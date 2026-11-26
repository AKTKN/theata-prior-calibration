"""Bounded-memory, reproducible multicore sampling to NumPy memory maps.

The same source, seed, stream name, total shots, chunk size and Stim build
produce identical samples when the worker count changes. Stim does not promise
bitwise reproducibility across versions or SIMD architectures. Prefix studies
should generate the maximum training set once and slice that stored set.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import stim
from threadpoolctl import threadpool_limits


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def shard_seed(seed: int, stream: str, shard: int) -> int:
    """Derive a uint64 seed without Python's process-randomized ``hash``."""
    if seed < 0 or shard < 0:
        raise ValueError("seed and shard must be nonnegative")
    key = json.dumps([int(seed), str(stream), int(shard)], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")


@dataclass(frozen=True)
class SampleSet:
    """Completed detection-event and observable samples stored on disk."""

    directory: Path
    metadata: dict

    @classmethod
    def open(cls, directory: str | Path, *, validate_observables: bool = True) -> "SampleSet":
        """Open a completed set; fitting can validate detections only.

        Set ``validate_observables=False`` for an inference stage whose input
        boundary must exclude observable-label files altogether.
        """
        directory = Path(directory).resolve()
        metadata = json.loads((directory / "samples.json").read_text(encoding="utf-8"))
        if metadata.get("status") != "complete":
            raise ValueError(f"Sample set is incomplete: {directory}")
        result = cls(directory, metadata)
        arrays = [(result.detector_path, result.num_detectors)]
        if validate_observables:
            arrays.append((result.observable_path, result.num_observables))
        for path, columns in arrays:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != (result.shots, columns) or array.dtype != np.bool_:
                raise ValueError(f"Unexpected sample array shape or dtype: {path}")
        return result

    @property
    def detector_path(self) -> Path:
        return self.directory / "detectors.npy"

    @property
    def observable_path(self) -> Path:
        return self.directory / "observables.npy"

    @property
    def shots(self) -> int:
        return int(self.metadata["config"]["shots"])

    @property
    def num_detectors(self) -> int:
        return int(self.metadata["num_detectors"])

    @property
    def num_observables(self) -> int:
        return int(self.metadata["num_observables"])

    def _prefix(self, path: Path, shots: int | None) -> np.ndarray:
        count = self.shots if shots is None else shots
        if count < 1 or count > self.shots:
            raise ValueError(f"shots must be in [1, {self.shots}]")
        return np.load(path, mmap_mode="r", allow_pickle=False)[:count]

    def load_detectors(self, shots: int | None = None) -> np.ndarray:
        return self._prefix(self.detector_path, shots)

    def load_observables(self, shots: int | None = None) -> np.ndarray:
        return self._prefix(self.observable_path, shots)


_SAMPLE_STATE: dict = {}
_SAMPLE_THREAD_LIMIT = None


def _initialize_sampler(model_text: str, source: str, indices: list[int], directory: str, seed: int, stream: str) -> None:
    global _SAMPLE_STATE, _SAMPLE_THREAD_LIMIT
    _SAMPLE_THREAD_LIMIT = threadpool_limits(limits=1)
    model = stim.DetectorErrorModel(model_text) if source == "dem" else stim.Circuit(model_text)
    _SAMPLE_STATE = {
        "model": model,
        "source": source,
        "indices": np.asarray(indices, dtype=np.int64),
        "detectors": np.load(Path(directory) / "detectors.npy", mmap_mode="r+", allow_pickle=False),
        "observables": np.load(Path(directory) / "observables.npy", mmap_mode="r+", allow_pickle=False),
        "seed": seed,
        "stream": stream,
    }


def _sample_shard(task: tuple[int, int, int]) -> int:
    shard, start, stop = task
    state = _SAMPLE_STATE
    seed = shard_seed(state["seed"], state["stream"], shard)
    if state["source"] == "dem":
        sampler = state["model"].compile_sampler(seed=seed)
        detectors, observables, _ = sampler.sample(shots=stop - start, return_errors=False)
    else:
        sampler = state["model"].compile_detector_sampler(seed=seed)
        detectors, observables = sampler.sample(shots=stop - start, separate_observables=True)
    state["detectors"][start:stop] = detectors[:, state["indices"]]
    state["observables"][start:stop] = observables
    state["detectors"].flush()
    state["observables"].flush()
    return shard


def sample_to_disk(
    model_text: str,
    output_dir: str | Path,
    *,
    shots: int,
    seed: int,
    stream: str = "train",
    source: str = "dem",
    detector_indices: Sequence[int] | None = None,
    workers: int = 1,
    chunk_shots: int = 65_536,
    resume: bool = True,
) -> SampleSet:
    """Sample an actual model, preserving observable labels for evaluation.

    Only selected detector columns are saved; indices refer to the full source
    model, in the order used by the projected decoder DEM. Calibration must not
    use the saved observable labels. Use distinct stream names for calibration
    and heldout evaluation. Concurrent calls must use different directories.
    """
    if source not in {"dem", "circuit"}:
        raise ValueError("source must be 'dem' or 'circuit'")
    if shots < 1 or workers < 1 or chunk_shots < 1 or seed < 0:
        raise ValueError("shots, workers and chunk_shots must be positive; seed must be nonnegative")
    model = stim.DetectorErrorModel(model_text) if source == "dem" else stim.Circuit(model_text)
    indices = list(range(model.num_detectors)) if detector_indices is None else [int(i) for i in detector_indices]
    if not indices or len(indices) != len(set(indices)) or min(indices) < 0 or max(indices) >= model.num_detectors:
        raise ValueError("detector_indices must be unique valid source detector indices and nonempty")
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "samples.json"
    config = {
        "model_sha256": hashlib.sha256(model_text.encode()).hexdigest(),
        "source": source,
        "shots": int(shots),
        "seed": int(seed),
        "stream": stream,
        "chunk_shots": int(chunk_shots),
        "detector_indices": indices,
        "stim_version": stim.__version__,
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config"] != config:
            raise ValueError(f"Sampling configuration differs from existing cache: {directory}")
        if not resume:
            raise FileExistsError(f"Sample cache already exists: {directory}")
        if manifest.get("status") == "complete":
            return SampleSet.open(directory)
        # Check the headers before a resumed worker opens writable memory maps.
        for filename, columns in (("detectors.npy", len(indices)), ("observables.npy", model.num_observables)):
            array = np.load(directory / filename, mmap_mode="r", allow_pickle=False)
            if array.shape != (shots, columns) or array.dtype != np.bool_:
                raise ValueError(f"Damaged incomplete sample cache: {directory / filename}")
    else:
        if (directory / "detectors.npy").exists() or (directory / "observables.npy").exists():
            raise FileExistsError(f"Untracked sample files already exist: {directory}")
        for filename, columns in (("detectors.npy", len(indices)), ("observables.npy", model.num_observables)):
            array = np.lib.format.open_memmap(directory / filename, mode="w+", dtype=np.bool_, shape=(shots, columns))
            array.flush()
            del array
        manifest = {
            "schema_version": 1,
            "status": "incomplete",
            "config": config,
            "num_source_detectors": model.num_detectors,
            "num_detectors": len(indices),
            "num_observables": model.num_observables,
            "completed_shards": [],
            "elapsed_seconds": 0.0,
        }
        _atomic_json(manifest_path, manifest)
    started = time.perf_counter()
    completed = set(manifest["completed_shards"])
    tasks = [
        (shard, start, min(shots, start + chunk_shots))
        for shard, start in enumerate(range(0, shots, chunk_shots))
        if shard not in completed
    ]
    initargs = (model_text, source, indices, str(directory), int(seed), stream)

    def record(shard: int) -> None:
        completed.add(shard)
        manifest["completed_shards"] = sorted(completed)
        _atomic_json(manifest_path, manifest)

    if workers == 1:
        _initialize_sampler(*initargs)
        try:
            for task in tasks:
                record(_sample_shard(task))
        finally:
            _SAMPLE_STATE.clear()
    elif tasks:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)), mp_context=mp.get_context("spawn"),
            initializer=_initialize_sampler, initargs=initargs,
        ) as executor:
            futures = [executor.submit(_sample_shard, task) for task in tasks]
            for future in as_completed(futures):
                record(future.result())
    manifest["elapsed_seconds"] += time.perf_counter() - started
    manifest["status"] = "complete"
    manifest["workers_last_run"] = int(workers)
    _atomic_json(manifest_path, manifest)
    return SampleSet.open(directory)
