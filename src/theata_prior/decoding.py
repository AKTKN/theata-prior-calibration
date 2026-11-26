"""Paired, multicore PyMatching evaluation on a fixed heldout sample set."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import time
from typing import Mapping

import numpy as np
import pymatching
import stim
from threadpoolctl import threadpool_limits

from .metrics import logical_error_statistics, paired_error_statistics
from .sampling import SampleSet


_DECODE_STATE: dict = {}
_DECODE_THREAD_LIMIT = None


def _initialize_decoders(directory: str, models: dict[str, str], baseline: str) -> None:
    global _DECODE_STATE, _DECODE_THREAD_LIMIT
    _DECODE_THREAD_LIMIT = threadpool_limits(limits=1)
    samples = SampleSet.open(directory)
    decoders = {}
    for name, text in models.items():
        dem = stim.DetectorErrorModel(text)
        if dem.num_detectors != samples.num_detectors or dem.num_observables != samples.num_observables:
            raise ValueError(f"Decoder {name!r} has dimensions inconsistent with the sample set")
        # The estimator's structure validator rejects non-graphlike mechanisms.
        # Recheck here so direct callers cannot silently drop a hyperedge.
        for instruction in dem.flattened():
            if instruction.type != "error":
                continue
            detectors = set()
            observables = set()
            for target in instruction.targets_copy():
                if target.is_separator():
                    raise ValueError("Evaluation requires an explicit independent graphlike DEM, without '^' decompositions")
                if target.is_relative_detector_id():
                    detectors.symmetric_difference_update({target.val})
                elif target.is_logical_observable_id():
                    observables.symmetric_difference_update({target.val})
            if len(detectors) > 2:
                raise ValueError("Evaluation requires a graphlike projected DEM; found a hyperedge")
            if not detectors and observables:
                raise ValueError("A logical-only mechanism is invisible to matching and detector-only calibration")
        decoders[name] = pymatching.Matching.from_detector_error_model(dem)
    _DECODE_STATE = {
        "detectors": samples.load_detectors(),
        "observables": samples.load_observables(),
        "decoders": decoders,
        "baseline": baseline,
    }


def _decode_shard(task: tuple[int, int]) -> dict:
    start, stop = task
    detectors = np.ascontiguousarray(_DECODE_STATE["detectors"][start:stop], dtype=np.uint8)
    observables = _DECODE_STATE["observables"][start:stop]
    errors = {}
    for name, decoder in _DECODE_STATE["decoders"].items():
        prediction = decoder.decode_batch(detectors)
        if prediction.shape != observables.shape:
            raise ValueError(f"Decoder {name!r} returned unexpected observable shape {prediction.shape}")
        errors[name] = np.any(prediction != observables, axis=1)
    baseline_errors = errors[_DECODE_STATE["baseline"]]
    return {
        "failures": {name: int(np.count_nonzero(values)) for name, values in errors.items()},
        "paired": {
            name: (int(np.count_nonzero(values & ~baseline_errors)), int(np.count_nonzero(~values & baseline_errors)))
            for name, values in errors.items() if name != _DECODE_STATE["baseline"]
        },
    }


def evaluate_models(
    samples: SampleSet | str | Path,
    models: Mapping[str, str],
    *,
    baseline: str = "actual",
    workers: int = 1,
    chunk_shots: int = 16_384,
    shots: int | None = None,
    confidence: float = 0.95,
) -> dict:
    """Evaluate all DEMs on identical test shots, with no training-data access.

    Each process constructs its own Matching objects and decodes C++ batches.
    Only integer counts cross process boundaries. ``models`` must contain
    independent graphlike DEMs; raw correlated source DEMs belong in sampling.
    """
    samples = SampleSet.open(samples) if not isinstance(samples, SampleSet) else samples
    count = samples.shots if shots is None else shots
    if count < 1 or count > samples.shots or workers < 1 or chunk_shots < 1:
        raise ValueError("Invalid shot count, worker count, or chunk size")
    if not models or baseline not in models:
        raise ValueError("models must be nonempty and contain the baseline name")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if samples.num_observables < 1:
        raise ValueError("Logical-error evaluation requires at least one observable")
    started = time.perf_counter()
    total_failures = {name: 0 for name in models}
    total_paired = {name: [0, 0] for name in models if name != baseline}
    tasks = [(start, min(count, start + chunk_shots)) for start in range(0, count, chunk_shots)]
    initargs = (str(samples.directory), dict(models), baseline)

    def accumulate(result: dict) -> None:
        for name, failures in result["failures"].items():
            total_failures[name] += failures
        for name, (candidate, reference) in result["paired"].items():
            total_paired[name][0] += candidate
            total_paired[name][1] += reference

    if workers == 1:
        _initialize_decoders(*initargs)
        try:
            for task in tasks:
                accumulate(_decode_shard(task))
        finally:
            _DECODE_STATE.clear()
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)), mp_context=mp.get_context("spawn"),
            initializer=_initialize_decoders, initargs=initargs,
        ) as executor:
            futures = [executor.submit(_decode_shard, task) for task in tasks]
            for future in as_completed(futures):
                accumulate(future.result())
    return {
        "shots": int(count),
        "baseline": baseline,
        "models": {name: logical_error_statistics(value, count, confidence) for name, value in total_failures.items()},
        "paired": {name: paired_error_statistics(a, b, count, confidence) for name, (a, b) in total_paired.items()},
        "elapsed_seconds": time.perf_counter() - started,
        "workers": int(workers),
        "chunk_shots": int(chunk_shots),
        "pymatching_version": pymatching.__version__,
    }
