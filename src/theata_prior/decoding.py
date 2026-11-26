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


def _boundary_free_components(num_detectors: int, supports: list[tuple[int, ...]]) -> tuple[np.ndarray, np.ndarray]:
    """Pack components without a boundary for a vectorized syndrome parity check."""
    parent = list(range(num_detectors))

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for support in supports:
        if len(support) == 2:
            i, j = support
            parent[root(j)] = root(i)
    boundaries = {root(support[0]) for support in supports if len(support) == 1}
    groups: dict[int, list[int]] = {}
    for detector in range(num_detectors):
        component = root(detector)
        if component not in boundaries:
            groups.setdefault(component, []).append(detector)
    lengths = np.array([len(group) for group in groups.values()], dtype=int)
    columns = np.array([detector for group in groups.values() for detector in group], dtype=int)
    return columns, np.cumsum(lengths) - lengths


def _initialize_decoders(directory: str, models: dict[str, str], baseline: str, allow_unmatched: bool) -> None:
    global _DECODE_STATE, _DECODE_THREAD_LIMIT
    _DECODE_THREAD_LIMIT = threadpool_limits(limits=1)
    samples = SampleSet.open(directory)
    decoders = {}
    components = {}
    for name, text in models.items():
        dem = stim.DetectorErrorModel(text)
        if dem.num_detectors != samples.num_detectors or dem.num_observables != samples.num_observables:
            raise ValueError(f"Decoder {name!r} has dimensions inconsistent with the sample set")
        # The estimator's structure validator rejects non-graphlike mechanisms.
        # Recheck here so direct callers cannot silently drop a hyperedge.
        supports = []
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
            if allow_unmatched and detectors and instruction.args_copy()[0] > 0:
                supports.append(tuple(detectors))
        decoders[name] = pymatching.Matching.from_detector_error_model(dem)
        if allow_unmatched:
            components[name] = _boundary_free_components(samples.num_detectors, supports)
    _DECODE_STATE = {
        "detectors": samples.load_detectors(),
        "observables": samples.load_observables(),
        "decoders": decoders,
        "components": components,
        "baseline": baseline,
    }


def _decode_available(decoder, detectors: np.ndarray, observables: np.ndarray,
                      components: tuple[np.ndarray, np.ndarray] | None) -> tuple[np.ndarray, np.ndarray]:
    mismatch = np.zeros(len(detectors), dtype=bool)
    unmatched = np.zeros_like(mismatch)
    if components is not None:
        columns, starts = components
        if len(columns):
            unmatched = np.any(np.bitwise_xor.reduceat(detectors[:, columns], starts, axis=1), axis=1)

    def decode(indices: np.ndarray, batch: np.ndarray) -> None:
        try:
            prediction = decoder.decode_batch(batch)
        except ValueError as error:
            if components is None or "No perfect matching could be found" not in str(error):
                raise
            if len(indices) == 1:
                unmatched[indices[0]] = True
            else:
                mid = len(indices) // 2
                decode(indices[:mid], batch[:mid])
                decode(indices[mid:], batch[mid:])
            return
        expected = observables if len(indices) == len(observables) else observables[indices]
        if prediction.shape != expected.shape:
            raise ValueError(f"Decoder returned unexpected observable shape {prediction.shape}")
        mismatch[indices] = np.any(prediction != expected, axis=1)

    available = np.flatnonzero(~unmatched)
    if len(available):
        batch = detectors if len(available) == len(detectors) else detectors[available]
        decode(available, batch)
    return mismatch, unmatched


def _decode_shard(task: tuple[int, int]) -> dict:
    start, stop = task
    detectors = np.ascontiguousarray(_DECODE_STATE["detectors"][start:stop], dtype=np.uint8)
    observables = _DECODE_STATE["observables"][start:stop]
    errors = {}
    unmatched_counts = {}
    for name, decoder in _DECODE_STATE["decoders"].items():
        mismatch, unmatched = _decode_available(decoder, detectors, observables, _DECODE_STATE["components"].get(name))
        errors[name] = mismatch | unmatched
        unmatched_counts[name] = int(np.count_nonzero(unmatched))
    baseline_errors = errors[_DECODE_STATE["baseline"]]
    return {
        "failures": {name: int(np.count_nonzero(values)) for name, values in errors.items()},
        "unmatched": unmatched_counts,
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
    allow_unmatched: bool = False,
) -> dict:
    """Evaluate all DEMs on identical test shots, with no training-data access.

    Each process constructs its own Matching objects and decodes C++ batches.
    Only integer counts cross process boundaries. ``models`` must contain
    independent graphlike DEMs. With allow_unmatched, infeasible syndromes
    count as failures and are also reported separately.
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
    total_unmatched = {name: 0 for name in models}
    total_paired = {name: [0, 0] for name in models if name != baseline}
    tasks = [(start, min(count, start + chunk_shots)) for start in range(0, count, chunk_shots)]
    initargs = (str(samples.directory), dict(models), baseline, allow_unmatched)

    def accumulate(result: dict) -> None:
        for name, failures in result["failures"].items():
            total_failures[name] += failures
            total_unmatched[name] += result["unmatched"][name]
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
    summaries = {name: logical_error_statistics(value, count, confidence) for name, value in total_failures.items()}
    if allow_unmatched:
        for name, summary in summaries.items():
            failures = total_unmatched[name]
            decoded = count - failures
            mismatches = total_failures[name] - failures
            conditional = logical_error_statistics(mismatches, decoded, confidence) if decoded else None
            summary.update(
                decoding_failures=failures, decoding_failure_rate=failures / count,
                successfully_decoded_shots=decoded, logical_mismatches=mismatches,
                conditional_logical_mismatch_rate=mismatches / decoded if decoded else None,
                conditional_ci_low=conditional["ci_low"] if conditional else None,
                conditional_ci_high=conditional["ci_high"] if conditional else None,
            )
    return {
        "shots": int(count),
        "baseline": baseline,
        "models": summaries,
        "paired": {name: paired_error_statistics(a, b, count, confidence) for name, (a, b) in total_paired.items()},
        "elapsed_seconds": time.perf_counter() - started,
        "workers": int(workers),
        "chunk_shots": int(chunk_shots),
        "pymatching_version": pymatching.__version__,
    }
