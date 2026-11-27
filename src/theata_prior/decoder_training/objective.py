"""Fixed-sample decoding used by both optimization and held-out evaluation."""
from __future__ import annotations

import time

import numpy as np
from threadpoolctl import threadpool_limits

from ..dem import GraphStructure
from ..sampling import SampleSet
from .adapters import Decoder, count_outcomes
from .weights import WeightModel


def decode_samples(predict, detectors: np.ndarray, observables: np.ndarray, chunk_shots: int,
                   indices: np.ndarray | None = None, *, keep_predictions: bool = False):
    """Decode a fixed subset in bounded chunks, preserving its shot order."""
    shots = len(detectors) if indices is None else len(indices)
    outcomes = np.empty(shots, dtype=np.uint8)
    predictions = np.empty((shots, observables.shape[1]), dtype=np.uint8) if keep_predictions else None
    for first in range(0, shots, chunk_shots):
        span = slice(first, min(first + chunk_shots, shots))
        selection = span if indices is None else indices[span]
        predicted, failed = predict(np.ascontiguousarray(detectors[selection], dtype=np.uint8))
        outcomes[span] = count_outcomes(predicted, failed, observables[selection])
        if keep_predictions:
            predictions[span] = predicted
    return outcomes, predictions


class Scorer:
    """Process-local decoder and read-only training/validation array references."""

    def __init__(self, structure: GraphStructure, train: SampleSet, train_shots: int,
                 validation: SampleSet, weights: WeightModel, decoder: str, bits_per_step: int,
                 chunk_shots: int):
        self.weights = weights
        self.decoder = Decoder(decoder, structure, bits_per_step)
        self.train = train.load_detectors(train_shots), train.load_observables(train_shots)
        self.validation = validation.load_detectors(), validation.load_observables()
        self.chunk_shots = chunk_shots

    def score(self, task: tuple[np.ndarray, np.ndarray | None, bool]) -> dict:
        theta, indices, validation = task
        started = time.perf_counter()
        realized = self.weights.realize(theta)
        predict = self.decoder.predictor(realized.priors)
        detectors, observables = self.validation if validation else self.train
        outcomes, _ = decode_samples(predict, detectors, observables, self.chunk_shots, indices)
        return {"failures": int(np.count_nonzero(outcomes)), "shots": len(outcomes),
                "decoder_failures": int(np.count_nonzero(outcomes == 2)),
                "seconds": time.perf_counter() - started}


_SCORER: Scorer | None = None
_THREAD_LIMIT = None


def initialize_worker(native_threads: int, *scorer_args) -> None:
    global _SCORER, _THREAD_LIMIT
    _SCORER = Scorer(*scorer_args)
    _THREAD_LIMIT = threadpool_limits(limits=native_threads)


def score_candidate(task: tuple[np.ndarray, np.ndarray | None, bool]) -> dict:
    return _SCORER.score(task)
