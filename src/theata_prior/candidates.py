"""Implicit complete family of local detector supports and sanitized metadata."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import stim

from .dem import ErrorSupport, GraphStructure

CANDIDATE_SCHEMA = "pairs_lexicographic_then_singletons_v1"


@dataclass(frozen=True)
class CandidateSet:
    num_detectors: int

    def __post_init__(self) -> None:
        if type(self.num_detectors) is not int or self.num_detectors < 1:
            raise ValueError("A positive integer detector count is required")

    @property
    def num_pairs(self) -> int:
        return self.num_detectors * (self.num_detectors - 1) // 2

    def __len__(self) -> int:
        return self.num_pairs + self.num_detectors

    @cached_property
    def pairs(self) -> tuple[np.ndarray, np.ndarray]:
        indices = np.triu_indices(self.num_detectors, k=1)
        for axis in indices:
            axis.setflags(write=False)
        return indices

    def index(self, support: tuple[int, ...]) -> int:
        m = self.num_detectors
        if len(support) == 1 and 0 <= support[0] < m:
            return self.num_pairs + support[0]
        if len(support) == 2:
            i, j = sorted(support)
            if 0 <= i < j < m:
                return i * (2 * m - i - 1) // 2 + j - i - 1
        raise ValueError(f"Invalid detector support {support}")

    def support(self, index: int) -> tuple[int, ...]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index >= self.num_pairs:
            return (int(index) - self.num_pairs,)
        left, right = self.pairs
        return int(left[index]), int(right[index])

    def matrix(self, values: np.ndarray) -> np.ndarray:
        """Place pairs symmetrically and singletons on the diagonal."""
        values = np.asarray(values)
        if values.shape != (len(self),):
            raise ValueError("Values must match the complete candidate family")
        matrix = np.empty((self.num_detectors, self.num_detectors), dtype=values.dtype)
        left, right = self.pairs
        matrix[left, right] = matrix[right, left] = values[:self.num_pairs]
        np.fill_diagonal(matrix, values[self.num_pairs:])
        return matrix

    def incident(self, pairs: np.ndarray) -> np.ndarray:
        """Sum undirected pair values incident on each detector."""
        i, j = self.pairs
        return (np.bincount(i, weights=pairs, minlength=self.num_detectors)
                + np.bincount(j, weights=pairs, minlength=self.num_detectors))


@dataclass(frozen=True)
class DetectorMetadata:
    detector_ids: tuple[int, ...]
    coordinates: tuple[tuple[float, ...], ...]
    num_observables: int

    def __post_init__(self) -> None:
        if not self.detector_ids or len(self.coordinates) != len(self.detector_ids):
            raise ValueError("One coordinate entry is required per detector")
        if len(set(self.detector_ids)) != len(self.detector_ids) or min(self.detector_ids) < 0:
            raise ValueError("Original detector IDs must be distinct and nonnegative")
        if type(self.num_observables) is not int or self.num_observables < 0:
            raise ValueError("Observable count must be a nonnegative integer")

    @property
    def candidates(self) -> CandidateSet:
        return CandidateSet(len(self.detector_ids))

    @classmethod
    def from_ideal(cls, path: str | Path, basis: str) -> DetectorMetadata:
        from .circuits import detector_sectors

        circuit = stim.Circuit.from_file(path)
        ids = tuple(i for i, sector in enumerate(detector_sectors(circuit)) if sector.lower() == basis)
        coordinates = circuit.get_detector_coordinates()
        return cls(ids, tuple(tuple(coordinates.get(i, ())) for i in ids), circuit.num_observables)

    def to_dict(self) -> dict:
        return {
            "candidate_schema": CANDIDATE_SCHEMA,
            "num_detectors": len(self.detector_ids),
            "detector_ids": list(self.detector_ids),
            "coordinates": [list(point) for point in self.coordinates],
            "num_observables": self.num_observables,
        }

    @classmethod
    def from_dict(cls, value: dict) -> DetectorMetadata:
        if value.get("candidate_schema") != CANDIDATE_SCHEMA:
            raise ValueError("Unrecognized detector candidate schema")
        metadata = cls(tuple(value["detector_ids"]), tuple(map(tuple, value["coordinates"])), value["num_observables"])
        if value.get("num_detectors") != len(metadata.detector_ids):
            raise ValueError("Detector metadata count is inconsistent")
        return metadata

    def graph(self, probabilities: np.ndarray, labels: np.ndarray | None = None) -> tuple[GraphStructure, np.ndarray]:
        """Emit positive supports, using zero observable dimensions for detector-only DEMs."""
        family = self.candidates
        p = np.asarray(probabilities, dtype=np.float64)
        if p.shape != (len(family),) or not np.all((p >= 0) & (p < 0.5)):
            raise ValueError("Candidate probabilities must be finite and in [0, 1/2)")
        if labels is not None:
            labels = np.asarray(labels)
            if self.num_observables != 1 or labels.shape != p.shape or not np.all((labels == 0) | (labels == 1)):
                raise ValueError("Expected one logical bit per candidate for a one-observable benchmark")
        active = np.flatnonzero(p > 0)
        edges = tuple(ErrorSupport(family.support(k), (0,) if labels is not None and labels[k] else ()) for k in active)
        return GraphStructure(family.num_detectors, int(labels is not None), edges, self.detector_ids,
                              self.coordinates), p[active]
