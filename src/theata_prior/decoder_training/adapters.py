"""MWPM and direct ldpc Union-find predictions on fixed H and B."""
from __future__ import annotations

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from scipy.sparse import csc_matrix

from ..decoding import _boundary_free_components, _infeasible_shots, _predict_available
from ..dem import GraphStructure
from .config import LDPC_VERSION
from .weights import llrs


class UnsupportedPriorError(ValueError):
    """The specified decoder cannot represent this prior vector."""


def incidence(structure: GraphStructure) -> tuple[csc_matrix, csc_matrix]:
    """Return detector and logical incidence, with columns in canonical edge order."""
    rows, columns, logical_rows, logical_columns = [], [], [], []
    for index, edge in enumerate(structure.edges):
        rows.extend(edge.detectors)
        columns.extend([index] * len(edge.detectors))
        logical_rows.extend(edge.observables)
        logical_columns.extend([index] * len(edge.observables))
    h = csc_matrix((np.ones(len(rows), dtype=np.uint8), (rows, columns)),
                   shape=(structure.num_detectors, structure.num_errors))
    b = csc_matrix((np.ones(len(logical_rows), dtype=np.uint8), (logical_rows, logical_columns)),
                   shape=(structure.num_observables, structure.num_errors))
    return h, b


def decoder_metadata(name: str, bits_per_step: int) -> dict:
    if name == "mwpm":
        return {"decoder": name, "package": "pymatching", "version": version("pymatching")}
    if name != "uf":
        raise ValueError(f"Unknown decoder {name!r}")
    try:
        installed = version("ldpc")
    except PackageNotFoundError as error:
        raise ValueError("UF requires pip install '.[decoder-training]'") from error
    if installed != LDPC_VERSION:
        raise ValueError(f"Unsupported ldpc {installed}. Install the verified ldpc {LDPC_VERSION} wheel.")
    return {"decoder": name, "package": "ldpc", "version": installed,
            "mode": "LLR-guided peeling", "bits_per_step": bits_per_step, "constructor_uf_method": ""}


@lru_cache(maxsize=1)
def verify_union_find() -> dict:
    """Exercise the installed wheel's peeling and LLR behavior before optimization."""
    from ldpc import UnionFindDecoder

    h = csc_matrix([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    u = UnionFindDecoder(h, uf_method="")
    s = np.array([1, 0], dtype=np.uint8)
    a, b = np.array([1., 9., 9.]), np.array([9., 1., 1.])
    first = u.decode(s, llrs=a, bits_per_step=1).copy()
    second = u.decode(s, llrs=b, bits_per_step=1).copy()
    all_neighbors = [u.decode(s, llrs=w, bits_per_step=0).tolist() for w in (a, b)]
    zero = u.decode(np.zeros_like(s), llrs=b, bits_per_step=1)
    if (first.tolist() != [1, 0, 0] or second.tolist() != [0, 1, 1]
            or np.any(zero) or zero.shape != (3,)
            or not np.array_equal(h @ first % 2, s) or not np.array_equal(h @ second % 2, s)
            or all_neighbors != [[1, 0, 0], [1, 0, 0]]):
        raise RuntimeError("The installed ldpc wheel failed the Union-find compatibility fixture")
    return {"bits_per_step_1": [first.tolist(), second.tolist()],
            "bits_per_step_0": all_neighbors, "zero_syndrome": zero.tolist()}


class Decoder:
    """Keep a fixed graph and one worker-local UF instance across candidates."""

    def __init__(self, name: str, structure: GraphStructure, bits_per_step: int = 1):
        self.metadata = decoder_metadata(name, bits_per_step)
        self.name, self.structure, self.bits_per_step = name, structure, bits_per_step
        self.components = _boundary_free_components(structure.num_detectors,
                                                     [edge.detectors for edge in structure.edges])
        self.h, self.b = incidence(structure)
        if name == "uf":
            from ldpc import UnionFindDecoder

            verify_union_find()
            # The pinned wheel types uf_method as str. A falsey string selects peeling.
            self.uf = UnionFindDecoder(self.h, uf_method="")

    def predictor(self, probabilities: np.ndarray):
        """Construct a candidate-specific batch predictor returning predictions and a failure mask."""
        if np.shape(probabilities) != (self.structure.num_errors,):
            raise ValueError("prior vector must match the canonical edge order")
        try:
            weights = llrs(probabilities)
        except ValueError as error:
            raise UnsupportedPriorError(str(error)) from error
        if self.name == "mwpm":
            from pymatching import Matching

            # PyMatching computes log((1-p)/p) internally. The ratio must be representable.
            with np.errstate(over="ignore"):
                if not np.isfinite((1 - probabilities) / probabilities).all():
                    raise UnsupportedPriorError("PyMatching cannot represent the resulting edge weight")
            matching = Matching.from_detector_error_model(self.structure.to_dem(probabilities))
            if matching.num_edges != self.structure.num_errors:
                raise UnsupportedPriorError("PyMatching omitted a canonical mechanism")

            def predict(detectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                return _predict_available(matching, detectors, self.structure.num_observables, self.components)

            return predict

        def predict(detectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            failed = _infeasible_shots(detectors, self.components)
            predictions = np.zeros((len(detectors), self.structure.num_observables), dtype=np.uint8)
            # The UF wrapper returns zero corrections for zero syndromes.
            for index in np.flatnonzero(~failed & np.any(detectors, axis=1)):
                syndrome = detectors[index]
                correction = self.uf.decode(syndrome, llrs=weights, bits_per_step=self.bits_per_step)
                if (correction.shape != (self.structure.num_errors,)
                        or not np.all((correction == 0) | (correction == 1))
                        or not np.array_equal((self.h @ correction) % 2, syndrome)):
                    failed[index] = True
                else:
                    predictions[index] = (self.b @ correction) % 2
            return predictions, failed

        return predict


def count_outcomes(prediction: np.ndarray, failed: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Encode success=0, logical mismatch=1, decoder failure=2 per shot."""
    if prediction.shape != observed.shape or failed.shape != (len(observed),):
        raise ValueError("decoder results must align with the observable data")
    return np.where(failed, 2, np.any(prediction != observed, axis=1)).astype(np.uint8)
