"""Detector moments and exact graphlike prior inversion (Chen et al. 2021).

Equation S11 estimates two-detector events.  Equations S14--S16 infer existing
boundary events after accounting for all incident two-detector events.  No
true prior or observed logical label enters this module's estimator.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import stim
from threadpoolctl import threadpool_limits

from .dem import GraphStructure


class EstimationError(ValueError):
    """The empirical moments are incompatible with the identifiable model branch."""


class SaturatedMarginalsError(EstimationError):
    """A detector's sample mean lies outside the positive-bias branch."""

    def __init__(self, means: np.ndarray, detector_ids: list[int]) -> None:
        self.detector_ids = detector_ids
        self.maximum_mean = float(means.max())
        super().__init__(
            f"Empirical detector means >= 1/2 at local IDs {detector_ids[:20]}; the positive-bias "
            "branch is not identifiable with these data. Increase calibration shots or "
            "change the experimental regime; this condition is never silently clipped."
        )


@dataclass(frozen=True)
class Moments:
    """Empirical raw moments: E[x_i] and E[x_i x_j] with normalization 1/N.

    ``shots=0`` is permitted for supplied exact population moments in algebraic
    validation.  Sample-derived moments always have strictly positive shots.
    """

    shots: int
    means: np.ndarray
    second_moments: np.ndarray

    def __post_init__(self) -> None:
        if isinstance(self.shots, bool) or int(self.shots) != self.shots:
            raise ValueError("shots must be an integer")
        object.__setattr__(self, "shots", int(self.shots))
        object.__setattr__(self, "means", np.asarray(self.means, dtype=np.float64))
        object.__setattr__(self, "second_moments", np.asarray(self.second_moments, dtype=np.float64))

    @property
    def covariance(self) -> np.ndarray:
        return self.second_moments - np.outer(self.means, self.means)

    def validate(self, num_detectors: int | None = None) -> None:
        mean = np.asarray(self.means)
        second = np.asarray(self.second_moments)
        m = len(mean) if mean.ndim == 1 else -1
        if m < 1 or (num_detectors is not None and m != num_detectors) or second.shape != (m, m):
            raise ValueError("Moment dimensions must match the graph's detector count")
        if self.shots < 0 or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(second)):
            raise ValueError("Finite moments and a nonnegative shot count are required")
        tol = 2e-12
        if np.any(mean < -tol) or np.any(mean > 1 + tol):
            raise ValueError("Detector means must be in [0,1]")
        if not np.allclose(second, second.T, atol=tol, rtol=0):
            raise ValueError("Second moment matrix must be symmetric")
        if not np.allclose(np.diag(second), mean, atol=tol, rtol=0):
            raise ValueError("For binary detectors, E[x_i^2] must equal E[x_i]")
        lower = np.maximum(0.0, mean[:, None] + mean[None, :] - 1)
        upper = np.minimum(mean[:, None], mean[None, :])
        if np.any(second < lower - tol) or np.any(second > upper + tol):
            raise ValueError("Second moments violate binary joint-probability bounds")


def compute_moments(
    detector_samples: np.ndarray,
    *,
    shots: int | None = None,
    workers: int = 1,
    chunk_shots: int = 16384,
    block_detectors: int = 128,
) -> Moments:
    """Compute full moments in bounded shot chunks and parallel detector blocks.

    A read-only NumPy memory map is accepted.  ``shots`` selects its first N
    rows, which permits nested training-size experiments without resampling.
    BLAS uses one thread per Python worker, preventing nested oversubscription.
    All products use float64 (never uint8 or boolean matrix multiplication).
    Memory is O(chunk_shots*M + M^2); no N-by-M float64 copy is retained.
    """
    samples = np.asanyarray(detector_samples)
    if samples.ndim != 2 or samples.shape[1] == 0:
        raise ValueError("detector_samples must have shape (N,M), with M > 0")
    n = samples.shape[0] if shots is None else int(shots)
    if n < 1 or n > samples.shape[0]:
        raise ValueError("shots must select a nonempty prefix of the sample array")
    if workers < 1 or chunk_shots < 1 or block_detectors < 1:
        raise ValueError("workers, chunk_shots and block_detectors must be positive")
    m = samples.shape[1]
    sums = np.zeros(m, dtype=np.float64)
    pair_sums = np.zeros((m, m), dtype=np.float64)
    blocks = [(i, min(m, i + block_detectors)) for i in range(0, m, block_detectors)]
    active_workers = min(int(workers), len(blocks))

    with threadpool_limits(limits=1, user_api="blas"):
        with ThreadPoolExecutor(max_workers=active_workers) as executor:
            for first in range(0, n, chunk_shots):
                chunk = np.asarray(samples[first:min(n, first + chunk_shots)], dtype=np.float64)
                if np.any((chunk != 0.0) & (chunk != 1.0)):
                    raise ValueError("Detector samples must contain only binary 0/1 values")
                sums += chunk.sum(axis=0)

                def product(block: tuple[int, int]) -> tuple[int, int, np.ndarray]:
                    left, right = block
                    return left, right, chunk[:, left:right].T @ chunk

                # Each task holds only a detector-row block; input chunk is shared.
                results = map(product, blocks) if active_workers == 1 else executor.map(product, blocks)
                for left, right, value in results:
                    pair_sums[left:right] += value
    return Moments(n, sums / n, pair_sums / n)


def pair_probability_matrix(moments: Moments) -> np.ndarray:
    """Equation S11, with zero diagonal and no probability regularization.

    Writing A=(1-2 mu_i)(1-2 mu_j), B=1-2 mu_i-2 mu_j+4 mu_ij,
    S11 is (1-sqrt(A/B))/2.  The equivalent expression below avoids loss
    of precision from subtracting two nearly equal numbers at low rates.
    Invalid pairs have NaN, and finite negative estimates remain negative.
    """
    moments.validate()
    mu = np.asarray(moments.means, dtype=np.float64)
    saturated = np.flatnonzero(mu >= 0.5).tolist()
    if saturated:
        raise SaturatedMarginalsError(mu, saturated)
    covariance = moments.covariance
    a = np.outer(1 - 2 * mu, 1 - 2 * mu)
    b = 1 - 2 * mu[:, None] - 2 * mu[None, :] + 4 * moments.second_moments
    result = np.full_like(b, np.nan, dtype=np.float64)
    valid = (a > 0) & (b > 0)
    result[valid] = 2 * covariance[valid] / (b[valid] + np.sqrt(a[valid] * b[valid]))
    np.fill_diagonal(result, 0.0)
    return result


@dataclass(frozen=True)
class EstimationResult:
    """Fitted decoder priors and unregularized statistics retained for auditing."""

    structure: GraphStructure
    probabilities: np.ndarray
    raw_probabilities: np.ndarray
    p_ij: np.ndarray
    boundary_probabilities: np.ndarray
    moments: Moments
    diagnostics: dict[str, Any]

    def to_dem(self) -> stim.DetectorErrorModel:
        return self.structure.to_dem(self.probabilities)

    def save(self, directory: str | Path) -> None:
        """Save the probability-free topology, raw moments, estimates and DEM."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.to_dem().to_file(out / "estimated.dem")
        (out / "structure.json").write_text(json.dumps(self.structure.to_dict(), indent=2) + "\n")
        (out / "diagnostics.json").write_text(json.dumps(self.diagnostics, indent=2, allow_nan=False) + "\n")
        np.save(out / "estimated_priors.npy", self.probabilities, allow_pickle=False)
        np.save(out / "raw_estimated_priors.npy", self.raw_probabilities, allow_pickle=False)
        np.save(out / "p_ij.npy", self.p_ij, allow_pickle=False)
        np.save(out / "boundary_probabilities.npy", self.boundary_probabilities, allow_pickle=False)
        np.savez_compressed(
            out / "moments.npz", shots=self.moments.shots,
            means=self.moments.means, second_moments=self.moments.second_moments,
        )


def estimate_priors(
    structure: GraphStructure,
    detector_samples: np.ndarray | None = None,
    *,
    moments: Moments | None = None,
    workers: int = 1,
    chunk_shots: int = 16384,
    block_detectors: int = 128,
    probability_floor: float = 1e-8,
    probability_ceiling: float = 0.499999,
    strict: bool = False,
) -> EstimationResult:
    """Estimate all known graph edges from detector samples alone.

    Pair edges use exact S11.  Existing one-detector boundary edges use
    S14--S16, equivalently p_iB=[1-(1-2 mu_i)/prod_j(1-2 p_ij)]/2.
    The product includes only known two-detector edges, not every nonzero
    sample correlation.  Nonedges in the full matrix remain diagnostics.

    Finite sampling can yield negative estimates or incompatible moments.
    Default behavior retains raw values, substitutes ``probability_floor``
    for invalid known-edge values, and clips valid estimates to the explicit
    [floor,ceiling] interval.  Boundary inversion uses these regularized pair
    priors, and residual moments are reported.  ``strict=True`` rejects
    nonphysical known-edge estimates instead.  Saturated marginal means
    always raise, regardless of strict.  Neither policy consults true priors.
    """
    if (detector_samples is None) == (moments is None):
        raise ValueError("Supply exactly one of detector_samples or moments")
    if not 0 < probability_floor < probability_ceiling < 0.5:
        raise ValueError("Require 0 < probability_floor < probability_ceiling < 1/2")
    if moments is None:
        moments = compute_moments(
            detector_samples, workers=workers, chunk_shots=chunk_shots,
            block_detectors=block_detectors,
        )
    moments.validate(structure.num_detectors)
    p_ij = pair_probability_matrix(moments)
    raw = np.full(structure.num_errors, np.nan, dtype=np.float64)
    fitted = np.empty(structure.num_errors, dtype=np.float64)
    boundary = np.full(structure.num_detectors, np.nan, dtype=np.float64)
    incident_log_bias = np.zeros(structure.num_detectors, dtype=np.float64)
    pair_indices: list[int] = []
    boundary_indices: list[int] = []
    known_pair_mask = np.zeros_like(p_ij, dtype=bool)

    def regularize(value: float, edge_index: int) -> float:
        if not math.isfinite(value) or value < -1e-12 or value >= 0.5:
            if strict:
                raise EstimationError(f"Nonphysical estimate {value} for edge {edge_index}: {structure.edges[edge_index]}")
        if not math.isfinite(value):
            return probability_floor
        return float(np.clip(value, probability_floor, probability_ceiling))

    for edge_index, edge in enumerate(structure.edges):
        if len(edge.detectors) == 2:
            i, j = edge.detectors
            pair_indices.append(edge_index)
            known_pair_mask[i, j] = known_pair_mask[j, i] = True
            raw[edge_index] = p_ij[i, j]
            fitted[edge_index] = regularize(float(raw[edge_index]), edge_index)
            log_bias = math.log1p(-2 * fitted[edge_index])
            incident_log_bias[i] += log_bias
            incident_log_bias[j] += log_bias
        else:
            boundary_indices.append(edge_index)

    for edge_index in boundary_indices:
        i = structure.edges[edge_index].detectors[0]
        log_ratio = math.log1p(-2 * float(moments.means[i])) - incident_log_bias[i]
        # -expm1(t)/2 evaluates (1-exp(t))/2 accurately when p_iB is small.
        value = -0.5 * math.expm1(log_ratio) if log_ratio < 709 else -math.inf
        boundary[i] = raw[edge_index] = value
        fitted[edge_index] = regularize(value, edge_index)

    fitted_log_bias = np.zeros(structure.num_detectors, dtype=np.float64)
    for edge, prior in zip(structure.edges, fitted, strict=True):
        for i in edge.detectors:
            fitted_log_bias[i] += math.log1p(-2 * float(prior))
    predicted_means = -0.5 * np.expm1(fitted_log_bias)
    residual = predicted_means - moments.means
    offdiag = np.triu(np.ones_like(p_ij, dtype=bool), 1)
    nonedges = offdiag & ~known_pair_mask
    raw_pairs = raw[np.array(pair_indices, dtype=int)]
    raw_boundary = raw[np.array(boundary_indices, dtype=int)]

    def counts(values: np.ndarray) -> dict[str, int]:
        finite = np.isfinite(values)
        return {
            "total": int(len(values)),
            "nonfinite": int(np.count_nonzero(~finite)),
            "negative": int(np.count_nonzero(finite & (values < 0))),
            "at_or_above_half": int(np.count_nonzero(finite & (values >= 0.5))),
            "below_probability_floor": int(np.count_nonzero(finite & (values < probability_floor))),
            "above_probability_ceiling": int(np.count_nonzero(finite & (values > probability_ceiling))),
            "regularized": int(np.count_nonzero(~finite | (values < probability_floor) | (values > probability_ceiling))),
        }

    finite_nonedge = p_ij[nonedges & np.isfinite(p_ij)]
    diagnostics: dict[str, Any] = {
        "method": "S11_exact_pairs_S14_S15_S16_boundaries",
        "calibration_shots": int(moments.shots),
        "probability_floor": probability_floor,
        "probability_ceiling": probability_ceiling,
        "strict": bool(strict),
        "pairs": counts(raw_pairs),
        "boundaries": counts(raw_boundary),
        "all_known_edges": counts(raw),
        "full_matrix_negative_offdiagonal": int(np.count_nonzero(offdiag & (p_ij < 0))),
        "full_matrix_invalid_offdiagonal": int(np.count_nonzero(offdiag & ~np.isfinite(p_ij))),
        "maximum_absolute_nonedge_estimate": float(np.max(np.abs(finite_nonedge))) if len(finite_nonedge) else None,
        "max_marginal_residual": float(np.max(np.abs(residual))),
        "rms_marginal_residual": float(np.sqrt(np.mean(residual**2))),
        "marginal_residuals": residual.tolist(),
        "predicted_marginals": predicted_means.tolist(),
        "boundary_uses_regularized_known_pair_priors": True,
        "p_ij_diagonal": "zero_by_plotting_convention",
        "raw_nonedge_estimates_are_not_added_to_topology": True,
        "true_priors_used": False,
        "logical_samples_used": False,
    }
    return EstimationResult(structure, fitted, raw, p_ij, boundary, moments, diagnostics)
