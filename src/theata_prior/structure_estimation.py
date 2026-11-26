"""Complete-family detector-only parity inversion and finite-shot selection."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time
from typing import Iterator

import numpy as np
from scipy.stats import hypergeom, norm
from threadpoolctl import threadpool_limits

from .candidates import CandidateSet
from .estimation import EstimationError, Moments, compute_moments, pair_probability_matrix

MAX_COUNT_SHOTS = 2**50


class ParityDomainError(EstimationError):
    """At least one required empirical parity is nonpositive."""


def attenuation(bias: np.ndarray) -> np.ndarray:
    """Return -log(bias), keeping undefined logarithms as NaN."""
    values = np.asarray(bias, dtype=np.float64)
    result = np.full_like(values, np.nan)
    valid = values > 0
    with np.errstate(divide="ignore"):
        # log1p is accurate near one. Plain log also supports biases below machine epsilon.
        near_one = valid & (values > 0.5)
        result[near_one] = -np.log1p(values[near_one] - 1)
        result[valid & ~near_one] = -np.log(values[valid & ~near_one])
    return result


def attenuation_to_prior(value: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return -0.5 * np.expm1(-value)


def detector_biases(moments: Moments, family: CandidateSet) -> tuple[np.ndarray, np.ndarray]:
    """Use integer sums for the empirical parity domain, including exact zeros."""
    i, j = family.pairs
    if moments.shots > 0:
        n = moments.shots
        count, joint = integer_counts(moments)
        return (n - 2 * count) / n, (n - 2 * count[i] - 2 * count[j] + 4 * joint[i, j]) / n
    means = moments.means
    return 1 - 2 * means, 1 - 2 * means[i] - 2 * means[j] + 4 * moments.second_moments[i, j]


@dataclass(frozen=True)
class RawInversion:
    marginal_bias: np.ndarray
    pair_bias: np.ndarray
    pair_probability_matrix: np.ndarray
    pair_attenuation: np.ndarray
    singleton_attenuation: np.ndarray

    @property
    def valid(self) -> bool:
        return bool(np.all(self.marginal_bias > 0) and np.all(self.pair_bias > 0))

    def probabilities(self, family: CandidateSet) -> np.ndarray:
        return np.concatenate((self.pair_probability_matrix[family.pairs],
                               attenuation_to_prior(self.singleton_attenuation)))

    def diagnostics(self) -> dict:
        return {
            "nonpositive_marginal_ids": np.flatnonzero(self.marginal_bias <= 0).tolist(),
            "nonpositive_pair_parities": int(np.count_nonzero(self.pair_bias <= 0)),
            "negative_pair_attenuations": int(np.count_nonzero(self.pair_attenuation < 0)),
            "negative_singleton_attenuations": int(np.count_nonzero(self.singleton_attenuation < 0)),
        }


def invert_detector_moments(moments: Moments) -> RawInversion:
    """Invert all pair and singleton parities, retaining signed feasibility diagnostics."""
    moments.validate()
    family = CandidateSet(len(moments.means))
    i, j = family.pairs
    b, pair_bias = detector_biases(moments, family)
    marginal = attenuation(b)
    pair = (marginal[i] + marginal[j] - attenuation(pair_bias)) / 2
    if np.all(b > 0):
        raw = pair_probability_matrix(moments)
    else:
        # Preserve S11 wherever both single-detector parities are defined.
        valid = np.flatnonzero(b > 0)
        raw = np.full_like(moments.second_moments, np.nan)
        if len(valid):
            ix = np.ix_(valid, valid)
            raw[ix] = pair_probability_matrix(Moments(moments.shots, moments.means[valid],
                                                       moments.second_moments[ix]))
        np.fill_diagonal(raw, 0)
    invalid_pairs = pair_bias <= 0
    raw[i[invalid_pairs], j[invalid_pairs]] = raw[j[invalid_pairs], i[invalid_pairs]] = np.nan
    return RawInversion(b, pair_bias, raw, pair, marginal - family.incident(pair))


def integer_counts(moments: Moments) -> tuple[np.ndarray, np.ndarray]:
    """Recover exact integer counts from float64 averages for supported shot counts."""
    if not 1 <= moments.shots <= MAX_COUNT_SHOTS:
        raise ValueError(f"Integer count recovery requires 1 <= N <= {MAX_COUNT_SHOTS}")
    counts = []
    tolerance = min(0.25, 2 * np.finfo(float).eps * moments.shots)
    for mean in (moments.means, moments.second_moments):
        scaled = mean * moments.shots
        rounded = np.rint(scaled)
        if not np.all(np.isfinite(rounded)) or np.any(np.abs(scaled - rounded) > tolerance):
            raise ValueError("Empirical moments do not correspond to integer counts")
        counts.append(rounded.astype(np.int64))
    return counts[0], counts[1]


def prefix_moments(samples: np.ndarray, sizes: tuple[int, ...], **resources) -> Iterator[Moments]:
    """Reuse exact cumulative counts across ordered nested calibration prefixes."""
    previous = 0
    count = np.zeros(samples.shape[1], dtype=np.int64)
    pairs = np.zeros((samples.shape[1], samples.shape[1]), dtype=np.int64)
    for n in sizes:
        if not previous < n <= min(len(samples), MAX_COUNT_SHOTS):
            raise ValueError("Prefix sizes must be increasing and within the sample array")
        added = compute_moments(samples[previous:n], **resources)
        extra_count, extra_pairs = integer_counts(added)
        count += extra_count
        pairs += extra_pairs
        yield Moments(n, count / n, pairs / n)
        previous = n


def fisher_pair_pvalues(moments: Moments, *, batch_pairs: int = 4096) -> np.ndarray:
    """One-sided Fisher independence p-values for the full pair family."""
    if batch_pairs < 1:
        raise ValueError("The Fisher batch size must be positive")
    count, joint = integer_counts(moments)
    i, j = CandidateSet(len(count)).pairs
    x, y, both = count[i], count[j], joint[i, j]
    n = moments.shots
    if np.any(count < 0) or np.any(count > n) or np.any(both < np.maximum(0, x + y - n)) or np.any(both > np.minimum(x, y)):
        raise ValueError("Invalid detector contingency counts")
    pvalues = np.ones(len(i))
    possible = np.flatnonzero((both > 0) & (x < n) & (y < n))
    for first in range(0, len(possible), batch_pairs):
        batch = possible[first:first + batch_pairs]
        pvalues[batch] = hypergeom.sf(both[batch] - 1, n, x[batch], y[batch])
    return pvalues


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Step-down Holm adjusted p-values in the original candidate ordering."""
    pvalues = np.asarray(pvalues, dtype=np.float64)
    if pvalues.ndim != 1 or not np.all((0 <= pvalues) & (pvalues <= 1)):
        raise ValueError("Holm requires a one-dimensional family of valid p-values")
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(pvalues[order] * (len(order) - np.arange(len(order)))))
    return adjusted


def singleton_influence_chunks(samples: np.ndarray, raw: RawInversion, *, chunk_shots: int = 8192,
                               block_detectors: int = 128, workers: int = 1) -> Iterator[np.ndarray]:
    """Yield bounded blocks of the complete-family singleton influence vector per shot."""
    m = len(raw.marginal_bias)
    if samples.ndim != 2 or samples.shape[1] != m or min(chunk_shots, block_detectors, workers) < 1:
        raise ValueError("Invalid influence dimensions or execution resources")
    if not raw.valid:
        raise ParityDomainError("Singleton influence requires positive detector and pair parities")
    family = CandidateSet(m)
    i, j = family.pairs
    weights = np.zeros((m, m))
    weights[i, j] = weights[j, i] = 1 / raw.pair_bias
    blocks = range(0, m, block_detectors)
    with threadpool_limits(limits=1, user_api="blas"), ThreadPoolExecutor(max_workers=min(workers, len(blocks))) as executor:
        for first in range(0, len(samples), chunk_shots):
            chunk = samples[first:first + chunk_shots]
            if not np.all((chunk == 0) | (chunk == 1)):
                raise ValueError("Detector samples must be binary")
            z = 1 - 2 * chunk.astype(np.float64)
            h = (z - raw.marginal_bias) / raw.marginal_bias
            common = (h.sum(axis=1, keepdims=True) + m - 1) / 2
            influence = np.empty_like(z)

            def fill_block(left: int) -> None:
                block = slice(left, left + block_detectors)
                influence[:, block] = ((m - 4) / 2 * h[:, block] + common
                                       - 0.5 * z[:, block] * (z @ weights[block].T))

            for _ in executor.map(fill_block, blocks):
                pass
            yield influence


def singleton_standard_errors(samples: np.ndarray, raw: RawInversion, **resources) -> np.ndarray:
    """Sample-variance standard errors with stable, bounded batch accumulation."""
    if len(samples) < 2:
        raise ValueError("Singleton uncertainty needs at least two shots")
    n = 0
    mean = np.zeros(samples.shape[1])
    centered_sum = np.zeros_like(mean)
    for chunk in singleton_influence_chunks(samples, raw, **resources):
        size = len(chunk)
        relative = chunk - chunk[0]
        relative_mean = relative.mean(axis=0)
        local_mean = chunk[0] + relative_mean
        delta = local_mean - mean
        centered_sum += ((relative - relative_mean) ** 2).sum(axis=0) + delta**2 * n * size / (n + size)
        mean += delta * size / (n + size)
        n += size
    return np.sqrt(centered_sum / ((n - 1) * n))


@dataclass(frozen=True)
class StructureFit:
    """Complete-order arrays and diagnostics, including unresolved fits."""

    arrays: dict[str, np.ndarray]
    diagnostics: dict

    @property
    def valid(self) -> bool:
        return self.diagnostics["status"] == "complete"


def fit_structure(samples: np.ndarray, *, moments: Moments | None = None, alpha_pairs: float = 0.025,
                  alpha_singletons: float = 0.025, probability_ceiling: float = 0.499999,
                  workers: int = 1, chunk_shots: int = 8192, block_detectors: int = 128) -> StructureFit:
    """Estimate detector support and rates using only calibration detectors and predefined policy."""
    if not (0 < alpha_pairs < 1 and 0 < alpha_singletons < 1 and 0 < probability_ceiling < 0.5):
        raise ValueError("Alphas must lie in (0, 1) and the ceiling in (0, 1/2)")
    if samples.ndim != 2 or len(samples) < 2:
        raise ValueError("Structure selection needs a detector array with at least two shots")
    resources = dict(workers=workers, chunk_shots=chunk_shots, block_detectors=block_detectors)
    started = time.perf_counter()
    if moments is None:
        moments = compute_moments(samples, **resources)
    if moments.shots != len(samples):
        raise ValueError("Statistics and calibration samples must cover the same prefix")
    moments.validate(samples.shape[1])
    raw = invert_detector_moments(moments)
    family = CandidateSet(samples.shape[1])
    pair_count = family.num_pairs
    marginals, coincidences = integer_counts(moments)
    arrays = dict(means=moments.means, second_moments=moments.second_moments,
                  marginal_counts=marginals, coincidence_counts=coincidences,
                  marginal_bias=raw.marginal_bias, pair_bias=raw.pair_bias,
                  raw_p_ij=raw.pair_probability_matrix, raw_probabilities=raw.probabilities(family),
                  raw_pair_attenuation=raw.pair_attenuation, raw_singleton_attenuation=raw.singleton_attenuation)
    diagnostics = dict(raw.diagnostics(), status="complete" if raw.valid else "invalid_detector_parity",
                       pair_family_size=pair_count, singleton_family_size=family.num_detectors,
                       num_unresolved=0 if raw.valid else len(family),
                       detector_inversion_seconds=time.perf_counter() - started)
    started = time.perf_counter()
    pair_pvalues = fisher_pair_pvalues(moments)
    pair_adjusted = holm_adjust(pair_pvalues)
    arrays.update(pair_pvalues=pair_pvalues, pair_adjusted_pvalues=pair_adjusted,
                  pair_selected=pair_adjusted <= alpha_pairs)
    if not raw.valid:
        diagnostics["support_statistics_seconds"] = time.perf_counter() - started
        return StructureFit(arrays, diagnostics)
    singleton_se = singleton_standard_errors(samples, raw, **resources)
    no_information = (singleton_se == 0) | (marginals == 0)
    statistic = np.divide(raw.singleton_attenuation, singleton_se, out=np.zeros_like(singleton_se),
                          where=~no_information)
    singleton_pvalues = np.where(no_information, 1.0, norm.sf(statistic))
    singleton_adjusted = holm_adjust(singleton_pvalues)
    selected = np.concatenate((arrays["pair_selected"], singleton_adjusted <= alpha_singletons))
    arrays.update(singleton_se=singleton_se, singleton_pvalues=singleton_pvalues,
                  singleton_adjusted_pvalues=singleton_adjusted, singleton_zero_information=no_information,
                  complete_selected_selected=selected)
    for method, decision in (("complete_clipped", np.ones(len(family), dtype=bool)), ("complete_selected", selected)):
        pairs = np.where(decision[:pair_count], np.clip(arrays["raw_probabilities"][:pair_count], 0, probability_ceiling), 0)
        residual = attenuation(raw.marginal_bias) - family.incident(-np.log1p(-2 * pairs))
        singleton_priors = attenuation_to_prior(residual)
        refit = np.concatenate((arrays["raw_probabilities"][:pair_count], singleton_priors))
        probabilities = np.concatenate((pairs, np.where(decision[pair_count:], np.clip(singleton_priors, 0,
                                                                                       probability_ceiling), 0)))
        active = probabilities > 0
        arrays.update({f"{method}_priors": probabilities, f"{method}_active": active,
                       f"{method}_refit_probabilities": refit, f"{method}_singleton_refit_attenuation": residual})
        diagnostics[method] = {
            "nonpositive_selected_refits": int(np.count_nonzero(decision & ~active)),
            "capped_selected_refits": int(np.count_nonzero(decision & (refit > probability_ceiling))),
            "negative_refit_singletons": int(np.count_nonzero(residual < 0)),
        }
    # Clipping has no hypothesis screen. Its selection mask is its positive support.
    arrays["complete_clipped_selected"] = arrays["complete_clipped_active"].copy()
    diagnostics["support_statistics_seconds"] = time.perf_counter() - started
    return StructureFit(arrays, diagnostics)
