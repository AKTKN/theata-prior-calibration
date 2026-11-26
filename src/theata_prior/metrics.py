"""Binomial and paired logical-error statistics.

Logical error rates are *per shot*, i.e. the probability that at least one
observable is predicted incorrectly. No conversion to a per-round rate is made.
"""
from __future__ import annotations

from math import sqrt
from statistics import NormalDist

from scipy.stats import beta


def wilson_interval(failures: int, shots: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return the two-sided Wilson score interval, including zero failures."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if shots <= 0 or failures < 0 or failures > shots:
        raise ValueError("require shots > 0 and 0 <= failures <= shots")
    z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    p = failures / shots
    denominator = 1.0 + z * z / shots
    center = (p + z * z / (2.0 * shots)) / denominator
    radius = z * sqrt(p * (1.0 - p) / shots + z * z / (4.0 * shots * shots)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def logical_error_statistics(failures: int, shots: int, confidence: float = 0.95) -> dict:
    low, high = wilson_interval(failures, shots, confidence)
    return {
        "shots": int(shots),
        "failures": int(failures),
        "logical_error_rate": failures / shots,
        "ci_low": low,
        "ci_high": high,
        "confidence": confidence,
        "ci_method": "wilson",
        "rate_unit": "per_shot",
    }


def _clopper_pearson(successes: int, total: int, confidence: float) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0
    low = 0.0 if successes == 0 else float(beta.ppf(tail, successes, total - successes + 1))
    high = 1.0 if successes == total else float(beta.ppf(1.0 - tail, successes + 1, total - successes))
    return low, high


def paired_error_statistics(
    candidate_only_failures: int,
    baseline_only_failures: int,
    shots: int,
    confidence: float = 0.95,
) -> dict:
    """Compare two decoders evaluated on identical shots.

    The rate difference is candidate minus baseline. The interval subtracts
    simultaneous Clopper--Pearson intervals for the two discordant categories.
    Bonferroni allocation gives coverage at least ``confidence`` under IID shots;
    it is conservative but remains informative when no disagreements occur.
    """
    a, b = candidate_only_failures, baseline_only_failures
    if shots <= 0 or min(a, b) < 0 or a + b > shots:
        raise ValueError("require shots > 0 and nonnegative discordant counts summing to <= shots")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    # Each marginal interval has failure probability alpha/2.
    marginal_confidence = (1.0 + confidence) / 2.0
    low_a, high_a = _clopper_pearson(a, shots, marginal_confidence)
    low_b, high_b = _clopper_pearson(b, shots, marginal_confidence)
    difference = (a - b) / shots
    variance = ((a + b) / shots - difference * difference) / (shots - 1) if shots > 1 else None
    return {
        "shots": int(shots),
        "candidate_only_failures": int(a),
        "baseline_only_failures": int(b),
        "difference": difference,
        "standard_error": sqrt(max(0.0, variance)) if variance is not None else None,
        "ci_low": max(-1.0, low_a - high_b),
        "ci_high": min(1.0, high_a - low_b),
        "confidence": confidence,
        "ci_method": "bonferroni_clopper_pearson_discordance",
    }
