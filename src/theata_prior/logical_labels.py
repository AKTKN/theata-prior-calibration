"""Optional supervised logical labels inferred from calibration D and L0."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .candidates import CandidateSet
from .estimation import Moments, compute_moments
from .structure_estimation import (
    attenuation, attenuation_to_prior, detector_biases, integer_counts, invert_detector_moments,
)

LABEL_POLICY = "complete_family_variant_attenuation_contrast_sign_tie_zero_v1"


@dataclass(frozen=True)
class LogicalParities:
    logical: float
    detector_logical: np.ndarray
    pair_logical: np.ndarray

    @classmethod
    def from_counts(cls, moments: Moments, n_logical: int, n_detector_logical: np.ndarray,
                    n_pair_logical: np.ndarray) -> LogicalParities:
        family = CandidateSet(len(moments.means))
        i, j = family.pairs
        detector, joint = integer_counts(moments)
        logical_bias_sum = moments.shots - 2 * n_logical
        signed_count = detector - 2 * n_detector_logical
        signed_coincidence = joint[i, j] - 2 * n_pair_logical[i, j]
        return cls(logical_bias_sum / moments.shots,
                   (logical_bias_sum - 2 * signed_count) / moments.shots,
                   (logical_bias_sum - 2 * signed_count[i] - 2 * signed_count[j] + 4 * signed_coincidence)
                   / moments.shots)


def prefix_logical_counts(detectors: np.ndarray, observables: np.ndarray, sizes: tuple[int, ...],
                          **resources) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Count L0=1 coincidences in bounded blocks for all nested prefixes."""
    if detectors.ndim != 2 or observables.shape != (len(detectors), 1):
        raise ValueError("Calibration observable samples must have shape (N, 1)")
    m = detectors.shape[1]
    logical_count = 0
    detector_count = np.zeros(m, dtype=np.int64)
    pair_count = np.zeros((m, m), dtype=np.int64)
    chunk_shots = resources.get("chunk_shots", 8192)
    if chunk_shots < 1:
        raise ValueError("Shot blocks must be positive")
    previous = 0
    for n in sizes:
        if not previous < n <= len(detectors):
            raise ValueError("Logical count prefixes must be increasing and within the sample arrays")
        for start in range(previous, n, chunk_shots):
            section = slice(start, min(n, start + chunk_shots))
            bits = observables[section, 0]
            if not np.all((bits == 0) | (bits == 1)):
                raise ValueError("Calibration logical outcomes must be binary")
            hits = bits == 1
            if np.any(hits):
                moments = compute_moments(detectors[section][hits], **resources)
                means, coincidences = integer_counts(moments)
                logical_count += moments.shots
                detector_count += means
                pair_count += coincidences
        yield logical_count, detector_count.copy(), pair_count.copy()
        previous = n


@dataclass(frozen=True)
class LogicalLabelFit:
    arrays: dict[str, np.ndarray]
    diagnostics: dict

    @property
    def valid(self) -> bool:
        return self.diagnostics["status"] == "complete"


def fit_logical_labels(moments: Moments, parities: LogicalParities, *,
                       raw_pair_attenuation: np.ndarray | None = None,
                       raw_singleton_attenuation: np.ndarray | None = None) -> LogicalLabelFit:
    """Estimate label bits from full-family parity contrasts, leaving detector rates fixed."""
    family = CandidateSet(len(moments.means))
    i, j = family.pairs
    if parities.detector_logical.shape != moments.means.shape or parities.pair_logical.shape != i.shape:
        raise ValueError("Augmented parity dimensions do not match the detector family")
    if raw_pair_attenuation is None or raw_singleton_attenuation is None:
        raw = invert_detector_moments(moments)
        raw_pair_attenuation, raw_singleton_attenuation = raw.pair_attenuation, raw.singleton_attenuation

    detector_bias, pair_bias = detector_biases(moments, family)
    lam = attenuation(detector_bias)
    lam_ij = attenuation(pair_bias)
    lam_l = float(attenuation(np.array(parities.logical)))
    lam_il = attenuation(parities.detector_logical)
    lam_ijl = attenuation(parities.pair_logical)
    diagnostics = dict(
        policy=LABEL_POLICY, calibration_observables_used=True,
        nonpositive_logical_bias=bool(parities.logical <= 0),
        nonpositive_detector_logical_parities=int(np.count_nonzero(parities.detector_logical <= 0)),
        nonpositive_pair_logical_parities=int(np.count_nonzero(parities.pair_logical <= 0)),
    )
    arrays = dict(logical_bias=np.array(parities.logical), detector_logical_bias=parities.detector_logical,
                  pair_logical_bias=parities.pair_logical)
    pair_1 = (lam[i] + lam[j] + lam_l - lam_ij - lam_il[i] - lam_il[j] + lam_ijl) / 4
    singleton_1 = (lam + lam_l - lam_il) / 2 - family.incident(pair_1)
    pure = lam_l - singleton_1.sum() - pair_1.sum()
    variant_1 = np.concatenate((pair_1, singleton_1))
    total = np.concatenate((raw_pair_attenuation, raw_singleton_attenuation))
    variant_0 = total - variant_1
    contrast = variant_1 - variant_0
    arrays.update(attenuation_label_0=variant_0, attenuation_label_1=variant_1,
                  raw_prior_label_0=attenuation_to_prior(variant_0), raw_prior_label_1=attenuation_to_prior(variant_1),
                  contrast=contrast, pure_logical_attenuation=np.array(pure))
    if not np.all(np.isfinite(contrast)) or not np.isfinite(pure):
        diagnostics.update(status="invalid_logical_parity", pure_logical_attenuation=None)
        return LogicalLabelFit(arrays, diagnostics)
    # A single nonnegative variant per support would leave the other at zero.
    discrepancy = np.minimum(np.abs(variant_0), np.abs(variant_1))
    arrays.update(label_bits=(contrast > 0).astype(np.uint8), ambiguous=(contrast == 0),
                  unique_variant_discrepancy=discrepancy)
    diagnostics.update(status="complete", pure_logical_attenuation=float(pure),
                       ambiguous_candidates=int(np.count_nonzero(arrays["ambiguous"])),
                       negative_variant_0=int(np.count_nonzero(variant_0 < 0)),
                       negative_variant_1=int(np.count_nonzero(variant_1 < 0)),
                       max_unique_variant_discrepancy=float(discrepancy.max()))
    return LogicalLabelFit(arrays, diagnostics)
