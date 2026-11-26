"""Evaluation-only comparisons in the canonical detector-support ordering."""
from __future__ import annotations

import numpy as np

from .candidates import CandidateSet
from .dem import GraphStructure, model_moments
from .estimation import Moments


def canonical_truth(family: CandidateSet, graph: GraphStructure,
                    probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if family.num_detectors != graph.num_detectors or graph.num_observables != 1:
        raise ValueError("Expected the declared detector family and one logical observable")
    priors = np.zeros(len(family))
    labels = np.zeros(len(family), dtype=np.uint8)
    for edge, prior in zip(graph.edges, probabilities, strict=True):
        index = family.index(edge.detectors)
        priors[index] = prior
        labels[index] = bool(edge.observables)
    return priors, labels


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def score_support(family: CandidateSet, truth: np.ndarray, estimated: np.ndarray | None,
                  selected: np.ndarray | None) -> list[dict]:
    """Report errors on the full family, keeping undefined fits separate from FN."""
    p = family.num_pairs
    groups = (("pair", slice(0, p)), ("singleton", slice(p, None)), ("combined", slice(None)))
    rows = []
    for group, span in groups:
        true = truth[span]
        positive = true > 0
        row = dict(support_type=group, candidates=len(true), true_active=int(positive.sum()))
        if estimated is None or selected is None:
            rows.append(dict(row, unresolved=len(true), selected=None, active=None, tp=None, fp=None, fn=None,
                             tn=None, precision=None, recall=None, f1=None, false_discovery_proportion=None,
                             exact_support_recovery=None, missed_true_prior_mass=None, false_positive_prior_mass=None))
            continue
        rate = estimated[span]
        found = rate > 0
        tp = int(np.count_nonzero(positive & found))
        fp = int(np.count_nonzero(~positive & found))
        fn = int(np.count_nonzero(positive & ~found))
        tn = len(true) - tp - fp - fn
        rows.append(dict(row, unresolved=0, selected=int(np.count_nonzero(selected[span])),
                         active=tp + fp, tp=tp, fp=fp, fn=fn, tn=tn,
                         precision=_ratio(tp, tp + fp), recall=_ratio(tp, tp + fn),
                         f1=_ratio(2 * tp, 2 * tp + fp + fn),
                         false_discovery_proportion=_ratio(fp, tp + fp),
                         exact_support_recovery=fp + fn == 0,
                         missed_true_prior_mass=float(true[~found].sum()),
                         false_positive_prior_mass=float(rate[~positive].sum())))
    return rows


def _error_stats(error: np.ndarray) -> tuple[float | None, float | None]:
    if not len(error):
        return None, None
    return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error**2)))


def score_priors(family: CandidateSet, truth: np.ndarray, estimated: np.ndarray,
                 predicted_graph: GraphStructure, positive_priors: np.ndarray, heldout: Moments,
                 raw: np.ndarray | None = None, refit: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
    true = truth > 0
    residual = estimated - truth
    mae, rmse = _error_stats(residual)
    true_mae, true_rmse = _error_stats(residual[true])
    mean, second = model_moments(predicted_graph, positive_priors)
    mean_mae, mean_rmse = _error_stats(mean - heldout.means)
    pair_mae, pair_rmse = _error_stats((second - heldout.second_moments)[family.pairs])
    spurious = np.where(true, 0, estimated)
    pair_count = family.num_pairs
    incident = family.incident(spurious[:pair_count]) + spurious[pair_count:]
    row = dict(mae=mae, rmse=rmse, true_support_mae=true_mae, true_support_rmse=true_rmse,
               false_positive_prior_mass=float(spurious.sum()),
               max_spurious_incident_rate=float(incident.max()), mean_spurious_incident_rate=float(incident.mean()),
               heldout_mean_mae=mean_mae, heldout_mean_rmse=mean_rmse,
               heldout_second_moment_mae=pair_mae, heldout_second_moment_rmse=pair_rmse)
    if raw is not None:
        finite = np.isfinite(raw)
        raw_mae, raw_rmse = _error_stats(raw[finite] - truth[finite])
        row.update(raw_mae=raw_mae, raw_rmse=raw_rmse, raw_finite=int(finite.sum()),
                   negative_raw_singletons=int(np.count_nonzero(raw[pair_count:] < 0)))
    if refit is not None:
        finite = np.isfinite(refit)
        refit_mae, refit_rmse = _error_stats(refit[finite] - truth[finite])
        row.update(refit_mae=refit_mae, refit_rmse=refit_rmse,
                   rates_changed_after_refit=int(np.count_nonzero(refit != estimated)))
    return row, incident


def score_labels(truth: np.ndarray, true_labels: np.ndarray, estimated: np.ndarray,
                 labels: dict[str, np.ndarray]) -> dict:
    true = truth > 0
    matched = true & (estimated > 0)
    denominator = int(matched.sum())
    errors = int(np.count_nonzero(matched & (labels["label_bits"] != true_labels)))
    ambiguous = int(np.count_nonzero(matched & labels["ambiguous"]))
    return {
        "matched_true_supports": denominator,
        "missed_true_supports": int(np.count_nonzero(true & ~matched)),
        "errors_on_matched_supports": errors,
        "error_rate": _ratio(errors, denominator),
        "ambiguous_on_matched_supports": ambiguous,
        "ambiguity_rate": _ratio(ambiguous, denominator),
        "mean_absolute_contrast": float(np.abs(labels["contrast"][matched]).mean()) if denominator else None,
        "mean_unique_variant_discrepancy": float(labels["unique_variant_discrepancy"][matched].mean()) if denominator else None,
        "pure_logical_attenuation": float(labels["pure_logical_attenuation"]),
    }
