"""Canonical mechanism, pair-matrix, and singleton diagnostics."""
from __future__ import annotations

import numpy as np

from ..dem import GraphStructure
from .grouping import GroupMap
from .weights import llrs


def prior_matrices(structure: GraphStructure, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return P(q), singleton values and the two structural support masks."""
    if np.shape(probabilities) != (structure.num_errors,):
        raise ValueError("prior vector must match the canonical edge order")
    size = structure.num_detectors
    pair = np.zeros((size, size))
    singleton = np.zeros(size)
    pair_mask = np.zeros_like(pair, dtype=bool)
    singleton_mask = np.zeros(size, dtype=bool)
    for edge, p in zip(structure.edges, probabilities, strict=True):
        i = edge.detectors[0]
        if len(edge.detectors) == 2:
            j = edge.detectors[1]
            pair[i, j] = pair[j, i] = p
            pair_mask[i, j] = pair_mask[j, i] = True
        else:
            singleton[i] = p
            singleton_mask[i] = True
    return pair, singleton, pair_mask, singleton_mask


def errors(delta: np.ndarray, *, finite_only: bool = False) -> dict:
    delta = np.asarray(delta)
    finite = delta[np.isfinite(delta)]
    total, count = delta.size, finite.size
    available = count > 0 and (finite_only or count == total)
    return {"status": "ok" if available else ("empty" if not total else "unavailable"),
            "rmse": float(np.sqrt(np.mean(finite ** 2))) if available else None,
            "mae": float(np.mean(np.abs(finite))) if available else None,
            "count": count, "total": total, "invalid_count": total - count,
            "coverage": count / total if total else None, "finite_only": finite_only}


def pair_errors(delta: np.ndarray, pair_mask: np.ndarray, *, finite_only: bool = False) -> dict[str, dict]:
    upper = np.triu(np.ones(delta.shape, dtype=bool), 1)
    return {"pair_matrix_support": errors(delta[upper & pair_mask], finite_only=finite_only),
            "pair_matrix_all": errors(delta[upper], finite_only=finite_only)}


def compare_priors(structure: GraphStructure, estimate: np.ndarray, actual: np.ndarray,
                   groups: GroupMap) -> dict[str, dict]:
    delta = estimate - actual
    is_pair = np.array([len(edge.detectors) == 2 for edge in structure.edges], dtype=bool)
    p_hat, _, mask, _ = prior_matrices(structure, estimate)
    p_actual, _, _, _ = prior_matrices(structure, actual)
    results = {"mechanisms": errors(delta), "pair_mechanisms": errors(delta[is_pair]),
               "singleton_mechanisms": errors(delta[~is_pair]), **pair_errors(p_hat - p_actual, mask)}
    try:
        estimated_weights, actual_weights = llrs(estimate), llrs(actual)
        results["normalized_weights"] = errors(
            estimated_weights / estimated_weights.mean() - actual_weights / actual_weights.mean())
    except ValueError as error:
        results["normalized_weights"] = {**errors(np.full(delta.shape, np.nan)), "reason": str(error)}
    for index, label in enumerate(groups.labels):
        mask = np.asarray(groups.edge_groups) == index
        results[f"group:{label}"] = {**errors(delta[mask]), "group_id": index,
                                    "signed_mean": float(delta[mask].mean())}
    return results
