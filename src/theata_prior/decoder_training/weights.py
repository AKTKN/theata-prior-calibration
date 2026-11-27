"""Bounded group corrections with a fixed mean realized decoder weight."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dem import GraphStructure
from .grouping import GroupMap

NORMALIZATION_TOL = 1e-11


def llrs(priors: np.ndarray) -> np.ndarray:
    q = np.asarray(priors, dtype=float)
    if not np.all(np.isfinite(q) & (q > 0) & (q < 0.5)):
        raise ValueError("decoder LLRs require finite probabilities strictly between 0 and 1/2")
    weights = np.asarray(np.log1p(-q) - np.log(q))
    near_half = q > 0.25
    weights[near_half] = np.log1p((1 - 2 * q[near_half]) / q[near_half])
    return weights


@dataclass(frozen=True)
class RealizedPrior:
    priors: np.ndarray
    weights: np.ndarray
    theta: np.ndarray
    scale: float
    clipped_low: int
    clipped_high: int
    normalization_error: float


class WeightModel:
    def __init__(self, initial: np.ndarray, groups: GroupMap, p_min: float, p_max: float):
        if not 0 < p_min < p_max < 0.5:
            raise ValueError("require 0 < p_min < p_max < 1/2")
        self.initial = np.asarray(initial, dtype=float).copy()
        if self.initial.shape != (len(groups.edge_groups),) or not len(self.initial):
            raise ValueError("initial priors must align with the edge-to-group map")
        llrs(self.initial)
        self.initial_clip_low = int(np.count_nonzero(self.initial < p_min))
        self.initial_clip_high = int(np.count_nonzero(self.initial > p_max))
        self.p_min, self.p_max = p_min, p_max
        self.initial = np.clip(self.initial, p_min, p_max)
        self.base = llrs(self.initial)
        self.target = float(self.base.mean())
        self.lo, self.hi = llrs(np.array([p_max, p_min]))
        self.groups = groups
        self.edge_groups = np.asarray(groups.edge_groups)
        self.degenerate = bool(np.all(self.base == self.lo) or np.all(self.base == self.hi))

    def realize(self, free_theta: np.ndarray) -> RealizedPrior:
        free_theta = np.asarray(free_theta, dtype=float)
        if free_theta.shape != (len(self.groups.free),) or not np.all(np.isfinite(free_theta) & (np.abs(free_theta) <= 300)):
            raise ValueError("free theta must be finite, bounded, and aligned to free groups")
        theta = np.zeros(len(self.groups.labels))
        theta[list(self.groups.free)] = free_theta
        if not np.any(theta) or self.degenerate:
            return RealizedPrior(self.initial.copy(), self.base.copy(), theta, 1.0, 0, 0, 0.0)
        unscaled = self.base * np.exp(theta[self.edge_groups])
        scale = self.target / unscaled.mean()
        trial = scale * unscaled
        bounded = np.clip(trial, self.lo, self.hi)
        if abs(bounded.mean() - self.target) > NORMALIZATION_TOL / 4:
            # Bisect in log scale to handle small weights and wide theta bounds.
            lower = np.log(self.lo) - np.log(unscaled.max())
            upper = np.log(self.hi) - np.log(unscaled.min())
            for _ in range(160):
                scale = float(np.exp((lower + upper) / 2))
                trial = scale * unscaled
                bounded = np.clip(trial, self.lo, self.hi)
                residual = bounded.mean() - self.target
                if abs(residual) <= NORMALIZATION_TOL / 4:
                    break
                if residual < 0:
                    lower = np.log(scale)
                else:
                    upper = np.log(scale)
        # This form of the logistic preserves subnormal probabilities.
        priors = np.clip(np.exp(-np.logaddexp(0, bounded)), self.p_min, self.p_max)
        realized = llrs(priors)
        error = self.check_mean(realized)
        return RealizedPrior(priors, realized, theta, float(scale), int(np.count_nonzero(trial < self.lo)),
                             int(np.count_nonzero(trial > self.hi)), error)

    def check_mean(self, weights: np.ndarray) -> float:
        if weights.shape != self.base.shape or not np.all(np.isfinite(weights)):
            raise ValueError("realized decoder weights must be finite and align with the edge order")
        error = float(weights.mean() - self.target)
        if abs(error) > NORMALIZATION_TOL:
            raise ArithmeticError(f"Realized weight normalization error {error:g} exceeds tolerance {NORMALIZATION_TOL:g}")
        return error

    def verify_serialization(self, structure: GraphStructure, dem, expected: RealizedPrior) -> float:
        """Check exported edge signatures, probabilities and the decoded scale."""
        instructions = [instruction for instruction in dem if instruction.type == "error"]
        original = [instruction for instruction in structure.to_dem(expected.priors) if instruction.type == "error"]
        if len(instructions) != len(original) or any(
            a.targets_copy() != b.targets_copy() for a, b in zip(instructions, original, strict=True)
        ):
            raise ValueError("exported DEM mechanism signatures differ from the optimizer model")
        exported = np.array([instruction.args_copy()[0] for instruction in instructions])
        if not np.array_equal(exported, expected.priors):
            raise ArithmeticError("exported prior probabilities differ from the realized optimizer model")
        return self.check_mean(llrs(exported))
