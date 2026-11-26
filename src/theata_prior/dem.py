"""Probability-free DEM topology and rigorously graphlike sector projections.

An ``error(p) A ^ B`` instruction is ONE Bernoulli event.  The separator is
never interpreted as independence here: all its targets are XORed before
projection.  A projection is accepted only when every surviving event has
one or two detectors and its logical action is identifiable from its support.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np
import stim


class DemStructureError(ValueError):
    """The model cannot be identified with the graphlike covariance method."""


@dataclass(frozen=True, order=True)
class ErrorSupport:
    """A single edge's local detector IDs and logical observable IDs; no prior."""

    detectors: tuple[int, ...]
    observables: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "detectors", tuple(int(x) for x in self.detectors))
        object.__setattr__(self, "observables", tuple(int(x) for x in self.observables))
        for label, values in (("detectors", self.detectors), ("observables", self.observables)):
            if tuple(sorted(set(values))) != values or any(x < 0 for x in values):
                raise DemStructureError(f"{label} must be sorted, distinct nonnegative IDs")
        if len(self.detectors) not in (1, 2):
            raise DemStructureError("Each identifiable graph edge must have 1 or 2 detectors")

    def to_dict(self) -> dict[str, list[int]]:
        return {"detectors": list(self.detectors), "observables": list(self.observables)}


@dataclass(frozen=True)
class GraphStructure:
    """Identifiable graphlike topology, explicitly excluding error probabilities.

    ``detector_ids[k]`` is the detector ID in the source circuit corresponding
    to local detector ``k``.  ``edges`` fixes the ordering of prior vectors.
    """

    num_detectors: int
    num_observables: int
    edges: tuple[ErrorSupport, ...]
    detector_ids: tuple[int, ...] = ()
    detector_coordinates: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.num_detectors < 1 or self.num_observables < 0:
            raise DemStructureError("Positive detector count and nonnegative observable count required")
        object.__setattr__(self, "edges", tuple(self.edges))
        ids = tuple(self.detector_ids) or tuple(range(self.num_detectors))
        if len(ids) != self.num_detectors or len(set(ids)) != len(ids) or any(i < 0 for i in ids):
            raise DemStructureError("detector_ids must identify each local detector exactly once")
        object.__setattr__(self, "detector_ids", tuple(int(x) for x in ids))
        coordinates = tuple(tuple(float(v) for v in c) for c in self.detector_coordinates)
        if coordinates and len(coordinates) != self.num_detectors:
            raise DemStructureError("detector_coordinates must have one entry per detector")
        object.__setattr__(self, "detector_coordinates", coordinates)
        seen: dict[tuple[int, ...], tuple[int, ...]] = {}
        for edge in self.edges:
            if any(i >= self.num_detectors for i in edge.detectors):
                raise DemStructureError("Edge detector ID exceeds num_detectors")
            if any(i >= self.num_observables for i in edge.observables):
                raise DemStructureError("Edge observable ID exceeds num_observables")
            if edge.detectors in seen:
                raise DemStructureError(
                    "Repeated detector support must be aggregated first; different logical "
                    "labels for one support are not identifiable from detector samples"
                )
            seen[edge.detectors] = edge.observables

    @property
    def num_errors(self) -> int:
        return len(self.edges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "num_detectors": self.num_detectors,
            "num_observables": self.num_observables,
            "detector_ids": list(self.detector_ids),
            "detector_coordinates": [list(c) for c in self.detector_coordinates],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GraphStructure":
        if value.get("schema_version", 1) != 1:
            raise DemStructureError("Unsupported graph structure schema version")
        return cls(
            num_detectors=int(value["num_detectors"]),
            num_observables=int(value["num_observables"]),
            edges=tuple(ErrorSupport(tuple(e["detectors"]), tuple(e["observables"])) for e in value["edges"]),
            detector_ids=tuple(value.get("detector_ids", ())),
            detector_coordinates=tuple(tuple(c) for c in value.get("detector_coordinates", ())),
        )

    def to_dem(self, probabilities: Sequence[float]) -> stim.DetectorErrorModel:
        """Attach an explicitly supplied prior vector to this topology."""
        p = np.asarray(probabilities, dtype=np.float64)
        if p.shape != (self.num_errors,) or not np.all(np.isfinite(p)):
            raise ValueError("A finite probability is required for each graph edge")
        if np.any(p < 0) or np.any(p >= 0.5):
            raise ValueError("This identifiable branch requires 0 <= p < 1/2")
        dem = stim.DetectorErrorModel()
        for edge, prior in zip(self.edges, p, strict=True):
            targets = [stim.target_relative_detector_id(i) for i in edge.detectors]
            targets.extend(stim.target_logical_observable_id(i) for i in edge.observables)
            dem.append("error", float(prior), targets)
        # Explicit declarations preserve isolated detectors and observables.
        for i in range(self.num_detectors):
            coords = self.detector_coordinates[i] if self.detector_coordinates else ()
            dem.append("detector", list(coords), [stim.target_relative_detector_id(i)])
        for i in range(self.num_observables):
            dem.append("logical_observable", [], [stim.target_logical_observable_id(i)])
        return dem


@dataclass(frozen=True)
class ProjectedModel:
    """Oracle marginal DEM kept separate from the prior-free structure."""

    structure: GraphStructure
    probabilities: np.ndarray
    dem: stim.DetectorErrorModel
    diagnostics: dict[str, Any]


def _toggle(targets: set[int], value: int) -> None:
    if value in targets:
        targets.remove(value)
    else:
        targets.add(value)


def project_dem(
    dem: stim.DetectorErrorModel,
    detector_ids: Iterable[int] | None = None,
) -> ProjectedModel:
    """Marginalize a DEM onto selected detectors while preserving logical labels.

    Repeated identical supports represent independent events whose XOR prior
    is ``(1-prod(1-2*p))/2``.  Original error-instruction components separated
    by ``^`` remain one event.  Events invisible both to the selected detectors
    and to logical observables integrate out.  Invisible logical events,
    support/logical ambiguities and residual hyperedges raise explicitly.

    This operation is exact for the input independent-Bernoulli DEM.  Any
    circuit-to-DEM approximation was made upstream and is not repaired here.
    """
    ids = tuple(range(dem.num_detectors)) if detector_ids is None else tuple(int(x) for x in detector_ids)
    if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= dem.num_detectors for i in ids):
        raise DemStructureError("detector_ids must be a nonempty selection of distinct valid IDs")
    local = {original: i for i, original in enumerate(ids)}
    groups: dict[ErrorSupport, float] = {}
    support_labels: dict[tuple[int, ...], tuple[int, ...]] = {}
    source_events = dropped = separators = merged = 0
    raw_weights: dict[str, int] = {}
    projected_weights: dict[str, int] = {}
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        source_events += 1
        prior = float(instruction.args_copy()[0])
        if not math.isfinite(prior) or not 0 <= prior <= 1:
            raise DemStructureError("Every input Bernoulli event must satisfy 0 <= p <= 1")
        all_detectors: set[int] = set()
        observables: set[int] = set()
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                _toggle(all_detectors, target.val)
            elif target.is_logical_observable_id():
                _toggle(observables, target.val)
            elif target.is_separator():
                separators += 1
        raw_weights[str(len(all_detectors))] = raw_weights.get(str(len(all_detectors)), 0) + 1
        selected = tuple(sorted(local[i] for i in all_detectors if i in local))
        logical = tuple(sorted(observables))
        projected_weights[str(len(selected))] = projected_weights.get(str(len(selected)), 0) + 1
        if not selected:
            if logical:
                raise DemStructureError(
                    f"An event flips L{logical} but no selected detector; its prior cannot "
                    "be identified from detector data. Choose the appropriate memory sector."
                )
            dropped += 1
            continue
        # An event invisible to all retained variables integrates out at ANY
        # valid Bernoulli probability, including a fair or deterministic flip.
        # Only surviving events need the positive-bias identifiable branch.
        if prior >= 0.5:
            raise DemStructureError("Every surviving Bernoulli event must satisfy 0 <= p < 1/2")
        if len(selected) > 2:
            raise DemStructureError(
                f"Residual hyperedge {selected} after projection: S11/S14-S16 do not "
                "identify its individual prior. No separator splitting is performed."
            )
        if selected in support_labels and support_labels[selected] != logical:
            raise DemStructureError(
                f"Detector support {selected} has logical labels {support_labels[selected]} "
                f"and {logical}; detector-only calibration cannot identify them separately"
            )
        support_labels[selected] = logical
        edge = ErrorSupport(selected, logical)
        if edge in groups:
            merged += 1
        # log(1-2p) sums more accurately than a long product at small p.
        groups[edge] = groups.get(edge, 0.0) + math.log1p(-2 * prior)
    edges = tuple(sorted(groups))
    probabilities = np.array([-0.5 * math.expm1(groups[e]) for e in edges], dtype=np.float64)
    if np.any(probabilities >= 0.5):
        raise DemStructureError("Merged edge probability is numerically saturated at 1/2")
    source_coords = dem.get_detector_coordinates()
    structure = GraphStructure(
        num_detectors=len(ids), num_observables=dem.num_observables, edges=edges,
        detector_ids=ids, detector_coordinates=tuple(tuple(source_coords.get(i, ())) for i in ids),
    )
    diagnostics = {
        "source_error_instructions": source_events,
        "discarded_invisible_nonlogical_events": dropped,
        "merged_identical_support_events": merged,
        "separator_targets_treated_as_correlated": separators,
        "source_detector_weight_histogram": raw_weights,
        "projected_detector_weight_histogram_before_aggregation": projected_weights,
        "projected_edges": len(edges),
        "projection_is_exact_for_independent_dem": True,
    }
    return ProjectedModel(structure, probabilities, structure.to_dem(probabilities), diagnostics)


def model_moments(structure: GraphStructure, probabilities: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Exact population moments of a graphlike model, for validation only.

    These oracle moments must never replace calibration samples in an experiment.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (structure.num_errors,) or not np.all(np.isfinite(p)) or np.any((p < 0) | (p >= 0.5)):
        raise ValueError("Invalid prior vector")
    m = structure.num_detectors
    log_bias = np.zeros(m, dtype=np.float64)
    shared_log_bias = np.zeros((m, m), dtype=np.float64)
    for edge, prior in zip(structure.edges, p, strict=True):
        value = math.log1p(-2 * float(prior))
        for i in edge.detectors:
            log_bias[i] += value
        if len(edge.detectors) == 2:
            i, j = edge.detectors
            shared_log_bias[i, j] += value
            shared_log_bias[j, i] += value
    bias = np.exp(log_bias)
    mean = -0.5 * np.expm1(log_bias)
    xor_bias = np.exp(log_bias[:, None] + log_bias[None, :] - 2 * shared_log_bias)
    second = (1 - bias[:, None] - bias[None, :] + xor_bias) * 0.25
    np.fill_diagonal(second, mean)
    return mean, second
