"""Deterministic edge groups using only the prior-free structure."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dem import GraphStructure


@dataclass(frozen=True)
class GroupMap:
    edge_groups: tuple[int, ...]
    labels: tuple[str, ...]
    sizes: tuple[int, ...]
    reference: int
    scheme: str
    features: str

    @property
    def free(self) -> tuple[int, ...]:
        return tuple(i for i in range(len(self.labels)) if i != self.reference)


def make_groups(structure: GraphStructure, scheme: str = "coarse", *,
                verified_coordinates: bool = False, manual: tuple[str, ...] | None = None) -> GroupMap:
    if not structure.edges:
        raise ValueError("decoder training requires at least one mechanism")
    coords = structure.detector_coordinates
    if scheme == "manual":
        if manual is None or isinstance(manual, str) or len(manual) != structure.num_errors:
            raise ValueError("manual grouping must have one group label per canonical edge")
        if any(not isinstance(label, str) or not label.strip() for label in manual):
            raise ValueError("manual group labels must be nonempty strings")
        edge_labels = list(manual)
        features = "User-supplied labels aligned to GraphStructure.edges"
    elif scheme == "coarse" and verified_coordinates and coords and all(
        len(coord) == 3 and np.isfinite(coord).all() for coord in coords
    ) and min(c[2] for c in coords) < max(c[2] for c in coords):
        first = min(c[2] for c in coords)
        last = max(c[2] for c in coords)
        edge_labels = []
        for edge in structure.edges:
            locations = [coords[i] for i in edge.detectors]
            times = [loc[2] for loc in locations]
            location = "initial" if first in times else "final" if last in times else "bulk"
            if len(locations) == 1:
                kind = "singleton"
            else:
                dx, dy, dt = np.abs(np.subtract(*locations))
                spatial = "same site" if dx == dy == 0 else "axial" if dx * dy == 0 else "diagonal"
                kind = f"pair {spatial}, {'same round' if dt == 0 else 'cross round'}"
            edge_labels.append(f"{kind} / {location}")
        features = ("Verified Stim (x, y, round) coordinates. Pair displacement is orientation invariant and "
                    "classified by spatial axis occupancy and whether time changes. Location uses only observed "
                    "endpoints: initial if touching first, else final if touching last, otherwise bulk.")
    elif scheme in ("coarse", "support"):
        scheme = "support_fallback" if scheme == "coarse" else "support"
        edge_labels = ["singleton" if len(edge.detectors) == 1 else "pair" for edge in structure.edges]
        features = "Detector target count. No space or time interpretation is used."
    else:
        raise ValueError(f"Unknown grouping scheme {scheme!r}")
    labels = tuple(sorted(set(edge_labels)))
    ids = {label: i for i, label in enumerate(labels)}
    edge_groups = tuple(ids[label] for label in edge_labels)
    sizes = tuple(int(n) for n in np.bincount(edge_groups))
    reference = max(range(len(labels)), key=lambda i: (sizes[i], -i))
    return GroupMap(edge_groups, labels, sizes, reference, scheme, features)
