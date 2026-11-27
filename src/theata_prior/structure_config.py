"""Strict configuration for the separate unknown-support experiment."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml

from .candidates import CANDIDATE_SCHEMA
from .config import ExperimentConfig, LEGACY_PHYSICAL_ERROR_RATES, THEATA_REVISION, _base_config, available_cpus
from .logical_labels import LABEL_POLICY
from .structure_estimation import MAX_COUNT_SHOTS

STRUCTURE_SCHEMA = 1
SUPPORT_METHOD = "fisher_pair_holm_and_complete_singleton_influence_wald_holm_v1"
RATE_POLICY = "zero_clip_exact_pairs_and_singleton_refit_with_upper_cap_v1"


@dataclass(frozen=True)
class StructureExperimentConfig:
    base: ExperimentConfig
    alpha_pairs: float = 0.025
    alpha_singletons: float = 0.025
    label_mode: str = "detector_only"
    case_jobs: int = 1

    def __post_init__(self) -> None:
        if self.base.physical_error_rates is None:
            raise ValueError("Structure learning uses physical_error_rates")
        for name in ("alpha_pairs", "alpha_singletons"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not 0 < value < 1:
                raise ValueError(f"{name} must be a number in (0, 1)")
        if self.label_mode not in ("detector_only", "calibration_observables"):
            raise ValueError("label_mode must be detector_only or calibration_observables")
        if type(self.case_jobs) is not int or self.case_jobs < 1:
            raise ValueError("case_jobs must be a positive integer")
        if max(self.base.training_shots) > MAX_COUNT_SHOTS:
            raise ValueError(f"Structure learning supports at most {MAX_COUNT_SHOTS} calibration shots")

    def to_base_config(self) -> ExperimentConfig:
        return replace(self.base, workers=min(self.base.workers, available_cpus()))

    def scientific_dict(self) -> dict:
        return {
            "schema_version": STRUCTURE_SCHEMA,
            "candidate_schema": CANDIDATE_SCHEMA,
            "support_method": SUPPORT_METHOD,
            "rate_policy": RATE_POLICY,
            "label_mode": self.label_mode,
            "label_policy": LABEL_POLICY if self.label_mode == "calibration_observables" else None,
            "alpha_pairs": self.alpha_pairs,
            "alpha_singletons": self.alpha_singletons,
            "theata_reference_revision": THEATA_REVISION,
            "base": self.base.scientific_dict(),
        }


def load_structure_config(path: str | Path, *, workers: int | None = None,
                          theata_path: str | None = None) -> StructureExperimentConfig:
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("configuration must be a YAML mapping")
    options = data.pop("structure", {})
    if not isinstance(options, dict):
        raise ValueError("structure options must be a YAML mapping")
    allowed = {field.name for field in fields(StructureExperimentConfig)} - {"base"}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unknown structure options: {sorted(unknown)}")
    data.setdefault("physical_error_rates", LEGACY_PHYSICAL_ERROR_RATES)
    base = _base_config(data, path, workers=workers, theata_path=theata_path)
    return StructureExperimentConfig(base, **options)
