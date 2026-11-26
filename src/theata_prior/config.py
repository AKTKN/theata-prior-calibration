"""Validated, serializable experiment configuration; no hidden oracle settings."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
import os
from pathlib import Path
from typing import Any

import yaml

THEATA_REVISION = "8bfb9b97de91bacac2f56725c27b4c71c6a91009"


def available_cpus() -> int:
    """Respect PBS/SLURM allocations before using machine-wide CPU counts."""
    for key in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NCPUS"):
        value = os.environ.get(key)
        if value and value.isdigit() and int(value) > 0:
            return int(value)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


@dataclass(frozen=True)
class ExperimentConfig:
    distances: tuple[int, ...] = (3,)
    basis: str = "z"
    rounds_per_distance: int = 1
    physical_error_rates: tuple[float, ...] = (0.0005, 0.001, 0.002, 0.004)
    training_shots: tuple[int, ...] = (1000, 10000, 100000)
    test_shots: int = 50000
    repetitions: int = 1
    seed: int = 20261126
    sample_source: str = "dem"
    workers: int = 1
    sample_chunk_shots: int = 16384
    decode_chunk_shots: int = 8192
    covariance_chunk_shots: int = 8192
    covariance_block_detectors: int = 128
    probability_floor: float = 1e-8
    probability_ceiling: float = 0.499999
    strict_estimation: bool = False
    theata_path: str = "vendor/theata"
    hardware: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "distances", tuple(self.distances))
        object.__setattr__(self, "training_shots", tuple(self.training_shots))
        object.__setattr__(self, "physical_error_rates", tuple(self.physical_error_rates))
        if not self.distances or any(type(d) is not int or d < 3 or d % 2 != 1 for d in self.distances):
            raise ValueError("distances must contain odd integers >= 3")
        if self.basis not in {"x", "z"}:
            raise ValueError("basis must be 'x' or 'z'")
        if self.sample_source not in {"dem", "circuit"}:
            raise ValueError("sample_source must be 'dem' or 'circuit'")
        for key in ("rounds_per_distance", "test_shots", "repetitions", "workers", "sample_chunk_shots", "decode_chunk_shots", "covariance_chunk_shots", "covariance_block_detectors"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not self.training_shots or any(type(n) is not int or n < 2 for n in self.training_shots):
            raise ValueError("training_shots must contain integers >= 2")
        if sorted(set(self.training_shots)) != list(self.training_shots):
            raise ValueError("training_shots must be strictly increasing")
        if not self.physical_error_rates or any(not math.isfinite(p) or not 0 < p < 0.5 for p in self.physical_error_rates):
            raise ValueError("physical_error_rates must be finite and in (0, 0.5)")
        if sorted(set(self.physical_error_rates)) != list(self.physical_error_rates):
            raise ValueError("physical_error_rates must be strictly increasing")
        if len(set(self.distances)) != len(self.distances):
            raise ValueError("distances must not repeat")
        if not 0 < self.probability_floor < self.probability_ceiling < 0.5:
            raise ValueError("require 0 < probability_floor < probability_ceiling < 0.5")
        if type(self.strict_estimation) is not bool:
            raise ValueError("strict_estimation must be a boolean")
        allowed = {"liz_capacity", "memory_capacity", "compiler_options", "noise_parameters", "time_parameters", "lookup_mode", "phonon_scale", "strict_lut"}
        unknown = set(self.hardware) - allowed
        if unknown:
            raise ValueError(f"unknown hardware keys: {sorted(unknown)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def scientific_dict(self) -> dict[str, Any]:
        """Fields defining samples, model, or inference, excluding execution resources."""
        value = self.to_dict()
        for key in ("workers", "decode_chunk_shots", "covariance_chunk_shots", "covariance_block_detectors", "theata_path"):
            value.pop(key)
        return value


def load_config(path: str | Path, *, workers: int | None = None, theata_path: str | None = None) -> ExperimentConfig:
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("configuration must be a YAML mapping")
    return _base_config(data, path, workers=workers, theata_path=theata_path)


def _base_config(data: dict, path: Path, *, workers: int | None, theata_path: str | None) -> ExperimentConfig:
    data = dict(data)
    unknown = set(data) - {f.name for f in fields(ExperimentConfig)}
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    if workers is not None:
        data["workers"] = workers
    if data.get("workers") == "auto":
        data["workers"] = available_cpus()
    selected_path = theata_path or os.environ.get("THEATA_PATH") or data.get("theata_path", "vendor/theata")
    # Config files normally reside in <project>/configs. Relative vendor paths
    # are relative to the project; arbitrary paths are relative to the YAML.
    base = path.parent.parent if path.parent.name == "configs" else path.parent
    selected = Path(selected_path).expanduser()
    data["theata_path"] = str((base / selected).resolve() if not selected.is_absolute() else selected.resolve())
    return ExperimentConfig(**data)

