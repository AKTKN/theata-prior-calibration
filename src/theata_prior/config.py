"""Validated, serializable experiment configuration; no hidden oracle settings."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
import os
from pathlib import Path
from typing import Any

import yaml

THEATA_REVISION = "8bfb9b97de91bacac2f56725c27b4c71c6a91009"
DEFAULT_COOL_THRESHOLDS = (0.25, 0.5, 1.0, 1.5, 2.0, 4.0, None)
LEGACY_PHYSICAL_ERROR_RATES = (0.0005, 0.001, 0.002, 0.004)


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
    physical_error_rates: tuple[float, ...] | None = None
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
    cool_thresholds: tuple[float | None, ...] | None = None
    liz_pt_thresholds: tuple[float, ...] | None = None
    rounds: int | None = None
    liz_pt_max_iters: int | None = None
    cooling_duration_us: float | None = None
    residual_phonon: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "distances", tuple(self.distances))
        object.__setattr__(self, "training_shots", tuple(self.training_shots))
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
        if len(set(self.distances)) != len(self.distances):
            raise ValueError("distances must not repeat")
        if not 0 < self.probability_floor < self.probability_ceiling < 0.5:
            raise ValueError("require 0 < probability_floor < probability_ceiling < 0.5")
        if type(self.strict_estimation) is not bool:
            raise ValueError("strict_estimation must be a boolean")
        allowed = {"liz_capacity", "memory_capacity", "compiler_options", "noise_parameters", "time_parameters", "lookup_mode", "phonon_scale", "strict_lut"}
        if not isinstance(self.hardware, dict):
            raise ValueError("hardware must be a mapping")
        unknown = set(self.hardware) - allowed
        if unknown:
            raise ValueError(f"unknown hardware keys: {sorted(unknown)}")
        options = self.hardware.get("compiler_options", {})
        if not isinstance(options, dict):
            raise ValueError("hardware.compiler_options must be a mapping")
        if "strict_lut" in self.hardware and type(self.hardware["strict_lut"]) is not bool:
            raise ValueError("hardware.strict_lut must be a boolean")
        lpt_defaults = {"rounds": 3, "liz_pt_max_iters": 32, "cooling_duration_us": 1000.0, "residual_phonon": 0.0}
        if self.liz_pt_thresholds is not None:
            if self.cool_thresholds is not None or self.physical_error_rates is not None:
                raise ValueError("Specify only one sweep parameter")
            if self.basis != "z" or any(d not in (3, 5, 7, 9, 11, 13) for d in self.distances):
                raise ValueError("The THeatA LPT workload supports Z memory at distances 3, 5, 7, 9, 11 and 13")
            if self.sample_source != "circuit" or self.hardware or self.rounds_per_distance != 1:
                raise ValueError("LPT uses circuit sampling, the paper hardware profile and a fixed rounds setting")
            values = self.liz_pt_thresholds
            if (not isinstance(values, (list, tuple)) or not values
                    or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values)
                    or len(set(map(float, values))) != len(values)):
                raise ValueError("liz_pt_thresholds must contain distinct finite nonnegative numbers")
            object.__setattr__(self, "liz_pt_thresholds", tuple(float(value or 0) for value in values))
            for name, default in lpt_defaults.items():
                value = getattr(self, name)
                value = default if value is None else value
                if name in ("rounds", "liz_pt_max_iters"):
                    valid = type(value) is int and value > 0
                else:
                    valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
                    if name == "cooling_duration_us":
                        valid = valid and value > 0
                if not valid:
                    raise ValueError(f"Invalid LPT {name}")
                object.__setattr__(self, name, value)
            return
        if any(getattr(self, name) is not None for name in lpt_defaults):
            raise ValueError("LPT settings require liz_pt_thresholds")
        if self.physical_error_rates is not None:
            if self.cool_thresholds is not None:
                raise ValueError("Specify only one of cool_thresholds and physical_error_rates")
            rates = tuple(self.physical_error_rates)
            if not rates or any(not math.isfinite(p) or not 0 < p < 0.5 for p in rates):
                raise ValueError("physical_error_rates must be finite and in (0, 0.5)")
            if tuple(sorted(set(rates))) != rates:
                raise ValueError("physical_error_rates must be strictly increasing")
            object.__setattr__(self, "physical_error_rates", rates)
        else:
            thresholds = DEFAULT_COOL_THRESHOLDS if self.cool_thresholds is None else self.cool_thresholds
            invalid_thresholds = "cool_thresholds must be a nonempty sequence of finite nonnegative numbers or null"
            if (not isinstance(thresholds, (list, tuple)) or not thresholds
                    or any(value is not None and type(value) not in (int, float) for value in thresholds)):
                raise ValueError(invalid_thresholds)
            try:
                thresholds = tuple(None if value is None else float(0 if value == 0 else value) for value in thresholds)
            except OverflowError as error:
                raise ValueError(invalid_thresholds) from error
            if any(value is not None and (not math.isfinite(value) or value < 0) for value in thresholds):
                raise ValueError(invalid_thresholds)
            if len(set(thresholds)) != len(thresholds):
                raise ValueError("cool_thresholds must not repeat")
            if "cool_threshold_transversal" in options:
                raise ValueError("Set cool_thresholds only; remove hardware.compiler_options.cool_threshold_transversal")
            object.__setattr__(self, "cool_thresholds", thresholds)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return {key: item for key, item in value.items() if item is not None}

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
    if "physical_error_rates" in data:
        raise ValueError("Replace physical_error_rates with cool_thresholds or liz_pt_thresholds for the calibration sweep")
    if "cool_thresholds" in data and data["cool_thresholds"] is None:
        raise ValueError("Use cool_thresholds: [null] for a no-cooling-only sweep")
    if "liz_pt_thresholds" in data:
        if data["liz_pt_thresholds"] is None:
            raise ValueError("liz_pt_thresholds must be a sequence")
        data.setdefault("sample_source", "circuit")
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

