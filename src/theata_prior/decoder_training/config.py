"""Scientific settings and nonoverlapping CPU allocations for decoder training."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
from pathlib import Path

import yaml

from ..config import ExperimentConfig, _base_config, available_cpus

LDPC_VERSION = "2.4.1"
DECODERS = ("mwpm", "uf")


def validate_decoders(values: tuple[str, ...] | list[str], field_name: str) -> tuple[str, ...]:
    if (not isinstance(values, (list, tuple)) or not values
            or any(value not in DECODERS for value in values) or len(set(values)) != len(values)):
        raise ValueError(f"{field_name} must contain distinct selections from {list(DECODERS)}")
    return tuple(values)


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class CEMConfig:
    population_size: int = 6
    elite_count: int = 2
    generations: int = 6
    initial_std: float = 0.5
    smoothing: float = 0.5
    min_std: float = 0.05
    theta_bounds: tuple[float, float] = (-2.0, 2.0)
    train_batch_shots: int | None = 512
    validation_interval: int = 2

    def __post_init__(self) -> None:
        for key in ("population_size", "elite_count", "generations", "validation_interval"):
            _positive(getattr(self, key), key)
        if self.elite_count > self.population_size:
            raise ValueError("elite_count cannot exceed population_size")
        if self.train_batch_shots is not None:
            _positive(self.train_batch_shots, "train_batch_shots")
        if not 0 < self.min_std <= self.initial_std or not math.isfinite(self.initial_std):
            raise ValueError("require 0 < min_std <= initial_std < infinity")
        if not 0 < self.smoothing <= 1:
            raise ValueError("smoothing must be in (0, 1]")
        bounds = tuple(self.theta_bounds)
        if len(bounds) != 2 or not -300 <= bounds[0] < 0 < bounds[1] <= 300:
            raise ValueError("theta_bounds must enclose 0 and lie in [-300, 300]")
        object.__setattr__(self, "theta_bounds", bounds)


@dataclass(frozen=True)
class Resources:
    total_workers: int = field(default_factory=lambda: min(2, available_cpus()))
    case_jobs: int = 1
    candidate_jobs: int | None = None
    sampling_jobs: int | None = None
    covariance_jobs: int = 1
    native_threads: int = 1

    def __post_init__(self) -> None:
        for key in ("total_workers", "case_jobs", "native_threads"):
            _positive(getattr(self, key), key)
        if self.candidate_jobs is None:
            object.__setattr__(self, "candidate_jobs", max(1, min(2, self.total_workers // (self.case_jobs * self.native_threads))))
        if self.sampling_jobs is None:
            object.__setattr__(self, "sampling_jobs", min(2, self.total_workers))
        for item in fields(self):
            _positive(getattr(self, item.name), item.name)
        if self.total_workers > available_cpus():
            raise ValueError("total_workers exceeds the available CPU allocation")
        peak = max(self.case_jobs * self.candidate_jobs * self.native_threads,
                   self.sampling_jobs, self.covariance_jobs)
        if peak > self.total_workers:
            raise ValueError("CPU allocation exceeds total_workers")


def _physical_defaults() -> ExperimentConfig:
    return ExperimentConfig(physical_error_rates=(0.001,), training_shots=(2048,), test_shots=4096,
                            sample_chunk_shots=2048, decode_chunk_shots=2048, covariance_chunk_shots=2048)


@dataclass(frozen=True)
class TrainingConfig:
    physical: ExperimentConfig = field(default_factory=_physical_defaults)
    rounds_by_distance: dict[int, int] = field(default_factory=dict)
    validation_shots: int = 2048
    optimizer_seeds: tuple[int, ...] = (0,)
    initializations: tuple[str, ...] = ("covariance",)
    covariance_failure: str = "skip"
    uniform_probability: float = 0.001
    p_min: float = 1e-9
    p_max: float = 0.49
    grouping: str = "coarse"
    manual_groups: tuple[str, ...] | dict[str, tuple[str, ...]] | None = None
    target_decoders: tuple[str, ...] = DECODERS
    evaluation_decoders: tuple[str, ...] = DECODERS
    ldpc_version: str = LDPC_VERSION
    uf_mode: str = "peeling"
    bits_per_step: int = 1
    cem: CEMConfig = field(default_factory=CEMConfig)
    resources: Resources = field(default_factory=Resources)
    plot_decoders: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _positive(self.validation_shots, "validation_shots")
        if self.physical.physical_error_rates is None:
            raise ValueError("Decoder training uses physical_error_rates")
        if self.physical.workers != 1:
            raise ValueError("set decoder-training worker allocations in resources")
        for distance, rounds in self.rounds_by_distance.items():
            if type(distance) is not int or distance not in self.physical.distances:
                raise ValueError("rounds_by_distance keys must be configured distances")
            _positive(rounds, "rounds_by_distance value")
        initializations = tuple(self.initializations)
        if (not initializations or len(set(initializations)) != len(initializations)
                or set(initializations) - {"covariance", "uniform"}):
            raise ValueError("initializations must contain distinct selections from ['covariance', 'uniform']")
        object.__setattr__(self, "initializations", initializations)
        for key in ("target_decoders", "evaluation_decoders", "plot_decoders"):
            values = getattr(self, key)
            if key != "plot_decoders" or values is not None:
                object.__setattr__(self, key, validate_decoders(values, key))
        seeds = tuple(self.optimizer_seeds)
        if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("optimizer_seeds must contain distinct nonnegative integers")
        object.__setattr__(self, "optimizer_seeds", seeds)
        if self.covariance_failure not in ("skip", "uniform"):
            raise ValueError("covariance_failure must be 'skip' or 'uniform'")
        if not 0 < self.p_min < self.p_max < 0.5 or not 0 < self.uniform_probability < 0.5:
            raise ValueError("require 0 < p_min < p_max < 0.5 and 0 < uniform_probability < 0.5")
        if self.grouping not in ("coarse", "support", "manual"):
            raise ValueError("grouping must be 'coarse', 'support', or 'manual'")
        if (self.grouping == "manual") != (self.manual_groups is not None):
            raise ValueError("manual grouping requires manual_groups and vice versa")
        if self.ldpc_version != LDPC_VERSION or self.uf_mode != "peeling":
            raise ValueError(f"supported UF configuration is ldpc {LDPC_VERSION}, mode 'peeling'")
        if type(self.bits_per_step) is not int or not 0 <= self.bits_per_step <= 2147483647:
            raise ValueError("bits_per_step must be a nonnegative C int")

    def scientific_dict(self) -> dict:
        value = asdict(self)
        del value["resources"]
        del value["plot_decoders"]
        value["physical"] = self.physical.scientific_dict()
        return value


def _validate_keys(values: dict, config_type: type) -> None:
    if not isinstance(values, dict):
        raise ValueError(f"{config_type.__name__} settings must be a mapping")
    unknown = set(values) - {item.name for item in fields(config_type)}
    if unknown:
        raise ValueError(f"unknown {config_type.__name__} keys: {sorted(unknown)}")


def config_from_dict(data: dict, path: str | Path, *, theata_path: str | None = None) -> TrainingConfig:
    """Validate YAML-compatible settings, resolving paths relative to the YAML."""
    _validate_keys(data, TrainingConfig)
    data = dict(data)
    physical = data.pop("physical", {})
    _validate_keys(physical, ExperimentConfig)
    defaults = _physical_defaults().to_dict()
    physical = _base_config({**defaults, **physical}, Path(path).resolve(), workers=None, theata_path=theata_path)
    cem = data.pop("cem", {})
    resources = data.pop("resources", {})
    _validate_keys(cem, CEMConfig)
    _validate_keys(resources, Resources)
    return TrainingConfig(physical=physical, cem=CEMConfig(**cem), resources=Resources(**resources), **data)


def load_training_config(path: str | Path, *, theata_path: str | None = None) -> TrainingConfig:
    return config_from_dict(yaml.safe_load(Path(path).read_text()), path, theata_path=theata_path)
