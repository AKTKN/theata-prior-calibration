"""Validated settings and identities for the physical thermal benchmarks."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from importlib import metadata
import json
import math
import os
from pathlib import Path
import subprocess

import yaml

from .circuits import _load_theata
from .config import THEATA_REVISION, available_cpus

COLOR_REVISION = "0eb35935c1e5ff30ba3db9def30a9d35bca2f16d"
DISTANCES = (3, 5, 7, 9, 11, 13)
PAPER_NOISE = {"p_R": 1e-4, "p_meas": 3.2e-3, "p_reset": 8e-4, "gamma_phi_per_s": 3.0}


@dataclass(frozen=True)
class Condition:
    label: str
    policy: str
    cap: float | None = None
    n_max: int | None = None


def _nonnegative(value) -> bool:
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


@dataclass(frozen=True)
class ThermalConfig:
    code: str
    sweep: str
    distances: tuple[int, ...]
    basis: str = "Z"
    rounds: int = 3
    caps: tuple[float, ...] = ()
    budgets: tuple[int, ...] = ()
    shots: int = 100000
    shard_shots: int = 2048
    seed: int = 20261127
    workers: int = 1
    theata_path: str = "vendor/theata"
    thermal_boundary: str = "reset_each_round"
    cooling_threshold: float = 1.5
    post_measure_cooling_us: float = 1000.0
    time_parameters: dict = field(default_factory=dict)
    noise_parameters: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in ("distances", "caps", "budgets"):
            value = getattr(self, key)
            if not isinstance(value, (tuple, list)):
                raise ValueError(f"{key} must be a sequence")
            object.__setattr__(self, key, tuple(value))
        if self.code not in ("surface", "color") or self.sweep not in ("cap", "cooling"):
            raise ValueError("require code surface/color and sweep cap/cooling")
        if self.basis not in (("Z",) if self.code == "surface" else ("X", "Z")):
            raise ValueError("surface supports Z memory; color supports X and Z")
        if self.thermal_boundary != "reset_each_round":
            raise ValueError("the pinned profile supports only reset_each_round")
        if (not self.distances or
                any(type(d) is not int or d not in DISTANCES for d in self.distances) or
                tuple(sorted(set(self.distances))) != self.distances):
            raise ValueError(f"distances must be ordered unique entries from {DISTANCES}")
        for key in ("rounds", "workers", "shots", "shard_shots", "seed"):
            value = getattr(self, key)
            if type(value) is not int or value < (0 if key == "seed" else 1):
                raise ValueError(f"invalid {key}")
        if self.sweep == "cap":
            if (not self.caps or self.budgets or any(not _nonnegative(x) for x in self.caps) or
                    tuple(sorted(set(self.caps))) != self.caps):
                raise ValueError("cap sweep requires ordered unique finite nonnegative caps")
            object.__setattr__(self, "caps", tuple(map(float, self.caps)))
        elif (not self.budgets or self.caps or
              any(type(x) is not int or x < 1 for x in self.budgets) or
              tuple(sorted(set(self.budgets))) != self.budgets):
            raise ValueError("cooling sweep requires ordered unique positive integer budgets")
        if not _nonnegative(self.cooling_threshold):
            raise ValueError("cooling_threshold must be finite and nonnegative")
        if not _nonnegative(self.post_measure_cooling_us) or self.post_measure_cooling_us == 0:
            raise ValueError("post_measure_cooling_us must be finite and positive")
        for key in ("time_parameters", "noise_parameters"):
            if not isinstance(getattr(self, key), dict):
                raise ValueError(f"{key} must be a mapping")
        unknown = self.noise_parameters.keys() - PAPER_NOISE.keys()
        if unknown:
            raise ValueError(f"unknown noise parameters: {sorted(unknown)}")
        for key, value in self.noise_parameters.items():
            if not _nonnegative(value) or (key != "gamma_phi_per_s" and value >= 0.5):
                raise ValueError(f"invalid noise parameter {key}")

    @property
    def conditions(self) -> tuple[Condition, ...]:
        if self.sweep == "cap":
            return (*[Condition(f"cap_{x!r}", "no_cooling", cap=x) for x in self.caps],
                    Condition("uncapped", "no_cooling"))
        return (Condition("no_cooling", "no_cooling"), Condition("postmeasure_liz", "postmeasure_liz"),
                *[Condition(f"mpt_{n}", "mpt", n_max=n) for n in self.budgets])


def load_thermal_config(path: str | Path, **overrides) -> ThermalConfig:
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("thermal config must contain a mapping")
    unknown = data.keys() - {f.name for f in fields(ThermalConfig)}
    missing = {"code", "sweep", "distances"} - data.keys()
    if unknown or missing:
        raise ValueError(f"thermal configuration has unknown keys {sorted(unknown)} and missing keys {sorted(missing)}")
    data.update({key: value for key, value in overrides.items() if value is not None})
    selected = os.environ.get("THEATA_PATH", data.get("theata_path", "vendor/theata"))
    selected = overrides.get("theata_path") or selected
    base = path.parent.parent if path.parent.name == "configs" else path.parent
    data["theata_path"] = str((base / Path(selected).expanduser()).resolve())
    if data.get("workers") == "auto":
        data["workers"] = available_cpus()
    return ThermalConfig(**data)


def resolved_time(upstream, config: ThermalConfig):
    time_type = upstream["estimator"].TimeParameters
    defaults = asdict(time_type())
    defaults.update(json.loads((upstream["root"] / "params/mainz_heating.json").read_text()))
    defaults.update(cool=1000.0, phonon_cool_residual_transversal2=0.0)
    unknown = config.time_parameters.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"unknown TimeParameters: {sorted(unknown)}")
    defaults.update(config.time_parameters)
    for key, value in defaults.items():
        if "transversal2" in key and value is None:
            value = defaults[key.replace("transversal2", "transversal")]
            defaults[key] = value
        if not _nonnegative(value) or (key in {"shuttle_speed", "shuttle_base_distance", "cool"} and value == 0):
            raise ValueError(f"invalid TimeParameters value for {key}")
    return time_type(**defaults)


def compiler_settings(upstream, config: ThermalConfig, condition: Condition):
    return upstream["qccd_compiler"].CompilerConfig(
        planner="v1", resident_coolant_ids=None, predictive_cool_actuator="ancilla",
        cool_ancilla_after_measure_us=(None if condition.policy == "no_cooling" else config.post_measure_cooling_us),
        predictive_cool_threshold_transversal=(config.cooling_threshold if condition.policy == "mpt" else None),
        predictive_cool_max_iters=condition.n_max or 1,
    )


def hardware_environment(config: ThermalConfig):
    root = Path(config.theata_path).resolve()
    try:
        revision = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        changes = subprocess.check_output(
            ["git", "-C", root, "status", "--porcelain", "--untracked-files=no"], text=True
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"Expected a pinned THeatA Git checkout at {root}") from exc
    if revision != THEATA_REVISION or changes:
        raise ValueError(f"Thermal benchmarks require clean THeatA revision {THEATA_REVISION}")
    upstream = _load_theata(root)
    packages = ["stim", "pymatching", "numpy", "scipy", "matplotlib", "PyYAML", "lark",
                "networkx", "threadpoolctl", "theata-prior-calibration"]
    revisions = {"theata": revision}
    if config.code == "color":
        try:
            package = metadata.distribution("color-code-stim")
        except metadata.PackageNotFoundError as exc:
            raise ValueError("Install the thermal dependencies with bash scripts/bootstrap.sh --thermal") from exc
        source = json.loads(package.read_text("direct_url.json") or "{}")
        if source.get("vcs_info", {}).get("commit_id") != COLOR_REVISION:
            raise ValueError(f"Install pinned color-code-stim {COLOR_REVISION} with the thermal extra")
        revisions["color-code-stim"] = COLOR_REVISION
        packages += ["color-code-stim", "igraph", "statsmodels", "CairoSVG"]
    settings = {
        "revisions": revisions,
        "versions": {name: metadata.version(name) for name in packages},
        "time_parameters": asdict(resolved_time(upstream, config)),
        "noise_parameters": PAPER_NOISE | config.noise_parameters,
        "lookup_mode": "strict",
        "noise_scale": 1.0,
        "thermal_boundary": config.thermal_boundary,
        "phonon_clock": "upstream serial planning clock; zero occupations at each round start",
        "idle_clock": "upstream lockstep wall clock; contiguous across rounds",
    }
    return upstream, settings
