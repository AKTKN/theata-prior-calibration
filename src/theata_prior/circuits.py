"""Compile Stim's standard surface-memory circuit through THeatA.

THeatA's stock recompiler accepts its own serial stabilizer circuit format.  It
cannot round-trip the standard Stim circuit.  This bridge instead expands each
standard Clifford into the native R/MS basis, compiles those operations, and
keeps Stim's measurement order and detector/observable annotations.  All native
quantum operations are ordered conservatively; THeatA may overlap transport.

The Pauli lookup is applied at the physical MS location (not after a folded
CNOT).  Idle dephasing is inserted before *every* native quantum operation using
its scheduled per-ion gap, avoiding commuting dephasing through basis changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping
import warnings

import numpy as np
import stim


THEATA_COMMIT = "8bfb9b97de91bacac2f56725c27b4c71c6a91009"


@dataclass(frozen=True)
class NativeStep:
    """One native Clifford/reset/measurement, with its compiled noise inputs."""

    gate: str
    targets: tuple[int, ...]
    idle_before_us: tuple[float, ...]
    ms_probabilities: tuple[float, ...] = ()


@dataclass
class CompiledSurface:
    ideal_circuit: stim.Circuit
    compiled_ideal_circuit: stim.Circuit
    noisy_circuit: stim.Circuit
    raw_dem: stim.DetectorErrorModel
    decomposed_dem: stim.DetectorErrorModel
    detector_sectors: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass
class PreparedSurface:
    """A compiled trace reusable across the physical-error-rate sweep.

    ``with_physical_error_rate(p)`` defines p as the arithmetic mean, over MS
    gate locations, of the total nonidentity PAULI_CHANNEL_2 probability.  If
    the unscaled hardware trace has mean m, discrete probabilities and the
    dephasing rate are multiplied by alpha=p/m.  Times and phonons are fixed.
    """

    ideal_circuit: stim.Circuit
    compiled_ideal_circuit: stim.Circuit
    detector_sectors: tuple[str, ...]
    steps: tuple[NativeStep | stim.CircuitInstruction, ...]
    noise_parameters: dict[str, float]
    metadata: dict[str, Any]

    def with_physical_error_rate(self, physical_error_rate: float) -> CompiledSurface:
        p = float(physical_error_rate)
        if not math.isfinite(p) or not 0 <= p < 0.5:
            raise ValueError("physical_error_rate must be finite and in [0, 0.5)")
        baseline = float(self.metadata["baseline_mean_ms_error"])
        if baseline <= 0:
            raise ValueError("Cannot normalize a zero-error MS lookup table")
        alpha = p / baseline
        noisy, noise_summary = inject_native_noise(self.steps, self.noise_parameters, alpha=alpha)

        # General PAULI_CHANNEL_2 outcomes are mutually exclusive.  Stim's
        # independent DEM conversion is explicitly an approximation here.
        raw_dem = noisy.detector_error_model(approximate_disjoint_errors=True)
        decomposed_dem = noisy.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True
        )
        metadata = dict(self.metadata)
        metadata.update(
            **noise_summary,
            physical_error_rate=p,
            dem_approximate_disjoint_errors=True,
            actual_dem_semantics="independent Pauli-event approximation to the physical circuit",
            circuit_sampling_semantics="exact sampling of the compiled stochastic Pauli circuit",
        )
        return CompiledSurface(
            ideal_circuit=self.ideal_circuit.copy(),
            compiled_ideal_circuit=self.compiled_ideal_circuit.copy(),
            noisy_circuit=noisy,
            raw_dem=raw_dem,
            decomposed_dem=decomposed_dem,
            detector_sectors=self.detector_sectors,
            metadata=metadata,
        )


def inject_native_noise(
    steps: tuple[NativeStep | stim.CircuitInstruction, ...],
    pars: Mapping[str, float],
    *,
    alpha: float = 1.0,
) -> tuple[stim.Circuit, dict[str, Any]]:
    """Apply physical noise in the native Pauli frame. Thermal runs use alpha=1."""
    scaled = {key: alpha * pars[key] for key in ("p_R", "p_reset", "p_meas")}
    for key, value in scaled.items():
        if not 0 <= value < 0.5:
            raise ValueError(f"Scaled {key}={value} is outside [0, 0.5); reduce p")
    gamma = alpha * pars["gamma_phi_per_s"]
    noisy = stim.Circuit()
    ms_totals: list[float] = []
    idle_total_us = 0.0
    idle_max_p = 0.0
    for step in steps:
        if isinstance(step, stim.CircuitInstruction):
            noisy.append(step)
            continue
        # A Z error before Z-reset is absorbed by reset.  After measuring
        # in X, we explicitly use H+M, so no basis-specific shortcut is used.
        if step.gate != "R":
            for qubit, gap in zip(step.targets, step.idle_before_us, strict=True):
                idle_total_us += gap
                probability = -math.expm1(-2 * gamma * gap * 1e-6) / 2
                idle_max_p = max(idle_max_p, probability)
                if probability > 0:
                    noisy.append("Z_ERROR", [qubit], probability)
        if step.gate == "M" and scaled["p_meas"] > 0:
            noisy.append("X_ERROR", step.targets, scaled["p_meas"])
        noisy.append(step.gate, step.targets)
        if step.gate == "R":
            if scaled["p_reset"] > 0:
                noisy.append("X_ERROR", step.targets, scaled["p_reset"])
        elif step.gate == "SQRT_XX":
            probabilities = [alpha * value for value in step.ms_probabilities]
            total = math.fsum(probabilities)
            if total >= 0.5 or any(value < 0 for value in probabilities):
                raise ValueError(f"Scaled MS error {total} is outside [0, 0.5); reduce p")
            ms_totals.append(total)
            if total > 0:
                noisy.append("PAULI_CHANNEL_2", step.targets, probabilities)
        elif step.gate != "M" and scaled["p_R"] > 0:
            noisy.append("DEPOLARIZE1", step.targets, scaled["p_R"])

    return noisy, {
        "noise_scale": alpha,
        "mean_ms_error": float(np.mean(ms_totals)),
        "min_ms_error": min(ms_totals),
        "max_ms_error": max(ms_totals),
        "scaled_noise_parameters": {**scaled, "gamma_phi_per_s": gamma},
        "idle_total_us": idle_total_us,
        "max_idle_probability": idle_max_p,
    }


def _load_theata(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    required = ("qccd_compiler.py", "dag_builder.py", "ast_def.py", "pauli_lookup.py",
                "ion_timeline.py", "estimator.py", "validator.py")
    for name in required:
        if not (root / "src" / name).is_file():
            raise FileNotFoundError(f"Not a THeatA checkout: missing {root / 'src' / name}")
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    modules = {}
    for name in (p[:-3] for p in required):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve().parent != root / "src":
            raise RuntimeError(
                f"Module {name} was already imported from another checkout; use a fresh process"
            )
        modules[name] = module
    modules["root"] = root
    return modules


def _native_gate_specs(name: str, targets: tuple[int, ...]):
    """Exact Clifford identities; no approximate angle recognition is used."""
    if name == "H":
        q, = targets
        return [("SQRT_Y", (q,)), ("X", (q,))]
    if name == "CX":
        control, target = targets
        return [("SQRT_Y", (control,)), ("SQRT_XX", targets),
                ("SQRT_X_DAG", (control,)), ("SQRT_X_DAG", (target,)),
                ("SQRT_Y_DAG", (control,))]
    raise ValueError(f"Unsupported Clifford block {name}")


def _validate_native_identities() -> None:
    for name, targets in (("H", (0,)), ("CX", (0, 1))):
        actual = stim.Circuit()
        expected = stim.Circuit()
        for gate, qubits in _native_gate_specs(name, targets):
            actual.append(gate, qubits)
        expected.append(name, targets)
        if actual.to_tableau() != expected.to_tableau():
            raise RuntimeError(f"Native {name} decomposition failed Clifford tableau equality")


def _schedule_grouped(operations: list[Any], upstream_scheduler: Any, timestep_type: Any) -> list[Any]:
    """THeatA's exact ASAP rule, with linear-time grouping of resulting steps.

    Adapted from the Apache-2.0-licensed THeatA scheduler; the grouping is
    modified here. See THIRD_PARTY_NOTICES.md and licenses/ for attribution.

    The upstream scheduler scans all N operations once per resulting timestep
    while grouping.  This implementation only changes that bookkeeping step;
    it uses upstream's zone sets, the same earliest-step rule, and preserves
    input order within a timestep.  The source checkout is never modified.
    """
    zone_last: dict[int, int] = {}
    step_zones: dict[int, set[int]] = {}
    groups: dict[int, list[Any]] = {}
    for operation in operations:
        zones = upstream_scheduler._zones_used(operation)
        step = max((zone_last.get(zone, -1) + 1 for zone in zones), default=0)
        while zones & step_zones.get(step, set()):
            step += 1
        groups.setdefault(step, []).append(operation)
        step_zones.setdefault(step, set()).update(zones)
        for zone in zones:
            zone_last[zone] = max(zone_last.get(zone, -1), step)
    return [timestep_type(ops=groups[step]) for step in sorted(groups)]


def detector_sectors(circuit: stim.Circuit) -> tuple[str, ...]:
    """Determine check basis from standard-generator ancilla H/MR operations.

    Detector (x,y) coordinates coincide with their measured check ancilla,
    including initial/final time-boundary detectors.
    """
    ancillas = {t.value for instruction in circuit.flattened() if instruction.name == "MR"
                for t in instruction.targets_copy()}
    x_ancillas = {t.value for instruction in circuit.flattened() if instruction.name == "H"
                  for t in instruction.targets_copy()} & ancillas
    coordinates = circuit.get_final_qubit_coordinates()
    basis_by_coordinate = {tuple(coordinates[q][:2]): ("X" if q in x_ancillas else "Z")
                           for q in ancillas}
    result = []
    for index, coordinate in circuit.get_detector_coordinates().items():
        if index != len(result) or tuple(coordinate[:2]) not in basis_by_coordinate:
            raise ValueError("Could not identify standard surface-code detector basis")
        result.append(basis_by_coordinate[tuple(coordinate[:2])])
    if len(result) != circuit.num_detectors:
        raise ValueError("Missing detector sector assignments")
    return tuple(result)


def _unroll_stim(circuit: stim.Circuit):
    """Expand repeats while retaining coordinate shifts as round markers."""
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            body = instruction.body_copy()
            for _ in range(instruction.repeat_count):
                yield from _unroll_stim(body)
        else:
            yield instruction


def native_stim_bridge(ideal: stim.Circuit, ast, *, keep_coordinate_shifts: bool = False):
    """Translate a noiseless Stim Clifford stream without changing its records."""
    instructions = _unroll_stim(ideal) if keep_coordinate_shifts else ideal.flattened()
    operations: list[Any] = []
    specifications: list[tuple[str, tuple[int, ...]]] = []
    stream: list[int | stim.CircuitInstruction] = []
    compiled_ideal = stim.Circuit()

    def append_native(gate: str, targets: tuple[int, ...]) -> None:
        if gate == "R":
            operation = ast.ResetOp(list(targets))
        elif gate == "M":
            operation = ast.MeasureOp(list(targets))
        elif gate == "SQRT_XX":
            operation = ast.GateOp("MS", list(targets), [0.25])
        else:
            angles = {
                "X": (math.pi, 0.0),
                "SQRT_Y": (math.pi / 2, math.pi / 2),
                "SQRT_Y_DAG": (math.pi / 2, 3 * math.pi / 2),
                "SQRT_X_DAG": (math.pi / 2, math.pi),
            }
            operation = ast.GateOp("R", list(targets), list(angles[gate]))
        stream.append(len(operations))
        operations.append(operation)
        specifications.append((gate, targets))
        compiled_ideal.append(gate, targets)

    for instruction in instructions:
        name = instruction.name
        targets = tuple(t.value for t in instruction.targets_copy())
        if name == "CX":
            if any(not target.is_qubit_target for target in instruction.targets_copy()):
                raise ValueError(f"Unsupported non-qubit target in {instruction}")
            for a, b in zip(targets[::2], targets[1::2], strict=True):
                for gate, qubits in _native_gate_specs("CX", (a, b)):
                    append_native(gate, qubits)
        elif name in {"H", "R", "RX", "M", "MX", "MR", "MRX"}:
            if any(not target.is_qubit_target for target in instruction.targets_copy()):
                raise ValueError(f"Unsupported non-qubit target in {instruction}")
            if any(instruction.gate_args_copy()):
                raise ValueError(f"Source circuit must be noiseless: {instruction}")
            for qubit in targets:
                if name in {"R", "RX"}:
                    append_native("R", (qubit,))
                if name in {"H", "RX", "MX", "MRX"}:
                    for gate, qubits in _native_gate_specs("H", (qubit,)):
                        append_native(gate, qubits)
                if name in {"M", "MX", "MR", "MRX"}:
                    append_native("M", (qubit,))
                if name in {"MR", "MRX"}:
                    append_native("R", (qubit,))
                if name == "MRX":
                    for gate, qubits in _native_gate_specs("H", (qubit,)):
                        append_native(gate, qubits)
        elif name in {"TICK", "QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"}:
            stream.append(instruction)
            compiled_ideal.append(instruction)
        else:
            raise ValueError(f"Unsupported Stim instruction: {name}")

    return operations, specifications, stream, compiled_ideal


def prepare_surface(
    distance: int,
    rounds: int,
    basis: str,
    theata_path: str | Path,
    *,
    liz_capacity: int = 4,
    memory_capacity: int | None = None,
    compiler_options: Mapping[str, Any] | None = None,
    noise_parameters: Mapping[str, float] | None = None,
    time_parameters: Mapping[str, Any] | None = None,
    lookup_mode: str = "strict",
    phonon_scale: float = 1.0,
    strict_lut: bool = False,
) -> PreparedSurface:
    """Generate a standard Stim surface circuit and compile its full operation list.

    Data preparation, all original ancillas, all rounds, and final destructive
    measurements are compiled.  No ancilla reuse or alternate stabilizer
    schedule is substituted.  The total-order DAG is intentional and recorded.
    """
    if isinstance(distance, bool) or int(distance) != distance or distance < 3 or distance % 2 != 1:
        raise ValueError("distance must be an odd integer at least 3")
    if isinstance(rounds, bool) or int(rounds) != rounds or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    basis = basis.upper()
    if basis not in {"X", "Z"}:
        raise ValueError("basis must be X or Z")
    if not math.isfinite(phonon_scale) or phonon_scale <= 0:
        raise ValueError("phonon_scale must be finite and positive")
    if liz_capacity < 2:
        raise ValueError("liz_capacity must be at least 2")
    _validate_native_identities()
    upstream = _load_theata(theata_path)
    ast = upstream["ast_def"]
    dag_module = upstream["dag_builder"]
    qccd = upstream["qccd_compiler"]
    parameters = upstream["estimator"].TimeParameters(**dict(time_parameters or {}))
    configuration = qccd.CompilerConfig(**dict(compiler_options or {}))
    if configuration.measure_batch_size != 1 or configuration.interleave:
        raise ValueError("This conservative standard-Stim bridge requires measure_batch_size=1 and interleave=False")
    configuration.validate()

    ideal = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis.lower()}", distance=int(distance), rounds=int(rounds)
    )
    ions = sorted(ideal.get_final_qubit_coordinates())
    capacity = max(20, (len(ions) + 1) // 2 + 4) if memory_capacity is None else memory_capacity
    if capacity < (len(ions) + 1) // 2:
        raise ValueError("memory_capacity cannot hold the split initial placement")
    # The upstream shuttle-distance model is 1000 - 5*n micrometers.  Rather
    # than allowing zero/negative duration for unusually large codes, fail.
    if parameters.shuttle_base_distance <= parameters.shuttle_ion_factor * capacity:
        raise ValueError("Memory capacity exceeds the positive-distance range of the THeatA shuttle model")
    architecture = qccd.QCCDArchitecture.simple_mlm(
        liz_capacity=liz_capacity, memory_capacity=capacity, temp_capacity=capacity
    )
    initial = {q: 1 if index < (len(ions) + 1) // 2 else 5 for index, q in enumerate(ions)}
    declarations = [ast.QubitDecl(q, ast.Location(initial[q], i)) for i, q in enumerate(ions)]
    operations, specifications, stream, compiled_ideal = native_stim_bridge(ideal, ast)

    program = ast.Program(declarations, operations)
    dag = dag_module.DAGBuilder(program, stabilizer_parallel=False).build()
    # Preserve the user's original schedule and all measurement record indices.
    for index in range(1, len(operations)):
        dag.add_edge(index - 1, index)
    dag.compute_entry_exit()
    compiler = qccd.QCCDCompiler(architecture, config=configuration, time_params=parameters)

    class GroupedScheduler(qccd.Scheduler):
        def schedule(self, ops):
            return _schedule_grouped(ops, self, ast.TimeStep)

    # This replaces only the scheduler instance; all compiler placement,
    # transport, phonon tracking, and timing APIs remain upstream's code.
    compiler._scheduler = GroupedScheduler()
    result = compiler.compile(dag, initial)
    for validator_name, values in (("validate_operations", result.flat_ops),
                                   ("validate_timesteps", result.timesteps)):
        validation = getattr(upstream["validator"], validator_name)(
            values, initial, architecture, verbose=False
        )
        if not validation.is_valid:
            raise RuntimeError(f"THeatA {validator_name} failed:\n{validation.summary()}")
    compiled_ops = [op for op in result.flat_ops if isinstance(op, (ast.GateOp, ast.ResetOp, ast.MeasureOp))]
    if len(compiled_ops) != len(operations):
        raise RuntimeError("THeatA inserted or removed a native quantum operation")
    for original, compiled in zip(operations, compiled_ops, strict=True):
        if type(original) is not type(compiled) or original.targets != compiled.targets:
            raise RuntimeError("THeatA changed the native operation order")
        if isinstance(original, ast.GateOp) and (original.type != compiled.type or original.params != compiled.params):
            raise RuntimeError("THeatA changed a native gate")

    intervals, duration_us = upstream["ion_timeline"].op_intervals_from_timesteps(result.timesteps, parameters)
    interval_by_id = {id(interval.op): interval for interval in intervals}
    if any(interval.t1 < interval.t0 for interval in intervals):
        raise ValueError("THeatA produced a negative operation duration")
    lookup = upstream["pauli_lookup"].PauliErrorLookup()
    gate_phonons = list(result.phonon_data["at_gates"])
    if len(gate_phonons) != sum(gate == "SQRT_XX" for gate, _ in specifications):
        raise RuntimeError("THeatA phonon trace does not match the MS operation count")
    last_end = {q: 0.0 for q in ions}
    steps: list[NativeStep] = []
    ms_index = 0
    raw_totals: list[float] = []
    trace: list[dict[str, Any]] = []
    clip_count = 0
    previous_native_start = -math.inf
    for (gate, targets), operation in zip(specifications, compiled_ops, strict=True):
        interval = interval_by_id[id(operation)]
        if interval.t0 < previous_native_start - 1e-8:
            raise RuntimeError("THeatA scheduling reordered native quantum operations")
        previous_native_start = interval.t0
        gaps = []
        for q in targets:
            gap = interval.t0 - last_end[q]
            if gap < -1e-8:
                raise RuntimeError(f"Overlapping scheduled quantum operations on ion {q}")
            gaps.append(max(0.0, gap))
            last_end[q] = interval.t1
        probabilities: tuple[float, ...] = ()
        if gate == "SQRT_XX":
            gate_record = gate_phonons[ms_index]
            phonons = gate_record["phonons"]
            if set(phonons) != set(targets):
                raise RuntimeError("MS target order does not match the phonon trace")
            n_bar = phonon_scale * max(float(value[1]) for value in phonons.values())
            if n_bar > lookup.n_bar_range[1]:
                clip_count += 1
                if strict_lut:
                    raise ValueError(f"MS phonon n_bar={n_bar} exceeds LUT maximum {lookup.n_bar_range[1]}")
            selected = lookup.select_n_bar(n_bar, mode=lookup_mode, warn_on_clip=False)
            probabilities = tuple(lookup.lookup(n_bar, mode=lookup_mode, warn_on_clip=False))
            total = math.fsum(probabilities)
            if total <= 0 or total >= 1 or any(not math.isfinite(v) or v < 0 for v in probabilities):
                raise ValueError("Invalid THeatA MS Pauli lookup probabilities")
            raw_totals.append(total)
            trace.append({"index": ms_index, "targets": list(targets), "n_bar": n_bar,
                          "n_bar_lookup": selected, "baseline_total_error": total,
                          "start_us": interval.t0, "end_us": interval.t1})
            ms_index += 1
        steps.append(NativeStep(gate, targets, tuple(gaps), probabilities))
    if clip_count:
        warnings.warn(
            f"THeatA LUT clipped {clip_count}/{len(raw_totals)} MS phonon values to {lookup.n_bar_range[1]}; "
            "see metadata.ms_gate_trace or use strict_lut=True", RuntimeWarning, stacklevel=2
        )
    pars = {"p_R": 1e-4, "p_reset": 8e-4, "p_meas": 3.2e-3, "gamma_phi_per_s": 3.0}
    unknown = set(noise_parameters or {}) - set(pars)
    if unknown:
        raise ValueError(f"Unknown noise parameters: {sorted(unknown)}")
    pars.update({key: float(value) for key, value in (noise_parameters or {}).items()})
    if any(not math.isfinite(value) or value < 0 for value in pars.values()):
        raise ValueError("Noise parameters must be finite and nonnegative")
    if any(pars[key] >= 0.5 for key in ("p_R", "p_reset", "p_meas")):
        raise ValueError("Baseline discrete noise probabilities must be below 0.5")

    root = upstream["root"]
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    hashes = {name: hashlib.sha256((root / "src" / name).read_bytes()).hexdigest()
              for name in ("qccd_compiler.py", "pauli_twirled_error_probabilities.csv")}
    metadata = {
        "distance": int(distance), "rounds": int(rounds), "basis": basis,
        "stim_generator": f"surface_code:rotated_memory_{basis.lower()}",
        "used_qubits": len(ions), "stim_num_qubits": ideal.num_qubits,
        "num_detectors": ideal.num_detectors, "num_observables": ideal.num_observables,
        "native_quantum_operations": len(operations), "ms_gates": len(raw_totals),
        "scheduler_implementation": "upstream ASAP rule with one-pass timestep grouping",
        "duration_us": duration_us, "compiler_stats": result.stats,
        "planning_clock_us": result.phonon_data.get("planning_clock_us"),
        "phonon_clock": "THeatA serial planner clock (upstream convention)",
        "idle_clock": "THeatA parallel scheduler wall clock",
        "upstream_sequential_validation": True, "upstream_parallel_validation": True,
        "initial_placement": initial, "liz_capacity": liz_capacity,
        "memory_capacity": capacity, "native_schedule": "total-order DAG; original Stim measurement order",
        "baseline_mean_ms_error": float(np.mean(raw_totals)),
        "baseline_min_ms_error": min(raw_totals), "baseline_max_ms_error": max(raw_totals),
        "physical_error_rate_definition": "mean total nonidentity probability at physical MS gates",
        "discrete_rate_scaling": "alpha = p / baseline_mean_ms_error; probabilities multiplied by alpha",
        "idle_rate_scaling": "gamma_phi multiplied by alpha; p_Z = (1-exp(-2*alpha*gamma*t))/2",
        "noise_parameters": pars, "time_parameters": asdict(parameters),
        "compiler_options": asdict(configuration), "lookup_mode": lookup_mode,
        "phonon_scale": phonon_scale, "lut_range": list(lookup.n_bar_range),
        "lut_clipped_gates": clip_count, "strict_lut": strict_lut,
        "ms_gate_trace": trace, "theata_commit": commit,
        "tested_theata_commit": THEATA_COMMIT, "theata_source_sha256": hashes,
        "stim_version": stim.__version__,
    }
    # This also checks detector determinism of the native rewrite exactly.
    compiled_ideal.detector_error_model()
    return PreparedSurface(
        ideal_circuit=ideal,
        compiled_ideal_circuit=compiled_ideal,
        detector_sectors=detector_sectors(ideal),
        steps=tuple(steps[item] if isinstance(item, int) else item for item in stream),
        noise_parameters=pars,
        metadata=metadata,
    )


def compile_surface(
    distance: int,
    rounds: int,
    basis: str,
    theata_path: str | Path,
    physical_error_rate: float,
    **kwargs: Any,
) -> CompiledSurface:
    """Convenience wrapper; use prepare_surface to reuse compilation in a sweep."""
    return prepare_surface(distance, rounds, basis, theata_path, **kwargs).with_physical_error_rate(physical_error_rate)
