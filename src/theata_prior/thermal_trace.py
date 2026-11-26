"""Thermal compilation and auditable traces for the two physical workloads."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from importlib import import_module
import math
from pathlib import Path
import sys

import stim

from .circuits import _schedule_grouped, _validate_native_identities, native_stim_bridge
from .thermal_config import Condition, ThermalConfig, compiler_settings


def paper_modules(upstream):
    for subdir in ("experiments", "circuits"):
        path = str(upstream["root"] / subdir)
        if path not in sys.path:
            sys.path.insert(0, path)
    names = ("exp_distance_scan", "exp_multi_round", "stim_recompiler", "exp_noise_injection", "qccd_parser")
    modules = [import_module(name) for name in names]
    if any(upstream["root"] not in Path(module.__file__).resolve().parents for module in modules):
        raise RuntimeError("Paper modules must come from the pinned THeatA checkout")
    return modules


def color_workload(distance: int, rounds: int, basis: str):
    from color_code_stim import ColorCode, NoiseModel

    return ColorCode(
        d=distance, rounds=rounds, circuit_type="tri", cnot_schedule="tri_optimal",
        superdense_circuit=False, temp_bdry_type=basis, noise_model=NoiseModel(),
        exclude_non_essential_pauli_detectors=True, _generate_dem=False,
    )


def _architecture(upstream, ions: int, parameters, extra: int = 0):
    capacity = max(20, (ions + 1) // 2 + extra + 4)
    if parameters.shuttle_base_distance <= parameters.shuttle_ion_factor * capacity:
        raise ValueError("Memory capacity exceeds the positive-distance range of the THeatA shuttle model")
    return upstream["qccd_compiler"].QCCDArchitecture.simple_mlm(
        liz_capacity=4, memory_capacity=capacity, temp_capacity=capacity,
    )


def _compile_round(upstream, dag, initial, architecture, settings, parameters, syndrome_ions):
    """Instrument v1's two-pass MPT and insist on explicitly disposable LIZ occupants."""
    qccd, ast = upstream["qccd_compiler"], upstream["ast_def"]

    class AuditedPlanner(qccd.MovementPlanner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.disposable = set()
            self.measurements = []
            self.laser_events = []
            self.gate_cursor = 0
            self.window_start = {q: 0.0 for q in initial}
            self.checkpoint = None

        def _annotate_gate(self, op, state):
            # Upstream infers roles from all MeasureOps, including terminal data readout.
            self._measure_ancillas = syndrome_ions
            if isinstance(op, ast.MeasureOp):
                self.disposable.update(set(op.targets) & syndrome_ions)
            elif isinstance(op, (ast.ResetOp, ast.GateOp)):
                self.disposable.difference_update(op.targets)
            return super()._annotate_gate(op, state)

        def _maybe_cool_after_measure(self, op, state):
            if isinstance(op, ast.MeasureOp) and set(op.targets) <= syndrome_ions:
                return super()._maybe_cool_after_measure(op, state)
            return None

        def _emit_cool_op(self, zone, state, dur):
            occupants = list(state.zone_contents[zone])
            if not occupants or not set(occupants) <= self.disposable:
                raise RuntimeError(f"Direct cooling would address a live or non-syndrome ion: {occupants}")
            output = super()._emit_cool_op(zone, state, dur)
            self.laser_events.append({"zone": zone, "ions": occupants, "duration_us": dur})
            return output

        def _on_measure_event(self, op, state):
            if not isinstance(op, ast.MeasureOp):
                return []
            gates = state.phonon_at_gates
            entry = {
                "measured_ions": list(op.targets),
                "eligible_actuator": set(op.targets) <= self.disposable and
                self.architecture.is_liz(state.locate(op.targets[0])),
                "preceding_ms_gates": list(range(self.gate_cursor, len(gates))),
                "trigger_activated": False,
                "episodes": [],
            }
            self.gate_cursor = len(gates)
            self.measurements.append(entry)
            self.checkpoint = entry
            ops = super()._on_measure_event(op, state)
            if self._record_windows:
                self._recorded_windows[-1]["ms_forecast_offsets"] = [
                    {ion: modes[1] - self.window_start.get(ion, 0.0)
                     for ion, modes in gates[index]["phonons"].items()}
                    for index in entry["preceding_ms_gates"]
                ]
                self.window_start = dict(state.phonon_transversal)
            self.checkpoint = None
            return ops

        def _maybe_predictive_cool(self, idx, op, state):
            table = self.window_table
            if idx >= len(table) or list(op.targets) != table[idx]["ancillas"]:
                raise RuntimeError(f"MPT measurement window {idx} diverged from the no-trigger pass")
            if not self.checkpoint["eligible_actuator"] or idx + 1 == len(table):
                return []
            nxt = table[idx + 1]
            predicted = {i: state.phonon_transversal[i] + nxt["delta_tr"].get(i, 0.0)
                         for i in sorted(nxt["data_ions"])}
            self.checkpoint["predicted_next_window_transversal"] = predicted
            self.checkpoint["predicted_gate_transversal"] = [
                {i: state.phonon_transversal[i] + delta for i, delta in offsets.items()}
                for offsets in nxt["ms_forecast_offsets"]
            ]
            self.checkpoint["trigger_activated"] = any(value > self.config.predictive_cool_threshold_transversal
                                                        for value in predicted.values())
            return super()._maybe_predictive_cool(idx, op, state)

        def _merge_cool_episode(self, ancilla, liz, target_zones, state):
            ops = super()._merge_cool_episode(ancilla, liz, target_zones, state)
            if ops:
                self.checkpoint["episodes"].append({
                    "memory_zones": target_zones,
                    "operation_counts": dict(Counter(type(op).__name__ for op in ops)),
                    "serial_duration_us": math.fsum(self._op_timer.operation_time(op) for op in ops),
                })
            return ops

    class GroupedScheduler(qccd.Scheduler):
        def schedule(self, ops):
            return _schedule_grouped(ops, self, ast.TimeStep)

    compiler = qccd.QCCDCompiler(architecture, config=settings, time_params=parameters)
    compiler._scheduler = GroupedScheduler()
    compiler._planner = AuditedPlanner(architecture, settings, time_params=parameters)
    result = compiler.compile(dag, initial)
    for validator_name, values in (("validate_operations", result.flat_ops), ("validate_timesteps", result.timesteps)):
        validation = getattr(upstream["validator"], validator_name)(values, initial, architecture, verbose=False)
        if not validation.is_valid:
            raise RuntimeError(f"THeatA {validator_name} failed:\n{validation.summary()}")
    intervals, duration = upstream["ion_timeline"].op_intervals_from_timesteps(result.timesteps, parameters)
    gates = [interval.op for interval in intervals if isinstance(interval.op, ast.GateOp) and interval.op.type == "MS"]
    gate_ops = [op for op in result.flat_ops if isinstance(op, ast.GateOp) and op.type == "MS"]
    if gates != gate_ops:
        raise RuntimeError("Lockstep schedule changed the MS order")
    by_id = {id(interval.op): interval for interval in intervals}
    trace = []
    for op, record in zip(gate_ops, result.phonon_data["at_gates"], strict=True):
        if tuple(record["phonons"]) != tuple(op.targets) or record["gate"] != f"MS({','.join(map(str, op.targets))})":
            raise RuntimeError("MS phonon trace identity differs from the compiled gate")
        interval = by_id[id(op)]
        trace.append({**record, "targets": list(op.targets), "start_us": interval.t0, "end_us": interval.t1})
    planner = compiler._planner
    for index, entry in enumerate(planner.measurements):
        following = planner.measurements[index + 1:index + 2]
        indices = following[0]["preceding_ms_gates"] if following else []
        entry["next_window_ms_gates"] = indices
        entry["next_window_data_ions"] = sorted({q for i in indices for q in trace[i]["targets"]} - syndrome_ions)
        entry["eligible_checkpoint"] = bool(entry["eligible_actuator"] and entry["next_window_data_ions"])
    initial_order = {zone: [ion for ion, z in initial.items() if z == zone] for zone in sorted(set(initial.values()))}
    audit = {
        "wall_us": duration,
        "planning_us": result.phonon_data["planning_clock_us"],
        "initial_zone_order": initial_order,
        "initial_phonons": [0.0, 0.0, 0.0],
        "final_zone_order": result.final_zone_contents,
        "operation_counts": dict(Counter(type(op).__name__ for op in result.flat_ops)),
        "measurements": planner.measurements,
        "post_last_measurement_ms_gates": list(range(planner.gate_cursor, len(trace))),
        "direct_cooling": planner.laser_events,
        "gate_phonons": trace,
    }
    cool_intervals = [iv for iv in intervals if isinstance(iv.op, ast.CoolOp)]
    if len(cool_intervals) != len(planner.laser_events):
        raise RuntimeError("Unaccounted CoolOp in compiled schedule")
    for event, interval in zip(planner.laser_events, cool_intervals, strict=True):
        if not math.isclose(interval.t1 - interval.t0, event["duration_us"], rel_tol=1e-9, abs_tol=1e-8):
            raise RuntimeError("Recorded CoolOp duration differs from its scheduled duration")
        event.update(start_us=interval.t0, end_us=interval.t1)
    return result, audit, by_id


def _ordered_placement(result):
    return {ion: zone for zone, ions in sorted(result.final_zone_contents.items()) for ion in ions}


def _surface_trace(upstream, config, distance, settings, parameters):
    distance_scan, multi_round, recompiler, _, parser = paper_modules(upstream)
    forward, reverse, observable = distance_scan.DISTANCE_CIRCUITS[distance]
    module = import_module(forward)
    data = set(module.DATA_QUBITS)
    syndrome = set(module.ANCILLA_QUBITS)
    initial = multi_round.placement_greedy_alt(module.DATA_QUBITS, module.ANCILLA_QUBITS, module)
    architecture = _architecture(upstream, len(data), parameters, extra=len(syndrome))
    results, audits = [], []
    for index in range(config.rounds):
        circuit = import_module((forward, reverse)[index % 2])
        dag = upstream["dag_builder"].DAGBuilder(parser.parse(circuit.CIRCUIT), stabilizer_parallel=False).build()
        result, audit, _ = _compile_round(upstream, dag, initial, architecture, settings, parameters, syndrome)
        results.append(result)
        audits.append(audit)
        initial = _ordered_placement(result)
    ideal = recompiler.recompile_multi_round(results, observable_column=observable)
    timeline = upstream["ion_timeline"]
    anchors, gaps = timeline.gaps_from_compile_results(results, parameters, mode="lockstep")
    ast = upstream["ast_def"]
    kinds = {ast.GateOp: "CX", ast.MeasureOp: "M", ast.ResetOp: "R"}
    anchor_records = [[kinds[type(op)], list(op.targets), gaps.gaps_by_op.get(id(op), {})] for op in anchors]
    signature = _stim_anchors(ideal)
    readout = signature[len(anchors):]
    if signature[:len(anchors)] != [(kind, targets) for kind, targets, _ in anchor_records] or (
            len(readout) != len(data) or {tuple(targets) for _, targets in readout} != {(q,) for q in data} or
            any(kind != "M" for kind, _ in readout)):
        raise RuntimeError("Reconstructed Stim anchors disagree with the compiled operations")
    trace = {"anchors": anchor_records, "trailing_idle_us": gaps.trailing_idle,
             "memory_capacity": architecture.get_memory_zones()[0].capacity,
             "source_schedule": [forward, reverse], "noise_convention": "upstream reconstructed CX, Approach B"}
    return ideal, ideal, audits, data, syndrome, trace


def _stim_anchors(circuit):
    signature = []
    for inst in circuit.flattened():
        if inst.name in ("CX", "R", "M"):
            qubits = [t.value for t in inst.targets_copy()]
            stride = 2 if inst.name == "CX" else 1
            signature.extend((inst.name, qubits[i:i + stride]) for i in range(0, len(qubits), stride))
    return signature


def _color_trace(upstream, config, distance, settings, parameters):
    source = color_workload(distance, config.rounds, config.basis)
    ideal = source.circuit
    syndrome = set(source.qubit_groups["anc_X"]["qid"]) | set(source.qubit_groups["anc_Z"]["qid"])
    data = set(source.qubit_groups["data"]["qid"])
    ions = sorted(data | syndrome)
    if data & syndrome or set(ideal.get_final_qubit_coordinates()) != set(ions):
        raise RuntimeError("Color-code source qubit roles disagree with the Tanner graph")
    _validate_native_identities()
    ast = upstream["ast_def"]
    operations, specifications, stream, native = native_stim_bridge(ideal, ast, keep_coordinate_shifts=True)
    boundaries = []
    cursor = 0
    for item in stream:
        if isinstance(item, int):
            cursor = item + 1
        elif item.name == "SHIFT_COORDS":
            if item.gate_args_copy() != [0.0, 0.0, 1.0]:
                raise RuntimeError("Unexpected color-code coordinate shift")
            boundaries.append(cursor)
    if len(boundaries) != config.rounds:
        raise RuntimeError("Color-code SHIFT_COORDS markers do not identify the configured rounds")
    # The final destructive readout stays in the last thermal segment.
    boundaries[-1] = len(operations)
    capacity = (len(ions) + 1) // 2
    architecture = _architecture(upstream, len(ions), parameters)
    initial = {q: 1 if i < capacity else 5 for i, q in enumerate(ions)}
    last_end = dict.fromkeys(ions, 0.0)
    native_steps, audits = [], []
    start, offset = 0, 0.0
    for end in boundaries:
        declarations = [ast.QubitDecl(q, ast.Location(z, i)) for i, (q, z) in enumerate(initial.items())]
        dag = upstream["dag_builder"].DAGBuilder(
            ast.Program(declarations, operations[start:end]), stabilizer_parallel=False
        ).build()
        for index in range(1, end - start):
            dag.add_edge(index - 1, index)
        dag.compute_entry_exit()
        result, audit, intervals = _compile_round(upstream, dag, initial, architecture, settings, parameters, syndrome)
        quantum = [op for op in result.flat_ops if isinstance(op, (ast.GateOp, ast.MeasureOp, ast.ResetOp))]
        for original, op in zip(operations[start:end], quantum, strict=True):
            if type(original) is not type(op) or original.targets != op.targets or (
                    isinstance(op, ast.GateOp) and (original.type, original.params) != (op.type, op.params)):
                raise RuntimeError("THeatA changed the color-code native operation order or identity")
        for spec, op in zip(specifications[start:end], quantum, strict=True):
            interval = intervals[id(op)]
            gaps = [offset + interval.t0 - last_end[q] for q in op.targets]
            if any(gap < -1e-7 for gap in gaps):
                raise RuntimeError("Overlapping native quantum operations on an ion")
            native_steps.append([spec[0], list(spec[1]), [max(0.0, gap) for gap in gaps]])
            last_end.update({q: offset + interval.t1 for q in op.targets})
        audits.append(audit)
        initial = _ordered_placement(result)
        start, offset = end, offset + audit["wall_us"]
    if (ideal.num_measurements != native.num_measurements or ideal.num_observables != native.num_observables or
            ideal.get_detector_coordinates() != native.get_detector_coordinates() or
            ideal.get_final_qubit_coordinates() != native.get_final_qubit_coordinates()):
        raise RuntimeError("Native rewrite changed measurement counts, observables or coordinates")
    trace = {"native_steps": native_steps, "source_schedule": "tri_optimal",
             "memory_capacity": architecture.get_memory_zones()[0].capacity,
             "noise_convention": "native SQRT_XX Pauli channel and per-native-operation idle dephasing"}
    return ideal, native, audits, data, syndrome, trace


def compile_trace(upstream, config: ThermalConfig, distance: int, condition: Condition, environment: dict):
    settings = compiler_settings(upstream, config, condition)
    parameters = upstream["estimator"].TimeParameters(**environment["time_parameters"])
    adapter = _surface_trace if config.code == "surface" else _color_trace
    source, ideal, rounds, data, syndrome, trace = adapter(upstream, config, distance, settings, parameters)
    ideal.detector_error_model()
    if any(record["post_last_measurement_ms_gates"] for record in rounds):
        raise RuntimeError("MS gates follow the last measurement of a thermal segment")
    for current, following in zip(rounds, rounds[1:]):
        before = [entry for entry in current["measurements"] if set(entry["measured_ions"]) <= syndrome]
        next_window = following["measurements"][0]["preceding_ms_gates"]
        before[-1]["across_reset_lookahead"] = {
            "ms_gates": next_window,
            "data_ions": sorted({q for i in next_window for q in following["gate_phonons"][i]["targets"]} & data),
            "eligible": False,
            "reason": "round reset discards any phonon reduction before the next measurement window",
        }
    counts = Counter()
    for record in rounds:
        counts.update(record["operation_counts"])
    checkpoints = [entry for record in rounds for entry in record["measurements"]]
    episodes = [episode for entry in checkpoints for episode in entry["episodes"]]
    laser = [event for record in rounds for event in record["direct_cooling"]]
    eligibility = sum(entry["eligible_checkpoint"] for entry in checkpoints)
    activations = sum(entry["trigger_activated"] for entry in checkpoints)
    status = "disabled" if condition.policy != "mpt" else (
        "policy_inactive" if not eligibility else ("executed" if episodes else "eligible_no_episode"))
    trace.update(
        code=config.code, distance=distance, basis=config.basis, rounds=config.rounds,
        compiler_options=asdict(settings), architecture="M-M-M-L-M-M-M, single LIZ",
        liz_capacity=4,
        data_ions=sorted(data), syndrome_ancilla_ions=sorted(syndrome), coolant_ions=[],
        round_traces=rounds, operation_counts=dict(counts),
        schedule_duration_us=math.fsum(record["wall_us"] for record in rounds),
        planning_duration_us=math.fsum(record["planning_us"] for record in rounds),
        cooling={"policy_status": status, "measurements": len(checkpoints),
                 "syndrome_measurements": sum(set(c["measured_ions"]) <= syndrome for c in checkpoints),
                 "compiled_data_readout_measurements": sum(set(c["measured_ions"]) <= data for c in checkpoints),
                 "eligible_checkpoints": eligibility,
                 "cross_reset_windows": sum("across_reset_lookahead" in entry for entry in checkpoints),
                 "trigger_activations": activations, "completed_loop_iterations": len(episodes),
                 "activated_checkpoints_with_episodes": sum(bool(entry["episodes"]) for entry in checkpoints),
                 "transport_episodes": len(episodes), "memory_visits": sum(len(e["memory_zones"]) for e in episodes),
                 "cool_ops": len(laser), "cooling_duration_us": math.fsum(event["duration_us"] for event in laser)},
    )
    if config.code == "color" and eligibility:
        raise RuntimeError("tri_optimal unexpectedly has an in-round postmeasurement MS window")
    return source, ideal, trace
