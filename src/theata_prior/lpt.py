"""Compile the paper's single-LIZ predictive-trigger surface workload."""
from __future__ import annotations

from dataclasses import asdict
from importlib import import_module
import math
from pathlib import Path
import subprocess

import stim

from .circuits import CompiledSurface, _load_theata
from .config import ExperimentConfig, THEATA_REVISION
from .thermal_config import PAPER_NOISE
from .thermal_noise import inject_thermal_noise, thermal_dem
from .thermal_trace import paper_modules


def compiler_options(config: ExperimentConfig, threshold: float) -> dict:
    # Upstream's archived run_lpt_theta_n32_1028.ps1, resident / temp arm.
    return {
        "planner": "v1",
        "resident_coolant_ids": [9000, 9001],
        "coolant_recool_us": config.cooling_duration_us,
        "cool_ancilla_after_measure_us": config.cooling_duration_us,
        "cool_ancilla_measure_interval": 1,
        "liz_pt_threshold_transversal": threshold,
        "liz_pt_target_transversal": None,
        "liz_pt_max_iters": config.liz_pt_max_iters,
        "liz_pt_park": "temp",
        "isolate_ancilla_recool": False,
        "cool_threshold_transversal": None,
        "merge_cool_threshold_transversal": None,
        "predictive_cool_threshold_transversal": None,
    }


def reconstructed_detector_sectors(circuit: stim.Circuit) -> tuple[str, ...]:
    """Label the paper recompiler's measurements from its reset/H/readout blocks."""
    x_ancillas: set[int] = set()
    measurements: list[str] = []
    sectors: list[str] = []
    for op in circuit.flattened():
        targets = [target.value for target in op.targets_copy()]
        if op.name == "R":
            x_ancillas.difference_update(targets)
        elif op.name == "H":
            # The recompiler encloses X-stabilizer CXs between two Hs.
            x_ancillas.update(targets)
        elif op.name == "M":
            measurements.extend("X" if qubit in x_ancillas else "Z" for qubit in targets)
        elif op.name == "DETECTOR":
            if not targets or any(not -len(measurements) <= offset < 0 for offset in targets):
                raise ValueError("Invalid detector record in the reconstructed circuit")
            bases = {measurements[offset] for offset in targets}
            if len(bases) != 1:
                raise ValueError("The reconstructed detector mixes X and Z measurements")
            sectors.append(bases.pop())
    if len(sectors) != circuit.num_detectors:
        raise ValueError("Missing basis labels for reconstructed detectors")
    return tuple(sectors)


def compile_lpt(config: ExperimentConfig, distance: int, rounds: int, threshold: float) -> CompiledSurface:
    root = Path(config.theata_path).resolve()
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        changes = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Expected a pinned THeatA Git checkout at {root}") from error
    if revision != THEATA_REVISION or changes:
        raise ValueError(f"LPT calibration requires clean THeatA revision {THEATA_REVISION}")
    upstream = _load_theata(root)
    distance_scan, _, recompiler, _, _ = paper_modules(upstream)
    scan = import_module("exp_cooling_distance_scan")
    if Path(scan.__file__).resolve().parent != root / "experiments":
        raise ValueError("The cooling scan must come from the pinned THeatA checkout")
    time_parameters = scan.load_time_parameters(
        str(root / "params/mainz_heating.json"),
        overrides={"cool": config.cooling_duration_us, **{
            f"phonon_cool_residual_{mode}": config.residual_phonon for mode in ("axial", "transversal", "transversal2")
        }},
    )
    options = compiler_options(config, threshold)
    # Upstream supplies greedy_alt, resident placement, the alternating circuit
    # order, and fresh phonons with carried-over ion order each round.
    compiled = scan.compile_rounds_d(distance, options, rounds, tp=time_parameters)
    forward, reverse, observable = distance_scan.DISTANCE_CIRCUITS[distance]
    ideal = recompiler.recompile_multi_round(compiled, observable_column=observable)
    anchors, gaps = upstream["ion_timeline"].gaps_from_compile_results(compiled, time_parameters, mode="lockstep")
    ast = upstream["ast_def"]
    kinds = {ast.GateOp: "CX", ast.MeasureOp: "M", ast.ResetOp: "R"}
    trace = {
        "code": "surface",
        "schedule_duration_us": gaps.t_end,
        "anchors": [[kinds[type(op)], list(op.targets), gaps.gaps_by_op.get(id(op), {})] for op in anchors],
        "trailing_idle_us": gaps.trailing_idle,
        "round_traces": [],
    }
    cooling = []
    for result in compiled:
        liz = result.initial_placement[options["resident_coolant_ids"][0]]
        if any(result.initial_placement[q] != liz or result.final_placement[q] != liz
               for q in options["resident_coolant_ids"]):
            raise RuntimeError("Resident coolant ions must remain assigned to the LIZ across rounds")
        round_cooling = []
        for index, op in enumerate(result.flat_ops):
            if isinstance(op, ast.CoolOp):
                # temp-park LPT cycles are the adjacent split / cool / merge
                # subsequence from the pinned planner.
                before = result.flat_ops[index - 1] if index else None
                after = result.flat_ops[index + 1] if index + 1 < len(result.flat_ops) else None
                lpt = (isinstance(before, ast.SplitOp) and isinstance(after, ast.MergeOp)
                       and before.src_zone_id == op.zone_id
                       and {after.zone_left, after.zone_right} == {op.zone_id, before.dst_zone_id})
                round_cooling.append({"zone": op.zone_id, "duration_us": op.duration_us or time_parameters.cool,
                                      "lpt_cycle": lpt})
        cooling.extend(round_cooling)
        trace["round_traces"].append({
            "initial_placement": result.initial_placement,
            "initial_phonons": [0.0, 0.0, 0.0],
            "final_zone_contents": result.final_zone_contents,
            "compiler_stats": result.stats,
            "cooling": round_cooling,
            "gate_phonons": [{**gate, "targets": list(gate["phonons"])}
                             for gate in result.phonon_data["at_gates"]],
        })
    noisy, noise = inject_thermal_noise(upstream, ideal, trace, cap=None, environment={"noise_parameters": PAPER_NOISE})
    ms_totals = [math.fsum(op.gate_args_copy()) for op in noisy if op.name == "PAULI_CHANNEL_2"]
    if len(ms_totals) != noise["ms_gates"]:
        raise RuntimeError("Reconstructed noisy circuit must contain one Pauli channel per MS gate")
    idle_gaps = [gap for _, _, ions in trace["anchors"] for gap in ions.values()]
    metadata = {
        **noise, "distance": distance, "rounds": rounds, "basis": "Z",
        "source_schedule": [forward, reverse], "compiler_options": asdict(compiled[0].config),
        "time_parameters": asdict(time_parameters), "noise_parameters": PAPER_NOISE, "noise_scale": 1.0,
        "theata_commit": revision, "liz_capacity": 4, "resident_coolant_ids": options["resident_coolant_ids"],
        "noise_convention": "THeatA reconstructed CX Approach B with per-ion lockstep idle dephasing",
        "thermal_boundary": "reset_each_round",
        "duration_us": gaps.t_end, "cooling_operations": len(cooling),
        "lpt_cooling_cycles": sum(item["lpt_cycle"] for item in cooling),
        "cooling_duration_us": math.fsum(item["duration_us"] for item in cooling),
        "mean_ms_phonon": noise["phonon_raw"]["mean"], "min_ms_phonon": noise["phonon_raw"]["min"],
        "max_ms_phonon": noise["phonon_raw"]["max"],
        "min_ms_error": min(ms_totals), "max_ms_error": max(ms_totals),
        "lut_out_of_range_gates": noise["lut_clipping_raw"]["above"],
        "lut_clipped_gates": noise["lut_clipping_used"]["above"],
        "idle_total_us": math.fsum((*idle_gaps, *gaps.trailing_idle.values())),
        "max_idle_probability": max((op.gate_args_copy()[0] for op in noisy if op.name == "Z_ERROR"), default=0.0),
        "trace": trace,
    }
    return CompiledSurface(
        ideal, ideal, noisy, noisy.detector_error_model(approximate_disjoint_errors=True),
        thermal_dem(noisy, "surface"), reconstructed_detector_sectors(ideal), metadata,
    )
