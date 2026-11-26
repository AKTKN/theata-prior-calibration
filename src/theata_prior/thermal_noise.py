"""Unnormalized hardware noise and condition-specific matching decoders."""
from __future__ import annotations

import math

import numpy as np
import stim

from .circuits import NativeStep, inject_native_noise
from .thermal_trace import _stim_anchors, color_workload, paper_modules


def _phonon_summary(values):
    return {"min": min(values), "mean": float(np.mean(values)), "median": float(np.median(values)), "max": max(values)}


def _ms_channels(upstream, trace: dict, cap: float | None):
    lookup = upstream["pauli_lookup"].PauliErrorLookup()
    records = [gate for round_trace in trace["round_traces"] for gate in round_trace["gate_phonons"]]
    raw = [max(modes[1] for modes in gate["phonons"].values()) for gate in records]
    used = raw if cap is None else [min(value, cap) for value in raw]
    selected = [lookup.select_n_bar(value, mode="strict", warn_on_clip=False) for value in used]
    # Cache exact LUT rows, selected by upward rounding and range clipping.
    channels = {row: tuple(lookup.lookup(row, mode="strict", warn_on_clip=False)) for row in set(selected)}
    for channel in channels.values():
        if len(channel) != 15 or any(not math.isfinite(p) or p < 0 for p in channel) or not 0 < math.fsum(channel) < 0.5:
            raise RuntimeError("Invalid THeatA MS Pauli channel")
    low, high = lookup.n_bar_range

    def clips(values):
        return {"below": sum(value < low for value in values), "above": sum(value > high for value in values)}

    probabilities = [channels[row] for row in selected]
    return records, probabilities, {
        "ms_gates": len(records),
        "phonon_raw": _phonon_summary(raw),
        "phonon_used": _phonon_summary(used),
        "phonon_lut_selected": _phonon_summary(selected),
        "lut_range": [low, high],
        "lut_clipping_raw": clips(raw),
        "lut_clipping_used": clips(used),
        "mean_ms_error": float(np.mean([math.fsum(channel) for channel in probabilities])),
        "per_gate_lut_selected": selected,
    }


def inject_thermal_noise(upstream, ideal, trace, cap, environment):
    records, channels, summary = _ms_channels(upstream, trace, cap)
    params = environment["noise_parameters"]
    if trace["code"] == "surface":
        _, _, _, injection, _ = paper_modules(upstream)
        anchor_signatures = [targets for kind, targets in _stim_anchors(ideal) if kind == "CX"]
        if anchor_signatures != [record["targets"] for record in records]:
            raise RuntimeError("Reconstructed CX targets disagree with the ordered physical MS trace")
        # The upstream injector reads rate[1] for each CX. The signature check
        # rules out the missing-entry fallback before calling it.
        rates = [(record["gate"], channel, selected, None, None) for record, channel, selected in
                 zip(records, channels, summary["per_gate_lut_selected"], strict=True)]
        noisy = injection.inject_noise_approach_b(ideal, rates, params, include_idle=False)
        ast = upstream["ast_def"]
        factories = {"CX": lambda ions: ast.GateOp("MS", ions, [0.25]), "R": ast.ResetOp, "M": ast.MeasureOp}
        anchor_ops, by_op = [], {}
        for kind, ions, gap in trace["anchors"]:
            operation = factories[kind](ions)
            anchor_ops.append(operation)
            by_op[id(operation)] = {int(ion): duration for ion, duration in gap.items()}
        gaps = upstream["ion_timeline"].IonGaps(
            gaps_by_op=by_op, trailing_idle={int(ion): gap for ion, gap in trace["trailing_idle_us"].items()},
            t_end=trace["schedule_duration_us"],
        )
        noisy = injection.inject_per_ion_idle(noisy, anchor_ops, gaps, params)
    else:
        steps, ms_index, cursor = [], 0, 0
        for instruction in ideal:
            name, targets = instruction.name, [t.value for t in instruction.targets_copy()]
            if name in ("DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS", "TICK"):
                steps.append(instruction)
                continue
            # Stim fuses adjacent same-name instructions. Expand in order
            # against the physical stream before inserting target-local noise.
            stride = 2 if name == "SQRT_XX" else 1
            for i in range(0, len(targets), stride):
                gate, ions, idle = trace["native_steps"][cursor]
                if name != gate or targets[i:i + stride] != ions:
                    raise RuntimeError("Native Stim operation does not match the compiled timing trace")
                probabilities = ()
                if name == "SQRT_XX":
                    if ions != records[ms_index]["targets"]:
                        raise RuntimeError("Physical MS operands disagree with the phonon trace")
                    probabilities = channels[ms_index]
                    ms_index += 1
                steps.append(NativeStep(gate, tuple(ions), tuple(idle), probabilities))
                cursor += 1
        if cursor != len(trace["native_steps"]) or ms_index != len(channels):
            raise RuntimeError("Unconsumed native timing or phonon records")
        noisy, native_summary = inject_native_noise(tuple(steps), params)
        summary.update(native_summary)
    return noisy, summary


def thermal_dem(circuit: stim.Circuit, code: str) -> stim.DetectorErrorModel:
    return circuit.detector_error_model(
        flatten_loops=True, approximate_disjoint_errors=True, decompose_errors=(code == "surface"),
    )


def color_decoder(noisy, dem, distance, rounds, basis):
    from color_code_stim.decoders import ConcatMatchingDecoder
    from color_code_stim.dem_utils.dem_manager import DemManager
    from color_code_stim.stim_utils import dem_to_parity_check

    class HardwareDemManager(DemManager):
        def _generate_dem(self):
            # The saved DEM was extracted directly from this noisy circuit,
            # without upstream's X/Z factorization of gate channels.
            return dem, *dem_to_parity_check(dem)

    source = color_workload(distance, rounds, basis)
    if (noisy.get_detector_coordinates() != source.circuit.get_detector_coordinates() or
            dem.get_detector_coordinates() != source.circuit.get_detector_coordinates()):
        raise RuntimeError("Detector metadata differs between color generator, compiled circuit and DEM")
    manager = HardwareDemManager(noisy, source.tanner_graph, "tri")
    if any(len(group) == 0 for group in manager.detector_ids_by_color.values()):
        raise ValueError("Concatenated matching requires detectors of all three colors")
    return ConcatMatchingDecoder(manager)


def color_pruning_audit(decoder, dem) -> dict:
    mechanisms = [instruction for instruction in dem if instruction.type == "error"]
    p = np.array([error.args_copy()[0] for error in mechanisms])
    per_color = {}
    missing = []
    for color, decomp in decoder.dem_manager.dems_decomposed.items():
        # Every retained original mechanism has a stage-2 column. Use the
        # actual mapping, including any omissions beyond the support-size rule.
        represented = set(decomp.error_map_matrices[1].nonzero()[1])
        discarded = sorted(set(range(len(mechanisms))) - represented)
        missing.append(set(discarded))
        per_color[color] = {
            "discarded_count": len(discarded),
            "discarded_probability_mass": float(p[discarded].sum()),
            "discarded_mechanism_indices": discarded,
        }
    any_color = sorted(set.union(*missing))
    all_colors = sorted(set.intersection(*missing))
    return {
        "mechanisms": len(mechanisms),
        "by_color": per_color,
        "discarded_any_color": len(any_color),
        "discarded_all_colors": len(all_colors),
        "discarded_all_colors_probability_mass": float(p[all_colors].sum()),
        "discarded_all_colors_mechanism_indices": all_colors,
        "discarded_any_color_probability_mass": float(p[any_color].sum()),
        "probability_mass_definition": "sum of decoder DEM mechanism probabilities, not a shot fraction",
        "approximation": "each color's two MWPM stages keep the edge-like mechanisms for that color",
        "status": "color_specific_pruning" if any_color else "zero_pruning",
    }
