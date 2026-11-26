"""Physical circuit, policy, and interrupted-run checks for the thermal workflows."""

from dataclasses import replace
import os
from pathlib import Path

import numpy as np
import pytest
import stim

from theata_prior import thermal_experiment
from theata_prior.circuits import native_stim_bridge
from theata_prior.io import read_json
from theata_prior.thermal_config import ThermalConfig, compiler_settings, hardware_environment
from theata_prior.thermal_noise import color_decoder, color_pruning_audit, inject_thermal_noise, thermal_dem
from theata_prior.thermal_trace import compile_trace, paper_modules


@pytest.fixture(scope="module")
def checkout():
    default = Path(__file__).resolve().parents[1] / "vendor/theata"
    path = Path(os.environ.get("THEATA_PATH", default))
    if not (path / "src" / "qccd_compiler.py").exists():
        pytest.skip("Requires the pinned THeatA checkout")
    return str(path)


def test_thermal_conditions_validate_caps_and_budgets():
    with pytest.raises(ValueError, match="finite"):
        ThermalConfig("surface", "cap", (3,), caps=(0, float("inf")))
    with pytest.raises(ValueError, match="positive"):
        ThermalConfig("surface", "cooling", (3,), budgets=(0, 1))
    with pytest.raises(ValueError, match="reset_each_round"):
        ThermalConfig("surface", "cap", (3,), caps=(0,), thermal_boundary="carry")
    config = ThermalConfig("surface", "cap", (3,), rounds=2, caps=(0,))
    assert config.conditions[-1].label == "uncapped"
    assert config.conditions[-1].cap is None


@pytest.mark.integration
def test_mrx_native_instrument_and_coordinate_shifts(checkout):
    config = ThermalConfig("surface", "cap", (3,), caps=(0,), theata_path=checkout)
    upstream, _ = hardware_environment(config)
    source = stim.Circuit("""
        RX 0 1
        REPEAT 3 {
            MRX 0 1
            DETECTOR(1, 2, 0, 0, 2) rec[-2]
            OBSERVABLE_INCLUDE(0) rec[-1]
            SHIFT_COORDS(0, 0, 1)
        }
        MRX 0 1
        DETECTOR(2, 3, 0, 0, 1) rec[-2]
    """)
    _, _, _, native = native_stim_bridge(source, upstream["ast_def"], keep_coordinate_shifts=True)
    assert native.num_measurements == source.num_measurements == 8
    assert native.get_detector_coordinates() == source.get_detector_coordinates()
    assert native.has_all_flows(source.flow_generators())
    assert source.has_all_flows(native.flow_generators())
    _, _, _, mrx = native_stim_bridge(stim.Circuit("MRX 0"), upstream["ast_def"])
    expected = stim.Circuit("H 0\nM 0\nR 0\nH 0")
    assert mrx.has_all_flows(expected.flow_generators())
    assert expected.has_all_flows(mrx.flow_generators())


@pytest.mark.integration
def test_surface_matches_reference_and_applies_only_virtual_cap(checkout):
    config = ThermalConfig("surface", "cap", (3,), caps=(0, 1.5, 20), theata_path=checkout)
    upstream, env = hardware_environment(config)
    _, ideal, trace = compile_trace(upstream, config, 3, config.conditions[0], env)
    _, _, recompiler, injection, _ = paper_modules(upstream)
    import exp_cooling_distance_scan as reference

    parameters = upstream["estimator"].TimeParameters(**env["time_parameters"])
    reference_config = compiler_settings(upstream, config, config.conditions[0])
    results = reference.compile_rounds_d(3, reference_config.to_dict(), 3, tp=parameters)
    reference_ideal = recompiler.recompile_multi_round(results, observable_column=[0, 1, 2])
    assert env["time_parameters"]["heating_rate_transversal_per_s"] == 4.8
    assert ideal == reference_ideal
    for result, record in zip(results, trace["round_traces"], strict=True):
        assert result.phonon_data["at_gates"] == [
            {key: gate[key] for key in ("gate", "phonons", "time_us")} for gate in record["gate_phonons"]
        ]
        assert result.final_zone_contents == record["final_zone_order"]
    raw = [gate for result in results for gate in result.phonon_data["at_gates"]]
    rates = injection.phonon_to_pauli_lookup_rates(raw, upstream["pauli_lookup"].PauliErrorLookup())
    anchors, gaps = upstream["ion_timeline"].gaps_from_compile_results(results, parameters)
    baseline = injection.inject_noise_approach_b(reference_ideal, rates, env["noise_parameters"], include_idle=False)
    baseline = injection.inject_per_ion_idle(baseline, anchors, gaps, env["noise_parameters"])
    uncapped, raw_metrics = inject_thermal_noise(upstream, ideal, trace, None, env)
    at_twenty, _ = inject_thermal_noise(upstream, ideal, trace, 20, env)
    cold, cold_metrics = inject_thermal_noise(upstream, ideal, trace, 0, env)
    assert uncapped == baseline == at_twenty
    assert 0 < cold_metrics["mean_ms_error"] < raw_metrics["mean_ms_error"]
    assert cold_metrics["phonon_lut_selected"]["min"] == cold_metrics["phonon_lut_selected"]["max"] == 0.01
    assert cold_metrics["phonon_raw"] == raw_metrics["phonon_raw"]
    assert [i for i in cold if i.name != "PAULI_CHANNEL_2"] == [i for i in uncapped if i.name != "PAULI_CHANNEL_2"]
    assert {"X_ERROR", "Z_ERROR", "DEPOLARIZE1", "PAULI_CHANNEL_2"} <= {i.name for i in cold}


@pytest.mark.integration
@pytest.mark.parametrize("distance", [3, 5])
def test_surface_mpt_executes_real_episodes(checkout, distance):
    config = ThermalConfig("surface", "cooling", (distance,), budgets=(1, 2), theata_path=checkout)
    upstream, env = hardware_environment(config)
    traces = [compile_trace(upstream, config, distance, condition, env)[2] for condition in config.conditions]
    uncooled, postmeasure, budget_one, budget_two = traces
    mpt = budget_two["cooling"]
    assert uncooled["cooling"]["cool_ops"] == 0
    assert postmeasure["cooling"]["cool_ops"] == postmeasure["cooling"]["syndrome_measurements"]
    assert mpt["completed_loop_iterations"] > budget_one["cooling"]["completed_loop_iterations"] > 0
    assert mpt["memory_visits"] >= mpt["completed_loop_iterations"]
    assert mpt["cooling_duration_us"] > budget_one["cooling"]["cooling_duration_us"]
    assert budget_two["schedule_duration_us"] > budget_one["schedule_duration_us"]
    assert budget_two["round_traces"][0]["gate_phonons"] != budget_one["round_traces"][0]["gate_phonons"]
    rounds = budget_two["round_traces"]
    for current, following in zip(rounds, rounds[1:]):
        assert current["final_zone_order"] == following["initial_zone_order"]
    for round_trace in rounds:
        for event in round_trace["direct_cooling"]:
            assert set(event["ions"]) <= set(budget_two["syndrome_ancilla_ions"])
            assert event["end_us"] - event["start_us"] == pytest.approx(1000)
        assert all(len(window["episodes"]) <= 2 for window in round_trace["measurements"])
        assert any(window.get("predicted_gate_transversal") for window in round_trace["measurements"])


@pytest.mark.integration
@pytest.mark.parametrize("distance,basis", [(3, "Z"), (3, "X"), (5, "Z"), (7, "Z")])
def test_color_preserves_instrument_and_reports_inactive_mpt(checkout, distance, basis):
    pytest.importorskip("color_code_stim")
    config = ThermalConfig("color", "cooling", (distance,), basis=basis, budgets=(1, 4), theata_path=checkout)
    upstream, env = hardware_environment(config)
    source, ideal, trace = compile_trace(upstream, config, distance, config.conditions[-1], env)
    assert trace["source_schedule"] == "tri_optimal"
    assert source.get_detector_coordinates() == ideal.get_detector_coordinates()
    assert source.num_measurements == ideal.num_measurements
    assert [i for i in source.flattened() if i.name in ("DETECTOR", "OBSERVABLE_INCLUDE")] == [
        i for i in ideal.flattened() if i.name in ("DETECTOR", "OBSERVABLE_INCLUDE")
    ]
    assert ideal.detector_error_model().num_errors == 0
    cooling = trace["cooling"]
    assert cooling["policy_status"] == "policy_inactive"
    assert cooling["compiled_data_readout_measurements"] == len(trace["data_ions"])
    assert cooling["cross_reset_windows"] == 2
    assert cooling["eligible_checkpoints"] == cooling["completed_loop_iterations"] == 0
    assert cooling["cool_ops"] == cooling["syndrome_measurements"]
    noisy, stats = inject_thermal_noise(upstream, ideal, trace, None, env)
    assert stats["noise_scale"] == 1
    _, control_ideal, control_trace = compile_trace(upstream, config, distance, config.conditions[1], env)
    control, _ = inject_thermal_noise(upstream, control_ideal, control_trace, None, env)
    assert noisy == control
    dem = thermal_dem(noisy, "color")
    decoder = color_decoder(noisy, dem, distance, 3, basis)
    audit = color_pruning_audit(decoder, dem)
    assert audit["status"] == "zero_pruning"
    assert audit["discarded_any_color_probability_mass"] == 0
    det, obs = noisy.compile_detector_sampler(seed=15).sample(17, separate_observables=True)
    preds = decoder.decode(det)
    assert np.asarray(preds).size == obs.size == 17


@pytest.mark.integration
def test_shards_match_serial_after_interruption(tmp_path, monkeypatch, checkout):
    config = ThermalConfig("surface", "cap", (3,), caps=(0, 1.5), shots=91, shard_shots=17, theata_path=checkout)
    expected = thermal_experiment.run_thermal(config, tmp_path / "serial")
    target = tmp_path / "interrupted"
    actual_decode = thermal_experiment._decode_shards
    completed = []

    def interrupt_after_one_shard(path, point_id, tasks):
        if completed:
            raise InterruptedError("injected interruption")
        actual_decode(path, point_id, tasks[:1])
        completed.append(path)

    with monkeypatch.context() as patch:
        patch.setattr(thermal_experiment, "_decode_shards", interrupt_after_one_shard)
        with pytest.raises(InterruptedError):
            thermal_experiment.run_thermal(config, target)
    [first] = target.glob("points/*/*/shards/*.json")
    initial_record, initial_mtime = read_json(first), first.stat().st_mtime_ns
    actual = thermal_experiment.run_thermal(replace(config, workers=2), target, resume=True)
    assert actual == expected
    assert read_json(first) == initial_record
    assert first.stat().st_mtime_ns == initial_mtime
    assert len(list(target.glob("points/*/*/shards/*.json"))) == 18
    assert len({row["trace_sha256"] for row in actual}) == 1
    with pytest.raises(ValueError, match="Incompatible"):
        thermal_experiment.run_thermal(replace(config, seed=9), target, resume=True)
