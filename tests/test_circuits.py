"""Native equivalence and actual THeatA compilation checks."""

from collections import Counter
import math
import os
from pathlib import Path
import warnings

import numpy as np
import pytest
import stim

from theata_prior.circuits import (
    _load_theata, _native_gate_specs, _schedule_grouped, detector_sectors, prepare_surface,
)
from theata_prior.config import ExperimentConfig
from theata_prior.experiment import build_models, evaluate_models_stage, fit_models, sample_models
from theata_prior.io import read_json


@pytest.mark.parametrize("gate,targets", [("H", (0,)), ("CX", (0, 1))])
def test_native_blocks_are_exact_cliffords(gate, targets):
    actual = stim.Circuit()
    expected = stim.Circuit()
    for native_gate, native_targets in _native_gate_specs(gate, targets):
        actual.append(native_gate, native_targets)
    expected.append(gate, targets)
    assert actual.to_tableau() == expected.to_tableau()


@pytest.mark.parametrize("basis", ["x", "z"])
def test_standard_detector_sector_counts(basis):
    circuit = stim.Circuit.generated(f"surface_code:rotated_memory_{basis}", distance=3, rounds=3)
    sectors = detector_sectors(circuit)
    assert len(sectors) == circuit.num_detectors
    assert Counter(sectors) == {basis.upper(): 16, ("Z" if basis == "x" else "X"): 8}


@pytest.fixture(scope="module")
def theata_path():
    root = Path(os.environ.get("THEATA_PATH", Path(__file__).resolve().parents[1] / "vendor" / "theata"))
    if not (root / "src" / "qccd_compiler.py").is_file():
        pytest.skip("Set THEATA_PATH to a THeatA checkout for integration tests")
    return root


@pytest.fixture(scope="module", params=["X", "Z"])
def prepared(request, theata_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return prepare_surface(3, 3, request.param, theata_path)


@pytest.mark.integration
def test_compiled_native_circuit_preserves_standard_detectors(prepared):
    reference = prepared.ideal_circuit
    rewritten = prepared.compiled_ideal_circuit
    assert reference.num_detectors == rewritten.num_detectors
    assert reference.num_measurements == rewritten.num_measurements
    assert reference.get_detector_coordinates() == rewritten.get_detector_coordinates()
    assert reference.get_final_qubit_coordinates() == rewritten.get_final_qubit_coordinates()
    # Local tableau identities prove the rewrite; this catches annotation and
    # measurement-order errors for both memory bases at the global boundaries.
    detectors, observables = rewritten.compile_detector_sampler(seed=19).sample(
        512, separate_observables=True
    )
    assert not detectors.any()
    assert not observables.any()
    assert prepared.metadata["used_qubits"] == 17
    assert prepared.metadata["upstream_sequential_validation"]
    assert prepared.metadata["upstream_parallel_validation"]


@pytest.mark.integration
def test_physical_error_rate_is_mean_ms_probability(prepared):
    target = 0.003
    compiled = prepared.with_physical_error_rate(target)
    totals = [sum(instruction.gate_args_copy()) for instruction in compiled.noisy_circuit
              if instruction.name == "PAULI_CHANNEL_2"]
    assert len(totals) == 72
    assert np.mean(totals) == pytest.approx(target, rel=1e-13)
    assert max(totals) > min(totals)
    assert compiled.metadata["dem_approximate_disjoint_errors"] is True


@pytest.mark.integration
def test_both_sector_projections_are_graphlike(prepared):
    compiled = prepared.with_physical_error_rate(0.003)
    for sector in ("X", "Z"):
        supports = []
        for instruction in compiled.raw_dem.flattened():
            if instruction.type == "error":
                targets = [target.val for target in instruction.targets_copy()
                           if target.is_relative_detector_id() and compiled.detector_sectors[target.val] == sector]
                supports.append(len(targets))
        assert max(supports) == 2
    zero = prepared.with_physical_error_rate(0)
    assert zero.raw_dem.num_errors == 0
    detectors, observables = zero.noisy_circuit.compile_detector_sampler(seed=14).sample(
        100, separate_observables=True
    )
    assert not detectors.any()
    assert not observables.any()


@pytest.mark.integration
def test_rejects_scaled_probabilities_outside_domain(prepared):
    with pytest.raises(ValueError, match="Scaled MS error"):
        prepared.with_physical_error_rate(0.499)


@pytest.mark.integration
def test_grouped_scheduler_exactly_matches_upstream(theata_path):
    upstream = _load_theata(theata_path)
    ast = upstream["ast_def"]
    scheduler = upstream["qccd_compiler"].Scheduler()
    # Overlapping/disjoint zone sets, zone-free barriers and shuffled commands
    # exercise ordering and simultaneity, not the implementation bookkeeping.
    operations = [
        ast.GateOp("R", [0], [1.0, 0.0], zone_id=3),
        ast.SwapOp([1, 2], zone_id=1),
        ast.SplitOp(ion_id=1, dst_zone_id=2, src_zone_id=1),
        ast.MergeOp(zone_left=2, zone_right=3),
        ast.BarrierOp([0, 1]),
        ast.GateOp("MS", [0, 1], [.25], zone_id=3),
        ast.SwapOp([2, 3], zone_id=5),
        ast.ResetOp([0], zone_id=3),
        ast.MeasureOp([0], zone_id=3),
    ] * 13
    expected = scheduler.schedule(operations)
    actual = _schedule_grouped(operations, scheduler, ast.TimeStep)
    assert actual == expected


def test_bad_surface_parameters_rejected_before_import():
    with pytest.raises(ValueError, match="odd integer"):
        prepare_surface(4, 3, "Z", "/missing")
    with pytest.raises(ValueError, match="positive integer"):
        prepare_surface(3, 0, "Z", "/missing")


@pytest.mark.integration
def test_cooling_sweep_uses_unscaled_native_noise_and_resumes(theata_path, tmp_path, monkeypatch):
    import theata_prior.circuits as circuits

    upstream = _load_theata(theata_path)
    lookup = upstream["pauli_lookup"].PauliErrorLookup()
    compiler = upstream["qccd_compiler"].QCCDCompiler
    original_compile = compiler.compile
    compiled_conditions = []
    prepared_conditions = []

    def record_compile(self, *args, **kwargs):
        result = original_compile(self, *args, **kwargs)
        compiled_conditions.append(result)
        return result

    def record_prepare(*args, **kwargs):
        result = prepare_surface(*args, **kwargs)
        prepared_conditions.append(result)
        return result

    monkeypatch.setattr(compiler, "compile", record_compile)
    monkeypatch.setattr(circuits, "prepare_surface", record_prepare)
    noise = {"p_R": 0.00015, "p_reset": 0.001, "p_meas": 0.003, "gamma_phi_per_s": 2.5}
    config = ExperimentConfig(cool_thresholds=(0.25, 1.5, 4, None), theata_path=str(theata_path),
                              training_shots=(512, 2048), test_shots=2048, sample_chunk_shots=512,
                              hardware={"noise_parameters": noise})
    root = tmp_path / "sweep"
    with pytest.warns(RuntimeWarning, match="LUT clipped"):
        selected = build_models(config, root)
    assert tuple(result.config.cool_threshold_transversal for result in compiled_conditions) == config.cool_thresholds
    assert len({case.directory for case in selected}) == len(selected) == len(compiled_conditions)
    metadata = [read_json(case.directory / "hardware.json") for case in selected]
    assert all(h["time_parameters"] == metadata[0]["time_parameters"] for h in metadata)
    for case, hardware, result, prepared in zip(selected, metadata, compiled_conditions, prepared_conditions, strict=True):
        assert hardware["compiler_options"]["cool_threshold_transversal"] == case.cool_threshold
        assert hardware["noise_parameters"] == noise
        assert hardware["phonon_scale"] == hardware["noise_scale"] == 1.0
        assert "discrete_rate_scaling" not in hardware and "scaled_noise_parameters" not in hardware
        assert hardware["cooling_operations"] == sum(isinstance(op, upstream["ast_def"].CoolOp) for op in result.flat_ops)
        assert hardware["cooling_duration_us"] == pytest.approx(hardware["cooling_operations"] * hardware["time_parameters"]["cool"])
        time_parameters = upstream["estimator"].TimeParameters(**hardware["time_parameters"])
        _, duration = upstream["ion_timeline"].op_intervals_from_timesteps(result.timesteps, time_parameters)
        assert hardware["duration_us"] == duration
        noisy = stim.Circuit.from_file(case.directory / "actual_noisy.stim")
        channels = [instruction.gate_args_copy() for instruction in noisy if instruction.name == "PAULI_CHANNEL_2"]
        assert len(channels) == hardware["ms_gates"]
        for channel, gate, recorded in zip(channels, hardware["ms_gate_trace"], result.phonon_data["at_gates"], strict=True):
            n_bar = max(phonons[1] for phonons in recorded["phonons"].values())
            assert gate["n_bar"] == n_bar
            assert channel == gate["pauli_probabilities"] == lookup.lookup(n_bar, warn_on_clip=False)
        totals = [math.fsum(channel) for channel in channels]
        assert hardware["mean_ms_error"] == np.mean(totals)
        assert hardware["min_ms_error"] == min(totals)
        assert hardware["max_ms_error"] == max(totals)
        assert {op.gate_args_copy()[0] for op in noisy if op.name == "DEPOLARIZE1"} == {noise["p_R"]}
        assert {op.gate_args_copy()[0] for op in noisy if op.name == "X_ERROR"} == {noise["p_reset"], noise["p_meas"]}
        idle = [(target.value, op.gate_args_copy()[0]) for op in noisy if op.name == "Z_ERROR"
                for target in op.targets_copy()]
        expected_idle = [(target, -math.expm1(-2 * noise["gamma_phi_per_s"] * gap * 1e-6) / 2)
                         for step in prepared.steps if isinstance(step, circuits.NativeStep) and step.gate != "R"
                         for target, gap in zip(step.targets, step.idle_before_us, strict=True) if gap > 0]
        assert idle == expected_idle
        if case.cool_threshold == 1.5:
            legacy = prepared.with_physical_error_rate(prepared.metadata["baseline_mean_ms_error"])
            assert legacy.metadata["noise_scale"] == 1.0
            assert noisy == legacy.noisy_circuit
            assert stim.DetectorErrorModel.from_file(case.directory / "actual_full.dem") == legacy.raw_dem
            assert stim.DetectorErrorModel.from_file(case.directory / "actual_full_decomposed.dem") == legacy.decomposed_dem

    assert metadata[-1]["cooling_operations"] == 0
    assert metadata[-1]["lut_clipped_gates"] > 0
    for field in ("cooling_operations", "duration_us", "mean_ms_error"):
        assert len({h[field] for h in metadata}) > 1
    assert build_models(config, root) == selected
    assert len(compiled_conditions) == len(selected)

    sample_models(config, root)
    streams = [read_json(case.directory / "samples_test/samples.json")["config"]["stream"] for case in selected]
    assert len(set(streams)) == len(selected)
    fit_models(config, root, skip_saturated_marginals=True)
    rows = evaluate_models_stage(config, root, skip_saturated_marginals=True)
    assert {row["cool_threshold"] for row in rows if row["decoder"] == "actual"} == set(config.cool_thresholds)
    assert all(any(row["cool_threshold"] == threshold and row["prior_rmse"] is not None for row in rows)
               for threshold in (0.25, 1.5, 4))
    replica = tmp_path / "replica"
    with pytest.warns(RuntimeWarning, match="LUT clipped"):
        repeated = build_models(config, replica)
    sample_models(config, replica)
    for original, copy in zip(selected, repeated, strict=True):
        for path in ("actual_noisy.stim", "actual_full.dem", "samples_test/detectors.npy",
                     "samples_train_000/detectors.npy"):
            assert (original.directory / path).read_bytes() == (copy.directory / path).read_bytes()


@pytest.mark.integration
def test_strict_lut_reports_failed_threshold_and_keeps_other_cases(theata_path, tmp_path, monkeypatch):
    import theata_prior.circuits as circuits

    config = ExperimentConfig(cool_thresholds=(None, 1.5), theata_path=str(theata_path), training_shots=(128,),
                              test_shots=128, hardware={"strict_lut": True})
    rejected, valid = build_models(config, tmp_path)
    assert not (rejected.directory / "actual_noisy.stim").exists()
    failed, ok = read_json(tmp_path / "physical_metrics.json")
    assert failed["cool_threshold"] is None
    assert failed["status"] == "lut_out_of_range"
    assert "exceeds LUT maximum 20.0" in failed["reason"]
    assert failed["mean_ms_error"] is None and failed["lut_clipped_gates"] == 0
    assert failed["max_ms_phonon"] > 20
    assert ok["cool_threshold"] == 1.5 and ok["status"] == "ok"
    assert sample_models(config, tmp_path) == [valid]

    def unexpected_compile(*args, **kwargs):
        pytest.fail("A recorded model or deterministic LUT rejection should be cached")

    monkeypatch.setattr(circuits, "prepare_surface", unexpected_compile)
    assert build_models(config, tmp_path) == [rejected, valid]
