"""Native equivalence and actual THeatA compilation checks."""

from collections import Counter
import os
from pathlib import Path
import warnings

import numpy as np
import pytest
import stim

from theata_prior.circuits import (
    _load_theata, _native_gate_specs, _schedule_grouped, detector_sectors, prepare_surface,
)


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
    root = Path(os.environ.get("THEATA_PATH", Path(__file__).resolve().parents[1] / ".deps" / "theata"))
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
