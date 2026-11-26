"""Algebraic and structural regression tests for covariance prior estimation."""

from itertools import product
import json

import numpy as np
import pytest
import stim

from theata_prior.dem import DemStructureError, ErrorSupport, GraphStructure, model_moments, project_dem
from theata_prior.estimation import EstimationError, Moments, compute_moments, estimate_priors


def exhaustive_moments(structure, probabilities):
    """Independent reference: enumerate every Bernoulli mechanism outcome."""
    mean = np.zeros(structure.num_detectors)
    second = np.zeros((structure.num_detectors, structure.num_detectors))
    for events in product((0, 1), repeat=structure.num_errors):
        weight = 1.0
        outcome = np.zeros(structure.num_detectors, dtype=np.uint8)
        for fired, edge, p in zip(events, structure.edges, probabilities, strict=True):
            weight *= p if fired else 1 - p
            if fired:
                outcome[list(edge.detectors)] ^= 1
        mean += weight * outcome
        second += weight * np.outer(outcome, outcome)
    return Moments(0, mean, second)


def test_exact_population_inversion_including_boundaries():
    structure = GraphStructure(
        4, 1,
        (ErrorSupport((0,), (0,)), ErrorSupport((0, 1)), ErrorSupport((0, 2)),
         ErrorSupport((1, 2)), ErrorSupport((2, 3)), ErrorSupport((3,))),
    )
    priors = np.array([0.03, 0.01, 0.017, 0.007, 0.021, 0.023])
    moments = exhaustive_moments(structure, priors)
    result = estimate_priors(structure, moments=moments, strict=True)
    np.testing.assert_allclose(result.probabilities, priors, rtol=1e-12, atol=1e-14)
    assert result.diagnostics["all_known_edges"]["regularized"] == 0
    assert result.diagnostics["max_marginal_residual"] < 1e-14
    np.testing.assert_array_equal(np.diag(result.p_ij), 0)
    np.testing.assert_allclose(model_moments(structure, priors)[0], moments.means, atol=1e-14)
    np.testing.assert_allclose(model_moments(structure, priors)[1], moments.second_moments, atol=1e-14)


def test_worked_covariance_example_recovers_one_percent_exactly():
    structure = GraphStructure(2, 0, (ErrorSupport((0,)), ErrorSupport((0, 1)), ErrorSupport((1,))))
    moments = Moments(10000, np.array([0.108, 0.108]), np.array([[0.108, 0.018], [0.018, 0.108]]))
    result = estimate_priors(structure, moments=moments)
    np.testing.assert_allclose(result.probabilities, [0.1, 0.01, 0.1], rtol=1e-14)
    assert result.p_ij[0, 1] == pytest.approx(0.01)


def test_counts_do_not_overflow_uint8_and_are_worker_independent():
    samples = np.tile(np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.uint8), (512, 1))
    expected = samples.astype(np.float64).T @ samples.astype(np.float64) / len(samples)
    one = compute_moments(samples, workers=1, chunk_shots=331, block_detectors=1)
    many = compute_moments(samples, workers=3, chunk_shots=331, block_detectors=1)
    np.testing.assert_array_equal(one.second_moments, many.second_moments)
    np.testing.assert_allclose(one.second_moments, expected)
    np.testing.assert_allclose(one.means, [2 / 3] * 3)
    prefix = compute_moments(samples, shots=1000, workers=2, block_detectors=1)
    np.testing.assert_allclose(prefix.means, samples[:1000].mean(axis=0))


def test_projection_marginalizes_correlated_components_without_splitting():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1 ^ D2 D3
        error(0.2) D0 D1
        error(0.04) D2
        detector(5, 6, 7) D0
        detector D3
    """)
    projected = project_dem(dem, (0, 1))
    assert projected.structure.edges == (ErrorSupport((0, 1)),)
    assert projected.probabilities.tolist() == pytest.approx([0.26])
    assert projected.diagnostics["separator_targets_treated_as_correlated"] == 1
    assert projected.diagnostics["discarded_invisible_nonlogical_events"] == 1
    assert projected.structure.detector_coordinates[0] == (5, 6, 7)
    assert projected.dem.num_detectors == 2


def test_same_instruction_repeated_targets_cancel_under_xor():
    dem = stim.DetectorErrorModel("error(0.1) D0 D1 L0 ^ D1 D2 L0\ndetector D2")
    projected = project_dem(dem)
    assert projected.structure.edges == (ErrorSupport((0, 2)),)
    assert projected.dem.num_observables == 1


@pytest.mark.parametrize("invisible_prior", [0.5, 0.9, 1.0])
def test_invisible_nonlogical_event_integrates_out_at_any_valid_prior(invisible_prior):
    actual = stim.DetectorErrorModel(
        f"error(0.031) D0 L0\nerror({invisible_prior}) D1\ndetector D2"
    )
    reference = stim.DetectorErrorModel("error(0.031) D0 L0\ndetector D2")
    projected = project_dem(actual, (0, 2))
    expected = project_dem(reference, (0, 2))
    assert projected.structure == expected.structure
    np.testing.assert_array_equal(projected.probabilities, expected.probabilities)
    assert projected.dem == expected.dem
    assert projected.diagnostics["discarded_invisible_nonlogical_events"] == 1
    # The same uncertain event cannot be retained in the calibrated sector.
    with pytest.raises(DemStructureError, match="surviving Bernoulli"):
        project_dem(actual, (0, 1, 2))


@pytest.mark.parametrize("text,ids,message", [
    ("error(0.01) D0 D1 ^ D2 D3", None, "Residual hyperedge"),
    ("error(0.01) D0 L0\nerror(0.02) D0", None, "logical labels"),
    ("error(0.01) D0 L0\ndetector D1", (1,), "no selected detector"),
    ("error(0.5) D0", None, "0 <= p < 1/2"),
])
def test_unidentifiable_topology_is_rejected(text, ids, message):
    with pytest.raises(DemStructureError, match=message):
        project_dem(stim.DetectorErrorModel(text), ids)


def test_structure_json_contains_no_priors_and_round_trips():
    projected = project_dem(stim.DetectorErrorModel("error(0.0123) D0 L0\ndetector D1"))
    encoded = json.dumps(projected.structure.to_dict())
    assert "0.0123" not in encoded
    assert "probabilit" not in encoded
    restored = GraphStructure.from_dict(json.loads(encoded))
    assert restored == projected.structure
    assert restored.to_dem([0.02]).num_detectors == 2
    assert restored.to_dem([0.02]).num_observables == 1


def test_regularization_is_reported_and_strict_mode_rejects_it():
    structure = GraphStructure(2, 0, (ErrorSupport((0,)), ErrorSupport((0, 1)), ErrorSupport((1,))))
    moments = Moments(1000, np.array([0.1, 0.1]), np.diag([0.1, 0.1]))
    result = estimate_priors(structure, moments=moments, probability_floor=1e-7)
    assert result.p_ij[0, 1] < 0
    assert result.probabilities[1] == 1e-7
    assert result.diagnostics["pairs"]["negative"] == 1
    assert result.diagnostics["pairs"]["regularized"] == 1
    with pytest.raises(EstimationError, match="Nonphysical"):
        estimate_priors(structure, moments=moments, strict=True)


def test_invalid_pair_moment_is_preserved_as_nan_and_diagnosed():
    structure = GraphStructure(2, 0, (ErrorSupport((0, 1)),))
    # Valid binary joint distribution but incompatible with independent p<1/2 edges.
    moments = Moments(100, np.array([0.4, 0.4]), np.diag([0.4, 0.4]))
    result = estimate_priors(structure, moments=moments)
    assert np.isnan(result.raw_probabilities[0])
    assert result.diagnostics["pairs"]["nonfinite"] == 1
    assert result.diagnostics["max_marginal_residual"] > 0.39


def test_half_probability_marginal_always_raises():
    structure = GraphStructure(1, 0, (ErrorSupport((0,)),))
    for strict in (False, True):
        with pytest.raises(EstimationError, match="not identifiable"):
            estimate_priors(structure, moments=Moments(10, np.array([0.5]), np.array([[0.5]])), strict=strict)


def test_boundary_clipping_and_residual_are_auditable(tmp_path):
    structure = GraphStructure(3, 0, (ErrorSupport((0,)), ErrorSupport((0, 1)), ErrorSupport((0, 2))))
    # Mutually correlated pair estimates overcount detector 0's marginal.
    means = np.array([0.1, 0.1, 0.1])
    second = np.diag(means)
    second[0, 1] = second[1, 0] = 0.09
    second[0, 2] = second[2, 0] = 0.09
    result = estimate_priors(structure, moments=Moments(100, means, second))
    assert result.raw_probabilities[0] < 0
    assert result.diagnostics["boundaries"]["negative"] == 1
    assert result.diagnostics["max_marginal_residual"] > 0
    result.save(tmp_path)
    assert stim.DetectorErrorModel.from_file(tmp_path / "estimated.dem").num_detectors == 3
    json.loads((tmp_path / "diagnostics.json").read_text())
    np.testing.assert_allclose(np.load(tmp_path / "p_ij.npy"), result.p_ij)


def test_prior_changes_do_not_change_the_estimator_structure():
    first = project_dem(stim.DetectorErrorModel("error(0.01) D0 D1\nerror(0.02) D0 L0"))
    second = project_dem(stim.DetectorErrorModel("error(0.19) D0 D1\nerror(0.24) D0 L0"))
    assert first.structure == second.structure
    moments = exhaustive_moments(first.structure, first.probabilities)
    a = estimate_priors(first.structure, moments=moments)
    b = estimate_priors(second.structure, moments=moments)
    np.testing.assert_array_equal(a.probabilities, b.probabilities)
