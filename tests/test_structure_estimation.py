from itertools import product

import numpy as np
import pytest
from scipy.stats import fisher_exact

from theata_prior.candidates import CandidateSet, DetectorMetadata
from theata_prior.estimation import Moments, compute_moments
from theata_prior.logical_labels import LogicalParities, fit_logical_labels, prefix_logical_counts
from theata_prior.structure_estimation import (
    attenuation, fisher_pair_pvalues, fit_structure, holm_adjust, integer_counts, invert_detector_moments,
    prefix_moments, singleton_influence_chunks, singleton_standard_errors,
)


def enumerate_faults(m, mechanisms):
    """Independent oracle over every microscopic fault pattern, including both L bits."""
    detectors, observables, weights = [], [], []
    for events in product((0, 1), repeat=len(mechanisms)):
        d = np.zeros(m, dtype=int)
        l = 0
        weight = 1.0
        for fired, (support, label, probability) in zip(events, mechanisms, strict=True):
            weight *= probability if fired else 1 - probability
            if fired:
                d[list(support)] ^= 1
                l ^= label
        detectors.append(d)
        observables.append(l)
        weights.append(weight)
    return np.array(detectors), np.array(observables), np.array(weights)


def weighted_moments(d, weights):
    return Moments(0, weights @ d, d.T @ (weights[:, None] * d))


def augmented_oracle(d, l, weights):
    family = CandidateSet(d.shape[1])
    i, j = family.pairs
    return LogicalParities(float(weights @ (1 - 2 * l)), weights @ (1 - 2 * (d ^ l[:, None])),
                           weights @ (1 - 2 * (d[:, i] ^ d[:, j] ^ l[:, None])))


def test_complete_candidate_order_detector_map_and_emitted_labels():
    metadata = DetectorMetadata((12, 2, 90, 3), ((0,), (1,), (2,), (3,)), 1)
    assert DetectorMetadata.from_dict(metadata.to_dict()) == metadata
    candidates = metadata.candidates
    assert len(candidates) == 10
    supports = [candidates.support(k) for k in range(len(candidates))]
    assert supports == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (0,), (1,), (2,), (3,)]
    assert [candidates.index(tuple(reversed(s))) for s in supports] == list(range(10))
    rates = np.zeros(len(candidates))
    rates[candidates.index((0, 3))] = 0.08
    graph, p = metadata.graph(rates, np.zeros(len(candidates), dtype=bool))
    dem = graph.to_dem(p)
    assert graph.detector_ids == metadata.detector_ids
    assert dem.num_detectors == 4 and dem.num_observables == 1
    assert dem.num_errors == 1 and len(graph.edges[0].observables) == 0


def test_population_exact_support_recovery_and_indirect_path_null():
    mechanisms = [((0, 3), 1, 0.07), ((0, 3), 1, 0.025), ((1, 3), 0, 0.12), ((1,), 1, 0.02)]
    d, _, weights = enumerate_faults(5, mechanisms)
    moments = weighted_moments(d, weights)
    candidates = CandidateSet(5)
    actual = np.zeros(len(candidates))
    for support, _, rate in mechanisms:
        k = candidates.index(support)
        actual[k] = actual[k] * (1 - rate) + (1 - actual[k]) * rate
    inverted = invert_detector_moments(moments)
    np.testing.assert_allclose(inverted.probabilities(candidates), actual, atol=2e-15, rtol=0)
    assert moments.covariance[0, 1] == pytest.approx(0, abs=2e-15)
    assert moments.covariance[0, 3] > 0 and moments.covariance[1, 3] > 0
    assert moments.means[4] == 0


def test_integer_counts_fisher_orientation_and_full_family_holm():
    n = 1_000_000
    counts = np.array([9, 7, 0, n])
    joint = np.diag(counts)
    joint[0, 1] = joint[1, 0] = 3
    joint[0, 3] = joint[3, 0] = 9
    joint[1, 3] = joint[3, 1] = 7
    mon = Moments(n, counts / n, joint / n)
    np.testing.assert_array_equal(integer_counts(mon)[0], counts)
    np.testing.assert_array_equal(integer_counts(mon)[1], joint)
    actual = fisher_pair_pvalues(mon, batch_pairs=1)
    expected = fisher_exact([[3, 6], [4, 999987]], alternative="greater").pvalue
    assert actual[0] == pytest.approx(expected, rel=1e-12)
    np.testing.assert_array_equal(actual[1:], np.ones(5))
    assert holm_adjust(actual)[0] == pytest.approx(6 * expected, rel=1e-12)
    np.testing.assert_allclose(holm_adjust([0.025, 0.01, 0.4, 1]), [0.075, 0.04, 0.8, 1])
    with pytest.raises(ValueError, match="integer counts"):
        integer_counts(Moments(15, np.array([0.5]), np.array([[0.5]])))


def test_singleton_influence_matches_finite_difference_and_monte_carlo():
    d, _, weights = enumerate_faults(3, [((0, 2), 0, 0.06), ((1, 2), 0, 0.035), ((1,), 0, 0.04)])
    raw = invert_detector_moments(weighted_moments(d, weights))
    analytic = np.concatenate(list(singleton_influence_chunks(d, raw, block_detectors=1, chunk_shots=2, workers=2)))
    step = 1e-6
    for k in range(len(d)):
        direction = np.eye(len(d))[k] - weights
        plus = invert_detector_moments(weighted_moments(d, weights + step * direction))
        minus = invert_detector_moments(weighted_moments(d, weights - step * direction))
        derivative = (plus.singleton_attenuation - minus.singleton_attenuation) / (2 * step)
        np.testing.assert_allclose(analytic[k], derivative, atol=2e-9, rtol=0)

    rng = np.random.default_rng(801)
    estimates, errors = [], []
    for index in rng.choice(len(d), size=(160, 800), p=weights):
        x = d[index].astype(float)
        sample = invert_detector_moments(Moments(len(x), x.mean(axis=0), x.T @ x / len(x)))
        estimates.append(sample.singleton_attenuation)
        errors.append(singleton_standard_errors(x, sample, chunk_shots=250))
    np.testing.assert_allclose(np.std(estimates, axis=0, ddof=1), np.mean(errors, axis=0), rtol=0.23)


def test_negative_estimates_unresolved_domains_and_exact_zero_rejections():
    negative = np.array([[0, 0]] * 70 + [[1, 0]] * 15 + [[0, 1]] * 15)
    result = fit_structure(negative)
    assert result.valid and result.arrays["raw_p_ij"][0, 1] < 0
    assert result.arrays["complete_clipped_priors"][0] == 0
    weak = np.array([[0, 0]] * 65 + [[1, 0]] * 15 + [[0, 1]] * 15 + [[1, 1]] * 5)
    result = fit_structure(weak)
    assert result.arrays["complete_clipped_priors"][0] > 0
    assert not result.arrays["complete_selected_selected"][0]
    assert result.arrays["complete_selected_priors"][0] == 0
    zeros = fit_structure(np.column_stack((weak, np.zeros(len(weak), dtype=int))))
    assert zeros.arrays["singleton_zero_information"][-1]
    assert zeros.arrays["singleton_pvalues"][-1] == 1
    for data, key in ((np.array([[0, 0]] * 40 + [[1, 0]] * 30 + [[0, 1]] * 30), "nonpositive_pair_parities"),
                      (np.array([[0, 0]] * 3 + [[1, 0]] + [[0, 1]] * 2), "nonpositive_pair_parities"),
                      (np.array([[0], [1]]), "nonpositive_marginal_ids")):
        invalid = fit_structure(data)
        assert not invalid.valid and invalid.diagnostics[key]
        assert "complete_selected_priors" not in invalid.arrays


def test_independent_label_oracle_variants_ambiguity_and_pure_logical_residual():
    mechanisms = [((0, 2), 1, 0.1), ((1, 2), 0, 0.03), ((0,), 0, 0.04), ((2,), 1, 0.07)]
    d, l, weights = enumerate_faults(4, mechanisms)
    fit = fit_logical_labels(weighted_moments(d, weights), augmented_oracle(d, l, weights))
    assert fit.valid
    family = CandidateSet(4)
    for support, label, probability in mechanisms:
        k = family.index(support)
        assert fit.arrays["label_bits"][k] == label
        assert fit.arrays[f"raw_prior_label_{label}"][k] == pytest.approx(probability)
        assert fit.arrays[f"attenuation_label_{1-label}"][k] == pytest.approx(0, abs=1e-14)
    assert fit.diagnostics["pure_logical_attenuation"] == pytest.approx(0, abs=1e-14)

    mixed = [((0, 1), 0, 0.1), ((0, 1), 1, 0.1), ((), 1, 0.05)]
    d, l, weights = enumerate_faults(2, mixed)
    fit = fit_logical_labels(weighted_moments(d, weights), augmented_oracle(d, l, weights))
    assert fit.arrays["contrast"][0] == pytest.approx(0, abs=1e-14)
    assert fit.arrays["unique_variant_discrepancy"][0] == pytest.approx(-np.log1p(-0.2))
    assert fit.diagnostics["pure_logical_attenuation"] == pytest.approx(-np.log1p(-0.1))
    tied = fit_logical_labels(Moments(0, np.zeros(2), np.zeros((2, 2))),
                              LogicalParities(1.0, np.ones(2), np.ones(1)))
    assert np.all(tied.arrays["ambiguous"]) and not np.any(tied.arrays["label_bits"])
    invalid = fit_logical_labels(weighted_moments(d, weights),
                                 LogicalParities(0, np.ones(2), np.ones(1)))
    assert not invalid.valid and invalid.diagnostics["nonpositive_logical_bias"]


def test_prefix_counts_and_discrete_decisions_are_worker_invariant():
    rng = np.random.default_rng(392)
    detectors = rng.random((582, 5)) < 0.07
    logical = rng.random((582, 1)) < 0.13
    sizes = (129, 582)
    for n, moments, signed in zip(sizes, prefix_moments(detectors, sizes, chunk_shots=30, workers=2),
                                  prefix_logical_counts(detectors, logical, sizes, chunk_shots=30, workers=2), strict=True):
        serial = compute_moments(detectors[:n], workers=1)
        np.testing.assert_array_equal(moments.second_moments, serial.second_moments)
        parities = LogicalParities.from_counts(moments, *signed)
        reference = augmented_oracle(detectors[:n], logical[:n, 0], np.full(n, 1 / n))
        np.testing.assert_allclose(parities.detector_logical, reference.detector_logical, atol=1e-14)
        np.testing.assert_allclose(parities.pair_logical, reference.pair_logical, atol=1e-14)
    serial = fit_structure(detectors, workers=1, chunk_shots=41, block_detectors=2)
    parallel = fit_structure(detectors, workers=2, chunk_shots=30, block_detectors=1)
    for key in ("complete_selected_selected", "complete_selected_active", "pair_pvalues"):
        np.testing.assert_array_equal(serial.arrays[key], parallel.arrays[key])
    np.testing.assert_allclose(serial.arrays["singleton_se"], parallel.arrays["singleton_se"], rtol=1e-12)
