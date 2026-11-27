"""Scientific and integration checks for grouped decoder-prior calibration."""
from dataclasses import replace
import inspect
from pathlib import Path
import re

import numpy as np
import pytest
import stim

from theata_prior.decoder_training.adapters import Decoder, count_outcomes, incidence, verify_union_find
from theata_prior.decoder_training.cem import optimize, update_proposal
from theata_prior.decoder_training.config import CEMConfig, Resources, TrainingConfig, load_training_config
from theata_prior.decoder_training.diagnostics import compare_priors, pair_errors, prior_matrices
from theata_prior.decoder_training.experiment import evaluate, fit, preflight, sample, train
from theata_prior.decoder_training.grouping import make_groups
from theata_prior.decoder_training.objective import Scorer
from theata_prior.decoder_training.plotting import PlotOptions, load_results, plot_results
from theata_prior.decoder_training.weights import WeightModel, llrs
from theata_prior.dem import ErrorSupport, GraphStructure, model_moments
from theata_prior.estimation import Moments, estimate_priors, pair_probability_matrix
from theata_prior.experiment import cases
from theata_prior.io import atomic_json, read_json
from theata_prior.sampling import SampleSet, sample_to_disk


@pytest.fixture
def structure():
    return GraphStructure(
        2, 1, (ErrorSupport((0,), (0,)), ErrorSupport((0, 1)), ErrorSupport((1,))),
        detector_ids=(0, 1), detector_coordinates=((1, 3, 0), (3, 1, 1)),
    )


def test_incidence_grouping_and_realized_normalization(structure):
    h, b = incidence(structure)
    np.testing.assert_array_equal(h.toarray(), [[1, 1, 0], [0, 1, 1]])
    np.testing.assert_array_equal(b.toarray(), [[1, 0, 0]])
    assert make_groups(structure).scheme == "support_fallback"
    coarse = make_groups(structure, verified_coordinates=True)
    assert "pair diagonal, cross round / initial" in coarse.labels
    reverse = replace(structure, detector_coordinates=tuple(reversed(structure.detector_coordinates)))
    assert "pair diagonal, cross round / initial" in make_groups(reverse, verified_coordinates=True).labels
    with pytest.raises(ValueError, match="one group label"):
        make_groups(structure, "manual", manual=("all",))

    groups = make_groups(structure, "manual", manual=("a", "b", "c"))
    initial = np.array([.025, .05, .48])
    transform = WeightModel(initial, groups, .03, .49)
    original = transform.realize(np.zeros(2))
    np.testing.assert_array_equal(original.priors, [.03, .05, .48])
    assert transform.initial_clip_low == 1
    adjusted = transform.realize(np.array([2., -2.]))
    assert adjusted.theta[groups.reference] == 0
    assert adjusted.clipped_high or adjusted.clipped_low
    assert np.all((adjusted.priors >= .03) & (adjusted.priors <= .49))
    np.testing.assert_allclose(adjusted.weights, llrs(adjusted.priors), rtol=0, atol=1e-14)
    assert abs(adjusted.weights.mean() - original.weights.mean()) < 1e-11
    dem = stim.DetectorErrorModel(str(structure.to_dem(adjusted.priors)))
    assert abs(transform.verify_serialization(structure, dem, adjusted)) < 1e-11
    assert len([instruction for instruction in dem if instruction.type == "error"]) == 3


def _probabilities(weights):
    return 1 / (1 + np.exp(np.asarray(weights, dtype=float)))


def test_mwpm_weighted_decision_and_scale_invariance(structure):
    syndromes = np.array([[1, 0], [0, 0], [1, 1], [0, 1]], dtype=np.uint8)
    decoder = Decoder("mwpm", structure)
    for weights, expected in (([1, 9, 9], 1), ([9, 1, 1], 0)):
        original, failures = decoder.predictor(_probabilities(weights))(syndromes)
        scaled, _ = decoder.predictor(_probabilities(2.7 * np.asarray(weights)))(syndromes)
        assert original[0, 0] == expected
        assert not failures.any()
        np.testing.assert_array_equal(original, scaled)
    with pytest.raises(ValueError, match="strictly between"):
        decoder.predictor(np.array([0.0, 0.1, 0.1]))


def test_ldpc_wheel_weighted_decision_peeling_and_reuse(structure):
    pytest.importorskip("ldpc")
    from ldpc import UnionFindDecoder

    h, _ = incidence(structure)
    with pytest.raises(TypeError, match="expected str"):
        UnionFindDecoder(h)
    with pytest.raises(TypeError, match="expected str"):
        UnionFindDecoder(h, uf_method=False)
    assert verify_union_find() == {
        "bits_per_step_1": [[1, 0, 0], [0, 1, 1]],
        "bits_per_step_0": [[1, 0, 0], [1, 0, 0]],
        "zero_syndrome": [0, 0, 0],
    }
    syndromes = np.tile(np.array([[1, 0], [0, 0], [1, 1], [0, 1]], dtype=np.uint8), (5, 1))
    reused = Decoder("uf", structure)
    for weights, logical in (([1, 9, 9], 1), ([9, 1, 1], 0), ([1, 9, 9], 1)):
        probabilities = _probabilities(weights)
        predictions, failures = reused.predictor(probabilities)(syndromes)
        fresh, _ = Decoder("uf", structure).predictor(probabilities)(syndromes)
        assert not failures.any()
        assert predictions[0, 0] == logical and predictions[1, 0] == 0
        np.testing.assert_array_equal(predictions, fresh)


@pytest.mark.parametrize("name", ["mwpm", "uf"])
def test_infeasible_shots_and_mismatches_are_disjoint(name):
    if name == "uf":
        pytest.importorskip("ldpc")
    structure = GraphStructure(3, 2, (ErrorSupport((0, 1), (0,)),))
    detector = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 0, 1]], dtype=np.uint8)
    logical = np.array([[1, 0], [0, 0], [0, 0], [0, 1]], dtype=np.uint8)
    prediction, failed = Decoder(name, structure).predictor(np.array([.1]))(detector)
    assert prediction.shape == logical.shape
    np.testing.assert_array_equal(failed, [True, False, False, True])
    np.testing.assert_array_equal(count_outcomes(prediction, failed, logical), [2, 1, 0, 2])


def test_pair_inversion_and_finite_coverage(structure):
    structure = replace(structure, num_detectors=3, detector_ids=(17, 29, 42), detector_coordinates=())
    probabilities = np.array([.03, .05, .02])
    means, second = model_moments(structure, probabilities)
    # Independently enumerate the eight latent signatures for this fixture.
    h, _ = incidence(structure)
    patterns = np.array([[i >> j & 1 for j in range(3)] for i in range(8)])
    detectors = patterns @ h.toarray().T % 2
    mass = np.prod(np.where(patterns, probabilities, 1 - probabilities), axis=1)
    np.testing.assert_allclose(means, mass @ detectors, atol=1e-14)
    np.testing.assert_allclose(second, detectors.T @ (mass[:, None] * detectors), atol=1e-14)
    moments = Moments(0, means, second)
    np.testing.assert_allclose(estimate_priors(structure, moments=moments).probabilities, probabilities)
    pair, singleton, mask, singleton_mask = prior_matrices(structure, probabilities)
    np.testing.assert_allclose(pair_probability_matrix(moments), pair, atol=1e-14)
    np.testing.assert_array_equal(np.diag(pair), [0, 0, 0])
    np.testing.assert_array_equal(singleton, [.03, .02, 0])
    np.testing.assert_array_equal(singleton_mask, [True, True, False])
    diff = pair_errors(np.array([[0., -.06, np.nan], [-.06, 0, 0], [np.nan, 0, 0]]), mask, finite_only=True)
    assert diff["pair_matrix_support"]["rmse"] == pytest.approx(.06)
    assert diff["pair_matrix_all"]["rmse"] == pytest.approx(.06 / np.sqrt(2))
    assert diff["pair_matrix_all"]["coverage"] == pytest.approx(2 / 3)
    assert diff["pair_matrix_all"]["invalid_count"] == 1
    assert pair_errors(pair, np.zeros_like(mask))["pair_matrix_support"]["status"] == "empty"
    stats = compare_priors(structure, probabilities * 2, probabilities, make_groups(structure))
    assert stats["pair_matrix_support"]["rmse"] > stats["pair_matrix_all"]["rmse"]


def test_flat_ties_preserve_proposal_and_cutoff_ties_split_elite_mass():
    config = replace(CEMConfig(), population_size=4, elite_count=2, smoothing=1.0)
    draws = np.array([[-1.5], [0.], [1.], [2.]])
    mean, variance = np.array([0.2]), np.array([.5])
    next_mean, next_variance, flat = update_proposal(draws, np.zeros(4), mean, variance, config)
    assert flat["flat"]
    np.testing.assert_array_equal(next_mean, mean)
    np.testing.assert_array_equal(next_variance, variance)
    shifted, _, diag = update_proposal(draws, np.array([1, 2, 2, 2]), mean, variance, config)
    np.testing.assert_allclose(diag["elite_weights"], [.5, 1/6, 1/6, 1/6])
    np.testing.assert_allclose(shifted, [-.25])


def _samples(structure, tmp_path):
    source = str(structure.to_dem([.22, .08, .07]))
    return [sample_to_disk(source, tmp_path / stream, shots=191, seed=420, stream=stream, chunk_shots=46)
            for stream in ("training", "validation")]


def _without_timing(value):
    if isinstance(value, list):
        return [_without_timing(child) for child in value]
    if isinstance(value, dict):
        return {key: _without_timing(child) for key, child in value.items() if key != "seconds"}
    return value


def test_tied_draws_can_be_selected_when_they_beat_controls(structure, tmp_path, monkeypatch):
    training, validation = _samples(structure, tmp_path)
    groups = make_groups(structure)
    config = CEMConfig(generations=1, train_batch_shots=20)

    def score(self, task):
        theta, indices, is_validation = task
        shots = len(self.validation[0]) if is_validation else len(indices)
        return {"failures": 1 if np.any(theta) else 2, "shots": shots, "decoder_failures": 0, "seconds": 0}

    monkeypatch.setattr(Scorer, "score", score)
    folder = tmp_path / "run"
    result = optimize(structure, training, validation, np.array([.09, .05, .05]), groups, config, folder,
                      train_shots=50, decoder="mwpm", seed=43)
    state = read_json(folder / "state.json")
    assert result["flat_generations"] == 1
    assert result["selected"]["generation"] == 1 and result["selected"]["failures"] == 1
    assert state["mean"] == [0] and state["variance"] == [config.initial_std ** 2]


def test_resume_worker_and_chunk_independence_and_dem_round_trip(structure, tmp_path):
    training, validation = _samples(structure, tmp_path)
    groups = make_groups(structure, "manual", manual=("left", "pair", "right"))
    config = CEMConfig(generations=3, train_batch_shots=19, validation_interval=2)
    initial = np.array([.09, .05, .05])
    common = (structure, training, validation, initial, groups, config)
    args = {"decoder": "mwpm", "seed": 17, "train_shots": 91}
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    optimize(*common, serial, chunk_shots=13, **args)
    assert optimize(*common, parallel, candidate_jobs=2, stop_after=1, **args)["status"] == "paused"
    optimize(*common, parallel, candidate_jobs=1, chunk_shots=32, **args)
    for path in ("state.json", *[f"history/generation_{i:05d}.json" for i in range(1, 4)]):
        assert _without_timing(read_json(serial / path)) == _without_timing(read_json(parallel / path))
    saved = stim.DetectorErrorModel.from_file(serial / "learned.dem")
    exported = np.array([line.args_copy()[0] for line in saved if line.type == "error"])
    frozen = read_json(serial / "state.json")["best_validation"]
    realized = WeightModel(initial, groups, 1e-9, .49).realize(frozen["theta"])
    np.testing.assert_array_equal(exported, realized.priors)
    from pymatching import Matching
    samples = validation.load_detectors().astype(np.uint8)
    predicted, _ = Decoder("mwpm", structure).predictor(realized.priors)(samples)
    np.testing.assert_array_equal(predicted, Matching.from_detector_error_model(saved).decode_batch(samples))
    with pytest.raises(ValueError, match="Optimization inputs changed"):
        optimize(*common, serial, **{**args, "seed": 18})
    with training.observable_path.open("r+b") as handle:
        handle.seek(-1, 2)
        byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match="Optimization inputs changed"):
        optimize(*common, serial, **args)


@pytest.mark.parametrize("case_status", ["no_free_parameters", "degenerate_mean_at_bound"])
def test_unsearchable_initialization_is_evaluated_and_exported(structure, tmp_path, case_status):
    training, validation = _samples(structure, tmp_path)
    groups = (make_groups(structure, "manual", manual=("one",) * 3)
              if case_status == "no_free_parameters" else make_groups(structure))
    initial = np.full(3, 1e-10 if case_status == "degenerate_mean_at_bound" else .01)
    result = optimize(structure, training, validation, initial, groups, CEMConfig(), tmp_path / "run",
                      train_shots=50, decoder="mwpm", seed=42)
    assert result["status"] == case_status
    assert result["optimizer_evaluations"] == result["distinct_train_shots"] == 0
    assert result["validation_evaluations"] == 1 and (tmp_path / "run/learned.dem").exists()


def _synthetic_case(config, root, structure, priors):
    case = next(cases(config.physical, root / "models"))
    folder = case.directory
    atomic_json(folder / "known_structure.json", structure.to_dict())
    atomic_json(folder / "model.json", {"detector_indices": list(structure.detector_ids)})
    structure.to_dem(priors).to_file(folder / "actual_full.dem")
    np.save(folder / "actual_priors.npy", priors)
    return case


def _config(**kwargs):
    config = TrainingConfig(resources=Resources(total_workers=1, candidate_jobs=1, sampling_jobs=1))
    return replace(config, physical=replace(config.physical, training_shots=(128, 256), test_shots=384,
                                            sample_chunk_shots=77, decode_chunk_shots=47),
                   validation_shots=96, cem=replace(config.cem, generations=2, train_batch_shots=48), **kwargs)


def test_synthetic_pipeline_uses_no_oracle_until_evaluation(structure, tmp_path, monkeypatch):
    pytest.importorskip("ldpc")
    config = _config(initializations=("covariance", "uniform"))
    case = _synthetic_case(config, tmp_path, structure, np.array([.12, .07, .05]))
    sample(config, tmp_path)
    path_open = Path.open
    numpy_load = np.load
    forbidden = re.compile(r"actual_|samples_test|samples_validation|observables.npy")

    def check(file):
        if isinstance(file, (str, Path)) and forbidden.search(str(file)):
            raise AssertionError(f"Forbidden file read before held-out evaluation: {file}")

    def guarded_open(path, *args, **kwargs):
        check(path)
        return path_open(path, *args, **kwargs)

    def guarded_load(file, *args, **kwargs):
        check(file)
        return numpy_load(file, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(Path, "open", guarded_open)
        m.setattr(np, "load", guarded_load)
        fits = fit(config, tmp_path)
    assert [row["calibration_shots"] for row in fits] == [128, 256]
    forbidden = re.compile(r"actual_|samples_test")
    with monkeypatch.context() as m:
        m.setattr(Path, "open", guarded_open)
        m.setattr(np, "load", guarded_load)
        runs = train(config, tmp_path)
    assert len(runs) == 8
    assert {"actual", "test", "oracle", "latent"}.isdisjoint(inspect.signature(optimize).parameters)
    rows = evaluate(config, tmp_path)
    assert len(rows) == 40 and all(row["status"] == "ok" for row in rows)
    assert len({row["test_sha256"] for row in rows}) == 1
    assert {row["evaluation_decoder"] for row in rows if row["method"] == "cem_mwpm"} == {"mwpm", "uf"}
    assert all((row["actual_n10"] - row["actual_n01"]) / row["test_shots"] == row["actual_delta"] for row in rows)
    assert all(row["failures"] == row["decoder_failures"] + row["logical_mismatches"] for row in rows)
    saved, diagnostics, index = load_results(tmp_path)
    assert saved == rows and len(index) == 4
    with np.load(tmp_path / index[0]["matrices"], allow_pickle=False) as arrays:
        assert "pair_discrepancy_cem_uf_vs_covariance_raw" in arrays
        assert "detector_covariance" in arrays
    assert any(row.get("finite_only") for row in diagnostics)
    paths = plot_results(tmp_path, PlotOptions(calibration_shots=256, initialization="uniform", formats=("png", "pdf")))
    assert len(paths) == 22 and all(path.stat().st_size > 1000 for path in paths)
    assert load_results(tmp_path, decoders=("mwpm", "uf")) == (saved, diagnostics, index)
    uf_rows, uf_diagnostics, uf_index = load_results(tmp_path, decoders=("uf",))
    assert {row["evaluation_decoder"] for row in uf_rows} == {"uf"}
    assert {row["method"] for row in uf_rows} == {"actual", "covariance", "initial", "cem_uf"}
    assert all(row["method"] != "cem_mwpm" for row in uf_diagnostics)
    assert all(set(row["traces"]) == {"cem_uf"} and "cem_mwpm" not in row["statuses"] for row in uf_index)
    uf_paths = plot_results(tmp_path, PlotOptions(decoders=("uf",), formats=("png",)))
    assert len(uf_paths) == 10 and {path.parent for path in uf_paths} == {tmp_path / "figures" / "uf"}
    assert "trace_cem_uf.png" in {path.name for path in uf_paths}
    assert "trace_cem_mwpm.png" not in {path.name for path in uf_paths}
    assert preflight(config, tmp_path, probe_shots=17)[0]["uf_timing_probe_shots"] == 17
    assert SampleSet.open(case.directory / "samples_test").shots == 384


@pytest.mark.parametrize("decoder", ["mwpm", "uf"])
def test_single_decoder_configuration_trains_and_evaluates_selected_decoder(structure, tmp_path, decoder):
    if decoder == "uf":
        pytest.importorskip("ldpc")
    selected = (decoder,)
    config = _config(target_decoders=selected, evaluation_decoders=selected, plot_decoders=selected)
    _synthetic_case(config, tmp_path, structure, np.array([.12, .07, .05]))
    sample(config, tmp_path)
    fit(config, tmp_path)
    assert {row["target_decoder"] for row in train(config, tmp_path)} == {decoder}
    assert {row["evaluation_decoder"] for row in evaluate(config, tmp_path)} == {decoder}
    other = "mwpm" if decoder == "uf" else "uf"
    with pytest.raises(ValueError, match=f"No saved evaluations for {other}"):
        load_results(tmp_path, decoders=(other,))


def test_saturated_covariance_is_unavailable_and_uniform_fallback_is_explicit(tmp_path):
    structure = GraphStructure(3, 1, (ErrorSupport((0,), (0,)), ErrorSupport((0, 1)), ErrorSupport((0, 2)),
                                     ErrorSupport((1,)), ErrorSupport((1, 2)), ErrorSupport((2,))))
    config = _config(covariance_failure="uniform", target_decoders=("mwpm",), evaluation_decoders=("mwpm",))
    case = _synthetic_case(config, tmp_path, structure, np.full(6, .20))
    sample(config, tmp_path)
    # Build valid binary moment data with a saturated training marginal independent of sampling fluctuations.
    detector_path = case.directory / "samples_train_000/detectors.npy"
    detectors = np.load(detector_path, mmap_mode="r+")
    detectors[:, 0] = True
    detectors.flush()
    assert all(row["status"] == "saturated_marginals" for row in fit(config, tmp_path))
    assert all(row["realized_initialization"] == "uniform_fallback" for row in train(config, tmp_path))
    rows = evaluate(config, tmp_path)
    assert all(row["status"] == "saturated_marginals" for row in rows if row["method"] == "covariance")
    assert all(row["covariance_paired_status"] == "saturated_marginals" for row in rows)
    assert plot_results(tmp_path, PlotOptions(formats=("png",)))


def test_configuration_and_cpu_budget(monkeypatch, tmp_path):
    from theata_prior.decoder_training import config

    monkeypatch.setattr(config, "available_cpus", lambda: 2)
    path = tmp_path / "experiment.yaml"
    path.write_text("physical:\n  distances: [3, 5]\nrounds_by_distance: {3: 3, 5: 5}\nplot_decoders: [uf]\n")
    configured = load_training_config(path)
    assert configured.rounds_by_distance == {3: 3, 5: 5}
    assert configured.plot_decoders == ("uf",)
    assert configured.scientific_dict() == replace(configured, plot_decoders=None).scientific_dict()
    with pytest.raises(ValueError, match="plot decoders"):
        PlotOptions(decoders=("uf", "uf"))
    with pytest.raises(ValueError, match="allocation exceeds"):
        Resources(total_workers=1, sampling_jobs=1, candidate_jobs=2)
    monkeypatch.setattr(config, "available_cpus", lambda: 1)
    assert Resources().candidate_jobs == Resources().sampling_jobs == 1
