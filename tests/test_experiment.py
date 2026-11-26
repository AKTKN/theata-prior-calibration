"""Scientific data-boundary and resumption tests for the stage orchestration.

The compiler is intentionally replaced by exact toy artifacts. The tests cover
the real on-disk sample -> fit -> paired-evaluation path without recompiling a
surface code in each test.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import stim

from theata_prior.config import ExperimentConfig
from theata_prior.dem import project_dem
from theata_prior.estimation import SaturatedMarginalsError
from theata_prior.experiment import (
    cases,
    evaluate_models_stage,
    fit_directory,
    fit_models,
    initialize_run,
    sample_models,
)
from theata_prior.io import atomic_json
from theata_prior.sampling import SampleSet, shard_seed


TOY_CIRCUIT = """R 0 1 2 3
X_ERROR(0.08) 0
E(0.03) X0 X1
E(0.02) X1 X2
X_ERROR(0.04) 3
E(0.01) X0 X3
M 0 1 2 3
DETECTOR rec[-4]
DETECTOR rec[-3]
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(0) rec[-2]
"""


def _config(**kwargs) -> ExperimentConfig:
    return ExperimentConfig(
        distances=(3,), physical_error_rates=(0.001,), training_shots=(256, 2048),
        test_shots=4096, workers=1, sample_chunk_shots=256, covariance_chunk_shots=256,
        decode_chunk_shots=256, **kwargs,
    )


def _write_toy_model(config: ExperimentConfig, root: Path):
    initialize_run(config, root)
    case, = cases(config, root)
    case.directory.mkdir(parents=True)
    circuit = stim.Circuit(TOY_CIRCUIT)
    actual = circuit.detector_error_model()
    marginal = project_dem(actual, [0, 1])
    actual.to_file(case.directory / "actual_full.dem")
    circuit.to_file(case.directory / "actual_noisy.stim")
    marginal.dem.to_file(case.directory / "actual_sector.dem")
    np.save(case.directory / "actual_priors.npy", marginal.probabilities)
    marginal.structure.to_dem(np.full(marginal.structure.num_errors, 0.001)).to_file(case.directory / "placeholder.dem")
    atomic_json(case.directory / "known_structure.json", marginal.structure.to_dict())
    atomic_json(case.directory / "model.json", {"detector_indices": [0, 1]})
    return case


def test_fit_has_no_oracle_or_logical_label_input(tmp_path, monkeypatch):
    config = _config()
    case = _write_toy_model(config, tmp_path)
    sample_models(config, tmp_path)
    fit_models(config, tmp_path)
    original = {
        count: np.load(fit_directory(case, 0, count) / "estimated_priors.npy").copy()
        for count in config.training_shots
    }

    # Corrupt oracle values, change the placeholder, and flip every training
    # observable label. These are deliberately not estimator inputs.
    np.save(case.directory / "actual_priors.npy", np.array([np.nan]))
    (case.directory / "placeholder.dem").write_text("error(0.49) D0 L0\n")
    labels = np.load(case.directory / "samples_train_000" / "observables.npy", mmap_mode="r+")
    labels[:] = ~labels
    labels.flush()
    del labels
    shutil.rmtree(case.directory / "fits")
    real_load = np.load

    def forbid_oracle_load(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).name in {"actual_priors.npy", "observables.npy"}:
            raise AssertionError(f"Fit tried to read forbidden data: {path}")
        return real_load(path, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(np, "load", forbid_oracle_load)
        fit_models(config, tmp_path)
    for count, values in original.items():
        np.testing.assert_array_equal(values, np.load(fit_directory(case, 0, count) / "estimated_priors.npy"))


def test_resume_allows_resources_but_rejects_scientific_change(tmp_path):
    config = _config()
    case = _write_toy_model(config, tmp_path)
    sample_models(config, tmp_path)
    path = case.directory / "samples_train_000" / "detectors.npy"
    before = np.load(path).copy()
    modified_ns = path.stat().st_mtime_ns
    resources = replace(config, workers=2, decode_chunk_shots=128, covariance_chunk_shots=128, covariance_block_detectors=1)
    sample_models(resources, tmp_path)
    np.testing.assert_array_equal(before, np.load(path))
    assert path.stat().st_mtime_ns == modified_ns
    with pytest.raises(ValueError, match="different experiment"):
        sample_models(replace(resources, seed=config.seed + 1), tmp_path)
    with pytest.raises(ValueError, match="different experiment"):
        initialize_run(replace(resources, sample_source="circuit"), tmp_path)
    np.testing.assert_array_equal(before, np.load(path))


def test_full_pipeline_uses_independent_training_and_paired_test_streams(tmp_path):
    for source in ("dem", "circuit"):
        config = _config(sample_source=source, repetitions=2)
        root = tmp_path / source
        case = _write_toy_model(config, root)
        sample_models(config, root)
        training = [SampleSet.open(case.directory / f"samples_train_{rep:03d}") for rep in range(2)]
        heldout = SampleSet.open(case.directory / "samples_test")
        sets = [*training, heldout]
        streams = [sample.metadata["config"]["stream"] for sample in sets]
        assert len(set(streams)) == 3
        assert len({shard_seed(config.seed, stream, 0) for stream in streams}) == 3
        assert all(sample.metadata["config"]["source"] == source for sample in sets)
        assert not np.array_equal(training[0].load_detectors(), training[1].load_detectors())
        assert not np.array_equal(training[0].load_detectors(), heldout.load_detectors(2048))
        np.testing.assert_array_equal(training[0].load_detectors(256), training[0].load_detectors()[:256])

        fit_models(config, root)
        rows = evaluate_models_stage(config, root)
        assert len(rows) == 1 + len(config.training_shots) * config.repetitions
        assert {row["shots"] for row in rows} == {config.test_shots}
        evaluation = json.loads((case.directory / "evaluation.json").read_text())
        assert evaluation["test_config"] == heldout.metadata["config"]
        baseline_errors = evaluation["models"]["actual"]["failures"]
        for name, paired in evaluation["paired"].items():
            assert paired["candidate_only_failures"] - paired["baseline_only_failures"] == evaluation["models"][name]["failures"] - baseline_errors
        for count in config.training_shots:
            for rep in range(config.repetitions):
                marker = json.loads((fit_directory(case, rep, count) / "fit_complete.json").read_text())
                assert marker["training_config"] == training[rep].metadata["config"]
                assert marker["shots"] == count


def test_opt_in_records_undefined_fit_and_evaluates_available_models(tmp_path):
    config = _config()
    case = _write_toy_model(config, tmp_path)
    sample_models(config, tmp_path)
    short, long = config.training_shots
    detectors = np.load(case.directory / "samples_train_000" / "detectors.npy", mmap_mode="r+")
    detectors[:short, 0] = True
    detectors.flush()
    del detectors

    with pytest.raises(SaturatedMarginalsError):
        fit_models(config, tmp_path)

    fit_models(config, tmp_path, skip_saturated_marginals=True)
    short_fit = fit_directory(case, 0, short)
    assert (short_fit / "fit_skipped.json").is_file()
    assert not (short_fit / "estimated.dem").exists()
    assert (fit_directory(case, 0, long) / "fit_complete.json").is_file()

    rows = evaluate_models_stage(config, tmp_path, skip_saturated_marginals=True)
    assert [(row["decoder"], row["calibration_shots"]) for row in rows] == [("actual", 0), ("estimated", long)]
    skipped, = json.loads((tmp_path / "fit_skips.json").read_text())
    assert skipped["calibration_shots"] == short
    assert skipped["saturated_detector_ids"] == [0]
    assert skipped["max_detector_mean"] == 1.0
    assert evaluate_models_stage(config, tmp_path, skip_saturated_marginals=True) == rows
