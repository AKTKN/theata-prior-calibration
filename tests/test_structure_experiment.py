from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pytest
import stim

from theata_prior.candidates import DetectorMetadata
from theata_prior.config import ExperimentConfig
from theata_prior.decoding import evaluate_models
from theata_prior.experiment import cases
from theata_prior.io import atomic_json, read_json
from theata_prior.sampling import sample_to_disk
from theata_prior.structure_config import StructureExperimentConfig, load_structure_config
from theata_prior.structure_experiment import evaluate_structure_stage, fit_structure_stage, label_structure_stage


MODEL = """error(0.1) D0 L0
error(0.035) D0 D2
error(0.06) D1
detector D0
detector D1
detector D2
logical_observable L0
"""


def _fixture(tmp_path, *, label_mode="detector_only", model=MODEL):
    base = ExperimentConfig(physical_error_rates=(0.002,), training_shots=(200, 1200), test_shots=700,
                            sample_chunk_shots=128, decode_chunk_shots=129, covariance_chunk_shots=91)
    config = StructureExperimentConfig(base, label_mode=label_mode)
    simulation = tmp_path / "simulation"
    case, = cases(base, simulation)
    blind = tmp_path / "blind" / case.directory.relative_to(simulation)
    metadata = DetectorMetadata((0, 1, 2), ((), (), ()), 1)
    atomic_json(blind / "detector_metadata.json", metadata.to_dict())
    for name in ("actual_full.dem", "actual_sector.dem"):
        (case.directory / name).parent.mkdir(parents=True, exist_ok=True)
        (case.directory / name).write_text(model)
    namespace = f"d3/r3/z/p{case.physical_error_rate.hex()}"
    settings = dict(seed=base.seed, chunk_shots=base.sample_chunk_shots)
    train = sample_to_disk(model, case.directory / "samples_train_000", shots=max(base.training_shots),
                           stream=f"{namespace}/train/0", **settings)
    test = sample_to_disk(model, case.directory / "samples_test", shots=base.test_shots,
                          stream=f"{namespace}/test", **settings)
    return config, case, blind, train, test


def _flip(path):
    data = np.load(path, mmap_mode="r+")
    data[:] = ~data
    data.flush()


def _arrays(folder, name):
    with np.load(folder / name, allow_pickle=False) as data:
        return dict(data)


@pytest.mark.parametrize("mode", ["detector_only", "calibration_observables"])
def test_fits_have_no_oracle_or_heldout_access(tmp_path, monkeypatch, mode):
    config, case, blind, train, test = _fixture(tmp_path, label_mode=mode)
    fit_structure_stage(config, tmp_path)
    label_structure_stage(config, tmp_path)
    folder = blind / "rep000_n1200"
    arrays = {"detector_fit.npz": _arrays(folder, "detector_fit.npz")}
    if mode == "calibration_observables":
        arrays["logical_labels.npz"] = _arrays(folder, "logical_labels.npz")
    for path in blind.glob("rep*"):
        shutil.rmtree(path)
    for path in (test.detector_path, test.observable_path):
        _flip(path)
    if mode == "detector_only":
        _flip(train.observable_path)
    for name in ("known_structure.json", "actual_priors.npy", "actual_sector.dem", "actual_full.dem", "model.json"):
        (case.directory / name).unlink(missing_ok=True)

    def no_oracle_access(self, *args, **kwargs):
        if self.name in {"known_structure.json", "actual_priors.npy", "actual_sector.dem", "actual_full.dem", "model.json"} or "samples_test" in self.parts:
            pytest.fail(f"Fitter opened prohibited file {self}")
        if mode == "detector_only" and self.name == "observables.npy":
            pytest.fail(f"Detector-only fit opened {self}")
        return real_open(self, *args, **kwargs)

    real_open = Path.open
    monkeypatch.setattr(Path, "open", no_oracle_access)
    fit_structure_stage(config, tmp_path)
    label_structure_stage(config, tmp_path)
    for name, expected in arrays.items():
        actual = _arrays(folder, name)
        for field, value in expected.items():
            np.testing.assert_array_equal(actual[field], value)
    dem = stim.DetectorErrorModel.from_file(folder / "complete_selected_detector_only.dem")
    rates = arrays["detector_fit.npz"]["complete_selected_priors"]
    active = arrays["detector_fit.npz"]["complete_selected_active"]
    np.testing.assert_array_equal(rates[~active], 0)
    assert dem.num_errors == np.count_nonzero(active) and dem.num_observables == 0
    resource_change = replace(config, base=replace(config.base, workers=2), case_jobs=2)
    before = (folder / "fit.json").read_bytes()
    fit_structure_stage(resource_change, tmp_path)
    assert (folder / "fit.json").read_bytes() == before
    with pytest.raises(ValueError, match="Cached identity differs"):
        fit_structure_stage(replace(config, alpha_pairs=0.05), tmp_path)


def test_invalid_fit_keeps_baseline_and_valid_prefix(tmp_path):
    config, _, blind, train, _ = _fixture(tmp_path)
    data = np.load(train.detector_path, mmap_mode="r+")
    data[:200, 0] = True
    data.flush()
    del data
    fit_structure_stage(config, tmp_path)
    tables = evaluate_structure_stage(config, tmp_path)
    assert read_json(blind / "rep000_n200" / "fit.json")["status"] == "invalid_detector_parity"
    assert any(r["method"] == "actual_prior" and r["status"] == "complete" and r["shots"] == 700
               for r in tables["decoding"])
    unavailable = [r for r in tables["support"] if r["method"] == "complete_selected"
                   and r["calibration_shots"] == 200 and r["support_type"] == "combined"]
    assert unavailable[0]["unresolved"] == 6 and unavailable[0]["fn"] is None
    assert any(r["status"] == "complete" and r["calibration_shots"] == 1200
               for r in tables["support"] if r["method"] == "complete_selected")
    assert not (tmp_path / "label_metrics.csv").exists()
    assert {"complete_clipped", "complete_selected"} <= {r["method"] for r in tables["skipped"]}


def test_pruned_decoder_counts_odd_components_without_oracle_fallback(tmp_path):
    sample = sample_to_disk(MODEL, tmp_path / "test", shots=601, seed=18, stream="test")
    pruned = "error(0.035) D0 D2\nerror(0.06) D1\ndetector D0\ndetector D1\ndetector D2\nlogical_observable L0\n"
    args = (sample, {"actual_prior": MODEL, "pruned": pruned})
    with pytest.raises(ValueError, match="No perfect matching"):
        evaluate_models(*args, baseline="actual_prior")
    rows = evaluate_models(*args, baseline="actual_prior", allow_unmatched=True, chunk_shots=37, workers=2)
    summary = rows["models"]["pruned"]
    data = sample.load_detectors()
    odd = int(np.count_nonzero(data[:, 0] ^ data[:, 2]))
    assert odd > 0
    assert summary["shots"] == 601 and summary["decoding_failures"] == odd
    assert summary["successfully_decoded_shots"] == 601 - odd
    assert summary["failures"] == odd and summary["conditional_logical_mismatch_rate"] == 0
    assert rows["paired"]["pruned"]["candidate_only_failures"] == odd


def test_strict_configuration_parser(tmp_path):
    path = tmp_path / "structure.yaml"
    path.write_text("structure:\n  alpha_pairs: 0.03\n  label_mode: detector_only\nworkers: auto\n")
    config = load_structure_config(path, workers=3)
    assert config.alpha_pairs == 0.03 and config.to_base_config().workers == 3
    path.write_text("structure:\n  unknown_key: 123\n")
    with pytest.raises(ValueError, match="Unknown structure options"):
        load_structure_config(path)
