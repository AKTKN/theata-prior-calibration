from __future__ import annotations

import json

import numpy as np
import pytest

from theata_prior.decoding import evaluate_models
from theata_prior.metrics import paired_error_statistics, wilson_interval
from theata_prior.sampling import SampleSet, sample_to_disk, shard_seed


MODEL = """error(0.1) D0
error(0.05) D0 D1
error(0.02) D1 L0
detector D0
detector D1
logical_observable L0
"""


def test_worker_count_does_not_change_samples(tmp_path):
    kwargs = dict(shots=2048, seed=492, stream="train", chunk_shots=256)
    serial = sample_to_disk(MODEL, tmp_path / "serial", workers=1, **kwargs)
    parallel = sample_to_disk(MODEL, tmp_path / "parallel", workers=2, **kwargs)
    np.testing.assert_array_equal(serial.load_detectors(), parallel.load_detectors())
    np.testing.assert_array_equal(serial.load_observables(), parallel.load_observables())
    assert serial.load_detectors(300).shape == (300, 2)
    test = sample_to_disk(MODEL, tmp_path / "test", workers=1, **{**kwargs, "stream": "test"})
    assert not np.array_equal(serial.load_detectors(), test.load_detectors())
    assert shard_seed(492, "train", 0) != shard_seed(492, "test", 0)


def test_paired_evaluation_and_multicore(tmp_path):
    samples = sample_to_disk(MODEL, tmp_path / "samples", shots=4096, seed=17, stream="heldout", chunk_shots=512)
    incorrect = MODEL.replace("error(0.1)", "error(0.0001)").replace("error(0.02)", "error(0.4)")
    models = {"actual": MODEL, "same": MODEL, "candidate": incorrect}
    serial = evaluate_models(samples, models, workers=1, chunk_shots=256)
    parallel = evaluate_models(samples, models, workers=2, chunk_shots=256)
    assert serial["models"] == parallel["models"]
    assert serial["paired"] == parallel["paired"]
    assert serial["paired"]["same"]["candidate_only_failures"] == 0
    assert serial["paired"]["same"]["baseline_only_failures"] == 0
    for name in ("same", "candidate"):
        pair = serial["paired"][name]
        assert pair["candidate_only_failures"] - pair["baseline_only_failures"] == serial["models"][name]["failures"] - serial["models"]["actual"]["failures"]


def test_resume_partial_cache_and_reject_different_source(tmp_path):
    output = tmp_path / "samples"
    kwargs = dict(shots=200, seed=42, chunk_shots=50)
    samples = sample_to_disk(MODEL, output, **kwargs)
    expected = samples.load_detectors().copy()
    manifest_path = output / "samples.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "incomplete"
    manifest["completed_shards"] = [0, 2]
    manifest_path.write_text(json.dumps(manifest))
    mmap = np.load(samples.detector_path, mmap_mode="r+")
    mmap[50:100] = False
    mmap[150:200] = False
    mmap.flush()
    del mmap
    resumed = sample_to_disk(MODEL, output, workers=2, **kwargs)
    np.testing.assert_array_equal(expected, resumed.load_detectors())
    with pytest.raises(ValueError, match="configuration differs"):
        sample_to_disk(MODEL.replace("0.1", "0.2"), output, **kwargs)
    with pytest.raises(FileExistsError):
        sample_to_disk(MODEL, output, resume=False, **kwargs)


def test_projection_and_circuit_sampling(tmp_path):
    import stim
    circuit = stim.Circuit("R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    samples = sample_to_disk(str(circuit), tmp_path / "circuit", source="circuit", shots=2000, seed=1, workers=2, chunk_shots=256)
    np.testing.assert_array_equal(samples.load_detectors(), samples.load_observables())
    subset = sample_to_disk(MODEL, tmp_path / "projection", detector_indices=[1], shots=100, seed=1)
    assert subset.load_detectors().shape == (100, 1)
    assert SampleSet.open(tmp_path / "projection").num_observables == 1


def test_confidence_intervals_at_zero_and_paired_difference():
    low, high = wilson_interval(0, 1000)
    assert low == pytest.approx(0.0)
    assert 0.0 < high < 0.01
    pair = paired_error_statistics(10, 3, 1000)
    assert pair["difference"] == pytest.approx(0.007)
    assert pair["ci_low"] < pair["difference"] < pair["ci_high"]
    identical = paired_error_statistics(0, 0, 1000)
    assert identical["ci_low"] < 0 < identical["ci_high"]


def test_hyperedge_decoder_is_rejected(tmp_path):
    model = "error(0.01) D0 D1 D2 L0\n"
    samples = sample_to_disk(model, tmp_path / "samples", shots=20, seed=1)
    with pytest.raises(ValueError, match="hyperedge"):
        evaluate_models(samples, {"actual": model})
    invisible = "error(0.1) D0\nerror(0.02) L0\n"
    samples = sample_to_disk(invisible, tmp_path / "invisible", shots=20, seed=1)
    with pytest.raises(ValueError, match="logical-only"):
        evaluate_models(samples, {"actual": invisible})
