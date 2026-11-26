"""Build, sample and resume the physical thermal parameter sweeps."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import csv
from dataclasses import asdict
from functools import lru_cache
from itertools import islice
import fcntl
import hashlib
import io
import multiprocessing as mp
import os
from pathlib import Path
import platform

import numpy as np
import pymatching
import stim
from threadpoolctl import threadpool_limits

from .config import available_cpus
from .io import atomic_json, atomic_text, fingerprint, read_json
from .metrics import _clopper_pearson, logical_error_statistics
from .sampling import shard_seed
from .thermal_config import ThermalConfig, compiler_settings, hardware_environment
from .thermal_noise import color_decoder, color_pruning_audit, inject_thermal_noise, thermal_dem
from .thermal_trace import compile_trace


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _check_text(path: Path, digest: str) -> str:
    text = path.read_text()
    if _sha(text) != digest:
        raise ValueError(f"Cached file does not match its fingerprint: {path}")
    return text


def _implementation():
    source = Path(__file__).parent
    return {name: _sha((source / name).read_text()) for name in (
        "thermal_config.py", "thermal_trace.py", "thermal_noise.py", "thermal_experiment.py",
        "circuits.py", "sampling.py", "metrics.py",
    )}


@contextmanager
def _locked_output(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another thermal process is writing to {directory}") from exc
        yield directory


def _initialize(config, directory, resume):
    upstream, environment = hardware_environment(config)
    compiler = {c.label: asdict(compiler_settings(upstream, config, c)) for c in config.conditions}
    science = {
        "code": config.code, "sweep": config.sweep, "basis": config.basis, "rounds": config.rounds,
        "distances": config.distances, "conditions": [asdict(c) for c in config.conditions],
        "compiler_options": compiler, "hardware": environment,
        "implementation_sha256": _implementation(),
        "machine_architecture": platform.machine(),
    }
    run = {"science": science, "seed": config.seed, "requested_shots": config.shots, "shard_shots": config.shard_shots}
    identity = fingerprint(run)
    path = directory / "run.json"
    if path.exists():
        if not resume:
            raise FileExistsError(f"Run already exists at {directory}. Use --resume to reuse its caches")
        if read_json(path)["fingerprint"] != identity:
            raise ValueError("Incompatible thermal run configuration, revisions, versions, or code. Use a new output directory")
    elif any(p.name != ".lock" for p in directory.iterdir()):
        raise ValueError(f"Output directory {directory} already contains files without a thermal run manifest")
    else:
        atomic_json(path, {**run, "fingerprint": identity})
    return environment, science


def _worker_init():
    global _THREAD_LIMIT
    _THREAD_LIMIT = threadpool_limits(limits=1)


def _run_tasks(function, tasks, workers):
    if workers == 1:
        with threadpool_limits(limits=1):
            for task in tasks:
                function(*task)
        return
    # Set before spawn so numerical imports also see the thread budget.
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update(dict.fromkeys(names, "1"))
    tasks = iter(tasks)
    try:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_worker_init) as executor:
            pending = {executor.submit(function, *task) for task in islice(tasks, 2 * workers)}
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for task in done:
                    task.result()
                pending.update(executor.submit(function, *task) for task in islice(tasks, len(done)))
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _point_path(directory, distance, label):
    return directory / "points" / f"d{distance}" / label


def _build_group(config, distance, conditions, directory, environment, implementation):
    from .circuits import _load_theata

    upstream = _load_theata(config.theata_path)
    first = conditions[0]
    compiler = asdict(compiler_settings(upstream, config, first))
    trace_identity = fingerprint({
        "code": config.code, "d": distance, "rounds": config.rounds, "basis": config.basis,
        "hardware": {key: environment[key] for key in ("revisions", "time_parameters", "thermal_boundary")},
        "stim_version": environment["versions"]["stim"], "compiler": compiler,
        "trace_implementation": {key: implementation[key] for key in ("thermal_trace.py", "circuits.py")},
    })
    trace_dir = directory / "traces" / f"d{distance}" / first.policy
    if first.n_max is not None:
        trace_dir /= str(first.n_max)
    manifest_path = trace_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["identity"] != trace_identity:
            raise ValueError(f"Incompatible trace cache: {trace_dir}")
        trace = read_json(trace_dir / "trace.json")
        if fingerprint(trace) != manifest["trace_sha256"]:
            raise ValueError(f"Cached trace does not match its fingerprint: {trace_dir}")
        _check_text(trace_dir / "source.stim", manifest["source_sha256"])
        ideal = stim.Circuit(_check_text(trace_dir / "compiled_ideal.stim", manifest["ideal_sha256"]))
    else:
        source, ideal, trace = compile_trace(upstream, config, distance, first, environment)
        # Normalize integer dict keys before hashing the JSON representation.
        atomic_json(trace_dir / "trace.json", trace)
        trace = read_json(trace_dir / "trace.json")
        source_text, ideal_text = str(source) + "\n", str(ideal) + "\n"
        atomic_text(trace_dir / "source.stim", source_text)
        atomic_text(trace_dir / "compiled_ideal.stim", ideal_text)
        manifest = {
            "identity": trace_identity, "trace_sha256": fingerprint(trace),
            "source_sha256": _sha(source_text), "ideal_sha256": _sha(ideal_text),
        }
        atomic_json(manifest_path, manifest)
    for condition in conditions:
        point = _point_path(directory, distance, condition.label)
        model_file = point / "model.json"
        identity = fingerprint({
            "code": config.code, "sweep": config.sweep, "condition": asdict(condition),
            "d": distance, "rounds": config.rounds, "basis": config.basis,
            "environment": environment, "trace": manifest, "implementation": implementation,
        })
        if model_file.exists():
            model = read_json(model_file)
            if model["point_id"] != identity:
                raise ValueError(f"Incompatible thermal model cache: {point}")
            _check_text(point / "noisy.stim", model["circuit_sha256"])
            _check_text(point / "decoder.dem", model["dem_sha256"])
            continue
        noisy, noise = inject_thermal_noise(upstream, ideal, trace, condition.cap, environment)
        dem = thermal_dem(noisy, config.code)
        noisy_text, dem_text = str(noisy) + "\n", str(dem) + "\n"
        decoder_id = "PyMatching MWPM"
        sector = "all detectors of upstream reconstructed Z memory"
        pruning = None
        if config.code == "color":
            decoder = color_decoder(noisy, dem, distance, config.rounds, config.basis)
            pruning = color_pruning_audit(decoder, dem)
            if pruning["discarded_any_color"]:
                atomic_json(point / "rejected_decoder_pruning.json", {
                    **pruning, "dem_sha256": _sha(dem_text), "point_id": identity,
                })
                atomic_text(point / "decoder.dem", dem_text)
                raise ValueError(f"Color decomposition discarded mechanisms. See {point / 'rejected_decoder_pruning.json'}")
            decoder_id = "color-code-stim concatenated MWPM"
            sector = f"{config.basis} memory, exclude_non_essential_pauli_detectors=True"
        else:
            pymatching.Matching.from_detector_error_model(dem)
        atomic_text(point / "noisy.stim", noisy_text)
        atomic_text(point / "decoder.dem", dem_text)
        atomic_json(point / "model.json", {
            "point_id": identity, "code": config.code, "sweep": config.sweep, "distance": distance,
            "rounds": config.rounds, "basis": config.basis, "condition": asdict(condition),
            "decoder": decoder_id, "detector_sector": sector, "detectors": noisy.num_detectors,
            "observables": noisy.num_observables, "source_schedule": trace["source_schedule"],
            "noise_convention": trace["noise_convention"],
            "decoder_dem_approximation": "Stim independent-event DEM with approximate_disjoint_errors=True",
            "sampling": "physical noisy Stim circuit",
            "color_pruning": pruning, "hardware": environment, "compiler_options": trace["compiler_options"],
            "source_circuit_sha256": manifest["source_sha256"], "ideal_sha256": manifest["ideal_sha256"],
            "trace_sha256": manifest["trace_sha256"], "trace_path": str(trace_dir.relative_to(directory)),
            "circuit_sha256": _sha(noisy_text), "dem_sha256": _sha(dem_text),
            "noise": noise, "cooling": trace["cooling"], "operation_counts": trace["operation_counts"],
            "schedule_duration_us": trace["schedule_duration_us"],
        })
    print(f"Built {config.code} d={distance} {trace_dir.relative_to(directory)}", flush=True)


def _build(config, directory, environment, science):
    tasks = ((config, distance, conditions, directory, environment, science["implementation_sha256"])
             for distance in config.distances
             for conditions in ([config.conditions] if config.sweep == "cap" else [(c,) for c in config.conditions]))
    task_count = len(config.distances) * (1 if config.sweep == "cap" else len(config.conditions))
    _run_tasks(_build_group, tasks, min(config.workers, available_cpus(), task_count))


def build_thermal(config: ThermalConfig, out, *, resume=False):
    with _locked_output(out) as directory:
        _build(config, directory, *_initialize(config, directory, resume))


@lru_cache(maxsize=1)
def _shot_model(path, point_id):
    model = read_json(path / "model.json")
    if model["point_id"] != point_id:
        raise ValueError(f"Unexpected point identity at {path}")
    noisy = stim.Circuit(_check_text(path / "noisy.stim", model["circuit_sha256"]))
    dem = stim.DetectorErrorModel(_check_text(path / "decoder.dem", model["dem_sha256"]))
    if model["code"] == "surface":
        matching = pymatching.Matching.from_detector_error_model(dem)
        return noisy, matching.decode_batch
    decoder = color_decoder(noisy, dem, model["distance"], model["rounds"], model["basis"])
    return noisy, decoder.decode


def _expected_shard(seed, point_id, shard, shots):
    return {"shard": shard, "seed": shard_seed(seed, point_id, shard), "shots": shots, "point_id": point_id}


def _shard_path(path, shard):
    return path / "shards" / f"{shard:06d}.json"


def _read_shard(path, expected):
    saved = read_json(path)
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Incompatible shot shard: {path}")
    failures = saved["failures"]
    if type(failures) is not int or not 0 <= failures <= expected["shots"]:
        raise ValueError(f"Invalid logical failure count: {path}")
    return failures


def _decode_shards(path, point_id, tasks):
    noisy, decode = _shot_model(path, point_id)
    for task in tasks:
        sampler = noisy.compile_detector_sampler(seed=task["seed"])
        detectors, observables = sampler.sample(task["shots"], separate_observables=True)
        predictions = decode(detectors)
        # The pinned color decoder squeezes the single-observable axis.
        if predictions.ndim == 1 and observables.shape[1] == 1:
            predictions = predictions[:, None]
        if predictions.shape != observables.shape:
            raise RuntimeError(f"Decoder shape {predictions.shape} differs from observables {observables.shape}")
        failures = int(np.count_nonzero(np.any(predictions != observables, axis=1)))
        atomic_json(_shard_path(path, task["shard"]), {**task, "failures": failures})


def _shards(config, point_id):
    for index, start in enumerate(range(0, config.shots, config.shard_shots)):
        yield _expected_shard(config.seed, point_id, index, min(config.shard_shots, config.shots - start))


def _points(config, directory):
    for distance in config.distances:
        for condition in config.conditions:
            path = _point_path(directory, distance, condition.label)
            yield path, read_json(path / "model.json")


def _pending_shards(config, directory):
    for path, model in _points(config, directory):
        pending = []
        for task in _shards(config, model["point_id"]):
            target = _shard_path(path, task["shard"])
            if target.exists():
                _read_shard(target, task)
            else:
                pending.append(task)
                if len(pending) == 2:
                    yield path, model["point_id"], pending
                    pending = []
        if pending:
            yield path, model["point_id"], pending


def _ratio_rows(rows):
    pairs = {(row["condition"], row["distance"]): row for row in rows}
    result = []
    for a in rows:
        b = pairs.get((a["condition"], a["distance"] + 2))
        if b is None:
            continue
        low_a, high_a = _clopper_pearson(a["failures"], a["shots"], 0.975)
        low_b, high_b = _clopper_pearson(b["failures"], b["shots"], 0.975)
        low = low_a / high_b
        high = high_a / low_b if low_b else None
        result.append({
            "condition": a["condition"], "cap": a["cap"], "n_max": a["n_max"],
            "d_from": a["distance"], "d_to": b["distance"],
            "lambda": a["logical_error_rate"] / b["logical_error_rate"] if b["failures"] else None,
            "zero_denominator_failures": b["failures"] == 0,
            "ci_low": low, "ci_high": high, "confidence": 0.95,
            "ci_method": "Bonferroni ratio of two 97.5% Clopper-Pearson intervals",
            "verdict": "above_1" if low > 1 else ("below_1" if high is not None and high < 1 else "unresolved"),
        })
    return result


def _crossings(ratios):
    by_pair = {}
    for row in ratios:
        if row["cap"] is not None:
            by_pair.setdefault((row["d_from"], row["d_to"]), []).append(row)
    result = []
    for (a, b), points in by_pair.items():
        for left, right in zip(points, points[1:]):
            if {left["verdict"], right["verdict"]} == {"above_1", "below_1"}:
                result.append({"d_from": a, "d_to": b, "sampled_cap_interval": [left["cap"], right["cap"]],
                               "endpoint_verdicts": [left["verdict"], right["verdict"]],
                               "method": "opposite pointwise confidence verdicts; no interpolation"})
    return result


def _csv(path, rows):
    if not rows:
        return
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, handle.getvalue())


def _summarize(config, directory):
    rows = []
    for path, model in _points(config, directory):
        shards = list(_shards(config, model["point_id"]))
        failures = sum(_read_shard(_shard_path(path, task["shard"]), task) for task in shards)
        noise, cooling, condition = (model[key] for key in ("noise", "cooling", "condition"))
        pruning = model["color_pruning"]
        rows.append({
            "code": model["code"], "distance": model["distance"], "rounds": model["rounds"], "basis": model["basis"],
            "condition": condition["label"], "policy": condition["policy"], "cap": condition["cap"],
            "n_max": condition["n_max"], "requested_shots": config.shots, "completed_shards": len(shards),
            **logical_error_statistics(failures, config.shots),
            "mean_ms_error": noise["mean_ms_error"], "phonon_raw_mean": noise["phonon_raw"]["mean"],
            "phonon_used_mean": noise["phonon_used"]["mean"], "phonon_lut_mean": noise["phonon_lut_selected"]["mean"],
            "raw_clip_below": noise["lut_clipping_raw"]["below"], "raw_clip_above": noise["lut_clipping_raw"]["above"],
            "used_clip_below": noise["lut_clipping_used"]["below"], "used_clip_above": noise["lut_clipping_used"]["above"],
            "policy_status": cooling["policy_status"], "eligible_checkpoints": cooling["eligible_checkpoints"],
            "trigger_activations": cooling["trigger_activations"], "mpt_iterations": cooling["completed_loop_iterations"],
            "transport_episodes": cooling["transport_episodes"], "memory_visits": cooling["memory_visits"],
            "cool_ops": cooling["cool_ops"], "cooling_duration_us": cooling["cooling_duration_us"],
            "schedule_duration_us": model["schedule_duration_us"],
            "decoder": model["decoder"], "detector_sector": model["detector_sector"],
            "pruned_any_color": pruning["discarded_any_color"] if pruning else None,
            "pruned_all_colors": pruning["discarded_all_colors"] if pruning else None,
            "pruned_probability_mass": pruning["discarded_any_color_probability_mass"] if pruning else None,
            "point_id": model["point_id"], "trace_sha256": model["trace_sha256"],
            "circuit_sha256": model["circuit_sha256"], "dem_sha256": model["dem_sha256"],
            "model_path": str((path / "model.json").relative_to(directory)),
        })
    ratios = _ratio_rows(rows)
    atomic_json(directory / "metrics.json", {"rows": rows, "ratios": ratios, "sampled_crossing_brackets": _crossings(ratios)})
    _csv(directory / "metrics.csv", rows)
    _csv(directory / "ratios.csv", ratios)
    return rows


def run_thermal(config: ThermalConfig, out, *, resume=False):
    from .thermal_plotting import plot_thermal

    with _locked_output(out) as directory:
        _build(config, directory, *_initialize(config, directory, resume))
        shards_per_point = (config.shots + config.shard_shots - 1) // config.shard_shots
        total_tasks = len(config.distances) * len(config.conditions) * ((shards_per_point + 1) // 2)
        _run_tasks(_decode_shards, _pending_shards(config, directory), min(config.workers, available_cpus(), total_tasks))
        rows = _summarize(config, directory)
        plot_thermal(directory)
        return rows
