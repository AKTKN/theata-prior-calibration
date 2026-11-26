"""Separate build, blind fit, supervised labels, and scoring for unknown supports."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import replace
import hashlib
import importlib.metadata
import multiprocessing as mp
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from typing import Callable, Iterator

import numpy as np
import stim

from . import __version__
from .candidates import DetectorMetadata
from .config import THEATA_REVISION
from .decoding import evaluate_models
from .dem import DemStructureError, GraphStructure, project_dem
from .estimation import EstimationError, Moments, compute_moments, estimate_priors
from .experiment import Case, build_models, cases, sample_models
from .io import atomic_json, atomic_text, fingerprint, read_json
from .logical_labels import LogicalParities, fit_logical_labels, prefix_logical_counts
from .sampling import SampleSet
from .structure_config import StructureExperimentConfig
from .structure_estimation import fit_structure, prefix_moments
from .support_metrics import canonical_truth, score_labels, score_priors, score_support

UNKNOWN_METHODS = ("complete_clipped", "complete_selected")
INFORMATION = {"actual_prior": "actual_support_priors_and_logical_labels",
               "known_support_estimated": "known_support_and_logical_labels"}


def _initialize(config: StructureExperimentConfig, output_dir: str | Path) -> Path:
    root = Path(output_dir).resolve()
    identity = dict(config.scientific_dict(), package_version=__version__)
    marker = root / "structure_run.json"
    digest = fingerprint(identity)
    previous = _cached(marker, digest)
    if previous is None:
        if (root / "run.json").exists():
            raise ValueError("Choose an output directory separate from the known-support experiment")
        versions = {name: importlib.metadata.version(name) for name in
                    ("stim", "pymatching", "numpy", "scipy", "matplotlib", "threadpoolctl", "lark", "networkx")}
        atomic_json(marker, dict(identity, fingerprint=digest, python=platform.python_version(),
                                 versions=versions, base_config=config.base.to_dict(),
                                 initial_resources={"workers": config.to_base_config().workers,
                                                    "case_jobs": config.case_jobs},
                                 sampling_regime="independent_dem" if config.base.sample_source == "dem"
                                 else "noisy_circuit_model_mismatch"))
    return root


def _cached(marker: Path, digest: str) -> dict | None:
    if not marker.exists():
        return None
    result = read_json(marker)
    if result["fingerprint"] != digest:
        raise ValueError(f"Cached identity differs at {marker}. Use a fresh output directory.")
    for name, expected in result.get("files", {}).items():
        if _file_hash(marker.parent / name) != expected:
            raise ValueError(f"Cached artifact changed at {marker.parent / name}")
    return result


def _file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
        scratch = Path(temporary) / path.name
        np.savez_compressed(scratch, **arrays)
        scratch.replace(path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _save_dem(path: Path, dem: stim.DetectorErrorModel) -> None:
    atomic_text(path, str(dem) + "\n")


def _report_time(root: Path, stage: str, started: float, **details: float) -> None:
    path = root / "timings.json"
    records = read_json(path) if path.exists() else {"stages": {}}
    try:
        import resource

        unit = 1 if platform.system() == "Darwin" else 1024
        peak_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
        peak_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * unit
    except ImportError:
        peak_self = peak_children = None
    current = records["stages"].get(stage, {})
    elapsed = time.perf_counter() - started
    totals = {name: current.get("cumulative_detail_seconds", {}).get(name, 0) + seconds
              for name, seconds in details.items()}
    records["stages"][stage] = {
        "last_seconds": elapsed, "last_detail_seconds": details,
        "cumulative_seconds": current.get("cumulative_seconds", 0) + elapsed,
        "cumulative_detail_seconds": totals, "peak_process_rss_bytes": peak_self,
        "peak_child_rss_bytes": peak_children,
    }
    atomic_json(path, records)


def _case_paths(config: StructureExperimentConfig, root: Path) -> Iterator[tuple[Case, Path, Path]]:
    simulation = root / "simulation"
    for case in cases(config.base, simulation):
        key = case.directory.relative_to(simulation)
        yield case, root / "blind" / key, root / "evaluation" / key


def _fit_path(folder: Path, repetition: int, shots: int) -> Path:
    return folder / f"rep{repetition:03d}_n{shots}"


def _resources(config: StructureExperimentConfig) -> dict:
    base = config.to_base_config()
    return dict(workers=base.workers, chunk_shots=base.covariance_chunk_shots,
                block_detectors=base.covariance_block_detectors)


def _run_case_jobs(config: StructureExperimentConfig, root: Path,
                   action: Callable[[StructureExperimentConfig, Case, Path], tuple[float, ...]]) -> list[tuple[float, ...]]:
    tasks = [(case, blind) for case, blind, _ in _case_paths(config, root)]
    budget = config.to_base_config().workers
    jobs = min(len(tasks), budget, config.case_jobs)
    if jobs == 1:
        return [action(config, case, blind) for case, blind in tasks]
    share = replace(config, base=replace(config.base, workers=budget // jobs))
    with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(action, share, case, blind) for case, blind in tasks]
        return [future.result() for future in futures]


def preflight_structure(config: StructureExperimentConfig) -> list[dict]:
    """Estimate numerical array sizes before invoking THeatA."""
    from .circuits import detector_sectors

    base = config.base
    result = []
    for d in base.distances:
        rounds = d * base.rounds_per_distance
        circuit = stim.Circuit.generated(f"surface_code:rotated_memory_{base.basis}", distance=d, rounds=rounds)
        m = sum(sector.lower() == base.basis for sector in detector_sectors(circuit))
        block_shots = min(base.covariance_chunk_shots, max(base.training_shots))
        n = max(base.training_shots)
        result.append(dict(distance=d, rounds=rounds, detectors=m, pairs=m * (m - 1) // 2,
                           candidates=m * (m + 1) // 2, float64_moment_bytes=8 * m * m,
                           working_array_budget_bytes=8 * (40 * m * m + 8 * block_shots * m),
                           memmap_sample_bytes_per_rate=(m + circuit.num_observables) *
                           (base.repetitions * n + base.test_shots),
                           moment_multiply_accumulates_per_max_prefix=n * m * m))
    return result


def build_structure(config: StructureExperimentConfig, output_dir: str | Path) -> None:
    root = _initialize(config, output_dir)
    started = time.perf_counter()
    atomic_json(root / "preflight.json", preflight_structure(config))
    todo = list(_case_paths(config, root))
    completed = sum((case.directory / "model.json").is_file() for case, _, _ in todo)
    if completed < len(todo):
        checkout = Path(config.base.theata_path)
        git_args = ["git", "-C", str(checkout)]
        revision = subprocess.run([*git_args, "rev-parse", "HEAD"], capture_output=True, text=True)
        if revision.returncode or revision.stdout.strip() != THEATA_REVISION:
            raise ValueError(f"Build requires the pinned THeatA checkout {THEATA_REVISION}. Run scripts/bootstrap.sh.")
        tracked_edits = subprocess.run([*git_args, "status", "--porcelain", "--untracked-files=no"],
                                       capture_output=True, text=True, check=True).stdout
        if tracked_edits.strip():
            raise ValueError("Build requires unmodified tracked files in the pinned THeatA checkout")
        if completed:
            archived = read_json(root / "simulation" / "run.json")
            versions = ("stim", "numpy", "scipy", "lark", "networkx")
            if any(archived["versions"][name] != importlib.metadata.version(name) for name in versions):
                raise ValueError("A partial model cache requires its original compilation dependencies")
    try:
        build_models(config.to_base_config(), root / "simulation")
    except DemStructureError as error:
        atomic_json(root / "build_failure.json", {"status": "unsupported_generative_model", "reason": str(error)})
        raise
    (root / "build_failure.json").unlink(missing_ok=True)
    for case, blind, _ in todo:
        meta = DetectorMetadata.from_ideal(case.directory / "stim_standard_ideal.stim", config.base.basis)
        if meta.num_observables != 1:
            raise ValueError("The structure benchmark requires one logical observable")
        path = blind / "detector_metadata.json"
        if path.exists() and read_json(path) != meta.to_dict():
            raise ValueError(f"Noiseless detector metadata changed at {path}")
        atomic_json(path, meta.to_dict())
    _report_time(root, "build", started)


def sample_structure(config: StructureExperimentConfig, output_dir: str | Path) -> None:
    root = _initialize(config, output_dir)
    started = time.perf_counter()
    sample_models(config.to_base_config(), root / "simulation")
    _report_time(root, "sampling", started)


def _training(config: StructureExperimentConfig, case: Case, repetition: int,
              metadata: DetectorMetadata, *, labels: bool = False) -> tuple[SampleSet, dict]:
    samples = SampleSet.open(case.directory / f"samples_train_{repetition:03d}", validate_observables=labels)
    sample_config = samples.metadata["config"]
    base = config.base
    stream = f"d{case.distance}/r{case.rounds}/{base.basis}/p{case.physical_error_rate.hex()}/train/{repetition}"
    expected = dict(seed=base.seed, stream=stream, source=base.sample_source, shots=max(base.training_shots),
                    chunk_shots=base.sample_chunk_shots, detector_indices=list(metadata.detector_ids),
                    stim_version=stim.__version__)
    if any(sample_config.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Calibration sample identity differs at {samples.directory}")
    identity = {"sample_config": sample_config, "detectors_sha256": _file_hash(samples.detector_path)}
    return samples, identity


def _fit_identity(config: StructureExperimentConfig, metadata: DetectorMetadata,
                  training: dict, n: int, repetition: int) -> dict:
    return dict(config=config.scientific_dict(), detector_metadata=metadata.to_dict(),
                training=training, calibration_shots=n, repetition=repetition)


def _label_identity(config: StructureExperimentConfig, detector_fit: dict, observable_hash: str) -> dict:
    return dict(detector_fit_fingerprint=detector_fit["fingerprint"],
                detector_fit_sha256=detector_fit["files"]["detector_fit.npz"],
                calibration_observables_sha256=observable_hash, label_mode=config.label_mode,
                label_policy=config.scientific_dict()["label_policy"])


def fit_structure_stage(config: StructureExperimentConfig, output_dir: str | Path) -> None:
    """Read sanitized metadata and training D only, even in the supervised regime."""
    root = _initialize(config, output_dir)
    started = time.perf_counter()
    results = _run_case_jobs(config, root, _fit_case)
    moments_time, statistics_time, inversion_time = (sum(times) for times in zip(*results, strict=True))
    _report_time(root, "detector_fit", started, moments=moments_time, support_statistics=statistics_time,
                 detector_inversion=inversion_time)


def _fit_case(config: StructureExperimentConfig, case: Case, blind: Path) -> tuple[float, ...]:
    moments_time = statistics_time = inversion_time = 0.0
    meta = DetectorMetadata.from_dict(read_json(blind / "detector_metadata.json"))
    for repetition in range(config.base.repetitions):
        samples, training = _training(config, case, repetition, meta)
        pending = {}
        for n in config.base.training_shots:
            path = _fit_path(blind, repetition, n) / "fit.json"
            identity = _fit_identity(config, meta, training, n, repetition)
            digest = fingerprint(identity)
            if _cached(path, digest) is None:
                pending[n] = (path, identity, digest)
        if not pending:
            continue
        data = samples.load_detectors()
        generator = prefix_moments(data, tuple(pending), **_resources(config))
        for n, (marker, identity, digest) in pending.items():
            moment_start = time.perf_counter()
            moments = next(generator)
            moment_seconds = time.perf_counter() - moment_start
            fitted = fit_structure(data[:n], moments=moments, alpha_pairs=config.alpha_pairs,
                                   alpha_singletons=config.alpha_singletons,
                                   probability_ceiling=config.base.probability_ceiling, **_resources(config))
            folder = marker.parent
            names = ["detector_fit.npz"]
            _save_npz(folder / names[0], **fitted.arrays)
            if fitted.valid:
                for method in UNKNOWN_METHODS:
                    graph, priors = meta.graph(fitted.arrays[f"{method}_priors"])
                    name = f"{method}_detector_only.dem"
                    _save_dem(folder / name, graph.to_dem(priors))
                    names.append(name)
            atomic_json(marker, dict(identity, fingerprint=digest, **fitted.diagnostics,
                                     moments_seconds=moment_seconds,
                                     uncompressed_array_bytes=sum(a.nbytes for a in fitted.arrays.values()),
                                     files={name: _file_hash(folder / name) for name in names}))
            moments_time += moment_seconds
            statistics_time += fitted.diagnostics["support_statistics_seconds"]
            inversion_time += fitted.diagnostics["detector_inversion_seconds"]
            print(f"structure fit: d={case.distance} p={case.physical_error_rate:g} rep={repetition} "
                  f"N={n}: {fitted.diagnostics['status']}", flush=True)
    return moments_time, statistics_time, inversion_time


def label_structure_stage(config: StructureExperimentConfig, output_dir: str | Path) -> None:
    """Attach calibration-observable labels to already fixed detector rates."""
    root = _initialize(config, output_dir)
    if config.label_mode == "detector_only":
        return
    started = time.perf_counter()
    results = _run_case_jobs(config, root, _label_case)
    count_seconds, label_seconds = (sum(times) for times in zip(*results, strict=True))
    _report_time(root, "labels", started, signed_counts=count_seconds, label_estimation=label_seconds)


def _label_case(config: StructureExperimentConfig, case: Case, blind: Path) -> tuple[float, ...]:
    count_seconds = label_seconds = 0.0
    meta = DetectorMetadata.from_dict(read_json(blind / "detector_metadata.json"))
    if meta.num_observables != 1:
        raise ValueError("Logical-label learning requires one observable")
    for repetition in range(config.base.repetitions):
        samples, training = _training(config, case, repetition, meta, labels=True)
        observable_hash = _file_hash(samples.observable_path)
        pending = {}
        for n in config.base.training_shots:
            folder = _fit_path(blind, repetition, n)
            detector_fit = _cached(folder / "fit.json", fingerprint(_fit_identity(config, meta, training, n, repetition)))
            if detector_fit is None:
                raise FileNotFoundError(folder / "fit.json")
            if detector_fit["status"] != "complete":
                continue
            identity = _label_identity(config, detector_fit, observable_hash)
            digest = fingerprint(identity)
            if _cached(folder / "labels.json", digest) is None:
                pending[n] = (folder, identity, digest)
        if not pending:
            continue
        counts = prefix_logical_counts(samples.load_detectors(), samples.load_observables(),
                                       tuple(pending), **_resources(config))
        for n, (folder, identity, digest) in pending.items():
            count_start = time.perf_counter()
            logical_count = next(counts)
            count_elapsed = time.perf_counter() - count_start
            label_start = time.perf_counter()
            fitted = _load_npz(folder / "detector_fit.npz")
            moments = Moments(n, fitted["means"], fitted["second_moments"])
            labels = fit_logical_labels(
                moments, LogicalParities.from_counts(moments, *logical_count),
                raw_pair_attenuation=fitted["raw_pair_attenuation"],
                raw_singleton_attenuation=fitted["raw_singleton_attenuation"],
            )
            names = ["logical_labels.npz"]
            label_arrays = dict(labels.arrays, logical_count=np.array(logical_count[0]),
                                detector_logical_counts=logical_count[1], pair_logical_counts=logical_count[2])
            _save_npz(folder / names[0], **label_arrays)
            if labels.valid:
                for method in UNKNOWN_METHODS:
                    graph, priors = meta.graph(fitted[f"{method}_priors"], labels.arrays["label_bits"])
                    name = f"{method}_calibration_observables.dem"
                    _save_dem(folder / name, graph.to_dem(priors))
                    names.append(name)
            label_elapsed = time.perf_counter() - label_start
            atomic_json(folder / "labels.json", dict(identity, fingerprint=digest, **labels.diagnostics,
                        signed_counts_seconds=count_elapsed, label_estimation_seconds=label_elapsed,
                        uncompressed_array_bytes=sum(a.nbytes for a in label_arrays.values()),
                        files={name: _file_hash(folder / name) for name in names}))
            count_seconds += count_elapsed
            label_seconds += label_elapsed
            print(f"structure labels: d={case.distance} p={case.physical_error_rate:g} rep={repetition} "
                  f"N={n}: {labels.diagnostics['status']}", flush=True)
    return count_seconds, label_seconds


def _write_table(path: Path, rows: list[dict], *, required: tuple[str, ...] = ()) -> None:
    import io

    buffer = io.StringIO(newline="")
    fields = list(dict.fromkeys((*required, *(key for row in rows for key in row))))
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def _method_information(method: str, label_mode: str) -> str:
    return INFORMATION.get(method, label_mode)


def _score_fit(config: StructureExperimentConfig, metadata: DetectorMetadata, reference: GraphStructure, actual_text: str,
               true_priors: np.ndarray, true_labels: np.ndarray, fit: dict, label_marker: dict | None,
               fitted_folder: Path, score_folder: Path, heldout: SampleSet, heldout_moments: Moments,
               common: dict) -> dict:
    """Score one fixed calibration prefix and compare all available decoders."""
    family = metadata.candidates
    n = common["calibration_shots"]
    arrays = _load_npz(fitted_folder / "detector_fit.npz")
    moments = Moments(n, arrays["means"], arrays["second_moments"])
    label_fit = (_load_npz(fitted_folder / "logical_labels.npz")
                 if label_marker and label_marker["status"] == "complete" else None)
    predictions = {"actual_prior": (true_priors, true_priors > 0)}
    models = {"actual_prior": actual_text}
    skips = []
    try:
        estimate = estimate_priors(reference, moments=moments, probability_floor=config.base.probability_floor,
                                   probability_ceiling=config.base.probability_ceiling, strict=config.base.strict_estimation)
    except EstimationError as error:
        skips.append(dict(common, method="known_support_estimated", stage="detector_fit", reason=str(error)))
    else:
        known_priors, _ = canonical_truth(family, estimate.structure, estimate.probabilities)
        predictions["known_support_estimated"] = known_priors, true_priors > 0
        dem = estimate.to_dem()
        _save_dem(score_folder / "known_support_estimated.dem", dem)
        models["known_support_estimated"] = str(dem)
    for method in UNKNOWN_METHODS:
        if fit["status"] != "complete":
            skips.append(dict(common, method=method, stage="detector_fit", reason=fit["status"],
                              nonpositive_marginal_ids=fit["nonpositive_marginal_ids"],
                              nonpositive_pair_parities=fit["nonpositive_pair_parities"]))
            continue
        predictions[method] = arrays[f"{method}_priors"], arrays[f"{method}_selected"]
        if label_fit is not None:
            models[method] = (fitted_folder / f"{method}_calibration_observables.dem").read_text()
        elif label_marker is not None:
            skips.append(dict(common, method=method, stage="logical_labels", reason=label_marker["status"],
                              nonpositive_logical_bias=label_marker["nonpositive_logical_bias"],
                              nonpositive_detector_logical_parities=label_marker["nonpositive_detector_logical_parities"],
                              nonpositive_pair_logical_parities=label_marker["nonpositive_pair_logical_parities"]))

    support_rows, prior_rows, label_rows = [], [], []
    comparison_arrays = {}
    for method in (*INFORMATION, *UNKNOWN_METHODS):
        prior = predictions.get(method)
        method_row = dict(common, method=method, information=_method_information(method, config.label_mode))
        selected_prior, selected = prior if prior is not None else (None, None)
        status = "complete" if prior is not None else "undefined"
        support_rows.extend(dict(method_row, status=status, **row)
                            for row in score_support(family, true_priors, selected_prior, selected))
        if prior is not None:
            graph, positive = metadata.graph(selected_prior)
            raw = arrays["raw_probabilities"] if method in UNKNOWN_METHODS else None
            refit = arrays.get(f"{method}_refit_probabilities")
            score, incident = score_priors(family, true_priors, selected_prior, graph, positive,
                                           heldout_moments, raw, refit)
            comparison_arrays[f"{method}_spurious_incident_rate"] = incident
            prior_rows.append(dict(method_row, status=status, **score))
        else:
            prior_rows.append(dict(method_row, status=status))
        if method in UNKNOWN_METHODS and config.label_mode == "calibration_observables":
            if label_fit is not None:
                label_rows.append(dict(method_row, status="complete",
                                       **score_labels(true_priors, true_labels, selected_prior, label_fit)))
            else:
                label_rows.append(dict(method_row, status="undefined"))

    _save_npz(score_folder / "comparison.npz", **comparison_arrays)
    decoded = evaluate_models(heldout, models, baseline="actual_prior", workers=config.to_base_config().workers,
                              chunk_shots=config.base.decode_chunk_shots, allow_unmatched=True)
    decode_rows, paired_rows = [], []
    for method in (*INFORMATION, *UNKNOWN_METHODS):
        method_row = dict(common, method=method, information=_method_information(method, config.label_mode))
        if method in decoded["models"]:
            decode_rows.append(dict(method_row, status="complete", **decoded["models"][method]))
        else:
            skip = next((entry for entry in skips if entry["method"] == method), None)
            reason = skip["reason"] if skip else "detector_only_without_logical_labels"
            decode_rows.append(dict(method_row, status=reason))
        if method in decoded["paired"]:
            paired_rows.append(dict(method_row, baseline="actual_prior",
                                    baseline_information=INFORMATION["actual_prior"], **decoded["paired"][method]))
    return dict(support=support_rows, prior=prior_rows, label=label_rows, decoding=decode_rows,
                paired=paired_rows, skipped=skips, decode_seconds=decoded["elapsed_seconds"])


def evaluate_structure_stage(config: StructureExperimentConfig, output_dir: str | Path) -> dict[str, list[dict]]:
    """Join truth and estimated models only here, then score shared heldout shots."""
    root = _initialize(config, output_dir)
    started = time.perf_counter()
    decode_seconds = heldout_seconds = 0.0
    tables = {name: [] for name in ("support", "prior", "label", "decoding", "paired", "skipped")}
    mode = config.label_mode
    for case, blind, evaluation in _case_paths(config, root):
        metadata = DetectorMetadata.from_dict(read_json(blind / "detector_metadata.json"))
        actual_text = (case.directory / "actual_sector.dem").read_text()
        oracle = project_dem(stim.DetectorErrorModel(actual_text))
        true_priors, true_labels = canonical_truth(metadata.candidates, oracle.structure, oracle.probabilities)
        reference = replace(
            oracle.structure, detector_ids=metadata.detector_ids,
            detector_coordinates=metadata.coordinates,
            edges=tuple(edge for edge, prior in zip(oracle.structure.edges, oracle.probabilities, strict=True)
                        if prior > 0),
        )
        _save_npz(evaluation / "truth.npz", priors=true_priors, labels=true_labels)
        heldout = SampleSet.open(case.directory / "samples_test")
        base = config.base
        source_name = "actual_full.dem" if base.sample_source == "dem" else "actual_noisy.stim"
        model_source_hash = hashlib.sha256((case.directory / source_name).read_text().encode()).hexdigest()
        stream = f"d{case.distance}/r{case.rounds}/{base.basis}/p{case.physical_error_rate.hex()}/test"
        sample_config = heldout.metadata["config"]
        expected = dict(seed=base.seed, stream=stream, source=base.sample_source, shots=base.test_shots,
                        chunk_shots=base.sample_chunk_shots, detector_indices=list(metadata.detector_ids),
                        stim_version=stim.__version__, model_sha256=model_source_hash)
        if any(sample_config.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Heldout sample identity differs at {heldout.directory}")
        test_identity = dict(sample_config=sample_config, detectors_sha256=_file_hash(heldout.detector_path),
                             observables_sha256=_file_hash(heldout.observable_path))
        heldout_moments = None
        baseline_recorded = False
        for repetition in range(base.repetitions):
            samples, training = _training(config, case, repetition, metadata, labels=mode == "calibration_observables")
            if training["sample_config"]["model_sha256"] != model_source_hash:
                raise ValueError(f"Training source differs from heldout source at {samples.directory}")
            observable_hash = _file_hash(samples.observable_path) if mode == "calibration_observables" else None
            for n in base.training_shots:
                fitted_folder = _fit_path(blind, repetition, n)
                fit_identity = _fit_identity(config, metadata, training, n, repetition)
                fit = _cached(fitted_folder / "fit.json", fingerprint(fit_identity))
                if fit is None:
                    raise FileNotFoundError(fitted_folder / "fit.json")
                labels = None
                if mode == "calibration_observables" and fit["status"] == "complete":
                    label_identity = _label_identity(config, fit, observable_hash)
                    labels = _cached(fitted_folder / "labels.json", fingerprint(label_identity))
                    if labels is None:
                        raise FileNotFoundError(fitted_folder / "labels.json")
                score_folder = _fit_path(evaluation, repetition, n)
                marker = score_folder / "score.json"
                identity = dict(detector_fit=fit["files"], logical_labels=labels["files"] if labels else None,
                                fit_fingerprint=fit["fingerprint"], heldout=test_identity,
                                actual_sector_sha256=hashlib.sha256(actual_text.encode()).hexdigest(),
                                pymatching_version=importlib.metadata.version("pymatching"))
                digest = fingerprint(identity)
                scored = _cached(marker, digest)
                if scored is None:
                    if heldout_moments is None:
                        holdout_start = time.perf_counter()
                        heldout_moments = compute_moments(heldout.load_detectors(), **_resources(config))
                        heldout_seconds += time.perf_counter() - holdout_start
                    common = dict(case=case.directory.relative_to(root / "simulation").as_posix(),
                                  distance=case.distance, rounds=case.rounds, basis=base.basis,
                                  physical_error_rate=case.physical_error_rate, repetition=repetition,
                                  calibration_shots=n, label_mode=mode,
                                  sample_source=base.sample_source, sampling_regime="independent_dem"
                                  if base.sample_source == "dem" else "noisy_circuit_model_mismatch")
                    scored = _score_fit(config, metadata, reference, actual_text, true_priors, true_labels, fit, labels,
                                        fitted_folder, score_folder, heldout, heldout_moments, common)
                    names = ["comparison.npz"]
                    if (score_folder / "known_support_estimated.dem").exists():
                        names.append("known_support_estimated.dem")
                    scored.update(identity=identity, fingerprint=digest,
                                  files={name: _file_hash(score_folder / name) for name in names})
                    atomic_json(marker, scored)
                    decode_seconds += scored["decode_seconds"]
                    print(f"structure score: d={case.distance} p={case.physical_error_rate:g} rep={repetition} "
                          f"N={n}: {scored['decode_seconds']:.2f}s decoding", flush=True)
                for name, records in tables.items():
                    for row in scored[name]:
                        if row["method"] == "actual_prior":
                            if baseline_recorded:
                                continue
                            row = dict(row, calibration_shots=0, repetition=-1)
                        records.append(row)
                baseline_recorded = True

    for name in ("support", "prior", "label", "decoding"):
        if name != "label" or mode == "calibration_observables":
            _write_table(root / f"{name}_metrics.csv", tables[name], required=("method", "status"))
    _write_table(root / "paired_comparisons.csv", tables["paired"],
                 required=("method", "baseline", "candidate_only_failures", "baseline_only_failures"))
    atomic_json(root / "metrics.json", tables)
    atomic_json(root / "skipped_conditions.json", tables["skipped"])
    _report_time(root, "evaluation", started, heldout_moments=heldout_seconds, decoding=decode_seconds)
    return tables


def plot_structure_stage(output_dir: str | Path) -> list[Path]:
    from .structure_plotting import plot_structure_results

    root = Path(output_dir).resolve()
    started = time.perf_counter()
    figures = plot_structure_results(root)
    _report_time(root, "plotting", started)
    return figures


def run_structure_experiment(config: StructureExperimentConfig, output_dir: str | Path) -> dict[str, list[dict]]:
    """Run the unknown-support experiment and report fixed-model and support metrics."""
    started = time.perf_counter()
    root = _initialize(config, output_dir)
    build_structure(config, root)
    sample_structure(config, root)
    fit_structure_stage(config, root)
    label_structure_stage(config, root)
    tables = evaluate_structure_stage(config, root)
    plot_structure_stage(root)
    _report_time(root, "total", started)
    return tables
