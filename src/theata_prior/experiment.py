"""Composable experiment stages and resumable parameter sweeps.

Only ``build_models`` handles oracle priors. ``fit_models`` reads the prior-free
topology and calibration detections; logical labels and true priors are used
only later, by evaluation and reporting.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import importlib.metadata
import io
from pathlib import Path
import platform
import subprocess
import time
from typing import Iterator

import numpy as np
import stim

from . import __version__
from .config import ExperimentConfig, THEATA_REVISION
from .dem import GraphStructure, project_dem
from .io import atomic_json, atomic_text, fingerprint, read_json


@dataclass(frozen=True)
class Case:
    directory: Path
    distance: int
    rounds: int
    physical_error_rate: float


def cases(config: ExperimentConfig, output_dir: str | Path) -> Iterator[Case]:
    root = Path(output_dir).resolve()
    for distance in config.distances:
        rounds = distance * config.rounds_per_distance
        for index, rate in enumerate(config.physical_error_rates):
            yield Case(root / f"d{distance}_r{rounds}_{config.basis}" / f"p{index:03d}_{rate:.8g}", distance, rounds, rate)


def _log(message: str) -> None:
    print(message, flush=True)


def initialize_run(config: ExperimentConfig, output_dir: str | Path) -> Path:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {"package_version": __version__, "scientific_config": config.scientific_dict(), "theata_reference_revision": THEATA_REVISION}
    digest = fingerprint(identity)
    marker = root / "run.json"
    if marker.exists():
        previous = read_json(marker)
        if previous["fingerprint"] != digest:
            raise ValueError("Output directory belongs to a different experiment. Choose a new --out directory; results are never silently overwritten.")
    else:
        versions = {}
        for name in ("stim", "pymatching", "numpy", "scipy", "matplotlib", "threadpoolctl", "lark", "networkx"):
            versions[name] = importlib.metadata.version(name)
        atomic_json(marker, {**identity, "fingerprint": digest, "config": config.to_dict(), "versions": versions,
                             "python": platform.python_version(), "platform": platform.platform(),
                             "logical_error_definition": "probability of any logical-observable mismatch per complete memory experiment",
                             "inference_target": "known, identifiable graphlike marginal in the memory basis",
                             "physical_error_definition": "mean total nonidentity Pauli probability over native MS gate locations"})
    return root


def build_models(config: ExperimentConfig, output_dir: str | Path) -> list[Case]:
    """Compile each distance once; generate actual models for all physical p."""
    from .circuits import prepare_surface

    root = initialize_run(config, output_dir)
    all_cases = list(cases(config, root))
    for distance in config.distances:
        selected = [case for case in all_cases if case.distance == distance]
        if all((case.directory / "model.json").exists() for case in selected):
            _log(f"build: d={distance} cached")
            continue
        started = time.perf_counter()
        _log(f"build: d={distance} compiling THeatA model")
        prepared = prepare_surface(distance, distance * config.rounds_per_distance, config.basis, config.theata_path, **config.hardware)
        for case in selected:
            folder = case.directory
            if (folder / "model.json").exists():
                continue
            folder.mkdir(parents=True, exist_ok=True)
            actual = prepared.with_physical_error_rate(case.physical_error_rate)
            detector_ids = tuple(i for i, sector in enumerate(actual.detector_sectors) if sector.lower() == config.basis)
            marginal = project_dem(actual.raw_dem, detector_ids)
            # Oracle files are isolated from the estimator's inputs.
            actual.ideal_circuit.to_file(folder / "stim_standard_ideal.stim")
            actual.compiled_ideal_circuit.to_file(folder / "theata_native_ideal.stim")
            actual.noisy_circuit.to_file(folder / "actual_noisy.stim")
            actual.raw_dem.to_file(folder / "actual_full.dem")
            actual.decomposed_dem.to_file(folder / "actual_full_decomposed.dem")
            marginal.dem.to_file(folder / "actual_sector.dem")
            np.save(folder / "actual_priors.npy", marginal.probabilities, allow_pickle=False)
            atomic_json(folder / "known_structure.json", marginal.structure.to_dict())
            marginal.structure.to_dem(np.full(marginal.structure.num_errors, 0.001)).to_file(folder / "placeholder.dem")
            atomic_json(folder / "hardware.json", actual.metadata)
            atomic_json(folder / "model.json", {"distance": distance, "rounds": case.rounds, "basis": config.basis,
                "physical_error_rate": case.physical_error_rate, "detector_indices": detector_ids,
                "full_detectors": actual.raw_dem.num_detectors, "sector_detectors": len(detector_ids),
                "sector_errors": marginal.structure.num_errors, "projection": marginal.diagnostics,
                "compile_seconds": time.perf_counter() - started,
                "source_sha256": {"dem": fingerprint(str(actual.raw_dem)), "circuit": fingerprint(str(actual.noisy_circuit))}})
            _log(f"build: d={distance} p={case.physical_error_rate:g}; {len(detector_ids)} detectors, {marginal.structure.num_errors} identifiable edges")
    return all_cases


def sample_models(config: ExperimentConfig, output_dir: str | Path) -> list[Case]:
    """Generate max-N calibration samples and independent paired test samples."""
    from .sampling import sample_to_disk

    initialize_run(config, output_dir)
    all_cases = list(cases(config, output_dir))
    for case in all_cases:
        metadata = read_json(case.directory / "model.json")
        source = case.directory / ("actual_full.dem" if config.sample_source == "dem" else "actual_noisy.stim")
        model_text = source.read_text()
        namespace = f"d{case.distance}/r{case.rounds}/{config.basis}/p{case.physical_error_rate.hex()}"
        common = dict(seed=config.seed, source=config.sample_source, detector_indices=metadata["detector_indices"], workers=config.workers, chunk_shots=config.sample_chunk_shots)
        sample_to_disk(model_text, case.directory / "samples_test", shots=config.test_shots, stream=namespace + "/test", **common)
        for rep in range(config.repetitions):
            sample_to_disk(model_text, case.directory / f"samples_train_{rep:03d}", shots=max(config.training_shots), stream=namespace + f"/train/{rep}", **common)
        _log(f"sample: d={case.distance} p={case.physical_error_rate:g}; train={max(config.training_shots)} x {config.repetitions}, test={config.test_shots}")
    return all_cases


def fit_directory(case: Case, rep: int, shots: int) -> Path:
    return case.directory / "fits" / f"rep{rep:03d}_n{shots}"


def fit_models(config: ExperimentConfig, output_dir: str | Path, *, skip_saturated_marginals: bool = False) -> list[Case]:
    """Infer priors from detections and structure, optionally recording undefined fits."""
    from .estimation import SaturatedMarginalsError, estimate_priors
    from .sampling import SampleSet

    initialize_run(config, output_dir)
    all_cases = list(cases(config, output_dir))
    for case in all_cases:
        structure = GraphStructure.from_dict(read_json(case.directory / "known_structure.json"))
        for rep in range(config.repetitions):
            samples = SampleSet.open(case.directory / f"samples_train_{rep:03d}", validate_observables=False)
            for count in config.training_shots:
                folder = fit_directory(case, rep, count)
                if (folder / "fit_complete.json").exists():
                    previous = read_json(folder / "fit_complete.json")
                    if (previous["structure_sha256"] != fingerprint(structure.to_dict())
                            or previous["training_config"] != samples.metadata["config"]):
                        raise ValueError(f"Fit inputs changed; use a fresh output directory: {folder}")
                    continue
                started = time.perf_counter()
                try:
                    fitted = estimate_priors(structure, samples.load_detectors(count), workers=config.workers,
                        chunk_shots=config.covariance_chunk_shots, block_detectors=config.covariance_block_detectors,
                        probability_floor=config.probability_floor, probability_ceiling=config.probability_ceiling,
                        strict=config.strict_estimation)
                except SaturatedMarginalsError as error:
                    if not skip_saturated_marginals or config.strict_estimation:
                        raise
                    atomic_json(folder / "fit_skipped.json", {
                        "distance": case.distance, "physical_error_rate": case.physical_error_rate,
                        "calibration_shots": count, "repetition": rep,
                        "saturated_detector_ids": error.detector_ids,
                        "max_detector_mean": error.maximum_mean, "reason": str(error),
                    })
                    _log(f"fit: d={case.distance} p={case.physical_error_rate:g} N={count} rep={rep}; "
                         f"undefined with {len(error.detector_ids)} detector means >= 1/2")
                    continue
                fitted.save(folder)
                atomic_json(folder / "fit_complete.json", {"shots": count, "repetition": rep, "seconds": time.perf_counter() - started,
                    "structure_sha256": fingerprint(structure.to_dict()), "training_config": samples.metadata["config"],
                    "estimator": "exact S11 and S14-S16; clipped decoder priors with raw estimates retained"})
                _log(f"fit: d={case.distance} p={case.physical_error_rate:g} N={count} rep={rep}; {time.perf_counter()-started:.2f}s")
    return all_cases


def evaluate_models_stage(config: ExperimentConfig, output_dir: str | Path, *, skip_saturated_marginals: bool = False) -> list[dict]:
    """Evaluate fitted and actual priors on identical heldout detector samples."""
    from .decoding import evaluate_models
    from .sampling import SampleSet

    root = initialize_run(config, output_dir)
    rows: list[dict] = []
    paired_rows: list[dict] = []
    skipped_fits: list[dict] = []
    for case in cases(config, root):
        models = {"actual": (case.directory / "actual_sector.dem").read_text()}
        labels: dict[str, tuple[int, int]] = {"actual": (0, -1)}
        fitted_folders: list[Path] = []
        for rep in range(config.repetitions):
            for count in config.training_shots:
                folder = fit_directory(case, rep, count)
                skipped = folder / "fit_skipped.json"
                if skip_saturated_marginals and skipped.is_file() and not (folder / "fit_complete.json").is_file():
                    skipped_fits.append(read_json(skipped))
                    continue
                name = f"estimated_n{count}_rep{rep}"
                models[name] = (folder / "estimated.dem").read_text()
                labels[name] = (count, rep)
                fitted_folders.append(folder)
        evaluation_path = case.directory / "evaluation.json"
        model_hash = fingerprint(models)
        samples = SampleSet.open(case.directory / "samples_test")
        if evaluation_path.exists():
            evaluation = read_json(evaluation_path)
            if evaluation["model_hash"] != model_hash or evaluation["test_config"] != samples.metadata["config"]:
                raise ValueError(f"Evaluation inputs changed; use a fresh output directory: {case.directory}")
        else:
            evaluation = evaluate_models(samples, models, baseline="actual", workers=config.workers, chunk_shots=config.decode_chunk_shots)
            evaluation["model_hash"] = model_hash
            evaluation["test_config"] = samples.metadata["config"]
            atomic_json(evaluation_path, evaluation)
        for name, metrics in evaluation["models"].items():
            count, rep = labels[name]
            row = dict(distance=case.distance, rounds=case.rounds, basis=config.basis,
                       physical_error_rate=case.physical_error_rate, calibration_shots=count, repetition=rep,
                       decoder="actual" if name == "actual" else "estimated", **metrics)
            rows.append(row)
            if name != "actual":
                paired_rows.append(dict(distance=case.distance, physical_error_rate=case.physical_error_rate,
                    calibration_shots=count, repetition=rep, **evaluation["paired"][name]))
        _write_prior_comparisons(case, fitted_folders)
        _log(f"decode: d={case.distance} p={case.physical_error_rate:g}; actual LER={evaluation['models']['actual']['logical_error_rate']:.6g}")
    _write_csv(root / "metrics.csv", rows)
    _write_csv(root / "paired_comparisons.csv", paired_rows)
    atomic_json(root / "metrics.json", rows)
    if skip_saturated_marginals:
        atomic_json(root / "fit_skips.json", skipped_fits)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    stream = io.StringIO(newline="")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def _write_prior_comparisons(case: Case, fitted_folders: list[Path]) -> None:
    """Post-fit oracle diagnostics; this function never changes fitted priors."""
    actual = np.load(case.directory / "actual_priors.npy", allow_pickle=False)
    structure = GraphStructure.from_dict(read_json(case.directory / "known_structure.json"))
    for folder in fitted_folders:
        estimated = np.load(folder / "estimated_priors.npy", allow_pickle=False)
        raw = np.load(folder / "raw_estimated_priors.npy", allow_pickle=False)
        _write_csv(folder / "prior_comparison.csv", [dict(edge=index, detectors=" ".join(map(str, edge.detectors)),
            observables=" ".join(map(str, edge.observables)), actual=float(p), estimated=float(q), raw_estimated=float(r))
            for index, (edge, p, q, r) in enumerate(zip(structure.edges, actual, estimated, raw, strict=True))])
        atomic_json(folder / "prior_error.json", {"mean_absolute_error": float(np.mean(np.abs(estimated-actual))),
            "root_mean_squared_error": float(np.sqrt(np.mean((estimated-actual)**2))),
            "max_absolute_error": float(np.max(np.abs(estimated-actual))), "num_edges": len(actual)})


def run_experiment(config: ExperimentConfig, output_dir: str | Path) -> list[dict]:
    """Execute all stages; safe to call again with the same scientific config."""
    from .plotting import plot_results

    build_models(config, output_dir)
    sample_models(config, output_dir)
    fit_models(config, output_dir)
    rows = evaluate_models_stage(config, output_dir)
    plot_results(output_dir)
    return rows

