"""Composable experiment stages and resumable parameter sweeps.

``fit_models`` reads the prior-free topology and calibration detections.
Evaluation uses the heldout logical labels and compares against the actual priors.
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
from typing import TYPE_CHECKING, Iterator

import numpy as np
import stim

from . import __version__
from .config import ExperimentConfig, THEATA_REVISION
from .dem import DemStructureError, GraphStructure, project_dem
from .io import atomic_json, atomic_text, fingerprint, read_json

if TYPE_CHECKING:
    from .circuits import CompiledSurface


PHYSICAL_METRICS = (
    "cooling_operations", "cooling_duration_us", "duration_us",
    "mean_ms_phonon", "min_ms_phonon", "max_ms_phonon",
    "mean_ms_error", "min_ms_error", "max_ms_error",
    "idle_total_us", "max_idle_probability", "ms_gates", "lut_out_of_range_gates", "lut_clipped_gates",
)


def threshold_label(value: float | None) -> str:
    return "No cooling" if value is None else str(value).removesuffix(".0")


def cooling_case_name(value: float | None, *, lpt: bool = False) -> str:
    if lpt:
        return f"lpt_{threshold_label(value)}"
    return "cool_off" if value is None else f"cool_{threshold_label(value)}"


@dataclass(frozen=True)
class Case:
    directory: Path
    distance: int
    rounds: int
    physical_error_rate: float | None = None
    cool_threshold: float | None = None
    lpt: bool = False

    def condition(self) -> dict:
        if self.physical_error_rate is not None:
            return {"physical_error_rate": self.physical_error_rate}
        if self.lpt:
            return {"cool_threshold": self.cool_threshold, "liz_pt_threshold": self.cool_threshold}
        return {"cool_threshold": self.cool_threshold}

    @property
    def label(self) -> str:
        if self.physical_error_rate is not None:
            return f"d={self.distance} p={self.physical_error_rate:g}"
        axis = "LPT threshold" if self.lpt else "threshold"
        return f"d={self.distance} r={self.rounds}, {axis}={threshold_label(self.cool_threshold)}"


def cases(config: ExperimentConfig, output_dir: str | Path, *,
          rounds_by_distance: dict[int, int] | None = None) -> Iterator[Case]:
    root = Path(output_dir).resolve()
    for distance in config.distances:
        rounds = (rounds_by_distance or {}).get(distance, config.rounds or distance * config.rounds_per_distance)
        parent = root / f"d{distance}_r{rounds}_{config.basis}"
        if config.liz_pt_thresholds is not None:
            for threshold in config.liz_pt_thresholds:
                yield Case(parent / cooling_case_name(threshold, lpt=True), distance, rounds,
                           cool_threshold=threshold, lpt=True)
        elif config.cool_thresholds is not None:
            for threshold in config.cool_thresholds:
                yield Case(parent / cooling_case_name(threshold), distance, rounds, cool_threshold=threshold)
        else:
            for index, rate in enumerate(config.physical_error_rates):
                yield Case(parent / f"p{index:03d}_{rate:.8g}", distance, rounds, rate)


def _log(message: str) -> None:
    print(message, flush=True)


def initialize_run(config: ExperimentConfig, output_dir: str | Path, *,
                   rounds_by_distance: dict[int, int] | None = None) -> Path:
    if config.liz_pt_thresholds is not None and rounds_by_distance:
        raise ValueError("The LPT sweep uses the same rounds setting for every distance")
    if rounds_by_distance and any(type(d) is not int or d not in config.distances
                                  or type(r) is not int or r < 1 for d, r in rounds_by_distance.items()):
        raise ValueError("rounds_by_distance must map configured distances to positive integer rounds")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {"package_version": __version__, "scientific_config": config.scientific_dict(), "theata_reference_revision": THEATA_REVISION}
    if config.liz_pt_thresholds is not None:
        from .lpt import compiler_options

        identity["sweep"] = "paper_liz_predictive_trigger"
        identity["compiler_options"] = [compiler_options(config, t) for t in config.liz_pt_thresholds]
    elif config.cool_thresholds is not None:
        identity["sweep"] = "native_cooling_threshold"
    if rounds_by_distance:
        identity["rounds_by_distance"] = rounds_by_distance
    digest = fingerprint(identity)
    marker = root / "run.json"
    if marker.exists():
        previous = read_json(marker)
        if previous["fingerprint"] != digest:
            raise ValueError("Output directory belongs to a different experiment. Choose a new --out directory; results are never silently overwritten.")
    else:
        if ((config.cool_thresholds is not None or config.liz_pt_thresholds is not None)
                and (any(root.glob("d*/p*")) or (root / "metrics.json").exists()
                     or (config.liz_pt_thresholds is not None and any(root.glob("d*/cool_*"))))):
            raise ValueError("Output directory already contains results. Choose a new --out directory.")
        versions = {}
        for name in ("stim", "pymatching", "numpy", "scipy", "matplotlib", "threadpoolctl", "lark", "networkx"):
            versions[name] = importlib.metadata.version(name)
        atomic_json(marker, {**identity, "fingerprint": digest, "config": config.to_dict(), "versions": versions,
                             "python": platform.python_version(), "platform": platform.platform(),
                             "logical_error_definition": "probability of any logical-observable mismatch per complete memory experiment",
                             "inference_target": "known, identifiable graphlike marginal in the memory basis",
                             "physical_error_definition": "mean total nonidentity Pauli probability over native MS gate locations"})
    return root


def build_models(config: ExperimentConfig, output_dir: str | Path, *,
                 rounds_by_distance: dict[int, int] | None = None) -> list[Case]:
    """Compile each cooling policy independently and save the noisy model."""
    from .circuits import LUTRangeError, default_memory_capacity, prepare_surface

    root = initialize_run(config, output_dir, rounds_by_distance=rounds_by_distance)
    all_cases = list(cases(config, root, rounds_by_distance=rounds_by_distance))
    if config.physical_error_rates is not None:
        # The separate structure and decoder-training experiments retain their
        # existing rate sweeps, model paths, and sampling identities.
        for distance in config.distances:
            pending = [case for case in all_cases if case.distance == distance
                       and not (case.directory / "model.json").exists()]
            if not pending:
                _log(f"build: d={distance} cached")
                continue
            started = time.perf_counter()
            _log(f"build: d={distance} compiling THeatA model")
            prepared = prepare_surface(distance, pending[0].rounds, config.basis, config.theata_path, **config.hardware)
            for case in pending:
                _write_model(config, case, prepared.with_physical_error_rate(case.physical_error_rate), started)
        return all_cases

    hardware = dict(config.hardware)
    compiler_options = hardware.pop("compiler_options", {})
    if config.liz_pt_thresholds is None and hardware.get("memory_capacity") is None:
        hardware["memory_capacity"] = default_memory_capacity(2 * max(config.distances)**2 - 1)
    physical_rows = []
    for case in all_cases:
        folder = case.directory
        model_path = folder / "model.json"
        failed = folder / "model_failed.json"
        identity = _case_identity(config, case)
        if model_path.exists():
            _log(f"build: {case.label} cached")
            model = read_json(model_path)
            physical_rows.append({**identity, **_case_physics(case, model), "status": "ok",
                                  **_projection_status(model)})
            continue
        if failed.exists():
            row = read_json(failed)
            physical_rows.append(row)
            _log(f"build: {case.label}; cached rejection: {row['reason']}")
            continue
        started = time.perf_counter()
        _log(f"build: {case.label} compiling THeatA model")
        try:
            if case.lpt:
                from .lpt import compile_lpt

                actual = compile_lpt(config, case.distance, case.rounds, case.cool_threshold)
            else:
                prepared = prepare_surface(case.distance, case.rounds, config.basis, config.theata_path,
                                           compiler_options={**compiler_options, "cool_threshold_transversal": case.cool_threshold},
                                           **hardware)
                actual = prepared.with_native_noise()
        except LUTRangeError as error:
            physical = _physical_metrics(error.metadata)
            atomic_json(folder / "hardware.json", error.metadata | case.condition() | physical)
            row = {**identity, **physical, "status": "lut_out_of_range", "reason": str(error)}
            atomic_json(failed, row)
            _log(f"build: {case.label}; {error}")
        else:
            physical = _case_physics(case, actual.metadata)
            model = _write_model(config, case, actual, started, physical=physical)
            row = {**identity, **physical, "status": "ok", **_projection_status(model)}
        physical_rows.append(row)
    atomic_json(root / "physical_metrics.json", physical_rows)
    _write_csv(root / "physical_metrics.csv", physical_rows)
    return all_cases


def _physical_metrics(metadata: dict) -> dict:
    return {key: metadata.get(key) for key in PHYSICAL_METRICS}


def _case_physics(case: Case, metadata: dict) -> dict:
    physical = _physical_metrics(metadata)
    if case.lpt:
        physical["lpt_cooling_cycles"] = metadata["lpt_cooling_cycles"]
    return physical


def _projection_status(metadata: dict) -> dict:
    return {"projection_error": metadata["projection_error"]} if metadata.get("projection_error") else {}


def _case_identity(config: ExperimentConfig, case: Case) -> dict:
    return dict(distance=case.distance, rounds=case.rounds, basis=config.basis, **case.condition())


def _write_model(config: ExperimentConfig, case: Case, actual: CompiledSurface, started: float, *,
                 physical: dict | None = None) -> dict:
    from .circuits import native_noise_circuit_text

    folder = case.directory
    folder.mkdir(parents=True, exist_ok=True)
    detector_ids = tuple(i for i, sector in enumerate(actual.detector_sectors) if sector.lower() == config.basis)
    projection_error = None
    try:
        marginal = project_dem(actual.raw_dem, detector_ids)
    except DemStructureError as error:
        if not case.lpt:
            raise
        marginal = None
        projection_error = str(error)
    # Oracle files are isolated from the estimator's inputs.
    if case.lpt:
        actual.compiled_ideal_circuit.to_file(folder / "theata_reconstructed_ideal.stim")
    else:
        actual.ideal_circuit.to_file(folder / "stim_standard_ideal.stim")
        actual.compiled_ideal_circuit.to_file(folder / "theata_native_ideal.stim")
    circuit_text = (native_noise_circuit_text(actual.noisy_circuit) if case.physical_error_rate is None
                    else str(actual.noisy_circuit))
    atomic_text(folder / "actual_noisy.stim", circuit_text + "\n")
    actual.raw_dem.to_file(folder / "actual_full.dem")
    actual.decomposed_dem.to_file(folder / "actual_full_decomposed.dem")
    if marginal is not None:
        marginal.dem.to_file(folder / "actual_sector.dem")
        np.save(folder / "actual_priors.npy", marginal.probabilities, allow_pickle=False)
        atomic_json(folder / "known_structure.json", marginal.structure.to_dict())
        marginal.structure.to_dem(np.full(marginal.structure.num_errors, 0.001)).to_file(folder / "placeholder.dem")
    atomic_json(folder / "hardware.json", actual.metadata | case.condition())
    record = {**_case_identity(config, case), **(physical or {}),
        "detector_indices": detector_ids, "full_detectors": actual.raw_dem.num_detectors,
        "sector_detectors": len(detector_ids), "sector_errors": marginal.structure.num_errors if marginal else None,
        "projection": marginal.diagnostics if marginal else None, "compile_seconds": time.perf_counter() - started,
        "source_sha256": {"dem": fingerprint(str(actual.raw_dem)), "circuit": fingerprint(circuit_text)}}
    if projection_error:
        record["projection_error"] = projection_error
        _log(f"build: {case.label}; sector calibration unavailable: {projection_error}")
    else:
        _log(f"build: {case.label}; {len(detector_ids)} detectors, {record['sector_errors']} identifiable edges")
    atomic_json(folder / "model.json", record)
    return record


def _available_cases(config: ExperimentConfig, root: str | Path,
                     rounds_by_distance: dict[int, int] | None) -> Iterator[Case]:
    for case in cases(config, root, rounds_by_distance=rounds_by_distance):
        if (case.directory / "model_failed.json").is_file() and not (case.directory / "model.json").is_file():
            continue
        yield case


def sample_models(config: ExperimentConfig, output_dir: str | Path, *,
                  rounds_by_distance: dict[int, int] | None = None) -> list[Case]:
    """Generate max-N calibration samples and independent paired test samples."""
    from .sampling import sample_to_disk

    initialize_run(config, output_dir, rounds_by_distance=rounds_by_distance)
    all_cases = list(_available_cases(config, output_dir, rounds_by_distance))
    for case in all_cases:
        metadata = read_json(case.directory / "model.json")
        source = case.directory / ("actual_full.dem" if config.sample_source == "dem" else "actual_noisy.stim")
        model_text = source.read_text()
        key = case.directory.name if case.physical_error_rate is None else f"p{case.physical_error_rate.hex()}"
        namespace = f"d{case.distance}/r{case.rounds}/{config.basis}/{key}"
        common = dict(seed=config.seed, source=config.sample_source, detector_indices=metadata["detector_indices"], workers=config.workers, chunk_shots=config.sample_chunk_shots)
        if case.lpt:
            # Identical stream and shard seeds generate the same physical shots.
            sample_to_disk(model_text, case.directory / "samples_test_full", shots=config.test_shots,
                           stream=namespace + "/test", **(common | {"detector_indices": None}))
            if metadata.get("projection_error"):
                continue
        sample_to_disk(model_text, case.directory / "samples_test", shots=config.test_shots, stream=namespace + "/test", **common)
        for rep in range(config.repetitions):
            sample_to_disk(model_text, case.directory / f"samples_train_{rep:03d}", shots=max(config.training_shots), stream=namespace + f"/train/{rep}", **common)
        _log(f"sample: {case.label}; train={max(config.training_shots)} x {config.repetitions}, test={config.test_shots}")
    return all_cases


def fit_directory(case: Case, rep: int, shots: int) -> Path:
    return case.directory / "fits" / f"rep{rep:03d}_n{shots}"


def fit_models(config: ExperimentConfig, output_dir: str | Path, *, skip_saturated_marginals: bool = False,
               rounds_by_distance: dict[int, int] | None = None) -> list[Case]:
    """Infer priors from detections and structure, optionally recording undefined fits."""
    from .estimation import SaturatedMarginalsError, estimate_priors
    from .sampling import SampleSet

    initialize_run(config, output_dir, rounds_by_distance=rounds_by_distance)
    all_cases = list(_available_cases(config, output_dir, rounds_by_distance))
    for case in all_cases:
        if case.lpt and read_json(case.directory / "model.json").get("projection_error"):
            continue
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
                        _log(f"fit: {case.label} N={count} rep={rep}; {error}")
                        raise
                    atomic_json(folder / "fit_skipped.json", {
                        **_case_identity(config, case),
                        "calibration_shots": count, "repetition": rep,
                        "saturated_detector_ids": error.detector_ids,
                        "max_detector_mean": error.maximum_mean, "reason": str(error),
                    })
                    _log(f"fit: {case.label} N={count} rep={rep}; "
                         f"undefined with {len(error.detector_ids)} detector means >= 1/2")
                    continue
                fitted.save(folder)
                atomic_json(folder / "fit_complete.json", {"shots": count, "repetition": rep, "seconds": time.perf_counter() - started,
                    "structure_sha256": fingerprint(structure.to_dict()), "training_config": samples.metadata["config"],
                    "estimator": "exact S11 and S14-S16; clipped decoder priors with raw estimates retained"})
                _log(f"fit: {case.label} N={count} rep={rep}; {time.perf_counter()-started:.2f}s")
    return all_cases


def evaluate_models_stage(config: ExperimentConfig, output_dir: str | Path, *, skip_saturated_marginals: bool = False,
                          rounds_by_distance: dict[int, int] | None = None) -> list[dict]:
    """Evaluate fitted and actual priors on identical heldout detector samples."""
    root = initialize_run(config, output_dir, rounds_by_distance=rounds_by_distance)
    rows: list[dict] = []
    paired_rows: list[dict] = []
    skipped_fits: list[dict] = []
    full_rows: list[dict] = []
    for case in _available_cases(config, root, rounds_by_distance):
        identity = _case_identity(config, case)
        metadata = read_json(case.directory / "model.json")
        physical = _case_physics(case, metadata) if case.physical_error_rate is None else {}
        if case.lpt:
            full = _evaluate_cached(config, case.directory, {
                "actual": (case.directory / "actual_full_decomposed.dem").read_text()
            }, sample_dir="samples_test_full", filename="evaluation_full.json", allow_decomposed=True)
            full_metrics = full["models"]["actual"]
            full_rows.append(dict(**identity, **physical, decoder="actual_full", decoder_scope="full_detector",
                                  calibration_shots=0, repetition=-1, **full_metrics))
            _log(f"decode: {case.label}; full-detector LER={full_metrics['logical_error_rate']:.6g}")
            if metadata.get("projection_error"):
                continue
        models = {"actual": (case.directory / "actual_sector.dem").read_text()}
        labels: dict[str, tuple[int, int]] = {"actual": (0, -1)}
        fitted_folders: dict[str, Path] = {}
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
                fitted_folders[name] = folder
        evaluation = _evaluate_cached(config, case.directory, models)
        prior_errors = _write_prior_comparisons(case, fitted_folders)
        for name, metrics in evaluation["models"].items():
            count, rep = labels[name]
            row = dict(**identity, **physical, calibration_shots=count, repetition=rep,
                       decoder="actual" if name == "actual" else "estimated", **metrics)
            if case.physical_error_rate is None:
                row["prior_rmse"] = prior_errors.get(name)
            if case.lpt:
                row["decoder_scope"] = "z_sector"
            rows.append(row)
            if name != "actual":
                paired_rows.append(dict(distance=case.distance, **case.condition(),
                    calibration_shots=count, repetition=rep, **evaluation["paired"][name]))
        _log(f"decode: {case.label}; actual LER={evaluation['models']['actual']['logical_error_rate']:.6g}")
    _write_csv(root / "metrics.csv", rows)
    _write_csv(root / "paired_comparisons.csv", paired_rows)
    atomic_json(root / "metrics.json", rows)
    if skip_saturated_marginals:
        atomic_json(root / "fit_skips.json", skipped_fits)
    if config.liz_pt_thresholds is not None:
        atomic_json(root / "full_detector_metrics.json", full_rows)
        _write_csv(root / "full_detector_metrics.csv", full_rows)
        ratios = _lpt_suppression_ratios(rows + full_rows)
        atomic_json(root / "suppression_ratios.json", ratios)
        _write_csv(root / "suppression_ratios.csv", ratios)
    return rows


def _evaluate_cached(config: ExperimentConfig, directory: Path, models: dict, *,
                     sample_dir: str = "samples_test", filename: str = "evaluation.json",
                     allow_decomposed: bool = False) -> dict:
    from .decoding import evaluate_models
    from .sampling import SampleSet

    path = directory / filename
    model_hash = fingerprint(models)
    samples = SampleSet.open(directory / sample_dir)
    if path.exists():
        evaluation = read_json(path)
        if evaluation["model_hash"] != model_hash or evaluation["test_config"] != samples.metadata["config"]:
            raise ValueError(f"Evaluation inputs changed; use a fresh output directory: {directory}")
        return evaluation
    evaluation = evaluate_models(samples, models, baseline="actual", workers=config.workers,
                                 chunk_shots=config.decode_chunk_shots, allow_decomposed=allow_decomposed)
    evaluation["model_hash"] = model_hash
    evaluation["test_config"] = samples.metadata["config"]
    atomic_json(path, evaluation)
    return evaluation


def _lpt_suppression_ratios(rows: list[dict]) -> list[dict]:
    rates = {}
    for row in rows:
        key = (row["liz_pt_threshold"], row["decoder"], row["calibration_shots"], row["distance"])
        rates.setdefault(key, []).append(row["logical_error_rate"])
    ratios = []
    for (threshold, decoder, shots, distance), values in sorted(rates.items()):
        following = rates.get((threshold, decoder, shots, distance + 2))
        if following is None:
            continue
        low_distance, high_distance = float(np.mean(values)), float(np.mean(following))
        ratios.append(dict(liz_pt_threshold=threshold, decoder=decoder, calibration_shots=shots,
                           d_from=distance, d_to=distance + 2, ler_from=low_distance, ler_to=high_distance,
                           suppression_ratio=low_distance / high_distance if high_distance else None))
    return ratios


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    stream = io.StringIO(newline="")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def _write_prior_comparisons(case: Case, fitted_folders: dict[str, Path]) -> dict[str, float]:
    """Post-fit oracle diagnostics; this function never changes fitted priors."""
    actual = np.load(case.directory / "actual_priors.npy", allow_pickle=False)
    structure = GraphStructure.from_dict(read_json(case.directory / "known_structure.json"))
    rmses = {}
    for name, folder in fitted_folders.items():
        estimated = np.load(folder / "estimated_priors.npy", allow_pickle=False)
        raw = np.load(folder / "raw_estimated_priors.npy", allow_pickle=False)
        _write_csv(folder / "prior_comparison.csv", [dict(edge=index, detectors=" ".join(map(str, edge.detectors)),
            observables=" ".join(map(str, edge.observables)), actual=float(p), estimated=float(q), raw_estimated=float(r))
            for index, (edge, p, q, r) in enumerate(zip(structure.edges, actual, estimated, raw, strict=True))])
        rmse = float(np.sqrt(np.mean((estimated - actual)**2)))
        rmses[name] = rmse
        atomic_json(folder / "prior_error.json", {"mean_absolute_error": float(np.mean(np.abs(estimated-actual))),
            "root_mean_squared_error": rmse,
            "max_absolute_error": float(np.max(np.abs(estimated-actual))), "num_edges": len(actual)})
    return rmses


def run_experiment(config: ExperimentConfig, output_dir: str | Path, *, skip_saturated_marginals: bool = False,
                   rounds_by_distance: dict[int, int] | None = None) -> list[dict]:
    """Execute all stages; safe to call again with the same scientific config."""
    from .plotting import plot_results

    build_models(config, output_dir, rounds_by_distance=rounds_by_distance)
    sample_models(config, output_dir, rounds_by_distance=rounds_by_distance)
    fit_models(config, output_dir, rounds_by_distance=rounds_by_distance,
               skip_saturated_marginals=skip_saturated_marginals)
    rows = evaluate_models_stage(config, output_dir, rounds_by_distance=rounds_by_distance,
                                 skip_saturated_marginals=skip_saturated_marginals)
    plot_results(output_dir)
    return rows

