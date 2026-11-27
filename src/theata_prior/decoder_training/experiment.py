"""Disk-backed stages for grouped decoder training on known-support DEMs."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from importlib.metadata import version
import itertools
import multiprocessing as mp
from pathlib import Path
import platform
import time

import numpy as np
from scipy.stats import rankdata
import stim
from threadpoolctl import threadpool_limits

from ..dem import GraphStructure
from ..estimation import (EstimationError, SaturatedMarginalsError, compute_moments,
                          estimate_priors, pair_probability_matrix)
from ..experiment import Case, build_models, cases
from ..io import atomic_json, atomic_text, fingerprint, read_json
from ..metrics import logical_error_statistics, paired_error_statistics
from ..sampling import SampleSet, sample_to_disk, shard_seed
from .adapters import Decoder, UnsupportedPriorError, decoder_metadata, verify_union_find
from .artifacts import atomic_arrays, file_digest, model_identity, sample_identity, single_writer, write_csv
from .cem import OBJECTIVE, optimize
from .config import TrainingConfig
from .diagnostics import compare_priors, pair_errors, prior_matrices
from .grouping import GroupMap, make_groups
from .objective import decode_samples
from .weights import WeightModel

SCHEMA = 1
STAGES = ("build", "sample", "fit", "train", "evaluate", "plot")


def _cases(config: TrainingConfig, output_dir: str | Path) -> list[Case]:
    return list(cases(config.physical, Path(output_dir) / "models", rounds_by_distance=config.rounds_by_distance))


def _root(config: TrainingConfig, output_dir: str | Path) -> Path:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {"schema_version": SCHEMA, "scientific_config": config.scientific_dict(),
                "scientific_versions": {package: version(package) for package in ("numpy", "scipy", "stim")}}
    marker = root / "decoder_training.json"
    digest = fingerprint(identity)
    if marker.exists():
        if read_json(marker)["fingerprint"] != digest:
            raise ValueError(f"Scientific settings or dependencies changed. Use a fresh --out directory: {root}")
    else:
        atomic_json(marker, {**identity, "fingerprint": digest, "objective": OBJECTIVE,
                            "information_regime": "Known fixed H,B. CEM uses observed D,L for training, "
                            "validation D,L only for selection, and common independent test D,L for evaluation.",
                            "validation_budget": "Additional to calibration_shots",
                            "python": platform.python_version(), "platform": platform.platform()})
    atomic_json(root / "execution.json", {"config": asdict(config), "available_resources": asdict(config.resources)})
    return root


def _structure(case: Case) -> GraphStructure:
    return GraphStructure.from_dict(read_json(case.directory / "known_structure.json"))


def _groups(config: TrainingConfig, case: Case, structure: GraphStructure) -> GroupMap:
    ideal = case.directory / "stim_standard_ideal.stim"
    verified = False
    if config.grouping == "coarse" and ideal.is_file():
        # Standard Stim surface circuits use (x, y, round). Confirm the bridge
        # and projection preserved those exact coordinates before using them.
        coordinates = stim.Circuit.from_file(ideal).get_detector_coordinates()
        verified = structure.detector_coordinates == tuple(tuple(coordinates.get(i, ())) for i in structure.detector_ids)
    manual = config.manual_groups
    if isinstance(manual, dict):
        key = f"d{case.distance}_r{case.rounds}_{config.physical.basis}"
        if key not in manual:
            raise ValueError(f"manual groups have no entry for {key}")
        manual = manual[key]
    return make_groups(structure, config.grouping, verified_coordinates=verified, manual=manual)


def _sample_dir(case: Case, split: str, rep: int | None = None) -> Path:
    return case.directory / (f"samples_{split}" if rep is None else f"samples_{split}_{rep:03d}")


def _fit_dir(case: Case, rep: int, shots: int) -> Path:
    return case.directory / "fits" / f"rep{rep:03d}_n{shots}"


def _training_dir(folder: Path, initialization: str, seed: int, decoder: str) -> Path:
    return folder / "training" / initialization / f"seed{seed}" / decoder


def _namespace(config: TrainingConfig, case: Case) -> str:
    return f"decoder-training/d{case.distance}/r{case.rounds}/{config.physical.basis}/p{case.physical_error_rate.hex()}"


def _covariance_fingerprint(config: TrainingConfig, structure: GraphStructure, identity: dict, shots: int) -> str:
    physical = config.physical
    return fingerprint({"structure": structure.to_dict(), "training": identity, "prefix": shots,
                        "ceiling": physical.probability_ceiling, "floor": physical.probability_floor,
                        "strict": physical.strict_estimation})


def _designs(config: TrainingConfig):
    return itertools.product(config.initializations, config.optimizer_seeds)


def _budgets(config: TrainingConfig):
    return itertools.product(range(config.physical.repetitions), config.physical.training_shots)


@single_writer
def build(config: TrainingConfig, output_dir: str | Path) -> list[Case]:
    root = _root(config, output_dir)
    with threadpool_limits(limits=config.resources.native_threads):
        selected = build_models(config.physical, root / "models", rounds_by_distance=config.rounds_by_distance)
    for case in selected:
        structure = _structure(case)
        atomic_json(case.directory / "groups.json", asdict(_groups(config, case, structure)))
    return selected


@single_writer
def sample(config: TrainingConfig, output_dir: str | Path) -> list[Case]:
    root = _root(config, output_dir)
    physical = config.physical
    selected = _cases(config, root)
    for case in selected:
        metadata = read_json(case.directory / "model.json")
        source = case.directory / ("actual_full.dem" if physical.sample_source == "dem" else "actual_noisy.stim")
        model = source.read_text()
        namespace = _namespace(config, case)
        splits = [("test", None, physical.test_shots)]
        for rep in range(physical.repetitions):
            splits.extend([("train", rep, max(physical.training_shots)), ("validation", rep, config.validation_shots)])
        for split, rep, shots in splits:
            stream = f"{namespace}/{split}" if rep is None else f"{namespace}/{split}/{rep}"
            sample_to_disk(model, _sample_dir(case, split, rep), shots=shots, seed=physical.seed, stream=stream,
                           source=physical.sample_source, detector_indices=metadata["detector_indices"],
                           workers=config.resources.sampling_jobs, chunk_shots=physical.sample_chunk_shots)
        print(f"sample: d={case.distance} p={case.physical_error_rate:g}; train, validation, and test ready", flush=True)
    return selected


@single_writer
def fit(config: TrainingConfig, output_dir: str | Path) -> list[dict]:
    root = _root(config, output_dir)
    physical = config.physical
    records = []
    for case in _cases(config, root):
        structure = _structure(case)
        for rep in range(physical.repetitions):
            train = SampleSet.open(_sample_dir(case, "train", rep), validate_observables=False)
            identity = sample_identity(train, include_observables=False)
            for shots in physical.training_shots:
                folder = _fit_dir(case, rep, shots)
                marker = folder / "covariance.json"
                digest = _covariance_fingerprint(config, structure, identity, shots)
                if marker.exists():
                    record = read_json(marker)
                    if record["fingerprint"] != digest or record["arrays_sha256"] != file_digest(folder / "covariance.npz"):
                        raise ValueError(f"Covariance fit inputs or artifacts changed: {folder}")
                    records.append(record)
                    continue
                started = time.perf_counter()
                moments = compute_moments(train.load_detectors(shots), workers=config.resources.covariance_jobs,
                                          chunk_shots=physical.covariance_chunk_shots,
                                          block_detectors=physical.covariance_block_detectors)
                arrays = {"means": moments.means, "second_moments": moments.second_moments,
                          "detector_covariance": moments.covariance, "detector_ids": np.asarray(structure.detector_ids)}
                record = {"distance": case.distance, "physical_error_rate": case.physical_error_rate,
                          "calibration_shots": shots, "replicate": rep, "status": "ok", "fingerprint": digest}
                try:
                    fitted = estimate_priors(structure, moments=moments, strict=physical.strict_estimation,
                                             probability_floor=physical.probability_floor,
                                             probability_ceiling=physical.probability_ceiling)
                    raw = fitted.p_ij
                    arrays.update(probabilities=fitted.probabilities, raw_mechanisms=fitted.raw_probabilities,
                                  raw_singletons=fitted.boundary_probabilities)
                    record["estimator_diagnostics"] = fitted.diagnostics
                    atomic_text(folder / "covariance.dem", str(fitted.to_dem()) + "\n")
                except SaturatedMarginalsError as error:
                    raw = np.full((structure.num_detectors, structure.num_detectors), np.nan)
                    np.fill_diagonal(raw, 0.0)
                    record.update(status="saturated_marginals", reason=str(error),
                                  saturated_detector_ids=error.detector_ids, maximum_mean=error.maximum_mean)
                except EstimationError as error:
                    raw = pair_probability_matrix(moments)
                    record.update(status="estimation_error", reason=str(error))
                valid = np.isfinite(raw)
                np.fill_diagonal(valid, False)
                arrays.update(pair_prior_cov_raw=raw, pair_prior_cov_raw_valid=valid)
                atomic_arrays(folder / "covariance.npz", **arrays)
                upper = np.triu_indices(structure.num_detectors, 1)
                record.update(raw_finite_pairs=int(valid[upper].sum()), total_unique_pairs=len(upper[0]),
                              raw_negative_pairs=int(np.count_nonzero(raw[upper] < 0)),
                              arrays_sha256=file_digest(folder / "covariance.npz"), seconds=time.perf_counter() - started)
                atomic_json(marker, record)
                records.append(record)
                print(f"fit: d={case.distance} p={case.physical_error_rate:g} N={shots} rep={rep}; "
                      f"{record['status']}, {record['raw_finite_pairs']}/{record['total_unique_pairs']} finite raw pairs", flush=True)
    write_csv(root / "covariance_fits.csv", [{k: v for k, v in record.items() if not isinstance(v, dict)}
                                             for record in records])
    return records


def _initial(config: TrainingConfig, covariance: np.ndarray | None, requested: str, size: int) -> tuple[str, np.ndarray | None]:
    if requested == "covariance" and covariance is not None:
        return "covariance", covariance
    if requested == "uniform" or config.covariance_failure == "uniform":
        return ("uniform" if requested == "uniform" else "uniform_fallback"), np.full(size, config.uniform_probability)
    return "covariance_unavailable", None


def _load_covariance(folder: Path) -> tuple[dict, dict]:
    record = read_json(folder / "covariance.json")
    path = folder / "covariance.npz"
    if file_digest(path) != record["arrays_sha256"]:
        raise ValueError(f"Inconsistent covariance fit artifact: {path}")
    with np.load(path, allow_pickle=False) as data:
        return record, dict(data)


def _train_case(task: tuple[TrainingConfig, Case]) -> list[dict]:
    config, case = task
    structure = _structure(case)
    groups = _groups(config, case, structure)
    records = []
    detector_identities = {}
    for rep, shots in _budgets(config):
        train = SampleSet.open(_sample_dir(case, "train", rep))
        validation = SampleSet.open(_sample_dir(case, "validation", rep))
        folder = _fit_dir(case, rep, shots)
        cov_record, cov = _load_covariance(folder)
        if rep not in detector_identities:
            detector_identities[rep] = sample_identity(train, include_observables=False)
        if cov_record["fingerprint"] != _covariance_fingerprint(config, structure, detector_identities[rep], shots):
            raise ValueError(f"Covariance fit does not match these training samples: {folder}")
        for initialization, seed in _designs(config):
            init_status, initial = _initial(config, cov.get("probabilities"), initialization, structure.num_errors)
            # The same data design and optimizer seed share a batch schedule across targets and initializations.
            run_seed = shard_seed(seed, f"{_namespace(config, case)}/rep{rep}/n{shots}", 0)
            for decoder in config.target_decoders:
                target = _training_dir(folder, initialization, seed, decoder)
                print(f"train: d={case.distance} p={case.physical_error_rate:g} N={shots} rep={rep} "
                      f"{initialization} seed={seed} {decoder}", flush=True)
                common = {"distance": case.distance, "rounds": case.rounds, "physical_error_rate": case.physical_error_rate,
                          "replicate": rep, "calibration_shots": shots, "validation_shots": validation.shots,
                          "test_shots": config.physical.test_shots, "initialization": initialization,
                          "realized_initialization": init_status, "optimizer_seed": seed, "derived_seed": run_seed,
                          "target_decoder": decoder}
                if initial is None:
                    result = {"status": init_status, "reason": cov_record["reason"]}
                else:
                    try:
                        result = optimize(
                            structure, train, validation, initial, groups, config.cem, target, train_shots=shots,
                            decoder=decoder, seed=run_seed, bits_per_step=config.bits_per_step,
                            p_min=config.p_min, p_max=config.p_max, candidate_jobs=config.resources.candidate_jobs,
                            native_threads=config.resources.native_threads, chunk_shots=config.physical.decode_chunk_shots,
                        )
                    except UnsupportedPriorError as error:
                        result = {"status": "unsupported_prior", "reason": str(error)}
                summary = {**result, **common}
                atomic_json(target / "training_status.json", summary)
                records.append(summary)
    return records


@single_writer
def train(config: TrainingConfig, output_dir: str | Path) -> list[dict]:
    root = _root(config, output_dir)
    metadata = [decoder_metadata(name, config.bits_per_step) for name in config.target_decoders]
    if "uf" in config.target_decoders:
        atomic_json(root / "uf_behavior.json", verify_union_find())
    atomic_json(root / "target_decoders.json", metadata)
    tasks = [(config, case) for case in _cases(config, root)]
    if config.resources.case_jobs == 1 or len(tasks) == 1:
        results = map(_train_case, tasks)
        records = list(itertools.chain.from_iterable(results))
    else:
        with ProcessPoolExecutor(max_workers=config.resources.case_jobs, mp_context=mp.get_context("spawn")) as executor:
            records = list(itertools.chain.from_iterable(executor.map(_train_case, tasks)))
    atomic_json(root / "training.json", records)
    write_csv(root / "training.csv", [{k: v for k, v in record.items() if not isinstance(v, dict)} for record in records])
    return records


def _evaluate_prior(decoder: Decoder, prior: np.ndarray, test: tuple[np.ndarray, np.ndarray],
                    chunk_shots: int) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    started = time.perf_counter()
    try:
        predict = decoder.predictor(prior)
    except UnsupportedPriorError as error:
        return {"status": "unsupported_prior", "reason": str(error), "test_shots": len(test[0])}, None, None
    outcomes, predictions = decode_samples(predict, *test, chunk_shots, keep_predictions=True)
    failures = int(np.count_nonzero(outcomes == 2))
    mismatches = int(np.count_nonzero(outcomes == 1))
    successful = len(outcomes) - failures
    stats = logical_error_statistics(failures + mismatches, len(outcomes))
    return {**stats, "status": "ok", "test_shots": len(outcomes), "successful_decodes": successful,
            "logical_mismatches": mismatches, "decoder_failures": failures,
            "failure_probability": stats["logical_error_rate"], "conditional_denominator": successful,
            "conditional_logical_mismatch_probability": mismatches / successful if successful else None,
            "seconds": time.perf_counter() - started}, outcomes, predictions


def _load_frozen(folder: Path, test_shots: int) -> tuple[dict, np.ndarray | None]:
    training = read_json(folder / "training_status.json")
    if training["test_shots"] != test_shots:
        raise ValueError(f"Frozen model test design changed: {folder}")
    if "artifacts" not in training:
        return training, None
    if any(file_digest(folder / name) != sha for name, sha in training["artifacts"].items()):
        raise ValueError(f"Frozen training artifacts changed: {folder}")
    with np.load(folder / "learned.npz", allow_pickle=False) as data:
        return training, data["probabilities"]


def _evaluate_case(config: TrainingConfig, case: Case, root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    structure = _structure(case)
    groups = _groups(config, case, structure)
    structure_sha = fingerprint(structure.to_dict())
    actual = np.load(case.directory / "actual_priors.npy", allow_pickle=False)
    test = SampleSet.open(_sample_dir(case, "test"))
    test_identity = sample_identity(test)
    test_arrays = test.load_detectors(), test.load_observables()
    if (test.shots != config.physical.test_shots or test.num_observables != structure.num_observables
            or tuple(test.metadata["config"]["detector_indices"]) != structure.detector_ids
            or test.metadata["config"]["stream"] != f"{_namespace(config, case)}/test"
            or test.metadata["config"]["seed"] != config.physical.seed
            or test.metadata["config"]["source"] != config.physical.sample_source
            or test.metadata["config"]["chunk_shots"] != config.physical.sample_chunk_shots):
        raise ValueError("Test shots do not match the configured case")
    decoders = {name: Decoder(name, structure, config.bits_per_step) for name in config.evaluation_decoders}
    evaluation, diagnostics, artifacts = [], [], []
    actual_pair, _, pair_mask, singleton_mask = prior_matrices(structure, actual)
    for rep, shots in _budgets(config):
        folder = _fit_dir(case, rep, shots)
        cov_record, cov = _load_covariance(folder)
        raw = cov["pair_prior_cov_raw"]
        # Reuse predictions for exact duplicate priors within this paired comparison.
        cached: dict[tuple[str, str], tuple] = {}
        for initialization, seed in _designs(config):
            design = folder / "evaluation" / initialization / f"seed{seed}"
            common = {"distance": case.distance, "rounds": case.rounds, "basis": config.physical.basis,
                      "physical_error_rate": case.physical_error_rate, "calibration_shots": shots,
                      "validation_shots": config.validation_shots, "replicate": rep,
                      "optimizer_seed": seed, "initialization": initialization}
            init_status, init = _initial(config, cov.get("probabilities"), initialization, structure.num_errors)
            prior_by_model = {"actual": actual, "covariance": cov.get("probabilities"),
                              "initial": None if init is None else WeightModel(init, groups, config.p_min, config.p_max).initial}
            unavailable = {"covariance": cov_record, "initial": {"status": init_status, "reason": cov_record.get("reason")}}
            training_by_model = {}
            for target in config.target_decoders:
                trained = _training_dir(folder, initialization, seed, target)
                training, prior = _load_frozen(trained, test.shots)
                method = f"cem_{target}"
                training_by_model[method] = training
                prior_by_model[method] = prior
                if prior is None:
                    unavailable[method] = training

            matrix_data = {"detector_ids": np.asarray(structure.detector_ids), "pair_support": pair_mask,
                           "singleton_support": singleton_mask, "pair_prior_cov_raw": raw,
                           "pair_prior_cov_raw_valid": cov["pair_prior_cov_raw_valid"],
                           "detector_covariance": cov["detector_covariance"]}
            traces = {method: str(_training_dir(folder, initialization, seed, method[4:]).relative_to(root))
                      for method in training_by_model if prior_by_model[method] is not None}
            statuses = {}
            for method, prior in prior_by_model.items():
                if prior is None:
                    statuses[method] = unavailable[method]["status"]
                    for metric in ("mechanisms", "pair_mechanisms", "singleton_mechanisms", "pair_matrix_support",
                                   "pair_matrix_all", "normalized_weights"):
                        diagnostics.append({**common, "method": method, "comparison": "actual",
                                            "metric": metric, "status": statuses[method]})
                    continue
                statuses[method] = "ok"
                pair, singleton, _, _ = prior_matrices(structure, prior)
                matrix_data.update({f"mechanisms_{method}": prior, f"pair_prior_{method}": pair,
                                    f"singleton_prior_{method}": singleton,
                                    f"pair_residual_vs_actual_{method}": pair - actual_pair})
                for metric, values in compare_priors(structure, prior, actual, groups).items():
                    diagnostics.append({**common, "method": method, "comparison": "actual", "metric": metric, **values})
                training = training_by_model.get(method, {})
                diagnostics.append({**common, "method": method, "comparison": "bounds", "metric": "clipping", "status": "ok",
                                    "at_or_below_p_min": int(np.count_nonzero(prior <= config.p_min)),
                                    "at_or_above_p_max": int(np.count_nonzero(prior >= config.p_max)),
                                    "normalization_error": training.get("normalization_error"),
                                    "clipped_weight_low": training.get("clipped_weight_low"),
                                    "clipped_weight_high": training.get("clipped_weight_high")})
                if training and prior_by_model["initial"] is not None:
                    initial = prior_by_model["initial"]
                    changed = rankdata(prior, method="dense") != rankdata(initial, method="dense")
                    diagnostics.append({**common, "method": method, "comparison": "initial", "metric": "weight_order",
                                        "status": "ok", "changed_dense_ranks": int(changed.sum()), "total": len(changed)})
                if method in training_by_model:
                    for reference, matrix in (("covariance", matrix_data.get("pair_prior_covariance")),
                                              ("covariance_raw", raw)):
                        if matrix is not None:
                            difference = pair - matrix
                            matrix_data[f"pair_discrepancy_{method}_vs_{reference}"] = difference
                            for metric, values in pair_errors(difference, pair_mask, finite_only=reference.endswith("_raw")).items():
                                diagnostics.append({**common, "method": method, "comparison": reference,
                                                    "metric": metric, **values})
            statuses["covariance_raw"] = cov_record["status"]
            for metric, values in pair_errors(raw - actual_pair, pair_mask, finite_only=True).items():
                diagnostics.append({**common, "method": "covariance_raw", "comparison": "actual", "metric": metric, **values})

            matrix_path = design / "prior_matrices.npz"
            atomic_arrays(matrix_path, **matrix_data)
            for name, decoder in decoders.items():
                by_method = {}
                for method, prior in prior_by_model.items():
                    if prior is None:
                        by_method[method] = ({"status": unavailable[method]["status"], "reason": unavailable[method].get("reason"),
                                              "test_shots": test.shots}, None, None)
                        continue
                    model_sha = model_identity(prior)
                    key = name, model_sha
                    reused = key in cached
                    if not reused:
                        cached[key] = _evaluate_prior(decoder, prior, test_arrays, config.physical.decode_chunk_shots)
                    stats, failures, predictions = cached[key]
                    by_method[method] = ({**stats, "prior_sha256": model_sha, "evaluation_reused": reused}, failures, predictions)

                for method, (stats, failures, predicted) in by_method.items():
                    _, init_failures, init_predict = by_method["initial"]
                    row = {**common, **stats, "method": method, "evaluation_decoder": name,
                           "decoder_package": decoder.metadata["package"], "decoder_version": decoder.metadata["version"],
                           "decoder_mode": decoder.metadata.get("mode"), "bits_per_step": decoder.metadata.get("bits_per_step"),
                           "structure_sha256": structure_sha, "test_sha256": fingerprint(test_identity),
                           "training_status": training_by_model.get(method, {}).get("status")}
                    for key in ("optimizer_evaluations", "validation_evaluations", "decoder_shot_evaluations", "distinct_train_shots"):
                        row[key] = training_by_model.get(method, {}).get(key)
                    if failures is not None and init_failures is not None and method in training_by_model:
                        common_success = (failures != 2) & (init_failures != 2)
                        disagreement = np.any(predicted != init_predict, axis=1)
                        row["prediction_disagreements_with_initial"] = int(np.count_nonzero(disagreement & common_success))
                        row["prediction_disagreement_shots"] = int(np.count_nonzero(common_success))
                    for baseline in ("actual", "covariance"):
                        baseline_stats, baseline_failures, _ = by_method[baseline]
                        row[f"{baseline}_paired_status"] = baseline_stats["status"] if failures is not None else stats["status"]
                        if failures is not None and baseline_failures is not None:
                            n10 = int(np.count_nonzero((failures != 0) & (baseline_failures == 0)))
                            n01 = int(np.count_nonzero((failures == 0) & (baseline_failures != 0)))
                            paired = paired_error_statistics(n10, n01, test.shots)
                            for key, value in (("n10", n10), ("n01", n01), ("delta", paired["difference"]),
                                               ("paired_ci_low", paired["ci_low"]), ("paired_ci_high", paired["ci_high"])):
                                row[f"{baseline}_{key}"] = value
                    evaluation.append(row)
            artifacts.append({**common, "matrices": str(matrix_path.relative_to(root)), "traces": traces,
                              "matrices_sha256": file_digest(matrix_path),
                              "statuses": statuses, "realized_initialization": init_status,
                              "sample_identity": test_identity, "structure_sha256": structure_sha})
        cached.clear()
    return evaluation, diagnostics, artifacts


def _evaluation_fingerprint(config: TrainingConfig, selected: list[Case]) -> str:
    inputs = []
    for case in selected:
        structure = _structure(case)
        test = sample_identity(SampleSet.open(_sample_dir(case, "test")))
        files = [case.directory / "actual_priors.npy"]
        for rep, shots in _budgets(config):
            folder = _fit_dir(case, rep, shots)
            files.extend([folder / "covariance.json", folder / "covariance.npz"])
            for initialization, seed in _designs(config):
                for decoder in config.target_decoders:
                    trained = _training_dir(folder, initialization, seed, decoder)
                    status_path = trained / "training_status.json"
                    files.append(status_path)
                    files.extend(trained / name for name in read_json(status_path).get("artifacts", {}))
        inputs.append({"structure": structure.to_dict(), "groups": asdict(_groups(config, case, structure)),
                       "test": test, "files": {str(path.relative_to(case.directory)): file_digest(path) for path in files}})
    return fingerprint({"inputs": inputs, "settings": config.scientific_dict(),
                        "decoders": [decoder_metadata(name, config.bits_per_step) for name in config.evaluation_decoders]})


@single_writer
def evaluate(config: TrainingConfig, output_dir: str | Path) -> list[dict]:
    root = _root(config, output_dir)
    selected = _cases(config, root)
    digest = _evaluation_fingerprint(config, selected)
    marker = root / "evaluation_complete.json"
    if marker.exists():
        completed = read_json(marker)
        if completed.get("fingerprint") == digest and all(
            (root / name).is_file() and file_digest(root / name) == sha
            for name, sha in completed["artifacts"].items()
        ):
            index = read_json(root / "evaluation_artifacts.json")
            if all((root / row["matrices"]).is_file() and file_digest(root / row["matrices"]) == row["matrices_sha256"]
                   for row in index):
                print(f"evaluate: {completed['rows']} held-out comparisons cached", flush=True)
                return read_json(root / "evaluation.json")
    (root / "complete.json").unlink(missing_ok=True)
    marker.unlink(missing_ok=True)
    evaluation, diagnostics, artifacts = [], [], []
    with threadpool_limits(limits=config.resources.native_threads):
        for case in selected:
            rows, errors, case_artifacts = _evaluate_case(config, case, root)
            evaluation.extend(rows)
            diagnostics.extend(errors)
            artifacts.extend(case_artifacts)
            print(f"evaluate: d={case.distance} p={case.physical_error_rate:g}; {len(rows)} held-out comparisons", flush=True)
    atomic_json(root / "evaluation.json", evaluation)
    write_csv(root / "evaluation.csv", evaluation)
    atomic_json(root / "prior_diagnostics.json", diagnostics)
    write_csv(root / "prior_diagnostics.csv", diagnostics)
    atomic_json(root / "evaluation_artifacts.json", artifacts)
    atomic_json(marker, {"schema_version": SCHEMA, "rows": len(evaluation), "fingerprint": digest,
                "artifacts": {name: file_digest(root / name) for name in (
                    "evaluation.json", "evaluation.csv", "prior_diagnostics.json",
                    "prior_diagnostics.csv", "evaluation_artifacts.json")}})
    return evaluation


def preflight(config: TrainingConfig, output_dir: str | Path, *, probe_shots: int = 0) -> list[dict]:
    """Report exact built topology if present, plus an optional train-only timing probe."""
    from ..circuits import detector_sectors

    if type(probe_shots) is not int or probe_shots < 0:
        raise ValueError("probe_shots must be a nonnegative integer")
    physical, cem = config.physical, config.cem
    selected = _cases(config, output_dir)
    checkpoint_count = 1 + cem.generations // cem.validation_interval + int(cem.generations % cem.validation_interval != 0)
    training_upper = sum(cem.generations * (cem.population_size + 2) * min(n, cem.train_batch_shots or n)
                         for n in physical.training_shots)
    copies = len(config.initializations) * len(config.optimizer_seeds) * physical.repetitions
    shot_upper = copies * (training_upper + len(physical.training_shots) * checkpoint_count * config.validation_shots)
    rows = []
    for case in selected:
        if (case.directory / "known_structure.json").exists():
            structure = _structure(case)
            groups = _groups(config, case, structure)
            m, e, g = structure.num_detectors, structure.num_errors, len(groups.labels)
            input_columns = m + structure.num_observables
        else:
            if probe_shots:
                raise ValueError("Timing probes require completed build and sample stages")
            ideal = stim.Circuit.generated(f"surface_code:rotated_memory_{physical.basis}",
                                           distance=case.distance, rounds=case.rounds)
            m = sum(sector.lower() == physical.basis for sector in detector_sectors(ideal))
            e, g = None, None
            input_columns = m + ideal.num_observables
        rows.append({"distance": case.distance, "rounds": case.rounds, "physical_error_rate": case.physical_error_rate,
                     "detectors": m, "mechanisms": e, "groups": g, "free_parameters": g - 1 if g is not None else None,
                     "sample_disk_bytes_estimate": input_columns * (physical.test_shots + physical.repetitions *
                                                                   (max(physical.training_shots) + config.validation_shots)),
                     "covariance_working_bytes_estimate": 8 * (4 * m * m + min(max(physical.training_shots), physical.covariance_chunk_shots) * m),
                     "candidate_batch_bytes_per_worker_estimate": 2 * physical.decode_chunk_shots * input_columns,
                     "optimization_conditions": copies * len(physical.training_shots) * len(config.target_decoders),
                     "decoder_shot_budget_upper": shot_upper * len(config.target_decoders),
                     "heldout_decoder_shot_budget_upper": copies * len(physical.training_shots) * physical.test_shots
                     * (3 + len(config.target_decoders)) * len(config.evaluation_decoders)})
        if probe_shots:
            data = SampleSet.open(_sample_dir(case, "train", 0))
            count = min(probe_shots, data.shots)
            priors = np.full(e, np.clip(config.uniform_probability, config.p_min, config.p_max))
            for name in config.target_decoders:
                with threadpool_limits(limits=config.resources.native_threads):
                    decoder = Decoder(name, structure, config.bits_per_step)
                    started = time.perf_counter()
                    predict = decoder.predictor(priors)
                    decode_samples(predict, data.load_detectors(count), data.load_observables(count), physical.decode_chunk_shots)
                seconds = time.perf_counter() - started
                rows[-1][f"{name}_timing_probe_shots"] = count
                rows[-1][f"{name}_timing_probe_seconds"] = seconds
                rows[-1][f"{name}_serial_training_seconds_estimate"] = seconds * shot_upper / count
    return rows


@single_writer
def run(config: TrainingConfig, output_dir: str | Path, *, resume: bool = True) -> list[dict]:
    """Run every stage in order and publish completion after plotting."""
    from .plotting import plot_results

    if not resume and (Path(output_dir) / "decoder_training.json").exists():
        raise FileExistsError(f"Run already exists: {output_dir}. Use resume or a fresh directory.")
    root = _root(config, output_dir)
    build(config, root)
    sample(config, root)
    fit(config, root)
    train(config, root)
    results = evaluate(config, root)
    paths = plot_results(root)
    atomic_json(root / "complete.json", {"schema_version": SCHEMA, "evaluation_rows": len(results),
                "figures": [str(path.relative_to(root)) for path in paths]})
    return results
