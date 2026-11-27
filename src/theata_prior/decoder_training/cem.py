"""Tie-aware, deterministic CEM on supervised training and validation shots."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import stim
from threadpoolctl import threadpool_limits

from ..dem import GraphStructure
from ..io import atomic_json, atomic_text, fingerprint, read_json
from ..sampling import SampleSet, shard_seed
from .adapters import decoder_metadata
from .artifacts import atomic_arrays, file_digest, model_identity, sample_identity, single_writer
from .config import CEMConfig
from .grouping import GroupMap
from .objective import Scorer, initialize_worker, score_candidate
from .weights import WeightModel

OBJECTIVE = "number of decoder failures plus any logical mismatch among successful decodes per shot"
CHECKPOINT_SCHEMA = 1


def update_proposal(draws: np.ndarray, failures: np.ndarray, mean: np.ndarray,
                    variance: np.ndarray, config: CEMConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fractionally allocate elite mass at the cutoff and smooth the Gaussian moments."""
    cutoff = int(np.partition(failures, config.elite_count - 1)[config.elite_count - 1])
    better, tied = failures < cutoff, failures == cutoff
    flat = bool(np.all(failures == cutoff))
    mass = np.zeros(len(draws))
    mass[better] = 1
    mass[tied] = (config.elite_count - int(better.sum())) / int(tied.sum())
    diagnostics = {"flat": flat, "elite_cutoff": cutoff, "elite_cutoff_ties": int(tied.sum()),
                   "elite_weights": mass / config.elite_count}
    if flat:
        return mean, variance, diagnostics
    mass /= config.elite_count
    elite_mean = mass @ draws
    elite_second = mass @ (draws ** 2)
    mix = config.smoothing
    new_mean = (1 - mix) * mean + mix * elite_mean
    second = (1 - mix) * (variance + mean ** 2) + mix * elite_second
    return new_mean, np.maximum(second - new_mean ** 2, config.min_std ** 2), diagnostics


def _validate_samples(structure: GraphStructure, train: SampleSet, validation: SampleSet) -> None:
    if train.directory == validation.directory:
        raise ValueError("training and validation must be separate sample sets")
    train_config, val_config = train.metadata["config"], validation.metadata["config"]
    for key in ("model_sha256", "source", "detector_indices", "stim_version"):
        if train_config[key] != val_config[key]:
            raise ValueError(f"training and validation sample sources disagree on {key}")
    if (train_config["seed"], train_config["stream"]) == (val_config["seed"], val_config["stream"]):
        raise ValueError("training and validation must use independent seed or stream identities")
    for samples in (train, validation):
        if (samples.num_detectors != structure.num_detectors or samples.num_observables != structure.num_observables
                or tuple(samples.metadata["config"]["detector_indices"]) != structure.detector_ids):
            raise ValueError("sample columns do not match the known detector and observable indexing")


def _checkpoint_key(checkpoint: dict) -> tuple:
    theta = np.asarray(checkpoint["theta"], dtype=float)
    return checkpoint["failures"], float(theta @ theta), checkpoint["generation"]


@single_writer
def optimize(structure: GraphStructure, train: SampleSet, validation: SampleSet, initial_priors: np.ndarray,
             groups: GroupMap, config: CEMConfig, output_dir: str | Path, *, train_shots: int,
             decoder: str, seed: int, bits_per_step: int = 1, p_min: float = 1e-9, p_max: float = 0.49,
             candidate_jobs: int = 1, native_threads: int = 1, chunk_shots: int = 2048,
             stop_after: int | None = None) -> dict:
    """Optimize with known H,B and observed D,L, using validation only for selection.

    stop_after pauses after an absolute generation number without changing the
    scientific schedule. Workers, native threads, and decode chunks may change
    when resuming.
    """
    _validate_samples(structure, train, validation)
    if not 1 <= train_shots <= train.shots or min(candidate_jobs, native_threads, chunk_shots) < 1:
        raise ValueError("invalid training prefix or execution allocation")
    if seed < 0 or (stop_after is not None and not 0 <= stop_after <= config.generations):
        raise ValueError("invalid optimizer seed or stop_after generation")
    folder = Path(output_dir)
    state_path, completion_path = folder / "state.json", folder / "training_complete.json"
    weights = WeightModel(initial_priors, groups, p_min, p_max)
    identity = {
        "schema_version": CHECKPOINT_SCHEMA, "structure": fingerprint(structure.to_dict()),
        "groups": fingerprint(asdict(groups)), "initial": model_identity(initial_priors),
        "realized_initial": model_identity(weights.initial), "train": sample_identity(train),
        "validation": sample_identity(validation), "train_shots": train_shots, "config": asdict(config),
        "p_min": p_min, "p_max": p_max, "seed": seed, "objective": OBJECTIVE,
        "decoder": decoder_metadata(decoder, bits_per_step), "numpy_version": np.__version__,
    }
    digest = fingerprint(identity)
    no_search = ("no_free_parameters" if not groups.free else "degenerate_mean_at_bound" if weights.degenerate else None)
    end = 0 if no_search else config.generations
    n_free = len(groups.free)
    initial = np.zeros(n_free)
    batch_size = min(train_shots, config.train_batch_shots or train_shots)
    limit = end if stop_after is None else min(stop_after, end)
    if state_path.exists():
        state = read_json(state_path)
        if state["fingerprint"] != digest:
            raise ValueError(f"Optimization inputs changed. Use a fresh output directory: {folder}")
    else:
        state = None
    if completion_path.exists():
        completed = read_json(completion_path)
        if (state is None or state["generation"] != end or completed["fingerprint"] != digest
                or any(file_digest(folder / name) != sha for name, sha in completed["artifacts"].items())):
            raise ValueError(f"Inconsistent completed training artifacts: {folder}")
        return completed

    args = (structure, train, train_shots, validation, weights, decoder, bits_per_step, chunk_shots)
    seen = np.zeros(train_shots, dtype=bool) if batch_size < train_shots else None
    with ExitStack() as stack:
        # Keep proposal arithmetic independent of decoder and worker allocation.
        stack.enter_context(threadpool_limits(limits=1))
        if candidate_jobs == 1:
            scorer = Scorer(*args)

            def score(tasks):
                with threadpool_limits(limits=native_threads):
                    return list(map(scorer.score, tasks))
        else:
            executor = stack.enter_context(ProcessPoolExecutor(
                max_workers=candidate_jobs, mp_context=mp.get_context("spawn"),
                initializer=initialize_worker, initargs=(native_threads, *args),
            ))

            def score(tasks):
                return list(executor.map(score_candidate, tasks))

        if state is None:
            init_score = score([(initial, None, True)])[0]
            checkpoint = {"generation": 0, "candidate_id": "initial", "theta": initial, **init_score}
            state = {"fingerprint": digest, "identity": identity, "generation": 0,
                     "mean": initial, "variance": np.full(n_free, config.initial_std ** 2),
                     "incumbent": initial, "best_validation": checkpoint,
                     "validation_history": [checkpoint],
                     "optimizer_evaluations": 0, "validation_evaluations": 1,
                     "decoder_shot_evaluations": validation.shots, "flat_generations": 0,
                     "distinct_train_shots": 0}
            atomic_json(folder / "initial.json", {
                "probabilities": weights.initial, "weights": weights.base, "target_mean_weight": weights.target,
                "input_clipped_to_p_min": weights.initial_clip_low, "input_clipped_to_p_max": weights.initial_clip_high,
                "model_sha256": model_identity(weights.initial), "all_weights_at_one_bound": weights.degenerate,
            })
            atomic_json(folder / "groups.json", asdict(groups))
            atomic_json(state_path, state)

        if seen is not None:
            for generation in range(1, state["generation"] + 1):
                saved = read_json(folder / "history" / f"generation_{generation:05d}.json")
                seen[saved["batch_indices"]] = True

        for generation in range(state["generation"] + 1, limit + 1):
            started = time.perf_counter()
            proposal_seed = shard_seed(seed, "cem-proposal", generation)
            batch_seed = shard_seed(seed, "cem-training-batch", generation)
            mean, variance = np.asarray(state["mean"]), np.asarray(state["variance"])
            draw_rng = np.random.default_rng(proposal_seed)
            draws = np.clip(draw_rng.normal(mean, np.sqrt(variance), (config.population_size, n_free)),
                            *config.theta_bounds)
            indices = None if seen is None else np.sort(np.random.default_rng(batch_seed).choice(
                train_shots, size=batch_size, replace=False))
            if seen is not None:
                seen[indices] = True

            # Keep a score per draw for elite mass, while decoding identical theta only once.
            unique: dict[tuple, int] = {}
            candidates = []
            thetas = [*draws, initial, np.asarray(state["incumbent"])]
            for index, theta in enumerate(thetas):
                role = "draw" if index < len(draws) else ("initial" if index == len(draws) else "incumbent")
                evaluation_index = unique.setdefault(tuple(theta), len(unique))
                candidates.append({"candidate_id": f"{generation}:{index}", "role": role, "theta": theta,
                                   "evaluation_index": evaluation_index})
            results = score([(np.asarray(theta), indices, False) for theta in unique])
            for candidate in candidates:
                candidate.update(results[candidate["evaluation_index"]])
            winner = min(candidates, key=lambda candidate: (
                candidate["failures"], float(candidate["theta"] @ candidate["theta"]),
                candidate["evaluation_index"],
            ))
            failures = np.array([candidate["failures"] for candidate in candidates[:len(draws)]])
            next_mean, next_variance, diagnostics = update_proposal(draws, failures, mean, variance, config)
            state["incumbent"] = winner["theta"]
            state["optimizer_evaluations"] += len(results)
            state["decoder_shot_evaluations"] += len(results) * batch_size
            state["flat_generations"] += int(diagnostics["flat"])
            validation_score = None
            if generation % config.validation_interval == 0 or generation == end:
                validation_score = {"generation": generation, "candidate_id": winner["candidate_id"],
                                    "theta": winner["theta"], **score([(winner["theta"], None, True)])[0]}
                state["validation_history"].append(validation_score)
                state["validation_evaluations"] += 1
                state["decoder_shot_evaluations"] += validation.shots
                if _checkpoint_key(validation_score) < _checkpoint_key(state["best_validation"]):
                    state["best_validation"] = validation_score
            record = {
                "generation": generation, "proposal_seed": proposal_seed, "batch_seed": batch_seed,
                "batch_id": fingerprint({"seed": batch_seed, "indices": indices, "prefix_shots": train_shots}),
                "batch_indices": indices, "batch_shots": batch_size, "candidates": candidates,
                "winner": winner["candidate_id"], "winner_failures": winner["failures"], "validation": validation_score,
                "proposal_mean": mean, "proposal_variance": variance, "next_mean": next_mean,
                "next_variance": next_variance, **diagnostics, "seconds": time.perf_counter() - started,
                "decoder_shot_evaluations": len(results) * batch_size + (validation.shots if validation_score else 0),
            }
            atomic_json(folder / "history" / f"generation_{generation:05d}.json", record)
            state.update(generation=generation, mean=next_mean, variance=next_variance,
                         distinct_train_shots=train_shots if seen is None else int(np.count_nonzero(seen)))
            atomic_json(state_path, state)
    if limit < end:
        return {"status": "paused", "generation": state["generation"], "fingerprint": digest}

    selected = weights.realize(state["best_validation"]["theta"])
    dem_path = folder / "learned.dem"
    atomic_text(dem_path, str(structure.to_dem(selected.priors)) + "\n")
    exported = stim.DetectorErrorModel.from_file(dem_path)
    error = weights.verify_serialization(structure, exported, selected)
    decoder_error = error
    if decoder == "mwpm":
        from pymatching import Matching

        graph = Matching.from_detector_error_model(exported)
        edge_weights = np.asarray([edge[2]["weight"] for edge in graph.edges()])
        decoder_error = weights.check_mean(edge_weights)
    atomic_arrays(folder / "learned.npz", theta=selected.theta, weights=selected.weights, probabilities=selected.priors)
    completed = {
        "status": no_search or "complete", "generation": state["generation"], "fingerprint": digest,
        "selected": state["best_validation"], "model_sha256": model_identity(selected.priors),
        "decoder": identity["decoder"], "mean_weight_target": weights.target, "normalization_error": error,
        "decoder_mean_weight_error": decoder_error,
        "weight_scale": selected.scale, "clipped_weight_low": selected.clipped_low,
        "clipped_weight_high": selected.clipped_high, "calibration_shots": train_shots,
        "validation_shots": validation.shots,
        **{key: state[key] for key in ("optimizer_evaluations", "validation_evaluations",
                                       "decoder_shot_evaluations", "flat_generations", "distinct_train_shots")},
        "artifacts": {name: file_digest(folder / name) for name in ("learned.dem", "learned.npz", "initial.json", "groups.json")},
    }
    atomic_json(completion_path, completed)
    return completed
