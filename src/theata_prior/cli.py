"""Command-line entry point shared by shell scripts and notebook examples."""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="THeatA cooling-threshold sweep for surface-code prior calibration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, description in (("run", "Execute all stages with resume"), ("build", "Compile standard Stim circuits through THeatA"),
        ("sample", "Sample actual model for calibration and heldout evaluation"), ("fit", "Estimate priors with exact S11 and S14-S16"),
        ("evaluate", "Decode actual and estimated DEMs on paired test shots")):
        child = subparsers.add_parser(command, help=description)
        child.add_argument("--config", default="configs/lpt_calibration.yaml")
        child.add_argument("--out", required=True, help="Experiment output directory; different configurations require different directories")
        child.add_argument("--workers", type=int, help="Override allocated CPU worker count")
        child.add_argument("--theata-path", help="Override the pinned upstream checkout location")
        if command in ("run", "fit", "evaluate"):
            child.add_argument("--skip-saturated-marginals", action="store_true",
                               help="Record undefined covariance fits and evaluate the remaining decoders")
    plot = subparsers.add_parser("plot", help="Recreate figures from saved metrics, without sampling or decoding")
    plot.add_argument("--out", required=True)
    structure = subparsers.add_parser("structure", help="Learn unknown graphlike detector supports")
    stages = structure.add_subparsers(dest="structure_stage", required=True)
    for stage, description in (
        ("run", "Execute all stages with resume"), ("build", "Compile the simulation models"),
        ("sample", "Generate independent calibration and heldout samples"), ("fit", "Fit with calibration detectors only"),
        ("label", "Attach optional calibration-observable labels"), ("evaluate", "Score and evaluate matching decoders"),
        ("plot", "Plot saved structure metrics"), ("preflight", "Estimate array sizes without compiling"),
    ):
        child = stages.add_parser(stage, help=description)
        if stage != "preflight":
            child.add_argument("--out", required=True)
        if stage != "plot":
            child.add_argument("--config", default="configs/structure_smoke.yaml")
            child.add_argument("--workers", type=int)
            child.add_argument("--theata-path")
    decoder = subparsers.add_parser("decoder-training", help="Train grouped priors for MWPM and ldpc Union-find")
    decoder_stages = decoder.add_subparsers(dest="decoder_stage", required=True)
    for stage, description in (
        ("preflight", "Estimate resource use and optionally time candidate decoding"),
        ("build", "Build the known-support physical models"), ("sample", "Generate the three independent sample streams"),
        ("fit", "Fit covariance priors from training detections"), ("train", "Train and freeze decoder priors"),
        ("evaluate", "Compare frozen models on paired held-out shots"),
        ("plot", "Recreate figures from saved results"), ("run", "Run or resume all stages"),
    ):
        child = decoder_stages.add_parser(stage, help=description)
        child.add_argument("--out", default="results/decoder_training_smoke")
        if stage != "plot":
            child.add_argument("--config", default="configs/decoder_training_smoke.yaml")
            child.add_argument("--theata-path")
        if stage == "preflight":
            child.add_argument("--probe-shots", type=int, default=0)
        if stage == "run":
            child.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
        if stage == "plot":
            child.add_argument("--distance", type=int)
            child.add_argument("--physical-error-rate", type=float)
            child.add_argument("--calibration-shots", type=int)
            child.add_argument("--replicate", type=int)
            child.add_argument("--optimizer-seed", type=int)
            child.add_argument("--initialization", choices=("covariance", "uniform"))
            child.add_argument("--decoders", nargs="+", choices=("mwpm", "uf"),
                               help="Select evaluation decoders and their learned priors")
            child.add_argument("--color-normalization", choices=("linear", "log"), default="linear")
            child.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("png", "pdf"))
    thermal = subparsers.add_parser("thermal", help="Physical phonon-cap and cooling-policy benchmarks")
    thermal_stages = thermal.add_subparsers(dest="thermal_stage", required=True)
    for stage, description in (
        ("build", "Compile traces and construct condition-specific decoders"),
        ("run", "Build, sample, decode and plot the thermal benchmark"),
        ("plot", "Regenerate thermal figures from saved results"),
    ):
        child = thermal_stages.add_parser(stage, help=description)
        child.add_argument("--out", required=True)
        if stage != "plot":
            child.add_argument("--config", required=True)
            child.add_argument("--workers", type=int)
            child.add_argument("--shots", type=int)
            child.add_argument("--shard-shots", type=int)
            child.add_argument("--seed", type=int)
            child.add_argument("--theata-path")
            child.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "decoder-training":
            stage = args.decoder_stage
            if stage == "plot":
                from .decoder_training.plotting import PlotOptions, plot_results

                options = PlotOptions(args.distance, args.physical_error_rate, args.calibration_shots, args.replicate,
                                      args.optimizer_seed, args.initialization, args.color_normalization, tuple(args.formats),
                                      decoders=args.decoders)
                for path in plot_results(args.out, options):
                    print(path)
                return
            from .decoder_training import experiment
            from .decoder_training.config import load_training_config

            config = load_training_config(args.config, theata_path=args.theata_path)
            if stage == "preflight":
                print(json.dumps(experiment.preflight(config, args.out, probe_shots=args.probe_shots), indent=2))
            elif stage == "run":
                experiment.run(config, args.out, resume=args.resume)
            else:
                actions = {"build": experiment.build, "sample": experiment.sample, "fit": experiment.fit,
                           "train": experiment.train, "evaluate": experiment.evaluate}
                actions[stage](config, args.out)
            return
        if args.command == "thermal":
            from .thermal_plotting import plot_thermal

            if args.thermal_stage == "plot":
                for path in plot_thermal(args.out):
                    print(path)
                return
            from .thermal_config import load_thermal_config
            from .thermal_experiment import build_thermal, run_thermal

            config = load_thermal_config(
                args.config, workers=args.workers, shots=args.shots, shard_shots=args.shard_shots,
                seed=args.seed, theata_path=args.theata_path,
            )
            action = run_thermal if args.thermal_stage == "run" else build_thermal
            action(config, args.out, resume=args.resume)
            return
        if args.command == "structure":
            from .structure_config import load_structure_config
            from . import structure_experiment as experiment

            stage = args.structure_stage
            if stage == "plot":
                for path in experiment.plot_structure_stage(args.out):
                    print(path)
                return
            config = load_structure_config(args.config, workers=args.workers, theata_path=args.theata_path)
            if stage == "preflight":
                print(json.dumps(experiment.preflight_structure(config), indent=2))
                return
            actions = {"run": experiment.run_structure_experiment, "build": experiment.build_structure,
                       "sample": experiment.sample_structure, "fit": experiment.fit_structure_stage,
                       "label": experiment.label_structure_stage, "evaluate": experiment.evaluate_structure_stage}
            actions[stage](config, args.out)
            return
        if args.command == "plot":
            from .plotting import plot_results
            paths = plot_results(args.out)
            for path in paths:
                print(path)
            return
        config = load_config(args.config, workers=args.workers, theata_path=args.theata_path)
        from .experiment import build_models, sample_models, fit_models, evaluate_models_stage, run_experiment
        actions = {"run": run_experiment, "build": build_models, "sample": sample_models, "fit": fit_models, "evaluate": evaluate_models_stage}
        options = {"skip_saturated_marginals": True} if getattr(args, "skip_saturated_marginals", False) else {}
        actions[args.command](config, args.out, **options)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()

