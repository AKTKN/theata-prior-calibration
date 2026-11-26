"""Command-line entry point shared by shell scripts and notebook examples."""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="THeatA surface-code prior calibration and MWPM benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, description in (("run", "Execute all stages with resume"), ("build", "Compile standard Stim circuits through THeatA"),
        ("sample", "Sample actual model for calibration and heldout evaluation"), ("fit", "Estimate priors with exact S11 and S14-S16"),
        ("evaluate", "Decode actual and estimated DEMs on paired test shots")):
        child = subparsers.add_parser(command, help=description)
        child.add_argument("--config", default="configs/demo.yaml")
        child.add_argument("--out", required=True, help="Experiment output directory; different configurations require different directories")
        child.add_argument("--workers", type=int, help="Override allocated CPU worker count")
        child.add_argument("--theata-path", help="Override the pinned upstream checkout location")
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
    args = parser.parse_args(argv)
    try:
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
        actions[args.command](config, args.out)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()

