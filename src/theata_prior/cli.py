"""Command-line entry point shared by shell scripts and notebook examples."""
from __future__ import annotations

import argparse
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
    args = parser.parse_args(argv)
    try:
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

