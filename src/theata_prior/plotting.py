"""Publication-friendly plots with explicit finite-shot uncertainty."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import NullLocator
import numpy as np

from .dem import GraphStructure
from .io import read_json


def load_metrics(output_dir: str | Path) -> list[dict]:
    return read_json(Path(output_dir) / "metrics.json")


def _style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18, "figure.dpi": 120,
        "savefig.dpi": 180, "pdf.fonttype": 42, "ps.fonttype": 42})


def _series(ax, groups: list[tuple[float, list[dict]]], *, color, label: str, linestyle: str = "-", marker: str = "o") -> None:
    """Show repeat means; uncertainty envelope does not pool reused test shots."""
    x = np.asarray([value for value, _ in groups])
    y = np.asarray([np.mean([r["logical_error_rate"] for r in rows]) if rows else np.nan for _, rows in groups])
    low = np.asarray([min((r["ci_low"] for r in rows), default=np.nan) for _, rows in groups])
    high = np.asarray([max((r["ci_high"] for r in rows), default=np.nan) for _, rows in groups])
    positive = y > 0
    zero = y == 0
    ax.plot(x, np.where(positive, y, np.nan), color=color, marker=marker, linestyle=linestyle, linewidth=1.7, markersize=4, label=label)
    ax.fill_between(x, np.where(low > 0, low, np.nan), high, color=color, alpha=0.10, linewidth=0)
    if np.any(zero):
        # A zero count is displayed at its Wilson upper bound, not at an
        # invented nonzero point estimate. CSV retains exactly zero.
        ax.scatter(x[zero], high[zero], marker="v", facecolors="none", edgecolors=color, s=35)


def _save(fig, target: Path) -> list[Path]:
    paths = [target.with_suffix(".png"), target.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_results(output_dir: str | Path) -> list[Path]:
    """Create LER-vs-p and LER-vs-calibration-N from stored evaluation only."""
    root = Path(output_dir)
    rows = load_metrics(root)
    if not rows:
        raise ValueError("No evaluation results")
    _style()
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    distances = sorted({r["distance"] for r in rows})
    counts = sorted({r["calibration_shots"] for r in rows if r["decoder"] == "estimated"})
    rates = sorted({r["physical_error_rate"] for r in rows})
    colors = plt.colormaps["viridis"](np.linspace(0.15, 0.85, max(len(counts), len(rates))))
    repeats = max(r["repetition"] for r in rows) + 1
    uncertainty = "Shading: 95% Wilson intervals" if repeats == 1 else "Lines: calibration-repeat means; shading: envelope of per-repeat 95% Wilson intervals"
    footnote = uncertainty + ". Open downward triangles: zero-count upper bounds."
    skips_path = root / "fit_skips.json"
    if skips_path.is_file() and read_json(skips_path):
        footnote += " Missing points indicate undefined fits."
    result: list[Path] = []

    fig, axes = plt.subplots(1, len(distances), figsize=(6 * len(distances), 4.5), squeeze=False, layout="constrained")
    for ax, distance in zip(axes[0], distances, strict=True):
        selected = [r for r in rows if r["distance"] == distance]
        actual = [(p, [r for r in selected if r["physical_error_rate"] == p and r["decoder"] == "actual"]) for p in rates]
        _series(ax, actual, color="#202c3b", label="Actual prior (same sector)", linestyle="--", marker="s")
        for index, count in enumerate(counts):
            groups = [(p, [r for r in selected if r["physical_error_rate"] == p and r["calibration_shots"] == count]) for p in rates]
            _series(ax, groups, color=colors[index], label=f"Estimated, N={count:,}")
        ax.set(xscale="log", yscale="log", xlabel="Mean physical MS error probability, p",
               ylabel="Logical error probability per memory experiment",
               title=f"Surface code d={distance}, rounds={selected[0]['rounds']}, {selected[0]['basis'].upper()} memory")
        ax.set_xticks(rates, labels=[f"{p:g}" for p in rates])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("MWPM with actual and covariance-estimated priors", fontsize=13)
    fig.supxlabel(footnote, fontsize=8)
    result += _save(fig, output / "ler_vs_physical_error")

    fig, axes = plt.subplots(1, len(distances), figsize=(6 * len(distances), 5.2), squeeze=False, layout="constrained")
    for ax, distance in zip(axes[0], distances, strict=True):
        selected = [r for r in rows if r["distance"] == distance]
        for index, p in enumerate(rates):
            group = [r for r in selected if r["physical_error_rate"] == p]
            groups = [(n, [r for r in group if r["calibration_shots"] == n]) for n in counts]
            _series(ax, groups, color=colors[index], label=f"Estimated, p={p:g}")
            actual, = [r for r in group if r["decoder"] == "actual"]
            if actual["logical_error_rate"] > 0:
                ax.axhline(actual["logical_error_rate"], color=colors[index], linestyle="--", linewidth=1.3)
            if actual["ci_low"] > 0:
                ax.axhspan(actual["ci_low"], actual["ci_high"], color=colors[index], alpha=0.05, linewidth=0)
        ax.set(xscale="log", yscale="log", xlabel="Calibration detector samples, N",
               ylabel="Logical error probability per memory experiment", title=f"Surface code d={distance}; fixed heldout test data")
        ax.plot([], [], color="gray", linestyle="--", label="Actual prior at matching p")
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2)
    fig.suptitle("Calibration sample size and decoding performance", fontsize=13)
    fig.supxlabel(footnote, fontsize=8)
    result += _save(fig, output / "ler_vs_calibration_shots")

    # Choose the largest available p and N, then the earliest repetition.
    examples = {}
    for row in sorted(rows, key=lambda r: (r["physical_error_rate"], r["calibration_shots"], -r["repetition"])):
        if row["decoder"] == "estimated":
            examples[row["distance"]] = row
    for model_file in sorted(root.glob("d*/p*/model.json")):
        metadata = read_json(model_file)
        example = examples.get(metadata["distance"])
        if example is None or metadata["physical_error_rate"] != example["physical_error_rate"]:
            continue
        folder = model_file.parent
        fit = folder / "fits" / f"rep{example['repetition']:03d}_n{example['calibration_shots']}"
        result += plot_prior_diagnostics(folder, fit, output / f"priors_d{metadata['distance']}")
    return result


def plot_prior_diagnostics(case_dir: str | Path, fit_dir: str | Path, target: str | Path) -> list[Path]:
    """Plot the raw p_ij matrix and edge prior recovery for an existing fit."""
    _style()
    case_dir, fit_dir, target = Path(case_dir), Path(fit_dir), Path(target)
    structure = GraphStructure.from_dict(read_json(case_dir / "known_structure.json"))
    actual = np.load(case_dir / "actual_priors.npy", allow_pickle=False)
    fitted = np.load(fit_dir / "estimated_priors.npy", allow_pickle=False)
    matrix = np.load(fit_dir / "p_ij.npy", allow_pickle=False)
    oracle = np.zeros_like(matrix)
    for edge, p in zip(structure.edges, actual, strict=True):
        if len(edge.detectors) == 2:
            i, j = edge.detectors
            oracle[i, j] = oracle[j, i] = p
    limit = max(float(np.nanmax(np.abs(matrix))), float(np.max(oracle)), 1e-6)
    norm = SymLogNorm(linthresh=max(limit/100, 1e-6), vmin=-limit, vmax=limit, base=10)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for ax, image, title in zip(axes[:2], (oracle, matrix), ("Actual pair-event probabilities", "Raw S11 estimates (diagonal = 0)"), strict=True):
        im = ax.imshow(image, origin="lower", cmap="RdBu_r", norm=norm, interpolation="nearest")
        ax.set(title=title, xlabel="Local detector j", ylabel="Local detector i")
        ax.grid(False)
    fig.colorbar(im, ax=list(axes[:2]), label="p_ij", shrink=0.8)
    boundary = np.array([len(edge.detectors) == 1 for edge in structure.edges])
    for mask, label, color in ((~boundary, "Pair edge (S11)", "#257c97"), (boundary, "Boundary edge (S14-S16)", "#ca7440")):
        axes[2].scatter(actual[mask], fitted[mask], s=15, alpha=0.7, color=color, label=label)
    values = np.r_[actual[actual > 0], fitted[fitted > 0]]
    low, high = max(float(values.min()) * 0.7, 1e-12), float(values.max()) * 1.3
    axes[2].plot([low, high], [low, high], linestyle="--", color="gray", linewidth=1)
    axes[2].set(xscale="log", yscale="log", xlim=(low, high), ylim=(low, high), xlabel="Actual edge prior", ylabel="Estimated decoder prior", title="Prior recovery; fixed clipping is visible")
    axes[2].legend(frameon=False, fontsize=8)
    metadata = read_json(case_dir / "model.json")
    count = read_json(fit_dir / "fit_complete.json")["shots"]
    fig.suptitle(f"d={metadata['distance']}, p={metadata['physical_error_rate']:g}, N={count:,}", fontsize=12)
    target.parent.mkdir(parents=True, exist_ok=True)
    return _save(fig, target)

