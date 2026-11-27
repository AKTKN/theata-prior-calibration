"""Plots from archived metrics and representative complete-family arrays."""
from __future__ import annotations

from pathlib import Path

from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullLocator
import numpy as np

from .candidates import DetectorMetadata
from .io import read_json
from .plotting import _save, _series, _style, plt, SymLogNorm
from .structure_experiment import UNKNOWN_METHODS, _fit_path, _load_npz

METHODS = {
    "actual_prior": ("Actual priors and labels", "#202c3b"),
    "known_support_estimated": ("Estimated priors, known support and labels", "#4967a3"),
    "complete_clipped": ("Complete clipped", "#bb6931"),
    "complete_selected": ("Complete selected", "#237e76"),
}


def _ler_label(method: str) -> str:
    name = METHODS[method][0]
    return f"{name}, calibration-observable labels" if method in UNKNOWN_METHODS else name


def _label_ler_ticks(ax) -> None:
    """Label percentages even when the panel spans less than one decade."""
    lower, upper = ax.get_ylim()
    upper = min(upper, 1.0)
    ax.set_ylim(lower, upper)
    for locator in (LogLocator(base=10, numticks=7), LogLocator(base=10, subs=(1, 2, 5), numticks=7),
                    MaxNLocator(nbins=5)):
        ticks = locator.tick_values(lower, upper)
        ticks = ticks[(lower <= ticks) & (ticks <= upper)]
        if 3 <= len(ticks) <= 7:
            break
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda rate, _: f"{100 * rate:.7g}%"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())


def _ler_plots(tables: dict, run: dict, output: Path) -> list[Path]:
    rows = [row for row in tables["decoding"] if row["status"] == "complete"]
    distances = run["base"]["distances"]
    rates = run["base"]["physical_error_rates"]
    sizes = run["base"]["training_shots"]
    regime = "Independent DEM sampling" if run["base"]["sample_source"] == "dem" else "Noisy circuit sampling (model mismatch)"
    footnote = ("Errors include logical mismatches and decoding failures.\n"
                "Lines average evaluated repeats. Shading spans 95% Wilson intervals. "
                "Open triangles mark zero-count upper bounds.\n"
                "Vertical scales vary by panel. Gaps indicate unavailable prior fits or logical labels.")
    methods = [method for method in METHODS if any(row["method"] == method for row in rows)]
    result = []
    for by_p, name in ((True, "ler_vs_physical_error"), (False, "ler_vs_calibration_shots")):
        if by_p:
            panels = [(d, None) for d in distances]
            shape = 1, len(distances)
        else:
            panels = [(d, p) for d in distances for p in rates]
            shape = len(distances), len(rates)
        fig, axes = plt.subplots(*shape, figsize=(5.5 * shape[1], 3.8 * shape[0] + 1.0),
                                 squeeze=False, layout="constrained")
        for ax, (d, panel_p) in zip(axes.flat, panels, strict=True):
            for method in methods:
                selected = [r for r in rows if r["distance"] == d and r["method"] == method]
                if by_p:
                    # The physical-rate sweep compares all methods at the same, largest calibration N.
                    groups = [(p, [r for r in selected if r["physical_error_rate"] == p
                                   and r["calibration_shots"] in (0, max(sizes))]) for p in rates]
                else:
                    matching_rate = [r for r in selected if r["physical_error_rate"] == panel_p]
                    groups = [(n, [r for r in matching_rate if r["calibration_shots"] in (0, n)]) for n in sizes]
                _series(ax, groups, color=METHODS[method][1], label=_ler_label(method),
                        linestyle="--" if method == "actual_prior" else "-")
            xvalues, xlabel = (rates, "Mean physical MS error probability") if by_p else (sizes, "Calibration shots")
            title = f"d={d}, N={max(sizes):,}" if by_p else f"d={d}, p={panel_p:g}"
            ax.set(xscale="log", yscale="log", xlabel=xlabel, ylabel="Error rate per memory shot", title=title)
            ax.set_xticks(xvalues, labels=[f"{x:g}" for x in xvalues])
            _label_ler_ticks(ax)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=min(2, len(methods)), frameon=False)
        fig.supxlabel(f"{regime}. {footnote}", fontsize=9)
        result += _save(fig, output / name)
    return result


def _metric_line(ax, rows: list[dict], sizes: list[int], key: str, method: str, *, style: str = "-",
                 label: str | None = None) -> None:
    values = []
    for n in sizes:
        at_n = [r[key] for r in rows if r["calibration_shots"] == n and r.get(key) is not None]
        values.append(float(np.mean(at_n)) if at_n else np.nan)
    ax.plot(sizes, values, linestyle=style, marker="o", color=METHODS[method][1],
            label=label or METHODS[method][0])


def _support_priors_labels(tables: dict, key: str, sizes: list[int], output: Path, run: dict) -> list[Path]:
    result = []
    fields = {"support": ("support_vs_calibration_shots", (2, 2)),
              "prior": ("prior_error_vs_calibration_shots", (1, 2)),
              "label": ("label_error_vs_calibration_shots", (1, 2))}
    for table, (name, shape) in fields.items():
        if table == "label" and run["label_mode"] != "calibration_observables":
            continue
        rows = [r for r in tables[table] if r["case"] == key]
        fig, axes = plt.subplots(*shape, figsize=(11, 3.65 * shape[0]), squeeze=False, layout="constrained")
        for method in UNKNOWN_METHODS:
            selected = [r for r in rows if r["method"] == method]
            if table == "support":
                for column, kind in enumerate(("pair", "singleton")):
                    group = [r for r in selected if r["support_type"] == kind]
                    for axis, keys in zip(axes[:, column], (("precision", "recall"), ("fp", "fn")), strict=True):
                        for metric, style in zip(keys, ("-", "--"), strict=True):
                            _metric_line(axis, group, sizes, metric, method, style=style,
                                         label=f"{METHODS[method][0]} {metric.upper() if metric in ('fp', 'fn') else metric}")
                    axes[0, column].set(title=f"{kind.capitalize()} support", ylim=(-0.02, 1.02))
                    axes[1, column].set(ylabel="Number of supports")
            elif table == "prior":
                for metric, label, style in (("true_support_rmse", "true supports", "-"), ("rmse", "all candidates", "--")):
                    _metric_line(axes[0, 0], selected, sizes, metric, method, style=style,
                                 label=f"{METHODS[method][0]}, {label}")
                _metric_line(axes[0, 1], selected, sizes, "false_positive_prior_mass", method)
                axes[0, 0].set(ylabel="Prior RMSE")
                axes[0, 1].set(ylabel="False-positive prior mass")
            else:
                for axis, metric, axis_label in zip(axes[0], ("error_rate", "ambiguity_rate"),
                                                    ("Label error rate", "Label ambiguity rate"), strict=True):
                    _metric_line(axis, selected, sizes, metric, method)
                    axis.set(ylabel=axis_label, ylim=(-0.02, 1.02))
        detail = "Calibration-observable labels on matched true supports. " if table == "label" else ""
        fig.suptitle(f"{detail}{key}. Means across calibration repeats.", fontsize=11)
        for axis in axes.flat:
            axis.set(xscale="log", xlabel="Calibration shots")
            axis.set_xticks(sizes, labels=[f"{n:,}" for n in sizes])
            axis.minorticks_off()
            axis.legend(frameon=False, fontsize=8)
        result += _save(fig, output / name)
    return result


def _diagnostic_figures(root: Path, key: str, n: int, repetition: int, output: Path) -> list[Path]:
    blind = root / "blind" / key
    family = DetectorMetadata.from_dict(read_json(blind / "detector_metadata.json")).candidates
    arrays = _load_npz(_fit_path(blind, repetition, n) / "detector_fit.npz")
    truth = _load_npz(root / "evaluation" / key / "truth.npz")["priors"] > 0
    decision = arrays.get("complete_selected_selected")
    active = arrays.get("complete_selected_active")
    raw = arrays["raw_p_ij"]
    finite = raw[np.isfinite(raw)]
    limit = max(1e-8, float(np.max(np.abs(finite))) if len(finite) else 0)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), layout="constrained")
    axes[0].imshow(family.matrix(truth), origin="lower", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    estimate = axes[1].imshow(raw, origin="lower", cmap="RdBu_r", interpolation="nearest",
                             norm=SymLogNorm(linthresh=limit / 100, vmin=-limit, vmax=limit))
    fig.colorbar(estimate, ax=axes[1], label="Raw pair prior")
    if decision is not None:
        axes[2].imshow(family.matrix(decision), origin="lower", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        delta = active.astype(int) - truth.astype(int)
        matrix = axes[3].imshow(family.matrix(delta), origin="lower", cmap="RdBu_r",
                                vmin=-1, vmax=1, interpolation="nearest")
        errors = fig.colorbar(matrix, ax=axes[3])
        errors.set_ticks([-1, 0, 1], labels=["Missed (FN)", "Agreement", "Extra (FP)"])
    else:
        for ax in axes[2:]:
            ax.text(0.5, 0.5, "Undefined fit", transform=ax.transAxes, ha="center")
    for ax, title in zip(axes, ("Actual support", "Raw S11", "Holm selection\n(before rate refit)",
                               "Final support errors"), strict=True):
        ax.set(title=title, xlabel="Local detector", ylabel="Local detector")
        ax.grid(False)
    fig.suptitle(f"{key}, N={n:,}, repeat {repetition}. Support diagonals indicate singletons.")
    result = _save(fig, output / "support_matrix_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for axis, span, title in zip(axes, (slice(0, family.num_pairs), slice(family.num_pairs, None)),
                                ("Pairs", "Singletons"), strict=True):
        values = arrays["raw_probabilities"][span]
        finite = np.isfinite(values)
        shown = values[finite]
        bins = np.histogram_bin_edges(shown, bins=35)
        for subset, label, color in ((truth[span], "True support", "#22796f"),
                                     (~truth[span], "Absent support", "#7357a3")):
            axis.hist(values[finite & subset], bins=bins, histtype="step", label=label, color=color)
        axis.axvline(0, color="#202c3b", linewidth=0.8)
        axis.set(title=title, xlabel="Raw prior", ylabel="Candidate count")
        axis.legend(frameon=False)
    fig.suptitle(f"{key}, N={n:,}, repeat {repetition}. Finite raw estimates.")
    return result + _save(fig, output / "raw_prior_histograms")


def plot_structure_results(output_dir: str | Path) -> list[Path]:
    """Produce sweep and diagnostic figures for a completed structure run."""
    root = Path(output_dir)
    run = read_json(root / "structure_run.json")
    tables = read_json(root / "metrics.json")
    _style()
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    result = _ler_plots(tables, run, output)
    sizes = run["base"]["training_shots"]
    representative = min(tables["decoding"], key=lambda r: (r["distance"], -r["physical_error_rate"]))
    key = representative["case"]
    result += _support_priors_labels(tables, key, sizes, output, run)
    result += _diagnostic_figures(root, key, max(sizes), 0, output)
    return result
