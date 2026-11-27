"""Publication-friendly plots with explicit finite-shot uncertainty."""
from __future__ import annotations

from math import expm1, log1p
from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import NullLocator
import numpy as np

from .dem import GraphStructure
from .experiment import cooling_case_name, threshold_label
from .io import read_json


def load_metrics(output_dir: str | Path) -> list[dict]:
    return read_json(Path(output_dir) / "metrics.json")


def _round_probability(probability: float, rounds: int) -> float:
    """Convert shot failure probability to an equivalent per-round probability."""
    if type(rounds) is not int or rounds < 1:
        raise ValueError("Syndrome extraction rounds must be a positive integer")
    if not 0 <= probability <= 1:
        raise ValueError("Shot failure probability must be in [0, 1]")
    if rounds == 1 or probability in (0, 1):
        return probability
    return -expm1(log1p(-probability) / rounds)


def _per_round_rows(rows: list[dict]) -> list[dict]:
    converted = []
    for row in rows:
        if row.get("rate_unit", "per_shot") != "per_shot":
            raise ValueError("Per-round conversion requires per-shot metrics")
        values = {key: _round_probability(row[key], row["rounds"])
                  for key in ("logical_error_rate", "ci_low", "ci_high")}
        converted.append({**row, **values, "rate_unit": "per_round"})
    return converted


def _style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18, "figure.dpi": 120,
        "savefig.dpi": 180, "pdf.fonttype": 42, "ps.fonttype": 42})


def _series(ax, groups: list[tuple[float | str, list[dict]]], *, color, label: str,
            linestyle: str = "-", marker: str = "o", intervals: bool = False) -> None:
    """Show repeat means; uncertainty envelope does not pool reused test shots."""
    x = np.asarray([value for value, _ in groups])
    y = np.asarray([np.mean([r["logical_error_rate"] for r in rows]) if rows else np.nan for _, rows in groups])
    low = np.asarray([min((r["ci_low"] for r in rows), default=np.nan) for _, rows in groups])
    high = np.asarray([max((r["ci_high"] for r in rows), default=np.nan) for _, rows in groups])
    positive = y > 0
    zero = y == 0
    ax.plot(x, np.where(positive, y, np.nan), color=color, marker=marker, linestyle=linestyle, linewidth=1.7, markersize=4, label=label)
    if intervals:
        bounded = low > 0
        ax.vlines(x[bounded], low[bounded], high[bounded], color=color, alpha=0.35, linewidth=1)
    else:
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


def plot_results(output_dir: str | Path, *, split_ler: bool = False, per_round: bool = False,
                 figure_dir: str | Path | None = None, comparison_shots: int | None = None,
                 include_no_cooling: bool = True) -> list[Path]:
    """Plot saved results and optionally compare distances at one calibration size."""
    root = Path(output_dir)
    rows = load_metrics(root)
    cooling = (root / "physical_metrics.json").is_file()
    if not rows and not cooling:
        raise ValueError("No evaluation results")
    if comparison_shots is not None and not cooling:
        raise ValueError("The fixed calibration-size comparison requires a cooling-threshold sweep")
    ylabel = "Logical error probability per memory experiment"
    interval_label = "95% Wilson intervals"
    if per_round and (root / "full_detector_metrics.json").is_file():
        raise ValueError("LPT figures use the total logical error probability over the configured rounds")
    if per_round:
        rows = _per_round_rows(rows)
        ylabel = "Logical error rate per syndrome extraction round"
        interval_label = "transformed 95% Wilson intervals"
    _style()
    output = root / "figures" if figure_dir is None else Path(figure_dir)
    output.mkdir(parents=True, exist_ok=True)
    if cooling:
        return _plot_cooling(root, output, rows, ylabel, interval_label, split_ler,
                             comparison_shots, include_no_cooling)
    distances = sorted({r["distance"] for r in rows})
    counts = sorted({r["calibration_shots"] for r in rows if r["decoder"] == "estimated"})
    rates = sorted({r["physical_error_rate"] for r in rows})
    colors = plt.colormaps["viridis"](np.linspace(0.15, 0.85, max(len(counts), len(rates))))
    repeats = max(r["repetition"] for r in rows) + 1
    if repeats == 1:
        uncertainty = f"Shading: {interval_label}"
    else:
        uncertainty = f"Lines: calibration-repeat means; shading: envelope of per-repeat {interval_label}"
    footnote = uncertainty + ". Open downward triangles: zero-count upper bounds."
    skips_path = root / "fit_skips.json"
    if skips_path.is_file() and read_json(skips_path):
        footnote += " Missing points indicate undefined fits."
    result: list[Path] = []

    panels = [[distance] for distance in distances] if split_ler else [distances]
    caption = fill(footnote, width=82) if split_ler else footnote
    for panel in panels:
        suffix = f"_d{panel[0]}" if split_ler else ""
        fig, axes = plt.subplots(1, len(panel), figsize=(6 * len(panel), 4.5), squeeze=False, layout="constrained")
        for ax, distance in zip(axes[0], panel, strict=True):
            selected = [r for r in rows if r["distance"] == distance]
            actual = [(p, [r for r in selected if r["physical_error_rate"] == p and r["decoder"] == "actual"]) for p in rates]
            _series(ax, actual, color="#202c3b", label="Actual prior (same sector)", linestyle="--", marker="s")
            for index, count in enumerate(counts):
                groups = [(p, [r for r in selected if r["physical_error_rate"] == p and r["calibration_shots"] == count]) for p in rates]
                _series(ax, groups, color=colors[index], label=f"Estimated, N={count:,}")
            ax.set(xscale="log", yscale="log", xlabel="Mean physical MS error probability, p",
                   ylabel=ylabel,
                   title=f"Surface code d={distance}, rounds={selected[0]['rounds']}, {selected[0]['basis'].upper()} memory")
            ax.set_xticks(rates, labels=[f"{p:g}" for p in rates])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.legend(frameon=False, fontsize=8, loc="best")
        fig.suptitle("MWPM with actual and covariance-estimated priors", fontsize=13)
        fig.supxlabel(caption, fontsize=8)
        result += _save(fig, output / f"ler_vs_physical_error{suffix}")

    for panel in panels:
        suffix = f"_d{panel[0]}" if split_ler else ""
        fig, axes = plt.subplots(1, len(panel), figsize=(6 * len(panel), 5.2), squeeze=False, layout="constrained")
        for ax, distance in zip(axes[0], panel, strict=True):
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
                   ylabel=ylabel, title=f"Surface code d={distance}; fixed heldout test data")
            ax.plot([], [], color="gray", linestyle="--", label="Actual prior at matching p")
            ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2)
        fig.suptitle("Calibration sample size and decoding performance", fontsize=13)
        fig.supxlabel(caption, fontsize=8)
        result += _save(fig, output / f"ler_vs_calibration_shots{suffix}")

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


def _threshold_figure(thresholds: list[float], off_panels: list[bool], height: float):
    fig = plt.figure(figsize=(7.2 * len(off_panels), height), layout="constrained")
    # Build each pair only where its categorical condition has data.
    pair = (4, 1.2) if thresholds else (0.001, 4)
    widths = [width for show_off in off_panels for width in (pair if show_off else (4,))]
    return fig, fig.add_gridspec(1, len(widths), width_ratios=widths)


def _threshold_axes(fig, grid, index: int, thresholds: list[float], no_cooling: bool, *,
                    xlabel: str = "Cooling threshold (transverse phonons)"):
    ax = fig.add_subplot(grid[0, index])
    ax.set(xlabel=xlabel)
    ax.set_xticks(thresholds, labels=[threshold_label(value) for value in thresholds],
                  rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    ax.xaxis.set_minor_locator(NullLocator())
    if not thresholds:
        ax.set_visible(False)
    off = None
    if no_cooling:
        off = fig.add_subplot(grid[0, index + 1], sharey=ax)
        off.xaxis.update_units(["No cooling"])
        off.set_xticks(["No cooling"])
        if thresholds:
            off.tick_params(axis="y", which="both", left=False, labelleft=False)
    return ax, off


def _plot_cooling(root: Path, output: Path, rows: list[dict], ylabel: str,
                  interval_label: str, split_ler: bool, comparison_shots: int | None,
                  include_no_cooling: bool) -> list[Path]:
    physical = read_json(root / "physical_metrics.json")
    config = read_json(root / "run.json")["config"]
    lpt = "liz_pt_thresholds" in config
    sweep_key = "liz_pt_thresholds" if lpt else "cool_thresholds"
    xlabel = "LPT threshold (transverse phonons)" if lpt else "Cooling threshold (transverse phonons)"
    if lpt:
        ylabel = f"Logical error probability over {config['rounds']} rounds"
    thresholds = sorted(value for value in config[sweep_key] if value is not None)
    no_cooling = include_no_cooling and any(r["cool_threshold"] is None and r["status"] == "ok" for r in physical)
    conditions = [*thresholds, None] if no_cooling else thresholds
    distances = sorted(config["distances"])
    counts = config["training_shots"]
    if comparison_shots is not None and (type(comparison_shots) is not int or comparison_shots not in counts):
        available = ", ".join(f"{count:,}" for count in counts)
        raise ValueError(f"comparison_shots must be a configured calibration size. Available: {available}")
    colors = plt.colormaps["viridis"](np.linspace(0.15, 0.85, max(len(counts), len(distances), len(thresholds) + no_cooling)))
    case_rows = {(d, threshold): [r for r in rows if r["distance"] == d and r["cool_threshold"] == threshold]
                 for d in distances for threshold in conditions}
    physical_rows = {(r["distance"], r["cool_threshold"]): r for r in physical}
    caption = f"Intervals show {interval_label} or the envelope across repetitions. Open triangles mark zero-count upper bounds."
    skips = root / "fit_skips.json"
    if skips.is_file() and read_json(skips):
        caption += " Undefined fits are listed in fit_skips.json."
    if any(row["status"] != "ok" for row in physical):
        caption += " LUT rejections are listed in physical_metrics.csv."
    result = []

    def scalar_plot(name, y_label, fields, values):
        show_off = no_cooling and any(np.isfinite(values(d, None, field)) for d in distances for field, _, _ in fields)
        if not thresholds and not show_off:
            return []
        fig, grid = _threshold_figure(thresholds, [show_off], 4.4)
        ax, off = _threshold_axes(fig, grid, 0, thresholds, show_off, xlabel=xlabel)
        for distance, color in zip(distances, colors):
            for field, label, style in fields:
                legend = f"d={distance}{label}"
                if thresholds:
                    ax.plot(thresholds, [values(distance, threshold, field) for threshold in thresholds],
                            color=color, marker="o", linestyle=style, label=legend)
                if off is not None:
                    off.plot(["No cooling"], [values(distance, None, field)],
                             color=color, marker="o" if style == "-" else "^", label=legend)
        main = ax if thresholds else off
        main.set_ylabel(y_label)
        main.legend(frameon=False, fontsize=8)
        return _save(fig, output / name)

    for name, label, fields, scale in (
        ("mean_ms_error_vs_cooling_threshold", "Mean MS error probability", [("mean_ms_error", "", "-")], 1),
        ("cooling_operations_vs_cooling_threshold", "Cooling operations", [("cooling_operations", "", "-")], 1),
        ("execution_time_vs_cooling_threshold", "Scheduled execution time (s)", [("duration_us", "", "-")], 1e6),
        ("ms_phonon_vs_cooling_threshold", "At-gate transverse phonons", [
            ("mean_ms_phonon", " mean", "-"), ("max_ms_phonon", " max", "--")], 1),
    ):
        def physical_value(distance, threshold, field):
            row = physical_rows[distance, threshold]
            if threshold is None and row["status"] != "ok":
                return np.nan
            value = row[field]
            return np.nan if value is None else value / scale

        result += scalar_plot(name, label, fields, physical_value)

    def rmse_value(distance, threshold, field):
        values = [r[field] for r in case_rows[distance, threshold]
                  if r["calibration_shots"] == counts[-1] and r[field] is not None]
        return np.mean(values) if values else np.nan

    result += scalar_plot("prior_rmse_vs_cooling_threshold", f"Mean DEM prior RMSE (N={counts[-1]:,})",
                          [("prior_rmse", "", "-")], rmse_value)
    if lpt:
        result += _plot_lpt_figures(root, output, case_rows, distances, counts, thresholds, xlabel,
                                    comparison_shots or counts[-1], config["rounds"])
    distances = [d for d in distances if any(case_rows[d, t] for t in conditions)]
    if not distances:
        return result

    if comparison_shots is not None:
        result += _plot_cooling_comparison(output, case_rows, distances, thresholds, comparison_shots,
                                            total_rounds=config["rounds"] if lpt else None)

    decoders = [(0, "#202c3b", "Actual prior", "--", "s")]
    decoders += [(n, colors[i], f"Estimated, N={n:,}", "-", "o") for i, n in enumerate(counts)]
    panels = [[d] for d in distances] if split_ler else [distances]
    for panel in panels:
        suffix = f"_d{panel[0]}" if split_ler else ""
        off_panels = [bool(case_rows.get((d, None))) for d in panel]
        fig, grid = _threshold_figure(thresholds, off_panels, 5.0)
        index = 0
        for distance, show_off in zip(panel, off_panels, strict=True):
            ax, off = _threshold_axes(fig, grid, index, thresholds, show_off, xlabel=xlabel)
            index += 1 + show_off
            main = ax if thresholds else off
            main.set(yscale="log", ylabel=ylabel, title=f"Surface code d={distance}")

            def groups(keys, count, x_values):
                return [(x, [r for r in case_rows[distance, key] if r["calibration_shots"] == count])
                        for key, x in zip(keys, x_values, strict=True)]

            for n, color, label, style, marker in decoders:
                if thresholds:
                    _series(ax, groups(thresholds, n, thresholds), color=color, label=label, linestyle=style, marker=marker)
                if off is not None:
                    _series(off, groups([None], n, ["No cooling"]), color=color, label=label,
                            linestyle="none", marker=marker, intervals=True)
            main.legend(frameon=False, fontsize=8)
        fig.supxlabel(fill(caption, width=85) if split_ler else caption, fontsize=8)
        result += _save(fig, output / f"ler_vs_cooling_threshold{suffix}")

    for distance in distances:
        suffix = f"_d{distance}"
        fig, ax = plt.subplots(figsize=(6, 5.2), layout="constrained")
        for index, threshold in enumerate(conditions):
            selected = case_rows[distance, threshold]
            if not selected:
                continue
            color = colors[index]
            label = threshold_label(threshold)
            _series(ax, [(n, [r for r in selected if r["calibration_shots"] == n]) for n in counts],
                    color=color, label=label)
            _series(ax, [(n, [r for r in selected if r["decoder"] == "actual"]) for n in counts],
                    color=color, label="_nolegend_", linestyle="--", marker="s")
        ax.set(xscale="log", yscale="log", xlabel="Calibration detector samples, N", ylabel=ylabel,
               title=f"Calibration size and decoding at d={distance}")
        ax.plot([], [], color="gray", linestyle="--", marker="s", label="Actual prior at matching threshold")
        ax.legend(frameon=False, fontsize=8)
        fig.supxlabel(fill(caption, width=85), fontsize=8)
        result += _save(fig, output / f"ler_vs_calibration_shots{suffix}")

        for field, xlabel, filename, scale in (
            ("mean_ms_error", "Mean MS error probability", "ler_vs_mean_ms_error", 1),
            ("duration_us", "Scheduled execution time (s)", "ler_vs_execution_time", 1e6),
        ):
            fig, ax = plt.subplots(figsize=(6, 4.7), layout="constrained")
            for n, color, label, _, marker in decoders:
                points = []
                for threshold in conditions:
                    selected = [r for r in case_rows[distance, threshold] if r["calibration_shots"] == n]
                    if not selected:
                        continue
                    x = selected[0][field] / scale
                    points.append((x, selected))
                    if threshold is None:
                        y = np.mean([r["logical_error_rate"] for r in selected])
                        ax.scatter(x, y or max(r["ci_high"] for r in selected), marker="D",
                                   facecolors="none", edgecolors=color, s=80)
                if points:
                    _series(ax, points, color=color, label=label, linestyle="none", marker=marker, intervals=True)
            ax.set(yscale="log", xlabel=xlabel, ylabel=ylabel, title=f"Cooling-policy trade-off at d={distance}")
            if case_rows.get((distance, None)):
                ax.scatter([], [], marker="D", facecolors="none", edgecolors="gray", label="No cooling")
            ax.legend(frameon=False, fontsize=8)
            fig.supxlabel(fill(caption, width=85), fontsize=8)
            result += _save(fig, output / f"{filename}{suffix}")

        estimated = [r for t in conditions for r in case_rows[distance, t] if r["decoder"] == "estimated"]
        if estimated:
            example = max(estimated, key=lambda r: (r["calibration_shots"], r["cool_threshold"] is None,
                r["cool_threshold"] or 0, -r["repetition"]))
            folder = (root / f"d{distance}_r{example['rounds']}_{example['basis']}"
                      / cooling_case_name(example["cool_threshold"], lpt=lpt))
            fit = folder / "fits" / f"rep{example['repetition']:03d}_n{example['calibration_shots']}"
            result += plot_prior_diagnostics(folder, fit, output / f"priors_d{distance}")
    return result


def _plot_lpt_figures(root, output, case_rows, distances, counts, thresholds, xlabel, selected_shots, rounds):
    full_rows = read_json(root / "full_detector_metrics.json")
    ratios = read_json(root / "suppression_ratios.json")
    result = []
    colors = plt.colormaps["viridis"](np.linspace(0.12, 0.8, max(len(distances), len(thresholds))))
    if full_rows:
        fig, grid = _threshold_figure(thresholds, [False], 4.8)
        ax, _ = _threshold_axes(fig, grid, 0, thresholds, False, xlabel=xlabel)
        for distance, color in zip(distances, colors):
            selected = [r for r in full_rows if r["distance"] == distance]
            _series(ax, [(t, [r for r in selected if r["liz_pt_threshold"] == t]) for t in thresholds],
                    color=color, label=f"d={distance}")
        ax.set(yscale="log", ylabel=f"Logical error probability over {rounds} rounds",
               title="Full-detector MWPM")
        ax.legend(frameon=False)
        fig.supxlabel("Shading shows 95% Wilson intervals. Open triangles mark zero-count upper bounds.", fontsize=8)
        result += _save(fig, output / "full_detector_ler_vs_lpt_threshold")

    pairs = sorted({(r["d_from"], r["d_to"]) for r in ratios})
    if pairs:
        fig, grid = _threshold_figure(thresholds, [False], 4.8)
        ax, _ = _threshold_axes(fig, grid, 0, thresholds, False, xlabel=xlabel)
        for pair, color in zip(pairs, colors):
            for decoder, shots, label, style in (("actual_full", 0, "full detector", "-"),
                                                  ("actual", 0, "sector actual", "--"),
                                                  ("estimated", selected_shots, f"sector estimated N={selected_shots:,}", ":")):
                selected = {r["liz_pt_threshold"]: r for r in ratios
                            if (r["d_from"], r["d_to"]) == pair
                            and (r["decoder"], r["calibration_shots"]) == (decoder, shots)}
                values = [selected.get(t, {}).get("suppression_ratio") for t in thresholds]
                if all(value is None for value in values):
                    continue
                ax.plot(thresholds, [np.nan if value is None else value for value in values],
                        marker="o", color=color, linestyle=style, label=f"{pair[0]} / {pair[1]} · {label}")
        ax.axhline(1, color="gray", linewidth=1)
        ax.set_ylabel("Logical error suppression ratio, LER(d) / LER(d+2)")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=8)
        result += _save(fig, output / "suppression_ratio_vs_lpt_threshold")

    for distance in distances:
        fig, ax = plt.subplots(figsize=(6.5, 4.4), layout="constrained")
        for threshold, color in zip(thresholds, colors):
            selected = case_rows[distance, threshold]
            rmses = [[r["prior_rmse"] for r in selected
                      if r["calibration_shots"] == count and r["prior_rmse"] is not None] for count in counts]
            if any(rmses):
                ax.plot(counts, [np.mean(values) if values else np.nan for values in rmses],
                        color=color, marker="o", label=threshold_label(threshold))
        if ax.lines:
            ax.set(xscale="log", xlabel="Calibration detector samples, N", ylabel="Mean DEM prior RMSE",
                   title=f"Calibration size and prior recovery at d={distance}")
            ax.legend(title="LPT threshold", frameon=False)
            result += _save(fig, output / f"prior_rmse_vs_calibration_shots_d{distance}")
        else:
            plt.close(fig)
    return result


def _plot_cooling_comparison(output: Path, case_rows: dict, distances: list[int],
                             thresholds: list[float], shots: int, *, total_rounds: int | None = None) -> list[Path]:
    selected = {key: [r for r in rows if r["calibration_shots"] in (0, shots)] for key, rows in case_rows.items()}
    for key, rows in selected.items():
        if total_rounds is None and rows and rows[0].get("rate_unit") != "per_round":
            selected[key] = _per_round_rows(rows)
    show_off = any(selected.get((d, None)) for d in distances)
    fig, grid = _threshold_figure(thresholds, [show_off], 5.3)
    xlabel = "LPT threshold (transverse phonons)" if total_rounds else "Cooling threshold (transverse phonons)"
    ax, off = _threshold_axes(fig, grid, 0, thresholds, show_off, xlabel=xlabel)
    main = ax if thresholds else off
    ylabel = (f"Logical error probability over {total_rounds} rounds" if total_rounds
              else "Logical error rate per syndrome extraction round")
    main.set(yscale="log", ylabel=ylabel)
    colors = plt.colormaps["viridis"](np.linspace(0.12, 0.62, len(distances)))
    for distance, color in zip(distances, colors, strict=True):
        for count, decoder, style, marker in ((0, "Actual", "--", "s"), (shots, "Estimated", "-", "o")):
            groups = [(t, [r for r in selected[distance, t] if r["calibration_shots"] == count])
                      for t in thresholds]
            off_group = [r for r in selected.get((distance, None), []) if r["calibration_shots"] == count]
            label = f"d={distance} · {decoder}"
            has_numeric = any(rows for _, rows in groups)
            if has_numeric:
                _series(ax, groups, color=color, label=label, linestyle=style, marker=marker, intervals=True)
            if off_group:
                _series(off, [("No cooling", off_group)], color=color,
                        label="_nolegend_" if has_numeric else label, linestyle="none", marker=marker, intervals=True)
    handles, labels = main.get_legend_handles_labels()
    if thresholds and off is not None:
        off_handles, off_labels = off.get_legend_handles_labels()
        handles += off_handles
        labels += off_labels
    fig.legend(handles, labels, loc="outside upper center", ncols=min(4, len(labels)), frameon=False, fontsize=9,
               title=f"Decoding by distance at N={shots:,}", title_fontsize=11)
    interval = "95% Wilson" if total_rounds else "Transformed 95% Wilson"
    fig.supxlabel(f"{interval} intervals or their range across repetitions.\n"
                  "Open triangles mark zero-count upper bounds.", fontsize=8)
    name = "sector_ler_vs_lpt_threshold" if total_rounds else "ler_per_round_vs_cooling_threshold"
    return _save(fig, output / f"{name}_n{shots}")


def _actual_pair_matrix(structure: GraphStructure, actual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((structure.num_detectors, structure.num_detectors))
    support = np.zeros(matrix.shape, dtype=bool)
    for edge, prior in zip(structure.edges, actual, strict=True):
        if len(edge.detectors) == 2:
            i, j = edge.detectors
            matrix[i, j] = matrix[j, i] = prior
            support[i, j] = support[j, i] = True
    return matrix, support


def prior_rmse(structure: GraphStructure, actual: np.ndarray, estimated: np.ndarray,
               raw_pair_estimates: np.ndarray) -> dict:
    """Compare canonical edge priors and finite raw S11 pairs against actual."""
    actual_pair, support = _actual_pair_matrix(structure, actual)
    if raw_pair_estimates.shape != actual_pair.shape or estimated.shape != actual.shape:
        raise ValueError("Prior estimates must match the saved detector and edge dimensions")
    pair_indices = np.triu_indices(structure.num_detectors, k=1)
    raw = raw_pair_estimates[pair_indices]
    pair_delta = raw - actual_pair[pair_indices]
    known = support[pair_indices]
    finite = np.isfinite(raw)

    def rmse(delta: np.ndarray) -> float | None:
        return float(np.sqrt(np.mean(delta ** 2))) if delta.size else None

    return {
        "prior_rmse": rmse(estimated - actual), "mechanisms": len(actual),
        "raw_pair_rmse_all": rmse(pair_delta[finite]),
        "raw_pair_rmse_support": rmse(pair_delta[finite & known]),
        "finite_pairs": int(np.count_nonzero(finite)), "total_pairs": len(finite),
        "finite_known_pairs": int(np.count_nonzero(finite & known)),
        "total_known_pairs": int(np.count_nonzero(known)),
    }


def plot_prior_diagnostics(case_dir: str | Path, fit_dir: str | Path, target: str | Path) -> list[Path]:
    """Plot the raw p_ij matrix and edge prior recovery for an existing fit."""
    _style()
    case_dir, fit_dir, target = Path(case_dir), Path(fit_dir), Path(target)
    structure = GraphStructure.from_dict(read_json(case_dir / "known_structure.json"))
    actual = np.load(case_dir / "actual_priors.npy", allow_pickle=False)
    fitted = np.load(fit_dir / "estimated_priors.npy", allow_pickle=False)
    matrix = np.load(fit_dir / "p_ij.npy", allow_pickle=False)
    oracle, _ = _actual_pair_matrix(structure, actual)
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
    condition = (f"threshold: {threshold_label(metadata['cool_threshold'])}" if "cool_threshold" in metadata
                 else f"p={metadata['physical_error_rate']:g}")
    fig.suptitle(f"d={metadata['distance']}, {condition}, N={count:,}", fontsize=12)
    target.parent.mkdir(parents=True, exist_ok=True)
    return _save(fig, target)

