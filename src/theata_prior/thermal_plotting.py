"""Regenerate thermal figures from recorded experiment and ratio statistics."""
from __future__ import annotations

import math
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np

from .io import read_json


def _positions(rows, sweep):
    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    if sweep == "cooling":
        position = dict(zip(conditions, range(len(conditions)), strict=True))
        controls = {"no_cooling": "No cooling", "postmeasure_liz": "Post-measure\nLIZ only"}
        labels = {row["condition"]: controls.get(row["condition"], str(row["n_max"])) for row in rows}
        return position, list(position.values()), [labels[key] for key in position]
    cap_by_condition = {row["condition"]: row["cap"] for row in rows if row["cap"] is not None}
    low, high = min(cap_by_condition.values()), max(cap_by_condition.values())
    span = high - low or max(high, 1)
    uncapped = high + 0.12 * span
    position = {**cap_by_condition, "uncapped": uncapped}
    ticks = list(cap_by_condition.values())
    if len(ticks) > 8:
        ticks = [value for value in MaxNLocator(nbins=6).tick_values(low, high) if low <= value <= high]
    return position, [*ticks, uncapped], [*[f"{x:g}" for x in ticks], "Uncapped"]


def _logical_series(axis, rows, x, color, label, control):
    estimates = [row["logical_error_rate"] or np.nan for row in rows]
    axis.plot(x, estimates, color=color, marker="o", ms=4, ls="none" if control else "-", label=label)
    for xx, estimate, row in zip(x, estimates, rows, strict=True):
        upper = row["ci_high"]
        if math.isnan(estimate):
            axis.scatter([xx], [upper], marker="v", color=color, s=23, zorder=4)
        else:
            low = row["ci_low"]
            axis.errorbar(xx, estimate, yerr=[[max(estimate - low, 0)], [max(upper - estimate, 0)]],
                          fmt="none", ecolor=color, capsize=2, alpha=0.75)


def _ratio_series(axis, rows, x, color, label, control, top):
    estimates = [row["lambda"] if row["lambda"] is not None else np.nan for row in rows]
    axis.plot(x, estimates, color=color, marker="o", ms=4, ls="none" if control else "-", label=label)
    for xx, estimate, row in zip(x, estimates, rows, strict=True):
        low, upper = row["ci_low"], row["ci_high"]
        axis.vlines(xx, low, top if upper is None else upper, color=color, alpha=0.65)
        if upper is None:
            axis.scatter([xx], [top], marker="^", color=color, s=21, zorder=4)
        elif not math.isnan(estimate):
            axis.plot([xx, xx], [low, upper], "_", color=color, ms=4)


def plot_thermal(out):
    directory = Path(out)
    results = read_json(directory / "metrics.json")
    science = read_json(directory / "run.json")["science"]
    rows, ratios = results["rows"], results["ratios"]
    sweep = science["sweep"]
    position, ticks, labels = _positions(rows, sweep)
    control_labels = {"uncapped"} if sweep == "cap" else {"no_cooling", "postmeasure_liz"}
    figure = Figure(figsize=(10.6, 7.1) if ratios else (10.6, 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2 if ratios else 1, 1, sharex=True, squeeze=False).ravel()
    palette = ["#245ea8", "#b64936", "#247358", "#7e4595", "#9c6018", "#5a6375"]
    family = "Rotated surface code" if science["code"] == "surface" else "Triangular color code"
    experiment = "virtual phonon cap" if sweep == "cap" else "memory-predictive cooling"
    figure.suptitle(f"{family} · {experiment}", fontsize=15)
    for index, distance in enumerate(science["distances"]):
        selected = [row for row in rows if row["distance"] == distance]
        for control in (False, True):
            subset = [row for row in selected if (row["condition"] in control_labels) == control]
            _logical_series(axes[0], subset, [position[row["condition"]] for row in subset], palette[index],
                            None if control else f"d = {distance}", control)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Logical error probability per shot")
    axes[0].legend(loc="best", ncols=min(3, len(science["distances"])))
    if ratios:
        top = 1.05 * max([1.0, *[max(row["ci_low"], row["ci_high"] or 0) for row in ratios]])
        axes[1].axhline(1, ls="--", color="#626a75", lw=1)
        for index, distance in enumerate(sorted({row["d_from"] for row in ratios})):
            selected = [row for row in ratios if row["d_from"] == distance]
            for control in (False, True):
                subset = [row for row in selected if (row["condition"] in control_labels) == control]
                _ratio_series(axes[1], subset, [position[row["condition"]] for row in subset], palette[index],
                              None if control else f"{distance} / {distance + 2}", control, top)
        # Preserve zero on the axis and keep the Lambda=1 neighborhood readable
        # when a finite ratio interval spans several orders of magnitude.
        if top > 10:
            axes[1].set_yscale("symlog", linthresh=0.01, linscale=0.55)
        axes[1].set_ylim(bottom=0)
        axes[1].set_ylabel(r"$\Lambda = p_L(d)/p_L(d+2)$")
        axes[1].legend(loc="best", ncols=min(3, len(science["distances"]) - 1))
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(ticks, labels)
    axes[-1].set_xlabel("Gate-time phonon cap" if sweep == "cap" else "Maximum MPT cycles per checkpoint")
    note = f"{science['rounds']} syndrome rounds · {science['basis']} memory · 95% confidence intervals"
    if any(row["failures"] == 0 for row in rows):
        note += " · ▼ zero failures (upper bound)"
    if any(row["ci_high"] is None for row in ratios):
        note += " · ▲ unbounded ratio interval"
    if sweep == "cooling" and all(row["policy_status"] == "policy_inactive" for row in rows if row["n_max"] is not None):
        note += "\nMPT was inactive. Budget points share the post-measurement LIZ-only physical model."
    figure.supxlabel(note, fontsize=9)
    folder = directory / "figures"
    folder.mkdir(exist_ok=True)
    paths = [folder / f"{science['code']}_{sweep}.{extension}" for extension in ("png", "pdf")]
    for path in paths:
        figure.savefig(path, dpi=175)
    return paths
