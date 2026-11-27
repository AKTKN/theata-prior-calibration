"""Publication figures recreated from held-out tables and saved checkpoints."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np

from ..io import atomic_json, read_json
from .artifacts import file_digest, single_writer, write_csv

METHODS = {
    "actual": ("Actual", "#333333"),
    "covariance": ("Covariance", "#0072B2"),
    "initial": ("Initialization", "#7A5195"),
    "cem_mwpm": ("MWPM trained", "#D55E00"),
    "cem_uf": ("UF trained", "#009E73"),
}
DECODERS = {"mwpm": "MWPM", "uf": "ldpc UF"}
METRICS = {"mechanisms": "All mechanisms", "pair_mechanisms": "Pair mechanisms",
           "singleton_mechanisms": "Singleton mechanisms"}


@dataclass(frozen=True)
class PlotOptions:
    distance: int | None = None
    physical_error_rate: float | None = None
    calibration_shots: int | None = None
    replicate: int | None = None
    optimizer_seed: int | None = None
    initialization: str | None = None
    color_normalization: str = "linear"
    formats: tuple[str, ...] = ("png", "pdf")

    def __post_init__(self) -> None:
        if self.color_normalization not in ("linear", "log"):
            raise ValueError("color_normalization must be linear or log")
        if not self.formats or set(self.formats) - {"png", "pdf"}:
            raise ValueError("formats must contain png and/or pdf")


def _legend(fig, methods, *, intervals: bool = False) -> None:
    handles = [Line2D([], [], label=METHODS[m][0], marker="o", color=METHODS[m][1]) for m in METHODS if m in methods]
    if intervals:
        handles.append(Line2D([], [], label="Zero failures at upper 95% Wilson limit", marker="v",
                              markersize=5, color="0.3", linestyle="None"))
    fig.legend(handles=handles, loc="outside lower center", ncols=min(len(handles), 4), frameon=False)


def _missing(axis, rows, x_key: str) -> None:
    groups = defaultdict(set)
    for row in rows:
        if row["status"] != "ok":
            groups[(row["method"], row["status"])].add(row[x_key])
    if groups:
        notes = []
        for (method, status), positions in groups.items():
            where = f" at {', '.join(f'{x:g}' for x in sorted(positions))}" if x_key != "method" else ""
            notes.append(f"{METHODS[method][0]}{where} · {status.replace('_', ' ')}")
        axis.text(.02, .02, "Unavailable\n" + "\n".join(notes), transform=axis.transAxes,
                  va="bottom", fontsize=7, bbox={"facecolor": "white", "alpha": .92, "linewidth": 0})


def _failure_points(axis, rows: list[dict], values: list[float], color: str, *, connect: bool) -> None:
    regular = [(x, row) for x, row in zip(values, rows, strict=True)
               if row["status"] == "ok" and row["failures"] > 0]
    if connect:
        axis.plot(values, [row["failure_probability"] if row["status"] == "ok" and row["failures"] else np.nan
                           for row in rows], color=color, alpha=.65, linewidth=1)
    if regular:
        x, records = zip(*regular)
        y = np.array([row["failure_probability"] for row in records])
        ci = np.array([[row["ci_low"], row["ci_high"]] for row in records]).T
        axis.errorbar(x, y, yerr=np.maximum(0, np.array([y - ci[0], ci[1] - y])),
                      fmt="o", color=color, alpha=.7, capsize=2, markersize=3)
    zero = [(x, row["ci_high"]) for x, row in zip(values, rows, strict=True)
            if row["status"] == "ok" and not row["failures"]]
    if zero:
        axis.scatter(*zip(*zero), marker="v", color=color, s=24)


def _by_keys(rows, keys):
    return sorted({tuple(row[key] for key in keys) for row in rows})


def _sweep(rows, x_key, facet_keys, *, failure: bool):
    facets = _by_keys(rows, facet_keys)
    columns = min(3, len(facets))
    fig, axes = plt.subplots((len(facets) + columns - 1) // columns, columns, squeeze=False,
                             figsize=(4.05 * columns, 3.05 * ((len(facets) + columns - 1) // columns)),
                             layout="constrained")
    for axis, values in zip(axes.flat, facets):
        panel = [row for row in rows if tuple(row[key] for key in facet_keys) == values]
        positions = sorted({row[x_key] for row in panel})
        gap = min(np.diff(np.log(positions))) if len(positions) > 1 else .3
        offset_step = min(.015, gap / 12)
        sort_keys = ("method", "replicate", "optimizer_seed", "initialization", "rounds", "basis")
        for series in _by_keys(panel, sort_keys):
            curve = sorted((row for row in panel if tuple(row[key] for key in sort_keys) == series),
                           key=lambda row: row[x_key])
            offset = (list(METHODS).index(series[0]) - (len(METHODS) - 1) / 2) * offset_step
            x = [row[x_key] * np.exp(offset) for row in curve]
            color = METHODS[series[0]][1]
            if failure:
                _failure_points(axis, curve, x, color, connect=True)
            else:
                axis.plot(x, [row.get("rmse", np.nan) if row["status"] == "ok" else np.nan for row in curve],
                          color=color, marker="o", linewidth=1, markersize=3, alpha=.7)
        title = {"distance": lambda x: f"d = {x}", "physical_error_rate": lambda x: f"p = {x:g}",
                 "evaluation_decoder": DECODERS.get, "metric": METRICS.get}
        axis.set_title(" · ".join(title[key](value) for key, value in zip(facet_keys, values, strict=True)))
        axis.set_xscale("log")
        axis.set_xticks(positions, [f"{x:g}" if x_key == "physical_error_rate" else f"{x:,}" for x in positions])
        axis.set_xlim(positions[0] * np.exp(-.1), positions[-1] * np.exp(.1))
        axis.set_xlabel("Physical error probability" if x_key == "physical_error_rate" else "Calibration shots")
        axis.set_ylabel("Failure probability per memory shot" if failure else "Prior RMSE")
        if failure:
            axis.set_yscale("log")
            if all(row["status"] != "ok" for row in panel):
                axis.set_ylim(1e-6, 1)
        else:
            axis.set_ylim(bottom=0)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 2))
            axis.yaxis.set_major_formatter(formatter)
        axis.grid(axis="y", alpha=.2)
        _missing(axis, panel, x_key)
    for axis in list(axes.flat)[len(facets):]:
        axis.set_visible(False)
    _legend(fig, {row["method"] for row in rows}, intervals=failure)
    return fig


def _transfer(rows, subtitle):
    names = sorted({row["evaluation_decoder"] for row in rows})
    fig, axes = plt.subplots(1, len(names), figsize=(4.1 * len(names), 3.6), squeeze=False, layout="constrained")
    for axis, name in zip(axes[0], names, strict=True):
        panel = [row for row in rows if row["evaluation_decoder"] == name]
        methods = [method for method in METHODS if any(row["method"] == method for row in panel)]
        for index, method in enumerate(methods):
            row = next(row for row in panel if row["method"] == method)
            _failure_points(axis, [row], [index], METHODS[method][1], connect=False)
        axis.set_xticks(range(len(methods)), [METHODS[method][0] for method in methods], rotation=24, ha="right")
        axis.set_ylabel("Failure probability per memory shot")
        axis.set_title(f"Evaluated with {DECODERS[name]}")
        axis.set_yscale("log")
        if all(row["status"] != "ok" for row in panel):
            axis.set_ylim(1e-6, 1)
        axis.grid(axis="y", alpha=.2)
        _missing(axis, panel, "method")
    fig.suptitle(subtitle + "\nTriangles show zero failures at the upper 95% Wilson limit", fontsize=9)
    return fig


def _matrix_figure(data, available, methods, subtitle, *, residual=False, log=False):
    prefix = "pair_residual_vs_actual_" if residual else "pair_prior_"
    arrays = [data[prefix + method] for method in methods if prefix + method in data]
    max_abs = max((float(np.max(np.abs(array))) for array in arrays), default=0.0) or 1.0
    positives = [float(np.min(array[array > 0])) for array in arrays if np.any(array > 0)]
    if residual:
        norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    elif log and positives:
        norm = LogNorm(vmin=min(min(positives), max_abs / 1.01), vmax=max_abs)
    else:
        norm = Normalize(vmin=0, vmax=max_abs)
    cmap = plt.get_cmap("RdBu_r" if residual else "viridis").with_extremes(bad="white")
    fig, axes = plt.subplots(1, len(methods), figsize=(3.1 * len(methods), 3.6), squeeze=False, layout="constrained")
    ticks = np.unique(np.linspace(0, len(data["detector_ids"]) - 1, min(5, len(data["detector_ids"])), dtype=int))
    for axis, method in zip(axes[0], methods, strict=True):
        axis.set_title(METHODS[method][0])
        key = prefix + method
        if key in data:
            array = np.ma.array(data[key], mask=~data["pair_support"])
            if log and not residual:
                array = np.ma.masked_less_equal(array, 0)
            axis.imshow(array, norm=norm, cmap=cmap, interpolation="nearest")
        else:
            axis.text(.5, .5, available[method].replace("_", " "), transform=axis.transAxes, ha="center")
        positions = ticks if key in data else []
        axis.set_xticks(positions, data["detector_ids"][positions])
        axis.set_yticks(positions, data["detector_ids"][positions])
        axis.set_xlabel("Source detector")
    axes[0, 0].set_ylabel("Source detector")
    label = "Signed prior residual versus actual" if residual else "Pair mechanism probability"
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axes[0]), shrink=.73, label=label)
    fig.suptitle(subtitle + (" · white denotes absent pairs or zeros masked for log scale" if log and not residual
                            else " · white denotes absent pairs and diagonal"), fontsize=9)
    return fig


def _raw(data, status, subtitle):
    values = data["pair_prior_cov_raw"]
    mask = data["pair_prior_cov_raw_valid"]
    limit = float(np.max(np.abs(values[mask]))) if mask.any() else 0.0
    limit = limit or 1.0
    fig, axis = plt.subplots(figsize=(5.2, 4.3), layout="constrained")
    image = axis.imshow(np.ma.array(values, mask=~mask), cmap=plt.get_cmap("RdBu_r").with_extremes(bad="#B8B8B8"),
                        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), interpolation="nearest")
    ticks = np.unique(np.linspace(0, len(data["detector_ids"]) - 1, min(7, len(data["detector_ids"])), dtype=int))
    axis.set(xticks=ticks, xticklabels=data["detector_ids"][ticks], yticks=ticks, yticklabels=data["detector_ids"][ticks],
             xlabel="Source detector", ylabel="Source detector", title="Raw covariance-derived pair probabilities")
    fig.colorbar(image, ax=axis, label="Signed pair prior estimate")
    fig.suptitle(subtitle + "\nGray marks invalid entries and the diagonal", fontsize=9)
    if status != "ok":
        axis.text(.02, .02, status.replace("_", " "), transform=axis.transAxes, fontsize=8,
                  bbox={"facecolor": "white", "alpha": .9, "linewidth": 0})
    return fig


def _singletons(data, statuses, subtitle):
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 5), sharex=True, layout="constrained")
    respective = data["singleton_support"]
    x = data["detector_ids"][respective]
    actual = data["singleton_prior_actual"][respective]
    methods = []
    missing = []
    for method, (_, color) in METHODS.items():
        key = f"singleton_prior_{method}"
        if key in data:
            methods.append(method)
            y = data[key][respective]
            axes[0].plot(x, y, marker="o", markersize=3, linewidth=.8, color=color)
            if method != "actual":
                axes[1].plot(x, y - actual, marker="o", markersize=3, linewidth=.8, color=color)
        elif method in statuses:
            missing.append(f"{METHODS[method][0]} · {statuses[method].replace('_', ' ')}")
    axes[0].set_ylabel("Singleton prior probability")
    axes[1].set_ylabel("Residual versus actual")
    axes[1].set_xlabel("Source detector with singleton support")
    for axis in axes:
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
        axis.grid(axis="y", alpha=.2)
    if not len(x):
        missing.append("No singleton supports")
    if missing:
        axes[0].text(.02, .97, "\n".join(missing), transform=axes[0].transAxes, va="top", fontsize=8)
    fig.suptitle(subtitle, fontsize=9)
    _legend(fig, methods)
    return fig


def _trace(path: Path, method: str, subtitle: str):
    state = read_json(path / "state.json")
    records = [read_json(path / "history" / f"generation_{generation:05d}.json")
               for generation in range(1, state["generation"] + 1)]
    fig, (train, val) = plt.subplots(2, 1, sharex=True, figsize=(7.0, 5.0), layout="constrained")
    x = [row["generation"] for row in records]
    if records:
        for role, label, color in (("initial", "Initialization", "#7A5195"), ("incumbent", "Incoming incumbent", "#0072B2")):
            values = [next(candidate["failures"] for candidate in row["candidates"] if candidate["role"] == role)
                      / row["batch_shots"] for row in records]
            train.scatter(x, values, label=label, s=20, color=color, alpha=.75)
        rates = [row["winner_failures"] / row["batch_shots"] for row in records]
        train.scatter(x, rates, marker="x", s=28, label="Current-batch winner", color="#D55E00")
        flat = [(row["generation"], rate) for row, rate in zip(records, rates, strict=True) if row["flat"]]
        if flat:
            train.scatter(*zip(*flat), facecolors="none", edgecolors="#444444", s=75, label="Flat stochastic scores")
        train.legend(frameon=False, fontsize=8, ncols=2)
    else:
        train.text(.5, .5, "Only the initialization was scored", ha="center", transform=train.transAxes)
    checkpoints = state["validation_history"]
    val.plot([row["generation"] for row in checkpoints], [row["failures"] / row["shots"] for row in checkpoints],
             color=METHODS[method][1], marker="o", markersize=4)
    selected = state["best_validation"]
    val.scatter(selected["generation"], selected["failures"] / selected["shots"], marker="*", color="#333333",
                s=100, zorder=3, label="Selected checkpoint")
    val.legend(frameon=False, fontsize=8)
    train.set_ylabel("Training-batch failure probability")
    val.set_ylabel("Validation failure probability")
    val.set_xlabel("CEM generation")
    fig.suptitle(f"{METHODS[method][0]} · {subtitle}\nTraining scores use the batch for each generation", fontsize=9)
    return fig


def _variation(rows):
    """Summarize spread between trained models while keeping the shared tests paired."""
    keys = ("distance", "rounds", "basis", "physical_error_rate", "calibration_shots",
            "initialization", "method", "evaluation_decoder")
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["status"] == "ok":
            groups[tuple(row[key] for key in keys)][row["replicate"]].append(row["failure_probability"])
    output = []
    for key, replicates in sorted(groups.items()):
        rates = [np.mean(seeds) for seeds in replicates.values()]
        optimizer_sd = [np.std(seeds, ddof=1) for seeds in replicates.values() if len(seeds) > 1]
        output.append({**dict(zip(keys, key, strict=True)), "data_replicates": len(rates),
                       "mean_of_replicate_rates": float(np.mean(rates)),
                       "sd_between_data_replicate_mean_rates": float(np.std(rates, ddof=1)) if len(rates) > 1 else None,
                       "mean_within_replicate_optimizer_seed_sd": float(np.mean(optimizer_sd)) if optimizer_sd else None,
                       "replicates_with_multiple_optimizer_seeds": len(optimizer_sd)})
    return output


def load_results(output_dir: str | Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Read completed held-out results, numerical diagnostics and the artifact index."""
    root = Path(output_dir)
    marker = read_json(root / "evaluation_complete.json")
    if any(file_digest(root / name) != digest for name, digest in marker["artifacts"].items()):
        raise ValueError("Held-out result tables differ from their completion marker")
    return tuple(read_json(root / name) for name in (
        "evaluation.json", "prior_diagnostics.json", "evaluation_artifacts.json"))


@single_writer
def plot_results(output_dir: str | Path, options: PlotOptions | None = None) -> list[Path]:
    """Plot saved results. Selectors choose the detailed design and the rate-sweep budget."""
    options = options or PlotOptions()
    root = Path(output_dir).resolve()
    rows, errors, index = load_results(root)
    selectors = {key: getattr(options, key) for key in ("distance", "physical_error_rate", "calibration_shots",
                                                        "replicate", "optimizer_seed", "initialization")
                 if getattr(options, key) is not None}
    eligible = [row for row in index if all(row[key] == value for key, value in selectors.items())]
    if not eligible:
        raise ValueError(f"No saved design matches plot selectors {selectors}")
    keys = ("distance", "physical_error_rate", "calibration_shots", "replicate", "optimizer_seed", "initialization")
    detailed = min(eligible, key=lambda row: tuple(row[key] for key in keys))
    budget = detailed["calibration_shots"]
    detail_rows = [row for row in rows if all(row[key] == detailed[key] for key in keys)]
    region = root / "figures"
    region.mkdir(parents=True, exist_ok=True)
    write_csv(region / "replicate_variation.csv", _variation(rows))
    atomic_json(region / "selection.json", {"options": asdict(options), "detailed_design": {key: detailed[key] for key in keys},
                "fixed_model_intervals": "95% Wilson per held-out model", "replicate_variation": "No pooling of reused test shots"})
    subtitle = (f"d = {detailed['distance']} · r = {detailed['rounds']} · p = {detailed['physical_error_rate']:g} · "
                f"N = {budget:,} · data replicate {detailed['replicate']} · optimizer seed {detailed['optimizer_seed']} · "
                f"{detailed['realized_initialization'].replace('_', ' ')} initialization")
    paths = []

    def save(name, fig):
        for extension in options.formats:
            path = region / f"{name}.{extension}"
            fig.savefig(path, dpi=250, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)

    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"}):
        save("failure_vs_physical_error", _sweep([row for row in rows if row["calibration_shots"] == budget],
             "physical_error_rate", ("distance", "evaluation_decoder"), failure=True))
        save("failure_vs_calibration_shots", _sweep(rows, "calibration_shots",
             ("distance", "physical_error_rate", "evaluation_decoder"), failure=True))
        save("cross_decoder_transfer", _transfer(detail_rows, subtitle))
        prior_rows = [row for row in errors if row["comparison"] == "actual" and row["metric"] in METRICS]
        save("prior_rmse_vs_calibration_shots", _sweep(prior_rows, "calibration_shots",
             ("distance", "physical_error_rate", "metric"), failure=False))
        save("prior_rmse_vs_physical_error", _sweep([row for row in prior_rows if row["calibration_shots"] == budget],
             "physical_error_rate", ("distance", "metric"), failure=False))
        matrix_path = root / detailed["matrices"]
        if file_digest(matrix_path) != detailed["matrices_sha256"]:
            raise ValueError(f"Saved prior matrices differ from the evaluation index: {matrix_path}")
        with np.load(matrix_path, allow_pickle=False) as archive:
            matrices = dict(archive)
        methods = [method for method in METHODS if method in detailed["statuses"] and method != "initial"]
        status = detailed["statuses"]
        save("pair_priors", _matrix_figure(matrices, status, methods, subtitle, log=options.color_normalization == "log"))
        save("pair_residuals_vs_actual", _matrix_figure(matrices, status, methods, subtitle, residual=True))
        save("raw_covariance_pair_priors", _raw(matrices, status["covariance_raw"], subtitle))
        save("singleton_priors", _singletons(matrices, status, subtitle))
        for method, path in detailed["traces"].items():
            save(f"trace_{method}", _trace(root / path, method, subtitle))
    return paths
