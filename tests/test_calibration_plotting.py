import json

import numpy as np
import pytest

from theata_prior import plotting


THRESHOLDS = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0, None]


def _write_results(root, *, valid_no_cooling: bool):
    config = {"distances": [5, 7], "cool_thresholds": THRESHOLDS, "training_shots": [100, 1000]}
    physical = []
    metrics = []
    for distance in config["distances"]:
        for threshold in THRESHOLDS:
            identity = {"distance": distance, "cool_threshold": threshold}
            valid = threshold is not None or valid_no_cooling
            physical.append(dict(**identity, status="ok" if valid else "lut_out_of_range",
                                 mean_ms_error=0.01 if valid else None, cooling_operations=40,
                                 duration_us=1200, mean_ms_phonon=2.5, max_ms_phonon=3.0))
            if valid:
                for shots, rate in ((0, 0.55), (100, 0.50), (1000, 0.15)):
                    metrics.append(dict(**identity, rounds=distance, basis="z", calibration_shots=shots,
                                        decoder="estimated" if shots else "actual", repetition=0,
                                        logical_error_rate=rate, ci_low=rate - 0.02, ci_high=rate + 0.02,
                                        prior_rmse=0.01 if shots else None, mean_ms_error=0.01, duration_us=1200))
    for name, data in (("run", {"config": config}), ("physical_metrics", physical), ("metrics", metrics)):
        (root / f"{name}.json").write_text(json.dumps(data))


@pytest.mark.parametrize("valid_no_cooling,include_no_cooling,has_panel", [
    (False, True, False), (True, False, False), (True, True, True),
])
def test_cooling_plots_at_fixed_calibration_size(tmp_path, monkeypatch, valid_no_cooling,
                                                 include_no_cooling, has_panel):
    _write_results(tmp_path, valid_no_cooling=valid_no_cooling)
    monkeypatch.setattr(plotting, "plot_prior_diagnostics", lambda *args: [])
    checked = set()

    def inspect_figure(fig, target):
        try:
            if target.name == "mean_ms_error_vs_cooling_threshold":
                assert len(fig.axes) == 1 + has_panel
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                ticks = fig.axes[0].get_xticklabels()
                assert [tick.get_text() for tick in ticks] == ["0.25", "0.5", "1", "1.5", "2", "4"]
                boxes = [tick.get_window_extent(renderer) for tick in ticks]
                assert all(left.x1 < right.x0 for left, right in zip(boxes, boxes[1:]))
                checked.add("physical")
            elif target.name == "ler_vs_cooling_threshold_d5":
                assert len(fig.axes) == 1 + has_panel
                checked.add("ler")
            elif target.name == "ler_per_round_vs_cooling_threshold_n1000":
                assert len(fig.axes) == 1 + has_panel
                ax = fig.axes[0]
                assert ax.get_xscale() == "linear"
                assert ax.get_yscale() == "log"
                assert ax.get_ylabel() == "Logical error rate per syndrome extraction round"
                assert len(ax.lines) == 4
                for line, (distance, rate) in zip(ax.lines, [(5, 0.55), (5, 0.15), (7, 0.55), (7, 0.15)],
                                                  strict=True):
                    np.testing.assert_array_equal(line.get_xdata(), THRESHOLDS[:-1])
                    np.testing.assert_allclose(line.get_ydata(), 1 - (1 - rate)**(1 / distance))
                checked.add("comparison")
            return []
        finally:
            plotting.plt.close(fig)

    monkeypatch.setattr(plotting, "_save", inspect_figure)
    plotting.plot_results(tmp_path, split_ler=True, comparison_shots=1000, include_no_cooling=include_no_cooling)
    assert checked == {"physical", "ler", "comparison"}


def test_cooling_plot_rejects_an_unconfigured_calibration_size(tmp_path):
    _write_results(tmp_path, valid_no_cooling=False)
    with pytest.raises(ValueError, match="Available: 100, 1,000"):
        plotting.plot_results(tmp_path, comparison_shots=120)
