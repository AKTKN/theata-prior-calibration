import pytest

from theata_prior.structure_plotting import _label_ler_ticks, plt


@pytest.mark.parametrize("limits", [(0.034, 1.04), (0.48, 0.52), (0.00013, 0.00019)])
def test_ler_panels_show_percent_ticks_even_within_one_decade(limits):
    fig, ax = plt.subplots()
    try:
        ax.set(yscale="log", ylim=limits)
        _label_ler_ticks(ax)
        fig.canvas.draw()
        ticks = ax.get_yticks()
        labels = [label.get_text() for label in ax.get_yticklabels()]
        assert 3 <= len(ticks) <= 7
        assert all(limits[0] <= tick <= min(limits[1], 1) for tick in ticks)
        assert all(label.endswith("%") for label in labels)
        assert len(set(labels)) == len(ticks)
        if limits[1] > 1:
            assert "5%" in labels and "100%" in labels
    finally:
        plt.close(fig)
