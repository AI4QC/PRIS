from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from experiments.pris_composition_holdout_20260829.plot_figures import (
    load_results,
    make_overview_figure,
    make_set4_sensitivity_figure,
    save_figure,
)


HERE = Path(__file__).resolve().parent


def test_si_figures_have_expected_panels_and_no_axes_titles() -> None:
    counts, metrics, per_class = load_results(HERE / "results")
    overview = make_overview_figure(counts, metrics)
    sensitivity = make_set4_sensitivity_figure(metrics, per_class)
    try:
        assert len(overview.axes) == 3
        assert len(sensitivity.axes) == 2
        for figure in (overview, sensitivity):
            assert all(axis.get_title() == "" for axis in figure.axes)
            assert all(
                not any(line.get_visible() for line in axis.get_xgridlines())
                for axis in figure.axes
            )
    finally:
        plt.close(overview)
        plt.close(sensitivity)


def test_save_figure_writes_editable_and_preview_formats(tmp_path: Path) -> None:
    counts, metrics, _ = load_results(HERE / "results")
    figure = make_overview_figure(counts, metrics)
    save_figure(figure, tmp_path, "candidate")
    for suffix in ("pdf", "svg", "png"):
        path = tmp_path / f"candidate.{suffix}"
        assert path.exists()
        assert path.stat().st_size > 1_000
