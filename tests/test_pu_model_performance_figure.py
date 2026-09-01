import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.pu_model_performance_audit import plot_roc_confusion as plot


def test_default_frozen_validation_inputs_exist():
    for directory in (plot.DEFAULT_A, plot.DEFAULT_B):
        assert (directory / "bag_validation_metrics.csv").is_file()
        assert (directory / "macro_validation_curves.csv").is_file()
        assert (directory / "run_metadata.json").is_file()


def _toy_model(model: str = "CGCNN-PU") -> dict:
    return {
        "model": model,
        "roc": pd.DataFrame(
            {
                "x": [0.0, 0.5, 1.0],
                "mean": [0.0, 0.8, 1.0],
                "q025": [0.0, 0.7, 1.0],
                "q975": [0.0, 0.9, 1.0],
            }
        ),
        "auc_mean": 0.90,
        "auc_sd": 0.01,
        "n_bags": 5,
        "matrix": np.array([[90.0, 10.0], [15.0, 85.0]]),
        "row_pct": np.array([[90.0, 10.0], [15.0, 85.0]]),
    }


def test_pu_model_si_axis_labels_start_with_capitals():
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    data = _toy_model()
    plot.draw_roc(axes[0], data, plot.BLUE, "a")
    plot.draw_confusion(axes[1], data, "b")

    labels = [
        label
        for ax in fig.axes
        for label in (ax.get_xlabel(), ax.get_ylabel())
        if label.strip()
    ]
    plt.close(fig)

    assert "False-positive rate" in labels
    assert "True-positive rate" in labels
    assert "Threshold = 0.5" in labels
    assert "Validation label" in labels
    assert "Row percentage" in labels


def test_pu_performance_figure_has_no_titles_or_explanatory_footer():
    model_a = _toy_model("CGCNN-PU")
    model_b = _toy_model("MatterSim-1M-MLP-PU")

    fig, axes = plot.build_figure(model_a, model_b)
    try:
        assert fig._suptitle is None
        assert all(ax.get_title() == "" for ax in axes)
        assert [text.get_text() for text in fig.texts] == [
            "CGCNN-PU",
            "MatterSim-1M-MLP-PU",
        ]
    finally:
        plt.close(fig)


def test_all_new_figure_sources_avoid_titles():
    sources = [
        plot.ROOT / "experiments/pu_synthesizability_20260821/plot_merged_fig45_nature.py",
        plot.ROOT / "experiments/property_design_20260821/plot_synthesis_score_design.py",
        plot.ROOT / "experiments/pu_synthesizability_20260821/plot_si_l4_contribution.py",
        plot.ROOT / "experiments/pu_model_performance_audit/plot_roc_confusion.py",
        plot.ROOT / "experiments/pu_synthesizability_20260821/render_moved_si_panels.py",
    ]
    for source in sources:
        text = source.read_text(encoding="utf-8")
        assert ".set_title(" not in text, source
        assert ".suptitle(" not in text, source
