#!/usr/bin/env python3
"""Plot standard validation diagnostics for the two CSAgent PU classifiers.

The inputs are the raw, per-bag validation summaries produced by
``evaluate_remote_confusion.py``.  Each model is shown with a mean ROC curve
and its bag-wise 95% envelope, alongside a confusion matrix at the frozen
0.5 decision threshold.  The labels are experimental positives and sampled
pseudo-negatives; this is deliberately not called an independent test set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / 'src'))
import figure_palette  # noqa: F401
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_A = ROOT / "experiments/pu_model_performance_audit/raw_A"
DEFAULT_B = ROOT / "experiments/pu_model_performance_audit/raw_B_opt"
DEFAULT_OUT = ROOT / "experiments/pu_model_performance_audit/standard_roc_confusion"
CM = 1 / 2.54
BLUE = "#005B93"
ORANGE = "#9861B0"
INK = "#1F2022"
GREY = "#757779"


def style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def clean(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)


def panel_letter(ax: plt.Axes, value: str) -> None:
    ax.text(-0.14, 1.05, value, transform=ax.transAxes, fontsize=10,
            fontweight="bold", color=INK, ha="left", va="bottom")


def load_model(directory: Path) -> dict:
    metrics_path = directory / "bag_validation_metrics.csv"
    curves_path = directory / "macro_validation_curves.csv"
    metadata_path = directory / "run_metadata.json"
    for path in (metrics_path, curves_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(path)
    metrics = pd.read_csv(metrics_path)
    curves = pd.read_csv(curves_path)
    required_metrics = {
        "model", "bag", "n", "n_positive", "n_pseudo_negative",
        "decision_threshold", "true_negative", "false_positive",
        "false_negative", "true_positive", "roc_auc",
    }
    missing = sorted(required_metrics - set(metrics.columns))
    if missing:
        raise ValueError(f"{metrics_path} missing columns: {missing}")
    required_curves = {"model", "curve", "x", "mean", "q025", "q975"}
    missing = sorted(required_curves - set(curves.columns))
    if missing:
        raise ValueError(f"{curves_path} missing columns: {missing}")
    if metrics["model"].isna().any():
        raise ValueError(f"{metrics_path} contains missing model names")
    model_names = metrics["model"].unique().tolist()
    if len(model_names) != 1:
        raise ValueError(f"expected one model in {metrics_path}, found {model_names}")
    model = model_names[0]
    metrics = metrics.sort_values("bag").reset_index(drop=True)
    roc = curves.loc[curves["curve"].eq("roc")].sort_values("x")
    if roc.empty:
        raise ValueError(f"no ROC curve in {curves_path}")
    # Average counts over bags so the matrix is readable as one representative
    # validation bag.  Row percentages are computed after averaging.
    count_cols = ["true_negative", "false_positive", "false_negative", "true_positive"]
    counts = metrics[count_cols].astype(float).mean(axis=0).to_numpy().reshape(2, 2)
    # rows: pseudo-negative, experimental positive; columns: predicted negative, positive
    matrix = np.array([[counts[0, 0], counts[0, 1]], [counts[1, 0], counts[1, 1]]])
    row_pct = matrix / matrix.sum(axis=1, keepdims=True) * 100
    auc = metrics["roc_auc"].astype(float)
    return {
        "directory": str(directory),
        "model": model,
        "metrics": metrics,
        "roc": roc,
        "matrix": matrix,
        "row_pct": row_pct,
        "auc_mean": float(auc.mean()),
        "auc_sd": float(auc.std(ddof=1)),
        "n_bags": int(metrics["bag"].nunique()),
        "n_per_bag": int(metrics["n"].iloc[0]),
        "n_positive_per_bag": int(metrics["n_positive"].iloc[0]),
        "n_negative_per_bag": int(metrics["n_pseudo_negative"].iloc[0]),
        "thresholds": sorted(metrics["decision_threshold"].astype(float).unique().tolist()),
        "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
    }


def draw_roc(ax: plt.Axes, data: dict, color: str, letter: str) -> None:
    roc = data["roc"]
    x = roc["x"].to_numpy(float)
    mean = roc["mean"].to_numpy(float)
    lo = roc["q025"].to_numpy(float)
    hi = roc["q975"].to_numpy(float)
    ax.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0)
    ax.plot(x, mean, color=color, lw=1.45,
            label=f"ROC-AUC {data['auc_mean']:.4f} ± {data['auc_sd']:.4f}")
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.75, ls="--", label="Chance")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1]); ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.legend(frameon=False, loc="lower right", handlelength=1.35,
              borderpad=0.0, labelspacing=0.2)
    ax.text(0.03, 0.96, f"{data['n_bags']} validation bags",
            transform=ax.transAxes, va="top", color=GREY, fontsize=7.5)
    clean(ax); panel_letter(ax, letter)


def draw_confusion(ax: plt.Axes, data: dict, letter: str) -> None:
    matrix = data["matrix"]
    pct = data["row_pct"]
    im = ax.imshow(pct, cmap="palseq", norm=Normalize(vmin=0, vmax=100),
                   interpolation="nearest", aspect="equal")
    for i in range(2):
        for j in range(2):
            count = matrix[i, j]
            value = pct[i, j]
            ax.text(j, i, f"{count:,.0f}\n({value:.1f}%)", ha="center", va="center",
                    fontsize=8.2, color="white" if value > 55 else INK,
                    fontweight="bold")
    ax.set_xticks([0, 1], ["predicted\nnegative", "predicted\npositive"])
    ax.set_yticks([0, 1], ["pseudo-negative", "experimental\npositive"])
    ax.set_xlabel("Threshold = 0.5")
    ax.set_ylabel("Validation label")
    # A compact colour key is useful in SI but is not a grid or a second metric.
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.035, ticks=[0, 50, 100])
    cb.set_label("Row percentage", fontsize=7.5)
    cb.ax.tick_params(labelsize=7, width=0.5, length=2)
    cb.outline.set_linewidth(0.4)
    clean(ax); panel_letter(ax, letter)


def build_figure(a: dict, b: dict) -> tuple[plt.Figure, list[plt.Axes]]:
    """Build the four-panel SI figure without titles or explanatory footer."""
    style()
    fig = plt.figure(figsize=(18.3 * CM, 13.0 * CM), facecolor="white")
    gs = fig.add_gridspec(2, 2, left=0.10, right=0.96, bottom=0.10,
                          top=0.92, wspace=0.46, hspace=0.58)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    draw_roc(axes[0], a, BLUE, "a")
    draw_confusion(axes[1], a, "b")
    draw_roc(axes[2], b, ORANGE, "c")
    draw_confusion(axes[3], b, "d")
    # These two labels identify the model associated with each row; they are
    # not panel titles and are the only free-standing text retained.
    fig.text(0.10, 0.955, a["model"], ha="left", va="bottom", fontsize=9.5,
             fontweight="bold", color=BLUE)
    fig.text(0.10, 0.485, b["model"], ha="left", va="bottom", fontsize=9.5,
             fontweight="bold", color=ORANGE)
    return fig, axes


def render(output_dir: Path, raw_a: Path, raw_b: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    a = load_model(raw_a); b = load_model(raw_b)
    fig, _ = build_figure(a, b)
    stem = output_dir / "pu_models_roc_auc_confusion"
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white",
                metadata={"Title": "PU model validation ROC and confusion matrices"})
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)
    summary = {
        "figure": str(stem.with_suffix(".pdf")),
        "models": {
            "A": {k: a[k] for k in ["model", "auc_mean", "auc_sd", "n_bags", "n_per_bag", "thresholds"]},
            "B": {k: b[k] for k in ["model", "auc_mean", "auc_sd", "n_bags", "n_per_bag", "thresholds"]},
        },
        "protocol": "validation only; experimental positives plus sampled pseudo-negatives; no independent test split",
        "confusion_matrix": {
            "threshold": 0.5,
            "counts_are": "mean per bag",
            "row_percentages": "within validation label",
        },
    }
    (output_dir / "STATUS.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Standard PU-model validation diagnostics\n\n"
        "The four panels are mean ROC curves with bag-wise 95% envelopes and "
        "confusion matrices at threshold 0.5 for CGCNN-PU and MatterSim-1M-MLP-PU. "
        "The labels are experimental positives and sampled pseudo-negatives, not "
        "observed synthesis-failure labels; no independent test split was available.\n",
        encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-a", type=Path, default=DEFAULT_A)
    parser.add_argument("--raw-b", type=Path, default=DEFAULT_B)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(render(args.output_dir, args.raw_a, args.raw_b), ensure_ascii=False))


if __name__ == "__main__":
    main()
