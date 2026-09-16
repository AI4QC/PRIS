#!/usr/bin/env python3
"""Build standalone, validation-only SI audit figures and tables.

The first pass uses the immutable per-bag ``DONE best vAUC`` lines from the
remote training logs.  If ``bag_validation_metrics.csv`` from the remote
re-evaluation is present, its PR-AUC and score summaries are added without
changing the reported validation protocol.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

A_AUC = [
    .9779,.9763,.9768,.9765,.9749,.9762,.9778,.9756,.9771,.9760,
    .9771,.9765,.9766,.9762,.9749,.9772,.9766,.9774,.9753,.9766,
    .9772,.9754,.9766,.9764,.9780,.9759,.9760,.9770,.9772,.9765,
    .9764,.9771,.9764,.9769,.9755,.9763,.9768,.9758,.9758,.9762,
    .9774,.9759,.9768,.9770,.9766,.9771,.9753,.9775,.9781,.9763,
]
B_AUC = [
    .9468,.9479,.9457,.9467,.9449,.9459,.9446,.9465,.9484,.9467,
    .9470,.9479,.9474,.9470,.9461,.9474,.9465,.9474,.9462,.9469,
    .9473,.9473,.9486,.9456,.9482,.9468,.9457,.9467,.9446,.9474,
    .9466,.9464,.9464,.9464,.9465,.9456,.9460,.9454,.9459,.9471,
    .9471,.9464,.9468,.9459,.9469,.9463,.9465,.9459,.9478,.9443,
]

MODELS = {
    "CGCNN-PU OOB": {
        "short": "CGCNN-PU",
        "auc": A_AUC,
        "color": "#0072B2",
        "log": "/data1/home/hzxy10/csllm/logs/train_200535_1.out … train_200537_50.out",
        "architecture": "CGCNN trained from scratch; 50 PU bags; OOB full-pool score",
    },
    "MatterSim-1M-MLP-PU OOB": {
        "short": "MatterSim-1M-MLP-PU",
        "auc": B_AUC,
        "color": "#D55E00",
        "log": "/data1/home/hzxy10/csllm/logs/headB_200736.out",
        "architecture": "Frozen MatterSim-v1.0.0-1M embedding + 50 MLP PU heads; OOB full-pool score",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    # Immutable source-log values; each bag is one row.
    rows = []
    for model, cfg in MODELS.items():
        for bag, auc in enumerate(cfg["auc"], 1):
            rows.append({
                "model": model,
                "bag": bag,
                "validation_metric": "ROC-AUC",
                "roc_auc": auc,
                "metric_protocol": "per-bag validation (positive holdout + sampled pseudo-negative)",
                "independent_test": False,
            })
    write_csv(ROOT / "per_bag_validation_auc.csv", rows)

    summaries = {}
    for model, cfg in MODELS.items():
        x = np.asarray(cfg["auc"], dtype=float)
        summaries[model] = {
            "n_bags": len(x),
            "roc_auc_mean": float(x.mean()),
            "roc_auc_sd": float(x.std(ddof=1)),
            "roc_auc_median": float(np.median(x)),
            "roc_auc_min": float(x.min()),
            "roc_auc_max": float(x.max()),
            "roc_auc_range": [float(x.min()), float(x.max())],
            "metric_status": "validation-only; no independent test split",
            "log_source": cfg["log"],
            "architecture": cfg["architecture"],
        }
    # Merge exact PR-AUC/score summaries if the remote recomputation has landed.
    # Prefer the corrected MatterSim rerun; retain the old remote location as
    # a backwards-compatible fallback for earlier audit snapshots.
    metrics_path = ROOT / "raw_B_opt" / "bag_validation_metrics.csv"
    if not metrics_path.exists():
        metrics_path = ROOT / "remote" / "bag_validation_metrics.csv"
    if metrics_path.exists():
        remote = pd.read_csv(metrics_path)
        model_alias = {
            "CGCNN-PU": "CGCNN-PU OOB",
            "MatterSim-1M-MLP-PU": "MatterSim-1M-MLP-PU OOB",
        }
        for model, sub in remote.groupby("model"):
            target = model_alias.get(model, model)
            if target not in summaries:
                continue
            summaries[target]["pr_auc_mean"] = float(sub.pr_auc.mean())
            summaries[target]["pr_auc_sd"] = float(sub.pr_auc.std(ddof=1))
            summaries[target]["evaluated_rows_min"] = int(sub.n.min())
            summaries[target]["evaluated_rows_max"] = int(sub.n.max())
    summary_payload = {
        "independent_test_available": False,
        "validation_rows_per_bag": 39596,
        "validation_positive_per_bag": 19798,
        "validation_pseudo_negative_per_bag": 19798,
        "raw_score_re_evaluation": {
            "slurm_job": 209663 if (ROOT / "raw_B_opt").exists() else 209577,
            "status": "completed_corrected_embedding_path" if (ROOT / "raw_B_opt").exists() else "stopped_before_matterSim_pass",
            "pr_auc_available": bool((ROOT / "raw_B_opt" / "bag_validation_metrics.csv").exists()),
            "reason": "Each embedding shard materialized once before validation-row extraction." if (ROOT / "raw_B_opt").exists() else "The remote job stalled after the CGCNN pass; no raw prediction curve tables were produced.",
        },
        "models": summaries,
    }
    with (ROOT / "model_summary.json").open("w") as f:
        json.dump(summary_payload, f, indent=2, sort_keys=True)

    # A compact per-model figure: validation AUC distribution plus a clearly
    # separated protocol panel.  This is intentionally not a ROC claim: the
    # available source logs contain AUC values, while raw validation scores are
    # generated asynchronously by evaluate_remote.py.
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    for model, cfg in MODELS.items():
        x = np.asarray(cfg["auc"], dtype=float)
        fig, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
        ax.boxplot(x, vert=True, widths=0.36, patch_artist=True,
                   boxprops={"facecolor": cfg["color"], "alpha": .20,
                             "edgecolor": cfg["color"]},
                   medianprops={"color": "#111111", "linewidth": 1.5},
                   whiskerprops={"color": cfg["color"]},
                   capprops={"color": cfg["color"]},
                   flierprops={"marker": "", "markersize": 0})
        rng = np.random.default_rng(20260823)
        ax.scatter(1 + rng.normal(0, .045, len(x)), x, s=24,
                   color=cfg["color"], alpha=.75, edgecolor="white", linewidth=.3,
                   zorder=3)
        ax.set_xlim(.65, 1.35)
        ax.set_ylim(max(.45, x.min() - .004), min(1.0, x.max() + .004))
        ax.set_xticks([1], ["50 PU bags"])
        ax.set_ylabel("Validation ROC-AUC")
        s = summaries[model]
        ax.set_title(model, loc="left", fontweight="bold", fontsize=12)
        ax.text(.02, .98,
                f"mean {s['roc_auc_mean']:.4f} ± {s['roc_auc_sd']:.4f} (SD)\n"
                f"range {s['roc_auc_min']:.4f}–{s['roc_auc_max']:.4f}\n"
                "n=39,596 per bag (19,798 + 19,798)\n"
                "validation labels: experimental positives vs\n"
                "sampled unlabeled pseudo-negatives",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": .9,
                      "boxstyle": "round,pad=.35"})
        ax.text(.98, .04, "No independent test split\nOOB refers to full-pool scoring",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=8.5,
                color="#444444")
        stem = "cgc nn" if False else cfg["short"].lower().replace("-", "_")
        stem = stem.replace(" ", "_")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(ROOT / f"figure_{stem}_validation_performance.{ext}", dpi=220)
        plt.close(fig)

    # Human-readable status and provenance.
    status = "COMPLETE_WITH_VALIDATION_ONLY"
    (ROOT / "STATUS.md").write_text(
        "# PU model performance audit\n\n"
        f"Status: **{status}**\n\n"
        "This package reports per-bag validation performance from the frozen "
        "CSAgent-v2 artifacts. No independent test split or observed negative "
        "synthesis-failure labels exist in the source checkout. The word OOB "
        "refers to the full-pool CLscore construction, not to an independent "
        "test accuracy estimate.\n\n"
        "The standard ROC/AUC + threshold=0.5 confusion-matrix figure is in "
        "`standard_roc_confusion/`. The corrected MatterSim rerun (Slurm 209663) "
        "is stored in `raw_B_opt/`; all values remain validation-only because "
        "no independent test split exists.\n",
        encoding="utf-8")
    (ROOT / "README.md").write_text(
        "# PU learning performance audit\n\n"
        "An independent performance package; it does not modify the paper.\n\n"
        "- `CGCNN-PU OOB`: CGCNN trained from scratch, 50 PU bags.\n"
        "- `MatterSim-1M-MLP-PU OOB`: frozen MatterSim-v1.0.0-1M embeddings and 50 MLP heads.\n"
        "- `per_bag_validation_auc.csv`: the best validation ROC-AUC saved per bag in the raw logs.\n"
        "- `figure_*_validation_performance.{png,pdf,svg}`: per-model validation distributions, "
        "labelled explicitly as having no independent test split.\n"
        "- `standard_roc_confusion/pu_models_roc_auc_confusion.pdf`: ROC/AUC and the "
        "threshold=0.5 confusion matrix.\n"
        "- `raw_B_opt/`: the MatterSim 50-bag output after fixing the repeated decompression "
        "of the embeddings.\n"
        "- `performance_audit_report.md/.pdf`: provenance, split protocol, metrics and the "
        "recommendation on aggregating A and B.\n"
        "- `provenance_audit.json`: model provenance at file and line level, with checkable "
        "evidence.\n"
        "- `remote/`: the protocol and log provenance audit; the state of the earlier 209577 "
        "is kept here.\n\n"
        "The validation labels are experimental positives and per-bag sampled unlabelled "
        "pseudo-negatives, and may not be read as a real detection rate for failed synthesis;\n"
        "OOB means only the out-of-bag average used when scoring the whole pool, not an "
        "independent test set.\n",
        encoding="utf-8")

    manifest = {
        "status": status,
        "files": {},
        "source_artifacts": {
            "A_logs": "/data1/home/hzxy10/csllm/logs/train_200535_1.out … train_200537_50.out",
            "B_log": "/data1/home/hzxy10/csllm/logs/headB_200736.out",
            "split_script": "pu_screening_v2/02_split_bagging.py",
            "A_score_script": "pu_screening_v2/04_predict_clscore.py",
            "B_score_script": "pu_screening_v2/07_embed_pu_head.py",
        },
        "independent_test_available": False,
    }
    # Include nested provenance files as well as top-level deliverables.  The
    # manifest is intended to make the standalone package auditable after it is
    # copied away from this checkout.
    for p in ROOT.rglob("*"):
        if p.is_file() and p.name not in {"manifest.json"}:
            key = str(p.relative_to(ROOT))
            manifest["files"][key] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
