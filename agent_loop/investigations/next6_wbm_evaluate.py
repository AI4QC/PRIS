#!/usr/bin/env python3
"""Apply one frozen x0 rule to the WBM test partition exactly once."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import beta

from src.next6_wbm_build import sha256_file
from src.next6_wbm_calibrate import CandidateSpec, score_candidate
from src.next6_wbm_protocol import clopper_pearson_upper, evaluate_triage


def apply_frozen_rule(features: pd.DataFrame, frozen: dict[str, object]) -> pd.DataFrame:
    """Return scores and fail-open decisions without looking at any labels."""

    formula = dict(frozen["formula"])
    spec = CandidateSpec(
        name=str(formula["name"]),
        family=str(formula["family"]),
        mode=str(formula["mode"]),
        pack_low=float(formula["pack_low"]),
        pack_high=float(formula["pack_high"]),
        pack_weight=float(formula["pack_weight"]),
        scale_penalty=float(formula["scale_penalty"]),
        complexity=int(formula["complexity"]),
    )
    threshold = float(dict(frozen["threshold"])["threshold"])
    score = score_candidate(features, spec)
    decision = np.full(len(features), "KEEP", dtype=object)
    decision[~np.isfinite(score)] = "ABSTAIN"
    decision[np.isfinite(score) & (score >= threshold)] = "REJECT"
    return pd.DataFrame(
        {
            "material_id": features["material_id"].astype(str).to_numpy(),
            "score": score,
            "decision": decision,
        }
    )


def _binomial_lower(successes: int, trials: int, confidence: float) -> float:
    if trials == 0 or successes == 0:
        return 0.0
    return float(beta.ppf(1.0 - confidence, successes, trials - successes + 1))


def classification_metrics(
    stable: Iterable[bool],
    decisions: Iterable[str],
    scores: Iterable[float],
    *,
    top_ks: Sequence[int] = (10_000,),
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Report stable classification and actual DFT triage accounting."""

    y = np.asarray(stable, dtype=bool)
    decision = np.asarray(decisions, dtype=object)
    score = np.asarray(scores, dtype=float)
    if len(y) != len(decision) or len(y) != len(score):
        raise ValueError("stable, decisions, and scores must align")
    triage = evaluate_triage(y, decision)
    predicted_stable = decision != "REJECT"
    true_positive = int((predicted_stable & y).sum())
    predicted_positive = int(predicted_stable.sum())
    stable_precision = (
        true_positive / predicted_positive if predicted_positive else float("nan")
    )
    stable_recall = float(triage["stable_recall"])
    stable_f1 = (
        2 * stable_precision * stable_recall / (stable_precision + stable_recall)
        if stable_precision + stable_recall > 0
        else 0.0
    )
    prevalence = float(y.mean()) if len(y) else float("nan")
    metrics: dict[str, float | int] = {
        **triage,
        "prevalence": prevalence,
        "stable_precision": float(stable_precision),
        "stable_f1": float(stable_f1),
        "daf": float(stable_precision / prevalence) if prevalence > 0 else float("nan"),
        "stable_recall_lower": float(
            1.0
            - clopper_pearson_upper(
                int(triage["stable_false_rejects"]),
                int(triage["n_stable"]),
                confidence=confidence,
            )
        ),
        "dft_savings_lower": _binomial_lower(
            int(triage["n_reject"]), len(y), confidence
        ),
    }
    order = np.argsort(np.where(np.isfinite(score), score, np.inf), kind="stable")
    for requested in top_ks:
        k = min(int(requested), len(y))
        metrics[f"top_{requested}_precision"] = (
            float(y[order[:k]].mean()) if k else float("nan")
        )
    return metrics


def _cluster_bootstrap(
    predictions: pd.DataFrame,
    *,
    reps: int,
    seed: int = 20260801,
) -> dict[str, float]:
    reject = predictions["decision"].eq("REJECT")
    stable = predictions["stable"].astype(bool)
    summary = (
        predictions.assign(
            _n=1,
            _stable=stable.astype(int),
            _reject=reject.astype(int),
            _stable_false_reject=(reject & stable).astype(int),
        )
        .groupby("formula_key", sort=False)
        .agg(
            n=("_n", "sum"),
            n_stable=("_stable", "sum"),
            n_reject=("_reject", "sum"),
            stable_false_rejects=("_stable_false_reject", "sum"),
        )
    )
    if summary.empty or reps <= 0:
        return {}
    values = summary.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    recalls = np.empty(reps, dtype=float)
    savings = np.empty(reps, dtype=float)
    for index in range(reps):
        sample = values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
        n, n_stable, n_reject, false_rejects = sample
        recalls[index] = (
            1.0 - false_rejects / n_stable if n_stable > 0 else np.nan
        )
        savings[index] = n_reject / n
    return {
        "cluster_bootstrap_reps": int(reps),
        "stable_recall_cluster_lo": float(np.nanquantile(recalls, 0.05)),
        "stable_recall_cluster_hi": float(np.nanquantile(recalls, 0.95)),
        "dft_savings_cluster_lo": float(np.quantile(savings, 0.05)),
        "dft_savings_cluster_hi": float(np.quantile(savings, 0.95)),
    }


def run_test_evaluation(
    artifact_dir: Path,
    frozen_rule_path: Path,
    output_dir: Path,
    *,
    bootstrap_reps: int = 1_000,
    allow_rerun: bool = False,
) -> dict[str, float | int]:
    """Open the outer test labels once, evaluate, and log that opening."""

    artifact_dir = Path(artifact_dir)
    frozen_rule_path = Path(frozen_rule_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_x_path = artifact_dir / "test_x0_features.parquet"
    test_y_path = artifact_dir / "test_labels.parquet"
    opening_path = artifact_dir / "TEST_OPENINGS.jsonl"
    if opening_path.exists() and opening_path.stat().st_size > 0 and not allow_rerun:
        raise RuntimeError(f"test labels already opened: {opening_path}")

    frozen = json.loads(frozen_rule_path.read_text(encoding="utf-8"))
    features = pd.read_parquet(test_x_path)
    prediction = apply_frozen_rule(features, frozen)
    opening = {
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "single frozen-rule WBM test evaluation",
        "frozen_rule_sha256": sha256_file(frozen_rule_path),
        "test_labels_sha256": sha256_file(test_y_path),
    }
    with opening_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(opening, sort_keys=True) + "\n")

    labels = pd.read_parquet(test_y_path)
    if prediction["material_id"].duplicated().any() or labels["material_id"].duplicated().any():
        raise ValueError("test material_id must be unique")
    joined = labels.merge(prediction, on="material_id", how="inner", validate="one_to_one")
    if len(joined) != len(labels) or len(joined) != len(prediction):
        raise ValueError("test feature and label material_id sets differ")

    metrics = classification_metrics(
        joined["stable"].to_numpy(dtype=bool),
        joined["decision"].to_numpy(dtype=object),
        joined["score"].to_numpy(dtype=float),
    )
    metrics.update(_cluster_bootstrap(joined, reps=bootstrap_reps))
    metrics["n_compositions"] = int(joined["formula_key"].nunique())
    joined.to_parquet(output_dir / "test_predictions.parquet", index=False)
    metrics_path = output_dir / "test_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "opening": opening,
        "test_predictions_sha256": sha256_file(output_dir / "test_predictions.parquet"),
        "test_metrics_sha256": sha256_file(metrics_path),
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--frozen-rule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=1_000)
    parser.add_argument("--allow-rerun", action="store_true")
    args = parser.parse_args()
    metrics = run_test_evaluation(
        args.artifacts,
        args.frozen_rule,
        args.output,
        bootstrap_reps=args.bootstrap_reps,
        allow_rerun=args.allow_rerun,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
