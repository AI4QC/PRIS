#!/usr/bin/env python3
"""Retrospective paired diagnostics for ELEMENTA x0 screening rules.

The bootstrap sampling unit is a composition group (``rk``), not a structure.
This module is deliberately diagnostic: it compares already-opened test outputs
and therefore cannot turn a post-hoc hypothesis into a confirmatory result.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.next6_wbm_build import sha256_file


DECISIONS = {"KEEP", "REJECT", "ABSTAIN"}
PAIR_COLUMNS = (
    "sid",
    "rk",
    "delta_e",
    "exact_min",
    "near_min",
    "valuable",
    "high_energy",
    "decision",
    "formula",
    "alpha",
)
_SUFFIX_RE = re.compile(r"_([0-9]{2})$")


def _filter_predictions(
    predictions: pd.DataFrame,
    *,
    formula: str | None = None,
    alpha: float,
) -> pd.DataFrame:
    missing = set(PAIR_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    mask = np.isclose(
        pd.to_numeric(predictions["alpha"], errors="coerce").to_numpy(float),
        float(alpha),
        rtol=0.0,
        atol=1e-12,
    )
    if formula is not None:
        mask &= predictions["formula"].eq(formula).to_numpy(bool)
    result = predictions.loc[mask].copy()
    if result.empty:
        target = f"formula={formula!r}, " if formula is not None else ""
        raise ValueError(f"no predictions for {target}alpha={alpha}")
    if not set(result["decision"].astype(str)) <= DECISIONS:
        raise ValueError("unknown triage decision")
    return result


def _paired_rows(
    predictions: pd.DataFrame,
    *,
    baseline_formula: str,
    candidate_formula: str,
    alpha: float,
) -> pd.DataFrame:
    baseline = _filter_predictions(
        predictions, formula=baseline_formula, alpha=alpha
    ).sort_values("sid", kind="stable")
    candidate = _filter_predictions(
        predictions, formula=candidate_formula, alpha=alpha
    ).sort_values("sid", kind="stable")
    if baseline["sid"].duplicated().any() or candidate["sid"].duplicated().any():
        raise ValueError("each formula/alpha must contain one row per sid")
    if set(baseline["sid"]) != set(candidate["sid"]):
        raise ValueError("baseline and candidate sid sets differ")

    common = [
        "sid",
        "rk",
        "delta_e",
        "exact_min",
        "near_min",
        "valuable",
        "high_energy",
    ]
    paired = baseline[common + ["decision"]].merge(
        candidate[common + ["decision"]],
        on="sid",
        how="inner",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    for column in ("rk", "exact_min", "near_min", "valuable", "high_energy"):
        if not paired[f"{column}_baseline"].eq(
            paired[f"{column}_candidate"]
        ).all():
            raise ValueError(f"paired label mismatch for {column}")
    if not np.allclose(
        pd.to_numeric(paired["delta_e_baseline"], errors="coerce"),
        pd.to_numeric(paired["delta_e_candidate"], errors="coerce"),
        rtol=0.0,
        atol=1e-12,
        equal_nan=False,
    ):
        raise ValueError("paired label mismatch for delta_e")
    return pd.DataFrame(
        {
            "sid": paired["sid"],
            "rk": paired["rk_baseline"],
            "delta_e": paired["delta_e_baseline"],
            "exact_min": paired["exact_min_baseline"].astype(bool),
            "near_min": paired["near_min_baseline"].astype(bool),
            "valuable": paired["valuable_baseline"].astype(bool),
            "high_energy": paired["high_energy_baseline"].astype(bool),
            "decision_baseline": paired["decision_baseline"].astype(str),
            "decision_candidate": paired["decision_candidate"].astype(str),
        }
    ).sort_values(["rk", "sid"], kind="stable", ignore_index=True)


def _group_components(paired: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    rows: list[dict[str, float | str]] = []
    for rk, group in paired.groupby("rk", sort=True):
        row: dict[str, float | str] = {"rk": str(rk), "n": float(len(group))}
        exact = group["exact_min"].to_numpy(bool)
        near = group["near_min"].to_numpy(bool)
        valuable = group["valuable"].to_numpy(bool)
        high = group["high_energy"].to_numpy(bool)
        if not exact.any() or not near.any():
            raise ValueError(f"composition {rk!r} lacks exact/near minimum labels")
        for method in ("baseline", "candidate"):
            decision = group[f"decision_{method}"].to_numpy(str)
            reject = decision == "REJECT"
            abstain = decision == "ABSTAIN"
            row[f"{method}_reject"] = float(reject.sum())
            row[f"{method}_abstain"] = float(abstain.sum())
            row[f"{method}_exact_retained"] = float((exact & ~reject).any())
            row[f"{method}_near_retained"] = float((near & ~reject).any())
            row[f"{method}_valuable_retained"] = float((valuable & ~reject).sum())
            row[f"{method}_valuable_total"] = float(valuable.sum())
            row[f"{method}_high_rejected"] = float((high & reject).sum())
            row[f"{method}_high_total"] = float(high.sum())
            row[f"{method}_all_rejected"] = float(reject.all())
        rows.append(row)
    groups = pd.DataFrame(rows).sort_values("rk", kind="stable", ignore_index=True)
    ones = np.ones(len(groups), dtype=float)
    components: dict[str, dict[str, np.ndarray]] = {}
    for method in ("baseline", "candidate"):
        n = groups["n"].to_numpy(float)
        reject = groups[f"{method}_reject"].to_numpy(float)
        components[method] = {
            "dft_savings_num": reject,
            "dft_savings_den": n,
            "macro_dft_savings_num": reject / n,
            "macro_dft_savings_den": ones,
            "abstention_rate_num": groups[f"{method}_abstain"].to_numpy(float),
            "abstention_rate_den": n,
            "exact_min_retention_num": groups[
                f"{method}_exact_retained"
            ].to_numpy(float),
            "exact_min_retention_den": ones,
            "near_min_retention_num": groups[
                f"{method}_near_retained"
            ].to_numpy(float),
            "near_min_retention_den": ones,
            "valuable_item_recall_num": groups[
                f"{method}_valuable_retained"
            ].to_numpy(float),
            "valuable_item_recall_den": groups[
                f"{method}_valuable_total"
            ].to_numpy(float),
            "high_energy_removal_recall_num": groups[
                f"{method}_high_rejected"
            ].to_numpy(float),
            "high_energy_removal_recall_den": groups[
                f"{method}_high_total"
            ].to_numpy(float),
            "reject_high_energy_precision_num": groups[
                f"{method}_high_rejected"
            ].to_numpy(float),
            "reject_high_energy_precision_den": reject,
            "all_rejected_group_rate_num": groups[
                f"{method}_all_rejected"
            ].to_numpy(float),
            "all_rejected_group_rate_den": ones,
        }
    return components


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    total_denominator = float(np.sum(denominator))
    if total_denominator <= 0:
        return float("nan")
    return float(np.sum(numerator) / total_denominator)


def _ci(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return [float("nan"), float("nan")]
    low, high = np.quantile(finite, [0.025, 0.975])
    return [float(low), float(high)]


def paired_cluster_bootstrap(
    predictions: pd.DataFrame,
    *,
    baseline_formula: str,
    candidate_formula: str,
    alpha: float,
    n_resamples: int = 20_000,
    seed: int = 20260801,
    batch_size: int = 1_000,
) -> dict[str, object]:
    """Compare two frozen decision vectors with an rk-clustered bootstrap."""

    if n_resamples <= 0 or batch_size <= 0:
        raise ValueError("n_resamples and batch_size must be positive")
    paired = _paired_rows(
        predictions,
        baseline_formula=baseline_formula,
        candidate_formula=candidate_formula,
        alpha=alpha,
    )
    components = _group_components(paired)
    metric_names = sorted(
        key.removesuffix("_num")
        for key in components["baseline"]
        if key.endswith("_num")
    )
    n_groups = len(next(iter(components["baseline"].values())))
    rng = np.random.default_rng(seed)
    draws: dict[str, dict[str, list[np.ndarray]]] = {
        metric: {"baseline": [], "candidate": [], "difference": []}
        for metric in metric_names
    }
    remaining = int(n_resamples)
    while remaining:
        size = min(batch_size, remaining)
        indices = rng.integers(0, n_groups, size=(size, n_groups))
        for metric in metric_names:
            samples: dict[str, np.ndarray] = {}
            for method in ("baseline", "candidate"):
                numerator = components[method][f"{metric}_num"][indices].sum(axis=1)
                denominator = components[method][f"{metric}_den"][indices].sum(axis=1)
                sample = np.full(size, np.nan, dtype=float)
                np.divide(numerator, denominator, out=sample, where=denominator > 0)
                samples[method] = sample
                draws[metric][method].append(sample)
            draws[metric]["difference"].append(
                samples["candidate"] - samples["baseline"]
            )
        remaining -= size

    metrics: dict[str, dict[str, object]] = {}
    for metric in metric_names:
        baseline = _ratio(
            components["baseline"][f"{metric}_num"],
            components["baseline"][f"{metric}_den"],
        )
        candidate = _ratio(
            components["candidate"][f"{metric}_num"],
            components["candidate"][f"{metric}_den"],
        )
        baseline_draws = np.concatenate(draws[metric]["baseline"])
        candidate_draws = np.concatenate(draws[metric]["candidate"])
        difference_draws = np.concatenate(draws[metric]["difference"])
        metrics[metric] = {
            "baseline": baseline,
            "baseline_ci_95": _ci(baseline_draws),
            "candidate": candidate,
            "candidate_ci_95": _ci(candidate_draws),
            "difference": float(candidate - baseline),
            "difference_ci_95": _ci(difference_draws),
        }
    return {
        "method": "paired percentile bootstrap over rk composition clusters",
        "confidence_level": 0.95,
        "alpha": float(alpha),
        "baseline_formula": baseline_formula,
        "candidate_formula": candidate_formula,
        "n_rows": int(len(paired)),
        "n_groups": int(paired["rk"].nunique()),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "difference_direction": "candidate_minus_baseline",
        "metrics": metrics,
    }


def _attach_material_suffix(
    predictions: pd.DataFrame, labels: pd.DataFrame, *, alpha: float
) -> pd.DataFrame:
    filtered = _filter_predictions(predictions, alpha=alpha)
    required = {"sid", "material"}
    if required - set(labels.columns):
        raise ValueError("labels must contain sid and material")
    label_keys = labels[["sid", "material"]].copy()
    if label_keys["sid"].duplicated().any():
        raise ValueError("label sid must be unique")
    work = filtered.merge(label_keys, on="sid", how="left", validate="many_to_one")
    if work["material"].isna().any():
        raise ValueError("prediction sid missing from labels")
    work["suffix"] = work["material"].astype(str).map(
        lambda value: (_SUFFIX_RE.search(value).group(1) if _SUFFIX_RE.search(value) else "other")
    )
    return work


def suffix_diagnostics(
    predictions: pd.DataFrame, labels: pd.DataFrame, *, alpha: float
) -> pd.DataFrame:
    """Summarize decisions and DFT labels by generator-order suffix."""

    work = _attach_material_suffix(predictions, labels, alpha=alpha)
    records: list[dict[str, object]] = []
    for (formula, suffix), group in work.groupby(["formula", "suffix"], sort=True):
        decision = group["decision"].astype(str).to_numpy()
        reject = decision == "REJECT"
        abstain = decision == "ABSTAIN"
        high = group["high_energy"].astype(bool).to_numpy()
        delta = pd.to_numeric(group["delta_e"], errors="coerce").to_numpy(float)
        score = pd.to_numeric(group.get("score"), errors="coerce").to_numpy(float)
        records.append(
            {
                "formula": str(formula),
                "alpha": float(alpha),
                "suffix": str(suffix),
                "n": int(len(group)),
                "n_reject": int(reject.sum()),
                "n_abstain": int(abstain.sum()),
                "reject_rate": float(reject.mean()),
                "abstention_rate": float(abstain.mean()),
                "mean_delta_e": float(np.mean(delta)),
                "median_delta_e": float(np.median(delta)),
                "exact_min_rate": float(group["exact_min"].astype(bool).mean()),
                "valuable_rate": float(group["valuable"].astype(bool).mean()),
                "high_energy_rate": float(high.mean()),
                "high_energy_removal_recall": (
                    float((high & reject).sum() / high.sum()) if high.any() else np.nan
                ),
                "reject_high_energy_precision": (
                    float((high & reject).sum() / reject.sum()) if reject.any() else np.nan
                ),
                "mean_score": float(np.nanmean(score)),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["formula", "suffix"], kind="stable", ignore_index=True
    )


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def score_energy_correlations(
    predictions: pd.DataFrame, labels: pd.DataFrame, *, alpha: float
) -> pd.DataFrame:
    """Report raw and suffix-adjusted rank correlations for each formula."""

    work = _attach_material_suffix(predictions, labels, alpha=alpha)
    records: list[dict[str, object]] = []
    for formula, group in work.groupby("formula", sort=True):
        score = pd.to_numeric(group["score"], errors="coerce")
        delta = pd.to_numeric(group["delta_e"], errors="coerce")
        finite = score.notna() & delta.notna() & np.isfinite(score) & np.isfinite(delta)
        clean = group.loc[finite, ["suffix"]].copy()
        clean["score_rank"] = score.loc[finite].rank(method="average").to_numpy(float)
        clean["delta_rank"] = delta.loc[finite].rank(method="average").to_numpy(float)
        score_rank = clean["score_rank"].to_numpy(float)
        delta_rank = clean["delta_rank"].to_numpy(float)
        score_residual = (
            clean["score_rank"]
            - clean.groupby("suffix", sort=False)["score_rank"].transform("mean")
        ).to_numpy(float)
        delta_residual = (
            clean["delta_rank"]
            - clean.groupby("suffix", sort=False)["delta_rank"].transform("mean")
        ).to_numpy(float)
        records.append(
            {
                "formula": str(formula),
                "alpha": float(alpha),
                "n_finite": int(len(clean)),
                "n_suffixes": int(clean["suffix"].nunique()),
                "raw_spearman": _pearson(score_rank, delta_rank),
                "suffix_residualized_rank_correlation": _pearson(
                    score_residual, delta_residual
                ),
            }
        )
    return pd.DataFrame(records).sort_values("formula", kind="stable", ignore_index=True)


def run_diagnostics(
    predictions_path: Path,
    labels_path: Path,
    output_dir: Path,
    *,
    baseline_formula: str,
    candidate_formula: str,
    alpha: float = 0.03,
    n_resamples: int = 20_000,
    seed: int = 20260801,
) -> dict[str, object]:
    predictions_path = Path(predictions_path)
    labels_path = Path(labels_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions = pd.read_parquet(predictions_path)
    labels = pd.read_parquet(labels_path)
    paired = paired_cluster_bootstrap(
        predictions,
        baseline_formula=baseline_formula,
        candidate_formula=candidate_formula,
        alpha=alpha,
        n_resamples=n_resamples,
        seed=seed,
    )
    paired_path = output_dir / "paired_cluster_bootstrap.json"
    paired_path.write_text(
        json.dumps(paired, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    suffix_path = output_dir / "suffix_diagnostics.parquet"
    suffix_diagnostics(predictions, labels, alpha=alpha).to_parquet(
        suffix_path, index=False
    )
    correlation_path = output_dir / "score_energy_correlations.parquet"
    score_energy_correlations(predictions, labels, alpha=alpha).to_parquet(
        correlation_path, index=False
    )
    manifest: dict[str, object] = {
        "diagnostic_status": "post-hoc retrospective comparison after test exposure",
        "alpha": float(alpha),
        "baseline_formula": baseline_formula,
        "candidate_formula": candidate_formula,
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "inputs_sha256": {
            predictions_path.name: sha256_file(predictions_path),
            labels_path.name: sha256_file(labels_path),
        },
        "outputs_sha256": {
            paired_path.name: sha256_file(paired_path),
            suffix_path.name: sha256_file(suffix_path),
            correlation_path.name: sha256_file(correlation_path),
        },
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args(argv)
    manifest = run_diagnostics(
        args.predictions,
        args.labels,
        args.output,
        baseline_formula=args.baseline,
        candidate_formula=args.candidate,
        alpha=args.alpha,
        n_resamples=args.resamples,
        seed=args.seed,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "paired_cluster_bootstrap",
    "run_diagnostics",
    "score_energy_correlations",
    "suffix_diagnostics",
]
