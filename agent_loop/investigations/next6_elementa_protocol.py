#!/usr/bin/env python3
"""Leakage-resistant group protocol for the exploratory ELEMENTA x0 migration."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import beta


VALID_DECISIONS = frozenset({"KEEP", "REJECT", "ABSTAIN"})
NEAR_MIN_EV_PER_ATOM = 0.001
VALUABLE_EV_PER_ATOM = 0.05
HIGH_ENERGY_EV_PER_ATOM = 0.20


def _hash_bucket(text: str, modulus: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def elementa_stage(composition_key: str) -> str:
    """Assign an entire composition to the frozen 20/20/20/40 workflow."""

    key = str(composition_key)
    outer = _hash_bucket(f"elementa-x0-migration-v1|outer|{key}", 10_000)
    if outer < 4_000:
        inner = _hash_bucket(f"elementa-x0-migration-v1|inner|{key}", 2)
        return "search_calibration" if inner == 0 else "formula_selection"
    if outer < 6_000:
        return "threshold_calibration"
    return "test"


def attach_energy_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Attach within-composition endpoint labels without changing row order."""

    required = {"rk", "e_per_atom"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"missing label columns: {sorted(missing)}")
    out = data.copy()
    energy = pd.to_numeric(out["e_per_atom"], errors="coerce")
    if not np.isfinite(energy.to_numpy(dtype=float)).all():
        raise ValueError("endpoint energy labels must all be finite")
    out["delta_e"] = energy - energy.groupby(out["rk"], sort=False).transform("min")
    tolerance = 1e-12
    out["exact_min"] = out["delta_e"] <= tolerance
    out["near_min"] = out["delta_e"] <= NEAR_MIN_EV_PER_ATOM + tolerance
    out["valuable"] = out["delta_e"] <= VALUABLE_EV_PER_ATOM + tolerance
    out["high_energy"] = out["delta_e"] >= HIGH_ENERGY_EV_PER_ATOM - tolerance
    return out


def group_conformal_threshold(
    calibration: pd.DataFrame,
    *,
    alpha: float,
    group_column: str = "rk",
    valuable_column: str = "valuable",
    score_column: str = "score",
    supported_column: str = "supported",
    within_group: str = "max",
) -> dict[str, float | int | str]:
    """Calibrate a group-level threshold protecting every scored valuable row.

    Unsupported or non-finite rows abstain and therefore cannot be rejected.
    The deployment rule is strictly ``score > threshold`` so calibration ties
    remain protected.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if within_group not in {"max", "min"}:
        raise ValueError("within_group must be 'max' or 'min'")
    required = {group_column, valuable_column, score_column, supported_column}
    missing = required - set(calibration.columns)
    if missing:
        raise ValueError(f"missing calibration columns: {sorted(missing)}")
    maxima: list[float] = []
    for _, group in calibration.groupby(group_column, sort=False):
        score = pd.to_numeric(group[score_column], errors="coerce").to_numpy(dtype=float)
        eligible = (
            group[valuable_column].to_numpy(dtype=bool)
            & group[supported_column].to_numpy(dtype=bool)
            & np.isfinite(score)
        )
        if eligible.any():
            reducer = np.max if within_group == "max" else np.min
            maxima.append(float(reducer(score[eligible])))
        else:
            maxima.append(-np.inf)
    n_groups = len(maxima)
    order_index = int(math.ceil((n_groups + 1) * (1.0 - alpha)))
    threshold = (
        float("inf")
        if n_groups == 0 or order_index > n_groups
        else float(np.sort(np.asarray(maxima, dtype=float))[order_index - 1])
    )
    return {
        "alpha": float(alpha),
        "n_groups": n_groups,
        "order_index": order_index,
        "threshold": threshold,
        "within_group": within_group,
    }


def apply_group_threshold(
    scores: Iterable[float],
    supported: Iterable[bool],
    threshold: float,
) -> np.ndarray:
    """Apply fail-open three-state decisions; larger scores are worse."""

    score = np.asarray(scores, dtype=float)
    support = np.asarray(supported, dtype=bool)
    if score.ndim != 1 or support.ndim != 1 or len(score) != len(support):
        raise ValueError("scores and supported must be aligned one-dimensional arrays")
    finite_support = support & np.isfinite(score)
    decision = np.full(len(score), "ABSTAIN", dtype=object)
    decision[finite_support] = "KEEP"
    decision[finite_support & (score > float(threshold))] = "REJECT"
    return decision


def _binomial_lower(successes: int, trials: int, confidence: float = 0.95) -> float:
    if trials <= 0 or successes <= 0:
        return 0.0
    return float(beta.ppf(1.0 - confidence, successes, trials - successes + 1))


def evaluate_group_triage(data: pd.DataFrame) -> dict[str, float | int]:
    """Evaluate DFT savings, retention, and regret with compositions as groups."""

    required = {"rk", "delta_e", "decision"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"missing evaluation columns: {sorted(missing)}")
    decision = data["decision"].astype(str).to_numpy(dtype=object)
    unexpected = set(decision.tolist()) - VALID_DECISIONS
    if unexpected:
        raise ValueError(f"unexpected decisions: {sorted(unexpected)}")
    delta = pd.to_numeric(data["delta_e"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(delta).all() or (delta < -1e-10).any():
        raise ValueError("delta_e must be finite and nonnegative")
    reject = decision == "REJECT"
    abstain = decision == "ABSTAIN"
    exact = delta <= 1e-12
    near = delta <= NEAR_MIN_EV_PER_ATOM + 1e-12
    valuable = delta <= VALUABLE_EV_PER_ATOM + 1e-12
    high = delta >= HIGH_ENERGY_EV_PER_ATOM - 1e-12

    work = pd.DataFrame(
        {
            "rk": data["rk"].astype(str).to_numpy(),
            "delta_e": delta,
            "reject": reject,
            "exact": exact,
            "near": near,
            "valuable": valuable,
        }
    )
    group_rows: list[dict[str, float | bool | int]] = []
    for _, group in work.groupby("rk", sort=False):
        survivors = ~group["reject"].to_numpy(dtype=bool)
        group_delta = group["delta_e"].to_numpy(dtype=float)
        group_rows.append(
            {
                "size": len(group),
                "n_reject": int(group["reject"].sum()),
                "n_survivors": int(survivors.sum()),
                "exact_retained": bool(
                    np.any(group["exact"].to_numpy(dtype=bool) & survivors)
                ),
                "near_retained": bool(
                    np.any(group["near"].to_numpy(dtype=bool) & survivors)
                ),
                "valuable_all_retained": bool(
                    not np.any(group["valuable"].to_numpy(dtype=bool) & ~survivors)
                ),
                "regret": float(np.min(group_delta[survivors])) if survivors.any() else np.inf,
            }
        )
    groups = pd.DataFrame(group_rows)
    n = len(data)
    n_groups = len(groups)
    n_reject = int(reject.sum())
    n_abstain = int(abstain.sum())
    exact_success = int(groups["exact_retained"].sum()) if n_groups else 0
    near_success = int(groups["near_retained"].sum()) if n_groups else 0
    valuable_group_success = int(groups["valuable_all_retained"].sum()) if n_groups else 0
    valuable_count = int(valuable.sum())
    valuable_reject = int((valuable & reject).sum())
    high_count = int(high.sum())
    high_reject = int((high & reject).sum())
    regret = groups["regret"].to_numpy(dtype=float) if n_groups else np.asarray([])
    all_rejected = int(np.isinf(regret).sum())
    finite_regret = regret[np.isfinite(regret)]
    if all_rejected:
        regret_p95 = regret_max = float("inf")
    elif len(finite_regret):
        regret_p95 = float(np.quantile(finite_regret, 0.95))
        regret_max = float(np.max(finite_regret))
    else:
        regret_p95 = regret_max = float("nan")

    exact_retention = exact_success / n_groups if n_groups else float("nan")
    near_retention = near_success / n_groups if n_groups else float("nan")
    valuable_group_rate = valuable_group_success / n_groups if n_groups else float("nan")
    valuable_recall = (
        1.0 - valuable_reject / valuable_count if valuable_count else float("nan")
    )
    return {
        "n": n,
        "n_groups": n_groups,
        "n_reject": n_reject,
        "n_abstain": n_abstain,
        "coverage": float(1.0 - n_abstain / n) if n else 0.0,
        "dft_savings": float(n_reject / n) if n else 0.0,
        "macro_dft_savings": float((groups["n_reject"] / groups["size"]).mean())
        if n_groups
        else 0.0,
        "abstention_rate": float(n_abstain / n) if n else 0.0,
        "exact_min_retention": float(exact_retention),
        "exact_min_retention_lower": _binomial_lower(exact_success, n_groups),
        "near_min_retention": float(near_retention),
        "near_min_retention_lower": _binomial_lower(near_success, n_groups),
        "valuable_item_recall": float(valuable_recall),
        "valuable_all_retained_group_rate": float(valuable_group_rate),
        "valuable_group_retention_lower": _binomial_lower(
            valuable_group_success, n_groups
        ),
        "high_energy_removal_recall": float(high_reject / high_count)
        if high_count
        else float("nan"),
        "reject_high_energy_precision": float(high_reject / n_reject)
        if n_reject
        else float("nan"),
        "all_rejected_groups": all_rejected,
        "survivors_per_group_min": int(groups["n_survivors"].min()) if n_groups else 0,
        "survivors_per_group_median": float(groups["n_survivors"].median())
        if n_groups
        else 0.0,
        "regret_median": float(np.median(finite_regret)) if len(finite_regret) else float("nan"),
        "regret_p95": regret_p95,
        "regret_max": regret_max,
    }


__all__ = [
    "HIGH_ENERGY_EV_PER_ATOM",
    "NEAR_MIN_EV_PER_ATOM",
    "VALUABLE_EV_PER_ATOM",
    "apply_group_threshold",
    "attach_energy_labels",
    "elementa_stage",
    "evaluate_group_triage",
    "group_conformal_threshold",
]
