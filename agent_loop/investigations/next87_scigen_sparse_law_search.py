"""NEXT87 finite, physics-directed sparse-law search for SCIGEN x0 structures."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from itertools import combinations, product
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.next23_relaxation_rule import wilson_lower_bound
from src.next23_evaluate import _roc_auc
from src.next85_scigen_label_free_features import (
    CATALOGUE_NAME as FEATURE_CATALOGUE_NAME,
    FEATURE_NAMES,
    MANIFEST_NAME as FEATURE_MANIFEST_NAME,
    PROTOCOL as FEATURE_PROTOCOL,
)
from src.next86_scigen_endpoint_router import (
    ENDPOINT_NAME,
    MANIFEST_NAME as ENDPOINT_MANIFEST_NAME,
    PROTOCOL as ENDPOINT_PROTOCOL,
)
from src.next86_scigen_term_catalogue import (
    CATALOGUE_NAME as TERM_CATALOGUE_NAME,
    MANIFEST_NAME as TERM_MANIFEST_NAME,
    PROTOCOL as TERM_PROTOCOL,
)


PROTOCOL = "2026-08-03-next87-scigen-sparse-hinge-search-v1"
FORMULA_KIND = "nonnegative_hinge_sum"
MISSING_POLICY = "KEEP"
WEIGHT_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
GROUP_FOLD_SALT = "NEXT87_SCIGEN_GROUP_FOLD_V1"
GROUP_FOLDS = 5
DEFAULT_GATES = {
    "coverage_lower": 0.90,
    "protected_recall_lower": 0.95,
    "severe_rejection_precision_lower": 0.80,
    "savings_lower": 0.02,
    "pooled_extreme_auc": 0.75,
    "macro_lattice_auc": 0.65,
    "worst_lattice_auc": 0.55,
    "evaluable_lattices": 8,
}
PAIR_SHORTLIST = 16
TRIPLE_SHORTLIST = 12
MAX_TERMS_PER_PHYSICS_GROUP = 3
FOLD_MIN_PRECISION = 0.70
FOLD_MIN_PROTECTED_RECALL = 0.93
MANIFEST_NAME = "MANIFEST.json"
FORMULA_NAME = "NEXT87_FROZEN_FORMULA.json"
EVALUATION_NAME = "NEXT87_DISCOVERY_EVALUATION.json"
FOLD_DIAGNOSTICS_NAME = "NEXT87_FOLD_DIAGNOSTICS.json"
SEARCH_RECORD_NAME = "next87_complete_candidate_search.parquet"
PREDICTION_NAMES = {
    role: f"next87_frozen_predictions_{role}.parquet" for role in FEATURE_NAMES
}
EXPECTED_INPUT_SHA256 = {
    "feature_manifest": "8dcb8118f85ee4a3acbf0905f01c2b173d58742a1e16dcd6004adbbbedcf63cc",
    "feature_catalogue": "f34b09a4a9f18b0202b8daf606b7baab7bdae826871bcc60a4be858a8c1cc96a",
    "features_discovery": "7031d86e4fb6e469c674d208f680ace1dbe5e11e45f3d4b2befefd747efdde16",
    "features_internal_validation": "f266e6143bc23d9e131b5ec788676b520db928aa46a57a1fcba6fd8530a80c8a",
    "features_internal_replication": "2d420ac76f8b9e1ea6a7908df92a4db1198bc0ef0b2d410875225d51536214b2",
    "term_manifest": "5b80f948a35a40ef79438ea1902b92a40dd07c35a4b541826252eb92cf96f1eb",
    "term_catalogue": "e8f9fe532c15673c0a74737632b0145d43f6494cb1ea7e94e7380198fd4e4dee",
    "discovery_endpoint_manifest": "35792117310f04daa8c383bddb5d4012084d47c7d904706d86cbe33e0a55a6ea",
    "discovery_endpoint": "f86cff6f5e9124ee82aae13911ffe55a125c6fe111fc1f64122a610febf67958",
    "design": "39127f4d2b5ddba176f7904ed498f98e0326fa902e1c3ede79fbbcf320c13ee9",
    "implementation": "21a618ccfd2610446f0bd4c7d5b8478d3c64ffd12fd37f49349d12bd619cb8e7",
}


def assign_group_folds(groups: object) -> np.ndarray:
    """Assign whole reduced-formula groups to deterministic hash folds."""

    values = np.asarray(groups, dtype=object)
    if values.ndim != 1:
        raise ValueError("NEXT87 groups must be one-dimensional")
    result = np.empty(len(values), dtype=np.int8)
    for index, value in enumerate(values):
        group = str(value)
        payload = f"{GROUP_FOLD_SALT}|{group}".encode("utf-8")
        result[index] = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % GROUP_FOLDS
    return result


def decision_metrics(
    *, supported: object, reject: object, distortion_ratio: object
) -> dict[str, object]:
    """Evaluate NEXT87 decisions; middle rows never enter precision."""

    endpoint = np.asarray(distortion_ratio, dtype=float)
    supported_array = np.asarray(supported, dtype=bool)
    reject_array = np.asarray(reject, dtype=bool)
    if (
        endpoint.ndim != 1
        or supported_array.shape != endpoint.shape
        or reject_array.shape != endpoint.shape
        or not np.isfinite(endpoint).all()
    ):
        raise ValueError("NEXT87 decision arrays differ")
    reject_array = reject_array & supported_array
    protected = endpoint <= 1.0
    severe = endpoint >= 2.0
    extremes = protected | severe
    rows = len(endpoint)
    n_supported = int(supported_array.sum())
    n_rejected = int(reject_array.sum())
    n_protected = int(protected.sum())
    protected_kept = int((protected & ~reject_array).sum())
    rejected_extremes = int((extremes & reject_array).sum())
    severe_total = int(severe.sum())
    severe_rejected = int((severe & reject_array).sum())
    metrics: dict[str, object] = {
        "rows": rows,
        "supported": n_supported,
        "rejected": n_rejected,
        "protected": n_protected,
        "protected_kept": protected_kept,
        "rejected_extremes": rejected_extremes,
        "severe_total": severe_total,
        "severe_rejected": severe_rejected,
        "coverage": n_supported / rows if rows else 0.0,
        "coverage_lower": wilson_lower_bound(n_supported, rows),
        "protected_recall": protected_kept / n_protected if n_protected else 0.0,
        "protected_recall_lower": wilson_lower_bound(protected_kept, n_protected),
        "severe_rejection_precision": (
            severe_rejected / rejected_extremes if rejected_extremes else 0.0
        ),
        "severe_rejection_precision_lower": wilson_lower_bound(
            severe_rejected, rejected_extremes
        ),
        "savings": n_rejected / rows if rows else 0.0,
        "savings_lower": wilson_lower_bound(n_rejected, rows),
        "severe_recall": severe_rejected / severe_total if severe_total else 0.0,
    }
    metrics["passes_operating_gates"] = all(
        float(metrics[name]) >= float(DEFAULT_GATES[name])
        for name in (
            "coverage_lower",
            "protected_recall_lower",
            "severe_rejection_precision_lower",
            "savings_lower",
        )
    )
    return metrics


def _wilson_lower_array(successes: np.ndarray, trials: np.ndarray) -> np.ndarray:
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    result = np.zeros(np.broadcast_shapes(successes.shape, trials.shape), dtype=float)
    successes, trials = np.broadcast_arrays(successes, trials)
    valid = (trials > 0.0) & (successes >= 0.0) & (successes <= trials)
    if not valid.any():
        return result
    z = NormalDist().inv_cdf(0.95)
    proportion = np.zeros_like(successes, dtype=float)
    proportion[valid] = successes[valid] / trials[valid]
    denominator = 1.0 + z * z / trials[valid]
    center = proportion[valid] + z * z / (2.0 * trials[valid])
    radius = z * np.sqrt(
        proportion[valid] * (1.0 - proportion[valid]) / trials[valid]
        + z * z / (4.0 * trials[valid] * trials[valid])
    )
    result[valid] = (center - radius) / denominator
    return result


def select_threshold(
    *,
    score: object,
    supported: object,
    distortion_ratio: object,
    gates: Mapping[str, float] | None = None,
    row_mask: object | None = None,
) -> dict[str, object] | None:
    """Select the highest-precision feasible threshold on one data partition."""

    gate_values = {
        name: float((gates or DEFAULT_GATES)[name])
        for name in (
            "coverage_lower",
            "protected_recall_lower",
            "severe_rejection_precision_lower",
            "savings_lower",
        )
    }
    scores = np.asarray(score, dtype=float)
    support = np.asarray(supported, dtype=bool)
    endpoint = np.asarray(distortion_ratio, dtype=float)
    if scores.ndim != 1 or support.shape != scores.shape or endpoint.shape != scores.shape:
        raise ValueError("NEXT87 threshold arrays differ")
    mask = np.ones(len(scores), dtype=bool) if row_mask is None else np.asarray(row_mask, dtype=bool)
    if mask.shape != scores.shape or not np.isfinite(endpoint[mask]).all():
        raise ValueError("NEXT87 threshold mask differs")
    local_scores = scores[mask]
    local_support = support[mask] & np.isfinite(local_scores)
    local_endpoint = endpoint[mask]
    rows = len(local_scores)
    n_supported = int(local_support.sum())
    coverage_lower = wilson_lower_bound(n_supported, rows)
    if not n_supported or coverage_lower < gate_values["coverage_lower"]:
        return None

    order = np.argsort(-local_scores[local_support], kind="stable")
    sorted_scores = local_scores[local_support][order]
    sorted_endpoint = local_endpoint[local_support][order]
    group_ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(sorted_scores) - 1]
    thresholds = sorted_scores[group_ends]
    rejected = group_ends + 1
    protected_sorted = sorted_endpoint <= 1.0
    severe_sorted = sorted_endpoint >= 2.0
    rejected_protected = np.cumsum(protected_sorted, dtype=int)[group_ends]
    rejected_severe = np.cumsum(severe_sorted, dtype=int)[group_ends]
    rejected_extremes = rejected_protected + rejected_severe
    n_protected = int((local_endpoint <= 1.0).sum())
    protected_kept = n_protected - rejected_protected
    recall_lower = _wilson_lower_array(
        protected_kept, np.full(len(thresholds), n_protected, dtype=int)
    )
    precision_lower = _wilson_lower_array(rejected_severe, rejected_extremes)
    savings_lower = _wilson_lower_array(rejected, np.full(len(thresholds), rows, dtype=int))
    feasible = (
        (recall_lower >= gate_values["protected_recall_lower"])
        & (precision_lower >= gate_values["severe_rejection_precision_lower"])
        & (savings_lower >= gate_values["savings_lower"])
    )
    candidates = np.flatnonzero(feasible)
    if not len(candidates):
        return None
    best = max(
        candidates.tolist(),
        key=lambda index: (
            float(precision_lower[index]),
            int(rejected_severe[index]),
            float(savings_lower[index]),
            float(thresholds[index]),
        ),
    )
    threshold = float(thresholds[best])
    local_reject = local_support & (local_scores >= threshold)
    metrics = decision_metrics(
        supported=local_support,
        reject=local_reject,
        distortion_ratio=local_endpoint,
    )
    return {"threshold": threshold, "metrics": metrics}


def auc_diagnostics(
    *,
    score: object,
    supported: object,
    distortion_ratio: object,
    lattice_class: object,
) -> dict[str, object]:
    """Compute protected-versus-severe AUC overall and by lattice class."""

    scores = np.asarray(score, dtype=float)
    support = np.asarray(supported, dtype=bool)
    endpoint = np.asarray(distortion_ratio, dtype=float)
    lattices = np.asarray(lattice_class, dtype=object)
    if (
        scores.ndim != 1
        or support.shape != scores.shape
        or endpoint.shape != scores.shape
        or lattices.shape != scores.shape
    ):
        raise ValueError("NEXT87 AUC arrays differ")
    extremes = ((endpoint <= 1.0) | (endpoint >= 2.0)) & support & np.isfinite(scores)
    severe = endpoint >= 2.0
    pooled = _roc_auc(scores[extremes], severe[extremes]) if extremes.any() else None
    records: list[dict[str, object]] = []
    aucs: list[float] = []
    for lattice in sorted(map(str, np.unique(lattices))):
        mask = extremes & (lattices.astype(str) == lattice)
        value = _roc_auc(scores[mask], severe[mask]) if mask.any() else None
        record = {
            "lattice_class": lattice,
            "supported_extremes": int(mask.sum()),
            "protected": int(((endpoint <= 1.0) & mask).sum()),
            "severe": int((severe & mask).sum()),
            "auc": value,
        }
        records.append(record)
        if value is not None:
            aucs.append(float(value))
    return {
        "pooled_extreme_auc": pooled,
        "macro_lattice_auc": float(np.mean(aucs)) if aucs else None,
        "worst_lattice_auc": float(np.min(aucs)) if aucs else None,
        "evaluable_lattices": len(aucs),
        "lattices": records,
    }


def _term_risk(
    features: pd.DataFrame, term: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    feature = term.get("feature")
    direction = term.get("direction")
    transform = term.get("transform")
    center = term.get("center")
    scale = term.get("scale")
    if (
        not isinstance(term.get("term_id"), str)
        or not isinstance(feature, str)
        or feature not in features
        or direction not in (-1, 1)
        or transform not in {"log1p_nonnegative", "asinh"}
        or not isinstance(center, (int, float))
        or not math.isfinite(float(center))
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(scale))
        or float(scale) <= 0.0
    ):
        raise ValueError("NEXT87 eligible term schema differs")
    raw = pd.to_numeric(features[feature], errors="coerce").to_numpy(float)
    transformed = _transformed_column(raw, str(transform))
    supported = np.isfinite(transformed)
    normalized = int(direction) * (transformed - float(center)) / float(scale)
    risk = np.maximum(0.0, normalized)
    risk[~supported] = 0.0
    return risk, supported


def _pooled_auc(
    score: np.ndarray,
    supported: np.ndarray,
    endpoint: np.ndarray,
    mask: np.ndarray | None = None,
) -> float | None:
    rows = np.ones(len(score), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    extreme = rows & supported & np.isfinite(score) & ((endpoint <= 1.0) | (endpoint >= 2.0))
    return _roc_auc(score[extreme], endpoint[extreme] >= 2.0) if extreme.any() else None


def _operating_pass(metrics: Mapping[str, object], gates: Mapping[str, float]) -> bool:
    return all(
        float(metrics[name]) >= float(gates[name])
        for name in (
            "coverage_lower",
            "protected_recall_lower",
            "severe_rejection_precision_lower",
            "savings_lower",
        )
    )


def _search_rank(
    metrics: Mapping[str, object],
    pooled_auc: float | None,
    term_count: int,
    gates: Mapping[str, float],
) -> tuple[float, ...]:
    ratios = [
        float(metrics[name]) / float(gates[name])
        for name in (
            "coverage_lower",
            "protected_recall_lower",
            "severe_rejection_precision_lower",
            "savings_lower",
        )
    ]
    ratios.append(
        float(pooled_auc) / float(gates["pooled_extreme_auc"])
        if pooled_auc is not None
        else -1.0
    )
    preliminary_pass = _operating_pass(metrics, gates) and (
        pooled_auc is not None and pooled_auc >= float(gates["pooled_extreme_auc"])
    )
    return (
        1.0 if preliminary_pass else 0.0,
        min(ratios),
        float(metrics["severe_rejection_precision_lower"]),
        float(metrics["protected_recall_lower"]),
        float(metrics["severe_rejected"]),
        float(metrics["savings_lower"]),
        float(pooled_auc) if pooled_auc is not None else -1.0,
        -float(term_count),
    )


def _formula_from_spec(
    terms: list[Mapping[str, object]],
    indices: tuple[int, ...],
    weights: tuple[float, ...],
    threshold: float,
) -> dict[str, object]:
    return {
        "kind": FORMULA_KIND,
        "missing_policy": MISSING_POLICY,
        "terms": [
            {
                "term_id": str(terms[index]["term_id"]),
                "feature": str(terms[index]["feature"]),
                "group": str(terms[index].get("group", "unspecified")),
                "direction": int(terms[index]["direction"]),
                "transform": str(terms[index]["transform"]),
                "center": float(terms[index]["center"]),
                "scale": float(terms[index]["scale"]),
                "weight": float(weight),
            }
            for index, weight in zip(indices, weights, strict=True)
        ],
        "threshold": float(threshold),
    }


def _candidate_identity(
    terms: list[Mapping[str, object]], indices: tuple[int, ...], weights: tuple[float, ...]
) -> tuple[str, str]:
    term_ids = tuple(str(terms[index]["term_id"]) for index in indices)
    term_list_key = json.dumps(term_ids, separators=(",", ":"))
    formula_key = json.dumps(
        {"term_ids": term_ids, "weights": weights},
        sort_keys=True,
        separators=(",", ":"),
    )
    return term_list_key, formula_key


def _pauling_baseline(features: pd.DataFrame, endpoint: np.ndarray) -> dict[str, object]:
    if "pauling_p2_p5_decision" not in features:
        raise ValueError("NEXT87 Pauling baseline decision is missing")
    decisions = features["pauling_p2_p5_decision"].astype(str).to_numpy()
    if set(np.unique(decisions)) - {"KEEP", "REJECT", "ABSTAIN"}:
        raise ValueError("NEXT87 Pauling baseline decision differs")
    supported = decisions != "ABSTAIN"
    reject = decisions == "REJECT"
    metrics = decision_metrics(
        supported=supported, reject=reject, distortion_ratio=endpoint
    )
    binary_auc = _roc_auc(
        reject[((endpoint <= 1.0) | (endpoint >= 2.0))].astype(float),
        endpoint[((endpoint <= 1.0) | (endpoint >= 2.0))] >= 2.0,
    )
    return {**metrics, "binary_reject_auc_all_extremes": binary_auc}


def search_scigen_sparse_law(
    *,
    features: pd.DataFrame,
    distortion_ratio: object,
    eligible_terms: list[Mapping[str, object]],
    gates: Mapping[str, float] = DEFAULT_GATES,
    pair_shortlist: int = PAIR_SHORTLIST,
    triple_shortlist: int = TRIPLE_SHORTLIST,
    max_terms_per_group: int = MAX_TERMS_PER_PHYSICS_GROUP,
) -> dict[str, object]:
    """Run the finite staged NEXT87 search using discovery labels only."""

    endpoint = np.asarray(distortion_ratio, dtype=float)
    required = {"material_id", "reduced_formula", "lattice_class", "pauling_p2_p5_decision"}
    if (
        len(features) != len(endpoint)
        or endpoint.ndim != 1
        or not np.isfinite(endpoint).all()
        or required - set(features.columns)
        or features["material_id"].astype(str).duplicated().any()
        or not (endpoint <= 1.0).any()
        or not (endpoint >= 2.0).any()
    ):
        raise ValueError("NEXT87 discovery arrays differ")
    if not eligible_terms or pair_shortlist <= 0 or triple_shortlist <= 0 or max_terms_per_group <= 0:
        raise ValueError("NEXT87 finite catalogue controls differ")
    terms = sorted((dict(term) for term in eligible_terms), key=lambda term: str(term.get("term_id")))
    term_ids = [str(term.get("term_id")) for term in terms]
    if len(term_ids) != len(set(term_ids)) or any(not term_id for term_id in term_ids):
        raise ValueError("NEXT87 term identities differ")

    risks: list[np.ndarray] = []
    supports: list[np.ndarray] = []
    for term in terms:
        risk, support = _term_risk(features, term)
        risks.append(risk)
        supports.append(support)
    risk_matrix = np.column_stack(risks)
    support_matrix = np.column_stack(supports)
    folds = assign_group_folds(features["reduced_formula"].astype(str).to_numpy())
    if set(np.unique(folds)) != set(range(GROUP_FOLDS)):
        raise ValueError("NEXT87 discovery groups do not populate all folds")

    single_order: list[tuple[float, str, int]] = []
    for index, term_id in enumerate(term_ids):
        auc = _pooled_auc(risk_matrix[:, index], support_matrix[:, index], endpoint)
        single_order.append((-(float(auc) if auc is not None else -1.0), term_id, index))
    single_order.sort()
    shortlist: list[int] = []
    group_counts: dict[str, int] = {}
    for _negative_auc, _term_id, index in single_order:
        group = str(terms[index].get("group", "unspecified"))
        if group_counts.get(group, 0) >= max_terms_per_group:
            continue
        shortlist.append(index)
        group_counts[group] = group_counts.get(group, 0) + 1
        if len(shortlist) >= min(pair_shortlist, len(terms)):
            break
    ranked_shortlist = list(shortlist)
    pair_indices = sorted(ranked_shortlist, key=lambda index: term_ids[index])
    ranked_triple_shortlist = ranked_shortlist[
        : min(triple_shortlist, len(ranked_shortlist))
    ]
    triple_indices = sorted(
        ranked_triple_shortlist, key=lambda index: term_ids[index]
    )

    specs: list[tuple[str, tuple[int, ...], tuple[float, ...]]] = [
        ("single", (index,), (1.0,)) for index in range(len(terms))
    ]
    for indices in combinations(pair_indices, 2):
        for ratio in WEIGHT_GRID:
            specs.append(("pair", tuple(indices), (1.0, float(ratio))))
    for indices in combinations(triple_indices, 3):
        for right_weights in product(WEIGHT_GRID, repeat=2):
            specs.append(
                (
                    "triple",
                    tuple(indices),
                    (1.0, float(right_weights[0]), float(right_weights[1])),
                )
            )

    fold_winners: list[dict[str, object] | None] = [None] * GROUP_FOLDS
    internal_records: list[dict[str, object]] = []
    for stage, indices, weights in specs:
        score = np.sum(
            risk_matrix[:, indices] * np.asarray(weights, dtype=float)[None, :], axis=1
        )
        supported = np.all(support_matrix[:, indices], axis=1)
        score = np.asarray(score, dtype=float)
        score[~supported] = np.nan
        selected = select_threshold(
            score=score,
            supported=supported,
            distortion_ratio=endpoint,
            gates=gates,
        )
        pooled_auc = _pooled_auc(score, supported, endpoint)
        if selected is None:
            metrics = decision_metrics(
                supported=supported,
                reject=np.zeros(len(features), dtype=bool),
                distortion_ratio=endpoint,
            )
            threshold = None
        else:
            metrics = dict(selected["metrics"])
            threshold = float(selected["threshold"])
        rank = _search_rank(metrics, pooled_auc, len(indices), gates)
        term_list_key, formula_key = _candidate_identity(terms, indices, weights)
        fold_train_summaries: list[dict[str, object]] = []
        for held_out in range(GROUP_FOLDS):
            train = folds != held_out
            fold_selected = select_threshold(
                score=score,
                supported=supported,
                distortion_ratio=endpoint,
                gates=gates,
                row_mask=train,
            )
            fold_auc = _pooled_auc(score, supported, endpoint, train)
            if fold_selected is None:
                fold_metrics = decision_metrics(
                    supported=supported[train],
                    reject=np.zeros(int(train.sum()), dtype=bool),
                    distortion_ratio=endpoint[train],
                )
                fold_threshold = None
            else:
                fold_metrics = dict(fold_selected["metrics"])
                fold_threshold = float(fold_selected["threshold"])
            fold_rank = _search_rank(fold_metrics, fold_auc, len(indices), gates)
            fold_train_summaries.append(
                {
                    "held_out_fold": held_out,
                    "threshold": fold_threshold,
                    "rank": list(fold_rank),
                }
            )
            winner = fold_winners[held_out]
            if (
                winner is None
                or fold_rank > winner["rank_tuple"]
                or (fold_rank == winner["rank_tuple"] and formula_key < winner["formula_key"])
            ):
                fold_winners[held_out] = {
                    "held_out_fold": held_out,
                    "term_list_key": term_list_key,
                    "term_ids": [term_ids[index] for index in indices],
                    "formula_key": formula_key,
                    "rank_tuple": fold_rank,
                    "rank": list(fold_rank),
                    "threshold": fold_threshold,
                }
        internal_records.append(
            {
                "stage": stage,
                "indices": indices,
                "weights_tuple": weights,
                "term_ids": [term_ids[index] for index in indices],
                "weights": list(weights),
                "term_list_key": term_list_key,
                "formula_key": formula_key,
                "threshold": threshold,
                "metrics": metrics,
                "pooled_extreme_auc": pooled_auc,
                "rank_tuple": rank,
                "rank": list(rank),
                "fold_train_summaries": fold_train_summaries,
            }
        )

    if any(winner is None for winner in fold_winners):
        raise RuntimeError("NEXT87 fold search produced no winner")
    win_counts: dict[str, int] = {}
    for winner in fold_winners:
        key = str(winner["term_list_key"])
        win_counts[key] = win_counts.get(key, 0) + 1
    stable_term_lists = {key for key, count in win_counts.items() if count >= 4}
    pauling = _pauling_baseline(features, endpoint)

    evaluated_finalists: list[dict[str, object]] = []
    records_by_rank = sorted(
        internal_records,
        key=lambda record: (record["rank_tuple"], tuple(-ord(char) for char in record["formula_key"])),
        reverse=True,
    )
    finalist_pool = [
        record for record in records_by_rank if record["term_list_key"] in stable_term_lists
    ]
    if not finalist_pool:
        finalist_pool = records_by_rank[:1]
    for record in finalist_pool:
        indices = record["indices"]
        weights = record["weights_tuple"]
        score = np.sum(
            risk_matrix[:, indices] * np.asarray(weights, dtype=float)[None, :], axis=1
        )
        supported = np.all(support_matrix[:, indices], axis=1)
        score[~supported] = np.nan
        threshold = record["threshold"]
        reject = (
            supported & (score >= float(threshold))
            if threshold is not None
            else np.zeros(len(features), dtype=bool)
        )
        diagnostics = auc_diagnostics(
            score=score,
            supported=supported,
            distortion_ratio=endpoint,
            lattice_class=features["lattice_class"].astype(str).to_numpy(),
        )
        fold_diagnostics: list[dict[str, object]] = []
        all_fold_support = True
        all_raw_fold_gates = True
        for held_out in range(GROUP_FOLDS):
            train = folds != held_out
            test = folds == held_out
            fold_selected = select_threshold(
                score=score,
                supported=supported,
                distortion_ratio=endpoint,
                gates=gates,
                row_mask=train,
            )
            support_lower = wilson_lower_bound(int(supported[test].sum()), int(test.sum()))
            support_pass = support_lower >= float(gates["coverage_lower"])
            all_fold_support &= support_pass
            if fold_selected is None:
                fold_metrics = decision_metrics(
                    supported=supported[test],
                    reject=np.zeros(int(test.sum()), dtype=bool),
                    distortion_ratio=endpoint[test],
                )
                fold_threshold = None
            else:
                fold_threshold = float(fold_selected["threshold"])
                fold_metrics = decision_metrics(
                    supported=supported[test],
                    reject=supported[test] & (score[test] >= fold_threshold),
                    distortion_ratio=endpoint[test],
                )
            raw_pass = (
                fold_selected is not None
                and float(fold_metrics["severe_rejection_precision"]) >= FOLD_MIN_PRECISION
                and float(fold_metrics["protected_recall"]) >= FOLD_MIN_PROTECTED_RECALL
            )
            all_raw_fold_gates &= raw_pass
            fold_diagnostics.append(
                {
                    "held_out_fold": held_out,
                    "train_threshold": fold_threshold,
                    "support_coverage_lower": support_lower,
                    "passes_support_gate": support_pass,
                    "metrics": fold_metrics,
                    "passes_raw_fold_gates": raw_pass,
                }
            )
        metrics = dict(record["metrics"])
        auc_pass = (
            diagnostics["pooled_extreme_auc"] is not None
            and float(diagnostics["pooled_extreme_auc"]) >= float(gates["pooled_extreme_auc"])
            and diagnostics["macro_lattice_auc"] is not None
            and float(diagnostics["macro_lattice_auc"]) >= float(gates["macro_lattice_auc"])
            and diagnostics["worst_lattice_auc"] is not None
            and float(diagnostics["worst_lattice_auc"]) >= float(gates["worst_lattice_auc"])
            and int(diagnostics["evaluable_lattices"]) >= int(gates["evaluable_lattices"])
        )
        beats_pauling = (
            int(metrics["severe_rejected"]) > int(pauling["severe_rejected"])
            and float(metrics["severe_rejection_precision_lower"])
            > float(pauling["severe_rejection_precision_lower"])
        )
        stable_count = win_counts.get(str(record["term_list_key"]), 0)
        passes = bool(
            threshold is not None
            and _operating_pass(metrics, gates)
            and auc_pass
            and stable_count >= 4
            and all_fold_support
            and all_raw_fold_gates
            and beats_pauling
        )
        evaluated_finalists.append(
            {
                "record": record,
                "score": score,
                "supported": supported,
                "reject": reject,
                "diagnostics": diagnostics,
                "fold_diagnostics": fold_diagnostics,
                "stable_count": stable_count,
                "beats_pauling": beats_pauling,
                "passes": passes,
            }
        )
    selected = max(
        evaluated_finalists,
        key=lambda item: (
            1 if item["passes"] else 0,
            item["record"]["rank_tuple"],
            tuple(-ord(char) for char in item["record"]["formula_key"]),
        ),
    )
    selected_record = selected["record"]
    selected_threshold = selected_record["threshold"]
    formula = (
        _formula_from_spec(
            terms,
            selected_record["indices"],
            selected_record["weights_tuple"],
            float(selected_threshold),
        )
        if selected_threshold is not None
        else None
    )
    diagnostics = selected["diagnostics"]
    discovery_metrics = {
        **selected_record["metrics"],
        **{
            key: diagnostics[key]
            for key in (
                "pooled_extreme_auc",
                "macro_lattice_auc",
                "worst_lattice_auc",
                "evaluable_lattices",
            )
        },
        "beats_pauling_severe_count_and_precision_lower": selected["beats_pauling"],
    }
    public_records = [
        {
            key: value
            for key, value in record.items()
            if key not in {"indices", "weights_tuple", "rank_tuple"}
        }
        for record in internal_records
    ]
    public_winners = [
        {key: value for key, value in winner.items() if key != "rank_tuple"}
        for winner in fold_winners
    ]
    return {
        "selected_formula": formula,
        "discovery_metrics": discovery_metrics,
        "lattice_diagnostics": diagnostics["lattices"],
        "fold_diagnostics": selected["fold_diagnostics"],
        "fold_stability": {
            "fold_winners": public_winners,
            "term_list_win_counts": win_counts,
            "stable_term_lists": sorted(stable_term_lists),
            "selected_term_list_win_count": selected["stable_count"],
        },
        "pauling_baseline": pauling,
        "passes_discovery_gates": bool(selected["passes"]),
        "candidate_count": len(internal_records),
        "single_shortlist_term_ids": [term_ids[index] for index in ranked_shortlist],
        "triple_shortlist_term_ids": [
            term_ids[index] for index in ranked_triple_shortlist
        ],
        "search_records": public_records,
        "score": selected["score"],
        "supported": selected["supported"],
        "reject": selected["reject"],
        "fold": folds,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read_json(path: Path, *, role: str) -> dict[str, object]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _publish_directory_no_replace(staging: Path, target: Path) -> None:
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    os.rename(staging, target)


def _search_record_table(records: list[Mapping[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in records:
        metrics = record["metrics"]
        rows.append(
            {
                "stage": record["stage"],
                "term_ids_json": json.dumps(record["term_ids"], separators=(",", ":")),
                "weights_json": json.dumps(record["weights"], separators=(",", ":")),
                "term_list_key": record["term_list_key"],
                "formula_key": record["formula_key"],
                "threshold": record["threshold"],
                "pooled_extreme_auc": record["pooled_extreme_auc"],
                "rank_json": json.dumps(record["rank"], separators=(",", ":")),
                "fold_train_summaries_json": json.dumps(
                    record["fold_train_summaries"], sort_keys=True, separators=(",", ":")
                ),
                **{f"metric_{key}": value for key, value in metrics.items()},
            }
        )
    return pd.DataFrame(rows)


def run_scigen_sparse_search(
    *,
    feature_dir: Path,
    term_catalogue_dir: Path,
    discovery_endpoint_dir: Path,
    design_path: Path,
    implementation_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run NEXT87 without any interface for locked validation/replication labels."""

    feature_root = Path(feature_dir).resolve()
    term_root = Path(term_catalogue_dir).resolve()
    endpoint_root = Path(discovery_endpoint_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "feature_manifest": feature_root / FEATURE_MANIFEST_NAME,
        "feature_catalogue": feature_root / FEATURE_CATALOGUE_NAME,
        **{
            f"features_{role}": feature_root / FEATURE_NAMES[role]
            for role in FEATURE_NAMES
        },
        "term_manifest": term_root / TERM_MANIFEST_NAME,
        "term_catalogue": term_root / TERM_CATALOGUE_NAME,
        "discovery_endpoint_manifest": endpoint_root / ENDPOINT_MANIFEST_NAME,
        "discovery_endpoint": endpoint_root / ENDPOINT_NAME,
        "design": Path(design_path).resolve(),
        "implementation": Path(implementation_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT87 input is missing")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT87 formal input identity differs")

    feature_manifest = _read_json(paths["feature_manifest"], role="NEXT85 manifest")
    feature_outputs = feature_manifest.get("outputs_sha256")
    if (
        feature_manifest.get("protocol") != FEATURE_PROTOCOL
        or feature_manifest.get("labels_opened") is not False
        or feature_manifest.get("endpoint_payloads_opened") is not False
        or feature_manifest.get("relaxed_structures_opened") is not False
        or not isinstance(feature_outputs, Mapping)
        or feature_outputs.get(FEATURE_CATALOGUE_NAME) != hashes["feature_catalogue"]
        or any(
            feature_outputs.get(FEATURE_NAMES[role]) != hashes[f"features_{role}"]
            for role in FEATURE_NAMES
        )
    ):
        raise ValueError("NEXT85 label-free feature provenance differs")
    term_manifest = _read_json(paths["term_manifest"], role="NEXT86 term manifest")
    term_outputs = term_manifest.get("outputs_sha256")
    term_catalogue = _read_json(paths["term_catalogue"], role="NEXT86 term catalogue")
    if (
        term_manifest.get("protocol") != TERM_PROTOCOL
        or term_manifest.get("labels_opened") is not False
        or term_manifest.get("endpoint_payloads_opened") is not False
        or not isinstance(term_outputs, Mapping)
        or term_outputs.get(TERM_CATALOGUE_NAME) != hashes["term_catalogue"]
        or term_catalogue.get("protocol") != TERM_PROTOCOL
        or term_catalogue.get("labels_opened") is not False
        or not isinstance(term_catalogue.get("eligible_terms"), list)
    ):
        raise ValueError("NEXT86 term catalogue provenance differs")
    endpoint_manifest = _read_json(
        paths["discovery_endpoint_manifest"], role="NEXT86 discovery endpoint manifest"
    )
    endpoint_outputs = endpoint_manifest.get("outputs_sha256")
    if (
        endpoint_manifest.get("protocol") != ENDPOINT_PROTOCOL
        or endpoint_manifest.get("partition_role") != "discovery"
        or not isinstance(endpoint_outputs, Mapping)
        or endpoint_outputs.get(ENDPOINT_NAME) != hashes["discovery_endpoint"]
    ):
        raise ValueError("NEXT86 discovery endpoint provenance differs")

    feature_tables: dict[str, pd.DataFrame] = {}
    for role in FEATURE_NAMES:
        table = pd.read_parquet(paths[f"features_{role}"])
        if (
            "material_id" not in table
            or "partition_role" not in table
            or table["material_id"].astype(str).duplicated().any()
            or set(table["partition_role"].astype(str)) != {role}
        ):
            raise ValueError(f"NEXT87 {role} feature identity differs")
        feature_tables[role] = table
    endpoints = pd.read_parquet(paths["discovery_endpoint"])
    if (
        {"material_id", "partition_role", "distortion_ratio"} - set(endpoints.columns)
        or endpoints["material_id"].astype(str).duplicated().any()
        or set(endpoints["partition_role"].astype(str)) != {"discovery"}
    ):
        raise ValueError("NEXT87 discovery endpoint table differs")
    discovery = feature_tables["discovery"].merge(
        endpoints.loc[:, ["material_id", "distortion_ratio"]],
        on="material_id",
        how="inner",
        validate="one_to_one",
    )
    if len(discovery) != len(feature_tables["discovery"]) or len(discovery) != len(endpoints):
        raise ValueError("NEXT87 discovery identity join differs")
    result = search_scigen_sparse_law(
        features=discovery,
        distortion_ratio=discovery["distortion_ratio"].to_numpy(float),
        eligible_terms=term_catalogue["eligible_terms"],
    )

    source_path = Path(__file__).resolve()
    source_hash = _sha256_file(source_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    output_paths: list[Path] = []
    try:
        search_path = staging / SEARCH_RECORD_NAME
        _search_record_table(result["search_records"]).to_parquet(search_path, index=False)
        output_paths.append(search_path)
        evaluation = {
            "protocol": PROTOCOL,
            "status": (
                "discovery_gates_passed_predictions_frozen"
                if result["passes_discovery_gates"]
                else "discovery_gates_failed_stop_without_lockbox_opening"
            ),
            "passes_discovery_gates": result["passes_discovery_gates"],
            "candidate_count": result["candidate_count"],
            "selected_formula": result["selected_formula"],
            "discovery_metrics": result["discovery_metrics"],
            "pauling_baseline": result["pauling_baseline"],
            "lattice_diagnostics": result["lattice_diagnostics"],
            "fold_stability": result["fold_stability"],
            "single_shortlist_term_ids": result["single_shortlist_term_ids"],
            "triple_shortlist_term_ids": result["triple_shortlist_term_ids"],
        }
        evaluation_path = staging / EVALUATION_NAME
        evaluation_path.write_bytes(_json_bytes(evaluation))
        output_paths.append(evaluation_path)
        folds_path = staging / FOLD_DIAGNOSTICS_NAME
        folds_path.write_bytes(_json_bytes(result["fold_diagnostics"]))
        output_paths.append(folds_path)

        if result["passes_discovery_gates"]:
            formula = {
                **result["selected_formula"],
                "protocol": PROTOCOL,
                "training_partition": "SCIGEN discovery only",
                "validation_endpoint_opened": False,
                "replication_endpoint_opened": False,
            }
            formula_path = staging / FORMULA_NAME
            formula_path.write_bytes(_json_bytes(formula))
            formula_sha256 = _sha256_file(formula_path)
            output_paths.append(formula_path)
            for role, table in feature_tables.items():
                score, supported, reject = apply_scigen_formula(table, formula)
                predictions = pd.DataFrame(
                    {
                        "material_id": table["material_id"].astype(str).to_numpy(),
                        "partition_role": role,
                        "next87_score": score,
                        "next87_supported": supported,
                        "next87_reject": reject,
                        "next87_decision": np.where(reject, "REJECT", "KEEP"),
                        "formula_sha256": formula_sha256,
                    }
                ).sort_values("material_id", kind="stable", ignore_index=True)
                prediction_path = staging / PREDICTION_NAMES[role]
                predictions.to_parquet(prediction_path, index=False)
                output_paths.append(prediction_path)

        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "mode": "discovery_only_sparse_search_and_conditional_all_partition_prediction_freeze",
            "passes_discovery_gates": result["passes_discovery_gates"],
            "discovery_endpoint_opened": True,
            "validation_endpoint_opened": False,
            "replication_endpoint_opened": False,
            "relaxed_structures_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "model_or_proxy_potential_used": False,
            "physical_relaxation_executed": False,
            "candidate_count": result["candidate_count"],
            "prediction_partitions_frozen": (
                list(FEATURE_NAMES) if result["passes_discovery_gates"] else []
            ),
            "inputs_sha256": {
                name: {"path": str(path), "sha256": hashes[name]}
                for name, path in paths.items()
            },
            "executed_source_sha256": {
                "src/next87_scigen_sparse_law_search.py": source_hash
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
            "scientific_improvement_claim": bool(result["passes_discovery_gates"]),
            "universal_or_dft_equivalence_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if any(_sha256_file(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT87 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT87 source changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _transformed_column(values: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if transform == "log1p_nonnegative":
        valid = np.isfinite(values) & (values >= -1.0e-12)
        result = np.full(values.shape, np.nan, dtype=float)
        result[valid] = np.log1p(np.maximum(values[valid], 0.0))
        return result
    if transform == "asinh":
        return np.arcsinh(values)
    raise ValueError(f"unknown NEXT87 transform: {transform}")


def apply_scigen_formula(
    features: pd.DataFrame, formula: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply an explicit frozen hinge formula with strict fail-open semantics."""

    if formula.get("kind") != FORMULA_KIND:
        raise ValueError("NEXT87 formula kind differs")
    if formula.get("missing_policy") != MISSING_POLICY:
        raise ValueError("NEXT87 formula missing policy differs")
    terms = formula.get("terms")
    if not isinstance(terms, list) or not 1 <= len(terms) <= 3:
        raise ValueError("NEXT87 formulas require one to three terms")

    supported = np.ones(len(features), dtype=bool)
    score = np.zeros(len(features), dtype=float)
    for term in terms:
        if not isinstance(term, Mapping):
            raise ValueError("NEXT87 formula term must be an object")
        term_id = term.get("term_id")
        feature = term.get("feature")
        direction = term.get("direction")
        transform = term.get("transform")
        center = term.get("center")
        scale = term.get("scale")
        weight = term.get("weight")
        if not isinstance(term_id, str) or not term_id:
            raise ValueError("NEXT87 term id differs")
        if not isinstance(feature, str) or feature not in features:
            raise ValueError(f"NEXT87 formula feature is missing: {feature}")
        if direction not in (-1, 1):
            raise ValueError("NEXT87 term direction differs")
        if transform not in {"log1p_nonnegative", "asinh"}:
            raise ValueError("NEXT87 term transform differs")
        if not isinstance(center, (int, float)) or not math.isfinite(float(center)):
            raise ValueError("NEXT87 term center differs")
        if (
            not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            or float(scale) <= 0.0
        ):
            raise ValueError("NEXT87 term scale differs")
        if (
            not isinstance(weight, (int, float))
            or float(weight) not in WEIGHT_GRID
        ):
            raise ValueError("NEXT87 term weight differs")

        raw = pd.to_numeric(features[feature], errors="coerce").to_numpy(float)
        transformed = _transformed_column(raw, str(transform))
        finite = np.isfinite(transformed)
        supported &= finite
        normalized = int(direction) * (transformed - float(center)) / float(scale)
        score += float(weight) * np.maximum(0.0, normalized)

    threshold = formula.get("threshold")
    if (
        not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
    ):
        raise ValueError("NEXT87 formula threshold differs")
    score[~supported] = np.nan
    reject = supported & (score >= float(threshold))
    return score, supported, reject


__all__ = [
    "FORMULA_KIND",
    "MANIFEST_NAME",
    "MISSING_POLICY",
    "PREDICTION_NAMES",
    "PROTOCOL",
    "WEIGHT_GRID",
    "apply_scigen_formula",
    "assign_group_folds",
    "auc_diagnostics",
    "decision_metrics",
    "search_scigen_sparse_law",
    "select_threshold",
    "run_scigen_sparse_search",
]
