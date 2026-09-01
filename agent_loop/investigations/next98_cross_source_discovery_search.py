"""Finite cross-source discovery search for a transferable no-DFT law."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.next87_scigen_sparse_law_search import (
    _pauling_baseline,
    _term_risk,
    _wilson_lower_array,
    assign_group_folds,
    auc_diagnostics,
    decision_metrics,
    wilson_lower_bound,
)
from src.next85_scigen_label_free_features import (
    CATALOGUE_NAME as SCIGEN_FEATURE_CATALOGUE_NAME,
    FEATURE_NAMES as SCIGEN_FEATURE_NAMES,
    MANIFEST_NAME as SCIGEN_FEATURE_MANIFEST_NAME,
    PROTOCOL as SCIGEN_FEATURE_PROTOCOL,
)
from src.next86_scigen_endpoint_router import (
    ENDPOINT_NAME as SCIGEN_ENDPOINT_NAME,
    MANIFEST_NAME as SCIGEN_ENDPOINT_MANIFEST_NAME,
    PROTOCOL as SCIGEN_ENDPOINT_PROTOCOL,
)
from src.next86_scigen_term_catalogue import (
    CATALOGUE_NAME as SCIGEN_TERM_CATALOGUE_NAME,
    MANIFEST_NAME as SCIGEN_TERM_MANIFEST_NAME,
    PROTOCOL as SCIGEN_TERM_PROTOCOL,
)
from src.next93_wyformer_source_lockbox import _sha256_file, _write_json
from src.next93b_wyformer_blind_lockbox import (
    ENDPOINT_NAME as WYFORMER_ENDPOINT_NAME,
    MANIFEST_NAME as WYFORMER_ENDPOINT_MANIFEST_NAME,
    PROTOCOL as WYFORMER_ENDPOINT_PROTOCOL,
)
from src.next94_wyformer_label_free_features import (
    FEATURE_NAMES as WYFORMER_FEATURE_NAMES,
    MANIFEST_NAME as WYFORMER_FEATURE_MANIFEST_NAME,
    PROTOCOL as WYFORMER_FEATURE_PROTOCOL,
)
from src.next95_wyformer_sparse_law_search import (
    DEFAULT_GATES,
    _endpoint_numeric,
    recalibrate_terms,
)
from src.next96_wyformer_dual_operating_candidate import pauling_dominance


PROTOCOL = "2026-08-04-next98-cross-source-discovery-search-v1"
MANIFEST_NAME = "MANIFEST.json"
EVALUATION_NAME = "NEXT98_CROSS_SOURCE_DISCOVERY_EVALUATION.json"
CATALOGUE_NAME = "NEXT98_CROSS_SOURCE_TERM_CATALOGUE.json"
SEARCH_NAME = "next98_cross_source_candidate_search.parquet"
GROUP_FOLDS = 5
BROAD_MIN_PRECISION_LOWER = 0.45
TOP_PER_VIEW = 120
AUC_GATES = {
    "pooled_extreme_auc": 0.75,
    "macro_lattice_auc": 0.60,
    "worst_lattice_auc": 0.55,
    "evaluable_lattices": 5,
}
EXPECTED_INPUT_SHA256 = {
    "scigen_feature_manifest": "8dcb8118f85ee4a3acbf0905f01c2b173d58742a1e16dcd6004adbbbedcf63cc",
    "scigen_feature_catalogue": "f34b09a4a9f18b0202b8daf606b7baab7bdae826871bcc60a4be858a8c1cc96a",
    "scigen_features": "7031d86e4fb6e469c674d208f680ace1dbe5e11e45f3d4b2befefd747efdde16",
    "scigen_term_manifest": "5b80f948a35a40ef79438ea1902b92a40dd07c35a4b541826252eb92cf96f1eb",
    "scigen_term_catalogue": "e8f9fe532c15673c0a74737632b0145d43f6494cb1ea7e94e7380198fd4e4dee",
    "scigen_endpoint_manifest": "35792117310f04daa8c383bddb5d4012084d47c7d904706d86cbe33e0a55a6ea",
    "scigen_endpoint": "f86cff6f5e9124ee82aae13911ffe55a125c6fe111fc1f64122a610febf67958",
    "scigen_search_records": "438c98681ddf7366bccaf88f36221142c1851873d89632c9d04196bffed7dac2",
    "wyformer_feature_manifest": "fb66f7c5caade419a46b9a3fa6fef1bc5b3afa3eebeb95a4bc53baddabc0f659",
    "wyformer_features": "c515baec0fccef5bc03c7672f1d4e1aca278f5ed4d7b6f1bf7f66c734e2b87f7",
    "wyformer_endpoint_manifest": "3cf3a196ab497851131d5d1604f272d15121c19a943eeb3103a268e7e8b332f5",
    "wyformer_endpoint": "f39836e62a1da03ed823479e87d6f75fc0d01da60a8c0a2faa696638cc2fb9d7",
    "wyformer_search_records": "36c18f23e2b7c8d5ad7df16da34205a2dedbd1cf1e5ba544299f501653a87c35",
    "design": "a4ade9d106a4b03d46894a2c4cbaa601286efb5270237152483b3e1af545e86b",
}


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def build_source_fold_cells(
    *, source: object, folds: object
) -> list[dict[str, object]]:
    """Return two source aggregates plus five deterministic folds per source."""

    source_values = np.asarray(source, dtype=object)
    fold_values = np.asarray(folds, dtype=int)
    if (
        source_values.ndim != 1
        or fold_values.shape != source_values.shape
        or len(source_values) == 0
        or set(np.unique(fold_values)) - set(range(GROUP_FOLDS))
    ):
        raise ValueError("NEXT98 source-fold arrays differ")
    cells: list[dict[str, object]] = []
    for source_name in sorted(str(value) for value in np.unique(source_values)):
        source_mask = source_values.astype(str) == source_name
        cells.append(
            {
                "cell_id": f"{source_name}:all",
                "source_dataset": source_name,
                "fold": None,
                "kind": "source_aggregate",
                "mask": source_mask,
            }
        )
        for fold in range(GROUP_FOLDS):
            mask = source_mask & (fold_values == fold)
            if not mask.any():
                raise ValueError("NEXT98 source-by-formula fold is empty")
            cells.append(
                {
                    "cell_id": f"{source_name}:fold{fold}",
                    "source_dataset": source_name,
                    "fold": fold,
                    "kind": "source_fold",
                    "mask": mask,
                }
            )
    return cells


def _threshold_tables(
    *,
    score: object,
    supported: object,
    endpoint: object,
    cells: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    scores = np.asarray(score, dtype=float)
    support = np.asarray(supported, dtype=bool)
    endpoints = np.asarray(endpoint, dtype=float)
    if (
        scores.ndim != 1
        or support.shape != scores.shape
        or endpoints.shape != scores.shape
        or not np.isfinite(endpoints).all()
    ):
        raise ValueError("NEXT98 threshold arrays differ")
    effective = support & np.isfinite(scores)
    if not effective.any():
        return None
    effective_indices = np.flatnonzero(effective)
    order = np.argsort(-scores[effective], kind="stable")
    sorted_indices = effective_indices[order]
    sorted_scores = scores[sorted_indices]
    group_ends = np.r_[
        np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]),
        len(sorted_scores) - 1,
    ]
    thresholds = sorted_scores[group_ends]
    cell_count = len(cells)
    threshold_count = len(thresholds)
    rejected = np.zeros((cell_count, threshold_count), dtype=int)
    rejected_protected = np.zeros_like(rejected)
    rejected_severe = np.zeros_like(rejected)
    coverage_lower = np.zeros(cell_count, dtype=float)
    total_rows = np.zeros(cell_count, dtype=int)
    total_protected = np.zeros(cell_count, dtype=int)
    total_severe = np.zeros(cell_count, dtype=int)
    for cell_index, cell in enumerate(cells):
        mask = np.asarray(cell["mask"], dtype=bool)
        if mask.shape != scores.shape or not mask.any():
            raise ValueError("NEXT98 threshold cell differs")
        total_rows[cell_index] = int(mask.sum())
        total_protected[cell_index] = int((mask & (endpoints <= 1.0)).sum())
        total_severe[cell_index] = int((mask & (endpoints >= 2.0)).sum())
        coverage_lower[cell_index] = wilson_lower_bound(
            int((mask & effective).sum()), int(total_rows[cell_index])
        )
        sorted_cell = mask[sorted_indices]
        rejected[cell_index] = np.cumsum(sorted_cell, dtype=int)[group_ends]
        rejected_protected[cell_index] = np.cumsum(
            sorted_cell & (endpoints[sorted_indices] <= 1.0), dtype=int
        )[group_ends]
        rejected_severe[cell_index] = np.cumsum(
            sorted_cell & (endpoints[sorted_indices] >= 2.0), dtype=int
        )[group_ends]
    protected_kept = total_protected[:, None] - rejected_protected
    recall_lower = _wilson_lower_array(
        protected_kept,
        np.broadcast_to(total_protected[:, None], protected_kept.shape),
    )
    rejected_extremes = rejected_protected + rejected_severe
    precision_lower = _wilson_lower_array(rejected_severe, rejected_extremes)
    savings_lower = _wilson_lower_array(
        rejected, np.broadcast_to(total_rows[:, None], rejected.shape)
    )
    severe_recall = np.divide(
        rejected_severe,
        total_severe[:, None],
        out=np.zeros_like(rejected_severe, dtype=float),
        where=total_severe[:, None] > 0,
    )
    return {
        "thresholds": thresholds,
        "effective": effective,
        "coverage_lower": coverage_lower,
        "rejected": rejected,
        "protected_kept": protected_kept,
        "rejected_severe": rejected_severe,
        "precision_lower": precision_lower,
        "recall_lower": recall_lower,
        "savings_lower": savings_lower,
        "severe_recall": severe_recall,
    }


def _selected_cell_records(
    *,
    threshold: float,
    score: np.ndarray,
    supported: np.ndarray,
    endpoint: np.ndarray,
    cells: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    reject = supported & np.isfinite(score) & (score >= float(threshold))
    records: list[dict[str, object]] = []
    for cell in cells:
        mask = np.asarray(cell["mask"], dtype=bool)
        records.append(
            {
                "cell_id": str(cell["cell_id"]),
                "source_dataset": str(cell["source_dataset"]),
                "fold": cell["fold"],
                "kind": str(cell["kind"]),
                "metrics": decision_metrics(
                    supported=supported[mask],
                    reject=reject[mask],
                    distortion_ratio=endpoint[mask],
                ),
            }
        )
    return records


def select_safe_threshold_across_cells(
    *,
    score: object,
    supported: object,
    endpoint: object,
    cells: Sequence[Mapping[str, object]],
    gates: Mapping[str, float] = DEFAULT_GATES,
) -> dict[str, object] | None:
    """Choose a minimax SAFE threshold that passes all source/fold cells."""

    tables = _threshold_tables(
        score=score, supported=supported, endpoint=endpoint, cells=cells
    )
    if tables is None:
        return None
    coverage = np.asarray(tables["coverage_lower"], dtype=float)
    if np.any(coverage < float(gates["coverage_lower"])):
        return None
    feasible = (
        np.all(
            np.asarray(tables["recall_lower"])
            >= float(gates["protected_recall_lower"]),
            axis=0,
        )
        & np.all(
            np.asarray(tables["precision_lower"])
            >= float(gates["severe_rejection_precision_lower"]),
            axis=0,
        )
        & np.all(
            np.asarray(tables["savings_lower"]) >= float(gates["savings_lower"]),
            axis=0,
        )
    )
    candidates = np.flatnonzero(feasible)
    if not len(candidates):
        return None
    severe_recall = np.asarray(tables["severe_recall"], dtype=float)
    precision = np.asarray(tables["precision_lower"], dtype=float)
    savings = np.asarray(tables["savings_lower"], dtype=float)
    severe_rejected = np.asarray(tables["rejected_severe"], dtype=int)
    thresholds = np.asarray(tables["thresholds"], dtype=float)
    best = max(
        candidates.tolist(),
        key=lambda index: (
            float(np.min(severe_recall[:, index])),
            float(np.min(precision[:, index])),
            float(np.min(savings[:, index])),
            int(np.sum(severe_rejected[:, index])),
            float(thresholds[index]),
        ),
    )
    score_array = np.asarray(score, dtype=float)
    support_array = np.asarray(supported, dtype=bool) & np.isfinite(score_array)
    endpoint_array = np.asarray(endpoint, dtype=float)
    threshold = float(thresholds[best])
    return {
        "threshold": threshold,
        "passes_all_cells": True,
        "worst_cell_severe_recall": float(np.min(severe_recall[:, best])),
        "worst_cell_precision_lower": float(np.min(precision[:, best])),
        "worst_cell_savings_lower": float(np.min(savings[:, best])),
        "cell_records": _selected_cell_records(
            threshold=threshold,
            score=score_array,
            supported=support_array,
            endpoint=endpoint_array,
            cells=cells,
        ),
    }


def diagnose_safe_threshold_feasibility(
    *,
    score: object,
    supported: object,
    endpoint: object,
    cells: Sequence[Mapping[str, object]],
    gates: Mapping[str, float] = DEFAULT_GATES,
) -> dict[str, object]:
    """Find the threshold satisfying the largest number of frozen SAFE cells."""

    tables = _threshold_tables(
        score=score, supported=supported, endpoint=endpoint, cells=cells
    )
    if tables is None:
        raise ValueError("NEXT98 SAFE diagnostic has no supported score")
    coverage_ok = np.asarray(tables["coverage_lower"], dtype=float)[:, None] >= float(
        gates["coverage_lower"]
    )
    recall_ok = np.asarray(tables["recall_lower"], dtype=float) >= float(
        gates["protected_recall_lower"]
    )
    precision_ok = np.asarray(tables["precision_lower"], dtype=float) >= float(
        gates["severe_rejection_precision_lower"]
    )
    savings_ok = np.asarray(tables["savings_lower"], dtype=float) >= float(
        gates["savings_lower"]
    )
    cell_pass = coverage_ok & recall_ok & precision_ok & savings_ok
    pass_counts = cell_pass.sum(axis=0)
    severe_recall = np.asarray(tables["severe_recall"], dtype=float)
    precision = np.asarray(tables["precision_lower"], dtype=float)
    savings = np.asarray(tables["savings_lower"], dtype=float)
    thresholds = np.asarray(tables["thresholds"], dtype=float)
    best = max(
        range(len(thresholds)),
        key=lambda index: (
            int(pass_counts[index]),
            float(np.min(severe_recall[:, index])),
            float(np.min(precision[:, index])),
            float(np.min(savings[:, index])),
            float(thresholds[index]),
        ),
    )
    score_array = np.asarray(score, dtype=float)
    support_array = np.asarray(supported, dtype=bool) & np.isfinite(score_array)
    endpoint_array = np.asarray(endpoint, dtype=float)
    records = _selected_cell_records(
        threshold=float(thresholds[best]),
        score=score_array,
        supported=support_array,
        endpoint=endpoint_array,
        cells=cells,
    )
    for index, record in enumerate(records):
        record["passes_safe_gates"] = bool(cell_pass[index, best])
        record["gate_components"] = {
            "coverage": bool(coverage_ok[index, 0]),
            "protected_recall": bool(recall_ok[index, best]),
            "severe_precision": bool(precision_ok[index, best]),
            "savings": bool(savings_ok[index, best]),
        }
    return {
        "threshold": float(thresholds[best]),
        "passing_cells": int(pass_counts[best]),
        "total_cells": len(cells),
        "failing_cell_ids": [
            str(record["cell_id"])
            for record in records
            if not bool(record["passes_safe_gates"])
        ],
        "cell_records": records,
    }


def select_broad_threshold_across_cells(
    *,
    score: object,
    supported: object,
    endpoint: object,
    cells: Sequence[Mapping[str, object]],
    pauling_by_cell: Mapping[str, Mapping[str, object]],
    safe_threshold: float,
    broad_min_precision_lower: float = BROAD_MIN_PRECISION_LOWER,
) -> dict[str, object] | None:
    """Choose a lower threshold that Pareto-dominates Pauling in every cell."""

    tables = _threshold_tables(
        score=score, supported=supported, endpoint=endpoint, cells=cells
    )
    if tables is None:
        return None
    thresholds = np.asarray(tables["thresholds"], dtype=float)
    feasible = thresholds < float(safe_threshold)
    aggregate_indices: list[int] = []
    for cell_index, cell in enumerate(cells):
        cell_id = str(cell["cell_id"])
        baseline = pauling_by_cell.get(cell_id)
        if baseline is None:
            raise ValueError("NEXT98 Pauling cell baseline is missing")
        feasible &= np.asarray(tables["coverage_lower"])[cell_index] > float(
            baseline["coverage_lower"]
        )
        feasible &= np.asarray(tables["protected_kept"])[cell_index] >= int(
            baseline["protected_kept"]
        )
        feasible &= np.asarray(tables["rejected_severe"])[cell_index] > int(
            baseline["severe_rejected"]
        )
        feasible &= np.asarray(tables["precision_lower"])[cell_index] > float(
            baseline["severe_rejection_precision_lower"]
        )
        feasible &= np.asarray(tables["savings_lower"])[cell_index] > float(
            baseline["savings_lower"]
        )
        if cell.get("kind") == "source_aggregate":
            aggregate_indices.append(cell_index)
    if aggregate_indices:
        feasible &= np.all(
            np.asarray(tables["precision_lower"])[aggregate_indices]
            >= float(broad_min_precision_lower),
            axis=0,
        )
    candidates = np.flatnonzero(feasible)
    if not len(candidates):
        return None
    severe_recall = np.asarray(tables["severe_recall"], dtype=float)
    precision = np.asarray(tables["precision_lower"], dtype=float)
    savings = np.asarray(tables["savings_lower"], dtype=float)
    severe_rejected = np.asarray(tables["rejected_severe"], dtype=int)
    best = max(
        candidates.tolist(),
        key=lambda index: (
            float(np.min(severe_recall[:, index])),
            float(np.min(precision[:, index])),
            int(np.sum(severe_rejected[:, index])),
            float(np.min(savings[:, index])),
            -float(thresholds[index]),
        ),
    )
    score_array = np.asarray(score, dtype=float)
    support_array = np.asarray(supported, dtype=bool) & np.isfinite(score_array)
    endpoint_array = np.asarray(endpoint, dtype=float)
    threshold = float(thresholds[best])
    cell_records = _selected_cell_records(
        threshold=threshold,
        score=score_array,
        supported=support_array,
        endpoint=endpoint_array,
        cells=cells,
    )
    for record in cell_records:
        record["pauling_metrics"] = dict(pauling_by_cell[str(record["cell_id"])])
        record["pauling_dominance"] = pauling_dominance(
            record["metrics"], record["pauling_metrics"]
        )
    return {
        "threshold": threshold,
        "passes_all_cells": True,
        "worst_cell_severe_recall": float(np.min(severe_recall[:, best])),
        "worst_cell_precision_lower": float(np.min(precision[:, best])),
        "worst_cell_savings_lower": float(np.min(savings[:, best])),
        "cell_records": cell_records,
    }


def _as_list(value: object) -> list[object]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    raise ValueError("NEXT98 candidate list differs")


def _candidate_key(term_ids: Sequence[str], weights: Sequence[float]) -> str:
    return json.dumps(
        {"term_ids": list(term_ids), "weights": [float(value) for value in weights]},
        sort_keys=True,
        separators=(",", ":"),
    )


def _top_rows(
    frame: pd.DataFrame,
    *,
    by: Sequence[str],
    ascending: Sequence[bool],
    top_per_view: int,
) -> pd.DataFrame:
    available = [column for column in by if column in frame.columns]
    local_ascending = [ascending[list(by).index(column)] for column in available]
    if "formula_key" in frame.columns and "formula_key" not in available:
        available.append("formula_key")
        local_ascending.append(True)
    return frame.sort_values(
        available, ascending=local_ascending, kind="mergesort", na_position="last"
    ).head(int(top_per_view))


def build_candidate_specs(
    *,
    scigen_records: pd.DataFrame,
    wyformer_records: pd.DataFrame,
    eligible_terms: Sequence[Mapping[str, object]],
    top_per_view: int = TOP_PER_VIEW,
) -> list[dict[str, object]]:
    """Build a deterministic finite union of prior complete-search top slices."""

    if top_per_view <= 0:
        raise ValueError("NEXT98 top slice differs")
    eligible_ids = {str(term["term_id"]) for term in eligible_terms}
    selected_frames: list[tuple[str, pd.DataFrame]] = []
    scigen_views = [
        (
            [
                "metric_passes_operating_gates",
                "metric_severe_rejection_precision_lower",
                "metric_severe_rejected",
                "pooled_extreme_auc",
            ],
            [False, False, False, False],
        ),
        (["metric_severe_rejected", "metric_severe_rejection_precision_lower"], [False, False]),
        (["pooled_extreme_auc", "metric_severe_rejected"], [False, False]),
        (["metric_savings_lower", "metric_severe_rejected"], [False, False]),
    ]
    wyformer_views = [
        (
            [
                "passes_all_discovery_gates",
                "passing_fixed_folds",
                "severe_rejection_precision_lower",
                "severe_rejected",
            ],
            [False, False, False, False],
        ),
        (["severe_rejected", "severe_rejection_precision_lower"], [False, False]),
        (["pooled_extreme_auc", "severe_rejected"], [False, False]),
        (["savings_lower", "severe_rejected"], [False, False]),
    ]
    for by, ascending in scigen_views:
        selected_frames.append(
            (
                "next87_top_slice",
                _top_rows(
                    scigen_records,
                    by=by,
                    ascending=ascending,
                    top_per_view=top_per_view,
                ),
            )
        )
    for by, ascending in wyformer_views:
        selected_frames.append(
            (
                "next95_top_slice",
                _top_rows(
                    wyformer_records,
                    by=by,
                    ascending=ascending,
                    top_per_view=top_per_view,
                ),
            )
        )
    specs: dict[str, dict[str, object]] = {}
    for origin, selected in selected_frames:
        for _, row in selected.iterrows():
            if origin.startswith("next87"):
                term_ids = [str(value) for value in json.loads(str(row["term_ids_json"]))]
                weights = [float(value) for value in json.loads(str(row["weights_json"]))]
            else:
                term_ids = [str(value) for value in _as_list(row["term_ids"])]
                weights = [float(value) for value in _as_list(row["weights"])]
            if (
                not 1 <= len(term_ids) <= 3
                or len(weights) != len(term_ids)
                or len(set(term_ids)) != len(term_ids)
                or any(term_id not in eligible_ids for term_id in term_ids)
                or any(not math.isfinite(weight) or weight <= 0 for weight in weights)
            ):
                continue
            key = _candidate_key(term_ids, weights)
            if key not in specs:
                specs[key] = {
                    "candidate_key": key,
                    "term_ids": term_ids,
                    "weights": weights,
                    "origins": [origin],
                }
            elif origin not in specs[key]["origins"]:
                specs[key]["origins"].append(origin)
    for term_id in sorted(eligible_ids):
        key = _candidate_key([term_id], [1.0])
        specs.setdefault(
            key,
            {
                "candidate_key": key,
                "term_ids": [term_id],
                "weights": [1.0],
                "origins": ["all_eligible_singles"],
            },
        )
    return [specs[key] for key in sorted(specs)]


def _auc_pass(diagnostics: Mapping[str, object]) -> bool:
    return bool(
        diagnostics["pooled_extreme_auc"] is not None
        and float(diagnostics["pooled_extreme_auc"])
        >= AUC_GATES["pooled_extreme_auc"]
        and diagnostics["macro_lattice_auc"] is not None
        and float(diagnostics["macro_lattice_auc"])
        >= AUC_GATES["macro_lattice_auc"]
        and diagnostics["worst_lattice_auc"] is not None
        and float(diagnostics["worst_lattice_auc"])
        >= AUC_GATES["worst_lattice_auc"]
        and int(diagnostics["evaluable_lattices"])
        >= AUC_GATES["evaluable_lattices"]
    )


def search_cross_source_law(
    *,
    features: pd.DataFrame,
    endpoint: object,
    eligible_terms: Sequence[Mapping[str, object]],
    candidate_specs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Evaluate the finite catalogue under source aggregate and fold gates."""

    required = {
        "material_id",
        "source_dataset",
        "reduced_formula",
        "crystal_system",
        "pauling_p2_p5_decision",
    }
    endpoint_array = np.asarray(endpoint, dtype=float)
    if (
        required - set(features.columns)
        or len(features) != len(endpoint_array)
        or not np.isfinite(endpoint_array).all()
        or set(features["source_dataset"].astype(str)) != {"scigen", "wyformer"}
    ):
        raise ValueError("NEXT98 cross-source discovery arrays differ")
    terms = {str(term["term_id"]): dict(term) for term in eligible_terms}
    risk_by_term: dict[str, np.ndarray] = {}
    support_by_term: dict[str, np.ndarray] = {}
    for term_id, term in terms.items():
        risk, support = _term_risk(features, term)
        risk_by_term[term_id] = risk
        support_by_term[term_id] = support
    folds = assign_group_folds(features["reduced_formula"].astype(str).to_numpy())
    cells = build_source_fold_cells(
        source=features["source_dataset"].astype(str).to_numpy(), folds=folds
    )
    pauling_by_cell = {
        str(cell["cell_id"]): _pauling_baseline(
            features.loc[np.asarray(cell["mask"], dtype=bool)],
            endpoint_array[np.asarray(cell["mask"], dtype=bool)],
        )
        for cell in cells
    }
    source_masks = {
        source: features["source_dataset"].astype(str).to_numpy() == source
        for source in ("scigen", "wyformer")
    }
    records: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for spec in candidate_specs:
        term_ids = [str(value) for value in spec["term_ids"]]
        weights = np.asarray(spec["weights"], dtype=float)
        if any(term_id not in terms for term_id in term_ids):
            continue
        score = np.sum(
            np.column_stack([risk_by_term[term_id] for term_id in term_ids])
            * weights[None, :],
            axis=1,
        )
        supported = np.all(
            np.column_stack([support_by_term[term_id] for term_id in term_ids]), axis=1
        )
        score = np.asarray(score, dtype=float)
        score[~supported] = np.nan
        source_diagnostics: dict[str, object] = {}
        auc_all_sources = True
        for source, mask in source_masks.items():
            diagnostics = auc_diagnostics(
                score=score[mask],
                supported=supported[mask],
                distortion_ratio=endpoint_array[mask],
                lattice_class=features.loc[mask, "crystal_system"].astype(str).to_numpy(),
            )
            diagnostics["passes_auc_gates"] = _auc_pass(diagnostics)
            source_diagnostics[source] = diagnostics
            auc_all_sources &= bool(diagnostics["passes_auc_gates"])
        safe = select_safe_threshold_across_cells(
            score=score,
            supported=supported,
            endpoint=endpoint_array,
            cells=cells,
        )
        broad = None
        if safe is not None:
            broad = select_broad_threshold_across_cells(
                score=score,
                supported=supported,
                endpoint=endpoint_array,
                cells=cells,
                pauling_by_cell=pauling_by_cell,
                safe_threshold=float(safe["threshold"]),
            )
        passed = bool(auc_all_sources and safe is not None and broad is not None)
        record = {
            "candidate_key": str(spec["candidate_key"]),
            "term_ids_json": json.dumps(term_ids, separators=(",", ":")),
            "weights_json": json.dumps(weights.tolist(), separators=(",", ":")),
            "origins_json": json.dumps(list(spec.get("origins", [])), separators=(",", ":")),
            "term_count": len(term_ids),
            "safe_threshold": None if safe is None else float(safe["threshold"]),
            "broad_threshold": None if broad is None else float(broad["threshold"]),
            "safe_worst_cell_severe_recall": None
            if safe is None
            else float(safe["worst_cell_severe_recall"]),
            "safe_worst_cell_precision_lower": None
            if safe is None
            else float(safe["worst_cell_precision_lower"]),
            "broad_worst_cell_severe_recall": None
            if broad is None
            else float(broad["worst_cell_severe_recall"]),
            "scigen_pooled_auc": source_diagnostics["scigen"]["pooled_extreme_auc"],
            "scigen_macro_auc": source_diagnostics["scigen"]["macro_lattice_auc"],
            "scigen_worst_auc": source_diagnostics["scigen"]["worst_lattice_auc"],
            "wyformer_pooled_auc": source_diagnostics["wyformer"]["pooled_extreme_auc"],
            "wyformer_macro_auc": source_diagnostics["wyformer"]["macro_lattice_auc"],
            "wyformer_worst_auc": source_diagnostics["wyformer"]["worst_lattice_auc"],
            "passes_source_auc_gates": bool(auc_all_sources),
            "passes_safe_all_cells": safe is not None,
            "passes_broad_all_cells": broad is not None,
            "passes_all_discovery_gates": passed,
        }
        records.append(record)
        safe_recall = (
            float(safe["worst_cell_severe_recall"]) if safe is not None else -1.0
        )
        safe_precision = (
            float(safe["worst_cell_precision_lower"]) if safe is not None else -1.0
        )
        worst_source_auc = min(
            float(source_diagnostics[source]["pooled_extreme_auc"] or -1.0)
            for source in source_masks
        )
        rank = (
            int(passed),
            int(safe is not None),
            int(broad is not None),
            int(auc_all_sources),
            safe_recall,
            safe_precision,
            worst_source_auc,
            -len(term_ids),
        )
        identity = str(spec["candidate_key"])
        if best is None or rank > best["rank"] or (
            rank == best["rank"] and identity < best["identity"]
        ):
            formula_terms = [
                {**terms[term_id], "weight": float(weight)}
                for term_id, weight in zip(term_ids, weights)
            ]
            best = {
                "rank": rank,
                "identity": identity,
                "record": record,
                "formula": {
                    "kind": "nonnegative_sum_of_at_most_three_one_sided_robust_hinges",
                    "missing_policy": "ABSTAIN",
                    "terms": formula_terms,
                    "safe_threshold": None
                    if safe is None
                    else float(safe["threshold"]),
                    "broad_threshold": None
                    if broad is None
                    else float(broad["threshold"]),
                },
                "safe": safe,
                "broad": broad,
                "source_diagnostics": source_diagnostics,
            }
    if best is None:
        raise RuntimeError("NEXT98 search produced no candidate")
    return {
        "candidate_records": records,
        "candidate_count": len(records),
        "cells": [
            {key: value for key, value in cell.items() if key != "mask"}
            for cell in cells
        ],
        "pauling_by_cell": pauling_by_cell,
        "selected": best,
    }


def run_cross_source_discovery_search(
    *,
    scigen_feature_dir: Path,
    scigen_term_catalogue_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    scigen_search_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    wyformer_search_dir: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the finite discovery-only cross-source search."""

    scigen_feature_root = Path(scigen_feature_dir).resolve()
    scigen_term_root = Path(scigen_term_catalogue_dir).resolve()
    scigen_endpoint_root = Path(scigen_discovery_endpoint_dir).resolve()
    scigen_search_root = Path(scigen_search_dir).resolve()
    wyformer_feature_root = Path(wyformer_feature_dir).resolve()
    wyformer_endpoint_root = Path(wyformer_discovery_endpoint_dir).resolve()
    wyformer_search_root = Path(wyformer_search_dir).resolve()
    paths = {
        "scigen_feature_manifest": scigen_feature_root
        / SCIGEN_FEATURE_MANIFEST_NAME,
        "scigen_feature_catalogue": scigen_feature_root
        / SCIGEN_FEATURE_CATALOGUE_NAME,
        "scigen_features": scigen_feature_root
        / SCIGEN_FEATURE_NAMES["discovery"],
        "scigen_term_manifest": scigen_term_root / SCIGEN_TERM_MANIFEST_NAME,
        "scigen_term_catalogue": scigen_term_root / SCIGEN_TERM_CATALOGUE_NAME,
        "scigen_endpoint_manifest": scigen_endpoint_root
        / SCIGEN_ENDPOINT_MANIFEST_NAME,
        "scigen_endpoint": scigen_endpoint_root / SCIGEN_ENDPOINT_NAME,
        "scigen_search_records": scigen_search_root
        / "next87_complete_candidate_search.parquet",
        "wyformer_feature_manifest": wyformer_feature_root
        / WYFORMER_FEATURE_MANIFEST_NAME,
        "wyformer_features": wyformer_feature_root
        / WYFORMER_FEATURE_NAMES["discovery"],
        "wyformer_endpoint_manifest": wyformer_endpoint_root
        / WYFORMER_ENDPOINT_MANIFEST_NAME,
        "wyformer_endpoint": wyformer_endpoint_root / WYFORMER_ENDPOINT_NAME,
        "wyformer_search_records": wyformer_search_root
        / "next95_complete_candidate_search.parquet",
        "design": Path(design_path).resolve(),
    }
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT98 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT98 formal input identity differs")

    scigen_feature_manifest = _read_json(paths["scigen_feature_manifest"])
    scigen_term_manifest = _read_json(paths["scigen_term_manifest"])
    scigen_endpoint_manifest = _read_json(paths["scigen_endpoint_manifest"])
    wyformer_feature_manifest = _read_json(paths["wyformer_feature_manifest"])
    wyformer_endpoint_manifest = _read_json(paths["wyformer_endpoint_manifest"])
    if (
        scigen_feature_manifest.get("protocol") != SCIGEN_FEATURE_PROTOCOL
        or scigen_feature_manifest.get("labels_opened") is not False
        or scigen_feature_manifest.get("endpoint_payloads_opened") is not False
        or scigen_feature_manifest.get("dft_values_used_by_features") is not False
        or scigen_term_manifest.get("protocol") != SCIGEN_TERM_PROTOCOL
        or scigen_endpoint_manifest.get("protocol") != SCIGEN_ENDPOINT_PROTOCOL
        or scigen_endpoint_manifest.get("partition_role") != "discovery"
        or scigen_endpoint_manifest.get("energy_columns_retained") is not False
        or scigen_endpoint_manifest.get("relaxed_structures_opened") is not False
    ):
        raise ValueError("NEXT98 SCIGEN discovery provenance differs")
    if (
        wyformer_feature_manifest.get("protocol") != WYFORMER_FEATURE_PROTOCOL
        or wyformer_feature_manifest.get("labels_opened") is not False
        or wyformer_feature_manifest.get("endpoint_payloads_opened") is not False
        or wyformer_feature_manifest.get("dft_values_used_by_features") is not False
        or wyformer_endpoint_manifest.get("protocol") != WYFORMER_ENDPOINT_PROTOCOL
        or wyformer_endpoint_manifest.get("partition_role") != "discovery"
        or wyformer_endpoint_manifest.get("endpoint_sha256")
        != input_hashes["wyformer_endpoint"]
    ):
        raise ValueError("NEXT98 WyFormer discovery provenance differs")

    scigen_features = pd.read_parquet(paths["scigen_features"])
    scigen_endpoints = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_features = pd.read_parquet(paths["wyformer_features"])
    wyformer_endpoints = pd.read_parquet(paths["wyformer_endpoint"])
    if (
        scigen_features["material_id"].astype(str).duplicated().any()
        or scigen_endpoints["material_id"].astype(str).duplicated().any()
        or wyformer_features["material_id"].astype(str).duplicated().any()
        or wyformer_endpoints["material_id"].astype(str).duplicated().any()
    ):
        raise ValueError("NEXT98 discovery ids are duplicated")
    scigen = scigen_features.merge(
        scigen_endpoints.loc[:, ["material_id", "distortion_ratio"]],
        on="material_id",
        how="inner",
        validate="one_to_one",
    )
    wyformer = wyformer_features.merge(
        wyformer_endpoints.loc[:, ["material_id", "endpoint_stratum"]],
        on="material_id",
        how="inner",
        validate="one_to_one",
    )
    if len(scigen) != len(scigen_features) or len(scigen) != len(scigen_endpoints):
        raise ValueError("NEXT98 SCIGEN row accounting differs")
    if len(wyformer) != len(wyformer_features) or len(wyformer) != len(wyformer_endpoints):
        raise ValueError("NEXT98 WyFormer row accounting differs")
    scigen = scigen.copy()
    wyformer = wyformer.copy()
    scigen["material_id"] = "scigen:" + scigen["material_id"].astype(str)
    wyformer["material_id"] = "wyformer:" + wyformer["material_id"].astype(str)
    scigen["source_dataset"] = "scigen"
    wyformer["source_dataset"] = "wyformer"
    scigen["crystal_system"] = scigen["lattice_class"].astype(str)
    endpoint = np.concatenate(
        [
            pd.to_numeric(scigen["distortion_ratio"], errors="coerce").to_numpy(float),
            _endpoint_numeric(wyformer["endpoint_stratum"]),
        ]
    )
    combined = pd.concat([scigen, wyformer], ignore_index=True, sort=False)
    if not np.isfinite(endpoint).all() or len(combined) != len(endpoint):
        raise ValueError("NEXT98 endpoint conversion differs")

    template_catalogue = _read_json(paths["scigen_term_catalogue"])
    template_terms = template_catalogue.get("eligible_terms")
    if not isinstance(template_terms, list):
        raise ValueError("NEXT98 term templates differ")
    eligible_terms, excluded_terms = recalibrate_terms(combined, template_terms)
    scigen_records = pd.read_parquet(paths["scigen_search_records"])
    wyformer_records = pd.read_parquet(paths["wyformer_search_records"])
    candidate_specs = build_candidate_specs(
        scigen_records=scigen_records,
        wyformer_records=wyformer_records,
        eligible_terms=eligible_terms,
    )
    started = time.perf_counter()
    result = search_cross_source_law(
        features=combined,
        endpoint=endpoint,
        eligible_terms=eligible_terms,
        candidate_specs=candidate_specs,
    )
    elapsed = time.perf_counter() - started
    selected = result["selected"]
    passes = bool(selected["record"]["passes_all_discovery_gates"])

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256_file(source_path)
    output_paths: list[Path] = []
    try:
        catalogue = {
            "protocol": PROTOCOL,
            "calibration": "pooled label-free SCIGEN plus WyFormer discovery x0 features",
            "eligible_terms": eligible_terms,
            "excluded_terms": excluded_terms,
            "candidate_generation": {
                "top_per_view": TOP_PER_VIEW,
                "prior_complete_searches": ["NEXT87", "NEXT95"],
                "all_eligible_singles_included": True,
                "candidate_count": len(candidate_specs),
            },
        }
        evaluation = {
            "protocol": PROTOCOL,
            "evaluation_mode": "cross_source_discovery_only_no_validation_outputs",
            "rows": {
                "scigen": int(len(scigen)),
                "wyformer": int(len(wyformer)),
                "total": int(len(combined)),
            },
            "candidate_count": int(result["candidate_count"]),
            "elapsed_seconds": elapsed,
            "safe_gates": dict(DEFAULT_GATES),
            "source_auc_gates": dict(AUC_GATES),
            "broad_min_severe_precision_lower": BROAD_MIN_PRECISION_LOWER,
            "selected_record": selected["record"],
            "selected_formula": selected["formula"],
            "selected_safe": selected["safe"],
            "selected_broad": selected["broad"],
            "selected_source_diagnostics": selected["source_diagnostics"],
            "pauling_by_cell": result["pauling_by_cell"],
            "cells": result["cells"],
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
        }
        catalogue_path = staging / CATALOGUE_NAME
        evaluation_path = staging / EVALUATION_NAME
        search_path = staging / SEARCH_NAME
        _write_json(catalogue_path, catalogue)
        _write_json(evaluation_path, evaluation)
        pd.DataFrame(result["candidate_records"]).to_parquet(search_path, index=False)
        output_paths.extend([catalogue_path, evaluation_path, search_path])
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "opened_validation_outputs_used": False,
            "scigen_replication_endpoint_opened": False,
            "wyformer_replication_endpoint_opened": False,
            "formula_or_threshold_changed_after_search": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "physical_relaxation_executed": False,
            "scientific_improvement_claim": False,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": {
                "src/next98_cross_source_discovery_search.py": source_hash
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT98 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT98 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "BROAD_MIN_PRECISION_LOWER",
    "build_source_fold_cells",
    "build_candidate_specs",
    "diagnose_safe_threshold_feasibility",
    "run_cross_source_discovery_search",
    "search_cross_source_law",
    "select_broad_threshold_across_cells",
    "select_safe_threshold_across_cells",
]
