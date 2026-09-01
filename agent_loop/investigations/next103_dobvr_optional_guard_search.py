#!/usr/bin/env python3
"""Cross-source discovery search for one optional DOBVR risk guard."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.next87_scigen_sparse_law_search import (
    _pauling_baseline,
    _term_risk,
    _transformed_column,
    assign_group_folds,
    auc_diagnostics,
)
from src.next85_scigen_label_free_features import (
    FEATURE_NAMES as SCIGEN_FEATURE_NAMES,
)
from src.next86_scigen_endpoint_router import (
    ENDPOINT_NAME as SCIGEN_ENDPOINT_NAME,
)
from src.next93_wyformer_source_lockbox import _sha256_file, _write_json
from src.next93b_wyformer_blind_lockbox import (
    ENDPOINT_NAME as WYFORMER_ENDPOINT_NAME,
)
from src.next94_wyformer_label_free_features import (
    FEATURE_NAMES as WYFORMER_FEATURE_NAMES,
)
from src.next95_wyformer_sparse_law_search import DEFAULT_GATES, _endpoint_numeric
from src.next98_cross_source_discovery_search import (
    AUC_GATES,
    BROAD_MIN_PRECISION_LOWER,
    CATALOGUE_NAME as NEXT98_CATALOGUE_NAME,
    MANIFEST_NAME as NEXT98_MANIFEST_NAME,
    PROTOCOL as NEXT98_PROTOCOL,
    _auc_pass,
    _read_json,
    _selected_cell_records,
    _threshold_tables,
    build_source_fold_cells,
    select_broad_threshold_across_cells,
)
from src.next98b_cross_source_exhaustive_search import (
    MANIFEST_NAME as NEXT98B_MANIFEST_NAME,
    PROTOCOL as NEXT98B_PROTOCOL,
    SEARCH_NAME as NEXT98B_SEARCH_NAME,
)
from src.next102_cross_source_dobvr_features import (
    FEATURE_COLUMNS as NEXT102_FEATURE_COLUMNS,
    FEATURE_NAMES as NEXT102_FEATURE_NAMES,
    MANIFEST_NAME as NEXT102_MANIFEST_NAME,
    PROTOCOL as NEXT102_PROTOCOL,
)


PROTOCOL = "2026-08-04-next103-dobvr-optional-guard-search-v1"
OPTIONAL_WEIGHT_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
MIN_SOURCE_COVERAGE = 0.15
EXPECTED_AUC_PASSING_BASES = 67
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT103_OPTIONAL_TERM_CATALOGUE.json"
EVALUATION_NAME = "NEXT103_DISCOVERY_EVALUATION.json"
SEARCH_NAME = "next103_optional_guard_candidate_search.parquet"
EXPECTED_INPUT_SHA256 = {
    "scigen_features": "7031d86e4fb6e469c674d208f680ace1dbe5e11e45f3d4b2befefd747efdde16",
    "scigen_endpoint": "f86cff6f5e9124ee82aae13911ffe55a125c6fe111fc1f64122a610febf67958",
    "wyformer_features": "c515baec0fccef5bc03c7672f1d4e1aca278f5ed4d7b6f1bf7f66c734e2b87f7",
    "wyformer_endpoint": "f39836e62a1da03ed823479e87d6f75fc0d01da60a8c0a2faa696638cc2fb9d7",
    "next98_manifest": "5fcd924b125767e52ac1826203595692af868ab35366899e12b82aea2726e32c",
    "next98_term_catalogue": "f2165f548a56cda04559a11a0d575f0654d3e8a17cf3b85b76e7974ea65dee41",
    "next98b_manifest": "b20d2f500ce74a6fd8b1a8a992bca3fff3ee5952fc38c09d3ad34ca317c3084d",
    "next98b_search_records": "748a4623ecfc725636837f3944b70482a97b2df39a495a81e3f8e09f5d09a4e4",
    "next102_manifest": "1dc5040d1fb226f2a5c0e1ea9ff38fb97cc6015abeb9e9a7ca394f997f7523c5",
    "next102_scigen_features": "2fa8108a3273ecd20e5268de3015fe70f0aee038f7011316f1eb0a1c4aafca39",
    "next102_wyformer_features": "6d92ac670d34b8801777fcf24d62fee22684dad1e6e9cc5c82f55ba957996686",
    "design": "d291ec9fe434bea2b4057e46a9fe1ae98a109a4b02c9aac246302e9e8ac70584",
}


def _template(
    feature: str, direction: int, group: str, support_column: str
) -> dict[str, object]:
    label = "high" if direction == 1 else "low"
    return {
        "term_id": f"{feature}__{label}",
        "feature": feature,
        "direction": direction,
        "transform": "log1p_nonnegative",
        "group": group,
        "support_column": support_column,
    }


OPTIONAL_TERM_TEMPLATES = tuple(
    [
        _template(name, direction, "dobvr_strict", "dobvr_supported")
        for name, direction in (
            ("dobvr_best_mismatch_rms", 1),
            ("dobvr_best_mismatch_q95", 1),
            ("dobvr_median_mismatch_rms", 1),
            ("dobvr_best_parameter_generic_fraction", 1),
            ("dobvr_assignment_log_count", 1),
            ("dobvr_runner_up_gap_rms", -1),
        )
    ]
    + [
        _template(name, direction, "dobvr_expanded", "dobvrb_supported")
        for name, direction in (
            ("dobvrb_best_mismatch_rms", 1),
            ("dobvrb_best_mismatch_q95", 1),
            ("dobvrb_median_mismatch_rms", 1),
            ("dobvrb_best_parameter_generic_fraction", 1),
            ("dobvrb_assignment_log_count", 1),
            ("dobvrb_best_catalogue_tier", 1),
            ("dobvrb_runner_up_gap_rms", -1),
            ("dobvrb_core_assignment_fraction", -1),
            ("dobvrb_best_eneg_margin", -1),
        )
    ]
)


def compose_optional_guard_score(
    *,
    base_score: object,
    base_supported: object,
    guard_risk: object,
    guard_active: object,
    guard_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Add an active guard without changing the base law's support mask."""

    base = np.asarray(base_score, dtype=float)
    supported = np.asarray(base_supported, dtype=bool)
    risk = np.asarray(guard_risk, dtype=float)
    active = np.asarray(guard_active, dtype=bool)
    weight = float(guard_weight)
    if (
        base.ndim != 1
        or supported.shape != base.shape
        or risk.shape != base.shape
        or active.shape != base.shape
        or not np.isfinite(base[supported]).all()
        or not np.isfinite(risk[active]).all()
        or not np.isfinite(weight)
        or weight < 0.0
    ):
        raise ValueError("NEXT103 optional-guard arrays differ")
    score = base.copy()
    score += weight * np.where(active, risk, 0.0)
    score[~supported] = np.nan
    return score, supported.copy()


def calibrate_optional_terms(
    features: pd.DataFrame,
    *,
    templates: Sequence[Mapping[str, object]],
    min_source_coverage: float = 0.15,
    min_unique_values: int = 8,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Freeze pooled label-free transforms with a per-source activation floor."""

    if (
        not 0.0 < float(min_source_coverage) <= 1.0
        or type(min_unique_values) is not int
        or min_unique_values < 2
        or "source_dataset" not in features
        or set(features["source_dataset"].astype(str)) != {"scigen", "wyformer"}
    ):
        raise ValueError("NEXT103 optional calibration boundary differs")
    eligible: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    source_values = features["source_dataset"].astype(str).to_numpy()
    for template in sorted(templates, key=lambda item: str(item.get("term_id"))):
        term_id = template.get("term_id")
        feature = template.get("feature")
        direction = template.get("direction")
        transform = template.get("transform")
        group = template.get("group")
        support_column = template.get("support_column")
        reason: str | None = None
        if (
            not isinstance(term_id, str)
            or not isinstance(feature, str)
            or feature not in features
            or direction not in {-1, 1}
            or transform not in {"log1p_nonnegative", "asinh"}
            or not isinstance(group, str)
            or not isinstance(support_column, str)
            or support_column not in features
        ):
            reason = "template_schema_or_feature_missing"
            transformed = np.full(len(features), np.nan)
            active = np.zeros(len(features), dtype=bool)
        else:
            raw = pd.to_numeric(features[feature], errors="coerce").to_numpy(float)
            transformed = _transformed_column(raw, str(transform))
            active = (
                features[support_column].fillna(False).astype(bool).to_numpy()
                & np.isfinite(transformed)
            )
        source_coverage = {
            source: float(active[source_values == source].mean())
            for source in ("scigen", "wyformer")
        }
        values = transformed[active]
        if reason is None and min(source_coverage.values()) < float(
            min_source_coverage
        ):
            reason = "source_coverage_below_gate"
        if reason is None and len(np.unique(values)) < min_unique_values:
            reason = "fewer_than_required_unique_values"
        if reason is None:
            q10, center, q90 = np.quantile(values, [0.1, 0.5, 0.9])
            scale = float((q90 - q10) / 2.0)
            if not math.isfinite(scale) or scale <= 1.0e-12:
                reason = "robust_scale_degenerate"
        if reason is not None:
            excluded.append(
                {
                    "term_id": term_id,
                    "feature": feature,
                    "source_coverage": source_coverage,
                    "reason": reason,
                }
            )
            continue
        eligible.append(
            {
                **dict(template),
                "center": float(center),
                "scale": float(scale),
                "finite_rows": int(active.sum()),
                "source_coverage": source_coverage,
                "unique_transformed_values": int(len(np.unique(values))),
                "transformed_q10": float(q10),
                "transformed_q90": float(q90),
                "missing_policy": "OPTIONAL_GUARD_OFF_KEEP_BASE",
            }
        )
    return eligible, excluded


def _candidate_key(
    *,
    base_term_ids: Sequence[str],
    base_weights: Sequence[float],
    optional_term_id: str | None,
    optional_weight: float,
) -> str:
    return json.dumps(
        {
            "base_term_ids": list(base_term_ids),
            "base_weights": [float(value) for value in base_weights],
            "optional_term_id": optional_term_id,
            "optional_weight": float(optional_weight),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def build_optional_guard_candidate_specs(
    *,
    base_records: pd.DataFrame,
    old_term_ids: set[str],
    optional_terms: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Add zero or one finite-grid optional guard to every AUC-passing base."""

    required = {"passes_source_auc_gates", "term_ids_json", "weights_json"}
    if required - set(base_records.columns):
        raise ValueError("NEXT103 base candidate columns differ")
    optional_ids = sorted(str(term["term_id"]) for term in optional_terms)
    specs: dict[str, dict[str, object]] = {}
    for _, row in base_records.iterrows():
        if not bool(row["passes_source_auc_gates"]):
            continue
        term_ids = [str(value) for value in json.loads(str(row["term_ids_json"]))]
        weights = [float(value) for value in json.loads(str(row["weights_json"]))]
        if (
            not 1 <= len(term_ids) <= 3
            or len(weights) != len(term_ids)
            or len(set(term_ids)) != len(term_ids)
            or any(term_id not in old_term_ids for term_id in term_ids)
            or any(not math.isfinite(weight) or weight <= 0.0 for weight in weights)
        ):
            raise ValueError("NEXT103 base formula differs")

        def add(optional_term_id: str | None, optional_weight: float) -> None:
            key = _candidate_key(
                base_term_ids=term_ids,
                base_weights=weights,
                optional_term_id=optional_term_id,
                optional_weight=optional_weight,
            )
            specs[key] = {
                "candidate_key": key,
                "base_term_ids": term_ids,
                "base_weights": weights,
                "optional_term_id": optional_term_id,
                "optional_weight": float(optional_weight),
            }

        add(None, 0.0)
        for optional_term_id in optional_ids:
            for optional_weight in OPTIONAL_WEIGHT_GRID:
                add(optional_term_id, optional_weight)
    return [specs[key] for key in sorted(specs)]


def _optional_term_risk(
    features: pd.DataFrame, term: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    risk, finite = _term_risk(features, term)
    support_column = term.get("support_column")
    if not isinstance(support_column, str) or support_column not in features:
        raise ValueError("NEXT103 optional support column differs")
    active = (
        finite
        & features[support_column].fillna(False).astype(bool).to_numpy()
    )
    risk = np.asarray(risk, dtype=float)
    risk[~active] = 0.0
    return risk, active


def select_safe_and_diagnostic_once(
    *,
    score: object,
    supported: object,
    endpoint: object,
    cells: Sequence[Mapping[str, object]],
    gates: Mapping[str, float] = DEFAULT_GATES,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    """Match NEXT98 SAFE selection and diagnostic from one threshold table."""

    tables = _threshold_tables(
        score=score, supported=supported, endpoint=endpoint, cells=cells
    )
    if tables is None:
        raise ValueError("NEXT103 SAFE diagnostic has no supported score")
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
    severe_rejected = np.asarray(tables["rejected_severe"], dtype=int)
    thresholds = np.asarray(tables["thresholds"], dtype=float)
    score_array = np.asarray(score, dtype=float)
    support_array = np.asarray(supported, dtype=bool) & np.isfinite(score_array)
    endpoint_array = np.asarray(endpoint, dtype=float)

    diagnostic_index = max(
        range(len(thresholds)),
        key=lambda index: (
            int(pass_counts[index]),
            float(np.min(severe_recall[:, index])),
            float(np.min(precision[:, index])),
            float(np.min(savings[:, index])),
            float(thresholds[index]),
        ),
    )
    diagnostic_records = _selected_cell_records(
        threshold=float(thresholds[diagnostic_index]),
        score=score_array,
        supported=support_array,
        endpoint=endpoint_array,
        cells=cells,
    )
    for index, record in enumerate(diagnostic_records):
        record["passes_safe_gates"] = bool(cell_pass[index, diagnostic_index])
        record["gate_components"] = {
            "coverage": bool(coverage_ok[index, 0]),
            "protected_recall": bool(recall_ok[index, diagnostic_index]),
            "severe_precision": bool(precision_ok[index, diagnostic_index]),
            "savings": bool(savings_ok[index, diagnostic_index]),
        }
    diagnostic = {
        "threshold": float(thresholds[diagnostic_index]),
        "passing_cells": int(pass_counts[diagnostic_index]),
        "total_cells": len(cells),
        "failing_cell_ids": [
            str(record["cell_id"])
            for record in diagnostic_records
            if not bool(record["passes_safe_gates"])
        ],
        "cell_records": diagnostic_records,
    }

    feasible = np.all(cell_pass, axis=0)
    candidates = np.flatnonzero(feasible)
    if not len(candidates):
        return None, diagnostic
    safe_index = max(
        candidates.tolist(),
        key=lambda index: (
            float(np.min(severe_recall[:, index])),
            float(np.min(precision[:, index])),
            float(np.min(savings[:, index])),
            int(np.sum(severe_rejected[:, index])),
            float(thresholds[index]),
        ),
    )
    safe_threshold = float(thresholds[safe_index])
    safe = {
        "threshold": safe_threshold,
        "passes_all_cells": True,
        "worst_cell_severe_recall": float(np.min(severe_recall[:, safe_index])),
        "worst_cell_precision_lower": float(np.min(precision[:, safe_index])),
        "worst_cell_savings_lower": float(np.min(savings[:, safe_index])),
        "cell_records": _selected_cell_records(
            threshold=safe_threshold,
            score=score_array,
            supported=support_array,
            endpoint=endpoint_array,
            cells=cells,
        ),
    }
    return safe, diagnostic


def search_optional_guard_laws(
    *,
    features: pd.DataFrame,
    endpoint: object,
    old_terms: Sequence[Mapping[str, object]],
    optional_terms: Sequence[Mapping[str, object]],
    candidate_specs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Evaluate every frozen base plus zero/one fail-open DOBVR guard."""

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
        raise ValueError("NEXT103 cross-source discovery arrays differ")
    old_by_id = {str(term["term_id"]): dict(term) for term in old_terms}
    optional_by_id = {
        str(term["term_id"]): dict(term) for term in optional_terms
    }
    old_risk: dict[str, np.ndarray] = {}
    old_support: dict[str, np.ndarray] = {}
    for term_id, term in old_by_id.items():
        old_risk[term_id], old_support[term_id] = _term_risk(features, term)
    optional_risk: dict[str, np.ndarray] = {}
    optional_active: dict[str, np.ndarray] = {}
    for term_id, term in optional_by_id.items():
        optional_risk[term_id], optional_active[term_id] = _optional_term_risk(
            features, term
        )

    folds = assign_group_folds(features["reduced_formula"].astype(str).to_numpy())
    source_values = features["source_dataset"].astype(str).to_numpy()
    cells = build_source_fold_cells(source=source_values, folds=folds)
    pauling_by_cell = {
        str(cell["cell_id"]): _pauling_baseline(
            features.loc[np.asarray(cell["mask"], dtype=bool)],
            endpoint_array[np.asarray(cell["mask"], dtype=bool)],
        )
        for cell in cells
    }
    source_masks = {
        source: source_values == source for source in ("scigen", "wyformer")
    }
    base_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    records: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for spec in candidate_specs:
        base_term_ids = [str(value) for value in spec["base_term_ids"]]
        base_weights = np.asarray(spec["base_weights"], dtype=float)
        optional_term_id_value = spec.get("optional_term_id")
        optional_term_id = (
            None
            if optional_term_id_value is None
            else str(optional_term_id_value)
        )
        optional_weight = float(spec.get("optional_weight", 0.0))
        if (
            not base_term_ids
            or len(base_weights) != len(base_term_ids)
            or any(term_id not in old_by_id for term_id in base_term_ids)
            or (optional_term_id is not None and optional_term_id not in optional_by_id)
            or not np.isfinite(base_weights).all()
            or np.any(base_weights <= 0.0)
            or not math.isfinite(optional_weight)
            or optional_weight < 0.0
        ):
            raise ValueError("NEXT103 candidate specification differs")
        base_key = json.dumps(
            [base_term_ids, base_weights.tolist()], separators=(",", ":")
        )
        if base_key not in base_cache:
            base_score = np.sum(
                np.column_stack([old_risk[term_id] for term_id in base_term_ids])
                * base_weights[None, :],
                axis=1,
            )
            base_supported = np.all(
                np.column_stack(
                    [old_support[term_id] for term_id in base_term_ids]
                ),
                axis=1,
            )
            base_score = np.asarray(base_score, dtype=float)
            base_score[~base_supported] = np.nan
            base_cache[base_key] = (base_score, base_supported)
        base_score, base_supported = base_cache[base_key]
        if optional_term_id is None:
            guard_risk = np.zeros(len(features), dtype=float)
            guard_active = np.zeros(len(features), dtype=bool)
        else:
            guard_risk = optional_risk[optional_term_id]
            guard_active = optional_active[optional_term_id]
        score, supported = compose_optional_guard_score(
            base_score=base_score,
            base_supported=base_supported,
            guard_risk=guard_risk,
            guard_active=guard_active,
            guard_weight=optional_weight,
        )

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
        safe, safe_diagnostic = select_safe_and_diagnostic_once(
            score=score,
            supported=supported,
            endpoint=endpoint_array,
            cells=cells,
        )
        safe_passing_cells = int(safe_diagnostic["passing_cells"])
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
        activation_by_source = {
            source: float(guard_active[mask].mean())
            for source, mask in source_masks.items()
        }
        record = {
            "candidate_key": str(spec["candidate_key"]),
            "base_term_ids_json": json.dumps(base_term_ids, separators=(",", ":")),
            "base_weights_json": json.dumps(base_weights.tolist(), separators=(",", ":")),
            "optional_term_id": optional_term_id,
            "optional_weight": optional_weight,
            "term_count": len(base_term_ids) + int(optional_term_id is not None),
            "supported_rows": int(supported.sum()),
            "support_coverage": float(supported.mean()),
            "optional_active_rows": int(guard_active.sum()),
            "optional_activation_coverage": float(guard_active.mean()),
            "optional_activation_by_source_json": json.dumps(
                activation_by_source, sort_keys=True, separators=(",", ":")
            ),
            "safe_passing_cells": safe_passing_cells,
            "safe_threshold": None if safe is None else float(safe["threshold"]),
            "broad_threshold": None if broad is None else float(broad["threshold"]),
            "safe_worst_cell_severe_recall": None
            if safe is None
            else float(safe["worst_cell_severe_recall"]),
            "safe_worst_cell_precision_lower": None
            if safe is None
            else float(safe["worst_cell_precision_lower"]),
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
            safe_passing_cells,
            safe_recall,
            safe_precision,
            worst_source_auc,
            -record["term_count"],
        )
        identity = str(spec["candidate_key"])
        if best is None or rank > best["rank"] or (
            rank == best["rank"] and identity < best["identity"]
        ):
            formula_terms = [
                {**old_by_id[term_id], "weight": float(weight)}
                for term_id, weight in zip(
                    base_term_ids, base_weights, strict=True
                )
            ]
            optional_formula = (
                None
                if optional_term_id is None
                else {
                    **optional_by_id[optional_term_id],
                    "weight": optional_weight,
                }
            )
            best = {
                "rank": rank,
                "identity": identity,
                "record": record,
                "formula": {
                    "kind": "base_nonnegative_hinge_sum_plus_one_optional_dobvr_guard",
                    "base_missing_policy": "ABSTAIN",
                    "optional_missing_policy": "OPTIONAL_GUARD_OFF_KEEP_BASE",
                    "base_terms": formula_terms,
                    "optional_term": optional_formula,
                    "safe_threshold": None
                    if safe is None
                    else float(safe["threshold"]),
                    "broad_threshold": None
                    if broad is None
                    else float(broad["threshold"]),
                },
                "safe": safe,
                "safe_diagnostic": safe_diagnostic,
                "broad": broad,
                "source_diagnostics": source_diagnostics,
            }
    if best is None:
        raise RuntimeError("NEXT103 search produced no candidate")
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


def run_dobvr_optional_guard_search(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    next98_dir: Path,
    next98b_dir: Path,
    next102_dir: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the preregistered discovery-only optional-guard search."""

    scigen_feature_root = Path(scigen_feature_dir).resolve()
    scigen_endpoint_root = Path(scigen_discovery_endpoint_dir).resolve()
    wyformer_feature_root = Path(wyformer_feature_dir).resolve()
    wyformer_endpoint_root = Path(wyformer_discovery_endpoint_dir).resolve()
    next98_root = Path(next98_dir).resolve()
    next98b_root = Path(next98b_dir).resolve()
    next102_root = Path(next102_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "scigen_features": scigen_feature_root
        / SCIGEN_FEATURE_NAMES["discovery"],
        "scigen_endpoint": scigen_endpoint_root / SCIGEN_ENDPOINT_NAME,
        "wyformer_features": wyformer_feature_root
        / WYFORMER_FEATURE_NAMES["discovery"],
        "wyformer_endpoint": wyformer_endpoint_root / WYFORMER_ENDPOINT_NAME,
        "next98_manifest": next98_root / NEXT98_MANIFEST_NAME,
        "next98_term_catalogue": next98_root / NEXT98_CATALOGUE_NAME,
        "next98b_manifest": next98b_root / NEXT98B_MANIFEST_NAME,
        "next98b_search_records": next98b_root / NEXT98B_SEARCH_NAME,
        "next102_manifest": next102_root / NEXT102_MANIFEST_NAME,
        "next102_scigen_features": next102_root / NEXT102_FEATURE_NAMES["scigen"],
        "next102_wyformer_features": next102_root
        / NEXT102_FEATURE_NAMES["wyformer"],
        "design": Path(design_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT103 discovery input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT103 formal input identity differs")

    next98_manifest = _read_json(paths["next98_manifest"])
    next98b_manifest = _read_json(paths["next98b_manifest"])
    next102_manifest = _read_json(paths["next102_manifest"])
    if (
        next98_manifest.get("protocol") != NEXT98_PROTOCOL
        or next98_manifest.get("opened_validation_outputs_used") is not False
        or next98_manifest.get("scigen_replication_endpoint_opened") is not False
        or next98_manifest.get("wyformer_replication_endpoint_opened") is not False
        or next98b_manifest.get("protocol") != NEXT98B_PROTOCOL
        or next98b_manifest.get("passes_all_cross_source_discovery_gates") is not False
        or next98b_manifest.get("opened_validation_outputs_used") is not False
        or next98b_manifest.get("scigen_replication_endpoint_opened") is not False
        or next98b_manifest.get("wyformer_replication_endpoint_opened") is not False
    ):
        raise ValueError("NEXT103 prior search provenance differs")
    next102_outputs = next102_manifest.get("outputs_sha256")
    if (
        next102_manifest.get("protocol") != NEXT102_PROTOCOL
        or next102_manifest.get("partitions_read") != ["discovery"]
        or next102_manifest.get("labels_opened") is not False
        or next102_manifest.get("endpoint_payloads_opened") is not False
        or next102_manifest.get("validation_geometry_opened") is not False
        or next102_manifest.get("replication_geometry_opened") is not False
        or next102_manifest.get("dft_values_used_by_features") is not False
        or not isinstance(next102_outputs, Mapping)
        or next102_outputs.get(NEXT102_FEATURE_NAMES["scigen"])
        != input_hashes["next102_scigen_features"]
        or next102_outputs.get(NEXT102_FEATURE_NAMES["wyformer"])
        != input_hashes["next102_wyformer_features"]
    ):
        raise ValueError("NEXT103 NEXT102 feature provenance differs")

    old_tables = {
        "scigen": pd.read_parquet(paths["scigen_features"]),
        "wyformer": pd.read_parquet(paths["wyformer_features"]),
    }
    new_tables = {
        "scigen": pd.read_parquet(paths["next102_scigen_features"]),
        "wyformer": pd.read_parquet(paths["next102_wyformer_features"]),
    }
    feature_tables: dict[str, pd.DataFrame] = {}
    for source in ("scigen", "wyformer"):
        old = old_tables[source]
        new = new_tables[source]
        if (
            old["material_id"].astype(str).duplicated().any()
            or new["material_id"].astype(str).duplicated().any()
            or set(NEXT102_FEATURE_COLUMNS) - set(new.columns)
        ):
            raise ValueError(f"NEXT103 {source} feature identity differs")
        merged = old.merge(
            new.loc[:, ["material_id", *NEXT102_FEATURE_COLUMNS]],
            on="material_id",
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(old) or len(merged) != len(new):
            raise ValueError(f"NEXT103 {source} feature row accounting differs")
        merged = merged.copy()
        merged["source_dataset"] = source
        if source == "scigen":
            merged["crystal_system"] = merged["lattice_class"].astype(str)
        merged["material_id"] = source + ":" + merged["material_id"].astype(str)
        feature_tables[source] = merged
    feature_combined = pd.concat(
        [feature_tables["scigen"], feature_tables["wyformer"]],
        ignore_index=True,
        sort=False,
    )
    optional_terms, excluded_optional_terms = calibrate_optional_terms(
        feature_combined,
        templates=OPTIONAL_TERM_TEMPLATES,
        min_source_coverage=MIN_SOURCE_COVERAGE,
        min_unique_values=8,
    )

    # Endpoint tables are opened only after the label-free optional catalogue is frozen.
    scigen_endpoints = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_endpoints = pd.read_parquet(paths["wyformer_endpoint"])
    if (
        scigen_endpoints["material_id"].astype(str).duplicated().any()
        or wyformer_endpoints["material_id"].astype(str).duplicated().any()
    ):
        raise ValueError("NEXT103 discovery endpoint identities are duplicated")
    scigen_endpoint_frame = pd.DataFrame(
        {
            "material_id": "scigen:"
            + scigen_endpoints["material_id"].astype(str),
            "_endpoint_numeric": pd.to_numeric(
                scigen_endpoints["distortion_ratio"], errors="coerce"
            ),
        }
    )
    wyformer_endpoint_frame = pd.DataFrame(
        {
            "material_id": "wyformer:"
            + wyformer_endpoints["material_id"].astype(str),
            "_endpoint_numeric": _endpoint_numeric(
                wyformer_endpoints["endpoint_stratum"]
            ),
        }
    )
    endpoint_frame = pd.concat(
        [scigen_endpoint_frame, wyformer_endpoint_frame], ignore_index=True
    )
    combined = feature_combined.merge(
        endpoint_frame,
        on="material_id",
        how="inner",
        validate="one_to_one",
    )
    if len(combined) != len(feature_combined) or len(combined) != len(endpoint_frame):
        raise ValueError("NEXT103 endpoint row accounting differs")
    endpoint = pd.to_numeric(
        combined.pop("_endpoint_numeric"), errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(endpoint).all():
        raise ValueError("NEXT103 endpoint conversion differs")

    old_catalogue = _read_json(paths["next98_term_catalogue"])
    old_terms = old_catalogue.get("eligible_terms")
    if not isinstance(old_terms, list):
        raise ValueError("NEXT103 old term catalogue differs")
    old_term_ids = {str(term["term_id"]) for term in old_terms}
    base_records = pd.read_parquet(paths["next98b_search_records"])
    passing_bases = int(base_records["passes_source_auc_gates"].eq(True).sum())
    if require_formal_inputs and passing_bases != EXPECTED_AUC_PASSING_BASES:
        raise ValueError("NEXT103 AUC-passing base count differs")
    specs = build_optional_guard_candidate_specs(
        base_records=base_records,
        old_term_ids=old_term_ids,
        optional_terms=optional_terms,
    )
    started = time.perf_counter()
    result = search_optional_guard_laws(
        features=combined,
        endpoint=endpoint,
        old_terms=old_terms,
        optional_terms=optional_terms,
        candidate_specs=specs,
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
            "calibration_stage": "label_free_before_endpoint_join",
            "min_source_coverage": MIN_SOURCE_COVERAGE,
            "optional_weight_grid": list(OPTIONAL_WEIGHT_GRID),
            "expected_auc_passing_bases": EXPECTED_AUC_PASSING_BASES,
            "observed_auc_passing_bases": passing_bases,
            "templates": list(OPTIONAL_TERM_TEMPLATES),
            "eligible_optional_terms": optional_terms,
            "excluded_optional_terms": excluded_optional_terms,
            "candidate_count": len(specs),
            "candidate_grammar": "each AUC-passing NEXT98b base plus zero or one optional DOBVR guard",
            "optional_missing_policy": "OPTIONAL_GUARD_OFF_KEEP_BASE",
        }
        evaluation = {
            "protocol": PROTOCOL,
            "evaluation_mode": "cross_source_discovery_only_optional_dobvr_guard",
            "rows": {
                "scigen": int(len(feature_tables["scigen"])),
                "wyformer": int(len(feature_tables["wyformer"])),
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
            "selected_safe_diagnostic": selected["safe_diagnostic"],
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
            "candidate_count": int(result["candidate_count"]),
            "eligible_optional_term_count": len(optional_terms),
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
                "src/next103_dobvr_optional_guard_search.py": source_hash
            },
            "outputs_sha256": {
                path.name: _sha256_file(path) for path in output_paths
            },
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT103 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT103 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "OPTIONAL_WEIGHT_GRID",
    "OPTIONAL_TERM_TEMPLATES",
    "PROTOCOL",
    "build_optional_guard_candidate_specs",
    "calibrate_optional_terms",
    "compose_optional_guard_score",
    "run_dobvr_optional_guard_search",
    "select_safe_and_diagnostic_once",
    "search_optional_guard_laws",
]
