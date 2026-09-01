"""Finite sparse discovery search on the WyFormer discovery endpoint only."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from itertools import combinations, product
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.next23_relaxation_rule import wilson_lower_bound
from src.next87_scigen_sparse_law_search import (
    WEIGHT_GRID,
    _pauling_baseline,
    _pooled_auc,
    _term_risk,
    _transformed_column,
    _wilson_lower_array,
    assign_group_folds,
    auc_diagnostics,
    decision_metrics,
    select_threshold,
)
from src.next93_wyformer_source_lockbox import _sha256_file, _write_json
from src.next93b_wyformer_blind_lockbox import (
    ENDPOINT_NAME,
    MANIFEST_NAME as ENDPOINT_MANIFEST_NAME,
    PROTOCOL as ENDPOINT_PROTOCOL,
)
from src.next94_wyformer_label_free_features import (
    CATALOGUE_NAME as FEATURE_CATALOGUE_NAME,
    FEATURE_NAMES,
    MANIFEST_NAME as FEATURE_MANIFEST_NAME,
    PROTOCOL as FEATURE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next95-wyformer-discovery-sparse-law-search-v1"
MANIFEST_NAME = "MANIFEST.json"
TERM_CATALOGUE_NAME = "NEXT95_WYFORMER_TERM_CATALOGUE.json"
EVALUATION_NAME = "NEXT95_WYFORMER_DISCOVERY_EVALUATION.json"
SEARCH_RECORD_NAME = "next95_complete_candidate_search.parquet"
FORMULA_KIND = "nonnegative_sum_of_at_most_three_one_sided_robust_hinges"
MISSING_POLICY = "ABSTAIN"
GROUP_FOLDS = 5
PAIR_SHORTLIST = 16
TRIPLE_SHORTLIST = 12
MAX_TERMS_PER_GROUP = 3
MIN_TERM_COVERAGE = 0.90
DEFAULT_GATES = {
    "coverage_lower": 0.90,
    "protected_recall_lower": 0.90,
    "severe_rejection_precision_lower": 0.80,
    "savings_lower": 0.02,
    "pooled_extreme_auc": 0.75,
    "macro_lattice_auc": 0.60,
    "worst_lattice_auc": 0.55,
    "evaluable_lattices": 5,
}
EXPECTED_INPUT_SHA256 = {
    "feature_manifest": "fb66f7c5caade419a46b9a3fa6fef1bc5b3afa3eebeb95a4bc53baddabc0f659",
    "feature_discovery": "c515baec0fccef5bc03c7672f1d4e1aca278f5ed4d7b6f1bf7f66c734e2b87f7",
    "feature_catalogue": "2fcec0f8564294ec1267546532c974a6e059e9f48b3b30bf95dc3dd58ca80991",
    "template_term_catalogue": "e8f9fe532c15673c0a74737632b0145d43f6494cb1ea7e94e7380198fd4e4dee",
    "discovery_endpoint_manifest": "3cf3a196ab497851131d5d1604f272d15121c19a943eeb3103a268e7e8b332f5",
    "discovery_endpoint": "f39836e62a1da03ed823479e87d6f75fc0d01da60a8c0a2faa696638cc2fb9d7",
    "design": "db9e05470132d57002b62b408b4c0ed3ee39201a61fe6586610b70f1123cbc77",
}


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _endpoint_numeric(strata: pd.Series) -> np.ndarray:
    mapping = {"protected": 1.0, "middle": 1.5, "severe": 2.0}
    values = strata.astype(str).map(mapping)
    if values.isna().any():
        raise ValueError("WyFormer endpoint stratum differs")
    return values.to_numpy(float)


def recalibrate_terms(
    features: pd.DataFrame,
    template_terms: Sequence[Mapping[str, object]],
    *,
    min_coverage: float = MIN_TERM_COVERAGE,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Refit label-free robust transforms on discovery x0 features only."""

    if not 0 < float(min_coverage) <= 1:
        raise ValueError("minimum term coverage differs")
    eligible: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for template in sorted(template_terms, key=lambda item: str(item.get("term_id"))):
        term_id = template.get("term_id")
        feature = template.get("feature")
        transform = template.get("transform")
        direction = template.get("direction")
        group = template.get("group")
        reason: str | None = None
        if (
            not isinstance(term_id, str)
            or not isinstance(feature, str)
            or feature not in features
            or transform not in {"log1p_nonnegative", "asinh"}
            or direction not in {-1, 1}
            or not isinstance(group, str)
        ):
            reason = "template_schema_or_feature_missing"
            raw = np.array([], dtype=float)
            transformed = raw
        else:
            raw = pd.to_numeric(features[feature], errors="coerce").to_numpy(float)
            transformed = _transformed_column(raw, str(transform))
        finite = np.isfinite(transformed)
        coverage = float(finite.mean()) if len(finite) else 0.0
        values = transformed[finite]
        if reason is None and coverage < float(min_coverage):
            reason = "coverage_below_gate"
        if reason is None and len(np.unique(values)) < 8:
            reason = "fewer_than_eight_unique_values"
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
                    "coverage": coverage,
                    "reason": reason,
                }
            )
            continue
        eligible.append(
            {
                "term_id": term_id,
                "feature": feature,
                "direction": int(direction),
                "transform": str(transform),
                "group": str(group),
                "center": float(center),
                "scale": float(scale),
                "coverage": coverage,
                "finite_rows": int(finite.sum()),
                "unique_transformed_values": int(len(np.unique(values))),
                "transformed_q10": float(q10),
                "transformed_q90": float(q90),
            }
        )
    return eligible, excluded


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


def _fixed_fold_metrics(
    *,
    score: np.ndarray,
    supported: np.ndarray,
    endpoint: np.ndarray,
    folds: np.ndarray,
    threshold: float,
    gates: Mapping[str, float],
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    passing = 0
    for held_out in range(GROUP_FOLDS):
        mask = folds == held_out
        metrics = decision_metrics(
            supported=supported[mask],
            reject=supported[mask] & (score[mask] >= threshold),
            distortion_ratio=endpoint[mask],
        )
        passed = _operating_pass(metrics, gates)
        passing += int(passed)
        records.append(
            {
                "held_out_fold": held_out,
                "rows": int(mask.sum()),
                "passes_fixed_formula_and_threshold_gates": passed,
                "metrics": metrics,
            }
        )
    return {
        "passing_folds": passing,
        "passes_all_folds": passing == GROUP_FOLDS,
        "folds": records,
    }


def evaluate_fixed_threshold_folds(
    *,
    score: object,
    supported: object,
    endpoint: object,
    reduced_formula: object,
    threshold: float,
    gates: Mapping[str, float] = DEFAULT_GATES,
) -> dict[str, object]:
    scores = np.asarray(score, dtype=float)
    support = np.asarray(supported, dtype=bool)
    endpoint_array = np.asarray(endpoint, dtype=float)
    groups = np.asarray(reduced_formula, dtype=object)
    if (
        scores.ndim != 1
        or support.shape != scores.shape
        or endpoint_array.shape != scores.shape
        or groups.shape != scores.shape
        or not math.isfinite(float(threshold))
    ):
        raise ValueError("fixed-fold arrays differ")
    folds = assign_group_folds(groups)
    return _fixed_fold_metrics(
        score=scores,
        supported=support,
        endpoint=endpoint_array,
        folds=folds,
        threshold=float(threshold),
        gates=gates,
    )


def _select_threshold_beating_pauling(
    *,
    score: np.ndarray,
    supported: np.ndarray,
    endpoint: np.ndarray,
    gates: Mapping[str, float],
    pauling: Mapping[str, object],
) -> dict[str, object] | None:
    local_support = supported & np.isfinite(score)
    rows = len(score)
    n_supported = int(local_support.sum())
    if (
        not n_supported
        or wilson_lower_bound(n_supported, rows) < float(gates["coverage_lower"])
    ):
        return None
    order = np.argsort(-score[local_support], kind="stable")
    sorted_scores = score[local_support][order]
    sorted_endpoint = endpoint[local_support][order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(sorted_scores) - 1]
    thresholds = sorted_scores[ends]
    rejected = ends + 1
    protected = sorted_endpoint <= 1.0
    severe = sorted_endpoint >= 2.0
    rejected_protected = np.cumsum(protected, dtype=int)[ends]
    rejected_severe = np.cumsum(severe, dtype=int)[ends]
    rejected_extremes = rejected_protected + rejected_severe
    n_protected = int((endpoint <= 1.0).sum())
    recall_lower = _wilson_lower_array(
        n_protected - rejected_protected,
        np.full(len(ends), n_protected, dtype=int),
    )
    precision_lower = _wilson_lower_array(rejected_severe, rejected_extremes)
    savings_lower = _wilson_lower_array(rejected, np.full(len(ends), rows, dtype=int))
    feasible = (
        (recall_lower >= float(gates["protected_recall_lower"]))
        & (precision_lower >= float(gates["severe_rejection_precision_lower"]))
        & (savings_lower >= float(gates["savings_lower"]))
        & (rejected_severe > int(pauling["severe_rejected"]))
        & (
            precision_lower
            > float(pauling["severe_rejection_precision_lower"])
        )
    )
    indices = np.flatnonzero(feasible)
    if not len(indices):
        return None
    best = max(
        indices.tolist(),
        key=lambda index: (
            float(precision_lower[index]),
            int(rejected_severe[index]),
            float(savings_lower[index]),
            float(thresholds[index]),
        ),
    )
    threshold = float(thresholds[best])
    reject = local_support & (score >= threshold)
    return {
        "threshold": threshold,
        "metrics": decision_metrics(
            supported=local_support, reject=reject, distortion_ratio=endpoint
        ),
    }


def _formula(
    terms: Sequence[Mapping[str, object]],
    indices: tuple[int, ...],
    weights: tuple[float, ...],
    threshold: float,
) -> dict[str, object]:
    return {
        "kind": FORMULA_KIND,
        "missing_policy": MISSING_POLICY,
        "threshold": float(threshold),
        "terms": [
            {**dict(terms[index]), "weight": float(weight)}
            for index, weight in zip(indices, weights)
        ],
    }


def search_wyformer_sparse_law(
    *,
    features: pd.DataFrame,
    endpoint: np.ndarray,
    eligible_terms: Sequence[Mapping[str, object]],
    gates: Mapping[str, float] = DEFAULT_GATES,
    pair_shortlist: int = PAIR_SHORTLIST,
    triple_shortlist: int = TRIPLE_SHORTLIST,
) -> dict[str, object]:
    required = {
        "material_id",
        "reduced_formula",
        "crystal_system",
        "pauling_p2_p5_decision",
    }
    endpoint = np.asarray(endpoint, dtype=float)
    if (
        required - set(features.columns)
        or len(features) != len(endpoint)
        or not np.isfinite(endpoint).all()
        or not (endpoint <= 1).any()
        or not (endpoint >= 2).any()
    ):
        raise ValueError("NEXT95 discovery arrays differ")
    terms = sorted((dict(term) for term in eligible_terms), key=lambda term: str(term["term_id"]))
    if not terms:
        raise ValueError("NEXT95 has no eligible terms")
    risks, supports = zip(*(_term_risk(features, term) for term in terms))
    risk_matrix = np.column_stack(risks)
    support_matrix = np.column_stack(supports)
    single_order = sorted(
        range(len(terms)),
        key=lambda index: (
            -(
                _pooled_auc(risk_matrix[:, index], support_matrix[:, index], endpoint)
                or -1.0
            ),
            str(terms[index]["term_id"]),
        ),
    )
    shortlist: list[int] = []
    group_counts: dict[str, int] = {}
    for index in single_order:
        group = str(terms[index]["group"])
        if group_counts.get(group, 0) >= MAX_TERMS_PER_GROUP:
            continue
        shortlist.append(index)
        group_counts[group] = group_counts.get(group, 0) + 1
        if len(shortlist) >= min(pair_shortlist, len(terms)):
            break
    pair_indices = sorted(shortlist, key=lambda index: str(terms[index]["term_id"]))
    triple_indices = sorted(
        shortlist[: min(triple_shortlist, len(shortlist))],
        key=lambda index: str(terms[index]["term_id"]),
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

    folds = assign_group_folds(features["reduced_formula"].astype(str).to_numpy())
    pauling = _pauling_baseline(features, endpoint)
    records: list[dict[str, object]] = []
    best_payload: dict[str, object] | None = None
    for stage, indices, weights in specs:
        score = np.sum(
            risk_matrix[:, indices] * np.asarray(weights, dtype=float)[None, :], axis=1
        )
        supported = np.all(support_matrix[:, indices], axis=1)
        score = np.asarray(score, dtype=float)
        score[~supported] = np.nan
        selected = _select_threshold_beating_pauling(
            score=score,
            supported=supported,
            endpoint=endpoint,
            gates=gates,
            pauling=pauling,
        )
        if selected is None:
            selected = select_threshold(
                score=score,
                supported=supported,
                distortion_ratio=endpoint,
                gates=gates,
            )
        pooled_auc = _pooled_auc(score, supported, endpoint)
        if selected is None:
            threshold = None
            metrics = decision_metrics(
                supported=supported,
                reject=np.zeros(len(endpoint), dtype=bool),
                distortion_ratio=endpoint,
            )
            fold_result = {"passing_folds": 0, "passes_all_folds": False, "folds": []}
        else:
            threshold = float(selected["threshold"])
            metrics = dict(selected["metrics"])
            fold_result = _fixed_fold_metrics(
                score=score,
                supported=supported,
                endpoint=endpoint,
                folds=folds,
                threshold=threshold,
                gates=gates,
            )
        beats_pauling = bool(
            int(metrics["severe_rejected"]) > int(pauling["severe_rejected"])
            and float(metrics["severe_rejection_precision_lower"])
            > float(pauling["severe_rejection_precision_lower"])
        )
        operating = threshold is not None and _operating_pass(metrics, gates)
        diagnostics: dict[str, object] | None = None
        auc_pass = False
        if operating and fold_result["passes_all_folds"] and beats_pauling:
            diagnostics = auc_diagnostics(
                score=score,
                supported=supported,
                distortion_ratio=endpoint,
                lattice_class=features["crystal_system"].astype(str).to_numpy(),
            )
            auc_pass = bool(
                diagnostics["pooled_extreme_auc"] is not None
                and float(diagnostics["pooled_extreme_auc"]) >= float(gates["pooled_extreme_auc"])
                and diagnostics["macro_lattice_auc"] is not None
                and float(diagnostics["macro_lattice_auc"]) >= float(gates["macro_lattice_auc"])
                and diagnostics["worst_lattice_auc"] is not None
                and float(diagnostics["worst_lattice_auc"]) >= float(gates["worst_lattice_auc"])
                and int(diagnostics["evaluable_lattices"]) >= int(gates["evaluable_lattices"])
            )
        passed = bool(operating and fold_result["passes_all_folds"] and beats_pauling and auc_pass)
        rank = (
            int(passed),
            int(fold_result["passing_folds"]),
            int(operating),
            int(beats_pauling),
            float(metrics["severe_rejection_precision_lower"]),
            int(metrics["severe_rejected"]),
            float(pooled_auc) if pooled_auc is not None else -1.0,
            float(metrics["savings_lower"]),
            -len(indices),
        )
        formula_key = "+".join(
            f"{weight:g}*{terms[index]['term_id']}" for index, weight in zip(indices, weights)
        )
        record = {
            "stage": stage,
            "term_ids": [str(terms[index]["term_id"]) for index in indices],
            "weights": list(weights),
            "formula_key": formula_key,
            "threshold": threshold,
            "coverage_lower": float(metrics["coverage_lower"]),
            "protected_recall_lower": float(metrics["protected_recall_lower"]),
            "severe_rejection_precision_lower": float(
                metrics["severe_rejection_precision_lower"]
            ),
            "savings_lower": float(metrics["savings_lower"]),
            "severe_rejected": int(metrics["severe_rejected"]),
            "pooled_extreme_auc": pooled_auc,
            "passing_fixed_folds": int(fold_result["passing_folds"]),
            "passes_operating_gates": bool(operating),
            "beats_pauling": beats_pauling,
            "passes_all_discovery_gates": passed,
        }
        records.append(record)
        identity = tuple(str(item) for item in (formula_key, threshold))
        if best_payload is None or rank > best_payload["rank"] or (
            rank == best_payload["rank"] and identity < best_payload["identity"]
        ):
            if diagnostics is None and threshold is not None:
                diagnostics = auc_diagnostics(
                    score=score,
                    supported=supported,
                    distortion_ratio=endpoint,
                    lattice_class=features["crystal_system"].astype(str).to_numpy(),
                )
            best_payload = {
                "rank": rank,
                "identity": identity,
                "record": record,
                "formula": _formula(terms, indices, weights, threshold)
                if threshold is not None
                else None,
                "metrics": metrics,
                "fold_result": fold_result,
                "diagnostics": diagnostics,
                "score": score,
                "supported": supported,
            }
    if best_payload is None:
        raise RuntimeError("NEXT95 search produced no candidate")
    return {
        "eligible_terms": terms,
        "candidate_records": records,
        "candidate_count": len(records),
        "pauling": pauling,
        "selected": best_payload,
    }


def _publish_directory(staging: Path, target: Path) -> None:
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    os.replace(staging, target)


def run_wyformer_sparse_search(
    *,
    feature_dir: Path,
    template_term_catalogue_path: Path,
    discovery_endpoint_dir: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Open only the discovery endpoint and run the finite NEXT95 search."""

    feature_root = Path(feature_dir).resolve()
    endpoint_root = Path(discovery_endpoint_dir).resolve()
    paths = {
        "feature_manifest": feature_root / FEATURE_MANIFEST_NAME,
        "feature_discovery": feature_root / FEATURE_NAMES["discovery"],
        "feature_catalogue": feature_root / FEATURE_CATALOGUE_NAME,
        "template_term_catalogue": Path(template_term_catalogue_path).resolve(),
        "discovery_endpoint_manifest": endpoint_root / ENDPOINT_MANIFEST_NAME,
        "discovery_endpoint": endpoint_root / ENDPOINT_NAME,
        "design": Path(design_path).resolve(),
    }
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT95 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT95 formal input identity differs")

    feature_manifest = _read_json(paths["feature_manifest"])
    endpoint_manifest = _read_json(paths["discovery_endpoint_manifest"])
    if (
        feature_manifest.get("protocol") != FEATURE_PROTOCOL
        or feature_manifest.get("labels_opened") is not False
        or feature_manifest.get("endpoint_payloads_opened") is not False
        or feature_manifest.get("dft_values_used_by_features") is not False
        or feature_manifest.get("learned_energy_force_stress_proxy_used") is not False
    ):
        raise ValueError("NEXT94 feature provenance differs")
    if (
        endpoint_manifest.get("protocol") != ENDPOINT_PROTOCOL
        or endpoint_manifest.get("partition_role") != "discovery"
        or endpoint_manifest.get("endpoint_payload_opened") is not False
        or endpoint_manifest.get("endpoint_sha256") != input_hashes["discovery_endpoint"]
    ):
        raise ValueError("NEXT93b discovery endpoint provenance differs")

    features = pd.read_parquet(paths["feature_discovery"])
    endpoint_frame = pd.read_parquet(paths["discovery_endpoint"])
    if endpoint_frame["material_id"].duplicated().any():
        raise ValueError("NEXT95 discovery endpoint ids are duplicated")
    merged = features.merge(
        endpoint_frame[["material_id", "endpoint_stratum"]],
        on="material_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(features) or len(merged) != len(endpoint_frame):
        raise ValueError("NEXT95 discovery row accounting differs")
    endpoint = _endpoint_numeric(merged["endpoint_stratum"])
    template_catalogue = _read_json(paths["template_term_catalogue"])
    template_terms = template_catalogue.get("eligible_terms")
    if not isinstance(template_terms, list):
        raise ValueError("NEXT86 label-free template terms differ")
    terms, excluded = recalibrate_terms(merged, template_terms)
    started = time.perf_counter()
    result = search_wyformer_sparse_law(
        features=merged,
        endpoint=endpoint,
        eligible_terms=terms,
    )
    selected = result["selected"]
    selected_record = dict(selected["record"])
    passed = bool(selected_record["passes_all_discovery_gates"])
    evaluation = {
        "protocol": PROTOCOL,
        "endpoint_definition": {
            "protected": "DFT succeeded and corrected e_hull <= 0.10 eV/atom",
            "middle": "DFT succeeded and 0.10 < corrected e_hull < 0.50 eV/atom",
            "severe": "DFT failed or corrected e_hull >= 0.50 eV/atom",
        },
        "gates": DEFAULT_GATES,
        "candidate_count": int(result["candidate_count"]),
        "eligible_term_count": len(terms),
        "excluded_term_count": len(excluded),
        "selected_candidate": selected_record,
        "selected_formula": selected["formula"],
        "selected_metrics": selected["metrics"],
        "selected_fixed_fold_diagnostics": selected["fold_result"],
        "selected_auc_diagnostics": selected["diagnostics"],
        "pauling_baseline": result["pauling"],
        "passes_all_discovery_gates": passed,
        "validation_authorized": passed,
        "replication_authorized": False,
        "scientific_improvement_claim": False,
    }
    term_catalogue = {
        "protocol": PROTOCOL,
        "template_catalogue_sha256": input_hashes["template_term_catalogue"],
        "statistics_partition": "discovery_x0_features_only",
        "labels_used_for_transforms": False,
        "eligible_terms": terms,
        "excluded_terms": excluded,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256_file(source_path)
    try:
        _write_json(staging / TERM_CATALOGUE_NAME, term_catalogue)
        _write_json(staging / EVALUATION_NAME, evaluation)
        pd.DataFrame(result["candidate_records"]).to_parquet(
            staging / SEARCH_RECORD_NAME, index=False
        )
        output_paths = [
            staging / TERM_CATALOGUE_NAME,
            staging / EVALUATION_NAME,
            staging / SEARCH_RECORD_NAME,
        ]
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "elapsed_seconds": time.perf_counter() - started,
            "discovery_endpoint_opened": True,
            "validation_endpoint_opened": False,
            "replication_endpoint_opened": False,
            "passes_discovery_gates": passed,
            "prediction_partitions_frozen": [],
            "dft_calculation_executed": False,
            "dft_values_used_by_formula_at_execution": False,
            "learned_energy_force_stress_proxy_used": False,
            "physical_relaxation_executed": False,
            "scientific_improvement_claim": False,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": {
                "src/next95_wyformer_sparse_law_search.py": source_hash
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT95 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT95 source changed before publication")
        _publish_directory(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "DEFAULT_GATES",
    "EVALUATION_NAME",
    "MANIFEST_NAME",
    "PROTOCOL",
    "SEARCH_RECORD_NAME",
    "TERM_CATALOGUE_NAME",
    "evaluate_fixed_threshold_folds",
    "recalibrate_terms",
    "run_wyformer_sparse_search",
    "search_wyformer_sparse_law",
]
