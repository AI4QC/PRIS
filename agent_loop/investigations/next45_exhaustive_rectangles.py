#!/usr/bin/env python3
"""Exhaustive two-condition analytic rectangles over NEXT43/NEXT44 features."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from itertools import combinations
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from src.next19_feature_build import _publish_directory_no_replace, _sha256, _strict_json
from src.next23_evaluate import _continuous_diagnostics, _decision_metrics
from src.next23_relaxation_rule import (
    ENDPOINT_COLUMN,
    PRIMARY_GATES,
    PROTECTED_MAX,
    SEVERE_MIN,
    SUBSTANTIAL_MIN,
    wilson_lower_bound,
)
from src.next43_finite_law_search import (
    _baseline_metrics,
    _validate_inputs as _validate_next43_inputs,
    apply_formula,
    deterministic_split,
)
from src.next44_rich_law_search import (
    ALL_FEATURE_NAMES,
    PROTOCOL as NEXT44_SEARCH_PROTOCOL,
    SEARCH_NAME as NEXT44_SEARCH_NAME,
    _validate_rich_table,
    combine_feature_tables,
)
from src.next43_analytic_feature_bank import CANDIDATE_FEATURE_NAMES as BASE_FEATURE_NAMES
from src.next44_rich_analytic_features import CANDIDATE_FEATURE_NAMES as RICH_FEATURE_NAMES


PROTOCOL = "2026-08-03-next45-exhaustive-analytic-rectangles-v1"
FORMULA_NAME = "NEXT45_DEVELOPMENT_RECTANGLE.json"
SEARCH_NAME = "NEXT45_EXHAUSTIVE_RECTANGLE_SEARCH.json"
PREDICTION_NAME = "next45_development_evaluation.parquet"
MANIFEST_NAME = "MANIFEST.json"
TAIL_FRACTIONS = (0.30, 0.40, 0.50, 0.60, 0.70)


def bitset_decision_metrics(
    *, supported: np.ndarray, reject: np.ndarray, endpoint: np.ndarray
) -> dict[str, object]:
    """Reference-compatible public wrapper used by bitset contract tests."""

    return _decision_metrics(supported=supported, reject=reject, endpoint=endpoint)


def _mask_bits(mask: np.ndarray) -> int:
    packed = np.packbits(np.asarray(mask, dtype=np.uint8), bitorder="little")
    return int.from_bytes(packed.tobytes(), "little")


def _metrics_from_bits(
    *,
    supported: int,
    reject: int,
    rows: int,
    protected: int,
    changed: int,
    substantial: int,
    severe: int,
) -> dict[str, object]:
    reject &= supported
    n_supported = supported.bit_count()
    n_rejected = reject.bit_count()
    n_protected = protected.bit_count()
    protected_kept = n_protected - (protected & reject).bit_count()
    changed_rejected = (changed & reject).bit_count()
    substantial_total = substantial.bit_count()
    severe_total = severe.bit_count()
    metrics: dict[str, object] = {
        "rows": rows,
        "supported": n_supported,
        "rejected": n_rejected,
        "protected": n_protected,
        "protected_kept": protected_kept,
        "changed_rejected": changed_rejected,
        "coverage": n_supported / rows if rows else 0.0,
        "coverage_lower": wilson_lower_bound(n_supported, rows),
        "protected_recall": protected_kept / n_protected if n_protected else 0.0,
        "protected_recall_lower": wilson_lower_bound(protected_kept, n_protected),
        "rejection_precision": changed_rejected / n_rejected if n_rejected else 0.0,
        "rejection_precision_lower": wilson_lower_bound(changed_rejected, n_rejected),
        "savings": n_rejected / rows if rows else 0.0,
        "savings_lower": wilson_lower_bound(n_rejected, rows),
        "substantial_total": substantial_total,
        "substantial_recall": (
            (substantial & reject).bit_count() / substantial_total
            if substantial_total
            else 0.0
        ),
        "severe_total": severe_total,
        "severe_recall": (
            (severe & reject).bit_count() / severe_total if severe_total else 0.0
        ),
    }
    metrics["passes_primary_gates"] = all(
        float(metrics[name]) >= cutoff for name, cutoff in PRIMARY_GATES.items()
    )
    return metrics


def _rank(metrics: Mapping[str, object]) -> tuple[float, ...]:
    ratios = [float(metrics[name]) / float(cutoff) for name, cutoff in PRIMARY_GATES.items()]
    return (
        1.0 if bool(metrics["passes_primary_gates"]) else 0.0,
        min(ratios),
        float(metrics["protected_recall_lower"]),
        float(metrics["rejection_precision_lower"]),
        float(metrics["savings_lower"]),
        float(metrics["coverage_lower"]),
        float(metrics["severe_recall"]),
        float(metrics["substantial_recall"]),
    )


def _quantile(values: np.ndarray, q: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, q, method="inverted_cdf"))


def search_exhaustive_rectangles(
    *,
    features: pd.DataFrame,
    material_ids: Sequence[str],
    endpoint: Sequence[float],
    split: Sequence[str],
    candidate_features: Sequence[str] = ALL_FEATURE_NAMES,
    tail_fractions: Sequence[float] = TAIL_FRACTIONS,
) -> dict[str, object]:
    """Select an exhaustive rectangle on discovery and open validation once."""

    material_ids = np.asarray(material_ids, dtype=str)
    endpoint = np.asarray(endpoint, dtype=float)
    split = np.asarray(split, dtype=object)
    tails = tuple(float(value) for value in tail_fractions)
    if (
        len(features) != len(material_ids)
        or endpoint.shape != (len(features),)
        or split.shape != (len(features),)
        or set(np.unique(split)) != {"discovery", "validation"}
        or not np.isfinite(endpoint).all()
        or len(set(material_ids)) != len(material_ids)
        or not tails
        or any(not 0.0 < value < 1.0 for value in tails)
    ):
        raise ValueError("NEXT45 development arrays or tail grid differ")
    discovery_mask = split == "discovery"
    validation_mask = split == "validation"
    discovery_endpoint = endpoint[discovery_mask]
    n_discovery = len(discovery_endpoint)
    protected_bits = _mask_bits(discovery_endpoint <= PROTECTED_MAX)
    changed_bits = _mask_bits(discovery_endpoint > PROTECTED_MAX)
    substantial_bits = _mask_bits(discovery_endpoint >= SUBSTANTIAL_MIN)
    severe_bits = _mask_bits(discovery_endpoint >= SEVERE_MIN)

    infos: list[dict[str, object]] = []
    for feature in candidate_features:
        if feature not in features:
            continue
        values = pd.to_numeric(features[feature], errors="coerce").to_numpy(float)
        finite_discovery = np.isfinite(values[discovery_mask])
        if wilson_lower_bound(int(finite_discovery.sum()), n_discovery) < PRIMARY_GATES["coverage_lower"]:
            continue
        finite_values = values[discovery_mask][finite_discovery]
        if len(finite_values) < 4:
            continue
        center = float(np.median(finite_values))
        q25, q75 = np.quantile(finite_values, (0.25, 0.75))
        scale = float(q75 - q25)
        if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0.0:
            continue
        support_bits = _mask_bits(finite_discovery)
        conditions: list[dict[str, object]] = []
        seen: set[tuple[int, float]] = set()
        for direction in (-1, 1):
            risk = int(direction) * (values[discovery_mask] - center) / scale
            for tail in tails:
                cutoff = _quantile(risk[finite_discovery], 1.0 - tail)
                identity = (int(direction), float(cutoff))
                if identity in seen:
                    continue
                seen.add(identity)
                condition = finite_discovery & (risk >= cutoff)
                conditions.append(
                    {
                        "bits": _mask_bits(condition),
                        "term": {
                            "feature": str(feature),
                            "direction": int(direction),
                            "center": center,
                            "scale": scale,
                            "cutoff": float(cutoff),
                        },
                    }
                )
        infos.append(
            {
                "feature": str(feature),
                "support_bits": support_bits,
                "conditions": conditions,
            }
        )
    if not infos:
        raise ValueError("NEXT45 has no coverage-qualified variable feature")

    best_formula: dict[str, object] | None = None
    best_metrics: dict[str, object] | None = None
    best_rank: tuple[float, ...] | None = None
    best_key: str | None = None
    candidate_count = 0

    def consider(support: int, reject: int, terms: list[dict[str, object]]) -> None:
        nonlocal best_formula, best_metrics, best_rank, best_key, candidate_count
        metrics = _metrics_from_bits(
            supported=support,
            reject=reject,
            rows=n_discovery,
            protected=protected_bits,
            changed=changed_bits,
            substantial=substantial_bits,
            severe=severe_bits,
        )
        formula = {
            "kind": "conjunctive",
            "terms": terms,
            "missing_policy": "KEEP",
        }
        rank = _rank(metrics)
        key = json.dumps(formula, sort_keys=True, separators=(",", ":"), allow_nan=False)
        candidate_count += 1
        if best_rank is None or rank > best_rank or (rank == best_rank and key < str(best_key)):
            best_formula = formula
            best_metrics = metrics
            best_rank = rank
            best_key = key

    for info in infos:
        support = int(info["support_bits"])
        for condition in info["conditions"]:
            consider(support, int(condition["bits"]), [dict(condition["term"])])
    for left, right in combinations(infos, 2):
        support = int(left["support_bits"]) & int(right["support_bits"])
        if wilson_lower_bound(support.bit_count(), n_discovery) < PRIMARY_GATES["coverage_lower"]:
            continue
        for left_condition in left["conditions"]:
            left_bits = int(left_condition["bits"])
            for right_condition in right["conditions"]:
                consider(
                    support,
                    left_bits & int(right_condition["bits"]),
                    [dict(left_condition["term"]), dict(right_condition["term"])],
                )
    if best_formula is None or best_metrics is None:
        raise RuntimeError("NEXT45 exhaustive catalogue produced no formula")
    score, supported, reject = apply_formula(features, best_formula)
    validation_metrics = _decision_metrics(
        supported=supported[validation_mask],
        reject=reject[validation_mask],
        endpoint=endpoint[validation_mask],
    )
    full_metrics = _decision_metrics(
        supported=supported, reject=reject, endpoint=endpoint
    )
    diagnostics = {
        role: _continuous_diagnostics(score[mask], supported[mask], endpoint[mask])
        for role, mask in (
            ("discovery", discovery_mask),
            ("validation", validation_mask),
            ("full_development", np.ones(len(features), dtype=bool)),
        )
    }
    return {
        "selected_formula": best_formula,
        "discovery_metrics": best_metrics,
        "validation_metrics": validation_metrics,
        "full_development_metrics": full_metrics,
        "continuous_diagnostics": diagnostics,
        "passes_both_internal_splits": bool(
            best_metrics["passes_primary_gates"]
            and validation_metrics["passes_primary_gates"]
        ),
        "candidate_count": candidate_count,
        "coverage_qualified_feature_count": len(infos),
        "score": score,
        "supported": supported,
        "reject": reject,
    }


def _validate_next44_search(path: Path, manifest_path: Path) -> dict[str, object]:
    if path.name != NEXT44_SEARCH_NAME:
        raise ValueError("NEXT44 search filename differs")
    manifest = _strict_json(manifest_path, role="NEXT44 search manifest")
    outputs = manifest.get("outputs_sha256")
    if (
        manifest.get("protocol") != NEXT44_SEARCH_PROTOCOL
        or manifest.get("validation_labels_used_for_selection") is not False
        or manifest.get("law_execution_dft_values_read") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get(path.name) != _sha256(path)
    ):
        raise ValueError("NEXT44 comparison search contract differs")
    return _strict_json(path, role="NEXT44 search result")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def run_exhaustive_search(
    *,
    base_feature_path: Path,
    base_manifest_path: Path,
    rich_feature_path: Path,
    rich_manifest_path: Path,
    evaluation_path: Path,
    evaluation_manifest_path: Path,
    next44_search_path: Path,
    next44_search_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Publish the exhaustive development rectangle and comparisons."""

    input_paths = {
        "next43_features": Path(base_feature_path).resolve(),
        "next43_feature_manifest": Path(base_manifest_path).resolve(),
        "next44_features": Path(rich_feature_path).resolve(),
        "next44_feature_manifest": Path(rich_manifest_path).resolve(),
        "development_evaluation": Path(evaluation_path).resolve(),
        "development_evaluation_manifest": Path(evaluation_manifest_path).resolve(),
        "next44_search": Path(next44_search_path).resolve(),
        "next44_search_manifest": Path(next44_search_manifest_path).resolve(),
    }
    if any(not path.is_file() for path in input_paths.values()):
        raise FileNotFoundError("NEXT45 exhaustive-search input is missing")
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing existing output: {target}")
    base, evaluation = _validate_next43_inputs(
        feature_path=input_paths["next43_features"],
        feature_manifest_path=input_paths["next43_feature_manifest"],
        evaluation_path=input_paths["development_evaluation"],
        evaluation_manifest_path=input_paths["development_evaluation_manifest"],
    )
    rich = _validate_rich_table(
        input_paths["next44_features"], input_paths["next44_feature_manifest"]
    )
    previous = _validate_next44_search(
        input_paths["next44_search"], input_paths["next44_search_manifest"]
    )
    combined = combine_feature_tables(
        base,
        rich,
        base_features=BASE_FEATURE_NAMES,
        rich_features=RICH_FEATURE_NAMES,
    )
    material_ids = combined.material_id.astype(str).to_numpy()
    if material_ids.tolist() != evaluation.material_id.astype(str).tolist():
        raise ValueError("NEXT45 feature and endpoint identity differs")
    endpoint = evaluation[ENDPOINT_COLUMN].to_numpy(float)
    split = deterministic_split(material_ids)
    result = search_exhaustive_rectangles(
        features=combined,
        material_ids=material_ids,
        endpoint=endpoint,
        split=split,
        candidate_features=ALL_FEATURE_NAMES,
        tail_fractions=TAIL_FRACTIONS,
    )
    masks = {
        "discovery": split == "discovery",
        "validation": split == "validation",
        "full_development": np.ones(len(split), dtype=bool),
    }
    previous_formula = previous.get("selected_formula")
    if not isinstance(previous_formula, Mapping):
        raise ValueError("NEXT44 selected formula is missing")
    previous_score, previous_supported, previous_reject = apply_formula(
        combined, previous_formula
    )
    previous_metrics = {
        role: _decision_metrics(
            supported=previous_supported[mask],
            reject=previous_reject[mask],
            endpoint=endpoint[mask],
        )
        for role, mask in masks.items()
    }
    formula_document = {
        "protocol": PROTOCOL,
        "role": "development rectangle; not frozen for unseen confirmation",
        "formula": result["selected_formula"],
        "missing_policy": "KEEP",
        "execution_input": "one_raw_pre_dft_pre_mlip_x0_only",
        "execution_uses_dft": False,
        "execution_uses_endpoint_or_later_geometry": False,
        "execution_uses_mlip_or_model_potential": False,
        "execution_runs_physical_relaxation": False,
        "passes_both_internal_splits": result["passes_both_internal_splits"],
        "confirmation_candidate_ready": result["passes_both_internal_splits"],
        "requires_unseen_source_qualified_confirmation": True,
    }
    search_document = {
        key: value
        for key, value in result.items()
        if key not in {"score", "supported", "reject"}
    }
    search_document.update(
        {
            "protocol": PROTOCOL,
            "data_role": "NEXT42 opened converged endpoints used for development only",
            "candidate_feature_count": len(ALL_FEATURE_NAMES),
            "tail_fractions": list(TAIL_FRACTIONS),
            "split_counts": {role: int(mask.sum()) for role, mask in masks.items()},
            "primary_gates": dict(PRIMARY_GATES),
            "next44_selected_formula_recomputed_metrics": previous_metrics,
            "frozen_baselines": _baseline_metrics(evaluation, masks),
            "scientific_confirmation": False,
        }
    )
    prediction = pd.DataFrame(
        {
            "material_id": material_ids,
            "split_role": split,
            "analytic_supported": result["supported"],
            "analytic_score": result["score"],
            "analytic_reject": result["reject"],
            "next44_score": previous_score,
            "next44_supported": previous_supported,
            "next44_reject": previous_reject,
            ENDPOINT_COLUMN: endpoint,
        }
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        formula_path = staging / FORMULA_NAME
        search_path = staging / SEARCH_NAME
        prediction_path = staging / PREDICTION_NAME
        formula_path.write_bytes(_json_bytes(formula_document))
        search_path.write_bytes(_json_bytes(search_document))
        prediction.to_parquet(prediction_path, index=False)
        repository = Path(__file__).resolve().parents[1]
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "evidence_role": "exhaustive rectangular development, not confirmation",
            "development_labels_opened": True,
            "offline_dft_final_geometry_label_used": True,
            "offline_dft_force_convergence_filter_used": True,
            "offline_dft_energy_used": False,
            "law_execution_dft_values_read": False,
            "law_execution_endpoint_or_later_geometry_read": False,
            "law_execution_mlip_or_model_potential_used": False,
            "law_execution_learned_energy_force_stress_proxy_used": False,
            "law_execution_physical_relaxation_executed": False,
            "thresholds_fit_on_discovery_only": True,
            "validation_labels_used_for_selection": False,
            "passes_both_internal_splits": result["passes_both_internal_splits"],
            "scientific_confirmation": False,
            "inputs_sha256": {role: _sha256(path) for role, path in input_paths.items()},
            "executed_source_sha256": {
                "src/next43_finite_law_search.py": _sha256(
                    repository / "src/next43_finite_law_search.py"
                ),
                "src/next44_rich_law_search.py": _sha256(
                    repository / "src/next44_rich_law_search.py"
                ),
                "src/next45_exhaustive_rectangles.py": _sha256(
                    repository / "src/next45_exhaustive_rectangles.py"
                ),
            },
            "outputs_sha256": {
                path.name: _sha256(path)
                for path in (formula_path, search_path, prediction_path)
            },
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if any(_sha256(path) != manifest["inputs_sha256"][role] for role, path in input_paths.items()):
            raise RuntimeError("NEXT45 input changed during exhaustive search")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next43-features", required=True, type=Path)
    parser.add_argument("--next43-feature-manifest", required=True, type=Path)
    parser.add_argument("--next44-features", required=True, type=Path)
    parser.add_argument("--next44-feature-manifest", required=True, type=Path)
    parser.add_argument("--development-evaluation", required=True, type=Path)
    parser.add_argument("--development-evaluation-manifest", required=True, type=Path)
    parser.add_argument("--next44-search", required=True, type=Path)
    parser.add_argument("--next44-search-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run_exhaustive_search(
        base_feature_path=args.next43_features,
        base_manifest_path=args.next43_feature_manifest,
        rich_feature_path=args.next44_features,
        rich_manifest_path=args.next44_feature_manifest,
        evaluation_path=args.development_evaluation,
        evaluation_manifest_path=args.development_evaluation_manifest,
        next44_search_path=args.next44_search,
        next44_search_manifest_path=args.next44_search_manifest,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "passes_both_internal_splits": manifest["passes_both_internal_splits"],
                "output": str(args.output_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
