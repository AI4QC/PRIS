#!/usr/bin/env python3
"""Search and freeze a deliberately small NEXT26 analytic law family."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next23_evaluate import _roc_auc
from src.next26_evaluate import PRIMARY_GATES, decision_metrics
from src.next26_omc25 import severe_dft_response
from src.next26_packing import FEATURE_COLUMNS, FORBIDDEN_TOKENS


PROTOCOL = "2026-08-03-next26-omc25-packing-law-development-v1"
FROZEN_RULE_NAME = "FROZEN_RULE.json"
CANDIDATES_NAME = "development_candidates.parquet"
MANIFEST_NAME = "MANIFEST.json"

# Direction is fixed by physical interpretation, not learned sign fitting.
SIGNED_TERMS: Mapping[str, tuple[str, int]] = {
    "over_cov_packing": ("cov_packing", +1),
    "over_density": ("density_proxy", +1),
    "short_nonbond_q01": ("nonbond_vdw_q01", -1),
    "nonbond_clash": ("nonbond_clash_frac085", +1),
    "small_volume": ("volume_pa", -1),
    "bond_dispersion": ("bond_ratio_sd", +1),
    "cell_anisotropy": ("cell_anisotropy", +1),
    "under_cov_packing": ("cov_packing", -1),
    "under_density": ("density_proxy", -1),
    "large_volume": ("volume_pa", +1),
}
ABS_FEATURES = ("cov_packing", "density_proxy", "volume_pa")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _validate_ids(frame: pd.DataFrame, *, role: str) -> pd.Series:
    if "material_id" not in frame or frame["material_id"].isna().any() or frame["material_id"].duplicated().any():
        raise ValueError(f"{role} material IDs are invalid")
    ids = frame["material_id"].astype(str)
    if not ids.map(bool).all():
        raise ValueError(f"{role} material IDs are empty")
    return ids


def _validate_feature_contract(features: pd.DataFrame) -> None:
    forbidden = [
        str(column)
        for column in features.columns
        if column != "material_id" and any(token in str(column).lower() for token in FORBIDDEN_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"feature table crossed no-DFT contract: {forbidden}")
    required = {feature for feature, _ in SIGNED_TERMS.values()} | set(ABS_FEATURES)
    if not required.issubset(features.columns):
        raise ValueError(f"feature table lacks analytic terms: {sorted(required-set(features.columns))}")


def _robust_parameters(features: pd.DataFrame) -> dict[str, dict[str, float]]:
    required = sorted({feature for feature, _ in SIGNED_TERMS.values()} | set(ABS_FEATURES))
    parameters: dict[str, dict[str, float]] = {}
    for feature in required:
        values = pd.to_numeric(features[feature], errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            continue
        q25, median, q75 = np.quantile(finite, [0.25, 0.5, 0.75])
        scale = float(q75 - q25)
        if math.isfinite(scale) and scale > 0:
            parameters[feature] = {"median": float(median), "scale_iqr": scale}
    return parameters


def _candidate_catalogue(parameters: Mapping[str, Mapping[str, float]]) -> list[dict[str, object]]:
    signed = [name for name, (feature, _sign) in SIGNED_TERMS.items() if feature in parameters]
    result: list[dict[str, object]] = []
    for name in signed:
        feature, sign = SIGNED_TERMS[name]
        result.append(
            {
                "name": name,
                "formula_family": "signed_robust_z",
                "terms": [{"name": name, "feature": feature, "sign": sign, "transform": "signed"}],
            }
        )
    for first, second in itertools.combinations(signed, 2):
        feature_a, sign_a = SIGNED_TERMS[first]
        feature_b, sign_b = SIGNED_TERMS[second]
        if feature_a == feature_b:
            continue
        result.append(
            {
                "name": f"{first}+{second}",
                "formula_family": "equal_weight_signed_robust_z_sum",
                "terms": [
                    {"name": first, "feature": feature_a, "sign": sign_a, "transform": "signed"},
                    {"name": second, "feature": feature_b, "sign": sign_b, "transform": "signed"},
                ],
            }
        )
        result.append(
            {
                "name": f"AND({first},{second})",
                "formula_family": "conjunctive_signed_robust_z_min",
                "terms": [
                    {"name": first, "feature": feature_a, "sign": sign_a, "transform": "signed"},
                    {"name": second, "feature": feature_b, "sign": sign_b, "transform": "signed"},
                ],
            }
        )
    for feature in ABS_FEATURES:
        if feature in parameters:
            result.append(
                {
                    "name": f"absolute_{feature}",
                    "formula_family": "absolute_robust_z",
                    "terms": [{"name": f"absolute_{feature}", "feature": feature, "sign": 1, "transform": "absolute"}],
                }
            )
    return result


def score_rule(
    features: pd.DataFrame,
    *,
    terms: Sequence[Mapping[str, object]],
    parameters: Mapping[str, Mapping[str, float]],
    formula_family: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    contributions: list[np.ndarray] = []
    support = np.ones(len(features), dtype=bool)
    for term in terms:
        feature = str(term["feature"])
        if feature not in features or feature not in parameters:
            raise ValueError(f"rule feature is unavailable: {feature}")
        values = pd.to_numeric(features[feature], errors="coerce").to_numpy(float)
        support &= np.isfinite(values)
        median = float(parameters[feature]["median"])
        scale = float(parameters[feature]["scale_iqr"])
        if not math.isfinite(median) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid robust parameters for {feature}")
        standardized = (values - median) / scale
        if term.get("transform") == "absolute":
            contribution = np.abs(standardized)
        elif term.get("transform") == "signed" and int(term.get("sign", 0)) in {-1, 1}:
            contribution = int(term["sign"]) * standardized
        else:
            raise ValueError("unsupported analytic term transform")
        contributions.append(np.where(np.isfinite(contribution), contribution, 0.0))
    if formula_family == "conjunctive_signed_robust_z_min":
        if len(contributions) != 2:
            raise ValueError("conjunctive formula requires exactly two terms")
        score = np.minimum(contributions[0], contributions[1])
    else:
        score = np.sum(np.vstack(contributions), axis=0)
    score[~support] = np.nan
    return score, support


def _best_threshold(
    score: np.ndarray, support: np.ndarray, endpoint_positive: np.ndarray
) -> tuple[dict[str, object] | None, dict[str, object] | None, list[dict[str, object]]]:
    thresholds = np.unique(score[support & np.isfinite(score)])[::-1]
    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    diagnostic: dict[str, object] | None = None
    for threshold in thresholds:
        reject = support & (score >= threshold)
        metrics = decision_metrics(supported=support, reject=reject, endpoint_positive=endpoint_positive)
        record = {"threshold": float(threshold), **metrics}
        rows.append(record)
        ratios = tuple(
            float(metrics[name]) / cutoff for name, cutoff in PRIMARY_GATES.items()
        )
        diagnostic_rank = (
            min(ratios),
            sum(min(value, 1.0) for value in ratios),
            float(metrics["endpoint_positive_precision_lower"]),
            float(metrics["endpoint_negative_protection_lower"]),
            float(metrics["savings_lower"]),
        )
        if diagnostic is None or diagnostic_rank > diagnostic["_diagnostic_rank"]:
            diagnostic = {**record, "_diagnostic_rank": diagnostic_rank}
        if not bool(metrics["passes_primary_gates"]):
            continue
        rank = (
            float(metrics["endpoint_positive_precision_lower"]),
            float(metrics["endpoint_negative_protection_lower"]),
            float(metrics["savings_lower"]),
            int(metrics["rejected"]),
        )
        if best is None or rank > best["_rank"]:
            best = {**record, "_rank": rank}
    return best, diagnostic, rows


def search_and_freeze(
    *, features: pd.DataFrame, endpoints: pd.DataFrame, output_dir: Path
) -> dict[str, object]:
    """Search development labels and freeze at most a two-term analytic law."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    features = features.copy()
    endpoints = endpoints.copy()
    feature_ids = _validate_ids(features, role="features")
    endpoint_ids = _validate_ids(endpoints, role="endpoints")
    _validate_feature_contract(features)
    if set(feature_ids) != set(endpoint_ids):
        raise ValueError("development feature and endpoint IDs differ")
    merged = features.merge(endpoints, on="material_id", how="inner", validate="one_to_one")
    merged = merged.sort_values("material_id", kind="stable").reset_index(drop=True)
    endpoint_positive = severe_dft_response(merged).to_numpy(bool)
    parameters = _robust_parameters(merged)
    catalogue = _candidate_catalogue(parameters)
    candidate_rows: list[dict[str, object]] = []
    eligible: list[dict[str, object]] = []
    for candidate in catalogue:
        terms = candidate["terms"]
        assert isinstance(terms, list)
        score, support = score_rule(
            merged,
            terms=terms,
            parameters=parameters,
            formula_family=str(candidate["formula_family"]),
        )
        best, diagnostic, threshold_rows = _best_threshold(score, support, endpoint_positive)
        auc = _roc_auc(score[support], endpoint_positive[support]) if support.any() else None
        summary = {
            "candidate": candidate["name"],
            "formula_family": candidate["formula_family"],
            "terms_json": json.dumps(terms, sort_keys=True, separators=(",", ":")),
            "support": int(support.sum()),
            "endpoint_auc": auc,
            "eligible": best is not None,
        }
        if best is None:
            assert diagnostic is not None
            diagnostic = {
                key: value
                for key, value in diagnostic.items()
                if key != "_diagnostic_rank"
            }
            candidate_rows.append(
                {**summary, **diagnostic, "diagnostic_threshold_only": True}
            )
            continue
        best = {key: value for key, value in best.items() if key != "_rank"}
        candidate_rows.append(
            {**summary, **best, "diagnostic_threshold_only": False}
        )
        eligible.append({**candidate, "score": score, "support": support, "metrics": best, "auc": auc})

    selected: dict[str, object] | None = None
    if eligible:
        def selection_rank(item: Mapping[str, object]) -> tuple[float, float, float, float, int, str]:
            metrics = item["metrics"]
            assert isinstance(metrics, Mapping)
            return (
                float(metrics["endpoint_positive_precision_lower"]),
                float(metrics["endpoint_negative_protection_lower"]),
                float(metrics["savings_lower"]),
                float(item["auc"] or 0.0),
                -len(item["terms"]),  # type: ignore[arg-type]
                str(item["name"]),
            )
        selected = max(eligible, key=selection_rank)

    frozen_at = datetime.now(timezone.utc).isoformat()
    if selected is None:
        rule: dict[str, object] = {
            "protocol": PROTOCOL,
            "eligible": False,
            "frozen_at_utc": frozen_at,
            "formula_family": None,
            "selected_candidate": None,
            "terms": [],
            "parameters": parameters,
            "threshold": None,
        }
    else:
        metrics = selected["metrics"]
        assert isinstance(metrics, Mapping)
        rule = {
            "protocol": PROTOCOL,
            "eligible": True,
            "frozen_at_utc": frozen_at,
            "formula_family": selected["formula_family"],
            "selected_candidate": selected["name"],
            "terms": selected["terms"],
            "parameters": parameters,
            "threshold": float(metrics["threshold"]),
            "development_metrics": dict(metrics),
            "endpoint_auc": selected["auc"],
        }
    rule.update(
        {
            "endpoint_definition": {
                "force0_max_ge": 1.0,
                "force0_rms_ge": 0.40,
                "energy_drop_pa_ge": 0.040,
                "stress0_norm_ge": 0.030,
                "combination": "logical_or",
            },
            "primary_gates": dict(PRIMARY_GATES),
            "missing_policy": "fail_open_do_not_reject",
            "maximum_terms": 2,
            "dense_model_used": False,
            "model_or_proxy_potential_used": False,
            "prospective_labels_opened": False,
        }
    )
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["eligible", "endpoint_positive_precision_lower", "endpoint_negative_protection_lower", "savings_lower", "candidate"],
        ascending=[False, False, False, False, True],
        kind="stable",
    )
    source_hashes = {"src/next26_development.py": _sha256(Path(__file__).resolve())}
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "frozen_at_utc": frozen_at,
        "development_labels_opened": True,
        "prospective_labels_opened": False,
        "model_or_proxy_potential_used": False,
        "dense_model_used": False,
        "eligible": bool(rule["eligible"]),
        "counts": {
            "rows": len(merged),
            "endpoint_positives": int(endpoint_positive.sum()),
            "endpoint_negatives": int((~endpoint_positive).sum()),
            "candidate_formulas": len(catalogue),
            "eligible_formulas": len(eligible),
        },
        "executed_source_sha256": source_hashes,
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        rule_path = staging / FROZEN_RULE_NAME
        candidates_path = staging / CANDIDATES_NAME
        rule_path.write_bytes(_json_bytes(rule))
        candidates.to_parquet(candidates_path, index=False)
        manifest["outputs_sha256"] = {
            FROZEN_RULE_NAME: _sha256(rule_path),
            CANDIDATES_NAME: _sha256(candidates_path),
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(Path(__file__).resolve()) != source_hashes["src/next26_development.py"]:
            raise RuntimeError("development source changed before publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--endpoints", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = search_and_freeze(
        features=pd.read_parquet(args.features),
        endpoints=pd.read_parquet(args.endpoints),
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FROZEN_RULE_NAME", "search_and_freeze", "score_rule"]
