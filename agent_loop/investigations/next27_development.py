#!/usr/bin/env python3
"""Freeze a periodic contact-pressure law that transports across dev shards."""

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
from src.next26_development import score_rule
from src.next26_evaluate import decision_metrics
from src.next26_omc25 import severe_dft_response
from src.next27_periodic_packing import NEXT27_FEATURE_COLUMNS


PROTOCOL = "2026-08-03-next27-shard-stable-periodic-pressure-law-v1"
FROZEN_RULE_NAME = "FROZEN_RULE.json"
CANDIDATES_NAME = "development_candidates.parquet"
MANIFEST_NAME = "MANIFEST.json"
SHARD_GATES: Mapping[str, float] = {
    "coverage_lower": 0.95,
    "endpoint_negative_protection": 0.95,
    "endpoint_positive_precision": 0.80,
    "savings": 0.05,
}
PROSPECTIVE_GATES: Mapping[str, float] = {
    "coverage_lower": 0.95,
    "endpoint_negative_protection_lower": 0.95,
    "endpoint_positive_precision_lower": 0.75,
    "savings_lower": 0.04,
}
SIGNED_TERMS: Mapping[str, tuple[str, int]] = {
    "short_periodic_min": ("periodic_nonbond_vdw_min", -1),
    "short_periodic_q01": ("periodic_nonbond_vdw_q01", -1),
    "short_periodic_q05": ("periodic_nonbond_vdw_q05", -1),
    "overlap2_pressure": ("periodic_overlap2_pa", +1),
    "overlap3_pressure": ("periodic_overlap3_pa", +1),
    "repulsion12_pressure": ("periodic_repulsion12_pa", +1),
    "contact_coord100": ("periodic_contact_coord100", +1),
    "contact_coord105": ("periodic_contact_coord105", +1),
    "contact_coord110": ("periodic_contact_coord110", +1),
    "short_nearest_mean": ("periodic_nearest_mean", -1),
    "short_nearest_q10": ("periodic_nearest_q10", -1),
    "periodic_pairs": ("periodic_pairs_pa", +1),
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _validate(features: pd.DataFrame, endpoints: pd.DataFrame) -> pd.DataFrame:
    required = {"material_id", "development_shard", *NEXT27_FEATURE_COLUMNS}
    if not required.issubset(features.columns):
        raise ValueError(f"development features lack columns: {sorted(required-set(features.columns))}")
    for frame, role in ((features, "features"), (endpoints, "endpoints")):
        if "material_id" not in frame or frame["material_id"].isna().any() or frame["material_id"].duplicated().any():
            raise ValueError(f"{role} material IDs are invalid")
    if set(features["material_id"].astype(str)) != set(endpoints["material_id"].astype(str)):
        raise ValueError("development feature and endpoint IDs differ")
    if features["development_shard"].isna().any() or features["development_shard"].nunique() < 2:
        raise ValueError("at least two exact development shards are required")
    merged = features.merge(endpoints, on="material_id", how="inner", validate="one_to_one")
    return merged.sort_values("material_id", kind="stable").reset_index(drop=True)


def _parameters(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for feature in NEXT27_FEATURE_COLUMNS:
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            continue
        q25, median, q75 = np.quantile(finite, [0.25, 0.5, 0.75])
        scale = float(q75 - q25)
        if math.isfinite(scale) and scale > 0:
            result[feature] = {"median": float(median), "scale_iqr": scale}
    return result


def _catalogue(parameters: Mapping[str, Mapping[str, float]]) -> list[dict[str, object]]:
    names = [name for name, (feature, _sign) in SIGNED_TERMS.items() if feature in parameters]
    result: list[dict[str, object]] = []
    for name in names:
        feature, sign = SIGNED_TERMS[name]
        result.append(
            {
                "name": name,
                "formula_family": "signed_robust_z",
                "terms": [{"name": name, "feature": feature, "sign": sign, "transform": "signed"}],
            }
        )
    for first, second in itertools.combinations(names, 2):
        feature_a, sign_a = SIGNED_TERMS[first]
        feature_b, sign_b = SIGNED_TERMS[second]
        terms = [
            {"name": first, "feature": feature_a, "sign": sign_a, "transform": "signed"},
            {"name": second, "feature": feature_b, "sign": sign_b, "transform": "signed"},
        ]
        result.extend(
            [
                {
                    "name": f"{first}+{second}",
                    "formula_family": "equal_weight_signed_robust_z_sum",
                    "terms": terms,
                },
                {
                    "name": f"AND({first},{second})",
                    "formula_family": "conjunctive_signed_robust_z_min",
                    "terms": terms,
                },
            ]
        )
    return result


def _passes(metrics: Mapping[str, object], gates: Mapping[str, float]) -> bool:
    return all(float(metrics[name]) >= cutoff for name, cutoff in gates.items())


def _best_threshold(
    score: np.ndarray,
    support: np.ndarray,
    positive: np.ndarray,
    shards: np.ndarray,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    best: dict[str, object] | None = None
    diagnostic: dict[str, object] | None = None
    shard_names = sorted(set(str(value) for value in shards))
    for threshold in np.unique(score[support & np.isfinite(score)])[::-1]:
        reject = support & (score >= threshold)
        pooled = decision_metrics(supported=support, reject=reject, endpoint_positive=positive)
        by_shard: dict[str, dict[str, object]] = {}
        for name in shard_names:
            mask = shards.astype(str) == name
            by_shard[name] = decision_metrics(
                supported=support[mask], reject=reject[mask], endpoint_positive=positive[mask]
            )
        eligible = _passes(pooled, PROSPECTIVE_GATES) and all(
            _passes(metrics, SHARD_GATES) for metrics in by_shard.values()
        )
        worst_precision = min(float(metrics["endpoint_positive_precision"]) for metrics in by_shard.values())
        worst_savings = min(float(metrics["savings"]) for metrics in by_shard.values())
        worst_protection = min(float(metrics["endpoint_negative_protection"]) for metrics in by_shard.values())
        rank = (
            int(eligible),
            worst_precision,
            sum(float(metrics["endpoint_positive_precision"]) for metrics in by_shard.values()),
            worst_protection,
            worst_savings,
            int(pooled["rejected"]),
        )
        record = {
            "threshold": float(threshold),
            "pooled_metrics": pooled,
            "shard_metrics": by_shard,
            "eligible": eligible,
            "rank": rank,
        }
        if diagnostic is None or rank > diagnostic["rank"]:
            diagnostic = record
        if eligible and (best is None or rank > best["rank"]):
            best = record
    assert diagnostic is not None
    return best, diagnostic


def search_and_freeze(
    *, features: pd.DataFrame, endpoints: pd.DataFrame, output_dir: Path
) -> dict[str, object]:
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    merged = _validate(features.copy(), endpoints.copy())
    positive = severe_dft_response(merged).to_numpy(bool)
    shards = merged["development_shard"].astype(str).to_numpy()
    parameters = _parameters(merged)
    catalogue = _catalogue(parameters)
    rows: list[dict[str, object]] = []
    eligible_candidates: list[dict[str, object]] = []
    for candidate in catalogue:
        terms = candidate["terms"]
        assert isinstance(terms, list)
        score, support = score_rule(
            merged,
            terms=terms,
            parameters=parameters,
            formula_family=str(candidate["formula_family"]),
        )
        best, diagnostic = _best_threshold(score, support, positive, shards)
        chosen = best or diagnostic
        auc = _roc_auc(score[support], positive[support])
        pooled = chosen["pooled_metrics"]
        by_shard = chosen["shard_metrics"]
        assert isinstance(pooled, Mapping) and isinstance(by_shard, Mapping)
        rows.append(
            {
                "candidate": candidate["name"],
                "formula_family": candidate["formula_family"],
                "terms_json": json.dumps(terms, sort_keys=True, separators=(",", ":")),
                "threshold": chosen["threshold"],
                "eligible": best is not None,
                "endpoint_auc": auc,
                "pooled_rejected": pooled["rejected"],
                "pooled_precision": pooled["endpoint_positive_precision"],
                "pooled_precision_lower": pooled["endpoint_positive_precision_lower"],
                "pooled_protection_lower": pooled["endpoint_negative_protection_lower"],
                "pooled_savings_lower": pooled["savings_lower"],
                "shard_metrics_json": json.dumps(by_shard, sort_keys=True, separators=(",", ":")),
            }
        )
        if best is not None:
            eligible_candidates.append(
                {**candidate, "threshold_record": best, "auc": auc}
            )

    selected: dict[str, object] | None = None
    if eligible_candidates:
        def rank_candidate(item: Mapping[str, object]) -> tuple[object, ...]:
            record = item["threshold_record"]
            assert isinstance(record, Mapping)
            rank = record["rank"]
            assert isinstance(rank, tuple)
            return (*rank, float(item["auc"] or 0.0), -len(item["terms"]), str(item["name"]))  # type: ignore[arg-type]
        selected = max(eligible_candidates, key=rank_candidate)

    frozen_at = datetime.now(timezone.utc).isoformat()
    if selected is None:
        rule: dict[str, object] = {
            "protocol": PROTOCOL,
            "eligible": False,
            "frozen_at_utc": frozen_at,
            "selected_candidate": None,
            "formula_family": None,
            "terms": [],
            "parameters": parameters,
            "threshold": None,
        }
    else:
        record = selected["threshold_record"]
        assert isinstance(record, Mapping)
        rule = {
            "protocol": PROTOCOL,
            "eligible": True,
            "frozen_at_utc": frozen_at,
            "selected_candidate": selected["name"],
            "formula_family": selected["formula_family"],
            "terms": selected["terms"],
            "parameters": parameters,
            "threshold": float(record["threshold"]),
            "development_pooled_metrics": record["pooled_metrics"],
            "development_shard_metrics": record["shard_metrics"],
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
            "shard_development_gates": dict(SHARD_GATES),
            "prospective_aggregate_gates": dict(PROSPECTIVE_GATES),
            "maximum_terms": 2,
            "missing_policy": "fail_open_do_not_reject",
            "model_or_proxy_potential_used": False,
            "prospective_labels_opened": False,
        }
    )
    candidates = pd.DataFrame(rows).sort_values(
        ["eligible", "pooled_precision_lower", "pooled_savings_lower", "endpoint_auc", "candidate"],
        ascending=[False, False, False, False, True],
        kind="stable",
    )
    source_hashes = {"src/next27_development.py": _sha256(Path(__file__).resolve())}
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "frozen_at_utc": frozen_at,
        "development_labels_opened": True,
        "prospective_labels_opened": False,
        "development_shards": sorted(set(shards)),
        "eligible": bool(rule["eligible"]),
        "model_or_proxy_potential_used": False,
        "counts": {
            "rows": len(merged),
            "endpoint_positives": int(positive.sum()),
            "candidate_formulas": len(catalogue),
            "eligible_formulas": len(eligible_candidates),
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
        if _sha256(Path(__file__).resolve()) != source_hashes["src/next27_development.py"]:
            raise RuntimeError("NEXT27 development source changed")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", nargs="+", required=True, type=Path)
    parser.add_argument("--endpoints", nargs="+", required=True, type=Path)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if not (len(args.features) == len(args.endpoints) == len(args.shards)):
        raise ValueError("features, endpoints, and shards must have equal lengths")
    feature_parts: list[pd.DataFrame] = []
    endpoint_parts: list[pd.DataFrame] = []
    for feature_path, endpoint_path, shard in zip(args.features, args.endpoints, args.shards, strict=True):
        feature = pd.read_parquet(feature_path)
        feature["development_shard"] = str(shard)
        feature_parts.append(feature)
        endpoint_parts.append(pd.read_parquet(endpoint_path))
    result = search_and_freeze(
        features=pd.concat(feature_parts, ignore_index=True),
        endpoints=pd.concat(endpoint_parts, ignore_index=True),
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FROZEN_RULE_NAME", "PROSPECTIVE_GATES", "search_and_freeze"]
