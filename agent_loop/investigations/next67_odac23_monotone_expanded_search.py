#!/usr/bin/env python3
"""Monotone expanded additive search over the NEXT65 robust discovery data."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from itertools import combinations, product
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next57_odac23_discovery_search import (
    DOMAIN_GATE,
    GATES,
    PROTECTED_MAX,
    REJECTION_FRACTIONS,
    SEVERE_MIN,
    _auc_diagnostics,
    _decision_metrics,
    _gate_rank,
    _safe_auc,
    _term,
    apply_odac23_formula,
)
from src.next60_odac23_robust_scaffold_endpoint import (
    ENDPOINT_COLUMN,
    PROTOCOL as ENDPOINT_PROTOCOL,
    ROLE_LABELS_NAME,
    ROLE_MANIFEST_NAME,
)
from src.next65_odac23_physics_couplings import (
    FEATURES_NAME as SOURCE_FEATURES_NAME,
    NEXT65_FEATURE_NAMES,
    PROTOCOL as SOURCE_FEATURE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next67-odac23-monotone-expanded-additive-search-v1"
DESIGN_SHA256 = "e4cb7ebecbbf045f94b8a4b250a25e37acce2a5823f2738995b3918e0da392a7"
EXPECTED_FEATURE_MANIFEST_SHA256 = (
    "8a858b58f6772a50b1ee3ea900bef9d66eb5636efadfeef878d9c7740011de5c"
)
EXPECTED_ENDPOINT_FIREWALL_SHA256 = (
    "9dbd3f78d2505ba96b33715e6409cd8524e9b909f4134af0020b933dff2f769f"
)
EXPECTED_DISCOVERY_MANIFEST_SHA256 = (
    "6ca39eb42629d626559618474f75aa6bb6571a38a928b3b16512b5d987b76137"
)
EXPECTED_DISCOVERY_LABEL_SHA256 = (
    "1a7c78fd87bb3f5795e59fa3c3799fbbb07a1629b90d472aef7e73740ce7f08a"
)
EXPECTED_NEXT61_FORMULA_SHA256 = (
    "779ffaea31818d850eb4ebe35ca73281de07c3028cd73817dac5206aca65bc62"
)
EXPECTED_NEXT64_FORMULA_SHA256 = (
    "e41d72fec72baf7e8d38bae622ff3748abd4c52c64651c9df4615b4591007890"
)
PAIR_SHORTLIST = 40
TRIPLE_SHORTLIST = 18
PAIR_WEIGHTS = (0.25, 0.5, 1.0, 2.0, 4.0)
TRIPLE_WEIGHTS = (0.5, 1.0, 2.0)
FORMULA_NAME = "NEXT67_ODAC23_MONOTONE_EXPANDED_CANDIDATE.json"
SEARCH_NAME = "NEXT67_ODAC23_MONOTONE_EXPANDED_SEARCH.json"
PREDICTIONS_NAME = "next67_odac23_monotone_expanded_predictions.parquet"
MANIFEST_NAME = "MANIFEST.json"


def _strata(features: pd.DataFrame) -> np.ndarray:
    defective = np.asarray(features["defective"], dtype=bool)
    oms = np.asarray(features["open_metal_site"], dtype=bool)
    return np.asarray(
        [f"defective={int(left)}|oms={int(right)}" for left, right in zip(defective, oms, strict=True)],
        dtype=str,
    )


def search_expanded_additive_rule(
    *,
    features: pd.DataFrame,
    endpoint: Sequence[float],
    candidate_features: Sequence[str] = NEXT65_FEATURE_NAMES,
    anchor_features: Sequence[str] = (),
) -> dict[str, object]:
    endpoint = np.asarray(endpoint, dtype=float)
    required = {
        "combined_supported",
        "periodic_dimension_max",
        "periodic_framework_fraction",
        "defective",
        "open_metal_site",
    }
    if (
        len(features) != len(endpoint)
        or not required.issubset(features.columns)
        or not np.isfinite(endpoint).all()
        or not ((endpoint <= PROTECTED_MAX).any() and (endpoint >= SEVERE_MIN).any())
    ):
        raise ValueError("NEXT67 discovery arrays differ")
    strata = _strata(features)
    dimension = pd.to_numeric(features["periodic_dimension_max"], errors="coerce").to_numpy(float)
    fraction = pd.to_numeric(features["periodic_framework_fraction"], errors="coerce").to_numpy(float)
    base_supported = (
        np.asarray(features["combined_supported"], dtype=bool)
        & np.isfinite(dimension)
        & np.isfinite(fraction)
        & (dimension >= 1.0)
        & (fraction >= 0.5)
    )
    extreme = (endpoint <= PROTECTED_MAX) | (endpoint >= SEVERE_MIN)
    truth = endpoint >= SEVERE_MIN
    infos = []
    for feature in candidate_features:
        if feature not in features:
            continue
        values = pd.to_numeric(features[feature], errors="coerce").to_numpy(float)
        finite = base_supported & np.isfinite(values)
        if int(finite.sum()) < 20:
            continue
        center = float(np.median(values[finite]))
        q25, q75 = np.quantile(values[finite], (0.25, 0.75))
        scale = float(q75 - q25)
        if not math.isfinite(scale) or scale <= 1.0e-12:
            continue
        raw_aucs = []
        for stratum in sorted(set(strata.tolist())):
            mask = finite & extreme & (strata == stratum)
            auc = _safe_auc(values[mask], truth[mask])
            if auc is not None:
                raw_aucs.append(auc)
        if not raw_aucs:
            continue
        direction = 1 if float(np.mean(raw_aucs)) >= 0.5 else -1
        directional = [auc if direction == 1 else 1.0 - auc for auc in raw_aucs]
        risk = direction * (values - center) / scale
        pooled = _safe_auc(risk[finite & extreme], truth[finite & extreme])
        if pooled is None:
            continue
        infos.append(
            {
                "feature": feature,
                "direction": direction,
                "center": center,
                "scale": scale,
                "risk": risk,
                "pooled_directional_auc": pooled,
                "macro_stratum_directional_auc": float(np.mean(directional)),
                "worst_stratum_directional_auc": float(np.min(directional)),
            }
        )
    if not infos:
        raise ValueError("NEXT67 has no evaluable feature")
    infos.sort(
        key=lambda item: (
            -float(item["worst_stratum_directional_auc"]),
            -float(item["macro_stratum_directional_auc"]),
            -float(item["pooled_directional_auc"]),
            str(item["feature"]),
        )
    )
    by_name = {str(info["feature"]): info for info in infos}

    def pool(size: int) -> list[Mapping[str, object]]:
        selected = list(infos[: min(size, len(infos))])
        selected_names = {str(info["feature"]) for info in selected}
        for feature in anchor_features:
            if feature in by_name and feature not in selected_names:
                selected.append(by_name[feature])
                selected_names.add(feature)
        return selected

    best = None
    considered = 0

    def consider(selected: Sequence[Mapping[str, object]], weights: Sequence[float]) -> None:
        nonlocal best, considered
        raw_score = np.zeros(len(features), dtype=float)
        finite = base_supported.copy()
        for info, weight in zip(selected, weights, strict=True):
            risk = np.asarray(info["risk"], dtype=float)
            finite &= np.isfinite(risk)
            raw_score += float(weight) * risk
        if not finite.any():
            return
        score_for_auc = raw_score.copy()
        score_for_auc[~finite] = np.nan
        aucs = _auc_diagnostics(
            score=score_for_auc, supported=finite, endpoint=endpoint, strata=strata
        )
        if aucs["pooled_extreme_auc"] is None:
            return
        thresholds = {
            float(np.quantile(raw_score[finite], 1.0 - rejection, method="inverted_cdf"))
            for rejection in REJECTION_FRACTIONS
        }
        for threshold in sorted(thresholds):
            formula = {
                "kind": "additive",
                "terms": [
                    _term(info, weight)
                    for info, weight in zip(selected, weights, strict=True)
                ],
                "threshold": threshold,
                "missing_policy": "KEEP",
                "domain_gate": dict(DOMAIN_GATE),
            }
            score, supported, reject = apply_odac23_formula(features, formula)
            metrics = _decision_metrics(supported=supported, reject=reject, endpoint=endpoint)
            rank = _gate_rank(metrics, aucs, len(selected))
            key = json.dumps(formula, sort_keys=True, separators=(",", ":"))
            considered += 1
            record = (rank, key, formula, metrics, aucs, score, supported, reject)
            if best is None or rank > best[0] or (rank == best[0] and key < best[1]):
                best = record

    for info in infos:
        consider((info,), (1.0,))
    pair_pool = pool(PAIR_SHORTLIST)
    for selected in combinations(pair_pool, 2):
        for weight in PAIR_WEIGHTS:
            consider(selected, (1.0, weight))
    triple_pool = pool(TRIPLE_SHORTLIST)
    for selected in combinations(triple_pool, 3):
        for second, third in product(TRIPLE_WEIGHTS, repeat=2):
            consider(selected, (1.0, second, third))
    if best is None:
        raise RuntimeError("NEXT67 finite catalogue is empty")
    rank, _key, formula, metrics, aucs, score, supported, reject = best
    return {
        "selected_formula": formula,
        "discovery_metrics": {
            **metrics,
            **{
                key: aucs[key]
                for key in (
                    "pooled_extreme_auc",
                    "macro_stratum_auc",
                    "worst_stratum_auc",
                    "evaluable_strata",
                )
            },
        },
        "stratum_diagnostics": aucs["strata"],
        "passes_discovery_gates": bool(rank[0] == 1.0),
        "candidate_count": considered,
        "evaluable_feature_count": len(infos),
        "pair_pool": [str(info["feature"]) for info in pair_pool],
        "triple_pool": [str(info["feature"]) for info in triple_pool],
        "feature_diagnostics": [
            {key: value for key, value in info.items() if key != "risk"} for info in infos
        ],
        "rank": list(rank),
        "score": score,
        "supported": supported,
        "reject": reject,
    }


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("NEXT67 JSON must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _formula_features(path: Path) -> list[str]:
    formula = _read_json(path)
    terms = formula.get("terms")
    if not isinstance(terms, list):
        raise ValueError("NEXT67 anchor formula differs")
    return [str(term["feature"]) for term in terms if isinstance(term, Mapping)]


def run_monotone_expanded_search(
    *,
    feature_dir: Path,
    endpoint_firewall_manifest_path: Path,
    discovery_dir: Path,
    design_path: Path,
    next61_formula_path: Path,
    next64_formula_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    feature_dir = Path(feature_dir).resolve()
    discovery_dir = Path(discovery_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "features": feature_dir / SOURCE_FEATURES_NAME,
        "feature_manifest": feature_dir / MANIFEST_NAME,
        "endpoint_firewall": Path(endpoint_firewall_manifest_path).resolve(),
        "discovery_labels": discovery_dir / ROLE_LABELS_NAME,
        "discovery_manifest": discovery_dir / ROLE_MANIFEST_NAME,
        "design": Path(design_path).resolve(),
        "next61_formula": Path(next61_formula_path).resolve(),
        "next64_formula": Path(next64_formula_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT67 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    expected = {
        "feature_manifest": EXPECTED_FEATURE_MANIFEST_SHA256,
        "endpoint_firewall": EXPECTED_ENDPOINT_FIREWALL_SHA256,
        "discovery_manifest": EXPECTED_DISCOVERY_MANIFEST_SHA256,
        "discovery_labels": EXPECTED_DISCOVERY_LABEL_SHA256,
        "design": DESIGN_SHA256,
        "next61_formula": EXPECTED_NEXT61_FORMULA_SHA256,
        "next64_formula": EXPECTED_NEXT64_FORMULA_SHA256,
    }
    if any(hashes[name] != digest for name, digest in expected.items()):
        raise ValueError("NEXT67 frozen input hash differs")
    feature_manifest = _read_json(paths["feature_manifest"])
    endpoint_firewall = _read_json(paths["endpoint_firewall"])
    discovery_manifest = _read_json(paths["discovery_manifest"])
    feature_outputs = feature_manifest.get("outputs_sha256")
    discovery_outputs = discovery_manifest.get("outputs_sha256")
    if (
        feature_manifest.get("protocol") != SOURCE_FEATURE_PROTOCOL
        or feature_manifest.get("labels_opened") is not False
        or not isinstance(feature_outputs, Mapping)
        or feature_outputs.get(SOURCE_FEATURES_NAME) != hashes["features"]
        or endpoint_firewall.get("protocol") != ENDPOINT_PROTOCOL
        or endpoint_firewall.get("internal_validation_endpoint_values_summarized_or_inspected") is not False
        or endpoint_firewall.get("internal_replication_endpoint_values_summarized_or_inspected") is not False
        or discovery_manifest.get("protocol") != ENDPOINT_PROTOCOL
        or discovery_manifest.get("partition_role") != "discovery"
        or not isinstance(discovery_outputs, Mapping)
        or discovery_outputs.get(ROLE_LABELS_NAME) != hashes["discovery_labels"]
    ):
        raise ValueError("NEXT67 discovery-only provenance differs")
    anchors = sorted(
        set(_formula_features(paths["next61_formula"]) + _formula_features(paths["next64_formula"]))
    )
    features_all = pd.read_parquet(paths["features"])
    features = features_all[features_all["partition_role"].eq("discovery")].copy()
    labels = pd.read_parquet(paths["discovery_labels"])
    joined = features.merge(labels, on="material_id", how="inner", validate="one_to_one")
    if len(joined) != len(labels) or set(labels["partition_role"]) != {"discovery"}:
        raise ValueError("NEXT67 discovery identity differs")
    result = search_expanded_additive_rule(
        features=joined,
        endpoint=joined[ENDPOINT_COLUMN].to_numpy(float),
        candidate_features=NEXT65_FEATURE_NAMES,
        anchor_features=anchors,
    )
    formula = {
        **result["selected_formula"],
        "protocol": PROTOCOL,
        "training_partition": "ODAC23 official train / robust discovery only",
        "endpoint_definition": {
            "column": ENDPOINT_COLUMN,
            "protected_max_angstrom": PROTECTED_MAX,
            "severe_min_angstrom": SEVERE_MIN,
            "minimum_adsorbate_configurations": 4,
            "common_translation_removed": True,
        },
        "gates": GATES,
        "candidate_feature_count": len(NEXT65_FEATURE_NAMES),
        "anchor_features": anchors,
        "feature_artifact_sha256": hashes["features"],
        "scientific_status": "advance_to_internal_validation"
        if result["passes_discovery_gates"]
        else "discovery_failure_diagnostic_only",
    }
    search_record = {
        key: value
        for key, value in result.items()
        if key not in {"score", "supported", "reject", "selected_formula"}
    }
    endpoint = joined[ENDPOINT_COLUMN].to_numpy(float)
    predictions = pd.DataFrame(
        {
            "material_id": joined["material_id"].astype(str),
            "partition_role": "discovery",
            ENDPOINT_COLUMN: endpoint,
            "protected": endpoint <= PROTECTED_MAX,
            "severe": endpoint >= SEVERE_MIN,
            "risk_score": result["score"],
            "supported": result["supported"],
            "reject": result["reject"],
        }
    )
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "monotone_expanded_robust_discovery_search",
        "robust_discovery_labels_opened": True,
        "internal_validation_labels_opened": False,
        "internal_replication_labels_opened": False,
        "official_validation_or_test_payload_deserialized": False,
        "dft_values_used_by_executable_formula": False,
        "relaxed_coordinates_used_by_executable_formula": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "passes_discovery_gates": result["passes_discovery_gates"],
        "counts": {
            "rows": len(joined),
            "protected": int(predictions["protected"].sum()),
            "severe": int(predictions["severe"].sum()),
            "supported": int(predictions["supported"].sum()),
            "rejected": int(predictions["reject"].sum()),
            "candidate_features": len(NEXT65_FEATURE_NAMES),
            "candidate_formulas": int(result["candidate_count"]),
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next67_odac23_monotone_expanded_search.py": source_hash
        },
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        formula_path = staging / FORMULA_NAME
        search_path = staging / SEARCH_NAME
        predictions_path = staging / PREDICTIONS_NAME
        formula_path.write_bytes(_json_bytes(formula))
        search_path.write_bytes(_json_bytes(search_record))
        predictions.to_parquet(predictions_path, index=False)
        manifest["outputs_sha256"] = {
            path.name: _sha256(path)
            for path in (formula_path, search_path, predictions_path)
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT67 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT67 input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--endpoint-firewall", type=Path, required=True)
    parser.add_argument("--discovery-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--next61-formula", type=Path, required=True)
    parser.add_argument("--next64-formula", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = run_monotone_expanded_search(
        feature_dir=args.feature_dir,
        endpoint_firewall_manifest_path=args.endpoint_firewall,
        discovery_dir=args.discovery_dir,
        design_path=args.design,
        next61_formula_path=args.next61_formula,
        next64_formula_path=args.next64_formula,
        output_dir=args.output_dir,
    )
    print(json.dumps({"passes": manifest["passes_discovery_gates"], **manifest["counts"]}, indent=2, sort_keys=True))


__all__ = ["PROTOCOL", "run_monotone_expanded_search", "search_expanded_additive_rule"]


if __name__ == "__main__":
    main()
