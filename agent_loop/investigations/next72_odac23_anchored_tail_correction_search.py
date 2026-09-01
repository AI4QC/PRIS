#!/usr/bin/env python3
"""Finite one/two-term tail corrections around the sealed NEXT67 x0 axis."""

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
)
from src.next60_odac23_robust_scaffold_endpoint import (
    ENDPOINT_COLUMN,
    PROTOCOL as ENDPOINT_PROTOCOL,
    ROLE_LABELS_NAME,
    ROLE_MANIFEST_NAME,
)
from src.next65_odac23_physics_couplings import NEXT65_FEATURE_NAMES
from src.next67_odac23_monotone_expanded_search import PROTOCOL as ANCHOR_PROTOCOL
from src.next68_odac23_sparse_stable_law import apply_sparse_formula
from src.next70_odac23_metal_donor_bond_valence_features import (
    FEATURES_NAME as SOURCE_FEATURES_NAME,
    METAL_DONOR_BV_FEATURE_NAMES,
    PROTOCOL as SOURCE_FEATURE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next72-odac23-anchored-tail-correction-search-v1"
DESIGN_SHA256 = "188230c2cf56f2f06915f21cfd893d71bcb68167244a2dc20962d4df1e54e274"
EXPECTED_FEATURE_MANIFEST_SHA256 = (
    "928a0bbfa1120e2c92bac2e9d3f0046a1d440c24beb72f652e477eb827874f14"
)
EXPECTED_FEATURE_SHA256 = (
    "d3684af21c70e3be18ae4aed8dd9a505209cfb2d91e9639911aae72da77ca6dc"
)
EXPECTED_ANCHOR_SHA256 = (
    "9ea83c8c1f70acf619c8e9f163f4468de316972c33b71605fa75260ce013ed58"
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
CANDIDATE_FEATURE_NAMES = tuple(NEXT65_FEATURE_NAMES) + tuple(METAL_DONOR_BV_FEATURE_NAMES)
SINGLE_WEIGHTS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
PAIR_WEIGHTS = (0.25, 0.5, 1.0, 2.0, 4.0)
PAIR_SHORTLIST = 24
FORMULA_NAME = "NEXT72_ODAC23_ANCHORED_TAIL_CANDIDATE.json"
SEARCH_NAME = "NEXT72_ODAC23_ANCHORED_TAIL_SEARCH.json"
PREDICTIONS_NAME = "next72_odac23_anchored_tail_predictions.parquet"
MANIFEST_NAME = "MANIFEST.json"


def _strata(features: pd.DataFrame) -> np.ndarray:
    defective = np.asarray(features["defective"], dtype=bool)
    oms = np.asarray(features["open_metal_site"], dtype=bool)
    return np.asarray(
        [f"defective={int(left)}|oms={int(right)}" for left, right in zip(defective, oms, strict=True)],
        dtype=str,
    )


def _executable_formula(terms: list[dict[str, object]], threshold: float = 0.0) -> dict[str, object]:
    return {
        "kind": "additive",
        "terms": terms,
        "threshold": float(threshold),
        "missing_policy": "KEEP",
        "domain_gate": dict(DOMAIN_GATE),
    }


def search_anchored_tail_correction(
    *,
    features: pd.DataFrame,
    endpoint: Sequence[float],
    anchor_formula: Mapping[str, object],
    candidate_features: Sequence[str] = CANDIDATE_FEATURE_NAMES,
    single_weights: Sequence[float] = SINGLE_WEIGHTS,
    pair_weights: Sequence[float] = PAIR_WEIGHTS,
    pair_shortlist: int = PAIR_SHORTLIST,
) -> dict[str, object]:
    """Exhaust one/two-term standardized corrections around frozen anchor terms."""

    endpoint = np.asarray(endpoint, dtype=float)
    required = {
        "combined_supported",
        "periodic_dimension_max",
        "periodic_framework_fraction",
        "defective",
        "open_metal_site",
    }
    anchor_terms_raw = anchor_formula.get("terms")
    if (
        len(features) != len(endpoint)
        or not required.issubset(features.columns)
        or not np.isfinite(endpoint).all()
        or not isinstance(anchor_terms_raw, list)
        or not 1 <= len(anchor_terms_raw) <= 6
        or not all(isinstance(term, Mapping) for term in anchor_terms_raw)
        or not single_weights
        or not pair_weights
        or type(pair_shortlist) is not int
        or pair_shortlist < 1
        or any(float(value) <= 0.0 for value in (*single_weights, *pair_weights))
    ):
        raise ValueError("NEXT72 frozen catalogue differs")
    anchor_terms = [
        {
            "feature": str(term["feature"]),
            "direction": int(term["direction"]),
            "center": float(term["center"]),
            "scale": float(term["scale"]),
            "weight": float(term["weight"]),
        }
        for term in anchor_terms_raw
    ]
    if len(anchor_terms) + 2 > 8:
        raise ValueError("NEXT72 anchor leaves no correction capacity")
    apply_sparse_formula(features, _executable_formula(anchor_terms))
    anchor_names = {str(term["feature"]) for term in anchor_terms}
    dimension = pd.to_numeric(features["periodic_dimension_max"], errors="coerce").to_numpy(float)
    fraction = pd.to_numeric(features["periodic_framework_fraction"], errors="coerce").to_numpy(float)
    domain = (
        np.asarray(features["combined_supported"], dtype=bool)
        & np.isfinite(dimension)
        & np.isfinite(fraction)
        & (dimension >= DOMAIN_GATE["periodic_dimension_max_min"])
        & (fraction >= DOMAIN_GATE["periodic_framework_fraction_min"])
    )
    infos = []
    for feature in candidate_features:
        if feature in anchor_names or feature not in features or any(
            info["feature"] == feature for info in infos
        ):
            continue
        values = pd.to_numeric(features[feature], errors="coerce").to_numpy(float)
        finite = domain & np.isfinite(values)
        if int(finite.sum()) / max(1, int(domain.sum())) < 0.95:
            continue
        center = float(np.median(values[finite]))
        q25, q75 = np.quantile(values[finite], (0.25, 0.75))
        scale = float(q75 - q25)
        if not math.isfinite(scale) or scale <= 1.0e-12:
            continue
        infos.append({"feature": str(feature), "center": center, "scale": scale})
    if not infos:
        raise ValueError("NEXT72 has no evaluable correction feature")
    strata = _strata(features)
    best = None
    term_keys: set[str] = set()
    candidate_count = 0
    formula_count = 0

    def evaluate_terms(terms: list[dict[str, object]]):
        nonlocal best, candidate_count, formula_count
        term_key = json.dumps(terms, sort_keys=True, separators=(",", ":"))
        if term_key in term_keys:
            return None
        term_keys.add(term_key)
        formula_count += 1
        provisional = _executable_formula(terms)
        raw_score, supported, _reject = apply_sparse_formula(features, provisional)
        aucs = _auc_diagnostics(
            score=raw_score, supported=supported, endpoint=endpoint, strata=strata
        )
        thresholds = {
            float(np.quantile(raw_score[supported], 1.0 - rejection, method="inverted_cdf"))
            for rejection in REJECTION_FRACTIONS
        }
        local_best = None
        for threshold in sorted(thresholds):
            formula = {**provisional, "threshold": threshold}
            score, formula_supported, reject = apply_sparse_formula(features, formula)
            metrics = _decision_metrics(
                supported=formula_supported, reject=reject, endpoint=endpoint
            )
            rank = _gate_rank(metrics, aucs, len(terms))
            key = json.dumps(formula, sort_keys=True, separators=(",", ":"))
            candidate_count += 1
            record = (rank, key, formula, metrics, aucs, score, formula_supported, reject)
            if local_best is None or rank > local_best[0] or (
                rank == local_best[0] and key < local_best[1]
            ):
                local_best = record
            if best is None or rank > best[0] or (rank == best[0] and key < best[1]):
                best = record
        return local_best

    evaluate_terms(anchor_terms)
    guard_records = []
    for info in infos:
        feature_best = None
        feature_best_term = None
        for direction in (-1, 1):
            for weight in single_weights:
                term = {
                    "feature": info["feature"],
                    "direction": direction,
                    "center": info["center"],
                    "scale": info["scale"],
                    "weight": float(weight),
                }
                local = evaluate_terms(anchor_terms + [term])
                if local is not None and (
                    feature_best is None
                    or local[0] > feature_best[0]
                    or (local[0] == feature_best[0] and local[1] < feature_best[1])
                ):
                    feature_best = local
                    feature_best_term = term
        if feature_best is not None and feature_best_term is not None:
            guard_records.append(
                {
                    **info,
                    "best_rank": list(feature_best[0]),
                    "best_term": feature_best_term,
                }
            )
    guard_records.sort(
        key=lambda row: (tuple(-float(value) for value in row["best_rank"]), row["feature"])
    )
    shortlist = guard_records[: min(pair_shortlist, len(guard_records))]
    info_by_name = {str(info["feature"]): info for info in infos}
    for left_record, right_record in combinations(shortlist, 2):
        left = info_by_name[str(left_record["feature"])]
        right = info_by_name[str(right_record["feature"])]
        for left_direction, right_direction in product((-1, 1), repeat=2):
            for left_weight, right_weight in product(pair_weights, repeat=2):
                guard_terms = [
                    {
                        "feature": left["feature"],
                        "direction": left_direction,
                        "center": left["center"],
                        "scale": left["scale"],
                        "weight": float(left_weight),
                    },
                    {
                        "feature": right["feature"],
                        "direction": right_direction,
                        "center": right["center"],
                        "scale": right["scale"],
                        "weight": float(right_weight),
                    },
                ]
                guard_terms.sort(key=lambda term: str(term["feature"]))
                evaluate_terms(anchor_terms + guard_terms)
    if best is None:
        raise RuntimeError("NEXT72 finite correction catalogue is empty")
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
        "candidate_count": candidate_count,
        "unique_term_list_count": formula_count,
        "usable_guard_feature_count": len(infos),
        "pair_shortlist": [str(row["feature"]) for row in shortlist],
        "guard_feature_ranking": guard_records,
        "rank": list(rank),
        "score": score,
        "supported": supported,
        "reject": reject,
    }


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("NEXT72 JSON must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def run_anchored_tail_search(
    *,
    feature_dir: Path,
    endpoint_firewall_manifest_path: Path,
    discovery_dir: Path,
    design_path: Path,
    anchor_formula_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Run discovery-only finite correction search and publish immutably."""

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
        "anchor_formula": Path(anchor_formula_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT72 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    expected = {
        "features": EXPECTED_FEATURE_SHA256,
        "feature_manifest": EXPECTED_FEATURE_MANIFEST_SHA256,
        "endpoint_firewall": EXPECTED_ENDPOINT_FIREWALL_SHA256,
        "discovery_manifest": EXPECTED_DISCOVERY_MANIFEST_SHA256,
        "discovery_labels": EXPECTED_DISCOVERY_LABEL_SHA256,
        "design": DESIGN_SHA256,
        "anchor_formula": EXPECTED_ANCHOR_SHA256,
    }
    if any(hashes[name] != digest for name, digest in expected.items()):
        raise ValueError("NEXT72 frozen input hash differs")
    feature_manifest = _read_json(paths["feature_manifest"])
    endpoint_firewall = _read_json(paths["endpoint_firewall"])
    discovery_manifest = _read_json(paths["discovery_manifest"])
    anchor = _read_json(paths["anchor_formula"])
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
        or anchor.get("protocol") != ANCHOR_PROTOCOL
    ):
        raise ValueError("NEXT72 discovery-only provenance differs")
    features_all = pd.read_parquet(paths["features"])
    features = features_all[features_all["partition_role"].eq("discovery")].copy()
    labels = pd.read_parquet(paths["discovery_labels"])
    if set(labels["partition_role"]) != {"discovery"}:
        raise ValueError("NEXT72 received non-discovery labels")
    joined = features.merge(labels, on="material_id", how="inner", validate="one_to_one")
    if len(joined) != len(labels):
        raise ValueError("NEXT72 discovery identity differs")
    result = search_anchored_tail_correction(
        features=joined,
        endpoint=joined[ENDPOINT_COLUMN].to_numpy(float),
        anchor_formula=anchor,
        candidate_features=CANDIDATE_FEATURE_NAMES,
    )
    formula = {
        **result["selected_formula"],
        "protocol": PROTOCOL,
        "training_partition": "ODAC23 official train / robust discovery only",
        "formula_family": "sealed NEXT67 physical axis plus zero, one, or two finite corrections",
        "anchor_formula_sha256": hashes["anchor_formula"],
        "endpoint_definition": {
            "column": ENDPOINT_COLUMN,
            "protected_max_angstrom": PROTECTED_MAX,
            "severe_min_angstrom": SEVERE_MIN,
            "minimum_adsorbate_configurations": 4,
            "common_translation_removed": True,
        },
        "gates": GATES,
        "candidate_feature_count": len(CANDIDATE_FEATURE_NAMES),
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
        "mode": "finite_anchored_tail_correction_robust_discovery_search",
        "robust_discovery_labels_opened": True,
        "internal_validation_labels_opened": False,
        "internal_replication_labels_opened": False,
        "official_validation_or_test_payload_deserialized": False,
        "dft_calculation_executed": False,
        "dft_values_used_by_executable_formula": False,
        "relaxed_coordinates_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "executable_is_fixed_explicit_weighted_sum": True,
        "passes_discovery_gates": result["passes_discovery_gates"],
        "counts": {
            "rows": len(joined),
            "protected": int(predictions["protected"].sum()),
            "severe": int(predictions["severe"].sum()),
            "supported": int(predictions["supported"].sum()),
            "rejected": int(predictions["reject"].sum()),
            "candidate_features": len(CANDIDATE_FEATURE_NAMES),
            "usable_guard_features": int(result["usable_guard_feature_count"]),
            "candidate_thresholds": int(result["candidate_count"]),
            "unique_term_lists": int(result["unique_term_list_count"]),
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next72_odac23_anchored_tail_correction_search.py": source_hash
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
            path.name: _sha256(path) for path in (formula_path, search_path, predictions_path)
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT72 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT72 input changed before publication")
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
    parser.add_argument("--anchor-formula", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = run_anchored_tail_search(
        feature_dir=args.feature_dir,
        endpoint_firewall_manifest_path=args.endpoint_firewall,
        discovery_dir=args.discovery_dir,
        design_path=args.design,
        anchor_formula_path=args.anchor_formula,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {"passes": manifest["passes_discovery_gates"], **manifest["counts"]},
            indent=2,
            sort_keys=True,
        )
    )


__all__ = ["PROTOCOL", "run_anchored_tail_search", "search_anchored_tail_correction"]


if __name__ == "__main__":
    main()
