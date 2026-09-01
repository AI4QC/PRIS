#!/usr/bin/env python3
"""Frozen search with light attenuation of the dominant capped family."""

from __future__ import annotations

import argparse
import hashlib
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

import src.next130_coordination_protection_search as n130
import src.next132_extended_coordination_protection_search as n132
import src.next133_compactness_protection as n133
import src.next135_conjunctive_compactness_search as n135
import src.next161_family_dominance_attenuation_audit as n161
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next87_scigen_sparse_law_search import _term_risk


PROTOCOL = "2026-08-08-next162-family-dominance-attenuation-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT162_FAMILY_DOMINANCE_ATTENUATION_CATALOGUE.json"
EVALUATION_NAME = "NEXT162_DISCOVERY_EVALUATION.json"
SEARCH_NAME = "next162_family_dominance_attenuation_candidate_search.parquet"
EXPECTED_DESIGN_SHA256 = "b0c2d4bff3669a380160c9d481619f317c3c0c41c8e151966ba21353f9a40105"
EXPECTED_BASE_COUNT = 11
EXPECTED_CANDIDATE_COUNT = 176
EXPECTED_BASE_FORMULA_SHA256 = "d1f8763331cbe36f54e898e4efc88d0f88d2ae5d6284883acc4850e58d9678b5"
EXPECTED_CANDIDATE_KEY_SHA256 = "f5ad03d87fa11a06aee0c0aec07eb8a70848353497f8305f9003e5056d7823aa"
AGGREGATION = "family_capmean_dominance_attenuation"
CONTRIBUTION_CAP = 0.5
DOMINANT_FAMILY_ATTENUATION = 0.1
FAMILY_PREFIXES = n161.FAMILY_PREFIXES
COORDINATION_WEIGHTS = (0.0, 0.5, 1.0, 2.0)
PACKING_WEIGHTS = (0.0, 0.1, 0.25, 0.5)
SEARCH_WORKERS = 4
COORDINATION_TERM_ID = n130.PROTECTION_TERM_ID
PACKING_TERM_ID = n135.PACKING_PRODUCT_TERM_ID
EXPECTED_INPUT_SHA256 = {
    **n135.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next161_manifest": "f3521475f86c1ceb656256490a2c386311c686f0df54fc5e80292a708205d542",
    "next161_audit": "86b7e1d3c7e74573edc1740c36a4fc0d328b37734e1ea6f5ef82fc3b24882162",
    "next161_table": "6c3750e1c9ac99cd93fba4d5ff1080348451c47bf5416a21d3d240645e484a1e",
}


def family_dominance_attenuated_sum(
    contributions: object, term_ids: Sequence[str]
) -> np.ndarray:
    """Cap contributions, average within each family, then sum families."""

    values = np.asarray(contributions, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] < 4
        or len(term_ids) != values.shape[1]
        or np.any(~np.isfinite(values))
        or np.any(values < -1.0e-12)
    ):
        raise ValueError("NEXT162 contribution matrix differs")
    values = np.maximum(values, 0.0)
    members: dict[str, list[int]] = {family: [] for family in FAMILY_PREFIXES}
    for index, term_id in enumerate(term_ids):
        matches = [
            family
            for family, prefixes in FAMILY_PREFIXES.items()
            if str(term_id).startswith(prefixes)
        ]
        if len(matches) != 1:
            raise ValueError("NEXT162 term-to-family assignment differs")
        members[matches[0]].append(index)
    if any(not indices for indices in members.values()):
        raise ValueError("NEXT162 family coverage differs")
    capped = np.minimum(values, CONTRIBUTION_CAP)
    family_means = np.column_stack(
        [capped[:, indices].mean(axis=1) for indices in members.values()]
    )
    return family_means.sum(axis=1) - (
        DOMINANT_FAMILY_ATTENUATION * family_means.max(axis=1)
    )


def compose_family_dominance_attenuation_score(
    *,
    contributions: object,
    term_support: object,
    term_ids: Sequence[str],
    coordination_protection: object,
    coordination_active: object,
    coordination_weight: float,
    packing_protection: object,
    packing_active: object,
    packing_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cap each physical contribution, then subtract active protections."""

    values = np.asarray(contributions, dtype=float)
    supports = np.asarray(term_support, dtype=bool)
    coordination = np.asarray(coordination_protection, dtype=float)
    coordination_on = np.asarray(coordination_active, dtype=bool)
    packing = np.asarray(packing_protection, dtype=float)
    packing_on = np.asarray(packing_active, dtype=bool)
    alpha = float(coordination_weight)
    beta = float(packing_weight)
    if (
        values.ndim != 2
        or values.shape[1] < 4
        or len(term_ids) != values.shape[1]
        or supports.shape != values.shape
        or coordination.shape != (len(values),)
        or coordination_on.shape != (len(values),)
        or packing.shape != (len(values),)
        or packing_on.shape != (len(values),)
        or not math.isfinite(alpha)
        or alpha < 0.0
        or not math.isfinite(beta)
        or beta < 0.0
    ):
        raise ValueError("NEXT162 score arrays differ")
    supported = supports.all(axis=1)
    if (
        np.any(supported[:, None] & (~np.isfinite(values) | (values < -1.0e-12)))
        or np.any(coordination_on & (~np.isfinite(coordination) | (coordination < -1.0e-12)))
        or np.any(packing_on & (~np.isfinite(packing) | (packing < -1.0e-12)))
    ):
        raise ValueError("NEXT162 supported score values differ")
    score = np.full(len(values), np.nan, dtype=float)
    if supported.any():
        score[supported] = family_dominance_attenuated_sum(
            values[supported], term_ids
        )
    active = supported & coordination_on
    score[active] -= alpha * coordination[active]
    active = supported & packing_on
    score[active] -= beta * packing[active]
    score[supported] = np.maximum(0.0, score[supported])
    return score, supported


def build_candidate_specs(
    *, bases: pd.DataFrame, physical_term_ids: set[str]
) -> list[dict[str, object]]:
    """Enumerate the frozen 11 x 4 x 4 family-consensus universe."""

    if {"term_ids_json", "weights_json"} - set(bases.columns):
        raise ValueError("NEXT162 base schema differs")
    specs: dict[str, dict[str, object]] = {}
    for _, row in bases.iterrows():
        term_ids = [str(value) for value in json.loads(str(row["term_ids_json"]))]
        weights = [float(value) for value in json.loads(str(row["weights_json"]))]
        if (
            len(term_ids) < 4
            or len(term_ids) != len(weights)
            or len(set(term_ids)) != len(term_ids)
            or any(term_id not in physical_term_ids for term_id in term_ids)
            or any(not math.isfinite(weight) or weight <= 0.0 for weight in weights)
        ):
            raise ValueError("NEXT162 physical base formula differs")
        family_members = {
            family: [
                term_id
                for term_id in term_ids
                if term_id.startswith(prefixes)
            ]
            for family, prefixes in FAMILY_PREFIXES.items()
        }
        if (
            any(not members for members in family_members.values())
            or sum(len(members) for members in family_members.values())
            != len(term_ids)
        ):
            raise ValueError("NEXT162 frozen family membership differs")
        for coordination_weight in COORDINATION_WEIGHTS:
            for packing_weight in PACKING_WEIGHTS:
                payload = {
                    "aggregation": AGGREGATION,
                    "contribution_cap": CONTRIBUTION_CAP,
                    "dominant_family_attenuation": DOMINANT_FAMILY_ATTENUATION,
                    "family_prefixes": {
                        family: list(prefixes)
                        for family, prefixes in FAMILY_PREFIXES.items()
                    },
                    "base_term_ids": term_ids,
                    "base_weights": weights,
                    "coordination_protection_term_id": (
                        None if coordination_weight == 0.0 else COORDINATION_TERM_ID
                    ),
                    "coordination_protection_weight": coordination_weight,
                    "conjunctive_term_ids": (
                        [] if packing_weight == 0.0 else [PACKING_TERM_ID]
                    ),
                    "conjunctive_weights": (
                        [] if packing_weight == 0.0 else [packing_weight]
                    ),
                }
                key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
                specs[key] = {"candidate_key": key, **payload}
    return [specs[key] for key in sorted(specs)]


def materialize_candidates(
    *,
    features: pd.DataFrame,
    physical_terms: Sequence[Mapping[str, object]],
    specs: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    """Encode each exact family-consensus score as an evaluator-compatible term."""

    physical_by_id = {str(term["term_id"]): dict(term) for term in physical_terms}
    if len(physical_by_id) != len(physical_terms):
        raise ValueError("NEXT162 physical term identities are duplicated")
    if {
        n130.n129.FEATURE_NAME,
        n130.n129.SUPPORT_COLUMN,
        n135.PACKING_PRODUCT_FEATURE,
        n135.PACKING_PRODUCT_SUPPORT,
    } - set(features.columns):
        raise ValueError("NEXT162 protection feature schema differs")
    coordination = pd.to_numeric(
        features[n130.n129.FEATURE_NAME], errors="coerce"
    ).to_numpy(float)
    coordination_active = features[n130.n129.SUPPORT_COLUMN].eq(True).to_numpy()
    packing = pd.to_numeric(
        features[n135.PACKING_PRODUCT_FEATURE], errors="coerce"
    ).to_numpy(float)
    packing_active = features[n135.PACKING_PRODUCT_SUPPORT].eq(True).to_numpy()
    risk_cache = {
        term_id: _term_risk(features, term)
        for term_id, term in physical_by_id.items()
    }
    columns: dict[str, np.ndarray] = {}
    virtual_terms: list[dict[str, object]] = []
    runtime: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_spec in specs:
        spec = dict(raw_spec)
        key = str(spec["candidate_key"])
        term_ids = [str(value) for value in spec["base_term_ids"]]
        weights = [float(value) for value in spec["base_weights"]]
        if (
            key in seen
            or spec.get("aggregation") != AGGREGATION
            or float(spec.get("contribution_cap", -1.0)) != CONTRIBUTION_CAP
            or float(spec.get("dominant_family_attenuation", -1.0))
            != DOMINANT_FAMILY_ATTENUATION
            or spec.get("family_prefixes")
            != {
                family: list(prefixes)
                for family, prefixes in FAMILY_PREFIXES.items()
            }
            or len(term_ids) != len(weights)
            or any(term_id not in risk_cache for term_id in term_ids)
        ):
            raise ValueError("NEXT162 candidate materialization differs")
        seen.add(key)
        contribution_columns = []
        support_columns = []
        for term_id, weight in zip(term_ids, weights, strict=True):
            risk, supported = risk_cache[term_id]
            contribution_columns.append(weight * risk)
            support_columns.append(supported)
        score, supported = compose_family_dominance_attenuation_score(
            contributions=np.column_stack(contribution_columns),
            term_support=np.column_stack(support_columns),
            term_ids=term_ids,
            coordination_protection=coordination,
            coordination_active=coordination_active,
            coordination_weight=float(spec["coordination_protection_weight"]),
            packing_protection=packing,
            packing_active=packing_active,
            packing_weight=(
                0.0
                if not spec["conjunctive_weights"]
                else float(spec["conjunctive_weights"][0])
            ),
        )
        maximum = float(np.max(score[supported])) if supported.any() else 0.0
        divisor = max(1.0, maximum / 50.0)
        encoded = np.full(len(features), np.nan, dtype=float)
        encoded[supported] = np.sinh(score[supported] / divisor)
        if not np.isfinite(encoded[supported]).all():
            raise ValueError("NEXT162 virtual score encoding overflowed")
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        virtual_id = f"next162_virtual_candidate__{digest}"
        feature_name = f"_{virtual_id}_value"
        columns[feature_name] = encoded
        virtual_terms.append(
            {
                "term_id": virtual_id,
                "feature": feature_name,
                "direction": 1,
                "transform": "asinh",
                "center": 0.0,
                "scale": 1.0 / divisor,
                "group": "next162_family_dominance_attenuation_candidate",
                "encoding": "asinh_sinh_exact_family_dominance_attenuation_score",
                "physical_candidate_key": key,
            }
        )
        runtime.append(
            {
                "candidate_key": key,
                "base_term_ids": [virtual_id],
                "base_weights": [1.0],
                "optional_term_id": None,
                "optional_weight": 0.0,
            }
        )
    return (
        pd.concat([features.reset_index(drop=True), pd.DataFrame(columns)], axis=1),
        virtual_terms,
        runtime,
    )


def _paths(
    roots: Mapping[str, Path], freeze_path: Path, design_path: Path
) -> dict[str, Path]:
    paths = n135._paths(roots, freeze_path)
    paths.update(
        {
            "design": design_path,
            "next161_manifest": roots["next161"] / n161.MANIFEST_NAME,
            "next161_audit": roots["next161"] / n161.AUDIT_NAME,
            "next161_table": roots["next161"] / n161.TABLE_NAME,
        }
    )
    return paths


def run_family_dominance_attenuation_search(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    next98_dir: Path,
    next110_dir: Path,
    next111_dir: Path,
    next113_dir: Path,
    next114_dir: Path,
    next116_dir: Path,
    next117_dir: Path,
    next120_dir: Path,
    next121_dir: Path,
    next122_dir: Path,
    next124_dir: Path,
    next125_dir: Path,
    next129_dir: Path,
    next130_dir: Path,
    next133_dir: Path,
    next134_dir: Path,
    next161_dir: Path,
    next135_freeze_path: Path,
    design_path: Path,
    output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{
            f"next{stage}": Path(value).resolve()
            for stage, value in (
                (98, next98_dir),
                (110, next110_dir),
                (111, next111_dir),
                (113, next113_dir),
                (114, next114_dir),
                (116, next116_dir),
                (117, next117_dir),
                (120, next120_dir),
                (121, next121_dir),
                (122, next122_dir),
                (124, next124_dir),
                (125, next125_dir),
                (129, next129_dir),
                (130, next130_dir),
                (133, next133_dir),
                (134, next134_dir),
                (161, next161_dir),
            )
        },
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots, Path(next135_freeze_path).resolve(), Path(design_path).resolve()
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT162 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            key
            for key in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(key) != EXPECTED_INPUT_SHA256.get(key)
        )
        raise ValueError(f"NEXT162 formal input identity differs: {differing}")
    manifest161 = json.loads(paths["next161_manifest"].read_text())
    audit161 = json.loads(paths["next161_audit"].read_text())
    if (
        manifest161.get("protocol") != n161.PROTOCOL
        or manifest161.get("eligible_statistic_count") != 4
        or manifest161.get("opened_validation_outputs_used") is not False
        or manifest161.get("dft_values_used_by_executable_formula") is not False
        or manifest161.get("outputs_sha256", {}).get(n161.AUDIT_NAME)
        != input_hashes["next161_audit"]
        or audit161.get("eligible_statistics")
        != [
            "family_capmean_attenuation_0p1",
            "family_capmean_attenuation_0p25",
            "family_capmean_attenuation_0p5",
            "family_capmean_attenuation_0p75",
        ]
        or audit161.get("selected_statistic", {}).get("statistic")
        != "family_capmean_attenuation_0p1"
        or audit161.get("validation_or_replication_opened") is not False
        or audit161.get("dft_values_used_by_executable_formula") is not False
    ):
        raise ValueError("NEXT162 attenuation-audit provenance differs")

    extended, feature_tables, old_terms, mhcr_terms = n130._join_label_free_features(paths)
    compact_frames = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next133_{source}_features"]).copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        compact_frames.append(table)
    extended = extended.merge(
        pd.concat(compact_frames, ignore_index=True),
        on="material_id",
        how="inner",
        validate="one_to_one",
    )
    conjunctive = n135.materialize_conjunctive_features(extended)
    extended = pd.concat(
        [extended.reset_index(drop=True), conjunctive.reset_index(drop=True)], axis=1
    )
    physical_terms = [*old_terms, *mhcr_terms]
    physical_ids = {str(term["term_id"]) for term in physical_terms}
    all_bases = n130.n127.select_next125_bases(
        pd.read_parquet(paths["next125_search_records"])
    )
    bases = n132.select_extended_bases(
        pd.read_parquet(paths["next130_search_records"]), all_bases
    )
    specs = build_candidate_specs(bases=bases, physical_term_ids=physical_ids)
    base_formulas = sorted(
        n130.n127._formula_identity(
            json.loads(str(row["term_ids_json"])),
            json.loads(str(row["weights_json"])),
        )
        for _, row in bases.iterrows()
    )
    base_formula_sha = hashlib.sha256("\n".join(base_formulas).encode()).hexdigest()
    candidate_sha = hashlib.sha256(
        "\n".join(str(spec["candidate_key"]) for spec in specs).encode()
    ).hexdigest()
    if require_formal_inputs and (
        len(bases) != EXPECTED_BASE_COUNT
        or len(specs) != EXPECTED_CANDIDATE_COUNT
        or base_formula_sha != EXPECTED_BASE_FORMULA_SHA256
        or candidate_sha != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT162 frozen candidate universe differs")

    scigen_endpoint = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_endpoint = pd.read_parquet(paths["wyformer_endpoint"])
    endpoint_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "material_id": "scigen:"
                    + scigen_endpoint["material_id"].astype(str),
                    "_endpoint": pd.to_numeric(
                        scigen_endpoint["distortion_ratio"], errors="coerce"
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "material_id": "wyformer:"
                    + wyformer_endpoint["material_id"].astype(str),
                    "_endpoint": n130.n125.n121.prior._endpoint_numeric(
                        wyformer_endpoint["endpoint_stratum"]
                    ),
                }
            ),
        ],
        ignore_index=True,
    )
    combined = extended.merge(
        endpoint_frame, on="material_id", how="inner", validate="one_to_one"
    )
    endpoint = pd.to_numeric(combined.pop("_endpoint"), errors="coerce").to_numpy(float)
    combined, virtual_terms, runtime = materialize_candidates(
        features=combined, physical_terms=physical_terms, specs=specs
    )
    started = time.perf_counter()
    result = n130.n125.search_optional_guard_laws_parallel(
        features=combined,
        endpoint=endpoint,
        old_terms=virtual_terms,
        optional_terms=[],
        candidate_specs=runtime,
        workers=search_workers,
    )
    elapsed = time.perf_counter() - started

    def decorate(record: dict[str, object]) -> None:
        payload = json.loads(str(record["candidate_key"]))
        evaluated = json.loads(str(record["base_term_ids_json"]))
        record["evaluation_virtual_term_id"] = str(evaluated[0])
        record["aggregation"] = AGGREGATION
        record["base_term_ids_json"] = json.dumps(
            payload["base_term_ids"], separators=(",", ":")
        )
        record["base_weights_json"] = json.dumps(
            payload["base_weights"], separators=(",", ":")
        )
        record["coordination_protection_weight"] = float(
            payload["coordination_protection_weight"]
        )
        packing_weight = (
            0.0
            if not payload["conjunctive_weights"]
            else float(payload["conjunctive_weights"][0])
        )
        record["packing_protection_weight"] = packing_weight
        record["family_prefixes_json"] = json.dumps(
            payload["family_prefixes"], sort_keys=True, separators=(",", ":")
        )
        record["score_composition"] = (
            "max(0,sum_family(capmean)-0.1*max_family(capmean)"
            "-alpha*coordination-beta*coordination_covalent_packing)"
        )

    for record in result["candidate_records"]:
        decorate(record)
    selected = result["selected"]
    if "evaluation_virtual_term_id" not in selected["record"]:
        decorate(selected["record"])
    payload = json.loads(str(selected["record"]["candidate_key"]))
    physical_by_id = {str(term["term_id"]): dict(term) for term in physical_terms}
    formula = selected["formula"]
    formula["evaluation_virtual_term_id"] = str(formula["base_terms"][0]["term_id"])
    formula["base_terms"] = [
        {**physical_by_id[str(term_id)], "weight": float(weight)}
        for term_id, weight in zip(
            payload["base_term_ids"], payload["base_weights"], strict=True
        )
    ]
    formula["aggregation"] = AGGREGATION
    formula["contribution_cap"] = CONTRIBUTION_CAP
    formula["dominant_family_attenuation"] = DOMINANT_FAMILY_ATTENUATION
    formula["family_prefixes"] = payload["family_prefixes"]
    formula["within_family_reduction"] = "arithmetic_mean"
    formula["across_family_reduction"] = "sum"
    formula["coordination_protection"] = {
        "term_id": COORDINATION_TERM_ID,
        "feature": n130.n129.FEATURE_NAME,
        "weight": float(payload["coordination_protection_weight"]),
        "missing_policy": "TERM_OFF_KEEP_BASE",
    }
    formula["conjunctive_protection"] = {
        "term_id": PACKING_TERM_ID,
        "feature": n135.PACKING_PRODUCT_FEATURE,
        "weight": float(selected["record"]["packing_protection_weight"]),
        "missing_policy": "TERM_OFF_KEEP_BASE",
    }
    formula["score_composition"] = selected["record"]["score_composition"]
    formula["kind"] = (
        "family_dominance_attenuated_physical_base_with_bounded_protections"
    )
    passes = bool(selected["record"]["passes_all_discovery_gates"])
    records_frame = pd.DataFrame(result["candidate_records"])
    counts = {}
    for (alpha, beta), frame in records_frame.groupby(
        ["coordination_protection_weight", "packing_protection_weight"], sort=True
    ):
        counts[f"alpha={alpha:g},beta={beta:g}"] = {
            "candidates": int(len(frame)),
            "passes_source_auc_gates": int(frame["passes_source_auc_gates"].sum()),
            "passes_safe_all_cells": int(frame["passes_safe_all_cells"].sum()),
            "passes_broad_all_cells": int(frame["passes_broad_all_cells"].sum()),
            "passes_all_discovery_gates": int(
                frame["passes_all_discovery_gates"].sum()
            ),
        }
    catalogue = {
        "protocol": PROTOCOL,
        "design_sha256": input_hashes["design"],
        "next161_selected_statistic": "family_capmean_attenuation_0p1",
        "contribution_cap": CONTRIBUTION_CAP,
        "dominant_family_attenuation": DOMINANT_FAMILY_ATTENUATION,
        "family_prefixes": {
            family: list(prefixes) for family, prefixes in FAMILY_PREFIXES.items()
        },
        "within_family_reduction": "arithmetic_mean",
        "across_family_reduction": "sum",
        "base_count": len(bases),
        "candidate_count": len(specs),
        "coordination_weight_grid": list(COORDINATION_WEIGHTS),
        "packing_weight_grid": list(PACKING_WEIGHTS),
        "base_formula_sha256": base_formula_sha,
        "candidate_key_sha256": candidate_sha,
        "active_score": (
            "max(0,sum_family(capmean)-0.1*max_family(capmean)"
            "-alpha*coordination-beta*coordination_covalent_packing)"
        ),
        "base_support_unchanged": True,
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    catalogue_sha = hashlib.sha256(
        json.dumps(catalogue, indent=2, sort_keys=True).encode() + b"\n"
    ).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    source_paths = {
        "src/next130_coordination_protection_search.py": Path(n130.__file__).resolve(),
        "src/next135_conjunctive_compactness_search.py": Path(n135.__file__).resolve(),
        "src/next161_family_dominance_attenuation_audit.py": Path(n161.__file__).resolve(),
        "src/next162_family_dominance_attenuation_search.py": Path(__file__).resolve(),
    }
    source_hashes = {
        name: _sha256_file(path) for name, path in source_paths.items()
    }
    output_paths: list[Path] = []
    try:
        catalogue_path = staging / CATALOGUE_NAME
        evaluation_path = staging / EVALUATION_NAME
        search_path = staging / SEARCH_NAME
        _write_json(
            catalogue_path,
            {**catalogue, "label_free_catalogue_sha256": catalogue_sha},
        )
        _write_json(
            evaluation_path,
            {
                "protocol": PROTOCOL,
                "evaluation_mode": "fixed_family_dominance_attenuation_search",
                "rows": {
                    "scigen": int(len(feature_tables["scigen"])),
                    "wyformer": int(len(feature_tables["wyformer"])),
                    "total": int(len(combined)),
                },
                "base_count": len(bases),
                "candidate_count": int(result["candidate_count"]),
                "elapsed_seconds": elapsed,
                "search_workers": search_workers,
                "counts_by_protection_weights": counts,
                "safe_gates": dict(n130.n125.n121.prior.DEFAULT_GATES),
                "source_auc_gates": dict(n130.n125.n121.prior.AUC_GATES),
                "broad_min_severe_precision_lower": n130.n125.n121.prior.BROAD_MIN_PRECISION_LOWER,
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
                "requires_unopened_internal_validation_before_claim": True,
            },
        )
        records_frame.to_parquet(search_path, index=False)
        output_paths.extend([catalogue_path, evaluation_path, search_path])
        manifest = {
            "protocol": PROTOCOL,
            "label_free_catalogue_sha256": catalogue_sha,
            "base_count": len(bases),
            "candidate_count": int(result["candidate_count"]),
            "search_workers": search_workers,
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "requires_unopened_internal_validation_before_claim": True,
            "family_dominance_attenuation_branch_terminated": not passes,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True,
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
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {
                path.name: _sha256_file(path) for path in output_paths
            },
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT162 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT162 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scigen-feature-dir", type=Path, required=True)
    parser.add_argument("--scigen-discovery-endpoint-dir", type=Path, required=True)
    parser.add_argument("--wyformer-feature-dir", type=Path, required=True)
    parser.add_argument("--wyformer-discovery-endpoint-dir", type=Path, required=True)
    for stage in (
        98,
        110,
        111,
        113,
        114,
        116,
        117,
        120,
        121,
        122,
        124,
        125,
        129,
        130,
        133,
        134,
        161,
    ):
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_family_dominance_attenuation_search(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{
            f"next{stage}_dir": getattr(args, f"next{stage}_dir")
            for stage in (
                98,
                110,
                111,
                113,
                114,
                116,
                117,
                120,
                121,
                122,
                124,
                125,
                129,
                130,
                133,
                134,
                161,
            )
        },
        next135_freeze_path=args.next135_freeze_path,
        design_path=args.design_path,
        output_dir=args.output_dir,
        search_workers=args.search_workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DOMINANT_FAMILY_ATTENUATION",
    "EXPECTED_CANDIDATE_COUNT",
    "EXPECTED_CANDIDATE_KEY_SHA256",
    "FAMILY_PREFIXES",
    "build_candidate_specs",
    "compose_family_dominance_attenuation_score",
    "family_dominance_attenuated_sum",
    "run_family_dominance_attenuation_search",
]
