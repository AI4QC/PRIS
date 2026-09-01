#!/usr/bin/env python3
"""Frozen subtractive coordination-protection search over NEXT125 AUC+SAFE12 laws."""

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

import src.next125_mhcr_frontier_rescue as n125
import src.next127_hall_profile_persistence_rescue as n127
import src.next129_coordination_protection as n129
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next87_scigen_sparse_law_search import _term_risk


PROTOCOL = "2026-08-08-next130-coordination-protection-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT130_COORDINATION_PROTECTION_SEARCH_CATALOGUE.json"
EVALUATION_NAME = "NEXT130_DISCOVERY_EVALUATION.json"
SEARCH_NAME = "next130_coordination_protection_candidate_search.parquet"
EXPECTED_FREEZE_SHA256 = "0636a9075f50ed4e2239a66d069e68443bd31ffe755897ba43947106632a7028"
EXPECTED_BASE_COUNT = 260
EXPECTED_CANDIDATE_COUNT = 1_560
EXPECTED_BASE_KEY_SHA256 = "a83e10e4c2d6cca3f2ee3c6bb5ba77f3856983cc355259de771b081dcb802f2e"
EXPECTED_BASE_FORMULA_SHA256 = "3139fa905c68fd0321f5922996b14271249ada5c644a9392522f04c4dda4ab95"
EXPECTED_CANDIDATE_KEY_SHA256 = "bad1a9c16c54ecb90ae94fc39deec7da2901c98f0f4c6f038810191f1d012730"
EXPECTED_NEXT125_MANIFEST_SHA256 = "305b1a6044ee43b17a56edd8e7630819955328d35416fa5bd8c178eddf12dac9"
EXPECTED_NEXT129_MANIFEST_SHA256 = "77ca67ae73eb5709577f3b2a172cb55bdba5e7a908cfcec20b424f1828549c21"
PROTECTION_TERM_ID = "coordination_protection__high"
PROTECTION_WEIGHTS = (0.10, 0.25, 0.50, 1.00, 2.00)
SEARCH_WORKERS = 4
BASE_REPRODUCTION_AUC_TOLERANCE = n125.BASE_REPRODUCTION_AUC_TOLERANCE

EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n125.EXPECTED_INPUT_SHA256.items()
        if key != "freeze"
    },
    "next125_manifest": EXPECTED_NEXT125_MANIFEST_SHA256,
    "next125_catalogue": "b4230b22e5362c8678e38053180c1d5d6d2faad6016e11827bf93cb6824fc546",
    "next125_evaluation": "fd22aafb75c45243ae39cb53f35fb98d0e1c869e9ea293886928c4a816c879d6",
    "next125_search_records": "df4b01c18d12cec5f9a7e7ce79c5f292edd494def5b35c53a98374fd26869907",
    "next129_manifest": EXPECTED_NEXT129_MANIFEST_SHA256,
    "next129_catalogue": "347d785fe8b02f324d4019cf8837a6b1b7c9bfae7e5c8f10f37b48356e4337a6",
    "next129_scigen_features": "7d5bd6fe0e71019b64b426a13dadcf38f29889904d9c4feb05c668d3b8bd1392",
    "next129_wyformer_features": "3aa414e3623223014fb0c8fc691cfc8822df1ff2a7424c6b83d9b7da7bc47b05",
    "freeze": EXPECTED_FREEZE_SHA256,
}


def apply_protection_score(
    *,
    base_score: object,
    base_supported: object,
    protection: object,
    protection_active: object,
    protection_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract bounded protection without changing the base applicability domain."""

    base = np.asarray(base_score, dtype=float)
    supported = np.asarray(base_supported, dtype=bool)
    evidence = np.asarray(protection, dtype=float)
    active = np.asarray(protection_active, dtype=bool)
    weight = float(protection_weight)
    if (
        base.ndim != 1
        or supported.shape != base.shape
        or evidence.shape != base.shape
        or active.shape != base.shape
        or not math.isfinite(weight)
        or weight < 0.0
        or np.any(supported & (~np.isfinite(base) | (base < -1.0e-12)))
        or np.any(active & (~np.isfinite(evidence) | (evidence < -1.0e-12)))
    ):
        raise ValueError("NEXT130 protection-score arrays differ")
    result = np.full(len(base), np.nan, dtype=float)
    result[supported] = np.maximum(0.0, base[supported])
    subtract = supported & active
    result[subtract] = np.maximum(
        0.0,
        base[subtract] - weight * evidence[subtract],
    )
    return result, supported.copy()


def build_candidate_specs(
    *, bases: pd.DataFrame, old_term_ids: set[str]
) -> list[dict[str, object]]:
    """Enumerate each frozen base alone and with five subtractive weights."""

    if {"term_ids_json", "weights_json"} - set(bases.columns):
        raise ValueError("NEXT130 candidate base schema differs")
    specs: dict[str, dict[str, object]] = {}
    for _, row in bases.iterrows():
        term_ids = [str(value) for value in json.loads(str(row["term_ids_json"]))]
        weights = [float(value) for value in json.loads(str(row["weights_json"]))]
        if (
            not term_ids
            or len(term_ids) != len(weights)
            or len(set(term_ids)) != len(term_ids)
            or any(term_id not in old_term_ids for term_id in term_ids)
            or any(not math.isfinite(weight) or weight <= 0.0 for weight in weights)
        ):
            raise ValueError("NEXT130 physical base formula differs")
        for protection_term_id, protection_weight in (
            (None, 0.0),
            *((PROTECTION_TERM_ID, weight) for weight in PROTECTION_WEIGHTS),
        ):
            payload = {
                "base_term_ids": term_ids,
                "base_weights": weights,
                "protection_term_id": protection_term_id,
                "protection_weight": protection_weight,
            }
            key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            specs[key] = {"candidate_key": key, **payload}
    return [specs[key] for key in sorted(specs)]


def materialize_protected_candidates(
    *,
    features: pd.DataFrame,
    bases: pd.DataFrame,
    base_virtual_terms: Sequence[Mapping[str, object]],
    base_virtual_by_formula: Mapping[str, str],
    physical_specs: Sequence[Mapping[str, object]],
) -> tuple[
    pd.DataFrame,
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, str],
]:
    """Encode every protected score exactly as one evaluator-compatible term."""

    if {n129.FEATURE_NAME, n129.SUPPORT_COLUMN} - set(features.columns):
        raise ValueError("NEXT130 protection feature schema differs")
    protection = pd.to_numeric(features[n129.FEATURE_NAME], errors="coerce").to_numpy(float)
    protection_active = features[n129.SUPPORT_COLUMN].eq(True).to_numpy()
    if (
        np.any(protection_active & ~np.isfinite(protection))
        or np.any(protection_active & (protection < -1.0e-12))
        or np.any(protection_active & (protection > n129.CLIP_NORMALIZED + 1.0e-12))
    ):
        raise ValueError("NEXT130 protection values differ")
    base_terms_by_id = {
        str(term["term_id"]): dict(term) for term in base_virtual_terms
    }
    if len(base_terms_by_id) != len(base_virtual_terms):
        raise ValueError("NEXT130 virtual base identities are duplicated")
    expected_formulas = {
        n127._formula_identity(
            json.loads(str(row["term_ids_json"])),
            json.loads(str(row["weights_json"])),
        )
        for _, row in bases.iterrows()
    }
    if set(base_virtual_by_formula) != expected_formulas:
        raise ValueError("NEXT130 virtual base mapping differs")
    base_risks = {
        term_id: _term_risk(features, term)
        for term_id, term in base_terms_by_id.items()
    }
    columns: dict[str, np.ndarray] = {}
    protected_terms: list[dict[str, object]] = []
    runtime_specs: list[dict[str, object]] = []
    mapping: dict[str, str] = {}
    for spec_raw in physical_specs:
        spec = dict(spec_raw)
        key = str(spec["candidate_key"])
        formula_identity = n127._formula_identity(
            spec["base_term_ids"], spec["base_weights"]
        )
        base_term_id = base_virtual_by_formula.get(formula_identity)
        if base_term_id is None or base_term_id not in base_risks:
            raise ValueError("NEXT130 physical-to-virtual base mapping is incomplete")
        protection_term_id = spec.get("protection_term_id")
        weight = float(spec.get("protection_weight", 0.0))
        if protection_term_id is None:
            if weight != 0.0:
                raise ValueError("NEXT130 inactive protection weight differs")
            evaluation_term_id = base_term_id
        else:
            if protection_term_id != PROTECTION_TERM_ID or weight not in PROTECTION_WEIGHTS:
                raise ValueError("NEXT130 protection specification differs")
            score, supported = apply_protection_score(
                base_score=base_risks[base_term_id][0],
                base_supported=base_risks[base_term_id][1],
                protection=protection,
                protection_active=protection_active,
                protection_weight=weight,
            )
            maximum = float(np.max(score[supported])) if supported.any() else 0.0
            divisor = max(1.0, maximum / 50.0)
            encoded = np.full(len(features), np.nan, dtype=float)
            encoded[supported] = np.sinh(score[supported] / divisor)
            if not np.isfinite(encoded[supported]).all():
                raise ValueError("NEXT130 virtual protected encoding overflowed")
            digest = hashlib.sha256(key.encode()).hexdigest()[:24]
            evaluation_term_id = f"next130_virtual_candidate__{digest}"
            feature_name = f"_{evaluation_term_id}_value"
            if evaluation_term_id in base_terms_by_id or evaluation_term_id in {
                term["term_id"] for term in protected_terms
            }:
                raise ValueError("NEXT130 virtual candidate identity collision")
            columns[feature_name] = encoded
            protected_terms.append(
                {
                    "term_id": evaluation_term_id,
                    "feature": feature_name,
                    "direction": 1,
                    "transform": "asinh",
                    "center": 0.0,
                    "scale": 1.0 / divisor,
                    "group": "next130_subtractive_coordination_protection",
                    "encoding": "asinh_sinh_exact_max_zero_base_minus_weight_protection",
                    "physical_candidate_key": key,
                }
            )
        if key in mapping:
            raise ValueError("NEXT130 physical candidate identity is duplicated")
        mapping[key] = evaluation_term_id
        runtime_specs.append(
            {
                "candidate_key": key,
                "base_term_ids": [evaluation_term_id],
                "base_weights": [1.0],
                "optional_term_id": None,
                "optional_weight": 0.0,
            }
        )
    return (
        pd.concat([features.reset_index(drop=True), pd.DataFrame(columns)], axis=1),
        [*base_virtual_terms, *protected_terms],
        runtime_specs,
        mapping,
    )


def verify_base_reproduction(
    *, result_records: Sequence[Mapping[str, object]], prior: pd.DataFrame
) -> None:
    """Prove pure bases exactly reproduce the published NEXT125 diagnostics."""

    metrics = (
        "scigen_pooled_auc",
        "scigen_macro_auc",
        "scigen_worst_auc",
        "wyformer_pooled_auc",
        "wyformer_macro_auc",
        "wyformer_worst_auc",
    )
    observed: dict[str, Mapping[str, object]] = {}
    for record in result_records:
        payload = json.loads(str(record["candidate_key"]))
        if payload.get("protection_term_id") is not None:
            continue
        key = n127._formula_identity(
            payload["base_term_ids"], payload["base_weights"]
        )
        if key in observed:
            raise RuntimeError("NEXT130 base-only formula identities are duplicated")
        observed[key] = record
    expected: dict[str, Mapping[str, object]] = {}
    for _, row in prior.iterrows():
        key = n127._formula_identity(
            json.loads(str(row["term_ids_json"])),
            json.loads(str(row["weights_json"])),
        )
        if key in expected:
            raise RuntimeError("NEXT130 prior base formula identities are duplicated")
        expected[key] = row
    if set(observed) != set(expected):
        raise RuntimeError("NEXT130 base-only reproduction identities differ")
    for key, row in expected.items():
        record = observed[key]
        source = row["_prior_record"]
        if any(
            not math.isclose(
                float(record[name]),
                float(source[name]),
                rel_tol=0.0,
                abs_tol=BASE_REPRODUCTION_AUC_TOLERANCE,
            )
            for name in metrics
        ) or any(
            bool(record[name]) != bool(source[name])
            for name in ("passes_source_auc_gates", "passes_safe_all_cells")
        ) or int(record["safe_passing_cells"]) != int(source["safe_passing_cells"]):
            raise RuntimeError("NEXT130 base-only diagnostics do not reproduce NEXT125")


def _paths(roots: Mapping[str, Path], freeze_path: Path) -> dict[str, Path]:
    paths = n125._paths(roots, freeze_path)
    paths.update(
        {
            "next125_manifest": roots["next125"] / n125.MANIFEST_NAME,
            "next125_catalogue": roots["next125"] / n125.CATALOGUE_NAME,
            "next125_evaluation": roots["next125"] / n125.EVALUATION_NAME,
            "next125_search_records": roots["next125"] / n125.SEARCH_NAME,
            "next129_manifest": roots["next129"] / n129.MANIFEST_NAME,
            "next129_catalogue": roots["next129"] / n129.CATALOGUE_NAME,
            "next129_scigen_features": roots["next129"] / n129.FEATURE_FILES["scigen"],
            "next129_wyformer_features": roots["next129"] / n129.FEATURE_FILES["wyformer"],
        }
    )
    return paths


def _join_label_free_features(
    paths: Mapping[str, Path],
) -> tuple[pd.DataFrame, Mapping[str, pd.DataFrame], list[dict[str, object]], list[dict[str, object]]]:
    features, feature_tables, old_terms = n125.prior._reconstruct_label_free_table(paths)
    retained = sorted(
        {str(spec["raw_feature"]) for spec in n125.FROZEN_TERM_SPECS}
        | {str(spec["support_column"]) for spec in n125.FROZEN_TERM_SPECS}
    )
    mhcr_frames: list[pd.DataFrame] = []
    protection_frames: list[pd.DataFrame] = []
    for source in ("scigen", "wyformer"):
        mhcr = pd.read_parquet(paths[f"next124_{source}_features"])
        if mhcr["material_id"].astype(str).duplicated().any() or set(retained) - set(mhcr.columns):
            raise ValueError("NEXT130 MHCR feature schema differs")
        mhcr_frame = mhcr.loc[:, ["material_id", *retained]].copy()
        mhcr_frame["material_id"] = source + ":" + mhcr_frame["material_id"].astype(str)
        mhcr_frames.append(mhcr_frame)
        protection = pd.read_parquet(paths[f"next129_{source}_features"])
        required = {"material_id", n129.FEATURE_NAME, n129.SUPPORT_COLUMN}
        if required - set(protection.columns) or protection["material_id"].astype(str).duplicated().any():
            raise ValueError("NEXT130 protection artifact schema differs")
        protection_frame = protection.loc[:, sorted(required)].copy()
        protection_frame["material_id"] = source + ":" + protection_frame["material_id"].astype(str)
        protection_frames.append(protection_frame)
    mhcr_all = pd.concat(mhcr_frames, ignore_index=True, sort=False)
    joined = features.merge(mhcr_all, on="material_id", how="inner", validate="one_to_one")
    if len(joined) != len(features) or len(joined) != len(mhcr_all):
        raise ValueError("NEXT130 MHCR row accounting differs")
    extended, mhcr_terms = n125.materialize_mhcr_tail_terms(joined)
    protection_all = pd.concat(protection_frames, ignore_index=True, sort=False)
    extended = extended.merge(
        protection_all, on="material_id", how="inner", validate="one_to_one"
    )
    if len(extended) != len(features) or len(protection_all) != len(features):
        raise ValueError("NEXT130 protection row accounting differs")
    return extended, feature_tables, old_terms, mhcr_terms


def run_coordination_protection_search(
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
    freeze_path: Path,
    output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen 1,560-candidate subtractive-protection discovery search."""

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
            )
        },
    }
    target = Path(output_dir).resolve()
    paths = _paths(roots, freeze_path)
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT130 discovery input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            key
            for key in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(key) != EXPECTED_INPUT_SHA256.get(key)
        )
        raise ValueError(f"NEXT130 formal input identity differs: {differing}")

    manifest125 = json.loads(paths["next125_manifest"].read_text())
    manifest129 = json.loads(paths["next129_manifest"].read_text())
    if (
        manifest125.get("protocol") != n125.PROTOCOL
        or manifest125.get("passes_all_cross_source_discovery_gates") is not False
        or manifest125.get("opened_validation_outputs_used") is not False
        or manifest125.get("scigen_replication_endpoint_opened") is not False
        or manifest125.get("wyformer_replication_endpoint_opened") is not False
        or manifest125.get("dft_values_used_by_executable_formula") is not False
        or manifest129.get("protocol") != n129.PROTOCOL
        or manifest129.get("labels_opened") is not False
        or manifest129.get("endpoint_payloads_opened") is not False
        or manifest129.get("opened_validation_outputs_used") is not False
        or manifest129.get("dft_values_used_by_features") is not False
        or manifest129.get("learned_energy_force_stress_proxy_used") is not False
        or manifest129.get("physical_relaxation_executed") is not False
    ):
        raise ValueError("NEXT130 prior provenance differs")
    for manifest, expected in (
        (
            manifest125,
            {
                n125.CATALOGUE_NAME: "next125_catalogue",
                n125.EVALUATION_NAME: "next125_evaluation",
                n125.SEARCH_NAME: "next125_search_records",
            },
        ),
        (
            manifest129,
            {
                n129.CATALOGUE_NAME: "next129_catalogue",
                n129.FEATURE_FILES["scigen"]: "next129_scigen_features",
                n129.FEATURE_FILES["wyformer"]: "next129_wyformer_features",
            },
        ),
    ):
        outputs = manifest.get("outputs_sha256")
        if not isinstance(outputs, Mapping) or any(
            outputs.get(filename) != input_hashes[key]
            for filename, key in expected.items()
        ):
            raise ValueError("NEXT130 prior output identity differs")

    extended, feature_tables, old_terms, mhcr_terms = _join_label_free_features(paths)
    all_physical_terms = [*old_terms, *mhcr_terms]
    physical_term_ids = {str(term["term_id"]) for term in all_physical_terms}
    if len(physical_term_ids) != len(all_physical_terms) or PROTECTION_TERM_ID in physical_term_ids:
        raise ValueError("NEXT130 inherited term identities differ")
    prior_records = pd.read_parquet(paths["next125_search_records"])
    bases = n127.select_next125_bases(prior_records)
    physical_specs = build_candidate_specs(
        bases=bases, old_term_ids=physical_term_ids
    )
    base_keys = sorted(bases["prior_candidate_key"].astype(str))
    base_formulas = sorted(
        n127._formula_identity(
            json.loads(str(row["term_ids_json"])),
            json.loads(str(row["weights_json"])),
        )
        for _, row in bases.iterrows()
    )
    base_key_sha = hashlib.sha256("\n".join(base_keys).encode()).hexdigest()
    base_formula_sha = hashlib.sha256("\n".join(base_formulas).encode()).hexdigest()
    candidate_key_sha = hashlib.sha256(
        "\n".join(str(spec["candidate_key"]) for spec in physical_specs).encode()
    ).hexdigest()
    if require_formal_inputs and (
        len(bases) != EXPECTED_BASE_COUNT
        or len(set(base_formulas)) != len(bases)
        or len(physical_specs) != EXPECTED_CANDIDATE_COUNT
        or base_key_sha != EXPECTED_BASE_KEY_SHA256
        or base_formula_sha != EXPECTED_BASE_FORMULA_SHA256
        or candidate_key_sha != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT130 frozen candidate universe differs")

    scigen_endpoints = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_endpoints = pd.read_parquet(paths["wyformer_endpoint"])
    endpoint_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "material_id": "scigen:" + scigen_endpoints["material_id"].astype(str),
                    "_endpoint_numeric": pd.to_numeric(
                        scigen_endpoints["distortion_ratio"], errors="coerce"
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "material_id": "wyformer:" + wyformer_endpoints["material_id"].astype(str),
                    "_endpoint_numeric": n125.n121.prior._endpoint_numeric(
                        wyformer_endpoints["endpoint_stratum"]
                    ),
                }
            ),
        ],
        ignore_index=True,
    )
    combined = extended.merge(
        endpoint_frame, on="material_id", how="inner", validate="one_to_one"
    )
    if len(combined) != len(extended) or len(combined) != len(endpoint_frame):
        raise ValueError("NEXT130 endpoint row accounting differs")
    endpoint = pd.to_numeric(combined.pop("_endpoint_numeric"), errors="coerce").to_numpy(float)
    if not np.isfinite(endpoint).all():
        raise ValueError("NEXT130 endpoint conversion differs")

    combined, base_virtual_terms, base_virtual_by_formula = n127.materialize_virtual_bases(
        features=combined,
        bases=bases,
        old_terms=old_terms,
        mhcr_terms=mhcr_terms,
    )
    combined, virtual_terms, runtime_specs, virtual_by_candidate = materialize_protected_candidates(
        features=combined,
        bases=bases,
        base_virtual_terms=base_virtual_terms,
        base_virtual_by_formula=base_virtual_by_formula,
        physical_specs=physical_specs,
    )
    if (
        len(runtime_specs) != len(physical_specs)
        or len(virtual_by_candidate) != len(physical_specs)
        or len({str(term["term_id"]) for term in virtual_terms}) != len(virtual_terms)
    ):
        raise RuntimeError("NEXT130 runtime candidate accounting differs")

    started = time.perf_counter()
    result = n125.search_optional_guard_laws_parallel(
        features=combined,
        endpoint=endpoint,
        old_terms=virtual_terms,
        optional_terms=[],
        candidate_specs=runtime_specs,
        workers=search_workers,
    )
    elapsed = time.perf_counter() - started
    verify_base_reproduction(result_records=result["candidate_records"], prior=bases)

    physical_term_by_id = {
        str(term["term_id"]): dict(term) for term in all_physical_terms
    }

    def decorate_record(record: dict[str, object]) -> None:
        payload = json.loads(str(record["candidate_key"]))
        evaluated_ids = json.loads(str(record["base_term_ids_json"]))
        record["evaluation_virtual_term_id"] = str(evaluated_ids[0])
        record["base_term_ids_json"] = json.dumps(
            payload["base_term_ids"], separators=(",", ":")
        )
        record["base_weights_json"] = json.dumps(
            payload["base_weights"], separators=(",", ":")
        )
        record["protection_term_id"] = payload["protection_term_id"]
        record["protection_weight"] = float(payload["protection_weight"])
        record["score_composition"] = (
            "base"
            if payload["protection_term_id"] is None
            else "max(0,base-weight*coordination_protection)"
        )
        record["physical_base_term_count"] = len(payload["base_term_ids"])
        record["physical_term_count"] = len(payload["base_term_ids"]) + int(
            payload["protection_term_id"] is not None
        )

    for record in result["candidate_records"]:
        decorate_record(record)
    selected_record = result["selected"]["record"]
    if "evaluation_virtual_term_id" not in selected_record:
        decorate_record(selected_record)
    selected_payload = json.loads(str(selected_record["candidate_key"]))
    selected_formula = result["selected"]["formula"]
    selected_formula["evaluation_virtual_term_id"] = str(
        selected_formula["base_terms"][0]["term_id"]
    )
    selected_formula["base_terms"] = [
        {**physical_term_by_id[str(term_id)], "weight": float(weight)}
        for term_id, weight in zip(
            selected_payload["base_term_ids"],
            selected_payload["base_weights"],
            strict=True,
        )
    ]
    selected_formula["protection_term"] = (
        None
        if selected_payload["protection_term_id"] is None
        else {
            "term_id": PROTECTION_TERM_ID,
            "feature": n129.FEATURE_NAME,
            "support_column": n129.SUPPORT_COLUMN,
            "raw_feature": n129.RAW_FEATURE,
            "definition": "clip(max(0,(log1p(raw)-center)/scale),0,clip_normalized)",
            "center": n129.CENTER,
            "scale": n129.SCALE,
            "clip_normalized": n129.CLIP_NORMALIZED,
            "weight": float(selected_payload["protection_weight"]),
            "polarity": "subtractive_protection",
        }
    )
    selected_formula["score_composition"] = (
        "base"
        if selected_payload["protection_term_id"] is None
        else "max(0,base_score-protection_weight*coordination_protection)"
    )
    selected_formula["protection_missing_policy"] = "PROTECTION_OFF_KEEP_BASE"
    selected_formula["nested_mhcr_missing_policy"] = "OPTIONAL_GUARD_OFF_KEEP_PRE_MHCR_BASE"
    selected_formula["kind"] = "next125_auc_safe12_base_with_subtractive_high_coordination_protection"
    selected = result["selected"]
    passes = bool(selected_record["passes_all_discovery_gates"])

    records_frame = pd.DataFrame(result["candidate_records"])
    counts_by_weight: dict[str, object] = {}
    for label, frame in records_frame.groupby(
        records_frame["protection_weight"].map(lambda value: f"{float(value):g}"),
        sort=True,
    ):
        counts_by_weight[str(label)] = {
            "candidates": int(len(frame)),
            "passes_source_auc_gates": int(frame["passes_source_auc_gates"].sum()),
            "passes_safe_all_cells": int(frame["passes_safe_all_cells"].sum()),
            "passes_broad_all_cells": int(frame["passes_broad_all_cells"].sum()),
            "passes_all_discovery_gates": int(frame["passes_all_discovery_gates"].sum()),
        }
    label_free_catalogue = {
        "protocol": PROTOCOL,
        "freeze_sha256": input_hashes["freeze"],
        "base_protocol": n125.PROTOCOL,
        "protection_protocol": n129.PROTOCOL,
        "protection_term_id": PROTECTION_TERM_ID,
        "protection_feature": n129.FEATURE_NAME,
        "protection_support_column": n129.SUPPORT_COLUMN,
        "protection_definition": "clip(max(0,(log1p(raw)-center)/scale),0,clip_normalized)",
        "active_score": "max(0,base_score-weight*coordination_protection)",
        "inactive_score": "base_score",
        "base_support_unchanged": True,
        "weight_grid": list(PROTECTION_WEIGHTS),
        "frontier_base_count": len(bases),
        "candidate_count": len(physical_specs),
        "frontier_base_key_sha256": base_key_sha,
        "frontier_base_formula_sha256": base_formula_sha,
        "candidate_key_sha256": candidate_key_sha,
        "new_protection_feature_joined_to_endpoint_before_freeze": False,
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    label_free_catalogue_sha = hashlib.sha256(
        json.dumps(label_free_catalogue, indent=2, sort_keys=True).encode() + b"\n"
    ).hexdigest()

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next103_dobvr_optional_guard_search.py": repository_root / "src/next103_dobvr_optional_guard_search.py",
        "src/next125_mhcr_frontier_rescue.py": repository_root / "src/next125_mhcr_frontier_rescue.py",
        "src/next127_hall_profile_persistence_rescue.py": repository_root / "src/next127_hall_profile_persistence_rescue.py",
        "src/next129_coordination_protection.py": repository_root / "src/next129_coordination_protection.py",
        "src/next130_coordination_protection_search.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    output_paths: list[Path] = []
    try:
        catalogue_path = staging / CATALOGUE_NAME
        evaluation_path = staging / EVALUATION_NAME
        search_path = staging / SEARCH_NAME
        _write_json(
            catalogue_path,
            {**label_free_catalogue, "label_free_catalogue_sha256": label_free_catalogue_sha},
        )
        _write_json(
            evaluation_path,
            {
                "protocol": PROTOCOL,
                "evaluation_mode": "fixed_subtractive_high_coordination_protection_broad_rescue",
                "rows": {
                    "scigen": int(len(feature_tables["scigen"])),
                    "wyformer": int(len(feature_tables["wyformer"])),
                    "total": int(len(combined)),
                },
                "frontier_base_count": len(bases),
                "candidate_count": int(result["candidate_count"]),
                "elapsed_seconds": elapsed,
                "search_workers": search_workers,
                "base_only_reproduced_next125": True,
                "counts_by_protection_weight": counts_by_weight,
                "safe_gates": dict(n125.n121.prior.DEFAULT_GATES),
                "source_auc_gates": dict(n125.n121.prior.AUC_GATES),
                "broad_min_severe_precision_lower": n125.n121.prior.BROAD_MIN_PRECISION_LOWER,
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
            "label_free_catalogue_sha256": label_free_catalogue_sha,
            "frontier_base_count": len(bases),
            "candidate_count": int(result["candidate_count"]),
            "search_workers": search_workers,
            "base_only_reproduced_next125": True,
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "requires_unopened_internal_validation_before_claim": True,
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
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT130 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT130 source changed before publication")
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
    for stage in (98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124, 125, 129):
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--freeze-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_coordination_protection_search(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{
            f"next{stage}_dir": getattr(args, f"next{stage}_dir")
            for stage in (98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124, 125, 129)
        },
        freeze_path=args.freeze_path,
        output_dir=args.output_dir,
        search_workers=args.search_workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_CANDIDATE_COUNT",
    "EXPECTED_CANDIDATE_KEY_SHA256",
    "EXPECTED_FREEZE_SHA256",
    "PROTECTION_TERM_ID",
    "PROTECTION_WEIGHTS",
    "apply_protection_score",
    "build_candidate_specs",
    "materialize_protected_candidates",
    "run_coordination_protection_search",
    "verify_base_reproduction",
]
