#!/usr/bin/env python3
"""Frozen dual compactness-protection search over coordination-protected bases."""

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
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next87_scigen_sparse_law_search import _term_risk


PROTOCOL = "2026-08-08-next134-compactness-protection-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT134_COMPACTNESS_PROTECTION_CATALOGUE.json"
EVALUATION_NAME = "NEXT134_DISCOVERY_EVALUATION.json"
SEARCH_NAME = "next134_compactness_protection_candidate_search.parquet"
EXPECTED_FREEZE_SHA256 = "b00fe65b66d60e678bc8b85dc382dc5e8a50b3a8643c5ece6c69e864162dde83"
EXPECTED_BASE_COUNT = 11
EXPECTED_CONFIGURATION_COUNT = 49
EXPECTED_CANDIDATE_COUNT = 539
EXPECTED_BASE_KEY_SHA256 = n132.EXPECTED_BASE_KEY_SHA256
EXPECTED_BASE_FORMULA_SHA256 = n132.EXPECTED_BASE_FORMULA_SHA256
EXPECTED_CONFIGURATION_SHA256 = "052947b00ae68a028c2ad9307602fbf46cb6e723dc546d0c99aa28abeb13bfae"
EXPECTED_CANDIDATE_KEY_SHA256 = "82155cb2f9e26cf81488bfe7f3ba7cef872298491883221ba5c165c5ad7e3f1a"
COORDINATION_WEIGHT = 2.0
COMPACTNESS_WEIGHTS = (0.10, 0.25, 0.50, 1.00, 2.00, 4.00)
PACKING_TERM_ID = "covalent_packing_protection__high"
VOLUME_TERM_ID = "low_volume_protection__high"
TERM_IDS = (PACKING_TERM_ID, VOLUME_TERM_ID)
SEARCH_WORKERS = 4
EXPECTED_INPUT_SHA256 = {
    **{key: value for key, value in n130.EXPECTED_INPUT_SHA256.items() if key != "freeze"},
    "next130_manifest": "8c672fdcd5b97a282604ebd49678d698e24c5f5f4e90412fb056844131d0119e",
    "next130_catalogue": "3c1386bd338ccccfc777825053e4f440171a83695450dc13bfa4e88723cf9857",
    "next130_evaluation": "87b7672aa6c2224597c0b0a3b582a2c353db76426ef7de956d89493d7ef4a019",
    "next130_search_records": "223dfb259e7b62e423bc5739f01ba18f3107aedd84a5370a8351fc94fc9f8cb0",
    "next133_manifest": "7bef252da5eb9a1e31066529fbb785fca741998d9b334a6bcc17c0b7d747eb97",
    "next133_catalogue": "7063eb41ba4d5aa3d50a4b7b4248b1b25137b69ba2a087b079000def525b143f",
    "next133_scigen_features": "b9dd53c46e941a8777e89bf309d14a81044c0308f17d895fd65328bc828f348a",
    "next133_wyformer_features": "fde1e319bf909a4320c97f14275e57af62f64d3097d780a62bc3edeb29612ca7",
    "freeze": EXPECTED_FREEZE_SHA256,
}


def build_compactness_configurations() -> list[dict[str, object]]:
    configurations = [{"term_ids": [], "weights": []}]
    for term_id in TERM_IDS:
        for weight in COMPACTNESS_WEIGHTS:
            configurations.append({"term_ids": [term_id], "weights": [weight]})
    for packing_weight in COMPACTNESS_WEIGHTS:
        for volume_weight in COMPACTNESS_WEIGHTS:
            configurations.append(
                {
                    "term_ids": [PACKING_TERM_ID, VOLUME_TERM_ID],
                    "weights": [packing_weight, volume_weight],
                }
            )
    return configurations


def compose_compactness_protection_score(
    *,
    base_score: object,
    base_supported: object,
    protections: Sequence[object],
    active: Sequence[object],
    weights: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base_score, dtype=float)
    supported = np.asarray(base_supported, dtype=bool)
    if (
        base.ndim != 1
        or supported.shape != base.shape
        or len(protections) != len(active)
        or len(protections) != len(weights)
        or np.any(supported & (~np.isfinite(base) | (base < -1.0e-12)))
    ):
        raise ValueError("NEXT134 compactness score arrays differ")
    subtraction = np.zeros(len(base), dtype=float)
    for raw_values, raw_active, raw_weight in zip(protections, active, weights, strict=True):
        values = np.asarray(raw_values, dtype=float)
        term_active = np.asarray(raw_active, dtype=bool)
        weight = float(raw_weight)
        if (
            values.shape != base.shape
            or term_active.shape != base.shape
            or not math.isfinite(weight)
            or weight <= 0.0
            or np.any(term_active & (~np.isfinite(values) | (values < -1.0e-12)))
        ):
            raise ValueError("NEXT134 compactness term differs")
        subtraction[term_active] += weight * values[term_active]
    score = np.full(len(base), np.nan, dtype=float)
    score[supported] = np.maximum(0.0, base[supported] - subtraction[supported])
    return score, supported.copy()


def build_candidate_specs(
    *, bases: pd.DataFrame, physical_term_ids: set[str]
) -> list[dict[str, object]]:
    configurations = build_compactness_configurations()
    specs: dict[str, dict[str, object]] = {}
    for _, row in bases.iterrows():
        base_ids = [str(value) for value in json.loads(str(row["term_ids_json"]))]
        base_weights = [float(value) for value in json.loads(str(row["weights_json"]))]
        if (
            not base_ids
            or len(base_ids) != len(base_weights)
            or any(term_id not in physical_term_ids for term_id in base_ids)
        ):
            raise ValueError("NEXT134 base formula differs")
        for configuration in configurations:
            payload = {
                "base_term_ids": base_ids,
                "base_weights": base_weights,
                "coordination_protection_term_id": n130.PROTECTION_TERM_ID,
                "coordination_protection_weight": COORDINATION_WEIGHT,
                "compactness_term_ids": list(configuration["term_ids"]),
                "compactness_weights": list(configuration["weights"]),
            }
            key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            specs[key] = {"candidate_key": key, **payload}
    return [specs[key] for key in sorted(specs)]


def materialize_coordination_bases(
    *,
    features: pd.DataFrame,
    bases: pd.DataFrame,
    base_virtual_terms: Sequence[Mapping[str, object]],
    base_virtual_by_formula: Mapping[str, str],
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, str]]:
    base_by_id = {str(term["term_id"]): dict(term) for term in base_virtual_terms}
    risks = {term_id: _term_risk(features, term) for term_id, term in base_by_id.items()}
    protection = pd.to_numeric(features[n130.n129.FEATURE_NAME], errors="coerce").to_numpy(float)
    active = features[n130.n129.SUPPORT_COLUMN].eq(True).to_numpy()
    columns: dict[str, np.ndarray] = {}
    terms: list[dict[str, object]] = []
    mapping: dict[str, str] = {}
    for _, row in bases.iterrows():
        identity = n130.n127._formula_identity(
            json.loads(str(row["term_ids_json"])), json.loads(str(row["weights_json"]))
        )
        source_id = base_virtual_by_formula.get(identity)
        if source_id is None or source_id not in risks:
            raise ValueError("NEXT134 coordination base mapping differs")
        score, supported = n130.apply_protection_score(
            base_score=risks[source_id][0],
            base_supported=risks[source_id][1],
            protection=protection,
            protection_active=active,
            protection_weight=COORDINATION_WEIGHT,
        )
        maximum = float(np.max(score[supported])) if supported.any() else 0.0
        divisor = max(1.0, maximum / 50.0)
        encoded = np.full(len(features), np.nan, dtype=float)
        encoded[supported] = np.sinh(score[supported] / divisor)
        digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
        term_id = f"next134_coordination_base__{digest}"
        feature_name = f"_{term_id}_value"
        columns[feature_name] = encoded
        terms.append(
            {
                "term_id": term_id,
                "feature": feature_name,
                "direction": 1,
                "transform": "asinh",
                "center": 0.0,
                "scale": 1.0 / divisor,
                "group": "next130_coordination_weight2_base",
            }
        )
        mapping[identity] = term_id
    return pd.concat([features.reset_index(drop=True), pd.DataFrame(columns)], axis=1), terms, mapping


def materialize_candidates(
    *,
    features: pd.DataFrame,
    coordination_terms: Sequence[Mapping[str, object]],
    coordination_by_formula: Mapping[str, str],
    specs: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    coordination_by_id = {str(term["term_id"]): dict(term) for term in coordination_terms}
    base_risks = {term_id: _term_risk(features, term) for term_id, term in coordination_by_id.items()}
    feature_spec = {
        PACKING_TERM_ID: (n133.PACKING_FEATURE, n133.PACKING_SUPPORT),
        VOLUME_TERM_ID: (n133.VOLUME_FEATURE, n133.VOLUME_SUPPORT),
    }
    values = {
        term_id: pd.to_numeric(features[name], errors="coerce").to_numpy(float)
        for term_id, (name, _) in feature_spec.items()
    }
    active = {
        term_id: features[support].eq(True).to_numpy()
        for term_id, (_, support) in feature_spec.items()
    }
    columns: dict[str, np.ndarray] = {}
    terms: list[dict[str, object]] = []
    runtime: list[dict[str, object]] = []
    for spec_raw in specs:
        spec = dict(spec_raw)
        identity = n130.n127._formula_identity(spec["base_term_ids"], spec["base_weights"])
        base_id = coordination_by_formula.get(identity)
        term_ids = [str(value) for value in spec["compactness_term_ids"]]
        weights = [float(value) for value in spec["compactness_weights"]]
        if base_id is None or len(term_ids) != len(weights) or any(term_id not in values for term_id in term_ids):
            raise ValueError("NEXT134 candidate configuration differs")
        score, supported = compose_compactness_protection_score(
            base_score=base_risks[base_id][0],
            base_supported=base_risks[base_id][1],
            protections=[values[term_id] for term_id in term_ids],
            active=[active[term_id] for term_id in term_ids],
            weights=weights,
        )
        maximum = float(np.max(score[supported])) if supported.any() else 0.0
        divisor = max(1.0, maximum / 50.0)
        encoded = np.full(len(features), np.nan, dtype=float)
        encoded[supported] = np.sinh(score[supported] / divisor)
        key = str(spec["candidate_key"])
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        virtual_id = f"next134_virtual_candidate__{digest}"
        feature_name = f"_{virtual_id}_value"
        if not np.isfinite(encoded[supported]).all():
            raise ValueError("NEXT134 virtual candidate encoding differs")
        columns[feature_name] = encoded
        terms.append(
            {
                "term_id": virtual_id,
                "feature": feature_name,
                "direction": 1,
                "transform": "asinh",
                "center": 0.0,
                "scale": 1.0 / divisor,
                "group": "next134_compactness_protected_candidate",
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
    return pd.concat([features.reset_index(drop=True), pd.DataFrame(columns)], axis=1), terms, runtime


def verify_base_reproduction(
    *, result_records: Sequence[Mapping[str, object]], prior: pd.DataFrame
) -> None:
    metrics = (
        "scigen_pooled_auc", "scigen_macro_auc", "scigen_worst_auc",
        "wyformer_pooled_auc", "wyformer_macro_auc", "wyformer_worst_auc",
    )
    observed: dict[str, Mapping[str, object]] = {}
    for record in result_records:
        payload = json.loads(str(record["candidate_key"]))
        if payload["compactness_term_ids"]:
            continue
        identity = n130.n127._formula_identity(payload["base_term_ids"], payload["base_weights"])
        observed[identity] = record
    expected = {
        n130.n127._formula_identity(
            json.loads(str(row["term_ids_json"])), json.loads(str(row["weights_json"]))
        ): row["_next130_record"]
        for _, row in prior.iterrows()
    }
    if set(observed) != set(expected):
        raise RuntimeError("NEXT134 base reproduction identities differ")
    for identity, source in expected.items():
        record = observed[identity]
        if any(
            not math.isclose(float(record[name]), float(source[name]), rel_tol=0.0, abs_tol=n130.BASE_REPRODUCTION_AUC_TOLERANCE)
            for name in metrics
        ) or any(
            bool(record[name]) != bool(source[name])
            for name in ("passes_source_auc_gates", "passes_safe_all_cells")
        ) or int(record["safe_passing_cells"]) != int(source["safe_passing_cells"]):
            raise RuntimeError("NEXT134 base diagnostics do not reproduce NEXT130")


def _paths(roots: Mapping[str, Path], freeze_path: Path) -> dict[str, Path]:
    paths = n130._paths(roots, freeze_path)
    paths.update(
        {
            "next130_manifest": roots["next130"] / n130.MANIFEST_NAME,
            "next130_catalogue": roots["next130"] / n130.CATALOGUE_NAME,
            "next130_evaluation": roots["next130"] / n130.EVALUATION_NAME,
            "next130_search_records": roots["next130"] / n130.SEARCH_NAME,
            "next133_manifest": roots["next133"] / n133.MANIFEST_NAME,
            "next133_catalogue": roots["next133"] / n133.CATALOGUE_NAME,
            "next133_scigen_features": roots["next133"] / n133.FEATURE_FILES["scigen"],
            "next133_wyformer_features": roots["next133"] / n133.FEATURE_FILES["wyformer"],
        }
    )
    return paths


def run_compactness_protection_search(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    next98_dir: Path, next110_dir: Path, next111_dir: Path, next113_dir: Path,
    next114_dir: Path, next116_dir: Path, next117_dir: Path, next120_dir: Path,
    next121_dir: Path, next122_dir: Path, next124_dir: Path, next125_dir: Path,
    next129_dir: Path, next130_dir: Path, next133_dir: Path,
    freeze_path: Path,
    output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{f"next{stage}": Path(value).resolve() for stage, value in (
            (98,next98_dir),(110,next110_dir),(111,next111_dir),(113,next113_dir),
            (114,next114_dir),(116,next116_dir),(117,next117_dir),(120,next120_dir),
            (121,next121_dir),(122,next122_dir),(124,next124_dir),(125,next125_dir),
            (129,next129_dir),(130,next130_dir),(133,next133_dir),
        )},
    }
    target = Path(output_dir).resolve()
    paths = _paths(roots, freeze_path)
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT134 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(key for key in set(input_hashes) | set(EXPECTED_INPUT_SHA256) if input_hashes.get(key) != EXPECTED_INPUT_SHA256.get(key))
        raise ValueError(f"NEXT134 formal input identity differs: {differing}")
    manifest130 = json.loads(paths["next130_manifest"].read_text())
    manifest133 = json.loads(paths["next133_manifest"].read_text())
    if (
        manifest130.get("protocol") != n130.PROTOCOL
        or manifest133.get("protocol") != n133.PROTOCOL
        or manifest130.get("opened_validation_outputs_used") is not False
        or manifest133.get("opened_validation_outputs_used") is not False
        or manifest130.get("dft_values_used_by_executable_formula") is not False
        or manifest133.get("dft_values_used_by_features") is not False
        or manifest133.get("endpoint_payloads_opened") is not False
    ):
        raise ValueError("NEXT134 prior provenance differs")

    extended, feature_tables, old_terms, mhcr_terms = n130._join_label_free_features(paths)
    compact_frames = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next133_{source}_features"])
        table = table.copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        compact_frames.append(table)
    compact = pd.concat(compact_frames, ignore_index=True, sort=False)
    extended = extended.merge(compact, on="material_id", how="inner", validate="one_to_one")
    if len(extended) != len(compact):
        raise ValueError("NEXT134 compactness row accounting differs")
    physical_terms = [*old_terms, *mhcr_terms]
    physical_ids = {str(term["term_id"]) for term in physical_terms}
    all_bases = n130.n127.select_next125_bases(pd.read_parquet(paths["next125_search_records"]))
    bases = n132.select_extended_bases(pd.read_parquet(paths["next130_search_records"]), all_bases)
    specs = build_candidate_specs(bases=bases, physical_term_ids=physical_ids)
    configurations = build_compactness_configurations()
    base_keys = sorted(str(value["candidate_key"]) for value in bases["_next130_record"])
    base_formulas = sorted(n130.n127._formula_identity(json.loads(str(row["term_ids_json"])), json.loads(str(row["weights_json"]))) for _, row in bases.iterrows())
    config_ids = sorted(json.dumps(config, sort_keys=True, separators=(",", ":")) for config in configurations)
    base_key_sha = hashlib.sha256("\n".join(base_keys).encode()).hexdigest()
    base_formula_sha = hashlib.sha256("\n".join(base_formulas).encode()).hexdigest()
    config_sha = hashlib.sha256("\n".join(config_ids).encode()).hexdigest()
    candidate_sha = hashlib.sha256("\n".join(str(spec["candidate_key"]) for spec in specs).encode()).hexdigest()
    if require_formal_inputs and (
        len(bases) != EXPECTED_BASE_COUNT or len(configurations) != EXPECTED_CONFIGURATION_COUNT
        or len(specs) != EXPECTED_CANDIDATE_COUNT or base_key_sha != EXPECTED_BASE_KEY_SHA256
        or base_formula_sha != EXPECTED_BASE_FORMULA_SHA256 or config_sha != EXPECTED_CONFIGURATION_SHA256
        or candidate_sha != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT134 frozen candidate universe differs")

    scigen_endpoint = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_endpoint = pd.read_parquet(paths["wyformer_endpoint"])
    endpoint_frame = pd.concat([
        pd.DataFrame({"material_id":"scigen:"+scigen_endpoint["material_id"].astype(str),"_endpoint":pd.to_numeric(scigen_endpoint["distortion_ratio"],errors="coerce")}),
        pd.DataFrame({"material_id":"wyformer:"+wyformer_endpoint["material_id"].astype(str),"_endpoint":n130.n125.n121.prior._endpoint_numeric(wyformer_endpoint["endpoint_stratum"])}),
    ], ignore_index=True)
    combined = extended.merge(endpoint_frame, on="material_id", how="inner", validate="one_to_one")
    endpoint = pd.to_numeric(combined.pop("_endpoint"), errors="coerce").to_numpy(float)
    combined, base_virtual_terms, base_virtual_by_formula = n130.n127.materialize_virtual_bases(
        features=combined, bases=bases, old_terms=old_terms, mhcr_terms=mhcr_terms
    )
    combined, coordination_terms, coordination_by_formula = materialize_coordination_bases(
        features=combined, bases=bases, base_virtual_terms=base_virtual_terms,
        base_virtual_by_formula=base_virtual_by_formula,
    )
    combined, virtual_terms, runtime = materialize_candidates(
        features=combined, coordination_terms=coordination_terms,
        coordination_by_formula=coordination_by_formula, specs=specs,
    )
    started = time.perf_counter()
    result = n130.n125.search_optional_guard_laws_parallel(
        features=combined, endpoint=endpoint, old_terms=virtual_terms,
        optional_terms=[], candidate_specs=runtime, workers=search_workers,
    )
    elapsed = time.perf_counter() - started
    verify_base_reproduction(result_records=result["candidate_records"], prior=bases)

    physical_by_id = {str(term["term_id"]): dict(term) for term in physical_terms}
    def decorate(record: dict[str, object]) -> None:
        payload = json.loads(str(record["candidate_key"]))
        evaluated = json.loads(str(record["base_term_ids_json"]))
        record["evaluation_virtual_term_id"] = str(evaluated[0])
        record["base_term_ids_json"] = json.dumps(payload["base_term_ids"], separators=(",", ":"))
        record["base_weights_json"] = json.dumps(payload["base_weights"], separators=(",", ":"))
        record["coordination_protection_weight"] = COORDINATION_WEIGHT
        record["compactness_term_ids_json"] = json.dumps(payload["compactness_term_ids"], separators=(",", ":"))
        record["compactness_weights_json"] = json.dumps(payload["compactness_weights"], separators=(",", ":"))
        record["compactness_term_count"] = len(payload["compactness_term_ids"])
        record["score_composition"] = "max(0,coordination_weight2_base-sum(compactness_weights*protections))"
    for record in result["candidate_records"]:
        decorate(record)
    selected = result["selected"]
    if "evaluation_virtual_term_id" not in selected["record"]:
        decorate(selected["record"])
    payload = json.loads(str(selected["record"]["candidate_key"]))
    formula = selected["formula"]
    formula["evaluation_virtual_term_id"] = str(formula["base_terms"][0]["term_id"])
    formula["base_terms"] = [{**physical_by_id[str(term_id)],"weight":float(weight)} for term_id,weight in zip(payload["base_term_ids"],payload["base_weights"],strict=True)]
    formula["coordination_protection"] = {"term_id":n130.PROTECTION_TERM_ID,"feature":n130.n129.FEATURE_NAME,"weight":COORDINATION_WEIGHT,"missing_policy":"TERM_OFF_KEEP_BASE"}
    compact_meta = {
        PACKING_TERM_ID:{"feature":n133.PACKING_FEATURE,"raw_feature":n133.PACKING_RAW_FEATURE,"support_column":n133.PACKING_SUPPORT},
        VOLUME_TERM_ID:{"feature":n133.VOLUME_FEATURE,"raw_feature":n133.VOLUME_RAW_FEATURE,"support_column":n133.VOLUME_SUPPORT},
    }
    formula["compactness_protections"] = [{"term_id":term_id,**compact_meta[term_id],"weight":float(weight)} for term_id,weight in zip(payload["compactness_term_ids"],payload["compactness_weights"],strict=True)]
    formula["score_composition"] = "max(0,coordination_weight2_base-sum(compactness_weights*protections))"
    formula["kind"] = "next130_coordination_base_with_optional_compactness_protection"
    passes = bool(selected["record"]["passes_all_discovery_gates"])
    records_frame = pd.DataFrame(result["candidate_records"])
    counts = {
        str(count): {
            "candidates": int(len(frame)),
            "passes_source_auc_gates": int(frame["passes_source_auc_gates"].sum()),
            "passes_safe_all_cells": int(frame["passes_safe_all_cells"].sum()),
            "passes_broad_all_cells": int(frame["passes_broad_all_cells"].sum()),
            "passes_all_discovery_gates": int(frame["passes_all_discovery_gates"].sum()),
        }
        for count, frame in records_frame.groupby("compactness_term_count", sort=True)
    }
    catalogue = {
        "protocol": PROTOCOL, "freeze_sha256": input_hashes["freeze"],
        "base_count": len(bases), "configuration_count": len(configurations),
        "candidate_count": len(specs), "weight_grid": list(COMPACTNESS_WEIGHTS),
        "term_ids": list(TERM_IDS), "base_key_sha256": base_key_sha,
        "base_formula_sha256": base_formula_sha, "configuration_sha256": config_sha,
        "candidate_key_sha256": candidate_sha,
        "active_score":"max(0,coordination_weight2_base-sum(compactness_weights*protections))",
        "base_support_unchanged": True, "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    catalogue_sha = hashlib.sha256(json.dumps(catalogue,indent=2,sort_keys=True).encode()+b"\n").hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next130_coordination_protection_search.py":Path(n130.__file__).resolve(),
        "src/next133_compactness_protection.py":Path(n133.__file__).resolve(),
        "src/next134_compactness_protection_search.py":Path(__file__).resolve(),
    }
    source_hashes = {name:_sha256_file(path) for name,path in source_paths.items()}
    output_paths: list[Path] = []
    try:
        catalogue_path=staging/CATALOGUE_NAME; evaluation_path=staging/EVALUATION_NAME; search_path=staging/SEARCH_NAME
        _write_json(catalogue_path,{**catalogue,"label_free_catalogue_sha256":catalogue_sha})
        _write_json(evaluation_path,{
            "protocol":PROTOCOL,"evaluation_mode":"fixed_dual_compactness_protection",
            "rows":{"scigen":int(len(feature_tables["scigen"])),"wyformer":int(len(feature_tables["wyformer"])),"total":int(len(combined))},
            "base_count":len(bases),"configuration_count":len(configurations),"candidate_count":int(result["candidate_count"]),
            "elapsed_seconds":elapsed,"search_workers":search_workers,"base_only_reproduced_next130":True,
            "counts_by_compactness_term_count":counts,"safe_gates":dict(n130.n125.n121.prior.DEFAULT_GATES),
            "source_auc_gates":dict(n130.n125.n121.prior.AUC_GATES),"broad_min_severe_precision_lower":n130.n125.n121.prior.BROAD_MIN_PRECISION_LOWER,
            "selected_record":selected["record"],"selected_formula":selected["formula"],"selected_safe":selected["safe"],
            "selected_safe_diagnostic":selected["safe_diagnostic"],"selected_broad":selected["broad"],
            "selected_source_diagnostics":selected["source_diagnostics"],"pauling_by_cell":result["pauling_by_cell"],"cells":result["cells"],
            "passes_all_cross_source_discovery_gates":passes,"freeze_authorized":passes,
            "requires_unopened_internal_validation_before_claim":True,
        })
        records_frame.to_parquet(search_path,index=False); output_paths.extend([catalogue_path,evaluation_path,search_path])
        manifest={
            "protocol":PROTOCOL,"label_free_catalogue_sha256":catalogue_sha,"base_count":len(bases),
            "configuration_count":len(configurations),"candidate_count":int(result["candidate_count"]),"search_workers":search_workers,
            "base_only_reproduced_next130":True,"passes_all_cross_source_discovery_gates":passes,"freeze_authorized":passes,
            "requires_unopened_internal_validation_before_claim":True,"scigen_discovery_endpoint_opened":True,
            "wyformer_discovery_endpoint_opened":True,"discovery_outcomes_used_as_offline_labels":True,
            "opened_validation_outputs_used":False,"scigen_replication_endpoint_opened":False,"wyformer_replication_endpoint_opened":False,
            "formula_or_threshold_changed_after_search":False,"dft_calculation_executed":False,
            "dft_values_used_by_executable_formula":False,"learned_energy_force_stress_proxy_used":False,
            "physical_relaxation_executed":False,"scientific_improvement_claim":False,
            "inputs_sha256":input_hashes,"executed_source_sha256":source_hashes,
            "outputs_sha256":{path.name:_sha256_file(path) for path in output_paths},
        }
        _write_json(staging/MANIFEST_NAME,manifest)
        if any(_sha256_file(path)!=input_hashes[name] for name,path in paths.items()): raise RuntimeError("NEXT134 input changed before publication")
        if any(_sha256_file(path)!=source_hashes[name] for name,path in source_paths.items()): raise RuntimeError("NEXT134 source changed before publication")
        os.replace(staging,target); return manifest
    except Exception:
        shutil.rmtree(staging,ignore_errors=True); raise


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scigen-feature-dir",type=Path,required=True); parser.add_argument("--scigen-discovery-endpoint-dir",type=Path,required=True)
    parser.add_argument("--wyformer-feature-dir",type=Path,required=True); parser.add_argument("--wyformer-discovery-endpoint-dir",type=Path,required=True)
    for stage in (98,110,111,113,114,116,117,120,121,122,124,125,129,130,133): parser.add_argument(f"--next{stage}-dir",type=Path,required=True)
    parser.add_argument("--freeze-path",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--search-workers",type=int,default=SEARCH_WORKERS); parser.add_argument("--allow-nonformal-inputs",action="store_true")
    args=parser.parse_args()
    manifest=run_compactness_protection_search(
        scigen_feature_dir=args.scigen_feature_dir,scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{f"next{stage}_dir":getattr(args,f"next{stage}_dir") for stage in (98,110,111,113,114,116,117,120,121,122,124,125,129,130,133)},
        freeze_path=args.freeze_path,output_dir=args.output_dir,search_workers=args.search_workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest,indent=2,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
