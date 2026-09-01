#!/usr/bin/env python3
"""Frozen higher-weight continuation of subtractive coordination protection."""

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
import src.next131_protected_broad_residual_diagnostic as n131
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next87_scigen_sparse_law_search import _term_risk


PROTOCOL = "2026-08-08-next132-extended-coordination-protection-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT132_EXTENDED_COORDINATION_PROTECTION_CATALOGUE.json"
EVALUATION_NAME = "NEXT132_DISCOVERY_EVALUATION.json"
SEARCH_NAME = "next132_extended_coordination_protection_candidate_search.parquet"
EXPECTED_FREEZE_SHA256 = "61c9f483da2c33ed58d2e2fcdb7ef9f291f0c0f1adecf2e90f819017728564de"
EXPECTED_BASE_COUNT = 11
EXPECTED_CANDIDATE_COUNT = 66
EXPECTED_BASE_KEY_SHA256 = "00010ff4bbad2c9dda7430dd4a9c3f7112a16ec2694fd6447d72eb70b6b1cd9d"
EXPECTED_BASE_FORMULA_SHA256 = "d1f8763331cbe36f54e898e4efc88d0f88d2ae5d6284883acc4850e58d9678b5"
EXPECTED_CANDIDATE_KEY_SHA256 = "4bcd863569762907339200327fbaa9b08fed3c4e9c8516da3f1e2af50b1f5cd2"
PROTECTION_TERM_ID = n130.PROTECTION_TERM_ID
EXTENDED_WEIGHTS = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
SEARCH_WORKERS = 4
BASE_REPRODUCTION_AUC_TOLERANCE = n130.BASE_REPRODUCTION_AUC_TOLERANCE
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n130.EXPECTED_INPUT_SHA256.items()
        if key != "freeze"
    },
    "next130_manifest": n131.EXPECTED_NEXT130_MANIFEST_SHA256,
    "next130_catalogue": "3c1386bd338ccccfc777825053e4f440171a83695450dc13bfa4e88723cf9857",
    "next130_evaluation": "87b7672aa6c2224597c0b0a3b582a2c353db76426ef7de956d89493d7ef4a019",
    "next130_search_records": "223dfb259e7b62e423bc5739f01ba18f3107aedd84a5370a8351fc94fc9f8cb0",
    "next131_manifest": "a2498a1c5304a5336cf043ec4c7a8742b1173ac24290a9a936366f9352b3a59c",
    "next131_diagnostic": "f7f32d1ac963040b6e3284b814f13b7850f094225c65f6b9de8d0fa78218742c",
    "next131_per_candidate": "884827829e5295d23c077ffbc808a37215019ebb9fe81b2f474e87c728fb765c",
    "freeze": EXPECTED_FREEZE_SHA256,
}


def select_extended_bases(
    next130_records: pd.DataFrame, next125_bases: pd.DataFrame
) -> pd.DataFrame:
    """Match weight-2 AUC+SAFE12 formulas back to their nested NEXT125 bases."""

    required = {
        "candidate_key",
        "passes_source_auc_gates",
        "passes_safe_all_cells",
        "protection_term_id",
        "protection_weight",
    }
    if required - set(next130_records.columns):
        raise ValueError("NEXT132 NEXT130 candidate schema differs")
    keep = (
        next130_records["passes_source_auc_gates"].fillna(False).astype(bool)
        & next130_records["passes_safe_all_cells"].fillna(False).astype(bool)
        & next130_records["protection_term_id"].eq(PROTECTION_TERM_ID)
        & pd.to_numeric(next130_records["protection_weight"], errors="coerce").eq(2.0)
    )
    selected130: dict[str, dict[str, object]] = {}
    for _, row in next130_records.loc[keep].iterrows():
        payload = json.loads(str(row["candidate_key"]))
        identity = n130.n127._formula_identity(
            payload["base_term_ids"], payload["base_weights"]
        )
        if identity in selected130:
            raise ValueError("NEXT132 selected base formula is duplicated")
        selected130[identity] = row.to_dict()
    rows: list[dict[str, object]] = []
    for _, base in next125_bases.iterrows():
        identity = n130.n127._formula_identity(
            json.loads(str(base["term_ids_json"])),
            json.loads(str(base["weights_json"])),
        )
        source = selected130.get(identity)
        if source is not None:
            rows.append({**base.to_dict(), "_next130_record": source})
    result = pd.DataFrame(rows)
    if len(result) != len(selected130):
        raise ValueError("NEXT132 selected NEXT125 base mapping is incomplete")
    result["_selection_sort_key"] = result["_next130_record"].map(
        lambda value: str(value["candidate_key"])
    )
    return result.sort_values("_selection_sort_key").drop(
        columns="_selection_sort_key"
    ).reset_index(drop=True)


def build_extended_candidate_specs(
    *, bases: pd.DataFrame, old_term_ids: set[str]
) -> list[dict[str, object]]:
    """Enumerate the frozen six-weight continuation for every selected base."""

    if {"term_ids_json", "weights_json"} - set(bases.columns):
        raise ValueError("NEXT132 candidate base schema differs")
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
            raise ValueError("NEXT132 physical base formula differs")
        for protection_weight in EXTENDED_WEIGHTS:
            payload = {
                "base_term_ids": term_ids,
                "base_weights": weights,
                "protection_term_id": PROTECTION_TERM_ID,
                "protection_weight": protection_weight,
            }
            key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            specs[key] = {"candidate_key": key, **payload}
    return [specs[key] for key in sorted(specs)]


def materialize_extended_candidates(
    *,
    features: pd.DataFrame,
    bases: pd.DataFrame,
    base_virtual_terms: Sequence[Mapping[str, object]],
    base_virtual_by_formula: Mapping[str, str],
    physical_specs: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    """Encode each higher-weight protected score as an exact virtual term."""

    protection = pd.to_numeric(
        features[n130.n129.FEATURE_NAME], errors="coerce"
    ).to_numpy(float)
    active = features[n130.n129.SUPPORT_COLUMN].eq(True).to_numpy()
    if np.any(active & (~np.isfinite(protection) | (protection < -1.0e-12))):
        raise ValueError("NEXT132 protection feature differs")
    base_by_id = {str(term["term_id"]): dict(term) for term in base_virtual_terms}
    risks = {term_id: _term_risk(features, term) for term_id, term in base_by_id.items()}
    expected = {
        n130.n127._formula_identity(
            json.loads(str(row["term_ids_json"])),
            json.loads(str(row["weights_json"])),
        )
        for _, row in bases.iterrows()
    }
    if set(base_virtual_by_formula) != expected:
        raise ValueError("NEXT132 virtual base mapping differs")
    columns: dict[str, np.ndarray] = {}
    terms: list[dict[str, object]] = []
    runtime: list[dict[str, object]] = []
    for spec_raw in physical_specs:
        spec = dict(spec_raw)
        key = str(spec["candidate_key"])
        identity = n130.n127._formula_identity(
            spec["base_term_ids"], spec["base_weights"]
        )
        base_id = base_virtual_by_formula.get(identity)
        weight = float(spec["protection_weight"])
        if base_id is None or base_id not in risks or weight not in EXTENDED_WEIGHTS:
            raise ValueError("NEXT132 protected candidate specification differs")
        score, supported = n130.apply_protection_score(
            base_score=risks[base_id][0],
            base_supported=risks[base_id][1],
            protection=protection,
            protection_active=active,
            protection_weight=weight,
        )
        maximum = float(np.max(score[supported])) if supported.any() else 0.0
        divisor = max(1.0, maximum / 50.0)
        encoded = np.full(len(features), np.nan, dtype=float)
        encoded[supported] = np.sinh(score[supported] / divisor)
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        term_id = f"next132_virtual_candidate__{digest}"
        feature_name = f"_{term_id}_value"
        if not np.isfinite(encoded[supported]).all() or term_id in {
            term["term_id"] for term in terms
        }:
            raise ValueError("NEXT132 virtual candidate encoding differs")
        columns[feature_name] = encoded
        terms.append(
            {
                "term_id": term_id,
                "feature": feature_name,
                "direction": 1,
                "transform": "asinh",
                "center": 0.0,
                "scale": 1.0 / divisor,
                "group": "next132_extended_subtractive_coordination_protection",
                "encoding": "asinh_sinh_exact_max_zero_base_minus_weight_protection",
                "physical_candidate_key": key,
            }
        )
        runtime.append(
            {
                "candidate_key": key,
                "base_term_ids": [term_id],
                "base_weights": [1.0],
                "optional_term_id": None,
                "optional_weight": 0.0,
            }
        )
    return (
        pd.concat([features.reset_index(drop=True), pd.DataFrame(columns)], axis=1),
        terms,
        runtime,
    )


def verify_weight2_reproduction(
    *, result_records: Sequence[Mapping[str, object]], prior: pd.DataFrame
) -> None:
    """Prove all weight-2 controls reproduce their published NEXT130 metrics."""

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
        if float(payload["protection_weight"]) != 2.0:
            continue
        identity = n130.n127._formula_identity(
            payload["base_term_ids"], payload["base_weights"]
        )
        observed[identity] = record
    expected: dict[str, Mapping[str, object]] = {}
    for _, row in prior.iterrows():
        identity = n130.n127._formula_identity(
            json.loads(str(row["term_ids_json"])),
            json.loads(str(row["weights_json"])),
        )
        expected[identity] = row["_next130_record"]
    if set(observed) != set(expected):
        raise RuntimeError("NEXT132 weight-2 reproduction identities differ")
    for identity, source in expected.items():
        record = observed[identity]
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
            raise RuntimeError("NEXT132 weight-2 diagnostics do not reproduce NEXT130")


def _paths(roots: Mapping[str, Path], freeze_path: Path) -> dict[str, Path]:
    paths = n130._paths(roots, freeze_path)
    paths.update(
        {
            "next130_manifest": roots["next130"] / n130.MANIFEST_NAME,
            "next130_catalogue": roots["next130"] / n130.CATALOGUE_NAME,
            "next130_evaluation": roots["next130"] / n130.EVALUATION_NAME,
            "next130_search_records": roots["next130"] / n130.SEARCH_NAME,
            "next131_manifest": roots["next131"] / n131.MANIFEST_NAME,
            "next131_diagnostic": roots["next131"] / n131.DIAGNOSTIC_NAME,
            "next131_per_candidate": roots["next131"] / n131.PER_CANDIDATE_NAME,
        }
    )
    return paths


def run_extended_coordination_protection_search(
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
    next131_dir: Path,
    freeze_path: Path,
    output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen 66-candidate higher-weight discovery continuation."""

    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{
            f"next{stage}": Path(value).resolve()
            for stage, value in (
                (98, next98_dir), (110, next110_dir), (111, next111_dir),
                (113, next113_dir), (114, next114_dir), (116, next116_dir),
                (117, next117_dir), (120, next120_dir), (121, next121_dir),
                (122, next122_dir), (124, next124_dir), (125, next125_dir),
                (129, next129_dir), (130, next130_dir), (131, next131_dir),
            )
        },
    }
    target = Path(output_dir).resolve()
    paths = _paths(roots, freeze_path)
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT132 discovery input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            key
            for key in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(key) != EXPECTED_INPUT_SHA256.get(key)
        )
        raise ValueError(f"NEXT132 formal input identity differs: {differing}")
    manifest130 = json.loads(paths["next130_manifest"].read_text())
    manifest131 = json.loads(paths["next131_manifest"].read_text())
    if (
        manifest130.get("protocol") != n130.PROTOCOL
        or manifest131.get("protocol") != n131.PROTOCOL
        or manifest130.get("opened_validation_outputs_used") is not False
        or manifest131.get("opened_validation_outputs_used") is not False
        or manifest130.get("dft_values_used_by_executable_formula") is not False
        or manifest131.get("dft_values_used_by_executable_formula") is not False
        or manifest130.get("scigen_replication_endpoint_opened") is not False
        or manifest131.get("wyformer_replication_endpoint_opened") is not False
    ):
        raise ValueError("NEXT132 prior provenance differs")

    extended, feature_tables, old_terms, mhcr_terms = n130._join_label_free_features(paths)
    physical_terms = [*old_terms, *mhcr_terms]
    physical_ids = {str(term["term_id"]) for term in physical_terms}
    all_bases = n130.n127.select_next125_bases(
        pd.read_parquet(paths["next125_search_records"])
    )
    bases = select_extended_bases(
        pd.read_parquet(paths["next130_search_records"]), all_bases
    )
    specs = build_extended_candidate_specs(bases=bases, old_term_ids=physical_ids)
    base_keys = sorted(
        str(value["candidate_key"]) for value in bases["_next130_record"]
    )
    base_formulas = sorted(
        n130.n127._formula_identity(
            json.loads(str(row["term_ids_json"])),
            json.loads(str(row["weights_json"])),
        )
        for _, row in bases.iterrows()
    )
    base_key_sha = hashlib.sha256("\n".join(base_keys).encode()).hexdigest()
    base_formula_sha = hashlib.sha256("\n".join(base_formulas).encode()).hexdigest()
    candidate_sha = hashlib.sha256(
        "\n".join(str(spec["candidate_key"]) for spec in specs).encode()
    ).hexdigest()
    if require_formal_inputs and (
        len(bases) != EXPECTED_BASE_COUNT
        or len(specs) != EXPECTED_CANDIDATE_COUNT
        or base_key_sha != EXPECTED_BASE_KEY_SHA256
        or base_formula_sha != EXPECTED_BASE_FORMULA_SHA256
        or candidate_sha != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT132 frozen candidate universe differs")

    scigen_endpoint = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_endpoint = pd.read_parquet(paths["wyformer_endpoint"])
    endpoint_frame = pd.concat(
        [
            pd.DataFrame({
                "material_id": "scigen:" + scigen_endpoint["material_id"].astype(str),
                "_endpoint": pd.to_numeric(scigen_endpoint["distortion_ratio"], errors="coerce"),
            }),
            pd.DataFrame({
                "material_id": "wyformer:" + wyformer_endpoint["material_id"].astype(str),
                "_endpoint": n130.n125.n121.prior._endpoint_numeric(
                    wyformer_endpoint["endpoint_stratum"]
                ),
            }),
        ], ignore_index=True,
    )
    combined = extended.merge(endpoint_frame, on="material_id", how="inner", validate="one_to_one")
    endpoint = pd.to_numeric(combined.pop("_endpoint"), errors="coerce").to_numpy(float)
    if len(combined) != len(extended) or not np.isfinite(endpoint).all():
        raise ValueError("NEXT132 endpoint row accounting differs")
    combined, base_virtual_terms, base_virtual_by_formula = n130.n127.materialize_virtual_bases(
        features=combined, bases=bases, old_terms=old_terms, mhcr_terms=mhcr_terms
    )
    combined, virtual_terms, runtime = materialize_extended_candidates(
        features=combined,
        bases=bases,
        base_virtual_terms=base_virtual_terms,
        base_virtual_by_formula=base_virtual_by_formula,
        physical_specs=specs,
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
    verify_weight2_reproduction(result_records=result["candidate_records"], prior=bases)

    physical_by_id = {str(term["term_id"]): dict(term) for term in physical_terms}
    def decorate(record: dict[str, object]) -> None:
        payload = json.loads(str(record["candidate_key"]))
        evaluated = json.loads(str(record["base_term_ids_json"]))
        record["evaluation_virtual_term_id"] = str(evaluated[0])
        record["base_term_ids_json"] = json.dumps(payload["base_term_ids"], separators=(",", ":"))
        record["base_weights_json"] = json.dumps(payload["base_weights"], separators=(",", ":"))
        record["protection_term_id"] = PROTECTION_TERM_ID
        record["protection_weight"] = float(payload["protection_weight"])
        record["score_composition"] = "max(0,base-weight*coordination_protection)"
        record["physical_term_count"] = len(payload["base_term_ids"]) + 1
    for record in result["candidate_records"]:
        decorate(record)
    selected = result["selected"]
    if "evaluation_virtual_term_id" not in selected["record"]:
        decorate(selected["record"])
    payload = json.loads(str(selected["record"]["candidate_key"]))
    formula = selected["formula"]
    formula["evaluation_virtual_term_id"] = str(formula["base_terms"][0]["term_id"])
    formula["base_terms"] = [
        {**physical_by_id[str(term_id)], "weight": float(weight)}
        for term_id, weight in zip(payload["base_term_ids"], payload["base_weights"], strict=True)
    ]
    formula["protection_term"] = {
        "term_id": PROTECTION_TERM_ID,
        "feature": n130.n129.FEATURE_NAME,
        "raw_feature": n130.n129.RAW_FEATURE,
        "support_column": n130.n129.SUPPORT_COLUMN,
        "center": n130.n129.CENTER,
        "scale": n130.n129.SCALE,
        "clip_normalized": n130.n129.CLIP_NORMALIZED,
        "weight": float(payload["protection_weight"]),
        "polarity": "subtractive_protection",
    }
    formula["score_composition"] = "max(0,base_score-protection_weight*coordination_protection)"
    formula["protection_missing_policy"] = "PROTECTION_OFF_KEEP_BASE"
    formula["nested_mhcr_missing_policy"] = "OPTIONAL_GUARD_OFF_KEEP_PRE_MHCR_BASE"
    formula["kind"] = "next125_auc_safe12_base_with_extended_subtractive_coordination_protection"
    passes = bool(selected["record"]["passes_all_discovery_gates"])
    records_frame = pd.DataFrame(result["candidate_records"])
    counts_by_weight = {
        f"{float(weight):g}": {
            "candidates": int(len(frame)),
            "passes_source_auc_gates": int(frame["passes_source_auc_gates"].sum()),
            "passes_safe_all_cells": int(frame["passes_safe_all_cells"].sum()),
            "passes_broad_all_cells": int(frame["passes_broad_all_cells"].sum()),
            "passes_all_discovery_gates": int(frame["passes_all_discovery_gates"].sum()),
        }
        for weight, frame in records_frame.groupby("protection_weight", sort=True)
    }
    catalogue = {
        "protocol": PROTOCOL,
        "freeze_sha256": input_hashes["freeze"],
        "selection_protocol": n130.PROTOCOL,
        "diagnostic_protocol": n131.PROTOCOL,
        "selected_base_count": len(bases),
        "weight_grid": list(EXTENDED_WEIGHTS),
        "candidate_count": len(specs),
        "selected_base_key_sha256": base_key_sha,
        "selected_base_formula_sha256": base_formula_sha,
        "candidate_key_sha256": candidate_sha,
        "active_score": "max(0,base_score-weight*coordination_protection)",
        "base_support_unchanged": True,
        "new_weights_joined_to_endpoint_before_freeze": False,
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    catalogue_sha = hashlib.sha256(
        json.dumps(catalogue, indent=2, sort_keys=True).encode() + b"\n"
    ).hexdigest()

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next125_mhcr_frontier_rescue.py": Path(n130.n125.__file__).resolve(),
        "src/next127_hall_profile_persistence_rescue.py": Path(n130.n127.__file__).resolve(),
        "src/next130_coordination_protection_search.py": Path(n130.__file__).resolve(),
        "src/next132_extended_coordination_protection_search.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    output_paths: list[Path] = []
    try:
        catalogue_path = staging / CATALOGUE_NAME
        evaluation_path = staging / EVALUATION_NAME
        search_path = staging / SEARCH_NAME
        _write_json(catalogue_path, {**catalogue, "label_free_catalogue_sha256": catalogue_sha})
        _write_json(evaluation_path, {
            "protocol": PROTOCOL,
            "evaluation_mode": "fixed_extended_subtractive_coordination_protection",
            "rows": {
                "scigen": int(len(feature_tables["scigen"])),
                "wyformer": int(len(feature_tables["wyformer"])),
                "total": int(len(combined)),
            },
            "selected_base_count": len(bases),
            "candidate_count": int(result["candidate_count"]),
            "elapsed_seconds": elapsed,
            "search_workers": search_workers,
            "weight2_reproduced_next130": True,
            "counts_by_protection_weight": counts_by_weight,
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
        })
        records_frame.to_parquet(search_path, index=False)
        output_paths.extend([catalogue_path, evaluation_path, search_path])
        manifest = {
            "protocol": PROTOCOL,
            "label_free_catalogue_sha256": catalogue_sha,
            "selected_base_count": len(bases),
            "candidate_count": int(result["candidate_count"]),
            "search_workers": search_workers,
            "weight2_reproduced_next130": True,
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
            raise RuntimeError("NEXT132 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT132 source changed before publication")
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
    for stage in (98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124, 125, 129, 130, 131):
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--freeze-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_extended_coordination_protection_search(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{
            f"next{stage}_dir": getattr(args, f"next{stage}_dir")
            for stage in (98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124, 125, 129, 130, 131)
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
    "EXTENDED_WEIGHTS",
    "PROTECTION_TERM_ID",
    "build_extended_candidate_specs",
    "materialize_extended_candidates",
    "run_extended_coordination_protection_search",
    "select_extended_bases",
    "verify_weight2_reproduction",
]
