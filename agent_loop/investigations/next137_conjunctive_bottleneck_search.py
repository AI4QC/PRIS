#!/usr/bin/env python3
"""Frozen coordination-by-compactness bottleneck protection search."""

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
import src.next134_compactness_protection_search as n134
import src.next135_conjunctive_compactness_search as n135
import src.next136_conjunctive_broad_residual_diagnostic as n136
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next87_scigen_sparse_law_search import _term_risk


PROTOCOL = "2026-08-08-next137-conjunctive-bottleneck-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT137_CONJUNCTIVE_BOTTLENECK_CATALOGUE.json"
EVALUATION_NAME = "NEXT137_DISCOVERY_EVALUATION.json"
SEARCH_NAME = "next137_conjunctive_bottleneck_candidate_search.parquet"
EXPECTED_FREEZE_SHA256 = "5d537b3c6f5996dec7f75c09a2c4036c5d5be0a7be0c33b7c802a394966f7d4e"
EXPECTED_BASE_COUNT = 11
EXPECTED_CONFIGURATION_COUNT = 49
EXPECTED_CANDIDATE_COUNT = 539
EXPECTED_CONFIGURATION_SHA256 = "3974f4fedaee1aeead96cb8a90911fdc20cf71644334a544a7be522dba1f1d75"
EXPECTED_CANDIDATE_KEY_SHA256 = "863db9388a87b314e08c86764e0e12d10ec103d10afba6abee509822adfb3e62"
PACKING_MIN_TERM_ID = "coordination_covalent_packing_min__high"
VOLUME_MIN_TERM_ID = "coordination_low_volume_min__high"
PACKING_MIN_FEATURE = "coordination_covalent_packing_min"
VOLUME_MIN_FEATURE = "coordination_low_volume_min"
PACKING_MIN_SUPPORT = "coordination_covalent_packing_min_supported"
VOLUME_MIN_SUPPORT = "coordination_low_volume_min_supported"
TERM_IDS = (PACKING_MIN_TERM_ID, VOLUME_MIN_TERM_ID)
WEIGHTS = (0.10, 0.25, 0.50, 1.00, 2.00, 4.00)
SEARCH_WORKERS = 4
EXPECTED_INPUT_SHA256 = {
    **{key: value for key, value in n135.EXPECTED_INPUT_SHA256.items() if key != "freeze"},
    "freeze": EXPECTED_FREEZE_SHA256,
    "next136_manifest": "4df132a07a6c8a22eb2dd22b45a0010aea9c649f1f010deea9eae745f97d9f6f",
    "next136_diagnostic": "625423884cb283a71102ac551c6fd221d1dfebd183b53b7f06e7aad8fc02eb88",
    "next136_per_candidate": "718675a5ef88e93c0c7fe734b589b0ddb7cb5464f6a800d3f330f120fa7ffe42",
}


def materialize_bottleneck_features(table: pd.DataFrame) -> pd.DataFrame:
    """Create normalized joint bottlenecks that require both operands."""

    required = {
        n130.n129.FEATURE_NAME,
        n130.n129.SUPPORT_COLUMN,
        n133.PACKING_FEATURE,
        n133.PACKING_SUPPORT,
        n133.VOLUME_FEATURE,
        n133.VOLUME_SUPPORT,
    }
    if required - set(table.columns):
        raise ValueError("NEXT137 bottleneck feature schema differs")
    coordination = pd.to_numeric(
        table[n130.n129.FEATURE_NAME], errors="coerce"
    ).to_numpy(float)
    coordination_support = table[n130.n129.SUPPORT_COLUMN].eq(True).to_numpy()
    normalized_coordination = np.clip(
        coordination / n130.n129.CLIP_NORMALIZED, 0.0, 1.0
    )
    result: dict[str, object] = {}
    for feature, support, clip, out_feature, out_support in (
        (
            n133.PACKING_FEATURE,
            n133.PACKING_SUPPORT,
            n133.PACKING_CLIP,
            PACKING_MIN_FEATURE,
            PACKING_MIN_SUPPORT,
        ),
        (
            n133.VOLUME_FEATURE,
            n133.VOLUME_SUPPORT,
            n133.VOLUME_CLIP,
            VOLUME_MIN_FEATURE,
            VOLUME_MIN_SUPPORT,
        ),
    ):
        values = pd.to_numeric(table[feature], errors="coerce").to_numpy(float)
        active = coordination_support & table[support].eq(True).to_numpy()
        if np.any(active & (~np.isfinite(coordination) | ~np.isfinite(values))):
            raise ValueError("NEXT137 supported bottleneck operand differs")
        bottleneck = np.full(len(table), np.nan, dtype=float)
        bottleneck[active] = np.minimum(
            normalized_coordination[active],
            np.clip(values[active] / clip, 0.0, 1.0),
        )
        result[out_feature] = bottleneck
        result[out_support] = active
    return pd.DataFrame(result, index=table.index)


def build_bottleneck_configurations() -> list[dict[str, object]]:
    configurations = [{"term_ids": [], "weights": []}]
    for term_id in TERM_IDS:
        for weight in WEIGHTS:
            configurations.append({"term_ids": [term_id], "weights": [weight]})
    for packing_weight in WEIGHTS:
        for volume_weight in WEIGHTS:
            configurations.append(
                {
                    "term_ids": [PACKING_MIN_TERM_ID, VOLUME_MIN_TERM_ID],
                    "weights": [packing_weight, volume_weight],
                }
            )
    return configurations


def build_candidate_specs(
    *, bases: pd.DataFrame, physical_term_ids: set[str]
) -> list[dict[str, object]]:
    specs: dict[str, dict[str, object]] = {}
    for _, row in bases.iterrows():
        base_ids = [str(value) for value in json.loads(str(row["term_ids_json"]))]
        base_weights = [float(value) for value in json.loads(str(row["weights_json"]))]
        if (
            not base_ids
            or len(base_ids) != len(base_weights)
            or any(term_id not in physical_term_ids for term_id in base_ids)
        ):
            raise ValueError("NEXT137 base formula differs")
        for configuration in build_bottleneck_configurations():
            payload = {
                "base_term_ids": base_ids,
                "base_weights": base_weights,
                "coordination_protection_term_id": n130.PROTECTION_TERM_ID,
                "coordination_protection_weight": 2.0,
                "bottleneck_term_ids": list(configuration["term_ids"]),
                "bottleneck_weights": list(configuration["weights"]),
            }
            key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            specs[key] = {"candidate_key": key, **payload}
    return [specs[key] for key in sorted(specs)]


def materialize_candidates(
    *,
    features: pd.DataFrame,
    coordination_terms: Sequence[Mapping[str, object]],
    coordination_by_formula: Mapping[str, str],
    specs: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    base_by_id = {str(term["term_id"]): dict(term) for term in coordination_terms}
    base_risks = {
        term_id: _term_risk(features, term) for term_id, term in base_by_id.items()
    }
    feature_spec = {
        PACKING_MIN_TERM_ID: (PACKING_MIN_FEATURE, PACKING_MIN_SUPPORT),
        VOLUME_MIN_TERM_ID: (VOLUME_MIN_FEATURE, VOLUME_MIN_SUPPORT),
    }
    values = {
        term_id: features[name].to_numpy(float)
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
        identity = n130.n127._formula_identity(
            spec["base_term_ids"], spec["base_weights"]
        )
        base_id = coordination_by_formula.get(identity)
        term_ids = [str(value) for value in spec["bottleneck_term_ids"]]
        weights = [float(value) for value in spec["bottleneck_weights"]]
        if (
            base_id is None
            or len(term_ids) != len(weights)
            or any(term_id not in values for term_id in term_ids)
        ):
            raise ValueError("NEXT137 candidate configuration differs")
        score, supported = n134.compose_compactness_protection_score(
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
        virtual_id = (
            f"next137_virtual_candidate__{hashlib.sha256(key.encode()).hexdigest()[:24]}"
        )
        feature_name = f"_{virtual_id}_value"
        if not np.isfinite(encoded[supported]).all():
            raise ValueError("NEXT137 virtual candidate encoding differs")
        columns[feature_name] = encoded
        terms.append(
            {
                "term_id": virtual_id,
                "feature": feature_name,
                "direction": 1,
                "transform": "asinh",
                "center": 0.0,
                "scale": 1.0 / divisor,
                "group": "next137_conjunctive_bottleneck_candidate",
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
        terms,
        runtime,
    )


def verify_base_reproduction(
    *, result_records: Sequence[Mapping[str, object]], prior: pd.DataFrame
) -> None:
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
        if payload["bottleneck_term_ids"]:
            continue
        observed[
            n130.n127._formula_identity(
                payload["base_term_ids"], payload["base_weights"]
            )
        ] = record
    expected = {
        n130.n127._formula_identity(
            json.loads(str(row["term_ids_json"])),
            json.loads(str(row["weights_json"])),
        ): row["_next130_record"]
        for _, row in prior.iterrows()
    }
    if set(observed) != set(expected):
        raise RuntimeError("NEXT137 base reproduction identities differ")
    for identity, source in expected.items():
        record = observed[identity]
        if any(
            not math.isclose(
                float(record[name]),
                float(source[name]),
                rel_tol=0.0,
                abs_tol=n130.BASE_REPRODUCTION_AUC_TOLERANCE,
            )
            for name in metrics
        ) or any(
            bool(record[name]) != bool(source[name])
            for name in ("passes_source_auc_gates", "passes_safe_all_cells")
        ) or int(record["safe_passing_cells"]) != int(source["safe_passing_cells"]):
            raise RuntimeError("NEXT137 base diagnostics do not reproduce NEXT130")


def _paths(roots: Mapping[str, Path], freeze_path: Path) -> dict[str, Path]:
    paths = n135._paths(roots, freeze_path)
    paths.update(
        {
            "next136_manifest": roots["next136"] / n136.MANIFEST_NAME,
            "next136_diagnostic": roots["next136"] / n136.DIAGNOSTIC_NAME,
            "next136_per_candidate": roots["next136"] / n136.PER_CANDIDATE_NAME,
        }
    )
    return paths


def run_conjunctive_bottleneck_search(
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
    next136_dir: Path,
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
                (136, next136_dir),
            )
        },
    }
    target = Path(output_dir).resolve()
    paths = _paths(roots, Path(freeze_path).resolve())
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT137 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            key
            for key in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(key) != EXPECTED_INPUT_SHA256.get(key)
        )
        raise ValueError(f"NEXT137 formal input identity differs: {differing}")
    manifest136 = json.loads(paths["next136_manifest"].read_text())
    outputs136 = manifest136.get("outputs_sha256")
    if (
        manifest136.get("protocol") != n136.PROTOCOL
        or manifest136.get("new_formula_searched") is not False
        or manifest136.get("opened_validation_outputs_used") is not False
        or manifest136.get("dft_values_used_by_executable_formula") is not False
        or not isinstance(outputs136, Mapping)
        or outputs136.get(n136.DIAGNOSTIC_NAME) != input_hashes["next136_diagnostic"]
        or outputs136.get(n136.PER_CANDIDATE_NAME) != input_hashes["next136_per_candidate"]
    ):
        raise ValueError("NEXT137 diagnostic provenance differs")

    extended, feature_tables, old_terms, mhcr_terms = n130._join_label_free_features(paths)
    compact_frames: list[pd.DataFrame] = []
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
    bottleneck = materialize_bottleneck_features(extended)
    extended = pd.concat(
        [extended.reset_index(drop=True), bottleneck.reset_index(drop=True)], axis=1
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
    configurations = build_bottleneck_configurations()
    base_keys = sorted(str(value["candidate_key"]) for value in bases["_next130_record"])
    base_formulas = sorted(
        n130.n127._formula_identity(
            json.loads(str(row["term_ids_json"])),
            json.loads(str(row["weights_json"])),
        )
        for _, row in bases.iterrows()
    )
    config_ids = sorted(
        json.dumps(config, sort_keys=True, separators=(",", ":"))
        for config in configurations
    )
    base_key_sha = hashlib.sha256("\n".join(base_keys).encode()).hexdigest()
    base_formula_sha = hashlib.sha256("\n".join(base_formulas).encode()).hexdigest()
    config_sha = hashlib.sha256("\n".join(config_ids).encode()).hexdigest()
    candidate_sha = hashlib.sha256(
        "\n".join(str(spec["candidate_key"]) for spec in specs).encode()
    ).hexdigest()
    if require_formal_inputs and (
        len(bases) != EXPECTED_BASE_COUNT
        or len(configurations) != EXPECTED_CONFIGURATION_COUNT
        or len(specs) != EXPECTED_CANDIDATE_COUNT
        or base_key_sha != n132.EXPECTED_BASE_KEY_SHA256
        or base_formula_sha != n132.EXPECTED_BASE_FORMULA_SHA256
        or config_sha != EXPECTED_CONFIGURATION_SHA256
        or candidate_sha != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT137 frozen candidate universe differs")

    scigen_endpoint = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_endpoint = pd.read_parquet(paths["wyformer_endpoint"])
    endpoint_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "material_id": "scigen:" + scigen_endpoint["material_id"].astype(str),
                    "_endpoint": pd.to_numeric(
                        scigen_endpoint["distortion_ratio"], errors="coerce"
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "material_id": "wyformer:" + wyformer_endpoint["material_id"].astype(str),
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
    combined, base_virtual_terms, base_virtual_by_formula = n130.n127.materialize_virtual_bases(
        features=combined,
        bases=bases,
        old_terms=old_terms,
        mhcr_terms=mhcr_terms,
    )
    combined, coordination_terms, coordination_by_formula = n134.materialize_coordination_bases(
        features=combined,
        bases=bases,
        base_virtual_terms=base_virtual_terms,
        base_virtual_by_formula=base_virtual_by_formula,
    )
    combined, virtual_terms, runtime = materialize_candidates(
        features=combined,
        coordination_terms=coordination_terms,
        coordination_by_formula=coordination_by_formula,
        specs=specs,
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
    verify_base_reproduction(result_records=result["candidate_records"], prior=bases)

    physical_by_id = {str(term["term_id"]): dict(term) for term in physical_terms}

    def decorate(record: dict[str, object]) -> None:
        payload = json.loads(str(record["candidate_key"]))
        evaluated = json.loads(str(record["base_term_ids_json"]))
        record["evaluation_virtual_term_id"] = str(evaluated[0])
        record["base_term_ids_json"] = json.dumps(
            payload["base_term_ids"], separators=(",", ":")
        )
        record["base_weights_json"] = json.dumps(
            payload["base_weights"], separators=(",", ":")
        )
        record["coordination_protection_weight"] = 2.0
        record["bottleneck_term_ids_json"] = json.dumps(
            payload["bottleneck_term_ids"], separators=(",", ":")
        )
        record["bottleneck_weights_json"] = json.dumps(
            payload["bottleneck_weights"], separators=(",", ":")
        )
        record["bottleneck_term_count"] = len(payload["bottleneck_term_ids"])
        record["score_composition"] = (
            "max(0,coordination_weight2_base-sum(weights*conjunctive_minima))"
        )

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
        for term_id, weight in zip(
            payload["base_term_ids"], payload["base_weights"], strict=True
        )
    ]
    formula["coordination_protection"] = {
        "term_id": n130.PROTECTION_TERM_ID,
        "feature": n130.n129.FEATURE_NAME,
        "weight": 2.0,
        "missing_policy": "TERM_OFF_KEEP_BASE",
    }
    term_meta = {
        PACKING_MIN_TERM_ID: {
            "feature": PACKING_MIN_FEATURE,
            "definition": "min(normalized_coordination,normalized_covalent_packing)",
        },
        VOLUME_MIN_TERM_ID: {
            "feature": VOLUME_MIN_FEATURE,
            "definition": "min(normalized_coordination,normalized_low_volume)",
        },
    }
    formula["bottleneck_protections"] = [
        {"term_id": term_id, **term_meta[term_id], "weight": float(weight)}
        for term_id, weight in zip(
            payload["bottleneck_term_ids"], payload["bottleneck_weights"], strict=True
        )
    ]
    formula["score_composition"] = (
        "max(0,coordination_weight2_base-sum(weights*conjunctive_minima))"
    )
    formula["kind"] = "next130_coordination_base_with_optional_conjunctive_bottleneck_protection"
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
        for count, frame in records_frame.groupby("bottleneck_term_count", sort=True)
    }
    catalogue = {
        "protocol": PROTOCOL,
        "freeze_sha256": input_hashes["freeze"],
        "base_count": len(bases),
        "configuration_count": len(configurations),
        "candidate_count": len(specs),
        "weight_grid": list(WEIGHTS),
        "term_ids": list(TERM_IDS),
        "base_key_sha256": base_key_sha,
        "base_formula_sha256": base_formula_sha,
        "configuration_sha256": config_sha,
        "candidate_key_sha256": candidate_sha,
        "active_score": "max(0,coordination_weight2_base-sum(weights*conjunctive_minima))",
        "base_support_unchanged": True,
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    catalogue_sha = hashlib.sha256(
        json.dumps(catalogue, indent=2, sort_keys=True).encode() + b"\n"
    ).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next130_coordination_protection_search.py": Path(n130.__file__).resolve(),
        "src/next134_compactness_protection_search.py": Path(n134.__file__).resolve(),
        "src/next137_conjunctive_bottleneck_search.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
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
                "evaluation_mode": "fixed_conjunctive_bottleneck_protection",
                "rows": {
                    "scigen": int(len(feature_tables["scigen"])),
                    "wyformer": int(len(feature_tables["wyformer"])),
                    "total": int(len(combined)),
                },
                "base_count": len(bases),
                "configuration_count": len(configurations),
                "candidate_count": int(result["candidate_count"]),
                "elapsed_seconds": elapsed,
                "search_workers": search_workers,
                "base_only_reproduced_next130": True,
                "counts_by_bottleneck_term_count": counts,
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
        manifest = {
            "protocol": PROTOCOL,
            "label_free_catalogue_sha256": catalogue_sha,
            "base_count": len(bases),
            "configuration_count": len(configurations),
            "candidate_count": int(result["candidate_count"]),
            "search_workers": search_workers,
            "base_only_reproduced_next130": True,
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "requires_unopened_internal_validation_before_claim": True,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True,
            "opened_validation_outputs_used": False,
            "scigen_replication_endpoint_opened": False,
            "wyformer_replication_endpoint_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "physical_relaxation_executed": False,
            "formula_or_threshold_changed_after_search": False,
            "scientific_improvement_claim": False,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {
                CATALOGUE_NAME: _sha256_file(catalogue_path),
                EVALUATION_NAME: _sha256_file(evaluation_path),
                SEARCH_NAME: _sha256_file(search_path),
            },
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT137 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT137 source changed before publication")
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
    for stage in (98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124, 125, 129, 130, 133, 134, 136):
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--freeze-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_conjunctive_bottleneck_search(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{
            f"next{stage}_dir": getattr(args, f"next{stage}_dir")
            for stage in (98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124, 125, 129, 130, 133, 134, 136)
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
    "build_bottleneck_configurations",
    "build_candidate_specs",
    "materialize_bottleneck_features",
    "materialize_candidates",
    "run_conjunctive_bottleneck_search",
]
