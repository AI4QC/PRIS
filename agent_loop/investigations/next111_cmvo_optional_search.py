#!/usr/bin/env python3
"""Frozen discovery-only rescue search with Brown-free CMVO obstruction terms."""

from __future__ import annotations

import argparse
import hashlib
import itertools
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

from src.next85_scigen_label_free_features import FEATURE_NAMES as SCIGEN_FEATURE_NAMES
from src.next86_scigen_endpoint_router import ENDPOINT_NAME as SCIGEN_ENDPOINT_NAME
from src.next93_wyformer_source_lockbox import _sha256_file, _write_json
from src.next93b_wyformer_blind_lockbox import ENDPOINT_NAME as WYFORMER_ENDPOINT_NAME
from src.next94_wyformer_label_free_features import FEATURE_NAMES as WYFORMER_FEATURE_NAMES
from src.next95_wyformer_sparse_law_search import DEFAULT_GATES, _endpoint_numeric
from src.next98_cross_source_discovery_search import (
    AUC_GATES,
    BROAD_MIN_PRECISION_LOWER,
    CATALOGUE_NAME as NEXT98_CATALOGUE_NAME,
    MANIFEST_NAME as NEXT98_MANIFEST_NAME,
    PROTOCOL as NEXT98_PROTOCOL,
    _read_json,
)
from src.next98b_cross_source_exhaustive_search import (
    MANIFEST_NAME as NEXT98B_MANIFEST_NAME,
    PROTOCOL as NEXT98B_PROTOCOL,
    SEARCH_NAME as NEXT98B_SEARCH_NAME,
)
from src.next103_dobvr_optional_guard_search import search_optional_guard_laws
from src.next107_two_axis_cmvf_guard_search import (
    _configuration_id,
    _decorate_result,
    build_two_axis_candidate_specs,
    materialize_composite_guard_terms,
)
from src.next108_near_miss_cmvf_rescue import (
    MANIFEST_NAME as NEXT108_MANIFEST_NAME,
    NEAR_MISS_TOLERANCE,
    PROTOCOL as NEXT108_PROTOCOL,
    select_near_miss_bases,
)
from src.next110_cross_source_cmvo_features import (
    CATALOGUE_NAME as NEXT110_CATALOGUE_NAME,
    FEATURE_COLUMNS as NEXT110_FEATURE_COLUMNS,
    FEATURE_FILES as NEXT110_FEATURE_FILES,
    MANIFEST_NAME as NEXT110_MANIFEST_NAME,
    PROTOCOL as NEXT110_PROTOCOL,
)


PROTOCOL = "2026-08-08-next111-cmvo-optional-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT111_CMVO_TERM_CATALOGUE.json"
EVALUATION_NAME = "NEXT111_DISCOVERY_EVALUATION.json"
SEARCH_NAME = "next111_cmvo_optional_candidate_search.parquet"
SINGLE_WEIGHT_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
PAIR_WEIGHT_GRID = (0.25, 0.5, 1.0, 2.0)
EXPECTED_NEAR_MISS_BASES = 353
EXPECTED_CONFIGURATION_COUNT = 63
EXPECTED_CANDIDATE_COUNT = 22_592

FROZEN_TERM_SPECS = (
    {
        "term_id": "cmvo_core_min_interval_slack__high",
        "raw_feature": "cmvo_core_min_interval_slack",
        "group": "cmvo_core",
        "support_column": "cmvo_core_supported",
        "center": 0.07142857142857142,
        "scale": 0.24999999999999986,
        "raw_p995": 0.5587653898768803,
        "clip_normalized": 1.9493472737932367,
    },
    {
        "term_id": "cmvo_core_global_balance_gap__high",
        "raw_feature": "cmvo_core_global_balance_gap",
        "group": "cmvo_core",
        "support_column": "cmvo_core_supported",
        "center": 0.14285714285714285,
        "scale": 0.4666666666666667,
        "raw_p995": 0.9,
        "clip_normalized": 1.6224489795918366,
    },
    {
        "term_id": "cmvo_core_component_balance_gap__high",
        "raw_feature": "cmvo_core_component_balance_gap",
        "group": "cmvo_core",
        "support_column": "cmvo_core_supported",
        "center": 0.14285714285714285,
        "scale": 0.5,
        "raw_p995": 1.0,
        "clip_normalized": 1.7142857142857144,
    },
)
EXCLUDED_TERM_SPECS = (
    "cmvo_core_unserved_site_fraction",
    "cmvo_expanded_min_interval_slack",
    "cmvo_expanded_global_balance_gap",
    "cmvo_expanded_component_balance_gap",
    "cmvo_expanded_unserved_site_fraction",
)

EXPECTED_INPUT_SHA256 = {
    "scigen_features": "7031d86e4fb6e469c674d208f680ace1dbe5e11e45f3d4b2befefd747efdde16",
    "scigen_endpoint": "f86cff6f5e9124ee82aae13911ffe55a125c6fe111fc1f64122a610febf67958",
    "wyformer_features": "c515baec0fccef5bc03c7672f1d4e1aca278f5ed4d7b6f1bf7f66c734e2b87f7",
    "wyformer_endpoint": "f39836e62a1da03ed823479e87d6f75fc0d01da60a8c0a2faa696638cc2fb9d7",
    "next98_manifest": "5fcd924b125767e52ac1826203595692af868ab35366899e12b82aea2726e32c",
    "next98_term_catalogue": "f2165f548a56cda04559a11a0d575f0654d3e8a17cf3b85b76e7974ea65dee41",
    "next98b_manifest": "b20d2f500ce74a6fd8b1a8a992bca3fff3ee5952fc38c09d3ad34ca317c3084d",
    "next98b_search_records": "748a4623ecfc725636837f3944b70482a97b2df39a495a81e3f8e09f5d09a4e4",
    "next108_manifest": "005f276ab2627863f8f942e2e15ab8a3c2b9868439ca00c4782d8ff94eeefda5",
    "next110_manifest": "06000213e80de7afa2e13f9dd67561ff2b56a9a10fede90260c269ad57dc03b3",
    "next110_catalogue": "5a9e66a87779555f91019ac1873a5b2974154e51b2911986b3911c5d69b5ac01",
    "next110_scigen_features": "023c0662fabd73df0a7f47c1e10dc7e229fb0b5cde6f2d76c34c3c6efc1bb31e",
    "next110_wyformer_features": "fd225555c8cadd2219df6fec679c74c78a9a5c15065f23553d7e6d1eec681c94",
    "freeze": "2ae6cd2389ef1e445a0f7da1224356a64f62d624b6ee5e1465fc595489084a29",
}


def _json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def build_cmvo_guard_configurations(
    term_specs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Enumerate three singles and every same-mode pair on frozen grids."""

    terms = sorted((dict(term) for term in term_specs), key=lambda item: str(item.get("term_id")))
    term_ids = [term.get("term_id") for term in terms]
    if (
        len(terms) != 3
        or len(set(term_ids)) != len(terms)
        or any(not isinstance(term_id, str) for term_id in term_ids)
        or any(term.get("group") != "cmvo_core" for term in terms)
    ):
        raise ValueError("NEXT111 frozen CMVO term identity differs")
    configurations: dict[str, dict[str, object]] = {}

    def add(selected: Sequence[Mapping[str, object]], weights: Sequence[float]) -> None:
        components = [
            {
                "term_id": str(term["term_id"]),
                "group": str(term["group"]),
                "weight": float(weight),
            }
            for term, weight in zip(selected, weights, strict=True)
        ]
        configuration_id = _configuration_id(components)
        configurations[configuration_id] = {
            "configuration_id": configuration_id,
            "components": components,
        }

    for term in terms:
        for weight in SINGLE_WEIGHT_GRID:
            add([term], [weight])
    for first, second in itertools.combinations(terms, 2):
        for first_weight in PAIR_WEIGHT_GRID:
            for second_weight in PAIR_WEIGHT_GRID:
                add([first, second], [first_weight, second_weight])
    return [configurations[key] for key in sorted(configurations)]


def build_cmvo_candidate_specs(
    *,
    base_records: pd.DataFrame,
    old_term_ids: set[str],
    configurations: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Attach the unchanged base or one frozen CMVO configuration."""

    accepted = base_records.copy()
    if "passes_source_auc_gates" not in accepted:
        raise ValueError("NEXT111 near-miss base flag is missing")
    accepted["passes_source_auc_gates"] = True
    return build_two_axis_candidate_specs(
        base_records=accepted,
        old_term_ids=old_term_ids,
        configurations=configurations,
    )


def materialize_cmvo_tail_terms(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Encode the three frozen clipped IQR tails reversibly through asinh."""

    columns: dict[str, np.ndarray] = {}
    terms: list[dict[str, object]] = []
    for raw_spec in FROZEN_TERM_SPECS:
        spec = dict(raw_spec)
        term_id = str(spec["term_id"])
        raw_feature = str(spec["raw_feature"])
        raw_support = str(spec["support_column"])
        if raw_feature not in features or raw_support not in features:
            raise ValueError("NEXT111 frozen CMVO feature is missing")
        center = float(spec["center"])
        scale = float(spec["scale"])
        cap = float(spec["clip_normalized"])
        if not all(math.isfinite(value) for value in (center, scale, cap)) or scale <= 0.0 or cap <= 0.0:
            raise ValueError("NEXT111 frozen CMVO calibration differs")
        raw = pd.to_numeric(features[raw_feature], errors="coerce").to_numpy(float)
        active = (
            features[raw_support].fillna(False).astype(bool).to_numpy()
            & np.isfinite(raw)
        )
        tail = np.zeros(len(features), dtype=float)
        tail[active] = np.clip(
            np.maximum(0.0, (raw[active] - center) / scale),
            0.0,
            cap,
        )
        encoded = np.sinh(tail)
        if not np.isfinite(encoded).all():
            raise ValueError("NEXT111 reversible CMVO encoding overflowed")
        feature_name = f"_{term_id}_encoded"
        support_name = f"_{term_id}_active"
        columns[feature_name] = encoded
        columns[support_name] = active
        terms.append(
            {
                "term_id": term_id,
                "feature": feature_name,
                "direction": 1,
                "transform": "asinh",
                "group": str(spec["group"]),
                "support_column": support_name,
                "center": 0.0,
                "scale": 1.0,
                "raw_feature": raw_feature,
                "physical_center": center,
                "physical_scale": scale,
                "raw_p995": float(spec["raw_p995"]),
                "clip_normalized": cap,
                "missing_policy": "OPTIONAL_GUARD_OFF_KEEP_BASE",
                "encoding": "asinh_sinh_exact_clipped_iqr_tail",
            }
        )
    extended = pd.concat(
        [features.reset_index(drop=True), pd.DataFrame(columns)], axis=1
    )
    return extended, terms


def run_cmvo_optional_search(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    next98_dir: Path,
    next98b_dir: Path,
    next108_dir: Path,
    next110_dir: Path,
    freeze_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only near-miss CMVO rescue search."""

    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        "next98": Path(next98_dir).resolve(),
        "next98b": Path(next98b_dir).resolve(),
        "next108": Path(next108_dir).resolve(),
        "next110": Path(next110_dir).resolve(),
    }
    target = Path(output_dir).resolve()
    paths = {
        "scigen_features": roots["scigen_features"] / SCIGEN_FEATURE_NAMES["discovery"],
        "scigen_endpoint": roots["scigen_endpoint"] / SCIGEN_ENDPOINT_NAME,
        "wyformer_features": roots["wyformer_features"] / WYFORMER_FEATURE_NAMES["discovery"],
        "wyformer_endpoint": roots["wyformer_endpoint"] / WYFORMER_ENDPOINT_NAME,
        "next98_manifest": roots["next98"] / NEXT98_MANIFEST_NAME,
        "next98_term_catalogue": roots["next98"] / NEXT98_CATALOGUE_NAME,
        "next98b_manifest": roots["next98b"] / NEXT98B_MANIFEST_NAME,
        "next98b_search_records": roots["next98b"] / NEXT98B_SEARCH_NAME,
        "next108_manifest": roots["next108"] / NEXT108_MANIFEST_NAME,
        "next110_manifest": roots["next110"] / NEXT110_MANIFEST_NAME,
        "next110_catalogue": roots["next110"] / NEXT110_CATALOGUE_NAME,
        "next110_scigen_features": roots["next110"] / NEXT110_FEATURE_FILES["scigen"],
        "next110_wyformer_features": roots["next110"] / NEXT110_FEATURE_FILES["wyformer"],
        "freeze": Path(freeze_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT111 discovery input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT111 formal input identity differs")

    next98_manifest = _read_json(paths["next98_manifest"])
    next98b_manifest = _read_json(paths["next98b_manifest"])
    next108_manifest = _read_json(paths["next108_manifest"])
    next110_manifest = _read_json(paths["next110_manifest"])
    if (
        next98_manifest.get("protocol") != NEXT98_PROTOCOL
        or next98_manifest.get("opened_validation_outputs_used") is not False
        or next98_manifest.get("scigen_replication_endpoint_opened") is not False
        or next98_manifest.get("wyformer_replication_endpoint_opened") is not False
        or next98b_manifest.get("protocol") != NEXT98B_PROTOCOL
        or next98b_manifest.get("opened_validation_outputs_used") is not False
        or next98b_manifest.get("scigen_replication_endpoint_opened") is not False
        or next98b_manifest.get("wyformer_replication_endpoint_opened") is not False
        or next108_manifest.get("protocol") != NEXT108_PROTOCOL
        or next108_manifest.get("passes_all_cross_source_discovery_gates") is not False
        or next108_manifest.get("freeze_authorized") is not False
        or next108_manifest.get("opened_validation_outputs_used") is not False
        or next108_manifest.get("scigen_replication_endpoint_opened") is not False
        or next108_manifest.get("wyformer_replication_endpoint_opened") is not False
        or next110_manifest.get("protocol") != NEXT110_PROTOCOL
        or next110_manifest.get("labels_opened") is not False
        or next110_manifest.get("endpoint_payloads_opened") is not False
        or next110_manifest.get("validation_geometry_opened") is not False
        or next110_manifest.get("replication_geometry_opened") is not False
        or next110_manifest.get("dft_values_used_by_features") is not False
    ):
        raise ValueError("NEXT111 prior provenance differs")
    next110_outputs = next110_manifest.get("outputs_sha256")
    if not isinstance(next110_outputs, Mapping) or any(
        next110_outputs.get(filename) != input_hashes[key]
        for filename, key in {
            NEXT110_CATALOGUE_NAME: "next110_catalogue",
            NEXT110_FEATURE_FILES["scigen"]: "next110_scigen_features",
            NEXT110_FEATURE_FILES["wyformer"]: "next110_wyformer_features",
        }.items()
    ):
        raise ValueError("NEXT111 NEXT110 output provenance differs")

    old_tables = {
        "scigen": pd.read_parquet(paths["scigen_features"]),
        "wyformer": pd.read_parquet(paths["wyformer_features"]),
    }
    new_tables = {
        "scigen": pd.read_parquet(paths["next110_scigen_features"]),
        "wyformer": pd.read_parquet(paths["next110_wyformer_features"]),
    }
    feature_tables: dict[str, pd.DataFrame] = {}
    for source in ("scigen", "wyformer"):
        old, new = old_tables[source], new_tables[source]
        if (
            old["material_id"].astype(str).duplicated().any()
            or new["material_id"].astype(str).duplicated().any()
            or set(NEXT110_FEATURE_COLUMNS) - set(new.columns)
        ):
            raise ValueError(f"NEXT111 {source} feature identity differs")
        merged = old.merge(
            new.loc[:, ["material_id", *NEXT110_FEATURE_COLUMNS]],
            on="material_id",
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(old) or len(merged) != len(new):
            raise ValueError(f"NEXT111 {source} feature row accounting differs")
        merged = merged.copy()
        merged["source_dataset"] = source
        if source == "scigen":
            merged["crystal_system"] = merged["lattice_class"].astype(str)
        merged["material_id"] = source + ":" + merged["material_id"].astype(str)
        feature_tables[source] = merged
    feature_combined = pd.concat(
        [feature_tables["scigen"], feature_tables["wyformer"]],
        ignore_index=True,
        sort=False,
    )

    tail_features, eligible_terms = materialize_cmvo_tail_terms(feature_combined)
    configurations = build_cmvo_guard_configurations(FROZEN_TERM_SPECS)
    extended_features, composite_terms, composite_mapping = (
        materialize_composite_guard_terms(
            features=tail_features,
            eligible_terms=eligible_terms,
            configurations=configurations,
        )
    )
    old_catalogue = _read_json(paths["next98_term_catalogue"])
    old_terms = old_catalogue.get("eligible_terms")
    if not isinstance(old_terms, list):
        raise ValueError("NEXT111 old term catalogue differs")
    old_term_ids = {str(term["term_id"]) for term in old_terms}
    all_base_records = pd.read_parquet(paths["next98b_search_records"])
    near_miss_bases = select_near_miss_bases(
        all_base_records, tolerance=NEAR_MISS_TOLERANCE
    )
    specs = build_cmvo_candidate_specs(
        base_records=near_miss_bases,
        old_term_ids=old_term_ids,
        configurations=configurations,
    )
    label_free_catalogue = {
        "protocol": PROTOCOL,
        "calibration_stage": "label_free_before_endpoint_join",
        "freeze_sha256": input_hashes["freeze"],
        "eligible_optional_terms": eligible_terms,
        "excluded_raw_features": list(EXCLUDED_TERM_SPECS),
        "single_weight_grid": list(SINGLE_WEIGHT_GRID),
        "pair_weight_grid": list(PAIR_WEIGHT_GRID),
        "guard_configurations": configurations,
        "configuration_count": len(configurations),
        "near_miss_tolerance": NEAR_MISS_TOLERANCE,
        "near_miss_base_count": len(near_miss_bases),
        "near_miss_base_key_sha256": hashlib.sha256(
            "\n".join(sorted(near_miss_bases["candidate_key"].astype(str))).encode("utf-8")
        ).hexdigest(),
        "candidate_count": len(specs),
        "candidate_key_sha256": hashlib.sha256(
            "\n".join(spec["candidate_key"] for spec in specs).encode("utf-8")
        ).hexdigest(),
        "optional_missing_policy": "OPTIONAL_GUARD_OFF_KEEP_BASE",
    }
    label_free_catalogue_sha256 = hashlib.sha256(
        _json_bytes(label_free_catalogue)
    ).hexdigest()
    if require_formal_inputs and (
        len(near_miss_bases) != EXPECTED_NEAR_MISS_BASES
        or len(configurations) != EXPECTED_CONFIGURATION_COUNT
        or len(specs) != EXPECTED_CANDIDATE_COUNT
    ):
        raise ValueError("NEXT111 frozen candidate count differs")

    # Discovery endpoints are opened only after the complete catalogue is hashed.
    scigen_endpoints = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_endpoints = pd.read_parquet(paths["wyformer_endpoint"])
    if (
        scigen_endpoints["material_id"].astype(str).duplicated().any()
        or wyformer_endpoints["material_id"].astype(str).duplicated().any()
    ):
        raise ValueError("NEXT111 discovery endpoint identities are duplicated")
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
                    "_endpoint_numeric": _endpoint_numeric(
                        wyformer_endpoints["endpoint_stratum"]
                    ),
                }
            ),
        ],
        ignore_index=True,
    )
    combined = extended_features.merge(
        endpoint_frame, on="material_id", how="inner", validate="one_to_one"
    )
    if len(combined) != len(extended_features) or len(combined) != len(endpoint_frame):
        raise ValueError("NEXT111 endpoint row accounting differs")
    endpoint = pd.to_numeric(combined.pop("_endpoint_numeric"), errors="coerce").to_numpy(float)
    if not np.isfinite(endpoint).all():
        raise ValueError("NEXT111 endpoint conversion differs")

    started = time.perf_counter()
    result = search_optional_guard_laws(
        features=combined,
        endpoint=endpoint,
        old_terms=old_terms,
        optional_terms=composite_terms,
        candidate_specs=specs,
    )
    elapsed = time.perf_counter() - started
    _decorate_result(
        result=result,
        mapping=composite_mapping,
        eligible_terms=eligible_terms,
    )
    result["selected"]["formula"]["kind"] = (
        "base_nonnegative_hinge_sum_plus_up_to_two_cmvo_obstruction_terms"
    )
    selected = result["selected"]
    passes = bool(selected["record"]["passes_all_discovery_gates"])

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next103_dobvr_optional_guard_search.py": repository_root / "src/next103_dobvr_optional_guard_search.py",
        "src/next107_two_axis_cmvf_guard_search.py": repository_root / "src/next107_two_axis_cmvf_guard_search.py",
        "src/next108_near_miss_cmvf_rescue.py": repository_root / "src/next108_near_miss_cmvf_rescue.py",
        "src/next111_cmvo_optional_search.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    output_paths: list[Path] = []
    try:
        catalogue = {
            **label_free_catalogue,
            "label_free_catalogue_sha256": label_free_catalogue_sha256,
        }
        evaluation = {
            "protocol": PROTOCOL,
            "evaluation_mode": "cross_source_discovery_near_miss_cmvo_rescue",
            "rows": {
                "scigen": int(len(feature_tables["scigen"])),
                "wyformer": int(len(feature_tables["wyformer"])),
                "total": int(len(combined)),
            },
            "near_miss_base_count": len(near_miss_bases),
            "configuration_count": len(configurations),
            "candidate_count": int(result["candidate_count"]),
            "elapsed_seconds": elapsed,
            "safe_gates": dict(DEFAULT_GATES),
            "source_auc_gates": dict(AUC_GATES),
            "broad_min_severe_precision_lower": BROAD_MIN_PRECISION_LOWER,
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
            "adaptive_discovery_search": True,
            "requires_unopened_internal_validation_before_claim": True,
        }
        catalogue_path = staging / CATALOGUE_NAME
        evaluation_path = staging / EVALUATION_NAME
        search_path = staging / SEARCH_NAME
        _write_json(catalogue_path, catalogue)
        _write_json(evaluation_path, evaluation)
        pd.DataFrame(result["candidate_records"]).to_parquet(search_path, index=False)
        output_paths.extend([catalogue_path, evaluation_path, search_path])
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "label_free_catalogue_sha256": label_free_catalogue_sha256,
            "near_miss_base_count": len(near_miss_bases),
            "configuration_count": len(configurations),
            "candidate_count": int(result["candidate_count"]),
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "adaptive_discovery_search": True,
            "requires_unopened_internal_validation_before_claim": True,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
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
            raise RuntimeError("NEXT111 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT111 source changed before publication")
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
    parser.add_argument("--next98-dir", type=Path, required=True)
    parser.add_argument("--next98b-dir", type=Path, required=True)
    parser.add_argument("--next108-dir", type=Path, required=True)
    parser.add_argument("--next110-dir", type=Path, required=True)
    parser.add_argument("--freeze-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = run_cmvo_optional_search(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        next98_dir=args.next98_dir,
        next98b_dir=args.next98b_dir,
        next108_dir=args.next108_dir,
        next110_dir=args.next110_dir,
        freeze_path=args.freeze_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
