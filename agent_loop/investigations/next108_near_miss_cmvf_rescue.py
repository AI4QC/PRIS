#!/usr/bin/env python3
"""Rescue near-miss cross-source bases with the frozen expanded CMVF pair."""

from __future__ import annotations

import argparse
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
from src.next103_dobvr_optional_guard_search import (
    OPTIONAL_WEIGHT_GRID,
    calibrate_optional_terms,
    search_optional_guard_laws,
)
from src.next105_cross_source_cmvf_features import (
    FEATURE_COLUMNS as NEXT105_FEATURE_COLUMNS,
    FEATURE_FILES as NEXT105_FEATURE_FILES,
    MANIFEST_NAME as NEXT105_MANIFEST_NAME,
    PROTOCOL as NEXT105_PROTOCOL,
)
from src.next106_cmvf_optional_guard_search import (
    MIN_SOURCE_COVERAGE,
    OPTIONAL_TERM_TEMPLATES,
)
from src.next107_two_axis_cmvf_guard_search import (
    CATALOGUE_NAME as NEXT107_CATALOGUE_NAME,
    EVALUATION_NAME as NEXT107_EVALUATION_NAME,
    MANIFEST_NAME as NEXT107_MANIFEST_NAME,
    PROTOCOL as NEXT107_PROTOCOL,
    SEARCH_NAME as NEXT107_SEARCH_NAME,
    _decorate_result,
    build_two_axis_candidate_specs,
    build_two_axis_guard_configurations,
    materialize_composite_guard_terms,
)


PROTOCOL = "2026-08-04-next108-near-miss-cmvf-rescue-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT108_RESCUE_CATALOGUE.json"
EVALUATION_NAME = "NEXT108_DISCOVERY_EVALUATION.json"
SEARCH_NAME = "next108_near_miss_rescue_search.parquet"
NEAR_MISS_TOLERANCE = 0.01
EXPECTED_NEAR_MISS_BASES = 353
EXPECTED_CONFIGURATION_COUNT = 25
EXPECTED_CANDIDATE_COUNT = 9_178
RESCUE_TERM_IDS = (
    "cmvf_expanded_overload__high",
    "cmvf_expanded_reallocation__high",
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
    "next105_manifest": "a2340605d9e8f97165ed8fad10c33f401dc17cdade6c5552e0867923fe5002e3",
    "next105_scigen_features": "d4d7974439ea9a39cf9db0bf458c13253f80e1baf5d9faf31594182473e2a90a",
    "next105_wyformer_features": "299f5ab2060aebaa4c5915aac7543fadc16728ffc055a3bd341373d820aeba99",
    "next107_manifest": "388099bf9513e252488a301be1814ade74ad227376dcf740ab833b2666eca9ed",
    "next107_catalogue": "731e2021b716c084ea106ae83dcbf9eee49f52d74d927e6672cbf114ee990777",
    "next107_evaluation": "65fe4767d52c1807d288e2d6b3e2a1a9d86c93f8c791f57cd2067e5e068d6f82",
    "next107_search_records": "bc06ee907f223ab800a623f44ef2b924260b51fcdb080da2161fbced0a684a53",
    "design": "0e38cd3ba3cc632d64b2d96eb90c245c26fdbe0570d5afc4def0ec563c30d87f",
}


def select_near_miss_bases(
    base_records: pd.DataFrame, *, tolerance: float = NEAR_MISS_TOLERANCE
) -> pd.DataFrame:
    """Select bases no farther than one frozen margin below every AUC gate."""

    thresholds = {
        "scigen_pooled_auc": float(AUC_GATES["pooled_extreme_auc"]),
        "scigen_macro_auc": float(AUC_GATES["macro_lattice_auc"]),
        "scigen_worst_auc": float(AUC_GATES["worst_lattice_auc"]),
        "wyformer_pooled_auc": float(AUC_GATES["pooled_extreme_auc"]),
        "wyformer_macro_auc": float(AUC_GATES["macro_lattice_auc"]),
        "wyformer_worst_auc": float(AUC_GATES["worst_lattice_auc"]),
    }
    required = {
        "candidate_key",
        "term_ids_json",
        "weights_json",
        "passes_source_auc_gates",
        *thresholds,
    }
    margin = float(tolerance)
    if (
        required - set(base_records.columns)
        or not math.isfinite(margin)
        or margin < 0.0
    ):
        raise ValueError("NEXT108 near-miss base schema differs")
    keep = np.ones(len(base_records), dtype=bool)
    for column, threshold in thresholds.items():
        values = pd.to_numeric(base_records[column], errors="coerce").to_numpy(float)
        keep &= np.isfinite(values) & (values >= threshold - margin - 1e-12)
    return base_records.loc[keep].copy().reset_index(drop=True)


def build_rescue_configurations(
    eligible_terms: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Enumerate only the 5x5 expanded overload/reallocation weight pair."""

    by_id = {str(term["term_id"]): dict(term) for term in eligible_terms}
    if any(term_id not in by_id for term_id in RESCUE_TERM_IDS):
        raise ValueError("NEXT108 rescue CMVF terms are missing")
    configurations = build_two_axis_guard_configurations(
        [by_id[term_id] for term_id in RESCUE_TERM_IDS]
    )
    pairs = [
        configuration
        for configuration in configurations
        if len(configuration["components"]) == 2
    ]
    if any(
        tuple(component["term_id"] for component in configuration["components"])
        != RESCUE_TERM_IDS
        for configuration in pairs
    ):
        raise ValueError("NEXT108 rescue pair identity differs")
    return pairs


def build_rescue_candidate_specs(
    *,
    base_records: pd.DataFrame,
    old_term_ids: set[str],
    configurations: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Enumerate every base that already passed the frozen near-miss filter."""

    accepted = base_records.copy()
    if "passes_source_auc_gates" not in accepted:
        raise ValueError("NEXT108 near-miss base flag is missing")
    accepted["passes_source_auc_gates"] = True
    return build_two_axis_candidate_specs(
        base_records=accepted,
        old_term_ids=old_term_ids,
        configurations=configurations,
    )


def run_near_miss_cmvf_rescue(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    next98_dir: Path,
    next98b_dir: Path,
    next105_dir: Path,
    next107_dir: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only near-miss expanded-CMVF rescue."""

    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        "next98": Path(next98_dir).resolve(),
        "next98b": Path(next98b_dir).resolve(),
        "next105": Path(next105_dir).resolve(),
        "next107": Path(next107_dir).resolve(),
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
        "next105_manifest": roots["next105"] / NEXT105_MANIFEST_NAME,
        "next105_scigen_features": roots["next105"] / NEXT105_FEATURE_FILES["scigen"],
        "next105_wyformer_features": roots["next105"] / NEXT105_FEATURE_FILES["wyformer"],
        "next107_manifest": roots["next107"] / NEXT107_MANIFEST_NAME,
        "next107_catalogue": roots["next107"] / NEXT107_CATALOGUE_NAME,
        "next107_evaluation": roots["next107"] / NEXT107_EVALUATION_NAME,
        "next107_search_records": roots["next107"] / NEXT107_SEARCH_NAME,
        "design": Path(design_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT108 discovery input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT108 formal input identity differs")

    next98_manifest = _read_json(paths["next98_manifest"])
    next98b_manifest = _read_json(paths["next98b_manifest"])
    next105_manifest = _read_json(paths["next105_manifest"])
    next107_manifest = _read_json(paths["next107_manifest"])
    next107_catalogue = _read_json(paths["next107_catalogue"])
    if (
        next98_manifest.get("protocol") != NEXT98_PROTOCOL
        or next98_manifest.get("opened_validation_outputs_used") is not False
        or next98_manifest.get("scigen_replication_endpoint_opened") is not False
        or next98_manifest.get("wyformer_replication_endpoint_opened") is not False
        or next98b_manifest.get("protocol") != NEXT98B_PROTOCOL
        or next98b_manifest.get("opened_validation_outputs_used") is not False
        or next98b_manifest.get("scigen_replication_endpoint_opened") is not False
        or next98b_manifest.get("wyformer_replication_endpoint_opened") is not False
        or next105_manifest.get("protocol") != NEXT105_PROTOCOL
        or next105_manifest.get("labels_opened") is not False
        or next105_manifest.get("endpoint_payloads_opened") is not False
        or next105_manifest.get("validation_geometry_opened") is not False
        or next105_manifest.get("replication_geometry_opened") is not False
        or next105_manifest.get("dft_values_used_by_features") is not False
        or next107_manifest.get("protocol") != NEXT107_PROTOCOL
        or next107_manifest.get("candidate_count") != 12_127
        or next107_manifest.get("passes_all_cross_source_discovery_gates") is not False
        or next107_manifest.get("freeze_authorized") is not False
        or next107_manifest.get("opened_validation_outputs_used") is not False
        or next107_manifest.get("scigen_replication_endpoint_opened") is not False
        or next107_manifest.get("wyformer_replication_endpoint_opened") is not False
        or next107_catalogue.get("protocol") != NEXT107_PROTOCOL
        or not isinstance(next107_catalogue.get("eligible_optional_terms"), list)
    ):
        raise ValueError("NEXT108 prior provenance differs")
    for manifest, prefix, names in (
        (
            next105_manifest,
            "next105",
            {
                NEXT105_FEATURE_FILES["scigen"]: "next105_scigen_features",
                NEXT105_FEATURE_FILES["wyformer"]: "next105_wyformer_features",
            },
        ),
        (
            next107_manifest,
            "next107",
            {
                NEXT107_CATALOGUE_NAME: "next107_catalogue",
                NEXT107_EVALUATION_NAME: "next107_evaluation",
                NEXT107_SEARCH_NAME: "next107_search_records",
            },
        ),
    ):
        outputs = manifest.get("outputs_sha256")
        if not isinstance(outputs, Mapping) or any(
            outputs.get(filename) != input_hashes[key]
            for filename, key in names.items()
        ):
            raise ValueError(f"NEXT108 {prefix} output provenance differs")

    old_tables = {
        "scigen": pd.read_parquet(paths["scigen_features"]),
        "wyformer": pd.read_parquet(paths["wyformer_features"]),
    }
    new_tables = {
        "scigen": pd.read_parquet(paths["next105_scigen_features"]),
        "wyformer": pd.read_parquet(paths["next105_wyformer_features"]),
    }
    feature_tables: dict[str, pd.DataFrame] = {}
    for source in ("scigen", "wyformer"):
        old, new = old_tables[source], new_tables[source]
        if (
            old["material_id"].astype(str).duplicated().any()
            or new["material_id"].astype(str).duplicated().any()
            or set(NEXT105_FEATURE_COLUMNS) - set(new.columns)
        ):
            raise ValueError(f"NEXT108 {source} feature identity differs")
        merged = old.merge(
            new.loc[:, ["material_id", *NEXT105_FEATURE_COLUMNS]],
            on="material_id",
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(old) or len(merged) != len(new):
            raise ValueError(f"NEXT108 {source} feature row accounting differs")
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
    eligible_terms, excluded_terms = calibrate_optional_terms(
        feature_combined,
        templates=OPTIONAL_TERM_TEMPLATES,
        min_source_coverage=MIN_SOURCE_COVERAGE,
        min_unique_values=8,
    )
    if eligible_terms != next107_catalogue["eligible_optional_terms"]:
        raise ValueError("NEXT108 label-free CMVF calibration differs from NEXT107")
    configurations = build_rescue_configurations(eligible_terms)
    extended_features, composite_terms, composite_mapping = (
        materialize_composite_guard_terms(
            features=feature_combined,
            eligible_terms=eligible_terms,
            configurations=configurations,
        )
    )

    # Endpoint payloads are opened only after the label-free rescue grammar exists.
    scigen_endpoints = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_endpoints = pd.read_parquet(paths["wyformer_endpoint"])
    if (
        scigen_endpoints["material_id"].astype(str).duplicated().any()
        or wyformer_endpoints["material_id"].astype(str).duplicated().any()
    ):
        raise ValueError("NEXT108 discovery endpoint identities are duplicated")
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
        raise ValueError("NEXT108 endpoint row accounting differs")
    endpoint = pd.to_numeric(combined.pop("_endpoint_numeric"), errors="coerce").to_numpy(float)
    if not np.isfinite(endpoint).all():
        raise ValueError("NEXT108 endpoint conversion differs")

    old_catalogue = _read_json(paths["next98_term_catalogue"])
    old_terms = old_catalogue.get("eligible_terms")
    if not isinstance(old_terms, list):
        raise ValueError("NEXT108 old term catalogue differs")
    old_term_ids = {str(term["term_id"]) for term in old_terms}
    all_base_records = pd.read_parquet(paths["next98b_search_records"])
    near_miss_bases = select_near_miss_bases(
        all_base_records, tolerance=NEAR_MISS_TOLERANCE
    )
    specs = build_rescue_candidate_specs(
        base_records=near_miss_bases,
        old_term_ids=old_term_ids,
        configurations=configurations,
    )
    if require_formal_inputs and (
        len(near_miss_bases) != EXPECTED_NEAR_MISS_BASES
        or len(configurations) != EXPECTED_CONFIGURATION_COUNT
        or len(specs) != EXPECTED_CANDIDATE_COUNT
    ):
        raise ValueError("NEXT108 frozen candidate count differs")

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
    selected = result["selected"]
    passes = bool(selected["record"]["passes_all_discovery_gates"])

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next103_dobvr_optional_guard_search.py": repository_root / "src/next103_dobvr_optional_guard_search.py",
        "src/next106_cmvf_optional_guard_search.py": repository_root / "src/next106_cmvf_optional_guard_search.py",
        "src/next107_two_axis_cmvf_guard_search.py": repository_root / "src/next107_two_axis_cmvf_guard_search.py",
        "src/next108_near_miss_cmvf_rescue.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    output_paths: list[Path] = []
    try:
        catalogue = {
            "protocol": PROTOCOL,
            "calibration_stage": "label_free_before_endpoint_join",
            "near_miss_tolerance": NEAR_MISS_TOLERANCE,
            "near_miss_base_count": len(near_miss_bases),
            "near_miss_definition": {
                "pooled_auc_min": AUC_GATES["pooled_extreme_auc"] - NEAR_MISS_TOLERANCE,
                "macro_auc_min": AUC_GATES["macro_lattice_auc"] - NEAR_MISS_TOLERANCE,
                "worst_auc_min": AUC_GATES["worst_lattice_auc"] - NEAR_MISS_TOLERANCE,
            },
            "optional_weight_grid": list(OPTIONAL_WEIGHT_GRID),
            "eligible_optional_terms": eligible_terms,
            "excluded_optional_terms": excluded_terms,
            "rescue_term_ids": list(RESCUE_TERM_IDS),
            "guard_configurations": configurations,
            "configuration_count": len(configurations),
            "candidate_count": len(specs),
            "candidate_grammar": "each 0.01-AUC-near-miss NEXT98b base plus zero or the expanded overload/reallocation pair",
            "optional_missing_policy": "OPTIONAL_GUARD_OFF_KEEP_BASE",
        }
        evaluation = {
            "protocol": PROTOCOL,
            "evaluation_mode": "cross_source_discovery_near_miss_expanded_cmvf_pair_rescue",
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
            raise RuntimeError("NEXT108 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT108 source changed before publication")
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
    parser.add_argument("--next105-dir", type=Path, required=True)
    parser.add_argument("--next107-dir", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = run_near_miss_cmvf_rescue(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        next98_dir=args.next98_dir,
        next98b_dir=args.next98b_dir,
        next105_dir=args.next105_dir,
        next107_dir=args.next107_dir,
        design_path=args.design_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "PROTOCOL",
    "build_rescue_candidate_specs",
    "build_rescue_configurations",
    "run_near_miss_cmvf_rescue",
    "select_near_miss_bases",
]


if __name__ == "__main__":
    raise SystemExit(main())
