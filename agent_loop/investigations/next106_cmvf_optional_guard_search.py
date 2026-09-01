#!/usr/bin/env python3
"""Cross-source discovery search for one optional NEXT104 CMVF guard."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping

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
    build_optional_guard_candidate_specs,
    calibrate_optional_terms,
    compose_optional_guard_score,
    search_optional_guard_laws,
    select_safe_and_diagnostic_once,
)
from src.next105_cross_source_cmvf_features import (
    FEATURE_COLUMNS as NEXT105_FEATURE_COLUMNS,
    FEATURE_FILES as NEXT105_FEATURE_FILES,
    MANIFEST_NAME as NEXT105_MANIFEST_NAME,
    PROTOCOL as NEXT105_PROTOCOL,
)


PROTOCOL = "2026-08-04-next106-cmvf-optional-guard-search-v1"
MIN_SOURCE_COVERAGE = 0.15
EXPECTED_AUC_PASSING_BASES = 67
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT106_OPTIONAL_TERM_CATALOGUE.json"
EVALUATION_NAME = "NEXT106_DISCOVERY_EVALUATION.json"
SEARCH_NAME = "next106_optional_guard_candidate_search.parquet"


def _template(mode: str, metric: str) -> dict[str, object]:
    feature = f"cmvf_{mode}_{metric}"
    return {
        "term_id": f"{feature}__high",
        "feature": feature,
        "direction": 1,
        "transform": "log1p_nonnegative",
        "group": f"cmvf_{mode}",
        "support_column": f"cmvf_{mode}_supported",
    }


OPTIONAL_TERM_TEMPLATES = tuple(
    _template(mode, metric)
    for mode in ("core", "expanded")
    for metric in ("reallocation", "overload", "log_scale_mismatch")
)

# NEXT105 identities are filled only after its atomic discovery-only publication.
# The formal runner rejects placeholders, so no endpoint search can run early.
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
    "design": "a340d68b9277102ae97ab0a736f882064dd01bce9c0019f1c50130aec5a0854e",
}


def run_cmvf_optional_guard_search(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    next98_dir: Path,
    next98b_dir: Path,
    next105_dir: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only CMVF optional-guard search."""

    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        "next98": Path(next98_dir).resolve(),
        "next98b": Path(next98b_dir).resolve(),
        "next105": Path(next105_dir).resolve(),
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
        "design": Path(design_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT106 discovery input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT106 formal input identity differs")

    next98_manifest = _read_json(paths["next98_manifest"])
    next98b_manifest = _read_json(paths["next98b_manifest"])
    next105_manifest = _read_json(paths["next105_manifest"])
    if (
        next98_manifest.get("protocol") != NEXT98_PROTOCOL
        or next98_manifest.get("opened_validation_outputs_used") is not False
        or next98_manifest.get("scigen_replication_endpoint_opened") is not False
        or next98_manifest.get("wyformer_replication_endpoint_opened") is not False
        or next98b_manifest.get("protocol") != NEXT98B_PROTOCOL
        or next98b_manifest.get("passes_all_cross_source_discovery_gates") is not False
        or next98b_manifest.get("opened_validation_outputs_used") is not False
        or next98b_manifest.get("scigen_replication_endpoint_opened") is not False
        or next98b_manifest.get("wyformer_replication_endpoint_opened") is not False
    ):
        raise ValueError("NEXT106 prior search provenance differs")
    next105_outputs = next105_manifest.get("outputs_sha256")
    if (
        next105_manifest.get("protocol") != NEXT105_PROTOCOL
        or next105_manifest.get("partitions_read") != ["discovery"]
        or next105_manifest.get("labels_opened") is not False
        or next105_manifest.get("endpoint_payloads_opened") is not False
        or next105_manifest.get("validation_geometry_opened") is not False
        or next105_manifest.get("replication_geometry_opened") is not False
        or next105_manifest.get("dft_values_used_by_features") is not False
        or next105_manifest.get("solver_thread_environment")
        != {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        or not isinstance(next105_outputs, Mapping)
        or next105_outputs.get(NEXT105_FEATURE_FILES["scigen"])
        != input_hashes["next105_scigen_features"]
        or next105_outputs.get(NEXT105_FEATURE_FILES["wyformer"])
        != input_hashes["next105_wyformer_features"]
    ):
        raise ValueError("NEXT106 NEXT105 feature provenance differs")

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
            raise ValueError(f"NEXT106 {source} feature identity differs")
        merged = old.merge(
            new.loc[:, ["material_id", *NEXT105_FEATURE_COLUMNS]],
            on="material_id",
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(old) or len(merged) != len(new):
            raise ValueError(f"NEXT106 {source} feature row accounting differs")
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
    optional_terms, excluded_optional_terms = calibrate_optional_terms(
        feature_combined,
        templates=OPTIONAL_TERM_TEMPLATES,
        min_source_coverage=MIN_SOURCE_COVERAGE,
        min_unique_values=8,
    )

    # Endpoint payloads are opened only after the label-free catalogue is frozen.
    scigen_endpoints = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_endpoints = pd.read_parquet(paths["wyformer_endpoint"])
    if (
        scigen_endpoints["material_id"].astype(str).duplicated().any()
        or wyformer_endpoints["material_id"].astype(str).duplicated().any()
    ):
        raise ValueError("NEXT106 discovery endpoint identities are duplicated")
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
    combined = feature_combined.merge(
        endpoint_frame, on="material_id", how="inner", validate="one_to_one"
    )
    if len(combined) != len(feature_combined) or len(combined) != len(endpoint_frame):
        raise ValueError("NEXT106 endpoint row accounting differs")
    endpoint = pd.to_numeric(combined.pop("_endpoint_numeric"), errors="coerce").to_numpy(float)
    if not np.isfinite(endpoint).all():
        raise ValueError("NEXT106 endpoint conversion differs")

    old_catalogue = _read_json(paths["next98_term_catalogue"])
    old_terms = old_catalogue.get("eligible_terms")
    if not isinstance(old_terms, list):
        raise ValueError("NEXT106 old term catalogue differs")
    old_term_ids = {str(term["term_id"]) for term in old_terms}
    base_records = pd.read_parquet(paths["next98b_search_records"])
    passing_bases = int(base_records["passes_source_auc_gates"].eq(True).sum())
    if require_formal_inputs and passing_bases != EXPECTED_AUC_PASSING_BASES:
        raise ValueError("NEXT106 AUC-passing base count differs")
    specs = build_optional_guard_candidate_specs(
        base_records=base_records,
        old_term_ids=old_term_ids,
        optional_terms=optional_terms,
    )
    started = time.perf_counter()
    result = search_optional_guard_laws(
        features=combined,
        endpoint=endpoint,
        old_terms=old_terms,
        optional_terms=optional_terms,
        candidate_specs=specs,
    )
    elapsed = time.perf_counter() - started
    selected = result["selected"]
    selected["formula"]["kind"] = "base_nonnegative_hinge_sum_plus_one_optional_cmvf_guard"
    passes = bool(selected["record"]["passes_all_discovery_gates"])

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next103_dobvr_optional_guard_search.py": repository_root / "src/next103_dobvr_optional_guard_search.py",
        "src/next106_cmvf_optional_guard_search.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    output_paths: list[Path] = []
    try:
        catalogue = {
            "protocol": PROTOCOL,
            "calibration_stage": "label_free_before_endpoint_join",
            "min_source_coverage": MIN_SOURCE_COVERAGE,
            "optional_weight_grid": list(OPTIONAL_WEIGHT_GRID),
            "expected_auc_passing_bases": EXPECTED_AUC_PASSING_BASES,
            "observed_auc_passing_bases": passing_bases,
            "templates": list(OPTIONAL_TERM_TEMPLATES),
            "eligible_optional_terms": optional_terms,
            "excluded_optional_terms": excluded_optional_terms,
            "candidate_count": len(specs),
            "candidate_grammar": "each AUC-passing NEXT98b base plus zero or one optional CMVF guard",
            "optional_missing_policy": "OPTIONAL_GUARD_OFF_KEEP_BASE",
        }
        evaluation = {
            "protocol": PROTOCOL,
            "evaluation_mode": "cross_source_discovery_only_optional_cmvf_guard",
            "rows": {
                "scigen": int(len(feature_tables["scigen"])),
                "wyformer": int(len(feature_tables["wyformer"])),
                "total": int(len(combined)),
            },
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
            "candidate_count": int(result["candidate_count"]),
            "eligible_optional_term_count": len(optional_terms),
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
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
            raise RuntimeError("NEXT106 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT106 source changed before publication")
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
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = run_cmvf_optional_guard_search(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        next98_dir=args.next98_dir,
        next98b_dir=args.next98b_dir,
        next105_dir=args.next105_dir,
        design_path=args.design_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "OPTIONAL_TERM_TEMPLATES",
    "OPTIONAL_WEIGHT_GRID",
    "PROTOCOL",
    "build_optional_guard_candidate_specs",
    "calibrate_optional_terms",
    "compose_optional_guard_score",
    "run_cmvf_optional_guard_search",
    "select_safe_and_diagnostic_once",
    "search_optional_guard_laws",
]


if __name__ == "__main__":
    raise SystemExit(main())
