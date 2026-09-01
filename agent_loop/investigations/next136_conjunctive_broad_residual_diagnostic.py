#!/usr/bin/env python3
"""Diagnose exact BROAD residuals of published NEXT135 SAFE12 candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd

import src.next130_coordination_protection_search as n130
import src.next134_compactness_protection_search as n134
import src.next135_conjunctive_compactness_search as n135
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next128_broad_residual_diagnostic import diagnose_broad_threshold_tables
from src.next87_scigen_sparse_law_search import _term_risk, assign_group_folds
from src.next98_cross_source_discovery_search import (
    _pauling_baseline,
    _threshold_tables,
    build_source_fold_cells,
)


PROTOCOL = "2026-08-08-next136-conjunctive-broad-residual-diagnostic-v1"
MANIFEST_NAME = "MANIFEST.json"
DIAGNOSTIC_NAME = "NEXT136_CONJUNCTIVE_BROAD_RESIDUAL_DIAGNOSTIC.json"
PER_CANDIDATE_NAME = "next136_conjunctive_broad_residual_by_candidate.parquet"
EXPECTED_DESIGN_SHA256 = "408d78d234a6c0e7e92689947af5084e154a122f0335d71093d9a78652db1377"
EXPECTED_NEXT135_MANIFEST_SHA256 = "f77fef70a1c01b6e335c7938556813cbfcb42d3ac5b605f662bf48e0d772397c"
EXPECTED_SAFE_CANDIDATE_COUNT = 119
EXPECTED_INPUT_SHA256 = {
    **n135.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next135_manifest": EXPECTED_NEXT135_MANIFEST_SHA256,
    "next135_catalogue": "beee64d4c0ab86a3a0038b86e0a9dec7846fcb0ebaa5dbef4df66fad25aaafac",
    "next135_evaluation": "985fa3f3dd8fb9a2395c86bbe467885e3c5523750a032fc3a749969c14d87488",
    "next135_search_records": "a60fdc4a1611f54303cdc484f0792e4de7cf7808c1efe9149f4b898d2bc3b35b",
}


def select_safe_candidates(records: pd.DataFrame) -> pd.DataFrame:
    """Select only published SAFE12 candidates and preserve their identities."""

    required = {
        "candidate_key",
        "safe_threshold",
        "passes_safe_all_cells",
        "conjunctive_term_ids_json",
        "conjunctive_weights_json",
        "conjunctive_term_count",
    }
    if required - set(records.columns) or records["candidate_key"].astype(str).duplicated().any():
        raise ValueError("NEXT136 published candidate schema differs")
    selected = records.loc[records["passes_safe_all_cells"].fillna(False).astype(bool)].copy()
    thresholds = pd.to_numeric(selected["safe_threshold"], errors="coerce")
    if selected.empty or not np.isfinite(thresholds.to_numpy(float)).all():
        raise ValueError("NEXT136 published safe threshold differs")
    selected["safe_threshold"] = thresholds
    for _, row in selected.iterrows():
        term_ids = json.loads(str(row["conjunctive_term_ids_json"]))
        weights = json.loads(str(row["conjunctive_weights_json"]))
        if not isinstance(term_ids, list) or not isinstance(weights, list) or len(term_ids) != len(weights) or len(term_ids) != int(row["conjunctive_term_count"]):
            raise ValueError("NEXT136 published conjunctive identity differs")
    return selected.sort_values("candidate_key").reset_index(drop=True)


def _closest(frame: pd.DataFrame) -> pd.Series:
    return frame.sort_values(
        ["failed_constraint_count", "normalized_shortfall_sum", "best_threshold", "candidate_key"]
    ).iloc[0]


def summarize_groups(records: pd.DataFrame) -> dict[str, object]:
    """Summarize closest BROAD residual by term count and exact configuration."""

    def record(frame: pd.DataFrame) -> dict[str, object]:
        best = _closest(frame)
        minimum_count = int(best["failed_constraint_count"])
        return {
            "candidate_count": int(len(frame)),
            "minimum_failed_constraint_count": minimum_count,
            "minimum_normalized_shortfall_sum_at_best_count": float(
                frame.loc[
                    frame["failed_constraint_count"].eq(minimum_count),
                    "normalized_shortfall_sum",
                ].min()
            ),
            "closest_candidate_key": str(best["candidate_key"]),
        }

    by_count = {
        str(int(count)): record(frame)
        for count, frame in records.groupby("conjunctive_term_count", sort=True)
    }
    by_configuration: dict[str, object] = {}
    for (term_ids, weights), frame in records.groupby(
        ["conjunctive_term_ids_json", "conjunctive_weights_json"], sort=True
    ):
        by_configuration[f"{term_ids}@{weights}"] = record(frame)
    return {
        "by_conjunctive_term_count": by_count,
        "by_configuration": by_configuration,
    }


def _paths(
    roots: Mapping[str, Path], next135_freeze_path: Path, design_path: Path
) -> dict[str, Path]:
    paths = n135._paths(roots, next135_freeze_path)
    paths.update(
        {
            "design": design_path,
            "next135_manifest": roots["next135"] / n135.MANIFEST_NAME,
            "next135_catalogue": roots["next135"] / n135.CATALOGUE_NAME,
            "next135_evaluation": roots["next135"] / n135.EVALUATION_NAME,
            "next135_search_records": roots["next135"] / n135.SEARCH_NAME,
        }
    )
    return paths


def run_conjunctive_broad_residual_diagnostic(
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
    next135_dir: Path,
    next135_freeze_path: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Reconstruct and diagnose all published NEXT135 SAFE12 candidates."""

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
                (135, next135_dir),
            )
        },
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots,
        Path(next135_freeze_path).resolve(),
        Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT136 diagnostic input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            key
            for key in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(key) != EXPECTED_INPUT_SHA256.get(key)
        )
        raise ValueError(f"NEXT136 formal input identity differs: {differing}")

    manifest135 = json.loads(paths["next135_manifest"].read_text())
    outputs135 = manifest135.get("outputs_sha256")
    expected_outputs = {
        n135.CATALOGUE_NAME: "next135_catalogue",
        n135.EVALUATION_NAME: "next135_evaluation",
        n135.SEARCH_NAME: "next135_search_records",
    }
    if (
        manifest135.get("protocol") != n135.PROTOCOL
        or manifest135.get("passes_all_cross_source_discovery_gates") is not False
        or manifest135.get("opened_validation_outputs_used") is not False
        or manifest135.get("scigen_replication_endpoint_opened") is not False
        or manifest135.get("wyformer_replication_endpoint_opened") is not False
        or manifest135.get("dft_values_used_by_executable_formula") is not False
        or not isinstance(outputs135, Mapping)
        or any(outputs135.get(filename) != input_hashes[key] for filename, key in expected_outputs.items())
    ):
        raise ValueError("NEXT136 prior provenance differs")

    extended, _, old_terms, mhcr_terms = n130._join_label_free_features(paths)
    compact_frames: list[pd.DataFrame] = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next133_{source}_features"]).copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        compact_frames.append(table)
    compact = pd.concat(compact_frames, ignore_index=True)
    extended = extended.merge(compact, on="material_id", how="inner", validate="one_to_one")
    conjunctive = n135.materialize_conjunctive_features(extended)
    extended = pd.concat(
        [extended.reset_index(drop=True), conjunctive.reset_index(drop=True)], axis=1
    )

    physical_terms = [*old_terms, *mhcr_terms]
    physical_ids = {str(term["term_id"]) for term in physical_terms}
    all_bases = n130.n127.select_next125_bases(
        pd.read_parquet(paths["next125_search_records"])
    )
    bases = n135.n132.select_extended_bases(
        pd.read_parquet(paths["next130_search_records"]), all_bases
    )
    specs = n135.build_candidate_specs(bases=bases, physical_term_ids=physical_ids)
    if require_formal_inputs and len(specs) != n135.EXPECTED_CANDIDATE_COUNT:
        raise ValueError("NEXT136 frozen candidate universe differs")

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
    if len(combined) != len(extended) or not np.isfinite(endpoint).all():
        raise ValueError("NEXT136 endpoint row accounting differs")
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
    combined, virtual_terms, runtime = n135.materialize_candidates(
        features=combined,
        coordination_terms=coordination_terms,
        coordination_by_formula=coordination_by_formula,
        specs=specs,
    )
    virtual_by_candidate = {
        str(spec["candidate_key"]): str(spec["base_term_ids"][0]) for spec in runtime
    }
    virtual_by_id = {str(term["term_id"]): term for term in virtual_terms}
    if len(virtual_by_candidate) != len(specs) or set(virtual_by_candidate.values()) != set(virtual_by_id):
        raise ValueError("NEXT136 virtual candidate mapping differs")

    published_all = pd.read_parquet(paths["next135_search_records"])
    if set(published_all["candidate_key"].astype(str)) != set(virtual_by_candidate):
        raise ValueError("NEXT136 published candidate universe differs")
    published = select_safe_candidates(published_all)
    if require_formal_inputs and len(published) != EXPECTED_SAFE_CANDIDATE_COUNT:
        raise ValueError("NEXT136 published SAFE12 count differs")

    folds = assign_group_folds(combined["reduced_formula"].astype(str).to_numpy())
    sources = combined["source_dataset"].astype(str).to_numpy()
    cells = build_source_fold_cells(source=sources, folds=folds)
    pauling_by_cell = {
        str(cell["cell_id"]): _pauling_baseline(
            combined.loc[np.asarray(cell["mask"], dtype=bool)],
            endpoint[np.asarray(cell["mask"], dtype=bool)],
        )
        for cell in cells
    }

    records: list[dict[str, object]] = []
    component_frequency: Counter[str] = Counter()
    for _, row in published.iterrows():
        candidate_key = str(row["candidate_key"])
        virtual_id = virtual_by_candidate[candidate_key]
        score, supported = _term_risk(combined, virtual_by_id[virtual_id])
        tables = _threshold_tables(
            score=score,
            supported=supported,
            endpoint=endpoint,
            cells=cells,
        )
        if tables is None:
            raise RuntimeError("NEXT136 candidate has no supported threshold table")
        diagnostic = diagnose_broad_threshold_tables(
            tables=tables,
            cells=cells,
            pauling_by_cell=pauling_by_cell,
            safe_threshold=float(row["safe_threshold"]),
        )
        if diagnostic["passes_broad"] or bool(row["passes_broad_all_cells"]):
            raise RuntimeError("NEXT136 contradicts published NEXT135 BROAD result")
        for failure in diagnostic["failures"]:
            component_frequency[f"{failure['cell_id']}::{failure['component']}"] += 1
        records.append(
            {
                "candidate_key": candidate_key,
                "conjunctive_term_count": int(row["conjunctive_term_count"]),
                "conjunctive_term_ids_json": str(row["conjunctive_term_ids_json"]),
                "conjunctive_weights_json": str(row["conjunctive_weights_json"]),
                "safe_threshold": float(row["safe_threshold"]),
                "best_threshold": diagnostic["best_threshold"],
                "failed_constraint_count": diagnostic["failed_constraint_count"],
                "normalized_shortfall_sum": diagnostic["normalized_shortfall_sum"],
                "eligible_threshold_count": diagnostic["eligible_threshold_count"],
                "failures_json": json.dumps(
                    diagnostic["failures"], sort_keys=True, separators=(",", ":")
                ),
            }
        )
    per_candidate = pd.DataFrame(records)
    closest = _closest(per_candidate)
    grouped = summarize_groups(per_candidate)
    distribution = {
        str(int(key)): int(value)
        for key, value in per_candidate["failed_constraint_count"]
        .value_counts()
        .sort_index()
        .items()
    }
    summary = {
        "protocol": PROTOCOL,
        "diagnostic_mode": "offline_discovery_label_next135_broad_constraint_residual",
        "safe_candidate_count": int(len(per_candidate)),
        "failed_constraint_count_distribution": distribution,
        **grouped,
        "global_closest": {
            "candidate_key": str(closest["candidate_key"]),
            "conjunctive_term_count": int(closest["conjunctive_term_count"]),
            "conjunctive_term_ids": json.loads(str(closest["conjunctive_term_ids_json"])),
            "conjunctive_weights": json.loads(str(closest["conjunctive_weights_json"])),
            "safe_threshold": float(closest["safe_threshold"]),
            "best_threshold": float(closest["best_threshold"]),
            "failed_constraint_count": int(closest["failed_constraint_count"]),
            "normalized_shortfall_sum": float(closest["normalized_shortfall_sum"]),
            "failures": json.loads(str(closest["failures_json"])),
        },
        "failure_frequency_at_per_candidate_optima": dict(component_frequency.most_common()),
        "cells": [
            {key: value for key, value in cell.items() if key != "mask"} for cell in cells
        ],
        "pauling_by_cell": pauling_by_cell,
        "new_formula_searched": False,
        "validation_or_replication_opened": False,
        "dft_calculation_executed": False,
        "dft_values_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "physical_relaxation_executed": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next128_broad_residual_diagnostic.py": Path(
            diagnose_broad_threshold_tables.__code__.co_filename
        ).resolve(),
        "src/next135_conjunctive_compactness_search.py": Path(n135.__file__).resolve(),
        "src/next136_conjunctive_broad_residual_diagnostic.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    try:
        diagnostic_path = staging / DIAGNOSTIC_NAME
        per_candidate_path = staging / PER_CANDIDATE_NAME
        _write_json(diagnostic_path, summary)
        per_candidate.to_parquet(per_candidate_path, index=False)
        manifest = {
            "protocol": PROTOCOL,
            "safe_candidate_count": int(len(per_candidate)),
            "new_formula_searched": False,
            "discovery_outcomes_used_as_offline_labels": True,
            "opened_validation_outputs_used": False,
            "scigen_replication_endpoint_opened": False,
            "wyformer_replication_endpoint_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "physical_relaxation_executed": False,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {
                DIAGNOSTIC_NAME: _sha256_file(diagnostic_path),
                PER_CANDIDATE_NAME: _sha256_file(per_candidate_path),
            },
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT136 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT136 source changed before publication")
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
    for stage in (98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124, 125, 129, 130, 133, 134, 135):
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_conjunctive_broad_residual_diagnostic(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{
            f"next{stage}_dir": getattr(args, f"next{stage}_dir")
            for stage in (98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124, 125, 129, 130, 133, 134, 135)
        },
        next135_freeze_path=args.next135_freeze_path,
        design_path=args.design_path,
        output_dir=args.output_dir,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "run_conjunctive_broad_residual_diagnostic",
    "select_safe_candidates",
    "summarize_groups",
]
