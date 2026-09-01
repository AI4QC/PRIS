#!/usr/bin/env python3
"""Diagnose exact BROAD residuals of published NEXT130 SAFE12 candidates."""

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
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next128_broad_residual_diagnostic import diagnose_broad_threshold_tables
from src.next87_scigen_sparse_law_search import _term_risk, assign_group_folds
from src.next98_cross_source_discovery_search import (
    _pauling_baseline,
    _threshold_tables,
    build_source_fold_cells,
)


PROTOCOL = "2026-08-08-next131-protected-broad-residual-diagnostic-v1"
MANIFEST_NAME = "MANIFEST.json"
DIAGNOSTIC_NAME = "NEXT131_PROTECTED_BROAD_RESIDUAL_DIAGNOSTIC.json"
PER_CANDIDATE_NAME = "next131_protected_broad_residual_by_candidate.parquet"
EXPECTED_DESIGN_SHA256 = "ea0e6d6e4c0cf1e00360177bca55bb4c0de299de1d717fecfc50c5a09fe71555"
EXPECTED_NEXT130_MANIFEST_SHA256 = "8c672fdcd5b97a282604ebd49678d698e24c5f5f4e90412fb056844131d0119e"
EXPECTED_SAFE_CANDIDATE_COUNT = 454
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n130.EXPECTED_INPUT_SHA256.items()
        if key != "freeze"
    },
    "design": EXPECTED_DESIGN_SHA256,
    "next130_manifest": EXPECTED_NEXT130_MANIFEST_SHA256,
    "next130_catalogue": "3c1386bd338ccccfc777825053e4f440171a83695450dc13bfa4e88723cf9857",
    "next130_evaluation": "87b7672aa6c2224597c0b0a3b582a2c353db76426ef7de956d89493d7ef4a019",
    "next130_search_records": "223dfb259e7b62e423bc5739f01ba18f3107aedd84a5370a8351fc94fc9f8cb0",
}


def select_safe_candidates(records: pd.DataFrame) -> pd.DataFrame:
    """Select only published SAFE12 candidates, preserving their thresholds."""

    required = {
        "candidate_key",
        "safe_threshold",
        "passes_safe_all_cells",
        "protection_term_id",
        "protection_weight",
    }
    if required - set(records.columns) or records["candidate_key"].astype(str).duplicated().any():
        raise ValueError("NEXT131 published candidate schema differs")
    keep = records["passes_safe_all_cells"].fillna(False).astype(bool)
    selected = records.loc[keep].copy()
    thresholds = pd.to_numeric(selected["safe_threshold"], errors="coerce")
    if selected.empty or not np.isfinite(thresholds.to_numpy(float)).all():
        raise ValueError("NEXT131 published safe threshold differs")
    selected["safe_threshold"] = thresholds
    return selected.sort_values("candidate_key").reset_index(drop=True)


def _paths(roots: Mapping[str, Path], design_path: Path) -> dict[str, Path]:
    paths = n130._paths(roots, design_path)
    paths["design"] = paths.pop("freeze")
    paths.update(
        {
            "next130_manifest": roots["next130"] / n130.MANIFEST_NAME,
            "next130_catalogue": roots["next130"] / n130.CATALOGUE_NAME,
            "next130_evaluation": roots["next130"] / n130.EVALUATION_NAME,
            "next130_search_records": roots["next130"] / n130.SEARCH_NAME,
        }
    )
    return paths


def run_protected_broad_residual_diagnostic(
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
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Reconstruct and diagnose all published NEXT130 SAFE12 candidates."""

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
            )
        },
    }
    target = Path(output_dir).resolve()
    paths = _paths(roots, Path(design_path).resolve())
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT131 diagnostic input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            key
            for key in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(key) != EXPECTED_INPUT_SHA256.get(key)
        )
        raise ValueError(f"NEXT131 formal input identity differs: {differing}")
    manifest130 = json.loads(paths["next130_manifest"].read_text())
    if (
        manifest130.get("protocol") != n130.PROTOCOL
        or manifest130.get("passes_all_cross_source_discovery_gates") is not False
        or manifest130.get("opened_validation_outputs_used") is not False
        or manifest130.get("scigen_replication_endpoint_opened") is not False
        or manifest130.get("wyformer_replication_endpoint_opened") is not False
        or manifest130.get("dft_values_used_by_executable_formula") is not False
    ):
        raise ValueError("NEXT131 prior provenance differs")
    outputs130 = manifest130.get("outputs_sha256")
    expected_outputs = {
        n130.CATALOGUE_NAME: "next130_catalogue",
        n130.EVALUATION_NAME: "next130_evaluation",
        n130.SEARCH_NAME: "next130_search_records",
    }
    if not isinstance(outputs130, Mapping) or any(
        outputs130.get(filename) != input_hashes[key]
        for filename, key in expected_outputs.items()
    ):
        raise ValueError("NEXT131 prior output identity differs")

    extended, _, old_terms, mhcr_terms = n130._join_label_free_features(paths)
    all_physical_terms = [*old_terms, *mhcr_terms]
    physical_ids = {str(term["term_id"]) for term in all_physical_terms}
    bases = n130.n127.select_next125_bases(
        pd.read_parquet(paths["next125_search_records"])
    )
    physical_specs = n130.build_candidate_specs(
        bases=bases, old_term_ids=physical_ids
    )
    if len(physical_specs) != n130.EXPECTED_CANDIDATE_COUNT:
        raise ValueError("NEXT131 frozen candidate universe differs")

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
        raise ValueError("NEXT131 endpoint row accounting differs")
    combined, base_virtual_terms, base_virtual_by_formula = n130.n127.materialize_virtual_bases(
        features=combined,
        bases=bases,
        old_terms=old_terms,
        mhcr_terms=mhcr_terms,
    )
    combined, virtual_terms, runtime_specs, virtual_by_candidate = n130.materialize_protected_candidates(
        features=combined,
        bases=bases,
        base_virtual_terms=base_virtual_terms,
        base_virtual_by_formula=base_virtual_by_formula,
        physical_specs=physical_specs,
    )
    if len(runtime_specs) != n130.EXPECTED_CANDIDATE_COUNT:
        raise ValueError("NEXT131 runtime candidate universe differs")

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
    virtual_by_id = {str(term["term_id"]): term for term in virtual_terms}
    published = select_safe_candidates(
        pd.read_parquet(paths["next130_search_records"])
    )
    if require_formal_inputs and len(published) != EXPECTED_SAFE_CANDIDATE_COUNT:
        raise ValueError("NEXT131 published SAFE12 count differs")

    records: list[dict[str, object]] = []
    component_frequency: Counter[str] = Counter()
    for _, row in published.iterrows():
        candidate_key = str(row["candidate_key"])
        virtual_id = virtual_by_candidate.get(candidate_key)
        if virtual_id is None or virtual_id not in virtual_by_id:
            raise RuntimeError("NEXT131 virtual candidate mapping is incomplete")
        score, supported = _term_risk(combined, virtual_by_id[virtual_id])
        tables = _threshold_tables(
            score=score,
            supported=supported,
            endpoint=endpoint,
            cells=cells,
        )
        if tables is None:
            raise RuntimeError("NEXT131 candidate has no supported threshold table")
        diagnostic = diagnose_broad_threshold_tables(
            tables=tables,
            cells=cells,
            pauling_by_cell=pauling_by_cell,
            safe_threshold=float(row["safe_threshold"]),
        )
        if diagnostic["passes_broad"] or bool(row["passes_broad_all_cells"]):
            raise RuntimeError("NEXT131 contradicts published NEXT130 BROAD result")
        for failure in diagnostic["failures"]:
            component_frequency[
                f"{failure['cell_id']}::{failure['component']}"
            ] += 1
        records.append(
            {
                "candidate_key": candidate_key,
                "protection_term_id": row["protection_term_id"],
                "protection_weight": float(row["protection_weight"]),
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
    closest = per_candidate.sort_values(
        [
            "failed_constraint_count",
            "normalized_shortfall_sum",
            "best_threshold",
            "candidate_key",
        ]
    ).iloc[0]
    closest_failures = json.loads(str(closest["failures_json"]))
    distribution = {
        str(int(key)): int(value)
        for key, value in per_candidate["failed_constraint_count"]
        .value_counts()
        .sort_index()
        .items()
    }
    by_weight: dict[str, object] = {}
    for weight, frame in per_candidate.groupby("protection_weight", sort=True):
        best = frame.sort_values(
            ["failed_constraint_count", "normalized_shortfall_sum", "best_threshold", "candidate_key"]
        ).iloc[0]
        by_weight[f"{float(weight):g}"] = {
            "candidate_count": int(len(frame)),
            "minimum_failed_constraint_count": int(best["failed_constraint_count"]),
            "minimum_normalized_shortfall_sum_at_best_count": float(
                frame.loc[
                    frame["failed_constraint_count"].eq(best["failed_constraint_count"]),
                    "normalized_shortfall_sum",
                ].min()
            ),
        }
    summary = {
        "protocol": PROTOCOL,
        "diagnostic_mode": "offline_discovery_label_next130_broad_constraint_residual",
        "safe_candidate_count": len(per_candidate),
        "failed_constraint_count_distribution": distribution,
        "by_protection_weight": by_weight,
        "global_closest": {
            "candidate_key": str(closest["candidate_key"]),
            "protection_term_id": closest["protection_term_id"],
            "protection_weight": float(closest["protection_weight"]),
            "safe_threshold": float(closest["safe_threshold"]),
            "best_threshold": float(closest["best_threshold"]),
            "failed_constraint_count": int(closest["failed_constraint_count"]),
            "normalized_shortfall_sum": float(closest["normalized_shortfall_sum"]),
            "failures": closest_failures,
        },
        "failure_frequency_at_per_candidate_optima": dict(
            component_frequency.most_common()
        ),
        "cells": [
            {key: value for key, value in cell.items() if key != "mask"}
            for cell in cells
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
        "src/next130_coordination_protection_search.py": Path(n130.__file__).resolve(),
        "src/next131_protected_broad_residual_diagnostic.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    try:
        diagnostic_path = staging / DIAGNOSTIC_NAME
        per_candidate_path = staging / PER_CANDIDATE_NAME
        _write_json(diagnostic_path, summary)
        per_candidate.to_parquet(per_candidate_path, index=False)
        manifest = {
            "protocol": PROTOCOL,
            "safe_candidate_count": len(per_candidate),
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
            raise RuntimeError("NEXT131 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT131 source changed before publication")
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
    for stage in (98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124, 125, 129, 130):
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_protected_broad_residual_diagnostic(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{
            f"next{stage}_dir": getattr(args, f"next{stage}_dir")
            for stage in (98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124, 125, 129, 130)
        },
        design_path=args.design_path,
        output_dir=args.output_dir,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_protected_broad_residual_diagnostic", "select_safe_candidates"]
