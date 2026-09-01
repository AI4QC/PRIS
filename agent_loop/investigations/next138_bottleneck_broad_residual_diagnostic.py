#!/usr/bin/env python3
"""Diagnose exact BROAD residuals of published NEXT137 SAFE12 candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd

import src.next130_coordination_protection_search as n130
import src.next134_compactness_protection_search as n134
import src.next136_conjunctive_broad_residual_diagnostic as n136
import src.next137_conjunctive_bottleneck_search as n137
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next128_broad_residual_diagnostic import diagnose_broad_threshold_tables
from src.next87_scigen_sparse_law_search import _term_risk, assign_group_folds
from src.next98_cross_source_discovery_search import (
    _pauling_baseline,
    _threshold_tables,
    build_source_fold_cells,
)


PROTOCOL = "2026-08-08-next138-bottleneck-broad-residual-diagnostic-v1"
MANIFEST_NAME = "MANIFEST.json"
DIAGNOSTIC_NAME = "NEXT138_BOTTLENECK_BROAD_RESIDUAL_DIAGNOSTIC.json"
PER_CANDIDATE_NAME = "next138_bottleneck_broad_residual_by_candidate.parquet"
EXPECTED_DESIGN_SHA256 = "4df66dc62b650ca04fa0d8d503f371872292598496f8dc5b1e0c55700881a72e"
EXPECTED_NEXT137_MANIFEST_SHA256 = "734f54ab0893cb6d13d32b0dee12ca78e4ae0fa9ba637caa53fc5c7b4f9823c2"
EXPECTED_SAFE_CANDIDATE_COUNT = 66
EXPECTED_INPUT_SHA256 = {
    **n137.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next137_manifest": EXPECTED_NEXT137_MANIFEST_SHA256,
    "next137_catalogue": "b5bcc6c608ce5b3e91c142f5f631ba1271d48f39918ce3f4f762a41825cceaa7",
    "next137_evaluation": "59614e4bad7e8c1faa657857db3327c77fa185b1357a4995ae7f74a73c070675",
    "next137_search_records": "44be6c75f3e2df3de1181ba54868c58939af2ee0485452830afa73d43fb0f481",
}


def select_safe_candidates(records: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate_key",
        "safe_threshold",
        "passes_safe_all_cells",
        "bottleneck_term_ids_json",
        "bottleneck_weights_json",
        "bottleneck_term_count",
    }
    if required - set(records.columns) or records["candidate_key"].astype(str).duplicated().any():
        raise ValueError("NEXT138 published candidate schema differs")
    selected = records.loc[records["passes_safe_all_cells"].fillna(False).astype(bool)].copy()
    thresholds = pd.to_numeric(selected["safe_threshold"], errors="coerce")
    if selected.empty or not np.isfinite(thresholds.to_numpy(float)).all():
        raise ValueError("NEXT138 published safe threshold differs")
    selected["safe_threshold"] = thresholds
    for _, row in selected.iterrows():
        term_ids = json.loads(str(row["bottleneck_term_ids_json"]))
        weights = json.loads(str(row["bottleneck_weights_json"]))
        if (
            not isinstance(term_ids, list)
            or not isinstance(weights, list)
            or len(term_ids) != len(weights)
            or len(term_ids) != int(row["bottleneck_term_count"])
        ):
            raise ValueError("NEXT138 published bottleneck identity differs")
    return selected.sort_values("candidate_key").reset_index(drop=True)


def _closest(frame: pd.DataFrame) -> pd.Series:
    return frame.sort_values(
        ["failed_constraint_count", "normalized_shortfall_sum", "best_threshold", "candidate_key"]
    ).iloc[0]


def _group_summary(records: pd.DataFrame) -> dict[str, object]:
    def summarize(frame: pd.DataFrame) -> dict[str, object]:
        best = _closest(frame)
        count = int(best["failed_constraint_count"])
        return {
            "candidate_count": int(len(frame)),
            "minimum_failed_constraint_count": count,
            "minimum_normalized_shortfall_sum_at_best_count": float(
                frame.loc[
                    frame["failed_constraint_count"].eq(count),
                    "normalized_shortfall_sum",
                ].min()
            ),
            "closest_candidate_key": str(best["candidate_key"]),
        }

    by_count = {
        str(int(count)): summarize(frame)
        for count, frame in records.groupby("bottleneck_term_count", sort=True)
    }
    by_configuration = {
        f"{term_ids}@{weights}": summarize(frame)
        for (term_ids, weights), frame in records.groupby(
            ["bottleneck_term_ids_json", "bottleneck_weights_json"], sort=True
        )
    }
    return {
        "by_bottleneck_term_count": by_count,
        "by_configuration": by_configuration,
    }


def _paths(
    roots: Mapping[str, Path], next137_freeze_path: Path, design_path: Path
) -> dict[str, Path]:
    paths = n137._paths(roots, next137_freeze_path)
    paths.update(
        {
            "design": design_path,
            "next137_manifest": roots["next137"] / n137.MANIFEST_NAME,
            "next137_catalogue": roots["next137"] / n137.CATALOGUE_NAME,
            "next137_evaluation": roots["next137"] / n137.EVALUATION_NAME,
            "next137_search_records": roots["next137"] / n137.SEARCH_NAME,
        }
    )
    return paths


def run_bottleneck_broad_residual_diagnostic(
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
    next137_dir: Path,
    next137_freeze_path: Path,
    design_path: Path,
    output_dir: Path,
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
                (137, next137_dir),
            )
        },
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots,
        Path(next137_freeze_path).resolve(),
        Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT138 diagnostic input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            key
            for key in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(key) != EXPECTED_INPUT_SHA256.get(key)
        )
        raise ValueError(f"NEXT138 formal input identity differs: {differing}")
    manifest137 = json.loads(paths["next137_manifest"].read_text())
    outputs137 = manifest137.get("outputs_sha256")
    expected_outputs = {
        n137.CATALOGUE_NAME: "next137_catalogue",
        n137.EVALUATION_NAME: "next137_evaluation",
        n137.SEARCH_NAME: "next137_search_records",
    }
    if (
        manifest137.get("protocol") != n137.PROTOCOL
        or manifest137.get("passes_all_cross_source_discovery_gates") is not False
        or manifest137.get("opened_validation_outputs_used") is not False
        or manifest137.get("dft_values_used_by_executable_formula") is not False
        or not isinstance(outputs137, Mapping)
        or any(outputs137.get(name) != input_hashes[key] for name, key in expected_outputs.items())
    ):
        raise ValueError("NEXT138 prior provenance differs")

    extended, _, old_terms, mhcr_terms = n130._join_label_free_features(paths)
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
    bottleneck = n137.materialize_bottleneck_features(extended)
    extended = pd.concat(
        [extended.reset_index(drop=True), bottleneck.reset_index(drop=True)], axis=1
    )
    physical_terms = [*old_terms, *mhcr_terms]
    physical_ids = {str(term["term_id"]) for term in physical_terms}
    all_bases = n130.n127.select_next125_bases(
        pd.read_parquet(paths["next125_search_records"])
    )
    bases = n137.n132.select_extended_bases(
        pd.read_parquet(paths["next130_search_records"]), all_bases
    )
    specs = n137.build_candidate_specs(bases=bases, physical_term_ids=physical_ids)
    if require_formal_inputs and len(specs) != n137.EXPECTED_CANDIDATE_COUNT:
        raise ValueError("NEXT138 frozen candidate universe differs")

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
        raise ValueError("NEXT138 endpoint row accounting differs")
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
    combined, virtual_terms, runtime = n137.materialize_candidates(
        features=combined,
        coordination_terms=coordination_terms,
        coordination_by_formula=coordination_by_formula,
        specs=specs,
    )
    virtual_by_candidate = {
        str(spec["candidate_key"]): str(spec["base_term_ids"][0]) for spec in runtime
    }
    virtual_by_id = {str(term["term_id"]): term for term in virtual_terms}
    if set(virtual_by_candidate.values()) != set(virtual_by_id):
        raise ValueError("NEXT138 virtual candidate mapping differs")

    published_all = pd.read_parquet(paths["next137_search_records"])
    if set(published_all["candidate_key"].astype(str)) != set(virtual_by_candidate):
        raise ValueError("NEXT138 published candidate universe differs")
    published = select_safe_candidates(published_all)
    if require_formal_inputs and len(published) != EXPECTED_SAFE_CANDIDATE_COUNT:
        raise ValueError("NEXT138 published SAFE12 count differs")

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
    frequency: Counter[str] = Counter()
    for _, row in published.iterrows():
        key = str(row["candidate_key"])
        score, supported = _term_risk(combined, virtual_by_id[virtual_by_candidate[key]])
        tables = _threshold_tables(
            score=score, supported=supported, endpoint=endpoint, cells=cells
        )
        if tables is None:
            raise RuntimeError("NEXT138 candidate has no supported threshold table")
        diagnostic = diagnose_broad_threshold_tables(
            tables=tables,
            cells=cells,
            pauling_by_cell=pauling_by_cell,
            safe_threshold=float(row["safe_threshold"]),
        )
        if diagnostic["passes_broad"] or bool(row["passes_broad_all_cells"]):
            raise RuntimeError("NEXT138 contradicts published NEXT137 BROAD result")
        for failure in diagnostic["failures"]:
            frequency[f"{failure['cell_id']}::{failure['component']}"] += 1
        records.append(
            {
                "candidate_key": key,
                "bottleneck_term_count": int(row["bottleneck_term_count"]),
                "bottleneck_term_ids_json": str(row["bottleneck_term_ids_json"]),
                "bottleneck_weights_json": str(row["bottleneck_weights_json"]),
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
    summary = {
        "protocol": PROTOCOL,
        "diagnostic_mode": "offline_discovery_label_next137_broad_constraint_residual",
        "safe_candidate_count": int(len(per_candidate)),
        "failed_constraint_count_distribution": {
            str(int(key)): int(value)
            for key, value in per_candidate["failed_constraint_count"]
            .value_counts()
            .sort_index()
            .items()
        },
        **_group_summary(per_candidate),
        "global_closest": {
            "candidate_key": str(closest["candidate_key"]),
            "bottleneck_term_count": int(closest["bottleneck_term_count"]),
            "bottleneck_term_ids": json.loads(str(closest["bottleneck_term_ids_json"])),
            "bottleneck_weights": json.loads(str(closest["bottleneck_weights_json"])),
            "safe_threshold": float(closest["safe_threshold"]),
            "best_threshold": float(closest["best_threshold"]),
            "failed_constraint_count": int(closest["failed_constraint_count"]),
            "normalized_shortfall_sum": float(closest["normalized_shortfall_sum"]),
            "failures": json.loads(str(closest["failures_json"])),
        },
        "failure_frequency_at_per_candidate_optima": dict(frequency.most_common()),
        "cells": [{key: value for key, value in cell.items() if key != "mask"} for cell in cells],
        "pauling_by_cell": pauling_by_cell,
        "branch_termination_rule": "terminate_coordination_by_compactness_if_no_candidate_reduces_fixed_six_failures",
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
        "src/next137_conjunctive_bottleneck_search.py": Path(n137.__file__).resolve(),
        "src/next138_bottleneck_broad_residual_diagnostic.py": Path(__file__).resolve(),
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
            "coordination_by_compactness_branch_terminated": bool(
                int(closest["failed_constraint_count"]) >= 6
            ),
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
            raise RuntimeError("NEXT138 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT138 source changed before publication")
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
    for stage in (98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124, 125, 129, 130, 133, 134, 136, 137):
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next137-freeze-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_bottleneck_broad_residual_diagnostic(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{
            f"next{stage}_dir": getattr(args, f"next{stage}_dir")
            for stage in (98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124, 125, 129, 130, 133, 134, 136, 137)
        },
        next137_freeze_path=args.next137_freeze_path,
        design_path=args.design_path,
        output_dir=args.output_dir,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_bottleneck_broad_residual_diagnostic", "select_safe_candidates"]
