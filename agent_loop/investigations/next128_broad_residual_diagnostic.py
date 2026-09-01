#!/usr/bin/env python3
"""Diagnose the remaining strict BROAD residual of frozen NEXT125 laws."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import src.next125_mhcr_frontier_rescue as n125
import src.next127_hall_profile_persistence_rescue as n127
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next87_scigen_sparse_law_search import _term_risk, assign_group_folds
from src.next98_cross_source_discovery_search import (
    _pauling_baseline,
    _threshold_tables,
    build_source_fold_cells,
)


PROTOCOL = "2026-08-08-next128-broad-residual-diagnostic-v1"
MANIFEST_NAME = "MANIFEST.json"
DIAGNOSTIC_NAME = "NEXT128_BROAD_RESIDUAL_DIAGNOSTIC.json"
PER_LAW_NAME = "next128_broad_residual_by_law.parquet"
EXPECTED_DESIGN_SHA256 = "f69346fae76c3d3678580e454bbe4c0694a0910ff1ed5a8b4a7c2b5068b29d1b"
EXPECTED_NEXT125_MANIFEST_SHA256 = "305b1a6044ee43b17a56edd8e7630819955328d35416fa5bd8c178eddf12dac9"
EXPECTED_NEXT127_MANIFEST_SHA256 = "3cd6ea08acd0c64db2420c53522192c35c2828f0a40de0dfc0a62bcf0362b295"
EXPECTED_LAW_COUNT = 260
BROAD_MIN_PRECISION_LOWER = n125.n121.prior.BROAD_MIN_PRECISION_LOWER


def _failure(
    *,
    cell_id: str,
    component: str,
    comparator: str,
    observed: float,
    required: float,
    normalized_shortfall: float,
) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "component": component,
        "comparator": comparator,
        "observed": float(observed),
        "required": float(required),
        "normalized_shortfall": float(max(0.0, normalized_shortfall)),
    }


def diagnose_broad_threshold_tables(
    *,
    tables: Mapping[str, object],
    cells: Sequence[Mapping[str, object]],
    pauling_by_cell: Mapping[str, Mapping[str, object]],
    safe_threshold: float,
    broad_min_precision_lower: float = BROAD_MIN_PRECISION_LOWER,
) -> dict[str, object]:
    """Return the closest threshold under the exact strict BROAD inequalities."""

    thresholds = np.asarray(tables["thresholds"], dtype=float)
    coverage = np.asarray(tables["coverage_lower"], dtype=float)
    protected = np.asarray(tables["protected_kept"], dtype=float)
    severe = np.asarray(tables["rejected_severe"], dtype=float)
    precision = np.asarray(tables["precision_lower"], dtype=float)
    savings = np.asarray(tables["savings_lower"], dtype=float)
    cell_count = len(cells)
    if (
        thresholds.ndim != 1
        or coverage.shape != (cell_count,)
        or any(array.shape != (cell_count, len(thresholds)) for array in (protected, severe, precision, savings))
        or not math.isfinite(float(safe_threshold))
        or not math.isfinite(float(broad_min_precision_lower))
    ):
        raise ValueError("NEXT128 threshold table schema differs")
    eligible = np.flatnonzero(np.isfinite(thresholds) & (thresholds < float(safe_threshold)))
    if not len(eligible):
        return {
            "passes_broad": False,
            "best_threshold": None,
            "failed_constraint_count": None,
            "normalized_shortfall_sum": None,
            "failures": [],
            "eligible_threshold_count": 0,
        }

    candidates: list[dict[str, object]] = []
    for threshold_index in eligible.tolist():
        failures: list[dict[str, object]] = []
        for cell_index, cell in enumerate(cells):
            cell_id = str(cell["cell_id"])
            baseline = pauling_by_cell[cell_id]
            comparisons = (
                (
                    "coverage_lower",
                    ">",
                    float(coverage[cell_index]),
                    float(baseline["coverage_lower"]),
                    max(0.0, float(baseline["coverage_lower"]) - float(coverage[cell_index])),
                ),
                (
                    "protected_kept",
                    ">=",
                    float(protected[cell_index, threshold_index]),
                    float(baseline["protected_kept"]),
                    max(0.0, float(baseline["protected_kept"]) - float(protected[cell_index, threshold_index]))
                    / max(1.0, float(baseline["protected_kept"])),
                ),
                (
                    "severe_rejected",
                    ">",
                    float(severe[cell_index, threshold_index]),
                    float(baseline["severe_rejected"]),
                    max(0.0, float(baseline["severe_rejected"]) + 1.0 - float(severe[cell_index, threshold_index]))
                    / max(1.0, float(baseline["severe_rejected"]) + 1.0),
                ),
                (
                    "severe_precision_lower",
                    ">",
                    float(precision[cell_index, threshold_index]),
                    float(baseline["severe_rejection_precision_lower"]),
                    max(0.0, float(baseline["severe_rejection_precision_lower"]) - float(precision[cell_index, threshold_index])),
                ),
                (
                    "savings_lower",
                    ">",
                    float(savings[cell_index, threshold_index]),
                    float(baseline["savings_lower"]),
                    max(0.0, float(baseline["savings_lower"]) - float(savings[cell_index, threshold_index])),
                ),
            )
            for component, comparator, observed, required, shortfall in comparisons:
                passed = observed >= required if comparator == ">=" else observed > required
                if not passed:
                    failures.append(
                        _failure(
                            cell_id=cell_id,
                            component=component,
                            comparator=comparator,
                            observed=observed,
                            required=required,
                            normalized_shortfall=shortfall,
                        )
                    )
            if cell.get("kind") == "source_aggregate":
                observed = float(precision[cell_index, threshold_index])
                if observed < float(broad_min_precision_lower):
                    failures.append(
                        _failure(
                            cell_id=cell_id,
                            component="aggregate_precision_lower",
                            comparator=">=",
                            observed=observed,
                            required=float(broad_min_precision_lower),
                            normalized_shortfall=float(broad_min_precision_lower) - observed,
                        )
                    )
        candidates.append(
            {
                "threshold": float(thresholds[threshold_index]),
                "failed_constraint_count": len(failures),
                "normalized_shortfall_sum": float(
                    sum(float(item["normalized_shortfall"]) for item in failures)
                ),
                "failures": failures,
            }
        )
    best = min(
        candidates,
        key=lambda item: (
            int(item["failed_constraint_count"]),
            float(item["normalized_shortfall_sum"]),
            float(item["threshold"]),
        ),
    )
    return {
        "passes_broad": int(best["failed_constraint_count"]) == 0,
        "best_threshold": best["threshold"],
        "failed_constraint_count": best["failed_constraint_count"],
        "normalized_shortfall_sum": best["normalized_shortfall_sum"],
        "failures": best["failures"],
        "eligible_threshold_count": len(eligible),
    }


def run_broad_residual_diagnostic(
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
    next127_dir: Path,
    design_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Reconstruct and diagnose all 260 frozen NEXT125 AUC+SAFE12 laws."""

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
                (127, next127_dir),
            )
        },
    }
    target = Path(output_dir).resolve()
    paths = n125._paths(roots, Path(design_path).resolve())
    paths["design"] = paths.pop("freeze")
    paths.update(
        {
            "next125_manifest": roots["next125"] / n125.MANIFEST_NAME,
            "next125_search_records": roots["next125"] / n125.SEARCH_NAME,
            "next127_manifest": roots["next127"] / n127.MANIFEST_NAME,
            "next127_evaluation": roots["next127"] / n127.EVALUATION_NAME,
            "next127_search_records": roots["next127"] / n127.SEARCH_NAME,
        }
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT128 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if (
        input_hashes["design"] != EXPECTED_DESIGN_SHA256
        or input_hashes["next125_manifest"] != EXPECTED_NEXT125_MANIFEST_SHA256
        or input_hashes["next127_manifest"] != EXPECTED_NEXT127_MANIFEST_SHA256
    ):
        raise ValueError("NEXT128 formal input identity differs")
    manifest125 = json.loads(paths["next125_manifest"].read_text())
    manifest127 = json.loads(paths["next127_manifest"].read_text())
    if (
        manifest125.get("protocol") != n125.PROTOCOL
        or manifest127.get("protocol") != n127.PROTOCOL
        or manifest125.get("opened_validation_outputs_used") is not False
        or manifest127.get("opened_validation_outputs_used") is not False
        or manifest125.get("scigen_replication_endpoint_opened") is not False
        or manifest127.get("wyformer_replication_endpoint_opened") is not False
        or manifest125.get("dft_values_used_by_executable_formula") is not False
        or manifest127.get("dft_values_used_by_executable_formula") is not False
    ):
        raise ValueError("NEXT128 prior provenance differs")

    features, _, old_terms = n125.prior._reconstruct_label_free_table(paths)
    retained = sorted(
        {str(spec["raw_feature"]) for spec in n125.FROZEN_TERM_SPECS}
        | {str(spec["support_column"]) for spec in n125.FROZEN_TERM_SPECS}
    )
    mhcr_frames: list[pd.DataFrame] = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next124_{source}_features"])
        frame = table.loc[:, ["material_id", *retained]].copy()
        frame["material_id"] = source + ":" + frame["material_id"].astype(str)
        mhcr_frames.append(frame)
    mhcr = pd.concat(mhcr_frames, ignore_index=True, sort=False)
    extended = features.merge(mhcr, on="material_id", how="inner", validate="one_to_one")
    if len(extended) != len(features) or len(mhcr) != len(features):
        raise ValueError("NEXT128 MHCR row accounting differs")
    extended, mhcr_terms = n125.materialize_mhcr_tail_terms(extended)
    bases = n127.select_next125_bases(pd.read_parquet(paths["next125_search_records"]))
    extended, virtual_terms, virtual_by_formula = n127.materialize_virtual_bases(
        features=extended,
        bases=bases,
        old_terms=old_terms,
        mhcr_terms=mhcr_terms,
    )
    if len(bases) != EXPECTED_LAW_COUNT or len(virtual_terms) != EXPECTED_LAW_COUNT:
        raise ValueError("NEXT128 law count differs")

    scigen_endpoint = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_endpoint = pd.read_parquet(paths["wyformer_endpoint"])
    endpoint_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "material_id": "scigen:" + scigen_endpoint["material_id"].astype(str),
                    "_endpoint": pd.to_numeric(scigen_endpoint["distortion_ratio"], errors="coerce"),
                }
            ),
            pd.DataFrame(
                {
                    "material_id": "wyformer:" + wyformer_endpoint["material_id"].astype(str),
                    "_endpoint": n125.n121.prior._endpoint_numeric(
                        wyformer_endpoint["endpoint_stratum"]
                    ),
                }
            ),
        ],
        ignore_index=True,
    )
    combined = extended.merge(endpoint_frame, on="material_id", how="inner", validate="one_to_one")
    endpoint = pd.to_numeric(combined.pop("_endpoint"), errors="coerce").to_numpy(float)
    if len(combined) != len(extended) or not np.isfinite(endpoint).all():
        raise ValueError("NEXT128 endpoint row accounting differs")
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
    records: list[dict[str, object]] = []
    component_frequency: Counter[str] = Counter()
    for _, row in bases.iterrows():
        formula = n127._formula_identity(
            json.loads(str(row["term_ids_json"])),
            json.loads(str(row["weights_json"])),
        )
        virtual_id = virtual_by_formula[formula]
        score, supported = _term_risk(combined, virtual_by_id[virtual_id])
        tables = _threshold_tables(
            score=score,
            supported=supported,
            endpoint=endpoint,
            cells=cells,
        )
        if tables is None:
            raise RuntimeError("NEXT128 law has no supported threshold table")
        safe_threshold = float(row["_prior_record"]["safe_threshold"])
        diagnostic = diagnose_broad_threshold_tables(
            tables=tables,
            cells=cells,
            pauling_by_cell=pauling_by_cell,
            safe_threshold=safe_threshold,
        )
        if diagnostic["passes_broad"]:
            raise RuntimeError("NEXT128 contradicts published NEXT125 BROAD result")
        for failure in diagnostic["failures"]:
            component_frequency[f"{failure['cell_id']}::{failure['component']}"] += 1
        records.append(
            {
                "prior_candidate_key": str(row["prior_candidate_key"]),
                "safe_threshold": safe_threshold,
                "best_threshold": diagnostic["best_threshold"],
                "failed_constraint_count": diagnostic["failed_constraint_count"],
                "normalized_shortfall_sum": diagnostic["normalized_shortfall_sum"],
                "eligible_threshold_count": diagnostic["eligible_threshold_count"],
                "failures_json": json.dumps(diagnostic["failures"], sort_keys=True, separators=(",", ":")),
            }
        )
    per_law = pd.DataFrame(records)
    closest = per_law.sort_values(
        ["failed_constraint_count", "normalized_shortfall_sum", "best_threshold", "prior_candidate_key"]
    ).iloc[0]
    closest_failures = json.loads(str(closest["failures_json"]))
    distribution = {
        str(int(key)): int(value)
        for key, value in per_law["failed_constraint_count"].value_counts().sort_index().items()
    }
    summary = {
        "protocol": PROTOCOL,
        "diagnostic_mode": "offline_discovery_label_broad_constraint_residual",
        "law_count": len(per_law),
        "failed_constraint_count_distribution": distribution,
        "global_closest": {
            "prior_candidate_key": str(closest["prior_candidate_key"]),
            "safe_threshold": float(closest["safe_threshold"]),
            "best_threshold": float(closest["best_threshold"]),
            "failed_constraint_count": int(closest["failed_constraint_count"]),
            "normalized_shortfall_sum": float(closest["normalized_shortfall_sum"]),
            "failures": closest_failures,
        },
        "failure_frequency_at_per_law_optima": dict(component_frequency.most_common()),
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
        "src/next125_mhcr_frontier_rescue.py": Path(n125.__file__).resolve(),
        "src/next127_hall_profile_persistence_rescue.py": Path(n127.__file__).resolve(),
        "src/next128_broad_residual_diagnostic.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    try:
        diagnostic_path = staging / DIAGNOSTIC_NAME
        per_law_path = staging / PER_LAW_NAME
        _write_json(diagnostic_path, summary)
        per_law.to_parquet(per_law_path, index=False)
        manifest = {
            "protocol": PROTOCOL,
            "law_count": len(per_law),
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
                PER_LAW_NAME: _sha256_file(per_law_path),
            },
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT128 input changed before publication")
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
    for stage in (98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124, 125, 127):
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = run_broad_residual_diagnostic(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{
            f"next{stage}_dir": getattr(args, f"next{stage}_dir")
            for stage in (98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124, 125, 127)
        },
        design_path=args.design_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["diagnose_broad_threshold_tables", "run_broad_residual_diagnostic"]
