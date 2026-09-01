#!/usr/bin/env python3
"""Audit raw x0 features inside the exact NEXT229 rejected frontier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd

import src.next229_margin_local_broad_diagnostic as n229
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


n227 = n229.n228.n227
n207 = n227.n207
PROTOCOL = "2026-08-09-next230-stagewise-margin-local-feature-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT230_STAGEWISE_MARGIN_LOCAL_FEATURE_CATALOGUE.json"
AUDIT_NAME = "NEXT230_STAGEWISE_MARGIN_LOCAL_FEATURE_AUDIT.json"
TABLE_NAME = "next230_stagewise_margin_local_feature_audit.parquet"
EXPECTED_DESIGN_SHA256 = (
    "3ced88d86829c1d979474909248605eede321266a338f950a506aadf8e985556"
)
EXPECTED_BASE_CANDIDATE_KEY_SHA256 = (
    "3115dd8189d1125c3863f09a107c083fd81f538d3ac27351273cd8a4bbe41b5a"
)
EXPECTED_BASE_THRESHOLD = 0.10672744194580967
EXPECTED_BASE_FAILED_COUNT = 5
EXPECTED_BASE_SHORTFALL = 0.16431186635663908
EXPECTED_BASE_SUPPORT_COUNT = n227.EXPECTED_BASE_SUPPORT_COUNT
EXPECTED_FEATURE_COUNT = n227.EXPECTED_FEATURE_COUNT
EXPECTED_FEATURE_NAME_SHA256 = n227.EXPECTED_FEATURE_NAME_SHA256
EXPECTED_HYPOTHESIS_COUNT = n227.EXPECTED_HYPOTHESIS_COUNT
PROTECTION_DIRECTIONS = n227.PROTECTION_DIRECTIONS
MINIMUM_COVERAGE = n227.MINIMUM_COVERAGE
MINIMUM_CLASS_COUNT = n227.MINIMUM_CLASS_COUNT
MINIMUM_AGGREGATE_AUC = n227.MINIMUM_AGGREGATE_AUC
MINIMUM_MACRO_AUC = n227.MINIMUM_MACRO_AUC
MINIMUM_WORST_AUC = n227.MINIMUM_WORST_AUC
EXPECTED_COHORT_COUNTS = {
    "scigen:all": (249, 3086),
    "scigen:fold0": (50, 629),
    "scigen:fold1": (54, 621),
    "scigen:fold2": (49, 609),
    "scigen:fold3": (53, 614),
    "scigen:fold4": (43, 613),
    "wyformer:all": (345, 522),
    "wyformer:fold0": (79, 97),
    "wyformer:fold1": (59, 104),
    "wyformer:fold2": (72, 114),
    "wyformer:fold3": (73, 111),
    "wyformer:fold4": (62, 96),
}
REQUIRED_STAGES = (*n229.REQUIRED_STAGES, 229)
REQUIRED_DESIGN_STAGES = (*n229.REQUIRED_DESIGN_STAGES, 229)
BOUNDARY_FLAGS = n229.BOUNDARY_FLAGS
EXPECTED_NEXT229_SOURCE_SHA256 = (
    "aedcb8241f529bdcfc22cf03557eaaa9a7d70ad170f4faf8f1b0dcce68499b68"
)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n229.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next229_design": n229.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next229_manifest": (
        "04eb558f666d9ed651f1013fd69b9947693ec655b604df54fb8ba6255b10277a"
    ),
    "next229_diagnostic": (
        "c2d754450c7db6ae24fab473efe61a3f97e43e157ccf955bb52e57e4c02524ce"
    ),
    "next229_table": (
        "6717774c35b262dfc5af333237058c76977df6aab964b18a2abe54c338af7376"
    ),
}


select_auditable_features = n227.select_auditable_features
audit_one_source = n227.audit_one_source
select_residual_hypothesis = n227.select_residual_hypothesis
build_rejected_extreme_cohort = n227.build_rejected_extreme_cohort


def _paths(
    *,
    roots: Mapping[str, Path],
    next135_freeze_path: Path,
    design_paths: Mapping[int, Path],
    design_path: Path,
) -> dict[str, Path]:
    paths = n229._paths(
        roots=roots,
        next135_freeze_path=next135_freeze_path,
        design_paths={
            stage: design_paths[stage] for stage in n229.REQUIRED_DESIGN_STAGES
        },
        design_path=design_paths[229],
    )
    paths["next229_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next229_manifest": roots["next229"] / n229.MANIFEST_NAME,
            "next229_diagnostic": roots["next229"] / n229.DIAGNOSTIC_NAME,
            "next229_table": roots["next229"] / n229.TABLE_NAME,
        }
    )
    return paths


def _verify_next229(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    str,
    str,
    dict[str, object],
    str,
    dict[str, object],
    tuple[str, ...],
    pd.DataFrame,
    dict[str, object],
]:
    prior_paths = dict(paths)
    prior_paths["design"] = paths["next229_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next229_design"]
    prior = n229._verify_next228(prior_paths, prior_hashes)
    manifest = json.loads(paths["next229_manifest"].read_text())
    diagnostic = json.loads(paths["next229_diagnostic"].read_text())
    table = pd.read_parquet(paths["next229_table"])
    closest = dict(diagnostic["global_closest"])
    key = str(closest["candidate_key"])
    expected_outputs = {
        n229.DIAGNOSTIC_NAME: input_hashes["next229_diagnostic"],
        n229.TABLE_NAME: input_hashes["next229_table"],
    }
    if (
        manifest.get("protocol") != n229.PROTOCOL
        or manifest.get("candidate_count") != n229.EXPECTED_CANDIDATE_COUNT
        or manifest.get("candidate_key_sha256")
        != n229.EXPECTED_CANDIDATE_KEY_SHA256
        or manifest.get("strict_improvement_over_next224_diagnostic") is not True
        or manifest.get("new_formula_searched") is not False
        or manifest.get("new_formula_selected") is not False
        or manifest.get("outputs_sha256") != expected_outputs
        or manifest.get("executed_source_sha256", {}).get(
            "src/next229_margin_local_broad_diagnostic.py"
        )
        != EXPECTED_NEXT229_SOURCE_SHA256
        or _sha256_file(Path(n229.__file__).resolve())
        != EXPECTED_NEXT229_SOURCE_SHA256
        or any(manifest.get(k) is not value for k, value in BOUNDARY_FLAGS.items())
        or diagnostic.get("protocol") != n229.PROTOCOL
        or diagnostic.get("candidate_count") != n229.EXPECTED_CANDIDATE_COUNT
        or diagnostic.get("new_formula_searched") is not False
        or hashlib.sha256(key.encode()).hexdigest()
        != EXPECTED_BASE_CANDIDATE_KEY_SHA256
        or int(closest["failed_constraint_count"]) != EXPECTED_BASE_FAILED_COUNT
        or not math.isclose(
            float(closest["normalized_shortfall_sum"]),
            EXPECTED_BASE_SHORTFALL,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            float(closest["best_threshold"]),
            EXPECTED_BASE_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or len(table) != n229.EXPECTED_CANDIDATE_COUNT
        or n229.candidate_key_sha256(table) != n229.EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT230 NEXT229 provenance differs")
    return (*prior, closest)


def _reconstruct_next229_frontier(
    *,
    paths: Mapping[str, Path],
    eligible: tuple[str, ...],
    eligible214: tuple[str, ...],
    primary_key: str,
    base_start_key: str,
    formula214: Mapping[str, object],
    current_key: str,
    formula222: Mapping[str, object],
    eligible227: tuple[str, ...],
    published228: pd.DataFrame,
    closest229: Mapping[str, object],
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    combined, feature_tables, score224, support, endpoint, _ = (
        n227._reconstruct_next224_frontier(
            paths=paths,
            eligible=eligible,
            eligible214=eligible214,
            primary_key=primary_key,
            base_start_key=base_start_key,
            formula214=formula214,
            current_key=current_key,
            formula222=formula222,
        )
    )
    diagnostic224 = json.loads(paths["next224_diagnostic"].read_text())
    base_key = str(diagnostic224["global_closest"]["candidate_key"])
    all_specs = n229.n228.build_margin_local_candidate_specs(
        base_candidate_key=base_key,
        eligible_hypotheses=eligible227,
        features=combined,
        base_score=score224,
        base_support=support,
    )
    key = str(closest229["candidate_key"])
    specs = [spec for spec in all_specs if str(spec["candidate_key"]) == key]
    if len(specs) != 1:
        raise ValueError("NEXT230 NEXT229 specification differs")
    virtual, terms, runtime, _ = n229.n228.materialize_margin_local_candidates(
        features=combined,
        base_score=score224,
        base_support=support,
        specs=specs,
    )
    n164 = n229.n228.n223.n222.n215.n214.n164
    score, got_support = n164._term_risk(virtual, terms[0])
    if (
        not np.array_equal(got_support, support)
        or int(support.sum()) != EXPECTED_BASE_SUPPORT_COUNT
    ):
        raise RuntimeError("NEXT230 NEXT229 support differs")
    expected = published228.loc[published228["candidate_key"].eq(key)]
    if len(expected) != 1:
        raise ValueError("NEXT230 NEXT228 published record differs")
    evaluator = (
        n229.n228.n223.n222.n215.n214.n212.n210.n208.n205.n203.n202.n200.n194.n130.n125
        .search_optional_guard_laws_parallel
    )
    rerun = evaluator(
        features=virtual,
        endpoint=endpoint,
        old_terms=terms,
        optional_terms=[],
        candidate_specs=runtime,
        workers=1,
    )
    n164._verify_reproduction(rerun=rerun["candidate_records"], published=expected)
    return combined, feature_tables, score, support, endpoint, specs[0]


def run_stagewise_margin_local_feature_audit(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    stage_dirs: Mapping[int, Path],
    next135_freeze_path: Path,
    design_paths: Mapping[int, Path],
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Audit the exact NEXT229 rejected frontier without searching a law."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs):
        raise ValueError("NEXT230 stage directory universe differs")
    if not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT230 design path universe differs")
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{
            f"next{stage}": Path(stage_dirs[stage]).resolve()
            for stage in REQUIRED_STAGES
        },
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots=roots,
        next135_freeze_path=Path(next135_freeze_path).resolve(),
        design_paths={
            stage: Path(design_paths[stage]).resolve()
            for stage in REQUIRED_DESIGN_STAGES
        },
        design_path=Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT230 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT230 formal input identity differs: {differing}")
    (
        eligible,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        current_key,
        formula222,
        eligible227,
        published228,
        closest229,
    ) = _verify_next229(paths, input_hashes)
    combined, _, score, support, endpoint, _ = _reconstruct_next229_frontier(
        paths=paths,
        eligible=eligible,
        eligible214=eligible214,
        primary_key=primary_key,
        base_start_key=base_start_key,
        formula214=formula214,
        current_key=current_key,
        formula222=formula222,
        eligible227=eligible227,
        published228=published228,
        closest229=closest229,
    )
    cohort = build_rejected_extreme_cohort(
        score=score,
        support=support,
        endpoint=endpoint,
        threshold=EXPECTED_BASE_THRESHOLD,
    )
    n164 = n229.n228.n223.n222.n215.n214.n164
    folds = n164.assign_group_folds(
        combined["reduced_formula"].astype(str).to_numpy()
    )
    source = combined["source_dataset"].astype(str).to_numpy()
    protected = endpoint <= 1.0
    severe = endpoint >= 2.0
    cohort_counts: dict[str, tuple[int, int]] = {}
    for source_name in ("scigen", "wyformer"):
        for fold in (None, 0, 1, 2, 3, 4):
            mask = cohort & (source == source_name)
            if fold is not None:
                mask &= folds == fold
            cell_id = f"{source_name}:{'all' if fold is None else f'fold{fold}'}"
            cohort_counts[cell_id] = (
                int((mask & protected).sum()),
                int((mask & severe).sum()),
            )
    if cohort_counts != EXPECTED_COHORT_COUNTS:
        raise ValueError("NEXT230 rejected cohort counts differ")
    feature_names = select_auditable_features(combined)
    feature_sha = hashlib.sha256("\n".join(feature_names).encode()).hexdigest()
    if (
        len(feature_names) != EXPECTED_FEATURE_COUNT
        or feature_sha != EXPECTED_FEATURE_NAME_SHA256
    ):
        raise ValueError("NEXT230 frozen feature universe differs")

    rows: list[dict[str, object]] = []
    for feature in feature_names:
        values = pd.to_numeric(combined[feature], errors="coerce").to_numpy(float)
        for direction in PROTECTION_DIRECTIONS:
            source_results = {}
            for source_name in ("scigen", "wyformer"):
                mask = source == source_name
                source_results[source_name] = audit_one_source(
                    values=values[mask],
                    endpoint=endpoint[mask],
                    cohort=cohort[mask],
                    folds=folds[mask],
                    direction=direction,
                )
            scigen = source_results["scigen"]
            wyformer = source_results["wyformer"]
            aggregate_aucs = [
                float(scigen["aggregate_auc"]),
                float(wyformer["aggregate_auc"]),
            ]
            rows.append(
                {
                    "hypothesis": f"{feature}__{direction}",
                    "feature": feature,
                    "direction": direction,
                    "passes_raw_gates": bool(
                        scigen["passes_source_gates"]
                        and wyformer["passes_source_gates"]
                    ),
                    "ranking_min_worst_fold_auc": float(
                        min(
                            float(scigen["worst_fold_auc"]),
                            float(wyformer["worst_fold_auc"]),
                        )
                    ),
                    "ranking_min_aggregate_auc": float(min(aggregate_aucs)),
                    "ranking_mean_aggregate_auc": float(np.mean(aggregate_aucs)),
                    "scigen_aggregate_auc": scigen["aggregate_auc"],
                    "scigen_macro_fold_auc": scigen["macro_fold_auc"],
                    "scigen_worst_fold_auc": scigen["worst_fold_auc"],
                    "scigen_minimum_cell_coverage": scigen[
                        "minimum_cell_coverage"
                    ],
                    "wyformer_aggregate_auc": wyformer["aggregate_auc"],
                    "wyformer_macro_fold_auc": wyformer["macro_fold_auc"],
                    "wyformer_worst_fold_auc": wyformer["worst_fold_auc"],
                    "wyformer_minimum_cell_coverage": wyformer[
                        "minimum_cell_coverage"
                    ],
                    "source_audits_json": json.dumps(
                        source_results, sort_keys=True, separators=(",", ":")
                    ),
                }
            )
    raw_table = pd.DataFrame(rows)
    if len(raw_table) != EXPECTED_HYPOTHESIS_COUNT:
        raise RuntimeError("NEXT230 hypothesis count differs")
    table, selected = select_residual_hypothesis(raw_table)
    eligible_table = table.loc[table["eligible_for_search"]]
    eligible_names = sorted(eligible_table["hypothesis"].astype(str))
    eligible_sha = hashlib.sha256("\n".join(eligible_names).encode()).hexdigest()

    catalogue = {
        "protocol": PROTOCOL,
        "design_sha256": input_hashes["design"],
        "base_candidate_key_sha256": EXPECTED_BASE_CANDIDATE_KEY_SHA256,
        "base_threshold": EXPECTED_BASE_THRESHOLD,
        "feature_selection_policy": {
            "blocked_exact_names": sorted(n207.BLOCKED_EXACT_NAMES),
            "blocked_prefixes": ["_", "pauling_"],
            "blocked_suffixes": ["_supported", "_site_count", "_edge_count"],
            "numeric_only": True,
        },
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "feature_name_sha256": feature_sha,
        "directions": PROTECTION_DIRECTIONS,
        "hypothesis_count": len(table),
        "gates": {
            "minimum_coverage": MINIMUM_COVERAGE,
            "minimum_class_count": MINIMUM_CLASS_COUNT,
            "minimum_aggregate_auc": MINIMUM_AGGREGATE_AUC,
            "minimum_macro_auc": MINIMUM_MACRO_AUC,
            "minimum_worst_auc": MINIMUM_WORST_AUC,
            "opposite_direction_veto": True,
        },
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    audit = {
        "protocol": PROTOCOL,
        "audit_mode": "fixed_next229_rejected_extreme_margin_feature_audit",
        "base_candidate_key_sha256": EXPECTED_BASE_CANDIDATE_KEY_SHA256,
        "base_threshold": EXPECTED_BASE_THRESHOLD,
        "base_failed_constraint_count": EXPECTED_BASE_FAILED_COUNT,
        "base_normalized_shortfall_sum": EXPECTED_BASE_SHORTFALL,
        "cohort_counts": {
            key: {
                "protected_rejected": value[0],
                "severe_rejected": value[1],
            }
            for key, value in sorted(cohort_counts.items())
        },
        "feature_count": len(feature_names),
        "hypothesis_count": len(table),
        "raw_gate_passing_count": int(table["passes_raw_gates"].sum()),
        "eligible_hypothesis_count": int(len(eligible_table)),
        "eligible_hypotheses": eligible_names,
        "eligible_hypothesis_sha256": eligible_sha,
        "selected_hypothesis": selected,
        "next231_search_authorized": bool(selected is not None),
        "new_formula_searched": False,
        "new_formula_selected": False,
        "validation_or_replication_opened": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    source_paths = {
        "src/next207_residual_x0_feature_audit.py": Path(n207.__file__).resolve(),
        "src/next229_margin_local_broad_diagnostic.py": Path(n229.__file__).resolve(),
        "src/next230_stagewise_margin_local_feature_audit.py": Path(
            __file__
        ).resolve(),
    }
    source_hashes = {
        name: _sha256_file(path) for name, path in source_paths.items()
    }
    try:
        catalogue_path = staging / CATALOGUE_NAME
        audit_path = staging / AUDIT_NAME
        table_path = staging / TABLE_NAME
        _write_json(catalogue_path, catalogue)
        _write_json(audit_path, audit)
        table.to_parquet(table_path, index=False)
        outputs = [catalogue_path, audit_path, table_path]
        manifest = {
            "protocol": PROTOCOL,
            "feature_count": len(feature_names),
            "feature_name_sha256": feature_sha,
            "hypothesis_count": len(table),
            "eligible_hypothesis_count": int(len(eligible_table)),
            "eligible_hypothesis_sha256": eligible_sha,
            "next229_frontier_reproduced": True,
            "next231_search_authorized": bool(selected is not None),
            "stagewise_margin_local_branch_terminated": selected is None,
            "new_formula_searched": False,
            "new_formula_selected": False,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True,
            **BOUNDARY_FLAGS,
            "scientific_improvement_claim": False,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {
                path.name: _sha256_file(path) for path in outputs
            },
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(
            _sha256_file(path) != input_hashes[name]
            for name, path in paths.items()
        ):
            raise RuntimeError("NEXT230 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT230 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scigen-feature-dir", type=Path, required=True)
    parser.add_argument(
        "--scigen-discovery-endpoint-dir", type=Path, required=True
    )
    parser.add_argument("--wyformer-feature-dir", type=Path, required=True)
    parser.add_argument(
        "--wyformer-discovery-endpoint-dir", type=Path, required=True
    )
    for stage in REQUIRED_STAGES:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    for stage in REQUIRED_DESIGN_STAGES:
        parser.add_argument(f"--next{stage}-design-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_stagewise_margin_local_feature_audit(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        stage_dirs={
            stage: getattr(args, f"next{stage}_dir") for stage in REQUIRED_STAGES
        },
        next135_freeze_path=args.next135_freeze_path,
        design_paths={
            stage: getattr(args, f"next{stage}_design_path")
            for stage in REQUIRED_DESIGN_STAGES
        },
        design_path=args.design_path,
        output_dir=args.output_dir,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "audit_one_source",
    "build_rejected_extreme_cohort",
    "run_stagewise_margin_local_feature_audit",
    "select_auditable_features",
    "select_residual_hypothesis",
]


if __name__ == "__main__":
    raise SystemExit(main())
