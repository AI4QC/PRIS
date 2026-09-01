#!/usr/bin/env python3
"""Audit raw x0 features inside the exact NEXT224 rejected frontier."""

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

import src.next207_residual_x0_feature_audit as n207
import src.next226_agreement_gated_broad_diagnostic as n226
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next227-margin-local-feature-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT227_MARGIN_LOCAL_FEATURE_CATALOGUE.json"
AUDIT_NAME = "NEXT227_MARGIN_LOCAL_FEATURE_AUDIT.json"
TABLE_NAME = "next227_margin_local_feature_audit.parquet"
EXPECTED_DESIGN_SHA256 = (
    "abad6c2f3fdf87399da6eafc0f2d46d6cff545b7ffc273adeb1ca4060d0ed3c1"
)
EXPECTED_BASE_CANDIDATE_KEY_SHA256 = (
    "3f87102463cc283bcb3e4d1c45e434e04c7f7d2d32167801b79d7db8035559e4"
)
EXPECTED_BASE_THRESHOLD = 0.1520033762332462
EXPECTED_BASE_FAILED_COUNT = 6
EXPECTED_BASE_SHORTFALL = 0.1461217358987499
EXPECTED_BASE_SUPPORT_COUNT = 18017
EXPECTED_FEATURE_COUNT = n207.EXPECTED_FEATURE_COUNT
EXPECTED_FEATURE_NAME_SHA256 = n207.EXPECTED_FEATURE_NAME_SHA256
EXPECTED_HYPOTHESIS_COUNT = n207.EXPECTED_HYPOTHESIS_COUNT
PROTECTION_DIRECTIONS = n207.PROTECTION_DIRECTIONS
MINIMUM_COVERAGE = n207.MINIMUM_COVERAGE
MINIMUM_CLASS_COUNT = n207.MINIMUM_CLASS_COUNT
MINIMUM_AGGREGATE_AUC = n207.MINIMUM_AGGREGATE_AUC
MINIMUM_MACRO_AUC = n207.MINIMUM_MACRO_AUC
MINIMUM_WORST_AUC = n207.MINIMUM_WORST_AUC
EXPECTED_COHORT_COUNTS = {
    "scigen:all": (231, 2895),
    "scigen:fold0": (47, 601),
    "scigen:fold1": (49, 582),
    "scigen:fold2": (45, 572),
    "scigen:fold3": (46, 578),
    "scigen:fold4": (44, 562),
    "wyformer:all": (317, 508),
    "wyformer:fold0": (80, 95),
    "wyformer:fold1": (52, 101),
    "wyformer:fold2": (65, 112),
    "wyformer:fold3": (67, 106),
    "wyformer:fold4": (53, 94),
}
REQUIRED_STAGES = (*n226.REQUIRED_STAGES, 226)
REQUIRED_DESIGN_STAGES = (*n226.REQUIRED_DESIGN_STAGES, 226)
BOUNDARY_FLAGS = n226.BOUNDARY_FLAGS
EXPECTED_NEXT226_SOURCE_SHA256 = (
    "7ac1db914e04ff0dab5b77c127d6713ce8074593ba42c6dcaa06266874bffeeb"
)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n226.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next226_design": n226.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next226_manifest": (
        "01b81007835f2a7f7fce6626530c97a16baaad662dfd16ec30b56711c21e1191"
    ),
    "next226_diagnostic": (
        "37ab991f4b494200d4602c29cabf788d47c7aae1ec45cb5b0254274672710900"
    ),
    "next226_table": (
        "b33c3d9d6b0565fca2d49685a802d0f2f8d32453380a70ee21d9f74159fd2567"
    ),
}


select_auditable_features = n207.select_auditable_features
audit_one_source = n207.audit_one_source
select_residual_hypothesis = n207.select_residual_hypothesis


def build_rejected_extreme_cohort(
    *, score: object, support: object, endpoint: object, threshold: float
) -> np.ndarray:
    """Return supported, finite, rejected protected/severe discovery rows."""

    values = np.asarray(score, dtype=float)
    supported = np.asarray(support, dtype=bool)
    outcomes = np.asarray(endpoint, dtype=float)
    cutoff = float(threshold)
    if (
        values.ndim != 1
        or supported.shape != values.shape
        or outcomes.shape != values.shape
        or not math.isfinite(cutoff)
        or np.any(~np.isfinite(outcomes))
    ):
        raise ValueError("NEXT227 rejected cohort inputs differ")
    extreme = (outcomes <= 1.0) | (outcomes >= 2.0)
    return supported & np.isfinite(values) & (values >= cutoff) & extreme


def _paths(
    *,
    roots: Mapping[str, Path],
    next135_freeze_path: Path,
    design_paths: Mapping[int, Path],
    design_path: Path,
) -> dict[str, Path]:
    paths = n226._paths(
        roots=roots,
        next135_freeze_path=next135_freeze_path,
        design_paths={
            stage: design_paths[stage]
            for stage in n226.REQUIRED_DESIGN_STAGES
        },
        design_path=design_paths[226],
    )
    paths["next226_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next226_manifest": roots["next226"] / n226.MANIFEST_NAME,
            "next226_diagnostic": roots["next226"] / n226.DIAGNOSTIC_NAME,
            "next226_table": roots["next226"] / n226.TABLE_NAME,
        }
    )
    return paths


def _verify_next226(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    str,
    str,
    dict[str, object],
    str,
    dict[str, object],
]:
    prior_paths = dict(paths)
    prior_paths["design"] = paths["next226_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next226_design"]
    (
        eligible,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        current_key,
        formula222,
        _,
        _,
    ) = n226._verify_next225(prior_paths, prior_hashes)
    manifest = json.loads(paths["next226_manifest"].read_text())
    diagnostic = json.loads(paths["next226_diagnostic"].read_text())
    table = pd.read_parquet(paths["next226_table"])
    expected_outputs = {
        n226.DIAGNOSTIC_NAME: input_hashes["next226_diagnostic"],
        n226.TABLE_NAME: input_hashes["next226_table"],
    }
    if (
        manifest.get("protocol") != n226.PROTOCOL
        or manifest.get("candidate_count") != n226.EXPECTED_CANDIDATE_COUNT
        or manifest.get("candidate_key_sha256")
        != n226.EXPECTED_CANDIDATE_KEY_SHA256
        or manifest.get("strict_residual_improvement_observed") is not True
        or manifest.get("strict_improvement_over_next224_diagnostic") is not False
        or manifest.get("new_formula_searched") is not False
        or manifest.get("new_formula_selected") is not False
        or manifest.get("outputs_sha256") != expected_outputs
        or any(
            manifest.get(key) is not value
            for key, value in BOUNDARY_FLAGS.items()
        )
        or manifest.get("executed_source_sha256", {}).get(
            "src/next226_agreement_gated_broad_diagnostic.py"
        )
        != EXPECTED_NEXT226_SOURCE_SHA256
        or _sha256_file(Path(n226.__file__).resolve())
        != EXPECTED_NEXT226_SOURCE_SHA256
        or diagnostic.get("protocol") != n226.PROTOCOL
        or diagnostic.get("candidate_count") != n226.EXPECTED_CANDIDATE_COUNT
        or diagnostic.get("new_formula_searched") is not False
        or len(table) != n226.EXPECTED_CANDIDATE_COUNT
        or n226.candidate_key_sha256(table)
        != n226.EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT227 NEXT226 provenance differs")
    return (
        eligible,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        current_key,
        formula222,
    )


def _reconstruct_next224_frontier(
    *,
    paths: Mapping[str, Path],
    eligible: tuple[str, ...],
    eligible214: tuple[str, ...],
    primary_key: str,
    base_start_key: str,
    formula214: Mapping[str, object],
    current_key: str,
    formula222: Mapping[str, object],
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    n223 = n226.n225.n223
    n222 = n223.n222
    combined, feature_tables, base_score, support, endpoint = (
        n222.n215._reconstruct_next214_final(
            paths=paths,
            eligible=eligible214,
            primary_key=primary_key,
            start_key=base_start_key,
            formula=formula214,
        )
    )
    current = n223._reconstruct_next222_delta(
        features=combined,
        base_score=base_score,
        support=support,
        formula=formula222,
    )
    next214_table = pd.read_parquet(paths["next214_search"])
    accepted = next214_table.loc[
        next214_table["depth"].eq(3)
        & next214_table["proposed_hypothesis"].eq(
            "steric_overlap2_vector_q95__protected_low"
        )
        & next214_table["proposed_amplitude_fraction"].eq(0.0625)
    ]
    if len(accepted) != 1:
        raise ValueError("NEXT227 NEXT214 base identity differs")
    initial = n222.n220.build_signed_candidate_specs(
        base_candidate_key=str(accepted.iloc[0]["candidate_key"]),
        eligible_hypotheses=eligible,
        features=combined,
        base_score=base_score,
        base_support=support,
    )
    normalizations = {
        str(spec["hypothesis"]): dict(spec)
        for spec in initial
        if spec["hypothesis"] is not None
    }
    all_specs = n223.build_dual_evidence_candidate_specs(
        current_path_key=current_key,
        current_terms=[dict(value) for value in formula222["terms"]],
        normalizations=normalizations,
    )
    diagnostic = json.loads(paths["next224_diagnostic"].read_text())
    closest = dict(diagnostic["global_closest"])
    key = str(closest["candidate_key"])
    if (
        hashlib.sha256(key.encode()).hexdigest()
        != EXPECTED_BASE_CANDIDATE_KEY_SHA256
        or int(closest["failed_constraint_count"])
        != EXPECTED_BASE_FAILED_COUNT
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
    ):
        raise ValueError("NEXT227 NEXT224 frontier identity differs")
    specs = [spec for spec in all_specs if str(spec["candidate_key"]) == key]
    if len(specs) != 1:
        raise ValueError("NEXT227 NEXT224 specification differs")
    virtual, terms, _, _ = n223.materialize_dual_evidence_candidates(
        features=combined,
        base_score=base_score,
        current_delta=current,
        base_support=support,
        specs=specs,
    )
    score, got_support = n222.n215.n214.n164._term_risk(virtual, terms[0])
    if (
        not np.array_equal(got_support, support)
        or int(support.sum()) != EXPECTED_BASE_SUPPORT_COUNT
    ):
        raise RuntimeError("NEXT227 NEXT224 support differs")
    return combined, feature_tables, score, support, endpoint, base_score


def run_margin_local_feature_audit(
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
    """Audit the frozen NEXT224 rejected frontier without searching a law."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs):
        raise ValueError("NEXT227 stage directory universe differs")
    if not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT227 design path universe differs")
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
        raise FileNotFoundError("NEXT227 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT227 formal input identity differs: {differing}")
    (
        eligible,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        current_key,
        formula222,
    ) = _verify_next226(paths, input_hashes)
    combined, _, score, support, endpoint, _ = _reconstruct_next224_frontier(
        paths=paths,
        eligible=eligible,
        eligible214=eligible214,
        primary_key=primary_key,
        base_start_key=base_start_key,
        formula214=formula214,
        current_key=current_key,
        formula222=formula222,
    )
    cohort = build_rejected_extreme_cohort(
        score=score,
        support=support,
        endpoint=endpoint,
        threshold=EXPECTED_BASE_THRESHOLD,
    )
    n164 = n226.n225.n222.n215.n214.n164
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
        raise ValueError("NEXT227 rejected cohort counts differ")
    feature_names = select_auditable_features(combined)
    feature_sha = hashlib.sha256("\n".join(feature_names).encode()).hexdigest()
    if (
        len(feature_names) != EXPECTED_FEATURE_COUNT
        or feature_sha != EXPECTED_FEATURE_NAME_SHA256
    ):
        raise ValueError("NEXT227 frozen feature universe differs")

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
        raise RuntimeError("NEXT227 hypothesis count differs")
    table, selected = select_residual_hypothesis(raw_table)
    eligible_table = table.loc[table["eligible_for_search"]]
    eligible_names = sorted(eligible_table["hypothesis"].astype(str))

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
        "audit_mode": "fixed_next224_rejected_extreme_margin_feature_audit",
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
        "eligible_hypothesis_sha256": hashlib.sha256(
            "\n".join(eligible_names).encode()
        ).hexdigest(),
        "selected_hypothesis": selected,
        "next228_search_authorized": bool(selected is not None),
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
        "src/next226_agreement_gated_broad_diagnostic.py": Path(
            n226.__file__
        ).resolve(),
        "src/next227_margin_local_feature_audit.py": Path(__file__).resolve(),
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
            "eligible_hypothesis_sha256": audit["eligible_hypothesis_sha256"],
            "next224_frontier_reproduced": True,
            "next228_search_authorized": bool(selected is not None),
            "margin_local_branch_terminated": selected is None,
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
            raise RuntimeError("NEXT227 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT227 source changed before publication")
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
        parser.add_argument(
            f"--next{stage}-design-path", type=Path, required=True
        )
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_margin_local_feature_audit(
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
    "run_margin_local_feature_audit",
    "select_auditable_features",
    "select_residual_hypothesis",
]


if __name__ == "__main__":
    raise SystemExit(main())
