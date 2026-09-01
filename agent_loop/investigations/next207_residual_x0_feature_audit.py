#!/usr/bin/env python3
"""Audit raw x0 features inside the frozen NEXT206 rejected cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import src.next206_motif_exception_depth_broad_residual as n206
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


n205 = n206.n205
PROTOCOL = "2026-08-08-next207-residual-x0-feature-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT207_RESIDUAL_X0_FEATURE_CATALOGUE.json"
AUDIT_NAME = "NEXT207_RESIDUAL_X0_FEATURE_AUDIT.json"
TABLE_NAME = "next207_residual_x0_feature_audit.parquet"
EXPECTED_DESIGN_SHA256 = (
    "9818f9b38c8ad78d02f7d635709b1017d09bc94a02e4bd61016e0e80454291c4"
)
EXPECTED_CANDIDATE_KEY_SHA256 = (
    "26ee85c8dbb8f810eb5baf8c8be07f61d390f2c02c9e45d147c786212b7acc38"
)
EXPECTED_RESIDUAL_THRESHOLD = 0.16344427817025572
EXPECTED_FEATURE_COUNT = 242
EXPECTED_FEATURE_NAME_SHA256 = (
    "87a20f191ca47b6fb3e9f0255ae8d1e98bcf41e21991af3d290ff222c446f07c"
)
PROTECTION_DIRECTIONS = ("protected_high", "protected_low")
EXPECTED_HYPOTHESIS_COUNT = EXPECTED_FEATURE_COUNT * len(PROTECTION_DIRECTIONS)
MINIMUM_COVERAGE = 0.90
MINIMUM_CLASS_COUNT = 20
MINIMUM_AGGREGATE_AUC = 0.55
MINIMUM_MACRO_AUC = 0.53
MINIMUM_WORST_AUC = 0.50
BLOCKED_EXACT_NAMES = {
    "raw_material_id",
    "source_member_bytes",
    "generated_space_group",
    "natoms",
    "geom_species_count",
}
EXPECTED_COHORT_COUNTS = {
    "scigen:all": (316, 3110),
    "scigen:fold0": (59, 643),
    "scigen:fold1": (67, 618),
    "scigen:fold2": (64, 626),
    "scigen:fold3": (59, 614),
    "scigen:fold4": (67, 609),
    "wyformer:all": (331, 521),
    "wyformer:fold0": (75, 95),
    "wyformer:fold1": (56, 101),
    "wyformer:fold2": (69, 115),
    "wyformer:fold3": (75, 114),
    "wyformer:fold4": (56, 96),
}
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n206.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next205_design": n206.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next206_manifest": "ebe97175c2bf50242ab7113263729803e6ca0cc13fd589d556f0ed7eae392f74",
    "next206_diagnostic": "a657bfc54da364ef2f90cce3bc0d665944f8ce522aac8b321c9a9c94d3b43af4",
    "next206_table": "6555b691556bf1102d03c0fb451155f5b4619532d3baa86898cc8fcc322ec58e",
}


def blocked_feature_name(name: str) -> bool:
    """Return whether a numeric column is forbidden by the frozen schema."""

    value = str(name)
    return (
        value in BLOCKED_EXACT_NAMES
        or value.startswith("_")
        or value.startswith("pauling_")
        or value.endswith("_supported")
        or value.endswith("_site_count")
        or value.endswith("_edge_count")
    )


def select_auditable_features(frame: pd.DataFrame) -> tuple[str, ...]:
    """Select the exact sorted numeric, raw, non-identifier x0 columns."""

    if not isinstance(frame, pd.DataFrame) or frame.columns.astype(str).duplicated().any():
        raise ValueError("NEXT207 feature schema differs")
    return tuple(
        sorted(
            str(name)
            for name in frame.columns
            if not blocked_feature_name(str(name))
            and pd.api.types.is_numeric_dtype(frame[name])
        )
    )


def directional_risk(values: object, direction: str) -> np.ndarray:
    """Map an exact protection direction to a severe-positive risk score."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or direction not in PROTECTION_DIRECTIONS:
        raise ValueError("NEXT207 protection direction differs")
    return -array if direction == "protected_high" else array.copy()


def audit_one_source(
    *,
    values: object,
    endpoint: object,
    cohort: object,
    folds: object,
    direction: str,
    expected_folds: Sequence[int] = tuple(range(5)),
    minimum_coverage: float = MINIMUM_COVERAGE,
    minimum_class_count: int = MINIMUM_CLASS_COUNT,
    minimum_aggregate_auc: float = MINIMUM_AGGREGATE_AUC,
    minimum_macro_auc: float = MINIMUM_MACRO_AUC,
    minimum_worst_auc: float = MINIMUM_WORST_AUC,
) -> dict[str, object]:
    """Audit one directional feature within one source and its frozen folds."""

    raw = np.asarray(values, dtype=float)
    outcome = np.asarray(endpoint, dtype=float)
    selected = np.asarray(cohort, dtype=bool)
    fold_values = np.asarray(folds, dtype=int)
    expected = tuple(int(value) for value in expected_folds)
    if (
        raw.ndim != 1
        or outcome.shape != raw.shape
        or selected.shape != raw.shape
        or fold_values.shape != raw.shape
        or not expected
        or len(set(expected)) != len(expected)
        or type(minimum_class_count) is not int
        or minimum_class_count <= 0
        or any(
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            for value in (
                minimum_coverage,
                minimum_aggregate_auc,
                minimum_macro_auc,
                minimum_worst_auc,
            )
        )
    ):
        raise ValueError("NEXT207 source audit inputs differ")
    risk = directional_risk(raw, direction)
    extreme = (outcome <= 1.0) | (outcome >= 2.0)
    base = selected & extreme
    finite = np.isfinite(risk)
    records: list[dict[str, object]] = []
    aucs: list[float] = []
    for fold in (None, *expected):
        cell = base & ((fold_values == fold) if fold is not None else True)
        supported = cell & finite
        protected_count = int((supported & (outcome <= 1.0)).sum())
        severe_count = int((supported & (outcome >= 2.0)).sum())
        coverage = float(supported.sum() / max(int(cell.sum()), 1))
        auc = (
            n205.n203.n202.n200.n194.n87._roc_auc(
                risk[supported], outcome[supported] >= 2.0
            )
            if protected_count and severe_count
            else None
        )
        count_and_coverage_pass = bool(
            int(cell.sum()) > 0
            and coverage >= float(minimum_coverage)
            and protected_count >= minimum_class_count
            and severe_count >= minimum_class_count
        )
        record = {
            "fold": fold,
            "cohort_rows": int(cell.sum()),
            "supported_rows": int(supported.sum()),
            "coverage": coverage,
            "protected": protected_count,
            "severe": severe_count,
            "auc": auc,
            "passes_count_and_coverage": count_and_coverage_pass,
        }
        records.append(record)
        if fold is not None and auc is not None:
            aucs.append(float(auc))
    aggregate = records[0]
    macro = float(np.mean(aucs)) if len(aucs) == len(expected) else None
    worst = float(np.min(aucs)) if len(aucs) == len(expected) else None
    passes = bool(
        all(bool(record["passes_count_and_coverage"]) for record in records)
        and aggregate["auc"] is not None
        and float(aggregate["auc"]) >= float(minimum_aggregate_auc)
        and macro is not None
        and macro >= float(minimum_macro_auc)
        and worst is not None
        and worst >= float(minimum_worst_auc)
    )
    return {
        "aggregate_auc": aggregate["auc"],
        "macro_fold_auc": macro,
        "worst_fold_auc": worst,
        "minimum_cell_coverage": float(
            min(float(record["coverage"]) for record in records)
        ),
        "passes_source_gates": passes,
        "cell_records": records,
    }


def select_residual_hypothesis(
    records: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    """Apply the opposite-direction veto and frozen deterministic ranking."""

    required = {
        "hypothesis",
        "feature",
        "direction",
        "passes_raw_gates",
        "ranking_min_worst_fold_auc",
        "ranking_min_aggregate_auc",
        "ranking_mean_aggregate_auc",
    }
    if (
        not isinstance(records, pd.DataFrame)
        or required - set(records.columns)
        or records["hypothesis"].astype(str).duplicated().any()
        or set(records["direction"].astype(str)) - set(PROTECTION_DIRECTIONS)
    ):
        raise ValueError("NEXT207 audit record schema differs")
    table = records.copy()
    raw = table["passes_raw_gates"].fillna(False).astype(bool)
    passing_per_feature = raw.groupby(table["feature"].astype(str)).transform("sum")
    table["opposite_direction_veto_passed"] = passing_per_feature == 1
    table["eligible_for_search"] = raw & table["opposite_direction_veto_passed"]
    table = table.sort_values(
        [
            "eligible_for_search",
            "ranking_min_worst_fold_auc",
            "ranking_min_aggregate_auc",
            "ranking_mean_aggregate_auc",
            "hypothesis",
        ],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    eligible = table.loc[table["eligible_for_search"]]
    selected = None if eligible.empty else eligible.iloc[0].to_dict()
    return table, selected


def _paths(
    roots: Mapping[str, Path],
    freeze_path: Path,
    next202_design_path: Path,
    next205_design_path: Path,
    design_path: Path,
) -> dict[str, Path]:
    paths = n206._paths(
        roots, freeze_path, next202_design_path, next205_design_path
    )
    paths["next205_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next206_manifest": roots["next206"] / n206.MANIFEST_NAME,
            "next206_diagnostic": roots["next206"] / n206.DIAGNOSTIC_NAME,
            "next206_table": roots["next206"] / n206.PER_CANDIDATE_NAME,
        }
    )
    return paths


def _verify_prior(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> dict[str, object]:
    prior_paths = dict(paths)
    prior_paths["design"] = paths["next205_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next205_design"]
    n206._verify_next205(prior_paths, prior_hashes)
    manifest = json.loads(paths["next206_manifest"].read_text())
    diagnostic = json.loads(paths["next206_diagnostic"].read_text())
    expected_outputs = {
        n206.DIAGNOSTIC_NAME: input_hashes["next206_diagnostic"],
        n206.PER_CANDIDATE_NAME: input_hashes["next206_table"],
    }
    if (
        manifest.get("protocol") != n206.PROTOCOL
        or manifest.get("candidate_count") != n206.EXPECTED_CANDIDATE_COUNT
        or manifest.get("candidate_key_sha256")
        != n206.EXPECTED_CANDIDATE_KEY_SHA256
        or manifest.get("next205_records_reproduced") is not True
        or manifest.get("motif_exception_depth_broad_residual_diagnosed")
        is not True
        or manifest.get("motif_exception_depth_branch_closed") is not True
        or manifest.get("new_formula_searched") is not False
        or manifest.get("new_formula_selected") is not False
        or manifest.get("opened_validation_outputs_used") is not False
        or manifest.get("scigen_replication_endpoint_opened") is not False
        or manifest.get("wyformer_replication_endpoint_opened") is not False
        or manifest.get("dft_calculation_executed") is not False
        or manifest.get("dft_values_used_by_executable_formula") is not False
        or manifest.get("learned_energy_force_stress_proxy_used") is not False
        or manifest.get("model_or_proxy_potential_used") is not False
        or manifest.get("physical_relaxation_executed") is not False
        or manifest.get("outputs_sha256") != expected_outputs
        or manifest.get("executed_source_sha256", {}).get(
            "src/next206_motif_exception_depth_broad_residual.py"
        )
        != _sha256_file(Path(n206.__file__).resolve())
        or diagnostic.get("protocol") != n206.PROTOCOL
        or diagnostic.get("candidate_count") != n206.EXPECTED_CANDIDATE_COUNT
        or diagnostic.get("candidate_key_sha256")
        != n206.EXPECTED_CANDIDATE_KEY_SHA256
        or diagnostic.get("new_formula_searched") is not False
        or diagnostic.get("validation_outputs_opened") is not False
    ):
        raise ValueError("NEXT207 NEXT206 provenance differs")
    return diagnostic


def run_residual_x0_feature_audit(
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
    next163_dir: Path,
    next164_dir: Path,
    next168_dir: Path,
    next173_dir: Path,
    next179_dir: Path,
    next180_dir: Path,
    next181_dir: Path,
    next182_dir: Path,
    next183_dir: Path,
    next184_dir: Path,
    next185_dir: Path,
    next186_dir: Path,
    next188_dir: Path,
    next190_dir: Path,
    next192_dir: Path,
    next194_dir: Path,
    next199_dir: Path,
    next200_dir: Path,
    next201_dir: Path,
    next202_dir: Path,
    next203_dir: Path,
    next204_dir: Path,
    next205_dir: Path,
    next206_dir: Path,
    next135_freeze_path: Path,
    next202_design_path: Path,
    next205_design_path: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT207 residual feature audit."""

    stage_values = (
        (98, next98_dir), (110, next110_dir), (111, next111_dir),
        (113, next113_dir), (114, next114_dir), (116, next116_dir),
        (117, next117_dir), (120, next120_dir), (121, next121_dir),
        (122, next122_dir), (124, next124_dir), (125, next125_dir),
        (129, next129_dir), (130, next130_dir), (133, next133_dir),
        (134, next134_dir), (163, next163_dir), (164, next164_dir),
        (168, next168_dir), (173, next173_dir), (179, next179_dir),
        (180, next180_dir), (181, next181_dir), (182, next182_dir),
        (183, next183_dir), (184, next184_dir), (185, next185_dir),
        (186, next186_dir), (188, next188_dir), (190, next190_dir),
        (192, next192_dir),
    )
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{f"next{stage}": Path(value).resolve() for stage, value in stage_values},
        "next194": Path(next194_dir).resolve(),
        "next199": Path(next199_dir).resolve(),
        "next200": Path(next200_dir).resolve(),
        "next201": Path(next201_dir).resolve(),
        "next202": Path(next202_dir).resolve(),
        "next203": Path(next203_dir).resolve(),
        "next204": Path(next204_dir).resolve(),
        "next205": Path(next205_dir).resolve(),
        "next206": Path(next206_dir).resolve(),
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots,
        Path(next135_freeze_path).resolve(),
        Path(next202_design_path).resolve(),
        Path(next205_design_path).resolve(),
        Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT207 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT207 formal input identity differs: {differing}")
    diagnostic206 = _verify_prior(paths, input_hashes)
    eligible_names = json.loads(paths["next202_audit"].read_text())[
        "eligible_hypotheses"
    ]
    (
        combined,
        _,
        base_key,
        base_score,
        base_support,
        certificates,
        endpoint,
    ) = n205._reconstruct_discovery(paths=paths, eligible_names=eligible_names)

    closest = diagnostic206.get("global_closest", {})
    candidate_key = str(closest.get("candidate_key", ""))
    threshold = float(closest.get("best_threshold", float("nan")))
    if (
        hashlib.sha256(candidate_key.encode()).hexdigest()
        != EXPECTED_CANDIDATE_KEY_SHA256
        or threshold != EXPECTED_RESIDUAL_THRESHOLD
    ):
        raise ValueError("NEXT207 residual candidate identity differs")
    spec = json.loads(candidate_key)
    certificate_name = str(spec.get("certificate_hypothesis", ""))
    if certificate_name not in certificates:
        raise ValueError("NEXT207 residual certificate identity differs")
    score, support, _ = n205.motif_exception_depth_score(
        base_score=base_score,
        base_support=base_support,
        certificate=certificates[certificate_name],
        certificate_cutoff=float(spec["certificate_cutoff"]),
        pardon_depth=float(spec["pardon_depth"]),
    )
    source = combined["source_dataset"].astype(str).to_numpy()
    folds = n205.n203.n202.n200.n194.n87.assign_group_folds(
        combined["reduced_formula"].astype(str).to_numpy()
    )
    extreme = (endpoint <= 1.0) | (endpoint >= 2.0)
    cohort = support & np.isfinite(score) & (score >= threshold) & extreme
    cohort_counts = {}
    for source_name in ("scigen", "wyformer"):
        source_mask = source == source_name
        for fold in (None, *range(5)):
            mask = cohort & source_mask & (
                (folds == fold) if fold is not None else True
            )
            cell_id = f"{source_name}:{'all' if fold is None else f'fold{fold}'}"
            cohort_counts[cell_id] = (
                int((mask & (endpoint <= 1.0)).sum()),
                int((mask & (endpoint >= 2.0)).sum()),
            )
    if cohort_counts != EXPECTED_COHORT_COUNTS:
        raise ValueError("NEXT207 residual cohort accounting differs")

    feature_names = select_auditable_features(combined)
    feature_sha = hashlib.sha256("\n".join(feature_names).encode()).hexdigest()
    if require_formal_inputs and (
        len(feature_names) != EXPECTED_FEATURE_COUNT
        or feature_sha != EXPECTED_FEATURE_NAME_SHA256
    ):
        raise ValueError("NEXT207 frozen feature universe differs")

    rows: list[dict[str, object]] = []
    for feature in feature_names:
        values = pd.to_numeric(combined[feature], errors="coerce").to_numpy(float)
        source_results = {}
        for direction in PROTECTION_DIRECTIONS:
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
            row = {
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
            rows.append(row)
    raw_table = pd.DataFrame(rows)
    if len(raw_table) != EXPECTED_HYPOTHESIS_COUNT:
        raise RuntimeError("NEXT207 hypothesis count differs")
    table, selected = select_residual_hypothesis(raw_table)
    eligible = table.loc[table["eligible_for_search"]]
    eligible_names_out = sorted(eligible["hypothesis"].astype(str).tolist())

    catalogue = {
        "protocol": PROTOCOL,
        "design_sha256": input_hashes["design"],
        "residual_candidate_key_sha256": EXPECTED_CANDIDATE_KEY_SHA256,
        "residual_threshold": threshold,
        "feature_selection_policy": {
            "blocked_exact_names": sorted(BLOCKED_EXACT_NAMES),
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
        "audit_mode": "fixed_next206_rejected_extreme_cohort_x0_feature_audit",
        "residual_candidate_key_sha256": EXPECTED_CANDIDATE_KEY_SHA256,
        "residual_threshold": threshold,
        "cohort_counts": {
            key: {"protected_rejected": value[0], "severe_rejected": value[1]}
            for key, value in sorted(cohort_counts.items())
        },
        "feature_count": len(feature_names),
        "hypothesis_count": len(table),
        "raw_gate_passing_count": int(table["passes_raw_gates"].sum()),
        "eligible_hypothesis_count": int(len(eligible)),
        "eligible_hypotheses": eligible_names_out,
        "selected_hypothesis": selected,
        "next208_search_authorized": bool(selected is not None),
        "new_formula_searched": False,
        "new_formula_selected": False,
        "validation_or_replication_opened": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    source_paths = {
        "src/next205_motif_exception_depth_search.py": Path(
            n205.__file__
        ).resolve(),
        "src/next206_motif_exception_depth_broad_residual.py": Path(
            n206.__file__
        ).resolve(),
        "src/next207_residual_x0_feature_audit.py": Path(__file__).resolve(),
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
            "eligible_hypothesis_count": int(len(eligible)),
            "next206_residual_candidate_reproduced": True,
            "next208_search_authorized": bool(selected is not None),
            "existing_x0_feature_branch_terminated": selected is None,
            "new_formula_searched": False,
            "new_formula_selected": False,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True,
            "opened_validation_outputs_used": False,
            "scigen_replication_endpoint_opened": False,
            "wyformer_replication_endpoint_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "model_or_proxy_potential_used": False,
            "physical_relaxation_executed": False,
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
            raise RuntimeError("NEXT207 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT207 source changed before publication")
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
    stages = (
        98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124, 125,
        129, 130, 133, 134, 163, 164, 168, 173, 179, 180, 181, 182,
        183, 184, 185, 186, 188, 190, 192,
    )
    for stage in stages:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    for stage in (194, 199, 200, 201, 202, 203, 204, 205, 206):
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--next202-design-path", type=Path, required=True)
    parser.add_argument("--next205-design-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_residual_x0_feature_audit(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{f"next{stage}_dir": getattr(args, f"next{stage}_dir") for stage in stages},
        **{
            f"next{stage}_dir": getattr(args, f"next{stage}_dir")
            for stage in (194, 199, 200, 201, 202, 203, 204, 205, 206)
        },
        next135_freeze_path=args.next135_freeze_path,
        next202_design_path=args.next202_design_path,
        next205_design_path=args.next205_design_path,
        design_path=args.design_path,
        output_dir=args.output_dir,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "audit_one_source",
    "blocked_feature_name",
    "directional_risk",
    "run_residual_x0_feature_audit",
    "select_auditable_features",
    "select_residual_hypothesis",
]


if __name__ == "__main__":
    raise SystemExit(main())
