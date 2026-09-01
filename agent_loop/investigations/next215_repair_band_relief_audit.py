#!/usr/bin/env python3
"""Audit raw x0 protection certificates inside the NEXT214 repair band."""

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

import src.next207_residual_x0_feature_audit as n207
import src.next214_forward_stagewise_risk_lift as n214
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-08-next215-repair-band-relief-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT215_REPAIR_BAND_RELIEF_CATALOGUE.json"
AUDIT_NAME = "NEXT215_REPAIR_BAND_RELIEF_AUDIT.json"
TABLE_NAME = "next215_repair_band_relief_audit.parquet"
EXPECTED_DESIGN_SHA256 = (
    "b8f1465d2dc0b4ee56ccbefcc162cf5d7d60fd64a4330ca140de0eb7caa2c5a5"
)
EXPECTED_NEXT214_SOURCE_SHA256 = (
    "fb1c4b7d2ca0db9af12b6ebf9b88e538f4694ac6f96a83bb01bfc84d9d71f149"
)
EXPECTED_NEXT214_FINAL_PATH_KEY_SHA256 = (
    "bf9c47811a270ba46f0acb5a982f42c45cacf54c8801733333289b03bf99810e"
)
EXPECTED_NEXT214_UNCHANGED_KEY_SHA256 = (
    "3bac3825e0a4ac36caf41e8170f6f266876ab830ea9676610ed4c019db3b461f"
)
EXPECTED_NEXT214_SHORTFALL = 0.26893426117441227
REPAIR_LOWER_THRESHOLD = 0.17470215862148156
REPAIR_UPPER_THRESHOLD = 0.570892727856757
PROTECTION_DIRECTIONS = n207.PROTECTION_DIRECTIONS
EXPECTED_FEATURE_COUNT = n207.EXPECTED_FEATURE_COUNT
EXPECTED_FEATURE_NAME_SHA256 = n207.EXPECTED_FEATURE_NAME_SHA256
EXPECTED_HYPOTHESIS_COUNT = EXPECTED_FEATURE_COUNT * len(PROTECTION_DIRECTIONS)
MINIMUM_COVERAGE = n207.MINIMUM_COVERAGE
MINIMUM_CLASS_COUNT = n207.MINIMUM_CLASS_COUNT
MINIMUM_AGGREGATE_AUC = n207.MINIMUM_AGGREGATE_AUC
MINIMUM_MACRO_AUC = n207.MINIMUM_MACRO_AUC
MINIMUM_WORST_AUC = n207.MINIMUM_WORST_AUC
USED_NEXT214_FEATURES = frozenset(
    {"scbv_mismatch_max", "nm_site_max", "steric_overlap2_vector_q95"}
)
EXPECTED_FINAL_TERMS = (
    {
        "amplitude_fraction": 0.0625,
        "direction": "protected_low",
        "feature": "scbv_mismatch_max",
        "hypothesis": "scbv_mismatch_max__protected_low",
        "q_hi": 1.8461993346107197,
        "q_lo": 0.15502600238558073,
    },
    {
        "amplitude_fraction": 0.0625,
        "direction": "protected_low",
        "feature": "nm_site_max",
        "hypothesis": "nm_site_max__protected_low",
        "q_hi": 0.031296021794211364,
        "q_lo": -0.10243926308528957,
    },
    {
        "amplitude_fraction": 0.0625,
        "direction": "protected_low",
        "feature": "steric_overlap2_vector_q95",
        "hypothesis": "steric_overlap2_vector_q95__protected_low",
        "q_hi": 0.08051816031092843,
        "q_lo": 1.1874805672990879e-08,
    },
)
EXPECTED_REPAIR_COHORT_COUNTS = {
    "scigen:all": (221, 1432),
    "scigen:fold0": (42, 302),
    "scigen:fold1": (49, 269),
    "scigen:fold2": (40, 301),
    "scigen:fold3": (42, 282),
    "scigen:fold4": (48, 278),
    "wyformer:all": (317, 369),
    "wyformer:fold0": (72, 68),
    "wyformer:fold1": (53, 72),
    "wyformer:fold2": (66, 84),
    "wyformer:fold3": (70, 81),
    "wyformer:fold4": (56, 64),
}
EXPECTED_ABOVE_SAFE_COUNTS = {
    "scigen:all": (84, 1674),
    "wyformer:all": (9, 150),
}
BOUNDARY_FLAGS = n214.n212.n210.n208.BOUNDARY_FLAGS
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n214.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next214_design": n214.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next214_manifest": (
        "35e1da8433880ea5dd9436f68b6d18b8fed9e135aaf2bb457c528063cc8eed18"
    ),
    "next214_catalogue": (
        "f11d9549feb54d4acdb4aa740a0fdd76bbb8b2bcfe9b0cc5bbf0b4e4f9015ac6"
    ),
    "next214_evaluation": (
        "cd05d9f7a684eed330603c45ac84b4098363d904b7b66533ba03aa6129a97c29"
    ),
    "next214_formula": (
        "1ba0de2b8f77e393c11849e5a013722a4b453a7f744894fb4d8cd61d24a3e412"
    ),
    "next214_search": (
        "a22f3f350345c94400f673aba4ff308ab0ba579c5be88f6b377aaa1a930b4afc"
    ),
}


def directional_protection(values: object, direction: str) -> np.ndarray:
    """Map a protection direction to a protected-positive score."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or direction not in PROTECTION_DIRECTIONS:
        raise ValueError("NEXT215 protection direction differs")
    return array.copy() if direction == "protected_high" else -array


def ranking_auc_value(value: object) -> float:
    """Map an unavailable AUC to a deterministic noncompetitive rank value."""

    if value is None:
        return float("-inf")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("NEXT215 ranking AUC differs")
    return result


def repair_band_mask(
    *,
    score: object,
    support: object,
    endpoint: object,
    lower: float = REPAIR_LOWER_THRESHOLD,
    upper: float = REPAIR_UPPER_THRESHOLD,
) -> np.ndarray:
    """Return the lower-inclusive, upper-exclusive extreme repair cohort."""

    values = np.asarray(score, dtype=float)
    supported = np.asarray(support, dtype=bool)
    outcome = np.asarray(endpoint, dtype=float)
    low = float(lower)
    high = float(upper)
    if (
        values.ndim != 1
        or supported.shape != values.shape
        or outcome.shape != values.shape
        or not math.isfinite(low)
        or not math.isfinite(high)
        or not high > low
    ):
        raise ValueError("NEXT215 repair-band inputs differ")
    extreme = (outcome <= 1.0) | (outcome >= 2.0)
    return (
        supported
        & np.isfinite(values)
        & (values >= low)
        & (values < high)
        & extreme
    )


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
    """Audit one protected-positive feature within one source and its folds."""

    raw = np.asarray(values, dtype=float)
    outcome = np.asarray(endpoint, dtype=float)
    selected = np.asarray(cohort, dtype=bool)
    fold_values = np.asarray(folds, dtype=int)
    expected = tuple(int(value) for value in expected_folds)
    gates = (
        minimum_coverage,
        minimum_aggregate_auc,
        minimum_macro_auc,
        minimum_worst_auc,
    )
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
            for value in gates
        )
    ):
        raise ValueError("NEXT215 source audit inputs differ")
    protection = directional_protection(raw, direction)
    extreme = (outcome <= 1.0) | (outcome >= 2.0)
    base = selected & extreme
    finite = np.isfinite(protection)
    records: list[dict[str, object]] = []
    fold_aucs: list[float] = []
    for fold in (None, *expected):
        cell = base & ((fold_values == fold) if fold is not None else True)
        usable = cell & finite
        protected_count = int((usable & (outcome <= 1.0)).sum())
        severe_count = int((usable & (outcome >= 2.0)).sum())
        coverage = float(usable.sum() / max(int(cell.sum()), 1))
        auc = (
            n207.n205.n203.n202.n200.n194.n87._roc_auc(
                protection[usable], outcome[usable] <= 1.0
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
        records.append(
            {
                "fold": fold,
                "cohort_rows": int(cell.sum()),
                "supported_rows": int(usable.sum()),
                "coverage": coverage,
                "protected": protected_count,
                "severe": severe_count,
                "auc": auc,
                "passes_count_and_coverage": count_and_coverage_pass,
            }
        )
        if fold is not None and auc is not None:
            fold_aucs.append(float(auc))
    aggregate = records[0]
    macro = float(np.mean(fold_aucs)) if len(fold_aucs) == len(expected) else None
    worst = float(np.min(fold_aucs)) if len(fold_aucs) == len(expected) else None
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


def select_relief_hypotheses(
    records: pd.DataFrame,
    *,
    used_features: frozenset[str] = USED_NEXT214_FEATURES,
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    """Apply raw gates, direction vetoes, the path veto, and frozen ranking."""

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
        or not isinstance(used_features, frozenset)
    ):
        raise ValueError("NEXT215 audit record schema differs")
    table = records.copy()
    raw = table["passes_raw_gates"].fillna(False).astype(bool)
    feature = table["feature"].astype(str)
    opposite = raw.groupby(feature).transform("sum") > 1
    already_used = feature.isin(used_features)
    eligible = raw & ~opposite & ~already_used
    table["opposite_direction_passes"] = opposite
    table["already_in_next214_path"] = already_used
    table["eligible_for_search"] = eligible
    table["ineligibility_reason"] = np.select(
        [already_used, opposite, ~raw],
        [
            "already_in_next214_path",
            "opposite_direction_passed",
            "raw_gates_failed",
        ],
        default=None,
    )
    ranked = table.loc[eligible].sort_values(
        [
            "ranking_min_worst_fold_auc",
            "ranking_min_aggregate_auc",
            "ranking_mean_aggregate_auc",
            "hypothesis",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    selected = None if ranked.empty else ranked.iloc[0].to_dict()
    return table, selected


def _paths(
    roots: Mapping[str, Path],
    freeze_path: Path,
    next202_design_path: Path,
    next205_design_path: Path,
    next207_design_path: Path,
    next208_design_path: Path,
    next209_design_path: Path,
    next210_design_path: Path,
    next211_design_path: Path,
    next212_design_path: Path,
    next213_design_path: Path,
    next214_design_path: Path,
    design_path: Path,
) -> dict[str, Path]:
    paths = n214._paths(
        roots,
        freeze_path,
        next202_design_path,
        next205_design_path,
        next207_design_path,
        next208_design_path,
        next209_design_path,
        next210_design_path,
        next211_design_path,
        next212_design_path,
        next213_design_path,
        next214_design_path,
    )
    paths["next214_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next214_manifest": roots["next214"] / n214.MANIFEST_NAME,
            "next214_catalogue": roots["next214"] / n214.CATALOGUE_NAME,
            "next214_evaluation": roots["next214"] / n214.EVALUATION_NAME,
            "next214_formula": roots["next214"] / n214.FORMULA_NAME,
            "next214_search": roots["next214"] / n214.SEARCH_NAME,
        }
    )
    return paths


def _verify_next214(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> tuple[tuple[str, ...], str, str, dict[str, object]]:
    """Verify NEXT214 and return the frozen reconstruction identities."""

    prior_paths = dict(paths)
    prior_paths["design"] = paths["next214_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next214_design"]
    eligible, primary_key, start_key, _ = n214._verify_next213(
        prior_paths, prior_hashes
    )
    manifest = json.loads(paths["next214_manifest"].read_text())
    catalogue = json.loads(paths["next214_catalogue"].read_text())
    evaluation = json.loads(paths["next214_evaluation"].read_text())
    formula = json.loads(paths["next214_formula"].read_text())
    published = pd.read_parquet(paths["next214_search"])
    expected_outputs = {
        n214.CATALOGUE_NAME: input_hashes["next214_catalogue"],
        n214.EVALUATION_NAME: input_hashes["next214_evaluation"],
        n214.FORMULA_NAME: input_hashes["next214_formula"],
        n214.SEARCH_NAME: input_hashes["next214_search"],
    }
    unchanged = published.loc[
        published["depth"].eq(4) & published["proposed_hypothesis"].isna()
    ]
    accepted = published.loc[
        published["depth"].eq(3)
        & published["proposed_hypothesis"].eq(
            "steric_overlap2_vector_q95__protected_low"
        )
        & published["proposed_amplitude_fraction"].eq(0.0625)
    ]
    stages = evaluation.get("stage_summaries", [])
    if (
        manifest.get("protocol") != n214.PROTOCOL
        or manifest.get("passes_all_cross_source_discovery_gates") is not False
        or manifest.get("freeze_authorized") is not False
        or manifest.get("stop_reason") != "no_strict_residual_improvement"
        or manifest.get("final_term_count") != 3
        or manifest.get("stage_count") != 2
        or manifest.get("total_candidate_evaluations") != 417
        or manifest.get("outputs_sha256") != expected_outputs
        or any(manifest.get(key) is not value for key, value in BOUNDARY_FLAGS.items())
        or manifest.get("executed_source_sha256", {}).get(
            "src/next214_forward_stagewise_risk_lift.py"
        )
        != EXPECTED_NEXT214_SOURCE_SHA256
        or _sha256_file(Path(n214.__file__).resolve())
        != EXPECTED_NEXT214_SOURCE_SHA256
        or catalogue.get("protocol") != n214.PROTOCOL
        or catalogue.get("design_sha256") != input_hashes["next214_design"]
        or catalogue.get("normalization_refit") is not False
        or catalogue.get("base_support_unchanged") is not True
        or evaluation.get("protocol") != n214.PROTOCOL
        or evaluation.get("all_discovery_gates_passed") is not False
        or evaluation.get("freeze_authorized") is not False
        or evaluation.get("stop_reason") != "no_strict_residual_improvement"
        or evaluation.get("final_term_count") != 3
        or not math.isclose(
            float(evaluation.get("final_normalized_shortfall_sum")),
            EXPECTED_NEXT214_SHORTFALL,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or len(stages) != 2
        or not math.isclose(
            float(stages[-1].get("best_threshold")),
            REPAIR_LOWER_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or formula.get("protocol") != n214.PROTOCOL
        or formula.get("terms") != list(EXPECTED_FINAL_TERMS)
        or formula.get("term_count") != 3
        or formula.get("dft_values_used_by_executable_formula") is not False
        or formula.get("learned_energy_force_stress_proxy_used") is not False
        or formula.get("model_or_proxy_potential_used") is not False
        or formula.get("physical_relaxation_executed") is not False
        or len(published) != 417
        or len(unchanged) != 1
        or len(accepted) != 1
    ):
        raise ValueError("NEXT215 NEXT214 provenance differs")
    unchanged_row = unchanged.iloc[0]
    accepted_row = accepted.iloc[0]
    if (
        hashlib.sha256(str(unchanged_row["candidate_key"]).encode()).hexdigest()
        != EXPECTED_NEXT214_UNCHANGED_KEY_SHA256
        or hashlib.sha256(str(accepted_row["candidate_key"]).encode()).hexdigest()
        != EXPECTED_NEXT214_FINAL_PATH_KEY_SHA256
        or not bool(unchanged_row["passes_source_auc_gates"])
        or not bool(unchanged_row["passes_safe_all_cells"])
        or bool(unchanged_row["passes_broad_all_cells"])
        or not math.isclose(
            float(unchanged_row["safe_threshold"]),
            REPAIR_UPPER_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    ):
        raise ValueError("NEXT215 NEXT214 final candidate differs")
    return eligible, primary_key, start_key, formula


def _reconstruct_next214_final(
    *,
    paths: Mapping[str, Path],
    eligible: Sequence[str],
    primary_key: str,
    start_key: str,
    formula: Mapping[str, object],
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Reconstruct the exact published three-term NEXT214 score."""

    combined, feature_tables, base_key, base_score, base_support, endpoint = (
        n214.n212.n210.n208._reconstruct_next206(paths=paths)
    )
    next210_specs = n214.n212.n210.build_candidate_specs(
        base_candidate_key=base_key,
        eligible_hypotheses=eligible,
        features=combined,
        base_score=base_score,
        base_support=base_support,
        residual_threshold=n207.EXPECTED_RESIDUAL_THRESHOLD,
    )
    primary_matches = [
        spec for spec in next210_specs if str(spec["candidate_key"]) == primary_key
    ]
    if len(primary_matches) != 1:
        raise ValueError("NEXT215 primary path identity differs")
    primary = primary_matches[0]
    primary_values = pd.to_numeric(
        combined[str(primary["feature"])], errors="coerce"
    ).to_numpy(float)
    current_score, support, _ = n214.n212.n210.residual_risk_lift_score(
        base_score=base_score,
        base_support=base_support,
        feature_values=primary_values,
        direction=str(primary["direction"]),
        q_lo=float(primary["q_lo"]),
        q_hi=float(primary["q_hi"]),
        residual_threshold=float(primary["residual_threshold"]),
        amplitude_fraction=float(primary["amplitude_fraction"]),
        risk_scale=float(primary["risk_scale"]),
    )
    two_specs = n214.n212.build_candidate_specs(
        anchor_spec=primary, next210_specs=next210_specs
    )
    second_matches = [
        spec for spec in two_specs if str(spec["candidate_key"]) == start_key
    ]
    if len(second_matches) != 1:
        raise ValueError("NEXT215 two-signal path identity differs")
    second = second_matches[0]
    second_values = pd.to_numeric(
        combined[str(second["secondary_feature"])], errors="coerce"
    ).to_numpy(float)
    current_score, support, _ = n214.n212.anchored_two_signal_score(
        anchor_score=current_score,
        activation_score=base_score,
        base_support=support,
        feature_values=second_values,
        direction=str(second["secondary_direction"]),
        q_lo=float(second["secondary_q_lo"]),
        q_hi=float(second["secondary_q_hi"]),
        residual_threshold=float(second["residual_threshold"]),
        amplitude_fraction=float(second["secondary_amplitude_fraction"]),
        risk_scale=float(second["risk_scale"]),
    )
    terms = formula.get("terms")
    if terms != list(EXPECTED_FINAL_TERMS):
        raise ValueError("NEXT215 final formula identity differs")
    for term in list(terms)[2:]:
        values = pd.to_numeric(
            combined[str(term["feature"])], errors="coerce"
        ).to_numpy(float)
        current_score, support, _ = n214.n212.anchored_two_signal_score(
            anchor_score=current_score,
            activation_score=base_score,
            base_support=support,
            feature_values=values,
            direction=str(term["direction"]),
            q_lo=float(term["q_lo"]),
            q_hi=float(term["q_hi"]),
            residual_threshold=float(formula["residual_threshold"]),
            amplitude_fraction=float(term["amplitude_fraction"]),
            risk_scale=float(formula["risk_scale"]),
        )
    if not np.array_equal(support, base_support):
        raise RuntimeError("NEXT215 reconstruction changed base support")
    return combined, feature_tables, current_score, support, endpoint


def _cohort_counts(
    *,
    cohort: np.ndarray,
    source: np.ndarray,
    folds: np.ndarray,
    endpoint: np.ndarray,
    include_folds: bool = True,
) -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}
    fold_values: tuple[int | None, ...] = (
        (None, *range(5)) if include_folds else (None,)
    )
    for source_name in ("scigen", "wyformer"):
        for fold in fold_values:
            mask = cohort & (source == source_name)
            if fold is not None:
                mask &= folds == fold
            cell_id = f"{source_name}:{'all' if fold is None else f'fold{fold}'}"
            counts[cell_id] = (
                int((mask & (endpoint <= 1.0)).sum()),
                int((mask & (endpoint >= 2.0)).sum()),
            )
    return counts


def run_repair_band_relief_audit(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    next98_dir: Path, next110_dir: Path, next111_dir: Path,
    next113_dir: Path, next114_dir: Path, next116_dir: Path,
    next117_dir: Path, next120_dir: Path, next121_dir: Path,
    next122_dir: Path, next124_dir: Path, next125_dir: Path,
    next129_dir: Path, next130_dir: Path, next133_dir: Path,
    next134_dir: Path, next163_dir: Path, next164_dir: Path,
    next168_dir: Path, next173_dir: Path, next179_dir: Path,
    next180_dir: Path, next181_dir: Path, next182_dir: Path,
    next183_dir: Path, next184_dir: Path, next185_dir: Path,
    next186_dir: Path, next188_dir: Path, next190_dir: Path,
    next192_dir: Path, next194_dir: Path, next199_dir: Path,
    next200_dir: Path, next201_dir: Path, next202_dir: Path,
    next203_dir: Path, next204_dir: Path, next205_dir: Path,
    next206_dir: Path, next207_dir: Path, next208_dir: Path,
    next209_dir: Path, next210_dir: Path, next211_dir: Path,
    next212_dir: Path, next213_dir: Path, next214_dir: Path,
    next135_freeze_path: Path, next202_design_path: Path,
    next205_design_path: Path, next207_design_path: Path,
    next208_design_path: Path, next209_design_path: Path,
    next210_design_path: Path, next211_design_path: Path,
    next212_design_path: Path, next213_design_path: Path,
    next214_design_path: Path, design_path: Path, output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only repair-band feature audit."""

    early_stages = (
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
    later_stages = (
        (194, next194_dir), (199, next199_dir), (200, next200_dir),
        (201, next201_dir), (202, next202_dir), (203, next203_dir),
        (204, next204_dir), (205, next205_dir), (206, next206_dir),
        (207, next207_dir), (208, next208_dir), (209, next209_dir),
        (210, next210_dir), (211, next211_dir), (212, next212_dir),
        (213, next213_dir), (214, next214_dir),
    )
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{
            f"next{stage}": Path(value).resolve()
            for stage, value in (*early_stages, *later_stages)
        },
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots,
        Path(next135_freeze_path).resolve(),
        Path(next202_design_path).resolve(),
        Path(next205_design_path).resolve(),
        Path(next207_design_path).resolve(),
        Path(next208_design_path).resolve(),
        Path(next209_design_path).resolve(),
        Path(next210_design_path).resolve(),
        Path(next211_design_path).resolve(),
        Path(next212_design_path).resolve(),
        Path(next213_design_path).resolve(),
        Path(next214_design_path).resolve(),
        Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT215 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT215 formal input identity differs: {differing}")
    eligible214, primary_key, start_key, formula = _verify_next214(
        paths, input_hashes
    )
    combined, feature_tables, score, support, endpoint = (
        _reconstruct_next214_final(
            paths=paths,
            eligible=eligible214,
            primary_key=primary_key,
            start_key=start_key,
            formula=formula,
        )
    )
    source = combined["source_dataset"].astype(str).to_numpy()
    folds = n214.n164.assign_group_folds(
        combined["reduced_formula"].astype(str).to_numpy()
    )
    cohort = repair_band_mask(score=score, support=support, endpoint=endpoint)
    cohort_counts = _cohort_counts(
        cohort=cohort,
        source=source,
        folds=folds,
        endpoint=endpoint,
    )
    if cohort_counts != EXPECTED_REPAIR_COHORT_COUNTS:
        raise ValueError("NEXT215 repair-band cohort accounting differs")
    above_safe = (
        support
        & np.isfinite(score)
        & (score >= REPAIR_UPPER_THRESHOLD)
        & ((endpoint <= 1.0) | (endpoint >= 2.0))
    )
    above_safe_counts = _cohort_counts(
        cohort=above_safe,
        source=source,
        folds=folds,
        endpoint=endpoint,
        include_folds=False,
    )
    if above_safe_counts != EXPECTED_ABOVE_SAFE_COUNTS:
        raise ValueError("NEXT215 above-SAFE cohort accounting differs")

    feature_names = n207.select_auditable_features(combined)
    feature_sha = hashlib.sha256("\n".join(feature_names).encode()).hexdigest()
    if require_formal_inputs and (
        len(feature_names) != EXPECTED_FEATURE_COUNT
        or feature_sha != EXPECTED_FEATURE_NAME_SHA256
    ):
        raise ValueError("NEXT215 frozen feature universe differs")

    rows: list[dict[str, object]] = []
    for feature in feature_names:
        values = pd.to_numeric(combined[feature], errors="coerce").to_numpy(float)
        for direction in PROTECTION_DIRECTIONS:
            source_results: dict[str, dict[str, object]] = {}
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
            aggregate_aucs = (
                ranking_auc_value(scigen["aggregate_auc"]),
                ranking_auc_value(wyformer["aggregate_auc"]),
            )
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
                            ranking_auc_value(scigen["worst_fold_auc"]),
                            ranking_auc_value(wyformer["worst_fold_auc"]),
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
        raise RuntimeError("NEXT215 hypothesis count differs")
    table, selected = select_relief_hypotheses(raw_table)
    table = table.sort_values("hypothesis", kind="mergesort").reset_index(drop=True)
    eligible = table.loc[table["eligible_for_search"]]
    eligible_names = sorted(eligible["hypothesis"].astype(str).tolist())
    eligible_sha = hashlib.sha256("\n".join(eligible_names).encode()).hexdigest()

    catalogue = {
        "protocol": PROTOCOL,
        "design_sha256": input_hashes["design"],
        "next214_final_path_key_sha256": EXPECTED_NEXT214_FINAL_PATH_KEY_SHA256,
        "repair_lower_threshold": REPAIR_LOWER_THRESHOLD,
        "repair_upper_threshold": REPAIR_UPPER_THRESHOLD,
        "repair_interval": "lower_inclusive_upper_exclusive",
        "feature_selection_policy": {
            "reused_exactly_from": n207.PROTOCOL,
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
        "used_next214_features_vetoed": sorted(USED_NEXT214_FEATURES),
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
        "audit_mode": "fixed_next214_repair_band_x0_protection_feature_audit",
        "next214_final_path_key_sha256": EXPECTED_NEXT214_FINAL_PATH_KEY_SHA256,
        "repair_lower_threshold": REPAIR_LOWER_THRESHOLD,
        "repair_upper_threshold": REPAIR_UPPER_THRESHOLD,
        "cohort_counts": {
            key: {"protected": value[0], "severe": value[1]}
            for key, value in sorted(cohort_counts.items())
        },
        "above_safe_counts": {
            key: {"protected": value[0], "severe": value[1]}
            for key, value in sorted(above_safe_counts.items())
        },
        "rows": {
            "scigen": int(len(feature_tables["scigen"])),
            "wyformer": int(len(feature_tables["wyformer"])),
            "total": int(len(combined)),
        },
        "feature_count": len(feature_names),
        "hypothesis_count": len(table),
        "raw_gate_passing_count": int(table["passes_raw_gates"].sum()),
        "eligible_hypothesis_count": int(len(eligible)),
        "eligible_hypothesis_sha256": eligible_sha,
        "eligible_hypotheses": eligible_names,
        "selected_hypothesis": selected,
        "next216_search_authorized": bool(eligible_names),
        "predeclared_stop": None if eligible_names else "no_eligible_hypothesis",
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
        "src/next214_forward_stagewise_risk_lift.py": Path(n214.__file__).resolve(),
        "src/next215_repair_band_relief_audit.py": Path(__file__).resolve(),
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
            "raw_gate_passing_count": int(table["passes_raw_gates"].sum()),
            "eligible_hypothesis_count": int(len(eligible)),
            "eligible_hypothesis_sha256": eligible_sha,
            "next214_final_candidate_reproduced": True,
            "repair_band_counts_reproduced": True,
            "above_safe_counts_reproduced": True,
            "next216_search_authorized": bool(eligible_names),
            "repair_band_branch_terminated": not bool(eligible_names),
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
            raise RuntimeError("NEXT215 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT215 source changed before publication")
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
    early_stages = (
        98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124, 125,
        129, 130, 133, 134, 163, 164, 168, 173, 179, 180, 181, 182,
        183, 184, 185, 186, 188, 190, 192,
    )
    later_stages = (
        194, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
        210, 211, 212, 213, 214,
    )
    for stage in early_stages + later_stages:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    for stage in (202, 205, 207, 208, 209, 210, 211, 212, 213, 214):
        parser.add_argument(f"--next{stage}-design-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_repair_band_relief_audit(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{
            f"next{stage}_dir": getattr(args, f"next{stage}_dir")
            for stage in early_stages + later_stages
        },
        next135_freeze_path=args.next135_freeze_path,
        **{
            f"next{stage}_design_path": getattr(args, f"next{stage}_design_path")
            for stage in (202, 205, 207, 208, 209, 210, 211, 212, 213, 214)
        },
        design_path=args.design_path,
        output_dir=args.output_dir,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "REPAIR_LOWER_THRESHOLD",
    "REPAIR_UPPER_THRESHOLD",
    "USED_NEXT214_FEATURES",
    "audit_one_source",
    "directional_protection",
    "ranking_auc_value",
    "repair_band_mask",
    "run_repair_band_relief_audit",
    "select_relief_hypotheses",
]


if __name__ == "__main__":
    raise SystemExit(main())
