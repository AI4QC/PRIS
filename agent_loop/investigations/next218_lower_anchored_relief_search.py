#!/usr/bin/env python3
"""Search frozen lower-anchored x0 protection relief inside the NEXT214 band."""

from __future__ import annotations

import argparse
import hashlib
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

import src.next217_repair_band_broad_diagnostic as n217
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


n216 = n217.n216
n215 = n216.n215
PROTOCOL = "2026-08-08-next218-lower-anchored-relief-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT218_LOWER_ANCHORED_RELIEF_CATALOGUE.json"
EVALUATION_NAME = "NEXT218_DISCOVERY_EVALUATION.json"
FORMULA_NAME = "NEXT218_FROZEN_CANDIDATE.json"
SEARCH_NAME = "next218_lower_anchored_relief_search.parquet"
EXPECTED_DESIGN_SHA256 = (
    "298c17b9553a9fe3ce703d1e38b9a0abcb6b028deb2f43e11908b04a7af29673"
)
EXPECTED_NEXT217_SOURCE_SHA256 = (
    "fb7cb23dc50becb0ce9756472a411a1b045d6b7f39b3e670a3da74b3d891bd2c"
)
EXPECTED_ELIGIBLE_COUNT = n216.EXPECTED_ELIGIBLE_COUNT
EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256 = n216.EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256
AMPLITUDE_FRACTIONS = n216.AMPLITUDE_FRACTIONS
EXPECTED_CANDIDATE_COUNT = 1 + EXPECTED_ELIGIBLE_COUNT * len(AMPLITUDE_FRACTIONS)
SEARCH_WORKERS = n216.SEARCH_WORKERS
SCORE_COMPOSITION = (
    "next214_score_if_outside_lower_inclusive_upper_exclusive_repair_band_"
    "or_missing_else_lower_plus_next214_score_minus_lower_times_one_minus_"
    "amplitude_times_bounded_protection_certificate"
)
BOUNDARY_FLAGS = n216.BOUNDARY_FLAGS
EARLY_STAGES = (
    98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124, 125,
    129, 130, 133, 134, 163, 164, 168, 173, 179, 180, 181, 182,
    183, 184, 185, 186, 188, 190, 192,
)
LATER_STAGES = (
    194, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
    210, 211, 212, 213, 214, 215, 216, 217,
)
REQUIRED_STAGES = EARLY_STAGES + LATER_STAGES
REQUIRED_DESIGN_STAGES = (
    202, 205, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217,
)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n217.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next217_design": n217.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next217_manifest": (
        "3478bd3d348d83f9cf952b8e357c2e7f9b86d3208c3b2fcbaefcf79db0df1d62"
    ),
    "next217_diagnostic": (
        "ae4e29bcc407b6e9b0a94c8bc5daa88f348f3cab1ded3416bfdb114525247a78"
    ),
    "next217_table": (
        "a2a4a8723a0372a7442cd7ad141e5f9f0d9c5250c7d18fe5202c19651b8aa061"
    ),
}


def lower_anchored_relief_score(
    *,
    base_score: object,
    base_support: object,
    feature_values: object,
    direction: str | None,
    q_lo: float | None,
    q_hi: float | None,
    lower: float,
    upper: float,
    amplitude_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contract only the excess above the repair band's lower boundary."""

    score = np.asarray(base_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    values = np.asarray(feature_values, dtype=float)
    low = float(lower)
    high = float(upper)
    amplitude = float(amplitude_fraction)
    if (
        score.ndim != 1
        or support.shape != score.shape
        or values.shape != score.shape
        or not math.isfinite(low)
        or not math.isfinite(high)
        or not high > low
        or not math.isfinite(amplitude)
        or not 0.0 <= amplitude <= 1.0
        or np.any(~np.isfinite(score[support]))
        or np.any(score[support] < -1.0e-12)
    ):
        raise ValueError("NEXT218 score arrays or relief parameters differ")
    if direction is None and q_lo is None and q_hi is None and amplitude == 0.0:
        return score.copy(), support.copy(), np.zeros(score.shape, dtype=bool)
    if (
        direction not in n215.PROTECTION_DIRECTIONS
        or q_lo is None
        or q_hi is None
        or amplitude <= 0.0
    ):
        raise ValueError("NEXT218 relief specification differs")
    protection = n216.bounded_directional_protection(
        values, direction, float(q_lo), float(q_hi)
    )
    active = (
        support
        & (score >= low)
        & (score < high)
        & np.isfinite(protection)
    )
    corrected = score.copy()
    corrected[active] = low + (score[active] - low) * (
        1.0 - amplitude * protection[active]
    )
    if (
        np.any(corrected[active] < low - 1.0e-12)
        or np.any(corrected[active] > score[active] + 1.0e-12)
        or np.any(corrected[active] >= high + 1.0e-12)
    ):
        raise RuntimeError("NEXT218 relief crossed a frozen repair boundary")
    return corrected, support.copy(), active


def build_anchored_candidate_specs(
    *,
    base_candidate_key: str,
    eligible_hypotheses: Sequence[str],
    features: pd.DataFrame,
    base_score: object,
    base_support: object,
    lower: float = n215.REPAIR_LOWER_THRESHOLD,
    upper: float = n215.REPAIR_UPPER_THRESHOLD,
    amplitude_fractions: Sequence[float] = AMPLITUDE_FRACTIONS,
) -> list[dict[str, object]]:
    """Build the exact NEXT216 grid with lower-anchored candidate identities."""

    prior = n216.build_candidate_specs(
        base_candidate_key=base_candidate_key,
        eligible_hypotheses=eligible_hypotheses,
        features=features,
        base_score=base_score,
        base_support=base_support,
        lower=lower,
        upper=upper,
        amplitude_fractions=amplitude_fractions,
    )
    specs: list[dict[str, object]] = []
    for raw in prior:
        payload = {key: value for key, value in raw.items() if key != "candidate_key"}
        payload["score_composition"] = SCORE_COMPOSITION
        specs.append(
            {
                **payload,
                "candidate_key": json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                ),
            }
        )
    if len({str(spec["candidate_key"]) for spec in specs}) != len(specs):
        raise RuntimeError("NEXT218 candidate keys are not unique")
    return specs


def materialize_anchored_candidates(
    *,
    features: pd.DataFrame,
    base_score: object,
    base_support: object,
    specs: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    """Encode every exact anchored score as one evaluator term."""

    score = np.asarray(base_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    raw_specs = [dict(value) for value in specs]
    if (
        not isinstance(features, pd.DataFrame)
        or score.shape != (len(features),)
        or support.shape != score.shape
        or not raw_specs
        or len({str(spec.get("candidate_key", "")) for spec in raw_specs})
        != len(raw_specs)
    ):
        raise ValueError("NEXT218 materializer inputs differ")
    columns: dict[str, np.ndarray] = {}
    terms: list[dict[str, object]] = []
    runtime: list[dict[str, object]] = []
    for spec in raw_specs:
        feature = spec.get("feature")
        if feature is None:
            values = np.full(len(features), np.nan)
        elif str(feature) in features.columns:
            values = pd.to_numeric(
                features[str(feature)], errors="coerce"
            ).to_numpy(float)
        else:
            raise ValueError("NEXT218 materializer feature differs")
        corrected, got_support, _ = lower_anchored_relief_score(
            base_score=score,
            base_support=support,
            feature_values=values,
            direction=spec.get("direction"),
            q_lo=spec.get("q_lo"),
            q_hi=spec.get("q_hi"),
            lower=float(spec["lower_threshold"]),
            upper=float(spec["upper_threshold"]),
            amplitude_fraction=float(spec["amplitude_fraction"]),
        )
        maximum = float(np.max(corrected[got_support])) if got_support.any() else 0.0
        divisor = max(1.0, maximum / 50.0)
        encoded = np.full(len(features), np.nan)
        encoded[got_support] = np.sinh(corrected[got_support] / divisor)
        key = str(spec["candidate_key"])
        term_id = (
            "next218_virtual_candidate__"
            f"{hashlib.sha256(key.encode()).hexdigest()[:24]}"
        )
        column = f"_{term_id}_value"
        columns[column] = encoded
        terms.append(
            {
                "term_id": term_id,
                "feature": column,
                "direction": 1,
                "transform": "asinh",
                "center": 0.0,
                "scale": 1.0 / divisor,
                "group": "next218_lower_anchored_relief",
                "encoding": "asinh_sinh_exact_lower_anchored_relief_score",
                "physical_candidate_key": key,
            }
        )
        runtime.append(
            {
                "candidate_key": key,
                "base_term_ids": [term_id],
                "base_weights": [1.0],
                "optional_term_id": None,
                "optional_weight": 0.0,
            }
        )
    return (
        pd.concat([features.reset_index(drop=True), pd.DataFrame(columns)], axis=1),
        terms,
        runtime,
    )


def _paths(
    *,
    roots: Mapping[str, Path],
    next135_freeze_path: Path,
    design_paths: Mapping[int, Path],
    design_path: Path,
) -> dict[str, Path]:
    paths = n217._paths(
        roots,
        next135_freeze_path,
        design_paths[202],
        design_paths[205],
        design_paths[207],
        design_paths[208],
        design_paths[209],
        design_paths[210],
        design_paths[211],
        design_paths[212],
        design_paths[213],
        design_paths[214],
        design_paths[215],
        design_paths[216],
        design_paths[217],
    )
    paths["next217_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next217_manifest": roots["next217"] / n217.MANIFEST_NAME,
            "next217_diagnostic": roots["next217"] / n217.DIAGNOSTIC_NAME,
            "next217_table": roots["next217"] / n217.TABLE_NAME,
        }
    )
    return paths


def _verify_next217(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> tuple[tuple[str, ...], tuple[str, ...], str, str, dict[str, object]]:
    """Verify the closed NEXT217 branch and return reconstruction identities."""

    prior_paths = dict(paths)
    prior_paths["design"] = paths["next217_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next217_design"]
    (
        eligible,
        eligible214,
        primary_key,
        start_key,
        formula214,
        _,
        _,
    ) = n217._verify_next216(prior_paths, prior_hashes)
    manifest = json.loads(paths["next217_manifest"].read_text())
    diagnostic = json.loads(paths["next217_diagnostic"].read_text())
    table = pd.read_parquet(paths["next217_table"])
    expected_outputs = {
        n217.DIAGNOSTIC_NAME: input_hashes["next217_diagnostic"],
        n217.TABLE_NAME: input_hashes["next217_table"],
    }
    closest = diagnostic.get("global_closest", {})
    if (
        manifest.get("protocol") != n217.PROTOCOL
        or manifest.get("candidate_count") != n217.EXPECTED_CANDIDATE_COUNT
        or manifest.get("candidate_key_sha256")
        != n217.EXPECTED_CANDIDATE_KEY_SHA256
        or manifest.get("next216_all_gate_candidate_count") != 0
        or manifest.get("new_formula_searched") is not False
        or manifest.get("new_formula_selected") is not False
        or manifest.get("repair_band_relief_branch_closed") is not True
        or manifest.get("outputs_sha256") != expected_outputs
        or any(manifest.get(key) is not value for key, value in BOUNDARY_FLAGS.items())
        or manifest.get("executed_source_sha256", {}).get(
            "src/next217_repair_band_broad_diagnostic.py"
        )
        != EXPECTED_NEXT217_SOURCE_SHA256
        or _sha256_file(Path(n217.__file__).resolve())
        != EXPECTED_NEXT217_SOURCE_SHA256
        or diagnostic.get("protocol") != n217.PROTOCOL
        or diagnostic.get("candidate_count") != n217.EXPECTED_CANDIDATE_COUNT
        or diagnostic.get("candidate_key_sha256")
        != n217.EXPECTED_CANDIDATE_KEY_SHA256
        or diagnostic.get("improves_over_next214_global_residual") is not False
        or diagnostic.get("repair_band_relief_branch_closed") is not True
        or closest.get("hypothesis") is not None
        or closest.get("failed_constraint_count") != n217.EXPECTED_BASE_FAILED_COUNT
        or not math.isclose(
            float(closest.get("normalized_shortfall_sum", math.nan)),
            n217.EXPECTED_BASE_SHORTFALL,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or len(table) != n217.EXPECTED_CANDIDATE_COUNT
    ):
        raise ValueError("NEXT218 NEXT217 provenance differs")
    return eligible, eligible214, primary_key, start_key, formula214


def run_lower_anchored_relief_search(
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
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only lower-anchored relief search."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs):
        raise ValueError("NEXT218 stage directory universe differs")
    if not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT218 design path universe differs")
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
    if type(search_workers) is not int or search_workers <= 0:
        raise ValueError("NEXT218 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT218 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT218 formal input identity differs: {differing}")
    eligible, eligible214, primary_key, start_key, formula214 = _verify_next217(
        paths, input_hashes
    )
    combined, feature_tables, base_score, base_support, endpoint = (
        n215._reconstruct_next214_final(
            paths=paths,
            eligible=eligible214,
            primary_key=primary_key,
            start_key=start_key,
            formula=formula214,
        )
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
        raise ValueError("NEXT218 NEXT214 base identity differs")
    base_candidate_key = str(accepted.iloc[0]["candidate_key"])
    if (
        hashlib.sha256(base_candidate_key.encode()).hexdigest()
        != n215.EXPECTED_NEXT214_FINAL_PATH_KEY_SHA256
    ):
        raise ValueError("NEXT218 NEXT214 base key differs")
    specs = build_anchored_candidate_specs(
        base_candidate_key=base_candidate_key,
        eligible_hypotheses=eligible,
        features=combined,
        base_score=base_score,
        base_support=base_support,
    )
    if len(specs) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("NEXT218 candidate universe differs")
    combined_virtual, terms, runtime = materialize_anchored_candidates(
        features=combined,
        base_score=base_score,
        base_support=base_support,
        specs=specs,
    )
    started = time.perf_counter()
    result = (
        n215.n214.n212.n210.n208.n205.n203.n202.n200.n194.n130.n125
        .search_optional_guard_laws_parallel(
            features=combined_virtual,
            endpoint=endpoint,
            old_terms=terms,
            optional_terms=[],
            candidate_specs=runtime,
            workers=search_workers,
        )
    )
    elapsed = time.perf_counter() - started
    if int(result["candidate_count"]) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("NEXT218 evaluator count differs")

    spec_by_key = {str(spec["candidate_key"]): spec for spec in specs}
    source = combined["source_dataset"].astype(str).to_numpy()

    def decorate(record: dict[str, object]) -> None:
        spec = spec_by_key[str(record["candidate_key"])]
        feature = spec["feature"]
        values = (
            np.full(len(combined), np.nan)
            if feature is None
            else pd.to_numeric(combined[str(feature)], errors="coerce").to_numpy(float)
        )
        corrected, support, active = lower_anchored_relief_score(
            base_score=base_score,
            base_support=base_support,
            feature_values=values,
            direction=spec["direction"],
            q_lo=spec["q_lo"],
            q_hi=spec["q_hi"],
            lower=float(spec["lower_threshold"]),
            upper=float(spec["upper_threshold"]),
            amplitude_fraction=float(spec["amplitude_fraction"]),
        )
        unchanged_region = support & (
            (base_score < n215.REPAIR_LOWER_THRESHOLD)
            | (base_score >= n215.REPAIR_UPPER_THRESHOLD)
            | ~np.isfinite(values)
        )
        if (
            not np.array_equal(corrected[unchanged_region], base_score[unchanged_region])
            or np.any(corrected[active] < n215.REPAIR_LOWER_THRESHOLD - 1.0e-12)
        ):
            raise RuntimeError("NEXT218 changed a frozen score region")
        record.update(
            {
                "amplitude_denominator": spec["amplitude_denominator"],
                "amplitude_fraction": spec["amplitude_fraction"],
                "amplitude_numerator": spec["amplitude_numerator"],
                "base_candidate_key_sha256": (
                    n215.EXPECTED_NEXT214_FINAL_PATH_KEY_SHA256
                ),
                "direction": spec["direction"],
                "feature": feature,
                "hypothesis": spec["hypothesis"],
                "anchored_active_rows": int(active.sum()),
                "anchored_active_scigen": int((active & (source == "scigen")).sum()),
                "anchored_active_wyformer": int(
                    (active & (source == "wyformer")).sum()
                ),
                "missing_policy": spec["missing_policy"],
                "q_hi": spec["q_hi"],
                "q_lo": spec["q_lo"],
                "quantile_method": spec["quantile_method"],
                "repair_lower_threshold": spec["lower_threshold"],
                "repair_upper_threshold": spec["upper_threshold"],
                "score_composition": SCORE_COMPOSITION,
            }
        )

    for record in result["candidate_records"]:
        decorate(record)
    selected = result["selected"]
    if "feature" not in selected["record"]:
        decorate(selected["record"])
    records = pd.DataFrame(result["candidate_records"])
    auc_safe_non_broad = (
        records["passes_source_auc_gates"].fillna(False).astype(bool)
        & records["passes_safe_all_cells"].fillna(False).astype(bool)
        & ~records["passes_broad_all_cells"].fillna(False).astype(bool)
    )
    next219_keys = sorted(
        records.loc[auc_safe_non_broad, "candidate_key"].astype(str)
    )
    next219_sha = hashlib.sha256("\n".join(next219_keys).encode()).hexdigest()
    selected_spec = spec_by_key[str(selected["record"]["candidate_key"])]
    formula = {
        "protocol": PROTOCOL,
        "kind": "lower_anchored_repair_band_raw_x0_relief_no_dft_score",
        "base_candidate_key_sha256": n215.EXPECTED_NEXT214_FINAL_PATH_KEY_SHA256,
        "feature": selected_spec["feature"],
        "direction": selected_spec["direction"],
        "q_lo": selected_spec["q_lo"],
        "q_hi": selected_spec["q_hi"],
        "quantile_method": "inverted_cdf",
        "amplitude_fraction": selected_spec["amplitude_fraction"],
        "repair_lower_threshold": n215.REPAIR_LOWER_THRESHOLD,
        "repair_upper_threshold": n215.REPAIR_UPPER_THRESHOLD,
        "missing_policy": "TERM_OFF_KEEP_NEXT214_SCORE",
        "support_policy": "UNCHANGED_FROM_NEXT214",
        "score_composition": SCORE_COMPOSITION,
        "dft_values_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
    }
    passes = bool(selected["record"]["passes_all_discovery_gates"])
    counts = {
        name: int(records[name].fillna(False).astype(bool).sum())
        for name in (
            "passes_source_auc_gates",
            "passes_safe_all_cells",
            "passes_broad_all_cells",
            "passes_all_discovery_gates",
        )
    }
    counts["passes_auc_and_safe_but_not_broad"] = int(auc_safe_non_broad.sum())
    catalogue = {
        "protocol": PROTOCOL,
        "design_sha256": input_hashes["design"],
        "base_candidate_key_sha256": n215.EXPECTED_NEXT214_FINAL_PATH_KEY_SHA256,
        "eligible_hypotheses": list(eligible),
        "eligible_hypothesis_count": len(eligible),
        "eligible_hypothesis_sha256": EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256,
        "normalization_fit_uses_endpoint": False,
        "normalization_population": "all_rows_in_fixed_repair_band",
        "normalization_quantiles": [1 / 16, 15 / 16],
        "quantile_method": "inverted_cdf",
        "amplitude_fractions": list(AMPLITUDE_FRACTIONS),
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "repair_lower_threshold": n215.REPAIR_LOWER_THRESHOLD,
        "repair_upper_threshold": n215.REPAIR_UPPER_THRESHOLD,
        "score_composition": SCORE_COMPOSITION,
        "base_support_unchanged": True,
        "active_scores_cannot_cross_lower_boundary": True,
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    evaluation = {
        "protocol": PROTOCOL,
        "evaluation_mode": "fixed_lower_anchored_repair_band_x0_relief_search",
        "next214_final_candidate_reproduced": True,
        "next215_eligible_hypotheses_reproduced": True,
        "next217_closed_branch_reproduced": True,
        "rows": {
            "scigen": int(len(feature_tables["scigen"])),
            "wyformer": int(len(feature_tables["wyformer"])),
            "total": int(len(combined)),
        },
        "candidate_count": int(result["candidate_count"]),
        "elapsed_seconds": elapsed,
        "search_workers": search_workers,
        "counts": counts,
        "selected_record": selected["record"],
        "selected_formula": formula,
        "selected_safe": selected["safe"],
        "selected_safe_diagnostic": selected["safe_diagnostic"],
        "selected_broad": selected["broad"],
        "selected_source_diagnostics": selected["source_diagnostics"],
        "pauling_by_cell": result["pauling_by_cell"],
        "cells": result["cells"],
        "passes_all_cross_source_discovery_gates": passes,
        "freeze_authorized": passes,
        "next219_diagnostic_authorized": bool(not passes and next219_keys),
        "next219_candidate_count": len(next219_keys),
        "next219_candidate_key_sha256": next219_sha,
        "requires_unopened_internal_validation_before_claim": True,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    source_paths = {
        "src/next216_repair_band_relief_search.py": Path(n216.__file__).resolve(),
        "src/next217_repair_band_broad_diagnostic.py": Path(n217.__file__).resolve(),
        "src/next218_lower_anchored_relief_search.py": Path(__file__).resolve(),
    }
    source_hashes = {
        name: _sha256_file(path) for name, path in source_paths.items()
    }
    try:
        catalogue_path = staging / CATALOGUE_NAME
        evaluation_path = staging / EVALUATION_NAME
        formula_path = staging / FORMULA_NAME
        search_path = staging / SEARCH_NAME
        _write_json(catalogue_path, catalogue)
        _write_json(evaluation_path, evaluation)
        _write_json(formula_path, formula)
        records.to_parquet(search_path, index=False)
        outputs = [catalogue_path, evaluation_path, formula_path, search_path]
        manifest = {
            "protocol": PROTOCOL,
            "candidate_count": int(result["candidate_count"]),
            "eligible_hypothesis_count": len(eligible),
            "eligible_hypothesis_sha256": EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256,
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "next219_diagnostic_authorized": bool(not passes and next219_keys),
            "next219_candidate_count": len(next219_keys),
            "next219_candidate_key_sha256": next219_sha,
            "requires_unopened_internal_validation_before_claim": True,
            "lower_anchored_relief_branch_terminated": bool(
                not passes and not next219_keys
            ),
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True,
            **BOUNDARY_FLAGS,
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
            raise RuntimeError("NEXT218 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT218 source changed before publication")
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
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_lower_anchored_relief_search(
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
        search_workers=args.search_workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "AMPLITUDE_FRACTIONS",
    "EXPECTED_CANDIDATE_COUNT",
    "SCORE_COMPOSITION",
    "build_anchored_candidate_specs",
    "lower_anchored_relief_score",
    "materialize_anchored_candidates",
    "run_lower_anchored_relief_search",
]


if __name__ == "__main__":
    raise SystemExit(main())
