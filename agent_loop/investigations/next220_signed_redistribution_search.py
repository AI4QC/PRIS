#!/usr/bin/env python3
"""Search frozen signed x0 protection redistribution inside the repair band."""

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

import src.next219_lower_anchored_broad_diagnostic as n219
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


n218 = n219.n218
n216 = n218.n216
n215 = n218.n215
PROTOCOL = "2026-08-08-next220-signed-redistribution-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT220_SIGNED_REDISTRIBUTION_CATALOGUE.json"
EVALUATION_NAME = "NEXT220_DISCOVERY_EVALUATION.json"
FORMULA_NAME = "NEXT220_FROZEN_CANDIDATE.json"
SEARCH_NAME = "next220_signed_redistribution_search.parquet"
EXPECTED_DESIGN_SHA256 = (
    "45c6008000ffbb24a2b6fff494a447c6409abe89b9899d6a46ee92bdf23a8961"
)
EXPECTED_NEXT219_SOURCE_SHA256 = (
    "69a3d0b4cd0b9e2eb284b491b36e7a87cc852f6141480e251a75faf248fccba8"
)
EXPECTED_ELIGIBLE_COUNT = n218.EXPECTED_ELIGIBLE_COUNT
EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256 = n218.EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256
BETA_DENOMINATOR = 64
BETA_FRACTIONS = (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4)
EXPECTED_CANDIDATE_COUNT = 1 + EXPECTED_ELIGIBLE_COUNT * len(BETA_FRACTIONS)
SEARCH_WORKERS = n218.SEARCH_WORKERS
SCORE_COMPOSITION = (
    "next214_score_if_outside_lower_inclusive_upper_exclusive_repair_band_"
    "or_missing_else_next214_score_plus_beta_times_band_width_times_one_"
    "minus_two_times_bounded_protection_certificate"
)
BOUNDARY_FLAGS = n218.BOUNDARY_FLAGS
REQUIRED_STAGES = n219.REQUIRED_STAGES + (219,)
REQUIRED_DESIGN_STAGES = n219.REQUIRED_DESIGN_STAGES + (219,)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n219.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next219_design": n219.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next219_manifest": (
        "29b2dec027e7686decb06dd03d72b2ed96a4461b7c99029b7a3e08337614ad32"
    ),
    "next219_diagnostic": (
        "1327c39f97ab282361258f43582e8a3d054ac2665e2c1e725e513ba4a5b6d132"
    ),
    "next219_table": (
        "c84ad7ab729843a3a3949dbb3d226801f11bdf79ff1a19a9cc25c3f4bd783e0f"
    ),
}


def signed_redistribution_score(
    *,
    base_score: object,
    base_support: object,
    feature_values: object,
    direction: str | None,
    q_lo: float | None,
    q_hi: float | None,
    lower: float,
    upper: float,
    beta_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Redistribute in-band score symmetrically around half protection."""

    score = np.asarray(base_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    values = np.asarray(feature_values, dtype=float)
    low = float(lower)
    high = float(upper)
    beta = float(beta_fraction)
    if (
        score.ndim != 1
        or support.shape != score.shape
        or values.shape != score.shape
        or not math.isfinite(low)
        or not math.isfinite(high)
        or not high > low
        or not math.isfinite(beta)
        or not 0.0 <= beta <= 0.25
        or np.any(~np.isfinite(score[support]))
        or np.any(score[support] < -1.0e-12)
    ):
        raise ValueError("NEXT220 score arrays or redistribution parameters differ")
    if direction is None and q_lo is None and q_hi is None and beta == 0.0:
        return score.copy(), support.copy(), np.zeros(score.shape, dtype=bool)
    if (
        direction not in n215.PROTECTION_DIRECTIONS
        or q_lo is None
        or q_hi is None
        or beta <= 0.0
    ):
        raise ValueError("NEXT220 redistribution specification differs")
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
    corrected[active] = score[active] + beta * (high - low) * (
        1.0 - 2.0 * protection[active]
    )
    if np.any(corrected[active] < -1.0e-12):
        raise RuntimeError("NEXT220 redistribution produced negative risk")
    return corrected, support.copy(), active


def _beta_numerator(value: float) -> int:
    numerator = int(round(float(value) * BETA_DENOMINATOR))
    if numerator not in {1, 2, 4, 8, 16} or not math.isclose(
        float(value),
        numerator / BETA_DENOMINATOR,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("NEXT220 beta fraction differs")
    return numerator


def build_signed_candidate_specs(
    *,
    base_candidate_key: str,
    eligible_hypotheses: Sequence[str],
    features: pd.DataFrame,
    base_score: object,
    base_support: object,
    lower: float = n215.REPAIR_LOWER_THRESHOLD,
    upper: float = n215.REPAIR_UPPER_THRESHOLD,
    beta_fractions: Sequence[float] = BETA_FRACTIONS,
) -> list[dict[str, object]]:
    """Build the unchanged base plus exact signed redistribution grid."""

    if not isinstance(base_candidate_key, str) or not base_candidate_key:
        raise ValueError("NEXT220 base candidate key must be nonempty")
    if not isinstance(features, pd.DataFrame):
        raise ValueError("NEXT220 feature table differs")
    score = np.asarray(base_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    low = float(lower)
    high = float(upper)
    if (
        score.shape != (len(features),)
        or support.shape != score.shape
        or not math.isfinite(low)
        or not math.isfinite(high)
        or not high > low
        or np.any(~np.isfinite(score[support]))
    ):
        raise ValueError("NEXT220 normalization population differs")
    names = tuple(sorted(str(value) for value in eligible_hypotheses))
    if not names or len(names) != len(set(names)):
        raise ValueError("NEXT220 eligible hypothesis universe differs")
    betas = tuple(float(value) for value in beta_fractions)
    numerators = tuple(_beta_numerator(value) for value in betas)
    if not betas or len(numerators) != len(set(numerators)):
        raise ValueError("NEXT220 beta grid differs")
    fit_mask = support & np.isfinite(score) & (score >= low) & (score < high)
    if not fit_mask.any():
        raise ValueError("NEXT220 endpoint-blind normalization population is empty")
    common = {
        "base_candidate_key": base_candidate_key,
        "beta_denominator": BETA_DENOMINATOR,
        "lower_threshold": low,
        "upper_threshold": high,
        "missing_policy": "TERM_OFF_KEEP_NEXT214_SCORE",
        "quantile_method": "inverted_cdf",
        "score_composition": SCORE_COMPOSITION,
    }
    payloads: list[dict[str, object]] = [
        {
            **common,
            "beta_fraction": 0.0,
            "beta_numerator": 0,
            "direction": None,
            "feature": None,
            "hypothesis": None,
            "q_hi": None,
            "q_lo": None,
        }
    ]
    for hypothesis in names:
        try:
            feature, direction = hypothesis.rsplit("__", 1)
        except ValueError as error:
            raise ValueError("NEXT220 eligible hypothesis identity differs") from error
        if (
            not feature
            or direction not in n215.PROTECTION_DIRECTIONS
            or feature not in features.columns
        ):
            raise ValueError("NEXT220 eligible hypothesis identity differs")
        fit_values = pd.to_numeric(
            features.loc[fit_mask, feature], errors="coerce"
        ).to_numpy(float)
        q_lo, q_hi = n216.robust_protection_cutoffs(fit_values)
        for beta, numerator in zip(betas, numerators, strict=True):
            payloads.append(
                {
                    **common,
                    "beta_fraction": beta,
                    "beta_numerator": numerator,
                    "direction": direction,
                    "feature": feature,
                    "hypothesis": hypothesis,
                    "q_hi": q_hi,
                    "q_lo": q_lo,
                }
            )
    specs = [
        {
            **payload,
            "candidate_key": json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ),
        }
        for payload in payloads
    ]
    if len({str(spec["candidate_key"]) for spec in specs}) != len(specs):
        raise RuntimeError("NEXT220 candidate keys are not unique")
    return specs


def materialize_signed_candidates(
    *,
    features: pd.DataFrame,
    base_score: object,
    base_support: object,
    specs: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    """Encode every exact signed score as one evaluator term."""

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
        raise ValueError("NEXT220 materializer inputs differ")
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
            raise ValueError("NEXT220 materializer feature differs")
        corrected, got_support, _ = signed_redistribution_score(
            base_score=score,
            base_support=support,
            feature_values=values,
            direction=spec.get("direction"),
            q_lo=spec.get("q_lo"),
            q_hi=spec.get("q_hi"),
            lower=float(spec["lower_threshold"]),
            upper=float(spec["upper_threshold"]),
            beta_fraction=float(spec["beta_fraction"]),
        )
        maximum = float(np.max(corrected[got_support])) if got_support.any() else 0.0
        divisor = max(1.0, maximum / 50.0)
        encoded = np.full(len(features), np.nan)
        encoded[got_support] = np.sinh(corrected[got_support] / divisor)
        key = str(spec["candidate_key"])
        term_id = (
            "next220_virtual_candidate__"
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
                "group": "next220_signed_redistribution",
                "encoding": "asinh_sinh_exact_signed_redistribution_score",
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
    paths = n219._paths(
        roots=roots,
        next135_freeze_path=next135_freeze_path,
        design_paths={
            stage: design_paths[stage] for stage in n219.REQUIRED_DESIGN_STAGES
        },
        design_path=design_paths[219],
    )
    paths["next219_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next219_manifest": roots["next219"] / n219.MANIFEST_NAME,
            "next219_diagnostic": roots["next219"] / n219.DIAGNOSTIC_NAME,
            "next219_table": roots["next219"] / n219.TABLE_NAME,
        }
    )
    return paths


def _verify_next219(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> tuple[tuple[str, ...], tuple[str, ...], str, str, dict[str, object]]:
    """Verify the closed NEXT219 branch and return reconstruction identities."""

    prior_paths = dict(paths)
    prior_paths["design"] = paths["next219_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next219_design"]
    (
        eligible,
        eligible214,
        primary_key,
        start_key,
        formula214,
        _,
        _,
    ) = n219._verify_next218(prior_paths, prior_hashes)
    manifest = json.loads(paths["next219_manifest"].read_text())
    diagnostic = json.loads(paths["next219_diagnostic"].read_text())
    table = pd.read_parquet(paths["next219_table"])
    expected_outputs = {
        n219.DIAGNOSTIC_NAME: input_hashes["next219_diagnostic"],
        n219.TABLE_NAME: input_hashes["next219_table"],
    }
    closest = diagnostic.get("global_closest", {})
    if (
        manifest.get("protocol") != n219.PROTOCOL
        or manifest.get("candidate_count") != n219.EXPECTED_CANDIDATE_COUNT
        or manifest.get("candidate_key_sha256")
        != n219.EXPECTED_CANDIDATE_KEY_SHA256
        or manifest.get("strict_residual_improvement_observed") is not False
        or manifest.get("new_formula_searched") is not False
        or manifest.get("new_formula_selected") is not False
        or manifest.get("lower_anchored_relief_branch_closed") is not True
        or manifest.get("outputs_sha256") != expected_outputs
        or any(manifest.get(key) is not value for key, value in BOUNDARY_FLAGS.items())
        or manifest.get("executed_source_sha256", {}).get(
            "src/next219_lower_anchored_broad_diagnostic.py"
        )
        != EXPECTED_NEXT219_SOURCE_SHA256
        or _sha256_file(Path(n219.__file__).resolve())
        != EXPECTED_NEXT219_SOURCE_SHA256
        or diagnostic.get("protocol") != n219.PROTOCOL
        or diagnostic.get("candidate_count") != n219.EXPECTED_CANDIDATE_COUNT
        or diagnostic.get("candidate_key_sha256")
        != n219.EXPECTED_CANDIDATE_KEY_SHA256
        or diagnostic.get("improves_over_next214_global_residual") is not False
        or diagnostic.get("lower_anchored_relief_branch_closed") is not True
        or closest.get("hypothesis") is not None
        or closest.get("failed_constraint_count") != EXPECTED_BASE_FAILED_COUNT
        or not math.isclose(
            float(closest.get("normalized_shortfall_sum", math.nan)),
            EXPECTED_BASE_SHORTFALL,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or len(table) != n219.EXPECTED_CANDIDATE_COUNT
    ):
        raise ValueError("NEXT220 NEXT219 provenance differs")
    return eligible, eligible214, primary_key, start_key, formula214


EXPECTED_BASE_FAILED_COUNT = n219.EXPECTED_BASE_FAILED_COUNT
EXPECTED_BASE_SHORTFALL = n219.EXPECTED_BASE_SHORTFALL


def run_signed_redistribution_search(
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
    """Run the frozen discovery-only signed redistribution search."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs):
        raise ValueError("NEXT220 stage directory universe differs")
    if not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT220 design path universe differs")
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
        raise ValueError("NEXT220 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT220 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT220 formal input identity differs: {differing}")
    eligible, eligible214, primary_key, start_key, formula214 = _verify_next219(
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
        raise ValueError("NEXT220 NEXT214 base identity differs")
    base_candidate_key = str(accepted.iloc[0]["candidate_key"])
    if (
        hashlib.sha256(base_candidate_key.encode()).hexdigest()
        != n215.EXPECTED_NEXT214_FINAL_PATH_KEY_SHA256
    ):
        raise ValueError("NEXT220 NEXT214 base key differs")
    specs = build_signed_candidate_specs(
        base_candidate_key=base_candidate_key,
        eligible_hypotheses=eligible,
        features=combined,
        base_score=base_score,
        base_support=base_support,
    )
    if len(specs) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("NEXT220 candidate universe differs")
    combined_virtual, terms, runtime = materialize_signed_candidates(
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
        raise RuntimeError("NEXT220 evaluator count differs")

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
        corrected, support, active = signed_redistribution_score(
            base_score=base_score,
            base_support=base_support,
            feature_values=values,
            direction=spec["direction"],
            q_lo=spec["q_lo"],
            q_hi=spec["q_hi"],
            lower=float(spec["lower_threshold"]),
            upper=float(spec["upper_threshold"]),
            beta_fraction=float(spec["beta_fraction"]),
        )
        unchanged_region = support & (
            (base_score < n215.REPAIR_LOWER_THRESHOLD)
            | (base_score >= n215.REPAIR_UPPER_THRESHOLD)
            | ~np.isfinite(values)
        )
        if not np.array_equal(corrected[unchanged_region], base_score[unchanged_region]):
            raise RuntimeError("NEXT220 changed a frozen outside-band score")
        record.update(
            {
                "base_candidate_key_sha256": (
                    n215.EXPECTED_NEXT214_FINAL_PATH_KEY_SHA256
                ),
                "beta_denominator": spec["beta_denominator"],
                "beta_fraction": spec["beta_fraction"],
                "beta_numerator": spec["beta_numerator"],
                "direction": spec["direction"],
                "feature": feature,
                "hypothesis": spec["hypothesis"],
                "redistribution_active_rows": int(active.sum()),
                "redistribution_active_scigen": int(
                    (active & (source == "scigen")).sum()
                ),
                "redistribution_active_wyformer": int(
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
    next221_keys = sorted(
        records.loc[auc_safe_non_broad, "candidate_key"].astype(str)
    )
    next221_sha = hashlib.sha256("\n".join(next221_keys).encode()).hexdigest()
    selected_spec = spec_by_key[str(selected["record"]["candidate_key"])]
    formula = {
        "protocol": PROTOCOL,
        "kind": "signed_repair_band_raw_x0_redistribution_no_dft_score",
        "base_candidate_key_sha256": n215.EXPECTED_NEXT214_FINAL_PATH_KEY_SHA256,
        "feature": selected_spec["feature"],
        "direction": selected_spec["direction"],
        "q_lo": selected_spec["q_lo"],
        "q_hi": selected_spec["q_hi"],
        "quantile_method": "inverted_cdf",
        "beta_fraction": selected_spec["beta_fraction"],
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
        "beta_fractions": list(BETA_FRACTIONS),
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "repair_lower_threshold": n215.REPAIR_LOWER_THRESHOLD,
        "repair_upper_threshold": n215.REPAIR_UPPER_THRESHOLD,
        "score_composition": SCORE_COMPOSITION,
        "base_support_unchanged": True,
        "signed_zero_centered_redistribution": True,
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    evaluation = {
        "protocol": PROTOCOL,
        "evaluation_mode": "fixed_signed_repair_band_x0_redistribution_search",
        "next214_final_candidate_reproduced": True,
        "next215_eligible_hypotheses_reproduced": True,
        "next219_closed_branch_reproduced": True,
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
        "next221_diagnostic_authorized": bool(not passes and next221_keys),
        "next221_candidate_count": len(next221_keys),
        "next221_candidate_key_sha256": next221_sha,
        "requires_unopened_internal_validation_before_claim": True,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    source_paths = {
        "src/next219_lower_anchored_broad_diagnostic.py": Path(
            n219.__file__
        ).resolve(),
        "src/next220_signed_redistribution_search.py": Path(__file__).resolve(),
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
            "next221_diagnostic_authorized": bool(not passes and next221_keys),
            "next221_candidate_count": len(next221_keys),
            "next221_candidate_key_sha256": next221_sha,
            "requires_unopened_internal_validation_before_claim": True,
            "signed_redistribution_branch_terminated": bool(
                not passes and not next221_keys
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
            raise RuntimeError("NEXT220 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT220 source changed before publication")
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
    manifest = run_signed_redistribution_search(
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
    "BETA_FRACTIONS",
    "EXPECTED_CANDIDATE_COUNT",
    "SCORE_COMPOSITION",
    "build_signed_candidate_specs",
    "materialize_signed_candidates",
    "run_signed_redistribution_search",
    "signed_redistribution_score",
]


if __name__ == "__main__":
    raise SystemExit(main())
