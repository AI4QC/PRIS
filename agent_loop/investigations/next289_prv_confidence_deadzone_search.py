#!/usr/bin/env python3
"""Frozen confidence-deadzone PRV search on the NEXT224 frontier."""

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

import src.next256_dvci_feature_audit as n256
import src.next269_prv_margin_local_search as n269
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


n268 = n269.n268
n227 = n269.n227
n223 = n269.n223
PROTOCOL = "2026-08-09-next289-prv-confidence-deadzone-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT289_PRV_CONFIDENCE_DEADZONE_CATALOGUE.json"
EVALUATION_NAME = "NEXT289_DISCOVERY_EVALUATION.json"
FORMULA_NAME = "NEXT289_FROZEN_CANDIDATE.json"
SEARCH_NAME = "next289_prv_confidence_deadzone_search.parquet"
SCORE_COMPOSITION = "nonnegative_next224_plus_triangular_margin_local_confidence_deadzone_prv_term"
DEADZONE_DENOMINATOR = 4
DEADZONE_FRACTIONS = (1 / 2, 3 / 4)
ANALYTIC_DEADZONE_FRACTIONS = (0.0, *DEADZONE_FRACTIONS)
LOCAL_WIDTH_DENOMINATOR = n269.LOCAL_WIDTH_DENOMINATOR
LOCAL_WIDTH_FRACTIONS = n269.LOCAL_WIDTH_FRACTIONS
AMPLITUDE_DENOMINATOR = n269.AMPLITUDE_DENOMINATOR
AMPLITUDE_FRACTIONS = n269.AMPLITUDE_FRACTIONS
EXPECTED_ELIGIBLE_HYPOTHESIS_COUNT = n269.EXPECTED_ELIGIBLE_HYPOTHESIS_COUNT
EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256 = n269.EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256
EXPECTED_CANDIDATE_COUNT = 1 + (
    EXPECTED_ELIGIBLE_HYPOTHESIS_COUNT
    * len(DEADZONE_FRACTIONS)
    * len(LOCAL_WIDTH_FRACTIONS)
    * len(AMPLITUDE_FRACTIONS)
)
EXPECTED_ELIGIBLE_COUNT = EXPECTED_CANDIDATE_COUNT - 1
EXPECTED_DESIGN_SHA256 = (
    "cd9cfd9277a6fd416bd64dd19c9c8c373536cd8efd7c859e73d37404367fc0ea"
)
EXPECTED_BASE_CANDIDATE_KEY_SHA256 = n269.EXPECTED_BASE_CANDIDATE_KEY_SHA256
EXPECTED_BASE_THRESHOLD = n269.EXPECTED_BASE_THRESHOLD
EXPECTED_BASE_SUPPORT_COUNT = n269.EXPECTED_BASE_SUPPORT_COUNT
REPAIR_WIDTH = n269.REPAIR_WIDTH
SEARCH_WORKERS = n269.SEARCH_WORKERS
BOUNDARY_FLAGS = n269.BOUNDARY_FLAGS
REQUIRED_STAGES = n269.REQUIRED_STAGES
REQUIRED_DESIGN_STAGES = n269.REQUIRED_DESIGN_STAGES
EXPECTED_INPUT_SHA256 = {
    **n269.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
}


def confidence_deadzone_signed_protection(
    *, protection: object, deadzone_fraction: float
) -> np.ndarray:
    """Map bounded protection to a signed tail confidence with a central zero."""

    certificate = np.asarray(protection, dtype=float)
    deadzone = float(deadzone_fraction)
    finite = np.isfinite(certificate)
    if (
        certificate.ndim != 1
        or deadzone not in ANALYTIC_DEADZONE_FRACTIONS
        or np.any((certificate[finite] < 0.0) | (certificate[finite] > 1.0))
    ):
        raise ValueError("NEXT289 confidence deadzone inputs differ")
    result = np.full(certificate.shape, np.nan, dtype=float)
    signed = 2.0 * certificate[finite] - 1.0
    result[finite] = np.sign(signed) * np.maximum(
        (np.abs(signed) - deadzone) / (1.0 - deadzone), 0.0
    )
    return result


def prv_confidence_deadzone_score(
    *,
    base_score: object,
    base_support: object,
    protection: object,
    threshold: float,
    repair_width: float,
    deadzone_fraction: float,
    local_width_fraction: float,
    amplitude_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply a triangular signed PRV correction only outside its confidence deadzone."""

    base = np.asarray(base_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    certificate = np.asarray(protection, dtype=float)
    cutoff = float(threshold)
    width = float(repair_width)
    deadzone = float(deadzone_fraction)
    fraction = float(local_width_fraction)
    amplitude = float(amplitude_fraction)
    if (
        base.ndim != 1
        or support.shape != base.shape
        or certificate.shape != base.shape
        or not math.isfinite(cutoff)
        or not math.isfinite(width)
        or width <= 0.0
        or deadzone not in ANALYTIC_DEADZONE_FRACTIONS
        or fraction not in LOCAL_WIDTH_FRACTIONS
        or amplitude not in AMPLITUDE_FRACTIONS
        or np.any(~np.isfinite(base[support]))
    ):
        raise ValueError("NEXT289 confidence-deadzone score inputs differ")
    signed = confidence_deadzone_signed_protection(
        protection=certificate, deadzone_fraction=deadzone
    )
    local_width = fraction * width
    distance = np.abs(base - cutoff)
    local_weight = np.maximum(0.0, 1.0 - distance / local_width)
    at_edge = (distance >= local_width) | np.isclose(
        distance,
        local_width,
        rtol=0.0,
        atol=1.0e-15 * max(1.0, local_width),
    )
    local_weight[at_edge] = 0.0
    local_finite = support & np.isfinite(signed) & (local_weight > 0.0)
    active = local_finite & ((deadzone == 0.0) | (np.abs(signed) > 0.0))
    effective_weight = np.zeros(base.shape, dtype=float)
    effective_weight[active] = local_weight[active]
    score = np.full(base.shape, np.nan, dtype=float)
    score[support] = base[support]
    score[active] = np.maximum(
        0.0,
        base[active]
        - amplitude
        * local_width
        * effective_weight[active]
        * signed[active],
    )
    if np.any(~np.isfinite(score[support])) or np.any(score[support] < 0.0):
        raise RuntimeError("NEXT289 confidence-deadzone score postcondition failed")
    return score, support.copy(), active, effective_weight


def _grid_numerator(value: float, denominator: int, label: str) -> int:
    scaled = float(value) * denominator
    numerator = int(round(scaled))
    if numerator <= 0 or not math.isclose(
        scaled, numerator, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(f"NEXT289 {label} grid differs")
    return numerator


def build_prv_confidence_deadzone_specs(
    *,
    base_candidate_key: str,
    eligible_table: pd.DataFrame,
    threshold: float = EXPECTED_BASE_THRESHOLD,
    repair_width: float = REPAIR_WIDTH,
    deadzone_fractions: Sequence[float] = DEADZONE_FRACTIONS,
    local_width_fractions: Sequence[float] = LOCAL_WIDTH_FRACTIONS,
    amplitude_fractions: Sequence[float] = AMPLITUDE_FRACTIONS,
) -> list[dict[str, object]]:
    """Build one base control and the complete frozen confidence-tail grammar."""

    deadzones = tuple(float(value) for value in deadzone_fractions)
    if (
        not deadzones
        or deadzones != tuple(value for value in DEADZONE_FRACTIONS if value in deadzones)
        or len(deadzones) != len(set(deadzones))
    ):
        raise ValueError("NEXT289 deadzone grid differs")
    deadzone_numerators = tuple(
        _grid_numerator(value, DEADZONE_DENOMINATOR, "deadzone")
        for value in deadzones
    )
    raw_specs = n269.build_prv_candidate_specs(
        base_candidate_key=base_candidate_key,
        eligible_table=eligible_table,
        threshold=threshold,
        repair_width=repair_width,
        local_width_fractions=local_width_fractions,
        amplitude_fractions=amplitude_fractions,
    )
    control = {
        key: value for key, value in raw_specs[0].items() if key != "candidate_key"
    }
    control.update(
        {
            "deadzone_fraction": 0.0,
            "deadzone_numerator": 0,
            "deadzone_denominator": DEADZONE_DENOMINATOR,
            "score_composition": SCORE_COMPOSITION,
        }
    )
    payloads: list[dict[str, object]] = [control]
    for raw in raw_specs[1:]:
        base_payload = {
            key: value for key, value in raw.items() if key != "candidate_key"
        }
        for deadzone, numerator in zip(
            deadzones, deadzone_numerators, strict=True
        ):
            payloads.append(
                {
                    **base_payload,
                    "deadzone_fraction": deadzone,
                    "deadzone_numerator": numerator,
                    "deadzone_denominator": DEADZONE_DENOMINATOR,
                    "score_composition": SCORE_COMPOSITION,
                }
            )
    result = [
        {
            **payload,
            "candidate_key": json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ),
        }
        for payload in payloads
    ]
    if len({str(spec["candidate_key"]) for spec in result}) != len(result):
        raise RuntimeError("NEXT289 candidate keys are not unique")
    return result


def materialize_prv_confidence_deadzone_candidates(
    *,
    features: pd.DataFrame,
    base_score: object,
    base_support: object,
    specs: Sequence[Mapping[str, object]],
) -> tuple[
    pd.DataFrame,
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, dict[str, int]],
]:
    """Encode each exact confidence-deadzone score as one evaluator term."""

    base = np.asarray(base_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    raw_specs = [dict(value) for value in specs]
    if (
        not isinstance(features, pd.DataFrame)
        or base.shape != (len(features),)
        or support.shape != base.shape
        or np.any(~np.isfinite(base[support]))
        or not raw_specs
        or len({str(spec.get("candidate_key", "")) for spec in raw_specs})
        != len(raw_specs)
        or "source_dataset" not in features.columns
    ):
        raise ValueError("NEXT289 materializer inputs differ")
    source = features["source_dataset"].astype(str).to_numpy()
    protection_cache: dict[str, np.ndarray] = {}
    columns: dict[str, np.ndarray] = {}
    terms: list[dict[str, object]] = []
    runtime: list[dict[str, object]] = []
    activity: dict[str, dict[str, int]] = {}
    for spec in raw_specs:
        feature = spec.get("feature")
        if feature is None:
            score = np.full(len(features), np.nan, dtype=float)
            score[support] = base[support]
            got_support = support.copy()
            active = np.zeros(len(features), dtype=bool)
        else:
            feature = str(feature)
            if feature not in features.columns:
                raise ValueError("NEXT289 materializer feature differs")
            hypothesis = str(spec["hypothesis"])
            if hypothesis not in protection_cache:
                values = pd.to_numeric(features[feature], errors="coerce").to_numpy(float)
                protection_cache[hypothesis] = n256.bounded_protection(
                    values=values,
                    direction=str(spec["direction"]),
                    q_lo=float(spec["q_lo"]),
                    q_hi=float(spec["q_hi"]),
                )
            score, got_support, active, _ = prv_confidence_deadzone_score(
                base_score=base,
                base_support=support,
                protection=protection_cache[hypothesis],
                threshold=float(spec["base_threshold"]),
                repair_width=float(spec["repair_width"]),
                deadzone_fraction=float(spec["deadzone_fraction"]),
                local_width_fraction=float(spec["local_width_fraction"]),
                amplitude_fraction=float(spec["amplitude_fraction"]),
            )
        if not np.array_equal(got_support, support):
            raise RuntimeError("NEXT289 support changed")
        maximum = float(np.max(score[got_support])) if got_support.any() else 0.0
        divisor = max(1.0, maximum / 50.0)
        encoded = np.full(len(features), np.nan, dtype=float)
        encoded[got_support] = np.sinh(score[got_support] / divisor)
        key = str(spec["candidate_key"])
        term_id = "next289_virtual_candidate__" + hashlib.sha256(
            key.encode()
        ).hexdigest()[:24]
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
                "group": "next289_prv_confidence_deadzone",
                "encoding": "asinh_sinh_exact_prv_confidence_deadzone_score",
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
        activity[key] = {
            "rows": int(active.sum()),
            "scigen": int((active & (source == "scigen")).sum()),
            "wyformer": int((active & (source == "wyformer")).sum()),
        }
    virtual = pd.concat([features.reset_index(drop=True), pd.DataFrame(columns)], axis=1)
    return virtual, terms, runtime, activity


def select_best_new_record(records: pd.DataFrame) -> pd.Series | None:
    return n269.select_best_new_record(records)


def _formula_from_spec(spec: Mapping[str, object] | None) -> dict[str, object]:
    if spec is None or spec.get("eligible_new_candidate") is not True:
        return {
            "protocol": PROTOCOL,
            "selected": False,
            "reason": "NO_ELIGIBLE_AUC_SAFE_CANDIDATE",
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "model_or_proxy_potential_used": False,
            "physical_relaxation_executed": False,
        }
    return {
        "protocol": PROTOCOL,
        "kind": "prv_triangular_margin_local_confidence_deadzone_x0_no_dft_score",
        "selected": True,
        "base_protocol": n223.PROTOCOL,
        "base_candidate_key_sha256": EXPECTED_BASE_CANDIDATE_KEY_SHA256,
        "base_threshold": EXPECTED_BASE_THRESHOLD,
        "repair_width": REPAIR_WIDTH,
        "hypothesis": spec["hypothesis"],
        "feature": spec["feature"],
        "direction": spec["direction"],
        "q_lo": spec["q_lo"],
        "q_hi": spec["q_hi"],
        "deadzone_fraction": spec["deadzone_fraction"],
        "local_width_fraction": spec["local_width_fraction"],
        "amplitude_fraction": spec["amplitude_fraction"],
        "nonnegative_floor": 0.0,
        "normalization_population": "ALL_FINITE_COMBINED_DISCOVERY",
        "normalization_fit_uses_endpoint": False,
        "support_policy": "UNCHANGED_FROM_NEXT214",
        "missing_policy": "TERM_OFF_KEEP_NEXT224_SCORE",
        "score_composition": SCORE_COMPOSITION,
        "dft_values_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
    }


def run_prv_confidence_deadzone_search(
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
    """Run the complete frozen discovery-only NEXT289 search."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs):
        raise ValueError("NEXT289 stage directory universe differs")
    if not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT289 design path universe differs")
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{f"next{stage}": Path(stage_dirs[stage]).resolve() for stage in REQUIRED_STAGES},
    }
    target = Path(output_dir).resolve()
    paths = n269._paths(
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
        raise ValueError("NEXT289 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT289 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT289 formal input identity differs: {differing}")
    (
        eligible_prior,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        current_key,
        formula222,
        prv_tables,
        eligible268,
    ) = n269._verify_next268(paths, input_hashes)
    combined, feature_tables, base_score, support, endpoint, _ = (
        n227._reconstruct_next224_frontier(
            paths=paths,
            eligible=eligible_prior,
            eligible214=eligible214,
            primary_key=primary_key,
            base_start_key=base_start_key,
            formula214=formula214,
            current_key=current_key,
            formula222=formula222,
        )
    )
    n269._attach_prv_features(
        combined=combined,
        feature_tables=feature_tables,
        prv_tables=prv_tables,
    )
    diagnostic224 = json.loads(paths["next224_diagnostic"].read_text())
    base_key = str(diagnostic224["global_closest"]["candidate_key"])
    if (
        hashlib.sha256(base_key.encode()).hexdigest()
        != EXPECTED_BASE_CANDIDATE_KEY_SHA256
        or int(support.sum()) != EXPECTED_BASE_SUPPORT_COUNT
    ):
        raise ValueError("NEXT289 NEXT224 base identity differs")
    specs = build_prv_confidence_deadzone_specs(
        base_candidate_key=base_key,
        eligible_table=eligible268,
    )
    eligible_count = sum(bool(spec["eligible_new_candidate"]) for spec in specs)
    if len(specs) != EXPECTED_CANDIDATE_COUNT or eligible_count != EXPECTED_ELIGIBLE_COUNT:
        raise RuntimeError("NEXT289 frozen candidate universe differs")
    virtual, terms, runtime, activity = materialize_prv_confidence_deadzone_candidates(
        features=combined,
        base_score=base_score,
        base_support=support,
        specs=specs,
    )
    runtime_by_key = {str(value["candidate_key"]): value for value in runtime}
    eligible_runtime = [
        runtime_by_key[str(spec["candidate_key"])]
        for spec in specs
        if spec["eligible_new_candidate"]
    ]
    fixed_runtime = [runtime_by_key[str(specs[0]["candidate_key"])]]
    evaluator = (
        n223.n222.n215.n214.n212.n210.n208.n205.n203.n202.n200.n194.n130.n125
        .search_optional_guard_laws_parallel
    )
    started = time.perf_counter()
    eligible_result = evaluator(
        features=virtual,
        endpoint=endpoint,
        old_terms=terms,
        optional_terms=[],
        candidate_specs=eligible_runtime,
        workers=search_workers,
    )
    fixed_result = evaluator(
        features=virtual,
        endpoint=endpoint,
        old_terms=terms,
        optional_terms=[],
        candidate_specs=fixed_runtime,
        workers=search_workers,
    )
    elapsed = time.perf_counter() - started
    if (
        int(eligible_result["candidate_count"]) != EXPECTED_ELIGIBLE_COUNT
        or int(fixed_result["candidate_count"]) != 1
        or eligible_result["cells"] != fixed_result["cells"]
        or eligible_result["pauling_by_cell"] != fixed_result["pauling_by_cell"]
    ):
        raise RuntimeError("NEXT289 evaluator accounting differs")
    spec_by_key = {str(spec["candidate_key"]): spec for spec in specs}
    raw_records = [
        *eligible_result["candidate_records"],
        *fixed_result["candidate_records"],
    ]
    for record in raw_records:
        spec = spec_by_key[str(record["candidate_key"])]
        record.update(
            {
                "hypothesis": spec["hypothesis"],
                "feature": spec["feature"],
                "direction": spec["direction"],
                "q_lo": spec["q_lo"],
                "q_hi": spec["q_hi"],
                "deadzone_fraction": spec["deadzone_fraction"],
                "deadzone_numerator": spec["deadzone_numerator"],
                "local_width_fraction": spec["local_width_fraction"],
                "local_width_numerator": spec["local_width_numerator"],
                "amplitude_fraction": spec["amplitude_fraction"],
                "amplitude_numerator": spec["amplitude_numerator"],
                "eligible_new_candidate": spec["eligible_new_candidate"],
                "is_reproduction_control": spec["is_reproduction_control"],
                "confidence_active_rows": activity[str(record["candidate_key"])]["rows"],
                "confidence_active_scigen": activity[str(record["candidate_key"])]["scigen"],
                "confidence_active_wyformer": activity[str(record["candidate_key"])]["wyformer"],
                "normalization_population": spec["normalization_population"],
                "missing_policy": spec["missing_policy"],
                "score_composition": SCORE_COMPOSITION,
            }
        )
    records = pd.DataFrame(raw_records).sort_values(
        "candidate_key", kind="mergesort"
    ).reset_index(drop=True)
    no_op_key = str(specs[0]["candidate_key"])
    no_op = records.loc[records["candidate_key"].eq(no_op_key)]
    table223 = pd.read_parquet(paths["next223_search"])
    reference_no_op = table223.loc[table223["candidate_key"].eq(base_key)]
    if len(no_op) != 1 or len(reference_no_op) != 1:
        raise RuntimeError("NEXT289 no-op reproduction identity differs")
    n223._assert_record_reproduction(no_op.iloc[0], reference_no_op.iloc[0])
    eligible_frame = records.loc[records["eligible_new_candidate"].astype(bool)]
    auc_safe_mask = (
        eligible_frame["passes_source_auc_gates"].fillna(False).astype(bool)
        & eligible_frame["passes_safe_all_cells"].fillna(False).astype(bool)
    )
    selected_row = select_best_new_record(records)
    selected: dict[str, object] | None = None
    selected_spec: dict[str, object] | None = None
    if selected_row is not None:
        selected_key = str(selected_row["candidate_key"])
        selected_result = evaluator(
            features=virtual,
            endpoint=endpoint,
            old_terms=terms,
            optional_terms=[],
            candidate_specs=[runtime_by_key[selected_key]],
            workers=1,
        )
        if str(selected_result["selected"]["record"]["candidate_key"]) != selected_key:
            raise RuntimeError("NEXT289 selected candidate reproduction differs")
        selected = selected_result["selected"]
        selected_spec = spec_by_key[selected_key]
        if selected_spec.get("eligible_new_candidate") is not True:
            raise RuntimeError("NEXT289 reproduction control was selected")
        for name, value in selected_row.items():
            if name in selected["record"]:
                selected["record"][name] = value
    passes = bool(
        eligible_frame["passes_all_discovery_gates"].fillna(False).astype(bool).any()
    )
    if selected is not None and passes != bool(
        selected["record"]["passes_all_discovery_gates"]
    ):
        raise RuntimeError("NEXT289 all-gate selection differs")
    if passes and selected is None:
        raise RuntimeError("NEXT289 all-gate candidate was not selected")
    diagnostic_mask = auc_safe_mask & ~eligible_frame[
        "passes_broad_all_cells"
    ].fillna(False).astype(bool)
    diagnostic_keys = sorted(
        eligible_frame.loc[diagnostic_mask, "candidate_key"].astype(str)
    )
    diagnostic_sha = hashlib.sha256("\n".join(diagnostic_keys).encode()).hexdigest()
    formula = _formula_from_spec(selected_spec)
    catalogue = {
        "protocol": PROTOCOL,
        "design_sha256": input_hashes["design"],
        "base_candidate_key_sha256": EXPECTED_BASE_CANDIDATE_KEY_SHA256,
        "base_threshold": EXPECTED_BASE_THRESHOLD,
        "repair_width": REPAIR_WIDTH,
        "eligible_hypotheses": list(eligible268["hypothesis"].astype(str)),
        "eligible_hypothesis_count": len(eligible268),
        "eligible_hypothesis_sha256": EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256,
        "deadzone_fractions": list(DEADZONE_FRACTIONS),
        "local_width_fractions": list(LOCAL_WIDTH_FRACTIONS),
        "amplitude_fractions": list(AMPLITUDE_FRACTIONS),
        "candidate_count": len(records),
        "eligible_new_candidate_count": len(eligible_frame),
        "reproduction_control_count": 1,
        "normalization_population": "ALL_FINITE_COMBINED_DISCOVERY",
        "normalization_fit_uses_endpoint": False,
        "base_support_unchanged": True,
        "central_confidence_region_keeps_next224_score": True,
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    evaluation = {
        "protocol": PROTOCOL,
        "evaluation_mode": "frozen_prv_confidence_deadzone_discovery_search",
        "next224_frontier_reproduced": True,
        "rows": {
            "scigen": int(len(feature_tables["scigen"])),
            "wyformer": int(len(feature_tables["wyformer"])),
            "total": int(len(combined)),
        },
        "candidate_count": len(records),
        "eligible_new_candidate_count": len(eligible_frame),
        "reproduction_control_count": 1,
        "elapsed_seconds": elapsed,
        "search_workers": search_workers,
        "counts_all": n223._gate_counts(records),
        "counts_eligible_new": n223._gate_counts(eligible_frame),
        "selected_record": None if selected is None else selected["record"],
        "selected_formula": formula,
        "selected_safe": None if selected is None else selected["safe"],
        "selected_safe_diagnostic": None if selected is None else selected["safe_diagnostic"],
        "selected_broad": None if selected is None else selected["broad"],
        "selected_source_diagnostics": None if selected is None else selected["source_diagnostics"],
        "pauling_by_cell": eligible_result["pauling_by_cell"],
        "cells": eligible_result["cells"],
        "passes_all_cross_source_discovery_gates": passes,
        "freeze_authorized": passes,
        "next290_diagnostic_authorized": bool(not passes and diagnostic_keys),
        "next290_candidate_count": len(diagnostic_keys),
        "next290_candidate_key_sha256": diagnostic_sha,
        "requires_unopened_internal_validation_before_claim": True,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next256_dvci_feature_audit.py": Path(n256.__file__).resolve(),
        "src/next269_prv_margin_local_search.py": Path(n269.__file__).resolve(),
        "src/next289_prv_confidence_deadzone_search.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
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
            "candidate_count": len(records),
            "eligible_new_candidate_count": len(eligible_frame),
            "reproduction_control_count": 1,
            "eligible_hypothesis_count": len(eligible268),
            "eligible_hypothesis_sha256": EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256,
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "next290_diagnostic_authorized": bool(not passes and diagnostic_keys),
            "next290_candidate_count": len(diagnostic_keys),
            "next290_candidate_key_sha256": diagnostic_sha,
            "requires_unopened_internal_validation_before_claim": True,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True,
            **BOUNDARY_FLAGS,
            "scientific_improvement_claim": False,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {path.name: _sha256_file(path) for path in outputs},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT289 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT289 source changed before publication")
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
    manifest = run_prv_confidence_deadzone_search(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        stage_dirs={stage: getattr(args, f"next{stage}_dir") for stage in REQUIRED_STAGES},
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


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AMPLITUDE_FRACTIONS",
    "DEADZONE_FRACTIONS",
    "EXPECTED_CANDIDATE_COUNT",
    "EXPECTED_ELIGIBLE_COUNT",
    "LOCAL_WIDTH_FRACTIONS",
    "REQUIRED_DESIGN_STAGES",
    "REQUIRED_STAGES",
    "SCORE_COMPOSITION",
    "build_prv_confidence_deadzone_specs",
    "confidence_deadzone_signed_protection",
    "materialize_prv_confidence_deadzone_candidates",
    "prv_confidence_deadzone_score",
    "run_prv_confidence_deadzone_search",
    "select_best_new_record",
]
