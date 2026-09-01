#!/usr/bin/env python3
"""Frozen threshold-local signed redistribution search on NEXT224."""

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

import src.next227_margin_local_feature_audit as n227
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


n223 = n227.n226.n225.n223
n216 = n223.n222.n220.n216
PROTOCOL = "2026-08-09-next228-margin-local-redistribution-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT228_MARGIN_LOCAL_REDISTRIBUTION_CATALOGUE.json"
EVALUATION_NAME = "NEXT228_DISCOVERY_EVALUATION.json"
FORMULA_NAME = "NEXT228_FROZEN_CANDIDATE.json"
SEARCH_NAME = "next228_margin_local_redistribution_search.parquet"
SCORE_COMPOSITION = "nonnegative_next224_plus_triangular_margin_local_signed_term"
LOCAL_WIDTH_DENOMINATOR = 64
LOCAL_WIDTH_FRACTIONS = (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4)
AMPLITUDE_DENOMINATOR = 4
AMPLITUDE_FRACTIONS = (1 / 4, 1 / 2, 1.0)
EXPECTED_ELIGIBLE_HYPOTHESIS_COUNT = 26
EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256 = (
    "a499927148c493fbc0769e0d57e71609e31fbe0bfcc2c819a21172a02ce4d8ee"
)
EXPECTED_CANDIDATE_COUNT = 1 + (
    EXPECTED_ELIGIBLE_HYPOTHESIS_COUNT
    * len(LOCAL_WIDTH_FRACTIONS)
    * len(AMPLITUDE_FRACTIONS)
)
EXPECTED_ELIGIBLE_COUNT = EXPECTED_CANDIDATE_COUNT - 1
EXPECTED_DESIGN_SHA256 = n227.EXPECTED_DESIGN_SHA256
EXPECTED_BASE_CANDIDATE_KEY_SHA256 = n227.EXPECTED_BASE_CANDIDATE_KEY_SHA256
EXPECTED_BASE_THRESHOLD = n227.EXPECTED_BASE_THRESHOLD
EXPECTED_BASE_SUPPORT_COUNT = n227.EXPECTED_BASE_SUPPORT_COUNT
EXPECTED_BASE_FAILED_COUNT = n227.EXPECTED_BASE_FAILED_COUNT
EXPECTED_BASE_SHORTFALL = n227.EXPECTED_BASE_SHORTFALL
REPAIR_WIDTH = (
    n223.n222.n215.REPAIR_UPPER_THRESHOLD
    - n223.n222.n215.REPAIR_LOWER_THRESHOLD
)
SEARCH_WORKERS = n223.SEARCH_WORKERS
BOUNDARY_FLAGS = n227.BOUNDARY_FLAGS
REQUIRED_STAGES = (*n227.REQUIRED_STAGES, 227)
REQUIRED_DESIGN_STAGES = (*n227.REQUIRED_DESIGN_STAGES, 227)
EXPECTED_NEXT227_SOURCE_SHA256 = (
    "e092b45d8e3d01cb8c5754e7de0bd6908ea00aa49b4286234074db23758d4ef1"
)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n227.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next227_design": n227.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next227_manifest": (
        "f23664d364c4329cb7301b7c7eefd34606b5a52a365f32aed863cd84d7e8550b"
    ),
    "next227_catalogue": (
        "b545755026b5b94d00515d53264e3397d16b3a4e00a52a8e38ba08b373f0bfee"
    ),
    "next227_audit": (
        "72f3ca905dcf142408926c4e3b61feb38d5b4087d2467b4b4f85d75def1b63b5"
    ),
    "next227_table": (
        "fad6264181c767bb5b2302de6c8e2732673f38214f4aadde354af14910676e8f"
    ),
}


def margin_local_redistribution_score(
    *,
    base_score: object,
    base_support: object,
    feature_values: object,
    direction: str,
    q_lo: float,
    q_hi: float,
    threshold: float,
    repair_width: float,
    local_width_fraction: float,
    amplitude_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the frozen triangular signed correction around one threshold."""

    base = np.asarray(base_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    raw = np.asarray(feature_values, dtype=float)
    cutoff = float(threshold)
    width = float(repair_width)
    fraction = float(local_width_fraction)
    amplitude = float(amplitude_fraction)
    if (
        base.ndim != 1
        or support.shape != base.shape
        or raw.shape != base.shape
        or direction not in n227.PROTECTION_DIRECTIONS
        or not math.isfinite(float(q_lo))
        or not math.isfinite(float(q_hi))
        or not float(q_hi) > float(q_lo)
        or not math.isfinite(cutoff)
        or not math.isfinite(width)
        or width <= 0.0
        or fraction not in LOCAL_WIDTH_FRACTIONS
        or amplitude not in AMPLITUDE_FRACTIONS
        or np.any(~np.isfinite(base[support]))
    ):
        raise ValueError("NEXT228 margin-local score inputs differ")
    protection = n216.bounded_directional_protection(
        raw, direction, float(q_lo), float(q_hi)
    )
    h = fraction * width
    distance = np.abs(base - cutoff)
    local_weight = np.maximum(0.0, 1.0 - distance / h)
    at_edge = (distance >= h) | np.isclose(
        distance, h, rtol=0.0, atol=1.0e-15 * max(1.0, h)
    )
    local_weight[at_edge] = 0.0
    active = support & np.isfinite(protection) & (local_weight > 0.0)
    effective_weight = np.zeros(base.shape, dtype=float)
    effective_weight[active] = local_weight[active]
    score = np.full(base.shape, np.nan, dtype=float)
    score[support] = base[support]
    score[active] = np.maximum(
        0.0,
        base[active]
        + amplitude
        * h
        * effective_weight[active]
        * (1.0 - 2.0 * protection[active]),
    )
    return score, support.copy(), active, effective_weight, protection


def _grid_numerator(value: float, denominator: int, label: str) -> int:
    scaled = float(value) * denominator
    numerator = int(round(scaled))
    if (
        numerator <= 0
        or not math.isclose(scaled, numerator, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise ValueError(f"NEXT228 {label} grid differs")
    return numerator


def build_margin_local_candidate_specs(
    *,
    base_candidate_key: str,
    eligible_hypotheses: Sequence[str],
    features: pd.DataFrame,
    base_score: object,
    base_support: object,
    threshold: float = EXPECTED_BASE_THRESHOLD,
    repair_width: float = REPAIR_WIDTH,
    local_width_fractions: Sequence[float] = LOCAL_WIDTH_FRACTIONS,
    amplitude_fractions: Sequence[float] = AMPLITUDE_FRACTIONS,
) -> list[dict[str, object]]:
    """Build one no-op and the complete feature x width x amplitude grammar."""

    if not isinstance(base_candidate_key, str) or not base_candidate_key:
        raise ValueError("NEXT228 base candidate key differs")
    if not isinstance(features, pd.DataFrame):
        raise ValueError("NEXT228 feature table differs")
    base = np.asarray(base_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    cutoff = float(threshold)
    width = float(repair_width)
    if (
        base.shape != (len(features),)
        or support.shape != base.shape
        or np.any(~np.isfinite(base[support]))
        or not math.isfinite(cutoff)
        or not math.isfinite(width)
        or width <= 0.0
    ):
        raise ValueError("NEXT228 base population differs")
    hypotheses = tuple(sorted(str(value) for value in eligible_hypotheses))
    if not hypotheses or len(hypotheses) != len(set(hypotheses)):
        raise ValueError("NEXT228 eligible hypothesis universe differs")
    fractions = tuple(float(value) for value in local_width_fractions)
    amplitudes = tuple(float(value) for value in amplitude_fractions)
    fraction_numerators = tuple(
        _grid_numerator(value, LOCAL_WIDTH_DENOMINATOR, "local-width")
        for value in fractions
    )
    amplitude_numerators = tuple(
        _grid_numerator(value, AMPLITUDE_DENOMINATOR, "amplitude")
        for value in amplitudes
    )
    if (
        not fractions
        or fractions
        != tuple(value for value in LOCAL_WIDTH_FRACTIONS if value in fractions)
        or len(fractions) != len(set(fractions))
        or not amplitudes
        or amplitudes
        != tuple(value for value in AMPLITUDE_FRACTIONS if value in amplitudes)
        or len(amplitudes) != len(set(amplitudes))
    ):
        raise ValueError("NEXT228 frozen grid differs")
    common = {
        "base_candidate_key_sha256": hashlib.sha256(
            base_candidate_key.encode()
        ).hexdigest(),
        "base_threshold": cutoff,
        "repair_width": width,
        "local_width_denominator": LOCAL_WIDTH_DENOMINATOR,
        "amplitude_denominator": AMPLITUDE_DENOMINATOR,
        "normalization_population": "ALL_FINITE_COMBINED_DISCOVERY",
        "quantile_method": "inverted_cdf",
        "missing_policy": "TERM_OFF_KEEP_NEXT224_SCORE",
        "support_policy": "UNCHANGED_FROM_NEXT214",
        "score_composition": SCORE_COMPOSITION,
    }
    payloads: list[dict[str, object]] = [
        {
            **common,
            "hypothesis": None,
            "feature": None,
            "direction": None,
            "q_lo": None,
            "q_hi": None,
            "local_width_fraction": 0.0,
            "local_width_numerator": 0,
            "amplitude_fraction": 0.0,
            "amplitude_numerator": 0,
            "eligible_new_candidate": False,
            "is_reproduction_control": True,
        }
    ]
    for hypothesis in hypotheses:
        try:
            feature, direction = hypothesis.rsplit("__", 1)
        except ValueError as error:
            raise ValueError("NEXT228 eligible hypothesis identity differs") from error
        if (
            not feature
            or direction not in n227.PROTECTION_DIRECTIONS
            or feature not in features.columns
        ):
            raise ValueError("NEXT228 eligible hypothesis identity differs")
        values = pd.to_numeric(features[feature], errors="coerce").to_numpy(float)
        q_lo, q_hi = n216.robust_protection_cutoffs(values[np.isfinite(values)])
        for fraction, fraction_numerator in zip(
            fractions, fraction_numerators, strict=True
        ):
            for amplitude, amplitude_numerator in zip(
                amplitudes, amplitude_numerators, strict=True
            ):
                payloads.append(
                    {
                        **common,
                        "hypothesis": hypothesis,
                        "feature": feature,
                        "direction": direction,
                        "q_lo": q_lo,
                        "q_hi": q_hi,
                        "local_width_fraction": fraction,
                        "local_width_numerator": fraction_numerator,
                        "amplitude_fraction": amplitude,
                        "amplitude_numerator": amplitude_numerator,
                        "eligible_new_candidate": True,
                        "is_reproduction_control": False,
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
        raise RuntimeError("NEXT228 candidate keys are not unique")
    return specs


def materialize_margin_local_candidates(
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
    """Encode each exact physical score as one virtual evaluator term."""

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
        raise ValueError("NEXT228 materializer inputs differ")
    source = features["source_dataset"].astype(str).to_numpy()
    raw_cache: dict[str, np.ndarray] = {}
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
                raise ValueError("NEXT228 materializer feature differs")
            if feature not in raw_cache:
                raw_cache[feature] = pd.to_numeric(
                    features[feature], errors="coerce"
                ).to_numpy(float)
            score, got_support, active, _, _ = margin_local_redistribution_score(
                base_score=base,
                base_support=support,
                feature_values=raw_cache[feature],
                direction=str(spec["direction"]),
                q_lo=float(spec["q_lo"]),
                q_hi=float(spec["q_hi"]),
                threshold=float(spec["base_threshold"]),
                repair_width=float(spec["repair_width"]),
                local_width_fraction=float(spec["local_width_fraction"]),
                amplitude_fraction=float(spec["amplitude_fraction"]),
            )
        if not np.array_equal(got_support, support):
            raise RuntimeError("NEXT228 support changed")
        maximum = float(np.max(score[got_support])) if got_support.any() else 0.0
        divisor = max(1.0, maximum / 50.0)
        encoded = np.full(len(features), np.nan, dtype=float)
        encoded[got_support] = np.sinh(score[got_support] / divisor)
        key = str(spec["candidate_key"])
        term_id = (
            "next228_virtual_candidate__"
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
                "group": "next228_margin_local_redistribution",
                "encoding": "asinh_sinh_exact_margin_local_score",
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
    virtual = pd.concat(
        [features.reset_index(drop=True), pd.DataFrame(columns)], axis=1
    )
    return virtual, terms, runtime, activity


def _paths(
    *,
    roots: Mapping[str, Path],
    next135_freeze_path: Path,
    design_paths: Mapping[int, Path],
    design_path: Path,
) -> dict[str, Path]:
    paths = n227._paths(
        roots=roots,
        next135_freeze_path=next135_freeze_path,
        design_paths={
            stage: design_paths[stage] for stage in n227.REQUIRED_DESIGN_STAGES
        },
        design_path=design_paths[227],
    )
    paths["next227_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next227_manifest": roots["next227"] / n227.MANIFEST_NAME,
            "next227_catalogue": roots["next227"] / n227.CATALOGUE_NAME,
            "next227_audit": roots["next227"] / n227.AUDIT_NAME,
            "next227_table": roots["next227"] / n227.TABLE_NAME,
        }
    )
    return paths


def _verify_next227(
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
]:
    prior_paths = dict(paths)
    prior_paths["design"] = paths["next227_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next227_design"]
    prior = n227._verify_next226(prior_paths, prior_hashes)
    manifest = json.loads(paths["next227_manifest"].read_text())
    catalogue = json.loads(paths["next227_catalogue"].read_text())
    audit = json.loads(paths["next227_audit"].read_text())
    table = pd.read_parquet(paths["next227_table"])
    eligible = tuple(sorted(str(value) for value in audit["eligible_hypotheses"]))
    table_eligible = tuple(
        sorted(
            table.loc[
                table["eligible_for_search"].fillna(False).astype(bool),
                "hypothesis",
            ].astype(str)
        )
    )
    expected_outputs = {
        n227.CATALOGUE_NAME: input_hashes["next227_catalogue"],
        n227.AUDIT_NAME: input_hashes["next227_audit"],
        n227.TABLE_NAME: input_hashes["next227_table"],
    }
    if (
        manifest.get("protocol") != n227.PROTOCOL
        or manifest.get("eligible_hypothesis_count")
        != EXPECTED_ELIGIBLE_HYPOTHESIS_COUNT
        or manifest.get("eligible_hypothesis_sha256")
        != EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256
        or manifest.get("next228_search_authorized") is not True
        or manifest.get("new_formula_searched") is not False
        or manifest.get("new_formula_selected") is not False
        or manifest.get("outputs_sha256") != expected_outputs
        or manifest.get("executed_source_sha256", {}).get(
            "src/next227_margin_local_feature_audit.py"
        )
        != EXPECTED_NEXT227_SOURCE_SHA256
        or _sha256_file(Path(n227.__file__).resolve())
        != EXPECTED_NEXT227_SOURCE_SHA256
        or catalogue.get("design_sha256") != EXPECTED_DESIGN_SHA256
        or audit.get("eligible_hypothesis_count")
        != EXPECTED_ELIGIBLE_HYPOTHESIS_COUNT
        or audit.get("eligible_hypothesis_sha256")
        != EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256
        or audit.get("next228_search_authorized") is not True
        or eligible != table_eligible
        or len(eligible) != EXPECTED_ELIGIBLE_HYPOTHESIS_COUNT
        or hashlib.sha256("\n".join(eligible).encode()).hexdigest()
        != EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256
        or any(manifest.get(key) is not value for key, value in BOUNDARY_FLAGS.items())
    ):
        raise ValueError("NEXT228 NEXT227 provenance differs")
    return (*prior, eligible)


def _formula_from_spec(spec: Mapping[str, object] | None) -> dict[str, object]:
    if spec is None:
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
        "kind": "triangular_margin_local_signed_x0_no_dft_score",
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


def run_margin_local_redistribution_search(
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
    """Run the complete frozen discovery-only NEXT228 search."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs):
        raise ValueError("NEXT228 stage directory universe differs")
    if not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT228 design path universe differs")
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
        raise ValueError("NEXT228 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT228 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT228 formal input identity differs: {differing}")
    (
        eligible_prior,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        current_key,
        formula222,
        eligible227,
    ) = _verify_next227(paths, input_hashes)
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
    diagnostic224 = json.loads(paths["next224_diagnostic"].read_text())
    base_key = str(diagnostic224["global_closest"]["candidate_key"])
    if (
        hashlib.sha256(base_key.encode()).hexdigest()
        != EXPECTED_BASE_CANDIDATE_KEY_SHA256
        or int(support.sum()) != EXPECTED_BASE_SUPPORT_COUNT
    ):
        raise ValueError("NEXT228 NEXT224 base identity differs")
    specs = build_margin_local_candidate_specs(
        base_candidate_key=base_key,
        eligible_hypotheses=eligible227,
        features=combined,
        base_score=base_score,
        base_support=support,
    )
    eligible_count = sum(bool(spec["eligible_new_candidate"]) for spec in specs)
    if len(specs) != EXPECTED_CANDIDATE_COUNT or eligible_count != EXPECTED_ELIGIBLE_COUNT:
        raise RuntimeError("NEXT228 frozen candidate universe differs")
    virtual, terms, runtime, activity = materialize_margin_local_candidates(
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
        raise RuntimeError("NEXT228 evaluator accounting differs")
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
                "local_width_fraction": spec["local_width_fraction"],
                "local_width_numerator": spec["local_width_numerator"],
                "amplitude_fraction": spec["amplitude_fraction"],
                "amplitude_numerator": spec["amplitude_numerator"],
                "eligible_new_candidate": spec["eligible_new_candidate"],
                "is_reproduction_control": spec["is_reproduction_control"],
                "local_active_rows": activity[str(record["candidate_key"])]["rows"],
                "local_active_scigen": activity[str(record["candidate_key"])]["scigen"],
                "local_active_wyformer": activity[str(record["candidate_key"])]["wyformer"],
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
        raise RuntimeError("NEXT228 no-op reproduction identity differs")
    n223._assert_record_reproduction(no_op.iloc[0], reference_no_op.iloc[0])
    eligible_frame = records.loc[records["eligible_new_candidate"].astype(bool)]
    auc_safe_mask = (
        eligible_frame["passes_source_auc_gates"].fillna(False).astype(bool)
        & eligible_frame["passes_safe_all_cells"].fillna(False).astype(bool)
    )
    selected_row = (
        n223.select_best_eligible_record(records) if bool(auc_safe_mask.any()) else None
    )
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
            raise RuntimeError("NEXT228 selected candidate reproduction differs")
        selected = selected_result["selected"]
        selected_spec = spec_by_key[selected_key]
        for name, value in selected_row.items():
            if name in selected["record"]:
                selected["record"][name] = value
    passes = bool(
        eligible_frame["passes_all_discovery_gates"].fillna(False).astype(bool).any()
    )
    if selected is not None and passes != bool(
        selected["record"]["passes_all_discovery_gates"]
    ):
        raise RuntimeError("NEXT228 all-gate selection differs")
    if passes and selected is None:
        raise RuntimeError("NEXT228 all-gate candidate was not selected")
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
        "eligible_hypotheses": list(eligible227),
        "eligible_hypothesis_count": len(eligible227),
        "eligible_hypothesis_sha256": EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256,
        "local_width_fractions": list(LOCAL_WIDTH_FRACTIONS),
        "amplitude_fractions": list(AMPLITUDE_FRACTIONS),
        "candidate_count": len(records),
        "eligible_new_candidate_count": len(eligible_frame),
        "reproduction_control_count": 1,
        "normalization_population": "ALL_FINITE_COMBINED_DISCOVERY",
        "normalization_fit_uses_endpoint": False,
        "base_support_unchanged": True,
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    evaluation = {
        "protocol": PROTOCOL,
        "evaluation_mode": "frozen_margin_local_redistribution_discovery_search",
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
        "next229_diagnostic_authorized": bool(not passes and diagnostic_keys),
        "next229_candidate_count": len(diagnostic_keys),
        "next229_candidate_key_sha256": diagnostic_sha,
        "requires_unopened_internal_validation_before_claim": True,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    source_paths = {
        "src/next227_margin_local_feature_audit.py": Path(n227.__file__).resolve(),
        "src/next228_margin_local_redistribution_search.py": Path(__file__).resolve(),
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
            "candidate_count": len(records),
            "eligible_new_candidate_count": len(eligible_frame),
            "reproduction_control_count": 1,
            "eligible_hypothesis_count": len(eligible227),
            "eligible_hypothesis_sha256": EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256,
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "next229_diagnostic_authorized": bool(not passes and diagnostic_keys),
            "next229_candidate_count": len(diagnostic_keys),
            "next229_candidate_key_sha256": diagnostic_sha,
            "requires_unopened_internal_validation_before_claim": True,
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
            raise RuntimeError("NEXT228 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT228 source changed before publication")
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
    manifest = run_margin_local_redistribution_search(
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
    "EXPECTED_ELIGIBLE_COUNT",
    "LOCAL_WIDTH_FRACTIONS",
    "build_margin_local_candidate_specs",
    "margin_local_redistribution_score",
    "materialize_margin_local_candidates",
    "run_margin_local_redistribution_search",
]


if __name__ == "__main__":
    raise SystemExit(main())
