#!/usr/bin/env python3
"""Search frozen continuous raw-x0 risk lifts above the NEXT206 residual."""

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

import src.next209_residual_x0_broad_diagnostic as n209
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


n208 = n209.n208
PROTOCOL = "2026-08-08-next210-residual-risk-lift-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT210_RESIDUAL_RISK_LIFT_CATALOGUE.json"
EVALUATION_NAME = "NEXT210_DISCOVERY_EVALUATION.json"
FORMULA_NAME = "NEXT210_FROZEN_CANDIDATE.json"
SEARCH_NAME = "next210_residual_risk_lift_search.parquet"
EXPECTED_DESIGN_SHA256 = (
    "6cdf054a87a4a07ca761d63af24fbd519b27bcb1957247aef67f3a3054cacd70"
)
EXPECTED_ELIGIBLE_COUNT = n208.EXPECTED_ELIGIBLE_COUNT
EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256 = n208.EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256
AMPLITUDE_DENOMINATOR = 16
AMPLITUDE_FRACTIONS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
EXPECTED_CANDIDATE_COUNT = 1 + EXPECTED_ELIGIBLE_COUNT * len(AMPLITUDE_FRACTIONS)
RISK_SCALE = n208.n205.n203.SAFE_THRESHOLD - n208.n205.n203.BROAD_THRESHOLD
SEARCH_WORKERS = 4
SCORE_COMPOSITION = (
    "base_plus_amplitude_times_frozen_safe_broad_gap_times_bounded_directional_"
    "risk_if_current_score_at_or_above_residual_threshold_else_keep_base"
)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n209.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next209_design": n209.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next209_manifest": (
        "0956cbf629092e25f63769e46775ea172ebcd41992338baa6b69dbc28616758d"
    ),
    "next209_diagnostic": (
        "d35960c9c08241a846fd0bde3267173f48ff77bfcaada24c8868d6268ce55b18"
    ),
    "next209_table": (
        "702346e58a04afcba08c7be73c25aca14f80b17f52338b5523bb8de2cbdd6507"
    ),
}


def robust_risk_cutoffs(values: object) -> tuple[float, float]:
    """Return endpoint-blind 1/16 and 15/16 inverted-CDF cutoffs."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("NEXT210 robust risk cutoff inputs differ")
    finite = array[np.isfinite(array)]
    if not len(finite):
        raise ValueError("NEXT210 robust risk cutoffs have no finite values")
    q_lo = float(np.quantile(finite, 1 / 16, method="inverted_cdf"))
    q_hi = float(np.quantile(finite, 15 / 16, method="inverted_cdf"))
    if not q_hi > q_lo:
        raise ValueError("NEXT210 robust risk cutoffs are degenerate")
    return q_lo, q_hi


def bounded_directional_risk(
    values: object, direction: str, q_lo: float, q_hi: float
) -> np.ndarray:
    """Map raw feature values to a bounded severe-positive risk in [0,1]."""

    array = np.asarray(values, dtype=float)
    low = float(q_lo)
    high = float(q_hi)
    if direction not in n208.n207.PROTECTION_DIRECTIONS:
        raise ValueError("NEXT210 protection direction differs")
    if array.ndim != 1 or not math.isfinite(low) or not math.isfinite(high) or not high > low:
        raise ValueError("NEXT210 bounded risk inputs differ")
    risk = (
        (high - array) / (high - low)
        if direction == "protected_high"
        else (array - low) / (high - low)
    )
    return np.clip(risk, 0.0, 1.0)


def residual_risk_lift_score(
    *,
    base_score: object,
    base_support: object,
    feature_values: object,
    direction: str | None,
    q_lo: float | None,
    q_hi: float | None,
    residual_threshold: float,
    amplitude_fraction: float,
    risk_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add one bounded nonnegative risk lift without changing base support."""

    score = np.asarray(base_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    values = np.asarray(feature_values, dtype=float)
    threshold = float(residual_threshold)
    amplitude = float(amplitude_fraction)
    scale = float(risk_scale)
    if (
        score.ndim != 1
        or support.shape != score.shape
        or values.shape != score.shape
        or not math.isfinite(threshold)
        or threshold < 0.0
        or not math.isfinite(amplitude)
        or amplitude < 0.0
        or not math.isfinite(scale)
        or scale <= 0.0
        or np.any(~np.isfinite(score[support]))
        or np.any(score[support] < -1.0e-12)
    ):
        raise ValueError("NEXT210 base score arrays or lift parameters differ")
    if direction is None and q_lo is None and q_hi is None and amplitude == 0.0:
        return score.copy(), support.copy(), np.zeros(score.shape, dtype=bool)
    if direction not in n208.n207.PROTECTION_DIRECTIONS:
        raise ValueError("NEXT210 protection direction differs")
    if q_lo is None or q_hi is None or amplitude <= 0.0:
        raise ValueError("NEXT210 lift specification differs")
    risk = bounded_directional_risk(values, direction, q_lo, q_hi)
    active = support & (score >= threshold) & np.isfinite(risk)
    corrected = score.copy()
    corrected[active] = score[active] + amplitude * scale * risk[active]
    if np.any(corrected[active] + 1.0e-12 < score[active]):
        raise RuntimeError("NEXT210 risk lift reduced a score")
    return corrected, support.copy(), active


def _amplitude_numerator(value: float) -> int:
    numerator = int(round(float(value) * AMPLITUDE_DENOMINATOR))
    allowed = {1, 2, 4, 8, 16}
    if numerator not in allowed or not math.isclose(
        float(value), numerator / AMPLITUDE_DENOMINATOR,
        rel_tol=0.0, abs_tol=1.0e-15,
    ):
        raise ValueError("NEXT210 amplitude fraction differs")
    return numerator


def build_candidate_specs(
    *,
    base_candidate_key: str,
    eligible_hypotheses: Sequence[str],
    features: pd.DataFrame,
    base_score: object,
    base_support: object,
    residual_threshold: float,
    amplitude_fractions: Sequence[float] = AMPLITUDE_FRACTIONS,
    risk_scale: float = RISK_SCALE,
) -> list[dict[str, object]]:
    """Build the base plus exact feature/amplitude continuous-lift grid."""

    if not isinstance(base_candidate_key, str) or not base_candidate_key:
        raise ValueError("NEXT210 base candidate key must be nonempty")
    if not isinstance(features, pd.DataFrame):
        raise ValueError("NEXT210 feature table differs")
    score = np.asarray(base_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    threshold = float(residual_threshold)
    scale = float(risk_scale)
    if (
        score.shape != (len(features),)
        or support.shape != score.shape
        or not math.isfinite(threshold)
        or threshold < 0.0
        or not math.isfinite(scale)
        or scale <= 0.0
        or np.any(~np.isfinite(score[support]))
    ):
        raise ValueError("NEXT210 normalization population differs")
    names = tuple(sorted(str(value) for value in eligible_hypotheses))
    if not names or len(names) != len(set(names)):
        raise ValueError("NEXT210 eligible hypothesis universe differs")
    amplitudes = tuple(float(value) for value in amplitude_fractions)
    numerators = tuple(_amplitude_numerator(value) for value in amplitudes)
    if not amplitudes or len(numerators) != len(set(numerators)):
        raise ValueError("NEXT210 amplitude grid differs")
    fit_mask = support & np.isfinite(score) & (score >= threshold)
    if not fit_mask.any():
        raise ValueError("NEXT210 endpoint-blind normalization population is empty")

    base_payload = {
        "amplitude_denominator": AMPLITUDE_DENOMINATOR,
        "amplitude_fraction": 0.0,
        "amplitude_numerator": 0,
        "base_candidate_key": base_candidate_key,
        "direction": None,
        "feature": None,
        "hypothesis": None,
        "missing_policy": "TERM_OFF_KEEP_BASE",
        "q_hi": None,
        "q_lo": None,
        "quantile_method": "inverted_cdf",
        "residual_threshold": threshold,
        "risk_scale": scale,
        "score_composition": SCORE_COMPOSITION,
    }
    payloads: list[dict[str, object]] = [base_payload]
    for hypothesis in names:
        try:
            feature, direction = hypothesis.rsplit("__", 1)
        except ValueError as error:
            raise ValueError("NEXT210 eligible hypothesis identity differs") from error
        if (
            not feature
            or direction not in n208.n207.PROTECTION_DIRECTIONS
            or feature not in features.columns
        ):
            raise ValueError("NEXT210 eligible hypothesis identity differs")
        fit_values = pd.to_numeric(
            features.loc[fit_mask, feature], errors="coerce"
        ).to_numpy(float)
        q_lo, q_hi = robust_risk_cutoffs(fit_values)
        for amplitude, numerator in zip(amplitudes, numerators, strict=True):
            payloads.append(
                {
                    "amplitude_denominator": AMPLITUDE_DENOMINATOR,
                    "amplitude_fraction": amplitude,
                    "amplitude_numerator": numerator,
                    "base_candidate_key": base_candidate_key,
                    "direction": direction,
                    "feature": feature,
                    "hypothesis": hypothesis,
                    "missing_policy": "TERM_OFF_KEEP_BASE",
                    "q_hi": q_hi,
                    "q_lo": q_lo,
                    "quantile_method": "inverted_cdf",
                    "residual_threshold": threshold,
                    "risk_scale": scale,
                    "score_composition": SCORE_COMPOSITION,
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
        raise RuntimeError("NEXT210 candidate keys are not unique")
    return specs


def materialize_residual_risk_lift_candidates(
    *,
    features: pd.DataFrame,
    base_score: object,
    base_support: object,
    specs: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    """Encode every continuous-lift score as one exact virtual term."""

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
        raise ValueError("NEXT210 materializer inputs differ")
    columns: dict[str, np.ndarray] = {}
    terms: list[dict[str, object]] = []
    runtime: list[dict[str, object]] = []
    for spec in raw_specs:
        feature = spec.get("feature")
        if feature is None:
            feature_values = np.full(len(features), np.nan)
        elif str(feature) in features.columns:
            feature_values = pd.to_numeric(
                features[str(feature)], errors="coerce"
            ).to_numpy(float)
        else:
            raise ValueError("NEXT210 materializer feature differs")
        corrected, corrected_support, _ = residual_risk_lift_score(
            base_score=score,
            base_support=support,
            feature_values=feature_values,
            direction=spec.get("direction"),
            q_lo=spec.get("q_lo"),
            q_hi=spec.get("q_hi"),
            residual_threshold=float(spec["residual_threshold"]),
            amplitude_fraction=float(spec["amplitude_fraction"]),
            risk_scale=float(spec["risk_scale"]),
        )
        maximum = (
            float(np.max(corrected[corrected_support]))
            if corrected_support.any()
            else 0.0
        )
        divisor = max(1.0, maximum / 50.0)
        encoded = np.full(len(features), np.nan)
        encoded[corrected_support] = np.sinh(
            corrected[corrected_support] / divisor
        )
        key = str(spec["candidate_key"])
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        virtual_id = f"next210_virtual_candidate__{digest}"
        column = f"_{virtual_id}_value"
        columns[column] = encoded
        terms.append(
            {
                "term_id": virtual_id,
                "feature": column,
                "direction": 1,
                "transform": "asinh",
                "center": 0.0,
                "scale": 1.0 / divisor,
                "group": "next210_residual_risk_lift",
                "encoding": "asinh_sinh_exact_residual_risk_lift_score",
                "physical_candidate_key": key,
            }
        )
        runtime.append(
            {
                "candidate_key": key,
                "base_term_ids": [virtual_id],
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
    roots: Mapping[str, Path],
    freeze_path: Path,
    next202_design_path: Path,
    next205_design_path: Path,
    next207_design_path: Path,
    next208_design_path: Path,
    next209_design_path: Path,
    design_path: Path,
) -> dict[str, Path]:
    paths = n209._paths(
        roots,
        freeze_path,
        next202_design_path,
        next205_design_path,
        next207_design_path,
        next208_design_path,
        next209_design_path,
    )
    paths["next209_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next209_manifest": roots["next209"] / n209.MANIFEST_NAME,
            "next209_diagnostic": roots["next209"] / n209.DIAGNOSTIC_NAME,
            "next209_table": roots["next209"] / n209.TABLE_NAME,
        }
    )
    return paths


def _verify_next209(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> tuple[str, ...]:
    prior_paths = dict(paths)
    prior_paths["design"] = paths["next209_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next209_design"]
    eligible, _ = n209._verify_next208(prior_paths, prior_hashes)
    manifest = json.loads(paths["next209_manifest"].read_text())
    diagnostic = json.loads(paths["next209_diagnostic"].read_text())
    table = pd.read_parquet(paths["next209_table"])
    expected_outputs = {
        n209.DIAGNOSTIC_NAME: input_hashes["next209_diagnostic"],
        n209.TABLE_NAME: input_hashes["next209_table"],
    }
    if (
        manifest.get("protocol") != n209.PROTOCOL
        or manifest.get("candidate_count") != n209.EXPECTED_CANDIDATE_COUNT
        or manifest.get("candidate_key_sha256")
        != n209.EXPECTED_CANDIDATE_KEY_SHA256
        or manifest.get("only_candidate_is_unchanged_next206_base") is not True
        or manifest.get("new_x0_exception_candidates_reaching_auc_and_safe_count") != 0
        or manifest.get("next208_record_reproduced") is not True
        or manifest.get("next206_global_closest_residual_reproduced") is not True
        or manifest.get("existing_raw_x0_single_exception_branch_closed") is not True
        or manifest.get("new_formula_searched") is not False
        or manifest.get("new_formula_selected") is not False
        or manifest.get("outputs_sha256") != expected_outputs
        or any(manifest.get(key) is not value for key, value in n208.BOUNDARY_FLAGS.items())
        or manifest.get("executed_source_sha256", {}).get(
            "src/next209_residual_x0_broad_diagnostic.py"
        )
        != _sha256_file(Path(n209.__file__).resolve())
        or diagnostic.get("protocol") != n209.PROTOCOL
        or diagnostic.get("existing_raw_x0_single_exception_branch_closed") is not True
        or diagnostic.get("new_formula_searched") is not False
        or len(table) != n209.EXPECTED_CANDIDATE_COUNT
    ):
        raise ValueError("NEXT210 NEXT209 provenance differs")
    return eligible


def run_residual_risk_lift_search(
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
    next209_dir: Path,
    next135_freeze_path: Path,
    next202_design_path: Path,
    next205_design_path: Path,
    next207_design_path: Path,
    next208_design_path: Path,
    next209_design_path: Path,
    design_path: Path,
    output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT210 continuous risk-lift search."""

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
        **{
            f"next{stage}": Path(value).resolve()
            for stage, value in (
                (194, next194_dir), (199, next199_dir), (200, next200_dir),
                (201, next201_dir), (202, next202_dir), (203, next203_dir),
                (204, next204_dir), (205, next205_dir), (206, next206_dir),
                (207, next207_dir), (208, next208_dir), (209, next209_dir),
            )
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
        Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(search_workers) is not int or search_workers <= 0:
        raise ValueError("NEXT210 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT210 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT210 formal input identity differs: {differing}")
    eligible = _verify_next209(paths, input_hashes)
    (
        combined, feature_tables, candidate_key, base_score, base_support, endpoint,
    ) = n208._reconstruct_next206(paths=paths)
    specs = build_candidate_specs(
        base_candidate_key=candidate_key,
        eligible_hypotheses=eligible,
        features=combined,
        base_score=base_score,
        base_support=base_support,
        residual_threshold=n208.n207.EXPECTED_RESIDUAL_THRESHOLD,
    )
    if len(specs) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("NEXT210 candidate universe differs")
    combined_virtual, terms, runtime = materialize_residual_risk_lift_candidates(
        features=combined,
        base_score=base_score,
        base_support=base_support,
        specs=specs,
    )
    started = time.perf_counter()
    result = n208.n205.n203.n202.n200.n194.n130.n125.search_optional_guard_laws_parallel(
        features=combined_virtual,
        endpoint=endpoint,
        old_terms=terms,
        optional_terms=[],
        candidate_specs=runtime,
        workers=search_workers,
    )
    elapsed = time.perf_counter() - started
    if int(result["candidate_count"]) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("NEXT210 evaluator count differs")

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
        _, _, active = residual_risk_lift_score(
            base_score=base_score,
            base_support=base_support,
            feature_values=values,
            direction=spec["direction"],
            q_lo=spec["q_lo"],
            q_hi=spec["q_hi"],
            residual_threshold=float(spec["residual_threshold"]),
            amplitude_fraction=float(spec["amplitude_fraction"]),
            risk_scale=float(spec["risk_scale"]),
        )
        record.update(
            {
                "amplitude_denominator": spec["amplitude_denominator"],
                "amplitude_fraction": spec["amplitude_fraction"],
                "amplitude_numerator": spec["amplitude_numerator"],
                "base_candidate_key": candidate_key,
                "direction": spec["direction"],
                "feature": feature,
                "hypothesis": spec["hypothesis"],
                "lift_active_rows": int(active.sum()),
                "lift_active_scigen": int((active & (source == "scigen")).sum()),
                "lift_active_wyformer": int((active & (source == "wyformer")).sum()),
                "missing_policy": spec["missing_policy"],
                "q_hi": spec["q_hi"],
                "q_lo": spec["q_lo"],
                "quantile_method": spec["quantile_method"],
                "residual_threshold": spec["residual_threshold"],
                "risk_scale": spec["risk_scale"],
                "score_composition": SCORE_COMPOSITION,
            }
        )

    for record in result["candidate_records"]:
        decorate(record)
    selected = result["selected"]
    if "feature" not in selected["record"]:
        decorate(selected["record"])
    records = pd.DataFrame(result["candidate_records"])
    all_gate = records["passes_all_discovery_gates"].fillna(False).astype(bool)
    auc_safe_non_broad = (
        records["passes_source_auc_gates"].fillna(False).astype(bool)
        & records["passes_safe_all_cells"].fillna(False).astype(bool)
        & ~records["passes_broad_all_cells"].fillna(False).astype(bool)
    )
    next211_keys = sorted(records.loc[auc_safe_non_broad, "candidate_key"].astype(str))
    next211_sha = hashlib.sha256("\n".join(next211_keys).encode()).hexdigest()
    selected_spec = spec_by_key[str(selected["record"]["candidate_key"])]
    formula = {
        "protocol": PROTOCOL,
        "kind": "continuous_residual_raw_x0_risk_lift_no_dft_score",
        "base_candidate_key": candidate_key,
        "base_candidate_key_sha256": n208.n207.EXPECTED_CANDIDATE_KEY_SHA256,
        "feature": selected_spec["feature"],
        "direction": selected_spec["direction"],
        "q_lo": selected_spec["q_lo"],
        "q_hi": selected_spec["q_hi"],
        "quantile_method": "inverted_cdf",
        "amplitude_fraction": selected_spec["amplitude_fraction"],
        "risk_scale": selected_spec["risk_scale"],
        "residual_threshold": n208.n207.EXPECTED_RESIDUAL_THRESHOLD,
        "missing_policy": "TERM_OFF_KEEP_BASE",
        "support_policy": "UNCHANGED_FROM_NEXT206",
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
        "base_candidate_key_sha256": n208.n207.EXPECTED_CANDIDATE_KEY_SHA256,
        "eligible_hypotheses": list(eligible),
        "eligible_hypothesis_count": len(eligible),
        "eligible_hypothesis_sha256": EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256,
        "normalization_fit_uses_endpoint": False,
        "normalization_quantiles": [1 / 16, 15 / 16],
        "quantile_method": "inverted_cdf",
        "amplitude_fractions": list(AMPLITUDE_FRACTIONS),
        "risk_scale": RISK_SCALE,
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "score_composition": SCORE_COMPOSITION,
        "base_support_unchanged": True,
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    evaluation = {
        "protocol": PROTOCOL,
        "evaluation_mode": "fixed_continuous_residual_raw_x0_risk_lift_search",
        "next206_residual_candidate_reproduced": True,
        "next207_eligible_hypotheses_reproduced": True,
        "next209_hard_exception_branch_closure_reproduced": True,
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
        "next211_diagnostic_authorized": bool(not passes and next211_keys),
        "next211_candidate_count": len(next211_keys),
        "next211_candidate_key_sha256": next211_sha,
        "requires_unopened_internal_validation_before_claim": True,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next208_residual_x0_exception_search.py": Path(n208.__file__).resolve(),
        "src/next209_residual_x0_broad_diagnostic.py": Path(n209.__file__).resolve(),
        "src/next210_residual_risk_lift_search.py": Path(__file__).resolve(),
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
            "candidate_count": int(result["candidate_count"]),
            "eligible_hypothesis_count": len(eligible),
            "eligible_hypothesis_sha256": EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256,
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "next211_diagnostic_authorized": bool(not passes and next211_keys),
            "next211_candidate_count": len(next211_keys),
            "next211_candidate_key_sha256": next211_sha,
            "requires_unopened_internal_validation_before_claim": True,
            "continuous_residual_risk_lift_branch_terminated": bool(
                not passes and not next211_keys
            ),
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True,
            **n208.BOUNDARY_FLAGS,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {path.name: _sha256_file(path) for path in outputs},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT210 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT210 source changed before publication")
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
    for stage in (194, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209):
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--next202-design-path", type=Path, required=True)
    parser.add_argument("--next205-design-path", type=Path, required=True)
    parser.add_argument("--next207-design-path", type=Path, required=True)
    parser.add_argument("--next208-design-path", type=Path, required=True)
    parser.add_argument("--next209-design-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_residual_risk_lift_search(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{f"next{stage}_dir": getattr(args, f"next{stage}_dir") for stage in stages},
        **{
            f"next{stage}_dir": getattr(args, f"next{stage}_dir")
            for stage in (194, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209)
        },
        next135_freeze_path=args.next135_freeze_path,
        next202_design_path=args.next202_design_path,
        next205_design_path=args.next205_design_path,
        next207_design_path=args.next207_design_path,
        next208_design_path=args.next208_design_path,
        next209_design_path=args.next209_design_path,
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
    "RISK_SCALE",
    "bounded_directional_risk",
    "build_candidate_specs",
    "materialize_residual_risk_lift_candidates",
    "residual_risk_lift_score",
    "robust_risk_cutoffs",
    "run_residual_risk_lift_search",
]


if __name__ == "__main__":
    raise SystemExit(main())
