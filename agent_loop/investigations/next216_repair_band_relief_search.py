#!/usr/bin/env python3
"""Search frozen continuous x0 protection relief inside the NEXT214 band."""

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

import src.next215_repair_band_relief_audit as n215
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-08-next216-repair-band-relief-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT216_REPAIR_BAND_RELIEF_CATALOGUE.json"
EVALUATION_NAME = "NEXT216_DISCOVERY_EVALUATION.json"
FORMULA_NAME = "NEXT216_FROZEN_CANDIDATE.json"
SEARCH_NAME = "next216_repair_band_relief_search.parquet"
EXPECTED_DESIGN_SHA256 = (
    "f01ffa96607ffb489aa9b01b00a47235c03d8efae5b7530553b5dfa5bf0c3a96"
)
EXPECTED_NEXT215_SOURCE_SHA256 = (
    "d6e5232b004a934f05ef7c7cc5d4c1237474fa31e9420146239cff402c8e8e11"
)
EXPECTED_ELIGIBLE_COUNT = 22
EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256 = (
    "2e5000a319188a6191922a499b8151e28bb603ba06e70cff8750ec582e887b41"
)
AMPLITUDE_DENOMINATOR = 16
AMPLITUDE_FRACTIONS = (1 / 16, 1 / 8, 1 / 4, 1 / 2)
EXPECTED_CANDIDATE_COUNT = 1 + EXPECTED_ELIGIBLE_COUNT * len(AMPLITUDE_FRACTIONS)
SEARCH_WORKERS = 4
SCORE_COMPOSITION = (
    "next214_score_if_outside_lower_inclusive_upper_exclusive_repair_band_"
    "or_missing_else_next214_score_times_one_minus_amplitude_times_bounded_"
    "protection_certificate"
)
BOUNDARY_FLAGS = n215.BOUNDARY_FLAGS
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n215.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next215_design": n215.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next215_manifest": (
        "44bc4c90ae4ca689effaa0a1dc2de05b03792b972d96e32039ed85ff7cd0c9c3"
    ),
    "next215_catalogue": (
        "56e0bbd8e0ca178dc1d98b3ecdb449870b384de6e485143d2908c6446a7f4b85"
    ),
    "next215_audit": (
        "2e99e076af8e35bcd2522e179a504ce5ae9e0a8f6758ae454ff2a71c33b28489"
    ),
    "next215_table": (
        "e73fc703484e364f5f4ac6b8a5897b727064a95b974b194b55edaf563b72a9bb"
    ),
}


def robust_protection_cutoffs(values: object) -> tuple[float, float]:
    """Return endpoint-blind 1/16 and 15/16 inverted-CDF cutoffs."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("NEXT216 robust protection cutoff inputs differ")
    finite = array[np.isfinite(array)]
    if not len(finite):
        raise ValueError("NEXT216 robust protection cutoffs have no finite values")
    q_lo = float(np.quantile(finite, 1 / 16, method="inverted_cdf"))
    q_hi = float(np.quantile(finite, 15 / 16, method="inverted_cdf"))
    if not q_hi > q_lo:
        raise ValueError("NEXT216 robust protection cutoffs are degenerate")
    return q_lo, q_hi


def bounded_directional_protection(
    values: object, direction: str, q_lo: float, q_hi: float
) -> np.ndarray:
    """Map raw values to a bounded protected-positive certificate in [0,1]."""

    array = np.asarray(values, dtype=float)
    low = float(q_lo)
    high = float(q_hi)
    if direction not in n215.PROTECTION_DIRECTIONS:
        raise ValueError("NEXT216 protection direction differs")
    if (
        array.ndim != 1
        or not math.isfinite(low)
        or not math.isfinite(high)
        or not high > low
    ):
        raise ValueError("NEXT216 bounded protection inputs differ")
    protection = (
        (array - low) / (high - low)
        if direction == "protected_high"
        else (high - array) / (high - low)
    )
    return np.clip(protection, 0.0, 1.0)


def repair_band_relief_score(
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
    """Attenuate one score only inside the frozen repair band."""

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
        raise ValueError("NEXT216 score arrays or relief parameters differ")
    if direction is None and q_lo is None and q_hi is None and amplitude == 0.0:
        return score.copy(), support.copy(), np.zeros(score.shape, dtype=bool)
    if (
        direction not in n215.PROTECTION_DIRECTIONS
        or q_lo is None
        or q_hi is None
        or amplitude <= 0.0
    ):
        raise ValueError("NEXT216 relief specification differs")
    protection = bounded_directional_protection(values, direction, q_lo, q_hi)
    active = (
        support
        & (score >= low)
        & (score < high)
        & np.isfinite(protection)
    )
    corrected = score.copy()
    corrected[active] = score[active] * (1.0 - amplitude * protection[active])
    if (
        np.any(corrected[active] > score[active] + 1.0e-12)
        or np.any(corrected[active] < -1.0e-12)
    ):
        raise RuntimeError("NEXT216 relief violated monotone nonnegative score")
    return corrected, support.copy(), active


def _amplitude_numerator(value: float) -> int:
    numerator = int(round(float(value) * AMPLITUDE_DENOMINATOR))
    if numerator not in {1, 2, 4, 8} or not math.isclose(
        float(value),
        numerator / AMPLITUDE_DENOMINATOR,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("NEXT216 amplitude fraction differs")
    return numerator


def build_candidate_specs(
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
    """Build the unchanged base plus exact feature/amplitude relief grid."""

    if not isinstance(base_candidate_key, str) or not base_candidate_key:
        raise ValueError("NEXT216 base candidate key must be nonempty")
    if not isinstance(features, pd.DataFrame):
        raise ValueError("NEXT216 feature table differs")
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
        raise ValueError("NEXT216 normalization population differs")
    names = tuple(sorted(str(value) for value in eligible_hypotheses))
    if not names or len(names) != len(set(names)):
        raise ValueError("NEXT216 eligible hypothesis universe differs")
    amplitudes = tuple(float(value) for value in amplitude_fractions)
    numerators = tuple(_amplitude_numerator(value) for value in amplitudes)
    if not amplitudes or len(numerators) != len(set(numerators)):
        raise ValueError("NEXT216 amplitude grid differs")
    fit_mask = support & np.isfinite(score) & (score >= low) & (score < high)
    if not fit_mask.any():
        raise ValueError("NEXT216 endpoint-blind normalization population is empty")
    common = {
        "amplitude_denominator": AMPLITUDE_DENOMINATOR,
        "base_candidate_key": base_candidate_key,
        "lower_threshold": low,
        "upper_threshold": high,
        "missing_policy": "TERM_OFF_KEEP_NEXT214_SCORE",
        "quantile_method": "inverted_cdf",
        "score_composition": SCORE_COMPOSITION,
    }
    payloads: list[dict[str, object]] = [
        {
            **common,
            "amplitude_fraction": 0.0,
            "amplitude_numerator": 0,
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
            raise ValueError("NEXT216 eligible hypothesis identity differs") from error
        if (
            not feature
            or direction not in n215.PROTECTION_DIRECTIONS
            or feature not in features.columns
        ):
            raise ValueError("NEXT216 eligible hypothesis identity differs")
        fit_values = pd.to_numeric(
            features.loc[fit_mask, feature], errors="coerce"
        ).to_numpy(float)
        q_lo, q_hi = robust_protection_cutoffs(fit_values)
        for amplitude, numerator in zip(amplitudes, numerators, strict=True):
            payloads.append(
                {
                    **common,
                    "amplitude_fraction": amplitude,
                    "amplitude_numerator": numerator,
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
        raise RuntimeError("NEXT216 candidate keys are not unique")
    return specs


def materialize_repair_band_candidates(
    *,
    features: pd.DataFrame,
    base_score: object,
    base_support: object,
    specs: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    """Encode every exact physical relief score as one evaluator term."""

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
        raise ValueError("NEXT216 materializer inputs differ")
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
            raise ValueError("NEXT216 materializer feature differs")
        corrected, got_support, _ = repair_band_relief_score(
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
            "next216_virtual_candidate__"
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
                "group": "next216_repair_band_relief",
                "encoding": "asinh_sinh_exact_repair_band_relief_score",
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
    next215_design_path: Path,
    design_path: Path,
) -> dict[str, Path]:
    paths = n215._paths(
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
        next215_design_path,
    )
    paths["next215_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next215_manifest": roots["next215"] / n215.MANIFEST_NAME,
            "next215_catalogue": roots["next215"] / n215.CATALOGUE_NAME,
            "next215_audit": roots["next215"] / n215.AUDIT_NAME,
            "next215_table": roots["next215"] / n215.TABLE_NAME,
        }
    )
    return paths


def _verify_next215(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> tuple[tuple[str, ...], tuple[str, ...], str, str, dict[str, object]]:
    """Verify NEXT215 and return its frozen eligible identities."""

    prior_paths = dict(paths)
    prior_paths["design"] = paths["next215_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next215_design"]
    eligible214, primary_key, start_key, formula214 = n215._verify_next214(
        prior_paths, prior_hashes
    )
    manifest = json.loads(paths["next215_manifest"].read_text())
    catalogue = json.loads(paths["next215_catalogue"].read_text())
    audit = json.loads(paths["next215_audit"].read_text())
    table = pd.read_parquet(paths["next215_table"])
    expected_outputs = {
        n215.CATALOGUE_NAME: input_hashes["next215_catalogue"],
        n215.AUDIT_NAME: input_hashes["next215_audit"],
        n215.TABLE_NAME: input_hashes["next215_table"],
    }
    eligible = tuple(sorted(str(value) for value in audit.get("eligible_hypotheses", [])))
    eligible_table = tuple(
        sorted(
            table.loc[
                table["eligible_for_search"].fillna(False).astype(bool),
                "hypothesis",
            ].astype(str)
        )
    )
    eligible_sha = hashlib.sha256("\n".join(eligible).encode()).hexdigest()
    if (
        manifest.get("protocol") != n215.PROTOCOL
        or manifest.get("feature_count") != n215.EXPECTED_FEATURE_COUNT
        or manifest.get("hypothesis_count") != n215.EXPECTED_HYPOTHESIS_COUNT
        or manifest.get("raw_gate_passing_count") != 23
        or manifest.get("eligible_hypothesis_count") != EXPECTED_ELIGIBLE_COUNT
        or manifest.get("eligible_hypothesis_sha256")
        != EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256
        or manifest.get("next216_search_authorized") is not True
        or manifest.get("new_formula_searched") is not False
        or manifest.get("new_formula_selected") is not False
        or manifest.get("outputs_sha256") != expected_outputs
        or any(manifest.get(key) is not value for key, value in BOUNDARY_FLAGS.items())
        or manifest.get("executed_source_sha256", {}).get(
            "src/next215_repair_band_relief_audit.py"
        )
        != EXPECTED_NEXT215_SOURCE_SHA256
        or _sha256_file(Path(n215.__file__).resolve())
        != EXPECTED_NEXT215_SOURCE_SHA256
        or catalogue.get("protocol") != n215.PROTOCOL
        or catalogue.get("design_sha256") != input_hashes["next215_design"]
        or catalogue.get("feature_count") != n215.EXPECTED_FEATURE_COUNT
        or catalogue.get("hypothesis_count") != n215.EXPECTED_HYPOTHESIS_COUNT
        or audit.get("protocol") != n215.PROTOCOL
        or audit.get("next216_search_authorized") is not True
        or audit.get("raw_gate_passing_count") != 23
        or audit.get("eligible_hypothesis_count") != EXPECTED_ELIGIBLE_COUNT
        or audit.get("eligible_hypothesis_sha256")
        != EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256
        or len(eligible) != EXPECTED_ELIGIBLE_COUNT
        or eligible_sha != EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256
        or eligible != eligible_table
        or len(table) != n215.EXPECTED_HYPOTHESIS_COUNT
    ):
        raise ValueError("NEXT216 NEXT215 provenance differs")
    return eligible, eligible214, primary_key, start_key, formula214


def run_repair_band_relief_search(
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
    next215_dir: Path,
    next135_freeze_path: Path, next202_design_path: Path,
    next205_design_path: Path, next207_design_path: Path,
    next208_design_path: Path, next209_design_path: Path,
    next210_design_path: Path, next211_design_path: Path,
    next212_design_path: Path, next213_design_path: Path,
    next214_design_path: Path, next215_design_path: Path,
    design_path: Path, output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only repair-band relief search."""

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
        (213, next213_dir), (214, next214_dir), (215, next215_dir),
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
        Path(next215_design_path).resolve(),
        Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(search_workers) is not int or search_workers <= 0:
        raise ValueError("NEXT216 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT216 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT216 formal input identity differs: {differing}")
    eligible, eligible214, primary_key, start_key, formula214 = _verify_next215(
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
        raise ValueError("NEXT216 NEXT214 base identity differs")
    base_candidate_key = str(accepted.iloc[0]["candidate_key"])
    if (
        hashlib.sha256(base_candidate_key.encode()).hexdigest()
        != n215.EXPECTED_NEXT214_FINAL_PATH_KEY_SHA256
    ):
        raise ValueError("NEXT216 NEXT214 base key differs")
    specs = build_candidate_specs(
        base_candidate_key=base_candidate_key,
        eligible_hypotheses=eligible,
        features=combined,
        base_score=base_score,
        base_support=base_support,
    )
    if len(specs) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("NEXT216 candidate universe differs")
    combined_virtual, terms, runtime = materialize_repair_band_candidates(
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
        raise RuntimeError("NEXT216 evaluator count differs")

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
        corrected, support, active = repair_band_relief_score(
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
        if not np.array_equal(corrected[unchanged_region], base_score[unchanged_region]):
            raise RuntimeError("NEXT216 changed a frozen outside-band score")
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
                "relief_active_rows": int(active.sum()),
                "relief_active_scigen": int((active & (source == "scigen")).sum()),
                "relief_active_wyformer": int((active & (source == "wyformer")).sum()),
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
    all_gate = records["passes_all_discovery_gates"].fillna(False).astype(bool)
    auc_safe_non_broad = (
        records["passes_source_auc_gates"].fillna(False).astype(bool)
        & records["passes_safe_all_cells"].fillna(False).astype(bool)
        & ~records["passes_broad_all_cells"].fillna(False).astype(bool)
    )
    next217_keys = sorted(
        records.loc[auc_safe_non_broad, "candidate_key"].astype(str)
    )
    next217_sha = hashlib.sha256("\n".join(next217_keys).encode()).hexdigest()
    selected_spec = spec_by_key[str(selected["record"]["candidate_key"])]
    formula = {
        "protocol": PROTOCOL,
        "kind": "continuous_repair_band_raw_x0_relief_no_dft_score",
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
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    evaluation = {
        "protocol": PROTOCOL,
        "evaluation_mode": "fixed_continuous_repair_band_x0_relief_search",
        "next214_final_candidate_reproduced": True,
        "next215_eligible_hypotheses_reproduced": True,
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
        "next217_diagnostic_authorized": bool(not passes and next217_keys),
        "next217_candidate_count": len(next217_keys),
        "next217_candidate_key_sha256": next217_sha,
        "requires_unopened_internal_validation_before_claim": True,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    source_paths = {
        "src/next214_forward_stagewise_risk_lift.py": Path(
            n215.n214.__file__
        ).resolve(),
        "src/next215_repair_band_relief_audit.py": Path(n215.__file__).resolve(),
        "src/next216_repair_band_relief_search.py": Path(__file__).resolve(),
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
            "next217_diagnostic_authorized": bool(not passes and next217_keys),
            "next217_candidate_count": len(next217_keys),
            "next217_candidate_key_sha256": next217_sha,
            "requires_unopened_internal_validation_before_claim": True,
            "repair_band_relief_branch_terminated": bool(
                not passes and not next217_keys
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
            raise RuntimeError("NEXT216 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT216 source changed before publication")
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
        210, 211, 212, 213, 214, 215,
    )
    for stage in early_stages + later_stages:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    for stage in (202, 205, 207, 208, 209, 210, 211, 212, 213, 214, 215):
        parser.add_argument(f"--next{stage}-design-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_repair_band_relief_search(
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
            for stage in (202, 205, 207, 208, 209, 210, 211, 212, 213, 214, 215)
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
    "bounded_directional_protection",
    "build_candidate_specs",
    "materialize_repair_band_candidates",
    "repair_band_relief_score",
    "robust_protection_cutoffs",
    "run_repair_band_relief_search",
]


if __name__ == "__main__":
    raise SystemExit(main())
