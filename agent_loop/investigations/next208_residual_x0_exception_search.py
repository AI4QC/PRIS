#!/usr/bin/env python3
"""Search frozen raw-x0 exceptions for the NEXT206 residual cohort."""

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

import src.next207_residual_x0_feature_audit as n207
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


n206 = n207.n206
n205 = n207.n205
PROTOCOL = "2026-08-08-next208-residual-x0-exception-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT208_RESIDUAL_X0_EXCEPTION_CATALOGUE.json"
EVALUATION_NAME = "NEXT208_DISCOVERY_EVALUATION.json"
FORMULA_NAME = "NEXT208_FROZEN_CANDIDATE.json"
SEARCH_NAME = "next208_residual_x0_exception_search.parquet"
EXPECTED_DESIGN_SHA256 = (
    "a5acfe5db166577e7f39effb99f4109adbe8f1de7378a8ab557b72e9246988c3"
)
EXPECTED_ELIGIBLE_COUNT = 44
EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256 = (
    "9d5ccc3ca8dd31c2b4b230330d141f9a05202900cd1f0e243f4140efb60ec24a"
)
EXCEPTION_DENOMINATOR = 16
EXCEPTION_FRACTIONS = tuple(k / EXCEPTION_DENOMINATOR for k in range(1, 16))
EXPECTED_CANDIDATE_COUNT = 1 + EXPECTED_ELIGIBLE_COUNT * len(EXCEPTION_FRACTIONS)
SEARCH_WORKERS = 4
SCORE_COMPOSITION = (
    "zero_if_current_score_at_or_above_residual_threshold_and_finite_feature_"
    "satisfies_directional_cutoff_else_keep_base"
)
BOUNDARY_FLAGS = {
    "dft_calculation_executed": False,
    "dft_values_used_by_executable_formula": False,
    "learned_energy_force_stress_proxy_used": False,
    "model_or_proxy_potential_used": False,
    "physical_relaxation_executed": False,
    "opened_validation_outputs_used": False,
    "scigen_replication_endpoint_opened": False,
    "wyformer_replication_endpoint_opened": False,
}
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n207.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next207_design": n207.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next207_manifest": (
        "6535b0d315059e3e60169da8164d024ae820a3a6d1e65d477b9a08ca85799a33"
    ),
    "next207_catalogue": (
        "e8b9951944ad7b85ac66cb6e58dfd495068b2ffe6854735a82fd26cbca22e875"
    ),
    "next207_audit": (
        "0ac52f065f3f77132b9365144b6d77ee7a36b270bfcde19e70b0530c228d88de"
    ),
    "next207_table": (
        "f89a637ca74c3e2f1d5653cce2c414c2af4fa461b73d893f7f9d996c5e2ea3c1"
    ),
}


def empirical_exception_cutoff(
    values: object, direction: str, exception_fraction: float
) -> float:
    """Return one endpoint-blind empirical cutoff with inverted-CDF semantics."""

    array = np.asarray(values, dtype=float)
    fraction = float(exception_fraction)
    if (
        array.ndim != 1
        or direction not in n207.PROTECTION_DIRECTIONS
        or not math.isfinite(fraction)
        or not 0.0 < fraction < 1.0
    ):
        if direction not in n207.PROTECTION_DIRECTIONS:
            raise ValueError("NEXT208 protection direction differs")
        raise ValueError("NEXT208 empirical cutoff inputs differ")
    finite = array[np.isfinite(array)]
    if not len(finite):
        raise ValueError("NEXT208 empirical cutoff has no finite values")
    quantile = fraction if direction == "protected_low" else 1.0 - fraction
    return float(np.quantile(finite, quantile, method="inverted_cdf"))


def residual_x0_exception_score(
    *,
    base_score: object,
    base_support: object,
    feature_values: object,
    direction: str | None,
    cutoff: float | None,
    residual_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one fixed zero-score raw-x0 exception without changing support."""

    score = np.asarray(base_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    values = np.asarray(feature_values, dtype=float)
    threshold = float(residual_threshold)
    if (
        score.ndim != 1
        or support.shape != score.shape
        or values.shape != score.shape
        or not math.isfinite(threshold)
        or threshold < 0.0
        or np.any(~np.isfinite(score[support]))
        or np.any(score[support] < -1.0e-12)
    ):
        raise ValueError("NEXT208 base score arrays or threshold differ")
    if direction is None and cutoff is None:
        return score.copy(), support.copy(), np.zeros(score.shape, dtype=bool)
    if direction not in n207.PROTECTION_DIRECTIONS:
        raise ValueError("NEXT208 protection direction differs")
    if cutoff is None or not math.isfinite(float(cutoff)):
        raise ValueError("NEXT208 cutoff differs")
    finite = np.isfinite(values)
    safe = (
        values >= float(cutoff)
        if direction == "protected_high"
        else values <= float(cutoff)
    )
    active = support & (score >= threshold) & finite & safe
    corrected = score.copy()
    corrected[active] = 0.0
    return corrected, support.copy(), active


def _fraction_numerator(value: float) -> int:
    numerator = int(round(float(value) * EXCEPTION_DENOMINATOR))
    if (
        numerator not in range(1, EXCEPTION_DENOMINATOR)
        or not math.isclose(
            float(value), numerator / EXCEPTION_DENOMINATOR,
            rel_tol=0.0, abs_tol=1.0e-15,
        )
    ):
        raise ValueError("NEXT208 exception fraction differs")
    return numerator


def build_candidate_specs(
    *,
    base_candidate_key: str,
    eligible_hypotheses: Sequence[str],
    features: pd.DataFrame,
    base_score: object,
    base_support: object,
    residual_threshold: float,
    exception_fractions: Sequence[float] = EXCEPTION_FRACTIONS,
) -> list[dict[str, object]]:
    """Build the base plus the exact endpoint-blind feature/fraction grid."""

    if not isinstance(base_candidate_key, str) or not base_candidate_key:
        raise ValueError("NEXT208 base candidate key must be nonempty")
    if not isinstance(features, pd.DataFrame):
        raise ValueError("NEXT208 feature table differs")
    score = np.asarray(base_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    threshold = float(residual_threshold)
    if (
        score.shape != (len(features),)
        or support.shape != score.shape
        or not math.isfinite(threshold)
        or threshold < 0.0
        or np.any(~np.isfinite(score[support]))
    ):
        raise ValueError("NEXT208 cutoff-fit population differs")
    names = tuple(sorted(str(name) for name in eligible_hypotheses))
    if not names or len(set(names)) != len(names):
        raise ValueError("NEXT208 eligible hypothesis universe differs")
    fractions = tuple(float(value) for value in exception_fractions)
    numerators = tuple(_fraction_numerator(value) for value in fractions)
    if not fractions or len(set(numerators)) != len(numerators):
        raise ValueError("NEXT208 exception fraction grid differs")
    fit_mask = support & np.isfinite(score) & (score >= threshold)
    if not fit_mask.any():
        raise ValueError("NEXT208 endpoint-blind cutoff-fit population is empty")

    parsed: list[tuple[str, str, str]] = []
    for hypothesis in names:
        try:
            feature, direction = hypothesis.rsplit("__", 1)
        except ValueError as error:
            raise ValueError("NEXT208 eligible hypothesis identity differs") from error
        if (
            not feature
            or direction not in n207.PROTECTION_DIRECTIONS
            or feature not in features.columns
        ):
            raise ValueError("NEXT208 eligible hypothesis identity differs")
        parsed.append((hypothesis, feature, direction))

    base_payload = {
        "base_candidate_key": base_candidate_key,
        "comparison": None,
        "cutoff": None,
        "direction": None,
        "exception_fraction_denominator": EXCEPTION_DENOMINATOR,
        "exception_fraction_numerator": 0,
        "feature": None,
        "hypothesis": None,
        "missing_policy": "TERM_OFF_KEEP_BASE",
        "residual_threshold": threshold,
        "score_composition": SCORE_COMPOSITION,
    }
    payloads: list[dict[str, object]] = [base_payload]
    for hypothesis, feature, direction in parsed:
        fit_values = pd.to_numeric(
            features.loc[fit_mask, feature], errors="coerce"
        ).to_numpy(float)
        for fraction, numerator in zip(fractions, numerators, strict=True):
            cutoff = empirical_exception_cutoff(
                fit_values, direction, fraction
            )
            payloads.append(
                {
                    "base_candidate_key": base_candidate_key,
                    "comparison": ">=" if direction == "protected_high" else "<=",
                    "cutoff": cutoff,
                    "direction": direction,
                    "exception_fraction_denominator": EXCEPTION_DENOMINATOR,
                    "exception_fraction_numerator": numerator,
                    "feature": feature,
                    "hypothesis": hypothesis,
                    "missing_policy": "TERM_OFF_KEEP_BASE",
                    "residual_threshold": threshold,
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
        raise RuntimeError("NEXT208 candidate keys are not unique")
    return specs


def materialize_residual_x0_exception_candidates(
    *,
    features: pd.DataFrame,
    base_score: object,
    base_support: object,
    specs: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    """Encode every corrected score as one exactly recoverable virtual term."""

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
        raise ValueError("NEXT208 materializer inputs differ")
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
            raise ValueError("NEXT208 materializer feature differs")
        corrected, corrected_support, _ = residual_x0_exception_score(
            base_score=score,
            base_support=support,
            feature_values=feature_values,
            direction=spec.get("direction"),
            cutoff=spec.get("cutoff"),
            residual_threshold=float(spec["residual_threshold"]),
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
        virtual_id = f"next208_virtual_candidate__{digest}"
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
                "group": "next208_residual_x0_exception",
                "encoding": "asinh_sinh_exact_residual_x0_exception_score",
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
    design_path: Path,
) -> dict[str, Path]:
    paths = n207._paths(
        roots,
        freeze_path,
        next202_design_path,
        next205_design_path,
        next207_design_path,
    )
    paths["next207_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next207_manifest": roots["next207"] / n207.MANIFEST_NAME,
            "next207_catalogue": roots["next207"] / n207.CATALOGUE_NAME,
            "next207_audit": roots["next207"] / n207.AUDIT_NAME,
            "next207_table": roots["next207"] / n207.TABLE_NAME,
        }
    )
    return paths


def _verify_next207(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> tuple[str, ...]:
    prior_paths = dict(paths)
    prior_paths["design"] = paths["next207_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next207_design"]
    n207._verify_prior(prior_paths, prior_hashes)
    manifest = json.loads(paths["next207_manifest"].read_text())
    catalogue = json.loads(paths["next207_catalogue"].read_text())
    audit = json.loads(paths["next207_audit"].read_text())
    table = pd.read_parquet(paths["next207_table"])
    eligible = tuple(sorted(str(value) for value in audit.get("eligible_hypotheses", [])))
    eligible_sha = hashlib.sha256("\n".join(eligible).encode()).hexdigest()
    expected_outputs = {
        n207.CATALOGUE_NAME: input_hashes["next207_catalogue"],
        n207.AUDIT_NAME: input_hashes["next207_audit"],
        n207.TABLE_NAME: input_hashes["next207_table"],
    }
    table_eligible = tuple(
        sorted(
            table.loc[
                table["eligible_for_search"].fillna(False).astype(bool),
                "hypothesis",
            ].astype(str)
        )
    )
    if (
        manifest.get("protocol") != n207.PROTOCOL
        or manifest.get("feature_count") != n207.EXPECTED_FEATURE_COUNT
        or manifest.get("hypothesis_count") != n207.EXPECTED_HYPOTHESIS_COUNT
        or manifest.get("eligible_hypothesis_count") != EXPECTED_ELIGIBLE_COUNT
        or manifest.get("next206_residual_candidate_reproduced") is not True
        or manifest.get("next208_search_authorized") is not True
        or manifest.get("new_formula_searched") is not False
        or manifest.get("new_formula_selected") is not False
        or manifest.get("outputs_sha256") != expected_outputs
        or any(manifest.get(key) is not value for key, value in BOUNDARY_FLAGS.items())
        or manifest.get("executed_source_sha256", {}).get(
            "src/next207_residual_x0_feature_audit.py"
        )
        != _sha256_file(Path(n207.__file__).resolve())
        or catalogue.get("protocol") != n207.PROTOCOL
        or catalogue.get("feature_name_sha256") != n207.EXPECTED_FEATURE_NAME_SHA256
        or audit.get("protocol") != n207.PROTOCOL
        or audit.get("eligible_hypothesis_count") != EXPECTED_ELIGIBLE_COUNT
        or audit.get("next208_search_authorized") is not True
        or eligible_sha != EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256
        or table_eligible != eligible
    ):
        raise ValueError("NEXT208 NEXT207 provenance differs")
    return eligible


def _reconstruct_next206(
    *, paths: Mapping[str, Path]
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    str,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    motif_eligible = json.loads(paths["next202_audit"].read_text())[
        "eligible_hypotheses"
    ]
    (
        combined,
        feature_tables,
        _,
        base_score,
        base_support,
        certificates,
        endpoint,
    ) = n205._reconstruct_discovery(
        paths=paths, eligible_names=motif_eligible
    )
    diagnostic = json.loads(paths["next206_diagnostic"].read_text())
    closest = diagnostic.get("global_closest", {})
    candidate_key = str(closest.get("candidate_key", ""))
    threshold = float(closest.get("best_threshold", float("nan")))
    if (
        hashlib.sha256(candidate_key.encode()).hexdigest()
        != n207.EXPECTED_CANDIDATE_KEY_SHA256
        or threshold != n207.EXPECTED_RESIDUAL_THRESHOLD
    ):
        raise ValueError("NEXT208 residual candidate identity differs")
    spec = json.loads(candidate_key)
    certificate_name = str(spec.get("certificate_hypothesis", ""))
    if certificate_name not in certificates:
        raise ValueError("NEXT208 residual certificate identity differs")
    score, support, _ = n205.motif_exception_depth_score(
        base_score=base_score,
        base_support=base_support,
        certificate=certificates[certificate_name],
        certificate_cutoff=float(spec["certificate_cutoff"]),
        pardon_depth=float(spec["pardon_depth"]),
    )
    return combined, feature_tables, candidate_key, score, support, endpoint


def run_residual_x0_exception_search(
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
    next207_dir: Path,
    next135_freeze_path: Path,
    next202_design_path: Path,
    next205_design_path: Path,
    next207_design_path: Path,
    design_path: Path,
    output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT208 raw-x0 exception search."""

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
        "next207": Path(next207_dir).resolve(),
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots,
        Path(next135_freeze_path).resolve(),
        Path(next202_design_path).resolve(),
        Path(next205_design_path).resolve(),
        Path(next207_design_path).resolve(),
        Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(search_workers) is not int or search_workers <= 0:
        raise ValueError("NEXT208 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT208 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT208 formal input identity differs: {differing}")
    eligible = _verify_next207(paths, input_hashes)
    if (
        len(eligible) != EXPECTED_ELIGIBLE_COUNT
        or hashlib.sha256("\n".join(eligible).encode()).hexdigest()
        != EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256
    ):
        raise ValueError("NEXT208 eligible hypothesis universe differs")
    (
        combined,
        feature_tables,
        candidate_key,
        base_score,
        base_support,
        endpoint,
    ) = _reconstruct_next206(paths=paths)

    specs = build_candidate_specs(
        base_candidate_key=candidate_key,
        eligible_hypotheses=eligible,
        features=combined,
        base_score=base_score,
        base_support=base_support,
        residual_threshold=n207.EXPECTED_RESIDUAL_THRESHOLD,
    )
    if len(specs) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("NEXT208 candidate universe differs")
    combined_with_virtual, virtual_terms, runtime = (
        materialize_residual_x0_exception_candidates(
            features=combined,
            base_score=base_score,
            base_support=base_support,
            specs=specs,
        )
    )
    started = time.perf_counter()
    result = n205.n203.n202.n200.n194.n130.n125.search_optional_guard_laws_parallel(
        features=combined_with_virtual,
        endpoint=endpoint,
        old_terms=virtual_terms,
        optional_terms=[],
        candidate_specs=runtime,
        workers=search_workers,
    )
    elapsed = time.perf_counter() - started
    if int(result["candidate_count"]) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("NEXT208 evaluator count differs")

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
        _, _, active = residual_x0_exception_score(
            base_score=base_score,
            base_support=base_support,
            feature_values=values,
            direction=spec["direction"],
            cutoff=spec["cutoff"],
            residual_threshold=float(spec["residual_threshold"]),
        )
        record.update(
            {
                "base_candidate_key": candidate_key,
                "comparison": spec["comparison"],
                "cutoff": spec["cutoff"],
                "direction": spec["direction"],
                "exception_active_rows": int(active.sum()),
                "exception_active_scigen": int((active & (source == "scigen")).sum()),
                "exception_active_wyformer": int((active & (source == "wyformer")).sum()),
                "exception_fraction_denominator": spec[
                    "exception_fraction_denominator"
                ],
                "exception_fraction_numerator": spec[
                    "exception_fraction_numerator"
                ],
                "feature": feature,
                "hypothesis": spec["hypothesis"],
                "missing_policy": spec["missing_policy"],
                "residual_threshold": spec["residual_threshold"],
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
    next209_keys = sorted(records.loc[auc_safe_non_broad, "candidate_key"].astype(str))
    next209_key_sha = hashlib.sha256("\n".join(next209_keys).encode()).hexdigest()
    selected_spec = spec_by_key[str(selected["record"]["candidate_key"])]
    base_spec = json.loads(candidate_key)
    base_core_formula = json.loads(paths["next163_evaluation"].read_text())[
        "selected_formula"
    ]
    base_formula = {
        "kind": "motif_conjunction_protected_exception_depth_no_dft_score",
        "core_formula": base_core_formula,
        "candidate_key": candidate_key,
        "certificate_hypothesis": base_spec["certificate_hypothesis"],
        "certificate_cutoff": base_spec["certificate_cutoff"],
        "conjunction": base_spec["conjunction"],
        "floor_threshold": base_spec["floor_threshold"],
        "pardon_depth": base_spec["pardon_depth"],
        "secondary_feature": base_spec["secondary_feature"],
        "score_composition": base_spec["score_composition"],
    }
    formula = {
        "protocol": PROTOCOL,
        "kind": "residual_raw_x0_protected_exception_no_dft_score",
        "base_candidate_key": candidate_key,
        "base_candidate_key_sha256": n207.EXPECTED_CANDIDATE_KEY_SHA256,
        "base_formula": base_formula,
        "feature": selected_spec["feature"],
        "direction": selected_spec["direction"],
        "comparison": selected_spec["comparison"],
        "cutoff": selected_spec["cutoff"],
        "exception_fraction_numerator": selected_spec[
            "exception_fraction_numerator"
        ],
        "exception_fraction_denominator": selected_spec[
            "exception_fraction_denominator"
        ],
        "residual_threshold": n207.EXPECTED_RESIDUAL_THRESHOLD,
        "missing_policy": "TERM_OFF_KEEP_BASE",
        "support_policy": "UNCHANGED_FROM_NEXT206",
        "score_composition": SCORE_COMPOSITION,
        **{
            key: value
            for key, value in BOUNDARY_FLAGS.items()
            if key
            in {
                "dft_values_used_by_executable_formula",
                "learned_energy_force_stress_proxy_used",
                "model_or_proxy_potential_used",
                "physical_relaxation_executed",
            }
        },
    }
    passes = bool(selected["record"]["passes_all_discovery_gates"])
    counts = {
        "passes_source_auc_gates": int(
            records["passes_source_auc_gates"].fillna(False).astype(bool).sum()
        ),
        "passes_safe_all_cells": int(
            records["passes_safe_all_cells"].fillna(False).astype(bool).sum()
        ),
        "passes_broad_all_cells": int(
            records["passes_broad_all_cells"].fillna(False).astype(bool).sum()
        ),
        "passes_all_discovery_gates": int(all_gate.sum()),
        "passes_auc_and_safe_but_not_broad": int(auc_safe_non_broad.sum()),
    }
    catalogue = {
        "protocol": PROTOCOL,
        "design_sha256": input_hashes["design"],
        "base_candidate_key_sha256": n207.EXPECTED_CANDIDATE_KEY_SHA256,
        "residual_threshold": n207.EXPECTED_RESIDUAL_THRESHOLD,
        "eligible_hypotheses": list(eligible),
        "eligible_hypothesis_count": len(eligible),
        "eligible_hypothesis_sha256": EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256,
        "cutoff_fit_mask": "support_and_finite_score_and_score_ge_residual_threshold",
        "cutoff_fit_uses_endpoint": False,
        "quantile_method": "inverted_cdf",
        "exception_fraction_denominator": EXCEPTION_DENOMINATOR,
        "exception_fraction_numerators": list(range(1, EXCEPTION_DENOMINATOR)),
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "score_composition": SCORE_COMPOSITION,
        "base_support_unchanged": True,
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    evaluation = {
        "protocol": PROTOCOL,
        "evaluation_mode": "fixed_residual_raw_x0_exception_search",
        "next206_residual_candidate_reproduced": True,
        "next207_eligible_hypotheses_reproduced": True,
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
        "next209_diagnostic_authorized": bool(not passes and next209_keys),
        "next209_candidate_count": len(next209_keys),
        "next209_candidate_key_sha256": next209_key_sha,
        "requires_unopened_internal_validation_before_claim": True,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    source_paths = {
        "src/next205_motif_exception_depth_search.py": Path(n205.__file__).resolve(),
        "src/next206_motif_exception_depth_broad_residual.py": Path(
            n206.__file__
        ).resolve(),
        "src/next207_residual_x0_feature_audit.py": Path(n207.__file__).resolve(),
        "src/next208_residual_x0_exception_search.py": Path(__file__).resolve(),
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
            "search_workers": search_workers,
            "next206_residual_candidate_reproduced": True,
            "next207_eligible_hypotheses_reproduced": True,
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "next209_diagnostic_authorized": bool(not passes and next209_keys),
            "next209_candidate_count": len(next209_keys),
            "next209_candidate_key_sha256": next209_key_sha,
            "requires_unopened_internal_validation_before_claim": True,
            "residual_x0_exception_search_branch_terminated": bool(
                not passes and not next209_keys
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
            raise RuntimeError("NEXT208 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT208 source changed before publication")
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
    for stage in (194, 199, 200, 201, 202, 203, 204, 205, 206, 207):
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--next202-design-path", type=Path, required=True)
    parser.add_argument("--next205-design-path", type=Path, required=True)
    parser.add_argument("--next207-design-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_residual_x0_exception_search(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{f"next{stage}_dir": getattr(args, f"next{stage}_dir") for stage in stages},
        **{
            f"next{stage}_dir": getattr(args, f"next{stage}_dir")
            for stage in (194, 199, 200, 201, 202, 203, 204, 205, 206, 207)
        },
        next135_freeze_path=args.next135_freeze_path,
        next202_design_path=args.next202_design_path,
        next205_design_path=args.next205_design_path,
        next207_design_path=args.next207_design_path,
        design_path=args.design_path,
        output_dir=args.output_dir,
        search_workers=args.search_workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "BOUNDARY_FLAGS",
    "EXCEPTION_FRACTIONS",
    "EXPECTED_CANDIDATE_COUNT",
    "EXPECTED_ELIGIBLE_COUNT",
    "build_candidate_specs",
    "empirical_exception_cutoff",
    "materialize_residual_x0_exception_candidates",
    "residual_x0_exception_score",
    "run_residual_x0_exception_search",
]


if __name__ == "__main__":
    raise SystemExit(main())
