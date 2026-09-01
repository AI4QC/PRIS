#!/usr/bin/env python3
"""Finite no-DFT search over local directional-rigidity attenuation."""

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

import src.next130_coordination_protection_search as n130
import src.next135_conjunctive_compactness_search as n135
import src.next163_interior_family_attenuation_search as n163
import src.next164_interior_attenuation_broad_residual as n164
import src.next168_periodic_local_directional_rigidity as n168
import src.next169_periodic_local_directional_rigidity_audit as n169
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next87_scigen_sparse_law_search import _term_risk


PROTOCOL = "2026-08-08-next170-local-directional-rigidity-attenuation-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT170_LOCAL_DIRECTIONAL_RIGIDITY_ATTENUATION_CATALOGUE.json"
EVALUATION_NAME = "NEXT170_DISCOVERY_EVALUATION.json"
FORMULA_NAME = "NEXT170_FROZEN_CANDIDATE.json"
SEARCH_NAME = "next170_local_directional_rigidity_attenuation_search.parquet"
EXPECTED_DESIGN_SHA256 = "2e8f91013caafefefc61af8058a29a8513b13b2ca6e66a439209a6261b08803f"
EXPECTED_BASE_CANDIDATE_KEY_SHA256 = "1d0ea8331f38aa69cfdedbe664d5ceb46c14e166e121bae92d9e14dd4fc6109e"
ELIGIBLE_FEATURES = (
    "pldr_crystalnn_tightness_min",
    "pldr_crystalnn_tightness_q10",
    "pldr_crystalnn_tightness_mean",
    "pldr_crystalnn_volume_q10",
    "pldr_crystalnn_volume_mean",
)
ATTENUATIONS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40)
EXPECTED_CANDIDATE_COUNT = 1 + len(ELIGIBLE_FEATURES) * len(ATTENUATIONS)
SCORE_COMPOSITION = "max(0,base_score*(1-alpha*local_directional_rigidity))"
SEARCH_WORKERS = 4
EXPECTED_INPUT_SHA256 = {
    **n169.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next169_manifest": "13249710d94e1950ffed4b84a40eca7f754bd292bb794275095d81cf1bbab643",
    "next169_audit": "4135933b256e6820e27093446f4afc6e83a7cb341670add8ef6b01fbaf74920a",
    "next169_table": "af58c9c5d59d209baa2727bb3f39ec3f36a50c1498c8410ef07172354bee53a0",
}


def attenuate_score(
    *,
    base_score: object,
    base_support: object,
    feature: object,
    attenuation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one bounded attenuation with missing-feature keep-base policy."""

    score = np.asarray(base_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    rigidity = np.asarray(feature, dtype=float)
    alpha = float(attenuation)
    if (
        score.ndim != 1
        or support.shape != score.shape
        or rigidity.shape != score.shape
        or alpha not in {0.0, *ATTENUATIONS}
        or np.any(~np.isfinite(score[support]))
        or np.any(score[support] < -1.0e-12)
    ):
        raise ValueError("NEXT170 attenuation input differs")
    finite = np.isfinite(rigidity)
    if np.any((rigidity[finite] < -1.0e-12) | (rigidity[finite] > 1.0 + 1.0e-12)):
        raise ValueError("NEXT170 rigidity feature is outside [0,1]")
    bounded = np.clip(rigidity[finite], 0.0, 1.0)
    corrected = score.copy()
    corrected[finite] = np.maximum(
        0.0, score[finite] * (1.0 - alpha * bounded)
    )
    return corrected, support.copy()


def build_candidate_specs(*, base_candidate_key: str) -> list[dict[str, object]]:
    """Build the exact 41-member preregistered candidate universe."""

    if not isinstance(base_candidate_key, str) or not base_candidate_key:
        raise ValueError("NEXT170 base candidate key must be nonempty")
    pairs: list[tuple[str | None, float]] = [(None, 0.0)]
    pairs.extend(
        (feature, alpha)
        for feature in ELIGIBLE_FEATURES
        for alpha in ATTENUATIONS
    )
    specs = []
    for feature, alpha in pairs:
        payload = {
            "attenuation": alpha,
            "base_candidate_key": base_candidate_key,
            "directional_rigidity_feature": feature,
            "missing_policy": "TERM_OFF_KEEP_BASE",
            "score_composition": SCORE_COMPOSITION,
        }
        specs.append(
            {
                **payload,
                "candidate_key": json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                ),
            }
        )
    if len(specs) != EXPECTED_CANDIDATE_COUNT or len(
        {str(spec["candidate_key"]) for spec in specs}
    ) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("NEXT170 candidate universe differs")
    return specs


def materialize_attenuation_candidates(
    *,
    features: pd.DataFrame,
    base_score: object,
    base_support: object,
    specs: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    """Encode every corrected score as one evaluator-compatible virtual term."""

    score = np.asarray(base_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    if score.shape != (len(features),) or support.shape != score.shape:
        raise ValueError("NEXT170 base score shape differs")
    if set(ELIGIBLE_FEATURES) - set(features.columns):
        raise ValueError("NEXT170 eligible feature schema differs")
    columns: dict[str, np.ndarray] = {}
    terms: list[dict[str, object]] = []
    runtime: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_spec in specs:
        spec = dict(raw_spec)
        key = str(spec.get("candidate_key", ""))
        feature_name = spec.get("directional_rigidity_feature")
        alpha = float(spec.get("attenuation", np.nan))
        if (
            not key
            or key in seen
            or feature_name not in {None, *ELIGIBLE_FEATURES}
            or (feature_name is None and alpha != 0.0)
            or (feature_name is not None and alpha not in ATTENUATIONS)
        ):
            raise ValueError("NEXT170 candidate spec differs")
        seen.add(key)
        feature = (
            np.full(len(features), np.nan, dtype=float)
            if feature_name is None
            else pd.to_numeric(features[str(feature_name)], errors="coerce").to_numpy(float)
        )
        corrected, corrected_support = attenuate_score(
            base_score=score,
            base_support=support,
            feature=feature,
            attenuation=alpha,
        )
        maximum = float(np.max(corrected[corrected_support])) if corrected_support.any() else 0.0
        divisor = max(1.0, maximum / 50.0)
        encoded = np.full(len(features), np.nan, dtype=float)
        encoded[corrected_support] = np.sinh(corrected[corrected_support] / divisor)
        if not np.isfinite(encoded[corrected_support]).all():
            raise ValueError("NEXT170 virtual score encoding overflowed")
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        virtual_id = f"next170_virtual_candidate__{digest}"
        value_column = f"_{virtual_id}_value"
        columns[value_column] = encoded
        terms.append(
            {
                "term_id": virtual_id,
                "feature": value_column,
                "direction": 1,
                "transform": "asinh",
                "center": 0.0,
                "scale": 1.0 / divisor,
                "group": "next170_local_directional_rigidity_attenuation",
                "encoding": "asinh_sinh_exact_attenuated_score",
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
    if len(seen) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("NEXT170 materialized candidate count differs")
    return (
        pd.concat([features.reset_index(drop=True), pd.DataFrame(columns)], axis=1),
        terms,
        runtime,
    )


def _paths(
    roots: Mapping[str, Path], freeze_path: Path, design_path: Path
) -> dict[str, Path]:
    paths = n169._paths(roots, freeze_path, design_path)
    paths.update(
        {
            "next169_manifest": roots["next169"] / n169.MANIFEST_NAME,
            "next169_audit": roots["next169"] / n169.AUDIT_NAME,
            "next169_table": roots["next169"] / n169.TABLE_NAME,
        }
    )
    return paths


def _verify_base_reproduction(
    *, record: Mapping[str, object], published: pd.DataFrame, candidate_key: str
) -> None:
    matches = published.loc[published["candidate_key"].astype(str) == candidate_key]
    if len(matches) != 1:
        raise RuntimeError("NEXT170 published base identity differs")
    row = matches.iloc[0]
    metrics = (
        "scigen_pooled_auc",
        "scigen_macro_auc",
        "scigen_worst_auc",
        "wyformer_pooled_auc",
        "wyformer_macro_auc",
        "wyformer_worst_auc",
        "safe_threshold",
    )
    booleans = (
        "passes_source_auc_gates",
        "passes_safe_all_cells",
        "passes_broad_all_cells",
        "passes_all_discovery_gates",
    )
    for metric in metrics:
        actual = float("nan") if record.get(metric) is None else float(record[metric])
        expected = float("nan") if pd.isna(row[metric]) else float(row[metric])
        if not (
            (math.isnan(actual) and math.isnan(expected))
            or math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
        ):
            raise RuntimeError(f"NEXT170 base reproduction differs: {metric}")
    if any(bool(record[name]) != bool(row[name]) for name in booleans):
        raise RuntimeError("NEXT170 base gate reproduction differs")


def run_local_directional_rigidity_attenuation_search(
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
    next169_dir: Path,
    next135_freeze_path: Path,
    design_path: Path,
    output_dir: Path,
    search_workers: int = 4,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the formal frozen NEXT170 discovery search."""

    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{
            f"next{stage}": Path(value).resolve()
            for stage, value in (
                (98, next98_dir),
                (110, next110_dir),
                (111, next111_dir),
                (113, next113_dir),
                (114, next114_dir),
                (116, next116_dir),
                (117, next117_dir),
                (120, next120_dir),
                (121, next121_dir),
                (122, next122_dir),
                (124, next124_dir),
                (125, next125_dir),
                (129, next129_dir),
                (130, next130_dir),
                (133, next133_dir),
                (134, next134_dir),
                (163, next163_dir),
                (164, next164_dir),
                (168, next168_dir),
                (169, next169_dir),
            )
        },
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots, Path(next135_freeze_path).resolve(), Path(design_path).resolve()
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(search_workers) is not int or search_workers <= 0:
        raise ValueError("search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT170 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            key
            for key in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(key) != EXPECTED_INPUT_SHA256.get(key)
        )
        raise ValueError(f"NEXT170 formal input identity differs: {differing}")

    manifest169 = json.loads(paths["next169_manifest"].read_text())
    audit169 = json.loads(paths["next169_audit"].read_text())
    table169 = pd.read_parquet(paths["next169_table"])
    eligible_from_table = tuple(
        sorted(
            table169.loc[
                table169["eligible_for_search"].fillna(False).astype(bool), "feature"
            ].astype(str)
        )
    )
    if (
        manifest169.get("protocol") != n169.PROTOCOL
        or manifest169.get("eligible_hypothesis_count") != len(ELIGIBLE_FEATURES)
        or manifest169.get("periodic_local_directional_rigidity_branch_terminated")
        is not False
        or manifest169.get("new_formula_searched") is not False
        or manifest169.get("opened_validation_outputs_used") is not False
        or manifest169.get("scigen_replication_endpoint_opened") is not False
        or manifest169.get("wyformer_replication_endpoint_opened") is not False
        or manifest169.get("dft_calculation_executed") is not False
        or manifest169.get("dft_values_used_by_executable_formula") is not False
        or manifest169.get("learned_energy_force_stress_proxy_used") is not False
        or manifest169.get("physical_relaxation_executed") is not False
        or manifest169.get("outputs_sha256", {}).get(n169.AUDIT_NAME)
        != input_hashes["next169_audit"]
        or manifest169.get("outputs_sha256", {}).get(n169.TABLE_NAME)
        != input_hashes["next169_table"]
        or manifest169.get("executed_source_sha256", {}).get(
            "src/next169_periodic_local_directional_rigidity_audit.py"
        )
        != _sha256_file(Path(n169.__file__).resolve())
        or tuple(sorted(ELIGIBLE_FEATURES)) != eligible_from_table
        or tuple(sorted(audit169.get("eligible_hypotheses", ())))
        != tuple(
            sorted(
                f"{feature.removeprefix('pldr_')}__high"
                for feature in ELIGIBLE_FEATURES
            )
        )
    ):
        raise ValueError("NEXT170 eligibility provenance differs")

    diagnostic164 = json.loads(paths["next164_diagnostic"].read_text())
    closest = diagnostic164.get("global_closest", {})
    base_candidate_key = str(closest.get("candidate_key", ""))
    if (
        hashlib.sha256(base_candidate_key.encode()).hexdigest()
        != EXPECTED_BASE_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT170 base candidate identity differs")

    extended, feature_tables, old_terms, mhcr_terms = n130._join_label_free_features(paths)
    compact_frames = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next133_{source}_features"]).copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        compact_frames.append(table)
    extended = extended.merge(
        pd.concat(compact_frames, ignore_index=True),
        on="material_id",
        how="inner",
        validate="one_to_one",
    )
    conjunctive = n135.materialize_conjunctive_features(extended)
    extended = pd.concat(
        [extended.reset_index(drop=True), conjunctive.reset_index(drop=True)], axis=1
    )
    rigidity_frames = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next168_{source}_features"]).copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        rigidity_frames.append(table)
    combined = extended.merge(
        pd.concat(rigidity_frames, ignore_index=True),
        on="material_id",
        how="inner",
        validate="one_to_one",
    )

    physical_terms = [*old_terms, *mhcr_terms]
    physical_ids = {str(term["term_id"]) for term in physical_terms}
    all_bases = n130.n127.select_next125_bases(
        pd.read_parquet(paths["next125_search_records"])
    )
    bases = n135.n132.select_extended_bases(
        pd.read_parquet(paths["next130_search_records"]), all_bases
    )
    base_specs = n163.build_candidate_specs(
        bases=bases, physical_term_ids=physical_ids
    )
    selected_base_specs = [
        spec for spec in base_specs if str(spec["candidate_key"]) == base_candidate_key
    ]
    if len(selected_base_specs) != 1:
        raise ValueError("NEXT170 base reconstruction differs")
    combined, base_terms, base_runtime = n163.materialize_candidates(
        features=combined,
        physical_terms=physical_terms,
        specs=selected_base_specs,
    )
    if len(base_terms) != 1 or len(base_runtime) != 1:
        raise RuntimeError("NEXT170 base materialization differs")
    base_score, base_support = _term_risk(combined, base_terms[0])

    scigen_endpoint = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_endpoint = pd.read_parquet(paths["wyformer_endpoint"])
    endpoint_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "material_id": "scigen:" + scigen_endpoint["material_id"].astype(str),
                    "_endpoint": pd.to_numeric(
                        scigen_endpoint["distortion_ratio"], errors="coerce"
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "material_id": "wyformer:" + wyformer_endpoint["material_id"].astype(str),
                    "_endpoint": n130.n125.n121.prior._endpoint_numeric(
                        wyformer_endpoint["endpoint_stratum"]
                    ),
                }
            ),
        ],
        ignore_index=True,
    )
    combined = combined.merge(
        endpoint_frame, on="material_id", how="inner", validate="one_to_one"
    )
    endpoint = pd.to_numeric(combined.pop("_endpoint"), errors="coerce").to_numpy(float)
    if not np.isfinite(endpoint).all() or len(combined) != len(base_score):
        raise ValueError("NEXT170 endpoint row accounting differs")

    specs = build_candidate_specs(base_candidate_key=base_candidate_key)
    combined, virtual_terms, runtime = materialize_attenuation_candidates(
        features=combined,
        base_score=base_score,
        base_support=base_support,
        specs=specs,
    )
    started = time.perf_counter()
    result = n130.n125.search_optional_guard_laws_parallel(
        features=combined,
        endpoint=endpoint,
        old_terms=virtual_terms,
        optional_terms=[],
        candidate_specs=runtime,
        workers=search_workers,
    )
    elapsed = time.perf_counter() - started
    if int(result["candidate_count"]) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("NEXT170 evaluator candidate count differs")

    spec_by_key = {str(spec["candidate_key"]): spec for spec in specs}

    def decorate(record: dict[str, object]) -> None:
        spec = spec_by_key[str(record["candidate_key"])]
        record["base_candidate_key"] = base_candidate_key
        record["directional_rigidity_feature"] = spec[
            "directional_rigidity_feature"
        ]
        record["attenuation"] = float(spec["attenuation"])
        record["missing_policy"] = "TERM_OFF_KEEP_BASE"
        record["score_composition"] = SCORE_COMPOSITION

    for record in result["candidate_records"]:
        decorate(record)
    selected = result["selected"]
    if "attenuation" not in selected["record"]:
        decorate(selected["record"])
    base_records = [
        record
        for record in result["candidate_records"]
        if record["directional_rigidity_feature"] is None
    ]
    if len(base_records) != 1:
        raise RuntimeError("NEXT170 unmodified base count differs")
    _verify_base_reproduction(
        record=base_records[0],
        published=pd.read_parquet(paths["next163_search"]),
        candidate_key=base_candidate_key,
    )

    selected_spec = spec_by_key[str(selected["record"]["candidate_key"])]
    prior_evaluation = json.loads(paths["next163_evaluation"].read_text())
    frozen_formula = {
        "protocol": PROTOCOL,
        "kind": "local_directional_rigidity_attenuated_no_dft_score",
        "base_candidate_key": base_candidate_key,
        "base_formula": prior_evaluation["selected_formula"],
        "directional_rigidity_feature": selected_spec[
            "directional_rigidity_feature"
        ],
        "attenuation": float(selected_spec["attenuation"]),
        "feature_definition": "site_direction_gram_bounded_certificate",
        "missing_policy": "TERM_OFF_KEEP_BASE",
        "score_composition": SCORE_COMPOSITION,
        "dft_values_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "physical_relaxation_executed": False,
    }
    passes = bool(selected["record"]["passes_all_discovery_gates"])
    records_frame = pd.DataFrame(result["candidate_records"])
    counts = {}
    for (feature, alpha), frame in records_frame.assign(
        directional_rigidity_feature=records_frame[
            "directional_rigidity_feature"
        ].fillna("BASE")
    ).groupby(["directional_rigidity_feature", "attenuation"], sort=True):
        counts[f"feature={feature},alpha={float(alpha):g}"] = {
            "candidates": int(len(frame)),
            "passes_source_auc_gates": int(frame["passes_source_auc_gates"].sum()),
            "passes_safe_all_cells": int(frame["passes_safe_all_cells"].sum()),
            "passes_broad_all_cells": int(frame["passes_broad_all_cells"].sum()),
            "passes_all_discovery_gates": int(
                frame["passes_all_discovery_gates"].sum()
            ),
        }
    catalogue = {
        "protocol": PROTOCOL,
        "design_sha256": input_hashes["design"],
        "base_candidate_key_sha256": EXPECTED_BASE_CANDIDATE_KEY_SHA256,
        "base_endpoint_reproduced": True,
        "eligible_features": ELIGIBLE_FEATURES,
        "attenuation_grid": ATTENUATIONS,
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "score_composition": SCORE_COMPOSITION,
        "missing_policy": "TERM_OFF_KEEP_BASE",
        "base_support_unchanged": True,
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next163_interior_family_attenuation_search.py": Path(n163.__file__).resolve(),
        "src/next168_periodic_local_directional_rigidity.py": Path(n168.__file__).resolve(),
        "src/next169_periodic_local_directional_rigidity_audit.py": Path(n169.__file__).resolve(),
        "src/next170_local_directional_rigidity_attenuation_search.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    try:
        catalogue_path = staging / CATALOGUE_NAME
        evaluation_path = staging / EVALUATION_NAME
        formula_path = staging / FORMULA_NAME
        search_path = staging / SEARCH_NAME
        _write_json(catalogue_path, catalogue)
        _write_json(
            evaluation_path,
            {
                "protocol": PROTOCOL,
                "evaluation_mode": "fixed_local_directional_rigidity_attenuation_search",
                "base_endpoint_reproduced": True,
                "rows": {
                    "scigen": int(len(feature_tables["scigen"])),
                    "wyformer": int(len(feature_tables["wyformer"])),
                    "total": int(len(combined)),
                },
                "candidate_count": int(result["candidate_count"]),
                "elapsed_seconds": elapsed,
                "search_workers": search_workers,
                "counts_by_feature_and_attenuation": counts,
                "safe_gates": dict(n130.n125.n121.prior.DEFAULT_GATES),
                "source_auc_gates": dict(n130.n125.n121.prior.AUC_GATES),
                "broad_min_severe_precision_lower": n130.n125.n121.prior.BROAD_MIN_PRECISION_LOWER,
                "selected_record": selected["record"],
                "selected_formula": frozen_formula,
                "selected_safe": selected["safe"],
                "selected_safe_diagnostic": selected["safe_diagnostic"],
                "selected_broad": selected["broad"],
                "selected_source_diagnostics": selected["source_diagnostics"],
                "pauling_by_cell": result["pauling_by_cell"],
                "cells": result["cells"],
                "passes_all_cross_source_discovery_gates": passes,
                "freeze_authorized": passes,
                "requires_unopened_internal_validation_before_claim": True,
            },
        )
        _write_json(formula_path, frozen_formula)
        records_frame.to_parquet(search_path, index=False)
        output_paths = [catalogue_path, evaluation_path, formula_path, search_path]
        manifest = {
            "protocol": PROTOCOL,
            "candidate_count": int(result["candidate_count"]),
            "search_workers": search_workers,
            "base_endpoint_reproduced": True,
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "requires_unopened_internal_validation_before_claim": True,
            "local_directional_rigidity_attenuation_branch_terminated": not passes,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True,
            "opened_validation_outputs_used": False,
            "scigen_replication_endpoint_opened": False,
            "wyformer_replication_endpoint_opened": False,
            "formula_or_threshold_changed_after_search": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "physical_relaxation_executed": False,
            "scientific_improvement_claim": False,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {
                path.name: _sha256_file(path) for path in output_paths
            },
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT170 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name] for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT170 source changed before publication")
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
        98,
        110,
        111,
        113,
        114,
        116,
        117,
        120,
        121,
        122,
        124,
        125,
        129,
        130,
        133,
        134,
        163,
        164,
        168,
        169,
    )
    for stage in stages:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_local_directional_rigidity_attenuation_search(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{f"next{stage}_dir": getattr(args, f"next{stage}_dir") for stage in stages},
        next135_freeze_path=args.next135_freeze_path,
        design_path=args.design_path,
        output_dir=args.output_dir,
        search_workers=args.search_workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ATTENUATIONS",
    "ELIGIBLE_FEATURES",
    "EXPECTED_CANDIDATE_COUNT",
    "attenuate_score",
    "build_candidate_specs",
    "materialize_attenuation_candidates",
    "run_local_directional_rigidity_attenuation_search",
]


if __name__ == "__main__":
    raise SystemExit(main())
