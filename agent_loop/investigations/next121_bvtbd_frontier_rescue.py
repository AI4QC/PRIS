#!/usr/bin/env python3
"""Frozen BVTBD rescue of the NEXT117 cross-source discovery frontier."""

from __future__ import annotations

import argparse
import hashlib
import itertools
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

import src.next117_hcid_frontier_rescue as prior
from src.next103_dobvr_optional_guard_search import (
    _optional_term_risk,
    search_optional_guard_laws,
)
from src.next120_cross_source_bvtbd_features import (
    CATALOGUE_NAME as NEXT120_CATALOGUE_NAME,
    FEATURE_COLUMNS as NEXT120_FEATURE_COLUMNS,
    FEATURE_FILES as NEXT120_FEATURE_FILES,
    MANIFEST_NAME as NEXT120_MANIFEST_NAME,
    PROTOCOL as NEXT120_PROTOCOL,
)


PROTOCOL = "2026-08-08-next121-bvtbd-frontier-rescue-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT121_BVTBD_FRONTIER_CATALOGUE.json"
EVALUATION_NAME = "NEXT121_DISCOVERY_EVALUATION.json"
SEARCH_NAME = "next121_bvtbd_frontier_candidate_search.parquet"
EXPECTED_FREEZE_SHA256 = (
    "b2d85550b09a2785f59890ef3ab957fd4974c06fa7be76c12d01242a404dcec8"
)
SINGLE_WEIGHT_GRID = (0.10, 0.25, 0.50, 1.00, 2.00)
PAIR_WEIGHT_GRID = (0.10, 0.25, 0.50, 1.00)
FRONTIER_AUC_SAFE_CELLS = 11
BASE_REPRODUCTION_AUC_TOLERANCE = prior.BASE_REPRODUCTION_AUC_TOLERANCE
EXPECTED_FRONTIER_BASES = 507
EXPECTED_AUC_FRONTIER_BASES = 44
EXPECTED_SAFE_FRONTIER_BASES = 463
EXPECTED_CONFIGURATION_COUNT = 116
EXPECTED_CANDIDATE_COUNT = 59_319
EXPECTED_CANDIDATE_KEY_SHA256 = (
    "cc5c201ef12f7c76bc169ce7a1cbebde6bfcab34a103bfb9a6a0e3e55d77f7aa"
)
EXPECTED_BASE_CANDIDATE_KEY_SHA256 = (
    "b9482783f66225d2f4f93758badb1ab525e7406968e0d08a269f561db71bd6fd"
)
EXPECTED_BASE_FORMULA_SHA256 = (
    "b2a9a42e2c3e38cc88940b7396f67d829040d0b16b97f31b4852b96b6d3d4883"
)
NUMERICAL_ZERO_TOLERANCE = 1.0e-12
PHYSICAL_BUDGET = 0.10

FROZEN_TERM_SPECS = (
    {
        "term_id": "bvtbd_required_linf_budget_decades__high",
        "raw_key": "required_budget_decades",
        "group": "bvtbd_core",
        "center": 0.0,
        "scale": 1.0,
        "raw_p995": 7.708746100480892,
        "clip_normalized": 7.708746100480892,
        "nonzero_rows": 10_431,
    },
    {
        "term_id": "bvtbd_deformation_debt_tau10__high",
        "raw_key": "tau10_debt_excess",
        "group": "bvtbd_core",
        "center": 0.0,
        "scale": 1.0,
        "raw_p995": 0.9999999363235441,
        "clip_normalized": 0.9999999363235441,
        "nonzero_rows": 10_290,
    },
    {
        "term_id": "bvtbd_coordinate_localization__high",
        "raw_key": "coordinate_localization_excess",
        "group": "bvtbd_core",
        "center": 0.0,
        "scale": 1.0,
        "raw_p995": 0.8832898728495564,
        "clip_normalized": 0.8832898728495564,
        "nonzero_rows": 4_521,
    },
    {
        "term_id": "bvtbd_cell_strain_budget_decades__high",
        "raw_key": "cell_strain_budget_decades",
        "group": "bvtbd_core",
        "center": 0.0,
        "scale": 1.0,
        "raw_p995": 6.370519077999772,
        "clip_normalized": 6.370519077999772,
        "nonzero_rows": 10_376,
    },
)
EXCLUDED_FEATURES = (
    "bvtbd_unbounded_residual_fraction_due_existing_bvtc_overlap",
    "bvtbd_tau01_and_tau03_due_nested_budget_collinearity",
    "bvtbd_atomic_motion_max_due_required_budget_overlap",
    "unsupported_status_due_missingness_not_risk",
)

EXPECTED_INPUT_SHA256 = {
    **{key: value for key, value in prior.EXPECTED_INPUT_SHA256.items() if key != "freeze"},
    "next117_manifest": "02638dbf07b5b471e23c786e4951426bf22d1d5b28e83908a6738db58b5c2f3f",
    "next117_catalogue": "728e176f900274e4f7be20822ac363fe4fb7acca357ca5b747b21ab19184a6e7",
    "next117_search_records": "b40619d2bcdf555c5dfcb477683d0a66fe52994c8a6c3473021f0ceb7516818b",
    "next120_manifest": "119b342e1fe603f55b0c7a6aedee48eb61733749558a1590f34e9a37f35ab684",
    "next120_catalogue": "3a72546fbfe8954498a24acd6048dc8e3d909020248c332dcb97b02cf2ec908b",
    "next120_scigen_features": "1f8e9e17981b114d4c7dd881e0ad7aaff2c251aca63ad3b6965a2879d94e8249",
    "next120_wyformer_features": "dafa685d47d02a1ed6c20c1ec3ecc4e54aaa571119b104365032705dd4974892",
    "freeze": EXPECTED_FREEZE_SHA256,
}


def _derived_bvtbd_raw(features: pd.DataFrame) -> dict[str, np.ndarray]:
    required_columns = {
        "natoms",
        "bvtbd_supported",
        "bvtbd_required_linf_budget",
        "bvtbd_minimum_motion_rms",
        "bvtbd_cell_strain_frobenius",
        "bvtbd_deformation_debt_tau10",
    }
    if required_columns - set(features.columns):
        raise ValueError("NEXT121 BVTBD derivation columns differ")
    required = pd.to_numeric(
        features["bvtbd_required_linf_budget"], errors="coerce"
    ).to_numpy(float)
    rms = pd.to_numeric(
        features["bvtbd_minimum_motion_rms"], errors="coerce"
    ).to_numpy(float)
    cell = pd.to_numeric(
        features["bvtbd_cell_strain_frobenius"], errors="coerce"
    ).to_numpy(float)
    debt = pd.to_numeric(
        features["bvtbd_deformation_debt_tau10"], errors="coerce"
    ).to_numpy(float)
    natoms = pd.to_numeric(features["natoms"], errors="coerce").to_numpy(float)
    supported = features["bvtbd_supported"].fillna(False).astype(bool).to_numpy()
    if np.any(supported & (~np.isfinite(natoms) | (natoms < 2.0))):
        raise ValueError("NEXT121 site count differs")

    required_decades = np.maximum(
        0.0,
        np.log10(np.maximum(required / PHYSICAL_BUDGET, 1.0)),
    )
    debt_excess = np.maximum(0.0, (debt - 0.50) / 0.50)
    solution_norm = rms * np.sqrt(3.0 * natoms + 6.0)
    localization = np.divide(
        required,
        solution_norm,
        out=np.zeros_like(required),
        where=(
            np.isfinite(required)
            & np.isfinite(solution_norm)
            & (solution_norm > NUMERICAL_ZERO_TOLERANCE)
        ),
    )
    inconsistent = (
        np.isfinite(required)
        & (required > NUMERICAL_ZERO_TOLERANCE)
        & (~np.isfinite(solution_norm) | (solution_norm <= NUMERICAL_ZERO_TOLERANCE))
    )
    if (supported & inconsistent).any():
        raise ValueError("NEXT121 minimum-norm diagnostics are inconsistent")
    localization_excess = np.maximum(0.0, (localization - 0.50) / 0.50)
    cell_decades = np.maximum(
        0.0,
        np.log10(np.maximum(cell / PHYSICAL_BUDGET, 1.0)),
    )
    return {
        "required_budget_decades": required_decades,
        "tau10_debt_excess": debt_excess,
        "coordinate_localization_excess": localization_excess,
        "cell_strain_budget_decades": cell_decades,
    }


def materialize_bvtbd_tail_terms(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Derive and reversibly encode the four frozen BVTBD high tails."""

    raw_values = _derived_bvtbd_raw(features)
    columns: dict[str, np.ndarray] = {}
    terms: list[dict[str, object]] = []
    supported = features["bvtbd_supported"].fillna(False).astype(bool).to_numpy()
    for raw_spec in FROZEN_TERM_SPECS:
        spec = dict(raw_spec)
        term_id = str(spec["term_id"])
        raw_key = str(spec["raw_key"])
        raw = raw_values[raw_key]
        center = float(spec["center"])
        scale = float(spec["scale"])
        cap = float(spec["clip_normalized"])
        if not all(math.isfinite(value) for value in (center, scale, cap)) or scale <= 0.0 or cap <= 0.0:
            raise ValueError("NEXT121 frozen BVTBD calibration differs")
        active = supported & np.isfinite(raw)
        tail = np.zeros(len(features), dtype=float)
        tail[active] = np.clip(
            np.maximum(0.0, (raw[active] - center) / scale),
            0.0,
            cap,
        )
        encoded = np.sinh(tail)
        if not np.isfinite(encoded).all():
            raise ValueError("NEXT121 reversible BVTBD encoding overflowed")
        raw_name = f"_{term_id}_physical_raw"
        feature_name = f"_{term_id}_encoded"
        support_name = f"_{term_id}_active"
        columns[raw_name] = raw
        columns[feature_name] = encoded
        columns[support_name] = active
        terms.append(
            {
                "term_id": term_id,
                "feature": feature_name,
                "direction": 1,
                "transform": "asinh",
                "group": str(spec["group"]),
                "support_column": support_name,
                "center": 0.0,
                "scale": 1.0,
                "raw_feature": raw_name,
                "physical_center": center,
                "physical_scale": scale,
                "raw_p995": float(spec["raw_p995"]),
                "clip_normalized": cap,
                "nonzero_rows": int(spec["nonzero_rows"]),
                "numerical_zero_tolerance": NUMERICAL_ZERO_TOLERANCE,
                "missing_policy": "OPTIONAL_GUARD_OFF_KEEP_BASE",
                "encoding": "asinh_sinh_exact_clipped_frozen_bvtbd_tail",
            }
        )
    return (
        pd.concat([features.reset_index(drop=True), pd.DataFrame(columns)], axis=1),
        terms,
    )


def build_bvtbd_guard_configurations(
    term_specs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Enumerate four singles and every pair on the frozen grids."""

    terms = sorted(
        (dict(term) for term in term_specs), key=lambda item: str(item.get("term_id"))
    )
    term_ids = [term.get("term_id") for term in terms]
    if (
        len(terms) != 4
        or len(set(term_ids)) != 4
        or any(not isinstance(term_id, str) for term_id in term_ids)
        or any(term.get("group") != "bvtbd_core" for term in terms)
    ):
        raise ValueError("NEXT121 frozen BVTBD term identity differs")
    configurations: dict[str, dict[str, object]] = {}

    def add(selected: Sequence[Mapping[str, object]], weights: Sequence[float]) -> None:
        components = [
            {
                "term_id": str(term["term_id"]),
                "group": str(term["group"]),
                "weight": float(weight),
            }
            for term, weight in zip(selected, weights, strict=True)
        ]
        configuration_id = prior._configuration_id(components)
        configurations[configuration_id] = {
            "configuration_id": configuration_id,
            "components": components,
        }

    for term in terms:
        for weight in SINGLE_WEIGHT_GRID:
            add([term], [weight])
    for first, second in itertools.combinations(terms, 2):
        for first_weight in PAIR_WEIGHT_GRID:
            for second_weight in PAIR_WEIGHT_GRID:
                add([first, second], [first_weight, second_weight])
    return [configurations[key] for key in sorted(configurations)]


def materialize_bvtbd_composite_guard_terms(
    *,
    features: pd.DataFrame,
    eligible_terms: Sequence[Mapping[str, object]],
    configurations: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, dict[str, object]]]:
    """Encode weighted BVTBD sums while enforcing the NEXT121 weight grids."""

    by_id = {str(term["term_id"]): dict(term) for term in eligible_terms}
    if len(by_id) != len(eligible_terms):
        raise ValueError("NEXT121 eligible BVTBD term IDs are duplicated")
    risks: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    for term_id, term in by_id.items():
        risks[term_id], active[term_id] = _optional_term_risk(features, term)

    columns: dict[str, object] = {}
    composite_terms: list[dict[str, object]] = []
    mapping: dict[str, dict[str, object]] = {}
    for raw_configuration in configurations:
        configuration = dict(raw_configuration)
        configuration_id = str(configuration.get("configuration_id"))
        components_raw = configuration.get("components")
        if not isinstance(components_raw, list) or len(components_raw) not in (1, 2):
            raise ValueError("NEXT121 guard component count differs")
        components = [dict(component) for component in components_raw]
        term_ids = [str(component.get("term_id")) for component in components]
        groups = {str(component.get("group")) for component in components}
        weights = [float(component.get("weight")) for component in components]
        allowed_weights = SINGLE_WEIGHT_GRID if len(components) == 1 else PAIR_WEIGHT_GRID
        if (
            configuration_id != prior._configuration_id(components)
            or len(set(term_ids)) != len(term_ids)
            or any(term_id not in by_id for term_id in term_ids)
            or groups != {"bvtbd_core"}
            or any(str(by_id[term_id].get("group")) not in groups for term_id in term_ids)
            or any(weight not in allowed_weights for weight in weights)
        ):
            raise ValueError("NEXT121 BVTBD guard configuration differs")

        combined = np.zeros(len(features), dtype=float)
        is_active = np.ones(len(features), dtype=bool)
        for term_id, weight in zip(term_ids, weights, strict=True):
            combined += weight * risks[term_id]
            is_active &= active[term_id]
        combined[~is_active] = 0.0
        if not np.isfinite(combined).all() or np.any(combined < 0.0):
            raise ValueError("NEXT121 composite BVTBD risk is not finite")
        maximum = float(np.max(combined)) if len(combined) else 0.0
        divisor = max(1.0, maximum / 50.0)
        encoded = np.sinh(combined / divisor)
        if not np.isfinite(encoded).all():
            raise ValueError("NEXT121 reversible BVTBD guard encoding overflowed")

        term_id = prior._composite_term_id(configuration_id)
        feature_name = f"_{term_id}_value"
        support_name = f"_{term_id}_active"
        columns[feature_name] = encoded
        columns[support_name] = is_active
        composite_terms.append(
            {
                "term_id": term_id,
                "feature": feature_name,
                "direction": 1,
                "transform": "asinh",
                "group": "bvtbd_core",
                "support_column": support_name,
                "center": 0.0,
                "scale": 1.0 / divisor,
                "configuration_id": configuration_id,
                "components": components,
                "encoding": "asinh_sinh_exact_weighted_bvtbd_risk_sum",
            }
        )
        mapping[term_id] = {
            "configuration_id": configuration_id,
            "components": components,
        }
    extended = pd.concat(
        [features.reset_index(drop=True), pd.DataFrame(columns)], axis=1
    )
    return extended, composite_terms, mapping


def select_frontier_bases(records: pd.DataFrame) -> pd.DataFrame:
    """Freeze the AUC-best-SAFE and all-SAFE conflict fronts from NEXT117."""

    required = {
        "candidate_key",
        "base_term_ids_json",
        "base_weights_json",
        "optional_term_ids_json",
        "optional_weights_json",
        "passes_source_auc_gates",
        "safe_passing_cells",
        "passes_safe_all_cells",
    }
    if required - set(records.columns):
        raise ValueError("NEXT121 frontier base schema differs")
    auc = records["passes_source_auc_gates"].fillna(False).astype(bool)
    safe_all = records["passes_safe_all_cells"].fillna(False).astype(bool)
    safe_cells = pd.to_numeric(records["safe_passing_cells"], errors="coerce")
    keep = (auc & safe_cells.eq(FRONTIER_AUC_SAFE_CELLS)) | safe_all
    selected_rows: list[dict[str, object]] = []
    for _, row in records.loc[keep].iterrows():
        base_ids = [str(value) for value in json.loads(str(row["base_term_ids_json"]))]
        base_weights = [
            float(value) for value in json.loads(str(row["base_weights_json"]))
        ]
        optional_ids = [
            str(value) for value in json.loads(str(row["optional_term_ids_json"]))
        ]
        optional_weights = [
            float(value) for value in json.loads(str(row["optional_weights_json"]))
        ]
        term_ids = [*base_ids, *optional_ids]
        weights = [*base_weights, *optional_weights]
        if (
            len(term_ids) not in tuple(range(4, 10))
            or len(term_ids) != len(weights)
            or len(set(term_ids)) != len(term_ids)
            or any(not math.isfinite(weight) or weight <= 0.0 for weight in weights)
        ):
            raise ValueError("NEXT121 flattened frontier formula differs")
        selected_rows.append(
            {
                "prior_candidate_key": str(row["candidate_key"]),
                "term_ids_json": json.dumps(term_ids, separators=(",", ":")),
                "weights_json": json.dumps(weights, separators=(",", ":")),
                "frontier_route": "auc"
                if bool(row["passes_source_auc_gates"])
                else "safe",
            }
        )
    return pd.DataFrame(
        selected_rows,
        columns=(
            "prior_candidate_key",
            "term_ids_json",
            "weights_json",
            "frontier_route",
        ),
    ).reset_index(drop=True)


def build_frontier_candidate_specs(
    *,
    base_records: pd.DataFrame,
    old_term_ids: set[str],
    configurations: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Attach zero, one, or two frozen BVTBD terms to every frontier base."""

    required = {"prior_candidate_key", "term_ids_json", "weights_json"}
    if required - set(base_records.columns):
        raise ValueError("NEXT121 frontier candidate columns differ")
    configuration_ids = sorted(
        str(configuration["configuration_id"]) for configuration in configurations
    )
    specs: dict[str, dict[str, object]] = {}
    for _, row in base_records.iterrows():
        term_ids = [str(value) for value in json.loads(str(row["term_ids_json"]))]
        weights = [float(value) for value in json.loads(str(row["weights_json"]))]
        if (
            len(term_ids) not in tuple(range(4, 10))
            or len(weights) != len(term_ids)
            or len(set(term_ids)) != len(term_ids)
            or any(term_id not in old_term_ids for term_id in term_ids)
            or any(not math.isfinite(weight) or weight <= 0.0 for weight in weights)
        ):
            raise ValueError("NEXT121 frontier base formula differs")

        def add(configuration_id: str | None) -> None:
            payload = {
                "base_term_ids": term_ids,
                "base_weights": weights,
                "optional_configuration_id": configuration_id,
            }
            key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            specs[key] = {
                "candidate_key": key,
                "base_term_ids": term_ids,
                "base_weights": weights,
                "optional_term_id": None
                if configuration_id is None
                else prior._composite_term_id(configuration_id),
                "optional_weight": 0.0 if configuration_id is None else 1.0,
                "optional_configuration_id": configuration_id,
            }

        add(None)
        for configuration_id in configuration_ids:
            add(configuration_id)
    return [specs[key] for key in sorted(specs)]


def _paths(roots: Mapping[str, Path], freeze_path: Path) -> dict[str, Path]:
    return {
        "scigen_features": roots["scigen_features"] / prior.SCIGEN_FEATURE_NAMES["discovery"],
        "scigen_endpoint": roots["scigen_endpoint"] / prior.SCIGEN_ENDPOINT_NAME,
        "wyformer_features": roots["wyformer_features"] / prior.WYFORMER_FEATURE_NAMES["discovery"],
        "wyformer_endpoint": roots["wyformer_endpoint"] / prior.WYFORMER_ENDPOINT_NAME,
        "next98_manifest": roots["next98"] / prior.NEXT98_MANIFEST_NAME,
        "next98_term_catalogue": roots["next98"] / prior.NEXT98_CATALOGUE_NAME,
        "next110_manifest": roots["next110"] / prior.NEXT110_MANIFEST_NAME,
        "next110_catalogue": roots["next110"] / prior.NEXT110_CATALOGUE_NAME,
        "next110_scigen_features": roots["next110"] / prior.NEXT110_FEATURE_FILES["scigen"],
        "next110_wyformer_features": roots["next110"] / prior.NEXT110_FEATURE_FILES["wyformer"],
        "next111_manifest": roots["next111"] / prior.NEXT111_MANIFEST_NAME,
        "next111_catalogue": roots["next111"] / prior.NEXT111_CATALOGUE_NAME,
        "next113_manifest": roots["next113"] / prior.NEXT113_MANIFEST_NAME,
        "next113_catalogue": roots["next113"] / prior.NEXT113_CATALOGUE_NAME,
        "next113_scigen_features": roots["next113"] / prior.NEXT113_FEATURE_FILES["scigen"],
        "next113_wyformer_features": roots["next113"] / prior.NEXT113_FEATURE_FILES["wyformer"],
        "next114_manifest": roots["next114"] / prior.NEXT114_MANIFEST_NAME,
        "next114_catalogue": roots["next114"] / prior.NEXT114_CATALOGUE_NAME,
        "next114_search_records": roots["next114"] / prior.NEXT114_SEARCH_NAME,
        "next116_manifest": roots["next116"] / prior.NEXT116_MANIFEST_NAME,
        "next116_catalogue": roots["next116"] / prior.NEXT116_CATALOGUE_NAME,
        "next116_scigen_features": roots["next116"] / prior.NEXT116_FEATURE_FILES["scigen"],
        "next116_wyformer_features": roots["next116"] / prior.NEXT116_FEATURE_FILES["wyformer"],
        "next117_manifest": roots["next117"] / prior.MANIFEST_NAME,
        "next117_catalogue": roots["next117"] / prior.CATALOGUE_NAME,
        "next117_search_records": roots["next117"] / prior.SEARCH_NAME,
        "next120_manifest": roots["next120"] / NEXT120_MANIFEST_NAME,
        "next120_catalogue": roots["next120"] / NEXT120_CATALOGUE_NAME,
        "next120_scigen_features": roots["next120"] / NEXT120_FEATURE_FILES["scigen"],
        "next120_wyformer_features": roots["next120"] / NEXT120_FEATURE_FILES["wyformer"],
        "freeze": Path(freeze_path).resolve(),
    }


def run_bvtbd_frontier_rescue(
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
    freeze_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen BVTBD rescue after hashing its label-free catalogue."""

    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        "next98": Path(next98_dir).resolve(),
        "next110": Path(next110_dir).resolve(),
        "next111": Path(next111_dir).resolve(),
        "next113": Path(next113_dir).resolve(),
        "next114": Path(next114_dir).resolve(),
        "next116": Path(next116_dir).resolve(),
        "next117": Path(next117_dir).resolve(),
        "next120": Path(next120_dir).resolve(),
    }
    target = Path(output_dir).resolve()
    paths = _paths(roots, freeze_path)
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT121 discovery input is missing")
    input_hashes = {name: prior._sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT121 formal input identity differs")

    manifests = {
        "next98": prior._read_json(paths["next98_manifest"]),
        "next110": prior._read_json(paths["next110_manifest"]),
        "next111": prior._read_json(paths["next111_manifest"]),
        "next113": prior._read_json(paths["next113_manifest"]),
        "next114": prior._read_json(paths["next114_manifest"]),
        "next116": prior._read_json(paths["next116_manifest"]),
        "next117": prior._read_json(paths["next117_manifest"]),
        "next120": prior._read_json(paths["next120_manifest"]),
    }
    if (
        manifests["next98"].get("protocol") != prior.NEXT98_PROTOCOL
        or manifests["next110"].get("protocol") != prior.NEXT110_PROTOCOL
        or manifests["next111"].get("protocol") != prior.NEXT111_PROTOCOL
        or manifests["next113"].get("protocol") != prior.NEXT113_PROTOCOL
        or manifests["next114"].get("protocol") != prior.NEXT114_PROTOCOL
        or manifests["next116"].get("protocol") != prior.NEXT116_PROTOCOL
        or manifests["next117"].get("protocol") != prior.PROTOCOL
        or manifests["next117"].get("passes_all_cross_source_discovery_gates") is not False
        or manifests["next117"].get("opened_validation_outputs_used") is not False
        or manifests["next117"].get("scigen_replication_endpoint_opened") is not False
        or manifests["next117"].get("wyformer_replication_endpoint_opened") is not False
        or manifests["next120"].get("protocol") != NEXT120_PROTOCOL
        or manifests["next120"].get("labels_opened") is not False
        or manifests["next120"].get("endpoint_payloads_opened") is not False
        or manifests["next120"].get("validation_geometry_opened") is not False
        or manifests["next120"].get("replication_geometry_opened") is not False
        or manifests["next120"].get("dft_values_used_by_features") is not False
    ):
        raise ValueError("NEXT121 prior provenance differs")
    for manifest, expected in (
        (
            manifests["next117"],
            {
                prior.CATALOGUE_NAME: "next117_catalogue",
                prior.SEARCH_NAME: "next117_search_records",
            },
        ),
        (
            manifests["next120"],
            {
                NEXT120_CATALOGUE_NAME: "next120_catalogue",
                NEXT120_FEATURE_FILES["scigen"]: "next120_scigen_features",
                NEXT120_FEATURE_FILES["wyformer"]: "next120_wyformer_features",
            },
        ),
    ):
        outputs = manifest.get("outputs_sha256")
        if not isinstance(outputs, Mapping) or any(
            outputs.get(filename) != input_hashes[key]
            for filename, key in expected.items()
        ):
            raise ValueError("NEXT121 prior output provenance differs")

    old_tables = {
        "scigen": pd.read_parquet(paths["scigen_features"]),
        "wyformer": pd.read_parquet(paths["wyformer_features"]),
    }
    cmvo_tables = {
        "scigen": pd.read_parquet(paths["next110_scigen_features"]),
        "wyformer": pd.read_parquet(paths["next110_wyformer_features"]),
    }
    morphology_tables = {
        "scigen": pd.read_parquet(paths["next113_scigen_features"]),
        "wyformer": pd.read_parquet(paths["next113_wyformer_features"]),
    }
    hcid_tables = {
        "scigen": pd.read_parquet(paths["next116_scigen_features"]),
        "wyformer": pd.read_parquet(paths["next116_wyformer_features"]),
    }
    bvtbd_tables = {
        "scigen": pd.read_parquet(paths["next120_scigen_features"]),
        "wyformer": pd.read_parquet(paths["next120_wyformer_features"]),
    }
    feature_tables: dict[str, pd.DataFrame] = {}
    for source in ("scigen", "wyformer"):
        tables = (
            old_tables[source],
            cmvo_tables[source],
            morphology_tables[source],
            hcid_tables[source],
            bvtbd_tables[source],
        )
        if any(table["material_id"].astype(str).duplicated().any() for table in tables):
            raise ValueError(f"NEXT121 {source} feature identities are duplicated")
        if (
            set(prior.NEXT110_FEATURE_COLUMNS) - set(cmvo_tables[source].columns)
            or set(prior.NEXT113_FEATURE_COLUMNS) - set(morphology_tables[source].columns)
            or set(prior.NEXT116_FEATURE_COLUMNS) - set(hcid_tables[source].columns)
            or set(NEXT120_FEATURE_COLUMNS) - set(bvtbd_tables[source].columns)
        ):
            raise ValueError(f"NEXT121 {source} feature schema differs")
        merged = old_tables[source].merge(
            cmvo_tables[source].loc[:, ["material_id", *prior.NEXT110_FEATURE_COLUMNS]],
            on="material_id",
            how="inner",
            validate="one_to_one",
        ).merge(
            morphology_tables[source].loc[:, ["material_id", *prior.NEXT113_FEATURE_COLUMNS]],
            on="material_id",
            how="inner",
            validate="one_to_one",
        ).merge(
            hcid_tables[source].loc[:, ["material_id", *prior.NEXT116_FEATURE_COLUMNS]],
            on="material_id",
            how="inner",
            validate="one_to_one",
        ).merge(
            bvtbd_tables[source].loc[:, ["material_id", *NEXT120_FEATURE_COLUMNS]],
            on="material_id",
            how="inner",
            validate="one_to_one",
        )
        if any(len(merged) != len(table) for table in tables):
            raise ValueError(f"NEXT121 {source} feature row accounting differs")
        merged = merged.copy()
        merged["source_dataset"] = source
        if source == "scigen":
            merged["crystal_system"] = merged["lattice_class"].astype(str)
        merged["material_id"] = source + ":" + merged["material_id"].astype(str)
        feature_tables[source] = merged
    feature_combined = pd.concat(
        [feature_tables["scigen"], feature_tables["wyformer"]],
        ignore_index=True,
        sort=False,
    )

    cmvo_features, cmvo_terms = prior.materialize_cmvo_tail_terms(feature_combined)
    morphology_features, morphology_terms = prior.materialize_cmvom_tail_terms(cmvo_features)
    hcid_features, hcid_terms = prior.materialize_hcid_tail_terms(morphology_features)
    bvtbd_features, eligible_terms = materialize_bvtbd_tail_terms(hcid_features)
    configurations = build_bvtbd_guard_configurations(FROZEN_TERM_SPECS)
    extended_features, composite_terms, composite_mapping = materialize_bvtbd_composite_guard_terms(
        features=bvtbd_features,
        eligible_terms=eligible_terms,
        configurations=configurations,
    )

    old_catalogue = prior._read_json(paths["next98_term_catalogue"])
    next111_catalogue = prior._read_json(paths["next111_catalogue"])
    next114_catalogue = prior._read_json(paths["next114_catalogue"])
    next117_catalogue = prior._read_json(paths["next117_catalogue"])
    old_terms_raw = old_catalogue.get("eligible_terms")
    if (
        not isinstance(old_terms_raw, list)
        or next111_catalogue.get("eligible_optional_terms") != cmvo_terms
        or next114_catalogue.get("eligible_optional_terms") != morphology_terms
        or next117_catalogue.get("eligible_optional_terms") != hcid_terms
    ):
        raise ValueError("NEXT121 prior term catalogue differs")
    old_terms = [*old_terms_raw, *cmvo_terms, *morphology_terms, *hcid_terms]
    old_term_ids = {str(term["term_id"]) for term in old_terms}
    if len(old_term_ids) != len(old_terms):
        raise ValueError("NEXT121 old term identities are duplicated")

    prior_records = pd.read_parquet(paths["next117_search_records"])
    frontier_bases = select_frontier_bases(prior_records)
    prior_by_key = {
        str(row["candidate_key"]): row.to_dict()
        for _, row in prior_records.iterrows()
    }
    frontier_bases["_prior_record"] = [
        prior_by_key[str(key)] for key in frontier_bases["prior_candidate_key"]
    ]
    specs = build_frontier_candidate_specs(
        base_records=frontier_bases,
        old_term_ids=old_term_ids,
        configurations=configurations,
    )
    base_keys = sorted(frontier_bases["prior_candidate_key"].astype(str))
    base_formulas = sorted(
        prior._formula_identity(
            json.loads(str(row["term_ids_json"])),
            json.loads(str(row["weights_json"])),
        )
        for _, row in frontier_bases.iterrows()
    )
    base_key_sha = hashlib.sha256("\n".join(base_keys).encode("utf-8")).hexdigest()
    base_formula_sha = hashlib.sha256(
        "\n".join(base_formulas).encode("utf-8")
    ).hexdigest()
    auc_frontier_count = int(frontier_bases["frontier_route"].eq("auc").sum())
    safe_frontier_count = int(frontier_bases["frontier_route"].eq("safe").sum())
    candidate_key_sha = hashlib.sha256(
        "\n".join(spec["candidate_key"] for spec in specs).encode("utf-8")
    ).hexdigest()
    label_free_catalogue = {
        "protocol": PROTOCOL,
        "calibration_stage": "bvtbd_label_free_before_endpoint_join",
        "freeze_sha256": input_hashes["freeze"],
        "eligible_optional_terms": eligible_terms,
        "excluded_features": list(EXCLUDED_FEATURES),
        "single_weight_grid": list(SINGLE_WEIGHT_GRID),
        "pair_weight_grid": list(PAIR_WEIGHT_GRID),
        "guard_configurations": configurations,
        "configuration_count": len(configurations),
        "frontier_base_count": len(frontier_bases),
        "auc_frontier_base_count": auc_frontier_count,
        "safe_frontier_base_count": safe_frontier_count,
        "frontier_base_key_sha256": base_key_sha,
        "frontier_base_formula_sha256": base_formula_sha,
        "candidate_count": len(specs),
        "candidate_key_sha256": candidate_key_sha,
        "optional_missing_policy": "OPTIONAL_GUARD_OFF_KEEP_BASE",
        "new_bvtbd_features_joined_to_endpoint_before_freeze": False,
        "discovery_endpoints_previously_opened_by_next117": True,
    }
    label_free_catalogue_sha256 = hashlib.sha256(
        prior._json_bytes(label_free_catalogue)
    ).hexdigest()
    if require_formal_inputs and (
        len(frontier_bases) != EXPECTED_FRONTIER_BASES
        or auc_frontier_count != EXPECTED_AUC_FRONTIER_BASES
        or safe_frontier_count != EXPECTED_SAFE_FRONTIER_BASES
        or base_key_sha != EXPECTED_BASE_CANDIDATE_KEY_SHA256
        or base_formula_sha != EXPECTED_BASE_FORMULA_SHA256
        or len(configurations) != EXPECTED_CONFIGURATION_COUNT
        or len(specs) != EXPECTED_CANDIDATE_COUNT
        or candidate_key_sha != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT121 frozen candidate universe differs")

    # These discovery outcomes were opened in prior stages. Re-read them only
    # after the new label-free catalogue and complete candidate universe exist.
    scigen_endpoints = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_endpoints = pd.read_parquet(paths["wyformer_endpoint"])
    if (
        scigen_endpoints["material_id"].astype(str).duplicated().any()
        or wyformer_endpoints["material_id"].astype(str).duplicated().any()
    ):
        raise ValueError("NEXT121 discovery endpoint identities are duplicated")
    endpoint_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "material_id": "scigen:" + scigen_endpoints["material_id"].astype(str),
                    "_endpoint_numeric": pd.to_numeric(
                        scigen_endpoints["distortion_ratio"], errors="coerce"
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "material_id": "wyformer:" + wyformer_endpoints["material_id"].astype(str),
                    "_endpoint_numeric": prior._endpoint_numeric(
                        wyformer_endpoints["endpoint_stratum"]
                    ),
                }
            ),
        ],
        ignore_index=True,
    )
    combined = extended_features.merge(
        endpoint_frame, on="material_id", how="inner", validate="one_to_one"
    )
    if len(combined) != len(extended_features) or len(combined) != len(endpoint_frame):
        raise ValueError("NEXT121 endpoint row accounting differs")
    endpoint = pd.to_numeric(
        combined.pop("_endpoint_numeric"), errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(endpoint).all():
        raise ValueError("NEXT121 endpoint conversion differs")

    started = time.perf_counter()
    result = search_optional_guard_laws(
        features=combined,
        endpoint=endpoint,
        old_terms=old_terms,
        optional_terms=composite_terms,
        candidate_specs=specs,
    )
    elapsed = time.perf_counter() - started
    prior._decorate_result(
        result=result,
        mapping=composite_mapping,
        eligible_terms=eligible_terms,
    )
    prior._verify_base_reproduction(
        result_records=result["candidate_records"],
        prior=frontier_bases,
    )
    result["selected"]["formula"]["kind"] = (
        "next117_frontier_base_plus_up_to_two_closed_form_bvtbd_terms"
    )
    selected = result["selected"]
    passes = bool(selected["record"]["passes_all_discovery_gates"])

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next98_cross_source_discovery_search.py": repository_root / "src/next98_cross_source_discovery_search.py",
        "src/next103_dobvr_optional_guard_search.py": repository_root / "src/next103_dobvr_optional_guard_search.py",
        "src/next107_two_axis_cmvf_guard_search.py": repository_root / "src/next107_two_axis_cmvf_guard_search.py",
        "src/next111_cmvo_optional_search.py": repository_root / "src/next111_cmvo_optional_search.py",
        "src/next114_cmvom_frontier_rescue.py": repository_root / "src/next114_cmvom_frontier_rescue.py",
        "src/next117_hcid_frontier_rescue.py": repository_root / "src/next117_hcid_frontier_rescue.py",
        "src/next119_bounded_valence_transport.py": repository_root / "src/next119_bounded_valence_transport.py",
        "src/next120_cross_source_bvtbd_features.py": repository_root / "src/next120_cross_source_bvtbd_features.py",
        "src/next121_bvtbd_frontier_rescue.py": Path(__file__).resolve(),
    }
    source_hashes = {
        name: prior._sha256_file(path) for name, path in source_paths.items()
    }
    output_paths: list[Path] = []
    try:
        catalogue = {
            **label_free_catalogue,
            "label_free_catalogue_sha256": label_free_catalogue_sha256,
        }
        evaluation = {
            "protocol": PROTOCOL,
            "evaluation_mode": "adaptive_cross_source_discovery_bvtbd_frontier_rescue",
            "rows": {
                "scigen": int(len(feature_tables["scigen"])),
                "wyformer": int(len(feature_tables["wyformer"])),
                "total": int(len(combined)),
            },
            "frontier_base_count": len(frontier_bases),
            "configuration_count": len(configurations),
            "candidate_count": int(result["candidate_count"]),
            "elapsed_seconds": elapsed,
            "base_only_reproduced_next117": True,
            "safe_gates": dict(prior.DEFAULT_GATES),
            "source_auc_gates": dict(prior.AUC_GATES),
            "broad_min_severe_precision_lower": prior.BROAD_MIN_PRECISION_LOWER,
            "selected_record": selected["record"],
            "selected_formula": selected["formula"],
            "selected_safe": selected["safe"],
            "selected_safe_diagnostic": selected["safe_diagnostic"],
            "selected_broad": selected["broad"],
            "selected_source_diagnostics": selected["source_diagnostics"],
            "pauling_by_cell": result["pauling_by_cell"],
            "cells": result["cells"],
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "requires_unopened_internal_validation_before_claim": True,
        }
        catalogue_path = staging / CATALOGUE_NAME
        evaluation_path = staging / EVALUATION_NAME
        search_path = staging / SEARCH_NAME
        prior._write_json(catalogue_path, catalogue)
        prior._write_json(evaluation_path, evaluation)
        pd.DataFrame(result["candidate_records"]).to_parquet(search_path, index=False)
        output_paths.extend([catalogue_path, evaluation_path, search_path])
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "label_free_catalogue_sha256": label_free_catalogue_sha256,
            "frontier_base_count": len(frontier_bases),
            "configuration_count": len(configurations),
            "candidate_count": int(result["candidate_count"]),
            "base_only_reproduced_next117": True,
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "requires_unopened_internal_validation_before_claim": True,
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
                path.name: prior._sha256_file(path) for path in output_paths
            },
        }
        prior._write_json(staging / MANIFEST_NAME, manifest)
        if any(prior._sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT121 input changed before publication")
        if any(prior._sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT121 source changed before publication")
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
    for stage in (98, 110, 111, 113, 114, 116, 117, 120):
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--freeze-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = run_bvtbd_frontier_rescue(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        next98_dir=args.next98_dir,
        next110_dir=args.next110_dir,
        next111_dir=args.next111_dir,
        next113_dir=args.next113_dir,
        next114_dir=args.next114_dir,
        next116_dir=args.next116_dir,
        next117_dir=args.next117_dir,
        next120_dir=args.next120_dir,
        freeze_path=args.freeze_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_CANDIDATE_COUNT",
    "EXPECTED_FREEZE_SHA256",
    "EXPECTED_INPUT_SHA256",
    "FROZEN_TERM_SPECS",
    "PROTOCOL",
    "build_bvtbd_guard_configurations",
    "build_frontier_candidate_specs",
    "materialize_bvtbd_composite_guard_terms",
    "materialize_bvtbd_tail_terms",
    "run_bvtbd_frontier_rescue",
    "select_frontier_bases",
]
