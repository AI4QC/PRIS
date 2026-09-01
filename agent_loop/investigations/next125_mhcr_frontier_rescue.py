#!/usr/bin/env python3
"""Frozen MHCR rescue of the NEXT122 cross-source discovery frontier."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import src.next122_safe12_bvtc_prlr_rescue as prior
import src.next121_bvtbd_frontier_rescue as n121
from src.next103_dobvr_optional_guard_search import (
    _optional_term_risk,
    search_optional_guard_laws,
)
from src.next124_cross_source_mhcr_features import (
    CATALOGUE_NAME as NEXT124_CATALOGUE_NAME,
    FEATURE_FILES as NEXT124_FEATURE_FILES,
    MANIFEST_NAME as NEXT124_MANIFEST_NAME,
    PROTOCOL as NEXT124_PROTOCOL,
)


PROTOCOL = "2026-08-08-next125-mhcr-frontier-rescue-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT125_MHCR_FRONTIER_CATALOGUE.json"
EVALUATION_NAME = "NEXT125_DISCOVERY_EVALUATION.json"
SEARCH_NAME = "next125_mhcr_frontier_candidate_search.parquet"
EXPECTED_FREEZE_SHA256 = (
    "e89ebcb4604ed4ae9f13fd205b6a5acf07b521948f62723d9d1bd68a85d10dde"
)
SINGLE_WEIGHT_GRID = (0.10, 0.25, 0.50, 1.00)
PAIR_WEIGHT_GRID = (0.10, 0.25, 0.50, 1.00)
FRONTIER_AUC_SAFE_CELLS = 11
SAFE_FRONTIER_LIMIT = 256
EXPECTED_FRONTIER_BASES = 506
EXPECTED_AUC_FRONTIER_BASES = 250
EXPECTED_SAFE_FRONTIER_BASES = 256
EXPECTED_CONFIGURATION_COUNT = 112
EXPECTED_CANDIDATE_COUNT = 57_178
EXPECTED_BASE_CANDIDATE_KEY_SHA256 = (
    "2aa85a2ae5e2b4cf11a8f64c2d4e329e50829533cd70067fc7d609a68378b814"
)
EXPECTED_BASE_FORMULA_SHA256 = (
    "02c0426478fd1aad0835c5b146c73d60b6336a0c0aafde54751b677063297316"
)
EXPECTED_CONFIGURATION_ID_SHA256 = (
    "0f8f330d9a559da633d06a14166755bd251350452b186a7ff289a86d99c055c2"
)
EXPECTED_CANDIDATE_KEY_SHA256 = (
    "2380e72e270eb332592aa39ef45a3978512a0dc6933971415be27b027d7da392"
)
BASE_REPRODUCTION_AUC_TOLERANCE = prior.BASE_REPRODUCTION_AUC_TOLERANCE
NUMERICAL_ZERO_TOLERANCE = 1.0e-12
SEARCH_WORKERS = 12

_PARALLEL_SEARCH_PAYLOAD: tuple[
    pd.DataFrame,
    np.ndarray,
    Sequence[Mapping[str, object]],
    Sequence[Mapping[str, object]],
] | None = None

FROZEN_TERM_SPECS = (
    {
        "term_id": "mhcr_core_positive_deficit_gain_tau50__high",
        "raw_feature": "mhcr_core_positive_deficit_gain_tau50",
        "support_column": "mhcr_core_supported",
        "group": "mhcr_core",
        "center": 0.0,
        "scale": 1.0,
        "clip_normalized": 1.0,
        "scigen_nonzero": 569,
        "wyformer_nonzero": 158,
        "scigen_p995": 0.4444444444444444,
        "wyformer_p995": 0.42857142857142855,
    },
    {
        "term_id": "mhcr_core_negative_deficit_gain_tau50__high",
        "raw_feature": "mhcr_core_negative_deficit_gain_tau50",
        "support_column": "mhcr_core_supported",
        "group": "mhcr_core",
        "center": 0.0,
        "scale": 1.0,
        "clip_normalized": 1.0,
        "scigen_nonzero": 452,
        "wyformer_nonzero": 223,
        "scigen_p995": 0.5,
        "wyformer_p995": 0.5,
    },
    {
        "term_id": "mhcr_expanded_positive_deficit_gain_tau50__high",
        "raw_feature": "mhcr_expanded_positive_deficit_gain_tau50",
        "support_column": "mhcr_expanded_supported",
        "group": "mhcr_expanded",
        "center": 0.0,
        "scale": 1.0,
        "clip_normalized": 1.0,
        "scigen_nonzero": 710,
        "wyformer_nonzero": 229,
        "scigen_p995": 0.5,
        "wyformer_p995": 0.5,
    },
    {
        "term_id": "mhcr_expanded_negative_deficit_gain_tau50__high",
        "raw_feature": "mhcr_expanded_negative_deficit_gain_tau50",
        "support_column": "mhcr_expanded_supported",
        "group": "mhcr_expanded",
        "center": 0.0,
        "scale": 1.0,
        "clip_normalized": 1.0,
        "scigen_nonzero": 1175,
        "wyformer_nonzero": 964,
        "scigen_p995": 0.5,
        "wyformer_p995": 0.6666666666666666,
    },
)

EXPECTED_INPUT_SHA256 = {
    **{key: value for key, value in prior.EXPECTED_INPUT_SHA256.items() if key != "freeze"},
    "next122_manifest": "c4d094d58f34dd69ba7a1528e072dfcd349d2d17b306faf6dfcdfa7208faac50",
    "next122_catalogue": "777e5b1e39979d0f8d1e5375e7998a11d059619ed23a4a0a06fb4c964971801b",
    "next122_evaluation": "5fb1a70b34ea677753f80f5893360e774fd159754cb49259a3ba35064f6bd189",
    "next122_search_records": "64a3a26286b862512dd9647b22dff8b87fb5360bdc487e6e4c94e25d50f46d2a",
    "next124_manifest": "32e7e4a7d9c74aea3ce029c654dfc9216ab466dda39459538f178d9e16beb8bb",
    "next124_catalogue": "8a5a3ad4cf123996309169caf9f94edd022f00617c697f1d6da66584b44f81d3",
    "next124_scigen_features": "50002b41a430278788f9c097a997651c34bd9acc350cad6078122731150c5ac7",
    "next124_wyformer_features": "76bf3129268b1adb53a0cf82a3c6f0a2786176f8e4a6f3e49ea44195b89b3932",
    "freeze": EXPECTED_FREEZE_SHA256,
}


def materialize_mhcr_tail_terms(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Reversibly encode the four frozen sparse MHCR high tails."""

    columns: dict[str, np.ndarray] = {}
    terms: list[dict[str, object]] = []
    for spec_raw in FROZEN_TERM_SPECS:
        spec = dict(spec_raw)
        source_name = str(spec["raw_feature"])
        source_support = str(spec["support_column"])
        if source_name not in features or source_support not in features:
            raise ValueError("NEXT125 MHCR feature schema differs")
        raw = pd.to_numeric(features[source_name], errors="coerce").to_numpy(float)
        supported = features[source_support].fillna(False).astype(bool).to_numpy()
        if np.any(supported & (~np.isfinite(raw) | (raw < -1.0e-10) | (raw > 1.0 + 1.0e-10))):
            raise ValueError("NEXT125 supported MHCR value is invalid")
        active = supported & np.isfinite(raw)
        physical = np.full(len(features), np.nan, dtype=float)
        physical[active] = np.clip(raw[active], 0.0, 1.0)
        tail = np.zeros(len(features), dtype=float)
        tail[active] = physical[active]
        encoded = np.sinh(tail)
        if not np.isfinite(encoded).all():
            raise ValueError("NEXT125 MHCR encoding overflowed")
        term_id = str(spec["term_id"])
        raw_name = f"_{term_id}_physical_raw"
        encoded_name = f"_{term_id}_encoded"
        active_name = f"_{term_id}_active"
        columns[raw_name] = physical
        columns[encoded_name] = encoded
        columns[active_name] = active
        terms.append(
            {
                "term_id": term_id,
                "feature": encoded_name,
                "direction": 1,
                "transform": "asinh",
                "group": str(spec["group"]),
                "support_column": active_name,
                "center": 0.0,
                "scale": 1.0,
                "raw_feature": raw_name,
                "source_raw_feature": source_name,
                "physical_center": 0.0,
                "physical_scale": 1.0,
                "clip_normalized": 1.0,
                "scigen_nonzero": int(spec["scigen_nonzero"]),
                "wyformer_nonzero": int(spec["wyformer_nonzero"]),
                "scigen_p995": float(spec["scigen_p995"]),
                "wyformer_p995": float(spec["wyformer_p995"]),
                "numerical_zero_tolerance": NUMERICAL_ZERO_TOLERANCE,
                "missing_policy": "OPTIONAL_GUARD_OFF_KEEP_BASE",
                "encoding": "asinh_sinh_exact_clipped_frozen_mhcr_tail",
            }
        )
    return (
        pd.concat([features.reset_index(drop=True), pd.DataFrame(columns)], axis=1),
        terms,
    )


def build_mhcr_guard_configurations(
    term_specs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Enumerate four singles and every pair on the fixed weight grid."""

    terms = sorted((dict(term) for term in term_specs), key=lambda item: str(item.get("term_id")))
    identities = [str(term.get("term_id")) for term in terms]
    if (
        len(terms) != 4
        or len(set(identities)) != 4
        or any(term.get("group") not in {"mhcr_core", "mhcr_expanded"} for term in terms)
        or any(not identity.startswith("mhcr_") or not identity.endswith("__high") for identity in identities)
    ):
        raise ValueError("NEXT125 frozen MHCR term identity differs")
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
        configuration_id = n121.prior._configuration_id(components)
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


def materialize_mhcr_composite_guard_terms(
    *,
    features: pd.DataFrame,
    eligible_terms: Sequence[Mapping[str, object]],
    configurations: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, dict[str, object]]]:
    """Encode every fixed one/two-term fail-open MHCR guard."""

    by_id = {str(term["term_id"]): dict(term) for term in eligible_terms}
    if len(by_id) != len(eligible_terms):
        raise ValueError("NEXT125 eligible MHCR term IDs are duplicated")
    risks: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    for term_id, term in by_id.items():
        risks[term_id], active[term_id] = _optional_term_risk(features, term)
    columns: dict[str, np.ndarray] = {}
    composite_terms: list[dict[str, object]] = []
    mapping: dict[str, dict[str, object]] = {}
    for raw_configuration in configurations:
        configuration = dict(raw_configuration)
        configuration_id = str(configuration.get("configuration_id"))
        components_raw = configuration.get("components")
        if not isinstance(components_raw, list) or len(components_raw) not in (1, 2):
            raise ValueError("NEXT125 MHCR guard component count differs")
        components = [dict(component) for component in components_raw]
        term_ids = [str(component.get("term_id")) for component in components]
        weights = [float(component.get("weight")) for component in components]
        allowed = SINGLE_WEIGHT_GRID if len(components) == 1 else PAIR_WEIGHT_GRID
        if (
            configuration_id != n121.prior._configuration_id(components)
            or len(set(term_ids)) != len(term_ids)
            or any(term_id not in by_id for term_id in term_ids)
            or any(weight not in allowed for weight in weights)
            or any(str(component.get("group")) != str(by_id[term_id].get("group")) for component, term_id in zip(components, term_ids, strict=True))
        ):
            raise ValueError("NEXT125 MHCR guard configuration differs")
        combined = np.zeros(len(features), dtype=float)
        is_active = np.ones(len(features), dtype=bool)
        for term_id, weight in zip(term_ids, weights, strict=True):
            combined += weight * risks[term_id]
            is_active &= active[term_id]
        combined[~is_active] = 0.0
        if not np.isfinite(combined).all() or np.any(combined < 0.0):
            raise ValueError("NEXT125 composite MHCR risk is invalid")
        maximum = float(np.max(combined)) if len(combined) else 0.0
        divisor = max(1.0, maximum / 50.0)
        encoded = np.sinh(combined / divisor)
        if not np.isfinite(encoded).all():
            raise ValueError("NEXT125 composite MHCR encoding overflowed")
        term_id = n121.prior._composite_term_id(configuration_id)
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
                "group": "mhcr_composite",
                "support_column": support_name,
                "center": 0.0,
                "scale": 1.0 / divisor,
                "configuration_id": configuration_id,
                "components": components,
                "encoding": "asinh_sinh_exact_weighted_mhcr_risk_sum",
            }
        )
        mapping[term_id] = {
            "configuration_id": configuration_id,
            "components": components,
        }
    return (
        pd.concat([features.reset_index(drop=True), pd.DataFrame(columns)], axis=1),
        composite_terms,
        mapping,
    )


def select_frontier_bases(
    records: pd.DataFrame,
    *,
    safe_limit: int = SAFE_FRONTIER_LIMIT,
) -> pd.DataFrame:
    """Select every AUC+SAFE11 law and the top-margin SAFE12 laws."""

    metrics = (
        "scigen_pooled_auc",
        "scigen_macro_auc",
        "scigen_worst_auc",
        "wyformer_pooled_auc",
        "wyformer_macro_auc",
        "wyformer_worst_auc",
    )
    required = {
        "candidate_key",
        "base_term_ids_json",
        "base_weights_json",
        "passes_source_auc_gates",
        "safe_passing_cells",
        "passes_safe_all_cells",
        *metrics,
    }
    if required - set(records.columns) or type(safe_limit) is not int or safe_limit < 1:
        raise ValueError("NEXT125 frontier base schema differs")
    frame = records.copy()
    thresholds = np.asarray([0.75, 0.60, 0.55, 0.75, 0.60, 0.55], dtype=float)
    values = frame.loc[:, metrics].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("NEXT125 prior AUC values differ")
    frame["_minimum_auc_margin"] = np.min(values - thresholds[None, :], axis=1)
    auc = frame["passes_source_auc_gates"].fillna(False).astype(bool)
    safe = frame["passes_safe_all_cells"].fillna(False).astype(bool)
    safe_cells = pd.to_numeric(frame["safe_passing_cells"], errors="coerce")
    auc_rows = frame.loc[auc & safe_cells.eq(FRONTIER_AUC_SAFE_CELLS)].sort_values("candidate_key")
    safe_rows = frame.loc[safe].sort_values(
        ["_minimum_auc_margin", "candidate_key"], ascending=[False, True]
    ).head(safe_limit)
    selected_rows: list[dict[str, object]] = []
    for route, table in (("auc_safe11", auc_rows), ("safe12", safe_rows)):
        for _, row in table.iterrows():
            term_ids = [str(value) for value in json.loads(str(row["base_term_ids_json"]))]
            weights = [float(value) for value in json.loads(str(row["base_weights_json"]))]
            if (
                not term_ids
                or len(term_ids) != len(weights)
                or len(set(term_ids)) != len(term_ids)
                or any(not math.isfinite(weight) or weight <= 0.0 for weight in weights)
            ):
                raise ValueError("NEXT125 flattened frontier formula differs")
            selected_rows.append(
                {
                    "prior_candidate_key": str(row["candidate_key"]),
                    "term_ids_json": json.dumps(term_ids, separators=(",", ":")),
                    "weights_json": json.dumps(weights, separators=(",", ":")),
                    "frontier_route": route,
                }
            )
    result = pd.DataFrame(
        selected_rows,
        columns=("prior_candidate_key", "term_ids_json", "weights_json", "frontier_route"),
    )
    if result["prior_candidate_key"].duplicated().any():
        raise ValueError("NEXT125 frontier base identities overlap")
    return result.reset_index(drop=True)


def build_frontier_candidate_specs(
    *,
    base_records: pd.DataFrame,
    old_term_ids: set[str],
    configurations: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Attach base, one, or two frozen MHCR terms to every frontier law."""

    required = {"prior_candidate_key", "term_ids_json", "weights_json"}
    if required - set(base_records.columns):
        raise ValueError("NEXT125 frontier candidate columns differ")
    configuration_ids = sorted(str(item["configuration_id"]) for item in configurations)
    specs: dict[str, dict[str, object]] = {}
    for _, row in base_records.iterrows():
        term_ids = [str(value) for value in json.loads(str(row["term_ids_json"]))]
        weights = [float(value) for value in json.loads(str(row["weights_json"]))]
        if (
            not term_ids
            or len(term_ids) != len(weights)
            or len(set(term_ids)) != len(term_ids)
            or any(term_id not in old_term_ids for term_id in term_ids)
            or any(not math.isfinite(weight) or weight <= 0.0 for weight in weights)
        ):
            raise ValueError("NEXT125 frontier base formula differs")

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
                "optional_term_id": None if configuration_id is None else n121.prior._composite_term_id(configuration_id),
                "optional_weight": 0.0 if configuration_id is None else 1.0,
                "optional_configuration_id": configuration_id,
            }

        add(None)
        for configuration_id in configuration_ids:
            add(configuration_id)
    return [specs[key] for key in sorted(specs)]


def verify_base_reproduction(
    *, result_records: Sequence[Mapping[str, object]], prior: pd.DataFrame
) -> None:
    """Prove pure NEXT122 bases reproduce before inspecting MHCR variants."""

    metrics = (
        "scigen_pooled_auc",
        "scigen_macro_auc",
        "scigen_worst_auc",
        "wyformer_pooled_auc",
        "wyformer_macro_auc",
        "wyformer_worst_auc",
    )
    observed: dict[str, Mapping[str, object]] = {}
    for record in result_records:
        if record.get("optional_configuration_id") is not None:
            continue
        key = n121.prior._formula_identity(
            json.loads(str(record["base_term_ids_json"])),
            json.loads(str(record["base_weights_json"])),
        )
        if key in observed:
            raise RuntimeError("NEXT125 base-only formula identities are duplicated")
        observed[key] = record

    expected: dict[str, Mapping[str, object]] = {}
    for _, row in prior.iterrows():
        key = n121.prior._formula_identity(
            json.loads(str(row["term_ids_json"])),
            json.loads(str(row["weights_json"])),
        )
        if key in expected:
            raise RuntimeError("NEXT125 prior base formula identities are duplicated")
        expected[key] = row
    if set(observed) != set(expected):
        raise RuntimeError("NEXT125 base-only reproduction identities differ")

    for key, row in expected.items():
        record = observed[key]
        source = row["_prior_record"]
        if any(
            not math.isclose(
                float(record[name]),
                float(source[name]),
                rel_tol=0.0,
                abs_tol=BASE_REPRODUCTION_AUC_TOLERANCE,
            )
            for name in metrics
        ) or any(
            bool(record[name]) != bool(source[name])
            for name in ("passes_source_auc_gates", "passes_safe_all_cells")
        ) or int(record["safe_passing_cells"]) != int(source["safe_passing_cells"]):
            raise RuntimeError("NEXT125 base-only diagnostics do not reproduce NEXT122")


def _search_candidate_shard(
    candidate_specs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    payload = _PARALLEL_SEARCH_PAYLOAD
    if payload is None:
        raise RuntimeError("NEXT125 parallel search payload is unavailable")
    features, endpoint, old_terms, optional_terms = payload
    return search_optional_guard_laws(
        features=features,
        endpoint=endpoint,
        old_terms=old_terms,
        optional_terms=optional_terms,
        candidate_specs=candidate_specs,
    )


def search_optional_guard_laws_parallel(
    *,
    features: pd.DataFrame,
    endpoint: object,
    old_terms: Sequence[Mapping[str, object]],
    optional_terms: Sequence[Mapping[str, object]],
    candidate_specs: Sequence[Mapping[str, object]],
    workers: int = SEARCH_WORKERS,
) -> dict[str, object]:
    """Evaluate fixed contiguous shards and merge by the original rank rule."""

    specs = list(candidate_specs)
    if type(workers) is not int or workers < 1 or not specs:
        raise ValueError("NEXT125 search workers or candidate universe differs")
    if workers == 1 or len(specs) == 1:
        return search_optional_guard_laws(
            features=features,
            endpoint=endpoint,
            old_terms=old_terms,
            optional_terms=optional_terms,
            candidate_specs=specs,
        )
    worker_count = min(workers, len(specs))
    shard_size = math.ceil(len(specs) / worker_count)
    shards = [specs[start : start + shard_size] for start in range(0, len(specs), shard_size)]
    global _PARALLEL_SEARCH_PAYLOAD
    if _PARALLEL_SEARCH_PAYLOAD is not None:
        raise RuntimeError("NEXT125 parallel search is not reentrant")
    _PARALLEL_SEARCH_PAYLOAD = (
        features,
        np.asarray(endpoint, dtype=float),
        list(old_terms),
        list(optional_terms),
    )
    try:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=len(shards),
            mp_context=context,
        ) as executor:
            results = list(executor.map(_search_candidate_shard, shards))
    finally:
        _PARALLEL_SEARCH_PAYLOAD = None
    reference = results[0]
    records = [record for result in results for record in result["candidate_records"]]
    if (
        len(records) != len(specs)
        or len({str(record["candidate_key"]) for record in records}) != len(records)
        or any(result["cells"] != reference["cells"] for result in results[1:])
        or any(result["pauling_by_cell"] != reference["pauling_by_cell"] for result in results[1:])
    ):
        raise RuntimeError("NEXT125 parallel shard accounting differs")
    selected = reference["selected"]
    for result in results[1:]:
        candidate = result["selected"]
        if candidate["rank"] > selected["rank"] or (
            candidate["rank"] == selected["rank"]
            and candidate["identity"] < selected["identity"]
        ):
            selected = candidate
    return {
        "candidate_records": records,
        "candidate_count": len(records),
        "cells": reference["cells"],
        "pauling_by_cell": reference["pauling_by_cell"],
        "selected": selected,
    }


def _paths(roots: Mapping[str, Path], freeze_path: Path) -> dict[str, Path]:
    paths = prior._paths(roots, freeze_path)
    paths.update(
        {
            "next122_manifest": roots["next122"] / prior.MANIFEST_NAME,
            "next122_catalogue": roots["next122"] / prior.CATALOGUE_NAME,
            "next122_evaluation": roots["next122"] / prior.EVALUATION_NAME,
            "next122_search_records": roots["next122"] / prior.SEARCH_NAME,
            "next124_manifest": roots["next124"] / NEXT124_MANIFEST_NAME,
            "next124_catalogue": roots["next124"] / NEXT124_CATALOGUE_NAME,
            "next124_scigen_features": roots["next124"] / NEXT124_FEATURE_FILES["scigen"],
            "next124_wyformer_features": roots["next124"] / NEXT124_FEATURE_FILES["wyformer"],
        }
    )
    return paths


def run_mhcr_frontier_rescue(
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
    freeze_path: Path,
    output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen MHCR frontier rescue after hashing every candidate."""

    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{f"next{stage}": Path(value).resolve() for stage, value in (
            (98, next98_dir), (110, next110_dir), (111, next111_dir),
            (113, next113_dir), (114, next114_dir), (116, next116_dir),
            (117, next117_dir), (120, next120_dir), (121, next121_dir),
            (122, next122_dir), (124, next124_dir),
        )},
    }
    target = Path(output_dir).resolve()
    paths = _paths(roots, freeze_path)
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT125 discovery input is missing")
    input_hashes = {name: n121.prior._sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT125 formal input identity differs")

    manifest122 = n121.prior._read_json(paths["next122_manifest"])
    manifest124 = n121.prior._read_json(paths["next124_manifest"])
    if (
        manifest122.get("protocol") != prior.PROTOCOL
        or manifest122.get("passes_all_cross_source_discovery_gates") is not False
        or manifest122.get("opened_validation_outputs_used") is not False
        or manifest122.get("scigen_replication_endpoint_opened") is not False
        or manifest122.get("wyformer_replication_endpoint_opened") is not False
        or manifest122.get("dft_values_used_by_executable_formula") is not False
        or manifest124.get("protocol") != NEXT124_PROTOCOL
        or manifest124.get("labels_opened") is not False
        or manifest124.get("endpoint_payloads_opened") is not False
        or manifest124.get("validation_geometry_opened") is not False
        or manifest124.get("replication_geometry_opened") is not False
        or manifest124.get("dft_values_used_by_features") is not False
    ):
        raise ValueError("NEXT125 prior provenance differs")
    for manifest, expected in (
        (
            manifest122,
            {
                prior.CATALOGUE_NAME: "next122_catalogue",
                prior.EVALUATION_NAME: "next122_evaluation",
                prior.SEARCH_NAME: "next122_search_records",
            },
        ),
        (
            manifest124,
            {
                NEXT124_CATALOGUE_NAME: "next124_catalogue",
                NEXT124_FEATURE_FILES["scigen"]: "next124_scigen_features",
                NEXT124_FEATURE_FILES["wyformer"]: "next124_wyformer_features",
            },
        ),
    ):
        outputs = manifest.get("outputs_sha256")
        if not isinstance(outputs, Mapping) or any(
            outputs.get(filename) != input_hashes[key] for filename, key in expected.items()
        ):
            raise ValueError("NEXT125 prior output provenance differs")

    features, feature_tables, old_terms = prior._reconstruct_label_free_table(paths)
    mhcr_tables = {
        "scigen": pd.read_parquet(paths["next124_scigen_features"]),
        "wyformer": pd.read_parquet(paths["next124_wyformer_features"]),
    }
    retained_columns = sorted(
        {
            str(spec["raw_feature"])
            for spec in FROZEN_TERM_SPECS
        }
        | {str(spec["support_column"]) for spec in FROZEN_TERM_SPECS}
    )
    mhcr_frames: list[pd.DataFrame] = []
    for source in ("scigen", "wyformer"):
        table = mhcr_tables[source]
        if table["material_id"].astype(str).duplicated().any() or set(retained_columns) - set(table.columns):
            raise ValueError(f"NEXT125 {source} MHCR feature schema differs")
        frame = table.loc[:, ["material_id", *retained_columns]].copy()
        frame["material_id"] = source + ":" + frame["material_id"].astype(str)
        mhcr_frames.append(frame)
    mhcr = pd.concat(mhcr_frames, ignore_index=True, sort=False)
    joined = features.merge(mhcr, on="material_id", how="inner", validate="one_to_one")
    if len(joined) != len(features) or len(joined) != len(mhcr):
        raise ValueError("NEXT125 MHCR feature row accounting differs")
    mhcr_features, eligible_terms = materialize_mhcr_tail_terms(joined)
    configurations = build_mhcr_guard_configurations(FROZEN_TERM_SPECS)
    extended, composite_terms, mapping = materialize_mhcr_composite_guard_terms(
        features=mhcr_features,
        eligible_terms=eligible_terms,
        configurations=configurations,
    )
    old_term_ids = {str(term["term_id"]) for term in old_terms}
    if len(old_term_ids) != len(old_terms):
        raise ValueError("NEXT125 prior term identities are duplicated")

    prior_records = pd.read_parquet(paths["next122_search_records"])
    bases = select_frontier_bases(prior_records)
    prior_by_key = {str(row["candidate_key"]): row.to_dict() for _, row in prior_records.iterrows()}
    bases["_prior_record"] = [prior_by_key[str(key)] for key in bases["prior_candidate_key"]]
    specs = build_frontier_candidate_specs(
        base_records=bases,
        old_term_ids=old_term_ids,
        configurations=configurations,
    )
    base_keys = sorted(bases["prior_candidate_key"].astype(str))
    base_formulas = sorted(
        n121.prior._formula_identity(
            json.loads(str(row["term_ids_json"])),
            json.loads(str(row["weights_json"])),
        )
        for _, row in bases.iterrows()
    )
    base_key_sha = hashlib.sha256("\n".join(base_keys).encode()).hexdigest()
    base_formula_sha = hashlib.sha256("\n".join(base_formulas).encode()).hexdigest()
    configuration_sha = hashlib.sha256(
        "\n".join(sorted(item["configuration_id"] for item in configurations)).encode()
    ).hexdigest()
    candidate_key_sha = hashlib.sha256(
        "\n".join(spec["candidate_key"] for spec in specs).encode()
    ).hexdigest()
    auc_count = int(bases["frontier_route"].eq("auc_safe11").sum())
    safe_count = int(bases["frontier_route"].eq("safe12").sum())
    label_free_catalogue = {
        "protocol": PROTOCOL,
        "calibration_stage": "mhcr_label_free_before_new_feature_endpoint_join",
        "freeze_sha256": input_hashes["freeze"],
        "eligible_optional_terms": eligible_terms,
        "excluded_features": ["all tau05/tau10/tau25 nested lower-coverage gains"],
        "single_weight_grid": list(SINGLE_WEIGHT_GRID),
        "pair_weight_grid": list(PAIR_WEIGHT_GRID),
        "guard_configurations": configurations,
        "configuration_count": len(configurations),
        "frontier_base_count": len(bases),
        "auc_safe11_base_count": auc_count,
        "safe12_base_count": safe_count,
        "frontier_base_key_sha256": base_key_sha,
        "frontier_base_formula_sha256": base_formula_sha,
        "configuration_id_sha256": configuration_sha,
        "candidate_count": len(specs),
        "candidate_key_sha256": candidate_key_sha,
        "optional_missing_policy": "OPTIONAL_GUARD_OFF_KEEP_BASE",
        "new_mhcr_features_joined_to_endpoint_before_freeze": False,
        "discovery_endpoints_previously_opened_by_next122": True,
    }
    label_free_catalogue_sha256 = hashlib.sha256(
        n121.prior._json_bytes(label_free_catalogue)
    ).hexdigest()
    if require_formal_inputs and (
        len(bases) != EXPECTED_FRONTIER_BASES
        or auc_count != EXPECTED_AUC_FRONTIER_BASES
        or safe_count != EXPECTED_SAFE_FRONTIER_BASES
        or base_key_sha != EXPECTED_BASE_CANDIDATE_KEY_SHA256
        or base_formula_sha != EXPECTED_BASE_FORMULA_SHA256
        or len(configurations) != EXPECTED_CONFIGURATION_COUNT
        or configuration_sha != EXPECTED_CONFIGURATION_ID_SHA256
        or len(specs) != EXPECTED_CANDIDATE_COUNT
        or candidate_key_sha != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT125 frozen candidate universe differs")

    scigen_endpoints = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_endpoints = pd.read_parquet(paths["wyformer_endpoint"])
    endpoint_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "material_id": "scigen:" + scigen_endpoints["material_id"].astype(str),
                    "_endpoint_numeric": pd.to_numeric(scigen_endpoints["distortion_ratio"], errors="coerce"),
                }
            ),
            pd.DataFrame(
                {
                    "material_id": "wyformer:" + wyformer_endpoints["material_id"].astype(str),
                    "_endpoint_numeric": n121.prior._endpoint_numeric(wyformer_endpoints["endpoint_stratum"]),
                }
            ),
        ],
        ignore_index=True,
    )
    combined = extended.merge(endpoint_frame, on="material_id", how="inner", validate="one_to_one")
    if len(combined) != len(extended) or len(combined) != len(endpoint_frame):
        raise ValueError("NEXT125 endpoint row accounting differs")
    endpoint = pd.to_numeric(combined.pop("_endpoint_numeric"), errors="coerce").to_numpy(float)
    if not np.isfinite(endpoint).all():
        raise ValueError("NEXT125 endpoint conversion differs")

    started = time.perf_counter()
    result = search_optional_guard_laws_parallel(
        features=combined,
        endpoint=endpoint,
        old_terms=old_terms,
        optional_terms=composite_terms,
        candidate_specs=specs,
        workers=search_workers,
    )
    elapsed = time.perf_counter() - started
    n121.prior._decorate_result(
        result=result,
        mapping=mapping,
        eligible_terms=eligible_terms,
    )
    verify_base_reproduction(result_records=result["candidate_records"], prior=bases)
    result["selected"]["formula"]["kind"] = "next122_frontier_base_plus_up_to_two_multiscale_hall_contact_terms"
    selected = result["selected"]
    passes = bool(selected["record"]["passes_all_discovery_gates"])

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next103_dobvr_optional_guard_search.py": repository_root / "src/next103_dobvr_optional_guard_search.py",
        "src/next117_hcid_frontier_rescue.py": repository_root / "src/next117_hcid_frontier_rescue.py",
        "src/next121_bvtbd_frontier_rescue.py": repository_root / "src/next121_bvtbd_frontier_rescue.py",
        "src/next122_safe12_bvtc_prlr_rescue.py": repository_root / "src/next122_safe12_bvtc_prlr_rescue.py",
        "src/next123_multiscale_hall_contact_robustness.py": repository_root / "src/next123_multiscale_hall_contact_robustness.py",
        "src/next124_cross_source_mhcr_features.py": repository_root / "src/next124_cross_source_mhcr_features.py",
        "src/next125_mhcr_frontier_rescue.py": Path(__file__).resolve(),
    }
    source_hashes = {name: n121.prior._sha256_file(path) for name, path in source_paths.items()}
    output_paths: list[Path] = []
    try:
        catalogue_path = staging / CATALOGUE_NAME
        evaluation_path = staging / EVALUATION_NAME
        search_path = staging / SEARCH_NAME
        n121.prior._write_json(
            catalogue_path,
            {**label_free_catalogue, "label_free_catalogue_sha256": label_free_catalogue_sha256},
        )
        n121.prior._write_json(
            evaluation_path,
            {
                "protocol": PROTOCOL,
                "evaluation_mode": "adaptive_cross_source_discovery_mhcr_frontier_rescue",
                "rows": {
                    "scigen": int(len(feature_tables["scigen"])),
                    "wyformer": int(len(feature_tables["wyformer"])),
                    "total": int(len(combined)),
                },
                "frontier_base_count": len(bases),
                "configuration_count": len(configurations),
                "candidate_count": int(result["candidate_count"]),
                "elapsed_seconds": elapsed,
                "search_workers": search_workers,
                "base_only_reproduced_next122": True,
                "safe_gates": dict(n121.prior.DEFAULT_GATES),
                "source_auc_gates": dict(n121.prior.AUC_GATES),
                "broad_min_severe_precision_lower": n121.prior.BROAD_MIN_PRECISION_LOWER,
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
            },
        )
        pd.DataFrame(result["candidate_records"]).to_parquet(search_path, index=False)
        output_paths.extend([catalogue_path, evaluation_path, search_path])
        manifest = {
            "protocol": PROTOCOL,
            "label_free_catalogue_sha256": label_free_catalogue_sha256,
            "frontier_base_count": len(bases),
            "configuration_count": len(configurations),
            "candidate_count": int(result["candidate_count"]),
            "search_workers": search_workers,
            "base_only_reproduced_next122": True,
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
            "outputs_sha256": {path.name: n121.prior._sha256_file(path) for path in output_paths},
        }
        n121.prior._write_json(staging / MANIFEST_NAME, manifest)
        if any(n121.prior._sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT125 input changed before publication")
        if any(n121.prior._sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT125 source changed before publication")
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
    for stage in (98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124):
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--freeze-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    args = parser.parse_args()
    manifest = run_mhcr_frontier_rescue(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{f"next{stage}_dir": getattr(args, f"next{stage}_dir") for stage in (98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124)},
        freeze_path=args.freeze_path,
        output_dir=args.output_dir,
        search_workers=args.search_workers,
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
    "build_frontier_candidate_specs",
    "build_mhcr_guard_configurations",
    "materialize_mhcr_composite_guard_terms",
    "materialize_mhcr_tail_terms",
    "run_mhcr_frontier_rescue",
    "search_optional_guard_laws_parallel",
    "select_frontier_bases",
]
