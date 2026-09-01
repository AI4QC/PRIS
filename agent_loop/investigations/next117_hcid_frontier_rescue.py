#!/usr/bin/env python3
"""Frozen generalized-Hall rescue of the NEXT114 AUC--SAFE frontier."""

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

from src.next85_scigen_label_free_features import FEATURE_NAMES as SCIGEN_FEATURE_NAMES
from src.next86_scigen_endpoint_router import ENDPOINT_NAME as SCIGEN_ENDPOINT_NAME
from src.next93_wyformer_source_lockbox import _sha256_file, _write_json
from src.next93b_wyformer_blind_lockbox import ENDPOINT_NAME as WYFORMER_ENDPOINT_NAME
from src.next94_wyformer_label_free_features import FEATURE_NAMES as WYFORMER_FEATURE_NAMES
from src.next95_wyformer_sparse_law_search import DEFAULT_GATES, _endpoint_numeric
from src.next98_cross_source_discovery_search import (
    AUC_GATES,
    BROAD_MIN_PRECISION_LOWER,
    CATALOGUE_NAME as NEXT98_CATALOGUE_NAME,
    MANIFEST_NAME as NEXT98_MANIFEST_NAME,
    PROTOCOL as NEXT98_PROTOCOL,
    _read_json,
)
from src.next103_dobvr_optional_guard_search import search_optional_guard_laws
from src.next107_two_axis_cmvf_guard_search import (
    _composite_term_id,
    _configuration_id,
    _decorate_result,
    materialize_composite_guard_terms,
)
from src.next110_cross_source_cmvo_features import (
    CATALOGUE_NAME as NEXT110_CATALOGUE_NAME,
    FEATURE_COLUMNS as NEXT110_FEATURE_COLUMNS,
    FEATURE_FILES as NEXT110_FEATURE_FILES,
    MANIFEST_NAME as NEXT110_MANIFEST_NAME,
    PROTOCOL as NEXT110_PROTOCOL,
)
from src.next111_cmvo_optional_search import (
    CATALOGUE_NAME as NEXT111_CATALOGUE_NAME,
    MANIFEST_NAME as NEXT111_MANIFEST_NAME,
    PROTOCOL as NEXT111_PROTOCOL,
    materialize_cmvo_tail_terms,
)
from src.next113_cross_source_cmvom_features import (
    CATALOGUE_NAME as NEXT113_CATALOGUE_NAME,
    FEATURE_COLUMNS as NEXT113_FEATURE_COLUMNS,
    FEATURE_FILES as NEXT113_FEATURE_FILES,
    MANIFEST_NAME as NEXT113_MANIFEST_NAME,
    PROTOCOL as NEXT113_PROTOCOL,
)
from src.next114_cmvom_frontier_rescue import (
    CATALOGUE_NAME as NEXT114_CATALOGUE_NAME,
    MANIFEST_NAME as NEXT114_MANIFEST_NAME,
    PROTOCOL as NEXT114_PROTOCOL,
    SEARCH_NAME as NEXT114_SEARCH_NAME,
    materialize_cmvom_tail_terms,
)
from src.next116_cross_source_hcid_features import (
    CATALOGUE_NAME as NEXT116_CATALOGUE_NAME,
    FEATURE_COLUMNS as NEXT116_FEATURE_COLUMNS,
    FEATURE_FILES as NEXT116_FEATURE_FILES,
    MANIFEST_NAME as NEXT116_MANIFEST_NAME,
    PROTOCOL as NEXT116_PROTOCOL,
)


PROTOCOL = "2026-08-08-next117-hcid-frontier-rescue-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT117_HCID_FRONTIER_CATALOGUE.json"
EVALUATION_NAME = "NEXT117_DISCOVERY_EVALUATION.json"
SEARCH_NAME = "next117_hcid_frontier_candidate_search.parquet"
EXPECTED_FREEZE_SHA256 = "121de88c03b4261f944a1a6665f2244218397cd9c8580ecab8c1353eed1443e1"
SINGLE_WEIGHT_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
PAIR_WEIGHT_GRID = (0.25, 0.5, 1.0, 2.0)
BASE_REPRODUCTION_AUC_TOLERANCE = 2.0e-5
EXPECTED_FRONTIER_BASES = 97
EXPECTED_AUC_FRONTIER_BASES = 90
EXPECTED_SAFE_FRONTIER_BASES = 7
EXPECTED_CONFIGURATION_COUNT = 116
EXPECTED_CANDIDATE_COUNT = 11_349
EXPECTED_BASE_CANDIDATE_KEY_SHA256 = (
    "05fd0f109e96818ecc2e0b929589e9a05b7e5447a8ad79cebff4f8a7609347ca"
)
EXPECTED_BASE_FORMULA_SHA256 = (
    "b48f33d59662b659073dbe6e6159384f9f1cbcb4b8b805b81dee40f3c4d5d0e6"
)
NUMERICAL_ZERO_TOLERANCE = 1.0e-12

FROZEN_TERM_SPECS = (
    {
        "term_id": "hcid_core_positive_local_density__high",
        "raw_key": "local_density",
        "group": "hcid_core",
        "center": 0.16666666666666663,
        "scale": 0.5419642857142857,
        "raw_p995": 0.9066315406976742,
        "clip_normalized": 1.3653388120506236,
        "nonzero_rows": 4173,
    },
    {
        "term_id": "hcid_core_positive_localization_gain__high",
        "raw_key": "localization_gain",
        "group": "hcid_core",
        "center": 0.0,
        "scale": 0.08333333333333337,
        "raw_p995": 0.056677350427349726,
        "clip_normalized": 0.6801282051281964,
        "nonzero_rows": 53,
    },
    {
        "term_id": "hcid_core_positive_origin_localization__high",
        "raw_key": "origin_localization",
        "group": "hcid_core",
        "center": 0.0,
        "scale": 0.25,
        "raw_p995": 0.1428571428571429,
        "clip_normalized": 0.5714285714285716,
        "nonzero_rows": 53,
    },
    {
        "term_id": "hcid_core_positive_neighbor_bottleneck__high",
        "raw_key": "neighbor_bottleneck",
        "group": "hcid_core",
        "center": 0.0,
        "scale": 0.5,
        "raw_p995": 0.5507575757575721,
        "clip_normalized": 1.1015151515151442,
        "nonzero_rows": 89,
    },
)
EXCLUDED_FEATURES = (
    "hcid_core_positive_global_deficit_as_standalone",
    "hcid_core_positive_origin_site_fraction_max",
    "hcid_core_origin_face_range",
    "hcid_core_negative_raw_terms",
    "hcid_expanded_raw_terms",
)

EXPECTED_INPUT_SHA256 = {
    "scigen_features": "7031d86e4fb6e469c674d208f680ace1dbe5e11e45f3d4b2befefd747efdde16",
    "scigen_endpoint": "f86cff6f5e9124ee82aae13911ffe55a125c6fe111fc1f64122a610febf67958",
    "wyformer_features": "c515baec0fccef5bc03c7672f1d4e1aca278f5ed4d7b6f1bf7f66c734e2b87f7",
    "wyformer_endpoint": "f39836e62a1da03ed823479e87d6f75fc0d01da60a8c0a2faa696638cc2fb9d7",
    "next98_manifest": "5fcd924b125767e52ac1826203595692af868ab35366899e12b82aea2726e32c",
    "next98_term_catalogue": "f2165f548a56cda04559a11a0d575f0654d3e8a17cf3b85b76e7974ea65dee41",
    "next110_manifest": "06000213e80de7afa2e13f9dd67561ff2b56a9a10fede90260c269ad57dc03b3",
    "next110_catalogue": "5a9e66a87779555f91019ac1873a5b2974154e51b2911986b3911c5d69b5ac01",
    "next110_scigen_features": "023c0662fabd73df0a7f47c1e10dc7e229fb0b5cde6f2d76c34c3c6efc1bb31e",
    "next110_wyformer_features": "fd225555c8cadd2219df6fec679c74c78a9a5c15065f23553d7e6d1eec681c94",
    "next111_manifest": "b832f9036be564288654eeba9afca6455bd6b78eb33bdedc3fe469f3b9a9bfc4",
    "next111_catalogue": "ca817af567caa123f87eb9e310baf491515d20179c9c4a1aab69fdcd4e52b392",
    "next113_manifest": "61c3fec7505d4216410da038ea9d85c28992d1d3fa6d230e2c90146d75b6f78c",
    "next113_catalogue": "530faa6f38289942e97dca010e8ccdc3a4ac5e59beadda3712db080d0383d212",
    "next113_scigen_features": "e6b4f1257dc50e2358c31d0eef5402cf8712e795594cc1b8df5cd7a825dcc692",
    "next113_wyformer_features": "aa45232d5c94fb43401e5aa804ed56c229a8630ea1c3642c984ac789546c3038",
    "next114_manifest": "50c899c2e3eca655dffadfd81a50dd1044006a968e3fb2b2e5434b1515f9578a",
    "next114_catalogue": "483f4fc75d70feee017050797ddfe1f7319c8959c9c96ad7c4870a97f91e7546",
    "next114_search_records": "0c08170251d70613c7268c8b04c3f0f929eed7699dbe9a8787ea8b0f21ceb4c8",
    "next116_manifest": "15e1b175f85ee7419b511beaaf4ad4feb72d7377a1dc52156665c8b708c7ab60",
    "next116_catalogue": "e12f1060f65b05469a0c20e365d6a9f6c2e93bf267dd319577d9fbf0718c5ebc",
    "next116_scigen_features": "f9c42a29c40389720dabeab3d66cf75064e57cc5945fcd54582bfa827eda62b2",
    "next116_wyformer_features": "934552c0d310b1b2dc5933a05cca9bc3903d38e94d5f559349086f835691555a",
    "freeze": EXPECTED_FREEZE_SHA256,
}


def _json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _formula_identity(term_ids: Sequence[str], weights: Sequence[float]) -> str:
    return json.dumps(
        {
            "term_ids": [str(value) for value in term_ids],
            "weights": [float(value) for value in weights],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def select_frontier_bases(records: pd.DataFrame) -> pd.DataFrame:
    """Freeze NEXT114's AUC-best-SAFE and all-SAFE conflict routes."""

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
        raise ValueError("NEXT117 frontier base schema differs")
    auc = records["passes_source_auc_gates"].fillna(False).astype(bool)
    safe_all = records["passes_safe_all_cells"].fillna(False).astype(bool)
    safe_cells = pd.to_numeric(records["safe_passing_cells"], errors="coerce")
    if not auc.any():
        raise ValueError("NEXT117 prior frontier has no AUC-passing base")
    auc_best_safe = int(safe_cells.loc[auc].max())
    keep = (auc & safe_cells.eq(auc_best_safe)) | safe_all
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
            len(term_ids) not in (4, 5, 6, 7)
            or len(term_ids) != len(weights)
            or len(set(term_ids)) != len(term_ids)
            or any(not math.isfinite(weight) or weight <= 0.0 for weight in weights)
        ):
            raise ValueError("NEXT117 flattened frontier formula differs")
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


def _derived_hcid_raw(features: pd.DataFrame) -> dict[str, np.ndarray]:
    required = {
        "hcid_core_supported",
        "hcid_core_positive_global_deficit",
        "hcid_core_positive_local_density",
        "hcid_core_positive_origin_site_fraction_min",
        "hcid_core_positive_neighbor_site_fraction_min",
    }
    if required - set(features.columns):
        raise ValueError("NEXT117 frozen HCID operand is missing")
    supported = features["hcid_core_supported"].fillna(False).astype(bool).to_numpy()
    global_deficit = pd.to_numeric(
        features["hcid_core_positive_global_deficit"], errors="coerce"
    ).to_numpy(float)
    local_density = pd.to_numeric(
        features["hcid_core_positive_local_density"], errors="coerce"
    ).to_numpy(float)
    origin_fraction = pd.to_numeric(
        features["hcid_core_positive_origin_site_fraction_min"], errors="coerce"
    ).to_numpy(float)
    neighbor_fraction = pd.to_numeric(
        features["hcid_core_positive_neighbor_site_fraction_min"], errors="coerce"
    ).to_numpy(float)
    active_obstruction = supported & np.isfinite(global_deficit) & (
        global_deficit > NUMERICAL_ZERO_TOLERANCE
    )
    raw = {
        "local_density": local_density.copy(),
        "localization_gain": np.maximum(0.0, local_density - global_deficit),
        "origin_localization": np.where(
            active_obstruction, np.maximum(0.0, 1.0 - origin_fraction), 0.0
        ),
        "neighbor_bottleneck": np.where(
            active_obstruction, np.maximum(0.0, 1.0 - neighbor_fraction), 0.0
        ),
    }
    for values in raw.values():
        values[np.isfinite(values) & (np.abs(values) <= NUMERICAL_ZERO_TOLERANCE)] = 0.0
        values[~supported] = np.nan
    return raw


def materialize_hcid_tail_terms(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Derive and reversibly encode the four frozen HCID high tails."""

    raw_values = _derived_hcid_raw(features)
    columns: dict[str, np.ndarray] = {}
    terms: list[dict[str, object]] = []
    supported = features["hcid_core_supported"].fillna(False).astype(bool).to_numpy()
    for raw_spec in FROZEN_TERM_SPECS:
        spec = dict(raw_spec)
        term_id = str(spec["term_id"])
        raw_key = str(spec["raw_key"])
        raw = raw_values[raw_key]
        center = float(spec["center"])
        scale = float(spec["scale"])
        cap = float(spec["clip_normalized"])
        if not all(math.isfinite(value) for value in (center, scale, cap)) or scale <= 0.0 or cap <= 0.0:
            raise ValueError("NEXT117 frozen HCID calibration differs")
        active = supported & np.isfinite(raw)
        tail = np.zeros(len(features), dtype=float)
        tail[active] = np.clip(
            np.maximum(0.0, (raw[active] - center) / scale),
            0.0,
            cap,
        )
        encoded = np.sinh(tail)
        if not np.isfinite(encoded).all():
            raise ValueError("NEXT117 reversible HCID encoding overflowed")
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
                "encoding": "asinh_sinh_exact_clipped_frozen_hcid_tail",
            }
        )
    return (
        pd.concat([features.reset_index(drop=True), pd.DataFrame(columns)], axis=1),
        terms,
    )


def build_hcid_guard_configurations(
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
        or any(term.get("group") != "hcid_core" for term in terms)
    ):
        raise ValueError("NEXT117 frozen HCID term identity differs")
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
        configuration_id = _configuration_id(components)
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


def build_frontier_candidate_specs(
    *,
    base_records: pd.DataFrame,
    old_term_ids: set[str],
    configurations: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Attach zero, one, or two frozen HCID terms to every frontier base."""

    required = {"prior_candidate_key", "term_ids_json", "weights_json"}
    if required - set(base_records.columns):
        raise ValueError("NEXT117 frontier candidate columns differ")
    configuration_ids = sorted(
        str(configuration["configuration_id"]) for configuration in configurations
    )
    specs: dict[str, dict[str, object]] = {}
    for _, row in base_records.iterrows():
        term_ids = [str(value) for value in json.loads(str(row["term_ids_json"]))]
        weights = [float(value) for value in json.loads(str(row["weights_json"]))]
        if (
            len(term_ids) not in (4, 5, 6, 7)
            or len(weights) != len(term_ids)
            or len(set(term_ids)) != len(term_ids)
            or any(term_id not in old_term_ids for term_id in term_ids)
            or any(not math.isfinite(weight) or weight <= 0.0 for weight in weights)
        ):
            raise ValueError("NEXT117 frontier base formula differs")

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
                else _composite_term_id(configuration_id),
                "optional_weight": 0.0 if configuration_id is None else 1.0,
                "optional_configuration_id": configuration_id,
            }

        add(None)
        for configuration_id in configuration_ids:
            add(configuration_id)
    return [specs[key] for key in sorted(specs)]


def _verify_base_reproduction(
    *, result_records: Sequence[Mapping[str, object]], prior: pd.DataFrame
) -> None:
    """Prove HCID materialization did not alter NEXT114 frontier bases."""

    observed: dict[str, Mapping[str, object]] = {}
    for record in result_records:
        if record.get("optional_configuration_id") is None:
            key = _formula_identity(
                json.loads(str(record["base_term_ids_json"])),
                json.loads(str(record["base_weights_json"])),
            )
            observed[key] = record
    metrics = (
        "scigen_pooled_auc",
        "scigen_macro_auc",
        "scigen_worst_auc",
        "wyformer_pooled_auc",
        "wyformer_macro_auc",
        "wyformer_worst_auc",
    )
    if len(observed) != len(prior):
        raise RuntimeError("NEXT117 base-only reproduction count differs")
    for _, row in prior.iterrows():
        term_ids = json.loads(str(row["term_ids_json"]))
        weights = json.loads(str(row["weights_json"]))
        key = _formula_identity(term_ids, weights)
        record = observed.get(key)
        if record is None:
            raise RuntimeError("NEXT117 base-only formula identity differs")
        source_row = row["_prior_record"]
        if any(
            not math.isclose(
                float(record[name]),
                float(source_row[name]),
                rel_tol=0.0,
                abs_tol=BASE_REPRODUCTION_AUC_TOLERANCE,
            )
            for name in metrics
        ) or any(
            bool(record[name]) != bool(source_row[name])
            for name in ("passes_source_auc_gates", "passes_safe_all_cells")
        ) or int(record["safe_passing_cells"]) != int(source_row["safe_passing_cells"]):
            raise RuntimeError("NEXT117 base-only diagnostics do not reproduce NEXT114")


def run_hcid_frontier_rescue(
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
    freeze_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen HCID rescue after hashing its label-free catalogue."""

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
    }
    target = Path(output_dir).resolve()
    paths = {
        "scigen_features": roots["scigen_features"] / SCIGEN_FEATURE_NAMES["discovery"],
        "scigen_endpoint": roots["scigen_endpoint"] / SCIGEN_ENDPOINT_NAME,
        "wyformer_features": roots["wyformer_features"] / WYFORMER_FEATURE_NAMES["discovery"],
        "wyformer_endpoint": roots["wyformer_endpoint"] / WYFORMER_ENDPOINT_NAME,
        "next98_manifest": roots["next98"] / NEXT98_MANIFEST_NAME,
        "next98_term_catalogue": roots["next98"] / NEXT98_CATALOGUE_NAME,
        "next110_manifest": roots["next110"] / NEXT110_MANIFEST_NAME,
        "next110_catalogue": roots["next110"] / NEXT110_CATALOGUE_NAME,
        "next110_scigen_features": roots["next110"] / NEXT110_FEATURE_FILES["scigen"],
        "next110_wyformer_features": roots["next110"] / NEXT110_FEATURE_FILES["wyformer"],
        "next111_manifest": roots["next111"] / NEXT111_MANIFEST_NAME,
        "next111_catalogue": roots["next111"] / NEXT111_CATALOGUE_NAME,
        "next113_manifest": roots["next113"] / NEXT113_MANIFEST_NAME,
        "next113_catalogue": roots["next113"] / NEXT113_CATALOGUE_NAME,
        "next113_scigen_features": roots["next113"] / NEXT113_FEATURE_FILES["scigen"],
        "next113_wyformer_features": roots["next113"] / NEXT113_FEATURE_FILES["wyformer"],
        "next114_manifest": roots["next114"] / NEXT114_MANIFEST_NAME,
        "next114_catalogue": roots["next114"] / NEXT114_CATALOGUE_NAME,
        "next114_search_records": roots["next114"] / NEXT114_SEARCH_NAME,
        "next116_manifest": roots["next116"] / NEXT116_MANIFEST_NAME,
        "next116_catalogue": roots["next116"] / NEXT116_CATALOGUE_NAME,
        "next116_scigen_features": roots["next116"] / NEXT116_FEATURE_FILES["scigen"],
        "next116_wyformer_features": roots["next116"] / NEXT116_FEATURE_FILES["wyformer"],
        "freeze": Path(freeze_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT117 discovery input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT117 formal input identity differs")

    manifests = {
        "next98": _read_json(paths["next98_manifest"]),
        "next110": _read_json(paths["next110_manifest"]),
        "next111": _read_json(paths["next111_manifest"]),
        "next113": _read_json(paths["next113_manifest"]),
        "next114": _read_json(paths["next114_manifest"]),
        "next116": _read_json(paths["next116_manifest"]),
    }
    if (
        manifests["next98"].get("protocol") != NEXT98_PROTOCOL
        or manifests["next110"].get("protocol") != NEXT110_PROTOCOL
        or manifests["next110"].get("labels_opened") is not False
        or manifests["next110"].get("endpoint_payloads_opened") is not False
        or manifests["next111"].get("protocol") != NEXT111_PROTOCOL
        or manifests["next111"].get("passes_all_cross_source_discovery_gates") is not False
        or manifests["next111"].get("opened_validation_outputs_used") is not False
        or manifests["next111"].get("scigen_replication_endpoint_opened") is not False
        or manifests["next111"].get("wyformer_replication_endpoint_opened") is not False
        or manifests["next113"].get("protocol") != NEXT113_PROTOCOL
        or manifests["next113"].get("labels_opened") is not False
        or manifests["next113"].get("validation_geometry_opened") is not False
        or manifests["next113"].get("replication_geometry_opened") is not False
        or manifests["next114"].get("protocol") != NEXT114_PROTOCOL
        or manifests["next114"].get("passes_all_cross_source_discovery_gates") is not False
        or manifests["next114"].get("opened_validation_outputs_used") is not False
        or manifests["next114"].get("scigen_replication_endpoint_opened") is not False
        or manifests["next114"].get("wyformer_replication_endpoint_opened") is not False
        or manifests["next116"].get("protocol") != NEXT116_PROTOCOL
        or manifests["next116"].get("labels_opened") is not False
        or manifests["next116"].get("endpoint_payloads_opened") is not False
        or manifests["next116"].get("validation_geometry_opened") is not False
        or manifests["next116"].get("replication_geometry_opened") is not False
        or manifests["next116"].get("dft_values_used_by_features") is not False
    ):
        raise ValueError("NEXT117 prior provenance differs")
    provenance_outputs = (
        (
            manifests["next110"],
            {
                NEXT110_CATALOGUE_NAME: "next110_catalogue",
                NEXT110_FEATURE_FILES["scigen"]: "next110_scigen_features",
                NEXT110_FEATURE_FILES["wyformer"]: "next110_wyformer_features",
            },
        ),
        (
            manifests["next111"],
            {NEXT111_CATALOGUE_NAME: "next111_catalogue"},
        ),
        (
            manifests["next113"],
            {
                NEXT113_CATALOGUE_NAME: "next113_catalogue",
                NEXT113_FEATURE_FILES["scigen"]: "next113_scigen_features",
                NEXT113_FEATURE_FILES["wyformer"]: "next113_wyformer_features",
            },
        ),
        (
            manifests["next114"],
            {
                NEXT114_CATALOGUE_NAME: "next114_catalogue",
                NEXT114_SEARCH_NAME: "next114_search_records",
            },
        ),
        (
            manifests["next116"],
            {
                NEXT116_CATALOGUE_NAME: "next116_catalogue",
                NEXT116_FEATURE_FILES["scigen"]: "next116_scigen_features",
                NEXT116_FEATURE_FILES["wyformer"]: "next116_wyformer_features",
            },
        ),
    )
    for manifest, expected in provenance_outputs:
        outputs = manifest.get("outputs_sha256")
        if not isinstance(outputs, Mapping) or any(
            outputs.get(filename) != input_hashes[key]
            for filename, key in expected.items()
        ):
            raise ValueError("NEXT117 prior output provenance differs")

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
    feature_tables: dict[str, pd.DataFrame] = {}
    for source in ("scigen", "wyformer"):
        tables = (
            old_tables[source],
            cmvo_tables[source],
            morphology_tables[source],
            hcid_tables[source],
        )
        if any(table["material_id"].astype(str).duplicated().any() for table in tables):
            raise ValueError(f"NEXT117 {source} feature identities are duplicated")
        if (
            set(NEXT110_FEATURE_COLUMNS) - set(cmvo_tables[source].columns)
            or set(NEXT113_FEATURE_COLUMNS) - set(morphology_tables[source].columns)
            or set(NEXT116_FEATURE_COLUMNS) - set(hcid_tables[source].columns)
        ):
            raise ValueError(f"NEXT117 {source} feature schema differs")
        merged = old_tables[source].merge(
            cmvo_tables[source].loc[:, ["material_id", *NEXT110_FEATURE_COLUMNS]],
            on="material_id",
            how="inner",
            validate="one_to_one",
        ).merge(
            morphology_tables[source].loc[:, ["material_id", *NEXT113_FEATURE_COLUMNS]],
            on="material_id",
            how="inner",
            validate="one_to_one",
        ).merge(
            hcid_tables[source].loc[:, ["material_id", *NEXT116_FEATURE_COLUMNS]],
            on="material_id",
            how="inner",
            validate="one_to_one",
        )
        if any(len(merged) != len(table) for table in tables):
            raise ValueError(f"NEXT117 {source} feature row accounting differs")
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

    cmvo_features, cmvo_terms = materialize_cmvo_tail_terms(feature_combined)
    morphology_features, morphology_terms = materialize_cmvom_tail_terms(cmvo_features)
    hcid_features, eligible_terms = materialize_hcid_tail_terms(morphology_features)
    configurations = build_hcid_guard_configurations(FROZEN_TERM_SPECS)
    extended_features, composite_terms, composite_mapping = materialize_composite_guard_terms(
        features=hcid_features,
        eligible_terms=eligible_terms,
        configurations=configurations,
    )
    old_catalogue = _read_json(paths["next98_term_catalogue"])
    old_terms_raw = old_catalogue.get("eligible_terms")
    next111_catalogue = _read_json(paths["next111_catalogue"])
    next114_catalogue = _read_json(paths["next114_catalogue"])
    if (
        not isinstance(old_terms_raw, list)
        or next111_catalogue.get("eligible_optional_terms") != cmvo_terms
        or next114_catalogue.get("eligible_optional_terms") != morphology_terms
    ):
        raise ValueError("NEXT117 prior term catalogue differs")
    old_terms = [*old_terms_raw, *cmvo_terms, *morphology_terms]
    old_term_ids = {str(term["term_id"]) for term in old_terms}
    if len(old_term_ids) != len(old_terms):
        raise ValueError("NEXT117 old term identities are duplicated")

    prior_records = pd.read_parquet(paths["next114_search_records"])
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
        _formula_identity(
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
    label_free_catalogue = {
        "protocol": PROTOCOL,
        "calibration_stage": "hcid_label_free_before_endpoint_join",
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
        "candidate_key_sha256": hashlib.sha256(
            "\n".join(spec["candidate_key"] for spec in specs).encode("utf-8")
        ).hexdigest(),
        "optional_missing_policy": "OPTIONAL_GUARD_OFF_KEEP_BASE",
        "new_hcid_features_joined_to_endpoint_before_freeze": False,
        "discovery_endpoints_previously_opened_by_next114": True,
    }
    label_free_catalogue_sha256 = hashlib.sha256(
        _json_bytes(label_free_catalogue)
    ).hexdigest()
    if require_formal_inputs and (
        len(frontier_bases) != EXPECTED_FRONTIER_BASES
        or auc_frontier_count != EXPECTED_AUC_FRONTIER_BASES
        or safe_frontier_count != EXPECTED_SAFE_FRONTIER_BASES
        or base_key_sha != EXPECTED_BASE_CANDIDATE_KEY_SHA256
        or base_formula_sha != EXPECTED_BASE_FORMULA_SHA256
        or len(configurations) != EXPECTED_CONFIGURATION_COUNT
        or len(specs) != EXPECTED_CANDIDATE_COUNT
    ):
        raise ValueError("NEXT117 frozen candidate universe differs")

    # The two discovery endpoints are read only after the HCID catalogue hash exists.
    scigen_endpoints = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_endpoints = pd.read_parquet(paths["wyformer_endpoint"])
    if (
        scigen_endpoints["material_id"].astype(str).duplicated().any()
        or wyformer_endpoints["material_id"].astype(str).duplicated().any()
    ):
        raise ValueError("NEXT117 discovery endpoint identities are duplicated")
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
                    "_endpoint_numeric": _endpoint_numeric(
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
        raise ValueError("NEXT117 endpoint row accounting differs")
    endpoint = pd.to_numeric(
        combined.pop("_endpoint_numeric"), errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(endpoint).all():
        raise ValueError("NEXT117 endpoint conversion differs")

    started = time.perf_counter()
    result = search_optional_guard_laws(
        features=combined,
        endpoint=endpoint,
        old_terms=old_terms,
        optional_terms=composite_terms,
        candidate_specs=specs,
    )
    elapsed = time.perf_counter() - started
    _decorate_result(
        result=result,
        mapping=composite_mapping,
        eligible_terms=eligible_terms,
    )
    _verify_base_reproduction(
        result_records=result["candidate_records"],
        prior=frontier_bases,
    )
    result["selected"]["formula"]["kind"] = (
        "next114_frontier_base_plus_up_to_two_generalized_hall_local_obstruction_terms"
    )
    selected = result["selected"]
    passes = bool(selected["record"]["passes_all_discovery_gates"])

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next87_scigen_sparse_law_search.py": repository_root / "src/next87_scigen_sparse_law_search.py",
        "src/next95_wyformer_sparse_law_search.py": repository_root / "src/next95_wyformer_sparse_law_search.py",
        "src/next98_cross_source_discovery_search.py": repository_root / "src/next98_cross_source_discovery_search.py",
        "src/next103_dobvr_optional_guard_search.py": repository_root / "src/next103_dobvr_optional_guard_search.py",
        "src/next107_two_axis_cmvf_guard_search.py": repository_root / "src/next107_two_axis_cmvf_guard_search.py",
        "src/next111_cmvo_optional_search.py": repository_root / "src/next111_cmvo_optional_search.py",
        "src/next114_cmvom_frontier_rescue.py": repository_root / "src/next114_cmvom_frontier_rescue.py",
        "src/next115_hall_cut_interval_deficit.py": repository_root / "src/next115_hall_cut_interval_deficit.py",
        "src/next116_cross_source_hcid_features.py": repository_root / "src/next116_cross_source_hcid_features.py",
        "src/next117_hcid_frontier_rescue.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    output_paths: list[Path] = []
    try:
        catalogue = {
            **label_free_catalogue,
            "label_free_catalogue_sha256": label_free_catalogue_sha256,
        }
        evaluation = {
            "protocol": PROTOCOL,
            "evaluation_mode": "adaptive_cross_source_discovery_hcid_frontier_rescue",
            "rows": {
                "scigen": int(len(feature_tables["scigen"])),
                "wyformer": int(len(feature_tables["wyformer"])),
                "total": int(len(combined)),
            },
            "frontier_base_count": len(frontier_bases),
            "configuration_count": len(configurations),
            "candidate_count": int(result["candidate_count"]),
            "elapsed_seconds": elapsed,
            "base_only_reproduced_next114": True,
            "safe_gates": dict(DEFAULT_GATES),
            "source_auc_gates": dict(AUC_GATES),
            "broad_min_severe_precision_lower": BROAD_MIN_PRECISION_LOWER,
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
        _write_json(catalogue_path, catalogue)
        _write_json(evaluation_path, evaluation)
        pd.DataFrame(result["candidate_records"]).to_parquet(search_path, index=False)
        output_paths.extend([catalogue_path, evaluation_path, search_path])
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "label_free_catalogue_sha256": label_free_catalogue_sha256,
            "frontier_base_count": len(frontier_bases),
            "configuration_count": len(configurations),
            "candidate_count": int(result["candidate_count"]),
            "base_only_reproduced_next114": True,
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "requires_unopened_internal_validation_before_claim": True,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
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
            raise RuntimeError("NEXT117 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT117 source changed before publication")
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
    parser.add_argument("--next98-dir", type=Path, required=True)
    parser.add_argument("--next110-dir", type=Path, required=True)
    parser.add_argument("--next111-dir", type=Path, required=True)
    parser.add_argument("--next113-dir", type=Path, required=True)
    parser.add_argument("--next114-dir", type=Path, required=True)
    parser.add_argument("--next116-dir", type=Path, required=True)
    parser.add_argument("--freeze-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = run_hcid_frontier_rescue(
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
        freeze_path=args.freeze_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
