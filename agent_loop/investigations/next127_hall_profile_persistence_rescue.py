#!/usr/bin/env python3
"""Frozen HPP rescue of the NEXT125 AUC+SAFE12 discovery frontier."""

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

import src.next125_mhcr_frontier_rescue as n125
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next103_dobvr_optional_guard_search import (
    _optional_term_risk,
    compose_optional_guard_score,
)
from src.next87_scigen_sparse_law_search import _term_risk
from src.next126_hall_profile_persistence import (
    CATALOGUE_NAME as NEXT126_CATALOGUE_NAME,
    FEATURE_FILES as NEXT126_FEATURE_FILES,
    FEATURE_NAME,
    MANIFEST_NAME as NEXT126_MANIFEST_NAME,
    OUTPUT_SUPPORT_COLUMN,
    PROTOCOL as NEXT126_PROTOCOL,
)


PROTOCOL = "2026-08-08-next127-hall-profile-persistence-rescue-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT127_HPP_RESCUE_CATALOGUE.json"
EVALUATION_NAME = "NEXT127_DISCOVERY_EVALUATION.json"
SEARCH_NAME = "next127_hpp_rescue_candidate_search.parquet"
EXPECTED_FREEZE_SHA256 = "b6a925292e9d8d6687bc44ad29bbc83d024ae1d6149a7cbfd697e8650e4d0297"
EXPECTED_BASE_COUNT = 260
EXPECTED_CANDIDATE_COUNT = 1_300
EXPECTED_BASE_KEY_SHA256 = "a83e10e4c2d6cca3f2ee3c6bb5ba77f3856983cc355259de771b081dcb802f2e"
EXPECTED_BASE_FORMULA_SHA256 = "3139fa905c68fd0321f5922996b14271249ada5c644a9392522f04c4dda4ab95"
EXPECTED_CANDIDATE_KEY_SHA256 = "21f1d580ed8d8a409420cee44ec6163d058cf9df642a8ebdb7107286a455ef2a"
EXPECTED_NEXT125_MANIFEST_SHA256 = "305b1a6044ee43b17a56edd8e7630819955328d35416fa5bd8c178eddf12dac9"
EXPECTED_NEXT126_MANIFEST_SHA256 = "a9dafbcb40a21dec881c60237f2bfc248fe57cdc1d680b417957bbb17e6ea0c4"
OPTIONAL_TERM_ID = "mhpp_expanded_negative_weak_contact_persistence__high"
OPTIONAL_WEIGHTS = (0.10, 0.25, 0.50, 1.00)
SEARCH_WORKERS = 12
BASE_REPRODUCTION_AUC_TOLERANCE = n125.BASE_REPRODUCTION_AUC_TOLERANCE


def select_next125_bases(records: pd.DataFrame) -> pd.DataFrame:
    """Flatten every published NEXT125 AUC+SAFE12 formula."""

    required = {
        "candidate_key",
        "base_term_ids_json",
        "base_weights_json",
        "optional_term_ids_json",
        "optional_weights_json",
        "passes_source_auc_gates",
        "passes_safe_all_cells",
    }
    if required - set(records.columns):
        raise ValueError("NEXT127 base schema differs")
    keep = records["passes_source_auc_gates"].fillna(False).astype(bool) & records[
        "passes_safe_all_cells"
    ].fillna(False).astype(bool)
    selected: list[dict[str, object]] = []
    for _, row in records.loc[keep].sort_values("candidate_key").iterrows():
        term_ids = [
            *[str(value) for value in json.loads(str(row["base_term_ids_json"]))],
            *[str(value) for value in json.loads(str(row["optional_term_ids_json"]))],
        ]
        weights = [
            *[float(value) for value in json.loads(str(row["base_weights_json"]))],
            *[float(value) for value in json.loads(str(row["optional_weights_json"]))],
        ]
        if (
            not term_ids
            or len(term_ids) != len(weights)
            or len(set(term_ids)) != len(term_ids)
            or any(not math.isfinite(value) or value <= 0.0 for value in weights)
        ):
            raise ValueError("NEXT127 flattened base formula differs")
        selected.append(
            {
                "prior_candidate_key": str(row["candidate_key"]),
                "term_ids_json": json.dumps(term_ids, separators=(",", ":")),
                "weights_json": json.dumps(weights, separators=(",", ":")),
                "_prior_record": row.to_dict(),
            }
        )
    result = pd.DataFrame(
        selected,
        columns=("prior_candidate_key", "term_ids_json", "weights_json", "_prior_record"),
    )
    if result["prior_candidate_key"].duplicated().any():
        raise ValueError("NEXT127 prior candidate identities are duplicated")
    return result.reset_index(drop=True)


def build_candidate_specs(
    *, bases: pd.DataFrame, old_term_ids: set[str]
) -> list[dict[str, object]]:
    """Attach no HPP term or one of four frozen HPP weights."""

    required = {"term_ids_json", "weights_json"}
    if required - set(bases.columns):
        raise ValueError("NEXT127 candidate base columns differ")
    specs: dict[str, dict[str, object]] = {}
    for _, row in bases.iterrows():
        term_ids = [str(value) for value in json.loads(str(row["term_ids_json"]))]
        weights = [float(value) for value in json.loads(str(row["weights_json"]))]
        if (
            not term_ids
            or len(term_ids) != len(weights)
            or len(set(term_ids)) != len(term_ids)
            or any(term_id not in old_term_ids for term_id in term_ids)
        ):
            raise ValueError("NEXT127 candidate base formula differs")
        for optional_term_id, optional_weight in (
            (None, 0.0),
            *((OPTIONAL_TERM_ID, weight) for weight in OPTIONAL_WEIGHTS),
        ):
            payload = {
                "base_term_ids": term_ids,
                "base_weights": weights,
                "optional_term_id": optional_term_id,
                "optional_weight": optional_weight,
            }
            key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            specs[key] = {"candidate_key": key, **payload}
    return [specs[key] for key in sorted(specs)]


def materialize_virtual_bases(
    *,
    features: pd.DataFrame,
    bases: pd.DataFrame,
    old_terms: Sequence[Mapping[str, object]],
    mhcr_terms: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, str]]:
    """Encode each nested fail-open NEXT125 law as one exact base channel."""

    old_by_id = {str(term["term_id"]): dict(term) for term in old_terms}
    mhcr_by_id = {str(term["term_id"]): dict(term) for term in mhcr_terms}
    if len(old_by_id) != len(old_terms) or len(mhcr_by_id) != len(mhcr_terms):
        raise ValueError("NEXT127 virtual-base term identities are duplicated")
    old_risk = {term_id: _term_risk(features, term) for term_id, term in old_by_id.items()}
    mhcr_risk = {
        term_id: _optional_term_risk(features, term)
        for term_id, term in mhcr_by_id.items()
    }
    columns: dict[str, np.ndarray] = {}
    virtual_terms: list[dict[str, object]] = []
    mapping: dict[str, str] = {}
    for _, row in bases.iterrows():
        source = row["_prior_record"]
        base_ids = [str(value) for value in json.loads(str(source["base_term_ids_json"]))]
        base_weights = [float(value) for value in json.loads(str(source["base_weights_json"]))]
        guard_ids = [str(value) for value in json.loads(str(source["optional_term_ids_json"]))]
        guard_weights = [float(value) for value in json.loads(str(source["optional_weights_json"]))]
        if (
            not base_ids
            or len(base_ids) != len(base_weights)
            or len(guard_ids) not in (1, 2)
            or len(guard_ids) != len(guard_weights)
            or any(term_id not in old_by_id for term_id in base_ids)
            or any(term_id not in mhcr_by_id for term_id in guard_ids)
        ):
            raise ValueError("NEXT127 nested base formula differs")
        base_score = np.sum(
            np.column_stack([old_risk[term_id][0] for term_id in base_ids])
            * np.asarray(base_weights, dtype=float)[None, :],
            axis=1,
        )
        base_supported = np.all(
            np.column_stack([old_risk[term_id][1] for term_id in base_ids]),
            axis=1,
        )
        guard_score = np.sum(
            np.column_stack([mhcr_risk[term_id][0] for term_id in guard_ids])
            * np.asarray(guard_weights, dtype=float)[None, :],
            axis=1,
        )
        guard_active = np.all(
            np.column_stack([mhcr_risk[term_id][1] for term_id in guard_ids]),
            axis=1,
        )
        score, supported = compose_optional_guard_score(
            base_score=base_score,
            base_supported=base_supported,
            guard_risk=guard_score,
            guard_active=guard_active,
            guard_weight=1.0,
        )
        maximum = float(np.max(score[supported])) if supported.any() else 0.0
        divisor = max(1.0, maximum / 50.0)
        encoded = np.full(len(features), np.nan, dtype=float)
        encoded[supported] = np.sinh(score[supported] / divisor)
        if not np.isfinite(encoded[supported]).all():
            raise ValueError("NEXT127 virtual-base encoding overflowed")
        digest = hashlib.sha256(str(row["prior_candidate_key"]).encode()).hexdigest()[:24]
        term_id = f"next127_virtual_base__{digest}"
        feature_name = f"_{term_id}_value"
        formula_identity = _formula_identity(
            json.loads(str(row["term_ids_json"])),
            json.loads(str(row["weights_json"])),
        )
        if formula_identity in mapping or term_id in {term["term_id"] for term in virtual_terms}:
            raise ValueError("NEXT127 virtual-base identity collision")
        columns[feature_name] = encoded
        virtual_terms.append(
            {
                "term_id": term_id,
                "feature": feature_name,
                "direction": 1,
                "transform": "asinh",
                "center": 0.0,
                "scale": 1.0 / divisor,
                "group": "next125_nested_fail_open_base",
                "encoding": "asinh_sinh_exact_nested_optional_guard_score",
                "prior_candidate_key": str(row["prior_candidate_key"]),
            }
        )
        mapping[formula_identity] = term_id
    return (
        pd.concat([features.reset_index(drop=True), pd.DataFrame(columns)], axis=1),
        virtual_terms,
        mapping,
    )


def verify_base_reproduction(
    *, result_records: Sequence[Mapping[str, object]], prior: pd.DataFrame
) -> None:
    """Prove every pure base reproduces its published NEXT125 diagnostics."""

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
        if record.get("optional_term_id") is not None:
            continue
        payload = json.loads(str(record["candidate_key"]))
        key = n125.n121.prior._formula_identity(
            payload["base_term_ids"],
            payload["base_weights"],
        )
        if key in observed:
            raise RuntimeError("NEXT127 base-only formula identities are duplicated")
        observed[key] = record
    expected: dict[str, Mapping[str, object]] = {}
    for _, row in prior.iterrows():
        key = n125.n121.prior._formula_identity(
            json.loads(str(row["term_ids_json"])),
            json.loads(str(row["weights_json"])),
        )
        if key in expected:
            raise RuntimeError("NEXT127 prior base formula identities are duplicated")
        expected[key] = row
    if set(observed) != set(expected):
        raise RuntimeError("NEXT127 base-only reproduction identities differ")
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
            raise RuntimeError("NEXT127 base-only diagnostics do not reproduce NEXT125")


def _formula_identity(term_ids: Sequence[str], weights: Sequence[float]) -> str:
    return json.dumps(
        {"term_ids": list(term_ids), "weights": list(weights)},
        sort_keys=True,
        separators=(",", ":"),
    )


def _paths(roots: Mapping[str, Path], freeze_path: Path) -> dict[str, Path]:
    paths = n125._paths(roots, freeze_path)
    paths.update(
        {
            "next125_manifest": roots["next125"] / n125.MANIFEST_NAME,
            "next125_catalogue": roots["next125"] / n125.CATALOGUE_NAME,
            "next125_evaluation": roots["next125"] / n125.EVALUATION_NAME,
            "next125_search_records": roots["next125"] / n125.SEARCH_NAME,
            "next126_manifest": roots["next126"] / NEXT126_MANIFEST_NAME,
            "next126_catalogue": roots["next126"] / NEXT126_CATALOGUE_NAME,
            "next126_scigen_features": roots["next126"] / NEXT126_FEATURE_FILES["scigen"],
            "next126_wyformer_features": roots["next126"] / NEXT126_FEATURE_FILES["wyformer"],
        }
    )
    return paths


def run_hall_profile_persistence_rescue(
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
    next126_dir: Path,
    freeze_path: Path,
    output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen 1,300-candidate HPP discovery search."""

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
                (126, next126_dir),
            )
        },
    }
    target = Path(output_dir).resolve()
    paths = _paths(roots, freeze_path)
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT127 discovery input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if input_hashes["freeze"] != EXPECTED_FREEZE_SHA256:
        raise ValueError("NEXT127 freeze identity differs")

    manifest125 = json.loads(paths["next125_manifest"].read_text())
    manifest126 = json.loads(paths["next126_manifest"].read_text())
    if (
        input_hashes["next125_manifest"] != EXPECTED_NEXT125_MANIFEST_SHA256
        or input_hashes["next126_manifest"] != EXPECTED_NEXT126_MANIFEST_SHA256
        or manifest125.get("protocol") != n125.PROTOCOL
        or manifest125.get("passes_all_cross_source_discovery_gates") is not False
        or manifest125.get("opened_validation_outputs_used") is not False
        or manifest125.get("scigen_replication_endpoint_opened") is not False
        or manifest125.get("wyformer_replication_endpoint_opened") is not False
        or manifest125.get("dft_values_used_by_executable_formula") is not False
        or manifest126.get("protocol") != NEXT126_PROTOCOL
        or manifest126.get("labels_opened") is not False
        or manifest126.get("endpoint_payloads_opened") is not False
        or manifest126.get("validation_geometry_opened") is not False
        or manifest126.get("replication_geometry_opened") is not False
        or manifest126.get("dft_values_used_by_features") is not False
    ):
        raise ValueError("NEXT127 prior provenance differs")
    prior_inputs = manifest125.get("inputs_sha256")
    if not isinstance(prior_inputs, Mapping) or any(
        input_hashes.get(name) != value
        for name, value in prior_inputs.items()
        if name != "freeze"
    ):
        raise ValueError("NEXT127 inherited input identity differs")
    for manifest, expected in (
        (
            manifest125,
            {
                n125.CATALOGUE_NAME: "next125_catalogue",
                n125.EVALUATION_NAME: "next125_evaluation",
                n125.SEARCH_NAME: "next125_search_records",
            },
        ),
        (
            manifest126,
            {
                NEXT126_CATALOGUE_NAME: "next126_catalogue",
                NEXT126_FEATURE_FILES["scigen"]: "next126_scigen_features",
                NEXT126_FEATURE_FILES["wyformer"]: "next126_wyformer_features",
            },
        ),
    ):
        outputs = manifest.get("outputs_sha256")
        if not isinstance(outputs, Mapping) or any(
            outputs.get(filename) != input_hashes[key] for filename, key in expected.items()
        ):
            raise ValueError("NEXT127 prior output identity differs")

    features, feature_tables, old_terms = n125.prior._reconstruct_label_free_table(paths)
    mhcr_frames: list[pd.DataFrame] = []
    retained = sorted(
        {str(spec["raw_feature"]) for spec in n125.FROZEN_TERM_SPECS}
        | {str(spec["support_column"]) for spec in n125.FROZEN_TERM_SPECS}
    )
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next124_{source}_features"])
        frame = table.loc[:, ["material_id", *retained]].copy()
        frame["material_id"] = source + ":" + frame["material_id"].astype(str)
        mhcr_frames.append(frame)
    mhcr = pd.concat(mhcr_frames, ignore_index=True, sort=False)
    joined = features.merge(mhcr, on="material_id", how="inner", validate="one_to_one")
    if len(joined) != len(features) or len(joined) != len(mhcr):
        raise ValueError("NEXT127 MHCR row accounting differs")
    extended, mhcr_terms = n125.materialize_mhcr_tail_terms(joined)

    hpp_frames: list[pd.DataFrame] = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next126_{source}_features"])
        if table["material_id"].astype(str).duplicated().any():
            raise ValueError("NEXT127 HPP material identities are duplicated")
        frame = table.loc[:, ["material_id", FEATURE_NAME, OUTPUT_SUPPORT_COLUMN]].copy()
        frame["material_id"] = source + ":" + frame["material_id"].astype(str)
        hpp_frames.append(frame)
    hpp = pd.concat(hpp_frames, ignore_index=True, sort=False)
    extended = extended.merge(hpp, on="material_id", how="inner", validate="one_to_one")
    if len(extended) != len(features) or len(hpp) != len(features):
        raise ValueError("NEXT127 HPP row accounting differs")
    hpp_values = pd.to_numeric(extended[FEATURE_NAME], errors="coerce").to_numpy(float)
    hpp_supported = extended[OUTPUT_SUPPORT_COLUMN].eq(True).to_numpy()
    if (
        not np.isfinite(hpp_values[hpp_supported]).all()
        or np.any(hpp_values[hpp_supported] < 0.0)
        or np.any(hpp_values[hpp_supported] > 1.0)
    ):
        raise ValueError("NEXT127 HPP values differ")
    optional_term = {
        "term_id": OPTIONAL_TERM_ID,
        "feature": FEATURE_NAME,
        "raw_feature": FEATURE_NAME,
        "direction": 1,
        "transform": "asinh",
        "center": 0.0,
        "scale": 1.0,
        "clip_normalized": 1.0,
        "group": "mhpp_expanded_negative",
        "support_column": OUTPUT_SUPPORT_COLUMN,
        "missing_policy": "OPTIONAL_GUARD_OFF_KEEP_BASE",
    }
    all_old_terms = [*old_terms, *mhcr_terms]
    old_term_ids = {str(term["term_id"]) for term in all_old_terms}
    if len(old_term_ids) != len(all_old_terms) or OPTIONAL_TERM_ID in old_term_ids:
        raise ValueError("NEXT127 inherited term identities differ")

    prior_records = pd.read_parquet(paths["next125_search_records"])
    bases = select_next125_bases(prior_records)
    specs = build_candidate_specs(bases=bases, old_term_ids=old_term_ids)
    extended, virtual_terms, virtual_by_formula = materialize_virtual_bases(
        features=extended,
        bases=bases,
        old_terms=old_terms,
        mhcr_terms=mhcr_terms,
    )
    runtime_specs: list[dict[str, object]] = []
    for spec in specs:
        formula_identity = _formula_identity(
            spec["base_term_ids"], spec["base_weights"]
        )
        virtual_term_id = virtual_by_formula.get(formula_identity)
        if virtual_term_id is None:
            raise RuntimeError("NEXT127 runtime base mapping is incomplete")
        runtime_specs.append(
            {
                **spec,
                "base_term_ids": [virtual_term_id],
                "base_weights": [1.0],
            }
        )
    base_keys = sorted(bases["prior_candidate_key"].astype(str))
    base_formulas = sorted(
        _formula_identity(
            json.loads(str(row["term_ids_json"])),
            json.loads(str(row["weights_json"])),
        )
        for _, row in bases.iterrows()
    )
    base_key_sha = hashlib.sha256("\n".join(base_keys).encode()).hexdigest()
    base_formula_sha = hashlib.sha256("\n".join(base_formulas).encode()).hexdigest()
    candidate_key_sha = hashlib.sha256(
        "\n".join(spec["candidate_key"] for spec in specs).encode()
    ).hexdigest()
    if require_formal_inputs and (
        len(bases) != EXPECTED_BASE_COUNT
        or len(set(base_formulas)) != len(bases)
        or len(specs) != EXPECTED_CANDIDATE_COUNT
        or base_key_sha != EXPECTED_BASE_KEY_SHA256
        or base_formula_sha != EXPECTED_BASE_FORMULA_SHA256
        or candidate_key_sha != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT127 frozen candidate universe differs")

    scigen_endpoints = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_endpoints = pd.read_parquet(paths["wyformer_endpoint"])
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
                    "_endpoint_numeric": n125.n121.prior._endpoint_numeric(
                        wyformer_endpoints["endpoint_stratum"]
                    ),
                }
            ),
        ],
        ignore_index=True,
    )
    combined = extended.merge(endpoint_frame, on="material_id", how="inner", validate="one_to_one")
    if len(combined) != len(extended) or len(combined) != len(endpoint_frame):
        raise ValueError("NEXT127 endpoint row accounting differs")
    endpoint = pd.to_numeric(combined.pop("_endpoint_numeric"), errors="coerce").to_numpy(float)
    if not np.isfinite(endpoint).all():
        raise ValueError("NEXT127 endpoint conversion differs")

    started = time.perf_counter()
    result = n125.search_optional_guard_laws_parallel(
        features=combined,
        endpoint=endpoint,
        old_terms=virtual_terms,
        optional_terms=[optional_term],
        candidate_specs=runtime_specs,
        workers=search_workers,
    )
    elapsed = time.perf_counter() - started
    verify_base_reproduction(result_records=result["candidate_records"], prior=bases)
    physical_term_by_id = {str(term["term_id"]): dict(term) for term in all_old_terms}

    def decorate_record(record: dict[str, object]) -> None:
        payload = json.loads(str(record["candidate_key"]))
        evaluated_ids = json.loads(str(record["base_term_ids_json"]))
        record["evaluation_virtual_base_term_id"] = str(evaluated_ids[0])
        record["base_term_ids_json"] = json.dumps(
            payload["base_term_ids"], separators=(",", ":")
        )
        record["base_weights_json"] = json.dumps(
            payload["base_weights"], separators=(",", ":")
        )
        record["physical_base_term_count"] = len(payload["base_term_ids"])
        record["physical_term_count"] = len(payload["base_term_ids"]) + int(
            record.get("optional_term_id") is not None
        )

    for record in result["candidate_records"]:
        decorate_record(record)
    selected_record = result["selected"]["record"]
    if "evaluation_virtual_base_term_id" not in selected_record:
        decorate_record(selected_record)
    selected_payload = json.loads(str(selected_record["candidate_key"]))
    selected_formula = result["selected"]["formula"]
    selected_formula["evaluation_virtual_base_term_id"] = str(
        selected_formula["base_terms"][0]["term_id"]
    )
    selected_formula["base_terms"] = [
        {**physical_term_by_id[str(term_id)], "weight": float(weight)}
        for term_id, weight in zip(
            selected_payload["base_term_ids"],
            selected_payload["base_weights"],
            strict=True,
        )
    ]
    selected_formula["nested_mhcr_missing_policy"] = "OPTIONAL_GUARD_OFF_KEEP_PRE_MHCR_BASE"
    selected_formula["kind"] = "next125_auc_safe12_base_plus_optional_hall_profile_persistence"
    selected = result["selected"]
    passes = bool(selected["record"]["passes_all_discovery_gates"])
    label_free_catalogue = {
        "protocol": PROTOCOL,
        "freeze_sha256": input_hashes["freeze"],
        "feature_name": FEATURE_NAME,
        "optional_term": optional_term,
        "weight_grid": list(OPTIONAL_WEIGHTS),
        "frontier_base_count": len(bases),
        "candidate_count": len(specs),
        "frontier_base_key_sha256": base_key_sha,
        "frontier_base_formula_sha256": base_formula_sha,
        "candidate_key_sha256": candidate_key_sha,
        "optional_missing_policy": "OPTIONAL_GUARD_OFF_KEEP_BASE",
        "new_hpp_feature_joined_to_endpoint_before_freeze": False,
    }
    label_free_catalogue_sha = hashlib.sha256(
        json.dumps(label_free_catalogue, indent=2, sort_keys=True).encode() + b"\n"
    ).hexdigest()

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next103_dobvr_optional_guard_search.py": repository_root / "src/next103_dobvr_optional_guard_search.py",
        "src/next125_mhcr_frontier_rescue.py": repository_root / "src/next125_mhcr_frontier_rescue.py",
        "src/next126_hall_profile_persistence.py": repository_root / "src/next126_hall_profile_persistence.py",
        "src/next127_hall_profile_persistence_rescue.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    output_paths: list[Path] = []
    try:
        catalogue_path = staging / CATALOGUE_NAME
        evaluation_path = staging / EVALUATION_NAME
        search_path = staging / SEARCH_NAME
        _write_json(
            catalogue_path,
            {**label_free_catalogue, "label_free_catalogue_sha256": label_free_catalogue_sha},
        )
        _write_json(
            evaluation_path,
            {
                "protocol": PROTOCOL,
                "evaluation_mode": "fixed_hall_profile_persistence_broad_rescue",
                "rows": {
                    "scigen": int(len(feature_tables["scigen"])),
                    "wyformer": int(len(feature_tables["wyformer"])),
                    "total": int(len(combined)),
                },
                "frontier_base_count": len(bases),
                "candidate_count": int(result["candidate_count"]),
                "elapsed_seconds": elapsed,
                "search_workers": search_workers,
                "base_only_reproduced_next125": True,
                "safe_gates": dict(n125.n121.prior.DEFAULT_GATES),
                "source_auc_gates": dict(n125.n121.prior.AUC_GATES),
                "broad_min_severe_precision_lower": n125.n121.prior.BROAD_MIN_PRECISION_LOWER,
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
            "label_free_catalogue_sha256": label_free_catalogue_sha,
            "frontier_base_count": len(bases),
            "candidate_count": int(result["candidate_count"]),
            "search_workers": search_workers,
            "base_only_reproduced_next125": True,
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
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT127 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT127 source changed before publication")
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
    for stage in (98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124, 125, 126):
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--freeze-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    args = parser.parse_args()
    manifest = run_hall_profile_persistence_rescue(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{
            f"next{stage}_dir": getattr(args, f"next{stage}_dir")
            for stage in (98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124, 125, 126)
        },
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
    "OPTIONAL_TERM_ID",
    "build_candidate_specs",
    "run_hall_profile_persistence_rescue",
    "select_next125_bases",
    "verify_base_reproduction",
]
