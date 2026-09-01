#!/usr/bin/env python3
"""Frozen one-parameter homotopy from summed to top-two-trimmed physical risk."""

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
import src.next132_extended_coordination_protection_search as n132
import src.next135_conjunctive_compactness_search as n135
import src.next152_trimmed_joint_base_search as n152
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next87_scigen_sparse_law_search import _term_risk


PROTOCOL = "2026-08-08-next153-top2-attenuation-homotopy-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT153_TOP2_ATTENUATION_HOMOTOPY_CATALOGUE.json"
EVALUATION_NAME = "NEXT153_DISCOVERY_EVALUATION.json"
SEARCH_NAME = "next153_top2_attenuation_homotopy_candidate_search.parquet"
EXPECTED_DESIGN_SHA256 = "2f72079a4a259e1b054bb245eda764afb264266af875b52fd81990462a2f9664"
EXPECTED_BASE_COUNT = 11
EXPECTED_CANDIDATE_COUNT = 66
EXPECTED_BASE_FORMULA_SHA256 = n152.EXPECTED_BASE_FORMULA_SHA256
EXPECTED_CANDIDATE_KEY_SHA256 = "6a7047514b4926a514b9fdb39c55ab594009444c03f3cc331201aeaec4dd8c31"
AGGREGATION = "top2_attenuated_sum"
GAMMAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
PACKING_WEIGHT = 0.5
SEARCH_WORKERS = 4
EXPECTED_INPUT_SHA256 = {
    **n135.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next135_manifest": "f77fef70a1c01b6e335c7938556813cbfcb42d3ac5b605f662bf48e0d772397c",
    "next135_catalogue": "beee64d4c0ab86a3a0038b86e0a9dec7846fcb0ebaa5dbef4df66fad25aaafac",
    "next135_evaluation": "985fa3f3dd8fb9a2395c86bbe467885e3c5523750a032fc3a749969c14d87488",
    "next135_search": "a60fdc4a1611f54303cdc484f0792e4de7cf7808c1efe9149f4b898d2bc3b35b",
    "next152_manifest": "9470a71d7988b9cb9c34605c45cdbc19f1a1720a6b50d95bb93bf99b6e6442ee",
    "next152_catalogue": "32c4e6254b65dabae42c2d277077ca253c1353650fd5a0ff412396aed1b52dae",
    "next152_evaluation": "3d053e5bc2c20fcafacf60307b085f5662c0063de5d719d83735859aa3063486",
    "next152_search": "5aff039487c7205da39b9b6c5989e7a1c331d65d777891d78cd62eed5d77f445",
}


def top2_attenuated_sum(contributions: object, gamma: float) -> np.ndarray:
    """Attenuate, rather than discard, the two largest row contributions."""

    values = np.asarray(contributions, dtype=float)
    attenuation = float(gamma)
    if (
        values.ndim != 2
        or values.shape[1] < 3
        or np.any(~np.isfinite(values))
        or np.any(values < -1.0e-12)
        or not math.isfinite(attenuation)
        or attenuation < 0.0
        or attenuation > 1.0
    ):
        raise ValueError("NEXT153 attenuated contribution matrix differs")
    values = np.maximum(values, 0.0)
    largest = np.partition(values, kth=values.shape[1] - 2, axis=1)[:, -2:]
    return np.maximum(
        0.0, values.sum(axis=1) - attenuation * largest.sum(axis=1)
    )


def compose_homotopy_score(
    *,
    contributions: object,
    term_support: object,
    gamma: float,
    coordination_protection: object,
    coordination_active: object,
    packing_protection: object,
    packing_active: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen gamma-dependent coordination and fixed packing protection."""

    values = np.asarray(contributions, dtype=float)
    supports = np.asarray(term_support, dtype=bool)
    coordination = np.asarray(coordination_protection, dtype=float)
    coordination_on = np.asarray(coordination_active, dtype=bool)
    packing = np.asarray(packing_protection, dtype=float)
    packing_on = np.asarray(packing_active, dtype=bool)
    attenuation = float(gamma)
    if (
        values.ndim != 2
        or values.shape[1] < 3
        or supports.shape != values.shape
        or coordination.shape != (len(values),)
        or coordination_on.shape != (len(values),)
        or packing.shape != (len(values),)
        or packing_on.shape != (len(values),)
        or attenuation not in GAMMAS
    ):
        raise ValueError("NEXT153 score arrays differ")
    supported = supports.all(axis=1)
    if (
        np.any(supported[:, None] & (~np.isfinite(values) | (values < -1.0e-12)))
        or np.any(
            coordination_on
            & (~np.isfinite(coordination) | (coordination < -1.0e-12))
        )
        or np.any(packing_on & (~np.isfinite(packing) | (packing < -1.0e-12)))
    ):
        raise ValueError("NEXT153 supported score values differ")
    result = np.full(len(values), np.nan, dtype=float)
    if supported.any():
        result[supported] = top2_attenuated_sum(values[supported], attenuation)
    alpha = 2.0 * (1.0 - attenuation)
    active = supported & coordination_on
    result[active] -= alpha * coordination[active]
    active = supported & packing_on
    result[active] -= PACKING_WEIGHT * packing[active]
    result[supported] = np.maximum(0.0, result[supported])
    return result, supported


def build_candidate_specs(
    *, bases: pd.DataFrame, physical_term_ids: set[str]
) -> list[dict[str, object]]:
    specs: dict[str, dict[str, object]] = {}
    for _, row in bases.iterrows():
        term_ids = [str(value) for value in json.loads(str(row["term_ids_json"]))]
        weights = [float(value) for value in json.loads(str(row["weights_json"]))]
        if (
            len(term_ids) < 3
            or len(term_ids) != len(weights)
            or len(set(term_ids)) != len(term_ids)
            or any(term_id not in physical_term_ids for term_id in term_ids)
            or any(not math.isfinite(weight) or weight <= 0.0 for weight in weights)
        ):
            raise ValueError("NEXT153 physical base formula differs")
        for gamma in GAMMAS:
            coordination_weight = 2.0 * (1.0 - gamma)
            payload = {
                "aggregation": AGGREGATION,
                "top2_attenuation": gamma,
                "base_term_ids": term_ids,
                "base_weights": weights,
                "coordination_protection_term_id": (
                    None
                    if coordination_weight == 0.0
                    else n130.PROTECTION_TERM_ID
                ),
                "coordination_protection_weight": coordination_weight,
                "conjunctive_term_ids": [n135.PACKING_PRODUCT_TERM_ID],
                "conjunctive_weights": [PACKING_WEIGHT],
            }
            key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            specs[key] = {"candidate_key": key, **payload}
    return [specs[key] for key in sorted(specs)]


def materialize_candidates(
    *,
    features: pd.DataFrame,
    physical_terms: Sequence[Mapping[str, object]],
    specs: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    physical_by_id = {str(term["term_id"]): dict(term) for term in physical_terms}
    if len(physical_by_id) != len(physical_terms):
        raise ValueError("NEXT153 physical term identities are duplicated")
    required = {
        n130.n129.FEATURE_NAME,
        n130.n129.SUPPORT_COLUMN,
        n135.PACKING_PRODUCT_FEATURE,
        n135.PACKING_PRODUCT_SUPPORT,
    }
    if required - set(features.columns):
        raise ValueError("NEXT153 protection feature schema differs")
    coordination = pd.to_numeric(
        features[n130.n129.FEATURE_NAME], errors="coerce"
    ).to_numpy(float)
    coordination_active = features[n130.n129.SUPPORT_COLUMN].eq(True).to_numpy()
    packing = pd.to_numeric(
        features[n135.PACKING_PRODUCT_FEATURE], errors="coerce"
    ).to_numpy(float)
    packing_active = features[n135.PACKING_PRODUCT_SUPPORT].eq(True).to_numpy()
    risk_cache = {
        term_id: _term_risk(features, term)
        for term_id, term in physical_by_id.items()
    }
    columns: dict[str, np.ndarray] = {}
    virtual_terms: list[dict[str, object]] = []
    runtime: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_spec in specs:
        spec = dict(raw_spec)
        key = str(spec["candidate_key"])
        term_ids = [str(value) for value in spec["base_term_ids"]]
        weights = [float(value) for value in spec["base_weights"]]
        if (
            key in seen
            or spec.get("aggregation") != AGGREGATION
            or len(term_ids) != len(weights)
            or any(term_id not in risk_cache for term_id in term_ids)
        ):
            raise ValueError("NEXT153 candidate materialization differs")
        seen.add(key)
        contribution_columns = []
        support_columns = []
        for term_id, weight in zip(term_ids, weights, strict=True):
            risk, supported = risk_cache[term_id]
            contribution_columns.append(weight * risk)
            support_columns.append(supported)
        score, supported = compose_homotopy_score(
            contributions=np.column_stack(contribution_columns),
            term_support=np.column_stack(support_columns),
            gamma=float(spec["top2_attenuation"]),
            coordination_protection=coordination,
            coordination_active=coordination_active,
            packing_protection=packing,
            packing_active=packing_active,
        )
        maximum = float(np.max(score[supported])) if supported.any() else 0.0
        divisor = max(1.0, maximum / 50.0)
        encoded = np.full(len(features), np.nan, dtype=float)
        encoded[supported] = np.sinh(score[supported] / divisor)
        if not np.isfinite(encoded[supported]).all():
            raise ValueError("NEXT153 virtual score encoding overflowed")
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        virtual_id = f"next153_virtual_candidate__{digest}"
        feature_name = f"_{virtual_id}_value"
        columns[feature_name] = encoded
        virtual_terms.append(
            {
                "term_id": virtual_id,
                "feature": feature_name,
                "direction": 1,
                "transform": "asinh",
                "center": 0.0,
                "scale": 1.0 / divisor,
                "group": "next153_top2_attenuation_homotopy_candidate",
                "encoding": "asinh_sinh_exact_top2_attenuated_score",
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
        virtual_terms,
        runtime,
    )


def _paths(
    roots: Mapping[str, Path], freeze_path: Path, design_path: Path
) -> dict[str, Path]:
    paths = n135._paths(roots, freeze_path)
    paths.update(
        {
            "design": design_path,
            "next135_manifest": roots["next135"] / n135.MANIFEST_NAME,
            "next135_catalogue": roots["next135"] / n135.CATALOGUE_NAME,
            "next135_evaluation": roots["next135"] / n135.EVALUATION_NAME,
            "next135_search": roots["next135"] / n135.SEARCH_NAME,
            "next152_manifest": roots["next152"] / n152.MANIFEST_NAME,
            "next152_catalogue": roots["next152"] / n152.CATALOGUE_NAME,
            "next152_evaluation": roots["next152"] / n152.EVALUATION_NAME,
            "next152_search": roots["next152"] / n152.SEARCH_NAME,
        }
    )
    return paths


def _prior_endpoint_key(payload: Mapping[str, object], stage: int) -> str:
    gamma = float(payload["top2_attenuation"])
    if stage == 135 and gamma == 0.0:
        prior = {
            "base_term_ids": payload["base_term_ids"],
            "base_weights": payload["base_weights"],
            "coordination_protection_term_id": n130.PROTECTION_TERM_ID,
            "coordination_protection_weight": 2.0,
            "conjunctive_term_ids": [n135.PACKING_PRODUCT_TERM_ID],
            "conjunctive_weights": [PACKING_WEIGHT],
        }
    elif stage == 152 and gamma == 1.0:
        prior = {
            "aggregation": n152.AGGREGATION,
            "base_term_ids": payload["base_term_ids"],
            "base_weights": payload["base_weights"],
            "coordination_protection_term_id": None,
            "coordination_protection_weight": 0.0,
            "conjunctive_term_ids": [n135.PACKING_PRODUCT_TERM_ID],
            "conjunctive_weights": [PACKING_WEIGHT],
        }
    else:
        raise ValueError("NEXT153 endpoint key requested for an interior candidate")
    return json.dumps(prior, sort_keys=True, separators=(",", ":"))


def verify_endpoint_reproduction(
    *,
    result_records: Sequence[Mapping[str, object]],
    next135_records: pd.DataFrame,
    next152_records: pd.DataFrame,
) -> None:
    metrics = (
        "scigen_pooled_auc",
        "scigen_macro_auc",
        "scigen_worst_auc",
        "wyformer_pooled_auc",
        "wyformer_macro_auc",
        "wyformer_worst_auc",
    )
    booleans = (
        "passes_source_auc_gates",
        "passes_safe_all_cells",
        "passes_broad_all_cells",
        "passes_all_discovery_gates",
    )
    priors = {
        135: next135_records.set_index("candidate_key", drop=False),
        152: next152_records.set_index("candidate_key", drop=False),
    }
    counts = {135: 0, 152: 0}
    for record in result_records:
        payload = json.loads(str(record["candidate_key"]))
        gamma = float(payload["top2_attenuation"])
        if gamma not in (0.0, 1.0):
            continue
        stage = 135 if gamma == 0.0 else 152
        key = _prior_endpoint_key(payload, stage)
        if key not in priors[stage].index:
            raise RuntimeError(f"NEXT153 endpoint candidate is missing from NEXT{stage}")
        prior = priors[stage].loc[key]
        if any(
            not math.isclose(
                float(record[name]), float(prior[name]), rel_tol=0.0, abs_tol=1.0e-12
            )
            for name in metrics
        ) or any(bool(record[name]) != bool(prior[name]) for name in booleans):
            raise RuntimeError(f"NEXT153 does not reproduce NEXT{stage} endpoint")
        counts[stage] += 1
    if counts != {135: EXPECTED_BASE_COUNT, 152: EXPECTED_BASE_COUNT}:
        raise RuntimeError("NEXT153 endpoint reproduction counts differ")


def run_top2_attenuation_homotopy_search(
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
    next135_dir: Path,
    next152_dir: Path,
    next135_freeze_path: Path,
    design_path: Path,
    output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
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
                (135, next135_dir),
                (152, next152_dir),
            )
        },
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots, Path(next135_freeze_path).resolve(), Path(design_path).resolve()
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT153 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            key
            for key in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(key) != EXPECTED_INPUT_SHA256.get(key)
        )
        raise ValueError(f"NEXT153 formal input identity differs: {differing}")
    manifest135 = json.loads(paths["next135_manifest"].read_text())
    manifest152 = json.loads(paths["next152_manifest"].read_text())
    if (
        manifest135.get("protocol") != n135.PROTOCOL
        or manifest135.get("passes_all_cross_source_discovery_gates") is not False
        or manifest135.get("opened_validation_outputs_used") is not False
        or manifest152.get("protocol") != n152.PROTOCOL
        or manifest152.get("trimmed_joint_base_branch_terminated") is not True
        or manifest152.get("opened_validation_outputs_used") is not False
        or manifest152.get("dft_values_used_by_executable_formula") is not False
    ):
        raise ValueError("NEXT153 prior provenance differs")

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
    physical_terms = [*old_terms, *mhcr_terms]
    physical_ids = {str(term["term_id"]) for term in physical_terms}
    all_bases = n130.n127.select_next125_bases(
        pd.read_parquet(paths["next125_search_records"])
    )
    bases = n132.select_extended_bases(
        pd.read_parquet(paths["next130_search_records"]), all_bases
    )
    specs = build_candidate_specs(bases=bases, physical_term_ids=physical_ids)
    base_formulas = sorted(
        n130.n127._formula_identity(
            json.loads(str(row["term_ids_json"])),
            json.loads(str(row["weights_json"])),
        )
        for _, row in bases.iterrows()
    )
    base_formula_sha = hashlib.sha256("\n".join(base_formulas).encode()).hexdigest()
    candidate_sha = hashlib.sha256(
        "\n".join(str(spec["candidate_key"]) for spec in specs).encode()
    ).hexdigest()
    if require_formal_inputs and (
        len(bases) != EXPECTED_BASE_COUNT
        or len(specs) != EXPECTED_CANDIDATE_COUNT
        or base_formula_sha != EXPECTED_BASE_FORMULA_SHA256
        or candidate_sha != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT153 frozen candidate universe differs")

    scigen_endpoint = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_endpoint = pd.read_parquet(paths["wyformer_endpoint"])
    endpoint_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "material_id": "scigen:"
                    + scigen_endpoint["material_id"].astype(str),
                    "_endpoint": pd.to_numeric(
                        scigen_endpoint["distortion_ratio"], errors="coerce"
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "material_id": "wyformer:"
                    + wyformer_endpoint["material_id"].astype(str),
                    "_endpoint": n130.n125.n121.prior._endpoint_numeric(
                        wyformer_endpoint["endpoint_stratum"]
                    ),
                }
            ),
        ],
        ignore_index=True,
    )
    combined = extended.merge(
        endpoint_frame, on="material_id", how="inner", validate="one_to_one"
    )
    endpoint = pd.to_numeric(combined.pop("_endpoint"), errors="coerce").to_numpy(float)
    combined, virtual_terms, runtime = materialize_candidates(
        features=combined, physical_terms=physical_terms, specs=specs
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
    verify_endpoint_reproduction(
        result_records=result["candidate_records"],
        next135_records=pd.read_parquet(paths["next135_search"]),
        next152_records=pd.read_parquet(paths["next152_search"]),
    )

    def decorate(record: dict[str, object]) -> None:
        payload = json.loads(str(record["candidate_key"]))
        evaluated = json.loads(str(record["base_term_ids_json"]))
        record["evaluation_virtual_term_id"] = str(evaluated[0])
        record["aggregation"] = AGGREGATION
        record["top2_attenuation"] = float(payload["top2_attenuation"])
        record["base_term_ids_json"] = json.dumps(
            payload["base_term_ids"], separators=(",", ":")
        )
        record["base_weights_json"] = json.dumps(
            payload["base_weights"], separators=(",", ":")
        )
        record["coordination_protection_weight"] = float(
            payload["coordination_protection_weight"]
        )
        record["packing_protection_weight"] = PACKING_WEIGHT
        record["score_composition"] = (
            "max(0,sum(weighted_base)-gamma*top2_sum"
            "-2*(1-gamma)*coordination-0.5*coordination_covalent_packing)"
        )

    for record in result["candidate_records"]:
        decorate(record)
    selected = result["selected"]
    if "evaluation_virtual_term_id" not in selected["record"]:
        decorate(selected["record"])
    payload = json.loads(str(selected["record"]["candidate_key"]))
    physical_by_id = {str(term["term_id"]): dict(term) for term in physical_terms}
    formula = selected["formula"]
    formula["evaluation_virtual_term_id"] = str(formula["base_terms"][0]["term_id"])
    formula["base_terms"] = [
        {**physical_by_id[str(term_id)], "weight": float(weight)}
        for term_id, weight in zip(
            payload["base_term_ids"], payload["base_weights"], strict=True
        )
    ]
    formula["aggregation"] = AGGREGATION
    formula["top2_attenuation"] = float(payload["top2_attenuation"])
    formula["coordination_protection"] = {
        "term_id": n130.PROTECTION_TERM_ID,
        "feature": n130.n129.FEATURE_NAME,
        "weight": float(payload["coordination_protection_weight"]),
        "missing_policy": "TERM_OFF_KEEP_BASE",
    }
    formula["conjunctive_protection"] = {
        "term_id": n135.PACKING_PRODUCT_TERM_ID,
        "feature": n135.PACKING_PRODUCT_FEATURE,
        "weight": PACKING_WEIGHT,
        "missing_policy": "TERM_OFF_KEEP_BASE",
    }
    formula["score_composition"] = selected["record"]["score_composition"]
    formula["kind"] = "top2_attenuated_physical_base_homotopy"
    passes = bool(selected["record"]["passes_all_discovery_gates"])
    records_frame = pd.DataFrame(result["candidate_records"])
    counts = {
        f"gamma={gamma:g}": {
            "candidates": int(len(frame)),
            "passes_source_auc_gates": int(frame["passes_source_auc_gates"].sum()),
            "passes_safe_all_cells": int(frame["passes_safe_all_cells"].sum()),
            "passes_broad_all_cells": int(frame["passes_broad_all_cells"].sum()),
            "passes_all_discovery_gates": int(
                frame["passes_all_discovery_gates"].sum()
            ),
        }
        for gamma, frame in records_frame.groupby("top2_attenuation", sort=True)
    }
    catalogue = {
        "protocol": PROTOCOL,
        "design_sha256": input_hashes["design"],
        "base_count": len(bases),
        "candidate_count": len(specs),
        "gamma_grid": list(GAMMAS),
        "coordination_weight_formula": "2*(1-gamma)",
        "packing_weight": PACKING_WEIGHT,
        "base_formula_sha256": base_formula_sha,
        "candidate_key_sha256": candidate_sha,
        "active_score": (
            "max(0,sum(weighted_base)-gamma*top2_sum"
            "-2*(1-gamma)*coordination-0.5*coordination_covalent_packing)"
        ),
        "base_support_unchanged": True,
        "gamma0_reproduces_next135": True,
        "gamma1_reproduces_next152": True,
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    catalogue_sha = hashlib.sha256(
        json.dumps(catalogue, indent=2, sort_keys=True).encode() + b"\n"
    ).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    source_paths = {
        "src/next135_conjunctive_compactness_search.py": Path(n135.__file__).resolve(),
        "src/next152_trimmed_joint_base_search.py": Path(n152.__file__).resolve(),
        "src/next153_top2_attenuation_homotopy_search.py": Path(__file__).resolve(),
    }
    source_hashes = {
        name: _sha256_file(path) for name, path in source_paths.items()
    }
    output_paths: list[Path] = []
    try:
        catalogue_path = staging / CATALOGUE_NAME
        evaluation_path = staging / EVALUATION_NAME
        search_path = staging / SEARCH_NAME
        _write_json(
            catalogue_path,
            {**catalogue, "label_free_catalogue_sha256": catalogue_sha},
        )
        _write_json(
            evaluation_path,
            {
                "protocol": PROTOCOL,
                "evaluation_mode": "fixed_top2_attenuation_homotopy",
                "rows": {
                    "scigen": int(len(feature_tables["scigen"])),
                    "wyformer": int(len(feature_tables["wyformer"])),
                    "total": int(len(combined)),
                },
                "base_count": len(bases),
                "candidate_count": int(result["candidate_count"]),
                "elapsed_seconds": elapsed,
                "search_workers": search_workers,
                "gamma0_reproduced_next135": True,
                "gamma1_reproduced_next152": True,
                "counts_by_gamma": counts,
                "safe_gates": dict(n130.n125.n121.prior.DEFAULT_GATES),
                "source_auc_gates": dict(n130.n125.n121.prior.AUC_GATES),
                "broad_min_severe_precision_lower": n130.n125.n121.prior.BROAD_MIN_PRECISION_LOWER,
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
        records_frame.to_parquet(search_path, index=False)
        output_paths.extend([catalogue_path, evaluation_path, search_path])
        manifest = {
            "protocol": PROTOCOL,
            "label_free_catalogue_sha256": catalogue_sha,
            "base_count": len(bases),
            "candidate_count": int(result["candidate_count"]),
            "search_workers": search_workers,
            "gamma0_reproduced_next135": True,
            "gamma1_reproduced_next152": True,
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "requires_unopened_internal_validation_before_claim": True,
            "top2_attenuation_homotopy_branch_terminated": not passes,
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
            raise RuntimeError("NEXT153 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT153 source changed before publication")
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
        135,
        152,
    )
    for stage in stages:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_top2_attenuation_homotopy_search(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{
            f"next{stage}_dir": getattr(args, f"next{stage}_dir")
            for stage in stages
        },
        next135_freeze_path=args.next135_freeze_path,
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
    "EXPECTED_CANDIDATE_COUNT",
    "EXPECTED_CANDIDATE_KEY_SHA256",
    "build_candidate_specs",
    "compose_homotopy_score",
    "run_top2_attenuation_homotopy_search",
    "top2_attenuated_sum",
]
