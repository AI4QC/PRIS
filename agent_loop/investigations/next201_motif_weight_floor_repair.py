#!/usr/bin/env python3
"""Finite no-DFT repair search using the weakest-site motif weight floor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import src.next182_local_family_closure_attenuation_search as n182
import src.next200_cross_source_motif_audit as n200
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-08-next201-motif-weight-floor-repair-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT201_MOTIF_WEIGHT_FLOOR_CATALOGUE.json"
EVALUATION_NAME = "NEXT201_DISCOVERY_EVALUATION.json"
FORMULA_NAME = "NEXT201_FROZEN_CANDIDATE.json"
SEARCH_NAME = "next201_motif_weight_floor_repair.parquet"
EXPECTED_DESIGN_SHA256 = (
    "bfd3cdd2b3c7c7b32fe9ee16df11fc05dec94a62e8eaa1273aaa64771b991e26"
)
EXPECTED_BASE_CANDIDATE_KEY_SHA256 = n200.n194.n192.n190.n186_candidate_key_sha256()
BROAD_THRESHOLD = n200.BROAD_THRESHOLD
SAFE_THRESHOLD = n200.SAFE_THRESHOLD
REPAIR_WIDTH = SAFE_THRESHOLD - BROAD_THRESHOLD
FLOOR_THRESHOLDS = tuple(1.0 - 2.0**-power for power in range(0, 11))
ATTENUATIONS = (0.25, 0.50, 0.75, 1.00, 1.50, 2.00)
EXPECTED_CANDIDATE_COUNT = 1 + len(FLOOR_THRESHOLDS) * len(ATTENUATIONS)
SEARCH_WORKERS = 4
SCORE_COMPOSITION = (
    "base_score_if_outside_frozen_repair_interval_or_motif_missing_else_"
    "max(0,base_score-alpha*(safe_threshold-broad_threshold)*"
    "clip((clip(motif_weight_sum_min,0,1)-tau)/(1-tau),0,1))"
)
EXPECTED_INPUT_SHA256 = {
    **n200.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next200_manifest": "ed61510182b00f9ba1addfd3a5f1792abd0d307e800e71df708d71e9f643de78",
    "next200_audit": "b34dae03e567718a576a66930eda5b8f454377d8e79a69f16d253bc3e1e2a093",
    "next200_table": "df91717a931ffc18a285256993f1abaf6cc09a435f821f122cd2b5aedfb201fd",
}


def motif_weight_floor_certificate(
    values: object, *, floor_threshold: float
) -> np.ndarray:
    """Map weakest-site CrystalNN weight to a fixed ramp ending at one."""

    tau = float(floor_threshold)
    if not np.isfinite(tau) or tau < 0.0 or tau >= 1.0:
        raise ValueError("NEXT201 floor threshold differs")
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("NEXT201 motif schema differs")
    result = np.full(array.shape, np.nan, dtype=float)
    finite = np.isfinite(array)
    bounded = np.clip(array[finite], 0.0, 1.0)
    result[finite] = np.clip((bounded - tau) / (1.0 - tau), 0.0, 1.0)
    return result


def motif_weight_floor_repair_score(
    *,
    base_score: object,
    base_support: object,
    certificate: object,
    attenuation: float,
    broad_threshold: float = BROAD_THRESHOLD,
    safe_threshold: float = SAFE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Repair only the original `[BROAD, SAFE)` interval; missing keeps base."""

    score = np.asarray(base_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    values = np.asarray(certificate, dtype=float)
    alpha = float(attenuation)
    broad = float(broad_threshold)
    safe = float(safe_threshold)
    if (
        score.ndim != 1
        or support.shape != score.shape
        or values.shape != score.shape
        or not np.isfinite(alpha)
        or alpha < 0.0
        or not np.isfinite([broad, safe]).all()
        or not broad < safe
    ):
        raise ValueError("NEXT201 repair schema differs")
    finite_certificate = np.isfinite(values)
    if np.any(
        (values[finite_certificate] < -1.0e-12)
        | (values[finite_certificate] > 1.0 + 1.0e-12)
    ):
        raise ValueError("NEXT201 certificate is outside bounds")
    corrected_support = support & np.isfinite(score)
    active = (
        corrected_support
        & finite_certificate
        & (score >= broad)
        & (score < safe)
    )
    corrected = np.full(score.shape, np.nan, dtype=float)
    corrected[corrected_support] = score[corrected_support]
    corrected[active] = np.maximum(
        0.0,
        score[active]
        - alpha * (safe - broad) * np.clip(values[active], 0.0, 1.0),
    )
    return corrected, corrected_support, active


def build_candidate_specs(*, base_candidate_key: str) -> list[dict[str, object]]:
    """Build the exact base plus frozen floor/attenuation product."""

    if not isinstance(base_candidate_key, str) or not base_candidate_key:
        raise ValueError("NEXT201 base candidate key must be nonempty")
    pairs: list[tuple[float | None, float]] = [(None, 0.0)]
    pairs.extend(
        (floor, attenuation)
        for floor in FLOOR_THRESHOLDS
        for attenuation in ATTENUATIONS
    )
    specs = []
    for floor, attenuation in pairs:
        payload = {
            "attenuation": attenuation,
            "base_candidate_key": base_candidate_key,
            "broad_threshold": BROAD_THRESHOLD,
            "certificate_feature": (
                None if floor is None else "motif_weight_sum_min"
            ),
            "floor_threshold": floor,
            "missing_policy": "TERM_OFF_KEEP_BASE",
            "repair_width": REPAIR_WIDTH,
            "safe_threshold": SAFE_THRESHOLD,
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
    if (
        len(specs) != EXPECTED_CANDIDATE_COUNT
        or len({str(spec["candidate_key"]) for spec in specs})
        != EXPECTED_CANDIDATE_COUNT
    ):
        raise RuntimeError("NEXT201 candidate universe differs")
    return specs


def materialize_motif_weight_floor_candidates(
    *,
    features: pd.DataFrame,
    base_score: object,
    base_support: object,
    motif_weight_sum_min: object,
    specs: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    """Encode every corrected score as an exactly recoverable virtual term."""

    score = np.asarray(base_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    motif = np.asarray(motif_weight_sum_min, dtype=float)
    if (
        score.shape != (len(features),)
        or support.shape != score.shape
        or motif.shape != score.shape
    ):
        raise ValueError("NEXT201 base or motif shape differs")
    expected_specs = build_candidate_specs(
        base_candidate_key=(
            str(specs[0].get("base_candidate_key", "")) if specs else ""
        )
    )
    if [str(spec.get("candidate_key", "")) for spec in specs] != [
        str(spec["candidate_key"]) for spec in expected_specs
    ]:
        raise ValueError("NEXT201 candidate specs differ")

    columns: dict[str, np.ndarray] = {}
    terms: list[dict[str, object]] = []
    runtime: list[dict[str, object]] = []
    for raw in specs:
        spec = dict(raw)
        key = str(spec["candidate_key"])
        floor = spec["floor_threshold"]
        certificate = (
            np.full(len(features), np.nan)
            if floor is None
            else motif_weight_floor_certificate(motif, floor_threshold=float(floor))
        )
        corrected, corrected_support, _ = motif_weight_floor_repair_score(
            base_score=score,
            base_support=support,
            certificate=certificate,
            attenuation=float(spec["attenuation"]),
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
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        virtual_id = f"next201_virtual_candidate__{digest}"
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
                "group": "next201_motif_weight_floor",
                "encoding": "asinh_sinh_exact_motif_weight_floor_repair_score",
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
    roots: Mapping[str, Path], freeze_path: Path, design_path: Path
) -> dict[str, Path]:
    paths = n200._paths(roots, freeze_path, design_path)
    paths.update(
        {
            "next200_manifest": roots["next200"] / n200.MANIFEST_NAME,
            "next200_audit": roots["next200"] / n200.AUDIT_NAME,
            "next200_table": roots["next200"] / n200.TABLE_NAME,
        }
    )
    return paths


def _verify_next200(paths: Mapping[str, Path], hashes: Mapping[str, str]) -> None:
    manifest = json.loads(paths["next200_manifest"].read_text())
    audit = json.loads(paths["next200_audit"].read_text())
    selected = audit.get("selected_hypothesis")
    if (
        manifest.get("protocol") != n200.PROTOCOL
        or manifest.get("hypothesis_count") != len(n200.HYPOTHESES)
        or manifest.get("eligible_hypothesis_count") != 1
        or manifest.get("next201_search_authorized") is not True
        or manifest.get("new_formula_searched") is not False
        or manifest.get("opened_validation_outputs_used") is not False
        or manifest.get("scigen_replication_endpoint_opened") is not False
        or manifest.get("wyformer_replication_endpoint_opened") is not False
        or manifest.get("dft_calculation_executed") is not False
        or manifest.get("dft_values_used_by_executable_formula") is not False
        or manifest.get("learned_energy_force_stress_proxy_used") is not False
        or manifest.get("model_or_proxy_potential_used") is not False
        or manifest.get("physical_relaxation_executed") is not False
        or manifest.get("outputs_sha256")
        != {
            n200.AUDIT_NAME: hashes["next200_audit"],
            n200.TABLE_NAME: hashes["next200_table"],
        }
        or manifest.get("executed_source_sha256", {}).get(
            "src/next200_cross_source_motif_audit.py"
        )
        != _sha256_file(Path(n200.__file__).resolve())
    ):
        raise ValueError("NEXT201 NEXT200 provenance differs")
    if (
        audit.get("protocol") != n200.PROTOCOL
        or audit.get("eligible_hypotheses")
        != ["motif_weight_sum_min__protected_high"]
        or not isinstance(selected, dict)
        or selected.get("feature") != "motif_weight_sum_min"
        or selected.get("direction") != 1
        or audit.get("new_formula_searched") is not False
        or audit.get("validation_or_replication_opened") is not False
    ):
        raise ValueError("NEXT201 NEXT200 audit boundary differs")


def run_motif_weight_floor_search(
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
    next135_freeze_path: Path,
    design_path: Path,
    output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT201 search and publish atomically."""

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
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots, Path(next135_freeze_path).resolve(), Path(design_path).resolve()
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(search_workers) is not int or search_workers <= 0:
        raise ValueError("NEXT201 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT201 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT201 formal input identity differs: {differing}")
    _verify_next200(paths, input_hashes)

    diagnostic164 = json.loads(paths["next164_diagnostic"].read_text())
    base_key = str(diagnostic164.get("global_closest", {}).get("candidate_key", ""))
    if hashlib.sha256(base_key.encode()).hexdigest() != EXPECTED_BASE_CANDIDATE_KEY_SHA256:
        raise ValueError("NEXT201 base candidate identity differs")

    combined, feature_tables, old_terms, mhcr_terms = n200.n194.n130._join_label_free_features(paths)
    compact_frames = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next133_{source}_features"]).copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        compact_frames.append(table)
    combined = combined.merge(
        pd.concat(compact_frames, ignore_index=True),
        on="material_id", how="inner", validate="one_to_one",
    )
    combined = pd.concat(
        [
            combined.reset_index(drop=True),
            n200.n194.n135.materialize_conjunctive_features(combined).reset_index(drop=True),
        ],
        axis=1,
    )
    closure_frames = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next179_{source}_features"]).copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        closure_frames.append(table)
    combined = combined.merge(
        pd.concat(closure_frames, ignore_index=True),
        on="material_id", how="inner", validate="one_to_one",
    )
    motif_frames = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next199_{source}_features"])[
            ["material_id", "motif_weight_sum_min"]
        ].copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        motif_frames.append(table)
    motif_table = pd.concat(motif_frames, ignore_index=True)
    combined = combined.merge(
        motif_table, on="material_id", how="inner", validate="one_to_one"
    )
    if len(combined) != len(motif_table):
        raise ValueError("NEXT201 motif row accounting differs")

    physical_terms = [*old_terms, *mhcr_terms]
    physical_ids = {str(term["term_id"]) for term in physical_terms}
    all_bases = n200.n194.n130.n127.select_next125_bases(
        pd.read_parquet(paths["next125_search_records"])
    )
    bases = n200.n194.n135.n132.select_extended_bases(
        pd.read_parquet(paths["next130_search_records"]), all_bases
    )
    base_specs = n200.n194.n163.build_candidate_specs(
        bases=bases, physical_term_ids=physical_ids
    )
    selected_specs = [
        spec for spec in base_specs if str(spec["candidate_key"]) == base_key
    ]
    if len(selected_specs) != 1:
        raise ValueError("NEXT201 base reconstruction differs")
    combined, base_terms, base_runtime = n200.n194.n163.materialize_candidates(
        features=combined, physical_terms=physical_terms, specs=selected_specs
    )
    if len(base_terms) != 1 or len(base_runtime) != 1:
        raise RuntimeError("NEXT201 base materialization differs")
    base_score, base_support = n200.n194.n87._term_risk(combined, base_terms[0])

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
                    "_endpoint": n200.n194.n130.n125.n121.prior._endpoint_numeric(
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
    if len(endpoint) != len(base_score) or not np.isfinite(endpoint).all():
        raise ValueError("NEXT201 endpoint row accounting differs")

    specs = build_candidate_specs(base_candidate_key=base_key)
    combined, virtual_terms, runtime = materialize_motif_weight_floor_candidates(
        features=combined,
        base_score=base_score,
        base_support=base_support,
        motif_weight_sum_min=pd.to_numeric(
            combined["motif_weight_sum_min"], errors="coerce"
        ).to_numpy(float),
        specs=specs,
    )
    started = time.perf_counter()
    result = n200.n194.n130.n125.search_optional_guard_laws_parallel(
        features=combined,
        endpoint=endpoint,
        old_terms=virtual_terms,
        optional_terms=[],
        candidate_specs=runtime,
        workers=search_workers,
    )
    elapsed = time.perf_counter() - started
    if int(result["candidate_count"]) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("NEXT201 evaluator count differs")

    spec_by_key = {str(spec["candidate_key"]): spec for spec in specs}

    def decorate(record: dict[str, object]) -> None:
        spec = spec_by_key[str(record["candidate_key"])]
        record.update(
            {
                "base_candidate_key": base_key,
                "certificate_feature": spec["certificate_feature"],
                "floor_threshold": spec["floor_threshold"],
                "attenuation": float(spec["attenuation"]),
                "repair_width": REPAIR_WIDTH,
                "broad_threshold": BROAD_THRESHOLD,
                "safe_threshold_frozen": SAFE_THRESHOLD,
                "missing_policy": "TERM_OFF_KEEP_BASE",
                "score_composition": SCORE_COMPOSITION,
            }
        )

    for record in result["candidate_records"]:
        decorate(record)
    selected = result["selected"]
    if "floor_threshold" not in selected["record"]:
        decorate(selected["record"])
    base_records = [
        record
        for record in result["candidate_records"]
        if record["floor_threshold"] is None
    ]
    if len(base_records) != 1:
        raise RuntimeError("NEXT201 base candidate differs")
    n182.n181.n175.n170._verify_base_reproduction(
        record=base_records[0],
        published=pd.read_parquet(paths["next163_search"]),
        candidate_key=base_key,
    )
    selected_spec = spec_by_key[str(selected["record"]["candidate_key"])]
    prior = json.loads(paths["next163_evaluation"].read_text())
    formula = {
        "protocol": PROTOCOL,
        "kind": "motif_weight_floor_repair_no_dft_score",
        "base_candidate_key": base_key,
        "base_formula": prior["selected_formula"],
        "certificate_feature": selected_spec["certificate_feature"],
        "floor_threshold": selected_spec["floor_threshold"],
        "certificate_definition": (
            "clip((clip(motif_weight_sum_min,0,1)-tau)/(1-tau),0,1)"
        ),
        "attenuation": float(selected_spec["attenuation"]),
        "repair_width": REPAIR_WIDTH,
        "broad_threshold": BROAD_THRESHOLD,
        "safe_threshold": SAFE_THRESHOLD,
        "interval_policy": "BROAD_INCLUSIVE_SAFE_EXCLUSIVE_ON_ORIGINAL_BASE_SCORE",
        "missing_policy": "TERM_OFF_KEEP_BASE",
        "score_composition": SCORE_COMPOSITION,
        "dft_values_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "physical_relaxation_executed": False,
    }
    passes = bool(selected["record"]["passes_all_discovery_gates"])
    records = pd.DataFrame(result["candidate_records"])
    counts = {}
    grouped = records.assign(
        floor_threshold=records["floor_threshold"].fillna(-1.0)
    ).groupby(["floor_threshold", "attenuation"], sort=True)
    for (floor, attenuation), frame in grouped:
        label = "BASE" if float(floor) < 0.0 else f"{float(floor):.12g}"
        counts[f"floor={label},alpha={float(attenuation):g}"] = {
            "candidates": int(len(frame)),
            "passes_source_auc_gates": int(frame["passes_source_auc_gates"].sum()),
            "passes_safe_all_cells": int(frame["passes_safe_all_cells"].sum()),
            "passes_broad_all_cells": int(frame["passes_broad_all_cells"].sum()),
            "passes_all_discovery_gates": int(frame["passes_all_discovery_gates"].sum()),
        }
    catalogue = {
        "protocol": PROTOCOL,
        "design_sha256": input_hashes["design"],
        "base_candidate_key_sha256": EXPECTED_BASE_CANDIDATE_KEY_SHA256,
        "base_endpoint_reproduced": True,
        "authorized_hypothesis": "motif_weight_sum_min__protected_high",
        "floor_threshold_grid": FLOOR_THRESHOLDS,
        "attenuation_grid": ATTENUATIONS,
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "repair_width": REPAIR_WIDTH,
        "broad_threshold": BROAD_THRESHOLD,
        "safe_threshold": SAFE_THRESHOLD,
        "score_composition": SCORE_COMPOSITION,
        "base_support_unchanged": True,
        "outside_interval_exactly_unchanged": True,
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next163_interior_family_attenuation_search.py": Path(
            n200.n194.n163.__file__
        ).resolve(),
        "src/next199_cross_source_motif_features.py": Path(
            n200.n199.__file__
        ).resolve(),
        "src/next200_cross_source_motif_audit.py": Path(n200.__file__).resolve(),
        "src/next201_motif_weight_floor_repair.py": Path(__file__).resolve(),
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
                "evaluation_mode": "fixed_motif_weight_floor_repair_search",
                "base_endpoint_reproduced": True,
                "rows": {
                    "scigen": int(len(feature_tables["scigen"])),
                    "wyformer": int(len(feature_tables["wyformer"])),
                    "total": int(len(combined)),
                },
                "candidate_count": int(result["candidate_count"]),
                "elapsed_seconds": elapsed,
                "search_workers": search_workers,
                "counts_by_floor_and_attenuation": counts,
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
                "requires_unopened_internal_validation_before_claim": True,
            },
        )
        _write_json(formula_path, formula)
        records.to_parquet(search_path, index=False)
        manifest = {
            "protocol": PROTOCOL,
            "candidate_count": int(result["candidate_count"]),
            "search_workers": search_workers,
            "base_endpoint_reproduced": True,
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "requires_unopened_internal_validation_before_claim": True,
            "motif_weight_floor_search_branch_terminated": not passes,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True,
            "opened_validation_outputs_used": False,
            "scigen_replication_endpoint_opened": False,
            "wyformer_replication_endpoint_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "model_or_proxy_potential_used": False,
            "physical_relaxation_executed": False,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {
                CATALOGUE_NAME: _sha256_file(catalogue_path),
                EVALUATION_NAME: _sha256_file(evaluation_path),
                FORMULA_NAME: _sha256_file(formula_path),
                SEARCH_NAME: _sha256_file(search_path),
            },
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT201 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT201 source changed before publication")
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
    parser.add_argument("--next194-dir", type=Path, required=True)
    parser.add_argument("--next199-dir", type=Path, required=True)
    parser.add_argument("--next200-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_motif_weight_floor_search(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{f"next{stage}_dir": getattr(args, f"next{stage}_dir") for stage in stages},
        next194_dir=args.next194_dir,
        next199_dir=args.next199_dir,
        next200_dir=args.next200_dir,
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
    "EXPECTED_CANDIDATE_COUNT",
    "FLOOR_THRESHOLDS",
    "build_candidate_specs",
    "materialize_motif_weight_floor_candidates",
    "motif_weight_floor_certificate",
    "motif_weight_floor_repair_score",
    "run_motif_weight_floor_search",
]


if __name__ == "__main__":
    raise SystemExit(main())
