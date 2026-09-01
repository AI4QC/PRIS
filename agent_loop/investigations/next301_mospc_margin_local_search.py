#!/usr/bin/env python3
"""Frozen one-term MOSPC margin-local search on the NEXT224 frontier."""

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

import src.next261_pvbp_margin_local_search as n261
import src.next300_mospc_feature_audit as n300
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


n227 = n300.n268.n227
n223 = n261.n223
PROTOCOL = "2026-08-09-next301-mospc-margin-local-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT301_MOSPC_MARGIN_LOCAL_CATALOGUE.json"
EVALUATION_NAME = "NEXT301_DISCOVERY_EVALUATION.json"
FORMULA_NAME = "NEXT301_FROZEN_CANDIDATE.json"
SEARCH_NAME = "next301_mospc_margin_local_search.parquet"
SCORE_COMPOSITION = "nonnegative_next224_plus_triangular_margin_local_signed_mospc_term"
LOCAL_WIDTH_DENOMINATOR = n261.LOCAL_WIDTH_DENOMINATOR
LOCAL_WIDTH_FRACTIONS = n261.LOCAL_WIDTH_FRACTIONS
AMPLITUDE_DENOMINATOR = n261.AMPLITUDE_DENOMINATOR
AMPLITUDE_FRACTIONS = n261.AMPLITUDE_FRACTIONS
EXPECTED_ELIGIBLE_HYPOTHESIS_COUNT = 5
EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256 = (
    "baa906df0e3c037fbd69b418980e4b627a2db4b24e397de25aa7c8bf592ab5c7"
)
EXPECTED_CANDIDATE_COUNT = 1 + (
    EXPECTED_ELIGIBLE_HYPOTHESIS_COUNT
    * len(LOCAL_WIDTH_FRACTIONS)
    * len(AMPLITUDE_FRACTIONS)
)
EXPECTED_ELIGIBLE_COUNT = EXPECTED_CANDIDATE_COUNT - 1
EXPECTED_DESIGN_SHA256 = n300.EXPECTED_INPUT_SHA256["design"]
EXPECTED_BASE_CANDIDATE_KEY_SHA256 = n227.EXPECTED_BASE_CANDIDATE_KEY_SHA256
EXPECTED_BASE_THRESHOLD = n227.EXPECTED_BASE_THRESHOLD
EXPECTED_BASE_SUPPORT_COUNT = n227.EXPECTED_BASE_SUPPORT_COUNT
REPAIR_WIDTH = n261.REPAIR_WIDTH
SEARCH_WORKERS = n223.SEARCH_WORKERS
BOUNDARY_FLAGS = n300.BOUNDARY_FLAGS
REQUIRED_STAGES = (*n300.REQUIRED_STAGES, 300)
REQUIRED_DESIGN_STAGES = (*n300.REQUIRED_DESIGN_STAGES, 300)
EXPECTED_NEXT261_SOURCE_SHA256 = (
    "5283f4b6a58e313a735cd4f3245028a173caa77f0338a6d3e04e5a7244256599"
)
EXPECTED_NEXT300_SOURCE_SHA256 = (
    "8723fcb52e9ccb605bead336719caad6f68077deb32e706bc14148690aa36783"
)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n300.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next300_design": n300.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next300_manifest": (
        "e7f739817774dd3eded49be91ab712bd08a51093c69017c947b07dff7f76d2ee"
    ),
    "next300_catalogue": (
        "03445c57d1625f38b48939c7104c39384e4bfa67a7263cced990c594f85b9443"
    ),
    "next300_audit": (
        "91186ef228558dface94944567abf9464dab437b0222a4e7d2dda0cde7bf2faf"
    ),
    "next300_table": (
        "ff66f52a0d92bf5515ba77964894fd7fa44638fdc0d35a37b0b5d902b1818d5d"
    ),
}


def mospc_margin_local_score(
    *,
    base_score: object,
    base_support: object,
    protection: object,
    threshold: float,
    repair_width: float,
    local_width_fraction: float,
    amplitude_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        return n261.pvbp_margin_local_score(
            base_score=base_score,
            base_support=base_support,
            protection=protection,
            threshold=threshold,
            repair_width=repair_width,
            local_width_fraction=local_width_fraction,
            amplitude_fraction=amplitude_fraction,
        )
    except ValueError as exc:
        raise ValueError("NEXT301 margin-local score inputs differ") from exc


def build_mospc_candidate_specs(
    *,
    base_candidate_key: str,
    eligible_table: pd.DataFrame,
    threshold: float = EXPECTED_BASE_THRESHOLD,
    repair_width: float = REPAIR_WIDTH,
    local_width_fractions: Sequence[float] = LOCAL_WIDTH_FRACTIONS,
    amplitude_fractions: Sequence[float] = AMPLITUDE_FRACTIONS,
) -> list[dict[str, object]]:
    raw_specs = n261.build_pvbp_candidate_specs(
        base_candidate_key=base_candidate_key,
        eligible_table=eligible_table,
        threshold=threshold,
        repair_width=repair_width,
        local_width_fractions=local_width_fractions,
        amplitude_fractions=amplitude_fractions,
    )
    result: list[dict[str, object]] = []
    for raw in raw_specs:
        payload = {key: value for key, value in raw.items() if key != "candidate_key"}
        payload["score_composition"] = SCORE_COMPOSITION
        result.append(
            {
                **payload,
                "candidate_key": json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                ),
            }
        )
    if len({str(spec["candidate_key"]) for spec in result}) != len(result):
        raise RuntimeError("NEXT301 candidate keys are not unique")
    return result


def materialize_mospc_candidates(
    *,
    features: pd.DataFrame,
    base_score: object,
    base_support: object,
    specs: Sequence[Mapping[str, object]],
) -> tuple[
    pd.DataFrame,
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, dict[str, int]],
]:
    virtual, terms, runtime, activity = n261.n257.materialize_dvci_candidates(
        features=features,
        base_score=base_score,
        base_support=base_support,
        specs=specs,
    )
    rename: dict[str, str] = {}
    id_map: dict[str, str] = {}
    for term in terms:
        key = str(term["physical_candidate_key"])
        new_id = "next301_virtual_candidate__" + hashlib.sha256(
            key.encode()
        ).hexdigest()[:24]
        old_id = str(term["term_id"])
        old_column = str(term["feature"])
        new_column = f"_{new_id}_value"
        id_map[old_id] = new_id
        rename[old_column] = new_column
        term.update(
            {
                "term_id": new_id,
                "feature": new_column,
                "group": "next301_mospc_margin_local",
            }
        )
    virtual = virtual.rename(columns=rename)
    for spec in runtime:
        spec["base_term_ids"] = [id_map[str(value)] for value in spec["base_term_ids"]]
    return virtual, terms, runtime, activity


def _paths(
    *,
    roots: Mapping[str, Path],
    next135_freeze_path: Path,
    design_paths: Mapping[int, Path],
    design_path: Path,
) -> dict[str, Path]:
    paths = n300._paths(
        roots=roots,
        next135_freeze_path=next135_freeze_path,
        design_paths={
            stage: design_paths[stage] for stage in n300.REQUIRED_DESIGN_STAGES
        },
        design_path=design_paths[300],
    )
    paths["next300_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next300_manifest": roots["next300"] / n300.MANIFEST_NAME,
            "next300_catalogue": roots["next300"] / n300.CATALOGUE_NAME,
            "next300_audit": roots["next300"] / n300.AUDIT_NAME,
            "next300_table": roots["next300"] / n300.TABLE_NAME,
        }
    )
    return paths


def _verify_next300(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    str,
    str,
    dict[str, object],
    dict[str, object],
    str,
    dict[str, object],
    dict[str, pd.DataFrame],
    pd.DataFrame,
]:
    base_paths = {key: paths[key] for key in n300.EXPECTED_INPUT_SHA256}
    base_paths["design"] = paths["next300_design"]
    base_hashes = {key: input_hashes[key] for key in n300.EXPECTED_INPUT_SHA256}
    base_hashes["design"] = input_hashes["next300_design"]
    prior = n300._verify_next299(base_paths, base_hashes)
    manifest = json.loads(paths["next300_manifest"].read_text())
    catalogue = json.loads(paths["next300_catalogue"].read_text())
    audit = json.loads(paths["next300_audit"].read_text())
    table = pd.read_parquet(paths["next300_table"])
    eligible_table = table.loc[
        table["eligible_for_search"].fillna(False).astype(bool)
    ].sort_values("hypothesis", kind="mergesort").reset_index(drop=True)
    eligible = tuple(eligible_table["hypothesis"].astype(str))
    expected_outputs = {
        n300.CATALOGUE_NAME: input_hashes["next300_catalogue"],
        n300.AUDIT_NAME: input_hashes["next300_audit"],
        n300.TABLE_NAME: input_hashes["next300_table"],
    }
    if (
        manifest.get("protocol") != n300.PROTOCOL
        or manifest.get("eligible_hypothesis_count")
        != EXPECTED_ELIGIBLE_HYPOTHESIS_COUNT
        or manifest.get("eligible_hypothesis_sha256")
        != EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256
        or manifest.get("next301_search_authorized") is not True
        or manifest.get("new_formula_searched") is not False
        or manifest.get("new_formula_selected") is not False
        or manifest.get("outputs_sha256") != expected_outputs
        or manifest.get("executed_source_sha256", {}).get(
            "src/next300_mospc_feature_audit.py"
        )
        != EXPECTED_NEXT300_SOURCE_SHA256
        or _sha256_file(Path(n300.__file__).resolve())
        != EXPECTED_NEXT300_SOURCE_SHA256
        or _sha256_file(Path(n261.__file__).resolve())
        != EXPECTED_NEXT261_SOURCE_SHA256
        or catalogue.get("design_sha256") != EXPECTED_DESIGN_SHA256
        or audit.get("eligible_hypothesis_count")
        != EXPECTED_ELIGIBLE_HYPOTHESIS_COUNT
        or audit.get("eligible_hypothesis_sha256")
        != EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256
        or audit.get("next301_search_authorized") is not True
        or len(eligible) != EXPECTED_ELIGIBLE_HYPOTHESIS_COUNT
        or hashlib.sha256("\n".join(eligible).encode()).hexdigest()
        != EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256
        or any(manifest.get(key) is not value for key, value in BOUNDARY_FLAGS.items())
    ):
        raise ValueError("NEXT301 NEXT300 provenance differs")
    return (*prior, eligible_table)


def _formula_from_spec(spec: Mapping[str, object] | None) -> dict[str, object]:
    if spec is None or spec.get("eligible_new_candidate") is not True:
        return {
            "protocol": PROTOCOL,
            "selected": False,
            "reason": "NO_ELIGIBLE_AUC_SAFE_CANDIDATE",
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "model_or_proxy_potential_used": False,
            "physical_relaxation_executed": False,
        }
    return {
        "protocol": PROTOCOL,
        "kind": "mospc_triangular_margin_local_x0_no_dft_score",
        "selected": True,
        "base_protocol": n223.PROTOCOL,
        "base_candidate_key_sha256": EXPECTED_BASE_CANDIDATE_KEY_SHA256,
        "base_threshold": EXPECTED_BASE_THRESHOLD,
        "repair_width": REPAIR_WIDTH,
        "hypothesis": spec["hypothesis"],
        "feature": spec["feature"],
        "direction": spec["direction"],
        "q_lo": spec["q_lo"],
        "q_hi": spec["q_hi"],
        "local_width_fraction": spec["local_width_fraction"],
        "amplitude_fraction": spec["amplitude_fraction"],
        "nonnegative_floor": 0.0,
        "normalization_population": "ALL_FINITE_COMBINED_DISCOVERY",
        "normalization_fit_uses_endpoint": False,
        "support_policy": "UNCHANGED_FROM_NEXT214",
        "missing_policy": "TERM_OFF_KEEP_NEXT224_SCORE",
        "score_composition": SCORE_COMPOSITION,
        "dft_values_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
    }


def select_best_new_record(records: pd.DataFrame) -> pd.Series | None:
    try:
        return n261.select_best_new_record(records)
    except ValueError as exc:
        raise ValueError("NEXT301 reporting selection schema differs") from exc


def _attach_mospc_features(
    *,
    combined: pd.DataFrame,
    feature_tables: Mapping[str, pd.DataFrame],
    mospc_tables: Mapping[str, pd.DataFrame],
) -> None:
    source = combined["source_dataset"].astype(str).to_numpy()
    for source_name in ("scigen", "wyformer"):
        indexed = n300._index_mospc_by_prefixed_material_id(
            table=mospc_tables[source_name],
            source=source_name,
            expected_material_ids=feature_tables[source_name]["material_id"],
        )
        mask = source == source_name
        ordered_ids = combined.loc[mask, "material_id"].astype(str)
        for name in n300.n299.FEATURE_NAMES:
            combined.loc[mask, name] = ordered_ids.map(indexed[name]).to_numpy()


def run_mospc_margin_local_search(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    stage_dirs: Mapping[int, Path],
    next135_freeze_path: Path,
    design_paths: Mapping[int, Path],
    design_path: Path,
    output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the complete frozen discovery-only NEXT301 search."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs):
        raise ValueError("NEXT301 stage directory universe differs")
    if not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT301 design path universe differs")
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{f"next{stage}": Path(stage_dirs[stage]).resolve() for stage in REQUIRED_STAGES},
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots=roots,
        next135_freeze_path=Path(next135_freeze_path).resolve(),
        design_paths={
            stage: Path(design_paths[stage]).resolve()
            for stage in REQUIRED_DESIGN_STAGES
        },
        design_path=Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(search_workers) is not int or search_workers <= 0:
        raise ValueError("NEXT301 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT301 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT301 formal input identity differs: {differing}")
    (
        eligible_prior,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        current_key,
        formula222,
        mospc_tables,
        eligible300,
    ) = _verify_next300(paths, input_hashes)
    combined, feature_tables, base_score, support, endpoint, _ = (
        n227._reconstruct_next224_frontier(
            paths=paths,
            eligible=eligible_prior,
            eligible214=eligible214,
            primary_key=primary_key,
            base_start_key=base_start_key,
            formula214=formula214,
            current_key=current_key,
            formula222=formula222,
        )
    )
    _attach_mospc_features(
        combined=combined,
        feature_tables=feature_tables,
        mospc_tables=mospc_tables,
    )
    diagnostic224 = json.loads(paths["next224_diagnostic"].read_text())
    base_key = str(diagnostic224["global_closest"]["candidate_key"])
    if (
        hashlib.sha256(base_key.encode()).hexdigest()
        != EXPECTED_BASE_CANDIDATE_KEY_SHA256
        or int(support.sum()) != EXPECTED_BASE_SUPPORT_COUNT
    ):
        raise ValueError("NEXT301 NEXT224 base identity differs")
    specs = build_mospc_candidate_specs(
        base_candidate_key=base_key,
        eligible_table=eligible300,
    )
    eligible_count = sum(bool(spec["eligible_new_candidate"]) for spec in specs)
    if len(specs) != EXPECTED_CANDIDATE_COUNT or eligible_count != EXPECTED_ELIGIBLE_COUNT:
        raise RuntimeError("NEXT301 frozen candidate universe differs")
    virtual, terms, runtime, activity = materialize_mospc_candidates(
        features=combined,
        base_score=base_score,
        base_support=support,
        specs=specs,
    )
    runtime_by_key = {str(value["candidate_key"]): value for value in runtime}
    eligible_runtime = [
        runtime_by_key[str(spec["candidate_key"])]
        for spec in specs
        if spec["eligible_new_candidate"]
    ]
    fixed_runtime = [runtime_by_key[str(specs[0]["candidate_key"])]]
    evaluator = (
        n223.n222.n215.n214.n212.n210.n208.n205.n203.n202.n200.n194.n130.n125
        .search_optional_guard_laws_parallel
    )
    started = time.perf_counter()
    eligible_result = evaluator(
        features=virtual,
        endpoint=endpoint,
        old_terms=terms,
        optional_terms=[],
        candidate_specs=eligible_runtime,
        workers=search_workers,
    )
    fixed_result = evaluator(
        features=virtual,
        endpoint=endpoint,
        old_terms=terms,
        optional_terms=[],
        candidate_specs=fixed_runtime,
        workers=search_workers,
    )
    elapsed = time.perf_counter() - started
    if (
        int(eligible_result["candidate_count"]) != EXPECTED_ELIGIBLE_COUNT
        or int(fixed_result["candidate_count"]) != 1
        or eligible_result["cells"] != fixed_result["cells"]
        or eligible_result["pauling_by_cell"] != fixed_result["pauling_by_cell"]
    ):
        raise RuntimeError("NEXT301 evaluator accounting differs")
    spec_by_key = {str(spec["candidate_key"]): spec for spec in specs}
    raw_records = [
        *eligible_result["candidate_records"],
        *fixed_result["candidate_records"],
    ]
    for record in raw_records:
        spec = spec_by_key[str(record["candidate_key"])]
        record.update(
            {
                "hypothesis": spec["hypothesis"],
                "feature": spec["feature"],
                "direction": spec["direction"],
                "q_lo": spec["q_lo"],
                "q_hi": spec["q_hi"],
                "local_width_fraction": spec["local_width_fraction"],
                "local_width_numerator": spec["local_width_numerator"],
                "amplitude_fraction": spec["amplitude_fraction"],
                "amplitude_numerator": spec["amplitude_numerator"],
                "eligible_new_candidate": spec["eligible_new_candidate"],
                "is_reproduction_control": spec["is_reproduction_control"],
                "local_active_rows": activity[str(record["candidate_key"])]["rows"],
                "local_active_scigen": activity[str(record["candidate_key"])]["scigen"],
                "local_active_wyformer": activity[str(record["candidate_key"])]["wyformer"],
                "normalization_population": spec["normalization_population"],
                "missing_policy": spec["missing_policy"],
                "score_composition": SCORE_COMPOSITION,
            }
        )
    records = pd.DataFrame(raw_records).sort_values(
        "candidate_key", kind="mergesort"
    ).reset_index(drop=True)
    no_op_key = str(specs[0]["candidate_key"])
    no_op = records.loc[records["candidate_key"].eq(no_op_key)]
    table223 = pd.read_parquet(paths["next223_search"])
    reference_no_op = table223.loc[table223["candidate_key"].eq(base_key)]
    if len(no_op) != 1 or len(reference_no_op) != 1:
        raise RuntimeError("NEXT301 no-op reproduction identity differs")
    n223._assert_record_reproduction(no_op.iloc[0], reference_no_op.iloc[0])
    eligible_frame = records.loc[records["eligible_new_candidate"].astype(bool)]
    auc_safe_mask = (
        eligible_frame["passes_source_auc_gates"].fillna(False).astype(bool)
        & eligible_frame["passes_safe_all_cells"].fillna(False).astype(bool)
    )
    selected_row = select_best_new_record(records)
    selected: dict[str, object] | None = None
    selected_spec: dict[str, object] | None = None
    if selected_row is not None:
        selected_key = str(selected_row["candidate_key"])
        selected_result = evaluator(
            features=virtual,
            endpoint=endpoint,
            old_terms=terms,
            optional_terms=[],
            candidate_specs=[runtime_by_key[selected_key]],
            workers=1,
        )
        if str(selected_result["selected"]["record"]["candidate_key"]) != selected_key:
            raise RuntimeError("NEXT301 selected candidate reproduction differs")
        selected = selected_result["selected"]
        selected_spec = spec_by_key[selected_key]
        if selected_spec.get("eligible_new_candidate") is not True:
            raise RuntimeError("NEXT301 reproduction control was selected")
        for name, value in selected_row.items():
            if name in selected["record"]:
                selected["record"][name] = value
    passes = bool(
        eligible_frame["passes_all_discovery_gates"].fillna(False).astype(bool).any()
    )
    if selected is not None and passes != bool(
        selected["record"]["passes_all_discovery_gates"]
    ):
        raise RuntimeError("NEXT301 all-gate selection differs")
    if passes and selected is None:
        raise RuntimeError("NEXT301 all-gate candidate was not selected")
    diagnostic_mask = auc_safe_mask & ~eligible_frame[
        "passes_broad_all_cells"
    ].fillna(False).astype(bool)
    diagnostic_keys = sorted(
        eligible_frame.loc[diagnostic_mask, "candidate_key"].astype(str)
    )
    diagnostic_sha = hashlib.sha256("\n".join(diagnostic_keys).encode()).hexdigest()
    formula = _formula_from_spec(selected_spec)
    catalogue = {
        "protocol": PROTOCOL,
        "design_sha256": input_hashes["design"],
        "base_candidate_key_sha256": EXPECTED_BASE_CANDIDATE_KEY_SHA256,
        "base_threshold": EXPECTED_BASE_THRESHOLD,
        "repair_width": REPAIR_WIDTH,
        "eligible_hypotheses": list(eligible300["hypothesis"].astype(str)),
        "eligible_hypothesis_count": len(eligible300),
        "eligible_hypothesis_sha256": EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256,
        "local_width_fractions": list(LOCAL_WIDTH_FRACTIONS),
        "amplitude_fractions": list(AMPLITUDE_FRACTIONS),
        "candidate_count": len(records),
        "eligible_new_candidate_count": len(eligible_frame),
        "reproduction_control_count": 1,
        "normalization_population": "ALL_FINITE_COMBINED_DISCOVERY",
        "normalization_fit_uses_endpoint": False,
        "base_support_unchanged": True,
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    evaluation = {
        "protocol": PROTOCOL,
        "evaluation_mode": "frozen_mospc_margin_local_discovery_search",
        "next224_frontier_reproduced": True,
        "rows": {
            "scigen": int(len(feature_tables["scigen"])),
            "wyformer": int(len(feature_tables["wyformer"])),
            "total": int(len(combined)),
        },
        "candidate_count": len(records),
        "eligible_new_candidate_count": len(eligible_frame),
        "reproduction_control_count": 1,
        "elapsed_seconds": elapsed,
        "search_workers": search_workers,
        "counts_all": n223._gate_counts(records),
        "counts_eligible_new": n223._gate_counts(eligible_frame),
        "selected_record": None if selected is None else selected["record"],
        "selected_formula": formula,
        "selected_safe": None if selected is None else selected["safe"],
        "selected_safe_diagnostic": None if selected is None else selected["safe_diagnostic"],
        "selected_broad": None if selected is None else selected["broad"],
        "selected_source_diagnostics": None if selected is None else selected["source_diagnostics"],
        "pauling_by_cell": eligible_result["pauling_by_cell"],
        "cells": eligible_result["cells"],
        "passes_all_cross_source_discovery_gates": passes,
        "freeze_authorized": passes,
        "next302_diagnostic_authorized": bool(not passes and diagnostic_keys),
        "next302_candidate_count": len(diagnostic_keys),
        "next302_candidate_key_sha256": diagnostic_sha,
        "requires_unopened_internal_validation_before_claim": True,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next261_pvbp_margin_local_search.py": Path(n261.__file__).resolve(),
        "src/next300_mospc_feature_audit.py": Path(n300.__file__).resolve(),
        "src/next301_mospc_margin_local_search.py": Path(__file__).resolve(),
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
            "candidate_count": len(records),
            "eligible_new_candidate_count": len(eligible_frame),
            "reproduction_control_count": 1,
            "eligible_hypothesis_count": len(eligible300),
            "eligible_hypothesis_sha256": EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256,
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "next302_diagnostic_authorized": bool(not passes and diagnostic_keys),
            "next302_candidate_count": len(diagnostic_keys),
            "next302_candidate_key_sha256": diagnostic_sha,
            "requires_unopened_internal_validation_before_claim": True,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True,
            **BOUNDARY_FLAGS,
            "scientific_improvement_claim": False,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {path.name: _sha256_file(path) for path in outputs},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT301 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT301 source changed before publication")
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
    for stage in REQUIRED_STAGES:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    for stage in REQUIRED_DESIGN_STAGES:
        parser.add_argument(f"--next{stage}-design-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_mospc_margin_local_search(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        stage_dirs={stage: getattr(args, f"next{stage}_dir") for stage in REQUIRED_STAGES},
        next135_freeze_path=args.next135_freeze_path,
        design_paths={
            stage: getattr(args, f"next{stage}_design_path")
            for stage in REQUIRED_DESIGN_STAGES
        },
        design_path=args.design_path,
        output_dir=args.output_dir,
        search_workers=args.search_workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
