#!/usr/bin/env python3
"""Frozen sparse low-residual exemption search on coordination-weight-2 bases."""

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
import src.next134_compactness_protection_search as n134
import src.next135_conjunctive_compactness_search as n135
import src.next142_threshold_local_retention_audit as n142
import src.next143_coulomb_steric_balance_protection as n143
import src.next145_coulomb_steric_balance_broad_residual_diagnostic as n145
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next87_scigen_sparse_law_search import _term_risk


PROTOCOL = "2026-08-08-next146-conditional-balance-exemption-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT146_CONDITIONAL_BALANCE_EXEMPTION_SEARCH_CATALOGUE.json"
EVALUATION_NAME = "NEXT146_DISCOVERY_EVALUATION.json"
SEARCH_NAME = "next146_conditional_balance_exemption_candidate_search.parquet"
EXEMPTION_TERM_ID = "coulomb_steric_low_residual_exemption__high"
CUTOFFS = (0.05, 0.1, 0.25, 0.5)
WEIGHTS = (0.1, 0.25, 0.5, 1.0, 2.0)
SEARCH_WORKERS = 4
EXPECTED_FREEZE_SHA256 = "b672f2d8da126ca78f56ee62b2dea71b62227d599444d5a81e71edc60f67d4a1"
EXPECTED_CUTOFF_GRID_SHA256 = "c2bba8220bc9a9ee6aef1b393550d2801041a0c91ec260cb2f0e3f9c18f970ad"
EXPECTED_WEIGHT_GRID_SHA256 = "659780d2efe42b0ba01a6ae98bfc90493027eb32bc6ad1b456f8fff97f4512ef"
EXPECTED_BASE_COUNT = 11
EXPECTED_CANDIDATE_COUNT = 231
EXPECTED_CANDIDATE_KEY_SHA256 = "3ea4eb3c1817656f8e9ed24e35750cde30665255b48f223513e36fbb793727c4"
EXPECTED_INPUT_SHA256 = {
    **{key: value for key, value in n135.EXPECTED_INPUT_SHA256.items() if key != "freeze"},
    "freeze": EXPECTED_FREEZE_SHA256,
    "next142_manifest": "ad2a0152966bccda7deddd30b185e1cefdb22ba647d8332b45539313b746c2d7",
    "next142_audit": "8b407db841a508d66c243579f324ef4c20cdc355a1d58621dfdb4bf9ecc9b1ed",
    "next142_feature_table": "6f4efb1597c492cb5ee84c35f1a2dbcb4f8f8684fb5a84b0254e948994dd689a",
    "next143_manifest": "83a81449cd330782a57bfe2c710402c50d5d987478feb3faecdc0a4438a52766",
    "next143_catalogue": "111c8a8536d559e102174fb3be010950557ae352e85860d3c7e34d6a5aee86eb",
    "next143_scigen_features": "eb87ba3a7d79da4aaf1a3765e968edf57b22a98b4ba931f7e08e42941ef7f9fb",
    "next143_wyformer_features": "eda837cec63100b785c0e06f9e54e71e80f8eb9f238c3a55bde2ec44f267351c",
    "next145_manifest": "e6a82cc52370c9dba8e84f7453e9f70760670a8b70d0ff8101423b0ebeb2c501",
    "next145_diagnostic": "4b4268f0a6166db327b3b6912b47a92843a3ec3dc105278b0714e632a6ee8f87",
    "next145_per_candidate": "cadf28556f8afb478ab54ee5254efef4e68cd49679379e565329004d34e986fd",
}


def apply_conditional_balance_exemption(
    *,
    base_score: object,
    base_supported: object,
    residual: object,
    residual_supported: object,
    cutoff: float | None,
    weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = np.asarray(base_score, dtype=float)
    supported = np.asarray(base_supported, dtype=bool)
    values = np.asarray(residual, dtype=float)
    residual_ok = np.asarray(residual_supported, dtype=bool)
    if supported.shape != score.shape or values.shape != score.shape or residual_ok.shape != score.shape:
        raise ValueError("NEXT146 exemption arrays differ")
    baseline = cutoff is None and weight == 0.0
    active_config = cutoff in CUTOFFS and weight in WEIGHTS
    if not (baseline or active_config) or np.any(residual_ok & (~np.isfinite(values) | (values < 0.0) | (values > 1.0 + 1e-12))):
        raise ValueError("NEXT146 exemption values differ")
    result = np.zeros(score.shape, dtype=float)
    result[supported] = score[supported]
    use = np.zeros(score.shape, dtype=bool) if baseline else supported & residual_ok & (values <= float(cutoff))
    result[use] = np.maximum(0.0, result[use] - float(weight))
    return result, supported.copy(), use


def build_candidate_specs(
    *, bases: pd.DataFrame, physical_term_ids: set[str]
) -> list[dict[str, object]]:
    specs: dict[str, dict[str, object]] = {}
    for _, row in bases.iterrows():
        base_ids = [str(value) for value in json.loads(str(row["term_ids_json"]))]
        base_weights = [float(value) for value in json.loads(str(row["weights_json"]))]
        if not base_ids or len(base_ids) != len(base_weights) or any(term_id not in physical_term_ids for term_id in base_ids):
            raise ValueError("NEXT146 base formula differs")
        configurations = [(None, 0.0), *[(cutoff, weight) for cutoff in CUTOFFS for weight in WEIGHTS]]
        for cutoff, weight in configurations:
            payload = {
                "base_term_ids": base_ids,
                "base_weights": base_weights,
                "coordination_protection_term_id": n130.PROTECTION_TERM_ID,
                "coordination_protection_weight": 2.0,
                "conditional_exemption_term_id": None if cutoff is None else EXEMPTION_TERM_ID,
                "coulomb_steric_residual_cutoff": cutoff,
                "conditional_exemption_weight": float(weight),
            }
            key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            specs[key] = {"candidate_key": key, **payload}
    return [specs[key] for key in sorted(specs)]


def materialize_candidates(
    *,
    features: pd.DataFrame,
    coordination_terms: Sequence[Mapping[str, object]],
    coordination_by_formula: Mapping[str, str],
    specs: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    required = {n143.FEATURE_NAME, n143.SUPPORT_COLUMN}
    if required - set(features.columns):
        raise ValueError("NEXT146 protection feature schema differs")
    protection = pd.to_numeric(features[n143.FEATURE_NAME], errors="coerce").to_numpy(float)
    residual_supported = features[n143.SUPPORT_COLUMN].eq(True).to_numpy()
    if np.any(residual_supported & (~np.isfinite(protection) | (protection < 0.0) | (protection > 1.0 + 1e-12))):
        raise ValueError("NEXT146 protection feature values differ")
    residual = np.full(len(features), np.nan, dtype=float)
    residual[residual_supported] = 1.0 - protection[residual_supported]
    base_by_id = {str(term["term_id"]): dict(term) for term in coordination_terms}
    base_risks = {term_id: _term_risk(features, term) for term_id, term in base_by_id.items()}
    columns: dict[str, np.ndarray] = {}
    terms: list[dict[str, object]] = []
    runtime: list[dict[str, object]] = []
    for spec_raw in specs:
        spec = dict(spec_raw)
        identity = n130.n127._formula_identity(spec["base_term_ids"], spec["base_weights"])
        base_id = coordination_by_formula.get(identity)
        cutoff = spec["coulomb_steric_residual_cutoff"]
        weight = float(spec["conditional_exemption_weight"])
        expected_term = None if cutoff is None else EXEMPTION_TERM_ID
        if base_id is None or spec["conditional_exemption_term_id"] != expected_term:
            raise ValueError("NEXT146 candidate configuration differs")
        score, supported, _ = apply_conditional_balance_exemption(
            base_score=base_risks[base_id][0],
            base_supported=base_risks[base_id][1],
            residual=residual,
            residual_supported=residual_supported,
            cutoff=cutoff,
            weight=weight,
        )
        maximum = float(np.max(score[supported])) if supported.any() else 0.0
        divisor = max(1.0, maximum / 50.0)
        encoded = np.full(len(features), np.nan, dtype=float)
        encoded[supported] = np.sinh(score[supported] / divisor)
        key = str(spec["candidate_key"])
        virtual_id = f"next146_virtual_candidate__{hashlib.sha256(key.encode()).hexdigest()[:24]}"
        feature_name = f"_{virtual_id}_value"
        columns[feature_name] = encoded
        terms.append(
            {
                "term_id": virtual_id,
                "feature": feature_name,
                "direction": 1,
                "transform": "asinh",
                "center": 0.0,
                "scale": 1.0 / divisor,
                "group": "next146_conditional_balance_exemption_candidate",
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
    return pd.concat([features.reset_index(drop=True), pd.DataFrame(columns)], axis=1), terms, runtime


def verify_base_reproduction(
    *, result_records: Sequence[Mapping[str, object]], prior: pd.DataFrame
) -> None:
    metrics = (
        "scigen_pooled_auc", "scigen_macro_auc", "scigen_worst_auc",
        "wyformer_pooled_auc", "wyformer_macro_auc", "wyformer_worst_auc",
    )
    observed = {}
    for record in result_records:
        payload = json.loads(str(record["candidate_key"]))
        if payload["coulomb_steric_residual_cutoff"] is not None:
            continue
        observed[n130.n127._formula_identity(payload["base_term_ids"], payload["base_weights"])] = record
    expected = {
        n130.n127._formula_identity(json.loads(str(row["term_ids_json"])), json.loads(str(row["weights_json"]))): row["_next130_record"]
        for _, row in prior.iterrows()
    }
    if set(observed) != set(expected):
        raise RuntimeError("NEXT146 base reproduction identities differ")
    for identity, source in expected.items():
        record = observed[identity]
        if any(not math.isclose(float(record[name]), float(source[name]), rel_tol=0.0, abs_tol=n130.BASE_REPRODUCTION_AUC_TOLERANCE) for name in metrics) or any(bool(record[name]) != bool(source[name]) for name in ("passes_source_auc_gates", "passes_safe_all_cells")) or int(record["safe_passing_cells"]) != int(source["safe_passing_cells"]):
            raise RuntimeError("NEXT146 base diagnostics do not reproduce NEXT130")


def _paths(roots: Mapping[str, Path], freeze_path: Path) -> dict[str, Path]:
    paths = n135._paths(roots, freeze_path)
    paths.update(
        {
            "next142_manifest": roots["next142"] / n142.MANIFEST_NAME,
            "next142_audit": roots["next142"] / n142.AUDIT_NAME,
            "next142_feature_table": roots["next142"] / n142.FEATURE_TABLE_NAME,
            "next143_manifest": roots["next143"] / n143.MANIFEST_NAME,
            "next143_catalogue": roots["next143"] / n143.CATALOGUE_NAME,
            "next143_scigen_features": roots["next143"] / n143.FEATURE_FILES["scigen"],
            "next143_wyformer_features": roots["next143"] / n143.FEATURE_FILES["wyformer"],
            "next145_manifest": roots["next145"] / n145.MANIFEST_NAME,
            "next145_diagnostic": roots["next145"] / n145.DIAGNOSTIC_NAME,
            "next145_per_candidate": roots["next145"] / n145.PER_CANDIDATE_NAME,
        }
    )
    return paths


def run_conditional_balance_exemption_search(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    next98_dir: Path, next110_dir: Path, next111_dir: Path, next113_dir: Path,
    next114_dir: Path, next116_dir: Path, next117_dir: Path, next120_dir: Path,
    next121_dir: Path, next122_dir: Path, next124_dir: Path, next125_dir: Path,
    next129_dir: Path, next130_dir: Path, next133_dir: Path, next134_dir: Path,
    next142_dir: Path, next143_dir: Path, next145_dir: Path,
    freeze_path: Path,
    output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{f"next{stage}": Path(value).resolve() for stage, value in (
            (98,next98_dir),(110,next110_dir),(111,next111_dir),(113,next113_dir),
            (114,next114_dir),(116,next116_dir),(117,next117_dir),(120,next120_dir),
            (121,next121_dir),(122,next122_dir),(124,next124_dir),(125,next125_dir),
            (129,next129_dir),(130,next130_dir),(133,next133_dir),(134,next134_dir),
            (142,next142_dir),(143,next143_dir),(145,next145_dir),
        )},
    }
    target = Path(output_dir).resolve()
    paths = _paths(roots, Path(freeze_path).resolve())
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT146 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(key for key in set(input_hashes) | set(EXPECTED_INPUT_SHA256) if input_hashes.get(key) != EXPECTED_INPUT_SHA256.get(key))
        raise ValueError(f"NEXT146 formal input identity differs: {differing}")
    manifest142 = json.loads(paths["next142_manifest"].read_text())
    manifest143 = json.loads(paths["next143_manifest"].read_text())
    manifest145 = json.loads(paths["next145_manifest"].read_text())
    if (
        manifest142.get("protocol") != n142.PROTOCOL
        or manifest142.get("opened_validation_outputs_used") is not False
        or manifest142.get("outputs_sha256", {}).get(n142.AUDIT_NAME) != input_hashes["next142_audit"]
        or manifest142.get("outputs_sha256", {}).get(n142.FEATURE_TABLE_NAME) != input_hashes["next142_feature_table"]
        or manifest143.get("protocol") != n143.PROTOCOL
        or manifest143.get("labels_opened") is not False
        or manifest143.get("endpoint_payloads_opened") is not False
        or manifest143.get("dft_values_used_by_features") is not False
        or manifest143.get("outputs_sha256", {}).get(n143.FEATURE_FILES["scigen"]) != input_hashes["next143_scigen_features"]
        or manifest143.get("outputs_sha256", {}).get(n143.FEATURE_FILES["wyformer"]) != input_hashes["next143_wyformer_features"]
        or manifest145.get("protocol") != n145.PROTOCOL
        or manifest145.get("coulomb_steric_balance_branch_terminated") is not True
        or manifest145.get("opened_validation_outputs_used") is not False
        or manifest145.get("outputs_sha256", {}).get(n145.DIAGNOSTIC_NAME) != input_hashes["next145_diagnostic"]
    ):
        raise ValueError("NEXT146 prior provenance differs")

    extended, feature_tables, old_terms, mhcr_terms = n130._join_label_free_features(paths)
    protection_frames = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next143_{source}_features"]).copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        protection_frames.append(table)
    protection = pd.concat(protection_frames, ignore_index=True)
    extended = extended.merge(protection, on="material_id", how="inner", validate="one_to_one")
    if len(extended) != len(protection):
        raise ValueError("NEXT146 protection row accounting differs")
    physical_terms = [*old_terms, *mhcr_terms]
    physical_ids = {str(term["term_id"]) for term in physical_terms}
    all_bases = n130.n127.select_next125_bases(pd.read_parquet(paths["next125_search_records"]))
    bases = n132.select_extended_bases(pd.read_parquet(paths["next130_search_records"]), all_bases)
    specs = build_candidate_specs(bases=bases, physical_term_ids=physical_ids)
    candidate_sha = hashlib.sha256("\n".join(str(spec["candidate_key"]) for spec in specs).encode()).hexdigest()
    cutoff_sha = hashlib.sha256(json.dumps(list(CUTOFFS), separators=(",", ":")).encode()).hexdigest()
    weight_sha = hashlib.sha256(json.dumps(list(WEIGHTS), separators=(",", ":")).encode()).hexdigest()
    if require_formal_inputs and (
        len(bases) != EXPECTED_BASE_COUNT
        or len(specs) != EXPECTED_CANDIDATE_COUNT
        or candidate_sha != EXPECTED_CANDIDATE_KEY_SHA256
        or cutoff_sha != EXPECTED_CUTOFF_GRID_SHA256
        or weight_sha != EXPECTED_WEIGHT_GRID_SHA256
    ):
        raise ValueError("NEXT146 frozen candidate universe differs")

    scigen_endpoint = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_endpoint = pd.read_parquet(paths["wyformer_endpoint"])
    endpoint_frame = pd.concat(
        [
            pd.DataFrame({"material_id":"scigen:"+scigen_endpoint["material_id"].astype(str),"_endpoint":pd.to_numeric(scigen_endpoint["distortion_ratio"],errors="coerce")}),
            pd.DataFrame({"material_id":"wyformer:"+wyformer_endpoint["material_id"].astype(str),"_endpoint":n130.n125.n121.prior._endpoint_numeric(wyformer_endpoint["endpoint_stratum"])}),
        ], ignore_index=True,
    )
    combined = extended.merge(endpoint_frame, on="material_id", how="inner", validate="one_to_one")
    endpoint = pd.to_numeric(combined.pop("_endpoint"), errors="coerce").to_numpy(float)
    combined, base_virtual_terms, base_virtual_by_formula = n130.n127.materialize_virtual_bases(features=combined, bases=bases, old_terms=old_terms, mhcr_terms=mhcr_terms)
    combined, coordination_terms, coordination_by_formula = n134.materialize_coordination_bases(features=combined, bases=bases, base_virtual_terms=base_virtual_terms, base_virtual_by_formula=base_virtual_by_formula)
    combined, virtual_terms, runtime = materialize_candidates(features=combined, coordination_terms=coordination_terms, coordination_by_formula=coordination_by_formula, specs=specs)
    started = time.perf_counter()
    result = n130.n125.search_optional_guard_laws_parallel(features=combined, endpoint=endpoint, old_terms=virtual_terms, optional_terms=[], candidate_specs=runtime, workers=search_workers)
    elapsed = time.perf_counter() - started
    verify_base_reproduction(result_records=result["candidate_records"], prior=bases)

    physical_by_id = {str(term["term_id"]): dict(term) for term in physical_terms}
    def decorate(record: dict[str, object]) -> None:
        payload = json.loads(str(record["candidate_key"])); evaluated = json.loads(str(record["base_term_ids_json"]))
        record["evaluation_virtual_term_id"] = str(evaluated[0])
        record["base_term_ids_json"] = json.dumps(payload["base_term_ids"], separators=(",", ":"))
        record["base_weights_json"] = json.dumps(payload["base_weights"], separators=(",", ":"))
        record["coordination_protection_weight"] = 2.0
        record["conditional_exemption_term_id"] = payload["conditional_exemption_term_id"]
        record["coulomb_steric_residual_cutoff"] = payload["coulomb_steric_residual_cutoff"]
        record["conditional_exemption_weight"] = float(payload["conditional_exemption_weight"])
        record["score_composition"] = "max(0,coordination_weight2_base-exemption_weight*I(acsb_site_residual_max<=cutoff))"
    for record in result["candidate_records"]: decorate(record)
    selected = result["selected"]
    if "evaluation_virtual_term_id" not in selected["record"]: decorate(selected["record"])
    payload = json.loads(str(selected["record"]["candidate_key"])); formula = selected["formula"]
    formula["evaluation_virtual_term_id"] = str(formula["base_terms"][0]["term_id"])
    formula["base_terms"] = [{**physical_by_id[str(term_id)],"weight":float(weight)} for term_id,weight in zip(payload["base_term_ids"],payload["base_weights"],strict=True)]
    formula["coordination_protection"] = {"term_id":n130.PROTECTION_TERM_ID,"feature":n130.n129.FEATURE_NAME,"weight":2.0,"missing_policy":"TERM_OFF_KEEP_BASE"}
    formula["conditional_balance_exemption"] = None if payload["coulomb_steric_residual_cutoff"] is None else {"term_id":EXEMPTION_TERM_ID,"raw_feature":n143.RAW_FEATURE,"cutoff":float(payload["coulomb_steric_residual_cutoff"]),"weight":float(payload["conditional_exemption_weight"]),"missing_policy":"EXEMPTION_OFF_KEEP_BASE"}
    formula["score_composition"] = "max(0,coordination_weight2_base-exemption_weight*I(acsb_site_residual_max<=cutoff))"
    formula["kind"] = "next130_coordination_base_with_optional_conditional_balance_exemption"
    passes = bool(selected["record"]["passes_all_discovery_gates"])
    records_frame = pd.DataFrame(result["candidate_records"])
    records_frame["exemption_configuration"] = records_frame.apply(lambda row: "baseline" if pd.isna(row["coulomb_steric_residual_cutoff"]) else f"r<={float(row['coulomb_steric_residual_cutoff']):g}:w={float(row['conditional_exemption_weight']):g}",axis=1)
    counts = {str(config):{"candidates":int(len(frame)),"passes_source_auc_gates":int(frame["passes_source_auc_gates"].sum()),"passes_safe_all_cells":int(frame["passes_safe_all_cells"].sum()),"passes_broad_all_cells":int(frame["passes_broad_all_cells"].sum()),"passes_all_discovery_gates":int(frame["passes_all_discovery_gates"].sum())} for config,frame in records_frame.groupby("exemption_configuration",sort=True)}
    catalogue = {"protocol":PROTOCOL,"freeze_sha256":input_hashes["freeze"],"base_count":len(bases),"candidate_count":len(specs),"cutoff_grid":list(CUTOFFS),"cutoff_grid_sha256":cutoff_sha,"weight_grid":list(WEIGHTS),"weight_grid_sha256":weight_sha,"candidate_key_sha256":candidate_sha,"exemption_term_id":EXEMPTION_TERM_ID,"raw_feature":n143.RAW_FEATURE,"active_score":"max(0,coordination_weight2_base-exemption_weight*I(acsb_site_residual_max<=cutoff))","validation_outputs_opened":False,"dft_values_used_by_executable_formula":False,"analytic_coulomb_steric_vectors_used":True}
    catalogue_sha = hashlib.sha256(json.dumps(catalogue,indent=2,sort_keys=True).encode()+b"\n").hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {"src/next130_coordination_protection_search.py":Path(n130.__file__).resolve(),"src/next134_compactness_protection_search.py":Path(n134.__file__).resolve(),"src/next146_conditional_balance_exemption_search.py":Path(__file__).resolve()}
    source_hashes = {name:_sha256_file(path) for name,path in source_paths.items()}
    try:
        catalogue_path=staging/CATALOGUE_NAME; evaluation_path=staging/EVALUATION_NAME; search_path=staging/SEARCH_NAME
        _write_json(catalogue_path,{**catalogue,"label_free_catalogue_sha256":catalogue_sha})
        _write_json(evaluation_path,{"protocol":PROTOCOL,"evaluation_mode":"fixed_conditional_balance_exemption","rows":{"scigen":int(len(feature_tables["scigen"])),"wyformer":int(len(feature_tables["wyformer"])),"total":int(len(combined))},"base_count":len(bases),"candidate_count":int(result["candidate_count"]),"elapsed_seconds":elapsed,"search_workers":search_workers,"base_only_reproduced_next130":True,"counts_by_exemption_configuration":counts,"safe_gates":dict(n130.n125.n121.prior.DEFAULT_GATES),"source_auc_gates":dict(n130.n125.n121.prior.AUC_GATES),"broad_min_severe_precision_lower":n130.n125.n121.prior.BROAD_MIN_PRECISION_LOWER,"selected_record":selected["record"],"selected_formula":selected["formula"],"selected_safe":selected["safe"],"selected_safe_diagnostic":selected["safe_diagnostic"],"selected_broad":selected["broad"],"selected_source_diagnostics":selected["source_diagnostics"],"pauling_by_cell":result["pauling_by_cell"],"cells":result["cells"],"passes_all_cross_source_discovery_gates":passes,"freeze_authorized":passes,"requires_unopened_internal_validation_before_claim":True})
        records_frame.to_parquet(search_path,index=False)
        manifest={"protocol":PROTOCOL,"label_free_catalogue_sha256":catalogue_sha,"base_count":len(bases),"candidate_count":int(result["candidate_count"]),"search_workers":search_workers,"base_only_reproduced_next130":True,"passes_all_cross_source_discovery_gates":passes,"freeze_authorized":passes,"requires_unopened_internal_validation_before_claim":True,"scigen_discovery_endpoint_opened":True,"wyformer_discovery_endpoint_opened":True,"discovery_outcomes_used_as_offline_labels":True,"opened_validation_outputs_used":False,"scigen_replication_endpoint_opened":False,"wyformer_replication_endpoint_opened":False,"dft_calculation_executed":False,"dft_values_used_by_executable_formula":False,"learned_energy_force_stress_proxy_used":False,"analytic_coulomb_steric_vectors_used":True,"physical_relaxation_executed":False,"formula_or_threshold_changed_after_search":False,"scientific_improvement_claim":False,"inputs_sha256":input_hashes,"executed_source_sha256":source_hashes,"outputs_sha256":{CATALOGUE_NAME:_sha256_file(catalogue_path),EVALUATION_NAME:_sha256_file(evaluation_path),SEARCH_NAME:_sha256_file(search_path)}}
        _write_json(staging/MANIFEST_NAME,manifest)
        if any(_sha256_file(path)!=input_hashes[name] for name,path in paths.items()): raise RuntimeError("NEXT146 input changed before publication")
        if any(_sha256_file(path)!=source_hashes[name] for name,path in source_paths.items()): raise RuntimeError("NEXT146 source changed before publication")
        os.replace(staging,target); return manifest
    except Exception:
        shutil.rmtree(staging,ignore_errors=True); raise


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scigen-feature-dir",type=Path,required=True);parser.add_argument("--scigen-discovery-endpoint-dir",type=Path,required=True)
    parser.add_argument("--wyformer-feature-dir",type=Path,required=True);parser.add_argument("--wyformer-discovery-endpoint-dir",type=Path,required=True)
    for stage in (98,110,111,113,114,116,117,120,121,122,124,125,129,130,133,134,142,143,145): parser.add_argument(f"--next{stage}-dir",type=Path,required=True)
    parser.add_argument("--freeze-path",type=Path,required=True);parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--search-workers",type=int,default=SEARCH_WORKERS);parser.add_argument("--allow-nonformal-inputs",action="store_true")
    args=parser.parse_args()
    manifest=run_conditional_balance_exemption_search(scigen_feature_dir=args.scigen_feature_dir,scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,wyformer_feature_dir=args.wyformer_feature_dir,wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,**{f"next{stage}_dir":getattr(args,f"next{stage}_dir") for stage in (98,110,111,113,114,116,117,120,121,122,124,125,129,130,133,134,142,143,145)},freeze_path=args.freeze_path,output_dir=args.output_dir,search_workers=args.search_workers,require_formal_inputs=not args.allow_nonformal_inputs)
    print(json.dumps(manifest,indent=2,sort_keys=True));return 0


if __name__ == "__main__": raise SystemExit(main())


__all__=["apply_conditional_balance_exemption","build_candidate_specs","materialize_candidates","run_conditional_balance_exemption_search"]
