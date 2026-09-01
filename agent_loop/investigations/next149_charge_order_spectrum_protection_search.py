#!/usr/bin/env python3
"""Frozen formal-charge spectrum protection search on coordination-weight-2 bases."""

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
import src.next147_conditional_balance_exemption_broad_residual_diagnostic as n147
import src.next148_charge_order_spectrum_protection as n148
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next87_scigen_sparse_law_search import _term_risk


PROTOCOL = "2026-08-08-next149-charge-order-spectrum-protection-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT149_CHARGE_ORDER_SPECTRUM_PROTECTION_SEARCH_CATALOGUE.json"
EVALUATION_NAME = "NEXT149_DISCOVERY_EVALUATION.json"
SEARCH_NAME = "next149_charge_order_spectrum_protection_candidate_search.parquet"
PROTECTION_TERM_ID = "charge_order_spectrum_protection__high"
WEIGHTS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
SEARCH_WORKERS = 4
EXPECTED_FREEZE_SHA256 = "8c2c3877a355f21f48dfe227b3100e3d835cbde7048f29afad69f039991fa4f4"
EXPECTED_WEIGHT_GRID_SHA256 = "4420f38ada0fb6b6f0585f0db73d387e9425f62ac7b1735c7d16ab1922d3c7a7"
EXPECTED_BASE_COUNT = 11
EXPECTED_CANDIDATE_COUNT = 88
EXPECTED_CANDIDATE_KEY_SHA256 = "6bb64825118b08264598350af117d9dcd971066cdcd3cdac004d6ab2d530e146"
EXPECTED_INPUT_SHA256 = {
    **{key: value for key, value in n135.EXPECTED_INPUT_SHA256.items() if key != "freeze"},
    "freeze": EXPECTED_FREEZE_SHA256,
    "next142_manifest": "ad2a0152966bccda7deddd30b185e1cefdb22ba647d8332b45539313b746c2d7",
    "next142_audit": "8b407db841a508d66c243579f324ef4c20cdc355a1d58621dfdb4bf9ecc9b1ed",
    "next142_feature_table": "6f4efb1597c492cb5ee84c35f1a2dbcb4f8f8684fb5a84b0254e948994dd689a",
    "next147_manifest": "df83454b498675c8e493d31a9dbb39fb1581a04c6d7a281cc8d777c6a5c2c1e9",
    "next147_diagnostic": "09123e1817292b4bb0ed75f8622861dc26ac07e9c8f6ada5d67787388ea347bb",
    "next147_per_candidate": "e063a26a44f6bab9154830110f49781dcd71a74e3e0608b09bb6e746907bc6da",
    "next148_manifest": "c84de55c5e8ba186432f7fcb9b9f55e361f62717345166cad78e77ab933c617c",
    "next148_catalogue": "17113e203d4f1d9c7aed827736059a427e7386449071d1be3b4a5b0c97c49427",
    "next148_scigen_features": "2e5dbd427b64d1e4df461e5af856eaf463b1e509d2fbbce9c9c5b369003075a1",
    "next148_wyformer_features": "64b9d61c49ac18b3d2596e68038bfe49e95b7f4de6d02f54ac16b10dd3376749",
}


def apply_charge_order_spectrum_protection(
    *,
    base_score: object,
    base_supported: object,
    protection: object,
    protection_active: object,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(base_score, dtype=float)
    supported = np.asarray(base_supported, dtype=bool)
    values = np.asarray(protection, dtype=float)
    active = np.asarray(protection_active, dtype=bool)
    if supported.shape != score.shape or values.shape != score.shape or active.shape != score.shape:
        raise ValueError("NEXT149 protection arrays differ")
    if weight not in WEIGHTS or np.any(active & (~np.isfinite(values) | (values < 0.0))):
        raise ValueError("NEXT149 protection values differ")
    result = np.zeros(score.shape, dtype=float)
    result[supported] = score[supported]
    use = supported & active
    result[use] = np.maximum(0.0, result[use] - float(weight) * values[use])
    return result, supported.copy()


def build_candidate_specs(
    *, bases: pd.DataFrame, physical_term_ids: set[str]
) -> list[dict[str, object]]:
    specs: dict[str, dict[str, object]] = {}
    for _, row in bases.iterrows():
        base_ids = [str(value) for value in json.loads(str(row["term_ids_json"]))]
        base_weights = [float(value) for value in json.loads(str(row["weights_json"]))]
        if not base_ids or len(base_ids) != len(base_weights) or any(term_id not in physical_term_ids for term_id in base_ids):
            raise ValueError("NEXT149 base formula differs")
        for weight in WEIGHTS:
            payload = {
                "base_term_ids": base_ids,
                "base_weights": base_weights,
                "coordination_protection_term_id": n130.PROTECTION_TERM_ID,
                "coordination_protection_weight": 2.0,
                "charge_order_spectrum_protection_term_id": None if weight == 0.0 else PROTECTION_TERM_ID,
                "charge_order_spectrum_protection_weight": float(weight),
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
    required = {n148.FEATURE_NAME, n148.SUPPORT_COLUMN}
    if required - set(features.columns):
        raise ValueError("NEXT149 protection feature schema differs")
    protection = pd.to_numeric(features[n148.FEATURE_NAME], errors="coerce").to_numpy(float)
    active = features[n148.SUPPORT_COLUMN].eq(True).to_numpy()
    if np.any(active & (~np.isfinite(protection) | (protection < 0.0) | (protection > 1.0 + 1e-12))):
        raise ValueError("NEXT149 protection feature values differ")
    base_by_id = {str(term["term_id"]): dict(term) for term in coordination_terms}
    base_risks = {term_id: _term_risk(features, term) for term_id, term in base_by_id.items()}
    columns: dict[str, np.ndarray] = {}
    terms: list[dict[str, object]] = []
    runtime: list[dict[str, object]] = []
    for spec_raw in specs:
        spec = dict(spec_raw)
        identity = n130.n127._formula_identity(spec["base_term_ids"], spec["base_weights"])
        base_id = coordination_by_formula.get(identity)
        weight = float(spec["charge_order_spectrum_protection_weight"])
        expected_term = None if weight == 0.0 else PROTECTION_TERM_ID
        if base_id is None or spec["charge_order_spectrum_protection_term_id"] != expected_term:
            raise ValueError("NEXT149 candidate configuration differs")
        score, supported = apply_charge_order_spectrum_protection(
            base_score=base_risks[base_id][0],
            base_supported=base_risks[base_id][1],
            protection=protection,
            protection_active=active,
            weight=weight,
        )
        maximum = float(np.max(score[supported])) if supported.any() else 0.0
        divisor = max(1.0, maximum / 50.0)
        encoded = np.full(len(features), np.nan, dtype=float)
        encoded[supported] = np.sinh(score[supported] / divisor)
        key = str(spec["candidate_key"])
        virtual_id = f"next149_virtual_candidate__{hashlib.sha256(key.encode()).hexdigest()[:24]}"
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
                "group": "next149_charge_order_spectrum_protected_candidate",
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
        if float(payload["charge_order_spectrum_protection_weight"]) != 0.0:
            continue
        observed[n130.n127._formula_identity(payload["base_term_ids"], payload["base_weights"])] = record
    expected = {
        n130.n127._formula_identity(json.loads(str(row["term_ids_json"])), json.loads(str(row["weights_json"]))): row["_next130_record"]
        for _, row in prior.iterrows()
    }
    if set(observed) != set(expected):
        raise RuntimeError("NEXT149 base reproduction identities differ")
    for identity, source in expected.items():
        record = observed[identity]
        if any(not math.isclose(float(record[name]), float(source[name]), rel_tol=0.0, abs_tol=n130.BASE_REPRODUCTION_AUC_TOLERANCE) for name in metrics) or any(bool(record[name]) != bool(source[name]) for name in ("passes_source_auc_gates", "passes_safe_all_cells")) or int(record["safe_passing_cells"]) != int(source["safe_passing_cells"]):
            raise RuntimeError("NEXT149 base diagnostics do not reproduce NEXT130")


def _paths(roots: Mapping[str, Path], freeze_path: Path) -> dict[str, Path]:
    paths = n135._paths(roots, freeze_path)
    paths.update(
        {
            "next142_manifest": roots["next142"] / n142.MANIFEST_NAME,
            "next142_audit": roots["next142"] / n142.AUDIT_NAME,
            "next142_feature_table": roots["next142"] / n142.FEATURE_TABLE_NAME,
            "next147_manifest": roots["next147"] / n147.MANIFEST_NAME,
            "next147_diagnostic": roots["next147"] / n147.DIAGNOSTIC_NAME,
            "next147_per_candidate": roots["next147"] / n147.PER_CANDIDATE_NAME,
            "next148_manifest": roots["next148"] / n148.MANIFEST_NAME,
            "next148_catalogue": roots["next148"] / n148.CATALOGUE_NAME,
            "next148_scigen_features": roots["next148"] / n148.FEATURE_FILES["scigen"],
            "next148_wyformer_features": roots["next148"] / n148.FEATURE_FILES["wyformer"],
        }
    )
    return paths


def run_charge_order_spectrum_protection_search(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    next98_dir: Path, next110_dir: Path, next111_dir: Path, next113_dir: Path,
    next114_dir: Path, next116_dir: Path, next117_dir: Path, next120_dir: Path,
    next121_dir: Path, next122_dir: Path, next124_dir: Path, next125_dir: Path,
    next129_dir: Path, next130_dir: Path, next133_dir: Path, next134_dir: Path,
    next142_dir: Path, next147_dir: Path, next148_dir: Path,
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
            (142,next142_dir),(147,next147_dir),(148,next148_dir),
        )},
    }
    target = Path(output_dir).resolve()
    paths = _paths(roots, Path(freeze_path).resolve())
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT149 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(key for key in set(input_hashes) | set(EXPECTED_INPUT_SHA256) if input_hashes.get(key) != EXPECTED_INPUT_SHA256.get(key))
        raise ValueError(f"NEXT149 formal input identity differs: {differing}")
    manifest142 = json.loads(paths["next142_manifest"].read_text())
    manifest147 = json.loads(paths["next147_manifest"].read_text())
    manifest148 = json.loads(paths["next148_manifest"].read_text())
    if (
        manifest142.get("protocol") != n142.PROTOCOL
        or manifest142.get("opened_validation_outputs_used") is not False
        or manifest142.get("outputs_sha256", {}).get(n142.AUDIT_NAME) != input_hashes["next142_audit"]
        or manifest142.get("outputs_sha256", {}).get(n142.FEATURE_TABLE_NAME) != input_hashes["next142_feature_table"]
        or manifest147.get("protocol") != n147.PROTOCOL
        or manifest147.get("conditional_balance_exemption_branch_terminated") is not True
        or manifest147.get("opened_validation_outputs_used") is not False
        or manifest147.get("outputs_sha256", {}).get(n147.DIAGNOSTIC_NAME) != input_hashes["next147_diagnostic"]
        or manifest148.get("protocol") != n148.PROTOCOL
        or manifest148.get("labels_opened") is not False
        or manifest148.get("endpoint_payloads_opened") is not False
        or manifest148.get("dft_values_used_by_features") is not False
        or manifest148.get("outputs_sha256", {}).get(n148.FEATURE_FILES["scigen"]) != input_hashes["next148_scigen_features"]
        or manifest148.get("outputs_sha256", {}).get(n148.FEATURE_FILES["wyformer"]) != input_hashes["next148_wyformer_features"]
    ):
        raise ValueError("NEXT149 prior provenance differs")

    extended, feature_tables, old_terms, mhcr_terms = n130._join_label_free_features(paths)
    protection_frames = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next148_{source}_features"]).copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        protection_frames.append(table)
    protection = pd.concat(protection_frames, ignore_index=True)
    extended = extended.merge(protection, on="material_id", how="inner", validate="one_to_one")
    if len(extended) != len(protection):
        raise ValueError("NEXT149 protection row accounting differs")
    physical_terms = [*old_terms, *mhcr_terms]
    physical_ids = {str(term["term_id"]) for term in physical_terms}
    all_bases = n130.n127.select_next125_bases(pd.read_parquet(paths["next125_search_records"]))
    bases = n132.select_extended_bases(pd.read_parquet(paths["next130_search_records"]), all_bases)
    specs = build_candidate_specs(bases=bases, physical_term_ids=physical_ids)
    candidate_sha = hashlib.sha256("\n".join(str(spec["candidate_key"]) for spec in specs).encode()).hexdigest()
    weight_sha = hashlib.sha256(json.dumps(list(WEIGHTS), separators=(",", ":")).encode()).hexdigest()
    if require_formal_inputs and (
        len(bases) != EXPECTED_BASE_COUNT
        or len(specs) != EXPECTED_CANDIDATE_COUNT
        or candidate_sha != EXPECTED_CANDIDATE_KEY_SHA256
        or weight_sha != EXPECTED_WEIGHT_GRID_SHA256
    ):
        raise ValueError("NEXT149 frozen candidate universe differs")

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
        record["charge_order_spectrum_protection_term_id"] = payload["charge_order_spectrum_protection_term_id"]
        record["charge_order_spectrum_protection_weight"] = float(payload["charge_order_spectrum_protection_weight"])
        record["score_composition"] = "max(0,coordination_weight2_base-charge_order_spectrum_weight*charge_order_spectrum_protection)"
    for record in result["candidate_records"]: decorate(record)
    selected = result["selected"]
    if "evaluation_virtual_term_id" not in selected["record"]: decorate(selected["record"])
    payload = json.loads(str(selected["record"]["candidate_key"])); formula = selected["formula"]
    formula["evaluation_virtual_term_id"] = str(formula["base_terms"][0]["term_id"])
    formula["base_terms"] = [{**physical_by_id[str(term_id)],"weight":float(weight)} for term_id,weight in zip(payload["base_term_ids"],payload["base_weights"],strict=True)]
    formula["coordination_protection"] = {"term_id":n130.PROTECTION_TERM_ID,"feature":n130.n129.FEATURE_NAME,"weight":2.0,"missing_policy":"TERM_OFF_KEEP_BASE"}
    formula["charge_order_spectrum_protection"] = None if float(payload["charge_order_spectrum_protection_weight"]) == 0.0 else {"term_id":PROTECTION_TERM_ID,"feature":n148.FEATURE_NAME,"weight":float(payload["charge_order_spectrum_protection_weight"]),"missing_policy":"TERM_OFF_KEEP_BASE"}
    formula["score_composition"] = "max(0,coordination_weight2_base-charge_order_spectrum_weight*charge_order_spectrum_protection)"
    formula["kind"] = "next130_coordination_base_with_optional_charge_order_spectrum_protection"
    passes = bool(selected["record"]["passes_all_discovery_gates"])
    records_frame = pd.DataFrame(result["candidate_records"])
    counts = {f"{float(weight):g}":{"candidates":int(len(frame)),"passes_source_auc_gates":int(frame["passes_source_auc_gates"].sum()),"passes_safe_all_cells":int(frame["passes_safe_all_cells"].sum()),"passes_broad_all_cells":int(frame["passes_broad_all_cells"].sum()),"passes_all_discovery_gates":int(frame["passes_all_discovery_gates"].sum())} for weight,frame in records_frame.groupby("charge_order_spectrum_protection_weight",sort=True)}
    catalogue = {"protocol":PROTOCOL,"freeze_sha256":input_hashes["freeze"],"base_count":len(bases),"candidate_count":len(specs),"weight_grid":list(WEIGHTS),"weight_grid_sha256":weight_sha,"candidate_key_sha256":candidate_sha,"protection_term_id":PROTECTION_TERM_ID,"protection_feature":n148.FEATURE_NAME,"active_score":"max(0,coordination_weight2_base-charge_order_spectrum_weight*charge_order_spectrum_protection)","validation_outputs_opened":False,"dft_values_used_by_executable_formula":False,"analytic_formal_charge_spectrum_used":True}
    catalogue_sha = hashlib.sha256(json.dumps(catalogue,indent=2,sort_keys=True).encode()+b"\n").hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {"src/next130_coordination_protection_search.py":Path(n130.__file__).resolve(),"src/next134_compactness_protection_search.py":Path(n134.__file__).resolve(),"src/next149_charge_order_spectrum_protection_search.py":Path(__file__).resolve()}
    source_hashes = {name:_sha256_file(path) for name,path in source_paths.items()}
    try:
        catalogue_path=staging/CATALOGUE_NAME; evaluation_path=staging/EVALUATION_NAME; search_path=staging/SEARCH_NAME
        _write_json(catalogue_path,{**catalogue,"label_free_catalogue_sha256":catalogue_sha})
        _write_json(evaluation_path,{"protocol":PROTOCOL,"evaluation_mode":"fixed_charge_order_spectrum_protection","rows":{"scigen":int(len(feature_tables["scigen"])),"wyformer":int(len(feature_tables["wyformer"])),"total":int(len(combined))},"base_count":len(bases),"candidate_count":int(result["candidate_count"]),"elapsed_seconds":elapsed,"search_workers":search_workers,"base_only_reproduced_next130":True,"counts_by_charge_order_spectrum_weight":counts,"safe_gates":dict(n130.n125.n121.prior.DEFAULT_GATES),"source_auc_gates":dict(n130.n125.n121.prior.AUC_GATES),"broad_min_severe_precision_lower":n130.n125.n121.prior.BROAD_MIN_PRECISION_LOWER,"selected_record":selected["record"],"selected_formula":selected["formula"],"selected_safe":selected["safe"],"selected_safe_diagnostic":selected["safe_diagnostic"],"selected_broad":selected["broad"],"selected_source_diagnostics":selected["source_diagnostics"],"pauling_by_cell":result["pauling_by_cell"],"cells":result["cells"],"passes_all_cross_source_discovery_gates":passes,"freeze_authorized":passes,"requires_unopened_internal_validation_before_claim":True})
        records_frame.to_parquet(search_path,index=False)
        manifest={"protocol":PROTOCOL,"label_free_catalogue_sha256":catalogue_sha,"base_count":len(bases),"candidate_count":int(result["candidate_count"]),"search_workers":search_workers,"base_only_reproduced_next130":True,"passes_all_cross_source_discovery_gates":passes,"freeze_authorized":passes,"requires_unopened_internal_validation_before_claim":True,"scigen_discovery_endpoint_opened":True,"wyformer_discovery_endpoint_opened":True,"discovery_outcomes_used_as_offline_labels":True,"opened_validation_outputs_used":False,"scigen_replication_endpoint_opened":False,"wyformer_replication_endpoint_opened":False,"dft_calculation_executed":False,"dft_values_used_by_executable_formula":False,"learned_energy_force_stress_proxy_used":False,"analytic_formal_charge_spectrum_used":True,"physical_relaxation_executed":False,"formula_or_threshold_changed_after_search":False,"scientific_improvement_claim":False,"inputs_sha256":input_hashes,"executed_source_sha256":source_hashes,"outputs_sha256":{CATALOGUE_NAME:_sha256_file(catalogue_path),EVALUATION_NAME:_sha256_file(evaluation_path),SEARCH_NAME:_sha256_file(search_path)}}
        _write_json(staging/MANIFEST_NAME,manifest)
        if any(_sha256_file(path)!=input_hashes[name] for name,path in paths.items()): raise RuntimeError("NEXT149 input changed before publication")
        if any(_sha256_file(path)!=source_hashes[name] for name,path in source_paths.items()): raise RuntimeError("NEXT149 source changed before publication")
        os.replace(staging,target); return manifest
    except Exception:
        shutil.rmtree(staging,ignore_errors=True); raise


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scigen-feature-dir",type=Path,required=True);parser.add_argument("--scigen-discovery-endpoint-dir",type=Path,required=True)
    parser.add_argument("--wyformer-feature-dir",type=Path,required=True);parser.add_argument("--wyformer-discovery-endpoint-dir",type=Path,required=True)
    for stage in (98,110,111,113,114,116,117,120,121,122,124,125,129,130,133,134,142,147,148): parser.add_argument(f"--next{stage}-dir",type=Path,required=True)
    parser.add_argument("--freeze-path",type=Path,required=True);parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--search-workers",type=int,default=SEARCH_WORKERS);parser.add_argument("--allow-nonformal-inputs",action="store_true")
    args=parser.parse_args()
    manifest=run_charge_order_spectrum_protection_search(scigen_feature_dir=args.scigen_feature_dir,scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,wyformer_feature_dir=args.wyformer_feature_dir,wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,**{f"next{stage}_dir":getattr(args,f"next{stage}_dir") for stage in (98,110,111,113,114,116,117,120,121,122,124,125,129,130,133,134,142,147,148)},freeze_path=args.freeze_path,output_dir=args.output_dir,search_workers=args.search_workers,require_formal_inputs=not args.allow_nonformal_inputs)
    print(json.dumps(manifest,indent=2,sort_keys=True));return 0


if __name__ == "__main__": raise SystemExit(main())


__all__=["apply_charge_order_spectrum_protection","build_candidate_specs","materialize_candidates","run_charge_order_spectrum_protection_search"]
