#!/usr/bin/env python3
"""Audit whether severe structures distribute risk across multiple physical terms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import src.next130_coordination_protection_search as n130
import src.next132_extended_coordination_protection_search as n132
import src.next134_compactness_protection_search as n134
import src.next135_conjunctive_compactness_search as n135
import src.next136_conjunctive_broad_residual_diagnostic as n136
import src.next150_charge_order_spectrum_broad_residual_diagnostic as n150
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next87_scigen_sparse_law_search import _term_risk, assign_group_folds


PROTOCOL = "2026-08-08-next151-violation-multiplicity-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
AUDIT_NAME = "NEXT151_VIOLATION_MULTIPLICITY_AUDIT.json"
TABLE_NAME = "next151_violation_multiplicity_audit.parquet"
EXPECTED_DESIGN_SHA256 = "58671737bd5d372cb1f749d31664f58b9fce9fbd23feebe9326dc4366c249797"
EXPECTED_CANDIDATE_KEY_SHA256 = "44b9eabae5e1ff3014ef4746758bbc3a79a4f193bad94507dd17c7db0edd1919"
SAFE_THRESHOLD = 3.4014264642057306
BROAD_THRESHOLD = 0.8669460357541353
COUNT_THRESHOLDS = (0.1, 0.25, 0.5, 1.0)
FIXED_DIRECTIONS = {
    "sum_all": -1,
    "max_one": -1,
    "second_largest": -1,
    "third_largest": -1,
    "top2_sum": -1,
    "top3_sum": -1,
    "sum_minus_max": -1,
    "sum_minus_top2": -1,
    "effective_violation_count": -1,
    "max_fraction": 1,
    "count_gt_0p1": -1,
    "count_gt_0p25": -1,
    "count_gt_0p5": -1,
    "count_gt_1": -1,
}
EXPECTED_INPUT_SHA256 = {
    **n135.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next136_manifest": "4df132a07a6c8a22eb2dd22b45a0010aea9c649f1f010deea9eae745f97d9f6f",
    "next136_diagnostic": "625423884cb283a71102ac551c6fd221d1dfebd183b53b7f06e7aad8fc02eb88",
    "next136_per_candidate": "718675a5ef88e93c0c7fe734b589b0ddb7cb5464f6a800d3f330f120fa7ffe42",
    "next150_manifest": "d85318bc6549cf40f4fb09c347221d3dcadf9ad045400e3e65b47cf444f4df3d",
    "next150_diagnostic": "ec75d23842778c2200efbc69b3e2ce04605c30ad48f347ee81965f047ef91135",
    "next150_per_candidate": "187b0d03226e8c629ffa54cae6c7b3dbc6fa62cf8ff1ac7a5d02d0c0544c4b9e",
}


def multiplicity_statistics(contributions: object) -> dict[str, np.ndarray]:
    values = np.asarray(contributions, dtype=float)
    if values.ndim != 2 or values.shape[1] < 3 or np.any(~np.isfinite(values)) or np.any(values < -1e-12):
        raise ValueError("NEXT151 contribution matrix differs")
    values = np.maximum(values, 0.0)
    ordered = np.sort(values, axis=1)[:, ::-1]
    total = values.sum(axis=1)
    squared = np.square(values).sum(axis=1)
    effective = np.divide(np.square(total), squared, out=np.zeros_like(total), where=squared > 0.0)
    max_fraction = np.divide(ordered[:, 0], total, out=np.zeros_like(total), where=total > 0.0)
    result = {
        "sum_all": total,
        "max_one": ordered[:, 0],
        "second_largest": ordered[:, 1],
        "third_largest": ordered[:, 2],
        "top2_sum": ordered[:, :2].sum(axis=1),
        "top3_sum": ordered[:, :3].sum(axis=1),
        "sum_minus_max": total - ordered[:, 0],
        "sum_minus_top2": total - ordered[:, :2].sum(axis=1),
        "effective_violation_count": effective,
        "max_fraction": max_fraction,
    }
    for threshold in COUNT_THRESHOLDS:
        suffix = f"{threshold:g}".replace(".", "p")
        result[f"count_gt_{suffix}"] = (values > threshold).sum(axis=1).astype(float)
    if set(result) != set(FIXED_DIRECTIONS):
        raise RuntimeError("NEXT151 statistic schema differs")
    return result


def _evaluate_auc(
    *, values: np.ndarray, protected: np.ndarray, folds: np.ndarray, direction: int
) -> dict[str, object] | None:
    x = np.asarray(values, dtype=float)
    y = np.asarray(protected, dtype=bool)
    group = np.asarray(folds, dtype=int)
    finite = np.isfinite(x)
    if x.shape != y.shape or group.shape != y.shape or not (finite & y).any() or not (finite & ~y).any():
        return None
    pooled = float(roc_auc_score(y[finite], direction * x[finite]))
    fold_aucs: list[float | None] = []
    for fold in range(5):
        mask = finite & (group == fold)
        fold_aucs.append(None if np.unique(y[mask]).size != 2 else float(roc_auc_score(y[mask], direction * x[mask])))
    evaluable = [value for value in fold_aucs if value is not None]
    return {
        "pooled_auc": pooled,
        "macro_auc": float(np.mean(evaluable)) if evaluable else None,
        "worst_auc": float(np.min(evaluable)) if evaluable else None,
        "evaluable_folds": len(evaluable),
        "fold_aucs": fold_aucs,
        "protected": int((finite & y).sum()),
        "severe": int((finite & ~y).sum()),
    }


def _paths(roots: Mapping[str, Path], freeze_path: Path, design_path: Path) -> dict[str, Path]:
    paths = n135._paths(roots, freeze_path)
    paths.update(
        {
            "design": design_path,
            "next136_manifest": roots["next136"] / n136.MANIFEST_NAME,
            "next136_diagnostic": roots["next136"] / n136.DIAGNOSTIC_NAME,
            "next136_per_candidate": roots["next136"] / n136.PER_CANDIDATE_NAME,
            "next150_manifest": roots["next150"] / n150.MANIFEST_NAME,
            "next150_diagnostic": roots["next150"] / n150.DIAGNOSTIC_NAME,
            "next150_per_candidate": roots["next150"] / n150.PER_CANDIDATE_NAME,
        }
    )
    return paths


def run_violation_multiplicity_audit(
    *,
    scigen_feature_dir: Path, scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path, wyformer_discovery_endpoint_dir: Path,
    next98_dir: Path, next110_dir: Path, next111_dir: Path, next113_dir: Path,
    next114_dir: Path, next116_dir: Path, next117_dir: Path, next120_dir: Path,
    next121_dir: Path, next122_dir: Path, next124_dir: Path, next125_dir: Path,
    next129_dir: Path, next130_dir: Path, next133_dir: Path, next134_dir: Path,
    next136_dir: Path, next150_dir: Path,
    next135_freeze_path: Path, design_path: Path, output_dir: Path,
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
            (136,next136_dir),(150,next150_dir),
        )},
    }
    target = Path(output_dir).resolve()
    paths = _paths(roots, Path(next135_freeze_path).resolve(), Path(design_path).resolve())
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT151 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(key for key in set(input_hashes) | set(EXPECTED_INPUT_SHA256) if input_hashes.get(key) != EXPECTED_INPUT_SHA256.get(key))
        raise ValueError(f"NEXT151 formal input identity differs: {differing}")
    manifest136 = json.loads(paths["next136_manifest"].read_text())
    manifest150 = json.loads(paths["next150_manifest"].read_text())
    if (
        manifest136.get("protocol") != n136.PROTOCOL
        or manifest136.get("opened_validation_outputs_used") is not False
        or manifest136.get("outputs_sha256", {}).get(n136.DIAGNOSTIC_NAME) != input_hashes["next136_diagnostic"]
        or manifest150.get("protocol") != n150.PROTOCOL
        or manifest150.get("charge_order_spectrum_branch_terminated") is not True
        or manifest150.get("opened_validation_outputs_used") is not False
        or manifest150.get("outputs_sha256", {}).get(n150.DIAGNOSTIC_NAME) != input_hashes["next150_diagnostic"]
    ):
        raise ValueError("NEXT151 prior provenance differs")

    closest = json.loads(paths["next136_diagnostic"].read_text())["global_closest"]
    candidate_key = str(closest["candidate_key"])
    if (
        hashlib.sha256(candidate_key.encode()).hexdigest() != EXPECTED_CANDIDATE_KEY_SHA256
        or not np.isclose(float(closest["safe_threshold"]), SAFE_THRESHOLD, rtol=0.0, atol=1e-15)
        or not np.isclose(float(closest["best_threshold"]), BROAD_THRESHOLD, rtol=0.0, atol=1e-15)
    ):
        raise ValueError("NEXT151 frozen candidate differs")

    extended, _, old_terms, mhcr_terms = n130._join_label_free_features(paths)
    compact_frames = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next133_{source}_features"]).copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        compact_frames.append(table)
    extended = extended.merge(pd.concat(compact_frames, ignore_index=True), on="material_id", how="inner", validate="one_to_one")
    conjunctive = n135.materialize_conjunctive_features(extended)
    extended = pd.concat([extended.reset_index(drop=True), conjunctive.reset_index(drop=True)], axis=1)
    physical_terms = [*old_terms, *mhcr_terms]
    physical_by_id = {str(term["term_id"]): dict(term) for term in physical_terms}
    all_bases = n130.n127.select_next125_bases(pd.read_parquet(paths["next125_search_records"]))
    bases = n132.select_extended_bases(pd.read_parquet(paths["next130_search_records"]), all_bases)
    selected_specs = [
        spec for spec in n135.build_candidate_specs(bases=bases, physical_term_ids=set(physical_by_id))
        if spec["candidate_key"] == candidate_key
    ]
    if len(selected_specs) != 1:
        raise ValueError("NEXT151 candidate reconstruction differs")
    spec = selected_specs[0]

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
    combined, virtual_terms, _ = n135.materialize_candidates(features=combined, coordination_terms=coordination_terms, coordination_by_formula=coordination_by_formula, specs=selected_specs)
    published_score, published_support = _term_risk(combined, virtual_terms[0])

    contribution_columns = []
    contribution_support = np.ones(len(combined), dtype=bool)
    contribution_terms = []
    for term_id, weight in zip(spec["base_term_ids"], spec["base_weights"], strict=True):
        risk, supported = _term_risk(combined, physical_by_id[str(term_id)])
        contribution_columns.append(float(weight) * risk)
        contribution_support &= supported
        contribution_terms.append({"term_id":str(term_id),"weight":float(weight)})
    if not np.array_equal(contribution_support, published_support):
        raise ValueError("NEXT151 contribution support differs from published score")
    contributions = np.column_stack(contribution_columns)
    stats = multiplicity_statistics(contributions)
    sources = combined["source_dataset"].astype(str).to_numpy()
    folds = assign_group_folds(combined["reduced_formula"].astype(str).to_numpy())
    extremes = published_support & ((endpoint <= 1.0) | (endpoint >= 2.0))
    shell = extremes & (published_score >= BROAD_THRESHOLD) & (published_score < SAFE_THRESHOLD)
    populations = {
        "scigen_shell": shell & (sources == "scigen"),
        "wyformer_shell": shell & (sources == "wyformer"),
        "scigen_full": extremes & (sources == "scigen"),
        "wyformer_full": extremes & (sources == "wyformer"),
    }
    population_counts = {
        name:{"rows":int(mask.sum()),"protected":int((mask&(endpoint<=1.0)).sum()),"severe":int((mask&(endpoint>=2.0)).sum())}
        for name,mask in populations.items()
    }
    if min(population_counts[name][label] for name in populations for label in ("protected","severe")) < 1:
        raise ValueError("NEXT151 audit population differs")

    records = []
    for name in sorted(stats):
        direction = FIXED_DIRECTIONS[name]
        evaluations = {}
        for population, mask in populations.items():
            result = _evaluate_auc(values=stats[name][mask], protected=endpoint[mask] <= 1.0, folds=folds[mask], direction=direction)
            if result is None:
                raise ValueError("NEXT151 AUC population differs")
            evaluations[population] = result
        key_aucs = [
            float(evaluations["scigen_shell"]["worst_auc"]),
            float(evaluations["wyformer_shell"]["pooled_auc"]),
            float(evaluations["scigen_full"]["pooled_auc"]),
            float(evaluations["wyformer_full"]["pooled_auc"]),
        ]
        eligible = bool(
            key_aucs[0] >= 0.55 and key_aucs[1] >= 0.55 and key_aucs[2] >= 0.50 and key_aucs[3] >= 0.50
            and int(evaluations["scigen_shell"]["evaluable_folds"]) == 5
        )
        record: dict[str, object] = {
            "aggregation":name,"direction":direction,"eligible_for_search":eligible,
            "ranking_min_auc":float(min(key_aucs)),"ranking_mean_auc":float(np.mean(key_aucs)),
        }
        for population, result in evaluations.items():
            for metric in ("pooled_auc","macro_auc","worst_auc","evaluable_folds","protected","severe"):
                record[f"{population}_{metric}"] = result[metric]
            record[f"{population}_fold_aucs_json"] = json.dumps(result["fold_aucs"], separators=(",", ":"))
        records.append(record)
    table = pd.DataFrame(records).sort_values(["eligible_for_search","ranking_min_auc","ranking_mean_auc","aggregation"],ascending=[False,False,False,True]).reset_index(drop=True)
    eligible_names = table.loc[table["eligible_for_search"],"aggregation"].astype(str).tolist()
    selected = None if not eligible_names else table.iloc[0].to_dict()
    audit = {
        "protocol":PROTOCOL,
        "candidate_key_sha256":EXPECTED_CANDIDATE_KEY_SHA256,
        "safe_threshold":SAFE_THRESHOLD,"broad_threshold":BROAD_THRESHOLD,
        "base_term_count":len(contribution_terms),"base_terms":contribution_terms,
        "population_counts":population_counts,"fixed_directions":FIXED_DIRECTIONS,
        "eligibility_gates":{"scigen_shell_worst_auc":0.55,"wyformer_shell_pooled_auc":0.55,"full_source_pooled_auc":0.50,"required_scigen_shell_folds":5},
        "eligible_aggregations":eligible_names,"selected_aggregation":selected,
        "new_formula_searched":False,"validation_or_replication_opened":False,
        "dft_calculation_executed":False,"dft_values_used_by_executable_formula":False,
        "learned_energy_force_stress_proxy_used":False,"physical_relaxation_executed":False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {"src/next135_conjunctive_compactness_search.py":Path(n135.__file__).resolve(),"src/next151_violation_multiplicity_audit.py":Path(__file__).resolve()}
    source_hashes = {name:_sha256_file(path) for name,path in source_paths.items()}
    try:
        audit_path=staging/AUDIT_NAME;table_path=staging/TABLE_NAME
        _write_json(audit_path,audit);table.to_parquet(table_path,index=False)
        manifest={"protocol":PROTOCOL,"eligible_aggregation_count":len(eligible_names),"new_formula_searched":False,"discovery_outcomes_used_as_offline_labels":True,"opened_validation_outputs_used":False,"scigen_replication_endpoint_opened":False,"wyformer_replication_endpoint_opened":False,"dft_calculation_executed":False,"dft_values_used_by_executable_formula":False,"learned_energy_force_stress_proxy_used":False,"physical_relaxation_executed":False,"inputs_sha256":input_hashes,"executed_source_sha256":source_hashes,"outputs_sha256":{AUDIT_NAME:_sha256_file(audit_path),TABLE_NAME:_sha256_file(table_path)}}
        _write_json(staging/MANIFEST_NAME,manifest)
        if any(_sha256_file(path)!=input_hashes[name] for name,path in paths.items()):raise RuntimeError("NEXT151 input changed")
        if any(_sha256_file(path)!=source_hashes[name] for name,path in source_paths.items()):raise RuntimeError("NEXT151 source changed")
        os.replace(staging,target);return manifest
    except Exception:
        shutil.rmtree(staging,ignore_errors=True);raise


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scigen-feature-dir",type=Path,required=True);p.add_argument("--scigen-discovery-endpoint-dir",type=Path,required=True)
    p.add_argument("--wyformer-feature-dir",type=Path,required=True);p.add_argument("--wyformer-discovery-endpoint-dir",type=Path,required=True)
    for stage in (98,110,111,113,114,116,117,120,121,122,124,125,129,130,133,134,136,150):p.add_argument(f"--next{stage}-dir",type=Path,required=True)
    p.add_argument("--next135-freeze-path",type=Path,required=True);p.add_argument("--design-path",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--allow-nonformal-inputs",action="store_true")
    a=p.parse_args();m=run_violation_multiplicity_audit(scigen_feature_dir=a.scigen_feature_dir,scigen_discovery_endpoint_dir=a.scigen_discovery_endpoint_dir,wyformer_feature_dir=a.wyformer_feature_dir,wyformer_discovery_endpoint_dir=a.wyformer_discovery_endpoint_dir,**{f"next{stage}_dir":getattr(a,f"next{stage}_dir") for stage in (98,110,111,113,114,116,117,120,121,122,124,125,129,130,133,134,136,150)},next135_freeze_path=a.next135_freeze_path,design_path=a.design_path,output_dir=a.output_dir,require_formal_inputs=not a.allow_nonformal_inputs);print(json.dumps(m,indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())


__all__=["FIXED_DIRECTIONS","multiplicity_statistics","run_violation_multiplicity_audit"]
