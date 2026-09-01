#!/usr/bin/env python3
"""Audit structure-only features in the SAFE-to-BROAD incremental score shell."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import src.next130_coordination_protection_search as n130
import src.next132_extended_coordination_protection_search as n132
import src.next134_compactness_protection_search as n134
import src.next135_conjunctive_compactness_search as n135
import src.next136_conjunctive_broad_residual_diagnostic as n136
import src.next141_analytic_field_broad_residual_diagnostic as n141
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next87_scigen_sparse_law_search import _term_risk, assign_group_folds


PROTOCOL = "2026-08-08-next142-threshold-local-retention-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
AUDIT_NAME = "NEXT142_THRESHOLD_LOCAL_RETENTION_AUDIT.json"
FEATURE_TABLE_NAME = "next142_threshold_local_feature_audit.parquet"
EXPECTED_DESIGN_SHA256 = "f2de460809a01c729e4c7db47e14e7da2adcc5da5bdd9e29b04c4d4f4afc5245"
EXPECTED_CANDIDATE_KEY_SHA256 = "44b9eabae5e1ff3014ef4746758bbc3a79a4f193bad94507dd17c7db0edd1919"
SAFE_THRESHOLD = 3.4014264642057306
BROAD_THRESHOLD = 0.8669460357541353
MINIMUM_COVERAGE = 0.8
MINIMUM_UNIQUE = 10
EXPECTED_INPUT_SHA256 = {
    **n135.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next136_manifest": "4df132a07a6c8a22eb2dd22b45a0010aea9c649f1f010deea9eae745f97d9f6f",
    "next136_diagnostic": "625423884cb283a71102ac551c6fd221d1dfebd183b53b7f06e7aad8fc02eb88",
    "next136_per_candidate": "718675a5ef88e93c0c7fe734b589b0ddb7cb5464f6a800d3f330f120fa7ffe42",
    "next141_manifest": "5ed429591a771661007ae98166154ca747f3a7ae62f2629b8dffd7d7370a0317",
    "next141_diagnostic": "f33c45e6c47c9c3d78657b014061925929c173b8ed644032b89ef2ac3cb7e7a8",
    "next141_per_candidate": "68b9de38bfc11826cbffa7c3fa02c09f86279aa4ebd488c7ebb02059112f3452",
}


BLOCKED_SUBSTRINGS = (
    "endpoint", "label", "target", "energy", "force", "stress", "dft",
    "relax", "distortion", "pauling", "matter", "formation", "hull",
    "bandgap", "band_gap", "e_above", "virtual_candidate", "_encoded",
    "_physical_raw", "coord", "effective_cn", "packing", "volume",
    "bottleneck", "analytic_field", "aefi",
)


def blocked_feature_name(name: str) -> bool:
    lowered = str(name).lower()
    return any(token in lowered for token in BLOCKED_SUBSTRINGS)


def audit_one_source(
    *,
    values: object,
    protected: object,
    folds: object,
    direction: int | None,
    minimum_coverage: float = MINIMUM_COVERAGE,
    minimum_unique: int = MINIMUM_UNIQUE,
) -> dict[str, object] | None:
    x = np.asarray(values, dtype=float)
    y = np.asarray(protected, dtype=bool)
    group = np.asarray(folds, dtype=int)
    if x.shape != y.shape or group.shape != y.shape or not y.any() or not (~y).any():
        return None
    finite = np.isfinite(x)
    protected_coverage = float(finite[y].mean())
    severe_coverage = float(finite[~y].mean())
    if min(protected_coverage, severe_coverage) < minimum_coverage or np.unique(x[finite]).size < minimum_unique:
        return None
    if direction is None:
        raw_auc = float(roc_auc_score(y[finite], x[finite]))
        fixed_direction = 1 if raw_auc >= 0.5 else -1
    else:
        if direction not in (-1, 1):
            raise ValueError("NEXT142 feature direction differs")
        fixed_direction = int(direction)
    pooled = float(roc_auc_score(y[finite], fixed_direction * x[finite]))
    fold_aucs: list[float | None] = []
    for fold in range(5):
        mask = finite & (group == fold)
        if not mask.any() or np.unique(y[mask]).size != 2:
            fold_aucs.append(None)
        else:
            fold_aucs.append(float(roc_auc_score(y[mask], fixed_direction * x[mask])))
    evaluable = [value for value in fold_aucs if value is not None]
    return {
        "direction": fixed_direction,
        "pooled_auc": pooled,
        "macro_auc": float(np.mean(evaluable)) if evaluable else None,
        "worst_auc": float(np.min(evaluable)) if evaluable else None,
        "evaluable_folds": len(evaluable),
        "fold_aucs": fold_aucs,
        "protected_coverage": protected_coverage,
        "severe_coverage": severe_coverage,
        "finite_rows": int(finite.sum()),
        "unique_values": int(np.unique(x[finite]).size),
    }


def _paths(roots: Mapping[str, Path], original_freeze: Path, design: Path) -> dict[str, Path]:
    paths = n135._paths(roots, original_freeze)
    paths.update(
        {
            "design": design,
            "next136_manifest": roots["next136"] / n136.MANIFEST_NAME,
            "next136_diagnostic": roots["next136"] / n136.DIAGNOSTIC_NAME,
            "next136_per_candidate": roots["next136"] / n136.PER_CANDIDATE_NAME,
            "next141_manifest": roots["next141"] / n141.MANIFEST_NAME,
            "next141_diagnostic": roots["next141"] / n141.DIAGNOSTIC_NAME,
            "next141_per_candidate": roots["next141"] / n141.PER_CANDIDATE_NAME,
        }
    )
    return paths


def run_threshold_local_retention_audit(
    *,
    scigen_feature_dir: Path, scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path, wyformer_discovery_endpoint_dir: Path,
    next98_dir: Path, next110_dir: Path, next111_dir: Path, next113_dir: Path,
    next114_dir: Path, next116_dir: Path, next117_dir: Path, next120_dir: Path,
    next121_dir: Path, next122_dir: Path, next124_dir: Path, next125_dir: Path,
    next129_dir: Path, next130_dir: Path, next133_dir: Path, next134_dir: Path,
    next136_dir: Path, next141_dir: Path,
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
            (136,next136_dir),(141,next141_dir),
        )},
    }
    target = Path(output_dir).resolve()
    paths = _paths(roots, Path(next135_freeze_path).resolve(), Path(design_path).resolve())
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT142 audit input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(key for key in set(input_hashes) | set(EXPECTED_INPUT_SHA256) if input_hashes.get(key) != EXPECTED_INPUT_SHA256.get(key))
        raise ValueError(f"NEXT142 formal input identity differs: {differing}")
    manifest136 = json.loads(paths["next136_manifest"].read_text())
    manifest141 = json.loads(paths["next141_manifest"].read_text())
    if (
        manifest136.get("protocol") != n136.PROTOCOL
        or manifest136.get("opened_validation_outputs_used") is not False
        or manifest136.get("outputs_sha256", {}).get(n136.DIAGNOSTIC_NAME) != input_hashes["next136_diagnostic"]
        or manifest141.get("protocol") != n141.PROTOCOL
        or manifest141.get("analytic_field_branch_terminated") is not True
        or manifest141.get("opened_validation_outputs_used") is not False
        or manifest141.get("outputs_sha256", {}).get(n141.DIAGNOSTIC_NAME) != input_hashes["next141_diagnostic"]
    ):
        raise ValueError("NEXT142 prior provenance differs")

    closest = json.loads(paths["next136_diagnostic"].read_text())["global_closest"]
    candidate_key = str(closest["candidate_key"])
    if (
        hashlib.sha256(candidate_key.encode()).hexdigest() != EXPECTED_CANDIDATE_KEY_SHA256
        or not math.isclose(float(closest["safe_threshold"]), SAFE_THRESHOLD, rel_tol=0.0, abs_tol=1e-15)
        or not math.isclose(float(closest["best_threshold"]), BROAD_THRESHOLD, rel_tol=0.0, abs_tol=1e-15)
    ):
        raise ValueError("NEXT142 frozen shell identity differs")

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
    all_bases = n130.n127.select_next125_bases(pd.read_parquet(paths["next125_search_records"]))
    bases = n132.select_extended_bases(pd.read_parquet(paths["next130_search_records"]), all_bases)
    selected_specs = [spec for spec in n135.build_candidate_specs(bases=bases, physical_term_ids={str(term["term_id"]) for term in physical_terms}) if spec["candidate_key"] == candidate_key]
    if len(selected_specs) != 1:
        raise ValueError("NEXT142 selected candidate reconstruction differs")

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
    audit_features = combined
    combined, base_virtual_terms, base_virtual_by_formula = n130.n127.materialize_virtual_bases(features=combined, bases=bases, old_terms=old_terms, mhcr_terms=mhcr_terms)
    combined, coordination_terms, coordination_by_formula = n134.materialize_coordination_bases(features=combined, bases=bases, base_virtual_terms=base_virtual_terms, base_virtual_by_formula=base_virtual_by_formula)
    combined, virtual_terms, _ = n135.materialize_candidates(features=combined, coordination_terms=coordination_terms, coordination_by_formula=coordination_by_formula, specs=selected_specs)
    score, supported = _term_risk(combined, virtual_terms[0])
    sources = combined["source_dataset"].astype(str).to_numpy()
    folds = assign_group_folds(combined["reduced_formula"].astype(str).to_numpy())
    extremes = (endpoint <= 1.0) | (endpoint >= 2.0)
    shell = supported & (score >= BROAD_THRESHOLD) & (score < SAFE_THRESHOLD) & extremes
    shell_masks = {source: shell & (sources == source) for source in ("scigen", "wyformer")}
    shell_counts = {
        source: {
            "rows": int(mask.sum()),
            "protected": int((mask & (endpoint <= 1.0)).sum()),
            "severe": int((mask & (endpoint >= 2.0)).sum()),
        }
        for source, mask in shell_masks.items()
    }
    if shell_counts["scigen"]["protected"] == 0 or shell_counts["scigen"]["severe"] == 0:
        raise ValueError("NEXT142 SCIGEN shell class accounting differs")

    records: list[dict[str, object]] = []
    considered = 0
    for feature in audit_features.columns:
        if blocked_feature_name(feature) or not pd.api.types.is_numeric_dtype(audit_features[feature].dtype):
            continue
        considered += 1
        values = pd.to_numeric(audit_features[feature], errors="coerce").to_numpy(float)
        scigen_mask = shell_masks["scigen"]
        scigen = audit_one_source(
            values=values[scigen_mask],
            protected=endpoint[scigen_mask] <= 1.0,
            folds=folds[scigen_mask],
            direction=None,
        )
        if scigen is None:
            continue
        wyformer_mask = shell_masks["wyformer"]
        wyformer = audit_one_source(
            values=values[wyformer_mask],
            protected=endpoint[wyformer_mask] <= 1.0,
            folds=folds[wyformer_mask],
            direction=int(scigen["direction"]),
        )
        record: dict[str, object] = {
            "feature": feature,
            **{f"scigen_{key}": value for key, value in scigen.items() if key != "fold_aucs"},
            "scigen_fold_aucs_json": json.dumps(scigen["fold_aucs"], separators=(",", ":")),
        }
        if wyformer is None:
            record.update({
                "wyformer_pooled_auc": np.nan, "wyformer_macro_auc": np.nan,
                "wyformer_worst_auc": np.nan, "wyformer_evaluable_folds": 0,
                "wyformer_protected_coverage": np.nan, "wyformer_severe_coverage": np.nan,
                "wyformer_finite_rows": 0, "wyformer_unique_values": 0,
                "wyformer_fold_aucs_json": "[]", "source_direction_concordant": False,
            })
        else:
            record.update({f"wyformer_{key}": value for key, value in wyformer.items() if key not in ("fold_aucs", "direction")})
            record["wyformer_fold_aucs_json"] = json.dumps(wyformer["fold_aucs"], separators=(",", ":"))
            record["source_direction_concordant"] = float(wyformer["pooled_auc"]) >= 0.5
        records.append(record)
    table = pd.DataFrame(records)
    if table.empty:
        raise RuntimeError("NEXT142 audit retained no feature")
    table["all_scigen_folds_evaluable"] = table["scigen_evaluable_folds"].eq(5)
    table = table.sort_values(
        ["all_scigen_folds_evaluable", "scigen_worst_auc", "scigen_macro_auc", "scigen_pooled_auc", "wyformer_pooled_auc", "feature"],
        ascending=[False, False, False, False, False, True], na_position="last",
    ).reset_index(drop=True)
    top = []
    for _, row in table.head(25).iterrows():
        top.append({
            "feature": str(row["feature"]), "direction": int(row["scigen_direction"]),
            "scigen_pooled_auc": float(row["scigen_pooled_auc"]),
            "scigen_macro_auc": float(row["scigen_macro_auc"]),
            "scigen_worst_auc": float(row["scigen_worst_auc"]),
            "wyformer_pooled_auc": None if pd.isna(row["wyformer_pooled_auc"]) else float(row["wyformer_pooled_auc"]),
            "source_direction_concordant": bool(row["source_direction_concordant"]),
        })
    summary = {
        "protocol": PROTOCOL,
        "audit_mode": "safe_to_broad_incremental_rejection_shell",
        "candidate_key_sha256": EXPECTED_CANDIDATE_KEY_SHA256,
        "safe_threshold": SAFE_THRESHOLD, "broad_threshold": BROAD_THRESHOLD,
        "shell_counts": shell_counts, "numeric_features_considered": considered,
        "eligible_features": int(len(table)), "top_features": top,
        "new_formula_searched": False, "validation_or_replication_opened": False,
        "dft_calculation_executed": False, "dft_values_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False, "physical_relaxation_executed": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {"src/next135_conjunctive_compactness_search.py":Path(n135.__file__).resolve(),"src/next142_threshold_local_retention_audit.py":Path(__file__).resolve()}
    source_hashes = {name:_sha256_file(path) for name,path in source_paths.items()}
    try:
        audit_path=staging/AUDIT_NAME;table_path=staging/FEATURE_TABLE_NAME
        _write_json(audit_path,summary);table.to_parquet(table_path,index=False)
        manifest={"protocol":PROTOCOL,"new_formula_searched":False,"discovery_outcomes_used_as_offline_audit_labels":True,"opened_validation_outputs_used":False,"scigen_replication_endpoint_opened":False,"wyformer_replication_endpoint_opened":False,"dft_calculation_executed":False,"dft_values_used_by_executable_formula":False,"learned_energy_force_stress_proxy_used":False,"physical_relaxation_executed":False,"inputs_sha256":input_hashes,"executed_source_sha256":source_hashes,"outputs_sha256":{AUDIT_NAME:_sha256_file(audit_path),FEATURE_TABLE_NAME:_sha256_file(table_path)}}
        _write_json(staging/MANIFEST_NAME,manifest)
        if any(_sha256_file(path)!=input_hashes[name] for name,path in paths.items()):raise RuntimeError("NEXT142 input changed")
        if any(_sha256_file(path)!=source_hashes[name] for name,path in source_paths.items()):raise RuntimeError("NEXT142 source changed")
        os.replace(staging,target);return manifest
    except Exception:
        shutil.rmtree(staging,ignore_errors=True);raise


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scigen-feature-dir",type=Path,required=True);parser.add_argument("--scigen-discovery-endpoint-dir",type=Path,required=True)
    parser.add_argument("--wyformer-feature-dir",type=Path,required=True);parser.add_argument("--wyformer-discovery-endpoint-dir",type=Path,required=True)
    for stage in (98,110,111,113,114,116,117,120,121,122,124,125,129,130,133,134,136,141):parser.add_argument(f"--next{stage}-dir",type=Path,required=True)
    parser.add_argument("--next135-freeze-path",type=Path,required=True);parser.add_argument("--design-path",type=Path,required=True);parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--allow-nonformal-inputs",action="store_true")
    args=parser.parse_args()
    manifest=run_threshold_local_retention_audit(scigen_feature_dir=args.scigen_feature_dir,scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,wyformer_feature_dir=args.wyformer_feature_dir,wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,**{f"next{stage}_dir":getattr(args,f"next{stage}_dir") for stage in (98,110,111,113,114,116,117,120,121,122,124,125,129,130,133,134,136,141)},next135_freeze_path=args.next135_freeze_path,design_path=args.design_path,output_dir=args.output_dir,require_formal_inputs=not args.allow_nonformal_inputs)
    print(json.dumps(manifest,indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())


__all__=["audit_one_source","blocked_feature_name","run_threshold_local_retention_audit"]
