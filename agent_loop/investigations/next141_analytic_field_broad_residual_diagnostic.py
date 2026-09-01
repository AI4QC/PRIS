#!/usr/bin/env python3
"""Diagnose exact BROAD residuals of published NEXT140 SAFE12 candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd

import src.next130_coordination_protection_search as n130
import src.next132_extended_coordination_protection_search as n132
import src.next134_compactness_protection_search as n134
import src.next140_analytic_field_protection_search as n140
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next128_broad_residual_diagnostic import diagnose_broad_threshold_tables
from src.next87_scigen_sparse_law_search import _term_risk, assign_group_folds
from src.next98_cross_source_discovery_search import _pauling_baseline, _threshold_tables, build_source_fold_cells


PROTOCOL = "2026-08-08-next141-analytic-field-broad-residual-diagnostic-v1"
MANIFEST_NAME = "MANIFEST.json"
DIAGNOSTIC_NAME = "NEXT141_ANALYTIC_FIELD_BROAD_RESIDUAL_DIAGNOSTIC.json"
PER_CANDIDATE_NAME = "next141_analytic_field_broad_residual_by_candidate.parquet"
EXPECTED_DESIGN_SHA256 = "bac35d335c72c43eb83dc35682ff151268406d2587d40c1eca0f86d6a76fe905"
EXPECTED_SAFE_CANDIDATE_COUNT = 33
EXPECTED_INPUT_SHA256 = {
    **n140.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next140_manifest": "b61bf73cba33aea48c0073e322a3d77de334ce0f967964e7bb4f9f03ad280fae",
    "next140_catalogue": "28e9c0ced9c6b11c1f3f40fad31ae1ebf32d94d8da845ad8c550ca1f32137c90",
    "next140_evaluation": "33886b33bf7051a8e30fe400f967edb2cd687eff7849df2763ff338d82b35475",
    "next140_search_records": "aeed41e7108492bc321d75ad067a5195402246dfac4b1a768bfaabc445bd00fe",
}


def select_safe_candidates(records: pd.DataFrame) -> pd.DataFrame:
    required = {"candidate_key", "safe_threshold", "passes_safe_all_cells", "analytic_field_protection_weight"}
    if required - set(records.columns) or records["candidate_key"].astype(str).duplicated().any():
        raise ValueError("NEXT141 published candidate schema differs")
    selected = records.loc[records["passes_safe_all_cells"].fillna(False).astype(bool)].copy()
    selected["safe_threshold"] = pd.to_numeric(selected["safe_threshold"], errors="coerce")
    selected["analytic_field_protection_weight"] = pd.to_numeric(selected["analytic_field_protection_weight"], errors="coerce")
    if selected.empty or not np.isfinite(selected[["safe_threshold", "analytic_field_protection_weight"]].to_numpy(float)).all():
        raise ValueError("NEXT141 published threshold or weight differs")
    return selected.sort_values("candidate_key").reset_index(drop=True)


def _paths(roots: Mapping[str, Path], freeze_path: Path, design_path: Path) -> dict[str, Path]:
    paths = n140._paths(roots, freeze_path)
    paths.update({
        "design": design_path,
        "next140_manifest": roots["next140"] / n140.MANIFEST_NAME,
        "next140_catalogue": roots["next140"] / n140.CATALOGUE_NAME,
        "next140_evaluation": roots["next140"] / n140.EVALUATION_NAME,
        "next140_search_records": roots["next140"] / n140.SEARCH_NAME,
    })
    return paths


def run_analytic_field_broad_residual_diagnostic(
    *,
    scigen_feature_dir: Path, scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path, wyformer_discovery_endpoint_dir: Path,
    next98_dir: Path, next110_dir: Path, next111_dir: Path, next113_dir: Path,
    next114_dir: Path, next116_dir: Path, next117_dir: Path, next120_dir: Path,
    next121_dir: Path, next122_dir: Path, next124_dir: Path, next125_dir: Path,
    next129_dir: Path, next130_dir: Path, next133_dir: Path, next134_dir: Path,
    next138_dir: Path, next139_dir: Path, next140_dir: Path,
    next140_freeze_path: Path, design_path: Path, output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(), "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(), "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{f"next{stage}": Path(value).resolve() for stage, value in (
            (98,next98_dir),(110,next110_dir),(111,next111_dir),(113,next113_dir),(114,next114_dir),(116,next116_dir),(117,next117_dir),
            (120,next120_dir),(121,next121_dir),(122,next122_dir),(124,next124_dir),(125,next125_dir),(129,next129_dir),(130,next130_dir),
            (133,next133_dir),(134,next134_dir),(138,next138_dir),(139,next139_dir),(140,next140_dir),
        )},
    }
    target = Path(output_dir).resolve()
    paths = _paths(roots, Path(next140_freeze_path).resolve(), Path(design_path).resolve())
    if os.path.lexists(target): raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()): raise FileNotFoundError("NEXT141 diagnostic input is missing")
    input_hashes = {name:_sha256_file(path) for name,path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing=sorted(k for k in set(input_hashes)|set(EXPECTED_INPUT_SHA256) if input_hashes.get(k)!=EXPECTED_INPUT_SHA256.get(k));raise ValueError(f"NEXT141 formal input identity differs: {differing}")
    manifest140=json.loads(paths["next140_manifest"].read_text());outputs140=manifest140.get("outputs_sha256",{})
    if manifest140.get("protocol")!=n140.PROTOCOL or manifest140.get("passes_all_cross_source_discovery_gates") is not False or manifest140.get("opened_validation_outputs_used") is not False or outputs140.get(n140.SEARCH_NAME)!=input_hashes["next140_search_records"]:
        raise ValueError("NEXT141 prior provenance differs")

    extended,_,old,mh=n130._join_label_free_features(paths)
    frames=[]
    for source in ("scigen","wyformer"):
        table=pd.read_parquet(paths[f"next139_{source}_features"]).copy();table["material_id"]=source+":"+table["material_id"].astype(str);frames.append(table)
    extended=extended.merge(pd.concat(frames,ignore_index=True),on="material_id",validate="one_to_one")
    all_bases=n130.n127.select_next125_bases(pd.read_parquet(paths["next125_search_records"]));bases=n132.select_extended_bases(pd.read_parquet(paths["next130_search_records"]),all_bases)
    physical=[*old,*mh];specs=n140.build_candidate_specs(bases=bases,physical_term_ids={str(t["term_id"]) for t in physical})
    se=pd.read_parquet(paths["scigen_endpoint"]);wy=pd.read_parquet(paths["wyformer_endpoint"])
    endpoints=pd.concat([pd.DataFrame({"material_id":"scigen:"+se["material_id"].astype(str),"_endpoint":pd.to_numeric(se["distortion_ratio"],errors="coerce")}),pd.DataFrame({"material_id":"wyformer:"+wy["material_id"].astype(str),"_endpoint":n130.n125.n121.prior._endpoint_numeric(wy["endpoint_stratum"])})],ignore_index=True)
    combined=extended.merge(endpoints,on="material_id",validate="one_to_one");endpoint=pd.to_numeric(combined.pop("_endpoint"),errors="coerce").to_numpy(float)
    combined,bvt,bmap=n130.n127.materialize_virtual_bases(features=combined,bases=bases,old_terms=old,mhcr_terms=mh)
    combined,ct,cmap=n134.materialize_coordination_bases(features=combined,bases=bases,base_virtual_terms=bvt,base_virtual_by_formula=bmap)
    combined,vt,runtime=n140.materialize_candidates(features=combined,coordination_terms=ct,coordination_by_formula=cmap,specs=specs)
    by_key={str(s["candidate_key"]):str(s["base_term_ids"][0]) for s in runtime};by_id={str(t["term_id"]):t for t in vt}
    published=select_safe_candidates(pd.read_parquet(paths["next140_search_records"]))
    if require_formal_inputs and len(published)!=EXPECTED_SAFE_CANDIDATE_COUNT: raise ValueError("NEXT141 SAFE12 count differs")
    folds=assign_group_folds(combined["reduced_formula"].astype(str).to_numpy());sources=combined["source_dataset"].astype(str).to_numpy();cells=build_source_fold_cells(source=sources,folds=folds)
    pauling={str(c["cell_id"]):_pauling_baseline(combined.loc[np.asarray(c["mask"],bool)],endpoint[np.asarray(c["mask"],bool)]) for c in cells}
    records=[];frequency=Counter()
    for _,row in published.iterrows():
        key=str(row["candidate_key"]);score,support=_term_risk(combined,by_id[by_key[key]]);tables=_threshold_tables(score=score,supported=support,endpoint=endpoint,cells=cells)
        diagnostic=diagnose_broad_threshold_tables(tables=tables,cells=cells,pauling_by_cell=pauling,safe_threshold=float(row["safe_threshold"]))
        if diagnostic["passes_broad"] or bool(row["passes_broad_all_cells"]): raise RuntimeError("NEXT141 contradicts NEXT140")
        for failure in diagnostic["failures"]: frequency[f"{failure['cell_id']}::{failure['component']}"]+=1
        records.append({"candidate_key":key,"analytic_field_protection_weight":float(row["analytic_field_protection_weight"]),"safe_threshold":float(row["safe_threshold"]),"best_threshold":diagnostic["best_threshold"],"failed_constraint_count":diagnostic["failed_constraint_count"],"normalized_shortfall_sum":diagnostic["normalized_shortfall_sum"],"failures_json":json.dumps(diagnostic["failures"],sort_keys=True,separators=(",",":"))})
    table=pd.DataFrame(records);table=table.sort_values(["failed_constraint_count","normalized_shortfall_sum","candidate_key"]).reset_index(drop=True);closest=table.iloc[0]
    by_weight={}
    for weight,frame in table.groupby("analytic_field_protection_weight",sort=True):
        best=frame.sort_values(["failed_constraint_count","normalized_shortfall_sum","candidate_key"]).iloc[0];by_weight[f"{float(weight):g}"]={"candidate_count":int(len(frame)),"minimum_failed_constraint_count":int(best["failed_constraint_count"]),"minimum_normalized_shortfall_sum":float(best["normalized_shortfall_sum"])}
    positive=[value for key,value in by_weight.items() if float(key)>0];terminated=all(v["minimum_failed_constraint_count"]>=6 and v["minimum_normalized_shortfall_sum"]>=0.792624940220628 for v in positive)
    summary={"protocol":PROTOCOL,"safe_candidate_count":int(len(table)),"failed_constraint_count_distribution":{str(int(k)):int(v) for k,v in table["failed_constraint_count"].value_counts().sort_index().items()},"by_analytic_field_weight":by_weight,"global_closest":{"candidate_key":str(closest["candidate_key"]),"analytic_field_protection_weight":float(closest["analytic_field_protection_weight"]),"failed_constraint_count":int(closest["failed_constraint_count"]),"normalized_shortfall_sum":float(closest["normalized_shortfall_sum"]),"failures":json.loads(str(closest["failures_json"]))},"failure_frequency_at_per_candidate_optima":dict(frequency.most_common()),"analytic_field_branch_terminated":terminated,"new_formula_searched":False,"validation_or_replication_opened":False,"dft_calculation_executed":False,"dft_values_used_by_executable_formula":False,"learned_energy_force_stress_proxy_used":False,"analytic_ewald_derivative_used":True,"physical_relaxation_executed":False}
    target.parent.mkdir(parents=True,exist_ok=True);staging=Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-",dir=target.parent));source_paths={"src/next128_broad_residual_diagnostic.py":Path(diagnose_broad_threshold_tables.__code__.co_filename).resolve(),"src/next140_analytic_field_protection_search.py":Path(n140.__file__).resolve(),"src/next141_analytic_field_broad_residual_diagnostic.py":Path(__file__).resolve()};source_hashes={n:_sha256_file(p) for n,p in source_paths.items()}
    try:
        dp=staging/DIAGNOSTIC_NAME;tp=staging/PER_CANDIDATE_NAME;_write_json(dp,summary);table.to_parquet(tp,index=False)
        manifest={"protocol":PROTOCOL,"safe_candidate_count":int(len(table)),"analytic_field_branch_terminated":terminated,"new_formula_searched":False,"discovery_outcomes_used_as_offline_labels":True,"opened_validation_outputs_used":False,"scigen_replication_endpoint_opened":False,"wyformer_replication_endpoint_opened":False,"dft_calculation_executed":False,"dft_values_used_by_executable_formula":False,"learned_energy_force_stress_proxy_used":False,"analytic_ewald_derivative_used":True,"physical_relaxation_executed":False,"inputs_sha256":input_hashes,"executed_source_sha256":source_hashes,"outputs_sha256":{DIAGNOSTIC_NAME:_sha256_file(dp),PER_CANDIDATE_NAME:_sha256_file(tp)}};_write_json(staging/MANIFEST_NAME,manifest)
        if any(_sha256_file(p)!=input_hashes[n] for n,p in paths.items()): raise RuntimeError("NEXT141 input changed")
        if any(_sha256_file(p)!=source_hashes[n] for n,p in source_paths.items()): raise RuntimeError("NEXT141 source changed")
        os.replace(staging,target);return manifest
    except Exception: shutil.rmtree(staging,ignore_errors=True);raise


def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--scigen-feature-dir",type=Path,required=True);p.add_argument("--scigen-discovery-endpoint-dir",type=Path,required=True);p.add_argument("--wyformer-feature-dir",type=Path,required=True);p.add_argument("--wyformer-discovery-endpoint-dir",type=Path,required=True)
    for s in (98,110,111,113,114,116,117,120,121,122,124,125,129,130,133,134,138,139,140):p.add_argument(f"--next{s}-dir",type=Path,required=True)
    p.add_argument("--next140-freeze-path",type=Path,required=True);p.add_argument("--design-path",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--allow-nonformal-inputs",action="store_true");a=p.parse_args()
    m=run_analytic_field_broad_residual_diagnostic(scigen_feature_dir=a.scigen_feature_dir,scigen_discovery_endpoint_dir=a.scigen_discovery_endpoint_dir,wyformer_feature_dir=a.wyformer_feature_dir,wyformer_discovery_endpoint_dir=a.wyformer_discovery_endpoint_dir,**{f"next{s}_dir":getattr(a,f"next{s}_dir") for s in (98,110,111,113,114,116,117,120,121,122,124,125,129,130,133,134,138,139,140)},next140_freeze_path=a.next140_freeze_path,design_path=a.design_path,output_dir=a.output_dir,require_formal_inputs=not a.allow_nonformal_inputs);print(json.dumps(m,indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())


__all__=["run_analytic_field_broad_residual_diagnostic","select_safe_candidates"]
