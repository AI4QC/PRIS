#!/usr/bin/env python3
"""Finite no-DFT search attenuating only local-geometry family risk."""

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

import src.next130_coordination_protection_search as n130
import src.next135_conjunctive_compactness_search as n135
import src.next163_interior_family_attenuation_search as n163
import src.next164_interior_attenuation_broad_residual as n164
import src.next179_strong_neighborhood_directional_closure as n179
import src.next180_strong_neighborhood_directional_closure_audit as n180
import src.next181_strong_closure_repair_width_search as n181
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next87_scigen_sparse_law_search import _term_risk


PROTOCOL = "2026-08-08-next182-local-family-closure-attenuation-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT182_LOCAL_FAMILY_CLOSURE_ATTENUATION_CATALOGUE.json"
EVALUATION_NAME = "NEXT182_DISCOVERY_EVALUATION.json"
FORMULA_NAME = "NEXT182_FROZEN_CANDIDATE.json"
SEARCH_NAME = "next182_local_family_closure_attenuation_search.parquet"
EXPECTED_DESIGN_SHA256 = "c4076e316da6bc650e6dd5f4ca694cc865dca636dfdaf6be1986978430b7eb1a"
EXPECTED_BASE_CANDIDATE_KEY_SHA256 = n181.EXPECTED_BASE_CANDIDATE_KEY_SHA256
BROAD_THRESHOLD = n181.BROAD_THRESHOLD
SAFE_THRESHOLD = n181.SAFE_THRESHOLD
ELIGIBLE_FEATURES = n181.ELIGIBLE_FEATURES
ATTENUATIONS = (0.25, 0.50, 0.75, 1.00)
EXPECTED_CANDIDATE_COUNT = 1 + len(ELIGIBLE_FEATURES) * len(ATTENUATIONS)
LOCAL_FAMILY = "local_geometry"
LOCAL_PREFIXES = n163.FAMILY_PREFIXES[LOCAL_FAMILY]
SCORE_COMPOSITION = (
    "base_score_if_outside_frozen_repair_interval_else_"
    "max(0,base_score-alpha*local_geometry_family_mean*strong_closure)"
)
SEARCH_WORKERS = 4
EXPECTED_INPUT_SHA256 = {
    **n181.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next181_manifest": "dd64907f0caa49e6104dc13d8d3d309fbe70485aa6d976220961a0b6f0ea342e",
    "next181_catalogue": "17adb2c30f7cfbe6ab98038a3d36fddcfb35bdd9ec274429838b2ae84b6a6189",
    "next181_evaluation": "eaa6db6a27b8ec1a92f70bc3341f7e44b7f5a0a31148ecf7dc7f42fa3963b423",
    "next181_formula": "e3fbb95b33eff7ffad2947820d051133615edfe28a422e7a1e7ad8497a8744d6",
    "next181_search": "a03e898f6f13ee3c1aca28154a9ef9662a73bc38f6a26281b52bdef3519c3a7d",
}


def local_family_closure_score(
    *,
    base_score: object,
    base_support: object,
    local_family: object,
    feature: object,
    attenuation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Attenuate only the bounded local-geometry family contribution."""

    score = np.asarray(base_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    local = np.asarray(local_family, dtype=float)
    closure = np.asarray(feature, dtype=float)
    alpha = float(attenuation)
    if (
        score.ndim != 1
        or support.shape != score.shape
        or local.shape != score.shape
        or closure.shape != score.shape
        or alpha not in {0.0, *ATTENUATIONS}
        or np.any(~np.isfinite(score[support]))
        or np.any(score[support] < -1.0e-12)
        or np.any(~np.isfinite(local[support]))
        or np.any((local[support] < -1.0e-12) | (local[support] > n163.CONTRIBUTION_CAP + 1.0e-12))
    ):
        raise ValueError("NEXT182 local family or protection input differs")
    finite = np.isfinite(closure)
    if np.any((closure[finite] < -1.0e-12) | (closure[finite] > 1.0 + 1.0e-12)):
        raise ValueError("NEXT182 strong closure is outside [0,1]")
    active = support & finite & (score >= BROAD_THRESHOLD) & (score < SAFE_THRESHOLD) & (alpha > 0.0)
    corrected = score.copy()
    corrected[active] = np.maximum(0.0, score[active] - alpha * local[active] * np.clip(closure[active], 0.0, 1.0))
    return corrected, support.copy(), active


def build_candidate_specs(*, base_candidate_key: str) -> list[dict[str, object]]:
    if not isinstance(base_candidate_key, str) or not base_candidate_key:
        raise ValueError("NEXT182 base candidate key must be nonempty")
    pairs: list[tuple[str | None, float]] = [(None, 0.0)]
    pairs.extend((feature, alpha) for feature in ELIGIBLE_FEATURES for alpha in ATTENUATIONS)
    specs = []
    for feature, alpha in pairs:
        payload = {
            "attenuation": alpha,
            "base_candidate_key": base_candidate_key,
            "broad_threshold": BROAD_THRESHOLD,
            "local_family": LOCAL_FAMILY,
            "local_prefixes": list(LOCAL_PREFIXES),
            "missing_policy": "TERM_OFF_KEEP_BASE",
            "safe_threshold": SAFE_THRESHOLD,
            "score_composition": SCORE_COMPOSITION,
            "strong_closure_feature": feature,
        }
        specs.append({**payload, "candidate_key": json.dumps(payload, sort_keys=True, separators=(",", ":"))})
    if len(specs) != EXPECTED_CANDIDATE_COUNT or len({str(x["candidate_key"]) for x in specs}) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("NEXT182 candidate universe differs")
    return specs


def materialize_local_family_candidates(
    *,
    features: pd.DataFrame,
    base_score: object,
    base_support: object,
    local_family: object,
    specs: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    score = np.asarray(base_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    local = np.asarray(local_family, dtype=float)
    if score.shape != (len(features),) or support.shape != score.shape or local.shape != score.shape:
        raise ValueError("NEXT182 base/local shape differs")
    if set(ELIGIBLE_FEATURES) - set(features.columns):
        raise ValueError("NEXT182 eligible feature schema differs")
    columns: dict[str, np.ndarray] = {}
    terms: list[dict[str, object]] = []
    runtime: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in specs:
        spec = dict(raw)
        key = str(spec.get("candidate_key", ""))
        feature_name = spec.get("strong_closure_feature")
        alpha = float(spec.get("attenuation", np.nan))
        if (
            not key or key in seen or feature_name not in {None, *ELIGIBLE_FEATURES}
            or (feature_name is None and alpha != 0.0)
            or (feature_name is not None and alpha not in ATTENUATIONS)
            or spec.get("local_family") != LOCAL_FAMILY
            or tuple(spec.get("local_prefixes", ())) != tuple(LOCAL_PREFIXES)
        ):
            raise ValueError("NEXT182 candidate spec differs")
        seen.add(key)
        feature = np.full(len(features), np.nan) if feature_name is None else pd.to_numeric(features[str(feature_name)], errors="coerce").to_numpy(float)
        corrected, corrected_support, _ = local_family_closure_score(
            base_score=score, base_support=support, local_family=local, feature=feature, attenuation=alpha
        )
        maximum = float(np.max(corrected[corrected_support])) if corrected_support.any() else 0.0
        divisor = max(1.0, maximum / 50.0)
        encoded = np.full(len(features), np.nan)
        encoded[corrected_support] = np.sinh(corrected[corrected_support] / divisor)
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        virtual_id = f"next182_virtual_candidate__{digest}"
        column = f"_{virtual_id}_value"
        columns[column] = encoded
        terms.append({
            "term_id": virtual_id, "feature": column, "direction": 1, "transform": "asinh", "center": 0.0,
            "scale": 1.0 / divisor, "group": "next182_local_family_closure_attenuation",
            "encoding": "asinh_sinh_exact_local_family_closure_score", "physical_candidate_key": key,
        })
        runtime.append({"candidate_key": key, "base_term_ids": [virtual_id], "base_weights": [1.0], "optional_term_id": None, "optional_weight": 0.0})
    if len(seen) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("NEXT182 materialized candidate count differs")
    return pd.concat([features.reset_index(drop=True), pd.DataFrame(columns)], axis=1), terms, runtime


def _paths(roots: Mapping[str, Path], freeze_path: Path, design_path: Path) -> dict[str, Path]:
    paths = n181._paths(roots, freeze_path, design_path)
    paths.update({
        "next181_manifest": roots["next181"] / n181.MANIFEST_NAME,
        "next181_catalogue": roots["next181"] / n181.CATALOGUE_NAME,
        "next181_evaluation": roots["next181"] / n181.EVALUATION_NAME,
        "next181_formula": roots["next181"] / n181.FORMULA_NAME,
        "next181_search": roots["next181"] / n181.SEARCH_NAME,
    })
    return paths


def _local_family_values(
    *, features: pd.DataFrame, physical_terms: Sequence[Mapping[str, object]], base_spec: Mapping[str, object], base_support: np.ndarray
) -> np.ndarray:
    physical_by_id = {str(term["term_id"]): term for term in physical_terms}
    ids = [str(x) for x in base_spec["base_term_ids"]]
    weights = [float(x) for x in base_spec["base_weights"]]
    indices = [i for i, term_id in enumerate(ids) if term_id.startswith(LOCAL_PREFIXES)]
    if not indices:
        raise ValueError("NEXT182 local family is empty")
    columns = []
    supports = []
    for term_id, weight in zip(ids, weights, strict=True):
        risk, supported = _term_risk(features, physical_by_id[term_id])
        columns.append(weight * risk)
        supports.append(supported)
    support_matrix = np.column_stack(supports)
    if not np.array_equal(support_matrix.all(axis=1), base_support):
        raise RuntimeError("NEXT182 physical/base support differs")
    contributions = np.column_stack(columns)
    local = np.full(len(features), np.nan)
    local[base_support] = np.minimum(contributions[base_support][:, indices], n163.CONTRIBUTION_CAP).mean(axis=1)
    return local


def run_local_family_closure_attenuation_search(
    *,
    scigen_feature_dir: Path, scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path, wyformer_discovery_endpoint_dir: Path,
    next98_dir: Path, next110_dir: Path, next111_dir: Path, next113_dir: Path,
    next114_dir: Path, next116_dir: Path, next117_dir: Path, next120_dir: Path,
    next121_dir: Path, next122_dir: Path, next124_dir: Path, next125_dir: Path,
    next129_dir: Path, next130_dir: Path, next133_dir: Path, next134_dir: Path,
    next163_dir: Path, next164_dir: Path, next168_dir: Path, next173_dir: Path,
    next179_dir: Path, next180_dir: Path, next181_dir: Path,
    next135_freeze_path: Path, design_path: Path, output_dir: Path,
    search_workers: int = SEARCH_WORKERS, require_formal_inputs: bool = True,
) -> dict[str, object]:
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(), "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(), "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{f"next{s}": Path(v).resolve() for s,v in (
            (98,next98_dir),(110,next110_dir),(111,next111_dir),(113,next113_dir),(114,next114_dir),(116,next116_dir),
            (117,next117_dir),(120,next120_dir),(121,next121_dir),(122,next122_dir),(124,next124_dir),(125,next125_dir),
            (129,next129_dir),(130,next130_dir),(133,next133_dir),(134,next134_dir),(163,next163_dir),(164,next164_dir),
            (168,next168_dir),(173,next173_dir),(179,next179_dir),(180,next180_dir),(181,next181_dir),
        )},
    }
    target = Path(output_dir).resolve()
    paths = _paths(roots, Path(next135_freeze_path).resolve(), Path(design_path).resolve())
    if os.path.lexists(target): raise FileExistsError(str(target))
    if type(search_workers) is not int or search_workers <= 0: raise ValueError("search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()): raise FileNotFoundError("NEXT182 input is missing")
    input_hashes = {name: _sha256_file(path) for name,path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(k for k in set(input_hashes)|set(EXPECTED_INPUT_SHA256) if input_hashes.get(k)!=EXPECTED_INPUT_SHA256.get(k))
        raise ValueError(f"NEXT182 formal input identity differs: {differing}")

    manifest181 = json.loads(paths["next181_manifest"].read_text())
    if (
        manifest181.get("protocol") != n181.PROTOCOL or manifest181.get("candidate_count") != n181.EXPECTED_CANDIDATE_COUNT
        or manifest181.get("base_endpoint_reproduced") is not True or manifest181.get("freeze_authorized") is not False
        or manifest181.get("passes_all_cross_source_discovery_gates") is not False
        or manifest181.get("strong_closure_repair_width_branch_terminated") is not True
        or manifest181.get("opened_validation_outputs_used") is not False
        or manifest181.get("dft_calculation_executed") is not False or manifest181.get("dft_values_used_by_executable_formula") is not False
        or manifest181.get("outputs_sha256", {}).get(n181.SEARCH_NAME) != input_hashes["next181_search"]
        or manifest181.get("executed_source_sha256", {}).get("src/next181_strong_closure_repair_width_search.py") != _sha256_file(Path(n181.__file__).resolve())
    ): raise ValueError("NEXT182 NEXT181 provenance differs")

    diagnostic164 = json.loads(paths["next164_diagnostic"].read_text())
    base_key = str(diagnostic164.get("global_closest", {}).get("candidate_key", ""))
    if hashlib.sha256(base_key.encode()).hexdigest() != EXPECTED_BASE_CANDIDATE_KEY_SHA256: raise ValueError("NEXT182 base identity differs")
    extended, feature_tables, old_terms, mhcr_terms = n130._join_label_free_features(paths)
    compact=[]
    for source in ("scigen","wyformer"):
        t=pd.read_parquet(paths[f"next133_{source}_features"]).copy(); t["material_id"]=source+":"+t["material_id"].astype(str); compact.append(t)
    extended=extended.merge(pd.concat(compact,ignore_index=True),on="material_id",how="inner",validate="one_to_one")
    extended=pd.concat([extended.reset_index(drop=True),n135.materialize_conjunctive_features(extended).reset_index(drop=True)],axis=1)
    closure=[]
    for source in ("scigen","wyformer"):
        t=pd.read_parquet(paths[f"next179_{source}_features"]).copy(); t["material_id"]=source+":"+t["material_id"].astype(str); closure.append(t)
    combined=extended.merge(pd.concat(closure,ignore_index=True),on="material_id",how="inner",validate="one_to_one")
    physical_terms=[*old_terms,*mhcr_terms]; physical_ids={str(t["term_id"]) for t in physical_terms}
    all_bases=n130.n127.select_next125_bases(pd.read_parquet(paths["next125_search_records"]))
    bases=n135.n132.select_extended_bases(pd.read_parquet(paths["next130_search_records"]),all_bases)
    base_specs=n163.build_candidate_specs(bases=bases,physical_term_ids=physical_ids)
    selected_specs=[s for s in base_specs if str(s["candidate_key"])==base_key]
    if len(selected_specs)!=1: raise ValueError("NEXT182 base reconstruction differs")
    combined,base_terms,base_runtime=n163.materialize_candidates(features=combined,physical_terms=physical_terms,specs=selected_specs)
    base_score,base_support=_term_risk(combined,base_terms[0])
    local=_local_family_values(features=combined,physical_terms=physical_terms,base_spec=selected_specs[0],base_support=base_support)
    se=pd.read_parquet(paths["scigen_endpoint"]); we=pd.read_parquet(paths["wyformer_endpoint"])
    ef=pd.concat([
        pd.DataFrame({"material_id":"scigen:"+se.material_id.astype(str),"_endpoint":pd.to_numeric(se.distortion_ratio,errors="coerce")}),
        pd.DataFrame({"material_id":"wyformer:"+we.material_id.astype(str),"_endpoint":n130.n125.n121.prior._endpoint_numeric(we.endpoint_stratum)}),
    ],ignore_index=True)
    combined=combined.merge(ef,on="material_id",how="inner",validate="one_to_one")
    endpoint=pd.to_numeric(combined.pop("_endpoint"),errors="coerce").to_numpy(float)
    if not np.isfinite(endpoint).all() or len(endpoint)!=len(base_score): raise ValueError("NEXT182 endpoint accounting differs")
    specs=build_candidate_specs(base_candidate_key=base_key)
    combined,virtual_terms,runtime=materialize_local_family_candidates(features=combined,base_score=base_score,base_support=base_support,local_family=local,specs=specs)
    started=time.perf_counter(); result=n130.n125.search_optional_guard_laws_parallel(features=combined,endpoint=endpoint,old_terms=virtual_terms,optional_terms=[],candidate_specs=runtime,workers=search_workers); elapsed=time.perf_counter()-started
    if int(result["candidate_count"])!=EXPECTED_CANDIDATE_COUNT: raise RuntimeError("NEXT182 evaluator count differs")
    spec_by_key={str(s["candidate_key"]):s for s in specs}
    def decorate(record:dict[str,object])->None:
        spec=spec_by_key[str(record["candidate_key"])]; record.update({"base_candidate_key":base_key,"strong_closure_feature":spec["strong_closure_feature"],"attenuation":float(spec["attenuation"]),"broad_threshold":BROAD_THRESHOLD,"safe_threshold_frozen":SAFE_THRESHOLD,"local_family":LOCAL_FAMILY,"local_prefixes_json":json.dumps(list(LOCAL_PREFIXES),separators=(",",":")),"missing_policy":"TERM_OFF_KEEP_BASE","score_composition":SCORE_COMPOSITION})
    for record in result["candidate_records"]: decorate(record)
    selected=result["selected"]
    if "attenuation" not in selected["record"]: decorate(selected["record"])
    base_records=[r for r in result["candidate_records"] if r["strong_closure_feature"] is None]
    n181.n175.n170._verify_base_reproduction(record=base_records[0],published=pd.read_parquet(paths["next163_search"]),candidate_key=base_key)
    selected_spec=spec_by_key[str(selected["record"]["candidate_key"])]
    prior=json.loads(paths["next163_evaluation"].read_text())
    formula={"protocol":PROTOCOL,"kind":"local_geometry_family_strong_closure_attenuation_no_dft_score","base_candidate_key":base_key,"base_formula":prior["selected_formula"],"strong_closure_feature":selected_spec["strong_closure_feature"],"attenuation":float(selected_spec["attenuation"]),"local_family":LOCAL_FAMILY,"local_prefixes":list(LOCAL_PREFIXES),"contribution_cap":n163.CONTRIBUTION_CAP,"broad_threshold":BROAD_THRESHOLD,"safe_threshold":SAFE_THRESHOLD,"interval_policy":"BROAD_INCLUSIVE_SAFE_EXCLUSIVE_ON_ORIGINAL_BASE_SCORE","missing_policy":"TERM_OFF_KEEP_BASE","score_composition":SCORE_COMPOSITION,"dft_values_used_by_executable_formula":False,"learned_energy_force_stress_proxy_used":False,"physical_relaxation_executed":False}
    passes=bool(selected["record"]["passes_all_discovery_gates"]); records=pd.DataFrame(result["candidate_records"])
    counts={}
    for (feature,alpha),frame in records.assign(strong_closure_feature=records.strong_closure_feature.fillna("BASE")).groupby(["strong_closure_feature","attenuation"],sort=True):
        counts[f"feature={feature},alpha={float(alpha):g}"]={"candidates":int(len(frame)),"passes_source_auc_gates":int(frame.passes_source_auc_gates.sum()),"passes_safe_all_cells":int(frame.passes_safe_all_cells.sum()),"passes_broad_all_cells":int(frame.passes_broad_all_cells.sum()),"passes_all_discovery_gates":int(frame.passes_all_discovery_gates.sum())}
    catalogue={"protocol":PROTOCOL,"design_sha256":input_hashes["design"],"base_candidate_key_sha256":EXPECTED_BASE_CANDIDATE_KEY_SHA256,"base_endpoint_reproduced":True,"eligible_features":ELIGIBLE_FEATURES,"attenuation_grid":ATTENUATIONS,"candidate_count":EXPECTED_CANDIDATE_COUNT,"local_family":LOCAL_FAMILY,"local_prefixes":LOCAL_PREFIXES,"contribution_cap":n163.CONTRIBUTION_CAP,"broad_threshold":BROAD_THRESHOLD,"safe_threshold":SAFE_THRESHOLD,"score_composition":SCORE_COMPOSITION,"base_support_unchanged":True,"outside_interval_exactly_unchanged":True,"validation_outputs_opened":False,"dft_values_used_by_executable_formula":False}
    target.parent.mkdir(parents=True,exist_ok=True); staging=Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-",dir=target.parent))
    source_paths={"src/next163_interior_family_attenuation_search.py":Path(n163.__file__).resolve(),"src/next179_strong_neighborhood_directional_closure.py":Path(n179.__file__).resolve(),"src/next182_local_family_closure_attenuation_search.py":Path(__file__).resolve()}; source_hashes={k:_sha256_file(v) for k,v in source_paths.items()}
    try:
        cp=staging/CATALOGUE_NAME; ep=staging/EVALUATION_NAME; fp=staging/FORMULA_NAME; sp=staging/SEARCH_NAME
        _write_json(cp,catalogue); _write_json(ep,{"protocol":PROTOCOL,"evaluation_mode":"fixed_local_family_closure_attenuation_search","base_endpoint_reproduced":True,"rows":{"scigen":int(len(feature_tables["scigen"])),"wyformer":int(len(feature_tables["wyformer"])),"total":int(len(combined))},"candidate_count":int(result["candidate_count"]),"elapsed_seconds":elapsed,"search_workers":search_workers,"counts_by_feature_and_attenuation":counts,"selected_record":selected["record"],"selected_formula":formula,"selected_safe":selected["safe"],"selected_safe_diagnostic":selected["safe_diagnostic"],"selected_broad":selected["broad"],"selected_source_diagnostics":selected["source_diagnostics"],"pauling_by_cell":result["pauling_by_cell"],"cells":result["cells"],"passes_all_cross_source_discovery_gates":passes,"freeze_authorized":passes,"requires_unopened_internal_validation_before_claim":True}); _write_json(fp,formula); records.to_parquet(sp,index=False)
        outputs=[cp,ep,fp,sp]; manifest={"protocol":PROTOCOL,"candidate_count":int(result["candidate_count"]),"search_workers":search_workers,"base_endpoint_reproduced":True,"passes_all_cross_source_discovery_gates":passes,"freeze_authorized":passes,"requires_unopened_internal_validation_before_claim":True,"local_family_closure_attenuation_branch_terminated":not passes,"scigen_discovery_endpoint_opened":True,"wyformer_discovery_endpoint_opened":True,"discovery_outcomes_used_as_offline_labels":True,"opened_validation_outputs_used":False,"scigen_replication_endpoint_opened":False,"wyformer_replication_endpoint_opened":False,"formula_or_threshold_changed_after_search":False,"dft_calculation_executed":False,"dft_values_used_by_executable_formula":False,"learned_energy_force_stress_proxy_used":False,"physical_relaxation_executed":False,"scientific_improvement_claim":False,"inputs_sha256":input_hashes,"executed_source_sha256":source_hashes,"outputs_sha256":{p.name:_sha256_file(p) for p in outputs}}
        _write_json(staging/MANIFEST_NAME,manifest)
        if any(_sha256_file(p)!=input_hashes[n] for n,p in paths.items()): raise RuntimeError("NEXT182 input changed before publication")
        if any(_sha256_file(p)!=source_hashes[n] for n,p in source_paths.items()): raise RuntimeError("NEXT182 source changed before publication")
        os.replace(staging,target); return manifest
    except Exception:
        shutil.rmtree(staging,ignore_errors=True); raise


def main()->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--scigen-feature-dir",type=Path,required=True); p.add_argument("--scigen-discovery-endpoint-dir",type=Path,required=True); p.add_argument("--wyformer-feature-dir",type=Path,required=True); p.add_argument("--wyformer-discovery-endpoint-dir",type=Path,required=True)
    stages=(98,110,111,113,114,116,117,120,121,122,124,125,129,130,133,134,163,164,168,173,179,180,181)
    for s in stages:p.add_argument(f"--next{s}-dir",type=Path,required=True)
    p.add_argument("--next135-freeze-path",type=Path,required=True);p.add_argument("--design-path",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--search-workers",type=int,default=SEARCH_WORKERS);p.add_argument("--allow-nonformal-inputs",action="store_true");a=p.parse_args()
    m=run_local_family_closure_attenuation_search(scigen_feature_dir=a.scigen_feature_dir,scigen_discovery_endpoint_dir=a.scigen_discovery_endpoint_dir,wyformer_feature_dir=a.wyformer_feature_dir,wyformer_discovery_endpoint_dir=a.wyformer_discovery_endpoint_dir,**{f"next{s}_dir":getattr(a,f"next{s}_dir") for s in stages},next135_freeze_path=a.next135_freeze_path,design_path=a.design_path,output_dir=a.output_dir,search_workers=a.search_workers,require_formal_inputs=not a.allow_nonformal_inputs);print(json.dumps(m,indent=2,sort_keys=True));return 0


__all__=["ATTENUATIONS","BROAD_THRESHOLD","ELIGIBLE_FEATURES","EXPECTED_CANDIDATE_COUNT","SAFE_THRESHOLD","build_candidate_specs","local_family_closure_score","materialize_local_family_candidates","run_local_family_closure_attenuation_search"]
if __name__=="__main__":raise SystemExit(main())
