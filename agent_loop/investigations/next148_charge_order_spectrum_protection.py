#!/usr/bin/env python3
"""Materialize frozen long-wavelength formal-charge order protection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

import src.next133_compactness_protection as n133
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-08-next148-charge-order-spectrum-protection-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT148_CHARGE_ORDER_SPECTRUM_PROTECTION_CATALOGUE.json"
RAW_FEATURE = "csf_long_fraction"
RAW_SUPPORT = "next43_charge_spectrum_supported"
FEATURE_NAME = "charge_order_spectrum_protection"
SUPPORT_COLUMN = "charge_order_spectrum_protection_supported"
FEATURE_FILES = {
    "scigen": "next148_scigen_charge_order_spectrum_protection.parquet",
    "wyformer": "next148_wyformer_charge_order_spectrum_protection.parquet",
}
EXPECTED_DESIGN_SHA256 = "95677c0bab64d7e8492d5232e3f4c2140d471d59b722ec0194fb343d3a39b6f2"
EXPECTED_INPUT_SHA256 = {
    **{key:value for key,value in n133.EXPECTED_INPUT_SHA256.items() if key != "design"},
    "design": EXPECTED_DESIGN_SHA256,
}


def materialize_charge_order_protection(table: pd.DataFrame) -> pd.DataFrame:
    required={"material_id",RAW_FEATURE,RAW_SUPPORT}
    if required-set(table.columns) or table["material_id"].astype(str).duplicated().any():
        raise ValueError("NEXT148 feature schema differs")
    raw=pd.to_numeric(table[RAW_FEATURE],errors="coerce").to_numpy(float);declared=table[RAW_SUPPORT].eq(True).to_numpy()
    if np.any(declared & (~np.isfinite(raw) | (raw < -1e-12) | (raw > 1.0+1e-12))):
        raise ValueError("NEXT148 supported charge spectrum differs")
    active=declared & np.isfinite(raw) & (raw>=-1e-12) & (raw<=1.0+1e-12)
    values=np.full(len(table),np.nan,dtype=float);values[active]=1.0-np.clip(raw[active],0.0,1.0)
    return pd.DataFrame({"material_id":table["material_id"].astype(str),FEATURE_NAME:values,SUPPORT_COLUMN:active})


def build_charge_order_protection(*,scigen_feature_dir:Path,wyformer_feature_dir:Path,design_path:Path,output_dir:Path,require_formal_inputs:bool=True)->dict[str,object]:
    roots={"scigen":Path(scigen_feature_dir).resolve(),"wyformer":Path(wyformer_feature_dir).resolve()};target=Path(output_dir).resolve()
    paths={"scigen_manifest":roots["scigen"]/"MANIFEST.json","scigen_features":roots["scigen"]/"features_discovery.parquet","wyformer_manifest":roots["wyformer"]/"MANIFEST.json","wyformer_features":roots["wyformer"]/"wyformer_x0_features_discovery.parquet","design":Path(design_path).resolve()}
    if os.path.lexists(target):raise FileExistsError(str(target))
    if any(not p.is_file() for p in paths.values()):raise FileNotFoundError("NEXT148 input is missing")
    hashes={n:_sha256_file(p) for n,p in paths.items()}
    if require_formal_inputs and hashes!=EXPECTED_INPUT_SHA256:raise ValueError("NEXT148 formal input identity differs")
    for source,protocol,filename,key in (("scigen",n133.SCIGEN_PROTOCOL,"features_discovery.parquet","scigen_features"),("wyformer",n133.WYFORMER_PROTOCOL,"wyformer_x0_features_discovery.parquet","wyformer_features")):
        m=json.loads(paths[f"{source}_manifest"].read_text())
        if m.get("protocol")!=protocol or m.get("labels_opened") is not False or m.get("endpoint_payloads_opened") is not False or m.get("dft_values_used_by_features") is not False or m.get("outputs_sha256",{}).get(filename)!=hashes[key]:raise ValueError("NEXT148 prior provenance differs")
    target.parent.mkdir(parents=True,exist_ok=True);staging=Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-",dir=target.parent));outputs=[];diagnostics={}
    try:
        for source in ("scigen","wyformer"):
            raw=pd.read_parquet(paths[f"{source}_features"],columns=["material_id",RAW_FEATURE,RAW_SUPPORT]);result=materialize_charge_order_protection(raw);path=staging/FEATURE_FILES[source];result.to_parquet(path,index=False);outputs.append(path)
            active=result[SUPPORT_COLUMN].eq(True).to_numpy();values=result[FEATURE_NAME].to_numpy(float);diagnostics[source]={"rows":int(len(result)),"supported":int(active.sum()),"positive":int((active&(values>0)).sum()),"full_protection":int((active&np.isclose(values,1.0)).sum()),"quantiles":{name:float(value) for name,value in zip(("q50","q75","q90","q95","q99","max"),np.quantile(values[active],[.5,.75,.9,.95,.99,1]))}}
        catalogue={"protocol":PROTOCOL,"feature":{"name":FEATURE_NAME,"raw_feature":RAW_FEATURE,"raw_support":RAW_SUPPORT,"definition":"1 - clip(raw,0,1)","support_column":SUPPORT_COLUMN,"mechanism":"complement_of_long_wavelength_formal_charge_spectrum_fraction"},"missing_policy":"TERM_OFF_KEEP_BASE","diagnostics":diagnostics,"labels_opened":False,"endpoint_columns_present":False}
        cp=staging/CATALOGUE_NAME;_write_json(cp,catalogue);outputs.append(cp);source_paths={"src/next148_charge_order_spectrum_protection.py":Path(__file__).resolve()};source_hashes={n:_sha256_file(p) for n,p in source_paths.items()}
        manifest={"protocol":PROTOCOL,"inputs_sha256":hashes,"executed_source_sha256":source_hashes,"outputs_sha256":{p.name:_sha256_file(p) for p in outputs},"diagnostics":diagnostics,"labels_opened":False,"endpoint_payloads_opened":False,"opened_validation_outputs_used":False,"scigen_replication_endpoint_opened":False,"wyformer_replication_endpoint_opened":False,"dft_calculation_executed":False,"dft_values_used_by_features":False,"learned_energy_force_stress_proxy_used":False,"analytic_formal_charge_spectrum_used":True,"physical_relaxation_executed":False,"scientific_improvement_claim":False};_write_json(staging/MANIFEST_NAME,manifest)
        if any(_sha256_file(p)!=hashes[n] for n,p in paths.items()):raise RuntimeError("NEXT148 input changed")
        if any(_sha256_file(p)!=source_hashes[n] for n,p in source_paths.items()):raise RuntimeError("NEXT148 source changed")
        os.replace(staging,target);return manifest
    except Exception:shutil.rmtree(staging,ignore_errors=True);raise


def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--scigen-feature-dir",type=Path,required=True);p.add_argument("--wyformer-feature-dir",type=Path,required=True);p.add_argument("--design-path",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--allow-nonformal-inputs",action="store_true");a=p.parse_args();m=build_charge_order_protection(scigen_feature_dir=a.scigen_feature_dir,wyformer_feature_dir=a.wyformer_feature_dir,design_path=a.design_path,output_dir=a.output_dir,require_formal_inputs=not a.allow_nonformal_inputs);print(json.dumps(m,indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())


__all__=["build_charge_order_protection","materialize_charge_order_protection"]
