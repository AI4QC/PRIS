#!/usr/bin/env python3
"""Formalize the opened-data HEA entropy-packing discovery."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from ase import Atoms
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy.stats import rankdata

from src.next19_feature_build import _publish_directory_no_replace, _sha256
import src.next552_hea_analytic_feature_freeze as n552
import src.next553_hea_development_search as n553
import src.next555_hea_extreme_waste_search as n555
import src.next558_hea_packing_deficit_validation as n558


PROTOCOL = "2026-08-14-next559-hea-entropy-packing-discovery-v1"
DESIGN_SHA256 = "65e08885a13daeea0ecc6fb27651d058471978d2823b344b48fee6498e703584"
PACKING_RAW = "primitive_covalent_packing_fraction"
PACKING_RISK = "epcu_packing_deficit_risk"
ENTROPY_RAW = "composition_entropy"
ENTROPY_RISK = "epcu_entropy_risk"
SCORE = "epcu_risk"
TABLE_NAME = "next559_hea_entropy_packing_discovery.parquet"
RESULT_NAME = "NEXT559_HEA_ENTROPY_PACKING_DISCOVERY.json"
FORMULA_NAME = "NEXT559_FROZEN_EPCU_FORMULA.json"
MANIFEST_NAME = "MANIFEST.json"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def composition_entropy(atoms: Atoms) -> float:
    counts = np.asarray(list(Counter(np.asarray(atoms.numbers, dtype=int)).values()), dtype=float)
    fractions = counts / counts.sum()
    return float(-(fractions * np.log(fractions)).sum())


def _formula_entropy(formula: str) -> float:
    composition = Composition(formula)
    fractions = np.asarray(
        [composition[element] / composition.num_atoms for element in composition.elements],
        dtype=float,
    )
    return float(-(fractions * np.log(fractions)).sum())


def _midrank(values: object, *, reverse: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    base = -values if reverse else values
    return (rankdata(base, method="average") - 0.5) / len(values)


def entropy_packing_union(entropy_risk: object, packing_risk: object) -> np.ndarray:
    entropy_risk = np.asarray(entropy_risk, dtype=float)
    packing_risk = np.asarray(packing_risk, dtype=float)
    if entropy_risk.shape != packing_risk.shape:
        raise ValueError("NEXT559 risk arrays differ")
    return 1.0 - (1.0 - entropy_risk) * (1.0 - packing_risk)


def build_discovery(
    *, next552_dir: Path, next555_dir: Path, next558_dir: Path,
    design_path: Path, output_dir: Path,
) -> dict[str, object]:
    root552, root555, root558 = map(
        lambda path: Path(path).resolve(), (next552_dir, next555_dir, next558_dir)
    )
    design_path = Path(design_path).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "design": design_path,
        "features": root552 / n552.TABLE_NAME,
        "development_endpoints": root555 / n555.ENDPOINT_TABLE_NAME,
        "validation_joined": root558 / n558.TABLE_NAME,
        "next552_manifest": root552 / n552.MANIFEST_NAME,
        "next555_manifest": root555 / n555.MANIFEST_NAME,
        "next558_manifest": root558 / n558.MANIFEST_NAME,
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT559 input is missing")
    if _sha256(design_path) != DESIGN_SHA256:
        raise ValueError("NEXT559 design identity differs")
    manifest558 = json.loads(paths["next558_manifest"].read_text())
    if (
        manifest558.get("protocol") != n558.PROTOCOL
        or manifest558.get("validation_endpoint_values_opened") is not True
        or manifest558.get("confirmation_pass") is not False
    ):
        raise ValueError("NEXT559 opened discovery state differs")
    features = pd.read_parquet(paths["features"])
    development = pd.read_parquet(paths["development_endpoints"])
    validation = pd.read_parquet(paths["validation_joined"])
    endpoints = pd.concat(
        [
            development[["fid", "dft_waste", "waste_severity", "protected"]],
            validation[["fid", "dft_waste", "waste_severity", "protected"]],
        ],
        ignore_index=True,
    )
    table = features.merge(endpoints, on="fid", validate="one_to_one")
    if len(table) != 2_400:
        raise ValueError("NEXT559 opened discovery join differs")
    table[ENTROPY_RAW] = table["reduced_formula"].astype(str).map(_formula_entropy)
    table[ENTROPY_RISK] = _midrank(table[ENTROPY_RAW])
    table[PACKING_RISK] = _midrank(table[PACKING_RAW], reverse=True)
    table[SCORE] = entropy_packing_union(table[ENTROPY_RISK], table[PACKING_RISK])
    masks = {
        "overall": np.ones(len(table), dtype=bool),
        "old_development": table["partition"].astype(str).eq("development").to_numpy(),
        "old_validation": table["partition"].astype(str).eq("validation").to_numpy(),
        "ordered": table["size_family"].astype(str).eq("ordered").to_numpy(),
        "sqs": table["size_family"].astype(str).eq("sqs").to_numpy(),
    }
    metrics = {
        name: n553._score_metrics(table, table[SCORE].to_numpy(float), mask)
        for name, mask in masks.items()
    }
    components = {
        name: {
            "packing": n553._score_metrics(table, table[PACKING_RISK].to_numpy(float), mask),
            "entropy": n553._score_metrics(table, table[ENTROPY_RISK].to_numpy(float), mask),
        }
        for name, mask in masks.items()
    }
    evidence_pass = bool(
        metrics["overall"]["roc_auc"] >= 0.75
        and metrics["old_development"]["roc_auc"] >= 0.75
        and metrics["old_validation"]["roc_auc"] >= 0.75
        and metrics["ordered"]["roc_auc"] >= 0.80
        and metrics["sqs"]["roc_auc"] >= 0.72
        and metrics["overall"]["spearman_severity"] >= 0.40
    )
    if not evidence_pass:
        raise RuntimeError(f"NEXT559 discovery evidence differs: {metrics}")
    result = {
        "protocol": PROTOCOL,
        "metrics": metrics,
        "components": components,
        "opened_data_discovery": True,
        "discovery_evidence_pass": evidence_pass,
        "scientific_improvement_claim": False,
    }
    formula = {
        "protocol": PROTOCOL,
        "name": "entropy-packing capacity union",
        "short_name": "EPCU",
        "composition_entropy": "H=-sum_i x_i ln(x_i)",
        "packing_fraction": "phi=sum_i 4*pi*r_cov_i^3/(3*V_cell)",
        "risk_percentiles": ["u_H=midrank(H)", "u_phi=midrank(-phi)"],
        "formula": "EPCU=1-(1-u_H)*(1-u_phi)",
        "operating_rule": "reject highest-risk 15 percent; FID breaks ties",
        "endpoint_fitted_coefficients": False,
        "next560_endpoints_opened": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    try:
        table_path, result_path, formula_path = (
            staging / TABLE_NAME, staging / RESULT_NAME, staging / FORMULA_NAME
        )
        table.to_parquet(table_path, index=False)
        result_path.write_bytes(_json_bytes(result))
        formula_path.write_bytes(_json_bytes(formula))
        outputs = {
            TABLE_NAME: _sha256(table_path), RESULT_NAME: _sha256(result_path),
            FORMULA_NAME: _sha256(formula_path),
        }
        manifest = {
            "protocol": PROTOCOL,
            "inputs_sha256": {
                name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()
            },
            "outputs_sha256": outputs,
            "executed_source_sha256": {
                "src/next559_hea_entropy_packing_discovery.py": source_hash
            },
            "opened_data_discovery": True,
            "next560_endpoint_values_opened": False,
            "dft_values_used_by_executable_formula": False,
            "next560_cohort_authorized": True,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next552-dir", required=True, type=Path)
    parser.add_argument("--next555-dir", required=True, type=Path)
    parser.add_argument("--next558-dir", required=True, type=Path)
    parser.add_argument("--design-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = build_discovery(
        next552_dir=args.next552_dir, next555_dir=args.next555_dir,
        next558_dir=args.next558_dir, design_path=args.design_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_discovery", "composition_entropy", "entropy_packing_union"]
