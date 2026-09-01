#!/usr/bin/env python3
"""Freeze three composition-only mechanism candidates before new endpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.next19_feature_build import _publish_directory_no_replace, _sha256
import src.next552_hea_analytic_feature_freeze as n552
import src.next553_hea_development_search as n553
import src.next555_hea_extreme_waste_search as n555
import src.next558_hea_packing_deficit_validation as n558
import src.next561_hea_entropy_packing_confirmation as n561
import src.next562_hea_stable_analytic_union_search as n562


PROTOCOL = "2026-08-14-next565-hea-mechanism-formula-family-v1"
DESIGN_SHA256 = "71941600edae2cebd3b55e2e589077abb107368446160504d1bca7a59dcb5a31"
FORMULA_NAME = "NEXT565_FROZEN_MECHANISM_FORMULA_FAMILY.json"
RESULT_NAME = "NEXT565_HEA_MECHANISM_FORMULA_EVIDENCE.json"
MANIFEST_NAME = "MANIFEST.json"
CANDIDATE_NAMES = ("MEMAX", "MEPU24", "ZEPU24")


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def candidate_scores(u_h: object, u_m: object, u_z: object) -> dict[str, np.ndarray]:
    u_h, u_m, u_z = map(lambda value: np.asarray(value, dtype=float), (u_h, u_m, u_z))
    if u_h.shape != u_m.shape or u_h.shape != u_z.shape:
        raise ValueError("NEXT565 candidate risk arrays differ")
    return {
        "MEMAX": np.maximum(u_h, u_m),
        "MEPU24": 1.0 - (1.0 - u_h**2) * (1.0 - u_m**4),
        "ZEPU24": 1.0 - (1.0 - u_h**2) * (1.0 - u_z**4),
    }


def build_family(
    *, next562_dir: Path, next555_dir: Path, next558_dir: Path,
    next561_dir: Path, design_path: Path, output_dir: Path,
) -> dict[str, object]:
    roots = [Path(value).resolve() for value in (next562_dir, next555_dir, next558_dir, next561_dir)]
    root562, root555, root558, root561 = roots
    design_path, target = Path(design_path).resolve(), Path(output_dir).resolve()
    paths = {
        "design": design_path,
        "next562_manifest": root562 / n562.MANIFEST_NAME,
        "next562_table": root562 / n562.TABLE_NAME,
        "next555_endpoints": root555 / n555.ENDPOINT_TABLE_NAME,
        "next558_table": root558 / n558.TABLE_NAME,
        "next561_table": root561 / n561.TABLE_NAME,
        "next561_manifest": root561 / n561.MANIFEST_NAME,
        "formula_source": Path(__file__).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT565 input is missing")
    if _sha256(design_path) != DESIGN_SHA256:
        raise ValueError("NEXT565 design identity differs")
    manifest562 = json.loads(paths["next562_manifest"].read_text())
    manifest561 = json.loads(paths["next561_manifest"].read_text())
    if (
        manifest562.get("protocol") != n562.PROTOCOL
        or manifest562.get("discovery_pass") is not False
        or manifest562.get("outputs_sha256", {}).get(n562.TABLE_NAME) != _sha256(paths["next562_table"])
        or manifest561.get("protocol") != n561.PROTOCOL
        or manifest561.get("confirmation_pass") is not False
    ):
        raise ValueError("NEXT565 opened development identity differs")
    table = pd.read_parquet(paths["next562_table"])
    endpoints = pd.concat(
        [
            pd.read_parquet(path)[
                ["fid", "e_above_hull", "disp_p90", "cell_logstrain_max", "volume_logchange"]
            ]
            for path in (paths["next555_endpoints"], paths["next558_table"], paths["next561_table"])
        ],
        ignore_index=True,
    ).drop_duplicates("fid")
    table = table.merge(endpoints, on="fid", validate="one_to_one")
    table["energy_extreme"] = table["e_above_hull"].to_numpy(float) >= n555.ENERGY_HULL_THRESHOLD
    table["geometric_extreme"] = (
        (table["disp_p90"].to_numpy(float) >= n553.DISPLACEMENT_P90_THRESHOLD)
        | (table["cell_logstrain_max"].to_numpy(float) >= n553.CELL_LOGSTRAIN_THRESHOLD)
        | (table["volume_logchange"].to_numpy(float) >= n553.VOLUME_LOGCHANGE_THRESHOLD)
    )
    u_h = n552._midrank(table["composition_ideal_entropy"])
    u_m = n552._midrank(table["composition_atomic_mass_cv"])
    u_z = n552._midrank(table["composition_atomic_number_cv"])
    scores = candidate_scores(u_h, u_m, u_z)
    masks = n562._masks(table)
    evidence = {
        "protocol": PROTOCOL,
        "counts": {
            "rows": len(table),
            "energy_extreme": int(table["energy_extreme"].sum()),
            "geometric_extreme": int(table["geometric_extreme"].sum()),
            "mechanism_overlap": int((table["energy_extreme"] & table["geometric_extreme"]).sum()),
        },
        "mechanism_univariate_auc": {
            "entropy_to_energy": float(roc_auc_score(table["energy_extreme"], u_h)),
            "mass_cv_to_geometry": float(roc_auc_score(table["geometric_extreme"], u_m)),
            "atomic_number_cv_to_geometry": float(roc_auc_score(table["geometric_extreme"], u_z)),
        },
        "candidate_metrics": {
            name: {
                stratum: n553._score_metrics(table, score, mask)
                for stratum, mask in masks.items()
            }
            for name, score in scores.items()
        },
        "opened_data_formulation": True,
        "scientific_success_claim": False,
    }
    formula = {
        "protocol": PROTOCOL,
        "normalization": "full-candidate-batch midrank percentiles",
        "components": {
            "u_H": "midrank(H=-sum_i x_i ln(x_i))",
            "u_M": "midrank(weighted_std(atomic_mass)/weighted_mean(atomic_mass))",
            "u_Z": "midrank(weighted_std(atomic_number)/weighted_mean(atomic_number))",
        },
        "candidates": {
            "MEPU24": "1-(1-u_H^2)*(1-u_M^4)",
            "ZEPU24": "1-(1-u_H^2)*(1-u_Z^4)",
            "MEMAX": "max(u_H,u_M)",
        },
        "candidate_order": list(CANDIDATE_NAMES),
        "endpoint_fitted_coefficients": False,
        "dft_inputs_at_execution": False,
        "next566_endpoints_opened": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_hash = _sha256(Path(__file__).resolve())
    try:
        result_path, formula_path = staging / RESULT_NAME, staging / FORMULA_NAME
        result_path.write_bytes(_json_bytes(evidence))
        formula_path.write_bytes(_json_bytes(formula))
        outputs = {RESULT_NAME: _sha256(result_path), FORMULA_NAME: _sha256(formula_path)}
        manifest = {
            "protocol": PROTOCOL,
            "inputs_sha256": {
                name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()
            },
            "outputs_sha256": outputs,
            "executed_source_sha256": {
                "src/next565_hea_mechanism_formula_family.py": source_hash
            },
            "opened_data_formulation": True,
            "next566_endpoint_values_opened": False,
            "next566_cohort_authorized": True,
            "dft_values_used_by_executable_formula": False,
            "model_or_proxy_potential_used": False,
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
    parser.add_argument("--next562-dir", required=True, type=Path)
    parser.add_argument("--next555-dir", required=True, type=Path)
    parser.add_argument("--next558-dir", required=True, type=Path)
    parser.add_argument("--next561-dir", required=True, type=Path)
    parser.add_argument("--design-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = build_family(
        next562_dir=args.next562_dir, next555_dir=args.next555_dir,
        next558_dir=args.next558_dir, next561_dir=args.next561_dir,
        design_path=args.design_path, output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_family", "candidate_scores"]
