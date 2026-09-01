#!/usr/bin/env python3
"""Freeze label-free SSSP-BVC features, decisions, and Pauling controls on WBM x0."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import pandas as pd

import src.next12_pauling_controls as n12
import src.next22_bond_valence_equilibrium as n22
import src.next267_periodic_radical_voronoi_packing as n267
import src.next411_same_sign_shell_purity as n411
import src.next529_sssp_bvc_development_freeze as n529
import src.next530_sssp_bvc_wbm_cohort as n530
from src.next347_periodic_allocation_redistribution_capacity import _sha256_file


PROTOCOL = "2026-08-13-next531-sssp-bvc-wbm-label-free-features-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT531_SSSP_BVC_FEATURE_CATALOGUE.json"
TABLE_NAME = "next531_sssp_bvc_wbm_predictions.parquet"
MINIMUM_SSSP_COVERAGE = 0.90
MINIMUM_UNIQUE = 20
BOUNDARY_FLAGS = {
    "wbm_summary_opened": False,
    "external_endpoint_opened": False,
    "relaxed_structures_opened": False,
    "dft_calculation_executed": False,
    "dft_values_used_by_features": False,
    "learned_energy_force_stress_proxy_used": False,
    "model_or_proxy_potential_used": False,
    "physical_relaxation_executed": False,
}
EXPECTED_INPUT_SHA256 = {
    "design": n529.DESIGN_SHA256,
    "next529_manifest": "3f5bfa89726bfa7edc8daa898169c3e9259c5d3d29e1d12c2674fb4343f17705",
    "next529_formula": "b50e194273e83f06e26bd4f4e9c904cd692dc9fa9d874aebb0181c4fcfa849be",
    "next530_manifest": "c794f740d056816c9cefd3acef61a17a3ffb7aaf2143061cb93d41870ab9bb6b",
    "next530_metadata": "f9922c13dccd4b3f4b1fad8f991f16910ed92d8c88e6ccf437469c435da318b5",
    "next530_geometry": "0623839699cd50de5705f23b5b21d48e6a4843ca1b069d8d9aeafe8f5bb36ef6",
    "next411_source": "172543534328a387b7d2b12ffd6cad919793ace56ec1124dd6e228f96d8cc9a4",
    "next22_source": "79635d205bfaefbabdc63012b482f64fe259f1f487abd5f351783b944ff8ed93",
    "pauling_source": "b37f1a84326e8104b38bd61398ee10cf9f6421007fead5d004a0601cb5159c43",
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def apply_frozen_formula(table: pd.DataFrame) -> pd.DataFrame:
    required = {
        n411.FEATURE_NAMES[0], "sssp_supported", n529.SCBV_FEATURE, "scbv_supported"
    }
    if required - set(table):
        raise ValueError("NEXT531 feature table differs")
    applied = n529.apply_sssp_bvc(
        sssp=pd.to_numeric(table[n411.FEATURE_NAMES[0]], errors="coerce").to_numpy(float),
        sssp_supported=table["sssp_supported"].fillna(False).to_numpy(bool),
        scbv=pd.to_numeric(table[n529.SCBV_FEATURE], errors="coerce").to_numpy(float),
        scbv_supported=table["scbv_supported"].fillna(False).to_numpy(bool),
        sssp_threshold=n529.SSSP_THRESHOLD,
        scbv_threshold=n529.EXPECTED_SCBV_THRESHOLD,
    )
    result = table.copy()
    result["risk_score"] = applied["risk"]
    result["formula_supported"] = applied["supported"]
    result["reject"] = applied["reject"]
    result["sssp_bvc_decision"] = np.where(
        ~applied["supported"], "ABSTAIN", np.where(applied["reject"], "REJECT", "KEEP")
    )
    return result


def label_free_gate_statistics(table: pd.DataFrame) -> dict[str, object]:
    shell = pd.to_numeric(table[n411.FEATURE_NAMES[0]], errors="coerce").to_numpy(float)
    mismatch = pd.to_numeric(table[n529.SCBV_FEATURE], errors="coerce").to_numpy(float)
    shell_support = table["sssp_supported"].fillna(False).to_numpy(bool)
    mismatch_support = table["scbv_supported"].fillna(False).to_numpy(bool)
    if (
        not np.array_equal(shell_support, shell_support & np.isfinite(shell))
        or not np.array_equal(mismatch_support, mismatch_support & np.isfinite(mismatch))
    ):
        raise ValueError("NEXT531 support semantics differ")
    shell_unique = int(np.unique(np.round(shell[shell_support], 10)).size)
    mismatch_unique = int(np.unique(np.round(mismatch[mismatch_support], 10)).size)
    statistics = {
        "rows": int(len(table)),
        "sssp_supported": int(shell_support.sum()),
        "sssp_coverage": float(shell_support.mean()),
        "sssp_unique_rounded_10": shell_unique,
        "scbv_supported": int(mismatch_support.sum()),
        "scbv_coverage": float(mismatch_support.mean()),
        "scbv_unique_rounded_10": mismatch_unique,
    }
    statistics["passes"] = bool(
        statistics["sssp_coverage"] >= MINIMUM_SSSP_COVERAGE
        and shell_unique >= MINIMUM_UNIQUE
        and mismatch_unique >= MINIMUM_UNIQUE
    )
    return statistics


def _error_row(exc: Exception) -> dict[str, object]:
    row = {
        n411.FEATURE_NAMES[0]: math.nan,
        "sssp_supported": False,
        "sssp_failure": f"{type(exc).__name__}: {exc}",
        "sssp_site_count": 0,
        "sssp_edge_count": 0,
        "sssp_min_site_purity": math.nan,
        "sssp_valence_policy": None,
        n529.SCBV_FEATURE: math.nan,
        "scbv_supported": False,
        "scbv_failure": f"upstream parse failed: {type(exc).__name__}: {exc}",
        "pauling_feature_error": f"upstream parse failed: {type(exc).__name__}: {exc}",
    }
    for name in n12.RULES:
        row[f"pauling_{name}_value"] = math.nan
        row[f"pauling_{name}_decision"] = "ABSTAIN"
    row["pauling_p2_p5_decision"] = "ABSTAIN"
    return row


def _compute_payload(item):
    material_id, payload = item
    try:
        atoms = n267.n85._parse_frame(payload, strict_output=True).atoms
    except Exception as exc:
        return material_id, _error_row(exc)
    row = {}
    try:
        row.update(n411.compute_sssp_row(atoms))
    except Exception as exc:
        row.update(_error_row(exc))
    try:
        structure = n267.AseAtomsAdaptor.get_structure(atoms)
        result = n22.compute_scale_calibrated_bond_valence_features(
            structure, graph_mode="voronoi"
        )
    except Exception as exc:
        result = None
        scbv_failure = f"{type(exc).__name__}: {exc}"
    else:
        scbv_failure = result.failure_reason
    row["scbv_supported"] = bool(result is not None and result.supported)
    row["scbv_failure"] = None if row["scbv_supported"] else scbv_failure
    row[n529.SCBV_FEATURE] = (
        float(result.features[n529.SCBV_FEATURE]) if row["scbv_supported"] else math.nan
    )
    try:
        features, error = n12._classical_features(atoms)
    except Exception as exc:
        features, error = None, f"calculator failed: {type(exc).__name__}: {exc}"
    values = dict(features) if isinstance(features, dict) else {}
    row["pauling_feature_error"] = error
    decisions = []
    for name, rule in n12.RULES.items():
        value = values.get(str(rule["feature"]), np.nan)
        decision = n12._rule_decision(
            value, operator=str(rule["operator"]), threshold=float(rule["threshold"])
        )
        row[f"pauling_{name}_value"] = value
        row[f"pauling_{name}_decision"] = decision
        decisions.append(decision)
    row["pauling_p2_p5_decision"] = n12._combined_decision(decisions)
    return material_id, row


def _compute_many(payloads, workers: int):
    if workers == 1:
        return [_compute_payload(item) for item in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_compute_payload, payloads, chunksize=8))


def build_wbm_features(
    *,
    next529_dir: Path,
    next530_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    freeze, cohort = Path(next529_dir).resolve(), Path(next530_dir).resolve()
    paths = {
        "design": Path(design_path).resolve(),
        "next529_manifest": freeze / n529.MANIFEST_NAME,
        "next529_formula": freeze / n529.FORMULA_NAME,
        "next530_manifest": cohort / n530.MANIFEST_NAME,
        "next530_metadata": cohort / n530.METADATA_NAME,
        "next530_geometry": cohort / n530.GEOMETRY_NAME,
        "next411_source": Path(n411.__file__).resolve(),
        "next22_source": Path(n22.__file__).resolve(),
        "pauling_source": Path(n12.__file__).resolve(),
    }
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("NEXT531 workers differ")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT531 input is missing")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name for name in set(hashes) | set(EXPECTED_INPUT_SHA256)
            if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT531 formal input identity differs: {differing}")
    freeze_manifest = _read_json(paths["next529_manifest"])
    cohort_manifest = _read_json(paths["next530_manifest"])
    formula = _read_json(paths["next529_formula"])
    if (
        freeze_manifest.get("wbm_external_endpoint_opened") is not False
        or cohort_manifest.get("protocol") != n530.PROTOCOL
        or cohort_manifest.get("next531_label_free_features_authorized") is not True
        or cohort_manifest.get("labels_opened") is not False
        or cohort_manifest.get("wbm_summary_opened") is not False
        or cohort_manifest.get("relaxed_structures_opened") is not False
        or formula.get("sssp_threshold") != n529.SSSP_THRESHOLD
        or formula.get("scbv_threshold") != n529.EXPECTED_SCBV_THRESHOLD
        or formula.get("dft_inputs") != []
    ):
        raise ValueError("NEXT531 frozen provenance differs")
    metadata = pd.read_parquet(paths["next530_metadata"])
    required = {
        "material_id", "rk", "formula", "natoms", "partition_role", "input_role"
    }
    if (
        required - set(metadata)
        or len(metadata) != n530.SAMPLE_SIZE
        or metadata["material_id"].astype(str).duplicated().any()
        or set(metadata["partition_role"].astype(str)) != {"external_validation"}
    ):
        raise ValueError("NEXT531 metadata differs")
    metadata = metadata.sort_values("material_id", kind="mergesort").reset_index(drop=True)
    payloads = n267.n85._archive_payloads(
        paths["next530_geometry"], metadata["material_id"].astype(str).tolist()
    )
    started = time.perf_counter()
    computed = _compute_many(payloads, workers)
    computed_frame = pd.DataFrame(
        [{"material_id": material_id, **row} for material_id, row in computed]
    )
    if (
        computed_frame["material_id"].astype(str).duplicated().any()
        or set(computed_frame["material_id"].astype(str))
        != set(metadata["material_id"].astype(str))
    ):
        raise RuntimeError("NEXT531 feature material identity differs")
    table = metadata.merge(computed_frame, on="material_id", validate="one_to_one")
    statistics = label_free_gate_statistics(table)
    if statistics["passes"] is not True:
        raise RuntimeError("NEXT531 label-free feature gates failed")
    table = apply_frozen_formula(table)
    counts = {
        **statistics,
        "formula_supported": int(table["formula_supported"].sum()),
        "rejected": int(table["reject"].sum()),
        "pauling_decisions": {
            value: int(table["pauling_p2_p5_decision"].eq(value).sum())
            for value in ("KEEP", "REJECT", "ABSTAIN")
        },
    }
    catalogue = {
        "protocol": PROTOCOL,
        "features": [n411.FEATURE_NAMES[0], n529.SCBV_FEATURE],
        "formula_sha256": hashes["next529_formula"],
        "formula": formula,
        "label_free_statistics": statistics,
        "pauling_control": "frozen P2-P5 analytic control from NEXT12",
        "endpoint_fields_present": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_hash = _sha256_file(Path(__file__).resolve())
    try:
        table_path, catalogue_path = staging / TABLE_NAME, staging / CATALOGUE_NAME
        table.to_parquet(table_path, index=False)
        catalogue_path.write_bytes(_json_bytes(catalogue))
        manifest = {
            "protocol": PROTOCOL,
            "mode": "label_free_wbm_x0_features_predictions_and_pauling_controls",
            "counts": counts,
            "elapsed_seconds": time.perf_counter() - started,
            **BOUNDARY_FLAGS,
            "formula_or_threshold_modified": False,
            "inputs_sha256": {
                name: {"path": str(path), "sha256": hashes[name]}
                for name, path in paths.items()
            },
            "executed_source_sha256": {
                "src/next531_sssp_bvc_wbm_features.py": source_hash
            },
            "outputs_sha256": {
                TABLE_NAME: _sha256_file(table_path),
                CATALOGUE_NAME: _sha256_file(catalogue_path),
            },
            "next532_external_evaluation_authorized": True,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256_file(Path(__file__).resolve()) != source_hash:
            raise RuntimeError("NEXT531 source changed before publication")
        if any(_sha256_file(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT531 input changed before publication")
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next529-dir", type=Path, required=True)
    parser.add_argument("--next530-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_wbm_features(
        next529_dir=args.next529_dir, next530_dir=args.next530_dir,
        design_path=args.design, output_dir=args.output_dir, workers=args.workers,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


__all__ = [
    "BOUNDARY_FLAGS", "CATALOGUE_NAME", "MANIFEST_NAME", "PROTOCOL", "TABLE_NAME",
    "apply_frozen_formula", "build_wbm_features", "label_free_gate_statistics",
]


if __name__ == "__main__":
    main()
