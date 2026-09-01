#!/usr/bin/env python3
"""Freeze unchanged SSSP on all ODAC23 x0 roles without opening endpoints."""

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
import zipfile

import numpy as np
import pandas as pd

from src.next11_geometry_only_frames import _parse_frame
import src.next411_same_sign_shell_purity as n411
from src.next347_periodic_allocation_redistribution_capacity import _sha256_file


PROTOCOL = "2026-08-13-next533-odac23-sssp-label-free-features-v1"
NEXT54_PROTOCOL = "2026-08-03-next54-odac23-train-representative-selection-v1"
NEXT60_PROTOCOL = "2026-08-03-next60-odac23-robust-scaffold-endpoint-v1"
PARTITIONS = ("discovery", "internal_validation", "internal_replication")
MINIMUM_UNIQUE = 20
EXECUTABLE_INPUT_BOUNDARY = ("composition", "one raw initial fully periodic geometry")
BOUNDARY_FLAGS = {
    "endpoint_values_opened": False,
    "internal_replication_endpoint_values_opened": False,
    "relaxed_structures_opened": False,
    "dft_calculation_executed": False,
    "dft_values_used_by_features": False,
    "learned_energy_force_stress_proxy_used": False,
    "model_or_proxy_potential_used": False,
    "physical_relaxation_executed": False,
}
TABLE_NAME = "next533_odac23_sssp_label_free_features.parquet"
CATALOGUE_NAME = "NEXT533_ODAC23_SSSP_FEATURE_CATALOGUE.json"
MANIFEST_NAME = "MANIFEST.json"
EXPECTED_ROWS = 7_815
EXPECTED_INPUT_SHA256 = {
    "design": "74db90a13d3eb30e392ab3b1723f8bfb92e1a08882c3577fd8e94172161c37af",
    "next54_manifest": "9ea1f0e6c04c8619dd295aa1579da15b51d8241971b3adacb716fdbf93290927",
    "next54_metadata": "d7236197f7cea953f312fdc7d0776a2380d4c9febb8f47546d3a66ced5d96c10",
    "next54_geometry": "43ede2d8e0ac562ebf79643716395a951bebc3dfb0b1ff510c1b37c394b30ec2",
    "next60_firewall": "9dbd3f78d2505ba96b33715e6409cd8524e9b909f4134af0020b933dff2f769f",
    "next411_source": "172543534328a387b7d2b12ffd6cad919793ace56ec1124dd6e228f96d8cc9a4",
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def role_gate_statistics(table: pd.DataFrame) -> dict[str, object]:
    required = {
        "partition_role", n411.FEATURE_NAMES[0], "sssp_supported", "sssp_failure"
    }
    if required - set(table):
        raise ValueError("NEXT533 feature table differs")
    values = pd.to_numeric(table[n411.FEATURE_NAMES[0]], errors="coerce").to_numpy(float)
    support = table["sssp_supported"].fillna(False).to_numpy(bool)
    if not np.array_equal(support, support & np.isfinite(values)):
        raise ValueError("NEXT533 support semantics differ")
    if set(table["partition_role"].astype(str)) != set(PARTITIONS):
        raise ValueError("NEXT533 partition roles differ")
    records = {}
    passes = True
    roles = table["partition_role"].astype(str).to_numpy()
    for role in PARTITIONS:
        mask = roles == role
        finite = mask & support
        unique = int(np.unique(np.round(values[finite], 10)).size)
        records[role] = {
            "rows": int(mask.sum()),
            "supported": int(finite.sum()),
            "unsupported": int((mask & ~support).sum()),
            "coverage": float(finite.sum() / mask.sum()) if mask.any() else 0.0,
            "finite_unique_rounded_10": unique,
            "passes_nondegeneracy": unique >= MINIMUM_UNIQUE,
        }
        passes &= unique >= MINIMUM_UNIQUE
    return {"rows": int(len(table)), "partitions": records, "passes": bool(passes)}


def _parse_error_row(exc: Exception) -> dict[str, object]:
    return {
        n411.FEATURE_NAMES[0]: math.nan,
        "sssp_supported": False,
        "sssp_failure": f"upstream parse failed: {type(exc).__name__}: {exc}",
        "sssp_site_count": 0,
        "sssp_edge_count": 0,
        "sssp_min_site_purity": math.nan,
        "sssp_valence_policy": None,
    }


def _compute_payload(item: tuple[str, bytes]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        atoms = _parse_frame(payload, strict_output=True).atoms
        row = n411.compute_sssp_row(atoms)
    except Exception as exc:
        row = _parse_error_row(exc)
    return material_id, row


def _compute_many(payloads: list[tuple[str, bytes]], workers: int):
    if workers == 1:
        return [_compute_payload(item) for item in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_compute_payload, payloads, chunksize=4))


def build_odac23_sssp_features(
    *, next54_dir: Path, next60_firewall_path: Path, design_path: Path,
    output_dir: Path, workers: int = 16, require_formal_inputs: bool = True,
) -> dict[str, object]:
    cohort = Path(next54_dir).resolve()
    paths = {
        "design": Path(design_path).resolve(),
        "next54_manifest": cohort / "MANIFEST.json",
        "next54_metadata": cohort / "next54_odac23_selected_metadata.parquet",
        "next54_geometry": cohort / "next54_odac23_selected_x0.zip",
        "next60_firewall": Path(next60_firewall_path).resolve(),
        "next411_source": Path(n411.__file__).resolve(),
    }
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("NEXT533 workers differ")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT533 input is missing")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name for name in set(hashes) | set(EXPECTED_INPUT_SHA256)
            if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT533 formal input identity differs: {differing}")
    cohort_manifest = _read_json(paths["next54_manifest"])
    firewall = _read_json(paths["next60_firewall"])
    outputs = cohort_manifest.get("outputs_sha256")
    if (
        cohort_manifest.get("protocol") != NEXT54_PROTOCOL
        or cohort_manifest.get("selection_frozen_before_row_labels_opened") is not True
        or cohort_manifest.get("law_execution_dft_values_read") is not False
        or cohort_manifest.get("law_execution_relaxed_geometry_read") is not False
        or cohort_manifest.get("model_or_proxy_potential_used") is not False
        or cohort_manifest.get("physical_relaxation_executed") is not False
        or not isinstance(outputs, dict)
        or outputs.get(paths["next54_metadata"].name) != hashes["next54_metadata"]
        or outputs.get(paths["next54_geometry"].name) != hashes["next54_geometry"]
        or firewall.get("protocol") != NEXT60_PROTOCOL
        or firewall.get("internal_replication_endpoint_values_summarized_or_inspected")
        is not False
    ):
        raise ValueError("NEXT533 frozen provenance differs")
    metadata = pd.read_parquet(paths["next54_metadata"])
    required = {"material_id", "framework_name", "natoms", "partition_role", "input_role"}
    if (
        required - set(metadata) or len(metadata) != EXPECTED_ROWS
        or metadata["material_id"].astype(str).duplicated().any()
        or set(metadata["partition_role"].astype(str)) != set(PARTITIONS)
    ):
        raise ValueError("NEXT533 cohort metadata differs")
    metadata = metadata.sort_values("material_id", kind="mergesort").reset_index(drop=True)
    expected_ids = metadata["material_id"].astype(str).tolist()
    with zipfile.ZipFile(paths["next54_geometry"]) as archive:
        names = archive.namelist()
        if (
            len(names) != EXPECTED_ROWS or len(names) != len(set(names))
            or any(Path(name).suffix != ".extxyz" or Path(name).name != name for name in names)
        ):
            raise ValueError("NEXT533 geometry inventory differs")
        by_id = {Path(name).stem: name for name in names}
        if set(by_id) != set(expected_ids):
            raise ValueError("NEXT533 geometry identity differs")
        payloads = [(material_id, archive.read(by_id[material_id])) for material_id in expected_ids]
    started = time.perf_counter()
    computed = _compute_many(payloads, workers)
    rows = pd.DataFrame([{"material_id": material_id, **row} for material_id, row in computed])
    if rows["material_id"].astype(str).duplicated().any() or set(rows["material_id"]) != set(expected_ids):
        raise RuntimeError("NEXT533 computed identity differs")
    table = metadata.merge(rows, on="material_id", validate="one_to_one")
    statistics = role_gate_statistics(table)
    if statistics["passes"] is not True:
        raise RuntimeError("NEXT533 role nondegeneracy gate failed")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        table.to_parquet(staging / TABLE_NAME, index=False)
        catalogue = {
            "protocol": PROTOCOL,
            "feature": n411.FEATURE_NAMES[0],
            "direction_from_prior_external_sources": "lower_is_riskier",
            "optional_increment_missing_policy": "zero_increment",
            "minimum_unique_per_role": MINIMUM_UNIQUE,
            "statistics": statistics,
            "endpoint_columns_present": False,
            "internal_replication_labels_opened": False,
        }
        (staging / CATALOGUE_NAME).write_bytes(_json_bytes(catalogue))
        outputs_hash = {
            name: _sha256_file(staging / name) for name in (TABLE_NAME, CATALOGUE_NAME)
        }
        manifest = {
            "protocol": PROTOCOL,
            "mode": "all_role_odac23_x0_only_sssp_feature_freeze",
            "counts": statistics,
            "workers": workers,
            "elapsed_seconds": time.perf_counter() - started,
            "inputs_sha256": {
                name: {"path": str(path), "sha256": hashes[name]}
                for name, path in paths.items()
            },
            "executed_source_sha256": {
                "src/next533_odac23_sssp_label_free_features.py": _sha256_file(Path(__file__).resolve())
            },
            "outputs_sha256": outputs_hash,
            **BOUNDARY_FLAGS,
            "next534_two_partition_development_authorized": True,
            "next535_replication_prediction_authorized": False,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        shutil.move(str(staging), str(target))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next54-dir", type=Path, required=True)
    parser.add_argument("--next60-firewall", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = build_odac23_sssp_features(
        next54_dir=args.next54_dir,
        next60_firewall_path=args.next60_firewall,
        design_path=args.design,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
