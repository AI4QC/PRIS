#!/usr/bin/env python3
"""Freeze NEXT537 PBAAA for every ODAC23 x0 role without opening endpoints."""

from __future__ import annotations

import argparse
from collections import Counter
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
from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next54_odac23_train_selection import (
    GEOMETRY_NAME as NEXT54_GEOMETRY_NAME,
    MANIFEST_NAME as NEXT54_MANIFEST_NAME,
    METADATA_NAME as NEXT54_METADATA_NAME,
    PROTOCOL as NEXT54_PROTOCOL,
)
from src.next537_periodic_bond_angle_affine_accommodation import (
    BOUNDARY_FLAGS,
    FEATURE_NAMES,
    PROTOCOL as NEXT537_PROTOCOL,
    compute_periodic_bond_angle_affine_accommodation,
)


PROTOCOL = "2026-08-13-next539-odac23-pbaaa-label-free-features-v1"
NEXT60_PROTOCOL = "2026-08-03-next60-odac23-robust-scaffold-endpoint-v1"
NEXT538_PROTOCOL = "2026-08-13-next538-pbaaa-label-blind-probe-v2"
ROLES = ("discovery", "internal_validation", "internal_replication")
FEATURE_NAME = FEATURE_NAMES[0]
MINIMUM_UNIQUE = 20
EXPECTED_ROWS = 7_815
TABLE_NAME = "next539_odac23_pbaaa_label_free_features.parquet"
CATALOGUE_NAME = "NEXT539_ODAC23_PBAAA_FEATURE_CATALOGUE.json"
MANIFEST_NAME = "MANIFEST.json"
EXPECTED_INPUT_SHA256 = {
    "design": "68bee1dea45492f1bf7349965dcd149497e91e9ed6a1aa29098ce3bc0b01ceac",
    "next54_manifest": "9ea1f0e6c04c8619dd295aa1579da15b51d8241971b3adacb716fdbf93290927",
    "next54_metadata": "d7236197f7cea953f312fdc7d0776a2380d4c9febb8f47546d3a66ced5d96c10",
    "next54_geometry": "43ede2d8e0ac562ebf79643716395a951bebc3dfb0b1ff510c1b37c394b30ec2",
    "next60_firewall": "9dbd3f78d2505ba96b33715e6409cd8524e9b909f4134af0020b933dff2f769f",
    "next537_source": "ccaedb99e9a62589a184ff40c3b9f66c6c5928c386b262e2c9cbf945a2a1a918",
    "next538_manifest": "f488f3ff89ccf4e71237dbf5993fcc3218f5954434d1de9105a9f89a00e8bc4f",
    "next538_record": "bef570cc9c2cdd1cd8fefafde9688ada4f62e467c223bf15c4bd689811d87835",
    "next538_table": "14cece935bc1c49d9ca7173e61c51b4ee5bedab769e5fe3c0419e110d6bc4d7d",
}


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"NEXT539 {path.name} must contain an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def role_gate_statistics(table: pd.DataFrame) -> dict[str, object]:
    required = {"partition_role", FEATURE_NAME, "pbaaa_supported", "pbaaa_failure"}
    if required - set(table):
        raise ValueError("NEXT539 feature table differs")
    values = pd.to_numeric(table[FEATURE_NAME], errors="coerce").to_numpy(float)
    support = table["pbaaa_supported"].fillna(False).to_numpy(bool)
    if not np.array_equal(support, np.isfinite(values)):
        raise ValueError("NEXT539 support semantics differ")
    if set(table["partition_role"].astype(str)) != set(ROLES):
        raise ValueError("NEXT539 partition roles differ")
    records = {}
    passes = True
    roles = table["partition_role"].astype(str).to_numpy()
    for role in ROLES:
        mask = roles == role
        finite = mask & support
        unique = int(np.unique(np.round(values[finite], 8)).size)
        records[role] = {
            "rows": int(mask.sum()),
            "supported": int(finite.sum()),
            "unsupported": int((mask & ~support).sum()),
            "coverage": float(finite.sum() / mask.sum()) if mask.any() else 0.0,
            "finite_unique_rounded_8": unique,
            "passes_nondegeneracy": unique >= MINIMUM_UNIQUE,
        }
        passes &= unique >= MINIMUM_UNIQUE
    return {"rows": int(len(table)), "partitions": records, "passes": bool(passes)}


def _failure_row(exc: Exception) -> dict[str, object]:
    return {
        FEATURE_NAME: math.nan,
        "pbaaa_supported": False,
        "pbaaa_failure": f"upstream parse failed: {type(exc).__name__}: {exc}",
        "pbaaa_primitive_reduced": None,
        "pbaaa_site_count": 0,
        "pbaaa_edge_constraint_count": 0,
        "pbaaa_angle_constraint_count": 0,
        "pbaaa_direct_rank": 0,
        "pbaaa_atomic_rank": 0,
        "pbaaa_mu_min": math.nan,
        "pbaaa_mu_median": math.nan,
        "pbaaa_mu_max": math.nan,
        "pbaaa_factorization_residual": math.nan,
        "pbaaa_runtime_seconds": 0.0,
    }


def _compute_payload(item: tuple[str, bytes]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    started = time.perf_counter()
    try:
        atoms = _parse_frame(payload, strict_output=True).atoms
        result = compute_periodic_bond_angle_affine_accommodation(atoms)
        eigenvalues = np.asarray(result.generalized_eigenvalues, dtype=float)
        row = {
            FEATURE_NAME: (
                float(result.features[FEATURE_NAME]) if result.supported else math.nan
            ),
            "pbaaa_supported": result.supported,
            "pbaaa_failure": result.failure_reason,
            "pbaaa_primitive_reduced": result.primitive_reduced,
            "pbaaa_site_count": result.site_count,
            "pbaaa_edge_constraint_count": result.edge_constraint_count,
            "pbaaa_angle_constraint_count": result.angle_constraint_count,
            "pbaaa_direct_rank": result.direct_rank,
            "pbaaa_atomic_rank": result.atomic_rank,
            "pbaaa_mu_min": float(eigenvalues.min()) if len(eigenvalues) else math.nan,
            "pbaaa_mu_median": (
                float(np.median(eigenvalues)) if len(eigenvalues) else math.nan
            ),
            "pbaaa_mu_max": float(eigenvalues.max()) if len(eigenvalues) else math.nan,
            "pbaaa_factorization_residual": result.factorization_residual,
            "pbaaa_runtime_seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        row = _failure_row(exc)
        row["pbaaa_runtime_seconds"] = time.perf_counter() - started
    return material_id, row


def _compute_many(payloads: list[tuple[str, bytes]], workers: int):
    if workers == 1:
        iterator = map(_compute_payload, payloads)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_compute_payload, payloads, chunksize=2)
    rows = []
    try:
        for index, result in enumerate(iterator, start=1):
            rows.append(result)
            if index % 100 == 0 or index == len(payloads):
                print(f"NEXT539 ODAC23 PBAAA: {index}/{len(payloads)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return rows


def build_odac23_pbaaa_features(
    *,
    next54_dir: Path,
    next60_firewall_path: Path,
    next538_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    next54 = Path(next54_dir).resolve()
    next538 = Path(next538_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "design": Path(design_path).resolve(),
        "next54_manifest": next54 / NEXT54_MANIFEST_NAME,
        "next54_metadata": next54 / NEXT54_METADATA_NAME,
        "next54_geometry": next54 / NEXT54_GEOMETRY_NAME,
        "next60_firewall": Path(next60_firewall_path).resolve(),
        "next537_source": Path(
            compute_periodic_bond_angle_affine_accommodation.__code__.co_filename
        ).resolve(),
        "next538_manifest": next538 / "MANIFEST.json",
        "next538_record": next538 / "NEXT538_PBAAA_LABEL_BLIND_PROBE.json",
        "next538_table": next538 / "next538_pbaaa_label_blind_probe.parquet",
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("NEXT539 workers differ")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT539 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(hashes) | set(EXPECTED_INPUT_SHA256)
            if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT539 formal input identity differs: {differing}")
    next54_manifest = _read_json(paths["next54_manifest"])
    firewall = _read_json(paths["next60_firewall"])
    probe_manifest = _read_json(paths["next538_manifest"])
    probe_record = _read_json(paths["next538_record"])
    next54_outputs = next54_manifest.get("outputs_sha256")
    probe_outputs = probe_manifest.get("outputs_sha256")
    if (
        next54_manifest.get("protocol") != NEXT54_PROTOCOL
        or next54_manifest.get("selection_frozen_before_row_labels_opened") is not True
        or not isinstance(next54_outputs, dict)
        or next54_outputs.get(NEXT54_METADATA_NAME) != hashes["next54_metadata"]
        or next54_outputs.get(NEXT54_GEOMETRY_NAME) != hashes["next54_geometry"]
        or firewall.get("protocol") != NEXT60_PROTOCOL
        or firewall.get("internal_replication_endpoint_values_summarized_or_inspected")
        is not False
        or probe_manifest.get("protocol") != NEXT538_PROTOCOL
        or probe_manifest.get("next539_full_build_authorized") is not True
        or probe_manifest.get("endpoint_or_label_paths_constructed_or_read") is not False
        or probe_record.get("next539_full_build_authorized") is not True
        or probe_record.get("endpoint_or_label_values_opened") is not False
        or not isinstance(probe_outputs, dict)
        or probe_outputs.get(paths["next538_record"].name) != hashes["next538_record"]
        or probe_outputs.get(paths["next538_table"].name) != hashes["next538_table"]
    ):
        raise ValueError("NEXT539 label-free provenance differs")

    metadata = pd.read_parquet(paths["next54_metadata"])
    if (
        len(metadata) != EXPECTED_ROWS
        or metadata["material_id"].astype(str).duplicated().any()
        or set(metadata["partition_role"].astype(str)) != set(ROLES)
    ):
        raise ValueError("NEXT539 metadata differs")
    metadata = metadata.sort_values("material_id", kind="mergesort").reset_index(drop=True)
    material_ids = metadata["material_id"].astype(str).tolist()
    with zipfile.ZipFile(paths["next54_geometry"]) as archive:
        names = archive.namelist()
        by_id = {Path(name).stem: name for name in names}
        if len(names) != EXPECTED_ROWS or set(by_id) != set(material_ids):
            raise ValueError("NEXT539 geometry inventory differs")
        payloads = [(material_id, archive.read(by_id[material_id])) for material_id in material_ids]
    started = time.perf_counter()
    computed = _compute_many(payloads, workers)
    rows = pd.DataFrame([{"material_id": material_id, **row} for material_id, row in computed])
    if rows["material_id"].astype(str).duplicated().any() or set(rows["material_id"]) != set(
        material_ids
    ):
        raise RuntimeError("NEXT539 computed identity differs")
    table = metadata.merge(rows, on="material_id", validate="one_to_one")
    statistics = role_gate_statistics(table)
    if statistics["passes"] is not True:
        raise RuntimeError("NEXT539 role nondegeneracy gate failed")
    failures = Counter(
        table.loc[~table["pbaaa_supported"], "pbaaa_failure"].astype(str).tolist()
    )
    runtime_values = table["pbaaa_runtime_seconds"].to_numpy(float)

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    try:
        table_path = staging / TABLE_NAME
        catalogue_path = staging / CATALOGUE_NAME
        table.to_parquet(table_path, index=False)
        catalogue = {
            "protocol": PROTOCOL,
            "candidate_protocol": NEXT537_PROTOCOL,
            "feature": FEATURE_NAME,
            "direction": "risk_high",
            "missing_policy": "zero_increment",
            "minimum_unique_per_role": MINIMUM_UNIQUE,
            "statistics": statistics,
            "failure_counts": dict(sorted(failures.items())),
            "runtime_seconds": {
                "median": float(np.median(runtime_values)),
                "p95": float(np.quantile(runtime_values, 0.95)),
                "maximum": float(np.max(runtime_values)),
                "wall": time.perf_counter() - started,
            },
            "endpoint_columns_present": False,
            "internal_replication_labels_opened": False,
        }
        catalogue_path.write_bytes(_json_bytes(catalogue))
        outputs = {
            table_path.name: _sha256(table_path),
            catalogue_path.name: _sha256(catalogue_path),
        }
        manifest = {
            "protocol": PROTOCOL,
            "mode": "all_role_odac23_x0_only_pbaaa_feature_freeze",
            "counts": statistics,
            "workers": workers,
            "outputs_sha256": outputs,
            "inputs_sha256": {
                name: {"path": str(path), "sha256": hashes[name]}
                for name, path in paths.items()
            },
            "executed_source_sha256": {
                "src/next539_odac23_pbaaa_label_free_features.py": source_hash,
                "src/next537_periodic_bond_angle_affine_accommodation.py": hashes[
                    "next537_source"
                ],
            },
            "endpoint_values_opened": False,
            "internal_replication_endpoint_values_opened": False,
            **BOUNDARY_FLAGS,
            "next540_two_partition_development_authorized": True,
            "next541_replication_prediction_authorized": False,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT539 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT539 input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next54-dir", type=Path, required=True)
    parser.add_argument("--next60-firewall", type=Path, required=True)
    parser.add_argument("--next538-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_odac23_pbaaa_features(
        next54_dir=args.next54_dir,
        next60_firewall_path=args.next60_firewall,
        next538_dir=args.next538_dir,
        design_path=args.design,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
