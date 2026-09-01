#!/usr/bin/env python3
"""Freeze a fixed x0-only analytic feature catalogue on the NEXT551 HEA cohort."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

from ase import Atoms
from ase.io import read
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next43_analytic_feature_bank import (
    CANDIDATE_FEATURE_NAMES as NEXT43_FEATURE_NAMES,
    compute_analytic_feature_row,
)
from src.next546_lisi_analytic_mechanism_search import (
    PRIMITIVE_FEATURE_NAMES,
    primitive_geometry_features,
)
import src.next551_hea_initial_cohort as n551


PROTOCOL = "2026-08-13-next552-hea-analytic-feature-freeze-v1"
FEATURE_NAMES = tuple(NEXT43_FEATURE_NAMES) + tuple(PRIMITIVE_FEATURE_NAMES)
TABLE_NAME = "next552_hea_x0_analytic_features.parquet"
CATALOGUE_NAME = "NEXT552_FEATURE_CATALOGUE.json"
MANIFEST_NAME = "MANIFEST.json"
MINIMUM_PARTITION_COVERAGE = 0.95
MINIMUM_UNIQUE = 100
MAXIMUM_POINT_MASS = 0.10
MINIMUM_SEARCHABLE_DIRECTIONS = 8
EXPECTED_INPUT_SHA256 = {
    "design": n551.DESIGN_SHA256,
    "next551_manifest": "75373a82ac45d62cc8b25df26cdba578c3e53beae470ad7f920cfc7d0f7da368",
    "next551_metadata": "4e45f924e03afa027780898bbcfd54f8d7422ddfaed8ceec04a3de2943aa73e1",
    "next551_cohort": "436979eb6b6cb19f806c22cd50a4b63d71caa126365e371fc07d9b05945cc448",
    "next551_geometry": "4e09984bc1801d1f72ed84c62557da52a7efcb6b722338b70a856b15436b75fb",
    "next551_source": "ea6cfaefe1d73b86dadc29a567f8e2a98d720601a7c56e962577bce6e7dd8036",
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _midrank(values: object, *, reverse: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    count = int(finite.sum())
    if count:
        base = -values[finite] if reverse else values[finite]
        result[finite] = (rankdata(base, method="average") - 0.5) / count
    return result


def add_frozen_risk_percentiles(
    table: pd.DataFrame, feature_names: tuple[str, ...]
) -> pd.DataFrame:
    if "fid" not in table or "partition" not in table or set(feature_names) - set(table):
        raise ValueError("NEXT552 raw feature table differs")
    result = table.copy()
    for feature in feature_names:
        values = pd.to_numeric(result[feature], errors="coerce").to_numpy(float)
        result[f"{feature}__risk_high"] = _midrank(values)
        result[f"{feature}__risk_low"] = _midrank(values, reverse=True)
    return result


def _decode_geometry(payload: bytes) -> Atoms:
    try:
        atoms = read(io.StringIO(payload.decode("utf-8")), format="extxyz", index=0)
    except Exception as exc:
        raise ValueError("NEXT552 invalid x0 extxyz") from exc
    if (
        not isinstance(atoms, Atoms)
        or len(atoms) < 1
        or not np.all(atoms.pbc)
        or atoms.calc is not None
        or atoms.info
        or set(atoms.arrays) != {"numbers", "positions"}
    ):
        raise ValueError("NEXT552 x0 geometry firewall differs")
    return atoms


def _compute_payload(item: tuple[str, bytes]) -> dict[str, object]:
    fid, payload = item
    started = time.perf_counter()
    try:
        atoms = _decode_geometry(payload)
        values = compute_analytic_feature_row(atoms)
        values.update(primitive_geometry_features(atoms))
        if set(FEATURE_NAMES) - set(values):
            raise RuntimeError("NEXT552 analytic feature schema differs")
        return {
            "fid": fid,
            **values,
            "feature_runtime_seconds": time.perf_counter() - started,
            "next552_failure": None,
        }
    except Exception as exc:
        return {
            "fid": fid,
            **{name: math.nan for name in FEATURE_NAMES},
            "feature_runtime_seconds": time.perf_counter() - started,
            "next552_failure": f"{type(exc).__name__}: {exc}",
        }


def _compute_many(payloads: list[tuple[str, bytes]], workers: int) -> list[dict[str, object]]:
    if workers == 1:
        iterator = map(_compute_payload, payloads)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_compute_payload, payloads, chunksize=2)
    rows: list[dict[str, object]] = []
    try:
        for offset, row in enumerate(iterator, start=1):
            rows.append(row)
            if offset % 100 == 0 or offset == len(payloads):
                print(f"NEXT552 x0-only analytic features: {offset}/{len(payloads)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return rows


def _feature_catalogue(table: pd.DataFrame) -> dict[str, object]:
    partitions = table["partition"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    searchable = 0
    for feature in FEATURE_NAMES:
        values = pd.to_numeric(table[feature], errors="coerce").to_numpy(float)
        finite = np.isfinite(values)
        rounded = np.round(values[finite], 12)
        counts = Counter(rounded.tolist())
        coverage = {
            partition: float(np.isfinite(values[partitions == partition]).mean())
            for partition in ("development", "validation")
        }
        unique = len(counts)
        point_mass = max(counts.values(), default=len(table)) / len(table)
        accepted = bool(
            all(value >= MINIMUM_PARTITION_COVERAGE for value in coverage.values())
            and unique >= MINIMUM_UNIQUE
            and point_mass <= MAXIMUM_POINT_MASS
        )
        if accepted:
            searchable += 2
        rows.append(
            {
                "feature": feature,
                "directions": ["high", "low"],
                "partition_coverage": coverage,
                "unique_rounded_12": unique,
                "maximum_point_mass": float(point_mass),
                "searchable": accepted,
            }
        )
    return {
        "protocol": PROTOCOL,
        "raw_feature_count": len(FEATURE_NAMES),
        "direction_count": 2 * len(FEATURE_NAMES),
        "searchable_direction_count": searchable,
        "features": rows,
        "full_cohort_midrank_normalization_frozen_before_endpoints": True,
    }


def build_feature_freeze(
    *, next551_dir: Path, design_path: Path, output_dir: Path, workers: int = 8,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    upstream = Path(next551_dir).resolve()
    design_path = Path(design_path).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "design": design_path,
        "next551_manifest": upstream / n551.MANIFEST_NAME,
        "next551_metadata": upstream / n551.METADATA_NAME,
        "next551_cohort": upstream / n551.COHORT_NAME,
        "next551_geometry": upstream / n551.GEOMETRY_NAME,
        "next551_source": Path(n551.__file__).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("NEXT552 workers differ")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT552 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name for name in set(hashes) | set(EXPECTED_INPUT_SHA256)
            if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT552 formal input identity differs: {differing}")
    manifest = json.loads(paths["next551_manifest"].read_text())
    outputs = manifest.get("outputs_sha256")
    if (
        manifest.get("protocol") != n551.PROTOCOL
        or manifest.get("dft_energy_force_stress_values_opened") is not False
        or manifest.get("final_or_relaxed_structures_opened") is not False
        or manifest.get("next552_feature_freeze_authorized") is not True
        or not isinstance(outputs, dict)
        or outputs.get(n551.METADATA_NAME) != hashes["next551_metadata"]
        or outputs.get(n551.COHORT_NAME) != hashes["next551_cohort"]
        or outputs.get(n551.GEOMETRY_NAME) != hashes["next551_geometry"]
    ):
        raise ValueError("NEXT552 upstream x0 firewall differs")
    metadata = pd.read_parquet(paths["next551_metadata"])
    if len(metadata) != n551.EXPECTED_SELECTED_ROWS or metadata["fid"].duplicated().any():
        raise ValueError("NEXT552 metadata identity differs")
    wanted = set(metadata["fid"].astype(str))
    with zipfile.ZipFile(paths["next551_geometry"]) as archive:
        names = archive.namelist()
        if len(names) != len(wanted):
            raise ValueError("NEXT552 x0 archive count differs")
        payloads = []
        for name in names:
            fid = Path(name).stem
            if fid not in wanted or not name.endswith(".extxyz"):
                raise ValueError("NEXT552 x0 archive identity differs")
            payloads.append((fid, archive.read(name)))
    payloads.sort(key=lambda item: item[0])
    features = pd.DataFrame(_compute_many(payloads, workers))
    table = metadata.merge(features, on="fid", validate="one_to_one")
    table = add_frozen_risk_percentiles(table, FEATURE_NAMES)
    table = table.sort_values(["partition", "size_family", "fid"], kind="mergesort").reset_index(drop=True)
    catalogue = _feature_catalogue(table)
    failed = int(table["next552_failure"].notna().sum())
    gates = {
        "rows": len(table),
        "whole_structure_failures": failed,
        "searchable_direction_count": catalogue["searchable_direction_count"],
    }
    gates["passes"] = bool(
        len(table) == n551.EXPECTED_SELECTED_ROWS
        and failed == 0
        and catalogue["searchable_direction_count"] >= MINIMUM_SEARCHABLE_DIRECTIONS
    )
    if gates["passes"] is not True:
        raise RuntimeError(f"NEXT552 label-blind gates failed: {gates}")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    try:
        table_path = staging / TABLE_NAME
        catalogue_path = staging / CATALOGUE_NAME
        table.to_parquet(table_path, index=False)
        catalogue_path.write_bytes(_json_bytes(catalogue))
        outputs_out = {TABLE_NAME: _sha256(table_path), CATALOGUE_NAME: _sha256(catalogue_path)}
        manifest_out = {
            "protocol": PROTOCOL,
            "gates": gates,
            "inputs_sha256": {
                name: {"path": str(path), "sha256": hashes[name]}
                for name, path in paths.items()
            },
            "outputs_sha256": outputs_out,
            "executed_source_sha256": {
                "src/next552_hea_analytic_feature_freeze.py": source_hash
            },
            "input_role": "unrelaxed_x0_geometry_only",
            "endpoint_files_accepted_by_interface": False,
            "endpoint_values_opened": False,
            "final_or_relaxed_structures_opened": False,
            "dft_values_used_by_features_or_ranks": False,
            "model_or_proxy_potential_used": False,
            "coordinates_or_cell_modified": False,
            "normalization_fit_uses_endpoint": False,
            "next553_development_endpoint_opening_authorized": True,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest_out))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT552 source changed before publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next551-dir", required=True, type=Path)
    parser.add_argument("--design-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_feature_freeze(
        next551_dir=args.next551_dir,
        design_path=args.design_path,
        output_dir=args.output_dir,
        workers=args.workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["add_frozen_risk_percentiles", "build_feature_freeze"]
