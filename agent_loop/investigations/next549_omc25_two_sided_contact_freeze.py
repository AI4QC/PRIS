#!/usr/bin/env python3
"""Freeze the x0-only two-sided contact-envelope score on OMC25 rows."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
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
from src.next32_inorganic_response_features import _canonical_periodic_ratios, _resolve_radii


PROTOCOL = "2026-08-13-next549-omc25-two-sided-contact-freeze-v1"
DESIGN_SHA256 = "1c031a8b2077aa7702072c3685b196dc78a3b0f3e5113a04a180b421896ca5d7"
EXPECTED_ROWS = 3_099
MINIMUM_COVERAGE = 0.99
MINIMUM_UNIQUE = 100
MAXIMUM_POINT_MASS = 0.05
TABLE_NAME = "next549_omc25_tcse_predictions.parquet"
FORMULA_NAME = "NEXT549_TCSE_FORMULA.json"
MANIFEST_NAME = "MANIFEST.json"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def two_sided_contact_features(atoms: Atoms) -> dict[str, float]:
    """Return the two frozen raw contact quantiles from one untouched x0 frame."""

    if len(atoms) < 1 or not np.all(atoms.pbc):
        raise ValueError("NEXT549 requires a nonempty fully periodic structure")
    cell = np.asarray(atoms.cell.array, dtype=float)
    if not np.isfinite(cell).all() or abs(float(np.linalg.det(cell))) <= 1.0e-10:
        raise ValueError("NEXT549 periodic cell differs")
    radii = _resolve_radii(np.asarray(atoms.numbers, dtype=int), None)
    pairs = _canonical_periodic_ratios(atoms, radii)
    ratios = np.asarray([value for _left, _right, value in pairs], dtype=float)
    if not len(ratios) or not np.isfinite(ratios).all():
        raise ValueError("NEXT549 periodic contact population differs")
    values = {
        "contact_ratio_q10": float(np.quantile(ratios, 0.10)),
        "contact_ratio_q50": float(np.quantile(ratios, 0.50)),
    }
    if not np.isfinite(list(values.values())).all():
        raise RuntimeError("NEXT549 contact quantiles are nonfinite")
    return values


def _decode_geometry_payload(payload: bytes) -> Atoms:
    try:
        atoms = read(io.StringIO(payload.decode("utf-8")), format="extxyz", index=0)
    except Exception as exc:
        raise ValueError("NEXT549 invalid extxyz geometry payload") from exc
    if not isinstance(atoms, Atoms) or len(atoms) < 1 or not np.all(atoms.pbc):
        raise ValueError("NEXT549 geometry is not a fully periodic ASE Atoms")
    if atoms.calc is not None or set(atoms.arrays) != {"numbers", "positions"}:
        raise ValueError("NEXT549 geometry retained non-geometric fields")
    if atoms.info:
        raise ValueError("NEXT549 geometry retained non-geometric metadata")
    return atoms


def _read_geometry_payload(archive_path: Path, member: str) -> tuple[Atoms, str]:
    with zipfile.ZipFile(Path(archive_path)) as archive:
        payload = archive.read(member)
    return _decode_geometry_payload(payload), hashlib.sha256(payload).hexdigest()


def _compute_payload(item: tuple[str, bytes, str]) -> dict[str, object]:
    material_id, payload, source_shard = item
    started = time.perf_counter()
    base: dict[str, object] = {
        "material_id": material_id,
        "source_shard": source_shard,
        "x0_member_sha256": hashlib.sha256(payload).hexdigest(),
    }
    try:
        atoms = _decode_geometry_payload(payload)
        values = two_sided_contact_features(atoms)
        return {
            **base,
            "natoms": len(atoms),
            "contact_supported": True,
            "contact_failure": None,
            **values,
            "feature_runtime_seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            **base,
            "natoms": 0,
            "contact_supported": False,
            "contact_failure": f"{type(exc).__name__}: {exc}",
            "contact_ratio_q10": math.nan,
            "contact_ratio_q50": math.nan,
            "feature_runtime_seconds": time.perf_counter() - started,
        }


def _compute_many(payloads: list[tuple[str, bytes, str]], workers: int) -> list[dict[str, object]]:
    if workers == 1:
        iterator = map(_compute_payload, payloads)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_compute_payload, payloads, chunksize=4)
    rows: list[dict[str, object]] = []
    try:
        for offset, row in enumerate(iterator, start=1):
            rows.append(row)
            if offset % 100 == 0 or offset == len(payloads):
                print(f"NEXT549 x0-only contact features: {offset}/{len(payloads)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return rows


def _percentile(values: object) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    count = int(finite.sum())
    if count:
        result[finite] = (rankdata(values[finite], method="average") - 0.5) / count
    return result


def freeze_two_sided_scores(table: pd.DataFrame) -> pd.DataFrame:
    required = {"material_id", "contact_ratio_q10", "contact_ratio_q50"}
    if required - set(table):
        raise ValueError("NEXT549 raw feature table differs")
    result = table.copy()
    q10 = pd.to_numeric(result["contact_ratio_q10"], errors="coerce").to_numpy(float)
    q50 = pd.to_numeric(result["contact_ratio_q50"], errors="coerce").to_numpy(float)
    result["risk_low_q10"] = _percentile(-q10)
    result["risk_high_q50"] = _percentile(q50)
    result["tcse_risk"] = np.maximum(
        result["risk_low_q10"].to_numpy(float),
        result["risk_high_q50"].to_numpy(float),
    )
    return result


def _freeze_gates(table: pd.DataFrame, *, expected_rows: int) -> dict[str, object]:
    values: dict[str, object] = {"rows": len(table), "expected_rows": expected_rows}
    for column in ("risk_low_q10", "risk_high_q50", "tcse_risk"):
        score = pd.to_numeric(table[column], errors="coerce").to_numpy(float)
        finite = np.isfinite(score)
        rounded = np.round(score[finite], 12)
        counts = Counter(rounded.tolist())
        values[column] = {
            "supported": int(finite.sum()),
            "coverage": float(finite.mean()),
            "unique_rounded_12": len(counts),
            "maximum_point_mass": float(max(counts.values(), default=len(table)) / len(table)),
            "bounded": bool(finite.any() and np.all((score[finite] >= 0) & (score[finite] <= 1))),
        }
    tcse = values["tcse_risk"]
    values["passes"] = bool(
        len(table) == expected_rows
        and table["material_id"].nunique() == expected_rows
        and all(values[name]["coverage"] >= MINIMUM_COVERAGE for name in (
            "risk_low_q10", "risk_high_q50", "tcse_risk"
        ))
        and tcse["unique_rounded_12"] >= MINIMUM_UNIQUE
        and tcse["maximum_point_mass"] <= MAXIMUM_POINT_MASS
        and all(values[name]["bounded"] for name in (
            "risk_low_q10", "risk_high_q50", "tcse_risk"
        ))
    )
    return values


def _load_predictions(paths: list[Path], expected_rows: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    required = {"material_id", "source_shard", "next31_risk_score", "input_role"}
    for path in paths:
        frame = pd.read_parquet(path)
        if required - set(frame) or not frame["input_role"].astype(str).eq(
            "unrelaxed_x0_geometry_only"
        ).all():
            raise ValueError(f"NEXT549 label-free prediction schema differs: {path}")
        frames.append(frame[["material_id", "source_shard", "next31_risk_score"]].copy())
    result = pd.concat(frames, ignore_index=True)
    if len(result) != expected_rows or result["material_id"].duplicated().any():
        raise ValueError("NEXT549 prediction cohort identity differs")
    return result.sort_values("material_id", kind="mergesort").reset_index(drop=True)


def _payloads_from_archives(
    x0_dirs: list[Path], cohort: pd.DataFrame
) -> tuple[list[tuple[str, bytes, str]], list[dict[str, object]]]:
    wanted = set(cohort["material_id"].astype(str))
    shard_by_id = dict(zip(cohort["material_id"].astype(str), cohort["source_shard"].astype(str)))
    found: dict[str, bytes] = {}
    inputs: list[dict[str, object]] = []
    for directory in x0_dirs:
        manifest_path = directory / MANIFEST_NAME
        geometry_path = directory / "geometry_only_frames.zip"
        if not manifest_path.is_file() or not geometry_path.is_file():
            raise FileNotFoundError(f"NEXT549 x0 artifact differs: {directory}")
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("output_role") != "unrelaxed_x0_geometry_only"
            or manifest.get("labels_opened") is not False
            or manifest.get("relaxed_structures_opened") is not False
            or manifest.get("model_or_proxy_potential_used") is not False
            or manifest.get("coordinates_or_cell_modified") is not False
        ):
            raise ValueError(f"NEXT549 x0 firewall differs: {directory}")
        inputs.append(
            {
                "directory": str(directory),
                "manifest_sha256": _sha256(manifest_path),
                "geometry_sha256": _sha256(geometry_path),
            }
        )
        with zipfile.ZipFile(geometry_path) as archive:
            for name in archive.namelist():
                if not name.endswith(".extxyz"):
                    raise ValueError(f"NEXT549 unexpected x0 member: {name}")
                material_id = Path(name).stem
                if material_id not in wanted:
                    continue
                if material_id in found:
                    raise ValueError(f"NEXT549 duplicate x0 geometry: {material_id}")
                found[material_id] = archive.read(name)
    missing = sorted(wanted - set(found))
    extra = sorted(set(found) - wanted)
    if missing or extra:
        raise ValueError(
            f"NEXT549 x0 geometry identity differs: missing={missing[:5]}, extra={extra[:5]}"
        )
    payloads = [(sid, found[sid], shard_by_id[sid]) for sid in sorted(wanted)]
    return payloads, inputs


def build_prediction_freeze(
    *,
    prediction_tables: list[Path],
    x0_dirs: list[Path],
    design_path: Path,
    output_dir: Path,
    workers: int = 8,
    expected_rows: int = EXPECTED_ROWS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    target = Path(output_dir).resolve()
    design_path = Path(design_path).resolve()
    prediction_tables = [Path(path).resolve() for path in prediction_tables]
    x0_dirs = [Path(path).resolve() for path in x0_dirs]
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if not design_path.is_file() or any(not path.is_file() for path in prediction_tables):
        raise FileNotFoundError("NEXT549 input is missing")
    if not prediction_tables or not x0_dirs or type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("NEXT549 execution parameters differ")
    if require_formal_inputs and _sha256(design_path) != DESIGN_SHA256:
        raise ValueError("NEXT549 design identity differs")
    cohort = _load_predictions(prediction_tables, expected_rows)
    payloads, x0_inputs = _payloads_from_archives(x0_dirs, cohort)
    features = pd.DataFrame(_compute_many(payloads, workers))
    table = cohort.merge(features, on=["material_id", "source_shard"], validate="one_to_one")
    table = freeze_two_sided_scores(table)
    table = table.sort_values(["source_shard", "material_id"], kind="mergesort").reset_index(drop=True)
    gates = _freeze_gates(table, expected_rows=expected_rows)
    if gates["passes"] is not True:
        raise RuntimeError(f"NEXT549 label-blind gates failed: {gates}")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    try:
        table_path = staging / TABLE_NAME
        formula_path = staging / FORMULA_NAME
        table.to_parquet(table_path, index=False)
        formula_path.write_bytes(
            _json_bytes(
                {
                    "protocol": PROTOCOL,
                    "name": "two_sided_contact_envelope",
                    "short_name": "TCSE",
                    "raw_features": {
                        "local_crowding": "q10(d_ij/(r_cov_i+r_cov_j))",
                        "global_underpacking": "q50(d_ij/(r_cov_i+r_cov_j))",
                    },
                    "percentiles": {
                        "risk_low_q10": "midrank(-q10)",
                        "risk_high_q50": "midrank(q50)",
                    },
                    "formula": "TCSE=max(risk_low_q10,risk_high_q50)",
                    "coefficients_fitted_to_endpoint": False,
                    "endpoint_dependent_normalization": False,
                    "dft_values_used_by_executable_formula": False,
                }
            )
        )
        outputs = {TABLE_NAME: _sha256(table_path), FORMULA_NAME: _sha256(formula_path)}
        manifest = {
            "protocol": PROTOCOL,
            "design_sha256": _sha256(design_path),
            "gates": gates,
            "prediction_inputs": [
                {"path": str(path), "sha256": _sha256(path)} for path in prediction_tables
            ],
            "x0_inputs": x0_inputs,
            "outputs_sha256": outputs,
            "executed_source_sha256": {
                "src/next549_omc25_two_sided_contact_freeze.py": source_hash
            },
            "input_role": "unrelaxed_x0_geometry_only",
            "endpoint_files_accepted_by_interface": False,
            "endpoint_values_opened_by_next549": False,
            "dft_energy_force_stress_used": False,
            "later_or_relaxed_geometry_used": False,
            "model_or_proxy_potential_used": False,
            "coordinates_or_cell_modified": False,
            "historical_omc25_endpoint_contamination_exists": True,
            "retrospective_transfer_only": True,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash or _sha256(design_path) != DESIGN_SHA256:
            raise RuntimeError("NEXT549 source or design changed before publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-table", action="append", required=True, type=Path)
    parser.add_argument("--x0-dir", action="append", required=True, type=Path)
    parser.add_argument("--design-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args(argv)
    result = build_prediction_freeze(
        prediction_tables=args.prediction_table,
        x0_dirs=args.x0_dir,
        design_path=args.design_path,
        output_dir=args.output_dir,
        workers=args.workers,
        expected_rows=args.expected_rows,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_read_geometry_payload",
    "build_prediction_freeze",
    "freeze_two_sided_scores",
    "two_sided_contact_features",
]

