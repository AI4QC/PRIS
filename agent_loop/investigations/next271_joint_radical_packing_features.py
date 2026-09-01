#!/usr/bin/env python3
"""Materialize the frozen joint radical-packing feature bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd

import src.next267_periodic_radical_voronoi_packing as n267
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next271-joint-radical-packing-features-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT271_JOINT_RADICAL_FEATURE_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "next271_scigen_joint_radical_features.parquet",
    "wyformer": "next271_wyformer_joint_radical_features.parquet",
}
INPUT_MANIFEST_NAME = n267.MANIFEST_NAME
INPUT_CATALOGUE_NAME = n267.CATALOGUE_NAME
INPUT_FEATURE_FILES = n267.FEATURE_FILES
FEATURE_NAMES = (
    "prvj_joint_min",
    "prvj_joint_harmonic",
    "prvj_joint_geometric",
    "prvj_joint_product",
    "prvj_joint_mean",
    "prvj_joint_max",
    "prvj_joint_l1_gap",
    "prvj_volume_minus_chebyshev",
    "prvj_chebyshev_minus_volume",
    "prvj_volume_excess",
    "prvj_chebyshev_excess",
    "prvj_balance_weighted_joint",
)
VOLUME_Q_LO = 0.021517581455692707
VOLUME_Q_HI = 0.6977318301246591
CHEBYSHEV_Q_LO = 0.011985598809152042
CHEBYSHEV_Q_HI = 0.28180046821941024
VOLUME_SPAN = VOLUME_Q_HI - VOLUME_Q_LO
CHEBYSHEV_SPAN = CHEBYSHEV_Q_HI - CHEBYSHEV_Q_LO
OUTPUT_GRID = 1.0e12
EXPECTED_DESIGN_SHA256 = (
    "2a769be7b45ce26fe5da7d06569645e3cee3a1669a608db9781312f6cd331950"
)
EXPECTED_INPUT_SHA256 = {
    "next267_manifest": (
        "65dc247f58e2ea49a9956f63f1ea547a6b01254617e72a0d3a8d15c504c6dc82"
    ),
    "next267_catalogue": (
        "b974cbaac11d00535be0c5566390c0716cd7b116705cec7974d3684c644d3eaa"
    ),
    "next267_scigen": (
        "2e400676b94110fa9e64715840f26873855416a12df199181c8150df6d4fe7c0"
    ),
    "next267_wyformer": (
        "ae6a9b76e39e603541a1065aea46fdac6ad3e3ab633e9e55a210dfa35977b827"
    ),
    "design": EXPECTED_DESIGN_SHA256,
}
BOUNDARY_FLAGS = n267.BOUNDARY_FLAGS


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def joint_radical_values(*, volume_cv: float, chebyshev_cv: float) -> dict[str, float]:
    """Evaluate the frozen twelve-term joint transform for one structure."""

    volume = float(volume_cv)
    chebyshev = float(chebyshev_cv)
    if (
        not math.isfinite(volume)
        or not math.isfinite(chebyshev)
        or volume < 0.0
        or chebyshev < 0.0
    ):
        raise ValueError("NEXT271 CV inputs must be finite nonnegative values")
    x = max(0.0, (volume - VOLUME_Q_LO) / VOLUME_SPAN)
    y = max(0.0, (chebyshev - CHEBYSHEV_Q_LO) / CHEBYSHEV_SPAN)
    lower = min(x, y)
    upper = max(x, y)
    total = x + y
    raw = {
        "prvj_joint_min": lower,
        "prvj_joint_harmonic": 0.0 if total == 0.0 else 2.0 * x * y / total,
        "prvj_joint_geometric": math.sqrt(x * y),
        "prvj_joint_product": x * y,
        "prvj_joint_mean": 0.5 * total,
        "prvj_joint_max": upper,
        "prvj_joint_l1_gap": abs(x - y),
        "prvj_volume_minus_chebyshev": x - y,
        "prvj_chebyshev_minus_volume": y - x,
        "prvj_volume_excess": max(0.0, x - y),
        "prvj_chebyshev_excess": max(0.0, y - x),
        "prvj_balance_weighted_joint": 0.0 if upper == 0.0 else lower * lower / upper,
    }
    values = {name: _quantize(raw[name]) for name in FEATURE_NAMES}
    if tuple(values) != FEATURE_NAMES or not np.isfinite(list(values.values())).all():
        raise RuntimeError("NEXT271 joint feature schema or values differ")
    return values


def transform_prv_table(table: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Append frozen joint features while preserving NEXT267 support exactly."""

    if source not in FEATURE_FILES:
        raise ValueError("NEXT271 source differs")
    required = {
        "material_id",
        "prv_volume_ratio_cv",
        "prv_chebyshev_ratio_cv",
        "prv_supported",
        "prv_failure",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"NEXT271 input columns differ: {missing}")
    if table["material_id"].astype(str).duplicated().any():
        raise ValueError("NEXT271 material_id values are not unique")
    collisions = sorted((set(FEATURE_NAMES) | {"prvj_supported", "prvj_failure"}) & set(table.columns))
    if collisions:
        raise ValueError(f"NEXT271 output columns already exist: {collisions}")
    output = table.copy(deep=True)
    support = output["prv_supported"]
    if support.isna().any() or not support.map(lambda value: type(value) in (bool, np.bool_)).all():
        raise ValueError("NEXT271 support column differs")
    support_mask = support.astype(bool).to_numpy()
    for name in FEATURE_NAMES:
        output[name] = math.nan
    for index in np.flatnonzero(support_mask):
        row = output.iloc[int(index)]
        values = joint_radical_values(
            volume_cv=float(row["prv_volume_ratio_cv"]),
            chebyshev_cv=float(row["prv_chebyshev_ratio_cv"]),
        )
        for name, value in values.items():
            output.iat[int(index), output.columns.get_loc(name)] = value
    unsupported = ~support_mask
    if unsupported.any():
        block = output.loc[unsupported, list(FEATURE_NAMES)]
        if not block.isna().all().all():
            raise RuntimeError("NEXT271 unsupported rows received features")
    finite = np.isfinite(
        output.loc[support_mask, list(FEATURE_NAMES)].to_numpy(dtype=float)
    )
    if not finite.all():
        raise RuntimeError("NEXT271 supported feature values are non-finite")
    output["prvj_supported"] = support_mask
    output["prvj_failure"] = output["prv_failure"]
    if output["material_id"].astype(str).tolist() != table["material_id"].astype(str).tolist():
        raise RuntimeError("NEXT271 row identity/order differs")
    return output


def _input_paths(*, next267_dir: Path, design_path: Path) -> dict[str, Path]:
    root = Path(next267_dir).resolve()
    return {
        "next267_manifest": root / INPUT_MANIFEST_NAME,
        "next267_catalogue": root / INPUT_CATALOGUE_NAME,
        "next267_scigen": root / INPUT_FEATURE_FILES["scigen"],
        "next267_wyformer": root / INPUT_FEATURE_FILES["wyformer"],
        "design": Path(design_path).resolve(),
    }


def _counts(table: pd.DataFrame) -> dict[str, object]:
    supported = table["prvj_supported"].astype(bool)
    return {
        "rows": len(table),
        "supported": int(supported.sum()),
        "failures": int((~supported).sum()),
        "finite_feature_counts": {
            name: int(np.isfinite(pd.to_numeric(table[name], errors="coerce")).sum())
            for name in FEATURE_NAMES
        },
    }


def build_joint_radical_features(
    *,
    next267_dir: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Publish the frozen NEXT271 feature tables atomically."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = _input_paths(next267_dir=next267_dir, design_path=design_path)
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT271 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT271 formal input identity differs: {differing}")
    source_tables = {
        source: pd.read_parquet(paths[f"next267_{source}"])
        for source in FEATURE_FILES
    }
    transformed = {
        source: transform_prv_table(table, source=source)
        for source, table in source_tables.items()
    }
    counts = {source: _counts(table) for source, table in transformed.items()}
    authorized = all(
        details["supported"]
        == next(iter(details["finite_feature_counts"].values()))
        and len(set(details["finite_feature_counts"].values())) == 1
        for details in counts.values()
    )
    if not authorized:
        raise RuntimeError("NEXT271 finite supported feature certificate failed")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256_file(source_path)
    try:
        catalogue = {
            "protocol": PROTOCOL,
            "feature_names": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "base_features": ["prv_volume_ratio_cv", "prv_chebyshev_ratio_cv"],
            "normalization": {
                "volume_q_lo": VOLUME_Q_LO,
                "volume_q_hi": VOLUME_Q_HI,
                "chebyshev_q_lo": CHEBYSHEV_Q_LO,
                "chebyshev_q_hi": CHEBYSHEV_Q_HI,
                "lower_clip": 0.0,
                "upper_clip": None,
                "output_grid": OUTPUT_GRID,
            },
            "support_policy": "EXACT_NEXT267_SUPPORT_NO_IMPUTATION",
            "endpoint_or_label_input_accepted": False,
        }
        catalogue_path = staging / CATALOGUE_NAME
        _write_json(catalogue_path, catalogue)
        outputs = [catalogue_path]
        for source, table in transformed.items():
            path = staging / FEATURE_FILES[source]
            table.to_parquet(path, index=False)
            outputs.append(path)
        manifest = {
            "protocol": PROTOCOL,
            "counts": counts,
            "feature_count": len(FEATURE_NAMES),
            "feature_names": list(FEATURE_NAMES),
            "next267_support_preserved": True,
            "next272_audit_authorized": authorized,
            "validation_outputs_opened": False,
            "replication_outputs_opened": False,
            **BOUNDARY_FLAGS,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": {
                "src/next271_joint_radical_packing_features.py": source_hash,
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in outputs},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT271 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT271 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next267-dir", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = build_joint_radical_features(
        next267_dir=args.next267_dir,
        design_path=args.design_path,
        output_dir=args.output_dir,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
