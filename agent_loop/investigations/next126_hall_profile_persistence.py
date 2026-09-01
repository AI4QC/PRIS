#!/usr/bin/env python3
"""Endpoint-free Hall-profile persistence derived from frozen MHCR gains."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd

from src.next102_cross_source_dobvr_features import (
    _read_json,
    _sha256_file,
    _write_json,
)
from src.next124_cross_source_mhcr_features import (
    CATALOGUE_NAME as NEXT124_CATALOGUE_NAME,
    FEATURE_FILES as NEXT124_FEATURE_FILES,
    MANIFEST_NAME as NEXT124_MANIFEST_NAME,
    PROTOCOL as NEXT124_PROTOCOL,
)


PROTOCOL = "2026-08-08-next126-hall-profile-persistence-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT126_HALL_PROFILE_PERSISTENCE_CATALOGUE.json"
FEATURE_NAME = "mhpp_expanded_negative_weak_contact_persistence"
FEATURE_FILES = {
    "scigen": "scigen_discovery_mhpp_features.parquet",
    "wyformer": "wyformer_discovery_mhpp_features.parquet",
}
SOURCE_GAIN_COLUMNS = tuple(
    f"mhcr_expanded_negative_deficit_gain_tau{code}"
    for code in ("05", "10", "25", "50")
)
SUPPORT_COLUMN = "mhcr_expanded_supported"
OUTPUT_SUPPORT_COLUMN = "mhpp_supported"
OUTPUT_FAILURE_COLUMN = "mhpp_failure"
NUMERICAL_ZERO_TOLERANCE = 1.0e-12
EXPECTED_INPUT_SHA256 = {
    "next124_manifest": "32e7e4a7d9c74aea3ce029c654dfc9216ab466dda39459538f178d9e16beb8bb",
    "next124_catalogue": "8a5a3ad4cf123996309169caf9f94edd022f00617c697f1d6da66584b44f81d3",
    "next124_scigen_features": "50002b41a430278788f9c097a997651c34bd9acc350cad6078122731150c5ac7",
    "next124_wyformer_features": "76bf3129268b1adb53a0cf82a3c6f0a2786176f8e4a6f3e49ea44195b89b3932",
    "design": "18defe3121f2a5e4d2c84f38960fcd3f83f86cfb50b56442de7271cc3c3cae51",
}


def compute_hall_profile_persistence(
    g05: float,
    g10: float,
    g25: float,
    g50: float,
    *,
    tolerance: float = NUMERICAL_ZERO_TOLERANCE,
) -> float:
    """Return the normalized weak-contact left-step persistence of g50."""

    values = np.asarray([g05, g10, g25, g50], dtype=float)
    if (
        not math.isfinite(tolerance)
        or tolerance <= 0.0
        or not np.isfinite(values).all()
        or np.any(values < -tolerance)
        or np.any(values > 1.0 + tolerance)
        or np.any(values[1:] + tolerance < values[:-1])
    ):
        raise ValueError("Hall profile must be finite, bounded, and monotone")
    values = np.clip(values, 0.0, 1.0)
    if values[3] <= tolerance:
        return 0.0
    persistence = float(
        (0.05 * values[0] + 0.15 * values[1] + 0.25 * values[2])
        / (0.45 * values[3])
    )
    if not math.isfinite(persistence) or persistence < -tolerance or persistence > 1.0 + tolerance:
        raise ValueError("Hall profile persistence is outside [0, 1]")
    return float(min(1.0, max(0.0, persistence)))


def materialize_hall_profile_persistence(table: pd.DataFrame) -> pd.DataFrame:
    """Materialize HPP while preserving expanded-MHCR abstention exactly."""

    required = {
        "material_id",
        SUPPORT_COLUMN,
        "mhcr_expanded_failure",
        *SOURCE_GAIN_COLUMNS,
    }
    if required - set(table.columns) or table["material_id"].astype(str).duplicated().any():
        raise ValueError("NEXT126 source feature schema differs")
    supported = table[SUPPORT_COLUMN].eq(True).to_numpy()
    values = table.loc[:, SOURCE_GAIN_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    result = np.full(len(table), np.nan, dtype=float)
    for index in np.flatnonzero(supported):
        result[index] = compute_hall_profile_persistence(*values[index])
    failure = table["mhcr_expanded_failure"].astype(object).where(~supported, None)
    return pd.DataFrame(
        {
            "material_id": table["material_id"].astype(str).to_numpy(),
            FEATURE_NAME: result,
            OUTPUT_SUPPORT_COLUMN: supported,
            OUTPUT_FAILURE_COLUMN: failure.to_numpy(),
        }
    )


def _source_diagnostics(result: pd.DataFrame, source: pd.DataFrame) -> dict[str, object]:
    supported = result[OUTPUT_SUPPORT_COLUMN].eq(True).to_numpy()
    values = pd.to_numeric(result[FEATURE_NAME], errors="coerce").to_numpy(float)
    tau50 = pd.to_numeric(source[SOURCE_GAIN_COLUMNS[-1]], errors="coerce").to_numpy(float)
    finite = supported & np.isfinite(values) & np.isfinite(tau50)
    if int(finite.sum()) != int(supported.sum()):
        raise RuntimeError("NEXT126 supported feature is nonfinite")
    quantiles = np.quantile(values[finite], [0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 1.0])
    correlation = pd.Series(values[finite]).corr(pd.Series(tau50[finite]), method="spearman")
    return {
        "rows": int(len(result)),
        "supported": int(supported.sum()),
        "nonzero": int((finite & (values > NUMERICAL_ZERO_TOLERANCE)).sum()),
        "quantiles": {
            name: float(value)
            for name, value in zip(("q50", "q75", "q90", "q95", "q99", "q995", "max"), quantiles)
        },
        "spearman_vs_tau50": None if not math.isfinite(float(correlation)) else float(correlation),
    }


def build_cross_source_hall_profile_persistence(
    *,
    next124_dir: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build NEXT126 from NEXT124 without reading outcomes or endpoints."""

    root = Path(next124_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "next124_manifest": root / NEXT124_MANIFEST_NAME,
        "next124_catalogue": root / NEXT124_CATALOGUE_NAME,
        "next124_scigen_features": root / NEXT124_FEATURE_FILES["scigen"],
        "next124_wyformer_features": root / NEXT124_FEATURE_FILES["wyformer"],
        "design": Path(design_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT126 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT126 formal input identity differs")
    prior = _read_json(paths["next124_manifest"])
    outputs = prior.get("outputs_sha256")
    if (
        prior.get("protocol") != NEXT124_PROTOCOL
        or prior.get("labels_opened") is not False
        or prior.get("endpoint_payloads_opened") is not False
        or prior.get("validation_geometry_opened") is not False
        or prior.get("replication_geometry_opened") is not False
        or prior.get("dft_values_used_by_features") is not False
        or prior.get("learned_energy_force_stress_proxy_used") is not False
        or prior.get("physical_relaxation_executed") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get(NEXT124_CATALOGUE_NAME) != input_hashes["next124_catalogue"]
        or outputs.get(NEXT124_FEATURE_FILES["scigen"]) != input_hashes["next124_scigen_features"]
        or outputs.get(NEXT124_FEATURE_FILES["wyformer"]) != input_hashes["next124_wyformer_features"]
    ):
        raise ValueError("NEXT126 prior provenance differs")

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next126_hall_profile_persistence.py": Path(__file__).resolve(),
        "src/next124_cross_source_mhcr_features.py": repository_root / "src/next124_cross_source_mhcr_features.py",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    output_paths: list[Path] = []
    diagnostics: dict[str, object] = {}
    try:
        for source in ("scigen", "wyformer"):
            source_table = pd.read_parquet(paths[f"next124_{source}_features"])
            result = materialize_hall_profile_persistence(source_table)
            output_path = staging / FEATURE_FILES[source]
            result.to_parquet(output_path, index=False)
            output_paths.append(output_path)
            diagnostics[source] = _source_diagnostics(result, source_table)
        catalogue = {
            "protocol": PROTOCOL,
            "feature_name": FEATURE_NAME,
            "definition": "(0.05*g05+0.15*g10+0.25*g25)/(0.45*g50), zero when g50<=1e-12",
            "source_gain_columns": list(SOURCE_GAIN_COLUMNS),
            "source_support_column": SUPPORT_COLUMN,
            "output_support_column": OUTPUT_SUPPORT_COLUMN,
            "risk_polarity": "high",
            "range": [0.0, 1.0],
            "missing_policy": "OPTIONAL_GUARD_OFF_KEEP_BASE",
            "endpoint_columns_present": False,
            "labels_opened": False,
            "diagnostics": diagnostics,
        }
        catalogue_path = staging / CATALOGUE_NAME
        _write_json(catalogue_path, catalogue)
        output_paths.append(catalogue_path)
        manifest = {
            "protocol": PROTOCOL,
            "mode": "endpoint_free_derived_mhcr_profile_persistence",
            "inputs_sha256": input_hashes,
            "executed_source_sha256": {name: _sha256_file(path) for name, path in source_paths.items()},
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
            "diagnostics": diagnostics,
            "labels_opened": False,
            "endpoint_payloads_opened": False,
            "validation_geometry_opened": False,
            "replication_geometry_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_features": False,
            "learned_energy_force_stress_proxy_used": False,
            "physical_relaxation_executed": False,
            "scientific_improvement_claim": False,
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT126 input changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next124-dir", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_cross_source_hall_profile_persistence(
        next124_dir=args.next124_dir,
        design_path=args.design_path,
        output_dir=args.output_dir,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FEATURE_NAME",
    "PROTOCOL",
    "build_cross_source_hall_profile_persistence",
    "compute_hall_profile_persistence",
    "materialize_hall_profile_persistence",
]
