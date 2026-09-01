#!/usr/bin/env python3
"""Materialize frozen low analytic-Ewald-field protection features."""

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


PROTOCOL = "2026-08-08-next139-analytic-field-protection-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT139_ANALYTIC_FIELD_PROTECTION_CATALOGUE.json"
RAW_FEATURE = "aefi_field_max"
RAW_SUPPORT = "next43_analytic_field_supported"
FEATURE_NAME = "analytic_field_balance_protection"
SUPPORT_COLUMN = "analytic_field_balance_protection_supported"
CENTER = 1.060255159285863
SCALE = 1.1838398971398365
CLIP_NORMALIZED = 0.8956068821868941
FEATURE_FILES = {
    "scigen": "next139_scigen_analytic_field_protection.parquet",
    "wyformer": "next139_wyformer_analytic_field_protection.parquet",
}
EXPECTED_DESIGN_SHA256 = "36a6c5183b0959d285173044a60b8f1ad2ac413d5a99ad51d1a5c74319ae935c"
EXPECTED_INPUT_SHA256 = {
    **{key: value for key, value in n133.EXPECTED_INPUT_SHA256.items() if key != "design"},
    "design": EXPECTED_DESIGN_SHA256,
}


def materialize_analytic_field_protection(table: pd.DataFrame) -> pd.DataFrame:
    required = {"material_id", RAW_FEATURE, RAW_SUPPORT}
    if required - set(table.columns) or table["material_id"].astype(str).duplicated().any():
        raise ValueError("NEXT139 analytic-field feature schema differs")
    raw = pd.to_numeric(table[RAW_FEATURE], errors="coerce").to_numpy(float)
    declared = table[RAW_SUPPORT].eq(True).to_numpy()
    if np.any(declared & (~np.isfinite(raw) | (raw < 0.0))):
        raise ValueError("NEXT139 supported analytic field differs")
    active = declared & np.isfinite(raw) & (raw >= 0.0)
    protection = np.full(len(table), np.nan, dtype=float)
    transformed = np.log1p(raw[active])
    protection[active] = np.clip(
        np.maximum(0.0, (CENTER - transformed) / SCALE),
        0.0,
        CLIP_NORMALIZED,
    )
    return pd.DataFrame(
        {
            "material_id": table["material_id"].astype(str),
            FEATURE_NAME: protection,
            SUPPORT_COLUMN: active,
        }
    )


def build_analytic_field_protection(
    *,
    scigen_feature_dir: Path,
    wyformer_feature_dir: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    scigen = Path(scigen_feature_dir).resolve()
    wyformer = Path(wyformer_feature_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "scigen_manifest": scigen / "MANIFEST.json",
        "scigen_features": scigen / "features_discovery.parquet",
        "wyformer_manifest": wyformer / "MANIFEST.json",
        "wyformer_features": wyformer / "wyformer_x0_features_discovery.parquet",
        "design": Path(design_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT139 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            key
            for key in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(key) != EXPECTED_INPUT_SHA256.get(key)
        )
        raise ValueError(f"NEXT139 formal input identity differs: {differing}")
    for source, protocol, filename, hash_key in (
        ("scigen", n133.SCIGEN_PROTOCOL, "features_discovery.parquet", "scigen_features"),
        ("wyformer", n133.WYFORMER_PROTOCOL, "wyformer_x0_features_discovery.parquet", "wyformer_features"),
    ):
        manifest = json.loads(paths[f"{source}_manifest"].read_text())
        if (
            manifest.get("protocol") != protocol
            or manifest.get("labels_opened") is not False
            or manifest.get("endpoint_payloads_opened") is not False
            or manifest.get("dft_values_used_by_features") is not False
            or manifest.get("learned_energy_force_stress_proxy_used") is not False
            or manifest.get("outputs_sha256", {}).get(filename) != input_hashes[hash_key]
        ):
            raise ValueError("NEXT139 prior provenance differs")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    output_paths: list[Path] = []
    diagnostics: dict[str, object] = {}
    try:
        for source in ("scigen", "wyformer"):
            raw = pd.read_parquet(
                paths[f"{source}_features"],
                columns=["material_id", RAW_FEATURE, RAW_SUPPORT],
            )
            result = materialize_analytic_field_protection(raw)
            path = staging / FEATURE_FILES[source]
            result.to_parquet(path, index=False)
            output_paths.append(path)
            supported = result[SUPPORT_COLUMN].eq(True).to_numpy()
            values = pd.to_numeric(result[FEATURE_NAME], errors="coerce").to_numpy(float)
            diagnostics[source] = {
                "rows": int(len(result)),
                "supported": int(supported.sum()),
                "positive": int((supported & (values > 0.0)).sum()),
                "clipped": int((supported & np.isclose(values, CLIP_NORMALIZED)).sum()),
                "quantiles": {
                    name: float(value)
                    for name, value in zip(
                        ("q50", "q75", "q90", "q95", "q99", "max"),
                        np.quantile(values[supported], [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]),
                    )
                },
            }
        catalogue = {
            "protocol": PROTOCOL,
            "feature": {
                "name": FEATURE_NAME,
                "raw_feature": RAW_FEATURE,
                "raw_support": RAW_SUPPORT,
                "definition": "clip(max(0,(center-log1p(raw))/scale),0,clip)",
                "center": CENTER,
                "scale": SCALE,
                "clip": CLIP_NORMALIZED,
                "support_column": SUPPORT_COLUMN,
                "mechanism": "low_dimensionless_analytic_ewald_field_maximum",
            },
            "missing_policy": "TERM_OFF_KEEP_BASE",
            "diagnostics": diagnostics,
            "labels_opened": False,
            "endpoint_columns_present": False,
        }
        catalogue_path = staging / CATALOGUE_NAME
        _write_json(catalogue_path, catalogue)
        output_paths.append(catalogue_path)
        source_paths = {"src/next139_analytic_field_protection.py": Path(__file__).resolve()}
        source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
        manifest = {
            "protocol": PROTOCOL,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
            "diagnostics": diagnostics,
            "labels_opened": False,
            "endpoint_payloads_opened": False,
            "opened_validation_outputs_used": False,
            "scigen_replication_endpoint_opened": False,
            "wyformer_replication_endpoint_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_features": False,
            "learned_energy_force_stress_proxy_used": False,
            "analytic_ewald_derivative_used": True,
            "physical_relaxation_executed": False,
            "scientific_improvement_claim": False,
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT139 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT139 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scigen-feature-dir", type=Path, required=True)
    parser.add_argument("--wyformer-feature-dir", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = build_analytic_field_protection(
        scigen_feature_dir=args.scigen_feature_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        design_path=args.design_path,
        output_dir=args.output_dir,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_analytic_field_protection", "materialize_analytic_field_protection"]
