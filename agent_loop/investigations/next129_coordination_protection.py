#!/usr/bin/env python3
"""Endpoint-free bounded high-coordination protection evidence."""

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

from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next85_scigen_label_free_features import PROTOCOL as SCIGEN_PROTOCOL
from src.next94_wyformer_label_free_features import PROTOCOL as WYFORMER_PROTOCOL


PROTOCOL = "2026-08-08-next129-coordination-protection-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT129_COORDINATION_PROTECTION_CATALOGUE.json"
FEATURE_NAME = "coordination_protection"
RAW_FEATURE = "cov_coord110_mean"
SUPPORT_COLUMN = "coordination_protection_supported"
FEATURE_FILES = {
    "scigen": "scigen_discovery_coordination_protection.parquet",
    "wyformer": "wyformer_discovery_coordination_protection.parquet",
}
CENTER = 2.1671471220989416
SCALE = 0.5873264716128193
CLIP_NORMALIZED = 0.9209581129860017
EXPECTED_INPUT_SHA256 = {
    "scigen_manifest": "8dcb8118f85ee4a3acbf0905f01c2b173d58742a1e16dcd6004adbbbedcf63cc",
    "scigen_features": "7031d86e4fb6e469c674d208f680ace1dbe5e11e45f3d4b2befefd747efdde16",
    "wyformer_manifest": "fb66f7c5caade419a46b9a3fa6fef1bc5b3afa3eebeb95a4bc53baddabc0f659",
    "wyformer_features": "c515baec0fccef5bc03c7672f1d4e1aca278f5ed4d7b6f1bf7f66c734e2b87f7",
    "design": "cf12320eae4e3e0260ae9adc7ca31a43e25071d86ac436a69a35a96f4092819b",
}


def compute_coordination_protection(value: float) -> float | None:
    """Return bounded protection evidence, or None for invalid raw input."""

    raw = float(value)
    if not math.isfinite(raw) or raw < 0.0:
        return None
    normalized = max(0.0, (math.log1p(raw) - CENTER) / SCALE)
    return float(min(CLIP_NORMALIZED, normalized))


def materialize_coordination_protection(table: pd.DataFrame) -> pd.DataFrame:
    """Materialize protection and keep invalid rows inactive/fail-open."""

    if {"material_id", RAW_FEATURE} - set(table.columns) or table["material_id"].astype(str).duplicated().any():
        raise ValueError("NEXT129 source feature schema differs")
    raw = pd.to_numeric(table[RAW_FEATURE], errors="coerce").to_numpy(float)
    values = np.full(len(table), np.nan, dtype=float)
    supported = np.zeros(len(table), dtype=bool)
    for index, value in enumerate(raw):
        result = compute_coordination_protection(float(value))
        if result is not None:
            values[index] = result
            supported[index] = True
    return pd.DataFrame(
        {
            "material_id": table["material_id"].astype(str).to_numpy(),
            FEATURE_NAME: values,
            SUPPORT_COLUMN: supported,
        }
    )


def build_coordination_protection(
    *,
    scigen_feature_dir: Path,
    wyformer_feature_dir: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build the protection artifact without reading any endpoint."""

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
        raise FileNotFoundError("NEXT129 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT129 formal input identity differs")
    manifests = {
        "scigen": json.loads(paths["scigen_manifest"].read_text()),
        "wyformer": json.loads(paths["wyformer_manifest"].read_text()),
    }
    for source, expected_protocol, filename, hash_key in (
        ("scigen", SCIGEN_PROTOCOL, "features_discovery.parquet", "scigen_features"),
        ("wyformer", WYFORMER_PROTOCOL, "wyformer_x0_features_discovery.parquet", "wyformer_features"),
    ):
        manifest = manifests[source]
        if (
            manifest.get("protocol") != expected_protocol
            or manifest.get("labels_opened") is not False
            or manifest.get("endpoint_payloads_opened") is not False
            or manifest.get("dft_values_used_by_features") is not False
            or manifest.get("learned_energy_force_stress_proxy_used") is not False
            or manifest.get("physical_relaxation_executed") is not False
            or manifest.get("outputs_sha256", {}).get(filename) != input_hashes[hash_key]
        ):
            raise ValueError("NEXT129 prior provenance differs")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    output_paths: list[Path] = []
    diagnostics: dict[str, object] = {}
    try:
        for source in ("scigen", "wyformer"):
            raw = pd.read_parquet(paths[f"{source}_features"], columns=["material_id", RAW_FEATURE])
            result = materialize_coordination_protection(raw)
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
                        ("q50", "q75", "q90", "q95", "q99", "q995", "max"),
                        np.quantile(values[supported], [0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 1.0]),
                    )
                },
            }
        catalogue = {
            "protocol": PROTOCOL,
            "raw_feature": RAW_FEATURE,
            "feature_name": FEATURE_NAME,
            "support_column": SUPPORT_COLUMN,
            "definition": "clip(max(0,(log1p(raw)-center)/scale),0,clip_normalized)",
            "center": CENTER,
            "scale": SCALE,
            "clip_normalized": CLIP_NORMALIZED,
            "polarity": "protection_high",
            "missing_policy": "PROTECTION_OFF_KEEP_BASE",
            "diagnostics": diagnostics,
            "labels_opened": False,
            "endpoint_columns_present": False,
        }
        catalogue_path = staging / CATALOGUE_NAME
        _write_json(catalogue_path, catalogue)
        output_paths.append(catalogue_path)
        source_paths = {"src/next129_coordination_protection.py": Path(__file__).resolve()}
        manifest = {
            "protocol": PROTOCOL,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": {name: _sha256_file(path) for name, path in source_paths.items()},
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
            "physical_relaxation_executed": False,
            "scientific_improvement_claim": False,
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT129 input changed before publication")
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
    manifest = build_coordination_protection(
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


__all__ = [
    "CLIP_NORMALIZED",
    "FEATURE_NAME",
    "build_coordination_protection",
    "compute_coordination_protection",
    "materialize_coordination_protection",
]
