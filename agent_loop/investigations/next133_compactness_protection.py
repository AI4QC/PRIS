#!/usr/bin/env python3
"""Endpoint-free bounded compactness protection features."""

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


PROTOCOL = "2026-08-08-next133-compactness-protection-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT133_COMPACTNESS_PROTECTION_CATALOGUE.json"
VOLUME_RAW_FEATURE = "geom_volume_pa"
PACKING_RAW_FEATURE = "geom_covalent_packing"
VOLUME_FEATURE = "low_volume_protection"
PACKING_FEATURE = "covalent_packing_protection"
VOLUME_SUPPORT = "low_volume_protection_supported"
PACKING_SUPPORT = "covalent_packing_protection_supported"
VOLUME_CENTER = 3.0858220121448285
VOLUME_SCALE = 0.6305067898025083
VOLUME_CLIP = 1.5310711399624055
PACKING_CENTER = 0.5102962511091282
PACKING_SCALE = 0.15390578713507463
PACKING_CLIP = 1.9773347262377292
FEATURE_FILES = {
    "scigen": "scigen_discovery_compactness_protection.parquet",
    "wyformer": "wyformer_discovery_compactness_protection.parquet",
}
EXPECTED_INPUT_SHA256 = {
    "scigen_manifest": "8dcb8118f85ee4a3acbf0905f01c2b173d58742a1e16dcd6004adbbbedcf63cc",
    "scigen_features": "7031d86e4fb6e469c674d208f680ace1dbe5e11e45f3d4b2befefd747efdde16",
    "wyformer_manifest": "fb66f7c5caade419a46b9a3fa6fef1bc5b3afa3eebeb95a4bc53baddabc0f659",
    "wyformer_features": "c515baec0fccef5bc03c7672f1d4e1aca278f5ed4d7b6f1bf7f66c734e2b87f7",
    "design": "408e848ca03507708af4b34a49793290654e351e3edd7d2381cc9c5dc0763072",
}


def compute_low_volume_protection(value: float) -> float | None:
    raw = float(value)
    if not math.isfinite(raw) or raw <= 0.0:
        return None
    normalized = max(0.0, (VOLUME_CENTER - math.log(raw)) / VOLUME_SCALE)
    return float(min(VOLUME_CLIP, normalized))


def compute_covalent_packing_protection(value: float) -> float | None:
    raw = float(value)
    if not math.isfinite(raw) or raw < 0.0:
        return None
    normalized = max(0.0, (math.log1p(raw) - PACKING_CENTER) / PACKING_SCALE)
    return float(min(PACKING_CLIP, normalized))


def materialize_compactness_protection(table: pd.DataFrame) -> pd.DataFrame:
    required = {"material_id", VOLUME_RAW_FEATURE, PACKING_RAW_FEATURE}
    if required - set(table.columns) or table["material_id"].astype(str).duplicated().any():
        raise ValueError("NEXT133 source feature schema differs")
    result: dict[str, object] = {"material_id": table["material_id"].astype(str).to_numpy()}
    for raw_name, feature_name, support_name, function in (
        (VOLUME_RAW_FEATURE, VOLUME_FEATURE, VOLUME_SUPPORT, compute_low_volume_protection),
        (PACKING_RAW_FEATURE, PACKING_FEATURE, PACKING_SUPPORT, compute_covalent_packing_protection),
    ):
        raw = pd.to_numeric(table[raw_name], errors="coerce").to_numpy(float)
        values = np.full(len(table), np.nan, dtype=float)
        support = np.zeros(len(table), dtype=bool)
        for index, value in enumerate(raw):
            computed = function(float(value))
            if computed is not None:
                values[index] = computed
                support[index] = True
        result[feature_name] = values
        result[support_name] = support
    return pd.DataFrame(result)


def build_compactness_protection(
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
        raise FileNotFoundError("NEXT133 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT133 formal input identity differs")
    for source, protocol, filename, hash_key in (
        ("scigen", SCIGEN_PROTOCOL, "features_discovery.parquet", "scigen_features"),
        ("wyformer", WYFORMER_PROTOCOL, "wyformer_x0_features_discovery.parquet", "wyformer_features"),
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
            raise ValueError("NEXT133 prior provenance differs")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    output_paths: list[Path] = []
    diagnostics: dict[str, object] = {}
    try:
        for source in ("scigen", "wyformer"):
            raw = pd.read_parquet(
                paths[f"{source}_features"],
                columns=["material_id", VOLUME_RAW_FEATURE, PACKING_RAW_FEATURE],
            )
            result = materialize_compactness_protection(raw)
            path = staging / FEATURE_FILES[source]
            result.to_parquet(path, index=False)
            output_paths.append(path)
            diagnostics[source] = {}
            for feature, support_name, clip in (
                (VOLUME_FEATURE, VOLUME_SUPPORT, VOLUME_CLIP),
                (PACKING_FEATURE, PACKING_SUPPORT, PACKING_CLIP),
            ):
                supported = result[support_name].eq(True).to_numpy()
                values = pd.to_numeric(result[feature], errors="coerce").to_numpy(float)
                diagnostics[source][feature] = {
                    "rows": int(len(result)),
                    "supported": int(supported.sum()),
                    "positive": int((supported & (values > 0.0)).sum()),
                    "clipped": int((supported & np.isclose(values, clip)).sum()),
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
            "features": {
                VOLUME_FEATURE: {
                    "raw_feature": VOLUME_RAW_FEATURE,
                    "definition": "clip(max(0,(center-log(raw))/scale),0,clip)",
                    "center": VOLUME_CENTER,
                    "scale": VOLUME_SCALE,
                    "clip": VOLUME_CLIP,
                    "support_column": VOLUME_SUPPORT,
                },
                PACKING_FEATURE: {
                    "raw_feature": PACKING_RAW_FEATURE,
                    "definition": "clip(max(0,(log1p(raw)-center)/scale),0,clip)",
                    "center": PACKING_CENTER,
                    "scale": PACKING_SCALE,
                    "clip": PACKING_CLIP,
                    "support_column": PACKING_SUPPORT,
                },
            },
            "missing_policy": "TERM_OFF_KEEP_BASE",
            "diagnostics": diagnostics,
            "labels_opened": False,
            "endpoint_columns_present": False,
        }
        catalogue_path = staging / CATALOGUE_NAME
        _write_json(catalogue_path, catalogue)
        output_paths.append(catalogue_path)
        source_paths = {"src/next133_compactness_protection.py": Path(__file__).resolve()}
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
            raise RuntimeError("NEXT133 input changed before publication")
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
    manifest = build_compactness_protection(
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
