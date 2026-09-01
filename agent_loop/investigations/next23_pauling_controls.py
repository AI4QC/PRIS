#!/usr/bin/env python3
"""Apply frozen Pauling 2--5 controls to the NEXT23 blind x0 cohort."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Callable, Mapping, Sequence

from ase import Atoms
import numpy as np
import pandas as pd

from src.next11_geometry_only_frames import _load_archive_only
from src.next12_pauling_controls import (
    DECISIONS,
    RULES,
    _classical_features,
    _combined_decision,
    _rule_decision,
)
from src.next13d_acsc_dft_pairs import _sha256_file, _strict_json
from src.next14_wbm_holdout import (
    GEOMETRY_NAME,
    MANIFEST_NAME,
    METADATA_NAME,
    _json_bytes,
    _publish_directory_no_replace,
)
from src.next23_wbm_holdout import PROTOCOL as HOLDOUT_PROTOCOL


PROTOCOL = "2026-08-02-next23-wbm-pauling-controls-v1"
OUTPUT_NAME = "next23_pauling_controls.parquet"
FeatureCalculator = Callable[
    [Atoms], tuple[Mapping[str, object] | None, str | None]
]


def _load_holdout(
    *, metadata_path: Path, frames_zip_path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, list[Atoms], Mapping[str, object]]:
    metadata_data = metadata_path.read_bytes()
    geometry_hash = _sha256_file(frames_zip_path)
    manifest = _strict_json(manifest_path.read_bytes(), role="NEXT23 holdout manifest")
    if (
        manifest.get("protocol") != HOLDOUT_PROTOCOL
        or manifest.get("input_role") != "unrelaxed_x0_geometry_only"
        or manifest.get("labels_opened") is not False
        or manifest.get("endpoint_artifacts_opened") is not False
        or manifest.get("relaxed_structures_opened") is not False
    ):
        raise ValueError("NEXT23 holdout isolation contract differs")
    outputs = manifest.get("outputs_sha256")
    expected = {
        METADATA_NAME: hashlib.sha256(metadata_data).hexdigest(),
        GEOMETRY_NAME: geometry_hash,
    }
    if not isinstance(outputs, Mapping) or any(
        outputs.get(name) != digest for name, digest in expected.items()
    ):
        raise ValueError("NEXT23 holdout output hash differs from manifest")
    metadata = pd.read_parquet(io.BytesIO(metadata_data))
    required = {"material_id", "rk", "formula", "natoms", "input_role"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"NEXT23 metadata lacks columns: {sorted(missing)}")
    metadata = metadata.loc[:, sorted(required)].copy()
    metadata["material_id"] = metadata["material_id"].astype(str)
    if (
        metadata["material_id"].duplicated().any()
        or not metadata["input_role"].eq("unrelaxed_x0_geometry_only").all()
    ):
        raise ValueError("NEXT23 metadata identity or role differs")
    metadata = metadata.sort_values("material_id", kind="stable", ignore_index=True)
    expected_ids = metadata["material_id"].tolist()
    loaded_ids, structures = _load_archive_only(frames_zip_path, expected_ids)
    if loaded_ids != expected_ids:
        raise ValueError("NEXT23 geometry order differs")
    if any(
        len(atoms) != int(natoms)
        for atoms, natoms in zip(structures, metadata["natoms"], strict=True)
    ):
        raise ValueError("NEXT23 atom count differs from geometry")
    return metadata, structures, manifest


def run_next23_pauling_controls(
    *,
    metadata_path: Path,
    frames_zip_path: Path,
    holdout_manifest_path: Path,
    output_dir: Path,
    feature_calculator: FeatureCalculator | None = None,
) -> dict[str, object]:
    """Apply fixed classical rules without labels, endpoints, or refitting."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "metadata": Path(metadata_path).resolve(),
        "geometry_only_frames": Path(frames_zip_path).resolve(),
        "holdout_manifest": Path(holdout_manifest_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    input_hashes = {role: _sha256_file(path) for role, path in paths.items()}
    metadata, structures, holdout_manifest = _load_holdout(
        metadata_path=paths["metadata"],
        frames_zip_path=paths["geometry_only_frames"],
        manifest_path=paths["holdout_manifest"],
    )
    calculator = _classical_features if feature_calculator is None else feature_calculator
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    for upstream, atoms in zip(
        metadata.to_dict("records"), structures, strict=True
    ):
        try:
            features, error = calculator(atoms)
        except Exception as exc:
            features, error = None, f"calculator failed: {type(exc).__name__}: {exc}"
        values = dict(features) if isinstance(features, Mapping) else {}
        row: dict[str, object] = {
            "material_id": str(upstream["material_id"]),
            "rk": str(upstream["rk"]),
            "formula": str(upstream["formula"]),
            "natoms": int(upstream["natoms"]),
            "pauling_feature_error": error,
        }
        decisions: list[str] = []
        for name, rule in RULES.items():
            value = values.get(str(rule["feature"]), np.nan)
            decision = _rule_decision(
                value,
                operator=str(rule["operator"]),
                threshold=float(rule["threshold"]),
            )
            row[f"{name}_value"] = value
            row[f"{name}_decision"] = decision
            decisions.append(decision)
        row["pauling_p2_p5_decision"] = _combined_decision(decisions)
        rows.append(row)
    elapsed = time.perf_counter() - started
    table = pd.DataFrame(rows).sort_values(
        "material_id", kind="stable", ignore_index=True
    )
    if len(table) != len(metadata) or table["material_id"].duplicated().any():
        raise RuntimeError("NEXT23 Pauling accounting differs")

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next23_pauling_controls.py": Path(__file__).resolve(),
        "src/next12_pauling_controls.py": repository_root
        / "src/next12_pauling_controls.py",
        "src/apply_rules.py": repository_root / "src/apply_rules.py",
        "src/discriminate.py": repository_root / "src/discriminate.py",
    }
    source_hashes = {
        relative: _sha256_file(path) for relative, path in source_paths.items()
    }
    counts: dict[str, object] = {
        "rows": len(table),
        "feature_error_rows": int(table["pauling_feature_error"].notna().sum()),
        "combined": {
            decision: int(table["pauling_p2_p5_decision"].eq(decision).sum())
            for decision in DECISIONS
        },
    }
    for name in RULES:
        counts[name] = {
            decision: int(table[f"{name}_decision"].eq(decision).sum())
            for decision in DECISIONS
        }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "next23_identical_blind_cohort_pauling_controls",
        "evidence_role": "label-free comparator before blind label opening",
        "labels_opened": False,
        "endpoint_artifacts_opened": False,
        "thresholds_refit": False,
        "rules": RULES,
        "counts": counts,
        "execution": {"wall_time_seconds": elapsed},
        "inputs_sha256": {
            role: {"path": str(path), "sha256": input_hashes[role]}
            for role, path in paths.items()
        },
        "executed_source_sha256": source_hashes,
        "production_protocol_eligible": bool(
            feature_calculator is None
            and holdout_manifest.get("production_protocol_eligible") is True
        ),
        "scientific_improvement_claim": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        output_path = staging / OUTPUT_NAME
        table.to_parquet(output_path, index=False)
        manifest["outputs_sha256"] = {OUTPUT_NAME: _sha256_file(output_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        for role, path in paths.items():
            if _sha256_file(path) != input_hashes[role]:
                raise RuntimeError(f"input {role} changed before publication")
        for relative, path in source_paths.items():
            if _sha256_file(path) != source_hashes[relative]:
                raise RuntimeError(f"source {relative} changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--frames-zip", required=True, type=Path)
    parser.add_argument("--holdout-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    run_next23_pauling_controls(
        metadata_path=arguments.metadata,
        frames_zip_path=arguments.frames_zip,
        holdout_manifest_path=arguments.holdout_manifest,
        output_dir=arguments.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["OUTPUT_NAME", "run_next23_pauling_controls", "main"]
