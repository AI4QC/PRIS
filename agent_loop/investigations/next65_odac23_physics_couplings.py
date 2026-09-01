#!/usr/bin/env python3
"""Predefined multiplicative/ratio couplings over sealed x0-only features."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next63_odac23_metal_chemistry_features import (
    FEATURES_NAME as SOURCE_FEATURES_NAME,
    MANIFEST_NAME as SOURCE_MANIFEST_NAME,
    NEXT63_FEATURE_NAMES,
    PROTOCOL as SOURCE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next65-odac23-predefined-physics-couplings-v1"
DESIGN_SHA256 = "d867713cd22752f95cb891eb0fa15170f5efe3648b484ab9aefc1c1b5dd86324"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "65d5d7bfc7d2ffa28c1dae27beb46b529af2e02c0301ae7d19c72e911b7e37e0"
)
EPSILON = 1.0e-6
FEATURES_NAME = "next65_odac23_physics_coupling_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
INTERACTION_FEATURE_NAMES = (
    "density_metal_donor_strain",
    "density_metal_donor_distance_tail",
    "density_metal_ligand_stretch",
    "density_directional_confinement",
    "hetero_directional_confinement",
    "metal_strain_to_donor_order",
    "metal_stretch_to_global_order",
    "hinge_metal_strain",
    "hinge_metal_stretch",
    "void_metal_strain",
    "coordination_ambiguity_strain",
    "electrochemical_metal_strain",
)
NEXT65_FEATURE_NAMES = tuple(NEXT63_FEATURE_NAMES) + INTERACTION_FEATURE_NAMES
_INPUTS = (
    "atom_density",
    "metal_donor_ratio_std",
    "metal_donor_distance_q95",
    "metal_ligand_ratio_q95",
    "bond_orientation_lambda_min",
    "heteroatomic_edge_fraction",
    "donor_motif_order_strength_min",
    "metal_donor_ratio_max",
    "motif_order_strength_min",
    "degree2_bend_q95",
    "volume_per_atom",
    "donor_motif_cn_entropy_q95",
    "metal_donor_en_gap_q95",
)


@dataclass(frozen=True)
class PhysicsCouplingResult:
    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def compute_physics_couplings(row: Mapping[str, object]) -> PhysicsCouplingResult:
    try:
        values = {name: float(row[name]) for name in _INPUTS}
        if not np.isfinite(list(values.values())).all():
            raise ValueError("coupling input is non-finite")
        if (
            values["bond_orientation_lambda_min"] < 0.0
            or values["donor_motif_order_strength_min"] < 0.0
            or values["motif_order_strength_min"] < 0.0
        ):
            raise ValueError("coupling divisor is negative")
        strain = values["metal_donor_ratio_std"]
        bend = values["degree2_bend_q95"]
        features = {
            "density_metal_donor_strain": values["atom_density"] * strain,
            "density_metal_donor_distance_tail": values["atom_density"]
            * values["metal_donor_distance_q95"],
            "density_metal_ligand_stretch": values["atom_density"]
            * values["metal_ligand_ratio_q95"],
            "density_directional_confinement": values["atom_density"]
            / max(values["bond_orientation_lambda_min"], EPSILON),
            "hetero_directional_confinement": values["heteroatomic_edge_fraction"]
            / max(values["bond_orientation_lambda_min"], EPSILON),
            "metal_strain_to_donor_order": strain
            / max(values["donor_motif_order_strength_min"], EPSILON),
            "metal_stretch_to_global_order": values["metal_donor_ratio_max"]
            / max(values["motif_order_strength_min"], EPSILON),
            "hinge_metal_strain": bend * strain,
            "hinge_metal_stretch": bend * values["metal_ligand_ratio_q95"],
            "void_metal_strain": values["volume_per_atom"] * strain,
            "coordination_ambiguity_strain": values["donor_motif_cn_entropy_q95"]
            * strain,
            "electrochemical_metal_strain": values["metal_donor_en_gap_q95"]
            * strain,
        }
        if tuple(features) != INTERACTION_FEATURE_NAMES or not np.isfinite(
            list(features.values())
        ).all():
            raise ValueError("NEXT65 coupling schema differs")
        return PhysicsCouplingResult(True, None, features)
    except Exception as exc:
        return PhysicsCouplingResult(False, f"{type(exc).__name__}: {exc}", {})


def _strict_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid NEXT63 manifest") from exc
    if not isinstance(value, dict):
        raise ValueError("NEXT63 manifest must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def build_physics_coupling_batch(
    *, source_dir: Path, design_path: Path, output_dir: Path
) -> dict[str, object]:
    source_dir = Path(source_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "features": source_dir / SOURCE_FEATURES_NAME,
        "manifest": source_dir / SOURCE_MANIFEST_NAME,
        "design": Path(design_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT65 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if (
        hashes["manifest"] != EXPECTED_SOURCE_MANIFEST_SHA256
        or hashes["design"] != DESIGN_SHA256
    ):
        raise ValueError("NEXT65 frozen input hash differs")
    source_manifest = _strict_json(paths["manifest"])
    outputs = source_manifest.get("outputs_sha256")
    if (
        source_manifest.get("protocol") != SOURCE_PROTOCOL
        or source_manifest.get("labels_opened") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get(SOURCE_FEATURES_NAME) != hashes["features"]
    ):
        raise ValueError("NEXT65 x0-only provenance differs")
    source = pd.read_parquet(paths["features"])
    if source.empty or source["material_id"].duplicated().any():
        raise ValueError("NEXT65 source identity differs")
    rows = []
    failures: Counter[str] = Counter()
    for _, row in source.iterrows():
        result = compute_physics_couplings(row)
        output: dict[str, object] = {
            "physics_coupling_supported": result.supported,
            "physics_coupling_failure": result.failure_reason,
        }
        output.update(
            {
                name: float(result.features[name]) if result.supported else math.nan
                for name in INTERACTION_FEATURE_NAMES
            }
        )
        if not result.supported:
            failures[result.failure_reason or "unsupported"] += 1
        rows.append(output)
    table = pd.concat([source.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    table["next65_supported"] = (
        table["next63_supported"].astype(bool)
        & table["physics_coupling_supported"].astype(bool)
        & np.isfinite(table.loc[:, NEXT65_FEATURE_NAMES]).all(axis=1)
    )
    source_path = Path(__file__).resolve()
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "predefined_algebraic_couplings_over_sealed_x0_features",
        "input_role": "one_raw_unrelaxed_framework_x0_geometry_only",
        "labels_opened": False,
        "endpoint_columns_selected": False,
        "dft_or_energy_proxy_used_at_execution": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "epsilon": EPSILON,
        "feature_columns": list(NEXT65_FEATURE_NAMES),
        "interaction_feature_columns": list(INTERACTION_FEATURE_NAMES),
        "counts": {
            "rows": len(table),
            "physics_coupling_supported": int(table["physics_coupling_supported"].sum()),
            "next65_supported": int(table["next65_supported"].sum()),
            "failures": dict(sorted(failures.items())),
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next65_odac23_physics_couplings.py": _sha256(source_path)
        },
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        feature_path = staging / FEATURES_NAME
        table.to_parquet(feature_path, index=False)
        manifest["outputs_sha256"] = {FEATURES_NAME: _sha256(feature_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != manifest["executed_source_sha256"][
            "src/next65_odac23_physics_couplings.py"
        ]:
            raise RuntimeError("NEXT65 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT65 input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_physics_coupling_batch(
        source_dir=args.source_dir, design_path=args.design, output_dir=args.output_dir
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


__all__ = [
    "FEATURES_NAME",
    "INTERACTION_FEATURE_NAMES",
    "NEXT65_FEATURE_NAMES",
    "PROTOCOL",
    "PhysicsCouplingResult",
    "build_physics_coupling_batch",
    "compute_physics_couplings",
]


if __name__ == "__main__":
    main()
