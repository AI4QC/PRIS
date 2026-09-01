#!/usr/bin/env python3
"""Build sealed NEXT22 bond-valence features from sanitized x0 geometry."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import tempfile

import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor

from src.next11_geometry_only_frames import _load_archive_only
from src.next19_feature_build import (
    MANIFEST_NAME,
    _publish_directory_no_replace,
    _sha256,
    _strict_json,
    _validate_source_manifest,
    validate_geometry_metadata,
)
from src.next19_valence_transport import (
    build_periodic_edge_geometry,
    infer_valence_assignment,
)
from src.next22_bond_valence_equilibrium import (
    FEATURE_NAMES,
    PROTOCOL as FEATURE_PROTOCOL,
    bond_valence_features_from_periodic_geometry,
)


PROTOCOL = "2026-08-02-next22-geometry-only-feature-build-v2"
FEATURE_NAME = "next22_bond_valence_equilibrium_features.parquet"
FORBIDDEN_COLUMN_TOKENS = (
    "energy",
    "force",
    "stress",
    "relax",
    "mattersim",
    "dft",
    "endpoint",
    "label",
    "target",
)


def build_feature_batch(
    *,
    archive_path: Path,
    source_manifest_path: Path,
    metadata_path: Path,
    source_role: str,
    output_dir: Path,
    graph_mode: str = "voronoi",
) -> None:
    """Build one deterministic, no-replace SCBVE feature directory."""

    if graph_mode not in {"crystalnn", "voronoi"}:
        raise ValueError("unsupported graph mode")
    if not isinstance(source_role, str) or not source_role.strip():
        raise ValueError("source role must be a nonempty string")
    archive = Path(archive_path)
    source_manifest_candidate = Path(source_manifest_path)
    metadata_candidate = Path(metadata_path)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(str(target))
    if archive.name != "geometry_only_frames.zip" or not archive.is_file():
        raise ValueError("geometry-only archive path/name is invalid")
    source_manifest = _strict_json(source_manifest_candidate, role="source manifest")
    _validate_source_manifest(source_manifest, archive_path=archive)
    metadata = validate_geometry_metadata(pd.read_parquet(metadata_candidate))
    expected_sids = tuple(metadata["material_id"].tolist())
    sids, atoms_list = _load_archive_only(archive, expected_sids)
    if sids != list(expected_sids):
        raise ValueError("geometry archive SID order mismatch")
    if [len(atoms) for atoms in atoms_list] != metadata["natoms"].tolist():
        raise ValueError("geometry archive atom counts mismatch metadata")

    failures: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    for metadata_row, atoms in zip(
        metadata.to_dict(orient="records"), atoms_list, strict=True
    ):
        structure = AseAtomsAdaptor.get_structure(atoms)
        assignment = infer_valence_assignment(structure)
        geometry = (
            build_periodic_edge_geometry(
                structure,
                assignment.values,
                graph_mode=graph_mode,
            )
            if assignment.values is not None
            else None
        )
        result = (
            bond_valence_features_from_periodic_geometry(
                structure,
                assignment.values,
                geometry,
            )
            if assignment.values is not None and geometry is not None
            else None
        )
        supported = bool(result is not None and result.supported)
        error = (
            result.failure_reason
            if result is not None
            else assignment.failure_reason or "valence assignment is unsupported"
        )
        if not supported:
            failures[error or "unknown"] += 1
        row: dict[str, object] = {
            "material_id": metadata_row["material_id"],
            "rk": metadata_row["rk"],
            "formula": metadata_row["formula"],
            "natoms": int(metadata_row["natoms"]),
            "input_role": metadata_row["input_role"],
            "valence_supported": bool(assignment.supported),
            "valence_policy": assignment.policy,
            "valence_failure": assignment.failure_reason,
            "scbv_supported": supported,
            "scbv_failure": error,
        }
        for feature in FEATURE_NAMES:
            row[feature] = float(result.features[feature]) if supported else math.nan
        rows.append(row)
    features = pd.DataFrame(rows)
    forbidden = [
        column
        for column in features
        if any(token in column.lower() for token in FORBIDDEN_COLUMN_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"feature output crossed no-DFT contract: {forbidden}")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        feature_path = staging / FEATURE_NAME
        features.to_parquet(feature_path, index=False)
        manifest = {
            "protocol": PROTOCOL,
            "feature_protocol": FEATURE_PROTOCOL,
            "source_role": source_role,
            "input_role": "unrelaxed_x0_geometry_only",
            "graph_mode": graph_mode,
            "endpoint_fields_read": False,
            "model_or_proxy_potential_used": False,
            "coordinates_or_cell_modified": False,
            "same_composition_candidates_used": False,
            "scientific_improvement_claim": False,
            "feature_names": list(FEATURE_NAMES),
            "counts": {
                "rows": int(len(features)),
                "atoms": int(features["natoms"].sum()),
                "valence_supported": int(features["valence_supported"].sum()),
                "supported": int(features["scbv_supported"].sum()),
                "valence_policy": {
                    str(key): int(value)
                    for key, value in features["valence_policy"]
                    .fillna("unsupported")
                    .value_counts()
                    .sort_index()
                    .items()
                },
            },
            "failure_counts": dict(sorted(failures.items())),
            "inputs_sha256": {
                "geometry_only_frames.zip": _sha256(archive),
                "source_manifest": _sha256(source_manifest_candidate),
                "metadata": _sha256(metadata_candidate),
            },
            "executed_source_sha256": {
                "src/next19_valence_transport.py": _sha256(
                    Path(__file__).with_name("next19_valence_transport.py")
                ),
                "src/next22_bond_valence_equilibrium.py": _sha256(
                    Path(__file__).with_name("next22_bond_valence_equilibrium.py")
                ),
                "src/next22_feature_build.py": _sha256(Path(__file__)),
            },
            "outputs_sha256": {FEATURE_NAME: _sha256(feature_path)},
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if not pd.read_parquet(feature_path).equals(features):
            raise ValueError("feature batch prepublication validation failed")
        _publish_directory_no_replace(staging, target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--source-role", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph-mode", default="voronoi")
    args = parser.parse_args()
    build_feature_batch(
        archive_path=args.archive,
        source_manifest_path=args.source_manifest,
        metadata_path=args.metadata,
        source_role=args.source_role,
        output_dir=args.output,
        graph_mode=args.graph_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FEATURE_NAME", "MANIFEST_NAME", "build_feature_batch"]
