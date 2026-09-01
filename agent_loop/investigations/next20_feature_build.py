#!/usr/bin/env python3
"""Build no-replace NEXT20 features from sanitized x0 geometry archives."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Sequence

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
from src.next20_valence_rigidity import (
    FEATURE_NAMES,
    PROTOCOL as FEATURE_PROTOCOL,
    rigidity_features_from_periodic_geometry,
)


PROTOCOL = "2026-08-02-next20-geometry-only-feature-build-v1"
FEATURE_NAME = "next20_valence_rigidity_features.parquet"
GRAPH_MODES = ("voronoi",)
CHARGE_WEIGHT_EXPONENTS = (0.0, 0.5)
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


def _configuration_prefix(graph_mode: str, exponent: float) -> str:
    if exponent == 0.0:
        suffix = "q0"
    elif exponent == 0.5:
        suffix = "q05"
    else:
        raise ValueError("unsupported charge-weight exponent")
    return f"{graph_mode}_{suffix}"


def _validate_catalogue(
    graph_modes: Sequence[str], charge_weight_exponents: Sequence[float]
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    modes = tuple(str(value) for value in graph_modes)
    exponents = tuple(float(value) for value in charge_weight_exponents)
    if not modes or len(set(modes)) != len(modes):
        raise ValueError("graph modes must be nonempty and unique")
    if any(mode not in {"crystalnn", "voronoi"} for mode in modes):
        raise ValueError("unsupported graph mode")
    if not exponents or len(set(exponents)) != len(exponents):
        raise ValueError("charge-weight exponents must be nonempty and unique")
    if any(value not in {0.0, 0.5} for value in exponents):
        raise ValueError("unsupported charge-weight exponent")
    return modes, exponents


def build_feature_batch(
    *,
    archive_path: Path,
    source_manifest_path: Path,
    metadata_path: Path,
    source_role: str,
    output_dir: Path,
    graph_modes: Sequence[str] = GRAPH_MODES,
    charge_weight_exponents: Sequence[float] = CHARGE_WEIGHT_EXPONENTS,
) -> None:
    """Build one deterministic identifier-bearing SIVR feature directory."""

    modes, exponents = _validate_catalogue(graph_modes, charge_weight_exponents)
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

    prefixes = [
        _configuration_prefix(mode, exponent)
        for mode in modes
        for exponent in exponents
    ]
    failures = {prefix: Counter() for prefix in prefixes}
    rows: list[dict[str, object]] = []
    for metadata_row, atoms in zip(
        metadata.to_dict(orient="records"), atoms_list, strict=True
    ):
        row: dict[str, object] = {
            "material_id": metadata_row["material_id"],
            "rk": metadata_row["rk"],
            "formula": metadata_row["formula"],
            "natoms": int(metadata_row["natoms"]),
            "input_role": metadata_row["input_role"],
        }
        structure = AseAtomsAdaptor.get_structure(atoms)
        assignment = infer_valence_assignment(structure)
        valences = assignment.values
        row["valence_supported"] = bool(assignment.supported)
        row["valence_policy"] = assignment.policy
        row["valence_failure"] = assignment.failure_reason
        for mode in modes:
            geometry = (
                build_periodic_edge_geometry(structure, valences, graph_mode=mode)
                if valences is not None
                else None
            )
            for exponent in exponents:
                prefix = _configuration_prefix(mode, exponent)
                if geometry is None or valences is None:
                    result = None
                    error = assignment.failure_reason or "valence assignment is unsupported"
                elif not geometry.supported:
                    result = None
                    error = geometry.failure_reason or "periodic graph is unsupported"
                else:
                    result = rigidity_features_from_periodic_geometry(
                        structure,
                        valences,
                        geometry,
                        charge_weight_exponent=exponent,
                    )
                    error = result.failure_reason
                supported = bool(result is not None and result.supported)
                row[f"{prefix}__supported"] = supported
                row[f"{prefix}__failure"] = error
                if not supported:
                    failures[prefix][error or "unknown"] += 1
                for feature in FEATURE_NAMES:
                    row[f"{prefix}__{feature}"] = (
                        float(result.features[feature]) if supported else math.nan
                    )
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
        supported_counts = {
            prefix: int(features[f"{prefix}__supported"].sum())
            for prefix in prefixes
        }
        manifest = {
            "protocol": PROTOCOL,
            "feature_protocol": FEATURE_PROTOCOL,
            "source_role": source_role,
            "input_role": "unrelaxed_x0_geometry_only",
            "endpoint_fields_read": False,
            "model_or_proxy_potential_used": False,
            "coordinates_or_cell_modified": False,
            "same_composition_candidates_used": False,
            "scientific_improvement_claim": False,
            "catalogue": {
                "graph_modes": list(modes),
                "charge_weight_exponents": list(exponents),
                "feature_names": list(FEATURE_NAMES),
            },
            "counts": {
                "rows": int(len(features)),
                "atoms": int(features["natoms"].sum()),
                "valence_supported": int(features["valence_supported"].sum()),
                "valence_policy": {
                    str(key): int(value)
                    for key, value in features["valence_policy"]
                    .fillna("unsupported")
                    .value_counts()
                    .sort_index()
                    .items()
                },
                "configuration_supported": supported_counts,
            },
            "failure_counts": {
                prefix: dict(sorted(counter.items()))
                for prefix, counter in failures.items()
            },
            "inputs_sha256": {
                "geometry_only_frames.zip": _sha256(archive),
                "source_manifest": _sha256(source_manifest_candidate),
                "metadata": _sha256(metadata_candidate),
            },
            "executed_source_sha256": {
                "src/next19_valence_transport.py": _sha256(
                    Path(__file__).with_name("next19_valence_transport.py")
                ),
                "src/next20_valence_rigidity.py": _sha256(
                    Path(__file__).with_name("next20_valence_rigidity.py")
                ),
                "src/next20_feature_build.py": _sha256(Path(__file__)),
            },
            "outputs_sha256": {FEATURE_NAME: _sha256(feature_path)},
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        reloaded = pd.read_parquet(feature_path)
        if not reloaded.equals(features):
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
    args = parser.parse_args()
    build_feature_batch(
        archive_path=args.archive,
        source_manifest_path=args.source_manifest,
        metadata_path=args.metadata,
        source_role=args.source_role,
        output_dir=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FEATURE_NAME", "MANIFEST_NAME", "build_feature_batch"]
