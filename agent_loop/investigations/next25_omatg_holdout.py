#!/usr/bin/env python3
"""Seal OMatG's complete NEXT25 output as canonical generated x0 geometry."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from ase.io import read
import numpy as np
import pandas as pd

from src.next11_geometry_only_frames import _ParsedFrame, _write_deterministic_archive
from src.next19_feature_build import _publish_directory_no_replace
from src.next25_omatg_compositions import (
    COHORT_NAME as SOURCE_COHORT_NAME,
    PROTOCOL as COMPOSITION_PROTOCOL,
)
from src.next25_omatg_run import GENERATED_NAME, PROTOCOL as GENERATION_PROTOCOL


PROTOCOL = "2026-08-03-next25-omatg-generated-x0-sanitize-v1"
METADATA_NAME = "holdout_metadata.parquet"
GEOMETRY_NAME = "geometry_only_frames.zip"
MANIFEST_NAME = "MANIFEST.json"
# Filled only after the formal generation run is sealed.  A blank value makes
# accidental formal publication impossible during development and tests.
FORMAL_GENERATION_MANIFEST_SHA256 = (
    "892f6b04de445fcd1547013ac5e6d99a2b5374fdd9c8056070a1ee7aa5788263"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _strict_json(path: Path, *, role: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be an object")
    return value


def _hash_record(path: Path, digest: str) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": digest}


def _validate_manifests(
    *,
    composition_manifest: Mapping[str, object],
    generation_manifest: Mapping[str, object],
    input_hashes: Mapping[str, str],
    row_count: int,
    atom_count: int,
) -> None:
    if (
        composition_manifest.get("protocol") != COMPOSITION_PROTOCOL
        or composition_manifest.get("input_role") != "composition_only"
        or composition_manifest.get("reference_geometry_fields_accessed") is not False
        or composition_manifest.get("property_label_fields_accessed") is not False
        or composition_manifest.get("labels_opened") is not False
    ):
        raise ValueError("composition manifest crossed the label-free boundary")
    composition_outputs = composition_manifest.get("outputs_sha256")
    composition_counts = composition_manifest.get("counts")
    if (
        not isinstance(composition_outputs, Mapping)
        or composition_outputs.get(SOURCE_COHORT_NAME)
        != input_hashes["composition_cohort"]
        or not isinstance(composition_counts, Mapping)
        or composition_counts.get("selected_rows") != row_count
        or composition_counts.get("selected_atoms") != atom_count
    ):
        raise ValueError("composition manifest hashes or counts differ")

    boundary = (
        generation_manifest.get("protocol") == GENERATION_PROTOCOL
        and generation_manifest.get("input_role") == "composition_only"
        and generation_manifest.get("output_role") == "raw_unrelaxed_generator_x0"
        and generation_manifest.get("all_generator_outputs_retained") is True
        and generation_manifest.get("post_generation_validity_filter_used") is False
        and generation_manifest.get("reference_geometry_fields_accessed") is False
        and generation_manifest.get("property_label_fields_accessed") is False
        and generation_manifest.get("dft_or_relaxed_structures_accessed") is False
        and generation_manifest.get("energy_or_force_model_used") is False
        and generation_manifest.get("physical_relaxation_used") is False
        and generation_manifest.get("runtime_config_contains_reference_paths") is False
    )
    if not boundary:
        raise ValueError("generation manifest crossed the label-free boundary")
    generation_counts = generation_manifest.get("counts")
    generation_outputs = generation_manifest.get("outputs_sha256")
    generation_inputs = generation_manifest.get("inputs_sha256")
    if (
        not isinstance(generation_counts, Mapping)
        or generation_counts.get("composition_rows") != row_count
        or generation_counts.get("generated_frames") != row_count
        or not isinstance(generation_outputs, Mapping)
        or generation_outputs.get(GENERATED_NAME) != input_hashes["generated_xyz"]
        or not isinstance(generation_inputs, Mapping)
    ):
        raise ValueError("generation manifest output hash or counts differ")
    for role, manifest_role in (
        ("composition_cohort", "composition_cohort"),
        ("composition_manifest", "composition_manifest"),
    ):
        record = generation_inputs.get(manifest_role)
        if not isinstance(record, Mapping) or record.get("sha256") != input_hashes[role]:
            raise ValueError(f"generation manifest {manifest_role} hash differs")


def _verify_unchanged(paths: Mapping[str, Path], hashes: Mapping[str, str]) -> None:
    for role, path in paths.items():
        if _sha256(path) != hashes[role]:
            raise ValueError(f"{role} changed before publication")


def freeze_omatg_x0(
    *,
    composition_cohort_path: Path,
    composition_manifest_path: Path,
    generated_xyz_path: Path,
    generation_manifest_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Publish every OMatG output frame without validity-based selection."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "composition_cohort": Path(composition_cohort_path).resolve(),
        "composition_manifest": Path(composition_manifest_path).resolve(),
        "generated_xyz": Path(generated_xyz_path).resolve(),
        "generation_manifest": Path(generation_manifest_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    input_hashes = {role: _sha256(path) for role, path in paths.items()}
    formal_identity = (
        bool(FORMAL_GENERATION_MANIFEST_SHA256)
        and input_hashes["generation_manifest"]
        == FORMAL_GENERATION_MANIFEST_SHA256
    )
    if require_formal_inputs and not formal_identity:
        raise ValueError("formal NEXT25 generation manifest identity differs")
    try:
        cohort = pd.read_parquet(
            paths["composition_cohort"],
            columns=[
                "material_id",
                "formula",
                "atomic_numbers_json",
                "natoms",
                "selection_rank",
                "input_role",
            ],
        )
    except Exception as exc:
        raise ValueError("invalid composition cohort") from exc
    if (
        cohort.empty
        or cohort["material_id"].isna().any()
        or cohort["material_id"].duplicated().any()
        or cohort["material_id"].tolist() != sorted(cohort["material_id"].tolist())
        or cohort["selection_rank"].tolist() != list(range(len(cohort)))
        or not cohort["input_role"].eq("composition_only").all()
    ):
        raise ValueError("composition cohort identity or order differs")
    numeric_natoms = pd.to_numeric(cohort["natoms"], errors="raise")
    if not numeric_natoms.map(lambda value: float(value).is_integer() and value > 0).all():
        raise ValueError("composition atom counts must be positive integers")
    cohort["natoms"] = numeric_natoms.astype(int)
    composition_manifest = _strict_json(
        paths["composition_manifest"], role="composition manifest"
    )
    generation_manifest = _strict_json(
        paths["generation_manifest"], role="generation manifest"
    )
    _validate_manifests(
        composition_manifest=composition_manifest,
        generation_manifest=generation_manifest,
        input_hashes=input_hashes,
        row_count=len(cohort),
        atom_count=int(cohort["natoms"].sum()),
    )

    try:
        raw_frames = read(paths["generated_xyz"], index=":", format="extxyz")
    except Exception as exc:
        raise ValueError("could not read generated extended XYZ") from exc
    if len(raw_frames) != len(cohort):
        raise ValueError("generated frame count differs from composition cohort")
    # OMatG iterates the LMDB cursor directly. LMDB orders the decimal ASCII
    # keys bytewise (0, 1, 10, 100, ...), not numerically. Validate that exact
    # generator order, then key frames by the already frozen material IDs so
    # the canonical archive returns to ascending selection rank.
    generator_rows = sorted(
        cohort.to_dict("records"),
        key=lambda row: str(int(row["selection_rank"])).encode("ascii"),
    )
    frames: dict[str, _ParsedFrame] = {}
    for row, atoms in zip(generator_rows, raw_frames, strict=True):
        try:
            expected_numbers = sorted(json.loads(str(row["atomic_numbers_json"])))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid composition atomic_numbers_json") from exc
        if sorted(atoms.get_atomic_numbers().tolist()) != expected_numbers:
            raise ValueError(f"generated composition differs for {row['material_id']}")
        if len(atoms) != int(row["natoms"]):
            raise ValueError(f"generated atom count differs for {row['material_id']}")
        if (
            atoms.calc is not None
            or bool(atoms.info)
            or set(atoms.arrays) != {"numbers", "positions"}
        ):
            raise ValueError("generated frame contains non-geometry metadata")
        if (
            not np.all(atoms.pbc)
            or not np.all(np.isfinite(atoms.positions))
            or not np.all(np.isfinite(atoms.cell.array))
        ):
            raise ValueError("generated frame contains invalid periodic geometry")
        frames[str(row["material_id"])] = _ParsedFrame(
            atoms=atoms.copy(), dropped_comment_fields=(), dropped_atom_properties=()
        )

    metadata = pd.DataFrame(
        {
            "material_id": cohort["material_id"].astype(str),
            "rk": "omatg_mp20_csp_linear_ode",
            "formula": cohort["formula"].astype(str),
            "natoms": cohort["natoms"].astype(int),
            "input_role": "unrelaxed_x0_geometry_only",
        }
    )
    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next25_omatg_holdout.py": Path(__file__).resolve(),
        "src/next11_geometry_only_frames.py": repository_root
        / "src/next11_geometry_only_frames.py",
    }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "generated_source_label_free_transport_cohort",
        "source_protocol": GENERATION_PROTOCOL,
        "source_generator": "omatg_mp20_csp_linear_ode",
        "input_role": "unrelaxed_x0_geometry_only",
        "all_generator_outputs_retained": True,
        "post_generation_validity_filter_used": False,
        "labels_opened": False,
        "endpoint_artifacts_opened": False,
        "relaxed_structures_opened": False,
        "model_or_proxy_potential_used": False,
        "coordinates_or_cell_modified": False,
        "same_composition_candidates_used": False,
        "generator_frame_order": "ascending_lmdb_decimal_byte_key",
        "canonical_output_order": "ascending_selection_rank",
        "counts": {
            "rows": len(metadata),
            "frames": len(frames),
            "atoms": int(metadata["natoms"].sum()),
        },
        "inputs_sha256": {
            role: _hash_record(paths[role], input_hashes[role]) for role in paths
        },
        "executed_source_sha256": {
            relative: _sha256(path) for relative, path in source_paths.items()
        },
        "production_protocol_eligible": bool(formal_identity),
        "scientific_improvement_claim": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        metadata_path = staging / METADATA_NAME
        geometry_path = staging / GEOMETRY_NAME
        metadata.to_parquet(metadata_path, index=False)
        _write_deterministic_archive(geometry_path, frames)
        manifest["outputs_sha256"] = {
            METADATA_NAME: _sha256(metadata_path),
            GEOMETRY_NAME: _sha256(geometry_path),
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        _verify_unchanged(paths, input_hashes)
        for relative, path in source_paths.items():
            if _sha256(path) != manifest["executed_source_sha256"][relative]:
                raise ValueError(f"executed source changed before publication: {relative}")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-cohort", type=Path, required=True)
    parser.add_argument("--composition-manifest", type=Path, required=True)
    parser.add_argument("--generated-xyz", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = freeze_omatg_x0(
        composition_cohort_path=args.composition_cohort,
        composition_manifest_path=args.composition_manifest,
        generated_xyz_path=args.generated_xyz,
        generation_manifest_path=args.generation_manifest,
        output_dir=args.output_dir,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
