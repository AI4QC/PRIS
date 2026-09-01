#!/usr/bin/env python3
"""Freeze a label-free WBM cohort disjoint from exposed development IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence
import zipfile

import pandas as pd

from src.next14_wbm_holdout import (
    FROZEN_FORMAL_INPUT_SHA256,
    GEOMETRY_NAME,
    MANIFEST_NAME,
    METADATA_NAME,
    _archive_member,
    _atoms_from_text,
    _canonical_frame,
    _json_bytes,
    _publish_directory_no_replace,
    _sha256_file,
    _validated_upstream,
    _zip_info,
)
from src.next6_wbm_protocol import reduced_formula_key


PROTOCOL = "2026-08-02-next23-wbm-relaxation-change-holdout-v1"
SELECTION_SALT = "next23-wbm-relaxation-change-blind-v1"
FORMAL_SAMPLE_SIZE = 8192
FORMAL_MIN_ATOMS = 2
FORMAL_MAX_ATOMS = 12
FROZEN_FORMAL_EXCLUSION_SHA256 = (
    "ace914af28d6d1e82bbdd2a4ca0d7be39dc024fa9a98192c8ce770dfc5c75861"
)
FORBIDDEN_EXCLUSION_COLUMN_TOKENS = (
    "energy",
    "force",
    "stress",
    "relax",
    "dft",
    "endpoint",
    "label",
    "target",
)


def selection_key(material_id: str) -> str:
    if type(material_id) is not str or not material_id:
        raise ValueError("material_id must be a nonempty exact string")
    return hashlib.sha256(
        f"{SELECTION_SALT}|{material_id}".encode("utf-8")
    ).hexdigest()


def _canonical_id_digest(material_ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(material_ids)).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()


def _validated_exclusions(path: Path, source_ids: set[str]) -> tuple[str, ...]:
    frame = pd.read_parquet(path)
    forbidden = [
        str(column)
        for column in frame.columns
        if any(
            token in str(column).lower()
            for token in FORBIDDEN_EXCLUSION_COLUMN_TOKENS
        )
    ]
    if forbidden:
        raise ValueError(f"exclusion metadata crossed label-free contract: {forbidden}")
    if "material_id" not in frame or frame["material_id"].isna().any():
        raise ValueError("exclusion metadata lacks material IDs")
    material_ids = frame["material_id"].astype(str)
    if material_ids.duplicated().any():
        raise ValueError("exclusion material IDs must be unique")
    unknown = sorted(set(material_ids) - source_ids)
    if unknown:
        raise ValueError(f"exclusion IDs are absent from WBM test IDs: {unknown[:3]}")
    return tuple(sorted(material_ids.tolist()))


def freeze_disjoint_wbm_holdout(
    *,
    test_features_path: Path,
    wbm_manifest_path: Path,
    initial_zip_path: Path,
    exclusion_metadata_path: Path,
    output_dir: Path,
    sample_size: int = FORMAL_SAMPLE_SIZE,
    min_atoms: int = FORMAL_MIN_ATOMS,
    max_atoms: int = FORMAL_MAX_ATOMS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Select and sanitize WBM x0 structures after excluding exposed IDs."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    if type(sample_size) is not int or sample_size <= 0:
        raise ValueError("sample_size must be a positive exact integer")
    if (
        type(min_atoms) is not int
        or type(max_atoms) is not int
        or not 1 <= min_atoms <= max_atoms
    ):
        raise ValueError("atom bounds must be ordered positive exact integers")
    paths = {
        "test_x0_features": Path(test_features_path).resolve(),
        "wbm_manifest": Path(wbm_manifest_path).resolve(),
        "initial_zip": Path(initial_zip_path).resolve(),
        "exclusion_metadata": Path(exclusion_metadata_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    input_hashes = {role: _sha256_file(path) for role, path in paths.items()}
    if require_formal_inputs and (
        {role: input_hashes[role] for role in FROZEN_FORMAL_INPUT_SHA256}
        != dict(FROZEN_FORMAL_INPUT_SHA256)
        or input_hashes["exclusion_metadata"]
        != FROZEN_FORMAL_EXCLUSION_SHA256
        or sample_size != FORMAL_SAMPLE_SIZE
        or min_atoms != FORMAL_MIN_ATOMS
        or max_atoms != FORMAL_MAX_ATOMS
    ):
        raise ValueError("formal NEXT23 inputs or sampling constants differ")

    features = _validated_upstream(
        features_path=paths["test_x0_features"],
        manifest_path=paths["wbm_manifest"],
        initial_zip_path=paths["initial_zip"],
    )
    source_ids = set(features["material_id"])
    exclusions = _validated_exclusions(paths["exclusion_metadata"], source_ids)
    excluded_ids = set(exclusions)

    size_eligible: list[tuple[str, str, int]] = []
    with zipfile.ZipFile(paths["initial_zip"]) as archive:
        for material_id in features["material_id"].tolist():
            info = _archive_member(archive, material_id)
            try:
                natoms = int(archive.read(info).splitlines()[0].strip())
            except (IndexError, ValueError) as exc:
                raise ValueError(f"invalid WBM atom count: {material_id}") from exc
            if min_atoms <= natoms <= max_atoms:
                size_eligible.append((selection_key(material_id), material_id, natoms))
    selection_eligible = [
        row for row in size_eligible if row[1] not in excluded_ids
    ]
    selection_eligible.sort()
    if len(selection_eligible) < sample_size:
        raise ValueError(
            f"only {len(selection_eligible)} eligible WBM rows after exclusions "
            f"for sample_size={sample_size}"
        )
    selected = selection_eligible[:sample_size]
    ranks = {
        material_id: rank
        for rank, (_key, material_id, _natoms) in enumerate(selected)
    }

    structures = {}
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(paths["initial_zip"]) as archive:
        for key, material_id, expected_natoms in selected:
            text = archive.read(_archive_member(archive, material_id)).decode("utf-8")
            atoms = _atoms_from_text(text, expected_id=material_id)
            if len(atoms) != expected_natoms:
                raise ValueError(f"WBM atom count changed while freezing: {material_id}")
            formula = atoms.get_chemical_formula(mode="hill")
            structures[material_id] = atoms
            rows.append(
                {
                    "material_id": material_id,
                    "rk": reduced_formula_key(formula),
                    "formula": formula,
                    "natoms": len(atoms),
                    "selection_rank": ranks[material_id],
                    "selection_key_sha256": key,
                    "selection_salt": SELECTION_SALT,
                    "input_role": "unrelaxed_x0_geometry_only",
                }
            )
    metadata = pd.DataFrame(rows).sort_values(
        "material_id", kind="stable", ignore_index=True
    )
    if (
        len(metadata) != sample_size
        or metadata["material_id"].duplicated().any()
        or set(metadata["material_id"]) & excluded_ids
    ):
        raise RuntimeError("NEXT23 holdout accounting or disjointness differs")

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next23_wbm_holdout.py": Path(__file__).resolve(),
        "src/next14_wbm_holdout.py": repository_root / "src/next14_wbm_holdout.py",
        "src/next6_wbm_features.py": repository_root / "src/next6_wbm_features.py",
        "src/next6_wbm_protocol.py": repository_root / "src/next6_wbm_protocol.py",
        "src/next11_geometry_only_frames.py": repository_root
        / "src/next11_geometry_only_frames.py",
    }
    source_hashes = {
        relative: _sha256_file(path) for relative, path in source_paths.items()
    }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "development_disjoint_label_free_wbm_holdout",
        "evidence_role": "frozen blind relaxation-change evaluation cohort",
        "input_role": "unrelaxed_x0_geometry_only",
        "labels_opened": False,
        "relaxed_structures_opened": False,
        "endpoint_artifacts_opened": False,
        "selection": {
            "salt": SELECTION_SALT,
            "sample_size": sample_size,
            "minimum_atoms": min_atoms,
            "maximum_atoms": max_atoms,
            "ranking": (
                "ascending SHA-256(salt|material_id) among size-eligible test "
                "rows after exact development-ID exclusion"
            ),
            "excluded_material_ids_sha256": _canonical_id_digest(exclusions),
            "label_fields_available": [],
        },
        "counts": {
            "source_test_rows": len(features),
            "source_exclusion_rows": len(exclusions),
            "size_eligible_rows": len(size_eligible),
            "excluded_size_eligible_rows": sum(
                material_id in excluded_ids
                for _key, material_id, _natoms in size_eligible
            ),
            "selection_eligible_rows": len(selection_eligible),
            "selected_rows": len(metadata),
            "total_atoms": int(metadata["natoms"].sum()),
        },
        "inputs_sha256": {
            role: {"path": str(paths[role]), "sha256": digest}
            for role, digest in input_hashes.items()
        },
        "executed_source_sha256": source_hashes,
        "production_protocol_eligible": bool(require_formal_inputs),
        "scientific_improvement_claim": False,
    }

    def verify_unchanged() -> None:
        for role, path in paths.items():
            if _sha256_file(path) != input_hashes[role]:
                raise RuntimeError(f"input {role} changed before publication")
        for relative, path in source_paths.items():
            if _sha256_file(path) != source_hashes[relative]:
                raise RuntimeError(f"source {relative} changed before publication")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        metadata_path = staging / METADATA_NAME
        geometry_path = staging / GEOMETRY_NAME
        metadata.to_parquet(metadata_path, index=False)
        with zipfile.ZipFile(geometry_path, "x") as archive:
            for material_id in sorted(structures):
                archive.writestr(
                    _zip_info(f"{material_id}.extxyz"),
                    _canonical_frame(structures[material_id]),
                )
        manifest["outputs_sha256"] = {
            METADATA_NAME: _sha256_file(metadata_path),
            GEOMETRY_NAME: _sha256_file(geometry_path),
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        verify_unchanged()
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-features", required=True, type=Path)
    parser.add_argument("--wbm-manifest", required=True, type=Path)
    parser.add_argument("--initial-zip", required=True, type=Path)
    parser.add_argument("--exclude-metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    freeze_disjoint_wbm_holdout(
        test_features_path=arguments.test_features,
        wbm_manifest_path=arguments.wbm_manifest,
        initial_zip_path=arguments.initial_zip,
        exclusion_metadata_path=arguments.exclude_metadata,
        output_dir=arguments.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORMAL_SAMPLE_SIZE",
    "GEOMETRY_NAME",
    "MANIFEST_NAME",
    "METADATA_NAME",
    "freeze_disjoint_wbm_holdout",
    "selection_key",
]
