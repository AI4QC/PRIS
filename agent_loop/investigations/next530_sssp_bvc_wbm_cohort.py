#!/usr/bin/env python3
"""Freeze a new WBM x0 cohort disjoint from prior NEXT14/NEXT23 cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Sequence
import zipfile

import pandas as pd

import src.next14_wbm_holdout as n14
import src.next23_wbm_holdout as n23
import src.next529_sssp_bvc_development_freeze as n529
from src.next6_wbm_protocol import reduced_formula_key
from src.next347_periodic_allocation_redistribution_capacity import _sha256_file


PROTOCOL = "2026-08-13-next530-sssp-bvc-new-disjoint-wbm-cohort-v1"
SELECTION_SALT = "next530-sssp-bvc-wbm-relaxation-v1"
SAMPLE_SIZE = 8_192
MIN_ATOMS = 2
MAX_ATOMS = 12
MANIFEST_NAME = "MANIFEST.json"
METADATA_NAME = "next530_wbm_x0_metadata.parquet"
GEOMETRY_NAME = "next530_wbm_x0_geometry.zip"
EXPECTED_INPUT_SHA256 = {
    "design": n529.DESIGN_SHA256,
    "next529_manifest": "3f5bfa89726bfa7edc8daa898169c3e9259c5d3d29e1d12c2674fb4343f17705",
    "next529_formula": "b50e194273e83f06e26bd4f4e9c904cd692dc9fa9d874aebb0181c4fcfa849be",
    "test_features": "91ac6dc5bda3d9bb27ba390b7f108631b2f4466fae1cc3101f385bd5d69a171f",
    "wbm_manifest": "e08a30ee817986f24b72309e41c2026142205af6a4850dc30ab2529efa47a8cd",
    "initial_zip": "8d783b938f510624577cdbef1d2e3c232cc04476c4b581c894a7ca1b172ba0d0",
    "next14_exclusions": "ace914af28d6d1e82bbdd2a4ca0d7be39dc024fa9a98192c8ce770dfc5c75861",
    "next23_exclusions": "919deba21caae3c709790a52d40a44282ff983844c2559267f5f716f01fccbcd",
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def selection_key(material_id: str) -> str:
    if type(material_id) is not str or not material_id:
        raise ValueError("material_id must be a nonempty exact string")
    return hashlib.sha256(f"{SELECTION_SALT}|{material_id}".encode()).hexdigest()


def validated_exclusion_union(
    paths: Sequence[Path], *, source_ids: set[str]
) -> tuple[str, ...]:
    if not paths:
        raise ValueError("NEXT530 exclusion paths are empty")
    result: set[str] = set()
    for path in paths:
        result.update(n23._validated_exclusions(Path(path), source_ids))
    return tuple(sorted(result))


def freeze_wbm_cohort(
    *,
    test_features_path: Path,
    wbm_manifest_path: Path,
    initial_zip_path: Path,
    next14_exclusion_path: Path,
    next23_exclusion_path: Path,
    next529_dir: Path,
    design_path: Path,
    output_dir: Path,
    sample_size: int = SAMPLE_SIZE,
    min_atoms: int = MIN_ATOMS,
    max_atoms: int = MAX_ATOMS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    freeze = Path(next529_dir).resolve()
    paths = {
        "design": Path(design_path).resolve(),
        "next529_manifest": freeze / n529.MANIFEST_NAME,
        "next529_formula": freeze / n529.FORMULA_NAME,
        "test_features": Path(test_features_path).resolve(),
        "wbm_manifest": Path(wbm_manifest_path).resolve(),
        "initial_zip": Path(initial_zip_path).resolve(),
        "next14_exclusions": Path(next14_exclusion_path).resolve(),
        "next23_exclusions": Path(next23_exclusion_path).resolve(),
    }
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if (
        type(sample_size) is not int or sample_size <= 0
        or type(min_atoms) is not int or type(max_atoms) is not int
        or not 1 <= min_atoms <= max_atoms
    ):
        raise ValueError("NEXT530 cohort constants differ")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT530 input is missing")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and (
        hashes != EXPECTED_INPUT_SHA256
        or sample_size != SAMPLE_SIZE or min_atoms != MIN_ATOMS or max_atoms != MAX_ATOMS
    ):
        differing = sorted(
            name for name in set(hashes) | set(EXPECTED_INPUT_SHA256)
            if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT530 formal input identity differs: {differing}")
    prior_manifest = _read_json(paths["next529_manifest"])
    formula = _read_json(paths["next529_formula"])
    if (
        prior_manifest.get("protocol") != n529.PROTOCOL
        or prior_manifest.get("next530_wbm_cohort_authorized") is not True
        or prior_manifest.get("wbm_external_endpoint_opened") is not False
        or formula.get("dft_inputs") != []
        or formula.get("learned_model_inputs") != []
        or formula.get("relaxation_inputs") != []
    ):
        raise ValueError("NEXT530 formula authorization differs")
    features = n23._validated_upstream(
        features_path=paths["test_features"],
        manifest_path=paths["wbm_manifest"],
        initial_zip_path=paths["initial_zip"],
    )
    source_ids = set(features["material_id"].astype(str))
    exclusions = validated_exclusion_union(
        [paths["next14_exclusions"], paths["next23_exclusions"]],
        source_ids=source_ids,
    )
    excluded = set(exclusions)
    size_eligible = []
    with zipfile.ZipFile(paths["initial_zip"]) as archive:
        for material_id in features["material_id"].astype(str):
            info = n14._archive_member(archive, material_id)
            try:
                natoms = int(archive.read(info).splitlines()[0].strip())
            except (IndexError, ValueError) as exc:
                raise ValueError(f"invalid WBM atom count: {material_id}") from exc
            if min_atoms <= natoms <= max_atoms:
                size_eligible.append((selection_key(material_id), material_id, natoms))
    eligible = [row for row in size_eligible if row[1] not in excluded]
    eligible.sort()
    if len(eligible) < sample_size:
        raise ValueError("NEXT530 insufficient rows after exact exclusions")
    selected = eligible[:sample_size]
    structures = {}
    rows = []
    with zipfile.ZipFile(paths["initial_zip"]) as archive:
        for rank, (key, material_id, expected_natoms) in enumerate(selected):
            text = archive.read(n14._archive_member(archive, material_id)).decode("utf-8")
            atoms = n14._atoms_from_text(text, expected_id=material_id)
            if len(atoms) != expected_natoms:
                raise ValueError(f"NEXT530 atom count changed: {material_id}")
            formula_text = atoms.get_chemical_formula(mode="hill")
            structures[material_id] = atoms
            rows.append(
                {
                    "material_id": material_id,
                    "rk": reduced_formula_key(formula_text),
                    "formula": formula_text,
                    "natoms": len(atoms),
                    "selection_rank": rank,
                    "selection_key_sha256": key,
                    "selection_salt": SELECTION_SALT,
                    "partition_role": "external_validation",
                    "input_role": "unrelaxed_x0_geometry_only",
                }
            )
    metadata = pd.DataFrame(rows).sort_values("material_id", kind="mergesort").reset_index(drop=True)
    if (
        len(metadata) != sample_size
        or metadata["material_id"].duplicated().any()
        or set(metadata["material_id"]) & excluded
    ):
        raise RuntimeError("NEXT530 disjoint cohort identity differs")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_hash = _sha256_file(Path(__file__).resolve())
    try:
        metadata_path, geometry_path = staging / METADATA_NAME, staging / GEOMETRY_NAME
        metadata.to_parquet(metadata_path, index=False)
        with zipfile.ZipFile(geometry_path, "x") as archive:
            for material_id in sorted(structures):
                archive.writestr(
                    n14._zip_info(f"{material_id}.extxyz"),
                    n14._canonical_frame(structures[material_id]),
                )
        manifest = {
            "protocol": PROTOCOL,
            "mode": "new_disjoint_label_free_wbm_external_cohort",
            "evidence_role": "SSSP-BVC external relaxation-change validation",
            "input_role": "unrelaxed_x0_geometry_only",
            "labels_opened": False,
            "wbm_summary_opened": False,
            "relaxed_structures_opened": False,
            "endpoint_artifacts_opened": False,
            "selection": {
                "salt": SELECTION_SALT,
                "sample_size": sample_size,
                "minimum_atoms": min_atoms,
                "maximum_atoms": max_atoms,
                "excluded_material_ids_sha256": n23._canonical_id_digest(exclusions),
                "excluded_material_id_count": len(exclusions),
                "label_fields_available": [],
            },
            "counts": {
                "source_test_rows": len(features),
                "exclusion_union_rows": len(exclusions),
                "size_eligible_rows": len(size_eligible),
                "excluded_size_eligible_rows": sum(row[1] in excluded for row in size_eligible),
                "selection_eligible_rows": len(eligible),
                "selected_rows": len(metadata),
                "total_atoms": int(metadata["natoms"].sum()),
            },
            "inputs_sha256": {
                name: {"path": str(path), "sha256": hashes[name]}
                for name, path in paths.items()
            },
            "executed_source_sha256": {
                "src/next530_sssp_bvc_wbm_cohort.py": source_hash
            },
            "outputs_sha256": {
                METADATA_NAME: _sha256_file(metadata_path),
                GEOMETRY_NAME: _sha256_file(geometry_path),
            },
            "next531_label_free_features_authorized": True,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256_file(Path(__file__).resolve()) != source_hash:
            raise RuntimeError("NEXT530 source changed before publication")
        if any(_sha256_file(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT530 input changed before publication")
        n14._publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--wbm-manifest", type=Path, required=True)
    parser.add_argument("--initial-zip", type=Path, required=True)
    parser.add_argument("--next14-exclusions", type=Path, required=True)
    parser.add_argument("--next23-exclusions", type=Path, required=True)
    parser.add_argument("--next529-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = freeze_wbm_cohort(
        test_features_path=args.test_features, wbm_manifest_path=args.wbm_manifest,
        initial_zip_path=args.initial_zip, next14_exclusion_path=args.next14_exclusions,
        next23_exclusion_path=args.next23_exclusions, next529_dir=args.next529_dir,
        design_path=args.design, output_dir=args.output_dir,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


__all__ = [
    "GEOMETRY_NAME", "MANIFEST_NAME", "MAX_ATOMS", "METADATA_NAME", "MIN_ATOMS",
    "PROTOCOL", "SAMPLE_SIZE", "SELECTION_SALT", "freeze_wbm_cohort",
    "selection_key", "validated_exclusion_union",
]


if __name__ == "__main__":
    main()
