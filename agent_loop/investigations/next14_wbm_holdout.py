"""Freeze a deterministic label-free small-cell WBM external holdout."""

from __future__ import annotations

import argparse
import errno
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Mapping, Sequence
import zipfile

from ase import Atoms
import numpy as np
import pandas as pd

from src.next6_wbm_features import parse_extxyz
from src.next6_wbm_protocol import reduced_formula_key
from src.next11_geometry_only_frames import _canonical_frame
from src.next12_dft_queue import _json_bytes, _zip_info
from src.next13d_acsc_dft_pairs import _sha256_file, _strict_json
from src.next8_mattersim_committee_features import (
    _atomic_publish_directory_no_replace,
)


PROTOCOL = "2026-08-02-next14-wbm-acsc-label-free-holdout-v1"
UPSTREAM_PROTOCOL = "2026-08-01-dft-pre-screening-design-v1"
SELECTION_SALT = "next14-wbm-acsc-external-v1"
FORMAL_SAMPLE_SIZE = 2048
FORMAL_MIN_ATOMS = 2
FORMAL_MAX_ATOMS = 12
GEOMETRY_NAME = "geometry_only_frames.zip"
METADATA_NAME = "holdout_metadata.parquet"
MANIFEST_NAME = "MANIFEST.json"
FROZEN_FORMAL_INPUT_SHA256: Mapping[str, str] = {
    "test_x0_features": "91ac6dc5bda3d9bb27ba390b7f108631b2f4466fae1cc3101f385bd5d69a171f",
    "wbm_manifest": "e08a30ee817986f24b72309e41c2026142205af6a4850dc30ab2529efa47a8cd",
    "initial_zip": "8d783b938f510624577cdbef1d2e3c232cc04476c4b581c894a7ca1b172ba0d0",
}


def selection_key(material_id: str) -> str:
    if type(material_id) is not str or not material_id:
        raise ValueError("material_id must be a nonempty exact string")
    return hashlib.sha256(f"{SELECTION_SALT}|{material_id}".encode("utf-8")).hexdigest()


def _publish_directory_no_replace(source: Path, target: Path) -> None:
    """Publish safely on filesystems without renameat2/RENAME_NOREPLACE."""

    try:
        _atomic_publish_directory_no_replace(source, target)
        return
    except OSError as exc:
        if exc.errno != errno.EINVAL or not source.is_dir() or os.path.lexists(target):
            raise
    # mkdir is the exclusive no-overwrite reservation. Files become complete
    # when MANIFEST.json is moved last; a failed publication removes only this
    # directory created by the current invocation.
    target.mkdir(parents=False, exist_ok=False)
    try:
        members = sorted(
            source.iterdir(),
            key=lambda path: (path.name == MANIFEST_NAME, path.name),
        )
        if any(not member.is_file() for member in members):
            raise OSError("fallback publication supports regular files only")
        for member in members:
            os.rename(member, target / member.name)
        source.rmdir()
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def _validated_upstream(
    *, features_path: Path, manifest_path: Path, initial_zip_path: Path
) -> pd.DataFrame:
    data = features_path.read_bytes()
    manifest = _strict_json(manifest_path.read_bytes(), role="WBM artifact manifest")
    if manifest.get("protocol") != UPSTREAM_PROTOCOL or manifest.get("input_role") != "unrelaxed_x0_only":
        raise ValueError("WBM upstream protocol or input role differs")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(features_path.name) != hashlib.sha256(data).hexdigest():
        raise ValueError("WBM test feature hash differs from its manifest")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping) or inputs.get("initial_zip_sha256") != _sha256_file(initial_zip_path):
        raise ValueError("WBM initial archive hash differs from its manifest")
    features = pd.read_parquet(io.BytesIO(data))
    if "material_id" not in features or features["material_id"].isna().any():
        raise ValueError("WBM test features lack material IDs")
    features = features.loc[:, ["material_id"]].copy()
    features["material_id"] = features["material_id"].astype(str)
    if features["material_id"].duplicated().any():
        raise ValueError("WBM test material IDs must be unique")
    return features.sort_values("material_id", kind="stable", ignore_index=True)


def _archive_member(archive: zipfile.ZipFile, material_id: str) -> zipfile.ZipInfo:
    name = f"{material_id}.extxyz"
    try:
        info = archive.getinfo(name)
    except KeyError as exc:
        raise FileNotFoundError(f"WBM initial frame missing: {name}") from exc
    path = PurePosixPath(info.filename)
    if info.is_dir() or path.parent != PurePosixPath(".") or path.suffix != ".extxyz" or info.flag_bits & 0x1:
        raise ValueError(f"unsafe WBM initial frame member: {info.filename}")
    return info


def _atoms_from_text(text: str, *, expected_id: str) -> Atoms:
    frame = parse_extxyz(text)
    if frame.material_id != expected_id:
        raise ValueError(f"WBM frame ID differs: expected {expected_id}, found {frame.material_id}")
    atoms = Atoms(
        symbols=list(frame.species),
        positions=np.asarray(frame.cart_coords, dtype=np.float64),
        cell=np.asarray(frame.lattice, dtype=np.float64),
        pbc=True,
    )
    if len(atoms) < 1 or not np.all(np.isfinite(atoms.positions)) or not np.all(np.isfinite(atoms.cell.array)) or atoms.get_volume() <= 0.0:
        raise ValueError(f"invalid WBM initial geometry: {expected_id}")
    return atoms


def freeze_wbm_holdout(
    *,
    test_features_path: Path,
    wbm_manifest_path: Path,
    initial_zip_path: Path,
    output_dir: Path,
    sample_size: int = FORMAL_SAMPLE_SIZE,
    min_atoms: int = FORMAL_MIN_ATOMS,
    max_atoms: int = FORMAL_MAX_ATOMS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Select and sanitize WBM x0 structures without reading any label table."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    if type(sample_size) is not int or sample_size <= 0:
        raise ValueError("sample_size must be a positive exact integer")
    if type(min_atoms) is not int or type(max_atoms) is not int or not 1 <= min_atoms <= max_atoms:
        raise ValueError("atom bounds must be ordered positive exact integers")
    paths = {
        "test_x0_features": Path(test_features_path).resolve(),
        "wbm_manifest": Path(wbm_manifest_path).resolve(),
        "initial_zip": Path(initial_zip_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    input_hashes = {role: _sha256_file(path) for role, path in paths.items()}
    if require_formal_inputs and (
        input_hashes != dict(FROZEN_FORMAL_INPUT_SHA256)
        or sample_size != FORMAL_SAMPLE_SIZE
        or min_atoms != FORMAL_MIN_ATOMS
        or max_atoms != FORMAL_MAX_ATOMS
    ):
        raise ValueError("formal WBM holdout inputs or sampling constants differ")
    features = _validated_upstream(
        features_path=paths["test_x0_features"],
        manifest_path=paths["wbm_manifest"],
        initial_zip_path=paths["initial_zip"],
    )

    eligible: list[tuple[str, str, int]] = []
    with zipfile.ZipFile(paths["initial_zip"]) as archive:
        for material_id in features["material_id"].tolist():
            info = _archive_member(archive, material_id)
            try:
                first_line = archive.read(info).splitlines()[0]
                natoms = int(first_line.strip())
            except (IndexError, ValueError) as exc:
                raise ValueError(f"invalid WBM atom count: {material_id}") from exc
            if min_atoms <= natoms <= max_atoms:
                eligible.append((selection_key(material_id), material_id, natoms))
    eligible.sort()
    if len(eligible) < sample_size:
        raise ValueError(f"only {len(eligible)} eligible WBM rows for sample_size={sample_size}")
    selected = eligible[:sample_size]
    selection_rank = {material_id: rank for rank, (_key, material_id, _n) in enumerate(selected)}

    structures: dict[str, Atoms] = {}
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
                    "selection_rank": selection_rank[material_id],
                    "selection_key_sha256": key,
                    "selection_salt": SELECTION_SALT,
                    "input_role": "unrelaxed_x0_geometry_only",
                }
            )
    metadata = pd.DataFrame(rows).sort_values("material_id", kind="stable", ignore_index=True)
    if len(metadata) != sample_size or metadata["material_id"].duplicated().any():
        raise RuntimeError("WBM frozen holdout accounting differs")

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next14_wbm_holdout.py": Path(__file__).resolve(),
        "src/next6_wbm_features.py": repository_root / "src/next6_wbm_features.py",
        "src/next6_wbm_protocol.py": repository_root / "src/next6_wbm_protocol.py",
        "src/next11_geometry_only_frames.py": repository_root / "src/next11_geometry_only_frames.py",
    }
    source_hashes = {relative: _sha256_file(path) for relative, path in source_paths.items()}
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "external_source_label_free_small_cell_holdout",
        "evidence_role": "external-source retrospective; not a fresh project-wide lockbox",
        "labels_opened": False,
        "relaxed_structures_opened": False,
        "endpoint_artifacts_opened": False,
        "selection": {
            "salt": SELECTION_SALT,
            "sample_size": sample_size,
            "minimum_atoms": min_atoms,
            "maximum_atoms": max_atoms,
            "ranking": "ascending SHA-256(salt|material_id) among all size-eligible test rows",
            "label_fields_available": [],
        },
        "counts": {
            "source_test_rows": len(features),
            "eligible_rows": len(eligible),
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
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
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
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    freeze_wbm_holdout(
        test_features_path=arguments.test_features,
        wbm_manifest_path=arguments.wbm_manifest,
        initial_zip_path=arguments.initial_zip,
        output_dir=arguments.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
