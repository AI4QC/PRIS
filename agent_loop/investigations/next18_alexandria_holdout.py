"""Sanitize the fixed Alexandria two-shard batch into geometry-only complete groups."""

from __future__ import annotations

import argparse
import bz2
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterator, Mapping, Sequence
import zipfile

from ase import Atoms
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from src.next11_geometry_only_frames import _canonical_frame
from src.next13d_acsc_dft_pairs import _json_bytes, _sha256_file
from src.next14_wbm_holdout import _publish_directory_no_replace


PROTOCOL = "2026-08-02-next18-alexandria-two-shard-geometry-holdout-v1"
METADATA_NAME = "alexandria_x0_metadata.parquet"
GEOMETRY_NAME = "geometry_only_frames.zip"
MANIFEST_NAME = "MANIFEST.json"
OFFICIAL_BASE_URL = "https://alexandria.icams.rub.de/data/geo_opt_paths/2025.07.02/pbe/"
EXPECTED_SOURCE_ROWS = 20_000
EXPECTED_SOURCE_GROUPS = 19_806
EXPECTED_SELECTED_GROUPS = 185
EXPECTED_SELECTED_ROWS = 379
FROZEN_FORMAL_SHA256: Mapping[str, str] = {
    "pbe_0000": "9f83c116839d528a6c625ad158b060298a969f76ce69dd1e29a74806376e389d",
    "pbe_0001": "dff2091cc3a8eaf38472ef0487d1bd678bda3d62ad6c91e226a5a187058387dd",
}


def iter_bz2_object(path: Path, *, chunk_chars: int = 1 << 20) -> Iterator[tuple[str, object]]:
    """Stream key/value pairs from one top-level JSON object without whole-file loading."""

    if type(chunk_chars) is not int or chunk_chars <= 0:
        raise ValueError("chunk_chars must be a positive exact integer")
    decoder = json.JSONDecoder()
    with bz2.open(Path(path), "rt", encoding="utf-8") as stream:
        buffer = ""
        position = 0

        def refill() -> bool:
            nonlocal buffer, position
            chunk = stream.read(chunk_chars)
            if not chunk:
                return False
            buffer = buffer[position:] + chunk
            position = 0
            return True

        def skip_whitespace() -> None:
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    return
                if not refill():
                    raise ValueError("truncated Alexandria JSON")

        def consume(expected: str) -> None:
            nonlocal position
            skip_whitespace()
            if buffer[position] != expected:
                raise ValueError(f"expected {expected!r} in Alexandria JSON")
            position += 1

        def decode_one() -> object:
            nonlocal position
            while True:
                skip_whitespace()
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    position = end
                    return value
                except json.JSONDecodeError as exc:
                    if not refill():
                        raise ValueError("invalid Alexandria JSON value") from exc

        if not refill():
            raise ValueError("empty Alexandria shard")
        consume("{")
        count = 0
        while True:
            skip_whitespace()
            if buffer[position] == "}":
                position += 1
                break
            if count:
                consume(",")
            key = decode_one()
            if type(key) is not str or not key:
                raise ValueError("Alexandria material ID must be a nonempty string")
            consume(":")
            value = decode_one()
            yield key, value
            count += 1
        trailing = buffer[position:] + stream.read()
        if trailing.strip():
            raise ValueError("trailing data after Alexandria JSON object")


def _initial_structure(calculations: object) -> Structure:
    if not isinstance(calculations, list):
        raise ValueError("Alexandria trajectory must be a list")
    for calculation in calculations:
        if not isinstance(calculation, Mapping):
            continue
        steps = calculation.get("steps")
        if isinstance(steps, list) and steps:
            first = steps[0]
            if not isinstance(first, Mapping) or not isinstance(first.get("structure"), Mapping):
                raise ValueError("Alexandria first ionic step lacks a structure")
            try:
                structure = Structure.from_dict(first["structure"])
            except Exception as exc:
                raise ValueError("invalid Alexandria initial structure") from exc
            if len(structure) <= 0 or not all(structure.lattice.pbc):
                raise ValueError("Alexandria initial structure must be nonempty and periodic")
            return structure
    raise ValueError("Alexandria trajectory lacks a nonempty calculation")


def _clean_atoms(structure: Structure) -> Atoms:
    converted = AseAtomsAdaptor.get_atoms(structure)
    return Atoms(
        numbers=converted.get_atomic_numbers(),
        positions=converted.get_positions(),
        cell=converted.cell.array,
        pbc=True,
    )


def build_alexandria_holdout(
    *,
    shard_0000_path: Path,
    shard_0001_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
    expected_source_rows: int = EXPECTED_SOURCE_ROWS,
    expected_source_groups: int = EXPECTED_SOURCE_GROUPS,
    expected_selected_groups: int = EXPECTED_SELECTED_GROUPS,
    expected_selected_rows: int = EXPECTED_SELECTED_ROWS,
) -> dict[str, object]:
    """Select every repeated x0 composition in the fixed two-shard batch."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing existing output: {target}")
    paths = {
        "pbe_0000": Path(shard_0000_path).resolve(),
        "pbe_0001": Path(shard_0001_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    input_hashes = {role: _sha256_file(path) for role, path in paths.items()}
    if require_formal_inputs and input_hashes != dict(FROZEN_FORMAL_SHA256):
        raise ValueError("formal Alexandria shard identities differ")
    invariants = (
        expected_source_rows,
        expected_source_groups,
        expected_selected_groups,
        expected_selected_rows,
    )
    if any(type(value) is not int or value <= 0 for value in invariants):
        raise ValueError("Alexandria count invariants must be positive exact integers")

    inventory: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for role, path in paths.items():
        for material_id, calculations in iter_bz2_object(path):
            if material_id in seen_ids:
                raise ValueError(f"duplicate Alexandria material ID: {material_id}")
            seen_ids.add(material_id)
            structure = _initial_structure(calculations)
            inventory.append(
                {
                    "material_id": material_id,
                    "rk": structure.composition.reduced_formula,
                    "formula": structure.composition.formula.replace(" ", ""),
                    "natoms": len(structure),
                    "source_shard": role,
                }
            )
    inventory_table = pd.DataFrame(inventory)
    if len(inventory_table) != expected_source_rows:
        raise ValueError("Alexandria source row count differs")
    group_sizes = inventory_table.groupby("rk", sort=True).size()
    if len(group_sizes) != expected_source_groups:
        raise ValueError("Alexandria source composition count differs")
    selected_groups = set(group_sizes[group_sizes >= 2].index.astype(str))
    selected = inventory_table.loc[inventory_table["rk"].isin(selected_groups)].sort_values(
        "material_id", kind="stable", ignore_index=True
    )
    if len(selected_groups) != expected_selected_groups or len(selected) != expected_selected_rows:
        raise ValueError("Alexandria repeated-composition cohort count differs")
    expected_ids = set(selected["material_id"].astype(str))

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        geometry_path = staging / GEOMETRY_NAME
        written: set[str] = set()
        with zipfile.ZipFile(
            geometry_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            archive.comment = b""
            selected_payloads: dict[str, bytes] = {}
            for path in paths.values():
                for material_id, calculations in iter_bz2_object(path):
                    if material_id not in expected_ids:
                        continue
                    structure = _initial_structure(calculations)
                    selected_payloads[material_id] = _canonical_frame(_clean_atoms(structure))
            if set(selected_payloads) != expected_ids:
                raise ValueError("Alexandria selected geometry coverage differs")
            for material_id in sorted(selected_payloads):
                name = f"{material_id}.extxyz"
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.extra = b""
                info.comment = b""
                archive.writestr(
                    info,
                    selected_payloads[material_id],
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
                written.add(material_id)
        if written != expected_ids:
            raise ValueError("Alexandria geometry archive lost selected IDs")
        metadata = selected.copy()
        metadata["input_role"] = "unrelaxed_x0_geometry_only"
        metadata_path = staging / METADATA_NAME
        metadata.to_parquet(metadata_path, index=False)
        selected_size_histogram = {
            str(size): int(count)
            for size, count in Counter(
                metadata.groupby("rk", sort=True).size().astype(int).tolist()
            ).items()
        }
        repo_root = Path(__file__).resolve().parents[1]
        source_paths = {
            "src/next18_alexandria_holdout.py": Path(__file__).resolve(),
            "src/next11_geometry_only_frames.py": repo_root / "src/next11_geometry_only_frames.py",
        }
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "evidence_role": "external Alexandria source geometry-only falsification cohort",
            "official_source_base_url": OFFICIAL_BASE_URL,
            "fixed_source_shards": ["pbe_0000.json.bz2", "pbe_0001.json.bz2"],
            "input_role": "unrelaxed_x0_geometry_only",
            "selection": {
                "unit": "complete reduced-composition group within the fixed two-shard batch",
                "rule": "include every reduced composition with at least two x0 candidates",
                "sampled": False,
                "endpoint_fields_used": False,
            },
            "raw_container_endpoint_bytes_present": True,
            "raw_container_bytes_read_by_sanitizer": True,
            "endpoint_fields_accessed_by_sanitizer": False,
            "downstream_geometry_artifacts_include_endpoint_fields": False,
            "fresh_never_read_lockbox": False,
            "counts": {
                "source_rows": len(inventory_table),
                "source_groups": len(group_sizes),
                "selected_rows": len(metadata),
                "selected_groups": int(metadata["rk"].nunique()),
                "selected_group_size_histogram": dict(sorted(selected_size_histogram.items())),
                "selected_atoms": int(metadata["natoms"].sum()),
            },
            "inputs_sha256": {
                role: {"path": str(paths[role]), "sha256": digest}
                for role, digest in input_hashes.items()
            },
            "outputs_sha256": {
                METADATA_NAME: _sha256_file(metadata_path),
                GEOMETRY_NAME: _sha256_file(geometry_path),
            },
            "executed_source_sha256": {
                name: _sha256_file(path) for name, path in source_paths.items()
            },
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-0000", required=True, type=Path)
    parser.add_argument("--shard-0001", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    build_alexandria_holdout(
        shard_0000_path=args.shard_0000,
        shard_0001_path=args.shard_0001,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
