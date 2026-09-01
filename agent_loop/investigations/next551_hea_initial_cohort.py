#!/usr/bin/env python3
"""Select and publish a geometry-only HEA x0 cohort without opening DFT endpoints."""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from ase import Atoms
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from src.next11_geometry_only_frames import _ParsedFrame, _write_deterministic_archive
from src.next19_feature_build import _publish_directory_no_replace, _sha256


PROTOCOL = "2026-08-13-next551-hea-initial-cohort-v1"
DESIGN_SHA256 = "0ad0ea4c16327941e2c14cdaf82341f9b4f4ad05a19abbf14ab236db914eb743"
SOURCE_SHA256 = "42722654902f03efde4e1d53284741033f4c43b77d7ba5ced29e4cf873c455fe"
SOURCE_BYTES = 452_858_376
EXPECTED_SOURCE_ROWS = 84_024
EXPECTED_ELIGIBLE_ROWS = 83_797
SELECT_PER_FAMILY = 1_200
EXPECTED_SELECTED_ROWS = 2_400
EXPECTED_HEADER = (
    "fid",
    "reduced_formula",
    "chemical_system",
    "lattice",
    "nelements",
    "NIONS",
    "space_group_number",
    "volume_per_atom",
    "pressure",
    "stress",
    "e_per_atom",
    "Ef_per_atom",
    "e_above_hull",
    "magmom",
    "charge",
    "structure_ini_as_dict",
    "structure_as_dict",
    "kpt",
)
AUDIT_FIDS = frozenset(
    {
        "nar8898099", "nar8898101", "nar8898102", "nar8898103", "nar8898104",
        "nar8898105", "nar8898106", "nar8898107", "nar8898108", "nar8898109",
        "nar8898114", "nar8898115", "nar8898117", "nar8898118", "nar8898119",
        "nar8898120", "nar8898121", "nar8898122", "nar8898123", "nar8898125",
        "nar8898126", "nar8898127", "nar8898129", "nar8898131", "nar8898132",
        "nar8898133", "nar8898134", "nar8898139", "nar8898141", "nar8898142",
        "nar8898143",
    }
)
METADATA_NAME = "next551_hea_x0_metadata.parquet"
COHORT_NAME = "NEXT551_HEA_COHORT.json"
GEOMETRY_NAME = "geometry_only_frames.zip"
MANIFEST_NAME = "MANIFEST.json"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _project_csv_record(
    record: bytes, *, copy_indices: set[int], presence_indices: set[int] = frozenset()
) -> tuple[dict[int, bytes], dict[int, bool], int]:
    """Project selected RFC4180 fields while never copying skipped field bytes."""

    requested = set(copy_indices)
    presence_requested = set(presence_indices)
    copied: dict[int, bytes] = {}
    present: dict[int, bool] = {}
    index = 0
    position = 0
    in_quotes = False
    field_start = True
    content_present = False
    buffer = bytearray() if index in requested else None

    def finish_field() -> None:
        nonlocal index, buffer, content_present, field_start
        if index in requested:
            copied[index] = bytes(buffer or b"")
        if index in presence_requested:
            present[index] = content_present
        index += 1
        buffer = bytearray() if index in requested else None
        content_present = False
        field_start = True

    while position < len(record):
        byte = record[position]
        if in_quotes:
            if byte == 0x22:
                if position + 1 < len(record) and record[position + 1] == 0x22:
                    if buffer is not None:
                        buffer.append(0x22)
                    content_present = True
                    position += 2
                    continue
                in_quotes = False
                position += 1
                continue
            if buffer is not None:
                buffer.append(byte)
            content_present = True
            position += 1
            continue
        if field_start and byte == 0x22:
            in_quotes = True
            field_start = False
            position += 1
            continue
        if byte == 0x2C:
            finish_field()
            position += 1
            continue
        if byte in (0x0A, 0x0D):
            finish_field()
            while position < len(record) and record[position] in (0x0A, 0x0D):
                position += 1
            if position != len(record):
                raise ValueError("NEXT551 CSV record has trailing bytes")
            break
        if byte == 0x22:
            raise ValueError("NEXT551 CSV record has a quote outside a quoted field")
        if buffer is not None:
            buffer.append(byte)
        content_present = True
        field_start = False
        position += 1
    else:
        if in_quotes:
            raise ValueError("NEXT551 CSV record has an unterminated quote")
        finish_field()
    if in_quotes:
        raise ValueError("NEXT551 CSV record has an unterminated quote")
    return copied, present, index


def cohort_hash(fid: str) -> str:
    return hashlib.sha256(f"NEXT551-cohort-v1|{fid}".encode()).hexdigest()


def _split_hash(chemical_system: str) -> str:
    return hashlib.sha256(f"NEXT551-split-v2|{chemical_system}".encode()).hexdigest()


def balanced_partition_map(rows: list[dict[str, object]]) -> dict[str, str]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        system = str(row["chemical_system"])
        family = str(row["size_family"])
        if not system or family not in {"ordered", "sqs"}:
            raise ValueError("NEXT551 split metadata differs")
        counts[system][family] += 1
        counts[system]["total"] += 1
    totals = {
        "development": Counter({"ordered": 0, "sqs": 0, "total": 0}),
        "validation": Counter({"ordered": 0, "sqs": 0, "total": 0}),
    }
    assignment: dict[str, str] = {}
    ordered_systems = sorted(
        counts,
        key=lambda system: (-counts[system]["total"], _split_hash(system), system),
    )
    for system in ordered_systems:
        preferred = "development" if bytes.fromhex(_split_hash(system))[0] % 2 == 0 else "validation"
        candidates: list[tuple[tuple[object, ...], str]] = []
        for partition in ("development", "validation"):
            other = "validation" if partition == "development" else "development"
            after_ordered = totals[partition]["ordered"] + counts[system]["ordered"]
            after_sqs = totals[partition]["sqs"] + counts[system]["sqs"]
            after_total = totals[partition]["total"] + counts[system]["total"]
            score = (
                max(
                    abs(after_ordered - totals[other]["ordered"]),
                    abs(after_sqs - totals[other]["sqs"]),
                ),
                abs(after_total - totals[other]["total"]),
                0 if partition == preferred else 1,
                partition,
            )
            candidates.append((score, partition))
        chosen = min(candidates)[1]
        assignment[system] = chosen
        totals[chosen].update(counts[system])
    return assignment


def _size_family(nions: int) -> str | None:
    if 2 <= nions <= 8:
        return "ordered"
    if nions in {27, 64, 125}:
        return "sqs"
    return None


def _header_indices(handle: object) -> dict[str, int]:
    header = handle.readline()
    if not isinstance(header, bytes):
        raise TypeError("NEXT551 source must be opened in binary mode")
    names = tuple(value.decode("utf-8") for value in header.rstrip(b"\r\n").split(b","))
    if names != EXPECTED_HEADER:
        raise ValueError("NEXT551 source header differs")
    return {name: index for index, name in enumerate(names)}


def _scan_label_free_metadata(source_path: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    rows: list[dict[str, object]] = []
    with source_path.open("rb") as handle:
        indices = _header_indices(handle)
        wanted_names = ("fid", "reduced_formula", "chemical_system", "nelements", "NIONS")
        wanted = {indices[name] for name in wanted_names}
        initial_index = indices["structure_ini_as_dict"]
        for row_index, record in enumerate(handle, start=1):
            copied, present, field_count = _project_csv_record(
                record, copy_indices=wanted, presence_indices={initial_index}
            )
            if field_count != len(EXPECTED_HEADER):
                raise ValueError(f"NEXT551 source field count differs at row {row_index}")
            decoded = {
                name: copied[indices[name]].decode("utf-8") for name in wanted_names
            }
            try:
                nelements = int(decoded["nelements"])
                nions = int(decoded["NIONS"])
            except ValueError as exc:
                raise ValueError(f"NEXT551 label-free integers differ at row {row_index}") from exc
            family = _size_family(nions)
            rows.append(
                {
                    "fid": decoded["fid"],
                    "reduced_formula": decoded["reduced_formula"],
                    "chemical_system": decoded["chemical_system"],
                    "nelements": nelements,
                    "nions": nions,
                    "size_family": family,
                    "initial_structure_present": bool(present[initial_index]),
                    "source_row_index": row_index,
                }
            )
    if len(rows) != EXPECTED_SOURCE_ROWS or len({str(row["fid"]) for row in rows}) != len(rows):
        raise ValueError("NEXT551 source row identity differs")
    eligible = [
        row for row in rows
        if row["initial_structure_present"] and row["size_family"] in {"ordered", "sqs"}
    ]
    if len(eligible) != EXPECTED_ELIGIBLE_ROWS:
        raise ValueError("NEXT551 eligible source count differs")
    by_fid = {str(row["fid"]): row for row in rows}
    if not AUDIT_FIDS.issubset(by_fid) or any(
        by_fid[fid]["initial_structure_present"] for fid in AUDIT_FIDS
    ):
        raise ValueError("NEXT551 schema-audit exclusion identity differs")
    stats = {
        "source_rows": len(rows),
        "eligible_rows": len(eligible),
        "empty_initial_rows": len(rows) - len(eligible),
    }
    return eligible, stats


def _select_cohort(eligible: list[dict[str, object]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for family in ("ordered", "sqs"):
        members = [row for row in eligible if row["size_family"] == family]
        members.sort(key=lambda row: (cohort_hash(str(row["fid"])), str(row["fid"])))
        if len(members) < SELECT_PER_FAMILY:
            raise ValueError(f"NEXT551 source lacks {family} rows")
        selected.extend(dict(row) for row in members[:SELECT_PER_FAMILY])
    assignment = balanced_partition_map(selected)
    for row in selected:
        row["partition"] = assignment[str(row["chemical_system"])]
        row["cohort_hash"] = cohort_hash(str(row["fid"]))
    selected.sort(key=lambda row: str(row["fid"]))
    return selected


def _sanitized_atoms(value: object, fid: str) -> Atoms:
    if not isinstance(value, dict):
        raise ValueError(f"NEXT551 initial structure is not a dictionary: {fid}")
    structure = Structure.from_dict(value)
    atoms = AseAtomsAdaptor.get_atoms(structure)
    clean = Atoms(
        numbers=np.asarray(atoms.numbers, dtype=int),
        positions=np.asarray(atoms.positions, dtype=float),
        cell=np.asarray(atoms.cell.array, dtype=float),
        pbc=True,
    )
    if (
        len(clean) < 1
        or not np.all(clean.pbc)
        or clean.calc is not None
        or clean.info
        or set(clean.arrays) != {"numbers", "positions"}
        or not np.isfinite(clean.positions).all()
        or not np.isfinite(clean.cell.array).all()
        or abs(float(np.linalg.det(clean.cell.array))) <= 1.0e-10
    ):
        raise ValueError(f"NEXT551 sanitized geometry differs: {fid}")
    return clean


def _geometry_digest(atoms: Atoms) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(atoms.numbers, dtype="<i8").tobytes())
    digest.update(np.asarray(atoms.cell.array, dtype="<f8").tobytes())
    digest.update(np.asarray(atoms.positions, dtype="<f8").tobytes())
    return digest.hexdigest()


def _extract_selected_initial_structures(
    source_path: Path, selected: list[dict[str, object]]
) -> tuple[dict[str, _ParsedFrame], dict[str, str]]:
    selected_ids = {str(row["fid"]) for row in selected}
    frames: dict[str, _ParsedFrame] = {}
    hashes: dict[str, str] = {}
    with source_path.open("rb") as handle:
        indices = _header_indices(handle)
        fid_index = indices["fid"]
        initial_index = indices["structure_ini_as_dict"]
        for record in handle:
            first, _present, field_count = _project_csv_record(
                record, copy_indices={fid_index}
            )
            if field_count != len(EXPECTED_HEADER):
                raise ValueError("NEXT551 extraction source field count differs")
            fid = first[fid_index].decode("utf-8")
            if fid not in selected_ids:
                continue
            projected, _presence, repeated_count = _project_csv_record(
                record, copy_indices={fid_index, initial_index}
            )
            if repeated_count != len(EXPECTED_HEADER):
                raise ValueError("NEXT551 selected extraction field count differs")
            raw = projected[initial_index].decode("utf-8")
            if not raw:
                raise ValueError(f"NEXT551 selected initial structure is empty: {fid}")
            try:
                value = ast.literal_eval(raw)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"NEXT551 initial structure encoding differs: {fid}") from exc
            atoms = _sanitized_atoms(value, fid)
            frames[fid] = _ParsedFrame(atoms, (), ())
            hashes[fid] = _geometry_digest(atoms)
    if set(frames) != selected_ids or set(hashes) != selected_ids:
        missing = sorted(selected_ids - set(frames))
        raise ValueError(f"NEXT551 selected initial structures are missing: {missing[:5]}")
    return frames, hashes


def _blind_gates(metadata: pd.DataFrame) -> dict[str, object]:
    cross = pd.crosstab(metadata["partition"], metadata["size_family"])
    partitions: dict[str, object] = {}
    for partition in ("development", "validation"):
        subset = metadata.loc[metadata["partition"].eq(partition)]
        partitions[partition] = {
            "rows": len(subset),
            "ordered": int((subset["size_family"] == "ordered").sum()),
            "sqs": int((subset["size_family"] == "sqs").sum()),
            "chemical_systems": int(subset["chemical_system"].nunique()),
        }
    unique_geometries = int(metadata["x0_geometry_sha256"].nunique())
    result = {
        "rows": len(metadata),
        "unique_fids": int(metadata["fid"].nunique()),
        "families": {
            family: int((metadata["size_family"] == family).sum())
            for family in ("ordered", "sqs")
        },
        "partitions": partitions,
        "unique_geometry_hashes": unique_geometries,
        "unique_geometry_fraction": unique_geometries / len(metadata),
    }
    result["passes"] = bool(
        len(metadata) == EXPECTED_SELECTED_ROWS
        and result["unique_fids"] == EXPECTED_SELECTED_ROWS
        and all(result["families"][family] == SELECT_PER_FAMILY for family in ("ordered", "sqs"))
        and all(
            partitions[partition]["rows"] >= 900
            and partitions[partition]["ordered"] >= 400
            and partitions[partition]["sqs"] >= 400
            and partitions[partition]["chemical_systems"] >= 100
            for partition in ("development", "validation")
        )
        and result["unique_geometry_fraction"] >= 0.99
    )
    return result


def build_initial_cohort(
    *, source_csv: Path, design_path: Path, output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    source_csv = Path(source_csv).resolve()
    design_path = Path(design_path).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if not source_csv.is_file() or not design_path.is_file():
        raise FileNotFoundError("NEXT551 input is missing")
    source_sha = _sha256(source_csv)
    design_sha = _sha256(design_path)
    if require_formal_inputs and (
        source_sha != SOURCE_SHA256
        or source_csv.stat().st_size != SOURCE_BYTES
        or design_sha != DESIGN_SHA256
    ):
        raise ValueError("NEXT551 formal input identity differs")
    eligible, source_stats = _scan_label_free_metadata(source_csv)
    selected = _select_cohort(eligible)
    frames, geometry_hashes = _extract_selected_initial_structures(source_csv, selected)
    for row in selected:
        row["x0_geometry_sha256"] = geometry_hashes[str(row["fid"])]
        row["input_role"] = "unrelaxed_x0_geometry_only"
        row["natoms_decoded"] = len(frames[str(row["fid"])].atoms)
        if row["natoms_decoded"] != row["nions"]:
            raise ValueError(f"NEXT551 decoded atom count differs: {row['fid']}")
    metadata = pd.DataFrame(selected)
    gates = _blind_gates(metadata)
    if gates["passes"] is not True:
        raise RuntimeError(f"NEXT551 label-blind gates failed: {gates}")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    try:
        metadata_path = staging / METADATA_NAME
        cohort_path = staging / COHORT_NAME
        geometry_path = staging / GEOMETRY_NAME
        metadata.to_parquet(metadata_path, index=False)
        cohort_path.write_bytes(_json_bytes(selected))
        _write_deterministic_archive(geometry_path, frames)
        outputs = {
            METADATA_NAME: _sha256(metadata_path),
            COHORT_NAME: _sha256(cohort_path),
            GEOMETRY_NAME: _sha256(geometry_path),
        }
        manifest = {
            "protocol": PROTOCOL,
            "source": {
                "url": "https://zenodo.org/records/10854500",
                "path": str(source_csv),
                "bytes": source_csv.stat().st_size,
                "sha256": source_sha,
                "published_md5": "4754d35ac163bb8804ef2f24ace659f7",
            },
            "design_sha256": design_sha,
            "source_counts": source_stats,
            "gates": gates,
            "schema_audit_fids_permanently_excluded": sorted(AUDIT_FIDS),
            "outputs_sha256": outputs,
            "executed_source_sha256": {
                "src/next551_hea_initial_cohort.py": source_hash
            },
            "input_columns_copied_or_decoded": [
                "fid", "reduced_formula", "chemical_system", "nelements", "NIONS",
                "structure_ini_as_dict for selected rows only",
            ],
            "endpoint_or_final_structure_columns_copied_or_decoded": False,
            "dft_energy_force_stress_values_opened": False,
            "final_or_relaxed_structures_opened": False,
            "model_or_proxy_potential_used": False,
            "coordinates_or_cell_modified": False,
            "selection_uses_endpoint_values": False,
            "next552_feature_freeze_authorized": True,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_csv) != source_sha or _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT551 source changed before publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--design-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_initial_cohort(
        source_csv=args.source_csv,
        design_path=args.design_path,
        output_dir=args.output_dir,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_project_csv_record",
    "balanced_partition_map",
    "build_initial_cohort",
    "cohort_hash",
]
