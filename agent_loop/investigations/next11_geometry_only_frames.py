"""Build and consume a sealed geometry-only x0 archive for next11.

The builder reads only the frozen x0 frame archive plus the two tables needed
to select ``development_gate & strict_x0_ok`` rows.  Non-geometry extxyz
tokens are segmented only to locate fields and enforce row widths; their
values are never converted or exported.  The consumer intentionally does not
open any of the raw input paths recorded by the manifest.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import sys
import tempfile
from typing import Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd
from ase import Atoms
from ase.data import atomic_numbers


PROTOCOL = "2026-08-02-next11-geometry-only-frames-v1"
OUTPUT_ARCHIVE_NAME = "geometry_only_frames.zip"
MANIFEST_NAME = "MANIFEST.json"
EXECUTED_SOURCE_RELATIVE = "src/next11_geometry_only_frames.py"
GEOMETRY_SCHEMA = {
    "archive_member_pattern": "{sid}.extxyz",
    "comment_keys": ["Lattice", "Properties", "pbc"],
    "properties": ["species:S:1", "pos:R:3"],
    "retained_fields": ["species", "pos", "Lattice", "pbc"],
    "dropped_values_preserved": False,
    "ase_metadata_preserved": False,
}
_MANIFEST_KEYS = {
    "protocol",
    "mode",
    "endpoint_label_artifacts_opened",
    "raw_x0_archive_bytes_read",
    "raw_x0_nongeometry_values_converted_or_exported",
    "input_role",
    "selection",
    "geometry_schema",
    "dropped_field_names",
    "inputs_sha256",
    "executed_source_sha256",
    "integrity",
    "counts",
    "sid_order_sha256",
    "outputs_sha256",
    "scientific_improvement_claim",
}
_INPUT_ROLES = ("raw_frames", "committee_features", "threshold_roles")
_SID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_FIELD_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]*")
_RAW_REQUIRED_COMMENT_KEYS = frozenset({"Lattice", "Properties", "pbc"})
_CANONICAL_PROPERTIES = "species:S:1:pos:R:3"


@dataclass(frozen=True, slots=True)
class _Snapshot:
    path: Path
    data: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class _ParsedFrame:
    atoms: Atoms
    dropped_comment_fields: tuple[str, ...]
    dropped_atom_properties: tuple[str, ...]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(path: Path) -> _Snapshot:
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    data = candidate.read_bytes()
    return _Snapshot(candidate, data, _sha256_bytes(data))


def _hash_record(snapshot: _Snapshot) -> dict[str, str]:
    return {"path": str(snapshot.path.resolve()), "sha256": snapshot.sha256}


def _strict_json_document(data: bytes, *, role: str) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{role} contains nonstandard JSON constant {value}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{role} contains duplicate key {key}")
            result[key] = value
        return result

    try:
        document = json.loads(
            data.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role} JSON") from exc
    if not isinstance(document, dict):
        raise ValueError(f"invalid {role} document")
    return document


def _exact_string_column(table: pd.DataFrame, column: str, *, role: str) -> None:
    if table[column].isna().any() or not table[column].map(
        lambda value: type(value) is str and bool(value)
    ).all():
        raise ValueError(f"{role} {column} must contain nonempty exact strings")


def _read_projected_parquet(
    data: bytes,
    *,
    columns: Sequence[str],
    role: str,
) -> pd.DataFrame:
    try:
        return pd.read_parquet(io.BytesIO(data), columns=list(columns))
    except Exception as exc:
        raise ValueError(f"could not read required {role} columns") from exc


def _selected_strict_sids(
    features_data: bytes,
    roles_data: bytes,
) -> tuple[list[str], dict[str, int]]:
    features = _read_projected_parquet(
        features_data,
        columns=("sid", "rk", "stage", "strict_x0_ok"),
        role="committee features",
    )
    roles = _read_projected_parquet(
        roles_data,
        columns=("sid", "rk", "stage", "threshold_role"),
        role="threshold roles",
    )
    for table, role, columns in (
        (features, "committee features", ("sid", "rk", "stage")),
        (roles, "threshold roles", ("sid", "rk", "stage", "threshold_role")),
    ):
        for column in columns:
            _exact_string_column(table, column, role=role)
        if table["sid"].duplicated().any():
            raise ValueError(f"{role} sid values must be unique")
    if not features["strict_x0_ok"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise ValueError("committee features strict_x0_ok must be exactly boolean")
    allowed_roles = {"threshold_fit", "development_gate"}
    if not set(roles["threshold_role"]).issubset(allowed_roles):
        raise ValueError("threshold roles contain unsupported threshold_role")

    selected_roles = roles.loc[
        roles["threshold_role"].eq("development_gate"),
        ["sid", "rk", "stage"],
    ].copy()
    if selected_roles.empty:
        raise ValueError("development_gate selection must be nonempty")
    selected = selected_roles.merge(
        features[["sid", "rk", "stage", "strict_x0_ok"]],
        on="sid",
        how="left",
        suffixes=("_role", "_feature"),
        validate="one_to_one",
        indicator=True,
    )
    if not selected["_merge"].eq("both").all():
        raise ValueError("development_gate sid coverage differs from features")
    if not selected["rk_role"].eq(selected["rk_feature"]).all():
        raise ValueError("development_gate rk values differ between inputs")
    if not selected["stage_role"].eq(selected["stage_feature"]).all():
        raise ValueError("development_gate stage values differ between inputs")
    if not selected["stage_role"].eq("threshold_calibration").all():
        raise ValueError("development_gate rows must be threshold_calibration")
    selected = selected.sort_values("sid", kind="stable").reset_index(drop=True)
    strict_sids = selected.loc[selected["strict_x0_ok"], "sid"].tolist()
    if not strict_sids:
        raise ValueError("strict development_gate selection must be nonempty")
    _canonical_expected_sids(strict_sids)
    return strict_sids, {
        "feature_rows": len(features),
        "role_assignment_rows": len(roles),
        "development_gate_rows": len(selected),
        "strict_rows": len(strict_sids),
    }


def _canonical_expected_sids(expected_sids: Sequence[str]) -> tuple[str, ...]:
    values = list(expected_sids)
    if not values:
        raise ValueError("expected_sids must be nonempty")
    if any(type(sid) is not str or _SID.fullmatch(sid) is None for sid in values):
        raise ValueError("expected_sids must contain safe nonempty exact sid strings")
    if len(values) != len(set(values)):
        raise ValueError("expected_sids must be unique")
    return tuple(sorted(values))


def _comment_fields(
    comment: str,
    *,
    strict_output: bool,
) -> tuple[dict[str, str], tuple[str, ...]]:
    try:
        tokens = shlex.split(comment, posix=True)
    except ValueError as exc:
        raise ValueError("invalid extxyz comment quoting") from exc
    required: dict[str, str] = {}
    extra_names: set[str] = set()
    seen: set[str] = set()
    for token in tokens:
        if "=" not in token:
            raise ValueError("extxyz comment tokens must be named fields")
        key, value = token.split("=", 1)
        if _FIELD_NAME.fullmatch(key) is None:
            raise ValueError("invalid extxyz comment field name")
        if key in seen:
            raise ValueError(f"duplicate extxyz comment field {key}")
        seen.add(key)
        if key in _RAW_REQUIRED_COMMENT_KEYS:
            required[key] = value
        else:
            extra_names.add(key)
    if set(required) != _RAW_REQUIRED_COMMENT_KEYS:
        missing = sorted(_RAW_REQUIRED_COMMENT_KEYS - set(required))
        raise ValueError(f"extxyz comment is missing required geometry fields: {missing}")
    if strict_output and extra_names:
        raise ValueError("geometry-only comment keys are not exact")
    return required, tuple(sorted(extra_names))


def _property_schema(value: str) -> tuple[tuple[str, str, int, int], ...]:
    pieces = value.split(":")
    if not pieces or len(pieces) % 3:
        raise ValueError("invalid extxyz Properties schema")
    fields: list[tuple[str, str, int, int]] = []
    seen: set[str] = set()
    offset = 0
    for index in range(0, len(pieces), 3):
        name, kind, width_text = pieces[index : index + 3]
        if _FIELD_NAME.fullmatch(name) is None or name in seen:
            raise ValueError("invalid or duplicate extxyz property name")
        if kind not in {"S", "R", "I", "L"}:
            raise ValueError("unsupported extxyz property type")
        try:
            width = int(width_text)
        except ValueError as exc:
            raise ValueError("invalid extxyz property width") from exc
        if width <= 0 or str(width) != width_text:
            raise ValueError("invalid extxyz property width")
        fields.append((name, kind, width, offset))
        seen.add(name)
        offset += width
    by_name = {name: (kind, width) for name, kind, width, _ in fields}
    if by_name.get("species") != ("S", 1):
        raise ValueError("Properties must define species:S:1")
    if by_name.get("pos") != ("R", 3):
        raise ValueError("Properties must define pos:R:3")
    return tuple(fields)


def _parse_pbc(value: str) -> np.ndarray:
    tokens = value.split()
    if len(tokens) != 3:
        raise ValueError("pbc must contain exactly three flags")
    parsed: list[bool] = []
    for token in tokens:
        normalized = token.casefold()
        if normalized in {"t", "true", "1"}:
            parsed.append(True)
        elif normalized in {"f", "false", "0"}:
            parsed.append(False)
        else:
            raise ValueError("invalid pbc flag")
    result = np.asarray(parsed, dtype=bool)
    if not np.all(result):
        raise ValueError("geometry-only PHSC input must be fully periodic")
    return result


def _finite_cell(value: str) -> np.ndarray:
    try:
        values = [float(token) for token in value.split()]
    except ValueError as exc:
        raise ValueError("Lattice must contain numeric values") from exc
    if len(values) != 9:
        raise ValueError("Lattice must contain exactly nine values")
    cell = np.asarray(values, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(cell)):
        raise ValueError("Lattice values must be finite")
    determinant = float(np.linalg.det(cell))
    if not math.isfinite(determinant) or determinant == 0.0:
        raise ValueError("Lattice must be nonsingular")
    return cell


def _parse_frame(data: bytes, *, strict_output: bool) -> _ParsedFrame:
    try:
        text = data.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("extxyz member is not strict UTF-8") from exc
    lines = text.splitlines()
    if len(lines) < 2:
        raise ValueError("extxyz member lacks atom count or comment")
    try:
        natoms = int(lines[0])
    except ValueError as exc:
        raise ValueError("invalid extxyz atom count") from exc
    if natoms < 1:
        raise ValueError("extxyz atom count must be positive")
    if len(lines) != natoms + 2:
        raise ValueError("extxyz member must contain a single frame")
    fields, dropped_comment = _comment_fields(lines[1], strict_output=strict_output)
    schema = _property_schema(fields["Properties"])
    if strict_output and fields["Properties"] != _CANONICAL_PROPERTIES:
        raise ValueError("geometry-only Properties schema is not exact")
    cell = _finite_cell(fields["Lattice"])
    pbc = _parse_pbc(fields["pbc"])
    total_width = sum(width for _name, _kind, width, _offset in schema)
    species_field = next(item for item in schema if item[0] == "species")
    position_field = next(item for item in schema if item[0] == "pos")
    species_offset = species_field[3]
    position_offset = position_field[3]
    species: list[str] = []
    positions: list[list[float]] = []
    for line in lines[2:]:
        try:
            tokens = shlex.split(line, posix=True)
        except ValueError as exc:
            raise ValueError("invalid extxyz atom-line quoting") from exc
        if len(tokens) != total_width:
            raise ValueError("atom line width differs from Properties schema")
        symbol = tokens[species_offset]
        if symbol not in atomic_numbers or atomic_numbers[symbol] <= 0:
            raise ValueError(f"invalid chemical species {symbol!r}")
        try:
            xyz = [float(value) for value in tokens[position_offset : position_offset + 3]]
        except ValueError as exc:
            raise ValueError("positions must contain numeric values") from exc
        species.append(symbol)
        positions.append(xyz)
    position_array = np.asarray(positions, dtype=np.float64)
    if position_array.shape != (natoms, 3) or not np.all(np.isfinite(position_array)):
        raise ValueError("positions must be finite N x 3 values")
    atoms = Atoms(
        symbols=species,
        positions=position_array,
        cell=cell,
        pbc=pbc,
    )
    dropped_atom_property_names = tuple(
        sorted(name for name, _kind, _width, _offset in schema if name not in {"species", "pos"})
    )
    if strict_output and dropped_atom_property_names:
        raise ValueError("geometry-only Properties schema contains extra properties")
    return _ParsedFrame(atoms, dropped_comment, dropped_atom_property_names)


def _safe_raw_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".extxyz":
            raise ValueError("raw frame archive contains an unsafe or non-extxyz member")
        sid = path.stem
        if _SID.fullmatch(sid) is None:
            raise ValueError("raw frame archive contains an invalid member sid")
        if sid in members:
            raise ValueError(f"raw frame archive has duplicate member stem sid: {sid}")
        if info.flag_bits & 0x1:
            raise ValueError("raw frame archive contains an encrypted member")
        members[sid] = info
    if not members:
        raise ValueError("raw frame archive contains no frames")
    return members


def _selected_raw_frames(
    archive_data: bytes,
    expected_sids: Sequence[str],
) -> tuple[dict[str, _ParsedFrame], int]:
    try:
        context = zipfile.ZipFile(io.BytesIO(archive_data))
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("invalid raw x0 frame archive") from exc
    selected: dict[str, _ParsedFrame] = {}
    with context as archive:
        members = _safe_raw_members(archive)
        for sid in _canonical_expected_sids(expected_sids):
            info = members.get(sid)
            if info is None:
                raise ValueError(f"strict development_gate sid lacks x0 frame: {sid}")
            try:
                selected[sid] = _parse_frame(archive.read(info), strict_output=False)
            except Exception as exc:
                raise ValueError(f"invalid strict x0 frame for sid {sid}: {exc}") from exc
    return selected, len(members)


def _format_float(value: float) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("geometry contains a nonfinite value")
    if numeric == 0.0:
        return "0"
    return format(numeric, ".17g")


def _canonical_frame(atoms: Atoms) -> bytes:
    if atoms.calc is not None or atoms.info or set(atoms.arrays) != {"numbers", "positions"}:
        raise ValueError("geometry-only atoms contain calculator, info, or extra arrays")
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    cell = np.asarray(atoms.cell, dtype=np.float64)
    pbc = np.asarray(atoms.pbc, dtype=bool)
    if (
        len(atoms) < 1
        or numbers.shape != (len(atoms),)
        or np.any(numbers <= 0)
        or positions.shape != (len(atoms), 3)
        or cell.shape != (3, 3)
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(cell))
        or not np.all(pbc)
    ):
        raise ValueError("invalid geometry-only atoms")
    lattice = " ".join(_format_float(value) for value in cell.reshape(-1))
    comment = (
        f'Lattice="{lattice}" Properties="{_CANONICAL_PROPERTIES}" '
        'pbc="T T T"'
    )
    symbols = atoms.get_chemical_symbols()
    lines = [str(len(atoms)), comment]
    lines.extend(
        " ".join([symbol, *(_format_float(value) for value in xyz)])
        for symbol, xyz in zip(symbols, positions, strict=True)
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_deterministic_archive(
    path: Path,
    frames: Mapping[str, _ParsedFrame],
) -> None:
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        archive.comment = b""
        for sid in sorted(frames):
            info = zipfile.ZipInfo(f"{sid}.extxyz", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.extra = b""
            info.comment = b""
            archive.writestr(
                info,
                _canonical_frame(frames[sid].atoms),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def _sid_order_sha256(sids: Sequence[str]) -> str:
    return _sha256_bytes(("\n".join(sids) + "\n").encode("utf-8"))


def _atomic_publish_directory_no_replace(source: Path, target: Path) -> None:
    unsupported_errno = getattr(errno, "ENOTSUP", errno.EINVAL)
    if sys.platform != "linux":
        raise OSError(
            unsupported_errno,
            "atomic no-replace directory publication is unsupported",
            str(target),
        )
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as exc:
        raise OSError(
            unsupported_errno,
            "atomic no-replace directory publication is unsupported",
            str(target),
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(error, os.strerror(error), str(target))
    raise OSError(error, os.strerror(error), str(target))


def _validate_hash_record(value: object, *, role: str) -> None:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"manifest {role} input hash record is invalid")
    path = value.get("path")
    digest = value.get("sha256")
    if type(path) is not str or not path:
        raise ValueError(f"manifest {role} input path is invalid")
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"manifest {role} input sha256 is invalid")


def _validated_manifest(
    manifest: Mapping[str, object],
    *,
    archive_path: Path,
    expected_sids: tuple[str, ...],
    total_atoms: int,
) -> None:
    if set(manifest) != _MANIFEST_KEYS:
        raise ValueError("geometry-only manifest top-level schema is not exact")
    expected_scalars = {
        "protocol": PROTOCOL,
        "mode": "development_gate",
        "endpoint_label_artifacts_opened": False,
        "raw_x0_archive_bytes_read": True,
        "raw_x0_nongeometry_values_converted_or_exported": False,
        "input_role": "unrelaxed_x0_geometry_only",
        "scientific_improvement_claim": False,
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected or type(manifest.get(key)) is not type(expected):
            raise ValueError(f"geometry-only manifest {key} mismatch")
    if manifest.get("selection") != {
        "stage": "threshold_calibration",
        "threshold_role": "development_gate",
        "strict_x0_ok": True,
    }:
        raise ValueError("geometry-only manifest selection mismatch")
    if manifest.get("geometry_schema") != GEOMETRY_SCHEMA:
        raise ValueError("geometry-only manifest schema mismatch")
    dropped = manifest.get("dropped_field_names")
    if not isinstance(dropped, dict) or set(dropped) != {"comment", "atom_properties"}:
        raise ValueError("geometry-only dropped field-name schema mismatch")
    for role in ("comment", "atom_properties"):
        values = dropped.get(role)
        if (
            not isinstance(values, list)
            or any(type(value) is not str or _FIELD_NAME.fullmatch(value) is None for value in values)
            or values != sorted(set(values))
        ):
            raise ValueError("geometry-only dropped field names must be sorted unique names")
    inputs = manifest.get("inputs_sha256")
    if not isinstance(inputs, dict) or tuple(inputs) != tuple(sorted(_INPUT_ROLES)):
        if not isinstance(inputs, dict) or set(inputs) != set(_INPUT_ROLES):
            raise ValueError("geometry-only manifest input roles mismatch")
    assert isinstance(inputs, dict)
    for role in _INPUT_ROLES:
        _validate_hash_record(inputs.get(role), role=role)
    source_path = Path(__file__).resolve()
    if manifest.get("executed_source_sha256") != {
        EXECUTED_SOURCE_RELATIVE: _sha256_file(source_path)
    }:
        raise ValueError("geometry-only executed source hash closure mismatch")
    if manifest.get("integrity") != {"prepublish_rehash": "passed"}:
        raise ValueError("geometry-only manifest integrity status mismatch")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, dict) or set(outputs) != {OUTPUT_ARCHIVE_NAME}:
        raise ValueError("geometry-only output hash schema mismatch")
    if outputs[OUTPUT_ARCHIVE_NAME] != _sha256_file(archive_path):
        raise ValueError("geometry-only archive hash mismatch")
    if manifest.get("sid_order_sha256") != _sid_order_sha256(expected_sids):
        raise ValueError("geometry-only sid order hash mismatch")
    counts = manifest.get("counts")
    count_keys = {
        "feature_rows",
        "role_assignment_rows",
        "development_gate_rows",
        "strict_rows",
        "output_frames",
        "total_atoms",
        "raw_archive_file_members",
    }
    if not isinstance(counts, dict) or set(counts) != count_keys:
        raise ValueError("geometry-only manifest count schema mismatch")
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("geometry-only manifest counts must be nonnegative integers")
    if counts["strict_rows"] != len(expected_sids) or counts["output_frames"] != len(expected_sids):
        raise ValueError("geometry-only manifest frame counts mismatch")
    if counts["total_atoms"] != total_atoms:
        raise ValueError("geometry-only manifest atom count mismatch")
    if counts["development_gate_rows"] < counts["strict_rows"]:
        raise ValueError("geometry-only manifest selection counts are inconsistent")


def _load_archive_only(
    archive_path: Path,
    expected_sids: tuple[str, ...],
) -> tuple[list[str], list[Atoms]]:
    path = Path(archive_path)
    if path.name != OUTPUT_ARCHIVE_NAME or not path.is_file():
        raise ValueError("geometry-only archive path/name is invalid")
    try:
        context = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("invalid geometry-only archive") from exc
    structures: list[Atoms] = []
    with context as archive:
        if archive.comment:
            raise ValueError("geometry-only archive comment must be empty")
        infos = archive.infolist()
        if any(info.is_dir() for info in infos):
            raise ValueError("geometry-only archive must not contain directories")
        expected_names = [f"{sid}.extxyz" for sid in expected_sids]
        names = [info.filename for info in infos]
        if names != expected_names:
            if any(PurePosixPath(name).parent != PurePosixPath(".") for name in names):
                raise ValueError("geometry-only members must be root-level")
            raise ValueError("geometry-only archive exact sid coverage mismatch")
        for info in infos:
            if PurePosixPath(info.filename).parent != PurePosixPath("."):
                raise ValueError("geometry-only members must be root-level")
            if info.date_time != (1980, 1, 1, 0, 0, 0):
                raise ValueError("geometry-only member timestamp is not canonical")
            if (
                info.extra
                or info.comment
                or info.compress_type != zipfile.ZIP_DEFLATED
                or info.create_system != 3
                or info.external_attr != 0o100644 << 16
            ):
                raise ValueError("geometry-only member metadata is not canonical")
            try:
                parsed = _parse_frame(archive.read(info), strict_output=True)
            except Exception as exc:
                raise ValueError(
                    f"invalid geometry-only frame for {PurePosixPath(info.filename).stem}: {exc}"
                ) from exc
            atoms = parsed.atoms
            if atoms.calc is not None or atoms.info or set(atoms.arrays) != {"numbers", "positions"}:
                raise ValueError("geometry-only loader retained forbidden ASE metadata")
            structures.append(atoms)
    return list(expected_sids), structures


def load_geometry_only_archive(
    *,
    archive_path: Path,
    manifest_path: Path,
    expected_sids: Sequence[str],
) -> tuple[list[str], list[Atoms]]:
    """Load a sealed sanitized archive without touching its recorded raw inputs."""

    canonical_sids = _canonical_expected_sids(expected_sids)
    manifest_candidate = Path(manifest_path)
    if manifest_candidate.name != MANIFEST_NAME or not manifest_candidate.is_file():
        raise ValueError("geometry-only manifest path/name is invalid")
    manifest = _strict_json_document(
        manifest_candidate.read_bytes(), role="geometry-only manifest"
    )
    sids, structures = _load_archive_only(Path(archive_path), canonical_sids)
    _validated_manifest(
        manifest,
        archive_path=Path(archive_path),
        expected_sids=canonical_sids,
        total_atoms=sum(len(atoms) for atoms in structures),
    )
    return sids, structures


def validate_geometry_only_archive(
    *,
    archive_path: Path,
    manifest_path: Path,
    expected_sids: Sequence[str],
) -> tuple[str, ...]:
    """Validate the sealed archive/manifest closure and return canonical sids."""

    sids, _structures = load_geometry_only_archive(
        archive_path=archive_path,
        manifest_path=manifest_path,
        expected_sids=expected_sids,
    )
    return tuple(sids)


def build_geometry_only_frames(
    *,
    raw_frames_zip_path: Path,
    committee_features_path: Path,
    role_assignments_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Seal the strict development-gate subset as a geometry-only archive."""

    target = Path(output_dir)
    if os.path.lexists(target):
        raise FileExistsError(target)
    snapshots = {
        "raw_frames": _snapshot(raw_frames_zip_path),
        "committee_features": _snapshot(committee_features_path),
        "threshold_roles": _snapshot(role_assignments_path),
    }
    source_path = Path(__file__).resolve()
    source_sha256 = _sha256_file(source_path)
    strict_sids, selection_counts = _selected_strict_sids(
        snapshots["committee_features"].data,
        snapshots["threshold_roles"].data,
    )
    frames, raw_member_count = _selected_raw_frames(
        snapshots["raw_frames"].data,
        strict_sids,
    )
    dropped_comment = sorted(
        {
            name
            for frame in frames.values()
            for name in frame.dropped_comment_fields
        }
    )
    dropped_atom_property_names = sorted(
        {
            name
            for frame in frames.values()
            for name in frame.dropped_atom_properties
        }
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    archive_path = staging / OUTPUT_ARCHIVE_NAME
    manifest_path = staging / MANIFEST_NAME
    try:
        _write_deterministic_archive(archive_path, frames)

        def verify_unchanged() -> None:
            for role, snapshot in snapshots.items():
                if _sha256_file(snapshot.path) != snapshot.sha256:
                    raise RuntimeError(f"input {role} changed after initial snapshot")
            if _sha256_file(source_path) != source_sha256:
                raise RuntimeError("executed sanitizer source changed during build")

        verify_unchanged()
        canonical_sids = tuple(sorted(strict_sids))
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "mode": "development_gate",
            "endpoint_label_artifacts_opened": False,
            "raw_x0_archive_bytes_read": True,
            "raw_x0_nongeometry_values_converted_or_exported": False,
            "input_role": "unrelaxed_x0_geometry_only",
            "selection": {
                "stage": "threshold_calibration",
                "threshold_role": "development_gate",
                "strict_x0_ok": True,
            },
            "geometry_schema": dict(GEOMETRY_SCHEMA),
            "dropped_field_names": {
                "comment": dropped_comment,
                "atom_properties": dropped_atom_property_names,
            },
            "inputs_sha256": {
                role: _hash_record(snapshots[role]) for role in sorted(snapshots)
            },
            "executed_source_sha256": {
                EXECUTED_SOURCE_RELATIVE: source_sha256
            },
            "integrity": {"prepublish_rehash": "passed"},
            "counts": {
                **selection_counts,
                "output_frames": len(frames),
                "total_atoms": sum(len(frame.atoms) for frame in frames.values()),
                "raw_archive_file_members": raw_member_count,
            },
            "sid_order_sha256": _sid_order_sha256(canonical_sids),
            "outputs_sha256": {
                OUTPUT_ARCHIVE_NAME: _sha256_file(archive_path)
            },
            "scientific_improvement_claim": False,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        validate_geometry_only_archive(
            archive_path=archive_path,
            manifest_path=manifest_path,
            expected_sids=canonical_sids,
        )
        verify_unchanged()
        _atomic_publish_directory_no_replace(staging, target)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-frames-zip", type=Path, required=True)
    parser.add_argument("--committee-features", type=Path, required=True)
    parser.add_argument("--role-assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_geometry_only_frames(
        raw_frames_zip_path=args.raw_frames_zip,
        committee_features_path=args.committee_features,
        role_assignments_path=args.role_assignments,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
