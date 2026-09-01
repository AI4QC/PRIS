#!/usr/bin/env python3
"""Sanitize OMC25 x0 geometry and, separately, derive DFT-only endpoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import zlib

from ase import Atoms
from ase.io.jsonio import decode
import lmdb
import numpy as np
import pandas as pd

from src.next11_geometry_only_frames import _ParsedFrame, _write_deterministic_archive
from src.next19_feature_build import _publish_directory_no_replace, _sha256


PROTOCOL = "2026-08-03-next26-omc25-x0-sanitize-v1"
ENDPOINT_PROTOCOL = "2026-08-03-next26-omc25-dft-endpoint-v1"
METADATA_NAME = "holdout_metadata.parquet"
GEOMETRY_NAME = "geometry_only_frames.zip"
MANIFEST_NAME = "MANIFEST.json"
ENDPOINT_NAME = "next26_dft_endpoints.parquet"
FORCE_MAX_THRESHOLD = 1.0
FORCE_RMS_THRESHOLD = 0.40
ENERGY_DROP_THRESHOLD = 0.040
STRESS_NORM_THRESHOLD = 0.030


@dataclass(frozen=True, slots=True)
class TrajectoryRef:
    material_id: str
    csd_refcode: str
    z_value: int
    genarris_step: str
    xtal_id: str
    lmdb_keys: tuple[int, ...]
    frame_indices: tuple[int, ...]


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _skip_space(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _string_end(text: str, position: int) -> int:
    if position >= len(text) or text[position] != '"':
        raise ValueError("expected JSON string")
    position += 1
    while position < len(text):
        if text[position] == "\\":
            position += 2
        elif text[position] == '"':
            return position + 1
        else:
            position += 1
    raise ValueError("unterminated JSON string")


def _value_end(text: str, position: int) -> int:
    """Locate a JSON value without converting any of its numeric contents."""

    position = _skip_space(text, position)
    if position >= len(text):
        raise ValueError("missing JSON value")
    if text[position] == '"':
        return _string_end(text, position)
    if text[position] in "[{":
        pairs = {"[": "]", "{": "}"}
        stack = [pairs[text[position]]]
        position += 1
        while position < len(text) and stack:
            token = text[position]
            if token == '"':
                position = _string_end(text, position)
                continue
            if token in pairs:
                stack.append(pairs[token])
            elif token in "]}":
                if token != stack.pop():
                    raise ValueError("mismatched JSON container")
            position += 1
        if stack:
            raise ValueError("unterminated JSON container")
        return position
    while position < len(text) and text[position] not in ",}":
        position += 1
    return position


def _top_level_fields(payload: bytes, wanted: Iterable[str]) -> dict[str, str]:
    """Return raw top-level values while skipping every unrequested value."""

    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("record is not UTF-8 JSON") from exc
    wanted_set = set(wanted)
    result: dict[str, str] = {}
    position = _skip_space(text, 0)
    if position >= len(text) or text[position] != "{":
        raise ValueError("record must be a JSON object")
    position += 1
    decoder = json.JSONDecoder()
    while True:
        position = _skip_space(text, position)
        if position < len(text) and text[position] == "}":
            position += 1
            break
        try:
            key, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSON object key") from exc
        if type(key) is not str:
            raise ValueError("JSON object key must be a string")
        position = _skip_space(text, position)
        if position >= len(text) or text[position] != ":":
            raise ValueError("JSON object key lacks colon")
        start = _skip_space(text, position + 1)
        end = _value_end(text, start)
        if key in wanted_set:
            if key in result:
                raise ValueError(f"duplicate requested JSON field {key}")
            result[key] = text[start:end]
        position = _skip_space(text, end)
        if position >= len(text):
            raise ValueError("unterminated JSON object")
        if text[position] == ",":
            position += 1
            continue
        if text[position] == "}":
            position += 1
            break
        raise ValueError("invalid JSON object separator")
    if _skip_space(text, position) != len(text):
        raise ValueError("trailing JSON data")
    missing = wanted_set - set(result)
    if missing:
        raise ValueError(f"record lacks required fields: {sorted(missing)}")
    return result


def _metadata_from_payload(payload: bytes) -> tuple[tuple[str, int, str, str], int]:
    raw = _top_level_fields(payload, {"data"})["data"]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid OMC25 metadata") from exc
    if not isinstance(data, dict):
        raise ValueError("OMC25 data field must be an object")
    try:
        refcode = str(data["csd_refcode"])
        z_value = int(data["z_value"])
        step = str(data["genarris_step"])
        xtal_id = str(data["xtal.id"])
        sid = str(data["sid"])
        frame = int(sid.rsplit("-", 1)[1])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("invalid OMC25 trajectory metadata") from exc
    if (
        not refcode
        or z_value <= 0
        or step not in {"gener", "press"}
        or not xtal_id
        or frame < 0
        or sid != f"{refcode}-{z_value}-{step}-{xtal_id}-{frame}"
    ):
        raise ValueError("inconsistent OMC25 sid metadata")
    return (refcode, z_value, step, xtal_id), frame


def project_x0_payload(payload: bytes) -> Atoms:
    """Decode only numbers, positions, PBC, and cell from one OMC25 record."""

    fields = _top_level_fields(payload, {"numbers", "positions", "pbc", "cell"})
    try:
        numbers = np.asarray(decode(fields["numbers"]), dtype=int)
        positions = np.asarray(decode(fields["positions"]), dtype=float)
        pbc = np.asarray(decode(fields["pbc"]), dtype=bool)
        cell = np.asarray(decode(fields["cell"]), dtype=float)
    except Exception as exc:
        raise ValueError("invalid OMC25 geometry fields") from exc
    if (
        numbers.ndim != 1
        or len(numbers) < 1
        or np.any(numbers < 1)
        or np.any(numbers > 118)
        or positions.shape != (len(numbers), 3)
        or cell.shape != (3, 3)
        or pbc.shape != (3,)
        or not np.all(pbc)
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(cell))
        or not math.isfinite(float(np.linalg.det(cell)))
        or abs(float(np.linalg.det(cell))) < 1e-12
    ):
        raise ValueError("invalid fully periodic OMC25 geometry")
    return Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)


def _catalogue(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ValueError("invalid OMC25 starting-crystal catalogue") from exc
    required = {
        "csd_refcode",
        "z_value",
        "genarris_step",
        "xtal.id",
        "split",
        "nframes",
        "xtal.natoms",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"catalogue lacks columns: {sorted(required-set(frame.columns))}")
    frame = frame.loc[frame["split"].astype(str).eq("val")].copy()
    for column in ("z_value", "nframes", "xtal.natoms"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    keys = ["csd_refcode", "z_value", "genarris_step", "xtal.id"]
    if frame.empty or frame.duplicated(keys).any() or (frame["nframes"] <= 0).any():
        raise ValueError("catalogue trajectory identities are invalid")
    return frame


def scan_complete_trajectories(
    *, db_path: Path, catalogue_path: Path
) -> list[TrajectoryRef]:
    """Metadata-only scan; endpoint numeric values are skipped, not parsed."""

    catalogue = _catalogue(Path(catalogue_path))
    indexed = catalogue.set_index(
        ["csd_refcode", "z_value", "genarris_step", "xtal.id"]
    )
    groups: dict[tuple[str, int, str, str], list[tuple[int, int]]] = defaultdict(list)
    env = lmdb.open(
        str(Path(db_path)), subdir=False, readonly=True, lock=False, readahead=False
    )
    try:
        with env.begin() as transaction:
            for key, value in transaction.cursor():
                if key == b"nextid":
                    continue
                try:
                    numeric_key = int(key)
                except ValueError as exc:
                    raise ValueError(f"unexpected non-record LMDB key {key!r}") from exc
                identity, frame = _metadata_from_payload(zlib.decompress(value))
                if identity not in indexed.index:
                    raise ValueError(f"LMDB trajectory is absent from val catalogue: {identity}")
                groups[identity].append((frame, numeric_key))
    finally:
        env.close()

    result: list[TrajectoryRef] = []
    for identity, members in groups.items():
        members.sort()
        frames = tuple(item[0] for item in members)
        keys = tuple(item[1] for item in members)
        if len(frames) != len(set(frames)) or len(keys) != len(set(keys)):
            raise ValueError(f"duplicate trajectory frame or LMDB key: {identity}")
        expected = int(indexed.loc[identity, "nframes"])
        if len(frames) != expected:
            continue
        if not frames or frames[0] != 0 or any(b <= a for a, b in zip(frames, frames[1:])):
            raise ValueError(f"complete trajectory lacks chronological frame zero: {identity}")
        refcode, z_value, step, xtal_id = identity
        result.append(
            TrajectoryRef(
                material_id=f"{refcode}-{z_value}-{step}-{xtal_id}",
                csd_refcode=refcode,
                z_value=z_value,
                genarris_step=step,
                xtal_id=xtal_id,
                lmdb_keys=keys,
                frame_indices=frames,
            )
        )
    result.sort(key=lambda row: row.material_id)
    if not result:
        raise ValueError("LMDB contains no complete OMC25 trajectory")
    if len({row.material_id for row in result}) != len(result):
        raise ValueError("material IDs are not unique")
    return result


def sanitize_x0(
    *,
    db_path: Path,
    catalogue_path: Path,
    output_dir: Path,
    exclude_refcodes: Iterable[str] = (),
) -> dict[str, object]:
    """Publish one raw frame-zero geometry per complete eligible trajectory."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {"aselmdb": Path(db_path).resolve(), "catalogue": Path(catalogue_path).resolve()}
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    input_hashes = {role: _sha256(path) for role, path in paths.items()}
    complete = scan_complete_trajectories(db_path=paths["aselmdb"], catalogue_path=paths["catalogue"])
    excluded = frozenset(str(value) for value in exclude_refcodes)
    selected = [row for row in complete if row.csd_refcode not in excluded]
    if not selected:
        raise ValueError("x0 selection is empty")
    frames: dict[str, _ParsedFrame] = {}
    env = lmdb.open(str(paths["aselmdb"]), subdir=False, readonly=True, lock=False, readahead=False)
    try:
        with env.begin() as transaction:
            for row in selected:
                compressed = transaction.get(str(row.lmdb_keys[0]).encode())
                if compressed is None:
                    raise ValueError(f"missing x0 LMDB key for {row.material_id}")
                payload = zlib.decompress(compressed)
                identity, frame = _metadata_from_payload(payload)
                expected = (row.csd_refcode, row.z_value, row.genarris_step, row.xtal_id)
                if identity != expected or frame != 0:
                    raise ValueError(f"x0 identity differs for {row.material_id}")
                atoms = project_x0_payload(payload)
                frames[row.material_id] = _ParsedFrame(atoms, (), ())
    finally:
        env.close()
    metadata = pd.DataFrame(
        {
            "material_id": [row.material_id for row in selected],
            "csd_refcode": [row.csd_refcode for row in selected],
            "z_value": [row.z_value for row in selected],
            "genarris_step": [row.genarris_step for row in selected],
            "xtal_id": [row.xtal_id for row in selected],
            "natoms": [len(frames[row.material_id].atoms) for row in selected],
            "input_role": "unrelaxed_x0_geometry_only",
        }
    )
    source_paths = {
        "src/next26_omc25.py": Path(__file__).resolve(),
        "src/next11_geometry_only_frames.py": Path(__file__).resolve().parent / "next11_geometry_only_frames.py",
    }
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "input_role": "omc25_trajectory_records",
        "output_role": "unrelaxed_x0_geometry_only",
        "raw_records_contain_dft_labels": True,
        "record_metadata_fields_parsed": True,
        "endpoint_numeric_fields_parsed": False,
        "label_values_exported": False,
        "labels_opened": False,
        "relaxed_structures_opened": False,
        "model_or_proxy_potential_used": False,
        "coordinates_or_cell_modified": False,
        "selection_uses_endpoint_values": False,
        "excluded_development_refcodes_sha256": __import__("hashlib").sha256(("\n".join(sorted(excluded))+"\n").encode()).hexdigest(),
        "counts": {
            "complete_trajectories": len(complete),
            "excluded_by_refcode": len(complete) - len(selected),
            "selected_rows": len(selected),
            "selected_atoms": int(metadata["natoms"].sum()),
        },
        "inputs_sha256": {
            role: {"path": str(path), "sha256": input_hashes[role]} for role, path in paths.items()
        },
        "executed_source_sha256": source_hashes,
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
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
        for role, path in paths.items():
            if _sha256(path) != input_hashes[role]:
                raise RuntimeError(f"input {role} changed before publication")
        for name, path in source_paths.items():
            if _sha256(path) != source_hashes[name]:
                raise RuntimeError(f"source {name} changed before publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def _record_from_transaction(transaction: lmdb.Transaction, key: int) -> Mapping[str, object]:
    compressed = transaction.get(str(key).encode())
    if compressed is None:
        raise ValueError(f"missing LMDB record key {key}")
    value = decode(zlib.decompress(compressed).decode())
    if not isinstance(value, Mapping):
        raise ValueError(f"LMDB key {key} is not an ASE record")
    return value


def _endpoint_row(row: TrajectoryRef, initial: Mapping[str, object], final: Mapping[str, object]) -> dict[str, object]:
    numbers = np.asarray(initial["numbers"], dtype=int)
    final_numbers = np.asarray(final["numbers"], dtype=int)
    if not np.array_equal(numbers, final_numbers) or len(numbers) < 1:
        raise ValueError(f"trajectory composition changed: {row.material_id}")
    n = len(numbers)
    p0, p1 = np.asarray(initial["positions"], float), np.asarray(final["positions"], float)
    c0, c1 = np.asarray(initial["cell"], float), np.asarray(final["cell"], float)
    f0, f1 = np.asarray(initial["forces"], float), np.asarray(final["forces"], float)
    stress0 = np.asarray(initial["stress"], float)
    if p0.shape != (n, 3) or p1.shape != (n, 3) or f0.shape != (n, 3) or f1.shape != (n, 3):
        raise ValueError(f"invalid endpoint arrays: {row.material_id}")
    frac0 = p0 @ np.linalg.inv(c0)
    frac1 = p1 @ np.linalg.inv(c1)
    difference = frac1 - frac0
    difference -= np.rint(difference)
    displacement = np.linalg.norm(difference @ ((c0 + c1) / 2.0), axis=1)
    singular = np.linalg.svd(c1 @ np.linalg.inv(c0), compute_uv=False)
    force0 = np.linalg.norm(f0, axis=1)
    force1 = np.linalg.norm(f1, axis=1)
    energy_drop_pa = (float(initial["energy"]) - float(final["energy"])) / n
    values = {
        "material_id": row.material_id,
        "csd_refcode": row.csd_refcode,
        "nframes": len(row.lmdb_keys),
        "frame_first": row.frame_indices[0],
        "frame_last": row.frame_indices[-1],
        "natoms": n,
        "energy_drop_pa": energy_drop_pa,
        "force0_max": float(force0.max()),
        "force0_rms": float(np.sqrt(np.mean(force0**2))),
        "force1_max": float(force1.max()),
        "stress0_norm": float(np.linalg.norm(stress0)),
        "disp_rms": float(np.sqrt(np.mean(displacement**2))),
        "disp_p90": float(np.quantile(displacement, 0.9)),
        "disp_max": float(displacement.max()),
        "cell_logstrain_max": float(np.max(np.abs(np.log(singular)))),
        "volume_logchange": float(abs(math.log(abs(np.linalg.det(c1)) / abs(np.linalg.det(c0))))),
    }
    numeric = [value for key, value in values.items() if key not in {"material_id", "csd_refcode"}]
    if not np.all(np.isfinite(np.asarray(numeric, dtype=float))):
        raise ValueError(f"nonfinite endpoint: {row.material_id}")
    return values


def build_endpoint_table(
    *, db_path: Path, catalogue_path: Path, exclude_refcodes: Iterable[str] = ()
) -> pd.DataFrame:
    """Decode DFT labels for complete trajectories; never call before freeze."""

    complete = scan_complete_trajectories(db_path=Path(db_path), catalogue_path=Path(catalogue_path))
    excluded = frozenset(str(value) for value in exclude_refcodes)
    selected = [row for row in complete if row.csd_refcode not in excluded]
    env = lmdb.open(str(Path(db_path)), subdir=False, readonly=True, lock=False, readahead=False)
    rows: list[dict[str, object]] = []
    try:
        with env.begin() as transaction:
            for row in selected:
                initial = _record_from_transaction(transaction, row.lmdb_keys[0])
                final = _record_from_transaction(transaction, row.lmdb_keys[-1])
                rows.append(_endpoint_row(row, initial, final))
    finally:
        env.close()
    return pd.DataFrame(rows).sort_values("material_id", kind="stable").reset_index(drop=True)


def severe_dft_response(frame: pd.DataFrame) -> pd.Series:
    required = {"force0_max", "force0_rms", "energy_drop_pa", "stress0_norm"}
    if not required.issubset(frame.columns):
        raise ValueError(f"endpoint table lacks columns: {sorted(required-set(frame.columns))}")
    numeric = frame.loc[:, sorted(required)].apply(pd.to_numeric, errors="raise")
    if not np.all(np.isfinite(numeric.to_numpy(float))):
        raise ValueError("endpoint table contains nonfinite primary values")
    return (
        (numeric["force0_max"] >= FORCE_MAX_THRESHOLD)
        | (numeric["force0_rms"] >= FORCE_RMS_THRESHOLD)
        | (numeric["energy_drop_pa"] >= ENERGY_DROP_THRESHOLD)
        | (numeric["stress0_norm"] >= STRESS_NORM_THRESHOLD)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sanitize = sub.add_parser("sanitize-x0")
    sanitize.add_argument("--db", required=True, type=Path)
    sanitize.add_argument("--catalogue", required=True, type=Path)
    sanitize.add_argument("--output-dir", required=True, type=Path)
    sanitize.add_argument("--exclude-refcodes", type=Path)
    endpoint = sub.add_parser("endpoints")
    endpoint.add_argument("--db", required=True, type=Path)
    endpoint.add_argument("--catalogue", required=True, type=Path)
    endpoint.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "sanitize-x0":
        excluded: set[str] = set()
        if args.exclude_refcodes:
            excluded = {line.strip() for line in args.exclude_refcodes.read_text().splitlines() if line.strip()}
        result = sanitize_x0(
            db_path=args.db,
            catalogue_path=args.catalogue,
            output_dir=args.output_dir,
            exclude_refcodes=excluded,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        table = build_endpoint_table(db_path=args.db, catalogue_path=args.catalogue)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(args.output, index=False)
        print(json.dumps({"rows": len(table), "sha256": _sha256(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ENDPOINT_NAME",
    "GEOMETRY_NAME",
    "MANIFEST_NAME",
    "METADATA_NAME",
    "TrajectoryRef",
    "build_endpoint_table",
    "project_x0_payload",
    "sanitize_x0",
    "scan_complete_trajectories",
    "severe_dft_response",
]
