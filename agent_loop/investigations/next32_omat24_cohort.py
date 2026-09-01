"""Geometry-only OMat24 cohort projection and separately decoded DFT endpoints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
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

from src.next26_omc25 import _top_level_fields, project_x0_payload
from src.next11_geometry_only_frames import _ParsedFrame, _write_deterministic_archive
from src.next19_feature_build import _publish_directory_no_replace, _sha256


PROTOCOL = "2026-08-03-next32-omat24-inorganic-response-cohort-v1"
ENDPOINT_PROTOCOL = "2026-08-03-next32-omat24-dft-response-endpoints-v1"
COHORT_NAME = "next32_cohort.parquet"
GEOMETRY_NAME = "geometry_only_frames.zip"
PARENT_NAME = "parent_ids.txt"
ENDPOINT_NAME = "next32_dft_endpoints.parquet"
MANIFEST_NAME = "MANIFEST.json"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read_manifest(path: Path, *, role: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role} manifest") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid {role} manifest")
    return value


def _json_scalar(raw: str, *, role: str) -> str:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {role}") from exc
    if type(value) is not str or not value:
        raise ValueError(f"invalid {role}")
    return value


def project_identity_geometry(payload: bytes) -> tuple[dict[str, str], Atoms]:
    """Project identity and geometry while skipping every DFT numeric field."""

    top = _top_level_fields(payload, {"data"})
    nested = _top_level_fields(
        top["data"].encode("utf-8"), {"sid", "parent_id", "task_type"}
    )
    metadata = {
        name: _json_scalar(nested[name], role=name)
        for name in ("sid", "parent_id", "task_type")
    }
    atoms = project_x0_payload(payload)
    if atoms.calc is not None or atoms.info:
        raise ValueError("geometry projection retained calculator information")
    return metadata, atoms


def _project_identity(payload: bytes) -> dict[str, str]:
    top = _top_level_fields(payload, {"data"})
    nested = _top_level_fields(
        top["data"].encode("utf-8"), {"sid", "parent_id", "task_type"}
    )
    return {
        name: _json_scalar(nested[name], role=name)
        for name in ("sid", "parent_id", "task_type")
    }


def select_parent_unique(
    rows: Sequence[Mapping[str, object]],
    *,
    salt: str,
    limit: int,
    exclude_parent_ids: Iterable[str] = (),
) -> list[dict[str, object]]:
    """Select a deterministic, parent-disjoint cohort without endpoint values."""

    if not salt or not isinstance(limit, int) or limit <= 0:
        raise ValueError("selection salt and limit are invalid")
    excluded = frozenset(str(value) for value in exclude_parent_ids)
    by_parent: dict[str, tuple[str, dict[str, object]]] = {}
    seen_sids: set[str] = set()
    seen_keys: set[int] = set()
    for raw in rows:
        try:
            sid = str(raw["sid"])
            parent = str(raw["parent_id"])
            record_key = int(raw["record_key"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid OMat24 metadata row") from exc
        if not sid or not parent or record_key <= 0:
            raise ValueError("invalid OMat24 metadata identity")
        if sid in seen_sids or record_key in seen_keys:
            raise ValueError("duplicate OMat24 sid or record key")
        seen_sids.add(sid)
        seen_keys.add(record_key)
        if parent in excluded:
            continue
        digest = hashlib.sha256(f"{salt}|{parent}|{sid}".encode("utf-8")).hexdigest()
        row = dict(raw)
        previous = by_parent.get(parent)
        if previous is None or digest < previous[0]:
            by_parent[parent] = (digest, row)
    ordered = sorted(by_parent.values(), key=lambda item: (item[0], str(item[1]["sid"])))
    if len(ordered) < limit:
        raise ValueError("not enough unique eligible OMat24 parents")
    return [row for _, row in ordered[:limit]]


def decode_dft_endpoint(payload: bytes) -> dict[str, float]:
    """Decode one selected OMat24 DFT record after predictions are frozen."""

    fields = _top_level_fields(payload, {"numbers", "energy", "forces", "stress"})
    try:
        numbers = np.asarray(decode(fields["numbers"]), dtype=int)
        energy = float(decode(fields["energy"]))
        forces = np.asarray(decode(fields["forces"]), dtype=float)
        stress = np.asarray(decode(fields["stress"]), dtype=float)
    except Exception as exc:
        raise ValueError("invalid OMat24 DFT endpoint") from exc
    if numbers.ndim != 1 or len(numbers) < 1:
        raise ValueError("endpoint numbers are invalid")
    if forces.shape != (len(numbers), 3) or not np.isfinite(forces).all():
        raise ValueError("endpoint forces have invalid shape or values")
    if stress.shape not in {(6,), (3, 3)} or not np.isfinite(stress).all():
        raise ValueError("endpoint stress has invalid shape or values")
    if not np.isfinite(energy):
        raise ValueError("endpoint energy is not finite")
    magnitudes = np.linalg.norm(forces, axis=1)
    return {
        "force_max": float(magnitudes.max()),
        "force_rms": float(np.sqrt(np.mean(magnitudes**2))),
        "stress_norm": float(np.linalg.norm(stress)),
        "energy_per_atom": float(energy / len(numbers)),
    }


def _record_payload(transaction: lmdb.Transaction, record_key: int) -> bytes:
    compressed = transaction.get(str(record_key).encode())
    if compressed is None:
        raise ValueError(f"missing OMat24 LMDB record {record_key}")
    try:
        return zlib.decompress(compressed)
    except zlib.error as exc:
        raise ValueError(f"invalid compressed OMat24 record {record_key}") from exc


def sanitize_omat24_cohort(
    *,
    db_path: Path,
    source_name: str,
    salt: str,
    limit: int,
    output_dir: Path,
    exclude_parent_ids: Iterable[str] = (),
) -> dict[str, object]:
    """Publish a parent-unique geometry-only cohort without endpoint conversion."""

    database = Path(db_path).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if not database.is_file():
        raise FileNotFoundError(str(database))
    if not source_name:
        raise ValueError("source name is empty")
    database_hash = _sha256(database)
    excluded = frozenset(str(value) for value in exclude_parent_ids)
    env = lmdb.open(
        str(database), subdir=False, readonly=True, lock=False, readahead=False
    )
    scanned: list[dict[str, object]] = []
    try:
        with env.begin() as transaction:
            for key, compressed in transaction.cursor():
                if key in {b"nextid", b"metadata", b"deleted_ids"}:
                    continue
                try:
                    record_key = int(key)
                    payload = zlib.decompress(compressed)
                except (ValueError, zlib.error) as exc:
                    raise ValueError(f"invalid OMat24 LMDB key {key!r}") from exc
                identity = _project_identity(payload)
                scanned.append({**identity, "record_key": record_key})
        selected = select_parent_unique(
            scanned,
            salt=salt,
            limit=limit,
            exclude_parent_ids=excluded,
        )
        frames: dict[str, _ParsedFrame] = {}
        rows: list[dict[str, object]] = []
        with env.begin() as transaction:
            for selected_row in selected:
                record_key = int(selected_row["record_key"])
                payload = _record_payload(transaction, record_key)
                identity, atoms = project_identity_geometry(payload)
                for name in ("sid", "parent_id", "task_type"):
                    if identity[name] != str(selected_row[name]):
                        raise ValueError("OMat24 identity changed between scan and projection")
                material_id = f"{source_name}::{identity['sid']}"
                frames[material_id] = _ParsedFrame(atoms, (), ())
                rows.append(
                    {
                        "material_id": material_id,
                        "source_name": source_name,
                        "sid": identity["sid"],
                        "parent_id": identity["parent_id"],
                        "task_type": identity["task_type"],
                        "record_key": record_key,
                        "natoms": len(atoms),
                        "input_role": "unrelaxed_x0_geometry_only",
                    }
                )
    finally:
        env.close()
    metadata = pd.DataFrame(rows).sort_values("material_id", kind="stable", ignore_index=True)
    if metadata.material_id.duplicated().any() or metadata.parent_id.duplicated().any():
        raise ValueError("sanitized OMat24 identities are duplicated")
    source_hash = _sha256(Path(__file__).resolve())
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "source_name": source_name,
        "input_role": "omat24_records_containing_geometry_and_dft_labels",
        "output_role": "unrelaxed_x0_geometry_only",
        "raw_records_contain_dft_labels": True,
        "record_identity_fields_parsed": True,
        "endpoint_numeric_fields_parsed": False,
        "label_values_exported": False,
        "labels_opened": False,
        "relaxed_structures_opened": False,
        "model_or_proxy_potential_used": False,
        "selection_uses_endpoint_values": False,
        "physical_never_read_lockbox": False,
        "salt": salt,
        "excluded_parent_ids_sha256": hashlib.sha256(
            ("\n".join(sorted(excluded)) + "\n").encode()
        ).hexdigest(),
        "counts": {
            "raw_records": len(scanned),
            "selected_rows": len(metadata),
            "selected_parents": int(metadata.parent_id.nunique()),
            "selected_atoms": int(metadata.natoms.sum()),
        },
        "inputs_sha256": {"aselmdb": {"path": str(database), "sha256": database_hash}},
        "executed_source_sha256": {"src/next32_omat24_cohort.py": source_hash},
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        metadata_path = staging / COHORT_NAME
        geometry_path = staging / GEOMETRY_NAME
        parent_path = staging / PARENT_NAME
        metadata.to_parquet(metadata_path, index=False)
        _write_deterministic_archive(geometry_path, frames)
        parent_path.write_text(
            "\n".join(sorted(metadata.parent_id.astype(str))) + "\n", encoding="utf-8"
        )
        manifest["outputs_sha256"] = {
            name: _sha256(staging / name)
            for name in (COHORT_NAME, GEOMETRY_NAME, PARENT_NAME)
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(database) != database_hash or _sha256(Path(__file__).resolve()) != source_hash:
            raise RuntimeError("OMat24 cohort input changed during publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def publish_dft_endpoints(
    *,
    db_path: Path,
    metadata_path: Path,
    cohort_manifest_path: Path,
    identity_lock_path: Path,
    identity_lock_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Decode selected DFT values only after validating an immutable identity lock."""

    paths = {
        "aselmdb": Path(db_path).resolve(),
        "metadata": Path(metadata_path).resolve(),
        "cohort_manifest": Path(cohort_manifest_path).resolve(),
        "identity_lock": Path(identity_lock_path).resolve(),
        "identity_lock_manifest": Path(identity_lock_manifest_path).resolve(),
    }
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("OMat24 endpoint input is missing")
    hashes = {role: _sha256(path) for role, path in paths.items()}
    cohort_manifest = _read_manifest(paths["cohort_manifest"], role="cohort")
    lock_manifest = _read_manifest(paths["identity_lock_manifest"], role="identity lock")
    cohort_outputs = cohort_manifest.get("outputs_sha256")
    lock_outputs = lock_manifest.get("outputs_sha256")
    if (
        cohort_manifest.get("protocol") != PROTOCOL
        or cohort_manifest.get("labels_opened") is not False
        or not isinstance(cohort_outputs, Mapping)
        or cohort_outputs.get(paths["metadata"].name) != hashes["metadata"]
        or lock_manifest.get("labels_opened") is not False
        or not isinstance(lock_outputs, Mapping)
        or lock_outputs.get(paths["identity_lock"].name) != hashes["identity_lock"]
    ):
        raise ValueError("OMat24 identity lock crossed the pre-label boundary")
    metadata = pd.read_parquet(paths["metadata"])
    lock = pd.read_parquet(paths["identity_lock"], columns=["material_id"])
    required = {"material_id", "source_name", "sid", "parent_id", "record_key"}
    if not required.issubset(metadata.columns):
        raise ValueError("OMat24 cohort metadata lacks endpoint identity fields")
    for frame, role in ((metadata, "metadata"), (lock, "identity lock")):
        if frame.material_id.isna().any() or frame.material_id.astype(str).duplicated().any():
            raise ValueError(f"OMat24 {role} identities are invalid")
    if set(metadata.material_id.astype(str)) != set(lock.material_id.astype(str)):
        raise ValueError("OMat24 identity lock differs from cohort metadata")
    env = lmdb.open(
        str(paths["aselmdb"]), subdir=False, readonly=True, lock=False, readahead=False
    )
    rows: list[dict[str, object]] = []
    try:
        with env.begin() as transaction:
            for row in metadata.to_dict("records"):
                payload = _record_payload(transaction, int(row["record_key"]))
                identity = _project_identity(payload)
                if identity["sid"] != str(row["sid"]) or identity["parent_id"] != str(
                    row["parent_id"]
                ):
                    raise ValueError("OMat24 endpoint identity differs from cohort")
                endpoint = decode_dft_endpoint(payload)
                rows.append(
                    {
                        "material_id": str(row["material_id"]),
                        "source_name": str(row["source_name"]),
                        "parent_id": str(row["parent_id"]),
                        **endpoint,
                    }
                )
    finally:
        env.close()
    endpoints = pd.DataFrame(rows).sort_values("material_id", kind="stable", ignore_index=True)
    source_hash = _sha256(Path(__file__).resolve())
    manifest: dict[str, object] = {
        "protocol": ENDPOINT_PROTOCOL,
        "labels_opened": True,
        "identity_lock": "exact frozen material_id set",
        "physical_never_read_lockbox": False,
        "counts": {"rows": len(endpoints), "sources": int(endpoints.source_name.nunique())},
        "inputs_sha256_before_opening": hashes,
        "executed_source_sha256": {"src/next32_omat24_cohort.py": source_hash},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        endpoint_path = staging / ENDPOINT_NAME
        endpoints.to_parquet(endpoint_path, index=False)
        manifest["outputs_sha256"] = {ENDPOINT_NAME: _sha256(endpoint_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if any(_sha256(path) != hashes[role] for role, path in paths.items()):
            raise RuntimeError("OMat24 endpoint input changed during publication")
        if _sha256(Path(__file__).resolve()) != source_hash:
            raise RuntimeError("OMat24 endpoint source changed during publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


__all__ = [
    "PROTOCOL",
    "decode_dft_endpoint",
    "project_identity_geometry",
    "publish_dft_endpoints",
    "sanitize_omat24_cohort",
    "select_parent_unique",
]
