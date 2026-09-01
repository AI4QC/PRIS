#!/usr/bin/env python3
"""Opaque acquisition receipt for the future ODAC23 framework lockbox."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from src.next19_feature_build import _publish_directory_no_replace, _sha256


PROTOCOL = "2026-08-03-next51-odac23-opaque-source-receipt-v1"
OFFICIAL_URL = "https://dl.fbaipublicfiles.com/dac/datasets/odac23_is2r.tar.gz"
OFFICIAL_DOCUMENTATION = "https://fair-chem.github.io/dac/datasets/odac23.html"
PUBLICATION_DOI = "10.1021/acscentsci.3c01629"
EXPECTED_SIZE = 848_157_819
EXPECTED_MD5 = "f7f2f58669a30abae8cb9ba1b7f2bcd2"
EXPECTED_SHA256 = "13a26f00a6a26a95ab0706bf77b5dc1598cc689bda8846913bec0cc643152496"
RECEIPT_NAME = "ODAC23_SOURCE_RECEIPT.json"
MANIFEST_NAME = "MANIFEST.json"


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def freeze_odac23_source_receipt(
    *,
    source_archive_path: Path,
    output_dir: Path,
    expected_size: int = EXPECTED_SIZE,
    expected_md5: str = EXPECTED_MD5,
    expected_sha256: str = EXPECTED_SHA256,
) -> dict[str, object]:
    """Hash the archive as opaque bytes and publish a no-payload-access receipt."""

    source = Path(source_archive_path).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if not source.is_file():
        raise FileNotFoundError(str(source))
    if type(expected_size) is not int or expected_size <= 0:
        raise ValueError("ODAC23 expected byte count is invalid")
    actual_size = source.stat().st_size
    actual_md5 = _md5(source)
    actual_sha256 = _sha256(source)
    if (
        actual_size != expected_size
        or actual_md5 != expected_md5
        or actual_sha256 != expected_sha256
    ):
        raise ValueError("ODAC23 opaque source identity differs")
    receipt = {
        "protocol": PROTOCOL,
        "source": {
            "path": str(source),
            "official_url": OFFICIAL_URL,
            "official_documentation": OFFICIAL_DOCUMENTATION,
            "publication_doi": PUBLICATION_DOI,
            "bytes": actual_size,
            "md5": actual_md5,
            "sha256": actual_sha256,
            "level_of_theory": "PBE+D3 in VASP, per official documentation",
            "license": "CC BY 4.0",
        },
        "archive_members_opened": False,
        "archive_member_names_listed": False,
        "payload_deserialized": False,
        "structures_opened": False,
        "energies_opened": False,
        "forces_opened": False,
        "labels_opened": False,
        "future_role": "official-split framework development and independent OOD confirmation",
        "scientific_confirmation": False,
    }
    source_code = Path(__file__).resolve()
    source_hash = _sha256(source_code)
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "opaque_source_receipt_only",
        "archive_members_opened": False,
        "payload_deserialized": False,
        "labels_opened": False,
        "inputs_sha256": {
            "source_archive": {"path": str(source), "sha256": actual_sha256}
        },
        "executed_source_sha256": {
            "src/next51_odac23_lockbox.py": source_hash
        },
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        receipt_path = staging / RECEIPT_NAME
        receipt_path.write_bytes(_json_bytes(receipt))
        manifest["outputs_sha256"] = {RECEIPT_NAME: _sha256(receipt_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if (
            source.stat().st_size != actual_size
            or _md5(source) != actual_md5
            or _sha256(source) != actual_sha256
        ):
            raise RuntimeError("ODAC23 source changed before receipt publication")
        if _sha256(source_code) != source_hash:
            raise RuntimeError("NEXT51 source changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


__all__ = [
    "EXPECTED_MD5",
    "EXPECTED_SHA256",
    "EXPECTED_SIZE",
    "MANIFEST_NAME",
    "OFFICIAL_URL",
    "PROTOCOL",
    "RECEIPT_NAME",
    "freeze_odac23_source_receipt",
]
