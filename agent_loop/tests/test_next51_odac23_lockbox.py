from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.next51_odac23_lockbox import RECEIPT_NAME, freeze_odac23_source_receipt


def test_source_receipt_hashes_opaque_bytes_without_opening_payload(tmp_path: Path) -> None:
    archive = tmp_path / "opaque.tar.gz"
    archive.write_bytes(b"opaque-not-a-real-tar")
    payload = archive.read_bytes()
    output = tmp_path / "receipt"

    manifest = freeze_odac23_source_receipt(
        source_archive_path=archive,
        output_dir=output,
        expected_size=len(payload),
        expected_md5=hashlib.md5(payload).hexdigest(),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )
    receipt = json.loads((output / RECEIPT_NAME).read_text())

    assert receipt["archive_members_opened"] is False
    assert receipt["payload_deserialized"] is False
    assert receipt["labels_opened"] is False
    assert manifest["outputs_sha256"][RECEIPT_NAME]

    with pytest.raises(FileExistsError):
        freeze_odac23_source_receipt(
            source_archive_path=archive,
            output_dir=output,
            expected_size=len(payload),
            expected_md5=hashlib.md5(payload).hexdigest(),
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )
