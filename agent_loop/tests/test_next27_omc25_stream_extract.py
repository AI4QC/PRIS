"""Opaque streaming extraction contracts for future OMC25 holdout shards."""

from __future__ import annotations

import io
from pathlib import Path
import tarfile

import pytest


def _archive() -> io.BytesIO:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for index in range(5):
            lock = b"lock"
            info = tarfile.TarInfo(f"val/data{index:04d}.aselmdb-lock")
            info.size = len(lock)
            archive.addfile(info, io.BytesIO(lock))
            payload = bytes([index]) * (100 + index)
            info = tarfile.TarInfo(f"val/data{index:04d}.aselmdb")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    stream.seek(0)
    return stream


def test_stream_extractor_skips_prior_main_members_and_publishes_opaque_bytes(
    tmp_path: Path,
) -> None:
    from src.next27_omc25_stream_extract import extract_members

    output = tmp_path / "members"
    manifest = extract_members(
        stream=_archive(), output_dir=output, skip_main=2, take_main=2
    )

    assert sorted(path.name for path in output.glob("*.aselmdb")) == [
        "data0002.aselmdb",
        "data0003.aselmdb",
    ]
    assert (output / "data0002.aselmdb").read_bytes() == bytes([2]) * 102
    assert (output / "data0003.aselmdb").read_bytes() == bytes([3]) * 103
    assert manifest["opaque_bytes_only"] is True
    assert manifest["scientific_fields_decoded"] is False
    with pytest.raises(FileExistsError):
        extract_members(stream=_archive(), output_dir=output, skip_main=0, take_main=1)


def test_stream_extractor_requires_positive_bounded_selection(tmp_path: Path) -> None:
    from src.next27_omc25_stream_extract import extract_members

    with pytest.raises(ValueError):
        extract_members(stream=_archive(), output_dir=tmp_path / "x", skip_main=-1, take_main=1)
    with pytest.raises(ValueError):
        extract_members(stream=_archive(), output_dir=tmp_path / "y", skip_main=0, take_main=0)

