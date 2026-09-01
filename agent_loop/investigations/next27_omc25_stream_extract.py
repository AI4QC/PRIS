#!/usr/bin/env python3
"""Extract selected OMC25 ASE-LMDB members as opaque bytes from stdin."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile
from typing import BinaryIO

from src.next19_feature_build import _publish_directory_no_replace, _sha256


PROTOCOL = "2026-08-03-next27-omc25-opaque-stream-extract-v1"
MANIFEST_NAME = "MANIFEST.json"
MEMBER = re.compile(r"val/(data[0-9]{4}\.aselmdb)")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def extract_members(
    *,
    stream: BinaryIO,
    output_dir: Path,
    skip_main: int,
    take_main: int,
    source_url: str | None = None,
) -> dict[str, object]:
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if skip_main < 0 or take_main <= 0 or take_main > 16:
        raise ValueError("invalid bounded main-member selection")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    selected: list[dict[str, object]] = []
    main_seen = 0
    try:
        try:
            archive_context = tarfile.open(fileobj=stream, mode="r|gz")
        except tarfile.TarError as exc:
            raise ValueError("invalid gzip tar stream") from exc
        with archive_context as archive:
            for member in archive:
                if not member.isfile():
                    continue
                match = MEMBER.fullmatch(member.name)
                if match is None:
                    continue
                index = main_seen
                main_seen += 1
                if index < skip_main:
                    continue
                if len(selected) >= take_main:
                    break
                basename = match.group(1)
                if PurePosixPath(basename).name != basename:
                    raise ValueError("unsafe OMC25 member name")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"could not stream member {member.name}")
                output_path = staging / basename
                with output_path.open("xb") as destination:
                    shutil.copyfileobj(source, destination, length=1 << 20)
                if output_path.stat().st_size != member.size:
                    raise ValueError(f"streamed size differs for {member.name}")
                selected.append(
                    {
                        "archive_member": member.name,
                        "path": basename,
                        "size": member.size,
                        "sha256": _sha256(output_path),
                        "main_member_index": index,
                    }
                )
                if len(selected) == take_main:
                    break
        if len(selected) != take_main:
            raise ValueError(
                f"stream ended after selecting {len(selected)} of {take_main} members"
            )
        source_hashes = {
            "src/next27_omc25_stream_extract.py": _sha256(Path(__file__).resolve())
        }
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "source_url": source_url,
            "skip_main": skip_main,
            "take_main": take_main,
            "opaque_bytes_only": True,
            "scientific_fields_decoded": False,
            "dft_values_opened": False,
            "members": selected,
            "executed_source_sha256": source_hashes,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(Path(__file__).resolve()) != source_hashes[
            "src/next27_omc25_stream_extract.py"
        ]:
            raise RuntimeError("stream extractor source changed before publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--skip-main", required=True, type=int)
    parser.add_argument("--take-main", required=True, type=int)
    parser.add_argument("--source-url")
    args = parser.parse_args(argv)
    result = extract_members(
        stream=sys.stdin.buffer,
        output_dir=args.output_dir,
        skip_main=args.skip_main,
        take_main=args.take_main,
        source_url=args.source_url,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["extract_members"]
