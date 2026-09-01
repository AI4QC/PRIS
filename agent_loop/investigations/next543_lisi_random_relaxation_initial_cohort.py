#!/usr/bin/env python3
"""Select Li--Si random relaxations and extract only their first x0 structure."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile

import numpy as np
from pymatgen.core import Structure

from src.next19_feature_build import _publish_directory_no_replace, _sha256


PROTOCOL = "2026-08-13-next543-lisi-random-relaxation-initial-cohort-v1"
DESIGN_SHA256 = "ebbca0e3badc20892f41f6436c8a783a67927d5b21390fb7072c773065af1b51"
SELECTION_SALT = "NEXT543-v1|"
PREFIXES = ("Li1Si1_02", "Li2Si1_02", "Li7Si2_03", "Li15Si4_02")
EXCLUDED_PREFIX = "Li13Si4_02"
PER_PREFIX = 50
EXPECTED_ROWS = len(PREFIXES) * PER_PREFIX
BUCKET = "gresearch"
ROOT_PREFIX = "crystal-relaxations"
INVENTORY_NAME = "NEXT543_LISI_RR_REMOTE_INVENTORY.json"
COHORT_NAME = "NEXT543_LISI_RR_INITIAL_COHORT.json"
GEOMETRY_NAME = "next543_lisi_rr_x0_geometry_only.zip"
MANIFEST_NAME = "MANIFEST.json"
EXPECTED_AUDIT_SHA256 = {
    "02_10000.json": "9fb5239fa41870d3d2ee2f865b0257fa2ca31cd981eb823a4d2143546b7a1df6",
    "Li13Si4_02_summary.txt": "4e0905d0fe7748e364271a6e1a718c71f066d44a1bc67ab14c91c9b04b4b2ded",
}
_STRUCTURE_ARRAY = re.compile(rb'"structure"\s*:\s*\[')


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _selection_hash(name: str) -> str:
    return hashlib.sha256((SELECTION_SALT + name).encode()).hexdigest()


def select_remote_objects(
    items: list[dict[str, object]], *, prefixes: tuple[str, ...] = PREFIXES, per_prefix: int = PER_PREFIX
) -> list[dict[str, object]]:
    if type(per_prefix) is not int or per_prefix < 1:
        raise ValueError("NEXT543 per-prefix selection count differs")
    selected: list[dict[str, object]] = []
    for prefix in prefixes:
        expected_start = f"{ROOT_PREFIX}/{prefix}/data/"
        eligible = [
            item
            for item in items
            if str(item.get("name", "")).startswith(expected_start)
            and str(item.get("name", "")).endswith(".json")
            and int(item.get("size", 0)) > 0
        ]
        eligible.sort(key=lambda item: (_selection_hash(str(item["name"])), str(item["name"])))
        if len(eligible) < per_prefix:
            raise ValueError(f"NEXT543 insufficient nonempty objects for {prefix}")
        for item in eligible[:per_prefix]:
            selected.append(
                {
                    "prefix": prefix,
                    "object_name": str(item["name"]),
                    "size": int(item["size"]),
                    "md5_base64": str(item["md5Hash"]),
                    "generation": str(item["generation"]),
                    "selection_hash": _selection_hash(str(item["name"])),
                }
            )
    selected.sort(key=lambda row: (str(row["prefix"]), str(row["selection_hash"])))
    return selected


def _list_prefix(prefix: str) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    token: str | None = None
    while True:
        query = {
            "prefix": f"{ROOT_PREFIX}/{prefix}/data/",
            "maxResults": "1000",
        }
        if token:
            query["pageToken"] = token
        url = (
            f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?"
            + urllib.parse.urlencode(query)
        )
        with urllib.request.urlopen(url, timeout=120) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise ValueError("NEXT543 GCS listing schema differs")
        rows = value.get("items", [])
        if not isinstance(rows, list):
            raise ValueError("NEXT543 GCS listing items differ")
        items.extend(row for row in rows if isinstance(row, dict))
        token_value = value.get("nextPageToken")
        token = str(token_value) if token_value else None
        if token is None:
            break
    return items


def _md5_base64(path: Path) -> str:
    digest = hashlib.md5(path.read_bytes(), usedforsecurity=False).digest()
    return base64.b64encode(digest).decode("ascii")


def _download_one(item: dict[str, object], source_dir: Path) -> dict[str, object]:
    object_name = str(item["object_name"])
    prefix = str(item["prefix"])
    target = source_dir / prefix / "data" / Path(object_name).name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == int(item["size"]):
        if _md5_base64(target) == str(item["md5_base64"]):
            return {**item, "local_path": str(target), "local_sha256": _sha256(target)}
    temporary = target.with_name(target.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    url = "https://storage.googleapis.com/" + BUCKET + "/" + urllib.parse.quote(
        object_name, safe="/"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "newpauling-NEXT543/1"})
    with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    if temporary.stat().st_size != int(item["size"]) or _md5_base64(temporary) != str(
        item["md5_base64"]
    ):
        raise ValueError(f"NEXT543 downloaded object identity differs: {object_name}")
    os.replace(temporary, target)
    return {**item, "local_path": str(target), "local_sha256": _sha256(target)}


def _balanced_object_end(data: mmap.mmap, start: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(data)):
        byte = data[index]
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte == 0x7B:
            depth += 1
        elif byte == 0x7D:
            depth -= 1
            if depth == 0:
                return index + 1
            if depth < 0:
                break
    raise ValueError("NEXT543 first structure object is not balanced")


def _extract_first_structure_dict(path: Path) -> tuple[dict[str, object], dict[str, int]]:
    """Decode exactly the first structure object, not DFT arrays or later frames."""
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        match = _STRUCTURE_ARRAY.search(data)
        if match is None:
            raise ValueError("NEXT543 structure array marker is missing")
        start = data.find(b"{", match.end())
        if start < 0:
            raise ValueError("NEXT543 first structure object is missing")
        end = _balanced_object_end(data, start)
        structure_bytes = bytes(data[start:end])
        file_size = len(data)
    value = json.loads(structure_bytes)
    if not isinstance(value, dict):
        raise ValueError("NEXT543 first structure schema differs")
    return value, {
        "file_size": file_size,
        "first_structure_start": start,
        "first_structure_end": end,
        "structure_objects_decoded": 1,
    }


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    return info


def build_initial_cohort(
    *,
    source_dir: Path,
    audit_dir: Path,
    design_path: Path,
    output_dir: Path,
    download_workers: int = 8,
) -> dict[str, object]:
    source = Path(source_dir).resolve()
    audit = Path(audit_dir).resolve()
    design = Path(design_path).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(download_workers) is not int or not 1 <= download_workers <= 16:
        raise ValueError("NEXT543 download workers differ")
    if not design.is_file() or _sha256(design) != DESIGN_SHA256:
        raise ValueError("NEXT543 design identity differs")
    audit_hashes = {
        name: _sha256(audit / name) for name in EXPECTED_AUDIT_SHA256
    }
    if audit_hashes != EXPECTED_AUDIT_SHA256:
        raise ValueError("NEXT543 excluded audit identity differs")
    if any(source.rglob("summary.txt")):
        raise ValueError("NEXT543 formal source directory contains an endpoint summary")

    remote_items: list[dict[str, object]] = []
    for prefix in PREFIXES:
        rows = _list_prefix(prefix)
        print(f"NEXT543 remote inventory {prefix}: {len(rows)} objects", flush=True)
        remote_items.extend(rows)
    selected = select_remote_objects(remote_items)
    if (
        len(selected) != EXPECTED_ROWS
        or Counter(str(row["prefix"]) for row in selected)
        != Counter({prefix: PER_PREFIX for prefix in PREFIXES})
        or any(EXCLUDED_PREFIX in str(row["object_name"]) for row in selected)
    ):
        raise RuntimeError("NEXT543 frozen remote selection differs")

    downloaded: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=download_workers) as executor:
        for offset, row in enumerate(
            executor.map(lambda item: _download_one(item, source), selected), start=1
        ):
            downloaded.append(row)
            if offset % 20 == 0 or offset == len(selected):
                print(f"NEXT543 downloads verified: {offset}/{len(selected)}", flush=True)
    downloaded.sort(key=lambda row: (str(row["prefix"]), str(row["selection_hash"])))

    cohort: list[dict[str, object]] = []
    structures: dict[str, bytes] = {}
    scan_audit: dict[str, dict[str, int]] = {}
    for offset, row in enumerate(downloaded, start=1):
        path = Path(str(row["local_path"]))
        structure_dict, scan = _extract_first_structure_dict(path)
        structure = Structure.from_dict(structure_dict)
        if (
            len(structure) < 1
            or not np.isfinite(structure.lattice.matrix).all()
            or structure.volume <= 1.0e-10
        ):
            raise ValueError(f"NEXT543 x0 geometry differs: {path.name}")
        trajectory_id = f"{row['prefix']}__{path.stem}"
        payload = _json_bytes(structure.as_dict())
        structures[f"{trajectory_id}.json"] = payload
        scan_audit[trajectory_id] = scan
        cohort.append(
            {
                "trajectory_id": trajectory_id,
                "prefix": row["prefix"],
                "object_name": row["object_name"],
                "local_source_sha256": row["local_sha256"],
                "x0_structure_sha256": hashlib.sha256(payload).hexdigest(),
                "formula": structure.composition.reduced_formula,
                "n_sites": len(structure),
                "volume": float(structure.volume),
            }
        )
        if offset % 25 == 0 or offset == len(downloaded):
            print(f"NEXT543 first structures extracted: {offset}/{len(downloaded)}", flush=True)
    if len({row["trajectory_id"] for row in cohort}) != EXPECTED_ROWS:
        raise RuntimeError("NEXT543 trajectory identities differ")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    try:
        inventory_path = staging / INVENTORY_NAME
        cohort_path = staging / COHORT_NAME
        geometry_path = staging / GEOMETRY_NAME
        inventory_path.write_bytes(_json_bytes(downloaded))
        cohort_path.write_bytes(_json_bytes(cohort))
        with zipfile.ZipFile(geometry_path, "w") as archive:
            for name in sorted(structures):
                archive.writestr(_zip_info(name), structures[name])
        outputs = {
            inventory_path.name: _sha256(inventory_path),
            cohort_path.name: _sha256(cohort_path),
            geometry_path.name: _sha256(geometry_path),
        }
        scan_starts = [value["first_structure_start"] for value in scan_audit.values()]
        scan_ends = [value["first_structure_end"] for value in scan_audit.values()]
        manifest = {
            "protocol": PROTOCOL,
            "selection": {
                "salt": SELECTION_SALT,
                "prefixes": list(PREFIXES),
                "excluded_prefix": EXCLUDED_PREFIX,
                "per_prefix": PER_PREFIX,
                "rows": EXPECTED_ROWS,
                "nonempty_json_required": True,
                "object_size_used_for_ordering": False,
                "endpoint_or_summary_value_used_for_selection": False,
            },
            "inputs_sha256": {
                "design": {"path": str(design), "sha256": DESIGN_SHA256},
                "excluded_schema_audit": {
                    name: {"path": str(audit / name), "sha256": value}
                    for name, value in audit_hashes.items()
                },
            },
            "outputs_sha256": outputs,
            "executed_source_sha256": {
                "src/next543_lisi_random_relaxation_initial_cohort.py": source_hash
            },
            "counts": {
                "rows": len(cohort),
                "prefixes": dict(sorted(Counter(row["prefix"] for row in cohort).items())),
            },
            "initial_extractor": {
                "method": "mmap literal structure array plus first balanced object only",
                "structure_objects_decoded_per_file": 1,
                "minimum_first_structure_start": min(scan_starts),
                "maximum_first_structure_end": max(scan_ends),
            },
            "remote_files_containing_endpoint_bytes_downloaded": True,
            "dft_force_stress_energy_values_decoded_or_inspected": False,
            "later_structure_objects_decoded_or_inspected": False,
            "summary_files_downloaded_or_read": False,
            "endpoint_values_opened": False,
            "geometry_only_x0_archive_created": True,
            "next544_prediction_freeze_authorized": True,
            "next545_endpoint_access_authorized": False,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash or _sha256(design) != DESIGN_SHA256:
            raise RuntimeError("NEXT543 source or design changed before publication")
        if any(_sha256(Path(str(row["local_path"]))) != row["local_sha256"] for row in downloaded):
            raise RuntimeError("NEXT543 downloaded source changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_initial_cohort(
        source_dir=args.source_dir,
        audit_dir=args.audit_dir,
        design_path=args.design,
        output_dir=args.output_dir,
        download_workers=args.download_workers,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
