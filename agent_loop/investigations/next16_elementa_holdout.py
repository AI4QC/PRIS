"""Select complete ELEMENTA x0 composition groups without reading endpoint labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence
import zipfile

import pandas as pd

from src.next13d_acsc_dft_pairs import _json_bytes, _sha256_file, _strict_json
from src.next14_wbm_holdout import _publish_directory_no_replace
from src.next11_geometry_only_frames import _canonical_frame, _parse_frame


PROTOCOL = "2026-08-02-next16-elementa-group-holdout-v2"
UPSTREAM_PROTOCOL = "2026-08-01-dft-pre-screening-design-v1"
METADATA_NAME = "holdout_metadata.parquet"
GEOMETRY_NAME = "geometry_only_frames.zip"
MANIFEST_NAME = "MANIFEST.json"
SELECTION_SALT = "next16-elementa-complete-groups-v1"
FROZEN_GROUP_COUNT = 400
FROZEN_FORMAL_SHA256: Mapping[str, str] = {
    "features": "5917eb0cc05bc55effa9b7776694021ac45d2b9dd47cbfc64ad52573647b6bff",
    "geometry": "8c63e02932fcb0158c5d917702a4a863f01b0e0ec55fdddff426a24613e10457",
    "upstream_manifest": "ae3b192f3d28400fffd2eb818e574e60bb9400a4b993d196cc1ba2fcac0ebb99",
}


def _group_key(value: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}|{value}".encode("utf-8")).hexdigest()


def _read_atoms(payload: bytes):
    try:
        atoms = _parse_frame(payload, strict_output=False).atoms
    except Exception as exc:
        raise ValueError(f"invalid ELEMENTA x0 frame: {type(exc).__name__}: {exc}") from exc
    if len(atoms) <= 0 or not atoms.pbc.all():
        raise ValueError("ELEMENTA x0 frame must be nonempty and fully periodic")
    return atoms


def build_elementa_holdout(
    *,
    features_path: Path,
    frames_zip_path: Path,
    upstream_manifest_path: Path,
    output_dir: Path,
    group_count: int = FROZEN_GROUP_COUNT,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Hash-select full composition groups from the geometry-only upstream artifact."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing existing output: {target}")
    if type(group_count) is not int or group_count <= 0:
        raise ValueError("group_count must be a positive exact integer")
    paths = {
        "features": Path(features_path).resolve(),
        "geometry": Path(frames_zip_path).resolve(),
        "upstream_manifest": Path(upstream_manifest_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    input_hashes = {role: _sha256_file(path) for role, path in paths.items()}
    if require_formal_inputs:
        if input_hashes != dict(FROZEN_FORMAL_SHA256):
            raise ValueError("formal ELEMENTA x0 inputs differ")
        if group_count != FROZEN_GROUP_COUNT:
            raise ValueError("formal ELEMENTA group count differs")

    manifest = _strict_json(paths["upstream_manifest"].read_bytes(), role="ELEMENTA x0 manifest")
    if manifest.get("protocol") != UPSTREAM_PROTOCOL or manifest.get("input_role") != "unrelaxed_x0_only":
        raise ValueError("ELEMENTA x0 upstream contract differs")
    outputs = manifest.get("outputs_sha256")
    expected_outputs = {
        paths["features"].name: input_hashes["features"],
        paths["geometry"].name: input_hashes["geometry"],
    }
    if not isinstance(outputs, Mapping) or any(outputs.get(name) != digest for name, digest in expected_outputs.items()):
        raise ValueError("ELEMENTA x0 output hashes differ from manifest")

    source = pd.read_parquet(paths["features"])
    required = {"sid", "rk", "material", "input_role"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"ELEMENTA x0 features lack columns: {sorted(missing)}")
    source = source.loc[:, sorted(required)].copy()
    source["sid"] = source["sid"].astype(str)
    source["rk"] = source["rk"].astype(str)
    if source["sid"].duplicated().any() or source["sid"].isna().any():
        raise ValueError("ELEMENTA x0 SIDs must be unique")
    if not source["input_role"].eq("unrelaxed_x0_only").all():
        raise ValueError("ELEMENTA source includes a non-x0 input role")
    groups = sorted(source["rk"].unique(), key=lambda value: (_group_key(str(value)), str(value)))
    if group_count > len(groups):
        raise ValueError("group_count exceeds available complete groups")
    selected_groups = set(groups[:group_count])
    selected = source.loc[source["rk"].isin(selected_groups)].sort_values(
        "sid", kind="stable", ignore_index=True
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        with zipfile.ZipFile(paths["geometry"], "r") as upstream:
            members = upstream.namelist()
            expected_members = {f"{sid}.extxyz" for sid in source["sid"]}
            if len(members) != len(set(members)) or set(members) != expected_members:
                raise ValueError("ELEMENTA geometry archive member set differs")
            metadata_rows: list[dict[str, object]] = []
            geometry_path = staging / GEOMETRY_NAME
            with zipfile.ZipFile(
                geometry_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
            ) as output:
                output.comment = b""
                for row in selected.itertuples(index=False):
                    name = f"{row.sid}.extxyz"
                    payload = upstream.read(name)
                    atoms = _read_atoms(payload)
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.create_system = 3
                    info.external_attr = 0o100644 << 16
                    info.extra = b""
                    info.comment = b""
                    output.writestr(
                        info,
                        _canonical_frame(atoms),
                        compress_type=zipfile.ZIP_DEFLATED,
                        compresslevel=9,
                    )
                    metadata_rows.append(
                        {
                            "material_id": str(row.sid),
                            "rk": str(row.rk),
                            "formula": atoms.get_chemical_formula(mode="hill"),
                            "natoms": len(atoms),
                            "input_role": "unrelaxed_x0_geometry_only",
                        }
                    )
        metadata = pd.DataFrame(metadata_rows)
        if metadata["material_id"].duplicated().any() or metadata["rk"].nunique() != group_count:
            raise ValueError("ELEMENTA holdout lost rows or complete groups")
        metadata_path = staging / METADATA_NAME
        metadata.to_parquet(metadata_path, index=False)
        result: dict[str, object] = {
            "protocol": PROTOCOL,
            "evidence_role": "historical ELEMENTA external-source retrospective",
            "input_role": "unrelaxed_x0_geometry_only",
            "endpoint_bytes_read_by_execution": False,
            "labels_previously_opened_elsewhere": True,
            "fresh_lockbox": False,
            "selection": {
                "unit": "complete reduced-composition group",
                "salt": SELECTION_SALT,
                "algorithm": "ascending SHA256(salt|rk)",
                "selected_groups": group_count,
            },
            "counts": {
                "source_rows": len(source),
                "source_groups": int(source["rk"].nunique()),
                "selected_rows": len(metadata),
                "selected_groups": int(metadata["rk"].nunique()),
                "selected_atoms": int(metadata["natoms"].sum()),
            },
            "inputs_sha256": {
                role: {"path": str(path), "sha256": input_hashes[role]}
                for role, path in paths.items()
            },
            "outputs_sha256": {
                METADATA_NAME: _sha256_file(metadata_path),
                GEOMETRY_NAME: _sha256_file(geometry_path),
            },
            "executed_source_sha256": {
                "src/next16_elementa_holdout.py": _sha256_file(Path(__file__).resolve())
            },
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(result))
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--frames-zip", required=True, type=Path)
    parser.add_argument("--upstream-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    build_elementa_holdout(
        features_path=arguments.features,
        frames_zip_path=arguments.frames_zip,
        upstream_manifest_path=arguments.upstream_manifest,
        output_dir=arguments.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
