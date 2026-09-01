#!/usr/bin/env python3
"""Seal the existing prospective SSAGEN x0 cohort as canonical geometry only."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Mapping, Sequence
import zipfile

import pandas as pd
from pymatgen.core import Composition

from src.next11_geometry_only_frames import (
    _parse_frame,
    _write_deterministic_archive,
)
from src.next19_feature_build import _publish_directory_no_replace
from src.next12_prospective_cohort import PROTOCOL as SOURCE_PROTOCOL


PROTOCOL = "2026-08-03-next24-ssagen-x0-sanitize-v1"
METADATA_NAME = "holdout_metadata.parquet"
GEOMETRY_NAME = "geometry_only_frames.zip"
MANIFEST_NAME = "MANIFEST.json"
SOURCE_COHORT_NAME = "cohort.parquet"
SOURCE_GEOMETRY_NAME = "geometry_only_frames.zip"
SOURCE_COLUMNS = (
    "sid",
    "generator",
    "generation_status",
    "natoms",
    "formula",
    "geometry_sha256",
    "archive_member",
)
FORMAL_INPUT_SHA256: Mapping[str, str] = {
    "cohort": "fc08be4f1b28dc82f4a26aeb49819b914ad8df7c7c7ee3887dea7a6c61095215",
    "geometry_only_frames": "3b392bdd38120dae579dc22b1b51e7c30bcbbed0e72c9b462c1bce16eda96959",
    "source_manifest": "8649853dcbb40a081183b671101fdf2933f30358ad1e5cb5b8694e8e451a846a",
}
_SID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path, *, role: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _hash_record(path: Path, digest: str) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": digest}


def _validate_source_manifest(
    manifest: Mapping[str, object], *, input_hashes: Mapping[str, str]
) -> None:
    if (
        manifest.get("protocol") != SOURCE_PROTOCOL
        or manifest.get("mode") != "prospective_x0_geometry_freeze"
        or manifest.get("all_attempts_retained") is not True
        or manifest.get("labels_opened") is not False
        or manifest.get("energy_or_force_models_called") is not False
        or manifest.get("scientific_improvement_claim") is not False
    ):
        raise ValueError("source manifest crossed the label-free boundary")
    outputs = manifest.get("outputs_sha256")
    expected = {
        SOURCE_COHORT_NAME: input_hashes["cohort"],
        SOURCE_GEOMETRY_NAME: input_hashes["geometry_only_frames"],
    }
    if not isinstance(outputs, Mapping) or any(
        outputs.get(name) != digest for name, digest in expected.items()
    ):
        raise ValueError("source manifest output hash differs")


def _generator_rk(values: pd.Series) -> str:
    if values.isna().any():
        raise ValueError("generator identity contains nulls")
    exact = values.astype(str)
    if exact.nunique(dropna=False) != 1:
        raise ValueError("generator identity must be unique")
    normalized = re.sub(r"[^a-z0-9]+", "_", exact.iloc[0].lower()).strip("_")
    if not normalized:
        raise ValueError("generator identity is empty")
    return normalized


def _formula_counts(formula: str) -> Counter[str]:
    try:
        amounts = Composition(str(formula)).get_el_amt_dict()
    except Exception as exc:
        raise ValueError(f"invalid source formula: {formula!r}") from exc
    counts: Counter[str] = Counter()
    for symbol, amount in amounts.items():
        numeric = float(amount)
        integer = int(round(numeric))
        if integer <= 0 or abs(numeric - integer) > 1e-10:
            raise ValueError(f"source formula is not an exact integer composition: {formula!r}")
        counts[str(symbol)] = integer
    return counts


def _load_source(
    *, cohort_path: Path, frames_zip_path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    input_hashes = {
        "cohort": _sha256(cohort_path),
        "geometry_only_frames": _sha256(frames_zip_path),
        "source_manifest": _sha256(manifest_path),
    }
    manifest = _strict_json(manifest_path, role="SSAGEN source manifest")
    _validate_source_manifest(manifest, input_hashes=input_hashes)
    try:
        table = pd.read_parquet(cohort_path, columns=list(SOURCE_COLUMNS))
    except Exception as exc:
        raise ValueError("could not read required SSAGEN cohort columns") from exc
    if table.empty:
        raise ValueError("SSAGEN cohort is empty")
    for column in ("sid", "generator", "generation_status", "formula", "geometry_sha256", "archive_member"):
        if table[column].isna().any():
            raise ValueError(f"SSAGEN cohort {column} contains nulls")
        table[column] = table[column].astype(str)
    if table["sid"].duplicated().any() or not table["sid"].map(
        lambda value: _SID.fullmatch(value) is not None
    ).all():
        raise ValueError("SSAGEN cohort IDs must be safe and unique")
    if not table["generation_status"].eq("generated").all():
        raise ValueError("every SSAGEN attempt must be generated before full-cohort screening")
    numeric_natoms = pd.to_numeric(table["natoms"], errors="raise")
    if not (numeric_natoms > 0).all() or not numeric_natoms.map(
        lambda value: float(value).is_integer()
    ).all():
        raise ValueError("SSAGEN atom counts must be positive integers")
    table["natoms"] = numeric_natoms.astype(int)
    expected_members = [f"frames/{sid}.extxyz" for sid in table["sid"]]
    if table["archive_member"].tolist() != expected_members:
        raise ValueError("SSAGEN archive member identities differ")
    counts = manifest.get("counts")
    expected_counts = {
        "attempts": len(table),
        "generated": len(table),
        "failed": 0,
        "archive_frames": len(table),
        "total_atoms": int(table["natoms"].sum()),
    }
    if not isinstance(counts, Mapping) or any(
        counts.get(key) != value for key, value in expected_counts.items()
    ):
        raise ValueError("SSAGEN source manifest counts differ")
    return table, {"manifest": manifest, "input_hashes": input_hashes}


def freeze_ssagen_x0(
    *,
    cohort_path: Path,
    frames_zip_path: Path,
    source_manifest_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Publish a canonical, label-free copy of every generated SSAGEN x0."""

    paths = {
        "cohort": Path(cohort_path).resolve(),
        "geometry_only_frames": Path(frames_zip_path).resolve(),
        "source_manifest": Path(source_manifest_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    table, source = _load_source(
        cohort_path=paths["cohort"],
        frames_zip_path=paths["geometry_only_frames"],
        manifest_path=paths["source_manifest"],
    )
    input_hashes = source["input_hashes"]
    assert isinstance(input_hashes, Mapping)
    formal_identity = dict(input_hashes) == dict(FORMAL_INPUT_SHA256)
    if require_formal_inputs and not formal_identity:
        raise ValueError("formal SSAGEN input identities differ")
    generator_rk = _generator_rk(table["generator"])

    frames: dict[str, object] = {}
    try:
        archive_context = zipfile.ZipFile(paths["geometry_only_frames"])
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("invalid SSAGEN geometry archive") from exc
    with archive_context as archive:
        if archive.comment:
            raise ValueError("SSAGEN source archive comment must be empty")
        infos = archive.infolist()
        if any(info.is_dir() or info.flag_bits & 0x1 for info in infos):
            raise ValueError("SSAGEN source archive contains an invalid member")
        names = [info.filename for info in infos]
        expected_names = table["archive_member"].tolist()
        if names != expected_names or any(
            PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
            for name in names
        ):
            raise ValueError("SSAGEN source archive exact frame coverage differs")
        for row, info in zip(table.to_dict("records"), infos, strict=True):
            payload = archive.read(info)
            if hashlib.sha256(payload).hexdigest() != row["geometry_sha256"]:
                raise ValueError(f"SSAGEN frame hash differs for {row['sid']}")
            parsed = _parse_frame(payload, strict_output=True)
            atoms = parsed.atoms
            if len(atoms) != int(row["natoms"]):
                raise ValueError(f"SSAGEN atom count differs for {row['sid']}")
            if Counter(atoms.get_chemical_symbols()) != _formula_counts(row["formula"]):
                raise ValueError(f"SSAGEN formula differs for {row['sid']}")
            frames[str(row["sid"])] = parsed

    metadata = pd.DataFrame(
        {
            "material_id": table["sid"].astype(str),
            "rk": generator_rk,
            "formula": table["formula"].astype(str),
            "natoms": table["natoms"].astype(int),
            "input_role": "unrelaxed_x0_geometry_only",
        }
    ).sort_values("material_id", kind="stable", ignore_index=True)

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next24_ssagen_holdout.py": Path(__file__).resolve(),
        "src/next11_geometry_only_frames.py": repository_root
        / "src/next11_geometry_only_frames.py",
    }
    source_hashes = {relative: _sha256(path) for relative, path in source_paths.items()}
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "generated_source_label_free_transport_cohort",
        "source_protocol": SOURCE_PROTOCOL,
        "source_generator": generator_rk,
        "input_role": "unrelaxed_x0_geometry_only",
        "all_generated_attempts_retained": True,
        "labels_opened": False,
        "endpoint_artifacts_opened": False,
        "relaxed_structures_opened": False,
        "endpoint_fields_accessed_by_sanitizer": False,
        "model_or_proxy_potential_used": False,
        "coordinates_or_cell_modified": False,
        "same_composition_candidates_used": False,
        "counts": {
            "rows": len(metadata),
            "frames": len(frames),
            "atoms": int(metadata["natoms"].sum()),
        },
        "inputs_sha256": {
            role: _hash_record(paths[role], str(input_hashes[role]))
            for role in paths
        },
        "executed_source_sha256": source_hashes,
        "production_protocol_eligible": bool(formal_identity),
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
        for relative, path in source_paths.items():
            if _sha256(path) != source_hashes[relative]:
                raise RuntimeError(f"source {relative} changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--frames-zip", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    freeze_ssagen_x0(
        cohort_path=arguments.cohort,
        frames_zip_path=arguments.frames_zip,
        source_manifest_path=arguments.source_manifest,
        output_dir=arguments.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GEOMETRY_NAME",
    "MANIFEST_NAME",
    "METADATA_NAME",
    "PROTOCOL",
    "freeze_ssagen_x0",
    "main",
]
