"""Extract only SCIGEN pre-DFT POSCAR geometry into composition-isolated splits."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Mapping
import zipfile

from ase import Atoms
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Poscar

from src.next11_geometry_only_frames import _canonical_frame
from src.next12_dft_queue import _json_bytes
from src.next13d_acsc_dft_pairs import _sha256_file
from src.next14_wbm_holdout import _publish_directory_no_replace
from src.next83_scigen_source_audit import (
    AUDIT_NAME,
    EXPECTED_ARCHIVE_SHA256,
    PROTOCOL as SOURCE_AUDIT_PROTOCOL,
    ROOT,
)


PROTOCOL = "2026-08-03-next84-scigen-composition-isolated-x0-lockbox-v1"
METADATA_NAME = "scigen_x0_metadata.parquet"
IDENTITIES_NAME = "PARTITION_IDENTITIES.json"
MANIFEST_NAME = "MANIFEST.json"
PARTITIONS = ("discovery", "internal_validation", "internal_replication")
GEOMETRY_NAMES = {
    "discovery": "geometry_discovery.zip",
    "internal_validation": "geometry_internal_validation.zip",
    "internal_replication": "geometry_internal_replication.zip",
}
SPLIT_SALT = "NEXT84_SCIGEN_COMPOSITION_SPLIT_V1"
EXPECTED_SOURCE_AUDIT_SHA256 = (
    "de470fbae91f486d63aebdc4672d7e249ca18e6cd259cb835d6cc2729dee2132"
)
EXPECTED_SOURCE_AUDIT_MANIFEST_SHA256 = (
    "6fbdf2b62e07e590d726fe5bb5dc7bf9900a6f917ed1b4c35ffe27d2a6106b87"
)
EXPECTED_DESIGN_SHA256 = (
    "39127f4d2b5ddba176f7904ed498f98e0326fa902e1c3ede79fbbcf320c13ee9"
)
ALLOWED_LATTICE_CLASSES = frozenset(
    {"tri", "hon", "kag", "sqr", "elt", "sns", "tsq", "srt", "snh", "trh", "lieb"}
)
_ID_RE = re.compile(r"^(?P<lattice>[a-z]+)_(?P<run>[0-9]+)_(?P<index>[0-9]+)$")


def assign_partition(reduced_formula: str) -> str:
    if type(reduced_formula) is not str or not reduced_formula.strip():
        raise ValueError("reduced_formula must be a nonempty exact string")
    digest = hashlib.sha256(
        f"{SPLIT_SALT}|{reduced_formula}".encode("utf-8")
    ).digest()
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if unit < 0.55:
        return "discovery"
    if unit < 0.75:
        return "internal_validation"
    return "internal_replication"


def _read_json(path: Path, *, role: str) -> Mapping[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _validate_source_audit(
    *,
    source_archive_path: Path,
    source_audit_path: Path,
    source_audit_manifest_path: Path,
) -> str:
    audit = _read_json(source_audit_path, role="NEXT83 source audit")
    manifest = _read_json(source_audit_manifest_path, role="NEXT83 source audit manifest")
    outputs = manifest.get("outputs_sha256")
    if (
        audit.get("protocol") != SOURCE_AUDIT_PROTOCOL
        or manifest.get("protocol") != SOURCE_AUDIT_PROTOCOL
        or not isinstance(outputs, Mapping)
        or outputs.get(AUDIT_NAME) != _sha256_file(source_audit_path)
    ):
        raise ValueError("NEXT83 audit output hash or protocol differs")
    if (
        audit.get("labels_opened") is not False
        or audit.get("endpoint_payloads_opened") is not False
        or audit.get("relaxed_structures_opened") is not False
    ):
        raise ValueError("NEXT83 source audit is not label-sealed")
    source = audit.get("source")
    if not isinstance(source, Mapping) or source.get("sha256") != _sha256_file(source_archive_path):
        raise ValueError("SCIGEN source archive differs from NEXT83 audit")
    return str(source["sha256"])


def _poscar_inventory(archive: zipfile.ZipFile) -> list[tuple[str, zipfile.ZipInfo, str]]:
    rows: list[tuple[str, zipfile.ZipInfo, str]] = []
    seen: set[str] = set()
    for info in archive.infolist():
        parts = PurePosixPath(info.filename).parts
        if len(parts) != 3 or parts[0] != ROOT or parts[2] != "POSCAR":
            continue
        material_id = parts[1]
        match = _ID_RE.fullmatch(material_id)
        if match is None or match.group("lattice") not in ALLOWED_LATTICE_CLASSES:
            raise ValueError(f"invalid SCIGEN material ID: {material_id}")
        if material_id in seen:
            raise ValueError("duplicate SCIGEN POSCAR identity")
        seen.add(material_id)
        rows.append((material_id, info, match.group("lattice")))
    rows.sort(key=lambda value: value[0])
    if not rows:
        raise ValueError("SCIGEN archive has no POSCAR members")
    return rows


def _geometry_only_atoms(payload: bytes) -> tuple[Atoms, str, str]:
    try:
        structure = Poscar.from_str(payload.decode("utf-8")).structure
    except Exception as exc:
        raise ValueError(f"invalid SCIGEN POSCAR: {type(exc).__name__}: {exc}") from exc
    if len(structure) <= 0 or not all(structure.lattice.pbc):
        raise ValueError("SCIGEN POSCAR must be nonempty and fully periodic")
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = None
    atoms.info.clear()
    for name in list(atoms.arrays):
        if name not in {"numbers", "positions"}:
            del atoms.arrays[name]
    # Canonical serialization is also the strict finite/periodic geometry check.
    _canonical_frame(atoms)
    reduced_formula = structure.composition.reduced_formula
    chemical_system = "-".join(sorted(element.symbol for element in structure.composition.elements))
    if not reduced_formula or not chemical_system:
        raise ValueError("SCIGEN POSCAR composition is empty")
    return atoms, reduced_formula, chemical_system


def _write_frame(archive: zipfile.ZipFile, material_id: str, atoms: Atoms) -> None:
    info = zipfile.ZipInfo(f"{material_id}.extxyz", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.extra = b""
    info.comment = b""
    archive.writestr(
        info,
        _canonical_frame(atoms),
        compress_type=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    )


def _identity_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode("utf-8")).hexdigest()


def build_scigen_geometry_lockbox(
    *,
    source_archive_path: Path,
    source_audit_path: Path,
    source_audit_manifest_path: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Read only POSCAR payloads and publish physically separated x0 geometry."""

    source = Path(source_archive_path).resolve()
    audit_path = Path(source_audit_path).resolve()
    audit_manifest_path = Path(source_audit_manifest_path).resolve()
    design = Path(design_path).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in (source, audit_path, audit_manifest_path, design)):
        raise FileNotFoundError("NEXT84 input is missing")
    input_hashes = {
        "source_archive": _sha256_file(source),
        "source_audit": _sha256_file(audit_path),
        "source_audit_manifest": _sha256_file(audit_manifest_path),
        "design": _sha256_file(design),
    }
    if require_formal_inputs and input_hashes != {
        "source_archive": EXPECTED_ARCHIVE_SHA256,
        "source_audit": EXPECTED_SOURCE_AUDIT_SHA256,
        "source_audit_manifest": EXPECTED_SOURCE_AUDIT_MANIFEST_SHA256,
        "design": EXPECTED_DESIGN_SHA256,
    }:
        raise ValueError("NEXT84 formal input identity differs")
    audited_source_sha = _validate_source_audit(
        source_archive_path=source,
        source_audit_path=audit_path,
        source_audit_manifest_path=audit_manifest_path,
    )
    if audited_source_sha != input_hashes["source_archive"]:
        raise ValueError("NEXT84 source identity differs after audit validation")

    source_code = Path(__file__).resolve()
    source_code_hash = _sha256_file(source_code)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    metadata_rows: list[dict[str, object]] = []
    try:
        zip_outputs = {
            role: zipfile.ZipFile(
                staging / filename,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
                strict_timestamps=True,
            )
            for role, filename in GEOMETRY_NAMES.items()
        }
        try:
            for archive in zip_outputs.values():
                archive.comment = b""
            with zipfile.ZipFile(source) as source_zip:
                inventory = _poscar_inventory(source_zip)
                for material_id, info, lattice_class in inventory:
                    atoms, reduced_formula, chemical_system = _geometry_only_atoms(
                        source_zip.read(info)
                    )
                    partition = assign_partition(reduced_formula)
                    _write_frame(zip_outputs[partition], material_id, atoms)
                    metadata_rows.append(
                        {
                            "material_id": material_id,
                            "lattice_class": lattice_class,
                            "reduced_formula": reduced_formula,
                            "chemical_system": chemical_system,
                            "natoms": len(atoms),
                            "partition_role": partition,
                            "input_role": "raw_generated_pre_dft_unrelaxed_x0",
                            "source_member": info.filename,
                            "source_member_crc32": f"{info.CRC:08x}",
                            "source_member_bytes": int(info.file_size),
                        }
                    )
        finally:
            for archive in zip_outputs.values():
                archive.close()

        metadata = pd.DataFrame(metadata_rows).sort_values(
            "material_id", kind="stable", ignore_index=True
        )
        if metadata.empty or metadata["material_id"].duplicated().any():
            raise RuntimeError("NEXT84 geometry identity accounting differs")
        if metadata.groupby("reduced_formula")["partition_role"].nunique().max() != 1:
            raise RuntimeError("NEXT84 reduced-formula group leaked across partitions")
        metadata_path = staging / METADATA_NAME
        metadata.to_parquet(metadata_path, index=False)

        partitions: dict[str, object] = {}
        for role in PARTITIONS:
            part = metadata[metadata["partition_role"].eq(role)]
            lattice_counts = Counter(part["lattice_class"].astype(str))
            partitions[role] = {
                "rows": len(part),
                "reduced_formula_groups": int(part["reduced_formula"].nunique()),
                "material_id_list_sha256": _identity_hash(part["material_id"].astype(str).tolist()),
                "lattice_class_counts": dict(sorted(lattice_counts.items())),
                "geometry_archive": GEOMETRY_NAMES[role],
            }
        identities = {
            "protocol": PROTOCOL,
            "split_salt": SPLIT_SALT,
            "split_unit_interval": {
                "discovery": [0.0, 0.55],
                "internal_validation": [0.55, 0.75],
                "internal_replication": [0.75, 1.0],
            },
            "group_key": "reduced_formula",
            "rows": len(metadata),
            "reduced_formula_groups": int(metadata["reduced_formula"].nunique()),
            "partitions": partitions,
            "labels_opened": False,
        }
        identities_path = staging / IDENTITIES_NAME
        identities_path.write_bytes(_json_bytes(identities))

        output_paths = [
            metadata_path,
            identities_path,
            *(staging / GEOMETRY_NAMES[role] for role in PARTITIONS),
        ]
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "mode": "geometry_only_scigen_poscar_extraction_and_composition_split",
            "source_member_payload_allowlist": f"{ROOT}/<material_id>/POSCAR",
            "poscar_payloads_opened": len(metadata),
            "endpoint_payloads_opened": False,
            "relaxed_structures_opened": False,
            "labels_opened": False,
            "dft_values_used_by_features": False,
            "model_or_proxy_potential_used": False,
            "physical_relaxation_executed": False,
            "counts": {
                "rows": len(metadata),
                "reduced_formula_groups": int(metadata["reduced_formula"].nunique()),
                **{role: int(metadata["partition_role"].eq(role).sum()) for role in PARTITIONS},
            },
            "inputs_sha256": {
                name: {"path": str(path), "sha256": input_hashes[name]}
                for name, path in {
                    "source_archive": source,
                    "source_audit": audit_path,
                    "source_audit_manifest": audit_manifest_path,
                    "design": design,
                }.items()
            },
            "executed_source_sha256": {
                "src/next84_scigen_geometry_lockbox.py": source_code_hash
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if any(_sha256_file(path) != input_hashes[name] for name, path in {
            "source_archive": source,
            "source_audit": audit_path,
            "source_audit_manifest": audit_manifest_path,
            "design": design,
        }.items()):
            raise RuntimeError("NEXT84 input changed before publication")
        if _sha256_file(source_code) != source_code_hash:
            raise RuntimeError("NEXT84 source changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--source-audit-manifest", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    build_scigen_geometry_lockbox(
        source_archive_path=args.source_archive,
        source_audit_path=args.source_audit,
        source_audit_manifest_path=args.source_audit_manifest,
        design_path=args.design,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "GEOMETRY_NAMES",
    "MANIFEST_NAME",
    "METADATA_NAME",
    "PARTITIONS",
    "PROTOCOL",
    "assign_partition",
    "build_scigen_geometry_lockbox",
]
