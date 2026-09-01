"""Freeze the official SCIGEN DFT source identity without opening endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Mapping, Sequence
import zipfile

from src.next12_dft_queue import _json_bytes
from src.next13d_acsc_dft_pairs import _sha256_file
from src.next14_wbm_holdout import _publish_directory_no_replace


PROTOCOL = "2026-08-03-next83-scigen-official-source-audit-v1"
AUDIT_NAME = "SCIGEN_SOURCE_AUDIT.json"
MANIFEST_NAME = "MANIFEST.json"
EXPECTED_ARCHIVE_SIZE = 28_446_599
EXPECTED_ARCHIVE_MD5 = "fc217e45c5dd8920d08c523177546d45"
EXPECTED_ARCHIVE_SHA256 = (
    "7eb1b48200329e8d294d013c56767c2219020731dc9a44e36c23b83ac0914068"
)
EXPECTED_METADATA_SHA256 = (
    "82d9696e672b0206c0e3d3a023891a55d656927e18cb8b7db78cd667dc60af63"
)
EXPECTED_DESIGN_SHA256 = (
    "39127f4d2b5ddba176f7904ed498f98e0326fa902e1c3ede79fbbcf320c13ee9"
)
EXPECTED_ROWS = 24_742
EXPECTED_FIGSHARE_ARTICLE_ID = 26_082_733
EXPECTED_FIGSHARE_FILE_ID = 57_245_942
EXPECTED_DOI = "10.6084/m9.figshare.26082733.v3"
EXPECTED_LICENSE = "CC BY 4.0"
EXPECTED_ARCHIVE_NAME = "03_scigen_materials_relaxed.zip"
ROOT = "03_scigen_materials_relaxed"
EXPECTED_SUPPLEMENTARY_TABLES = (
    "si_table_elt.csv",
    "si_table_hon.csv",
    "si_table_kag.csv",
    "si_table_lieb.csv",
    "si_table_snh.csv",
    "si_table_sns.csv",
    "si_table_sqr.csv",
    "si_table_srt.csv",
    "si_table_trh.csv",
    "si_table_tri.csv",
    "si_table_tsq.csv",
)


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, Mapping):
        raise ValueError("Figshare metadata must be a JSON object")
    return value


def _official_file(metadata: Mapping[str, object]) -> Mapping[str, object]:
    files = metadata.get("files")
    if not isinstance(files, list):
        raise ValueError("Figshare metadata lacks files")
    matches = [
        value
        for value in files
        if isinstance(value, Mapping)
        and value.get("id") == EXPECTED_FIGSHARE_FILE_ID
        and value.get("name") == EXPECTED_ARCHIVE_NAME
    ]
    if len(matches) != 1:
        raise ValueError("Figshare metadata lacks the unique official SCIGEN archive")
    return matches[0]


def _member_audit(
    archive_path: Path,
    *,
    expected_rows: int,
    expected_supplementary_tables: Sequence[str],
) -> dict[str, object]:
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise ValueError("SCIGEN archive has duplicate member names")
    if any(info.is_dir() for info in infos):
        raise ValueError("SCIGEN archive unexpectedly contains directory members")

    poscar: dict[str, zipfile.ZipInfo] = {}
    contcar: dict[str, zipfile.ZipInfo] = {}
    output_names: list[str] = []
    supplementary: list[str] = []
    for info in infos:
        parts = PurePosixPath(info.filename).parts
        if len(parts) == 3 and parts[0] == ROOT and parts[2] in {"POSCAR", "CONTCAR"}:
            material_id = parts[1]
            if not material_id or "/" in material_id:
                raise ValueError("SCIGEN material ID is invalid")
            target = poscar if parts[2] == "POSCAR" else contcar
            if material_id in target:
                raise ValueError("SCIGEN archive repeats a structure member")
            target[material_id] = info
        elif parts == (ROOT, "output.dat"):
            output_names.append(info.filename)
        elif len(parts) == 2 and parts[0] == ROOT and parts[1].startswith("si_table_") and parts[1].endswith(".csv"):
            supplementary.append(parts[1])
        else:
            raise ValueError(f"unexpected SCIGEN archive member: {info.filename}")
    if set(poscar) != set(contcar) or len(poscar) != expected_rows:
        raise ValueError("SCIGEN POSCAR/CONTCAR identity pairing differs")
    if len(output_names) != 1:
        raise ValueError("SCIGEN aggregate output table count differs")
    if tuple(sorted(supplementary)) != tuple(sorted(expected_supplementary_tables)):
        raise ValueError("SCIGEN supplementary table inventory differs")
    material_ids = sorted(poscar)
    identity_sha256 = hashlib.sha256(
        ("\n".join(material_ids) + "\n").encode("utf-8")
    ).hexdigest()
    return {
        "counts": {
            "zip_members": len(infos),
            "unique_material_ids": len(material_ids),
            "poscar_members": len(poscar),
            "contcar_members": len(contcar),
            "output_table_members": len(output_names),
            "supplementary_table_members": len(supplementary),
        },
        "material_id_list_sha256": identity_sha256,
        "supplementary_tables": sorted(supplementary),
    }


def audit_scigen_source(
    *,
    source_archive_path: Path,
    figshare_metadata_path: Path,
    design_path: Path,
    output_dir: Path,
    expected_rows: int = EXPECTED_ROWS,
    expected_supplementary_tables: Sequence[str] = EXPECTED_SUPPLEMENTARY_TABLES,
    expected_size: int = EXPECTED_ARCHIVE_SIZE,
    expected_md5: str = EXPECTED_ARCHIVE_MD5,
    expected_sha256: str = EXPECTED_ARCHIVE_SHA256,
    require_formal_inputs: bool = True,
    prior_label_free_poscar_probe_count: int = 0,
) -> dict[str, object]:
    """Audit archive identity and central-directory structure without payload reads."""

    source = Path(source_archive_path).resolve()
    metadata_path = Path(figshare_metadata_path).resolve()
    design = Path(design_path).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in (source, metadata_path, design)):
        raise FileNotFoundError("NEXT83 source, metadata, or design input is missing")
    if type(expected_rows) is not int or expected_rows <= 0:
        raise ValueError("expected_rows must be a positive exact integer")
    if (
        type(prior_label_free_poscar_probe_count) is not int
        or prior_label_free_poscar_probe_count < 0
    ):
        raise ValueError("prior POSCAR probe count is invalid")

    actual_size = source.stat().st_size
    actual_md5 = _md5_file(source)
    actual_sha256 = _sha256_file(source)
    metadata_sha256 = _sha256_file(metadata_path)
    design_sha256 = _sha256_file(design)
    if require_formal_inputs and (
        actual_size != expected_size
        or actual_md5 != expected_md5
        or actual_sha256 != expected_sha256
        or metadata_sha256 != EXPECTED_METADATA_SHA256
        or design_sha256 != EXPECTED_DESIGN_SHA256
    ):
        raise ValueError("SCIGEN source identity differs from the frozen formal input")

    metadata = _strict_json(metadata_path)
    license_value = metadata.get("license")
    if (
        metadata.get("id") != EXPECTED_FIGSHARE_ARTICLE_ID
        or metadata.get("doi") != EXPECTED_DOI
        or not isinstance(license_value, Mapping)
        or license_value.get("name") != EXPECTED_LICENSE
    ):
        raise ValueError("SCIGEN Figshare article metadata differs")
    official_file = _official_file(metadata)
    if (
        official_file.get("size") != actual_size
        or official_file.get("computed_md5") != actual_md5
    ):
        raise ValueError("SCIGEN local archive differs from Figshare metadata")

    inventory = _member_audit(
        source,
        expected_rows=expected_rows,
        expected_supplementary_tables=expected_supplementary_tables,
    )
    audit = {
        "protocol": PROTOCOL,
        "source": {
            "path": str(source),
            "figshare_article_id": EXPECTED_FIGSHARE_ARTICLE_ID,
            "figshare_file_id": EXPECTED_FIGSHARE_FILE_ID,
            "doi": EXPECTED_DOI,
            "license": EXPECTED_LICENSE,
            "bytes": actual_size,
            "md5": actual_md5,
            "sha256": actual_sha256,
        },
        **inventory,
        "zip_central_directory_opened": True,
        "archive_member_payloads_opened_by_audit": False,
        "endpoint_payloads_opened": False,
        "relaxed_structures_opened": False,
        "labels_opened": False,
        "prior_label_free_poscar_probe_count": prior_label_free_poscar_probe_count,
        "prior_probe_role": "eligibility_and_runtime_probe_only",
        "eligible_execution_input_member": f"{ROOT}/<material_id>/POSCAR",
        "forbidden_pre_freeze_members": [
            f"{ROOT}/<material_id>/CONTCAR",
            f"{ROOT}/output.dat",
            f"{ROOT}/si_table_*.csv",
        ],
        "scientific_improvement_claim": False,
    }
    source_code = Path(__file__).resolve()
    source_code_hash = _sha256_file(source_code)
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "source_identity_and_zip_central_directory_audit",
        "zip_central_directory_opened": True,
        "archive_member_payloads_opened_by_audit": False,
        "labels_opened": False,
        "relaxed_structures_opened": False,
        "inputs_sha256": {
            "source_archive": {"path": str(source), "sha256": actual_sha256},
            "figshare_metadata": {
                "path": str(metadata_path),
                "sha256": metadata_sha256,
            },
            "design": {"path": str(design), "sha256": design_sha256},
        },
        "executed_source_sha256": {
            "src/next83_scigen_source_audit.py": source_code_hash
        },
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        audit_path = staging / AUDIT_NAME
        audit_path.write_bytes(_json_bytes(audit))
        manifest["outputs_sha256"] = {AUDIT_NAME: _sha256_file(audit_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if (
            source.stat().st_size != actual_size
            or _md5_file(source) != actual_md5
            or _sha256_file(source) != actual_sha256
            or _sha256_file(metadata_path) != metadata_sha256
            or _sha256_file(design) != design_sha256
            or _sha256_file(source_code) != source_code_hash
        ):
            raise RuntimeError("NEXT83 input or source changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--figshare-metadata", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-label-free-poscar-probe-count", type=int, default=0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    audit_scigen_source(
        source_archive_path=args.source_archive,
        figshare_metadata_path=args.figshare_metadata,
        design_path=args.design,
        output_dir=args.output_dir,
        prior_label_free_poscar_probe_count=args.prior_label_free_poscar_probe_count,
    )


if __name__ == "__main__":
    main()


__all__ = ["AUDIT_NAME", "MANIFEST_NAME", "PROTOCOL", "audit_scigen_source"]
