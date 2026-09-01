"""Blind reroute of the official WyFormer x0/DFT cohort.

NEXT93 exposed aggregate labels for its validation and replication partitions.
Those artifacts remain preserved but are invalid for confirmation.  This
module fixes a new split salt before routing and never returns or manifests
partition-specific endpoint statistics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import pandas as pd
from pymatgen.core import Structure

from src.next93_wyformer_source_lockbox import (
    ARTICLE_ID,
    ENDPOINT_NAME,
    EXPECTED_ARTICLE_METADATA_SHA256,
    EXPECTED_DFT,
    EXPECTED_RAW,
    EXPECTED_README_MD5,
    INPUT_ROLE,
    PARTITIONS,
    _composition_key_from_structure_payload,
    _crystal_system,
    _endpoint_stratum,
    _md5_file,
    _sha256_file,
    _stable_material_id,
    _validate_figshare_metadata,
    _write_json,
)


PROTOCOL = "2026-08-03-next93b-wyformer-blind-reroute-v1"
SPLIT_SALT = "NEXT93B_WYFORMER_BLIND_REDUCED_FORMULA_SPLIT_V1"
EXPECTED_DESIGN_SHA256 = "5d456da527c94dfe6990ae4795ac88a732e50e39397920ef2e2846f0e2838848"
MANIFEST_NAME = "MANIFEST.json"
AUDIT_NAME = "NEXT93B_WYFORMER_BLIND_SOURCE_AUDIT.json"
METADATA_NAME = "wyformer_x0_metadata.parquet"
GEOMETRY_NAMES = {
    role: f"wyformer_x0_geometry_{role}.parquet" for role in PARTITIONS
}


def _blind_partition_for_reduced_formula(reduced_formula: str) -> str:
    if not isinstance(reduced_formula, str) or not reduced_formula:
        raise ValueError("reduced_formula must be a non-empty string")
    digest = hashlib.sha256(f"{SPLIT_SALT}\0{reduced_formula}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    if bucket < 5_500:
        return "discovery"
    if bucket < 7_750:
        return "internal_validation"
    return "internal_replication"


def _publish_directory(staging: Path, target: Path) -> None:
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    os.replace(staging, target)


def build_wyformer_blind_lockbox(
    *,
    raw_x0_path: Path,
    dft_success_path: Path,
    figshare_metadata_path: Path,
    readme_path: Path,
    design_path: Path,
    cohort_output_dir: Path,
    endpoint_output_dirs: Mapping[str, Path],
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Publish a new blind split without exposing endpoint aggregates."""

    inputs = {
        "raw_x0": Path(raw_x0_path).resolve(),
        "dft_source": Path(dft_success_path).resolve(),
        "figshare_metadata": Path(figshare_metadata_path).resolve(),
        "readme": Path(readme_path).resolve(),
        "design": Path(design_path).resolve(),
    }
    if any(not path.is_file() for path in inputs.values()):
        raise FileNotFoundError("NEXT93b input is missing")
    cohort_target = Path(cohort_output_dir).resolve()
    if set(endpoint_output_dirs) != set(PARTITIONS):
        raise ValueError("endpoint output roles differ")
    endpoint_targets = {role: Path(endpoint_output_dirs[role]).resolve() for role in PARTITIONS}
    all_targets = [cohort_target, *endpoint_targets.values()]
    if len(set(all_targets)) != len(all_targets):
        raise ValueError("NEXT93b output directories must be distinct")
    existing = [target for target in all_targets if os.path.lexists(target)]
    if existing:
        raise FileExistsError(str(existing[0]))

    input_sha256 = {name: _sha256_file(path) for name, path in inputs.items()}
    input_md5 = {name: _md5_file(path) for name, path in inputs.items()}
    article = json.loads(inputs["figshare_metadata"].read_text(encoding="utf-8"))
    if require_formal_inputs:
        if inputs["raw_x0"].stat().st_size != EXPECTED_RAW["size"]:
            raise ValueError("formal raw x0 size differs")
        if inputs["dft_source"].stat().st_size != EXPECTED_DFT["size"]:
            raise ValueError("formal DFT source size differs")
        if input_md5["raw_x0"] != EXPECTED_RAW["md5"]:
            raise ValueError("formal raw x0 MD5 differs")
        if input_md5["dft_source"] != EXPECTED_DFT["md5"]:
            raise ValueError("formal DFT source MD5 differs")
        if input_sha256["figshare_metadata"] != EXPECTED_ARTICLE_METADATA_SHA256:
            raise ValueError("formal Figshare metadata snapshot differs")
        if input_md5["readme"] != EXPECTED_README_MD5:
            raise ValueError("formal README identity differs")
        if input_sha256["design"] != EXPECTED_DESIGN_SHA256:
            raise ValueError("formal reroute design identity differs")
        _validate_figshare_metadata(article)

    raw = pd.read_csv(inputs["raw_x0"])
    dft = pd.read_csv(inputs["dft_source"])
    if {"material_id", "structure"} - set(raw.columns):
        raise ValueError("raw x0 columns differ")
    if {"material_id", "structure", "dft_e_above_hull_corrected"} - set(dft.columns):
        raise ValueError("DFT source columns differ")
    if require_formal_inputs and (len(raw) != EXPECTED_RAW["rows"] or len(dft) != EXPECTED_DFT["rows"]):
        raise ValueError("formal source row count differs")

    raw = raw.copy()
    dft = dft.copy()
    raw["_composition_key"] = raw["structure"].map(_composition_key_from_structure_payload)
    dft["_composition_key"] = dft["structure"].map(_composition_key_from_structure_payload)
    raw_counts = raw["_composition_key"].value_counts()
    dft_counts = dft["_composition_key"].value_counts()
    if set(dft_counts.index) - set(raw_counts.index):
        raise ValueError("DFT composition is absent from raw x0 source")
    unique_raw = raw[raw["_composition_key"].map(raw_counts).eq(1)].copy()
    unique_raw["_dft_count"] = unique_raw["_composition_key"].map(dft_counts).fillna(0).astype(int)
    if unique_raw["_dft_count"].gt(1).any():
        raise ValueError("a unique raw composition maps to multiple DFT rows")
    dft_unique = dft[dft["_composition_key"].map(dft_counts).eq(1)].set_index(
        "_composition_key", drop=False
    )

    metadata_rows: list[dict[str, object]] = []
    geometry_rows = {role: [] for role in PARTITIONS}
    endpoint_rows = {role: [] for role in PARTITIONS}
    seen_ids: set[str] = set()
    for raw_record in unique_raw.to_dict(orient="records"):
        composition_key = str(raw_record["_composition_key"])
        structure = Structure.from_dict(json.loads(str(raw_record["structure"])))
        material_id = _stable_material_id(raw_record["material_id"])
        if material_id in seen_ids:
            raise ValueError("stable x0 material id is duplicated")
        seen_ids.add(material_id)
        reduced_formula = structure.composition.reduced_formula
        role = _blind_partition_for_reduced_formula(reduced_formula)
        group_value = raw_record.get("group")
        space_group = None if pd.isna(group_value) else int(group_value)
        metadata_rows.append(
            {
                "material_id": material_id,
                "raw_material_id": int(raw_record["material_id"]),
                "full_composition_key": composition_key,
                "reduced_formula": reduced_formula,
                "chemical_system": "-".join(
                    sorted(str(element) for element in structure.composition.elements)
                ),
                "natoms": len(structure),
                "generated_space_group": space_group,
                "crystal_system": _crystal_system(space_group),
                "partition_role": role,
                "input_role": INPUT_ROLE,
            }
        )
        geometry_rows[role].append(
            {"material_id": material_id, "structure_json": str(raw_record["structure"])}
        )

        succeeded = int(raw_record["_dft_count"]) == 1
        energy: float | None = None
        published_id: object = None
        if succeeded:
            dft_record = dft_unique.loc[composition_key]
            energy = float(dft_record["dft_e_above_hull_corrected"])
            if not math.isfinite(energy):
                raise ValueError("successful DFT energy above hull is non-finite")
            published_id = dft_record["material_id"]
        endpoint_rows[role].append(
            {
                "material_id": material_id,
                "dft_succeeded": bool(succeeded),
                "dft_e_above_hull_corrected": energy,
                "endpoint_stratum": _endpoint_stratum(
                    dft_succeeded=succeeded, dft_e_above_hull=energy
                ),
                "published_permuted_dft_material_id": published_id,
            }
        )

    metadata = pd.DataFrame(metadata_rows).sort_values("material_id", kind="stable")
    if metadata["material_id"].duplicated().any():
        raise ValueError("formal cohort material ids differ")
    if not metadata.groupby("reduced_formula")["partition_role"].nunique().eq(1).all():
        raise ValueError("reduced-formula group crosses partitions")
    partition_rows = {
        role: int(metadata["partition_role"].eq(role).sum()) for role in PARTITIONS
    }
    audit = {
        "protocol": PROTOCOL,
        "article_id": ARTICLE_ID,
        "source_url": "https://figshare.com/articles/dataset/WyFormer_generated_structures/29094701",
        "raw_x0_figshare_file_id": EXPECTED_RAW["figshare_file_id"],
        "dft_success_figshare_file_id": EXPECTED_DFT["figshare_file_id"],
        "published_material_id_used_for_pairing": False,
        "pairing_key": "exact_full_cell_element_occupancies",
        "raw_rows": int(len(raw)),
        "formal_unique_x0_rows": int(len(unique_raw)),
        "excluded_ambiguous_raw_rows": int(len(raw) - len(unique_raw)),
        "prior_v1_aggregate_leakage_recorded": True,
        "input_sha256": input_sha256,
        "input_md5": input_md5,
    }

    for target in all_targets:
        target.parent.mkdir(parents=True, exist_ok=True)
    staging_cohort = Path(
        tempfile.mkdtemp(prefix=f".{cohort_target.name}.staging-", dir=cohort_target.parent)
    )
    staging_endpoints = {
        role: Path(
            tempfile.mkdtemp(
                prefix=f".{endpoint_targets[role].name}.staging-",
                dir=endpoint_targets[role].parent,
            )
        )
        for role in PARTITIONS
    }
    published: list[Path] = []
    try:
        metadata.to_parquet(staging_cohort / METADATA_NAME, index=False)
        _write_json(staging_cohort / AUDIT_NAME, audit)
        output_hashes = {
            METADATA_NAME: _sha256_file(staging_cohort / METADATA_NAME),
            AUDIT_NAME: _sha256_file(staging_cohort / AUDIT_NAME),
        }
        endpoint_hashes: dict[str, str] = {}
        for role in PARTITIONS:
            geometry = pd.DataFrame(
                geometry_rows[role], columns=["material_id", "structure_json"]
            ).sort_values("material_id", kind="stable")
            endpoint = pd.DataFrame(
                endpoint_rows[role],
                columns=[
                    "material_id",
                    "dft_succeeded",
                    "dft_e_above_hull_corrected",
                    "endpoint_stratum",
                    "published_permuted_dft_material_id",
                ],
            ).sort_values("material_id", kind="stable")
            geometry_path = staging_cohort / GEOMETRY_NAMES[role]
            endpoint_path = staging_endpoints[role] / ENDPOINT_NAME
            geometry.to_parquet(geometry_path, index=False)
            endpoint.to_parquet(endpoint_path, index=False)
            output_hashes[GEOMETRY_NAMES[role]] = _sha256_file(geometry_path)
            endpoint_hashes[role] = _sha256_file(endpoint_path)
            endpoint_manifest = {
                "protocol": PROTOCOL,
                "partition_role": role,
                "rows": int(len(endpoint)),
                "endpoint_payload_opened": False,
                "formula_or_threshold_fitted": False,
                "endpoint_definition_frozen_by_design_sha256": input_sha256["design"],
                "endpoint_sha256": endpoint_hashes[role],
            }
            _write_json(staging_endpoints[role] / MANIFEST_NAME, endpoint_manifest)

        source_sha256 = _sha256_file(Path(__file__).resolve())
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "source_sha256": source_sha256,
            "input_role": INPUT_ROLE,
            "prior_v1_validation_and_replication_invalidated": True,
            "prior_v1_artifacts_preserved": True,
            "labels_opened_by_feature_builder": False,
            "discovery_endpoint_opened": False,
            "validation_endpoint_opened": False,
            "replication_endpoint_opened": False,
            "relaxed_structures_published": False,
            "learned_proxy_execution_input": False,
            "split_salt": SPLIT_SALT,
            "split_unit": "whole_reduced_formula_group",
            "partition_rows": partition_rows,
            "input_sha256": input_sha256,
            "outputs_sha256": output_hashes,
            "endpoint_sha256": endpoint_hashes,
        }
        _write_json(staging_cohort / MANIFEST_NAME, manifest)

        for role in PARTITIONS:
            _publish_directory(staging_endpoints[role], endpoint_targets[role])
            published.append(endpoint_targets[role])
        _publish_directory(staging_cohort, cohort_target)
        published.append(cohort_target)
        return manifest
    except Exception:
        for staging in [staging_cohort, *staging_endpoints.values()]:
            if staging.exists():
                shutil.rmtree(staging)
        for target in reversed(published):
            if target.exists():
                shutil.rmtree(target)
        raise


__all__ = [
    "ENDPOINT_NAME",
    "GEOMETRY_NAMES",
    "MANIFEST_NAME",
    "METADATA_NAME",
    "PARTITIONS",
    "PROTOCOL",
    "SPLIT_SALT",
    "build_wyformer_blind_lockbox",
]
