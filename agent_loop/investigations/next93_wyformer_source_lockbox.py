"""Freeze a provenance-safe WyFormer x0 cohort and isolated DFT endpoints.

The executable-law side of this protocol receives only the raw DiffCSP++
structure.  Published DFT results are used here solely to construct physically
separate offline endpoint payloads.  Published material identifiers are not
used for pairing because the source README says they were permuted.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Mapping

import pandas as pd
from pymatgen.core import Structure


PROTOCOL = "2026-08-03-next93-wyformer-official-source-lockbox-v1"
ARTICLE_ID = 29094701
PARTITIONS = ("discovery", "internal_validation", "internal_replication")
SPLIT_SALT = "NEXT93_WYFORMER_REDUCED_FORMULA_SPLIT_V1"
PROTECTED_MAX_EV_PER_ATOM = 0.10
SEVERE_MIN_EV_PER_ATOM = 0.50
INPUT_ROLE = "raw_generated_pre_chgnet_pre_dft_unrelaxed_x0"

MANIFEST_NAME = "MANIFEST.json"
AUDIT_NAME = "NEXT93_WYFORMER_SOURCE_AUDIT.json"
METADATA_NAME = "wyformer_x0_metadata.parquet"
GEOMETRY_NAMES = {
    role: f"wyformer_x0_geometry_{role}.parquet" for role in PARTITIONS
}
ENDPOINT_NAME = "wyformer_dft_screening_endpoints.parquet"

EXPECTED_DESIGN_SHA256 = "db9e05470132d57002b62b408b4c0ed3ee39201a61fe6586610b70f1123cbc77"
EXPECTED_ARTICLE_METADATA_SHA256 = (
    "6121efdcc2d17c8857bd54504da0bb9d9c50ec997ad07e9d2c546f5c2b359621"
)
EXPECTED_README_MD5 = "6cf255309977c869a5ae69942c3fa6e2"
EXPECTED_RAW = {
    "figshare_file_id": 54711179,
    "rows": 9999,
    "size": 7_541_258,
    "md5": "1792fbb96eeafd5bfa2f5a7b75012e3a",
}
EXPECTED_DFT = {
    "figshare_file_id": 54711188,
    "rows": 9623,
    "size": 18_472_069,
    "md5": "07f9d4c713f5bed2f584b6b801096b58",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_json_bytes(value))


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _composition_key_from_structure_payload(payload: str | Mapping[str, object]) -> str:
    """Return an exact, site-order-independent full-cell composition key."""

    data = json.loads(payload) if isinstance(payload, str) else dict(payload)
    sites = data.get("sites")
    if not isinstance(sites, list) or not sites:
        raise ValueError("structure payload has no sites")
    counts: Counter[str] = Counter()
    exact: dict[str, Fraction] = {}
    for site in sites:
        if not isinstance(site, Mapping) or not isinstance(site.get("species"), list):
            raise ValueError("structure site has no species list")
        for species in site["species"]:
            if not isinstance(species, Mapping) or not isinstance(species.get("element"), str):
                raise ValueError("structure species entry differs")
            symbol = str(species["element"])
            occupancy = Fraction(str(species.get("occu", 1))).limit_denominator(1_000_000)
            if occupancy <= 0:
                raise ValueError("structure has non-positive occupancy")
            exact[symbol] = exact.get(symbol, Fraction(0)) + occupancy
            counts[symbol] += 1
    if not exact:
        raise ValueError("structure composition is empty")
    # Include exact occupancies rather than merely the number of sites.  This
    # supports ordered formal data and fails safely for disordered payloads.
    return json.dumps(
        [[symbol, _fraction_text(exact[symbol])] for symbol in sorted(exact)],
        separators=(",", ":"),
    )


def _partition_for_reduced_formula(reduced_formula: str) -> str:
    if not isinstance(reduced_formula, str) or not reduced_formula:
        raise ValueError("reduced_formula must be a non-empty string")
    digest = hashlib.sha256(f"{SPLIT_SALT}\0{reduced_formula}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    if bucket < 5_500:
        return "discovery"
    if bucket < 7_750:
        return "internal_validation"
    return "internal_replication"


def _endpoint_stratum(*, dft_succeeded: bool, dft_e_above_hull: float | None) -> str:
    if not dft_succeeded:
        return "severe"
    if dft_e_above_hull is None or not math.isfinite(float(dft_e_above_hull)):
        raise ValueError("a successful DFT row requires finite corrected energy above hull")
    value = float(dft_e_above_hull)
    if value <= PROTECTED_MAX_EV_PER_ATOM:
        return "protected"
    if value >= SEVERE_MIN_EV_PER_ATOM:
        return "severe"
    return "middle"


def _crystal_system(space_group: int | None) -> str:
    if space_group is None:
        return "unknown"
    number = int(space_group)
    if 1 <= number <= 2:
        return "triclinic"
    if number <= 15:
        return "monoclinic"
    if number <= 74:
        return "orthorhombic"
    if number <= 142:
        return "tetragonal"
    if number <= 167:
        return "trigonal"
    if number <= 194:
        return "hexagonal"
    if number <= 230:
        return "cubic"
    return "unknown"


def _stable_material_id(raw_material_id: object) -> str:
    try:
        return f"WYF-MP20-{int(raw_material_id):05d}"
    except (TypeError, ValueError, OverflowError):
        digest = hashlib.sha256(str(raw_material_id).encode("utf-8")).hexdigest()[:16]
        return f"WYF-MP20-{digest}"


def _validate_figshare_metadata(metadata: Mapping[str, object]) -> None:
    if int(metadata.get("id", -1)) != ARTICLE_ID:
        raise ValueError("Figshare article id differs")
    files = metadata.get("files")
    if not isinstance(files, list):
        raise ValueError("Figshare files metadata differs")
    by_id = {int(item["id"]): item for item in files if isinstance(item, Mapping) and "id" in item}
    for expected in (EXPECTED_RAW, EXPECTED_DFT):
        item = by_id.get(int(expected["figshare_file_id"]))
        if item is None:
            raise ValueError("required Figshare file id is missing")
        if int(item.get("size", -1)) != expected["size"] or item.get("computed_md5") != expected["md5"]:
            raise ValueError("required Figshare file identity differs")


def _publish_directory(staging: Path, target: Path) -> None:
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    os.replace(staging, target)


def build_wyformer_source_lockbox(
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
    """Pair unique compositions and publish x0/endpoint payloads separately."""

    inputs = {
        "raw_x0": Path(raw_x0_path).resolve(),
        "dft_success": Path(dft_success_path).resolve(),
        "figshare_metadata": Path(figshare_metadata_path).resolve(),
        "readme": Path(readme_path).resolve(),
        "design": Path(design_path).resolve(),
    }
    if any(not path.is_file() for path in inputs.values()):
        raise FileNotFoundError("NEXT93 input is missing")
    cohort_target = Path(cohort_output_dir).resolve()
    if set(endpoint_output_dirs) != set(PARTITIONS):
        raise ValueError("endpoint output roles differ")
    endpoint_targets = {role: Path(endpoint_output_dirs[role]).resolve() for role in PARTITIONS}
    all_targets = [cohort_target, *endpoint_targets.values()]
    if len(set(all_targets)) != len(all_targets):
        raise ValueError("NEXT93 output directories must be distinct")
    existing = [target for target in all_targets if os.path.lexists(target)]
    if existing:
        raise FileExistsError(str(existing[0]))

    input_sha256 = {name: _sha256_file(path) for name, path in inputs.items()}
    input_md5 = {name: _md5_file(path) for name, path in inputs.items()}
    metadata_json = json.loads(inputs["figshare_metadata"].read_text(encoding="utf-8"))
    if require_formal_inputs:
        if inputs["raw_x0"].stat().st_size != EXPECTED_RAW["size"]:
            raise ValueError("formal raw x0 size differs")
        if inputs["dft_success"].stat().st_size != EXPECTED_DFT["size"]:
            raise ValueError("formal DFT source size differs")
        if input_md5["raw_x0"] != EXPECTED_RAW["md5"]:
            raise ValueError("formal raw x0 MD5 differs")
        if input_md5["dft_success"] != EXPECTED_DFT["md5"]:
            raise ValueError("formal DFT source MD5 differs")
        if input_sha256["figshare_metadata"] != EXPECTED_ARTICLE_METADATA_SHA256:
            raise ValueError("formal Figshare metadata snapshot differs")
        if input_md5["readme"] != EXPECTED_README_MD5:
            raise ValueError("formal README identity differs")
        if input_sha256["design"] != EXPECTED_DESIGN_SHA256:
            raise ValueError("formal design identity differs")
        _validate_figshare_metadata(metadata_json)

    raw = pd.read_csv(inputs["raw_x0"])
    dft = pd.read_csv(inputs["dft_success"])
    if {"material_id", "structure"} - set(raw.columns):
        raise ValueError("raw x0 columns differ")
    required_dft = {"material_id", "structure", "dft_e_above_hull_corrected"}
    if required_dft - set(dft.columns):
        raise ValueError("DFT source columns differ")
    if require_formal_inputs and (len(raw) != EXPECTED_RAW["rows"] or len(dft) != EXPECTED_DFT["rows"]):
        raise ValueError("formal source row count differs")

    raw = raw.copy()
    dft = dft.copy()
    raw["_composition_key"] = raw["structure"].map(_composition_key_from_structure_payload)
    dft["_composition_key"] = dft["structure"].map(_composition_key_from_structure_payload)
    raw_counts = raw["_composition_key"].value_counts()
    dft_counts = dft["_composition_key"].value_counts()
    unknown_dft_keys = set(dft_counts.index) - set(raw_counts.index)
    if unknown_dft_keys:
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
        chemical_system = "-".join(sorted(str(el) for el in structure.composition.elements))
        role = _partition_for_reduced_formula(reduced_formula)
        group_value = raw_record.get("group")
        space_group = None if pd.isna(group_value) else int(group_value)
        metadata_rows.append(
            {
                "material_id": material_id,
                "raw_material_id": int(raw_record["material_id"]),
                "full_composition_key": composition_key,
                "reduced_formula": reduced_formula,
                "chemical_system": chemical_system,
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

        dft_succeeded = int(raw_record["_dft_count"]) == 1
        energy: float | None = None
        published_dft_id: object = None
        if dft_succeeded:
            dft_record = dft_unique.loc[composition_key]
            energy = float(dft_record["dft_e_above_hull_corrected"])
            if not math.isfinite(energy):
                raise ValueError("successful DFT energy above hull is non-finite")
            published_dft_id = dft_record["material_id"]
        endpoint_rows[role].append(
            {
                "material_id": material_id,
                "dft_succeeded": bool(dft_succeeded),
                "dft_e_above_hull_corrected": energy,
                "endpoint_stratum": _endpoint_stratum(
                    dft_succeeded=dft_succeeded, dft_e_above_hull=energy
                ),
                "published_permuted_dft_material_id": published_dft_id,
            }
        )

    metadata_frame = pd.DataFrame(metadata_rows).sort_values("material_id", kind="stable")
    if metadata_frame["material_id"].duplicated().any():
        raise ValueError("formal cohort material ids differ")
    group_roles = metadata_frame.groupby("reduced_formula")["partition_role"].nunique()
    if not group_roles.eq(1).all():
        raise ValueError("reduced-formula group crosses partitions")

    pairing = {
        "raw_rows": int(len(raw)),
        "dft_success_rows": int(len(dft)),
        "raw_full_composition_keys": int(raw_counts.size),
        "formal_unique_x0_rows": int(len(unique_raw)),
        "excluded_ambiguous_raw_rows": int(len(raw) - len(unique_raw)),
        "matched_dft_success_rows": int(unique_raw["_dft_count"].eq(1).sum()),
        "unmatched_dft_failure_rows": int(unique_raw["_dft_count"].eq(0).sum()),
        "published_material_id_used_for_pairing": False,
        "pairing_key": "exact_full_cell_element_occupancies",
    }
    audit = {
        "protocol": PROTOCOL,
        "article_id": ARTICLE_ID,
        "source_url": "https://figshare.com/articles/dataset/WyFormer_generated_structures/29094701",
        "raw_x0_figshare_file_id": EXPECTED_RAW["figshare_file_id"],
        "dft_success_figshare_file_id": EXPECTED_DFT["figshare_file_id"],
        "raw_x0_role": INPUT_ROLE,
        "dft_role": "offline_endpoint_only_after_chgnet_prerelax_and_mp_gga_double_relax_static",
        "less_strict_dft_gga_relax_1_excluded": True,
        "pairing": pairing,
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
        metadata_frame.to_parquet(staging_cohort / METADATA_NAME, index=False)
        _write_json(staging_cohort / AUDIT_NAME, audit)
        cohort_output_hashes: dict[str, str] = {
            METADATA_NAME: _sha256_file(staging_cohort / METADATA_NAME),
            AUDIT_NAME: _sha256_file(staging_cohort / AUDIT_NAME),
        }
        partition_counts: dict[str, dict[str, int]] = {}
        endpoint_hashes: dict[str, str] = {}
        for role in PARTITIONS:
            geometry_frame = pd.DataFrame(
                geometry_rows[role], columns=["material_id", "structure_json"]
            ).sort_values("material_id", kind="stable")
            endpoint_frame = pd.DataFrame(
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
            geometry_frame.to_parquet(geometry_path, index=False)
            endpoint_frame.to_parquet(endpoint_path, index=False)
            cohort_output_hashes[GEOMETRY_NAMES[role]] = _sha256_file(geometry_path)
            endpoint_hash = _sha256_file(endpoint_path)
            endpoint_hashes[role] = endpoint_hash
            partition_counts[role] = {
                "rows": int(len(endpoint_frame)),
                "dft_success": int(endpoint_frame["dft_succeeded"].sum()),
                "dft_failure": int((~endpoint_frame["dft_succeeded"]).sum()),
                "protected": int(endpoint_frame["endpoint_stratum"].eq("protected").sum()),
                "middle": int(endpoint_frame["endpoint_stratum"].eq("middle").sum()),
                "severe": int(endpoint_frame["endpoint_stratum"].eq("severe").sum()),
            }
            endpoint_manifest = {
                "protocol": PROTOCOL,
                "partition_role": role,
                "endpoint_definition": {
                    "protected": "dft_succeeded and e_hull_corrected <= 0.10 eV/atom",
                    "middle": "dft_succeeded and 0.10 < e_hull_corrected < 0.50 eV/atom",
                    "severe": "dft_failed or e_hull_corrected >= 0.50 eV/atom",
                },
                "endpoint_payload_opened": False,
                "formula_or_threshold_fitted": False,
                "rows": int(len(endpoint_frame)),
                "endpoint_sha256": endpoint_hash,
                "source_input_sha256": input_sha256,
            }
            _write_json(staging_endpoints[role] / MANIFEST_NAME, endpoint_manifest)

        cohort_manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "input_role": INPUT_ROLE,
            "labels_opened_by_feature_builder": False,
            "discovery_endpoint_opened": False,
            "validation_endpoint_opened": False,
            "replication_endpoint_opened": False,
            "relaxed_structures_published": False,
            "learned_proxy_execution_input": False,
            "split_salt": SPLIT_SALT,
            "split_unit": "whole_reduced_formula_group",
            "pairing": pairing,
            "partition_counts": partition_counts,
            "input_sha256": input_sha256,
            "outputs_sha256": cohort_output_hashes,
            "endpoint_sha256": endpoint_hashes,
        }
        _write_json(staging_cohort / MANIFEST_NAME, cohort_manifest)

        # Publish endpoint directories first and the cohort manifest last.  A
        # consumer cannot observe a cohort that points to absent endpoints.
        for role in PARTITIONS:
            _publish_directory(staging_endpoints[role], endpoint_targets[role])
            published.append(endpoint_targets[role])
        _publish_directory(staging_cohort, cohort_target)
        published.append(cohort_target)
        return cohort_manifest
    except Exception:
        for staging in [staging_cohort, *staging_endpoints.values()]:
            if staging.exists():
                shutil.rmtree(staging)
        # Only remove directories published by this failed transaction.
        for target in reversed(published):
            if target.exists():
                shutil.rmtree(target)
        raise


__all__ = [
    "AUDIT_NAME",
    "ENDPOINT_NAME",
    "GEOMETRY_NAMES",
    "MANIFEST_NAME",
    "METADATA_NAME",
    "PARTITIONS",
    "PROTOCOL",
    "build_wyformer_source_lockbox",
]
