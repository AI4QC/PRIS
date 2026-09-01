"""Mechanically route SCIGEN DFT distortion endpoints into physical splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping
import zipfile

import pandas as pd

from src.next12_dft_queue import _json_bytes
from src.next13d_acsc_dft_pairs import _sha256_file
from src.next14_wbm_holdout import _publish_directory_no_replace
from src.next84_scigen_geometry_lockbox import (
    MANIFEST_NAME as COHORT_MANIFEST_NAME,
    METADATA_NAME,
    PARTITIONS,
    PROTOCOL as COHORT_PROTOCOL,
)
from src.next86_scigen_term_catalogue import (
    CATALOGUE_NAME,
    MANIFEST_NAME as TERM_MANIFEST_NAME,
    PROTOCOL as TERM_PROTOCOL,
)


PROTOCOL = "2026-08-03-next86-scigen-physical-endpoint-routing-v1"
ENDPOINT_NAME = "scigen_dft_distortion_endpoints.parquet"
MANIFEST_NAME = "MANIFEST.json"
SOURCE_MEMBER = "03_scigen_materials_relaxed/output.dat"
FORCE_THRESHOLD = 0.01
LATTICE_THRESHOLDS = {
    "tri": (1.0, 0.5),
    "hon": (1.0, 0.5),
    "kag": (1.0, 0.5),
    "sqr": (1.0, 0.5),
    "elt": (1.0, 0.5),
    "sns": (1.0, 0.5),
    "lieb": (1.0, 0.5),
    "tsq": (2.0, 1.0),
    "srt": (2.0, 1.0),
    "snh": (2.0, 1.0),
    "trh": (3.0, 1.5),
}
EXPECTED_INPUT_SHA256 = {
    "source_archive": "7eb1b48200329e8d294d013c56767c2219020731dc9a44e36c23b83ac0914068",
    "cohort_manifest": "dc5bf33c6ce6dc2c10bcd3704688055058145fbe7269ada23ffbe4b141d75fe7",
    "metadata": "f91455f23b0a96f60fd1c779249e2be46a7ecf94fcdde2b146426a95aac05bde",
    "term_manifest": "5b80f948a35a40ef79438ea1902b92a40dd07c35a4b541826252eb92cf96f1eb",
    "term_catalogue": "e8f9fe532c15673c0a74737632b0145d43f6494cb1ea7e94e7380198fd4e4dee",
    "design": "39127f4d2b5ddba176f7904ed498f98e0326fa902e1c3ede79fbbcf320c13ee9",
}


def _read_json(path: Path, *, role: str) -> Mapping[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _parse_endpoint_payload(payload: bytes) -> pd.DataFrame:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("SCIGEN endpoint table is not strict UTF-8") from exc
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or not stripped.replace("-", "").strip()
        ):
            continue
        parts = stripped.split()
        if len(parts) != 9:
            raise ValueError(f"SCIGEN endpoint row {line_number} has {len(parts)} fields")
        material_id = parts[0]
        try:
            maximum_force = float(parts[5])
            lattice_change = float(parts[7])
            coordinate_change = float(parts[8])
        except ValueError as exc:
            raise ValueError(f"SCIGEN endpoint row {line_number} is not numeric") from exc
        values = (maximum_force, lattice_change, coordinate_change)
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError(f"SCIGEN endpoint row {line_number} is nonfinite or negative")
        rows.append(
            {
                "material_id": material_id,
                "final_max_force_ev_per_a": maximum_force,
                "lattice_change_a": lattice_change,
                "coordinate_change_a": coordinate_change,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty or table["material_id"].duplicated().any():
        raise ValueError("SCIGEN endpoint identity is empty or duplicated")
    return table.sort_values("material_id", kind="stable", ignore_index=True)


def _identity_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode("utf-8")).hexdigest()


def route_scigen_endpoints(
    *,
    source_archive_path: Path,
    cohort_dir: Path,
    term_catalogue_dir: Path,
    design_path: Path,
    output_dirs: Mapping[str, Path],
    require_formal_inputs: bool = True,
) -> dict[str, dict[str, object]]:
    """Route endpoint rows without printing, ranking, plotting, or summarizing values."""

    source = Path(source_archive_path).resolve()
    cohort = Path(cohort_dir).resolve()
    terms = Path(term_catalogue_dir).resolve()
    design = Path(design_path).resolve()
    targets = {role: Path(output_dirs[role]).resolve() for role in PARTITIONS}
    if set(output_dirs) != set(PARTITIONS):
        raise ValueError("NEXT86 endpoint output roles differ")
    if len(set(targets.values())) != len(PARTITIONS):
        raise ValueError("NEXT86 endpoint outputs must be physically distinct")
    if any(os.path.lexists(path) for path in targets.values()):
        raise FileExistsError("refusing to overwrite a SCIGEN endpoint split")
    paths = {
        "source_archive": source,
        "cohort_manifest": cohort / COHORT_MANIFEST_NAME,
        "metadata": cohort / METADATA_NAME,
        "term_manifest": terms / TERM_MANIFEST_NAME,
        "term_catalogue": terms / CATALOGUE_NAME,
        "design": design,
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT86 endpoint router input is missing")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT86 endpoint formal input identity differs")

    cohort_manifest = _read_json(paths["cohort_manifest"], role="NEXT84 manifest")
    cohort_outputs = cohort_manifest.get("outputs_sha256")
    if (
        cohort_manifest.get("protocol") != COHORT_PROTOCOL
        or cohort_manifest.get("labels_opened") is not False
        or cohort_manifest.get("endpoint_payloads_opened") is not False
        or cohort_manifest.get("relaxed_structures_opened") is not False
        or not isinstance(cohort_outputs, Mapping)
        or cohort_outputs.get(METADATA_NAME) != hashes["metadata"]
    ):
        raise ValueError("NEXT84 geometry-lockbox provenance differs")
    term_manifest = _read_json(paths["term_manifest"], role="NEXT86 term manifest")
    term_outputs = term_manifest.get("outputs_sha256")
    term_catalogue = _read_json(paths["term_catalogue"], role="NEXT86 term catalogue")
    if (
        term_manifest.get("protocol") != TERM_PROTOCOL
        or term_manifest.get("labels_opened") is not False
        or term_manifest.get("endpoint_payloads_opened") is not False
        or not isinstance(term_outputs, Mapping)
        or term_outputs.get(CATALOGUE_NAME) != hashes["term_catalogue"]
        or term_catalogue.get("protocol") != TERM_PROTOCOL
        or term_catalogue.get("labels_opened") is not False
    ):
        raise ValueError("NEXT86 term catalogue was not frozen label-free")

    metadata = pd.read_parquet(paths["metadata"])
    required = {"material_id", "lattice_class", "partition_role"}
    if required - set(metadata.columns) or metadata["material_id"].duplicated().any():
        raise ValueError("NEXT84 metadata identity differs")
    if set(metadata["partition_role"]) - set(PARTITIONS):
        raise ValueError("NEXT84 partition roles differ")

    with zipfile.ZipFile(source) as archive:
        payload = archive.read(SOURCE_MEMBER)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    endpoints = _parse_endpoint_payload(payload)
    metadata_core = metadata.loc[:, ["material_id", "lattice_class", "partition_role"]].copy()
    if set(endpoints["material_id"]) != set(metadata_core["material_id"]):
        raise ValueError("SCIGEN endpoint and x0 cohort identity sets differ")
    joined = metadata_core.merge(endpoints, on="material_id", how="inner", validate="one_to_one")
    if len(joined) != len(metadata_core):
        raise ValueError("SCIGEN endpoint and x0 cohort identity join differs")

    lattice_thresholds: list[float] = []
    coordinate_thresholds: list[float] = []
    for lattice_class in joined["lattice_class"].astype(str):
        if lattice_class not in LATTICE_THRESHOLDS:
            raise ValueError(f"unknown SCIGEN lattice class: {lattice_class}")
        lattice_threshold, coordinate_threshold = LATTICE_THRESHOLDS[lattice_class]
        lattice_thresholds.append(lattice_threshold)
        coordinate_thresholds.append(coordinate_threshold)
    joined["lattice_threshold_a"] = lattice_thresholds
    joined["coordinate_threshold_a"] = coordinate_thresholds
    ratios = pd.DataFrame(
        {
            "force": joined["final_max_force_ev_per_a"].to_numpy(float) / FORCE_THRESHOLD,
            "lattice": joined["lattice_change_a"].to_numpy(float)
            / joined["lattice_threshold_a"].to_numpy(float),
            "coordinate": joined["coordinate_change_a"].to_numpy(float)
            / joined["coordinate_threshold_a"].to_numpy(float),
        }
    ).max(axis=1)
    joined["distortion_ratio"] = ratios.to_numpy(float)
    joined["protected"] = joined["distortion_ratio"].le(1.0)
    joined["middle"] = joined["distortion_ratio"].gt(1.0) & joined[
        "distortion_ratio"
    ].lt(2.0)
    joined["severe"] = joined["distortion_ratio"].ge(2.0)
    joined = joined.sort_values("material_id", kind="stable", ignore_index=True)

    source_code = Path(__file__).resolve()
    source_hash = _sha256_file(source_code)
    staging: dict[str, Path] = {}
    manifests: dict[str, dict[str, object]] = {}
    try:
        for role in PARTITIONS:
            target = targets[role]
            target.parent.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
            staging[role] = stage
            table = joined[joined["partition_role"].eq(role)].copy()
            if table.empty:
                raise ValueError(f"SCIGEN endpoint split {role} is empty")
            endpoint_path = stage / ENDPOINT_NAME
            table.to_parquet(endpoint_path, index=False)
            manifest: dict[str, object] = {
                "protocol": PROTOCOL,
                "mode": "mechanical_physical_partition_endpoint_routing",
                "partition_role": role,
                "lockbox_state": (
                    "discovery_available_after_routing"
                    if role == "discovery"
                    else "unopened_for_model_development"
                ),
                "rows": len(table),
                "material_id_list_sha256": _identity_hash(
                    table["material_id"].astype(str).tolist()
                ),
                "endpoint_definition": {
                    "distortion_ratio": "max(F_max/0.01, d_latt/T_latt, d_xyz/T_xyz)",
                    "protected_max": 1.0,
                    "severe_min": 2.0,
                    "force_threshold_ev_per_a": FORCE_THRESHOLD,
                    "lattice_thresholds_a": {
                        key: value[0] for key, value in LATTICE_THRESHOLDS.items()
                    },
                    "coordinate_thresholds_a": {
                        key: value[1] for key, value in LATTICE_THRESHOLDS.items()
                    },
                },
                "source_endpoint_member_sha256": payload_sha256,
                "endpoint_values_mechanically_parsed_and_routed": True,
                "endpoint_values_summarized_or_inspected": False,
                "energy_columns_retained": False,
                "relaxed_structures_opened": False,
                "supplementary_tables_opened": False,
                "inputs_sha256": {
                    name: {"path": str(path), "sha256": hashes[name]}
                    for name, path in paths.items()
                },
                "executed_source_sha256": {
                    "src/next86_scigen_endpoint_router.py": source_hash
                },
                "outputs_sha256": {ENDPOINT_NAME: _sha256_file(endpoint_path)},
                "scientific_improvement_claim": False,
            }
            (stage / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
            manifests[role] = manifest
        if any(_sha256_file(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT86 endpoint router input changed before publication")
        if _sha256_file(source_code) != source_hash:
            raise RuntimeError("NEXT86 endpoint router source changed before publication")
        for role in PARTITIONS:
            _publish_directory_no_replace(staging[role], targets[role])
    except Exception:
        for stage in staging.values():
            shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifests


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--term-catalogue-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--discovery-output-dir", type=Path, required=True)
    parser.add_argument("--validation-output-dir", type=Path, required=True)
    parser.add_argument("--replication-output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    route_scigen_endpoints(
        source_archive_path=args.source_archive,
        cohort_dir=args.cohort_dir,
        term_catalogue_dir=args.term_catalogue_dir,
        design_path=args.design,
        output_dirs={
            "discovery": args.discovery_output_dir,
            "internal_validation": args.validation_output_dir,
            "internal_replication": args.replication_output_dir,
        },
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ENDPOINT_NAME",
    "MANIFEST_NAME",
    "PROTOCOL",
    "route_scigen_endpoints",
]
