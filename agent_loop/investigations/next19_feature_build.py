"""Seal pure-analytic NEXT19 descriptors from sanitized x0 geometry archives."""

from __future__ import annotations

import argparse
from collections import Counter
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor

from src.next11_geometry_only_frames import (
    _atomic_publish_directory_no_replace,
    _load_archive_only,
)
from src.next19_valence_transport import (
    FEATURE_NAMES,
    PROTOCOL as FEATURE_PROTOCOL,
    build_periodic_edge_geometry,
    edge_priors_from_periodic_geometry,
    infer_valence_assignment,
    solve_valence_transport,
)


PROTOCOL = "2026-08-02-next19-geometry-only-feature-build-v2"
FEATURE_NAME = "next19_valence_transport_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
GRAPH_MODES = ("crystalnn", "voronoi")
ALPHAS = (0.0, 2.0, 4.0, 6.0)
FORBIDDEN_METADATA_TOKENS = (
    "energy",
    "force",
    "stress",
    "relax",
    "mattersim",
    "dft",
    "endpoint",
    "label",
    "target",
)
REQUIRED_METADATA = ("material_id", "rk", "formula", "natoms", "input_role")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path, *, role: str) -> dict[str, object]:
    candidate = Path(path)
    if not candidate.is_file():
        raise ValueError(f"{role} is not a file")
    try:
        value = json.loads(candidate.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value


def validate_geometry_metadata(table: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonically sort identifier-only geometry metadata."""

    missing = set(REQUIRED_METADATA) - set(table)
    if missing:
        raise ValueError(f"geometry metadata missing columns: {sorted(missing)}")
    for column in table.columns:
        lowered = str(column).lower()
        if any(token in lowered for token in FORBIDDEN_METADATA_TOKENS):
            raise ValueError(f"forbidden metadata column: {column}")
    result = table.copy()
    for column in ("material_id", "rk", "formula", "input_role"):
        if result[column].isna().any():
            raise ValueError(f"geometry metadata {column} contains nulls")
        result[column] = result[column].astype(str)
        if result[column].eq("").any():
            raise ValueError(f"geometry metadata {column} contains empty strings")
    if result["material_id"].duplicated().any():
        raise ValueError("geometry metadata material_id is not unique")
    if not result["input_role"].eq("unrelaxed_x0_geometry_only").all():
        raise ValueError("geometry metadata input role mismatch")
    numeric_natoms = pd.to_numeric(result["natoms"], errors="raise")
    if not np.equal(numeric_natoms, np.floor(numeric_natoms)).all() or np.any(
        numeric_natoms <= 0
    ):
        raise ValueError("geometry metadata natoms must be positive integers")
    result["natoms"] = numeric_natoms.astype(int)
    return result.sort_values("material_id", kind="stable").reset_index(drop=True)


def _validate_source_manifest(
    manifest: Mapping[str, object], *, archive_path: Path
) -> None:
    direct_x0_contract = manifest.get("input_role") == "unrelaxed_x0_geometry_only"
    legacy_wbm_x0_contract = (
        manifest.get("protocol")
        == "2026-08-02-next14-wbm-acsc-label-free-holdout-v1"
        and manifest.get("mode") == "external_source_label_free_small_cell_holdout"
        and manifest.get("endpoint_artifacts_opened") is False
        and manifest.get("labels_opened") is False
        and manifest.get("relaxed_structures_opened") is False
        and manifest.get("production_protocol_eligible") is True
    )
    if not (direct_x0_contract or legacy_wbm_x0_contract):
        raise ValueError("source manifest input role mismatch")
    if manifest.get("scientific_improvement_claim") is not False:
        raise ValueError("source manifest scientific claim flag mismatch")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping):
        raise ValueError("source manifest output hashes missing")
    expected = outputs.get("geometry_only_frames.zip")
    if not isinstance(expected, str) or expected != _sha256(archive_path):
        raise ValueError("source archive hash mismatch")
    false_flags = (
        "endpoint_fields_accessed",
        "endpoint_fields_accessed_by_sanitizer",
        "endpoint_bytes_read_by_execution",
        "downstream_geometry_artifacts_include_endpoint_fields",
    )
    for key in false_flags:
        if key in manifest and manifest.get(key) is not False:
            raise ValueError(f"source manifest {key} must be false")


def _alpha_tag(alpha: float) -> str:
    value = float(alpha)
    if value.is_integer():
        return str(int(value))
    return format(value, ".8g").replace(".", "p")


def _configuration_prefix(mode: str, alpha: float) -> str:
    return f"{mode}_a{_alpha_tag(alpha)}"


def _validate_catalogue(
    graph_modes: Sequence[str], alphas: Sequence[float]
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    modes = tuple(str(value) for value in graph_modes)
    alpha_values = tuple(float(value) for value in alphas)
    if not modes or len(set(modes)) != len(modes) or not set(modes).issubset(GRAPH_MODES):
        raise ValueError("graph modes are not a unique subset of the frozen catalogue")
    if (
        not alpha_values
        or len(set(alpha_values)) != len(alpha_values)
        or not set(alpha_values).issubset(ALPHAS)
        or not all(math.isfinite(value) for value in alpha_values)
    ):
        raise ValueError("alphas are not a unique subset of the frozen catalogue")
    return modes, alpha_values


def _configuration_result(geometry, *, alpha: float) -> tuple[bool, str | None, dict[str, float]]:
    priors = edge_priors_from_periodic_geometry(geometry, alpha=alpha)
    if not priors.supported:
        return False, priors.failure_reason, {}
    solution = solve_valence_transport(
        cation_supply=priors.cation_supply,
        anion_demand=priors.anion_demand,
        edges=priors.edges,
    )
    if not solution.supported:
        return False, solution.failure_reason, {}
    return (
        True,
        None,
        {
            "vt_overload": float(solution.overload),
            "vt_reallocation": float(solution.reallocation),
            "vt_anion_mismatch_max": float(solution.max_anion_mismatch),
            "vt_periodic_edge_count": float(len(priors.edges)),
            "vt_cation_count": float(len(priors.cation_supply)),
            "vt_anion_count": float(len(priors.anion_demand)),
        },
    )


def _publish_directory_no_replace(source: Path, target: Path) -> None:
    """Atomically publish, with a cooperative lock fallback for older filesystems."""

    try:
        _atomic_publish_directory_no_replace(source, target)
        return
    except OSError as exc:
        unsupported = {
            errno.EINVAL,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }
        if exc.errno not in unsupported:
            raise
    lock = target.parent / f".{target.name}.publish.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"publication lock already exists: {lock}") from exc
    try:
        if target.exists():
            raise FileExistsError(str(target))
        os.rename(source, target)
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)


def build_feature_batch(
    *,
    archive_path: Path,
    source_manifest_path: Path,
    metadata_path: Path,
    source_role: str,
    output_dir: Path,
    graph_modes: Sequence[str] = GRAPH_MODES,
    alphas: Sequence[float] = ALPHAS,
) -> None:
    """Build one deterministic, no-replace NEXT19 feature directory."""

    modes, alpha_values = _validate_catalogue(graph_modes, alphas)
    if not isinstance(source_role, str) or not source_role.strip():
        raise ValueError("source role must be a nonempty string")
    archive = Path(archive_path)
    source_manifest_candidate = Path(source_manifest_path)
    metadata_candidate = Path(metadata_path)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(str(target))
    if archive.name != "geometry_only_frames.zip" or not archive.is_file():
        raise ValueError("geometry-only archive path/name is invalid")
    source_manifest = _strict_json(source_manifest_candidate, role="source manifest")
    _validate_source_manifest(source_manifest, archive_path=archive)
    metadata = validate_geometry_metadata(pd.read_parquet(metadata_candidate))
    expected_sids = tuple(metadata["material_id"].tolist())
    sids, atoms_list = _load_archive_only(archive, expected_sids)
    if sids != list(expected_sids):
        raise ValueError("geometry archive SID order mismatch")
    if [len(atoms) for atoms in atoms_list] != metadata["natoms"].tolist():
        raise ValueError("geometry archive atom counts mismatch metadata")

    rows: list[dict[str, object]] = []
    failures: dict[str, Counter[str]] = {
        _configuration_prefix(mode, alpha): Counter()
        for mode in modes
        for alpha in alpha_values
    }
    for metadata_row, atoms in zip(
        metadata.to_dict(orient="records"), atoms_list, strict=True
    ):
        row: dict[str, object] = {
            "material_id": metadata_row["material_id"],
            "rk": metadata_row["rk"],
            "formula": metadata_row["formula"],
            "natoms": int(metadata_row["natoms"]),
            "input_role": metadata_row["input_role"],
        }
        structure = AseAtomsAdaptor.get_structure(atoms)
        assignment = infer_valence_assignment(structure)
        valences = assignment.values
        valence_error = assignment.failure_reason
        row["valence_supported"] = assignment.supported
        row["valence_policy"] = assignment.policy
        row["valence_failure"] = valence_error
        for mode in modes:
            geometry = (
                build_periodic_edge_geometry(structure, valences, graph_mode=mode)
                if valences is not None
                else None
            )
            for alpha in alpha_values:
                prefix = _configuration_prefix(mode, alpha)
                if geometry is None:
                    supported, error, values = False, valence_error, {}
                elif not geometry.supported:
                    supported, error, values = False, geometry.failure_reason, {}
                else:
                    supported, error, values = _configuration_result(
                        geometry, alpha=alpha
                    )
                row[f"{prefix}__supported"] = bool(supported)
                row[f"{prefix}__failure"] = error
                if not supported:
                    failures[prefix][error or "unknown"] += 1
                for feature in FEATURE_NAMES:
                    row[f"{prefix}__{feature}"] = (
                        float(values[feature]) if supported else math.nan
                    )
        rows.append(row)
    features = pd.DataFrame(rows)
    forbidden_columns = [
        column
        for column in features
        if any(token in column.lower() for token in FORBIDDEN_METADATA_TOKENS)
    ]
    if forbidden_columns:
        raise ValueError(f"feature output crossed no-DFT contract: {forbidden_columns}")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        feature_path = staging / FEATURE_NAME
        features.to_parquet(feature_path, index=False)
        supported_counts = {
            prefix: int(features[f"{prefix}__supported"].sum())
            for prefix in failures
        }
        manifest = {
            "protocol": PROTOCOL,
            "feature_protocol": FEATURE_PROTOCOL,
            "source_role": source_role,
            "input_role": "unrelaxed_x0_geometry_only",
            "endpoint_fields_read": False,
            "model_or_proxy_potential_used": False,
            "coordinates_or_cell_modified": False,
            "scientific_improvement_claim": False,
            "catalogue": {
                "graph_modes": list(modes),
                "alphas": list(alpha_values),
                "feature_names": list(FEATURE_NAMES),
            },
            "counts": {
                "rows": int(len(features)),
                "atoms": int(features["natoms"].sum()),
                "valence_supported": int(features["valence_supported"].sum()),
                "valence_policy": {
                    str(key): int(value)
                    for key, value in features["valence_policy"]
                    .fillna("unsupported")
                    .value_counts()
                    .sort_index()
                    .items()
                },
                "configuration_supported": supported_counts,
            },
            "failure_counts": {
                prefix: dict(sorted(counter.items()))
                for prefix, counter in failures.items()
            },
            "inputs_sha256": {
                "geometry_only_frames.zip": _sha256(archive),
                "source_manifest": _sha256(source_manifest_candidate),
                "metadata": _sha256(metadata_candidate),
            },
            "executed_source_sha256": {
                "src/next19_valence_transport.py": _sha256(
                    Path(__file__).with_name("next19_valence_transport.py")
                ),
                "src/next19_feature_build.py": _sha256(Path(__file__)),
            },
            "outputs_sha256": {FEATURE_NAME: _sha256(feature_path)},
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        reloaded = pd.read_parquet(feature_path)
        if not reloaded.equals(features) or _sha256(feature_path) != manifest[
            "outputs_sha256"
        ][FEATURE_NAME]:
            raise ValueError("feature batch prepublication validation failed")
        _publish_directory_no_replace(staging, target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--source-role", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    build_feature_batch(
        archive_path=args.archive,
        source_manifest_path=args.source_manifest,
        metadata_path=args.metadata,
        source_role=args.source_role,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
