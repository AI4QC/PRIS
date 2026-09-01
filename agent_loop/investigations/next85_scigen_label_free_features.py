"""Compute frozen analytic and Pauling features from SCIGEN x0 geometry only."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping, Sequence
import zipfile

from ase import Atoms
import numpy as np
import pandas as pd

from src.next11_geometry_only_frames import _parse_frame
from src.next12_dft_queue import _json_bytes
from src.next12_pauling_controls import (
    DECISIONS,
    RULES,
    _classical_features,
    _combined_decision,
    _rule_decision,
)
from src.next13d_acsc_dft_pairs import _sha256_file
from src.next14_wbm_holdout import _publish_directory_no_replace
from src.next43_analytic_feature_bank import (
    CANDIDATE_FEATURE_NAMES as NEXT43_FEATURE_NAMES,
    compute_analytic_feature_row,
)
from src.next44_rich_analytic_features import (
    CANDIDATE_FEATURE_NAMES as NEXT44_FEATURE_NAMES,
    compute_rich_feature_row,
)
from src.next80_periodic_repulsive_load_resolvability import (
    PRLR_FEATURE_NAMES,
    compute_periodic_repulsive_load_resolvability,
)
from src.next84_scigen_geometry_lockbox import (
    GEOMETRY_NAMES,
    MANIFEST_NAME as COHORT_MANIFEST_NAME,
    METADATA_NAME as COHORT_METADATA_NAME,
    PARTITIONS,
    PROTOCOL as COHORT_PROTOCOL,
)


PROTOCOL = "2026-08-03-next85-scigen-frozen-label-free-analytic-features-v1"
FEATURE_NAMES = {
    "discovery": "features_discovery.parquet",
    "internal_validation": "features_internal_validation.parquet",
    "internal_replication": "features_internal_replication.parquet",
}
CATALOGUE_NAME = "FEATURE_CATALOGUE.json"
MANIFEST_NAME = "MANIFEST.json"
EXPECTED_INPUT_SHA256 = {
    "cohort_manifest": "dc5bf33c6ce6dc2c10bcd3704688055058145fbe7269ada23ffbe4b141d75fe7",
    "metadata": "f91455f23b0a96f60fd1c779249e2be46a7ecf94fcdde2b146426a95aac05bde",
    "geometry_discovery": "e561ef12343c66dcc72bcabf6b8719ad727e01c9582a094e281da73b862ab575",
    "geometry_internal_validation": "d79d6c5466a4dcb06fb22df3c2aa118687fd5f1434efc7aede25b4a6444ea278",
    "geometry_internal_replication": "7c335c6a49fa9a1674c893ceae70654e7365a51fadf6124d60a10d2f86f5c087",
    "design": "39127f4d2b5ddba176f7904ed498f98e0326fa902e1c3ede79fbbcf320c13ee9",
}
UPSTREAM_SOURCE_HASHES = {
    "src/next43_analytic_feature_bank.py": "9212af77b86491ae71214f810bb316612d3037f87f645d57b712faebec0c4d24",
    "src/next44_rich_analytic_features.py": "fc8d2c99ecd183ef132e8a9d6af3bac71022367219ec2d014ac10fc210e9d373",
    "src/next80_periodic_repulsive_load_resolvability.py": "9bf93d4a541e900605b35d872e6eca9d8ee0c8b43c5f84395be956f5338c0655",
    "src/next12_pauling_controls.py": "b37f1a84326e8104b38bd61398ee10cf9f6421007fead5d004a0601cb5159c43",
}
ALL_ANALYTIC_FEATURES = tuple(
    dict.fromkeys((*NEXT43_FEATURE_NAMES, *NEXT44_FEATURE_NAMES, *PRLR_FEATURE_NAMES))
)
ALWAYS_PRESENT_STATUS_COLUMNS = (
    "next43_error",
    "next44_error",
    "next80_error",
    "next80_supported",
    "next80_failure",
    "pauling_feature_error",
    "pauling_p2_p5_decision",
    *(f"pauling_{name}_value" for name in RULES),
    *(f"pauling_{name}_decision" for name in RULES),
)
IDENTITY_ONLY_COLUMNS = (
    "material_id",
    "lattice_class",
    "reduced_formula",
    "chemical_system",
    "partition_role",
)


def _finite_or_nan(value: object) -> object:
    if isinstance(value, (bool, str)) or value is None:
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return number if math.isfinite(number) else math.nan


def compute_scigen_feature_row(atoms: Atoms) -> dict[str, object]:
    """Compute every frozen x0-only family with independent fail-open errors."""

    row: dict[str, object] = {name: math.nan for name in ALL_ANALYTIC_FEATURES}
    row["next43_error"] = None
    row["next44_error"] = None
    row["next80_error"] = None
    row["pauling_feature_error"] = None

    try:
        values43 = compute_analytic_feature_row(atoms)
    except Exception as exc:
        row["next43_error"] = f"{type(exc).__name__}: {exc}"
    else:
        for name, value in values43.items():
            target_name = name if name in NEXT43_FEATURE_NAMES else f"next43_{name}"
            row[target_name] = _finite_or_nan(value)

    try:
        values44 = compute_rich_feature_row(atoms)
    except Exception as exc:
        row["next44_error"] = f"{type(exc).__name__}: {exc}"
    else:
        for name, value in values44.items():
            target_name = name if name in NEXT44_FEATURE_NAMES else f"next44_{name}"
            row[target_name] = _finite_or_nan(value)

    try:
        prlr = compute_periodic_repulsive_load_resolvability(atoms)
    except Exception as exc:
        row["next80_error"] = f"{type(exc).__name__}: {exc}"
    else:
        for name in PRLR_FEATURE_NAMES:
            value = prlr.features.get(name, math.nan)
            row[name] = _finite_or_nan(value)
        row["next80_supported"] = bool(prlr.supported)
        row["next80_failure"] = prlr.failure_reason

    try:
        pauling_values, pauling_error = _classical_features(atoms)
    except Exception as exc:
        pauling_values = None
        pauling_error = f"{type(exc).__name__}: {exc}"
    row["pauling_feature_error"] = pauling_error
    decisions: list[str] = []
    values = dict(pauling_values) if isinstance(pauling_values, Mapping) else {}
    for name, rule in RULES.items():
        value = values.get(str(rule["feature"]), np.nan)
        decision = _rule_decision(
            value,
            operator=str(rule["operator"]),
            threshold=float(rule["threshold"]),
        )
        row[f"pauling_{name}_value"] = _finite_or_nan(value)
        row[f"pauling_{name}_decision"] = decision
        decisions.append(decision)
    row["pauling_p2_p5_decision"] = _combined_decision(decisions)
    return row


def _compute_payload(item: tuple[str, bytes]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    parsed = _parse_frame(payload, strict_output=True)
    return material_id, compute_scigen_feature_row(parsed.atoms)


def _archive_payloads(path: Path, expected_ids: Sequence[str]) -> list[tuple[str, bytes]]:
    expected = tuple(sorted(str(value) for value in expected_ids))
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names != sorted(names) or any(not name.endswith(".extxyz") for name in names):
            raise ValueError("NEXT84 geometry archive inventory differs")
        ids = tuple(Path(name).stem for name in names)
        if ids != expected:
            raise ValueError("NEXT84 geometry archive identity differs")
        return [(material_id, archive.read(name)) for material_id, name in zip(ids, names, strict=True)]


def _compute_many(
    payloads: Sequence[tuple[str, bytes]], *, workers: int
) -> list[tuple[str, dict[str, object]]]:
    if workers == 1:
        return [_compute_payload(item) for item in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_compute_payload, payloads, chunksize=8))


def _read_manifest(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, Mapping):
        raise ValueError("NEXT84 manifest must be a JSON object")
    return value


def build_scigen_label_free_features(
    *,
    cohort_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Freeze all analytic features for all partitions before endpoint opening."""

    cohort = Path(cohort_dir).resolve()
    design = Path(design_path).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive exact integer")
    paths = {
        "cohort_manifest": cohort / COHORT_MANIFEST_NAME,
        "metadata": cohort / COHORT_METADATA_NAME,
        **{
            f"geometry_{role}": cohort / GEOMETRY_NAMES[role]
            for role in PARTITIONS
        },
        "design": design,
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT85 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT85 formal input identity differs")

    cohort_manifest = _read_manifest(paths["cohort_manifest"])
    outputs = cohort_manifest.get("outputs_sha256")
    if (
        cohort_manifest.get("protocol") != COHORT_PROTOCOL
        or cohort_manifest.get("labels_opened") is not False
        or cohort_manifest.get("endpoint_payloads_opened") is not False
        or cohort_manifest.get("relaxed_structures_opened") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get(COHORT_METADATA_NAME) != input_hashes["metadata"]
        or any(outputs.get(GEOMETRY_NAMES[role]) != input_hashes[f"geometry_{role}"] for role in PARTITIONS)
    ):
        raise ValueError("NEXT84 geometry-lockbox provenance differs")
    metadata = pd.read_parquet(paths["metadata"])
    required = {
        "material_id",
        "lattice_class",
        "reduced_formula",
        "chemical_system",
        "natoms",
        "partition_role",
        "input_role",
    }
    if required - set(metadata.columns):
        raise ValueError("NEXT84 metadata columns differ")
    if (
        metadata["material_id"].duplicated().any()
        or set(metadata["partition_role"]) - set(PARTITIONS)
        or not metadata["input_role"].eq("raw_generated_pre_dft_unrelaxed_x0").all()
    ):
        raise ValueError("NEXT84 metadata identity or input role differs")

    repository_root = Path(__file__).resolve().parents[1]
    upstream_hashes = {
        relative: _sha256_file(repository_root / relative)
        for relative in UPSTREAM_SOURCE_HASHES
    }
    if require_formal_inputs and upstream_hashes != UPSTREAM_SOURCE_HASHES:
        raise ValueError("NEXT85 frozen upstream feature source differs")
    source_path = Path(__file__).resolve()
    source_hash = _sha256_file(source_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    started = time.perf_counter()
    counts: dict[str, object] = {}
    output_paths: list[Path] = []
    try:
        for role in PARTITIONS:
            part_meta = metadata[metadata["partition_role"].eq(role)].copy()
            part_meta = part_meta.sort_values("material_id", kind="stable", ignore_index=True)
            payloads = _archive_payloads(
                paths[f"geometry_{role}"], part_meta["material_id"].astype(str).tolist()
            )
            computed = _compute_many(payloads, workers=workers)
            computed_table = pd.DataFrame(
                [{"material_id": material_id, **row} for material_id, row in computed]
            )
            if computed_table.empty:
                computed_table = pd.DataFrame(columns=["material_id", *ALL_ANALYTIC_FEATURES])
            table = part_meta.merge(
                computed_table, on="material_id", how="left", validate="one_to_one"
            )
            if len(table) != len(part_meta):
                raise RuntimeError(f"NEXT85 {role} row accounting differs")
            for name in ALWAYS_PRESENT_STATUS_COLUMNS:
                if name not in table:
                    table[name] = None
            feature_path = staging / FEATURE_NAMES[role]
            table.to_parquet(feature_path, index=False)
            output_paths.append(feature_path)
            finite_counts = {
                name: int(np.isfinite(pd.to_numeric(table[name], errors="coerce")).sum())
                for name in ALL_ANALYTIC_FEATURES
            }
            counts[role] = {
                "rows": len(table),
                "full_row_errors": int(
                    table[["next43_error", "next44_error", "next80_error"]]
                    .notna()
                    .any(axis=1)
                    .sum()
                ),
                "pauling_feature_errors": int(table["pauling_feature_error"].notna().sum()),
                "pauling_decisions": {
                    decision: int(table["pauling_p2_p5_decision"].eq(decision).sum())
                    for decision in DECISIONS
                },
                "finite_feature_counts": finite_counts,
            }

        catalogue = {
            "protocol": PROTOCOL,
            "feature_names": list(ALL_ANALYTIC_FEATURES),
            "feature_count": len(ALL_ANALYTIC_FEATURES),
            "families": {
                "NEXT43": list(NEXT43_FEATURE_NAMES),
                "NEXT44": list(NEXT44_FEATURE_NAMES),
                "NEXT80": list(PRLR_FEATURE_NAMES),
                "Pauling_controls": list(RULES),
            },
            "identity_only_columns_excluded_from_candidate_laws": list(IDENTITY_ONLY_COLUMNS),
            "endpoint_columns_present": False,
            "labels_opened": False,
        }
        catalogue_path = staging / CATALOGUE_NAME
        catalogue_path.write_bytes(_json_bytes(catalogue))
        output_paths.append(catalogue_path)
        elapsed = time.perf_counter() - started
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "mode": "all_partition_label_free_analytic_and_pauling_feature_freeze",
            "workers": workers,
            "elapsed_seconds": elapsed,
            "counts": counts,
            "labels_opened": False,
            "endpoint_payloads_opened": False,
            "relaxed_structures_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_features": False,
            "learned_energy_force_stress_proxy_used": False,
            "model_or_proxy_potential_used": False,
            "physical_relaxation_executed": False,
            "inputs_sha256": {
                name: {"path": str(path), "sha256": input_hashes[name]}
                for name, path in paths.items()
            },
            "upstream_source_sha256": upstream_hashes,
            "executed_source_sha256": {
                "src/next85_scigen_label_free_features.py": source_hash
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT85 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT85 source changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser


def main() -> None:
    args = _parser().parse_args()
    build_scigen_label_free_features(
        cohort_dir=args.cohort_dir,
        design_path=args.design,
        output_dir=args.output_dir,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "FEATURE_NAMES",
    "MANIFEST_NAME",
    "PROTOCOL",
    "build_scigen_label_free_features",
    "compute_scigen_feature_row",
]
