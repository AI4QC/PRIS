#!/usr/bin/env python3
"""Frozen elemental-table and metal-donor descriptors from one raw MOF x0."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from ase import Atoms
import numpy as np
import pandas as pd
from pymatgen.core import Element

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next26_packing import _radii
from src.next49_framework_topology import (
    _canonical_covalent_edges,
    _directed_adjacency,
    _environment_versions,
    _is_metal,
    _strict_geometry,
)
from src.next52_site_resolved_motif_features import DONOR_NUMBERS
from src.next54_odac23_train_selection import (
    GEOMETRY_NAME as SOURCE_GEOMETRY_NAME,
    MANIFEST_NAME as SELECTION_MANIFEST_NAME,
    PROTOCOL as SELECTION_PROTOCOL,
)
from src.next55_odac23_analytic_features import _load_archive
from src.next58_odac23_shared_motif_features import (
    FEATURES_NAME as SOURCE_FEATURES_NAME,
    MANIFEST_NAME as SOURCE_MANIFEST_NAME,
    NEXT58_FEATURE_NAMES,
    PROTOCOL as SOURCE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next63-odac23-metal-chemistry-features-v1"
DESIGN_SHA256 = "2da0004b9f2c5c828ba2c4e1aafef0ac422b40e72e7edd9cbb2c25195a3b60e1"
EXPECTED_SELECTION_MANIFEST_SHA256 = (
    "9ea1f0e6c04c8619dd295aa1579da15b51d8241971b3adacb716fdbf93290927"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "4d0c2b667ea67233444d41b4f2c8035ab5eb047fad93342e9efb568c04ec5946"
)
FEATURES_NAME = "next63_odac23_metal_chemistry_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
HALOGEN_NUMBERS = frozenset((9, 17, 35, 53))
METAL_CHEMISTRY_FEATURE_NAMES = (
    "oxygen_fraction",
    "nitrogen_fraction",
    "sulfur_fraction",
    "phosphorus_fraction",
    "halogen_fraction",
    "metal_species_count",
    "metal_composition_entropy",
    "metal_atomic_number_mean",
    "metal_atomic_number_std",
    "metal_atomic_number_min",
    "metal_atomic_number_max",
    "metal_group_mean",
    "metal_group_std",
    "metal_group_min",
    "metal_group_max",
    "metal_row_mean",
    "metal_row_std",
    "metal_electronegativity_mean",
    "metal_electronegativity_std",
    "metal_electronegativity_min",
    "metal_electronegativity_max",
    "metal_mendeleev_mean",
    "metal_mendeleev_std",
    "metal_ionization_energy_mean",
    "metal_ionization_energy_std",
    "metal_ionization_energy_min",
    "metal_ionization_energy_max",
    "metal_atomic_radius_mean",
    "metal_atomic_radius_std",
    "metal_common_oxidation_mean",
    "metal_common_oxidation_std",
    "metal_common_oxidation_min",
    "metal_common_oxidation_max",
    "metal_max_oxidation_mean",
    "metal_max_oxidation_std",
    "metal_charge_density_proxy_mean",
    "metal_charge_density_proxy_max",
    "metal_hardness_proxy_mean",
    "metal_coordination_per_oxidation_mean",
    "metal_coordination_per_oxidation_std",
    "metal_coordination_per_oxidation_q95",
    "metal_donor_en_gap_mean",
    "metal_donor_en_gap_q95",
    "metal_donor_distance_mean",
    "metal_donor_distance_q95",
    "metal_donor_ratio_mean",
    "metal_donor_ratio_std",
    "metal_donor_ratio_max",
)
NEXT63_FEATURE_NAMES = tuple(NEXT58_FEATURE_NAMES) + METAL_CHEMISTRY_FEATURE_NAMES


@dataclass(frozen=True)
class MetalChemistryResult:
    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def _stats(values: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
    )


def _positive_common_oxidation(element: Element) -> float:
    values = [float(value) for value in element.common_oxidation_states if value > 0]
    if not values and float(element.max_oxidation_state) > 0:
        values = [float(element.max_oxidation_state)]
    if not values:
        raise ValueError(f"metal {element.symbol} has no positive oxidation proxy")
    return float(np.mean(values))


def compute_metal_chemistry_features(atoms: Atoms) -> MetalChemistryResult:
    try:
        numbers, _positions, _cell = _strict_geometry(atoms)
        covalent, _vdw = _radii(numbers)
        covalent = np.asarray(covalent, dtype=float)
        edges = _canonical_covalent_edges(atoms, covalent)
        adjacency = _directed_adjacency(len(atoms), edges)
        metal_mask = np.asarray([_is_metal(number) for number in numbers], dtype=bool)
        metal_indices = np.flatnonzero(metal_mask)
        if not len(metal_indices):
            raise ValueError("framework has no recognized metal")
        elements = [Element.from_Z(int(numbers[index])) for index in metal_indices]
        atomic_numbers = numbers[metal_indices].astype(float)
        groups = np.asarray([float(element.group) for element in elements])
        rows = np.asarray([float(element.row) for element in elements])
        electronegativities = np.asarray([float(element.X) for element in elements])
        mendeleev = np.asarray([float(element.mendeleev_no) for element in elements])
        ionization = np.asarray([float(element.ionization_energy) for element in elements])
        radii = np.asarray([float(element.atomic_radius) for element in elements])
        common_oxidation = np.asarray(
            [_positive_common_oxidation(element) for element in elements], dtype=float
        )
        max_oxidation = np.asarray(
            [float(element.max_oxidation_state) for element in elements], dtype=float
        )
        if not np.isfinite(
            np.concatenate(
                [groups, rows, electronegativities, mendeleev, ionization, radii, common_oxidation, max_oxidation]
            )
        ).all() or np.any(radii <= 0.0) or np.any(common_oxidation <= 0.0):
            raise ValueError("frozen metal elemental property is unavailable")
        unique, counts = np.unique(numbers[metal_indices], return_counts=True)
        probabilities = counts / counts.sum()
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        coordination = np.asarray([len(adjacency[index]) for index in metal_indices], dtype=float)
        coord_per_oxidation = coordination / common_oxidation

        all_en = np.asarray(
            [float(Element.from_Z(int(number)).X or 0.0) for number in numbers], dtype=float
        )
        donor_gaps = []
        donor_distances = []
        donor_ratios = []
        for edge in edges:
            left_metal = bool(metal_mask[edge.first])
            right_metal = bool(metal_mask[edge.second])
            if left_metal == right_metal:
                continue
            metal_index = edge.first if left_metal else edge.second
            donor_index = edge.second if left_metal else edge.first
            if int(numbers[donor_index]) not in DONOR_NUMBERS:
                continue
            donor_gaps.append(abs(all_en[donor_index] - all_en[metal_index]))
            donor_distances.append(float(np.linalg.norm(edge.vector)))
            donor_ratios.append(float(edge.ratio))
        gaps = np.asarray(donor_gaps if donor_gaps else [0.0], dtype=float)
        distances = np.asarray(donor_distances if donor_distances else [0.0], dtype=float)
        ratios = np.asarray(donor_ratios if donor_ratios else [0.0], dtype=float)

        z_mean, z_std, z_min, z_max = _stats(atomic_numbers)
        group_mean, group_std, group_min, group_max = _stats(groups)
        en_mean, en_std, en_min, en_max = _stats(electronegativities)
        ion_mean, ion_std, ion_min, ion_max = _stats(ionization)
        ox_mean, ox_std, ox_min, ox_max = _stats(common_oxidation)
        values = {
            "oxygen_fraction": float(np.mean(numbers == 8)),
            "nitrogen_fraction": float(np.mean(numbers == 7)),
            "sulfur_fraction": float(np.mean(numbers == 16)),
            "phosphorus_fraction": float(np.mean(numbers == 15)),
            "halogen_fraction": float(np.mean(np.isin(numbers, tuple(HALOGEN_NUMBERS)))),
            "metal_species_count": float(len(unique)),
            "metal_composition_entropy": entropy,
            "metal_atomic_number_mean": z_mean,
            "metal_atomic_number_std": z_std,
            "metal_atomic_number_min": z_min,
            "metal_atomic_number_max": z_max,
            "metal_group_mean": group_mean,
            "metal_group_std": group_std,
            "metal_group_min": group_min,
            "metal_group_max": group_max,
            "metal_row_mean": float(np.mean(rows)),
            "metal_row_std": float(np.std(rows)),
            "metal_electronegativity_mean": en_mean,
            "metal_electronegativity_std": en_std,
            "metal_electronegativity_min": en_min,
            "metal_electronegativity_max": en_max,
            "metal_mendeleev_mean": float(np.mean(mendeleev)),
            "metal_mendeleev_std": float(np.std(mendeleev)),
            "metal_ionization_energy_mean": ion_mean,
            "metal_ionization_energy_std": ion_std,
            "metal_ionization_energy_min": ion_min,
            "metal_ionization_energy_max": ion_max,
            "metal_atomic_radius_mean": float(np.mean(radii)),
            "metal_atomic_radius_std": float(np.std(radii)),
            "metal_common_oxidation_mean": ox_mean,
            "metal_common_oxidation_std": ox_std,
            "metal_common_oxidation_min": ox_min,
            "metal_common_oxidation_max": ox_max,
            "metal_max_oxidation_mean": float(np.mean(max_oxidation)),
            "metal_max_oxidation_std": float(np.std(max_oxidation)),
            "metal_charge_density_proxy_mean": float(np.mean(common_oxidation / radii**3)),
            "metal_charge_density_proxy_max": float(np.max(common_oxidation / radii**3)),
            "metal_hardness_proxy_mean": float(np.mean(ionization / radii)),
            "metal_coordination_per_oxidation_mean": float(np.mean(coord_per_oxidation)),
            "metal_coordination_per_oxidation_std": float(np.std(coord_per_oxidation)),
            "metal_coordination_per_oxidation_q95": float(np.quantile(coord_per_oxidation, 0.95)),
            "metal_donor_en_gap_mean": float(np.mean(gaps)),
            "metal_donor_en_gap_q95": float(np.quantile(gaps, 0.95)),
            "metal_donor_distance_mean": float(np.mean(distances)),
            "metal_donor_distance_q95": float(np.quantile(distances, 0.95)),
            "metal_donor_ratio_mean": float(np.mean(ratios)),
            "metal_donor_ratio_std": float(np.std(ratios)),
            "metal_donor_ratio_max": float(np.max(ratios)),
        }
        if tuple(values) != METAL_CHEMISTRY_FEATURE_NAMES or not np.isfinite(
            list(values.values())
        ).all():
            raise ValueError("NEXT63 metal chemistry schema differs")
        return MetalChemistryResult(True, None, values)
    except Exception as exc:
        return MetalChemistryResult(False, f"{type(exc).__name__}: {exc}", {})


def _feature_record(atoms: Atoms) -> dict[str, object]:
    result = compute_metal_chemistry_features(atoms)
    row: dict[str, object] = {
        "metal_chemistry_supported": result.supported,
        "metal_chemistry_failure": result.failure_reason,
    }
    row.update(
        {
            name: float(result.features[name]) if result.supported else math.nan
            for name in METAL_CHEMISTRY_FEATURE_NAMES
        }
    )
    return row


def _strict_json(path: Path, role: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def build_metal_chemistry_batch(
    *, selection_dir: Path, source_dir: Path, design_path: Path, output_dir: Path, workers: int = 1
) -> dict[str, object]:
    selection_dir = Path(selection_dir).resolve()
    source_dir = Path(source_dir).resolve()
    target = Path(output_dir).resolve()
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("NEXT63 workers must be 1 through 64")
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "geometry": selection_dir / SOURCE_GEOMETRY_NAME,
        "selection_manifest": selection_dir / SELECTION_MANIFEST_NAME,
        "source_features": source_dir / SOURCE_FEATURES_NAME,
        "source_manifest": source_dir / SOURCE_MANIFEST_NAME,
        "design": Path(design_path).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT63 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if (
        hashes["selection_manifest"] != EXPECTED_SELECTION_MANIFEST_SHA256
        or hashes["source_manifest"] != EXPECTED_SOURCE_MANIFEST_SHA256
        or hashes["design"] != DESIGN_SHA256
    ):
        raise ValueError("NEXT63 frozen input hash differs")
    selection_manifest = _strict_json(paths["selection_manifest"], "NEXT54 manifest")
    source_manifest = _strict_json(paths["source_manifest"], "NEXT58 manifest")
    selection_outputs = selection_manifest.get("outputs_sha256")
    source_outputs = source_manifest.get("outputs_sha256")
    if (
        selection_manifest.get("protocol") != SELECTION_PROTOCOL
        or not isinstance(selection_outputs, Mapping)
        or selection_outputs.get(paths["geometry"].name) != hashes["geometry"]
        or source_manifest.get("protocol") != SOURCE_PROTOCOL
        or source_manifest.get("labels_opened") is not False
        or not isinstance(source_outputs, Mapping)
        or source_outputs.get(paths["source_features"].name) != hashes["source_features"]
    ):
        raise ValueError("NEXT63 x0-only provenance differs")
    source = pd.read_parquet(paths["source_features"])
    if source.empty or source["material_id"].duplicated().any():
        raise ValueError("NEXT63 source feature identity differs")
    material_ids = tuple(source["material_id"].astype(str))
    structures = _load_archive(paths["geometry"], material_ids)
    rows = []
    failures: Counter[str] = Counter()
    if workers == 1:
        iterator = map(_feature_record, structures)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_feature_record, structures, chunksize=4)
    try:
        for index, row in enumerate(iterator, start=1):
            rows.append(row)
            if not bool(row["metal_chemistry_supported"]):
                failures[str(row["metal_chemistry_failure"] or "unsupported")] += 1
            if index % 100 == 0 or index == len(structures):
                print(f"NEXT63 ODAC23 metal chemistry: {index}/{len(structures)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    table = pd.concat([source.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    table["next63_supported"] = (
        table["combined_supported"].astype(bool)
        & table["metal_chemistry_supported"].astype(bool)
        & np.isfinite(table.loc[:, NEXT63_FEATURE_NAMES]).all(axis=1)
    )
    source_path = Path(__file__).resolve()
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "selected_odac23_label_free_frozen_metal_chemistry",
        "input_role": "one_raw_unrelaxed_framework_x0_geometry_only",
        "labels_opened": False,
        "relaxed_coordinate_payloads_opened": False,
        "endpoint_columns_selected": False,
        "model_or_proxy_potential_used": False,
        "dft_or_energy_proxy_used_at_execution": False,
        "physical_relaxation_executed": False,
        "feature_columns": list(NEXT63_FEATURE_NAMES),
        "metal_chemistry_feature_columns": list(METAL_CHEMISTRY_FEATURE_NAMES),
        "worker_processes": workers,
        "counts": {
            "rows": len(table),
            "metal_chemistry_supported": int(table["metal_chemistry_supported"].sum()),
            "next63_supported": int(table["next63_supported"].sum()),
            "failures": dict(sorted(failures.items())),
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next63_odac23_metal_chemistry_features.py": _sha256(source_path)
        },
        "environment_versions": {
            **_environment_versions(),
            "matminer": importlib.metadata.version("matminer"),
        },
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        feature_path = staging / FEATURES_NAME
        table.to_parquet(feature_path, index=False)
        manifest["outputs_sha256"] = {FEATURES_NAME: _sha256(feature_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != manifest["executed_source_sha256"][
            "src/next63_odac23_metal_chemistry_features.py"
        ]:
            raise RuntimeError("NEXT63 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT63 input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_metal_chemistry_batch(
        selection_dir=args.selection_dir,
        source_dir=args.source_dir,
        design_path=args.design,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


__all__ = [
    "FEATURES_NAME",
    "METAL_CHEMISTRY_FEATURE_NAMES",
    "NEXT63_FEATURE_NAMES",
    "PROTOCOL",
    "MetalChemistryResult",
    "build_metal_chemistry_batch",
    "compute_metal_chemistry_features",
]


if __name__ == "__main__":
    main()
