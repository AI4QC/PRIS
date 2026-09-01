#!/usr/bin/env python3
"""Label-free local metal-donor bond-valence descriptors for ODAC23 x0."""

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

from src.advanced_local_features import resolve_bond_valence_parameter
from src.elec_feat import bv_table
from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next20_valence_rigidity import _tabulated_radius
from src.next26_packing import _radii
from src.next49_framework_topology import (
    _canonical_covalent_edges,
    _directed_adjacency,
    _environment_versions,
    _is_metal,
    _strict_geometry,
)
from src.next54_odac23_train_selection import (
    GEOMETRY_NAME as SOURCE_GEOMETRY_NAME,
    MANIFEST_NAME as SOURCE_MANIFEST_NAME,
    METADATA_NAME as SOURCE_METADATA_NAME,
    PROTOCOL as SOURCE_PROTOCOL,
)
from src.next55_odac23_analytic_features import _load_archive
from src.next65_odac23_physics_couplings import (
    FEATURES_NAME as BASE_FEATURES_NAME,
    NEXT65_FEATURE_NAMES,
    PROTOCOL as BASE_FEATURE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next70-odac23-metal-donor-bond-valence-features-v1"
DESIGN_SHA256 = "73fa4ef954ca8d0f85b26479c1e7e93cf36d4d24ab3027e5b9962a24b1d8735f"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "9ea1f0e6c04c8619dd295aa1579da15b51d8241971b3adacb716fdbf93290927"
)
EXPECTED_BASE_MANIFEST_SHA256 = (
    "8a858b58f6772a50b1ee3ea900bef9d66eb5636efadfeef878d9c7740011de5c"
)
DONOR_VALENCE_BY_NUMBER = {
    7: -3,
    8: -2,
    9: -1,
    15: -3,
    16: -2,
    17: -1,
    35: -1,
    53: -1,
}
METAL_DONOR_BV_FEATURE_NAMES = (
    "metal_donor_bv_mismatch_rms",
    "metal_donor_bv_mismatch_q50",
    "metal_donor_bv_mismatch_q90",
    "metal_donor_bv_mismatch_q95",
    "metal_donor_bv_mismatch_max",
    "metal_donor_bv_signed_mismatch_mean",
    "metal_donor_bv_signed_mismatch_std",
    "metal_donor_bv_deficit_q90",
    "metal_donor_bv_deficit_max",
    "metal_donor_bv_excess_q90",
    "metal_donor_bv_excess_max",
    "metal_donor_bv_underbonded_fraction_025",
    "metal_donor_bv_underbonded_fraction_050",
    "metal_donor_bv_overbonded_fraction_025",
    "metal_donor_bv_overbonded_fraction_050",
    "metal_donor_bv_vector_asymmetry_rms",
    "metal_donor_bv_vector_asymmetry_q90",
    "metal_donor_bv_vector_asymmetry_max",
    "metal_donor_bv_effective_cn_mean",
    "metal_donor_bv_effective_cn_min",
    "metal_donor_bv_selected_oxidation_mean",
    "metal_donor_bv_selected_oxidation_std",
    "metal_donor_bv_selected_oxidation_min",
    "metal_donor_bv_selected_oxidation_max",
    "metal_donor_bv_oxidation_ambiguity_gap_mean",
    "metal_donor_bv_oxidation_ambiguity_gap_min",
    "metal_donor_bv_parameter_exact_fraction_mean",
    "metal_donor_bv_parameter_exact_fraction_min",
    "metal_donor_bv_parameter_generic_fraction_mean",
    "metal_donor_bv_parameter_generic_fraction_max",
    "metal_donor_bv_donor_count_mean",
    "metal_donor_bv_donor_count_min",
    "metal_donor_bv_evaluable_metal_fraction",
)
FEATURES_NAME = "next70_odac23_metal_donor_bond_valence_features.parquet"
MANIFEST_NAME = "MANIFEST.json"


@dataclass(frozen=True)
class MetalDonorBondValenceResult:
    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def _candidate_oxidation_states(element: Element) -> tuple[int, ...]:
    common = sorted(
        {
            int(round(float(value)))
            for value in element.common_oxidation_states
            if 0.0 < float(value) <= 8.0 and math.isclose(float(value), round(float(value)))
        }
    )
    if common:
        return tuple(common)
    listed = sorted(
        {
            int(round(float(value)))
            for value in element.oxidation_states
            if 0.0 < float(value) <= 8.0 and math.isclose(float(value), round(float(value)))
        }
    )
    if listed:
        return tuple(listed)
    return (1,)


def _resolve_strength(
    *, metal: Element, oxidation: int, donor: Element, donor_valence: int, distance: float, table
) -> tuple[float, str]:
    resolved = resolve_bond_valence_parameter(
        (metal.symbol, oxidation, donor.symbol, donor_valence),
        table,
        policy="frozen-fallback",
    )
    if resolved is None:
        metal_radius = _tabulated_radius(metal.symbol)
        donor_radius = _tabulated_radius(donor.symbol)
        if metal_radius is None or donor_radius is None:
            raise ValueError(f"missing radius-generic parameter for {metal.symbol}-{donor.symbol}")
        resolved = (metal_radius + donor_radius, 0.37, "radius_generic")
    r0, decay, source = resolved
    if not math.isfinite(float(r0)) or not math.isfinite(float(decay)) or float(decay) <= 0.0:
        raise ValueError("invalid local bond-valence parameter")
    strength = math.exp((float(r0) - distance) / float(decay))
    if not math.isfinite(strength) or strength <= 0.0:
        raise ValueError("invalid local bond-valence strength")
    return strength, str(source)


def _site_record(
    *,
    metal: Element,
    donor_rows: list[tuple[Element, int, float, np.ndarray]],
    table,
) -> dict[str, float]:
    candidates = []
    for oxidation in _candidate_oxidation_states(metal):
        strengths = []
        sources = []
        for donor, donor_valence, distance, _vector in donor_rows:
            strength, source = _resolve_strength(
                metal=metal,
                oxidation=oxidation,
                donor=donor,
                donor_valence=donor_valence,
                distance=distance,
                table=table,
            )
            strengths.append(strength)
            sources.append(source)
        bond_sum = float(np.sum(strengths))
        signed = (bond_sum - oxidation) / oxidation
        candidates.append((abs(signed), oxidation, signed, strengths, sources))
    candidates.sort(key=lambda row: (row[0], row[1]))
    best = candidates[0]
    ambiguity = float(candidates[1][0] - best[0]) if len(candidates) > 1 else 0.0
    _absolute, oxidation, signed, raw_strengths, sources = best
    strengths = np.asarray(raw_strengths, dtype=float)
    vectors = np.asarray([row[3] for row in donor_rows], dtype=float)
    distances = np.linalg.norm(vectors, axis=1)
    directions = vectors / distances[:, None]
    vector_sum = np.sum(strengths[:, None] * directions, axis=0)
    asymmetry = float(np.linalg.norm(vector_sum) / strengths.sum())
    probabilities = strengths / strengths.sum()
    effective_cn = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    source_array = np.asarray(sources, dtype=object)
    return {
        "signed": float(signed),
        "oxidation": float(oxidation),
        "ambiguity": ambiguity,
        "asymmetry": asymmetry,
        "effective_cn": effective_cn,
        "exact_fraction": float(np.mean(source_array == "exact")),
        "generic_fraction": float(
            np.mean(np.isin(source_array, ("brown_generic", "radius_generic")))
        ),
        "donor_count": float(len(donor_rows)),
    }


def compute_metal_donor_bond_valence_features(atoms: Atoms) -> MetalDonorBondValenceResult:
    """Compute intensive local bond-valence summaries from one framework x0."""

    try:
        numbers, _positions, _cell = _strict_geometry(atoms)
        covalent, _vdw = _radii(numbers)
        edges = _canonical_covalent_edges(atoms, np.asarray(covalent, dtype=float))
        adjacency = _directed_adjacency(len(atoms), edges)
        metal_indices = np.flatnonzero([_is_metal(number) for number in numbers])
        if not len(metal_indices):
            raise ValueError("framework has no recognized metal")
        table = bv_table()
        records = []
        for index in metal_indices:
            metal = Element.from_Z(int(numbers[index]))
            donor_rows = []
            for neighbour, _shift, vector, _ratio in adjacency[int(index)]:
                donor_number = int(numbers[neighbour])
                if donor_number not in DONOR_VALENCE_BY_NUMBER:
                    continue
                displacement = np.asarray(vector, dtype=float)
                distance = float(np.linalg.norm(displacement))
                if not math.isfinite(distance) or distance <= 1.0e-10:
                    continue
                donor_rows.append(
                    (
                        Element.from_Z(donor_number),
                        DONOR_VALENCE_BY_NUMBER[donor_number],
                        distance,
                        displacement,
                    )
                )
            if donor_rows:
                records.append(_site_record(metal=metal, donor_rows=donor_rows, table=table))
        if not records:
            raise ValueError("framework has no evaluable metal-donor site")
        signed = np.asarray([row["signed"] for row in records], dtype=float)
        absolute = np.abs(signed)
        deficit = np.maximum(-signed, 0.0)
        excess = np.maximum(signed, 0.0)
        asymmetry = np.asarray([row["asymmetry"] for row in records], dtype=float)
        effective_cn = np.asarray([row["effective_cn"] for row in records], dtype=float)
        oxidation = np.asarray([row["oxidation"] for row in records], dtype=float)
        ambiguity = np.asarray([row["ambiguity"] for row in records], dtype=float)
        exact = np.asarray([row["exact_fraction"] for row in records], dtype=float)
        generic = np.asarray([row["generic_fraction"] for row in records], dtype=float)
        donor_count = np.asarray([row["donor_count"] for row in records], dtype=float)
        values = {
            "metal_donor_bv_mismatch_rms": float(np.sqrt(np.mean(signed**2))),
            "metal_donor_bv_mismatch_q50": float(np.quantile(absolute, 0.50)),
            "metal_donor_bv_mismatch_q90": float(np.quantile(absolute, 0.90)),
            "metal_donor_bv_mismatch_q95": float(np.quantile(absolute, 0.95)),
            "metal_donor_bv_mismatch_max": float(np.max(absolute)),
            "metal_donor_bv_signed_mismatch_mean": float(np.mean(signed)),
            "metal_donor_bv_signed_mismatch_std": float(np.std(signed)),
            "metal_donor_bv_deficit_q90": float(np.quantile(deficit, 0.90)),
            "metal_donor_bv_deficit_max": float(np.max(deficit)),
            "metal_donor_bv_excess_q90": float(np.quantile(excess, 0.90)),
            "metal_donor_bv_excess_max": float(np.max(excess)),
            "metal_donor_bv_underbonded_fraction_025": float(np.mean(deficit > 0.25)),
            "metal_donor_bv_underbonded_fraction_050": float(np.mean(deficit > 0.50)),
            "metal_donor_bv_overbonded_fraction_025": float(np.mean(excess > 0.25)),
            "metal_donor_bv_overbonded_fraction_050": float(np.mean(excess > 0.50)),
            "metal_donor_bv_vector_asymmetry_rms": float(np.sqrt(np.mean(asymmetry**2))),
            "metal_donor_bv_vector_asymmetry_q90": float(np.quantile(asymmetry, 0.90)),
            "metal_donor_bv_vector_asymmetry_max": float(np.max(asymmetry)),
            "metal_donor_bv_effective_cn_mean": float(np.mean(effective_cn)),
            "metal_donor_bv_effective_cn_min": float(np.min(effective_cn)),
            "metal_donor_bv_selected_oxidation_mean": float(np.mean(oxidation)),
            "metal_donor_bv_selected_oxidation_std": float(np.std(oxidation)),
            "metal_donor_bv_selected_oxidation_min": float(np.min(oxidation)),
            "metal_donor_bv_selected_oxidation_max": float(np.max(oxidation)),
            "metal_donor_bv_oxidation_ambiguity_gap_mean": float(np.mean(ambiguity)),
            "metal_donor_bv_oxidation_ambiguity_gap_min": float(np.min(ambiguity)),
            "metal_donor_bv_parameter_exact_fraction_mean": float(np.mean(exact)),
            "metal_donor_bv_parameter_exact_fraction_min": float(np.min(exact)),
            "metal_donor_bv_parameter_generic_fraction_mean": float(np.mean(generic)),
            "metal_donor_bv_parameter_generic_fraction_max": float(np.max(generic)),
            "metal_donor_bv_donor_count_mean": float(np.mean(donor_count)),
            "metal_donor_bv_donor_count_min": float(np.min(donor_count)),
            "metal_donor_bv_evaluable_metal_fraction": float(len(records) / len(metal_indices)),
        }
        if tuple(values) != METAL_DONOR_BV_FEATURE_NAMES or not np.isfinite(
            list(values.values())
        ).all():
            raise ValueError("metal-donor bond-valence feature schema differs")
        return MetalDonorBondValenceResult(True, None, values)
    except Exception as exc:
        return MetalDonorBondValenceResult(False, f"{type(exc).__name__}: {exc}", {})


def _feature_record(atoms: Atoms) -> dict[str, object]:
    result = compute_metal_donor_bond_valence_features(atoms)
    row: dict[str, object] = {
        "metal_donor_bv_supported": result.supported,
        "metal_donor_bv_failure": result.failure_reason,
    }
    row.update(
        {
            name: float(result.features[name]) if result.supported else math.nan
            for name in METAL_DONOR_BV_FEATURE_NAMES
        }
    )
    return row


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("NEXT70 JSON must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def build_metal_donor_bond_valence_batch(
    *,
    source_dir: Path,
    base_feature_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 1,
) -> dict[str, object]:
    """Build all three partitions without opening an endpoint label file."""

    source_dir = Path(source_dir).resolve()
    base_feature_dir = Path(base_feature_dir).resolve()
    design_path = Path(design_path).resolve()
    target = Path(output_dir).resolve()
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("NEXT70 workers must be 1 through 64")
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "metadata": source_dir / SOURCE_METADATA_NAME,
        "geometry": source_dir / SOURCE_GEOMETRY_NAME,
        "source_manifest": source_dir / SOURCE_MANIFEST_NAME,
        "base_features": base_feature_dir / BASE_FEATURES_NAME,
        "base_manifest": base_feature_dir / MANIFEST_NAME,
        "design": design_path,
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT70 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if (
        hashes["source_manifest"] != EXPECTED_SOURCE_MANIFEST_SHA256
        or hashes["base_manifest"] != EXPECTED_BASE_MANIFEST_SHA256
        or hashes["design"] != DESIGN_SHA256
    ):
        raise ValueError("NEXT70 frozen input hash differs")
    source_manifest = _read_json(paths["source_manifest"])
    base_manifest = _read_json(paths["base_manifest"])
    source_outputs = source_manifest.get("outputs_sha256")
    base_outputs = base_manifest.get("outputs_sha256")
    if (
        source_manifest.get("protocol") != SOURCE_PROTOCOL
        or source_manifest.get("selection_frozen_before_row_labels_opened") is not True
        or source_manifest.get("validation_or_test_payload_deserialized") is not False
        or not isinstance(source_outputs, Mapping)
        or source_outputs.get(paths["metadata"].name) != hashes["metadata"]
        or source_outputs.get(paths["geometry"].name) != hashes["geometry"]
        or base_manifest.get("protocol") != BASE_FEATURE_PROTOCOL
        or base_manifest.get("labels_opened") is not False
        or not isinstance(base_outputs, Mapping)
        or base_outputs.get(BASE_FEATURES_NAME) != hashes["base_features"]
    ):
        raise ValueError("NEXT70 label-free provenance differs")
    metadata = pd.read_parquet(paths["metadata"])
    base = pd.read_parquet(paths["base_features"])
    material_ids = tuple(metadata["material_id"].astype(str))
    if (
        len(metadata) != len(base)
        or metadata["material_id"].duplicated().any()
        or base["material_id"].duplicated().any()
        or tuple(base["material_id"].astype(str)) != material_ids
        or not set(NEXT65_FEATURE_NAMES).issubset(base.columns)
    ):
        raise ValueError("NEXT70 base feature identity differs")
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
            if not bool(row["metal_donor_bv_supported"]):
                failures[str(row["metal_donor_bv_failure"])] += 1
            if index % 100 == 0 or index == len(structures):
                print(f"NEXT70 ODAC23 metal-donor bond valence: {index}/{len(structures)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    additions = pd.DataFrame(rows)
    table = pd.concat([base.reset_index(drop=True), additions], axis=1)
    if tuple(additions.loc[:, METAL_DONOR_BV_FEATURE_NAMES].columns) != METAL_DONOR_BV_FEATURE_NAMES:
        raise ValueError("NEXT70 output feature schema differs")
    if not table["combined_supported"].equals(base["combined_supported"]):
        raise RuntimeError("NEXT70 changed pre-existing support")
    source_path = Path(__file__).resolve()
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "selected_odac23_all_partitions_label_free_metal_donor_bond_valence",
        "input_role": "one_raw_unrelaxed_framework_x0_geometry_only",
        "labels_opened": False,
        "relaxed_coordinate_payloads_opened": False,
        "endpoint_columns_selected": False,
        "dft_calculation_or_value_used": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "same_composition_candidates_used": False,
        "missing_policy": "optional_family_fail_open_keep",
        "feature_columns": list(METAL_DONOR_BV_FEATURE_NAMES),
        "base_feature_columns": list(NEXT65_FEATURE_NAMES),
        "worker_processes": workers,
        "counts": {
            "rows": len(table),
            "base_supported": int(table["combined_supported"].sum()),
            "metal_donor_bv_supported": int(table["metal_donor_bv_supported"].sum()),
            "failures": dict(sorted(failures.items())),
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next70_odac23_metal_donor_bond_valence_features.py": _sha256(source_path)
        },
        "environment_versions": {
            **_environment_versions(),
            "ase": importlib.metadata.version("ase"),
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
            "src/next70_odac23_metal_donor_bond_valence_features.py"
        ]:
            raise RuntimeError("NEXT70 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT70 input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--base-feature-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_metal_donor_bond_valence_batch(
        source_dir=args.source_dir,
        base_feature_dir=args.base_feature_dir,
        design_path=args.design,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


__all__ = [
    "METAL_DONOR_BV_FEATURE_NAMES",
    "PROTOCOL",
    "build_metal_donor_bond_valence_batch",
    "compute_metal_donor_bond_valence_features",
]


if __name__ == "__main__":
    main()
