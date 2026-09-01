#!/usr/bin/env python3
"""Label-free local bond-valence descriptors for selected ODAC23 x0 frameworks."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
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
from pymatgen.io.ase import AseAtomsAdaptor

from src.advanced_local_features import resolve_bond_valence_parameter
from src.elec_feat import bv_table
from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next19_valence_transport import build_periodic_edge_geometry, infer_valence_assignment
from src.next20_valence_rigidity import _tabulated_radius
from src.next22_bond_valence_equilibrium import (
    FEATURE_NAMES as SCBV_ALL_FEATURE_NAMES,
    scale_calibrated_bond_valence_features,
)
from src.next49_framework_topology import _environment_versions, _is_metal
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


PROTOCOL = "2026-08-03-next69-odac23-local-bond-valence-features-v1"
DESIGN_SHA256 = "5715569c5b114e8326a664fc8821e125212464e19e183c7e7a12b79015e26223"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "9ea1f0e6c04c8619dd295aa1579da15b51d8241971b3adacb716fdbf93290927"
)
EXPECTED_BASE_MANIFEST_SHA256 = (
    "8a858b58f6772a50b1ee3ea900bef9d66eb5636efadfeef878d9c7740011de5c"
)
GRAPH_MODES = ("crystalnn", "voronoi")
GLOBAL_BV_FEATURE_NAMES = tuple(
    name for name in SCBV_ALL_FEATURE_NAMES if name not in {"scbv_edge_count", "scbv_site_count"}
)
METAL_BV_FEATURE_NAMES = (
    "metal_bv_mismatch_rms",
    "metal_bv_mismatch_q50",
    "metal_bv_mismatch_q90",
    "metal_bv_mismatch_q95",
    "metal_bv_mismatch_max",
    "metal_bv_deficit_q90",
    "metal_bv_deficit_max",
    "metal_bv_excess_q90",
    "metal_bv_excess_max",
    "metal_bv_vector_asymmetry_rms",
    "metal_bv_vector_asymmetry_q90",
    "metal_bv_vector_asymmetry_max",
    "metal_bv_effective_cn_mean",
    "metal_bv_effective_cn_min",
    "metal_bv_underbonded_fraction_025",
    "metal_bv_underbonded_fraction_050",
)
BOND_VALENCE_FEATURE_NAMES = GLOBAL_BV_FEATURE_NAMES + METAL_BV_FEATURE_NAMES
NEXT69_FEATURE_NAMES = tuple(
    f"{mode}_{name}" for mode in GRAPH_MODES for name in BOND_VALENCE_FEATURE_NAMES
)
FEATURES_NAME = "next69_odac23_local_bond_valence_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
_FORBIDDEN_TOKENS = ("energy", "force", "stress", "relax", "dft", "label", "target")


@dataclass(frozen=True)
class ODACBondValenceResult:
    any_supported: bool
    mode_supported: Mapping[str, bool]
    failures: Mapping[str, str | None]
    features: Mapping[str, float]


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def _resolved_graph_arrays(
    structure, charges: np.ndarray, geometry
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    table = bv_table()
    endpoints: list[tuple[int, int]] = []
    strengths: list[float] = []
    vectors: list[np.ndarray] = []
    sources: list[str] = []
    for edge in geometry.edges:
        left = int(edge.cation)
        right = int(edge.anion)
        key = (
            structure[left].specie.symbol,
            int(round(float(charges[left]))),
            structure[right].specie.symbol,
            int(round(float(charges[right]))),
        )
        resolved = resolve_bond_valence_parameter(key, table, policy="frozen-fallback")
        if resolved is None:
            left_radius = _tabulated_radius(structure[left].specie.symbol)
            right_radius = _tabulated_radius(structure[right].specie.symbol)
            if left_radius is None or right_radius is None:
                raise ValueError(
                    "bond-valence and radius-generic parameters are missing for "
                    f"{structure[left].specie.symbol}-{structure[right].specie.symbol}"
                )
            resolved = (left_radius + right_radius, 0.37, "radius_generic")
        r0, decay, source = resolved
        if not np.isfinite(r0) or not np.isfinite(decay) or decay <= 0.0:
            raise ValueError("bond-valence parameter is invalid")
        strength = math.exp((float(r0) - float(edge.distance)) / float(decay))
        if not math.isfinite(strength) or strength <= 0.0:
            raise ValueError("bond strength is invalid")
        image = np.asarray(edge.image, dtype=float)
        fractional = (
            np.asarray(structure[right].frac_coords, dtype=float)
            + image
            - np.asarray(structure[left].frac_coords, dtype=float)
        )
        vector = np.asarray(structure.lattice.get_cartesian_coords(fractional), dtype=float)
        endpoints.append((left, right))
        strengths.append(strength)
        vectors.append(vector)
        sources.append(str(source))
    return (
        np.asarray(endpoints, dtype=int),
        np.asarray(strengths, dtype=float),
        np.asarray(vectors, dtype=float),
        tuple(sources),
    )


def _metal_features(
    structure,
    charges: np.ndarray,
    endpoints: np.ndarray,
    strengths: np.ndarray,
    vectors: np.ndarray,
) -> dict[str, float]:
    n_sites = len(structure)
    distance = np.linalg.norm(vectors, axis=1)
    direction = vectors / distance[:, None]
    site_sum = np.zeros(n_sites, dtype=float)
    site_vector = np.zeros((n_sites, 3), dtype=float)
    site_strengths: list[list[float]] = [[] for _ in range(n_sites)]
    for index, (left, right) in enumerate(endpoints):
        value = float(strengths[index])
        vector = value * direction[index]
        site_sum[left] += value
        site_sum[right] += value
        site_vector[left] += vector
        site_vector[right] -= vector
        site_strengths[left].append(value)
        site_strengths[right].append(value)
    target = np.abs(charges)
    denominator = float(np.dot(site_sum, site_sum))
    global_scale = float(np.dot(site_sum, target) / denominator)
    charge_rms = float(np.sqrt(np.mean(target**2)))
    if not math.isfinite(global_scale) or global_scale <= 0.0 or charge_rms <= 0.0:
        raise ValueError("metal bond-valence scale is invalid")
    mismatch = (global_scale * site_sum - target) / charge_rms
    asymmetry = np.ones(n_sites, dtype=float)
    active = site_sum > 0.0
    asymmetry[active] = np.linalg.norm(site_vector[active], axis=1) / site_sum[active]
    effective_cn = np.zeros(n_sites, dtype=float)
    for index, values in enumerate(site_strengths):
        if not values:
            continue
        probabilities = np.asarray(values, dtype=float)
        probabilities /= probabilities.sum()
        effective_cn[index] = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    metals = np.asarray([_is_metal(int(site.specie.Z)) for site in structure], dtype=bool)
    if not metals.any():
        raise ValueError("structure has no metal site")
    metal_mismatch = mismatch[metals]
    absolute = np.abs(metal_mismatch)
    deficit = np.maximum(-metal_mismatch, 0.0)
    excess = np.maximum(metal_mismatch, 0.0)
    metal_asymmetry = asymmetry[metals]
    metal_cn = effective_cn[metals]
    values = {
        "metal_bv_mismatch_rms": float(np.sqrt(np.mean(metal_mismatch**2))),
        "metal_bv_mismatch_q50": _quantile(absolute, 0.50),
        "metal_bv_mismatch_q90": _quantile(absolute, 0.90),
        "metal_bv_mismatch_q95": _quantile(absolute, 0.95),
        "metal_bv_mismatch_max": float(np.max(absolute)),
        "metal_bv_deficit_q90": _quantile(deficit, 0.90),
        "metal_bv_deficit_max": float(np.max(deficit)),
        "metal_bv_excess_q90": _quantile(excess, 0.90),
        "metal_bv_excess_max": float(np.max(excess)),
        "metal_bv_vector_asymmetry_rms": float(np.sqrt(np.mean(metal_asymmetry**2))),
        "metal_bv_vector_asymmetry_q90": _quantile(metal_asymmetry, 0.90),
        "metal_bv_vector_asymmetry_max": float(np.max(metal_asymmetry)),
        "metal_bv_effective_cn_mean": float(np.mean(metal_cn)),
        "metal_bv_effective_cn_min": float(np.min(metal_cn)),
        "metal_bv_underbonded_fraction_025": float(np.mean(deficit > 0.25)),
        "metal_bv_underbonded_fraction_050": float(np.mean(deficit > 0.50)),
    }
    if tuple(values) != METAL_BV_FEATURE_NAMES or not np.isfinite(list(values.values())).all():
        raise ValueError("metal bond-valence feature schema is invalid")
    return values


def compute_odac23_local_bond_valence_features(atoms: Atoms) -> ODACBondValenceResult:
    """Compute two independent analytic bond-valence families from raw x0."""

    features = {name: math.nan for name in NEXT69_FEATURE_NAMES}
    supported = {mode: False for mode in GRAPH_MODES}
    failures: dict[str, str | None] = {mode: None for mode in GRAPH_MODES}
    try:
        if len(atoms) < 2 or not np.all(atoms.pbc):
            raise ValueError("periodic structure must have at least two atoms")
        structure = AseAtomsAdaptor.get_structure(atoms)
        assignment = infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(assignment.failure_reason or "valence assignment unsupported")
        charges = np.asarray(assignment.values, dtype=float)
    except Exception as exc:
        reason = f"valence_assignment: {type(exc).__name__}: {exc}"
        return ODACBondValenceResult(False, supported, {mode: reason for mode in GRAPH_MODES}, features)
    for mode in GRAPH_MODES:
        try:
            geometry = build_periodic_edge_geometry(structure, charges, graph_mode=mode)
            if not geometry.supported:
                raise ValueError(geometry.failure_reason or "periodic graph unsupported")
            endpoints, strengths, vectors, sources = _resolved_graph_arrays(
                structure, charges, geometry
            )
            global_result = scale_calibrated_bond_valence_features(
                charges=charges,
                endpoints=endpoints,
                strengths=strengths,
                vectors=vectors,
                parameter_sources=sources,
            )
            if not global_result.supported:
                raise ValueError(global_result.failure_reason or "global bond valence unsupported")
            mode_values = {
                **{name: float(global_result.features[name]) for name in GLOBAL_BV_FEATURE_NAMES},
                **_metal_features(structure, charges, endpoints, strengths, vectors),
            }
            if tuple(mode_values) != BOND_VALENCE_FEATURE_NAMES:
                raise ValueError("bond-valence mode feature schema differs")
            features.update({f"{mode}_{name}": value for name, value in mode_values.items()})
            supported[mode] = True
        except Exception as exc:
            failures[mode] = f"{type(exc).__name__}: {exc}"
    return ODACBondValenceResult(any(supported.values()), supported, failures, features)


def _feature_record(atoms: Atoms) -> dict[str, object]:
    result = compute_odac23_local_bond_valence_features(atoms)
    row: dict[str, object] = {"bond_valence_any_supported": result.any_supported}
    for mode in GRAPH_MODES:
        row[f"{mode}_bond_valence_supported"] = bool(result.mode_supported[mode])
        row[f"{mode}_bond_valence_failure"] = result.failures[mode]
    row.update(result.features)
    return row


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("NEXT69 JSON must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def build_odac23_local_bond_valence_batch(
    *,
    source_dir: Path,
    base_feature_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 1,
) -> dict[str, object]:
    """Build all partitions without opening any endpoint label payload."""

    source_dir = Path(source_dir).resolve()
    base_feature_dir = Path(base_feature_dir).resolve()
    design_path = Path(design_path).resolve()
    target = Path(output_dir).resolve()
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("NEXT69 workers must be 1 through 64")
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
        raise FileNotFoundError("NEXT69 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if (
        hashes["source_manifest"] != EXPECTED_SOURCE_MANIFEST_SHA256
        or hashes["base_manifest"] != EXPECTED_BASE_MANIFEST_SHA256
        or hashes["design"] != DESIGN_SHA256
    ):
        raise ValueError("NEXT69 frozen input hash differs")
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
        raise ValueError("NEXT69 label-free provenance differs")
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
        raise ValueError("NEXT69 base feature identity differs")
    structures = _load_archive(paths["geometry"], material_ids)
    rows = []
    failures = {mode: Counter() for mode in GRAPH_MODES}
    if workers == 1:
        iterator = map(_feature_record, structures)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_feature_record, structures, chunksize=2)
    try:
        for index, row in enumerate(iterator, start=1):
            rows.append(row)
            for mode in GRAPH_MODES:
                if not bool(row[f"{mode}_bond_valence_supported"]):
                    failures[mode][str(row[f"{mode}_bond_valence_failure"])] += 1
            if index % 100 == 0 or index == len(structures):
                print(f"NEXT69 ODAC23 bond valence: {index}/{len(structures)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    additions = pd.DataFrame(rows)
    if tuple(additions.loc[:, NEXT69_FEATURE_NAMES].columns) != NEXT69_FEATURE_NAMES:
        raise ValueError("NEXT69 output feature schema differs")
    table = pd.concat([base.reset_index(drop=True), additions], axis=1)
    if not table["combined_supported"].equals(base["combined_supported"]):
        raise RuntimeError("NEXT69 changed pre-existing support")

    source_path = Path(__file__).resolve()
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "selected_odac23_all_partitions_label_free_local_bond_valence",
        "input_role": "one_raw_unrelaxed_framework_x0_geometry_only",
        "labels_opened": False,
        "relaxed_coordinate_payloads_opened": False,
        "endpoint_columns_selected": False,
        "dft_calculation_or_value_used": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "same_composition_candidates_used": False,
        "missing_policy": "per_graph_fail_open_keep",
        "graph_modes": list(GRAPH_MODES),
        "feature_columns": list(NEXT69_FEATURE_NAMES),
        "base_feature_columns": list(NEXT65_FEATURE_NAMES),
        "worker_processes": workers,
        "counts": {
            "rows": len(table),
            "base_supported": int(table["combined_supported"].sum()),
            "any_bond_valence_supported": int(table["bond_valence_any_supported"].sum()),
            **{
                f"{mode}_supported": int(table[f"{mode}_bond_valence_supported"].sum())
                for mode in GRAPH_MODES
            },
            "failures": {
                mode: dict(sorted(counter.items())) for mode, counter in failures.items()
            },
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next69_odac23_local_bond_valence_features.py": _sha256(source_path),
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
            "src/next69_odac23_local_bond_valence_features.py"
        ]:
            raise RuntimeError("NEXT69 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT69 input changed before publication")
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
    manifest = build_odac23_local_bond_valence_batch(
        source_dir=args.source_dir,
        base_feature_dir=args.base_feature_dir,
        design_path=args.design,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


__all__ = [
    "BOND_VALENCE_FEATURE_NAMES",
    "NEXT69_FEATURE_NAMES",
    "PROTOCOL",
    "build_odac23_local_bond_valence_batch",
    "compute_odac23_local_bond_valence_features",
]


if __name__ == "__main__":
    main()
