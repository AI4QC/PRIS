#!/usr/bin/env python3
"""Label-free strong-neighborhood directional-closure features."""

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
from typing import Sequence
import zipfile

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from src.next11_geometry_only_frames import _parse_frame
from src.next19_valence_transport import (
    PeriodicEdgeGeometry,
    build_periodic_edge_geometry,
    infer_valence_assignment,
)
import src.next84_scigen_geometry_lockbox as n84
import src.next93b_wyformer_blind_lockbox as n93b
import src.next168_periodic_local_directional_rigidity as n168
import src.next173_weighted_local_directional_rigidity as n173
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-08-next179-strong-neighborhood-directional-closure-v1"
CATALOGUE_NAME = "NEXT179_STRONG_NEIGHBORHOOD_DIRECTIONAL_CLOSURE_CATALOGUE.json"
SCIGEN_NAME = "next179_scigen_strong_neighborhood_directional_closure.parquet"
WYFORMER_NAME = "next179_wyformer_strong_neighborhood_directional_closure.parquet"
MANIFEST_NAME = "MANIFEST.json"
GRAPH_MODES = n173.GRAPH_MODES
FEATURE_SUFFIXES = (
    "closure_min",
    "closure_q10",
    "closure_mean",
    "volume_q10",
    "volume_mean",
)
FEATURE_NAMES = tuple(
    f"psndc_{mode}_{suffix}" for mode in GRAPH_MODES for suffix in FEATURE_SUFFIXES
)
ROUND_OFF_TOLERANCE = n173.ROUND_OFF_TOLERANCE
EXPECTED_ROWS = n173.EXPECTED_ROWS
EXPECTED_DESIGN_SHA256 = "0eb16c0103bac07eeb6bdf3fac65edd76c95b863caa30e64aaa8d9576c736663"
EXPECTED_INPUT_SHA256 = {
    **n173.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
}


def strong_neighborhood_directional_closure_features(
    *,
    n_sites: int,
    endpoints: Sequence[Sequence[int]] | np.ndarray,
    vectors: Sequence[Sequence[float]] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """Summarize per-site max-relative strong-axis closure certificates."""

    if not isinstance(n_sites, (int, np.integer)) or int(n_sites) < 2:
        raise ValueError("n_sites must be an integer of at least two")
    n_sites = int(n_sites)
    pair = np.asarray(endpoints, dtype=int)
    vector = np.asarray(vectors, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if pair.ndim != 2 or pair.shape[1:] != (2,) or len(pair) < 1:
        raise ValueError("endpoints must have nonempty shape (E,2)")
    if vector.shape != (len(pair), 3):
        raise ValueError("vectors must have shape (E,3)")
    if weight.shape != (len(pair),) or not np.isfinite(weight).all() or np.any(weight <= 0.0):
        raise ValueError("weights must have finite positive shape (E,)")
    if (
        np.any(pair < 0)
        or np.any(pair >= n_sites)
        or np.any(pair[:, 0] == pair[:, 1])
    ):
        raise ValueError("endpoints contain invalid site indices")
    if not np.isfinite(vector).all():
        raise ValueError("vectors must be finite")
    distance = np.linalg.norm(vector, axis=1)
    if not np.isfinite(distance).all() or np.any(distance <= 0.0):
        raise ValueError("vectors must have finite positive lengths")
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("prefix must be a nonempty string")

    direction = vector / distance[:, None]
    max_weight = np.zeros(n_sites, dtype=float)
    for edge_index, (left, right) in enumerate(pair):
        max_weight[left] = max(max_weight[left], weight[edge_index])
        max_weight[right] = max(max_weight[right], weight[edge_index])
    tensors = np.zeros((n_sites, 3, 3), dtype=float)
    for edge_index, (left, right) in enumerate(pair):
        outer = np.outer(direction[edge_index], direction[edge_index])
        tensors[left] += (weight[edge_index] / max_weight[left]) * outer
        tensors[right] += (weight[edge_index] / max_weight[right]) * outer

    closure = np.zeros(n_sites, dtype=float)
    volume = np.zeros(n_sites, dtype=float)
    for site_index in np.flatnonzero(max_weight):
        eigenvalues = np.linalg.eigvalsh(tensors[site_index])
        if (
            not np.isfinite(eigenvalues).all()
            or float(eigenvalues[0]) < -ROUND_OFF_TOLERANCE
        ):
            raise RuntimeError("strong-neighborhood Gram spectrum is invalid")
        eigenvalues = np.maximum(eigenvalues, 0.0)
        closure[site_index] = float(np.clip(eigenvalues[0], 0.0, 1.0))
        volume[site_index] = float(
            np.clip(np.prod(eigenvalues), 0.0, 1.0)
        )

    result = {
        f"{prefix}_closure_min": float(closure.min()),
        f"{prefix}_closure_q10": n168._inverted_cdf(closure, 0.10),
        f"{prefix}_closure_mean": float(closure.mean()),
        f"{prefix}_volume_q10": n168._inverted_cdf(volume, 0.10),
        f"{prefix}_volume_mean": float(volume.mean()),
    }
    expected = tuple(f"{prefix}_{suffix}" for suffix in FEATURE_SUFFIXES)
    values = np.asarray(list(result.values()), dtype=float)
    if (
        tuple(result) != expected
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise RuntimeError("NEXT179 feature schema or bounds differ")
    return result


def _weighted_geometry(
    structure, edges: Sequence[PeriodicEdgeGeometry]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return n173._weighted_geometry(structure, edges)


def compute_strong_neighborhood_directional_closure(structure) -> dict[str, object]:
    """Compute both frozen graph modes with independent fail-open support."""

    row: dict[str, object] = {name: math.nan for name in FEATURE_NAMES}
    for mode in GRAPH_MODES:
        row[f"psndc_{mode}_supported"] = False
        row[f"psndc_{mode}_failure"] = None
    assignment = infer_valence_assignment(structure)
    if not assignment.supported or assignment.values is None:
        reason = assignment.failure_reason or "formal valence assignment failed"
        for mode in GRAPH_MODES:
            row[f"psndc_{mode}_failure"] = reason
        return row
    for mode in GRAPH_MODES:
        geometry = build_periodic_edge_geometry(
            structure, assignment.values, graph_mode=mode
        )
        if not geometry.supported:
            row[f"psndc_{mode}_failure"] = geometry.failure_reason
            continue
        try:
            endpoints, vectors, weights = _weighted_geometry(structure, geometry.edges)
            row.update(
                strong_neighborhood_directional_closure_features(
                    n_sites=len(structure),
                    endpoints=endpoints,
                    vectors=vectors,
                    weights=weights,
                    prefix=f"psndc_{mode}",
                )
            )
        except Exception as exc:
            row[f"psndc_{mode}_failure"] = f"{type(exc).__name__}: {exc}"
            continue
        row[f"psndc_{mode}_supported"] = True
    expected = {
        *FEATURE_NAMES,
        *(f"psndc_{mode}_supported" for mode in GRAPH_MODES),
        *(f"psndc_{mode}_failure" for mode in GRAPH_MODES),
    }
    if set(row) != expected:
        raise RuntimeError("NEXT179 row schema differs")
    return row


def _failure_row(reason: str) -> dict[str, object]:
    row: dict[str, object] = {name: math.nan for name in FEATURE_NAMES}
    for mode in GRAPH_MODES:
        row[f"psndc_{mode}_supported"] = False
        row[f"psndc_{mode}_failure"] = reason
    return row


def _compute_scigen_payload(item: tuple[str, bytes]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        atoms = _parse_frame(payload, strict_output=True).atoms
        structure = AseAtomsAdaptor.get_structure(atoms)
        row = compute_strong_neighborhood_directional_closure(structure)
    except Exception as exc:
        row = _failure_row(f"structure_parse: {type(exc).__name__}: {exc}")
    return material_id, row


def _compute_wyformer_payload(item: tuple[str, str]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        structure = Structure.from_dict(json.loads(payload))
        row = compute_strong_neighborhood_directional_closure(structure)
    except Exception as exc:
        row = _failure_row(f"structure_parse: {type(exc).__name__}: {exc}")
    return material_id, row


def _label_free_statistics(frame: pd.DataFrame) -> dict[str, object]:
    statistics: dict[str, object] = {}
    for mode in GRAPH_MODES:
        mask = frame[f"psndc_{mode}_supported"].fillna(False).astype(bool)
        mode_stats: dict[str, object] = {
            "rows": int(len(frame)),
            "supported": int(mask.sum()),
            "features": {},
        }
        for name in FEATURE_NAMES:
            if not name.startswith(f"psndc_{mode}_"):
                continue
            values = frame.loc[mask, name].to_numpy(dtype=float)
            if len(values) and not np.isfinite(values).all():
                raise RuntimeError(f"NEXT179 non-finite supported feature: {name}")
            mode_stats["features"][name] = {
                "unique_rounded_12": int(len(np.unique(np.round(values, 12)))),
                "minimum": float(values.min()) if len(values) else None,
                "q10": n168._inverted_cdf(values, 0.10) if len(values) else None,
                "median": n168._inverted_cdf(values, 0.50) if len(values) else None,
                "q90": n168._inverted_cdf(values, 0.90) if len(values) else None,
                "maximum": float(values.max()) if len(values) else None,
            }
        statistics[mode] = mode_stats
    return statistics


def build_strong_neighborhood_directional_closure_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 12,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build discovery geometry features without accepting endpoint paths."""

    scigen = Path(scigen_cohort_dir).resolve()
    wyformer = Path(wyformer_cohort_dir).resolve()
    design = Path(design_path).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive exact integer")
    paths = {
        "design": design,
        "next19_source": Path(build_periodic_edge_geometry.__code__.co_filename).resolve(),
        "scigen_manifest": scigen / n84.MANIFEST_NAME,
        "scigen_geometry_discovery": scigen / n84.GEOMETRY_NAMES["discovery"],
        "wyformer_manifest": wyformer / n93b.MANIFEST_NAME,
        "wyformer_geometry_discovery": wyformer / n93b.GEOMETRY_NAMES["discovery"],
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT179 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            key
            for key in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(key) != EXPECTED_INPUT_SHA256.get(key)
        )
        raise ValueError(f"NEXT179 formal input identity differs: {differing}")

    scigen_manifest = json.loads(paths["scigen_manifest"].read_text())
    wyformer_manifest = json.loads(paths["wyformer_manifest"].read_text())
    if (
        scigen_manifest.get("protocol") != n84.PROTOCOL
        or scigen_manifest.get("labels_opened") is not False
        or scigen_manifest.get("endpoint_payloads_opened") is not False
        or scigen_manifest.get("relaxed_structures_opened") is not False
        or scigen_manifest.get("dft_values_used_by_features") is not False
        or scigen_manifest.get("model_or_proxy_potential_used") is not False
        or scigen_manifest.get("physical_relaxation_executed") is not False
        or scigen_manifest.get("outputs_sha256", {}).get(
            n84.GEOMETRY_NAMES["discovery"]
        )
        != input_hashes["scigen_geometry_discovery"]
        or wyformer_manifest.get("protocol") != n93b.PROTOCOL
        or wyformer_manifest.get("labels_opened_by_feature_builder") is not False
        or wyformer_manifest.get("discovery_endpoint_opened") is not False
        or wyformer_manifest.get("validation_endpoint_opened") is not False
        or wyformer_manifest.get("replication_endpoint_opened") is not False
        or wyformer_manifest.get("relaxed_structures_published") is not False
        or wyformer_manifest.get("learned_proxy_execution_input") is not False
        or wyformer_manifest.get("outputs_sha256", {}).get(
            n93b.GEOMETRY_NAMES["discovery"]
        )
        != input_hashes["wyformer_geometry_discovery"]
    ):
        raise ValueError("NEXT179 geometry provenance differs")

    with zipfile.ZipFile(paths["scigen_geometry_discovery"]) as archive:
        names = archive.namelist()
        if names != sorted(names) or any(not name.endswith(".extxyz") for name in names):
            raise ValueError("NEXT179 SCIGEN geometry inventory differs")
        scigen_payloads = [(Path(name).stem, archive.read(name)) for name in names]
    wyformer_geometry = pd.read_parquet(paths["wyformer_geometry_discovery"])
    if (
        set(wyformer_geometry.columns) != {"material_id", "structure_json"}
        or wyformer_geometry["material_id"].astype(str).duplicated().any()
    ):
        raise ValueError("NEXT179 WyFormer geometry inventory differs")
    wyformer_geometry = wyformer_geometry.sort_values("material_id")
    wyformer_payloads = list(
        zip(
            wyformer_geometry["material_id"].astype(str),
            wyformer_geometry["structure_json"].astype(str),
            strict=True,
        )
    )
    if (
        len(scigen_payloads) != EXPECTED_ROWS["scigen"]
        or len(wyformer_payloads) != EXPECTED_ROWS["wyformer"]
        or len({item[0] for item in scigen_payloads}) != len(scigen_payloads)
    ):
        raise ValueError("NEXT179 discovery row identity differs")

    started = time.perf_counter()
    if workers == 1:
        scigen_results = [_compute_scigen_payload(item) for item in scigen_payloads]
        wyformer_results = [_compute_wyformer_payload(item) for item in wyformer_payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            scigen_results = list(
                executor.map(_compute_scigen_payload, scigen_payloads, chunksize=8)
            )
            wyformer_results = list(
                executor.map(_compute_wyformer_payload, wyformer_payloads, chunksize=8)
            )
    elapsed = time.perf_counter() - started

    frames: dict[str, pd.DataFrame] = {}
    expected_columns = {
        "material_id",
        *FEATURE_NAMES,
        *(f"psndc_{mode}_supported" for mode in GRAPH_MODES),
        *(f"psndc_{mode}_failure" for mode in GRAPH_MODES),
    }
    for source, results in (("scigen", scigen_results), ("wyformer", wyformer_results)):
        frame = pd.DataFrame(
            [{"material_id": material_id, **row} for material_id, row in results]
        )
        if (
            len(frame) != EXPECTED_ROWS[source]
            or frame["material_id"].astype(str).duplicated().any()
            or set(frame.columns) != expected_columns
        ):
            raise RuntimeError(f"NEXT179 {source} output schema differs")
        frames[source] = frame
    statistics = {
        source: _label_free_statistics(frame) for source, frame in frames.items()
    }
    catalogue = {
        "protocol": PROTOCOL,
        "graph_modes": GRAPH_MODES,
        "feature_names": FEATURE_NAMES,
        "definition": "site_H_sum_neighbor_weight_over_site_max_times_direction_outer_product_closure_lambda_min_volume_determinant_clipped_to_one",
        "weight_source": "unchanged_NEXT19_PeriodicEdgeGeometry.neighbor_weight",
        "normalization": "per_site_max_incident_neighbor_weight",
        "quantile_method": "inverted_cdf",
        "roundoff_tolerance": ROUND_OFF_TOLERANCE,
        "label_free_statistics": statistics,
        "workers": workers,
        "elapsed_seconds": elapsed,
        "labels_or_endpoints_opened": False,
        "validation_geometry_opened": False,
        "replication_geometry_opened": False,
        "dft_calculation_executed": False,
        "dft_values_used_by_features": False,
        "learned_energy_force_stress_proxy_used": False,
        "physical_relaxation_executed": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    source_paths = {
        "src/next19_valence_transport.py": paths["next19_source"],
        "src/next173_weighted_local_directional_rigidity.py": Path(n173.__file__).resolve(),
        "src/next179_strong_neighborhood_directional_closure.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    try:
        catalogue_path = staging / CATALOGUE_NAME
        scigen_path = staging / SCIGEN_NAME
        wyformer_path = staging / WYFORMER_NAME
        _write_json(catalogue_path, catalogue)
        frames["scigen"].to_parquet(scigen_path, index=False)
        frames["wyformer"].to_parquet(wyformer_path, index=False)
        manifest = {
            "protocol": PROTOCOL,
            "discovery_rows": EXPECTED_ROWS,
            "label_free_statistics": statistics,
            "labels_or_endpoints_opened": False,
            "validation_geometry_opened": False,
            "replication_geometry_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_features": False,
            "learned_energy_force_stress_proxy_used": False,
            "physical_relaxation_executed": False,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {
                CATALOGUE_NAME: _sha256_file(catalogue_path),
                SCIGEN_NAME: _sha256_file(scigen_path),
                WYFORMER_NAME: _sha256_file(wyformer_path),
            },
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT179 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT179 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scigen-cohort-dir", type=Path, required=True)
    parser.add_argument("--wyformer-cohort-dir", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    manifest = build_strong_neighborhood_directional_closure_features(
        scigen_cohort_dir=args.scigen_cohort_dir,
        wyformer_cohort_dir=args.wyformer_cohort_dir,
        design_path=args.design_path,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "FEATURE_NAMES",
    "GRAPH_MODES",
    "_failure_row",
    "build_strong_neighborhood_directional_closure_features",
    "compute_strong_neighborhood_directional_closure",
    "strong_neighborhood_directional_closure_features",
]


if __name__ == "__main__":
    raise SystemExit(main())
