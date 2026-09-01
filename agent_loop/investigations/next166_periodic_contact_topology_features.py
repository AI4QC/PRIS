#!/usr/bin/env python3
"""Freeze periodic opposite-sign contact-network topology from raw x0 geometry."""

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
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-08-next166-periodic-contact-topology-features-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT166_PERIODIC_CONTACT_TOPOLOGY_CATALOGUE.json"
SCIGEN_NAME = "next166_scigen_periodic_contact_topology.parquet"
WYFORMER_NAME = "next166_wyformer_periodic_contact_topology.parquet"
EXPECTED_DESIGN_SHA256 = "aba7edfd790ab479ef4b27c28713156fb52eb532590da4e6b11afc382265ad3a"
EXPECTED_INPUT_SHA256 = {
    "design": EXPECTED_DESIGN_SHA256,
    "next19_source": "f1195a7ef519827f8da1704b9abe773bcee105eff1bdf6dfd5b8eabba1b94712",
    "scigen_manifest": "dc5bf33c6ce6dc2c10bcd3704688055058145fbe7269ada23ffbe4b141d75fe7",
    "scigen_geometry_discovery": "e561ef12343c66dcc72bcabf6b8719ad727e01c9582a094e281da73b862ab575",
    "wyformer_manifest": "e0539d556538cb4c052431bc6a1e5c1663bc3de427677dbc8a446dcc3b4fbc54",
    "wyformer_geometry_discovery": "f1ce5ae4fba8c13fcbf3e25de4f596b919d9b41da5b072d9a28eefeaffc69784",
}
EXPECTED_ROWS = {"scigen": 13_470, "wyformer": 5_232}


GRAPH_MODES = ("voronoi", "crystalnn")
FEATURE_SUFFIXES = (
    "rank_max",
    "rank_mean",
    "rank0_fraction",
    "rank1_fraction",
    "rank2_fraction",
    "rank3_fraction",
)
FEATURE_NAMES = tuple(
    f"pct_{mode}_{suffix}" for mode in GRAPH_MODES for suffix in FEATURE_SUFFIXES
)


def _failure_row(message: str) -> dict[str, object]:
    row: dict[str, object] = {name: math.nan for name in FEATURE_NAMES}
    for mode in GRAPH_MODES:
        row[f"pct_{mode}_supported"] = False
        row[f"pct_{mode}_failure"] = str(message)
    return row


def _integer_vector_rank(vectors: Sequence[np.ndarray]) -> int:
    nonzero = [
        np.asarray(vector, dtype=np.int64)
        for vector in vectors
        if np.any(np.asarray(vector, dtype=np.int64) != 0)
    ]
    if not nonzero:
        return 0
    first = nonzero[0]
    cross = None
    for second in nonzero[1:]:
        candidate = np.cross(first, second)
        if np.any(candidate != 0):
            cross = candidate
            break
    if cross is None:
        return 1
    if any(int(np.dot(cross, third)) != 0 for third in nonzero):
        return 3
    return 2


def periodic_component_ranks(
    n_sites: int, edges: Sequence[PeriodicEdgeGeometry]
) -> np.ndarray:
    """Return the exact periodic translation rank of every site's component."""

    if type(n_sites) is not int or n_sites <= 0:
        raise ValueError("NEXT166 n_sites must be a positive exact integer")
    adjacency: list[list[tuple[int, np.ndarray]]] = [
        [] for _ in range(n_sites)
    ]
    validated: list[tuple[int, int, np.ndarray]] = []
    for edge in edges:
        left = int(edge.cation)
        right = int(edge.anion)
        image_values = tuple(edge.image)
        if (
            left < 0
            or left >= n_sites
            or right < 0
            or right >= n_sites
            or len(image_values) != 3
            or any(int(value) != value for value in image_values)
        ):
            raise ValueError("NEXT166 periodic edge schema differs")
        image = np.asarray(image_values, dtype=np.int64)
        adjacency[left].append((right, image))
        adjacency[right].append((left, -image))
        validated.append((left, right, image))

    ranks = np.zeros(n_sites, dtype=np.int8)
    assigned = np.zeros(n_sites, dtype=bool)
    for root in range(n_sites):
        if assigned[root]:
            continue
        potentials = {root: np.zeros(3, dtype=np.int64)}
        stack = [root]
        assigned[root] = True
        component = []
        while stack:
            left = stack.pop()
            component.append(left)
            for right, image in adjacency[left]:
                if right in potentials:
                    continue
                potentials[right] = potentials[left] + image
                assigned[right] = True
                stack.append(right)
        residuals = [
            potentials[left] + image - potentials[right]
            for left, right, image in validated
            if left in potentials and right in potentials
        ]
        rank = _integer_vector_rank(residuals)
        ranks[np.asarray(component, dtype=int)] = rank
    return ranks


def periodic_topology_features(
    n_sites: int,
    edges: Sequence[PeriodicEdgeGeometry],
    *,
    prefix: str,
) -> dict[str, float]:
    """Aggregate exact component ranks into bounded site-weighted features."""

    ranks = periodic_component_ranks(n_sites, edges)
    result = {
        f"{prefix}_rank_max": float(ranks.max() / 3.0),
        f"{prefix}_rank_mean": float(ranks.mean() / 3.0),
        **{
            f"{prefix}_rank{rank}_fraction": float(np.mean(ranks == rank))
            for rank in range(4)
        },
    }
    values = np.asarray(list(result.values()), dtype=float)
    fractions = np.asarray(
        [result[f"{prefix}_rank{rank}_fraction"] for rank in range(4)]
    )
    if (
        not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or not math.isclose(float(fractions.sum()), 1.0, abs_tol=1.0e-12)
    ):
        raise RuntimeError("NEXT166 topology feature bounds differ")
    return result


def compute_periodic_contact_topology(structure) -> dict[str, object]:
    """Compute both frozen graph modes with independent fail-open support."""

    row: dict[str, object] = {name: math.nan for name in FEATURE_NAMES}
    for mode in GRAPH_MODES:
        row[f"pct_{mode}_supported"] = False
        row[f"pct_{mode}_failure"] = None
    assignment = infer_valence_assignment(structure)
    if not assignment.supported or assignment.values is None:
        reason = assignment.failure_reason or "formal valence assignment failed"
        for mode in GRAPH_MODES:
            row[f"pct_{mode}_failure"] = reason
        return row
    for mode in GRAPH_MODES:
        geometry = build_periodic_edge_geometry(
            structure, assignment.values, graph_mode=mode
        )
        if not geometry.supported:
            row[f"pct_{mode}_failure"] = geometry.failure_reason
            continue
        try:
            values = periodic_topology_features(
                len(structure), geometry.edges, prefix=f"pct_{mode}"
            )
        except Exception as exc:
            row[f"pct_{mode}_failure"] = f"{type(exc).__name__}: {exc}"
            continue
        row.update(values)
        row[f"pct_{mode}_supported"] = True
    expected = {
        *FEATURE_NAMES,
        *(f"pct_{mode}_supported" for mode in GRAPH_MODES),
        *(f"pct_{mode}_failure" for mode in GRAPH_MODES),
    }
    if set(row) != expected:
        raise RuntimeError("NEXT166 row schema differs")
    return row


def _compute_scigen_payload(
    item: tuple[str, bytes],
) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        atoms = _parse_frame(payload, strict_output=True).atoms
        structure = AseAtomsAdaptor.get_structure(atoms)
        return material_id, compute_periodic_contact_topology(structure)
    except Exception as exc:
        return material_id, _failure_row(
            f"structure_parse: {type(exc).__name__}: {exc}"
        )


def _compute_wyformer_payload(
    item: tuple[str, str],
) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        structure = Structure.from_dict(json.loads(payload))
        return material_id, compute_periodic_contact_topology(structure)
    except Exception as exc:
        return material_id, _failure_row(
            f"structure_parse: {type(exc).__name__}: {exc}"
        )


def build_periodic_contact_topology_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 12,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build discovery-only topology features without any endpoint argument."""

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
        raise FileNotFoundError("NEXT166 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            key
            for key in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(key) != EXPECTED_INPUT_SHA256.get(key)
        )
        raise ValueError(f"NEXT166 formal input identity differs: {differing}")

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
        raise ValueError("NEXT166 geometry provenance differs")

    with zipfile.ZipFile(paths["scigen_geometry_discovery"]) as archive:
        names = archive.namelist()
        if names != sorted(names) or any(
            not name.endswith(".extxyz") for name in names
        ):
            raise ValueError("NEXT166 SCIGEN geometry inventory differs")
        scigen_payloads = [
            (Path(name).stem, archive.read(name)) for name in names
        ]
    wyformer_geometry = pd.read_parquet(paths["wyformer_geometry_discovery"])
    if (
        set(wyformer_geometry.columns) != {"material_id", "structure_json"}
        or wyformer_geometry["material_id"].astype(str).duplicated().any()
    ):
        raise ValueError("NEXT166 WyFormer geometry inventory differs")
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
        raise ValueError("NEXT166 discovery row identity differs")

    started = time.perf_counter()
    if workers == 1:
        scigen_results = [_compute_scigen_payload(item) for item in scigen_payloads]
        wyformer_results = [
            _compute_wyformer_payload(item) for item in wyformer_payloads
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            scigen_results = list(
                executor.map(_compute_scigen_payload, scigen_payloads, chunksize=8)
            )
            wyformer_results = list(
                executor.map(_compute_wyformer_payload, wyformer_payloads, chunksize=8)
            )
    elapsed = time.perf_counter() - started

    frames = {}
    for source, results in (
        ("scigen", scigen_results),
        ("wyformer", wyformer_results),
    ):
        frame = pd.DataFrame(
            [{"material_id": material_id, **row} for material_id, row in results]
        )
        if (
            len(frame) != EXPECTED_ROWS[source]
            or frame["material_id"].astype(str).duplicated().any()
            or set(frame.columns)
            != {
                "material_id",
                *FEATURE_NAMES,
                *(f"pct_{mode}_supported" for mode in GRAPH_MODES),
                *(f"pct_{mode}_failure" for mode in GRAPH_MODES),
            }
        ):
            raise RuntimeError(f"NEXT166 {source} output schema differs")
        frames[source] = frame
    coverage = {
        source: {
            mode: {
                "rows": int(len(frame)),
                "supported": int(
                    frame[f"pct_{mode}_supported"].fillna(False).astype(bool).sum()
                ),
            }
            for mode in GRAPH_MODES
        }
        for source, frame in frames.items()
    }
    catalogue = {
        "protocol": PROTOCOL,
        "graph_modes": GRAPH_MODES,
        "feature_names": FEATURE_NAMES,
        "periodic_rank_definition": (
            "exact_Q_rank_of_integer_cycle_closure_translations_site_weighted"
        ),
        "coverage": coverage,
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
        "src/next166_periodic_contact_topology_features.py": Path(__file__).resolve(),
    }
    source_hashes = {
        name: _sha256_file(path) for name, path in source_paths.items()
    }
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
            "coverage": coverage,
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
        if any(
            _sha256_file(path) != input_hashes[name]
            for name, path in paths.items()
        ):
            raise RuntimeError("NEXT166 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT166 source changed before publication")
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
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = build_periodic_contact_topology_features(
        scigen_cohort_dir=args.scigen_cohort_dir,
        wyformer_cohort_dir=args.wyformer_cohort_dir,
        design_path=args.design_path,
        output_dir=args.output_dir,
        workers=args.workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FEATURE_NAMES",
    "GRAPH_MODES",
    "build_periodic_contact_topology_features",
    "compute_periodic_contact_topology",
    "periodic_component_ranks",
    "periodic_topology_features",
]
