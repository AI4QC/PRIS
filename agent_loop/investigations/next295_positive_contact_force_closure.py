#!/usr/bin/env python3
"""Build sign-sensitive positive contact-force-closure features from raw x0."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping, Sequence

from ase import Atoms
import numpy as np
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.optimize import linprog

import src.next168_periodic_local_directional_rigidity as n168
import src.next19_valence_transport as n19
import src.next267_periodic_radical_voronoi_packing as n267
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next295-positive-contact-force-closure-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT295_PCFC_FEATURE_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "next295_scigen_positive_contact_force_closure_features.parquet",
    "wyformer": "next295_wyformer_positive_contact_force_closure_features.parquet",
}
METRIC_NAMES = (
    "uniform_closure",
    "weighted_closure",
    "uniform_equilibrium",
    "weighted_equilibrium",
)
STATISTIC_NAMES = ("min", "q10", "mean")
FEATURE_NAMES = tuple(
    [
        f"pcfc_{metric}_{statistic}"
        for metric in METRIC_NAMES
        for statistic in STATISTIC_NAMES
    ]
    + ["pcfc_locally_enclosed_fraction"]
)
FEATURE_DIRECTIONS = {name: "protected_high" for name in FEATURE_NAMES}
EXPECTED_ROWS = {"scigen": 13_470, "wyformer": 5_232}
EXPECTED_DESIGN_SHA256 = (
    "431edb15c01b0423c60b3613c32f67c74ba4b31b92607efcfecc426b91e1518b"
)
EXPECTED_INPUT_SHA256 = {
    "scigen_manifest": "dc5bf33c6ce6dc2c10bcd3704688055058145fbe7269ada23ffbe4b141d75fe7",
    "scigen_metadata": "f91455f23b0a96f60fd1c779249e2be46a7ecf94fcdde2b146426a95aac05bde",
    "scigen_discovery_geometry": "e561ef12343c66dcc72bcabf6b8719ad727e01c9582a094e281da73b862ab575",
    "wyformer_manifest": "e0539d556538cb4c052431bc6a1e5c1663bc3de427677dbc8a446dcc3b4fbc54",
    "wyformer_metadata": "3b152b4b84c8d3f7ff5e85611dc1fd2728296f150e907ac4578ce55d2b27dd2b",
    "wyformer_discovery_geometry": "f1ce5ae4fba8c13fcbf3e25de4f596b919d9b41da5b072d9a28eefeaffc69784",
    "design": EXPECTED_DESIGN_SHA256,
    "next19_source": "f1195a7ef519827f8da1704b9abe773bcee105eff1bdf6dfd5b8eabba1b94712",
    "next168_source": "f4cb114cd14a41c3f416d7782c2b45715b9b73fd07090795192d530e4bb2ad24",
    "next267_source": "8f1e7ed9eb73a81a5755d455ffc05aab6f539cbd66afbbbfc384ca88391adca1",
}
BOUNDARY_FLAGS = n267.BOUNDARY_FLAGS
MINIMUM_FORMAL_COVERAGE = 0.90
DIRECTION_NORM_TOLERANCE = 1.0e-12
RANK_RELATIVE_TOLERANCE = 1.0e-10
CLOSURE_NUMERICAL_TOLERANCE = 1.0e-12
LP_RESIDUAL_TOLERANCE = 1.0e-9
ENCLOSURE_TOLERANCE = 1.0e-9
OUTPUT_GRID = 10_000_000_000


@dataclass(frozen=True)
class PCFCFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    edge_count: int
    enclosed_site_count: int
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> PCFCFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return PCFCFeatureResult(False, reason, 0, 0, 0, {})


def _quantized(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def _bounded(value: float, *, tolerance: float, label: str) -> float:
    value = float(value)
    if (
        not math.isfinite(value)
        or value < -tolerance
        or value > 1.0 + tolerance
    ):
        raise ValueError(f"NEXT295 {label} differs")
    return _quantized(float(np.clip(value, 0.0, 1.0)))


def _normalized_directions(directions: object) -> np.ndarray:
    vector = np.asarray(directions, dtype=float)
    if vector.ndim != 2 or vector.shape[1:] != (3,) or not np.isfinite(vector).all():
        raise ValueError("NEXT295 directions differ")
    if len(vector) == 0:
        return vector.copy()
    norm = np.linalg.norm(vector, axis=1)
    if not np.isfinite(norm).all() or np.any(norm <= DIRECTION_NORM_TOLERANCE):
        raise ValueError("NEXT295 directions differ")
    return vector / norm[:, None]


def _normalized_prior(prior: object, count: int) -> np.ndarray:
    weight = np.asarray(prior, dtype=float)
    if (
        weight.shape != (count,)
        or not np.isfinite(weight).all()
        or (count and np.any(weight <= 0.0))
    ):
        raise ValueError("NEXT295 prior differs")
    if count == 0:
        return weight.copy()
    total = math.fsum(float(value) for value in weight)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=CLOSURE_NUMERICAL_TOLERANCE):
        raise ValueError("NEXT295 prior differs")
    return weight / total


def positive_equilibrium_fraction(directions: object, prior: object) -> float:
    """Return the frozen strict-positive convex-balance margin in ``[0, 1]``."""

    direction = _normalized_directions(directions)
    q = _normalized_prior(prior, len(direction))
    if len(direction) < 4:
        return 0.0
    singular = np.linalg.svd(direction, compute_uv=False)
    if (
        singular.shape != (3,)
        or not np.isfinite(singular).all()
        or singular[0] <= 0.0
        or int(np.count_nonzero(singular > singular[0] * RANK_RELATIVE_TOLERANCE)) < 3
    ):
        return 0.0

    count = len(direction)
    objective = np.zeros(count + 1, dtype=float)
    objective[-1] = -1.0
    inequality = np.column_stack((-np.eye(count), q))
    equality = np.vstack(
        (
            np.column_stack((direction.T, np.zeros(3, dtype=float))),
            np.append(np.ones(count, dtype=float), 0.0),
        )
    )
    program = linprog(
        objective,
        A_ub=inequality,
        b_ub=np.zeros(count, dtype=float),
        A_eq=equality,
        b_eq=np.asarray([0.0, 0.0, 0.0, 1.0]),
        bounds=[(0.0, None)] * count + [(0.0, 1.0)],
        method="highs",
        options={
            "primal_feasibility_tolerance": 1.0e-10,
            "dual_feasibility_tolerance": 1.0e-10,
        },
    )
    if program.status == 2:
        return 0.0
    if (
        not program.success
        or program.x.shape != (count + 1,)
        or not np.isfinite(program.x).all()
    ):
        raise ValueError(f"NEXT295 equilibrium linear program differs: {program.message}")
    force = np.asarray(program.x[:-1], dtype=float)
    alpha = float(program.x[-1])
    if (
        np.max(np.abs(direction.T @ force)) > LP_RESIDUAL_TOLERANCE
        or abs(float(force.sum()) - 1.0) > LP_RESIDUAL_TOLERANCE
        or np.min(force) < -LP_RESIDUAL_TOLERANCE
        or np.min(force - alpha * q) < -LP_RESIDUAL_TOLERANCE
        or alpha < -LP_RESIDUAL_TOLERANCE
        or alpha > 1.0 + LP_RESIDUAL_TOLERANCE
    ):
        raise ValueError("NEXT295 equilibrium certificate residual differs")
    return _bounded(alpha, tolerance=LP_RESIDUAL_TOLERANCE, label="equilibrium")


def site_pcfc_metrics(directions: object, weights: object) -> dict[str, float]:
    """Return five PCFC metrics for one crystallographic site."""

    direction = _normalized_directions(directions)
    graph_weight = np.asarray(weights, dtype=float)
    if (
        graph_weight.shape != (len(direction),)
        or not np.isfinite(graph_weight).all()
        or (len(direction) and np.any(graph_weight <= DIRECTION_NORM_TOLERANCE))
    ):
        raise ValueError("NEXT295 weights differ")
    if len(direction) == 0:
        return {
            "uniform_closure": 0.0,
            "weighted_closure": 0.0,
            "uniform_equilibrium": 0.0,
            "weighted_equilibrium": 0.0,
            "locally_enclosed": 0.0,
        }
    uniform = np.full(len(direction), 1.0 / len(direction), dtype=float)
    weighted = graph_weight / math.fsum(float(value) for value in graph_weight)

    def closure(prior: np.ndarray) -> float:
        resultant = np.einsum("i,ij->j", prior, direction)
        return _bounded(
            1.0 - float(np.linalg.norm(resultant)),
            tolerance=CLOSURE_NUMERICAL_TOLERANCE,
            label="directional closure",
        )

    uniform_equilibrium = positive_equilibrium_fraction(direction, uniform)
    weighted_equilibrium = positive_equilibrium_fraction(direction, weighted)
    result = {
        "uniform_closure": closure(uniform),
        "weighted_closure": closure(weighted),
        "uniform_equilibrium": uniform_equilibrium,
        "weighted_equilibrium": weighted_equilibrium,
        "locally_enclosed": float(uniform_equilibrium > ENCLOSURE_TOLERANCE),
    }
    if tuple(result) != (*METRIC_NAMES, "locally_enclosed"):
        raise RuntimeError("NEXT295 site metric schema differs")
    return result


def _mean(values: np.ndarray) -> float:
    return float(math.fsum(float(value) for value in values) / len(values))


def positive_contact_force_closure_features(
    *,
    n_sites: int,
    endpoints: Sequence[Sequence[int]] | np.ndarray,
    vectors: Sequence[Sequence[float]] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Aggregate frozen PCFC site certificates over a periodic edge graph."""

    if not isinstance(n_sites, (int, np.integer)) or int(n_sites) < 2:
        raise ValueError("NEXT295 n_sites differs")
    n_sites = int(n_sites)
    raw_pair = np.asarray(endpoints)
    if raw_pair.ndim != 2 or raw_pair.shape[1:] != (2,) or len(raw_pair) < 1:
        raise ValueError("NEXT295 endpoints differ")
    try:
        numeric_pair = np.asarray(raw_pair, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("NEXT295 endpoints differ") from exc
    if (
        not np.isfinite(numeric_pair).all()
        or not np.equal(numeric_pair, np.rint(numeric_pair)).all()
    ):
        raise ValueError("NEXT295 endpoints differ")
    pair = numeric_pair.astype(int)
    if (
        np.any(pair < 0)
        or np.any(pair >= n_sites)
        or np.any(pair[:, 0] == pair[:, 1])
    ):
        raise ValueError("NEXT295 endpoints differ")
    vector = np.asarray(vectors, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if vector.shape != (len(pair), 3) or not np.isfinite(vector).all():
        raise ValueError("NEXT295 vectors differ")
    distance = np.linalg.norm(vector, axis=1)
    if not np.isfinite(distance).all() or np.any(distance <= DIRECTION_NORM_TOLERANCE):
        raise ValueError("NEXT295 vectors differ")
    if (
        weight.shape != (len(pair),)
        or not np.isfinite(weight).all()
        or np.any(weight <= DIRECTION_NORM_TOLERANCE)
    ):
        raise ValueError("NEXT295 weights differ")
    direction = vector / distance[:, None]
    site_directions: list[list[np.ndarray]] = [[] for _ in range(n_sites)]
    site_weights: list[list[float]] = [[] for _ in range(n_sites)]
    for index, (left, right) in enumerate(pair):
        site_directions[int(left)].append(direction[index])
        site_weights[int(left)].append(float(weight[index]))
        site_directions[int(right)].append(-direction[index])
        site_weights[int(right)].append(float(weight[index]))
    populations = {name: np.zeros(n_sites, dtype=float) for name in METRIC_NAMES}
    enclosed = np.zeros(n_sites, dtype=float)
    for site in range(n_sites):
        directions_at_site = np.asarray(site_directions[site], dtype=float).reshape((-1, 3))
        weights_at_site = np.asarray(site_weights[site], dtype=float)
        metrics = site_pcfc_metrics(directions_at_site, weights_at_site)
        for name in METRIC_NAMES:
            populations[name][site] = metrics[name]
        enclosed[site] = metrics["locally_enclosed"]
    features: dict[str, float] = {}
    for name in METRIC_NAMES:
        values = populations[name]
        features[f"pcfc_{name}_min"] = _quantized(float(values.min()))
        features[f"pcfc_{name}_q10"] = _quantized(
            float(np.quantile(values, 0.10, method="inverted_cdf"))
        )
        features[f"pcfc_{name}_mean"] = _quantized(_mean(values))
    features["pcfc_locally_enclosed_fraction"] = _quantized(_mean(enclosed))
    values = np.asarray(list(features.values()), dtype=float)
    if (
        tuple(features) != FEATURE_NAMES
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise RuntimeError("NEXT295 aggregate feature schema or bounds differ")
    return features


def _geometry_only_atoms(atoms: Atoms) -> Atoms:
    try:
        pbc = np.asarray(atoms.pbc, dtype=bool)
        cell = np.asarray(atoms.cell.array, dtype=float)
        positions = np.asarray(atoms.positions, dtype=float)
        numbers = np.asarray(atoms.numbers)
    except Exception as exc:
        raise ValueError("NEXT295 features require exact periodic geometry-only Atoms") from exc
    if (
        not isinstance(atoms, Atoms)
        or len(atoms) < 2
        or atoms.calc is not None
        or bool(atoms.info)
        or set(atoms.arrays) != {"numbers", "positions"}
        or pbc.shape != (3,)
        or not np.all(pbc)
        or cell.shape != (3, 3)
        or positions.shape != (len(atoms), 3)
        or numbers.shape != (len(atoms),)
        or not np.isfinite(cell).all()
        or not np.isfinite(positions).all()
        or not np.isfinite(numbers).all()
        or abs(float(np.linalg.det(cell))) <= DIRECTION_NORM_TOLERANCE
    ):
        raise ValueError("NEXT295 features require exact periodic geometry-only Atoms")
    return atoms.copy()


def compute_pcfc_features(atoms: Atoms) -> PCFCFeatureResult:
    """Compute PCFC from element identities and one raw periodic geometry."""

    try:
        work = _geometry_only_atoms(atoms)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT295 formal valence assignment failed"
            )
        geometry = n19.build_periodic_edge_geometry(
            structure, assignment.values, graph_mode="crystalnn"
        )
        if not geometry.supported or not geometry.edges:
            raise ValueError(geometry.failure_reason or "NEXT295 CrystalNN graph unsupported")
        endpoints, vectors = n168._vectors_from_edges(structure, geometry.edges)
        weights = np.asarray(
            [float(edge.neighbor_weight) for edge in geometry.edges], dtype=float
        )
        features = positive_contact_force_closure_features(
            n_sites=len(structure),
            endpoints=endpoints,
            vectors=vectors,
            weights=weights,
        )
        enclosed_fraction = features["pcfc_locally_enclosed_fraction"]
        enclosed_count = int(round(enclosed_fraction * len(structure)))
        return PCFCFeatureResult(
            True,
            None,
            len(structure),
            len(geometry.edges),
            enclosed_count,
            features,
        )
    except Exception as exc:
        return _failure(exc)


def compute_pcfc_row(atoms: Atoms) -> dict[str, object]:
    result = compute_pcfc_features(atoms)
    row: dict[str, object] = {name: math.nan for name in FEATURE_NAMES}
    row.update(
        {
            "pcfc_supported": bool(result.supported),
            "pcfc_failure": result.failure_reason,
            "pcfc_site_count": result.site_count,
            "pcfc_edge_count": result.edge_count,
            "pcfc_enclosed_site_count": result.enclosed_site_count,
        }
    )
    if result.supported:
        row.update({name: float(result.features[name]) for name in FEATURE_NAMES})
    return row


def _error_row(exc: Exception) -> dict[str, object]:
    return {
        **{name: math.nan for name in FEATURE_NAMES},
        "pcfc_supported": False,
        "pcfc_failure": f"{type(exc).__name__}: {exc}",
        "pcfc_site_count": 0,
        "pcfc_edge_count": 0,
        "pcfc_enclosed_site_count": 0,
    }


def _compute_scigen_payload(item: tuple[str, bytes]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        atoms = n267.n85._parse_frame(payload, strict_output=True).atoms
        return material_id, compute_pcfc_row(atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_wyformer_payload(item: tuple[str, str]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        structure = n267.Structure.from_dict(json.loads(payload))
        atoms = n267.AseAtomsAdaptor.get_atoms(structure)
        return material_id, compute_pcfc_row(atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_many(payloads, *, source: str, workers: int):
    worker = _compute_scigen_payload if source == "scigen" else _compute_wyformer_payload
    if workers == 1:
        return [worker(item) for item in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(worker, payloads, chunksize=8))


def _label_free_statistics(table: pd.DataFrame) -> dict[str, object]:
    supported = table["pcfc_supported"].fillna(False).astype(bool)
    statistics: dict[str, object] = {}
    for name in FEATURE_NAMES:
        values = pd.to_numeric(table.loc[supported, name], errors="coerce").to_numpy(float)
        statistics[name] = {
            "unique_rounded_10": int(len(np.unique(np.round(values, 10)))),
            "minimum": float(values.min()),
            "q10": float(np.quantile(values, 0.10, method="inverted_cdf")),
            "median": float(np.quantile(values, 0.50, method="inverted_cdf")),
            "q90": float(np.quantile(values, 0.90, method="inverted_cdf")),
            "maximum": float(values.max()),
        }
    return statistics


def build_cross_source_pcfc_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build NEXT295 from physically isolated discovery geometry only."""

    scigen = Path(scigen_cohort_dir).resolve()
    wyformer = Path(wyformer_cohort_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "scigen_manifest": scigen / n267.n85.COHORT_MANIFEST_NAME,
        "scigen_metadata": scigen / n267.n85.COHORT_METADATA_NAME,
        "scigen_discovery_geometry": scigen / n267.n85.GEOMETRY_NAMES["discovery"],
        "wyformer_manifest": wyformer / n267.n94.COHORT_MANIFEST_NAME,
        "wyformer_metadata": wyformer / n267.n94.COHORT_METADATA_NAME,
        "wyformer_discovery_geometry": wyformer / n267.n94.GEOMETRY_NAMES["discovery"],
        "design": Path(design_path).resolve(),
        "next19_source": Path(n19.__file__).resolve(),
        "next168_source": Path(n168.__file__).resolve(),
        "next267_source": Path(n267.__file__).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("NEXT295 workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT295 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT295 formal input identity differs: {differing}")
    scigen_manifest = n267._read_manifest(paths["scigen_manifest"])
    wyformer_manifest = n267._read_manifest(paths["wyformer_manifest"])
    if (
        scigen_manifest.get("protocol") != n267.n85.COHORT_PROTOCOL
        or scigen_manifest.get("labels_opened") is not False
        or scigen_manifest.get("endpoint_payloads_opened") is not False
        or scigen_manifest.get("relaxed_structures_opened") is not False
        or wyformer_manifest.get("protocol") != n267.n94.COHORT_PROTOCOL
        or wyformer_manifest.get("discovery_endpoint_opened") is not False
        or wyformer_manifest.get("validation_endpoint_opened") is not False
        or wyformer_manifest.get("replication_endpoint_opened") is not False
        or wyformer_manifest.get("relaxed_structures_published") is not False
    ):
        raise ValueError("NEXT295 discovery geometry provenance differs")
    metadata = {
        "scigen": pd.read_parquet(paths["scigen_metadata"]),
        "wyformer": pd.read_parquet(paths["wyformer_metadata"]),
    }
    discovery: dict[str, pd.DataFrame] = {}
    for source, frame in metadata.items():
        required = {
            "material_id",
            "reduced_formula",
            "chemical_system",
            "natoms",
            "partition_role",
            "input_role",
        }
        if required - set(frame.columns) or frame["material_id"].astype(str).duplicated().any():
            raise ValueError(f"NEXT295 {source} metadata differs")
        selected = frame.loc[frame["partition_role"].eq("discovery")].copy()
        selected = selected.sort_values("material_id", kind="mergesort").reset_index(drop=True)
        if len(selected) != EXPECTED_ROWS[source]:
            raise ValueError(f"NEXT295 {source} discovery identity differs")
        discovery[source] = selected
    payloads = {
        "scigen": n267.n85._archive_payloads(
            paths["scigen_discovery_geometry"],
            discovery["scigen"]["material_id"].astype(str).tolist(),
        ),
        "wyformer": n267.n94._payloads(
            paths["wyformer_discovery_geometry"],
            discovery["wyformer"]["material_id"].astype(str).tolist(),
        ),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    test_path = source_path.parent.parent / "tests/test_next295_positive_contact_force_closure.py"
    executed_hashes = {
        "src/next295_positive_contact_force_closure.py": _sha256_file(source_path),
        "tests/test_next295_positive_contact_force_closure.py": _sha256_file(test_path),
    }
    started = time.perf_counter()
    try:
        computed = {
            source: _compute_many(payloads[source], source=source, workers=workers)
            for source in ("scigen", "wyformer")
        }
        counts: dict[str, object] = {}
        outputs: list[Path] = []
        statistics: dict[str, object] = {}
        for source in ("scigen", "wyformer"):
            computed_frame = pd.DataFrame(
                [{"material_id": material_id, **row} for material_id, row in computed[source]]
            )
            if (
                computed_frame["material_id"].astype(str).duplicated().any()
                or set(computed_frame["material_id"].astype(str))
                != set(discovery[source]["material_id"].astype(str))
            ):
                raise RuntimeError(f"NEXT295 {source} material identity differs")
            table = discovery[source].merge(
                computed_frame, on="material_id", how="left", validate="one_to_one"
            )
            supported = table["pcfc_supported"].fillna(False).astype(bool)
            values = np.column_stack(
                [
                    pd.to_numeric(table[name], errors="coerce").to_numpy(float)
                    for name in FEATURE_NAMES
                ]
            )
            finite = np.isfinite(values)
            sites = pd.to_numeric(table["pcfc_site_count"], errors="coerce")
            edges = pd.to_numeric(table["pcfc_edge_count"], errors="coerce")
            enclosed = pd.to_numeric(table["pcfc_enclosed_site_count"], errors="coerce")
            coverage = float(supported.mean())
            if (
                len(table) != EXPECTED_ROWS[source]
                or coverage < MINIMUM_FORMAL_COVERAGE
                or not finite[supported].all()
                or finite[~supported].any()
                or not ((values[supported] >= 0.0) & (values[supported] <= 1.0)).all()
                or not (sites[supported] >= 2).all()
                or not (edges[supported] > 0).all()
                or not (enclosed[supported] >= 0).all()
                or not (enclosed[supported] <= sites[supported]).all()
            ):
                raise RuntimeError(f"NEXT295 {source} support certificate differs")
            output = staging / FEATURE_FILES[source]
            table.to_parquet(output, index=False)
            outputs.append(output)
            failures = Counter(table.loc[~supported, "pcfc_failure"].astype(str))
            counts[source] = {
                "rows": int(len(table)),
                "supported": int(supported.sum()),
                "failures": int((~supported).sum()),
                "coverage": coverage,
                "failure_counts": dict(sorted(failures.items())),
                "finite_feature_counts": {
                    name: int(finite[:, index].sum())
                    for index, name in enumerate(FEATURE_NAMES)
                },
                "site_count": int(sites[supported].sum()),
                "edge_count": int(edges[supported].sum()),
                "enclosed_site_count": int(enclosed[supported].sum()),
            }
            statistics[source] = _label_free_statistics(table)
        catalogue = {
            "protocol": PROTOCOL,
            "feature_names": list(FEATURE_NAMES),
            "feature_directions": FEATURE_DIRECTIONS,
            "feature_count": len(FEATURE_NAMES),
            "metric_names": list(METRIC_NAMES),
            "statistic_names": list(STATISTIC_NAMES),
            "graph_mode": "crystalnn",
            "valence_protocol": n19.__name__,
            "equilibrium_program": "nonnegative_normalized_contact_balance_with_prior_floor",
            "rank_relative_tolerance": RANK_RELATIVE_TOLERANCE,
            "lp_residual_tolerance": LP_RESIDUAL_TOLERANCE,
            "enclosure_tolerance": ENCLOSURE_TOLERANCE,
            "quantile_method": "inverted_cdf",
            "output_grid": OUTPUT_GRID,
            "minimum_formal_coverage": MINIMUM_FORMAL_COVERAGE,
            "label_free_statistics": statistics,
            "source_partitions_read": {"scigen": ["discovery"], "wyformer": ["discovery"]},
            "endpoint_columns_present": False,
            "labels_opened": False,
        }
        catalogue_path = staging / CATALOGUE_NAME
        _write_json(catalogue_path, catalogue)
        outputs.append(catalogue_path)
        manifest = {
            "protocol": PROTOCOL,
            "mode": "physically_isolated_discovery_x0_positive_contact_force_closure_freeze",
            "workers": workers,
            "elapsed_seconds": time.perf_counter() - started,
            "counts": counts,
            "next296_audit_authorized": True,
            "source_partitions_read": {"scigen": ["discovery"], "wyformer": ["discovery"]},
            "labels_opened": False,
            "endpoint_fields_read": False,
            "internal_validation_geometry_opened": False,
            "internal_replication_geometry_opened": False,
            **BOUNDARY_FLAGS,
            "scientific_improvement_claim": False,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": executed_hashes,
            "outputs_sha256": {path.name: _sha256_file(path) for path in outputs},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT295 input changed before publication")
        if (
            _sha256_file(source_path)
            != executed_hashes["src/next295_positive_contact_force_closure.py"]
            or _sha256_file(test_path)
            != executed_hashes["tests/test_next295_positive_contact_force_closure.py"]
        ):
            raise RuntimeError("NEXT295 executed artifact changed before publication")
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
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = build_cross_source_pcfc_features(
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
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "METRIC_NAMES",
    "PROTOCOL",
    "build_cross_source_pcfc_features",
    "compute_pcfc_features",
    "positive_contact_force_closure_features",
    "positive_equilibrium_fraction",
    "site_pcfc_metrics",
]
