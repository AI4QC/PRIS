#!/usr/bin/env python3
"""Periodic bond-valence Hodge loop features from raw geometry only."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

from ase import Atoms
import numpy as np
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor

import src.next19_valence_transport as n19
import src.next267_periodic_radical_voronoi_packing as n267
import src.next295_positive_contact_force_closure as n295
import src.next38_bond_valence_transport_compatibility_features as n38
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next307-periodic-bond-valence-hodge-loop-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT307_PBVHL_FEATURE_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "next307_scigen_periodic_bond_valence_hodge_loop_features.parquet",
    "wyformer": "next307_wyformer_periodic_bond_valence_hodge_loop_features.parquet",
}
FEATURE_NAMES = (
    "pbvhl_cycle_fraction",
    "pbvhl_cycle_rms",
    "pbvhl_cycle_q90",
    "pbvhl_site_rms_q90",
)
FEATURE_DIRECTIONS = {name: "protected_low" for name in FEATURE_NAMES}
EXPECTED_ROWS = {"scigen": 13_470, "wyformer": 5_232}
MINIMUM_FORMAL_COVERAGE = 0.95
EXPECTED_DESIGN_SHA256 = (
    "94059fdd017bbf6ee61e249724ebb22ba141894f91fdd521e1b54f9f09ed6702"
)
EXPECTED_INPUT_SHA256 = {
    "design": EXPECTED_DESIGN_SHA256,
    "next19_source": "f1195a7ef519827f8da1704b9abe773bcee105eff1bdf6dfd5b8eabba1b94712",
    "next38_source": "597b584055cacc2f4d1f24051aab09bb8a2fcbd7adee964c5cb983971a657ef6",
    "next267_source": "8f1e7ed9eb73a81a5755d455ffc05aab6f539cbd66afbbbfc384ca88391adca1",
    "next295_source": "4b92811e7f3c7ac60c1506104a18d2bd9d0fe06c6202e7f34cb996b32cd649a3",
    "scigen_manifest": "dc5bf33c6ce6dc2c10bcd3704688055058145fbe7269ada23ffbe4b141d75fe7",
    "scigen_metadata": "f91455f23b0a96f60fd1c779249e2be46a7ecf94fcdde2b146426a95aac05bde",
    "scigen_discovery_geometry": "e561ef12343c66dcc72bcabf6b8719ad727e01c9582a094e281da73b862ab575",
    "wyformer_manifest": "e0539d556538cb4c052431bc6a1e5c1663bc3de427677dbc8a446dcc3b4fbc54",
    "wyformer_metadata": "3b152b4b84c8d3f7ff5e85611dc1fd2728296f150e907ac4578ce55d2b27dd2b",
    "wyformer_discovery_geometry": "f1ce5ae4fba8c13fcbf3e25de4f596b919d9b41da5b072d9a28eefeaffc69784",
}


@dataclass(frozen=True)
class PBVHLFeatureResult:
    """Fail-open result for one periodic bond-valence loop projection."""

    supported: bool
    failure_reason: str | None
    site_count: int
    edge_count: int
    incidence_rank: int
    cycle_dimension: int
    loop_divergence_max: float
    valence_policy: str | None
    parameter_exact_fraction: float
    parameter_generic_fraction: float
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> PBVHLFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return PBVHLFeatureResult(
        False,
        reason,
        0,
        0,
        0,
        0,
        math.nan,
        None,
        math.nan,
        math.nan,
        {},
    )


def _inverted_cdf(values: object, quantile: float) -> float:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("PBVHL quantile population differs")
    ordered = np.sort(array, kind="stable")
    index = max(0, int(math.ceil(float(quantile) * len(ordered))) - 1)
    return float(ordered[min(index, len(ordered) - 1)])


def bond_valence_hodge_loop_features(
    *,
    n_sites: int,
    endpoints: object,
    bond_valences: object,
) -> PBVHLFeatureResult:
    """Project a positive oriented edge field onto an incidence cycle space."""

    try:
        if type(n_sites) is not int or n_sites < 2:
            raise ValueError("PBVHL site count differs")
        raw_pair = np.asarray(endpoints)
        if (
            raw_pair.ndim != 2
            or raw_pair.shape[1:] != (2,)
            or len(raw_pair) < 1
            or not np.isfinite(raw_pair.astype(float)).all()
            or not np.equal(raw_pair.astype(float), np.rint(raw_pair.astype(float))).all()
        ):
            raise ValueError("PBVHL endpoint population differs")
        pair = raw_pair.astype(int)
        values = np.asarray(bond_valences, dtype=float)
        if (
            values.shape != (len(pair),)
            or not np.isfinite(values).all()
            or np.any(values <= 0.0)
        ):
            raise ValueError("PBVHL bond-valence population differs")
        if (
            np.any(pair < 0)
            or np.any(pair >= n_sites)
            or np.any(pair[:, 0] == pair[:, 1])
        ):
            raise ValueError("PBVHL endpoint indices differ")

        incidence = np.zeros((n_sites, len(pair)), dtype=float)
        columns = np.arange(len(pair))
        incidence[pair[:, 0], columns] = 1.0
        incidence[pair[:, 1], columns] = -1.0
        _left, singular, right = np.linalg.svd(incidence, full_matrices=False)
        if not len(singular) or float(singular[0]) <= 0.0:
            raise ValueError("PBVHL incidence spectrum differs")
        tolerance = (
            np.finfo(float).eps * max(incidence.shape) * float(singular[0])
        )
        rank = int(np.sum(singular > tolerance))
        cycle_dimension = int(len(pair) - rank)
        if rank <= 0 or cycle_dimension <= 0:
            raise ValueError("PBVHL graph has no positive cycle dimension")
        row_basis = right[:rank]
        gradient = row_basis.T @ (row_basis @ values)
        loop = values - gradient
        divergence = incidence @ loop
        divergence_max = float(np.max(np.abs(divergence)))
        divergence_tolerance = 1.0e-10 * max(
            1.0, float(np.linalg.norm(values)), float(np.linalg.norm(loop))
        )
        if not math.isfinite(divergence_max) or divergence_max > divergence_tolerance:
            raise RuntimeError("PBVHL loop projection is not divergence free")

        mean_valence = float(np.mean(values))
        valence_norm = float(np.linalg.norm(values))
        loop_norm = float(np.linalg.norm(loop))
        if not mean_valence > 0.0 or not valence_norm > 0.0:
            raise RuntimeError("PBVHL normalization differs")
        normalized = np.abs(loop) / mean_valence
        site_rms: list[float] = []
        for site in range(n_sites):
            selected = (pair[:, 0] == site) | (pair[:, 1] == site)
            if not selected.any():
                raise ValueError("PBVHL graph contains an isolated site")
            site_rms.append(float(np.sqrt(np.mean(normalized[selected] ** 2))))
        cycle_fraction = float(loop_norm / valence_norm)
        if cycle_fraction < 0.0 or cycle_fraction > 1.0 + 1.0e-12:
            raise RuntimeError("PBVHL cycle fraction differs")
        features = {
            "pbvhl_cycle_fraction": float(np.clip(cycle_fraction, 0.0, 1.0)),
            "pbvhl_cycle_rms": float(np.sqrt(np.mean(normalized**2))),
            "pbvhl_cycle_q90": _inverted_cdf(normalized, 0.90),
            "pbvhl_site_rms_q90": _inverted_cdf(site_rms, 0.90),
        }
        feature_values = np.asarray(list(features.values()), dtype=float)
        if (
            tuple(features) != FEATURE_NAMES
            or not np.isfinite(feature_values).all()
            or np.any(feature_values < 0.0)
        ):
            raise RuntimeError("PBVHL feature schema differs")
        return PBVHLFeatureResult(
            True,
            None,
            n_sites,
            len(pair),
            rank,
            cycle_dimension,
            divergence_max,
            None,
            math.nan,
            math.nan,
            features,
        )
    except Exception as exc:
        return _failure(exc)


def _resolved_bond_valence_field(
    structure, charges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    geometry = n19.build_periodic_edge_geometry(
        structure, charges, graph_mode="voronoi"
    )
    if not geometry.supported:
        raise ValueError(geometry.failure_reason or "PBVHL periodic graph failed")
    parameters = n38.bv_table()
    endpoints: list[tuple[int, int]] = []
    values: list[float] = []
    sources: list[str] = []
    for edge in geometry.edges:
        cation = int(edge.cation)
        anion = int(edge.anion)
        key = (
            structure[cation].specie.symbol,
            int(round(float(charges[cation]))),
            structure[anion].specie.symbol,
            int(round(float(charges[anion]))),
        )
        resolved = n38.resolve_bond_valence_parameter(
            key, parameters, policy="frozen-fallback"
        )
        if resolved is None:
            cation_radius = n38._tabulated_radius(structure[cation].specie.symbol)
            anion_radius = n38._tabulated_radius(structure[anion].specie.symbol)
            if cation_radius is None or anion_radius is None:
                raise ValueError("PBVHL bond-valence and radius parameters are missing")
            resolved = (cation_radius + anion_radius, 0.37, "radius_generic")
        r0, decay, source = resolved
        if (
            not math.isfinite(float(r0))
            or not math.isfinite(float(decay))
            or float(decay) <= 0.0
            or str(source) not in n38.PARAMETER_SOURCES
        ):
            raise ValueError("PBVHL bond-valence parameter differs")
        displacement_fractional = (
            np.asarray(structure[anion].frac_coords, dtype=float)
            + np.asarray(edge.image, dtype=float)
            - np.asarray(structure[cation].frac_coords, dtype=float)
        )
        displacement = np.asarray(
            structure.lattice.get_cartesian_coords(displacement_fractional),
            dtype=float,
        )
        distance = float(np.linalg.norm(displacement))
        if not math.isfinite(distance) or distance <= 0.0:
            raise ValueError("PBVHL edge distance differs")
        try:
            bond_valence = math.exp((float(r0) - distance) / float(decay))
        except OverflowError as exc:
            raise ValueError("PBVHL bond valence overflowed") from exc
        if not math.isfinite(bond_valence) or bond_valence <= 0.0:
            raise ValueError("PBVHL bond valence differs")
        endpoints.append((cation, anion))
        values.append(bond_valence)
        sources.append(str(source))
    return (
        np.asarray(endpoints, dtype=int),
        np.asarray(values, dtype=float),
        tuple(sources),
    )


def compute_pbvhl_features(atoms: Atoms) -> PBVHLFeatureResult:
    """Compute PBVHL from elements and one raw unrelaxed periodic geometry."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "PBVHL valence assignment failed"
            )
        charges = np.asarray(assignment.values, dtype=float)
        endpoints, bond_valences, sources = _resolved_bond_valence_field(
            structure, charges
        )
        result = bond_valence_hodge_loop_features(
            n_sites=len(structure),
            endpoints=endpoints,
            bond_valences=bond_valences,
        )
        if not result.supported:
            return result
        source_array = np.asarray(sources, dtype=object)
        exact_fraction = float(np.mean(source_array == "exact"))
        generic_fraction = float(
            np.mean(np.isin(source_array, ("brown_generic", "radius_generic")))
        )
        return replace(
            result,
            valence_policy=str(assignment.policy),
            parameter_exact_fraction=exact_fraction,
            parameter_generic_fraction=generic_fraction,
        )
    except Exception as exc:
        return _failure(exc)


def compute_pbvhl_row(atoms: Atoms) -> dict[str, object]:
    result = compute_pbvhl_features(atoms)
    row: dict[str, object] = {name: math.nan for name in FEATURE_NAMES}
    row.update(
        {
            "pbvhl_supported": bool(result.supported),
            "pbvhl_failure": result.failure_reason,
            "pbvhl_site_count": result.site_count,
            "pbvhl_edge_count": result.edge_count,
            "pbvhl_incidence_rank": result.incidence_rank,
            "pbvhl_cycle_dimension": result.cycle_dimension,
            "pbvhl_cycle_dimension_fraction": (
                float(result.cycle_dimension / result.edge_count)
                if result.edge_count
                else math.nan
            ),
            "pbvhl_loop_divergence_max": result.loop_divergence_max,
            "pbvhl_valence_policy": result.valence_policy,
            "pbvhl_parameter_exact_fraction": result.parameter_exact_fraction,
            "pbvhl_parameter_generic_fraction": result.parameter_generic_fraction,
        }
    )
    if result.supported:
        row.update({name: float(result.features[name]) for name in FEATURE_NAMES})
    return row


def _error_row(exc: Exception) -> dict[str, object]:
    return {
        **{name: math.nan for name in FEATURE_NAMES},
        "pbvhl_supported": False,
        "pbvhl_failure": f"{type(exc).__name__}: {exc}",
        "pbvhl_site_count": 0,
        "pbvhl_edge_count": 0,
        "pbvhl_incidence_rank": 0,
        "pbvhl_cycle_dimension": 0,
        "pbvhl_cycle_dimension_fraction": math.nan,
        "pbvhl_loop_divergence_max": math.nan,
        "pbvhl_valence_policy": None,
        "pbvhl_parameter_exact_fraction": math.nan,
        "pbvhl_parameter_generic_fraction": math.nan,
    }


def _compute_scigen_payload(item: tuple[str, bytes]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        atoms = n267.n85._parse_frame(payload, strict_output=True).atoms
        return material_id, compute_pbvhl_row(atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_wyformer_payload(item: tuple[str, str]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        structure = n267.Structure.from_dict(json.loads(payload))
        atoms = n267.AseAtomsAdaptor.get_atoms(structure)
        return material_id, compute_pbvhl_row(atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_many(payloads, *, source: str, workers: int):
    worker = _compute_scigen_payload if source == "scigen" else _compute_wyformer_payload
    if workers == 1:
        return [worker(item) for item in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(worker, payloads, chunksize=8))


def _label_free_statistics(table: pd.DataFrame) -> dict[str, object]:
    supported = table["pbvhl_supported"].fillna(False).astype(bool)
    statistics: dict[str, object] = {}
    for name in FEATURE_NAMES:
        values = pd.to_numeric(table.loc[supported, name], errors="coerce").to_numpy(float)
        statistics[name] = {
            "unique_rounded_10": int(len(np.unique(np.round(values, 10)))),
            "exact_zero_fraction": float(np.mean(values < 1.0e-10)),
            "minimum": float(values.min()),
            "q10": _inverted_cdf(values, 0.10),
            "median": _inverted_cdf(values, 0.50),
            "q90": _inverted_cdf(values, 0.90),
            "maximum": float(values.max()),
        }
    return statistics


def build_cross_source_pbvhl_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build NEXT307 from physically isolated discovery geometry only."""

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
        "next38_source": Path(n38.__file__).resolve(),
        "next267_source": Path(n267.__file__).resolve(),
        "next295_source": Path(n295.__file__).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("NEXT307 workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT307 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT307 formal input identity differs: {differing}")

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
        raise ValueError("NEXT307 discovery geometry provenance differs")

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
            raise ValueError(f"NEXT307 {source} metadata differs")
        selected = frame.loc[frame["partition_role"].eq("discovery")].copy()
        selected = selected.sort_values("material_id", kind="mergesort").reset_index(drop=True)
        if len(selected) != EXPECTED_ROWS[source]:
            raise ValueError(f"NEXT307 {source} discovery identity differs")
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
    test_path = source_path.parent.parent / "tests/test_next307_periodic_bond_valence_hodge_loop.py"
    executed_hashes = {
        "src/next307_periodic_bond_valence_hodge_loop.py": _sha256_file(source_path),
        "tests/test_next307_periodic_bond_valence_hodge_loop.py": _sha256_file(test_path),
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
                raise RuntimeError(f"NEXT307 {source} material identity differs")
            table = discovery[source].merge(
                computed_frame, on="material_id", how="left", validate="one_to_one"
            )
            supported = table["pbvhl_supported"].fillna(False).astype(bool)
            values = np.column_stack(
                [pd.to_numeric(table[name], errors="coerce").to_numpy(float) for name in FEATURE_NAMES]
            )
            finite = np.isfinite(values)
            sites = pd.to_numeric(table["pbvhl_site_count"], errors="coerce")
            edges = pd.to_numeric(table["pbvhl_edge_count"], errors="coerce")
            ranks = pd.to_numeric(table["pbvhl_incidence_rank"], errors="coerce")
            cycles = pd.to_numeric(table["pbvhl_cycle_dimension"], errors="coerce")
            divergence = pd.to_numeric(table["pbvhl_loop_divergence_max"], errors="coerce")
            exact = pd.to_numeric(table["pbvhl_parameter_exact_fraction"], errors="coerce")
            generic = pd.to_numeric(table["pbvhl_parameter_generic_fraction"], errors="coerce")
            coverage = float(supported.mean())
            if (
                len(table) != EXPECTED_ROWS[source]
                or coverage < MINIMUM_FORMAL_COVERAGE
                or not finite[supported].all()
                or finite[~supported].any()
                or not (values[supported] >= 0.0).all()
                or not (values[supported, 0] <= 1.0 + 1.0e-12).all()
                or not (sites[supported] >= 2).all()
                or not (edges[supported] > ranks[supported]).all()
                or not (cycles[supported] == edges[supported] - ranks[supported]).all()
                or not (cycles[supported] > 0).all()
                or not np.isfinite(divergence[supported]).all()
                or not (divergence[supported] <= 1.0e-8).all()
                or not ((exact[supported] >= 0.0) & (exact[supported] <= 1.0)).all()
                or not ((generic[supported] >= 0.0) & (generic[supported] <= 1.0)).all()
                or not (sites[~supported] == 0).all()
                or not (edges[~supported] == 0).all()
                or not (ranks[~supported] == 0).all()
                or not (cycles[~supported] == 0).all()
            ):
                raise RuntimeError(f"NEXT307 {source} support certificate differs")
            output = staging / FEATURE_FILES[source]
            table.to_parquet(output, index=False)
            outputs.append(output)
            failures = Counter(table.loc[~supported, "pbvhl_failure"].astype(str))
            statistics[source] = _label_free_statistics(table)
            if any(
                statistics[source][name]["unique_rounded_10"] < 2
                for name in FEATURE_NAMES
            ):
                raise RuntimeError(f"NEXT307 {source} feature degeneracy differs")
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
                "minimum_cycle_dimension": int(cycles[supported].min()),
                "maximum_cycle_dimension": int(cycles[supported].max()),
                "maximum_loop_divergence": float(divergence[supported].max()),
            }

        catalogue = {
            "protocol": PROTOCOL,
            "feature_names": list(FEATURE_NAMES),
            "feature_directions": FEATURE_DIRECTIONS,
            "feature_count": len(FEATURE_NAMES),
            "graph": "NEXT19 opposite-sign periodic Voronoi multigraph",
            "edge_field": "exp((R0-distance)/B) under NEXT38 frozen parameter policy",
            "incidence_orientation": "cation_plus_anion_minus",
            "rank_tolerance": "eps_times_max_shape_times_sigma_max",
            "loop_projection": "bond_valence_minus_row_space_projection",
            "quantile_method": "inverted_cdf",
            "quantile": 0.90,
            "directions_frozen_before_outcome": True,
            "label_free_statistics": statistics,
            "next308_audit_authorized": True,
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "model_or_proxy_potential_used": False,
            "physical_relaxation_executed": False,
        }
        catalogue_path = staging / CATALOGUE_NAME
        _write_json(catalogue_path, catalogue)
        outputs.append(catalogue_path)
        manifest = {
            "protocol": PROTOCOL,
            "mode": "physically_isolated_discovery_x0_periodic_bond_valence_hodge_loop_freeze",
            "workers": workers,
            "elapsed_seconds": float(time.perf_counter() - started),
            "counts": counts,
            "source_partitions_read": {"scigen": ["discovery"], "wyformer": ["discovery"]},
            "labels_opened": False,
            "endpoint_fields_read": False,
            "internal_validation_geometry_opened": False,
            "internal_replication_geometry_opened": False,
            "opened_validation_outputs_used": False,
            "scigen_replication_endpoint_opened": False,
            "wyformer_replication_endpoint_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "model_or_proxy_potential_used": False,
            "physical_relaxation_executed": False,
            "scientific_improvement_claim": False,
            "next308_audit_authorized": True,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": executed_hashes,
            "outputs_sha256": {path.name: _sha256_file(path) for path in outputs},
        }
        manifest_path = staging / MANIFEST_NAME
        _write_json(manifest_path, manifest)
        if (
            _sha256_file(source_path)
            != executed_hashes["src/next307_periodic_bond_valence_hodge_loop.py"]
            or _sha256_file(test_path)
            != executed_hashes["tests/test_next307_periodic_bond_valence_hodge_loop.py"]
        ):
            raise RuntimeError("NEXT307 executed artifact changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scigen-cohort-dir", type=Path, required=True)
    parser.add_argument("--wyformer-cohort-dir", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    arguments = parser.parse_args(argv)
    manifest = build_cross_source_pbvhl_features(
        scigen_cohort_dir=arguments.scigen_cohort_dir,
        wyformer_cohort_dir=arguments.wyformer_cohort_dir,
        design_path=arguments.design_path,
        output_dir=arguments.output_dir,
        workers=arguments.workers,
        require_formal_inputs=not arguments.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CATALOGUE_NAME",
    "EXPECTED_DESIGN_SHA256",
    "FEATURE_DIRECTIONS",
    "FEATURE_FILES",
    "FEATURE_NAMES",
    "MANIFEST_NAME",
    "MINIMUM_FORMAL_COVERAGE",
    "PBVHLFeatureResult",
    "PROTOCOL",
    "bond_valence_hodge_loop_features",
    "build_cross_source_pbvhl_features",
    "compute_pbvhl_features",
]
