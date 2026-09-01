#!/usr/bin/env python3
"""Local positive enclosure of unique radical-cell facet directions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ase import Atoms
import numpy as np

import src.next267_periodic_radical_voronoi_packing as n267
import src.next279_radical_packing_autocorrelation as n279
import src.next295_positive_contact_force_closure as n295


PROTOCOL = "2026-08-13-next327-radical-facet-positive-enclosure-v1"
FEATURE_NAMES = ("rfpe_uniform_equilibrium_q10",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 10_000_000_000
DIRECTION_NORM_TOLERANCE = 1.0e-12
BOUNDARY_FLAGS = {
    "dft_calculation_executed": False,
    "dft_values_used_by_executable_formula": False,
    "learned_energy_force_stress_proxy_used": False,
    "model_or_proxy_potential_used": False,
    "physical_relaxation_executed": False,
    "opened_validation_outputs_used": False,
    "scigen_replication_endpoint_opened": False,
    "wyformer_replication_endpoint_opened": False,
}


@dataclass(frozen=True)
class RFPEFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    directed_contact_count: int
    minimum_unique_facet_count: int
    maximum_unique_facet_count: int
    features: dict[str, float]


def _failure(exc: Exception | str) -> RFPEFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return RFPEFeatureResult(False, reason, 0, 0, 0, 0, {})


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def unique_facet_directions(vectors: object) -> np.ndarray:
    """Normalize and deterministically deduplicate coincident facet normals."""

    raw = np.asarray(vectors, dtype=float)
    if raw.ndim != 2 or raw.shape[1:] != (3,) or not np.isfinite(raw).all():
        raise ValueError("NEXT327 vectors differ")
    if len(raw) == 0:
        return np.empty((0, 3), dtype=float)
    norms = np.linalg.norm(raw, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= DIRECTION_NORM_TOLERANCE):
        raise ValueError("NEXT327 vectors differ")
    direction = raw / norms[:, None]
    keys = np.rint(direction * OUTPUT_GRID).astype(np.int64)
    retained: dict[tuple[int, int, int], np.ndarray] = {}
    for key, value in zip(keys, direction, strict=True):
        retained.setdefault(tuple(int(item) for item in key), value)
    return np.asarray([retained[key] for key in sorted(retained)], dtype=float)


def radical_facet_positive_enclosure_features(
    *,
    n_sites: int,
    centers: Sequence[int] | np.ndarray,
    vectors: Sequence[Sequence[float]] | np.ndarray,
) -> dict[str, float]:
    """Return the frozen q10 of per-site uniform positive-enclosure margins."""

    if not isinstance(n_sites, (int, np.integer)) or int(n_sites) < 1:
        raise ValueError("NEXT327 n_sites differs")
    count = int(n_sites)
    raw_center = np.asarray(centers)
    if raw_center.ndim != 1:
        raise ValueError("NEXT327 centers differ")
    try:
        numeric_center = np.asarray(raw_center, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("NEXT327 centers differ") from exc
    if (
        not np.isfinite(numeric_center).all()
        or not np.equal(numeric_center, np.rint(numeric_center)).all()
    ):
        raise ValueError("NEXT327 centers differ")
    center = numeric_center.astype(int)
    vector = np.asarray(vectors, dtype=float)
    if vector.ndim != 2 or vector.shape[1:] != (3,) or len(vector) != len(center):
        raise ValueError("NEXT327 vectors differ")
    if np.any(center < 0) or np.any(center >= count):
        raise ValueError("NEXT327 centers differ")
    if not np.isfinite(vector).all():
        raise ValueError("NEXT327 vectors differ")
    norms = np.linalg.norm(vector, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= DIRECTION_NORM_TOLERANCE):
        raise ValueError("NEXT327 vectors differ")

    margins = np.zeros(count, dtype=float)
    for site in range(count):
        directions = unique_facet_directions(vector[center == site])
        if len(directions) == 0:
            margins[site] = 0.0
            continue
        prior = np.full(len(directions), 1.0 / len(directions), dtype=float)
        margins[site] = n295.positive_equilibrium_fraction(directions, prior)
    q10 = float(np.quantile(margins, 0.10, method="inverted_cdf"))
    value = _quantize(q10)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise RuntimeError("NEXT327 feature bounds differ")
    return {FEATURE_NAMES[0]: value}


def _geometry_only_atoms(atoms: Atoms) -> Atoms:
    try:
        pbc = np.asarray(atoms.pbc, dtype=bool)
        cell = np.asarray(atoms.cell.array, dtype=float)
        positions = np.asarray(atoms.positions, dtype=float)
        numbers = np.asarray(atoms.numbers)
    except Exception as exc:
        raise ValueError("NEXT327 features require exact periodic geometry-only Atoms") from exc
    if (
        not isinstance(atoms, Atoms)
        or len(atoms) < 1
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
        raise ValueError("NEXT327 features require exact periodic geometry-only Atoms")
    return atoms.copy()


def compute_rfpe_features(atoms: Atoms) -> RFPEFeatureResult:
    """Compute RFPE from element identities and one raw periodic geometry."""

    try:
        work = _geometry_only_atoms(atoms)
        radii = np.asarray(
            [n267._tabulated_radius(str(symbol)) for symbol in work.get_chemical_symbols()],
            dtype=float,
        )
        cells, contacts = n279.periodic_radical_cells_and_contacts(work, radii=radii)
        if len(cells) != len(work) or any(cell.empty for cell in cells):
            raise ValueError("NEXT327 graph requires every labelled power cell to be nonempty")
        if not contacts:
            raise ValueError("NEXT327 graph has no active-facet contact incidence")
        if not n279.contacts_are_reciprocal(contacts):
            raise ValueError("NEXT327 active-facet contacts are not reciprocal")
        centers = np.asarray([contact.center for contact in contacts], dtype=int)
        vectors = np.asarray([contact.displacement for contact in contacts], dtype=float)
        counts = np.asarray(
            [len(unique_facet_directions(vectors[centers == site])) for site in range(len(work))],
            dtype=int,
        )
        if np.any(counts < 4):
            raise ValueError("NEXT327 site has fewer than four unique active facets")
        features = radical_facet_positive_enclosure_features(
            n_sites=len(work), centers=centers, vectors=vectors
        )
        return RFPEFeatureResult(
            True,
            None,
            len(work),
            len(contacts),
            int(counts.min()),
            int(counts.max()),
            features,
        )
    except Exception as exc:
        return _failure(exc)


def compute_rfpe_row(atoms: Atoms) -> dict[str, object]:
    result = compute_rfpe_features(atoms)
    row: dict[str, object] = {FEATURE_NAMES[0]: math.nan}
    row.update(
        {
            "rfpe_supported": bool(result.supported),
            "rfpe_failure": result.failure_reason,
            "rfpe_site_count": result.site_count,
            "rfpe_directed_contact_count": result.directed_contact_count,
            "rfpe_minimum_unique_facet_count": result.minimum_unique_facet_count,
            "rfpe_maximum_unique_facet_count": result.maximum_unique_facet_count,
        }
    )
    if result.supported:
        row[FEATURE_NAMES[0]] = float(result.features[FEATURE_NAMES[0]])
    return row


def build_cross_source_rfpe_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    probe_result_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build the frozen label-free RFPE tables after probe authorization."""

    raise NotImplementedError("NEXT327 formal builder awaits a passing frozen probe")


__all__ = [
    "BOUNDARY_FLAGS",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "PROTOCOL",
    "RFPEFeatureResult",
    "build_cross_source_rfpe_features",
    "compute_rfpe_features",
    "compute_rfpe_row",
    "radical_facet_positive_enclosure_features",
    "unique_facet_directions",
]
