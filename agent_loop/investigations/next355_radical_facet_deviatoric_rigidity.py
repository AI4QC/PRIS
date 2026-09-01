#!/usr/bin/env python3
"""Deviatoric rigidity of the raw radius-weighted radical-facet graph."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ase import Atoms
import numpy as np

import src.next267_periodic_radical_voronoi_packing as n267
import src.next295_positive_contact_force_closure as n295
import src.next339_periodic_geometric_homogenized_transmissivity as n339
from src.next351_periodic_deviatoric_strain_rigidity import (
    periodic_deviatoric_strain_retention,
)
import src.next351_periodic_deviatoric_strain_rigidity as n351


PROTOCOL = "2026-08-13-next355-radical-facet-deviatoric-rigidity-v1"
DESIGN_SHA256 = "cd86db09780a28eb4ddbc993837a46ab9f6852c9bcf35c6bdd719752c6d59059"
FEATURE_NAMES = ("rfdr_deviatoric_retention_floor",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 10_000_000_000
RECIPROCAL_AREA_RELATIVE_TOLERANCE = n339.RECIPROCAL_AREA_RELATIVE_TOLERANCE
VOLUME_TILING_RELATIVE_TOLERANCE = n267.VOLUME_TILING_RELATIVE_TOLERANCE
BOUNDARY_FLAGS = dict(n295.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class RFDRFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    edge_count: int
    minimum_facet_area: float
    maximum_reciprocal_area_relative_error: float
    maximum_orthogonality_residual: float
    volume_tiling_relative_error: float
    affine_minimum_eigenvalue: float
    features: dict[str, float]


def _failure(exc: Exception | str) -> RFDRFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return RFDRFeatureResult(
        False, reason, 0, 0, math.nan, math.nan, math.nan, math.nan, math.nan, {}
    )


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def compute_rfdr_features(atoms: Atoms) -> RFDRFeatureResult:
    """Compute RFDR from elements and one raw unrelaxed periodic geometry."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        radii = np.asarray(
            [n267._tabulated_radius(str(symbol)) for symbol in work.get_chemical_symbols()],
            dtype=float,
        )
        graph = n339.periodic_power_facet_graph(work, radii=radii)
        result = periodic_deviatoric_strain_retention(
            n_sites=graph.site_count,
            endpoints=graph.endpoints,
            displacements=graph.displacements,
            weights=graph.conductances,
        )
        value = _quantize(result.retention_floor)
        if not 0.0 <= value <= 1.0:
            raise RuntimeError("NEXT355 feature domain differs")
        return RFDRFeatureResult(
            True,
            None,
            graph.site_count,
            len(graph.endpoints),
            graph.minimum_facet_area,
            graph.maximum_reciprocal_area_relative_error,
            result.maximum_orthogonality_residual,
            graph.volume_tiling_relative_error,
            float(np.linalg.eigvalsh(result.affine_gram)[0]),
            {FEATURE_NAMES[0]: value},
        )
    except Exception as exc:
        return _failure(exc)


def compute_rfdr_row(atoms: Atoms) -> dict[str, object]:
    result = compute_rfdr_features(atoms)
    return {
        FEATURE_NAMES[0]: result.features.get(FEATURE_NAMES[0], math.nan),
        "rfdr_supported": bool(result.supported),
        "rfdr_failure": result.failure_reason,
        "rfdr_site_count": int(result.site_count),
        "rfdr_edge_count": int(result.edge_count),
        "rfdr_minimum_facet_area": result.minimum_facet_area,
        "rfdr_maximum_reciprocal_area_relative_error": (
            result.maximum_reciprocal_area_relative_error
        ),
        "rfdr_maximum_orthogonality_residual": result.maximum_orthogonality_residual,
        "rfdr_volume_tiling_relative_error": result.volume_tiling_relative_error,
        "rfdr_affine_minimum_eigenvalue": result.affine_minimum_eigenvalue,
    }


__all__ = [
    "BOUNDARY_FLAGS",
    "DESIGN_SHA256",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "PROTOCOL",
    "RECIPROCAL_AREA_RELATIVE_TOLERANCE",
    "RFDRFeatureResult",
    "VOLUME_TILING_RELATIVE_TOLERANCE",
    "compute_rfdr_features",
    "compute_rfdr_row",
    "periodic_deviatoric_strain_retention",
]
