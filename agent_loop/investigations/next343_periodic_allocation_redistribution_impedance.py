#!/usr/bin/env python3
"""Global impedance of radius-target allocation on raw power-facet graphs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from ase import Atoms
import numpy as np

import src.next267_periodic_radical_voronoi_packing as n267
import src.next331_radical_facet_minimum_participation as n331
import src.next339_periodic_geometric_homogenized_transmissivity as n339


PROTOCOL = "2026-08-13-next343-periodic-allocation-redistribution-impedance-v1"
FEATURE_NAMES = ("pari_allocation_redistribution_protection",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 10_000_000_000
ALLOCATION_TOLERANCE = 1.0e-12
POISSON_RESIDUAL_TOLERANCE = 1.0e-8
ENERGY_TOLERANCE = 1.0e-12
BOUNDARY_FLAGS = dict(n331.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class AllocationImpedanceResult:
    protection: float
    impedance_ratio: float
    global_energy: float
    local_energy: float
    allocation_total_variation: float
    maximum_poisson_residual: float


@dataclass(frozen=True)
class PARIFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    edge_count: int
    allocation_total_variation: float
    impedance_ratio: float
    maximum_poisson_residual: float
    volume_tiling_relative_error: float
    features: dict[str, float]


def _failure(exc: Exception | str) -> PARIFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return PARIFeatureResult(False, reason, 0, 0, math.nan, math.nan, math.nan, math.nan, {})


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def allocation_redistribution_protection(
    *,
    observed: object,
    target: object,
    endpoints: object,
    conductances: object,
) -> AllocationImpedanceResult:
    """Solve the frozen graph Poisson problem for a zero-sum allocation source."""

    actual = np.asarray(observed, dtype=float)
    desired = np.asarray(target, dtype=float)
    if (
        actual.ndim != 1
        or desired.shape != actual.shape
        or len(actual) < 1
        or not np.isfinite(actual).all()
        or not np.isfinite(desired).all()
        or np.any(actual < 0.0)
        or np.any(desired < 0.0)
        or not math.isclose(math.fsum(actual.tolist()), 1.0, abs_tol=ALLOCATION_TOLERANCE)
        or not math.isclose(math.fsum(desired.tolist()), 1.0, abs_tol=ALLOCATION_TOLERANCE)
    ):
        raise ValueError("NEXT343 allocation population differs")
    pair_raw = np.asarray(endpoints)
    conductance = np.asarray(conductances, dtype=float)
    if pair_raw.ndim != 2 or pair_raw.shape[1:] != (2,) or len(pair_raw) < 1:
        raise ValueError("NEXT343 edge population differs")
    try:
        pair_numeric = np.asarray(pair_raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("NEXT343 edge population differs") from exc
    if (
        not np.isfinite(pair_numeric).all()
        or not np.equal(pair_numeric, np.rint(pair_numeric)).all()
    ):
        raise ValueError("NEXT343 edge population differs")
    pair = pair_numeric.astype(int)
    if (
        conductance.shape != (len(pair),)
        or not np.isfinite(conductance).all()
        or np.any(conductance <= 0.0)
        or np.any(pair < 0)
        or np.any(pair >= len(actual))
    ):
        raise ValueError("NEXT343 edge population differs")

    source = actual - desired
    total_variation = 0.5 * math.fsum(np.abs(source).tolist())
    if float(np.max(np.abs(source))) <= ALLOCATION_TOLERANCE:
        return AllocationImpedanceResult(1.0, 0.0, 0.0, 0.0, total_variation, 0.0)

    incidence = np.zeros((len(pair), len(actual)), dtype=float)
    rows = np.arange(len(pair), dtype=int)
    nonself = pair[:, 0] != pair[:, 1]
    incidence[rows[nonself], pair[nonself, 0]] = -1.0
    incidence[rows[nonself], pair[nonself, 1]] = 1.0
    laplacian = incidence.T @ (conductance[:, None] * incidence)
    degree = np.diag(laplacian)
    if np.any((np.abs(source) > ALLOCATION_TOLERANCE) & (degree <= 0.0)):
        raise ValueError("NEXT343 nonzero allocation source has zero graph degree")
    potential = np.linalg.pinv(laplacian, rcond=1.0e-12, hermitian=True) @ source
    residual = laplacian @ potential - source
    residual_scale = max(1.0, float(np.max(np.abs(source))))
    maximum_residual = float(np.max(np.abs(residual)) / residual_scale)
    if not math.isfinite(maximum_residual) or maximum_residual > POISSON_RESIDUAL_TOLERANCE:
        raise ValueError(f"NEXT343 Poisson residual differs: {maximum_residual:.12g}")
    global_energy = float(source @ potential)
    local_energy = math.fsum(
        float(source[index]) ** 2 / float(degree[index])
        for index in range(len(source))
        if abs(float(source[index])) > ALLOCATION_TOLERANCE
    )
    energy_scale = max(1.0, abs(global_energy), abs(local_energy))
    if (
        not math.isfinite(global_energy)
        or not math.isfinite(local_energy)
        or global_energy < -ENERGY_TOLERANCE * energy_scale
        or local_energy <= 0.0
    ):
        raise ValueError("NEXT343 redistribution energy differs")
    global_energy = max(0.0, global_energy)
    ratio = global_energy / local_energy
    protection = 1.0 / (1.0 + ratio)
    if (
        not math.isfinite(ratio)
        or ratio < 0.0
        or not math.isfinite(protection)
        or protection <= 0.0
        or protection > 1.0
    ):
        raise ValueError("NEXT343 redistribution impedance differs")
    return AllocationImpedanceResult(
        protection,
        ratio,
        global_energy,
        local_energy,
        total_variation,
        maximum_residual,
    )


def compute_pari_features(atoms: Atoms) -> PARIFeatureResult:
    try:
        work = n331._geometry_only_atoms(atoms)
        radii = np.asarray(
            [n267._tabulated_radius(str(symbol)) for symbol in work.get_chemical_symbols()],
            dtype=float,
        )
        graph = n339.periodic_power_facet_graph(work, radii=radii)
        cells = n267.periodic_radical_cells(work, radii=radii)
        if len(cells) != len(work) or any(cell.empty for cell in cells):
            raise ValueError("NEXT343 requires every labelled power cell to be nonempty")
        volume = abs(float(np.linalg.det(np.asarray(work.cell.array, dtype=float))))
        cell_volumes = np.asarray([cell.volume for cell in cells], dtype=float)
        tiling_error = abs(math.fsum(cell_volumes.tolist()) - volume) / volume
        if tiling_error > n267.VOLUME_TILING_RELATIVE_TOLERANCE:
            raise ValueError(f"NEXT343 volume-tiling certificate differs: {tiling_error:.12g}")
        observed = cell_volumes / volume
        target = radii**3 / math.fsum((radii**3).tolist())
        result = allocation_redistribution_protection(
            observed=observed,
            target=target,
            endpoints=graph.endpoints,
            conductances=graph.conductances,
        )
        value = _quantize(result.protection)
        if value <= 0.0:
            raise ValueError("NEXT343 positive protection quantized to zero")
        return PARIFeatureResult(
            True,
            None,
            len(work),
            len(graph.endpoints),
            result.allocation_total_variation,
            result.impedance_ratio,
            result.maximum_poisson_residual,
            max(tiling_error, graph.volume_tiling_relative_error),
            {FEATURE_NAMES[0]: value},
        )
    except Exception as exc:
        return _failure(exc)


def compute_pari_row(atoms: Atoms) -> dict[str, object]:
    result = compute_pari_features(atoms)
    row: dict[str, object] = {FEATURE_NAMES[0]: math.nan}
    row.update(
        {
            "pari_supported": bool(result.supported),
            "pari_failure": result.failure_reason,
            "pari_site_count": result.site_count,
            "pari_edge_count": result.edge_count,
            "pari_allocation_total_variation": result.allocation_total_variation,
            "pari_impedance_ratio": result.impedance_ratio,
            "pari_maximum_poisson_residual": result.maximum_poisson_residual,
            "pari_volume_tiling_relative_error": result.volume_tiling_relative_error,
        }
    )
    if result.supported:
        row[FEATURE_NAMES[0]] = float(result.features[FEATURE_NAMES[0]])
    return row


def build_cross_source_pari_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    probe_result_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    raise NotImplementedError("NEXT343 formal builder awaits a passing frozen probe")


__all__ = [
    "BOUNDARY_FLAGS",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "POISSON_RESIDUAL_TOLERANCE",
    "PROTOCOL",
    "AllocationImpedanceResult",
    "PARIFeatureResult",
    "allocation_redistribution_protection",
    "build_cross_source_pari_features",
    "compute_pari_features",
    "compute_pari_row",
]
