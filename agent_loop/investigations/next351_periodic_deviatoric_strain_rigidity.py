#!/usr/bin/env python3
"""Periodic deviatoric strain rigidity from one raw geometry.

The descriptor is a kinematic projection on a frozen analytic contact graph.
It does not evaluate an energy, force, stress, learned model, or relaxation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ase import Atoms
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor

import src.next19_valence_transport as n19
import src.next267_periodic_radical_voronoi_packing as n267
import src.next295_positive_contact_force_closure as n295


PROTOCOL = "2026-08-13-next351-periodic-deviatoric-strain-rigidity-v1"
FEATURE_NAMES = ("pdsr_deviatoric_retention_floor",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 10_000_000_000
AFFINE_EIGENVALUE_RELATIVE_TOLERANCE = 1.0e-12
SPECTRUM_TOLERANCE = 1.0e-8
ORTHOGONALITY_TOLERANCE = 1.0e-8
BOUNDARY_FLAGS = dict(n295.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class DeviatoricRetentionResult:
    retention_floor: float
    generalized_eigenvalues: tuple[float, float, float, float, float]
    affine_gram: np.ndarray
    retained_gram: np.ndarray
    maximum_orthogonality_residual: float


@dataclass(frozen=True)
class PDSRFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    edge_count: int
    maximum_orthogonality_residual: float
    affine_minimum_eigenvalue: float
    features: dict[str, float]


def _failure(exc: Exception | str) -> PDSRFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return PDSRFeatureResult(False, reason, 0, 0, math.nan, math.nan, {})


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def deviatoric_strain_basis() -> np.ndarray:
    """Return a fixed Frobenius-orthonormal symmetric trace-free basis."""

    root2 = math.sqrt(2.0)
    basis = np.asarray(
        [
            [[1.0 / root2, 0.0, 0.0], [0.0, -1.0 / root2, 0.0], [0.0, 0.0, 0.0]],
            [
                [1.0 / math.sqrt(6.0), 0.0, 0.0],
                [0.0, 1.0 / math.sqrt(6.0), 0.0],
                [0.0, 0.0, -2.0 / math.sqrt(6.0)],
            ],
            [[0.0, 1.0 / root2, 0.0], [1.0 / root2, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0 / root2], [0.0, 0.0, 0.0], [1.0 / root2, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0 / root2], [0.0, 1.0 / root2, 0.0]],
        ],
        dtype=float,
    )
    gram = np.einsum("aij,bij->ab", basis, basis)
    if not np.allclose(gram, np.eye(5), rtol=0.0, atol=1.0e-15):
        raise RuntimeError("NEXT351 deviatoric basis is not orthonormal")
    if not np.allclose(np.trace(basis, axis1=1, axis2=2), 0.0, rtol=0.0, atol=1.0e-15):
        raise RuntimeError("NEXT351 deviatoric basis is not trace-free")
    return basis


def periodic_deviatoric_strain_retention(
    *,
    n_sites: int,
    endpoints: object,
    displacements: object,
    weights: object,
) -> DeviatoricRetentionResult:
    """Return the weakest internally unrelievable deviatoric strain fraction."""

    if type(n_sites) is not int or n_sites < 1:
        raise ValueError("NEXT351 site count differs")
    pair_raw = np.asarray(endpoints)
    vector = np.asarray(displacements, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if pair_raw.ndim != 2 or pair_raw.shape[1:] != (2,) or len(pair_raw) < 5:
        raise ValueError("NEXT351 edge population differs")
    try:
        pair_numeric = np.asarray(pair_raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("NEXT351 edge population differs") from exc
    if (
        not np.isfinite(pair_numeric).all()
        or not np.equal(pair_numeric, np.rint(pair_numeric)).all()
    ):
        raise ValueError("NEXT351 edge population differs")
    pair = pair_numeric.astype(int)
    if (
        vector.shape != (len(pair), 3)
        or weight.shape != (len(pair),)
        or not np.isfinite(vector).all()
        or not np.isfinite(weight).all()
        or np.any(weight <= 0.0)
        or np.any(pair < 0)
        or np.any(pair >= n_sites)
    ):
        raise ValueError("NEXT351 edge population differs")
    length = np.linalg.norm(vector, axis=1)
    if not np.isfinite(length).all() or np.any(length <= 1.0e-12):
        raise ValueError("NEXT351 edge population differs")

    direction = vector / length[:, None]
    root_weight = np.sqrt(weight)
    internal = np.zeros((len(pair), 3 * n_sites), dtype=float)
    derivative = root_weight[:, None] * direction / length[:, None]
    for edge, (left, right) in enumerate(pair):
        internal[edge, 3 * left : 3 * left + 3] -= derivative[edge]
        internal[edge, 3 * right : 3 * right + 3] += derivative[edge]
    basis = deviatoric_strain_basis()
    affine = root_weight[:, None] * np.einsum(
        "ei,aij,ej->ea", direction, basis, direction
    )
    affine_gram = affine.T @ affine
    affine_gram = 0.5 * (affine_gram + affine_gram.T)
    affine_values, affine_vectors = np.linalg.eigh(affine_gram)
    affine_scale = max(float(affine_values[-1]), np.finfo(float).tiny)
    if (
        not np.isfinite(affine_values).all()
        or float(affine_values[0])
        <= AFFINE_EIGENVALUE_RELATIVE_TOLERANCE * affine_scale
    ):
        raise ValueError("NEXT351 affine deviatoric Gram is not positive definite")

    correction, _, _, _ = np.linalg.lstsq(internal, affine, rcond=1.0e-12)
    retained = affine - internal @ correction
    numerator = float(np.linalg.norm(internal.T @ retained, ord="fro"))
    denominator = float(
        np.linalg.norm(internal, ord="fro") * np.linalg.norm(affine, ord="fro")
    )
    residual = 0.0 if denominator <= np.finfo(float).tiny else numerator / denominator
    if not math.isfinite(residual) or residual > ORTHOGONALITY_TOLERANCE:
        raise ValueError(f"NEXT351 projection orthogonality residual differs: {residual:.12g}")

    retained_gram = retained.T @ retained
    retained_gram = 0.5 * (retained_gram + retained_gram.T)
    inverse_root = affine_vectors @ np.diag(1.0 / np.sqrt(affine_values)) @ affine_vectors.T
    normalized = inverse_root @ retained_gram @ inverse_root
    normalized = 0.5 * (normalized + normalized.T)
    spectrum = np.linalg.eigvalsh(normalized)
    if (
        not np.isfinite(spectrum).all()
        or float(spectrum[0]) < -SPECTRUM_TOLERANCE
        or float(spectrum[-1]) > 1.0 + SPECTRUM_TOLERANCE
    ):
        raise ValueError("NEXT351 generalized spectrum differs")
    spectrum = np.clip(spectrum, 0.0, 1.0)
    return DeviatoricRetentionResult(
        float(spectrum[0]),
        tuple(float(value) for value in spectrum),
        affine_gram,
        retained_gram,
        residual,
    )


def compute_pdsr_features(atoms: Atoms) -> PDSRFeatureResult:
    """Compute PDSR from composition and one raw unrelaxed periodic geometry."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(assignment.failure_reason or "NEXT351 valence assignment failed")
        geometry = n19.build_periodic_edge_geometry(
            structure, assignment.values, graph_mode="voronoi"
        )
        if not geometry.supported:
            raise ValueError(geometry.failure_reason or "NEXT351 periodic graph failed")
        endpoints: list[tuple[int, int]] = []
        vectors: list[np.ndarray] = []
        weights: list[float] = []
        for edge in geometry.edges:
            left = int(edge.cation)
            right = int(edge.anion)
            fractional = (
                np.asarray(structure[right].frac_coords, dtype=float)
                + np.asarray(edge.image, dtype=float)
                - np.asarray(structure[left].frac_coords, dtype=float)
            )
            endpoints.append((left, right))
            vectors.append(structure.lattice.get_cartesian_coords(fractional))
            weights.append(float(edge.neighbor_weight))
        result = periodic_deviatoric_strain_retention(
            n_sites=len(structure),
            endpoints=np.asarray(endpoints, dtype=int),
            displacements=np.asarray(vectors, dtype=float),
            weights=np.asarray(weights, dtype=float),
        )
        feature = _quantize(result.retention_floor)
        if not 0.0 <= feature <= 1.0:
            raise RuntimeError("NEXT351 feature domain differs")
        features = {FEATURE_NAMES[0]: feature}
        return PDSRFeatureResult(
            True,
            None,
            len(structure),
            len(endpoints),
            result.maximum_orthogonality_residual,
            float(np.linalg.eigvalsh(result.affine_gram)[0]),
            features,
        )
    except Exception as exc:
        return _failure(exc)


def compute_pdsr_row(atoms: Atoms) -> dict[str, object]:
    result = compute_pdsr_features(atoms)
    return {
        FEATURE_NAMES[0]: result.features.get(FEATURE_NAMES[0], math.nan),
        "pdsr_supported": bool(result.supported),
        "pdsr_failure": result.failure_reason,
        "pdsr_site_count": int(result.site_count),
        "pdsr_edge_count": int(result.edge_count),
        "pdsr_maximum_orthogonality_residual": result.maximum_orthogonality_residual,
        "pdsr_affine_minimum_eigenvalue": result.affine_minimum_eigenvalue,
    }


__all__ = [
    "BOUNDARY_FLAGS",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "ORTHOGONALITY_TOLERANCE",
    "PROTOCOL",
    "PDSRFeatureResult",
    "DeviatoricRetentionResult",
    "compute_pdsr_features",
    "compute_pdsr_row",
    "deviatoric_strain_basis",
    "periodic_deviatoric_strain_retention",
]
