#!/usr/bin/env python3
"""Periodic bond-angle affine accommodation from one raw x0 geometry."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Mapping, Sequence

from ase import Atoms
import numpy as np
from scipy import linalg
import spglib

from src.next26_packing import _radii
from src.next49_framework_topology import (
    _canonical_covalent_edges,
    _component_dimensions,
    _directed_adjacency,
    _strict_geometry,
)


PROTOCOL = "2026-08-13-next537-periodic-bond-angle-affine-accommodation-v2"
DESIGN_SHA256 = "68bee1dea45492f1bf7349965dcd149497e91e9ed6a1aa29098ce3bc0b01ceac"
FEATURE_NAMES = ("pbaaa_periodic_bond_angle_affine_accommodation",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "risk_high"}
PRIMITIVE_SYMPREC = 1.0e-5
OUTPUT_GRID = 1.0e8
BOUNDARY_FLAGS = {
    "dft_calculation_executed": False,
    "dft_values_used": False,
    "relaxed_structures_used": False,
    "learned_energy_force_stress_proxy_used": False,
    "model_or_proxy_potential_used": False,
    "physical_or_virtual_coordinate_relaxation_executed": False,
}


@dataclass(frozen=True)
class PBAAAResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    edge_constraint_count: int
    angle_constraint_count: int
    direct_rank: int
    atomic_rank: int
    generalized_eigenvalues: tuple[float, ...]
    factorization_residual: float
    primitive_reduced: bool | None
    features: Mapping[str, float]


def _failure(exc: object) -> PBAAAResult:
    if isinstance(exc, Exception):
        reason = f"{type(exc).__name__}: {exc}"
    else:
        reason = str(exc)
    return PBAAAResult(False, reason, 0, 0, 0, 0, 0, (), math.nan, None, {})


def _kelvin_bond(unit: np.ndarray) -> np.ndarray:
    x, y, z = unit
    root_two = math.sqrt(2.0)
    return np.asarray(
        [x * x, y * y, z * z, root_two * y * z, root_two * x * z, root_two * x * y],
        dtype=float,
    )


def _kelvin_angle(
    first_vector: np.ndarray,
    second_vector: np.ndarray,
    first_gradient: np.ndarray,
    second_gradient: np.ndarray,
) -> np.ndarray:
    tensor = np.outer(first_gradient, first_vector) + np.outer(
        second_gradient, second_vector
    )
    tensor = 0.5 * (tensor + tensor.T)
    root_two = math.sqrt(2.0)
    return np.asarray(
        [
            tensor[0, 0],
            tensor[1, 1],
            tensor[2, 2],
            root_two * tensor[1, 2],
            root_two * tensor[0, 2],
            root_two * tensor[0, 1],
        ],
        dtype=float,
    )


def _constraint_matrices(
    *, n_sites: int, endpoints: np.ndarray, vectors: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, int]:
    atomic_rows: list[np.ndarray] = []
    affine_rows: list[np.ndarray] = []
    adjacency: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(n_sites)]
    for (raw_left, raw_right), raw_vector in zip(endpoints, vectors, strict=True):
        left = int(raw_left)
        right = int(raw_right)
        vector = np.asarray(raw_vector, dtype=float)
        distance = float(np.linalg.norm(vector))
        unit = vector / distance
        atomic = np.zeros(3 * n_sites, dtype=float)
        atomic[3 * left : 3 * left + 3] -= unit / distance
        atomic[3 * right : 3 * right + 3] += unit / distance
        atomic_rows.append(atomic)
        affine_rows.append(_kelvin_bond(unit))
        adjacency[left].append((right, vector))
        adjacency[right].append((left, -vector))
    edge_count = len(atomic_rows)

    angle_count = 0
    for center, neighbours in enumerate(adjacency):
        for (left, first), (right, second) in combinations(neighbours, 2):
            first_length = float(np.linalg.norm(first))
            second_length = float(np.linalg.norm(second))
            first_unit = first / first_length
            second_unit = second / second_length
            cosine = float(np.clip(first_unit @ second_unit, -1.0, 1.0))
            first_gradient = (second_unit - cosine * first_unit) / first_length
            second_gradient = (first_unit - cosine * second_unit) / second_length
            atomic = np.zeros(3 * n_sites, dtype=float)
            atomic[3 * left : 3 * left + 3] += first_gradient
            atomic[3 * right : 3 * right + 3] += second_gradient
            atomic[3 * center : 3 * center + 3] -= first_gradient + second_gradient
            affine = _kelvin_angle(first, second, first_gradient, second_gradient)
            if float(np.linalg.norm(atomic)) + float(np.linalg.norm(affine)) <= 1.0e-12:
                continue
            atomic_rows.append(atomic)
            affine_rows.append(affine)
            angle_count += 1
    return (
        np.asarray(atomic_rows, dtype=float),
        np.asarray(affine_rows, dtype=float),
        edge_count,
        angle_count,
    )


def _project_affine_columns(
    atomic: np.ndarray, affine: np.ndarray
) -> tuple[np.ndarray, int]:
    """Project affine constraints with a permutation-invariant SVD column basis."""

    if (
        atomic.ndim != 2
        or affine.ndim != 2
        or atomic.shape[0] != affine.shape[0]
        or affine.shape[1] != 6
        or not np.isfinite(atomic).all()
        or not np.isfinite(affine).all()
    ):
        raise ValueError("NEXT537 projection matrix differs")
    left, singular, _right = linalg.svd(
        atomic, full_matrices=False, lapack_driver="gesdd"
    )
    if not len(singular) or float(singular[0]) <= 0.0:
        return affine.copy(), 0
    tolerance = np.finfo(float).eps * max(atomic.shape) * float(singular[0])
    rank = int(np.sum(singular > tolerance))
    if rank == 0:
        return affine.copy(), 0
    basis = left[:, :rank]
    return affine - basis @ (basis.T @ affine), rank


def periodic_bond_angle_affine_accommodation(
    *,
    n_sites: int,
    endpoints: Sequence[Sequence[int]] | np.ndarray,
    vectors: Sequence[Sequence[float]] | np.ndarray,
) -> PBAAAResult:
    """Measure the softest affine strain of a fixed periodic constraint graph."""

    try:
        if not isinstance(n_sites, (int, np.integer)) or int(n_sites) < 1:
            raise ValueError("NEXT537 site population differs")
        n_sites = int(n_sites)
        raw_pair = np.asarray(endpoints)
        vector = np.asarray(vectors, dtype=float)
        if (
            raw_pair.ndim != 2
            or raw_pair.shape[1:] != (2,)
            or len(raw_pair) < 1
            or not np.isfinite(raw_pair.astype(float)).all()
            or not np.equal(raw_pair.astype(float), np.rint(raw_pair.astype(float))).all()
        ):
            raise ValueError("NEXT537 endpoint population differs")
        pair = raw_pair.astype(int)
        if np.any(pair < 0) or np.any(pair >= n_sites):
            raise ValueError("NEXT537 endpoint index differs")
        if vector.shape != (len(pair), 3) or not np.isfinite(vector).all():
            raise ValueError("NEXT537 periodic vector population differs")
        distance = np.linalg.norm(vector, axis=1)
        if np.any(~np.isfinite(distance)) or np.any(distance <= 1.0e-12):
            raise ValueError("NEXT537 periodic vector length differs")

        atomic, affine, edge_count, angle_count = _constraint_matrices(
            n_sites=n_sites, endpoints=pair, vectors=vector
        )
        if (
            atomic.ndim != 2
            or atomic.shape[1] != 3 * n_sites
            or affine.shape != (len(atomic), 6)
            or not np.isfinite(atomic).all()
            or not np.isfinite(affine).all()
        ):
            raise RuntimeError("NEXT537 constraint matrix differs")

        residual, atomic_rank = _project_affine_columns(atomic, affine)
        direct_gram = affine.T @ affine
        residual_gram = residual.T @ residual
        direct_gram = 0.5 * (direct_gram + direct_gram.T)
        residual_gram = 0.5 * (residual_gram + residual_gram.T)
        direct_values, direct_vectors = np.linalg.eigh(direct_gram)
        if not np.isfinite(direct_values).all() or float(direct_values[-1]) <= 0.0:
            raise RuntimeError("NEXT537 direct strain spectrum differs")
        tolerance = (
            np.finfo(float).eps
            * max(affine.shape)
            * float(direct_values[-1])
        )
        direct_rank = int(np.sum(direct_values > tolerance))
        orthogonal = float(
            np.linalg.norm(atomic.T @ residual)
            / max(
                np.linalg.norm(atomic) * np.linalg.norm(residual),
                np.finfo(float).tiny,
            )
        )
        if direct_rank < 6:
            generalized = np.asarray([], dtype=float)
            risk = 1.0
        else:
            inverse_root = (
                direct_vectors
                @ np.diag(1.0 / np.sqrt(direct_values))
                @ direct_vectors.T
            )
            whitened = inverse_root @ residual_gram @ inverse_root
            whitened = 0.5 * (whitened + whitened.T)
            generalized = np.linalg.eigvalsh(whitened)
            if np.any(~np.isfinite(generalized)):
                raise RuntimeError("NEXT537 generalized spectrum differs")
            if float(generalized.min()) < -1.0e-8 or float(generalized.max()) > 1.0 + 1.0e-8:
                raise RuntimeError("NEXT537 generalized spectrum leaves unit interval")
            generalized = np.clip(generalized, 0.0, 1.0)
            risk = 1.0 - float(generalized.min())
        quantized = float(np.rint(np.clip(risk, 0.0, 1.0) * OUTPUT_GRID) / OUTPUT_GRID)
        values = {FEATURE_NAMES[0]: quantized}
        return PBAAAResult(
            True,
            None,
            n_sites,
            edge_count,
            angle_count,
            direct_rank,
            int(atomic_rank),
            tuple(float(value) for value in generalized),
            orthogonal,
            None,
            values,
        )
    except Exception as exc:
        return _failure(exc)


def _primitive_representation(atoms: Atoms) -> tuple[Atoms, bool]:
    numbers, _positions, cell = _strict_geometry(atoms)
    fractional = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    primitive = spglib.find_primitive(
        (cell, fractional, numbers), symprec=PRIMITIVE_SYMPREC
    )
    if primitive is None:
        return atoms.copy(), False
    primitive_cell, primitive_fractional, primitive_numbers = primitive
    candidate = Atoms(
        numbers=np.asarray(primitive_numbers, dtype=int),
        scaled_positions=np.asarray(primitive_fractional, dtype=float),
        cell=np.asarray(primitive_cell, dtype=float),
        pbc=True,
    )
    _strict_geometry(candidate)
    return candidate, len(candidate) < len(atoms)


def compute_periodic_bond_angle_affine_accommodation(atoms: Atoms) -> PBAAAResult:
    """Build the periodic framework graph and evaluate PBAAA without moving atoms."""

    try:
        working, reduced = _primitive_representation(atoms)
        numbers, _positions, _cell = _strict_geometry(working)
        covalent, _van_der_waals = _radii(numbers)
        covalent = np.asarray(covalent, dtype=float)
        if (
            covalent.shape != (len(working),)
            or not np.isfinite(covalent).all()
            or np.any(covalent <= 0.0)
        ):
            raise ValueError("NEXT537 covalent radii differ")
        edges = _canonical_covalent_edges(working, covalent)
        adjacency = _directed_adjacency(len(working), edges)
        periodic_components = [
            members
            for members, dimension in _component_dimensions(adjacency)
            if dimension > 0
        ]
        active = {index for members in periodic_components for index in members}
        if not active:
            raise ValueError("NEXT537 geometry has no periodic covalent component")
        ordered = sorted(active)
        remap = {old: new for new, old in enumerate(ordered)}
        selected = [
            edge
            for edge in edges
            if int(edge.first) in active and int(edge.second) in active
        ]
        result = periodic_bond_angle_affine_accommodation(
            n_sites=len(ordered),
            endpoints=np.asarray(
                [(remap[int(edge.first)], remap[int(edge.second)]) for edge in selected],
                dtype=int,
            ),
            vectors=np.asarray([edge.vector for edge in selected], dtype=float),
        )
        if not result.supported:
            return result
        return PBAAAResult(
            **{
                **result.__dict__,
                "primitive_reduced": bool(reduced),
            }
        )
    except Exception as exc:
        return _failure(exc)


__all__ = [
    "BOUNDARY_FLAGS",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "PBAAAResult",
    "PROTOCOL",
    "compute_periodic_bond_angle_affine_accommodation",
    "periodic_bond_angle_affine_accommodation",
]
