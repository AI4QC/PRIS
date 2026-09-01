#!/usr/bin/env python3
"""Periodic coordination-reciprocity likelihood from raw x0 geometry only."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Mapping
import warnings

from ase import Atoms
import numpy as np
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.io.ase import AseAtomsAdaptor

import src.next267_periodic_radical_voronoi_packing as n267


PROTOCOL = "2026-08-13-next375-periodic-coordination-reciprocity-likelihood-v1"
_DESIGN_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/plans/2026-08-13-next375-next378-periodic-coordination-reciprocity-likelihood.md"
)
DESIGN_SHA256 = hashlib.sha256(_DESIGN_PATH.read_bytes()).hexdigest()
FEATURE_NAMES = ("pcrl_reciprocity_deficit",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_low"}
OUTPUT_GRID = 1.0e10
REVERSE_ANGLE_TOLERANCE = 1.0e-8
NUMERICAL_TOLERANCE = 1.0e-12
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
class PCRLFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    directed_face_count: int
    selected_directed_count: int
    unreciprocated_directed_count: int
    maximum_reverse_angle_error: float
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> PCRLFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return PCRLFeatureResult(False, reason, 0, 0, 0, 0, math.nan, {})


def _quantized(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("NEXT375 value is non-finite")
    return float(np.rint(value * OUTPUT_GRID) / OUTPUT_GRID)


def _bounded(value: float, *, label: str) -> float:
    if (
        not math.isfinite(value)
        or value < -NUMERICAL_TOLERANCE
        or value > 1.0 + NUMERICAL_TOLERANCE
    ):
        raise ValueError(f"NEXT375 {label} is outside [0,1]")
    return float(np.clip(value, 0.0, 1.0))


def _reverse(key: tuple[int, int, int, int, int]) -> tuple[int, int, int, int, int]:
    left, right, x, y, z = key
    return right, left, -x, -y, -z


def coordination_reciprocity_likelihood(
    *, n_sites: int, endpoints: object, solid_angles: object
) -> PCRLFeatureResult:
    """Evaluate reciprocal consistency of independently preferred prefixes."""

    try:
        if type(n_sites) is not int or n_sites < 1:
            raise ValueError("NEXT375 site population differs")
        raw_endpoints = np.asarray(endpoints)
        raw_angles = np.asarray(solid_angles, dtype=float)
        if (
            raw_endpoints.ndim != 2
            or raw_endpoints.shape[1:] != (5,)
            or len(raw_endpoints) == 0
            or raw_angles.shape != (len(raw_endpoints),)
            or np.any(~np.isfinite(raw_angles))
            or np.any(raw_angles <= 0.0)
        ):
            raise ValueError("NEXT375 directed facet population differs")
        endpoint_float = np.asarray(raw_endpoints, dtype=float)
        if (
            np.any(~np.isfinite(endpoint_float))
            or np.any(endpoint_float != np.floor(endpoint_float))
        ):
            raise ValueError("NEXT375 periodic contact key differs")
        endpoint_int = endpoint_float.astype(int)
        if (
            np.any(endpoint_int[:, :2] < 0)
            or np.any(endpoint_int[:, :2] >= n_sites)
        ):
            raise ValueError("NEXT375 periodic contact index differs")
        keys = [tuple(int(value) for value in row) for row in endpoint_int.tolist()]
        if len(set(keys)) != len(keys):
            raise ValueError("NEXT375 duplicate directed facet differs")
        for key in keys:
            if key[0] == key[1] and key[2:] == (0, 0, 0):
                raise ValueError("NEXT375 zero-image self contact differs")
        angle_by_key = dict(zip(keys, raw_angles.tolist(), strict=True))
        shared: dict[tuple[int, int, int, int, int], float] = {}
        reverse_errors: list[float] = []
        for key in sorted(angle_by_key):
            reverse = _reverse(key)
            if reverse not in angle_by_key:
                raise ValueError("NEXT375 reverse incidence is incomplete")
            error = abs(float(angle_by_key[key]) - float(angle_by_key[reverse]))
            reverse_errors.append(error)
            if error > REVERSE_ANGLE_TOLERANCE:
                raise ValueError("NEXT375 reverse solid angle differs")
            shared[key] = 0.5 * (
                float(angle_by_key[key]) + float(angle_by_key[reverse])
            )

        contacts_by_site: list[list[tuple[tuple[int, int, int, int, int], float]]] = [
            [] for _ in range(n_sites)
        ]
        for key in sorted(shared):
            contacts_by_site[key[0]].append((key, shared[key]))
        if any(not contacts for contacts in contacts_by_site):
            raise ValueError("NEXT375 site has no periodic Voronoi contact")

        selected: set[tuple[int, int, int, int, int]] = set()
        for contacts in contacts_by_site:
            ordered = sorted(contacts, key=lambda item: (-item[1], item[0]))
            maximum = float(ordered[0][1])
            ratios = np.asarray(
                [_quantized(float(weight) / maximum) for _, weight in ordered],
                dtype=float,
            )
            ratios = np.concatenate((ratios, np.asarray((0.0,))))
            gaps = ratios[:-1] - ratios[1:]
            if np.any(gaps < -NUMERICAL_TOLERANCE):
                raise RuntimeError("NEXT375 ordered likelihood gap differs")
            largest = float(np.max(gaps))
            tied = np.flatnonzero(np.abs(gaps - largest) <= NUMERICAL_TOLERANCE)
            prefix_length = int(tied[-1]) + 1
            selected.update(key for key, _ in ordered[:prefix_length])
        if not selected:
            raise RuntimeError("NEXT375 selected coordination population is empty")

        numerator = math.fsum(
            shared[key] for key in selected if _reverse(key) not in selected
        )
        denominator = math.fsum(shared[key] for key in selected)
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise RuntimeError("NEXT375 selected coordination weight differs")
        deficit = _quantized(
            _bounded(numerator / denominator, label="reciprocity deficit")
        )
        return PCRLFeatureResult(
            True,
            None,
            n_sites,
            len(keys),
            len(selected),
            sum(_reverse(key) not in selected for key in selected),
            max(reverse_errors, default=0.0),
            {FEATURE_NAMES[0]: deficit},
        )
    except Exception as exc:
        return _failure(exc)


def _strict_reduced_atoms(atoms: Atoms) -> Atoms:
    if (
        not isinstance(atoms, Atoms)
        or len(atoms) < 1
        or np.asarray(atoms.pbc, dtype=bool).shape != (3,)
        or not np.all(atoms.pbc)
        or atoms.calc is not None
        or bool(atoms.info)
        or set(atoms.arrays) != {"numbers", "positions"}
    ):
        raise ValueError("NEXT375 features require exact periodic geometry-only Atoms")
    cell = np.asarray(atoms.cell.array, dtype=float)
    positions = np.asarray(atoms.positions, dtype=float)
    if (
        cell.shape != (3, 3)
        or positions.shape != (len(atoms), 3)
        or not np.isfinite(cell).all()
        or not np.isfinite(positions).all()
        or abs(float(np.linalg.det(cell))) <= 1.0e-12
    ):
        raise ValueError("NEXT375 features require exact periodic geometry-only Atoms")
    try:
        return n267._validated_reduced_atoms(atoms)
    except Exception as exc:
        raise ValueError(
            "NEXT375 features require exact periodic geometry-only Atoms"
        ) from exc


def compute_pcrl_features(atoms: Atoms) -> PCRLFeatureResult:
    """Compute the frozen geometric coordination-reciprocity deficit."""

    try:
        work = _strict_reduced_atoms(atoms)
        structure = AseAtomsAdaptor.get_structure(work)
        finder = VoronoiNN(weight="solid_angle", tol=0, cutoff=13)
        angle_by_key: dict[tuple[int, int, int, int, int], float] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for center in range(len(structure)):
                for item in finder.get_nn_info(structure, center):
                    try:
                        neighbor = int(item["site_index"])
                        image_float = np.asarray(item["image"], dtype=float)
                        image = np.rint(image_float).astype(int)
                        angle = float(item["poly_info"]["solid_angle"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if (
                        neighbor < 0
                        or neighbor >= len(structure)
                        or image_float.shape != (3,)
                        or not np.isfinite(image_float).all()
                        or not np.allclose(image_float, image, rtol=0.0, atol=1.0e-10)
                        or not math.isfinite(angle)
                        or angle <= 0.0
                    ):
                        continue
                    key = (center, neighbor, *(int(value) for value in image))
                    previous = angle_by_key.get(key)
                    if previous is not None and not math.isclose(
                        previous, angle, rel_tol=0.0, abs_tol=REVERSE_ANGLE_TOLERANCE
                    ):
                        raise ValueError("NEXT375 duplicate Voronoi facet differs")
                    angle_by_key[key] = angle if previous is None else 0.5 * (previous + angle)
        return coordination_reciprocity_likelihood(
            n_sites=len(structure),
            endpoints=np.asarray(sorted(angle_by_key), dtype=int),
            solid_angles=np.asarray(
                [angle_by_key[key] for key in sorted(angle_by_key)], dtype=float
            ),
        )
    except Exception as exc:
        return _failure(exc)


def compute_pcrl_row(atoms: Atoms) -> dict[str, object]:
    result = compute_pcrl_features(atoms)
    return {
        FEATURE_NAMES[0]: result.features.get(FEATURE_NAMES[0], math.nan),
        "pcrl_supported": bool(result.supported),
        "pcrl_failure": result.failure_reason,
        "pcrl_site_count": int(result.site_count),
        "pcrl_directed_face_count": int(result.directed_face_count),
        "pcrl_selected_directed_count": int(result.selected_directed_count),
        "pcrl_unreciprocated_directed_count": int(
            result.unreciprocated_directed_count
        ),
        "pcrl_maximum_reverse_angle_error": result.maximum_reverse_angle_error,
    }


__all__ = [
    "BOUNDARY_FLAGS",
    "DESIGN_SHA256",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "NUMERICAL_TOLERANCE",
    "OUTPUT_GRID",
    "PCRLFeatureResult",
    "PROTOCOL",
    "REVERSE_ANGLE_TOLERANCE",
    "compute_pcrl_features",
    "compute_pcrl_row",
    "coordination_reciprocity_likelihood",
]
