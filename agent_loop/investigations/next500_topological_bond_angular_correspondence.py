"""Frozen NEXT500 topological bond/angular correspondence (no DFT)."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence
import warnings

from ase import Atoms
import numpy as np
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.io.ase import AseAtomsAdaptor

import src.next19_valence_transport as n19
import src.next267_periodic_radical_voronoi_packing as n267
import src.next295_positive_contact_force_closure as n295
import src.next440_path_constrained_apriori_bond_positivity as n440


PROTOCOL = "2026-08-13-next500-topological-bond-angular-correspondence-v1"
DESIGN_SHA256 = "8884d37ebabf6d7653dd83b154274b9b5268256c744f49bf24d495a54077430a"
FEATURE_NAMES = ("tbac_topological_bond_angular_correspondence",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 1.0e10
ANGLE_DUPLICATE_TOLERANCE = 1.0e-8
BOUNDARY_FLAGS = dict(n440.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class TBACResult:
    supported: bool
    failure_reason: str | None
    feasible: bool | None
    site_count: int
    edge_count: int
    edge_strengths: tuple[float, ...]
    solid_angles: tuple[float, ...]
    angular_targets_cation: tuple[float, ...]
    angular_targets_anion: tuple[float, ...]
    negative_edge_count: int
    normalized_mismatch: float
    maximum_equality_residual: float
    maximum_path_residual: float
    valence_policy: str | None
    features: Mapping[str, float]


def _failure(exc: object) -> TBACResult:
    reason = str(exc)
    if not reason.startswith(type(exc).__name__) and isinstance(exc, Exception):
        reason = f"{type(exc).__name__}: {reason}"
    return TBACResult(
        False,
        reason,
        None,
        0,
        0,
        (),
        (),
        (),
        (),
        0,
        math.nan,
        math.nan,
        math.nan,
        None,
        {},
    )


def _zero(*, site_count: int, edge_count: int) -> TBACResult:
    return TBACResult(
        True,
        None,
        False,
        int(site_count),
        int(edge_count),
        (),
        (),
        (),
        (),
        0,
        1.0,
        0.0,
        0.0,
        None,
        {FEATURE_NAMES[0]: 0.0},
    )


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def topological_bond_angular_correspondence(
    *,
    charges: Sequence[float] | object,
    endpoints: Sequence[Sequence[int]] | object,
    solid_angles: Sequence[float] | object,
) -> TBACResult:
    """Compare the curl-free a-priori edge field with angular charge shares."""

    try:
        charge = np.asarray(charges, dtype=float)
        raw_pair = np.asarray(endpoints)
        angle = np.asarray(solid_angles, dtype=float)
        if charge.ndim != 1 or len(charge) < 2 or not np.isfinite(charge).all():
            raise ValueError("NEXT500 charges differ")
        magnitude = float(np.abs(charge).sum())
        if (
            magnitude <= 0.0
            or abs(float(charge.sum()))
            > n440.CHARGE_TOLERANCE * max(1.0, magnitude)
            or np.any(charge == 0.0)
            or not np.any(charge > 0.0)
            or not np.any(charge < 0.0)
        ):
            raise ValueError("NEXT500 formal charges must be neutral and nonzero")
        if (
            raw_pair.ndim != 2
            or raw_pair.shape[1:] != (2,)
            or len(raw_pair) < 1
            or not np.isfinite(raw_pair.astype(float)).all()
            or not np.equal(
                raw_pair.astype(float), np.rint(raw_pair.astype(float))
            ).all()
        ):
            raise ValueError("NEXT500 endpoint population differs")
        pair = raw_pair.astype(int)
        if (
            np.any(pair < 0)
            or np.any(pair >= len(charge))
            or np.any(pair[:, 0] == pair[:, 1])
            or not np.all(charge[pair[:, 0]] > 0.0)
            or not np.all(charge[pair[:, 1]] < 0.0)
        ):
            raise ValueError("NEXT500 cation-anion edge orientation differs")
        if (
            angle.shape != (len(pair),)
            or not np.isfinite(angle).all()
            or np.any(angle <= 0.0)
        ):
            raise ValueError("NEXT500 solid-angle population differs")
        degree = np.bincount(pair.ravel(), minlength=len(charge))
        if np.any(degree == 0):
            raise ValueError("NEXT500 graph contains an isolated charged site")

        field = n440.path_constrained_apriori_bond_positivity(
            charges=charge, endpoints=pair
        )
        if not field.supported:
            raise ValueError(field.failure_reason or "NEXT500 path field failed")
        if not field.feasible:
            return _zero(site_count=len(charge), edge_count=len(pair))
        strength = np.asarray(field.edge_strengths, dtype=float)
        if strength.shape != angle.shape or not np.isfinite(strength).all():
            raise RuntimeError("NEXT500 path-field population differs")

        angular_mass = np.zeros(len(charge), dtype=float)
        np.add.at(angular_mass, pair[:, 0], angle)
        np.add.at(angular_mass, pair[:, 1], angle)
        if np.any(~np.isfinite(angular_mass)) or np.any(angular_mass <= 0.0):
            raise RuntimeError("NEXT500 site angular mass differs")
        target_cation = (
            np.abs(charge[pair[:, 0]]) * angle / angular_mass[pair[:, 0]]
        )
        target_anion = (
            np.abs(charge[pair[:, 1]]) * angle / angular_mass[pair[:, 1]]
        )
        targets = np.concatenate((target_cation, target_anion))
        repeated_strength = np.concatenate((strength, strength))
        numerator = math.fsum(np.abs(repeated_strength - targets).tolist())
        denominator = math.fsum((np.abs(repeated_strength) + targets).tolist())
        if (
            not math.isfinite(numerator)
            or not math.isfinite(denominator)
            or numerator < 0.0
            or denominator <= 0.0
            or numerator > denominator + 1.0e-10 * denominator
        ):
            raise RuntimeError("NEXT500 normalized mismatch differs")
        mismatch = float(np.clip(numerator / denominator, 0.0, 1.0))
        correspondence = _quantize(1.0 - mismatch)
        if (
            not math.isfinite(correspondence)
            or correspondence < 0.0
            or correspondence > 1.0
        ):
            raise RuntimeError("NEXT500 bounded feature differs")
        return TBACResult(
            True,
            None,
            True,
            len(charge),
            len(pair),
            tuple(float(value) for value in strength),
            tuple(float(value) for value in angle),
            tuple(float(value) for value in target_cation),
            tuple(float(value) for value in target_anion),
            int(np.sum(strength < -n440.SOLVE_RESIDUAL_TOLERANCE)),
            mismatch,
            float(field.maximum_equality_residual),
            float(field.maximum_path_residual),
            None,
            {FEATURE_NAMES[0]: correspondence},
        )
    except Exception as exc:
        return _failure(exc)


def _cation_voronoi_solid_angles(
    structure: object,
    charges: Sequence[float] | np.ndarray,
    edges: Sequence[n19.PeriodicEdgeGeometry],
) -> tuple[float, ...]:
    """Return cation-side raw ordinary-Voronoi solid angles in edge order."""

    charge = np.asarray(charges, dtype=float)
    working = structure.copy()
    working.add_oxidation_state_by_site(charge.tolist())
    finder = VoronoiNN(weight="solid_angle", tol=0.0, cutoff=13.0)
    by_key: dict[tuple[int, int, tuple[int, int, int]], float] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cation in np.flatnonzero(charge > 0.0):
            for item in finder.get_nn_info(working, int(cation)):
                try:
                    anion = int(item["site_index"])
                    image_float = np.asarray(item["image"], dtype=float)
                    image = np.rint(image_float).astype(int)
                    angle = float(item["poly_info"]["solid_angle"])
                except (KeyError, TypeError, ValueError):
                    continue
                if (
                    anion < 0
                    or anion >= len(charge)
                    or charge[anion] >= 0.0
                    or image_float.shape != (3,)
                    or not np.isfinite(image_float).all()
                    or not np.allclose(
                        image_float, image, rtol=0.0, atol=1.0e-10
                    )
                    or not math.isfinite(angle)
                    or angle <= 0.0
                ):
                    continue
                key = (
                    int(cation),
                    anion,
                    tuple(int(value) for value in image),
                )
                previous = by_key.get(key)
                if previous is not None and not math.isclose(
                    previous,
                    angle,
                    rel_tol=0.0,
                    abs_tol=ANGLE_DUPLICATE_TOLERANCE,
                ):
                    raise ValueError("NEXT500 duplicate Voronoi solid angle differs")
                by_key[key] = angle if previous is None else 0.5 * (previous + angle)
    ordered: list[float] = []
    for edge in edges:
        key = (int(edge.cation), int(edge.anion), tuple(int(v) for v in edge.image))
        if key not in by_key:
            raise ValueError("NEXT500 opposite-sign Voronoi edge lacks solid angle")
        ordered.append(float(by_key[key]))
    return tuple(ordered)


def compute_tbac_features(atoms: Atoms) -> TBACResult:
    """Compute TBAC from composition and one raw unrelaxed geometry only."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT500 valence assignment failed"
            )
        charge = np.asarray(assignment.values, dtype=float)
        if charge.shape != (len(structure),):
            raise ValueError("NEXT500 valence population differs")
        geometry = n19.build_periodic_edge_geometry(
            structure, charge, graph_mode="voronoi"
        )
        if not geometry.supported:
            reason = str(geometry.failure_reason or "NEXT500 periodic graph failed")
            if "no opposite-sign periodic neighbor" in reason:
                return replace(
                    _zero(site_count=len(structure), edge_count=0),
                    valence_policy=str(assignment.policy),
                )
            raise ValueError(reason)
        edges = tuple(geometry.edges)
        angles = _cation_voronoi_solid_angles(structure, charge, edges)
        result = topological_bond_angular_correspondence(
            charges=charge,
            endpoints=tuple(
                (int(edge.cation), int(edge.anion)) for edge in edges
            ),
            solid_angles=angles,
        )
        if not result.supported:
            return result
        return replace(result, valence_policy=str(assignment.policy))
    except Exception as exc:
        result = _failure(exc)
        if "NEXT295" in str(exc):
            return replace(
                result,
                failure_reason=str(exc).replace("NEXT295", "NEXT500"),
            )
        return result


def compute_tbac_row(atoms: Atoms) -> dict[str, object]:
    result = compute_tbac_features(atoms)
    return {
        FEATURE_NAMES[0]: (
            float(result.features[FEATURE_NAMES[0]])
            if result.supported
            else math.nan
        ),
        "tbac_supported": bool(result.supported),
        "tbac_failure": result.failure_reason,
        "tbac_feasible": result.feasible,
        "tbac_site_count": int(result.site_count),
        "tbac_edge_count": int(result.edge_count),
        "tbac_negative_edge_count": int(result.negative_edge_count),
        "tbac_normalized_mismatch": float(result.normalized_mismatch),
        "tbac_maximum_equality_residual": float(
            result.maximum_equality_residual
        ),
        "tbac_maximum_path_residual": float(result.maximum_path_residual),
        "tbac_valence_policy": result.valence_policy,
    }


__all__ = [
    "BOUNDARY_FLAGS",
    "DESIGN_SHA256",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "PROTOCOL",
    "TBACResult",
    "compute_tbac_features",
    "compute_tbac_row",
    "topological_bond_angular_correspondence",
]
