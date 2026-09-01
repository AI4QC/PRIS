"""Pure analytic valence-transport descriptors for raw crystal geometry.

This module deliberately has no calculator, relaxation, endpoint, or model
adapter.  Its core operation is a sparse linear transport problem over a
periodic cation-anion graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog


PROTOCOL = "2026-08-02-next19-valence-transport-v2"
FORBIDDEN_FEATURE_TOKENS = (
    "energy",
    "force",
    "stress",
    "relax",
    "mattersim",
    "dft",
)
FEATURE_NAMES = (
    "vt_overload",
    "vt_reallocation",
    "vt_anion_mismatch_max",
    "vt_periodic_edge_count",
    "vt_cation_count",
    "vt_anion_count",
)


@dataclass(frozen=True, order=True)
class EdgePrior:
    """One directed cation-to-anion periodic edge and its prior bond valence."""

    cation: int
    anion: int
    image: tuple[int, int, int]
    prior: float


@dataclass(frozen=True)
class TransportSolution:
    """Fail-open result of the two-stage valence transport solve."""

    supported: bool
    failure_reason: str | None
    overload: float
    reallocation: float
    max_anion_mismatch: float
    kappa: float
    edge_flow: tuple[float, ...]
    canonical_edges: tuple[EdgePrior, ...]


@dataclass(frozen=True)
class GraphPriorResult:
    """Geometry-only periodic graph converted to cation-normalized priors."""

    supported: bool
    failure_reason: str | None
    edges: tuple[EdgePrior, ...]
    cation_supply: Mapping[int, float]
    anion_demand: Mapping[int, float]


@dataclass(frozen=True, order=True)
class PeriodicEdgeGeometry:
    """One opposite-sign periodic neighbor before alpha reweighting."""

    cation: int
    anion: int
    image: tuple[int, int, int]
    distance: float
    neighbor_weight: float


@dataclass(frozen=True)
class PeriodicGeometryResult:
    """Cached periodic geometry shared by the fixed alpha catalogue."""

    supported: bool
    failure_reason: str | None
    edges: tuple[PeriodicEdgeGeometry, ...]
    cation_supply: Mapping[int, float]
    anion_demand: Mapping[int, float]


@dataclass(frozen=True)
class StructureFeatureResult:
    """Fail-open per-structure descriptor result."""

    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


@dataclass(frozen=True)
class ValenceAssignment:
    """Auditable analytic charge-sign and magnitude assignment."""

    supported: bool
    values: tuple[float, ...] | None
    policy: str | None
    failure_reason: str | None


def validate_feature_names(names: Iterable[str]) -> None:
    """Reject feature schemas that cross the pure analytic input boundary."""

    for name in names:
        lowered = str(name).lower()
        if any(token in lowered for token in FORBIDDEN_FEATURE_TOKENS):
            raise ValueError(f"forbidden feature field: {name}")


def _unsupported(reason: str) -> TransportSolution:
    return TransportSolution(
        supported=False,
        failure_reason=reason,
        overload=math.nan,
        reallocation=math.nan,
        max_anion_mismatch=math.nan,
        kappa=math.nan,
        edge_flow=(),
        canonical_edges=(),
    )


def _unsupported_graph(reason: str) -> GraphPriorResult:
    return GraphPriorResult(False, reason, (), {}, {})


def _unsupported_geometry(reason: str) -> PeriodicGeometryResult:
    return PeriodicGeometryResult(False, reason, (), {}, {})


def infer_valence_assignment(structure) -> ValenceAssignment:
    """Infer formal valences, then a scale-free electronegativity partition."""

    try:
        from src.discriminate import guess_oxi

        values, ok = guess_oxi(structure)
    except Exception as exc:
        values, ok = None, False
        integer_error = f"integer valence inference failed: {type(exc).__name__}"
    else:
        integer_error = None
    policy = "integer_oxidation_state" if ok else None
    if not ok:
        try:
            from src.apply_rules import frac_oxi

            values = frac_oxi(structure)
        except Exception as exc:
            values = None
            fractional_error = (
                f"fractional valence inference failed: {type(exc).__name__}"
            )
        else:
            fractional_error = None
        if values is not None:
            policy = "fractional_oxidation_state"
    else:
        fractional_error = None
    if values is not None:
        array = np.asarray(values, dtype=float)
        if (
            array.shape == (len(structure),)
            and np.isfinite(array).all()
            and np.any(array > 0.0)
            and np.any(array < 0.0)
            and abs(float(array.sum()))
            <= 1.0e-7 * max(1.0, float(np.abs(array).sum()))
        ):
            return ValenceAssignment(
                True,
                tuple(float(value) for value in array),
                policy,
                None,
            )

    try:
        from pymatgen.core.periodic_table import Element

        electronegativity = np.asarray(
            [float(Element(site.specie.symbol).X) for site in structure], dtype=float
        )
    except Exception as exc:
        return ValenceAssignment(
            False,
            None,
            None,
            f"electronegativity partition failed: {type(exc).__name__}",
        )
    if (
        electronegativity.shape != (len(structure),)
        or not np.isfinite(electronegativity).all()
    ):
        return ValenceAssignment(
            False, None, None, "electronegativity values are incomplete"
        )
    raw = float(electronegativity.mean()) - electronegativity
    scale = float(np.max(np.abs(raw))) if len(raw) else 0.0
    tolerance = max(1.0e-12, 1.0e-10 * scale)
    positive = raw > tolerance
    negative = raw < -tolerance
    if not positive.any() or not negative.any():
        detail = integer_error or fractional_error or "formal valence inference returned no assignment"
        return ValenceAssignment(False, None, None, detail)
    assigned = np.zeros(len(raw), dtype=float)
    assigned[positive] = raw[positive] / float(raw[positive].sum())
    assigned[negative] = raw[negative] / float(np.abs(raw[negative]).sum())
    if abs(float(assigned.sum())) > 1.0e-12:
        return ValenceAssignment(
            False, None, None, "electronegativity partition is not neutral"
        )
    return ValenceAssignment(
        True,
        tuple(float(value) for value in assigned),
        "electronegativity_partition",
        None,
    )


def infer_formal_valences(structure) -> tuple[tuple[float, ...] | None, str | None]:
    """Compatibility wrapper returning the analytic valence assignment."""

    assignment = infer_valence_assignment(structure)
    return assignment.values, assignment.failure_reason


def _neighbor_finder(graph_mode: str):
    if graph_mode == "crystalnn":
        from pymatgen.analysis.local_env import CrystalNN

        return CrystalNN(weighted_cn=True, x_diff_weight=0.0)
    if graph_mode == "voronoi":
        from pymatgen.analysis.local_env import VoronoiNN

        return VoronoiNN(weight="solid_angle", tol=0.0, cutoff=13.0)
    raise ValueError(f"unknown graph mode: {graph_mode}")


def build_periodic_edge_geometry(
    structure,
    formal_valences: Sequence[float],
    *,
    graph_mode: str,
) -> PeriodicGeometryResult:
    """Build the expensive periodic neighbor graph once per graph mode."""

    charges = np.asarray(formal_valences, dtype=float)
    if charges.shape != (len(structure),) or not np.isfinite(charges).all():
        raise ValueError("formal valences must match the finite structure sites")
    cations = tuple(int(index) for index in np.flatnonzero(charges > 0.0))
    anions = tuple(int(index) for index in np.flatnonzero(charges < 0.0))
    if not cations or not anions:
        return _unsupported_geometry("formal valences need both signs")
    supply = {index: float(charges[index]) for index in cations}
    demand = {index: float(abs(charges[index])) for index in anions}
    if not math.isclose(
        sum(supply.values()),
        sum(demand.values()),
        rel_tol=1.0e-8,
        abs_tol=1.0e-8,
    ):
        return _unsupported_geometry("formal charge is not neutral")
    working_structure = structure.copy()
    try:
        working_structure.add_oxidation_state_by_site(charges.tolist())
    except Exception as exc:
        return _unsupported_geometry(
            f"valence decoration failed: {type(exc).__name__}"
        )
    finder = _neighbor_finder(graph_mode)
    geometry_edges: list[PeriodicEdgeGeometry] = []
    for cation in cations:
        try:
            neighbor_info = finder.get_nn_info(working_structure, cation)
        except Exception as exc:
            return _unsupported_geometry(
                f"{graph_mode} neighbor construction failed: {type(exc).__name__}"
            )
        raw: dict[tuple[int, tuple[int, int, int]], tuple[float, float]] = {}
        origin = np.asarray(working_structure[cation].coords, dtype=float)
        for info in neighbor_info:
            try:
                anion = int(info["site_index"])
                if anion not in demand:
                    continue
                image = tuple(int(round(float(value))) for value in info["image"])
                if len(image) != 3:
                    continue
                neighbor_coords = np.asarray(info["site"].coords, dtype=float)
                distance = float(np.linalg.norm(neighbor_coords - origin))
                weight = float(info.get("weight", 1.0))
            except (KeyError, TypeError, ValueError):
                continue
            if not (math.isfinite(distance) and distance > 1.0e-8):
                continue
            if not (math.isfinite(weight) and weight > 0.0):
                continue
            key = (anion, image)
            previous = raw.get(key)
            if previous is None or weight > previous[1]:
                raw[key] = (distance, weight)
        if not raw:
            return _unsupported_geometry("cation has no opposite-sign periodic neighbor")
        for (anion, image), (distance, neighbor_weight) in sorted(raw.items()):
            geometry_edges.append(
                PeriodicEdgeGeometry(
                    cation,
                    anion,
                    image,
                    float(distance),
                    float(neighbor_weight),
                )
            )
    return PeriodicGeometryResult(
        supported=True,
        failure_reason=None,
        edges=tuple(sorted(geometry_edges)),
        cation_supply=supply,
        anion_demand=demand,
    )


def edge_priors_from_periodic_geometry(
    geometry: PeriodicGeometryResult, *, alpha: float
) -> GraphPriorResult:
    """Apply one cheap distance-decay setting to cached periodic geometry."""

    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and nonnegative")
    if not geometry.supported:
        return _unsupported_graph(geometry.failure_reason or "periodic graph unsupported")
    priors: list[EdgePrior] = []
    for cation in sorted(geometry.cation_supply):
        selected = [edge for edge in geometry.edges if edge.cation == cation]
        if not selected:
            return _unsupported_graph("cation has no cached periodic edge")
        minimum_distance = min(edge.distance for edge in selected)
        weighted = [
            edge.neighbor_weight
            * math.exp(
                -alpha * max(0.0, edge.distance / minimum_distance - 1.0)
            )
            for edge in selected
        ]
        normalizer = sum(weighted)
        if not math.isfinite(normalizer) or normalizer <= 0.0:
            return _unsupported_graph("cation geometric prior has zero weight")
        for edge, value in zip(selected, weighted, strict=True):
            priors.append(
                EdgePrior(
                    edge.cation,
                    edge.anion,
                    edge.image,
                    float(geometry.cation_supply[cation] * value / normalizer),
                )
            )
    return GraphPriorResult(
        supported=True,
        failure_reason=None,
        edges=tuple(sorted(priors)),
        cation_supply=geometry.cation_supply,
        anion_demand=geometry.anion_demand,
    )


def build_edge_priors(
    structure,
    formal_valences: Sequence[float],
    *,
    graph_mode: str,
    alpha: float,
) -> GraphPriorResult:
    """Build periodic opposite-sign edges and cation-normalized priors."""

    geometry = build_periodic_edge_geometry(
        structure, formal_valences, graph_mode=graph_mode
    )
    return edge_priors_from_periodic_geometry(
        geometry,
        alpha=alpha,
    )


def compute_valence_transport_features(
    structure,
    formal_valences: Sequence[float] | None = None,
    *,
    graph_mode: str,
    alpha: float,
) -> StructureFeatureResult:
    """Compute the exact pure-analytic NEXT19 descriptor schema."""

    if formal_valences is None:
        inferred, error = infer_formal_valences(structure)
        if inferred is None:
            return StructureFeatureResult(False, error, {})
        formal_valences = inferred
    graph = build_edge_priors(
        structure, formal_valences, graph_mode=graph_mode, alpha=alpha
    )
    if not graph.supported:
        return StructureFeatureResult(False, graph.failure_reason, {})
    solution = solve_valence_transport(
        cation_supply=graph.cation_supply,
        anion_demand=graph.anion_demand,
        edges=graph.edges,
    )
    if not solution.supported:
        return StructureFeatureResult(False, solution.failure_reason, {})
    features = {
        "vt_overload": float(solution.overload),
        "vt_reallocation": float(solution.reallocation),
        "vt_anion_mismatch_max": float(solution.max_anion_mismatch),
        "vt_periodic_edge_count": float(len(graph.edges)),
        "vt_cation_count": float(len(graph.cation_supply)),
        "vt_anion_count": float(len(graph.anion_demand)),
    }
    if tuple(features) != FEATURE_NAMES or not np.isfinite(list(features.values())).all():
        return StructureFeatureResult(False, "computed feature schema is invalid", {})
    return StructureFeatureResult(True, None, features)


def _positive_mapping(
    values: Mapping[int, float], *, role: str
) -> tuple[tuple[int, ...], np.ndarray]:
    if not values:
        raise ValueError(f"{role} must not be empty")
    keys = tuple(sorted(int(key) for key in values))
    if len(keys) != len(values):
        raise ValueError(f"{role} keys are not unique integers")
    array = np.asarray([values[key] for key in keys], dtype=float)
    if not np.isfinite(array).all() or np.any(array <= 0.0):
        raise ValueError(f"{role} must contain finite positive values")
    return keys, array


def _canonical_edges(
    edges: Sequence[EdgePrior],
    *,
    cations: set[int],
    anions: set[int],
) -> tuple[EdgePrior, ...]:
    canonical = tuple(
        sorted(
            EdgePrior(
                int(edge.cation),
                int(edge.anion),
                tuple(int(value) for value in edge.image),
                float(edge.prior),
            )
            for edge in edges
        )
    )
    if not canonical:
        raise ValueError("transport graph has no edges")
    identities = [(edge.cation, edge.anion, edge.image) for edge in canonical]
    if len(set(identities)) != len(identities):
        raise ValueError("transport graph contains duplicate periodic edges")
    for edge in canonical:
        if edge.cation not in cations or edge.anion not in anions:
            raise ValueError("transport edge references an unknown endpoint")
        if len(edge.image) != 3:
            raise ValueError("periodic image must have three integers")
        if not math.isfinite(edge.prior) or edge.prior <= 0.0:
            raise ValueError("edge priors must be finite and positive")
    return canonical


def solve_valence_transport(
    *,
    cation_supply: Mapping[int, float],
    anion_demand: Mapping[int, float],
    edges: Sequence[EdgePrior],
    tolerance: float = 1.0e-8,
) -> TransportSolution:
    """Solve minimum edge overload then minimum L1 charge reallocation.

    Invalid numeric input raises ``ValueError``.  A chemically well-formed but
    infeasible graph returns ``supported=False`` so callers can abstain.
    """

    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    cation_keys, supply = _positive_mapping(cation_supply, role="cation supply")
    anion_keys, demand = _positive_mapping(anion_demand, role="anion demand")
    if abs(float(supply.sum() - demand.sum())) > tolerance * max(
        1.0, float(supply.sum()), float(demand.sum())
    ):
        return _unsupported("formal charge is not neutral")
    canonical = _canonical_edges(
        edges, cations=set(cation_keys), anions=set(anion_keys)
    )
    cation_row = {key: index for index, key in enumerate(cation_keys)}
    anion_row = {
        key: len(cation_keys) + index for index, key in enumerate(anion_keys)
    }
    edge_count = len(canonical)
    equality = np.zeros((len(cation_keys) + len(anion_keys), edge_count), dtype=float)
    prior = np.asarray([edge.prior for edge in canonical], dtype=float)
    for column, edge in enumerate(canonical):
        equality[cation_row[edge.cation], column] = 1.0
        equality[anion_row[edge.anion], column] = 1.0
    rhs = np.concatenate([supply, demand])
    prior_by_cation = equality[: len(cation_keys)] @ prior
    if np.any(prior_by_cation <= tolerance):
        return _unsupported("cation has no positive-prior edge")
    if not np.allclose(prior_by_cation, supply, atol=tolerance, rtol=tolerance):
        raise ValueError("edge priors must sum to each cation supply")

    objective = np.zeros(edge_count + 1, dtype=float)
    objective[-1] = 1.0
    equality_with_kappa = np.pad(equality, ((0, 0), (0, 1)))
    capacity = np.zeros((edge_count, edge_count + 1), dtype=float)
    capacity[np.arange(edge_count), np.arange(edge_count)] = 1.0
    capacity[:, -1] = -prior
    first = linprog(
        objective,
        A_ub=capacity,
        b_ub=np.zeros(edge_count, dtype=float),
        A_eq=equality_with_kappa,
        b_eq=rhs,
        bounds=[(0.0, None)] * edge_count + [(1.0, None)],
        method="highs",
    )
    if not first.success or first.x is None:
        return _unsupported(f"overload solve failed: {first.message}")
    kappa = max(1.0, float(first.x[-1]))

    # Minimize sum(u_e) subject to |s_e - p_e| <= u_e and the already
    # minimized overload cap.  The small tolerance avoids false infeasibility
    # from the first LP's floating-point optimum.
    second_objective = np.concatenate(
        [np.zeros(edge_count, dtype=float), np.ones(edge_count, dtype=float)]
    )
    abs_upper = np.zeros((2 * edge_count, 2 * edge_count), dtype=float)
    abs_rhs = np.empty(2 * edge_count, dtype=float)
    for index in range(edge_count):
        abs_upper[2 * index, index] = 1.0
        abs_upper[2 * index, edge_count + index] = -1.0
        abs_rhs[2 * index] = prior[index]
        abs_upper[2 * index + 1, index] = -1.0
        abs_upper[2 * index + 1, edge_count + index] = -1.0
        abs_rhs[2 * index + 1] = -prior[index]
    # HiGHS 1.11 can incorrectly presolve a symmetric network as infeasible
    # when every upper bound is only ~1e-8 above a feasible equality point.
    # A fixed 1e-6 relative slack is far below descriptor precision while
    # keeping the second stage inside the first-stage overload face.
    capacity_slack = max(1.0e-6, 100.0 * tolerance)
    second = linprog(
        second_objective,
        A_ub=abs_upper,
        b_ub=abs_rhs,
        A_eq=np.pad(equality, ((0, 0), (0, edge_count))),
        b_eq=rhs,
        bounds=[(0.0, (kappa + capacity_slack) * value) for value in prior]
        + [(0.0, None)] * edge_count,
        method="highs",
    )
    if not second.success or second.x is None:
        return _unsupported(f"reallocation solve failed: {second.message}")
    flow = np.asarray(second.x[:edge_count], dtype=float)
    absolute_deviation = np.asarray(second.x[edge_count:], dtype=float)
    residual = equality @ flow - rhs
    if np.max(np.abs(residual)) > 100.0 * tolerance:
        return _unsupported("transport equality residual exceeded tolerance")
    total_charge = float(supply.sum())
    reallocation = float(absolute_deviation.sum() / (2.0 * total_charge))
    if reallocation <= 10.0 * tolerance:
        reallocation = 0.0
    prior_anion = equality[len(cation_keys) :] @ prior
    mismatch = float(np.max(np.abs(prior_anion - demand) / demand))
    return TransportSolution(
        supported=True,
        failure_reason=None,
        overload=max(0.0, kappa - 1.0),
        reallocation=max(0.0, reallocation),
        max_anion_mismatch=max(0.0, mismatch),
        kappa=kappa,
        edge_flow=tuple(float(value) for value in flow),
        canonical_edges=canonical,
    )


validate_feature_names(FEATURE_NAMES)
