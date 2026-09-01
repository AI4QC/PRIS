from __future__ import annotations

import math

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from src.next104_convex_mixed_valence_flow import (
    FEATURE_NAMES,
    PROTOCOL,
    ConvexMixedValenceFeatureResult,
    ConvexMixedValenceFlowResult,
    compute_convex_mixed_valence_flow,
    solve_convex_mixed_valence_flow,
)


def _mixed_network() -> tuple[list[tuple[float, float]], np.ndarray, np.ndarray]:
    bounds = [(2.0, 3.0)] * 3 + [(-2.0, -2.0)] * 4
    endpoints = np.asarray(
        [(cation, anion) for cation in range(3) for anion in range(3, 7)],
        dtype=int,
    )
    return bounds, endpoints, np.ones(len(endpoints), dtype=float)


def test_solver_accepts_exactly_compatible_mixed_valence_network() -> None:
    # Three Fe sites carry the convex mixed value +8/3 while four O sites
    # remain -2.  No element-uniform integer Fe state can realize Fe3O4.
    bounds, endpoints, priors = _mixed_network()
    result = solve_convex_mixed_valence_flow(
        signed_charge_bounds=bounds,
        endpoints=endpoints,
        raw_priors=priors,
    )

    assert isinstance(result, ConvexMixedValenceFlowResult)
    assert result.supported, result.failure_reason
    assert result.reallocation == pytest.approx(0.0, abs=1.0e-10)
    assert result.overload == pytest.approx(0.0, abs=1.0e-9)
    assert result.log_scale_mismatch == pytest.approx(
        abs(math.log(8.0 / 12.0)), abs=1.0e-9
    )


def test_solver_abstains_when_charge_intervals_cannot_balance() -> None:
    result = solve_convex_mixed_valence_flow(
        signed_charge_bounds=[(1.0, 1.0), (-2.0, -2.0)],
        endpoints=np.asarray([(0, 1)], dtype=int),
        raw_priors=np.asarray([1.0]),
    )

    assert not result.supported
    assert result.reallocation is None
    assert result.overload is None
    assert result.log_scale_mismatch is None
    assert "balance" in str(result.failure_reason).lower()


def test_solver_abstains_when_an_opposite_sign_site_is_isolated() -> None:
    result = solve_convex_mixed_valence_flow(
        signed_charge_bounds=[(1.0, 1.0), (-1.0, -1.0), (-1.0, -1.0)],
        endpoints=np.asarray([(0, 1)], dtype=int),
        raw_priors=np.asarray([1.0]),
    )

    assert not result.supported
    assert "isolated" in str(result.failure_reason).lower()


def test_solver_is_invariant_to_site_and_edge_permutation() -> None:
    bounds, endpoints, priors = _mixed_network()
    reference = solve_convex_mixed_valence_flow(
        signed_charge_bounds=bounds,
        endpoints=endpoints,
        raw_priors=priors,
    )
    old_to_new = np.asarray([2, 0, 1, 6, 3, 5, 4], dtype=int)
    permuted_bounds: list[tuple[float, float] | None] = [None] * len(bounds)
    for old, new in enumerate(old_to_new):
        permuted_bounds[int(new)] = bounds[old]
    edge_order = np.asarray([11, 3, 8, 0, 6, 10, 2, 9, 5, 1, 7, 4])
    permuted = solve_convex_mixed_valence_flow(
        signed_charge_bounds=[value for value in permuted_bounds if value is not None],
        endpoints=old_to_new[endpoints][edge_order],
        raw_priors=priors[edge_order],
    )

    assert reference.supported and permuted.supported
    assert permuted.reallocation == pytest.approx(reference.reallocation, abs=1.0e-10)
    assert permuted.overload == pytest.approx(reference.overload, abs=1.0e-9)
    assert permuted.log_scale_mismatch == pytest.approx(
        reference.log_scale_mismatch, abs=1.0e-9
    )


def test_solver_is_invariant_to_integer_supercell_replication() -> None:
    bounds, endpoints, priors = _mixed_network()
    primitive = solve_convex_mixed_valence_flow(
        signed_charge_bounds=bounds,
        endpoints=endpoints,
        raw_priors=priors,
    )
    replicated = solve_convex_mixed_valence_flow(
        signed_charge_bounds=bounds + bounds,
        endpoints=np.vstack([endpoints, endpoints + len(bounds)]),
        raw_priors=np.concatenate([priors, priors]),
    )

    assert primitive.supported and replicated.supported
    assert replicated.reallocation == pytest.approx(primitive.reallocation, abs=1.0e-10)
    assert replicated.overload == pytest.approx(primitive.overload, abs=1.0e-9)
    assert replicated.log_scale_mismatch == pytest.approx(
        primitive.log_scale_mismatch, abs=1.0e-9
    )


def test_common_prior_scaling_changes_only_the_raw_scale_mismatch() -> None:
    bounds, endpoints, priors = _mixed_network()
    reference = solve_convex_mixed_valence_flow(
        signed_charge_bounds=bounds,
        endpoints=endpoints,
        raw_priors=priors,
    )
    doubled = solve_convex_mixed_valence_flow(
        signed_charge_bounds=bounds,
        endpoints=endpoints,
        raw_priors=2.0 * priors,
    )

    assert reference.supported and doubled.supported
    assert doubled.reallocation == pytest.approx(reference.reallocation, abs=1.0e-10)
    assert doubled.overload == pytest.approx(reference.overload, abs=1.0e-9)
    assert doubled.log_scale_mismatch == pytest.approx(
        reference.log_scale_mismatch + math.log(2.0), abs=1.0e-9
    )


def _binary_structure(left: str, right: str) -> Structure:
    return Structure(
        Lattice.cubic(4.2),
        [left, right],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def test_structure_schema_stays_inside_the_no_dft_boundary() -> None:
    forbidden = (
        "energy",
        "force",
        "stress",
        "relax",
        "dft",
        "model",
        "proxy",
        "mattersim",
    )
    assert PROTOCOL == "2026-08-04-next104-convex-mixed-valence-flow-v1"
    assert FEATURE_NAMES == (
        "cmvf_reallocation",
        "cmvf_overload",
        "cmvf_log_scale_mismatch",
        "cmvf_domain_width_mean",
        "cmvf_domain_width_max",
        "cmvf_sign_pattern_log_count",
    )
    assert not any(
        token in name.lower() for name in FEATURE_NAMES for token in forbidden
    )


def test_structure_evaluation_is_pure_finite_and_deterministic() -> None:
    structure = _binary_structure("Cs", "Cl")
    species = tuple(str(site.specie) for site in structure)
    lattice = structure.lattice.matrix.copy()
    coordinates = structure.frac_coords.copy()

    first = compute_convex_mixed_valence_flow(
        structure, graph_mode="voronoi", catalogue_mode="core"
    )
    second = compute_convex_mixed_valence_flow(
        structure, graph_mode="voronoi", catalogue_mode="core"
    )

    assert isinstance(first, ConvexMixedValenceFeatureResult)
    assert first == second
    assert first.supported, first.failure_reason
    assert tuple(first.features) == FEATURE_NAMES
    assert all(math.isfinite(value) for value in first.features.values())
    assert first.features["cmvf_domain_width_mean"] == 0.0
    assert len(first.catalogue_sha256) == 64
    assert first.pymatgen_version
    assert first.scipy_version
    assert tuple(str(site.specie) for site in structure) == species
    assert np.array_equal(structure.lattice.matrix, lattice)
    assert np.array_equal(structure.frac_coords, coordinates)


def test_expanded_catalogue_supports_nao_without_changing_core_result() -> None:
    structure = _binary_structure("Na", "O")

    core = compute_convex_mixed_valence_flow(
        structure, graph_mode="voronoi", catalogue_mode="core"
    )
    expanded = compute_convex_mixed_valence_flow(
        structure, graph_mode="voronoi", catalogue_mode="expanded"
    )

    assert not core.supported
    assert expanded.supported, expanded.failure_reason
    assert expanded.features["cmvf_domain_width_mean"] >= 0.0
    assert core.catalogue_sha256 != expanded.catalogue_sha256


def test_sign_pattern_bound_abstains_without_truncating() -> None:
    structure = Structure(
        Lattice.cubic(7.0),
        ["B", "C", "N"],
        [[0.0, 0.0, 0.0], [0.35, 0.35, 0.35], [0.7, 0.7, 0.7]],
    )

    result = compute_convex_mixed_valence_flow(
        structure,
        graph_mode="voronoi",
        catalogue_mode="expanded",
        max_sign_patterns=1,
    )

    assert not result.supported
    assert result.features == {}
    assert "sign pattern" in str(result.failure_reason).lower()
    assert "exceeds 1" in str(result.failure_reason).lower()


def test_structure_features_are_supercell_invariant() -> None:
    primitive = _binary_structure("Cs", "Cl")
    supercell = primitive.copy()
    supercell.make_supercell([2, 1, 1])

    small = compute_convex_mixed_valence_flow(
        primitive, graph_mode="voronoi", catalogue_mode="core"
    )
    large = compute_convex_mixed_valence_flow(
        supercell, graph_mode="voronoi", catalogue_mode="core"
    )

    assert small.supported, small.failure_reason
    assert large.supported, large.failure_reason
    for name in FEATURE_NAMES:
        assert large.features[name] == pytest.approx(
            small.features[name], rel=1.0e-8, abs=1.0e-9
        )
