from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from src.next109_convex_mixed_valence_obstruction import (
    FEATURE_NAMES,
    PROTOCOL,
    ConvexMixedValenceObstructionFeatureResult,
    MixedValenceObstructionResult,
    compute_convex_mixed_valence_obstruction,
    solve_mixed_valence_obstruction,
)


def test_solver_returns_zero_for_an_exactly_feasible_network() -> None:
    result = solve_mixed_valence_obstruction(
        signed_charge_bounds=[(1.0, 1.0), (-1.0, -1.0)],
        endpoints=np.asarray([(0, 1)], dtype=int),
    )

    assert isinstance(result, MixedValenceObstructionResult)
    assert result.supported, result.failure_reason
    assert result.min_interval_slack == pytest.approx(0.0, abs=1.0e-10)
    assert result.global_balance_gap == pytest.approx(0.0, abs=1.0e-12)
    assert result.component_balance_gap == pytest.approx(0.0, abs=1.0e-12)
    assert result.unserved_site_fraction == pytest.approx(0.0, abs=1.0e-12)


def test_solver_quantifies_global_charge_interval_imbalance() -> None:
    result = solve_mixed_valence_obstruction(
        signed_charge_bounds=[(1.0, 1.0), (-2.0, -2.0)],
        endpoints=np.asarray([(0, 1)], dtype=int),
    )

    assert result.supported, result.failure_reason
    assert result.min_interval_slack == pytest.approx(0.25, abs=1.0e-9)
    assert result.global_balance_gap == pytest.approx(0.5, abs=1.0e-12)
    assert result.component_balance_gap == pytest.approx(0.5, abs=1.0e-12)
    assert result.unserved_site_fraction == pytest.approx(0.0, abs=1.0e-12)


def _hall_obstructed_network() -> tuple[list[tuple[float, float]], np.ndarray]:
    # Total +3/-3 charge and one connected component are both balanced, but
    # the +2 site can reach only the -1 site.  This is a genuine subset cut.
    return (
        [(2.0, 2.0), (1.0, 1.0), (-1.0, -1.0), (-2.0, -2.0)],
        np.asarray([(0, 2), (1, 2), (1, 3)], dtype=int),
    )


def test_solver_detects_disconnected_component_imbalance() -> None:
    result = solve_mixed_valence_obstruction(
        signed_charge_bounds=[
            (1.0, 1.0),
            (2.0, 2.0),
            (-2.0, -2.0),
            (-1.0, -1.0),
        ],
        endpoints=np.asarray([(0, 2), (1, 3)], dtype=int),
    )

    assert result.supported, result.failure_reason
    assert result.global_balance_gap == pytest.approx(0.0, abs=1.0e-12)
    assert result.component_balance_gap == pytest.approx(0.5, abs=1.0e-12)
    assert result.unserved_site_fraction == pytest.approx(0.0, abs=1.0e-12)
    assert 0.0 < result.min_interval_slack <= 1.0


def test_solver_detects_a_connected_hall_cut_obstruction() -> None:
    bounds, endpoints = _hall_obstructed_network()
    result = solve_mixed_valence_obstruction(
        signed_charge_bounds=bounds,
        endpoints=endpoints,
    )

    assert result.supported, result.failure_reason
    assert result.global_balance_gap == pytest.approx(0.0, abs=1.0e-12)
    assert result.component_balance_gap == pytest.approx(0.0, abs=1.0e-12)
    assert result.unserved_site_fraction == pytest.approx(0.0, abs=1.0e-12)
    assert 0.0 < result.min_interval_slack <= 1.0


def test_solver_exposes_an_isolated_site_without_abstaining() -> None:
    result = solve_mixed_valence_obstruction(
        signed_charge_bounds=[(1.0, 1.0), (1.0, 1.0), (-2.0, -2.0)],
        endpoints=np.asarray([(0, 2)], dtype=int),
    )

    assert result.supported, result.failure_reason
    assert result.global_balance_gap == pytest.approx(0.0, abs=1.0e-12)
    assert result.component_balance_gap == pytest.approx(1.0, abs=1.0e-12)
    assert result.unserved_site_fraction == pytest.approx(1.0 / 3.0, abs=1.0e-12)
    assert 0.0 < result.min_interval_slack <= 1.0


def test_solver_assigns_maximum_topological_obstruction_to_an_empty_graph() -> None:
    result = solve_mixed_valence_obstruction(
        signed_charge_bounds=[(1.0, 1.0), (-1.0, -1.0)],
        endpoints=np.empty((0, 2), dtype=int),
    )

    assert result.supported, result.failure_reason
    assert result.min_interval_slack == pytest.approx(1.0, abs=1.0e-12)
    assert result.global_balance_gap == pytest.approx(0.0, abs=1.0e-12)
    assert result.component_balance_gap == pytest.approx(1.0, abs=1.0e-12)
    assert result.unserved_site_fraction == pytest.approx(1.0, abs=1.0e-12)


def test_solver_is_invariant_to_site_and_edge_permutation() -> None:
    bounds, endpoints = _hall_obstructed_network()
    reference = solve_mixed_valence_obstruction(
        signed_charge_bounds=bounds,
        endpoints=endpoints,
    )
    old_to_new = np.asarray([1, 0, 3, 2], dtype=int)
    permuted_bounds: list[tuple[float, float] | None] = [None] * len(bounds)
    for old, new in enumerate(old_to_new):
        permuted_bounds[int(new)] = bounds[old]
    permuted = solve_mixed_valence_obstruction(
        signed_charge_bounds=[value for value in permuted_bounds if value is not None],
        endpoints=old_to_new[endpoints][np.asarray([2, 0, 1])],
    )

    assert reference.supported and permuted.supported
    assert permuted.min_interval_slack == pytest.approx(
        reference.min_interval_slack, abs=1.0e-10
    )
    assert permuted.global_balance_gap == pytest.approx(
        reference.global_balance_gap, abs=1.0e-12
    )
    assert permuted.component_balance_gap == pytest.approx(
        reference.component_balance_gap, abs=1.0e-12
    )
    assert permuted.unserved_site_fraction == pytest.approx(
        reference.unserved_site_fraction, abs=1.0e-12
    )


def test_solver_is_invariant_to_integer_supercell_replication() -> None:
    bounds, endpoints = _hall_obstructed_network()
    primitive = solve_mixed_valence_obstruction(
        signed_charge_bounds=bounds,
        endpoints=endpoints,
    )
    replicated = solve_mixed_valence_obstruction(
        signed_charge_bounds=bounds + bounds,
        endpoints=np.vstack([endpoints, endpoints + len(bounds)]),
    )

    assert primitive.supported and replicated.supported
    assert replicated.min_interval_slack == pytest.approx(
        primitive.min_interval_slack, abs=1.0e-10
    )
    assert replicated.global_balance_gap == pytest.approx(
        primitive.global_balance_gap, abs=1.0e-12
    )
    assert replicated.component_balance_gap == pytest.approx(
        primitive.component_balance_gap, abs=1.0e-12
    )
    assert replicated.unserved_site_fraction == pytest.approx(
        primitive.unserved_site_fraction, abs=1.0e-12
    )


def test_solver_is_invariant_to_common_charge_unit_scaling() -> None:
    bounds, endpoints = _hall_obstructed_network()
    reference = solve_mixed_valence_obstruction(
        signed_charge_bounds=bounds,
        endpoints=endpoints,
    )
    scaled = solve_mixed_valence_obstruction(
        signed_charge_bounds=[(7.0 * lower, 7.0 * upper) for lower, upper in bounds],
        endpoints=endpoints,
    )

    assert reference.supported and scaled.supported
    assert scaled.min_interval_slack == pytest.approx(
        reference.min_interval_slack, abs=1.0e-10
    )
    assert scaled.global_balance_gap == pytest.approx(
        reference.global_balance_gap, abs=1.0e-12
    )
    assert scaled.component_balance_gap == pytest.approx(
        reference.component_balance_gap, abs=1.0e-12
    )
    assert scaled.unserved_site_fraction == pytest.approx(
        reference.unserved_site_fraction, abs=1.0e-12
    )


def _binary_structure(left: str, right: str) -> Structure:
    return Structure(
        Lattice.cubic(4.2),
        [left, right],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def test_structure_schema_stays_inside_the_strict_no_dft_boundary() -> None:
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
    assert PROTOCOL == "2026-08-08-next109-convex-mixed-valence-obstruction-v1"
    assert FEATURE_NAMES == (
        "cmvo_min_interval_slack",
        "cmvo_global_balance_gap",
        "cmvo_component_balance_gap",
        "cmvo_unserved_site_fraction",
    )
    assert not any(
        token in name.lower() for name in FEATURE_NAMES for token in forbidden
    )


def test_structure_evaluation_is_pure_finite_and_deterministic() -> None:
    structure = _binary_structure("Cs", "Cl")
    species = tuple(str(site.specie) for site in structure)
    lattice = structure.lattice.matrix.copy()
    coordinates = structure.frac_coords.copy()

    first = compute_convex_mixed_valence_obstruction(
        structure, graph_mode="voronoi", catalogue_mode="core"
    )
    second = compute_convex_mixed_valence_obstruction(
        structure, graph_mode="voronoi", catalogue_mode="core"
    )

    assert isinstance(first, ConvexMixedValenceObstructionFeatureResult)
    assert first == second
    assert first.supported, first.failure_reason
    assert tuple(first.features) == FEATURE_NAMES
    assert all(np.isfinite(value) for value in first.features.values())
    assert first.features["cmvo_min_interval_slack"] == pytest.approx(
        0.0, abs=1.0e-9
    )
    assert len(first.catalogue_sha256) == 64
    assert first.pymatgen_version
    assert first.scipy_version
    assert tuple(str(site.specie) for site in structure) == species
    assert np.array_equal(structure.lattice.matrix, lattice)
    assert np.array_equal(structure.frac_coords, coordinates)


def test_expanded_catalogue_can_rescue_core_nao_interval_obstruction() -> None:
    structure = _binary_structure("Na", "O")

    core = compute_convex_mixed_valence_obstruction(
        structure, graph_mode="voronoi", catalogue_mode="core"
    )
    expanded = compute_convex_mixed_valence_obstruction(
        structure, graph_mode="voronoi", catalogue_mode="expanded"
    )

    assert core.supported, core.failure_reason
    assert expanded.supported, expanded.failure_reason
    assert core.features["cmvo_global_balance_gap"] > 0.0
    assert core.features["cmvo_min_interval_slack"] > 0.0
    assert expanded.features["cmvo_global_balance_gap"] == pytest.approx(
        0.0, abs=1.0e-12
    )
    assert expanded.features["cmvo_min_interval_slack"] == pytest.approx(
        0.0, abs=1.0e-9
    )
    assert core.catalogue_sha256 != expanded.catalogue_sha256


def test_sign_pattern_bound_abstains_without_truncating() -> None:
    structure = Structure(
        Lattice.cubic(7.0),
        ["B", "C", "N"],
        [[0.0, 0.0, 0.0], [0.35, 0.35, 0.35], [0.7, 0.7, 0.7]],
    )

    result = compute_convex_mixed_valence_obstruction(
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
    primitive = _binary_structure("Na", "O")
    supercell = primitive.copy()
    supercell.make_supercell([2, 1, 1])

    small = compute_convex_mixed_valence_obstruction(
        primitive, graph_mode="voronoi", catalogue_mode="core"
    )
    large = compute_convex_mixed_valence_obstruction(
        supercell, graph_mode="voronoi", catalogue_mode="core"
    )

    assert small.supported, small.failure_reason
    assert large.supported, large.failure_reason
    for name in FEATURE_NAMES:
        assert large.features[name] == pytest.approx(
            small.features[name], rel=1.0e-8, abs=1.0e-9
        )


def test_structure_obstruction_does_not_consult_brown_parameters(monkeypatch) -> None:
    import src.next104_convex_mixed_valence_flow as next104

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Brown parameters must stay outside NEXT109")

    monkeypatch.setattr(next104, "_brown_generic_strengths", fail_if_called)
    result = compute_convex_mixed_valence_obstruction(
        _binary_structure("Cs", "Cl"),
        graph_mode="voronoi",
        catalogue_mode="core",
    )

    assert result.supported, result.failure_reason
