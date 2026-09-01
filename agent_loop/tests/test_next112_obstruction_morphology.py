from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from src.next112_obstruction_morphology import (
    FEATURE_NAMES,
    PROTOCOL,
    ConvexMixedValenceObstructionMorphologyFeatureResult,
    MixedValenceObstructionMorphologyResult,
    compute_obstruction_morphology,
    solve_obstruction_morphology,
)


def _hall_obstructed_network() -> tuple[list[tuple[float, float]], np.ndarray]:
    return (
        [(2.0, 2.0), (1.0, 1.0), (-1.0, -1.0), (-2.0, -2.0)],
        np.asarray([(0, 2), (1, 2), (1, 3)], dtype=int),
    )


def _assert_morphology_bounded(result: MixedValenceObstructionMorphologyResult) -> None:
    assert result.supported, result.failure_reason
    assert tuple(result.morphology) == FEATURE_NAMES
    assert all(0.0 <= value <= 1.0 for value in result.morphology.values())


def test_feasible_network_has_zero_obstruction_morphology() -> None:
    result = solve_obstruction_morphology(
        signed_charge_bounds=[(1.0, 1.0), (-1.0, -1.0)],
        endpoints=np.asarray([(0, 1)], dtype=int),
    )

    assert isinstance(result, MixedValenceObstructionMorphologyResult)
    _assert_morphology_bounded(result)
    assert result.min_interval_slack == pytest.approx(0.0, abs=1.0e-10)
    assert result.global_balance_gap == pytest.approx(0.0, abs=1.0e-12)
    assert result.component_balance_gap == pytest.approx(0.0, abs=1.0e-12)
    assert result.unserved_site_fraction == pytest.approx(0.0, abs=1.0e-12)
    assert result.morphology == dict.fromkeys(FEATURE_NAMES, 0.0)


def test_global_imbalance_has_forced_one_side_slack() -> None:
    result = solve_obstruction_morphology(
        signed_charge_bounds=[(1.0, 1.0), (-2.0, -2.0)],
        endpoints=np.asarray([(0, 1)], dtype=int),
    )

    _assert_morphology_bounded(result)
    assert result.min_interval_slack == pytest.approx(0.25, abs=1.0e-9)
    assert result.morphology["cmvom_component_gap_site_mean"] == pytest.approx(0.5)
    assert result.morphology["cmvom_component_gap_site_rms"] == pytest.approx(0.5)
    assert result.morphology["cmvom_obstructed_site_fraction"] == pytest.approx(1.0)
    assert result.morphology["cmvom_localized_slack_severity"] == pytest.approx(0.125)
    assert result.morphology["cmvom_side_slack_asymmetry"] == pytest.approx(0.25)
    assert result.morphology["cmvom_side_slack_flexibility"] == pytest.approx(0.0)


def test_connected_hall_cut_is_local_without_component_imbalance() -> None:
    bounds, endpoints = _hall_obstructed_network()
    result = solve_obstruction_morphology(
        signed_charge_bounds=bounds,
        endpoints=endpoints,
    )

    _assert_morphology_bounded(result)
    assert result.min_interval_slack == pytest.approx(0.25, abs=1.0e-9)
    assert result.global_balance_gap == pytest.approx(0.0, abs=1.0e-12)
    assert result.component_balance_gap == pytest.approx(0.0, abs=1.0e-12)
    assert result.morphology["cmvom_component_gap_site_mean"] == pytest.approx(0.0)
    assert result.morphology["cmvom_component_gap_site_rms"] == pytest.approx(0.0)
    assert result.morphology["cmvom_obstructed_site_fraction"] == pytest.approx(0.0)
    assert result.morphology["cmvom_localized_slack_severity"] == pytest.approx(0.125)
    assert result.morphology["cmvom_side_slack_asymmetry"] == pytest.approx(0.0)


def test_disconnected_imbalance_is_widespread() -> None:
    result = solve_obstruction_morphology(
        signed_charge_bounds=[
            (1.0, 1.0),
            (2.0, 2.0),
            (-2.0, -2.0),
            (-1.0, -1.0),
        ],
        endpoints=np.asarray([(0, 2), (1, 3)], dtype=int),
    )

    _assert_morphology_bounded(result)
    assert result.morphology["cmvom_component_gap_site_mean"] == pytest.approx(0.5)
    assert result.morphology["cmvom_component_gap_site_rms"] == pytest.approx(0.5)
    assert result.morphology["cmvom_obstructed_site_fraction"] == pytest.approx(1.0)
    assert result.morphology["cmvom_localized_slack_severity"] == pytest.approx(0.125)
    assert result.morphology["cmvom_side_slack_asymmetry"] == pytest.approx(0.0)


def test_isolated_site_contributes_site_weighted_component_morphology() -> None:
    result = solve_obstruction_morphology(
        signed_charge_bounds=[(1.0, 1.0), (1.0, 1.0), (-2.0, -2.0)],
        endpoints=np.asarray([(0, 2)], dtype=int),
    )

    _assert_morphology_bounded(result)
    assert result.morphology["cmvom_component_gap_site_mean"] == pytest.approx(2.0 / 3.0)
    assert result.morphology["cmvom_component_gap_site_rms"] == pytest.approx(np.sqrt(0.5))
    assert result.morphology["cmvom_obstructed_site_fraction"] == pytest.approx(1.0)
    assert result.morphology["cmvom_localized_slack_severity"] == pytest.approx(1.0 / 6.0)
    assert result.morphology["cmvom_side_slack_asymmetry"] == pytest.approx(0.5)


def test_empty_graph_is_maximally_widespread_without_secondary_face() -> None:
    result = solve_obstruction_morphology(
        signed_charge_bounds=[(1.0, 1.0), (-1.0, -1.0)],
        endpoints=np.empty((0, 2), dtype=int),
    )

    _assert_morphology_bounded(result)
    assert result.min_interval_slack == pytest.approx(1.0)
    assert result.morphology == {
        "cmvom_component_gap_site_mean": 1.0,
        "cmvom_component_gap_site_rms": 1.0,
        "cmvom_obstructed_site_fraction": 1.0,
        "cmvom_localized_slack_severity": 0.0,
        "cmvom_side_slack_asymmetry": 0.0,
        "cmvom_side_slack_flexibility": 0.0,
    }


def test_side_slack_range_is_an_optimum_value_not_one_solver_vector() -> None:
    result = solve_obstruction_morphology(
        signed_charge_bounds=[(1.0, 1.0), (1.0, 1.0), (-1.0, -1.0), (-4.0, -3.0)],
        endpoints=np.asarray([(0, 2), (1, 2)], dtype=int),
    )

    _assert_morphology_bounded(result)
    assert result.min_interval_slack == pytest.approx(1.0)
    assert result.morphology["cmvom_side_slack_flexibility"] == pytest.approx(0.5)


def test_graph_features_are_permutation_replication_and_charge_scale_invariant() -> None:
    bounds, endpoints = _hall_obstructed_network()
    reference = solve_obstruction_morphology(
        signed_charge_bounds=bounds,
        endpoints=endpoints,
    )

    old_to_new = np.asarray([1, 0, 3, 2], dtype=int)
    permuted_bounds: list[tuple[float, float] | None] = [None] * len(bounds)
    for old, new in enumerate(old_to_new):
        permuted_bounds[int(new)] = bounds[old]
    permuted = solve_obstruction_morphology(
        signed_charge_bounds=[value for value in permuted_bounds if value is not None],
        endpoints=old_to_new[endpoints][np.asarray([2, 0, 1])],
    )
    replicated = solve_obstruction_morphology(
        signed_charge_bounds=bounds + bounds,
        endpoints=np.vstack([endpoints, endpoints + len(bounds)]),
    )
    scaled = solve_obstruction_morphology(
        signed_charge_bounds=[(7.0 * lower, 7.0 * upper) for lower, upper in bounds],
        endpoints=endpoints,
    )

    for observed in (permuted, replicated, scaled):
        _assert_morphology_bounded(observed)
        assert observed.min_interval_slack == pytest.approx(
            reference.min_interval_slack, abs=1.0e-9
        )
        assert observed.global_balance_gap == pytest.approx(
            reference.global_balance_gap, abs=1.0e-12
        )
        assert observed.component_balance_gap == pytest.approx(
            reference.component_balance_gap, abs=1.0e-12
        )
        assert observed.unserved_site_fraction == pytest.approx(
            reference.unserved_site_fraction, abs=1.0e-12
        )
        for name in FEATURE_NAMES:
            assert observed.morphology[name] == pytest.approx(
                reference.morphology[name], abs=1.0e-9
            )


def _binary_structure(left: str, right: str) -> Structure:
    return Structure(
        Lattice.cubic(4.2),
        [left, right],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def test_schema_stays_inside_the_strict_no_dft_boundary() -> None:
    forbidden = ("energy", "force", "stress", "relax", "dft", "model", "proxy", "mattersim")
    assert PROTOCOL == "2026-08-08-next112-obstruction-morphology-v1"
    assert FEATURE_NAMES == (
        "cmvom_component_gap_site_mean",
        "cmvom_component_gap_site_rms",
        "cmvom_obstructed_site_fraction",
        "cmvom_localized_slack_severity",
        "cmvom_side_slack_asymmetry",
        "cmvom_side_slack_flexibility",
    )
    assert not any(token in name.lower() for name in FEATURE_NAMES for token in forbidden)


def test_structure_evaluation_is_pure_finite_and_deterministic() -> None:
    structure = _binary_structure("Na", "O")
    species = tuple(str(site.specie) for site in structure)
    lattice = structure.lattice.matrix.copy()
    coordinates = structure.frac_coords.copy()

    first = compute_obstruction_morphology(
        structure, graph_mode="voronoi", catalogue_mode="core"
    )
    second = compute_obstruction_morphology(
        structure, graph_mode="voronoi", catalogue_mode="core"
    )

    assert isinstance(first, ConvexMixedValenceObstructionMorphologyFeatureResult)
    assert first == second
    assert first.supported, first.failure_reason
    assert tuple(first.features) == FEATURE_NAMES
    assert all(0.0 <= value <= 1.0 for value in first.features.values())
    assert len(first.catalogue_sha256) == 64
    assert first.pymatgen_version and first.scipy_version
    assert tuple(str(site.specie) for site in structure) == species
    assert np.array_equal(structure.lattice.matrix, lattice)
    assert np.array_equal(structure.frac_coords, coordinates)


def test_sign_pattern_bound_abstains_without_truncating() -> None:
    structure = Structure(
        Lattice.cubic(7.0),
        ["B", "C", "N"],
        [[0.0, 0.0, 0.0], [0.35, 0.35, 0.35], [0.7, 0.7, 0.7]],
    )

    result = compute_obstruction_morphology(
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

    small = compute_obstruction_morphology(
        primitive, graph_mode="voronoi", catalogue_mode="core"
    )
    large = compute_obstruction_morphology(
        supercell, graph_mode="voronoi", catalogue_mode="core"
    )

    assert small.supported, small.failure_reason
    assert large.supported, large.failure_reason
    for name in FEATURE_NAMES:
        assert large.features[name] == pytest.approx(
            small.features[name], rel=1.0e-8, abs=1.0e-9
        )


def test_structure_morphology_does_not_consult_brown_parameters(monkeypatch) -> None:
    import src.next104_convex_mixed_valence_flow as next104

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Brown parameters must stay outside NEXT112")

    monkeypatch.setattr(next104, "_brown_generic_strengths", fail_if_called)
    result = compute_obstruction_morphology(
        _binary_structure("Na", "O"), graph_mode="voronoi", catalogue_mode="core"
    )

    assert result.supported, result.failure_reason
