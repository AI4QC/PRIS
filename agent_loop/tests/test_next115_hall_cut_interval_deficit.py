from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from scipy.optimize import linprog

from src.next115_hall_cut_interval_deficit import (
    FEATURE_NAMES,
    PROTOCOL,
    HallCutIntervalDeficitFeatureResult,
    HallCutIntervalDeficitResult,
    compute_hall_cut_interval_deficit,
    solve_hall_cut_interval_deficit,
)


def _assert_bounded(result: HallCutIntervalDeficitResult) -> None:
    assert result.supported, result.failure_reason
    assert tuple(result.features) == FEATURE_NAMES
    assert all(0.0 <= value <= 1.0 for value in result.features.values())


def _side(result: HallCutIntervalDeficitResult, side: str) -> tuple[float, ...]:
    return tuple(
        result.features[f"hcid_{side}_{suffix}"]
        for suffix in (
            "global_deficit",
            "local_density",
            "origin_site_fraction_min",
            "origin_site_fraction_max",
            "neighbor_site_fraction_min",
        )
    )


def test_feasible_edge_has_zero_directional_certificates() -> None:
    result = solve_hall_cut_interval_deficit(
        signed_charge_bounds=[(1.0, 1.0), (-1.0, -1.0)],
        endpoints=np.asarray([(0, 1)], dtype=int),
    )

    assert isinstance(result, HallCutIntervalDeficitResult)
    _assert_bounded(result)
    assert result.positive_max_deficit == pytest.approx(0.0, abs=1.0e-10)
    assert result.negative_max_deficit == pytest.approx(0.0, abs=1.0e-10)
    assert result.features == dict.fromkeys(FEATURE_NAMES, 0.0)


def test_global_mismatch_is_directional_and_canonical() -> None:
    result = solve_hall_cut_interval_deficit(
        signed_charge_bounds=[(1.0, 1.0), (-2.0, -2.0)],
        endpoints=np.asarray([(0, 1)], dtype=int),
    )

    _assert_bounded(result)
    assert result.positive_max_deficit == pytest.approx(0.0)
    assert result.negative_max_deficit == pytest.approx(1.0)
    assert _side(result, "positive") == pytest.approx((0, 0, 0, 0, 0))
    assert _side(result, "negative") == pytest.approx((0.5, 0.5, 1, 1, 1))


def test_connected_hall_cut_retains_local_subset_scale() -> None:
    result = solve_hall_cut_interval_deficit(
        signed_charge_bounds=[
            (2.0, 2.0),
            (1.0, 1.0),
            (-1.0, -1.0),
            (-2.0, -2.0),
        ],
        endpoints=np.asarray([(0, 2), (1, 2), (1, 3)], dtype=int),
    )

    _assert_bounded(result)
    assert result.positive_max_deficit == pytest.approx(1.0)
    assert result.negative_max_deficit == pytest.approx(1.0)
    assert _side(result, "positive") == pytest.approx((1 / 3, 1 / 2, 1 / 2, 1 / 2, 1 / 2))
    assert _side(result, "negative") == pytest.approx((1 / 3, 1 / 2, 1 / 2, 1 / 2, 1 / 2))


def test_empty_graph_has_unit_deficits_and_zero_neighbor_support() -> None:
    result = solve_hall_cut_interval_deficit(
        signed_charge_bounds=[(1.0, 1.0), (-2.0, -2.0)],
        endpoints=np.empty((0, 2), dtype=int),
    )

    _assert_bounded(result)
    assert result.positive_max_deficit == pytest.approx(1.0)
    assert result.negative_max_deficit == pytest.approx(2.0)
    assert _side(result, "positive") == pytest.approx((1, 1, 1, 1, 0))
    assert _side(result, "negative") == pytest.approx((1, 1, 1, 1, 0))


def test_graph_input_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="same-sign"):
        solve_hall_cut_interval_deficit(
            signed_charge_bounds=[(-1.0, 1.0), (-1.0, -1.0)],
            endpoints=np.asarray([(0, 1)], dtype=int),
        )
    with pytest.raises(ValueError, match="ordered"):
        solve_hall_cut_interval_deficit(
            signed_charge_bounds=[(1.0, 1.0), (-1.0, -1.0)],
            endpoints=np.asarray([(1, 0)], dtype=int),
        )


def test_frozen_graph_schema_is_strictly_analytic() -> None:
    assert PROTOCOL == "2026-08-08-next115-hall-cut-interval-deficit-v1"
    assert FEATURE_NAMES == tuple(
        f"hcid_{side}_{suffix}"
        for side in ("positive", "negative")
        for suffix in (
            "global_deficit",
            "local_density",
            "origin_site_fraction_min",
            "origin_site_fraction_max",
            "neighbor_site_fraction_min",
        )
    )
    forbidden = (
        "energy",
        "force",
        "stress",
        "relax",
        "dft",
        "model",
        "mattersim",
        "chgnet",
        "mlip",
    )
    assert not any(
        token in name.lower() for token in forbidden for name in FEATURE_NAMES
    )


def _brute_max_deficit(
    low: np.ndarray,
    high: np.ndarray,
    origin: list[int],
    endpoints: list[tuple[int, int]],
) -> float:
    best = 0.0
    for mask in range(1 << len(origin)):
        chosen = {
            origin[index] for index in range(len(origin)) if mask & (1 << index)
        }
        neighborhood = {right for left, right in endpoints if left in chosen}
        best = max(
            best,
            sum(low[index] for index in chosen)
            - sum(high[index] for index in neighborhood),
        )
    return float(best)


def _direct_interval_flow_is_feasible(
    low: np.ndarray,
    high: np.ndarray,
    endpoints: list[tuple[int, int]],
) -> bool:
    if not endpoints:
        return False
    incidence = np.asarray(
        [
            [1.0 if site in edge else 0.0 for edge in endpoints]
            for site in range(len(low))
        ]
    )
    solved = linprog(
        np.zeros(len(endpoints), dtype=float),
        A_ub=np.vstack([incidence, -incidence]),
        b_ub=np.concatenate([high, -low]),
        bounds=[(0.0, None)] * len(endpoints),
        method="highs",
    )
    return bool(solved.success)


def test_primary_closure_lp_matches_exhaustive_subsets_and_flow_feasibility() -> None:
    rng = np.random.default_rng(20260808)
    for _ in range(96):
        n_positive = int(rng.integers(1, 5))
        n_negative = int(rng.integers(1, 5))
        positive = list(range(n_positive))
        negative = list(range(n_positive, n_positive + n_negative))
        low = rng.uniform(0.2, 4.0, n_positive + n_negative)
        high = low + rng.uniform(0.0, 3.0, n_positive + n_negative)
        endpoints = [
            (left, right)
            for left in positive
            for right in negative
            if rng.random() < 0.45
        ]
        signed_bounds = [
            (float(low[index]), float(high[index]))
            if index in positive
            else (-float(high[index]), -float(low[index]))
            for index in range(len(low))
        ]
        result = solve_hall_cut_interval_deficit(
            signed_charge_bounds=signed_bounds,
            endpoints=np.asarray(endpoints, dtype=int).reshape(-1, 2),
        )

        _assert_bounded(result)
        positive_oracle = _brute_max_deficit(low, high, positive, endpoints)
        reverse = [(right, left) for left, right in endpoints]
        negative_oracle = _brute_max_deficit(low, high, negative, reverse)
        assert result.positive_max_deficit == pytest.approx(
            positive_oracle, abs=1.0e-8
        )
        assert result.negative_max_deficit == pytest.approx(
            negative_oracle, abs=1.0e-8
        )
        zero_certificates = positive_oracle <= 1.0e-8 and negative_oracle <= 1.0e-8
        assert _direct_interval_flow_is_feasible(low, high, endpoints) == zero_certificates


def test_graph_values_are_permutation_replication_and_charge_scale_invariant() -> None:
    bounds = [(2.0, 2.0), (1.0, 1.0), (-1.0, -1.0), (-2.0, -2.0)]
    endpoints = np.asarray([(0, 2), (1, 2), (1, 3)], dtype=int)
    reference = solve_hall_cut_interval_deficit(
        signed_charge_bounds=bounds,
        endpoints=endpoints,
    )
    old_to_new = np.asarray([1, 0, 3, 2], dtype=int)
    permuted_bounds: list[tuple[float, float] | None] = [None] * len(bounds)
    for old, new in enumerate(old_to_new):
        permuted_bounds[int(new)] = bounds[old]
    permuted = solve_hall_cut_interval_deficit(
        signed_charge_bounds=[value for value in permuted_bounds if value is not None],
        endpoints=np.vstack(
            [old_to_new[endpoints][np.asarray([2, 0, 1])], old_to_new[endpoints[:1]]]
        ),
    )
    replicated = solve_hall_cut_interval_deficit(
        signed_charge_bounds=bounds + bounds,
        endpoints=np.vstack([endpoints, endpoints + len(bounds)]),
    )
    scaled = solve_hall_cut_interval_deficit(
        signed_charge_bounds=[(7.0 * lower, 7.0 * upper) for lower, upper in bounds],
        endpoints=endpoints,
    )

    for observed in (permuted, replicated, scaled):
        _assert_bounded(observed)
        for name in FEATURE_NAMES:
            assert observed.features[name] == pytest.approx(
                reference.features[name], abs=1.0e-8
            )


def test_symmetric_primary_face_publishes_extrema_not_one_cut_vector() -> None:
    # The violating first component may be selected alone or together with the
    # exactly balanced second component without changing the primary deficit.
    result = solve_hall_cut_interval_deficit(
        signed_charge_bounds=[
            (2.0, 2.0),
            (1.0, 1.0),
            (-1.0, -1.0),
            (-1.0, -1.0),
        ],
        endpoints=np.asarray([(0, 2), (1, 3)], dtype=int),
    )

    _assert_bounded(result)
    assert result.positive_max_deficit == pytest.approx(1.0)
    assert result.features["hcid_positive_origin_site_fraction_min"] == pytest.approx(0.5)
    assert result.features["hcid_positive_origin_site_fraction_max"] == pytest.approx(1.0)
    assert result.features["hcid_positive_neighbor_site_fraction_min"] == pytest.approx(0.5)


def _binary_structure(left: str, right: str) -> Structure:
    return Structure(
        Lattice.cubic(4.2),
        [left, right],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def test_structure_evaluation_is_pure_finite_and_deterministic() -> None:
    structure = _binary_structure("Na", "O")
    species = tuple(str(site.specie) for site in structure)
    lattice = structure.lattice.matrix.copy()
    coordinates = structure.frac_coords.copy()

    first = compute_hall_cut_interval_deficit(
        structure, graph_mode="voronoi", catalogue_mode="core"
    )
    second = compute_hall_cut_interval_deficit(
        structure, graph_mode="voronoi", catalogue_mode="core"
    )

    assert isinstance(first, HallCutIntervalDeficitFeatureResult)
    assert first == second
    assert first.supported, first.failure_reason
    assert tuple(first.features) == FEATURE_NAMES
    assert all(0.0 <= value <= 1.0 for value in first.features.values())
    assert len(first.catalogue_sha256) == 64
    assert first.pymatgen_version and first.scipy_version
    assert tuple(str(site.specie) for site in structure) == species
    assert np.array_equal(structure.lattice.matrix, lattice)
    assert np.array_equal(structure.frac_coords, coordinates)


def test_structure_wrapper_preserves_next109_sign_pattern_abstention() -> None:
    structure = Structure(
        Lattice.cubic(7.0),
        ["B", "C", "N"],
        [[0.0, 0.0, 0.0], [0.35, 0.35, 0.35], [0.7, 0.7, 0.7]],
    )

    result = compute_hall_cut_interval_deficit(
        structure,
        graph_mode="voronoi",
        catalogue_mode="expanded",
        max_sign_patterns=1,
    )

    assert not result.supported
    assert result.features == {}
    assert "sign pattern" in str(result.failure_reason).lower()
    assert "exceeds 1" in str(result.failure_reason).lower()


def test_structure_features_are_actual_supercell_invariant() -> None:
    primitive = _binary_structure("Na", "O")
    supercell = primitive.copy()
    supercell.make_supercell([2, 1, 1])

    small = compute_hall_cut_interval_deficit(
        primitive, graph_mode="voronoi", catalogue_mode="core"
    )
    large = compute_hall_cut_interval_deficit(
        supercell, graph_mode="voronoi", catalogue_mode="core"
    )

    assert small.supported, small.failure_reason
    assert large.supported, large.failure_reason
    for name in FEATURE_NAMES:
        assert large.features[name] == pytest.approx(
            small.features[name], rel=1.0e-8, abs=1.0e-8
        )


def test_structure_wrapper_does_not_consult_brown_parameters(monkeypatch) -> None:
    import src.next104_convex_mixed_valence_flow as next104

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Brown parameters must stay outside NEXT115")

    monkeypatch.setattr(next104, "_brown_generic_strengths", fail_if_called)
    result = compute_hall_cut_interval_deficit(
        _binary_structure("Na", "O"),
        graph_mode="voronoi",
        catalogue_mode="core",
    )

    assert result.supported, result.failure_reason
