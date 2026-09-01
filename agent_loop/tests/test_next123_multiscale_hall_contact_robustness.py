from __future__ import annotations

from itertools import combinations
import math

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from src.next123_multiscale_hall_contact_robustness import (
    FEATURE_NAMES,
    STRENGTH_THRESHOLDS,
    MultiscaleHallContactRobustnessFeatureResult,
    _weighted_opposite_sign_endpoints,
    compute_multiscale_hall_contact_robustness,
    solve_multiscale_hall_contact_robustness,
)
from src.next109_convex_mixed_valence_obstruction import _opposite_sign_endpoints


def _directional_oracle(
    bounds: np.ndarray,
    endpoints: np.ndarray,
    *,
    positive_origin: bool,
) -> float:
    low = np.min(np.abs(bounds), axis=1)
    high = np.max(np.abs(bounds), axis=1)
    positive = bounds[:, 0] > 0.0
    origin = np.flatnonzero(positive if positive_origin else ~positive)
    oriented = endpoints if positive_origin else endpoints[:, ::-1]
    best = 0.0
    for size in range(1, len(origin) + 1):
        for subset in combinations(origin.tolist(), size):
            chosen = set(subset)
            neighbors = {
                int(right) for left, right in oriented if int(left) in chosen
            }
            best = max(
                best,
                float(low[list(chosen)].sum() - high[list(neighbors)].sum()),
            )
    return best


def _feature(side: str, threshold: float) -> str:
    code = int(round(100 * threshold))
    return f"mhcr_{side}_deficit_gain_tau{code:02d}"


def test_one_edge_threshold_oracle_and_schema() -> None:
    result = solve_multiscale_hall_contact_robustness(
        signed_charge_bounds=[(1.0, 1.0), (-1.0, -1.0)],
        weighted_endpoints=[(0, 1, 0.20)],
    )
    assert result.supported
    assert tuple(result.features) == FEATURE_NAMES
    for side in ("positive", "negative"):
        assert result.features[_feature(side, 0.05)] == pytest.approx(0.0)
        assert result.features[_feature(side, 0.10)] == pytest.approx(0.0)
        assert result.features[_feature(side, 0.25)] == pytest.approx(1.0)
        assert result.features[_feature(side, 0.50)] == pytest.approx(1.0)


def test_seeded_small_graphs_match_exhaustive_subset_oracle() -> None:
    rng = np.random.default_rng(12345)
    for _ in range(60):
        n_positive = int(rng.integers(1, 4))
        n_negative = int(rng.integers(1, 4))
        positive = np.column_stack(
            [rng.uniform(0.2, 1.2, n_positive), rng.uniform(1.3, 2.5, n_positive)]
        )
        positive.sort(axis=1)
        negative_magnitude = np.column_stack(
            [rng.uniform(0.2, 1.2, n_negative), rng.uniform(1.3, 2.5, n_negative)]
        )
        negative_magnitude.sort(axis=1)
        negative = -negative_magnitude[:, ::-1]
        bounds = np.vstack([positive, negative])
        weighted: list[tuple[int, int, float]] = []
        for left in range(n_positive):
            for right in range(n_positive, n_positive + n_negative):
                if rng.random() < 0.65:
                    weighted.append((left, right, float(rng.uniform(0.01, 1.0))))
        result = solve_multiscale_hall_contact_robustness(
            signed_charge_bounds=bounds,
            weighted_endpoints=weighted,
        )
        assert result.supported
        full = np.asarray([(a, b) for a, b, _ in weighted], dtype=int).reshape(-1, 2)
        for side, positive_origin in (("positive", True), ("negative", False)):
            denominator = float(
                np.min(np.abs(bounds), axis=1)[
                    bounds[:, 0] > 0.0 if positive_origin else bounds[:, 1] < 0.0
                ].sum()
            )
            base = _directional_oracle(
                bounds, full, positive_origin=positive_origin
            )
            previous = -math.inf
            for threshold in STRENGTH_THRESHOLDS:
                retained = np.asarray(
                    [(a, b) for a, b, w in weighted if w >= threshold],
                    dtype=int,
                ).reshape(-1, 2)
                expected = max(
                    _directional_oracle(
                        bounds, retained, positive_origin=positive_origin
                    )
                    - base,
                    0.0,
                ) / denominator
                observed = result.features[_feature(side, threshold)]
                assert observed == pytest.approx(expected, abs=2.0e-9)
                assert observed + 1.0e-12 >= previous
                previous = observed


def test_duplicates_use_maximum_strength_deterministically() -> None:
    bounds = [(1.0, 1.0), (-1.0, -1.0)]
    duplicate = solve_multiscale_hall_contact_robustness(
        signed_charge_bounds=bounds,
        weighted_endpoints=[(0, 1, 0.10), (0, 1, 0.80), (0, 1, 0.25)],
    )
    single = solve_multiscale_hall_contact_robustness(
        signed_charge_bounds=bounds,
        weighted_endpoints=[(0, 1, 0.80)],
    )
    assert duplicate == single
    assert set(single.features.values()) == {0.0}


def test_permutation_charge_scale_and_exact_replication_invariance() -> None:
    bounds = np.asarray(
        [(1.0, 2.0), (0.5, 1.0), (-2.0, -1.0), (-1.5, -0.5)]
    )
    weighted = [(0, 2, 0.80), (0, 3, 0.20), (1, 2, 0.10)]
    base = solve_multiscale_hall_contact_robustness(
        signed_charge_bounds=bounds,
        weighted_endpoints=weighted,
    )
    scaled = solve_multiscale_hall_contact_robustness(
        signed_charge_bounds=7.5 * bounds,
        weighted_endpoints=weighted,
    )
    permutation = np.asarray([2, 0, 3, 1])
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(len(permutation))
    permuted_edges = [
        (int(inverse[left]), int(inverse[right]), strength)
        for left, right, strength in weighted
    ]
    permuted = solve_multiscale_hall_contact_robustness(
        signed_charge_bounds=bounds[permutation],
        weighted_endpoints=permuted_edges,
    )
    replicated_bounds = np.vstack([bounds, bounds])
    replicated_edges = [
        (left + shift, right + shift, strength)
        for shift in (0, len(bounds))
        for left, right, strength in weighted
    ]
    replicated = solve_multiscale_hall_contact_robustness(
        signed_charge_bounds=replicated_bounds,
        weighted_endpoints=replicated_edges,
    )
    assert base.supported
    assert scaled.features == pytest.approx(base.features, abs=2.0e-9)
    assert permuted.features == pytest.approx(base.features, abs=2.0e-9)
    assert replicated.features == pytest.approx(base.features, abs=2.0e-9)


@pytest.mark.parametrize(
    ("bounds", "weighted", "thresholds"),
    [
        ([(1.0, 1.0), (2.0, 2.0)], [], STRENGTH_THRESHOLDS),
        ([(1.0, 1.0), (-1.0, -1.0)], [(1, 0, 0.5)], STRENGTH_THRESHOLDS),
        ([(1.0, 1.0), (-1.0, -1.0)], [(0, 1, 0.0)], STRENGTH_THRESHOLDS),
        ([(1.0, 1.0), (-1.0, -1.0)], [(0, 1, 1.01)], STRENGTH_THRESHOLDS),
        ([(1.0, 1.0), (-1.0, -1.0)], [(0, 1, math.nan)], STRENGTH_THRESHOLDS),
        ([(1.0, 1.0), (-1.0, -1.0)], [(0, 1, 0.5)], (0.25, 0.10)),
    ],
)
def test_invalid_graph_inputs_fail_closed(bounds, weighted, thresholds) -> None:
    with pytest.raises(ValueError):
        solve_multiscale_hall_contact_robustness(
            signed_charge_bounds=bounds,
            weighted_endpoints=weighted,
            thresholds=thresholds,
        )


def _binary_structure(left: str = "Na", right: str = "O") -> Structure:
    return Structure(
        Lattice.cubic(4.2),
        [left, right],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def test_weighted_voronoi_endpoints_reproduce_next109_endpoint_set() -> None:
    structure = _binary_structure()
    symbols = tuple(str(site.specie.symbol) for site in structure)
    signs = {"Na": 1, "O": -1}
    legacy = _opposite_sign_endpoints(
        structure,
        symbols=symbols,
        sign_by_element=signs,
        graph_mode="voronoi",
    )
    weighted = _weighted_opposite_sign_endpoints(
        structure,
        symbols=symbols,
        sign_by_element=signs,
        graph_mode="voronoi",
    )

    assert isinstance(legacy, np.ndarray)
    assert isinstance(weighted, np.ndarray)
    assert weighted.shape[1:] == (3,)
    assert np.array_equal(weighted[:, :2].astype(int), legacy)
    assert np.all((weighted[:, 2] > 0.0) & (weighted[:, 2] <= 1.0))
    assert weighted[:, 2].max() == pytest.approx(1.0)


def test_structure_wrapper_is_pure_finite_and_deterministic() -> None:
    structure = _binary_structure()
    species = tuple(str(site.specie) for site in structure)
    lattice = structure.lattice.matrix.copy()
    coordinates = structure.frac_coords.copy()

    first = compute_multiscale_hall_contact_robustness(
        structure, graph_mode="voronoi", catalogue_mode="core"
    )
    second = compute_multiscale_hall_contact_robustness(
        structure, graph_mode="voronoi", catalogue_mode="core"
    )

    assert isinstance(first, MultiscaleHallContactRobustnessFeatureResult)
    assert first == second
    assert first.supported, first.failure_reason
    assert tuple(first.features) == FEATURE_NAMES
    assert all(0.0 <= value <= 1.0 for value in first.features.values())
    assert len(first.catalogue_sha256) == 64
    assert first.pymatgen_version and first.scipy_version
    assert tuple(str(site.specie) for site in structure) == species
    assert np.array_equal(structure.lattice.matrix, lattice)
    assert np.array_equal(structure.frac_coords, coordinates)


def test_structure_features_are_actual_supercell_invariant() -> None:
    primitive = _binary_structure()
    supercell = primitive.copy()
    supercell.make_supercell([2, 1, 1])
    small = compute_multiscale_hall_contact_robustness(
        primitive, graph_mode="voronoi", catalogue_mode="core"
    )
    large = compute_multiscale_hall_contact_robustness(
        supercell, graph_mode="voronoi", catalogue_mode="core"
    )

    assert small.supported, small.failure_reason
    assert large.supported, large.failure_reason
    assert large.features == pytest.approx(small.features, abs=2.0e-8)


def test_structure_wrapper_preserves_next109_sign_pattern_abstention() -> None:
    structure = Structure(
        Lattice.cubic(7.0),
        ["B", "C", "N"],
        [[0.0, 0.0, 0.0], [0.35, 0.35, 0.35], [0.7, 0.7, 0.7]],
    )
    result = compute_multiscale_hall_contact_robustness(
        structure,
        graph_mode="voronoi",
        catalogue_mode="expanded",
        max_sign_patterns=1,
    )
    assert not result.supported
    assert result.features == {}
    assert "sign pattern" in str(result.failure_reason).lower()
