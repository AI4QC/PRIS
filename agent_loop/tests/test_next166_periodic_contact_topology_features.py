from __future__ import annotations

import numpy as np
from pymatgen.core import Lattice, Structure

from src.next19_valence_transport import PeriodicEdgeGeometry
from src.next166_periodic_contact_topology_features import (
    FEATURE_NAMES,
    GRAPH_MODES,
    _failure_row,
    compute_periodic_contact_topology,
    periodic_component_ranks,
    periodic_topology_features,
)


def edge(u: int, v: int, image: tuple[int, int, int]) -> PeriodicEdgeGeometry:
    return PeriodicEdgeGeometry(u, v, image, 1.0, 1.0)


def test_periodic_component_ranks_are_exact_for_zero_through_three_dimensions() -> None:
    catalogues = {
        0: [edge(0, 1, (0, 0, 0))],
        1: [edge(0, 1, (0, 0, 0)), edge(0, 1, (1, 0, 0))],
        2: [
            edge(0, 1, (0, 0, 0)),
            edge(0, 1, (1, 0, 0)),
            edge(0, 1, (0, 1, 0)),
        ],
        3: [
            edge(0, 1, (0, 0, 0)),
            edge(0, 1, (1, 0, 0)),
            edge(0, 1, (0, 1, 0)),
            edge(0, 1, (0, 0, 1)),
        ],
    }
    for rank, edges in catalogues.items():
        assert periodic_component_ranks(2, edges).tolist() == [rank, rank]


def test_periodic_rank_is_invariant_to_edge_gauge_order_duplicates_and_site_order() -> None:
    edges = [
        edge(0, 1, (0, 0, 0)),
        edge(0, 1, (1, 0, 0)),
        edge(0, 1, (0, 1, 0)),
        edge(0, 1, (0, 0, 1)),
    ]
    reversed_edges = [
        edge(item.anion, item.cation, tuple(-x for x in item.image))
        for item in reversed(edges)
    ]
    duplicated = [*edges, *reversed_edges, *edges]
    expected = [3, 3]
    assert periodic_component_ranks(2, edges).tolist() == expected
    assert periodic_component_ranks(2, reversed_edges).tolist() == expected
    assert periodic_component_ranks(2, duplicated).tolist() == expected


def test_periodic_topology_features_are_site_weighted_and_bounded() -> None:
    edges = [
        edge(0, 1, (0, 0, 0)),
        edge(0, 1, (1, 0, 0)),
        edge(0, 1, (0, 1, 0)),
        edge(0, 1, (0, 0, 1)),
    ]
    features = periodic_topology_features(3, edges, prefix="pct_demo")
    assert set(features) == {
        "pct_demo_rank_max",
        "pct_demo_rank_mean",
        "pct_demo_rank0_fraction",
        "pct_demo_rank1_fraction",
        "pct_demo_rank2_fraction",
        "pct_demo_rank3_fraction",
    }
    np.testing.assert_allclose(features["pct_demo_rank_max"], 1.0)
    np.testing.assert_allclose(features["pct_demo_rank_mean"], 2.0 / 3.0)
    np.testing.assert_allclose(features["pct_demo_rank0_fraction"], 1.0 / 3.0)
    np.testing.assert_allclose(features["pct_demo_rank3_fraction"], 2.0 / 3.0)
    np.testing.assert_allclose(
        sum(features[f"pct_demo_rank{rank}_fraction"] for rank in range(4)),
        1.0,
    )
    assert all(0.0 <= value <= 1.0 for value in features.values())


def test_real_structure_topology_is_supercell_invariant() -> None:
    structure = Structure.from_spacegroup(
        225,
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    supercell = structure.copy()
    supercell.make_supercell([2, 1, 1])
    primitive_features = compute_periodic_contact_topology(structure)
    supercell_features = compute_periodic_contact_topology(supercell)
    assert set(FEATURE_NAMES) == {
        f"pct_{mode}_{suffix}"
        for mode in GRAPH_MODES
        for suffix in (
            "rank_max",
            "rank_mean",
            "rank0_fraction",
            "rank1_fraction",
            "rank2_fraction",
            "rank3_fraction",
        )
    }
    for mode in GRAPH_MODES:
        assert primitive_features[f"pct_{mode}_supported"] is True
        assert supercell_features[f"pct_{mode}_supported"] is True
        for name in FEATURE_NAMES:
            if name.startswith(f"pct_{mode}_"):
                np.testing.assert_allclose(
                    primitive_features[name], supercell_features[name], atol=1.0e-12
                )


def test_failure_row_preserves_complete_fail_open_schema() -> None:
    row = _failure_row("parse failed")
    assert all(np.isnan(row[name]) for name in FEATURE_NAMES)
    for mode in GRAPH_MODES:
        assert row[f"pct_{mode}_supported"] is False
        assert row[f"pct_{mode}_failure"] == "parse failed"
