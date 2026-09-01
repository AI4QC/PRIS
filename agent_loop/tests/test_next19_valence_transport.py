from __future__ import annotations

import math
import warnings

import pytest
import numpy as np
from pymatgen.core import Lattice, Structure

from src.next19_valence_transport import (
    FEATURE_NAMES,
    EdgePrior,
    build_edge_priors,
    build_periodic_edge_geometry,
    compute_valence_transport_features,
    edge_priors_from_periodic_geometry,
    infer_formal_valences,
    infer_valence_assignment,
    solve_valence_transport,
    validate_feature_names,
)


def test_feature_schema_contains_no_forbidden_energy_or_model_terms() -> None:
    validate_feature_names(FEATURE_NAMES)
    lowered = " ".join(FEATURE_NAMES).lower()
    for token in ("energy", "force", "stress", "relax", "mattersim", "dft"):
        assert token not in lowered


def test_feature_schema_rejects_endpoint_like_name() -> None:
    with pytest.raises(ValueError, match="forbidden feature field"):
        validate_feature_names(("vt_overload", "dft_energy"))


def test_balanced_one_to_one_graph_needs_no_overload_or_reallocation() -> None:
    result = solve_valence_transport(
        cation_supply={0: 1.0},
        anion_demand={1: 1.0},
        edges=(EdgePrior(0, 1, (0, 0, 0), 1.0),),
    )

    assert result.supported
    assert result.failure_reason is None
    assert result.overload == pytest.approx(0.0, abs=1.0e-9)
    assert result.reallocation == pytest.approx(0.0, abs=1.0e-9)
    assert result.max_anion_mismatch == pytest.approx(0.0, abs=1.0e-9)


def test_bottleneck_graph_reports_required_edge_overload() -> None:
    result = solve_valence_transport(
        cation_supply={0: 1.0, 1: 1.0},
        anion_demand={2: 1.5, 3: 0.5},
        edges=(
            EdgePrior(0, 2, (0, 0, 0), 0.5),
            EdgePrior(0, 3, (0, 0, 0), 0.5),
            EdgePrior(1, 2, (0, 0, 0), 0.5),
            EdgePrior(1, 3, (0, 0, 0), 0.5),
        ),
    )

    assert result.supported
    assert result.overload == pytest.approx(0.5, abs=1.0e-8)
    assert result.reallocation == pytest.approx(0.25, abs=1.0e-8)
    assert result.max_anion_mismatch == pytest.approx(1.0, abs=1.0e-8)


def test_disconnected_supply_and_demand_fail_open() -> None:
    result = solve_valence_transport(
        cation_supply={0: 1.0, 1: 1.0},
        anion_demand={2: 1.0, 3: 1.0},
        edges=(EdgePrior(0, 2, (0, 0, 0), 1.0),),
    )

    assert not result.supported
    assert result.failure_reason
    assert math.isnan(result.overload)
    assert math.isnan(result.reallocation)


def test_solver_is_invariant_to_edge_input_order() -> None:
    edges = (
        EdgePrior(0, 2, (0, 0, 0), 0.7),
        EdgePrior(0, 3, (0, 0, 0), 0.3),
        EdgePrior(1, 2, (0, 0, 0), 0.2),
        EdgePrior(1, 3, (0, 0, 0), 0.8),
    )
    kwargs = {
        "cation_supply": {0: 1.0, 1: 1.0},
        "anion_demand": {2: 1.0, 3: 1.0},
    }

    forward = solve_valence_transport(edges=edges, **kwargs)
    reverse = solve_valence_transport(edges=tuple(reversed(edges)), **kwargs)

    assert forward.supported and reverse.supported
    assert forward.overload == pytest.approx(reverse.overload, abs=1.0e-10)
    assert forward.reallocation == pytest.approx(reverse.reallocation, abs=1.0e-10)
    assert forward.max_anion_mismatch == pytest.approx(
        reverse.max_anion_mismatch, abs=1.0e-10
    )


def _cscl() -> Structure:
    return Structure(
        Lattice.cubic(4.2),
        ["Cs", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def _nacl() -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m",
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
    )


def test_infer_formal_valences_supports_simple_ionic_structure() -> None:
    values, error = infer_formal_valences(_nacl())

    assert error is None
    assert values is not None
    assert set(values) == {-1.0, 1.0}


def test_electronegativity_partition_fail_open_fallback_is_neutral() -> None:
    structure = Structure(
        Lattice.cubic(3.6),
        ["Cu", "Ni"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )

    assignment = infer_valence_assignment(structure)

    assert assignment.supported
    assert assignment.policy == "electronegativity_partition"
    assert assignment.values is not None
    assert sum(assignment.values) == pytest.approx(0.0, abs=1.0e-12)
    assert any(value > 0.0 for value in assignment.values)
    assert any(value < 0.0 for value in assignment.values)


@pytest.mark.parametrize("mode", ["crystalnn", "voronoi"])
def test_periodic_graph_retains_cscl_image_multiplicity(mode: str) -> None:
    graph = build_edge_priors(
        _cscl(), [1.0, -1.0], graph_mode=mode, alpha=2.0
    )

    assert graph.supported
    assert graph.failure_reason is None
    assert len(graph.edges) >= 8
    assert sum(edge.prior for edge in graph.edges) == pytest.approx(1.0)
    assert len({edge.image for edge in graph.edges}) >= 8


def test_structure_feature_api_is_pure_and_has_exact_schema() -> None:
    structure = _nacl()
    original_lattice = structure.lattice.matrix.copy()
    original_fractional = structure.frac_coords.copy()

    result = compute_valence_transport_features(
        structure, graph_mode="crystalnn", alpha=2.0
    )

    assert result.supported
    assert result.failure_reason is None
    assert tuple(result.features) == FEATURE_NAMES
    assert all(math.isfinite(value) for value in result.features.values())
    assert np.array_equal(structure.lattice.matrix, original_lattice)
    assert np.array_equal(structure.frac_coords, original_fractional)


def test_structure_without_opposite_sign_valences_abstains() -> None:
    structure = Structure(
        Lattice.cubic(4.0),
        ["Cu", "Ni"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    result = compute_valence_transport_features(
        structure,
        formal_valences=[1.0, 1.0],
        graph_mode="crystalnn",
        alpha=2.0,
    )

    assert not result.supported
    assert result.failure_reason == "formal valences need both signs"
    assert result.features == {}


def test_neighbor_builder_uses_valence_decorated_copy_without_radius_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        graph = build_edge_priors(
            _cscl(), [1.0, -1.0], graph_mode="crystalnn", alpha=2.0
        )

    assert graph.supported
    messages = [str(item.message) for item in caught]
    assert not any("No oxidation states specified" in message for message in messages)


@pytest.mark.parametrize("alpha", [0.0, 2.0, 6.0])
def test_cached_periodic_geometry_reproduces_direct_edge_priors(alpha: float) -> None:
    structure = _nacl()
    charges = [1.0 if site.specie.symbol == "Na" else -1.0 for site in structure]
    geometry = build_periodic_edge_geometry(
        structure, charges, graph_mode="crystalnn"
    )
    cached = edge_priors_from_periodic_geometry(geometry, alpha=alpha)
    direct = build_edge_priors(
        structure, charges, graph_mode="crystalnn", alpha=alpha
    )

    assert geometry.supported
    assert cached.supported and direct.supported
    assert cached.edges == direct.edges
    assert cached.cation_supply == direct.cation_supply
    assert cached.anion_demand == direct.anion_demand
