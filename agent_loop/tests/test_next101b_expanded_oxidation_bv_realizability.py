from __future__ import annotations

import math

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from src.next101b_expanded_oxidation_bv_realizability import (
    FEATURE_NAMES,
    PROTOCOL,
    ExpandedDOBVRFeatureResult,
    compute_expanded_discrete_oxidation_bv_realizability,
    enumerate_expanded_uniform_neutral_assignments,
)


def test_expanded_schema_and_protocol_are_frozen_and_no_dft() -> None:
    assert PROTOCOL == "2026-08-04-next101b-expanded-uniform-dobvr-v1"
    assert FEATURE_NAMES == (
        "dobvrb_neutral_assignment_count",
        "dobvrb_supported_assignment_fraction",
        "dobvrb_best_mismatch_rms",
        "dobvrb_best_mismatch_q95",
        "dobvrb_best_mismatch_max",
        "dobvrb_median_mismatch_rms",
        "dobvrb_runner_up_gap_rms",
        "dobvrb_best_parameter_exact_fraction",
        "dobvrb_best_parameter_generic_fraction",
        "dobvrb_best_mean_abs_oxidation",
        "dobvrb_best_max_abs_oxidation",
        "dobvrb_assignment_log_count",
        "dobvrb_best_catalogue_tier",
        "dobvrb_core_assignment_fraction",
        "dobvrb_best_eneg_margin",
    )
    forbidden = ("energy", "force", "stress", "relax", "dft", "model", "proxy")
    assert not any(
        token in name.lower() for name in FEATURE_NAMES for token in forbidden
    )


def _binary(species: list[str], lattice: float = 5.5) -> Structure:
    return Structure(
        Lattice.cubic(lattice),
        species,
        [[index / len(species)] * 3 for index in range(len(species))],
    )


def test_electronegativity_gate_removes_reversed_nacl_assignment() -> None:
    result = enumerate_expanded_uniform_neutral_assignments(_binary(["Na", "Cl"]))

    assert result.supported, result.failure_reason
    assert [assignment.element_states for assignment in result.assignments] == [
        (("Cl", -1), ("Na", 1))
    ]
    assert result.assignments[0].catalogue_tier == 0
    assert result.assignments[0].electronegativity_margin > 0.0


def test_expanded_table_supports_peroxide_like_nao_as_tier_one() -> None:
    result = enumerate_expanded_uniform_neutral_assignments(_binary(["Na", "O"]))

    assert result.supported, result.failure_reason
    assignment = next(
        item
        for item in result.assignments
        if item.element_states == (("Na", 1), ("O", -1))
    )
    assert assignment.catalogue_tier == 1


def _tio2() -> Structure:
    return Structure(
        Lattice.tetragonal(4.6, 3.0),
        ["Ti", "O", "O"],
        [[0.0, 0.0, 0.0], [0.305, 0.305, 0.0], [0.695, 0.695, 0.0]],
    )


def test_structure_evaluation_prioritizes_core_tier_and_is_pure() -> None:
    structure = _tio2()
    lattice = structure.lattice.matrix.copy()
    coordinates = structure.frac_coords.copy()

    result = compute_expanded_discrete_oxidation_bv_realizability(
        structure, graph_mode="voronoi"
    )

    assert isinstance(result, ExpandedDOBVRFeatureResult)
    assert result.supported, result.failure_reason
    assert result.best_assignment is not None
    assert result.best_assignment.element_states == (("O", -2), ("Ti", 4))
    assert result.best_assignment.catalogue_tier == 0
    assert tuple(result.features) == FEATURE_NAMES
    assert all(math.isfinite(value) for value in result.features.values())
    assert result.features["dobvrb_best_catalogue_tier"] == 0.0
    assert result.features["dobvrb_core_assignment_fraction"] > 0.0
    assert result.features["dobvrb_best_eneg_margin"] > 0.0
    assert np.array_equal(structure.lattice.matrix, lattice)
    assert np.array_equal(structure.frac_coords, coordinates)


def test_expanded_features_are_deterministic_and_supercell_invariant() -> None:
    primitive = _tio2()
    supercell = primitive.copy()
    supercell.make_supercell([2, 1, 1])

    first = compute_expanded_discrete_oxidation_bv_realizability(
        primitive, graph_mode="voronoi"
    )
    second = compute_expanded_discrete_oxidation_bv_realizability(
        primitive, graph_mode="voronoi"
    )
    large = compute_expanded_discrete_oxidation_bv_realizability(
        supercell, graph_mode="voronoi"
    )

    assert first == second
    assert first.supported and large.supported
    assert [item.element_states for item in first.assignments] == [
        item.element_states for item in large.assignments
    ]
    for name in FEATURE_NAMES:
        assert large.features[name] == pytest.approx(
            first.features[name], rel=1.0e-10, abs=1.0e-12
        )
