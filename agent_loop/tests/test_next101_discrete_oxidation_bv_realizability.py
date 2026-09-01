from __future__ import annotations

import math

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from src.next101_discrete_oxidation_bv_realizability import (
    FEATURE_NAMES,
    PROTOCOL,
    DOBVRFeatureResult,
    compute_discrete_oxidation_bv_realizability,
    enumerate_uniform_neutral_assignments,
)


def test_schema_and_protocol_stay_inside_no_dft_boundary() -> None:
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
    assert PROTOCOL == "2026-08-04-next101-uniform-dobvr-v1"
    assert FEATURE_NAMES == (
        "dobvr_neutral_assignment_count",
        "dobvr_supported_assignment_fraction",
        "dobvr_best_mismatch_rms",
        "dobvr_best_mismatch_q95",
        "dobvr_best_mismatch_max",
        "dobvr_median_mismatch_rms",
        "dobvr_runner_up_gap_rms",
        "dobvr_best_parameter_exact_fraction",
        "dobvr_best_parameter_generic_fraction",
        "dobvr_best_mean_abs_oxidation",
        "dobvr_best_max_abs_oxidation",
        "dobvr_assignment_log_count",
    )
    assert not any(
        token in name.lower() for name in FEATURE_NAMES for token in forbidden
    )


def _structure(species: list[str]) -> Structure:
    size = max(8.0, 2.0 * len(species))
    coordinates = [
        [index / len(species), index / len(species), index / len(species)]
        for index in range(len(species))
    ]
    return Structure(Lattice.cubic(size), species, coordinates)


def test_uniform_enumeration_finds_canonical_neutral_assignments() -> None:
    nacl = _structure(["Na", "Cl"])
    ferric_oxide = _structure(["Fe", "Fe", "O", "O", "O"])

    nacl_result = enumerate_uniform_neutral_assignments(nacl)
    ferric_result = enumerate_uniform_neutral_assignments(ferric_oxide)

    assert nacl_result.supported, nacl_result.failure_reason
    assert [assignment.element_states for assignment in nacl_result.assignments] == [
        (("Cl", -1), ("Na", 1))
    ]
    assert ferric_result.supported, ferric_result.failure_reason
    assert (("Fe", 3), ("O", -2)) in {
        assignment.element_states for assignment in ferric_result.assignments
    }
    for result in (nacl_result, ferric_result):
        for assignment in result.assignments:
            assert sum(assignment.site_charges) == 0
            assert any(charge > 0 for charge in assignment.site_charges)
            assert any(charge < 0 for charge in assignment.site_charges)
        assert result.assignments == tuple(sorted(result.assignments))
        assert len(result.catalogue_sha256) == 64
        assert result.pymatgen_version


def test_uniform_enumeration_is_supercell_invariant() -> None:
    primitive = _structure(["Fe", "Fe", "O", "O", "O"])
    supercell = primitive.copy()
    supercell.make_supercell([2, 1, 1])

    primitive_result = enumerate_uniform_neutral_assignments(primitive)
    supercell_result = enumerate_uniform_neutral_assignments(supercell)

    assert [a.element_states for a in primitive_result.assignments] == [
        a.element_states for a in supercell_result.assignments
    ]
    assert primitive_result.catalogue_sha256 == supercell_result.catalogue_sha256


def test_uniform_enumeration_abstains_when_no_neutral_assignment_exists() -> None:
    result = enumerate_uniform_neutral_assignments(_structure(["Cu"]))

    assert not result.supported
    assert result.assignments == ()
    assert "neutral" in str(result.failure_reason).lower()


def _cscl() -> Structure:
    return Structure(
        Lattice.cubic(4.2),
        ["Cs", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def test_structure_evaluation_is_pure_finite_and_deterministic() -> None:
    structure = _cscl()
    species = tuple(str(site.specie) for site in structure)
    lattice = structure.lattice.matrix.copy()
    coordinates = structure.frac_coords.copy()
    kwargs = {
        "graph_mode": "voronoi",
        "state_catalogue": {"Cs": (1,), "Cl": (-1,)},
    }

    first = compute_discrete_oxidation_bv_realizability(structure, **kwargs)
    second = compute_discrete_oxidation_bv_realizability(structure, **kwargs)

    assert isinstance(first, DOBVRFeatureResult)
    assert first == second
    assert first.supported, first.failure_reason
    assert first.best_assignment is not None
    assert tuple(first.features) == FEATURE_NAMES
    assert all(math.isfinite(value) for value in first.features.values())
    assert first.features["dobvr_neutral_assignment_count"] == 1.0
    assert first.features["dobvr_supported_assignment_fraction"] == 1.0
    assert first.features["dobvr_runner_up_gap_rms"] == 0.0
    assert tuple(str(site.specie) for site in structure) == species
    assert np.array_equal(structure.lattice.matrix, lattice)
    assert np.array_equal(structure.frac_coords, coordinates)


def test_structure_features_are_supercell_invariant() -> None:
    primitive = _cscl()
    supercell = primitive.copy()
    supercell.make_supercell([2, 1, 1])
    kwargs = {
        "graph_mode": "voronoi",
        "state_catalogue": {"Cs": (1,), "Cl": (-1,)},
    }

    small = compute_discrete_oxidation_bv_realizability(primitive, **kwargs)
    large = compute_discrete_oxidation_bv_realizability(supercell, **kwargs)

    assert small.supported, small.failure_reason
    assert large.supported, large.failure_reason
    for name in FEATURE_NAMES:
        assert large.features[name] == pytest.approx(
            small.features[name], rel=1.0e-10, abs=1.0e-12
        )


def test_multiple_neutral_explanations_are_exposed_without_label_selection() -> None:
    structure = Structure(
        Lattice.tetragonal(4.6, 3.0),
        ["Ti", "O", "O"],
        [[0.0, 0.0, 0.0], [0.305, 0.305, 0.0], [0.695, 0.695, 0.0]],
    )
    result = compute_discrete_oxidation_bv_realizability(
        structure,
        graph_mode="voronoi",
        state_catalogue={"Ti": (2, 4), "O": (-2, -1)},
    )

    assert result.supported, result.failure_reason
    assert result.features["dobvr_neutral_assignment_count"] == 2.0
    assert result.features["dobvr_supported_assignment_fraction"] > 0.0
    assert result.features["dobvr_runner_up_gap_rms"] >= 0.0
    assert result.best_assignment in result.assignments


def test_structure_evaluation_abstains_instead_of_inventing_charge_balance() -> None:
    result = compute_discrete_oxidation_bv_realizability(
        _cscl(),
        graph_mode="voronoi",
        state_catalogue={"Cs": (1,), "Cl": (1,)},
    )

    assert not result.supported
    assert result.features == {}
    assert result.best_assignment is None
    assert "neutral" in str(result.failure_reason).lower()
