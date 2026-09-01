from __future__ import annotations

import inspect

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
import pytest

import src.next295_positive_contact_force_closure as n295
from src.next299_minimal_opposite_sign_periodic_cage import (
    FEATURE_DIRECTIONS,
    FEATURE_NAMES,
    METRIC_NAMES,
    PROTOCOL,
    build_cross_source_mospc_features,
    compute_mospc_features,
    four_direction_equilibrium,
    minimal_opposite_sign_cage_for_site,
)


def _tetrahedron() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    ) / np.sqrt(3.0)


def _distorted_nacl() -> Atoms:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.set_cell(
        np.asarray(
            [[5.64, 0.0, 0.0], [0.27, 5.77, 0.0], [0.18, 0.31, 5.53]]
        ),
        scale_atoms=True,
    )
    atoms.positions[1] += np.asarray([0.08, -0.04, 0.06])
    atoms.wrap()
    return atoms


def _feature_vector(atoms: Atoms) -> np.ndarray:
    result = compute_mospc_features(atoms)
    assert result.supported, result.failure_reason
    return np.asarray([result.features[name] for name in FEATURE_NAMES], dtype=float)


def test_protocol_metric_feature_and_direction_universes_are_exact() -> None:
    assert PROTOCOL == "2026-08-09-next299-minimal-opposite-sign-periodic-cage-v1"
    assert METRIC_NAMES == (
        "uniform_closure",
        "inverse_square_closure",
        "uniform_equilibrium",
        "inverse_square_equilibrium",
    )
    assert FEATURE_NAMES == (
        "mospc_uniform_closure_min",
        "mospc_uniform_closure_q10",
        "mospc_uniform_closure_mean",
        "mospc_inverse_square_closure_min",
        "mospc_inverse_square_closure_q10",
        "mospc_inverse_square_closure_mean",
        "mospc_uniform_equilibrium_min",
        "mospc_uniform_equilibrium_q10",
        "mospc_uniform_equilibrium_mean",
        "mospc_inverse_square_equilibrium_min",
        "mospc_inverse_square_equilibrium_q10",
        "mospc_inverse_square_equilibrium_mean",
        "mospc_locally_enclosed_fraction",
    )
    assert FEATURE_DIRECTIONS == {name: "protected_high" for name in FEATURE_NAMES}


def test_four_direction_analytic_solution_is_exact_on_tetrahedron() -> None:
    directions = _tetrahedron()
    uniform = np.full(4, 0.25)
    assert four_direction_equilibrium(directions, uniform) == pytest.approx(1.0)
    weighted = np.asarray([0.4, 0.3, 0.2, 0.1])
    assert four_direction_equilibrium(directions, weighted) == pytest.approx(0.625)


def test_four_direction_solution_matches_frozen_lp() -> None:
    generator = np.random.default_rng(7)
    compared = 0
    for _ in range(100):
        directions = generator.normal(size=(4, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        prior = generator.random(4)
        prior /= prior.sum()
        analytic = four_direction_equilibrium(directions, prior)
        lp = n295.positive_equilibrium_fraction(directions, prior)
        assert analytic == pytest.approx(lp, abs=1.0e-9)
        compared += 1
    assert compared == 100


def test_four_direction_solution_returns_zero_for_one_sided_or_degenerate_set() -> None:
    one_sided = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
    )
    coplanar = np.asarray(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    )
    for directions in (one_sided, coplanar):
        assert four_direction_equilibrium(directions, np.full(4, 0.25)) == 0.0


def test_near_singular_augmented_balance_returns_zero_like_frozen_lp() -> None:
    directions = np.asarray(
        [
            [0.5585088689385969, 0.6782986401010200, 0.4774712537462714],
            [0.5585088689385970, -0.6782986401010197, 0.47747125374627153],
            [-0.6086711347861360, 0.6250959418324255, -0.48864559056926976],
            [-0.6086711347861361, -0.6250959418324253, -0.4886455905692699],
        ]
    )
    prior = np.full(4, 0.25)
    assert n295.positive_equilibrium_fraction(directions, prior) == 0.0
    assert four_direction_equilibrium(directions, prior) == 0.0


def test_large_negative_balance_returns_zero_before_residual_rejection() -> None:
    directions = np.asarray(
        [
            [-0.022349675122564677, -0.6402714353535345, -0.7678235351252529],
            [-0.022349675122564680, 0.6402714353535343, -0.7678235351252531],
            [0.495093330897518060, 0.4148424360565947, 0.7634057551180963],
            [0.495093354309434850, -0.4148423416836122, 0.7634057912179403],
        ]
    )
    prior = np.full(4, 0.25)
    assert n295.positive_equilibrium_fraction(directions, prior) == 0.0
    assert four_direction_equilibrium(directions, prior) == 0.0


def test_standard_crystals_retain_all_fourth_distance_ties() -> None:
    cases = (
        (bulk("NaCl", "rocksalt", a=5.64, cubic=True), 6),
        (bulk("CsCl", "cesiumchloride", a=4.12, cubic=True), 8),
        (bulk("ZnS", "zincblende", a=5.41, cubic=True), 4),
    )
    for atoms, expected_cage_size in cases:
        result = compute_mospc_features(atoms)
        assert result.supported, result.failure_reason
        assert result.min_cage_size == result.max_cage_size == expected_cage_size
        assert result.max_translation_range == 1
        np.testing.assert_allclose(
            [result.features[name] for name in FEATURE_NAMES],
            np.ones(len(FEATURE_NAMES)),
            rtol=0.0,
            atol=1.0e-10,
        )


def test_site_cage_publishes_a_strict_outside_distance_certificate() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    structure = AseAtomsAdaptor.get_structure(atoms)
    charges = np.asarray([1.0 if str(site.specie) == "Na" else -1.0 for site in structure])
    cage = minimal_opposite_sign_cage_for_site(
        structure=structure,
        formal_valences=charges,
        site_index=0,
    )
    assert len(cage.vectors) == 6
    assert cage.certified_range == 1
    assert cage.outside_lower_bound > cage.fourth_distance + cage.tie_tolerance
    with pytest.raises(ValueError, match="translation range"):
        minimal_opposite_sign_cage_for_site(
            structure=structure,
            formal_valences=charges,
            site_index=0,
            max_translation_range=0,
        )


def test_geometry_equivalences_preserve_features() -> None:
    atoms = _distorted_nacl()
    reference = _feature_vector(atoms)
    rotated = atoms.copy()
    rotated.rotate(31.0, "z", rotate_cell=True)
    translated = atoms.copy()
    translated.translate([0.173, 0.291, 0.419])
    translated.wrap()
    permuted = atoms[[3, 0, 6, 1, 7, 4, 2, 5]]
    rebased = atoms.copy()
    rebased.set_cell(
        np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
        @ atoms.cell.array,
        scale_atoms=False,
    )
    rebased.wrap()
    replicated = atoms.repeat((2, 1, 1))
    for equivalent in (rotated, translated, permuted, rebased, replicated):
        np.testing.assert_allclose(
            _feature_vector(equivalent), reference, rtol=0.0, atol=1.0e-8
        )


def test_geometry_boundary_fails_closed() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    with_calculator = atoms.copy()
    with_calculator.calc = Calculator()
    with_metadata = atoms.copy()
    with_metadata.info["outcome"] = 1
    with_array = atoms.copy()
    with_array.new_array("energy", np.zeros(len(with_array)))
    nonperiodic = atoms.copy()
    nonperiodic.pbc = False
    nonfinite = atoms.copy()
    nonfinite.positions[0, 0] = np.nan
    for changed in (
        with_calculator,
        with_metadata,
        with_array,
        nonperiodic,
        nonfinite,
    ):
        result = compute_mospc_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_builder_interface_is_discovery_geometry_only() -> None:
    parameters = tuple(inspect.signature(build_cross_source_mospc_features).parameters)
    assert parameters == (
        "scigen_cohort_dir",
        "wyformer_cohort_dir",
        "design_path",
        "output_dir",
        "workers",
        "require_formal_inputs",
    )
    assert not any(
        token in name
        for name in parameters
        for token in ("endpoint", "label", "validation", "replication", "relax")
    )


def test_builder_fails_closed_on_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT299 input is missing"):
        build_cross_source_mospc_features(
            scigen_cohort_dir=tmp_path / "scigen",
            wyformer_cohort_dir=tmp_path / "wyformer",
            design_path=tmp_path / "design",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
