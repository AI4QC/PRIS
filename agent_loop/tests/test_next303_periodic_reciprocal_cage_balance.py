from __future__ import annotations

import inspect

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

from src.next303_periodic_reciprocal_cage_balance import (
    FEATURE_DIRECTIONS,
    FEATURE_NAMES,
    PRIOR_NAMES,
    PROTOCOL,
    build_cross_source_prcb_features,
    compute_prcb_features,
)


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
    result = compute_prcb_features(atoms)
    assert result.supported, result.failure_reason
    return np.asarray([result.features[name] for name in FEATURE_NAMES], dtype=float)


def test_protocol_prior_feature_and_direction_universes_are_exact() -> None:
    assert PROTOCOL == "2026-08-09-next303-periodic-reciprocal-cage-balance-v1"
    assert PRIOR_NAMES == ("uniform", "inverse_square", "charge_inverse_square")
    assert FEATURE_NAMES == (
        "prcb_uniform_closure_min",
        "prcb_uniform_closure_q10",
        "prcb_uniform_closure_mean",
        "prcb_inverse_square_closure_min",
        "prcb_inverse_square_closure_q10",
        "prcb_inverse_square_closure_mean",
        "prcb_charge_inverse_square_closure_min",
        "prcb_charge_inverse_square_closure_q10",
        "prcb_charge_inverse_square_closure_mean",
        "prcb_mutual_site_fraction_min",
        "prcb_mutual_site_fraction_q10",
        "prcb_mutual_site_fraction_mean",
        "prcb_mutual_edge_fraction",
    )
    assert FEATURE_DIRECTIONS == {name: "protected_high" for name in FEATURE_NAMES}


def test_standard_crystals_have_exact_mutual_reciprocal_balance() -> None:
    cases = (
        (bulk("NaCl", "rocksalt", a=5.64, cubic=True), 24),
        (bulk("CsCl", "cesiumchloride", a=4.12, cubic=True), 8),
        (bulk("ZnS", "zincblende", a=5.41, cubic=True), 16),
    )
    for atoms, expected_edges in cases:
        result = compute_prcb_features(atoms)
        assert result.supported, result.failure_reason
        assert result.site_count == len(atoms)
        assert result.edge_count == result.mutual_edge_count == expected_edges
        assert result.max_translation_range == 1
        np.testing.assert_allclose(
            [result.features[name] for name in FEATURE_NAMES],
            np.ones(len(FEATURE_NAMES)),
            rtol=0.0,
            atol=1.0e-10,
        )


def test_distortion_produces_continuous_nontrivial_features() -> None:
    result = compute_prcb_features(_distorted_nacl())
    assert result.supported, result.failure_reason
    values = np.asarray([result.features[name] for name in FEATURE_NAMES])
    assert np.isfinite(values).all()
    assert ((0.0 <= values) & (values <= 1.0)).all()
    assert np.any(values < 0.99)
    assert result.mutual_edge_count <= result.edge_count


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
        result = compute_prcb_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_unsupported_chemistry_fails_open() -> None:
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
    result = compute_prcb_features(atoms)
    assert result.supported is False
    assert result.features == {}
    assert result.site_count == result.edge_count == result.mutual_edge_count == 0


def test_builder_interface_is_discovery_geometry_only() -> None:
    parameters = tuple(inspect.signature(build_cross_source_prcb_features).parameters)
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
    with pytest.raises(FileNotFoundError, match="NEXT303 input is missing"):
        build_cross_source_prcb_features(
            scigen_cohort_dir=tmp_path / "scigen",
            wyformer_cohort_dir=tmp_path / "wyformer",
            design_path=tmp_path / "design",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
