from __future__ import annotations

import inspect

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

from src.next307_periodic_bond_valence_hodge_loop import (
    FEATURE_DIRECTIONS,
    FEATURE_NAMES,
    PROTOCOL,
    bond_valence_hodge_loop_features,
    build_cross_source_pbvhl_features,
    compute_pbvhl_features,
)


def _four_cycle_values(values: object):
    return bond_valence_hodge_loop_features(
        n_sites=4,
        endpoints=np.asarray(((0, 2), (0, 3), (1, 2), (1, 3)), dtype=int),
        bond_valences=np.asarray(values, dtype=float),
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
    result = compute_pbvhl_features(atoms)
    assert result.supported, result.failure_reason
    return np.asarray([result.features[name] for name in FEATURE_NAMES], dtype=float)


def test_protocol_feature_and_direction_universes_are_exact() -> None:
    assert PROTOCOL == "2026-08-09-next307-periodic-bond-valence-hodge-loop-v1"
    assert FEATURE_NAMES == (
        "pbvhl_cycle_fraction",
        "pbvhl_cycle_rms",
        "pbvhl_cycle_q90",
        "pbvhl_site_rms_q90",
    )
    assert FEATURE_DIRECTIONS == {name: "protected_low" for name in FEATURE_NAMES}


def test_equal_four_cycle_obeys_loop_rule() -> None:
    result = _four_cycle_values([1.0, 1.0, 1.0, 1.0])
    assert result.supported, result.failure_reason
    assert result.incidence_rank == 3
    assert result.cycle_dimension == 1
    assert result.loop_divergence_max < 1.0e-14
    np.testing.assert_allclose(
        [result.features[name] for name in FEATURE_NAMES],
        np.zeros(len(FEATURE_NAMES)),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_perturbed_four_cycle_matches_analytic_cycle_projection() -> None:
    values = np.asarray([1.4, 1.0, 1.0, 1.0])
    result = _four_cycle_values(values)
    assert result.supported, result.failure_reason
    cycle_basis = np.asarray([1.0, -1.0, -1.0, 1.0])
    expected_cycle = cycle_basis * (
        float(values @ cycle_basis) / float(cycle_basis @ cycle_basis)
    )
    mean = float(values.mean())
    expected_abs = np.abs(expected_cycle) / mean
    expected_fraction = float(np.linalg.norm(expected_cycle) / np.linalg.norm(values))
    expected_rms = float(np.sqrt(np.mean(expected_abs**2)))
    np.testing.assert_allclose(
        result.features["pbvhl_cycle_fraction"], expected_fraction, atol=1.0e-14
    )
    np.testing.assert_allclose(
        result.features["pbvhl_cycle_rms"], expected_rms, atol=1.0e-14
    )
    np.testing.assert_allclose(
        result.features["pbvhl_cycle_q90"], expected_abs.max(), atol=1.0e-14
    )
    assert result.loop_divergence_max < 1.0e-14


def test_parallel_periodic_images_form_a_two_edge_loop() -> None:
    equal = bond_valence_hodge_loop_features(
        n_sites=2,
        endpoints=np.asarray(((0, 1), (0, 1))),
        bond_valences=np.asarray((2.0, 2.0)),
    )
    unequal = bond_valence_hodge_loop_features(
        n_sites=2,
        endpoints=np.asarray(((0, 1), (0, 1))),
        bond_valences=np.asarray((2.0, 1.0)),
    )
    assert equal.supported and unequal.supported
    assert equal.cycle_dimension == unequal.cycle_dimension == 1
    assert equal.features["pbvhl_cycle_fraction"] < 1.0e-14
    assert unequal.features["pbvhl_cycle_fraction"] > 0.0


def test_kernel_is_scale_order_and_exact_replication_invariant() -> None:
    endpoints = np.asarray(((0, 2), (0, 3), (1, 2), (1, 3)), dtype=int)
    values = np.asarray((1.4, 1.0, 0.8, 1.2), dtype=float)
    reference = bond_valence_hodge_loop_features(
        n_sites=4, endpoints=endpoints, bond_valences=values
    )
    order = np.asarray((2, 0, 3, 1))
    reordered = bond_valence_hodge_loop_features(
        n_sites=4, endpoints=endpoints[order], bond_valences=7.0 * values[order]
    )
    replicated = bond_valence_hodge_loop_features(
        n_sites=8,
        endpoints=np.vstack((endpoints, endpoints + 4)),
        bond_valences=np.concatenate((values, values)),
    )
    for result in (reordered, replicated):
        assert result.supported, result.failure_reason
        np.testing.assert_allclose(
            [result.features[name] for name in FEATURE_NAMES],
            [reference.features[name] for name in FEATURE_NAMES],
            rtol=0.0,
            atol=1.0e-13,
        )


@pytest.mark.parametrize(
    ("n_sites", "endpoints", "values"),
    (
        (1, np.asarray(((0, 0),)), np.asarray((1.0,))),
        (3, np.asarray(((0, 2), (1, 3))), np.asarray((1.0, 1.0))),
        (3, np.asarray(((0, 2),)), np.asarray((1.0, 2.0))),
        (3, np.asarray(((0, 2),)), np.asarray((0.0,))),
        (3, np.asarray(((0, 2),)), np.asarray((np.nan,))),
    ),
)
def test_kernel_fails_open_on_invalid_inputs(n_sites, endpoints, values) -> None:
    result = bond_valence_hodge_loop_features(
        n_sites=n_sites, endpoints=endpoints, bond_valences=values
    )
    assert result.supported is False
    assert result.features == {}


def test_standard_crystals_have_negligible_loop_residual() -> None:
    cases = (
        bulk("NaCl", "rocksalt", a=5.64, cubic=True),
        bulk("CsCl", "cesiumchloride", a=4.12, cubic=True),
        bulk("ZnS", "zincblende", a=5.41, cubic=True),
    )
    for atoms in cases:
        result = compute_pbvhl_features(atoms)
        assert result.supported, result.failure_reason
        assert result.edge_count > result.incidence_rank
        assert result.cycle_dimension > 0
        assert result.loop_divergence_max < 1.0e-12
        np.testing.assert_allclose(
            [result.features[name] for name in FEATURE_NAMES],
            np.zeros(len(FEATURE_NAMES)),
            rtol=0.0,
            atol=1.0e-10,
        )


def test_distortion_produces_continuous_nontrivial_features() -> None:
    result = compute_pbvhl_features(_distorted_nacl())
    assert result.supported, result.failure_reason
    values = np.asarray([result.features[name] for name in FEATURE_NAMES])
    assert np.isfinite(values).all()
    assert (values >= 0.0).all()
    assert 0.0 < values[0] <= 1.0
    assert np.any(values > 1.0e-8)
    assert result.loop_divergence_max < 1.0e-10


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
        result = compute_pbvhl_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_unsupported_chemistry_fails_open() -> None:
    result = compute_pbvhl_features(bulk("Cu", "fcc", a=3.61, cubic=True))
    assert result.supported is False
    assert result.features == {}
    assert result.site_count == result.edge_count == result.incidence_rank == 0


def test_builder_interface_is_discovery_geometry_only() -> None:
    parameters = tuple(inspect.signature(build_cross_source_pbvhl_features).parameters)
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
    with pytest.raises(FileNotFoundError, match="NEXT307 input is missing"):
        build_cross_source_pbvhl_features(
            scigen_cohort_dir=tmp_path / "scigen",
            wyformer_cohort_dir=tmp_path / "wyformer",
            design_path=tmp_path / "design",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
