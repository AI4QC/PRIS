from __future__ import annotations

import inspect

from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest
from sympy.physics.wigner import wigner_3j

from src.next247_third_order_voronoi_bond_order import (
    FEATURE_NAMES,
    aggregate_tvbo_features,
    build_cross_source_tvbo_features,
    compute_tvbo_features,
    normalized_third_order_invariant,
    third_order_site_values,
    weighted_spherical_harmonics,
    wigner_3j_equal_order,
    wigner_3j_terms,
)


def _facets() -> tuple[np.ndarray, np.ndarray]:
    normals = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    return normals, np.asarray([1.0, 2.0, 1.5, 0.7, 1.3, 2.4])


def test_all_cached_wigner_coefficients_match_sympy() -> None:
    expected_counts = {4: 61, 6: 127}
    for order in (4, 6):
        terms = wigner_3j_terms(order)
        assert len(terms) == expected_counts[order]
        for m1, m2, m3, coefficient in terms:
            reference = float(wigner_3j(order, order, order, m1, m2, m3))
            assert coefficient == pytest.approx(reference, abs=5.0e-17)
        assert wigner_3j_equal_order(order, order, order, order) == 0.0


def test_normalized_third_order_known_identity_and_scale_invariance() -> None:
    for order in (4, 6):
        vector = np.zeros(2 * order + 1, dtype=complex)
        vector[order] = 0.37
        expected = float(wigner_3j(order, order, order, 0, 0, 0))
        value = normalized_third_order_invariant(vector, order=order)
        scaled = normalized_third_order_invariant(7.0 * vector, order=order)
        assert value == pytest.approx(expected, abs=2.0e-15)
        assert scaled == pytest.approx(value, abs=2.0e-15)
        assert -1.0 <= value <= 1.0


def test_aligned_neighbors_preserve_third_order_and_have_zero_delta() -> None:
    normals, areas = _facets()
    vector = weighted_spherical_harmonics(normals=normals, areas=areas, order=4)
    raw, bar, delta = third_order_site_values(
        qlm=np.asarray([vector, vector]),
        neighbor_indices=[np.asarray([1]), np.asarray([0])],
        neighbor_weights=[np.asarray([1.0]), np.asarray([1.0])],
        order=4,
    )
    np.testing.assert_allclose(raw, bar, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(delta, 0.0, rtol=0.0, atol=2.0e-15)


def test_facet_order_does_not_change_third_order_value() -> None:
    normals, areas = _facets()
    reference = weighted_spherical_harmonics(normals=normals, areas=areas, order=6)
    reversed_vector = weighted_spherical_harmonics(
        normals=normals[::-1], areas=(3.0 * areas)[::-1], order=6
    )
    np.testing.assert_allclose(reference, reversed_vector, rtol=0.0, atol=2.0e-15)
    assert normalized_third_order_invariant(
        reference, order=6
    ) == pytest.approx(
        normalized_third_order_invariant(reversed_vector, order=6), abs=2.0e-15
    )


def test_aggregate_schema_and_linear_quantiles_are_exact() -> None:
    raw = np.asarray([-1.0, 1.0])
    bar = np.asarray([-0.5, 0.5])
    delta = np.asarray([0.5, 0.5])
    features = aggregate_tvbo_features(
        w4=raw,
        w6=raw[::-1],
        bar_w4=bar,
        bar_w6=bar[::-1],
        delta_w4=delta,
        delta_w6=delta,
    )
    assert tuple(features) == FEATURE_NAMES
    assert features["tvbo_w4_abs_q10"] == pytest.approx(1.0)
    assert features["tvbo_bar_w4_abs_q10"] == pytest.approx(0.5)
    assert features["tvbo_w4_coarse_delta_q90"] == pytest.approx(0.5)


def test_real_structure_is_rotation_scale_and_supercell_invariant() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    reference = compute_tvbo_features(atoms)
    rotated = atoms.copy()
    rotated.rotate(31.0, "z", rotate_cell=True)
    scaled = atoms.copy()
    scaled.set_cell(1.7 * scaled.cell.array, scale_atoms=True)
    supercell = atoms.repeat((2, 1, 1))
    for result in (
        reference,
        compute_tvbo_features(rotated),
        compute_tvbo_features(scaled),
        compute_tvbo_features(supercell),
    ):
        assert result.supported is True, result.failure_reason
        assert tuple(result.features) == FEATURE_NAMES
        np.testing.assert_allclose(
            [result.features[name] for name in FEATURE_NAMES],
            [reference.features[name] for name in FEATURE_NAMES],
            rtol=0.0,
            atol=3.0e-10,
        )


def test_geometry_boundary_fails_closed() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    with_calculator = atoms.copy()
    with_calculator.calc = Calculator()
    with_metadata = atoms.copy()
    with_metadata.info["outcome"] = 1
    assert compute_tvbo_features(with_calculator).supported is False
    assert compute_tvbo_features(with_metadata).supported is False


def test_builder_interface_is_discovery_geometry_only() -> None:
    parameters = tuple(inspect.signature(build_cross_source_tvbo_features).parameters)
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
    with pytest.raises(FileNotFoundError, match="NEXT247 input is missing"):
        build_cross_source_tvbo_features(
            scigen_cohort_dir=tmp_path / "scigen",
            wyformer_cohort_dir=tmp_path / "wyformer",
            design_path=tmp_path / "design",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
