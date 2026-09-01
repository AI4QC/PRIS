from __future__ import annotations

import inspect

from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

from src.next239_morphometric_voronoi_bond_order import (
    morphometric_site_invariants,
)
from src.next243_coarse_grained_voronoi_bond_order import (
    FEATURE_NAMES,
    aggregate_cmvbo_features,
    bond_order_magnitude,
    build_cross_source_cmvbo_features,
    coarse_grained_site_values,
    compute_cmvbo_features,
    weighted_spherical_harmonics,
)


def _octahedral_facets() -> tuple[np.ndarray, np.ndarray]:
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


def test_spherical_harmonic_kernel_matches_scalar_addition_theorem() -> None:
    normals, areas = _octahedral_facets()
    scalar = morphometric_site_invariants(normals=normals, areas=areas)
    for order, expected in ((4, scalar[0]), (6, scalar[1])):
        vector = weighted_spherical_harmonics(
            normals=normals, areas=areas, order=order
        )
        assert bond_order_magnitude(vector, order=order) == pytest.approx(
            expected, abs=2.0e-14
        )


def test_coherence_and_correlation_identity_and_antialignment() -> None:
    normals, areas = _octahedral_facets()
    vector = weighted_spherical_harmonics(normals=normals, areas=areas, order=4)
    identical = coarse_grained_site_values(
        qlm=np.asarray([vector, vector]),
        neighbor_indices=[np.asarray([1]), np.asarray([0])],
        neighbor_weights=[np.asarray([1.0]), np.asarray([1.0])],
        order=4,
    )
    np.testing.assert_allclose(identical[1], 1.0, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(identical[2], 1.0, rtol=0.0, atol=2.0e-14)
    opposite = coarse_grained_site_values(
        qlm=np.asarray([vector, -vector]),
        neighbor_indices=[np.asarray([1]), np.asarray([0])],
        neighbor_weights=[np.asarray([1.0]), np.asarray([1.0])],
        order=4,
    )
    np.testing.assert_allclose(opposite[0], 0.0, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(opposite[1], 0.0, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(opposite[2], -1.0, rtol=0.0, atol=2.0e-14)


def test_aggregate_schema_and_linear_quantile_are_exact() -> None:
    values = np.asarray([0.0, 1.0])
    correlations = np.asarray([-1.0, 1.0])
    features = aggregate_cmvbo_features(
        bar_q4=values,
        bar_q6=values[::-1],
        coherence_q4=values,
        coherence_q6=values[::-1],
        neighbor_corr_q4=correlations,
        neighbor_corr_q6=correlations[::-1],
    )
    assert tuple(features) == FEATURE_NAMES
    assert features["cmvbo_bar_q4_q10"] == pytest.approx(0.1)
    assert features["cmvbo_neighbor_corr_q4_q10"] == pytest.approx(-0.8)
    assert features["cmvbo_neighbor_corr_joint_q10"] == pytest.approx(0.0)


def test_real_structure_is_rotation_scale_and_supercell_invariant() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    reference = compute_cmvbo_features(atoms)
    rotated = atoms.copy()
    rotated.rotate(31.0, "z", rotate_cell=True)
    scaled = atoms.copy()
    scaled.set_cell(1.7 * scaled.cell.array, scale_atoms=True)
    supercell = atoms.repeat((2, 1, 1))
    for result in (
        reference,
        compute_cmvbo_features(rotated),
        compute_cmvbo_features(scaled),
        compute_cmvbo_features(supercell),
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
    assert compute_cmvbo_features(with_calculator).supported is False
    assert compute_cmvbo_features(with_metadata).supported is False


def test_builder_interface_is_discovery_geometry_only() -> None:
    parameters = tuple(
        inspect.signature(build_cross_source_cmvbo_features).parameters
    )
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
    with pytest.raises(FileNotFoundError, match="NEXT243 input is missing"):
        build_cross_source_cmvbo_features(
            scigen_cohort_dir=tmp_path / "scigen",
            wyformer_cohort_dir=tmp_path / "wyformer",
            design_path=tmp_path / "design",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
