from __future__ import annotations

import inspect

from ase.build import bulk
import numpy as np
import pytest

from src.next239_morphometric_voronoi_bond_order import (
    FEATURE_NAMES,
    aggregate_mvbo_features,
    build_cross_source_mvbo_features,
    compute_mvbo_features,
    morphometric_site_invariants,
)


def test_site_invariants_are_rotation_order_and_area_scale_invariant() -> None:
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
    areas = np.ones(6)
    angle = np.deg2rad(37.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    reference = morphometric_site_invariants(normals=normals, areas=areas)
    transformed = morphometric_site_invariants(
        normals=(normals @ rotation.T)[::-1], areas=(7.0 * areas)[::-1]
    )
    np.testing.assert_allclose(reference, transformed, rtol=0.0, atol=1.0e-12)
    assert reference[2] == pytest.approx(1.0)
    assert all(0.0 <= value <= 1.0 for value in reference)


def test_aggregate_schema_and_same_element_dispersion_are_exact() -> None:
    features = aggregate_mvbo_features(
        q4=np.asarray([0.2, 0.2, 0.7]),
        q6=np.asarray([0.4, 0.4, 0.1]),
        evenness=np.asarray([0.9, 0.8, 0.7]),
        atomic_numbers=np.asarray([11, 11, 17]),
    )
    assert tuple(features) == FEATURE_NAMES
    assert features["mvbo_facet_evenness_min"] == pytest.approx(0.7)
    assert features["mvbo_same_element_q46_dispersion_max"] == pytest.approx(0.0)


def test_real_structure_is_rotation_scale_and_supercell_invariant() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    reference = compute_mvbo_features(atoms)
    rotated = atoms.copy()
    rotated.rotate(31.0, "z", rotate_cell=True)
    scaled = atoms.copy()
    scaled.set_cell(1.7 * scaled.cell.array, scale_atoms=True)
    supercell = atoms.repeat((2, 1, 1))
    for result in (
        reference,
        compute_mvbo_features(rotated),
        compute_mvbo_features(scaled),
        compute_mvbo_features(supercell),
    ):
        assert result.supported is True
        assert tuple(result.features) == FEATURE_NAMES
        np.testing.assert_allclose(
            [result.features[name] for name in FEATURE_NAMES],
            [reference.features[name] for name in FEATURE_NAMES],
            rtol=0.0,
            atol=2.0e-10,
        )


def test_builder_interface_is_discovery_geometry_only() -> None:
    parameters = tuple(inspect.signature(build_cross_source_mvbo_features).parameters)
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
    with pytest.raises(FileNotFoundError, match="NEXT239 input is missing"):
        build_cross_source_mvbo_features(
            scigen_cohort_dir=tmp_path / "scigen",
            wyformer_cohort_dir=tmp_path / "wyformer",
            design_path=tmp_path / "design",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
