from __future__ import annotations

import inspect
import math

from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

from src.next263_local_angular_persistent_homology import (
    FEATURE_NAMES,
    aggregate_laph_features,
    build_cross_source_laph_features,
    compute_laph_features,
    vietoris_rips_intervals,
)


def test_triangle_and_tetrahedron_have_only_finite_h0_bars() -> None:
    triangle = np.asarray(
        [[1.0, 0.0, 0.0], [-0.5, math.sqrt(3) / 2, 0.0], [-0.5, -math.sqrt(3) / 2, 0.0]]
    )
    h0, h1 = vietoris_rips_intervals(triangle)
    assert len(h0) == 2
    np.testing.assert_allclose(h0, math.sqrt(3), rtol=0.0, atol=1.0e-10)
    assert h1 == ()

    tetrahedron = np.asarray(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float
    ) / math.sqrt(3)
    h0, h1 = vietoris_rips_intervals(tetrahedron)
    assert len(h0) == 3
    np.testing.assert_allclose(h0, math.sqrt(8 / 3), rtol=0.0, atol=1.0e-10)
    assert h1 == ()


def test_square_has_one_positive_h1_bar() -> None:
    square = np.asarray([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=float)
    h0, h1 = vietoris_rips_intervals(square)
    assert len(h0) == 3
    np.testing.assert_allclose(h0, math.sqrt(2), rtol=0.0, atol=1.0e-10)
    assert len(h1) == 1
    assert h1[0] == pytest.approx(2.0 - math.sqrt(2), abs=2.0e-10)


def test_barcode_is_permutation_and_rotation_invariant() -> None:
    rng = np.random.default_rng(7)
    points = rng.normal(size=(9, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1
    reference = vietoris_rips_intervals(points)
    assert vietoris_rips_intervals(points[[7, 1, 8, 0, 4, 2, 6, 5, 3]]) == reference
    assert vietoris_rips_intervals(points @ matrix) == reference


def test_aggregate_schema_uses_frozen_inverse_cdf() -> None:
    features = aggregate_laph_features(
        h0_death_mean=[1.0, 2.0, 3.0, 4.0],
        h0_death_cv=[0.1, 0.2, 0.3, 0.4],
        h1_persistence_density=[0.0, 0.1, 0.2, 0.3],
        h1_persistence_max=[0.0, 0.2, 0.4, 0.6],
    )
    assert tuple(features) == FEATURE_NAMES
    assert features["laph_h0_death_mean_q10"] == 1.0
    assert features["laph_h0_death_mean_q90"] == 4.0


def test_real_structure_is_rotation_scale_translation_and_supercell_invariant() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    reference = compute_laph_features(atoms)
    rotated = atoms.copy()
    rotated.rotate(31.0, "z", rotate_cell=True)
    scaled = atoms.copy()
    scaled.set_cell(1.7 * scaled.cell.array, scale_atoms=True)
    translated = atoms.copy()
    translated.translate([0.23, -0.41, 0.17])
    translated.wrap()
    supercell = atoms.repeat((2, 1, 1))
    for result in (
        reference,
        compute_laph_features(rotated),
        compute_laph_features(scaled),
        compute_laph_features(translated),
        compute_laph_features(supercell),
    ):
        assert result.supported is True, result.failure_reason
        assert tuple(result.features) == FEATURE_NAMES
        assert result.features == reference.features


def test_geometry_boundary_fails_closed() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    with_calculator = atoms.copy()
    with_calculator.calc = Calculator()
    with_metadata = atoms.copy()
    with_metadata.info["outcome"] = 1
    assert compute_laph_features(with_calculator).supported is False
    assert compute_laph_features(with_metadata).supported is False


def test_builder_interface_is_discovery_geometry_only() -> None:
    parameters = tuple(inspect.signature(build_cross_source_laph_features).parameters)
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
    with pytest.raises(FileNotFoundError, match="NEXT263 input is missing"):
        build_cross_source_laph_features(
            scigen_cohort_dir=tmp_path / "scigen",
            wyformer_cohort_dir=tmp_path / "wyformer",
            design_path=tmp_path / "design",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
