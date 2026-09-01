from __future__ import annotations

import inspect
import math

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next335_radical_facet_minkowski_tensor as n


def _box_halfspaces(extents: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    return np.vstack((np.eye(3), -np.eye(3))), np.asarray([*extents, *extents], dtype=float)


def _distorted_nacl() -> Atoms:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.set_cell(
        np.asarray([[5.64, 0.0, 0.0], [0.27, 5.77, 0.0], [0.18, 0.31, 5.53]]),
        scale_atoms=True,
    )
    atoms.positions[1] += np.asarray([0.08, -0.04, 0.06])
    atoms.wrap()
    return atoms


def test_frozen_schema_is_one_protected_high_q10() -> None:
    assert n.PROTOCOL == "2026-08-13-next335-radical-facet-minkowski-tensor-v1"
    assert n.FEATURE_NAMES == ("rfmt_surface_normal_beta_q10",)
    assert n.FEATURE_DIRECTIONS == {"rfmt_surface_normal_beta_q10": "protected_high"}


def test_surface_normal_beta_matches_isotropic_and_anisotropic_populations() -> None:
    normals = np.vstack((np.eye(3), -np.eye(3)))
    assert n.surface_normal_beta(normals=normals, areas=np.ones(6)) == pytest.approx(1.0)
    areas = np.asarray([24.0, 12.0, 8.0, 24.0, 12.0, 8.0])
    assert n.surface_normal_beta(normals=normals, areas=areas) == pytest.approx(1.0 / 3.0)
    assert n.surface_normal_beta(normals=normals, areas=37.0 * areas) == pytest.approx(1.0 / 3.0)


def test_surface_normal_beta_is_rotation_invariant_and_validates_input() -> None:
    normals = np.vstack((np.eye(3), -np.eye(3)))
    areas = np.asarray([24.0, 12.0, 8.0, 24.0, 12.0, 8.0])
    angle = 0.731
    axis = np.asarray([1.0, 2.0, -1.0], dtype=float)
    axis /= np.linalg.norm(axis)
    cross = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    rotation = (
        math.cos(angle) * np.eye(3)
        + (1.0 - math.cos(angle)) * np.outer(axis, axis)
        + math.sin(angle) * cross
    )
    assert n.surface_normal_beta(normals=normals @ rotation.T, areas=areas) == pytest.approx(1.0 / 3.0)
    with pytest.raises(ValueError, match="surface-normal population differs"):
        n.surface_normal_beta(normals=np.eye(3), areas=[1.0, 1.0])
    with pytest.raises(ValueError, match="surface-normal population differs"):
        n.surface_normal_beta(normals=np.eye(3), areas=[1.0, 0.0, 1.0])


def test_cube_and_rectangular_box_halfspaces_match_analytic_beta() -> None:
    normals, offsets = _box_halfspaces((1.0, 1.0, 1.0))
    cube = n.power_cell_minkowski_tensor(normals=normals, offsets=offsets, scale=1.0)
    assert not cube.empty
    assert cube.volume == pytest.approx(8.0)
    assert cube.surface_area == pytest.approx(24.0)
    assert cube.facet_count == 6
    assert cube.surface_normal_beta == pytest.approx(1.0)

    normals, offsets = _box_halfspaces((1.0, 2.0, 3.0))
    box = n.power_cell_minkowski_tensor(normals=normals, offsets=offsets, scale=3.0)
    assert box.volume == pytest.approx(48.0)
    assert box.surface_area == pytest.approx(88.0)
    assert box.surface_normal_beta == pytest.approx(1.0 / 3.0)
    np.testing.assert_allclose(box.tensor_eigenvalues, [2.0 / 11.0, 3.0 / 11.0, 6.0 / 11.0])


def _feature(atoms: Atoms) -> float:
    result = n.compute_rfmt_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_standard_and_distorted_crystals_have_finite_rfmt() -> None:
    for atoms in (
        bulk("NaCl", "rocksalt", a=5.64, cubic=True),
        bulk("CsCl", "cesiumchloride", a=4.12, cubic=True),
        bulk("ZnS", "zincblende", a=5.41, cubic=True),
        _distorted_nacl(),
    ):
        result = n.compute_rfmt_features(atoms)
        assert result.supported, result.failure_reason
        assert result.site_count == len(atoms)
        assert result.empty_cell_count == 0
        assert result.minimum_facet_count >= 4
        assert 0.0 < result.minimum_site_beta <= result.maximum_site_beta <= 1.0
        assert 0.0 < result.features[n.FEATURE_NAMES[0]] <= 1.0
        assert result.volume_tiling_relative_error <= n.n267.VOLUME_TILING_RELATIVE_TOLERANCE


def test_geometry_equivalences_preserve_rfmt() -> None:
    atoms = _distorted_nacl()
    reference = _feature(atoms)
    rotated = atoms.copy()
    rotated.rotate(31.0, "z", rotate_cell=True)
    translated = atoms.copy()
    translated.translate([0.173, 0.291, 0.419])
    translated.wrap()
    permuted = atoms[[3, 0, 6, 1, 7, 4, 2, 5]]
    rebased = atoms.copy()
    rebased.set_cell(
        np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=int) @ atoms.cell.array,
        scale_atoms=False,
    )
    rebased.wrap()
    replicated = atoms.repeat((2, 1, 1))
    for equivalent in (rotated, translated, permuted, rebased, replicated):
        assert _feature(equivalent) == pytest.approx(reference, abs=1.0e-8)


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
    for changed in (with_calculator, with_metadata, with_array, nonperiodic, nonfinite):
        result = n.compute_rfmt_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_row_schema_and_boundary_flags_are_exact() -> None:
    row = n.compute_rfmt_row(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    assert tuple(name for name in row if name.startswith("rfmt_")) == (
        "rfmt_surface_normal_beta_q10",
        "rfmt_supported",
        "rfmt_failure",
        "rfmt_site_count",
        "rfmt_empty_cell_count",
        "rfmt_minimum_facet_count",
        "rfmt_maximum_facet_count",
        "rfmt_minimum_site_beta",
        "rfmt_maximum_site_beta",
        "rfmt_volume_tiling_relative_error",
    )
    assert row["rfmt_supported"] is True
    assert n.BOUNDARY_FLAGS == {
        "dft_calculation_executed": False,
        "dft_values_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "opened_validation_outputs_used": False,
        "scigen_replication_endpoint_opened": False,
        "wyformer_replication_endpoint_opened": False,
    }


def test_builder_interface_is_discovery_geometry_only() -> None:
    parameters = tuple(inspect.signature(n.build_cross_source_rfmt_features).parameters)
    assert parameters == (
        "scigen_cohort_dir",
        "wyformer_cohort_dir",
        "design_path",
        "probe_result_path",
        "output_dir",
        "workers",
        "require_formal_inputs",
    )
    assert not any(
        token in name
        for name in parameters
        for token in ("endpoint", "label", "validation", "replication", "relax")
    )
