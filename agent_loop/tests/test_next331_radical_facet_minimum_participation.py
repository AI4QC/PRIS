from __future__ import annotations

import inspect
import math

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next331_radical_facet_minimum_participation as n


def _cube_halfspaces(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    normals = np.vstack((np.eye(3), -np.eye(3)))
    offsets = np.full(6, float(scale), dtype=float)
    return normals, offsets


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


def test_frozen_schema_is_one_protected_high_q10() -> None:
    assert n.PROTOCOL == "2026-08-13-next331-radical-facet-minimum-participation-v1"
    assert n.FEATURE_NAMES == ("rfmp_minimum_area_participation_q10",)
    assert n.FEATURE_DIRECTIONS == {
        "rfmp_minimum_area_participation_q10": "protected_high"
    }


def test_analytic_minimum_area_participation_and_scale_invariance() -> None:
    areas = np.asarray([1.0, 1.0, 1.0, 0.1])
    expected = 4.0 * 0.1 / 3.1
    assert n.facet_minimum_participation(areas) == pytest.approx(expected)
    assert n.facet_minimum_participation(37.0 * areas) == pytest.approx(expected)
    assert n.facet_minimum_participation(np.ones(6)) == pytest.approx(1.0)


def test_analytic_participation_refuses_invalid_populations() -> None:
    for areas in ([], [1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 0.0], [1.0, np.nan, 1.0, 1.0]):
        with pytest.raises(ValueError, match="facet areas differ"):
            n.facet_minimum_participation(areas)


def test_structure_q10_refuses_a_positive_value_below_the_output_grid() -> None:
    with pytest.raises(ValueError, match="quantized to zero"):
        n.structure_minimum_participation_q10([2.0e-14] * 4)
    assert n.structure_minimum_participation_q10([0.1, 0.2, 0.3, 0.4]) == 0.1


def test_cube_halfspaces_reconstruct_six_equal_facets() -> None:
    normals, offsets = _cube_halfspaces()
    cell = n.power_cell_facet_areas(normals=normals, offsets=offsets, scale=1.0)
    assert not cell.empty
    assert cell.volume == pytest.approx(8.0)
    assert cell.surface_area == pytest.approx(24.0)
    assert cell.facet_count == 6
    np.testing.assert_allclose(cell.facet_areas, np.full(6, 4.0), atol=1.0e-10)
    assert cell.minimum_participation == pytest.approx(1.0)


def test_rectangular_box_facet_measure_matches_analytic_areas() -> None:
    normals, _ = _cube_halfspaces()
    offsets = np.asarray([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    cell = n.power_cell_facet_areas(normals=normals, offsets=offsets, scale=3.0)
    np.testing.assert_allclose(
        np.sort(cell.facet_areas), np.asarray([8.0, 8.0, 12.0, 12.0, 24.0, 24.0]),
        atol=1.0e-10,
    )
    assert cell.surface_area == pytest.approx(88.0)
    assert cell.minimum_participation == pytest.approx(6.0 * 8.0 / 88.0)


def _feature(atoms: Atoms) -> float:
    result = n.compute_rfmp_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_standard_and_distorted_crystals_have_finite_rfmp() -> None:
    for atoms in (
        bulk("NaCl", "rocksalt", a=5.64, cubic=True),
        bulk("CsCl", "cesiumchloride", a=4.12, cubic=True),
        bulk("ZnS", "zincblende", a=5.41, cubic=True),
        _distorted_nacl(),
    ):
        result = n.compute_rfmp_features(atoms)
        assert result.supported, result.failure_reason
        assert result.site_count == len(atoms)
        assert result.empty_cell_count == 0
        assert result.minimum_facet_count >= 4
        assert result.minimum_facet_area > 0.0
        assert 0.0 < result.features[n.FEATURE_NAMES[0]] <= 1.0
        assert result.volume_tiling_relative_error <= n.n267.VOLUME_TILING_RELATIVE_TOLERANCE


def test_geometry_equivalences_preserve_rfmp() -> None:
    atoms = _distorted_nacl()
    reference = _feature(atoms)

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
    rotated = atoms.copy()
    rotated.positions = rotated.positions @ rotation.T
    rotated.set_cell(rotated.cell.array @ rotation.T, scale_atoms=False)
    translated = atoms.copy()
    translated.positions += [1.37, -0.62, 0.91]
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
        result = n.compute_rfmp_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_row_schema_and_boundary_flags_are_exact() -> None:
    row = n.compute_rfmp_row(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    assert tuple(name for name in row if name.startswith("rfmp_")) == (
        "rfmp_minimum_area_participation_q10",
        "rfmp_supported",
        "rfmp_failure",
        "rfmp_site_count",
        "rfmp_empty_cell_count",
        "rfmp_minimum_facet_count",
        "rfmp_maximum_facet_count",
        "rfmp_minimum_facet_area",
        "rfmp_volume_tiling_relative_error",
    )
    assert row["rfmp_supported"] is True
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
    parameters = tuple(inspect.signature(n.build_cross_source_rfmp_features).parameters)
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
