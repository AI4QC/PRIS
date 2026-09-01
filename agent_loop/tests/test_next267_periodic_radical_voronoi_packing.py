from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from src.next267_periodic_radical_voronoi_packing import (
    FEATURE_NAMES,
    build_cross_source_prv_features,
    compute_periodic_radical_voronoi_features,
    periodic_radical_cells,
)


def _asymmetric_structure() -> Atoms:
    return Atoms(
        ["Si", "O", "Na"],
        scaled_positions=[(0.10, 0.20, 0.30), (0.52, 0.63, 0.47), (0.81, 0.24, 0.72)],
        cell=[(4.1, 0.0, 0.0), (0.7, 4.4, 0.0), (0.4, 0.8, 4.8)],
        pbc=True,
    )


def _feature_vector(atoms: Atoms) -> np.ndarray:
    result = compute_periodic_radical_voronoi_features(atoms)
    assert result.supported, result.failure_reason
    return np.asarray([result.features[name] for name in FEATURE_NAMES], dtype=float)


def test_feature_catalogue_is_frozen() -> None:
    assert FEATURE_NAMES == (
        "prv_empty_cell_fraction",
        "prv_generator_excluded_fraction",
        "prv_sphere_crossing_fraction",
        "prv_allocation_total_variation",
        "prv_volume_ratio_q10",
        "prv_volume_ratio_q90",
        "prv_volume_ratio_cv",
        "prv_chebyshev_ratio_q10",
        "prv_chebyshev_ratio_q90",
        "prv_chebyshev_ratio_cv",
        "prv_centroid_offset_mean",
        "prv_centroid_offset_q90",
        "prv_vertex_anisotropy_mean",
        "prv_vertex_anisotropy_q90",
        "prv_facet_count_mean",
        "prv_facet_count_cv",
    )


def test_one_site_cubic_power_cell_has_analytic_volume_and_facets() -> None:
    atoms = Atoms("H", positions=[(0.0, 0.0, 0.0)], cell=[2.0, 2.0, 2.0], pbc=True)
    cells = periodic_radical_cells(atoms, radii=np.asarray([0.5]))
    assert len(cells) == 1 and not cells[0].empty
    assert cells[0].volume == pytest.approx(8.0, abs=1.0e-10)
    assert cells[0].chebyshev_radius == pytest.approx(1.0, abs=1.0e-10)
    assert cells[0].generator_margin == pytest.approx(1.0, abs=1.0e-10)
    assert cells[0].facet_count == 6
    assert cells[0].centroid_offset == pytest.approx(0.0, abs=1.0e-10)
    assert cells[0].vertex_anisotropy == pytest.approx(0.0, abs=1.0e-10)


def test_standard_crystals_are_finite_and_tile_the_periodic_cell() -> None:
    structures = (
        bulk("Cu", "fcc", a=3.6, cubic=True),
        bulk("Fe", "bcc", a=2.87, cubic=True),
        bulk("NaCl", "rocksalt", a=5.64, cubic=True),
        bulk("C", "diamond", a=3.57, cubic=True),
    )
    for atoms in structures:
        result = compute_periodic_radical_voronoi_features(atoms)
        assert result.supported, result.failure_reason
        assert result.site_count == len(atoms)
        assert result.volume_tiling_relative_error <= 1.0e-9
        assert np.isfinite([result.features[name] for name in FEATURE_NAMES]).all()


def test_radius_dominated_site_can_have_an_empty_power_cell() -> None:
    atoms = Atoms(
        ["Cs", "H"],
        positions=[(2.0, 4.0, 4.0), (2.2, 4.0, 4.0)],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    cells = periodic_radical_cells(atoms, radii=np.asarray([3.0, 0.1]))
    assert sum(cell.empty for cell in cells) == 1
    assert sum(cell.volume for cell in cells) == pytest.approx(
        atoms.get_volume(), rel=1.0e-7
    )


def test_geometry_equivalences_preserve_all_features() -> None:
    atoms = _asymmetric_structure()
    reference = _feature_vector(atoms)

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
    translated.positions += np.asarray([1.37, -0.62, 0.91])
    translated.wrap()

    permuted = atoms[[2, 0, 1]]

    rebased = atoms.copy()
    operation = np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
    rebased.set_cell(operation @ atoms.cell.array, scale_atoms=False)
    rebased.wrap()

    replicated = atoms.repeat((2, 1, 1))
    for equivalent in (rotated, translated, permuted, rebased, replicated):
        np.testing.assert_allclose(
            _feature_vector(equivalent), reference, rtol=0.0, atol=1.0e-8
        )


def test_malformed_geometry_fails_closed() -> None:
    nonperiodic = Atoms("H", positions=[(0.0, 0.0, 0.0)], cell=[2.0, 2.0, 2.0])
    result = compute_periodic_radical_voronoi_features(nonperiodic)
    assert not result.supported and "periodic" in str(result.failure_reason)

    coincident = Atoms(
        ["H", "He"],
        positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )
    result = compute_periodic_radical_voronoi_features(coincident)
    assert not result.supported and "zero-distance" in str(result.failure_reason)


def test_formal_builder_has_no_endpoint_or_validation_inputs() -> None:
    parameters = tuple(inspect.signature(build_cross_source_prv_features).parameters)
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
        for token in ("endpoint", "label", "validation", "replication", "relaxed")
    )
