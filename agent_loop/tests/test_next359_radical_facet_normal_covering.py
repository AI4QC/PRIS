from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next359_radical_facet_normal_covering as n


def _regular_tetrahedron_directions() -> np.ndarray:
    return np.asarray(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float
    ) / np.sqrt(3.0)


def _distorted_nacl() -> Atoms:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.set_cell(
        np.asarray([[5.64, 0.0, 0.0], [0.27, 5.77, 0.0], [0.18, 0.31, 5.53]]),
        scale_atoms=True,
    )
    atoms.positions[1] += np.asarray([0.08, -0.04, 0.06])
    atoms.wrap()
    return atoms


def test_frozen_schema_is_one_protected_high_feature() -> None:
    assert n.PROTOCOL == "2026-08-13-next359-radical-facet-normal-covering-v1"
    assert n.DESIGN_SHA256 == "b7a278ccaacc81800938edb21a10096aaa70aac3af3aeb3c336f53e940e52dac"
    assert n.FEATURE_NAMES == ("rfnc_directional_covering_floor_q10",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}


def test_analytic_tetrahedral_and_octahedral_covering_radii() -> None:
    tetrahedron = n.normal_covering_radius(_regular_tetrahedron_directions())
    octahedron = n.normal_covering_radius(
        np.asarray(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
            dtype=float,
        )
    )
    assert tetrahedron == pytest.approx(1.0 / 3.0, abs=1.0e-12)
    assert octahedron == pytest.approx(1.0 / np.sqrt(3.0), abs=1.0e-12)


def test_covering_radius_is_rotation_invariant_and_monotone_under_added_direction() -> None:
    directions = _regular_tetrahedron_directions()
    angle = np.deg2rad(37.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]],
        dtype=float,
    )
    reference = n.normal_covering_radius(directions)
    assert n.normal_covering_radius(directions @ rotation.T) == pytest.approx(
        reference, abs=1.0e-12
    )
    enriched = np.vstack((directions, np.asarray([[1.0, 0.0, 0.0]])))
    assert n.normal_covering_radius(enriched) >= reference - 1.0e-12


def test_rank_deficient_or_non_enclosing_directions_fail_closed() -> None:
    with pytest.raises(ValueError, match="three-dimensional"):
        n.normal_covering_radius(
            np.asarray([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=float)
        )
    with pytest.raises(ValueError, match="strictly enclose"):
        n.normal_covering_radius(
            np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
        )


def _feature(atoms: Atoms) -> float:
    result = n.compute_rfnc_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_distorted_crystal_has_finite_certified_feature() -> None:
    result = n.compute_rfnc_features(_distorted_nacl())
    assert result.supported, result.failure_reason
    assert result.site_count == len(_distorted_nacl())
    assert result.edge_count >= 3
    assert result.minimum_unique_facet_count >= 4
    assert result.minimum_facet_area > 0.0
    assert result.maximum_reciprocal_area_relative_error <= n.RECIPROCAL_AREA_RELATIVE_TOLERANCE
    assert result.volume_tiling_relative_error <= n.VOLUME_TILING_RELATIVE_TOLERANCE
    assert 0.0 < result.minimum_site_covering <= result.maximum_site_covering <= 1.0
    assert 0.0 <= result.features[n.FEATURE_NAMES[0]] <= 1.0


def test_geometry_equivalences_preserve_feature() -> None:
    atoms = _distorted_nacl()
    reference = _feature(atoms)
    rotated = atoms.copy(); rotated.rotate(31.0, "z", rotate_cell=True)
    translated = atoms.copy(); translated.translate([0.173, 0.291, 0.419]); translated.wrap()
    permuted = atoms[[3, 0, 6, 1, 7, 4, 2, 5]]
    rebased = atoms.copy(); rebased.set_cell(
        np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=int) @ atoms.cell.array,
        scale_atoms=False,
    ); rebased.wrap()
    replicated = atoms.repeat((2, 1, 1))
    for equivalent in (rotated, translated, permuted, rebased, replicated):
        assert _feature(equivalent) == pytest.approx(reference, abs=1.0e-8)


def test_geometry_boundary_fails_closed() -> None:
    atoms = _distorted_nacl()
    with_calculator = atoms.copy(); with_calculator.calc = Calculator()
    with_metadata = atoms.copy(); with_metadata.info["outcome"] = 1
    with_array = atoms.copy(); with_array.new_array("energy", np.zeros(len(with_array)))
    nonperiodic = atoms.copy(); nonperiodic.pbc = False
    nonfinite = atoms.copy(); nonfinite.positions[0, 0] = np.nan
    for changed in (with_calculator, with_metadata, with_array, nonperiodic, nonfinite):
        result = n.compute_rfnc_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_row_schema_and_boundary_flags_are_exact() -> None:
    row = n.compute_rfnc_row(_distorted_nacl())
    assert tuple(name for name in row if name.startswith("rfnc_")) == (
        "rfnc_directional_covering_floor_q10", "rfnc_supported", "rfnc_failure",
        "rfnc_site_count", "rfnc_edge_count", "rfnc_minimum_unique_facet_count",
        "rfnc_maximum_unique_facet_count", "rfnc_minimum_facet_area",
        "rfnc_maximum_reciprocal_area_relative_error",
        "rfnc_volume_tiling_relative_error", "rfnc_minimum_site_covering",
        "rfnc_maximum_site_covering",
    )
    assert row["rfnc_supported"] is True
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())
