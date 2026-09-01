from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next351_periodic_deviatoric_strain_rigidity as n351
import src.next355_radical_facet_deviatoric_rigidity as n


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
    assert n.PROTOCOL == "2026-08-13-next355-radical-facet-deviatoric-rigidity-v1"
    assert n.FEATURE_NAMES == ("rfdr_deviatoric_retention_floor",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}
    assert n.DESIGN_SHA256 == "cd86db09780a28eb4ddbc993837a46ab9f6852c9bcf35c6bdd719752c6d59059"


def test_rfdr_kernel_is_exactly_the_frozen_pdsr_kernel() -> None:
    assert n.periodic_deviatoric_strain_retention is n351.periodic_deviatoric_strain_retention
    vectors = np.asarray(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]],
        dtype=float,
    )
    result = n.periodic_deviatoric_strain_retention(
        n_sites=1,
        endpoints=np.zeros((6, 2), dtype=int),
        displacements=vectors,
        weights=np.arange(1.0, 7.0),
    )
    assert result.retention_floor == pytest.approx(1.0)
    assert result.generalized_eigenvalues == pytest.approx((1.0,) * 5)


def test_distorted_crystal_has_finite_certified_rfdr() -> None:
    result = n.compute_rfdr_features(_distorted_nacl())
    assert result.supported, result.failure_reason
    assert result.site_count == len(_distorted_nacl())
    assert result.edge_count >= 5
    assert result.minimum_facet_area > 0.0
    assert result.maximum_reciprocal_area_relative_error <= n.RECIPROCAL_AREA_RELATIVE_TOLERANCE
    assert result.volume_tiling_relative_error <= n.VOLUME_TILING_RELATIVE_TOLERANCE
    assert result.maximum_orthogonality_residual <= n351.ORTHOGONALITY_TOLERANCE
    assert 0.0 <= result.features[n.FEATURE_NAMES[0]] <= 1.0


def _feature(atoms: Atoms) -> float:
    result = n.compute_rfdr_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_geometry_equivalences_preserve_rfdr() -> None:
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
        result = n.compute_rfdr_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_row_schema_and_boundary_flags_are_exact() -> None:
    row = n.compute_rfdr_row(_distorted_nacl())
    assert tuple(name for name in row if name.startswith("rfdr_")) == (
        "rfdr_deviatoric_retention_floor", "rfdr_supported", "rfdr_failure",
        "rfdr_site_count", "rfdr_edge_count", "rfdr_minimum_facet_area",
        "rfdr_maximum_reciprocal_area_relative_error",
        "rfdr_maximum_orthogonality_residual", "rfdr_volume_tiling_relative_error",
        "rfdr_affine_minimum_eigenvalue",
    )
    assert row["rfdr_supported"] is True
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())

