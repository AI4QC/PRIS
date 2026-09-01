from __future__ import annotations

import inspect

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

from src.next327_radical_facet_positive_enclosure import (
    BOUNDARY_FLAGS,
    FEATURE_DIRECTIONS,
    FEATURE_NAMES,
    PROTOCOL,
    build_cross_source_rfpe_features,
    compute_rfpe_features,
    compute_rfpe_row,
    radical_facet_positive_enclosure_features,
    unique_facet_directions,
)


def _tetrahedron() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=float,
    )


def test_frozen_schema_is_one_protected_high_q10() -> None:
    assert PROTOCOL == "2026-08-13-next327-radical-facet-positive-enclosure-v1"
    assert FEATURE_NAMES == ("rfpe_uniform_equilibrium_q10",)
    assert FEATURE_DIRECTIONS == {
        "rfpe_uniform_equilibrium_q10": "protected_high"
    }


def test_balanced_tetrahedral_site_has_unit_margin() -> None:
    result = radical_facet_positive_enclosure_features(
        n_sites=1,
        centers=np.zeros(4, dtype=int),
        vectors=_tetrahedron(),
    )
    assert result == {"rfpe_uniform_equilibrium_q10": pytest.approx(1.0)}


def test_one_sided_site_has_zero_margin() -> None:
    result = radical_facet_positive_enclosure_features(
        n_sites=1,
        centers=np.zeros(4, dtype=int),
        vectors=np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
        ),
    )
    assert result == {"rfpe_uniform_equilibrium_q10": 0.0}


def test_geometrically_duplicate_facet_directions_are_removed() -> None:
    directions = np.vstack((_tetrahedron(), 3.0 * _tetrahedron()[0]))
    unique = unique_facet_directions(directions)
    assert unique.shape == (4, 3)
    result = radical_facet_positive_enclosure_features(
        n_sites=1,
        centers=np.zeros(len(directions), dtype=int),
        vectors=directions,
    )
    assert result == {"rfpe_uniform_equilibrium_q10": pytest.approx(1.0)}


def test_kernel_validates_site_alignment_and_finite_nonzero_vectors() -> None:
    with pytest.raises(ValueError, match="centers differ"):
        radical_facet_positive_enclosure_features(
            n_sites=1, centers=[0, 1], vectors=np.eye(2, 3)
        )
    with pytest.raises(ValueError, match="vectors differ"):
        radical_facet_positive_enclosure_features(
            n_sites=1, centers=[0], vectors=[[0.0, 0.0, 0.0]]
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


def _feature(atoms: Atoms) -> float:
    result = compute_rfpe_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[FEATURE_NAMES[0]])


def test_standard_and_distorted_crystals_have_finite_rfpe() -> None:
    for atoms in (
        bulk("NaCl", "rocksalt", a=5.64, cubic=True),
        bulk("CsCl", "cesiumchloride", a=4.12, cubic=True),
        bulk("ZnS", "zincblende", a=5.41, cubic=True),
        _distorted_nacl(),
    ):
        result = compute_rfpe_features(atoms)
        assert result.supported, result.failure_reason
        assert result.site_count == len(atoms)
        assert result.directed_contact_count >= 4 * len(atoms)
        assert result.minimum_unique_facet_count >= 4
        assert result.maximum_unique_facet_count >= result.minimum_unique_facet_count
        assert 0.0 <= result.features[FEATURE_NAMES[0]] <= 1.0


def test_geometry_equivalences_preserve_rfpe() -> None:
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
    for changed in (
        with_calculator,
        with_metadata,
        with_array,
        nonperiodic,
        nonfinite,
    ):
        result = compute_rfpe_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_row_schema_and_boundary_flags_are_exact() -> None:
    row = compute_rfpe_row(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    assert tuple(name for name in row if name.startswith("rfpe_")) == (
        "rfpe_uniform_equilibrium_q10",
        "rfpe_supported",
        "rfpe_failure",
        "rfpe_site_count",
        "rfpe_directed_contact_count",
        "rfpe_minimum_unique_facet_count",
        "rfpe_maximum_unique_facet_count",
    )
    assert row["rfpe_supported"] is True
    assert BOUNDARY_FLAGS == {
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
    parameters = tuple(inspect.signature(build_cross_source_rfpe_features).parameters)
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
