from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator

from src.next323_periodic_global_contact_equilibrium import (
    BOUNDARY_FLAGS,
    FEATURE_DIRECTIONS,
    FEATURE_NAMES,
    PROTOCOL,
    build_cross_source_pgce_features,
    compute_pgce_features,
    compute_pgce_row,
    positive_contact_equilibrium_floor,
)


def _reciprocal_population(
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    direction = np.asarray(directions, dtype=float)
    count = len(direction)
    centers = np.concatenate(
        (np.zeros(count, dtype=int), np.ones(count, dtype=int))
    )
    neighbors = np.concatenate(
        (np.ones(count, dtype=int), np.zeros(count, dtype=int))
    )
    translations = np.zeros((2 * count, 3), dtype=int)
    translations[:count, 0] = np.arange(count, dtype=int)
    translations[count:, 0] = -np.arange(count, dtype=int)
    vectors = np.vstack((direction, -direction))
    return centers, neighbors, translations, vectors


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


def _feature_value(atoms: Atoms) -> float:
    result = compute_pgce_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[FEATURE_NAMES[0]])


def test_frozen_schema_contains_one_protected_high_hypothesis() -> None:
    assert PROTOCOL == "2026-08-09-next323-periodic-global-contact-equilibrium-v1"
    assert FEATURE_NAMES == ("pgce_all_facet_participation_floor",)
    assert FEATURE_DIRECTIONS == {
        "pgce_all_facet_participation_floor": "protected_high"
    }


def test_uniform_positive_equilibrium_has_unit_participation_floor() -> None:
    directions = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    )
    centers, neighbors, translations, vectors = _reciprocal_population(directions)

    result = positive_contact_equilibrium_floor(
        n_sites=2,
        centers=centers,
        neighbors=neighbors,
        translations=translations,
        vectors=vectors,
    )

    assert result.supported
    assert result.error is None
    assert result.participation_floor == pytest.approx(1.0, abs=1.0e-10)
    assert result.maximum_equilibrium_residual <= 1.0e-9
    assert result.reciprocal_pair_count == 4


def test_imbalanced_positive_equilibrium_has_analytic_five_sixths_floor() -> None:
    directions = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    centers, neighbors, translations, vectors = _reciprocal_population(directions)

    result = positive_contact_equilibrium_floor(
        n_sites=2,
        centers=centers,
        neighbors=neighbors,
        translations=translations,
        vectors=vectors,
    )

    assert result.supported
    assert result.participation_floor == pytest.approx(5.0 / 6.0, abs=1.0e-9)
    assert result.maximum_equilibrium_residual <= 1.0e-9


def test_one_sided_population_has_zero_floor_without_becoming_unsupported() -> None:
    directions = np.eye(3)
    centers, neighbors, translations, vectors = _reciprocal_population(directions)

    result = positive_contact_equilibrium_floor(
        n_sites=2,
        centers=centers,
        neighbors=neighbors,
        translations=translations,
        vectors=vectors,
    )

    assert result.supported
    assert result.error is None
    assert result.participation_floor == 0.0
    assert np.isnan(result.maximum_equilibrium_residual)


def test_kernel_rejects_missing_or_nonopposite_reciprocal_incidences() -> None:
    centers, neighbors, translations, vectors = _reciprocal_population(np.eye(3))
    with pytest.raises(ValueError, match="reciprocal"):
        positive_contact_equilibrium_floor(
            n_sites=2,
            centers=centers[:-1],
            neighbors=neighbors[:-1],
            translations=translations[:-1],
            vectors=vectors[:-1],
        )
    corrupted = vectors.copy()
    corrupted[-1] = [0.0, 0.0, -2.0]
    with pytest.raises(ValueError, match="reciprocal"):
        positive_contact_equilibrium_floor(
            n_sites=2,
            centers=centers,
            neighbors=neighbors,
            translations=translations,
            vectors=corrupted,
        )


def test_standard_and_distorted_crystals_produce_finite_feature() -> None:
    cases = (
        bulk("NaCl", "rocksalt", a=5.64, cubic=True),
        bulk("CsCl", "cesiumchloride", a=4.12, cubic=True),
        bulk("ZnS", "zincblende", a=5.41, cubic=True),
        _distorted_nacl(),
    )
    for atoms in cases:
        result = compute_pgce_features(atoms)
        assert result.supported, result.failure_reason
        assert 0.0 <= result.features[FEATURE_NAMES[0]] <= 1.0
        assert result.site_count == len(atoms)
        assert result.directed_contact_count == 2 * result.reciprocal_pair_count


def test_geometry_equivalences_preserve_pgce_feature() -> None:
    atoms = _distorted_nacl()
    reference = _feature_value(atoms)
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
        assert _feature_value(equivalent) == pytest.approx(reference, abs=1.0e-10)


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
        result = compute_pgce_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_row_schema_and_boundary_flags_are_exact() -> None:
    row = compute_pgce_row(_distorted_nacl())
    assert tuple(row) == (
        "pgce_supported",
        "pgce_failure",
        "pgce_site_count",
        "pgce_directed_contact_count",
        "pgce_reciprocal_pair_count",
        "pgce_maximum_equilibrium_residual",
        *FEATURE_NAMES,
    )
    assert row["pgce_supported"] is True
    assert np.isfinite(row[FEATURE_NAMES[0]])
    assert BOUNDARY_FLAGS == {
        "dft_calculation_executed": False,
        "dft_values_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
    }


def test_builder_interface_is_discovery_geometry_only() -> None:
    parameters = tuple(inspect.signature(build_cross_source_pgce_features).parameters)
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


def test_builder_fails_closed_on_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT323 input is missing"):
        build_cross_source_pgce_features(
            scigen_cohort_dir=tmp_path / "scigen",
            wyformer_cohort_dir=tmp_path / "wyformer",
            design_path=tmp_path / "design",
            probe_result_path=tmp_path / "probe",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
