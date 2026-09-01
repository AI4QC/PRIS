from __future__ import annotations

import inspect

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

from src.next291_radius_weighted_delaunay_cage_isotropy import (
    FEATURE_DIRECTIONS,
    FEATURE_NAMES,
    METRIC_NAMES,
    PROTOCOL,
    aggregate_rwdci_features,
    build_cross_source_rwdci_features,
    compute_rwdci_features,
    weighted_cage_metrics,
)


def _regular_tetrahedron() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    ) / np.sqrt(3.0)


def _asymmetric_structure() -> Atoms:
    return Atoms(
        ["Si", "O", "Na"],
        scaled_positions=[
            (0.10, 0.20, 0.30),
            (0.52, 0.63, 0.47),
            (0.81, 0.24, 0.72),
        ],
        cell=[(4.1, 0.0, 0.0), (0.7, 4.4, 0.0), (0.4, 0.8, 4.8)],
        pbc=True,
    )


def _feature_vector(atoms: Atoms) -> np.ndarray:
    result = compute_rwdci_features(atoms)
    assert result.supported, result.failure_reason
    return np.asarray([result.features[name] for name in FEATURE_NAMES], dtype=float)


def test_protocol_metric_feature_and_direction_universes_are_exact() -> None:
    assert PROTOCOL == "2026-08-09-next291-radius-weighted-delaunay-cage-isotropy-v1"
    assert METRIC_NAMES == ("tightness", "volume", "eigenratio", "closure")
    assert FEATURE_NAMES == tuple(
        f"rwdci_{metric}_{statistic}"
        for metric in METRIC_NAMES
        for statistic in ("mean", "q10", "q25", "lower_quartile_mean")
    )
    assert FEATURE_DIRECTIONS == {name: "protected_high" for name in FEATURE_NAMES}


def test_equal_radius_regular_tetrahedron_is_exact() -> None:
    result = weighted_cage_metrics(
        vectors=_regular_tetrahedron(), radii=np.ones(4)
    )
    assert result == pytest.approx(
        {"tightness": 1.0, "volume": 1.0, "eigenratio": 1.0, "closure": 1.0}
    )


def test_unequal_radius_regular_tetrahedron_has_analytic_weighted_result() -> None:
    result = weighted_cage_metrics(
        vectors=_regular_tetrahedron(), radii=np.asarray([2.0, 1.0, 1.0, 1.0])
    )
    assert result == pytest.approx(
        {
            "tightness": 4.0 / 7.0,
            "volume": 27.0 * 4.0 * 4.0 * 13.0 / 21.0**3,
            "eigenratio": 4.0 / 13.0,
            "closure": 4.0 / 7.0,
        },
        abs=1.0e-14,
    )


def test_weighted_metric_guards_fail_closed() -> None:
    directions = _regular_tetrahedron()
    with pytest.raises(ValueError, match="cage population differs"):
        weighted_cage_metrics(vectors=directions[:3], radii=np.ones(3))
    with pytest.raises(ValueError, match="cage radii differ"):
        weighted_cage_metrics(vectors=directions, radii=[1.0, 1.0, 0.0, 1.0])
    changed = directions.copy()
    changed[0] = 0.0
    with pytest.raises(ValueError, match="cage distance differs"):
        weighted_cage_metrics(vectors=changed, radii=np.ones(4))
    changed = directions.copy()
    changed[0, 0] = np.nan
    with pytest.raises(ValueError, match="cage population differs"):
        weighted_cage_metrics(vectors=changed, radii=np.ones(4))


def test_inverse_cdf_aggregation_is_exactly_replication_invariant() -> None:
    population = np.asarray([0.6] * 6 + [1.0] * 8)
    result = aggregate_rwdci_features({name: population for name in METRIC_NAMES})
    replicated = aggregate_rwdci_features(
        {name: np.repeat(population, 8) for name in METRIC_NAMES}
    )
    assert result == pytest.approx(replicated, rel=0.0, abs=0.0)
    for metric in METRIC_NAMES:
        assert result[f"rwdci_{metric}_mean"] == pytest.approx(29.0 / 35.0)
        assert result[f"rwdci_{metric}_q10"] == pytest.approx(0.6)
        assert result[f"rwdci_{metric}_q25"] == pytest.approx(0.6)
        assert result[f"rwdci_{metric}_lower_quartile_mean"] == pytest.approx(0.6)


def test_analytic_one_site_cube_and_standard_crystals_are_supported() -> None:
    cubic = Atoms("H", positions=[(0.0, 0.0, 0.0)], cell=[2.0, 2.0, 2.0], pbc=True)
    result = compute_rwdci_features(cubic)
    assert result.supported, result.failure_reason
    assert result.incidence_count == 8
    assert result.min_cage_size == 8 and result.max_cage_size == 8
    assert result.volume_tiling_relative_error <= 1.0e-12
    assert np.asarray(list(result.features.values())) == pytest.approx(np.ones(16))

    for atoms in (
        bulk("Cu", "fcc", a=3.6, cubic=True),
        bulk("Fe", "bcc", a=2.87, cubic=True),
        bulk("NaCl", "rocksalt", a=5.64, cubic=True),
        bulk("C", "diamond", a=3.57, cubic=True),
    ):
        atoms.arrays.pop("initial_magmoms", None)
        result = compute_rwdci_features(atoms)
        assert result.supported, result.failure_reason
        assert result.incidence_count > 0
        assert result.min_cage_size >= 4
        assert result.volume_tiling_relative_error <= 1.0e-9
        values = np.asarray([result.features[name] for name in FEATURE_NAMES])
        assert np.isfinite(values).all()
        assert ((values >= 0.0) & (values <= 1.0)).all()


def test_geometry_equivalences_preserve_features() -> None:
    atoms = _asymmetric_structure()
    reference = _feature_vector(atoms)

    rotated = atoms.copy()
    rotated.rotate(31.0, "z", rotate_cell=True)
    translated = atoms.copy()
    translated.translate([0.173, 0.291, 0.419])
    translated.wrap()
    permuted = atoms[[2, 0, 1]]
    rebased = atoms.copy()
    rebased.set_cell(
        np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
        @ atoms.cell.array,
        scale_atoms=False,
    )
    rebased.wrap()
    replicated = atoms.repeat((2, 1, 1))

    for equivalent in (rotated, translated, permuted, rebased, replicated):
        np.testing.assert_allclose(
            _feature_vector(equivalent), reference, rtol=0.0, atol=1.0e-8
        )


def test_geometry_boundary_fails_closed() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    with_calculator = atoms.copy()
    with_calculator.calc = Calculator()
    with_metadata = atoms.copy()
    with_metadata.info["outcome"] = 1
    with_array = atoms.copy()
    with_array.new_array("energy", np.zeros(len(with_array)))
    nonperiodic = atoms.copy()
    nonperiodic.pbc = False
    for changed in (with_calculator, with_metadata, with_array, nonperiodic):
        result = compute_rwdci_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_builder_interface_is_discovery_geometry_only() -> None:
    parameters = tuple(inspect.signature(build_cross_source_rwdci_features).parameters)
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
    with pytest.raises(FileNotFoundError, match="NEXT291 input is missing"):
        build_cross_source_rwdci_features(
            scigen_cohort_dir=tmp_path / "scigen",
            wyformer_cohort_dir=tmp_path / "wyformer",
            design_path=tmp_path / "design",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
