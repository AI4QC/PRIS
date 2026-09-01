from __future__ import annotations

import inspect

from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

from src.next255_delaunay_void_cage_isotropy import (
    FEATURE_NAMES,
    METRIC_NAMES,
    PROTOCOL,
    aggregate_dvci_features,
    build_cross_source_dvci_features,
    compute_dvci_features,
    void_cage_metrics,
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


def test_protocol_metric_and_feature_universes_are_exact() -> None:
    assert PROTOCOL == "2026-08-09-next255-delaunay-void-cage-isotropy-v1"
    assert METRIC_NAMES == ("tightness", "volume", "eigenratio", "closure")
    assert FEATURE_NAMES == tuple(
        f"dvci_{metric}_{statistic}"
        for metric in METRIC_NAMES
        for statistic in ("mean", "q10", "q25", "lower_quartile_mean")
    )


def test_regular_tetrahedral_void_cage_is_exact_and_order_invariant() -> None:
    directions = _regular_tetrahedron()
    center_direction = directions[0]
    displacements = directions[1:] - center_direction
    result = void_cage_metrics(displacements)
    reversed_result = void_cage_metrics(displacements[::-1])
    assert result == pytest.approx(
        {"tightness": 1.0, "volume": 1.0, "eigenratio": 1.0, "closure": 1.0}
    )
    assert reversed_result == pytest.approx(result)


def test_anisotropic_cage_metrics_and_fail_closed_guards() -> None:
    directions = np.asarray(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    displacements = directions[1:] - directions[0]
    result = void_cage_metrics(displacements)
    assert result["tightness"] == pytest.approx(0.75)
    assert result["volume"] == pytest.approx(27.0 / 32.0)
    assert result["eigenratio"] == pytest.approx(0.5)
    assert result["closure"] == pytest.approx(1.0 - np.sqrt(1.0 / 8.0))
    with pytest.raises(ValueError, match="incident face population differs"):
        void_cage_metrics(displacements[:2])
    with pytest.raises(ValueError, match="bisector rank differs"):
        void_cage_metrics(np.asarray([[1.0, 0.0, 0.0]] * 3))
    bad = displacements.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="incident face population differs"):
        void_cage_metrics(bad)


def test_inverse_cdf_aggregates_are_exactly_replication_invariant() -> None:
    population = np.asarray([0.6] * 6 + [1.0] * 8)
    metrics = {name: population for name in METRIC_NAMES}
    result = aggregate_dvci_features(metrics)
    replicated = aggregate_dvci_features(
        {name: np.repeat(population, 8) for name in METRIC_NAMES}
    )
    assert result == pytest.approx(replicated, rel=0.0, abs=0.0)
    for metric in METRIC_NAMES:
        assert result[f"dvci_{metric}_mean"] == pytest.approx(29.0 / 35.0)
        assert result[f"dvci_{metric}_q10"] == pytest.approx(0.6)
        assert result[f"dvci_{metric}_q25"] == pytest.approx(0.6)
        assert result[f"dvci_{metric}_lower_quartile_mean"] == pytest.approx(0.6)
    assert np.quantile(population, 0.4, method="linear") != np.quantile(
        np.repeat(population, 8), 0.4, method="linear"
    )


def test_known_fcc_and_nacl_void_cages() -> None:
    fcc = compute_dvci_features(bulk("Cu", "fcc", a=3.6))
    nacl = compute_dvci_features(bulk("NaCl", "rocksalt", a=5.64))
    assert fcc.supported is True, fcc.failure_reason
    assert nacl.supported is True, nacl.failure_reason
    assert fcc.incidence_count == 14
    assert fcc.features["dvci_tightness_mean"] == pytest.approx(29.0 / 35.0)
    assert fcc.features["dvci_tightness_q10"] == pytest.approx(0.6)
    assert fcc.features["dvci_volume_q10"] == pytest.approx(0.864)
    assert fcc.features["dvci_eigenratio_q10"] == pytest.approx(0.5)
    assert fcc.features["dvci_closure_q10"] == pytest.approx(0.8)
    assert nacl.features["dvci_tightness_mean"] == pytest.approx(1.0)
    assert nacl.features["dvci_volume_mean"] == pytest.approx(1.0)
    assert nacl.features["dvci_eigenratio_mean"] == pytest.approx(1.0)
    assert nacl.features["dvci_closure_mean"] == pytest.approx(0.5)


def test_geometry_transform_and_supercell_invariance() -> None:
    atoms = bulk("Cu", "fcc", a=3.6)
    reference = compute_dvci_features(atoms)
    rotated = atoms.copy()
    rotated.rotate(31.0, "z", rotate_cell=True)
    scaled = atoms.copy()
    scaled.set_cell(1.7 * scaled.cell.array, scale_atoms=True)
    translated = atoms.copy()
    translated.translate([0.173, 0.291, 0.419])
    translated.wrap()
    supercell = atoms.repeat((2, 1, 1))
    for result in (
        reference,
        compute_dvci_features(rotated),
        compute_dvci_features(scaled),
        compute_dvci_features(translated),
        compute_dvci_features(supercell),
    ):
        assert result.supported is True, result.failure_reason
        assert tuple(result.features) == FEATURE_NAMES
        np.testing.assert_allclose(
            [result.features[name] for name in FEATURE_NAMES],
            [reference.features[name] for name in FEATURE_NAMES],
            rtol=0.0,
            atol=3.0e-10,
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
        result = compute_dvci_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_builder_interface_is_discovery_geometry_only() -> None:
    parameters = tuple(inspect.signature(build_cross_source_dvci_features).parameters)
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
    with pytest.raises(FileNotFoundError, match="NEXT255 input is missing"):
        build_cross_source_dvci_features(
            scigen_cohort_dir=tmp_path / "scigen",
            wyformer_cohort_dir=tmp_path / "wyformer",
            design_path=tmp_path / "design",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
