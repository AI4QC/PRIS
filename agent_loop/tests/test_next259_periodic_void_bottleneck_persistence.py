from __future__ import annotations

import inspect

from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

from src.next259_periodic_void_bottleneck_persistence import (
    FEATURE_NAMES,
    METRIC_NAMES,
    PROTOCOL,
    STATISTIC_NAMES,
    aggregate_pvbp_features,
    annotate_periodic_bottlenecks,
    build_cross_source_pvbp_features,
    compute_pvbp_features,
    translation_rank,
)


def test_protocol_metric_and_feature_universes_are_exact() -> None:
    assert PROTOCOL == "2026-08-09-next259-periodic-void-bottleneck-persistence-v1"
    assert METRIC_NAMES == (
        "isolation_any",
        "isolation_3d",
        "prominence_any",
        "radius",
    )
    assert STATISTIC_NAMES == ("mean", "q75", "q90", "upper_quartile_mean")
    assert FEATURE_NAMES == tuple(
        f"pvbp_{metric}_{statistic}"
        for metric in METRIC_NAMES
        for statistic in STATISTIC_NAMES
    )


def test_translation_rank_is_exact_and_order_invariant() -> None:
    assert translation_rank([]) == 0
    assert translation_rank([(2, 0, 0), (-1, 0, 0)]) == 1
    assert translation_rank([(2, 0, 0), (0, -3, 0), (2, -3, 0)]) == 2
    vectors = [(2, 0, 0), (0, -3, 0), (0, 0, 5), (2, -3, 5)]
    assert translation_rank(vectors) == 3
    assert translation_rank(list(reversed(vectors)) + [(0, 0, 0)]) == 3


def test_periodic_bottleneck_annotation_tracks_rank_onsets_and_new_members() -> None:
    edges = [
        (0, 0, (1, 0, 0), 0.9),
        (0, 0, (0, 1, 0), 0.8),
        (0, 0, (0, 0, 1), 0.7),
        (0, 1, (0, 0, 0), 0.6),
    ]
    any_rank, rank3 = annotate_periodic_bottlenecks(2, edges)
    np.testing.assert_allclose(any_rank, [0.9, 0.6], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(rank3, [0.7, 0.6], rtol=0.0, atol=0.0)

    reversed_and_duplicated = [
        (v, u, tuple(-value for value in image), capacity)
        for u, v, image, capacity in reversed(edges)
    ] + edges
    other_any, other_rank3 = annotate_periodic_bottlenecks(
        2, reversed_and_duplicated
    )
    np.testing.assert_allclose(other_any, any_rank, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(other_rank3, rank3, rtol=0.0, atol=0.0)


def test_periodic_bottleneck_annotation_rejects_invalid_graphs() -> None:
    with pytest.raises(ValueError, match="positive exact integer"):
        annotate_periodic_bottlenecks(0, [])
    with pytest.raises(ValueError, match="endpoint"):
        annotate_periodic_bottlenecks(2, [(0, 2, (0, 0, 0), 1.0)])
    with pytest.raises(ValueError, match="translation"):
        annotate_periodic_bottlenecks(2, [(0, 1, (0, 0), 1.0)])
    with pytest.raises(ValueError, match="capacity"):
        annotate_periodic_bottlenecks(2, [(0, 1, (0, 0, 0), np.nan)])


def test_upper_tail_aggregates_are_exactly_replication_invariant() -> None:
    population = np.asarray([0.1, 0.2, 0.2, 0.7, 0.9, 0.9, 0.9, 1.0])
    metrics = {name: population for name in METRIC_NAMES}
    result = aggregate_pvbp_features(metrics)
    replicated = aggregate_pvbp_features(
        {name: np.repeat(population, 7) for name in METRIC_NAMES}
    )
    assert result == pytest.approx(replicated, rel=0.0, abs=0.0)
    for metric in METRIC_NAMES:
        assert result[f"pvbp_{metric}_mean"] == pytest.approx(0.6125)
        assert result[f"pvbp_{metric}_q75"] == pytest.approx(0.9)
        assert result[f"pvbp_{metric}_q90"] == pytest.approx(1.0)
        assert result[f"pvbp_{metric}_upper_quartile_mean"] == pytest.approx(0.925)


def test_known_fcc_and_nacl_void_networks() -> None:
    fcc = compute_pvbp_features(bulk("Cu", "fcc", a=3.6))
    nacl = compute_pvbp_features(bulk("NaCl", "rocksalt", a=5.64))
    assert fcc.supported is True, fcc.failure_reason
    assert nacl.supported is True, nacl.failure_reason
    assert (fcc.node_count, fcc.edge_count) == (3, 8)
    assert (nacl.node_count, nacl.edge_count) == (2, 6)
    assert fcc.features["pvbp_radius_q75"] == pytest.approx(
        2.0 ** (-1.0 / 3.0), abs=3.0e-10
    )
    assert nacl.features["pvbp_radius_mean"] == pytest.approx(
        np.sqrt(3.0) / 2.0, abs=3.0e-10
    )
    assert nacl.features["pvbp_isolation_any_mean"] == pytest.approx(
        1.0 - np.sqrt(2.0 / 3.0), abs=3.0e-10
    )
    assert nacl.features["pvbp_isolation_3d_mean"] == pytest.approx(
        nacl.features["pvbp_isolation_any_mean"], abs=0.0
    )


def test_geometry_representation_and_supercell_invariance() -> None:
    atoms = bulk("Cu", "fcc", a=3.6)
    reference = compute_pvbp_features(atoms)
    rotated = atoms.copy()
    rotated.rotate(31.0, "z", rotate_cell=True)
    scaled = atoms.copy()
    scaled.set_cell(1.7 * scaled.cell.array, scale_atoms=True)
    translated = atoms.copy()
    translated.translate(0.173 * atoms.cell[0] - 0.291 * atoms.cell[1] + 0.419 * atoms.cell[2])
    translated.wrap()
    supercell = atoms.repeat((2, 1, 1))
    conventional = bulk("Cu", "fcc", a=3.6, cubic=True)
    for result in (
        reference,
        compute_pvbp_features(rotated),
        compute_pvbp_features(scaled),
        compute_pvbp_features(translated),
        compute_pvbp_features(supercell),
        compute_pvbp_features(conventional),
    ):
        assert result.supported is True, result.failure_reason
        assert tuple(result.features) == FEATURE_NAMES
        np.testing.assert_allclose(
            [result.features[name] for name in FEATURE_NAMES],
            [reference.features[name] for name in FEATURE_NAMES],
            rtol=0.0,
            atol=3.0e-10,
        )


def test_geometry_boundary_fails_open() -> None:
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
        result = compute_pvbp_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)
        assert result.features == {}


def test_builder_interface_is_discovery_geometry_only() -> None:
    parameters = tuple(inspect.signature(build_cross_source_pvbp_features).parameters)
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
    with pytest.raises(FileNotFoundError, match="NEXT259 input is missing"):
        build_cross_source_pvbp_features(
            scigen_cohort_dir=tmp_path / "scigen",
            wyformer_cohort_dir=tmp_path / "wyformer",
            design_path=tmp_path / "design",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
