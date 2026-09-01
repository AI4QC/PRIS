from __future__ import annotations

import inspect
import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from src.next168_periodic_local_directional_rigidity import (
    FEATURE_NAMES,
    GRAPH_MODES,
    _failure_row,
    build_periodic_local_directional_rigidity_features,
    compute_periodic_local_directional_rigidity,
    local_directional_rigidity_features,
)


def test_orthogonal_frame_has_unit_tightness_and_volume() -> None:
    features = local_directional_rigidity_features(
        n_sites=2,
        endpoints=np.asarray([(0, 1), (0, 1), (0, 1)]),
        vectors=np.eye(3),
        prefix="pldr_demo",
    )
    assert tuple(features) == (
        "pldr_demo_tightness_min",
        "pldr_demo_tightness_q10",
        "pldr_demo_tightness_mean",
        "pldr_demo_volume_q10",
        "pldr_demo_volume_mean",
    )
    np.testing.assert_allclose(list(features.values()), np.ones(5), atol=1.0e-12)


def test_planar_frame_has_zero_weak_direction_certificates() -> None:
    features = local_directional_rigidity_features(
        n_sites=2,
        endpoints=np.asarray([(0, 1), (0, 1), (0, 1)]),
        vectors=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
        prefix="pldr_demo",
    )
    np.testing.assert_allclose(list(features.values()), np.zeros(5), atol=1.0e-12)


def test_kernel_is_invariant_to_rotation_reversal_order_and_uniform_duplication() -> None:
    endpoints = np.asarray([(0, 1), (0, 1), (0, 1), (0, 1)])
    vectors = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [1.0, 1.0, 1.0]]
    )
    angle = np.pi / 5.0
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    expected = local_directional_rigidity_features(
        n_sites=2, endpoints=endpoints, vectors=vectors, prefix="pldr_demo"
    )
    order = np.asarray([3, 1, 0, 2])
    transformed = local_directional_rigidity_features(
        n_sites=2,
        endpoints=np.tile(endpoints[order][:, ::-1], (2, 1)),
        vectors=np.tile((-vectors[order]) @ rotation.T, (2, 1)),
        prefix="pldr_demo",
    )
    np.testing.assert_allclose(
        list(expected.values()), list(transformed.values()), atol=1.0e-12
    )


def test_kernel_validates_inputs_and_keeps_bounded_schema() -> None:
    with pytest.raises(ValueError, match="endpoints"):
        local_directional_rigidity_features(
            n_sites=2,
            endpoints=np.asarray([(0, 2)]),
            vectors=np.asarray([[1.0, 0.0, 0.0]]),
            prefix="pldr_demo",
        )
    features = local_directional_rigidity_features(
        n_sites=3,
        endpoints=np.asarray([(0, 1), (0, 1), (0, 1)]),
        vectors=np.eye(3),
        prefix="pldr_demo",
    )
    assert all(0.0 <= value <= 1.0 for value in features.values())
    assert features["pldr_demo_tightness_min"] == 0.0
    assert features["pldr_demo_volume_q10"] == 0.0


def test_real_structure_features_are_supercell_invariant() -> None:
    structure = Structure.from_spacegroup(
        225,
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    supercell = structure.copy()
    supercell.make_supercell([2, 1, 1])
    primitive = compute_periodic_local_directional_rigidity(structure)
    repeated = compute_periodic_local_directional_rigidity(supercell)
    assert set(FEATURE_NAMES) == {
        f"pldr_{mode}_{suffix}"
        for mode in GRAPH_MODES
        for suffix in (
            "tightness_min",
            "tightness_q10",
            "tightness_mean",
            "volume_q10",
            "volume_mean",
        )
    }
    for mode in GRAPH_MODES:
        assert primitive[f"pldr_{mode}_supported"] is True
        assert repeated[f"pldr_{mode}_supported"] is True
        for name in FEATURE_NAMES:
            if name.startswith(f"pldr_{mode}_"):
                np.testing.assert_allclose(
                    primitive[name], repeated[name], atol=1.0e-12
                )


def test_failure_row_preserves_complete_independent_fail_open_schema() -> None:
    row = _failure_row("parse failed")
    assert all(np.isnan(row[name]) for name in FEATURE_NAMES)
    for mode in GRAPH_MODES:
        assert row[f"pldr_{mode}_supported"] is False
        assert row[f"pldr_{mode}_failure"] == "parse failed"


def test_batch_builder_interface_has_no_endpoint_or_holdout_input(tmp_path) -> None:
    assert tuple(inspect.signature(build_periodic_local_directional_rigidity_features).parameters) == (
        "scigen_cohort_dir",
        "wyformer_cohort_dir",
        "design_path",
        "output_dir",
        "workers",
        "require_formal_inputs",
    )
    with pytest.raises(FileNotFoundError, match="NEXT168 input is missing"):
        build_periodic_local_directional_rigidity_features(
            scigen_cohort_dir=tmp_path / "missing-scigen",
            wyformer_cohort_dir=tmp_path / "missing-wyformer",
            design_path=tmp_path / "missing-design.md",
            output_dir=tmp_path / "output",
            workers=1,
            require_formal_inputs=False,
        )
