from __future__ import annotations

import inspect

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from src.next173_weighted_local_directional_rigidity import (
    FEATURE_NAMES,
    GRAPH_MODES,
    _failure_row,
    build_weighted_local_directional_rigidity_features,
    compute_weighted_local_directional_rigidity,
    weighted_local_directional_rigidity_features,
)


def test_equal_weight_orthogonal_frame_has_unit_certificates() -> None:
    features = weighted_local_directional_rigidity_features(
        n_sites=2,
        endpoints=np.asarray([(0, 1), (0, 1), (0, 1)]),
        vectors=np.eye(3),
        weights=np.ones(3),
        prefix="pwldr_demo",
    )
    np.testing.assert_allclose(list(features.values()), np.ones(5), atol=1.0e-12)


def test_salience_imbalance_changes_tensor_by_exact_weighted_gram() -> None:
    features = weighted_local_directional_rigidity_features(
        n_sites=2,
        endpoints=np.asarray([(0, 1), (0, 1), (0, 1)]),
        vectors=np.eye(3),
        weights=np.asarray([8.0, 1.0, 1.0]),
        prefix="pwldr_demo",
    )
    assert features["pwldr_demo_tightness_min"] == pytest.approx(0.3)
    assert features["pwldr_demo_tightness_q10"] == pytest.approx(0.3)
    assert features["pwldr_demo_tightness_mean"] == pytest.approx(0.3)
    assert features["pwldr_demo_volume_q10"] == pytest.approx(0.216)
    assert features["pwldr_demo_volume_mean"] == pytest.approx(0.216)


def test_kernel_is_invariant_to_rotation_reversal_order_and_weight_scale() -> None:
    endpoints = np.asarray([(0, 1), (0, 1), (0, 1), (0, 1)])
    vectors = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [1.0, 1.0, 1.0]]
    )
    weights = np.asarray([0.2, 0.4, 0.8, 1.0])
    angle = np.pi / 5.0
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    expected = weighted_local_directional_rigidity_features(
        n_sites=2,
        endpoints=endpoints,
        vectors=vectors,
        weights=weights,
        prefix="pwldr_demo",
    )
    order = np.asarray([3, 1, 0, 2])
    transformed = weighted_local_directional_rigidity_features(
        n_sites=2,
        endpoints=endpoints[order][:, ::-1],
        vectors=(-vectors[order]) @ rotation.T,
        weights=7.0 * weights[order],
        prefix="pwldr_demo",
    )
    np.testing.assert_allclose(
        list(expected.values()), list(transformed.values()), atol=1.0e-12
    )


def test_kernel_rejects_nonpositive_weights_and_keeps_isolated_sites_zero() -> None:
    with pytest.raises(ValueError, match="weights"):
        weighted_local_directional_rigidity_features(
            n_sites=2,
            endpoints=np.asarray([(0, 1)]),
            vectors=np.asarray([[1.0, 0.0, 0.0]]),
            weights=np.asarray([0.0]),
            prefix="pwldr_demo",
        )
    features = weighted_local_directional_rigidity_features(
        n_sites=3,
        endpoints=np.asarray([(0, 1), (0, 1), (0, 1)]),
        vectors=np.eye(3),
        weights=np.ones(3),
        prefix="pwldr_demo",
    )
    assert features["pwldr_demo_tightness_min"] == 0.0
    assert features["pwldr_demo_volume_q10"] == 0.0


def test_real_structure_features_are_supercell_invariant() -> None:
    structure = Structure.from_spacegroup(
        225,
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    supercell = structure.copy()
    supercell.make_supercell([2, 1, 1])
    primitive = compute_weighted_local_directional_rigidity(structure)
    repeated = compute_weighted_local_directional_rigidity(supercell)
    assert set(FEATURE_NAMES) == {
        f"pwldr_{mode}_{suffix}"
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
        assert primitive[f"pwldr_{mode}_supported"] is True
        assert repeated[f"pwldr_{mode}_supported"] is True
        for name in FEATURE_NAMES:
            if name.startswith(f"pwldr_{mode}_"):
                np.testing.assert_allclose(primitive[name], repeated[name], atol=1.0e-12)


def test_fail_open_schema_and_batch_interface_exclude_endpoint_paths(tmp_path) -> None:
    row = _failure_row("parse failed")
    assert all(np.isnan(row[name]) for name in FEATURE_NAMES)
    for mode in GRAPH_MODES:
        assert row[f"pwldr_{mode}_supported"] is False
        assert row[f"pwldr_{mode}_failure"] == "parse failed"
    assert tuple(
        inspect.signature(build_weighted_local_directional_rigidity_features).parameters
    ) == (
        "scigen_cohort_dir",
        "wyformer_cohort_dir",
        "design_path",
        "output_dir",
        "workers",
        "require_formal_inputs",
    )
    with pytest.raises(FileNotFoundError, match="NEXT173 input is missing"):
        build_weighted_local_directional_rigidity_features(
            scigen_cohort_dir=tmp_path / "missing-scigen",
            wyformer_cohort_dir=tmp_path / "missing-wyformer",
            design_path=tmp_path / "missing-design.md",
            output_dir=tmp_path / "output",
            workers=1,
            require_formal_inputs=False,
        )
