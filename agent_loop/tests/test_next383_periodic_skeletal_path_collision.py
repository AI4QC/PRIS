from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next383_periodic_skeletal_path_collision as n


def _directed(rows):
    endpoints = []
    weights = []
    for left, right, image, weight in rows:
        reverse = tuple(-int(value) for value in image)
        endpoints.extend(((left, right, *image), (right, left, *reverse)))
        weights.extend((weight, weight))
    return np.asarray(endpoints, dtype=int), np.asarray(weights, dtype=float)


def _distorted_nacl() -> Atoms:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.set_cell(
        np.asarray([[5.64, 0.0, 0.0], [0.21, 5.76, 0.0], [0.13, 0.29, 5.51]]),
        scale_atoms=True,
    )
    atoms.positions[1] += np.asarray([0.07, -0.03, 0.05])
    atoms.wrap()
    return atoms


def _feature(atoms: Atoms) -> float:
    result = n.compute_pspc_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_protocol_schema_depth_and_direction_are_frozen() -> None:
    assert n.PROTOCOL == "2026-08-13-next383-periodic-skeletal-path-collision-v1"
    assert n.FEATURE_NAMES == ("pspc_skeletal_nb3_collision_q10",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}
    assert n.WALK_DEPTH == 3
    assert len(n.DESIGN_SHA256) == 64


def test_periodic_chain_has_no_nonbacktracking_collision() -> None:
    endpoints, weights = _directed(((0, 0, (1, 0, 0), 1.0),))
    result = n.skeletal_path_collision(
        n_sites=1, endpoints=endpoints, solid_angles=weights
    )
    assert result.supported, result.failure_reason
    assert result.skeleton_threshold == 0.0
    assert result.total_walk_count == 2
    assert result.total_endpoint_count == 2
    assert result.site_collisions == (0.0,)
    assert result.features[n.FEATURE_NAMES[0]] == 0.0


def test_simple_cubic_net_has_hand_checkable_path_collision() -> None:
    endpoints, weights = _directed(
        (
            (0, 0, (1, 0, 0), 1.0),
            (0, 0, (0, 1, 0), 1.0),
            (0, 0, (0, 0, 1), 1.0),
        )
    )
    result = n.skeletal_path_collision(
        n_sites=1, endpoints=endpoints, solid_angles=weights
    )
    assert result.supported, result.failure_reason
    assert result.rank3_site_count == 1
    assert result.skeleton_threshold == 1.0
    assert result.total_walk_count == 150
    assert result.total_endpoint_count == 44
    assert result.site_collisions == pytest.approx((1.0 - 44.0 / 150.0,))
    assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(0.7066666667)


def test_skeletal_threshold_includes_complete_exact_tie_batch() -> None:
    endpoints, weights = _directed(
        (
            (0, 0, (1, 0, 0), 1.0),
            (0, 0, (0, 1, 0), 0.5),
            (0, 0, (0, 0, 1), 0.5),
        )
    )
    result = n.skeletal_path_collision(
        n_sites=1, endpoints=endpoints, solid_angles=weights
    )
    assert result.supported, result.failure_reason
    assert result.skeleton_threshold == 0.5
    assert result.skeleton_edge_count == 3


def test_kernel_is_order_scale_and_disjoint_replication_invariant() -> None:
    endpoints, weights = _directed(
        (
            (0, 0, (1, 0, 0), 0.8),
            (0, 0, (0, 1, 0), 0.6),
            (0, 0, (0, 0, 1), 0.4),
        )
    )
    order = np.asarray((4, 1, 5, 0, 3, 2))
    reference = n.skeletal_path_collision(
        n_sites=1, endpoints=endpoints, solid_angles=weights
    )
    reordered = n.skeletal_path_collision(
        n_sites=1, endpoints=endpoints[order], solid_angles=7.0 * weights[order]
    )
    replicated = n.skeletal_path_collision(
        n_sites=2,
        endpoints=np.vstack((endpoints, endpoints + np.asarray((1, 1, 0, 0, 0)))),
        solid_angles=np.concatenate((weights, weights)),
    )
    for result in (reference, reordered, replicated):
        assert result.supported, result.failure_reason
        assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(
            reference.features[n.FEATURE_NAMES[0]], abs=1e-14
        )


@pytest.mark.parametrize(
    ("n_sites", "endpoints", "weights"),
    (
        (2, [[0, 1, 0, 0, 0]], [1.0]),
        (2, [[0, 2, 0, 0, 0], [2, 0, 0, 0, 0]], [1.0, 1.0]),
        (2, [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], [1.0, 0.9]),
        (2, [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], [0.0, 0.0]),
        (
            2,
            [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
            [1.0, 1.0, 1.0],
        ),
    ),
)
def test_malformed_kernel_inputs_fail_closed(n_sites, endpoints, weights) -> None:
    result = n.skeletal_path_collision(
        n_sites=n_sites,
        endpoints=np.asarray(endpoints),
        solid_angles=np.asarray(weights),
    )
    assert result.supported is False
    assert result.features == {}


def test_periodic_voronoi_structure_is_supported_and_bounded() -> None:
    result = n.compute_pspc_features(_distorted_nacl())
    assert result.supported, result.failure_reason
    assert result.site_count == len(_distorted_nacl())
    assert result.directed_face_count == 2 * result.undirected_edge_count
    assert result.skeleton_edge_count <= result.undirected_edge_count
    assert result.maximum_reverse_angle_error <= 1e-8
    assert 0 <= result.features[n.FEATURE_NAMES[0]] <= 1


def test_structure_equivalences_preserve_feature() -> None:
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
        np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]]) @ atoms.cell.array,
        scale_atoms=False,
    )
    rebased.wrap()
    replicated = atoms.repeat((2, 1, 1))
    for equivalent in (rotated, translated, permuted, rebased, replicated):
        assert _feature(equivalent) == pytest.approx(reference, abs=1e-8)


def test_geometry_only_boundary_fails_closed() -> None:
    atoms = _distorted_nacl()
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
    for changed in (with_calculator, with_metadata, with_array, nonperiodic, nonfinite):
        result = n.compute_pspc_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_row_schema_and_boundary_flags_are_exact() -> None:
    row = n.compute_pspc_row(_distorted_nacl())
    assert tuple(name for name in row if name.startswith("pspc_")) == (
        "pspc_skeletal_nb3_collision_q10",
        "pspc_supported",
        "pspc_failure",
        "pspc_site_count",
        "pspc_directed_face_count",
        "pspc_undirected_edge_count",
        "pspc_rank3_site_count",
        "pspc_skeleton_edge_count",
        "pspc_skeleton_threshold",
        "pspc_total_walk_count",
        "pspc_total_endpoint_count",
        "pspc_maximum_reverse_angle_error",
    )
    assert row["pspc_supported"] is True
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())
