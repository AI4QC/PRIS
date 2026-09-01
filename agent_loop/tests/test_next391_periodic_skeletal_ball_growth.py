from __future__ import annotations

from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next391_periodic_skeletal_ball_growth as n


def _directed(rows):
    endpoints, weights = [], []
    for left, right, image, weight in rows:
        reverse = tuple(-int(x) for x in image)
        endpoints.extend(((left, right, *image), (right, left, *reverse)))
        weights.extend((weight, weight))
    return np.asarray(endpoints), np.asarray(weights)


def _nacl():
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.set_cell([[5.64, 0, 0], [0.21, 5.76, 0], [0.13, 0.29, 5.51]], scale_atoms=True)
    atoms.positions[1] += [0.07, -0.03, 0.05]; atoms.wrap()
    return atoms


def _feature(atoms):
    result = n.compute_psbg_features(atoms)
    assert result.supported, result.failure_reason
    return result.features[n.FEATURE_NAMES[0]]


def test_schema() -> None:
    assert n.BALL_RADIUS == 4
    assert n.FEATURE_NAMES == ("psbg_skeletal_ball4_growth_q10",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}


def test_chain_ball_count_is_nine() -> None:
    endpoints, weights = _directed(((0, 0, (1, 0, 0), 1),))
    result = n.skeletal_ball_growth(n_sites=1, endpoints=endpoints, solid_angles=weights)
    assert result.supported and result.site_ball_counts == (9,)
    assert result.site_growth == pytest.approx((9 / 73,))


def test_simple_cubic_ball_count_is_129() -> None:
    endpoints, weights = _directed(((0, 0, (1, 0, 0), 1), (0, 0, (0, 1, 0), 1), (0, 0, (0, 0, 1), 1)))
    result = n.skeletal_ball_growth(n_sites=1, endpoints=endpoints, solid_angles=weights)
    assert result.supported and result.site_ball_counts == (129,)
    assert result.site_growth == pytest.approx((129 / 193,))


def test_order_scale_ties_and_replication() -> None:
    endpoints, weights = _directed(((0, 0, (1, 0, 0), 1), (0, 0, (0, 1, 0), 0.5), (0, 0, (0, 0, 1), 0.5)))
    order = [4, 1, 5, 0, 3, 2]
    reference = n.skeletal_ball_growth(n_sites=1, endpoints=endpoints, solid_angles=weights)
    changed = n.skeletal_ball_growth(n_sites=1, endpoints=endpoints[order], solid_angles=7 * weights[order])
    replicated = n.skeletal_ball_growth(n_sites=2, endpoints=np.vstack((endpoints, endpoints + [1, 1, 0, 0, 0])), solid_angles=np.concatenate((weights, weights)))
    assert reference.skeleton_edge_count == 3 and reference.skeleton_threshold == 0.5
    for result in (reference, changed, replicated):
        assert result.supported and result.features[n.FEATURE_NAMES[0]] == pytest.approx(reference.features[n.FEATURE_NAMES[0]])


@pytest.mark.parametrize(("n_sites", "endpoints", "weights"), ((2, [[0, 1, 0, 0, 0]], [1]), (2, [[0, 2, 0, 0, 0], [2, 0, 0, 0, 0]], [1, 1]), (2, [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], [1, 0.9]), (2, [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], [0, 0])))
def test_malformed_fails(n_sites, endpoints, weights) -> None:
    assert not n.skeletal_ball_growth(n_sites=n_sites, endpoints=np.asarray(endpoints), solid_angles=np.asarray(weights)).supported


def test_real_equivalences_and_firewall() -> None:
    atoms = _nacl(); reference = _feature(atoms)
    rotated = atoms.copy(); rotated.rotate(31, "z", rotate_cell=True)
    translated = atoms.copy(); translated.translate([.173, .291, .419]); translated.wrap()
    permuted = atoms[[3, 0, 6, 1, 7, 4, 2, 5]]
    rebased = atoms.copy(); rebased.set_cell(np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]]) @ atoms.cell.array, scale_atoms=False); rebased.wrap()
    for equivalent in (rotated, translated, permuted, rebased, atoms.repeat((2, 1, 1))):
        assert _feature(equivalent) == pytest.approx(reference, abs=1e-8)
    bad = []
    x = atoms.copy(); x.calc = Calculator(); bad.append(x)
    x = atoms.copy(); x.info["outcome"] = 1; bad.append(x)
    x = atoms.copy(); x.new_array("energy", np.zeros(len(x))); bad.append(x)
    x = atoms.copy(); x.pbc = False; bad.append(x)
    for x in bad:
        result = n.compute_psbg_features(x)
        assert not result.supported and "geometry-only Atoms" in str(result.failure_reason)


def test_row_and_boundary() -> None:
    row = n.compute_psbg_row(_nacl())
    assert row["psbg_supported"] is True
    assert len(row) == 11 and all(value is False for value in n.BOUNDARY_FLAGS.values())
