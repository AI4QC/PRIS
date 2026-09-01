from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next407_beck_ecn_simplicity as n


def _kernel(endpoints: object):
    return n.beck_ecn_simplicity(
        symbols=("Sr", "Ni", "F", "F", "F", "F"),
        formal_valences=(2, 2, -1, -1, -1, -1),
        endpoints=np.asarray(endpoints),
    )


def _distorted_nacl() -> Atoms:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.set_cell(
        np.asarray([[5.64, 0.0, 0.0], [0.27, 5.77, 0.0], [0.18, 0.31, 5.53]]),
        scale_atoms=True,
    )
    atoms.positions[1] += np.asarray([0.08, -0.04, 0.06])
    atoms.wrap()
    return atoms


def _feature(atoms: Atoms) -> float:
    result = n.compute_becns_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_frozen_protocol_schema_direction_and_boundary_are_exact() -> None:
    assert n.PROTOCOL == "2026-08-13-next407-beck-ecn-simplicity-v1"
    assert len(n.DESIGN_SHA256) == 64
    assert n.FEATURE_NAMES == ("becns_beck_ecn_simplicity",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())


def test_adjacent_integer_distribution_has_zero_excess() -> None:
    result = _kernel(
        (
            (0, 2), (0, 3), (0, 4), (0, 5),
            (1, 2), (1, 3), (1, 4), (1, 5),
        )
    )
    assert result.supported, result.failure_reason
    assert result.beck_excess == pytest.approx(0.0, abs=1e-14)
    assert result.features[n.FEATURE_NAMES[0]] == 1.0


def test_fixed_total_segregation_strictly_reduces_simplicity() -> None:
    balanced = _kernel(
        (
            (0, 2), (0, 3), (0, 4), (0, 5),
            (1, 2), (1, 3), (1, 4), (1, 5),
        )
    )
    segregated = _kernel(
        (
            (0, 2), (0, 3), (0, 4), (0, 5),
            (1, 2), (1, 2), (1, 3), (1, 3),
        )
    )
    assert balanced.supported and segregated.supported
    assert segregated.beck_excess == pytest.approx(4.0, abs=1e-14)
    assert segregated.features[n.FEATURE_NAMES[0]] == pytest.approx(2.0 / 3.0)
    assert segregated.features[n.FEATURE_NAMES[0]] < balanced.features[n.FEATURE_NAMES[0]]


def test_kernel_is_order_and_disjoint_replication_invariant() -> None:
    endpoints = np.asarray(
        (
            (0, 2), (0, 3), (0, 4), (0, 5),
            (1, 2), (1, 2), (1, 3), (1, 3),
        )
    )
    reference = _kernel(endpoints)
    reordered = _kernel(endpoints[[7, 1, 5, 3, 0, 6, 4, 2]])
    replicated = n.beck_ecn_simplicity(
        symbols=("Sr", "Ni", "F", "F", "F", "F") * 2,
        formal_valences=(2, 2, -1, -1, -1, -1) * 2,
        endpoints=np.vstack((endpoints, endpoints + 6)),
    )
    for result in (reference, reordered, replicated):
        assert result.supported, result.failure_reason
        assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(
            reference.features[n.FEATURE_NAMES[0]], abs=1e-14
        )


@pytest.mark.parametrize(
    ("symbols", "charges", "endpoints"),
    (
        (("Na",), (1,), ((0, 0),)),
        (("Na", "Cl"), (1, -2), ((0, 1),)),
        (("Na", "Cl"), (1, -1), ((1, 0),)),
        (("Na", "Cl", "Cl"), (2, -1, -1), ((0, 1),)),
        (("Na", "Cl"), (1, -1), ((0, 2),)),
        (("Na", "Cl"), (0.5, -0.5), ((0, 1),)),
    ),
)
def test_malformed_or_isolated_inputs_fail_closed(
    symbols: object, charges: object, endpoints: object
) -> None:
    result = n.beck_ecn_simplicity(
        symbols=symbols, formal_valences=charges, endpoints=endpoints
    )
    assert result.supported is False
    assert result.features == {}


def test_distorted_crystal_and_equivalents_are_supported_and_invariant() -> None:
    atoms = _distorted_nacl()
    result = n.compute_becns_features(atoms)
    assert result.supported, result.failure_reason
    assert result.site_count == len(atoms)
    assert result.edge_count >= len(atoms)
    assert 0 < result.features[n.FEATURE_NAMES[0]] <= 1
    reference = _feature(atoms)
    rotated = atoms.copy(); rotated.rotate(31.0, "z", rotate_cell=True)
    translated = atoms.copy(); translated.translate([0.173, 0.291, 0.419]); translated.wrap()
    permuted = atoms[[3, 0, 6, 1, 7, 4, 2, 5]]
    rebased = atoms.copy(); rebased.set_cell(np.asarray([[1,1,0],[0,1,0],[0,0,1]]) @ atoms.cell.array, scale_atoms=False); rebased.wrap()
    for equivalent in (rotated, translated, permuted, rebased, atoms.repeat((2,1,1))):
        assert _feature(equivalent) == pytest.approx(reference, abs=1e-8)


def test_geometry_boundary_and_row_schema_are_exact() -> None:
    atoms = _distorted_nacl()
    changed = []
    item = atoms.copy(); item.calc = Calculator(); changed.append(item)
    item = atoms.copy(); item.info["outcome"] = 1; changed.append(item)
    item = atoms.copy(); item.new_array("energy", np.zeros(len(item))); changed.append(item)
    item = atoms.copy(); item.pbc = False; changed.append(item)
    item = atoms.copy(); item.positions[0, 0] = np.nan; changed.append(item)
    for item in changed:
        result = n.compute_becns_features(item)
        assert not result.supported
        assert "geometry-only Atoms" in str(result.failure_reason)
    row = n.compute_becns_row(atoms)
    assert tuple(name for name in row if name.startswith("becns_")) == (
        "becns_beck_ecn_simplicity", "becns_supported", "becns_failure",
        "becns_site_count", "becns_edge_count", "becns_cation_ecn_class_count",
        "becns_anion_type_count", "becns_beck_excess", "becns_valence_policy",
    )
    assert row["becns_supported"] is True
