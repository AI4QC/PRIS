from __future__ import annotations

import math

from ase import Atoms
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor

from src.next19_valence_transport import infer_valence_assignment
from src.next21_normalized_madelung import normalized_madelung_features
from src.next34_analytic_field_features import compute_analytic_field_features
from src.next35_coulomb_steric_balance_features import (
    compute_coulomb_steric_balance_features,
)
from src.next77_odac23_analytic_electrostatic_features import (
    ANALYTIC_ELECTROSTATIC_FEATURE_NAMES,
    NM_INVARIANT_FEATURE_NAMES,
    _shared_ewald_features,
    compute_odac23_analytic_electrostatic_features,
)


def test_analytic_electrostatics_are_finite_and_supercell_invariant() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    primitive = compute_odac23_analytic_electrostatic_features(atoms)
    repeated = compute_odac23_analytic_electrostatic_features(atoms.repeat((2, 1, 1)))

    assert primitive.supported
    assert repeated.supported
    assert tuple(primitive.features) == ANALYTIC_ELECTROSTATIC_FEATURE_NAMES
    for name in ANALYTIC_ELECTROSTATIC_FEATURE_NAMES:
        assert math.isfinite(primitive.features[name])
        assert math.isclose(
            primitive.features[name], repeated.features[name], rel_tol=2e-6, abs_tol=2e-8
        ), name


def test_analytic_electrostatics_fail_open_without_charge_partition() -> None:
    atoms = Atoms("Ar", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    result = compute_odac23_analytic_electrostatic_features(atoms)

    assert not result.supported
    assert result.features == {}


def test_shared_ewald_matches_original_family_implementations() -> None:
    structure = AseAtomsAdaptor.get_structure(bulk("NaCl", "rocksalt", a=5.64))
    assignment = infer_valence_assignment(structure)
    assert assignment.supported
    assert assignment.values is not None

    shared = _shared_ewald_features(structure, assignment.values)
    originals = (
        normalized_madelung_features(structure, assignment.values),
        compute_analytic_field_features(structure, assignment.values),
        compute_coulomb_steric_balance_features(structure, assignment.values),
    )
    assert all(result.supported for result in originals)
    expected = {
        **{name: originals[0].features[name] for name in NM_INVARIANT_FEATURE_NAMES},
        **originals[1].features,
        **originals[2].features,
    }
    assert tuple(shared) == tuple(expected)
    for name, value in shared.items():
        assert math.isclose(value, expected[name], rel_tol=1e-12, abs_tol=1e-14), name
