from __future__ import annotations

import inspect
import math

from ase import Atoms
import numpy as np
import pytest

import src.next279_radical_packing_autocorrelation as n


def _nacl() -> Atoms:
    return Atoms(
        symbols=["Na", "Cl"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=np.eye(3) * 5.6,
        pbc=True,
    )


def _asymmetric() -> Atoms:
    return Atoms(
        symbols=["Si", "O", "Na"],
        scaled_positions=[(0.10, 0.20, 0.30), (0.52, 0.63, 0.47), (0.81, 0.24, 0.72)],
        cell=[(4.1, 0.0, 0.0), (0.7, 4.4, 0.0), (0.4, 0.8, 4.8)],
        pbc=True,
    )


def test_two_site_autocorrelation_has_exact_analytic_values() -> None:
    result = n.packing_autocorrelation_features(
        values=[1.0, 4.0], contacts=[(0, 1), (1, 0)]
    )
    assert result["moran"] == pytest.approx(-1.0)
    assert result["geary"] == pytest.approx(2.0)
    assert result["absolute_moran"] == 0.0
    assert result["extreme_edge_fraction"] == pytest.approx(1.0)


def test_constant_residual_population_returns_exact_zeros() -> None:
    result = n.packing_autocorrelation_features(
        values=[2.0, 2.0, 2.0], contacts=[(0, 1), (1, 0), (1, 2), (2, 1)]
    )
    assert result == {
        "moran": 0.0,
        "geary": 0.0,
        "absolute_moran": 0.0,
        "extreme_edge_fraction": 0.0,
    }


@pytest.mark.parametrize(
    ("values", "contacts"),
    [
        ([1.0], []),
        ([1.0, np.nan], [(0, 1)]),
        ([1.0, -2.0], [(0, 1)]),
        ([1.0, 2.0], [(0, 2)]),
        ([1.0, 2.0], [(0, 1, 2)]),
    ],
)
def test_autocorrelation_refuses_invalid_population(values, contacts) -> None:
    with pytest.raises(ValueError, match="population differs"):
        n.packing_autocorrelation_features(values=values, contacts=contacts)


def test_radical_contacts_are_nonempty_reciprocal_and_replication_invariant() -> None:
    atoms = _nacl()
    radii = np.asarray([n.n267._tabulated_radius(symbol) for symbol in atoms.symbols])
    cells, contacts = n.periodic_radical_cells_and_contacts(atoms, radii=radii)
    assert len(cells) == 2
    assert contacts
    assert n.contacts_are_reciprocal(contacts)
    assert all(np.isfinite(contact.displacement).all() for contact in contacts)


def test_one_site_structure_abstains_from_graph_features() -> None:
    atoms = Atoms("Cu", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 3.6, pbc=True)
    result = n.compute_radical_packing_autocorrelation_features(atoms)
    assert not result.supported
    assert "at least two sites" in str(result.failure_reason)


def _feature_vector(atoms: Atoms) -> np.ndarray:
    result = n.compute_radical_packing_autocorrelation_features(atoms)
    assert result.supported, result.failure_reason
    assert tuple(result.features) == n.FEATURE_NAMES
    assert result.contact_count > 0 and result.contact_count % 2 == 0
    return np.asarray([result.features[name] for name in n.FEATURE_NAMES])


def test_real_features_are_representation_invariant() -> None:
    atoms = _asymmetric()
    reference = _feature_vector(atoms)

    angle = 0.731
    axis = np.asarray([1.0, 2.0, -1.0], dtype=float)
    axis /= np.linalg.norm(axis)
    cross = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    rotation = (
        math.cos(angle) * np.eye(3)
        + (1.0 - math.cos(angle)) * np.outer(axis, axis)
        + math.sin(angle) * cross
    )
    rotated = atoms.copy()
    rotated.positions = rotated.positions @ rotation.T
    rotated.set_cell(rotated.cell.array @ rotation.T, scale_atoms=False)

    translated = atoms.copy()
    translated.positions += [1.37, -0.62, 0.91]
    translated.wrap()

    permuted = atoms[[2, 0, 1]]

    rebased = atoms.copy()
    operation = np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
    rebased.set_cell(operation @ atoms.cell.array, scale_atoms=False)
    rebased.wrap()

    replicated = atoms.repeat((2, 1, 1))
    for equivalent in (rotated, translated, permuted, rebased, replicated):
        np.testing.assert_allclose(
            _feature_vector(equivalent), reference, rtol=0.0, atol=2.0e-9
        )


def test_malformed_geometry_fails_closed() -> None:
    atoms = _nacl()
    atoms.pbc = False
    result = n.compute_radical_packing_autocorrelation_features(atoms)
    assert not result.supported and "periodic" in str(result.failure_reason)


def test_builder_interface_has_no_endpoint_validation_or_replication_input() -> None:
    parameters = tuple(
        inspect.signature(n.build_radical_packing_autocorrelation_features).parameters
    )
    assert parameters == (
        "scigen_cohort_dir",
        "wyformer_cohort_dir",
        "design_path",
        "amendment_path",
        "second_amendment_path",
        "output_dir",
        "workers",
        "require_formal_inputs",
    )


def test_builder_fails_closed_on_missing_inputs(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT279 input is missing"):
        n.build_radical_packing_autocorrelation_features(
            scigen_cohort_dir=tmp_path / "scigen",
            wyformer_cohort_dir=tmp_path / "wyformer",
            design_path=tmp_path / "design",
            amendment_path=tmp_path / "amendment",
            second_amendment_path=tmp_path / "amendment2",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
