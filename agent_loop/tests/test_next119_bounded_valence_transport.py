from __future__ import annotations

import math

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.core.operations import SymmOp

from src.next119_bounded_valence_transport import (
    BUDGETS,
    CERTIFICATE_METHOD,
    FEATURE_NAMES,
    MAX_SITES,
    bounded_bond_valence_transport_features,
    bounded_transport_budget_certificate,
    compute_bounded_bond_valence_transport_features,
)
from src.next38_bond_valence_transport_compatibility_features import (
    bond_valence_transport_compatibility_features,
)


def _matrix(rows: int) -> np.ndarray:
    return np.zeros((rows, 12), dtype=float)


def test_scalar_radial_oracle_has_exact_half_residual_at_one_percent() -> None:
    jacobian = _matrix(1)
    jacobian[0, 0] = 1.0
    result = bounded_transport_budget_certificate(
        correction=[0.02],
        jacobian=jacobian,
        n_sites=2,
        characteristic_length=1.0,
    )
    assert result.supported
    assert result.failure_reason is None
    assert tuple(result.features) == FEATURE_NAMES
    assert math.isclose(result.features["bvtbd_unbounded_residual_fraction"], 0.0)
    assert math.isclose(result.features["bvtbd_required_linf_budget"], 0.02)
    assert math.isclose(result.features["bvtbd_residual_fraction_tau01"], 0.5)
    assert math.isclose(result.features["bvtbd_deformation_debt_tau01"], 0.5)
    assert result.features["bvtbd_residual_fraction_tau03"] < 1.0e-9
    assert result.features["bvtbd_deformation_debt_tau03"] < 1.0e-9


def test_multicoordinate_oracle_uses_frozen_minimum_norm_ray() -> None:
    jacobian = _matrix(2)
    jacobian[0, 0] = 1.0
    jacobian[1, 1] = 1.0
    result = bounded_transport_budget_certificate(
        correction=[0.02, 0.04],
        jacobian=jacobian,
        n_sites=2,
        characteristic_length=1.0,
    )
    assert result.supported
    assert math.isclose(result.features["bvtbd_required_linf_budget"], 0.04)
    # At tau=.01, the frozen minimum-norm ray is scaled by .01/.04=.25.
    # Its normalized residual is therefore exactly 1-.25=.75.  This differs
    # from a coordinate-wise box optimum and freezes the intended semantics.
    assert math.isclose(
        result.features["bvtbd_residual_fraction_tau01"],
        0.75,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        result.features["bvtbd_residual_fraction_tau03"],
        0.25,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    assert result.features["bvtbd_residual_fraction_tau10"] < 1.0e-12


def test_incompatible_floor_is_removed_from_deformation_debt() -> None:
    jacobian = _matrix(2)
    jacobian[0, 0] = 1.0
    result = bounded_transport_budget_certificate(
        correction=[0.02, 0.02],
        jacobian=jacobian,
        n_sites=2,
        characteristic_length=1.0,
    )
    assert result.supported
    floor = 1.0 / math.sqrt(2.0)
    assert math.isclose(
        result.features["bvtbd_unbounded_residual_fraction"], floor,
        rel_tol=0.0,
        abs_tol=1.0e-10,
    )
    # tau=0.01 leaves residual vector (-0.01, -0.02).  Normalizing its
    # sqrt(0.0005) norm by ||(0.02, 0.02)|| gives sqrt(5/8); removing the
    # orthogonal floor in quadrature leaves sqrt(1/8).
    expected_residual = math.sqrt(5.0 / 8.0)
    assert math.isclose(
        result.features["bvtbd_residual_fraction_tau01"], expected_residual,
        rel_tol=0.0,
        abs_tol=1.0e-8,
    )
    assert math.isclose(
        result.features["bvtbd_deformation_debt_tau01"], math.sqrt(1.0 / 8.0),
        rel_tol=0.0,
        abs_tol=1.0e-8,
    )


def test_zero_correction_is_exactly_zero() -> None:
    jacobian = _matrix(3)
    jacobian[:, :3] = np.eye(3)
    result = bounded_transport_budget_certificate(
        correction=np.zeros(3),
        jacobian=jacobian,
        n_sites=2,
        characteristic_length=2.0,
    )
    assert result.supported
    np.testing.assert_array_equal(list(result.features.values()), np.zeros(len(FEATURE_NAMES)))


def test_residual_is_monotone_with_budget_and_debt_is_bounded() -> None:
    rng = np.random.default_rng(119)
    jacobian = rng.normal(size=(8, 12))
    correction = rng.normal(size=8)
    result = bounded_transport_budget_certificate(
        correction=correction,
        jacobian=jacobian,
        n_sites=2,
        characteristic_length=1.7,
    )
    assert result.supported
    residuals = [
        result.features["bvtbd_residual_fraction_tau01"],
        result.features["bvtbd_residual_fraction_tau03"],
        result.features["bvtbd_residual_fraction_tau10"],
    ]
    assert residuals[0] >= residuals[1] >= residuals[2]
    assert all(0.0 <= value <= 1.0 + 1.0e-10 for value in residuals)
    assert all(
        0.0 <= result.features[name] <= 1.0 + 1.0e-10
        for name in (
            "bvtbd_deformation_debt_tau01",
            "bvtbd_deformation_debt_tau03",
            "bvtbd_deformation_debt_tau10",
        )
    )


def test_row_permutation_is_invariant() -> None:
    rng = np.random.default_rng(120)
    jacobian = rng.normal(size=(7, 12))
    correction = rng.normal(size=7)
    order = rng.permutation(len(correction))
    first = bounded_transport_budget_certificate(
        correction=correction,
        jacobian=jacobian,
        n_sites=2,
        characteristic_length=1.3,
    )
    second = bounded_transport_budget_certificate(
        correction=correction[order],
        jacobian=jacobian[order],
        n_sites=2,
        characteristic_length=1.3,
    )
    assert first.supported and second.supported
    np.testing.assert_allclose(
        list(first.features.values()),
        list(second.features.values()),
        rtol=1.0e-9,
        atol=1.0e-10,
    )


def test_joint_atomic_length_and_jacobian_rescaling_is_invariant() -> None:
    rng = np.random.default_rng(121)
    jacobian = rng.normal(size=(6, 12))
    correction = rng.normal(size=6)
    first = bounded_transport_budget_certificate(
        correction=correction,
        jacobian=jacobian,
        n_sites=2,
        characteristic_length=2.0,
    )
    rescaled = jacobian.copy()
    rescaled[:, :6] /= 4.0
    second = bounded_transport_budget_certificate(
        correction=correction,
        jacobian=rescaled,
        n_sites=2,
        characteristic_length=8.0,
    )
    assert first.supported and second.supported
    np.testing.assert_allclose(
        list(first.features.values()),
        list(second.features.values()),
        rtol=1.0e-9,
        atol=1.0e-10,
    )


def test_joint_target_and_jacobian_magnitude_rescaling_is_invariant() -> None:
    rng = np.random.default_rng(1219)
    jacobian = rng.normal(size=(9, 12))
    correction = rng.normal(size=9)
    first = bounded_transport_budget_certificate(
        correction=correction,
        jacobian=jacobian,
        n_sites=2,
        characteristic_length=1.4,
    )
    tiny = bounded_transport_budget_certificate(
        correction=1.0e-10 * correction,
        jacobian=1.0e-10 * jacobian,
        n_sites=2,
        characteristic_length=1.4,
    )
    assert first.supported and tiny.supported
    np.testing.assert_allclose(
        list(first.features.values()),
        list(tiny.features.values()),
        rtol=1.0e-8,
        atol=1.0e-10,
    )


def test_atomic_and_cell_motion_diagnostics_use_dimensionless_tensor_norms() -> None:
    jacobian = _matrix(2)
    jacobian[0, 0] = 1.0
    jacobian[1, -1] = 1.0
    result = bounded_transport_budget_certificate(
        correction=[0.02, 0.03],
        jacobian=jacobian,
        n_sites=2,
        characteristic_length=2.0,
    )
    assert result.supported
    assert math.isclose(result.features["bvtbd_atomic_motion_max"], 0.01)
    assert math.isclose(
        result.features["bvtbd_cell_strain_frobenius"],
        math.sqrt(2.0) * 0.03,
    )


def test_invalid_inputs_fail_open_without_partial_features() -> None:
    cases = [
        dict(correction=[1.0], jacobian=np.zeros((2, 12)), n_sites=2, characteristic_length=1.0),
        dict(correction=[np.nan], jacobian=np.zeros((1, 12)), n_sites=2, characteristic_length=1.0),
        dict(correction=[1.0], jacobian=np.zeros((1, 11)), n_sites=2, characteristic_length=1.0),
        dict(correction=[1.0], jacobian=np.zeros((1, 12)), n_sites=2, characteristic_length=0.0),
    ]
    for parameters in cases:
        result = bounded_transport_budget_certificate(**parameters)
        assert not result.supported
        assert result.failure_reason
        assert result.features == {}


def test_budget_constant_is_frozen() -> None:
    assert BUDGETS == (0.01, 0.03, 0.10)
    assert MAX_SITES == 64
    assert CERTIFICATE_METHOD == "closed_form_radial_minimum_norm_path"


def test_site_cap_fails_open_before_dense_certificate() -> None:
    result = bounded_transport_budget_certificate(
        correction=[1.0],
        jacobian=np.zeros((1, 3 * (MAX_SITES + 1) + 6)),
        n_sites=MAX_SITES + 1,
        characteristic_length=1.0,
    )
    assert not result.supported
    assert "site cap" in str(result.failure_reason)
    assert result.features == {}


def _nontrivial_bond_system() -> dict[str, object]:
    return {
        "charges": [1.0, 1.0, -1.0, -1.0],
        "endpoints": [[0, 2], [0, 3], [1, 2]],
        "vectors": [[1.0, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 1.2]],
        "strengths": [1.0, 1.0, 1.0],
        "decays": [0.37, 0.37, 0.37],
    }


def test_bond_system_adapter_reproduces_next38_unbounded_floor() -> None:
    system = _nontrivial_bond_system()
    bounded = bounded_bond_valence_transport_features(**system)
    prior = bond_valence_transport_compatibility_features(
        **system,
        parameter_sources=["exact", "exact", "exact"],
    )
    assert bounded.supported and prior.supported
    assert math.isclose(
        bounded.features["bvtbd_unbounded_residual_fraction"],
        prior.features["bvtc_incompatible_fraction"],
        rel_tol=0.0,
        abs_tol=1.0e-10,
    )


def test_bond_system_adapter_is_invariant_to_joint_length_scale() -> None:
    system = _nontrivial_bond_system()
    first = bounded_bond_valence_transport_features(**system)
    scaled = dict(system)
    scaled["vectors"] = (4.0 * np.asarray(system["vectors"], dtype=float)).tolist()
    scaled["decays"] = (4.0 * np.asarray(system["decays"], dtype=float)).tolist()
    second = bounded_bond_valence_transport_features(**scaled)
    assert first.supported and second.supported
    np.testing.assert_allclose(
        list(first.features.values()),
        list(second.features.values()),
        rtol=1.0e-9,
        atol=1.0e-10,
    )


def test_bond_system_adapter_fails_open_on_invalid_star_or_geometry() -> None:
    system = _nontrivial_bond_system()
    missing_star = dict(system)
    missing_star["endpoints"] = [[0, 2], [0, 3]]
    missing_star["vectors"] = [[1.0, 0.0, 0.0], [0.0, 1.1, 0.0]]
    missing_star["strengths"] = [1.0, 1.0]
    missing_star["decays"] = [0.37, 0.37]
    invalid_distance = dict(system)
    invalid_distance["vectors"] = [[0.0, 0.0, 0.0], *system["vectors"][1:]]
    for parameters in (missing_star, invalid_distance):
        result = bounded_bond_valence_transport_features(**parameters)
        assert not result.supported
        assert result.failure_reason
        assert result.features == {}


def test_structure_wrapper_is_representation_invariant() -> None:
    structure = Structure(
        Lattice.cubic(3.2),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.54, 0.5, 0.5]],
    )
    charges = [1.0, -1.0]
    translated = structure.copy()
    translated.translate_sites(
        range(len(translated)), [0.11, -0.17, 0.09], frac_coords=True
    )
    rotated = structure.copy()
    rotated.apply_operation(
        SymmOp.from_axis_angle_and_translation([0.3, 0.6, 0.2], 29.0),
        fractional=False,
    )
    permuted = Structure(
        structure.lattice,
        [site.specie for site in reversed(structure)],
        [site.frac_coords for site in reversed(structure)],
    )
    reference = compute_bounded_bond_valence_transport_features(structure, charges)
    variants = [
        compute_bounded_bond_valence_transport_features(translated, charges),
        compute_bounded_bond_valence_transport_features(rotated, charges),
        compute_bounded_bond_valence_transport_features(
            permuted, list(reversed(charges))
        ),
    ]
    assert reference.supported, reference.failure_reason
    for variant in variants:
        assert variant.supported, variant.failure_reason
        np.testing.assert_allclose(
            list(variant.features.values()),
            list(reference.features.values()),
            rtol=5.0e-5,
            atol=5.0e-8,
        )
