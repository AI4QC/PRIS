"""Contract tests for the label-free PHSC-v0 numerical core."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from ase import Atoms


def _base_atoms() -> Atoms:
    return Atoms(
        "H3",
        positions=[[1.2, 1.7, 2.1], [3.3, 1.1, 2.8], [2.4, 4.2, 1.4]],
        cell=[[8.0, 0.0, 0.0], [0.6, 9.0, 0.0], [0.2, 0.4, 10.0]],
        pbc=True,
    )


def _quadratic_oracle(
    reference: np.ndarray,
    hessian: np.ndarray,
    base_force: np.ndarray | None = None,
):
    flat_reference = np.asarray(reference, dtype=float).reshape(-1)
    flat_base_force = (
        np.zeros_like(flat_reference)
        if base_force is None
        else np.asarray(base_force, dtype=float).reshape(-1)
    )

    def oracle(atoms: Atoms) -> np.ndarray:
        displacement = atoms.get_positions().reshape(-1) - flat_reference
        return (flat_base_force - hessian @ displacement).reshape((-1, 3))

    return oracle


class _RecordingOracle:
    def __init__(self, function):
        self.function = function
        self.scaled_positions: list[np.ndarray] = []

    def __call__(self, atoms: Atoms) -> np.ndarray:
        self.scaled_positions.append(atoms.get_scaled_positions(wrap=False).copy())
        return self.function(atoms)


def test_frozen_constants_and_deterministic_helmert_basis() -> None:
    from src.next11_phsc import PHSC_VERSION, STEP_FRACTION, helmert_internal_basis

    q = helmert_internal_basis(4)

    assert PHSC_VERSION == "PHSC-v0"
    assert STEP_FRACTION == 2**-8
    assert q.shape == (12, 9)
    np.testing.assert_allclose(q.T @ q, np.eye(9), atol=2e-15)
    translation = np.kron(np.ones((4, 1)) / 2.0, np.eye(3))
    np.testing.assert_allclose(translation.T @ q, 0.0, atol=2e-15)

    # Freeze the Helmert sign and ordering, not merely its spanned subspace.
    expected_contrasts = np.array(
        [
            [1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(12)],
            [-1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(12)],
            [0.0, -2 / np.sqrt(6), 1 / np.sqrt(12)],
            [0.0, 0.0, -3 / np.sqrt(12)],
        ]
    )
    np.testing.assert_allclose(q, np.kron(expected_contrasts, np.eye(3)))


@pytest.mark.parametrize("n_atoms", [0, 1, -2, 2.5, True])
def test_helmert_basis_rejects_invalid_atom_counts(n_atoms: object) -> None:
    from src.next11_phsc import PHSCValidationError, helmert_internal_basis

    with pytest.raises(PHSCValidationError):
        helmert_internal_basis(n_atoms)  # type: ignore[arg-type]


def test_geometry_is_strictly_3d_periodic_nonsingular_mic_distinct_and_wrapped() -> None:
    from src.next11_phsc import canonicalize_phsc_geometry

    atoms = _base_atoms()
    atoms.positions[0] += 2 * atoms.cell[0] - atoms.cell[1]

    canonical, d_star = canonicalize_phsc_geometry(atoms)

    scaled = canonical.get_scaled_positions(wrap=False)
    assert np.all(scaled >= 0.0)
    assert np.all(scaled < 1.0)
    assert d_star > 0.0
    np.testing.assert_allclose(canonical.cell.array, atoms.cell.array)
    np.testing.assert_array_equal(canonical.pbc, [True, True, True])


@pytest.mark.parametrize(
    "atoms",
    [
        Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True),
        Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=np.eye(3), pbc=False),
        Atoms(
            "H2",
            positions=[[0, 0, 0], [1, 0, 0]],
            cell=[[1, 0, 0], [2, 0, 0], [0, 0, 1]],
            pbc=True,
        ),
        Atoms("H2", positions=[[0, 0, 0], [0, 0, 0]], cell=5 * np.eye(3), pbc=True),
        Atoms(
            "H2",
            positions=[[0, 0, 0], [5, 0, 0]],
            cell=5 * np.eye(3),
            pbc=True,
        ),
        Atoms(
            "H2",
            positions=[[0, 0, 0], [np.nan, 0, 0]],
            cell=5 * np.eye(3),
            pbc=True,
        ),
    ],
)
def test_geometry_validation_rejects_every_unsupported_case(atoms: Atoms) -> None:
    from src.next11_phsc import PHSCValidationError, canonicalize_phsc_geometry

    with pytest.raises(PHSCValidationError):
        canonicalize_phsc_geometry(atoms)


def test_four_force_samples_form_both_columns_with_frozen_denominators() -> None:
    from src.next11_phsc import hessian_columns_from_force_samples

    plus_h = np.array([[2.0, 5.0, -1.0], [0.0, 3.0, 7.0]])
    minus_h = np.array([[-2.0, 1.0, -3.0], [-4.0, 1.0, 1.0]])
    plus_h2 = np.array([[1.0, 4.0, -2.0], [-1.0, 2.5, 5.5]])
    minus_h2 = np.array([[-1.0, 2.0, -3.0], [-3.0, 1.5, 2.5]])

    column_h, column_h2 = hessian_columns_from_force_samples(
        plus_h, minus_h, plus_h2, minus_h2, h=0.25
    )

    np.testing.assert_allclose(column_h, -(plus_h - minus_h).reshape(-1) / 0.5)
    np.testing.assert_allclose(column_h2, -(plus_h2 - minus_h2).reshape(-1) / 0.25)


def test_public_probe_group_freezes_order_coordinate_and_wrapping() -> None:
    from src.next11_phsc import phsc_probe_group

    atoms = _base_atoms()
    base = atoms.get_positions().copy()
    h = 0.125

    probes = phsc_probe_group(atoms, coordinate=4, h=h)

    assert len(probes) == 4
    displacements = tuple(
        probe.get_positions().reshape(-1)[4] - base.reshape(-1)[4] for probe in probes
    )
    assert displacements == pytest.approx((h, -h, h / 2.0, -h / 2.0))
    for probe in probes:
        scaled = probe.get_scaled_positions(wrap=False)
        assert np.all(scaled >= 0.0)
        assert np.all(scaled < 1.0)


@pytest.mark.parametrize(
    ("coordinate", "h"),
    [(-1, 0.1), (9, 0.1), (1.5, 0.1), (True, 0.1), (0, 0.0), (0, np.nan)],
)
def test_public_probe_group_rejects_invalid_coordinate_or_step(
    coordinate: object, h: float
) -> None:
    from src.next11_phsc import PHSCValidationError, phsc_probe_group

    with pytest.raises(PHSCValidationError):
        phsc_probe_group(_base_atoms(), coordinate=coordinate, h=h)  # type: ignore[arg-type]


def test_unrepresentable_probe_abstains_before_any_oracle_call() -> None:
    from src.next11_phsc import (
        PHSCNumericalError,
        PHSCStatus,
        STEP_FRACTION,
        canonicalize_phsc_geometry,
        evaluate_phsc,
        phsc_probe_group,
    )

    atoms = Atoms(
        "H2",
        positions=[
            [5.0e19, 5.0e19, 5.0e19],
            [5.0e19 + 32768.0, 5.0e19, 5.0e19],
        ],
        cell=1.0e20 * np.eye(3),
        pbc=True,
    )
    canonical, d_star = canonicalize_phsc_geometry(atoms)
    # The separation remains representable through canonical wrapping, while
    # h=d*/256 is smaller than one Cartesian ULP at this coordinate magnitude.
    assert d_star == 32768.0
    assert STEP_FRACTION * d_star == 128.0

    with pytest.raises(PHSCNumericalError, match="representable|equivalent"):
        phsc_probe_group(canonical, coordinate=0, h=STEP_FRACTION * d_star)

    calls = 0

    def oracle(probe: Atoms) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.zeros((len(probe), 3))

    result = evaluate_phsc(atoms, oracle)

    assert calls == 0
    assert result.force_call_count == 0
    assert result.status is PHSCStatus.ABSTAIN_NUMERICAL_FAILURE
    assert result.negative is None
    assert result.error is not None and "representable" in result.error


@pytest.mark.parametrize(
    "samples,h",
    [
        ((np.zeros((2, 3)),) * 4, 0.0),
        ((np.zeros((2, 3)),) * 4, np.nan),
        ((np.zeros((2, 3)), np.zeros((3, 3)), np.zeros((2, 3)), np.zeros((2, 3))), 1.0),
        ((np.full((2, 3), np.nan),) * 4, 1.0),
    ],
)
def test_force_column_builder_rejects_invalid_direct_inputs(samples: tuple, h: float) -> None:
    from src.next11_phsc import PHSCValidationError, hessian_columns_from_force_samples

    with pytest.raises(PHSCValidationError):
        hessian_columns_from_force_samples(*samples, h=h)


@pytest.mark.parametrize(
    ("eigenvalues", "expected_status", "expected_negative"),
    [
        ([-3.0, 1.0, 2.0, 3.0, 4.0, 5.0], "resolved_negative", True),
        ([0.5, 1.0, 2.0, 3.0, 4.0, 5.0], "resolved_nonnegative", False),
        ([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "near_zero_or_inconsistent", False),
    ],
)
def test_exact_internal_quadratic_spectra_have_three_frozen_success_states(
    eigenvalues: list[float], expected_status: str, expected_negative: bool
) -> None:
    from src.next11_phsc import PHSCStatus, analyze_hessian_pair, helmert_internal_basis

    q = helmert_internal_basis(3)
    hessian = q @ np.diag(eigenvalues) @ q.T

    result = analyze_hessian_pair(hessian, hessian)

    assert result.status is PHSCStatus(expected_status)
    assert result.negative is expected_negative
    assert result.lambda_h == pytest.approx(min(eigenvalues), abs=2e-14)
    assert result.lambda_h2 == pytest.approx(min(eigenvalues), abs=2e-14)
    assert result.lambda_r == pytest.approx(min(eigenvalues), abs=2e-14)
    assert result.e_num == pytest.approx(0.0, abs=2e-14)
    assert result.u_num == pytest.approx(min(eigenvalues), abs=2e-14)
    assert result.l_num == pytest.approx(min(eigenvalues), abs=2e-14)
    assert result.tau_alg == pytest.approx(
        64 * 6 * np.finfo(np.float64).eps * max(1.0, max(abs(x) for x in eigenvalues))
    )


def test_two_scale_inconsistency_is_not_negative_evidence() -> None:
    from src.next11_phsc import PHSCStatus, analyze_hessian_pair, helmert_internal_basis

    q = helmert_internal_basis(2)
    h_h = q @ np.diag([-2.0, 1.0, 1.0]) @ q.T
    h_h2 = q @ np.diag([2.0, 1.0, 1.0]) @ q.T

    result = analyze_hessian_pair(h_h, h_h2)

    assert result.status is PHSCStatus.NEAR_ZERO_OR_INCONSISTENT
    assert result.negative is False
    assert result.lambda_h < 0.0 < result.lambda_h2
    assert result.e_num > 0.0


def test_consistent_scale_signs_are_insufficient_when_numerical_proxy_crosses_zero() -> None:
    from src.next11_phsc import PHSCStatus, analyze_hessian_pair, helmert_internal_basis

    q = helmert_internal_basis(2)
    both_negative_h = q @ np.diag([-10.0, 20.0, 20.0]) @ q.T
    both_negative_h2 = q @ np.diag([-1.0, 20.0, 20.0]) @ q.T
    negative = analyze_hessian_pair(both_negative_h, both_negative_h2)

    assert negative.lambda_h < 0.0
    assert negative.lambda_h2 < 0.0
    assert negative.u_num > 0.0
    assert negative.status is PHSCStatus.NEAR_ZERO_OR_INCONSISTENT
    assert negative.negative is False

    both_positive_h = q @ np.diag([10.0, 20.0, 20.0]) @ q.T
    both_positive_h2 = q @ np.diag([1.0, 20.0, 20.0]) @ q.T
    positive = analyze_hessian_pair(both_positive_h, both_positive_h2)

    assert positive.lambda_h > 0.0
    assert positive.lambda_h2 > 0.0
    assert positive.l_num < 0.0
    assert positive.status is PHSCStatus.NEAR_ZERO_OR_INCONSISTENT
    assert positive.negative is False


def test_algorithmic_tolerance_uses_strict_inequalities_at_exact_equality() -> None:
    from src.next11_phsc import PHSCStatus, classify_phsc_state

    tau = 0.25
    assert (
        classify_phsc_state(-tau, -2.0 * tau, -2.0 * tau, -3.0 * tau, tau)
        is PHSCStatus.NEAR_ZERO_OR_INCONSISTENT
    )
    assert (
        classify_phsc_state(-2.0 * tau, -tau, -2.0 * tau, -3.0 * tau, tau)
        is PHSCStatus.NEAR_ZERO_OR_INCONSISTENT
    )
    assert (
        classify_phsc_state(-2.0 * tau, -2.0 * tau, -tau, -3.0 * tau, tau)
        is PHSCStatus.NEAR_ZERO_OR_INCONSISTENT
    )
    assert (
        classify_phsc_state(tau, 2.0 * tau, 3.0 * tau, 2.0 * tau, tau)
        is PHSCStatus.NEAR_ZERO_OR_INCONSISTENT
    )
    assert (
        classify_phsc_state(2.0 * tau, tau, 3.0 * tau, 2.0 * tau, tau)
        is PHSCStatus.NEAR_ZERO_OR_INCONSISTENT
    )
    assert (
        classify_phsc_state(2.0 * tau, 2.0 * tau, 3.0 * tau, tau, tau)
        is PHSCStatus.NEAR_ZERO_OR_INCONSISTENT
    )
    assert (
        classify_phsc_state(-0.5, -0.5, -0.5, -0.75, tau)
        is PHSCStatus.RESOLVED_NEGATIVE
    )
    assert (
        classify_phsc_state(0.5, 0.5, 0.75, 0.5, tau)
        is PHSCStatus.RESOLVED_NONNEGATIVE
    )


def test_raw_antisymmetry_and_acoustic_translation_are_diagnostics_only() -> None:
    from src.next11_phsc import PHSCStatus, analyze_hessian_pair, helmert_internal_basis

    q = helmert_internal_basis(2)
    symmetric = q @ np.diag([2.0, 3.0, 4.0]) @ q.T
    antisymmetric = np.zeros((6, 6))
    antisymmetric[0, 1] = 7.0
    antisymmetric[1, 0] = -7.0
    translation_breaking = np.ones((6, 6)) * 0.25
    raw = symmetric + antisymmetric + translation_breaking

    result = analyze_hessian_pair(raw, raw)

    assert result.status is PHSCStatus.RESOLVED_NONNEGATIVE
    assert result.antisymmetric_norm_h == pytest.approx(7.0)
    assert result.antisymmetric_norm_h2 == pytest.approx(7.0)
    assert result.acoustic_residual_h > 0.0
    assert result.acoustic_residual_h2 > 0.0


def test_stationary_saddle_is_resolved_with_exact_12n_wrapped_force_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.next11_phsc as phsc
    from src.next11_phsc import PHSCStatus, evaluate_phsc, helmert_internal_basis

    atoms = _base_atoms()
    q = helmert_internal_basis(len(atoms))
    hessian = q @ np.diag([-4.0, 1.0, 2.0, 3.0, 5.0, 6.0]) @ q.T
    oracle = _RecordingOracle(_quadratic_oracle(atoms.positions, hessian))
    observed_coordinates: list[int] = []
    original_probe_group = phsc.phsc_probe_group

    def recording_probe_group(base: Atoms, coordinate: int, h: float):
        observed_coordinates.append(coordinate)
        return original_probe_group(base, coordinate, h)

    monkeypatch.setattr(phsc, "phsc_probe_group", recording_probe_group)

    result = evaluate_phsc(atoms, oracle)

    assert result.status is PHSCStatus.RESOLVED_NEGATIVE
    assert result.negative is True
    assert result.lambda_h == pytest.approx(-4.0, abs=2e-11)
    assert result.lambda_h2 == pytest.approx(-4.0, abs=4e-11)
    assert result.force_call_count == 12 * len(atoms)
    assert len(oracle.scaled_positions) == 12 * len(atoms)
    assert observed_coordinates == list(range(3 * len(atoms)))
    for scaled in oracle.scaled_positions:
        assert np.all(scaled >= 0.0)
        assert np.all(scaled < 1.0)


def test_negative_mode_orthogonal_to_positive_force_direction_is_found() -> None:
    from src.next11_phsc import PHSCStatus, evaluate_phsc, helmert_internal_basis

    atoms = _base_atoms()
    q = helmert_internal_basis(len(atoms))
    internal = np.diag([-2.5, 3.5, 4.0, 4.5, 5.0, 5.5])
    hessian = q @ internal @ q.T
    positive_force_direction = q[:, 1]
    assert positive_force_direction @ hessian @ positive_force_direction > 0.0

    result = evaluate_phsc(
        atoms,
        _quadratic_oracle(atoms.positions, hessian, positive_force_direction),
    )

    assert result.status is PHSCStatus.RESOLVED_NEGATIVE
    assert result.lambda_r == pytest.approx(-2.5, abs=3e-11)


def test_end_to_end_quadratic_oracle_is_correct_across_periodic_boundary() -> None:
    from ase.geometry import find_mic

    from src.next11_phsc import PHSCStatus, evaluate_phsc, helmert_internal_basis

    atoms = Atoms(
        "H2",
        positions=[[9.999, 2.0, 2.0], [5.0, 2.0, 2.0]],
        cell=10.0 * np.eye(3),
        pbc=True,
    )
    q = helmert_internal_basis(len(atoms))
    hessian = q @ np.diag([2.0, 3.0, 4.0]) @ q.T
    reference = atoms.get_positions().copy()
    crossed_boundary = False

    def oracle(probe: Atoms) -> np.ndarray:
        nonlocal crossed_boundary
        if probe.positions[0, 0] < 0.1:
            crossed_boundary = True
        vectors = probe.get_positions() - reference
        mic_vectors = find_mic(vectors, atoms.cell.array, pbc=True)[0]
        return (-hessian @ mic_vectors.reshape(-1)).reshape((-1, 3))

    result = evaluate_phsc(atoms, oracle)

    assert crossed_boundary is True
    assert result.status is PHSCStatus.RESOLVED_NONNEGATIVE
    assert result.lambda_h == pytest.approx(2.0, abs=3e-10)
    assert result.lambda_h2 == pytest.approx(2.0, abs=3e-10)
    assert result.lambda_r == pytest.approx(2.0, abs=3e-10)


def test_endpoint_result_matches_the_shared_matrix_analyzer() -> None:
    from src.next11_phsc import (
        PHSCStatus,
        analyze_hessian_pair,
        evaluate_phsc,
        helmert_internal_basis,
    )

    atoms = _base_atoms()
    q = helmert_internal_basis(len(atoms))
    hessian = q @ np.diag([0.75, 1.0, 2.0, 3.0, 4.0, 8.0]) @ q.T

    endpoint = evaluate_phsc(atoms, _quadratic_oracle(atoms.positions, hessian))
    shared = analyze_hessian_pair(hessian, hessian)

    assert endpoint.status is PHSCStatus.RESOLVED_NONNEGATIVE
    for field in (
        "lambda_h",
        "lambda_h2",
        "lambda_r",
        "e_num",
        "u_num",
        "l_num",
        "tau_alg",
    ):
        assert getattr(endpoint, field) == pytest.approx(getattr(shared, field), abs=5e-11)


def test_atom_permutation_is_covariant_for_spectrum_and_outcome() -> None:
    from src.next11_phsc import PHSCStatus, evaluate_phsc

    def pair_forces(atoms: Atoms) -> np.ndarray:
        pair_vectors = atoms.get_all_distances(mic=True, vector=True)
        return -1.5 * np.sum(pair_vectors, axis=1)

    atoms = _base_atoms()
    permuted = atoms[[2, 0, 1]]
    reference = evaluate_phsc(atoms, pair_forces)
    observed = evaluate_phsc(permuted, pair_forces)

    assert reference.status is PHSCStatus.RESOLVED_NEGATIVE
    assert observed.status is reference.status
    assert observed.lambda_h == pytest.approx(reference.lambda_h, abs=2e-10)
    assert observed.lambda_h2 == pytest.approx(reference.lambda_h2, abs=2e-10)
    assert observed.lambda_r == pytest.approx(reference.lambda_r, abs=2e-10)
    assert observed.e_num == pytest.approx(reference.e_num, abs=2e-10)


def test_oracle_failures_and_invalid_outputs_are_explicit_abstentions() -> None:
    from src.next11_phsc import PHSCStatus, evaluate_phsc

    atoms = _base_atoms()

    def exploding(_atoms: Atoms) -> np.ndarray:
        raise RuntimeError("oracle exploded")

    failure = evaluate_phsc(atoms, exploding)
    wrong_shape = evaluate_phsc(atoms, lambda _atoms: np.zeros((2, 3)))
    nonfinite = evaluate_phsc(atoms, lambda _atoms: np.full((3, 3), np.nan))

    assert failure.status is PHSCStatus.ABSTAIN_FORCE_FAILURE
    assert failure.force_call_count == 1
    assert failure.error is not None and "RuntimeError: oracle exploded" in failure.error
    assert "Traceback" not in failure.error
    assert wrong_shape.status is PHSCStatus.ABSTAIN_INVALID_FORCE
    assert wrong_shape.force_call_count == 1
    assert nonfinite.status is PHSCStatus.ABSTAIN_INVALID_FORCE
    assert nonfinite.force_call_count == 1
    assert failure.negative is None
    assert wrong_shape.negative is None


def test_late_force_failure_records_calls_and_never_becomes_instability_evidence() -> None:
    from src.next11_phsc import PHSCStatus, evaluate_phsc

    calls = 0

    def fails_on_call_seven(atoms: Atoms) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 7:
            raise ArithmeticError("seventh-call failure")
        return np.zeros((len(atoms), 3))

    result = evaluate_phsc(_base_atoms(), fails_on_call_seven)

    assert calls == 7
    assert result.force_call_count == 7
    assert result.status is PHSCStatus.ABSTAIN_FORCE_FAILURE
    assert result.negative is None


def test_unsupported_geometry_abstains_before_any_force_call() -> None:
    from src.next11_phsc import PHSCStatus, evaluate_phsc

    calls = 0

    def oracle(atoms: Atoms) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.zeros((len(atoms), 3))

    result = evaluate_phsc(
        Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=5 * np.eye(3), pbc=False),
        oracle,
    )

    assert calls == 0
    assert result.force_call_count == 0
    assert result.status is PHSCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY
    assert result.negative is None


@pytest.mark.parametrize(
    "matrix_pair",
    [
        (np.zeros((3, 3)), np.zeros((3, 3))),
        (np.zeros((6, 5)), np.zeros((6, 5))),
        (np.zeros((6, 6)), np.zeros((9, 9))),
        (np.full((6, 6), np.nan), np.zeros((6, 6))),
    ],
)
def test_matrix_analyzer_rejects_invalid_direct_inputs(matrix_pair: tuple) -> None:
    from src.next11_phsc import PHSCValidationError, analyze_hessian_pair

    with pytest.raises(PHSCValidationError):
        analyze_hessian_pair(*matrix_pair)


def test_public_results_are_immutable_and_status_strings_are_stable() -> None:
    from src.next11_phsc import PHSCResult, PHSCSpectralResult, PHSCStatus

    spectral = PHSCSpectralResult(
        status=PHSCStatus.NEAR_ZERO_OR_INCONSISTENT,
        negative=False,
        lambda_h=0.0,
        lambda_h2=0.0,
        lambda_r=0.0,
        e_num=0.0,
        u_num=0.0,
        l_num=0.0,
        tau_alg=1e-12,
        antisymmetric_norm_h=0.0,
        antisymmetric_norm_h2=0.0,
        acoustic_residual_h=0.0,
        acoustic_residual_h2=0.0,
    )
    result = PHSCResult(status=PHSCStatus.ABSTAIN_NUMERICAL_FAILURE, error="failed")

    with pytest.raises(FrozenInstanceError):
        spectral.lambda_h = 1.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.error = None  # type: ignore[misc]
    assert [status.value for status in PHSCStatus] == [
        "resolved_negative",
        "resolved_nonnegative",
        "near_zero_or_inconsistent",
        "abstain_unsupported_geometry",
        "abstain_force_failure",
        "abstain_invalid_force",
        "abstain_numerical_failure",
    ]
