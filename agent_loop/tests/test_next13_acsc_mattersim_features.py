"""Contract tests for the additive ACSC-v0 combined MatterSim probes."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from scipy.linalg import logm

from src.next10_lrrc_mattersim_features import BatchPrediction


def _atoms(symbol: str = "H2") -> Atoms:
    return Atoms(
        symbol,
        scaled_positions=[[0.15, 0.25, 0.35], [0.62, 0.71, 0.83]],
        cell=[[8.0, 0.0, 0.0], [0.4, 9.0, 0.0], [0.2, 0.3, 10.0]],
        pbc=True,
    )


class _CoupledQuadraticPredictor:
    """Analytic E/F oracle in the exact frozen ACSC generalized coordinates."""

    def __init__(self) -> None:
        from src.next11_phsc import canonicalize_phsc_geometry, helmert_internal_basis

        self.calls: list[list[Atoms]] = []
        base, self.d_star = canonicalize_phsc_geometry(_atoms())
        self.reference_cell = base.cell.array.copy()
        self.reference_scaled = base.get_scaled_positions(wrap=False).copy()
        self.q = helmert_internal_basis(2)
        self.atomic = np.diag([1.0, 2.0, 3.0])
        self.strain = np.diag([1.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    def _coordinates(self, atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
        from src.next12_chsc import strain_basis

        relative = np.linalg.solve(self.reference_cell, atoms.cell.array)
        strain_matrix = np.real_if_close(logm(relative.T), tol=1000)
        eta = np.einsum("aij,ij->a", strain_basis(), strain_matrix)
        affine = self.reference_scaled @ atoms.cell.array
        displacement = atoms.get_positions() - affine
        z = self.q.T @ displacement.reshape(-1) / self.d_star
        return z, eta

    def _values(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        z, eta = self._coordinates(atoms)
        coupling = 2.0 if int(atoms.numbers[0]) == 1 else 0.5
        cross = np.zeros((3, 6))
        cross[0, 0] = coupling
        n_atoms = len(atoms)
        energy = n_atoms * (
            0.5 * z @ self.atomic @ z
            + z @ cross @ eta
            + 0.5 * eta @ self.strain @ eta
        )
        internal_gradient = self.atomic @ z + cross @ eta
        force = -(n_atoms / self.d_star) * (self.q @ internal_gradient)
        return float(energy), force.reshape(n_atoms, 3)

    def __call__(self, structures: list[Atoms]) -> BatchPrediction:
        self.calls.append([atoms.copy() for atoms in structures])
        values = [self._values(atoms) for atoms in structures]
        return BatchPrediction(
            total_energies_ev=[energy for energy, _force in values],
            forces_ev_per_a=[force for _energy, force in values],
            stresses_ev_per_a3=[np.zeros((3, 3)) for _ in structures],
        )


def test_combined_batch_reuses_complete_chsc_probes_and_finds_incremental_saddle() -> None:
    from src.next11_phsc import PHSCStatus
    from src.next12_chsc import CHSCStatus
    from src.next13_acsc import ACSCStatus
    from src.next13_acsc_mattersim_features import evaluate_acsc_batch

    supplied = {"sid-z": _atoms("He2"), "sid-a": _atoms("H2")}
    predictor = _CoupledQuadraticPredictor()
    observed = evaluate_acsc_batch(
        ["sid-z", "sid-a"],
        [supplied["sid-z"], supplied["sid-a"]],
        predictor,
        structures_per_call=2,
    )

    assert [item.sid for item in observed] == ["sid-a", "sid-z"]
    assert [len(call) for call in predictor.calls] == [218]
    assert all(item.phsc.status is PHSCStatus.RESOLVED_NONNEGATIVE for item in observed)
    assert all(item.chsc.status is CHSCStatus.RESOLVED_NONNEGATIVE for item in observed)
    assert observed[0].acsc.status is ACSCStatus.RESOLVED_NEGATIVE
    assert observed[0].acsc.coupling_only_negative is True
    assert observed[0].acsc.lambda_r == pytest.approx(-1.0, abs=2e-8)
    assert observed[1].acsc.status is ACSCStatus.RESOLVED_NONNEGATIVE
    assert observed[1].acsc.coupling_only_negative is False
    assert all(item.acsc.prediction_evaluation_count == 109 for item in observed)
    assert all(item.cross_h.shape == (6, 6) for item in observed)
    assert all(item.cross_h2.shape == (6, 6) for item in observed)


def test_unsupported_geometry_abstains_before_predictor_evaluation() -> None:
    from src.next13_acsc import ACSCStatus
    from src.next13_acsc_mattersim_features import evaluate_acsc_batch

    invalid = Atoms("H", positions=[[0, 0, 0]], cell=5 * np.eye(3), pbc=True)
    predictor = _CoupledQuadraticPredictor()
    observed = evaluate_acsc_batch(
        ["bad", "good"], [invalid, _atoms()], predictor, structures_per_call=2
    )

    assert [len(call) for call in predictor.calls] == [109]
    bad, good = observed
    assert bad.sid == "bad"
    assert bad.acsc.status is ACSCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY
    assert bad.acsc.negative is None
    assert bad.acsc.prediction_evaluation_count == 0
    assert good.sid == "good"
    assert good.acsc.status is ACSCStatus.RESOLVED_NEGATIVE
