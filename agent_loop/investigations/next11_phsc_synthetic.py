"""Dataset-free, label-free synthetic falsification package for PHSC-v0."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable

import numpy as np
from ase import Atoms

from src.next11_phsc import (
    PHSC_VERSION,
    STEP_FRACTION,
    TAU_MULTIPLIER,
    PHSCResult,
    PHSCSpectralResult,
    PHSCStatus,
    analyze_hessian_pair,
    evaluate_phsc,
    helmert_internal_basis,
)
from src.next9_lrrc import LRRCResult, LRRCStatus, evaluate_lrrc


PROTOCOL = "2026-08-02-next11-phsc-synthetic-v1"
CASE_ORDER = (
    "positive_quadratic",
    "negative_quadratic",
    "stationary_saddle_lrrc_blind",
    "force_orthogonal_saddle",
    "translation_projection",
    "proper_rotation_covariance",
    "semidefinite_zero_ambiguous",
    "two_scale_inconsistent",
    "antisymmetric_diagnostic",
    "mass_invariance",
)

_NUMERICAL_ATOL = 5.0e-10
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


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
) -> Callable[[Atoms], np.ndarray]:
    flat_reference = np.asarray(reference, dtype=np.float64).reshape(-1).copy()
    matrix = np.asarray(hessian, dtype=np.float64).copy()
    force_at_reference = (
        np.zeros_like(flat_reference)
        if base_force is None
        else np.asarray(base_force, dtype=np.float64).reshape(-1).copy()
    )

    def oracle(atoms: Atoms) -> np.ndarray:
        displacement = atoms.get_positions().reshape(-1) - flat_reference
        return (force_at_reference - matrix @ displacement).reshape((-1, 3))

    return oracle


def _close(observed: float | None, expected: float, atol: float = _NUMERICAL_ATOL) -> bool:
    return bool(
        observed is not None
        and np.isfinite(observed)
        and np.isclose(observed, expected, rtol=0.0, atol=atol)
    )


def _phsc_payload(result: PHSCResult | PHSCSpectralResult) -> dict[str, object]:
    return {
        "phsc_status": result.status.value,
        "phsc_negative": result.negative,
        "lambda_h": result.lambda_h,
        "lambda_h2": result.lambda_h2,
        "lambda_r": result.lambda_r,
        "e_num": result.e_num,
        "u_num": result.u_num,
        "l_num": result.l_num,
        "tau_alg": result.tau_alg,
        "antisymmetric_norm_h": result.antisymmetric_norm_h,
        "antisymmetric_norm_h2": result.antisymmetric_norm_h2,
        "acoustic_residual_h": result.acoustic_residual_h,
        "acoustic_residual_h2": result.acoustic_residual_h2,
    }


def _lrrc_payload(result: LRRCResult) -> dict[str, object]:
    return {
        "lrrc_status": result.status.value,
        "lrrc_negative": result.negative,
        "lrrc_kappa_h": result.kappa_h,
        "lrrc_kappa_h2": result.kappa_h2,
        "lrrc_kappa_r": result.kappa_r,
        "lrrc_u_num": result.u_num,
    }


def _entry(
    name: str,
    expected: Mapping[str, object],
    observed: Mapping[str, object],
    passed: bool,
) -> dict[str, object]:
    return {
        "name": name,
        "expected": dict(expected),
        "observed": dict(observed),
        "passed": bool(passed),
    }


def _analytic_quadratic_entry(
    name: str,
    atoms: Atoms,
    hessian: np.ndarray,
    expected_status: PHSCStatus,
    expected_lambda: float,
) -> dict[str, object]:
    result = evaluate_phsc(atoms, _quadratic_oracle(atoms.positions, hessian))
    expected = {
        "phsc_status": expected_status.value,
        "phsc_negative": expected_status is PHSCStatus.RESOLVED_NEGATIVE,
        "lambda_min": expected_lambda,
    }
    observed = _phsc_payload(result)
    passed = bool(
        result.status is expected_status
        and result.negative is expected["phsc_negative"]
        and _close(result.lambda_h, expected_lambda)
        and _close(result.lambda_h2, expected_lambda)
        and _close(result.lambda_r, expected_lambda)
        and _close(result.e_num, 0.0)
    )
    return _entry(name, expected, observed, passed)


def _stationary_saddle_entry(
    atoms: Atoms, hessian: np.ndarray
) -> dict[str, object]:
    oracle = _quadratic_oracle(atoms.positions, hessian)
    lrrc = evaluate_lrrc(atoms, oracle)
    phsc = evaluate_phsc(atoms, oracle)
    expected = {
        "lrrc_status": LRRCStatus.STATIONARY_FALLBACK.value,
        "lrrc_negative": None,
        "phsc_status": PHSCStatus.RESOLVED_NEGATIVE.value,
        "phsc_negative": True,
        "lambda_min": -3.0,
    }
    observed = {**_lrrc_payload(lrrc), **_phsc_payload(phsc)}
    passed = bool(
        lrrc.status is LRRCStatus.STATIONARY_FALLBACK
        and lrrc.negative is None
        and phsc.status is PHSCStatus.RESOLVED_NEGATIVE
        and phsc.negative is True
        and _close(phsc.lambda_r, -3.0)
    )
    return _entry("stationary_saddle_lrrc_blind", expected, observed, passed)


def _force_orthogonal_saddle_entry(atoms: Atoms, q: np.ndarray) -> dict[str, object]:
    eigenvalues = np.array([-2.5, 3.5, 4.0, 4.5, 5.0, 5.5])
    hessian = q @ np.diag(eigenvalues) @ q.T
    base_force = q[:, 1]
    force_negative_mode_dot = float(base_force @ q[:, 0])
    oracle = _quadratic_oracle(atoms.positions, hessian, base_force)
    lrrc = evaluate_lrrc(atoms, oracle)
    phsc = evaluate_phsc(atoms, oracle)
    expected = {
        "force_mode_curvature": 3.5,
        "force_negative_mode_dot": 0.0,
        "lrrc_status": LRRCStatus.OK.value,
        "lrrc_negative": False,
        "phsc_status": PHSCStatus.RESOLVED_NEGATIVE.value,
        "phsc_negative": True,
        "lambda_min": -2.5,
    }
    observed = {
        "force_negative_mode_dot": force_negative_mode_dot,
        **_lrrc_payload(lrrc),
        **_phsc_payload(phsc),
    }
    passed = bool(
        lrrc.status is LRRCStatus.OK
        and _close(force_negative_mode_dot, 0.0, atol=2.0e-15)
        and lrrc.negative is False
        and _close(lrrc.kappa_h, 3.5)
        and _close(lrrc.kappa_h2, 3.5)
        and _close(lrrc.kappa_r, 3.5)
        and phsc.status is PHSCStatus.RESOLVED_NEGATIVE
        and phsc.negative is True
        and _close(phsc.lambda_r, -2.5)
    )
    return _entry("force_orthogonal_saddle", expected, observed, passed)


def _translation_projection_entry(q: np.ndarray) -> dict[str, object]:
    n_atoms = q.shape[0] // 3
    positive = q @ np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) @ q.T
    translation = np.kron(
        np.ones((n_atoms, 1), dtype=np.float64) / np.sqrt(float(n_atoms)),
        np.eye(3, dtype=np.float64),
    )
    contaminated = positive + translation @ np.diag([10.0, 20.0, 30.0]) @ translation.T
    reference = analyze_hessian_pair(positive, positive)
    observed_result = analyze_hessian_pair(contaminated, contaminated)
    expected = {
        "reference_status": PHSCStatus.RESOLVED_NONNEGATIVE.value,
        "contaminated_status": PHSCStatus.RESOLVED_NONNEGATIVE.value,
        "projected_lambda_unchanged": True,
        "acoustic_contamination_detected": True,
    }
    observed = {
        "reference_status": reference.status.value,
        "contaminated_status": observed_result.status.value,
        "reference_lambda_r": reference.lambda_r,
        "contaminated_lambda_r": observed_result.lambda_r,
        "reference_acoustic_residual": reference.acoustic_residual_h,
        "contaminated_acoustic_residual": observed_result.acoustic_residual_h,
    }
    passed = bool(
        reference.status is PHSCStatus.RESOLVED_NONNEGATIVE
        and observed_result.status is reference.status
        and _close(observed_result.lambda_r, reference.lambda_r, atol=5.0e-12)
        and observed_result.acoustic_residual_h
        > reference.acoustic_residual_h + 1.0
    )
    return _entry("translation_projection", expected, observed, passed)


def _semidefinite_entry(q: np.ndarray) -> dict[str, object]:
    hessian = q @ np.diag([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]) @ q.T
    result = analyze_hessian_pair(hessian, hessian)
    expected = {
        "phsc_status": PHSCStatus.NEAR_ZERO_OR_INCONSISTENT.value,
        "phsc_negative": False,
        "lambda_min": 0.0,
    }
    observed = _phsc_payload(result)
    passed = bool(
        result.status is PHSCStatus.NEAR_ZERO_OR_INCONSISTENT
        and result.negative is False
        and abs(result.lambda_r) <= result.tau_alg
    )
    return _entry("semidefinite_zero_ambiguous", expected, observed, passed)


def _proper_rotation_entry(q: np.ndarray) -> dict[str, object]:
    n_atoms = q.shape[0] // 3
    hessian = q @ np.diag([-3.0, 1.0, 2.0, 3.0, 4.0, 5.0]) @ q.T
    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    cartesian_rotation = np.kron(np.eye(n_atoms, dtype=np.float64), rotation)
    rotated_hessian = cartesian_rotation @ hessian @ cartesian_rotation.T
    reference = analyze_hessian_pair(hessian, hessian)
    rotated = analyze_hessian_pair(rotated_hessian, rotated_hessian)
    determinant = float(np.linalg.det(rotation))
    expected = {
        "proper_rotation_determinant": 1.0,
        "reference_status": PHSCStatus.RESOLVED_NEGATIVE.value,
        "rotated_status": PHSCStatus.RESOLVED_NEGATIVE.value,
        "primary_decision_unchanged": True,
        "spectrum_unchanged": True,
    }
    observed = {
        "rotation_determinant": determinant,
        "reference_status": reference.status.value,
        "rotated_status": rotated.status.value,
        "reference_negative": reference.negative,
        "rotated_negative": rotated.negative,
        "reference_lambda_r": reference.lambda_r,
        "rotated_lambda_r": rotated.lambda_r,
    }
    passed = bool(
        _close(determinant, 1.0, atol=1.0e-15)
        and reference.status is PHSCStatus.RESOLVED_NEGATIVE
        and rotated.status is reference.status
        and reference.negative is True
        and rotated.negative is True
        and _close(rotated.lambda_r, reference.lambda_r, atol=5.0e-12)
    )
    return _entry("proper_rotation_covariance", expected, observed, passed)


def _two_scale_inconsistent_entry(q: np.ndarray) -> dict[str, object]:
    h_h = q @ np.diag([-2.0, 1.0, 1.0, 1.0, 1.0, 1.0]) @ q.T
    h_h2 = q @ np.diag([2.0, 1.0, 1.0, 1.0, 1.0, 1.0]) @ q.T
    result = analyze_hessian_pair(h_h, h_h2)
    expected = {
        "phsc_status": PHSCStatus.NEAR_ZERO_OR_INCONSISTENT.value,
        "phsc_negative": False,
        "opposite_scale_signs": True,
    }
    observed = _phsc_payload(result)
    passed = bool(
        result.status is PHSCStatus.NEAR_ZERO_OR_INCONSISTENT
        and result.negative is False
        and result.lambda_h < 0.0 < result.lambda_h2
        and result.e_num > 0.0
    )
    return _entry("two_scale_inconsistent", expected, observed, passed)


def _antisymmetric_entry(q: np.ndarray) -> dict[str, object]:
    symmetric = q @ np.diag([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) @ q.T
    antisymmetric = np.zeros_like(symmetric)
    antisymmetric[0, 1] = 7.0
    antisymmetric[1, 0] = -7.0
    raw = symmetric + antisymmetric
    result = analyze_hessian_pair(raw, raw)
    expected = {
        "phsc_status": PHSCStatus.RESOLVED_NONNEGATIVE.value,
        "phsc_negative": False,
        "antisymmetric_norm": 7.0,
        "diagnostic_only": True,
    }
    observed = _phsc_payload(result)
    passed = bool(
        result.status is PHSCStatus.RESOLVED_NONNEGATIVE
        and result.negative is False
        and _close(result.antisymmetric_norm_h, 7.0, atol=5.0e-12)
        and _close(result.antisymmetric_norm_h2, 7.0, atol=5.0e-12)
    )
    return _entry("antisymmetric_diagnostic", expected, observed, passed)


def _mass_invariance_entry(atoms: Atoms, hessian: np.ndarray) -> dict[str, object]:
    changed = atoms.copy()
    changed.set_masses([1.0, 12.0, 197.0])
    reference = evaluate_phsc(atoms, _quadratic_oracle(atoms.positions, hessian))
    observed_result = evaluate_phsc(
        changed, _quadratic_oracle(changed.positions, hessian)
    )
    expected = {
        "reference_status": PHSCStatus.RESOLVED_NEGATIVE.value,
        "changed_status": PHSCStatus.RESOLVED_NEGATIVE.value,
        "primary_decision_unchanged": True,
        "spectrum_unchanged": True,
    }
    observed = {
        "reference_masses": [float(value) for value in atoms.get_masses()],
        "changed_masses": [float(value) for value in changed.get_masses()],
        "reference_status": reference.status.value,
        "changed_status": observed_result.status.value,
        "reference_negative": reference.negative,
        "changed_negative": observed_result.negative,
        "reference_lambda_r": reference.lambda_r,
        "changed_lambda_r": observed_result.lambda_r,
    }
    passed = bool(
        reference.status is PHSCStatus.RESOLVED_NEGATIVE
        and observed_result.status is reference.status
        and reference.negative is True
        and observed_result.negative is True
        and reference.lambda_r is not None
        and _close(observed_result.lambda_r, reference.lambda_r, atol=5.0e-12)
    )
    return _entry("mass_invariance", expected, observed, passed)


def _build_cases() -> list[dict[str, object]]:
    atoms = _base_atoms()
    q = helmert_internal_basis(len(atoms))
    positive = q @ np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) @ q.T
    negative = q @ np.diag([-3.0, 1.0, 2.0, 3.0, 4.0, 5.0]) @ q.T
    cases = [
        _analytic_quadratic_entry(
            "positive_quadratic",
            atoms,
            positive,
            PHSCStatus.RESOLVED_NONNEGATIVE,
            1.0,
        ),
        _analytic_quadratic_entry(
            "negative_quadratic",
            atoms,
            negative,
            PHSCStatus.RESOLVED_NEGATIVE,
            -3.0,
        ),
        _stationary_saddle_entry(atoms, negative),
        _force_orthogonal_saddle_entry(atoms, q),
        _translation_projection_entry(q),
        _proper_rotation_entry(q),
        _semidefinite_entry(q),
        _two_scale_inconsistent_entry(q),
        _antisymmetric_entry(q),
        _mass_invariance_entry(atoms, negative),
    ]
    if tuple(case["name"] for case in cases) != CASE_ORDER:
        raise AssertionError("synthetic case order differs from the frozen contract")
    return cases


def _source_paths() -> dict[str, Path]:
    return {
        "src/next11_phsc.py": Path(__file__).with_name("next11_phsc.py").resolve(),
        "src/next11_phsc_synthetic.py": Path(__file__).resolve(),
        "src/next9_lrrc.py": Path(__file__).with_name("next9_lrrc.py").resolve(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_manifest(
    source_paths: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    sources = dict(_source_paths() if source_paths is None else source_paths)
    cases = _build_cases()
    return {
        "protocol": PROTOCOL,
        "version": PHSC_VERSION,
        "evidence_scope": {
            "dataset_free": True,
            "label_free": True,
            "engineering_only": True,
            "mattersim_executed": False,
        },
        "inputs": {
            "datasets": [],
            "labels": [],
            "model_checkpoints": [],
        },
        "constants": {
            "step_fraction": STEP_FRACTION,
            "tau_multiplier": TAU_MULTIPLIER,
        },
        "formulas": {
            "hessian_columns": (
                "H_h[:,j]=-(F(x+h*e_j)-F(x-h*e_j))/(2h); "
                "H_h2[:,j]=-(F(x+h/2*e_j)-F(x-h/2*e_j))/h"
            ),
            "projection": "A_delta=Q.T*((H_delta+H_delta.T)/2)*Q",
            "richardson": "A_R=(4*A_h2-A_h)/3",
            "e_num": (
                "e_num=norm((A_h2-A_h)/3,2); numerical proxy, not a confidence bound"
            ),
            "resolved_negative": (
                "lambda_h<-tau_alg and lambda_h2<-tau_alg and "
                "lambda_R+e_num<-tau_alg"
            ),
        },
        "source_sha256": {
            logical_name: _sha256(path) for logical_name, path in sources.items()
        },
        "case_order": list(CASE_ORDER),
        "cases": cases,
        "known_limitations": [
            (
                "PHSC-v0 is a fixed-cell Gamma-point MLIP curvature diagnostic; "
                "it is not a full phonon or DFT-stability certificate."
            ),
            (
                "The two-scale numerical proxy is not a confidence bound or a "
                "rigorous truncation-error bound."
            ),
            (
                "These dataset-free and label-free cases establish engineering "
                "behavior only and make no scientific-improvement claim."
            ),
        ],
        "engineering_pass": all(case["passed"] is True for case in cases),
        "scientific_improvement_claim": False,
    }


def _verify_prepublish_source_hashes(
    manifest: Mapping[str, object], source_paths: Mapping[str, Path]
) -> None:
    recorded = manifest.get("source_sha256")
    if not isinstance(recorded, Mapping):
        raise RuntimeError("manifest source hashes are missing before publication")
    if set(recorded) != set(source_paths):
        raise RuntimeError("manifest source mapping changed before publication")
    for logical_name, expected_hash in recorded.items():
        if not isinstance(logical_name, str) or not isinstance(expected_hash, str):
            raise RuntimeError("manifest source hashes are invalid before publication")
        observed_hash = _sha256(source_paths[logical_name])
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"source {logical_name} changed before publication; refusing to publish"
            )


def _path_exists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _exists_error(path: Path) -> FileExistsError:
    return FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), os.fspath(path))


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a directory or fail closed without no-replace support."""

    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError as exc:
        raise NotImplementedError(
            "atomic no-replace directory publication is unavailable"
        ) from exc
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise NotImplementedError(
            "atomic no-replace directory publication is unavailable"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(target),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise _exists_error(target)
    if error_number in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
        raise NotImplementedError(
            "atomic no-replace directory publication is unavailable"
        )
    raise OSError(error_number, os.strerror(error_number), os.fspath(target))


def _remove_unpublished_temp(temp_dir: Path) -> None:
    manifest_path = temp_dir / "MANIFEST.json"
    if manifest_path.exists():
        manifest_path.unlink()
    if temp_dir.exists():
        temp_dir.rmdir()


def run(output_dir: str | os.PathLike[str]) -> Path:
    """Evaluate fixed synthetic cases and atomically publish one strict manifest."""

    target = Path(output_dir)
    if _path_exists(target):
        raise _exists_error(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    source_paths = _source_paths()
    manifest = _build_manifest(source_paths)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=os.fspath(target.parent))
    )
    try:
        manifest_path = temp_dir / "MANIFEST.json"
        payload = json.dumps(
            manifest,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _verify_prepublish_source_hashes(manifest, source_paths)
        _rename_noreplace(temp_dir, target)
    except Exception:
        _remove_unpublished_temp(temp_dir)
        raise
    return target


def main(argv: Sequence[str] | None = None) -> int:
    """CLI with an output directory as its sole application input."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    run(arguments.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
