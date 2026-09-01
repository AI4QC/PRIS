"""Dataset-free, label-free synthetic falsification package for CHSC-v0."""

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
from scipy.linalg import logm

from src.next12_chsc import (
    CHSC_VERSION,
    STEP_STRAIN,
    STRAIN_DIMENSION,
    TAU_MULTIPLIER,
    CHSCResult,
    CHSCSpectralResult,
    CHSCStatus,
    analyze_strain_hessian_pair,
    evaluate_chsc,
    strain_basis,
)


PROTOCOL = "2026-08-02-next12-chsc-synthetic-v1"
CASE_ORDER = (
    "positive_quadratic",
    "negative_quadratic",
    "rotated_saddle",
    "semidefinite_zero_ambiguous",
    "quartic_two_scale_inconsistent",
)

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_ATOL = 2.0e-8


def _base_atoms() -> Atoms:
    return Atoms(
        "Si2O4",
        scaled_positions=[
            [0.11, 0.23, 0.37],
            [0.57, 0.61, 0.79],
            [0.31, 0.73, 0.17],
            [0.83, 0.19, 0.43],
            [0.47, 0.41, 0.89],
            [0.71, 0.87, 0.59],
        ],
        cell=[[6.2, 0.0, 0.0], [0.7, 7.1, 0.0], [0.3, 0.5, 8.4]],
        pbc=True,
    )


def _strain_coordinates(reference_cell: np.ndarray, probe: Atoms) -> np.ndarray:
    relative = np.linalg.solve(reference_cell, np.asarray(probe.cell.array))
    strain = np.real_if_close(logm(relative.T), tol=1000)
    if np.iscomplexobj(strain) or not np.all(np.isfinite(strain)):
        raise RuntimeError("analytic strain logarithm was not finite and real")
    return np.einsum("aij,ij->a", strain_basis(), strain)


def _energy_oracle(
    atoms: Atoms,
    function: Callable[[np.ndarray], float],
) -> Callable[[Atoms], float]:
    reference_cell = np.asarray(atoms.cell.array, dtype=np.float64).copy()
    n_atoms = len(atoms)

    def oracle(probe: Atoms) -> float:
        return float(n_atoms * function(_strain_coordinates(reference_cell, probe)))

    return oracle


def _payload(result: CHSCResult | CHSCSpectralResult) -> dict[str, object]:
    return {
        "chsc_status": result.status.value,
        "chsc_negative": result.negative,
        "lambda_h": result.lambda_h,
        "lambda_h2": result.lambda_h2,
        "lambda_r": result.lambda_r,
        "e_num": result.e_num,
        "u_num": result.u_num,
        "l_num": result.l_num,
        "tau_alg": result.tau_alg,
        "antisymmetric_norm_h": result.antisymmetric_norm_h,
        "antisymmetric_norm_h2": result.antisymmetric_norm_h2,
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


def _close(value: float | None, expected: float, atol: float = _ATOL) -> bool:
    return bool(
        value is not None
        and np.isfinite(value)
        and np.isclose(value, expected, rtol=0.0, atol=atol)
    )


def _quadratic_entry(
    name: str,
    hessian: np.ndarray,
    expected_status: CHSCStatus,
    expected_lambda: float,
) -> dict[str, object]:
    atoms = _base_atoms()

    def energy(strain: np.ndarray) -> float:
        return float(0.5 * strain @ hessian @ strain)

    result = evaluate_chsc(atoms, _energy_oracle(atoms, energy))
    expected = {
        "chsc_status": expected_status.value,
        "chsc_negative": expected_status is CHSCStatus.RESOLVED_NEGATIVE,
        "lambda_min": expected_lambda,
    }
    passed = bool(
        result.status is expected_status
        and result.negative is expected["chsc_negative"]
        and _close(result.lambda_h, expected_lambda)
        and _close(result.lambda_h2, expected_lambda)
        and _close(result.lambda_r, expected_lambda)
        and _close(result.e_num, 0.0)
        and result.energy_call_count == 85
    )
    return _entry(name, expected, _payload(result), passed)


def _rotated_saddle_entry() -> dict[str, object]:
    angle = np.pi / 7.0
    rotation = np.eye(STRAIN_DIMENSION)
    rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    reference_hessian = np.diag([-2.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    rotated_hessian = rotation @ reference_hessian @ rotation.T
    reference = analyze_strain_hessian_pair(reference_hessian, reference_hessian)
    rotated = analyze_strain_hessian_pair(rotated_hessian, rotated_hessian)
    expected = {
        "reference_status": CHSCStatus.RESOLVED_NEGATIVE.value,
        "rotated_status": CHSCStatus.RESOLVED_NEGATIVE.value,
        "spectrum_unchanged": True,
    }
    observed = {
        "rotation_determinant": float(np.linalg.det(rotation)),
        "reference_status": reference.status.value,
        "rotated_status": rotated.status.value,
        "reference_lambda_r": reference.lambda_r,
        "rotated_lambda_r": rotated.lambda_r,
    }
    passed = bool(
        reference.status is CHSCStatus.RESOLVED_NEGATIVE
        and rotated.status is reference.status
        and _close(reference.lambda_r, rotated.lambda_r, atol=2.0e-12)
    )
    return _entry("rotated_saddle", expected, observed, passed)


def _zero_entry() -> dict[str, object]:
    hessian = np.diag([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    result = analyze_strain_hessian_pair(hessian, hessian)
    expected = {
        "chsc_status": CHSCStatus.NEAR_ZERO_OR_INCONSISTENT.value,
        "chsc_negative": False,
        "lambda_min": 0.0,
    }
    passed = bool(
        result.status is CHSCStatus.NEAR_ZERO_OR_INCONSISTENT
        and result.negative is False
        and abs(result.lambda_r) <= result.tau_alg
    )
    return _entry("semidefinite_zero_ambiguous", expected, _payload(result), passed)


def _quartic_inconsistent_entry() -> dict[str, object]:
    atoms = _base_atoms()
    beta = -1.0 / STEP_STRAIN**2

    def energy(strain: np.ndarray) -> float:
        return float(0.5 * strain @ strain + beta * strain[0] ** 4)

    result = evaluate_chsc(atoms, _energy_oracle(atoms, energy))
    expected = {
        "chsc_status": CHSCStatus.NEAR_ZERO_OR_INCONSISTENT.value,
        "chsc_negative": False,
        "opposite_scale_signs": True,
    }
    passed = bool(
        result.status is CHSCStatus.NEAR_ZERO_OR_INCONSISTENT
        and result.negative is False
        and result.lambda_h is not None
        and result.lambda_h2 is not None
        and result.lambda_h < 0.0 < result.lambda_h2
        and result.e_num is not None
        and result.e_num > 0.0
    )
    return _entry("quartic_two_scale_inconsistent", expected, _payload(result), passed)


def _build_cases() -> list[dict[str, object]]:
    cases = [
        _quadratic_entry(
            "positive_quadratic",
            np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            CHSCStatus.RESOLVED_NONNEGATIVE,
            1.0,
        ),
        _quadratic_entry(
            "negative_quadratic",
            np.diag([-3.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            CHSCStatus.RESOLVED_NEGATIVE,
            -3.0,
        ),
        _rotated_saddle_entry(),
        _zero_entry(),
        _quartic_inconsistent_entry(),
    ]
    if tuple(case["name"] for case in cases) != CASE_ORDER:
        raise AssertionError("synthetic case order differs from the frozen contract")
    return cases


def _source_paths() -> dict[str, Path]:
    return {
        "src/next12_chsc.py": Path(__file__).with_name("next12_chsc.py").resolve(),
        "src/next12_chsc_synthetic.py": Path(__file__).resolve(),
        "src/next11_phsc.py": Path(__file__).with_name("next11_phsc.py").resolve(),
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
        "version": CHSC_VERSION,
        "evidence_scope": {
            "dataset_free": True,
            "label_free": True,
            "engineering_only": True,
            "mattersim_executed": False,
        },
        "inputs": {"datasets": [], "labels": [], "model_checkpoints": []},
        "constants": {
            "strain_dimension": STRAIN_DIMENSION,
            "direction_count": 21,
            "step_strain": STEP_STRAIN,
            "tau_multiplier": TAU_MULTIPLIER,
            "energy_calls_per_structure": 85,
        },
        "formulas": {
            "deformation": "A(t)=A0*exp(t*sum_i(v_i*B_i)).T at fixed fractional coordinates",
            "directional_curvature": "q_h(v)=(E(+hv)-2*E(0)+E(-hv))/(N*h^2)",
            "richardson": "H_R=(4*H_h2-H_h)/3",
            "e_num": "norm((H_h2-H_h)/3,2); numerical proxy, not a confidence bound",
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
                "CHSC-v0 is a fixed-fractional-coordinate Gamma-point MLIP cell-curvature "
                "diagnostic, not a DFT elastic or thermodynamic stability certificate."
            ),
            "The two-scale numerical proxy is not a confidence or rigorous error bound.",
            "Dataset-free cases establish engineering behavior only.",
        ],
        "engineering_pass": all(case["passed"] is True for case in cases),
        "scientific_improvement_claim": False,
    }


def _verify_prepublish_source_hashes(
    manifest: Mapping[str, object], source_paths: Mapping[str, Path]
) -> None:
    recorded = manifest.get("source_sha256")
    if not isinstance(recorded, Mapping) or set(recorded) != set(source_paths):
        raise RuntimeError("manifest source hashes are invalid before publication")
    for logical_name, expected_hash in recorded.items():
        observed_hash = _sha256(source_paths[str(logical_name)])
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"source {logical_name} changed before publication; refusing to publish"
            )


def _path_exists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _exists_error(path: Path) -> FileExistsError:
    return FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), os.fspath(path))


def _rename_noreplace(source: Path, target: Path) -> None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError as exc:
        raise NotImplementedError("atomic no-replace publication is unavailable") from exc
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise NotImplementedError("atomic no-replace publication is unavailable")
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
    number = ctypes.get_errno()
    if number in (errno.EEXIST, errno.ENOTEMPTY):
        raise _exists_error(target)
    if number in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
        raise NotImplementedError("atomic no-replace publication is unavailable")
    raise OSError(number, os.strerror(number), os.fspath(target))


def _remove_temp(path: Path) -> None:
    manifest = path / "MANIFEST.json"
    if manifest.exists():
        manifest.unlink()
    if path.exists():
        path.rmdir()


def run(output_dir: str | os.PathLike[str]) -> Path:
    """Evaluate fixed synthetic cases and atomically publish a strict manifest."""

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
        payload = json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
        manifest_path = temp_dir / "MANIFEST.json"
        with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _verify_prepublish_source_hashes(manifest, source_paths)
        _rename_noreplace(temp_dir, target)
    except Exception:
        _remove_temp(temp_dir)
        raise
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    run(arguments.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
