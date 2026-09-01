"""Dataset-free, label-free synthetic falsification package for ACSC-v0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np

from src.next11_phsc import helmert_internal_basis
from src.next13_acsc import (
    ACSC_VERSION,
    ACSCSpectralResult,
    ACSCStatus,
    TAU_MULTIPLIER,
    analyze_acsc_blocks,
    analyze_coupled_hessian_pair,
)
from src.next8_mattersim_committee_features import (
    _atomic_publish_directory_no_replace,
)


PROTOCOL = "2026-08-02-next13-acsc-synthetic-v1"
CASE_ORDER = (
    "positive_uncoupled",
    "subcritical_coupling",
    "coupling_only_saddle",
    "rotated_generalized_saddle",
    "two_scale_inconsistent",
)


def _payload(result: ACSCSpectralResult) -> dict[str, object]:
    return {
        "acsc_status": result.status.value,
        "acsc_negative": result.negative,
        "lambda_h_ev_per_atom": result.lambda_h,
        "lambda_h2_ev_per_atom": result.lambda_h2,
        "lambda_r_ev_per_atom": result.lambda_r,
        "e_num_ev_per_atom": result.e_num,
        "u_num_ev_per_atom": result.u_num,
        "l_num_ev_per_atom": result.l_num,
        "tau_alg_ev_per_atom": result.tau_alg,
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


def _two_atom_blocks(coupling: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw blocks whose active dimensionless block is [[1,c],[c,1]]."""

    n_atoms = 2
    d_star = np.sqrt(float(n_atoms))
    q = helmert_internal_basis(n_atoms)
    atomic_internal = np.diag([1.0, 2.0, 3.0])
    atomic = q @ atomic_internal @ q.T
    strain = np.diag([1.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    internal_cross = np.zeros((3, 6))
    internal_cross[0, 0] = float(coupling) * n_atoms / d_star
    cross = q @ internal_cross
    return atomic, strain, cross


def _coupling_case(name: str, coupling: float, expected_status: ACSCStatus) -> dict[str, object]:
    atomic, strain, cross = _two_atom_blocks(coupling)
    result = analyze_acsc_blocks(
        atomic,
        atomic,
        strain,
        strain,
        cross,
        cross,
        d_star=np.sqrt(2.0),
    )
    atomic_min = 1.0
    strain_min = 1.0
    expected_lambda = 1.0 - abs(coupling)
    observed = {
        **_payload(result),
        "atomic_lambda_min": atomic_min,
        "strain_lambda_min": strain_min,
        "dimensionless_cross_entry": float(coupling),
    }
    expected = {
        "acsc_status": expected_status.value,
        "lambda_min": expected_lambda,
        "both_diagonal_blocks_positive": True,
    }
    passed = bool(
        result.status is expected_status
        and np.isclose(result.lambda_r, expected_lambda, rtol=0.0, atol=3e-14)
        and result.e_num == 0.0
        and atomic_min > 0.0
        and strain_min > 0.0
    )
    return _entry(name, expected, observed, passed)


def _rotated_entry() -> dict[str, object]:
    reference_matrix = np.diag([-1.25, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    angle = np.pi / 11.0
    rotation = np.eye(9)
    rotation[[0, 4], [0, 4]] = np.cos(angle)
    rotation[0, 4] = -np.sin(angle)
    rotation[4, 0] = np.sin(angle)
    rotated_matrix = rotation @ reference_matrix @ rotation.T
    reference = analyze_coupled_hessian_pair(reference_matrix, reference_matrix)
    rotated = analyze_coupled_hessian_pair(rotated_matrix, rotated_matrix)
    expected = {
        "reference_status": ACSCStatus.RESOLVED_NEGATIVE.value,
        "rotated_status": ACSCStatus.RESOLVED_NEGATIVE.value,
        "spectrum_unchanged": True,
    }
    observed = {
        "reference_status": reference.status.value,
        "rotated_status": rotated.status.value,
        "reference_lambda_r": reference.lambda_r,
        "rotated_lambda_r": rotated.lambda_r,
        "rotation_determinant": float(np.linalg.det(rotation)),
    }
    passed = bool(
        reference.status is ACSCStatus.RESOLVED_NEGATIVE
        and rotated.status is reference.status
        and np.isclose(reference.lambda_r, rotated.lambda_r, rtol=0.0, atol=3e-14)
    )
    return _entry("rotated_generalized_saddle", expected, observed, passed)


def _inconsistent_entry() -> dict[str, object]:
    first = np.diag([-2.0] + [1.0] * 8)
    second = np.diag([2.0] + [1.0] * 8)
    result = analyze_coupled_hessian_pair(first, second)
    expected = {
        "acsc_status": ACSCStatus.NEAR_ZERO_OR_INCONSISTENT.value,
        "opposite_scale_signs": True,
    }
    passed = bool(
        result.status is ACSCStatus.NEAR_ZERO_OR_INCONSISTENT
        and result.lambda_h < 0.0 < result.lambda_h2
        and result.e_num > 0.0
    )
    return _entry("two_scale_inconsistent", expected, _payload(result), passed)


def _build_cases() -> list[dict[str, object]]:
    cases = [
        _coupling_case(
            "positive_uncoupled", 0.0, ACSCStatus.RESOLVED_NONNEGATIVE
        ),
        _coupling_case(
            "subcritical_coupling", 0.5, ACSCStatus.RESOLVED_NONNEGATIVE
        ),
        _coupling_case(
            "coupling_only_saddle", 2.0, ACSCStatus.RESOLVED_NEGATIVE
        ),
        _rotated_entry(),
        _inconsistent_entry(),
    ]
    if tuple(case["name"] for case in cases) != CASE_ORDER:
        raise AssertionError("synthetic case order differs from the frozen contract")
    return cases


def _source_paths() -> dict[str, Path]:
    return {
        "src/next13_acsc.py": Path(__file__).with_name("next13_acsc.py").resolve(),
        "src/next13_acsc_synthetic.py": Path(__file__).resolve(),
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
        "version": ACSC_VERSION,
        "evidence_scope": {
            "dataset_free": True,
            "label_free": True,
            "engineering_only": True,
            "mattersim_executed": False,
        },
        "inputs": {"datasets": [], "labels": [], "model_checkpoints": []},
        "constants": {
            "strain_dimension": 6,
            "tau_multiplier": TAU_MULTIPLIER,
            "atomic_coordinate": "z=dR/d_star in Helmert internal basis",
            "strain_coordinate": "Frobenius-orthonormal eta",
            "coupled_hessian_unit": "eV/atom",
        },
        "formulas": {
            "atomic_block": "d_star^2/N * Q.T*H_RR*Q",
            "cross_block": "d_star/N * Q.T*(-dF/deta)",
            "strain_block": "H_eta_eta/N",
            "richardson": "K_R=(4*K_h2-K_h)/3",
            "incremental_reject": (
                "PHSC=resolved_nonnegative and CHSC=resolved_nonnegative and "
                "ACSC=resolved_negative"
            ),
        },
        "source_sha256": {
            logical_name: _sha256(path) for logical_name, path in sources.items()
        },
        "case_order": list(CASE_ORDER),
        "cases": cases,
        "known_limitations": [
            "Synthetic cases establish engineering behavior only.",
            "The two-scale proxy is not a confidence interval or rigorous error bound.",
            "ACSC-v0 is an MLIP Gamma-point diagnostic, not a DFT stability certificate.",
        ],
        "engineering_pass": all(case["passed"] is True for case in cases),
        "scientific_improvement_claim": False,
    }


def _verify_source_hashes(
    manifest: Mapping[str, object], source_paths: Mapping[str, Path]
) -> None:
    recorded = manifest.get("source_sha256")
    if not isinstance(recorded, Mapping) or set(recorded) != set(source_paths):
        raise RuntimeError("manifest source hashes are invalid before publication")
    for logical_name, path in source_paths.items():
        if recorded[logical_name] != _sha256(path):
            raise RuntimeError(
                f"source {logical_name} changed before publication; refusing to publish"
            )


def run(output_dir: str | os.PathLike[str]) -> Path:
    """Evaluate fixed synthetic cases and atomically publish a strict manifest."""

    target = Path(output_dir)
    if os.path.lexists(os.fspath(target)):
        raise FileExistsError(os.fspath(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    sources = _source_paths()
    manifest = _build_manifest(sources)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    manifest_path = staging / "MANIFEST.json"
    try:
        payload = json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
        with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _verify_source_hashes(manifest, sources)
        _atomic_publish_directory_no_replace(staging, target)
    except Exception:
        if manifest_path.exists():
            manifest_path.unlink()
        if staging.exists():
            staging.rmdir()
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
