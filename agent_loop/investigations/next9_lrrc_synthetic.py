"""Dataset-free analytic runner for the LRRC-v0 engineering contract."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Callable, Sequence

import numpy as np
from ase import Atoms

from src.next9_lrrc import (
    FORCE_RMS_FLOOR,
    LRRC_VERSION,
    STEP_FRACTION,
    Decision,
    LRRCResult,
    LRRCStatus,
    QuotaCRCRow,
    compose_decision,
    evaluate_lrrc,
    quota_crc,
)


CASE_ORDER = (
    "positive_harmonic",
    "inverted_harmonic",
    "translation_invariance",
    "rotation_invariance",
    "permutation_invariance",
    "pbc_wrapping_invariance",
    "exact_zero_force_saddle",
    "oracle_exception",
    "wrong_shape_force",
    "nonfinite_force",
    "decision_keep_or_negative",
    "decision_reject_or_nonnegative",
    "quota_fixed_ceil_sqrt_n",
    "quota_boundary_ties",
    "quota_abstain_unchanged",
    "quota_rejection_subset",
)

_EVALUATION_ORDER = CASE_ORDER[:7]
_SCALAR_DIAGNOSTIC_KEYS = (
    "d_star",
    "h",
    "kappa_h",
    "kappa_h2",
    "kappa_r",
    "error_proxy",
    "u_num",
)
_NUMERICAL_ATOL = 1.0e-10
_PAIR_D_STAR = math.sqrt(2.16)
_PAIR_H = STEP_FRACTION * _PAIR_D_STAR
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


@dataclass(frozen=True)
class _AnalyticCase:
    name: str
    atoms: Atoms
    oracle: Callable[[Atoms], np.ndarray]


def _base_atoms() -> Atoms:
    return Atoms(
        "H3",
        positions=[[0.4, 0.8, 1.1], [1.8, 0.6, 0.7], [3.2, 1.7, 0.9]],
        cell=[7.0, 8.0, 9.0],
        pbc=True,
    )


def _pair_quadratic_forces(atoms: Atoms, curvature: float = 1.0) -> np.ndarray:
    """Return pair-harmonic forces with LRRC curvature ``N * curvature``."""

    pair_vectors = atoms.get_all_distances(mic=True, vector=True)
    return curvature * np.sum(pair_vectors, axis=1)


def _analytic_cases() -> tuple[_AnalyticCase, ...]:
    base = _base_atoms()
    translated = base.copy()
    translated.translate([1.3, -0.4, 0.7])

    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rotated = base.copy()
    rotated.set_positions(base.get_positions() @ rotation.T)
    rotated.set_cell(np.asarray(base.cell) @ rotation.T)

    permuted = base[[2, 0, 1]]
    wrapped = base.copy()
    wrapped.positions[1] += wrapped.cell[0] - wrapped.cell[1]

    stationary = base.copy()
    stationary_reference = stationary.get_positions().copy()

    def exact_saddle_forces(probe: Atoms) -> np.ndarray:
        displacement = probe.get_positions() - stationary_reference
        forces = np.zeros_like(displacement)
        forces[:, 0] = displacement[:, 0]
        forces[:, 1] = -displacement[:, 1]
        return forces

    cases = (
        _AnalyticCase("positive_harmonic", base, _pair_quadratic_forces),
        _AnalyticCase(
            "inverted_harmonic",
            base,
            lambda atoms: _pair_quadratic_forces(atoms, curvature=-2.0),
        ),
        _AnalyticCase("translation_invariance", translated, _pair_quadratic_forces),
        _AnalyticCase("rotation_invariance", rotated, _pair_quadratic_forces),
        _AnalyticCase("permutation_invariance", permuted, _pair_quadratic_forces),
        _AnalyticCase("pbc_wrapping_invariance", wrapped, _pair_quadratic_forces),
        _AnalyticCase("exact_zero_force_saddle", stationary, exact_saddle_forces),
    )
    if tuple(case.name for case in cases) != _EVALUATION_ORDER:
        raise AssertionError("analytic evaluation order differs from the frozen contract")
    return cases


def _curvature_payload(result: LRRCResult) -> dict[str, object]:
    return {
        "status": result.status.value,
        "negative": result.negative,
        "d_star": result.d_star,
        "h": result.h,
        "kappa_h": result.kappa_h,
        "kappa_h2": result.kappa_h2,
        "kappa_r": result.kappa_r,
        "error_proxy": result.error_proxy,
        "u_num": result.u_num,
    }


def _curvatures_match(
    expected: Mapping[str, object], observed: Mapping[str, object]
) -> bool:
    if observed.get("status") != expected.get("status"):
        return False
    if observed.get("negative") is not expected.get("negative"):
        return False
    for key in _SCALAR_DIAGNOSTIC_KEYS:
        expected_value = expected.get(key)
        observed_value = observed.get(key)
        if not isinstance(expected_value, (int, float)) or not isinstance(
            observed_value, (int, float)
        ):
            return False
        if not math.isclose(
            float(observed_value),
            float(expected_value),
            rel_tol=0.0,
            abs_tol=_NUMERICAL_ATOL,
        ):
            return False
    return True


def _analytic_entry(
    name: str, result: LRRCResult, analytic_curvature: float, negative: bool
) -> dict[str, object]:
    expected: dict[str, object] = {
        "status": LRRCStatus.OK.value,
        "negative": negative,
        "d_star": _PAIR_D_STAR,
        "h": _PAIR_H,
        "kappa_h": analytic_curvature,
        "kappa_h2": analytic_curvature,
        "kappa_r": analytic_curvature,
        "error_proxy": 0.0,
        "u_num": analytic_curvature,
    }
    observed = _curvature_payload(result)
    return {
        "name": name,
        "expected": expected,
        "observed": observed,
        "passed": _curvatures_match(expected, observed),
    }


def _invariance_entry(
    name: str, reference: LRRCResult, observed_result: LRRCResult
) -> dict[str, object]:
    reference_payload = _curvature_payload(reference)
    expected = {"reference_case": "positive_harmonic", **reference_payload}
    observed = _curvature_payload(observed_result)
    return {
        "name": name,
        "expected": expected,
        "observed": observed,
        "passed": _curvatures_match(reference_payload, observed),
    }


def _status_entry(
    name: str, result: LRRCResult, expected_status: LRRCStatus
) -> dict[str, object]:
    expected = {"status": expected_status.value, "negative": None}
    observed = {"status": result.status.value, "negative": result.negative}
    return {
        "name": name,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def _decision_entry(
    name: str,
    baseline: Decision,
    lrrc: LRRCResult,
    expected_negative: bool,
    expected_decision: Decision,
) -> dict[str, object]:
    expected = {
        "baseline": baseline.value,
        "lrrc_negative": expected_negative,
        "decision": expected_decision.value,
    }
    observed = {
        "baseline": baseline.value,
        "lrrc_negative": lrrc.negative,
        "decision": compose_decision(baseline, lrrc).value,
    }
    return {
        "name": name,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def _quota_entries() -> list[dict[str, object]]:
    rows = (
        QuotaCRCRow("q0", "g", 0.1, Decision.REJECT),
        QuotaCRCRow("q1", "g", 0.2, Decision.REJECT),
        QuotaCRCRow("q2", "g", 0.2, Decision.REJECT),
        QuotaCRCRow("q3", "g", 0.2, Decision.REJECT),
        QuotaCRCRow("q4", "g", 0.8, Decision.REJECT),
        QuotaCRCRow("q5", "g", 1.0, Decision.REJECT),
        QuotaCRCRow("q_abstain", "g", 999.0, Decision.ABSTAIN),
    )
    output = quota_crc(rows)
    output_by_id = {row.row_id: row for row in output}
    eligible = [
        row
        for row in rows
        if row.supported and row.decision is not Decision.ABSTAIN
    ]
    k = math.ceil(math.sqrt(len(eligible)))
    boundary_score = sorted(row.score for row in eligible)[k - 1]

    fixed_expected = {
        "eligible_count": 6,
        "k": 3,
        "boundary_score": 0.2,
        "output_decisions": {
            "q0": Decision.KEEP.value,
            "q1": Decision.KEEP.value,
            "q2": Decision.KEEP.value,
            "q3": Decision.KEEP.value,
            "q4": Decision.REJECT.value,
            "q5": Decision.REJECT.value,
        },
    }
    fixed_observed = {
        "eligible_count": len(eligible),
        "k": k,
        "boundary_score": boundary_score,
        "output_decisions": {
            row.row_id: output_by_id[row.row_id].decision.value for row in eligible
        },
    }

    boundary_rows = [row for row in eligible if row.score == boundary_score]
    boundary_expected = {
        "boundary_ids": ["q1", "q2", "q3"],
        "boundary_decisions": [Decision.KEEP.value] * 3,
        "eligible_keep_count": 4,
        "kept_beyond_k_due_to_tie": True,
    }
    eligible_keep_count = sum(
        output_by_id[row.row_id].decision is Decision.KEEP for row in eligible
    )
    boundary_observed = {
        "boundary_ids": [row.row_id for row in boundary_rows],
        "boundary_decisions": [
            output_by_id[row.row_id].decision.value for row in boundary_rows
        ],
        "eligible_keep_count": eligible_keep_count,
        "kept_beyond_k_due_to_tie": eligible_keep_count > k,
    }

    abstain_expected = {
        "row_id": "q_abstain",
        "input_decision": Decision.ABSTAIN.value,
        "output_decision": Decision.ABSTAIN.value,
    }
    abstain_input = next(row for row in rows if row.row_id == "q_abstain")
    abstain_observed = {
        "row_id": abstain_input.row_id,
        "input_decision": abstain_input.decision.value,
        "output_decision": output_by_id[abstain_input.row_id].decision.value,
    }

    input_reject_ids = [
        row.row_id for row in rows if row.decision is Decision.REJECT
    ]
    output_reject_ids = [
        row.row_id for row in output if row.decision is Decision.REJECT
    ]
    subset_expected = {
        "input_reject_ids": ["q0", "q1", "q2", "q3", "q4", "q5"],
        "output_reject_ids": ["q4", "q5"],
        "is_subset": True,
    }
    subset_observed = {
        "input_reject_ids": input_reject_ids,
        "output_reject_ids": output_reject_ids,
        "is_subset": set(output_reject_ids).issubset(input_reject_ids),
    }

    contracts = (
        ("quota_fixed_ceil_sqrt_n", fixed_expected, fixed_observed),
        ("quota_boundary_ties", boundary_expected, boundary_observed),
        ("quota_abstain_unchanged", abstain_expected, abstain_observed),
        ("quota_rejection_subset", subset_expected, subset_observed),
    )
    return [
        {
            "name": name,
            "expected": expected,
            "observed": observed,
            "passed": observed == expected,
        }
        for name, expected, observed in contracts
    ]


def _source_paths() -> dict[str, Path]:
    return {
        "src/next9_lrrc.py": Path(__file__).with_name("next9_lrrc.py").resolve(),
        "src/next9_lrrc_synthetic.py": Path(__file__).resolve(),
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
    evaluations = {
        case.name: evaluate_lrrc(case.atoms, case.oracle)
        for case in _analytic_cases()
    }

    cases: list[dict[str, object]] = [
        _analytic_entry(
            "positive_harmonic", evaluations["positive_harmonic"], 3.0, False
        ),
        _analytic_entry(
            "inverted_harmonic", evaluations["inverted_harmonic"], -6.0, True
        ),
    ]
    for name in _EVALUATION_ORDER[2:6]:
        cases.append(
            _invariance_entry(
                name, evaluations["positive_harmonic"], evaluations[name]
            )
        )
    cases.append(
        _status_entry(
            "exact_zero_force_saddle",
            evaluations["exact_zero_force_saddle"],
            LRRCStatus.STATIONARY_FALLBACK,
        )
    )

    def raising_oracle(_atoms: Atoms) -> np.ndarray:
        raise RuntimeError("synthetic oracle exception")

    base = _base_atoms()
    failure_contracts = (
        (
            "oracle_exception",
            evaluate_lrrc(base, raising_oracle),
            LRRCStatus.ABSTAIN_FORCE_FAILURE,
        ),
        (
            "wrong_shape_force",
            evaluate_lrrc(base, lambda _atoms: np.zeros((2, 3))),
            LRRCStatus.ABSTAIN_INVALID_FORCE,
        ),
        (
            "nonfinite_force",
            evaluate_lrrc(base, lambda atoms: np.full((len(atoms), 3), np.nan)),
            LRRCStatus.ABSTAIN_INVALID_FORCE,
        ),
    )
    cases.extend(
        _status_entry(name, result, expected_status)
        for name, result, expected_status in failure_contracts
    )
    cases.extend(
        (
            _decision_entry(
                "decision_keep_or_negative",
                Decision.KEEP,
                evaluations["inverted_harmonic"],
                True,
                Decision.REJECT,
            ),
            _decision_entry(
                "decision_reject_or_nonnegative",
                Decision.REJECT,
                evaluations["positive_harmonic"],
                False,
                Decision.REJECT,
            ),
        )
    )
    cases.extend(_quota_entries())

    observed_order = tuple(case["name"] for case in cases)
    if observed_order != CASE_ORDER:
        raise AssertionError("manifest case order differs from the frozen contract")
    engineering_pass = all(case["passed"] is True for case in cases)
    return {
        "version": LRRC_VERSION,
        "constants": {
            "step_fraction": STEP_FRACTION,
            "force_rms_floor": FORCE_RMS_FLOOR,
        },
        "formulas": {
            "direction": (
                "u_i=(F_i-mean_j(F_j))/sqrt(mean_i(sum_a((F_ia-mean_j(F_ja))^2)))"
            ),
            "step": "h=step_fraction*d_star",
            "kappa_delta": (
                "kappa_delta=-(1/N)*sum_i(u_i dot "
                "(F(x+delta*u)-F(x-delta*u))/(2*delta))"
            ),
            "kappa_r": "kappa_r=(4*kappa_h2-kappa_h)/3",
            "error_proxy": "error_proxy=abs(kappa_h2-kappa_h)/3",
            "u_num": (
                "u_num=kappa_r+error_proxy; numerical proxy, not a confidence bound"
            ),
            "negative": "kappa_h<0 and kappa_h2<0 and u_num<0",
            "decision": "REJECT iff baseline REJECT or LRRC negative",
            "quota_crc": (
                "k=ceil(sqrt(n)); q=kth-smallest eligible score; "
                "score<=q becomes KEEP"
            ),
            "synthetic_pair_curvature": (
                "for F_i=c*sum_j(r_j-r_i) and mean(u)=0, kappa=N*c"
            ),
        },
        "source_sha256": {
            logical_name: _sha256(path) for logical_name, path in sources.items()
        },
        "case_order": list(CASE_ORDER),
        "cases": cases,
        "known_limitations": [
            (
                "LRRC-v0 probes one local force-derived direction; u_num is not a "
                "confidence bound or a formal stability certificate."
            ),
            (
                "An exact zero-force saddle triggers STATIONARY_FALLBACK and is not "
                "identified by LRRC-v0."
            ),
            (
                "These engineering cases read no datasets, import no MatterSim, and "
                "make no measured scientific-improvement claim."
            ),
        ],
        "engineering_pass": engineering_pass,
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
    """Atomically publish a directory, failing closed without a no-replace primitive."""

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
    """Run fixed analytic cases and atomically publish one strict manifest.

    The target must not exist. Publication uses a complete sibling temporary
    directory and a no-replace rename; existing content is never overwritten.
    Every recorded source is rehashed immediately before publication.
    """

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
    """CLI entry point with ``--output-dir`` as its sole application option."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    run(arguments.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
