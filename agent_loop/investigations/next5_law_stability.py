#!/usr/bin/env python3
"""True LOKO refits and final development gates for np-next-20260801e."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

from better_law_search import _candidate_record, pareto_beam
from next4_law_search import (
    _sha256,
    build_next4_candidate_sets,
    load_next4_search_frames,
    robust_pareto_beam,
    verify_isolated_manifest,
)
from next4_law_stability import (
    _held_metrics,
    iter_true_loko,
    signed_delta_summary,
)
from next5_law_search import EXPERIMENT_ID, paired_soft_strata


def loko_success_gate(
    discovery_deltas: Sequence[float],
    calibration_deltas: Sequence[float],
    *,
    minimum_allowed: float = -0.02,
) -> dict[str, object]:
    """Require non-negative macro delta and no held-kind collapse."""

    discovery = signed_delta_summary(discovery_deltas)
    calibration = signed_delta_summary(calibration_deltas)
    passes = bool(
        discovery["signed_mean"] >= 0
        and calibration["signed_mean"] >= 0
        and discovery["minimum"] >= minimum_allowed
        and calibration["minimum"] >= minimum_allowed
    )
    return {
        "minimum_allowed": minimum_allowed,
        "discovery": discovery,
        "calibration": calibration,
        "passes": passes,
    }


def validate_reference_contract(
    reference: Mapping[str, object],
    *,
    floor: float,
    current_input_sha256: Mapping[str, str],
    manifest_sha256: str,
) -> None:
    """Fail closed when a reference report is incompatible with this run."""

    protocol = reference.get("protocol", {})
    if protocol.get("experiment") != EXPERIMENT_ID:
        raise ValueError("reference experiment is incompatible")
    if float(protocol.get("floor", -1.0)) != float(floor):
        raise ValueError("reference floor is incompatible")
    provenance = reference.get("provenance", {})
    if dict(provenance.get("input_sha256", {})) != dict(current_input_sha256):
        raise ValueError("reference input hashes are incompatible")
    isolation = reference.get("isolation_audit", {})
    if isolation.get("manifest_sha256") != manifest_sha256:
        raise ValueError("reference isolated manifest is incompatible")


def validate_input_domain_audit(
    audit: Mapping[str, object],
    current_input_sha256: Mapping[str, str],
) -> None:
    """Require a passing descriptor-domain audit tied to current inputs."""

    if not audit.get("all_descriptor_keys_within_isolated_domains"):
        raise ValueError("input domain audit did not pass")
    descriptors = audit.get("descriptors", {})
    recorded = {
        name: entry.get("descriptor_sha256")
        for name, entry in descriptors.items()
        if name in current_input_sha256
    }
    if recorded != dict(current_input_sha256):
        raise ValueError("input domain audit descriptor hashes are incompatible")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated-dir", type=Path, required=True)
    parser.add_argument("--real-descriptors", type=Path, required=True)
    parser.add_argument("--bad-descriptors", type=Path, required=True)
    parser.add_argument("--real-sixfam", type=Path, required=True)
    parser.add_argument("--bad-sixfam", type=Path, required=True)
    parser.add_argument("--real-corrected", type=Path, required=True)
    parser.add_argument("--bad-corrected", type=Path, required=True)
    parser.add_argument("--real-guards", type=Path, required=True)
    parser.add_argument("--bad-guards", type=Path, required=True)
    parser.add_argument("--input-domain-audit", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--falsification-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--floor", type=float, default=0.98)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--max-rules", type=int, default=12)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--max-guard-targets", type=int, default=100)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--min-anion-rows", type=int, default=200)
    parser.add_argument("--min-cell-rows", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    started = time.time()
    input_paths = {
        "real_descriptors": args.real_descriptors,
        "bad_descriptors": args.bad_descriptors,
        "real_sixfam": args.real_sixfam,
        "bad_sixfam": args.bad_sixfam,
        "real_corrected": args.real_corrected,
        "bad_corrected": args.bad_corrected,
        "real_guards": args.real_guards,
        "bad_guards": args.bad_guards,
    }
    input_sha256 = {name: _sha256(path) for name, path in input_paths.items()}
    domain_audit = json.loads(
        args.input_domain_audit.read_text(encoding="utf-8")
    )
    validate_input_domain_audit(domain_audit, input_sha256)
    isolation_audit = verify_isolated_manifest(args.isolated_dir)
    reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
    validate_reference_contract(
        reference,
        floor=args.floor,
        current_input_sha256=input_sha256,
        manifest_sha256=str(isolation_audit["manifest_sha256"]),
    )
    falsification = json.loads(
        args.falsification_report.read_text(encoding="utf-8")
    )
    if falsification.get("protocol", {}).get("experiment") != EXPERIMENT_ID:
        raise SystemExit("falsification report experiment is incompatible")
    if falsification.get("provenance", {}).get("law_report_sha256") != _sha256(
        args.reference_report
    ):
        raise SystemExit("falsification report references a different law report")

    dr, cr, db, cb = load_next4_search_frames(
        args.isolated_dir,
        args.real_descriptors,
        args.bad_descriptors,
        args.real_sixfam,
        args.bad_sixfam,
        args.real_corrected,
        args.bad_corrected,
        args.real_guards,
        args.bad_guards,
    )
    floor_key = str(args.floor)
    full_existing = reference["frontiers"]["existing_loop"][floor_key][
        "discovery"
    ]["by_kind"]
    full_candidate = reference["frontiers"][
        reference["comparison_variant"]
    ][floor_key]["discovery"]["by_kind"]

    folds: dict[str, object] = {}
    discovery_deltas: list[float] = []
    calibration_deltas: list[float] = []
    fullfit_changes: list[float] = []
    for held, training_bad, held_discovery, held_calibration in iter_true_loko(db, cb):
        candidate_sets, counts = build_next4_candidate_sets(
            dr,
            training_bad,
            min_coverage=args.min_coverage,
            max_guard_targets=args.max_guard_targets,
            guard_min_real_satisfaction=args.floor,
        )
        existing_candidates = candidate_sets["existing_loop"]
        additive_candidates = candidate_sets["additive_corrected_loop"]
        baseline = pareto_beam(
            existing_candidates,
            real_size=len(dr),
            bad_size=len(training_bad),
            bad_kinds=training_bad["kind"].to_numpy(),
            satisfaction_floor=args.floor,
            max_rules=args.max_rules,
            width=24,
        )
        strata, floors, _metadata = paired_soft_strata(
            dr,
            baseline.real_mask,
            n_folds=args.n_folds,
            min_anion_rows=args.min_anion_rows,
            min_cell_rows=args.min_cell_rows,
        )
        candidate = robust_pareto_beam(
            additive_candidates,
            real_size=len(dr),
            bad_size=len(training_bad),
            bad_kinds=training_bad["kind"].to_numpy(),
            satisfaction_floor=args.floor,
            max_rules=args.max_rules,
            width=args.width,
            real_strata=strata,
            stratum_floors=floors,
        )
        base_discovery = _held_metrics(
            dr, held_discovery, existing_candidates, baseline
        )
        candidate_discovery = _held_metrics(
            dr, held_discovery, additive_candidates, candidate
        )
        base_calibration = _held_metrics(
            cr, held_calibration, existing_candidates, baseline
        )
        candidate_calibration = _held_metrics(
            cr, held_calibration, additive_candidates, candidate
        )
        if base_discovery is None or candidate_discovery is None:
            raise RuntimeError(f"held discovery kind unexpectedly empty: {held}")
        discovery_delta = float(
            candidate_discovery["rejection"] - base_discovery["rejection"]
        )
        discovery_deltas.append(discovery_delta)
        fullfit_change = float(
            candidate_discovery["rejection"] - float(full_candidate[held])
        )
        fullfit_changes.append(fullfit_change)
        calibration_delta = None
        if base_calibration is not None and candidate_calibration is not None:
            calibration_delta = float(
                candidate_calibration["rejection"] - base_calibration["rejection"]
            )
            calibration_deltas.append(calibration_delta)
        rules = [
            _candidate_record(additive_candidates[index])
            for index in candidate.indices
        ]
        rule_sha = hashlib.sha256(
            json.dumps(
                rules,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        folds[held] = {
            "training_kinds": sorted(set(training_bad["kind"]).difference({held})),
            "training_bad_count": len(training_bad),
            "held_discovery_count": len(held_discovery),
            "held_calibration_count": len(held_calibration),
            "robust_strata": len(strata),
            "candidate_counts": counts,
            "rules": rules,
            "rule_sha256": rule_sha,
            "existing_held_discovery": base_discovery,
            "candidate_held_discovery": candidate_discovery,
            "candidate_minus_existing_held_discovery_rejection": discovery_delta,
            "candidate_held_minus_fullfit_held_discovery_rejection": fullfit_change,
            "existing_fullfit_held_discovery_rejection": float(full_existing[held]),
            "candidate_fullfit_held_discovery_rejection": float(full_candidate[held]),
            "existing_held_calibration": base_calibration,
            "candidate_held_calibration": candidate_calibration,
            "candidate_minus_existing_held_calibration_rejection": calibration_delta,
        }
        print(
            f"held {held}: discovery delta {discovery_delta:+.4f}; "
            f"calibration delta "
            f"{calibration_delta if calibration_delta is not None else float('nan'):+.4f}; "
            f"full-fit change {fullfit_change:+.4f}",
            flush=True,
        )

    loko_gate = loko_success_gate(discovery_deltas, calibration_deltas)
    all295_gate = bool(
        falsification["comparisons"][floor_key][
            "combined_development_gate_before_loko"
        ]
    )
    report = {
        "protocol": {
            "experiment": EXPERIMENT_ID,
            "analysis": "true leave-one-perturbation-kind-out refit",
            "held_kind_exposed_to_search": False,
            "held_parent_exposed_via_other_kinds": True,
            "floor": args.floor,
            "robust_width": args.width,
            "full_anion_margin": reference["protocol"]["full_anion_margin"],
            "anion_by_fold_margin": reference["protocol"]["anion_by_fold_margin"],
            "lockbox_access": False,
        },
        "isolation_audit": isolation_audit,
        "input_domain_audit_sha256": _sha256(args.input_domain_audit),
        "folds": folds,
        "summary": {
            "candidate_minus_existing_held_discovery_rejection": signed_delta_summary(
                discovery_deltas
            ),
            "candidate_minus_existing_held_calibration_rejection": signed_delta_summary(
                calibration_deltas
            ),
            "candidate_held_minus_fullfit_held_discovery_rejection": signed_delta_summary(
                fullfit_changes
            ),
            "loko_stability_gate": loko_gate,
            "prior_all295_and_pre_loko_gate": all295_gate,
            "final_development_gate": bool(all295_gate and loko_gate["passes"]),
        },
        "provenance": {
            "runtime_seconds": time.time() - started,
            "reference_report_sha256": _sha256(args.reference_report),
            "falsification_report_sha256": _sha256(args.falsification_report),
            "input_sha256": input_sha256,
            "implementation_sha256": _sha256(Path(__file__)),
            "dependency_sha256": {
                "next4_law_stability": _sha256(
                    Path(__file__).with_name("next4_law_stability.py")
                ),
                "next5_law_search": _sha256(
                    Path(__file__).with_name("next5_law_search.py")
                ),
            },
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(
        f"LOKO gate={loko_gate['passes']}; final development gate="
        f"{report['summary']['final_development_gate']}",
        flush=True,
    )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
