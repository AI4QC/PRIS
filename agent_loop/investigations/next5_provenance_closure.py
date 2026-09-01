#!/usr/bin/env python3
"""Close provenance links across immutable np-next-20260801e artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from next4_law_search import _sha256
from next5_law_search import EXPERIMENT_ID


def validate_primary_binding(
    law: Mapping[str, object], audit: Mapping[str, object]
) -> None:
    """Tie every primary-search descriptor hash to the domain audit."""

    if not audit.get("all_descriptor_keys_within_isolated_domains"):
        raise ValueError("input-domain audit did not pass")
    law_hashes = dict(law.get("provenance", {}).get("input_sha256", {}))
    audit_descriptors = audit.get("descriptors", {})
    audit_hashes = {
        name: entry.get("descriptor_sha256")
        for name, entry in audit_descriptors.items()
    }
    if law_hashes != audit_hashes:
        raise ValueError("primary-search and domain-audit descriptor hashes differ")


def validate_falsification_contract(
    report: Mapping[str, object],
    *,
    law_sha256: str,
    fp_sha256: Mapping[str, str],
) -> None:
    """Verify the complete recorded all-295 contract before chaining it."""

    protocol = report.get("protocol", {})
    if int(protocol.get("n_population", -1)) != 295:
        raise ValueError("falsification population is not the frozen 295 rows")
    if protocol.get("missing_feature_semantics") != "unknown fails closed on all 295":
        raise ValueError("falsification does not use the frozen fail-closed protocol")
    for frontier in report.get("frontiers", {}).values():
        for evaluation in frontier.values():
            if int(evaluation.get("n", -1)) != 295:
                raise ValueError("falsification frontier does not contain 295 rows")
    provenance = report.get("provenance", {})
    if provenance.get("law_report_sha256") != law_sha256:
        raise ValueError("falsification references a different law report")
    recorded = {
        name: provenance.get(f"{name}_sha256") for name in fp_sha256
    }
    if recorded != dict(fp_sha256):
        raise ValueError("falsification descriptor hashes are incompatible")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--law-report", type=Path, required=True)
    parser.add_argument("--input-domain-audit", type=Path, required=True)
    parser.add_argument("--falsification-report", type=Path, required=True)
    parser.add_argument("--loko-report", type=Path, required=True)
    parser.add_argument("--consensus-report", type=Path, required=True)
    parser.add_argument("--fp-source", type=Path, required=True)
    parser.add_argument("--fp-p235", type=Path, required=True)
    parser.add_argument("--fp-guards", type=Path, required=True)
    parser.add_argument("--fp-sixfam", type=Path, required=True)
    parser.add_argument("--fp-corrected", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    law = json.loads(args.law_report.read_text(encoding="utf-8"))
    audit = json.loads(args.input_domain_audit.read_text(encoding="utf-8"))
    falsification = json.loads(
        args.falsification_report.read_text(encoding="utf-8")
    )
    loko = json.loads(args.loko_report.read_text(encoding="utf-8"))
    consensus = json.loads(args.consensus_report.read_text(encoding="utf-8"))
    if law.get("protocol", {}).get("experiment") != EXPERIMENT_ID:
        raise SystemExit("law report experiment is incompatible")
    validate_primary_binding(law, audit)

    law_sha = _sha256(args.law_report)
    audit_sha = _sha256(args.input_domain_audit)
    falsification_sha = _sha256(args.falsification_report)
    loko_sha = _sha256(args.loko_report)
    fp_paths = {
        "fp_p235": args.fp_p235,
        "fp_guards": args.fp_guards,
        "fp_sixfam": args.fp_sixfam,
        "fp_corrected": args.fp_corrected,
    }
    fp_sha = {name: _sha256(path) for name, path in fp_paths.items()}
    validate_falsification_contract(
        falsification, law_sha256=law_sha, fp_sha256=fp_sha
    )

    source = pd.read_parquet(args.fp_source, columns=["sid"])
    if len(source) != 295 or source["sid"].isna().any() or source["sid"].duplicated().any():
        raise SystemExit("frozen false-positive source is not 295 unique sid rows")
    if loko.get("protocol", {}).get("experiment") != EXPERIMENT_ID:
        raise SystemExit("LOKO report experiment is incompatible")
    if loko.get("provenance", {}).get("reference_report_sha256") != law_sha:
        raise SystemExit("LOKO report references a different law report")
    if loko.get("provenance", {}).get("falsification_report_sha256") != falsification_sha:
        raise SystemExit("LOKO report references a different falsification report")
    if loko.get("input_domain_audit_sha256") != audit_sha:
        raise SystemExit("LOKO report references a different input-domain audit")
    if consensus.get("protocol", {}).get("source_experiment") != EXPERIMENT_ID:
        raise SystemExit("consensus source experiment is incompatible")
    consensus_provenance = consensus.get("provenance", {})
    if consensus_provenance.get("reference_report_sha256") != law_sha:
        raise SystemExit("consensus references a different law report")
    if consensus_provenance.get("loko_report_sha256") != loko_sha:
        raise SystemExit("consensus references a different LOKO report")
    if consensus_provenance.get("input_domain_audit_sha256") != audit_sha:
        raise SystemExit("consensus references a different input-domain audit")

    output = {
        "protocol": {
            "experiment": "np-next-20260801e-provenance-closure",
            "source_experiment": EXPERIMENT_ID,
            "post_hoc_provenance_only": True,
            "result_metrics_recomputed": False,
            "lockbox_access": False,
        },
        "all_contracts_pass": True,
        "artifact_sha256": {
            "law_report": law_sha,
            "input_domain_audit": audit_sha,
            "falsification_report": falsification_sha,
            "loko_report": loko_sha,
            "consensus_report": _sha256(args.consensus_report),
        },
        "primary_descriptor_sha256": law["provenance"]["input_sha256"],
        "false_positive": {
            "source_rows": 295,
            "source_sha256": _sha256(args.fp_source),
            "descriptor_sha256": fp_sha,
            "protocol": "unknown fails closed on all 295",
        },
        "implementation_sha256": _sha256(Path(__file__)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print("closed primary, all-295, LOKO, and consensus provenance", flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
