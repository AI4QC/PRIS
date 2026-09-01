#!/usr/bin/env python3
"""Audit that every next4/next5 descriptor key belongs to isolated tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from next4_law_search import _sha256, verify_isolated_manifest


def audit_descriptor_domain(
    domain_path: Path,
    domain_key: str,
    descriptor_path: Path,
    descriptor_key: str,
) -> dict[str, object]:
    """Fail if a descriptor contains duplicate or non-isolated keys."""

    domain = pd.read_parquet(domain_path, columns=[domain_key])[domain_key]
    descriptor = pd.read_parquet(
        descriptor_path, columns=[descriptor_key]
    )[descriptor_key]
    if domain.isna().any():
        raise ValueError(f"isolated domain has null keys: {domain_key}")
    if descriptor.isna().any() or descriptor.duplicated().any():
        raise ValueError(
            f"descriptor has null or duplicate keys: {descriptor_key}"
        )
    domain_values = set(domain.astype(str))
    descriptor_values = set(descriptor.astype(str))
    extras = descriptor_values.difference(domain_values)
    if extras:
        raise ValueError(
            f"descriptor contains {len(extras)} keys outside isolated domain"
        )
    return {
        "domain_rows": int(len(domain)),
        "domain_unique_keys": int(len(domain_values)),
        "descriptor_rows": int(len(descriptor)),
        "extra_keys": 0,
        "missing_domain_keys": int(len(domain_values - descriptor_values)),
        "descriptor_keys_within_isolated_domain": True,
        "descriptor_sha256": _sha256(descriptor_path),
    }


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
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    isolated = verify_isolated_manifest(args.isolated_dir)
    real_domain = args.isolated_dir / "law_real.parquet"
    bad_domain = args.isolated_dir / "law_bad.parquet"
    specs = (
        ("real_descriptors", real_domain, "source_id", args.real_descriptors, "source_id"),
        ("real_sixfam", real_domain, "source_id", args.real_sixfam, "source_id"),
        ("real_corrected", real_domain, "source_id", args.real_corrected, "source_id"),
        ("real_guards", real_domain, "source_id", args.real_guards, "source_id"),
        ("bad_descriptors", bad_domain, "sid", args.bad_descriptors, "sid"),
        ("bad_sixfam", bad_domain, "sid", args.bad_sixfam, "sid"),
        ("bad_corrected", bad_domain, "sid", args.bad_corrected, "sid"),
        ("bad_guards", bad_domain, "parent", args.bad_guards, "parent"),
    )
    descriptors = {
        name: audit_descriptor_domain(domain, domain_key, path, descriptor_key)
        for name, domain, domain_key, path, descriptor_key in specs
    }
    output = {
        "protocol": {
            "experiment": "np-next-20260801e-input-domain-audit",
            "audit_only": True,
            "lockbox_access": False,
            "criterion": "all descriptor keys are subsets of physically isolated law-table key domains",
        },
        "isolated_manifest": isolated,
        "descriptors": descriptors,
        "all_descriptor_keys_within_isolated_domains": all(
            bool(entry["descriptor_keys_within_isolated_domain"])
            for entry in descriptors.values()
        ),
        "implementation_sha256": _sha256(Path(__file__)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(f"verified {len(descriptors)} descriptor key domains", flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
