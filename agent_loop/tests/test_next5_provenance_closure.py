"""Tests for post-hoc provenance closure across immutable next5 artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from next5_provenance_closure import main as closure_main  # noqa: E402
from next5_provenance_closure import validate_falsification_contract  # noqa: E402
from next5_provenance_closure import validate_primary_binding  # noqa: E402


def test_primary_binding_requires_all_law_inputs_in_passing_domain_audit():
    law = {"provenance": {"input_sha256": {"real_descriptors": "a"}}}
    audit = {
        "all_descriptor_keys_within_isolated_domains": True,
        "descriptors": {"real_descriptors": {"descriptor_sha256": "a"}},
    }
    validate_primary_binding(law, audit)
    audit["descriptors"]["real_descriptors"]["descriptor_sha256"] = "changed"
    with np.testing.assert_raises_regex(ValueError, "descriptor hashes"):
        validate_primary_binding(law, audit)


def test_falsification_contract_checks_population_protocol_and_descriptors():
    report = {
        "protocol": {
            "n_population": 295,
            "missing_feature_semantics": "unknown fails closed on all 295",
        },
        "frontiers": {
            "existing_loop": {"0.98": {"n": 295}},
            "candidate": {"0.98": {"n": 295}},
        },
        "provenance": {
            "law_report_sha256": "law",
            "fp_p235_sha256": "p235",
            "fp_guards_sha256": "guards",
            "fp_sixfam_sha256": "sixfam",
            "fp_corrected_sha256": "corrected",
        },
    }
    hashes = {
        "fp_p235": "p235",
        "fp_guards": "guards",
        "fp_sixfam": "sixfam",
        "fp_corrected": "corrected",
    }
    validate_falsification_contract(report, law_sha256="law", fp_sha256=hashes)
    report["protocol"]["n_population"] = 294
    with np.testing.assert_raises_regex(ValueError, "295"):
        validate_falsification_contract(
            report, law_sha256="law", fp_sha256=hashes
        )


def test_closure_cli_refuses_to_overwrite_before_loading_inputs(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("keep", encoding="utf-8")
    with np.testing.assert_raises_regex(SystemExit, "refusing to overwrite"):
        closure_main(
            [
                "--law-report",
                str(tmp_path / "law.json"),
                "--input-domain-audit",
                str(tmp_path / "audit.json"),
                "--falsification-report",
                str(tmp_path / "fp.json"),
                "--loko-report",
                str(tmp_path / "loko.json"),
                "--consensus-report",
                str(tmp_path / "consensus.json"),
                "--fp-source",
                str(tmp_path / "source.parquet"),
                "--fp-p235",
                str(tmp_path / "p235.parquet"),
                "--fp-guards",
                str(tmp_path / "guards.parquet"),
                "--fp-sixfam",
                str(tmp_path / "sixfam.parquet"),
                "--fp-corrected",
                str(tmp_path / "corrected.parquet"),
                "--out",
                str(output),
            ]
        )
    assert output.read_text(encoding="utf-8") == "keep"
