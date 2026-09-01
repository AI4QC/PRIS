"""Tests for descriptor-domain isolation auditing."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from next5_input_audit import audit_descriptor_domain  # noqa: E402


def test_descriptor_domain_accepts_unique_subset_and_reports_missing(tmp_path):
    base = tmp_path / "base.parquet"
    descriptor = tmp_path / "descriptor.parquet"
    pd.DataFrame({"source_id": ["a", "b", "c"]}).to_parquet(base)
    pd.DataFrame({"source_id": ["a", "c"], "value": [1, 2]}).to_parquet(
        descriptor
    )

    result = audit_descriptor_domain(
        base,
        "source_id",
        descriptor,
        "source_id",
    )

    assert result["domain_rows"] == 3
    assert result["descriptor_rows"] == 2
    assert result["extra_keys"] == 0
    assert result["missing_domain_keys"] == 1
    assert result["descriptor_keys_within_isolated_domain"] is True


def test_descriptor_domain_fails_on_extra_or_duplicate_keys(tmp_path):
    base = tmp_path / "base.parquet"
    pd.DataFrame({"sid": ["a", "b"]}).to_parquet(base)
    extra = tmp_path / "extra.parquet"
    pd.DataFrame({"sid": ["a", "x"]}).to_parquet(extra)
    duplicate = tmp_path / "duplicate.parquet"
    pd.DataFrame({"sid": ["a", "a"]}).to_parquet(duplicate)

    try:
        audit_descriptor_domain(base, "sid", extra, "sid")
    except ValueError as error:
        assert "outside isolated domain" in str(error)
    else:
        raise AssertionError("extra descriptor key was accepted")
    try:
        audit_descriptor_domain(base, "sid", duplicate, "sid")
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate descriptor key was accepted")


def test_isolated_domain_may_repeat_many_to_one_parent_keys(tmp_path):
    base = tmp_path / "bad.parquet"
    descriptor = tmp_path / "guards.parquet"
    pd.DataFrame({"parent": ["p1", "p1", "p2"]}).to_parquet(base)
    pd.DataFrame({"parent": ["p1", "p2"]}).to_parquet(descriptor)

    result = audit_descriptor_domain(base, "parent", descriptor, "parent")

    assert result["domain_rows"] == 3
    assert result["domain_unique_keys"] == 2
    assert result["missing_domain_keys"] == 0
