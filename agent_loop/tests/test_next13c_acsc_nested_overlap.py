"""Contracts for composing nested three-scale ACSC confirmation."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("old", "confirmed", "expected"),
    [
        ("KEEP", True, "REJECT"),
        ("ABSTAIN", True, "REJECT"),
        ("REJECT", True, "REJECT"),
        ("KEEP", False, "KEEP"),
        ("ABSTAIN", False, "ABSTAIN"),
        ("REJECT", False, "REJECT"),
    ],
)
def test_nested_confirmation_is_the_only_new_override(
    old: str, confirmed: bool, expected: str
) -> None:
    from src.next13c_acsc_nested_overlap import nested_decision

    assert nested_decision(old, confirmed) == expected


def test_cli_exposes_no_label_or_endpoint_argument() -> None:
    from src.next13c_acsc_nested_overlap import main

    for forbidden in ("--labels", "--endpoint", "--dft-results"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
