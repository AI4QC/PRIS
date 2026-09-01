"""Contracts for composing ACSC-v0 with the frozen label-free gate."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("old", "coupling_only", "expected"),
    [
        ("KEEP", True, "REJECT"),
        ("ABSTAIN", True, "REJECT"),
        ("REJECT", True, "REJECT"),
        ("KEEP", False, "KEEP"),
        ("ABSTAIN", False, "ABSTAIN"),
        ("REJECT", False, "REJECT"),
    ],
)
def test_resolved_coupling_negative_overrides_keep_or_abstain(
    old: str, coupling_only: bool, expected: str
) -> None:
    from src.next13_acsc_label_free_overlap import compose_decision

    assert compose_decision(old, coupling_only) == expected


@pytest.mark.parametrize("old", ["bad", "", None, 3])
def test_composition_rejects_unknown_old_decisions(old: object) -> None:
    from src.next13_acsc_label_free_overlap import compose_decision

    with pytest.raises(ValueError):
        compose_decision(old, False)  # type: ignore[arg-type]


def test_cli_exposes_no_label_or_endpoint_argument() -> None:
    from src.next13_acsc_label_free_overlap import main

    for forbidden in ("--labels", "--endpoint", "--dft-results"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
