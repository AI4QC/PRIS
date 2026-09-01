"""Contracts for frozen ACSC execution and decision composition on WBM."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("phsc", "chsc", "coupling", "nested", "expected"),
    [
        ("resolved_nonnegative", "resolved_nonnegative", False, False, ("KEEP", "KEEP", "KEEP")),
        ("resolved_negative", "resolved_nonnegative", False, False, ("REJECT", "REJECT", "REJECT")),
        ("resolved_nonnegative", "resolved_negative", False, False, ("REJECT", "REJECT", "REJECT")),
        ("near_zero_or_inconsistent", "resolved_nonnegative", False, False, ("KEEP", "KEEP", "KEEP")),
        ("abstain_energy_failure", "resolved_nonnegative", False, False, ("ABSTAIN", "ABSTAIN", "ABSTAIN")),
        ("resolved_nonnegative", "resolved_nonnegative", True, False, ("KEEP", "REJECT", "KEEP")),
        ("resolved_nonnegative", "resolved_nonnegative", True, True, ("KEEP", "REJECT", "REJECT")),
    ],
)
def test_mechanical_decision_composition_is_frozen(
    phsc: str, chsc: str, coupling: bool, nested: bool, expected: tuple[str, str, str]
) -> None:
    from src.next14_wbm_acsc_features import mechanical_decisions

    assert mechanical_decisions(phsc, chsc, coupling, nested) == expected


def test_nested_confirmation_requires_coupling_only_candidate() -> None:
    from src.next14_wbm_acsc_features import mechanical_decisions

    with pytest.raises(ValueError, match="nested"):
        mechanical_decisions(
            "resolved_nonnegative", "resolved_nonnegative", False, True
        )


def test_acsc_cli_exposes_no_label_or_refit_argument() -> None:
    from src.next14_wbm_acsc_features import main

    for forbidden in ("--labels", "--threshold", "--fit", "--dft-results"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2

