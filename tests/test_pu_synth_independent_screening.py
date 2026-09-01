from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.pu_synthesizability_20260821.independent_screening import (
    build_independent_frontier,
    build_operating_point_summary,
)


def _frame(
    scores: list[float],
    verdicts: list[str],
    *,
    cohort: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cohort": cohort,
            "cif_sha256": [f"{i:064x}" for i in range(len(scores))],
            "L4_verdict": verdicts,
            "S_syn": scores,
        }
    )


def test_independent_frontier_contains_formula_curve_and_one_fixed_l4_point() -> None:
    experimental = _frame(
        [0.0, 1.0, 2.0, 3.0],
        ["pass", "explicit_violation", "pass", "pass"],
        cohort="experimental",
    )
    pu = _frame(
        [-1.0, 0.5, 2.5, 4.0],
        ["explicit_violation", "pass", "pass", "pass"],
        cohort="pu_negative",
    )

    result = build_independent_frontier(
        experimental, pu, retentions=(0.75, 0.5)
    )

    assert set(result["method"]) == {"S_syn", "L4"}
    assert (result["method"] == "L4").sum() == 1
    formula = result.loc[result.method.eq("S_syn")]
    assert formula["experimental_retention"].between(0.0, 1.0).all()
    assert formula["pu_screened_rate"].between(0.0, 1.0).all()
    l4 = result.loc[result.method.eq("L4")].iloc[0]
    assert np.isclose(l4["experimental_retention"], 0.75)
    assert np.isclose(l4["pu_screened_rate"], 0.25)


def test_operating_summary_compares_formula_at_l4_retention_without_combining() -> None:
    experimental = _frame(
        [0.0, 1.0, 2.0, 3.0],
        ["pass", "explicit_violation", "pass", "pass"],
        cohort="experimental",
    )
    pu = _frame(
        [-1.0, 0.5, 2.5, 4.0],
        ["explicit_violation", "pass", "pass", "pass"],
        cohort="pu_negative",
    )
    summary = build_operating_point_summary(experimental, pu)

    assert set(summary["method"]) == {"L4", "S_syn"}
    assert not summary["combined"].any()
    assert summary.loc[summary.method.eq("L4"), "pu_screened_n"].iloc[0] == 1
    assert summary.loc[summary.method.eq("S_syn"), "experimental_retention"].iloc[0] == 0.75
