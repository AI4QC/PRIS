import numpy as np
import pandas as pd
import pytest

from src.next6_mattersim_evaluate import (
    join_mattersim_feature,
    mattersim_formula,
    mattersim_formulas,
)
from src.next6_elementa_search import score_formula


pytestmark = pytest.mark.filterwarnings("error")


def test_mattersim_join_is_keyed_and_non_strict_or_failed_rows_abstain():
    prepared = pd.DataFrame(
        {
            "sid": ["a", "b", "c"],
            "rk": ["x", "x", "x"],
            "strict_x0_ok": [True, True, False],
        }
    )
    predictions = pd.DataFrame(
        {
            "sid": ["c", "a", "b"],
            "rk": ["x", "x", "x"],
            "mattersim_feature_ok": [False, True, False],
            "mattersim_energy_per_atom": [np.nan, -2.0, -1.0],
            "mattersim_energy_total": [np.nan, -4.0, -2.0],
        }
    )
    got = join_mattersim_feature(prepared, predictions).set_index("sid")
    assert got.loc["a", "mattersim_energy_per_atom"] == -2.0
    assert bool(got.loc["a", "support_mattersim"])
    assert not bool(got.loc["b", "support_mattersim"])
    assert not bool(got.loc["c", "support_mattersim"])
    assert "mattersim_energy_total" not in got.columns

    scored = score_formula(got.reset_index(), mattersim_formula(), {}).set_index("sid")
    assert not bool(scored.loc["a", "decision_support"])
    # There is only one supported candidate in the group, so a relative rank
    # cannot be formed and must abstain rather than exploit row order.
    assert np.isnan(scored.loc["a", "score"])


def test_predicted_energy_gap_formula_keeps_continuous_within_group_margin():
    prepared = pd.DataFrame(
        {
            "sid": ["a", "b", "c"],
            "rk": ["x", "x", "x"],
            "strict_x0_ok": [True, True, True],
        }
    )
    predictions = pd.DataFrame(
        {
            "sid": ["b", "c", "a"],
            "rk": ["x", "x", "x"],
            "mattersim_feature_ok": [True, True, True],
            "mattersim_energy_per_atom": [-1.8, -1.0, -2.0],
        }
    )
    joined = join_mattersim_feature(prepared, predictions).set_index("sid")
    assert joined.mattersim_predicted_gap.to_dict() == pytest.approx(
        {"a": 0.0, "b": 0.2, "c": 1.0}
    )
    formulas = {formula.name: formula for formula in mattersim_formulas()}
    margin = formulas["mattersim_5m_predicted_gap"]
    assert margin.track == "cohort_margin"
