"""Contracts for label-free NEXT17 strict-relaxation group gaps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_strict_relaxation_constants_are_distinct_from_next16() -> None:
    from src.next17_strict_relax_gap import (
        STRICT_RELAX_FMAX_EV_PER_A,
        STRICT_RELAX_MAX_PREDICTION_STEPS,
    )

    assert STRICT_RELAX_FMAX_EV_PER_A == 0.005
    assert STRICT_RELAX_MAX_PREDICTION_STEPS == 64


def test_group_gap_is_relative_and_an_incomplete_group_fails_open() -> None:
    from src.next17_strict_relax_gap import add_group_relative_gaps

    frame = pd.DataFrame(
        {
            "material_id": ["a", "b", "c", "d", "e"],
            "rk": ["Li2O", "Li2O", "NaCl", "NaCl", "NaCl"],
            "supported": [True, True, True, False, True],
            "energy_ev_per_atom": [-3.0, -2.94, -4.0, np.nan, -3.9],
        }
    )
    got = add_group_relative_gaps(frame)
    assert got.loc[:1, "strict_group_supported"].tolist() == [True, True]
    assert got.loc[:1, "strict_relative_gap_ev_per_atom"].tolist() == pytest.approx(
        [0.0, 0.06]
    )
    assert got.loc[2:, "strict_group_supported"].tolist() == [False, False, False]
    assert got.loc[2:, "strict_relative_gap_ev_per_atom"].isna().all()


def test_group_gap_rejects_duplicate_identity_or_invalid_supported_energy() -> None:
    from src.next17_strict_relax_gap import add_group_relative_gaps

    duplicate = pd.DataFrame(
        {
            "material_id": ["a", "a"],
            "rk": ["Li2O", "Li2O"],
            "supported": [True, True],
            "energy_ev_per_atom": [-3.0, -2.9],
        }
    )
    with pytest.raises(ValueError, match="unique"):
        add_group_relative_gaps(duplicate)
    invalid = duplicate.assign(material_id=["a", "b"], energy_ev_per_atom=[-3.0, np.nan])
    with pytest.raises(ValueError, match="supported energy"):
        add_group_relative_gaps(invalid)


def test_cli_cannot_read_labels_or_change_the_frozen_protocol() -> None:
    from src.next17_strict_relax_gap import main

    for forbidden in ("--labels", "--threshold", "--fmax", "--max-steps", "--mp-reference"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
