from __future__ import annotations

import numpy as np
import pandas as pd

from src.next41_source_balanced_guard_search import scan_source_balanced_guards


def _source(*, invert_guard: bool = False) -> pd.DataFrame:
    protected = 100
    changed = 200
    endpoint = np.r_[np.zeros(protected), np.full(changed, 0.3)]
    guard = np.r_[np.zeros(protected), np.ones(changed)]
    if invert_guard:
        guard = 1.0 - guard
    return pd.DataFrame(
        {
            "material_id": [f"row-{i}" for i in range(protected + changed)],
            "score": np.ones(protected + changed),
            "score_supported": True,
            "contact_supported": True,
            "site_stats_fingerprint_init_final_norm_diff": endpoint,
            "cov_q01": -guard,
            "cov_q05": -guard,
            "cov_contact085_pa": guard,
            "cov_overlap2_pa": guard,
            "cov_site_overlap_q95": guard,
            "cov_site_overlap_max": guard,
        }
    )


def test_scan_requires_each_source_to_pass_and_can_find_clean_conjunction() -> None:
    result, table = scan_source_balanced_guards(
        {"source_a": _source(), "source_b": _source()}, quantile_count=5
    )

    assert result["eligible"] is True
    assert result["selected_candidate"] is not None
    selected = table.loc[table.candidate_id.eq(result["selected_candidate"])].iloc[0]
    assert selected.source_a_passes_primary_gates
    assert selected.source_b_passes_primary_gates


def test_scan_does_not_let_one_source_rescue_another() -> None:
    result, _table = scan_source_balanced_guards(
        {"source_a": _source(), "source_b": _source(invert_guard=True)},
        quantile_count=5,
    )

    assert result["eligible"] is False
    assert result["selected_candidate"] is None
