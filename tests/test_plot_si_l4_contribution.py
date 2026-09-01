import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import pandas as pd
import pytest

from experiments.pu_synthesizability_20260821 import (
    plot_si_l4_contribution as contribution,
)


def _toy_frame() -> pd.DataFrame:
    rows = []
    patterns = [
        ("a", {"D1"}),
        ("a", {"D1"}),  # duplicate release record; one unique CIF
        ("b", {"D7"}),
        ("c", {"D1", "D7"}),
        ("d", set()),
    ]
    for cif, active in patterns:
        row = {"cif_sha256": cif}
        for law in contribution.L4_LAWS:
            row[f"{law}_verdict"] = (
                "explicit_violation" if law in active else "pass"
            )
        row["L4_verdict"] = (
            "explicit_violation" if active else "pass"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_l4_contribution_is_unique_cif_and_overlap_aware():
    summary, metadata = contribution.summarize_cohort(
        _toy_frame(), cohort="experimental"
    )

    assert metadata["rows"] == 5
    assert metadata["unique_cifs"] == 4
    assert metadata["l4_union_mismatches"] == 0
    assert metadata["l4_violation_rate_pp"] == pytest.approx(75.0)
    d1 = summary.set_index("law").loc["D1"]
    d7 = summary.set_index("law").loc["D7"]
    assert d1["allocated_contribution_pp"] == pytest.approx(37.5)
    assert d7["allocated_contribution_pp"] == pytest.approx(37.5)
    assert d1["leave_one_out_loss_pp"] == pytest.approx(25.0)
    assert d7["leave_one_out_loss_pp"] == pytest.approx(25.0)
    assert summary["allocated_contribution_pp"].sum() == pytest.approx(75.0)


def test_l4_contribution_figure_has_no_title_grid_or_d2():
    experimental, exp_meta = contribution.summarize_cohort(
        _toy_frame(), cohort="experimental"
    )
    pu, pu_meta = contribution.summarize_cohort(
        _toy_frame(), cohort="pu_negative"
    )
    fig = contribution.draw_figure(
        pd.concat([experimental, pu], ignore_index=True),
        {"experimental": exp_meta, "pu_negative": pu_meta},
    )
    visible = "\n".join(
        item.get_text() for item in fig.findobj(Text) if item.get_visible()
    )

    assert fig._suptitle is None
    assert "D1" in visible and "D7" in visible and "D8" in visible
    assert "D2" not in visible
    assert "a" in visible and "b" in visible
    assert not any(line.get_visible() for ax in fig.axes for line in ax.get_xgridlines())
    assert not any(line.get_visible() for ax in fig.axes for line in ax.get_ygridlines())
    assert fig.axes[0].get_xlabel().startswith("Contribution")
    assert fig.axes[1].get_xlabel().startswith("Experimental")
    assert fig.axes[1].get_ylabel().startswith("Hard-negative")
    plt.close(fig)
