from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from experiments.next538_pbaaa_label_blind_probe import (
    PROBE_PER_ROLE,
    evaluate_probe_gates,
    numeric_novelty_audit,
    select_probe_ids,
    run_label_blind_probe,
)


def _metadata(rows_per_role: int = 40) -> pd.DataFrame:
    rows = []
    for role in ("discovery", "internal_validation", "internal_replication"):
        rows.extend(
            {
                "material_id": f"{role}-{index:03d}",
                "partition_role": role,
            }
            for index in range(rows_per_role)
        )
    return pd.DataFrame(rows)


def test_probe_selection_is_role_balanced_deterministic_and_order_invariant() -> None:
    metadata = _metadata()
    first = select_probe_ids(metadata)
    second = select_probe_ids(metadata.sample(frac=1.0, random_state=7))
    assert first == second
    assert len(first) == 3 * PROBE_PER_ROLE
    selected = metadata[metadata["material_id"].isin(first)]
    assert selected.groupby("partition_role").size().to_dict() == {
        "discovery": PROBE_PER_ROLE,
        "internal_replication": PROBE_PER_ROLE,
        "internal_validation": PROBE_PER_ROLE,
    }


def test_probe_gates_pass_exact_boundary_and_fail_on_any_role() -> None:
    frame = _metadata(PROBE_PER_ROLE)
    frame["supported"] = True
    frame["value"] = np.arange(len(frame), dtype=float) / len(frame)
    frame["runtime_seconds"] = 0.1
    invariance = {"tested": 8, "maximum_absolute_error": 1.0e-6}
    novelty = {"adequate_comparisons": 4, "maximum_absolute_spearman": 0.94}
    passed = evaluate_probe_gates(frame, invariance=invariance, novelty=novelty)
    assert passed["passes"] is True

    failed = frame.copy()
    mask = failed["partition_role"].eq("internal_validation")
    failed.loc[failed[mask].index[:5], "supported"] = False
    failed.loc[~failed["supported"], "value"] = np.nan
    stopped = evaluate_probe_gates(failed, invariance=invariance, novelty=novelty)
    assert stopped["passes"] is False
    assert stopped["roles"]["internal_validation"]["support_passes"] is False


def test_novelty_audit_detects_a_duplicate_and_ignores_inadequate_columns() -> None:
    candidate = np.linspace(0.0, 1.0, 60)
    prior = pd.DataFrame(
        {
            "duplicate": candidate,
            "different": np.sin(17.0 * candidate),
            "too_sparse": [1.0] * 20 + [np.nan] * 40,
            "constant": 1.0,
        }
    )
    audit = numeric_novelty_audit(candidate, prior, minimum_joint=40)
    assert audit["adequate_comparisons"] == 2
    assert audit["maximum_absolute_spearman"] == pytest.approx(1.0)
    assert audit["passes"] is False


def test_formal_probe_interface_has_no_endpoint_or_label_argument() -> None:
    parameters = set(inspect.signature(run_label_blind_probe).parameters)
    assert parameters == {
        "next54_dir",
        "next77_dir",
        "next533_dir",
        "design_path",
        "output_dir",
        "require_formal_inputs",
    }
