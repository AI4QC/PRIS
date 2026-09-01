from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from src.next87_scigen_sparse_law_search import _pauling_baseline
from src.next98_cross_source_discovery_search import (
    BROAD_MIN_PRECISION_LOWER,
    build_source_fold_cells,
    diagnose_safe_threshold_feasibility,
    run_cross_source_discovery_search,
    select_broad_threshold_across_cells,
    select_safe_threshold_across_cells,
)


def _cross_source_arrays() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    endpoint: list[float] = []
    score: list[float] = []
    material_index = 0
    for source in ("scigen", "wyformer"):
        for fold in range(5):
            for local_index in range(60):
                rows.append(
                    {
                        "material_id": f"m{material_index}",
                        "source_dataset": source,
                        "fold": fold,
                        "pauling_p2_p5_decision": (
                            "REJECT"
                            if local_index < 25
                            else "KEEP"
                            if local_index < 30
                            else "ABSTAIN"
                        ),
                    }
                )
                material_index += 1
                endpoint.append(1.0)
                score.append(1.0 if local_index < 20 else 0.0)
            for local_index in range(60):
                rows.append(
                    {
                        "material_id": f"m{material_index}",
                        "source_dataset": source,
                        "fold": fold,
                        "pauling_p2_p5_decision": (
                            "REJECT"
                            if local_index == 0
                            else "KEEP"
                            if local_index < 30
                            else "ABSTAIN"
                        ),
                    }
                )
                material_index += 1
                endpoint.append(2.0)
                score.append(4.0 if local_index < 40 else 1.0)
    frame = pd.DataFrame(rows)
    endpoint_array = np.asarray(endpoint, dtype=float)
    score_array = np.asarray(score, dtype=float)
    cells = build_source_fold_cells(
        source=frame["source_dataset"].to_numpy(),
        folds=frame["fold"].to_numpy(),
    )
    return frame, endpoint_array, score_array, cells


def test_runner_reads_discovery_only_and_has_no_validation_or_replication_input() -> None:
    parameters = inspect.signature(run_cross_source_discovery_search).parameters
    assert {"scigen_feature_dir", "wyformer_feature_dir"} <= set(parameters)
    assert {"scigen_discovery_endpoint_dir", "wyformer_discovery_endpoint_dir"} <= set(
        parameters
    )
    assert not any("validation" in name or "replication" in name for name in parameters)


def test_two_threshold_selector_enforces_all_source_fold_cells() -> None:
    frame, endpoint, score, cells = _cross_source_arrays()
    supported = np.ones(len(frame), dtype=bool)
    safe = select_safe_threshold_across_cells(
        score=score,
        supported=supported,
        endpoint=endpoint,
        cells=cells,
    )
    assert safe is not None
    assert safe["threshold"] == 4.0
    assert safe["passes_all_cells"] is True

    baselines = {
        str(cell["cell_id"]): _pauling_baseline(frame.loc[cell["mask"]], endpoint[cell["mask"]])
        for cell in cells
    }
    broad = select_broad_threshold_across_cells(
        score=score,
        supported=supported,
        endpoint=endpoint,
        cells=cells,
        pauling_by_cell=baselines,
        safe_threshold=float(safe["threshold"]),
        broad_min_precision_lower=BROAD_MIN_PRECISION_LOWER,
    )
    assert broad is not None
    assert broad["threshold"] == 1.0
    assert broad["passes_all_cells"] is True


def test_safe_selector_fails_if_one_formula_fold_has_no_savings() -> None:
    frame, endpoint, score, cells = _cross_source_arrays()
    supported = np.ones(len(frame), dtype=bool)
    target = (frame["source_dataset"] == "wyformer") & (frame["fold"] == 4)
    score[target.to_numpy()] = 0.0
    assert (
        select_safe_threshold_across_cells(
            score=score,
            supported=supported,
            endpoint=endpoint,
            cells=cells,
        )
        is None
    )
    diagnostic = diagnose_safe_threshold_feasibility(
        score=score,
        supported=supported,
        endpoint=endpoint,
        cells=cells,
    )
    assert diagnostic["passing_cells"] < len(cells)
    assert "wyformer:fold4" in diagnostic["failing_cell_ids"]
