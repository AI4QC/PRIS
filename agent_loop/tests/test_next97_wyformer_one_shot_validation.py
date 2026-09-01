from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.next87_scigen_sparse_law_search import assign_group_folds
from src.next97_wyformer_one_shot_validation import (
    BROAD_THRESHOLD,
    SAFE_THRESHOLD,
    evaluate_dual_partition,
    run_wyformer_one_shot_validation,
)


def _one_formula_per_fold() -> dict[int, str]:
    result: dict[int, str] = {}
    index = 0
    while len(result) < 5:
        formula = f"A{index + 1}B"
        fold = int(assign_group_folds(np.array([formula], dtype=object))[0])
        result.setdefault(fold, formula)
        index += 1
    return result


def _passing_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    formulas = _one_formula_per_fold()
    prediction_rows: list[dict[str, object]] = []
    endpoint_rows: list[dict[str, object]] = []
    material_index = 0
    for fold in range(5):
        formula = formulas[fold]
        for stratum, score in (("protected", 0.0), ("severe", 4.0)):
            for local_index in range(30):
                material_id = f"m{material_index}"
                material_index += 1
                if stratum == "protected":
                    pauling = "KEEP" if local_index < 15 else "ABSTAIN"
                else:
                    pauling = (
                        "REJECT"
                        if local_index == 0
                        else "KEEP"
                        if local_index < 15
                        else "ABSTAIN"
                    )
                prediction_rows.append(
                    {
                        "material_id": material_id,
                        "reduced_formula": formula,
                        "crystal_system": (
                            "cubic",
                            "hexagonal",
                            "monoclinic",
                            "orthorhombic",
                            "tetragonal",
                        )[local_index % 5],
                        "pauling_p2_p5_decision": pauling,
                        "score": score,
                        "supported": True,
                        "safe_decision": "REJECT" if score >= SAFE_THRESHOLD else "KEEP",
                        "broad_decision": "REJECT" if score >= BROAD_THRESHOLD else "KEEP",
                    }
                )
                endpoint_rows.append(
                    {"material_id": material_id, "endpoint_stratum": stratum}
                )
    return pd.DataFrame(prediction_rows), pd.DataFrame(endpoint_rows)


def test_runner_accepts_only_frozen_candidate_and_validation_endpoint() -> None:
    parameters = inspect.signature(run_wyformer_one_shot_validation).parameters
    assert set(parameters) >= {"frozen_dir", "validation_endpoint_dir", "output_dir"}
    assert "feature_dir" not in parameters
    assert not any("replication" in name for name in parameters)


def test_evaluator_requires_exact_frozen_inclusive_decisions() -> None:
    predictions, endpoints = _passing_frames()
    result = evaluate_dual_partition(predictions=predictions, endpoints=endpoints)
    assert result["passes_all_validation_gates"] is True
    assert result["safe_all_folds_pass"] is True
    assert result["broad_all_folds_dominate_pauling"] is True

    altered = predictions.copy()
    altered.loc[altered.index[-1], "broad_decision"] = "KEEP"
    with pytest.raises(ValueError, match="frozen dual decisions"):
        evaluate_dual_partition(predictions=altered, endpoints=endpoints)


def test_evaluator_rejects_identity_loss() -> None:
    predictions, endpoints = _passing_frames()
    with pytest.raises(ValueError, match="identity join"):
        evaluate_dual_partition(
            predictions=predictions.iloc[:-1].copy(), endpoints=endpoints
        )
