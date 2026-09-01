"""Contracts for the finite, development-only NEXT43 formula search."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hash_split_is_deterministic_disjoint_and_nonempty() -> None:
    from src.next43_finite_law_search import deterministic_split

    ids = [f"alex-{index:04d}" for index in range(200)]
    first = deterministic_split(ids)
    second = deterministic_split(ids)
    assert np.array_equal(first, second)
    assert set(np.unique(first)) == {"discovery", "validation"}
    assert 90 <= int((first == "discovery").sum()) <= 150


def test_additive_formula_is_explicit_and_missing_values_fail_open() -> None:
    from src.next43_finite_law_search import apply_formula

    table = pd.DataFrame({"a": [0.0, 3.0, np.nan], "b": [0.0, 2.0, 9.0]})
    formula = {
        "kind": "additive",
        "terms": [
            {"feature": "a", "direction": 1, "center": 0.0, "scale": 1.0, "weight": 1.0},
            {"feature": "b", "direction": 1, "center": 0.0, "scale": 1.0, "weight": 0.5},
        ],
        "threshold": 3.5,
        "missing_policy": "KEEP",
    }
    score, supported, reject = apply_formula(table, formula)
    assert score[:2].tolist() == [0.0, 4.0]
    assert supported.tolist() == [True, True, False]
    assert reject.tolist() == [False, True, False]
    assert np.isnan(score[2])


def test_conjunctive_formula_requires_every_analytic_condition() -> None:
    from src.next43_finite_law_search import apply_formula

    table = pd.DataFrame({"a": [2.0, 2.0, 0.0], "b": [3.0, 0.0, 3.0]})
    formula = {
        "kind": "conjunctive",
        "terms": [
            {"feature": "a", "direction": 1, "center": 0.0, "scale": 1.0, "cutoff": 1.0},
            {"feature": "b", "direction": 1, "center": 0.0, "scale": 1.0, "cutoff": 2.0},
        ],
        "missing_policy": "KEEP",
    }
    score, supported, reject = apply_formula(table, formula)
    assert supported.all()
    assert reject.tolist() == [True, False, False]
    assert score.tolist() == [1.0, -2.0, -1.0]


def test_candidate_selection_never_uses_validation_labels() -> None:
    from src.next43_finite_law_search import search_development_candidate

    ids = np.asarray([f"alex-{index:04d}" for index in range(600)])
    split = np.asarray(["discovery"] * 360 + ["validation"] * 240)
    signal = np.linspace(-3.0, 3.0, len(ids))
    distractor = np.sin(np.arange(len(ids)))
    features = pd.DataFrame({"signal": signal, "distractor": distractor})
    discovery_changed = signal[:360] > 0.0
    endpoint_a = np.where(
        np.r_[discovery_changed, signal[360:] > 1.0], 0.3, 0.0
    )
    endpoint_b = endpoint_a.copy()
    endpoint_b[360:] = np.where(endpoint_b[360:] > 0.1, 0.0, 0.3)
    result_a = search_development_candidate(
        features=features,
        material_ids=ids,
        endpoint=endpoint_a,
        split=split,
        candidate_features=("signal", "distractor"),
    )
    result_b = search_development_candidate(
        features=features,
        material_ids=ids,
        endpoint=endpoint_b,
        split=split,
        candidate_features=("signal", "distractor"),
    )
    assert result_a["selected_formula"] == result_b["selected_formula"]
    assert result_a["discovery_metrics"] == result_b["discovery_metrics"]
    assert result_a["validation_metrics"] != result_b["validation_metrics"]


def test_search_formula_contains_no_endpoint_or_calculator_term() -> None:
    from src.next43_finite_law_search import search_development_candidate

    ids = np.asarray([f"row-{index}" for index in range(300)])
    split = np.asarray(["discovery"] * 180 + ["validation"] * 120)
    feature = np.linspace(0.0, 1.0, len(ids))
    endpoint = np.where(feature > 0.45, 0.3, 0.0)
    result = search_development_candidate(
        features=pd.DataFrame({"analytic_x": feature}),
        material_ids=ids,
        endpoint=endpoint,
        split=split,
        candidate_features=("analytic_x",),
    )
    formula = result["selected_formula"]
    assert formula["missing_policy"] == "KEEP"
    assert {term["feature"] for term in formula["terms"]} == {"analytic_x"}
    rendered = str(formula).lower()
    assert not any(
        token in rendered
        for token in ("endpoint", "energy", "force", "relax", "dft", "model")
    )


def test_full_search_publishes_auditable_development_outputs(tmp_path: Path) -> None:
    from src import next43_finite_law_search as module
    from src.next42_alexandria_evaluate import PROTOCOL as EVALUATION_PROTOCOL
    from src.next43_analytic_feature_bank import (
        CANDIDATE_FEATURE_NAMES,
        FEATURE_NAME,
        PROTOCOL as FEATURE_PROTOCOL,
    )

    ids = [f"alex-{index:04d}" for index in range(300)]
    analytic = np.asarray(
        [int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) / 2**32 for value in ids]
    )
    feature_root = tmp_path / "feature"
    feature_root.mkdir()
    feature_table = pd.DataFrame(
        {
            "material_id": ids,
            **{
                name: analytic if name == CANDIDATE_FEATURE_NAMES[0] else np.nan
                for name in CANDIDATE_FEATURE_NAMES
            },
        }
    )
    feature_path = feature_root / FEATURE_NAME
    feature_table.to_parquet(feature_path, index=False)
    feature_manifest = feature_root / "MANIFEST.json"
    feature_manifest.write_text(
        json.dumps(
            {
                "protocol": FEATURE_PROTOCOL,
                "labels_opened": False,
                "endpoint_fields_read": False,
                "dft_values_used": False,
                "mlip_or_model_potential_used": False,
                "outputs_sha256": {FEATURE_NAME: _sha(feature_path)},
            }
        ),
        encoding="utf-8",
    )

    evaluation_root = tmp_path / "evaluation"
    evaluation_root.mkdir()
    endpoint = np.where(analytic > 0.72, 0.3, 0.0)
    evaluation = pd.DataFrame(
        {
            "material_id": ids,
            "source_family": "test",
            "natoms": 2,
            "next23_supported": True,
            "next23_reject": False,
            "pauling_p2_p5_decision": "ABSTAIN",
            "force_converged": True,
            "primary_evaluation_supported": True,
            "site_stats_fingerprint_init_final_norm_diff": endpoint,
        }
    )
    evaluation_path = evaluation_root / module.NEXT42_JOINED_NAME
    evaluation.to_parquet(evaluation_path, index=False)
    evaluation_manifest = evaluation_root / "MANIFEST.json"
    evaluation_manifest.write_text(
        json.dumps(
            {
                "protocol": EVALUATION_PROTOCOL,
                "production_protocol_eligible": True,
                "later_geometry_opened_after_prediction_freeze": True,
                "evaluation_only_dft_energy_read": False,
                "outputs_sha256": {
                    module.NEXT42_JOINED_NAME: _sha(evaluation_path)
                },
            }
        ),
        encoding="utf-8",
    )
    target = tmp_path / "search"
    manifest = module.run_finite_search(
        feature_path=feature_path,
        feature_manifest_path=feature_manifest,
        evaluation_path=evaluation_path,
        evaluation_manifest_path=evaluation_manifest,
        output_dir=target,
    )
    assert manifest["development_labels_opened"] is True
    assert manifest["law_execution_dft_values_read"] is False
    assert manifest["validation_labels_used_for_selection"] is False
    formula = json.loads((target / module.FORMULA_NAME).read_text())
    assert formula["formula"]["missing_policy"] == "KEEP"
    for name in (module.FORMULA_NAME, module.SEARCH_NAME, module.PREDICTION_NAME):
        assert manifest["outputs_sha256"][name] == _sha(target / name)
    with pytest.raises(FileExistsError):
        module.run_finite_search(
            feature_path=feature_path,
            feature_manifest_path=feature_manifest,
            evaluation_path=evaluation_path,
            evaluation_manifest_path=evaluation_manifest,
            output_dir=target,
        )
