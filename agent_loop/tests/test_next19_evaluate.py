from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.next19_evaluate import (
    candidate_thresholds,
    decisions_from_score,
    elementa_metrics,
    evaluate_development,
    formula_scores,
    join_feature_labels,
    proportion,
    search_catalogue,
    wbm_metrics,
    RESULT_NAME,
)


def test_decisions_fail_open_unsupported_rows() -> None:
    decisions = decisions_from_score(
        pd.Series([0.1, np.nan, 0.3]),
        pd.Series([True, False, True]),
        threshold=0.2,
    )

    assert decisions.tolist() == ["KEEP", "ABSTAIN", "REJECT"]


def test_candidate_thresholds_use_only_supported_wbm_scores() -> None:
    thresholds = candidate_thresholds(
        pd.Series([0.0, 1.0, 2.0, 3.0, 100.0]),
        pd.Series([True, True, True, True, False]),
        rejection_fractions=(0.25, 0.5),
    )

    assert thresholds == (2.0, 3.0)


def test_formula_scores_use_fixed_monotone_two_term_catalogue() -> None:
    table = pd.DataFrame(
        {
            "x__vt_overload": [1.0, 2.0],
            "x__vt_reallocation": [0.25, 0.5],
            "x__vt_anion_mismatch_max": [0.1, 0.2],
        }
    )

    score = formula_scores(table, prefix="x", formula="overload_plus_reallocation")

    assert score.tolist() == pytest.approx([1.25, 2.5])


def test_elementa_metrics_count_complete_group_failure() -> None:
    joined = pd.DataFrame(
        {
            "rk": ["a", "a", "b", "b"],
            "dft_group_regret_ev_per_atom": [0.0, 0.3, 0.0, 0.4],
        }
    )
    decisions = pd.Series(["KEEP", "REJECT", "REJECT", "REJECT"])

    metrics = elementa_metrics(joined, decisions)

    assert metrics["dft_savings"]["estimate"] == pytest.approx(0.75)
    assert metrics["group_minimum_recall"]["estimate"] == pytest.approx(0.5)
    assert metrics["valuable_recall"]["estimate"] == pytest.approx(0.5)
    assert metrics["high_energy_rejection_recall"]["estimate"] == pytest.approx(1.0)
    assert metrics["reject_precision_above_minimum"]["estimate"] == pytest.approx(
        2.0 / 3.0
    )
    assert metrics["all_rejected_groups"] == 1


def test_wbm_metrics_report_stable_and_valuable_recall() -> None:
    joined = pd.DataFrame(
        {
            "formula_key": ["a", "a", "b", "b"],
            "stable": [True, False, True, False],
            "e_above_hull_mp2020_corrected_ppd_mp": [0.0, 0.3, 0.04, 0.5],
        }
    )
    decisions = pd.Series(["KEEP", "REJECT", "REJECT", "REJECT"])

    metrics = wbm_metrics(joined, decisions)

    assert metrics["stable_recall"]["estimate"] == pytest.approx(0.5)
    assert metrics["valuable_recall"]["estimate"] == pytest.approx(0.5)
    assert metrics["high_energy_rejection_recall"]["estimate"] == pytest.approx(1.0)
    assert metrics["reject_precision_unstable"]["estimate"] == pytest.approx(1.0)
    assert metrics["all_rejected_groups"] == 1


def test_wilson_proportion_handles_empty_denominator() -> None:
    metric = proportion(0, 0)

    assert metric == {
        "numerator": 0,
        "denominator": 0,
        "estimate": None,
        "wilson_ci95": [None, None],
    }


def test_cross_source_search_selects_only_candidate_passing_both_sources() -> None:
    groups = 400
    material_ids = [f"m{index:04d}_{kind}" for index in range(groups) for kind in (0, 1)]
    group_ids = [f"g{index:04d}" for index in range(groups) for _kind in (0, 1)]
    score = np.tile([0.0, 1.0], groups)
    common = {
        "material_id": material_ids,
        "rk": group_ids,
        "x__supported": [True] * (2 * groups),
        "x__vt_overload": score,
        "x__vt_reallocation": score,
        "x__vt_anion_mismatch_max": score,
    }
    wbm = pd.DataFrame(
        {
            **common,
            "formula_key": group_ids,
            "stable": np.tile([True, False], groups),
            "e_above_hull_mp2020_corrected_ppd_mp": np.tile([0.0, 0.3], groups),
            "pauling_p2_p5_decision": np.tile(["KEEP", "REJECT"], groups),
        }
    )
    elementa = pd.DataFrame(
        {
            **common,
            "dft_group_regret_ev_per_atom": np.tile([0.0, 0.3], groups),
            "pauling_p2_p5_decision": np.tile(["KEEP", "REJECT"], groups),
        }
    )

    result = search_catalogue(
        wbm,
        elementa,
        prefixes=("x",),
        formulas=("overload",),
        rejection_fractions=(0.4,),
    )

    assert result["development_promotion"] is True
    assert result["selected_candidate"] == {
        "prefix": "x",
        "formula": "overload",
        "threshold": 1.0,
    }
    assert result["selected_metrics"]["wbm"]["stable_recall"]["estimate"] == 1.0
    assert result["selected_metrics"]["elementa"]["group_minimum_recall"][
        "estimate"
    ] == 1.0


def test_label_join_requires_exact_one_to_one_identifier_coverage() -> None:
    features = pd.DataFrame(
        {"material_id": ["b", "a"], "x__supported": [True, True]}
    )
    labels = pd.DataFrame({"material_id": ["a", "b"], "stable": [True, False]})

    joined = join_feature_labels(features, labels, source="unit")

    assert joined["material_id"].tolist() == ["a", "b"]
    with pytest.raises(ValueError, match="identifier coverage mismatch"):
        join_feature_labels(
            features,
            labels.iloc[:1].copy(),
            source="unit",
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_development_evaluator_separates_aggregate_and_private_outputs(
    tmp_path: Path,
) -> None:
    groups = 400
    material_ids = [f"m{index:04d}_{kind}" for index in range(groups) for kind in (0, 1)]
    group_ids = [f"g{index:04d}" for index in range(groups) for _kind in (0, 1)]
    score = np.tile([0.0, 1.0], groups)
    features = pd.DataFrame(
        {
            "material_id": material_ids,
            "rk": group_ids,
            "x__supported": [True] * (2 * groups),
            "x__vt_overload": score,
            "x__vt_reallocation": score,
            "x__vt_anion_mismatch_max": score,
        }
    )
    wbm_features = tmp_path / "wbm_features.parquet"
    elementa_features = tmp_path / "elementa_features.parquet"
    features.to_parquet(wbm_features, index=False)
    features.to_parquet(elementa_features, index=False)
    wbm_manifest = tmp_path / "wbm_manifest.json"
    elementa_manifest = tmp_path / "elementa_manifest.json"
    for manifest, feature_path in (
        (wbm_manifest, wbm_features),
        (elementa_manifest, elementa_features),
    ):
        manifest.write_text(
            json.dumps(
                {
                    "endpoint_fields_read": False,
                    "model_or_proxy_potential_used": False,
                    "outputs_sha256": {
                        feature_path.name: _sha256(feature_path)
                    },
                }
            )
            + "\n"
        )
    wbm_labels = tmp_path / "wbm_labels.parquet"
    pd.DataFrame(
        {
            "material_id": material_ids,
            "formula_key": group_ids,
            "stable": np.tile([True, False], groups),
            "e_above_hull_mp2020_corrected_ppd_mp": np.tile([0.0, 0.3], groups),
            "pauling_p2_p5_decision": np.tile(["KEEP", "REJECT"], groups),
        }
    ).to_parquet(wbm_labels, index=False)
    elementa_labels = tmp_path / "elementa_labels.parquet"
    pd.DataFrame(
        {
            "material_id": material_ids,
            "dft_group_regret_ev_per_atom": np.tile([0.0, 0.3], groups),
            "pauling_p2_p5_decision": np.tile(["KEEP", "REJECT"], groups),
        }
    ).to_parquet(elementa_labels, index=False)
    aggregate = tmp_path / "aggregate"
    private = tmp_path / "private"

    evaluate_development(
        wbm_features_path=wbm_features,
        wbm_feature_manifest_path=wbm_manifest,
        wbm_labels_path=wbm_labels,
        elementa_features_path=elementa_features,
        elementa_feature_manifest_path=elementa_manifest,
        elementa_labels_path=elementa_labels,
        aggregate_output_dir=aggregate,
        private_output_dir=private,
        formulas=("overload",),
        rejection_fractions=(0.4,),
    )

    payload = json.loads((aggregate / RESULT_NAME).read_text())
    assert payload["development_promotion"] is True
    assert "m0000_0" not in (aggregate / RESULT_NAME).read_text()
    assert (private / "joined_wbm.parquet").is_file()
    assert (private / "joined_elementa.parquet").is_file()
    with pytest.raises(FileExistsError):
        evaluate_development(
            wbm_features_path=wbm_features,
            wbm_feature_manifest_path=wbm_manifest,
            wbm_labels_path=wbm_labels,
            elementa_features_path=elementa_features,
            elementa_feature_manifest_path=elementa_manifest,
            elementa_labels_path=elementa_labels,
            aggregate_output_dir=aggregate,
            private_output_dir=private,
            formulas=("overload",),
            rejection_fractions=(0.4,),
        )
