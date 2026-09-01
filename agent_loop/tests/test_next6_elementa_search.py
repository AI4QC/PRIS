import numpy as np
import pandas as pd
import pytest

from src.next6_elementa_search import (
    FormulaSpec,
    TermSpec,
    candidate_catalog,
    calibrate_and_evaluate,
    choose_formula,
    fit_normalization,
    prepare_search_features,
    run_prepared_search,
    score_formula,
)


pytestmark = pytest.mark.filterwarnings("error")


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sid": ["a", "b", "c", "d", "e", "f"],
            "rk": ["x", "x", "x", "y", "y", "y"],
            "strict_x0_ok": [True, True, True, True, True, False],
            "support_geom": [True, True, True, True, True, True],
            "packing_bad": [3.0, 2.0, 1.0, 1.0, 3.0, 9.0],
            "ewald_bad": [2.0, 2.0, 0.0, 1.0, 4.0, 8.0],
        }
    )


def _relative_spec() -> FormulaSpec:
    return FormulaSpec(
        name="relative",
        track="cohort_relative",
        role="candidate",
        terms=(
            TermSpec("packing_bad", 1, 1, "geometry", "support_geom"),
            TermSpec("ewald_bad", 1, 1, "electrostatic", "support_geom"),
        ),
    )


def test_relative_formula_is_row_shuffle_invariant_and_averages_ties():
    data = _features()
    first = score_formula(data, _relative_spec(), normalization={}).set_index("sid")
    shuffled = score_formula(
        data.sample(frac=1, random_state=7), _relative_spec(), normalization={}
    ).set_index("sid")

    pd.testing.assert_series_equal(
        first.sort_index().score, shuffled.sort_index().score, check_names=False
    )
    assert first.loc["a", "score"] == pytest.approx(0.875)
    assert first.loc["b", "score"] == pytest.approx(0.625)
    assert not bool(first.loc["f", "decision_support"])


def test_absolute_normalization_uses_features_only_and_missing_support_abstains():
    spec = FormulaSpec(
        name="absolute",
        track="absolute_law",
        role="candidate",
        terms=(TermSpec("packing_bad", 1, 2, "geometry", "support_geom"),),
    )
    data = _features()
    normalization = fit_normalization(data, spec)
    assert normalization["packing_bad"]["center"] == 2.0
    assert normalization["packing_bad"]["scale"] == 2.0

    scored = score_formula(data, spec, normalization).set_index("sid")
    assert scored.loc["a", "score"] == pytest.approx(0.5)
    assert np.isnan(scored.loc["f", "score"])
    assert not bool(scored.loc["f", "decision_support"])


def test_catalog_is_finite_sparse_positive_and_keeps_baselines_distinct():
    catalog = candidate_catalog()
    assert len(catalog) == len({spec.name for spec in catalog})
    assert any(spec.name == "pauling_original_equal" and spec.role == "baseline" for spec in catalog)
    assert any(spec.name == "relative_ewald_packing_p2" and spec.role == "candidate" for spec in catalog)
    assert any(spec.name == "p9_robust_q95" for spec in catalog)
    assert all(term.weight > 0 for spec in catalog for term in spec.terms)
    assert all(len({term.block for term in spec.terms}) == len(spec.terms) for spec in catalog if spec.role == "candidate")


def test_formula_choice_enforces_safety_before_savings_and_is_deterministic():
    frontier = pd.DataFrame(
        {
            "name": ["unsafe", "complex", "simple", "baseline"],
            "role": ["candidate", "candidate", "candidate", "baseline"],
            "valuable_group_retention_lower": [0.99, 0.96, 0.96, 1.0],
            "exact_min_retention_lower": [0.90, 0.96, 0.96, 1.0],
            "regret_p95": [0.0, 0.01, 0.01, 0.0],
            "dft_savings": [0.9, 0.2, 0.2, 0.0],
            "complexity": [1, 3, 2, 1],
        }
    )
    chosen = choose_formula(frontier)
    assert chosen["name"] == "simple"


def test_group_min_formula_choice_uses_near_min_gate_not_all_valuable_gate():
    frontier = pd.DataFrame(
        {
            "name": ["safe", "unsafe"],
            "role": ["candidate", "candidate"],
            "valuable_group_retention_lower": [0.50, 0.99],
            "exact_min_retention_lower": [0.96, 0.90],
            "near_min_retention_lower": [0.97, 0.99],
            "regret_p95": [0.01, 0.0],
            "all_rejected_groups": [0, 0],
            "dft_savings": [0.30, 0.90],
            "complexity": [2, 1],
        }
    )
    assert choose_formula(frontier, gate="group_min")["name"] == "safe"
    with pytest.raises(ValueError, match="gate"):
        choose_formula(frontier, gate="unknown")


def test_feature_preparation_joins_by_sid_and_keeps_ambiguous_charge_only_for_robust_p9():
    scales = ("080", "090", "100", "110", "120")
    base = pd.DataFrame(
        {
            "sid": ["a", "b", "c"],
            "rk": ["x", "y", "z"],
            "material": ["A_01", "B_01", "C_01"],
            "geom_feature_ok": [True, True, True],
            "geom_min_pair_ratio": [0.5, 1.0, 1.0],
            "geom_packing_fraction": [1.0, 1.0, 1.0],
            "pauling_feature_ok": [True, True, True],
            "p2_mean_dev": [0.1, 0.2, 0.3],
            "p3_frac_edge_face": [0.0, 0.0, 0.0],
            "p4_violate": [0.0, 0.0, 0.0],
            "p5_n_distinct": [1.0, 2.0, 1.0],
            "shannon_feature_ok": [True, True, True],
            "bl_min": [1.0, 1.0, 1.0],
            "ewald_feature_ok": [True, True, True],
            "ewald_per_atom": [-2.0, -1.0, 0.0],
            "econ_mean": [4.0, 4.0, 4.0],
            "dist_rsd": [0.1, 0.1, 0.1],
            "bvs_feature_ok": [True, True, True],
            "gii": [0.1, 0.2, 0.3],
            "bv_param_cov": [1.0, 1.0, 1.0],
            "e_per_atom": [-99.0, -99.0, -99.0],
        }
    )
    for scale in scales:
        base[f"geom_repulsion_p2_l{scale}"] = [float(int(scale) / 100), 0.0, 0.0]
        base[f"geom_packing_l{scale}"] = [1.0, 1.0, 1.0]
    p9 = pd.DataFrame(
        {
            "sid": ["c", "b", "a"],
            "rk": ["z", "y", "x"],
            "material": ["C_01", "B_01", "A_01"],
            "strict_x0_ok": [False, True, True],
            "p9c_feature_ok": [False, True, True],
            "p9r_feature_ok": [False, True, True],
            "p9r_assignment_count": [0, 2, 1],
            "p9c_bond_mismatch_q95": [np.nan, 0.2, 0.1],
            "p9r_bond_mismatch_q95_min": [np.nan, 0.15, 0.1],
        }
    )

    got = prepare_search_features(base, p9).set_index("sid")
    assert "e_per_atom" not in got.columns
    assert got.loc["a", "min_pair_overlap"] == 1.0
    assert got.loc["a", "born_wbm_envelope"] == pytest.approx(0.8)
    assert bool(got.loc["a", "support_ewald"])
    assert not bool(got.loc["b", "support_ewald"])
    assert bool(got.loc["b", "support_p9r"])
    assert not bool(got.loc["c", "support_geom"])


def test_all_formulas_use_the_same_group_calibration_and_keyed_label_join():
    rows = []
    labels = []
    for group in range(20):
        for candidate, score, energy in (("good", 0.0, -2.0), ("bad", 1.0, -1.7)):
            sid = f"g{group}-{candidate}"
            rows.append(
                {
                    "sid": sid,
                    "rk": f"g{group}",
                    "strict_x0_ok": True,
                    "support_geom": True,
                    "packing_bad": score,
                }
            )
            labels.append({"sid": sid, "rk": f"g{group}", "e_per_atom": energy})
    features = pd.DataFrame(rows)
    endpoint = pd.DataFrame(labels).sample(frac=1, random_state=4)
    spec = FormulaSpec(
        name="law",
        track="absolute_law",
        role="candidate",
        terms=(TermSpec("packing_bad", 1, 1, "geometry", "support_geom"),),
    )
    normalization = {"packing_bad": {"center": 0.0, "scale": 1.0, "n": 40}}
    result, predictions = calibrate_and_evaluate(
        spec,
        normalization,
        features,
        endpoint,
        features,
        endpoint,
        alpha=0.2,
    )

    assert result["threshold"] == 0.0
    assert result["dft_savings"] == 0.5
    assert result["exact_min_retention"] == 1.0
    assert result["valuable_all_retained_group_rate"] == 1.0
    assert set(predictions.loc[predictions.decision == "REJECT", "sid"]) == {
        f"g{group}-bad" for group in range(20)
    }


def test_near_min_group_risk_can_reject_other_valuable_candidates():
    features = pd.DataFrame(
        {
            "sid": ["best", "nearby"],
            "rk": ["g", "g"],
            "strict_x0_ok": [True, True],
            "support_geom": [True, True],
            "packing_bad": [0.0, 1.0],
        }
    )
    labels = pd.DataFrame(
        {"sid": ["best", "nearby"], "rk": ["g", "g"], "e_per_atom": [-2.0, -1.96]}
    )
    spec = FormulaSpec(
        name="law",
        track="absolute_law",
        role="candidate",
        terms=(TermSpec("packing_bad", 1, 1, "geometry", "support_geom"),),
    )
    norm = {"packing_bad": {"center": 0.0, "scale": 1.0, "n": 2}}
    valuable_metrics, _ = calibrate_and_evaluate(
        spec, norm, features, labels, features, labels, alpha=0.5
    )
    group_metrics, decisions = calibrate_and_evaluate(
        spec,
        norm,
        features,
        labels,
        features,
        labels,
        alpha=0.5,
        protected_column="near_min",
        within_group="min",
    )
    assert valuable_metrics["dft_savings"] == 0.0
    assert group_metrics["dft_savings"] == 0.5
    assert group_metrics["risk_protected_column"] == "near_min"
    assert decisions.decision.tolist() == ["KEEP", "REJECT"]


def test_prepared_search_writes_frozen_artifacts_and_refuses_second_test_opening(tmp_path):
    rows = []
    labels = []
    for group in range(300):
        for candidate, score, energy in (("good", 0.0, -2.0), ("bad", 1.0, -1.7)):
            sid = f"g{group}-{candidate}"
            rows.append(
                {
                    "sid": sid,
                    "rk": f"g{group}",
                    "strict_x0_ok": True,
                    "support_geom": True,
                    "packing_bad": score,
                }
            )
            labels.append({"sid": sid, "rk": f"g{group}", "e_per_atom": energy})
    features = pd.DataFrame(rows)
    endpoint = pd.DataFrame(labels)
    spec = FormulaSpec(
        name="law",
        track="absolute_law",
        role="candidate",
        terms=(TermSpec("packing_bad", 1, 1, "geometry", "support_geom"),),
    )

    result = run_prepared_search(
        features,
        endpoint,
        tmp_path,
        specs=[spec],
        alpha_values=(0.2,),
    )
    assert result["selected_formula"] == "law"
    assert (tmp_path / "formula_selection_frontier.parquet").is_file()
    assert (tmp_path / "frozen_rules.json").is_file()
    assert (tmp_path / "test_metrics.parquet").is_file()
    assert (tmp_path / "TEST_OPENING.json").is_file()

    with pytest.raises(RuntimeError, match="already opened"):
        run_prepared_search(
            features,
            endpoint,
            tmp_path,
            specs=[spec],
            alpha_values=(0.2,),
        )
