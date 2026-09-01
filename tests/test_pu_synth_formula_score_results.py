from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.pu_synthesizability_20260821.analyze_formula_score_results import (
    _full_pool_formula_input_columns,
    analyze_binary_files,
    analyze_full_pool_files,
    build_binary_metrics,
    build_full_pool_correlations,
    build_pris_associations,
    build_score_distributions,
    join_full_pool_inputs,
    prepare_formula_scores,
)
from experiments.pu_synthesizability_20260821.formula_scores import (
    FrozenFormula,
    STABILITY_FEATURES,
    SYNTHESIS_FEATURES,
)


def _formula(name: str, features: tuple[str, ...]) -> FrozenFormula:
    return FrozenFormula(
        name=name,
        features=features,
        beta=np.ones(len(features), dtype=np.float64),
        impute_median={feature: 0.0 for feature in features},
        mu={feature: 0.0 for feature in features},
        sd={feature: 1.0 for feature in features},
        source_path=Path(f"{name}.json"),
    )


def _raw_formula_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "formula_syn_madz_mean": [1.0, 1.0],
            "formula_syn_wyckoff_econ_001": [2.0, 2.0],
            "formula_syn_bv_rel_mean": [3.0, 3.0],
            "formula_syn_vol_per_atom": [4.0, np.nan],
            "formula_syn_poly_deg_max": [5.0, 5.0],
            "formula_syn_frac_isolated": [6.0, 6.0],
            "formula_syn_valence_route": ["balance", "no_anion"],
            "formula_syn_historical_size_domain": [True, False],
            "formula_stab_econ_max": [1.0, 1.0],
            "formula_stab_gii": [2.0, 2.0],
            "formula_stab_cn_cat_max": [3.0, 3.0],
            "formula_stab_p2_max_dev": [4.0, 4.0],
            "formula_stab_wyckoff_econ_01": [5.0, 5.0],
            "formula_stab_econ_min": [6.0, np.nan],
            "formula_stab_valence_route": ["guess_oxi", "balance"],
            "formula_stab_historical_size_domain": [False, True],
        }
    )
    return frame


def test_prepare_formula_scores_uses_frozen_terms_and_removes_only_syn_d7_d8():
    got = prepare_formula_scores(
        _raw_formula_frame(),
        synthesis_formula=_formula("S_syn", SYNTHESIS_FEATURES),
        stability_formula=_formula("S_stab", STABILITY_FEATURES),
    )

    assert got.S_syn.tolist() == pytest.approx([21.0, 17.0])
    assert got.S_syn_no_D7_D8.tolist() == pytest.approx([16.0, 12.0])
    assert got.S_syn_no_D7_D8_madz.tolist() == pytest.approx([15.0, 11.0])
    assert got.S_stab.tolist() == pytest.approx([21.0, 15.0])
    assert got.S_syn_all_observed.tolist() == [True, False]
    assert got.S_stab_all_observed.tolist() == [True, False]
    assert got.S_syn_n_observed.tolist() == [6, 5]
    assert got.S_stab_n_observed.tolist() == [6, 5]


def test_prepare_formula_scores_supports_legacy_pilot_feature_names():
    frame = _raw_formula_frame().rename(
        columns={
            "formula_syn_madz_mean": "feature_madz_mean",
            "formula_syn_wyckoff_econ_001": "feature_wyckoff_econ",
            "formula_syn_bv_rel_mean": "feature_bv_rel_mean",
            "formula_syn_vol_per_atom": "feature_vol_per_atom",
            "formula_syn_poly_deg_max": "feature_poly_deg_max",
            "formula_syn_frac_isolated": "feature_frac_isolated",
            "formula_stab_econ_max": "feature_econ_max",
            "formula_stab_gii": "feature_gii",
            "formula_stab_cn_cat_max": "feature_cn_cat_max",
            "formula_stab_p2_max_dev": "feature_p2_max_dev",
            "formula_stab_wyckoff_econ_01": "wyckoff_econ_symprec_0p1",
            "formula_stab_econ_min": "feature_econ_min",
        }
    )

    got = prepare_formula_scores(
        frame,
        synthesis_formula=_formula("S_syn", SYNTHESIS_FEATURES),
        stability_formula=_formula("S_stab", STABILITY_FEATURES),
    )

    assert got.S_syn.tolist() == pytest.approx([21.0, 17.0])
    assert got.S_stab.tolist() == pytest.approx([21.0, 15.0])


def test_binary_metrics_use_midrank_ties_and_strict_95_percent_screening():
    frame = pd.DataFrame(
        {
            "cohort": ["experimental", "experimental", "pu_negative", "pu_negative"],
            "S_syn": [2.0, 2.0, 1.0, 2.0],
            "S_syn_no_D7_D8": [2.0, 2.0, 1.0, 2.0],
            "S_syn_no_D7_D8_madz": [2.0, 2.0, 1.0, 2.0],
            "S_stab": [2.0, 2.0, 1.0, 2.0],
            "S_syn_all_observed": [True] * 4,
            "S_stab_all_observed": [True] * 4,
            "formula_syn_historical_size_domain": [True] * 4,
            "formula_stab_historical_size_domain": [True] * 4,
            "charge_assignment_route": ["integer"] * 4,
            "provenance": ["lemat"] * 4,
        }
    )

    got = build_binary_metrics(frame, min_group_rows=1)
    row = got[
        (got.score_variant == "S_syn")
        & (got.analysis_scope == "raw")
        & (got.stratum_dimension == "overall")
    ].iloc[0]

    assert row.status == "ok"
    assert row.auroc == pytest.approx(0.75)
    assert row.cliffs_delta == pytest.approx(0.5)
    assert row.experimental_preserved_rate == pytest.approx(1.0)
    assert row.experimental_preserved_n == 2
    assert row.experimental_preserved_denominator == 2
    assert row.pu_screened_at_95pct_experimental == pytest.approx(0.5)
    assert row.pu_screened_n_at_95pct_experimental == 1
    assert row.pu_screened_denominator_at_95pct_experimental == 2
    assert row.screen_threshold == pytest.approx(2.0)


def test_outputs_include_raw_complete_case_charge_and_provenance_strata():
    frame = pd.DataFrame(
        {
            "cohort": ["experimental", "experimental", "pu_negative", "pu_negative"],
            "S_syn": [3.0, 2.0, 1.0, 0.0],
            "S_syn_no_D7_D8": [3.0, 2.0, 1.0, 0.0],
            "S_syn_no_D7_D8_madz": [3.0, 2.0, 1.0, 0.0],
            "S_stab": [4.0, 3.0, 2.0, 1.0],
            "S_syn_all_observed": [True, False, True, False],
            "S_stab_all_observed": [True, False, True, False],
            "formula_syn_historical_size_domain": [True, True, True, False],
            "formula_stab_historical_size_domain": [True, False, True, True],
            "charge_assignment_route": ["integer", "fractional", "integer", "fractional"],
            "provenance": ["lemat", "elementa", "lemat", "elementa"],
        }
    )

    distributions = build_score_distributions(frame)
    metrics = build_binary_metrics(frame, min_group_rows=1)

    assert set(distributions.analysis_scope) == {
        "raw",
        "complete_case",
        "complete_case_historical_size_domain",
    }
    assert {"overall", "charge_route", "provenance"}.issubset(
        distributions.stratum_dimension
    )
    complete_syn = distributions[
        (distributions.score_variant == "S_syn")
        & (distributions.analysis_scope == "complete_case")
        & (distributions.stratum_dimension == "overall")
        & (distributions.cohort == "experimental")
    ].iloc[0]
    assert complete_syn.n == 1
    historical_syn = distributions[
        (distributions.score_variant == "S_syn")
        & (
            distributions.analysis_scope
            == "complete_case_historical_size_domain"
        )
        & (distributions.stratum_dimension == "overall")
        & (distributions.cohort == "experimental")
    ].iloc[0]
    assert historical_syn.n == 1
    assert "S_syn_no_D7_D8_madz" in set(distributions.score_variant)
    assert {"overall", "charge_route", "provenance"}.issubset(
        metrics.stratum_dimension
    )


def test_full_pool_join_and_spearman_are_one_to_one_and_stratified():
    formula = _raw_formula_frame().iloc[[0, 0, 0, 0]].reset_index(drop=True)
    formula["orig_index"] = [13, 10, 12, 11]
    # Make both frozen scores increase with orig_index after the one-to-one join.
    formula["formula_syn_vol_per_atom"] = [3.0, 0.0, 2.0, 1.0]
    formula["formula_stab_econ_max"] = [3.0, 0.0, 2.0, 1.0]
    pris = pd.DataFrame(
        {
            "orig_index": [10, 11, 12, 13],
            "charge_assignment_route": ["integer", "integer", "fractional", "fractional"],
            "source_split": ["lemat", "lemat", "elementa", "elementa"],
        }
    )
    clscores = pd.DataFrame(
        {
            "orig_index": [10, 11, 12, 13],
            "CLscore_A": [0.0, 1.0, 2.0, 3.0],
            "CLscore_B": [3.0, 2.0, 1.0, 0.0],
            "CLscore_jang": [0.0, 1.0, 2.0, 3.0],
        }
    )

    joined = join_full_pool_inputs(formula, pris, clscores, expected_rows=4)
    scored = prepare_formula_scores(
        joined,
        synthesis_formula=_formula("S_syn", SYNTHESIS_FEATURES),
        stability_formula=_formula("S_stab", STABILITY_FEATURES),
    )
    got = build_full_pool_correlations(scored, min_group_rows=2)

    a = got[
        (got.score_variant == "S_syn")
        & (got.clscore == "CLscore_A")
        & (got.analysis_scope == "raw")
        & (got.stratum_dimension == "overall")
    ].iloc[0]
    b = got[
        (got.score_variant == "S_syn")
        & (got.clscore == "CLscore_B")
        & (got.analysis_scope == "raw")
        & (got.stratum_dimension == "overall")
    ].iloc[0]
    assert joined.orig_index.tolist() == [10, 11, 12, 13]
    assert a.spearman_rho == pytest.approx(1.0)
    assert b.spearman_rho == pytest.approx(-1.0)
    assert {"overall", "charge_route", "provenance"}.issubset(
        got.stratum_dimension
    )


def test_pris_associations_exclude_no_verdict_from_binary_metrics():
    frame = pd.DataFrame(
        {
            "S_syn": [3.0, 2.0, 1.0, 2.0, 100.0, -100.0],
            "S_syn_no_D7_D8": [3.0, 2.0, 1.0, 2.0, 100.0, -100.0],
            "S_syn_no_D7_D8_madz": [3.0, 2.0, 1.0, 2.0, 100.0, -100.0],
            "S_stab": [3.0, 2.0, 1.0, 2.0, 100.0, -100.0],
            "S_syn_all_observed": [True] * 6,
            "S_stab_all_observed": [True] * 6,
            "formula_syn_historical_size_domain": [True] * 6,
            "formula_stab_historical_size_domain": [True] * 6,
            "D7_verdict": [
                "pass",
                "pass",
                "explicit_violation",
                "explicit_violation",
                "no_verdict",
                "no_verdict",
            ],
            "L4_verdict": [
                "pass",
                "pass",
                "explicit_violation",
                "explicit_violation",
                "no_verdict",
                "no_verdict",
            ],
            "provenance": ["lemat", "lemat", "elementa", "elementa", "lemat", "elementa"],
            "charge_route": ["integer"] * 6,
        }
    )

    got = build_pris_associations(frame, min_group_rows=1)
    row = got[
        (got.score_variant == "S_syn")
        & (got.pris_outcome == "D7")
        & (got.analysis_scope == "raw")
        & (got.stratum_dimension == "overall")
    ].iloc[0]

    assert set(got.stratum_dimension) == {"overall", "provenance"}
    assert row.n_total == 6
    assert row.n_pass == 2
    assert row.n_explicit_violation == 2
    assert row.n_no_verdict == 2
    assert row.n_binary == 4
    assert row.auroc_high_score_predicts_pass == pytest.approx(0.875)
    assert row.cliffs_delta == pytest.approx(0.75)
    assert row.pass_median == pytest.approx(2.5)
    assert row.explicit_violation_median == pytest.approx(1.5)
    assert len(got) == 4 * 3 * 3 * 2

    invalid = frame.copy()
    invalid.loc[0, "D7_verdict"] = "unknown"
    with pytest.raises(ValueError, match="D7_verdict.*three-state"):
        build_pris_associations(invalid, min_group_rows=1)


def test_full_pool_join_rejects_duplicate_or_missing_formula_rows():
    formula = pd.DataFrame({"orig_index": [1, 1], "S_syn": [0.0, 0.0]})
    pris = pd.DataFrame(
        {
            "orig_index": [1, 2],
            "charge_assignment_route": ["integer", "integer"],
            "source_split": ["lemat", "lemat"],
        }
    )
    clscores = pd.DataFrame(
        {
            "orig_index": [1, 2],
            "CLscore_A": [0.0, 1.0],
            "CLscore_B": [0.0, 1.0],
            "CLscore_jang": [0.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="formula.*orig_index.*unique"):
        join_full_pool_inputs(formula, pris, clscores, expected_rows=2)

    formula = pd.DataFrame({"orig_index": [1], "S_syn": [0.0]})
    with pytest.raises(ValueError, match="formula rows matched"):
        join_full_pool_inputs(formula, pris, clscores, expected_rows=2)


def test_full_pool_join_requires_identical_formula_and_pris_pool_keys():
    formula = pd.DataFrame({"orig_index": [1, 2, 3]})
    pris = pd.DataFrame({"orig_index": [1, 2, 4]})
    clscores = pd.DataFrame(
        {
            "orig_index": [1, 2],
            "CLscore_A": [0.1, 0.2],
            "CLscore_B": [0.1, 0.2],
            "CLscore_jang": [0.1, 0.2],
        }
    )

    with pytest.raises(ValueError, match="formula and old PRIS full pools differ"):
        join_full_pool_inputs(formula, pris, clscores, expected_rows=2)


def test_binary_file_analysis_deduplicates_cifs_and_writes_compact_outputs(tmp_path):
    experimental = _raw_formula_frame().iloc[[0, 0]].reset_index(drop=True)
    experimental["record_index"] = [0, 1]
    experimental["cif_sha256"] = ["a" * 64, "b" * 64]
    experimental["charge_assignment_route"] = ["integer", "fractional"]
    experimental["source"] = ["lemat", "elementa"]
    experimental["formula_syn_valence_route"] = ["balance", "no_anion"]
    experimental["formula_stab_valence_route"] = ["guess_oxi", "balance"]
    pu = _raw_formula_frame().iloc[[0, 0, 0]].reset_index(drop=True)
    pu["record_index"] = [0, 1, 2]
    pu["cif_sha256"] = ["c" * 64, "d" * 64, "c" * 64]
    pu["charge_assignment_route"] = ["integer", "fractional", "integer"]
    pu["source"] = ["lemat", "elementa", "lemat"]
    pu["formula_syn_valence_route"] = ["balance", "no_anion", "balance"]
    pu["formula_stab_valence_route"] = ["guess_oxi", "balance", "guess_oxi"]
    experimental_path = tmp_path / "experimental.parquet"
    pu_path = tmp_path / "pu.parquet"
    experimental.to_parquet(experimental_path, index=False)
    pu.to_parquet(pu_path, index=False)

    output = tmp_path / "binary-analysis"
    summary = analyze_binary_files(
        experimental_inputs=[experimental_path],
        pu_inputs=[pu_path],
        output_dir=output,
        synthesis_formula=_formula("S_syn", SYNTHESIS_FEATURES),
        stability_formula=_formula("S_stab", STABILITY_FEATURES),
        expected_experimental_rows=2,
        expected_pu_rows=2,
        min_group_rows=1,
    )

    assert summary["rows"]["experimental_unique_cif"] == 2
    assert summary["rows"]["pu_negative_unique_cif"] == 2
    assert summary["rows"]["pu_negative_duplicate_extra_rows"] == 1
    assert (output / "score_distributions.csv").is_file()
    assert (output / "binary_metrics.csv").is_file()
    assert (output / "formula_coverage.csv").is_file()
    assert (output / "result_summary.json").is_file()
    metrics = pd.read_csv(output / "binary_metrics.csv")
    assert {
        "S_syn",
        "S_syn_no_D7_D8",
        "S_syn_no_D7_D8_madz",
        "S_stab",
    } == set(
        metrics.score_variant
    )
    coverage = pd.read_csv(output / "formula_coverage.csv")
    balance = coverage[
        (coverage.formula == "S_syn")
        & (coverage.cohort == "pu_negative")
        & (coverage.coverage_dimension == "valence_route")
        & (coverage.coverage_value == "balance")
    ].iloc[0]
    assert balance.n == 1
    assert balance.denominator == 2
    assert balance.fraction == pytest.approx(0.5)
    fully_observed = coverage[
        (coverage.formula == "S_syn")
        & (coverage.cohort == "experimental")
        & (coverage.coverage_dimension == "n_observed")
        & (coverage.coverage_value.astype(str) == "6")
    ].iloc[0]
    assert fully_observed.n == 2
    assert fully_observed.denominator == 2


def test_full_pool_file_analysis_joins_three_tables_and_writes_correlations(tmp_path):
    formula = _raw_formula_frame().iloc[[0, 0, 0, 0]].reset_index(drop=True)
    formula["orig_index"] = [13, 10, 12, 11]
    formula["formula_syn_vol_per_atom"] = [3.0, 0.0, 2.0, 1.0]
    formula["formula_stab_econ_max"] = [3.0, 0.0, 2.0, 1.0]
    pris = pd.DataFrame(
        {
            "orig_index": [10, 11, 12, 13],
            "charge_assignment_route": ["integer", "integer", "fractional", "fractional"],
            "source_split": ["lemat", "lemat", "elementa", "elementa"],
            "D7_verdict": ["pass", "pass", "explicit_violation", "no_verdict"],
            "L4_verdict": ["pass", "explicit_violation", "explicit_violation", "no_verdict"],
        }
    )
    clscores = pd.DataFrame(
        {
            "orig_index": [10, 11, 12, 13],
            "CLscore_A": [0.0, 1.0, 2.0, 3.0],
            "CLscore_B": [3.0, 2.0, 1.0, 0.0],
            "CLscore_jang": [0.0, 1.0, 2.0, 3.0],
        }
    )
    formula_path = tmp_path / "formula.parquet"
    pris_path = tmp_path / "pris.parquet"
    clscore_path = tmp_path / "clscores.parquet"
    formula.to_parquet(formula_path, index=False)
    pris.to_parquet(pris_path, index=False)
    clscores.to_parquet(clscore_path, index=False)

    output = tmp_path / "full-analysis"
    summary = analyze_full_pool_files(
        formula_inputs=[formula_path],
        pris_inputs=[pris_path],
        clscore_inputs=[clscore_path],
        output_dir=output,
        synthesis_formula=_formula("S_syn", SYNTHESIS_FEATURES),
        stability_formula=_formula("S_stab", STABILITY_FEATURES),
        expected_rows=4,
        min_group_rows=2,
    )

    assert summary["rows"]["joined_common_support"] == 4
    assert (output / "score_distributions.csv").is_file()
    assert (output / "score_correlations.csv").is_file()
    assert (output / "score_pris_associations.csv").is_file()
    assert (output / "result_summary.json").is_file()
    correlations = pd.read_csv(output / "score_correlations.csv")
    row = correlations[
        (correlations.score_variant == "S_syn")
        & (correlations.clscore == "CLscore_A")
        & (correlations.analysis_scope == "raw")
        & (correlations.stratum_dimension == "overall")
    ].iloc[0]
    assert row.spearman_rho == pytest.approx(1.0)


def test_full_pool_projection_excludes_cif_and_unused_legacy_columns():
    columns = _full_pool_formula_input_columns(
        _formula("S_syn", SYNTHESIS_FEATURES),
        _formula("S_stab", STABILITY_FEATURES),
    )

    assert "orig_index" in columns
    assert "charge_assignment_route" in columns
    assert "formula_syn_madz_mean" in columns
    assert "formula_stab_econ_min" in columns
    assert "cif_sha256" not in columns
    assert "formula_syn_valence_route" not in columns
    assert "formula_stab_valence_route" not in columns
    assert "feature_wyckoff_econ" not in columns
    assert "wyckoff_econ_symprec_0p1" not in columns


def test_full_pool_file_analysis_freezes_pool_rows_and_shard_counts(tmp_path):
    formula = _raw_formula_frame().iloc[[0, 0, 0, 0, 0, 0]].reset_index(drop=True)
    formula["orig_index"] = [5, 0, 4, 1, 3, 2]
    formula["charge_assignment_route"] = ["integer"] * 6
    pris = pd.DataFrame(
        {
            "orig_index": [0, 1, 2, 3, 4, 5],
            "charge_assignment_route": ["integer"] * 6,
            "D7_verdict": ["pass", "pass", "explicit_violation", "no_verdict", "pass", "explicit_violation"],
            "L4_verdict": ["pass", "explicit_violation", "explicit_violation", "no_verdict", "pass", "pass"],
        }
    )
    clscores = pd.DataFrame(
        {
            "orig_index": [0, 1, 3, 5],
            "CLscore_A": [0.0, 1.0, 3.0, 5.0],
            "CLscore_B": [5.0, 3.0, 1.0, 0.0],
            "CLscore_jang": [0.0, 1.0, 3.0, 5.0],
        }
    )
    formula_dir = tmp_path / "formula"
    pris_dir = tmp_path / "pris"
    formula_dir.mkdir()
    pris_dir.mkdir()
    for shard in range(2):
        formula.iloc[shard * 3 : (shard + 1) * 3].to_parquet(
            formula_dir / f"part-{shard:05d}.parquet", index=False
        )
        pris.iloc[shard * 3 : (shard + 1) * 3].to_parquet(
            pris_dir / f"part-{shard:05d}.parquet", index=False
        )
    clscore_path = tmp_path / "full_clscores_common.parquet"
    clscores.to_parquet(clscore_path, index=False)

    output = tmp_path / "analysis"
    summary = analyze_full_pool_files(
        formula_inputs=[formula_dir],
        pris_inputs=[pris_dir],
        clscore_inputs=[clscore_path],
        output_dir=output,
        synthesis_formula=_formula("S_syn", SYNTHESIS_FEATURES),
        stability_formula=_formula("S_stab", STABILITY_FEATURES),
        expected_rows=4,
        expected_pool_rows=6,
        expected_formula_files=2,
        expected_pris_files=2,
        expected_clscore_files=1,
        min_group_rows=2,
    )

    assert summary["rows"] == {
        "formula_input": 6,
        "old_pris_input": 6,
        "clscore_common_support": 4,
        "joined_common_support": 4,
    }
    assert summary["input_contract"]["formula_files"] == {
        "observed": 2,
        "expected": 2,
    }
    assert summary["input_contract"]["old_pris_files"] == {
        "observed": 2,
        "expected": 2,
    }
    assert summary["input_contract"]["clscore_files"] == {
        "observed": 1,
        "expected": 1,
    }

    with pytest.raises(ValueError, match="formula.*files"):
        analyze_full_pool_files(
            formula_inputs=[formula_dir],
            pris_inputs=[pris_dir],
            clscore_inputs=[clscore_path],
            output_dir=tmp_path / "wrong-shards",
            synthesis_formula=_formula("S_syn", SYNTHESIS_FEATURES),
            stability_formula=_formula("S_stab", STABILITY_FEATURES),
            expected_rows=4,
            expected_pool_rows=6,
            expected_formula_files=3,
            expected_pris_files=2,
            expected_clscore_files=1,
            min_group_rows=2,
        )

    with pytest.raises(ValueError, match="formula pool has 6 rows, expected 7"):
        analyze_full_pool_files(
            formula_inputs=[formula_dir],
            pris_inputs=[pris_dir],
            clscore_inputs=[clscore_path],
            output_dir=tmp_path / "wrong-rows",
            synthesis_formula=_formula("S_syn", SYNTHESIS_FEATURES),
            stability_formula=_formula("S_stab", STABILITY_FEATURES),
            expected_rows=4,
            expected_pool_rows=7,
            expected_formula_files=2,
            expected_pris_files=2,
            expected_clscore_files=1,
            min_group_rows=2,
        )


def test_provenance_uses_binary_source_and_full_pool_frozen_index_boundary():
    binary = pd.DataFrame(
        {
            "S_syn": [1.0, 2.0],
            "S_syn_no_D7_D8": [1.0, 2.0],
            "S_syn_no_D7_D8_madz": [1.0, 2.0],
            "S_stab": [1.0, 2.0],
            "S_syn_all_observed": [True, True],
            "S_stab_all_observed": [True, True],
            "formula_syn_historical_size_domain": [True, True],
            "formula_stab_historical_size_domain": [True, True],
            "source": ["icsd", "cod"],
            "source_split": ["train", "val"],
            "charge_assignment_route": ["integer", "integer"],
        }
    )
    binary_distribution = build_score_distributions(binary)
    binary_sources = set(
        binary_distribution.loc[
            binary_distribution.stratum_dimension.eq("provenance"), "stratum_value"
        ]
    )
    assert binary_sources == {"icsd", "cod"}

    full = binary.copy()
    full["orig_index"] = [6_120_139, 6_120_140]
    full_distribution = build_score_distributions(full)
    full_sources = set(
        full_distribution.loc[
            full_distribution.stratum_dimension.eq("provenance"), "stratum_value"
        ]
    )
    assert full_sources == {"lemat", "elementa"}
