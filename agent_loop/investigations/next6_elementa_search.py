#!/usr/bin/env python3
"""Finite, interpretable law/formula search for the ELEMENTA x0 migration.

Absolute-law and same-composition relative tracks are kept separate.  The
relative track is an operational cohort ranking rule, not an intrinsic crystal
law.  No score function accepts endpoint labels, IDs, material suffixes, or row
order as inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.next6_elementa_protocol import (
    apply_group_threshold,
    attach_energy_labels,
    elementa_stage,
    evaluate_group_triage,
    group_conformal_threshold,
)
from src.next6_wbm_build import sha256_file


@dataclass(frozen=True)
class TermSpec:
    column: str
    direction: int
    weight: float
    block: str
    support_column: str

    def __post_init__(self) -> None:
        if self.direction not in {-1, 1}:
            raise ValueError("term direction must be -1 or +1")
        if self.weight <= 0:
            raise ValueError("term weight must be positive")


@dataclass(frozen=True)
class FormulaSpec:
    name: str
    track: str
    role: str
    terms: tuple[TermSpec, ...]

    def __post_init__(self) -> None:
        if self.track not in {"absolute_law", "cohort_relative", "cohort_margin"}:
            raise ValueError("unknown formula track")
        if self.role not in {"baseline", "candidate"}:
            raise ValueError("unknown formula role")
        if not self.terms:
            raise ValueError("a formula needs at least one term")

    @property
    def complexity(self) -> int:
        return len(self.terms)


def _term(
    column: str,
    direction: int,
    block: str,
    support: str,
    weight: float = 1.0,
) -> TermSpec:
    return TermSpec(column, direction, weight, block, support)


def candidate_catalog() -> list[FormulaSpec]:
    """Return the frozen sparse physics catalog and explicit baselines."""

    p2 = _term("p2_mean_dev", 1, "topology", "support_p2")
    p3 = _term("p3_frac_edge_face", 1, "topology", "support_pauling")
    p4 = _term("p4_violate", 1, "topology", "support_pauling")
    p5 = _term("p5_penalty", 1, "topology", "support_p5")
    pack = _term("geom_packing_fraction", -1, "geometry", "support_geom")
    born = _term("born_wbm_envelope", 1, "geometry", "support_geom")
    overlap = _term("min_pair_overlap", 1, "geometry", "support_geom")
    ewald = _term("ewald_per_atom", 1, "electrostatic", "support_ewald")
    econ = _term("econ_mean", -1, "coordination", "support_econ")
    dist = _term("dist_rsd", 1, "distortion", "support_dist")
    blmin = _term("bl_min", -1, "shannon", "support_shannon")
    gii = _term("gii", 1, "bvs", "support_bvs_legacy")
    gii_strict = _term("gii", 1, "bvs", "support_bvs_strict")
    p9c = _term("p9c_bond_mismatch_q95", 1, "lewis", "support_p9c")
    p9r = _term("p9r_bond_mismatch_q95_min", 1, "lewis", "support_p9r")

    specs = [
        FormulaSpec("pauling_p2", "absolute_law", "baseline", (p2,)),
        FormulaSpec("pauling_p3", "absolute_law", "baseline", (p3,)),
        FormulaSpec("pauling_p4", "absolute_law", "baseline", (p4,)),
        FormulaSpec("pauling_p5", "absolute_law", "baseline", (p5,)),
        FormulaSpec(
            "pauling_original_equal", "absolute_law", "baseline", (p2, p3, p4, p5)
        ),
        FormulaSpec("born_wbm_envelope", "absolute_law", "baseline", (born,)),
        FormulaSpec("min_pair_overlap", "absolute_law", "baseline", (overlap,)),
        FormulaSpec("packing_low", "absolute_law", "baseline", (pack,)),
        FormulaSpec("ewald_per_atom", "absolute_law", "baseline", (ewald,)),
        FormulaSpec("econ_low", "absolute_law", "baseline", (econ,)),
        FormulaSpec("distortion_rsd", "absolute_law", "baseline", (dist,)),
        FormulaSpec("shannon_bl_min", "absolute_law", "baseline", (blmin,)),
        FormulaSpec("bvs_gii_legacy", "absolute_law", "baseline", (gii,)),
        FormulaSpec("bvs_gii_strict", "absolute_law", "baseline", (gii_strict,)),
        FormulaSpec("p9_corrected_q95", "absolute_law", "baseline", (p9c,)),
        FormulaSpec("p9_robust_q95", "absolute_law", "baseline", (p9r,)),
    ]

    # Physics directions are fixed.  Only small positive integer weights and at
    # most one term per block are admitted for candidate formulas.
    for track in ("absolute_law", "cohort_relative"):
        prefix = "absolute" if track == "absolute_law" else "relative"
        specs.extend(
            [
                FormulaSpec(f"{prefix}_ewald_packing", track, "candidate", (ewald, pack)),
                FormulaSpec(
                    f"{prefix}_ewald_packing_p2", track, "candidate", (ewald, pack, p2)
                ),
                FormulaSpec(
                    f"{prefix}_ewald_packing_p2light",
                    track,
                    "candidate",
                    (
                        _term("ewald_per_atom", 1, "electrostatic", "support_ewald", 2),
                        _term(
                            "geom_packing_fraction", -1, "geometry", "support_geom", 2
                        ),
                        _term("p2_mean_dev", 1, "topology", "support_p2", 1),
                    ),
                ),
                FormulaSpec(
                    f"{prefix}_ewald_born_p2", track, "candidate", (ewald, born, p2)
                ),
                FormulaSpec(
                    f"{prefix}_ewald_packing_p9r", track, "candidate", (ewald, pack, p9r)
                ),
                FormulaSpec(
                    f"{prefix}_ewald_packing_econ", track, "candidate", (ewald, pack, econ)
                ),
                FormulaSpec(
                    f"{prefix}_ewald_packing_dist", track, "candidate", (ewald, pack, dist)
                ),
            ]
        )
    return specs


def prepare_search_features(
    base_features: pd.DataFrame,
    p9_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build a label-free allowlisted feature table joined strictly by ``sid``."""

    scale_suffixes = ("080", "090", "100", "110", "120")
    base_columns = [
        "sid",
        "rk",
        "material",
        "geom_feature_ok",
        "geom_min_pair_ratio",
        "geom_packing_fraction",
        "pauling_feature_ok",
        "p2_mean_dev",
        "p3_frac_edge_face",
        "p4_violate",
        "p5_n_distinct",
        "shannon_feature_ok",
        "bl_min",
        "ewald_feature_ok",
        "ewald_per_atom",
        "econ_mean",
        "dist_rsd",
        "bvs_feature_ok",
        "gii",
        "bv_param_cov",
    ]
    for suffix in scale_suffixes:
        base_columns.extend(
            [f"geom_repulsion_p2_l{suffix}", f"geom_packing_l{suffix}"]
        )
    p9_columns = [
        "sid",
        "rk",
        "material",
        "strict_x0_ok",
        "p9c_feature_ok",
        "p9r_feature_ok",
        "p9r_assignment_count",
        "p9c_bond_mismatch_q95",
        "p9r_bond_mismatch_q95_min",
    ]
    missing_base = set(base_columns) - set(base_features.columns)
    missing_p9 = set(p9_columns) - set(p9_features.columns)
    if missing_base or missing_p9:
        raise ValueError(
            f"missing allowlisted features: base={sorted(missing_base)}, p9={sorted(missing_p9)}"
        )
    base = base_features[base_columns].copy()
    p9 = p9_features[p9_columns].copy()
    if base["sid"].duplicated().any() or p9["sid"].duplicated().any():
        raise ValueError("feature sid must be unique")
    joined = base.merge(
        p9,
        on="sid",
        how="inner",
        suffixes=("", "_p9"),
        validate="one_to_one",
    )
    if len(joined) != len(base) or len(joined) != len(p9):
        raise ValueError("base and P9 sid sets differ")
    if not joined["rk"].eq(joined["rk_p9"]).all() or not joined["material"].eq(
        joined["material_p9"]
    ).all():
        raise ValueError("base and P9 metadata differ")
    joined = joined.drop(columns=["rk_p9", "material_p9"])

    ratio = pd.to_numeric(joined["geom_min_pair_ratio"], errors="coerce").to_numpy(float)
    joined["min_pair_overlap"] = np.maximum(1.0 / ratio - 1.0, 0.0) ** 2
    scale_scores = []
    for suffix in scale_suffixes:
        repulsion = pd.to_numeric(
            joined[f"geom_repulsion_p2_l{suffix}"], errors="coerce"
        ).to_numpy(float)
        packing = pd.to_numeric(
            joined[f"geom_packing_l{suffix}"], errors="coerce"
        ).to_numpy(float)
        scale_scores.append(repulsion + 4.0 * np.maximum(packing - 1.2, 0.0) ** 2)
    joined["born_wbm_envelope"] = np.min(np.vstack(scale_scores), axis=0)
    joined["p5_penalty"] = np.maximum(
        pd.to_numeric(joined["p5_n_distinct"], errors="coerce").to_numpy(float) - 1.0,
        0.0,
    )

    strict = joined["strict_x0_ok"].fillna(False).to_numpy(bool)
    unique_charge = (
        pd.to_numeric(joined["p9r_assignment_count"], errors="coerce")
        .eq(1)
        .to_numpy(bool)
    )

    def finite(*columns: str) -> np.ndarray:
        return np.logical_and.reduce(
            [
                np.isfinite(pd.to_numeric(joined[column], errors="coerce").to_numpy(float))
                for column in columns
            ]
        )

    joined["support_geom"] = (
        strict
        & joined["geom_feature_ok"].fillna(False).to_numpy(bool)
        & finite(
            "geom_min_pair_ratio",
            "geom_packing_fraction",
            "born_wbm_envelope",
            "min_pair_overlap",
        )
    )
    joined["support_p2"] = strict & unique_charge & finite("p2_mean_dev")
    joined["support_p5"] = strict & unique_charge & finite("p5_penalty")
    joined["support_pauling"] = (
        strict
        & unique_charge
        & joined["pauling_feature_ok"].fillna(False).to_numpy(bool)
        & finite("p2_mean_dev", "p3_frac_edge_face", "p4_violate", "p5_penalty")
    )
    ewald_ok = (
        strict
        & unique_charge
        & joined["ewald_feature_ok"].fillna(False).to_numpy(bool)
    )
    joined["support_ewald"] = ewald_ok & finite("ewald_per_atom")
    joined["support_econ"] = ewald_ok & finite("econ_mean")
    joined["support_dist"] = ewald_ok & finite("dist_rsd")
    joined["support_shannon"] = (
        strict
        & unique_charge
        & joined["shannon_feature_ok"].fillna(False).to_numpy(bool)
        & finite("bl_min")
    )
    bvs_ok = (
        strict
        & unique_charge
        & joined["bvs_feature_ok"].fillna(False).to_numpy(bool)
        & finite("gii", "bv_param_cov")
    )
    joined["support_bvs_legacy"] = bvs_ok
    joined["support_bvs_strict"] = bvs_ok & np.isclose(
        pd.to_numeric(joined["bv_param_cov"], errors="coerce").to_numpy(float),
        1.0,
    )
    joined["support_p9c"] = (
        strict
        & unique_charge
        & joined["p9c_feature_ok"].fillna(False).to_numpy(bool)
        & finite("p9c_bond_mismatch_q95")
    )
    joined["support_p9r"] = (
        strict
        & joined["p9r_feature_ok"].fillna(False).to_numpy(bool)
        & finite("p9r_bond_mismatch_q95_min")
    )
    return joined


def fit_normalization(features: pd.DataFrame, spec: FormulaSpec) -> dict[str, dict[str, float]]:
    """Fit label-free median/IQR scales for an absolute formula."""

    if spec.track == "cohort_relative":
        return {}
    out: dict[str, dict[str, float]] = {}
    strict = features["strict_x0_ok"].fillna(False).to_numpy(dtype=bool)
    for term in spec.terms:
        values = pd.to_numeric(features[term.column], errors="coerce").to_numpy(dtype=float)
        support = features[term.support_column].fillna(False).to_numpy(dtype=bool)
        valid = strict & support & np.isfinite(values)
        if not valid.any():
            out[term.column] = {"center": 0.0, "scale": 1.0, "n": 0}
            continue
        center = float(np.median(values[valid]))
        q25, q75 = np.quantile(values[valid], [0.25, 0.75])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        out[term.column] = {"center": center, "scale": scale, "n": int(valid.sum())}
    return out


def _relative_bad_rank(
    features: pd.DataFrame,
    term: TermSpec,
    strict: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = pd.to_numeric(features[term.column], errors="coerce").to_numpy(dtype=float)
    base_support = features[term.support_column].fillna(False).to_numpy(dtype=bool)
    valid = strict & base_support & np.isfinite(values)
    bad = pd.Series(np.where(valid, term.direction * values, np.nan), index=features.index)
    rank = bad.groupby(features["rk"], sort=False).rank(method="average") - 1.0
    count = bad.groupby(features["rk"], sort=False).transform("count")
    ranked = (rank / (count - 1.0)).to_numpy(dtype=float)
    supported = valid & (count.to_numpy(dtype=float) > 1.0) & np.isfinite(ranked)
    ranked[~supported] = np.nan
    return ranked, supported


def score_formula(
    features: pd.DataFrame,
    spec: FormulaSpec,
    normalization: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Score one formula using feature allowlists only; larger is worse."""

    required = {"sid", "rk", "strict_x0_ok"}
    for term in spec.terms:
        required.update({term.column, term.support_column})
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"missing score columns: {sorted(missing)}")
    strict = features["strict_x0_ok"].fillna(False).to_numpy(dtype=bool)
    terms: list[np.ndarray] = []
    supports: list[np.ndarray] = []
    for term in spec.terms:
        if spec.track == "cohort_relative":
            values, supported = _relative_bad_rank(features, term, strict)
        else:
            raw = pd.to_numeric(features[term.column], errors="coerce").to_numpy(dtype=float)
            base_support = features[term.support_column].fillna(False).to_numpy(dtype=bool)
            params = normalization.get(term.column)
            if params is None:
                raise ValueError(f"missing normalization for {term.column}")
            values = term.direction * (
                raw - float(params["center"])
            ) / float(params["scale"])
            supported = strict & base_support & np.isfinite(values)
            values[~supported] = np.nan
        terms.append(term.weight * values)
        supports.append(supported)
    support = np.logical_and.reduce(supports)
    stacked = np.vstack(terms)
    score = np.sum(stacked, axis=0) / sum(term.weight for term in spec.terms)
    score[~support] = np.nan
    return pd.DataFrame(
        {
            "sid": features["sid"].astype(str).to_numpy(),
            "score": score,
            "decision_support": support,
        }
    )


def _join_scores_and_labels(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    if scores["sid"].duplicated().any() or labels["sid"].duplicated().any():
        raise ValueError("score and label sid values must be unique")
    labelled = attach_energy_labels(labels[["sid", "rk", "e_per_atom"]])
    joined = labelled.merge(scores, on="sid", how="inner", validate="one_to_one")
    if len(joined) != len(labelled) or len(joined) != len(scores):
        raise ValueError("score and label sid sets differ")
    return joined


def calibrate_and_evaluate(
    spec: FormulaSpec,
    normalization: Mapping[str, Mapping[str, float]],
    calibration_features: pd.DataFrame,
    calibration_labels: pd.DataFrame,
    evaluation_features: pd.DataFrame,
    evaluation_labels: pd.DataFrame,
    *,
    alpha: float,
    protected_column: str = "valuable",
    within_group: str = "max",
) -> tuple[dict[str, object], pd.DataFrame]:
    """Use the same group-conformal path for every candidate and baseline."""

    calibration = _join_scores_and_labels(
        score_formula(calibration_features, spec, normalization), calibration_labels
    )
    if protected_column not in calibration.columns:
        raise ValueError(f"unknown protected label column: {protected_column}")
    calibrated = group_conformal_threshold(
        calibration.assign(supported=calibration["decision_support"]),
        alpha=alpha,
        valuable_column=protected_column,
        within_group=within_group,
    )
    evaluation = _join_scores_and_labels(
        score_formula(evaluation_features, spec, normalization), evaluation_labels
    )
    evaluation["decision"] = apply_group_threshold(
        evaluation["score"].to_numpy(dtype=float),
        evaluation["decision_support"].to_numpy(dtype=bool),
        float(calibrated["threshold"]),
    )
    metrics: dict[str, object] = {
        "name": spec.name,
        "track": spec.track,
        "role": spec.role,
        "complexity": spec.complexity,
        "alpha": float(alpha),
        "threshold": float(calibrated["threshold"]),
        "calibration_n_groups": int(calibrated["n_groups"]),
        "calibration_order_index": int(calibrated["order_index"]),
        "risk_protected_column": protected_column,
        "risk_within_group": within_group,
        **evaluate_group_triage(evaluation),
    }
    evaluation["formula"] = spec.name
    evaluation["alpha"] = float(alpha)
    return metrics, evaluation


def choose_formula(frontier: pd.DataFrame, *, gate: str = "valuable_all") -> pd.Series:
    """Choose one candidate after fixed safety gates, then savings and simplicity."""

    if gate not in {"valuable_all", "group_min"}:
        raise ValueError("gate must be 'valuable_all' or 'group_min'")
    required = {
        "name",
        "role",
        "valuable_group_retention_lower",
        "exact_min_retention_lower",
        "regret_p95",
        "dft_savings",
        "complexity",
    }
    if gate == "group_min":
        required.update({"near_min_retention_lower", "all_rejected_groups"})
    missing = required - set(frontier.columns)
    if missing:
        raise ValueError(f"frontier is missing columns: {sorted(missing)}")
    safety = (
        frontier["role"].eq("candidate")
        & frontier["exact_min_retention_lower"].ge(0.95)
        & frontier["regret_p95"].le(0.05)
    )
    if gate == "valuable_all":
        safety &= frontier["valuable_group_retention_lower"].ge(0.95)
    else:
        safety &= (
            frontier["near_min_retention_lower"].ge(0.95)
            & frontier["all_rejected_groups"].eq(0)
        )
    eligible = frontier.loc[safety].copy()
    if eligible.empty:
        raise ValueError(f"no candidate formula passes the frozen {gate} safety gates")
    eligible = eligible.sort_values(
        ["dft_savings", "complexity", "name"],
        ascending=[False, True, True],
        kind="stable",
    )
    return eligible.iloc[0]


def run_prepared_search(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    output_dir: Path,
    *,
    specs: Sequence[FormulaSpec] | None = None,
    alpha_values: Sequence[float] = (0.05, 0.01),
    protected_column: str = "valuable",
    within_group: str = "max",
) -> dict[str, object]:
    """Run selection, independent threshold calibration, and one test opening."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    opening_path = output_dir / "TEST_OPENING.json"
    if opening_path.exists():
        raise RuntimeError(f"ELEMENTA migration test already opened: {opening_path}")
    if features["sid"].duplicated().any() or labels["sid"].duplicated().any():
        raise ValueError("prepared feature and label sid values must be unique")
    if set(features["sid"].astype(str)) != set(labels["sid"].astype(str)):
        raise ValueError("prepared feature and label sid sets differ")
    stages = features[["sid", "rk"]].copy()
    stages["stage"] = stages["rk"].astype(str).map(elementa_stage)
    stages.to_parquet(output_dir / "stage_assignments.parquet", index=False)
    label_map = labels[["sid", "rk", "e_per_atom"]].copy()

    def stage_data(stage: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        mask = stages["stage"].eq(stage).to_numpy()
        x = features.loc[mask].copy()
        ids = set(x["sid"].astype(str))
        y = label_map.loc[label_map["sid"].astype(str).isin(ids)].copy()
        if len(x) != len(y):
            raise ValueError(f"feature/label stage mismatch for {stage}")
        return x, y

    search_x, search_y = stage_data("search_calibration")
    selection_x, selection_y = stage_data("formula_selection")
    threshold_x, threshold_y = stage_data("threshold_calibration")
    test_x, test_y = stage_data("test")
    catalog = list(candidate_catalog() if specs is None else specs)
    if not catalog:
        raise ValueError("formula catalog is empty")
    normalizations = {
        spec.name: fit_normalization(search_x, spec) for spec in catalog
    }
    selection_alpha = float(alpha_values[0])
    frontier_rows: list[dict[str, object]] = []
    for spec in catalog:
        metrics, _ = calibrate_and_evaluate(
            spec,
            normalizations[spec.name],
            search_x,
            search_y,
            selection_x,
            selection_y,
            alpha=selection_alpha,
            protected_column=protected_column,
            within_group=within_group,
        )
        frontier_rows.append(metrics)
    frontier = pd.DataFrame(frontier_rows).sort_values("name", kind="stable")
    frontier_path = output_dir / "formula_selection_frontier.parquet"
    frontier.to_parquet(frontier_path, index=False)
    selection_gate = (
        "group_min"
        if protected_column in {"near_min", "exact_min"} and within_group == "min"
        else "valuable_all"
    )
    try:
        chosen_name: str | None = str(
            choose_formula(frontier, gate=selection_gate)["name"]
        )
    except ValueError:
        chosen_name = None

    final_specs = [spec for spec in catalog if spec.role == "baseline"]
    if chosen_name is not None:
        final_specs.append(next(spec for spec in catalog if spec.name == chosen_name))
    opening = {
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "single frozen-rule ELEMENTA x0 migration diagnostic",
        "evidence_class": "historically seen discovery; not confirmatory",
        "selected_formula": chosen_name or "null_keep_all",
        "n_test_rows": len(test_x),
        "n_test_groups": int(test_x["rk"].nunique()),
        "risk_protected_column": protected_column,
        "risk_within_group": within_group,
        "selection_gate": selection_gate,
    }
    opening_path.write_text(
        json.dumps(opening, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    metric_rows: list[dict[str, object]] = []
    prediction_tables: list[pd.DataFrame] = []
    frozen_entries: list[dict[str, object]] = []
    for spec in final_specs:
        for alpha in alpha_values:
            metrics, predictions = calibrate_and_evaluate(
                spec,
                normalizations[spec.name],
                threshold_x,
                threshold_y,
                test_x,
                test_y,
                alpha=float(alpha),
                protected_column=protected_column,
                within_group=within_group,
            )
            metric_rows.append(metrics)
            prediction_tables.append(predictions)
            frozen_entries.append(
                {
                    "formula": asdict(spec),
                    "normalization": normalizations[spec.name],
                    "alpha": float(alpha),
                    "threshold": metrics["threshold"],
                }
            )
    test_metrics = pd.DataFrame(metric_rows)
    test_metrics_path = output_dir / "test_metrics.parquet"
    test_metrics.to_parquet(test_metrics_path, index=False)
    predictions_path = output_dir / "test_predictions.parquet"
    if prediction_tables:
        pd.concat(prediction_tables, ignore_index=True).to_parquet(
            predictions_path, index=False
        )
    else:
        pd.DataFrame().to_parquet(predictions_path, index=False)
    frozen_path = output_dir / "frozen_rules.json"
    frozen_path.write_text(
        json.dumps(
            {
                "protocol": "2026-08-01-elementa-x0-migration-v1",
                "selected_formula": chosen_name or "null_keep_all",
                "risk_protected_column": protected_column,
                "risk_within_group": within_group,
                "selection_gate": selection_gate,
                "rules": frozen_entries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "protocol": "2026-08-01-elementa-x0-migration-v1",
        "selected_formula": chosen_name or "null_keep_all",
        "risk_protected_column": protected_column,
        "risk_within_group": within_group,
        "selection_gate": selection_gate,
        "counts": {
            stage: int((stages["stage"] == stage).sum())
            for stage in (
                "search_calibration",
                "formula_selection",
                "threshold_calibration",
                "test",
            )
        },
        "outputs_sha256": {
            path.name: sha256_file(path)
            for path in (
                output_dir / "stage_assignments.parquet",
                frontier_path,
                frozen_path,
                test_metrics_path,
                predictions_path,
                opening_path,
            )
        },
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def run_migration_search(
    elementa_dir: Path,
    p9_dir: Path,
    output_dir: Path,
    *,
    alpha_values: Sequence[float] = (0.05, 0.01),
    protected_column: str = "valuable",
    within_group: str = "max",
) -> dict[str, object]:
    """Load the additive x0 artifacts, prepare allowlisted inputs, and run the loop."""

    elementa_dir = Path(elementa_dir)
    p9_dir = Path(p9_dir)
    output_dir = Path(output_dir)
    if (output_dir / "TEST_OPENING.json").exists():
        raise RuntimeError(f"ELEMENTA migration test already opened: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = elementa_dir / "elementa_x0_features.parquet"
    p9_path = p9_dir / "elementa_x0_p9_features.parquet"
    labels_path = elementa_dir / "elementa_labels.parquet"
    prepared = prepare_search_features(
        pd.read_parquet(base_path), pd.read_parquet(p9_path)
    )
    prepared_path = output_dir / "prepared_x0_features.parquet"
    prepared.to_parquet(prepared_path, index=False)
    manifest = run_prepared_search(
        prepared,
        pd.read_parquet(labels_path),
        output_dir,
        alpha_values=alpha_values,
        protected_column=protected_column,
        within_group=within_group,
    )
    manifest["inputs_sha256"] = {
        base_path.name: sha256_file(base_path),
        p9_path.name: sha256_file(p9_path),
        labels_path.name: sha256_file(labels_path),
    }
    manifest["outputs_sha256"][prepared_path.name] = sha256_file(prepared_path)
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elementa", type=Path, required=True)
    parser.add_argument("--p9", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=(0.05, 0.01))
    parser.add_argument("--protected-column", choices=("valuable", "near_min", "exact_min"), default="valuable")
    parser.add_argument("--within-group", choices=("max", "min"), default="max")
    args = parser.parse_args(argv)
    manifest = run_migration_search(
        args.elementa,
        args.p9,
        args.output,
        alpha_values=tuple(args.alphas),
        protected_column=args.protected_column,
        within_group=args.within_group,
    )
    print(json.dumps({"selected_formula": manifest["selected_formula"], **manifest["counts"]}, sort_keys=True))
    return 0


__all__ = [
    "FormulaSpec",
    "TermSpec",
    "candidate_catalog",
    "calibrate_and_evaluate",
    "choose_formula",
    "fit_normalization",
    "prepare_search_features",
    "run_prepared_search",
    "run_migration_search",
    "score_formula",
]


if __name__ == "__main__":
    raise SystemExit(main())
