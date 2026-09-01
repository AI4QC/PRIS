#!/usr/bin/env python3
"""Leakage-resistant sparse formula search for same-composition ranking."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from better_search import (
    deterministic_group_folds,
    group_equal_accuracy,
    is_frozen_p1_search_feature,
    make_group_pairs,
)


@dataclass(frozen=True)
class SparsePairModel:
    feature_names: tuple[str, ...]
    coefficients: np.ndarray
    means: dict[str, float]
    scales: dict[str, float]
    l1_c: float
    max_terms: int

    def structure_scores(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = []
        for feature in self.feature_names:
            values = frame[feature].to_numpy(dtype=float)
            finite = np.isfinite(values)
            filled = np.where(finite, values, self.means[feature])
            matrix.append((filled - self.means[feature]) / self.scales[feature])
        if not matrix:
            return np.zeros(len(frame), dtype=float)
        return np.column_stack(matrix) @ self.coefficients


def _standardization(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for feature in feature_columns:
        values = frame[feature].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise ValueError(f"feature has no finite training values: {feature}")
        means[feature] = float(np.mean(finite))
        scale = float(np.std(finite))
        scales[feature] = scale if scale > 0 else 1.0
    return means, scales


def _standardized_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    means: Mapping[str, float],
    scales: Mapping[str, float],
) -> pd.DataFrame:
    transformed = frame[["rk", "e_hull"]].copy()
    for feature in feature_columns:
        values = frame[feature].to_numpy(dtype=float)
        filled = np.where(np.isfinite(values), values, means[feature])
        transformed[feature] = (filled - means[feature]) / scales[feature]
    return transformed


def fit_sparse_pair_model(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    l1_c: float,
    max_terms: int,
    min_gap: float = 0.0,
) -> SparsePairModel:
    """Select with L1, cap terms, then refit selected standardized terms."""

    if max_terms < 1:
        raise ValueError("max_terms must be positive")
    return _fit_sparse_pair_models(
        frame,
        feature_columns=feature_columns,
        l1_c=l1_c,
        term_counts=(max_terms,),
        min_gap=min_gap,
    )[max_terms]


def _fit_sparse_pair_models(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    l1_c: float,
    term_counts: Sequence[int],
    min_gap: float,
) -> dict[int, SparsePairModel]:
    """Fit one L1 selector and refit every requested sparse term count."""

    if not term_counts or min(term_counts) < 1:
        raise ValueError("term_counts must contain positive integers")
    means, scales = _standardization(frame, feature_columns)
    transformed = _standardized_frame(frame, feature_columns, means, scales)
    pairs = make_group_pairs(
        transformed,
        group_col="rk",
        target_col="e_hull",
        feature_cols=feature_columns,
        min_gap=min_gap,
    )
    from sklearn.linear_model import LogisticRegression

    selector = LogisticRegression(
        solver="liblinear",
        l1_ratio=1.0,
        C=l1_c,
        fit_intercept=False,
        max_iter=4000,
        random_state=0,
    )
    selector.fit(pairs.X, pairs.y, sample_weight=pairs.sample_weight)
    weights = selector.coef_[0]
    nonzero = np.flatnonzero(np.abs(weights) > 1e-10)
    if not len(nonzero):
        raise ValueError("L1 selector produced an empty formula")
    ranking = nonzero[np.argsort(-np.abs(weights[nonzero]))]
    models = {}
    for max_terms in sorted(set(term_counts)):
        selected = ranking[:max_terms]
        selected = np.asarray(sorted(selected.tolist()), dtype=int)
        refit = LogisticRegression(
            solver="liblinear",
            l1_ratio=0.0,
            C=100.0,
            fit_intercept=False,
            max_iter=4000,
            random_state=0,
        )
        refit.fit(
            pairs.X[:, selected],
            pairs.y,
            sample_weight=pairs.sample_weight,
        )
        names = tuple(feature_columns[index] for index in selected)
        models[max_terms] = SparsePairModel(
            feature_names=names,
            coefficients=refit.coef_[0].astype(float),
            means={name: means[name] for name in names},
            scales={name: scales[name] for name in names},
            l1_c=float(l1_c),
            max_terms=int(max_terms),
        )
    return models


def fixed_confidence_thresholds(
    training_scores: np.ndarray,
    *,
    coverages: Sequence[float],
) -> dict[str, float]:
    """Freeze numeric |score| thresholds from training scores only."""

    absolute = np.abs(np.asarray(training_scores, dtype=float))
    absolute = absolute[np.isfinite(absolute)]
    if not len(absolute):
        raise ValueError("training_scores contains no finite values")
    thresholds = {}
    for coverage in coverages:
        if not 0 < coverage <= 1:
            raise ValueError("coverages must lie in (0, 1]")
        threshold = (
            0.0
            if coverage == 1
            else float(np.quantile(absolute, 1 - coverage, method="higher"))
        )
        thresholds[f"{coverage:.2f}"] = threshold
    return thresholds


def evaluate_fixed_thresholds(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    thresholds: Mapping[str, float],
) -> dict[str, dict[str, float | int]]:
    """Apply already-frozen numeric thresholds to a new split."""

    values = np.asarray(scores, dtype=float)
    truth = np.asarray(labels, dtype=int)
    group_values = np.asarray(groups, dtype=object)
    if not (len(values) == len(truth) == len(group_values)):
        raise ValueError("scores, labels, and groups must have equal length")
    out: dict[str, dict[str, float | int]] = {}
    for name, threshold in thresholds.items():
        keep = np.isfinite(values) & (np.abs(values) >= threshold)
        if not keep.any():
            out[name] = {
                "threshold": float(threshold),
                "coverage": 0.0,
                "pooled_accuracy": float("nan"),
                "group_equal_accuracy": float("nan"),
                "group_equal_ci": [float("nan"), float("nan")],
                "n_pairs": 0,
                "n_groups": 0,
            }
            continue
        prediction = (values[keep] > 0).astype(int)
        out[name] = {
            "threshold": float(threshold),
            "coverage": float(keep.mean()),
            "pooled_accuracy": float(np.mean(prediction == truth[keep])),
            "group_equal_accuracy": group_equal_accuracy(
                truth[keep],
                prediction,
                group_values[keep],
            ),
            "group_equal_ci": _bootstrap_group_equal_ci(
                values[keep],
                truth[keep],
                group_values[keep],
            ),
            "n_pairs": int(keep.sum()),
            "n_groups": int(len(set(group_values[keep].tolist()))),
        }
    return out


def _pair_scores(
    model: SparsePairModel,
    frame: pd.DataFrame,
    *,
    min_gap: float,
):
    scored = frame[["rk", "e_hull"]].copy()
    scored["formula_score"] = model.structure_scores(frame)
    pairs = make_group_pairs(
        scored,
        group_col="rk",
        target_col="e_hull",
        feature_cols=["formula_score"],
        min_gap=min_gap,
    )
    return pairs, pairs.X[:, 0]


def inner_oof_confidence_thresholds(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    l1_c: float,
    max_terms: int,
    n_folds: int,
    seed: int,
    min_gap: float,
    coverages: Sequence[float] = (1.0, 0.5, 0.3, 0.2, 0.1, 0.05),
) -> tuple[dict[str, float], dict[str, int]]:
    """Fit numeric confidence cutoffs from grouped inner OOF scores only."""

    assignment = deterministic_group_folds(
        frame["rk"].to_numpy(dtype=object),
        n_splits=n_folds,
        seed=seed,
    )
    score_blocks = []
    validation_groups: set[str] = set()
    completed = 0
    for fold in range(n_folds):
        train = frame.loc[assignment != fold].reset_index(drop=True)
        validation = frame.loc[assignment == fold].reset_index(drop=True)
        model = fit_sparse_pair_model(
            train,
            feature_columns=feature_columns,
            l1_c=l1_c,
            max_terms=max_terms,
            min_gap=min_gap,
        )
        _, scores = _pair_scores(model, validation, min_gap=min_gap)
        score_blocks.append(np.asarray(scores, dtype=float))
        validation_groups.update(str(value) for value in validation["rk"].unique())
        completed += 1
    if completed != n_folds or not score_blocks:
        raise ValueError("inner OOF confidence fitting did not complete all folds")
    all_scores = np.concatenate(score_blocks)
    thresholds = fixed_confidence_thresholds(
        all_scores,
        coverages=coverages,
    )
    return thresholds, {
        "n_folds": completed,
        "n_oof_pairs": int(len(all_scores)),
        "n_oof_groups": int(len(validation_groups)),
    }


def _bootstrap_group_equal_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_bootstrap: int = 400,
    seed: int = 0,
) -> list[float]:
    correct = (np.asarray(scores) > 0).astype(int) == np.asarray(labels)
    group_frame = pd.DataFrame(
        {"group": np.asarray(groups, dtype=object), "correct": correct}
    )
    accuracies = group_frame.groupby("group", dropna=False)["correct"].mean().to_numpy()
    rng = np.random.default_rng(seed)
    draws = [
        float(rng.choice(accuracies, size=len(accuracies), replace=True).mean())
        for _ in range(n_bootstrap)
    ]
    return [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def _pair_metrics(pairs, scores: np.ndarray) -> dict[str, object]:
    prediction = (np.asarray(scores) > 0).astype(int)
    record: dict[str, object] = {
        "pooled_accuracy": float(np.mean(prediction == pairs.y)),
        "group_equal_accuracy": group_equal_accuracy(
            pairs.y,
            prediction,
            pairs.groups,
        ),
        "group_equal_ci": _bootstrap_group_equal_ci(
            scores,
            pairs.y,
            pairs.groups,
        ),
        "n_pairs": int(len(pairs.y)),
        "n_groups": int(len(set(pairs.groups.tolist()))),
    }
    gap_strata = {}
    bounds = (0.0, 0.025, 0.05, 0.10, float("inf"))
    for lower, upper in zip(bounds[:-1], bounds[1:]):
        selected = (pairs.target_gaps >= lower) & (pairs.target_gaps < upper)
        if not selected.any():
            continue
        label = f"{lower:.3f}-{'inf' if not np.isfinite(upper) else f'{upper:.3f}'}"
        gap_strata[label] = {
            "n_pairs": int(selected.sum()),
            "n_groups": int(len(set(pairs.groups[selected].tolist()))),
            "pooled_accuracy": float(
                np.mean(prediction[selected] == pairs.y[selected])
            ),
            "group_equal_accuracy": group_equal_accuracy(
                pairs.y[selected],
                prediction[selected],
                pairs.groups[selected],
            ),
        }
    record["gap_strata_eV_per_atom"] = gap_strata
    return record


def _select_configuration(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    c_grid: Sequence[float],
    term_grid: Sequence[int],
    n_folds: int,
    seed: int,
    min_gap: float,
) -> tuple[tuple[float, int], list[dict[str, object]]]:
    folds = deterministic_group_folds(
        frame["rk"].to_numpy(dtype=object),
        n_splits=n_folds,
        seed=seed,
    )
    accumulated = {
        (float(l1_c), int(max_terms)): []
        for l1_c in c_grid
        for max_terms in term_grid
    }
    for fold in range(n_folds):
        train = frame.loc[folds != fold].reset_index(drop=True)
        validation = frame.loc[folds == fold].reset_index(drop=True)
        for l1_c in c_grid:
            try:
                models = _fit_sparse_pair_models(
                    train,
                    feature_columns=feature_columns,
                    l1_c=l1_c,
                    term_counts=term_grid,
                    min_gap=min_gap,
                )
            except ValueError:
                continue
            for max_terms, model in models.items():
                try:
                    pairs, pair_score = _pair_scores(
                        model,
                        validation,
                        min_gap=min_gap,
                    )
                except ValueError:
                    continue
                accumulated[(float(l1_c), int(max_terms))].append(
                    group_equal_accuracy(
                        pairs.y,
                        (pair_score > 0).astype(int),
                        pairs.groups,
                    )
                )
    results = []
    for (l1_c, max_terms), scores in accumulated.items():
        if len(scores) != n_folds:
            continue
        results.append(
            {
                "l1_c": float(l1_c),
                "max_terms": int(max_terms),
                "mean_group_equal_accuracy": float(np.mean(scores)),
                "std_group_equal_accuracy": float(np.std(scores)),
                "fold_scores": [float(value) for value in scores],
            }
        )
    if not results:
        raise ValueError("no sparse configuration succeeded in all inner folds")
    winner = max(
        results,
        key=lambda row: (
            row["mean_group_equal_accuracy"],
            -row["std_group_equal_accuracy"],
            -row["max_terms"],
            -row["l1_c"],
        ),
    )
    return (winner["l1_c"], winner["max_terms"]), results


def _model_record(model: SparsePairModel) -> dict[str, object]:
    terms = []
    for feature, coefficient in zip(model.feature_names, model.coefficients):
        terms.append(
            {
                "feature": feature,
                "coefficient_on_z": float(coefficient),
                "training_mean": model.means[feature],
                "training_scale": model.scales[feature],
                "raw_difference_coefficient": float(
                    coefficient / model.scales[feature]
                ),
            }
        )
    return {
        "orientation": "higher score predicts lower E_hull within a composition",
        "expression": "S = sum_i coefficient_on_z_i * z_i",
        "l1_c": model.l1_c,
        "max_terms": model.max_terms,
        "n_terms": len(model.feature_names),
        "terms": terms,
    }


def _nested_search(
    discovery: pd.DataFrame,
    calibration: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    c_grid: Sequence[float],
    term_grid: Sequence[int],
    outer_folds: int,
    inner_folds: int,
    seed: int,
    min_gap: float,
    min_feature_coverage: float,
) -> dict[str, object]:
    outer_assignment = deterministic_group_folds(
        discovery["rk"].to_numpy(dtype=object),
        n_splits=outer_folds,
        seed=seed,
    )
    outer_rows = []
    selected_configs: list[tuple[float, int]] = []
    selected_features: list[str] = []
    for fold in range(outer_folds):
        train = discovery.loc[outer_assignment != fold].reset_index(drop=True)
        test = discovery.loc[outer_assignment == fold].reset_index(drop=True)
        fold_features = [
            feature
            for feature in feature_columns
            if np.isfinite(train[feature].to_numpy(dtype=float)).mean()
            >= min_feature_coverage
            and np.nanstd(train[feature].to_numpy(dtype=float)) > 1e-12
        ]
        if not fold_features:
            raise ValueError(f"outer fold {fold} has no eligible training features")
        config, inner_results = _select_configuration(
            train,
            fold_features,
            c_grid=c_grid,
            term_grid=term_grid,
            n_folds=inner_folds,
            seed=seed + 100 + fold,
            min_gap=min_gap,
        )
        model = fit_sparse_pair_model(
            train,
            feature_columns=fold_features,
            l1_c=config[0],
            max_terms=config[1],
            min_gap=min_gap,
        )
        pairs, scores = _pair_scores(model, test, min_gap=min_gap)
        metrics = _pair_metrics(pairs, scores)
        thresholds, threshold_audit = inner_oof_confidence_thresholds(
            train,
            feature_columns=fold_features,
            l1_c=config[0],
            max_terms=config[1],
            n_folds=inner_folds,
            seed=seed + 100 + fold,
            min_gap=min_gap,
        )
        selected_configs.append(config)
        selected_features.extend(model.feature_names)
        outer_rows.append(
            {
                "fold": fold,
                "n_train_structures": len(train),
                "n_test_structures": len(test),
                "selected_config": {
                    "l1_c": config[0],
                    "max_terms": config[1],
                },
                "selected_features": list(model.feature_names),
                "eligible_features_fit_on_outer_train": fold_features,
                "metrics": metrics,
                "confidence_thresholds_fit_on_inner_oof": thresholds,
                "confidence_threshold_audit": threshold_audit,
                "outer_abstention": evaluate_fixed_thresholds(
                    scores=scores,
                    labels=pairs.y,
                    groups=pairs.groups,
                    thresholds=thresholds,
                ),
                "inner_grid": inner_results,
            }
        )

    config_counts = Counter(selected_configs)
    consensus = max(
        config_counts,
        key=lambda config: (
            config_counts[config],
            -config[1],
            -config[0],
        ),
    )
    final_model = fit_sparse_pair_model(
        discovery,
        feature_columns=feature_columns,
        l1_c=consensus[0],
        max_terms=consensus[1],
        min_gap=min_gap,
    )
    discovery_pairs, discovery_scores = _pair_scores(
        final_model,
        discovery,
        min_gap=min_gap,
    )
    calibration_pairs, calibration_scores = _pair_scores(
        final_model,
        calibration,
        min_gap=min_gap,
    )
    thresholds, threshold_audit = inner_oof_confidence_thresholds(
        discovery,
        feature_columns=feature_columns,
        l1_c=consensus[0],
        max_terms=consensus[1],
        n_folds=inner_folds,
        seed=seed + 5000,
        min_gap=min_gap,
    )
    return {
        "outer_folds": outer_rows,
        "outer_group_equal_mean": float(
            np.mean(
                [
                    row["metrics"]["group_equal_accuracy"]
                    for row in outer_rows
                ]
            )
        ),
        "outer_group_equal_std": float(
            np.std(
                [
                    row["metrics"]["group_equal_accuracy"]
                    for row in outer_rows
                ]
            )
        ),
        "consensus_config": {
            "l1_c": consensus[0],
            "max_terms": consensus[1],
            "outer_selection_count": int(config_counts[consensus]),
        },
        "feature_stability": {
            feature: int(count_value)
            for feature, count_value in Counter(selected_features).most_common()
        },
        "final_formula": _model_record(final_model),
        "discovery_refit_metrics": _pair_metrics(
            discovery_pairs,
            discovery_scores,
        ),
        "calibration_metrics": _pair_metrics(
            calibration_pairs,
            calibration_scores,
        ),
        "confidence_thresholds_fit_on_discovery_grouped_oof": thresholds,
        "confidence_threshold_audit": threshold_audit,
        "discovery_refit_abstention": evaluate_fixed_thresholds(
            scores=discovery_scores,
            labels=discovery_pairs.y,
            groups=discovery_pairs.groups,
            thresholds=thresholds,
        ),
        "calibration_abstention": evaluate_fixed_thresholds(
            scores=calibration_scores,
            labels=calibration_pairs.y,
            groups=calibration_pairs.groups,
            thresholds=thresholds,
        ),
    }


def _load_formula_frame(
    features_dir: Path,
    real_descriptors: Path,
) -> pd.DataFrame:
    import formula2

    raw_splits = pd.read_parquet(
        features_dir / "real_rank.parquet",
        columns=["split"],
    )["split"]
    source_access_audit = {
        "source_tables_materialized_all_splits": True,
        "lockbox_access": True,
        "lockbox_rows_in_fit_or_evaluation": False,
        "materialized_lockbox_real_rank_rows": int(
            raw_splits.eq("lockbox").sum()
        ),
        "materialized_unknown_split_real_rank_rows": int(
            raw_splits.isna().sum()
        ),
        "reason": (
            "formula2.load reads monolithic real_rank and feature parquets "
            "before row filtering"
        ),
    }
    previous = formula2.F
    formula2.F = str(features_dir) + os.sep
    try:
        frame = formula2.load(phys=True)
    finally:
        formula2.F = previous
    extra = pd.read_parquet(real_descriptors).drop(columns=["split"], errors="ignore")
    if extra["source_id"].duplicated().any():
        raise ValueError("real descriptor source_id is not unique")
    before = len(frame)
    frame = frame.merge(extra, on="source_id", how="left", validate="one_to_one")
    if len(frame) != before:
        raise AssertionError("descriptor merge changed formula row count")
    if frame["split"].isna().any() or frame["split"].eq("lockbox").any():
        raise ValueError("formula frame contains lockbox or unknown rows")
    frame.attrs["source_access_audit"] = source_access_audit
    return frame


def _eligible_features(
    discovery: pd.DataFrame,
    *,
    include_new: bool,
    min_coverage: float,
    p1_vocabulary: str = "frozen",
) -> list[str]:
    if p1_vocabulary not in {"frozen", "expanded"}:
        raise ValueError("p1_vocabulary must be 'frozen' or 'expanded'")
    drop = {"source_id", "rk", "e_hull", "split", "anion", "sid", "parent", "kind"}
    features = []
    for column in discovery:
        if column in drop or discovery[column].dtype.kind != "f":
            continue
        is_new = column.startswith("bvloc_")
        if is_new and not include_new:
            continue
        if is_new and (
            column.endswith(("coverage", "count", "fraction"))
            or "parameter_" in column
        ):
            continue
        if (
            is_new
            and p1_vocabulary == "frozen"
            and not is_frozen_p1_search_feature(column)
        ):
            continue
        finite = np.isfinite(discovery[column].to_numpy(dtype=float))
        if finite.mean() < min_coverage:
            continue
        if np.nanstd(discovery[column].to_numpy(dtype=float)) <= 1e-12:
            continue
        features.append(column)
    return sorted(features)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def outer_fold_direction_gate(deltas: Sequence[float]) -> bool:
    """Require all positive folds, or one bounded negative fold at worst."""

    values = [float(value) for value in deltas]
    if not values or any(not np.isfinite(value) for value in values):
        return False
    positive = sum(value > 0 for value in values)
    if positive == len(values):
        return True
    return positive == len(values) - 1 and min(values) >= -0.01


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nested grouped search for an additive sparse ranking formula."
    )
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--real-descriptors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--min-gap", type=float, default=0.0)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--p1-vocabulary",
        choices=("frozen", "expanded"),
        default="frozen",
    )
    parser.add_argument(
        "--c-grid",
        type=float,
        nargs="+",
        default=[0.01, 0.03, 0.1, 0.3, 1.0],
    )
    parser.add_argument(
        "--term-grid",
        type=int,
        nargs="+",
        default=[3, 5, 7],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    if max(args.term_grid) > 7:
        raise SystemExit("formula complexity is frozen at at most seven terms")
    started = time.time()
    frame = _load_formula_frame(args.features_dir, args.real_descriptors)
    source_access_audit = dict(frame.attrs["source_access_audit"])
    discovery = frame[frame["split"].eq("discovery")].reset_index(drop=True)
    calibration = frame[frame["split"].eq("calibration")].reset_index(drop=True)
    print(
        f"discovery {len(discovery):,}/{discovery.rk.nunique():,} groups; "
        f"calibration {len(calibration):,}/{calibration.rk.nunique():,} groups; "
        "fit/evaluation lockbox rows absent; monolithic source did materialize "
        f"{source_access_audit['materialized_lockbox_real_rank_rows']:,} "
        "lockbox ranking rows",
        flush=True,
    )
    existing_features = _eligible_features(
        discovery,
        include_new=False,
        min_coverage=args.min_coverage,
        p1_vocabulary=args.p1_vocabulary,
    )
    additive_features = _eligible_features(
        discovery,
        include_new=True,
        min_coverage=args.min_coverage,
        p1_vocabulary=args.p1_vocabulary,
    )
    print(
        f"eligible features existing={len(existing_features)}, "
        f"additive={len(additive_features)}",
        flush=True,
    )
    variants = {}
    for name, features in (
        ("existing_formula_loop", existing_features),
        ("additive_bvloc_formula_loop", additive_features),
    ):
        print(f"nested search: {name}", flush=True)
        variants[name] = _nested_search(
            discovery,
            calibration,
            features,
            c_grid=args.c_grid,
            term_grid=args.term_grid,
            outer_folds=args.outer_folds,
            inner_folds=args.inner_folds,
            seed=args.seed,
            min_gap=args.min_gap,
            min_feature_coverage=args.min_coverage,
        )
        print(
            f"  outer={variants[name]['outer_group_equal_mean']:.4f}; "
            f"calibration="
            f"{variants[name]['calibration_metrics']['group_equal_accuracy']:.4f}",
            flush=True,
        )

    existing = variants["existing_formula_loop"]
    additive = variants["additive_bvloc_formula_loop"]
    fold_deltas = [
        additive["outer_folds"][index]["metrics"]["group_equal_accuracy"]
        - existing["outer_folds"][index]["metrics"]["group_equal_accuracy"]
        for index in range(args.outer_folds)
    ]
    final_new_features = [
        term["feature"]
        for term in additive["final_formula"]["terms"]
        if term["feature"].startswith("bvloc_")
    ]
    commitment_targets = ("1.00", "0.30", "0.10")
    outer_commitment = {}
    for target in commitment_targets:
        existing_scores = [
            float(row["outer_abstention"][target]["group_equal_accuracy"])
            for row in existing["outer_folds"]
        ]
        additive_scores = [
            float(row["outer_abstention"][target]["group_equal_accuracy"])
            for row in additive["outer_folds"]
        ]
        deltas = [
            new_value - old_value
            for old_value, new_value in zip(existing_scores, additive_scores)
        ]
        outer_commitment[target] = {
            "existing_group_equal_mean": float(np.mean(existing_scores)),
            "additive_group_equal_mean": float(np.mean(additive_scores)),
            "group_equal_accuracy_delta": float(np.mean(deltas)),
            "fold_deltas": deltas,
            "fold_direction_gate": outer_fold_direction_gate(deltas),
            "existing_realized_coverage_mean": float(
                np.mean(
                    [
                        row["outer_abstention"][target]["coverage"]
                        for row in existing["outer_folds"]
                    ]
                )
            ),
            "additive_realized_coverage_mean": float(
                np.mean(
                    [
                        row["outer_abstention"][target]["coverage"]
                        for row in additive["outer_folds"]
                    ]
                )
            ),
        }
    qualifying_targets = [
        target
        for target, metrics in outer_commitment.items()
        if metrics["group_equal_accuracy_delta"] >= 0.02
        and metrics["fold_direction_gate"]
    ]
    comparison = {
        "outer_group_equal_accuracy_delta": (
            additive["outer_group_equal_mean"] - existing["outer_group_equal_mean"]
        ),
        "calibration_group_equal_accuracy_delta": (
            additive["calibration_metrics"]["group_equal_accuracy"]
            - existing["calibration_metrics"]["group_equal_accuracy"]
        ),
        "outer_fold_deltas": fold_deltas,
        "outer_positive_folds": int(sum(delta > 0 for delta in fold_deltas)),
        "new_features_in_final_formula": final_new_features,
        "outer_fixed_commitment": outer_commitment,
        "qualifying_commitment_targets": qualifying_targets,
    }
    comparison["metric_gate_without_source_access_condition"] = bool(
        final_new_features
        and qualifying_targets
    )
    comparison["preliminary_gate"] = bool(
        comparison["metric_gate_without_source_access_condition"]
        and not (
            source_access_audit["source_tables_materialized_all_splits"]
            and source_access_audit["materialized_lockbox_real_rank_rows"] > 0
        )
    )
    comparison["status"] = (
        "pending temporal/source holdout; historical calibration is diagnostic"
    )
    report = {
        "protocol": {
            "selection_data": "discovery only",
            "outer_cv": "deterministic reduced-formula grouped",
            "inner_cv": "deterministic reduced-formula grouped",
            "pair_weighting": "each reduced-formula group sums to one",
            "antisymmetric_double_write": True,
            "max_terms": 7,
            "p1_vocabulary": args.p1_vocabulary,
            "abstention_threshold_source": (
                "numeric thresholds fit on grouped inner out-of-fold pair scores"
            ),
            "calibration_role": "historical diagnostic; previously adaptively reused",
            **source_access_audit,
            "min_gap_eV_per_atom": args.min_gap,
        },
        "counts": {
            "discovery_structures": len(discovery),
            "discovery_groups": int(discovery.rk.nunique()),
            "calibration_structures": len(calibration),
            "calibration_groups": int(calibration.rk.nunique()),
            "existing_features": len(existing_features),
            "additive_features": len(additive_features),
            "new_eligible_features": len(
                set(additive_features).difference(existing_features)
            ),
        },
        "variants": variants,
        "comparison": comparison,
        "provenance": {
            "runtime_seconds": time.time() - started,
            "descriptor_sha256": _hash_file(args.real_descriptors),
            "descriptor_metadata_sha256": _hash_file(
                args.real_descriptors.with_suffix(
                    args.real_descriptors.suffix + ".meta.json"
                )
            ),
            "implementation_sha256": _hash_file(Path(__file__)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
