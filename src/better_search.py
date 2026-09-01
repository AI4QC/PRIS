"""Shared, leakage-resistant primitives for the additive 2026-07-31 search.

This module intentionally contains no repository-specific file paths and no
lockbox access.  Experiment drivers must pass already selected data frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Collection, Mapping, Sequence

import numpy as np
import pandas as pd


FROZEN_P1_SEARCH_FEATURES = frozenset(
    f"bvloc_{charge}_{metric}_{aggregate}"
    for charge in ("cat", "an")
    for metric in (
        "absolute_mismatch",
        "effective_cn",
        "vector_asymmetry",
    )
    for aggregate in ("mean", "q95", "max")
)


def is_frozen_p1_search_feature(column: str) -> bool:
    """Return whether a column belongs to the pre-outcome P1 search vocabulary."""

    return column in FROZEN_P1_SEARCH_FEATURES


@dataclass(frozen=True)
class Rule:
    """One threshold rule with an optional threshold guard."""

    feature: str
    op: str
    threshold: float
    guard_feature: str | None = None
    guard_op: str | None = None
    guard_threshold: float | None = None


@dataclass(frozen=True)
class PairwiseDataset:
    """Antisymmetric within-group ranking examples."""

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    sample_weight: np.ndarray
    target_gaps: np.ndarray
    feature_names: tuple[str, ...]


def assert_no_forbidden_splits(
    frame: pd.DataFrame,
    *,
    allowed: Collection[str],
    split_col: str = "split",
) -> None:
    """Fail closed unless every row has an explicitly allowed split."""

    if split_col not in frame:
        raise ValueError(f"missing required split column: {split_col}")
    if frame[split_col].isna().any():
        raise ValueError(f"missing values in split column: {split_col}")
    observed = set(frame[split_col].astype(str).unique())
    forbidden = sorted(observed.difference(allowed))
    if forbidden:
        raise ValueError(f"forbidden splits present: {', '.join(forbidden)}")


def deterministic_group_folds(
    groups: Sequence[object],
    *,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """Assign whole groups to deterministic, approximately balanced folds."""

    values = np.asarray(groups, dtype=object)
    unique = np.asarray(sorted(set(values.tolist()), key=str), dtype=object)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(unique) < n_splits:
        raise ValueError("n_splits exceeds the number of unique groups")
    rng = np.random.default_rng(seed)
    shuffled = unique[rng.permutation(len(unique))]
    mapping = {group: index % n_splits for index, group in enumerate(shuffled)}
    return np.asarray([mapping[group] for group in values], dtype=int)


def _is_missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _compare(value: float, op: str, threshold: float) -> bool:
    if op == ">=":
        return value >= threshold
    if op == ">":
        return value > threshold
    if op == "<=":
        return value <= threshold
    if op == "<":
        return value < threshold
    raise ValueError(f"unsupported operator: {op}")


def _evaluate_one(row: Mapping[str, object], rule: Rule) -> bool | None:
    if rule.guard_feature is not None:
        if rule.guard_op is None or rule.guard_threshold is None:
            raise ValueError("guard_feature requires guard_op and guard_threshold")
        guard_value = row.get(rule.guard_feature)
        if _is_missing(guard_value):
            return None
        if not _compare(float(guard_value), rule.guard_op, rule.guard_threshold):
            return True

    value = row.get(rule.feature)
    if _is_missing(value):
        return None
    return _compare(float(value), rule.op, rule.threshold)


def evaluate_rule(
    row: Mapping[str, object],
    rules: Sequence[Rule],
) -> bool | None:
    """Evaluate an AND rule set with three-valued, fail-closed semantics."""

    saw_unknown = False
    for rule in rules:
        verdict = _evaluate_one(row, rule)
        if verdict is False:
            return False
        if verdict is None:
            saw_unknown = True
    return None if saw_unknown else True


def make_group_pairs(
    frame: pd.DataFrame,
    *,
    group_col: str,
    target_col: str,
    feature_cols: Sequence[str],
    min_gap: float = 0.0,
) -> PairwiseDataset:
    """Build doubled, antisymmetric comparisons within each target group.

    Lower target values are considered better.  Each original group receives
    total sample weight one, regardless of how many pair rows it contributes.
    """

    required = [group_col, target_col, *feature_cols]
    if min_gap < 0:
        raise ValueError("min_gap must be non-negative")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")

    rows: list[np.ndarray] = []
    labels: list[int] = []
    pair_groups: list[object] = []
    target_gaps: list[float] = []
    for group, subset in frame.groupby(group_col, sort=False, dropna=False):
        clean = subset.dropna(subset=[target_col, *feature_cols])
        for left, right in combinations(clean.index.tolist(), 2):
            left_target = float(clean.at[left, target_col])
            right_target = float(clean.at[right, target_col])
            if left_target == right_target or abs(left_target - right_target) < min_gap:
                continue
            difference = (
                clean.loc[left, feature_cols].to_numpy(dtype=float)
                - clean.loc[right, feature_cols].to_numpy(dtype=float)
            )
            left_is_better = int(left_target < right_target)
            rows.extend((difference, -difference))
            labels.extend((left_is_better, 1 - left_is_better))
            pair_groups.extend((group, group))
            target_gaps.extend(
                (abs(left_target - right_target), abs(left_target - right_target))
            )

    if not rows:
        raise ValueError("no non-tied within-group pairs could be constructed")

    group_array = np.asarray(pair_groups, dtype=object)
    counts = pd.Series(group_array).value_counts(dropna=False)
    weights = np.asarray([1.0 / counts[group] for group in group_array], dtype=float)
    return PairwiseDataset(
        X=np.vstack(rows),
        y=np.asarray(labels, dtype=int),
        groups=group_array,
        sample_weight=weights,
        target_gaps=np.asarray(target_gaps, dtype=float),
        feature_names=tuple(feature_cols),
    )


def group_equal_accuracy(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    groups: Sequence[object],
) -> float:
    """Average classification accuracy across groups with equal group weight."""

    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    group_array = np.asarray(groups, dtype=object)
    if not (len(truth) == len(prediction) == len(group_array)):
        raise ValueError("y_true, y_pred, and groups must have equal length")
    if not len(truth):
        raise ValueError("cannot score an empty dataset")
    correct = pd.DataFrame(
        {"group": group_array, "correct": truth == prediction}
    )
    return float(correct.groupby("group", dropna=False)["correct"].mean().mean())
