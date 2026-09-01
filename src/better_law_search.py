#!/usr/bin/env python3
"""Additive, discovery-only search for interpretable plausibility laws."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import heapq
import hashlib
from itertools import count
import json
import os
from pathlib import Path
import time
from typing import Mapping, Sequence
import zlib

import numpy as np
import pandas as pd

from better_search import is_frozen_p1_search_feature


@dataclass(frozen=True)
class LawCandidate:
    description: str
    feature: str
    family: str
    origin: str
    side: str
    thresholds: tuple[float, ...]
    real_mask: np.ndarray
    bad_mask: np.ndarray
    real_coverage: float
    bad_coverage: float
    guard_feature: str | None = None
    guard_side: str | None = None
    guard_threshold: float | None = None


@dataclass(frozen=True)
class BeamResult:
    indices: tuple[int, ...]
    real_mask: np.ndarray
    bad_mask: np.ndarray


def _finite_values(frame: pd.DataFrame, feature: str) -> tuple[np.ndarray, np.ndarray]:
    values = frame[feature].to_numpy(dtype=float)
    return values, np.isfinite(values)


def _one_sided_mask(values: np.ndarray, side: str, threshold: float) -> np.ndarray:
    finite = np.isfinite(values)
    comparison = values <= threshold if side == "hi" else values >= threshold
    return (~finite) | comparison


def apply_candidate(frame: pd.DataFrame, candidate: LawCandidate) -> np.ndarray:
    """Apply a fitted candidate without refitting any threshold."""

    values = frame[candidate.feature].to_numpy(dtype=float)
    if candidate.side == "band":
        lower, upper = candidate.thresholds
        finite = np.isfinite(values)
        target = (~finite) | ((values >= lower) & (values <= upper))
    else:
        target = _one_sided_mask(values, candidate.side, candidate.thresholds[0])
    if candidate.guard_feature is None:
        return target
    guard_values = frame[candidate.guard_feature].to_numpy(dtype=float)
    finite_guard = np.isfinite(guard_values)
    if candidate.guard_side == "hi":
        guard = finite_guard & (guard_values > float(candidate.guard_threshold))
    elif candidate.guard_side == "lo":
        guard = finite_guard & (guard_values <= float(candidate.guard_threshold))
    else:
        raise ValueError(f"unsupported guard side: {candidate.guard_side}")
    return (~guard) | target


def apply_serialized_law_set(
    frame: pd.DataFrame,
    records: Sequence[Mapping[str, object]],
) -> np.ndarray:
    """Apply rules emitted in a search report without refitting them."""

    mask = np.ones(len(frame), dtype=bool)
    for record in records:
        candidate = LawCandidate(
            description=str(record["description"]),
            feature=str(record["feature"]),
            family=str(record["family"]),
            origin=str(record["origin"]),
            side=str(record["side"]),
            thresholds=tuple(float(value) for value in record["thresholds"]),
            real_mask=np.empty(0, dtype=bool),
            bad_mask=np.empty(0, dtype=bool),
            real_coverage=float(record.get("real_coverage", float("nan"))),
            bad_coverage=float(record.get("bad_coverage", float("nan"))),
            guard_feature=(
                None
                if record.get("guard_feature") is None
                else str(record["guard_feature"])
            ),
            guard_side=(
                None
                if record.get("guard_side") is None
                else str(record["guard_side"])
            ),
            guard_threshold=(
                None
                if record.get("guard_threshold") is None
                else float(record["guard_threshold"])
            ),
        )
        mask &= apply_candidate(frame, candidate)
    return mask


def build_one_sided_candidates(
    real: pd.DataFrame,
    bad: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    alphas: Sequence[float],
    min_coverage: float,
    min_rejection: float,
    min_real_satisfaction: float,
    origin: str,
) -> list[LawCandidate]:
    """Fit all thresholds exclusively from real discovery rows."""

    candidates: list[LawCandidate] = []
    for feature in feature_columns:
        if feature not in real or feature not in bad:
            continue
        real_values, real_finite = _finite_values(real, feature)
        bad_values, bad_finite = _finite_values(bad, feature)
        real_coverage = float(real_finite.mean())
        bad_coverage = float(bad_finite.mean())
        if (
            real_coverage < min_coverage
            or bad_coverage < max(0.0, min_coverage - 0.05)
            or int(real_finite.sum()) < 4
        ):
            continue
        fitted = real_values[real_finite]
        for alpha in alphas:
            if not 0 < alpha < 0.5:
                raise ValueError("alphas must lie strictly between 0 and 0.5")
            for side, quantile, operator in (
                ("hi", 1 - alpha, "<="),
                ("lo", alpha, ">="),
            ):
                threshold = float(np.quantile(fitted, quantile))
                real_mask = _one_sided_mask(real_values, side, threshold)
                bad_mask = _one_sided_mask(bad_values, side, threshold)
                if real_mask.mean() < min_real_satisfaction:
                    continue
                if 1 - bad_mask.mean() < min_rejection:
                    continue
                candidates.append(
                    LawCandidate(
                        description=f"{feature} {operator} {threshold:.8g}",
                        feature=feature,
                        family="one-sided",
                        origin=origin,
                        side=side,
                        thresholds=(threshold,),
                        real_mask=real_mask,
                        bad_mask=bad_mask,
                        real_coverage=real_coverage,
                        bad_coverage=bad_coverage,
                    )
                )
    return candidates


def build_band_candidates(
    real: pd.DataFrame,
    bad: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    central_coverages: Sequence[float],
    min_coverage: float,
    min_rejection: float,
    min_real_satisfaction: float,
    origin: str,
) -> list[LawCandidate]:
    """Build central real-distribution bands that can reject either tail."""

    candidates: list[LawCandidate] = []
    for feature in feature_columns:
        if feature not in real or feature not in bad:
            continue
        real_values, real_finite = _finite_values(real, feature)
        bad_values, bad_finite = _finite_values(bad, feature)
        real_coverage = float(real_finite.mean())
        bad_coverage = float(bad_finite.mean())
        if (
            real_coverage < min_coverage
            or bad_coverage < max(0.0, min_coverage - 0.05)
            or int(real_finite.sum()) < 4
        ):
            continue
        fitted = real_values[real_finite]
        for central in central_coverages:
            if not 0 < central < 1:
                raise ValueError("central_coverages must lie strictly between 0 and 1")
            tail = (1 - central) / 2
            lower, upper = np.quantile(fitted, [tail, 1 - tail]).astype(float)
            real_mask = (~real_finite) | (
                (real_values >= lower) & (real_values <= upper)
            )
            bad_mask = (~bad_finite) | (
                (bad_values >= lower) & (bad_values <= upper)
            )
            if real_mask.mean() < min_real_satisfaction:
                continue
            if 1 - bad_mask.mean() < min_rejection:
                continue
            candidates.append(
                LawCandidate(
                    description=f"{lower:.8g} <= {feature} <= {upper:.8g}",
                    feature=feature,
                    family="band",
                    origin=origin,
                    side="band",
                    thresholds=(float(lower), float(upper)),
                    real_mask=real_mask,
                    bad_mask=bad_mask,
                    real_coverage=real_coverage,
                    bad_coverage=bad_coverage,
                )
            )
    return candidates


def build_guarded_candidates(
    real: pd.DataFrame,
    bad: pd.DataFrame,
    target_candidates: Sequence[LawCandidate],
    *,
    guard_columns: Sequence[str],
    guard_quantiles: Sequence[float],
    min_real_satisfaction: float,
    min_rejection: float,
    max_targets: int = 100,
) -> list[LawCandidate]:
    """Wrap strong but context-specific targets as ``if guard then target``."""

    targets = sorted(
        target_candidates,
        key=lambda candidate: 1 - float(candidate.bad_mask.mean()),
        reverse=True,
    )[:max_targets]
    candidates: list[LawCandidate] = []
    seen: set[tuple[str, str, float, bytes, bytes]] = set()
    for guard_feature in guard_columns:
        if guard_feature not in real or guard_feature not in bad:
            continue
        real_values, real_finite = _finite_values(real, guard_feature)
        bad_values, bad_finite = _finite_values(bad, guard_feature)
        if real_finite.mean() < 0.8 or int(real_finite.sum()) < 4:
            continue
        fitted = real_values[real_finite]
        for quantile in guard_quantiles:
            if not 0 < quantile < 1:
                raise ValueError("guard_quantiles must lie strictly between 0 and 1")
            threshold = float(np.quantile(fitted, quantile))
            for guard_side, operator in (("hi", ">"), ("lo", "<=")):
                if guard_side == "hi":
                    real_guard = real_finite & (real_values > threshold)
                    bad_guard = bad_finite & (bad_values > threshold)
                else:
                    real_guard = real_finite & (real_values <= threshold)
                    bad_guard = bad_finite & (bad_values <= threshold)
                applicability = float(real_guard.mean())
                if not 0.1 < applicability < 0.9:
                    continue
                for target in targets:
                    real_mask = (~real_guard) | target.real_mask
                    bad_mask = (~bad_guard) | target.bad_mask
                    if real_mask.mean() < min_real_satisfaction:
                        continue
                    if 1 - bad_mask.mean() < min_rejection:
                        continue
                    key = (
                        guard_feature,
                        guard_side,
                        threshold,
                        real_mask.tobytes(),
                        bad_mask.tobytes(),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        LawCandidate(
                            description=(
                                f"if {guard_feature} {operator} {threshold:.8g} "
                                f"then ({target.description})"
                            ),
                            feature=target.feature,
                            family=f"guarded-{target.family}",
                            origin=target.origin,
                            side=target.side,
                            thresholds=target.thresholds,
                            real_mask=real_mask,
                            bad_mask=bad_mask,
                            real_coverage=target.real_coverage,
                            bad_coverage=target.bad_coverage,
                            guard_feature=guard_feature,
                            guard_side=guard_side,
                            guard_threshold=threshold,
                        )
                    )
    return candidates


def evaluate_masks(
    *,
    real_mask: np.ndarray,
    bad_mask: np.ndarray,
    bad_groups: np.ndarray,
    bad_kinds: np.ndarray,
) -> dict[str, object]:
    real_pass = np.asarray(real_mask, dtype=bool)
    bad_pass = np.asarray(bad_mask, dtype=bool)
    groups = np.asarray(bad_groups, dtype=object)
    kinds = np.asarray(bad_kinds, dtype=object)
    if len(bad_pass) != len(groups) or len(bad_pass) != len(kinds):
        raise ValueError("bad mask, groups, and kinds must have equal length")
    rejected = ~bad_pass
    group_equal = (
        pd.DataFrame({"group": groups, "rejected": rejected})
        .groupby("group", dropna=False)["rejected"]
        .mean()
        .mean()
    )
    return {
        "satisfaction": float(real_pass.mean()),
        "rejection": float(rejected.mean()),
        "group_equal_rejection": float(group_equal),
        "by_kind": {
            str(kind): float(rejected[kinds == kind].mean())
            for kind in sorted(set(kinds.tolist()), key=str)
        },
    }


def leave_one_kind_out_frames(
    discovery_bad: pd.DataFrame,
    calibration_bad: pd.DataFrame,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Return deterministic no-leakage folds over discovery perturbation kinds.

    Only kinds present in discovery create folds: a calibration-only kind cannot
    influence model selection.  The held discovery and calibration frames are
    returned solely for evaluation after the search on ``training`` completes.
    """

    for name, frame in (
        ("discovery_bad", discovery_bad),
        ("calibration_bad", calibration_bad),
    ):
        if "kind" not in frame:
            raise ValueError(f"{name} is missing required column: kind")
        if frame["kind"].isna().any():
            raise ValueError(f"{name} contains unknown perturbation kinds")
    folds = []
    for held_kind in sorted(discovery_bad["kind"].astype(str).unique()):
        training = discovery_bad.loc[
            ~discovery_bad["kind"].astype(str).eq(held_kind)
        ].copy()
        held_discovery = discovery_bad.loc[
            discovery_bad["kind"].astype(str).eq(held_kind)
        ].copy()
        held_calibration = calibration_bad.loc[
            calibration_bad["kind"].astype(str).eq(held_kind)
        ].copy()
        if training.empty or held_discovery.empty:
            raise ValueError(
                f"leave-one-kind-out fold {held_kind} lacks training or holdout rows"
            )
        folds.append(
            (
                held_kind,
                training.reset_index(drop=True),
                held_discovery.reset_index(drop=True),
                held_calibration.reset_index(drop=True),
            )
        )
    if len(folds) < 2:
        raise ValueError("leave-one-kind-out requires at least two discovery kinds")
    return folds


def historical_rule_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Exact L1/L1'/L2/L3 offline masks used by the existing analysis."""

    def values(column: str) -> np.ndarray:
        return frame[column].to_numpy(dtype=float)

    def lower(column: str, threshold: float) -> np.ndarray:
        array = values(column)
        return (~np.isfinite(array)) | (array >= threshold)

    def upper(column: str, threshold: float) -> np.ndarray:
        array = values(column)
        return (~np.isfinite(array)) | (array <= threshold)

    bl = values("bl_min")
    fi = values("fi")
    coordination = values("cn_an_mean")
    d1a = lower("bl_min", 0.735)
    d1b = lower("bl_min", 0.804)
    d2 = (~(np.isfinite(fi) & (fi > 0.50))) | (
        (~np.isfinite(bl)) | (bl <= 1.05)
    )
    bl_mean = values("bl_mean")
    d3 = (~(np.isfinite(coordination) & (coordination <= 3.333))) | (
        (~np.isfinite(bl_mean)) | (bl_mean <= 1.081)
    )
    d4 = upper("madz_range", 31.45)
    d5 = upper("mad_max", 15.17)
    like = values("frac_like_bonds")
    d6 = (~(np.isfinite(fi) & (fi > 0.55))) | (
        (~np.isfinite(like)) | (like <= 1e-4)
    )
    return {
        "L1": d1a,
        "L1'": d1a & d2,
        "L2": d1b & d3 & d4 & d5,
        "L3": d1b & d3 & d4 & d5 & d6,
    }


def _beam_score(bad_mask: np.ndarray, bad_kinds: np.ndarray) -> tuple[float, float]:
    rejected = ~bad_mask
    by_kind = [
        float(rejected[bad_kinds == kind].mean())
        for kind in sorted(set(bad_kinds.tolist()), key=str)
        if np.any(bad_kinds == kind)
    ]
    minimum = min(by_kind) if by_kind else 0.0
    return float(rejected.mean()) + 0.5 * minimum, minimum


def pareto_beam(
    candidates: Sequence[LawCandidate],
    *,
    real_size: int,
    bad_size: int,
    bad_kinds: np.ndarray,
    satisfaction_floor: float,
    max_rules: int = 12,
    width: int = 24,
    min_gain: float = 0.0015,
    real_strata: Mapping[str, np.ndarray] | None = None,
    stratum_floors: Mapping[str, float] | None = None,
) -> BeamResult:
    """Search AND-combinations while preserving raw and efficient branches."""

    kinds = np.asarray(bad_kinds, dtype=object)
    if len(kinds) != bad_size:
        raise ValueError("bad_kinds length does not match bad_size")
    strata = dict(real_strata or {})
    floors = dict(stratum_floors or {})
    if set(strata) != set(floors):
        raise ValueError("real_strata and stratum_floors must have matching keys")
    for name, stratum in strata.items():
        if len(stratum) != real_size or not np.asarray(stratum, dtype=bool).any():
            raise ValueError(f"invalid real stratum: {name}")
    initial = BeamResult(
        indices=(),
        real_mask=np.ones(real_size, dtype=bool),
        bad_mask=np.ones(bad_size, dtype=bool),
    )
    frontier = [initial]
    best: BeamResult | None = None
    for _ in range(max_rules):
        half = max(width // 2, 1)
        raw_heap: list[tuple[tuple, int, BeamResult]] = []
        efficient_heap: list[tuple[tuple, int, BeamResult]] = []
        serial = count()

        def raw_key(state: BeamResult) -> tuple[float, float, float, tuple[int, ...]]:
            score, minimum = _beam_score(state.bad_mask, kinds)
            return score, minimum, state.real_mask.mean(), tuple(-i for i in state.indices)

        def efficient_key(
            state: BeamResult,
        ) -> tuple[float, float, float, tuple[int, ...]]:
            rejection = float((~state.bad_mask).mean())
            cost = max(1 - float(state.real_mask.mean()), 1e-8)
            _, minimum = _beam_score(state.bad_mask, kinds)
            return rejection / cost, minimum, rejection, tuple(-i for i in state.indices)

        def retain(
            heap: list[tuple[tuple, int, BeamResult]],
            key: tuple,
            state: BeamResult,
            limit: int,
        ) -> None:
            entry = (key, next(serial), state)
            if len(heap) < limit:
                heapq.heappush(heap, entry)
            elif key > heap[0][0]:
                heapq.heapreplace(heap, entry)

        for state in frontier:
            used = {candidates[index].feature for index in state.indices}
            start = state.indices[-1] + 1 if state.indices else 0
            for index in range(start, len(candidates)):
                candidate = candidates[index]
                if candidate.feature in used:
                    continue
                real_mask = state.real_mask & candidate.real_mask
                if real_mask.mean() < satisfaction_floor:
                    continue
                if any(
                    real_mask[np.asarray(stratum, dtype=bool)].mean()
                    < floors[name]
                    for name, stratum in strata.items()
                ):
                    continue
                bad_mask = state.bad_mask & candidate.bad_mask
                gain = state.bad_mask.mean() - bad_mask.mean()
                if gain <= min_gain:
                    continue
                expanded = BeamResult(
                    indices=(*state.indices, index),
                    real_mask=real_mask,
                    bad_mask=bad_mask,
                )
                retain(raw_heap, raw_key(expanded), expanded, half)
                retain(efficient_heap, efficient_key(expanded), expanded, width)
        if not raw_heap and not efficient_heap:
            break
        raw = [
            entry[2]
            for entry in sorted(raw_heap, key=lambda entry: entry[:2], reverse=True)
        ]
        raw_indices = {state.indices for state in raw}
        efficient = [
            entry[2]
            for entry in sorted(
                efficient_heap,
                key=lambda entry: entry[:2],
                reverse=True,
            )
            if entry[2].indices not in raw_indices
        ][: max(width - len(raw), 0)]
        frontier = raw + efficient
        for state in frontier:
            if best is None or raw_key(state) > raw_key(best):
                best = state
    if best is None:
        raise ValueError("no candidate combination satisfies the requested floor")
    return best


def _load_search_frames(
    features_dir: Path,
    real_descriptors: Path,
    bad_descriptors: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the historical tables and add descriptors without changing row sets."""

    import rules_final

    previous = rules_final.F
    rules_final.F = str(features_dir) + os.sep
    try:
        real, bad = rules_final.load(phys=True)
    finally:
        rules_final.F = previous

    real_extra = pd.read_parquet(real_descriptors).drop(
        columns=["split"],
        errors="ignore",
    )
    bad_extra = pd.read_parquet(bad_descriptors).drop(
        columns=["kind", "parent", "split"],
        errors="ignore",
    )
    if real_extra["source_id"].duplicated().any():
        raise ValueError("real descriptor source_id is not unique")
    if bad_extra["sid"].duplicated().any():
        raise ValueError("bad descriptor sid is not unique")
    real_count, bad_count = len(real), len(bad)
    real = real.merge(
        real_extra,
        on="source_id",
        how="left",
        validate="one_to_one",
    )
    bad = bad.merge(
        bad_extra,
        on="sid",
        how="left",
        validate="one_to_one",
    )
    if len(real) != real_count or len(bad) != bad_count:
        raise AssertionError("descriptor merge changed the historical row set")

    source_access_audit = {
        "source_tables_materialized_all_splits": True,
        "lockbox_access": True,
        "lockbox_rows_in_fit_or_evaluation": False,
        "materialized_lockbox_real_rows": int(real["split"].eq("lockbox").sum()),
        "materialized_lockbox_bad_rows": int(bad["psplit"].eq("lockbox").sum()),
        "reason": (
            "rules_final.load reads monolithic parquet tables before row filtering"
        ),
    }
    discovery_real = real[real["split"].eq("discovery")].reset_index(drop=True)
    calibration_real = real[real["split"].eq("calibration")].reset_index(drop=True)
    discovery_bad = bad[bad["psplit"].eq("discovery")].reset_index(drop=True)
    calibration_bad = bad[bad["psplit"].eq("calibration")].reset_index(drop=True)
    for frame in (
        discovery_real,
        calibration_real,
        discovery_bad,
        calibration_bad,
    ):
        frame.attrs["source_access_audit"] = source_access_audit
    for name, frame, column in (
        ("discovery real", discovery_real, "split"),
        ("calibration real", calibration_real, "split"),
        ("discovery bad", discovery_bad, "psplit"),
        ("calibration bad", calibration_bad, "psplit"),
    ):
        if frame.empty:
            raise ValueError(f"{name} is empty")
        if frame[column].isna().any() or frame[column].eq("lockbox").any():
            raise ValueError(f"{name} contains lockbox or unknown rows")
    return discovery_real, calibration_real, discovery_bad, calibration_bad


def _candidate_key(candidate: LawCandidate) -> tuple[object, ...]:
    return (
        candidate.feature,
        candidate.family,
        candidate.side,
        candidate.thresholds,
        candidate.guard_feature,
        candidate.guard_side,
        candidate.guard_threshold,
    )


def _deduplicate(candidates: Sequence[LawCandidate]) -> list[LawCandidate]:
    unique: dict[tuple[object, ...], LawCandidate] = {}
    for candidate in candidates:
        unique.setdefault(_candidate_key(candidate), candidate)
    return sorted(
        unique.values(),
        key=lambda candidate: tuple(str(value) for value in _candidate_key(candidate)),
    )


def _apply_result(
    frame: pd.DataFrame,
    candidates: Sequence[LawCandidate],
    result: BeamResult,
) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for index in result.indices:
        mask &= apply_candidate(frame, candidates[index])
    return mask


def _anion_satisfaction(
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    min_rows: int = 100,
) -> dict[str, dict[str, float | int]]:
    if "anion" not in frame:
        return {}
    rows = []
    for anion, subset in frame.groupby("anion", dropna=True):
        if len(subset) < min_rows:
            continue
        selected = mask[subset.index.to_numpy()]
        rows.append(
            (
                str(anion),
                {"n": int(len(subset)), "satisfaction": float(selected.mean())},
            )
        )
    return dict(rows)


def _candidate_record(candidate: LawCandidate) -> dict[str, object]:
    return {
        "description": candidate.description,
        "feature": candidate.feature,
        "family": candidate.family,
        "origin": candidate.origin,
        "side": candidate.side,
        "thresholds": list(candidate.thresholds),
        "real_coverage": candidate.real_coverage,
        "bad_coverage": candidate.bad_coverage,
        "guard_feature": candidate.guard_feature,
        "guard_side": candidate.guard_side,
        "guard_threshold": candidate.guard_threshold,
    }


def selected_rule_coverage(
    real: pd.DataFrame,
    bad: pd.DataFrame,
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Report finite-value coverage for every selected target or guard feature."""

    features = sorted(
        {
            str(value)
            for record in records
            for value in (record["feature"], record.get("guard_feature"))
            if value is not None
        }
    )
    missing = [
        f"{name}:{feature}"
        for name, frame in (("real", real), ("bad", bad))
        for feature in features
        if feature not in frame
    ]
    if missing:
        raise ValueError("selected rule feature missing: " + ", ".join(missing))
    real_by_feature = {
        feature: float(np.isfinite(real[feature].to_numpy(dtype=float)).mean())
        for feature in features
    }
    bad_by_feature = {
        feature: float(np.isfinite(bad[feature].to_numpy(dtype=float)).mean())
        for feature in features
    }
    return {
        "features": features,
        "real_by_feature": real_by_feature,
        "bad_by_feature": bad_by_feature,
        "real_min": min(real_by_feature.values(), default=1.0),
        "bad_min": min(bad_by_feature.values(), default=1.0),
    }


def law_preliminary_gate(
    *,
    new_descriptor_selected: bool,
    additive_satisfaction: float,
    base_satisfaction: float,
    rejection_delta: float,
    min_kind_rejection_delta: float,
    worst_anion_delta: float,
    real_coverage: float,
    bad_coverage: float,
    source_tables_materialized_lockbox_rows: bool,
) -> bool:
    """Evaluate the frozen law gate, including access and coverage conditions."""

    return bool(
        new_descriptor_selected
        and additive_satisfaction >= base_satisfaction - 0.005
        and (rejection_delta >= 0.02 or min_kind_rejection_delta >= 0.03)
        and worst_anion_delta >= -0.01
        and real_coverage >= 0.90
        and bad_coverage >= 0.90
        and not source_tables_materialized_lockbox_rows
    )


def _evaluate_result(
    real: pd.DataFrame,
    bad: pd.DataFrame,
    candidates: Sequence[LawCandidate],
    result: BeamResult,
    *,
    use_stored_masks: bool,
) -> dict[str, object]:
    real_mask = result.real_mask if use_stored_masks else _apply_result(
        real, candidates, result
    )
    bad_mask = result.bad_mask if use_stored_masks else _apply_result(
        bad, candidates, result
    )
    metrics = evaluate_masks(
        real_mask=real_mask,
        bad_mask=bad_mask,
        bad_groups=bad["parent"].to_numpy(),
        bad_kinds=bad["kind"].to_numpy(),
    )
    metrics["by_anion"] = _anion_satisfaction(real, real_mask)
    return metrics


def _fold_diagnostics(
    real: pd.DataFrame,
    bad: pd.DataFrame,
    real_mask: np.ndarray,
    bad_mask: np.ndarray,
    *,
    n_folds: int = 5,
) -> list[dict[str, object]]:
    real_fold = np.asarray(
        [
            (zlib.crc32(str(source).encode()) & 0x7FFFFFFF) % n_folds
            for source in real["source_id"]
        ]
    )
    bad_fold = np.asarray(
        [
            (zlib.crc32(str(parent).encode()) & 0x7FFFFFFF) % n_folds
            for parent in bad["parent"]
        ]
    )
    rows = []
    for fold in range(n_folds):
        rm = real_fold == fold
        bm = bad_fold == fold
        if not rm.any() or not bm.any():
            continue
        metrics = evaluate_masks(
            real_mask=real_mask[rm],
            bad_mask=bad_mask[bm],
            bad_groups=bad.loc[bm, "parent"].to_numpy(),
            bad_kinds=bad.loc[bm, "kind"].to_numpy(),
        )
        metrics["fold"] = fold
        metrics["n_real"] = int(rm.sum())
        metrics["n_bad"] = int(bm.sum())
        rows.append(metrics)
    return rows


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _historical_metrics(
    real: pd.DataFrame,
    bad: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    real_masks = historical_rule_masks(real)
    bad_masks = historical_rule_masks(bad)
    return {
        name: evaluate_masks(
            real_mask=real_masks[name],
            bad_mask=bad_masks[name],
            bad_groups=bad["parent"].to_numpy(),
            bad_kinds=bad["kind"].to_numpy(),
        )
        for name in real_masks
    }


def build_candidate_sets(
    real: pd.DataFrame,
    bad: pd.DataFrame,
    *,
    min_coverage: float,
    max_guard_targets: int,
    guard_min_real_satisfaction: float = 0.95,
    p1_vocabulary: str = "frozen",
) -> tuple[dict[str, list[LawCandidate]], dict[str, int]]:
    """Build the existing and additive pools with the production search grammar."""

    alphas = (
        0.0003,
        0.0007,
        0.0015,
        0.003,
        0.005,
        0.01,
        0.02,
        0.03,
        0.05,
        0.08,
        0.12,
    )
    old_features = [
        column
        for column in real
        if column in bad
        and real[column].dtype.kind == "f"
        and not column.startswith("bvloc_")
    ]
    diagnostic_suffixes = ("coverage", "count", "fraction")
    if p1_vocabulary not in {"frozen", "expanded"}:
        raise ValueError("p1_vocabulary must be 'frozen' or 'expanded'")
    new_features = [
        column
        for column in real
        if column in bad
        and column.startswith("bvloc_")
        and not column.endswith(diagnostic_suffixes)
        and "parameter_" not in column
        and (
            p1_vocabulary == "expanded"
            or is_frozen_p1_search_feature(column)
        )
    ]
    old_candidates = build_one_sided_candidates(
        real,
        bad,
        old_features,
        alphas=alphas,
        min_coverage=min_coverage,
        min_rejection=0.01,
        min_real_satisfaction=0.88,
        origin="existing",
    )
    new_one_sided = build_one_sided_candidates(
        real,
        bad,
        new_features,
        alphas=alphas,
        min_coverage=min_coverage,
        min_rejection=0.01,
        min_real_satisfaction=0.88,
        origin="bvloc-p1",
    )
    new_bands = build_band_candidates(
        real,
        bad,
        new_features,
        central_coverages=(0.99, 0.98, 0.95, 0.90),
        min_coverage=min_coverage,
        min_rejection=0.01,
        min_real_satisfaction=0.88,
        origin="bvloc-p1",
    )
    loose_targets = build_one_sided_candidates(
        real,
        bad,
        new_features,
        alphas=(0.15, 0.20, 0.25, 0.30),
        min_coverage=min_coverage,
        min_rejection=0.02,
        min_real_satisfaction=0.50,
        origin="bvloc-p1",
    )
    guarded = build_guarded_candidates(
        real,
        bad,
        [*new_one_sided, *new_bands, *loose_targets],
        guard_columns=(
            "mean_cn_cat",
            "z_cat_max",
            "cn_an_mean",
            "n_el",
            "cat_an_ratio",
            "fi",
            "dchi",
        ),
        guard_quantiles=(0.25, 0.5, 0.75),
        min_real_satisfaction=guard_min_real_satisfaction,
        min_rejection=0.02,
        max_targets=max_guard_targets,
    )
    existing = _deduplicate(old_candidates)
    additive = _deduplicate(
        [*existing, *new_one_sided, *new_bands, *guarded]
    )
    counts = {
        "old_features": len(old_features),
        "new_features_eligible": len(new_features),
        "old_candidates": len(existing),
        "new_one_sided_candidates": len(new_one_sided),
        "new_band_candidates": len(new_bands),
        "new_guarded_candidates": len(guarded),
        "combined_candidates": len(additive),
    }
    return {
        "existing_loop": existing,
        "additive_bvloc_loop": additive,
    }, counts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the additive PRIS law loop without touching old outputs."
    )
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--real-descriptors", type=Path, required=True)
    parser.add_argument("--bad-descriptors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--floors", type=float, nargs="+", default=[0.99, 0.98, 0.95])
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--max-rules", type=int, default=12)
    parser.add_argument("--max-guard-targets", type=int, default=100)
    parser.add_argument(
        "--p1-vocabulary",
        choices=("frozen", "expanded"),
        default="frozen",
        help="Use the pre-outcome P1 vocabulary or a separately labelled extension.",
    )
    parser.add_argument(
        "--paired-anion-guard",
        action="store_true",
        help="Require each discovery anion stratum to stay within 0.01 of the paired existing-loop result.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    if any(not 0 < floor < 1 for floor in args.floors):
        raise SystemExit("all --floors must lie strictly between 0 and 1")

    started = time.time()
    dr, cr, db, cb = _load_search_frames(
        args.features_dir,
        args.real_descriptors,
        args.bad_descriptors,
    )
    source_access_audit = dict(dr.attrs["source_access_audit"])
    print(
        f"discovery real/bad {len(dr):,}/{len(db):,}; "
        f"calibration real/bad {len(cr):,}/{len(cb):,}; "
        "fit/evaluation lockbox rows absent; monolithic source did materialize "
        f"{source_access_audit['materialized_lockbox_real_rows']:,} real and "
        f"{source_access_audit['materialized_lockbox_bad_rows']:,} bad lockbox rows",
        flush=True,
    )
    candidate_sets, candidate_counts = build_candidate_sets(
        dr,
        db,
        min_coverage=args.min_coverage,
        max_guard_targets=args.max_guard_targets,
        guard_min_real_satisfaction=min(args.floors),
        p1_vocabulary=args.p1_vocabulary,
    )
    old_candidates = candidate_sets["existing_loop"]
    additive_candidates = candidate_sets["additive_bvloc_loop"]
    print(
        f"candidates existing={candidate_counts['old_candidates']:,}; "
        f"new one-sided={candidate_counts['new_one_sided_candidates']:,}, "
        f"bands={candidate_counts['new_band_candidates']:,}, "
        f"guarded={candidate_counts['new_guarded_candidates']:,}; "
        f"combined={candidate_counts['combined_candidates']:,}",
        flush=True,
    )

    report: dict[str, object] = {
        "protocol": {
            "selection_split": "discovery only",
            "calibration_role": "historical diagnostic; previously adaptively reused",
            **source_access_audit,
            "missing_feature_offline_semantics": "pass/abstain",
            "bad_weighting": ["pooled", "parent-group-equal", "per-kind"],
            "candidate_threshold_source": "real discovery quantiles only",
            "floors": args.floors,
            "min_coverage": args.min_coverage,
            "p1_vocabulary": args.p1_vocabulary,
        },
        "counts": {
            "discovery_real": len(dr),
            "discovery_bad": len(db),
            "calibration_real": len(cr),
            "calibration_bad": len(cb),
            **candidate_counts,
        },
        "historical": {
            "discovery": _historical_metrics(dr, db),
            "calibration": _historical_metrics(cr, cb),
        },
        "frontiers": {},
    }
    variant_results: dict[str, dict[str, tuple[list[LawCandidate], BeamResult]]] = {}
    for variant, candidates in (
        ("existing_loop", old_candidates),
        ("additive_bvloc_loop", additive_candidates),
    ):
        variant_results[variant] = {}
        report["frontiers"][variant] = {}
        for floor in args.floors:
            print(f"search {variant} floor={floor:.3f}", flush=True)
            result = pareto_beam(
                candidates,
                real_size=len(dr),
                bad_size=len(db),
                bad_kinds=db["kind"].to_numpy(),
                satisfaction_floor=floor,
                max_rules=args.max_rules,
                width=args.width,
            )
            discovery = _evaluate_result(
                dr,
                db,
                candidates,
                result,
                use_stored_masks=True,
            )
            calibration = _evaluate_result(
                cr,
                cb,
                candidates,
                result,
                use_stored_masks=False,
            )
            calibration_real_mask = _apply_result(cr, candidates, result)
            calibration_bad_mask = _apply_result(cb, candidates, result)
            entry = {
                "rules": [
                    _candidate_record(candidates[index]) for index in result.indices
                ],
                "discovery": discovery,
                "calibration": calibration,
                "discovery_fold_diagnostic": _fold_diagnostics(
                    dr,
                    db,
                    result.real_mask,
                    result.bad_mask,
                ),
                "calibration_fold_diagnostic": _fold_diagnostics(
                    cr,
                    cb,
                    calibration_real_mask,
                    calibration_bad_mask,
                ),
            }
            report["frontiers"][variant][str(floor)] = entry
            variant_results[variant][str(floor)] = (candidates, result)
            print(
                f"  discovery sat/rej={discovery['satisfaction']:.4f}/"
                f"{discovery['rejection']:.4f}; calibration="
                f"{calibration['satisfaction']:.4f}/{calibration['rejection']:.4f}; "
                f"N={len(result.indices)}",
                flush=True,
            )

    if args.paired_anion_guard:
        variant = "additive_bvloc_anion_guarded_loop"
        candidates = additive_candidates
        variant_results[variant] = {}
        report["frontiers"][variant] = {}
        anion_counts = dr["anion"].value_counts()
        strata = {
            str(anion): dr["anion"].eq(anion).to_numpy()
            for anion in anion_counts[anion_counts >= 200].index
        }
        for floor in args.floors:
            key = str(floor)
            base_result = variant_results["existing_loop"][key][1]
            paired_floors = {
                anion: max(float(base_result.real_mask[mask].mean()) - 0.01, 0.0)
                for anion, mask in strata.items()
            }
            print(
                f"search {variant} floor={floor:.3f} "
                f"with {len(strata)} paired anion constraints",
                flush=True,
            )
            result = pareto_beam(
                candidates,
                real_size=len(dr),
                bad_size=len(db),
                bad_kinds=db["kind"].to_numpy(),
                satisfaction_floor=floor,
                max_rules=args.max_rules,
                width=args.width,
                real_strata=strata,
                stratum_floors=paired_floors,
            )
            discovery = _evaluate_result(
                dr,
                db,
                candidates,
                result,
                use_stored_masks=True,
            )
            calibration = _evaluate_result(
                cr,
                cb,
                candidates,
                result,
                use_stored_masks=False,
            )
            calibration_real_mask = _apply_result(cr, candidates, result)
            calibration_bad_mask = _apply_result(cb, candidates, result)
            report["frontiers"][variant][key] = {
                "rules": [
                    _candidate_record(candidates[index]) for index in result.indices
                ],
                "discovery": discovery,
                "calibration": calibration,
                "paired_discovery_anion_floors": paired_floors,
                "discovery_fold_diagnostic": _fold_diagnostics(
                    dr,
                    db,
                    result.real_mask,
                    result.bad_mask,
                ),
                "calibration_fold_diagnostic": _fold_diagnostics(
                    cr,
                    cb,
                    calibration_real_mask,
                    calibration_bad_mask,
                ),
            }
            variant_results[variant][key] = (candidates, result)
            print(
                f"  discovery sat/rej={discovery['satisfaction']:.4f}/"
                f"{discovery['rejection']:.4f}; calibration="
                f"{calibration['satisfaction']:.4f}/{calibration['rejection']:.4f}; "
                f"N={len(result.indices)}",
                flush=True,
            )

    comparison_variant = (
        "additive_bvloc_anion_guarded_loop"
        if args.paired_anion_guard
        else "additive_bvloc_loop"
    )
    comparisons = {}
    for floor in args.floors:
        key = str(floor)
        base = report["frontiers"]["existing_loop"][key]["calibration"]
        additive = report["frontiers"][comparison_variant][key]["calibration"]
        base_min = min(base["by_kind"].values())
        additive_min = min(additive["by_kind"].values())
        shared_anions = set(base["by_anion"]).intersection(additive["by_anion"])
        worst_anion_delta = min(
            (
                additive["by_anion"][anion]["satisfaction"]
                - base["by_anion"][anion]["satisfaction"]
                for anion in shared_anions
            ),
            default=0.0,
        )
        selected_rules = report["frontiers"][comparison_variant][key]["rules"]
        new_selected = any(rule["origin"] == "bvloc-p1" for rule in selected_rules)
        coverage = selected_rule_coverage(
            pd.concat([dr, cr], ignore_index=True),
            pd.concat([db, cb], ignore_index=True),
            selected_rules,
        )
        metric_gate = law_preliminary_gate(
            new_descriptor_selected=new_selected,
            additive_satisfaction=additive["satisfaction"],
            base_satisfaction=base["satisfaction"],
            rejection_delta=additive["rejection"] - base["rejection"],
            min_kind_rejection_delta=additive_min - base_min,
            worst_anion_delta=worst_anion_delta,
            real_coverage=float(coverage["real_min"]),
            bad_coverage=float(coverage["bad_min"]),
            source_tables_materialized_lockbox_rows=False,
        )
        comparisons[key] = {
            "calibration_satisfaction_delta": (
                additive["satisfaction"] - base["satisfaction"]
            ),
            "calibration_rejection_delta": additive["rejection"] - base["rejection"],
            "calibration_group_equal_rejection_delta": (
                additive["group_equal_rejection"] - base["group_equal_rejection"]
            ),
            "calibration_min_kind_rejection_delta": additive_min - base_min,
            "worst_shared_anion_satisfaction_delta": worst_anion_delta,
            "new_descriptor_selected": new_selected,
            "selected_rule_coverage": coverage,
            "metric_gate_without_source_access_condition": metric_gate,
            "preliminary_gate": law_preliminary_gate(
                new_descriptor_selected=new_selected,
                additive_satisfaction=additive["satisfaction"],
                base_satisfaction=base["satisfaction"],
                rejection_delta=additive["rejection"] - base["rejection"],
                min_kind_rejection_delta=additive_min - base_min,
                worst_anion_delta=worst_anion_delta,
                real_coverage=float(coverage["real_min"]),
                bad_coverage=float(coverage["bad_min"]),
                source_tables_materialized_lockbox_rows=bool(
                    source_access_audit[
                        "source_tables_materialized_all_splits"
                    ]
                    and (
                        source_access_audit["materialized_lockbox_real_rows"] > 0
                        or source_access_audit["materialized_lockbox_bad_rows"] > 0
                    )
                ),
            ),
            "status": "pending false-positive and perturbation-lineage falsification",
        }
    report["comparisons"] = comparisons
    report["comparison_variant"] = comparison_variant
    report["provenance"] = {
        "runtime_seconds": time.time() - started,
        "input_sha256": {
            "real_descriptors": _hash_file(args.real_descriptors),
            "bad_descriptors": _hash_file(args.bad_descriptors),
            "real_descriptor_metadata": _hash_file(
                args.real_descriptors.with_suffix(
                    args.real_descriptors.suffix + ".meta.json"
                )
            ),
            "bad_descriptor_metadata": _hash_file(
                args.bad_descriptors.with_suffix(
                    args.bad_descriptors.suffix + ".meta.json"
                )
            ),
        },
        "implementation_sha256": _hash_file(Path(__file__)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
