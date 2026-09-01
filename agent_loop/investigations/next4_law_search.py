#!/usr/bin/env python3
"""Robust anion-aware additive law search for np-next-20260801d."""

from __future__ import annotations

import argparse
import heapq
import hashlib
from itertools import count
import json
from pathlib import Path
import time
from typing import Mapping, Sequence
import zlib

import numpy as np
import pandas as pd

from better_law_search import (
    BeamResult,
    LawCandidate,
    _apply_result,
    _candidate_record,
    _deduplicate,
    _evaluate_result,
    build_band_candidates,
    build_guarded_candidates,
    build_one_sided_candidates,
    evaluate_masks,
    law_preliminary_gate,
    pareto_beam,
    selected_rule_coverage,
)
from next3_law_search import (
    NEXT3_GUARD_COLUMNS,
    build_next3_candidate_sets,
    load_next3_search_frames,
)

CORRECTED_SEARCH_FEATURES = frozenset(
    [
        f"p2c_{charge}_{metric}_{aggregate}"
        for charge in ("cat", "an")
        for metric in ("sa_effective_cn", "sa_like_fraction", "sa_max_fraction")
        for aggregate in ("mean", "q95", "max")
    ]
    + [
        f"p6c_{charge}_{metric}_{aggregate}"
        for charge in ("cat", "an")
        for metric in ("gap_ratio", "shell_width", "gap_pos")
        for aggregate in ("mean", "max")
    ]
    + [
        "p9c_bond_mismatch_mean",
        "p9c_bond_mismatch_q95",
        "p9c_bond_mismatch_max",
        "p9c_cat_site_mismatch_max",
    ]
)
TAINTED_FEATURE_PREFIXES = ("p2vor_", "p6gap_", "p7poly_", "p9lew_")
CORRECTED_PREFIXES = ("p2c_", "p6c_", "p7c_", "p9c_")
NEXT4_GENERAL_GUARD_COLUMNS = tuple(
    column for column in NEXT3_GUARD_COLUMNS if column != "p7poly_an_contact_min"
)


def _fixed_guard_mask(
    values: np.ndarray,
    *,
    side: str,
    threshold: float,
) -> np.ndarray:
    finite = np.isfinite(values)
    if side == "hi":
        return finite & (values > threshold)
    if side == "lo":
        return finite & (values <= threshold)
    raise ValueError(f"unsupported guard side: {side}")


def build_fixed_guard_candidates(
    real: pd.DataFrame,
    bad: pd.DataFrame,
    targets: Sequence[LawCandidate],
) -> list[LawCandidate]:
    """Add only the two pre-frozen mechanism guards for next4."""

    guarded: list[LawCandidate] = []
    for target in targets:
        if target.guard_feature is not None:
            continue
        guard: tuple[str, str, float] | None = None
        if target.feature.startswith("p2c_") and "sa_like_fraction" in target.feature:
            guard = ("p7c_an_short_contact_frac", "lo", 0.0)
        elif target.feature.startswith("bvloc_"):
            guard = ("bvloc_parameter_exact_fraction", "hi", 0.9)
        if guard is None:
            continue
        guard_feature, guard_side, guard_threshold = guard
        if guard_feature not in real or guard_feature not in bad:
            raise ValueError(f"missing fixed guard column: {guard_feature}")
        active_real = _fixed_guard_mask(
            real[guard_feature].to_numpy(dtype=float),
            side=guard_side,
            threshold=guard_threshold,
        )
        active_bad = _fixed_guard_mask(
            bad[guard_feature].to_numpy(dtype=float),
            side=guard_side,
            threshold=guard_threshold,
        )
        real_mask = (~active_real) | target.real_mask
        bad_mask = (~active_bad) | target.bad_mask
        guarded.append(
            LawCandidate(
                description=(
                    f"if {guard_feature} {guard_side} {guard_threshold:.8g} "
                    f"then ({target.description})"
                ),
                feature=target.feature,
                family=f"fixed-guarded-{target.family}",
                origin="next4-fixed-guard",
                side=target.side,
                thresholds=target.thresholds,
                real_mask=real_mask,
                bad_mask=bad_mask,
                real_coverage=target.real_coverage,
                bad_coverage=target.bad_coverage,
                guard_feature=guard_feature,
                guard_side=guard_side,
                guard_threshold=guard_threshold,
            )
        )
    return guarded


def build_next4_candidate_sets(
    real: pd.DataFrame,
    bad: pd.DataFrame,
    *,
    min_coverage: float,
    max_guard_targets: int,
    guard_min_real_satisfaction: float = 0.95,
    p1_vocabulary: str = "frozen",
) -> tuple[dict[str, list[LawCandidate]], dict[str, int]]:
    """Substitute corrected P2/P6/P9 candidates and add fixed P7/P1 guards."""

    prior, prior_counts = build_next3_candidate_sets(
        real,
        bad,
        min_coverage=min_coverage,
        max_guard_targets=max_guard_targets,
        guard_min_real_satisfaction=guard_min_real_satisfaction,
        p1_vocabulary=p1_vocabulary,
    )

    def retained(candidate: LawCandidate) -> bool:
        if candidate.feature.startswith(TAINTED_FEATURE_PREFIXES):
            return False
        if candidate.feature.startswith(CORRECTED_PREFIXES):
            return False
        if candidate.guard_feature == "p7poly_an_contact_min":
            return False
        return True

    existing = _deduplicate(
        [candidate for candidate in prior["existing_loop"] if retained(candidate)]
    )
    prior_additive = _deduplicate(
        [
            candidate
            for candidate in prior["additive_bvloc_loop"]
            if retained(candidate)
        ]
    )
    corrected_features = [
        feature
        for feature in sorted(CORRECTED_SEARCH_FEATURES)
        if feature in real and feature in bad
    ]
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
    one_sided = build_one_sided_candidates(
        real,
        bad,
        corrected_features,
        alphas=alphas,
        min_coverage=min_coverage,
        min_rejection=0.01,
        min_real_satisfaction=0.88,
        origin="next4-corrected",
    )
    bands = build_band_candidates(
        real,
        bad,
        corrected_features,
        central_coverages=(0.99, 0.98, 0.95, 0.90),
        min_coverage=min_coverage,
        min_rejection=0.01,
        min_real_satisfaction=0.88,
        origin="next4-corrected",
    )
    loose = build_one_sided_candidates(
        real,
        bad,
        corrected_features,
        alphas=(0.15, 0.20, 0.25, 0.30),
        min_coverage=min_coverage,
        min_rejection=0.02,
        min_real_satisfaction=0.50,
        origin="next4-corrected",
    )
    generally_guarded = build_guarded_candidates(
        real,
        bad,
        [*one_sided, *bands, *loose],
        guard_columns=NEXT4_GENERAL_GUARD_COLUMNS,
        guard_quantiles=(0.25, 0.5, 0.75),
        min_real_satisfaction=guard_min_real_satisfaction,
        min_rejection=0.02,
        max_targets=max_guard_targets,
    )
    fixed_targets = [
        *one_sided,
        *bands,
        *loose,
        *[
            candidate
            for candidate in prior_additive
            if candidate.feature.startswith("bvloc_")
            and candidate.guard_feature is None
        ],
    ]
    fixed_guarded = [
        candidate
        for candidate in build_fixed_guard_candidates(real, bad, fixed_targets)
        if float(candidate.real_mask.mean()) >= guard_min_real_satisfaction
        and 1.0 - float(candidate.bad_mask.mean()) >= 0.02
    ]
    additive = _deduplicate(
        [
            *prior_additive,
            *one_sided,
            *bands,
            *generally_guarded,
            *fixed_guarded,
        ]
    )
    counts = {
        **{f"prior_{key}": value for key, value in prior_counts.items()},
        "existing_candidates": len(existing),
        "corrected_features_eligible": len(corrected_features),
        "corrected_one_sided_candidates": len(one_sided),
        "corrected_band_candidates": len(bands),
        "corrected_guarded_candidates": len(generally_guarded),
        "fixed_guarded_candidates": len(fixed_guarded),
        "combined_candidates": len(additive),
    }
    return {
        "existing_loop": existing,
        "additive_corrected_loop": additive,
    }, counts


def deterministic_real_folds(
    frame: pd.DataFrame,
    *,
    n_folds: int = 4,
    source_column: str = "source_id",
) -> np.ndarray:
    """Assign stable CRC32 folds without depending on row order."""

    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    if source_column not in frame:
        raise ValueError(f"missing source column: {source_column}")
    return np.asarray(
        [
            (zlib.crc32(str(source).encode("utf-8")) & 0x7FFFFFFF) % n_folds
            for source in frame[source_column]
        ],
        dtype=int,
    )


def paired_robust_strata(
    real: pd.DataFrame,
    baseline_mask: Sequence[bool],
    *,
    n_folds: int = 4,
    min_anion_rows: int = 200,
    min_cell_rows: int = 50,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, float],
    dict[str, dict[str, float | int | str]],
]:
    """Build full-anion and anion-by-fold floors paired to a baseline mask."""

    if "anion" not in real:
        raise ValueError("real frame must contain an anion column")
    baseline = np.asarray(baseline_mask, dtype=bool)
    if baseline.shape != (len(real),):
        raise ValueError("baseline_mask length does not match real frame")
    if min_anion_rows < 1 or min_cell_rows < 1:
        raise ValueError("minimum stratum sizes must be positive")

    folds = deterministic_real_folds(real, n_folds=n_folds)
    counts = real["anion"].value_counts(dropna=True)
    eligible = sorted(str(anion) for anion, n in counts.items() if n >= min_anion_rows)
    values = real["anion"].astype("string").to_numpy(dtype=object)
    strata: dict[str, np.ndarray] = {}
    floors: dict[str, float] = {}
    metadata: dict[str, dict[str, float | int | str]] = {}

    def add(name: str, mask: np.ndarray, *, anion: str, fold: int | None) -> None:
        n = int(mask.sum())
        if n < min_cell_rows:
            return
        satisfaction = float(baseline[mask].mean())
        strata[name] = mask
        floors[name] = satisfaction
        metadata[name] = {
            "anion": anion,
            "fold": "all" if fold is None else int(fold),
            "n": n,
            "baseline_satisfaction": satisfaction,
        }

    for anion in eligible:
        anion_mask = values == anion
        add(f"anion:{anion}", anion_mask, anion=anion, fold=None)
        for fold in range(n_folds):
            add(
                f"anion:{anion}:fold:{fold}",
                anion_mask & (folds == fold),
                anion=anion,
                fold=fold,
            )
    return strata, floors, metadata


def _kind_rejections(
    bad_mask: np.ndarray,
    bad_kinds: np.ndarray,
) -> tuple[float, float]:
    pooled = float((~bad_mask).mean())
    kind_values = [
        float((~bad_mask[bad_kinds == kind]).mean())
        for kind in sorted(set(bad_kinds.tolist()))
        if np.any(bad_kinds == kind)
    ]
    return pooled, min(kind_values, default=0.0)


def robust_pareto_beam(
    candidates: Sequence[LawCandidate],
    *,
    real_size: int,
    bad_size: int,
    bad_kinds: np.ndarray,
    satisfaction_floor: float,
    max_rules: int = 12,
    width: int = 96,
    min_gain: float = 0.0015,
    real_strata: Mapping[str, np.ndarray] | None = None,
    stratum_floors: Mapping[str, float] | None = None,
) -> BeamResult:
    """Beam search with hard real strata and worst-kind-first selection."""

    kinds = np.asarray(bad_kinds, dtype=object)
    if kinds.shape != (bad_size,):
        raise ValueError("bad_kinds length does not match bad_size")
    if width < 2 or max_rules < 1:
        raise ValueError("width must be >=2 and max_rules must be positive")
    strata = {
        name: np.asarray(mask, dtype=bool)
        for name, mask in dict(real_strata or {}).items()
    }
    floors = dict(stratum_floors or {})
    if set(strata) != set(floors):
        raise ValueError("real_strata and stratum_floors must have matching keys")
    for name, mask in strata.items():
        if mask.shape != (real_size,) or not mask.any():
            raise ValueError(f"invalid real stratum: {name}")

    initial = BeamResult(
        indices=(),
        real_mask=np.ones(real_size, dtype=bool),
        bad_mask=np.ones(bad_size, dtype=bool),
    )
    frontier = [initial]
    best: BeamResult | None = None

    def primary_key(state: BeamResult) -> tuple[object, ...]:
        pooled, minimum = _kind_rejections(state.bad_mask, kinds)
        return (
            minimum,
            pooled,
            float(state.real_mask.mean()),
            -len(state.indices),
            tuple(-index for index in state.indices),
        )

    def efficient_key(state: BeamResult) -> tuple[object, ...]:
        pooled, minimum = _kind_rejections(state.bad_mask, kinds)
        cost = max(1.0 - float(state.real_mask.mean()), 1e-8)
        return (
            minimum,
            pooled / cost,
            pooled,
            float(state.real_mask.mean()),
            tuple(-index for index in state.indices),
        )

    for _ in range(max_rules):
        serial = count()
        half = max(width // 2, 1)
        primary_heap: list[tuple[tuple[object, ...], int, BeamResult]] = []
        efficient_heap: list[tuple[tuple[object, ...], int, BeamResult]] = []

        def retain(heap, key, state, limit):
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
                if candidate.real_mask.shape != (real_size,) or candidate.bad_mask.shape != (bad_size,):
                    raise ValueError("candidate masks do not match requested row counts")
                real_mask = state.real_mask & candidate.real_mask
                if float(real_mask.mean()) < satisfaction_floor:
                    continue
                if any(
                    float(real_mask[mask].mean()) < float(floors[name])
                    for name, mask in strata.items()
                ):
                    continue
                bad_mask = state.bad_mask & candidate.bad_mask
                gain = float(state.bad_mask.mean() - bad_mask.mean())
                if gain <= min_gain:
                    continue
                expanded = BeamResult(
                    indices=(*state.indices, index),
                    real_mask=real_mask,
                    bad_mask=bad_mask,
                )
                retain(primary_heap, primary_key(expanded), expanded, half)
                retain(efficient_heap, efficient_key(expanded), expanded, width)

        if not primary_heap and not efficient_heap:
            break
        primary = [
            entry[2]
            for entry in sorted(primary_heap, key=lambda entry: entry[:2], reverse=True)
        ]
        primary_indices = {state.indices for state in primary}
        efficient = [
            entry[2]
            for entry in sorted(efficient_heap, key=lambda entry: entry[:2], reverse=True)
            if entry[2].indices not in primary_indices
        ][: max(width - len(primary), 0)]
        frontier = primary + efficient
        for state in frontier:
            if best is None or primary_key(state) > primary_key(best):
                best = state
    if best is None:
        raise ValueError("no candidate combination satisfies the robust constraints")
    return best


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_isolated_manifest(isolated_dir: Path) -> dict[str, object]:
    """Verify the two downstream source tables against the isolation manifest."""

    manifest_path = isolated_dir / "isolated_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified: dict[str, object] = {}
    for table_name, filename, split_column in (
        ("law_real", "law_real.parquet", "split"),
        ("law_bad", "law_bad.parquet", "psplit"),
    ):
        expected = manifest["tables"][table_name]
        path = isolated_dir / filename
        actual_hash = _sha256(path)
        if actual_hash != expected["sha256"]:
            raise ValueError(f"isolated manifest hash mismatch: {filename}")
        frame = pd.read_parquet(path, columns=[split_column])
        counts = {
            str(key): int(value)
            for key, value in frame[split_column].value_counts().sort_index().items()
        }
        if len(frame) != int(expected["rows"]) or counts != expected["split_counts"]:
            raise ValueError(f"isolated manifest row contract mismatch: {filename}")
        if any(str(key).lower() == "lockbox" for key in counts):
            raise ValueError(f"isolated table contains lockbox rows: {filename}")
        verified[table_name] = {
            "sha256": actual_hash,
            "rows": len(frame),
            "split_counts": counts,
        }
    return {
        "manifest_sha256": _sha256(manifest_path),
        "tables": verified,
        "downstream_lockbox_rows": 0,
    }


def load_next4_search_frames(
    isolated_dir: Path,
    real_descriptors: Path,
    bad_descriptors: Path,
    real_sixfam: Path,
    bad_sixfam: Path,
    real_corrected: Path,
    bad_corrected: Path,
    real_guards: Path,
    bad_guards: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge corrected descriptors into the already isolated next3 frames."""

    dr, cr, db, cb = load_next3_search_frames(
        isolated_dir,
        real_descriptors,
        bad_descriptors,
        real_sixfam,
        bad_sixfam,
        real_guards,
        bad_guards,
    )
    corrected_real = pd.read_parquet(real_corrected).drop(
        columns=["split", "next4_valence_source"], errors="ignore"
    )
    corrected_bad = pd.read_parquet(bad_corrected).drop(
        columns=["kind", "parent", "split", "next4_valence_source"],
        errors="ignore",
    )
    if corrected_real["source_id"].duplicated().any():
        raise ValueError("duplicate source_id in corrected real descriptors")
    if corrected_bad["sid"].duplicated().any():
        raise ValueError("duplicate sid in corrected bad descriptors")
    output = []
    for name, frame, corrected, key in (
        ("discovery real", dr, corrected_real, "source_id"),
        ("calibration real", cr, corrected_real, "source_id"),
        ("discovery bad", db, corrected_bad, "sid"),
        ("calibration bad", cb, corrected_bad, "sid"),
    ):
        before = len(frame)
        merged = frame.merge(corrected, on=key, how="left", validate="one_to_one")
        if len(merged) != before:
            raise AssertionError(f"corrected descriptor merge changed {name} rows")
        merged.attrs["source_access_audit"] = frame.attrs["source_access_audit"]
        output.append(merged)
    return tuple(output)


def _worst_anion_delta(base: Mapping[str, object], candidate: Mapping[str, object]) -> float:
    base_by = base.get("by_anion", {})
    candidate_by = candidate.get("by_anion", {})
    shared = set(base_by).intersection(candidate_by)
    return min(
        (
            float(candidate_by[anion]["satisfaction"])
            - float(base_by[anion]["satisfaction"])
            for anion in shared
        ),
        default=0.0,
    )


def _minimum_kind(metrics: Mapping[str, object]) -> float:
    return min((float(value) for value in metrics["by_kind"].values()), default=0.0)


def _fold_comparisons(
    real: pd.DataFrame,
    bad: pd.DataFrame,
    baseline_real: np.ndarray,
    baseline_bad: np.ndarray,
    candidate_real: np.ndarray,
    candidate_bad: np.ndarray,
    *,
    n_folds: int,
) -> list[dict[str, object]]:
    real_folds = deterministic_real_folds(real, n_folds=n_folds)
    bad_folds = np.asarray(
        [
            (zlib.crc32(str(parent).encode("utf-8")) & 0x7FFFFFFF) % n_folds
            for parent in bad["parent"]
        ],
        dtype=int,
    )
    rows = []
    for fold in range(n_folds):
        rm = real_folds == fold
        bm = bad_folds == fold
        if not rm.any() or not bm.any():
            continue
        base = evaluate_masks(
            real_mask=baseline_real[rm],
            bad_mask=baseline_bad[bm],
            bad_groups=bad.loc[bm, "parent"].to_numpy(),
            bad_kinds=bad.loc[bm, "kind"].to_numpy(),
        )
        candidate = evaluate_masks(
            real_mask=candidate_real[rm],
            bad_mask=candidate_bad[bm],
            bad_groups=bad.loc[bm, "parent"].to_numpy(),
            bad_kinds=bad.loc[bm, "kind"].to_numpy(),
        )
        rows.append(
            {
                "fold": fold,
                "n_real": int(rm.sum()),
                "n_bad": int(bm.sum()),
                "baseline": base,
                "candidate": candidate,
                "satisfaction_delta": float(
                    candidate["satisfaction"] - base["satisfaction"]
                ),
                "rejection_delta": float(candidate["rejection"] - base["rejection"]),
                "minimum_kind_delta": float(
                    _minimum_kind(candidate) - _minimum_kind(base)
                ),
            }
        )
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated-dir", type=Path, required=True)
    parser.add_argument("--real-descriptors", type=Path, required=True)
    parser.add_argument("--bad-descriptors", type=Path, required=True)
    parser.add_argument("--real-sixfam", type=Path, required=True)
    parser.add_argument("--bad-sixfam", type=Path, required=True)
    parser.add_argument("--real-corrected", type=Path, required=True)
    parser.add_argument("--bad-corrected", type=Path, required=True)
    parser.add_argument("--real-guards", type=Path, required=True)
    parser.add_argument("--bad-guards", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--floor", type=float, default=0.98)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--max-rules", type=int, default=12)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--max-guard-targets", type=int, default=100)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--min-anion-rows", type=int, default=200)
    parser.add_argument("--min-cell-rows", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    if not 0 < args.floor < 1:
        raise SystemExit("--floor must lie strictly between zero and one")
    started = time.time()
    isolation_audit = verify_isolated_manifest(args.isolated_dir)
    dr, cr, db, cb = load_next4_search_frames(
        args.isolated_dir,
        args.real_descriptors,
        args.bad_descriptors,
        args.real_sixfam,
        args.bad_sixfam,
        args.real_corrected,
        args.bad_corrected,
        args.real_guards,
        args.bad_guards,
    )
    candidate_sets, candidate_counts = build_next4_candidate_sets(
        dr,
        db,
        min_coverage=args.min_coverage,
        max_guard_targets=args.max_guard_targets,
        guard_min_real_satisfaction=args.floor,
    )
    existing_candidates = candidate_sets["existing_loop"]
    additive_candidates = candidate_sets["additive_corrected_loop"]

    baseline_result = pareto_beam(
        existing_candidates,
        real_size=len(dr),
        bad_size=len(db),
        bad_kinds=db["kind"].to_numpy(),
        satisfaction_floor=args.floor,
        max_rules=args.max_rules,
        width=24,
    )
    baseline_discovery = _evaluate_result(
        dr, db, existing_candidates, baseline_result, use_stored_masks=True
    )
    strata, floors, strata_metadata = paired_robust_strata(
        dr,
        baseline_result.real_mask,
        n_folds=args.n_folds,
        min_anion_rows=args.min_anion_rows,
        min_cell_rows=args.min_cell_rows,
    )
    robust_result = robust_pareto_beam(
        additive_candidates,
        real_size=len(dr),
        bad_size=len(db),
        bad_kinds=db["kind"].to_numpy(),
        satisfaction_floor=args.floor,
        max_rules=args.max_rules,
        width=args.width,
        real_strata=strata,
        stratum_floors=floors,
    )
    robust_discovery = _evaluate_result(
        dr, db, additive_candidates, robust_result, use_stored_masks=True
    )

    # Freeze the selected rule object and its canonical hash before applying it
    # to the adaptively reused calibration split.
    baseline_rules = [
        _candidate_record(existing_candidates[index])
        for index in baseline_result.indices
    ]
    selected_rules = [
        _candidate_record(additive_candidates[index])
        for index in robust_result.indices
    ]
    selected_rule_sha256 = hashlib.sha256(
        json.dumps(
            selected_rules,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    baseline_calibration = _evaluate_result(
        cr, cb, existing_candidates, baseline_result, use_stored_masks=False
    )
    robust_calibration = _evaluate_result(
        cr, cb, additive_candidates, robust_result, use_stored_masks=False
    )
    baseline_cal_real = _apply_result(cr, existing_candidates, baseline_result)
    baseline_cal_bad = _apply_result(cb, existing_candidates, baseline_result)
    robust_cal_real = _apply_result(cr, additive_candidates, robust_result)
    robust_cal_bad = _apply_result(cb, additive_candidates, robust_result)

    coverage = selected_rule_coverage(
        pd.concat([dr, cr], ignore_index=True),
        pd.concat([db, cb], ignore_index=True),
        selected_rules,
    )
    discovery_worst = _worst_anion_delta(baseline_discovery, robust_discovery)
    calibration_worst = _worst_anion_delta(
        baseline_calibration, robust_calibration
    )
    comparison = {
        "discovery_satisfaction_delta": float(
            robust_discovery["satisfaction"] - baseline_discovery["satisfaction"]
        ),
        "discovery_rejection_delta": float(
            robust_discovery["rejection"] - baseline_discovery["rejection"]
        ),
        "discovery_minimum_kind_delta": float(
            _minimum_kind(robust_discovery) - _minimum_kind(baseline_discovery)
        ),
        "discovery_worst_shared_anion_delta": discovery_worst,
        "calibration_satisfaction_delta": float(
            robust_calibration["satisfaction"] - baseline_calibration["satisfaction"]
        ),
        "calibration_rejection_delta": float(
            robust_calibration["rejection"] - baseline_calibration["rejection"]
        ),
        "calibration_minimum_kind_delta": float(
            _minimum_kind(robust_calibration) - _minimum_kind(baseline_calibration)
        ),
        "calibration_worst_shared_anion_delta": calibration_worst,
        "selected_rule_coverage": coverage,
        "calibration_metric_gate": law_preliminary_gate(
            new_descriptor_selected=any(
                str(rule["origin"]).startswith("next4") for rule in selected_rules
            ),
            additive_satisfaction=float(robust_calibration["satisfaction"]),
            base_satisfaction=float(baseline_calibration["satisfaction"]),
            rejection_delta=float(
                robust_calibration["rejection"] - baseline_calibration["rejection"]
            ),
            min_kind_rejection_delta=float(
                _minimum_kind(robust_calibration)
                - _minimum_kind(baseline_calibration)
            ),
            worst_anion_delta=calibration_worst,
            real_coverage=float(coverage["real_min"]),
            bad_coverage=float(coverage["bad_min"]),
            source_tables_materialized_lockbox_rows=False,
        ),
        "status": "pending true LOKO and all-295 unknown-fails-closed falsification",
    }
    report = {
        "protocol": {
            "experiment": "np-next-20260801d",
            "design": "docs/plans/2026-08-01-robust-anion-search.md",
            "selection_split": "discovery only",
            "calibration_role": "historical diagnostic; adaptively reused",
            "floor": args.floor,
            "baseline_width": 24,
            "robust_width": args.width,
            "max_rules": args.max_rules,
            "robust_constraint": "zero empirical satisfaction drop vs existing loop in each eligible full-anion and anion-by-fold discovery stratum",
            "ranking": "minimum perturbation-kind rejection, then pooled rejection",
            "missing_feature_offline_semantics": "pass/abstain; falsification is unknown-fails-closed",
            "lockbox_access": False,
        },
        "isolation_audit": isolation_audit,
        "counts": {
            "discovery_real": len(dr),
            "discovery_bad": len(db),
            "calibration_real": len(cr),
            "calibration_bad": len(cb),
            "robust_strata": len(strata),
            **candidate_counts,
        },
        "robust_strata": {
            name: {
                **strata_metadata[name],
                "candidate_satisfaction": float(robust_result.real_mask[mask].mean()),
                "candidate_minus_baseline": float(
                    robust_result.real_mask[mask].mean() - floors[name]
                ),
            }
            for name, mask in strata.items()
        },
        "frontiers": {
            "existing_loop": {
                str(args.floor): {
                    "rules": baseline_rules,
                    "discovery": baseline_discovery,
                    "calibration": baseline_calibration,
                }
            },
            "additive_corrected_robust_loop": {
                str(args.floor): {
                    "rules": selected_rules,
                    "rule_sha256": selected_rule_sha256,
                    "discovery": robust_discovery,
                    "calibration": robust_calibration,
                }
            },
        },
        "discovery_fold_comparisons": _fold_comparisons(
            dr,
            db,
            baseline_result.real_mask,
            baseline_result.bad_mask,
            robust_result.real_mask,
            robust_result.bad_mask,
            n_folds=args.n_folds,
        ),
        "calibration_masks_summary": {
            "baseline_real_pass": int(baseline_cal_real.sum()),
            "baseline_bad_pass": int(baseline_cal_bad.sum()),
            "candidate_real_pass": int(robust_cal_real.sum()),
            "candidate_bad_pass": int(robust_cal_bad.sum()),
        },
        "comparison": comparison,
        "comparison_variant": "additive_corrected_robust_loop",
        "provenance": {
            "runtime_seconds": time.time() - started,
            "input_sha256": {
                "real_descriptors": _sha256(args.real_descriptors),
                "bad_descriptors": _sha256(args.bad_descriptors),
                "real_sixfam": _sha256(args.real_sixfam),
                "bad_sixfam": _sha256(args.bad_sixfam),
                "real_corrected": _sha256(args.real_corrected),
                "bad_corrected": _sha256(args.bad_corrected),
                "real_guards": _sha256(args.real_guards),
                "bad_guards": _sha256(args.bad_guards),
            },
            "implementation_sha256": _sha256(Path(__file__)),
            "design_sha256": _sha256(
                Path(__file__).resolve().parents[1]
                / "docs/plans/2026-08-01-robust-anion-search.md"
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(
        f"selected {len(selected_rules)} rules; calibration sat delta "
        f"{comparison['calibration_satisfaction_delta']:+.4f}, rejection "
        f"{comparison['calibration_rejection_delta']:+.4f}, minimum-kind "
        f"{comparison['calibration_minimum_kind_delta']:+.4f}, anion "
        f"{comparison['calibration_worst_shared_anion_delta']:+.4f}",
        flush=True,
    )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
