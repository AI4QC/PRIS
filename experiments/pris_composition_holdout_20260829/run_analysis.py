#!/usr/bin/env python3
"""Run the frozen PRIS composition-held-out sensitivity analysis."""

from __future__ import annotations
import os

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import (
    LAWSET_NAMES,
    attach_identity_and_novelty,
    cluster_bootstrap_matrix,
    cohort_masks,
    law_masks,
    load_analysis_tables,
    metric_estimates,
)


DEFAULT_FEATURE_ROOT = Path(os.environ.get("PRIS_FEATURES", "features/"))
DEFAULT_LAW_ROOT = Path(os.environ.get("PRIS_LAW_TABLES", "law_tables/"))
DEFAULT_REPLICATES = 10_000
DEFAULT_SEED = 20260829
COHORT_ORDER = (
    "heldout_all",
    "composition_shared",
    "composition_unseen",
    "chemical_system_unseen",
)
FEATURES = (
    "bl_min",
    "bl_mean",
    "cn_an_mean",
    "madz_range",
    "mad_max",
    "frac_like_bonds",
    "fi",
    "wyckoff_econ_001",
    "bv_rel_mean",
)
PUBLISHED_SATISFACTION = {
    "Set 1": 0.9918821974702662,
    "Set 1-prime": 0.9894279781008116,
    "Set 2": 0.9579006985085896,
    "Set 3": 0.917122899754578,
    "Set 4": 0.8180101944496885,
}
PUBLISHED_DETECTION = {
    "Set 1": 0.2890365448504983,
    "Set 1-prime": 0.3837209302325581,
    "Set 2": 0.6121262458471761,
    "Set 3": 0.7004429678848283,
    "Set 4": 0.9111295681063123,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cluster_unit(cohort: str) -> str:
    return "chemical_system_key" if cohort == "chemical_system_unseen" else "composition_key"


def _cohort_counts(
    real: pd.DataFrame,
    bad: pd.DataFrame,
    real_cohorts: dict[str, pd.Series],
    bad_cohorts: dict[str, pd.Series],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for population, frame, masks in (
        ("experimental", real, real_cohorts),
        ("chemically_damaged", bad, bad_cohorts),
    ):
        heldout_n = int(masks["heldout_all"].sum())
        for cohort in COHORT_ORDER:
            subset = frame.loc[masks[cohort]]
            rows.append(
                {
                    "population": population,
                    "cohort": cohort,
                    "n_rows": len(subset),
                    "n_parents": subset.parent.nunique() if "parent" in subset else np.nan,
                    "n_compositions": subset.composition_key.nunique(),
                    "n_chemical_systems": subset.chemical_system_key.nunique(),
                    "fraction_of_heldout_rows": len(subset) / heldout_n,
                }
            )
    return pd.DataFrame(rows)


def _evaluate_endpoint(
    frame: pd.DataFrame,
    masks: pd.DataFrame,
    cohorts: dict[str, pd.Series],
    *,
    endpoint: str,
    invert: bool,
    parent_column: str | None,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort in COHORT_ORDER:
        selected = cohorts[cohort]
        subset = frame.loc[selected].copy()
        success = masks.loc[selected, list(LAWSET_NAMES)].copy()
        if invert:
            success = ~success
        group_column = _cluster_unit(cohort)
        intervals = cluster_bootstrap_matrix(
            subset,
            success,
            group_column=group_column,
            parent_column=parent_column,
            replicates=replicates,
            seed=seed,
        )
        for lawset in LAWSET_NAMES:
            estimate = metric_estimates(
                subset,
                success[lawset].to_numpy(),
                group_column=group_column,
                parent_column=parent_column,
            )
            interval = intervals.loc[lawset]
            rows.append(
                {
                    "endpoint": endpoint,
                    "cohort": cohort,
                    "lawset": lawset,
                    "cluster_unit": group_column,
                    "n_rows": estimate["n_rows"],
                    "n_parents": subset[parent_column].nunique() if parent_column else np.nan,
                    "n_groups": estimate["n_groups"],
                    "n_success": estimate["n_success"],
                    "estimate_micro": estimate["estimate_micro"],
                    "micro_ci_low": interval.micro_ci_low,
                    "micro_ci_high": interval.micro_ci_high,
                    "estimate_group_equal": estimate["estimate_composition_equal"],
                    "group_equal_ci_low": interval.group_equal_ci_low,
                    "group_equal_ci_high": interval.group_equal_ci_high,
                }
            )
    return pd.DataFrame(rows)


def _evaluate_set4_per_class(
    bad: pd.DataFrame,
    bad_masks: pd.DataFrame,
    cohorts: dict[str, pd.Series],
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    detection = ~bad_masks["Set 4"]
    for cohort in COHORT_ORDER:
        group_column = _cluster_unit(cohort)
        for kind in ("S1", "S2", "S3", "S4", "S5"):
            selected = cohorts[cohort] & bad.kind.eq(kind)
            subset = bad.loc[selected].copy()
            success = detection.loc[selected].to_numpy()
            estimate = metric_estimates(
                subset,
                success,
                group_column=group_column,
                parent_column="parent",
            )
            interval = cluster_bootstrap_matrix(
                subset,
                pd.DataFrame({"Set 4": success}, index=subset.index),
                group_column=group_column,
                parent_column="parent",
                replicates=replicates,
                seed=seed,
            ).loc["Set 4"]
            rows.append(
                {
                    "cohort": cohort,
                    "damage_class": kind,
                    "cluster_unit": group_column,
                    "n_rows": estimate["n_rows"],
                    "n_parents": subset.parent.nunique(),
                    "n_groups": estimate["n_groups"],
                    "n_detected": estimate["n_success"],
                    "estimate_micro": estimate["estimate_micro"],
                    "micro_ci_low": interval.micro_ci_low,
                    "micro_ci_high": interval.micro_ci_high,
                    "estimate_group_equal": estimate["estimate_composition_equal"],
                    "group_equal_ci_low": interval.group_equal_ci_low,
                    "group_equal_ci_high": interval.group_equal_ci_high,
                }
            )
    return pd.DataFrame(rows)


def _feature_coverage(
    real: pd.DataFrame,
    bad: pd.DataFrame,
    real_cohorts: dict[str, pd.Series],
    bad_cohorts: dict[str, pd.Series],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for population, frame, cohorts in (
        ("experimental", real, real_cohorts),
        ("chemically_damaged", bad, bad_cohorts),
    ):
        for cohort in COHORT_ORDER:
            subset = frame.loc[cohorts[cohort]]
            for feature in FEATURES:
                finite = np.isfinite(subset[feature].to_numpy(dtype=float))
                rows.append(
                    {
                        "population": population,
                        "cohort": cohort,
                        "feature": feature,
                        "n_rows": len(subset),
                        "n_finite": int(finite.sum()),
                        "coverage": float(finite.mean()),
                    }
                )
    return pd.DataFrame(rows)


def _assert_published_reproduction(metrics: pd.DataFrame) -> None:
    heldout = metrics[metrics.cohort.eq("heldout_all")]
    for endpoint, expected in (
        ("experimental_satisfaction", PUBLISHED_SATISFACTION),
        ("damage_detection", PUBLISHED_DETECTION),
    ):
        observed = heldout[heldout.endpoint.eq(endpoint)].set_index("lawset").estimate_micro
        for lawset, target in expected.items():
            if not np.isclose(observed[lawset], target, rtol=0, atol=5e-13):
                raise AssertionError(
                    f"published reproduction failed for {endpoint}/{lawset}: "
                    f"{observed[lawset]} != {target}"
                )


def _write_results_markdown(
    output: Path,
    counts: pd.DataFrame,
    metrics: pd.DataFrame,
    per_class: pd.DataFrame,
    replicates: int,
    seed: int,
) -> None:
    real_counts = counts[counts.population.eq("experimental")].set_index("cohort")
    bad_counts = counts[counts.population.eq("chemically_damaged")].set_index("cohort")
    l4 = metrics[metrics.lawset.eq("Set 4")].set_index(["endpoint", "cohort"])

    def pct(value: float) -> str:
        return f"{100 * value:.2f}%"

    lines = [
        "# PRIS composition-held-out sensitivity: results",
        "",
        "The frozen PRIS rules and thresholds were evaluated without any refitting. "
        "Only the physically isolated discovery/calibration law tables were read; "
        "lockbox and unlabeled structures were not part of this analysis.",
        "",
        "## Overlap audit",
        "",
        f"Of 5,297 held-out experimental structures, {int(real_counts.loc['composition_shared','n_rows']):,} "
        f"({pct(real_counts.loc['composition_shared','fraction_of_heldout_rows'])}) have a reduced "
        f"composition seen in discovery and {int(real_counts.loc['composition_unseen','n_rows']):,} "
        f"({pct(real_counts.loc['composition_unseen','fraction_of_heldout_rows'])}) do not. "
        f"The corresponding damaged-structure counts are {int(bad_counts.loc['composition_shared','n_rows']):,} "
        f"and {int(bad_counts.loc['composition_unseen','n_rows']):,}.",
        "",
        "## Frozen Set 4",
        "",
        "| Cohort | Experimental satisfaction | Damage detection |",
        "|---|---:|---:|",
    ]
    for cohort, label in (
        ("heldout_all", "All held-out"),
        ("composition_shared", "Composition shared"),
        ("composition_unseen", "Composition unseen"),
        ("chemical_system_unseen", "Chemical system unseen"),
    ):
        sat = l4.loc[("experimental_satisfaction", cohort)]
        det = l4.loc[("damage_detection", cohort)]
        lines.append(
            f"| {label} | {pct(sat.estimate_micro)} "
            f"[{pct(sat.micro_ci_low)}, {pct(sat.micro_ci_high)}] | "
            f"{pct(det.estimate_micro)} [{pct(det.micro_ci_low)}, {pct(det.micro_ci_high)}] |"
        )
    lines.extend(
        [
            "",
            "The composition-unseen point estimates do not decrease relative to the "
            "composition-shared subset: Set 4 satisfaction changes by "
            f"{100 * (l4.loc[('experimental_satisfaction','composition_unseen')].estimate_micro - l4.loc[('experimental_satisfaction','composition_shared')].estimate_micro):+.2f} "
            "percentage points and damage detection by "
            f"{100 * (l4.loc[('damage_detection','composition_unseen')].estimate_micro - l4.loc[('damage_detection','composition_shared')].estimate_micro):+.2f} percentage points.",
            "",
            "## Set 4 damage classes on unseen compositions",
            "",
            "| Class | n | Detection |",
            "|---|---:|---:|",
        ]
    )
    unseen_class = per_class[per_class.cohort.eq("composition_unseen")]
    for row in unseen_class.itertuples():
        lines.append(
            f"| {row.damage_class} | {row.n_rows:,} | {pct(row.estimate_micro)} "
            f"[{pct(row.micro_ci_low)}, {pct(row.micro_ci_high)}] |"
        )
    lines.extend(
        [
            "",
            "Intervals are percentile intervals from whole-cluster resampling. "
            f"The analysis used {replicates:,} replicates with seed {seed}; composition "
            "is the cluster except for the chemical-system-unseen cohort, where the "
            "exact element set is the cluster.",
            "",
            "This is an outcome-blind subgroup sensitivity analysis of the "
            "existing calibration partition, not a newly collected external holdout. "
            "`Chemical system` means the exact sorted element set; no broader, post-hoc "
            "notion of chemical family is claimed.",
            "",
            "Missing frozen-law features retain the published convention of counting as "
            "satisfied. Feature coverage is therefore reported separately in "
            "`results/feature_coverage.csv`.",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--law-root", type=Path, default=DEFAULT_LAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    results_dir = output / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    real, bad, provenance = load_analysis_tables(args.feature_root, args.law_root)
    real, bad = attach_identity_and_novelty(real, bad, provenance)
    real_cohorts = cohort_masks(real, split_column="split")
    bad_cohorts = cohort_masks(bad, split_column="psplit")
    real_masks, bad_masks = law_masks(real), law_masks(bad)

    counts = _cohort_counts(real, bad, real_cohorts, bad_cohorts)
    metrics = pd.concat(
        [
            _evaluate_endpoint(
                real,
                real_masks,
                real_cohorts,
                endpoint="experimental_satisfaction",
                invert=False,
                parent_column=None,
                replicates=args.bootstrap_replicates,
                seed=args.seed,
            ),
            _evaluate_endpoint(
                bad,
                bad_masks,
                bad_cohorts,
                endpoint="damage_detection",
                invert=True,
                parent_column="parent",
                replicates=args.bootstrap_replicates,
                seed=args.seed,
            ),
        ],
        ignore_index=True,
    )
    _assert_published_reproduction(metrics)
    per_class = _evaluate_set4_per_class(
        bad,
        bad_masks,
        bad_cohorts,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    coverage = _feature_coverage(real, bad, real_cohorts, bad_cohorts)

    counts.to_csv(results_dir / "cohort_counts.csv", index=False)
    metrics.to_csv(results_dir / "metrics.csv", index=False)
    per_class.to_csv(results_dir / "set4_per_damage_class.csv", index=False)
    coverage.to_csv(results_dir / "feature_coverage.csv", index=False)

    inputs = [
        args.law_root / "law_real.parquet",
        args.law_root / "law_bad.parquet",
        args.feature_root / "law_real_aug.parquet",
        args.feature_root / "law_bad_aug.parquet",
        args.feature_root / "provenance.parquet",
    ]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {
        "analysis": "frozen PRIS composition-held-out sensitivity",
        "git_commit": commit,
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "interval": "two-sided 95% percentile cluster bootstrap",
        },
        "cohorts": {
            "heldout_all": "all calibration rows",
            "composition_shared": "calibration reduced composition present in discovery",
            "composition_unseen": "calibration reduced composition absent from discovery",
            "chemical_system_unseen": "calibration exact element set absent from discovery",
        },
        "missing_feature_convention": "missing counts as satisfying, matching frozen publication evaluation",
        "thresholds_frozen": True,
        "rule_search_or_refit": False,
        "input_files": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in inputs
        ],
    }
    (results_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    _write_results_markdown(
        output,
        counts,
        metrics,
        per_class,
        args.bootstrap_replicates,
        args.seed,
    )

    l4 = metrics[metrics.lawset.eq("Set 4")][
        ["endpoint", "cohort", "n_rows", "estimate_micro", "micro_ci_low", "micro_ci_high"]
    ]
    print(l4.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
