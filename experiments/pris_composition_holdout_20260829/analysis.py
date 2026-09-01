#!/usr/bin/env python3
"""Frozen PRIS evaluation on composition- and chemical-system-unseen subsets."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pymatgen.core import Composition


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.next6_wbm_protocol import reduced_formula_key  # noqa: E402


LAWSET_NAMES = ("Set 1", "Set 1-prime", "Set 2", "Set 3", "Set 4")
ALLOWED_SPLITS = frozenset({"discovery", "calibration"})


def canonical_composition_key(formula: str) -> str:
    """Reduce exact stoichiometric multiples to one deterministic key."""

    return reduced_formula_key(str(formula))


def canonical_chemical_system(value: str) -> str:
    """Return a deterministic key for an exact element set."""

    symbols = [part.strip() for part in str(value).split("-") if part.strip()]
    if not symbols:
        raise ValueError(f"empty chemical system: {value!r}")
    return "-".join(sorted(set(symbols)))


def _assert_allowed_splits(frame: pd.DataFrame, column: str, label: str) -> None:
    values = set(frame[column].dropna().astype(str))
    if frame[column].isna().any() or not values <= ALLOWED_SPLITS:
        raise ValueError(
            f"{label} must contain discovery and calibration only; observed "
            f"{sorted(values)} with {int(frame[column].isna().sum())} missing"
        )


def _filtered_parquet(path: Path, key: str, allowed: Iterable[str]) -> pd.DataFrame:
    """Materialize only rows whose identity belongs to the allowed analysis set."""

    allowed_values = sorted(set(map(str, allowed)))
    if not allowed_values:
        raise ValueError(f"no allowed keys supplied for {path}")
    result = pd.read_parquet(path, filters=[(key, "in", allowed_values)])
    unexpected = set(result[key].astype(str)) - set(allowed_values)
    if unexpected:
        raise AssertionError(f"{path.name} returned keys outside the allowlist")
    return result


def load_analysis_tables(
    feature_root: Path,
    law_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the physically isolated PRIS discovery/calibration tables.

    The base law tables contain no lockbox rows. Augmented features and
    provenance are predicate-filtered by their identities before materializing.
    """

    feature_root = Path(feature_root)
    law_root = Path(law_root)
    real = pd.read_parquet(law_root / "law_real.parquet")
    bad = pd.read_parquet(law_root / "law_bad.parquet")
    _assert_allowed_splits(real, "split", "real table")
    _assert_allowed_splits(bad, "psplit", "damaged table")
    if not real.source_id.is_unique:
        raise ValueError("real source_id must be unique")
    if not bad.sid.is_unique:
        raise ValueError("damaged sid must be unique")

    real_aug = _filtered_parquet(
        feature_root / "law_real_aug.parquet", "source_id", real.source_id
    )
    bad_aug = _filtered_parquet(
        feature_root / "law_bad_aug.parquet", "parent", bad.parent
    )
    bad_keys = pd.MultiIndex.from_frame(bad[["parent", "kind"]])
    bad_aug = bad_aug[
        pd.MultiIndex.from_frame(bad_aug[["parent", "kind"]]).isin(bad_keys)
    ].copy()
    if real_aug.source_id.duplicated().any():
        raise ValueError("real augmented source_id must be unique")
    if bad_aug.duplicated(["parent", "kind"]).any():
        raise ValueError("damaged augmented parent-kind key must be unique")

    n_real, n_bad = len(real), len(bad)
    real = real.merge(real_aug, on="source_id", how="left", validate="one_to_one")
    bad = bad.merge(bad_aug, on=["parent", "kind"], how="left", validate="one_to_one")
    if len(real) != n_real or len(bad) != n_bad:
        raise AssertionError("augmentation changed the isolated analysis population")

    allowed_ids = set(real.source_id) | set(bad.parent)
    provenance = _filtered_parquet(
        feature_root / "provenance.parquet", "source_id", allowed_ids
    )[["source_id", "formula", "chemical_system"]]
    if not provenance.source_id.is_unique:
        raise ValueError("provenance source_id must be unique")
    return real, bad, provenance


def attach_identity_and_novelty(
    real: pd.DataFrame,
    bad: pd.DataFrame,
    provenance: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach parent chemistry and label discovery-seen/unseen cohorts."""

    _assert_allowed_splits(real, "split", "real table")
    _assert_allowed_splits(bad, "psplit", "damaged table")
    if not real.source_id.is_unique:
        raise ValueError("real source_id must be unique")
    if not provenance.source_id.is_unique:
        raise ValueError("provenance source_id must be unique")

    metadata = provenance[["source_id", "formula", "chemical_system"]].copy()
    metadata["composition_key"] = metadata.formula.map(canonical_composition_key)
    metadata["chemical_system_key"] = metadata.chemical_system.map(
        canonical_chemical_system
    )
    formula_system = metadata.formula.map(
        lambda value: "-".join(
            sorted(element.symbol for element in Composition(str(value)).elements)
        )
    )
    if not formula_system.equals(metadata.chemical_system_key):
        raise ValueError("provenance formula and chemical_system disagree")

    n_real, n_bad = len(real), len(bad)
    real_out = real.merge(metadata, on="source_id", how="left", validate="one_to_one")
    bad_out = bad.merge(
        metadata.rename(columns={"source_id": "parent"}),
        on="parent",
        how="left",
        validate="many_to_one",
    )
    if len(real_out) != n_real or len(bad_out) != n_bad:
        raise AssertionError("identity mapping changed row counts")
    required = ["formula", "composition_key", "chemical_system_key"]
    if real_out[required].isna().any().any() or bad_out[required].isna().any().any():
        raise ValueError("incomplete parent chemistry mapping")

    parent_split = real_out[["source_id", "split"]].rename(
        columns={"source_id": "parent", "split": "_parent_split"}
    )
    bad_out = bad_out.merge(parent_split, on="parent", how="left", validate="many_to_one")
    if bad_out._parent_split.isna().any() or not bad_out.psplit.equals(
        bad_out._parent_split
    ):
        raise ValueError("damaged row does not inherit its parent split")
    bad_out = bad_out.drop(columns="_parent_split")

    discovery = real_out[real_out.split.eq("discovery")]
    seen_compositions = set(discovery.composition_key)
    seen_systems = set(discovery.chemical_system_key)
    for frame in (real_out, bad_out):
        frame["composition_seen_in_discovery"] = frame.composition_key.isin(
            seen_compositions
        )
        frame["chemical_system_seen_in_discovery"] = frame.chemical_system_key.isin(
            seen_systems
        )
    return real_out, bad_out


def cohort_masks(frame: pd.DataFrame, *, split_column: str) -> dict[str, pd.Series]:
    """Return prespecified calibration subsets."""

    heldout = frame[split_column].eq("calibration")
    shared = heldout & frame.composition_seen_in_discovery
    unseen = heldout & ~frame.composition_seen_in_discovery
    system_unseen = heldout & ~frame.chemical_system_seen_in_discovery
    if np.any(shared & unseen) or not np.all((shared | unseen) == heldout):
        raise AssertionError("composition cohorts do not partition held-out rows")
    if not np.all(system_unseen <= unseen):
        raise AssertionError("system-unseen must be nested inside composition-unseen")
    return {
        "heldout_all": heldout,
        "composition_shared": shared,
        "composition_unseen": unseen,
        "chemical_system_unseen": system_unseen,
    }


def _le(values: pd.Series, threshold: float) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    return np.where(np.isfinite(array), array <= threshold, True)


def _ge(values: pd.Series, threshold: float) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    return np.where(np.isfinite(array), array >= threshold, True)


def law_masks(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the five frozen PRIS sets at their published thresholds."""

    d1_735 = _ge(frame.bl_min, 0.735)
    d1_804 = _ge(frame.bl_min, 0.804)
    bl_mean = _le(frame.bl_mean, 1.081)
    cn = frame.cn_an_mean.to_numpy(dtype=float)
    d2 = np.where(
        np.isfinite(frame.fi.to_numpy(dtype=float)) & (frame.fi.to_numpy(dtype=float) > 0.50),
        _le(frame.bl_min, 1.05),
        True,
    )
    d3 = np.where(np.isfinite(cn) & (cn <= 3.333), bl_mean, True)
    d4 = _le(frame.madz_range, 31.45)
    d5 = _le(frame.mad_max, 15.17)
    fi = frame.fi.to_numpy(dtype=float)
    d6 = np.where(
        np.isfinite(fi) & (fi > 0.55),
        _le(frame.frac_like_bonds, 1.0e-4),
        True,
    )
    d7 = _le(frame.wyckoff_econ_001, 2.0 / 3.0)
    d8 = _le(frame.bv_rel_mean, 0.7143040821865658)
    l2 = d1_804 & d3 & d4 & d5
    l3 = l2 & d6
    return pd.DataFrame(
        {
            "Set 1": d1_735,
            "Set 1-prime": d1_735 & d2,
            "Set 2": l2,
            "Set 3": l3,
            "Set 4": l3 & d7 & d8,
        },
        index=frame.index,
    )


def _group_rates(
    frame: pd.DataFrame,
    successes: np.ndarray,
    *,
    group_column: str,
    parent_column: str | None,
) -> pd.Series:
    work = frame[[group_column] + ([parent_column] if parent_column else [])].copy()
    work["_success"] = np.asarray(successes, dtype=float)
    if parent_column:
        parent_rates = work.groupby([group_column, parent_column], sort=False)._success.mean()
        return parent_rates.groupby(level=0, sort=False).mean()
    return work.groupby(group_column, sort=False)._success.mean()


def metric_estimates(
    frame: pd.DataFrame,
    successes: np.ndarray,
    *,
    group_column: str = "composition_key",
    parent_column: str | None = None,
) -> dict[str, float | int]:
    """Return row-micro and group-equal estimates."""

    success = np.asarray(successes, dtype=bool)
    if len(frame) == 0 or len(success) != len(frame):
        raise ValueError("frame and successes must be aligned and non-empty")
    rates = _group_rates(
        frame,
        success,
        group_column=group_column,
        parent_column=parent_column,
    )
    return {
        "n_rows": int(len(frame)),
        "n_groups": int(len(rates)),
        "n_success": int(success.sum()),
        "estimate_micro": float(success.mean()),
        "estimate_composition_equal": float(rates.mean()),
    }


def cluster_bootstrap(
    frame: pd.DataFrame,
    successes: np.ndarray,
    *,
    group_column: str = "composition_key",
    parent_column: str | None = None,
    replicates: int = 2_000,
    seed: int = 20260829,
) -> dict[str, float]:
    """Percentile intervals from whole-group resampling."""

    result = cluster_bootstrap_matrix(
        frame,
        pd.DataFrame({"metric": np.asarray(successes, dtype=bool)}, index=frame.index),
        group_column=group_column,
        parent_column=parent_column,
        replicates=replicates,
        seed=seed,
    ).loc["metric"]
    return {
        "micro_ci_low": float(result.micro_ci_low),
        "micro_ci_high": float(result.micro_ci_high),
        "composition_equal_ci_low": float(result.group_equal_ci_low),
        "composition_equal_ci_high": float(result.group_equal_ci_high),
    }


def cluster_bootstrap_matrix(
    frame: pd.DataFrame,
    successes: pd.DataFrame,
    *,
    group_column: str,
    parent_column: str | None = None,
    replicates: int = 10_000,
    seed: int = 20260829,
    batch_size: int = 128,
) -> pd.DataFrame:
    """Bootstrap several aligned endpoints with the same cluster draws."""

    if replicates < 2:
        raise ValueError("replicates must be at least two")
    if len(frame) == 0 or len(successes) != len(frame):
        raise ValueError("frame and successes must be aligned and non-empty")
    success = successes.astype(float).copy()
    success.index = frame.index
    work = frame[[group_column] + ([parent_column] if parent_column else [])].join(
        success
    )
    grouped_success = work.groupby(group_column, sort=True)[list(success.columns)].sum()
    grouped_count = work.groupby(group_column, sort=True).size().rename("_count")
    if parent_column:
        parent_rates = work.groupby(
            [group_column, parent_column], sort=True
        )[list(success.columns)].mean()
        equal_rates = parent_rates.groupby(level=0, sort=True).mean()
    else:
        equal_rates = work.groupby(group_column, sort=True)[list(success.columns)].mean()
    equal_rates = equal_rates.reindex(grouped_success.index)

    sums = grouped_success.to_numpy(dtype=float)
    counts = grouped_count.reindex(grouped_success.index).to_numpy(dtype=float)
    rates = equal_rates.to_numpy(dtype=float)
    n_groups, n_metrics = sums.shape
    rng = np.random.default_rng(seed)
    micro = np.empty((replicates, n_metrics), dtype=float)
    equal = np.empty((replicates, n_metrics), dtype=float)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        sampled = rng.integers(0, n_groups, size=(stop - start, n_groups))
        micro[start:stop] = sums[sampled].sum(axis=1) / counts[sampled].sum(
            axis=1
        )[:, None]
        equal[start:stop] = rates[sampled].mean(axis=1)
    micro_ci = np.quantile(micro, [0.025, 0.975], axis=0)
    equal_ci = np.quantile(equal, [0.025, 0.975], axis=0)
    return pd.DataFrame(
        {
            "micro_ci_low": micro_ci[0],
            "micro_ci_high": micro_ci[1],
            "group_equal_ci_low": equal_ci[0],
            "group_equal_ci_high": equal_ci[1],
        },
        index=success.columns,
    )
