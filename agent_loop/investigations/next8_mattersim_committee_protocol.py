#!/usr/bin/env python3
"""Finite, label-free score catalog for the MatterSim 1M/5M committee."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import io
import json
from numbers import Real
from pathlib import Path
import platform
import shutil
import tempfile
import weakref

import numpy as np
import pandas as pd

from src.next6_elementa_diagnostics import paired_cluster_bootstrap
from src.next6_elementa_protocol import (
    apply_group_threshold,
    attach_energy_labels,
    evaluate_group_triage,
    group_conformal_threshold,
)
from src.next8_mattersim_committee_features import (
    FROZEN_CHECKPOINT_SHA256,
    OUTPUT_NAME as FEATURE_OUTPUT_NAME,
    PROTOCOL as FEATURE_PROTOCOL,
    _atomic_publish_directory_no_replace,
    _sha256_file,
)


FORMULA_NAMES = (
    "M5",
    "M1",
    "MIN",
    "MEAN",
    "MAX",
    "LCB",
    "AGREE99",
    "AGREE995",
    "AGREE_EF995",
    "CMEAN",
    "CMEAN_JOINT99",
)
DEVELOPMENT_STAGES = (
    "search_calibration",
    "formula_selection",
    "threshold_calibration",
)
QUANTILE_METHOD = "higher"
THRESHOLD_SPLIT_SALT = "next8-threshold-fit-gate-v1-20260801"
DEVELOPMENT_FREEZE_PROTOCOL = "2026-08-01-mattersim-committee-development-freeze-v1"
BOOTSTRAP_SEED = 20260801
PRODUCTION_BOOTSTRAP_RESAMPLES = 20_000
PRIMARY_MIN_EFFECTIVE_GROUPS = 99

_FEATURE_EXECUTED_SOURCE_RELATIVE = (
    "src/next8_mattersim_committee_features.py",
    "src/next6_mattersim_baseline.py",
    "src/next6_wbm_features.py",
)
_EXECUTED_SOURCE_RELATIVE = (
    "src/next8_mattersim_committee_protocol.py",
    "src/next6_elementa_protocol.py",
    "src/next6_elementa_diagnostics.py",
    "src/next6_wbm_build.py",
    "src/next8_mattersim_committee_features.py",
    "src/next6_mattersim_baseline.py",
    "src/next6_wbm_features.py",
    "src/next6_wbm_protocol.py",
)
OUTPUT_NAMES = (
    "threshold_role_assignments.parquet",
    "development_frontier.parquet",
    "threshold_fit_rules.parquet",
    "development_gate_metrics.parquet",
    "PAIRED_BOOTSTRAP.json",
    "IMPROVEMENT_GATE.json",
    "FROZEN_PROTOCOL.json",
    "MANIFEST.json",
)


@dataclass(frozen=True)
class TrackSpec:
    name: str
    protected_column: str
    protected_ev_per_atom: float
    within_group: str
    alpha: float


TRACKS = {
    "primary": TrackSpec("primary", "valuable", 0.05, "max", 0.01),
    "comparator": TrackSpec("comparator", "near_min", 0.001, "min", 0.035),
}
FORMULA_COST = {
    formula: (1 if formula == "M1" else 5 if formula == "M5" else 6)
    for formula in FORMULA_NAMES
}
FORMULA_COMPLEXITY = {
    "M5": 1,
    "M1": 1,
    "MIN": 2,
    "MEAN": 2,
    "MAX": 2,
    "LCB": 3,
    "AGREE99": 3,
    "AGREE995": 3,
    "AGREE_EF995": 4,
    "CMEAN": 3,
    "CMEAN_JOINT99": 5,
}

_FORMULA_DEFINITIONS = (
    {"name": "M5", "score": "g5"},
    {"name": "M1", "score": "g1"},
    {"name": "MIN", "score": "min(g1,g5)"},
    {"name": "MEAN", "score": "(g1+g5)/2"},
    {"name": "MAX", "score": "max(g1,g5)"},
    {"name": "LCB", "score": "max(0,(g1+g5)/2-uE)"},
    {
        "name": "AGREE99",
        "score": "(g1+g5)/2",
        "abstain_if": "uE>q99_energy",
    },
    {
        "name": "AGREE995",
        "score": "(g1+g5)/2",
        "abstain_if": "uE>q995_energy",
    },
    {
        "name": "AGREE_EF995",
        "score": "(g1+g5)/2",
        "abstain_if": "uE>q995_energy or uF>q995_force",
    },
    {
        "name": "CMEAN",
        "score": "(g1+g5)/2-min_rk((g1+g5)/2)",
    },
    {
        "name": "CMEAN_JOINT99",
        "score": "(g1+g5)/2-min_rk((g1+g5)/2)",
        "abstain_if": ("max(H_E(dE),H_Fmax(dFmax),H_Frms(dFrms))>qJ99"),
    },
)

_ENERGY_COLUMNS = (
    "m1_energy_ev_per_atom",
    "m5_energy_ev_per_atom",
)
_FORCE_COLUMNS = (
    "m1_fmax_ev_per_a",
    "m1_frms_ev_per_a",
    "m5_fmax_ev_per_a",
    "m5_frms_ev_per_a",
)
_KEY_COLUMNS = ("sid", "rk", "stage")
_FLAG_COLUMNS = (
    "committee_feature_ok",
    "m1_prediction_ok",
    "m5_prediction_ok",
)


class _CalibrationOriginToken:
    __slots__ = ("__weakref__",)


_ORIGIN_REGISTRY: weakref.WeakKeyDictionary[
    _CalibrationOriginToken, tuple[object, ...]
] = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class DisagreementCutoffs:
    """Search-calibration disagreement quantiles in eV/atom."""

    q99_ev_per_atom: float
    q995_ev_per_atom: float
    q995_force_ev_per_a: float
    eligible_row_count: int
    source_stage: str = "search_calibration"
    quantile_method: str = QUANTILE_METHOD
    calibration_fingerprint_sha256: str = ""
    joint_q99: float = 0.0
    joint_reference_n: int = 0
    joint_reference_n_rk: int = 0
    joint_weighting: str = "row"
    joint_ecdf_side: str = "right"
    joint_reference_dE: tuple[float, ...] = ()
    joint_reference_dFmax: tuple[float, ...] = ()
    joint_reference_dFrms: tuple[float, ...] = ()
    joint_reference_sha256: str = ""
    _origin_token: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._origin_token is None:
            raise ValueError("use derive_disagreement_cutoffs to construct cutoffs")


def _validated_features(features: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    required = set(_KEY_COLUMNS + _FLAG_COLUMNS + _ENERGY_COLUMNS + _FORCE_COLUMNS)
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"features are missing required columns: {missing}")

    result = features.loc[
        :, _KEY_COLUMNS + _FLAG_COLUMNS + _ENERGY_COLUMNS + _FORCE_COLUMNS
    ].copy()
    for column in _KEY_COLUMNS:
        if (
            not result[column]
            .map(lambda value: isinstance(value, str) and bool(value))
            .all()
        ):
            raise ValueError(f"{column} values must be nonempty exact strings")
    if result["sid"].duplicated().any() or result.duplicated(["sid", "rk"]).any():
        raise ValueError("features contain duplicate sid/rk keys")

    for column in _FLAG_COLUMNS:
        if (
            not result[column]
            .map(lambda value: isinstance(value, (bool, np.bool_)))
            .all()
        ):
            raise ValueError(f"{column} values must be exact booleans")
        result[column] = result[column].astype(bool)
    expected_committee = result["m1_prediction_ok"] & result["m5_prediction_ok"]
    if not result["committee_feature_ok"].equals(expected_committee):
        raise ValueError(
            "committee_feature_ok must equal the conjunction of "
            "m1_prediction_ok and m5_prediction_ok"
        )

    for model in ("m1", "m5"):
        column = f"{model}_energy_ev_per_atom"
        dtype = result[column].dtype
        if (
            not pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_complex_dtype(dtype)
        ):
            raise ValueError(f"{column} values must be real numeric values")
        values = result[column].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError(f"{column} values must be finite")
        claimed_success = result[f"{model}_prediction_ok"].to_numpy(dtype=bool)
        if (~np.isfinite(values[claimed_success])).any():
            raise ValueError(f"{column} values must be finite when prediction succeeds")
        result[column] = values
    for column in _FORCE_COLUMNS:
        dtype = result[column].dtype
        if (
            not pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_complex_dtype(dtype)
        ):
            raise ValueError(f"{column} values must be real numeric values")
        values = result[column].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError(f"{column} values must be finite")
        model = column.split("_", maxsplit=1)[0]
        claimed_success = result[f"{model}_prediction_ok"].to_numpy(dtype=bool)
        if (~np.isfinite(values[claimed_success])).any():
            raise ValueError(f"{column} values must be finite when prediction succeeds")
        if (np.isfinite(values) & (values < 0.0)).any():
            raise ValueError(f"{column} values must be nonnegative")
        result[column] = values
    return result


def _logical_calibration_fingerprint(features: pd.DataFrame) -> str:
    ordered = features.sort_values(["sid", "rk", "stage"], kind="stable").reset_index(
        drop=True
    )
    rows: list[list[object]] = []
    for record in ordered.to_dict("records"):
        rows.append(
            [
                record["sid"],
                record["rk"],
                record["stage"],
                bool(record["committee_feature_ok"]),
                bool(record["m1_prediction_ok"]),
                bool(record["m5_prediction_ok"]),
                float(record["m1_energy_ev_per_atom"]).hex(),
                float(record["m5_energy_ev_per_atom"]).hex(),
                float(record["m1_fmax_ev_per_a"]).hex(),
                float(record["m1_frms_ev_per_a"]).hex(),
                float(record["m5_fmax_ev_per_a"]).hex(),
                float(record["m5_frms_ev_per_a"]).hex(),
            ]
        )
    canonical = json.dumps(
        {
            "schema": "next8-label-free-calibration-logical-v1",
            "rows": rows,
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _joint_reference_fingerprint(
    *,
    dE: tuple[float, ...],
    dFmax: tuple[float, ...],
    dFrms: tuple[float, ...],
    n: int,
    n_rk: int,
    q99: float,
) -> str:
    canonical = json.dumps(
        {
            "ecdf_side": "right",
            "n": n,
            "n_rk": n_rk,
            "q99": float(q99).hex(),
            "quantile_method": QUANTILE_METHOD,
            "reference": {
                "dE": [float(value).hex() for value in dE],
                "dFmax": [float(value).hex() for value in dFmax],
                "dFrms": [float(value).hex() for value in dFrms],
            },
            "schema": "next8-joint-disagreement-ecdf-v1",
            "weighting": "row",
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _registered_field_tuple(
    cutoffs: DisagreementCutoffs,
) -> tuple[object, ...]:
    return (
        type(cutoffs.q99_ev_per_atom),
        float(cutoffs.q99_ev_per_atom).hex(),
        type(cutoffs.q995_ev_per_atom),
        float(cutoffs.q995_ev_per_atom).hex(),
        type(cutoffs.q995_force_ev_per_a),
        float(cutoffs.q995_force_ev_per_a).hex(),
        type(cutoffs.eligible_row_count),
        cutoffs.eligible_row_count,
        type(cutoffs.source_stage),
        cutoffs.source_stage,
        type(cutoffs.quantile_method),
        cutoffs.quantile_method,
        type(cutoffs.calibration_fingerprint_sha256),
        cutoffs.calibration_fingerprint_sha256,
        type(cutoffs.joint_q99),
        float(cutoffs.joint_q99).hex(),
        type(cutoffs.joint_reference_n),
        cutoffs.joint_reference_n,
        type(cutoffs.joint_reference_n_rk),
        cutoffs.joint_reference_n_rk,
        type(cutoffs.joint_weighting),
        cutoffs.joint_weighting,
        type(cutoffs.joint_ecdf_side),
        cutoffs.joint_ecdf_side,
        type(cutoffs.joint_reference_sha256),
        cutoffs.joint_reference_sha256,
    )


def _require_single_stage(features: pd.DataFrame, expected_stage: str) -> None:
    observed = set(features["stage"].tolist())
    if observed != {expected_stage}:
        raise ValueError(
            f"expected only {expected_stage!r}; observed stages={sorted(observed)!r}"
        )


def _group_gaps(features: pd.DataFrame) -> pd.DataFrame:
    result = features.loc[:, _KEY_COLUMNS + _ENERGY_COLUMNS + _FORCE_COLUMNS].copy()
    for model in ("m1", "m5"):
        energy = f"{model}_energy_ev_per_atom"
        gap = f"g{model.removeprefix('m')}_ev_per_atom"
        complete_column = f"{model}_group_complete"
        model_ready = features[f"{model}_prediction_ok"]
        model_complete = model_ready.groupby(features["rk"], sort=False).transform(
            "all"
        )
        result[complete_column] = model_complete.to_numpy(dtype=bool)
        eligible_energy = result[energy].where(result[complete_column])
        minimum = eligible_energy.groupby(result["rk"], sort=False).transform("min")
        with np.errstate(invalid="ignore", over="ignore"):
            result[gap] = eligible_energy - minimum
        values = result[gap].to_numpy(dtype=float)
        complete = result[complete_column].to_numpy(dtype=bool)
        if ((~np.isfinite(values) | (values < 0.0)) & complete).any():
            raise ValueError(f"complete-group {gap} must be finite and nonnegative")
    result["joint_group_complete"] = (
        result["m1_group_complete"] & result["m5_group_complete"]
    )
    joint_complete = result["joint_group_complete"].to_numpy(dtype=bool)
    for column in ("g1_ev_per_atom", "g5_ev_per_atom"):
        values = result[column].to_numpy(dtype=float)
        if ((~np.isfinite(values) | (values < 0.0)) & joint_complete).any():
            raise ValueError(f"complete-group {column} must be finite and nonnegative")
    with np.errstate(invalid="ignore", over="ignore"):
        result["disagreement_ev_per_atom"] = (
            result["g1_ev_per_atom"] - result["g5_ev_per_atom"]
        ).abs()
    disagreement = result["disagreement_ev_per_atom"].to_numpy(dtype=float)
    if ((~np.isfinite(disagreement) | (disagreement < 0.0)) & joint_complete).any():
        raise ValueError("complete-group disagreement must be finite and nonnegative")
    with np.errstate(invalid="ignore", over="ignore"):
        result["fmax_disagreement_ev_per_a"] = (
            result["m1_fmax_ev_per_a"] - result["m5_fmax_ev_per_a"]
        ).abs()
        result["frms_disagreement_ev_per_a"] = (
            result["m1_frms_ev_per_a"] - result["m5_frms_ev_per_a"]
        ).abs()
        result["force_disagreement_ev_per_a"] = np.maximum(
            result["fmax_disagreement_ev_per_a"],
            result["frms_disagreement_ev_per_a"],
        )
    for column in (
        "fmax_disagreement_ev_per_a",
        "frms_disagreement_ev_per_a",
        "force_disagreement_ev_per_a",
    ):
        force_disagreement = result[column].to_numpy(dtype=float)
        if (
            (~np.isfinite(force_disagreement) | (force_disagreement < 0.0))
            & joint_complete
        ).any():
            raise ValueError(f"complete-group {column} must be finite and nonnegative")
    return result


def derive_disagreement_cutoffs(features: pd.DataFrame) -> DisagreementCutoffs:
    """Derive frozen unlabeled disagreement gates from search calibration."""

    validated = _validated_features(features)
    _require_single_stage(validated, "search_calibration")
    gaps = _group_gaps(validated)
    eligible = gaps["joint_group_complete"].to_numpy(dtype=bool)
    values = gaps.loc[eligible, "disagreement_ev_per_atom"].to_numpy(dtype=float)
    fmax_values = gaps.loc[eligible, "fmax_disagreement_ev_per_a"].to_numpy(dtype=float)
    frms_values = gaps.loc[eligible, "frms_disagreement_ev_per_a"].to_numpy(dtype=float)
    force_values = gaps.loc[eligible, "force_disagreement_ev_per_a"].to_numpy(
        dtype=float
    )
    if len(values) == 0:
        raise ValueError("search_calibration has no complete finite groups")
    q99 = float(np.quantile(values, 0.99, method=QUANTILE_METHOD))
    q995 = float(np.quantile(values, 0.995, method=QUANTILE_METHOD))
    q995_force = float(np.quantile(force_values, 0.995, method=QUANTILE_METHOD))
    row_count = int(len(values))
    joint_reference_dE = tuple(float(value) for value in np.sort(values))
    joint_reference_dFmax = tuple(float(value) for value in np.sort(fmax_values))
    joint_reference_dFrms = tuple(float(value) for value in np.sort(frms_values))
    joint_scores = np.maximum.reduce(
        (
            np.searchsorted(joint_reference_dE, values, side="right").astype(float)
            / row_count,
            np.searchsorted(joint_reference_dFmax, fmax_values, side="right").astype(
                float
            )
            / row_count,
            np.searchsorted(joint_reference_dFrms, frms_values, side="right").astype(
                float
            )
            / row_count,
        )
    )
    joint_q99 = float(np.quantile(joint_scores, 0.99, method=QUANTILE_METHOD))
    joint_reference_n_rk = int(gaps.loc[eligible, "rk"].nunique())
    joint_reference_sha256 = _joint_reference_fingerprint(
        dE=joint_reference_dE,
        dFmax=joint_reference_dFmax,
        dFrms=joint_reference_dFrms,
        n=row_count,
        n_rk=joint_reference_n_rk,
        q99=joint_q99,
    )
    fingerprint = _logical_calibration_fingerprint(validated)
    origin_token = _CalibrationOriginToken()
    cutoffs = DisagreementCutoffs(
        q99_ev_per_atom=q99,
        q995_ev_per_atom=q995,
        q995_force_ev_per_a=q995_force,
        eligible_row_count=row_count,
        source_stage="search_calibration",
        quantile_method=QUANTILE_METHOD,
        calibration_fingerprint_sha256=fingerprint,
        joint_q99=joint_q99,
        joint_reference_n=row_count,
        joint_reference_n_rk=joint_reference_n_rk,
        joint_weighting="row",
        joint_ecdf_side="right",
        joint_reference_dE=joint_reference_dE,
        joint_reference_dFmax=joint_reference_dFmax,
        joint_reference_dFrms=joint_reference_dFrms,
        joint_reference_sha256=joint_reference_sha256,
        _origin_token=origin_token,
    )
    _ORIGIN_REGISTRY[origin_token] = _registered_field_tuple(cutoffs)
    return cutoffs


def _formula_values(gaps: pd.DataFrame) -> dict[str, np.ndarray]:
    g1 = gaps["g1_ev_per_atom"].to_numpy(dtype=float)
    g5 = gaps["g5_ev_per_atom"].to_numpy(dtype=float)
    disagreement = gaps["disagreement_ev_per_atom"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", over="ignore"):
        mean = 0.5 * g1 + 0.5 * g5
        joint_complete = gaps["joint_group_complete"].to_numpy(dtype=bool)
        mean_for_cmean = pd.Series(mean, index=gaps.index).where(joint_complete)
        cmean = (
            mean_for_cmean
            - mean_for_cmean.groupby(gaps["rk"], sort=False).transform("min")
        ).to_numpy(dtype=float)
        values = {
            "M5": g5,
            "M1": g1,
            "MIN": np.minimum(g1, g5),
            "MEAN": mean,
            "MAX": np.maximum(g1, g5),
            "LCB": np.maximum(0.0, mean - disagreement),
            "AGREE99": mean,
            "AGREE995": mean,
            "AGREE_EF995": mean,
            "CMEAN": cmean,
            "CMEAN_JOINT99": cmean,
        }
    for formula, formula_values in values.items():
        if formula == "M5":
            complete_column = "m5_group_complete"
        elif formula == "M1":
            complete_column = "m1_group_complete"
        else:
            complete_column = "joint_group_complete"
        complete = gaps[complete_column].to_numpy(dtype=bool)
        invalid = (~np.isfinite(formula_values) | (formula_values < 0.0)) & complete
        if invalid.any():
            raise ValueError(
                f"complete-group {formula} scores must be finite and nonnegative"
            )
    return values


def _joint_ecdf_scores(gaps: pd.DataFrame, cutoffs: DisagreementCutoffs) -> np.ndarray:
    n = cutoffs.joint_reference_n
    scores = np.maximum.reduce(
        (
            np.searchsorted(
                cutoffs.joint_reference_dE,
                gaps["disagreement_ev_per_atom"].to_numpy(dtype=float),
                side="right",
            ).astype(float)
            / n,
            np.searchsorted(
                cutoffs.joint_reference_dFmax,
                gaps["fmax_disagreement_ev_per_a"].to_numpy(dtype=float),
                side="right",
            ).astype(float)
            / n,
            np.searchsorted(
                cutoffs.joint_reference_dFrms,
                gaps["frms_disagreement_ev_per_a"].to_numpy(dtype=float),
                side="right",
            ).astype(float)
            / n,
        )
    )
    incomplete = ~gaps["joint_group_complete"].to_numpy(dtype=bool)
    scores[incomplete] = np.nan
    return scores


def construct_committee_scores(
    features: pd.DataFrame,
    *,
    cutoffs: DisagreementCutoffs,
    expected_stage: str,
) -> pd.DataFrame:
    """Construct the frozen long-form score catalog for one development stage."""

    if expected_stage not in DEVELOPMENT_STAGES:
        raise ValueError("expected_stage must be one of the three development stages")
    _validate_cutoffs(cutoffs)
    validated = _validated_features(features)
    _require_single_stage(validated, expected_stage)
    gaps = _group_gaps(validated)
    values = _formula_values(gaps)
    joint_ecdf_scores = _joint_ecdf_scores(gaps, cutoffs)
    parts: list[pd.DataFrame] = []
    for formula in FORMULA_NAMES:
        if formula == "M5":
            complete_column = "m5_group_complete"
            incomplete_reason = "incomplete_group_m5_failure"
        elif formula == "M1":
            complete_column = "m1_group_complete"
            incomplete_reason = "incomplete_group_m1_failure"
        else:
            complete_column = "joint_group_complete"
            incomplete_reason = "incomplete_group_committee_failure"
        incomplete = ~gaps[complete_column].to_numpy(dtype=bool)
        part = gaps[
            [
                "sid",
                "rk",
                "stage",
                "g1_ev_per_atom",
                "g5_ev_per_atom",
                "disagreement_ev_per_atom",
                "fmax_disagreement_ev_per_a",
                "frms_disagreement_ev_per_a",
                "force_disagreement_ev_per_a",
            ]
        ].copy()
        part.insert(3, "formula", formula)
        part["score_ev_per_atom"] = values[formula]
        part["joint_ecdf_score"] = joint_ecdf_scores
        part["state"] = "KEEP"
        part["abstain_reason"] = ""
        part.loc[incomplete, "score_ev_per_atom"] = np.nan
        part.loc[incomplete, "state"] = "ABSTAIN"
        part.loc[incomplete, "abstain_reason"] = incomplete_reason
        if formula in {"AGREE99", "AGREE995"}:
            threshold = (
                cutoffs.q99_ev_per_atom
                if formula == "AGREE99"
                else cutoffs.q995_ev_per_atom
            )
            above = (
                part["disagreement_ev_per_atom"].to_numpy(dtype=float) > threshold
            ) & ~incomplete
            part.loc[above, "score_ev_per_atom"] = np.nan
            part.loc[above, "state"] = "ABSTAIN"
            part.loc[above, "abstain_reason"] = "disagreement_above_threshold"
        elif formula == "AGREE_EF995":
            energy_above = (
                part["disagreement_ev_per_atom"].to_numpy(dtype=float)
                > cutoffs.q995_ev_per_atom
            ) & ~incomplete
            force_above = (
                part["force_disagreement_ev_per_a"].to_numpy(dtype=float)
                > cutoffs.q995_force_ev_per_a
            ) & ~incomplete
            either_above = energy_above | force_above
            part.loc[either_above, "score_ev_per_atom"] = np.nan
            part.loc[either_above, "state"] = "ABSTAIN"
            part.loc[
                energy_above & ~force_above, "abstain_reason"
            ] = "energy_disagreement_above_threshold"
            part.loc[
                force_above & ~energy_above, "abstain_reason"
            ] = "force_disagreement_above_threshold"
            part.loc[
                energy_above & force_above, "abstain_reason"
            ] = "energy_and_force_disagreement_above_threshold"
        elif formula == "CMEAN_JOINT99":
            joint_above = (
                part["joint_ecdf_score"].to_numpy(dtype=float) > cutoffs.joint_q99
            ) & ~incomplete
            part.loc[joint_above, "score_ev_per_atom"] = np.nan
            part.loc[joint_above, "state"] = "ABSTAIN"
            part.loc[
                joint_above, "abstain_reason"
            ] = "joint_ecdf_disagreement_above_threshold"
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _validate_cutoffs(cutoffs: DisagreementCutoffs) -> None:
    if not isinstance(cutoffs, DisagreementCutoffs):
        raise TypeError("cutoffs must be DisagreementCutoffs")
    token = cutoffs._origin_token
    if not isinstance(token, _CalibrationOriginToken):
        raise ValueError("calibration origin token is not registered")
    registered = _ORIGIN_REGISTRY.get(token)
    if registered is None:
        raise ValueError("calibration origin token is not registered")
    for name in (
        "q99_ev_per_atom",
        "q995_ev_per_atom",
        "q995_force_ev_per_a",
        "joint_q99",
    ):
        value = getattr(cutoffs, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a non-boolean real number")
    if type(cutoffs.eligible_row_count) is not int:
        raise ValueError("eligible_row_count must be a built-in int")
    if type(cutoffs.joint_reference_n) is not int:
        raise ValueError("joint_reference_n must be a built-in int")
    if type(cutoffs.joint_reference_n_rk) is not int:
        raise ValueError("joint_reference_n_rk must be a built-in int")
    if cutoffs.source_stage != "search_calibration":
        raise ValueError("cutoffs must come from search_calibration")
    if cutoffs.quantile_method != QUANTILE_METHOD:
        raise ValueError(f"quantile_method must be {QUANTILE_METHOD!r}")
    values = np.asarray(
        [
            cutoffs.q99_ev_per_atom,
            cutoffs.q995_ev_per_atom,
            cutoffs.q995_force_ev_per_a,
        ],
        dtype=float,
    )
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("disagreement cutoffs must be finite and nonnegative")
    if cutoffs.q995_ev_per_atom < cutoffs.q99_ev_per_atom:
        raise ValueError("q995 cutoff must be greater than or equal to q99")
    if cutoffs.eligible_row_count <= 0:
        raise ValueError("eligible_row_count must be a positive integer")
    joint_q99 = float(cutoffs.joint_q99)
    if not np.isfinite(joint_q99) or not 0.0 <= joint_q99 <= 1.0:
        raise ValueError("joint_q99 cutoff must be finite and in [0, 1]")
    if cutoffs.joint_reference_n != cutoffs.eligible_row_count:
        raise ValueError("joint cutoff reference n must equal eligible_row_count")
    if not 0 < cutoffs.joint_reference_n_rk <= cutoffs.joint_reference_n:
        raise ValueError("joint cutoff reference n_rk must be in [1, n]")
    if cutoffs.joint_weighting != "row":
        raise ValueError("joint cutoff weighting must be 'row'")
    if cutoffs.joint_ecdf_side != "right":
        raise ValueError("joint cutoff ECDF side must be 'right'")
    references = {
        "joint_reference_dE": cutoffs.joint_reference_dE,
        "joint_reference_dFmax": cutoffs.joint_reference_dFmax,
        "joint_reference_dFrms": cutoffs.joint_reference_dFrms,
    }
    for name, reference in references.items():
        if type(reference) is not tuple:
            raise ValueError(f"{name} must be an exact tuple")
        if len(reference) != cutoffs.joint_reference_n:
            raise ValueError(f"{name} must contain exactly joint_reference_n values")
        if any(
            isinstance(value, (bool, np.bool_)) or type(value) not in (int, float)
            for value in reference
        ):
            raise ValueError(f"{name} values must be built-in non-boolean real numbers")
        array = np.asarray(reference, dtype=float)
        if not np.isfinite(array).all() or (array < 0.0).any():
            raise ValueError(f"{name} values must be finite and nonnegative")
        if len(array) > 1 and (np.diff(array) < 0.0).any():
            raise ValueError(f"{name} must be sorted ascending")
    fingerprint = cutoffs.calibration_fingerprint_sha256
    if (
        type(fingerprint) is not str
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError(
            "calibration_fingerprint_sha256 must be 64 lowercase hex digits"
        )
    reference_fingerprint = cutoffs.joint_reference_sha256
    if (
        type(reference_fingerprint) is not str
        or len(reference_fingerprint) != 64
        or any(
            character not in "0123456789abcdef" for character in reference_fingerprint
        )
    ):
        raise ValueError("joint_reference_sha256 must be 64 lowercase hex digits")
    recomputed_reference_fingerprint = _joint_reference_fingerprint(
        dE=cutoffs.joint_reference_dE,
        dFmax=cutoffs.joint_reference_dFmax,
        dFrms=cutoffs.joint_reference_dFrms,
        n=cutoffs.joint_reference_n,
        n_rk=cutoffs.joint_reference_n_rk,
        q99=joint_q99,
    )
    if reference_fingerprint != recomputed_reference_fingerprint:
        raise ValueError("joint cutoff reference hash mismatch")
    if registered != _registered_field_tuple(cutoffs):
        raise ValueError("cutoff fields do not match the registered calibration origin")


def serialize_formula_catalog(cutoffs: DisagreementCutoffs) -> str:
    """Serialize only the mathematical catalog and unlabeled calibration facts."""

    _validate_cutoffs(cutoffs)
    payload = {
        "calibration": {
            "calibration_fingerprint_sha256": (cutoffs.calibration_fingerprint_sha256),
            "eligible_row_count": int(cutoffs.eligible_row_count),
            "n": int(cutoffs.eligible_row_count),
            "quantile_method": cutoffs.quantile_method,
            "q99_ev_per_atom": float(cutoffs.q99_ev_per_atom),
            "q995_ev_per_atom": float(cutoffs.q995_ev_per_atom),
            "q995_force_ev_per_a": float(cutoffs.q995_force_ev_per_a),
            "source_stage": cutoffs.source_stage,
            "joint_ecdf": {
                "n": int(cutoffs.joint_reference_n),
                "n_rk": int(cutoffs.joint_reference_n_rk),
                "weighting": cutoffs.joint_weighting,
                "side": cutoffs.joint_ecdf_side,
                "quantile_method": cutoffs.quantile_method,
                "qJ99": float(cutoffs.joint_q99),
                "sorted_reference": {
                    "dE": list(cutoffs.joint_reference_dE),
                    "dFmax": list(cutoffs.joint_reference_dFmax),
                    "dFrms": list(cutoffs.joint_reference_dFrms),
                },
                "reference_sha256": cutoffs.joint_reference_sha256,
            },
        },
        "formulas": _FORMULA_DEFINITIONS,
    }
    return json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _resolve_track(track: TrackSpec | str) -> TrackSpec:
    name = track.name if isinstance(track, TrackSpec) else track
    if type(name) is not str or name not in TRACKS:
        raise ValueError(f"unknown frozen track: {name!r}")
    canonical = TRACKS[name]
    if isinstance(track, TrackSpec) and track != canonical:
        raise ValueError("track does not match the frozen semantics")
    return canonical


def _metric_float(metrics: Mapping[str, object], name: str) -> float:
    try:
        value = metrics[name]
        if isinstance(value, (bool, np.bool_)):
            return float("nan")
        return float(value)
    except (KeyError, TypeError, ValueError):
        return float("nan")


def passes_safety_gate(metrics: Mapping[str, object], track: TrackSpec | str) -> bool:
    """Apply the frozen exact/protected/regret/survivor safety gate."""

    spec = _resolve_track(track)
    exact = _metric_float(metrics, "exact_min_retention_lower")
    protected = _metric_float(
        metrics,
        "valuable_group_retention_lower"
        if spec.name == "primary"
        else "near_min_retention_lower",
    )
    regret = _metric_float(metrics, "regret_p95")
    all_rejected = _metric_float(metrics, "all_rejected_groups")
    return bool(
        exact >= 0.95 and protected >= 0.95 and regret <= 0.05 and all_rejected == 0.0
    )


def select_primary_formula(frontier: pd.DataFrame) -> dict[str, object]:
    """Select one production formula; comparator only audits that formula."""

    if not isinstance(frontier, pd.DataFrame):
        raise TypeError("frontier must be a pandas DataFrame")
    required = {
        "formula",
        "track",
        "dft_savings",
        "exact_min_retention_lower",
        "near_min_retention_lower",
        "valuable_group_retention_lower",
        "regret_p95",
        "all_rejected_groups",
    }
    missing = sorted(required - set(frontier.columns))
    if missing:
        raise ValueError(f"frontier is missing columns: {missing}")
    expected_pairs = {(formula, track) for formula in FORMULA_NAMES for track in TRACKS}
    observed_pairs = list(zip(frontier["formula"], frontier["track"], strict=True))
    if len(observed_pairs) != len(set(observed_pairs)):
        raise ValueError("frontier contains duplicate formula/track rows")
    if set(observed_pairs) != expected_pairs:
        raise ValueError("frontier must contain the exact frozen formula/track grid")

    records = {
        (str(row["formula"]), str(row["track"])): row
        for row in frontier.to_dict("records")
    }
    candidates: list[dict[str, object]] = []
    for catalog_order, formula in enumerate(FORMULA_NAMES):
        primary = records[(formula, "primary")]
        comparator = records[(formula, "comparator")]
        if not (
            passes_safety_gate(primary, "primary")
            and passes_safety_gate(comparator, "comparator")
        ):
            continue
        savings = _metric_float(primary, "dft_savings")
        if not np.isfinite(savings):
            continue
        candidates.append(
            {
                "state": "selected",
                "name": formula,
                "primary_dft_savings": savings,
                "cost": FORMULA_COST[formula],
                "complexity": FORMULA_COMPLEXITY[formula],
                "catalog_order": catalog_order,
            }
        )
    if not candidates:
        return {"state": "null_keep_all", "name": "null_keep_all"}
    candidates.sort(
        key=lambda row: (
            -float(row["primary_dft_savings"]),
            int(row["cost"]),
            int(row["complexity"]),
            int(row["catalog_order"]),
        )
    )
    return candidates[0]


def _prepare_labels(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    *,
    expected_stage: str,
) -> pd.DataFrame:
    if not isinstance(labels, pd.DataFrame):
        raise TypeError("labels must be a pandas DataFrame")
    required = {"sid", "rk", "stage", "e_per_atom"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"labels are missing columns: {missing}")
    normalized = labels.loc[:, ["sid", "rk", "stage", "e_per_atom"]].copy()
    for column in ("sid", "rk", "stage"):
        if (
            not normalized[column]
            .map(lambda value: type(value) is str and bool(value))
            .all()
        ):
            raise ValueError(f"label {column} values must be nonempty exact strings")
    if (
        normalized["sid"].duplicated().any()
        or normalized.duplicated(["sid", "rk"]).any()
    ):
        raise ValueError("labels contain duplicate sid/rk keys")
    if set(normalized["stage"]) != {expected_stage}:
        raise ValueError(f"labels must contain only {expected_stage}")
    dtype = normalized["e_per_atom"].dtype
    if (
        not pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_complex_dtype(dtype)
    ):
        raise ValueError("label e_per_atom values must be real numeric values")
    energy = normalized["e_per_atom"].to_numpy(dtype=float)
    if not np.isfinite(energy).all():
        raise ValueError("label e_per_atom values must be finite")
    normalized["e_per_atom"] = energy

    feature_keys = set(
        zip(features["sid"], features["rk"], features["stage"], strict=True)
    )
    label_keys = set(
        zip(
            normalized["sid"],
            normalized["rk"],
            normalized["stage"],
            strict=True,
        )
    )
    if feature_keys != label_keys:
        raise ValueError("feature and label sid/rk/stage keys differ")
    labelled = attach_energy_labels(normalized[["sid", "rk", "e_per_atom"]])
    labelled["stage"] = expected_stage
    return labelled


def _labelled_scores(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    cutoffs: DisagreementCutoffs,
    expected_stage: str,
) -> pd.DataFrame:
    scores = construct_committee_scores(
        features, cutoffs=cutoffs, expected_stage=expected_stage
    )
    prepared_labels = _prepare_labels(labels, features, expected_stage=expected_stage)
    joined = scores.merge(
        prepared_labels,
        on=["sid", "rk", "stage"],
        how="inner",
        validate="many_to_one",
    )
    if len(joined) != len(scores):
        raise ValueError("score and label rows are not exactly aligned")
    joined["score"] = joined["score_ev_per_atom"]
    joined["supported"] = joined["state"].eq("KEEP") & np.isfinite(
        joined["score_ev_per_atom"].to_numpy(dtype=float)
    )
    return joined


def fit_provisional_thresholds(
    search_features: pd.DataFrame,
    search_labels: pd.DataFrame,
    *,
    cutoffs: DisagreementCutoffs,
) -> pd.DataFrame:
    """Fit all 11 x 2 provisional rules on search_calibration only."""

    validated_search = _validated_features(search_features)
    _require_single_stage(validated_search, "search_calibration")
    _validate_cutoffs(cutoffs)
    if cutoffs.calibration_fingerprint_sha256 != (
        _logical_calibration_fingerprint(validated_search)
    ):
        raise ValueError(
            "cutoff fingerprint must match the same search_calibration features"
        )
    labelled = _labelled_scores(
        validated_search,
        search_labels,
        cutoffs=cutoffs,
        expected_stage="search_calibration",
    )
    records: list[dict[str, object]] = []
    for formula in FORMULA_NAMES:
        formula_rows = labelled.loc[labelled["formula"].eq(formula)]
        for track in TRACKS.values():
            result = group_conformal_threshold(
                formula_rows,
                alpha=track.alpha,
                valuable_column=track.protected_column,
                score_column="score",
                supported_column="supported",
                within_group=track.within_group,
            )
            threshold = float(result["threshold"])
            if not np.isfinite(threshold):
                threshold = float("inf")
            records.append(
                {
                    "formula": formula,
                    "track": track.name,
                    "threshold": threshold,
                    "threshold_state": (
                        "finite" if np.isfinite(threshold) else "keep_all"
                    ),
                    "threshold_source_stage": "search_calibration",
                    "alpha": track.alpha,
                    "within_group": track.within_group,
                    "protected": track.protected_column,
                    "calibration_n_groups": int(result["n_groups"]),
                    "calibration_order_index": int(result["order_index"]),
                    "operator": "score > threshold",
                    "cost": FORMULA_COST[formula],
                    "complexity": FORMULA_COMPLEXITY[formula],
                }
            )
    return pd.DataFrame(records)


def _validated_provisional_thresholds(rules: pd.DataFrame) -> pd.DataFrame:
    required = {
        "formula",
        "track",
        "threshold",
        "threshold_source_stage",
        "alpha",
        "within_group",
        "operator",
    }
    missing = sorted(required - set(rules.columns))
    if missing:
        raise ValueError(f"provisional thresholds are missing columns: {missing}")
    out = rules.copy()
    pairs = list(zip(out["formula"], out["track"], strict=True))
    expected = {(formula, track) for formula in FORMULA_NAMES for track in TRACKS}
    if len(pairs) != len(set(pairs)) or set(pairs) != expected:
        raise ValueError("provisional thresholds do not match the frozen grid")
    if set(out["threshold_source_stage"]) != {"search_calibration"}:
        raise ValueError("provisional thresholds must come from search_calibration")
    if set(out["operator"]) != {"score > threshold"}:
        raise ValueError("provisional threshold operator mismatch")
    for row in out.to_dict("records"):
        track = TRACKS[str(row["track"])]
        if (
            float(row["alpha"]) != track.alpha
            or row["within_group"] != track.within_group
        ):
            raise ValueError("provisional threshold track semantics mismatch")
        threshold = row["threshold"]
        if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
            raise ValueError("provisional threshold must be a real number")
        if np.isnan(float(threshold)):
            raise ValueError("provisional threshold cannot be NaN")
        if np.isneginf(float(threshold)):
            raise ValueError(
                "provisional threshold must be finite or positive infinity"
            )
    return out


def evaluate_formula_selection(
    selection_features: pd.DataFrame,
    selection_labels: pd.DataFrame,
    *,
    cutoffs: DisagreementCutoffs,
    provisional_thresholds: pd.DataFrame,
) -> pd.DataFrame:
    """Apply, but never refit, provisional rules on formula_selection."""

    rules = _validated_provisional_thresholds(provisional_thresholds)
    labelled = _labelled_scores(
        selection_features,
        selection_labels,
        cutoffs=cutoffs,
        expected_stage="formula_selection",
    )
    records: list[dict[str, object]] = []
    for rule in rules.to_dict("records"):
        formula = str(rule["formula"])
        track = _resolve_track(str(rule["track"]))
        evaluation = labelled.loc[labelled["formula"].eq(formula)].copy()
        evaluation["decision"] = apply_group_threshold(
            evaluation["score"].to_numpy(dtype=float),
            evaluation["supported"].to_numpy(dtype=bool),
            float(rule["threshold"]),
        )
        metrics = evaluate_group_triage(evaluation)
        record = {
            "formula": formula,
            "track": track.name,
            "threshold": float(rule["threshold"]),
            "threshold_source_stage": "search_calibration",
            "evaluation_stage": "formula_selection",
            "alpha": track.alpha,
            "within_group": track.within_group,
            "protected": track.protected_column,
            "cost": FORMULA_COST[formula],
            "complexity": FORMULA_COMPLEXITY[formula],
            "search_calibration_n_groups": int(rule["calibration_n_groups"]),
            "search_calibration_order_index": int(rule["calibration_order_index"]),
            **metrics,
        }
        record["passes_safety_gate"] = passes_safety_gate(record, track)
        records.append(record)
    return pd.DataFrame(records)


def _validated_selection(selection: Mapping[str, object]) -> dict[str, str]:
    if not isinstance(selection, Mapping):
        raise TypeError("selection must be a mapping")
    state = selection.get("state")
    name = selection.get("name")
    if state == "null_keep_all" and name == "null_keep_all":
        return {"state": "null_keep_all", "name": "null_keep_all"}
    if state == "selected" and type(name) is str and name in FORMULA_NAMES:
        return {"state": "selected", "name": name}
    raise ValueError("selection is not a frozen primary formula or null_keep_all")


def _effective_formula_calibration_rows(
    formula_rows: pd.DataFrame, *, track: TrackSpec
) -> pd.DataFrame:
    """Keep only groups with a supported finite protected row for this rule."""

    scores = pd.to_numeric(formula_rows["score"], errors="coerce").to_numpy(dtype=float)
    eligible = (
        formula_rows[track.protected_column].to_numpy(dtype=bool)
        & formula_rows["supported"].to_numpy(dtype=bool)
        & np.isfinite(scores)
    )
    effective_groups = set(formula_rows.loc[eligible, "rk"].tolist())
    return formula_rows.loc[formula_rows["rk"].isin(effective_groups)].copy()


def fit_final_thresholds(
    threshold_fit_features: pd.DataFrame,
    threshold_fit_labels: pd.DataFrame,
    *,
    cutoffs: DisagreementCutoffs,
    selection: Mapping[str, object],
) -> pd.DataFrame:
    """Fit selected and M5 final rules on threshold_fit only."""

    selected = _validated_selection(selection)
    labelled = _labelled_scores(
        threshold_fit_features,
        threshold_fit_labels,
        cutoffs=cutoffs,
        expected_stage="threshold_calibration",
    )
    n_groups = int(labelled["rk"].nunique())
    records: list[dict[str, object]] = []
    for track in TRACKS.values():
        for role, formula in (
            ("selected", selected["name"]),
            ("m5_baseline", "M5"),
        ):
            if role == "selected" and selected["state"] == "null_keep_all":
                records.append(
                    {
                        "role": role,
                        "formula": "null_keep_all",
                        "track": track.name,
                        "threshold": float("inf"),
                        "threshold_state": "keep_all",
                        "threshold_source_role": "threshold_fit",
                        "alpha": track.alpha,
                        "within_group": track.within_group,
                        "protected": track.protected_column,
                        "operator": "KEEP_ALL",
                        "unsupported_decision": "KEEP",
                        "calibration_n_groups": n_groups,
                        "calibration_order_index": 0,
                    }
                )
                continue
            formula_rows = labelled.loc[labelled["formula"].eq(formula)]
            effective_rows = _effective_formula_calibration_rows(
                formula_rows, track=track
            )
            result = group_conformal_threshold(
                effective_rows,
                alpha=track.alpha,
                valuable_column=track.protected_column,
                score_column="score",
                supported_column="supported",
                within_group=track.within_group,
            )
            threshold = float(result["threshold"])
            if (
                track.name == "primary"
                and int(result["n_groups"]) < PRIMARY_MIN_EFFECTIVE_GROUPS
            ):
                threshold = float("inf")
            elif not np.isfinite(threshold):
                threshold = float("inf")
            operator = (
                "score > threshold" if np.isfinite(threshold) else "KEEP_ALL_SUPPORTED"
            )
            records.append(
                {
                    "role": role,
                    "formula": formula,
                    "track": track.name,
                    "threshold": threshold,
                    "threshold_state": (
                        "finite" if np.isfinite(threshold) else "keep_all"
                    ),
                    "threshold_source_role": "threshold_fit",
                    "alpha": track.alpha,
                    "within_group": track.within_group,
                    "protected": track.protected_column,
                    "operator": operator,
                    "unsupported_decision": "ABSTAIN",
                    "calibration_n_groups": int(result["n_groups"]),
                    "calibration_order_index": int(result["order_index"]),
                }
            )
    return pd.DataFrame(records)


def _validated_final_thresholds(
    rules: pd.DataFrame,
    *,
    selection: Mapping[str, object],
) -> pd.DataFrame:
    selected = _validated_selection(selection)
    if not isinstance(rules, pd.DataFrame):
        raise TypeError("final_thresholds must be a pandas DataFrame")
    required = {
        "role",
        "formula",
        "track",
        "threshold",
        "threshold_state",
        "threshold_source_role",
        "alpha",
        "within_group",
        "operator",
        "unsupported_decision",
    }
    missing = sorted(required - set(rules.columns))
    if missing:
        raise ValueError(f"final thresholds are missing columns: {missing}")
    out = rules.copy()
    pairs = list(zip(out["role"], out["track"], strict=True))
    expected_pairs = {
        (role, track) for role in ("selected", "m5_baseline") for track in TRACKS
    }
    if len(pairs) != len(set(pairs)) or set(pairs) != expected_pairs:
        raise ValueError("final thresholds must contain the exact role/track grid")
    if set(out["threshold_source_role"]) != {"threshold_fit"}:
        raise ValueError("final threshold source must be threshold_fit")
    for row in out.to_dict("records"):
        role = str(row["role"])
        track = _resolve_track(str(row["track"]))
        expected_formula = "M5" if role == "m5_baseline" else selected["name"]
        if row["formula"] != expected_formula:
            raise ValueError("final threshold formula does not match selection")
        if (
            float(row["alpha"]) != track.alpha
            or row["within_group"] != track.within_group
        ):
            raise ValueError("final threshold track semantics mismatch")
        threshold = row["threshold"]
        if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
            raise ValueError("final threshold must be a real number")
        threshold_value = float(threshold)
        if np.isnan(threshold_value) or np.isneginf(threshold_value):
            raise ValueError("final threshold must be finite or positive infinity")
        if expected_formula == "null_keep_all":
            if (
                not np.isposinf(threshold_value)
                or row["operator"] != "KEEP_ALL"
                or row["unsupported_decision"] != "KEEP"
            ):
                raise ValueError("null selection must be an exact keep-all rule")
        else:
            expected_operator = (
                "score > threshold"
                if np.isfinite(threshold_value)
                else "KEEP_ALL_SUPPORTED"
            )
            if (
                row["operator"] != expected_operator
                or row["unsupported_decision"] != "ABSTAIN"
            ):
                raise ValueError("final threshold deployment semantics mismatch")
        expected_state = "finite" if np.isfinite(threshold_value) else "keep_all"
        if row["threshold_state"] != expected_state:
            raise ValueError("final threshold_state is inconsistent with its value")
    return out


def _gate_method_rows(
    labelled: pd.DataFrame,
    prepared_labels: pd.DataFrame,
    *,
    rule: Mapping[str, object],
    method_id: str,
) -> pd.DataFrame:
    formula = str(rule["formula"])
    if formula == "null_keep_all":
        rows = prepared_labels.copy()
        rows["score"] = np.nan
        rows["supported"] = True
        rows["decision"] = "KEEP"
    else:
        rows = labelled.loc[labelled["formula"].eq(formula)].copy()
        rows["decision"] = apply_group_threshold(
            rows["score"].to_numpy(dtype=float),
            rows["supported"].to_numpy(dtype=bool),
            float(rule["threshold"]),
        )
    rows["formula"] = method_id
    rows["source_formula"] = formula
    rows["role"] = str(rule["role"])
    rows["alpha"] = float(rule["alpha"])
    rows["track"] = str(rule["track"])
    rows["threshold"] = float(rule["threshold"])
    columns = [
        "sid",
        "rk",
        "delta_e",
        "exact_min",
        "near_min",
        "valuable",
        "high_energy",
        "decision",
        "formula",
        "source_formula",
        "role",
        "alpha",
        "track",
        "threshold",
        "score",
        "supported",
    ]
    return rows.loc[:, columns].sort_values("sid", kind="stable").reset_index(drop=True)


def evaluate_development_gate(
    development_gate_features: pd.DataFrame,
    development_gate_labels: pd.DataFrame,
    *,
    cutoffs: DisagreementCutoffs,
    selection: Mapping[str, object],
    final_thresholds: pd.DataFrame,
    n_resamples: int = PRODUCTION_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_batch_size: int = 1_000,
) -> dict[str, object]:
    """Apply frozen final rules once on development_gate and compute gates."""

    if type(n_resamples) is not int or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive built-in int")
    if type(bootstrap_seed) is not int:
        raise ValueError("bootstrap_seed must be a built-in int")
    if type(bootstrap_batch_size) is not int or bootstrap_batch_size <= 0:
        raise ValueError("bootstrap_batch_size must be a positive built-in int")
    selected = _validated_selection(selection)
    rules = _validated_final_thresholds(final_thresholds, selection=selected)
    labelled = _labelled_scores(
        development_gate_features,
        development_gate_labels,
        cutoffs=cutoffs,
        expected_stage="threshold_calibration",
    )
    prepared_labels = _prepare_labels(
        development_gate_labels,
        development_gate_features,
        expected_stage="threshold_calibration",
    )

    prediction_parts: list[pd.DataFrame] = []
    metric_records: list[dict[str, object]] = []
    for track in TRACKS.values():
        track_rules = rules.loc[rules["track"].eq(track.name)]
        for role, method_id in (
            ("m5_baseline", "m5_baseline"),
            ("selected", "selected_candidate"),
        ):
            rule = track_rules.loc[track_rules["role"].eq(role)].iloc[0]
            predictions = _gate_method_rows(
                labelled,
                prepared_labels,
                rule=rule,
                method_id=method_id,
            )
            prediction_parts.append(predictions)
            metrics = evaluate_group_triage(predictions)
            metric_records.append(
                {
                    "track": track.name,
                    "method": method_id,
                    "source_formula": str(rule["formula"]),
                    "evaluation_role": "development_gate",
                    "threshold_source_role": "threshold_fit",
                    "threshold": float(rule["threshold"]),
                    "alpha": track.alpha,
                    **metrics,
                    "passes_safety_gate": passes_safety_gate(metrics, track),
                }
            )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    for track in TRACKS:
        track_rows = predictions.loc[predictions["track"].eq(track)]
        baseline = track_rows.loc[track_rows["formula"].eq("m5_baseline")]
        candidate = track_rows.loc[track_rows["formula"].eq("selected_candidate")]
        if (
            not baseline[["sid", "rk"]]
            .reset_index(drop=True)
            .equals(candidate[["sid", "rk"]].reset_index(drop=True))
        ):
            raise ValueError("selected and M5 gate rows are not exactly paired")

    primary_predictions = predictions.loc[predictions["track"].eq("primary")].copy()
    bootstrap = paired_cluster_bootstrap(
        primary_predictions,
        baseline_formula="m5_baseline",
        candidate_formula="selected_candidate",
        alpha=TRACKS["primary"].alpha,
        n_resamples=n_resamples,
        seed=bootstrap_seed,
        batch_size=bootstrap_batch_size,
    )
    bootstrap["batch_size"] = bootstrap_batch_size

    metrics_table = pd.DataFrame(metric_records)
    indexed_metrics = metrics_table.set_index(["track", "method"])
    primary_baseline = indexed_metrics.loc[("primary", "m5_baseline")]
    primary_candidate = indexed_metrics.loc[("primary", "selected_candidate")]
    comparator_candidate = indexed_metrics.loc[("comparator", "selected_candidate")]
    savings_delta = float(
        primary_candidate["dft_savings"] - primary_baseline["dft_savings"]
    )
    abstention_delta = float(
        primary_candidate["abstention_rate"] - primary_baseline["abstention_rate"]
    )
    savings_ci_lower = float(bootstrap["metrics"]["dft_savings"]["difference_ci_95"][0])
    valuable_ci_lower = float(
        bootstrap["metrics"]["valuable_item_recall"]["difference_ci_95"][0]
    )
    gate = {
        "evaluation_role": "development_gate",
        "threshold_source_role": "threshold_fit",
        "selection_state": selected["state"],
        "selected_formula": selected["name"],
        "dft_savings_delta": savings_delta,
        "dft_savings_delta_required": 0.03,
        "dft_savings_paired_ci_95_lower": savings_ci_lower,
        "valuable_item_recall_difference_ci_95_lower": valuable_ci_lower,
        "valuable_item_recall_noninferiority_floor": -0.005,
        "abstention_rate_delta": abstention_delta,
        "abstention_rate_delta_max": 0.01,
        "passes_dft_savings_magnitude": bool(savings_delta >= 0.03),
        "passes_dft_savings_paired_ci": bool(savings_ci_lower > 0.0),
        "passes_valuable_recall_noninferiority": bool(valuable_ci_lower >= -0.005),
        "passes_abstention_delta": bool(abstention_delta <= 0.01),
        "passes_primary_safety": passes_safety_gate(primary_candidate, "primary"),
        "passes_comparator_safety": passes_safety_gate(
            comparator_candidate, "comparator"
        ),
    }
    gate["passes_improvement_gate"] = bool(
        selected["state"] == "selected"
        and gate["passes_dft_savings_magnitude"]
        and gate["passes_dft_savings_paired_ci"]
        and gate["passes_valuable_recall_noninferiority"]
        and gate["passes_abstention_delta"]
        and gate["passes_primary_safety"]
        and gate["passes_comparator_safety"]
    )
    return {
        "predictions": predictions,
        "metrics": metrics_table,
        "paired_bootstrap": bootstrap,
        "improvement_gate": gate,
    }


def split_threshold_groups(
    threshold_features: pd.DataFrame,
    *,
    salt: str = THRESHOLD_SPLIT_SALT,
) -> pd.DataFrame:
    """Split whole threshold-calibration groups before endpoint labels open."""

    if salt != THRESHOLD_SPLIT_SALT:
        raise ValueError("split salt must equal the frozen protocol salt")
    validated = _validated_features(threshold_features)
    _require_single_stage(validated, "threshold_calibration")
    groups = sorted(set(validated["rk"].tolist()))
    if len(groups) < 2:
        raise ValueError(
            "threshold_calibration requires at least two complete rk groups"
        )
    digest_by_group = {
        rk: hashlib.sha256(f"{salt}\0{rk}".encode("utf-8")).hexdigest() for rk in groups
    }
    ordered_groups = sorted(groups, key=lambda rk: (digest_by_group[rk], rk))
    rank_by_group = {rk: index for index, rk in enumerate(ordered_groups)}
    fit_count = len(ordered_groups) // 2
    fit_groups = set(ordered_groups[:fit_count])
    assignment = validated[["sid", "rk", "stage"]].copy()
    assignment["threshold_role"] = np.where(
        assignment["rk"].isin(fit_groups),
        "threshold_fit",
        "development_gate",
    )
    assignment["split_rank"] = assignment["rk"].map(rank_by_group).astype(int)
    assignment["split_key_sha256"] = assignment["rk"].map(digest_by_group)
    assignment["split_salt"] = salt
    return assignment


@dataclass(frozen=True)
class _InputSnapshot:
    data: bytes
    sha256: str


def _snapshot_input(path: Path, *, role: str) -> _InputSnapshot:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise OSError(f"failed to snapshot {role}") from exc
    return _InputSnapshot(
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _read_parquet_snapshot(
    snapshot: _InputSnapshot,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(snapshot.data), columns=columns)


def _read_json_document(data: bytes, *, role: str) -> dict[str, object]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role} JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid {role} document")
    return payload


def _is_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_feature_manifest(
    manifest: Mapping[str, object],
    *,
    features_path: Path,
    feature_sha256: str,
    checkpoints: Mapping[str, Path],
    checkpoint_sha256: Mapping[str, str],
) -> dict[str, object]:
    if not isinstance(manifest, dict):
        raise ValueError("invalid feature manifest document")
    if manifest.get("protocol") != FEATURE_PROTOCOL:
        raise ValueError("feature manifest protocol mismatch")
    if manifest.get("mode") != "development":
        raise ValueError("feature manifest mode must be development")
    if manifest.get("production_protocol_eligible") is not True:
        raise ValueError(
            "feature manifest must be exactly production-protocol eligible"
        )
    if manifest.get("evidence_role") != "protocol_feature_generation":
        raise ValueError("feature manifest evidence role mismatch")
    adapter = manifest.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("mode") != "builtin_mattersim":
        raise ValueError("feature manifest adapter must be builtin_mattersim")
    implementation = adapter.get("implementation")
    if not isinstance(implementation, dict):
        raise ValueError("feature manifest adapter implementation is missing")
    if implementation.get("source_hash_verified") is not True:
        raise ValueError("feature manifest implementation source hash is not verified")
    stages = manifest.get("stages")
    if (
        not isinstance(stages, list)
        or len(stages) != len(DEVELOPMENT_STAGES)
        or len(set(stages)) != len(stages)
        or set(stages) != set(DEVELOPMENT_STAGES)
    ):
        raise ValueError(
            "feature manifest stages must be exactly the three development stages"
        )

    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, dict):
        raise ValueError("feature manifest outputs_sha256 is missing")
    recorded_feature_sha256 = outputs.get(features_path.name)
    if recorded_feature_sha256 != feature_sha256:
        raise ValueError("feature manifest feature output hash mismatch")
    if outputs != {features_path.name: feature_sha256}:
        raise ValueError("feature manifest output hash closure mismatch")
    if features_path.name != FEATURE_OUTPUT_NAME:
        raise ValueError("feature parquet name does not match its protocol")

    checkpoint_records = manifest.get("checkpoints")
    loaded_checkpoint_sha256 = manifest.get("predictor_loaded_checkpoint_sha256")
    if not isinstance(checkpoint_records, dict) or not isinstance(
        loaded_checkpoint_sha256, dict
    ):
        raise ValueError("feature manifest checkpoint records are missing")
    if set(checkpoint_records) != {"m1", "m5"} or set(loaded_checkpoint_sha256) != {
        "m1",
        "m5",
    }:
        raise ValueError("feature manifest checkpoints must be exactly m1 and m5")
    for model in ("m1", "m5"):
        record = checkpoint_records[model]
        if not isinstance(record, dict):
            raise ValueError("feature manifest checkpoint record is invalid")
        if record.get("sha256") != checkpoint_sha256[model]:
            raise ValueError("feature manifest checkpoint hash mismatch")
        if record.get("path") != str(checkpoints[model].resolve()):
            raise ValueError("feature manifest checkpoint path mismatch")
        if loaded_checkpoint_sha256.get(model) != checkpoint_sha256[model]:
            raise ValueError("feature predictor loaded checkpoint hash mismatch")

    repository_root = Path(__file__).resolve().parents[1]
    expected_sources = {
        relative: _sha256_file(repository_root / relative)
        for relative in _FEATURE_EXECUTED_SOURCE_RELATIVE
    }
    source_hashes = manifest.get("executed_source_sha256")
    if source_hashes != expected_sources:
        raise ValueError("feature manifest executed source hash closure mismatch")
    implementation_relative = _FEATURE_EXECUTED_SOURCE_RELATIVE[0]
    implementation_source_path = str(
        (repository_root / implementation_relative).resolve()
    )
    implementation_source_sha256 = source_hashes[implementation_relative]
    if implementation.get("source_path") != implementation_source_path:
        raise ValueError(
            "feature manifest implementation source path does not match "
            "the executed-source closure"
        )
    if (
        not _is_sha256(implementation.get("source_sha256"))
        or implementation.get("source_sha256") != implementation_source_sha256
    ):
        raise ValueError(
            "feature manifest implementation source hash does not match "
            "the executed-source closure"
        )
    inputs = manifest.get("inputs_sha256")
    if not isinstance(inputs, dict) or set(inputs) != {
        "frames",
        "metadata",
        "stage_assignments",
    }:
        raise ValueError("feature manifest input hash closure mismatch")
    for record in inputs.values():
        if (
            not isinstance(record, dict)
            or type(record.get("path")) is not str
            or not _is_sha256(record.get("sha256"))
        ):
            raise ValueError("feature manifest input hash record is invalid")
    if manifest.get("integrity") != {"prepublish_rehash": "passed"}:
        raise ValueError("feature manifest integrity record mismatch")
    return manifest


def _validate_full_development_features(features: pd.DataFrame) -> pd.DataFrame:
    validated = _validated_features(features)
    observed = set(validated["stage"].tolist())
    if observed != set(DEVELOPMENT_STAGES):
        raise ValueError(
            "features must contain exactly the three development stages; "
            "test or missing stages are forbidden"
        )
    stages_per_group = validated.groupby("rk", sort=False)["stage"].nunique()
    if (stages_per_group != 1).any():
        raise ValueError("each rk group must belong to exactly one stage")
    return validated


def _prepare_full_development_labels(
    labels: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    if not isinstance(labels, pd.DataFrame):
        raise TypeError("labels must be a pandas DataFrame")
    required = {"sid", "rk", "stage", "e_per_atom"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"labels are missing columns: {missing}")
    raw = labels.loc[:, ["sid", "rk", "stage", "e_per_atom"]].copy()
    if raw["sid"].duplicated().any() or raw.duplicated(["sid", "rk"]).any():
        raise ValueError("labels contain duplicate sid/rk keys")
    if set(raw["stage"].tolist()) != set(DEVELOPMENT_STAGES):
        raise ValueError("labels must contain exactly the three development stages")
    prepared = []
    for stage in DEVELOPMENT_STAGES:
        stage_features = features.loc[features["stage"].eq(stage)].copy()
        stage_labels = raw.loc[raw["stage"].eq(stage)].copy()
        prepared.append(
            _prepare_labels(stage_labels, stage_features, expected_stage=stage)
        )
    result = pd.concat(prepared, ignore_index=True)
    if len(result) != len(features):
        raise ValueError("feature and label rows are not exactly aligned")
    return result


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    raise TypeError(f"value is not strict-JSON serializable: {type(value)!r}")


def _write_strict_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(
            _json_safe(payload),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _strict_json_document(path: Path, *, role: str) -> dict[str, object]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid strict staged JSON: {role}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid strict staged document: {role}")
    return payload


def _validate_staged_artifacts(
    staging_dir: Path, *, manifest: Mapping[str, object]
) -> None:
    actual_names = {path.name for path in staging_dir.iterdir()}
    if actual_names != set(OUTPUT_NAMES):
        raise RuntimeError("staging directory does not contain exact artifacts")
    for name in (
        "threshold_role_assignments.parquet",
        "development_frontier.parquet",
        "threshold_fit_rules.parquet",
        "development_gate_metrics.parquet",
    ):
        pd.read_parquet(staging_dir / name)
    for name in (
        "PAIRED_BOOTSTRAP.json",
        "IMPROVEMENT_GATE.json",
        "FROZEN_PROTOCOL.json",
        "MANIFEST.json",
    ):
        _strict_json_document(staging_dir / name, role=name)
    output_hashes = manifest.get("outputs_sha256")
    if not isinstance(output_hashes, Mapping):
        raise RuntimeError("manifest output hashes are missing")
    if set(output_hashes) != set(OUTPUT_NAMES) - {"MANIFEST.json"}:
        raise RuntimeError("manifest output hash closure mismatch")
    for name, expected in output_hashes.items():
        if _sha256_file(staging_dir / str(name)) != expected:
            raise RuntimeError(f"staged output hash mismatch: {name}")


def _verify_unchanged(
    records: Sequence[tuple[str, Path, str]],
) -> None:
    for role, path, expected in records:
        try:
            current = _sha256_file(path)
        except OSError as exc:
            raise RuntimeError(f"{role} changed after initial hash") from exc
        if current != expected:
            raise RuntimeError(f"{role} changed after initial hash")


def _runtime_record() -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }


def _json_rule_records(rules: pd.DataFrame) -> list[dict[str, object]]:
    return [dict(_json_safe(record)) for record in rules.to_dict("records")]


def run_development_freeze(
    features_path: Path,
    labels_path: Path,
    feature_manifest_path: Path,
    output_dir: Path,
    *,
    checkpoints: Mapping[str, Path],
    n_resamples: int = PRODUCTION_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_batch_size: int = 1_000,
) -> dict[str, object]:
    """Select, calibrate, gate, and atomically freeze development evidence."""

    if type(n_resamples) is not int or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive built-in int")
    if type(bootstrap_seed) is not int:
        raise ValueError("bootstrap_seed must be a built-in int")
    if type(bootstrap_batch_size) is not int or bootstrap_batch_size <= 0:
        raise ValueError("bootstrap_batch_size must be a positive built-in int")
    features_path = Path(features_path)
    labels_path = Path(labels_path)
    feature_manifest_path = Path(feature_manifest_path)
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {output_dir}"
        )
    if set(checkpoints) != {"m1", "m5"}:
        raise ValueError("checkpoints must contain exactly m1 and m5")
    checkpoint_paths = {model: Path(checkpoints[model]) for model in ("m1", "m5")}
    for path in (
        features_path,
        labels_path,
        feature_manifest_path,
        *checkpoint_paths.values(),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        relative: repository_root / relative for relative in _EXECUTED_SOURCE_RELATIVE
    }
    input_paths = {
        "features": features_path,
        "labels": labels_path,
        "feature_manifest": feature_manifest_path,
    }
    feature_snapshot = _snapshot_input(features_path, role="features")
    feature_manifest_snapshot = _snapshot_input(
        feature_manifest_path, role="feature manifest"
    )
    feature_manifest = _read_json_document(
        feature_manifest_snapshot.data, role="feature manifest"
    )
    input_sha256 = {
        "features": feature_snapshot.sha256,
        "feature_manifest": feature_manifest_snapshot.sha256,
    }
    executed_source_sha256 = {
        relative: _sha256_file(path) for relative, path in source_paths.items()
    }
    checkpoint_sha256 = {
        model: _sha256_file(path) for model, path in checkpoint_paths.items()
    }
    if checkpoint_sha256 != dict(FROZEN_CHECKPOINT_SHA256):
        raise ValueError("checkpoint hash mismatch with frozen production identities")
    _validate_feature_manifest(
        feature_manifest,
        features_path=features_path,
        feature_sha256=input_sha256["features"],
        checkpoints=checkpoint_paths,
        checkpoint_sha256=checkpoint_sha256,
    )
    features = _validate_full_development_features(
        _read_parquet_snapshot(feature_snapshot)
    )
    search_features = features.loc[features["stage"].eq("search_calibration")].copy()
    cutoffs = derive_disagreement_cutoffs(search_features)
    threshold_features = features.loc[
        features["stage"].eq("threshold_calibration")
    ].copy()
    role_assignments = split_threshold_groups(threshold_features)

    # The label snapshot and its hash are deliberately created only after all
    # label-free validation, cutoff derivation, and deterministic role split.
    label_snapshot = _snapshot_input(labels_path, role="labels")
    input_sha256["labels"] = label_snapshot.sha256
    raw_labels = _read_parquet_snapshot(
        label_snapshot,
        columns=["sid", "rk", "stage", "e_per_atom"],
    )
    labels = _prepare_full_development_labels(raw_labels, features)
    feature_stages = {
        stage: features.loc[features["stage"].eq(stage)].copy()
        for stage in DEVELOPMENT_STAGES
    }
    label_stages = {
        stage: labels.loc[labels["stage"].eq(stage)].copy()
        for stage in DEVELOPMENT_STAGES
    }

    provisional_thresholds = fit_provisional_thresholds(
        feature_stages["search_calibration"],
        label_stages["search_calibration"],
        cutoffs=cutoffs,
    )
    frontier = evaluate_formula_selection(
        feature_stages["formula_selection"],
        label_stages["formula_selection"],
        cutoffs=cutoffs,
        provisional_thresholds=provisional_thresholds,
    )
    selection = select_primary_formula(frontier)

    threshold_roles = role_assignments.set_index("sid")["threshold_role"]
    fit_sids = set(threshold_roles.index[threshold_roles.eq("threshold_fit")].tolist())
    gate_sids = set(
        threshold_roles.index[threshold_roles.eq("development_gate")].tolist()
    )
    if fit_sids & gate_sids or fit_sids | gate_sids != set(threshold_features["sid"]):
        raise RuntimeError("threshold role assignment is not a partition")
    threshold_labels = label_stages["threshold_calibration"]
    fit_features = threshold_features.loc[
        threshold_features["sid"].isin(fit_sids)
    ].copy()
    fit_labels = threshold_labels.loc[threshold_labels["sid"].isin(fit_sids)].copy()
    gate_features = threshold_features.loc[
        threshold_features["sid"].isin(gate_sids)
    ].copy()
    gate_labels = threshold_labels.loc[threshold_labels["sid"].isin(gate_sids)].copy()
    final_rules = fit_final_thresholds(
        fit_features,
        fit_labels,
        cutoffs=cutoffs,
        selection=selection,
    )
    gate_result = evaluate_development_gate(
        gate_features,
        gate_labels,
        cutoffs=cutoffs,
        selection=selection,
        final_thresholds=final_rules,
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_batch_size=bootstrap_batch_size,
    )

    catalog_serialized = serialize_formula_catalog(cutoffs)
    catalog_sha256 = hashlib.sha256(catalog_serialized.encode("utf-8")).hexdigest()
    bootstrap_parameters = {
        "method": "paired percentile bootstrap over rk composition clusters",
        "seed": bootstrap_seed,
        "n_resamples": n_resamples,
        "batch_size": bootstrap_batch_size,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        artifact_paths = {name: staging_dir / name for name in OUTPUT_NAMES}
        role_assignments.to_parquet(
            artifact_paths["threshold_role_assignments.parquet"], index=False
        )
        frontier.to_parquet(artifact_paths["development_frontier.parquet"], index=False)
        final_rules.to_parquet(
            artifact_paths["threshold_fit_rules.parquet"], index=False
        )
        gate_result["metrics"].to_parquet(
            artifact_paths["development_gate_metrics.parquet"], index=False
        )
        _write_strict_json(
            artifact_paths["PAIRED_BOOTSTRAP.json"],
            gate_result["paired_bootstrap"],
        )
        _write_strict_json(
            artifact_paths["IMPROVEMENT_GATE.json"],
            gate_result["improvement_gate"],
        )
        development_artifacts_sha256 = {
            name: _sha256_file(path)
            for name, path in artifact_paths.items()
            if name
            in {
                "threshold_role_assignments.parquet",
                "development_frontier.parquet",
                "threshold_fit_rules.parquet",
                "development_gate_metrics.parquet",
                "PAIRED_BOOTSTRAP.json",
                "IMPROVEMENT_GATE.json",
            }
        }
        frozen: dict[str, object] = {
            "protocol": DEVELOPMENT_FREEZE_PROTOCOL,
            "state": "frozen",
            "split": {
                "salt": THRESHOLD_SPLIT_SALT,
                "ordering": "sha256(salt+'\\0'+rk),rk",
                "threshold_fit_groups": len(
                    set(
                        role_assignments.loc[
                            role_assignments["threshold_role"].eq("threshold_fit"),
                            "rk",
                        ]
                    )
                ),
                "development_gate_groups": len(
                    set(
                        role_assignments.loc[
                            role_assignments["threshold_role"].eq("development_gate"),
                            "rk",
                        ]
                    )
                ),
            },
            "catalog": {
                "serialized": catalog_serialized,
                "sha256": catalog_sha256,
            },
            "cutoffs": {
                "q99_ev_per_atom": cutoffs.q99_ev_per_atom,
                "q995_ev_per_atom": cutoffs.q995_ev_per_atom,
                "q995_force_ev_per_a": cutoffs.q995_force_ev_per_a,
                "eligible_row_count": cutoffs.eligible_row_count,
                "source_stage": cutoffs.source_stage,
                "quantile_method": cutoffs.quantile_method,
                "calibration_fingerprint_sha256": (
                    cutoffs.calibration_fingerprint_sha256
                ),
            },
            "cutoff_provenance": {
                "feature_sha256": input_sha256["features"],
                "feature_manifest_sha256": input_sha256["feature_manifest"],
                "protocol_code_sha256": executed_source_sha256[
                    "src/next8_mattersim_committee_protocol.py"
                ],
                "catalog_serialization_sha256": catalog_sha256,
            },
            "tracks": {
                name: {
                    "protected": track.protected_column,
                    "protected_ev_per_atom": track.protected_ev_per_atom,
                    "within_group": track.within_group,
                    "alpha": track.alpha,
                }
                for name, track in TRACKS.items()
            },
            "selection": dict(selection),
            "final_rules": _json_rule_records(final_rules),
            "bootstrap": bootstrap_parameters,
            "improvement_gate": gate_result["improvement_gate"],
            "development_artifacts_sha256": development_artifacts_sha256,
        }
        _write_strict_json(artifact_paths["FROZEN_PROTOCOL.json"], frozen)

        outputs_sha256 = {
            name: _sha256_file(path)
            for name, path in artifact_paths.items()
            if name != "MANIFEST.json"
        }
        manifest: dict[str, object] = {
            "protocol": DEVELOPMENT_FREEZE_PROTOCOL,
            "state": "frozen",
            "feature_protocol": FEATURE_PROTOCOL,
            "inputs_sha256": {
                role: {
                    "path": str(path.resolve()),
                    "sha256": input_sha256[role],
                }
                for role, path in input_paths.items()
            },
            "checkpoints": {
                model: {
                    "path": str(path.resolve()),
                    "sha256": checkpoint_sha256[model],
                }
                for model, path in checkpoint_paths.items()
            },
            "executed_source_sha256": executed_source_sha256,
            "catalog_serialization_sha256": catalog_sha256,
            "split": frozen["split"],
            "bootstrap": bootstrap_parameters,
            "runtime": _runtime_record(),
            "counts": {
                "feature_rows": len(features),
                "label_rows": len(labels),
                "search_calibration_groups": int(
                    feature_stages["search_calibration"]["rk"].nunique()
                ),
                "formula_selection_groups": int(
                    feature_stages["formula_selection"]["rk"].nunique()
                ),
                "threshold_fit_groups": int(fit_features["rk"].nunique()),
                "development_gate_groups": int(gate_features["rk"].nunique()),
            },
            "integrity": {"prepublish_rehash": "passed"},
            "outputs_sha256": outputs_sha256,
        }
        _write_strict_json(artifact_paths["MANIFEST.json"], manifest)
        _validate_staged_artifacts(staging_dir, manifest=manifest)

        unchanged_records = [
            *((role, path, input_sha256[role]) for role, path in input_paths.items()),
            *(
                (
                    f"checkpoint {model}",
                    path,
                    checkpoint_sha256[model],
                )
                for model, path in checkpoint_paths.items()
            ),
            *(
                (
                    f"executed source {relative}",
                    path,
                    executed_source_sha256[relative],
                )
                for relative, path in source_paths.items()
            ),
        ]
        _verify_unchanged(unchanged_records)
        _atomic_publish_directory_no_replace(staging_dir, output_dir)
        return dict(_json_safe(manifest))
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m1-checkpoint", type=Path, required=True)
    parser.add_argument("--m5-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--n-resamples", type=int, default=PRODUCTION_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-batch-size", type=int, default=1_000)
    args = parser.parse_args(argv)
    manifest = run_development_freeze(
        args.features,
        args.labels,
        args.feature_manifest,
        args.output,
        checkpoints={
            "m1": args.m1_checkpoint,
            "m5": args.m5_checkpoint,
        },
        n_resamples=args.n_resamples,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_batch_size=args.bootstrap_batch_size,
    )
    print(json.dumps(manifest["outputs_sha256"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "BOOTSTRAP_SEED",
    "DEVELOPMENT_FREEZE_PROTOCOL",
    "DEVELOPMENT_STAGES",
    "DisagreementCutoffs",
    "FORMULA_COMPLEXITY",
    "FORMULA_COST",
    "FORMULA_NAMES",
    "OUTPUT_NAMES",
    "PRODUCTION_BOOTSTRAP_RESAMPLES",
    "QUANTILE_METHOD",
    "THRESHOLD_SPLIT_SALT",
    "TRACKS",
    "TrackSpec",
    "construct_committee_scores",
    "derive_disagreement_cutoffs",
    "evaluate_development_gate",
    "evaluate_formula_selection",
    "fit_final_thresholds",
    "fit_provisional_thresholds",
    "main",
    "passes_safety_gate",
    "run_development_freeze",
    "select_primary_formula",
    "serialize_formula_catalog",
    "split_threshold_groups",
)
