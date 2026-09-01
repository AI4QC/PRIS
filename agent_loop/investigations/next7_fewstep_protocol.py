#!/usr/bin/env python3
"""Freeze the finite MatterSim few-step catalog using development data only."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.next6_elementa_protocol import (
    apply_group_threshold,
    attach_energy_labels,
    evaluate_group_triage,
    group_conformal_threshold,
)


FEATURE_PROTOCOL = "2026-08-01-mattersim-fewstep-prerelax-v1"
DEVELOPMENT_FREEZE_PROTOCOL = (
    "2026-08-01-mattersim-fewstep-development-freeze-v1"
)
EVIDENCE_ROLE = "historically seen discovery; not confirmatory"
DEVELOPMENT_STAGES = (
    "search_calibration",
    "formula_selection",
    "threshold_calibration",
)
OUTPUT_NAMES = (
    "development_frontier.parquet",
    "threshold_calibration_rules.parquet",
    "FROZEN_PROTOCOL.json",
    "MANIFEST.json",
)


@dataclass(frozen=True)
class FormulaSpec:
    """One member of the frozen, weight-free energy catalog."""

    name: str
    energy_source: tuple[str, ...]
    max_step: int
    cost: int

    @property
    def support_column(self) -> str:
        return f"k{self.max_step}_supported"

    def as_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "energy_source": list(self.energy_source),
            "max_step": self.max_step,
            "cost": self.cost,
        }


FROZEN_CATALOG = (
    FormulaSpec("S0", ("k0_energy_ev_per_atom",), 0, 1),
    FormulaSpec("S2", ("k2_energy_ev_per_atom",), 2, 3),
    FormulaSpec("S4", ("k4_energy_ev_per_atom",), 4, 5),
    FormulaSpec("S8", ("k8_energy_ev_per_atom",), 8, 9),
    FormulaSpec(
        "Sbest4",
        (
            "k0_energy_ev_per_atom",
            "k2_energy_ev_per_atom",
            "k4_energy_ev_per_atom",
        ),
        4,
        5,
    ),
    FormulaSpec(
        "Sbest8",
        (
            "k0_energy_ev_per_atom",
            "k2_energy_ev_per_atom",
            "k4_energy_ev_per_atom",
            "k8_energy_ev_per_atom",
        ),
        8,
        9,
    ),
)
CATALOG = FROZEN_CATALOG
CATALOG_BY_NAME = {formula.name: formula for formula in FROZEN_CATALOG}


@dataclass(frozen=True)
class TrackSpec:
    """One of the two and only two frozen risk semantics."""

    name: str
    protected: str
    protected_ev_per_atom: float
    within_group: str
    alpha: float

    def as_record(self) -> dict[str, object]:
        protected_gate = (
            "valuable_group_retention_lower"
            if self.name == "primary"
            else "near_min_retention_lower"
        )
        return {
            "name": self.name,
            "protected": self.protected,
            "protected_ev_per_atom": self.protected_ev_per_atom,
            "within_group": self.within_group,
            "alpha": self.alpha,
            "safety_gate": {
                "exact_min_retention_lower_min": 0.95,
                f"{protected_gate}_min": 0.95,
                "regret_p95_max_ev_per_atom": 0.05,
                "all_rejected_groups_max": 0,
            },
        }


TRACKS = {
    "primary": TrackSpec("primary", "valuable", 0.05, "max", 0.01),
    "comparator": TrackSpec(
        "comparator", "near_min", 0.001, "min", 0.035
    ),
}
FROZEN_TRACKS = TRACKS


def candidate_catalog() -> list[FormulaSpec]:
    """Return a fresh list containing exactly the six frozen formulas."""

    return list(FROZEN_CATALOG)


def _resolve_formula(formula: FormulaSpec | Mapping[str, object] | str) -> FormulaSpec:
    if isinstance(formula, FormulaSpec):
        canonical = CATALOG_BY_NAME.get(formula.name)
        if canonical != formula:
            raise ValueError("formula is not a member of the frozen catalog")
        return canonical
    if isinstance(formula, Mapping):
        name = formula.get("name")
    else:
        name = formula
    if not isinstance(name, str) or name not in CATALOG_BY_NAME:
        raise ValueError(f"unknown frozen formula: {name!r}")
    return CATALOG_BY_NAME[name]


def _resolve_track(track: TrackSpec | str) -> TrackSpec:
    name = track.name if isinstance(track, TrackSpec) else track
    if not isinstance(name, str) or name not in TRACKS:
        raise ValueError(f"unknown frozen track: {name!r}")
    canonical = TRACKS[name]
    if isinstance(track, TrackSpec) and track != canonical:
        raise ValueError("track does not match the frozen semantics")
    return canonical


def _normalize_key_columns(data: pd.DataFrame, *, table_name: str) -> pd.DataFrame:
    required = {"sid", "rk"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{table_name} is missing key columns: {sorted(missing)}")
    out = data.copy()
    if out["sid"].isna().any() or out["rk"].isna().any():
        raise ValueError(f"{table_name} sid/rk keys must be nonmissing")
    out["sid"] = out["sid"].astype(str)
    out["rk"] = out["rk"].astype(str)
    if out["sid"].duplicated().any():
        raise ValueError(f"{table_name} sid values must be unique")
    return out


def _validate_feature_stages(
    features: pd.DataFrame, *, require_all: bool
) -> pd.DataFrame:
    out = _normalize_key_columns(features, table_name="features")
    if "stage" not in out.columns:
        raise ValueError("features are missing stage")
    if out["stage"].isna().any():
        raise ValueError("feature stages must be nonmissing")
    out["stage"] = out["stage"].astype(str)
    observed = set(out["stage"])
    allowed = set(DEVELOPMENT_STAGES)
    if "test" in observed or not observed.issubset(allowed):
        raise ValueError("features must contain development stages only; test is forbidden")
    if require_all and observed != allowed:
        raise ValueError(
            "features must contain exactly the three development stages"
        )
    stages_per_group = out.groupby("rk", sort=False)["stage"].nunique()
    if (stages_per_group > 1).any():
        raise ValueError("each rk group must belong to exactly one development stage")
    return out


def _strict_support(values: pd.Series) -> np.ndarray:
    return np.asarray(
        [
            bool(value)
            if isinstance(value, (bool, np.bool_))
            else False
            for value in values.to_numpy(dtype=object)
        ],
        dtype=bool,
    )


def prepare_fewstep_scores(
    features: pd.DataFrame,
    formula: FormulaSpec | Mapping[str, object] | str,
) -> pd.DataFrame:
    """Compute same-composition gaps using only a formula's prefix support.

    Scores use no endpoint labels and no identifier-derived values.  If fewer
    than two candidates in a composition have finite, prefix-supported energy,
    every candidate in that composition is marked unsupported.
    """

    spec = _resolve_formula(formula)
    data = _validate_feature_stages(features, require_all=False)
    required = {*spec.energy_source, spec.support_column}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"features are missing {spec.name} columns: {sorted(missing)}"
        )
    energy_matrix = np.column_stack(
        [
            pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
            for column in spec.energy_source
        ]
    )
    finite_energy = np.isfinite(energy_matrix).all(axis=1)
    formula_energy = np.min(energy_matrix, axis=1)
    prefix_support = _strict_support(data[spec.support_column])
    preliminary_support = prefix_support & finite_energy
    support_count = (
        pd.Series(preliminary_support, index=data.index)
        .groupby(data["rk"], sort=False)
        .transform("sum")
        .to_numpy(dtype=int)
    )
    supported = preliminary_support & (support_count >= 2)
    supported_energy = pd.Series(
        np.where(supported, formula_energy, np.nan), index=data.index
    )
    group_min = supported_energy.groupby(data["rk"], sort=False).transform("min")
    score = np.where(
        supported,
        formula_energy - group_min.to_numpy(dtype=float),
        np.nan,
    )
    return pd.DataFrame(
        {
            "sid": data["sid"].to_numpy(dtype=object),
            "rk": data["rk"].to_numpy(dtype=object),
            "stage": data["stage"].to_numpy(dtype=object),
            "formula": spec.name,
            "formula_energy_ev_per_atom": formula_energy,
            "supported": supported,
            "score": score,
        }
    )


def _metric_float(metrics: Mapping[str, object], name: str) -> float:
    try:
        return float(metrics[name])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def passes_safety_gate(
    metrics: Mapping[str, object], track: TrackSpec | str
) -> bool:
    """Apply the immutable safety gate for one frozen track."""

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
        exact >= 0.95
        and protected >= 0.95
        and regret <= 0.05
        and all_rejected == 0.0
    )


def select_frozen_rule(
    frontier: pd.DataFrame, track: TrackSpec | str
) -> dict[str, object]:
    """Select by savings, cost, max step, then frozen catalog order."""

    track_spec = _resolve_track(track)
    required = {
        "name",
        "track",
        "dft_savings",
        "cost",
        "max_step",
        "exact_min_retention_lower",
        "near_min_retention_lower",
        "valuable_group_retention_lower",
        "regret_p95",
        "all_rejected_groups",
    }
    missing = required - set(frontier.columns)
    if missing:
        raise ValueError(f"frontier is missing columns: {sorted(missing)}")
    unknown_names = sorted(
        {
            repr(name)
            for name in frontier["name"].to_numpy(dtype=object)
            if not isinstance(name, str) or name not in CATALOG_BY_NAME
        }
    )
    if unknown_names:
        raise ValueError(
            f"frontier contains names outside the frozen catalog: {unknown_names}"
        )
    candidates = frontier.loc[frontier["track"].eq(track_spec.name)].copy()
    candidates = candidates.loc[
        [
            passes_safety_gate(row, track_spec)
            for row in candidates.to_dict("records")
        ]
    ]
    if candidates.empty:
        return {"state": "null_keep_all", "name": "null_keep_all"}
    catalog_order = {
        formula.name: index for index, formula in enumerate(FROZEN_CATALOG)
    }
    candidates["_catalog_order"] = candidates["name"].map(catalog_order)
    candidates = candidates.sort_values(
        ["dft_savings", "cost", "max_step", "_catalog_order"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    row = candidates.iloc[0].to_dict()
    return {"state": "selected", **row}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_document(path: Path, *, role: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role} JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid {role} document")
    return payload


def _validate_digest_map(value: object, *, role: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"feature manifest {role} must be a hash mapping")
    result: dict[str, str] = {}
    for key, digest in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"feature manifest {role} contains an invalid SHA-256")
        result[key] = digest
    return result


def _validate_feature_manifest(
    feature_manifest_path: Path,
    *,
    features_path: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
) -> dict[str, str]:
    manifest = _read_json_document(
        feature_manifest_path, role="feature manifest"
    )
    if manifest.get("protocol") != FEATURE_PROTOCOL:
        raise ValueError("feature manifest protocol mismatch")
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
    if manifest.get("evidence_role") != EVIDENCE_ROLE:
        raise ValueError("feature manifest evidence role mismatch")

    outputs = _validate_digest_map(
        manifest.get("outputs_sha256"), role="outputs_sha256"
    )
    expected_feature_sha256 = outputs.get(features_path.name)
    if expected_feature_sha256 is None:
        raise ValueError("feature manifest does not hash the feature parquet")
    if expected_feature_sha256 != _sha256_file(features_path):
        raise ValueError("feature manifest feature parquet hash mismatch")

    model = manifest.get("model")
    if not isinstance(model, dict):
        raise ValueError("feature manifest model record is missing")
    if model.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("feature manifest checkpoint hash mismatch")
    checkpoint_recorded_path = model.get("checkpoint")
    if not isinstance(checkpoint_recorded_path, str) or not checkpoint_recorded_path:
        raise ValueError("feature manifest checkpoint path is missing")
    checkpoint_name = Path(checkpoint_recorded_path).name

    inputs = _validate_digest_map(
        manifest.get("inputs_sha256"), role="inputs_sha256"
    )
    if inputs.get(checkpoint_name) != checkpoint_sha256:
        raise ValueError("feature manifest checkpoint input hash mismatch")
    feature_inputs = dict(inputs)
    feature_inputs.pop(checkpoint_name)
    if len(feature_inputs) != 4:
        raise ValueError(
            "feature manifest must contain exactly four feature-input hashes"
        )
    return feature_inputs


def _prepare_labels(
    labels: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    normalized = _normalize_key_columns(labels, table_name="labels")
    if "e_per_atom" not in normalized.columns:
        raise ValueError("labels are missing e_per_atom")
    feature_keys = set(zip(features["sid"], features["rk"], strict=True))
    label_keys = set(zip(normalized["sid"], normalized["rk"], strict=True))
    if feature_keys != label_keys:
        raise ValueError("feature and label sid/rk keys differ")
    if "stage" in normalized.columns:
        if normalized["stage"].isna().any():
            raise ValueError("label stages must be nonmissing")
        label_stages = normalized["stage"].astype(str)
        if not set(label_stages).issubset(set(DEVELOPMENT_STAGES)):
            raise ValueError("labels must contain development stages only")
        expected = features.set_index("sid")["stage"]
        observed = pd.Series(
            label_stages.to_numpy(dtype=object),
            index=normalized["sid"],
        )
        if not observed.sort_index().equals(expected.sort_index()):
            raise ValueError("feature and label stages differ")
    labelled = attach_energy_labels(normalized[["sid", "rk", "e_per_atom"]])
    labelled = labelled.merge(
        features[["sid", "rk", "stage"]],
        on=["sid", "rk"],
        how="inner",
        validate="one_to_one",
    )
    return labelled


def _join_scores_and_labels(
    scores: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    joined = scores.merge(
        labels,
        on=["sid", "rk", "stage"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(scores) or len(joined) != len(labels):
        raise ValueError("score and label rows are not aligned")
    return joined


def _frontier_record(
    search_features: pd.DataFrame,
    search_labels: pd.DataFrame,
    selection_features: pd.DataFrame,
    selection_labels: pd.DataFrame,
    *,
    formula: FormulaSpec,
    track: TrackSpec,
) -> dict[str, object]:
    calibration = _join_scores_and_labels(
        prepare_fewstep_scores(search_features, formula), search_labels
    )
    calibrated = group_conformal_threshold(
        calibration,
        alpha=track.alpha,
        valuable_column=track.protected,
        within_group=track.within_group,
    )
    evaluation = _join_scores_and_labels(
        prepare_fewstep_scores(selection_features, formula), selection_labels
    )
    evaluation["decision"] = apply_group_threshold(
        evaluation["score"].to_numpy(dtype=float),
        evaluation["supported"].to_numpy(dtype=bool),
        float(calibrated["threshold"]),
    )
    metrics = evaluate_group_triage(evaluation)
    record: dict[str, object] = {
        "name": formula.name,
        "track": track.name,
        "energy_source": "min(" + ",".join(formula.energy_source) + ")",
        "max_step": formula.max_step,
        "cost": formula.cost,
        "protected": track.protected,
        "protected_ev_per_atom": track.protected_ev_per_atom,
        "within_group": track.within_group,
        "alpha": track.alpha,
        "search_threshold": float(calibrated["threshold"]),
        "search_calibration_n_groups": int(calibrated["n_groups"]),
        "search_calibration_order_index": int(calibrated["order_index"]),
        **metrics,
    }
    record["passes_exact_min_retention_gate"] = bool(
        float(record["exact_min_retention_lower"]) >= 0.95
    )
    protected_metric = (
        "valuable_group_retention_lower"
        if track.name == "primary"
        else "near_min_retention_lower"
    )
    record["protected_retention_metric"] = protected_metric
    record["passes_protected_retention_gate"] = bool(
        float(record[protected_metric]) >= 0.95
    )
    record["passes_regret_gate"] = bool(float(record["regret_p95"]) <= 0.05)
    record["passes_all_rejected_gate"] = bool(
        int(record["all_rejected_groups"]) == 0
    )
    record["passes_safety_gate"] = passes_safety_gate(record, track)
    return record


def _calibrated_rule_record(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    formula: FormulaSpec,
    track: TrackSpec,
    role: str,
    selection_state: str,
) -> dict[str, object]:
    calibration = _join_scores_and_labels(
        prepare_fewstep_scores(features, formula), labels
    )
    result = group_conformal_threshold(
        calibration,
        alpha=track.alpha,
        valuable_column=track.protected,
        within_group=track.within_group,
    )
    threshold = float(result["threshold"])
    return {
        "track": track.name,
        "role": role,
        "selection_state": selection_state,
        "name": formula.name,
        "protected": track.protected,
        "protected_ev_per_atom": track.protected_ev_per_atom,
        "within_group": track.within_group,
        "alpha": track.alpha,
        "max_step": formula.max_step,
        "cost": formula.cost,
        "threshold": threshold,
        "threshold_state": "finite" if np.isfinite(threshold) else "keep_all",
        "operator": "score > threshold",
        "unsupported_decision": "ABSTAIN",
        "calibration_n_groups": int(result["n_groups"]),
        "calibration_order_index": int(result["order_index"]),
    }


def _null_rule_record(
    labels: pd.DataFrame, *, track: TrackSpec
) -> dict[str, object]:
    return {
        "track": track.name,
        "role": "selected",
        "selection_state": "null_keep_all",
        "name": "null_keep_all",
        "protected": track.protected,
        "protected_ev_per_atom": track.protected_ev_per_atom,
        "within_group": track.within_group,
        "alpha": track.alpha,
        "max_step": 0,
        "cost": 0,
        "threshold": float("inf"),
        "threshold_state": "keep_all",
        "operator": "KEEP_ALL",
        "unsupported_decision": "KEEP",
        "calibration_n_groups": int(labels["rk"].nunique()),
        "calibration_order_index": 0,
    }


def _json_rule(record: Mapping[str, object]) -> dict[str, object]:
    threshold = float(record["threshold"])
    return {
        "state": str(record["selection_state"]),
        "name": str(record["name"]),
        "protected": str(record["protected"]),
        "protected_ev_per_atom": float(record["protected_ev_per_atom"]),
        "within_group": str(record["within_group"]),
        "alpha": float(record["alpha"]),
        "max_step": int(record["max_step"]),
        "cost": int(record["cost"]),
        "threshold": threshold if np.isfinite(threshold) else None,
        "threshold_state": str(record["threshold_state"]),
        "operator": str(record["operator"]),
        "unsupported_decision": str(record["unsupported_decision"]),
        "calibration_n_groups": int(record["calibration_n_groups"]),
        "calibration_order_index": int(record["calibration_order_index"]),
    }


def _input_hashes_by_name(paths: Sequence[Path]) -> dict[str, str]:
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise ValueError("input artifact basenames must be unique")
    return {path.name: _sha256_file(path) for path in paths}


def _validate_staged_artifacts(
    staging_dir: Path,
    *,
    frozen: Mapping[str, object],
    manifest: Mapping[str, object],
) -> None:
    actual_names = {path.name for path in staging_dir.iterdir()}
    if actual_names != set(OUTPUT_NAMES):
        raise RuntimeError("staging directory does not contain the four artifacts")
    pd.read_parquet(staging_dir / "development_frontier.parquet")
    pd.read_parquet(staging_dir / "threshold_calibration_rules.parquet")
    loaded_frozen = _read_json_document(
        staging_dir / "FROZEN_PROTOCOL.json", role="staged frozen protocol"
    )
    loaded_manifest = _read_json_document(
        staging_dir / "MANIFEST.json", role="staged manifest"
    )
    if loaded_frozen != dict(frozen):
        raise RuntimeError("staged frozen protocol content mismatch")
    if loaded_manifest != dict(manifest):
        raise RuntimeError("staged manifest content mismatch")
    output_hashes = manifest.get("outputs_sha256")
    development_hashes = frozen.get("development_artifacts_sha256")
    if not isinstance(output_hashes, dict) or not isinstance(
        development_hashes, dict
    ):
        raise RuntimeError("staged artifact hash declarations are missing")
    for name, expected in output_hashes.items():
        if _sha256_file(staging_dir / str(name)) != expected:
            raise RuntimeError(f"staged output hash mismatch: {name}")
    for name, expected in development_hashes.items():
        if _sha256_file(staging_dir / str(name)) != expected:
            raise RuntimeError(f"staged development hash mismatch: {name}")


def _atomic_publish_directory_no_replace(source: Path, target: Path) -> None:
    """Atomically rename a directory without ever replacing ``target``."""

    unsupported_errno = getattr(errno, "ENOTSUP", errno.EINVAL)
    if sys.platform != "linux":
        raise OSError(
            unsupported_errno,
            "atomic no-replace directory publication is unsupported",
            str(target),
        )
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as exc:
        raise OSError(
            unsupported_errno,
            "atomic no-replace directory publication is unsupported",
            str(target),
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    ctypes.set_errno(0)
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(target),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno() or errno.EIO
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            f"refusing to overwrite existing output directory: {target}",
            str(target),
        )
    unsupported_errors = {
        errno.ENOSYS,
        errno.EINVAL,
        getattr(errno, "ENOTSUP", -1),
        getattr(errno, "EOPNOTSUPP", -1),
    }
    if error_number in unsupported_errors:
        raise OSError(
            error_number,
            "atomic no-replace directory publication is unsupported",
            str(target),
        )
    raise OSError(error_number, os.strerror(error_number), str(target))


def run_development_freeze(
    features_path: Path,
    labels_path: Path,
    feature_manifest_path: Path,
    output_dir: Path,
    *,
    checkpoint: Path,
) -> dict[str, object]:
    """Select and freeze rules without reading or emitting test evidence."""

    features_path = Path(features_path)
    labels_path = Path(labels_path)
    feature_manifest_path = Path(feature_manifest_path)
    output_dir = Path(output_dir)
    checkpoint = Path(checkpoint)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {output_dir}"
        )
    for path in (features_path, labels_path, feature_manifest_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    checkpoint_sha256 = _sha256_file(checkpoint)
    feature_inputs_sha256 = _validate_feature_manifest(
        feature_manifest_path,
        features_path=features_path,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
    )
    features = _validate_feature_stages(
        pd.read_parquet(features_path), require_all=True
    )
    if "evidence_role" not in features.columns or set(
        features["evidence_role"].astype(str)
    ) != {EVIDENCE_ROLE}:
        raise ValueError("feature evidence role mismatch")
    labels = _prepare_labels(pd.read_parquet(labels_path), features)

    feature_stages = {
        stage: features.loc[features["stage"].eq(stage)].copy()
        for stage in DEVELOPMENT_STAGES
    }
    label_stages = {
        stage: labels.loc[labels["stage"].eq(stage)].copy()
        for stage in DEVELOPMENT_STAGES
    }
    if any(table.empty for table in feature_stages.values()):
        raise ValueError("each development stage must contain at least one feature row")

    frontier_records = [
        _frontier_record(
            feature_stages["search_calibration"],
            label_stages["search_calibration"],
            feature_stages["formula_selection"],
            label_stages["formula_selection"],
            formula=formula,
            track=track,
        )
        for formula in FROZEN_CATALOG
        for track in TRACKS.values()
    ]
    frontier = pd.DataFrame(frontier_records)
    selections = {
        name: select_frozen_rule(frontier, track)
        for name, track in TRACKS.items()
    }

    rule_records: list[dict[str, object]] = []
    selected_rule_records: dict[str, dict[str, object]] = {}
    baseline_rule_records: dict[str, dict[str, object]] = {}
    for track_name, track in TRACKS.items():
        selection = selections[track_name]
        if selection["state"] == "null_keep_all":
            selected_rule = _null_rule_record(
                label_stages["threshold_calibration"], track=track
            )
        else:
            selected_rule = _calibrated_rule_record(
                feature_stages["threshold_calibration"],
                label_stages["threshold_calibration"],
                formula=CATALOG_BY_NAME[str(selection["name"])],
                track=track,
                role="selected",
                selection_state="selected",
            )
        baseline_rule = _calibrated_rule_record(
            feature_stages["threshold_calibration"],
            label_stages["threshold_calibration"],
            formula=CATALOG_BY_NAME["S0"],
            track=track,
            role="s0_baseline",
            selection_state="baseline",
        )
        selected_rule_records[track_name] = selected_rule
        baseline_rule_records[track_name] = baseline_rule
        rule_records.extend((selected_rule, baseline_rule))
    threshold_rules = pd.DataFrame(rule_records)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
        )
    )
    try:
        frontier_path = staging_dir / "development_frontier.parquet"
        rules_path = staging_dir / "threshold_calibration_rules.parquet"
        frozen_path = staging_dir / "FROZEN_PROTOCOL.json"
        manifest_path = staging_dir / "MANIFEST.json"
        frontier.to_parquet(frontier_path, index=False)
        threshold_rules.to_parquet(rules_path, index=False)

        source_dir = Path(__file__).resolve().parent
        code_sha256 = {
            "next7_mattersim_prerelax.py": _sha256_file(
                source_dir / "next7_mattersim_prerelax.py"
            ),
            "next7_mattersim_features.py": _sha256_file(
                source_dir / "next7_mattersim_features.py"
            ),
        }
        development_hashes = {
            frontier_path.name: _sha256_file(frontier_path),
            rules_path.name: _sha256_file(rules_path),
        }
        frozen: dict[str, object] = {
            "protocol": DEVELOPMENT_FREEZE_PROTOCOL,
            "state": "frozen",
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": checkpoint_sha256,
            "feature_inputs_sha256": feature_inputs_sha256,
            "code_sha256": code_sha256,
            "selection_code_sha256": _sha256_file(Path(__file__).resolve()),
            "catalog": [formula.as_record() for formula in FROZEN_CATALOG],
            "tracks": {
                name: track.as_record() for name, track in TRACKS.items()
            },
            "rules": {
                name: {
                    **_json_rule(selected_rule_records[name]),
                    "s0_baseline": _json_rule(baseline_rule_records[name]),
                }
                for name in TRACKS
            },
            "development_artifacts_sha256": development_hashes,
            "evidence_role": EVIDENCE_ROLE,
        }
        frozen_path.write_text(
            json.dumps(frozen, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        manifest: dict[str, object] = {
            "protocol": DEVELOPMENT_FREEZE_PROTOCOL,
            "state": "frozen",
            "evidence_role": EVIDENCE_ROLE,
            "inputs_sha256": _input_hashes_by_name(
                (features_path, labels_path, feature_manifest_path, checkpoint)
            ),
            "outputs_sha256": {
                path.name: _sha256_file(path)
                for path in (frontier_path, rules_path, frozen_path)
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _validate_staged_artifacts(
            staging_dir, frozen=frozen, manifest=manifest
        )
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite existing output directory: {output_dir}"
            )
        _atomic_publish_directory_no_replace(staging_dir, output_dir)
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = run_development_freeze(
        args.features,
        args.labels,
        args.feature_manifest,
        args.output,
        checkpoint=args.checkpoint,
    )
    print(json.dumps(manifest["outputs_sha256"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CATALOG",
    "DEVELOPMENT_FREEZE_PROTOCOL",
    "DEVELOPMENT_STAGES",
    "EVIDENCE_ROLE",
    "FROZEN_CATALOG",
    "FROZEN_TRACKS",
    "FormulaSpec",
    "TRACKS",
    "TrackSpec",
    "candidate_catalog",
    "main",
    "passes_safety_gate",
    "prepare_fewstep_scores",
    "run_development_freeze",
    "select_frozen_rule",
]
