"""Two-term DFT-free ranking rule for large OMC25 relaxation-energy response."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


PROTOCOL = "2026-08-03-next31-omc25-energy-ranking-v1"
Q05 = "periodic_nonbond_vdw_q05"
COORD105 = "periodic_contact_coord105"
PREDICTIONS_NAME = "next31_predictions.parquet"
LABEL_FREE_FEATURES_NAME = "next31_label_free_features.parquet"
FROZEN_RULE_NAME = "NEXT31_FROZEN_ENERGY_RULE.json"
MANIFEST_NAME = "MANIFEST.json"
FORBIDDEN_FEATURE_TOKENS = (
    "dft",
    "energy",
    "force",
    "stress",
    "endpoint",
    "label",
    "target",
    "relax",
    "mattersim",
    "mlip",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _rule_number(rule: Mapping[str, object], key: str, *, positive: bool = False) -> float:
    value = rule.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"frozen rule has invalid {key}")
    result = float(value)
    if positive and result <= 0.0:
        raise ValueError(f"frozen rule has non-positive {key}")
    return result


def compute_energy_risk(
    features: pd.DataFrame, rule: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the frozen two-term formula without reading an endpoint."""

    if rule.get("eligible") is not True:
        raise ValueError("frozen NEXT31 rule is not eligible")
    missing = {Q05, COORD105} - set(features.columns)
    if missing:
        raise ValueError(f"features lack columns: {sorted(missing)}")
    q05 = pd.to_numeric(features[Q05], errors="coerce").to_numpy(float)
    coord = pd.to_numeric(features[COORD105], errors="coerce").to_numpy(float)
    q_median = _rule_number(rule, "q05_median")
    q_iqr = _rule_number(rule, "q05_iqr", positive=True)
    c_median = _rule_number(rule, "coord105_median")
    c_iqr = _rule_number(rule, "coord105_iqr", positive=True)
    threshold = _rule_number(rule, "threshold")
    supported = np.isfinite(q05) & np.isfinite(coord)
    if "analytic_supported" in features:
        supported &= features["analytic_supported"].fillna(False).to_numpy(bool)
    score = np.full(len(features), np.nan, dtype=float)
    score[supported] = -(
        q05[supported] - q_median
    ) / q_iqr + (coord[supported] - c_median) / c_iqr
    reject = supported & (score >= threshold)
    return score, supported, reject


def fit_frozen_rule(
    *,
    features: pd.DataFrame,
    endpoints: pd.DataFrame,
    development_shards: Sequence[str],
    rejection_fraction: float = 0.025,
) -> dict[str, object]:
    """Fit robust constants on named exposed development shards only."""

    required_features = {"material_id", "source_shard", Q05, COORD105}
    required_endpoints = {"material_id", "source_shard", "energy_drop_pa"}
    if not required_features.issubset(features.columns) or not required_endpoints.issubset(
        endpoints.columns
    ):
        raise ValueError("development inputs lack required columns")
    shards = tuple(str(value) for value in development_shards)
    if not shards or len(set(shards)) != len(shards):
        raise ValueError("development shards must be unique and non-empty")
    if not 0.0 < rejection_fraction < 1.0:
        raise ValueError("rejection_fraction must lie in (0, 1)")
    feature_dev = features.loc[features["source_shard"].astype(str).isin(shards)].copy()
    endpoint_dev = endpoints.loc[
        endpoints["source_shard"].astype(str).isin(shards)
    ].copy()
    for frame in (feature_dev, endpoint_dev):
        frame["material_id"] = frame["material_id"].astype(str)
        frame["source_shard"] = frame["source_shard"].astype(str)
        if frame["material_id"].duplicated().any():
            raise ValueError("development material IDs must be unique")
    joined = feature_dev.merge(
        endpoint_dev.loc[:, ["material_id", "source_shard", "energy_drop_pa"]],
        on=["material_id", "source_shard"],
        validate="one_to_one",
    )
    if len(joined) != len(feature_dev) or set(joined["source_shard"]) != set(shards):
        raise ValueError("development endpoint coverage or shards differ")
    q05 = pd.to_numeric(joined[Q05], errors="coerce").to_numpy(float)
    coord = pd.to_numeric(joined[COORD105], errors="coerce").to_numpy(float)
    support = np.isfinite(q05) & np.isfinite(coord)
    if not support.any():
        raise ValueError("no supported development rows")
    q25, q_median, q75 = np.quantile(q05[support], [0.25, 0.5, 0.75])
    c25, c_median, c75 = np.quantile(coord[support], [0.25, 0.5, 0.75])
    q_iqr = float(q75 - q25)
    c_iqr = float(c75 - c25)
    if q_iqr <= 0.0 or c_iqr <= 0.0:
        raise ValueError("development term IQR is non-positive")
    score = np.full(len(joined), np.nan, dtype=float)
    score[support] = -(
        q05[support] - q_median
    ) / q_iqr + (coord[support] - c_median) / c_iqr
    threshold = float(
        np.quantile(score[support], 1.0 - rejection_fraction, method="higher")
    )
    reject = support & (score >= threshold)
    energy = pd.to_numeric(joined["energy_drop_pa"], errors="coerce").to_numpy(float)
    if not np.isfinite(energy).all():
        raise ValueError("development energy endpoint must be finite")
    positive = energy >= 0.04
    protected = energy <= 0.01
    return {
        "protocol": PROTOCOL,
        "eligible": True,
        "evidence_role": "historically_exposed_shard_level_development",
        "formula": "-(q05-q05_median)/q05_iqr + (coord105-coord105_median)/coord105_iqr",
        "q05_median": float(q_median),
        "q05_iqr": q_iqr,
        "coord105_median": float(c_median),
        "coord105_iqr": c_iqr,
        "threshold": threshold,
        "rejection_fraction": float(rejection_fraction),
        "development_shards": list(shards),
        "development_rows": len(joined),
        "confirmation_rows_used": 0,
        "confirmation_labels_used_for_selection": False,
        "dft_or_energy_proxy_used_at_execution": False,
        "development_counts": {
            "supported": int(support.sum()),
            "rejected": int(reject.sum()),
            "energy_positive": int(positive.sum()),
            "protected": int(protected.sum()),
            "rejected_positive": int((reject & positive).sum()),
            "rejected_protected": int((reject & protected).sum()),
        },
    }


def freeze_development_rule(
    *,
    feature_paths: Sequence[Path],
    endpoints_path: Path,
    development_shards: Sequence[str],
    output_dir: Path,
    rejection_fraction: float = 0.025,
) -> dict[str, object]:
    """Reproduce development constants and publish an immutable rule artifact."""

    paths = [Path(path).resolve() for path in feature_paths]
    endpoint_path = Path(endpoints_path).resolve()
    target = Path(output_dir).resolve()
    shards = tuple(str(value) for value in development_shards)
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if not paths or any(not path.is_file() for path in paths) or not endpoint_path.is_file():
        raise FileNotFoundError("development feature or endpoint input is missing")
    parts: list[pd.DataFrame] = []
    for index, path in enumerate(paths):
        frame = pd.read_parquet(path)
        forbidden = [
            str(column)
            for column in frame.columns
            if any(token in str(column).lower() for token in FORBIDDEN_FEATURE_TOKENS)
        ]
        if forbidden:
            raise ValueError(f"development feature crossed no-DFT contract: {forbidden}")
        if "source_shard" not in frame:
            if len(paths) != len(shards):
                raise ValueError("feature paths without shard columns must align to shards")
            frame["source_shard"] = shards[index]
        parts.append(frame)
    features = pd.concat(parts, ignore_index=True)
    endpoints = pd.read_parquet(
        endpoint_path,
        columns=["material_id", "source_shard", "energy_drop_pa"],
    )
    rule = fit_frozen_rule(
        features=features,
        endpoints=endpoints,
        development_shards=shards,
        rejection_fraction=rejection_fraction,
    )
    input_hashes = {
        "features": [{"path": str(path), "sha256": _sha256(path)} for path in paths],
        "endpoints": {"path": str(endpoint_path), "sha256": _sha256(endpoint_path)},
    }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "exposed_shard_development_rule_freeze",
        "labels_opened_for_development": True,
        "confirmation_labels_used_for_selection": False,
        "development_shards": list(shards),
        "inputs_sha256": input_hashes,
        "executed_source_sha256": {
            "src/next31_omc25_energy_ranking.py": _sha256(Path(__file__).resolve())
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        rule_path = staging / FROZEN_RULE_NAME
        rule_path.write_bytes(_json_bytes(rule))
        manifest["outputs_sha256"] = {FROZEN_RULE_NAME: _sha256(rule_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(endpoint_path) != input_hashes["endpoints"]["sha256"]:  # type: ignore[index]
            raise RuntimeError("development endpoint changed before rule publication")
        os.rename(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def assemble_label_free_features(
    *,
    feature_paths: Sequence[Path],
    feature_manifest_paths: Sequence[Path],
    source_shards: Sequence[str],
    output_dir: Path,
) -> dict[str, object]:
    """Validate label-free feature provenance and attach exact shard names."""

    paths = [Path(path).resolve() for path in feature_paths]
    manifest_paths = [Path(path).resolve() for path in feature_manifest_paths]
    shards = tuple(str(shard) for shard in source_shards)
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if not paths or not (len(paths) == len(manifest_paths) == len(shards)):
        raise ValueError("feature, manifest, and shard inputs must align")
    if len(set(shards)) != len(shards):
        raise ValueError("source shards must be unique")
    if any(not path.is_file() for path in (*paths, *manifest_paths)):
        raise FileNotFoundError("feature or manifest input is missing")

    parts: list[pd.DataFrame] = []
    inputs: list[dict[str, object]] = []
    for feature_path, manifest_path, shard in zip(
        paths, manifest_paths, shards, strict=True
    ):
        try:
            upstream = json.loads(manifest_path.read_text("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid upstream feature manifest") from exc
        outputs = upstream.get("outputs_sha256")
        if (
            upstream.get("labels_opened") is not False
            or upstream.get("endpoint_fields_read") is not False
            or upstream.get("relaxed_structures_opened") is not False
            or upstream.get("model_or_proxy_potential_used") is not False
            or not isinstance(outputs, Mapping)
            or outputs.get(feature_path.name) != _sha256(feature_path)
        ):
            raise ValueError("upstream feature crossed label-free boundary")
        frame = pd.read_parquet(feature_path)
        forbidden = [
            str(column)
            for column in frame.columns
            if any(token in str(column).lower() for token in FORBIDDEN_FEATURE_TOKENS)
        ]
        if forbidden:
            raise ValueError(f"feature input crossed no-DFT contract: {forbidden}")
        required = {"material_id", Q05, COORD105, "analytic_supported"}
        if not required.issubset(frame.columns):
            raise ValueError("feature input lacks required columns")
        if frame["material_id"].isna().any() or frame["material_id"].duplicated().any():
            raise ValueError("feature material IDs must be unique")
        if "source_shard" in frame and not frame["source_shard"].astype(str).eq(shard).all():
            raise ValueError("existing source shard disagrees with declared shard")
        frame = frame.copy()
        frame["material_id"] = frame["material_id"].astype(str)
        frame["source_shard"] = shard
        parts.append(frame)
        inputs.append(
            {
                "source_shard": shard,
                "features": {"path": str(feature_path), "sha256": _sha256(feature_path)},
                "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
            }
        )

    combined = pd.concat(parts, ignore_index=True).sort_values(
        ["source_shard", "material_id"], kind="stable", ignore_index=True
    )
    if combined["material_id"].duplicated().any():
        raise ValueError("combined feature material IDs must be unique")
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "prospective_label_free_feature_assembly",
        "labels_opened": False,
        "endpoint_fields_read": False,
        "relaxed_structures_opened": False,
        "dft_or_energy_proxy_used_at_execution": False,
        "model_or_proxy_potential_used": False,
        "source_shards": list(shards),
        "counts": {"rows": len(combined), "shards": len(shards)},
        "inputs_sha256": inputs,
        "executed_source_sha256": {
            "src/next31_omc25_energy_ranking.py": _sha256(Path(__file__).resolve())
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        output_path = staging / LABEL_FREE_FEATURES_NAME
        combined.to_parquet(output_path, index=False)
        manifest["outputs_sha256"] = {LABEL_FREE_FEATURES_NAME: _sha256(output_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        for item in inputs:
            for role in ("features", "manifest"):
                value = item[role]
                assert isinstance(value, Mapping)
                if _sha256(Path(str(value["path"]))) != value["sha256"]:
                    raise RuntimeError("label-free feature input changed before publication")
        os.rename(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def apply_frozen_rule(
    *,
    frozen_rule_path: Path,
    feature_paths: Sequence[Path],
    output_dir: Path,
) -> dict[str, object]:
    """Seal label-free predictions for one or more new OMC25 shards."""

    rule_path = Path(frozen_rule_path).resolve()
    paths = [Path(path).resolve() for path in feature_paths]
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if not rule_path.is_file() or not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("rule or feature input is missing")
    rule = json.loads(rule_path.read_text("utf-8"))
    parts: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path)
        forbidden = [
            str(column)
            for column in frame.columns
            if any(token in str(column).lower() for token in FORBIDDEN_FEATURE_TOKENS)
        ]
        if forbidden:
            raise ValueError(f"feature input crossed no-DFT contract: {forbidden}")
        required = {"material_id", Q05, COORD105}
        if not required.issubset(frame.columns):
            raise ValueError("feature input lacks required columns")
        if "source_shard" not in frame:
            frame["source_shard"] = path.parent.name
        score, supported, reject = compute_energy_risk(frame, rule)
        parts.append(
            pd.DataFrame(
                {
                    "material_id": frame["material_id"].astype(str),
                    "source_shard": frame["source_shard"].astype(str),
                    "analytic_supported": supported,
                    "next31_risk_score": score,
                    "reject": reject,
                    "input_role": "unrelaxed_x0_geometry_only",
                }
            )
        )
    predictions = pd.concat(parts, ignore_index=True).sort_values(
        "material_id", kind="stable", ignore_index=True
    )
    if predictions["material_id"].duplicated().any():
        raise ValueError("prediction material IDs must be unique")
    input_hashes = {
        "frozen_rule": {"path": str(rule_path), "sha256": _sha256(rule_path)},
        "features": [
            {"path": str(path), "sha256": _sha256(path)} for path in paths
        ],
    }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "label_free_prospective_energy_response_ranking",
        "labels_opened": False,
        "endpoint_artifacts_opened": False,
        "dft_or_energy_proxy_used_at_execution": False,
        "counts": {
            "rows": len(predictions),
            "supported": int(predictions["analytic_supported"].sum()),
            "rejected": int(predictions["reject"].sum()),
        },
        "inputs_sha256": input_hashes,
        "executed_source_sha256": {
            "src/next31_omc25_energy_ranking.py": _sha256(Path(__file__).resolve())
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        predictions_path = staging / PREDICTIONS_NAME
        predictions.to_parquet(predictions_path, index=False)
        manifest["outputs_sha256"] = {
            PREDICTIONS_NAME: _sha256(predictions_path)
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(rule_path) != input_hashes["frozen_rule"]["sha256"]:  # type: ignore[index]
            raise RuntimeError("frozen NEXT31 rule changed before publication")
        os.rename(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest
