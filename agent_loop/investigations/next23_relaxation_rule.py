#!/usr/bin/env python3
"""Freeze a finite analytic law on exposed relaxation-change development data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.next19_feature_build import _publish_directory_no_replace, _sha256


PROTOCOL = "2026-08-02-next23-relaxation-change-rule-freeze-v1"
ENDPOINT_COLUMN = "site_stats_fingerprint_init_final_norm_diff"
PROTECTED_MAX = 0.10
SUBSTANTIAL_MIN = 0.20
SEVERE_MIN = 0.50
ONE_SIDED_CONFIDENCE = 0.95
PRIMARY_GATES: Mapping[str, float] = {
    "coverage_lower": 0.90,
    "protected_recall_lower": 0.95,
    "rejection_precision_lower": 0.90,
    "savings_lower": 0.10,
}
FROZEN_RULE_NAME = "NEXT23_FROZEN_RELAXATION_RULE.json"
SCAN_NAME = "NEXT23_DEVELOPMENT_SCAN.json"
MANIFEST_NAME = "MANIFEST.json"
FORBIDDEN_FEATURE_COLUMN_TOKENS = (
    "energy",
    "force",
    "stress",
    "relax",
    "mattersim",
    "dft",
    "endpoint",
    "label",
    "target",
)


@dataclass(frozen=True)
class BaseTerm:
    source: str
    column: str
    direction: int


BASE_TERMS: Mapping[str, BaseTerm] = {
    "A": BaseTerm("sivr", "voronoi_q05__sivr_cell_anisotropy", 1),
    "B": BaseTerm("sivr", "voronoi_q0__sivr_cell_anisotropy", 1),
    "C": BaseTerm("sivr", "voronoi_q05__sivr_site_imbalance_max", 1),
    "D": BaseTerm("sivr", "voronoi_q05__sivr_site_imbalance_rms", 1),
    "E": BaseTerm("scbve", "scbv_vector_asymmetry_rms", 1),
    "F": BaseTerm("scbve", "scbv_vector_asymmetry_max", 1),
    "G": BaseTerm("sivr", "voronoi_q05__sivr_stiffness_min", -1),
    "H": BaseTerm("madelung", "nm_point_reduced", -1),
}
CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("A",),
    ("B",),
    ("C",),
    ("D",),
    ("E",),
    ("F",),
    ("G",),
    ("H",),
    ("A", "C"),
    ("A", "E"),
    ("A", "G"),
    ("A", "H"),
    ("B", "C"),
    ("B", "E"),
    ("A", "C", "E"),
    ("A", "E", "G"),
    ("A", "C", "H"),
)


def candidate_name(terms: Sequence[str]) -> str:
    return "+".join(terms)


def wilson_lower_bound(
    successes: int, trials: int, confidence: float = ONE_SIDED_CONFIDENCE
) -> float:
    """Return a one-sided Wilson lower confidence bound for a proportion."""

    if type(successes) is not int or type(trials) is not int:
        raise TypeError("Wilson counts must be exact integers")
    if trials <= 0 or successes < 0 or successes > trials:
        return 0.0
    z = NormalDist().inv_cdf(confidence)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = proportion + z * z / (2.0 * trials)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z * z / (4.0 * trials * trials)
    )
    return float((center - radius) / denominator)


def fit_robust_parameters(features: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Fit only deterministic development medians and interquartile ranges."""

    parameters: dict[str, dict[str, object]] = {}
    for key, term in BASE_TERMS.items():
        if term.column not in features:
            median = None
            scale = None
        else:
            values = pd.to_numeric(features[term.column], errors="coerce").to_numpy(
                dtype=float
            )
            finite = values[np.isfinite(values)]
            if finite.size:
                median_value = float(np.median(finite))
                q25, q75 = np.quantile(finite, [0.25, 0.75])
                scale_value = float(q75 - q25)
            else:
                median_value = math.nan
                scale_value = math.nan
            median = median_value if math.isfinite(median_value) else None
            scale = (
                scale_value
                if math.isfinite(scale_value) and scale_value > 0.0
                else None
            )
        parameters[key] = {
            **asdict(term),
            "median": median,
            "scale_iqr": scale,
        }
    return parameters


def score_candidate(
    features: pd.DataFrame,
    terms: Sequence[str],
    parameters: Mapping[str, Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate an equal-weight analytic score; unsupported rows fail open."""

    n_rows = len(features)
    score = np.zeros(n_rows, dtype=float)
    support = np.ones(n_rows, dtype=bool)
    for key in terms:
        if key not in BASE_TERMS or key not in parameters:
            raise ValueError(f"unknown frozen term: {key}")
        parameter = parameters[key]
        column = str(parameter["column"])
        median = parameter.get("median")
        scale = parameter.get("scale_iqr")
        direction = parameter.get("direction")
        if (
            column not in features
            or not isinstance(median, (int, float))
            or not math.isfinite(float(median))
            or not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            or float(scale) <= 0.0
            or direction not in {-1, 1}
        ):
            return np.full(n_rows, np.nan), np.zeros(n_rows, dtype=bool)
        values = pd.to_numeric(features[column], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        support &= finite
        score += int(direction) * (values - float(median)) / float(scale)
    score[~support] = np.nan
    return score, support


def _metrics(
    *, score: np.ndarray, support: np.ndarray, endpoint: np.ndarray, threshold: float
) -> dict[str, object]:
    n_rows = len(endpoint)
    reject = support & (score >= threshold)
    protected = endpoint <= PROTECTED_MAX
    changed = endpoint > PROTECTED_MAX
    substantial = endpoint >= SUBSTANTIAL_MIN
    severe = endpoint >= SEVERE_MIN
    n_supported = int(support.sum())
    n_rejected = int(reject.sum())
    n_protected = int(protected.sum())
    protected_kept = int((protected & ~reject).sum())
    changed_rejected = int((changed & reject).sum())
    substantial_rejected = int((substantial & reject).sum())
    severe_rejected = int((severe & reject).sum())
    metrics: dict[str, object] = {
        "threshold": float(threshold),
        "rows": n_rows,
        "supported": n_supported,
        "rejected": n_rejected,
        "protected": n_protected,
        "protected_kept": protected_kept,
        "changed_rejected": changed_rejected,
        "coverage": n_supported / n_rows,
        "coverage_lower": wilson_lower_bound(n_supported, n_rows),
        "protected_recall": protected_kept / n_protected if n_protected else 0.0,
        "protected_recall_lower": wilson_lower_bound(protected_kept, n_protected),
        "rejection_precision": changed_rejected / n_rejected if n_rejected else 0.0,
        "rejection_precision_lower": wilson_lower_bound(changed_rejected, n_rejected),
        "savings": n_rejected / n_rows,
        "savings_lower": wilson_lower_bound(n_rejected, n_rows),
        "substantial_recall": (
            substantial_rejected / int(substantial.sum())
            if substantial.any()
            else 0.0
        ),
        "severe_recall": severe_rejected / int(severe.sum()) if severe.any() else 0.0,
    }
    metrics["passes_primary_gates"] = all(
        float(metrics[name]) >= cutoff for name, cutoff in PRIMARY_GATES.items()
    )
    return metrics


def _validated_labels(
    labels: pd.DataFrame, expected_ids: Sequence[str]
) -> pd.DataFrame:
    if "material_id" not in labels or ENDPOINT_COLUMN not in labels:
        raise ValueError("development labels lack the declared endpoint")
    subset = labels.loc[
        labels["material_id"].astype(str).isin(expected_ids),
        ["material_id", ENDPOINT_COLUMN],
    ].copy()
    subset["material_id"] = subset["material_id"].astype(str)
    if subset["material_id"].duplicated().any() or set(subset["material_id"]) != set(
        expected_ids
    ):
        raise ValueError("development label IDs do not join one-to-one")
    subset[ENDPOINT_COLUMN] = pd.to_numeric(subset[ENDPOINT_COLUMN], errors="coerce")
    if not np.isfinite(subset[ENDPOINT_COLUMN].to_numpy(dtype=float)).all():
        raise ValueError("development endpoint must be finite")
    return subset


def select_candidate(features: pd.DataFrame, labels: pd.DataFrame) -> dict[str, object]:
    """Scan the frozen catalogue and deterministically select at most one law."""

    if "material_id" not in features or features["material_id"].isna().any():
        raise ValueError("development features lack material IDs")
    feature_ids = features["material_id"].astype(str)
    if feature_ids.duplicated().any():
        raise ValueError("development feature IDs must be unique")
    labels_valid = _validated_labels(labels, feature_ids.tolist())
    ordered = pd.DataFrame({"material_id": feature_ids}).merge(
        labels_valid, on="material_id", how="left", validate="one_to_one"
    )
    endpoint = ordered[ENDPOINT_COLUMN].to_numpy(dtype=float)
    parameters = fit_robust_parameters(features)

    scans: list[dict[str, object]] = []
    selected: dict[str, object] | None = None
    for catalogue_index, terms in enumerate(CANDIDATES):
        name = candidate_name(terms)
        score, support = score_candidate(features, terms, parameters)
        candidate_best: dict[str, object] | None = None
        if support.any():
            thresholds = np.unique(score[support])
            for threshold in thresholds:
                metrics = _metrics(
                    score=score,
                    support=support,
                    endpoint=endpoint,
                    threshold=float(threshold),
                )
                if not metrics["passes_primary_gates"]:
                    continue
                if candidate_best is None or (
                    int(metrics["rejected"]) > int(candidate_best["rejected"])
                    or (
                        int(metrics["rejected"]) == int(candidate_best["rejected"])
                        and float(metrics["threshold"])
                        < float(candidate_best["threshold"])
                    )
                ):
                    candidate_best = metrics
        record: dict[str, object] = {
            "catalogue_index": catalogue_index,
            "candidate": name,
            "terms": list(terms),
            "supported_rows": int(support.sum()),
            "eligible": candidate_best is not None,
            "best_metrics": candidate_best,
        }
        scans.append(record)
        if candidate_best is not None and (
            selected is None
            or int(candidate_best["rejected"])
            > int(selected["selected_metrics"]["rejected"])
        ):
            selected = {
                "catalogue_index": catalogue_index,
                "selected_candidate": name,
                "selected_terms": list(terms),
                "selected_metrics": candidate_best,
            }
    return {
        "protocol": PROTOCOL,
        "endpoint": {
            "column": ENDPOINT_COLUMN,
            "protected_max": PROTECTED_MAX,
            "substantial_min": SUBSTANTIAL_MIN,
            "severe_min": SEVERE_MIN,
            "role": "offline_development_label_only",
        },
        "primary_gates": dict(PRIMARY_GATES),
        "confidence": {
            "method": "one-sided Wilson lower bound",
            "level": ONE_SIDED_CONFIDENCE,
        },
        "base_parameters": parameters,
        "catalogue": [candidate_name(terms) for terms in CANDIDATES],
        "candidate_scans": scans,
        "eligible": selected is not None,
        "selected_candidate": (
            selected["selected_candidate"] if selected is not None else None
        ),
        "selected_terms": selected["selected_terms"] if selected is not None else None,
        "selected_metrics": (
            selected["selected_metrics"] if selected is not None else None
        ),
    }


def _validate_feature_table(
    frame: pd.DataFrame, *, role: str, expected_ids: Sequence[str]
) -> None:
    forbidden = [
        str(column)
        for column in frame.columns
        if any(
            token in str(column).lower()
            for token in FORBIDDEN_FEATURE_COLUMN_TOKENS
        )
    ]
    if forbidden:
        raise ValueError(f"{role} feature table crossed no-DFT contract: {forbidden}")
    if "material_id" not in frame or frame["material_id"].isna().any():
        raise ValueError(f"{role} features lack material IDs")
    ids = frame["material_id"].astype(str)
    if ids.duplicated().any() or set(ids) != set(expected_ids):
        raise ValueError(f"{role} feature IDs differ from development metadata")


def _merge_features(
    *,
    sivr: pd.DataFrame,
    madelung: pd.DataFrame,
    scbve: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    if "material_id" not in metadata or metadata["material_id"].isna().any():
        raise ValueError("development metadata lacks material IDs")
    ids = metadata["material_id"].astype(str)
    if ids.duplicated().any():
        raise ValueError("development metadata IDs must be unique")
    for role, frame in (
        ("sivr", sivr),
        ("madelung", madelung),
        ("scbve", scbve),
    ):
        _validate_feature_table(frame, role=role, expected_ids=ids.tolist())
    result = pd.DataFrame({"material_id": ids})
    for role, frame in (
        ("sivr", sivr),
        ("madelung", madelung),
        ("scbve", scbve),
    ):
        columns = [
            term.column for term in BASE_TERMS.values() if term.source == role
        ]
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f"{role} features lack frozen terms: {missing}")
        subset = frame.loc[:, ["material_id", *columns]].copy()
        subset["material_id"] = subset["material_id"].astype(str)
        result = result.merge(subset, on="material_id", how="left", validate="one_to_one")
    return result


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def freeze_development_rule(
    *,
    sivr_features_path: Path,
    madelung_features_path: Path,
    scbve_features_path: Path,
    labels_path: Path,
    development_metadata_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Publish a no-replace rule frozen entirely on the exposed cohort."""

    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(str(target))
    paths = {
        "sivr_features": Path(sivr_features_path).resolve(),
        "madelung_features": Path(madelung_features_path).resolve(),
        "scbve_features": Path(scbve_features_path).resolve(),
        "development_labels": Path(labels_path).resolve(),
        "development_metadata": Path(development_metadata_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    input_hashes = {role: _sha256(path) for role, path in paths.items()}
    metadata = pd.read_parquet(paths["development_metadata"])
    sivr = pd.read_parquet(paths["sivr_features"])
    madelung = pd.read_parquet(paths["madelung_features"])
    scbve = pd.read_parquet(paths["scbve_features"])
    labels = pd.read_parquet(paths["development_labels"])
    features = _merge_features(
        sivr=sivr, madelung=madelung, scbve=scbve, metadata=metadata
    )
    scan = select_candidate(features, labels)

    selected_terms = scan["selected_terms"]
    law = {
        "protocol": PROTOCOL,
        "eligible": scan["eligible"],
        "selected_candidate": scan["selected_candidate"],
        "selected_terms": selected_terms,
        "threshold": (
            scan["selected_metrics"]["threshold"]
            if scan["selected_metrics"] is not None
            else None
        ),
        "base_parameters": {
            key: scan["base_parameters"][key]
            for key in (selected_terms or [])
        },
        "score_definition": (
            "sum(direction * (feature - development_median) / development_IQR)"
        ),
        "missing_policy": "fail_open_do_not_reject",
        "reject_when": "supported and score >= threshold",
        "executable_inputs": "one_unrelaxed_structure_plus_element_tables",
        "dft_or_relaxed_input_used": False,
        "model_or_proxy_potential_used": False,
        "same_composition_candidates_used": False,
        "development_labels_opened": True,
        "blind_labels_opened": False,
        "primary_gates": dict(PRIMARY_GATES),
        "endpoint_operational_cutoffs": {
            "protected_max": PROTECTED_MAX,
            "substantial_min": SUBSTANTIAL_MIN,
            "severe_min": SEVERE_MIN,
        },
        "development_selected_metrics": scan["selected_metrics"],
    }
    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next23_relaxation_rule.py": Path(__file__).resolve(),
        "src/next20_valence_rigidity.py": repository_root
        / "src/next20_valence_rigidity.py",
        "src/next21_normalized_madelung.py": repository_root
        / "src/next21_normalized_madelung.py",
        "src/next22_bond_valence_equilibrium.py": repository_root
        / "src/next22_bond_valence_equilibrium.py",
    }
    source_hashes = {
        relative: _sha256(path) for relative, path in source_paths.items()
    }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "exposed_development_only_finite_analytic_search",
        "development_labels_opened": True,
        "blind_labels_opened": False,
        "relaxed_structures_opened_by_law": False,
        "candidate_count": len(CANDIDATES),
        "selected_candidate": scan["selected_candidate"],
        "inputs_sha256": {
            role: {"path": str(paths[role]), "sha256": digest}
            for role, digest in input_hashes.items()
        },
        "executed_source_sha256": source_hashes,
        "scientific_improvement_claim": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        rule_path = staging / FROZEN_RULE_NAME
        scan_path = staging / SCAN_NAME
        rule_path.write_bytes(_json_bytes(law))
        scan_path.write_bytes(_json_bytes(scan))
        manifest["outputs_sha256"] = {
            FROZEN_RULE_NAME: _sha256(rule_path),
            SCAN_NAME: _sha256(scan_path),
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        for role, path in paths.items():
            if _sha256(path) != input_hashes[role]:
                raise RuntimeError(f"input {role} changed before rule publication")
        for relative, path in source_paths.items():
            if _sha256(path) != source_hashes[relative]:
                raise RuntimeError(f"source {relative} changed before rule publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


__all__ = [
    "BASE_TERMS",
    "CANDIDATES",
    "FROZEN_RULE_NAME",
    "MANIFEST_NAME",
    "SCAN_NAME",
    "fit_robust_parameters",
    "freeze_development_rule",
    "score_candidate",
    "select_candidate",
    "wilson_lower_bound",
]
