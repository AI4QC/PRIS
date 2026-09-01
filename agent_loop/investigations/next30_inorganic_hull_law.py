"""Sparse single-x0 analytic screening laws for WBM hull-energy development.

The executable score consumes only precomputed analytic geometry/valence terms.
DFT hull values are permitted only in the separate offline discovery/evaluation
phases added below; they are never accepted by :func:`_score_formula`.
"""

from __future__ import annotations

from dataclasses import dataclass
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


SALT = "NEXT30-WBM-HULL-v1"
PROTECTED_HULL_MAX = 0.05
HIGH_HULL_MIN = 0.20
ONE_SIDED_95_Z = 1.6448536269514722
PRIMARY_GATES = {
    "coverage_lower": 0.90,
    "valuable_recall_lower": 0.95,
    "reject_precision_high_energy_lower": 0.90,
    "dft_savings_lower": 0.10,
}
FORBIDDEN_FEATURE_COLUMN_TOKENS = (
    "dft",
    "energy",
    "force",
    "stress",
    "hull",
    "endpoint",
    "label",
    "target",
    "relax",
    "mattersim",
    "mlip",
)


@dataclass(frozen=True)
class TermSpec:
    source: str
    column: str
    direction: int

    def __post_init__(self) -> None:
        if self.direction not in {-1, 1}:
            raise ValueError("term direction must be -1 or +1")


TERM_SPECS: Mapping[str, TermSpec] = {
    "sivr_edge_q95": TermSpec(
        "sivr", "voronoi_q0__sivr_edge_mismatch_q95", 1
    ),
    "sivr_edge_max": TermSpec(
        "sivr", "voronoi_q0__sivr_edge_mismatch_max", 1
    ),
    "sivr_site_rms": TermSpec(
        "sivr", "voronoi_q0__sivr_site_imbalance_rms", 1
    ),
    "sivr_cell_anisotropy": TermSpec(
        "sivr", "voronoi_q0__sivr_cell_anisotropy", 1
    ),
    "nm_weak_total": TermSpec("madelung", "nm_total_reduced", 1),
    "nm_site_spread": TermSpec("madelung", "nm_site_spread", 1),
    "nm_site_max": TermSpec("madelung", "nm_site_max", 1),
    "scbv_mismatch_rms": TermSpec("scbve", "scbv_mismatch_rms", 1),
    "scbv_mismatch_q95": TermSpec("scbve", "scbv_mismatch_q95", 1),
    "scbv_vector_rms": TermSpec(
        "scbve", "scbv_vector_asymmetry_rms", 1
    ),
    "scbv_isolated": TermSpec("scbve", "scbv_isolated_site_fraction", 1),
    "scbv_effective_cn_low": TermSpec(
        "scbve", "scbv_effective_cn_mean", -1
    ),
}

FORMULAS: tuple[tuple[str, ...], ...] = (
    *((name,) for name in TERM_SPECS),
    ("sivr_edge_q95", "sivr_cell_anisotropy"),
    ("sivr_edge_max", "sivr_cell_anisotropy"),
    ("sivr_site_rms", "sivr_cell_anisotropy"),
    ("nm_weak_total", "nm_site_spread"),
    ("nm_weak_total", "nm_site_max"),
    ("scbv_mismatch_rms", "scbv_vector_rms"),
    ("scbv_mismatch_q95", "scbv_vector_rms"),
    ("sivr_edge_q95", "scbv_mismatch_rms"),
    ("sivr_edge_max", "scbv_mismatch_q95"),
    ("sivr_cell_anisotropy", "scbv_vector_rms"),
    ("nm_weak_total", "scbv_mismatch_rms"),
    ("sivr_edge_q95", "nm_weak_total"),
    ("sivr_cell_anisotropy", "nm_site_spread"),
    ("sivr_edge_q95", "sivr_cell_anisotropy", "scbv_mismatch_rms"),
    ("sivr_edge_max", "sivr_cell_anisotropy", "scbv_vector_rms"),
    ("sivr_edge_q95", "nm_weak_total", "scbv_mismatch_rms"),
    ("sivr_cell_anisotropy", "nm_site_spread", "scbv_vector_rms"),
)

REJECTION_FRACTIONS = (0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30)
PROTOCOL = "2026-08-03-next30-inorganic-hull-law-v1"
SPLIT_NAME = "next30_split.parquet"
MANIFEST_NAME = "MANIFEST.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    def safe(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(key): safe(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(value) for value in item]
        if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            return None
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.bool_):
            return bool(item)
        return item

    return (
        json.dumps(safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _publish_directory(staging: Path, target: Path) -> None:
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    os.rename(staging, target)


def freeze_split(
    *,
    metadata_path: Path,
    output_dir: Path,
    development_size: int = 4096,
    expected_rows: int = 8192,
    salt: str = SALT,
) -> dict[str, object]:
    """Publish the deterministic partition without opening any label artifact."""

    source = Path(metadata_path).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if not source.is_file():
        raise FileNotFoundError(str(source))
    metadata = pd.read_parquet(source)
    forbidden = [
        str(column)
        for column in metadata.columns
        if any(token in str(column).lower() for token in FORBIDDEN_FEATURE_COLUMN_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"metadata crossed label-free split contract: {forbidden}")
    if len(metadata) != expected_rows or "material_id" not in metadata:
        raise ValueError("metadata row count or identity column differs")
    ids = metadata["material_id"].astype(str)
    if ids.duplicated().any():
        raise ValueError("metadata material IDs must be unique")
    split = deterministic_split(ids.tolist(), development_size=development_size, salt=salt)
    counts = split["partition"].value_counts().to_dict()
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "label_free_deterministic_development_confirmation_split",
        "salt": salt,
        "labels_opened": False,
        "endpoint_artifacts_opened": False,
        "input_role": "unrelaxed_x0_geometry_only",
        "counts": {
            "rows": len(split),
            "development": int(counts.get("development", 0)),
            "confirmation": int(counts.get("confirmation", 0)),
        },
        "inputs_sha256": {
            "metadata": {"path": str(source), "sha256": _sha256_file(source)}
        },
        "candidate_catalogue": {
            "terms": {
                name: {
                    "source": spec.source,
                    "column": spec.column,
                    "direction": spec.direction,
                }
                for name, spec in TERM_SPECS.items()
            },
            "formulas": [list(formula) for formula in FORMULAS],
            "rejection_fractions": list(REJECTION_FRACTIONS),
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        split_path = staging / SPLIT_NAME
        split.to_parquet(split_path, index=False)
        manifest["outputs_sha256"] = {SPLIT_NAME: _sha256_file(split_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256_file(source) != manifest["inputs_sha256"]["metadata"]["sha256"]:  # type: ignore[index]
            raise RuntimeError("metadata changed during split publication")
        _publish_directory(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def run_development(
    *,
    metadata_path: Path,
    split_path: Path,
    split_manifest_path: Path,
    feature_paths: Mapping[str, Path],
    labels_path: Path,
    output_dir: Path,
    label_column: str = "e_above_hull_mp2020_corrected_ppd_mp",
    term_specs: Mapping[str, TermSpec] = TERM_SPECS,
    formulas: Sequence[Sequence[str]] = FORMULAS,
    rejection_fractions: Sequence[float] = REJECTION_FRACTIONS,
) -> dict[str, object]:
    """Use development statistics only and publish a rule only after promotion."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    core_paths = {
        "metadata": Path(metadata_path).resolve(),
        "split": Path(split_path).resolve(),
        "split_manifest": Path(split_manifest_path).resolve(),
        "labels": Path(labels_path).resolve(),
    }
    expected_sources = {spec.source for spec in term_specs.values()}
    if set(feature_paths) != expected_sources:
        raise ValueError(f"feature paths must contain exactly {sorted(expected_sources)}")
    resolved_features = {
        source: Path(path).resolve() for source, path in feature_paths.items()
    }
    input_paths = {
        **core_paths,
        **{f"{source}_features": path for source, path in resolved_features.items()},
    }
    for role, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    input_hashes = {role: _sha256_file(path) for role, path in input_paths.items()}
    split_manifest = json.loads(core_paths["split_manifest"].read_text("utf-8"))
    outputs = split_manifest.get("outputs_sha256")
    if (
        split_manifest.get("protocol") != PROTOCOL
        or split_manifest.get("labels_opened") is not False
        or split_manifest.get("endpoint_artifacts_opened") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get(SPLIT_NAME) != input_hashes["split"]
    ):
        raise ValueError("NEXT30 split manifest is invalid")
    metadata = pd.read_parquet(core_paths["metadata"])
    split = pd.read_parquet(core_paths["split"])
    source_frames = {
        source: pd.read_parquet(path) for source, path in resolved_features.items()
    }
    features = merge_feature_sources(
        metadata=metadata, source_frames=source_frames, term_specs=term_specs
    )
    labels = pd.read_parquet(
        core_paths["labels"], columns=["material_id", label_column]
    )
    expected_ids = set(split["material_id"].astype(str))
    labels["material_id"] = labels["material_id"].astype(str)
    labels = labels.loc[labels["material_id"].isin(expected_ids)].copy()
    result = discover_rule(
        features=features,
        labels=labels,
        split=split,
        label_column=label_column,
        term_specs=term_specs,
        formulas=formulas,
        rejection_fractions=rejection_fractions,
    )
    selected = result["selected"]
    rule: dict[str, object] | None = None
    if isinstance(selected, Mapping):
        formula = [str(name) for name in selected["formula"]]
        rule = {
            "protocol": PROTOCOL,
            "eligible": True,
            "execution_boundary": "single_unrelaxed_x0_plus_frozen_element_tables_only",
            "dft_or_energy_proxy_used_at_execution": False,
            "selected_terms": formula,
            "threshold": float(selected["threshold"]),
            "development_rejection_fraction": float(selected["rejection_fraction"]),
            "term_specs": {
                name: {
                    "source": term_specs[name].source,
                    "column": term_specs[name].column,
                    "direction": term_specs[name].direction,
                }
                for name in formula
            },
            "term_parameters": {
                name: result["parameters"][name] for name in formula  # type: ignore[index]
            },
            "development_metrics": selected["metrics"],
            "confirmation_labels_used_for_selection": False,
            "source_label_artifact_historically_exposed": True,
        }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "historically_exposed_source_development_only_formula_scan",
        "promotion": bool(result["promotion"]),
        "development_rows": int(result["development_rows"]),
        "confirmation_rows_used": int(result["confirmation_rows_used"]),
        "confirmation_labels_used_for_selection": False,
        "source_label_artifact_historically_exposed": True,
        "label_column": label_column,
        "inputs_sha256": {
            role: {"path": str(path), "sha256": input_hashes[role]}
            for role, path in input_paths.items()
        },
        "executed_source_sha256": {
            "src/next30_inorganic_hull_law.py": _sha256_file(Path(__file__).resolve())
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        scan_path = staging / "NEXT30_DEVELOPMENT_SCAN.json"
        scan_path.write_bytes(_json_bytes(result))
        outputs_sha = {scan_path.name: _sha256_file(scan_path)}
        if rule is not None:
            rule_path = staging / "NEXT30_FROZEN_HULL_RULE.json"
            rule_path.write_bytes(_json_bytes(rule))
            outputs_sha[rule_path.name] = _sha256_file(rule_path)
        manifest["outputs_sha256"] = outputs_sha
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if any(_sha256_file(path) != input_hashes[role] for role, path in input_paths.items()):
            raise RuntimeError("NEXT30 development input changed before publication")
        _publish_directory(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def merge_feature_sources(
    *,
    metadata: pd.DataFrame,
    source_frames: Mapping[str, pd.DataFrame],
    term_specs: Mapping[str, TermSpec] = TERM_SPECS,
) -> pd.DataFrame:
    """Join only the exact analytic columns required by the frozen catalogue."""

    expected_sources = {spec.source for spec in term_specs.values()}
    if set(source_frames) != expected_sources:
        raise ValueError(f"feature sources must contain exactly {sorted(expected_sources)}")
    if "material_id" not in metadata or metadata["material_id"].isna().any():
        raise ValueError("metadata lacks material IDs")
    base = metadata.copy()
    base["material_id"] = base["material_id"].astype(str)
    if base["material_id"].duplicated().any():
        raise ValueError("metadata material IDs must be unique")
    expected_ids = base["material_id"].tolist()
    merged = base.sort_values("material_id", kind="stable", ignore_index=True)
    for source in sorted(expected_sources):
        columns = sorted(
            {spec.column for spec in term_specs.values() if spec.source == source}
        )
        raw = source_frames[source]
        missing = {"material_id", *columns} - set(raw.columns)
        if missing:
            raise ValueError(f"{source} lacks columns: {sorted(missing)}")
        selected = _validated_feature_frame(
            raw.loc[:, ["material_id", *columns]],
            expected_ids=expected_ids,
            role=f"{source} analytic features",
        )
        merged = merged.merge(selected, on="material_id", validate="one_to_one")
    return merged.sort_values("material_id", kind="stable", ignore_index=True)


def deterministic_split(
    material_ids: Sequence[str], *, development_size: int, salt: str = SALT
) -> pd.DataFrame:
    """Return a stable hash-ranked development/confirmation partition."""

    ids = [str(value) for value in material_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("material IDs must be unique")
    if not 0 < development_size < len(ids):
        raise ValueError("development_size must leave both partitions non-empty")
    rows = [
        {
            "material_id": material_id,
            "split_key_sha256": hashlib.sha256(
                f"{salt}|{material_id}".encode("utf-8")
            ).hexdigest(),
        }
        for material_id in ids
    ]
    out = pd.DataFrame(rows).sort_values(
        ["split_key_sha256", "material_id"], kind="stable", ignore_index=True
    )
    out["split_rank"] = np.arange(len(out), dtype=np.int64)
    out["partition"] = np.where(
        out["split_rank"].lt(development_size), "development", "confirmation"
    )
    return out.sort_values("material_id", kind="stable", ignore_index=True)


def _validated_feature_frame(
    frame: pd.DataFrame, *, expected_ids: Sequence[str], role: str
) -> pd.DataFrame:
    forbidden = [
        str(column)
        for column in frame.columns
        if any(token in str(column).lower() for token in FORBIDDEN_FEATURE_COLUMN_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"{role} crossed no-DFT contract: {forbidden}")
    if "material_id" not in frame or frame["material_id"].isna().any():
        raise ValueError(f"{role} lacks material IDs")
    out = frame.copy()
    out["material_id"] = out["material_id"].astype(str)
    expected = {str(value) for value in expected_ids}
    if out["material_id"].duplicated().any() or set(out["material_id"]) != expected:
        raise ValueError(f"{role} IDs differ from expected cohort")
    return out.sort_values("material_id", kind="stable", ignore_index=True)


def fit_term_parameters(
    development: pd.DataFrame, term_specs: Mapping[str, TermSpec]
) -> dict[str, dict[str, object]]:
    """Fit robust location/scale using only the caller-supplied rows."""

    parameters: dict[str, dict[str, object]] = {}
    for name, spec in term_specs.items():
        values = pd.to_numeric(development[spec.column], errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise ValueError(f"term {name} has no finite development values")
        q25, median, q75 = np.quantile(finite, [0.25, 0.5, 0.75])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= 0.0:
            parameters[name] = {
                "available": False,
                "median": float(median),
                "scale_iqr": None,
                "reason": "non_positive_development_iqr",
            }
        else:
            parameters[name] = {
                "available": True,
                "median": float(median),
                "scale_iqr": scale,
                "reason": None,
            }
    return parameters


def _score_formula(
    frame: pd.DataFrame,
    *,
    formula: Sequence[str],
    term_specs: Mapping[str, TermSpec],
    parameters: Mapping[str, Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return an equal-weight risk score; unsupported rows remain NaN/fail-open."""

    pieces: list[np.ndarray] = []
    for name in formula:
        spec = term_specs[name]
        parameter = parameters[name]
        if parameter.get("available", True) is not True:
            return np.full(len(frame), np.nan, dtype=float), np.zeros(
                len(frame), dtype=bool
            )
        values = pd.to_numeric(frame[spec.column], errors="coerce").to_numpy(float)
        z = spec.direction * (
            values - float(parameter["median"])
        ) / float(parameter["scale_iqr"])
        pieces.append(z)
    matrix = np.column_stack(pieces)
    supported = np.isfinite(matrix).all(axis=1)
    score = np.full(len(frame), np.nan, dtype=float)
    score[supported] = matrix[supported].sum(axis=1)
    return score, supported


def _wilson_lower(successes: int, total: int, *, z: float = ONE_SIDED_95_Z) -> float:
    if total <= 0:
        return math.nan
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - radius)


def _metric(successes: int, total: int) -> dict[str, float | int]:
    return {
        "numerator": int(successes),
        "denominator": int(total),
        "estimate": float(successes / total) if total else math.nan,
        "wilson_lower_onesided_95": _wilson_lower(successes, total),
    }


def decision_metrics(
    *, supported: np.ndarray, reject: np.ndarray, energy: np.ndarray
) -> dict[str, dict[str, float | int]]:
    """Evaluate a fail-open decision against offline WBM hull labels."""

    supported = np.asarray(supported, dtype=bool)
    reject = np.asarray(reject, dtype=bool)
    energy = np.asarray(energy, dtype=float)
    if not (len(supported) == len(reject) == len(energy)):
        raise ValueError("decision arrays have different lengths")
    if not np.isfinite(energy).all():
        raise ValueError("offline hull endpoint must be finite")
    if np.any(reject & ~supported):
        raise ValueError("unsupported rows must fail open")
    valuable = energy <= PROTECTED_HULL_MAX
    stable = energy <= 0.0
    high = energy >= HIGH_HULL_MIN
    return {
        "coverage": _metric(int(supported.sum()), len(energy)),
        "valuable_recall": _metric(int((valuable & ~reject).sum()), int(valuable.sum())),
        "stable_recall": _metric(int((stable & ~reject).sum()), int(stable.sum())),
        "reject_precision_high_energy": _metric(
            int((reject & high).sum()), int(reject.sum())
        ),
        "high_energy_rejection_recall": _metric(
            int((reject & high).sum()), int(high.sum())
        ),
        "dft_savings": _metric(int(reject.sum()), len(energy)),
    }


def _primary_clauses(
    metrics: Mapping[str, Mapping[str, float | int]],
) -> dict[str, bool]:
    keys = {
        "coverage_lower": "coverage",
        "valuable_recall_lower": "valuable_recall",
        "reject_precision_high_energy_lower": "reject_precision_high_energy",
        "dft_savings_lower": "dft_savings",
    }
    return {
        clause: bool(
            math.isfinite(
                value := float(metrics[metric]["wilson_lower_onesided_95"])
            )
            and value >= PRIMARY_GATES[clause]
        )
        for clause, metric in keys.items()
    }


def scan_development(
    frame: pd.DataFrame,
    *,
    energy: np.ndarray,
    term_specs: Mapping[str, TermSpec],
    formulas: Sequence[Sequence[str]],
    rejection_fractions: Sequence[float],
    parameters: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Scan a finite, predeclared catalogue using development rows only."""

    energy = np.asarray(energy, dtype=float)
    if len(frame) != len(energy):
        raise ValueError("development features and labels differ in length")
    rows: list[dict[str, object]] = []
    eligible: list[dict[str, object]] = []
    for formula_values in formulas:
        formula = tuple(str(value) for value in formula_values)
        if not formula or len(set(formula)) != len(formula):
            raise ValueError("formula terms must be non-empty and unique")
        score, supported = _score_formula(
            frame,
            formula=formula,
            term_specs=term_specs,
            parameters=parameters,
        )
        finite = score[supported]
        for rejection_fraction in rejection_fractions:
            fraction = float(rejection_fraction)
            if not 0.0 < fraction < 1.0:
                raise ValueError("rejection fractions must lie in (0, 1)")
            if len(finite):
                threshold = float(np.quantile(finite, 1.0 - fraction, method="higher"))
                reject = supported & (score >= threshold)
            else:
                threshold = math.nan
                reject = np.zeros(len(frame), dtype=bool)
            metrics = decision_metrics(
                supported=supported, reject=reject, energy=energy
            )
            clauses = _primary_clauses(metrics)
            candidate: dict[str, object] = {
                "formula": list(formula),
                "rejection_fraction": fraction,
                "threshold": threshold,
                "supported_rows": int(supported.sum()),
                "metrics": metrics,
                "clauses": clauses,
                "eligible": bool(all(clauses.values())),
            }
            rows.append(candidate)
            if candidate["eligible"]:
                eligible.append(candidate)
    selected = None
    if eligible:
        selected = sorted(
            eligible,
            key=lambda item: (
                -float(item["metrics"]["dft_savings"]["estimate"]),  # type: ignore[index]
                len(item["formula"]),  # type: ignore[arg-type]
                float(item["rejection_fraction"]),
                tuple(item["formula"]),  # type: ignore[arg-type]
            ),
        )[0]
    return {
        "promotion": selected is not None,
        "selected": selected,
        "candidates": rows,
    }


def discover_rule(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    split: pd.DataFrame,
    label_column: str,
    term_specs: Mapping[str, TermSpec],
    formulas: Sequence[Sequence[str]],
    rejection_fractions: Sequence[float],
) -> dict[str, object]:
    """Fit and scan exclusively on rows marked ``development``."""

    required_split = {"material_id", "partition"}
    if not required_split.issubset(split.columns):
        raise ValueError("split lacks required columns")
    split_frame = split.loc[:, ["material_id", "partition"]].copy()
    split_frame["material_id"] = split_frame["material_id"].astype(str)
    if split_frame["material_id"].duplicated().any() or not set(
        split_frame["partition"]
    ).issubset({"development", "confirmation"}):
        raise ValueError("split is invalid")
    expected_ids = split_frame["material_id"].tolist()
    validated = _validated_feature_frame(
        features, expected_ids=expected_ids, role="NEXT30 analytic features"
    )
    if "material_id" not in labels or label_column not in labels:
        raise ValueError("labels lack required columns")
    label_frame = labels.loc[:, ["material_id", label_column]].copy()
    label_frame["material_id"] = label_frame["material_id"].astype(str)
    if label_frame["material_id"].duplicated().any() or set(
        label_frame["material_id"]
    ) != set(expected_ids):
        raise ValueError("label IDs differ from split")
    joined = (
        split_frame.merge(validated, on="material_id", validate="one_to_one")
        .merge(label_frame, on="material_id", validate="one_to_one")
        .sort_values("material_id", kind="stable", ignore_index=True)
    )
    development = joined.loc[joined["partition"].eq("development")].copy()
    if development.empty:
        raise ValueError("development partition is empty")
    endpoint = pd.to_numeric(development[label_column], errors="coerce").to_numpy(float)
    if not np.isfinite(endpoint).all():
        raise ValueError("development hull endpoint must be finite")
    parameters = fit_term_parameters(development, term_specs)
    result = scan_development(
        development,
        energy=endpoint,
        term_specs=term_specs,
        formulas=formulas,
        rejection_fractions=rejection_fractions,
        parameters=parameters,
    )
    result["parameters"] = parameters
    result["development_rows"] = len(development)
    result["confirmation_rows_used"] = 0
    return result
