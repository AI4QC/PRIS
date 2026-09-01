#!/usr/bin/env python3
"""Zero-refit external QMOF validation of the frozen NEXT31 packing law."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

from ase import Atoms
from ase.io import read
import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.next11_geometry_only_frames import (
    _ParsedFrame,
    _load_archive_only,
    _write_deterministic_archive,
)
from src.next12_pauling_controls import DECISIONS, RULES, _classical_features
from src.next27_periodic_packing import compute_periodic_features
from src.next31_omc25_energy_ranking import compute_energy_risk
from src.next39_next23_predictions import _pauling_values
from src.next39_trajectory_evaluate import (
    _fingerprint_environment,
    exact_structure_fingerprint,
)


PROTOCOL = "2026-08-03-next48-qmof-external-validation-v1"
EXPECTED_QMOF_ARCHIVE_SHA256 = (
    "97d23c0b4f9e5a30888e53dc16222b90443ad7167c3284d2258615d9f44eceef"
)
EXPECTED_NEXT31_RULE_SHA256 = (
    "993d64b851c755fc5cc0d4b68ca7ca6994d4bdb7ed666f860d43a04925e254a8"
)
METADATA_MEMBER = "qmof_database/qmof.csv"
UNRELAXED_ARCHIVE_MEMBER = "qmof_database/unrelaxed_structures.zip"
RELAXED_ARCHIVE_MEMBER = "qmof_database/relaxed_structures.zip"
INITIAL_PREFIX = "unrelaxed_structures/other/"
RELAXED_PREFIX = "relaxed_structures/"
METADATA_COLUMNS = (
    "qmof_id",
    "name",
    "info.formula",
    "info.formula_reduced",
    "info.natoms",
    "info.source",
)
PROTECTED_MAX = 0.10
SUBSTANTIAL_MIN = 0.20
SEVERE_MIN = 0.50
GATES: Mapping[str, float] = {
    "coverage_lower_at_least": 0.95,
    "protected_recall_lower_at_least": 0.95,
    "reject_precision_lower_at_least": 0.70,
    "savings_lower_at_least": 0.02,
    "auc_substantial_at_least": 0.85,
}
PROTOCOL_NAME = "NEXT48_QMOF_PROTOCOL.json"
COHORT_NAME = "next48_qmof_cohort.parquet"
UNMATCHED_NAME = "NEXT48_QMOF_UNMATCHED.json"
MANIFEST_NAME = "MANIFEST.json"
GEOMETRY_NAME = "geometry_only_frames.zip"
FEATURES_NAME = "next48_qmof_label_free_features.parquet"
PREDICTIONS_NAME = "next48_qmof_predictions.parquet"
EVALUATION_NAME = "NEXT48_QMOF_EVALUATION.json"
JOINED_NAME = "next48_qmof_joined_evaluation.parquet"
ONE_SIDED_95_Z = 1.6448536269514722


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    def safe(item: object) -> object:
        if isinstance(item, dict):
            return {str(key): safe(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(value) for value in item]
        if isinstance(item, (np.integer, np.bool_)):
            return item.item()
        if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            return None
        return item

    return (json.dumps(safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _nested_members(outer: zipfile.ZipFile, member: str) -> tuple[str, ...]:
    try:
        payload = outer.read(member)
        with zipfile.ZipFile(io.BytesIO(payload)) as nested:
            return tuple(nested.namelist())
    except (KeyError, OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"invalid nested QMOF archive: {member}") from exc


def _whitelisted_metadata(outer: zipfile.ZipFile) -> pd.DataFrame:
    try:
        with outer.open(METADATA_MEMBER) as handle:
            table = pd.read_csv(handle, usecols=list(METADATA_COLUMNS))
    except (KeyError, OSError, UnicodeError, ValueError) as exc:
        raise ValueError("invalid QMOF metadata CSV") from exc
    if table.empty or any(column not in table for column in METADATA_COLUMNS):
        raise ValueError("QMOF metadata whitelist is incomplete")
    table = table.loc[:, list(METADATA_COLUMNS)].copy()
    for column in ("qmof_id", "name", "info.formula", "info.formula_reduced", "info.source"):
        if table[column].isna().any():
            raise ValueError(f"QMOF metadata {column} contains missing values")
        table[column] = table[column].astype(str)
    table["info.natoms"] = pd.to_numeric(table["info.natoms"], errors="raise").astype(int)
    if (table["info.natoms"] <= 0).any() or table["qmof_id"].duplicated().any():
        raise ValueError("QMOF metadata identity or atom counts are invalid")
    return table


def _cohort_from_archive(
    source_archive_path: Path,
) -> tuple[pd.DataFrame, list[str], int]:
    try:
        with zipfile.ZipFile(source_archive_path) as outer:
            metadata = _whitelisted_metadata(outer)
            initial_members = sorted(
                member
                for member in _nested_members(outer, UNRELAXED_ARCHIVE_MEMBER)
                if member.startswith(INITIAL_PREFIX) and member.endswith(".cif")
            )
            relaxed_members = {
                PurePosixPath(member).stem: member
                for member in _nested_members(outer, RELAXED_ARCHIVE_MEMBER)
                if member.startswith(RELAXED_PREFIX) and member.endswith(".cif")
            }
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("invalid QMOF source archive") from exc
    if not initial_members or not relaxed_members:
        raise ValueError("QMOF initial or relaxed member set is empty")
    initial_names = [PurePosixPath(member).stem for member in initial_members]
    if len(initial_names) != len(set(initial_names)):
        raise ValueError("QMOF initial member stems are not unique")
    by_name: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in metadata.to_dict("records"):
        by_name[str(row["name"])].append(row)
    rows: list[dict[str, object]] = []
    unmatched: list[str] = []
    for member, name in zip(initial_members, initial_names, strict=True):
        candidates = by_name.get(name, [])
        if len(candidates) != 1:
            unmatched.append(name)
            continue
        source = candidates[0]
        material_id = str(source["qmof_id"])
        relaxed_member = relaxed_members.get(material_id)
        if relaxed_member is None:
            unmatched.append(name)
            continue
        rows.append(
            {
                "material_id": material_id,
                "initial_name": name,
                "initial_member": member,
                "relaxed_member": relaxed_member,
                "formula": str(source["info.formula"]),
                "reduced_formula": str(source["info.formula_reduced"]),
                "natoms": int(source["info.natoms"]),
                "source_family": str(source["info.source"]),
                "input_role": "unrelaxed_x0_geometry_only",
            }
        )
    cohort = pd.DataFrame(rows).sort_values(
        ["initial_name", "material_id"], kind="stable", ignore_index=True
    )
    if cohort.empty or cohort["material_id"].duplicated().any():
        raise ValueError("QMOF eligible cohort identities are invalid")
    return cohort, sorted(unmatched), len(initial_members)


def _publish_directory(staging: Path, target: Path) -> None:
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    os.rename(staging, target)


def _strict_json(path: Path, *, role: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _validate_protocol_artifacts(
    *,
    source_archive_path: Path,
    frozen_rule_path: Path,
    cohort_path: Path,
    protocol_path: Path,
    protocol_manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, str]]:
    paths = {
        "source_archive": Path(source_archive_path).resolve(),
        "frozen_rule": Path(frozen_rule_path).resolve(),
        "cohort": Path(cohort_path).resolve(),
        "protocol": Path(protocol_path).resolve(),
        "protocol_manifest": Path(protocol_manifest_path).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT48 protocol or input artifact is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    protocol = _strict_json(paths["protocol"], role="NEXT48 protocol")
    manifest = _strict_json(paths["protocol_manifest"], role="NEXT48 protocol manifest")
    outputs = manifest.get("outputs_sha256")
    source_record = protocol.get("source_archive")
    rule_record = protocol.get("frozen_rule")
    endpoint = protocol.get("endpoint")
    if (
        protocol.get("protocol") != PROTOCOL
        or protocol.get("labels_opened") is not False
        or protocol.get("initial_coordinate_payloads_opened") is not False
        or protocol.get("relaxed_coordinate_payloads_opened") is not False
        or protocol.get("thresholds_refit") is not False
        or protocol.get("gates") != dict(GATES)
        or not isinstance(endpoint, Mapping)
        or endpoint.get("protected_max") != PROTECTED_MAX
        or endpoint.get("substantial_min") != SUBSTANTIAL_MIN
        or endpoint.get("severe_min") != SEVERE_MIN
        or not isinstance(source_record, Mapping)
        or source_record.get("sha256") != hashes["source_archive"]
        or not isinstance(rule_record, Mapping)
        or rule_record.get("sha256") != hashes["frozen_rule"]
        or not isinstance(outputs, Mapping)
        or outputs.get(paths["cohort"].name) != hashes["cohort"]
        or outputs.get(paths["protocol"].name) != hashes["protocol"]
    ):
        raise ValueError("NEXT48 frozen protocol binding differs")
    cohort = pd.read_parquet(paths["cohort"])
    required = {
        "material_id",
        "initial_name",
        "initial_member",
        "relaxed_member",
        "formula",
        "reduced_formula",
        "natoms",
        "source_family",
        "input_role",
    }
    if (
        not required.issubset(cohort.columns)
        or cohort.empty
        or cohort["material_id"].astype(str).duplicated().any()
        or not cohort["input_role"].eq("unrelaxed_x0_geometry_only").all()
    ):
        raise ValueError("NEXT48 cohort identity or role differs")
    cohort = cohort.sort_values(
        ["initial_name", "material_id"], kind="stable", ignore_index=True
    )
    return cohort, protocol, hashes


def _geometry_only_cif(payload: bytes) -> Atoms:
    try:
        parsed = read(io.BytesIO(payload), format="cif", index=-1)
    except Exception as exc:
        raise ValueError("invalid QMOF CIF geometry") from exc
    if not isinstance(parsed, Atoms):
        raise ValueError("QMOF CIF did not produce one structure")
    atoms = Atoms(
        numbers=np.asarray(parsed.numbers, dtype=int),
        positions=np.asarray(parsed.positions, dtype=float),
        cell=np.asarray(parsed.cell.array, dtype=float),
        pbc=True,
    )
    if (
        len(atoms) <= 0
        or not np.isfinite(atoms.positions).all()
        or not np.isfinite(atoms.cell.array).all()
        or abs(float(np.linalg.det(atoms.cell.array))) < 1e-12
    ):
        raise ValueError("QMOF CIF geometry is empty or nonperiodic")
    return atoms


def _pauling_abstention(error: str) -> dict[str, object]:
    row: dict[str, object] = {"pauling_feature_error": error}
    for name in RULES:
        row[f"{name}_value"] = math.nan
        row[f"{name}_decision"] = "ABSTAIN"
    row["pauling_p2_p5_decision"] = "ABSTAIN"
    return row


def freeze_qmof_predictions(
    *,
    source_archive_path: Path,
    frozen_rule_path: Path,
    cohort_path: Path,
    protocol_path: Path,
    protocol_manifest_path: Path,
    output_dir: Path,
    analytic_feature_calculator=None,
    pauling_feature_calculator=None,
) -> dict[str, object]:
    """Read unrelaxed CIFs only and seal NEXT31 plus Pauling decisions."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    cohort, _protocol, input_hashes = _validate_protocol_artifacts(
        source_archive_path=source_archive_path,
        frozen_rule_path=frozen_rule_path,
        cohort_path=cohort_path,
        protocol_path=protocol_path,
        protocol_manifest_path=protocol_manifest_path,
    )
    source = Path(source_archive_path).resolve()
    rule_path = Path(frozen_rule_path).resolve()
    try:
        rule = json.loads(rule_path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid frozen NEXT31 rule") from exc
    if not isinstance(rule, Mapping):
        raise ValueError("frozen NEXT31 rule must be an object")
    analytic_fn = compute_periodic_features if analytic_feature_calculator is None else analytic_feature_calculator
    pauling_fn = _classical_features if pauling_feature_calculator is None else pauling_feature_calculator
    frames: dict[str, _ParsedFrame] = {}
    rows: list[dict[str, object]] = []
    try:
        with zipfile.ZipFile(source) as outer:
            initial_bytes = outer.read(UNRELAXED_ARCHIVE_MEMBER)
        with zipfile.ZipFile(io.BytesIO(initial_bytes)) as initial_archive:
            names = set(initial_archive.namelist())
            for upstream in cohort.to_dict("records"):
                material_id = str(upstream["material_id"])
                member = str(upstream["initial_member"])
                row: dict[str, object] = {
                    "material_id": material_id,
                    "source_family": str(upstream["source_family"]),
                    "initial_name": str(upstream["initial_name"]),
                    "natoms_expected": int(upstream["natoms"]),
                    "x0_geometry_supported": False,
                    "analytic_feature_error": None,
                    "periodic_nonbond_vdw_q05": math.nan,
                    "periodic_contact_coord105": math.nan,
                }
                if member not in names:
                    row["analytic_feature_error"] = "initial CIF member is missing"
                    row.update(_pauling_abstention("initial CIF member is missing"))
                    rows.append(row)
                    continue
                try:
                    atoms = _geometry_only_cif(initial_archive.read(member))
                    if len(atoms) != int(upstream["natoms"]):
                        raise ValueError("initial CIF atom count differs from metadata")
                    row["x0_geometry_supported"] = True
                    frames[material_id] = _ParsedFrame(atoms, (), ())
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    row["analytic_feature_error"] = error
                    row.update(_pauling_abstention(error))
                    rows.append(row)
                    continue
                try:
                    values = dict(analytic_fn(atoms))
                    row["periodic_nonbond_vdw_q05"] = float(
                        values["periodic_nonbond_vdw_q05"]
                    )
                    row["periodic_contact_coord105"] = float(
                        values["periodic_contact_coord105"]
                    )
                except Exception as exc:
                    row["analytic_feature_error"] = f"{type(exc).__name__}: {exc}"
                row.update(_pauling_values(atoms, pauling_fn))
                rows.append(row)
    except (KeyError, OSError, zipfile.BadZipFile) as exc:
        raise ValueError("invalid QMOF unrelaxed nested archive") from exc
    features = pd.DataFrame(rows).sort_values(
        "material_id", kind="stable", ignore_index=True
    )
    features["analytic_supported"] = (
        features["x0_geometry_supported"].astype(bool)
        & np.isfinite(features["periodic_nonbond_vdw_q05"].to_numpy(float))
        & np.isfinite(features["periodic_contact_coord105"].to_numpy(float))
    )
    score, supported, reject = compute_energy_risk(features, rule)
    features["analytic_supported"] = supported
    predictions = features.copy()
    predictions["next31_risk_score"] = score
    predictions["reject"] = reject
    predictions["input_role"] = "unrelaxed_x0_geometry_only"
    if len(predictions) != len(cohort) or predictions["material_id"].duplicated().any():
        raise RuntimeError("NEXT48 prediction identity accounting differs")
    if bool((predictions["reject"] & ~predictions["analytic_supported"]).any()):
        raise RuntimeError("NEXT48 analytic failures did not fail open")

    repository = Path(__file__).resolve().parents[1]
    source_names = (
        "src/next11_geometry_only_frames.py",
        "src/next12_pauling_controls.py",
        "src/next27_periodic_packing.py",
        "src/next31_omc25_energy_ranking.py",
        "src/next39_next23_predictions.py",
        "src/next48_qmof_external_validation.py",
    )
    source_hashes = {name: _sha256(repository / name) for name in source_names}
    production = bool(
        analytic_feature_calculator is None
        and pauling_feature_calculator is None
        and input_hashes["source_archive"] == EXPECTED_QMOF_ARCHIVE_SHA256
        and input_hashes["frozen_rule"] == EXPECTED_NEXT31_RULE_SHA256
    )
    counts: dict[str, object] = {
        "rows": len(predictions),
        "x0_geometry_supported": int(predictions["x0_geometry_supported"].sum()),
        "analytic_supported": int(supported.sum()),
        "rejected": int(reject.sum()),
        "pauling": {
            decision: int(predictions["pauling_p2_p5_decision"].eq(decision).sum())
            for decision in DECISIONS
        },
        "sources": dict(sorted(Counter(predictions["source_family"]).items())),
    }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "external_qmof_x0_prediction_freeze",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_role": "unrelaxed_x0_geometry_only",
        "labels_opened": False,
        "endpoint_columns_selected": False,
        "relaxed_coordinate_payloads_opened": False,
        "dft_or_energy_proxy_used_at_execution": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "same_composition_candidates_used": False,
        "thresholds_refit": False,
        "missing_policy": "fail_open_do_not_reject",
        "counts": counts,
        "inputs_sha256": {
            "source_archive": input_hashes["source_archive"],
            "frozen_rule": input_hashes["frozen_rule"],
            "cohort": input_hashes["cohort"],
            "protocol": input_hashes["protocol"],
            "protocol_manifest": input_hashes["protocol_manifest"],
        },
        "executed_source_sha256": source_hashes,
        "production_protocol_eligible": production,
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        geometry_path = staging / GEOMETRY_NAME
        features_path = staging / FEATURES_NAME
        predictions_path = staging / PREDICTIONS_NAME
        _write_deterministic_archive(geometry_path, frames)
        features.to_parquet(features_path, index=False)
        predictions.to_parquet(predictions_path, index=False)
        manifest["outputs_sha256"] = {
            path.name: _sha256(path)
            for path in (geometry_path, features_path, predictions_path)
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        validation_paths = {
            "source_archive": Path(source_archive_path).resolve(),
            "frozen_rule": Path(frozen_rule_path).resolve(),
            "cohort": Path(cohort_path).resolve(),
            "protocol": Path(protocol_path).resolve(),
            "protocol_manifest": Path(protocol_manifest_path).resolve(),
        }
        if any(_sha256(path) != input_hashes[name] for name, path in validation_paths.items()):
            raise RuntimeError("NEXT48 prediction input changed during publication")
        if any(_sha256(repository / name) != digest for name, digest in source_hashes.items()):
            raise RuntimeError("NEXT48 prediction source changed during publication")
        _publish_directory(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def _wilson_lower(successes: int, total: int) -> float:
    if total <= 0:
        return math.nan
    probability = successes / total
    z = ONE_SIDED_95_Z
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        probability * (1.0 - probability) / total
        + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - radius)


def _metric(successes: int, total: int) -> dict[str, float | int]:
    return {
        "numerator": int(successes),
        "denominator": int(total),
        "estimate": float(successes / total) if total else math.nan,
        "wilson_lower_onesided_95": _wilson_lower(successes, total),
    }


def _relaxation_metrics(
    *,
    endpoint: np.ndarray,
    endpoint_supported: np.ndarray,
    method_supported: np.ndarray,
    reject: np.ndarray,
    score: np.ndarray | None,
    apply_gates: bool,
) -> dict[str, object]:
    endpoint = np.asarray(endpoint, dtype=float)
    endpoint_supported = np.asarray(endpoint_supported, dtype=bool)
    method_supported = np.asarray(method_supported, dtype=bool)
    reject = np.asarray(reject, dtype=bool)
    if not (
        len(endpoint)
        == len(endpoint_supported)
        == len(method_supported)
        == len(reject)
    ):
        raise ValueError("NEXT48 metric arrays have different lengths")
    if np.any(reject & ~method_supported):
        raise ValueError("NEXT48 method rejection crossed fail-open support")
    if not np.isfinite(endpoint[endpoint_supported]).all():
        raise ValueError("NEXT48 supported endpoints are not finite")
    protected = endpoint_supported & (endpoint <= PROTECTED_MAX)
    substantial = endpoint_supported & (endpoint >= SUBSTANTIAL_MIN)
    severe = endpoint_supported & (endpoint >= SEVERE_MIN)
    joint_support = endpoint_supported & method_supported
    evaluated_reject = endpoint_supported & reject
    auc = math.nan
    rho = math.nan
    if score is not None:
        score_array = np.asarray(score, dtype=float)
        if len(score_array) != len(endpoint):
            raise ValueError("NEXT48 score length differs")
        diagnostic = joint_support & np.isfinite(score_array)
        if diagnostic.any() and len(np.unique(substantial[diagnostic])) == 2:
            auc = float(roc_auc_score(substantial[diagnostic], score_array[diagnostic]))
        if diagnostic.sum() >= 2:
            rho = float(spearmanr(score_array[diagnostic], endpoint[diagnostic]).statistic)
    result: dict[str, object] = {
        "counts": {
            "rows": len(endpoint),
            "endpoint_supported": int(endpoint_supported.sum()),
            "method_supported": int(method_supported.sum()),
            "joint_supported": int(joint_support.sum()),
            "rejected": int(evaluated_reject.sum()),
            "protected": int(protected.sum()),
            "substantial": int(substantial.sum()),
            "severe": int(severe.sum()),
            "rejected_substantial": int((evaluated_reject & substantial).sum()),
            "rejected_protected": int((evaluated_reject & protected).sum()),
        },
        "endpoint_coverage": _metric(int(endpoint_supported.sum()), len(endpoint)),
        "coverage": _metric(int(joint_support.sum()), len(endpoint)),
        "protected_recall": _metric(
            int((protected & ~reject).sum()), int(protected.sum())
        ),
        "reject_precision": _metric(
            int((evaluated_reject & substantial).sum()), int(evaluated_reject.sum())
        ),
        "substantial_recall": _metric(
            int((evaluated_reject & substantial).sum()), int(substantial.sum())
        ),
        "savings": _metric(int(evaluated_reject.sum()), len(endpoint)),
        "auc_substantial": auc,
        "spearman_fingerprint_change": rho,
    }
    if apply_gates:
        clauses = {
            "coverage_lower_at_least": float(
                result["coverage"]["wilson_lower_onesided_95"]  # type: ignore[index]
            )
            >= GATES["coverage_lower_at_least"],
            "protected_recall_lower_at_least": float(
                result["protected_recall"]["wilson_lower_onesided_95"]  # type: ignore[index]
            )
            >= GATES["protected_recall_lower_at_least"],
            "reject_precision_lower_at_least": float(
                result["reject_precision"]["wilson_lower_onesided_95"]  # type: ignore[index]
            )
            >= GATES["reject_precision_lower_at_least"],
            "savings_lower_at_least": float(
                result["savings"]["wilson_lower_onesided_95"]  # type: ignore[index]
            )
            >= GATES["savings_lower_at_least"],
            "auc_substantial_at_least": math.isfinite(auc)
            and auc >= GATES["auc_substantial_at_least"],
        }
        result["clauses"] = clauses
        result["prospective_gate_pass"] = bool(all(clauses.values()))
    return result


def _validate_prediction_artifacts(
    *,
    predictions_path: Path,
    prediction_manifest_path: Path,
    geometry_path: Path,
    protocol_hashes: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[str, object], dict[str, str]]:
    paths = {
        "predictions": Path(predictions_path).resolve(),
        "prediction_manifest": Path(prediction_manifest_path).resolve(),
        "geometry": Path(geometry_path).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT48 prediction artifact is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    manifest = _strict_json(paths["prediction_manifest"], role="NEXT48 prediction manifest")
    outputs = manifest.get("outputs_sha256")
    inputs = manifest.get("inputs_sha256")
    if (
        manifest.get("protocol") != PROTOCOL
        or manifest.get("labels_opened") is not False
        or manifest.get("relaxed_coordinate_payloads_opened") is not False
        or manifest.get("thresholds_refit") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get(paths["predictions"].name) != hashes["predictions"]
    ):
        raise ValueError("NEXT48 prediction output hash or freeze contract differs")
    if outputs.get(paths["geometry"].name) != hashes["geometry"]:
        raise ValueError("NEXT48 geometry output hash differs")
    if not isinstance(inputs, Mapping) or any(
        inputs.get(name) != protocol_hashes[name]
        for name in (
            "source_archive",
            "cohort",
            "protocol",
            "protocol_manifest",
        )
    ):
        raise ValueError("NEXT48 prediction/protocol input binding differs")
    table = pd.read_parquet(paths["predictions"]).sort_values(
        "material_id", kind="stable", ignore_index=True
    )
    required = {
        "material_id",
        "source_family",
        "x0_geometry_supported",
        "analytic_supported",
        "next31_risk_score",
        "reject",
        "pauling_p2_p5_decision",
        "input_role",
    }
    if (
        not required.issubset(table.columns)
        or table.empty
        or table["material_id"].astype(str).duplicated().any()
        or not table["input_role"].eq("unrelaxed_x0_geometry_only").all()
        or bool((table["reject"].astype(bool) & ~table["analytic_supported"].astype(bool)).any())
    ):
        raise ValueError("NEXT48 prediction identity or fail-open contract differs")
    return table, manifest, hashes


def evaluate_qmof_relaxation(
    *,
    source_archive_path: Path,
    cohort_path: Path,
    protocol_path: Path,
    protocol_manifest_path: Path,
    predictions_path: Path,
    prediction_manifest_path: Path,
    geometry_path: Path,
    output_dir: Path,
    fingerprint_calculator=None,
) -> dict[str, object]:
    """Open mapped relaxed CIFs only after frozen predictions validate."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    protocol_value = _strict_json(Path(protocol_path), role="NEXT48 protocol")
    frozen_rule = protocol_value.get("frozen_rule")
    if not isinstance(frozen_rule, Mapping) or not isinstance(frozen_rule.get("path"), str):
        raise ValueError("NEXT48 protocol frozen-rule path is invalid")
    cohort, _protocol, protocol_hashes = _validate_protocol_artifacts(
        source_archive_path=source_archive_path,
        frozen_rule_path=Path(str(frozen_rule["path"])),
        cohort_path=cohort_path,
        protocol_path=protocol_path,
        protocol_manifest_path=protocol_manifest_path,
    )
    predictions, prediction_manifest, prediction_hashes = _validate_prediction_artifacts(
        predictions_path=predictions_path,
        prediction_manifest_path=prediction_manifest_path,
        geometry_path=geometry_path,
        protocol_hashes=protocol_hashes,
    )
    cohort_ids = set(cohort["material_id"].astype(str))
    if set(predictions["material_id"].astype(str)) != cohort_ids:
        raise ValueError("NEXT48 prediction identities differ from frozen cohort")
    geometry_ids = sorted(
        predictions.loc[predictions["x0_geometry_supported"].astype(bool), "material_id"].astype(str)
    )
    loaded_ids, initial_structures = _load_archive_only(
        Path(geometry_path).resolve(), tuple(geometry_ids)
    )
    if loaded_ids != geometry_ids:
        raise ValueError("NEXT48 x0 geometry identity differs")
    initial_by_id = dict(zip(loaded_ids, initial_structures, strict=True))
    calculator = exact_structure_fingerprint if fingerprint_calculator is None else fingerprint_calculator
    fingerprint_environment = (
        _fingerprint_environment()
        if fingerprint_calculator is None
        else {"calculator": "injected_test_double"}
    )
    input_paths = {
        "source_archive": Path(source_archive_path).resolve(),
        "cohort": Path(cohort_path).resolve(),
        "protocol": Path(protocol_path).resolve(),
        "protocol_manifest": Path(protocol_manifest_path).resolve(),
        "predictions": Path(predictions_path).resolve(),
        "prediction_manifest": Path(prediction_manifest_path).resolve(),
        "geometry": Path(geometry_path).resolve(),
    }
    preopening_hashes = {name: _sha256(path) for name, path in input_paths.items()}
    endpoint_rows: list[dict[str, object]] = []
    failures: Counter[str] = Counter()
    try:
        with zipfile.ZipFile(input_paths["source_archive"]) as outer:
            relaxed_bytes = outer.read(RELAXED_ARCHIVE_MEMBER)
        with zipfile.ZipFile(io.BytesIO(relaxed_bytes)) as relaxed_archive:
            members = set(relaxed_archive.namelist())
            for upstream in cohort.to_dict("records"):
                material_id = str(upstream["material_id"])
                member = str(upstream["relaxed_member"])
                supported = False
                change = math.nan
                volume_change = math.nan
                error: str | None = None
                try:
                    if material_id not in initial_by_id:
                        raise ValueError("initial geometry is unsupported")
                    if member not in members:
                        raise ValueError("relaxed CIF member is missing")
                    initial = initial_by_id[material_id]
                    final = _geometry_only_cif(relaxed_archive.read(member))
                    if Counter(initial.numbers.tolist()) != Counter(final.numbers.tolist()):
                        raise ValueError("initial and relaxed element counts differ")
                    first = np.asarray(calculator(initial), dtype=float)
                    second = np.asarray(calculator(final), dtype=float)
                    if (
                        first.ndim != 1
                        or second.shape != first.shape
                        or not np.isfinite(first).all()
                        or not np.isfinite(second).all()
                    ):
                        raise ValueError("QMOF structure fingerprints are invalid")
                    change = float(np.linalg.norm(second - first))
                    initial_volume = float(initial.get_volume())
                    final_volume = float(final.get_volume())
                    if initial_volume <= 0.0 or final_volume <= 0.0:
                        raise ValueError("QMOF structure volume is non-positive")
                    volume_change = abs(math.log(final_volume / initial_volume))
                    supported = True
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    failures[error] += 1
                endpoint_rows.append(
                    {
                        "material_id": material_id,
                        "endpoint_supported": supported,
                        "fingerprint_change": change,
                        "abs_log_volume_change": volume_change,
                        "endpoint_error": error,
                    }
                )
    except (KeyError, OSError, zipfile.BadZipFile) as exc:
        raise ValueError("invalid QMOF relaxed nested archive") from exc
    endpoints = pd.DataFrame(endpoint_rows).sort_values(
        "material_id", kind="stable", ignore_index=True
    )
    joined = predictions.merge(endpoints, on="material_id", validate="one_to_one")
    if len(joined) != len(cohort):
        raise ValueError("NEXT48 endpoint identities differ from frozen cohort")
    endpoint = joined["fingerprint_change"].to_numpy(float)
    endpoint_supported = joined["endpoint_supported"].to_numpy(bool)
    next31_metrics = _relaxation_metrics(
        endpoint=endpoint,
        endpoint_supported=endpoint_supported,
        method_supported=joined["analytic_supported"].to_numpy(bool),
        reject=joined["reject"].to_numpy(bool),
        score=joined["next31_risk_score"].to_numpy(float),
        apply_gates=True,
    )
    pauling_decision = joined["pauling_p2_p5_decision"].astype(str)
    pauling_metrics = _relaxation_metrics(
        endpoint=endpoint,
        endpoint_supported=endpoint_supported,
        method_supported=pauling_decision.ne("ABSTAIN").to_numpy(bool),
        reject=pauling_decision.eq("REJECT").to_numpy(bool),
        score=None,
        apply_gates=False,
    )
    per_source: dict[str, object] = {}
    for source_family, part in joined.groupby("source_family", sort=True):
        supported_endpoint = part["endpoint_supported"].to_numpy(bool)
        values = part["fingerprint_change"].to_numpy(float)
        decisions = part["pauling_p2_p5_decision"].astype(str)
        per_source[str(source_family)] = {
            "next31": _relaxation_metrics(
                endpoint=values,
                endpoint_supported=supported_endpoint,
                method_supported=part["analytic_supported"].to_numpy(bool),
                reject=part["reject"].to_numpy(bool),
                score=part["next31_risk_score"].to_numpy(float),
                apply_gates=False,
            ),
            "pauling_p2_p5": _relaxation_metrics(
                endpoint=values,
                endpoint_supported=supported_endpoint,
                method_supported=decisions.ne("ABSTAIN").to_numpy(bool),
                reject=decisions.eq("REJECT").to_numpy(bool),
                score=None,
                apply_gates=False,
            ),
        }
    endpoint_counts = {
        "supported": int(endpoint_supported.sum()),
        "protected": int((endpoint_supported & (endpoint <= PROTECTED_MAX)).sum()),
        "substantial": int((endpoint_supported & (endpoint >= SUBSTANTIAL_MIN)).sum()),
        "severe": int((endpoint_supported & (endpoint >= SEVERE_MIN)).sum()),
    }
    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels_opened_after_predictions_frozen": True,
        "thresholds_refit": False,
        "features_added_after_opening": False,
        "endpoint": {
            "name": "crystalnn_site_stats_fingerprint_l2_change",
            "protected_max": PROTECTED_MAX,
            "substantial_min": SUBSTANTIAL_MIN,
            "severe_min": SEVERE_MIN,
            "secondary": "absolute_log_volume_change",
        },
        "endpoint_counts": endpoint_counts,
        "endpoint_failure_counts": dict(sorted(failures.items())),
        "gates": dict(GATES),
        "next31": next31_metrics,
        "pauling_p2_p5": pauling_metrics,
        "per_source_descriptive_only": per_source,
        "fingerprint_environment": fingerprint_environment,
        "claim_boundary": (
            "QMOF PBE-D3(BJ) structural-relaxation response only; not formation "
            "energy, convex-hull, dynamical, or thermodynamic stability"
        ),
    }
    repository = Path(__file__).resolve().parents[1]
    source_names = (
        "src/next11_geometry_only_frames.py",
        "src/next31_omc25_energy_ranking.py",
        "src/next39_trajectory_evaluate.py",
        "src/next48_qmof_external_validation.py",
    )
    source_hashes = {name: _sha256(repository / name) for name in source_names}
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "opened_external_qmof_relaxed_geometry_evaluation",
        "labels_opened": True,
        "relaxed_coordinate_payloads_opened": True,
        "evaluation_only_dft_final_geometry_read": True,
        "evaluation_only_dft_energy_read": False,
        "law_execution_dft_values_read": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "thresholds_refit": False,
        "counts": {"rows": len(joined), **endpoint_counts},
        "inputs_sha256_before_endpoint_opening": preopening_hashes,
        "prediction_input_sha256": prediction_hashes,
        "executed_source_sha256": source_hashes,
        "production_protocol_eligible": bool(
            fingerprint_calculator is None
            and prediction_manifest.get("production_protocol_eligible") is True
        ),
        "frozen_rule_passes_primary_gates": bool(
            next31_metrics["prospective_gate_pass"]
        ),
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        joined_path = staging / JOINED_NAME
        result_path = staging / EVALUATION_NAME
        joined.to_parquet(joined_path, index=False)
        result_path.write_bytes(_json_bytes(result))
        manifest["outputs_sha256"] = {
            joined_path.name: _sha256(joined_path),
            result_path.name: _sha256(result_path),
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if any(_sha256(path) != preopening_hashes[name] for name, path in input_paths.items()):
            raise RuntimeError("NEXT48 evaluation input changed during publication")
        if any(_sha256(repository / name) != digest for name, digest in source_hashes.items()):
            raise RuntimeError("NEXT48 evaluation source changed during publication")
        _publish_directory(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def freeze_qmof_protocol(
    *,
    source_archive_path: Path,
    frozen_rule_path: Path,
    output_dir: Path,
    expected_archive_sha256: str = EXPECTED_QMOF_ARCHIVE_SHA256,
    expected_rule_sha256: str = EXPECTED_NEXT31_RULE_SHA256,
) -> dict[str, object]:
    """Freeze cohort identities and all gates before any CIF coordinates are parsed."""

    source = Path(source_archive_path).resolve()
    rule = Path(frozen_rule_path).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if not source.is_file() or not rule.is_file():
        raise FileNotFoundError("QMOF archive or frozen NEXT31 rule is missing")
    source_hash = _sha256(source)
    rule_hash = _sha256(rule)
    if source_hash != expected_archive_sha256:
        raise ValueError("QMOF archive hash differs")
    if rule_hash != expected_rule_sha256:
        raise ValueError("NEXT31 frozen rule hash differs")
    cohort, unmatched, initial_count = _cohort_from_archive(source)
    counts = {
        "initial_members": initial_count,
        "cohort": len(cohort),
        "unmatched": len(unmatched),
    }
    protocol: dict[str, object] = {
        "protocol": PROTOCOL,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_archive": {"path": str(source), "sha256": source_hash},
        "frozen_rule": {"path": str(rule), "sha256": rule_hash},
        "cohort_selection": (
            "all sorted unrelaxed_structures/other CIF stems with exactly one "
            "whitelisted metadata match and one qmof-id relaxed CIF"
        ),
        "counts": counts,
        "metadata_columns_selected": list(METADATA_COLUMNS),
        "endpoint_columns_selected": False,
        "metadata_member_opened": True,
        "nested_zip_central_directories_opened": True,
        "initial_coordinate_payloads_opened": False,
        "relaxed_coordinate_payloads_opened": False,
        "labels_opened": False,
        "thresholds_refit": False,
        "physical_never_read_lockbox": False,
        "archive_contains_accessible_endpoint_data": True,
        "endpoint": {
            "name": "crystalnn_site_stats_fingerprint_l2_change",
            "protected_max": PROTECTED_MAX,
            "substantial_min": SUBSTANTIAL_MIN,
            "severe_min": SEVERE_MIN,
        },
        "gates": dict(GATES),
        "primary_scope": "overall eligible QMOF cohort",
        "per_source_results_descriptive_only": True,
        "execution_contract": (
            "one unrelaxed x0 plus frozen analytic tables and deterministic geometry only"
        ),
        "forbidden_at_execution": [
            "DFT values or calculations",
            "relaxed structures or trajectories",
            "MLIP or learned energy-force-stress proxies",
            "physical relaxation",
            "same-composition alternatives",
        ],
        "claim_boundary": (
            "external PBE-D3(BJ) structural-relaxation response only; not formation "
            "energy, convex-hull, dynamical, or thermodynamic stability"
        ),
    }
    source_code = Path(__file__).resolve()
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "pre_coordinate_pre_endpoint_external_protocol_freeze",
        "labels_opened": False,
        "initial_coordinate_payloads_opened": False,
        "relaxed_coordinate_payloads_opened": False,
        "counts": counts,
        "inputs_sha256": {
            "source_archive": {"path": str(source), "sha256": source_hash},
            "frozen_rule": {"path": str(rule), "sha256": rule_hash},
        },
        "executed_source_sha256": {
            "src/next48_qmof_external_validation.py": _sha256(source_code)
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        cohort_path = staging / COHORT_NAME
        protocol_path = staging / PROTOCOL_NAME
        unmatched_path = staging / UNMATCHED_NAME
        cohort.to_parquet(cohort_path, index=False)
        protocol_path.write_bytes(_json_bytes(protocol))
        unmatched_path.write_bytes(
            _json_bytes({"count": len(unmatched), "initial_names": unmatched})
        )
        manifest["outputs_sha256"] = {
            path.name: _sha256(path)
            for path in (cohort_path, protocol_path, unmatched_path)
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source) != source_hash or _sha256(rule) != rule_hash:
            raise RuntimeError("NEXT48 protocol input changed during publication")
        if _sha256(source_code) != manifest["executed_source_sha256"][  # type: ignore[index]
            "src/next48_qmof_external_validation.py"
        ]:
            raise RuntimeError("NEXT48 protocol source changed during publication")
        _publish_directory(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


__all__ = [
    "COHORT_NAME",
    "EXPECTED_NEXT31_RULE_SHA256",
    "EXPECTED_QMOF_ARCHIVE_SHA256",
    "EVALUATION_NAME",
    "FEATURES_NAME",
    "GATES",
    "GEOMETRY_NAME",
    "MANIFEST_NAME",
    "PREDICTIONS_NAME",
    "PROTECTED_MAX",
    "PROTOCOL_NAME",
    "SEVERE_MIN",
    "SUBSTANTIAL_MIN",
    "UNMATCHED_NAME",
    "evaluate_qmof_relaxation",
    "freeze_qmof_predictions",
    "freeze_qmof_protocol",
]
