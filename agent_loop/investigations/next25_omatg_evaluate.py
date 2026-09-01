#!/usr/bin/env python3
"""Open MP-20 references once and evaluate frozen NEXT25 CSP screening."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import tempfile

from ase import Atoms
import lmdb
import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.stats import spearmanr

from src.next11_geometry_only_frames import (
    _ParsedFrame,
    _load_archive_only,
    _write_deterministic_archive,
)
from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next23_evaluate import _roc_auc
from src.next23_relaxation_rule import wilson_lower_bound
from src.next25_apply_rule import PROTOCOL as PREDICTION_PROTOCOL
from src.next25_omatg_compositions import PROTOCOL as COMPOSITION_PROTOCOL
from src.next25_omatg_holdout import PROTOCOL as HOLDOUT_PROTOCOL
from src.next25_pauling_controls import PROTOCOL as PAULING_PROTOCOL


PROTOCOL = "2026-08-03-next25-omatg-dft-reference-csp-evaluation-v1"
RESULT_NAME = "NEXT25_CSP_EVALUATION.json"
JOINED_NAME = "next25_csp_joined.parquet"
REFERENCE_NAME = "reference_geometry_frames.zip"
LABEL_OPENING_NAME = "LABEL_OPENING.json"
MANIFEST_NAME = "MANIFEST.json"
LTOL = 0.3
STOL = 0.5
ANGLE_TOL = 10.0
PRIMARY_GATES: Mapping[str, float] = {
    "coverage_lower": 0.90,
    "match_protection_recall_lower": 0.95,
    "nonmatch_rejection_precision_lower": 0.90,
    "savings_lower": 0.10,
}
FORMAL_TEST_LMDB_SHA256 = (
    "74223481f135274e54375c52e38ace5ab7a2403f367a4dd8352cfd1874b2986d"
)
FORMAL_COMPOSITION_COHORT_SHA256 = (
    "1b1a5b007b772967e4707f6c294545c202068c0b392ba087be66a7a293413f5d"
)
FORMAL_COMPOSITION_MANIFEST_SHA256 = (
    "d1177218f58b1393a48238b130571a1836adb16357dd9804b90af51fdf813ad3"
)
MatchFunction = Callable[[Atoms, Atoms], float | None]


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _read_json(path: Path, *, role: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be an object")
    return value


def _validate_output_hash(
    manifest: Mapping[str, object], path: Path, *, role: str
) -> None:
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(path.name) != _sha256(path):
        raise ValueError(f"{role} hash differs")


def _as_numpy(value: object, *, role: str) -> np.ndarray:
    try:
        if all(hasattr(value, method) for method in ("detach", "cpu", "numpy")):
            array = value.detach().cpu().numpy()  # type: ignore[union-attr]
        else:
            array = np.asarray(value)
    except Exception as exc:
        raise ValueError(f"invalid reference {role}") from exc
    return array


def extract_reference_geometry(record: Mapping[str, object]) -> Atoms:
    """Read exactly species, Cartesian positions, and cell from one label row."""

    try:
        numbers = _as_numpy(record["atomic_numbers"], role="atomic_numbers")
        positions = _as_numpy(record["pos"], role="positions")
        cell = _as_numpy(record["cell"], role="cell")
    except (KeyError, TypeError) as exc:
        raise ValueError("reference record lacks the geometry triplet") from exc
    if (
        numbers.ndim != 1
        or numbers.dtype.kind not in "iu"
        or len(numbers) < 1
        or np.any(numbers < 1)
        or np.any(numbers > 118)
        or positions.shape != (len(numbers), 3)
        or cell.shape != (3, 3)
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(cell))
    ):
        raise ValueError("invalid reference geometry triplet")
    return Atoms(
        numbers=numbers.astype(int),
        positions=positions.astype(float),
        cell=cell.astype(float),
        pbc=True,
    )


def default_match_pair(generated: Atoms, reference: Atoms) -> float | None:
    """Return OMatG benchmark normalized RMSD, or ``None`` for a non-match."""

    matcher = StructureMatcher(ltol=LTOL, stol=STOL, angle_tol=ANGLE_TOL)
    adaptor = AseAtomsAdaptor()
    result = matcher.get_rms_dist(
        adaptor.get_structure(generated), adaptor.get_structure(reference)
    )
    if result is None:
        return None
    value = float(result[0])
    if not math.isfinite(value) or value < 0 or value > STOL + 1e-12:
        raise ValueError("StructureMatcher returned an invalid normalized RMSD")
    return value


def _decision_metrics(
    *, supported: np.ndarray, reject: np.ndarray, reference_match: np.ndarray
) -> dict[str, object]:
    supported = np.asarray(supported, dtype=bool)
    reject = np.asarray(reject, dtype=bool) & supported
    reference_match = np.asarray(reference_match, dtype=bool)
    n_rows = len(reference_match)
    n_supported = int(supported.sum())
    n_rejected = int(reject.sum())
    n_matches = int(reference_match.sum())
    matched_kept = int((reference_match & ~reject).sum())
    nonmatch = ~reference_match
    nonmatches_rejected = int((nonmatch & reject).sum())
    metrics: dict[str, object] = {
        "rows": n_rows,
        "supported": n_supported,
        "rejected": n_rejected,
        "reference_matches": n_matches,
        "matched_kept": matched_kept,
        "nonmatches": int(nonmatch.sum()),
        "nonmatches_rejected": nonmatches_rejected,
        "coverage": n_supported / n_rows if n_rows else 0.0,
        "coverage_lower": wilson_lower_bound(n_supported, n_rows),
        "match_protection_recall": matched_kept / n_matches if n_matches else 0.0,
        "match_protection_recall_lower": wilson_lower_bound(matched_kept, n_matches),
        "nonmatch_rejection_precision": (
            nonmatches_rejected / n_rejected if n_rejected else 0.0
        ),
        "nonmatch_rejection_precision_lower": wilson_lower_bound(
            nonmatches_rejected, n_rejected
        ),
        "savings": n_rejected / n_rows if n_rows else 0.0,
        "savings_lower": wilson_lower_bound(n_rejected, n_rows),
        "nonmatch_recall": (
            nonmatches_rejected / int(nonmatch.sum()) if int(nonmatch.sum()) else 0.0
        ),
    }
    metrics["passes_primary_gates"] = all(
        float(metrics[name]) >= cutoff for name, cutoff in PRIMARY_GATES.items()
    )
    return metrics


def _analytic_subgroup_metrics(
    joined: pd.DataFrame, masks: Mapping[str, np.ndarray]
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, mask in masks.items():
        subset = joined.loc[np.asarray(mask, dtype=bool)]
        result[name] = _decision_metrics(
            supported=subset["analytic_supported"].to_numpy(bool),
            reject=subset["reject"].to_numpy(bool),
            reference_match=subset["reference_match"].to_numpy(bool),
        )
    return result


def _secondary_diagnostics(joined: pd.DataFrame) -> dict[str, object]:
    natoms = joined["natoms"].to_numpy(int)
    atom_subgroups = _analytic_subgroup_metrics(
        joined,
        {
            "2-5": (natoms >= 2) & (natoms <= 5),
            "6-10": (natoms >= 6) & (natoms <= 10),
            "11-20": (natoms >= 11) & (natoms <= 20),
        },
    )
    element_counts = np.asarray(
        [
            len(set(int(value) for value in json.loads(str(serialized))))
            for serialized in joined["atomic_numbers_json"]
        ],
        dtype=int,
    )
    element_subgroups = _analytic_subgroup_metrics(
        joined,
        {
            "unary": element_counts == 1,
            "binary": element_counts == 2,
            "ternary": element_counts == 3,
            "quaternary_plus": element_counts >= 4,
        },
    )

    analytic_decision = np.full(len(joined), "FAIL_OPEN", dtype=object)
    supported = joined["analytic_supported"].to_numpy(bool)
    reject = joined["reject"].to_numpy(bool) & supported
    analytic_decision[supported & ~reject] = "KEEP"
    analytic_decision[reject] = "REJECT"
    disagreement_frame = pd.DataFrame(
        {
            "analytic_decision": analytic_decision,
            "pauling_combined_decision": joined["pauling_p2_p5_decision"].astype(str),
            "reference_match": joined["reference_match"].to_numpy(bool),
        }
    )
    disagreement: list[dict[str, object]] = []
    for (analytic, pauling), subset in disagreement_frame.groupby(
        ["analytic_decision", "pauling_combined_decision"], sort=True
    ):
        matches = int(subset["reference_match"].sum())
        disagreement.append(
            {
                "analytic_decision": str(analytic),
                "pauling_combined_decision": str(pauling),
                "rows": len(subset),
                "reference_matches": matches,
                "reference_nonmatches": len(subset) - matches,
            }
        )

    score = joined["next23_risk_score"].to_numpy(float)
    finite = supported & np.isfinite(score)
    calibration: list[dict[str, object]] = []
    finite_frame = joined.loc[
        finite,
        ["next23_risk_score", "reference_match", "corrected_rmsd", "reject"],
    ].copy()
    if len(finite_frame):
        quantile = pd.qcut(
            finite_frame["next23_risk_score"], 4, labels=False, duplicates="drop"
        )
        finite_frame["risk_quantile"] = quantile.astype(int)
        for bin_index, subset in finite_frame.groupby("risk_quantile", sort=True):
            calibration.append(
                {
                    "risk_quantile": int(bin_index),
                    "rows": len(subset),
                    "risk_min": float(subset["next23_risk_score"].min()),
                    "risk_max": float(subset["next23_risk_score"].max()),
                    "reference_nonmatch_rate": float(
                        (~subset["reference_match"].to_numpy(bool)).mean()
                    ),
                    "mean_corrected_rmsd": float(subset["corrected_rmsd"].mean()),
                    "rejection_rate": float(subset["reject"].to_numpy(bool).mean()),
                }
            )
    return {
        "atom_count_subgroups": atom_subgroups,
        "element_count_subgroups": element_subgroups,
        "decision_disagreement": disagreement,
        "risk_score_calibration": calibration,
    }


def _parse_frozen_time(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("prediction was not frozen before reference opening")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("prediction frozen timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("prediction frozen timestamp lacks timezone")
    return parsed.astimezone(timezone.utc)


def _validate_label_free_inputs(
    paths: Mapping[str, Path], *, require_formal_inputs: bool
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, datetime, bool
]:
    prediction_manifest = _read_json(
        paths["prediction_manifest"], role="prediction manifest"
    )
    _validate_output_hash(
        prediction_manifest, paths["predictions"], role="prediction"
    )
    frozen_at = _parse_frozen_time(prediction_manifest.get("frozen_at_utc"))
    if (
        prediction_manifest.get("protocol") != PREDICTION_PROTOCOL
        or prediction_manifest.get("blind_labels_opened") is not False
        or prediction_manifest.get("endpoint_fields_read") is not False
        or prediction_manifest.get("thresholds_refit") is not False
        or prediction_manifest.get("formula_or_parameters_changed") is not False
    ):
        raise ValueError("prediction was not frozen before reference opening")

    pauling_manifest = _read_json(paths["pauling_manifest"], role="Pauling manifest")
    _validate_output_hash(
        pauling_manifest, paths["pauling_controls"], role="Pauling control"
    )
    if (
        pauling_manifest.get("protocol") != PAULING_PROTOCOL
        or pauling_manifest.get("labels_opened") is not False
        or pauling_manifest.get("endpoint_artifacts_opened") is not False
        or pauling_manifest.get("thresholds_refit") is not False
        or pauling_manifest.get("rules_changed") is not False
    ):
        raise ValueError("Pauling controls crossed the reference boundary")

    holdout_manifest = _read_json(paths["holdout_manifest"], role="holdout manifest")
    _validate_output_hash(holdout_manifest, paths["metadata"], role="holdout metadata")
    _validate_output_hash(holdout_manifest, paths["geometry"], role="holdout geometry")
    if (
        holdout_manifest.get("protocol") != HOLDOUT_PROTOCOL
        or holdout_manifest.get("labels_opened") is not False
        or holdout_manifest.get("endpoint_artifacts_opened") is not False
        or holdout_manifest.get("relaxed_structures_opened") is not False
        or holdout_manifest.get("all_generator_outputs_retained") is not True
    ):
        raise ValueError("holdout crossed the reference boundary")

    composition_manifest = _read_json(
        paths["composition_manifest"], role="composition manifest"
    )
    _validate_output_hash(
        composition_manifest, paths["composition_cohort"], role="composition cohort"
    )
    if (
        composition_manifest.get("protocol") != COMPOSITION_PROTOCOL
        or composition_manifest.get("input_role") != "composition_only"
        or composition_manifest.get("reference_geometry_fields_accessed") is not False
        or composition_manifest.get("labels_opened") is not False
    ):
        raise ValueError("composition selection crossed the reference boundary")
    production_chain_eligible = all(
        manifest.get("production_protocol_eligible") is True
        for manifest in (
            composition_manifest,
            holdout_manifest,
            prediction_manifest,
            pauling_manifest,
        )
    )
    if require_formal_inputs and not production_chain_eligible:
        raise ValueError("formal label-free chain is not production-eligible")
    source_inputs = composition_manifest.get("inputs_sha256")
    source_record = source_inputs.get("test_lmdb") if isinstance(source_inputs, Mapping) else None
    if not isinstance(source_record, Mapping) or source_record.get("sha256") != _sha256(
        paths["test_lmdb"]
    ):
        raise ValueError("MP-20 test LMDB identity differs from composition selection")

    metadata = pd.read_parquet(paths["metadata"])
    composition = pd.read_parquet(paths["composition_cohort"])
    predictions = pd.read_parquet(paths["predictions"])
    pauling = pd.read_parquet(paths["pauling_controls"])
    for role, frame in (
        ("metadata", metadata),
        ("composition", composition),
        ("predictions", predictions),
        ("Pauling", pauling),
    ):
        if "material_id" not in frame or frame["material_id"].isna().any():
            raise ValueError(f"{role} lacks material IDs")
        frame["material_id"] = frame["material_id"].astype(str)
        if frame["material_id"].duplicated().any():
            raise ValueError(f"{role} material IDs must be unique")
    expected = set(metadata["material_id"])
    if any(set(frame["material_id"]) != expected for frame in (composition, predictions, pauling)):
        raise ValueError("label-free inputs do not join one-to-one")
    if (
        not predictions["input_role"].eq("unrelaxed_x0_geometry_only").all()
        or predictions[["analytic_supported", "reject"]].isna().any().any()
    ):
        raise ValueError("frozen prediction columns differ")
    return (
        metadata,
        composition,
        predictions,
        pauling,
        frozen_at,
        production_chain_eligible,
    )


def _load_selected_references(
    *, test_lmdb_path: Path, composition: pd.DataFrame
) -> tuple[dict[str, Atoms], int]:
    try:
        environment = lmdb.open(
            str(test_lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=1,
        )
    except (lmdb.Error, OSError) as exc:
        raise ValueError("invalid MP-20 test LMDB") from exc
    references: dict[str, Atoms] = {}
    accessed = 0
    try:
        with environment.begin(write=False) as transaction:
            for row in composition.sort_values("material_id", kind="stable").to_dict("records"):
                source_index = int(row["source_index"])
                payload = transaction.get(str(source_index).encode("ascii"))
                if payload is None:
                    raise ValueError(f"missing MP-20 test row {source_index}")
                try:
                    record = pickle.loads(payload)
                except Exception as exc:
                    raise ValueError(f"invalid MP-20 test row {source_index}") from exc
                if not isinstance(record, Mapping):
                    raise ValueError("MP-20 reference row must be a mapping")
                atoms = extract_reference_geometry(record)
                expected_numbers = sorted(json.loads(str(row["atomic_numbers_json"])))
                if sorted(atoms.get_atomic_numbers().tolist()) != expected_numbers:
                    raise ValueError("reference composition differs from frozen cohort")
                references[str(row["material_id"])] = atoms
                accessed += 1
    finally:
        environment.close()
    return references, accessed


def evaluate_omatg_csp(
    *,
    composition_cohort_path: Path,
    composition_manifest_path: Path,
    test_lmdb_path: Path,
    cohort_metadata_path: Path,
    geometry_zip_path: Path,
    holdout_manifest_path: Path,
    predictions_path: Path,
    prediction_manifest_path: Path,
    pauling_controls_path: Path,
    pauling_manifest_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
    match_function: MatchFunction | None = None,
) -> dict[str, object]:
    """Open only selected references after frozen predictions and evaluate once."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "composition_cohort": Path(composition_cohort_path).resolve(),
        "composition_manifest": Path(composition_manifest_path).resolve(),
        "test_lmdb": Path(test_lmdb_path).resolve(),
        "metadata": Path(cohort_metadata_path).resolve(),
        "geometry": Path(geometry_zip_path).resolve(),
        "holdout_manifest": Path(holdout_manifest_path).resolve(),
        "predictions": Path(predictions_path).resolve(),
        "prediction_manifest": Path(prediction_manifest_path).resolve(),
        "pauling_controls": Path(pauling_controls_path).resolve(),
        "pauling_manifest": Path(pauling_manifest_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    input_hashes = {role: _sha256(path) for role, path in paths.items()}
    formal_identity = (
        input_hashes["test_lmdb"] == FORMAL_TEST_LMDB_SHA256
        and input_hashes["composition_cohort"]
        == FORMAL_COMPOSITION_COHORT_SHA256
        and input_hashes["composition_manifest"]
        == FORMAL_COMPOSITION_MANIFEST_SHA256
    )
    if require_formal_inputs and not formal_identity:
        raise ValueError("formal MP-20 or composition identity differs")

    # Every decision artifact is validated before reference record access.
    (
        metadata,
        composition,
        predictions,
        pauling,
        frozen_at,
        production_chain_eligible,
    ) = _validate_label_free_inputs(paths, require_formal_inputs=require_formal_inputs)
    opened_at_dt = datetime.now(timezone.utc)
    if frozen_at >= opened_at_dt:
        raise ValueError("prediction frozen timestamp does not predate reference opening")
    opened_at = opened_at_dt.isoformat()

    expected_ids = tuple(sorted(metadata["material_id"].astype(str)))
    loaded_ids, generated = _load_archive_only(paths["geometry"], expected_ids)
    if loaded_ids != list(expected_ids):
        raise ValueError("generated geometry order differs")
    references, accessed_rows = _load_selected_references(
        test_lmdb_path=paths["test_lmdb"], composition=composition
    )
    matcher = default_match_pair if match_function is None else match_function
    endpoint_rows: list[dict[str, object]] = []
    reference_frames: dict[str, _ParsedFrame] = {}
    for material_id, generated_atoms in zip(loaded_ids, generated, strict=True):
        reference = references[material_id]
        value = matcher(generated_atoms, reference)
        if value is not None and (
            not math.isfinite(float(value)) or float(value) < 0 or float(value) > STOL + 1e-12
        ):
            raise ValueError("match function returned an invalid RMSD")
        matched = value is not None
        endpoint_rows.append(
            {
                "material_id": material_id,
                "reference_match": matched,
                "matched_rmsd": float(value) if matched else np.nan,
                "corrected_rmsd": float(value) if matched else STOL,
            }
        )
        reference_frames[material_id] = _ParsedFrame(
            atoms=reference,
            dropped_comment_fields=(),
            dropped_atom_properties=(),
        )
    endpoint = pd.DataFrame(endpoint_rows)
    pauling_payload = pauling.drop(
        columns=[name for name in ("rk", "formula", "natoms") if name in pauling],
    )
    joined = (
        metadata.loc[:, ["material_id", "rk", "formula", "natoms"]]
        .merge(
            composition.loc[
                :,
                [
                    "material_id",
                    "source_index",
                    "selection_rank",
                    "atomic_numbers_json",
                ],
            ],
            on="material_id",
            validate="one_to_one",
        )
        .merge(predictions, on="material_id", validate="one_to_one")
        .merge(pauling_payload, on="material_id", validate="one_to_one")
        .merge(endpoint, on="material_id", validate="one_to_one")
        .sort_values("material_id", kind="stable", ignore_index=True)
    )
    analytic_metrics = _decision_metrics(
        supported=joined["analytic_supported"].to_numpy(bool),
        reject=joined["reject"].to_numpy(bool),
        reference_match=joined["reference_match"].to_numpy(bool),
    )
    pauling_columns = {
        "pauling_p2": "pauling_p2_decision",
        "pauling_p3": "pauling_p3_decision",
        "pauling_p4": "pauling_p4_decision",
        "pauling_p5": "pauling_p5_decision",
        "pauling_p2_p5_combined": "pauling_p2_p5_decision",
    }
    pauling_metrics: dict[str, object] = {}
    for name, column in pauling_columns.items():
        if column not in joined:
            raise ValueError(f"Pauling controls lack {column}")
        decisions = joined[column].astype(str).to_numpy()
        if not set(decisions) <= {"KEEP", "REJECT", "ABSTAIN"}:
            raise ValueError("Pauling decision vocabulary differs")
        pauling_metrics[name] = _decision_metrics(
            supported=decisions != "ABSTAIN",
            reject=decisions == "REJECT",
            reference_match=joined["reference_match"].to_numpy(bool),
        )
    safe_pauling_savings = [
        float(value["savings_lower"])
        for value in pauling_metrics.values()
        if bool(value["passes_primary_gates"])
    ]
    best_safe_pauling = max(safe_pauling_savings, default=0.0)
    beyond_pauling = bool(
        analytic_metrics["passes_primary_gates"]
        and float(analytic_metrics["savings_lower"]) > best_safe_pauling
    )
    supported = joined["analytic_supported"].to_numpy(bool)
    score = joined["next23_risk_score"].to_numpy(float)
    match_values = joined["reference_match"].to_numpy(bool)
    finite = supported & np.isfinite(score)
    if int(finite.sum()) >= 2:
        rho_value = spearmanr(
            score[finite], joined.loc[finite, "corrected_rmsd"].to_numpy(float)
        ).statistic
        rho = float(rho_value) if math.isfinite(float(rho_value)) else None
    else:
        rho = None
    continuous = {
        "supported_rows": int(finite.sum()),
        "auc_nonmatch": _roc_auc(score[finite], ~match_values[finite]),
        "spearman_risk_vs_corrected_rmsd": rho,
    }
    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "opened_at_utc": opened_at,
        "prediction_frozen_at_utc": frozen_at.isoformat(),
        "thresholds_refit_after_reference_opening": False,
        "formula_or_parameters_changed_after_reference_opening": False,
        "endpoint": {
            "name": "one_to_one_mp20_dft_reference_structure_match",
            "matching_tolerances": {
                "ltol": LTOL,
                "stol": STOL,
                "angle_tol_degrees": ANGLE_TOL,
            },
            "reference_rows_accessed": accessed_rows,
            "reference_matches": int(match_values.sum()),
            "reference_nonmatches": int((~match_values).sum()),
            "nonmatch_corrected_rmsd": STOL,
        },
        "primary_gates": dict(PRIMARY_GATES),
        "analytic_rule": analytic_metrics,
        "pauling_controls": pauling_metrics,
        "best_safe_pauling_savings_lower": best_safe_pauling,
        "beyond_pauling_on_this_endpoint": beyond_pauling,
        "continuous_diagnostics": continuous,
        "secondary_diagnostics": _secondary_diagnostics(joined),
        "claim_boundary": {
            "endpoint_is_csp_reference_recovery": True,
            "nonmatch_is_thermodynamic_instability": False,
            "convex_hull_stability_established": False,
            "alternate_polymorph_possible": True,
        },
        "production_protocol_eligible": bool(
            formal_identity and production_chain_eligible and match_function is None
        ),
    }
    label_opening = {
        "opened_at_utc": opened_at,
        "prediction_frozen_at_utc": frozen_at.isoformat(),
        "prediction_sha256_before_opening": input_hashes["predictions"],
        "pauling_sha256_before_opening": input_hashes["pauling_controls"],
        "mp20_test_lmdb_sha256": input_hashes["test_lmdb"],
        "selected_reference_rows_opened": accessed_rows,
        "thresholds_refit": False,
    }

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next25_omatg_evaluate.py": Path(__file__).resolve(),
        "src/next25_apply_rule.py": repository_root / "src/next25_apply_rule.py",
        "src/next25_pauling_controls.py": repository_root
        / "src/next25_pauling_controls.py",
        "docs/plans/2026-08-03-next25-omatg-blind-csp-design.md": repository_root
        / "docs/plans/2026-08-03-next25-omatg-blind-csp-design.md",
        "docs/plans/2026-08-03-next25-omatg-blind-csp-implementation.md": repository_root
        / "docs/plans/2026-08-03-next25-omatg-blind-csp-implementation.md",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        joined_path = staging / JOINED_NAME
        reference_path = staging / REFERENCE_NAME
        joined.to_parquet(joined_path, index=False)
        _write_deterministic_archive(reference_path, reference_frames)
        (staging / RESULT_NAME).write_bytes(_json_bytes(result))
        (staging / LABEL_OPENING_NAME).write_bytes(_json_bytes(label_opening))
        output_names = (JOINED_NAME, REFERENCE_NAME, RESULT_NAME, LABEL_OPENING_NAME)
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "mode": "post_freeze_one_shot_dft_reference_evaluation",
            "opened_at_utc": opened_at,
            "thresholds_refit": False,
            "formula_or_parameters_changed": False,
            "inputs_sha256": {
                role: {"path": str(path), "sha256": input_hashes[role]}
                for role, path in paths.items()
            },
            "executed_source_sha256": {
                relative: _sha256(path) for relative, path in source_paths.items()
            },
            "outputs_sha256": {
                name: _sha256(staging / name) for name in output_names
            },
            "production_protocol_eligible": result["production_protocol_eligible"],
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        for role, path in paths.items():
            if _sha256(path) != input_hashes[role]:
                raise RuntimeError(f"input {role} changed during reference evaluation")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-cohort", type=Path, required=True)
    parser.add_argument("--composition-manifest", type=Path, required=True)
    parser.add_argument("--test-lmdb", type=Path, required=True)
    parser.add_argument("--cohort-metadata", type=Path, required=True)
    parser.add_argument("--geometry-zip", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--pauling-controls", type=Path, required=True)
    parser.add_argument("--pauling-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_omatg_csp(
        composition_cohort_path=args.composition_cohort,
        composition_manifest_path=args.composition_manifest,
        test_lmdb_path=args.test_lmdb,
        cohort_metadata_path=args.cohort_metadata,
        geometry_zip_path=args.geometry_zip,
        holdout_manifest_path=args.holdout_manifest,
        predictions_path=args.predictions,
        prediction_manifest_path=args.prediction_manifest,
        pauling_controls_path=args.pauling_controls,
        pauling_manifest_path=args.pauling_manifest,
        output_dir=args.output_dir,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
