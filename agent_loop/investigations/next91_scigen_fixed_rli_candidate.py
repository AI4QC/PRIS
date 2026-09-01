"""NEXT91 endpoint-free freeze of the fixed SCIGEN RLI candidate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from src.next85_scigen_label_free_features import (
    CATALOGUE_NAME as FEATURE_CATALOGUE_NAME,
    FEATURE_NAMES,
    MANIFEST_NAME as FEATURE_MANIFEST_NAME,
    PROTOCOL as FEATURE_PROTOCOL,
)
from src.next86_scigen_term_catalogue import (
    CATALOGUE_NAME as TERM_CATALOGUE_NAME,
    MANIFEST_NAME as TERM_MANIFEST_NAME,
    PROTOCOL as TERM_PROTOCOL,
)
from src.next87_scigen_sparse_law_search import (
    EVALUATION_NAME as NEXT87_EVALUATION_NAME,
    FORMULA_KIND,
    MANIFEST_NAME as NEXT87_MANIFEST_NAME,
    MISSING_POLICY,
    PROTOCOL as NEXT87_PROTOCOL,
    SEARCH_RECORD_NAME as NEXT87_SEARCH_RECORD_NAME,
    _publish_directory_no_replace,
    _read_json,
    _sha256_file,
    apply_scigen_formula,
)


PROTOCOL = "2026-08-03-next91-scigen-fixed-rli-candidate-v1"
RLI_THRESHOLD = 3.915855102781074
RLI_TERM_WEIGHTS = (
    ("sivr_edge_mismatch_max__high", 1.0),
    ("sscp_load_rms__high", 4.0),
)
EXPECTED_TERM_PARAMETERS = {
    "sivr_edge_mismatch_max__high": {
        "feature": "sivr_edge_mismatch_max",
        "direction": 1,
        "transform": "log1p_nonnegative",
        "center": 0.34809689849136527,
        "scale": 0.2268496027212349,
    },
    "sscp_load_rms__high": {
        "feature": "sscp_load_rms",
        "direction": 1,
        "transform": "log1p_nonnegative",
        "center": 0.09650974330938514,
        "scale": 0.07475030243877033,
    },
}
EXPECTED_DISCOVERY_METRICS = {
    "metric_severe_rejected": 1359,
    "metric_severe_rejection_precision_lower": 0.9429092910402352,
    "metric_protected_recall_lower": 0.9746034559783944,
    "metric_coverage_lower": 0.9794285414401448,
    "metric_savings_lower": 0.11953000620096647,
    "pooled_extreme_auc": 0.7788719322036274,
}
MANIFEST_NAME = "MANIFEST.json"
FORMULA_NAME = "NEXT91_FROZEN_RLI_FORMULA.json"
PREDICTION_NAMES = {
    role: f"next91_frozen_rli_predictions_{role}.parquet" for role in FEATURE_NAMES
}
EXPECTED_INPUT_SHA256 = {
    "feature_manifest": "8dcb8118f85ee4a3acbf0905f01c2b173d58742a1e16dcd6004adbbbedcf63cc",
    "feature_catalogue": "f34b09a4a9f18b0202b8daf606b7baab7bdae826871bcc60a4be858a8c1cc96a",
    "features_discovery": "7031d86e4fb6e469c674d208f680ace1dbe5e11e45f3d4b2befefd747efdde16",
    "features_internal_validation": "f266e6143bc23d9e131b5ec788676b520db928aa46a57a1fcba6fd8530a80c8a",
    "features_internal_replication": "2d420ac76f8b9e1ea6a7908df92a4db1198bc0ef0b2d410875225d51536214b2",
    "term_manifest": "5b80f948a35a40ef79438ea1902b92a40dd07c35a4b541826252eb92cf96f1eb",
    "term_catalogue": "e8f9fe532c15673c0a74737632b0145d43f6494cb1ea7e94e7380198fd4e4dee",
    "next87_manifest": "2f8122d964aae2590963442162aae925e3437e4b947a19eaa856fa649ba6becc",
    "next87_evaluation": "49778a16d67998f1384704210f3bf12c3c65b8e2488c003930d3928b2a0c1be7",
    "next87_search": "438c98681ddf7366bccaf88f36221142c1851873d89632c9d04196bffed7dac2",
    "design": "6a61ed0acfc3bdb50b87fe9ccbfd8283b8af5d4ff521ffc6f41c91dc000737c5",
}


def build_frozen_rli_formula(
    eligible_terms: list[Mapping[str, object]],
) -> dict[str, object]:
    """Build the exact RLI formula only when NEXT86 parameters match."""

    by_id: dict[str, Mapping[str, object]] = {}
    for term in eligible_terms:
        term_id = str(term.get("term_id"))
        if not term_id or term_id in by_id:
            raise ValueError("NEXT91 term catalogue identity differs")
        by_id[term_id] = term
    formula_terms: list[dict[str, object]] = []
    for term_id, weight in RLI_TERM_WEIGHTS:
        term = by_id.get(term_id)
        expected = EXPECTED_TERM_PARAMETERS[term_id]
        if term is None or any(term.get(key) != value for key, value in expected.items()):
            raise ValueError(f"NEXT91 fixed RLI term differs: {term_id}")
        center = float(term["center"])
        scale = float(term["scale"])
        if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"NEXT91 fixed RLI term differs: {term_id}")
        formula_terms.append(
            {
                "term_id": term_id,
                "feature": str(term["feature"]),
                "group": str(term.get("group", "unspecified")),
                "direction": int(term["direction"]),
                "transform": str(term["transform"]),
                "center": center,
                "scale": scale,
                "weight": weight,
            }
        )
    return {
        "kind": FORMULA_KIND,
        "missing_policy": MISSING_POLICY,
        "terms": formula_terms,
        "threshold": RLI_THRESHOLD,
    }


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _validate_next87_candidate(search: pd.DataFrame) -> dict[str, object]:
    required = {"term_ids_json", "weights_json", "threshold", *EXPECTED_DISCOVERY_METRICS}
    if required - set(search.columns):
        raise ValueError("NEXT91 NEXT87 search columns differ")
    expected_ids = [term_id for term_id, _weight in RLI_TERM_WEIGHTS]
    matches = []
    for index, row in search.iterrows():
        try:
            term_ids = json.loads(str(row["term_ids_json"]))
            weights = [float(value) for value in json.loads(str(row["weights_json"]))]
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if (
            term_ids == expected_ids
            and weights == [weight for _term_id, weight in RLI_TERM_WEIGHTS]
            and math.isclose(float(row["threshold"]), RLI_THRESHOLD, abs_tol=1.0e-15)
        ):
            matches.append(index)
    if len(matches) != 1:
        raise ValueError("NEXT91 fixed RLI candidate row differs")
    row = search.loc[matches[0]]
    for name, expected in EXPECTED_DISCOVERY_METRICS.items():
        observed = float(row[name])
        if not math.isclose(observed, float(expected), rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"NEXT91 fixed RLI discovery metric differs: {name}")
    return {name: float(row[name]) for name in EXPECTED_DISCOVERY_METRICS}


def freeze_scigen_rli_candidate(
    *,
    feature_dir: Path,
    term_catalogue_dir: Path,
    next87_dir: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Freeze RLI predictions without accepting or opening any endpoint."""

    feature_root = Path(feature_dir).resolve()
    term_root = Path(term_catalogue_dir).resolve()
    next87_root = Path(next87_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "feature_manifest": feature_root / FEATURE_MANIFEST_NAME,
        "feature_catalogue": feature_root / FEATURE_CATALOGUE_NAME,
        **{
            f"features_{role}": feature_root / FEATURE_NAMES[role]
            for role in FEATURE_NAMES
        },
        "term_manifest": term_root / TERM_MANIFEST_NAME,
        "term_catalogue": term_root / TERM_CATALOGUE_NAME,
        "next87_manifest": next87_root / NEXT87_MANIFEST_NAME,
        "next87_evaluation": next87_root / NEXT87_EVALUATION_NAME,
        "next87_search": next87_root / NEXT87_SEARCH_RECORD_NAME,
        "design": Path(design_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT91 input is missing")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT91 formal input identity differs")

    feature_manifest = _read_json(paths["feature_manifest"], role="NEXT85 manifest")
    feature_outputs = feature_manifest.get("outputs_sha256")
    if (
        feature_manifest.get("protocol") != FEATURE_PROTOCOL
        or feature_manifest.get("labels_opened") is not False
        or feature_manifest.get("endpoint_payloads_opened") is not False
        or feature_manifest.get("relaxed_structures_opened") is not False
        or not isinstance(feature_outputs, Mapping)
        or feature_outputs.get(FEATURE_CATALOGUE_NAME) != hashes["feature_catalogue"]
        or any(
            feature_outputs.get(FEATURE_NAMES[role]) != hashes[f"features_{role}"]
            for role in FEATURE_NAMES
        )
    ):
        raise ValueError("NEXT91 received invalid NEXT85 provenance")
    term_manifest = _read_json(paths["term_manifest"], role="NEXT86 term manifest")
    term_catalogue = _read_json(paths["term_catalogue"], role="NEXT86 term catalogue")
    term_outputs = term_manifest.get("outputs_sha256")
    if (
        term_manifest.get("protocol") != TERM_PROTOCOL
        or term_manifest.get("labels_opened") is not False
        or term_manifest.get("endpoint_payloads_opened") is not False
        or not isinstance(term_outputs, Mapping)
        or term_outputs.get(TERM_CATALOGUE_NAME) != hashes["term_catalogue"]
        or term_catalogue.get("protocol") != TERM_PROTOCOL
        or term_catalogue.get("labels_opened") is not False
        or not isinstance(term_catalogue.get("eligible_terms"), list)
    ):
        raise ValueError("NEXT91 received invalid NEXT86 provenance")
    next87_manifest = _read_json(paths["next87_manifest"], role="NEXT87 manifest")
    next87_outputs = next87_manifest.get("outputs_sha256")
    next87_evaluation = _read_json(paths["next87_evaluation"], role="NEXT87 evaluation")
    if (
        next87_manifest.get("protocol") != NEXT87_PROTOCOL
        or next87_manifest.get("validation_endpoint_opened") is not False
        or next87_manifest.get("replication_endpoint_opened") is not False
        or not isinstance(next87_outputs, Mapping)
        or next87_outputs.get(NEXT87_EVALUATION_NAME) != hashes["next87_evaluation"]
        or next87_outputs.get(NEXT87_SEARCH_RECORD_NAME) != hashes["next87_search"]
        or next87_evaluation.get("protocol") != NEXT87_PROTOCOL
    ):
        raise ValueError("NEXT91 received invalid NEXT87 provenance")
    discovery_metrics = _validate_next87_candidate(pd.read_parquet(paths["next87_search"]))
    formula = {
        **build_frozen_rli_formula(term_catalogue["eligible_terms"]),
        "protocol": PROTOCOL,
        "candidate_name": "Rigidity-Load Incompatibility (RLI)",
        "selection_status": "post-discovery fixed candidate",
        "validation_endpoint_opened": False,
        "replication_endpoint_opened": False,
    }

    feature_tables: dict[str, pd.DataFrame] = {}
    for role in FEATURE_NAMES:
        table = pd.read_parquet(paths[f"features_{role}"])
        if (
            "material_id" not in table
            or "partition_role" not in table
            or table["material_id"].astype(str).duplicated().any()
            or set(table["partition_role"].astype(str)) != {role}
        ):
            raise ValueError(f"NEXT91 {role} feature identity differs")
        feature_tables[role] = table

    source_path = Path(__file__).resolve()
    source_hash = _sha256_file(source_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    output_paths: list[Path] = []
    try:
        formula_path = staging / FORMULA_NAME
        formula_path.write_bytes(_json_bytes(formula))
        formula_sha256 = _sha256_file(formula_path)
        output_paths.append(formula_path)
        counts: dict[str, object] = {}
        for role, table in feature_tables.items():
            score, supported, reject = apply_scigen_formula(table, formula)
            predictions = pd.DataFrame(
                {
                    "material_id": table["material_id"].astype(str).to_numpy(),
                    "partition_role": role,
                    "rli_score": score,
                    "rli_supported": supported,
                    "rli_reject": reject,
                    "rli_decision": np.where(reject, "REJECT", "KEEP"),
                    "formula_sha256": formula_sha256,
                }
            ).sort_values("material_id", kind="stable", ignore_index=True)
            prediction_path = staging / PREDICTION_NAMES[role]
            predictions.to_parquet(prediction_path, index=False)
            output_paths.append(prediction_path)
            counts[role] = {
                "rows": len(predictions),
                "supported": int(predictions["rli_supported"].sum()),
                "rejected": int(predictions["rli_reject"].sum()),
            }
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "mode": "endpoint_free_fixed_rli_formula_and_prediction_freeze",
            "candidate_name": "Rigidity-Load Incompatibility (RLI)",
            "endpoint_payloads_opened": False,
            "discovery_endpoint_opened": False,
            "validation_endpoint_opened": False,
            "replication_endpoint_opened": False,
            "relaxed_structures_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "model_or_proxy_potential_used": False,
            "physical_relaxation_executed": False,
            "discovery_metrics_from_frozen_next87_record": discovery_metrics,
            "counts": counts,
            "inputs_sha256": {
                name: {"path": str(path), "sha256": hashes[name]}
                for name, path in paths.items()
            },
            "executed_source_sha256": {
                "src/next91_scigen_fixed_rli_candidate.py": source_hash
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if any(_sha256_file(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT91 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT91 source changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


__all__ = [
    "PROTOCOL",
    "FORMULA_NAME",
    "MANIFEST_NAME",
    "PREDICTION_NAMES",
    "RLI_THRESHOLD",
    "RLI_TERM_WEIGHTS",
    "build_frozen_rli_formula",
    "freeze_scigen_rli_candidate",
]
