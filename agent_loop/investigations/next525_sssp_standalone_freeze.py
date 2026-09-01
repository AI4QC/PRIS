#!/usr/bin/env python3
"""Freeze a standalone SSSP prescreen using discovery endpoints only."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

import src.next411_same_sign_shell_purity as n411
import src.next412_sssp_formal_build as n412
from src.next23_evaluate import _roc_auc
from src.next87_scigen_sparse_law_search import (
    _pauling_baseline,
    assign_group_folds,
    auc_diagnostics,
    decision_metrics,
)
from src.next95_wyformer_sparse_law_search import _endpoint_numeric
from src.next347_periodic_allocation_redistribution_capacity import _sha256_file


PROTOCOL = "2026-08-13-next525-sssp-standalone-discovery-freeze-v1"
DESIGN_SHA256 = "99cbe8c820809a1eb412e8c8a5b61a0c3f70b5213f000ff663933e13fdac5df2"
FEATURE = n411.FEATURE_NAMES[0]
MINIMUM_PROTECTED_RECALL_LOWER = 0.95
MINIMUM_SAVINGS_LOWER = 0.02
EXPECTED_THRESHOLD = 0.5231805323
FORMULA_NAME = "NEXT525_FROZEN_SSSP_FORMULA.json"
EVALUATION_NAME = "NEXT525_SSSP_DISCOVERY_EVALUATION.json"
MANIFEST_NAME = "MANIFEST.json"
PREDICTION_NAMES = {
    "scigen": "next525_scigen_discovery_predictions.parquet",
    "wyformer": "next525_wyformer_discovery_predictions.parquet",
}
EXPECTED_INPUT_SHA256 = {
    "design": DESIGN_SHA256,
    "next411_source": "172543534328a387b7d2b12ffd6cad919793ace56ec1124dd6e228f96d8cc9a4",
    "next412_source": "b4ae4016a92217237d8eaccd2449fc7bfcee2193d31f615f0a79fd49f9fedaca",
    "next412_manifest": "cd0a222c4052446aadfdee2e4d04a4937c075d1a2062e32694cac3415cdaa02a",
    "next412_catalogue": "e3e7da654f276a21c64802696d436b81a153fe1eac30c9dc8b14fd44cb934053",
    "next412_scigen": "7f23aa49fdc6c7f048b8f48a14c1dd5402af78779aa00ec3e3c0f65ed02970a6",
    "next412_wyformer": "3af649c617dac28f5db687a32d736c043dbb1b5ba23206237d97e822fbcc0280",
    "scigen_feature_manifest": "8dcb8118f85ee4a3acbf0905f01c2b173d58742a1e16dcd6004adbbbedcf63cc",
    "scigen_features": "7031d86e4fb6e469c674d208f680ace1dbe5e11e45f3d4b2befefd747efdde16",
    "wyformer_feature_manifest": "fb66f7c5caade419a46b9a3fa6fef1bc5b3afa3eebeb95a4bc53baddabc0f659",
    "wyformer_features": "c515baec0fccef5bc03c7672f1d4e1aca278f5ed4d7b6f1bf7f66c734e2b87f7",
    "scigen_endpoint_manifest": "35792117310f04daa8c383bddb5d4012084d47c7d904706d86cbe33e0a55a6ea",
    "scigen_endpoint": "f86cff6f5e9124ee82aae13911ffe55a125c6fe111fc1f64122a610febf67958",
    "wyformer_endpoint_manifest": "3cf3a196ab497851131d5d1604f272d15121c19a943eeb3103a268e7e8b332f5",
    "wyformer_endpoint": "f39836e62a1da03ed823479e87d6f75fc0d01da60a8c0a2faa696638cc2fb9d7",
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def formula_payload(*, threshold: float) -> dict[str, object]:
    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("NEXT525 threshold differs")
    return {
        "protocol": PROTOCOL,
        "kind": "single_zero_dft_same_sign_shell_purity_threshold",
        "feature": FEATURE,
        "feature_formula": (
            "q10_inverted_cdf(min(1, nearest same-sign periodic distance / "
            "opposite-sign radical-Voronoi shell radius))"
        ),
        "direction": "protected_high",
        "risk_score": "-SSSP",
        "threshold": float(threshold),
        "reject_when": "supported and SSSP <= threshold",
        "missing_policy": "ABSTAIN",
        "input_boundary": ["composition", "one raw initial fully periodic geometry"],
        "dft_inputs": [],
        "learned_model_inputs": [],
        "relaxation_inputs": [],
        "formula_or_direction_changed_from_next411": False,
    }


def select_shared_threshold(
    *,
    sources: Mapping[str, Mapping[str, object]],
    minimum_protected_recall_lower: float = MINIMUM_PROTECTED_RECALL_LOWER,
    minimum_savings_lower: float = MINIMUM_SAVINGS_LOWER,
) -> dict[str, object]:
    """Select one observed SSSP cutoff using the frozen cross-source rank."""

    if (
        not sources
        or not math.isfinite(float(minimum_protected_recall_lower))
        or not 0.0 <= float(minimum_protected_recall_lower) <= 1.0
        or not math.isfinite(float(minimum_savings_lower))
        or not 0.0 <= float(minimum_savings_lower) <= 1.0
    ):
        raise ValueError("NEXT525 threshold gates differ")
    checked: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    observed: list[np.ndarray] = []
    for source, values in sorted(sources.items()):
        if set(values) != {"sssp", "supported", "endpoint"}:
            raise ValueError("NEXT525 source arrays differ")
        sssp = np.asarray(values["sssp"], dtype=float)
        supported = np.asarray(values["supported"], dtype=bool)
        endpoint = np.asarray(values["endpoint"], dtype=float)
        if (
            sssp.ndim != 1
            or supported.shape != sssp.shape
            or endpoint.shape != sssp.shape
            or not np.isfinite(endpoint).all()
            or not np.array_equal(supported, supported & np.isfinite(sssp))
            or not (endpoint <= 1.0).any()
            or not (endpoint >= 2.0).any()
        ):
            raise ValueError("NEXT525 source arrays differ")
        checked[str(source)] = (sssp, supported, endpoint)
        observed.append(sssp[supported])
    thresholds = np.unique(np.concatenate(observed))
    best: tuple[tuple[float, ...], float, dict[str, dict[str, object]]] | None = None
    for threshold in thresholds:
        metrics = {
            source: decision_metrics(
                supported=supported,
                reject=supported & (sssp <= float(threshold)),
                distortion_ratio=endpoint,
            )
            for source, (sssp, supported, endpoint) in checked.items()
        }
        if any(
            float(value["protected_recall_lower"])
            < float(minimum_protected_recall_lower)
            or float(value["savings_lower"]) < float(minimum_savings_lower)
            for value in metrics.values()
        ):
            continue
        rank = (
            min(float(value["severe_rejection_precision_lower"]) for value in metrics.values()),
            min(float(value["severe_recall"]) for value in metrics.values()),
            min(float(value["savings_lower"]) for value in metrics.values()),
            float(threshold),
        )
        if best is None or rank > best[0]:
            best = (rank, float(threshold), metrics)
    if best is None:
        raise RuntimeError("NEXT525 no shared threshold satisfies the frozen gates")
    return {
        "threshold": best[1],
        "rank": list(best[0]),
        "source_metrics": best[2],
        "candidate_count": int(len(thresholds)),
    }


def _fold_auc(frame: pd.DataFrame, *, score: np.ndarray, endpoint: np.ndarray, supported: np.ndarray):
    folds = assign_group_folds(frame["reduced_formula"].astype(str).to_numpy())
    records = []
    values = []
    for fold in range(5):
        mask = (folds == fold) & supported & ((endpoint <= 1.0) | (endpoint >= 2.0))
        auc = _roc_auc(score[mask], endpoint[mask] >= 2.0) if mask.any() else None
        records.append(
            {
                "fold": fold,
                "rows": int(mask.sum()),
                "protected": int((mask & (endpoint <= 1.0)).sum()),
                "severe": int((mask & (endpoint >= 2.0)).sum()),
                "auc": auc,
            }
        )
        if auc is not None:
            values.append(float(auc))
    return {
        "records": records,
        "macro_auc": float(np.mean(values)) if len(values) == 5 else None,
        "worst_auc": float(np.min(values)) if len(values) == 5 else None,
    }


def _joined_source(
    *, source: str, base: pd.DataFrame, sssp: pd.DataFrame, endpoint: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, str]:
    if source == "scigen":
        endpoint_column, lattice_column = "distortion_ratio", "lattice_class"
    elif source == "wyformer":
        endpoint_column, lattice_column = "endpoint_stratum", "crystal_system"
    else:
        raise ValueError("NEXT525 source differs")
    required_base = {
        "material_id", "reduced_formula", lattice_column, "pauling_p2_p5_decision"
    }
    required_sssp = {"material_id", FEATURE, "sssp_supported"}
    if (
        required_base - set(base)
        or required_sssp - set(sssp)
        or {"material_id", endpoint_column} - set(endpoint)
        or any(
            frame["material_id"].astype(str).duplicated().any()
            for frame in (base, sssp, endpoint)
        )
    ):
        raise ValueError(f"NEXT525 {source} discovery schema differs")
    joined = (
        base.merge(sssp[list(required_sssp)], on="material_id", validate="one_to_one")
        .merge(endpoint[["material_id", endpoint_column]], on="material_id", validate="one_to_one")
    )
    if len(joined) != len(base) or len(joined) != len(sssp) or len(joined) != len(endpoint):
        raise ValueError(f"NEXT525 {source} discovery identity differs")
    outcome = (
        pd.to_numeric(joined[endpoint_column], errors="coerce").to_numpy(float)
        if source == "scigen"
        else _endpoint_numeric(joined[endpoint_column])
    )
    if not np.isfinite(outcome).all():
        raise ValueError(f"NEXT525 {source} endpoint differs")
    return joined, outcome, lattice_column


def freeze_sssp_standalone(
    *,
    next412_dir: Path,
    scigen_feature_dir: Path,
    wyformer_feature_dir: Path,
    scigen_endpoint_dir: Path,
    wyformer_endpoint_dir: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Calibrate once on discovery and publish an immutable standalone law."""

    roots = {
        "next412": Path(next412_dir).resolve(),
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_endpoint_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_endpoint_dir).resolve(),
    }
    paths = {
        "design": Path(design_path).resolve(),
        "next411_source": Path(n411.__file__).resolve(),
        "next412_source": Path(n412.__file__).resolve(),
        "next412_manifest": roots["next412"] / n412.MANIFEST_NAME,
        "next412_catalogue": roots["next412"] / n412.CATALOGUE_NAME,
        "next412_scigen": roots["next412"] / n412.FEATURE_FILES["scigen"],
        "next412_wyformer": roots["next412"] / n412.FEATURE_FILES["wyformer"],
        "scigen_feature_manifest": roots["scigen_features"] / "MANIFEST.json",
        "scigen_features": roots["scigen_features"] / "features_discovery.parquet",
        "wyformer_feature_manifest": roots["wyformer_features"] / "MANIFEST.json",
        "wyformer_features": roots["wyformer_features"] / "wyformer_x0_features_discovery.parquet",
        "scigen_endpoint_manifest": roots["scigen_endpoint"] / "MANIFEST.json",
        "scigen_endpoint": roots["scigen_endpoint"] / "scigen_dft_distortion_endpoints.parquet",
        "wyformer_endpoint_manifest": roots["wyformer_endpoint"] / "MANIFEST.json",
        "wyformer_endpoint": roots["wyformer_endpoint"] / "wyformer_dft_screening_endpoints.parquet",
    }
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT525 input is missing")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name for name in set(hashes) | set(EXPECTED_INPUT_SHA256)
            if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT525 formal input identity differs: {differing}")
    manifests = {
        name: _read_json(paths[name])
        for name in (
            "next412_manifest", "scigen_feature_manifest", "wyformer_feature_manifest",
            "scigen_endpoint_manifest", "wyformer_endpoint_manifest",
        )
    }
    if (
        manifests["next412_manifest"].get("labels_opened") is not False
        or manifests["scigen_feature_manifest"].get("labels_opened") is not False
        or manifests["wyformer_feature_manifest"].get("labels_opened") is not False
        or manifests["scigen_endpoint_manifest"].get("partition_role") != "discovery"
        or manifests["wyformer_endpoint_manifest"].get("partition_role") != "discovery"
    ):
        raise ValueError("NEXT525 discovery provenance differs")
    tables = {
        "scigen": _joined_source(
            source="scigen",
            base=pd.read_parquet(paths["scigen_features"]),
            sssp=pd.read_parquet(paths["next412_scigen"]),
            endpoint=pd.read_parquet(paths["scigen_endpoint"]),
        ),
        "wyformer": _joined_source(
            source="wyformer",
            base=pd.read_parquet(paths["wyformer_features"]),
            sssp=pd.read_parquet(paths["next412_wyformer"]),
            endpoint=pd.read_parquet(paths["wyformer_endpoint"]),
        ),
    }
    arrays = {}
    for source, (frame, endpoint, _) in tables.items():
        sssp = pd.to_numeric(frame[FEATURE], errors="coerce").to_numpy(float)
        supported = frame["sssp_supported"].fillna(False).to_numpy(bool) & np.isfinite(sssp)
        arrays[source] = {"sssp": sssp, "supported": supported, "endpoint": endpoint}
    selected = select_shared_threshold(sources=arrays)
    if float(selected["threshold"]) != EXPECTED_THRESHOLD:
        raise RuntimeError("NEXT525 discovery threshold differs from the pre-freeze result")
    formula = formula_payload(threshold=float(selected["threshold"]))
    evaluation: dict[str, object] = {
        "protocol": PROTOCOL,
        "partition_role": "discovery",
        "threshold_selection": selected,
        "sources": {},
        "passes_discovery_feasibility": True,
        "validation_endpoint_opened": False,
        "replication_endpoint_opened": False,
    }
    predictions = {}
    for source, (frame, endpoint, lattice_column) in tables.items():
        sssp, supported = arrays[source]["sssp"], arrays[source]["supported"]
        score = -sssp
        reject = supported & (sssp <= float(selected["threshold"]))
        metrics = decision_metrics(supported=supported, reject=reject, distortion_ratio=endpoint)
        diagnostics = auc_diagnostics(
            score=score,
            supported=supported,
            distortion_ratio=endpoint,
            lattice_class=frame[lattice_column].astype(str).to_numpy(),
        )
        evaluation["sources"][source] = {
            "metrics": metrics,
            "auc_diagnostics": diagnostics,
            "formula_fold_auc": _fold_auc(
                frame, score=score, endpoint=endpoint, supported=supported
            ),
            "pauling": _pauling_baseline(frame, endpoint),
        }
        predictions[source] = pd.DataFrame(
            {
                "material_id": frame["material_id"].astype(str),
                "reduced_formula": frame["reduced_formula"].astype(str),
                "partition_role": "discovery",
                "endpoint": endpoint,
                FEATURE: sssp,
                "supported": supported,
                "risk_score": score,
                "reject": reject,
                "pauling_p2_p5_decision": frame["pauling_p2_p5_decision"].astype(str),
            }
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_hash = _sha256_file(Path(__file__).resolve())
    try:
        (staging / FORMULA_NAME).write_bytes(_json_bytes(formula))
        (staging / EVALUATION_NAME).write_bytes(_json_bytes(evaluation))
        for source, frame in predictions.items():
            frame.to_parquet(staging / PREDICTION_NAMES[source], index=False)
        output_paths = [
            staging / FORMULA_NAME,
            staging / EVALUATION_NAME,
            *(staging / name for name in PREDICTION_NAMES.values()),
        ]
        manifest = {
            "protocol": PROTOCOL,
            "mode": "discovery_only_shared_threshold_freeze",
            "formula_or_direction_changed_from_next411": False,
            "validation_endpoint_opened": False,
            "replication_endpoint_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "model_or_proxy_potential_used": False,
            "physical_relaxation_executed": False,
            "threshold": float(selected["threshold"]),
            "inputs_sha256": {
                name: {"path": str(path), "sha256": hashes[name]}
                for name, path in paths.items()
            },
            "executed_source_sha256": {
                "src/next525_sssp_standalone_freeze.py": source_hash
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
            "next526_holdout_feature_freeze_authorized": True,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256_file(Path(__file__).resolve()) != source_hash:
            raise RuntimeError("NEXT525 source changed before publication")
        if any(_sha256_file(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT525 input changed before publication")
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next412-dir", type=Path, required=True)
    parser.add_argument("--scigen-feature-dir", type=Path, required=True)
    parser.add_argument("--wyformer-feature-dir", type=Path, required=True)
    parser.add_argument("--scigen-endpoint-dir", type=Path, required=True)
    parser.add_argument("--wyformer-endpoint-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = freeze_sssp_standalone(
        next412_dir=args.next412_dir,
        scigen_feature_dir=args.scigen_feature_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        scigen_endpoint_dir=args.scigen_endpoint_dir,
        wyformer_endpoint_dir=args.wyformer_endpoint_dir,
        design_path=args.design,
        output_dir=args.output_dir,
    )
    print(json.dumps({"threshold": manifest["threshold"], "authorized": True}, indent=2))


__all__ = [
    "EXPECTED_THRESHOLD",
    "FORMULA_NAME",
    "MANIFEST_NAME",
    "PROTOCOL",
    "formula_payload",
    "freeze_sssp_standalone",
    "select_shared_threshold",
]


if __name__ == "__main__":
    main()
