#!/usr/bin/env python3
"""Freeze outcome-informed SSSP bond-valence coherence safeguard for external test."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

import src.next411_same_sign_shell_purity as n411
import src.next525_sssp_standalone_freeze as n525
import src.next527_sssp_one_shot_validation as n527
import src.next528_sssp_one_shot_replication as n528
from src.next87_scigen_sparse_law_search import decision_metrics
from src.next347_periodic_allocation_redistribution_capacity import _sha256_file


PROTOCOL = "2026-08-13-next529-sssp-bvc-development-freeze-v1"
DESIGN_SHA256 = "208f1ee146397520e26a827169f418ebcb04df090c50c54535232eb3a8ec83cb"
SSSP_THRESHOLD = n525.EXPECTED_THRESHOLD
EXPECTED_SCBV_THRESHOLD = 0.33695346214642063
SCBV_FEATURE = "scbv_mismatch_rms"
MINIMUM_PROTECTED_RECALL_LOWER = 0.95
MINIMUM_SAVINGS_LOWER = 0.02
FORMULA_NAME = "NEXT529_FROZEN_SSSP_BVC_FORMULA.json"
EVALUATION_NAME = "NEXT529_SSSP_BVC_SIX_CELL_DEVELOPMENT.json"
TABLE_NAME = "next529_sssp_bvc_development_predictions.parquet"
MANIFEST_NAME = "MANIFEST.json"
CELL_KEYS = (
    "scigen:discovery", "scigen:internal_validation", "scigen:internal_replication",
    "wyformer:discovery", "wyformer:internal_validation", "wyformer:internal_replication",
)
EXPECTED_INPUT_SHA256 = {
    "design": DESIGN_SHA256,
    "next525_manifest": "e15217dafaa1d86dc5a70640dd0ab96a99a9cc0bb04eff44ca850c88e4ff3140",
    "next525_scigen": "bf10f859f34c351c5f7a3ba16ea00d12ea3286751f345e8730364604750d86d0",
    "next525_wyformer": "17233df7aef62873463fadcea034f1de05895c14b1b93e9bd9d733c86f282c66",
    "next527_manifest": "f5c48c14b55713ad385d393523319b83ca29ef6079a689553b6d1fb4dd67fe32",
    "next527_scigen": "259a1bcd977ff7a03faf8b11aaa872345168648ed6bc23199b3cdca43dbd9d2d",
    "next527_wyformer": "a7e397cf24a84b992cd31f20960320e75b37061616030ef3a8c1a24ca46c3e64",
    "next528_manifest": "6adc7f78da729d9ad42d95d95995d17b6d3aa15b4c7315deab805f22abc48f66",
    "next528_scigen": "08b0ec33c675fa90c56795b75ccc6de30658beb546517bc12cce373709a8fd82",
    "next528_wyformer": "61d167717239bca24978d963b7748ce2683016e61549e0e5cbcfba6aea4d70d9",
    "scigen_discovery_features": "7031d86e4fb6e469c674d208f680ace1dbe5e11e45f3d4b2befefd747efdde16",
    "scigen_validation_features": "f266e6143bc23d9e131b5ec788676b520db928aa46a57a1fcba6fd8530a80c8a",
    "scigen_replication_features": "2d420ac76f8b9e1ea6a7908df92a4db1198bc0ef0b2d410875225d51536214b2",
    "wyformer_discovery_features": "c515baec0fccef5bc03c7672f1d4e1aca278f5ed4d7b6f1bf7f66c734e2b87f7",
    "wyformer_validation_features": "26d95746e8aa56087150737a62035f5d4c5ce51b1d2e10424ed6cb267ea1983c",
    "wyformer_replication_features": "7e52e8ab32b380882082ee9a9315c3d18b4d22fe100a83766060b86e50ff19d9",
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def apply_sssp_bvc(
    *,
    sssp: Sequence[float] | np.ndarray,
    sssp_supported: Sequence[bool] | np.ndarray,
    scbv: Sequence[float] | np.ndarray,
    scbv_supported: Sequence[bool] | np.ndarray,
    sssp_threshold: float,
    scbv_threshold: float,
) -> dict[str, np.ndarray]:
    shell = np.asarray(sssp, dtype=float)
    shell_support = np.asarray(sssp_supported, dtype=bool)
    mismatch = np.asarray(scbv, dtype=float)
    mismatch_support = np.asarray(scbv_supported, dtype=bool)
    if (
        shell.ndim != 1
        or shell_support.shape != shell.shape
        or mismatch.shape != shell.shape
        or mismatch_support.shape != shell.shape
        or not math.isfinite(float(sssp_threshold))
        or not 0.0 < float(sssp_threshold) <= 1.0
        or not math.isfinite(float(scbv_threshold))
        or float(scbv_threshold) <= 0.0
        or not np.array_equal(shell_support, shell_support & np.isfinite(shell))
        or not np.array_equal(mismatch_support, mismatch_support & np.isfinite(mismatch))
    ):
        raise ValueError("NEXT529 formula arrays differ")
    supported = shell_support.copy()
    risk = np.full(len(shell), np.nan, dtype=float)
    risk[supported] = 0.0
    both = supported & mismatch_support
    shell_deficit = np.maximum(0.0, float(sssp_threshold) - shell[both]) / float(sssp_threshold)
    mismatch_excess = np.maximum(0.0, mismatch[both] - float(scbv_threshold)) / float(scbv_threshold)
    risk[both] = shell_deficit * mismatch_excess
    reject = (
        both
        & (shell <= float(sssp_threshold))
        & (mismatch >= float(scbv_threshold))
    )
    return {"risk": risk, "supported": supported, "reject": reject}


def formula_payload(*, sssp_threshold: float, scbv_threshold: float) -> dict[str, object]:
    if (
        float(sssp_threshold) != SSSP_THRESHOLD
        or not math.isfinite(float(scbv_threshold))
        or float(scbv_threshold) <= 0.0
    ):
        raise ValueError("NEXT529 formula constants differ")
    return {
        "protocol": PROTOCOL,
        "name": "same_sign_shell_purity_with_bond_valence_coherence_safeguard",
        "abbreviation": "SSSP-BVC",
        "sssp_feature": n411.FEATURE_NAMES[0],
        "sssp_threshold": float(sssp_threshold),
        "scbv_feature": SCBV_FEATURE,
        "scbv_threshold": float(scbv_threshold),
        "risk_formula": (
            "max(0,(t_S-SSSP)/t_S) * max(0,(SCBV_M-t_M)/t_M)"
        ),
        "reject_when": "SSSP_supported and SCBV_supported and SSSP <= t_S and SCBV_M >= t_M",
        "sssp_missing_policy": "ABSTAIN",
        "scbv_missing_policy": "KEEP",
        "input_boundary": ["composition", "one raw initial fully periodic geometry"],
        "dft_inputs": [],
        "learned_model_inputs": [],
        "relaxation_inputs": [],
        "development_cells_are_exposed": True,
        "external_evidence_required": True,
    }


def select_scbv_threshold(
    *,
    cells: Mapping[str, Mapping[str, object]],
    sssp_threshold: float = SSSP_THRESHOLD,
    minimum_protected_recall_lower: float = MINIMUM_PROTECTED_RECALL_LOWER,
    minimum_savings_lower: float = MINIMUM_SAVINGS_LOWER,
) -> dict[str, object]:
    if not cells or float(sssp_threshold) != SSSP_THRESHOLD:
        raise ValueError("NEXT529 cell universe differs")
    checked = {}
    observed = []
    for name, cell in sorted(cells.items()):
        if set(cell) != {"sssp", "sssp_supported", "scbv", "endpoint"}:
            raise ValueError("NEXT529 cell arrays differ")
        shell = np.asarray(cell["sssp"], dtype=float)
        shell_support = np.asarray(cell["sssp_supported"], dtype=bool)
        mismatch = np.asarray(cell["scbv"], dtype=float)
        endpoint = np.asarray(cell["endpoint"], dtype=float)
        mismatch_support = np.isfinite(mismatch)
        if (
            shell.ndim != 1
            or shell_support.shape != shell.shape
            or mismatch.shape != shell.shape
            or endpoint.shape != shell.shape
            or not np.isfinite(endpoint).all()
            or not np.array_equal(shell_support, shell_support & np.isfinite(shell))
            or not (endpoint <= 1.0).any()
            or not (endpoint >= 2.0).any()
        ):
            raise ValueError("NEXT529 cell arrays differ")
        checked[str(name)] = (shell, shell_support, mismatch, mismatch_support, endpoint)
        observed.append(mismatch[mismatch_support])
    candidates = np.unique(np.concatenate(observed))
    candidates = candidates[candidates > 0.0]
    if not len(candidates):
        raise RuntimeError("NEXT529 SCBV threshold population is empty")
    best = None
    for threshold in candidates:
        metrics = {}
        for name, (shell, shell_support, mismatch, mismatch_support, endpoint) in checked.items():
            applied = apply_sssp_bvc(
                sssp=shell,
                sssp_supported=shell_support,
                scbv=mismatch,
                scbv_supported=mismatch_support,
                sssp_threshold=sssp_threshold,
                scbv_threshold=float(threshold),
            )
            metrics[name] = decision_metrics(
                supported=applied["supported"],
                reject=applied["reject"],
                distortion_ratio=endpoint,
            )
        if any(
            float(value["protected_recall_lower"]) < float(minimum_protected_recall_lower)
            or float(value["savings_lower"]) < float(minimum_savings_lower)
            for value in metrics.values()
        ):
            continue
        rank = (
            min(float(value["severe_rejection_precision_lower"]) for value in metrics.values()),
            min(float(value["severe_recall"]) for value in metrics.values()),
            min(float(value["protected_recall_lower"]) for value in metrics.values()),
            float(threshold),
        )
        if best is None or rank > best[0]:
            best = (rank, float(threshold), metrics)
    if best is None:
        raise RuntimeError("NEXT529 no safeguard threshold satisfies all development cells")
    return {
        "threshold": best[1], "rank": list(best[0]),
        "candidate_count": int(len(candidates)), "cell_metrics": best[2],
    }


def _join_cell(base_path: Path, prediction_path: Path) -> pd.DataFrame:
    base = pd.read_parquet(base_path, columns=["material_id", SCBV_FEATURE])
    prediction = pd.read_parquet(prediction_path)
    required = {
        "material_id", "reduced_formula", "endpoint",
        n411.FEATURE_NAMES[0], "supported", "reject",
    }
    if (
        required - set(prediction)
        or base["material_id"].astype(str).duplicated().any()
        or prediction["material_id"].astype(str).duplicated().any()
    ):
        raise ValueError("NEXT529 development input schema differs")
    joined = prediction.merge(base, on="material_id", validate="one_to_one")
    if len(joined) != len(base) or len(joined) != len(prediction):
        raise ValueError("NEXT529 development material identity differs")
    return joined


def freeze_sssp_bvc_development(
    *,
    next525_dir: Path,
    next527_dir: Path,
    next528_dir: Path,
    scigen_feature_dir: Path,
    wyformer_feature_dir: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    freeze, validation, replication = (
        Path(next525_dir).resolve(), Path(next527_dir).resolve(), Path(next528_dir).resolve()
    )
    sf, wf = Path(scigen_feature_dir).resolve(), Path(wyformer_feature_dir).resolve()
    paths = {
        "design": Path(design_path).resolve(),
        "next525_manifest": freeze / n525.MANIFEST_NAME,
        "next525_scigen": freeze / n525.PREDICTION_NAMES["scigen"],
        "next525_wyformer": freeze / n525.PREDICTION_NAMES["wyformer"],
        "next527_manifest": validation / n527.MANIFEST_NAME,
        "next527_scigen": validation / n527.PREDICTION_NAMES["scigen"],
        "next527_wyformer": validation / n527.PREDICTION_NAMES["wyformer"],
        "next528_manifest": replication / n528.MANIFEST_NAME,
        "next528_scigen": replication / n528.PREDICTION_NAMES["scigen"],
        "next528_wyformer": replication / n528.PREDICTION_NAMES["wyformer"],
        "scigen_discovery_features": sf / "features_discovery.parquet",
        "scigen_validation_features": sf / "features_internal_validation.parquet",
        "scigen_replication_features": sf / "features_internal_replication.parquet",
        "wyformer_discovery_features": wf / "wyformer_x0_features_discovery.parquet",
        "wyformer_validation_features": wf / "wyformer_x0_features_internal_validation.parquet",
        "wyformer_replication_features": wf / "wyformer_x0_features_internal_replication.parquet",
    }
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT529 input is missing")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name for name in set(hashes) | set(EXPECTED_INPUT_SHA256)
            if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT529 formal input identity differs: {differing}")
    manifests = {
        "next525": _read_json(paths["next525_manifest"]),
        "next527": _read_json(paths["next527_manifest"]),
        "next528": _read_json(paths["next528_manifest"]),
    }
    if (
        manifests["next525"].get("protocol") != n525.PROTOCOL
        or manifests["next527"].get("passes_all_validation_gates") is not True
        or manifests["next528"].get("passes_all_replication_gates") is not False
        or manifests["next528"].get("internal_replication_endpoint_values_opened") is not True
    ):
        raise ValueError("NEXT529 exposed-development provenance differs")
    cell_paths = {
        "scigen:discovery": (paths["scigen_discovery_features"], paths["next525_scigen"]),
        "scigen:internal_validation": (paths["scigen_validation_features"], paths["next527_scigen"]),
        "scigen:internal_replication": (paths["scigen_replication_features"], paths["next528_scigen"]),
        "wyformer:discovery": (paths["wyformer_discovery_features"], paths["next525_wyformer"]),
        "wyformer:internal_validation": (paths["wyformer_validation_features"], paths["next527_wyformer"]),
        "wyformer:internal_replication": (paths["wyformer_replication_features"], paths["next528_wyformer"]),
    }
    joined = {name: _join_cell(*cell_paths[name]) for name in CELL_KEYS}
    cells = {}
    for name, frame in joined.items():
        shell = pd.to_numeric(frame[n411.FEATURE_NAMES[0]], errors="coerce").to_numpy(float)
        cells[name] = {
            "sssp": shell,
            "sssp_supported": frame["supported"].to_numpy(bool) & np.isfinite(shell),
            "scbv": pd.to_numeric(frame[SCBV_FEATURE], errors="coerce").to_numpy(float),
            "endpoint": pd.to_numeric(frame["endpoint"], errors="coerce").to_numpy(float),
        }
    selected = select_scbv_threshold(cells=cells)
    if float(selected["threshold"]) != EXPECTED_SCBV_THRESHOLD:
        raise RuntimeError("NEXT529 development threshold differs from pre-freeze result")
    formula = formula_payload(
        sssp_threshold=SSSP_THRESHOLD, scbv_threshold=float(selected["threshold"])
    )
    output_frames = []
    cell_diagnostics = {}
    for name, frame in joined.items():
        values = cells[name]
        mismatch_support = np.isfinite(values["scbv"])
        applied = apply_sssp_bvc(
            sssp=values["sssp"], sssp_supported=values["sssp_supported"],
            scbv=values["scbv"], scbv_supported=mismatch_support,
            sssp_threshold=SSSP_THRESHOLD, scbv_threshold=EXPECTED_SCBV_THRESHOLD,
        )
        out = pd.DataFrame({
            "cell": name,
            "material_id": frame["material_id"].astype(str),
            "reduced_formula": frame["reduced_formula"].astype(str),
            "endpoint": values["endpoint"],
            n411.FEATURE_NAMES[0]: values["sssp"],
            SCBV_FEATURE: values["scbv"],
            "sssp_supported": values["sssp_supported"],
            "scbv_supported": mismatch_support,
            "risk_score": applied["risk"],
            "supported": applied["supported"],
            "reject": applied["reject"],
        })
        output_frames.append(out)
        cell_diagnostics[name] = {
            "rows": int(len(out)),
            "sssp_coverage": float(out["sssp_supported"].mean()),
            "scbv_coverage": float(out["scbv_supported"].mean()),
            "metrics": selected["cell_metrics"][name],
        }
    evaluation = {
        "protocol": PROTOCOL,
        "mode": "six_exposed_cells_outcome_informed_development",
        "selection": selected,
        "cells": cell_diagnostics,
        "external_evidence_required": True,
        "scientific_status": "freeze_for_new_disjoint_wbm_external_evaluation",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_hash = _sha256_file(Path(__file__).resolve())
    try:
        formula_path, evaluation_path, table_path = (
            staging / FORMULA_NAME, staging / EVALUATION_NAME, staging / TABLE_NAME
        )
        formula_path.write_bytes(_json_bytes(formula))
        evaluation_path.write_bytes(_json_bytes(evaluation))
        pd.concat(output_frames, ignore_index=True).to_parquet(table_path, index=False)
        outputs = [formula_path, evaluation_path, table_path]
        manifest = {
            "protocol": PROTOCOL,
            "mode": "outcome_informed_development_then_external_freeze",
            "all_six_development_cells_opened": True,
            "wbm_external_endpoint_opened": False,
            "formula_or_threshold_modified_after_freeze": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "model_or_proxy_potential_used": False,
            "physical_relaxation_executed": False,
            "inputs_sha256": {
                name: {"path": str(path), "sha256": hashes[name]}
                for name, path in paths.items()
            },
            "executed_source_sha256": {
                "src/next529_sssp_bvc_development_freeze.py": source_hash
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in outputs},
            "next530_wbm_cohort_authorized": True,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256_file(Path(__file__).resolve()) != source_hash:
            raise RuntimeError("NEXT529 source changed before publication")
        if any(_sha256_file(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT529 input changed before publication")
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next525-dir", type=Path, required=True)
    parser.add_argument("--next527-dir", type=Path, required=True)
    parser.add_argument("--next528-dir", type=Path, required=True)
    parser.add_argument("--scigen-feature-dir", type=Path, required=True)
    parser.add_argument("--wyformer-feature-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = freeze_sssp_bvc_development(
        next525_dir=args.next525_dir, next527_dir=args.next527_dir,
        next528_dir=args.next528_dir, scigen_feature_dir=args.scigen_feature_dir,
        wyformer_feature_dir=args.wyformer_feature_dir, design_path=args.design,
        output_dir=args.output_dir,
    )
    print(json.dumps({"authorized": manifest["next530_wbm_cohort_authorized"]}, indent=2))


__all__ = [
    "EXPECTED_SCBV_THRESHOLD", "FORMULA_NAME", "MANIFEST_NAME", "PROTOCOL",
    "SCBV_FEATURE", "SSSP_THRESHOLD", "apply_sssp_bvc", "formula_payload",
    "freeze_sssp_bvc_development", "select_scbv_threshold",
]


if __name__ == "__main__":
    main()
