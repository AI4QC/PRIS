#!/usr/bin/env python3
"""Retrospectively evaluate frozen NEXT549 TCSE scores on opened OMC25 endpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.next19_feature_build import _publish_directory_no_replace, _sha256
import src.next549_omc25_two_sided_contact_freeze as n549


PROTOCOL = "2026-08-13-next550-omc25-two-sided-contact-retrospective-v1"
ENERGY_POSITIVE_MIN = 0.04
PROTECTED_MAX = 0.01
TOP_FRACTION = 0.15
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 550_202_608
TABLE_NAME = "next550_omc25_tcse_joined.parquet"
RESULT_NAME = "NEXT550_OMC25_TCSE_RETROSPECTIVE.json"
MANIFEST_NAME = "MANIFEST.json"
SCORE_COLUMNS = (
    "tcse_risk",
    "risk_low_q10",
    "risk_high_q50",
    "next31_risk_score",
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _score_metrics(table: pd.DataFrame, column: str) -> dict[str, object]:
    score = pd.to_numeric(table[column], errors="coerce").to_numpy(float)
    energy = pd.to_numeric(table["energy_drop_pa"], errors="coerce").to_numpy(float)
    supported = np.isfinite(score) & np.isfinite(energy)
    values, response = score[supported], energy[supported]
    labels = response >= ENERGY_POSITIVE_MIN
    if supported.sum() < 2 or np.unique(labels).size != 2:
        return {"supported": int(supported.sum()), "coverage": float(supported.mean())}
    ids = table.loc[supported, "material_id"].astype(str).to_numpy()
    order = np.lexsort((ids, -values))
    top_n = max(1, math.ceil(TOP_FRACTION * len(order)))
    top = order[:top_n]
    protected = response <= PROTECTED_MAX
    prevalence = float(labels.mean())
    return {
        "supported": int(supported.sum()),
        "coverage": float(supported.mean()),
        "roc_auc": float(roc_auc_score(labels, values)),
        "spearman_energy_drop": float(spearmanr(values, response).statistic),
        "prevalence": prevalence,
        "top_15_percent": {
            "rows": top_n,
            "large_response": int(labels[top].sum()),
            "precision": float(labels[top].mean()),
            "recall": float(labels[top].sum() / labels.sum()),
            "protected": int(protected[top].sum()),
        },
    }


def _cluster_bootstrap(
    table: pd.DataFrame, *, draws: int, seed: int
) -> dict[str, object]:
    if type(draws) is not int or draws < 1:
        raise ValueError("NEXT550 bootstrap draws differ")
    ids = table["material_id"].astype(str).to_numpy()
    clusters = np.asarray([value.split("-", 1)[0] for value in ids], dtype=object)
    unique = np.asarray(sorted(set(clusters)), dtype=object)
    members = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    score = pd.to_numeric(table["tcse_risk"], errors="coerce").to_numpy(float)
    energy = pd.to_numeric(table["energy_drop_pa"], errors="coerce").to_numpy(float)
    finite = np.isfinite(score) & np.isfinite(energy)
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    rhos: list[float] = []
    for _ in range(draws):
        selected = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([members[cluster] for cluster in selected])
        index = index[finite[index]]
        labels = energy[index] >= ENERGY_POSITIVE_MIN
        if len(index) >= 2 and np.unique(labels).size == 2:
            aucs.append(float(roc_auc_score(labels, score[index])))
        rho = float(spearmanr(score[index], energy[index]).statistic) if len(index) >= 2 else math.nan
        if math.isfinite(rho):
            rhos.append(rho)

    def interval(values: list[float]) -> dict[str, object]:
        array = np.asarray(values, dtype=float)
        return {
            "valid": len(values),
            "lower_95": float(np.quantile(array, 0.025)) if len(array) else math.nan,
            "median": float(np.quantile(array, 0.5)) if len(array) else math.nan,
            "upper_95": float(np.quantile(array, 0.975)) if len(array) else math.nan,
        }

    return {
        "draws": draws,
        "seed": seed,
        "cluster": "csd_refcode parsed from material_id",
        "clusters": len(unique),
        "roc_auc": interval(aucs),
        "spearman": interval(rhos),
    }


def evaluate_scores(
    table: pd.DataFrame, *, bootstrap_draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED
) -> dict[str, object]:
    required = {"material_id", "source_shard", "energy_drop_pa", *SCORE_COLUMNS}
    if required - set(table) or table["material_id"].duplicated().any():
        raise ValueError("NEXT550 joined table schema differs")
    energy = pd.to_numeric(table["energy_drop_pa"], errors="coerce").to_numpy(float)
    if not np.isfinite(energy).all():
        raise ValueError("NEXT550 energy response contains nonfinite values")
    scores = {column: _score_metrics(table, column) for column in SCORE_COLUMNS}
    bootstrap = _cluster_bootstrap(table, draws=bootstrap_draws, seed=seed)
    tcse = scores["tcse_risk"]
    prevalence = tcse["prevalence"]
    component_auc = max(
        scores["risk_low_q10"]["roc_auc"], scores["risk_high_q50"]["roc_auc"]
    )
    clauses = {
        "coverage_at_least_0p99": tcse["coverage"] >= 0.99,
        "auc_at_least_0p70": tcse["roc_auc"] >= 0.70,
        "auc_cluster_lower_at_least_0p65": bootstrap["roc_auc"]["lower_95"] >= 0.65,
        "spearman_at_least_0p25": tcse["spearman_energy_drop"] >= 0.25,
        "spearman_cluster_lower_at_least_0p20": bootstrap["spearman"]["lower_95"] >= 0.20,
        "top15_precision_at_least_twice_prevalence": (
            tcse["top_15_percent"]["precision"] >= 2.0 * prevalence
        ),
        "top15_protected_is_zero": tcse["top_15_percent"]["protected"] == 0,
        "auc_component_margin_at_least_0p01": tcse["roc_auc"] - component_auc >= 0.01,
    }
    shard_metrics: dict[str, object] = {}
    for shard in sorted(table["source_shard"].astype(str).unique()):
        subset = table.loc[table["source_shard"].astype(str).eq(shard)]
        shard_metrics[shard] = _score_metrics(subset, "tcse_risk")
    return {
        "protocol": PROTOCOL,
        "thresholds": {
            "large_response_min_ev_per_atom": ENERGY_POSITIVE_MIN,
            "protected_max_ev_per_atom": PROTECTED_MAX,
            "top_fraction": TOP_FRACTION,
        },
        "counts": {
            "rows": len(table),
            "large_response": int((energy >= ENERGY_POSITIVE_MIN).sum()),
            "protected": int((energy <= PROTECTED_MAX).sum()),
            "source_shards": int(table["source_shard"].nunique()),
        },
        "scores": scores,
        "tcse_cluster_bootstrap": bootstrap,
        "tcse_source_shards": shard_metrics,
        "diagnostic_support_clauses": clauses,
        "diagnostic_support_pass": bool(all(clauses.values())),
        "retrospective_only": True,
        "historical_omc25_endpoint_contamination_exists": True,
        "prospective_or_unseen_confirmation": False,
        "scientific_success_claim": False,
    }


def build_evaluation(
    *,
    next549_dir: Path,
    endpoint_tables: list[Path],
    design_path: Path,
    output_dir: Path,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    upstream = Path(next549_dir).resolve()
    design_path = Path(design_path).resolve()
    endpoint_tables = [Path(path).resolve() for path in endpoint_tables]
    target = Path(output_dir).resolve()
    paths = {
        "manifest": upstream / n549.MANIFEST_NAME,
        "predictions": upstream / n549.TABLE_NAME,
        "formula": upstream / n549.FORMULA_NAME,
        "design": design_path,
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if not endpoint_tables or any(not path.is_file() for path in [*paths.values(), *endpoint_tables]):
        raise FileNotFoundError("NEXT550 input is missing")
    if require_formal_inputs and _sha256(design_path) != n549.DESIGN_SHA256:
        raise ValueError("NEXT550 design identity differs")
    manifest = json.loads(paths["manifest"].read_text())
    outputs = manifest.get("outputs_sha256")
    if (
        manifest.get("protocol") != n549.PROTOCOL
        or manifest.get("endpoint_values_opened_by_next549") is not False
        or manifest.get("dft_energy_force_stress_used") is not False
        or manifest.get("retrospective_transfer_only") is not True
        or not isinstance(outputs, dict)
        or outputs.get(n549.TABLE_NAME) != _sha256(paths["predictions"])
        or outputs.get(n549.FORMULA_NAME) != _sha256(paths["formula"])
    ):
        raise ValueError("NEXT550 prediction freeze identity differs")
    predictions = pd.read_parquet(paths["predictions"])
    endpoint_frames: list[pd.DataFrame] = []
    for path in endpoint_tables:
        frame = pd.read_parquet(path)
        required = {"material_id", "source_shard", "energy_drop_pa"}
        if required - set(frame):
            raise ValueError(f"NEXT550 endpoint schema differs: {path}")
        endpoint_frames.append(frame[["material_id", "source_shard", "energy_drop_pa"]].copy())
    endpoints = pd.concat(endpoint_frames, ignore_index=True)
    if endpoints["material_id"].duplicated().any():
        raise ValueError("NEXT550 endpoint identities overlap")
    joined = predictions.merge(
        endpoints, on=["material_id", "source_shard"], how="inner", validate="one_to_one"
    )
    if len(joined) != len(predictions) or len(joined) != len(endpoints):
        raise ValueError("NEXT550 prediction-endpoint coverage differs")
    joined = joined.sort_values(["source_shard", "material_id"], kind="mergesort").reset_index(drop=True)
    result = evaluate_scores(joined, bootstrap_draws=bootstrap_draws, seed=seed)

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    try:
        table_path = staging / TABLE_NAME
        result_path = staging / RESULT_NAME
        joined.to_parquet(table_path, index=False)
        result_path.write_bytes(_json_bytes(result))
        outputs_out = {TABLE_NAME: _sha256(table_path), RESULT_NAME: _sha256(result_path)}
        manifest_out = {
            "protocol": PROTOCOL,
            "design_sha256": _sha256(design_path),
            "next549_inputs_sha256": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in paths.items()
            },
            "endpoint_inputs_sha256": [
                {"path": str(path), "sha256": _sha256(path)} for path in endpoint_tables
            ],
            "outputs_sha256": outputs_out,
            "executed_source_sha256": {
                "src/next550_omc25_two_sided_contact_evaluate.py": source_hash
            },
            "endpoint_values_opened_by_next550": True,
            "endpoint_values_used_by_executable_formula": False,
            "historical_omc25_endpoint_contamination_exists": True,
            "retrospective_transfer_only": True,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest_out))
        if _sha256(source_path) != source_hash or _sha256(design_path) != n549.DESIGN_SHA256:
            raise RuntimeError("NEXT550 source or design changed before publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next549-dir", required=True, type=Path)
    parser.add_argument("--endpoint-table", action="append", required=True, type=Path)
    parser.add_argument("--design-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_evaluation(
        next549_dir=args.next549_dir,
        endpoint_tables=args.endpoint_table,
        design_path=args.design_path,
        output_dir=args.output_dir,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_evaluation", "evaluate_scores"]
