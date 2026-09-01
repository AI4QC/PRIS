#!/usr/bin/env python3
"""Open the sealed HEA validation endpoints once and test NEXT557."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

from ase import Atoms
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.next16_elementa_basin_hull import pauling_control
from src.next19_feature_build import _publish_directory_no_replace, _sha256
import src.next551_hea_initial_cohort as n551
import src.next552_hea_analytic_feature_freeze as n552
import src.next553_hea_development_search as n553
import src.next555_hea_extreme_waste_search as n555
import src.next557_hea_packing_deficit_freeze as n557


PROTOCOL = "2026-08-14-next558-hea-packing-deficit-validation-v1"
BOOTSTRAP_DRAWS = 5_000
BOOTSTRAP_SEED = 558_202_608
TABLE_NAME = "next558_hea_packing_deficit_validation.parquet"
RESULT_NAME = "NEXT558_HEA_PACKING_DEFICIT_VALIDATION.json"
MANIFEST_NAME = "MANIFEST.json"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _pauling_payload(item: tuple[str, Atoms]) -> dict[str, object]:
    fid, atoms = item
    result = pauling_control(atoms)
    decisions = [str(result.get(f"pauling_p{number}_decision", "ABSTAIN")) for number in range(2, 6)]
    supported = [decision for decision in decisions if decision != "ABSTAIN"]
    return {
        "fid": fid,
        "pauling_supported_rules": len(supported),
        "pauling_rejected_rules": sum(decision == "REJECT" for decision in supported),
        "pauling_risk": (
            sum(decision == "REJECT" for decision in supported) / len(supported)
            if supported else math.nan
        ),
        "pauling_feature_error": result.get("pauling_feature_error"),
    }


def _pauling_many(items: list[tuple[str, Atoms]], workers: int) -> list[dict[str, object]]:
    if workers == 1:
        iterator = map(_pauling_payload, items)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_pauling_payload, items, chunksize=4)
    rows: list[dict[str, object]] = []
    try:
        for row in iterator:
            rows.append(row)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return rows


def _cluster_bootstrap(
    table: pd.DataFrame, score: np.ndarray, *, draws: int, seed: int
) -> dict[str, object]:
    score = np.asarray(score, dtype=float)
    systems = table["chemical_system"].astype(str).to_numpy()
    unique = np.asarray(sorted(set(systems)), dtype=object)
    groups = {system: np.flatnonzero(systems == system) for system in unique}
    labels = table["dft_waste"].to_numpy(bool)
    severity = table["waste_severity"].to_numpy(float)
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    rhos: list[float] = []
    for _ in range(draws):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([groups[system] for system in chosen])
        index = index[np.isfinite(score[index])]
        if len(index) >= 2 and np.unique(labels[index]).size == 2:
            aucs.append(float(roc_auc_score(labels[index], score[index])))
        rho = float(spearmanr(score[index], severity[index]).statistic) if len(index) >= 2 else math.nan
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
        "clusters": len(unique),
        "roc_auc": interval(aucs),
        "spearman": interval(rhos),
    }


def evaluate_validation(
    table: pd.DataFrame, *, bootstrap_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    required = {
        "fid", "chemical_system", "size_family", n557.FEATURE, "dft_waste",
        "waste_severity", "protected", "pauling_risk",
    }
    if required - set(table) or table["fid"].duplicated().any():
        raise ValueError("NEXT558 validation table differs")
    score = pd.to_numeric(table[n557.FEATURE], errors="coerce").to_numpy(float)
    masks = {
        "overall": np.ones(len(table), dtype=bool),
        "ordered": table["size_family"].astype(str).eq("ordered").to_numpy(),
        "sqs": table["size_family"].astype(str).eq("sqs").to_numpy(),
    }
    candidate = {
        name: n553._score_metrics(table, score, mask) for name, mask in masks.items()
    }
    bootstrap = _cluster_bootstrap(table, score, draws=bootstrap_draws, seed=seed)
    pauling = n553._score_metrics(
        table,
        pd.to_numeric(table["pauling_risk"], errors="coerce").to_numpy(float),
        masks["overall"],
    )
    labels = table["dft_waste"].to_numpy(bool)
    protected = int(table["protected"].sum())
    overall = candidate["overall"]
    top = overall.get("top_15_percent", {})
    top_protected_fraction = top.get("protected", math.inf) / max(top.get("rows", 0), 1)
    clauses = {
        "rows_each_class_at_least_100": int(labels.sum()) >= 100 and int((~labels).sum()) >= 100,
        "protected_at_least_50": protected >= 50,
        "coverage_at_least_0p95": overall.get("coverage", 0.0) >= 0.95,
        "auc_at_least_0p70": overall.get("roc_auc", -math.inf) >= 0.70,
        "auc_cluster_lower_at_least_0p65": bootstrap["roc_auc"]["lower_95"] >= 0.65,
        "ordered_auc_at_least_0p62": candidate["ordered"].get("roc_auc", -math.inf) >= 0.62,
        "sqs_auc_at_least_0p62": candidate["sqs"].get("roc_auc", -math.inf) >= 0.62,
        "spearman_at_least_0p30": overall.get("spearman_severity", -math.inf) >= 0.30,
        "spearman_cluster_lower_at_least_0p20": bootstrap["spearman"]["lower_95"] >= 0.20,
        "top15_lift_at_least_1p50": top.get("lift", -math.inf) >= 1.50,
        "top15_protected_fraction_at_most_0p02": top_protected_fraction <= 0.02,
    }
    return {
        "protocol": PROTOCOL,
        "counts": {
            "rows": len(table),
            "positive": int(labels.sum()),
            "negative": int((~labels).sum()),
            "protected": protected,
            "chemical_systems": int(table["chemical_system"].nunique()),
        },
        "candidate": candidate,
        "cluster_bootstrap": bootstrap,
        "pauling_control": pauling,
        "top15_protected_fraction": float(top_protected_fraction),
        "confirmation_clauses": clauses,
        "confirmation_pass": bool(all(clauses.values())),
        "endpoint_is_development_calibrated": True,
        "scientific_success_claim": bool(all(clauses.values())),
        "claim_boundary": "HEA-domain extreme DFT-waste screening only",
    }


def build_validation(
    *, next551_dir: Path, next552_dir: Path, next557_dir: Path, source_csv: Path,
    design_path: Path, output_dir: Path, workers: int = 8,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, object]:
    root551 = Path(next551_dir).resolve()
    root552 = Path(next552_dir).resolve()
    root557 = Path(next557_dir).resolve()
    source_csv = Path(source_csv).resolve()
    design_path = Path(design_path).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "design": design_path,
        "source_csv": source_csv,
        "next551_manifest": root551 / n551.MANIFEST_NAME,
        "next551_metadata": root551 / n551.METADATA_NAME,
        "next551_geometry": root551 / n551.GEOMETRY_NAME,
        "next552_manifest": root552 / n552.MANIFEST_NAME,
        "next552_table": root552 / n552.TABLE_NAME,
        "next557_manifest": root557 / n557.MANIFEST_NAME,
        "next557_formula": root557 / n557.FORMULA_NAME,
        "next557_source": Path(n557.__file__).resolve(),
        "pauling_source": Path(pauling_control.__code__.co_filename).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT558 input is missing")
    if _sha256(design_path) != n557.DESIGN_SHA256 or _sha256(source_csv) != n551.SOURCE_SHA256:
        raise ValueError("NEXT558 formal source or design differs")
    manifest557 = json.loads(paths["next557_manifest"].read_text())
    outputs557 = manifest557.get("outputs_sha256")
    if (
        manifest557.get("protocol") != n557.PROTOCOL
        or manifest557.get("next558_validation_opening_authorized") is not True
        or manifest557.get("validation_endpoint_values_opened") is not False
        or manifest557.get("validation_final_structures_opened") is not False
        or not isinstance(outputs557, dict)
        or outputs557.get(n557.FORMULA_NAME) != _sha256(paths["next557_formula"])
    ):
        raise ValueError("NEXT558 frozen law identity differs")
    metadata = pd.read_parquet(paths["next551_metadata"])
    validation = metadata.loc[metadata["partition"].astype(str).eq("validation")].copy()
    validation_ids = set(validation["fid"].astype(str))
    if len(validation_ids) != 1_200:
        raise ValueError("NEXT558 validation identity differs")
    initial: dict[str, Atoms] = {}
    with zipfile.ZipFile(paths["next551_geometry"]) as archive:
        for name in archive.namelist():
            fid = Path(name).stem
            if fid in validation_ids:
                initial[fid] = n553._decode_initial(archive.read(name))
    if set(initial) != validation_ids:
        raise ValueError("NEXT558 validation x0 geometries differ")

    pauling_rows = pd.DataFrame(
        _pauling_many([(fid, initial[fid]) for fid in sorted(validation_ids)], workers)
    )
    endpoint_payloads, firewall = n553.extract_authorized_endpoint_payloads(
        source_csv, validation_ids
    )
    if firewall["source_rows_scanned"] != n551.EXPECTED_SOURCE_ROWS:
        raise ValueError("NEXT558 source row count differs")
    endpoint_rows = pd.DataFrame(
        [
            n553._endpoint_row(fid, initial[fid], endpoint_payloads[fid])
            for fid in sorted(validation_ids)
        ]
    )
    endpoints = n555.apply_extreme_waste_endpoint(
        validation.merge(endpoint_rows, on="fid", validate="one_to_one")
    )
    features = pd.read_parquet(paths["next552_table"])
    features = features.loc[features["partition"].astype(str).eq("validation")]
    table = features.merge(
        endpoints[
            ["fid", "e_above_hull", "disp_p90", "cell_logstrain_max", "volume_logchange",
             "dft_waste", "waste_severity", "protected"]
        ],
        on="fid", validate="one_to_one",
    ).merge(pauling_rows, on="fid", validate="one_to_one")
    if len(table) != 1_200:
        raise ValueError("NEXT558 validation join differs")
    result = evaluate_validation(table, bootstrap_draws=bootstrap_draws)
    result["endpoint_firewall"] = firewall

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    try:
        table_path = staging / TABLE_NAME
        result_path = staging / RESULT_NAME
        table.to_parquet(table_path, index=False)
        result_path.write_bytes(_json_bytes(result))
        outputs_out = {TABLE_NAME: _sha256(table_path), RESULT_NAME: _sha256(result_path)}
        manifest = {
            "protocol": PROTOCOL,
            "confirmation_pass": result["confirmation_pass"],
            "inputs_sha256": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in paths.items()
            },
            "outputs_sha256": outputs_out,
            "executed_source_sha256": {
                "src/next558_hea_packing_deficit_validation.py": source_hash
            },
            "validation_endpoint_values_opened": True,
            "validation_final_structures_opened": True,
            "dft_values_used_by_executable_formula": False,
            "model_or_proxy_potential_used": False,
            "coordinates_or_cell_modified_by_formula": False,
            "scientific_improvement_claim": result["confirmation_pass"],
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT558 source changed before publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next551-dir", required=True, type=Path)
    parser.add_argument("--next552-dir", required=True, type=Path)
    parser.add_argument("--next557-dir", required=True, type=Path)
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--design-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args(argv)
    manifest = build_validation(
        next551_dir=args.next551_dir,
        next552_dir=args.next552_dir,
        next557_dir=args.next557_dir,
        source_csv=args.source_csv,
        design_path=args.design_path,
        output_dir=args.output_dir,
        workers=args.workers,
        bootstrap_draws=args.bootstrap_draws,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_validation", "evaluate_validation"]
