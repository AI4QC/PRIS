#!/usr/bin/env python3
"""Open Li--Si DFT endpoints once and test the frozen MUPR screening law."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

from src.next19_feature_build import _publish_directory_no_replace, _sha256
import src.next541_jarvis_cdvae_initial_prediction_freeze as n541
import src.next542_jarvis_cdvae_dft_response as n542
import src.next543_lisi_random_relaxation_initial_cohort as n543
import src.next544_lisi_random_relaxation_prediction_freeze as n544


PROTOCOL = "2026-08-13-next545-lisi-random-relaxation-confirmation-v1"
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 545_202_608
MINIMUM_ENDPOINT_COVERAGE = 0.95
MINIMUM_CLASS_COUNT = 30
MINIMUM_PREFIX_CLASS_COUNT = 5
MINIMUM_MUPR_COVERAGE = 0.95
MINIMUM_AUC = 0.65
MINIMUM_AUC_LOWER = 0.55
MINIMUM_PAULING_MARGIN = 0.05
MINIMUM_COMPONENT_MARGIN = 0.02
MINIMUM_TOP_PRECISION = 0.60
MINIMUM_TOP_PRECISION_LOWER = 0.42
MINIMUM_TOP_RECALL = 0.20
MINIMUM_BOTTOM_NONWASTE = 0.75
MINIMUM_SPEARMAN = 0.35
MINIMUM_SPEARMAN_LOWER = 0.20
MINIMUM_PREFIX_AUC = 0.55
MINIMUM_PREFIXES_PASSING = 3
TABLE_NAME = "next545_lisi_rr_confirmation_endpoints.parquet"
RESULT_NAME = "NEXT545_LISI_RR_CONFIRMATION.json"
MANIFEST_NAME = "MANIFEST.json"
EXPECTED_INPUT_SHA256 = {
    "design": n543.DESIGN_SHA256,
    "next543_manifest": "5521ee2ad498f9529db0f88d82242efa46f930c5351fd3bf1fc0e069bbb51cf1",
    "next543_inventory": "7de250de1dba4b22670949e08049a5483c1f00f12a564c5790bf6f7702aa3eaf",
    "next543_cohort": "9022a3a0a81e3cca43f237f3b217391b892ba66781a83c09dc5055c1a02964bc",
    "next543_geometry": "1c249fbc66267c964eb840b06886acd5f21a4884b05533469567866753dd843e",
    "next543_source": "79f2d7e1b70677856f5596f1b28908cff53b3c037d27e36545abb64115d55cca",
    "next544_manifest": "963188b96d178b1d1e473f3d132b42c7b3a2387a1885cd1d0388d83ff2d812ed",
    "next544_table": "08adf0f0be41b21b79b64e1f951dc829695c5e6ce6a1cccc12f918fdb0740f4a",
    "next544_formula": "3a13c18b97838c038833c6f037f060d87a118fba9ee28ff5b28a9b9956f9f25b",
    "next544_source": "ad67d5065bbf14240233cd6de87448b1fd95d2d6d97f6020834e2c8fabbe699a",
}
_ENERGY = re.compile(r"^\s*(-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)\s+eV\s*$")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _summary_object_name(prefix: str) -> str:
    return f"{n543.ROOT_PREFIX}/{prefix}/summary.txt"


def _download_summary(prefix: str, source_dir: Path) -> dict[str, object]:
    object_name = _summary_object_name(prefix)
    target = source_dir / prefix / "summary.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        url = "https://storage.googleapis.com/" + n543.BUCKET + "/" + urllib.parse.quote(
            object_name, safe="/"
        )
        temporary = target.with_name(target.name + ".partial")
        if temporary.exists():
            temporary.unlink()
        with urllib.request.urlopen(url, timeout=300) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        os.replace(temporary, target)
    return {
        "prefix": prefix,
        "object_name": object_name,
        "local_path": str(target),
        "size": target.stat().st_size,
        "sha256": _sha256(target),
    }


def _parse_energy(value: str) -> float:
    if value.strip() == "None":
        return math.nan
    match = _ENERGY.fullmatch(value)
    return float(match.group(1)) if match else math.nan


def parse_summary_rows(path: Path, *, prefix: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for values in csv.reader(handle, skipinitialspace=True):
            if len(values) != 9:
                raise ValueError(f"NEXT545 summary field count differs for {prefix}")
            summary_id = values[0].strip()
            if not summary_id.startswith(prefix + "_"):
                raise ValueError(f"NEXT545 summary identifier differs for {prefix}")
            suffix = summary_id[len(prefix) + 1 :]
            rows.append(
                {
                    "summary_id": summary_id,
                    "prefix": prefix,
                    "object_suffix": suffix,
                    "calculation_message": values[1].strip(),
                    "final_energy_total_ev": _parse_energy(values[2]),
                    "initial_spacegroup": values[3].strip(),
                    "final_spacegroup": values[4].strip(),
                    "summary_initial_volume": float(values[5]) if values[5].strip() != "None" else math.nan,
                    "summary_final_volume": float(values[6]) if values[6].strip() != "None" else math.nan,
                    "summary_symmetry_operations": int(values[7]) if values[7].strip() != "None" else -1,
                    "summary_ionic_steps": int(values[8]) if values[8].strip() != "None" else -1,
                }
            )
    return rows


def _midrank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    count = int(finite.sum())
    if count:
        result[finite] = (rankdata(values[finite], method="average") - 0.5) / count
    return result


def apply_dft_waste_endpoint(table: pd.DataFrame) -> pd.DataFrame:
    required = {"prefix", "calculation_message", "final_energy_total_ev", "n_sites"}
    if required - set(table):
        raise ValueError("NEXT545 endpoint source columns differ")
    result = table.copy()
    energy = pd.to_numeric(result["final_energy_total_ev"], errors="coerce").to_numpy(float)
    sites = pd.to_numeric(result["n_sites"], errors="coerce").to_numpy(float)
    message = result["calculation_message"].fillna("").astype(str).str.strip()
    failed = message.ne("").to_numpy() | ~np.isfinite(energy) | ~np.isfinite(sites) | (sites <= 0)
    result["failed"] = failed
    result["final_energy_per_atom_ev"] = np.where(~failed, energy / sites, np.nan)
    result["energy_percentile"] = np.nan
    for prefix in sorted(result["prefix"].astype(str).unique()):
        mask = result["prefix"].astype(str).eq(prefix).to_numpy() & ~failed
        result.loc[mask, "energy_percentile"] = _midrank(
            result.loc[mask, "final_energy_per_atom_ev"].to_numpy(float)
        )
    percentile = pd.to_numeric(result["energy_percentile"], errors="coerce").to_numpy(float)
    result["dft_waste"] = failed | (percentile > 0.75)
    result["waste_severity"] = np.where(failed, 1.25, percentile)
    return result


def _extract_last_structure_dict(path: Path) -> tuple[dict[str, object], dict[str, int]]:
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        match = n543._STRUCTURE_ARRAY.search(data)
        if match is None:
            raise ValueError("NEXT545 structure array marker is missing")
        cursor = match.end()
        spans: list[tuple[int, int]] = []
        while True:
            while cursor < len(data) and data[cursor] in b" \t\r\n,":
                cursor += 1
            if cursor >= len(data) or data[cursor] == 0x5D:
                break
            if data[cursor] != 0x7B:
                raise ValueError("NEXT545 structure array member differs")
            end = n543._balanced_object_end(data, cursor)
            spans.append((cursor, end))
            cursor = end
        if not spans:
            raise ValueError("NEXT545 trajectory contains no structure")
        start, end = spans[-1]
        payload = bytes(data[start:end])
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("NEXT545 last structure schema differs")
    return value, {
        "structure_objects_scanned": len(spans),
        "structure_objects_decoded": 1,
        "last_structure_start": start,
        "last_structure_end": end,
    }


def _stratified_bootstrap(
    table: pd.DataFrame, *, draws: int, seed: int
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    prefixes = sorted(table["prefix"].astype(str).unique())
    groups = {
        prefix: np.flatnonzero(table["prefix"].astype(str).to_numpy() == prefix)
        for prefix in prefixes
    }
    labels = table["dft_waste"].to_numpy(bool)
    risk = table["mupr_risk"].to_numpy(float)
    severity = table["waste_severity"].to_numpy(float)
    aucs: list[float] = []
    rhos: list[float] = []
    degenerate_auc = 0
    degenerate_rho = 0
    for _ in range(draws):
        indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups.values()]
        )
        y = labels[indices]
        if np.unique(y).size == 2:
            aucs.append(float(roc_auc_score(y, risk[indices])))
        else:
            degenerate_auc += 1
        rho = float(spearmanr(risk[indices], severity[indices]).statistic)
        if math.isfinite(rho):
            rhos.append(rho)
        else:
            degenerate_rho += 1
    def interval(values: list[float], degenerate: int) -> dict[str, object]:
        array = np.asarray(values, dtype=float)
        return {
            "valid": len(values),
            "degenerate": degenerate,
            "lower": float(np.quantile(array, 0.025)) if len(array) else math.nan,
            "median": float(np.quantile(array, 0.5)) if len(array) else math.nan,
            "upper": float(np.quantile(array, 0.975)) if len(array) else math.nan,
        }
    return {
        "draws": draws,
        "seed": seed,
        "strata": prefixes,
        "roc_auc": interval(aucs, degenerate_auc),
        "spearman": interval(rhos, degenerate_rho),
    }


def evaluate_confirmation(
    table: pd.DataFrame, *, bootstrap_draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED
) -> dict[str, object]:
    required = {"trajectory_id", "prefix", "mupr_risk", "dft_waste", "waste_severity"}
    if required - set(table):
        raise ValueError("NEXT545 confirmation table differs")
    work = table.copy()
    work["mupr_risk"] = pd.to_numeric(work["mupr_risk"], errors="coerce")
    work["waste_severity"] = pd.to_numeric(work["waste_severity"], errors="coerce")
    work = work.loc[
        np.isfinite(work["mupr_risk"]) & np.isfinite(work["waste_severity"])
    ].copy()
    if work.empty or work["dft_waste"].nunique() != 2:
        raise ValueError("NEXT545 requires two endpoint classes")
    work["dft_waste"] = work["dft_waste"].astype(bool)
    ordered = work.sort_values(
        ["mupr_risk", "prefix", "trajectory_id"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    top_n = max(1, math.ceil(0.15 * len(ordered)))
    bottom_n = max(1, math.ceil(0.50 * len(ordered)))
    top = ordered.iloc[:top_n]
    bottom = ordered.iloc[-bottom_n:]
    positives = int(ordered["dft_waste"].sum())
    top_positives = int(top["dft_waste"].sum())
    lower, upper = n542._wilson(top_positives, top_n)
    auc = float(roc_auc_score(ordered["dft_waste"], ordered["mupr_risk"]))
    rho = float(spearmanr(ordered["mupr_risk"], ordered["waste_severity"]).statistic)
    return {
        "rows": len(ordered),
        "positives": positives,
        "negatives": len(ordered) - positives,
        "roc_auc": auc,
        "spearman": rho,
        "top_15_percent": {
            "rows": top_n,
            "positives": top_positives,
            "precision": top_positives / top_n,
            "precision_wilson_lower": lower,
            "precision_wilson_upper": upper,
            "recall": top_positives / positives,
        },
        "bottom_50_percent": {
            "rows": bottom_n,
            "nonwaste": int((~bottom["dft_waste"]).sum()),
            "nonwaste_fraction": float((~bottom["dft_waste"]).mean()),
        },
        "bootstrap": _stratified_bootstrap(ordered, draws=bootstrap_draws, seed=seed),
    }


def _comparator_results(table: pd.DataFrame, candidates: dict[str, np.ndarray]) -> dict[str, object]:
    labels = table["dft_waste"].to_numpy(bool)
    mupr = table["mupr_risk"].to_numpy(float)
    results: dict[str, object] = {}
    for name, score in candidates.items():
        score = np.asarray(score, dtype=float)
        mask = np.isfinite(score) & np.isfinite(mupr)
        if mask.sum() < 2 or np.unique(labels[mask]).size != 2:
            results[name] = {"supported": False, "rows": int(mask.sum())}
            continue
        comparator_auc = float(roc_auc_score(labels[mask], score[mask]))
        mupr_auc = float(roc_auc_score(labels[mask], mupr[mask]))
        results[name] = {
            "supported": True,
            "rows": int(mask.sum()),
            "comparator_auc": comparator_auc,
            "mupr_auc_same_rows": mupr_auc,
            "mupr_margin": mupr_auc - comparator_auc,
        }
    supported = {name: value for name, value in results.items() if value.get("supported") is True}
    best_name = max(supported, key=lambda name: supported[name]["comparator_auc"]) if supported else None
    return {"candidates": results, "best_name": best_name, "best": supported.get(best_name) if best_name else None}


def _pauling_scores(table: pd.DataFrame) -> dict[str, np.ndarray]:
    decision = table["pauling_p2_p5_decision"].astype(str)
    return {
        "pauling_violation_fraction": pd.to_numeric(table["pauling_violation_fraction"], errors="coerce").to_numpy(float),
        "pauling_p2_value": pd.to_numeric(table["pauling_p2_value"], errors="coerce").to_numpy(float),
        "pauling_p3_value": pd.to_numeric(table["pauling_p3_value"], errors="coerce").to_numpy(float),
        "pauling_p4_value": pd.to_numeric(table["pauling_p4_value"], errors="coerce").to_numpy(float),
        "negative_pauling_p5_value": -pd.to_numeric(table["pauling_p5_value"], errors="coerce").to_numpy(float),
        "pauling_combined_reject": np.where(
            decision.eq("REJECT"), 1.0, np.where(decision.eq("KEEP"), 0.0, np.nan)
        ),
    }


def _validate_freezes(paths: dict[str, Path], hashes: dict[str, str]) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    m543 = json.loads(paths["next543_manifest"].read_text())
    m544 = json.loads(paths["next544_manifest"].read_text())
    f544 = json.loads(paths["next544_formula"].read_text())
    if (
        m543.get("endpoint_values_opened") is not False
        or m543.get("summary_files_downloaded_or_read") is not False
        or m544.get("protocol") != n544.PROTOCOL
        or m544.get("endpoint_summary_or_values_opened") is not False
        or m544.get("later_structure_objects_opened") is not False
        or m544.get("predictions_and_gates_frozen_before_endpoint_access") is not True
        or m544.get("next545_endpoint_access_authorized") is not True
        or f544.get("short_name") != "MUPR"
        or f544.get("identical_to_next541_formula") is not True
        or f544.get("coefficients_fitted_to_endpoint") is not False
    ):
        raise ValueError("NEXT545 freeze firewall differs")
    predictions = pd.read_parquet(paths["next544_table"])
    inventory = json.loads(paths["next543_inventory"].read_text())
    cohort = json.loads(paths["next543_cohort"].read_text())
    if (
        len(predictions) != n543.EXPECTED_ROWS
        or len(inventory) != n543.EXPECTED_ROWS
        or len(cohort) != n543.EXPECTED_ROWS
        or predictions["trajectory_id"].duplicated().any()
    ):
        raise ValueError("NEXT545 frozen cohort identity differs")
    return predictions, inventory, cohort


def run_confirmation(
    *, source_dir: Path, next543_dir: Path, next544_dir: Path, design_path: Path,
    output_dir: Path, bootstrap_draws: int = BOOTSTRAP_DRAWS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    source = Path(source_dir).resolve()
    up543 = Path(next543_dir).resolve()
    up544 = Path(next544_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "design": Path(design_path).resolve(),
        "next543_manifest": up543 / n543.MANIFEST_NAME,
        "next543_inventory": up543 / n543.INVENTORY_NAME,
        "next543_cohort": up543 / n543.COHORT_NAME,
        "next543_geometry": up543 / n543.GEOMETRY_NAME,
        "next543_source": Path(n543.__file__).resolve(),
        "next544_manifest": up544 / n544.MANIFEST_NAME,
        "next544_table": up544 / n544.TABLE_NAME,
        "next544_formula": up544 / n544.FORMULA_NAME,
        "next544_source": Path(n544.__file__).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT545 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name for name in set(hashes) | set(EXPECTED_INPUT_SHA256)
            if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT545 formal input identity differs: {differing}")
    predictions, inventory, cohort = _validate_freezes(paths, hashes)
    inventory_by_object = {str(row["object_name"]): row for row in inventory}
    cohort_by_id = {str(row["trajectory_id"]): row for row in cohort}
    for row in cohort:
        trajectory_id = str(row["trajectory_id"])
        item = inventory_by_object[str(row["object_name"])]
        raw = Path(str(item["local_path"]))
        if not raw.is_file() or _sha256(raw) != str(item["local_sha256"]):
            raise ValueError(f"NEXT545 sealed raw trajectory differs: {trajectory_id}")

    # First endpoint access: download and parse only the four frozen summary files.
    summaries = [_download_summary(prefix, source) for prefix in n543.PREFIXES]
    summary_rows: list[dict[str, object]] = []
    for summary in summaries:
        summary_rows.extend(
            parse_summary_rows(Path(str(summary["local_path"])), prefix=str(summary["prefix"]))
        )
    summary_by_key = {
        (str(row["prefix"]), str(row["object_suffix"])): row for row in summary_rows
    }
    endpoint_rows: list[dict[str, object]] = []
    for row in cohort:
        object_stem = Path(str(row["object_name"])).stem
        suffix = object_stem.split("_", 1)[1]
        endpoint = summary_by_key.get((str(row["prefix"]), suffix))
        if endpoint is None:
            continue
        endpoint_rows.append({**row, **endpoint})
    endpoint = pd.DataFrame(endpoint_rows)
    table = predictions.merge(
        endpoint[[
            "trajectory_id", "calculation_message", "final_energy_total_ev",
            "initial_spacegroup", "final_spacegroup", "summary_initial_volume",
            "summary_final_volume", "summary_symmetry_operations", "summary_ionic_steps",
        ]],
        on="trajectory_id", how="inner", validate="one_to_one",
    )
    table = apply_dft_waste_endpoint(table)

    # Secondary response diagnostic: decode one final structure per selected trajectory.
    with zipfile.ZipFile(paths["next543_geometry"]) as archive:
        initial_structures = {
            Path(name).stem: Structure.from_dict(json.loads(archive.read(name)))
            for name in archive.namelist()
        }
    final_scan_counts: list[int] = []
    for index, row in table.iterrows():
        trajectory_id = str(row["trajectory_id"])
        source_row = cohort_by_id[trajectory_id]
        raw_item = inventory_by_object[str(source_row["object_name"])]
        final_dict, scan = _extract_last_structure_dict(Path(str(raw_item["local_path"])))
        final = Structure.from_dict(final_dict)
        initial = initial_structures[trajectory_id]
        final_scan_counts.append(scan["structure_objects_scanned"])
        match = n542._pair_match(initial, final)
        table.at[index, "final_structure_frames"] = scan["structure_objects_scanned"]
        table.at[index, "final_volume_log_response"] = abs(
            math.log((final.volume / len(final)) / (initial.volume / len(initial)))
        )
        if match is not None:
            tier, rms, maximum = match
            table.at[index, "final_match_tier"] = tier
            table.at[index, "final_normalized_rms"] = rms
            table.at[index, "final_normalized_max"] = maximum

    coverage = len(table) / n543.EXPECTED_ROWS
    screen = evaluate_confirmation(table, bootstrap_draws=bootstrap_draws, seed=BOOTSTRAP_SEED)
    pauling = _comparator_results(table, _pauling_scores(table))
    components = _comparator_results(
        table,
        {
            "contact_percentile": pd.to_numeric(table["contact_percentile"], errors="coerce").to_numpy(float),
            "sssp_percentile": pd.to_numeric(table["sssp_percentile"], errors="coerce").to_numpy(float),
            "pbaaa_percentile": pd.to_numeric(table["pbaaa_percentile"], errors="coerce").to_numpy(float),
        },
    )
    prefix_metrics: dict[str, object] = {}
    for prefix, group in table.groupby("prefix", sort=True):
        labels = group["dft_waste"].to_numpy(bool)
        prefix_metrics[str(prefix)] = {
            "rows": len(group),
            "positives": int(labels.sum()),
            "negatives": int((~labels).sum()),
            "roc_auc": float(roc_auc_score(labels, group["mupr_risk"]))
            if np.unique(labels).size == 2 else math.nan,
        }
    prefix_pass_count = sum(
        math.isfinite(value["roc_auc"]) and value["roc_auc"] > MINIMUM_PREFIX_AUC
        for value in prefix_metrics.values()
    )
    class_counts_pass = (
        min(screen["positives"], screen["negatives"]) >= MINIMUM_CLASS_COUNT
        and all(
            min(value["positives"], value["negatives"]) >= MINIMUM_PREFIX_CLASS_COUNT
            for value in prefix_metrics.values()
        )
    )
    best_pauling = pauling["best"]
    best_component = components["best"]
    gates = {
        "endpoint_coverage": {"value": coverage, "threshold": MINIMUM_ENDPOINT_COVERAGE, "passes": coverage >= MINIMUM_ENDPOINT_COVERAGE},
        "class_counts": {"positives": screen["positives"], "negatives": screen["negatives"], "minimum_overall": MINIMUM_CLASS_COUNT, "minimum_per_prefix": MINIMUM_PREFIX_CLASS_COUNT, "passes": class_counts_pass},
        "mupr_coverage": {"value": screen["rows"] / len(table), "threshold": MINIMUM_MUPR_COVERAGE, "passes": screen["rows"] / len(table) >= MINIMUM_MUPR_COVERAGE},
        "roc_auc": {"value": screen["roc_auc"], "lower": screen["bootstrap"]["roc_auc"]["lower"], "point_threshold": MINIMUM_AUC, "lower_threshold_strict": MINIMUM_AUC_LOWER, "passes": screen["roc_auc"] >= MINIMUM_AUC and screen["bootstrap"]["roc_auc"]["lower"] > MINIMUM_AUC_LOWER},
        "pauling_margin": {"best": pauling["best_name"], "value": best_pauling.get("mupr_margin") if isinstance(best_pauling, dict) else None, "threshold": MINIMUM_PAULING_MARGIN, "passes": isinstance(best_pauling, dict) and best_pauling["mupr_margin"] >= MINIMUM_PAULING_MARGIN},
        "component_margin": {"best": components["best_name"], "value": best_component.get("mupr_margin") if isinstance(best_component, dict) else None, "threshold": MINIMUM_COMPONENT_MARGIN, "passes": isinstance(best_component, dict) and best_component["mupr_margin"] >= MINIMUM_COMPONENT_MARGIN},
        "top_15_percent": {**screen["top_15_percent"], "precision_threshold": MINIMUM_TOP_PRECISION, "precision_lower_threshold": MINIMUM_TOP_PRECISION_LOWER, "recall_threshold": MINIMUM_TOP_RECALL, "passes": screen["top_15_percent"]["precision"] >= MINIMUM_TOP_PRECISION and screen["top_15_percent"]["precision_wilson_lower"] >= MINIMUM_TOP_PRECISION_LOWER and screen["top_15_percent"]["recall"] >= MINIMUM_TOP_RECALL},
        "bottom_50_percent": {**screen["bottom_50_percent"], "nonwaste_threshold": MINIMUM_BOTTOM_NONWASTE, "passes": screen["bottom_50_percent"]["nonwaste_fraction"] >= MINIMUM_BOTTOM_NONWASTE},
        "spearman": {"value": screen["spearman"], "lower": screen["bootstrap"]["spearman"]["lower"], "point_threshold": MINIMUM_SPEARMAN, "lower_threshold_strict": MINIMUM_SPEARMAN_LOWER, "passes": screen["spearman"] >= MINIMUM_SPEARMAN and screen["bootstrap"]["spearman"]["lower"] > MINIMUM_SPEARMAN_LOWER},
        "within_prefix": {"passing": prefix_pass_count, "required": MINIMUM_PREFIXES_PASSING, "auc_threshold_strict": MINIMUM_PREFIX_AUC, "passes": prefix_pass_count >= MINIMUM_PREFIXES_PASSING},
    }
    all_pass = all(value["passes"] for value in gates.values())
    result = {
        "protocol": PROTOCOL,
        "endpoint_definition": "failure or successful final-energy percentile above 0.75 within prefix",
        "endpoint_coverage": coverage,
        "screen": screen,
        "prefix_metrics": prefix_metrics,
        "pauling_comparisons": pauling,
        "single_mechanism_comparisons": components,
        "gates": gates,
        "all_gates_pass": all_pass,
        "scientific_success": all_pass,
        "claim_scope": "Li-Si AIRSS DFT-waste endpoint only",
        "retuning_after_endpoint_access_performed": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    try:
        table_path = staging / TABLE_NAME
        result_path = staging / RESULT_NAME
        table.to_parquet(table_path, index=False)
        result_path.write_bytes(_json_bytes(result))
        outputs = {table_path.name: _sha256(table_path), result_path.name: _sha256(result_path)}
        manifest = {
            "protocol": PROTOCOL,
            "inputs_sha256": {name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()},
            "endpoint_summary_sha256": {row["prefix"]: {"path": row["local_path"], "sha256": row["sha256"], "size": row["size"]} for row in summaries},
            "outputs_sha256": outputs,
            "executed_source_sha256": {"src/next545_lisi_random_relaxation_confirmation.py": source_hash},
            "predictions_and_gates_frozen_before_endpoint_access": True,
            "summary_endpoint_values_opened_offline": True,
            "final_structures_opened_offline": True,
            "dft_force_or_stress_values_decoded_or_used": False,
            "endpoint_used_by_executable_screen": False,
            "final_structure_scan_counts": {"minimum": min(final_scan_counts), "median": float(np.median(final_scan_counts)), "maximum": max(final_scan_counts)},
            "bootstrap_draws": bootstrap_draws,
            "all_gates_pass": all_pass,
            "scientific_success": all_pass,
            "standalone_report_authorized": all_pass,
            "canonical_report_or_paper_edit_authorized": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT545 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT545 frozen input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--next543-dir", type=Path, required=True)
    parser.add_argument("--next544-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_confirmation(
        source_dir=args.source_dir, next543_dir=args.next543_dir,
        next544_dir=args.next544_dir, design_path=args.design,
        output_dir=args.output_dir, bootstrap_draws=args.bootstrap_draws,
    )
    print(json.dumps({"gates": result["gates"], "all_gates_pass": result["all_gates_pass"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
