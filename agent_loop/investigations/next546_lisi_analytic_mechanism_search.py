#!/usr/bin/env python3
"""Bounded x0-only analytic mechanism search on the opened NEXT545 development set."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

from ase import Atoms
from ase.data import covalent_radii
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next32_inorganic_response_features import (
    _canonical_periodic_ratios,
    _resolve_radii,
)
from src.next43_analytic_feature_bank import (
    CANDIDATE_FEATURE_NAMES as NEXT43_FEATURE_NAMES,
    compute_analytic_feature_row,
)
import src.next411_same_sign_shell_purity as n411
import src.next537_periodic_bond_angle_affine_accommodation as n537
import src.next543_lisi_random_relaxation_initial_cohort as n543
import src.next545_lisi_random_relaxation_confirmation as n545


PROTOCOL = "2026-08-13-next546-lisi-analytic-mechanism-search-v1"
DESIGN_SHA256 = "fab8f66aff9294ccefa74ba77721c0e65170d5d725249997ff12e6b1261fb607"
DISCOVERY_PREFIXES = ("Li1Si1_02", "Li2Si1_02")
VALIDATION_PREFIXES = ("Li7Si2_03", "Li15Si4_02")
MINIMUM_SUPPORT = 0.80
MINIMUM_UNIQUE = 20
MINIMUM_UNIVARIATE_AUC = 0.58
MAXIMUM_RETAINED = 16
MAXIMUM_REDUNDANCY = 0.95
PAIR_FORMULAS = ("mean", "maximum", "union", "concurrence")
PRIMITIVE_FEATURE_NAMES = (
    "primitive_volume_per_atom",
    "primitive_covalent_packing_fraction",
    "primitive_cell_metric_anisotropy",
    "primitive_contact_ratio_mean",
    "primitive_contact_ratio_std",
    "primitive_contact_ratio_q10",
    "primitive_contact_ratio_q50",
    "primitive_contact_ratio_q90",
)
FEATURE_TABLE_NAME = "next546_lisi_x0_analytic_feature_bank.parquet"
UNIVARIATE_NAME = "NEXT546_UNIVARIATE_SCREEN.json"
PAIR_SEARCH_NAME = "NEXT546_PAIR_SEARCH.json"
FORMULA_NAME = "NEXT546_DEVELOPMENT_FORMULA.json"
MANIFEST_NAME = "MANIFEST.json"
EXPECTED_INPUT_SHA256 = {
    "design": DESIGN_SHA256,
    "next543_manifest": "5521ee2ad498f9529db0f88d82242efa46f930c5351fd3bf1fc0e069bbb51cf1",
    "next543_cohort": "9022a3a0a81e3cca43f237f3b217391b892ba66781a83c09dc5055c1a02964bc",
    "next543_geometry": "1c249fbc66267c964eb840b06886acd5f21a4884b05533469567866753dd843e",
    "next545_manifest": "5f84151ef187bd2427cafc49858c012575b7ceca6847256381da3a52add6caf2",
    "next545_table": "7f2075c36619db4fbf0e5e64cb043287874655f983ce5fbc6b1d8a2d1700446a",
    "next545_result": "fc989e927b6d0ec267d6caf25e1ab7fea35b573a289f5c07ea262c35c20f0c31",
    "next43_source": "9212af77b86491ae71214f810bb316612d3037f87f645d57b712faebec0c4d24",
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def primitive_geometry_features(atoms: Atoms) -> dict[str, float]:
    if len(atoms) < 1 or not np.all(atoms.pbc):
        raise ValueError("NEXT546 primitive geometry differs")
    cell = np.asarray(atoms.cell.array, dtype=float)
    volume = abs(float(np.linalg.det(cell)))
    if not np.isfinite(cell).all() or volume <= 1.0e-10:
        raise ValueError("NEXT546 primitive cell differs")
    radii = _resolve_radii(np.asarray(atoms.numbers, dtype=int), None)
    pairs = _canonical_periodic_ratios(atoms, radii)
    ratios = np.asarray([value for _i, _j, value in pairs], dtype=float)
    if not len(ratios) or not np.isfinite(ratios).all():
        raise ValueError("NEXT546 primitive contact population differs")
    singular = np.linalg.svd(cell, compute_uv=False)
    values = {
        "primitive_volume_per_atom": volume / len(atoms),
        "primitive_covalent_packing_fraction": float(
            np.sum((4.0 * math.pi / 3.0) * radii**3) / volume
        ),
        "primitive_cell_metric_anisotropy": float(singular.max() / singular.min()),
        "primitive_contact_ratio_mean": float(ratios.mean()),
        "primitive_contact_ratio_std": float(ratios.std()),
        "primitive_contact_ratio_q10": float(np.quantile(ratios, 0.10)),
        "primitive_contact_ratio_q50": float(np.quantile(ratios, 0.50)),
        "primitive_contact_ratio_q90": float(np.quantile(ratios, 0.90)),
    }
    if tuple(values) != PRIMITIVE_FEATURE_NAMES or not np.isfinite(list(values.values())).all():
        raise RuntimeError("NEXT546 primitive feature schema differs")
    return values


def symmetric_pair_score(u: np.ndarray, v: np.ndarray, formula: str) -> np.ndarray:
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    if u.shape != v.shape:
        raise ValueError("NEXT546 pair arrays differ")
    if formula == "mean":
        return (u + v) / 2.0
    if formula == "maximum":
        return np.maximum(u, v)
    if formula == "union":
        return 1.0 - (1.0 - u) * (1.0 - v)
    if formula == "concurrence":
        return np.minimum(u, v)
    raise ValueError("NEXT546 pair formula differs")


def _compute_payload(item: tuple[str, bytes]) -> dict[str, object]:
    trajectory_id, payload = item
    try:
        structure = Structure.from_dict(json.loads(payload))
        atoms = AseAtomsAdaptor.get_atoms(structure)
        row = compute_analytic_feature_row(atoms)
        row.update(primitive_geometry_features(atoms))
        return {"trajectory_id": trajectory_id, **row}
    except Exception as exc:
        return {"trajectory_id": trajectory_id, "next546_failure": f"{type(exc).__name__}: {exc}"}


def _compute_many(payloads: list[tuple[str, bytes]], workers: int) -> list[dict[str, object]]:
    if workers == 1:
        iterator = map(_compute_payload, payloads)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_compute_payload, payloads, chunksize=1)
    rows = []
    try:
        for offset, row in enumerate(iterator, start=1):
            rows.append(row)
            if offset % 25 == 0 or offset == len(payloads):
                print(f"NEXT546 analytic x0 features: {offset}/{len(payloads)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return rows


def _percentile(values: np.ndarray, direction: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    if finite.any():
        base = values[finite] if direction == "high" else -values[finite]
        result[finite] = (rankdata(base, method="average") - 0.5) / finite.sum()
    return result


def _metrics(score: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    supported = mask & np.isfinite(score)
    y = labels[supported]
    if supported.sum() < 2 or np.unique(y).size != 2:
        return {"rows": int(supported.sum()), "auc": math.nan, "top15_precision": math.nan}
    values = score[supported]
    order = np.argsort(-values, kind="mergesort")
    count = max(1, math.ceil(0.15 * len(order)))
    return {
        "rows": int(supported.sum()),
        "coverage": float(supported.sum() / mask.sum()),
        "auc": float(roc_auc_score(y, values)),
        "top15_precision": float(y[order[:count]].mean()),
    }


def run_bounded_search(table: pd.DataFrame, feature_names: tuple[str, ...]) -> dict[str, object]:
    prefixes = table["prefix"].astype(str).to_numpy()
    discovery = np.isin(prefixes, DISCOVERY_PREFIXES)
    validation = np.isin(prefixes, VALIDATION_PREFIXES)
    labels = table["dft_waste"].to_numpy(bool)
    failed = table["failed"].to_numpy(bool)
    high_energy = (~failed) & (table["energy_percentile"].to_numpy(float) > 0.75)
    directions: list[dict[str, object]] = []
    scores: dict[str, np.ndarray] = {}
    for feature in feature_names:
        values = pd.to_numeric(table[feature], errors="coerce").to_numpy(float)
        for direction in ("high", "low"):
            key = f"{feature}__risk_{direction}"
            score = _percentile(values, direction)
            scores[key] = score
            d = _metrics(score, labels, discovery)
            v = _metrics(score, labels, validation)
            combined = _metrics(score, labels, np.ones(len(table), dtype=bool))
            finite_d = np.isfinite(score[discovery])
            finite_v = np.isfinite(score[validation])
            unique_d = int(np.unique(np.round(score[discovery][finite_d], 12)).size)
            unique_v = int(np.unique(np.round(score[validation][finite_v], 12)).size)
            searchable = bool(
                d.get("coverage", 0.0) >= MINIMUM_SUPPORT
                and v.get("coverage", 0.0) >= MINIMUM_SUPPORT
                and unique_d >= MINIMUM_UNIQUE
                and unique_v >= MINIMUM_UNIQUE
            )
            enters = bool(
                searchable
                and d["auc"] >= MINIMUM_UNIVARIATE_AUC
                and v["auc"] >= MINIMUM_UNIVARIATE_AUC
            )
            failure_auc = _metrics(score, failed, np.ones(len(table), dtype=bool))["auc"]
            converged_mask = ~failed
            energy_auc = _metrics(score, high_energy, converged_mask)["auc"]
            directions.append(
                {
                    "key": key,
                    "feature": feature,
                    "direction": direction,
                    "discovery": d,
                    "validation": v,
                    "combined": combined,
                    "failure_auc": failure_auc,
                    "high_energy_auc": energy_auc,
                    "unique_discovery": unique_d,
                    "unique_validation": unique_v,
                    "searchable": searchable,
                    "enters_pair_search": enters,
                }
            )
    entrants = [row for row in directions if row["enters_pair_search"]]
    entrants.sort(
        key=lambda row: (
            -min(row["discovery"]["auc"], row["validation"]["auc"]),
            -row["combined"]["auc"],
            row["key"],
        )
    )
    retained: list[dict[str, object]] = []
    for row in entrants:
        candidate = scores[row["key"]]
        redundant = False
        for prior in retained:
            other = scores[prior["key"]]
            finite = np.isfinite(candidate) & np.isfinite(other)
            rho = float(spearmanr(candidate[finite], other[finite]).statistic)
            if math.isfinite(rho) and abs(rho) > MAXIMUM_REDUNDANCY:
                redundant = True
                break
        if not redundant:
            retained.append(row)
        if len(retained) == MAXIMUM_RETAINED:
            break

    pairs: list[dict[str, object]] = []
    for first_index, first in enumerate(retained):
        for second in retained[first_index + 1 :]:
            u, v = scores[first["key"]], scores[second["key"]]
            joint = np.isfinite(u) & np.isfinite(v)
            for formula in PAIR_FORMULAS:
                score = np.full(len(table), np.nan)
                score[joint] = symmetric_pair_score(u[joint], v[joint], formula)
                d = _metrics(score, labels, discovery)
                val = _metrics(score, labels, validation)
                combined = _metrics(score, labels, np.ones(len(table), dtype=bool))
                first_d = _metrics(u, labels, discovery)["auc"]
                second_d = _metrics(v, labels, discovery)["auc"]
                first_v = _metrics(u, labels, validation)["auc"]
                second_v = _metrics(v, labels, validation)["auc"]
                margin_d = d["auc"] - max(first_d, second_d)
                margin_v = val["auc"] - max(first_v, second_v)
                eligible = bool(
                    d.get("coverage", 0) >= MINIMUM_SUPPORT
                    and val.get("coverage", 0) >= MINIMUM_SUPPORT
                    and d["auc"] >= 0.65
                    and val["auc"] >= 0.65
                    and combined["auc"] >= 0.67
                    and d["top15_precision"] >= 0.50
                    and val["top15_precision"] >= 0.50
                    and margin_d >= 0.02
                    and margin_v >= 0.02
                )
                pairs.append(
                    {
                        "first": first["key"],
                        "second": second["key"],
                        "formula": formula,
                        "discovery": d,
                        "validation": val,
                        "combined": combined,
                        "component_margin_discovery": margin_d,
                        "component_margin_validation": margin_v,
                        "eligible": eligible,
                    }
                )
    eligible = [row for row in pairs if row["eligible"]]
    eligible.sort(
        key=lambda row: (
            -min(row["discovery"]["auc"], row["validation"]["auc"]),
            -min(row["component_margin_discovery"], row["component_margin_validation"]),
            -row["combined"]["auc"],
            row["first"], row["second"], row["formula"],
        )
    )
    return {
        "univariate": directions,
        "retained": [row["key"] for row in retained],
        "pairs": pairs,
        "eligible_count": len(eligible),
        "winner": eligible[0] if eligible else None,
    }


def build_and_search(
    *, next543_dir: Path, next545_dir: Path, design_path: Path,
    output_dir: Path, workers: int = 8, require_formal_inputs: bool = True,
) -> dict[str, object]:
    up543, up545 = Path(next543_dir).resolve(), Path(next545_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "design": Path(design_path).resolve(),
        "next543_manifest": up543 / n543.MANIFEST_NAME,
        "next543_cohort": up543 / n543.COHORT_NAME,
        "next543_geometry": up543 / n543.GEOMETRY_NAME,
        "next545_manifest": up545 / n545.MANIFEST_NAME,
        "next545_table": up545 / n545.TABLE_NAME,
        "next545_result": up545 / n545.RESULT_NAME,
        "next43_source": Path(compute_analytic_feature_row.__code__.co_filename).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT546 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(name for name in set(hashes) | set(EXPECTED_INPUT_SHA256) if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name))
        raise ValueError(f"NEXT546 formal input identity differs: {differing}")
    endpoint = pd.read_parquet(paths["next545_table"])
    with zipfile.ZipFile(paths["next543_geometry"]) as archive:
        payloads = [(Path(name).stem, archive.read(name)) for name in archive.namelist()]
    payloads.sort(key=lambda item: item[0])
    features = pd.DataFrame(_compute_many(payloads, workers))
    if "next546_failure" in features and features["next546_failure"].notna().any():
        raise RuntimeError("NEXT546 complete feature row failed")
    table = endpoint.merge(features, on="trajectory_id", how="left", validate="one_to_one", suffixes=("", "_analytic"))
    extra = (n411.FEATURE_NAMES[0], n537.FEATURE_NAMES[0])
    feature_names = tuple(NEXT43_FEATURE_NAMES) + PRIMITIVE_FEATURE_NAMES + extra
    if any(name not in table for name in feature_names):
        raise RuntimeError("NEXT546 feature bank schema differs")
    search = run_bounded_search(table, feature_names)
    winner = search["winner"]
    formula = {
        "protocol": PROTOCOL,
        "development_only": True,
        "winner_found": winner is not None,
        "winner": winner,
        "independent_confirmation_required": True,
        "scientific_success": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    try:
        table_path = staging / FEATURE_TABLE_NAME
        uni_path = staging / UNIVARIATE_NAME
        pair_path = staging / PAIR_SEARCH_NAME
        formula_path = staging / FORMULA_NAME
        table.to_parquet(table_path, index=False)
        uni_path.write_bytes(_json_bytes({"protocol": PROTOCOL, "rows": search["univariate"], "retained": search["retained"]}))
        pair_path.write_bytes(_json_bytes({"protocol": PROTOCOL, "eligible_count": search["eligible_count"], "winner": winner, "rows": search["pairs"]}))
        formula_path.write_bytes(_json_bytes(formula))
        outputs = {path.name: _sha256(path) for path in (table_path, uni_path, pair_path, formula_path)}
        manifest = {
            "protocol": PROTOCOL,
            "mode": "opened-development-endpoint-bounded-search",
            "split": {"discovery": list(DISCOVERY_PREFIXES), "validation": list(VALIDATION_PREFIXES)},
            "feature_count": len(feature_names),
            "retained_count": len(search["retained"]),
            "pair_candidates": len(search["pairs"]),
            "eligible_count": search["eligible_count"],
            "winner_found": winner is not None,
            "inputs_sha256": {name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()},
            "outputs_sha256": outputs,
            "executed_source_sha256": {"src/next546_lisi_analytic_mechanism_search.py": source_hash},
            "screen_inputs_x0_only": True,
            "dft_values_used_only_as_opened_development_labels": True,
            "formula_contains_dft_or_relaxed_input": False,
            "scientific_success": False,
            "standalone_success_report_authorized": False,
            "next547_formula_freeze_authorized": winner is not None,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT546 source changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next543-dir", type=Path, required=True)
    parser.add_argument("--next545-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_and_search(
        next543_dir=args.next543_dir, next545_dir=args.next545_dir,
        design_path=args.design, output_dir=args.output_dir, workers=args.workers,
    )
    print(json.dumps({key: manifest[key] for key in ("feature_count", "retained_count", "pair_candidates", "eligible_count", "winner_found")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
