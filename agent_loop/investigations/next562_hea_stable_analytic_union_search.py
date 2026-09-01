#!/usr/bin/env python3
"""Search a bounded coefficient-free HEA law on all opened x0 cohorts."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

from ase.data import atomic_masses, atomic_numbers, covalent_radii
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy.stats import spearmanr

from src.next19_feature_build import _publish_directory_no_replace, _sha256
import src.next552_hea_analytic_feature_freeze as n552
import src.next553_hea_development_search as n553
import src.next558_hea_packing_deficit_validation as n558
import src.next559_hea_entropy_packing_discovery as n559
import src.next560_hea_entropy_packing_cohort as n560
import src.next561_hea_entropy_packing_confirmation as n561


PROTOCOL = "2026-08-14-next562-hea-stable-analytic-union-search-v1"
DESIGN_SHA256 = "fdc40ba5f6f974717e79ab5c3ddc8377836d9daf86cf69ffdf9652b67f95bfa7"
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 562_202_608
MAXIMUM_RETAINED = 16
MAXIMUM_BOOTSTRAP_CANDIDATES = 20
MAXIMUM_REDUNDANCY = 0.95
TABLE_NAME = "next562_hea_opened_analytic_features.parquet"
SEARCH_NAME = "NEXT562_HEA_STABLE_ANALYTIC_UNION_SEARCH.json"
FORMULA_NAME = "NEXT562_FROZEN_ANALYTIC_UNION_FORMULA.json"
MANIFEST_NAME = "MANIFEST.json"


COMPOSITION_FEATURE_NAMES = (
    "composition_ideal_entropy",
    "composition_element_count",
    "composition_covalent_radius_mean",
    "composition_covalent_radius_std",
    "composition_covalent_radius_range",
    "composition_covalent_radius_cv",
    "composition_atomic_number_mean",
    "composition_atomic_number_std",
    "composition_atomic_number_range",
    "composition_atomic_number_cv",
    "composition_atomic_mass_mean",
    "composition_atomic_mass_std",
    "composition_atomic_mass_range",
    "composition_atomic_mass_cv",
)
RAW_FEATURE_NAMES = tuple(n552.FEATURE_NAMES) + COMPOSITION_FEATURE_NAMES
PROVENANCE_STRATA = (
    "old_development", "old_validation", "new_identity_known_system",
    "unseen_chemical_system",
)
EVALUATION_STRATA = PROVENANCE_STRATA + ("ordered", "sqs")


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _number(metrics: dict[str, object], key: str, default: float = -math.inf) -> float:
    try:
        value = float(metrics.get(key))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _weighted_summary(values: np.ndarray, fractions: np.ndarray) -> tuple[float, ...]:
    mean = float(np.sum(fractions * values))
    std = float(np.sqrt(np.sum(fractions * (values - mean) ** 2)))
    span = float(values.max() - values.min())
    return mean, std, span, std / mean if mean > 0 else math.nan


def composition_features(formula: str) -> dict[str, float]:
    composition = Composition(formula)
    elements = list(composition.elements)
    fractions = np.asarray(
        [composition[element] / composition.num_atoms for element in elements], dtype=float
    )
    numbers = np.asarray([atomic_numbers[element.symbol] for element in elements], dtype=int)
    entropy = float(-(fractions * np.log(fractions)).sum())
    values: dict[str, float] = {
        "composition_ideal_entropy": entropy,
        "composition_element_count": float(len(elements)),
    }
    for prefix, array in (
        ("composition_covalent_radius", np.asarray(covalent_radii[numbers], dtype=float)),
        ("composition_atomic_number", numbers.astype(float)),
        ("composition_atomic_mass", np.asarray(atomic_masses[numbers], dtype=float)),
    ):
        mean, std, span, cv = _weighted_summary(array, fractions)
        values[f"{prefix}_mean"] = mean
        values[f"{prefix}_std"] = std
        values[f"{prefix}_range"] = span
        values[f"{prefix}_cv"] = cv
    if set(values) != set(COMPOSITION_FEATURE_NAMES) or not np.isfinite(list(values.values())).all():
        raise ValueError(f"NEXT562 composition feature schema differs: {formula}")
    return values


def combine_risks(arrays: list[np.ndarray], operation: str) -> np.ndarray:
    if len(arrays) not in (2, 3):
        raise ValueError("NEXT562 combination arity differs")
    matrix = np.column_stack([np.asarray(array, dtype=float) for array in arrays])
    if operation == "mean":
        return matrix.mean(axis=1)
    if operation == "maximum":
        return matrix.max(axis=1)
    if operation == "minimum":
        return matrix.min(axis=1)
    if operation == "union":
        return 1.0 - np.prod(1.0 - matrix, axis=1)
    raise ValueError("NEXT562 combination operation differs")


def _masks(table: pd.DataFrame) -> dict[str, np.ndarray]:
    strata = table["development_stratum"].astype(str)
    families = table["size_family"].astype(str)
    return {
        "overall": np.ones(len(table), dtype=bool),
        **{name: strata.eq(name).to_numpy() for name in PROVENANCE_STRATA},
        "ordered": families.eq("ordered").to_numpy(),
        "sqs": families.eq("sqs").to_numpy(),
    }


def _metrics(table: pd.DataFrame, score: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, object]:
    return {name: n553._score_metrics(table, score, mask) for name, mask in masks.items()}


def _candidate_preeligible(
    metrics: dict[str, dict[str, object]], component_aucs: list[float]
) -> tuple[bool, dict[str, object]]:
    overall = metrics["overall"]
    top = overall.get("top_15_percent", {})
    top_fraction = float(top.get("protected", math.inf)) / max(int(top.get("rows", 0)), 1)
    overall_auc = _number(overall, "roc_auc")
    margins = [overall_auc - auc for auc in component_aucs]
    clauses = {
        "overall_auc_at_least_0p75": overall_auc >= 0.75,
        "provenance_auc_at_least_0p68": all(
            _number(metrics[name], "roc_auc") >= 0.68 for name in PROVENANCE_STRATA
        ),
        "families_auc_at_least_0p70": all(
            _number(metrics[name], "roc_auc") >= 0.70 for name in ("ordered", "sqs")
        ),
        "spearman_at_least_0p38": _number(overall, "spearman_severity") >= 0.38,
        "top15_lift_at_least_1p55": _number(top, "lift") >= 1.55,
        "top15_protected_fraction_at_most_0p02": top_fraction <= 0.02,
        "component_margins_at_least_0p02": all(margin >= 0.02 for margin in margins),
    }
    return bool(all(clauses.values())), {
        "clauses": clauses,
        "component_auc_margins": margins,
        "top15_protected_fraction": top_fraction,
    }


def run_bounded_search(
    table: pd.DataFrame, *, bootstrap_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    if set(RAW_FEATURE_NAMES) - set(table) or len(table) != 4_400:
        raise ValueError("NEXT562 opened feature table differs")
    masks = _masks(table)
    directions: list[dict[str, object]] = []
    scores: dict[str, np.ndarray] = {}
    for feature in RAW_FEATURE_NAMES:
        raw = pd.to_numeric(table[feature], errors="coerce").to_numpy(float)
        for direction, reverse in (("high", False), ("low", True)):
            key = f"{feature}__risk_{direction}"
            score = n552._midrank(raw, reverse=reverse)
            metrics = _metrics(table, score, masks)
            eligible = bool(
                _number(metrics["overall"], "roc_auc") >= 0.64
                and all(_number(metrics[name], "roc_auc") >= 0.55 for name in EVALUATION_STRATA)
            )
            directions.append({
                "key": key, "feature": feature, "direction": direction,
                "eligible": eligible, "metrics": metrics,
            })
            scores[key] = score
    entrants = [row for row in directions if row["eligible"]]
    entrants.sort(key=lambda row: (
        -min(_number(row["metrics"][name], "roc_auc") for name in EVALUATION_STRATA),
        -_number(row["metrics"]["overall"], "roc_auc"), row["key"],
    ))
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

    preliminary: list[dict[str, object]] = []
    for arity in (2, 3):
        operations = ("mean", "maximum", "minimum", "union") if arity == 2 else ("union",)
        for members in combinations(retained, arity):
            terms = [str(member["key"]) for member in members]
            component_aucs = [
                _number(member["metrics"]["overall"], "roc_auc") for member in members
            ]
            for operation in operations:
                score = combine_risks([scores[term] for term in terms], operation)
                metrics = _metrics(table, score, masks)
                eligible, diagnostic = _candidate_preeligible(metrics, component_aucs)
                if eligible:
                    preliminary.append({
                        "identity": f"{operation}({','.join(terms)})",
                        "terms": terms, "operation": operation, "metrics": metrics,
                        **diagnostic,
                    })
    preliminary.sort(key=lambda row: (
        -min(_number(row["metrics"][name], "roc_auc") for name in PROVENANCE_STRATA),
        -_number(row["metrics"]["overall"], "roc_auc"),
        len(row["terms"]), row["identity"],
    ))
    inspected: list[dict[str, object]] = []
    selected: dict[str, object] | None = None
    for offset, row in enumerate(preliminary[:MAXIMUM_BOOTSTRAP_CANDIDATES]):
        score = combine_risks([scores[term] for term in row["terms"]], str(row["operation"]))
        bootstrap = n558._cluster_bootstrap(
            table, score, draws=bootstrap_draws, seed=seed + offset
        )
        passed = _number(bootstrap["roc_auc"], "lower_95") >= 0.68
        evaluated = {**row, "cluster_bootstrap": bootstrap, "bootstrap_gate_pass": passed}
        inspected.append(evaluated)
        if passed:
            selected = evaluated
            break
    return {
        "protocol": PROTOCOL,
        "raw_feature_count": len(RAW_FEATURE_NAMES),
        "direction_count": len(directions),
        "eligible_direction_count": len(entrants),
        "retained_directions": retained,
        "preeligible_combination_count": len(preliminary),
        "preeligible_top_50": preliminary[:50],
        "bootstrap_inspected": inspected,
        "selected": selected,
        "discovery_pass": selected is not None,
        "opened_data_discovery": True,
        "scientific_success_claim": False,
    }


def build_search(
    *, next559_dir: Path, next560_dir: Path, next561_dir: Path,
    design_path: Path, output_dir: Path, workers: int = 8,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, object]:
    root559, root560, root561 = map(
        lambda value: Path(value).resolve(), (next559_dir, next560_dir, next561_dir)
    )
    design_path, target = Path(design_path).resolve(), Path(output_dir).resolve()
    paths = {
        "design": design_path,
        "next559_manifest": root559 / n559.MANIFEST_NAME,
        "next559_table": root559 / n559.TABLE_NAME,
        "next560_manifest": root560 / n560.MANIFEST_NAME,
        "next560_geometry": root560 / n560.GEOMETRY_NAME,
        "next561_manifest": root561 / n561.MANIFEST_NAME,
        "next561_table": root561 / n561.TABLE_NAME,
        "next552_source": Path(n552.__file__).resolve(),
        "analytic_bank_source": Path(n552.compute_analytic_feature_row.__code__.co_filename).resolve(),
        "primitive_feature_source": Path(n552.primitive_geometry_features.__code__.co_filename).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT562 input is missing")
    if _sha256(design_path) != DESIGN_SHA256:
        raise ValueError("NEXT562 design identity differs")
    manifest559 = json.loads(paths["next559_manifest"].read_text())
    manifest560 = json.loads(paths["next560_manifest"].read_text())
    manifest561 = json.loads(paths["next561_manifest"].read_text())
    if (
        manifest559.get("protocol") != n559.PROTOCOL
        or manifest559.get("outputs_sha256", {}).get(n559.TABLE_NAME) != _sha256(paths["next559_table"])
        or manifest560.get("protocol") != n560.PROTOCOL
        or manifest560.get("outputs_sha256", {}).get(n560.GEOMETRY_NAME) != _sha256(paths["next560_geometry"])
        or manifest561.get("protocol") != n561.PROTOCOL
        or manifest561.get("confirmation_pass") is not False
        or manifest561.get("next560_endpoint_values_opened") is not True
        or manifest561.get("outputs_sha256", {}).get(n561.TABLE_NAME) != _sha256(paths["next561_table"])
    ):
        raise ValueError("NEXT562 opened upstream identity differs")

    old = pd.read_parquet(paths["next559_table"])
    old["development_stratum"] = "old_" + old["partition"].astype(str)
    new = pd.read_parquet(paths["next561_table"])
    new["development_stratum"] = new["replication_stratum"].astype(str)
    wanted = set(new["fid"].astype(str))
    with zipfile.ZipFile(paths["next560_geometry"]) as archive:
        payloads = [(Path(name).stem, archive.read(name)) for name in archive.namelist()]
    if len(payloads) != len(wanted) or {fid for fid, _ in payloads} != wanted:
        raise ValueError("NEXT562 NEXT560 geometry identity differs")
    payloads.sort(key=lambda item: item[0])
    computed = pd.DataFrame(n552._compute_many(payloads, workers))
    existing_feature_columns = set(n552.FEATURE_NAMES) & set(new)
    if existing_feature_columns:
        for feature in existing_feature_columns:
            joined = new[["fid", feature]].merge(computed[["fid", feature]], on="fid")
            if not np.allclose(joined[f"{feature}_x"], joined[f"{feature}_y"], equal_nan=True):
                raise ValueError(f"NEXT562 recomputed feature differs: {feature}")
        new = new.drop(columns=sorted(existing_feature_columns))
    new = new.merge(computed, on="fid", validate="one_to_one")
    if int(new["next552_failure"].notna().sum()) != 0:
        raise RuntimeError("NEXT562 NEXT560 analytic feature failures")
    keep = [
        "fid", "reduced_formula", "chemical_system", "size_family",
        "development_stratum", "dft_waste", "waste_severity", "protected",
        *n552.FEATURE_NAMES,
    ]
    table = pd.concat([old[keep], new[keep]], ignore_index=True)
    comp = pd.DataFrame(
        [composition_features(formula) for formula in table["reduced_formula"].astype(str)]
    )
    table = pd.concat([table.reset_index(drop=True), comp], axis=1)
    if len(table) != 4_400 or table["fid"].duplicated().any():
        raise ValueError("NEXT562 combined opened cohort differs")
    result = run_bounded_search(table, bootstrap_draws=bootstrap_draws)
    selected = result["selected"]
    formula = {
        "protocol": PROTOCOL,
        "name": "stable analytic risk union" if selected else None,
        "short_name": "SARU" if selected else None,
        "terms": selected["terms"] if selected else [],
        "operation": selected["operation"] if selected else None,
        "normalization": "full-candidate-batch midrank risks",
        "endpoint_fitted_coefficients": False,
        "dft_inputs_at_execution": False,
        "next563_endpoint_values_opened": False,
        "discovery_pass": selected is not None,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    try:
        table_path, search_path, formula_path = (
            staging / TABLE_NAME, staging / SEARCH_NAME, staging / FORMULA_NAME
        )
        table.to_parquet(table_path, index=False)
        search_path.write_bytes(_json_bytes(result))
        formula_path.write_bytes(_json_bytes(formula))
        outputs = {
            TABLE_NAME: _sha256(table_path), SEARCH_NAME: _sha256(search_path),
            FORMULA_NAME: _sha256(formula_path),
        }
        manifest = {
            "protocol": PROTOCOL,
            "discovery_pass": selected is not None,
            "inputs_sha256": {
                name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()
            },
            "outputs_sha256": outputs,
            "executed_source_sha256": {
                "src/next562_hea_stable_analytic_union_search.py": source_hash
            },
            "opened_data_discovery": True,
            "dft_values_used_by_executable_formula": False,
            "model_or_proxy_potential_used": False,
            "coordinates_or_cell_modified_by_formula": False,
            "next563_endpoint_values_opened": False,
            "next563_cohort_authorized": selected is not None,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT562 source changed before publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next559-dir", required=True, type=Path)
    parser.add_argument("--next560-dir", required=True, type=Path)
    parser.add_argument("--next561-dir", required=True, type=Path)
    parser.add_argument("--design-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args(argv)
    manifest = build_search(
        next559_dir=args.next559_dir, next560_dir=args.next560_dir,
        next561_dir=args.next561_dir, design_path=args.design_path,
        output_dir=args.output_dir, workers=args.workers,
        bootstrap_draws=args.bootstrap_draws,
    )
    print(json.dumps(_json_ready(manifest), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_search", "combine_risks", "composition_features", "run_bounded_search"]
