#!/usr/bin/env python3
"""Open HEA development endpoints only and run the frozen bounded analytic search."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

from ase import Atoms
from ase.io import read
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.next19_feature_build import _publish_directory_no_replace, _sha256
import src.next551_hea_initial_cohort as n551
import src.next552_hea_analytic_feature_freeze as n552


PROTOCOL = "2026-08-13-next553-hea-development-search-v1"
ENERGY_HULL_THRESHOLD = 0.10
DISPLACEMENT_P90_THRESHOLD = 0.25
CELL_LOGSTRAIN_THRESHOLD = 0.08
VOLUME_LOGCHANGE_THRESHOLD = 0.10
PROTECTED_ENERGY_MAX = 0.025
PROTECTED_DISPLACEMENT_MAX = 0.10
PROTECTED_CELL_LOGSTRAIN_MAX = 0.03
PROTECTED_VOLUME_LOGCHANGE_MAX = 0.04
PAIR_FORMULAS = ("mean", "maximum", "union", "minimum")
MINIMUM_ENDPOINT_COVERAGE = 0.95
MINIMUM_CLASS_COUNT = 100
MINIMUM_FAMILY_CLASS_COUNT = 30
MINIMUM_DIRECTION_FAMILY_AUC = 0.60
MAXIMUM_RETAINED = 16
MAXIMUM_REDUNDANCY = 0.95
MINIMUM_PAIR_COVERAGE = 0.95
MINIMUM_PAIR_AUC = 0.70
MINIMUM_PAIR_FAMILY_AUC = 0.65
MINIMUM_PAIR_BOOTSTRAP_LOWER = 0.62
MINIMUM_TOP_LIFT = 1.75
MINIMUM_SPEARMAN = 0.30
MINIMUM_COMPONENT_MARGIN = 0.02
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 553_202_608
ENDPOINT_TABLE_NAME = "next553_hea_development_endpoints.parquet"
SEARCH_NAME = "NEXT553_HEA_DEVELOPMENT_SEARCH.json"
FORMULA_NAME = "NEXT553_FROZEN_FORMULA.json"
MANIFEST_NAME = "MANIFEST.json"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def extract_authorized_endpoint_payloads(
    source_csv: Path, authorized_fids: set[str]
) -> tuple[dict[str, dict[str, bytes]], dict[str, object]]:
    """Materialize endpoint fields only for explicitly authorized FIDs."""

    source_csv = Path(source_csv)
    result: dict[str, dict[str, bytes]] = {}
    source_rows = 0
    with source_csv.open("rb") as handle:
        indices = n551._header_indices(handle)
        fid_index = indices["fid"]
        endpoint_indices = {
            "e_above_hull": indices["e_above_hull"],
            "structure_as_dict": indices["structure_as_dict"],
        }
        for record in handle:
            source_rows += 1
            identity, _present, field_count = n551._project_csv_record(
                record, copy_indices={fid_index}
            )
            if field_count != len(n551.EXPECTED_HEADER):
                raise ValueError("NEXT553 source field count differs")
            fid = identity[fid_index].decode("utf-8")
            if fid not in authorized_fids:
                continue
            wanted = {fid_index, *endpoint_indices.values()}
            projected, _endpoint_presence, repeated_count = n551._project_csv_record(
                record, copy_indices=wanted
            )
            if repeated_count != len(n551.EXPECTED_HEADER) or fid in result:
                raise ValueError("NEXT553 authorized endpoint identity differs")
            result[fid] = {
                name: projected[index] for name, index in endpoint_indices.items()
            }
    missing = sorted(authorized_fids - set(result))
    if missing:
        raise ValueError(f"NEXT553 authorized endpoint rows are missing: {missing[:5]}")
    return result, {
        "source_rows_scanned": source_rows,
        "authorized_fids": len(authorized_fids),
        "endpoint_rows_materialized": len(result),
        "unauthorized_endpoint_rows_materialized": 0,
        "unauthorized_endpoint_fields_copied_or_decoded": False,
    }


def _decode_initial(payload: bytes) -> Atoms:
    atoms = read(io.StringIO(payload.decode("utf-8")), format="extxyz", index=0)
    if (
        not isinstance(atoms, Atoms)
        or len(atoms) < 1
        or not np.all(atoms.pbc)
        or atoms.calc is not None
        or atoms.info
        or set(atoms.arrays) != {"numbers", "positions"}
    ):
        raise ValueError("NEXT553 initial geometry firewall differs")
    return atoms


def _decode_final(payload: bytes, fid: str) -> Atoms:
    try:
        value = ast.literal_eval(payload.decode("utf-8"))
        structure = Structure.from_dict(value)
        atoms = AseAtomsAdaptor.get_atoms(structure)
    except Exception as exc:
        raise ValueError(f"NEXT553 final structure encoding differs: {fid}") from exc
    clean = Atoms(
        numbers=np.asarray(atoms.numbers, dtype=int),
        positions=np.asarray(atoms.positions, dtype=float),
        cell=np.asarray(atoms.cell.array, dtype=float),
        pbc=True,
    )
    if (
        len(clean) < 1
        or not np.all(clean.pbc)
        or not np.isfinite(clean.positions).all()
        or not np.isfinite(clean.cell.array).all()
        or abs(float(np.linalg.det(clean.cell.array))) <= 1.0e-10
    ):
        raise ValueError(f"NEXT553 final structure geometry differs: {fid}")
    return clean


def _endpoint_row(fid: str, initial: Atoms, payload: dict[str, bytes]) -> dict[str, object]:
    final = _decode_final(payload["structure_as_dict"], fid)
    if len(initial) != len(final) or not np.array_equal(initial.numbers, final.numbers):
        raise ValueError(f"NEXT553 initial/final site order differs: {fid}")
    try:
        e_above_hull = float(payload["e_above_hull"].decode("utf-8"))
    except ValueError as exc:
        raise ValueError(f"NEXT553 e_above_hull differs: {fid}") from exc
    cell0 = np.asarray(initial.cell.array, dtype=float)
    cell1 = np.asarray(final.cell.array, dtype=float)
    frac0 = np.asarray(initial.positions, dtype=float) @ np.linalg.inv(cell0)
    frac1 = np.asarray(final.positions, dtype=float) @ np.linalg.inv(cell1)
    delta = frac1 - frac0
    delta -= np.rint(delta)
    displacement = np.linalg.norm(delta @ ((cell0 + cell1) / 2.0), axis=1)
    singular = np.linalg.svd(cell1 @ np.linalg.inv(cell0), compute_uv=False)
    values = {
        "fid": fid,
        "e_above_hull": e_above_hull,
        "disp_rms": float(np.sqrt(np.mean(displacement**2))),
        "disp_p90": float(np.quantile(displacement, 0.90)),
        "disp_max": float(displacement.max()),
        "cell_logstrain_max": float(np.max(np.abs(np.log(singular)))),
        "volume_logchange": float(
            abs(math.log(abs(np.linalg.det(cell1)) / abs(np.linalg.det(cell0))))
        ),
    }
    if not np.isfinite(list(values.values())[1:]).all():
        raise ValueError(f"NEXT553 endpoint is nonfinite: {fid}")
    return values


def apply_frozen_endpoint_definition(table: pd.DataFrame) -> pd.DataFrame:
    required = {
        "fid", "e_above_hull", "disp_p90", "cell_logstrain_max", "volume_logchange"
    }
    if required - set(table):
        raise ValueError("NEXT553 endpoint table differs")
    result = table.copy()
    energy = pd.to_numeric(result["e_above_hull"], errors="coerce").to_numpy(float)
    displacement = pd.to_numeric(result["disp_p90"], errors="coerce").to_numpy(float)
    strain = pd.to_numeric(result["cell_logstrain_max"], errors="coerce").to_numpy(float)
    volume = pd.to_numeric(result["volume_logchange"], errors="coerce").to_numpy(float)
    if not np.isfinite(np.column_stack([energy, displacement, strain, volume])).all():
        raise ValueError("NEXT553 endpoint definition received nonfinite values")
    result["energetically_unstable"] = energy >= ENERGY_HULL_THRESHOLD
    result["large_geometric_response"] = (
        (displacement >= DISPLACEMENT_P90_THRESHOLD)
        | (strain >= CELL_LOGSTRAIN_THRESHOLD)
        | (volume >= VOLUME_LOGCHANGE_THRESHOLD)
    )
    result["dft_waste"] = (
        result["energetically_unstable"].to_numpy(bool)
        | result["large_geometric_response"].to_numpy(bool)
    )
    result["waste_severity"] = np.max(
        np.column_stack(
            [
                energy / ENERGY_HULL_THRESHOLD,
                displacement / DISPLACEMENT_P90_THRESHOLD,
                strain / CELL_LOGSTRAIN_THRESHOLD,
                volume / VOLUME_LOGCHANGE_THRESHOLD,
            ]
        ),
        axis=1,
    )
    result["protected"] = (
        (energy <= PROTECTED_ENERGY_MAX)
        & (displacement <= PROTECTED_DISPLACEMENT_MAX)
        & (strain <= PROTECTED_CELL_LOGSTRAIN_MAX)
        & (volume <= PROTECTED_VOLUME_LOGCHANGE_MAX)
    )
    return result


def symmetric_pair_score(u: object, v: object, formula: str) -> np.ndarray:
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    if u.shape != v.shape:
        raise ValueError("NEXT553 pair arrays differ")
    if formula == "mean":
        return (u + v) / 2.0
    if formula == "maximum":
        return np.maximum(u, v)
    if formula == "union":
        return 1.0 - (1.0 - u) * (1.0 - v)
    if formula == "minimum":
        return np.minimum(u, v)
    raise ValueError("NEXT553 pair formula differs")


def _score_metrics(table: pd.DataFrame, score: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    score = np.asarray(score, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    supported = mask & np.isfinite(score)
    work = table.loc[supported]
    values = score[supported]
    labels = work["dft_waste"].to_numpy(bool)
    denominator = int(mask.sum())
    if len(work) < 2 or np.unique(labels).size != 2:
        return {"rows": len(work), "coverage": len(work) / max(denominator, 1)}
    ids = work["fid"].astype(str).to_numpy()
    order = np.lexsort((ids, -values))
    top_n = max(1, math.ceil(0.15 * len(order)))
    top = order[:top_n]
    prevalence = float(labels.mean())
    return {
        "rows": len(work),
        "coverage": len(work) / denominator,
        "positives": int(labels.sum()),
        "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(labels, values)),
        "spearman_severity": float(
            spearmanr(values, work["waste_severity"].to_numpy(float)).statistic
        ),
        "top_15_percent": {
            "rows": top_n,
            "positives": int(labels[top].sum()),
            "precision": float(labels[top].mean()),
            "lift": float(labels[top].mean() / prevalence),
            "recall": float(labels[top].sum() / labels.sum()),
            "protected": int(work["protected"].to_numpy(bool)[top].sum()),
        },
    }


def _cluster_bootstrap_auc(
    table: pd.DataFrame, score: np.ndarray, *, draws: int, seed: int
) -> dict[str, object]:
    score = np.asarray(score, dtype=float)
    systems = table["chemical_system"].astype(str).to_numpy()
    unique = np.asarray(sorted(set(systems)), dtype=object)
    groups = {system: np.flatnonzero(systems == system) for system in unique}
    labels = table["dft_waste"].to_numpy(bool)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(draws):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([groups[system] for system in chosen])
        finite = np.isfinite(score[index])
        index = index[finite]
        if len(index) >= 2 and np.unique(labels[index]).size == 2:
            values.append(float(roc_auc_score(labels[index], score[index])))
    array = np.asarray(values, dtype=float)
    return {
        "draws": draws,
        "seed": seed,
        "valid": len(values),
        "clusters": len(unique),
        "lower_95": float(np.quantile(array, 0.025)) if len(array) else math.nan,
        "median": float(np.quantile(array, 0.5)) if len(array) else math.nan,
        "upper_95": float(np.quantile(array, 0.975)) if len(array) else math.nan,
    }


def run_bounded_search(
    table: pd.DataFrame, catalogue: dict[str, object], *, bootstrap_draws: int = BOOTSTRAP_DRAWS
) -> dict[str, object]:
    if not isinstance(catalogue.get("features"), list):
        raise ValueError("NEXT553 feature catalogue differs")
    development = table["partition"].astype(str).eq("development").to_numpy()
    if not development.all():
        raise ValueError("NEXT553 search table contains non-development rows")
    families = table["size_family"].astype(str).to_numpy()
    masks = {
        "overall": np.ones(len(table), dtype=bool),
        "ordered": families == "ordered",
        "sqs": families == "sqs",
    }
    directions: list[dict[str, object]] = []
    scores: dict[str, np.ndarray] = {}
    for row in catalogue["features"]:
        if row.get("searchable") is not True:
            continue
        feature = str(row["feature"])
        for direction in ("high", "low"):
            key = f"{feature}__risk_{direction}"
            score = pd.to_numeric(table[key], errors="coerce").to_numpy(float)
            scores[key] = score
            metrics = {name: _score_metrics(table, score, mask) for name, mask in masks.items()}
            enters = bool(
                metrics["ordered"].get("roc_auc", -math.inf) >= MINIMUM_DIRECTION_FAMILY_AUC
                and metrics["sqs"].get("roc_auc", -math.inf) >= MINIMUM_DIRECTION_FAMILY_AUC
            )
            directions.append(
                {"key": key, "feature": feature, "direction": direction, "metrics": metrics,
                 "enters_pair_search": enters}
            )
    entrants = [row for row in directions if row["enters_pair_search"]]
    entrants.sort(
        key=lambda row: (
            -min(row["metrics"]["ordered"]["roc_auc"], row["metrics"]["sqs"]["roc_auc"]),
            -row["metrics"]["overall"]["roc_auc"],
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
                metrics = {name: _score_metrics(table, score, mask) for name, mask in masks.items()}
                margins = {}
                for family in ("ordered", "sqs"):
                    margins[family] = metrics[family].get("roc_auc", -math.inf) - max(
                        first["metrics"][family].get("roc_auc", -math.inf),
                        second["metrics"][family].get("roc_auc", -math.inf),
                    )
                overall = metrics["overall"]
                preeligible = bool(
                    overall.get("coverage", 0.0) >= MINIMUM_PAIR_COVERAGE
                    and overall.get("roc_auc", -math.inf) >= MINIMUM_PAIR_AUC
                    and metrics["ordered"].get("roc_auc", -math.inf) >= MINIMUM_PAIR_FAMILY_AUC
                    and metrics["sqs"].get("roc_auc", -math.inf) >= MINIMUM_PAIR_FAMILY_AUC
                    and overall.get("spearman_severity", -math.inf) >= MINIMUM_SPEARMAN
                    and overall.get("top_15_percent", {}).get("lift", -math.inf) >= MINIMUM_TOP_LIFT
                    and overall.get("top_15_percent", {}).get("protected", math.inf) == 0
                    and margins["ordered"] >= MINIMUM_COMPONENT_MARGIN
                    and margins["sqs"] >= MINIMUM_COMPONENT_MARGIN
                )
                bootstrap = None
                eligible = False
                if preeligible:
                    bootstrap = _cluster_bootstrap_auc(
                        table, score, draws=bootstrap_draws,
                        seed=BOOTSTRAP_SEED + len(pairs),
                    )
                    eligible = bootstrap["lower_95"] >= MINIMUM_PAIR_BOOTSTRAP_LOWER
                pairs.append(
                    {
                        "formula_id": f"{formula}({first['key']},{second['key']})",
                        "first": first["key"],
                        "second": second["key"],
                        "formula": formula,
                        "metrics": metrics,
                        "component_margins": margins,
                        "preeligible": preeligible,
                        "cluster_bootstrap_auc": bootstrap,
                        "eligible": eligible,
                    }
                )
    eligible = [row for row in pairs if row["eligible"]]
    eligible.sort(
        key=lambda row: (
            -min(row["metrics"]["ordered"]["roc_auc"], row["metrics"]["sqs"]["roc_auc"]),
            -row["metrics"]["overall"]["roc_auc"],
            -row["metrics"]["overall"]["spearman_severity"],
            row["formula_id"],
        )
    )
    return {
        "protocol": PROTOCOL,
        "searchable_direction_count": len(directions),
        "univariate_entrant_count": len(entrants),
        "retained_direction_count": len(retained),
        "retained_directions": retained,
        "pair_candidate_count": len(pairs),
        "pair_candidates": pairs,
        "eligible_pair_count": len(eligible),
        "winner": eligible[0] if eligible else None,
        "validation_endpoints_opened": False,
    }


def _endpoint_class_gates(table: pd.DataFrame) -> dict[str, object]:
    labels = table["dft_waste"].to_numpy(bool)
    families: dict[str, object] = {}
    for family in ("ordered", "sqs"):
        subset = table.loc[table["size_family"].astype(str).eq(family)]
        families[family] = {
            "rows": len(subset),
            "positive": int(subset["dft_waste"].sum()),
            "negative": int((~subset["dft_waste"]).sum()),
        }
    result = {
        "rows": len(table),
        "positive": int(labels.sum()),
        "negative": int((~labels).sum()),
        "protected": int(table["protected"].sum()),
        "families": families,
    }
    result["passes"] = bool(
        len(table) >= MINIMUM_ENDPOINT_COVERAGE * 1_200
        and result["positive"] >= MINIMUM_CLASS_COUNT
        and result["negative"] >= MINIMUM_CLASS_COUNT
        and all(
            families[family]["positive"] >= MINIMUM_FAMILY_CLASS_COUNT
            and families[family]["negative"] >= MINIMUM_FAMILY_CLASS_COUNT
            for family in ("ordered", "sqs")
        )
    )
    return result


def build_development_search(
    *, next551_dir: Path, next552_dir: Path, source_csv: Path, design_path: Path,
    output_dir: Path, bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, object]:
    root551 = Path(next551_dir).resolve()
    root552 = Path(next552_dir).resolve()
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
        "next552_catalogue": root552 / n552.CATALOGUE_NAME,
        "next551_source": Path(n551.__file__).resolve(),
        "next552_source": Path(n552.__file__).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT553 input is missing")
    if _sha256(design_path) != n551.DESIGN_SHA256 or _sha256(source_csv) != n551.SOURCE_SHA256:
        raise ValueError("NEXT553 formal source or design identity differs")
    manifest552 = json.loads(paths["next552_manifest"].read_text())
    outputs552 = manifest552.get("outputs_sha256")
    if (
        manifest552.get("protocol") != n552.PROTOCOL
        or manifest552.get("endpoint_values_opened") is not False
        or manifest552.get("final_or_relaxed_structures_opened") is not False
        or manifest552.get("next553_development_endpoint_opening_authorized") is not True
        or not isinstance(outputs552, dict)
        or outputs552.get(n552.TABLE_NAME) != _sha256(paths["next552_table"])
        or outputs552.get(n552.CATALOGUE_NAME) != _sha256(paths["next552_catalogue"])
    ):
        raise ValueError("NEXT553 feature freeze identity differs")
    features = pd.read_parquet(paths["next552_table"])
    metadata = pd.read_parquet(paths["next551_metadata"])
    development = metadata.loc[metadata["partition"].astype(str).eq("development")].copy()
    validation_ids = set(
        metadata.loc[metadata["partition"].astype(str).eq("validation"), "fid"].astype(str)
    )
    development_ids = set(development["fid"].astype(str))
    if len(development_ids) != 1_200 or len(validation_ids) != 1_200 or development_ids & validation_ids:
        raise ValueError("NEXT553 frozen partition identity differs")
    endpoint_payloads, firewall = extract_authorized_endpoint_payloads(source_csv, development_ids)
    if firewall["source_rows_scanned"] != n551.EXPECTED_SOURCE_ROWS:
        raise ValueError("NEXT553 source row count differs")
    initial: dict[str, Atoms] = {}
    with zipfile.ZipFile(paths["next551_geometry"]) as archive:
        for name in archive.namelist():
            fid = Path(name).stem
            if fid in development_ids:
                initial[fid] = _decode_initial(archive.read(name))
    if set(initial) != development_ids:
        raise ValueError("NEXT553 development initial geometries differ")
    endpoint_rows = pd.DataFrame(
        [_endpoint_row(fid, initial[fid], endpoint_payloads[fid]) for fid in sorted(development_ids)]
    )
    endpoints = development.merge(endpoint_rows, on="fid", validate="one_to_one")
    endpoints = apply_frozen_endpoint_definition(endpoints)
    class_gates = _endpoint_class_gates(endpoints)
    if class_gates["passes"] is not True:
        search = {
            "protocol": PROTOCOL,
            "search_executed": False,
            "reason": "development endpoint class gates failed",
            "class_gates": class_gates,
            "winner": None,
            "validation_endpoints_opened": False,
        }
    else:
        search_table = features.merge(
            endpoints[
                ["fid", "e_above_hull", "disp_p90", "cell_logstrain_max",
                 "volume_logchange", "dft_waste", "waste_severity", "protected"]
            ],
            on="fid", validate="one_to_one",
        )
        if len(search_table) != len(development):
            raise ValueError("NEXT553 development feature-endpoint join differs")
        catalogue = json.loads(paths["next552_catalogue"].read_text())
        search = run_bounded_search(search_table, catalogue, bootstrap_draws=bootstrap_draws)
        search["search_executed"] = True
        search["class_gates"] = class_gates

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    try:
        endpoint_path = staging / ENDPOINT_TABLE_NAME
        search_path = staging / SEARCH_NAME
        endpoints.to_parquet(endpoint_path, index=False)
        search_path.write_bytes(_json_bytes(search))
        outputs_out = {ENDPOINT_TABLE_NAME: _sha256(endpoint_path), SEARCH_NAME: _sha256(search_path)}
        winner = search.get("winner")
        if isinstance(winner, dict):
            formula = {
                "protocol": PROTOCOL,
                "formula_id": winner["formula_id"],
                "first": winner["first"],
                "second": winner["second"],
                "formula": winner["formula"],
                "full_cohort_ranks_frozen_by_next552": True,
                "endpoint_fitted_coefficients": False,
                "validation_endpoints_opened": False,
            }
            formula_path = staging / FORMULA_NAME
            formula_path.write_bytes(_json_bytes(formula))
            outputs_out[FORMULA_NAME] = _sha256(formula_path)
        manifest_out = {
            "protocol": PROTOCOL,
            "class_gates": class_gates,
            "search_executed": bool(search.get("search_executed")),
            "eligible_pair_count": int(search.get("eligible_pair_count", 0)),
            "winner_frozen": isinstance(winner, dict),
            "next554_validation_opening_authorized": isinstance(winner, dict),
            "endpoint_firewall": firewall,
            "inputs_sha256": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in paths.items()
            },
            "outputs_sha256": outputs_out,
            "executed_source_sha256": {
                "src/next553_hea_development_search.py": source_hash
            },
            "development_endpoint_values_opened": True,
            "validation_endpoint_values_opened": False,
            "validation_final_structures_opened": False,
            "validation_endpoint_fields_copied_or_decoded": False,
            "dft_values_used_by_executable_formula": False,
            "model_or_proxy_potential_used": False,
            "coordinates_or_cell_modified_by_formula": False,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest_out))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT553 source changed before publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next551-dir", required=True, type=Path)
    parser.add_argument("--next552-dir", required=True, type=Path)
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--design-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args(argv)
    manifest = build_development_search(
        next551_dir=args.next551_dir,
        next552_dir=args.next552_dir,
        source_csv=args.source_csv,
        design_path=args.design_path,
        output_dir=args.output_dir,
        bootstrap_draws=args.bootstrap_draws,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "apply_frozen_endpoint_definition",
    "build_development_search",
    "extract_authorized_endpoint_payloads",
    "run_bounded_search",
    "symmetric_pair_score",
]
