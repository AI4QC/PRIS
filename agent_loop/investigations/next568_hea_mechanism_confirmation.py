#!/usr/bin/env python3
"""Open NEXT567 endpoints once and confirm or reject selected MEMAX."""

from __future__ import annotations

import argparse
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

from src.next16_elementa_basin_hull import pauling_control
from src.next19_feature_build import _publish_directory_no_replace, _sha256
import src.next551_hea_initial_cohort as n551
import src.next553_hea_development_search as n553
import src.next555_hea_extreme_waste_search as n555
import src.next558_hea_packing_deficit_validation as n558
import src.next565_hea_mechanism_formula_family as n565
import src.next566b_hea_mechanism_selection as n566b
import src.next567_hea_mechanism_confirmation_cohort as n567


PROTOCOL = "2026-08-14-next568-hea-mechanism-confirmation-v1"
BOOTSTRAP_DRAWS = 5_000
BOOTSTRAP_SEED = 568_202_608
TABLE_NAME = "next568_hea_mechanism_confirmation.parquet"
RESULT_NAME = "NEXT568_HEA_MECHANISM_CONFIRMATION.json"
MANIFEST_NAME = "MANIFEST.json"


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


def evaluate_confirmation(
    table: pd.DataFrame, selected: str, *, bootstrap_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    if selected not in n565.CANDIDATE_NAMES:
        raise ValueError("NEXT568 selected formula differs")
    components = n566b.COMPONENTS[selected]
    required = {
        "fid", "chemical_system", "size_family", "dft_waste", "waste_severity",
        "protected", "pauling_risk", selected, *components,
    }
    if required - set(table) or table["fid"].duplicated().any():
        raise ValueError("NEXT568 confirmation table differs")
    labels = table["dft_waste"].to_numpy(bool)
    families = table["size_family"].astype(str)
    masks = {
        "overall": np.ones(len(table), dtype=bool),
        "ordered": families.eq("ordered").to_numpy(),
        "sqs": families.eq("sqs").to_numpy(),
    }
    score = pd.to_numeric(table[selected], errors="coerce").to_numpy(float)
    metrics = {
        name: n553._score_metrics(table, score, mask) for name, mask in masks.items()
    }
    component_metrics = {
        component: n553._score_metrics(
            table, pd.to_numeric(table[component], errors="coerce").to_numpy(float),
            masks["overall"],
        )
        for component in components
    }
    bootstrap = n558._cluster_bootstrap(table, score, draws=bootstrap_draws, seed=seed)
    pauling_score = pd.to_numeric(table["pauling_risk"], errors="coerce").to_numpy(float)
    pauling_supported = np.isfinite(pauling_score)
    pauling = n553._score_metrics(table, pauling_score, masks["overall"])
    common = n553._score_metrics(table, score, pauling_supported)
    overall = metrics["overall"]
    overall_auc = _number(overall, "roc_auc")
    component_margins = {
        component: overall_auc - _number(value, "roc_auc")
        for component, value in component_metrics.items()
    }
    top = overall.get("top_15_percent", {})
    top_fraction = float(top.get("protected", math.inf)) / max(int(top.get("rows", 0)), 1)
    pauling_clause = (
        _number(common, "roc_auc") >= _number(pauling, "roc_auc") + 0.05
        if int(pauling_supported.sum()) >= 100
        else _number(overall, "coverage") >= 0.99
        and _number(pauling, "coverage", default=math.inf) <= 0.25
    )
    clauses = {
        "rows_each_class_at_least_200": int(labels.sum()) >= 200 and int((~labels).sum()) >= 200,
        "protected_at_least_75": int(table["protected"].sum()) >= 75,
        "coverage_at_least_0p99": _number(overall, "coverage") >= 0.99,
        "overall_auc_at_least_0p72": overall_auc >= 0.72,
        "auc_cluster_lower_at_least_0p68": _number(bootstrap["roc_auc"], "lower_95") >= 0.68,
        "ordered_auc_at_least_0p66": _number(metrics["ordered"], "roc_auc") >= 0.66,
        "sqs_auc_at_least_0p66": _number(metrics["sqs"], "roc_auc") >= 0.66,
        "spearman_at_least_0p35": _number(overall, "spearman_severity") >= 0.35,
        "spearman_cluster_lower_at_least_0p25": (
            _number(bootstrap["spearman"], "lower_95") >= 0.25
        ),
        "top15_lift_at_least_1p50": _number(top, "lift") >= 1.50,
        "top15_protected_fraction_at_most_0p02": top_fraction <= 0.02,
        "component_margins_at_least_0p02": all(
            margin >= 0.02 for margin in component_margins.values()
        ),
        "pauling_comparator_gate": bool(pauling_clause),
    }
    passed = bool(all(clauses.values()))
    return {
        "protocol": PROTOCOL,
        "selected_candidate": selected,
        "counts": {
            "rows": len(table), "positive": int(labels.sum()), "negative": int((~labels).sum()),
            "protected": int(table["protected"].sum()),
            "chemical_systems": int(table["chemical_system"].nunique()),
            "pauling_supported": int(pauling_supported.sum()),
        },
        "metrics": metrics,
        "component_metrics": component_metrics,
        "component_auc_margins": component_margins,
        "cluster_bootstrap": bootstrap,
        "pauling_control": pauling,
        "candidate_on_pauling_common_support": common,
        "top15_protected_fraction": top_fraction,
        "confirmation_clauses": clauses,
        "confirmation_pass": passed,
        "scientific_success_claim": passed,
        "claim_boundary": "same-source HEA extreme DFT-waste triage on new identities",
    }


def build_confirmation(
    *, next567_dir: Path, next566b_dir: Path, source_csv: Path,
    design_path: Path, output_dir: Path, workers: int = 8,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, object]:
    root567, root566b = Path(next567_dir).resolve(), Path(next566b_dir).resolve()
    source_csv, design_path, target = map(
        lambda value: Path(value).resolve(), (source_csv, design_path, output_dir)
    )
    paths = {
        "design": design_path,
        "source_csv": source_csv,
        "next567_manifest": root567 / n567.MANIFEST_NAME,
        "next567_metadata": root567 / n567.METADATA_NAME,
        "next567_geometry": root567 / n567.GEOMETRY_NAME,
        "next567_predictions": root567 / n567.PREDICTIONS_NAME,
        "next567_source": Path(n567.__file__).resolve(),
        "next566b_manifest": root566b / n566b.MANIFEST_NAME,
        "selected_formula": root566b / n566b.SELECTED_FORMULA_NAME,
        "pauling_source": Path(pauling_control.__code__.co_filename).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT568 input is missing")
    if _sha256(source_csv) != n551.SOURCE_SHA256 or _sha256(design_path) != n565.DESIGN_SHA256:
        raise ValueError("NEXT568 formal source or design differs")
    manifest567 = json.loads(paths["next567_manifest"].read_text())
    outputs567 = manifest567.get("outputs_sha256", {})
    selected = manifest567.get("selected_candidate")
    if (
        manifest567.get("protocol") != n567.PROTOCOL
        or manifest567.get("next568_endpoint_opening_authorized") is not True
        or manifest567.get("dft_values_opened") is not False
        or manifest567.get("gates", {}).get("passes") is not True
        or selected not in n565.CANDIDATE_NAMES
        or any(outputs567.get(name) != _sha256(root567 / name) for name in (
            n567.METADATA_NAME, n567.GEOMETRY_NAME, n567.PREDICTIONS_NAME
        ))
        or manifest567.get("executed_source_sha256", {}).get(
            "src/next567_hea_mechanism_confirmation_cohort.py"
        ) != _sha256(paths["next567_source"])
    ):
        raise ValueError("NEXT568 frozen prediction identity differs")
    metadata = pd.read_parquet(paths["next567_metadata"])
    predictions = pd.read_parquet(paths["next567_predictions"])
    fids = set(metadata["fid"].astype(str))
    if len(fids) != n567.EXPECTED_ROWS or set(predictions["fid"].astype(str)) != fids:
        raise ValueError("NEXT568 frozen cohort identity differs")
    initial: dict[str, Atoms] = {}
    with zipfile.ZipFile(paths["next567_geometry"]) as archive:
        for name in archive.namelist():
            fid = Path(name).stem
            if fid in fids:
                initial[fid] = n553._decode_initial(archive.read(name))
    if set(initial) != fids:
        raise ValueError("NEXT568 x0 geometry identity differs")
    pauling_rows = pd.DataFrame(
        n558._pauling_many([(fid, initial[fid]) for fid in sorted(fids)], workers)
    )
    endpoint_payloads, firewall = n553.extract_authorized_endpoint_payloads(source_csv, fids)
    endpoint_rows = pd.DataFrame(
        [n553._endpoint_row(fid, initial[fid], endpoint_payloads[fid]) for fid in sorted(fids)]
    )
    endpoints = n555.apply_extreme_waste_endpoint(
        metadata.merge(endpoint_rows, on="fid", validate="one_to_one")
    )
    endpoint_columns = [
        "fid", "e_above_hull", "disp_p90", "cell_logstrain_max", "volume_logchange",
        "dft_waste", "waste_severity", "protected",
    ]
    table = predictions.merge(
        endpoints[endpoint_columns], on="fid", validate="one_to_one"
    ).merge(pauling_rows, on="fid", validate="one_to_one")
    if len(table) != n567.EXPECTED_ROWS:
        raise ValueError("NEXT568 endpoint join differs")
    result = evaluate_confirmation(table, str(selected), bootstrap_draws=bootstrap_draws)
    result["endpoint_firewall"] = firewall

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    try:
        table_path, result_path = staging / TABLE_NAME, staging / RESULT_NAME
        table.to_parquet(table_path, index=False)
        result_path.write_bytes(_json_bytes(result))
        outputs = {TABLE_NAME: _sha256(table_path), RESULT_NAME: _sha256(result_path)}
        manifest = {
            "protocol": PROTOCOL,
            "selected_candidate": selected,
            "confirmation_pass": result["confirmation_pass"],
            "inputs_sha256": {
                name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()
            },
            "outputs_sha256": outputs,
            "executed_source_sha256": {
                "src/next568_hea_mechanism_confirmation.py": source_hash
            },
            "next567_endpoint_values_opened": True,
            "next567_final_structures_opened": True,
            "unauthorized_endpoint_rows_materialized": 0,
            "dft_values_used_by_executable_formula": False,
            "model_or_proxy_potential_used": False,
            "coordinates_or_cell_modified_by_formula": False,
            "independent_report_authorized": result["confirmation_pass"],
            "canonical_report_or_paper_edits_authorized": False,
            "scientific_improvement_claim": result["confirmation_pass"],
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next567-dir", required=True, type=Path)
    parser.add_argument("--next566b-dir", required=True, type=Path)
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--design-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args(argv)
    manifest = build_confirmation(
        next567_dir=args.next567_dir, next566b_dir=args.next566b_dir,
        source_csv=args.source_csv, design_path=args.design_path,
        output_dir=args.output_dir, workers=args.workers,
        bootstrap_draws=args.bootstrap_draws,
    )
    print(json.dumps(_json_ready(manifest), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_confirmation", "evaluate_confirmation"]
