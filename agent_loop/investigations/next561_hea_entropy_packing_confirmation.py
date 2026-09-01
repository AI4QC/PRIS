#!/usr/bin/env python3
"""Open NEXT560 endpoints once and confirm or reject the frozen EPCU law."""

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
import src.next559_hea_entropy_packing_discovery as n559
import src.next560_hea_entropy_packing_cohort as n560


PROTOCOL = "2026-08-14-next561-hea-entropy-packing-confirmation-v1"
BOOTSTRAP_DRAWS = 5_000
BOOTSTRAP_SEED = 561_202_608
TABLE_NAME = "next561_hea_entropy_packing_confirmation.parquet"
RESULT_NAME = "NEXT561_HEA_ENTROPY_PACKING_CONFIRMATION.json"
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


def _metric_number(metrics: dict[str, object], key: str, default: float = -math.inf) -> float:
    value = metrics.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def evaluate_confirmation(
    table: pd.DataFrame, *, bootstrap_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    required = {
        "fid", "chemical_system", "size_family", "replication_stratum",
        n559.SCORE, n559.ENTROPY_RISK, n559.PACKING_RISK,
        "dft_waste", "waste_severity", "protected", "pauling_risk",
    }
    if required - set(table) or table["fid"].duplicated().any():
        raise ValueError("NEXT561 confirmation table differs")
    score = pd.to_numeric(table[n559.SCORE], errors="coerce").to_numpy(float)
    entropy = pd.to_numeric(table[n559.ENTROPY_RISK], errors="coerce").to_numpy(float)
    packing = pd.to_numeric(table[n559.PACKING_RISK], errors="coerce").to_numpy(float)
    strata = table["replication_stratum"].astype(str)
    families = table["size_family"].astype(str)
    masks = {
        "overall": np.ones(len(table), dtype=bool),
        "unseen_chemical_system": strata.eq("unseen_chemical_system").to_numpy(),
        "new_identity_known_system": strata.eq("new_identity_known_system").to_numpy(),
        "ordered": families.eq("ordered").to_numpy(),
        "sqs": families.eq("sqs").to_numpy(),
    }
    candidate = {
        name: n553._score_metrics(table, score, mask) for name, mask in masks.items()
    }
    components = {
        "entropy": {
            name: n553._score_metrics(table, entropy, mask) for name, mask in masks.items()
        },
        "packing": {
            name: n553._score_metrics(table, packing, mask) for name, mask in masks.items()
        },
    }
    bootstrap = n558._cluster_bootstrap(
        table, score, draws=bootstrap_draws, seed=seed
    )
    pauling_score = pd.to_numeric(table["pauling_risk"], errors="coerce").to_numpy(float)
    pauling_supported = np.isfinite(pauling_score)
    pauling = n553._score_metrics(table, pauling_score, masks["overall"])
    candidate_common = n553._score_metrics(table, score, pauling_supported)

    labels = table["dft_waste"].to_numpy(bool)
    unseen_labels = labels[masks["unseen_chemical_system"]]
    overall = candidate["overall"]
    top = overall.get("top_15_percent", {})
    top_rows = int(top.get("rows", 0))
    top_protected_fraction = float(top.get("protected", math.inf)) / max(top_rows, 1)
    overall_auc = _metric_number(overall, "roc_auc")
    entropy_auc = _metric_number(components["entropy"]["overall"], "roc_auc")
    packing_auc = _metric_number(components["packing"]["overall"], "roc_auc")
    common_candidate_auc = _metric_number(candidate_common, "roc_auc")
    pauling_auc = _metric_number(pauling, "roc_auc")
    pauling_coverage = _metric_number(pauling, "coverage", default=math.inf)
    pauling_clause = (
        common_candidate_auc >= pauling_auc + 0.05
        if int(pauling_supported.sum()) >= 100
        else _metric_number(overall, "coverage") >= 0.99 and pauling_coverage <= 0.25
    )
    clauses = {
        "rows_each_class_at_least_100": int(labels.sum()) >= 100 and int((~labels).sum()) >= 100,
        "unseen_rows_each_class_at_least_30": (
            int(unseen_labels.sum()) >= 30 and int((~unseen_labels).sum()) >= 30
        ),
        "protected_at_least_50": int(table["protected"].sum()) >= 50,
        "coverage_at_least_0p99": _metric_number(overall, "coverage") >= 0.99,
        "overall_auc_at_least_0p72": overall_auc >= 0.72,
        "auc_cluster_lower_at_least_0p68": (
            _metric_number(bootstrap["roc_auc"], "lower_95") >= 0.68
        ),
        "unseen_auc_at_least_0p68": (
            _metric_number(candidate["unseen_chemical_system"], "roc_auc") >= 0.68
        ),
        "known_replication_auc_at_least_0p72": (
            _metric_number(candidate["new_identity_known_system"], "roc_auc") >= 0.72
        ),
        "ordered_auc_at_least_0p65": _metric_number(candidate["ordered"], "roc_auc") >= 0.65,
        "sqs_auc_at_least_0p65": _metric_number(candidate["sqs"], "roc_auc") >= 0.65,
        "spearman_at_least_0p35": _metric_number(overall, "spearman_severity") >= 0.35,
        "spearman_cluster_lower_at_least_0p25": (
            _metric_number(bootstrap["spearman"], "lower_95") >= 0.25
        ),
        "top15_lift_at_least_1p50": _metric_number(top, "lift") >= 1.50,
        "top15_protected_fraction_at_most_0p02": top_protected_fraction <= 0.02,
        "auc_margin_over_entropy_at_least_0p03": overall_auc >= entropy_auc + 0.03,
        "auc_margin_over_packing_at_least_0p03": overall_auc >= packing_auc + 0.03,
        "pauling_comparator_gate": bool(pauling_clause),
    }
    passed = bool(all(clauses.values()))
    return {
        "protocol": PROTOCOL,
        "counts": {
            "rows": len(table),
            "positive": int(labels.sum()),
            "negative": int((~labels).sum()),
            "unseen_positive": int(unseen_labels.sum()),
            "unseen_negative": int((~unseen_labels).sum()),
            "protected": int(table["protected"].sum()),
            "chemical_systems": int(table["chemical_system"].nunique()),
            "pauling_supported": int(pauling_supported.sum()),
        },
        "candidate": candidate,
        "components": components,
        "cluster_bootstrap": bootstrap,
        "pauling_control": pauling,
        "candidate_on_pauling_common_support": candidate_common,
        "top15_protected_fraction": top_protected_fraction,
        "confirmation_clauses": clauses,
        "confirmation_pass": passed,
        "endpoint_is_fixed_from_next555": True,
        "scientific_success_claim": passed,
        "claim_boundary": "HEA-domain extreme DFT-waste screening only",
    }


def build_confirmation(
    *, next560_dir: Path, next559_dir: Path, source_csv: Path,
    design_path: Path, output_dir: Path, workers: int = 8,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, object]:
    root560, root559 = Path(next560_dir).resolve(), Path(next559_dir).resolve()
    source_csv, design_path = Path(source_csv).resolve(), Path(design_path).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "design": design_path,
        "source_csv": source_csv,
        "next560_manifest": root560 / n560.MANIFEST_NAME,
        "next560_metadata": root560 / n560.METADATA_NAME,
        "next560_geometry": root560 / n560.GEOMETRY_NAME,
        "next560_predictions": root560 / n560.PREDICTIONS_NAME,
        "next560_source": Path(n560.__file__).resolve(),
        "next559_manifest": root559 / n559.MANIFEST_NAME,
        "next559_formula": root559 / n559.FORMULA_NAME,
        "next559_source": Path(n559.__file__).resolve(),
        "pauling_source": Path(pauling_control.__code__.co_filename).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT561 input is missing")
    if _sha256(source_csv) != n551.SOURCE_SHA256 or _sha256(design_path) != n559.DESIGN_SHA256:
        raise ValueError("NEXT561 formal source or design differs")
    manifest560 = json.loads(paths["next560_manifest"].read_text())
    outputs560 = manifest560.get("outputs_sha256")
    executed560 = manifest560.get("executed_source_sha256")
    if (
        manifest560.get("protocol") != n560.PROTOCOL
        or manifest560.get("next561_endpoint_opening_authorized") is not True
        or manifest560.get("endpoint_or_final_structure_columns_copied_or_decoded") is not False
        or manifest560.get("dft_values_opened") is not False
        or manifest560.get("selection_uses_endpoint_values") is not False
        or manifest560.get("gates", {}).get("passes") is not True
        or not isinstance(outputs560, dict)
        or not isinstance(executed560, dict)
        or executed560.get("src/next560_hea_entropy_packing_cohort.py") != _sha256(paths["next560_source"])
        or any(outputs560.get(name) != _sha256(root560 / name) for name in (
            n560.METADATA_NAME, n560.GEOMETRY_NAME, n560.PREDICTIONS_NAME
        ))
    ):
        raise ValueError("NEXT561 frozen prediction identity differs")
    manifest559 = json.loads(paths["next559_manifest"].read_text())
    if (
        manifest559.get("protocol") != n559.PROTOCOL
        or manifest559.get("next560_endpoint_values_opened") is not False
        or manifest559.get("outputs_sha256", {}).get(n559.FORMULA_NAME)
        != _sha256(paths["next559_formula"])
    ):
        raise ValueError("NEXT561 frozen formula identity differs")

    metadata = pd.read_parquet(paths["next560_metadata"])
    predictions = pd.read_parquet(paths["next560_predictions"])
    fids = set(metadata["fid"].astype(str))
    if (
        len(fids) != n560.EXPECTED_ROWS
        or set(predictions["fid"].astype(str)) != fids
        or predictions["fid"].duplicated().any()
    ):
        raise ValueError("NEXT561 frozen cohort identity differs")
    forbidden = {
        "e_above_hull", "structure_as_dict", "dft_waste", "waste_severity", "protected"
    }
    if forbidden & set(predictions):
        raise ValueError("NEXT561 prediction table contains endpoint fields")

    initial: dict[str, Atoms] = {}
    with zipfile.ZipFile(paths["next560_geometry"]) as archive:
        for name in archive.namelist():
            fid = Path(name).stem
            if fid in fids:
                initial[fid] = n553._decode_initial(archive.read(name))
    if set(initial) != fids:
        raise ValueError("NEXT561 x0 geometry identity differs")
    pauling_rows = pd.DataFrame(
        n558._pauling_many([(fid, initial[fid]) for fid in sorted(fids)], workers)
    )

    endpoint_payloads, firewall = n553.extract_authorized_endpoint_payloads(source_csv, fids)
    if firewall["source_rows_scanned"] != n551.EXPECTED_SOURCE_ROWS:
        raise ValueError("NEXT561 source row count differs")
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
    if len(table) != n560.EXPECTED_ROWS:
        raise ValueError("NEXT561 confirmation join differs")
    result = evaluate_confirmation(table, bootstrap_draws=bootstrap_draws)
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
            "confirmation_pass": result["confirmation_pass"],
            "inputs_sha256": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in paths.items()
            },
            "outputs_sha256": outputs,
            "executed_source_sha256": {
                "src/next561_hea_entropy_packing_confirmation.py": source_hash
            },
            "next560_endpoint_values_opened": True,
            "next560_final_structures_opened": True,
            "unauthorized_endpoint_rows_materialized": 0,
            "dft_values_used_by_executable_formula": False,
            "model_or_proxy_potential_used": False,
            "coordinates_or_cell_modified_by_formula": False,
            "scientific_improvement_claim": result["confirmation_pass"],
            "independent_report_authorized": result["confirmation_pass"],
            "canonical_report_or_paper_edits_authorized": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT561 source changed before publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next560-dir", required=True, type=Path)
    parser.add_argument("--next559-dir", required=True, type=Path)
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--design-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args(argv)
    manifest = build_confirmation(
        next560_dir=args.next560_dir, next559_dir=args.next559_dir,
        source_csv=args.source_csv, design_path=args.design_path,
        output_dir=args.output_dir, workers=args.workers,
        bootstrap_draws=args.bootstrap_draws,
    )
    print(json.dumps(_json_ready(manifest), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_confirmation", "evaluate_confirmation"]
