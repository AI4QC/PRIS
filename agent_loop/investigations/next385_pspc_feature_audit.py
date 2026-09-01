#!/usr/bin/env python3
"""Audit frozen PSPC on physically isolated discovery outcomes only."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from collections.abc import Mapping

import numpy as np
import pandas as pd

import src.next268_prv_feature_audit as n268
import src.next381_psnb_feature_audit as n381
import src.next383_periodic_skeletal_path_collision as n383
import src.next384_pspc_formal_build as n384
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-13-next385-pspc-feature-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT385_PSPC_FEATURE_CATALOGUE.json"
AUDIT_NAME = "NEXT385_PSPC_FEATURE_AUDIT.json"
TABLE_NAME = "next385_pspc_feature_audit.parquet"
HYPOTHESES = tuple((name, n383.FEATURE_DIRECTIONS[name]) for name in n383.FEATURE_NAMES)
QUANTILES = n268.QUANTILES
REQUIRED_STAGES = n381.REQUIRED_STAGES
REQUIRED_DESIGN_STAGES = n381.REQUIRED_DESIGN_STAGES
BOUNDARY_FLAGS = n381.BOUNDARY_FLAGS
EXPECTED_NEXT383_SOURCE_SHA256 = "41d2017374478b904ce40cc8767383cda13adb33c3aa194a6ce3fad7df7f61d5"
EXPECTED_NEXT384_SOURCE_SHA256 = "74ddc2032909a76e826d87806a3a903a13cfd95d037fbc05b9781579294854b7"
_REPOSITORY = Path(__file__).resolve().parents[1]
_NEXT381_DESIGN_PATH = _REPOSITORY / "docs/plans/2026-08-13-next379-next382-periodic-skeletal-net-bottleneck.md"
EXPECTED_INPUT_SHA256 = {
    **{key: value for key, value in n381.EXPECTED_INPUT_SHA256.items() if key != "design"},
    "next381_design": n381.EXPECTED_INPUT_SHA256["design"],
    "design": n383.DESIGN_SHA256,
    "next384_manifest": "293cf0b69dcfed3fdd565f3c89f4ed5b3d3704c60530042c9e6e4c566237d2c4",
    "next384_catalogue": "4639abbf6e0f2d9a1b9d8657a963574eb89e31a3c999ee4bb089ff148561769f",
    "next384_scigen": "544ed5bae733d9ec12d33a2b443426d6411247672e4d746bfd315a9743c0d5cb",
    "next384_wyformer": "8b20457e0f64e85957828e468a357f14761b4520a4a638b6c442020d8430805a",
}


def select_eligible_hypotheses(frame: pd.DataFrame):
    try:
        return n268.select_eligible_hypotheses(frame)
    except ValueError as exc:
        raise ValueError("NEXT385 audit ranking table differs") from exc


def bounded_protection(*, values: object, direction: str, q_lo: float, q_hi: float) -> np.ndarray:
    if direction != "protected_high":
        raise ValueError("NEXT385 bounded protection inputs differ")
    return n268.bounded_protection(values=values, direction=direction, q_lo=q_lo, q_hi=q_hi)


def _index_by_id(*, table: pd.DataFrame, source: str, expected_material_ids: object) -> pd.DataFrame:
    extra = table.copy()
    extra["material_id"] = source + ":" + extra["material_id"].astype(str)
    if set(extra["material_id"]) != set(pd.Series(expected_material_ids, dtype=str)):
        raise ValueError(f"NEXT385 {source} material identity differs")
    return extra.set_index("material_id")


def _paths(*, roots: Mapping[str, Path], next135_freeze_path: Path, design_paths: Mapping[int, Path], design_path: Path) -> dict[str, Path]:
    paths = n381._paths(
        roots=roots, next135_freeze_path=next135_freeze_path,
        design_paths=design_paths, design_path=_NEXT381_DESIGN_PATH,
    )
    paths["next381_design"] = paths.pop("design")
    paths.update({
        "design": design_path,
        "next384_manifest": roots["next384"] / n384.MANIFEST_NAME,
        "next384_catalogue": roots["next384"] / n384.CATALOGUE_NAME,
        "next384_scigen": roots["next384"] / n384.FEATURE_FILES["scigen"],
        "next384_wyformer": roots["next384"] / n384.FEATURE_FILES["wyformer"],
    })
    return paths


def _verify(paths: Mapping[str, Path], hashes: Mapping[str, str]):
    prior_paths = {key: paths[key] for key in n381.EXPECTED_INPUT_SHA256}
    prior_paths["design"] = paths["next381_design"]
    prior_hashes = {key: hashes[key] for key in n381.EXPECTED_INPUT_SHA256}
    prior_hashes["design"] = hashes["next381_design"]
    prior = n381._verify_next380(prior_paths, prior_hashes)
    manifest = json.loads(paths["next384_manifest"].read_text())
    catalogue = json.loads(paths["next384_catalogue"].read_text())
    tables = {source: pd.read_parquet(paths[f"next384_{source}"]) for source in ("scigen", "wyformer")}
    outputs = {
        n384.CATALOGUE_NAME: hashes["next384_catalogue"],
        n384.FEATURE_FILES["scigen"]: hashes["next384_scigen"],
        n384.FEATURE_FILES["wyformer"]: hashes["next384_wyformer"],
    }
    if (
        _sha256_file(Path(n383.__file__).resolve()) != EXPECTED_NEXT383_SOURCE_SHA256
        or _sha256_file(Path(n384.__file__).resolve()) != EXPECTED_NEXT384_SOURCE_SHA256
        or manifest.get("protocol") != n384.PROTOCOL
        or manifest.get("next385_audit_authorized") is not True
        or manifest.get("labels_opened") is not False
        or manifest.get("endpoint_fields_read") is not False
        or manifest.get("internal_validation_geometry_opened") is not False
        or manifest.get("internal_replication_geometry_opened") is not False
        or manifest.get("outputs_sha256") != outputs
        or any(manifest.get(key) is not value for key, value in BOUNDARY_FLAGS.items())
        or catalogue.get("feature_names") != list(n383.FEATURE_NAMES)
        or catalogue.get("feature_directions") != n383.FEATURE_DIRECTIONS
        or catalogue.get("feature_count") != 1
        or catalogue.get("walk_depth") != 3
        or catalogue.get("minimum_formal_coverage") != n384.MINIMUM_FORMAL_COVERAGE
    ):
        raise ValueError("NEXT385 NEXT384 provenance differs")
    for source, expected_rows in n384.EXPECTED_ROWS.items():
        table = tables[source]
        supported = table["pspc_supported"].fillna(False).astype(bool)
        values = pd.to_numeric(table[n383.FEATURE_NAMES[0]], errors="coerce").to_numpy(float)
        if (
            len(table) != expected_rows or table["material_id"].astype(str).duplicated().any()
            or int(supported.sum()) != int(manifest["counts"][source]["supported"])
            or float(supported.mean()) < n384.MINIMUM_FORMAL_COVERAGE
            or not np.isfinite(values[supported]).all() or np.isfinite(values[~supported]).any()
            or not ((values[supported] >= 0) & (values[supported] <= 1)).all()
            or not (pd.to_numeric(table.loc[supported, "pspc_directed_face_count"]) == 2 * pd.to_numeric(table.loc[supported, "pspc_undirected_edge_count"])).all()
            or not (pd.to_numeric(table.loc[supported, "pspc_skeleton_edge_count"]) <= pd.to_numeric(table.loc[supported, "pspc_undirected_edge_count"])).all()
            or not (pd.to_numeric(table.loc[supported, "pspc_maximum_reverse_angle_error"]) <= n383.n379.REVERSE_ANGLE_TOLERANCE).all()
        ):
            raise ValueError(f"NEXT385 NEXT384 {source} table differs")
    return (*prior[:-1], tables)


def run_pspc_feature_audit(
    *, scigen_feature_dir: Path, scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path, wyformer_discovery_endpoint_dir: Path,
    stage_dirs: Mapping[int, Path], next135_freeze_path: Path,
    design_paths: Mapping[int, Path], design_path: Path, next384_dir: Path,
    output_dir: Path, require_formal_inputs: bool = True,
) -> dict[str, object]:
    if not set(REQUIRED_STAGES).issubset(stage_dirs) or not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT385 input universe differs")
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{f"next{stage}": Path(stage_dirs[stage]).resolve() for stage in REQUIRED_STAGES},
        "next384": Path(next384_dir).resolve(),
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots=roots, next135_freeze_path=Path(next135_freeze_path).resolve(),
        design_paths={stage: Path(design_paths[stage]).resolve() for stage in REQUIRED_DESIGN_STAGES},
        design_path=Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT385 input is missing")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(name for name in set(hashes) | set(EXPECTED_INPUT_SHA256) if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name))
        raise ValueError(f"NEXT385 formal input identity differs: {differing}")
    eligible, eligible214, primary_key, base_start_key, formula214, current_key, formula222, tables = _verify(paths, hashes)
    n227 = n268.n227
    combined, feature_tables, score, support, endpoint, _ = n227._reconstruct_next224_frontier(
        paths=paths, eligible=eligible, eligible214=eligible214, primary_key=primary_key,
        base_start_key=base_start_key, formula214=formula214,
        current_key=current_key, formula222=formula222,
    )
    source = combined["source_dataset"].astype(str).to_numpy()
    for source_name in ("scigen", "wyformer"):
        indexed = _index_by_id(table=tables[source_name], source=source_name, expected_material_ids=feature_tables[source_name]["material_id"])
        mask = source == source_name
        ids = combined.loc[mask, "material_id"].astype(str)
        combined.loc[mask, n383.FEATURE_NAMES[0]] = ids.map(indexed[n383.FEATURE_NAMES[0]]).to_numpy()
    cohort = n227.build_rejected_extreme_cohort(score=score, support=support, endpoint=endpoint, threshold=n227.EXPECTED_BASE_THRESHOLD)
    folds = n227.n226.n225.n222.n215.n214.n164.assign_group_folds(combined["reduced_formula"].astype(str).to_numpy())
    protected, severe = endpoint <= 1, endpoint >= 2
    cohort_counts = {}
    for source_name in ("scigen", "wyformer"):
        for fold in (None, 0, 1, 2, 3, 4):
            mask = cohort & (source == source_name)
            if fold is not None:
                mask &= folds == fold
            cohort_counts[f"{source_name}:{'all' if fold is None else f'fold{fold}'}"] = (int((mask & protected).sum()), int((mask & severe).sum()))
    if cohort_counts != n227.EXPECTED_COHORT_COUNTS:
        raise ValueError("NEXT385 rejected cohort counts differ")
    rows = []
    for feature, direction in HYPOTHESES:
        values = pd.to_numeric(combined[feature], errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        q_lo, q_hi = np.quantile(finite, QUANTILES, method="inverted_cdf")
        mapped = bounded_protection(values=values, direction=direction, q_lo=float(q_lo), q_hi=float(q_hi))
        audits = {
            name: n227.audit_one_source(values=mapped[source == name], endpoint=endpoint[source == name], cohort=cohort[source == name], folds=folds[source == name], direction="protected_high")
            for name in ("scigen", "wyformer")
        }
        a, b = audits["scigen"], audits["wyformer"]
        rows.append({
            "hypothesis": f"{feature}__{direction}", "feature": feature, "direction": direction,
            "q_lo": float(q_lo), "q_hi": float(q_hi),
            "passes_raw_gates": bool(a["passes_source_gates"] and b["passes_source_gates"]),
            "ranking_min_worst_fold_auc": min(float(a["worst_fold_auc"]), float(b["worst_fold_auc"])),
            "ranking_min_aggregate_auc": min(float(a["aggregate_auc"]), float(b["aggregate_auc"])),
            "ranking_mean_aggregate_auc": np.mean([float(a["aggregate_auc"]), float(b["aggregate_auc"])]),
            **{f"scigen_{key}": a[key] for key in ("aggregate_auc", "macro_fold_auc", "worst_fold_auc", "minimum_cell_coverage")},
            **{f"wyformer_{key}": b[key] for key in ("aggregate_auc", "macro_fold_auc", "worst_fold_auc", "minimum_cell_coverage")},
            "source_audits_json": json.dumps(audits, sort_keys=True, separators=(",", ":")),
        })
    table, selected = select_eligible_hypotheses(pd.DataFrame(rows))
    eligible_names = sorted(table.loc[table["eligible_for_search"], "hypothesis"].astype(str))
    eligible_sha = hashlib.sha256("\n".join(eligible_names).encode()).hexdigest()
    catalogue = {
        "protocol": PROTOCOL, "design_sha256": hashes["design"],
        "features": list(n383.FEATURE_NAMES), "feature_directions": n383.FEATURE_DIRECTIONS,
        "hypotheses": [f"{f}__{d}" for f, d in HYPOTHESES], "hypothesis_count": 1,
        "quantiles": list(QUANTILES), "quantile_method": "inverted_cdf",
        "validation_outputs_opened": False, "dft_values_used_by_executable_formula": False,
    }
    audit = {
        "protocol": PROTOCOL, "audit_mode": "fixed_next224_rejected_extreme_pspc_audit",
        "base_candidate_key_sha256": n227.EXPECTED_BASE_CANDIDATE_KEY_SHA256,
        "base_threshold": n227.EXPECTED_BASE_THRESHOLD,
        "cohort_counts": {key: {"protected_rejected": value[0], "severe_rejected": value[1]} for key, value in sorted(cohort_counts.items())},
        "hypothesis_count": 1, "eligible_hypothesis_count": len(eligible_names),
        "eligible_hypotheses": eligible_names, "eligible_hypothesis_sha256": eligible_sha,
        "selected_hypothesis": selected, "next386_search_authorized": selected is not None,
        "new_formula_searched": False, "new_formula_selected": False,
        "validation_or_replication_opened": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    sources = {
        "src/next268_prv_feature_audit.py": Path(n268.__file__).resolve(),
        "src/next381_psnb_feature_audit.py": Path(n381.__file__).resolve(),
        "src/next383_periodic_skeletal_path_collision.py": Path(n383.__file__).resolve(),
        "src/next384_pspc_formal_build.py": Path(n384.__file__).resolve(),
        "src/next385_pspc_feature_audit.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in sources.items()}
    try:
        _write_json(staging / CATALOGUE_NAME, catalogue)
        _write_json(staging / AUDIT_NAME, audit)
        table.to_parquet(staging / TABLE_NAME, index=False)
        outputs = [staging / CATALOGUE_NAME, staging / AUDIT_NAME, staging / TABLE_NAME]
        manifest = {
            "protocol": PROTOCOL, "hypothesis_count": 1,
            "eligible_hypothesis_count": len(eligible_names), "eligible_hypothesis_sha256": eligible_sha,
            "next224_frontier_reproduced": True, "next386_search_authorized": selected is not None,
            "pspc_branch_terminated": selected is None, "new_formula_searched": False,
            "new_formula_selected": False, "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True, **BOUNDARY_FLAGS,
            "scientific_improvement_claim": False, "inputs_sha256": hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {path.name: _sha256_file(path) for path in outputs},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT385 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in sources.items()):
            raise RuntimeError("NEXT385 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = ["HYPOTHESES", "QUANTILES", "REQUIRED_DESIGN_STAGES", "REQUIRED_STAGES", "_index_by_id", "bounded_protection", "run_pspc_feature_audit", "select_eligible_hypotheses"]
