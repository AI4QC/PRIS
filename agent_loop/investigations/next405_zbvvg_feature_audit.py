#!/usr/bin/env python3
"""Audit frozen ZBVVG on physically isolated discovery outcomes only."""

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
import src.next403_zachara_over_linear_vector_gain as n403
import src.next404_zbvvg_formal_build as n404
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-13-next405-zbvvg-feature-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT405_ZBVVG_FEATURE_CATALOGUE.json"
AUDIT_NAME = "NEXT405_ZBVVG_FEATURE_AUDIT.json"
TABLE_NAME = "next405_zbvvg_feature_audit.parquet"
HYPOTHESES = tuple((name, n403.FEATURE_DIRECTIONS[name]) for name in n403.FEATURE_NAMES)
QUANTILES = n268.QUANTILES
REQUIRED_STAGES = n381.REQUIRED_STAGES
REQUIRED_DESIGN_STAGES = n381.REQUIRED_DESIGN_STAGES
BOUNDARY_FLAGS = n381.BOUNDARY_FLAGS
EXPECTED_NEXT403_SOURCE_SHA256 = "c3ea344de26590366318a5bc6e277654f179f9b54a55f37935e2e9c59eabbc8b"
EXPECTED_NEXT404_SOURCE_SHA256 = "63adcf83c6a43090b733e341ce56949f79c882cbcec1718b84020d32d2e2482e"
_REPOSITORY = Path(__file__).resolve().parents[1]
_NEXT381_DESIGN_PATH = _REPOSITORY / "docs/plans/2026-08-13-next379-next382-periodic-skeletal-net-bottleneck.md"
EXPECTED_INPUT_SHA256 = {
    **{key: value for key, value in n381.EXPECTED_INPUT_SHA256.items() if key != "design"},
    "next381_design": n381.EXPECTED_INPUT_SHA256["design"],
    "design": n403.DESIGN_SHA256,
    "next404_manifest": "99928c086d927714af879ebfae7513e09e62971015bb7de90e51c84d938c0620",
    "next404_catalogue": "c27b3d788ef69b50705b3c6cffc36b8300af9a85745240375b7ba577e60d10e6",
    "next404_scigen": "7d8800a909135f170b7775f1244b7f3015779e9fa9d2f9488e08b08b94fb4c6e",
    "next404_wyformer": "acde78b7caaef2e3953237e968374b1de014069d2baf782386143430481174fd",
}


def select_eligible_hypotheses(frame: pd.DataFrame):
    try:
        return n268.select_eligible_hypotheses(frame)
    except ValueError as exc:
        raise ValueError("NEXT405 audit ranking table differs") from exc


def bounded_protection(*, values: object, direction: str, q_lo: float, q_hi: float) -> np.ndarray:
    if direction != "protected_high":
        raise ValueError("NEXT405 bounded protection inputs differ")
    return n268.bounded_protection(
        values=values, direction=direction, q_lo=q_lo, q_hi=q_hi
    )


def _index_by_id(
    *, table: pd.DataFrame, source: str, expected_material_ids: object
) -> pd.DataFrame:
    extra = table.copy()
    extra["material_id"] = source + ":" + extra["material_id"].astype(str)
    if set(extra["material_id"]) != set(pd.Series(expected_material_ids, dtype=str)):
        raise ValueError(f"NEXT405 {source} material identity differs")
    return extra.set_index("material_id")


def _paths(
    *,
    roots: Mapping[str, Path],
    next135_freeze_path: Path,
    design_paths: Mapping[int, Path],
    design_path: Path,
) -> dict[str, Path]:
    paths = n381._paths(
        roots=roots,
        next135_freeze_path=next135_freeze_path,
        design_paths=design_paths,
        design_path=_NEXT381_DESIGN_PATH,
    )
    paths["next381_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next404_manifest": roots["next404"] / n404.MANIFEST_NAME,
            "next404_catalogue": roots["next404"] / n404.CATALOGUE_NAME,
            "next404_scigen": roots["next404"] / n404.FEATURE_FILES["scigen"],
            "next404_wyformer": roots["next404"] / n404.FEATURE_FILES["wyformer"],
        }
    )
    return paths


def _verify(paths: Mapping[str, Path], hashes: Mapping[str, str]):
    prior_paths = {key: paths[key] for key in n381.EXPECTED_INPUT_SHA256}
    prior_paths["design"] = paths["next381_design"]
    prior_hashes = {key: hashes[key] for key in n381.EXPECTED_INPUT_SHA256}
    prior_hashes["design"] = hashes["next381_design"]
    prior = n381._verify_next380(prior_paths, prior_hashes)
    manifest = json.loads(paths["next404_manifest"].read_text())
    catalogue = json.loads(paths["next404_catalogue"].read_text())
    tables = {
        source: pd.read_parquet(paths[f"next404_{source}"])
        for source in ("scigen", "wyformer")
    }
    outputs = {
        n404.CATALOGUE_NAME: hashes["next404_catalogue"],
        n404.FEATURE_FILES["scigen"]: hashes["next404_scigen"],
        n404.FEATURE_FILES["wyformer"]: hashes["next404_wyformer"],
    }
    if (
        _sha256_file(Path(n403.__file__).resolve()) != EXPECTED_NEXT403_SOURCE_SHA256
        or _sha256_file(Path(n404.__file__).resolve()) != EXPECTED_NEXT404_SOURCE_SHA256
        or manifest.get("protocol") != n404.PROTOCOL
        or manifest.get("next405_audit_authorized") is not True
        or manifest.get("labels_opened") is not False
        or manifest.get("endpoint_fields_read") is not False
        or manifest.get("internal_validation_geometry_opened") is not False
        or manifest.get("internal_replication_geometry_opened") is not False
        or manifest.get("outputs_sha256") != outputs
        or any(manifest.get(key) is not value for key, value in BOUNDARY_FLAGS.items())
        or catalogue.get("feature_names") != list(n403.FEATURE_NAMES)
        or catalogue.get("feature_directions") != n403.FEATURE_DIRECTIONS
        or catalogue.get("feature_count") != 1
        or catalogue.get("minimum_formal_coverage") != n404.MINIMUM_FORMAL_COVERAGE
    ):
        raise ValueError("NEXT405 NEXT404 provenance differs")
    for source, expected_rows in n404.EXPECTED_ROWS.items():
        table = tables[source]
        supported = table["zbvvg_supported"].fillna(False).astype(bool)
        values = pd.to_numeric(
            table[n403.FEATURE_NAMES[0]], errors="coerce"
        ).to_numpy(float)
        if (
            len(table) != expected_rows
            or table["material_id"].astype(str).duplicated().any()
            or int(supported.sum()) != int(manifest["counts"][source]["supported"])
            or float(supported.mean()) < n404.MINIMUM_FORMAL_COVERAGE
            or not np.isfinite(values[supported]).all()
            or np.isfinite(values[~supported]).any()
            or not ((values[supported] >= 0) & (values[supported] <= 1)).all()
            or not (
                pd.to_numeric(table.loc[supported, "zbvvg_minimum_degree"]) >= 1
            ).all()
            or not (
                pd.to_numeric(table.loc[supported, "zbvvg_maximum_degree"])
                >= pd.to_numeric(table.loc[supported, "zbvvg_minimum_degree"])
            ).all()
        ):
            raise ValueError(f"NEXT405 NEXT404 {source} table differs")
    return (*prior[:-1], tables)


def run_zbvvg_feature_audit(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    stage_dirs: Mapping[int, Path],
    next135_freeze_path: Path,
    design_paths: Mapping[int, Path],
    design_path: Path,
    next404_dir: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    if not set(REQUIRED_STAGES).issubset(stage_dirs) or not set(
        REQUIRED_DESIGN_STAGES
    ).issubset(design_paths):
        raise ValueError("NEXT405 input universe differs")
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{
            f"next{stage}": Path(stage_dirs[stage]).resolve()
            for stage in REQUIRED_STAGES
        },
        "next404": Path(next404_dir).resolve(),
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots=roots,
        next135_freeze_path=Path(next135_freeze_path).resolve(),
        design_paths={
            stage: Path(design_paths[stage]).resolve()
            for stage in REQUIRED_DESIGN_STAGES
        },
        design_path=Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT405 input is missing")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(hashes) | set(EXPECTED_INPUT_SHA256)
            if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT405 formal input identity differs: {differing}")
    (
        eligible,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        current_key,
        formula222,
        tables,
    ) = _verify(paths, hashes)
    n227 = n268.n227
    combined, feature_tables, score, support, endpoint, _ = (
        n227._reconstruct_next224_frontier(
            paths=paths,
            eligible=eligible,
            eligible214=eligible214,
            primary_key=primary_key,
            base_start_key=base_start_key,
            formula214=formula214,
            current_key=current_key,
            formula222=formula222,
        )
    )
    source = combined["source_dataset"].astype(str).to_numpy()
    for source_name in ("scigen", "wyformer"):
        indexed = _index_by_id(
            table=tables[source_name],
            source=source_name,
            expected_material_ids=feature_tables[source_name]["material_id"],
        )
        mask = source == source_name
        ids = combined.loc[mask, "material_id"].astype(str)
        combined.loc[mask, n403.FEATURE_NAMES[0]] = ids.map(
            indexed[n403.FEATURE_NAMES[0]]
        ).to_numpy()
    cohort = n227.build_rejected_extreme_cohort(
        score=score,
        support=support,
        endpoint=endpoint,
        threshold=n227.EXPECTED_BASE_THRESHOLD,
    )
    folds = n227.n226.n225.n222.n215.n214.n164.assign_group_folds(
        combined["reduced_formula"].astype(str).to_numpy()
    )
    protected, severe = endpoint <= 1, endpoint >= 2
    cohort_counts = {}
    for source_name in ("scigen", "wyformer"):
        for fold in (None, 0, 1, 2, 3, 4):
            mask = cohort & (source == source_name)
            if fold is not None:
                mask &= folds == fold
            cohort_counts[f"{source_name}:{'all' if fold is None else f'fold{fold}'}"] = (
                int((mask & protected).sum()),
                int((mask & severe).sum()),
            )
    if cohort_counts != n227.EXPECTED_COHORT_COUNTS:
        raise ValueError("NEXT405 rejected cohort counts differ")

    rows = []
    for feature, direction in HYPOTHESES:
        values = pd.to_numeric(combined[feature], errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        q_lo, q_hi = np.quantile(finite, QUANTILES, method="inverted_cdf")
        mapped = bounded_protection(
            values=values,
            direction=direction,
            q_lo=float(q_lo),
            q_hi=float(q_hi),
        )
        audits = {
            name: n227.audit_one_source(
                values=mapped[source == name],
                endpoint=endpoint[source == name],
                cohort=cohort[source == name],
                folds=folds[source == name],
                direction="protected_high",
            )
            for name in ("scigen", "wyformer")
        }
        a, b = audits["scigen"], audits["wyformer"]
        rows.append(
            {
                "hypothesis": f"{feature}__{direction}",
                "feature": feature,
                "direction": direction,
                "q_lo": float(q_lo),
                "q_hi": float(q_hi),
                "passes_raw_gates": bool(
                    a["passes_source_gates"] and b["passes_source_gates"]
                ),
                "ranking_min_worst_fold_auc": min(
                    float(a["worst_fold_auc"]), float(b["worst_fold_auc"])
                ),
                "ranking_min_aggregate_auc": min(
                    float(a["aggregate_auc"]), float(b["aggregate_auc"])
                ),
                "ranking_mean_aggregate_auc": np.mean(
                    [float(a["aggregate_auc"]), float(b["aggregate_auc"])]
                ),
                **{
                    f"scigen_{key}": a[key]
                    for key in (
                        "aggregate_auc",
                        "macro_fold_auc",
                        "worst_fold_auc",
                        "minimum_cell_coverage",
                    )
                },
                **{
                    f"wyformer_{key}": b[key]
                    for key in (
                        "aggregate_auc",
                        "macro_fold_auc",
                        "worst_fold_auc",
                        "minimum_cell_coverage",
                    )
                },
                "source_audits_json": json.dumps(
                    audits, sort_keys=True, separators=(",", ":")
                ),
            }
        )
    table, selected = select_eligible_hypotheses(pd.DataFrame(rows))
    eligible_names = sorted(
        table.loc[table["eligible_for_search"], "hypothesis"].astype(str)
    )
    eligible_sha = hashlib.sha256("\n".join(eligible_names).encode()).hexdigest()
    catalogue = {
        "protocol": PROTOCOL,
        "design_sha256": hashes["design"],
        "features": list(n403.FEATURE_NAMES),
        "feature_directions": n403.FEATURE_DIRECTIONS,
        "hypotheses": [f"{feature}__{direction}" for feature, direction in HYPOTHESES],
        "hypothesis_count": 1,
        "quantiles": list(QUANTILES),
        "quantile_method": "inverted_cdf",
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    audit = {
        "protocol": PROTOCOL,
        "audit_mode": "fixed_next224_rejected_extreme_zbvvg_audit",
        "base_candidate_key_sha256": n227.EXPECTED_BASE_CANDIDATE_KEY_SHA256,
        "base_threshold": n227.EXPECTED_BASE_THRESHOLD,
        "cohort_counts": {
            key: {"protected_rejected": value[0], "severe_rejected": value[1]}
            for key, value in sorted(cohort_counts.items())
        },
        "hypothesis_count": 1,
        "eligible_hypothesis_count": len(eligible_names),
        "eligible_hypotheses": eligible_names,
        "eligible_hypothesis_sha256": eligible_sha,
        "selected_hypothesis": selected,
        "next406_search_authorized": selected is not None,
        "new_formula_searched": False,
        "new_formula_selected": False,
        "validation_or_replication_opened": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    sources = {
        "src/next268_prv_feature_audit.py": Path(n268.__file__).resolve(),
        "src/next381_psnb_feature_audit.py": Path(n381.__file__).resolve(),
        "src/next403_zachara_over_linear_vector_gain.py": Path(n403.__file__).resolve(),
        "src/next404_zbvvg_formal_build.py": Path(n404.__file__).resolve(),
        "src/next405_zbvvg_feature_audit.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in sources.items()}
    try:
        _write_json(staging / CATALOGUE_NAME, catalogue)
        _write_json(staging / AUDIT_NAME, audit)
        table.to_parquet(staging / TABLE_NAME, index=False)
        outputs = [staging / CATALOGUE_NAME, staging / AUDIT_NAME, staging / TABLE_NAME]
        manifest = {
            "protocol": PROTOCOL,
            "hypothesis_count": 1,
            "eligible_hypothesis_count": len(eligible_names),
            "eligible_hypothesis_sha256": eligible_sha,
            "next224_frontier_reproduced": True,
            "next406_search_authorized": selected is not None,
            "zbvvg_branch_terminated": selected is None,
            "new_formula_searched": False,
            "new_formula_selected": False,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True,
            **BOUNDARY_FLAGS,
            "scientific_improvement_claim": False,
            "inputs_sha256": hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {path.name: _sha256_file(path) for path in outputs},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT405 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in sources.items()
        ):
            raise RuntimeError("NEXT405 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "HYPOTHESES",
    "QUANTILES",
    "REQUIRED_DESIGN_STAGES",
    "REQUIRED_STAGES",
    "_index_by_id",
    "bounded_protection",
    "run_zbvvg_feature_audit",
    "select_eligible_hypotheses",
]
