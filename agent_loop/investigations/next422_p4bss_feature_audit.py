#!/usr/bin/env python3
"""Audit frozen P4BSS on physically isolated discovery outcomes only."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

import src.next268_prv_feature_audit as n268
import src.next413_sssp_feature_audit as n413
import src.next420_pauling4_bond_strength_segregation as n420
import src.next421_p4bss_formal_build as n421
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-13-next422-p4bss-feature-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT422_P4BSS_FEATURE_CATALOGUE.json"
AUDIT_NAME = "NEXT422_P4BSS_FEATURE_AUDIT.json"
TABLE_NAME = "next422_p4bss_feature_audit.parquet"
HYPOTHESES = tuple((name, n420.FEATURE_DIRECTIONS[name]) for name in n420.FEATURE_NAMES)
QUANTILES = n268.QUANTILES
REQUIRED_STAGES = n413.REQUIRED_STAGES
REQUIRED_DESIGN_STAGES = n413.REQUIRED_DESIGN_STAGES
BOUNDARY_FLAGS = n413.BOUNDARY_FLAGS
EXPECTED_NEXT420_SOURCE_SHA256 = "6f931f3292d5d3d45b036b71ab85f39fd35587eadfc1a501905547db4c99062f"
EXPECTED_NEXT421_SOURCE_SHA256 = "5e0dff21a7b2c2114b5c46945ba35257118e9dc3ed8a078e7b01c3156501ce72"
_REPOSITORY = Path(__file__).resolve().parents[1]
_NEXT413_DESIGN_PATH = _REPOSITORY / "docs/plans/2026-08-13-next411-next414-same-sign-shell-purity.md"
EXPECTED_INPUT_SHA256 = {
    **{key: value for key, value in n413.EXPECTED_INPUT_SHA256.items() if key != "design"},
    "next413_design": n413.EXPECTED_INPUT_SHA256["design"],
    "design": n420.DESIGN_SHA256,
    "next421_manifest": "4fbeff1df494995d30cb861fc260bee8c16cf6825e4fd12329965ec047b67b1c",
    "next421_catalogue": "d10b347a9f2ebf79fde33a1de640ce2ac2cc5033a7950dee98f37c2af53cd55d",
    "next421_scigen": "0db525484d557b1e7769775a356da05fb92a2da0bd9ac4d187bea965a26d8577",
    "next421_wyformer": "fa6e07de08dccd4a42a2ab6af9e871f48b634b485c0efeea72894ddb0386c660",
}


def select_eligible_hypotheses(frame: pd.DataFrame):
    try:
        return n268.select_eligible_hypotheses(frame)
    except ValueError as exc:
        raise ValueError("NEXT422 audit ranking table differs") from exc


def bounded_protection(*, values: object, direction: str, q_lo: float, q_hi: float) -> np.ndarray:
    if direction != "protected_high":
        raise ValueError("NEXT422 bounded protection inputs differ")
    try:
        return n268.bounded_protection(values=values, direction=direction, q_lo=q_lo, q_hi=q_hi)
    except ValueError as exc:
        raise ValueError("NEXT422 bounded protection inputs differ") from exc


def _index_by_id(*, table: pd.DataFrame, source: str, expected_material_ids: object) -> pd.DataFrame:
    extra = table.copy()
    extra["material_id"] = source + ":" + extra["material_id"].astype(str)
    if set(extra["material_id"]) != set(pd.Series(expected_material_ids, dtype=str)):
        raise ValueError(f"NEXT422 {source} material identity differs")
    return extra.set_index("material_id")


def _strength_moments_are_consistent(values: object, expected: object, observed: object) -> np.ndarray:
    feature = np.asarray(values, dtype=float)
    left = np.asarray(expected, dtype=float)
    right = np.asarray(observed, dtype=float)
    if left.shape != feature.shape or right.shape != feature.shape:
        raise ValueError("NEXT422 strength-moment populations differ")
    recomputed = left / (left + right)
    tolerance = 0.5 / n420.OUTPUT_GRID + np.finfo(float).eps
    return (
        np.isfinite(feature) & np.isfinite(left) & np.isfinite(right)
        & (left > 0.0) & (right > 0.0)
        & (np.abs(feature - recomputed) <= tolerance)
    )


def _paths(
    *, roots: Mapping[str, Path], next135_freeze_path: Path,
    design_paths: Mapping[int, Path], design_path: Path,
) -> dict[str, Path]:
    paths = n413._paths(
        roots=roots,
        next135_freeze_path=next135_freeze_path,
        design_paths=design_paths,
        design_path=_NEXT413_DESIGN_PATH,
    )
    paths["next413_design"] = paths.pop("design")
    paths.update({
        "design": design_path,
        "next421_manifest": roots["next421"] / n421.MANIFEST_NAME,
        "next421_catalogue": roots["next421"] / n421.CATALOGUE_NAME,
        "next421_scigen": roots["next421"] / n421.FEATURE_FILES["scigen"],
        "next421_wyformer": roots["next421"] / n421.FEATURE_FILES["wyformer"],
    })
    return paths


def _verify(paths: Mapping[str, Path], hashes: Mapping[str, str]):
    prior_paths = {key: paths[key] for key in n413.EXPECTED_INPUT_SHA256}
    prior_paths["design"] = paths["next413_design"]
    prior_hashes = {key: hashes[key] for key in n413.EXPECTED_INPUT_SHA256}
    prior_hashes["design"] = hashes["next413_design"]
    prior = n413._verify(prior_paths, prior_hashes)
    manifest = json.loads(paths["next421_manifest"].read_text())
    catalogue = json.loads(paths["next421_catalogue"].read_text())
    tables = {
        source: pd.read_parquet(paths[f"next421_{source}"])
        for source in ("scigen", "wyformer")
    }
    outputs = {
        n421.CATALOGUE_NAME: hashes["next421_catalogue"],
        n421.FEATURE_FILES["scigen"]: hashes["next421_scigen"],
        n421.FEATURE_FILES["wyformer"]: hashes["next421_wyformer"],
    }
    if (
        _sha256_file(Path(n420.__file__).resolve()) != EXPECTED_NEXT420_SOURCE_SHA256
        or _sha256_file(Path(n421.__file__).resolve()) != EXPECTED_NEXT421_SOURCE_SHA256
        or manifest.get("protocol") != n421.PROTOCOL
        or manifest.get("next422_audit_authorized") is not True
        or manifest.get("labels_opened") is not False
        or manifest.get("endpoint_fields_read") is not False
        or manifest.get("internal_validation_geometry_opened") is not False
        or manifest.get("internal_replication_geometry_opened") is not False
        or manifest.get("outputs_sha256") != outputs
        or any(manifest.get(key) is not value for key, value in BOUNDARY_FLAGS.items())
        or catalogue.get("feature_names") != list(n420.FEATURE_NAMES)
        or catalogue.get("feature_directions") != n420.FEATURE_DIRECTIONS
        or catalogue.get("feature_count") != 1
        or catalogue.get("directions_frozen_before_outcome") is not True
        or catalogue.get("minimum_formal_coverage") != n421.MINIMUM_FORMAL_COVERAGE
    ):
        raise ValueError("NEXT422 NEXT421 provenance differs")
    for source, expected_rows in n421.EXPECTED_ROWS.items():
        table = tables[source]
        supported = table["p4bss_supported"].fillna(False).astype(bool)
        values = pd.to_numeric(table[n420.FEATURE_NAMES[0]], errors="coerce").to_numpy(float)
        sites = pd.to_numeric(table["p4bss_site_count"], errors="coerce")
        edges = pd.to_numeric(table["p4bss_edge_count"], errors="coerce")
        pairs = pd.to_numeric(table["p4bss_pair_count"], errors="coerce")
        expected = pd.to_numeric(table["p4bss_expected_product"], errors="coerce")
        observed = pd.to_numeric(table["p4bss_observed_product"], errors="coerce")
        if (
            len(table) != expected_rows
            or table["material_id"].astype(str).duplicated().any()
            or int(supported.sum()) != int(manifest["counts"][source]["supported"])
            or float(supported.mean()) < n421.MINIMUM_FORMAL_COVERAGE
            or not np.isfinite(values[supported]).all()
            or np.isfinite(values[~supported]).any()
            or not ((values[supported] > 0.0) & (values[supported] < 1.0)).all()
            or not (sites[supported] >= 2).all()
            or not (edges[supported] >= 2).all()
            or not (pairs[supported] >= 1).all()
            or not _strength_moments_are_consistent(values[supported], expected[supported], observed[supported]).all()
            or not (sites[~supported] == 0).all()
            or not (edges[~supported] == 0).all()
            or not (pairs[~supported] == 0).all()
        ):
            raise ValueError(f"NEXT422 NEXT421 {source} table differs")
    return (*prior[:-1], tables)


def run_p4bss_feature_audit(
    *, scigen_feature_dir: Path, scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path, wyformer_discovery_endpoint_dir: Path,
    stage_dirs: Mapping[int, Path], next135_freeze_path: Path,
    design_paths: Mapping[int, Path], design_path: Path,
    next412_dir: Path, next421_dir: Path, output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Audit exactly the one frozen P4BSS hypothesis."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs) or not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT422 input universe differs")
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{f"next{stage}": Path(stage_dirs[stage]).resolve() for stage in REQUIRED_STAGES},
        "next412": Path(next412_dir).resolve(),
        "next421": Path(next421_dir).resolve(),
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
        raise FileNotFoundError("NEXT422 input is missing")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(name for name in set(hashes) | set(EXPECTED_INPUT_SHA256) if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name))
        raise ValueError(f"NEXT422 formal input identity differs: {differing}")
    eligible, eligible214, primary_key, base_start_key, formula214, current_key, formula222, tables = _verify(paths, hashes)
    n227 = n268.n227
    combined, feature_tables, score, support, endpoint, _ = n227._reconstruct_next224_frontier(
        paths=paths, eligible=eligible, eligible214=eligible214,
        primary_key=primary_key, base_start_key=base_start_key,
        formula214=formula214, current_key=current_key, formula222=formula222,
    )
    source = combined["source_dataset"].astype(str).to_numpy()
    for source_name in ("scigen", "wyformer"):
        indexed = _index_by_id(table=tables[source_name], source=source_name, expected_material_ids=feature_tables[source_name]["material_id"])
        mask = source == source_name
        ids = combined.loc[mask, "material_id"].astype(str)
        combined.loc[mask, n420.FEATURE_NAMES[0]] = ids.map(indexed[n420.FEATURE_NAMES[0]]).to_numpy()
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
        raise ValueError("NEXT422 rejected cohort counts differ")

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
            "hypothesis": f"{feature}__{direction}", "feature": feature,
            "direction": direction, "q_lo": float(q_lo), "q_hi": float(q_hi),
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
        "features": list(n420.FEATURE_NAMES), "feature_directions": n420.FEATURE_DIRECTIONS,
        "hypotheses": [f"{feature}__{direction}" for feature, direction in HYPOTHESES],
        "hypothesis_count": 1, "quantiles": list(QUANTILES),
        "quantile_method": "inverted_cdf", "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    audit = {
        "protocol": PROTOCOL, "audit_mode": "fixed_next224_rejected_extreme_p4bss_audit",
        "base_candidate_key_sha256": n227.EXPECTED_BASE_CANDIDATE_KEY_SHA256,
        "base_threshold": n227.EXPECTED_BASE_THRESHOLD,
        "cohort_counts": {key: {"protected_rejected": value[0], "severe_rejected": value[1]} for key, value in sorted(cohort_counts.items())},
        "hypothesis_count": 1, "eligible_hypothesis_count": len(eligible_names),
        "eligible_hypotheses": eligible_names, "eligible_hypothesis_sha256": eligible_sha,
        "selected_hypothesis": selected, "next423_search_authorized": selected is not None,
        "new_formula_searched": False, "new_formula_selected": False,
        "validation_or_replication_opened": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    sources = {
        "src/next268_prv_feature_audit.py": Path(n268.__file__).resolve(),
        "src/next413_sssp_feature_audit.py": Path(n413.__file__).resolve(),
        "src/next420_pauling4_bond_strength_segregation.py": Path(n420.__file__).resolve(),
        "src/next421_p4bss_formal_build.py": Path(n421.__file__).resolve(),
        "src/next422_p4bss_feature_audit.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in sources.items()}
    try:
        _write_json(staging / CATALOGUE_NAME, catalogue)
        _write_json(staging / AUDIT_NAME, audit)
        table.to_parquet(staging / TABLE_NAME, index=False)
        outputs = [staging / CATALOGUE_NAME, staging / AUDIT_NAME, staging / TABLE_NAME]
        manifest = {
            "protocol": PROTOCOL, "hypothesis_count": 1,
            "eligible_hypothesis_count": len(eligible_names),
            "eligible_hypothesis_sha256": eligible_sha,
            "next224_frontier_reproduced": True,
            "next423_search_authorized": selected is not None,
            "p4bss_branch_terminated": selected is None,
            "new_formula_searched": False, "new_formula_selected": False,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True,
            **BOUNDARY_FLAGS, "scientific_improvement_claim": False,
            "inputs_sha256": hashes, "executed_source_sha256": source_hashes,
            "outputs_sha256": {path.name: _sha256_file(path) for path in outputs},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT422 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in sources.items()):
            raise RuntimeError("NEXT422 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _resolve_stage_dirs(root: Path) -> dict[int, Path]:
    base = Path(root).resolve()
    directories = tuple(path for path in base.iterdir() if path.is_dir())
    resolved = {}
    for stage in REQUIRED_STAGES:
        matches = [path for path in directories if path.name.startswith(f"next{stage}_")]
        if len(matches) != 1:
            raise FileNotFoundError(f"NEXT422 stage {stage} has {len(matches)} directories")
        resolved[stage] = matches[0]
    return resolved


def _resolve_design_paths() -> dict[int, Path]:
    repository = Path(__file__).resolve().parents[1]
    hashes = json.loads(
        (
            Path("$PRIS_ARCHIVE/")
            / "next413_same_sign_shell_purity_audit_v1/MANIFEST.json"
        ).read_text()
    )["inputs_sha256"]
    plans = tuple((repository / "docs/plans").glob("*.md"))
    by_hash = {_sha256_file(path): path for path in plans}
    resolved = {}
    for stage in REQUIRED_DESIGN_STAGES:
        digest = hashes[f"next{stage}_design"]
        if digest not in by_hash:
            raise FileNotFoundError(f"NEXT422 design {stage} is missing")
        resolved[stage] = by_hash[digest]
    return resolved


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.formal_root.resolve()
    manifest = run_p4bss_feature_audit(
        scigen_feature_dir=root / "next85_scigen_label_free_features_v1",
        scigen_discovery_endpoint_dir=root / "next86_scigen_discovery_endpoints_v1",
        wyformer_feature_dir=root / "next94_wyformer_label_free_features_v1",
        wyformer_discovery_endpoint_dir=root / "next93b_wyformer_blind_discovery_endpoint_lockbox_v1",
        stage_dirs=_resolve_stage_dirs(root),
        next135_freeze_path=Path(__file__).resolve().parents[1] / "docs/plans/2026-08-08-next135-conjunctive-compactness-search-freeze.json",
        design_paths=_resolve_design_paths(),
        design_path=Path(__file__).resolve().parents[1] / "docs/plans/2026-08-13-next420-next424-pauling4-bond-strength-segregation.md",
        next412_dir=root / "next412_same_sign_shell_purity_v1",
        next421_dir=root / "next421_pauling4_bond_strength_segregation_v1",
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HYPOTHESES", "QUANTILES", "REQUIRED_DESIGN_STAGES", "REQUIRED_STAGES",
    "_index_by_id", "_strength_moments_are_consistent", "bounded_protection",
    "run_p4bss_feature_audit", "select_eligible_hypotheses", "n420",
]
