#!/usr/bin/env python3
"""Audit frozen ECCC-CDE on physically isolated discovery outcomes only."""

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
import src.next515_eccc_conservative_domain_extension as n515
import src.next516_eccc_cde_formal_build as n516
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-13-next517-eccc-cde-feature-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT517_ECCC_CDE_FEATURE_CATALOGUE.json"
AUDIT_NAME = "NEXT517_ECCC_CDE_FEATURE_AUDIT.json"
TABLE_NAME = "next517_eccc_cde_feature_audit.parquet"
HYPOTHESES = tuple(
    (name, n515.FEATURE_DIRECTIONS[name]) for name in n515.FEATURE_NAMES
)
QUANTILES = n268.QUANTILES
REQUIRED_STAGES = n413.REQUIRED_STAGES
REQUIRED_DESIGN_STAGES = n413.REQUIRED_DESIGN_STAGES
BOUNDARY_FLAGS = n413.BOUNDARY_FLAGS
EXPECTED_NEXT515_SOURCE_SHA256 = "e35e44a848cf4b2f35892d1a0efd4dea25bc0a4fd01331a2384205557aeded9d"
EXPECTED_NEXT516_SOURCE_SHA256 = "50727cebf08e1bb74af721f301947f9f68c979491833e68dd76128f457d941ae"
_REPOSITORY = Path(__file__).resolve().parents[1]
_NEXT413_DESIGN_PATH = (
    _REPOSITORY / "docs/plans/2026-08-13-next411-next414-same-sign-shell-purity.md"
)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n413.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next413_design": n413.EXPECTED_INPUT_SHA256["design"],
    "design": n515.DESIGN_SHA256,
    "next516_manifest": "c2d44752008c9348a42da0a5bb3536c917bd09fb2400e58f6b1f706b2b3bfe8d",
    "next516_catalogue": "1f2f4caf8d3a2e39ce52c1bc120bbb55dfd8e0e4254aa6ac4e250bc7d65b5ab9",
    "next516_scigen": "a6c3c8778eebb90d4a6b5197c95bb1a28ba087b33f099226beb6e8505086dae9",
    "next516_wyformer": "e8643b564a1bc9b601374cfe6920560398d607449cde041f600bc9b1e59f9f7a",
}


def select_eligible_hypotheses(frame: pd.DataFrame):
    try:
        return n268.select_eligible_hypotheses(frame)
    except ValueError as exc:
        raise ValueError("NEXT517 audit ranking table differs") from exc


def bounded_protection(
    *, values: object, direction: str, q_lo: float, q_hi: float
) -> np.ndarray:
    if direction != "protected_high":
        raise ValueError("NEXT517 bounded protection inputs differ")
    try:
        return n268.bounded_protection(
            values=values, direction=direction, q_lo=q_lo, q_hi=q_hi
        )
    except ValueError as exc:
        raise ValueError("NEXT517 bounded protection inputs differ") from exc


def _index_by_id(
    *, table: pd.DataFrame, source: str, expected_material_ids: object
) -> pd.DataFrame:
    extra = table.copy()
    extra["material_id"] = source + ":" + extra["material_id"].astype(str)
    if set(extra["material_id"]) != set(pd.Series(expected_material_ids, dtype=str)):
        raise ValueError(f"NEXT517 {source} material identity differs")
    return extra.set_index("material_id")


def _cde_rows_are_consistent(
    values: object, unknown_cations: object, isolated_sites: object
) -> np.ndarray:
    feature = np.asarray(values, dtype=float)
    cations = np.asarray(unknown_cations, dtype=float)
    isolated = np.asarray(isolated_sites, dtype=float)
    if cations.shape != feature.shape or isolated.shape != feature.shape:
        raise ValueError("NEXT517 ECCC-CDE populations differ")
    return (
        np.isfinite(feature)
        & (feature >= 0.0)
        & (feature <= 1.0)
        & np.isfinite(cations)
        & np.isfinite(isolated)
        & (cations >= 0.0)
        & (isolated >= 0.0)
        & (cations == np.rint(cations))
        & (isolated == np.rint(isolated))
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
    paths.update(
        {
            "design": design_path,
            "next516_manifest": roots["next516"] / n516.MANIFEST_NAME,
            "next516_catalogue": roots["next516"] / n516.CATALOGUE_NAME,
            "next516_scigen": roots["next516"] / n516.FEATURE_FILES["scigen"],
            "next516_wyformer": roots["next516"] / n516.FEATURE_FILES["wyformer"],
        }
    )
    return paths


def _verify(paths: Mapping[str, Path], hashes: Mapping[str, str]):
    prior_paths = {key: paths[key] for key in n413.EXPECTED_INPUT_SHA256}
    prior_paths["design"] = paths["next413_design"]
    prior_hashes = {key: hashes[key] for key in n413.EXPECTED_INPUT_SHA256}
    prior_hashes["design"] = hashes["next413_design"]
    prior = n413._verify(prior_paths, prior_hashes)
    manifest = json.loads(paths["next516_manifest"].read_text())
    catalogue = json.loads(paths["next516_catalogue"].read_text())
    tables = {
        source: pd.read_parquet(paths[f"next516_{source}"])
        for source in ("scigen", "wyformer")
    }
    outputs = {
        n516.CATALOGUE_NAME: hashes["next516_catalogue"],
        n516.FEATURE_FILES["scigen"]: hashes["next516_scigen"],
        n516.FEATURE_FILES["wyformer"]: hashes["next516_wyformer"],
    }
    if (
        _sha256_file(Path(n515.__file__).resolve()) != EXPECTED_NEXT515_SOURCE_SHA256
        or _sha256_file(Path(n516.__file__).resolve()) != EXPECTED_NEXT516_SOURCE_SHA256
        or manifest.get("protocol") != n516.PROTOCOL
        or manifest.get("next517_audit_authorized") is not True
        or manifest.get("coverage_gate_passed") is not True
        or manifest.get("post_coverage_extension") is not True
        or manifest.get("new_novelty_claim") is not False
        or manifest.get("labels_opened") is not False
        or manifest.get("endpoint_fields_read") is not False
        or manifest.get("internal_validation_geometry_opened") is not False
        or manifest.get("internal_replication_geometry_opened") is not False
        or manifest.get("outputs_sha256") != outputs
        or catalogue.get("feature_names") != list(n515.FEATURE_NAMES)
        or catalogue.get("feature_directions") != n515.FEATURE_DIRECTIONS
        or catalogue.get("feature_count") != 1
        or catalogue.get("directions_frozen_before_outcome") is not True
        or catalogue.get("minimum_formal_coverage") != n516.MINIMUM_FORMAL_COVERAGE
    ):
        raise ValueError("NEXT517 NEXT516 provenance differs")
    for source, expected_rows in n516.EXPECTED_ROWS.items():
        table = tables[source]
        supported = table["eccc_cde_supported"].fillna(False).astype(bool)
        values = pd.to_numeric(
            table[n515.FEATURE_NAMES[0]], errors="coerce"
        ).to_numpy(float)
        sites = pd.to_numeric(
            table["eccc_cde_site_count"], errors="coerce"
        ).to_numpy(float)
        edges = pd.to_numeric(
            table["eccc_cde_edge_count"], errors="coerce"
        ).to_numpy(float)
        unknown_cations = pd.to_numeric(
            table["eccc_cde_unknown_cation_count"], errors="coerce"
        ).to_numpy(float)
        isolated_sites = pd.to_numeric(
            table["eccc_cde_isolated_site_count"], errors="coerce"
        ).to_numpy(float)
        if (
            len(table) != expected_rows
            or table["material_id"].astype(str).duplicated().any()
            or int(supported.sum()) != int(manifest["counts"][source]["supported"])
            or float(supported.mean()) < n516.MINIMUM_FORMAL_COVERAGE
            or not np.isfinite(values[supported]).all()
            or np.isfinite(values[~supported]).any()
            or not _cde_rows_are_consistent(
                values[supported], unknown_cations[supported], isolated_sites[supported]
            ).all()
            or not (sites[supported] >= 1).all()
            or not (edges[supported] >= 0).all()
            or not (sites[~supported] == 0).all()
            or not (edges[~supported] == 0).all()
        ):
            raise ValueError(f"NEXT517 NEXT516 {source} table differs")
    return (*prior[:-1], tables)


def run_eccc_cde_feature_audit(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    stage_dirs: Mapping[int, Path],
    next135_freeze_path: Path,
    design_paths: Mapping[int, Path],
    design_path: Path,
    next412_dir: Path,
    next516_dir: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Audit exactly the one frozen post-coverage ECCC-CDE hypothesis."""

    if (
        not set(REQUIRED_STAGES).issubset(stage_dirs)
        or not set(REQUIRED_DESIGN_STAGES).issubset(design_paths)
    ):
        raise ValueError("NEXT517 input universe differs")
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{
            f"next{stage}": Path(stage_dirs[stage]).resolve()
            for stage in REQUIRED_STAGES
        },
        "next412": Path(next412_dir).resolve(),
        "next516": Path(next516_dir).resolve(),
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
        raise FileNotFoundError("NEXT517 input is missing")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name for name in set(hashes) | set(EXPECTED_INPUT_SHA256)
            if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT517 formal input identity differs: {differing}")
    (
        eligible, eligible214, primary_key, base_start_key, formula214,
        current_key, formula222, tables,
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
        combined.loc[mask, n515.FEATURE_NAMES[0]] = ids.map(
            indexed[n515.FEATURE_NAMES[0]]
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
    cohort_counts: dict[str, tuple[int, int]] = {}
    for source_name in ("scigen", "wyformer"):
        for fold in (None, 0, 1, 2, 3, 4):
            mask = cohort & (source == source_name)
            if fold is not None:
                mask &= folds == fold
            cohort_counts[
                f"{source_name}:{'all' if fold is None else f'fold{fold}'}"
            ] = (int((mask & protected).sum()), int((mask & severe).sum()))
    if cohort_counts != n227.EXPECTED_COHORT_COUNTS:
        raise ValueError("NEXT517 rejected cohort counts differ")

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
                "ranking_mean_aggregate_auc": float(
                    np.mean([a["aggregate_auc"], b["aggregate_auc"]])
                ),
                **{
                    f"scigen_{key}": a[key]
                    for key in (
                        "aggregate_auc", "macro_fold_auc", "worst_fold_auc",
                        "minimum_cell_coverage",
                    )
                },
                **{
                    f"wyformer_{key}": b[key]
                    for key in (
                        "aggregate_auc", "macro_fold_auc", "worst_fold_auc",
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
        "features": list(n515.FEATURE_NAMES),
        "feature_directions": n515.FEATURE_DIRECTIONS,
        "hypotheses": [f"{feature}__{direction}" for feature, direction in HYPOTHESES],
        "hypothesis_count": 1,
        "quantiles": list(QUANTILES),
        "quantile_method": "inverted_cdf",
        "post_coverage_extension": True,
        "prospective_confirmation_claim": False,
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    audit = {
        "protocol": PROTOCOL,
        "audit_mode": "fixed_next224_rejected_extreme_post_coverage_eccc_cde_audit",
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
        "next518_search_authorized": selected is not None,
        "post_coverage_extension": True,
        "prospective_confirmation_claim": False,
        "new_formula_searched": False,
        "new_formula_selected": False,
        "validation_or_replication_opened": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    sources = {
        "src/next268_prv_feature_audit.py": Path(n268.__file__).resolve(),
        "src/next413_sssp_feature_audit.py": Path(n413.__file__).resolve(),
        "src/next515_eccc_conservative_domain_extension.py": Path(n515.__file__).resolve(),
        "src/next516_eccc_cde_formal_build.py": Path(n516.__file__).resolve(),
        "src/next517_eccc_cde_feature_audit.py": Path(__file__).resolve(),
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
            "next518_search_authorized": selected is not None,
            "eccc_cde_branch_terminated": selected is None,
            "post_coverage_extension": True,
            "prospective_confirmation_claim": False,
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
            raise RuntimeError("NEXT517 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in sources.items()
        ):
            raise RuntimeError("NEXT517 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _resolve_stage_dirs(root: Path) -> dict[int, Path]:
    directories = tuple(path for path in Path(root).resolve().iterdir() if path.is_dir())
    resolved = {}
    for stage in REQUIRED_STAGES:
        matches = [path for path in directories if path.name.startswith(f"next{stage}_")]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"NEXT517 stage {stage} has {len(matches)} directories"
            )
        resolved[stage] = matches[0]
    return resolved


def _resolve_design_paths() -> dict[int, Path]:
    hashes = json.loads(
        (
            Path("$PRIS_ARCHIVE/")
            / "next413_same_sign_shell_purity_audit_v1/MANIFEST.json"
        ).read_text()
    )["inputs_sha256"]
    plans = tuple((_REPOSITORY / "docs/plans").glob("*.md"))
    by_hash = {_sha256_file(path): path for path in plans}
    resolved = {}
    for stage in REQUIRED_DESIGN_STAGES:
        digest = hashes[f"next{stage}_design"]
        if digest not in by_hash:
            raise FileNotFoundError(f"NEXT517 design {stage} is missing")
        resolved[stage] = by_hash[digest]
    return resolved


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.formal_root.resolve()
    manifest = run_eccc_cde_feature_audit(
        scigen_feature_dir=root / "next85_scigen_label_free_features_v1",
        scigen_discovery_endpoint_dir=root / "next86_scigen_discovery_endpoints_v1",
        wyformer_feature_dir=root / "next94_wyformer_label_free_features_v1",
        wyformer_discovery_endpoint_dir=root / "next93b_wyformer_blind_discovery_endpoint_lockbox_v1",
        stage_dirs=_resolve_stage_dirs(root),
        next135_freeze_path=_REPOSITORY / "docs/plans/2026-08-08-next135-conjunctive-compactness-search-freeze.json",
        design_paths=_resolve_design_paths(),
        design_path=_REPOSITORY / "docs/plans/2026-08-13-next515-next519-eccc-conservative-domain-extension.md",
        next412_dir=root / "next412_same_sign_shell_purity_v1",
        next516_dir=root / "next516_eccc_conservative_domain_extension_v1",
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
