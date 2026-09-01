#!/usr/bin/env python3
"""Diagnose the frozen NEXT498 CCLAB-CDE BROAD residual."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd

import src.next262_pvbp_broad_diagnostic as n262
import src.next498_cclab_cde_margin_local_search as n498
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-13-next499-cclab-cde-broad-diagnostic-v1"
MANIFEST_NAME = "MANIFEST.json"
DIAGNOSTIC_NAME = "NEXT499_CCLAB_CDE_BROAD_DIAGNOSTIC.json"
TABLE_NAME = "next499_cclab_cde_broad_diagnostic.parquet"
EXPECTED_DESIGN_SHA256 = n498.EXPECTED_DESIGN_SHA256
EXPECTED_NEXT262_SOURCE_SHA256 = (
    "2e074b9575985fadf1e680b367c72ab2e3c9b21bbd996ad3e1af1f00b73f7421"
)
EXPECTED_NEXT498_SOURCE_SHA256 = (
    "fd3ec7cefc09eb03aaabda2a40e473b10053c54858238861c6a0e5fcc4bd8bfe"
)
EXPECTED_CANDIDATE_COUNT = 17
EXPECTED_CANDIDATE_KEY_SHA256 = (
    "78ea9d3d8e7ecd2c6870fb62ccee68b72b97a758fa0d65e6ebb3f0419ec72498"
)
EXPECTED_NEXT498_SEARCH_SHA256 = (
    "2a4eddd818a5e856647c727de932220cadb9808178461e6bbaef3ec2ead05150"
)
NEXT235_REFERENCE_FAILED_COUNT = n262.NEXT235_REFERENCE_FAILED_COUNT
NEXT235_REFERENCE_SHORTFALL = n262.NEXT235_REFERENCE_SHORTFALL
SEARCH_WORKERS = n498.SEARCH_WORKERS
BOUNDARY_FLAGS = n498.BOUNDARY_FLAGS
REQUIRED_STAGES = n498.REQUIRED_STAGES
REQUIRED_DESIGN_STAGES = n498.REQUIRED_DESIGN_STAGES
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n498.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next498_design": n498.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next498_manifest": (
        "08f0c4b10801c412137de8b004093f5bdbcb4df0e551679763f0e3988ddae445"
    ),
    "next498_catalogue": (
        "62f9f7779c8b74419c04f5bc999d3237671e0c2a7bb79eab3ec8a201a0e35c83"
    ),
    "next498_evaluation": (
        "d4c19a2cfbbdbe7cd6e41b2b9af598bd52fd868ed7482f26309de49d664a2c9a"
    ),
    "next498_formula": (
        "5ce73fdcbcb8769ea207f870109f722aa115e9b8f8da974c9488808106fccc61"
    ),
    "next498_search": EXPECTED_NEXT498_SEARCH_SHA256,
}


def select_diagnostic_candidates(published: pd.DataFrame) -> pd.DataFrame:
    try:
        return n262.select_diagnostic_candidates(published)
    except ValueError as exc:
        raise ValueError("NEXT499 published candidate columns differ") from exc


def candidate_key_sha256(frame: pd.DataFrame) -> str:
    try:
        return n262.candidate_key_sha256(frame)
    except ValueError as exc:
        raise ValueError("NEXT499 candidate identity table differs") from exc


def select_closest_residual(frame: pd.DataFrame) -> pd.Series:
    try:
        return n262.select_closest_residual(frame)
    except ValueError as exc:
        raise ValueError("NEXT499 residual population differs") from exc


def _paths(
    *,
    roots: Mapping[str, Path],
    next135_freeze_path: Path,
    design_paths: Mapping[int, Path],
    design_path: Path,
) -> dict[str, Path]:
    paths = n498._paths(
        roots=roots,
        next135_freeze_path=next135_freeze_path,
        design_paths=design_paths,
        design_path=design_path,
    )
    paths["next498_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next498_manifest": roots["next498"] / n498.MANIFEST_NAME,
            "next498_catalogue": roots["next498"] / n498.CATALOGUE_NAME,
            "next498_evaluation": roots["next498"] / n498.EVALUATION_NAME,
            "next498_formula": roots["next498"] / n498.FORMULA_NAME,
            "next498_search": roots["next498"] / n498.SEARCH_NAME,
        }
    )
    return paths


def _verify_next498(paths: Mapping[str, Path], hashes: Mapping[str, str]):
    prior_paths = {key: paths[key] for key in n498.EXPECTED_INPUT_SHA256}
    prior_paths["design"] = paths["next498_design"]
    prior_hashes = {key: hashes[key] for key in n498.EXPECTED_INPUT_SHA256}
    prior_hashes["design"] = hashes["next498_design"]
    prior = n498._verify_next497(prior_paths, prior_hashes)
    manifest = json.loads(paths["next498_manifest"].read_text())
    catalogue = json.loads(paths["next498_catalogue"].read_text())
    evaluation = json.loads(paths["next498_evaluation"].read_text())
    formula = json.loads(paths["next498_formula"].read_text())
    table = pd.read_parquet(paths["next498_search"])
    selected = select_diagnostic_candidates(table)
    expected_outputs = {
        n498.CATALOGUE_NAME: hashes["next498_catalogue"],
        n498.EVALUATION_NAME: hashes["next498_evaluation"],
        n498.FORMULA_NAME: hashes["next498_formula"],
        n498.SEARCH_NAME: hashes["next498_search"],
    }
    if (
        manifest.get("protocol") != n498.PROTOCOL
        or manifest.get("candidate_count") != n498.EXPECTED_CANDIDATE_COUNT
        or manifest.get("eligible_new_candidate_count")
        != n498.EXPECTED_ELIGIBLE_COUNT
        or manifest.get("passes_all_cross_source_discovery_gates") is not False
        or manifest.get("freeze_authorized") is not False
        or manifest.get("next499_diagnostic_authorized") is not True
        or manifest.get("next499_candidate_count") != EXPECTED_CANDIDATE_COUNT
        or manifest.get("next499_candidate_key_sha256")
        != EXPECTED_CANDIDATE_KEY_SHA256
        or manifest.get("post_coverage_extension") is not True
        or manifest.get("prospective_confirmation_claim") is not False
        or manifest.get("outputs_sha256") != expected_outputs
        or manifest.get("executed_source_sha256", {}).get(
            "src/next498_cclab_cde_margin_local_search.py"
        )
        != EXPECTED_NEXT498_SOURCE_SHA256
        or _sha256_file(Path(n498.__file__).resolve())
        != EXPECTED_NEXT498_SOURCE_SHA256
        or _sha256_file(Path(n262.__file__).resolve())
        != EXPECTED_NEXT262_SOURCE_SHA256
        or any(manifest.get(key) is not value for key, value in BOUNDARY_FLAGS.items())
        or catalogue.get("candidate_count") != n498.EXPECTED_CANDIDATE_COUNT
        or catalogue.get("candidate_grammar_inherited_unchanged_from")
        != n498.n261.PROTOCOL
        or catalogue.get("post_coverage_extension") is not True
        or catalogue.get("prospective_confirmation_claim") is not False
        or evaluation.get("passes_all_cross_source_discovery_gates") is not False
        or evaluation.get("next499_candidate_count") != EXPECTED_CANDIDATE_COUNT
        or evaluation.get("next499_candidate_key_sha256")
        != EXPECTED_CANDIDATE_KEY_SHA256
        or formula.get("kind")
        != "cclab_cde_triangular_margin_local_x0_no_dft_score"
        or formula.get("selected") is not True
        or formula.get("post_coverage_extension") is not True
        or formula.get("prospective_confirmation_claim") is not False
        or any(
            formula.get(key) is None
            for key in ("hypothesis", "feature", "direction")
        )
        or any(
            formula.get(key) is not False
            for key in (
                "dft_values_used_by_executable_formula",
                "learned_energy_force_stress_proxy_used",
                "model_or_proxy_potential_used",
                "physical_relaxation_executed",
            )
        )
        or len(table) != n498.EXPECTED_CANDIDATE_COUNT
        or len(selected) != EXPECTED_CANDIDATE_COUNT
        or candidate_key_sha256(selected) != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT499 NEXT498 provenance differs")
    return (*prior, selected)


def run_cclab_cde_broad_diagnostic(
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
    next496_dir: Path,
    next497_dir: Path,
    next498_dir: Path,
    output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Reproduce NEXT498 and compute its unchanged discovery BROAD residual."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs) or not set(
        REQUIRED_DESIGN_STAGES
    ).issubset(design_paths):
        raise ValueError("NEXT499 input universe differs")
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
        "next496": Path(next496_dir).resolve(),
        "next497": Path(next497_dir).resolve(),
        "next498": Path(next498_dir).resolve(),
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
    if type(search_workers) is not int or search_workers <= 0:
        raise ValueError("NEXT499 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT499 input is missing")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(hashes) | set(EXPECTED_INPUT_SHA256)
            if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT499 formal input identity differs: {differing}")
    (
        eligible_prior,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        current_key,
        formula222,
        cclab_tables,
        eligible497,
        published498,
    ) = _verify_next498(paths, hashes)
    combined, feature_tables, base_score, base_support, endpoint, _ = (
        n498.n227._reconstruct_next224_frontier(
            paths=paths,
            eligible=eligible_prior,
            eligible214=eligible214,
            primary_key=primary_key,
            base_start_key=base_start_key,
            formula214=formula214,
            current_key=current_key,
            formula222=formula222,
        )
    )
    n498._attach_cclab_cde_features(
        combined=combined,
        feature_tables=feature_tables,
        cclab_tables=cclab_tables,
    )
    diagnostic224 = json.loads(paths["next224_diagnostic"].read_text())
    base_key = str(diagnostic224["global_closest"]["candidate_key"])
    all_specs = n498.build_cclab_cde_candidate_specs(
        base_candidate_key=base_key,
        eligible_table=eligible497,
    )
    keys = set(published498["candidate_key"].astype(str))
    specs = [spec for spec in all_specs if str(spec["candidate_key"]) in keys]
    if len(specs) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("NEXT499 selected specification universe differs")
    virtual, terms, runtime, activity = n498.materialize_cclab_cde_candidates(
        features=combined,
        base_score=base_score,
        base_support=base_support,
        specs=specs,
    )
    evaluator = (
        n498.n223.n222.n215.n214.n212.n210.n208.n205.n203.n202.n200.n194.n130.n125
        .search_optional_guard_laws_parallel
    )
    rerun = evaluator(
        features=virtual,
        endpoint=endpoint,
        old_terms=terms,
        optional_terms=[],
        candidate_specs=runtime,
        workers=search_workers,
    )
    n164 = n498.n223.n222.n215.n214.n164
    n164._verify_reproduction(
        rerun=rerun["candidate_records"], published=published498
    )
    folds = n164.assign_group_folds(combined["reduced_formula"].astype(str).to_numpy())
    sources = combined["source_dataset"].astype(str).to_numpy()
    cells = n164.build_source_fold_cells(source=sources, folds=folds)
    pauling_by_cell = {
        str(cell["cell_id"]): n164._pauling_baseline(
            combined.loc[np.asarray(cell["mask"], dtype=bool)],
            endpoint[np.asarray(cell["mask"], dtype=bool)],
        )
        for cell in cells
    }
    spec_by_key = {str(spec["candidate_key"]): spec for spec in specs}
    published_by_key = published498.set_index("candidate_key", drop=False)
    records: list[dict[str, object]] = []
    frequency: Counter[str] = Counter()
    for term in terms:
        key = str(term["physical_candidate_key"])
        spec = spec_by_key[key]
        row = published_by_key.loc[key]
        score, support = n164._term_risk(virtual, term)
        if not np.array_equal(support, base_support):
            raise RuntimeError("NEXT499 candidate support differs from NEXT214")
        tables = n164._threshold_tables(
            score=score,
            supported=support,
            endpoint=endpoint,
            cells=cells,
        )
        if tables is None:
            raise RuntimeError("NEXT499 candidate has no threshold table")
        residual = n164.diagnose_broad_threshold_tables(
            tables=tables,
            cells=cells,
            pauling_by_cell=pauling_by_cell,
            safe_threshold=float(row["safe_threshold"]),
        )
        if residual["passes_broad"]:
            raise RuntimeError("NEXT499 contradicts NEXT498 BROAD result")
        for failure in residual["failures"]:
            frequency[f"{failure['cell_id']}::{failure['component']}"] += 1
        records.append(
            {
                "candidate_key": key,
                "hypothesis": str(spec["hypothesis"]),
                "feature": str(spec["feature"]),
                "direction": str(spec["direction"]),
                "q_lo": float(spec["q_lo"]),
                "q_hi": float(spec["q_hi"]),
                "local_width_fraction": float(spec["local_width_fraction"]),
                "amplitude_fraction": float(spec["amplitude_fraction"]),
                "local_active_rows": activity[key]["rows"],
                "safe_threshold": float(row["safe_threshold"]),
                "best_threshold": float(residual["best_threshold"]),
                "failed_constraint_count": int(residual["failed_constraint_count"]),
                "normalized_shortfall_sum": float(
                    residual["normalized_shortfall_sum"]
                ),
                "eligible_threshold_count": int(
                    residual["eligible_threshold_count"]
                ),
                "failures_json": json.dumps(
                    residual["failures"], sort_keys=True, separators=(",", ":")
                ),
            }
        )
    per_candidate = pd.DataFrame(records).sort_values(
        "candidate_key", kind="mergesort"
    ).reset_index(drop=True)
    if (
        len(per_candidate) != EXPECTED_CANDIDATE_COUNT
        or candidate_key_sha256(per_candidate) != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise RuntimeError("NEXT499 residual population differs")
    closest = select_closest_residual(per_candidate)
    improves = n498.n223.n222.strictly_improves(
        closest,
        {
            "failed_constraint_count": NEXT235_REFERENCE_FAILED_COUNT,
            "normalized_shortfall_sum": NEXT235_REFERENCE_SHORTFALL,
        },
    )

    def residual_record(row: pd.Series) -> dict[str, object]:
        return {
            "candidate_key": str(row["candidate_key"]),
            "hypothesis": str(row["hypothesis"]),
            "feature": str(row["feature"]),
            "direction": str(row["direction"]),
            "q_lo": float(row["q_lo"]),
            "q_hi": float(row["q_hi"]),
            "local_width_fraction": float(row["local_width_fraction"]),
            "amplitude_fraction": float(row["amplitude_fraction"]),
            "local_active_rows": int(row["local_active_rows"]),
            "safe_threshold": float(row["safe_threshold"]),
            "best_threshold": float(row["best_threshold"]),
            "failed_constraint_count": int(row["failed_constraint_count"]),
            "normalized_shortfall_sum": float(row["normalized_shortfall_sum"]),
            "failures": json.loads(str(row["failures_json"])),
        }

    summary = {
        "protocol": PROTOCOL,
        "diagnostic_mode": (
            "offline_discovery_label_next498_post_coverage_broad_residual"
        ),
        "candidate_count": len(per_candidate),
        "candidate_key_sha256": EXPECTED_CANDIDATE_KEY_SHA256,
        "next498_record_population_reproduced": True,
        "next498_candidate_universe_reproduced": True,
        "global_closest": residual_record(closest),
        "next235_reference_failed_constraint_count": NEXT235_REFERENCE_FAILED_COUNT,
        "next235_reference_normalized_shortfall_sum": NEXT235_REFERENCE_SHORTFALL,
        "normalized_shortfall_reduction_from_next235": (
            NEXT235_REFERENCE_SHORTFALL
            - float(closest["normalized_shortfall_sum"])
        ),
        "strict_improvement_over_next235_diagnostic": improves,
        "failure_frequency": dict(sorted(frequency.items())),
        "cclab_cde_post_coverage_branch_closed": True,
        "continuation_requires_distinct_preoutcome_freeze": True,
        "post_coverage_extension": True,
        "prospective_confirmation_claim": False,
        "new_formula_searched": False,
        "new_formula_selected": False,
        "validation_outputs_opened": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    sources_to_hash = {
        "src/next164_interior_attenuation_broad_residual.py": Path(
            n164.__file__
        ).resolve(),
        "src/next262_pvbp_broad_diagnostic.py": Path(n262.__file__).resolve(),
        "src/next498_cclab_cde_margin_local_search.py": Path(n498.__file__).resolve(),
        "src/next499_cclab_cde_broad_diagnostic.py": Path(__file__).resolve(),
    }
    source_hashes = {
        name: _sha256_file(path) for name, path in sources_to_hash.items()
    }
    try:
        diagnostic_path = staging / DIAGNOSTIC_NAME
        table_path = staging / TABLE_NAME
        _write_json(diagnostic_path, summary)
        per_candidate.to_parquet(table_path, index=False)
        outputs = [diagnostic_path, table_path]
        manifest = {
            "protocol": PROTOCOL,
            "candidate_count": len(per_candidate),
            "candidate_key_sha256": EXPECTED_CANDIDATE_KEY_SHA256,
            "next498_record_population_reproduced": True,
            "next498_candidate_universe_reproduced": True,
            "next498_all_gate_candidate_count": 0,
            "strict_residual_improvement_observed": improves,
            "strict_improvement_over_next235_diagnostic": improves,
            "new_formula_searched": False,
            "new_formula_selected": False,
            "cclab_cde_post_coverage_branch_closed": True,
            "continuation_requires_distinct_preoutcome_freeze": True,
            "post_coverage_extension": True,
            "prospective_confirmation_claim": False,
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
            raise RuntimeError("NEXT499 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in sources_to_hash.items()
        ):
            raise RuntimeError("NEXT499 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    args = parser.parse_args()
    root = args.formal_root.resolve()
    manifest = run_cclab_cde_broad_diagnostic(
        scigen_feature_dir=root / "next85_scigen_label_free_features_v1",
        scigen_discovery_endpoint_dir=root / "next86_scigen_discovery_endpoints_v1",
        wyformer_feature_dir=root / "next94_wyformer_label_free_features_v1",
        wyformer_discovery_endpoint_dir=(
            root / "next93b_wyformer_blind_discovery_endpoint_lockbox_v1"
        ),
        stage_dirs=n498.n497._resolve_stage_dirs(root),
        next135_freeze_path=(
            n498._REPOSITORY
            / "docs/plans/2026-08-08-next135-conjunctive-compactness-search-freeze.json"
        ),
        design_paths=n498.n497._resolve_design_paths(),
        design_path=(
            n498._REPOSITORY
            / "docs/plans/2026-08-13-next498-cclab-cde-margin-local-search.md"
        ),
        next412_dir=root / "next412_same_sign_shell_purity_v1",
        next496_dir=root / "next496_cclab_conservative_domain_extension_v1",
        next497_dir=root / "next497_cclab_cde_feature_audit_v1",
        next498_dir=root / "next498_cclab_cde_margin_local_search_v1",
        output_dir=args.output_dir,
        search_workers=args.search_workers,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
