#!/usr/bin/env python3
"""Diagnose the frozen NEXT208 AUC+SAFE, non-BROAD residual population."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import src.next164_interior_attenuation_broad_residual as n164
import src.next208_residual_x0_exception_search as n208
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-08-next209-residual-x0-broad-diagnostic-v1"
MANIFEST_NAME = "MANIFEST.json"
DIAGNOSTIC_NAME = "NEXT209_RESIDUAL_X0_BROAD_DIAGNOSTIC.json"
TABLE_NAME = "next209_residual_x0_broad_diagnostic.parquet"
EXPECTED_DESIGN_SHA256 = (
    "6a2b6debb1974231a9b962d8d2d46d7dc7dd6d82a688c31d6b5bc15966076493"
)
EXPECTED_CANDIDATE_COUNT = 1
EXPECTED_CANDIDATE_KEY_SHA256 = (
    "e1f1ab49dfe24fa449275bf24ab8882f2850be53ccc6422aa3674516d9feb312"
)
SEARCH_WORKERS = 4
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n208.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next208_design": n208.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next208_manifest": (
        "ba61979f3731bb0f26da817e71a313b15b6f42580c2d6edbc67ec91553bff9a0"
    ),
    "next208_catalogue": (
        "b8d35584f51d1e16a47fce17daa49bf9fdb891344a62a8052a2c0af27c480511"
    ),
    "next208_evaluation": (
        "3e1f3c58af5e917972b13ac1de9f556c5520520211a53df10c7b63e161ee8313"
    ),
    "next208_formula": (
        "58d4bc7a86784b8c08d921bb54e9ff7f6146eb15566142efa97c39547feff649"
    ),
    "next208_search": (
        "b04fbef394eeedca385cf7d3609e2a088ae9cfe09b774c46fb30c4e632396860"
    ),
}


def select_diagnostic_candidates(published: pd.DataFrame) -> pd.DataFrame:
    """Select the exact AUC+SAFE, non-BROAD NEXT208 population."""

    required = {
        "candidate_key",
        "passes_source_auc_gates",
        "passes_safe_all_cells",
        "passes_broad_all_cells",
    }
    if (
        not isinstance(published, pd.DataFrame)
        or required - set(published.columns)
        or published["candidate_key"].astype(str).duplicated().any()
    ):
        raise ValueError("NEXT209 published candidate schema differs")
    mask = (
        published["passes_source_auc_gates"].fillna(False).astype(bool)
        & published["passes_safe_all_cells"].fillna(False).astype(bool)
        & ~published["passes_broad_all_cells"].fillna(False).astype(bool)
    )
    return published.loc[mask].sort_values(
        "candidate_key", kind="mergesort"
    ).reset_index(drop=True)


def candidate_key_sha256(frame: pd.DataFrame) -> str:
    """Hash sorted candidate keys with the frozen newline-joined convention."""

    if (
        not isinstance(frame, pd.DataFrame)
        or "candidate_key" not in frame.columns
        or frame["candidate_key"].astype(str).duplicated().any()
    ):
        raise ValueError("NEXT209 candidate identity table differs")
    keys = sorted(frame["candidate_key"].astype(str))
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def verify_sole_inactive_base(frame: pd.DataFrame) -> pd.Series:
    """Require that the sole diagnostic survivor activates no x0 exception."""

    required = {
        "candidate_key",
        "feature",
        "direction",
        "comparison",
        "cutoff",
        "exception_fraction_numerator",
        "exception_active_rows",
        "exception_active_scigen",
        "exception_active_wyformer",
    }
    if not isinstance(frame, pd.DataFrame) or required - set(frame.columns) or len(frame) != 1:
        raise ValueError("NEXT209 sole candidate is not the inactive base")
    row = frame.iloc[0]
    if (
        any(not pd.isna(row[name]) for name in ("feature", "direction", "comparison", "cutoff"))
        or int(row["exception_fraction_numerator"]) != 0
        or int(row["exception_active_rows"]) != 0
        or int(row["exception_active_scigen"]) != 0
        or int(row["exception_active_wyformer"]) != 0
    ):
        raise ValueError("NEXT209 sole candidate is not the inactive base")
    return row


def select_closest_residual(frame: pd.DataFrame) -> pd.Series:
    """Select the closest residual under the frozen deterministic ordering."""

    required = {
        "candidate_key",
        "failed_constraint_count",
        "normalized_shortfall_sum",
    }
    if not isinstance(frame, pd.DataFrame) or frame.empty or required - set(frame.columns):
        raise ValueError("NEXT209 closest residual population differs")
    return frame.sort_values(
        ["failed_constraint_count", "normalized_shortfall_sum", "candidate_key"],
        kind="mergesort",
    ).iloc[0]


def _paths(
    roots: Mapping[str, Path],
    freeze_path: Path,
    next202_design_path: Path,
    next205_design_path: Path,
    next207_design_path: Path,
    next208_design_path: Path,
    design_path: Path,
) -> dict[str, Path]:
    paths = n208._paths(
        roots,
        freeze_path,
        next202_design_path,
        next205_design_path,
        next207_design_path,
        next208_design_path,
    )
    paths["next208_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next208_manifest": roots["next208"] / n208.MANIFEST_NAME,
            "next208_catalogue": roots["next208"] / n208.CATALOGUE_NAME,
            "next208_evaluation": roots["next208"] / n208.EVALUATION_NAME,
            "next208_formula": roots["next208"] / n208.FORMULA_NAME,
            "next208_search": roots["next208"] / n208.SEARCH_NAME,
        }
    )
    return paths


def _verify_next208(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> tuple[tuple[str, ...], pd.DataFrame]:
    prior_paths = dict(paths)
    prior_paths["design"] = paths["next208_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next208_design"]
    eligible = n208._verify_next207(prior_paths, prior_hashes)
    manifest = json.loads(paths["next208_manifest"].read_text())
    catalogue = json.loads(paths["next208_catalogue"].read_text())
    evaluation = json.loads(paths["next208_evaluation"].read_text())
    formula = json.loads(paths["next208_formula"].read_text())
    published = pd.read_parquet(paths["next208_search"])
    expected_outputs = {
        n208.CATALOGUE_NAME: input_hashes["next208_catalogue"],
        n208.EVALUATION_NAME: input_hashes["next208_evaluation"],
        n208.FORMULA_NAME: input_hashes["next208_formula"],
        n208.SEARCH_NAME: input_hashes["next208_search"],
    }
    if (
        manifest.get("protocol") != n208.PROTOCOL
        or manifest.get("candidate_count") != n208.EXPECTED_CANDIDATE_COUNT
        or manifest.get("eligible_hypothesis_count") != n208.EXPECTED_ELIGIBLE_COUNT
        or manifest.get("passes_all_cross_source_discovery_gates") is not False
        or manifest.get("freeze_authorized") is not False
        or manifest.get("next209_diagnostic_authorized") is not True
        or manifest.get("next209_candidate_count") != EXPECTED_CANDIDATE_COUNT
        or manifest.get("next209_candidate_key_sha256")
        != EXPECTED_CANDIDATE_KEY_SHA256
        or manifest.get("outputs_sha256") != expected_outputs
        or any(manifest.get(key) is not value for key, value in n208.BOUNDARY_FLAGS.items())
        or manifest.get("executed_source_sha256", {}).get(
            "src/next208_residual_x0_exception_search.py"
        )
        != _sha256_file(Path(n208.__file__).resolve())
        or catalogue.get("protocol") != n208.PROTOCOL
        or catalogue.get("candidate_count") != n208.EXPECTED_CANDIDATE_COUNT
        or catalogue.get("cutoff_fit_uses_endpoint") is not False
        or evaluation.get("protocol") != n208.PROTOCOL
        or evaluation.get("passes_all_cross_source_discovery_gates") is not False
        or evaluation.get("next209_diagnostic_authorized") is not True
        or evaluation.get("next209_candidate_count") != EXPECTED_CANDIDATE_COUNT
        or evaluation.get("next209_candidate_key_sha256")
        != EXPECTED_CANDIDATE_KEY_SHA256
        or formula.get("protocol") != n208.PROTOCOL
        or len(published) != n208.EXPECTED_CANDIDATE_COUNT
    ):
        raise ValueError("NEXT209 NEXT208 provenance differs")
    diagnostic = select_diagnostic_candidates(published)
    if (
        len(diagnostic) != EXPECTED_CANDIDATE_COUNT
        or candidate_key_sha256(diagnostic) != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT209 frozen diagnostic population differs")
    verify_sole_inactive_base(diagnostic)
    return eligible, diagnostic


def _same_failure_records(left: object, right: object) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def run_residual_x0_broad_diagnostic(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    next98_dir: Path,
    next110_dir: Path,
    next111_dir: Path,
    next113_dir: Path,
    next114_dir: Path,
    next116_dir: Path,
    next117_dir: Path,
    next120_dir: Path,
    next121_dir: Path,
    next122_dir: Path,
    next124_dir: Path,
    next125_dir: Path,
    next129_dir: Path,
    next130_dir: Path,
    next133_dir: Path,
    next134_dir: Path,
    next163_dir: Path,
    next164_dir: Path,
    next168_dir: Path,
    next173_dir: Path,
    next179_dir: Path,
    next180_dir: Path,
    next181_dir: Path,
    next182_dir: Path,
    next183_dir: Path,
    next184_dir: Path,
    next185_dir: Path,
    next186_dir: Path,
    next188_dir: Path,
    next190_dir: Path,
    next192_dir: Path,
    next194_dir: Path,
    next199_dir: Path,
    next200_dir: Path,
    next201_dir: Path,
    next202_dir: Path,
    next203_dir: Path,
    next204_dir: Path,
    next205_dir: Path,
    next206_dir: Path,
    next207_dir: Path,
    next208_dir: Path,
    next135_freeze_path: Path,
    next202_design_path: Path,
    next205_design_path: Path,
    next207_design_path: Path,
    next208_design_path: Path,
    design_path: Path,
    output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT209 residual diagnostic."""

    stage_values = (
        (98, next98_dir), (110, next110_dir), (111, next111_dir),
        (113, next113_dir), (114, next114_dir), (116, next116_dir),
        (117, next117_dir), (120, next120_dir), (121, next121_dir),
        (122, next122_dir), (124, next124_dir), (125, next125_dir),
        (129, next129_dir), (130, next130_dir), (133, next133_dir),
        (134, next134_dir), (163, next163_dir), (164, next164_dir),
        (168, next168_dir), (173, next173_dir), (179, next179_dir),
        (180, next180_dir), (181, next181_dir), (182, next182_dir),
        (183, next183_dir), (184, next184_dir), (185, next185_dir),
        (186, next186_dir), (188, next188_dir), (190, next190_dir),
        (192, next192_dir),
    )
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{f"next{stage}": Path(value).resolve() for stage, value in stage_values},
        "next194": Path(next194_dir).resolve(),
        "next199": Path(next199_dir).resolve(),
        "next200": Path(next200_dir).resolve(),
        "next201": Path(next201_dir).resolve(),
        "next202": Path(next202_dir).resolve(),
        "next203": Path(next203_dir).resolve(),
        "next204": Path(next204_dir).resolve(),
        "next205": Path(next205_dir).resolve(),
        "next206": Path(next206_dir).resolve(),
        "next207": Path(next207_dir).resolve(),
        "next208": Path(next208_dir).resolve(),
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots,
        Path(next135_freeze_path).resolve(),
        Path(next202_design_path).resolve(),
        Path(next205_design_path).resolve(),
        Path(next207_design_path).resolve(),
        Path(next208_design_path).resolve(),
        Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(search_workers) is not int or search_workers <= 0:
        raise ValueError("NEXT209 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT209 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT209 formal input identity differs: {differing}")
    eligible, published = _verify_next208(paths, input_hashes)
    published_row = verify_sole_inactive_base(published)
    (
        combined,
        _,
        candidate_key,
        base_score,
        base_support,
        endpoint,
    ) = n208._reconstruct_next206(paths=paths)
    specs = n208.build_candidate_specs(
        base_candidate_key=candidate_key,
        eligible_hypotheses=eligible,
        features=combined,
        base_score=base_score,
        base_support=base_support,
        residual_threshold=n208.n207.EXPECTED_RESIDUAL_THRESHOLD,
    )
    published_all = pd.read_parquet(paths["next208_search"])
    if (
        len(specs) != n208.EXPECTED_CANDIDATE_COUNT
        or {str(spec["candidate_key"]) for spec in specs}
        != set(published_all["candidate_key"].astype(str))
    ):
        raise ValueError("NEXT209 NEXT208 candidate universe differs")
    diagnostic_key = str(published_row["candidate_key"])
    selected_specs = [
        spec for spec in specs if str(spec["candidate_key"]) == diagnostic_key
    ]
    if len(selected_specs) != 1 or selected_specs[0]["feature"] is not None:
        raise ValueError("NEXT209 inactive-base spec differs")
    combined_virtual, terms, runtime = (
        n208.materialize_residual_x0_exception_candidates(
            features=combined,
            base_score=base_score,
            base_support=base_support,
            specs=selected_specs,
        )
    )
    rerun = n208.n205.n203.n202.n200.n194.n130.n125.search_optional_guard_laws_parallel(
        features=combined_virtual,
        endpoint=endpoint,
        old_terms=terms,
        optional_terms=[],
        candidate_specs=runtime,
        workers=search_workers,
    )
    n164._verify_reproduction(
        rerun=rerun["candidate_records"], published=published
    )

    score, supported = n208.n205.n203.n202.n200.n194.n87._term_risk(
        combined_virtual, terms[0]
    )
    np.testing.assert_allclose(score[supported], base_score[supported])
    if not np.array_equal(supported, base_support):
        raise RuntimeError("NEXT209 inactive base support differs")
    folds = n164.assign_group_folds(
        combined_virtual["reduced_formula"].astype(str).to_numpy()
    )
    sources = combined_virtual["source_dataset"].astype(str).to_numpy()
    cells = n164.build_source_fold_cells(source=sources, folds=folds)
    pauling_by_cell = {
        str(cell["cell_id"]): n164._pauling_baseline(
            combined_virtual.loc[np.asarray(cell["mask"], dtype=bool)],
            endpoint[np.asarray(cell["mask"], dtype=bool)],
        )
        for cell in cells
    }
    tables = n164._threshold_tables(
        score=score, supported=supported, endpoint=endpoint, cells=cells
    )
    if tables is None:
        raise RuntimeError("NEXT209 candidate has no threshold table")
    residual = n164.diagnose_broad_threshold_tables(
        tables=tables,
        cells=cells,
        pauling_by_cell=pauling_by_cell,
        safe_threshold=float(published_row["safe_threshold"]),
    )
    if residual["passes_broad"]:
        raise RuntimeError("NEXT209 contradicts NEXT208 BROAD result")
    reference = json.loads(paths["next206_diagnostic"].read_text())[
        "global_closest"
    ]
    if (
        int(residual["failed_constraint_count"])
        != int(reference["failed_constraint_count"])
        or not math.isclose(
            float(residual["normalized_shortfall_sum"]),
            float(reference["normalized_shortfall_sum"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            float(residual["best_threshold"]),
            float(reference["best_threshold"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not _same_failure_records(residual["failures"], reference["failures"])
    ):
        raise RuntimeError("NEXT209 does not reproduce NEXT206 closest residual")
    record = {
        "candidate_key": diagnostic_key,
        "feature": None,
        "direction": None,
        "cutoff": None,
        "exception_fraction_numerator": 0,
        "exception_active_rows": 0,
        "safe_threshold": float(published_row["safe_threshold"]),
        "best_threshold": float(residual["best_threshold"]),
        "failed_constraint_count": int(residual["failed_constraint_count"]),
        "normalized_shortfall_sum": float(residual["normalized_shortfall_sum"]),
        "eligible_threshold_count": int(residual["eligible_threshold_count"]),
        "failures_json": json.dumps(
            residual["failures"], sort_keys=True, separators=(",", ":")
        ),
    }
    per_candidate = pd.DataFrame([record])
    closest = select_closest_residual(per_candidate)
    frequency: Counter[str] = Counter(
        f"{failure['cell_id']}::{failure['component']}"
        for failure in residual["failures"]
    )
    summary = {
        "protocol": PROTOCOL,
        "diagnostic_mode": "offline_discovery_label_next208_broad_constraint_residual",
        "candidate_count": len(per_candidate),
        "candidate_key_sha256": candidate_key_sha256(published),
        "only_candidate_is_unchanged_next206_base": True,
        "new_x0_exception_candidates_reaching_auc_and_safe_count": 0,
        "next208_record_reproduced": True,
        "next206_global_closest_residual_reproduced": True,
        "global_closest": {
            "candidate_key": str(closest["candidate_key"]),
            "feature": None,
            "direction": None,
            "exception_active_rows": 0,
            "safe_threshold": float(closest["safe_threshold"]),
            "best_threshold": float(closest["best_threshold"]),
            "failed_constraint_count": int(closest["failed_constraint_count"]),
            "normalized_shortfall_sum": float(
                closest["normalized_shortfall_sum"]
            ),
            "failures": json.loads(str(closest["failures_json"])),
        },
        "failure_frequency": dict(sorted(frequency.items())),
        "existing_raw_x0_single_exception_branch_closed": True,
        "new_formula_searched": False,
        "new_formula_selected": False,
        "validation_outputs_opened": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    source_paths = {
        "src/next164_interior_attenuation_broad_residual.py": Path(
            n164.__file__
        ).resolve(),
        "src/next208_residual_x0_exception_search.py": Path(
            n208.__file__
        ).resolve(),
        "src/next209_residual_x0_broad_diagnostic.py": Path(__file__).resolve(),
    }
    source_hashes = {
        name: _sha256_file(path) for name, path in source_paths.items()
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
            "only_candidate_is_unchanged_next206_base": True,
            "new_x0_exception_candidates_reaching_auc_and_safe_count": 0,
            "next208_record_reproduced": True,
            "next206_global_closest_residual_reproduced": True,
            "new_formula_searched": False,
            "new_formula_selected": False,
            "existing_raw_x0_single_exception_branch_closed": True,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True,
            **n208.BOUNDARY_FLAGS,
            "scientific_improvement_claim": False,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {
                path.name: _sha256_file(path) for path in outputs
            },
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(
            _sha256_file(path) != input_hashes[name]
            for name, path in paths.items()
        ):
            raise RuntimeError("NEXT209 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT209 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scigen-feature-dir", type=Path, required=True)
    parser.add_argument("--scigen-discovery-endpoint-dir", type=Path, required=True)
    parser.add_argument("--wyformer-feature-dir", type=Path, required=True)
    parser.add_argument("--wyformer-discovery-endpoint-dir", type=Path, required=True)
    stages = (
        98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124, 125,
        129, 130, 133, 134, 163, 164, 168, 173, 179, 180, 181, 182,
        183, 184, 185, 186, 188, 190, 192,
    )
    for stage in stages:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    for stage in (194, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208):
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--next202-design-path", type=Path, required=True)
    parser.add_argument("--next205-design-path", type=Path, required=True)
    parser.add_argument("--next207-design-path", type=Path, required=True)
    parser.add_argument("--next208-design-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_residual_x0_broad_diagnostic(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{f"next{stage}_dir": getattr(args, f"next{stage}_dir") for stage in stages},
        **{
            f"next{stage}_dir": getattr(args, f"next{stage}_dir")
            for stage in (194, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208)
        },
        next135_freeze_path=args.next135_freeze_path,
        next202_design_path=args.next202_design_path,
        next205_design_path=args.next205_design_path,
        next207_design_path=args.next207_design_path,
        next208_design_path=args.next208_design_path,
        design_path=args.design_path,
        output_dir=args.output_dir,
        search_workers=args.search_workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "candidate_key_sha256",
    "run_residual_x0_broad_diagnostic",
    "select_closest_residual",
    "select_diagnostic_candidates",
    "verify_sole_inactive_base",
]


if __name__ == "__main__":
    raise SystemExit(main())
