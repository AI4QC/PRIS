#!/usr/bin/env python3
"""Diagnose the frozen NEXT212 AUC+SAFE, non-BROAD residual population."""

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
from typing import Mapping

import numpy as np
import pandas as pd

import src.next164_interior_attenuation_broad_residual as n164
import src.next212_two_signal_risk_lift_search as n212
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-08-next213-two-signal-broad-diagnostic-v1"
MANIFEST_NAME = "MANIFEST.json"
DIAGNOSTIC_NAME = "NEXT213_TWO_SIGNAL_BROAD_DIAGNOSTIC.json"
TABLE_NAME = "next213_two_signal_broad_diagnostic.parquet"
EXPECTED_DESIGN_SHA256 = (
    "215ce456a9b02bf525eececff9958537ea0b2100b27da26432f4300700bcb6e0"
)
EXPECTED_CANDIDATE_COUNT = 92
EXPECTED_CANDIDATE_KEY_SHA256 = (
    "97ee21cc2d73a01ff442fa0c4bf71cb8cd319dc05d0e3a53be3fbcde1d433b1a"
)
EXPECTED_NEXT212_SOURCE_SHA256 = (
    "bec4c1a8221af925e4f049af80b62c2f9a400df098c9500e81a5aa98c3b2f3a8"
)
SEARCH_WORKERS = 4
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n212.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next212_design": n212.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next212_manifest": (
        "67213591db6d05c687482fd19cec53e227270d7598acf854735ad9d3056dc3c1"
    ),
    "next212_catalogue": (
        "8324bedc0cf574b2b5bb5d68c35827d31cf08c88b4ef064cca9d52a219850e76"
    ),
    "next212_evaluation": (
        "c72b2c4a1da3eb06b2e065d786c9536ac6783b75f2c32444d74fb3e9ea75934e"
    ),
    "next212_formula": (
        "bdd91b3a0f4e011e45dc68c33322af68e3d2bf503e88a5a87c6c4fe210337823"
    ),
    "next212_search": (
        "3d34987ec0fe10ba113e99836d76eef288ba0b02f5158f83aa0ea0a402097237"
    ),
}


def select_diagnostic_candidates(published: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate_key", "passes_source_auc_gates", "passes_safe_all_cells",
        "passes_broad_all_cells",
    }
    if (
        not isinstance(published, pd.DataFrame)
        or required - set(published.columns)
        or published["candidate_key"].astype(str).duplicated().any()
    ):
        raise ValueError("NEXT213 published candidate schema differs")
    mask = (
        published["passes_source_auc_gates"].fillna(False).astype(bool)
        & published["passes_safe_all_cells"].fillna(False).astype(bool)
        & ~published["passes_broad_all_cells"].fillna(False).astype(bool)
    )
    return published.loc[mask].sort_values(
        "candidate_key", kind="mergesort"
    ).reset_index(drop=True)


def candidate_key_sha256(frame: pd.DataFrame) -> str:
    if (
        not isinstance(frame, pd.DataFrame)
        or "candidate_key" not in frame.columns
        or frame["candidate_key"].astype(str).duplicated().any()
    ):
        raise ValueError("NEXT213 candidate identity table differs")
    return hashlib.sha256(
        "\n".join(sorted(frame["candidate_key"].astype(str))).encode()
    ).hexdigest()


def select_closest_residual(frame: pd.DataFrame) -> pd.Series:
    required = {"candidate_key", "failed_constraint_count", "normalized_shortfall_sum"}
    if not isinstance(frame, pd.DataFrame) or frame.empty or required - set(frame.columns):
        raise ValueError("NEXT213 closest residual population differs")
    return frame.sort_values(
        ["failed_constraint_count", "normalized_shortfall_sum", "candidate_key"],
        kind="mergesort",
    ).iloc[0]


def _paths(
    roots: Mapping[str, Path], freeze_path: Path, next202_design_path: Path,
    next205_design_path: Path, next207_design_path: Path,
    next208_design_path: Path, next209_design_path: Path,
    next210_design_path: Path, next211_design_path: Path,
    next212_design_path: Path, design_path: Path,
) -> dict[str, Path]:
    paths = n212._paths(
        roots, freeze_path, next202_design_path, next205_design_path,
        next207_design_path, next208_design_path, next209_design_path,
        next210_design_path, next211_design_path, next212_design_path,
    )
    paths["next212_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next212_manifest": roots["next212"] / n212.MANIFEST_NAME,
            "next212_catalogue": roots["next212"] / n212.CATALOGUE_NAME,
            "next212_evaluation": roots["next212"] / n212.EVALUATION_NAME,
            "next212_formula": roots["next212"] / n212.FORMULA_NAME,
            "next212_search": roots["next212"] / n212.SEARCH_NAME,
        }
    )
    return paths


def _verify_next212(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> tuple[tuple[str, ...], str, pd.DataFrame, pd.DataFrame]:
    prior_paths = dict(paths)
    prior_paths["design"] = paths["next212_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next212_design"]
    eligible, anchor_key = n212._verify_next211(prior_paths, prior_hashes)
    manifest = json.loads(paths["next212_manifest"].read_text())
    catalogue = json.loads(paths["next212_catalogue"].read_text())
    evaluation = json.loads(paths["next212_evaluation"].read_text())
    formula = json.loads(paths["next212_formula"].read_text())
    published_all = pd.read_parquet(paths["next212_search"])
    expected_outputs = {
        n212.CATALOGUE_NAME: input_hashes["next212_catalogue"],
        n212.EVALUATION_NAME: input_hashes["next212_evaluation"],
        n212.FORMULA_NAME: input_hashes["next212_formula"],
        n212.SEARCH_NAME: input_hashes["next212_search"],
    }
    if (
        manifest.get("protocol") != n212.PROTOCOL
        or manifest.get("candidate_count") != n212.EXPECTED_CANDIDATE_COUNT
        or manifest.get("secondary_hypothesis_count")
        != n212.EXPECTED_SECONDARY_HYPOTHESIS_COUNT
        or manifest.get("anchor_candidate_key_sha256")
        != n212.EXPECTED_ANCHOR_KEY_SHA256
        or manifest.get("passes_all_cross_source_discovery_gates") is not False
        or manifest.get("freeze_authorized") is not False
        or manifest.get("next213_diagnostic_authorized") is not True
        or manifest.get("next213_candidate_count") != EXPECTED_CANDIDATE_COUNT
        or manifest.get("next213_candidate_key_sha256")
        != EXPECTED_CANDIDATE_KEY_SHA256
        or manifest.get("outputs_sha256") != expected_outputs
        or any(
            manifest.get(key) is not value
            for key, value in n212.n210.n208.BOUNDARY_FLAGS.items()
        )
        or manifest.get("executed_source_sha256", {}).get(
            "src/next212_two_signal_risk_lift_search.py"
        )
        != EXPECTED_NEXT212_SOURCE_SHA256
        or _sha256_file(Path(n212.__file__).resolve())
        != EXPECTED_NEXT212_SOURCE_SHA256
        or catalogue.get("protocol") != n212.PROTOCOL
        or catalogue.get("design_sha256") != input_hashes["next212_design"]
        or catalogue.get("candidate_count") != n212.EXPECTED_CANDIDATE_COUNT
        or catalogue.get("normalization_refit") is not False
        or catalogue.get("base_support_unchanged") is not True
        or evaluation.get("protocol") != n212.PROTOCOL
        or evaluation.get("candidate_count") != n212.EXPECTED_CANDIDATE_COUNT
        or evaluation.get("passes_all_cross_source_discovery_gates") is not False
        or evaluation.get("freeze_authorized") is not False
        or evaluation.get("next213_diagnostic_authorized") is not True
        or evaluation.get("next213_candidate_count") != EXPECTED_CANDIDATE_COUNT
        or evaluation.get("next213_candidate_key_sha256")
        != EXPECTED_CANDIDATE_KEY_SHA256
        or formula.get("protocol") != n212.PROTOCOL
        or formula.get("dft_values_used_by_executable_formula") is not False
        or formula.get("learned_energy_force_stress_proxy_used") is not False
        or formula.get("model_or_proxy_potential_used") is not False
        or formula.get("physical_relaxation_executed") is not False
        or len(published_all) != n212.EXPECTED_CANDIDATE_COUNT
    ):
        raise ValueError("NEXT213 NEXT212 provenance differs")
    published = select_diagnostic_candidates(published_all)
    if (
        len(published) != EXPECTED_CANDIDATE_COUNT
        or candidate_key_sha256(published) != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT213 frozen diagnostic population differs")
    return eligible, anchor_key, published_all, published


def run_two_signal_broad_diagnostic(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    next98_dir: Path, next110_dir: Path, next111_dir: Path,
    next113_dir: Path, next114_dir: Path, next116_dir: Path,
    next117_dir: Path, next120_dir: Path, next121_dir: Path,
    next122_dir: Path, next124_dir: Path, next125_dir: Path,
    next129_dir: Path, next130_dir: Path, next133_dir: Path,
    next134_dir: Path, next163_dir: Path, next164_dir: Path,
    next168_dir: Path, next173_dir: Path, next179_dir: Path,
    next180_dir: Path, next181_dir: Path, next182_dir: Path,
    next183_dir: Path, next184_dir: Path, next185_dir: Path,
    next186_dir: Path, next188_dir: Path, next190_dir: Path,
    next192_dir: Path, next194_dir: Path, next199_dir: Path,
    next200_dir: Path, next201_dir: Path, next202_dir: Path,
    next203_dir: Path, next204_dir: Path, next205_dir: Path,
    next206_dir: Path, next207_dir: Path, next208_dir: Path,
    next209_dir: Path, next210_dir: Path, next211_dir: Path,
    next212_dir: Path,
    next135_freeze_path: Path, next202_design_path: Path,
    next205_design_path: Path, next207_design_path: Path,
    next208_design_path: Path, next209_design_path: Path,
    next210_design_path: Path, next211_design_path: Path,
    next212_design_path: Path, design_path: Path, output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT213 residual diagnostic."""

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
        **{
            f"next{stage}": Path(value).resolve()
            for stage, value in (
                (194, next194_dir), (199, next199_dir), (200, next200_dir),
                (201, next201_dir), (202, next202_dir), (203, next203_dir),
                (204, next204_dir), (205, next205_dir), (206, next206_dir),
                (207, next207_dir), (208, next208_dir), (209, next209_dir),
                (210, next210_dir), (211, next211_dir), (212, next212_dir),
            )
        },
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots, Path(next135_freeze_path).resolve(),
        Path(next202_design_path).resolve(), Path(next205_design_path).resolve(),
        Path(next207_design_path).resolve(), Path(next208_design_path).resolve(),
        Path(next209_design_path).resolve(), Path(next210_design_path).resolve(),
        Path(next211_design_path).resolve(), Path(next212_design_path).resolve(),
        Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(search_workers) is not int or search_workers <= 0:
        raise ValueError("NEXT213 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT213 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT213 formal input identity differs: {differing}")
    eligible, anchor_key, published_all, published = _verify_next212(
        paths, input_hashes
    )
    combined, _, base_key, base_score, base_support, endpoint = (
        n212.n210.n208._reconstruct_next206(paths=paths)
    )
    next210_specs = n212.n210.build_candidate_specs(
        base_candidate_key=base_key, eligible_hypotheses=eligible,
        features=combined, base_score=base_score, base_support=base_support,
        residual_threshold=n212.n210.n208.n207.EXPECTED_RESIDUAL_THRESHOLD,
    )
    anchors = [spec for spec in next210_specs if str(spec["candidate_key"]) == anchor_key]
    if len(anchors) != 1:
        raise ValueError("NEXT213 frozen anchor specification differs")
    anchor_spec = anchors[0]
    anchor_values = pd.to_numeric(
        combined[str(anchor_spec["feature"])], errors="coerce"
    ).to_numpy(float)
    anchor_score, anchor_support, _ = n212.n210.residual_risk_lift_score(
        base_score=base_score, base_support=base_support,
        feature_values=anchor_values, direction=str(anchor_spec["direction"]),
        q_lo=float(anchor_spec["q_lo"]), q_hi=float(anchor_spec["q_hi"]),
        residual_threshold=float(anchor_spec["residual_threshold"]),
        amplitude_fraction=float(anchor_spec["amplitude_fraction"]),
        risk_scale=float(anchor_spec["risk_scale"]),
    )
    all_specs = n212.build_candidate_specs(
        anchor_spec=anchor_spec, next210_specs=next210_specs
    )
    if (
        len(all_specs) != n212.EXPECTED_CANDIDATE_COUNT
        or {str(spec["candidate_key"]) for spec in all_specs}
        != set(published_all["candidate_key"].astype(str))
    ):
        raise ValueError("NEXT213 NEXT212 candidate universe differs")
    diagnostic_keys = set(published["candidate_key"].astype(str))
    selected_specs = [
        spec for spec in all_specs if str(spec["candidate_key"]) in diagnostic_keys
    ]
    if len(selected_specs) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("NEXT213 selected specification universe differs")
    combined_virtual, terms, runtime = n212.materialize_two_signal_candidates(
        features=combined, anchor_score=anchor_score, activation_score=base_score,
        base_support=anchor_support, specs=selected_specs,
    )
    rerun = n212.n210.n208.n205.n203.n202.n200.n194.n130.n125.search_optional_guard_laws_parallel(
        features=combined_virtual, endpoint=endpoint, old_terms=terms,
        optional_terms=[], candidate_specs=runtime, workers=search_workers,
    )
    n164._verify_reproduction(rerun=rerun["candidate_records"], published=published)
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
    spec_by_key = {str(spec["candidate_key"]): spec for spec in selected_specs}
    published_by_key = published.set_index("candidate_key", drop=False)
    records: list[dict[str, object]] = []
    frequency: Counter[str] = Counter()
    for term in terms:
        key = str(term["physical_candidate_key"])
        spec = spec_by_key[key]
        row = published_by_key.loc[key]
        score, supported = n212.n210.n208.n205.n203.n202.n200.n194.n87._term_risk(
            combined_virtual, term
        )
        if not np.array_equal(supported, base_support):
            raise RuntimeError("NEXT213 candidate support differs from NEXT206")
        tables = n164._threshold_tables(
            score=score, supported=supported, endpoint=endpoint, cells=cells
        )
        if tables is None:
            raise RuntimeError("NEXT213 candidate has no threshold table")
        residual = n164.diagnose_broad_threshold_tables(
            tables=tables, cells=cells, pauling_by_cell=pauling_by_cell,
            safe_threshold=float(row["safe_threshold"]),
        )
        if residual["passes_broad"]:
            raise RuntimeError("NEXT213 contradicts NEXT212 BROAD result")
        for failure in residual["failures"]:
            frequency[f"{failure['cell_id']}::{failure['component']}"] += 1
        records.append(
            {
                "candidate_key": key,
                "secondary_hypothesis": spec["secondary_hypothesis"],
                "secondary_feature": spec["secondary_feature"],
                "secondary_direction": spec["secondary_direction"],
                "secondary_amplitude_fraction": float(
                    spec["secondary_amplitude_fraction"]
                ),
                "secondary_q_lo": spec["secondary_q_lo"],
                "secondary_q_hi": spec["secondary_q_hi"],
                "secondary_active_rows": int(row["secondary_active_rows"]),
                "safe_threshold": float(row["safe_threshold"]),
                "best_threshold": float(residual["best_threshold"]),
                "failed_constraint_count": int(residual["failed_constraint_count"]),
                "normalized_shortfall_sum": float(
                    residual["normalized_shortfall_sum"]
                ),
                "eligible_threshold_count": int(residual["eligible_threshold_count"]),
                "failures_json": json.dumps(
                    residual["failures"], sort_keys=True, separators=(",", ":")
                ),
            }
        )
    per_candidate = pd.DataFrame(records).sort_values(
        "candidate_key", kind="mergesort"
    ).reset_index(drop=True)
    closest = select_closest_residual(per_candidate)
    closest_shortfall = float(closest["normalized_shortfall_sum"])
    reference_shortfall = float(
        json.loads(paths["next211_diagnostic"].read_text())["global_closest"][
            "normalized_shortfall_sum"
        ]
    )
    summary = {
        "protocol": PROTOCOL,
        "diagnostic_mode": "offline_discovery_label_next212_broad_constraint_residual",
        "candidate_count": len(per_candidate),
        "candidate_key_sha256": candidate_key_sha256(published),
        "next212_record_population_reproduced": True,
        "next212_candidate_universe_reproduced": True,
        "global_closest": {
            "candidate_key": str(closest["candidate_key"]),
            "secondary_hypothesis": (
                None if pd.isna(closest["secondary_hypothesis"])
                else str(closest["secondary_hypothesis"])
            ),
            "secondary_feature": (
                None if pd.isna(closest["secondary_feature"])
                else str(closest["secondary_feature"])
            ),
            "secondary_direction": (
                None if pd.isna(closest["secondary_direction"])
                else str(closest["secondary_direction"])
            ),
            "secondary_amplitude_fraction": float(
                closest["secondary_amplitude_fraction"]
            ),
            "safe_threshold": float(closest["safe_threshold"]),
            "best_threshold": float(closest["best_threshold"]),
            "failed_constraint_count": int(closest["failed_constraint_count"]),
            "normalized_shortfall_sum": closest_shortfall,
            "failures": json.loads(str(closest["failures_json"])),
        },
        "next211_reference_normalized_shortfall_sum": reference_shortfall,
        "normalized_shortfall_reduction_from_next211": (
            reference_shortfall - closest_shortfall
        ),
        "improves_over_next211_global_residual": bool(
            closest_shortfall + 1.0e-12 < reference_shortfall
        ),
        "failure_frequency": dict(sorted(frequency.items())),
        "two_signal_risk_lift_branch_closed": True,
        "new_formula_searched": False,
        "new_formula_selected": False,
        "validation_outputs_opened": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next164_interior_attenuation_broad_residual.py": Path(n164.__file__).resolve(),
        "src/next212_two_signal_risk_lift_search.py": Path(n212.__file__).resolve(),
        "src/next213_two_signal_broad_diagnostic.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
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
            "next212_record_population_reproduced": True,
            "next212_candidate_universe_reproduced": True,
            "next212_all_gate_candidate_count": 0,
            "new_formula_searched": False,
            "new_formula_selected": False,
            "two_signal_risk_lift_branch_closed": True,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True,
            **n212.n210.n208.BOUNDARY_FLAGS,
            "scientific_improvement_claim": False,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {path.name: _sha256_file(path) for path in outputs},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT213 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT213 source changed before publication")
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
    later_stages = (
        194, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212,
    )
    for stage in stages + later_stages:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    for stage in (202, 205, 207, 208, 209, 210, 211, 212):
        parser.add_argument(f"--next{stage}-design-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_two_signal_broad_diagnostic(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{f"next{stage}_dir": getattr(args, f"next{stage}_dir") for stage in stages + later_stages},
        next135_freeze_path=args.next135_freeze_path,
        **{
            f"next{stage}_design_path": getattr(args, f"next{stage}_design_path")
            for stage in (202, 205, 207, 208, 209, 210, 211, 212)
        },
        design_path=args.design_path,
        output_dir=args.output_dir,
        search_workers=args.search_workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "candidate_key_sha256",
    "run_two_signal_broad_diagnostic",
    "select_closest_residual",
    "select_diagnostic_candidates",
]


if __name__ == "__main__":
    raise SystemExit(main())
