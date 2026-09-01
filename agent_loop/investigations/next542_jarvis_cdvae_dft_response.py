#!/usr/bin/env python3
"""One-shot JARVIS-CDVAE DFT structural-response evaluation of frozen NEXT541."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Poscar
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.next19_feature_build import _publish_directory_no_replace, _sha256
import src.next541_jarvis_cdvae_initial_prediction_freeze as n541


PROTOCOL = "2026-08-13-next542-jarvis-cdvae-dft-response-v1"
NEXT541_PROTOCOL = n541.PROTOCOL
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 54_254_213
MINIMUM_MAPPING_COVERAGE = 0.90
MINIMUM_CLASS_COUNT = 10
MINIMUM_MUPR_COVERAGE = 0.90
MINIMUM_AUC = 0.65
MINIMUM_AUC_LOWER = 0.50
MINIMUM_PAULING_MARGIN = 0.05
MINIMUM_TOP_PRECISION = 0.70
MINIMUM_TOP_PRECISION_LOWER = 0.45
MINIMUM_TOP_RECALL = 0.20
MINIMUM_BOTTOM_NONSEVERE = 0.80
MINIMUM_SPEARMAN = 0.30
MINIMUM_SPEARMAN_LOWER = 0.0
RMS_THRESHOLD = 0.15
MAX_THRESHOLD = 0.30
VOLUME_RATIO_THRESHOLD = 1.25
MAPPING_NAME = "next542_jarvis_cdvae_dft_response_mapping.parquet"
RESULT_NAME = "NEXT542_JARVIS_CDVAE_DFT_RESPONSE.json"
MANIFEST_NAME = "MANIFEST.json"
EXPECTED_INPUT_SHA256 = {
    "design": n541.DESIGN_SHA256,
    "initial_archive": n541.EXPECTED_INPUT_SHA256["initial_archive"],
    "endpoint_archive": n541.EXPECTED_INPUT_SHA256["endpoint_archive"],
    "next541_manifest": "06f4ebcd95173a4d3834df8dc6fb527dc7ac35639247281c61d42c0400d9e028",
    "next541_table": "8d13bcd73bff0c21ce3bc60ecdc84b00880735e5e04223cdabc1f661c51ad997",
    "next541_inventory": "2c346f2186f37c9a56edc5869a47e1f3c38a9eae0548add32c74da86cae800eb",
    "next541_formula": "f5bd63fa1648692a6d83c8fd71c4a5a98ba871dbd0161011539a41f22ae44445",
    "next541_source": "16f612791b256db05e782d4f65dabf8b95320fba48db9f624a7a765ff07cb80b",
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _matcher(tier: int) -> StructureMatcher:
    if tier == 0:
        settings = {"ltol": 0.20, "stol": 0.30, "angle_tol": 5.0}
    elif tier == 1:
        settings = {"ltol": 0.50, "stol": 0.50, "angle_tol": 15.0}
    else:
        raise ValueError("NEXT542 matcher tier differs")
    return StructureMatcher(
        **settings,
        primitive_cell=True,
        scale=True,
        attempt_supercell=True,
        allow_subset=False,
    )


def _pair_match(initial, final) -> tuple[int, float, float] | None:
    for tier in (0, 1):
        result = _matcher(tier).get_rms_dist(initial, final)
        if result is not None:
            rms, maximum = map(float, result)
            if np.isfinite([rms, maximum]).all():
                return tier, rms, maximum
    return None


def match_composition_group(
    initials: list[dict[str, object]], finals: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Globally map final files to distinct initial candidates with frozen costs."""

    initials = sorted(initials, key=lambda row: int(row["initial_index"]))
    finals = sorted(finals, key=lambda row: str(row["endpoint_filename"]))
    n_final, n_initial = len(finals), len(initials)
    if not n_final:
        return []
    pair: dict[tuple[int, int], tuple[int, float, float]] = {}
    cost = np.full((n_final, n_initial + n_final), 1.0e12, dtype=float)
    cost[:, n_initial:] = 1.0e9
    for i, final in enumerate(finals):
        for j, initial in enumerate(initials):
            result = _pair_match(initial["structure"], final["structure"])
            if result is None:
                continue
            tier, rms, maximum = result
            pair[(i, j)] = result
            cost[i, j] = (
                tier * 1.0e6
                + rms * 1.0e3
                + maximum
                + int(initial["initial_index"]) * 1.0e-9
            )
    row_indices, column_indices = linear_sum_assignment(cost)
    chosen = dict(zip(row_indices.tolist(), column_indices.tolist(), strict=True))
    rows: list[dict[str, object]] = []
    for i, final in enumerate(finals):
        j = chosen[i]
        base = {
            "endpoint_member": final.get("endpoint_member", ""),
            "endpoint_filename": str(final["endpoint_filename"]),
            "composition_key": final.get("composition_key", ""),
        }
        if j >= n_initial or (i, j) not in pair:
            rows.append(
                {
                    **base,
                    "mapped": False,
                    "mapping_failure": "no finite one-to-one match in frozen tiers",
                    "initial_index": -1,
                    "match_tier": -1,
                    "normalized_rms": math.nan,
                    "normalized_max": math.nan,
                }
            )
            continue
        tier, rms, maximum = pair[(i, j)]
        initial = initials[j]
        rows.append(
            {
                **base,
                "mapped": True,
                "mapping_failure": "",
                "initial_index": int(initial["initial_index"]),
                "match_tier": int(tier),
                "normalized_rms": float(rms),
                "normalized_max": float(maximum),
                "initial_volume_per_atom": float(
                    initial["structure"].volume / len(initial["structure"])
                ),
                "final_volume_per_atom": float(
                    final["structure"].volume / len(final["structure"])
                ),
            }
        )
    return rows


def endpoint_response(
    *,
    normalized_rms: float,
    normalized_max: float,
    initial_volume_per_atom: float,
    final_volume_per_atom: float,
    match_tier: int,
) -> dict[str, object]:
    values = np.asarray(
        [normalized_rms, normalized_max, initial_volume_per_atom, final_volume_per_atom],
        dtype=float,
    )
    if (
        not np.isfinite(values).all()
        or normalized_rms < 0.0
        or normalized_max < 0.0
        or initial_volume_per_atom <= 0.0
        or final_volume_per_atom <= 0.0
        or match_tier not in (0, 1)
    ):
        raise ValueError("NEXT542 response inputs differ")
    volume_log = abs(math.log(final_volume_per_atom / initial_volume_per_atom))
    severity = max(
        normalized_rms / RMS_THRESHOLD,
        normalized_max / MAX_THRESHOLD,
        volume_log / math.log(VOLUME_RATIO_THRESHOLD),
    )
    severe = bool(
        match_tier == 1
        or normalized_rms > RMS_THRESHOLD
        or normalized_max > MAX_THRESHOLD
        or volume_log > math.log(VOLUME_RATIO_THRESHOLD)
    )
    return {
        "volume_log_response": float(volume_log),
        "response_severity": float(severity),
        "severe_response": severe,
    }


def _wilson(successes: int, rows: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if rows <= 0 or not 0 <= successes <= rows:
        return math.nan, math.nan
    p = successes / rows
    denominator = 1.0 + z * z / rows
    center = (p + z * z / (2.0 * rows)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / rows + z * z / (4.0 * rows * rows)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _cluster_bootstrap(
    table: pd.DataFrame, *, draws: int, seed: int
) -> dict[str, object]:
    if type(draws) is not int or draws < 1:
        raise ValueError("NEXT542 bootstrap draws differ")
    clusters = sorted(table["composition_key"].astype(str).unique())
    groups = {
        cluster: np.flatnonzero(table["composition_key"].astype(str).to_numpy() == cluster)
        for cluster in clusters
    }
    rng = np.random.default_rng(seed)
    auc_values: list[float] = []
    rho_values: list[float] = []
    auc_degenerate = 0
    rho_degenerate = 0
    labels = table["severe_response"].to_numpy(bool)
    risks = table["mupr_risk"].to_numpy(float)
    severity = table["response_severity"].to_numpy(float)
    for _ in range(draws):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        indices = np.concatenate([groups[str(cluster)] for cluster in sampled])
        y = labels[indices]
        if np.unique(y).size == 2:
            auc_values.append(float(roc_auc_score(y, risks[indices])))
        else:
            auc_degenerate += 1
        rho = float(spearmanr(risks[indices], severity[indices]).statistic)
        if math.isfinite(rho):
            rho_values.append(rho)
        else:
            rho_degenerate += 1
    def interval(values: list[float]) -> dict[str, float | int]:
        if not values:
            return {"lower": math.nan, "median": math.nan, "upper": math.nan, "valid": 0}
        array = np.asarray(values, dtype=float)
        return {
            "lower": float(np.quantile(array, 0.025)),
            "median": float(np.quantile(array, 0.5)),
            "upper": float(np.quantile(array, 0.975)),
            "valid": len(values),
        }
    return {
        "draws": draws,
        "seed": seed,
        "clusters": len(clusters),
        "roc_auc": {**interval(auc_values), "degenerate": auc_degenerate},
        "spearman": {**interval(rho_values), "degenerate": rho_degenerate},
    }


def evaluate_screen(
    table: pd.DataFrame, *, bootstrap_draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED
) -> dict[str, object]:
    required = {
        "initial_index",
        "composition_key",
        "mupr_risk",
        "severe_response",
        "response_severity",
    }
    if required - set(table):
        raise ValueError("NEXT542 evaluation table differs")
    work = table.copy()
    work["mupr_risk"] = pd.to_numeric(work["mupr_risk"], errors="coerce")
    work["response_severity"] = pd.to_numeric(work["response_severity"], errors="coerce")
    work = work.loc[
        np.isfinite(work["mupr_risk"]) & np.isfinite(work["response_severity"])
    ].copy()
    if work.empty or work["severe_response"].nunique() != 2:
        raise ValueError("NEXT542 evaluation requires two endpoint classes")
    work["severe_response"] = work["severe_response"].astype(bool)
    ordered = work.sort_values(
        ["mupr_risk", "initial_index"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    top_rows = max(1, int(math.ceil(0.15 * len(ordered))))
    bottom_rows = max(1, int(math.ceil(0.50 * len(ordered))))
    top = ordered.iloc[:top_rows]
    bottom = ordered.iloc[-bottom_rows:]
    positives = int(top["severe_response"].sum())
    total_positives = int(ordered["severe_response"].sum())
    precision = positives / top_rows
    precision_lower, precision_upper = _wilson(positives, top_rows)
    auc = float(roc_auc_score(ordered["severe_response"], ordered["mupr_risk"]))
    rho = float(spearmanr(ordered["mupr_risk"], ordered["response_severity"]).statistic)
    bootstrap = _cluster_bootstrap(ordered, draws=bootstrap_draws, seed=seed)
    return {
        "rows": len(ordered),
        "positives": total_positives,
        "negatives": int(len(ordered) - total_positives),
        "roc_auc": auc,
        "spearman": rho,
        "top_15_percent": {
            "rows": top_rows,
            "positives": positives,
            "precision": precision,
            "precision_wilson_lower": precision_lower,
            "precision_wilson_upper": precision_upper,
            "recall": positives / total_positives,
        },
        "bottom_50_percent": {
            "rows": bottom_rows,
            "nonsevere": int((~bottom["severe_response"]).sum()),
            "nonsevere_fraction": float((~bottom["severe_response"]).mean()),
        },
        "bootstrap": bootstrap,
    }


def _pauling_comparisons(table: pd.DataFrame) -> dict[str, object]:
    candidates: dict[str, np.ndarray] = {
        "pauling_violation_fraction": pd.to_numeric(
            table["pauling_violation_fraction"], errors="coerce"
        ).to_numpy(float),
        "pauling_p2_value": pd.to_numeric(table["pauling_p2_value"], errors="coerce").to_numpy(float),
        "pauling_p3_value": pd.to_numeric(table["pauling_p3_value"], errors="coerce").to_numpy(float),
        "pauling_p4_value": pd.to_numeric(table["pauling_p4_value"], errors="coerce").to_numpy(float),
        "negative_pauling_p5_value": -pd.to_numeric(
            table["pauling_p5_value"], errors="coerce"
        ).to_numpy(float),
    }
    decision = table["pauling_p2_p5_decision"].astype(str)
    candidates["pauling_combined_reject"] = np.where(
        decision.eq("REJECT"), 1.0, np.where(decision.eq("KEEP"), 0.0, np.nan)
    )
    labels = table["severe_response"].to_numpy(bool)
    mupr = table["mupr_risk"].to_numpy(float)
    results: dict[str, object] = {}
    for name, score in candidates.items():
        mask = np.isfinite(score) & np.isfinite(mupr)
        if mask.sum() < 2 or np.unique(labels[mask]).size != 2:
            results[name] = {"rows": int(mask.sum()), "supported": False}
            continue
        results[name] = {
            "rows": int(mask.sum()),
            "supported": True,
            "pauling_auc": float(roc_auc_score(labels[mask], score[mask])),
            "mupr_auc_same_rows": float(roc_auc_score(labels[mask], mupr[mask])),
        }
        results[name]["mupr_margin"] = (
            results[name]["mupr_auc_same_rows"] - results[name]["pauling_auc"]
        )
    supported = {
        name: value
        for name, value in results.items()
        if isinstance(value, dict) and value.get("supported") is True
    }
    best_name = max(supported, key=lambda name: supported[name]["pauling_auc"]) if supported else None
    return {"candidates": results, "best_pauling": best_name, "best": supported.get(best_name) if best_name else None}


def _validate_next541(paths: dict[str, Path], hashes: dict[str, str]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    manifest = _read_json(paths["next541_manifest"])
    inventory = _read_json(paths["next541_inventory"])
    formula = _read_json(paths["next541_formula"])
    if not isinstance(manifest, dict) or not isinstance(inventory, list) or not isinstance(formula, dict):
        raise ValueError("NEXT542 NEXT541 artifact schema differs")
    outputs = manifest.get("outputs_sha256")
    source = manifest.get("executed_source_sha256")
    if (
        manifest.get("protocol") != NEXT541_PROTOCOL
        or manifest.get("endpoint_member_payloads_read") is not False
        or manifest.get("scores_frozen_before_endpoint_member_payloads_read") is not True
        or manifest.get("next542_endpoint_evaluation_authorized") is not True
        or manifest.get("forbidden_model_values_used") is not False
        or not isinstance(outputs, dict)
        or outputs.get(n541.TABLE_NAME) != hashes["next541_table"]
        or outputs.get(n541.INVENTORY_NAME) != hashes["next541_inventory"]
        or outputs.get(n541.FORMULA_NAME) != hashes["next541_formula"]
        or not isinstance(source, dict)
        or source.get("src/next541_jarvis_cdvae_initial_prediction_freeze.py")
        != hashes["next541_source"]
        or formula.get("short_name") != "MUPR"
        or formula.get("coefficients_fitted_to_endpoint") is not False
    ):
        raise ValueError("NEXT542 NEXT541 freeze contract differs")
    table = pd.read_parquet(paths["next541_table"])
    if (
        len(table) != n541.EXPECTED_CANDIDATES
        or table["initial_index"].duplicated().any()
        or not np.isfinite(pd.to_numeric(table["mupr_risk"], errors="coerce")).all()
    ):
        raise ValueError("NEXT542 frozen prediction table differs")
    return table, inventory


def run_dft_response_evaluation(
    *,
    initial_archive: Path,
    endpoint_archive: Path,
    design_path: Path,
    next541_dir: Path,
    output_dir: Path,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    freeze = Path(next541_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "design": Path(design_path).resolve(),
        "initial_archive": Path(initial_archive).resolve(),
        "endpoint_archive": Path(endpoint_archive).resolve(),
        "next541_manifest": freeze / n541.MANIFEST_NAME,
        "next541_table": freeze / n541.TABLE_NAME,
        "next541_inventory": freeze / n541.INVENTORY_NAME,
        "next541_formula": freeze / n541.FORMULA_NAME,
        "next541_source": Path(n541.__file__).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT542 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(hashes) | set(EXPECTED_INPUT_SHA256)
            if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT542 formal input identity differs: {differing}")
    predictions, inventory = _validate_next541(paths, hashes)

    # Reconstruct every possible x0 only after its score and hash are verified frozen.
    records, _masked_counts = n541._read_initial_records(paths["initial_archive"])
    initial_rows: list[dict[str, object]] = []
    adaptor = AseAtomsAdaptor()
    for row in predictions.itertuples(index=False):
        index = int(row.initial_index)
        atoms = n541._atoms_from_initial_record(records[index])
        geometry_bytes = np.asarray(atoms.cell.array, dtype="<f8").tobytes()
        geometry_bytes += np.asarray(atoms.get_scaled_positions(wrap=True), dtype="<f8").tobytes()
        geometry_bytes += "\0".join(atoms.get_chemical_symbols()).encode()
        import hashlib
        if hashlib.sha256(geometry_bytes).hexdigest() != str(row.initial_geometry_sha256):
            raise ValueError("NEXT542 reconstructed initial geometry hash differs")
        initial_rows.append(
            {
                "initial_index": index,
                "composition_key": str(row.composition_key),
                "structure": adaptor.get_structure(atoms),
            }
        )

    # This is the first authorized read of endpoint member payloads.
    final_rows: list[dict[str, object]] = []
    endpoint_parse_failures: list[dict[str, str]] = []
    with zipfile.ZipFile(paths["endpoint_archive"]) as archive:
        for item in inventory:
            try:
                payload = archive.read(str(item["endpoint_member"]))
                structure = Poscar.from_str(payload.decode("utf-8"), read_velocities=False).structure
            except Exception as exc:
                endpoint_parse_failures.append(
                    {
                        "endpoint_filename": str(item["endpoint_filename"]),
                        "failure": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            final_rows.append({**item, "structure": structure})

    mapping_rows: list[dict[str, object]] = []
    compositions = sorted({str(row["composition_key"]) for row in inventory})
    for composition in compositions:
        group_initials = [row for row in initial_rows if row["composition_key"] == composition]
        group_finals = [row for row in final_rows if row["composition_key"] == composition]
        mapping_rows.extend(match_composition_group(group_initials, group_finals))
    for failure in endpoint_parse_failures:
        item = next(row for row in inventory if row["endpoint_filename"] == failure["endpoint_filename"])
        mapping_rows.append(
            {
                "endpoint_member": item["endpoint_member"],
                "endpoint_filename": item["endpoint_filename"],
                "composition_key": item["composition_key"],
                "mapped": False,
                "mapping_failure": f"endpoint parse failed: {failure['failure']}",
                "initial_index": -1,
                "match_tier": -1,
                "normalized_rms": math.nan,
                "normalized_max": math.nan,
            }
        )
    mapping = pd.DataFrame(mapping_rows).sort_values("endpoint_filename", kind="mergesort")
    for index in mapping.index[mapping["mapped"]]:
        response = endpoint_response(
            normalized_rms=float(mapping.at[index, "normalized_rms"]),
            normalized_max=float(mapping.at[index, "normalized_max"]),
            initial_volume_per_atom=float(mapping.at[index, "initial_volume_per_atom"]),
            final_volume_per_atom=float(mapping.at[index, "final_volume_per_atom"]),
            match_tier=int(mapping.at[index, "match_tier"]),
        )
        for name, value in response.items():
            mapping.at[index, name] = value
    mapped = mapping.loc[mapping["mapped"]].merge(
        predictions, on="initial_index", how="left", validate="one_to_one", suffixes=("", "_frozen")
    )
    if mapped["mupr_risk"].isna().any():
        raise RuntimeError("NEXT542 mapped frozen prediction is missing")
    screen = evaluate_screen(mapped, bootstrap_draws=bootstrap_draws, seed=BOOTSTRAP_SEED)
    pauling = _pauling_comparisons(mapped)
    mapping_coverage = len(mapped) / n541.EXPECTED_ENDPOINT_FILES
    mupr_coverage = screen["rows"] / len(mapped) if len(mapped) else 0.0
    best = pauling["best"]
    gates = {
        "mapping_coverage": {
            "value": mapping_coverage,
            "threshold": MINIMUM_MAPPING_COVERAGE,
            "passes": mapping_coverage >= MINIMUM_MAPPING_COVERAGE,
        },
        "class_counts": {
            "severe": screen["positives"],
            "nonsevere": screen["negatives"],
            "minimum_each": MINIMUM_CLASS_COUNT,
            "passes": min(screen["positives"], screen["negatives"]) >= MINIMUM_CLASS_COUNT,
        },
        "mupr_coverage": {
            "value": mupr_coverage,
            "threshold": MINIMUM_MUPR_COVERAGE,
            "passes": mupr_coverage >= MINIMUM_MUPR_COVERAGE,
        },
        "roc_auc": {
            "value": screen["roc_auc"],
            "lower": screen["bootstrap"]["roc_auc"]["lower"],
            "point_threshold": MINIMUM_AUC,
            "lower_threshold_strict": MINIMUM_AUC_LOWER,
            "passes": screen["roc_auc"] >= MINIMUM_AUC
            and screen["bootstrap"]["roc_auc"]["lower"] > MINIMUM_AUC_LOWER,
        },
        "pauling_margin": {
            "best_pauling": pauling["best_pauling"],
            "value": best.get("mupr_margin") if isinstance(best, dict) else None,
            "threshold": MINIMUM_PAULING_MARGIN,
            "passes": isinstance(best, dict) and best["mupr_margin"] >= MINIMUM_PAULING_MARGIN,
        },
        "top_15_percent": {
            **screen["top_15_percent"],
            "precision_threshold": MINIMUM_TOP_PRECISION,
            "precision_lower_threshold": MINIMUM_TOP_PRECISION_LOWER,
            "recall_threshold": MINIMUM_TOP_RECALL,
            "passes": screen["top_15_percent"]["precision"] >= MINIMUM_TOP_PRECISION
            and screen["top_15_percent"]["precision_wilson_lower"]
            >= MINIMUM_TOP_PRECISION_LOWER
            and screen["top_15_percent"]["recall"] >= MINIMUM_TOP_RECALL,
        },
        "bottom_50_percent": {
            **screen["bottom_50_percent"],
            "nonsevere_threshold": MINIMUM_BOTTOM_NONSEVERE,
            "passes": screen["bottom_50_percent"]["nonsevere_fraction"]
            >= MINIMUM_BOTTOM_NONSEVERE,
        },
        "spearman": {
            "value": screen["spearman"],
            "lower": screen["bootstrap"]["spearman"]["lower"],
            "point_threshold": MINIMUM_SPEARMAN,
            "lower_threshold_strict": MINIMUM_SPEARMAN_LOWER,
            "passes": screen["spearman"] >= MINIMUM_SPEARMAN
            and screen["bootstrap"]["spearman"]["lower"] > MINIMUM_SPEARMAN_LOWER,
        },
    }
    all_gates_pass = all(value["passes"] for value in gates.values())
    result = {
        "protocol": PROTOCOL,
        "hypothesis": "frozen coefficient-free MUPR ranks severe DFT structural response",
        "mapping": {
            "endpoint_files": n541.EXPECTED_ENDPOINT_FILES,
            "mapped": len(mapped),
            "unmapped": int((~mapping["mapped"]).sum()),
            "coverage": mapping_coverage,
            "tier_counts": {
                str(key): int(value)
                for key, value in sorted(Counter(mapped["match_tier"].astype(int)).items())
            },
            "failure_counts": dict(
                sorted(Counter(mapping.loc[~mapping["mapped"], "mapping_failure"].astype(str)).items())
            ),
        },
        "screen": screen,
        "pauling_comparisons": pauling,
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "scientific_success": all_gates_pass,
        "claim_scope": "source-independent DFT structural-response proxy only; not energy stability",
        "retuning_after_endpoint_access_performed": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    try:
        mapping_path = staging / MAPPING_NAME
        result_path = staging / RESULT_NAME
        mapping.merge(
            predictions, on="initial_index", how="left", suffixes=("", "_frozen")
        ).to_parquet(mapping_path, index=False)
        result_path.write_bytes(_json_bytes(result))
        outputs = {
            mapping_path.name: _sha256(mapping_path),
            result_path.name: _sha256(result_path),
        }
        manifest = {
            "protocol": PROTOCOL,
            "inputs_sha256": {
                name: {"path": str(path), "sha256": hashes[name]}
                for name, path in paths.items()
            },
            "outputs_sha256": outputs,
            "executed_source_sha256": {
                "src/next542_jarvis_cdvae_dft_response.py": source_hash
            },
            "predictions_and_gates_frozen_before_endpoint_payload_access": True,
            "endpoint_member_payloads_read_offline": True,
            "endpoint_used_by_executable_screen": False,
            "dft_energy_force_or_stress_read": False,
            "relaxed_coordinates_used_only_as_offline_endpoint": True,
            "bootstrap_draws": bootstrap_draws,
            "all_gates_pass": all_gates_pass,
            "scientific_success": all_gates_pass,
            "report_authorized": all_gates_pass,
            "canonical_report_or_paper_edit_authorized": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT542 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT542 input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-archive", type=Path, required=True)
    parser.add_argument("--endpoint-archive", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--next541-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_dft_response_evaluation(
        initial_archive=args.initial_archive,
        endpoint_archive=args.endpoint_archive,
        design_path=args.design,
        next541_dir=args.next541_dir,
        output_dir=args.output_dir,
        bootstrap_draws=args.bootstrap_draws,
    )
    print(json.dumps({"mapping": result["mapping"], "gates": result["gates"], "all_gates_pass": result["all_gates_pass"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
