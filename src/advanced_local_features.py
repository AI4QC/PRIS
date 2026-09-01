#!/usr/bin/env python3
"""Additive local descriptors for the next PRIS search.

The first implemented family is the frozen-parameter bond-valence local
triplet: valence mismatch, entropy effective coordination, and
bond-strength-weighted vector asymmetry.
"""

from __future__ import annotations
import os

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def composition_guard_features(
    structure,
    formal_valences: Sequence[float],
) -> dict[str, float]:
    """Compute the same composition-only dchi/fi guard without fitted data."""

    charges = np.asarray(formal_valences, dtype=float)
    if len(structure) != len(charges):
        raise ValueError("structure and formal_valences must have equal length")
    negative_elements = {
        structure[index].specie.symbol
        for index in range(len(structure))
        if charges[index] < 0
    }
    if len(negative_elements) != 1:
        return {}
    anion = next(iter(negative_elements))
    from pymatgen.core.periodic_table import Element

    anion_x = Element(anion).X
    if anion_x is None or not np.isfinite(anion_x):
        return {}
    composition = structure.composition.get_el_amt_dict()
    cations = {}
    for element, amount in composition.items():
        if element == anion:
            continue
        value = Element(element).X
        if value is not None and np.isfinite(value):
            cations[element] = (float(amount), float(value))
    if not cations:
        return {}
    total = sum(amount for amount, _ in cations.values())
    deltas = [float(anion_x - value) for _, value in cations.values()]
    dchi = sum(
        amount * (float(anion_x) - value)
        for amount, value in cations.values()
    ) / total
    return {
        "dchi": float(dchi),
        "fi": float(1 - np.exp(-0.25 * dchi * dchi)),
        "dchi_min": float(min(deltas)),
    }


def resolve_bond_valence_parameter(
    key: tuple[str, int, str, int],
    parameters: Mapping[tuple[str, int, str, int], tuple[float, float]],
    *,
    policy: str,
) -> tuple[float, float, str] | None:
    """Resolve a frozen parameter and disclose which non-fitted fallback won."""

    if policy not in {"exact", "frozen-fallback"}:
        raise ValueError(f"unsupported parameter policy: {policy}")
    exact = parameters.get(key)
    if exact is not None:
        return float(exact[0]), float(exact[1]), "exact"
    if policy == "exact":
        return None
    element1, valence1, element2, valence2 = key
    candidates = [
        (
            abs(table_key[1] - valence1) + abs(table_key[3] - valence2),
            table_key,
            value,
        )
        for table_key, value in parameters.items()
        if table_key[0] == element1 and table_key[2] == element2
    ]
    if candidates:
        _, _, value = min(candidates, key=lambda item: (item[0], item[1]))
        return float(value[0]), float(value[1]), "nearest_valence"
    try:
        from pymatgen.analysis.bond_valence import BV_PARAMS
        from pymatgen.core import Element

        first, second = Element(element1), Element(element2)
        r1, c1 = BV_PARAMS[first]["r"], BV_PARAMS[first]["c"]
        r2, c2 = BV_PARAMS[second]["r"], BV_PARAMS[second]["c"]
        r0 = r1 + r2 - (
            r1
            * r2
            * (math.sqrt(c1) - math.sqrt(c2)) ** 2
            / (c1 * r1 + c2 * r2)
        )
        return float(r0), 0.37, "brown_generic"
    except Exception:
        return None


def site_bond_valence_statistics(
    strengths: np.ndarray,
    unit_vectors: np.ndarray,
    formal_valence: float,
) -> dict[str, float]:
    """Compute literature-defined bond-valence statistics for one site."""

    weights = np.asarray(strengths, dtype=float)
    vectors = np.asarray(unit_vectors, dtype=float)
    if weights.ndim != 1 or not len(weights):
        raise ValueError("strengths must be a non-empty one-dimensional array")
    if vectors.shape != (len(weights), 3):
        raise ValueError("unit_vectors must have shape (n_bonds, 3)")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("bond-valence strengths must be finite and positive")
    if not np.isfinite(formal_valence) or abs(formal_valence) <= 0:
        raise ValueError("formal_valence must be finite and non-zero")

    norms = np.linalg.norm(vectors, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("bond direction vectors must be finite and non-zero")
    directions = vectors / norms[:, None]
    total = float(weights.sum())
    probabilities = weights / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    relative = (total - abs(float(formal_valence))) / abs(float(formal_valence))
    asymmetry = float(np.linalg.norm((weights[:, None] * directions).sum(axis=0)))
    asymmetry /= total
    return {
        "relative_mismatch": float(relative),
        "absolute_mismatch": float(abs(relative)),
        "effective_cn": float(np.exp(entropy)),
        "vector_asymmetry": float(asymmetry),
    }


def _summarize(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def aggregate_bond_valence_sites(
    site_stats: Sequence[Mapping[str, float] | None],
    formal_valences: Sequence[float],
) -> dict[str, float]:
    """Aggregate site statistics separately for cations and anions."""

    if len(site_stats) != len(formal_valences):
        raise ValueError("site_stats and formal_valences must have equal length")
    charges = np.asarray(formal_valences, dtype=float)
    charged = np.isfinite(charges) & (charges != 0)
    covered = np.asarray([stats is not None for stats in site_stats], dtype=bool)
    denominator = int(charged.sum())
    out = {
        "bvloc_site_coverage": (
            float((covered & charged).sum() / denominator) if denominator else 0.0
        )
    }

    for short, mask in (("cat", charges > 0), ("an", charges < 0)):
        count = int(mask.sum())
        indices = [
            index
            for index in np.flatnonzero(mask)
            if site_stats[int(index)] is not None
        ]
        out[f"bvloc_{short}_coverage"] = float(len(indices) / count) if count else 0.0
        if not indices:
            continue
        for metric in (
            "relative_mismatch",
            "absolute_mismatch",
            "effective_cn",
            "vector_asymmetry",
        ):
            values = []
            for index in indices:
                stats = site_stats[index]
                assert stats is not None
                if metric in stats:
                    values.append(float(stats[metric]))
                elif metric == "absolute_mismatch":
                    values.append(abs(float(stats["relative_mismatch"])))
            if not values:
                continue
            for aggregate, value in _summarize(values).items():
                out[f"bvloc_{short}_{metric}_{aggregate}"] = value
    return out


def bond_valence_local_features(
    structure,
    formal_valences: Sequence[float],
    *,
    neighbors: Sequence[Sequence[Mapping[str, object]]] | None = None,
    parameters: Mapping[tuple[str, int, str, int], tuple[float, float]] | None = None,
    parameter_policy: str = "exact",
) -> dict[str, float]:
    """Compute local bond-valence descriptors for a periodic structure.

    ``neighbors`` and ``parameters`` are injectable so the mathematical
    contract can be tested without learning anything from the target data.
    Production calls use CrystalNN and the repository's frozen IUCr table.
    """

    charges = np.asarray(formal_valences, dtype=float)
    if len(structure) != len(charges):
        raise ValueError("structure and formal_valences must have equal length")
    if neighbors is None:
        from pymatgen.analysis.local_env import CrystalNN

        finder = CrystalNN(weighted_cn=False, x_diff_weight=0.0)
        built = []
        for index in range(len(structure)):
            try:
                built.append(finder.get_nn_info(structure, index))
            except Exception:
                built.append([])
        neighbors = built
    if len(neighbors) != len(structure):
        raise ValueError("neighbors must provide one list per structure site")
    if parameters is None:
        from elec_feat import bv_table

        parameters = bv_table()

    strengths: list[list[float]] = [[] for _ in structure]
    directions: list[list[np.ndarray]] = [[] for _ in structure]
    hit = 0
    miss = 0
    source_counts = {"exact": 0, "nearest_valence": 0, "brown_generic": 0}
    for center in range(len(structure)):
        if charges[center] <= 0:
            continue
        for neighbor in neighbors[center]:
            other = int(neighbor["site_index"])
            if other == center or other < 0 or other >= len(structure):
                continue
            if charges[other] >= 0:
                continue
            image_value = neighbor.get("image")
            image = (
                np.zeros(3, dtype=float)
                if image_value is None
                else np.asarray(image_value, dtype=float)
            )
            displacement = structure.lattice.get_cartesian_coords(
                structure[other].frac_coords + image - structure[center].frac_coords
            )
            distance = float(np.linalg.norm(displacement))
            if not np.isfinite(distance) or distance <= 0:
                continue
            key = (
                structure[center].specie.symbol,
                int(round(charges[center])),
                structure[other].specie.symbol,
                int(round(charges[other])),
            )
            resolved = resolve_bond_valence_parameter(
                key,
                parameters,
                policy=parameter_policy,
            )
            if resolved is None:
                miss += 1
                continue
            r0, decay, source = resolved
            if not np.isfinite(r0) or not np.isfinite(decay) or decay <= 0:
                miss += 1
                continue
            strength = float(np.exp((r0 - distance) / decay))
            if not np.isfinite(strength) or strength <= 0:
                miss += 1
                continue
            direction = displacement / distance
            strengths[center].append(strength)
            directions[center].append(direction)
            strengths[other].append(strength)
            directions[other].append(-direction)
            hit += 1
            source_counts[source] += 1

    site_stats: list[dict[str, float] | None] = []
    for index in range(len(structure)):
        if strengths[index]:
            site_stats.append(
                site_bond_valence_statistics(
                    np.asarray(strengths[index]),
                    np.asarray(directions[index]),
                    charges[index],
                )
            )
        else:
            site_stats.append(None)
    out = aggregate_bond_valence_sites(site_stats, charges)
    out["bvloc_bond_parameter_coverage"] = float(hit / (hit + miss)) if hit + miss else 0.0
    out["bvloc_bond_count"] = float(hit)
    out["bvloc_parameter_missing_count"] = float(miss)
    for source, count_value in source_counts.items():
        out[f"bvloc_parameter_{source}_fraction"] = (
            float(count_value / hit) if hit else 0.0
        )
    return out


def load_real_records(
    features_dir: str | Path,
    *,
    allowed_splits: Sequence[str] = ("discovery", "calibration"),
    max_sites: int = 80,
    limit: int = 0,
) -> list[dict[str, object]]:
    """Resolve permitted records; the monolithic index scan includes split metadata."""

    root = Path(features_dir)
    real = pd.read_parquet(
        root / "real_all.parquet",
        columns=["source_id", "split"],
    )
    selected = real[real["split"].isin(tuple(allowed_splits))].copy()
    observed = set(selected["split"].dropna().astype(str).unique())
    if not observed.issubset(set(allowed_splits)):
        raise ValueError(f"forbidden splits selected: {sorted(observed)}")
    provenance = pd.read_parquet(
        root / "provenance.parquet",
        columns=[
            "source_id",
            "blob_offset",
            "blob_length",
            "n_elements",
            "n_sites",
        ],
    )
    selected = selected.merge(
        provenance,
        on="source_id",
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    selected = selected[
        (selected["n_elements"] >= 2) & (selected["n_sites"] <= max_sites)
    ]
    if limit:
        selected = selected.iloc[:limit]
    return [
        {
            "sid": row.source_id,
            "off": int(row.blob_offset),
            "ln": int(row.blob_length),
            "split": str(row.split),
        }
        for row in selected.itertuples(index=False)
    ]


def load_bad_records(
    features_dir: str | Path,
    *,
    allowed_splits: Sequence[str] = ("discovery", "calibration"),
    max_sites: int = 80,
    limit: int = 0,
) -> list[dict[str, object]]:
    """Resolve exactly the perturbation IDs present in the existing baseline."""

    root = Path(features_dir)
    bad = pd.read_parquet(
        root / "phys_bad.parquet",
        columns=["sid", "kind", "parent"],
    )
    splits = pd.read_parquet(
        root / "splits.parquet",
        columns=["source_id", "split"],
    ).rename(columns={"source_id": "parent"})
    bad = bad.merge(splits, on="parent", how="left", validate="many_to_one")
    bad = bad[bad["split"].isin(tuple(allowed_splits))].copy()
    parents = (
        bad.groupby(["parent", "split"], sort=False)["kind"]
        .agg(lambda values: tuple(sorted(set(values))))
        .reset_index()
    )
    provenance = pd.read_parquet(
        root / "provenance.parquet",
        columns=[
            "source_id",
            "blob_offset",
            "blob_length",
            "n_elements",
            "n_sites",
        ],
    ).rename(columns={"source_id": "parent"})
    parents = parents.merge(
        provenance,
        on="parent",
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    parents = parents[
        (parents["n_elements"] >= 2) & (parents["n_sites"] <= max_sites)
    ]
    if limit:
        parents = parents.iloc[:limit]
    return [
        {
            "sid": row.parent,
            "off": int(row.blob_offset),
            "ln": int(row.blob_length),
            "split": str(row.split),
            "kinds": tuple(row.kind),
        }
        for row in parents.itertuples(index=False)
    ]


def load_false_positive_records(
    features_dir: str | Path,
    materials_database: str | Path,
    *,
    limit: int = 0,
) -> list[dict[str, object]]:
    """Resolve the already-frozen DFT-relaxed false-positive audit IDs."""

    root = Path(features_dir)
    audit_ids = (
        pd.read_parquet(root / "false_positive.parquet", columns=["sid"])["sid"]
        .dropna()
        .astype(str)
        .tolist()
    )
    if limit:
        audit_ids = audit_ids[:limit]
    resolved: dict[str, tuple[int, int]] = {}
    database = Path(materials_database)
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        for start in range(0, len(audit_ids), 500):
            batch = audit_ids[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            query = (
                "SELECT material_id, blob_offset, blob_length FROM materials "
                f"WHERE material_id IN ({placeholders})"
            )
            for material_id, offset, length in connection.execute(query, batch):
                resolved[str(material_id)] = (int(offset), int(length))
    finally:
        connection.close()
    return [
        {
            "sid": sid,
            "off": resolved[sid][0],
            "ln": resolved[sid][1],
            "split": "false_positive_audit",
        }
        for sid in audit_ids
        if sid in resolved
    ]


def _real_worker(record: Mapping[str, object]) -> dict[str, object] | None:
    from pymatgen.core import Structure

    from discriminate import guess_oxi, read_blob_cif

    try:
        structure = Structure.from_str(
            read_blob_cif(int(record["off"]), int(record["ln"])),
            fmt="cif",
        )
        valences, ok = guess_oxi(structure)
        if not ok:
            return None
        out: dict[str, object] = bond_valence_local_features(
            structure,
            valences,
            parameter_policy=str(record.get("parameter_policy", "exact")),
        )
        out.update(source_id=record["sid"], split=record["split"])
        return out
    except Exception:
        return None


def _bad_worker(record: Mapping[str, object]) -> list[dict[str, object]]:
    from pymatgen.core import Structure

    from discriminate import guess_oxi, read_blob_cif
    from make_negatives import perturb, swapped_val
    from phys_law import seed_of

    rows: list[dict[str, object]] = []
    try:
        structure = Structure.from_str(
            read_blob_cif(int(record["off"]), int(record["ln"])),
            fmt="cif",
        )
        valences, ok = guess_oxi(structure)
        if not ok:
            return rows
        rng = np.random.default_rng(seed_of(str(record["sid"])))
        wanted = set(record["kinds"])
        # Always advance the generator through the canonical sequence, even
        # when a baseline row is absent for one kind.
        for kind in ("S1", "S2", "S3", "S4", "S5"):
            changed = perturb(structure, kind, rng, valences)
            if changed is None or kind not in wanted:
                continue
            out: dict[str, object] = bond_valence_local_features(
                changed,
                swapped_val(changed, valences),
                parameter_policy=str(record.get("parameter_policy", "exact")),
            )
            out.update(
                sid=f"{record['sid']}_{kind}",
                kind=kind,
                parent=record["sid"],
                split=record["split"],
            )
            rows.append(out)
    except Exception:
        return rows
    return rows


def _false_positive_worker(
    record: Mapping[str, object],
) -> dict[str, object] | None:
    from pymatgen.core import Structure

    from discriminate import read_blob_cif
    from polymorph_rank2 import balance

    try:
        structure = Structure.from_str(
            read_blob_cif(int(record["off"]), int(record["ln"])),
            fmt="cif",
        )
        if len(structure) > 80:
            return None
        valence_map = balance(
            structure.composition.reduced_formula.replace(" ", "")
        )
        if valence_map is None:
            return None
        valences = [
            float(valence_map[site.specie.symbol]) for site in structure
        ]
        out: dict[str, object] = bond_valence_local_features(
            structure,
            valences,
            parameter_policy=str(record.get("parameter_policy", "exact")),
        )
        out.update(composition_guard_features(structure, valences))
        out.update(sid=record["sid"], split=record["split"])
        return out
    except Exception:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute additive frozen bond-valence local descriptors."
    )
    parser.add_argument("mode", choices=("real", "bad", "false-positive"))
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-sites", type=int, default=80)
    parser.add_argument(
        "--materials-database",
        type=Path,
        default=Path(
            Path(os.environ.get("PRIS_MATDATA_SQLITE", "materials.sqlite"))
        ),
    )
    parser.add_argument(
        "--parameter-policy",
        choices=("exact", "frozen-fallback"),
        default="exact",
    )
    parser.add_argument(
        "--allowed-splits",
        nargs="+",
        default=["discovery", "calibration"],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    permitted = {"discovery", "calibration"}
    if (
        args.mode != "false-positive"
        and not set(args.allowed_splits).issubset(permitted)
    ):
        raise SystemExit("only discovery and calibration splits are permitted")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")

    if args.mode == "false-positive":
        records = load_false_positive_records(
            args.features_dir,
            args.materials_database,
            limit=args.limit,
        )
    else:
        loader = load_real_records if args.mode == "real" else load_bad_records
        records = loader(
            args.features_dir,
            allowed_splits=args.allowed_splits,
            max_sites=args.max_sites,
            limit=args.limit,
        )
    for record in records:
        record["parameter_policy"] = args.parameter_policy
    print(f"{args.mode}: resolved {len(records):,} parent records", flush=True)

    worker = {
        "real": _real_worker,
        "bad": _bad_worker,
        "false-positive": _false_positive_worker,
    }[args.mode]
    single_result = args.mode in {"real", "false-positive"}
    rows: list[dict[str, object]] = []
    if args.workers == 1:
        results = map(worker, records)
        for index, result in enumerate(results, start=1):
            if single_result:
                if result is not None:
                    rows.append(result)
            else:
                rows.extend(result)
            if index % 1000 == 0:
                print(f"  {index:,}/{len(records):,} -> {len(rows):,}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = executor.map(worker, records, chunksize=args.chunksize)
            for index, result in enumerate(results, start=1):
                if single_result:
                    if result is not None:
                        rows.append(result)
                else:
                    rows.extend(result)
                if index % 1000 == 0:
                    print(
                        f"  {index:,}/{len(records):,} -> {len(rows):,}",
                        flush=True,
                    )
    if not rows:
        raise SystemExit("no descriptor rows were produced")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(args.out, index=False)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [str(value) for value in (argv if argv is not None else sys.argv[1:])],
        "mode": args.mode,
        "allowed_splits": (
            ["false_positive_audit"]
            if args.mode == "false-positive"
            else list(args.allowed_splits)
        ),
        "n_input_parents": len(records),
        "n_output_rows": len(frame),
        "split_counts": {
            str(key): int(value)
            for key, value in frame["split"].value_counts().sort_index().items()
        },
        "feature_columns": sorted(
            column for column in frame if column.startswith("bvloc_")
        ),
        "input_hashes": {
            name: _sha256(args.features_dir / name)
            for name in (
                ("real_all.parquet", "provenance.parquet")
                if args.mode == "real"
                else (
                    ("false_positive.parquet",)
                    if args.mode == "false-positive"
                    else (
                        "phys_bad.parquet",
                        "splits.parquet",
                        "provenance.parquet",
                    )
                )
            )
        },
        "implementation_sha256": _sha256(Path(__file__)),
        "lockbox_access": args.mode != "false-positive",
        "lockbox_payload_used": False,
        "lockbox_rows_in_output": False,
        "source_access_note": (
            "monolithic parquet metadata and whole-file hashes materialize/read "
            "all splits before permitted IDs are selected"
            if args.mode != "false-positive"
            else "false-positive audit inputs contain no experimental lockbox rows"
        ),
        "parameter_policy": args.parameter_policy,
        "fallback_policy": (
            "nearest frozen element-pair valence tuple, then pymatgen Brown generic"
            if args.parameter_policy == "frozen-fallback"
            else "none; missing exact tuples abstain"
        ),
        "neighbor_policy": "CrystalNN(weighted_cn=False, x_diff_weight=0.0)",
    }
    metadata_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {args.out} ({len(frame):,} rows)", flush=True)
    print(f"wrote {metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
