#!/usr/bin/env python3
"""Package the first-principles structures as a Supplementary Data set.

The property-conditioned inverse-design run produced 1,081 candidates; 261 were selected
for the pre-registered E4 experiment and 260 of them relaxed.  This writes those relaxed
cells out as CIFs beside an index that carries, for every candidate, what the generator
produced, what the screen said about it before any calculation, and what DFT found.

The comparison the index makes possible is the point of shipping it.  Every verdict in the
manuscript was computed on the generator's own coordinates; relaxing those coordinates with
DFT changes the symmetry, so the index reports the distinct-site fraction and space group
both before and after and lets a reader check the law against either.

    python src/dft_supplementary_data.py
"""
from __future__ import annotations

import json
import pathlib
import zipfile

import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = pathlib.Path(__file__).resolve().parent.parent
DFT = ROOT / "dft"
OUT = ROOT / "outputs" / "20260828_dft_supplementary_structures"
INVERSE = ROOT / "outputs/20260822_property_design_synthesis_score/inverse_scores.parquet"
ARCHIVE = ROOT / ("outputs/20260821_property_design/"
                  "mattergen_bulk400_progress_13shards_adaptive/generated_crystals_cif.zip")

SYMPREC = 0.01          # the tolerance every Law 7 verdict in the manuscript uses
LAW7_BOUND = 2.0 / 3.0


def symmetry(structure: Structure) -> tuple[str, int, int, float]:
    """Space group and distinct-site fraction at the manuscript's tolerance."""
    try:
        sga = SpacegroupAnalyzer(structure, symprec=SYMPREC)
        dataset = sga.get_symmetry_dataset()
        distinct = len(set(dataset.equivalent_atoms))
        return (sga.get_space_group_symbol(), sga.get_space_group_number(), distinct,
                distinct / len(structure))
    except Exception:
        # a cell spglib cannot reduce is P1 by definition of the failure
        return "P1", 1, len(structure), 1.0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cif_dir = OUT / "structures"
    cif_dir.mkdir(exist_ok=True)

    records = json.loads((DFT / "E4_design" / "collected.json").read_text())["records"]
    moduli = {r["candidate_id"]: r for r in
              json.loads((DFT / "E4_design" / "bulk_moduli.json").read_text())
              if r.get("candidate_id")}
    inverse = pd.read_parquet(INVERSE).set_index("candidate_id")

    with zipfile.ZipFile(ARCHIVE) as archive:
        rows = []
        for record in records:
            if record["status"] not in ("complete", "unconverged"):
                continue
            contcar = record["stage_results"].get("relax_cell", {}).get("contcar")
            if not contcar:
                continue
            cid = record["candidate_id"]
            relaxed = Structure.from_str(contcar, fmt="poscar")
            sg_r, num_r, sites_r, frac_r = symmetry(relaxed)

            meta = inverse.loc[cid]
            member = str(meta["archive_member"])
            generated = Structure.from_str(
                archive.read(member).decode("utf-8"), fmt="cif")
            sg_g, num_g, sites_g, frac_g = symmetry(generated)

            relaxed.to(filename=str(cif_dir / f"{cid}.cif"), fmt="cif")
            bulk = moduli.get(cid, {})
            rows.append({
                "candidate_id": cid,
                "formula": relaxed.composition.reduced_formula,
                "elements": " ".join(sorted({s.symbol for s in relaxed.species})),
                "n_atoms": len(relaxed),
                "volume_per_atom_a3": relaxed.volume / len(relaxed),
                "role_in_E4": record.get("role"),
                "pss": float(meta["synthesis_score"]),
                "screened_by_pss": bool(record.get("role") == "screened"),
                "spacegroup_generated": sg_g,
                "spacegroup_number_generated": num_g,
                "site_fraction_generated": frac_g,
                "law7_generated": bool(frac_g <= LAW7_BOUND),
                "spacegroup_dft_relaxed": sg_r,
                "spacegroup_number_dft_relaxed": num_r,
                "site_fraction_dft_relaxed": frac_r,
                "law7_dft_relaxed": bool(frac_r <= LAW7_BOUND),
                "proxy_bulk_modulus_gpa": bulk.get("uma_bulk_modulus_gpa"),
                "dft_bulk_modulus_gpa": bulk.get("dft_bulk_modulus_gpa"),
                "dft_bulk_modulus_pressure_derivative": bulk.get("bp"),
                "dft_equilibrium_volume_a3": bulk.get("v0_a3"),
                "birch_murnaghan_rms_ev": bulk.get("fit_rms_ev"),
                "relaxation_converged": record["status"] == "complete",
            })

    index = pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)
    index.to_csv(OUT / "index.csv", index=False)

    with zipfile.ZipFile(OUT / "dft_relaxed_structures.zip", "w",
                         compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(OUT / "index.csv", "index.csv")
        for cif in sorted(cif_dir.glob("*.cif")):
            bundle.write(cif, f"structures/{cif.name}")

    summary = {
        "n_structures": int(len(index)),
        "law7_satisfied_generated": int(index["law7_generated"].sum()),
        "law7_satisfied_dft_relaxed": int(index["law7_dft_relaxed"].sum()),
        "gained_law7_on_relaxation": int((~index["law7_generated"]
                                          & index["law7_dft_relaxed"]).sum()),
        "lost_law7_on_relaxation": int((index["law7_generated"]
                                        & ~index["law7_dft_relaxed"]).sum()),
        "median_site_fraction_generated": float(index["site_fraction_generated"].median()),
        "median_site_fraction_dft_relaxed": float(
            index["site_fraction_dft_relaxed"].median()),
        "law7_after_relaxation_by_role": {
            role: round(float(group["law7_dft_relaxed"].mean()), 4)
            for role, group in index.groupby("role_in_E4")
        },
        "elements": sorted({e for row in index["elements"] for e in row.split()}),
        "spacegroups_dft_relaxed": index["spacegroup_dft_relaxed"]
        .value_counts().to_dict(),
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
