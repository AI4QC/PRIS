#!/usr/bin/env python3
"""Referee fix: apply the FROZEN rule ladder to raw generator outputs (x0).

Population: all 256 SSAGen CIVAE-Transformer x0 frames from
$PRIS_ARCHIVE/next24_ssagen_x0/geometry_only_frames.zip
(unrelaxed geometry-only structures; every generated attempt retained,
per its MANIFEST: all_generated_attempts_retained=true).

Rule ladder: validity_rulesets.sets_of() with the identical feature code path
(apply_rules.features -> phys/elec/geom feats, guess_oxi integer valences,
aug(): spglib symprec 0.01 wyckoff economy + elec_feat bv_rel_mean),
thresholds are the published frozen values. Missing individual features count
as satisfying (the paper's evaluation convention) *only for structures with a
charge assignment*; structures where guess_oxi fails have no ladder verdict at
all and are reported as unfeaturisable (NOT counted as satisfying).

Baselines: min pair distance > 0.5 / 0.7 / 1.0 A (computed for all frames,
no charge assignment needed).
"""
import json, os, sys, zipfile, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "<repo>/src")

OUT = "<repo>/outputs/20260814_referee_fixes"
FRAMES = ("/tmp/claude-1000/-home-zhilong-workspace-newpauling/"
          "c65c6ce1-73ce-4feb-815b-553538256d7e/scratchpad/gen_frames")
DATA = "$PRIS_ARCHIVE/"

from validity_rulesets import sets_of, min_pair_dist
from discriminate import guess_oxi
import apply_rules as AR
from elec_feat import elec_feats
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read as ase_read


def aug(st_, val_):
    out_ = {}
    try:
        ds = SpacegroupAnalyzer(st_, symprec=0.01).get_symmetry_dataset()
        out_["wyckoff_econ"] = len(set(ds.equivalent_atoms)) / len(st_)
    except Exception:
        pass
    try:
        e = elec_feats(st_, val_)
        if e and "bv_rel_mean" in e:
            out_["bv_rel_mean"] = e["bv_rel_mean"]
    except Exception:
        pass
    return out_


def one(path, sid):
    atoms = ase_read(path)
    st = AseAtomsAdaptor.get_structure(atoms)
    row = {"material_id": sid, "natoms": len(st),
           "formula": st.composition.reduced_formula,
           "min_pair_dist": min_pair_dist(st)}
    val, ok = guess_oxi(st)
    row["guess_oxi_ok"] = bool(ok)
    fval = None
    if not ok:
        fval = AR.frac_oxi(st)
        row["frac_oxi_ok"] = fval is not None
    else:
        row["frac_oxi_ok"] = True
    row["featurisable"] = bool(ok)          # validity_rulesets convention
    row["featurisable_with_frac"] = bool(ok or fval is not None)
    use_val = val if ok else fval
    if use_val is None:
        return row
    f, err = AR.features(st)                # uses guess_oxi, frac fallback inside
    if f is None:
        row["featurisable"] = False
        row["featurisable_with_frac"] = False
        row["feature_error"] = err
        return row
    f.update(aug(st, use_val))
    for k in ("bl_min", "bl_mean", "cn_an_mean", "madz_range", "mad_max",
              "frac_like_bonds", "fi", "wyckoff_econ", "bv_rel_mean"):
        row[k] = f.get(k, np.nan)
    verdict = sets_of(f)
    for k, v in verdict.items():
        row[k] = bool(v)
    return row


def main():
    meta = pd.read_parquet(f"{DATA}/next24_ssagen_x0/holdout_metadata.parquet")
    sids = sorted(meta.material_id)         # deterministic order, full cohort
    rows = []
    for i, sid in enumerate(sids):
        p = os.path.join(FRAMES, sid + ".extxyz")
        try:
            rows.append(one(p, sid))
        except Exception as exc:
            rows.append({"material_id": sid, "featurisable": False,
                         "featurisable_with_frac": False,
                         "feature_error": f"{type(exc).__name__}: {exc}"})
        if (i + 1) % 32 == 0:
            print(f"{i+1}/{len(sids)}", flush=True)
    d = pd.DataFrame(rows)

    # merge label-free comparators (NOT ground-truth failure labels)
    pred = pd.read_parquet(
        f"{DATA}/next24_ssagen_predictions/next24_generated_transport_predictions.parquet"
    )[["material_id", "reject", "next23_risk_score"]].rename(
        columns={"reject": "next23_transport_reject"})
    pau = pd.read_parquet(
        f"{DATA}/next24_ssagen_pauling/next24_pauling_controls.parquet"
    )[["material_id", "pauling_feature_error", "pauling_p2_p5_decision"]]
    d = d.merge(pred, on="material_id", how="left").merge(pau, on="material_id", how="left")
    d.to_csv(f"{OUT}/generator_transfer.csv", index=False)

    n = len(d)
    feat = d.featurisable.fillna(False).astype(bool)
    sets_cols = ["L1", "D1_804", "L1'", "L2", "L3", "L4"]
    summ = {
        "protocol": "2026-08-14-referee-generator-transfer-v1",
        "population": ("all 256 SSAGen CIVAE-Transformer-500 unrelaxed x0 frames "
                       "(next24_ssagen_x0, all generated attempts retained); "
                       "no sampling needed - full cohort evaluated"),
        "n_structures": int(n),
        "unfeaturisable": {
            "definition": ("guess_oxi (integer, composition-based) charge assignment "
                           "failed - same gate as validity_rulesets parents; these "
                           "have no ladder verdict and are NOT counted as satisfying"),
            "n_guess_oxi_failed": int((~feat).sum()),
            "fraction_guess_oxi_failed": float((~feat).mean()),
            "n_also_failing_fractional_fallback": int((~d.featurisable_with_frac.fillna(False).astype(bool)).sum()),
            "fraction_also_failing_fractional_fallback": float((~d.featurisable_with_frac.fillna(False).astype(bool)).mean()),
        },
        "acceptance": {}, "acceptance_among_featurisable": {},
        "acceptance_with_frac_fallback": {},
    }
    for th in (0.5, 0.7, 1.0):
        m = (d.min_pair_dist > th)
        summ["acceptance"][f"min_pair_dist>{th}A"] = {
            "n_pass": int(m.sum()), "fraction_of_all": float(m.mean())}
    for c in sets_cols:
        col = d[c].fillna(False).astype(bool) & feat
        summ["acceptance"][c] = {"n_pass": int(col.sum()),
                                 "fraction_of_all": float(col.mean())}
        summ["acceptance_among_featurisable"][c] = {
            "n_featurisable": int(feat.sum()),
            "fraction": float(d.loc[feat, c].astype(bool).mean()) if feat.any() else None}
        featf = d.featurisable_with_frac.fillna(False).astype(bool)
        colf = d[c].fillna(False).astype(bool) & featf
        summ["acceptance_with_frac_fallback"][c] = {
            "note": ("apply_rules public convention: fractional mean-valence "
                     "fallback allowed when integer guess_oxi fails"),
            "n_featurisable": int(featf.sum()),
            "n_pass": int(colf.sum()),
            "fraction_of_all": float(colf.mean()),
            "fraction_among_featurisable": float(d.loc[featf, c].astype(bool).mean()) if featf.any() else None}

    # cross-tab vs label-free comparators
    summ["label_availability"] = {
        "dft_or_ground_truth_failure_label": ("NONE: VASP queue for this cohort was "
            "never executed (next12_vasp_pbe_queue MANIFEST: vasp_execution_completed=false; "
            "next24 manifests: labels_opened=false). No per-structure DFT convergence/"
            "failure label exists, so 'acceptance among labelled-failed vs labelled-ok' "
            "cannot be computed against ground truth."),
        "generator_generation_status": "all 256 attempts 'generated', error field empty",
        "available_label_free_comparators": [
            "next23_transport_reject: frozen transported relaxation-risk rule decision (a prediction, not a label)",
            "pauling_p2_p5_decision: Pauling rules 2-5 joint KEEP/REJECT/ABSTAIN (a control, not a label)"],
    }
    ct = {}
    for name, col in (("next23_transport_reject", d.next23_transport_reject),
                      ("pauling_p2_p5", d.pauling_p2_p5_decision)):
        sub = {}
        for g, gd in d.groupby(col.astype(str)):
            gfeat = gd.featurisable.fillna(False).astype(bool)
            sub[g] = {"n": int(len(gd)),
                      "L4_pass_fraction_of_group": float((gd["L4"].fillna(False).astype(bool) & gfeat).mean()),
                      "L1_pass_fraction_of_group": float((gd["L1"].fillna(False).astype(bool) & gfeat).mean()),
                      "unfeaturisable_fraction": float((~gfeat).mean())}
        ct[name] = sub
    summ["acceptance_by_label_free_comparator"] = ct

    with open(f"{OUT}/generator_transfer_summary.json", "w") as fh:
        json.dump(summ, fh, indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
