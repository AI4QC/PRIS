# NEXT30 design: a law for pre-screening high hull energy in inorganic crystals

Date: 2026-08-03  
Status: design frozen before implementation; no existing script, report or paper is modified

## Objective and boundaries

NEXT30 answers one question, closer to the overall goal than NEXT23's: can WBM candidates whose
DFT-relaxed hull energy is clearly high be safely screened out from an unrelaxed inorganic
crystal `x0` and a frozen element table alone? While the law executes it may not read or call
DFT, formation energies, the hull, relaxed structures, trajectories, energy/force/stress
surrogates, MatterSim/MLIP, geometry optimisation, or competing phases of the same composition.
WBM's DFT hull values serve as labels only during offline development and confirmation.

Three directions are workable: (A) a sparse equally weighted sum of a few physically directed
analytic terms; (B) fitting monotone sparse regression weights against the hull labels; (C)
routing to NEXT23/NEXT28 by periodic-graph connectivity. B might score higher in development, but
it looks more like an empirical model than a new law and carries the highest overfitting risk on
a single exposed source; C handles crystal type but does not address the energetic-stability
endpoint. A is therefore chosen.

## Data isolation and evidence level

Use the 8,192 WBM `x0` structures NEXT23 already sealed, together with their SIVR,
normalized-Madelung and SCBVE analytic features. Without reading any label value, order them by

```text
sha256("NEXT30-WBM-HULL-v1|" + material_id)
```

taking the first 4,096 as the development set and the last 4,096 as the confirmation set, and
seal the IDs, input hashes and the split manifest. The formula, normalisation constants and
thresholds may be chosen using the development labels only; afterwards the predictions on the
confirmation set are sealed first and only then are the confirmation labels read.

Because the same WBM label file was opened long ago by an earlier pipeline in this repository,
this can only be called "a procedural confirmation within a historical source", not a fresh
lockbox and not an external blind test. The confirmation set may take no part in candidate
ranking, threshold sweeps, or refitting after a failure.

## Candidate formulas

The base terms may only be closed-form scalars whose risk direction is physically determined in
advance: for SIVR, high edge mismatch, site imbalance and cell anisotropy are risk; for the
normalized Madelung terms, high weak-binding and site dispersion are risk; for SCBVE, high bond-
valence mismatch, bond-valence vector asymmetry and isolated sites are risk, and a low effective
coordination number is risk. Each term is robust-z scored using the development median and IQR;
no continuous weight is fitted.

The candidate catalogue is frozen as single terms, and as equally weighted sums of two or three
terms with a clear mechanistic connection, to a maximum of three. Each formula sweeps only the
fixed rejection fractions of the development score
`{0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30}`. When any required term is missing it
fails open and rejects nothing.

The development promotion gate requires all of the following one-sided 95% Wilson lower bounds:
analytic coverage `>=0.90`, protective recall at `E_hull<=0.05 eV/atom` `>=0.95`, rejection
precision at `E_hull>=0.20` `>=0.90`, and total savings `>=0.10`. Recall against the strict
stability label is reported separately, but `E_hull<=0` is not used as the sole gate, since it
differs considerably in sample definition. Among formulas that clear the gate, the unique formula
is fixed by largest savings, then fewest terms, then the more conservative threshold.

## Confirmation and the rules for concluding

The confirmation set uses the same four gates; AUC, Spearman and stratifications by WBM step,
atom count and valence strategy are also computed, and Pauling 2--5 are evaluated individually
and jointly on the same sample under the same fail-open semantics. Only if NEXT30 clears every
primary gate and none of the Pauling comparators does may we write "surpasses Pauling on this WBM
confirmation endpoint".

Even if it passes, it demonstrates pre-screening of high hull energy only, which is not formation-
energy regression, kinetic stability, synthesizability, or the full discriminating power of DFT.
If no formula clears the development stage, stop immediately and do not open the confirmation
set; if development passes and confirmation fails, keep the frozen formula and the failing result
and do not retune the thresholds.
