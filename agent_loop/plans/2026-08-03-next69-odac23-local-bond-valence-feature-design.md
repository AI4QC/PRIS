# NEXT69 ODAC23 local bond-valence x0 features

## Purpose

NEXT68 improved robust-discovery pooled AUC to 0.827 but failed the frozen
reject-precision gate.  NEXT69 adds a local, mechanistic descriptor family for
underbonded or directionally unbalanced metal environments.  It is an additive
feature artifact only; no decision threshold is selected here.

## Frozen inputs and execution boundary

Use the sealed NEXT54 framework-only raw x0 archive and the sealed NEXT65
feature table.  Build features for discovery, internal validation, and internal
replication together without opening any endpoint label file.  Inputs are one
unrelaxed framework geometry, frozen elemental/oxidation-state tables,
CrystalNN or Voronoi neighbor graphs, and analytic bond-valence arithmetic.
No DFT calculation/value, relaxed geometry, energy/force/stress model, proxy
potential, physical relaxation, or same-composition alternative is permitted.

## Frozen feature algorithm

Infer a neutral site-valence assignment with the existing NEXT19 deterministic
integer, fractional, then electronegativity fallback.  Independently construct
opposite-sign periodic graphs with frozen `crystalnn` and `voronoi` modes.  For
each graph resolve each bond's frozen bond-valence parameters using the existing
exact/nearest/fallback policy and evaluate the NEXT22 globally scale-calibrated
features, excluding the non-intensive edge and site counts.

Using the identical bonds, strengths, and one closed-form global amplitude,
also aggregate metal-site mismatch, deficit, excess, vector asymmetry,
effective coordination number, and fractions with normalized deficit above
0.25 or 0.50.  Prefix every feature with its graph mode.  A failed graph emits
NaN for that mode and is fail-open; it does not make the pre-existing NEXT65
row unsupported.  No label-dependent imputation or calibration is allowed.

## Publication

Merge by exact `material_id`, preserve all NEXT65 columns, append the frozen
NEXT69 columns and per-mode support/failure fields, and publish to a new
no-replace directory with source/input/output hashes.  Old artifacts remain
unchanged.
