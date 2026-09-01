# NEXT395--NEXT398 periodic bond-valence tensor-isotropy freeze

## Frozen boundary and mechanism audit

This additive branch is frozen before any NEXT395 value or outcome is opened.
The executable accepts only composition and one finite raw unrelaxed periodic
geometry.  It forbids every DFT calculation/value, learned energy/force/stress
proxy, model potential, relaxation, trajectory, and validation or replication
input.

The bond-valence vector-sum rule treats bond valence as directional flux and
expects the first directional moment around a spherically symmetric ion to be
near zero (Zachara, *Inorg. Chem.* 2007,
https://doi.org/10.1021/ic7011809; Brown, *Chem. Rev.* 2009,
https://doi.org/10.1021/cr900053k).  NEXT22 already implements that first
moment, so a new vector-sum descriptor would be a duplicate.  NEXT168 measures
an unweighted second directional moment, while NEXT367 measures bond-valence
share uniformity without directions.  The missing joint mechanism is whether
the bond-valence flux itself supplies balanced three-dimensional directional
support.  This is an exploratory extension of the established vector rule,
not a claimed literature law.  Legitimate lone-pair and steric anisotropy are
known caveats and therefore remain possible failure modes.

## Graph and sole formula

Use the existing strict geometry-only firewall and reduced representation,
NEXT19 composition-only formal-valence assignment, and the frozen opposite-sign
periodic Voronoi edge graph.  For edge `e` use the unchanged frozen-fallback
bond-valence parameters and

    s_e = exp((R0_e - r_e) / b_e),       u_e = d_e / ||d_e||.

At every site `i`, normalize the bond-valence-weighted second directional
moment

    T_i = sum_{e incident i} s_e u_e u_e^T / sum_{e incident i} s_e.

`T_i` is positive semidefinite with unit trace.  Define the local isotropy

    I_i = 3 lambda_min(T_i)

and the sole structure feature

    pbvti_bond_valence_tensor_isotropy_q10 = inverted-CDF q10_i I_i.

The sole direction is `protected_high`.  Values are quantized to `1e-10`.
No alternate graph, weight, moment, eigenvalue summary, quantile, species
subset, transform, direction, threshold, or conjunction is allowed.  Global
bond-valence scale is irrelevant because every site tensor is normalized by
its incident bond-valence sum.

## Pre-outcome loop

Analytic tests require exact isotropic and planar stars; sensitivity to a
mean-preserving directional redistribution of bond-valence; bond-valence-scale,
edge-order and disjoint-replication invariance; malformed-input failure;
distorted NaCl support; rigid, periodic and supercell equivalences; and the
no-DFT firewall.

The unchanged deterministic 80+80 discovery-only probe must satisfy per
source: support at least 72, finite `[0,1]`, at least 20 rounded values,
representation error at most `1e-8`, and maximum absolute Spearman strictly
below `0.90` against every adequate prior formal label-free control.  The probe
must additionally recompute the rejected NEXT387 vertex-bypass and NEXT391
ball-growth controls on the same rows.  It may read neither endpoint labels nor
validation/replication geometry.

Only a fully passing probe authorizes a complete discovery build with coverage
at least `0.95`.  Only a valid build authorizes one frozen
`pbvti_bond_valence_tensor_isotropy_q10__protected_high` audit in the exact
NEXT224 rejected-extreme cohort and unchanged `(1/16,15/16)` bounded map.
Per-source gates remain pooled/macro AUC `>=0.55`, worst-fold AUC `>=0.50`,
coverage `>=0.95`, and pooled/every-fold class counts `>=20`; both sources must
pass.  Failure creates no later search.  Passing authorizes only a separately
frozen search and does not establish validation or replacement of existing
laws.

Canonical papers, notes, README and preregistration remain untouched; all
results go only to the independent additive report pending user review.
