# NEXT399--NEXT402 Zachara bond-valence-vector closure freeze

## Frozen boundary and non-duplication audit

This additive branch is frozen before any NEXT399 value or outcome is opened.
The executable accepts only composition and one finite raw unrelaxed periodic
geometry.  It forbids every DFT calculation/value, learned energy/force/stress
proxy, model potential, relaxation, trajectory, validation and replication
input.

NEXT22 implements the usual linear bond-valence-vector resultant with vector
magnitude `s`.  Zachara instead derived the nonlinear normal-flux magnitude
`v=s(1-s/V)` and showed, among other consequences, that an asymmetric linear
two-coordinate environment has a zero resultant when the bonds are collinear
(Zachara, *Inorg. Chem.* 2007, https://doi.org/10.1021/ic7011809; Brown,
*Chem. Rev.* 2009, https://doi.org/10.1021/cr900053k).  Repository audit found
no implementation of this nonlinear magnitude.  NEXT173/NEXT179 use graph
neighbor weights, NEXT295 uses uniform or graph-weighted directions, and
NEXT395's rejected feature is a second moment; none uses `s(1-s/V)` in a first
directional moment.

## Graph and sole formula

Use the existing strict geometry-only firewall and reduced representation,
NEXT19 composition-only formal-valence assignment, the frozen opposite-sign
periodic Voronoi edge graph, and unchanged frozen-fallback bond-valence
parameters.  For every site `i`, rescale its incident raw bond valences to the
formal site magnitude before applying Zachara's formula:

    p_ie = s_e / sum_(f incident i) s_f,
    S_ie = |V_i| p_ie,
    v_ie = S_ie (1 - S_ie/|V_i|) = |V_i| p_ie(1-p_ie).

This is the standard local valence-sum normalization and guarantees
`0 <= S_ie <= |V_i|`; the common `|V_i|` cancels from the normalized
resultant.  Let `u_ie` point outward from site `i`.  Define

    C_i = 1 - ||sum_e v_ie u_ie|| / sum_e v_ie

when the denominator is positive, and `C_i=0` for a one-edge site.  The sole
feature is

    zbvvc_zachara_vector_closure_q10 = inverted-CDF q10_i C_i.

The sole direction is `protected_high`; values are quantized to `1e-10`.
No unnormalized alternative, linear magnitude, graph variant, quantile,
species subset, transform, direction, threshold or conjunction is allowed.

## Pre-outcome loop

Analytic tests require exact closure for unequal collinear two-coordination,
exact closure for an equal trigonal star, loss under angular strain, raw
bond-valence-scale and charge-magnitude invariance, edge order, rotation and
disjoint replication invariance, malformed-input failure, distorted NaCl,
rigid/periodic/supercell equivalences and the no-DFT firewall.

The unchanged deterministic 80+80 discovery-only probe must satisfy per
source: support at least 72, finite `[0,1]`, at least 20 rounded values,
representation error at most `1e-8`, and maximum absolute Spearman strictly
below `0.90` against every adequate prior formal label-free control.  It must
also recompute rejected NEXT387, NEXT391 and NEXT395 controls on the same rows.
No endpoint or later geometry may be opened.

Only a fully passing probe authorizes a complete discovery build with coverage
at least `0.95`.  Only a valid build authorizes one frozen
`zbvvc_zachara_vector_closure_q10__protected_high` audit in the exact NEXT224
rejected-extreme cohort and unchanged `(1/16,15/16)` bounded map.  Per-source
gates remain pooled/macro AUC `>=0.55`, worst-fold AUC `>=0.50`, coverage
`>=0.95`, and pooled/every-fold class counts `>=20`; both sources must pass.
Failure creates no later search.  Passing authorizes only a separately frozen
search and does not establish validation or replacement of existing laws.

Canonical papers, notes, README and preregistration remain untouched; all
results go only to the independent additive report pending user review.
