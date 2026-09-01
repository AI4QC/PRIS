# NEXT403--NEXT406 Zachara-over-linear vector-gain freeze

## Frozen boundary and residual mechanism

This additive branch is frozen before any NEXT403 value or outcome is opened.
The executable accepts only composition and one finite raw unrelaxed periodic
geometry.  It forbids every DFT calculation/value, learned energy/force/stress
proxy, model potential, relaxation, trajectory, validation and replication
input.

NEXT399 showed, without opening labels, that the absolute closure of Zachara's
nonlinear bond-valence vectors is rank-redundant with existing vector/closure
descriptors.  It did not test the distinctive claim of the nonlinear formula:
whether `v=s(1-s/V)` closes a given local coordination sphere better than the
traditional linear vector `v=s`.  This residual contrast is frozen only after
NEXT399 terminated, and before computing any residual value.

## Graph and sole formula

Use exactly the NEXT399 firewall, reduced representation, composition-only
formal valence, opposite-sign periodic Voronoi graph, frozen-fallback
bond-valence parameters, sitewise valence normalization and outward unit
directions.  Let

    p_ie = s_e / sum_(f incident i) s_f.

Define the traditional linear and Zachara nonlinear normalized closures

    L_i = 1 - ||sum_e p_ie u_ie||,
    z_ie = p_ie(1-p_ie),
    Z_i = 1 - ||sum_e z_ie u_ie|| / sum_e z_ie,

with `Z_i=0` when `sum_e z_ie=0`.  Both lie in `[0,1]`.  The sole gain and
structure feature are

    G_i = (1 + Z_i - L_i) / 2,
    zbvvg_zachara_over_linear_gain_q10 = inverted-CDF q10_i G_i.

`G_i=0.5` means no advantage; values above 0.5 mean the nonlinear normal-flux
formula closes better.  The sole direction is `protected_high`; values are
quantized to `1e-10`.  No ratio, positive-part clipping, alternate baseline,
graph, quantile, species subset, direction, threshold or conjunction is
allowed.

## Pre-outcome loop

Analytic tests require `G=0.8` for a collinear two-coordinate center with
shares `(0.2,0.8)`, `G=0.5` for equal collinear and equal trigonal stars,
closed bounds, raw bond-valence-scale and charge-magnitude invariance, edge
order, rotation and disjoint replication invariance, malformed failure,
distorted NaCl, rigid/periodic/supercell equivalences and no-DFT firewall.

The unchanged deterministic 80+80 discovery-only probe must satisfy per
source: support at least 72, finite `[0,1]`, at least 20 rounded values,
representation error at most `1e-8`, and maximum absolute Spearman strictly
below `0.90` against every adequate prior formal label-free control.  It must
also recompute rejected NEXT387, NEXT391, NEXT395 and NEXT399 controls on the
same rows.  No endpoint or later geometry may be opened.

Only a fully passing probe authorizes a complete discovery build with coverage
at least `0.95`.  Only a valid build authorizes one frozen
`zbvvg_zachara_over_linear_gain_q10__protected_high` audit in the exact NEXT224
rejected-extreme cohort and unchanged `(1/16,15/16)` bounded map.  Per-source
gates remain pooled/macro AUC `>=0.55`, worst-fold AUC `>=0.50`, coverage
`>=0.95`, and pooled/every-fold class counts `>=20`; both sources must pass.
Failure creates no later search.  Passing authorizes only a separately frozen
search and does not establish validation or replacement of existing laws.

Canonical papers, notes, README and preregistration remain untouched; all
results go only to the independent additive report pending user review.
