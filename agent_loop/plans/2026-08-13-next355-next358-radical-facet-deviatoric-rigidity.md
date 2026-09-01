# NEXT355--NEXT358 radical-facet deviatoric rigidity

Date: 2026-08-13

Status: frozen after the label-blind NEXT351 graph-support stop and before any
NEXT355 feature value or endpoint outcome is opened.

## Scope and information boundary

NEXT351 established without labels that the periodic deviatoric-strain
projection is nonredundant but that an opposite-sign Voronoi graph abstains too
often in WyFormer.  NEXT355 changes only the analytic contact representation:
it uses every reciprocal radius-weighted radical power-cell facet, without
inferring charge signs or deleting like-sign contacts.  No NEXT351 or PARC
outcome exists or is used.

The executable may use only composition, deterministic tabulated radii, and
one initial raw unrelaxed periodic geometry.  It must not use or execute DFT,
an energy/force/stress label or predictor, a learned potential, relaxation, a
trajectory, or any later geometry.  All changes remain additive and canonical
paper/report files remain untouched pending review.

## Frozen graph and law

Use the exact reciprocal radical power-facet graph already certified by
NEXT339.  Each undirected periodic facet edge has endpoints `(i,j)`, Cartesian
periodic displacement `d_e`, and fixed positive conductance

```text
w_e = A_e / |d_e|,
```

where `A_e` is the reciprocal shared-facet area.  Radii use the frozen
calculated-atomic-radius then atomic-radius fallback.  No cutoff, graph mode,
weight exponent, chemistry subgroup, or failure repair is searched.

Apply the NEXT351 kinematic kernel unchanged.  For its fixed orthonormal
five-dimensional symmetric trace-free strain basis, form the weighted
internal-displacement matrix `U`, affine extension matrix `D`, and

```text
H0 = D.T D,
H  = D.T (I - U U^+) D,
M  = H0^(-1/2) H H0^(-1/2).
```

The sole public feature is

```text
rfdr_deviatoric_retention_floor = lambda_min(M),
```

quantized to `1e-10`, with the sole frozen direction `protected_high`.
Rank-deficient `H0`, invalid facet certificates, or numerical violations
abstain.  The least-squares projection is kinematic only and does not alter the
input geometry.

## Certificates and sequential gates

Unit tests must retain NEXT351's zero/unit analytic limits and verify the
wrapper's geometry-only firewall, bounded spectrum, rigid transforms,
unimodular rebasing, and exact supercell invariance.  The power-facet graph
must retain reciprocal-area and volume-tiling certificates.

The label-blind probe uses the same deterministic 80 discovery structures per
source and the same mandatory gates: support at least 72/80, closed `[0,1]`
domain, at least 20 values at 10 decimals, transform error at most `1e-8`, and
absolute Spearman strictly below `0.90`.  Novelty is checked against every
formal label-free feature through NEXT347 and against NEXT351 PDSR recomputed
on the identical selected raw geometries.  Labels/endpoints remain unopened.

Only a passing probe authorizes the all-row NEXT355 label-free build.  Formal
coverage must be at least 0.90 in each of 13,470 SCIGEN and 5,232 WyFormer
rows.  Only a passing formal manifest authorizes NEXT356, which must reuse the
unchanged NEXT224/NEXT268/NEXT324 discovery audit: frozen cohort and
reduced-formula five-folds, inverted-CDF `1/16` and `15/16`, coverage 0.90,
class count 20, pooled AUC 0.55, macro AUC 0.53, worst-fold AUC 0.50, and
`protected_high` direction.  Validation and replication remain sealed.

If any source or gate fails, the branch terminates with no eligible hypothesis
and NEXT357/NEXT358 are not created.  No graph, formula, direction, chemistry,
unsupported-row policy, threshold, or gate may be changed after outcomes are
opened.  Results go only into the independent no-DFT report with exact hashes
and stop decisions.
