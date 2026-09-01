# NEXT98--NEXT100 cross-source discovery and replication design

Date: 2026-08-04

## Boundary

The executable law may use only one raw, unrelaxed generated crystal structure,
frozen element tables, and deterministic analytic geometry, graph, Voronoi,
bond-valence, electrostatic, linear-algebra, symmetry, and coordination
operations. It may not use a DFT calculation or value, a relaxed structure or
trajectory, a learned energy/force/stress proxy or MLIP, a physical relaxation,
or a same-composition alternative.

DFT-relaxed records are offline discovery or evaluation labels only. NEXT98 may
read the already designated SCIGEN discovery endpoint and the already designated
WyFormer discovery endpoint. It must not read either replication endpoint. The
opened NEXT92 and NEXT97 validation results are excluded from feature
calibration, candidate generation, threshold selection, ranking, and gates.

## NEXT98 finite search

1. Join the frozen label-free SCIGEN and WyFormer discovery features to their
   own discovery endpoints. Prefix material identifiers by source and retain a
   source indicator only for evaluation, never as a formula term.
2. Recalibrate the previously frozen analytic term templates on the pooled raw
   discovery x0 features without labels.
3. Build a finite candidate set from the union of deterministic top slices of
   the complete NEXT87 and NEXT95 discovery catalogues plus every eligible
   single term. Candidate formulas contain at most three nonnegative one-sided
   robust hinge terms and use only the previously enumerated weight grid.
4. For each score, choose two inclusive thresholds:
   - SAFE must satisfy the frozen operating gates in every source aggregate and
     every source-by-reduced-formula fold (2 aggregates plus 10 fold cells).
   - BROAD must Pareto-dominate the Pauling P2--P5 baseline on coverage lower
     bound, protected structures kept, severe structures rejected, severe
     precision lower bound, and savings lower bound in all 12 cells, and must
     meet severe-precision lower bound 0.45 in each source aggregate.
5. Each source must independently pass pooled AUC 0.75, macro crystal-system
   AUC 0.60, worst crystal-system AUC 0.55, and at least five evaluable crystal
   systems.
6. Rank passing candidates by the worst source-by-fold severe recall at SAFE,
   then worst SAFE precision lower bound, worst source AUC, total SAFE severe
   rejection, and lower complexity. Ties are lexical.

If no candidate passes every frozen discovery gate, stop and leave both
replication endpoints unopened.

## NEXT99 freeze

If NEXT98 passes, freeze the exact terms, transforms, centers, scales, weights,
SAFE threshold, and BROAD threshold. Before opening any replication endpoint,
write immutable prediction files for both the SCIGEN and WyFormer replication
feature partitions, including identities, grouping fields, Pauling decisions,
score, support, and both decisions. Record SHA-256 identities for all inputs and
outputs.

## NEXT100 two-source one-shot replication

Only after NEXT99 succeeds may NEXT100 open both replication endpoints once.
The formula and predictions cannot change. Each source must independently pass
the same SAFE operating, five-fold, and AUC gates, while BROAD must Pareto-
dominate Pauling in the source aggregate and all five folds. A scientific
cross-source improvement claim is permitted only if both sources pass every
gate. DFT equivalence and universality claims remain forbidden.

All code, artifacts, and reporting are additive. Existing scripts, reports,
paper files, and canonical documents remain untouched.
