# NEXT179/NEXT180 Strong-Neighborhood Directional Closure

## Scientific hypothesis

The normalized weighted Gram used by NEXT173 detects directional balance, but
normalization by total incident weight can still assign a nonzero certificate
when the third independent direction is supported only by weak neighbors. A
locally credible crystal skeleton should contain three independent directions
whose strengths are each comparable to the strongest incident neighbor.

For each site and each unchanged NEXT19 graph, let `w_max` be the largest
incident frozen neighbor weight and define

`H = sum_e (w_e / w_max) * u_e u_e^T`.

Publish two bounded site certificates:

- strong-axis closure: `C = min(1, lambda_min(H))`;
- strong-volume closure: `D = min(1, det(H))`.

Both equal one for three equal-weight orthogonal directions. A frame with
relative orthogonal strengths `1, r, r` has `C=r` and `D=r^2`, directly
penalizing a weak third axis. The construction is invariant to rotation,
direction reversal, edge order, and common weight scale. It uses no fitted
constant, label, energy, force, stress, or relaxation.

## NEXT179 label-free feature build

Use only the sealed discovery geometry cohorts, formal-valence assignment, and
unchanged Voronoi/CrystalNN periodic graphs. For both graph modes publish site
minimum, inverted-CDF q10, and mean of `C`, plus q10 and mean of `D`: exactly
10 bounded hypotheses. Missing or unsupported graph construction fails open
per graph mode. Do not accept endpoint paths.

## NEXT180 frozen discovery audit

Before opening discovery outcomes, freeze all ten hypotheses in the high
direction. Use the unchanged NEXT169/NEXT174 populations and gates:

- full-source support at least 0.90;
- SCIGEN repair-shell worst-fold AUC at least 0.55 in all five folds;
- WyFormer repair-shell pooled AUC at least 0.55;
- both full-source pooled AUC values at least 0.50.

Rank eligible hypotheses by minimum key AUC, then mean key AUC, then stable
hypothesis name. NEXT180 searches no formula. If none is eligible, terminate
the branch. If at least one is eligible, a later separately frozen stage may
search only those eligible features.

## Boundaries and outputs

- Discovery outcomes are offline audit labels only.
- No DFT calculation/value, learned energy/force/stress proxy, or relaxation.
- Validation and replication geometry/endpoints remain sealed.
- Preserve all prior scripts and artifacts; append results only to the
  standalone investigation report.

Publish atomically under:

- `$PRIS_ARCHIVE/next179_strong_neighborhood_directional_closure_v1`;
- `$PRIS_ARCHIVE/next180_strong_neighborhood_directional_closure_audit_v1`.
