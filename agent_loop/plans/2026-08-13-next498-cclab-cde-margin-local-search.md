# NEXT498 CCLAB-CDE margin-local search freeze

Date frozen: 2026-08-13

## Scope and status

NEXT498 is an additive, discovery-only follow-up to the post-coverage
CCLAB-CDE branch.  It does not modify or replace any existing script, report,
paper, note, README, or preregistration.  CCLAB-CDE was introduced only after
NEXT491 exposed a formal coverage failure, so this branch is exploratory and
cannot provide prospective confirmation.

The executable candidate may use only composition and one raw, initial,
unrelaxed periodic geometry.  It may not use a DFT calculation or value, a
learned energy/force/stress proxy, an ML interatomic potential, a physical or
model relaxation, a trajectory, a later geometry, or any validation or
replication output.  Discovery outcomes are permitted only as offline labels
after the candidate universe below is frozen.

## Frozen inputs

- Base frontier: the exact NEXT224 global-closest candidate, including its
  threshold, score, and support mask.
- New scalar: `cclab_cde_conservative_domain_extension` from the formal
  NEXT496 feature build.
- Hypothesis: exactly the one NEXT497 raw-gate survivor,
  `cclab_cde_conservative_domain_extension__protected_high`.
- Feature map: the NEXT497 bounded protection map with its already materialized
  inverted-CDF endpoints, `q_lo = 0.0731047591` and
  `q_hi = 0.8827890278`.  Missing values turn the new term off and retain the
  exact NEXT224 score.
- Repair width: the inherited frozen NEXT215 repair interval width.

## Frozen finite grammar

Reuse the audited NEXT261/NEXT414 triangular margin-local grammar without
adding a free parameter:

- local-width fractions: `1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1`;
- amplitude fractions: `1/4, 1/2, 1`;
- one no-op reproduction control;
- exactly `7 * 3 = 21` eligible new candidates and 22 records in total;
- nonnegative score floor at zero;
- support mask unchanged from NEXT214/NEXT224;
- feature normalization over all finite combined discovery rows and no endpoint
  field used in normalization.

For a protected-high feature, the signed correction lowers the rejection score
near the frozen threshold for high protection and raises it for low protection.
It is triangular in distance from the threshold, and zero outside the selected
local width.  There are no interactions or additional terms.

## Frozen evaluation and selection

Use the existing NEXT223 cross-source discovery evaluator and all inherited
cell, source-AUC, safety, broad, and Pauling-baseline gates unchanged.  The
no-op record must exactly reproduce the NEXT224 reference record.  Reporting
selection excludes the no-op and first requires both source-AUC gates and all
safe cells, then uses the inherited deterministic NEXT223 rank.

- If a selected candidate passes every cross-source discovery gate, freeze it
  only as a candidate requiring unopened internal validation before any claim.
- If source-AUC and safe-cell gates pass but broad-cell gates fail, authorize
  NEXT499 only as the predeclared broad-cell diagnostic; do not change the
  candidate universe after seeing it.
- If no new candidate clears source-AUC plus safe-cell gates, stop the branch.

In every case, internal validation and replication remain unopened, and the
manifest must state that no scientific improvement claim is made.
