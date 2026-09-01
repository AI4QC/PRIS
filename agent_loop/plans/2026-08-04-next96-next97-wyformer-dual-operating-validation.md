# NEXT96--NEXT97 WyFormer dual-operating candidate validation

## Amendment status

NEXT95's original single-threshold promotion gate required one operating point
to retain at least 90% of protected structures with at least 80% severe
precision **and** reject more severe structures than Pauling.  No searched
formula met that joint condition.  The strongest formula nevertheless passed
all safe operating, five-fold, pooled-AUC, and crystal-system-AUC gates; it
failed only the absolute severe-count comparison because Pauling achieves its
count by rejecting many protected structures.

This amendment is written after discovery and before any NEXT93b validation or
replication endpoint is opened.  It does not alter or overwrite NEXT95.  It
defines two fixed operating thresholds for one unchanged analytic score and a
new one-shot validation protocol.

## Frozen analytic score

For a raw generated x0 structure, define the robust one-sided hinge

`h_j(x) = max(0, direction_j * (transform_j(x) - center_j) / scale_j)`.

The score is

`R = h(cov_site_overlap_max) + h(scbv_mismatch_q95) + 0.5 h(sivr_edge_mismatch_rms)`.

All transforms, centers, scales, directions, and missing-value behavior are
copied byte-for-byte from the NEXT95 selected formula.  A row is supported
only when all three terms are finite; otherwise both operating points ABSTAIN
and fail open.

The score uses raw x0 analytic geometry, short-contact, bond-valence, and
valence-rigidity quantities only.  It uses no DFT calculation or value, no
relaxed structure, no trajectory, no learned energy/force/stress proxy, no
MLIP, and no same-composition alternative at execution time.

## Two fixed thresholds

1. **SAFE**: reject when `R >= 3.356904710858153`.
   This is the original NEXT95 selected threshold.
2. **BROAD**: reject when `R >= 0.5035394897502813`.
   This is the lowest discovery threshold that Pareto-dominates Pauling in
   every one of the five frozen reduced-formula folds on all of:
   coverage lower bound, protected structures kept, severe structures
   rejected, severe-precision lower bound, and savings lower bound.

The two thresholds are frozen here.  Neither may be changed after validation
is opened.  They are two operating strengths of the same law, not two fitted
models.

## Discovery evidence required before prediction freeze

SAFE must satisfy:

- coverage lower 95% bound >= 0.90;
- protected recall lower 95% bound >= 0.90;
- severe precision lower 95% bound >= 0.80;
- savings lower 95% bound >= 0.02;
- pooled protected-vs-severe AUC >= 0.75;
- macro crystal-system AUC >= 0.60;
- worst crystal-system AUC >= 0.55;
- at least five evaluable crystal systems;
- all five fixed formula-and-threshold folds pass the four operating gates.

BROAD must, in the pooled discovery set and separately in every frozen fold,
strictly exceed Pauling's supported-coverage lower bound, severe count,
severe-precision lower bound, and savings lower bound while keeping at least
as many protected structures.  Its pooled severe-precision lower bound must
also be at least 0.45.

Only after these facts are reproduced may predictions for all three x0 feature
partitions be frozen.  The freeze runner may read the already-open discovery
endpoint, but it has no validation or replication endpoint argument.

## NEXT97 one-shot internal validation

NEXT97 opens only the physically isolated NEXT93b internal-validation endpoint
and evaluates the already frozen predictions.

Validation passes only if:

- SAFE passes every aggregate operating and AUC gate listed above;
- SAFE passes the four operating gates in every one of five reduced-formula
  folds using the unchanged threshold;
- BROAD has pooled severe-precision lower bound >= 0.45;
- BROAD Pareto-dominates Pauling on the five comparison quantities in the
  pooled validation set and in every frozen validation fold;
- the score, both thresholds, transforms, missing policy, and predictions are
  unchanged from NEXT96.

Any failed condition stops the branch.  No formula or threshold adjustment is
allowed.  Internal replication is authorized only after a complete NEXT97
pass and otherwise remains unopened.

## Reporting

Whether validation passes or fails, write a new standalone report.  Do not
modify old reports, canonical reports, README files, or manuscript sources
before user review.
