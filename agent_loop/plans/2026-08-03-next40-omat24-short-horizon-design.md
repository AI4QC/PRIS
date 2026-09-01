# NEXT40 OMat24 short-horizon adaptive validation design

## Status and purpose

NEXT39 revealed, only after its predictions were frozen, that every trajectory
with latest observed step at least 20 exceeded the protected fingerprint-change
cutoff.  NEXT40 is therefore explicitly adaptive and cannot be described as an
independent first-shot confirmation.  It tests the unchanged NEXT23 rule on
parent-disjoint OMat24 trajectories whose latest observed step is 1--19 and
whose later geometries remain unopened.

## Selection boundary

- Exclude every parent frozen in NEXT39 before trajectory selection.
- Require `task_type == "Structure Optimization"`, step 0, and at least one
  later observed step.
- Require latest observed step between 1 and 19 inclusive.
- Keep one deterministic SHA-256-selected trajectory per remaining parent.
- Copy selected compressed LMDB records byte-for-byte without decoding geometry
  or DFT values.  The resulting source remains a mixed raw-record container;
  subsequent cohort projection reads step-0 geometry only.
- Use all eligible remaining parents.  Do not tune strata quotas from outcomes.

## Frozen law and evaluation

Reuse the unmodified NEXT39 cohort freezer, NEXT23 B+E prediction runner,
Pauling P2--P5 controls, exact Matbench Discovery fingerprint evaluator,
operational cutoffs, and four Wilson gates.  No NEXT23 parameter or threshold
may change.  Later sampled geometry is opened only after a new prediction
artifact is published and hash-verified.

## Interpretation

Because the horizon choice is a response to NEXT39, a pass would be adaptive
parent-disjoint validation, not untouched-source confirmation.  A failure or a
single-class endpoint must remain a negative/inconclusive result.  Step number
is evaluation design metadata and may never enter the executable law.
