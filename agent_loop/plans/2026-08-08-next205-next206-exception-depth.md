# NEXT205--NEXT206 Motif Exception Depth Search Plan

> **For Codex:** Execute this plan with `superpowers:executing-plans`, use
> test-driven development, preserve every existing script and artifact, and
> keep validation/replication sealed unless every frozen discovery gate passes.

**Goal:** Test whether the remaining SCIGEN protected-retention deficit is
caused by the shallow NEXT203 interval fold rather than by the motif
certificate itself, without using DFT values in the executable law.

**Architecture:** Reconstruct the exact NEXT203 discovery population and its
21 NEXT202-eligible motif-conjunction certificates. Keep the original repair
interval and nine certificate cutoffs fixed. Add one interpretable pardon-depth
parameter that moves certified rows to progressively lower risk while leaving
all other rows and support unchanged. Evaluate the exact product under the
unchanged dual-source AUC, SAFE12, and BROAD gates. If no candidate passes all
gates, diagnose the exact AUC+SAFE/non-BROAD residual population and close the
branch.

**Stack:** Python, NumPy, pandas, PyArrow, pytest; existing frozen discovery
feature/endpoint artifacts and evaluator only.

## Frozen scientific boundary

- Executable inputs: composition and initial unrelaxed geometry only.
- Discovery endpoints: offline labels only.
- Forbidden: DFT calculation or DFT value in the formula; learned energy,
  force, or stress proxy; model/proxy potential; relaxed geometry, trajectory,
  or physical relaxation.
- Validation and replication outputs remain unopened during NEXT205--NEXT206.
- Additive files only. Do not modify canonical papers, reports, notes,
  `README.md`, or `PREREG.md`.

## Frozen NEXT205 universe

Let `b` be the unchanged base score, `B` the frozen BROAD threshold, `S` the
frozen SAFE threshold, `c` a NEXT202-eligible certificate, and `k` one of the
existing NEXT203 cutoffs.

Activation is unchanged:

```text
active iff support and B <= b < S and finite(c) and c >= k
```

The new depth grid is frozen before reading NEXT205 outcomes:

```text
d in {0, 1/4, 1/2, 3/4, 1}
q_d = d * (B / S)
b' = q_d * b if active else b
```

For every active row, `b' < B`; `d=1` exactly reproduces NEXT203, while `d=0`
is a full protected pardon. The nine cutoff values remain exactly
`{1/16, 1/8, 3/16, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8}`. The universe is one
unchanged base plus `21 * 9 * 5 = 945` corrections, hence 946 candidates.

The missing policy remains `TERM_OFF_KEEP_BASE`. Base support, scores outside
the interval, and uncertified rows must be bitwise unchanged.

## Frozen gates and ordering

Use the unchanged evaluator and the exact gates inherited by NEXT203:

- both-source pooled/macro/worst-lattice AUC gates;
- all 12 SAFE operating cells;
- common-threshold BROAD dominance over Pauling in every required cell;
- unchanged support/coverage rules and deterministic evaluator ordering.

No validation or replication endpoint may be opened even if a candidate passes
discovery. A passing candidate is only a discovery freeze eligible for a new
standalone report and subsequent user review.

## Task 1: Implement NEXT205 by TDD

**Files:**

- Create: `tests/test_next205_motif_exception_depth_search.py`
- Create: `src/next205_motif_exception_depth_search.py`

Test the exact depth grid, exact candidate count, active-row formula, unchanged
inactive rows/support, exact `d=1` recovery of NEXT203, exact virtual-term
recovery, sealed formal interface, and fail-closed missing inputs. Observe RED,
implement, observe GREEN, then run the formal discovery search.

## Task 2: Conditionally implement NEXT206

**Files:**

- Create: `tests/test_next206_motif_exception_depth_broad_residual.py`
- Create: `src/next206_motif_exception_depth_broad_residual.py`

If NEXT205 has AUC+SAFE/non-BROAD candidates and no all-gate success, freeze the
published diagnostic population, test deterministic selection and residual
ordering, reproduce the published records, and publish diagnostic JSON,
per-candidate Parquet, and manifest only. Search no new formula in NEXT206.

## Task 3: Report and verification

Append NEXT205 and, if authorized, NEXT206 to the existing standalone report
only. Run targeted tests, compilation, full pytest, `git diff --check`, output
hash verification, report-fence checks, protected-path status checks, and
CodeGraph status. Keep the overall goal active unless a candidate clears all
frozen discovery gates and the requested standalone report is ready for user
review.
