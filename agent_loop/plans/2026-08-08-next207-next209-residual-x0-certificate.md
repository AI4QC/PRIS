# NEXT207--NEXT209 Residual X0 Certificate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task.

**Goal:** Find new interpretable initial-structure information that separates
the protected structures still rejected by the best NEXT206 no-DFT rule from
the severe structures it correctly rejects, without tuning the exhausted motif
cutoff or pardon-depth parameters.

**Architecture:** Reconstruct the exact NEXT206 global-closest candidate and
its fixed threshold, then audit a schema-frozen bank of raw, non-identifier,
label-free x0 features only inside its rejected extreme cohort. Require the
same protection direction, high finite coverage, and consistent AUC across two
source aggregates and all ten deterministic reduced-formula folds. Only an
eligible feature may authorize a separately frozen threshold search; otherwise
the existing-feature branch terminates and the next work must construct a new
physical descriptor family.

**Tech Stack:** Python 3.11, NumPy, pandas, PyArrow, pytest, existing
discovery-only feature artifacts and frozen cross-source evaluators.

## Non-negotiable scientific boundary

- Executable inputs are composition and the initial unrelaxed geometry only.
- Discovery outcomes are offline audit labels only.
- Forbidden: DFT calculation or DFT value in an executable formula; learned
  energy/force/stress proxy; model or proxy potential; relaxed structures,
  trajectories, or physical relaxation.
- Validation and replication outputs remain physically unopened throughout
  NEXT207. They may not be opened by NEXT208/NEXT209 unless a complete frozen
  discovery candidate passes every gate, and even then user review is required
  before canonical report or paper edits.
- Additive files only. Preserve every existing script and artifact. Do not
  modify `paper/`, `tex/`, `notes/`, `README.md`, or `PREREG.md`.
- Work in the user-authorized existing dirty checkout; do not commit or create
  another worktree.

## Frozen NEXT206 residual cohort

Reconstruct the exact NEXT206 global-closest candidate identified by candidate
key SHA-256
`26ee85c8dbb8f810eb5baf8c8be07f61d390f2c02c9e45d147c786212b7acc38`:

```text
certificate = weakest_site(tau=15/16) * global_dispersion_cleanliness
certificate cutoff = 5/8
pardon depth = 1/2
fixed residual threshold = 0.16344427817025572
```

The audit cohort is exactly:

```text
current candidate supported
AND finite current score
AND current score >= fixed residual threshold
AND (discovery endpoint <= 1 OR discovery endpoint >= 2)
```

The expected cohort accounting is frozen as:

| Cell | Protected rejected | Severe rejected |
|---|---:|---:|
| SCIGEN aggregate | 316 | 3110 |
| SCIGEN folds 0--4 | 59, 67, 64, 59, 67 | 643, 618, 626, 614, 609 |
| WyFormer aggregate | 331 | 521 |
| WyFormer folds 0--4 | 75, 56, 69, 75, 56 | 95, 101, 115, 114, 96 |

Any row-count mismatch fails closed.

## Frozen raw x0 feature universe

Start from the exact current reconstructed label-free table. A column is
auditable iff it is numeric and all of the following hold:

```text
name not in {
  raw_material_id, source_member_bytes, generated_space_group,
  natoms, geom_species_count
}
name does not start with "_" or "pauling_"
name does not end with "_supported", "_site_count", or "_edge_count"
```

This rule yields exactly 242 raw x0 features. Their sorted newline-joined name
list has SHA-256
`87a20f191ca47b6fb3e9f0255ae8d1e98bcf41e21991af3d290ff222c446f07c`.
Audit both exact directions for every feature:

```text
protected_low  -> risk score = +feature
protected_high -> risk score = -feature
```

The exact audit universe is therefore 484 hypotheses. Do not use source name,
fold, lattice label, material identifier, file size, support flags, Pauling
values, encoded virtual terms, or any discovery outcome as a feature.

## Frozen audit metrics and gates

For each hypothesis, compute severe-positive ROC AUC within the fixed rejected
extreme cohort separately for both source aggregates and all five folds per
source, using only finite feature values.

An hypothesis is eligible for NEXT208 only if all conditions hold:

- finite coverage is at least `0.90` in every aggregate/fold cell;
- each cell retains at least 20 protected and 20 severe rows;
- both source-aggregate AUCs are at least `0.55`;
- both source macro-fold AUCs are at least `0.53`;
- both source worst-fold AUCs are at least `0.50`;
- the opposite direction for the same feature is not also eligible;
- all values and identities reproduce deterministically.

Rank eligible hypotheses by descending minimum worst-fold AUC, then minimum
aggregate AUC, then mean of the two aggregate AUCs, then hypothesis name.
NEXT207 searches no formula and opens no validation/replication artifact.

### Task 1: Write NEXT207 audit contract tests

**Files:**

- Create: `tests/test_next207_residual_x0_feature_audit.py`

**Steps:**

1. Test the exact column-blocking policy and a small schema-derived universe.
2. Test high/low direction mapping and severe-positive AUC behavior.
3. Test per-cell coverage/count/AUC gates, deterministic eligibility ranking,
   sealed formal interface, and fail-closed missing inputs.
4. Run the test module and observe the expected missing-module RED failure.

### Task 2: Implement and run NEXT207

**Files:**

- Create: `src/next207_residual_x0_feature_audit.py`

**Steps:**

1. Implement only the tested schema, direction, audit, and ranking helpers.
2. Reconstruct and verify NEXT205/NEXT206 provenance, candidate identity,
   threshold, cohort counts, feature count, and feature-name digest.
3. Publish atomically: catalogue JSON, audit JSON, all-hypothesis Parquet, and
   manifest with complete input/source/output hashes and explicit no-DFT flags.
4. Run targeted tests, compile the source, then run the formal audit once.

### Task 3: Conditionally freeze NEXT208

**Files:**

- Create only after NEXT207 outcomes: a new dated NEXT208 design/plan.

**Steps:**

1. If zero hypotheses are eligible, terminate the existing-feature branch and
   proceed to a new physical x0 descriptor family; do not loosen gates.
2. If hypotheses are eligible, freeze a small raw-threshold grid derived only
   from label-free feature values, exact exception composition, candidate
   count, and unchanged AUC/SAFE/BROAD gates before evaluating formulas.
3. Implement NEXT208 by TDD and run it only under that written freeze.

### Task 4: Conditionally diagnose NEXT209

**Files:**

- Create: `tests/test_next209_residual_x0_broad_residual.py`
- Create: `src/next209_residual_x0_broad_residual.py`

If NEXT208 produces AUC+SAFE/non-BROAD candidates but no all-gate success,
freeze their exact identity digest, reproduce them, diagnose BROAD residuals,
and search no new formula in NEXT209.

### Task 5: Report and verification

Append verified NEXT207--NEXT209 outcomes to the standalone report only. Run
targeted tests, source compilation, full pytest, `git diff --check`, manifest
output-hash verification, report-fence checks, canonical-path status checks,
and CodeGraph status. Keep the overall goal active unless a candidate passes
all frozen discovery gates and the requested standalone report is ready for
user review.
