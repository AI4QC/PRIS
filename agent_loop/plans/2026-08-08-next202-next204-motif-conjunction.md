# Motif conjunction audit and search implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether weakest-site coordination confidence becomes a selective, transferable pre-DFT protection certificate when independently confirmed by low motif dispersion, then search only the eligible frozen certificates.

**Architecture:** NEXT202 is a discovery-only audit over a finite, prospectively frozen certificate set. NEXT203 is conditionally authorized and uses exact discrete protected exceptions inside the existing repair interval. NEXT204 is a diagnostic only if no NEXT203 candidate passes BROAD. Every stage is additive, atomically published, and keeps validation/replication paths absent.

**Tech Stack:** Python 3.11, NumPy, pandas/Parquet, existing NEXT164/NEXT200 evaluators, pytest, CodeGraph.

## Scientific design

The NEXT200 audit authorized `motif_weight_sum_min` in the high-protection
direction. NEXT201 showed that this feature alone is too concentrated near one
to define a selective exception. NEXT202 therefore requires an independent
motif-coherence check.

For `w = motif_weight_sum_min`, define six weakest-site confidence ramps

```text
W_tau = clip((clip(w, 0, 1) - tau) / (1 - tau), 0, 1)
tau in {0, 3/4, 15/16, 63/64, 255/256, 1023/1024}.
```

The two frozen secondary cleanliness certificates are:

```text
C_global = 1 / (1 + max(0, motif_global_dispersion_rms))
C_weight = clip(1 - 2 * max(0, motif_weight_sum_std), 0, 1).
```

The first is a scale-free reciprocal of the nonnegative global fingerprint
dispersion. The second uses the exact `[0, 1/2]` standard-deviation bound for
site-wise weights in `[0, 1]`. Missing input yields a missing certificate and
never grants protection.

For each secondary, floor, and conjunction in `{product, minimum}`, audit

```text
P = W_tau * C
P = min(W_tau, C)
```

in the high-protection direction. The exact NEXT202 universe is `2 * 6 * 2 =
24` hypotheses. No direction reversal, endpoint-derived quantile, learned
transform, DFT quantity, relaxed geometry, model potential, or trajectory is
allowed.

Reuse the unchanged gates:

- SCIGEN and WyFormer full support at least `0.90`;
- SCIGEN repair-shell worst-fold AUC at least `0.55` with all five folds;
- WyFormer repair-shell pooled AUC at least `0.55`;
- both full-source pooled AUCs at least `0.50`.

## Conditionally frozen NEXT203 search

If NEXT202 has no eligible hypothesis, terminate the branch. Otherwise use
only its eligible names. For each certificate, freeze cutoffs

`(1/16, 1/8, 3/16, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8)`.

An exception is active only when base support is true,
`BROAD <= base < SAFE`, the certificate is finite, and
`certificate >= cutoff`. Active scores become

`base * (BROAD / SAFE)`;

all other rows remain exactly unchanged. Search the unchanged base plus the
eligible-certificate/cutoff product with the existing source-AUC, SAFE-cell,
and BROAD-cell evaluator. Validation and replication remain sealed even if a
candidate passes discovery; a separate preregistered freeze would be required.

## NEXT204 diagnostic

If NEXT203 has AUC+SAFE/non-BROAD candidates, exactly reproduce that frozen
population and compute the existing BROAD threshold-table residual: failed
constraint count, normalized shortfall, best threshold, and failure frequency.
Do not search another formula in NEXT204.

### Task 1: Freeze NEXT202 design and tests

**Files:**

- Create: `tests/test_next202_motif_conjunction_audit.py`
- Create: `src/next202_motif_conjunction_audit.py`

**Step 1:** Write tests for the exact 24 hypotheses, both cleanliness maps,
common-support conjunction semantics, unchanged eligibility gates,
deterministic selection, sealed formal interface, and fail-closed missing input.

**Step 2:** Run
`python -m pytest -q tests/test_next202_motif_conjunction_audit.py`
and verify collection fails because the module does not exist.

**Step 3:** Implement the minimal additive audit module by reusing NEXT200 base
reconstruction and evaluator functions. Bind NEXT199--NEXT201 manifests and
current source hashes. Publish audit JSON, table Parquet, and manifest.

**Step 4:** Rerun the test and require all tests to pass.

### Task 2: Run NEXT202 formally

**Files:**

- Create externally: `$PRIS_ARCHIVE/next202_motif_conjunction_audit_v1/`

**Step 1:** Run the formal CLI with discovery-only SCIGEN/WyFormer endpoints,
NEXT98--NEXT201 inputs, the NEXT135 freeze, and this design.

**Step 2:** Verify all input/output hashes, 24 records, support accounting,
boundary flags, and selected/eligible hypotheses.

### Task 3: Conditionally implement NEXT203

**Files:**

- Create: `tests/test_next203_motif_conjunction_exception_search.py`
- Create: `src/next203_motif_conjunction_exception_search.py`

Write tests first for the exact cutoff grid, candidate count, interval-only
activation, fold-below-BROAD score, exact virtual-term recovery, sealed
interface, and fail-closed input. Watch RED, implement, watch GREEN, then run
the formal search only if NEXT202 authorizes it.

### Task 4: Conditionally implement NEXT204

**Files:**

- Create: `tests/test_next204_motif_conjunction_broad_residual.py`
- Create: `src/next204_motif_conjunction_broad_residual.py`

If needed, test exact selection of AUC+SAFE/non-BROAD candidates and
deterministic residual ordering before implementation. Formally reproduce the
published candidates and publish only diagnostic JSON, per-candidate Parquet,
and manifest.

### Task 5: Report and verification

**Files:**

- Modify: `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`

Append NEXT202--NEXT204 evidence without editing canonical paper/report files.
Run targeted tests, source compilation, full pytest, `git diff --check`, manifest
rehashing, Markdown-fence checks, canonical-path status, and CodeGraph status.

The repository is an established dirty additive research workspace; commit and
worktree steps are intentionally omitted to preserve the user's existing state.
