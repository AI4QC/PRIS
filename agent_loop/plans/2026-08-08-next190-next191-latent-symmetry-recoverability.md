# NEXT190 Latent-Symmetry Recoverability Audit and NEXT191 Search Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether approximate symmetry that can be recovered from a raw, unrelaxed structure supplies a transferable pre-DFT protection certificate for the SCIGEN structures lost at the BROAD operating point, then search a bounded correction only if the certificate passes every frozen cross-source audit gate.

**Architecture:** NEXT190 reuses the six label-free symmetry-recovery quantities already frozen before discovery endpoints were opened in NEXT85 and NEXT94. Their protection-high direction is fixed from the earlier independent OMat development result, not chosen from the current SCIGEN/WyFormer outcomes. NEXT191 is conditional on NEXT190 eligibility and applies only eligible, physically normalized certificates inside the unchanged repair interval.

**Tech Stack:** Python 3.11, NumPy, pandas/Parquet, pytest, the existing NEXT33 symmetry definitions and NEXT151/NEXT163/NEXT164/NEXT169 provenance/evaluation helpers.

## Frozen scientific design

NEXT186 and NEXT188 exclude local/nonlocal contradiction as either a transferable relief or severity signal. The positive full-source but negative repair-shell behavior in NEXT188 shows that another scalar derived from the existing base families is unlikely to add threshold-local information.

NEXT33 supplied an independent, already recorded mechanistic clue. On a different OMat development source, all six preregistered `high = risky` symmetry-recovery directions had AUC below 0.35. That result cannot be relabeled as a success on OMat, but it can prospectively fix a new cross-source hypothesis: a raw structure with stronger latent recovery of crystallographic operations may be a recoverably perturbed valid structure rather than an intrinsically incompatible one.

The exact NEXT190 universe is six protection-high hypotheses:

```text
sym_recovery_onset_rel__recoverable_high
sym_recovery_gain_log2__recoverable_high
sym_orbit_collapse__recoverable_high
sym_recovery_residual_rms_rel__recoverable_high
sym_recovery_residual_q95_rel__recoverable_high
sym_recovery_residual_max_rel__recoverable_high
```

For the audit, raw values are used because AUC is invariant under a monotone rescaling. Direction `+1` means a larger value predicts the protected endpoint. No opposite directions, feature cutoffs, conjunctions, source-specific choices, or binary rules may be added after outcomes are opened.

Support and audit gates are unchanged:

```text
SCIGEN full support >= 0.90
WyFormer full support >= 0.90
SCIGEN repair-shell worst-fold AUC >= 0.55 with all 5 folds evaluable
WyFormer repair-shell pooled AUC >= 0.55
SCIGEN full-extreme pooled AUC >= 0.50
WyFormer full-extreme pooled AUC >= 0.50
```

Eligible hypotheses are ranked by decreasing minimum key AUC, decreasing mean key AUC, then lexical name. NEXT190 searches no formula.

If and only if NEXT190 has at least one eligible certificate, NEXT191 searches:

```text
W = SAFE_THRESHOLD - BROAD_THRESHOLD
active iff BROAD_THRESHOLD <= base_score < SAFE_THRESHOLD
score = max(0, base_score - alpha * W * normalized_certificate)
alpha in (0.25, 0.50, 0.75, 1.00, 1.50, 2.00)
```

The normalizers are fixed by the existing symmetry algorithm, crystallographic operation bound, and dimensionless tolerance scale:

```text
onset_rel       -> clip(value / 0.12, 0, 1)
gain_log2       -> clip(value / log2(48), 0, 1)
orbit_collapse  -> clip(value, 0, 1)
residual_*_rel  -> clip(value / 0.24, 0, 1)
```

Here `0.12` is the maximum frozen relative symmetry tolerance, 48 is the maximum crystallographic point-group order used by the primitive-cell recovery calculation, and `0.24` is two maximum tolerance radii. Missing/unsupported certificates and all rows outside the interval retain the exact base score and support. Candidate count is `1 + eligible_count * 6`. No learned normalizer, binary cutoff, validation result, replication result, DFT value, energy/force/stress proxy, or relaxation may enter the executable formula.

Symmetry recovery can encode a generator's perturbation protocol. Passing NEXT190 therefore authorizes only NEXT191 discovery search; passing NEXT191 would still require the already sealed unseen validation/replication and unseen-source safeguards before a general law claim.

### Task 1: TDD and formal NEXT190 audit

**Files:**

- Create: `tests/test_next190_latent_symmetry_recoverability_audit.py`
- Create: `src/next190_latent_symmetry_recoverability_audit.py`
- Create externally: `$PRIS_ARCHIVE/next190_latent_symmetry_recoverability_audit_v1/`

Test the exact six hypotheses and direction, physical normalizers, unchanged eligibility, deterministic selection, discovery-only interface, and missing-input failure. Confirm RED before implementation. Require exact frozen feature, endpoint, base, NEXT188, plan, and source provenance. Reconstruct the same base score, fixed folds, repair shell, and full-extreme populations used by NEXT186. Publish atomically.

### Task 2: Conditionally TDD and run NEXT191

If NEXT190 has eligible certificates, create:

- `tests/test_next191_latent_symmetry_recoverability_search.py`
- `src/next191_latent_symmetry_recoverability_search.py`
- `$PRIS_ARCHIVE/next191_latent_symmetry_recoverability_search_v1/`

Test exact normalization, interval-only subtraction, missing/base fallback, six amplitudes, exact candidate count, virtual-term recovery, discovery-only interface, and missing-input failure. The formal runner may use only exact NEXT190-eligible certificates.

### Task 3: Diagnose and report

If NEXT191 runs but fails, publish a frozen BROAD residual diagnostic before changing the mechanism. Append NEXT186, NEXT188, and NEXT190/NEXT191 results to the standalone report. Keep all validation/replication endpoints sealed and do not modify canonical manuscript/report files.
