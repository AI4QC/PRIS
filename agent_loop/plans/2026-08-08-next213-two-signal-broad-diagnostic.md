# NEXT213 Anchored Two-Signal BROAD Diagnostic Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task.

**Goal:** Reproduce and diagnose the exact 92 NEXT212 candidates that pass
both-source AUC and every SAFE cell but fail BROAD, without searching a new
formula.

**Architecture:** Verify complete NEXT212 provenance, reconstruct its exact
216-candidate universe and fixed anchor, re-run only the frozen 92 keys, and
apply the unchanged threshold-table diagnostic. Rank residuals only by failed
constraint count, normalized shortfall sum, and candidate key. Publish an
additive diagnostic summary/table and close the two-signal branch.

**Tech Stack:** Python 3.11, NumPy, pandas, PyArrow, pytest, and existing
NEXT164/NEXT211/NEXT212 helpers.

## Frozen boundary and population

- Executable inputs are composition plus initial unrelaxed geometry only.
- No DFT calculation/value, learned energy/force/stress proxy, model or proxy
  potential, relaxed structure, trajectory, or physical relaxation is allowed.
- Validation and replication remain unopened.
- Additive files only; canonical manuscripts/reports remain untouched.
- No feature, direction, amplitude, cutoff, threshold, or gate may be changed.

The exact diagnostic filter is

```text
passes_source_auc_gates
AND passes_safe_all_cells
AND NOT passes_broad_all_cells
```

The frozen population has 92 sorted keys with newline-joined digest
`97ee21cc2d73a01ff442fa0c4bf71cb8cd319dc05d0e3a53be3fbcde1d433b1a`.

Formal NEXT212 identities are:

- design: `b2565b6e8135beac57ba3e3120692f1b91affb5ef79f57f783d8e355fec0feea`
- source: `bec4c1a8221af925e4f049af80b62c2f9a400df098c9500e81a5aa98c3b2f3a8`
- manifest: `67213591db6d05c687482fd19cec53e227270d7598acf854735ad9d3056dc3c1`
- catalogue: `8324bedc0cf574b2b5bb5d68c35827d31cf08c88b4ef064cca9d52a219850e76`
- evaluation: `c72b2c4a1da3eb06b2e065d786c9536ac6783b75f2c32444d74fb3e9ea75934e`
- frozen candidate: `bdd91b3a0f4e011e45dc68c33322af68e3d2bf503e88a5a87c6c4fe210337823`
- search table: `3d34987ec0fe10ba113e99836d76eef288ba0b02f5158f83aa0ea0a402097237`

## Tasks

1. Create `tests/test_next213_two_signal_broad_diagnostic.py`; observe
   missing-module RED for exact population selection, digest, deterministic
   residual ranking, discovery-only interface, and missing-input failure.
2. Create `src/next213_two_signal_broad_diagnostic.py`; verify all NEXT212
   identities, reconstruct all 216 keys, re-run the exact 92, diagnose each
   BROAD threshold table, and publish atomically.
3. Run formally into
   `$PRIS_ARCHIVE/next213_two_signal_broad_diagnostic_v1/`.
4. Append NEXT210--NEXT213 evidence to the standalone report and run targeted,
   compilation, full-suite, hash, boundary, report-fence, git, and CodeGraph
   verification. Do not commit in the dirty additive workspace.
