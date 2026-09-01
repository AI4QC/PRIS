# NEXT186 Local-Nonlocal Contradiction Relief Audit and NEXT187 Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test and, only if cross-source eligible, search a no-DFT correction that removes the locally driven risk contradicted by strong directional closure and lower independent nonlocal mechanism risks.

**Architecture:** NEXT186 reconstructs the exact capped NEXT163 family means and already published NEXT179 strong-closure features, derives 24 pre-registered local/nonlocal contradiction-relief hypotheses, and audits them on the unchanged NEXT180/NEXT183 populations and gates. NEXT187 is conditional on audit eligibility and searches a frozen finite amplitude family without changing the base, support, repair interval, or validation boundary.

**Tech Stack:** Python 3.11, NumPy, pandas/Parquet, pytest, existing NEXT130/NEXT151/NEXT163/NEXT179/NEXT183/NEXT185 provenance and evaluation helpers.

## Frozen scientific design

All executable inputs are initial composition and geometry only. DFT values, learned energy/force/stress proxies, physical relaxations, validation outputs, and replication outputs are forbidden. Discovery endpoints are offline labels only.

Let `L` be the exact capped `local_geometry` family mean and let `N_charge`, `N_contact`, and `N_valence` be the three exact capped nonlocal family means. Every family mean lies in `[0, 0.5]`. Define two parameter-free local-risk surpluses:

```text
surplus_max  = max(0, L - max(N_charge, N_contact, N_valence))
surplus_mean = max(0, L - mean(N_charge, N_contact, N_valence))
```

`surplus_max` is the strict contradiction: local risk exceeds every independent nonlocal family. `surplus_mean` is the less strict aggregate contradiction. A ratio form is excluded because it is unstable and arbitrarily sensitive when nonlocal risks approach zero.

The closure universe is exactly the six NEXT183 frozen features:

```text
psndc_crystalnn_closure_mean
psndc_crystalnn_closure_min
psndc_crystalnn_closure_q10
psndc_crystalnn_volume_mean
psndc_crystalnn_volume_q10
psndc_voronoi_closure_min
```

For closure `C` in `[0,1]` and surplus `S` in `[0,0.5]`, define two relief conjunctions in the original risk units:

```text
product = C * S
minimum = min(0.5 * C, S)
```

The NEXT186 universe is exactly `6 * 2 * 2 = 24` high-direction hypotheses. Names are `<closure>__<surplus>__<conjunction>__high`; lexicographic evaluation order is mandatory.

NEXT186 reuses unchanged eligibility gates:

- finite full support at least `0.90` in each source;
- SCIGEN repair-shell worst-fold AUC at least `0.55`, with exactly five evaluable folds;
- WYFormer repair-shell pooled AUC at least `0.55`;
- full-source pooled AUC at least `0.50` in each source;
- eligible ranking by decreasing minimum of the four key AUCs, decreasing mean, then name.

No formula is searched in NEXT186.

If at least one hypothesis is eligible, NEXT187 searches one unchanged base plus every eligible relief crossed with:

```text
alpha in (0.25, 0.50, 0.75, 1.00, 1.50, 2.00)
```

The frozen operator is:

```text
active iff BROAD_THRESHOLD <= base_score < SAFE_THRESHOLD
score = max(0, base_score - alpha * contradiction_relief)
```

Unsupported rows, missing relief, and rows outside the interval retain the exact base score and support. Candidate count is `1 + eligible_count * 6`. The candidate universe, amplitudes, thresholds, and gates may not change after NEXT186 outcomes are visible.

If no NEXT186 hypothesis is eligible, NEXT187 is not run; a discovery-only diagnostic may be published but may not redefine the hypothesis universe.

### Task 1: Freeze NEXT186 behavior with failing tests

**Files:**

- Create: `tests/test_next186_local_nonlocal_contradiction_relief_audit.py`

Test exact surplus formulas, missing/support/range behavior, exact product/minimum relief, exact 24 hypotheses, unchanged eligibility forwarding, deterministic selection, discovery-only interface, and missing-input failure. Run the test and confirm RED because the module is absent.

### Task 2: Implement and formally run NEXT186

**Files:**

- Create: `src/next186_local_nonlocal_contradiction_relief_audit.py`
- Create externally: `$PRIS_ARCHIVE/next186_local_nonlocal_contradiction_relief_audit_v1/`

Require exact NEXT185 provenance, reconstruct the base/families/closure tables, attach discovery endpoints, reproduce the frozen shell/full populations, audit all 24 hypotheses, and publish JSON/Parquet/manifest atomically. Verify all hashes and boundary flags.

### Task 3: Conditionally implement and run NEXT187

If NEXT186 has eligible hypotheses, use TDD to create:

- `tests/test_next187_local_nonlocal_contradiction_relief_search.py`
- `src/next187_local_nonlocal_contradiction_relief_search.py`
- `$PRIS_ARCHIVE/next187_local_nonlocal_contradiction_relief_search_v1/`

Tests must freeze the six amplitudes, exact candidate count, interval/missing/support behavior, exact virtual-term recovery, discovery-only interface, and fail-closed inputs. The formal runner must accept only the eligible hypotheses recorded by the exact NEXT186 audit.

### Task 4: Diagnose, report, and verify

If NEXT187 fails, publish a narrow BROAD residual diagnostic before changing mechanism. Append all results and hashes additively to `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`. Run the relevant full pytest set, compileall, formal manifest/source/output verification, `git diff --check`, canonical-path status checks, report integrity checks, and CodeGraph status. Do not modify `paper/`, `tex/`, `notes/`, `README.md`, or `PREREG.md`.
