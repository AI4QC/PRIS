# NEXT183 Conditional Nonlocal Cleanliness Closure Audit and NEXT184 Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether a no-DFT certificate that requires both strong local directional closure and independently clean nonlocal mechanisms can safely protect structures in the frozen repair interval, then search a finite additive formula family only if that certificate passes the frozen cross-source audit.

**Architecture:** NEXT183 reconstructs the exact frozen NEXT163 capped family contributions and the already published NEXT179 strong-neighborhood closure features. It derives 36 pre-registered conditional certificates, audits them on the unchanged NEXT180 shell/full populations, and publishes discovery-only evidence atomically. NEXT184 is conditional on at least one eligible certificate and uses only eligible certificates in a frozen finite attenuation grid; otherwise the formula search is not run and a residual diagnostic is permitted.

**Tech Stack:** Python 3.11, NumPy, pandas/Parquet, pytest, existing NEXT130/NEXT151/NEXT163/NEXT179/NEXT180/NEXT182 provenance and evaluation helpers.

## Frozen scientific design

The executable feature path is geometry/composition only. DFT energies, learned energy/force/stress proxies, relaxations, validation outputs, and replication outputs are forbidden. Discovery endpoints are used only as offline audit/search labels.

For each supported row, reconstruct the four capped NEXT163 family means using the exact selected base terms and weights:

```text
F_k = mean_j min(weight_j * risk_j, 0.5),  0 <= F_k <= 0.5
```

The nonlocal set is fixed to:

```text
charge_flow_feasibility
contact_robustness
valence_transport
```

`local_geometry` is deliberately excluded. Define three parameter-free cleanliness summaries:

```text
clean_max     = 1 - max(F_nonlocal) / 0.5
clean_mean    = 1 - mean(F_nonlocal) / 0.5
clean_product = product_k (1 - F_k / 0.5)
```

All values are clipped only for floating-point tolerance into `[0, 1]`; missing or unsupported rows remain missing.

The closure universe is exactly the six NEXT180-eligible features, in sorted feature-name order:

```text
psndc_crystalnn_closure_mean
psndc_crystalnn_closure_min
psndc_crystalnn_closure_q10
psndc_crystalnn_volume_mean
psndc_crystalnn_volume_q10
psndc_voronoi_closure_min
```

For every closure `C` and cleanliness `Q`, define two high-direction conjunctions:

```text
product: C * Q
minimum: min(C, Q)
```

This freezes exactly `6 * 3 * 2 = 36` NEXT183 hypotheses before opening outcomes. Hypothesis names are `<closure>__<cleanliness>__<conjunction>__high` and deterministic lexicographic ordering is mandatory.

NEXT183 reuses the exact NEXT180 populations and gates:

- Full finite support at least `0.90` in each source.
- SCIGEN repair-shell worst-fold AUC at least `0.55`, with exactly five evaluable folds.
- WYFormer repair-shell pooled AUC at least `0.55`.
- Full-source pooled AUC at least `0.50` in each source.
- Rank eligible rows by decreasing minimum of the four key AUCs, then decreasing mean, then hypothesis name.

No formula is searched in NEXT183.

If NEXT183 yields at least one eligible certificate, NEXT184 searches exactly one unchanged base plus every eligible certificate crossed with the frozen attenuation grid:

```text
alpha in (0.25, 0.50, 0.75, 1.00, 1.50, 2.00)
```

The score is:

```text
active iff BROAD_THRESHOLD <= base_score < SAFE_THRESHOLD
score = max(0, base_score - alpha * (SAFE_THRESHOLD - BROAD_THRESHOLD) * certificate)
```

Outside the interval, on unsupported rows, or when the certificate is missing, the exact base score and support are retained. Candidate count is `1 + eligible_count * 6`. Thresholds, candidate set, amplitudes, and selection gates may not change after NEXT183 outcomes are visible.

If NEXT183 has no eligible certificate, NEXT184 is not run. A discovery-only residual diagnostic may compare protected/severe certificate distributions and identify which frozen gate failed, but it may not alter the certificate universe retroactively.

### Task 1: Freeze NEXT183 public behavior with tests

**Files:**

- Create: `tests/test_next183_conditional_nonlocal_closure_audit.py`

**Step 1: Write failing tests**

Cover exact cleanliness formulas, support/missing behavior, range rejection, product/minimum conjunctions, the exact 36-hypothesis universe, unchanged eligibility gates, deterministic selection, discovery-only formal interface, and fail-closed missing inputs.

**Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_next183_conditional_nonlocal_closure_audit.py -q
```

Expected: collection fails because `src.next183_conditional_nonlocal_closure_audit` does not exist.

### Task 2: Implement NEXT183 audit

**Files:**

- Create: `src/next183_conditional_nonlocal_closure_audit.py`

**Step 1: Implement minimal pure functions**

Implement family reconstruction, the three cleanliness summaries, two conjunctions, exact hypothesis catalogue, eligibility forwarding, and deterministic selection.

**Step 2: Implement provenance-closed runner**

Require exact NEXT182 inputs/artifacts, verify NEXT182 terminated without opening validation/replication or using a DFT/proxy/relaxation executable feature, reconstruct the exact base/families, merge NEXT179 closure features, attach discovery endpoints, reproduce NEXT180 populations, evaluate all 36 hypotheses, and publish JSON/Parquet/manifest atomically.

**Step 3: Run tests to verify green**

Run the Task 1 pytest command. Expected: all tests pass.

### Task 3: Execute and verify formal NEXT183

**Files:**

- Create externally: `$PRIS_ARCHIVE/next183_conditional_nonlocal_closure_audit_v1/`

Run the formal CLI with the same frozen input roots as NEXT182 and the design path above. Verify every input/output/source SHA-256, manifest boundary flag, hypothesis count, and selected/eligible consistency. Do not open any validation or replication artifact.

### Task 4: Conditionally implement NEXT184 or a diagnostic

If and only if NEXT183 has eligible certificates, use TDD to add:

- `tests/test_next184_conditional_nonlocal_closure_search.py`
- `src/next184_conditional_nonlocal_closure_search.py`
- `$PRIS_ARCHIVE/next184_conditional_nonlocal_closure_search_v1/`

Tests must freeze the six attenuation values, exact candidate count, interval/missing/support behavior, score recovery through virtual terms, discovery-only interface, and missing-input failure. The runner must accept only the eligible certificates recorded by the exact NEXT183 audit and must use the existing frozen cross-source discovery evaluator without changing gates.

If NEXT183 has no eligible certificates, use TDD to add a narrowly scoped residual diagnostic instead; it must not search formulas or redefine certificates.

### Task 5: Report and verify

**Files:**

- Modify additively: `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`

Append NEXT183 and, if executed, NEXT184/diagnostic methods, hashes, metrics, outcome, and boundary statement. Do not modify `paper/`, `tex/`, `notes/`, `README.md`, or `PREREG.md`.

Run targeted tests, the complete NEXT142-through-new-stage suite, `compileall`, manifest hash verification, `git diff --check`, canonical-path status checks, and CodeGraph status. Report evidence without claiming a replacement unless every frozen discovery and unopened validation requirement is actually satisfied.
