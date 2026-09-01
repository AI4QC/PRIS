# NEXT197 Discrete Protected Exception Search and NEXT198 Diagnosis Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether the cross-source signed-local strong-closure certificate can create a genuinely discrete pre-DFT protected exception that dominates Pauling at the frozen BROAD operating objective, rather than merely applying another insufficient continuous attenuation.

**Architecture:** NEXT197 reconstructs the exact NEXT195 base and six NEXT194-eligible certificates, applies one preregistered interval-fold operator at nine analytic certificate cutoffs, and evaluates all candidates with the unchanged cross-source discovery evaluator. If no candidate passes all gates, NEXT198 exactly reproduces the AUC+SAFE/non-BROAD population and diagnoses its frozen BROAD residuals; if a candidate passes, its identity is frozen before any physically isolated validation endpoint can be opened.

**Tech Stack:** Python 3.11, NumPy, pandas/Parquet, pytest, and the existing NEXT130/NEXT163/NEXT164/NEXT194/NEXT195/NEXT196 provenance and evaluation helpers.

## Frozen scientific design

NEXT196 showed that every NEXT195 candidate still failed aggregate SCIGEN and all five SCIGEN-fold `protected_kept` constraints. The closest continuous correction reduced normalized shortfall only from `0.860419` to `0.816780` and recovered 24 aggregate protected rows, while 409 remained missing. Increasing the same subtraction amplitude is therefore closed.

Three approaches were considered:

1. extend the continuous attenuation grid, rejected because it does not change the decision topology and NEXT196 already shows the residual is far from zero;
2. refit the full mechanism-family decomposition, deferred because it changes many quantities simultaneously and has a much larger hypothesis burden;
3. introduce a binary protected exception backed by an independently audited certificate, selected because it is the smallest interpretable change that can reach a new ROC region.

The exact certificate universe is the six NEXT194-eligible hypotheses, verified from the immutable NEXT194 audit and table:

```text
psndc_crystalnn_closure_min__signed_local_safe__product__high
psndc_crystalnn_closure_min__signed_local_safe__minimum__high
psndc_crystalnn_closure_q10__signed_local_safe__product__high
psndc_crystalnn_closure_q10__signed_local_safe__minimum__high
psndc_crystalnn_volume_q10__signed_local_safe__product__high
psndc_crystalnn_volume_q10__signed_local_safe__minimum__high
```

For base risk `B`, support `M`, certificate `C in [0,1]`, and cutoff `tau`, define

```text
E_tau = M and finite(C) and BROAD <= B < SAFE and C >= tau

score_tau = B * (BROAD / SAFE), if E_tau
            B,                    otherwise.
```

Because `B < SAFE` on every exception row, the folded score is strictly below `BROAD`; multiplication by the frozen positive ratio preserves the base ordering within the exception set. The operator changes a decision region rather than adding another tunable amplitude. Unsupported rows, missing certificates, and rows outside the frozen repair interval retain the exact base score and support.

Cutoffs are fixed analytically on the bounded certificate scale before opening row outcomes:

```text
tau in (1/16, 1/8, 3/16, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8)
```

No empirical quantile, label-derived threshold, source-specific cutoff, alternative folding factor, opposite certificate direction, extra conjunction, cutoff interpolation, or post-result candidate may be added. Candidate count is exactly `1 + 6 * 9 = 55`, including the unchanged base.

Evaluation gates and evaluator semantics remain identical to NEXT195:

```text
SCIGEN pooled/macro/worst AUC >= 0.75/0.60/0.55
WyFormer pooled/macro/worst AUC >= 0.70/0.60/0.55
one common SAFE threshold passes all 12 frozen source/fold cells
one common BROAD threshold strictly dominates Pauling in every frozen cell
```

The published evaluator-selected record follows the existing deterministic ordering. A scientific replacement claim remains forbidden unless one candidate passes source AUC, SAFE, and BROAD simultaneously and then survives a separately frozen, physically isolated one-shot validation. NEXT197 itself has no validation or replication path.

All transforms, centers, scales, weights, cutoffs, graph features, and folding constants are deterministic functions of composition and raw initial unrelaxed geometry. No DFT calculation/value, learned energy/force/stress proxy, relaxed structure, trajectory, or physical relaxation enters the executable criterion. Discovery outcomes are offline search labels only.

### Task 1: TDD the interval-fold operator and frozen catalogue

**Files:**

- Create: `tests/test_next197_discrete_protected_exception_search.py`
- Create: `src/next197_discrete_protected_exception_search.py`

Write tests first for exact interval membership, cutoff equality, strict SAFE exclusion, missing-certificate/base fallback, support preservation, the exact nine rational cutoffs, the six immutable eligible hypotheses, exact 55-candidate catalogue, deterministic identities, discovery-only formal interface, and missing-input failure. Run the target test before implementation and confirm RED because the module does not exist.

### Task 2: Implement and formally run NEXT197

**External output:**

- Create atomically: `$PRIS_ARCHIVE/next197_discrete_protected_exception_search_v1/`

Require exact hashes and provenance for every inherited input through NEXT196, the NEXT194 eligible universe, NEXT195 reproduction artifacts, this plan, and the executed source. Reconstruct the selected NEXT164 base and the six certificates without reading validation/replication. Materialize all 55 scores, run the unchanged evaluator with four workers, reproduce the unchanged base metrics, and publish a manifest, catalogue, discovery evaluation, frozen formula record, and Parquet table.

### Task 3: Follow the preregistered branch

If no candidate passes all discovery gates, create NEXT198 as a diagnostic only. Its population is exactly every NEXT197 record with source-AUC and SAFE pass but BROAD failure, sorted by candidate key. It searches no formula and applies the unchanged NEXT164 residual evaluator, reporting failure frequencies, the global closest record, and the base residual reproduction.

If at least one candidate passes all discovery gates, freeze the evaluator-selected passing identity and first publish a new standalone discovery report. Before any validation access, write a separate one-shot protocol binding that candidate, file hashes, endpoint identity, metrics, and rejection rules. Canonical manuscript/report paths remain untouched pending user confirmation.

### Task 4: Verification

Run target tests, compile the new sources, then run the full test suite. Verify formal output hashes, input identities, exact candidate identities/counts, DFT/proxy/relaxation false flags, unopened validation/replication flags, report fence balance, `git diff --check`, canonical zero-diff, and CodeGraph health. Do not mark the persistent goal complete unless a frozen executable pre-DFT criterion survives every required discovery, validation, replication, and comparison gate.
