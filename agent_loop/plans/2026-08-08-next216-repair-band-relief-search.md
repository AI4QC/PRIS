# NEXT216 Repair-Band Relief Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task.

**Goal:** Test whether one NEXT215-stable raw x0 protection certificate can
continuously rescue protected structures from the fixed NEXT214 repair band
while preserving support and leaving the frozen high-risk region unchanged.

**Architecture:** Verify and reconstruct the exact NEXT214 three-term score and
the complete NEXT215 audit. Fit endpoint-blind robust cutoffs for each of the
22 frozen eligible features using all finite raw values in the fixed repair
band. Materialize the unchanged score plus four bounded multiplicative relief
strengths per feature, then run the unchanged dual-source AUC/SAFE/BROAD
evaluator. Validation and replication remain sealed regardless of outcome.

**Tech Stack:** Python 3.11, NumPy, pandas, PyArrow, pytest, and existing
NEXT164/NEXT210/NEXT214/NEXT215 helpers.

## Hard boundary

- Runtime inputs remain composition plus initial unrelaxed geometry only.
- No DFT calculation/value, learned energy/force/stress proxy, model or proxy
  potential, relaxed structure, trajectory, or physical relaxation may enter.
- Discovery endpoints remain offline evaluator labels only.
- Support is unchanged. Missing feature values keep the NEXT214 score.
- Rows below the repair interval and at or above its upper endpoint remain
  bitwise unchanged.
- Validation and replication artifacts remain physically unopened.
- Additive files only; do not edit canonical paper/report paths and do not
  commit in the user-authorized dirty checkout.

## Frozen NEXT215 provenance

- design: `b8f1465d2dc0b4ee56ccbefcc162cf5d7d60fd64a4330ca140de0eb7caa2c5a5`
- source: `d6e5232b004a934f05ef7c7cc5d4c1237474fa31e9420146239cff402c8e8e11`
- manifest: `44bc4c90ae4ca689effaa0a1dc2de05b03792b972d96e32039ed85ff7cd0c9c3`
- catalogue: `56e0bbd8e0ca178dc1d98b3ecdb449870b384de6e485143d2908c6446a7f4b85`
- audit: `2e99e076af8e35bcd2522e179a504ce5ae9e0a8f6758ae454ff2a71c33b28489`
- table: `e73fc703484e364f5f4ac6b8a5897b727064a95b974b194b55edaf563b72a9bb`

NEXT215 reproduced 242 raw features and 484 directional hypotheses. It found
23 raw-gate passes and 22 eligible hypotheses after the frozen used-feature
veto. Their sorted identity digest is
`2e5000a319188a6191922a499b8151e28bb603ba06e70cff8750ec582e887b41`:

```text
cov_q01__protected_high
cov_ratio_q50__protected_low
csf_gaussian_t040__protected_low
csf_gaussian_t060__protected_low
csf_long_fraction__protected_low
geom_covalent_radius_mean__protected_low
nm_reciprocal_reduced__protected_low
prlr_contact_weight_rms__protected_low
psndc_crystalnn_closure_mean__protected_high
psndc_crystalnn_closure_min__protected_high
psndc_crystalnn_closure_q10__protected_high
psndc_crystalnn_volume_mean__protected_high
psndc_crystalnn_volume_q10__protected_high
psndc_voronoi_closure_min__protected_high
psndc_voronoi_closure_q10__protected_high
psndc_voronoi_volume_mean__protected_high
psndc_voronoi_volume_q10__protected_high
scbv_cation_mismatch_rms__protected_low
scbv_effective_cn_min__protected_high
scbv_mismatch_q95__protected_low
scbv_mismatch_rms__protected_low
sivr_cell_hydro_abs__protected_low
```

Any count, identity, direction, digest, provenance hash, boundary flag, or
published metric mismatch fails closed.

## Frozen candidate construction

Reconstruct the NEXT214 final score `s`, support, and features exactly. The
repair band is fixed and lower-inclusive/upper-exclusive:

```text
lower = 0.17470215862148156
upper = 0.570892727856757
fit population = support AND finite(s) AND lower <= s < upper
```

The normalization population includes all sources and all endpoint values; no
endpoint is read to fit cutoffs. For each feature, compute `q_lo` and `q_hi` as
the 1/16 and 15/16 inverted-CDF quantiles of its finite fit-population values.
Degenerate cutoffs fail closed. Define the protected-positive certificate:

```text
protected_high: P = clip((x - q_lo) / (q_hi - q_lo), 0, 1)
protected_low:  P = clip((q_hi - x) / (q_hi - q_lo), 0, 1)
```

The amplitude grid is immutable:

```text
alpha in {1/16, 1/8, 1/4, 1/2}
```

For each feature/amplitude pair:

```text
active = support AND finite(s) AND lower <= s < upper AND finite(P)
s'[active] = s[active] * (1 - alpha * P[active])
s'[not active] = s[not active]
support' = support
```

Scores must remain finite and nonnegative on support, never increase on active
rows, and remain bitwise identical below `lower`, at/above `upper`, and where
the feature is missing. The candidate universe is exactly:

```text
1 unchanged NEXT214 score + 22 features * 4 amplitudes = 89 candidates
```

No conjunction, feature combination, width search, amplitude interpolation,
normalization refit, threshold prefilter, beam search, or manual override is
allowed.

## Frozen evaluation and decision

Run the unchanged dual-source evaluator used by NEXT210--NEXT214. Publish every
candidate and count exact passes for source AUC, SAFE, BROAD, and all gates.

- If at least one candidate passes all discovery gates, publish the evaluator's
  frozen selected candidate and keep validation sealed for user review.
- Otherwise, if any candidate passes AUC+SAFE but not BROAD, freeze its exact
  population and authorize diagnostic-only NEXT217.
- Otherwise, terminate this branch without loosening gates.

Formal output directory:

```text
$PRIS_ARCHIVE/next216_repair_band_relief_search_v1/
```

Publish atomically:

- `MANIFEST.json`
- `NEXT216_REPAIR_BAND_RELIEF_CATALOGUE.json`
- `NEXT216_DISCOVERY_EVALUATION.json`
- `NEXT216_FROZEN_CANDIDATE.json`
- `next216_repair_band_relief_search.parquet`

## Tasks

1. Create `tests/test_next216_repair_band_relief_search.py`; observe RED for
   robust cutoffs, certificate directions, exact interval behavior, candidate
   count/identities, sealed interface, and missing-input failure.
2. Create `src/next216_repair_band_relief_search.py`; implement only the frozen
   helpers, verify NEXT215, reconstruct NEXT214, materialize 89 candidates, run
   the unchanged evaluator, and publish atomically.
3. Run targeted tests and compilation, then execute once into the frozen
   external directory.
4. Independently verify source/design/manifest/output hashes, candidate and
   gate counts, no-DFT boundary flags, and unopened validation/replication.
5. If authorized, freeze a separate NEXT217 diagnostic plan before writing its
   code. Otherwise record the predeclared stop.
