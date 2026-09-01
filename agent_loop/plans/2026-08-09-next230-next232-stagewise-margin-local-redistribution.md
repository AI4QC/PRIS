# NEXT230--NEXT232 Stagewise Margin-Local Redistribution Plan

**Goal:** Test whether one additional fully pre-DFT, threshold-local signed
certificate can remove the remaining SCIGEN protection failures at the exact
five-failure NEXT229 frontier while preserving all currently passed SCIGEN
fold-4 and WyFormer BROAD constraints.

**Architecture:** NEXT230 reconstructs the exact NEXT229 global closest record
and audits the complete raw x0 feature bank only inside its rejected extreme
cohort. NEXT231 adds one triangular signed term, centered at the frozen
NEXT229 diagnostic threshold, for every source-stable eligible hypothesis and
every frozen width/amplitude pair. NEXT232 only reproduces the exact eligible
AUC+SAFE/non-BROAD population and computes the unchanged BROAD residual.

**Tech Stack:** Python 3.11, NumPy, pandas/Parquet, the existing NeoPauling
cross-source evaluator, pytest, and SHA-256 manifests.

Date: 2026-08-09

Status: frozen after the published NEXT227--NEXT229 outcomes and the
identity/cohort reconstruction below, but before any NEXT230 feature AUC or
NEXT231 candidate score is computed.

## Scientific rationale and alternatives

NEXT229 reduced the best diagnostic failed-constraint count from six to five.
The exact closest record now passes every WyFormer BROAD constraint and the
SCIGEN fold-4 constraints. Its remaining failures are aggregate SCIGEN and
SCIGEN folds 0--3 protected retention. This removes the previous fold-4
protection-versus-savings conflict but does not yet produce a valid law.

Three continuations were considered:

1. post-outcome adjustment of the first term's width, amplitude, or threshold,
   rejected because those values have already been inspected;
2. a source-specific SCIGEN exemption, rejected because source identity is not
   a physical executable input and would not generalize;
3. a second source-agnostic margin-local certificate, selected because it can
   rescue protected-like rows while lifting severe-like rows, is directly
   interpretable, and can be audited independently across both sources and all
   folds before its candidate outcomes are evaluated.

This is an exploratory stagewise branch. The NEXT229 diagnostic record was
selected using discovery outcomes and is not validated, promoted, or eligible
for a scientific claim merely by serving as the frozen base here.

## Immutable no-DFT and data boundary

- Executable quantities may use composition and initial, unrelaxed geometry
  only.
- The executable law must not use a DFT calculation or value; learned energy,
  force, or stress proxy; model or proxy potential; relaxed structure;
  trajectory; or physical relaxation.
- Discovery outcomes are offline labels used only by the feature audit and
  unchanged discovery evaluator.
- Use only the already opened SCIGEN and WyFormer discovery endpoints.
- Internal validation and replication endpoints remain physically sealed
  unless an eligible new candidate passes every discovery gate.
- Preserve every existing script, test, result, plan, report, and canonical
  artifact. Add new files and append only to the independent report.

## Frozen NEXT229 base identity

- Base candidate-key digest:
  `3115dd8189d1125c3863f09a107c083fd81f538d3ac27351273cd8a4bbe41b5a`.
- Base hypothesis: `sivr_cell_hydro_abs__protected_low`.
- Base first-stage local-width fraction: `1/4`.
- Base first-stage amplitude: `1`.
- Base first-stage cutoffs:
  `q_lo=9.854416130451319e-05`,
  `q_hi=0.022954334010962262`.
- Base failed-constraint count: `5`.
- Base normalized shortfall: `0.16431186635663908`.
- Base diagnostic decision threshold: `0.10672744194580967`.
- Base support: exact NEXT214 support, `18,017` rows.

The implementation must independently recompute the candidate-key digest and
all base metrics from the published NEXT229 diagnostic, reconstruct the exact
score from the NEXT224 score plus the published NEXT228 term, and reproduce
the published NEXT228 evaluator record before use. No coefficient is refitted.

## NEXT230 frozen residual feature audit

Select the exact 242 sorted numeric, raw, non-identifier x0 columns using the
existing NEXT207/NEXT227 schema policy. Their name digest must remain
`87a20f191ca47b6fb3e9f0255ae8d1e98bcf41e21991af3d290ff222c446f07c`.
Audit both `protected_high` and `protected_low` directions for every feature.

The cohort is fixed before feature values are ranked:

```text
supported
AND finite NEXT229-base score
AND score >= 0.10672744194580967
AND (endpoint <= 1 OR endpoint >= 2)
```

Expected protected/severe counts are:

| cell | protected | severe |
| --- | ---: | ---: |
| SCIGEN all | 249 | 3086 |
| SCIGEN folds 0--4 | 50, 54, 49, 53, 43 | 629, 621, 609, 614, 613 |
| WyFormer all | 345 | 522 |
| WyFormer folds 0--4 | 79, 59, 72, 73, 62 | 97, 104, 114, 111, 96 |

Use the exact NEXT227 source/fold gates independently for both sources:
minimum coverage `0.90`, minimum protected and severe count `20`, aggregate
AUC `0.55`, macro-fold AUC `0.53`, and worst-fold AUC `0.50`. Apply the exact
opposite-direction veto: a feature is eligible only when exactly one direction
passes all gates. Rank only for reporting by minimum worst-fold AUC, minimum
aggregate AUC, mean aggregate AUC, then hypothesis identity. NEXT231 must use
every eligible hypothesis, not only the reporting leader.

NEXT230 searches and selects no formula. If no hypothesis is eligible, close
the stagewise branch without running NEXT231.

## NEXT231 frozen executable grammar

Let `s1` be the exact NEXT229-base score, `t1` the frozen NEXT229 diagnostic
threshold, `W` the original NEXT214 repair width, `P` one eligible bounded
protection certificate, `f` a second-stage local-width fraction, and `beta` an
amplitude. Define

```text
h2 = f * W
local_weight2 = max(0, 1 - abs(s1 - t1) / h2)
local_delta2 = beta * h2 * local_weight2 * (1 - 2 * P)
score2 = max(0, s1 + local_delta2)
```

The proposed term is exactly zero at or beyond distance `h2`. Missing feature
values turn only the second-stage term off. Support remains exactly NEXT214
support. Certificate cutoffs are endpoint-blind 1/16 and 15/16 inverted-CDF
quantiles over all finite combined discovery feature values, using the
existing NEXT216 normalization.

Frozen grids:

- second-stage local-width fractions `f`:
  `{1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1}`;
- amplitudes `beta`: `{1/4, 1/2, 1}`.

The wider final two fractions are included before outcome inspection because
the 25th percentile and median SCIGEN protected-reject excesses are
`0.033149849947624946` and `0.22193547091008808`; the current aggregate
deficit requires rescuing 78 of 249 rejected protected structures, which a
width capped at `W/4` might not reach. The same signed term raises rather than
rescues severe-like rows, so wider candidates remain physically symmetric and
must still pass every source/fold gate.

For `K` NEXT230 eligible hypotheses, the complete catalogue contains one exact
NEXT229-base reproduction control and `21*K` eligible new candidates. No
feature pruning, family filtering, endpoint-dependent cutoff, source/fold
rule, coefficient fit, beam search, or manual override is allowed after
NEXT230.

Run the unchanged cross-source evaluator. Only eligible new candidates may be
selected. If any pass every discovery gate, select by the unchanged evaluator
rank and stop. Otherwise report the evaluator's best eligible AUC+SAFE record
without using BROAD residual to select it.

## NEXT232 frozen residual diagnostic and stopping rule

If NEXT231 has no all-gate candidate and at least one eligible AUC+SAFE/non-
BROAD candidate, reconstruct that exact population, verify its sorted-key
digest, reproduce all evaluator records, and compute the unchanged BROAD
threshold residual for every member. Rank by failed-constraint count,
normalized shortfall, and candidate key.

- The branch advances only if the global closest record strictly improves
  `(5, 0.16431186635663908)` under this lexicographic ordering.
- If it does not, close the stagewise margin-local branch.
- If it improves but remains non-BROAD, any continuation requires a new
  pre-outcome freeze.
- If an eligible candidate passes every discovery gate, freeze a separate
  validation protocol before opening validation data.

Validation and replication stay sealed in every non-all-gate case.

## Additive implementation and verification

Create only:

- `src/next230_stagewise_margin_local_feature_audit.py`
- `tests/test_next230_stagewise_margin_local_feature_audit.py`
- `src/next231_stagewise_margin_local_search.py`
- `tests/test_next231_stagewise_margin_local_search.py`
- `src/next232_stagewise_margin_local_broad_diagnostic.py`
- `tests/test_next232_stagewise_margin_local_broad_diagnostic.py`
- formal directories `next230_stagewise_margin_local_feature_audit_v1`,
  `next231_stagewise_margin_local_search_v1`, and, if authorized,
  `next232_stagewise_margin_local_broad_diagnostic_v1` under the external data
  root.

Use TDD for the cohort, exact base reconstruction, score, catalogue,
interfaces, provenance, and fail-closed behavior. After formal execution,
append exact results and hashes only to
`reports/2026-08-08-next115-next117-hcid-no-dft-search.md`. Run focused tests,
`py_compile`, the full pytest suite, independent manifest/hash checks,
`git diff --check`, trailing-whitespace and canonical-path checks, and
CodeGraph status. Do not modify `paper/`, `tex/`, `notes/`, `README.md`, or
`PREREG.md`.
