# NEXT233--NEXT235 Final Stagewise Margin-Local Redistribution Plan

**Goal:** Test whether one final, fully pre-DFT, threshold-local signed
certificate can close the five remaining SCIGEN protected-retention failures
of the exact NEXT232 frontier without losing the already passed SCIGEN fold-4
or WyFormer BROAD constraints.

**Architecture:** NEXT233 reconstructs the exact NEXT232 global closest record
and audits the complete raw x0 feature bank only in its rejected extreme
cohort. NEXT234 adds one final triangular signed term, centered at the frozen
NEXT232 diagnostic threshold, for every source-stable eligible hypothesis and
every frozen width/amplitude pair. NEXT235 only reproduces the exact eligible
AUC+SAFE/non-BROAD population and computes the unchanged BROAD residual. This
is the last permitted stagewise term in this branch.

**Tech Stack:** Python 3.11, NumPy, pandas/Parquet, the existing NeoPauling
cross-source evaluator, pytest, and SHA-256 manifests.

Date: 2026-08-09

Status: frozen after publication of NEXT232 and the identity/cohort
reconstruction below, but before any NEXT233 feature AUC or NEXT234 candidate
score is computed.

## Scientific rationale and stopping boundary

NEXT232 retained five failed constraints but reduced normalized shortfall from
`0.16431186635663908` to `0.12730274611313003`, a reduction of
`0.03700912024350905`. Its remaining failures are SCIGEN aggregate and folds
0--3 protected retention. SCIGEN fold 4 and every WyFormer BROAD constraint
pass. The improvement is distributed across four failing folds, so one more
source-agnostic residual certificate is scientifically testable.

The cost is an additional executable term selected using discovery outcomes.
To prevent indefinite stagewise overfitting, this protocol permits exactly one
final term. If NEXT235 remains non-BROAD, the stagewise margin-local branch is
closed even if its diagnostic residual improves. No fourth residual audit,
post-outcome grid expansion, feature substitution, or threshold adjustment is
allowed in this branch.

The NEXT232 diagnostic record was selected on discovery outcomes. It is not a
validated law and is not promoted merely by serving as the frozen base here.

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

## Frozen NEXT232 base identity

- Exact second-stage candidate-key digest:
  `6313dbca02ced34ec999f9c5b13616ad4e1d69d1ceef01bf3b3c57efe45ac913`.
- First stage: `sivr_cell_hydro_abs__protected_low`, width `1/4`, amplitude
  `1`, with the published NEXT228 cutoffs.
- Second stage: `nm_charge_concentration__protected_high`, width `1/4`,
  amplitude `1/2`, `q_lo=1.0`, `q_hi=4.596429084968457`.
- Base failed-constraint count: `5`.
- Base normalized shortfall: `0.12730274611313003`.
- Base diagnostic decision threshold: `0.13768812689345988`.
- Base support: exact NEXT214 support, `18,017` rows.
- Published second-stage active rows: `10,061` total, `7,682` SCIGEN and
  `2,379` WyFormer.

The implementation must independently verify every published NEXT232 input
and output hash, reproduce the 450-member NEXT232 diagnostic population,
reconstruct the selected score exactly from the published NEXT229 score and
NEXT231 term, reproduce its candidate-key digest and BROAD diagnostic record,
and only then use it as the base. No coefficient is refitted.

## NEXT233 frozen residual feature audit

Select the exact 242 sorted numeric, raw, non-identifier x0 columns under the
existing NEXT207/NEXT227 schema policy. Their name digest must remain
`87a20f191ca47b6fb3e9f0255ae8d1e98bcf41e21991af3d290ff222c446f07c`.
Audit both `protected_high` and `protected_low` directions for every feature.

The cohort is fixed before feature values are ranked:

```text
supported
AND finite NEXT232-base score
AND score >= 0.13768812689345988
AND (endpoint <= 1 OR endpoint >= 2)
```

Expected protected/severe counts are:

| cell | protected | severe |
| --- | ---: | ---: |
| SCIGEN all | 227 | 3041 |
| SCIGEN folds 0--4 | 47, 51, 47, 43, 39 | 624, 622, 606, 588, 601 |
| WyFormer all | 329 | 524 |
| WyFormer folds 0--4 | 74, 58, 66, 71, 60 | 97, 104, 113, 114, 96 |

Use the exact NEXT227/NEXT230 source/fold gates independently for both
sources: minimum coverage `0.90`, minimum protected and severe count `20`,
aggregate AUC `0.55`, macro-fold AUC `0.53`, and worst-fold AUC `0.50`.
Apply the exact opposite-direction veto: a feature is eligible only when
exactly one direction passes all gates. Rank only for reporting by minimum
worst-fold AUC, minimum aggregate AUC, mean aggregate AUC, then hypothesis
identity. NEXT234 must use every eligible hypothesis, not only the reporting
leader.

NEXT233 searches and selects no formula. If no hypothesis is eligible, close
the branch without running NEXT234.

## NEXT234 frozen executable grammar

Let `s2` be the exact NEXT232-base score, `t2` the frozen NEXT232 diagnostic
threshold, `W` the original NEXT214 repair width, `P` one eligible bounded
protection certificate, `f` a third-stage local-width fraction, and `beta` an
amplitude. Define

```text
h3 = f * W
local_weight3 = max(0, 1 - abs(s2 - t2) / h3)
local_delta3 = beta * h3 * local_weight3 * (1 - 2 * P)
score3 = max(0, s2 + local_delta3)
```

The new term is exactly zero at or beyond distance `h3`. Missing feature
values turn only the third-stage term off. Support remains exactly NEXT214
support. Certificate cutoffs are endpoint-blind 1/16 and 15/16 inverted-CDF
quantiles over all finite combined discovery feature values, using the
existing NEXT216 normalization.

Frozen grids:

- third-stage local-width fractions `f`:
  `{1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1}`;
- amplitudes `beta`: `{1/4, 1/2, 1}`.

The full width grid is frozen because the SCIGEN protected-reject score excess
has 25th percentile `0.03190657811758944`, median
`0.2317450305799884`, and maximum `0.7461707105595379`; aggregate passage
requires rescuing at least 56 currently rejected protected rows, while the
same signed term raises rather than rescues severe-like rows.

For `K` NEXT233 eligible hypotheses, the complete catalogue contains one exact
NEXT232-base reproduction control and `21*K` eligible new candidates. No
feature pruning, family filtering, endpoint-dependent cutoff, source/fold
rule, coefficient fit, beam search, or manual override is allowed after
NEXT233.

Run the unchanged cross-source evaluator. Only eligible new candidates may be
selected. If any pass every discovery gate, select by the unchanged evaluator
rank and stop. Otherwise report the evaluator's best eligible AUC+SAFE record
without using BROAD residual to select it.

## NEXT235 frozen diagnostic and terminal rule

If NEXT234 has no all-gate candidate and at least one eligible AUC+SAFE/non-
BROAD candidate, reconstruct that exact population, verify its sorted-key
digest, reproduce all evaluator records, and compute the unchanged BROAD
threshold residual for every member. Rank by failed-constraint count,
normalized shortfall, and candidate key.

- If an eligible candidate passes every discovery gate, do not run this
  diagnostic; freeze a separate validation protocol before opening validation
  data.
- Otherwise publish the closest diagnostic record and close this stagewise
  branch regardless of whether it improves `(5, 0.12730274611313003)`.
- A non-BROAD result cannot be promoted, and validation and replication stay
  sealed.

## Additive implementation and verification

Create only:

- `src/next233_final_stagewise_margin_local_feature_audit.py`
- `tests/test_next233_final_stagewise_margin_local_feature_audit.py`
- `src/next234_final_stagewise_margin_local_search.py`
- `tests/test_next234_final_stagewise_margin_local_search.py`
- `src/next235_final_stagewise_margin_local_broad_diagnostic.py`
- `tests/test_next235_final_stagewise_margin_local_broad_diagnostic.py`
- formal directories `next233_final_stagewise_margin_local_feature_audit_v1`,
  `next234_final_stagewise_margin_local_search_v1`, and, if authorized,
  `next235_final_stagewise_margin_local_broad_diagnostic_v1` under the
  external data root.

Use TDD for exact base reconstruction, cohort identity, score, catalogue,
interfaces, provenance, terminal stopping behavior, and fail-closed input
checks. After formal execution, append exact results and hashes only to
`reports/2026-08-08-next115-next117-hcid-no-dft-search.md`. Run focused tests,
`py_compile`, the full pytest suite, independent manifest/hash checks,
`git diff --check`, trailing-whitespace and canonical-path checks, and
CodeGraph status. Do not modify `paper/`, `tex/`, `notes/`, `README.md`, or
`PREREG.md`.
