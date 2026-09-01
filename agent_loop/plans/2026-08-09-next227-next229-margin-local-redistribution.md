# NEXT227--NEXT229 Margin-Local Redistribution Implementation Plan

**Goal:** Test whether a fully pre-DFT, decision-margin-local signed correction
can rescue the persistent SCIGEN protected false rejects at the strongest
NEXT224 diagnostic frontier without sacrificing WyFormer fold-4 savings.

**Architecture:** NEXT227 reconstructs the exact preregistered NEXT224 global
closest record and audits the complete raw x0 feature bank only inside its
rejected extreme cohort. NEXT228 uses every source-stable eligible hypothesis
from that audit in a complete frozen margin-local grammar. NEXT229 only
reproduces the exact eligible AUC+SAFE/non-BROAD population and computes the
unchanged BROAD residual.

**Tech Stack:** Python 3.11, NumPy, pandas/Parquet, the existing NeoPauling
cross-source evaluator, pytest, and SHA-256 manifests.

Date: 2026-08-09

Status: frozen after the published NEXT224--NEXT226 outcomes but before any
NEXT227 feature AUC or NEXT228 candidate score is computed.

## Scientific rationale and rejected alternatives

The strongest current result is the preauthorized NEXT224 diagnostic closest
record, with six failed constraints and normalized shortfall
`0.1461217358987499`. At its exact threshold `0.1520033762332462`, it rejects
231 protected SCIGEN structures and 421 total WyFormer fold-4 structures. The
lowest quartile of SCIGEN protected-reject score excess is
`0.02114152039688938`, while the median is much larger. Thus only a local
subset must be rescued to address the 60-structure aggregate SCIGEN deficit.

Three next mechanisms were considered:

1. further tuning of NEXT223 budgets or amplitudes, rejected because it would
   be post-outcome tuning of the same linear family;
2. agreement gating, already tested by NEXT225--NEXT226 and weaker than the
   NEXT224 frontier;
3. margin-local signed redistribution, selected because it changes only the
   decision neighborhood, can lower protected-like rows and raise severe-like
   rows with one source-agnostic rule, and leaves distant high-risk and
   low-risk structures unchanged.

The NEXT224 global closest record and threshold were selected by the
pre-outcome NEXT223--NEXT224 protocol. They are frozen exploratory inputs here,
not a validated or promoted law. Any claimed advancement must strictly improve
the exact NEXT224 tuple `(6, 0.1461217358987499)`.

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
- Preserve all existing scripts, results, plans, reports, and canonical
  artifacts. Add new files and append only to the independent report.

## Frozen NEXT224 base identity

- Base candidate-key digest:
  `3f87102463cc283bcb3e4d1c45e434e04c7f7d2d32167801b79d7db8035559e4`
  (the SHA-256 of the exact NEXT224 global-closest candidate key; the
  implementation must verify this before use).
- Base failed-constraint count: `6`.
- Base normalized shortfall: `0.1461217358987499`.
- Base decision threshold: `0.1520033762332462`.
- Base support: exact NEXT214 support, 18,017 rows.
- The base score must be reconstructed from the exact NEXT222 two-term path and
  the exact NEXT224 dual-evidence candidate; no coefficient is refitted.

The implementation must independently recompute the candidate-key digest from
the published diagnostic rather than trusting the literal above. If the
literal is incorrect, the stage must fail before any feature AUC is computed;
the plan may be corrected only as a pre-outcome identity fix.

## NEXT227 frozen feature audit

Select the exact 242 sorted numeric, raw, non-identifier x0 columns using the
existing NEXT207 schema policy. Their name digest must remain
`87a20f191ca47b6fb3e9f0255ae8d1e98bcf41e21991af3d290ff222c446f07c`.
Audit both `protected_high` and `protected_low` directions for every feature.

The cohort is fixed before feature values are ranked:

```text
supported
AND finite NEXT224 score
AND score >= 0.1520033762332462
AND (endpoint <= 1 OR endpoint >= 2)
```

Expected protected/severe counts are:

| cell | protected | severe |
| --- | ---: | ---: |
| SCIGEN all | 231 | 2895 |
| SCIGEN folds 0--4 | 47, 49, 45, 46, 44 | 601, 582, 572, 578, 562 |
| WyFormer all | 317 | 508 |
| WyFormer folds 0--4 | 80, 52, 65, 67, 53 | 95, 101, 112, 106, 94 |

Use the exact existing NEXT207 source/fold gates independently for both
sources: minimum coverage `0.90`, minimum protected and severe count `20`,
aggregate AUC `0.55`, macro-fold AUC `0.53`, and worst-fold AUC `0.50`.
Apply the exact opposite-direction veto: a feature is eligible only when
exactly one direction passes all gates. Rank only for reporting by minimum
worst-fold AUC, minimum aggregate AUC, mean aggregate AUC, then hypothesis
identity. The search must include every eligible hypothesis, not only the
reported top one.

NEXT227 searches and selects no formula. If no hypothesis is eligible, close
the branch without running NEXT228.

## NEXT228 frozen executable grammar

For exact NEXT224 score `s`, exact NEXT224 threshold `t`, original NEXT214
repair width `W`, an eligible bounded protection certificate `P`, local-width
fraction `f`, and amplitude `beta`, define

```text
h = f * W
local_weight = max(0, 1 - abs(s - t) / h)
local_delta = beta * h * local_weight * (1 - 2 * P)
score = max(0, s + local_delta)
```

The term lowers protected-like rows, raises severe-like rows, and is exactly
zero at or beyond distance `h` from the frozen threshold. Certificate cutoffs
are endpoint-blind 1/16 and 15/16 inverted-CDF quantiles over all finite
combined discovery feature values, using the existing NEXT216 normalization.

Frozen grids:

- local-width fractions `f`:
  `{1/64, 1/32, 1/16, 1/8, 1/4}`;
- amplitudes `beta`: `{1/4, 1/2, 1}`.

For `K` NEXT227 eligible hypotheses, the complete catalogue contains one exact
NEXT224 no-op control and `15*K` eligible new candidates. Missing proposed
feature values turn the local term off for that row. Support remains exactly
NEXT214 support. No feature pruning, family filtering, outcome-dependent
cutoff, source/fold rule, coefficient fit, beam search, or manual override is
allowed after NEXT227.

Run the unchanged cross-source evaluator. Only eligible new candidates may be
selected. If any pass every discovery gate, select by the unchanged evaluator
rank and stop. Otherwise report the evaluator's best eligible AUC+SAFE record
without using BROAD residual to select it.

## NEXT229 frozen residual diagnostic and stopping rule

If NEXT228 has no all-gate candidate and at least one eligible AUC+SAFE/non-
BROAD candidate, reconstruct that exact population, verify its sorted-key
digest, reproduce all evaluator records, and compute the unchanged BROAD
threshold residual for every member. Rank by failed-constraint count,
normalized shortfall, and candidate key.

- The branch advances only if the global closest record strictly improves
  `(6, 0.1461217358987499)`.
- If it does not, close the margin-local branch.
- If it improves but remains non-BROAD, any continuation requires a new
  pre-outcome freeze.
- If an eligible candidate passes every discovery gate, freeze a separate
  validation protocol before opening validation data.

Validation and replication stay sealed in every non-all-gate case.

## Additive implementation and verification

Create only:

- `src/next227_margin_local_feature_audit.py`
- `tests/test_next227_margin_local_feature_audit.py`
- `src/next228_margin_local_redistribution_search.py`
- `tests/test_next228_margin_local_redistribution_search.py`
- `src/next229_margin_local_broad_diagnostic.py`
- `tests/test_next229_margin_local_broad_diagnostic.py`
- formal directories `next227_margin_local_feature_audit_v1`,
  `next228_margin_local_redistribution_search_v1`, and, if authorized,
  `next229_margin_local_broad_diagnostic_v1` under the external data root.

Use TDD for audit, score, catalogue, interfaces, exact provenance, and
fail-closed behavior. After formal execution, append exact results and hashes
only to `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`. Run focused
tests, `py_compile`, the full pytest suite, independent manifest/hash checks,
`git diff --check`, trailing-whitespace and canonical-path checks, and
CodeGraph status. Do not modify `paper/`, `tex/`, `notes/`, `README.md`, or
`PREREG.md`.
