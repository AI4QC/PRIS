# NEXT236--NEXT238 Chemistry-Conditioned Certificate Search Plan

**Goal:** Test whether source transfer is limited by applying one absolute
feature scale across chemically different crystals, using an interpretable,
endpoint-blind chemistry-conditioned protection certificate in a fresh
one-term law rather than adding a fourth term to the closed NEXT229--NEXT235
stagewise branch.

**Architecture:** NEXT236 reconstructs the exact NEXT224 exploratory frontier,
builds frozen composition-only strata, and audits every raw x0 feature after
within-stratum endpoint-blind normalization. NEXT237 starts afresh from the
exact NEXT224 score and searches one chemistry-conditioned triangular
margin-local term over every eligible certificate and frozen parameter pair.
NEXT238 reproduces only the eligible AUC+SAFE/non-BROAD population and computes
the unchanged BROAD residual.

Date: 2026-08-09

Status: frozen after NEXT235 branch closure, before computing any conditioned
feature AUC, conditioned certificate cutoff, or NEXT237 candidate score.

## Why this is a different mechanism

The closed stagewise branch used globally normalized raw certificates and
successive score-local terms. Every one of the 484 final diagnostic candidates
still failed aggregate SCIGEN and folds 0--3 protected retention, despite
passing WyFormer and SCIGEN fold 4 in the closest record. This suggests a
transferable scale mismatch rather than a missing fourth additive term.

This branch changes the certificate itself. A raw geometric or bond-valence
quantity is interpreted relative to structures with similar composition-only
chemistry. It does not use source identity, fold identity, an outcome label,
or a learned energy surrogate at execution. It is not a continuation of the
NEXT235 formula: NEXT237 contains one proposed term on the exact NEXT224 base.

## Immutable no-DFT and data boundary

- Executable quantities may use composition and initial, unrelaxed geometry
  only.
- No DFT calculation or value; learned energy, force, or stress proxy; model
  or proxy potential; relaxed structure; trajectory; or physical relaxation
  may enter the executable law.
- Discovery outcomes are offline labels used only by the audit and unchanged
  evaluator.
- Only already opened SCIGEN and WyFormer discovery endpoints may be read.
- Validation and replication endpoints remain physically sealed unless a new
  candidate passes every frozen discovery gate.
- All scripts, tests, plans, results, and reports are additive. Do not modify
  `paper/`, `tex/`, `notes/`, `README.md`, or `PREREG.md`.

## Frozen NEXT224 base and comparison frontier

- NEXT224 base candidate-key digest:
  `3f87102463cc283bcb3e4d1c45e434e04c7f7d2d32167801b79d7db8035559e4`.
- NEXT224 diagnostic threshold: `0.1520033762332462`.
- NEXT224 failed constraints: `6`.
- NEXT224 normalized shortfall: `0.1461217358987499`.
- Support: exact NEXT214 support, `18,017` rows.
- Best overall diagnostic comparison after NEXT235:
  `(5, 0.12339543654931197)` under failed-count then normalized-shortfall
  lexicographic ordering.

The implementation must verify all published inputs and hashes and reproduce
the exact NEXT224 base score, support, evaluator record, and diagnostic before
use. The NEXT235 record is comparison-only and is never included as an
executable base term.

## NEXT236 frozen conditioned feature audit

Use the exact `242` sorted numeric, raw, non-identifier x0 features selected by
NEXT207/NEXT227, with name digest
`87a20f191ca47b6fb3e9f0255ae8d1e98bcf41e21991af3d290ff222c446f07c`.

The four conditioning axes are fixed composition-only quantities already in
that table:

1. `geom_electronegativity_mean`;
2. `geom_electronegativity_range`;
3. `geom_atomic_number_mean`;
4. `geom_covalent_radius_mean`.

For each axis, form four endpoint-blind strata with inverted-CDF 1/4, 1/2,
and 3/4 cut points over all finite combined discovery rows. Boundary values
go to the lower stratum. Missing conditioning values or target values make
only that certificate unavailable.

For every `(target feature, conditioning axis)` pair and each of
`protected_low` and `protected_high`, compute within each stratum the target's
1/16 and 15/16 inverted-CDF cutoffs over all finite combined discovery rows.
Map the target linearly to a bounded protection value in `[0,1]`, reversing
the orientation for `protected_low`. A stratum with non-finite or non-strict
cutoffs makes the certificate unavailable in that stratum. No outcome enters
strata or cutoffs.

Audit the resulting `242 * 4 * 2 = 1936` hypotheses only in the exact NEXT224
rejected extreme cohort:

```text
supported
AND finite NEXT224 score
AND score >= 0.1520033762332462
AND (endpoint <= 1 OR endpoint >= 2)
```

Reuse the exact source/fold gates: minimum coverage `0.90`, minimum protected
and severe count `20`, aggregate AUC `0.55`, macro-fold AUC `0.53`, and
worst-fold AUC `0.50`. Apply the opposite-direction veto separately for each
`(target, conditioner)` pair: exactly one direction must pass all gates.
Rank for reporting only by minimum worst-fold AUC, minimum aggregate AUC, mean
aggregate AUC, then hypothesis identity. NEXT237 must use all eligible
hypotheses.

NEXT236 searches and selects no formula. If none is eligible, close this
mechanism without running NEXT237.

## NEXT237 frozen fresh one-term grammar

Let `s` be exact NEXT224 score, `t` its frozen diagnostic threshold, `W` the
original NEXT214 repair width, and `P_c` one eligible chemistry-conditioned
certificate. Define

```text
h = f * W
local_weight = max(0, 1 - abs(s - t) / h)
local_delta = beta * h * local_weight * (1 - 2 * P_c)
score = max(0, s + local_delta)
```

The term is zero at and beyond distance `h`. A missing conditioned certificate
turns only the proposed term off. Support remains exactly NEXT214 support.

Frozen grids:

- width fractions `f`: `{1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1}`;
- amplitudes `beta`: `{1/4, 1/2, 1}`.

For `K` NEXT236 eligible hypotheses, the complete catalogue contains one
exact NEXT224 reproduction control and `21*K` eligible new candidates. No
feature pruning, source/fold rule, outcome-dependent stratum, cutoff refit,
beam search, or manual override is allowed after NEXT236.

Use the unchanged evaluator and its ranking. If any eligible candidate passes
every discovery gate, select it and stop for a separately frozen validation
protocol. Otherwise report the evaluator's best eligible AUC+SAFE record
without using BROAD residual for selection.

## NEXT238 frozen residual and stopping rule

If NEXT237 has no all-gate candidate and at least one eligible
AUC+SAFE/non-BROAD candidate, reproduce that exact population, verify its
sorted-key digest and evaluator records, and compute the unchanged BROAD
residual for every member. Rank by failed-constraint count, normalized
shortfall, and candidate key.

- Advance this mechanism only if the closest record strictly improves the
  overall NEXT235 comparison `(5, 0.12339543654931197)`.
- Otherwise close the chemistry-conditioned certificate branch.
- Any non-all-gate record remains exploratory and cannot open validation.
- Any continuation after a strict but non-all-gate improvement requires a new
  pre-outcome mechanism freeze; no inspected bin, axis, cutoff, or grid may be
  adjusted.

## Additive implementation and verification

Create only new NEXT236--NEXT238 scripts, tests, and formal external result
directories. Use TDD for stratum boundaries, endpoint-blind cutoff fitting,
missing behavior, exact base reproduction, candidate completeness,
provenance, and fail-closed interfaces. Append exact results only to the
independent report. Run focused tests, `py_compile`, full pytest, independent
hash checks, `git diff --check`, whitespace and forbidden-path checks, and
CodeGraph status.
