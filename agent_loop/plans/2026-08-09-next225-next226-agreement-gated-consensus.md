# NEXT225--NEXT226 Agreement-Gated Consensus Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task.

**Goal:** Test whether a fully pre-DFT correction that acts only when two
independent certificates agree can improve the NEXT222 cross-source
protection-versus-savings frontier without weakening any frozen discovery
gate.

**Architecture:** Reconstruct the exact NEXT222 two-term score rather than
promoting the outcome-selected NEXT224 diagnostic record. NEXT225 evaluates a
complete frozen unordered-pair agreement grammar over the exact 22
endpoint-blind NEXT215/NEXT216 certificates. NEXT226 only reproduces the exact
eligible AUC+SAFE/non-BROAD population and computes the unchanged BROAD
residual for every member.

**Tech Stack:** Python 3.11, NumPy, pandas/Parquet, the existing NeoPauling
cross-source evaluator, pytest, and SHA-256 manifests.

Date: 2026-08-09

Status: frozen before any NEXT225 candidate score is joined to discovery
outcomes.

## Scientific rationale and alternatives

NEXT224 found a strict but small residual improvement over NEXT222 while every
one of its 5147 diagnostic records still failed aggregate SCIGEN and SCIGEN
folds 0--3 on protected retention. The population then split between SCIGEN
fold-4 protection and WyFormer fold-4 savings. This indicates that the linear
dual-evidence allocation moves the frontier but applies substantial correction
when its two certificates conflict.

Three continuations were considered after that diagnostic:

1. promote or retune the NEXT224 residual winner, rejected because it was
   selected with BROAD outcomes and passed no complete discovery gate;
2. extend the amplitude or budget grids, rejected as post-outcome tuning of the
   same frozen family;
3. agreement-gated correction, selected because it changes the mechanism
   rather than the grid: only joint protected evidence produces protection
   relief, only joint severe evidence produces risk lift, and conflicting
   evidence produces little or no correction.

The selected mechanism is evaluated over the complete certificate universe.
No feature pair is chosen from NEXT224 outcomes.

## Immutable no-DFT and data boundary

- Executable quantities may use composition and initial, unrelaxed geometry
  only.
- The executable law must not use a DFT calculation or value; a learned
  energy, force, or stress proxy; a model or proxy potential; a relaxed
  structure; a trajectory; or physical relaxation.
- Discovery outcomes are offline labels only.
- Use only the already opened SCIGEN and WyFormer discovery endpoints.
- Internal validation and replication endpoints remain physically sealed
  unless an eligible new candidate passes every discovery gate.
- Preserve every existing script, result, report, plan, and canonical
  artifact. Add new scripts, tests, formal directories, and an independent
  report section only.

## Frozen inputs and certificate universe

- Base score/support: exact NEXT222 final two-term path and exact NEXT214
  support. The NEXT224 diagnostic winner is not a base term.
- Activation: original NEXT214 score in the exact lower-inclusive,
  upper-exclusive interval
  `[0.17470215862148156, 0.570892727856757)`.
- Certificate identities: exact 22 NEXT215 eligible hypotheses, digest
  `2e5000a319188a6191922a499b8151e28bb603ba06e70cff8750ec582e887b41`.
- Certificate cutoffs and directions: exact endpoint-blind NEXT216
  definitions; no refit or new cutoff.
- Pair universe: all 22 unordered pairs with replacement, in lexicographic
  hypothesis order; `22 * 23 / 2 = 253` pairs.
- Amplitude grid: exact NEXT220 grid
  `{1/64, 1/32, 1/16, 1/8, 1/4}`.
- Protection-budget fractions: exact grid `{1/4, 1/2, 3/4}`.
- Support remains identical to NEXT214. If either pair operand is missing, the
  proposed correction is zero for that row.

## Frozen executable grammar

For original NEXT214 score `s0`, exact NEXT222 cumulative delta `d222`, repair
width `W`, two bounded protection certificates `P_a` and `P_b`, amplitude
`beta`, and protection-budget fraction `lambda`, define on active rows

```text
risk_consensus = (1 - P_a) * (1 - P_b)
protection_consensus = P_a * P_b
pair_delta = 2 * beta * W * (
    (1 - lambda) * risk_consensus
    - lambda * protection_consensus
)
score = max(0, s0 + d222 + pair_delta)
```

The formula is symmetric in the two certificates, so only unordered pairs are
catalogued. At `lambda=1/2`, it becomes

```text
pair_delta = beta * W * (1 - P_a - P_b)
```

and exactly reproduces the corresponding equal-budget NEXT223 score. For a
diagonal pair `P_a=P_b=P`, this further reduces to the already tested
univariate score `beta * W * (1 - 2P)`. Therefore every equal-budget record is
a reproduction control and is ineligible as a new law.

The complete NEXT225 catalogue contains:

- one unchanged NEXT222 control;
- `253 * 5 * 3 = 3795` unordered-pair/amplitude/allocation records;
- `253 * 5 = 1265` equal-budget reproduction controls;
- among those controls, `231 * 5 = 1155` canonical off-diagonal NEXT223
  reproductions, 100 exact unused-feature NEXT222 depth-3 reproductions, and
  10 closed-form already-used-feature diagonal controls;
- `253 * 5 * 2 = 2530` eligible new asymmetric-budget candidates;
- `3796` total records.

No pair pruning, family filtering, outcome-dependent cutoff, coefficient
refit, source-specific rule, fold-specific rule, beam search, manual override,
or promotion of the NEXT224 diagnostic winner is allowed.

## Frozen evaluation and selection

Run the unchanged cross-source evaluator. Report source pooled/macro/worst AUC,
all 12 SAFE cells, BROAD, support, and complete gate counts separately for all
records, equal-budget controls, and eligible new candidates.

Only eligible new candidates may be selected. If one or more pass all
discovery gates, select deterministically using the unchanged evaluator order
and stop the discovery search. Otherwise select the unchanged evaluator's best
eligible AUC+SAFE record for reporting, without using BROAD residual to choose
it.

NEXT226 must reconstruct the exact eligible AUC+SAFE/non-BROAD population,
verify its sorted-key digest, and compute the unchanged BROAD threshold
diagnostic for every member. Rank by failed constraint count, normalized
shortfall sum, and candidate key. Compare the global closest eligible record
with the exact NEXT222 reference `(6, 0.1564570050830728)`. NEXT224 is an
outcome-informed diagnostic and is reported as a secondary comparison only;
it is not the frozen base and is not used as an acceptance threshold.

The branch is closed if no eligible candidate strictly improves the NEXT222
tuple. If it strictly improves but does not pass all gates, any continuation
requires another pre-outcome freeze. Validation and replication remain sealed
in either case. If an eligible candidate passes every discovery gate, freeze a
separate validation protocol before opening any validation endpoint.

## Task 1: NEXT225 score and catalogue helpers

**Files:**

- Create: `tests/test_next225_agreement_gated_consensus_search.py`
- Create: `src/next225_agreement_gated_consensus_search.py`

**Step 1: Write failing score and catalogue tests**

Test exact agreement products, symmetry under operand interchange, conflict
suppression, budget sensitivity, diagonal and off-diagonal equal-budget
identities, original-band activation, nonnegative flooring, unchanged support,
missing-pair term-off behavior, and the exact 3796/1265/2530 catalogue
partition.

**Step 2: Run focused tests and verify RED**

Run:

```bash
python -m pytest -q \
  tests/test_next225_agreement_gated_consensus_search.py
```

Expected: collection/import failure because the NEXT225 module does not exist.

**Step 3: Implement minimal score and catalogue helpers**

Implement `agreement_gated_consensus_score()` and
`build_agreement_candidate_specs()` with exact validation and deterministic
JSON keys. Expose constants for the frozen grids and candidate counts.

## Task 2: NEXT225 formal discovery search

**Files:**

- Modify: `tests/test_next225_agreement_gated_consensus_search.py`
- Modify: `src/next225_agreement_gated_consensus_search.py`
- Create formal directory:
  `$PRIS_ARCHIVE/next225_agreement_gated_consensus_search_v1`

Add interface and fail-closed tests for discovery-only endpoints, exact input
identity, output non-overwrite, complete control reproduction, and explicit
false flags for every forbidden mechanism. Reconstruct NEXT222, materialize
the complete frozen catalogue, run the unchanged evaluator, and publish
catalogue, evaluation, selected-formula, candidate Parquet, and manifest files
atomically.

## Task 3: NEXT226 exact BROAD residual diagnostic

**Files:**

- Create: `tests/test_next226_agreement_gated_broad_diagnostic.py`
- Create: `src/next226_agreement_gated_broad_diagnostic.py`
- Create formal directory:
  `$PRIS_ARCHIVE/next226_agreement_gated_broad_diagnostic_v1`

Test and implement exact eligible AUC+SAFE/non-BROAD filtering, sorted-key
digest, deterministic residual ordering, formal provenance, discovery-only
interfaces, and atomic publication. Reproduce NEXT225 evaluator records
exactly; search and select no new formula.

## Task 4: Report and verification

**Files:**

- Modify only the additive independent report:
  `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`

Append exact NEXT225--NEXT226 methods, results, limitations, formal paths, and
hashes. Do not modify `paper/`, `tex/`, `notes/`, `README.md`, or
`PREREG.md`. Run focused tests, `py_compile`, the full pytest suite,
independent manifest/hash checks, `git diff --check`, trailing-whitespace and
canonical-path status checks, and CodeGraph status.

Keep the overall goal active unless every discovery, validation, and
replication requirement is actually satisfied. Do not claim a replacement law
from discovery-only evidence.
