# NEXT87 SCIGEN sparse-search implementation protocol

Date: 2026-08-03

Status: post-discovery-opening implementation specification. This document
does not claim a pre-endpoint freeze. It makes the already frozen NEXT83--89
design executable without changing that design, the NEXT86 term catalogue, or
any gate. Internal-validation and internal-replication endpoints remain
physically unopened.

> **For Codex:** use `superpowers:executing-plans`,
> `superpowers:test-driven-development`, and
> `superpowers:verification-before-completion` while executing this protocol.

## Goal

Search the finite NEXT86 catalogue for an explicit one-to-three-term SCIGEN
pre-DFT law, freeze predictions before any internal endpoint is opened, and
stop without rescue if the precommitted discovery gates fail.

## Exact staged catalogue

1. Evaluate all 86 eligible one-term formulas with unit weight.
2. Rank singles using discovery-only threshold metrics and pooled extreme AUC.
   Retain at most 16 terms and at most three terms from one NEXT86 physics
   group. Deterministic ties use `term_id`.
3. Evaluate all unordered pairs from the 16-term shortlist. In lexicographic
   term order the first weight is 1 and the second is each member of
   `{0.25, 0.5, 1, 2, 4}`.
4. Use the first 12 members of the same single-term shortlist for triples.
   In lexicographic term order the first weight is 1 and the other two range
   independently over `{0.25, 0.5, 1, 2, 4}`.
5. Every score is exactly the frozen one-sided hinge sum. No sign, transform,
   center, scale, term, subgroup, or missing-value policy may be learned from
   an endpoint outside discovery.

## Threshold and metric semantics

- A row is supported only when every required raw term is finite and its
  frozen transform is valid; unsupported rows are `KEEP`.
- Candidate thresholds are distinct supported discovery scores and rejection
  uses `score >= threshold`.
- Threshold selection maximizes the one-sided 95% Wilson lower bound of
  severe-rejection precision subject to the frozen protected-recall and
  total-savings lower-bound gates. Ties prefer more rejected severe rows,
  higher savings lower bound, then a higher threshold.
- Precision is `rejected severe / rejected (severe + protected)`; middle rows
  do not enter its numerator or denominator. Savings uses all rows.
- Pooled and lattice AUC use only supported protected/severe rows.

## Five-fold stability

Assign whole `reduced_formula` groups with
`sha256("NEXT87_SCIGEN_GROUP_FOLD_V1|" + reduced_formula) % 5`.
For each held-out fold, repeat candidate ranking on the other four folds with
training-only thresholds. The exact unordered term-id list of the full-data
candidate must be the winning term list in at least four of five searches.
All terms must also attain the frozen support-coverage lower bound in every
held-out fold. For the final candidate, train a threshold on four folds and
apply it once to the held-out fold; every raw held-out precision must be at
least 0.70 and protected recall at least 0.93.

## Pauling comparison and publication

`pauling_p2_p5_decision == "REJECT"` is the only Pauling rejection. `ABSTAIN`
is unsupported and never converted to rejection. NEXT87 must reject more
severe discovery rows and have a higher severe-precision Wilson lower bound.

Publish without overwrite:

- the complete candidate search record;
- fold winners and out-of-fold diagnostics;
- selected formula and discovery evaluation;
- only after a discovery pass, frozen feature-only predictions for internal
  validation and internal replication;
- a manifest containing every input/output/source hash and explicit endpoint
  access flags.

The runner accepts only the discovery endpoint directory. It has no validation
or replication endpoint argument. Those endpoint files may be opened only by
subsequent one-shot evaluators after the frozen prediction gate authorizes it.

