# NEXT35 Coulomb--Steric Balance Implementation Plan

**Architecture:** Add one pure vector-balance kernel plus structure wrapper and
sealed batch builder, followed by a separate exact 18-formula scanner/freezer.
Reuse immutable NEXT33/NEXT34 code by import and record dependency hashes; never
edit their sealed sources or artifacts.

## Task 1: Pure balance kernel (TDD)

- Create `tests/test_next35_coulomb_steric_balance_features.py` first.
- Create `src/next35_coulomb_steric_balance_features.py` after the red import.
- Test exact-opposition zero risk, aligned/one-field-active high risk, invariance
  to independent positive rescaling of either field, rotation/permutation/
  replication invariance, deterministic q95, finite zero conventions, and exact
  feature schema with no forbidden DFT/model tokens.

## Task 2: Structure wrapper and sealed batch

- Test translation/wrapping/rotation/supercell invariance on perturbed NaCl and
  fail-open invalid valence/geometry cases.
- Compute the Ewald vector without emitting energy, and the rep12 pair vectors
  from the supplied x0 without coordinate modification.
- Validate exact NEXT32 cohort and NEXT34 feature hashes, copy only four frozen
  comparator columns, exclude `sid`/endpoint fields, record valence/failure
  counts and executed-source hashes, and publish no-replace.

## Task 3: Frozen scanner/freezer (TDD)

- Create `tests/test_next35_coulomb_steric_balance_rule.py` then
  `src/next35_coulomb_steric_balance_rule.py`.
- Test exactly 11 terms, 7 pairs, 18 formulas, 90 rows, all ACSB high-risk
  directions, unchanged six gates, deterministic ties, unsupported fail-open,
  immutable frozen constants, hash locking, and label-free replay.

## Task 4: Execute exposed development once

1. Run focused tests in the `newpauling` environment.
2. Publish and verify a 16-row label-free smoke artifact.
3. Publish and verify 4,096-row features while endpoint fields remain unopened.
4. Join the already exposed endpoints once and publish the exact 90-row scan.
5. Open confirmation only if all six gates pass; otherwise stop that branch.

## Task 5: Report and verify

- Add `reports/2026-08-03-next35-coulomb-steric-balance.md` only.
- Run focused and full repository tests, rehash every input/output/source/report,
  check CodeGraph health, and confirm canonical documents and confirmation
  sources remain untouched.
