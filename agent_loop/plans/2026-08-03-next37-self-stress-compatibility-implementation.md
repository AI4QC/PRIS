# NEXT37 Self-Stress Compatibility Implementation Plan

**Architecture:** Add a pure weighted compatibility-projection kernel and
structure/batch wrapper, followed by an exact 17-formula scanner/freezer. Reuse
the immutable NEXT19/NEXT20 graph/radius implementation and sealed NEXT36
comparators; record all dependency hashes.

## Task 1: Pure projection kernel (TDD)

- Create `tests/test_next37_self_stress_compatibility_features.py` first.
- Add `src/next37_self_stress_compatibility_features.py` after the red import.
- Test exact duplicate-row self-stress, pure load-generating mismatch,
  atomic-versus-affine decomposition, zero residual conventions, rank tolerance,
  finite schema, and rotation/permutation/scale/replication invariance.

## Task 2: Sealed feature batch

- Validate exact NEXT32 geometry and NEXT36 feature hashes.
- Rebuild only the frozen unweighted opposite-sign Voronoi graph from `x0`.
- Fail open unsupported charge/graph cases, copy only four fixed comparators,
  exclude `sid` and endpoint fields, and publish no-replace with all hashes.

## Task 3: Exact scanner/freezer (TDD)

- Create `tests/test_next37_self_stress_compatibility_rule.py` then source.
- Test exactly 10 terms, 7 pairs, 17 formulas, 85 rows, all SSCP high-risk
  directions, unchanged gates, zero-IQR local disabling, immutable replay,
  hash locking, no-overwrite, and label-free predictions.

## Task 4: Bounded execution

1. Run focused tests.
2. Publish and verify a 16-row smoke artifact.
3. Publish and verify 4,096 label-free development features.
4. Join the existing exposed endpoint once and publish exactly 85 rows.
5. Open confirmation only after all six gates pass.

## Task 5: Report and verification

- Add only `reports/2026-08-03-next37-self-stress-compatibility.md`.
- Run focused/full tests, verify all input/source/output/report hashes, check
  CodeGraph health and canonical-document diffs, and prove confirmation state.
