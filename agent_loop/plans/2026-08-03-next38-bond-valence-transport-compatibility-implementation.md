# NEXT38 Bond-Valence Transport Compatibility Implementation Plan

**Architecture:** Add a pure minimum-norm valence-correction/Jacobian-projection
kernel and a structure/batch wrapper, followed by an exact 17-formula
scanner/freezer. Reuse the immutable NEXT19 graph/valence implementation,
NEXT22 parameter policy, and sealed NEXT32/NEXT37 comparators; record every
dependency hash.

## Task 1: Pure correction and Jacobian kernel (TDD)

- Create `tests/test_next38_bond_valence_transport_compatibility_features.py`
  before source.
- Add `src/next38_bond_valence_transport_compatibility_features.py` after the
  red import.
- Test a balanced zero-correction graph, a fully compatible correction, a
  differential incompatibility, zero conventions, rank tolerance, finite
  schema, and rotation/atom-order/edge-order/exact-replication invariance.
- Verify cation-star prior sums and Jacobian row sums exactly within tolerance.

## Task 2: Sealed feature batch

- Validate exact NEXT32 geometry, NEXT32 feature, and NEXT37 feature hashes.
- Rebuild only the frozen opposite-sign Voronoi graph from `x0`.
- Resolve the frozen exact/fallback bond-valence parameters without label
  access, fail open unsupported graph/parameter/numerical cases, copy only the
  four fixed comparators, exclude identifiers other than `material_id` and
  geometry-only provenance, and publish no-replace with all hashes.

## Task 3: Exact scanner/freezer (TDD)

- Create `tests/test_next38_bond_valence_transport_compatibility_rule.py` before
  source.
- Test exactly 10 terms, 7 pairs, 17 formulae, 85 rows, all BVTC high-risk
  directions, unchanged gates, zero-IQR local disabling, immutable replay,
  hash locking, no-overwrite, and label-free predictions.

## Task 4: Bounded execution

1. Run focused tests.
2. Publish and verify a 16-row smoke artifact.
3. Publish and verify 4,096 label-free development features.
4. Join the existing exposed endpoint once and publish exactly 85 rows.
5. Open confirmation only after all six gates pass.

## Task 5: Report and verification

- Add only
  `reports/2026-08-03-next38-bond-valence-transport-compatibility.md`.
- Run focused/full tests, verify every input/source/output/report hash, check
  CodeGraph health and canonical-document diffs, and prove confirmation state.
