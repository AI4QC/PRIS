# NEXT34 Analytic Electrostatic-Field Imbalance Implementation Plan

**Architecture:** Add a pure analytic kernel and sealed geometry-only batch
builder, then a separate bounded development scanner/freezer. Reuse the
hash-locked NEXT32 cohort and NEXT33 features without modifying their files.
The shared research checkout is largely untracked, so work additively in place;
do not commit, reset, clean, overwrite, or edit canonical documents.

## Task 1: Analytic field kernel with TDD

- Create `tests/test_next34_analytic_field_features.py`.
- Create `src/next34_analytic_field_features.py`.
- First test exact schema and forbidden-token absence, fail-open behavior for
  invalid/non-neutral charges, translation/permutation/rotation/wrapping and
  primitive/supercell invariance, common charge-scale invariance, and increased
  field imbalance after a controlled NaCl displacement.
- Implement only the frozen Ewald field normalization and tensor statistic.
- Do not expose energy values or modify the input structure.

## Task 2: Sealed batch builder

- Extend the same test file with a tiny geometry-only ZIP, metadata, cohort
  manifest, and sealed NEXT33 feature fixture.
- Require exact protocol and SHA-256 matches, identity equality, no endpoint or
  `sid` fields, no calculator/attached result arrays, no-replace publication,
  source hashes, and explicit fail-open counts by valence policy/reason.
- Reuse `next11_geometry_only_frames._load_archive_only`, the NEXT19 valence
  cascade, and NEXT33 manifest conventions by import; do not edit those modules.

## Task 3: Frozen rule scanner/freezer with TDD

- Create `tests/test_next34_analytic_field_rule.py`.
- Create `src/next34_analytic_field_rule.py`.
- Test the exact 26 formula catalogue, 130 scan rows, high-risk AEFI directions,
  fixed fractions, unchanged six gates, deterministic robust-z/threshold ties,
  fail-open unsupported rows, no promotion when any gate fails, no-replace
  freezing, and label-free replay of a frozen rule.
- Reuse NEXT32 metric/Wilson helpers and join only the already exposed endpoint
  table inside the development scanner.

## Task 4: Real exposed-development execution

1. Run all focused NEXT34 tests in the `newpauling` conda environment.
2. Build a 16-row label-free smoke artifact and verify invariances/support.
3. Build the no-replace 4,096-row feature artifact from the exact NEXT32 cohort
   and sealed NEXT33 feature table; record wall time, counts, source hashes, and
   output SHA-256.
4. Verify the feature artifact before joining development endpoints.
5. Execute the exact 130-row frozen scan once and publish its manifest.
6. If and only if a candidate passes all gates, freeze and apply it before
   opening one confirmation source. Otherwise stop the confirmation branch.

## Task 5: Report and verification

- Write only a new standalone `reports/2026-08-03-next34-analytic-field.md`.
- Report the strongest failed or promoted candidates without post-hoc formula
  changes, compare with NEXT33/NEXT32 and same-cohort Pauling controls, and state
  the exact claim boundary.
- Run focused tests and the full `conda run -n newpauling python -m pytest -q`.
- Rehash every NEXT34 input, output, manifest source, and the report; check
  CodeGraph health and confirm no canonical paper/report was modified.
