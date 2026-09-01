# NEXT36 Weighted Charge-Spectrum Implementation Plan

**Architecture:** Add a pure reciprocal charge-spectrum kernel plus a sealed
batch builder, followed by a separate exact 17-formula scanner/freezer. Reuse
immutable NEXT19/NEXT35 code by import and record dependency hashes; do not edit
their sealed sources or artifacts.

## Task 1: Pure spectrum kernel (TDD)

- Create `tests/test_next36_charge_spectrum_features.py` before source.
- Add `src/next36_charge_spectrum_features.py` after the red import.
- Test neutrality validation, exact schema, finite outputs, translation/wrap,
  rotation, atom permutation, uniform scale, common charge-amplitude, and exact
  supercell invariance.
- Test that a deliberately long-wavelength separated charge decoration has
  larger long-scale Gaussian spectral weight than an alternating decoration.

## Task 2: Sealed label-free batch

- Validate the exact NEXT32 cohort and NEXT35 feature hashes.
- Infer neutral analytic charges from composition only; never read `sid` or any
  endpoint field.
- Copy only the four frozen comparators, fail open unsupported structures,
  record source/input/output hashes and counts, and publish no-replace.

## Task 3: Exact scanner/freezer (TDD)

- Create `tests/test_next36_charge_spectrum_rule.py` then
  `src/next36_charge_spectrum_rule.py`.
- Test exactly 10 terms, 7 pairs, 17 formulas and 85 rows, high-risk CSF
  directions, unchanged six gates, zero-IQR local disabling, immutable replay,
  hash locking, no-overwrite, and label-free predictions.

## Task 4: One bounded development execution

1. Run focused tests.
2. Publish and verify a 16-row label-free smoke artifact.
3. Publish and verify all 4,096 development features before endpoint access.
4. Open the already exposed endpoints once and publish exactly 85 scan rows.
5. Open confirmation only if all six gates pass; otherwise stop the branch.

## Task 5: Report and verification

- Add only `reports/2026-08-03-next36-charge-spectrum.md`.
- Run focused and full repository tests, verify all hashes, check CodeGraph,
  confirm canonical documents are unchanged, and prove confirmation remains
  unopened unless promotion occurred.
