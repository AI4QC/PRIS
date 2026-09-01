# NEXT60 ODAC23 robust scaffold-response endpoint

## Motivation

The prior exact-x0 endpoint assigns one invisible CO2/H2O placement to a law
that receives the framework only.  Train-only audit examples showed large
within-framework response variation across adsorbate placements, while the
official `fid` field was not a useful global identifier and `supercell`
distinguished genuinely different framework cells.  NEXT60 creates a new
additive endpoint artifact; all earlier endpoints and results remain unchanged.

## Scaffold identity

For each official train record, remove atoms with `tags != 0` and define a
scaffold-condition key from:

- framework-name prefix before `_w_`;
- exact integer `supercell` triplet;
- framework atom count and exact ordered atomic-number bytes;
- the full 3x3 cell rounded to a fixed 0.001 angstrom grid.

Initial positions and adsorbate identity/count/placement are deliberately not
part of this robustness identity.  The cell grid is fixed before the full train
scan.  A selected NEXT54 x0 is linked to one key by its exact NEXT53 geometry
hash; conflicting links fail closed.

## Translation-aligned offline DFT response

For framework atoms only, calculate fractional relaxed-minus-initial
displacements and wrap them by minimum image.  Estimate the periodic common
translation by: circular mean per fractional component; unwrap every atom
around that circular center; then take the componentwise median.  Subtract that
translation, wrap residuals again, convert through the fixed cell, and calculate
the residual displacement p95.

For each scaffold key, aggregate all adsorbate configurations by the median of
their aligned p95.  Retain raw/aligned quartiles and common-translation norms as
offline diagnostics, never executable inputs.  Require at least four official
train relaxations per key; lower-count keys are excluded from endpoint
development before label inspection.

- protected: robust aligned p95 <= 0.05 angstrom;
- severe: robust aligned p95 >= 0.20 angstrom.

Thresholds are unchanged from NEXT53 and frozen before the full scan.

## Firewall and next search

Route selected robust labels into the existing framework-isolated discovery,
internal-validation, and internal-replication roles without printing endpoint
summaries for locked roles.  NEXT58 features are already frozen.  The new
discovery search must reuse the exact NEXT59 finite catalogue and gates.  A
failure cannot open a lockbox; a passing formula is sealed before internal
validation is opened once.
