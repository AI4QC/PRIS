# NEXT123--NEXT125 MHCR implementation plan

> Use test-driven development with the mandated interpreter
> `python`.  Work additively in the
> current checkout, do not commit, and do not modify canonical documents.

## Goal

Implement, materialize, freeze, and evaluate the no-DFT multiscale Hall-contact
robustness certificate defined in the companion NEXT123 design.

## Task 1: graph-kernel RED tests

Create `tests/test_next123_multiscale_hall_contact_robustness.py` before the
source exists.  Cover a one-edge threshold oracle, exhaustive small-subset
oracles, threshold monotonicity, duplicate maximum-strength reduction,
permutation, common-charge scaling, exact graph replication, and invalid input.
Run the test and record the expected missing-module failure.

## Task 2: minimal graph implementation

Create `src/next123_multiscale_hall_contact_robustness.py`.  Validate signed
intervals and weighted oriented endpoints, solve only the primary exact closure
LP for the full and four thresholded graphs, normalize incremental deficits, and
return the frozen eight-feature schema.  Make Task 1 green before adding the
structure wrapper.

## Task 3: raw-structure wrapper

Add RED tests proving that the selected sign pattern and full endpoint set match
NEXT109/NEXT115, Voronoi strengths are normalized per origin, repeated calls are
deterministic, and a real pymatgen supercell remains supported.  Implement the
wrapper by reusing the frozen catalogue, sign-pattern ranking, and neighbor
finder.  Fail closed if the weighted and legacy endpoint sets differ.

## Task 4: NEXT124 label-free builder

Create focused tests and an additive multiprocessing builder.  Reuse the
NEXT116 frozen raw-structure catalogue and row identities.  Formal outputs go
only to
`$PRIS_ARCHIVE/next124_cross_source_mhcr_features_v1`.
Verify row counts, support accounting, finite ranges, monotonicity, source
hashes, and manifests.  No endpoint table may be read by the builder.

## Task 5: label-free feature and search freeze

Audit the eight features without outcomes.  Retain at most four terms using only
the rules frozen in the design.  Then select the frozen NEXT122 base frontier,
enumerate base/single/pair candidates on the fixed coefficient grid, hash the
complete universe, and write the NEXT125 freeze before rereading endpoints.

## Task 6: NEXT125 discovery-only evaluation

Reconstruct the complete label-free term table, reproduce every imported base,
evaluate only the fixed candidates under unchanged gates, publish atomically,
and independently verify all hashes.  Do not open validation or replication
unless every discovery gate passes.

## Task 7: standalone reporting and verification

Append the NEXT123--125 method and formal outcome to the existing standalone
NEXT115+ report or create another additive report if it becomes materially
large.  Run focused regressions, compile all new sources, verify formal hashes,
check CodeGraph freshness, and confirm no protected canonical path changed.
