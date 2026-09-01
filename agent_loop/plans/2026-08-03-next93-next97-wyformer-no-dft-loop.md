# NEXT93--NEXT97 WyFormer no-DFT discovery loop

## Objective

Test whether a deterministic, analytic law evaluated only on the raw generated
WyFormer/DiffCSP++ structure can reject candidates that will either fail the
published DFT relaxation or remain far above the reference convex hull.  DFT
data are offline labels only.  No DFT value, CHGNet result, relaxed structure,
trajectory, learned potential, or same-composition alternative may enter the
law at execution time.

This is an additive branch.  Existing scripts, artifacts, reports, README
files, and manuscript sources are not replaced or edited.

## Official source identity

- Figshare article: `29094701`, WyFormer generated structures.
- Raw x0 file: file id `54711179`, the 9,999-structure
  `mp_20/WyckoffTransformer/DiffCSP++10k/data.csv.gz` sample.
- Offline DFT file: file id `54711188`, the 9,623 successful
  `CHGNet_free/DFT/data.csv.gz` records produced with
  `MPGGADoubleRelaxStaticMaker`.
- The less strict one-stage `DFT-GGA-relax-1` file is excluded.
- Individual downloads must match the MD5 values in the Figshare API response.

The Figshare README states that indices were permuted at the CHGNet step.
Consequently, published `material_id` values must never be used to pair raw and
DFT rows.

## NEXT93: provenance-safe pairing and physical routing

1. Decode each raw and DFT structure and construct an exact full-cell
   composition key from element occupancies.
2. Retain only composition keys occurring exactly once in the raw x0 file.
3. Require each retained key to occur either zero times or exactly once in the
   DFT-success file.  More than one DFT row is an error, not a tie-break.
4. A unique DFT match is a successful calculation.  A missing DFT match is a
   failed calculation because the official source says failed relaxations are
   considered unstable.
5. Freeze whole reduced-formula groups into discovery, internal validation,
   and internal replication using SHA-256 and the salt
   `NEXT93_WYFORMER_REDUCED_FORMULA_SPLIT_V1`.  Labels do not affect the split.
6. Publish raw x0 payloads separately from endpoint payloads.  Validation and
   replication endpoints are not arguments to feature or discovery runners.

The formal endpoint strata are fixed before endpoint distributions are read:

- protected: DFT succeeded and corrected energy above hull is at most
  `0.10 eV/atom`;
- middle: DFT succeeded and `0.10 < e_hull < 0.50 eV/atom`;
- severe: DFT failed or corrected energy above hull is at least
  `0.50 eV/atom`.

Middle rows are excluded from binary AUC and severe-precision calculations.

## NEXT94: x0-only feature freeze

Before any discovery endpoint is opened, compute the already frozen analytic
feature families and Pauling P2--P5 controls for every x0 in every partition.
The computation may use deterministic geometry, graph, Voronoi, bond-valence,
electrostatic, linear-algebra, spglib, and CrystalNN operations plus frozen
element tables.  Every feature family fails open independently.

The feature runner accepts raw x0 cohort paths only.  It has no endpoint path.

## NEXT95: finite discovery search

Open only the discovery endpoint.  Search a finite, manifest-recorded family
of short monotone formulas using discovery-derived robust transforms.  Require
the same fixed formula and threshold to pass all composition folds; correlated
proxy substitutions do not count as stable formula recovery.

Minimum discovery gates:

- supported coverage lower 95% bound >= 0.90;
- protected recall lower 95% bound >= 0.90;
- severe precision lower 95% bound >= 0.80;
- savings lower 95% bound >= 0.02;
- pooled protected-vs-severe AUC >= 0.75;
- macro crystal-system AUC >= 0.60;
- worst evaluable crystal-system AUC >= 0.55;
- at least five evaluable crystal systems;
- reject more severe rows than Pauling P2--P5 while attaining a higher severe
  precision lower bound.

If no candidate passes, stop and publish a negative report.  Validation and
replication remain unopened.

## NEXT96: freeze before validation

If and only if discovery passes, publish the exact formula, transforms,
threshold, missing-value policy, source hashes, and predictions for all three
partitions without opening validation or replication endpoints.

## NEXT97: one-shot internal validation

Open the validation endpoint exactly once and evaluate the frozen predictions
against the unchanged gates.  Any failed gate stops the branch.  The formula
and threshold may not change.  Replication is authorized only after a complete
validation pass and is otherwise physically unopened.

## Reporting rule

Write a new standalone report after the branch stops or validates.  Do not
modify canonical reports or manuscript sources until the user reviews that
report.
