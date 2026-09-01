# NEXT25 OMatG Composition-Only Blind CSP Screening Design

## Purpose

NEXT25 tests whether the already frozen NEXT23 analytic `B+E` law transports
to a modern public crystal-structure-prediction generator.  It is a no-refit,
cross-source test.  The formula, feature definitions, robust centers/scales,
threshold, and fail-open policy remain immutable.  NEXT25 data may not select
or modify any of them.

The executable screen reads one raw, unrelaxed generated structure at a time.
It may use only composition, lattice, coordinates, tabulated elemental
properties, deterministic analytic geometry, bond valence, electrostatics,
and linear algebra.  It may not use DFT, an ML interatomic potential, an
energy proxy, a physical relaxation, a trajectory, a DFT reference, or a
same-composition alternative.  OMatG flow integration is the generator
forward pass, not a physical energy relaxation.

NEXT25 is additive.  It preserves all NEXT23/NEXT24 scripts and outputs and
does not modify `paper/`, any existing report, README, or canonical document.

## Frozen public sources

The generator source is official OMatG v1.2.1 at Git commit
`fcb9ba2c2cfd70505b0f142a5b3c44944d78e7f0`.  The model source is the official
Hugging Face repository `OMatG/MP-20-CSP` at revision
`87dcc2a222f849f4f3c381a8cfa47ede0971d364`.  NEXT25 uses
`Linear-ODE/checkpoint.ckpt` and its matching `train.yaml`, because the
official model card identifies the Linear-ODE checkpoints as the best released
variant for match rate and mean RMSD.  Exact downloaded file hashes are sealed
before generation.

The composition source is the `train.lmdb`, `val.lmdb`, and `test.lmdb` MP-20
snapshot shipped by that OMatG source revision.  Candidate selection may read
only `atomic_numbers` and row indices from these LMDB records.  It must not
read `pos`, `cell`, `band_gap`, or `ids` values.  Whole-file hashes are allowed
for provenance because hashing does not deserialize or expose record fields.

## Composition-only cohort

Eligibility is decided without geometry or property labels:

1. the test row contains 2--20 atoms;
2. its reduced formula occurs exactly once in the test split;
3. that reduced formula does not occur in the train or validation split;
4. no composition, oxidation-state, law-support, generator-output, or endpoint
   validity filter is applied.

Eligible rows are ordered by SHA-256 of a frozen public salt, source index,
full integer composition, and reduced formula.  The first 512 rows form the
cohort.  The published selection table contains only a new `material_id`, the
source row index, full and reduced formulas, atomic numbers, atom count, hash
rank, and `input_role=composition_only`.  It contains no coordinates, cell,
energy, band gap, material identifier, or endpoint.

The official OMatG LMDB schema is reproduced with dummy cubic cells and dummy
coordinates solely to carry the exact species list required by the predictor.
The generator receives one exact full composition per selected row, in the
published order, with one repeat.  Train and validation dataset arguments in
the prediction-only runtime also point to the composition-only LMDB, so the
runtime cannot accidentally open MP-20 test reference geometries.

## Frozen generation

Generation uses the official `Linear-ODE` checkpoint and its 210-step
configuration, with fixed seed `250803`, one output per input composition, no
post-generation validity filtering, and no energy/force model.  The formal run
uses CPU with eight fixed compute threads so it does not contend with unrelated
GPU training; batch size, hardware, and software environment are recorded.  A
short non-cohort smoke test may be used only to validate the runtime.

The raw extended-XYZ output is sanitized into the repository's canonical
geometry-only representation.  The sanitizer verifies exact row count and
order, full composition, atom count, periodic cell, finite values, and
per-frame hashes.  Every successful generated frame is retained, including
geometrically poor structures.  Generator failure is an explicit unsupported
row and fails open.

## Frozen analytic screen and Pauling controls

The screen is the immutable NEXT23 rule

`R = z(B) + z(E)`

where `B` is `voronoi_q0__sivr_cell_anisotropy`, `E` is
`scbv_vector_asymmetry_rms`, and all centers/scales come from the frozen
NEXT23 development artifact.  A supported structure is rejected when
`R >= 2.0327814658380157`; unsupported rows fail open.

Only the SIVR and SCBVE feature builders are run.  The unchanged repository
Pauling 2--5 operational controls are evaluated on the identical canonical
generated archive.  Analytic predictions, their manifests, and all source
hashes must be sealed before any reference geometry is extracted.

## One-shot DFT-reference endpoint

After prediction freeze, an endpoint builder may reopen only the 512 selected
test records and extract their MP-20 reference structures.  Generated and
reference structures are paired by the pre-frozen source row index and exact
full composition.  The primary endpoint is one-to-one Pymatgen
`StructureMatcher` agreement using the OMatG/CDVAE/DiffCSP benchmark
tolerances `ltol=0.3`, `stol=0.5`, and `angle_tol=10 degrees`.  Non-matches
receive corrected RMSD 0.5; matches retain their normalized RMSD.

The primary screening gates, all evaluated with one-sided 95% Wilson lower
bounds where applicable, are:

- analytic support coverage at least 0.90;
- keep recall among DFT-reference matches at least 0.95;
- non-match precision among rejected structures at least 0.90;
- total rejection/DFT-saving fraction at least 0.10.

The same gates are computed independently for each fixed Pauling control.
Risk-score AUC for non-match, rank correlation with corrected RMSD, atom-count
and chemistry subgroups, disagreement tables, and calibration curves are
secondary diagnostics.  No threshold or formula may be revisited after the
endpoint is opened.

## Claim boundary

A reference match is evidence that the generator recovered the unique MP-20
benchmark structure for that composition.  A non-match is not proof of
thermodynamic instability: the generated structure may be a different stable
or metastable polymorph, and the benchmark may not enumerate every polymorph.
The unique, train/validation-absent reduced-formula filter reduces index and
split ambiguity but does not remove that scientific limitation.

Therefore NEXT25 can establish transport on a DFT-reference CSP endpoint and
can test whether the law protects benchmark matches while rejecting
non-matches.  It cannot by itself establish convex-hull stability, positive
phonons, synthesizability, or universal pre-DFT screening performance.  Those
remain future endpoints requiring paired DFT calculations or traceable public
DFT data.

Any hash mismatch, duplicate selection, geometry/property field access during
selection, reference access before prediction freeze, order/composition
mismatch, non-finite generated geometry, changed law, output replacement, or
post-label refit is a hard error.  Per-structure analytic unsupported cases
remain explicit and fail open.
