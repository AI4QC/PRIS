# NEXT42 converged Alexandria source qualification design

Date: 2026-08-03

## Purpose

Qualify, or reject, a new cross-source structural-change evaluation source before any further law search.  NEXT40 cannot serve this role because its latest-observed OMat24 frames are right-censored: every apparent protected case ended before step 10.  NEXT42 therefore asks a narrower question first: do the locally available Alexandria geometry-optimization paths contain a sufficiently large cohort whose initial geometry is a genuine pre-DFT, pre-relaxation candidate and whose final geometry is demonstrably converged?

This is a source audit, not a scientific-improvement claim and not a formula search.

## Non-negotiable law boundary

The eventual executable law may use only one supplied raw `x0`, composition, cell, coordinates, frozen elemental tables, and deterministic analytic geometry, Voronoi, bond-valence, electrostatic, linear-algebra, or symmetry operations.  It may not execute DFT, read DFT values, use a relaxed structure or trajectory, invoke an MLIP or learned energy/force/stress proxy, physically relax the input, or compare same-composition alternatives.

DFT forces and final structures are permitted only after predictions are frozen and only as offline evaluation labels.

## Fixed candidate source

- Alexandria PBE geometry-optimization paths dated 2025-07-02.
- Formal local shards: `pbe_0000.json.bz2` and `pbe_0001.json.bz2`, already frozen by NEXT18.
- Official `benchmarks_pbe.csv` is used only as an identity exclusion list.
- Alexandria final-database `location` fields are used only for provenance classification; energy, hull, force, stress, relaxed geometry, and other endpoint fields must not be emitted by the provenance stage.

The two path shards contain 20,000 unique material IDs before exclusions.  No additional trajectory shard may be added after endpoint inspection.

## Qualification gates

A row is primary-eligible only if all gates are satisfied:

1. The material ID is absent from the official Alexandria benchmark list.
2. The provenance is mapped uniquely to an explicitly allowed source family.
3. That family has documentary evidence that the first DFT ionic geometry is the generated, enumerated, or substituted candidate rather than an MLIP-relaxed geometry.
4. The path has at least one non-empty calculation and exact atom/species order is preserved from the first to final structure.
5. The final force label is finite and the maximum per-atom force norm is at most `0.005 eV/angstrom`, matching the published Alexandria convergence criterion.
6. The initial structure is non-empty, fully periodic, and serializable after removing every endpoint field.

Rows from `m3gnet/*`, `orbital/*`, or any other explicitly MLIP-pre-relaxed workflow are excluded from the primary cohort.  Unknown or ambiguous provenance is excluded rather than guessed.  Excluded rows may be counted diagnostically but may not affect threshold selection or primary claims.

## Leakage order

1. Hash and inventory the fixed source containers.
2. Map identities and provenance without emitting scientific labels.
3. Apply benchmark and source-family exclusions.
4. Freeze the complete eligible identity set and geometry-only initial frames.
5. Apply the unchanged NEXT23 B+E rule and Pauling controls to initial frames; freeze predictions and hashes.
6. Only then open final structures and DFT force convergence labels.
7. If the source passes qualification, evaluate the frozen rules with the unchanged NEXT23 endpoint and gates.
8. Only after that evaluation may a finite new formula family be designed.  Confirmation rows must never be used to refit the already frozen NEXT23 constants.

## Stop conditions

Stop without a formula search if any of the following holds:

- pre-DFT rawness cannot be demonstrated from provenance;
- no source family survives benchmark and MLIP exclusions;
- final convergence cannot be verified;
- atom identities do not align;
- the supported, protected, or changed class is too small for the predefined Wilson gates to be meaningful.

Failure of source qualification is an informative negative result and must be reported as such.  It does not justify weakening the no-DFT law boundary.

## Deliverables

- additive source-audit code and tests;
- immutable external artifacts with manifests and SHA-256 identities;
- a standalone NEXT42 report explaining qualification, exclusions, and limitations;
- no edits to `README.md`, `PREREG.md`, `paper/`, `notes/`, or `tex/`.
