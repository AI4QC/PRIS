# NEXT24 Cross-Source Generated-Structure Screening Design

## Purpose

NEXT24 tests whether the already frozen NEXT23 analytic law transports from
WBM substitution structures to structures emitted by a crystal generator.  It
is a strict external-transport test: the formula, robust centers/scales,
threshold, feature definitions, missing-value policy, and Pauling controls are
all inherited unchanged.  No NEXT24 data may refit or select them.

The executable screen reads one raw unrelaxed structure at a time.  It may use
only composition, lattice, coordinates, tabulated elemental properties, and
deterministic analytic geometry, bond-valence, electrostatics, or linear
algebra.  It may not call DFT, a machine-learned interatomic potential, an
energy proxy, a relaxation, a trajectory, or same-composition alternatives.
DFT is permitted only after prediction freeze as an offline endpoint.

NEXT24 is additive.  It preserves every NEXT12 and NEXT23 artifact and does
not modify `paper/`, existing reports, README files, or canonical scientific
documents.

## Formal source cohort

The first transport cohort is the existing 256-attempt SSAGEN prospective x0
artifact at `outputs/20260802_next12_ssagen_prospective_x0/`.  Its upstream
manifest states that all attempts were retained, all 256 generated, no
energy/force model was called, and labels were not opened.  The generator is a
real CIVAE Transformer checkpoint trained on only 500 structures; therefore
this cohort is a useful weak-generator stress test but not representative of
modern production generators.

The sanitizer projects only `sid`, `generator`, `generation_status`, `natoms`,
`formula`, `geometry_sha256`, and `archive_member`.  It verifies upstream file
hashes, exact ID/frame coverage, per-frame hashes, full periodicity, finite
coordinates/cells, atom counts, and formula agreement.  It rewrites each frame
to the repository's canonical geometry-only representation without changing
numeric species, coordinates, or cell values.  The output metadata contains
only `material_id`, `rk`, `formula`, `natoms`, and
`input_role=unrelaxed_x0_geometry_only`.

## Frozen screen

The screen uses the immutable NEXT23 rule:

`R = z(B) + z(E)`

where `B` is `voronoi_q0__sivr_cell_anisotropy`, `E` is
`scbv_vector_asymmetry_rms`, and every `z` uses the median, IQR, and direction
stored in the frozen NEXT23 rule JSON.  A supported structure is rejected when
`R >= 2.0327814658380157`.  Missing or unsupported analytic terms fail open.

Only the NEXT20 SIVR and NEXT22 SCBVE builders are executed, because the
frozen law does not use a Madelung term.  The applier validates the original
NEXT23 law and manifest hashes, validates feature-table provenance and exact
cohort ID coverage, accepts no endpoint path, and seals predictions before any
DFT artifact is opened.

The unchanged repository Pauling 2--5 operational controls are evaluated on
the identical canonical x0 archive with the identical fail-open policy.  They
are comparators, not inputs to the new law.

## DFT endpoint and temporal boundary

The predeclared DFT protocol is the already published NEXT12 queue containing
an x0 static calculation and a full-cell PBE relaxation for every generated
attempt.  The local checkout currently contains the queue but no completed
VASP results.  NEXT24 therefore has two strictly separated states:

1. **prediction-only:** canonical x0, analytic features, frozen decisions, and
   Pauling controls are hash-sealed; no endpoint or relaxed structure is read;
2. **post-DFT evaluation:** only after matching VASP outputs exist, an additive
   evaluator may read convergence, x0/final energies, and final structures to
   compute relaxation failure and structural-reorganization endpoints without
   changing the law.

For the future endpoint, DFT failure/non-convergence and large x0-to-final
structural change are primary screening targets.  Convex-hull stability is a
separate thermodynamic target and requires compatible competing-phase data; it
must not be inferred from relaxation change alone.

An independently downloadable public generated-x0/DFT-final paired dataset may
be added as a second source only if both sides are traceably paired.  Relaxed
structures alone, MatterSim relaxations, aggregate benchmark metrics, and
unpaired paper CIFs are ineligible.

## Claims and failure policy

Before DFT endpoints exist, NEXT24 may report coverage, score distribution,
rejection fraction, chemistry/size strata, and disagreement with Pauling.  It
may not report precision, recall, stability prediction, superiority, or DFT
savings.

All publications are no-replace and include input/output/source hashes.  A
changed upstream hash, duplicate ID, unsafe ZIP member, non-canonical geometry,
label-like metadata, formula/atom-count mismatch, mutated coordinate/cell,
invalid frozen law, or source mutation before publication is a hard error.
Per-structure analytic unsupported cases are recorded and fail open.

