# NEXT23 Analytic Relaxation-Change Screening Design

## Purpose

NEXT23 tests whether a law evaluated independently on one raw, unrelaxed
structure can identify structures that will reorganize strongly during a
subsequent DFT relaxation.  It does **not** claim to predict convex-hull
thermodynamic stability.  That distinction is essential: hull stability
depends on competing phases, whereas structural reorganization is a property
that may leave a signal in the candidate geometry itself.

The executable law may read only the unrelaxed lattice, coordinates,
composition, tabulated elemental properties, and deterministic analytic
geometry/electrostatics/linear-algebra constructions.  It may not read DFT
energies, relaxed structures, forces, stresses, ML potentials, proxy energies,
relaxation trajectories, or same-composition alternatives.  DFT-derived data
appear only as offline development and frozen blind-evaluation labels.

All NEXT19--NEXT22 sources and results remain unchanged.  NEXT23 is additive.
It must not modify `paper/`, existing reports, or canonical documentation.

## Endpoint and claims

The offline endpoint is
`site_stats_fingerprint_init_final_norm_diff`, the L2 distance between the WBM
initial- and final-structure SiteStats fingerprints.  Larger values mean more
structural reorganization.  The upstream benchmark does not define a universal
physical cutoff, so NEXT23 predeclares operational tiers rather than calling
any one threshold a fundamental constant:

- protected low-change structure: endpoint `<= 0.10`;
- changed structure: endpoint `> 0.10`;
- substantial change: endpoint `>= 0.20`;
- severe change: endpoint `>= 0.50`.

The primary claim, if supported, is limited to safe pre-screening of WBM-like
unrelaxed structures for relaxation-change risk.  It is not evidence of
formation energy, convex-hull stability, synthesizability, or a complete
replacement for DFT.

## Cohorts and ordering

The existing NEXT14 WBM sample of 2,048 structures is the exposed development
cohort.  Its labels have already been inspected and it can never be called a
holdout again.

The blind cohort contains 8,192 WBM test structures with 2--12 atoms selected
by ascending SHA-256 of a new fixed salt and material ID after excluding every
development material ID.  Selection reads only the official test-ID table and
initial-geometry archive.  The selector publishes a no-replace directory with
metadata, canonical geometry frames, input hashes, source hashes, exclusion
hashes, and `labels_opened=false`.

The mandatory temporal order is:

1. freeze the finite formula catalogue and statistical gates in source/tests;
2. use development labels to choose exactly one formula and threshold;
3. publish a frozen-law JSON with feature names, directions, robust centers,
   robust scales, threshold, hashes, and `blind_labels_opened=false`;
4. freeze the disjoint geometry-only cohort;
5. compute analytic features and predictions without label access;
6. publish prediction hashes;
7. only then join the blind endpoint labels once and evaluate without refit.

## Frozen candidate catalogue

Every base term is an already implemented, scale-normalized analytic feature.
Risk directions are fixed before the blind cohort exists:

- `A`: high `voronoi_q05__sivr_cell_anisotropy`;
- `B`: high `voronoi_q0__sivr_cell_anisotropy`;
- `C`: high `voronoi_q05__sivr_site_imbalance_max`;
- `D`: high `voronoi_q05__sivr_site_imbalance_rms`;
- `E`: high `scbv_vector_asymmetry_rms`;
- `F`: high `scbv_vector_asymmetry_max`;
- `G`: low `voronoi_q05__sivr_stiffness_min`;
- `H`: low `nm_point_reduced`.

Each term is transformed on development data as

`z_j = direction_j * (x_j - median_j) / IQR_j`,

with a candidate unsupported when a required term is missing or its IQR is not
positive.  Candidate risk scores use equal unit weights only:

`A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `A+C`, `A+E`, `A+G`, `A+H`,
`B+C`, `B+E`, `A+C+E`, `A+E+G`, and `A+C+H`.

No learned weights, nonlinear fit, feature additions, or formula edits are
allowed after blind-cohort generation.  Unsupported rows fail open and are not
rejected.

For each candidate, the development threshold is the score cut that maximizes
rejected fraction among cuts satisfying every primary gate.  Ties are resolved
lexicographically by candidate catalogue order and then by the smallest numeric
threshold.  If no candidate passes, NEXT23 stops before blind label opening.

## Statistical gates

All proportions use one-sided 95% Wilson confidence bounds.  The frozen blind
candidate passes only if all primary gates hold:

- analytic feature coverage lower bound `>= 0.90`;
- protected-structure recall lower bound `>= 0.95`;
- changed-structure precision among rejects lower bound `>= 0.90`;
- rejected-fraction lower bound `>= 0.10`.

Secondary diagnostics, never used for refitting, include ROC AUC for the
continuous endpoint tiers, Spearman correlation with the continuous endpoint,
recall/enrichment for the `>=0.20` and `>=0.50` tiers, WBM substitution-step
strata, atom-count strata, chemistry strata, and bootstrap uncertainty.

Pauling controls are evaluated from the same blind x0 structures with the same
missing-value fail-open policy.  "Beyond Pauling" requires the frozen NEXT23
law to pass all primary gates and to provide greater safe rejected fraction
than every Pauling control on the identical cohort.  No claim against DFT's
thermodynamic-stability decision is permitted from this endpoint.

## Failure policy

All output directories are no-replace.  Input/source hashes are checked before
publication.  Duplicate IDs, development/blind overlap, label-like fields in
geometry or prediction artifacts, mutated coordinates/cells, missing manifest
provenance, non-finite law parameters, or a changed source after freeze are hard
errors.  Feature-construction failures are recorded and fail open per row.

If development has no eligible candidate, keep the negative result and continue
analytic feature discovery without touching the blind endpoint.  If blind
evaluation fails, preserve the frozen prediction and result as falsification;
never tune on it or relabel it as development.

