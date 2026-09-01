# NEXT43 analytic law search design

## Status

Development-only search after NEXT42 endpoint opening. Any formula discovered here is unconfirmed until it is frozen and evaluated on unseen Alexandria path shards.

## Objective

Find a compact DFT-free rejection law that predicts large raw-x0 to converged-DFT structural change while preserving structures whose fingerprint change is at most 0.10.

## Non-negotiable execution boundary

The executable descriptor and rule may use one raw, unrelaxed, pre-DFT, pre-MLIP periodic structure only: elements, cell, and coordinates. It may use frozen elemental tables, analytic radii, Voronoi graphs, bond-valence parameters, classical Ewald derivatives, deterministic linear algebra, and spglib. It may not use DFT values, endpoint structures, trajectories, MLIP/model potentials, learned energy/force/stress proxies, physical relaxation, or same-composition alternatives.

DFT endpoints are allowed only in the offline NEXT43 formula-development program.

## Data roles

- NEXT42 geometry-only x0 archive: descriptor input.
- NEXT42 joined converged evaluation: development labels only.
- Unseen Alexandria shards: future confirmation; they must not be opened until the selected formula and thresholds are frozen.

## Descriptor bank

Reuse the already tested single-x0 kernels without changing them:

- periodic covalent contacts;
- size/valence rigidity, normalized Madelung, and bond-valence equilibrium;
- approximate symmetry recovery and directional steric imbalance;
- valence transport at a finite alpha grid;
- analytic Ewald-field imbalance;
- Coulomb--steric vector balance;
- charge-spectrum descriptors;
- self-stress compatibility;
- bond-valence transport compatibility.

Every family is fail-open. Unsupported or non-finite values may reduce formula coverage but may never cause rejection.

## Finite formula catalogue

Use a deterministic material-id hash split fixed before search:

- discovery: 60%;
- internal validation: 40%.

Only discovery labels select directions, robust centers/scales, thresholds, feature shortlist, and the final development candidate. The catalogue contains:

1. one-feature high/low threshold laws;
2. two-feature robust-standardized additive scores with a finite coefficient grid;
3. two-feature conjunctive extreme-condition laws;
4. equal-weight three-feature additive and conjunctive laws over a discovery-only finite shortlist.

Thresholds come from a fixed rejection-fraction grid. Missing values fail open. Candidate selection is lexicographic: maximize the minimum normalized primary-gate margin, then protected recall lower bound, rejection-precision lower bound, savings lower bound, and continuous AUC. Internal validation is reported once for the selected discovery candidate and is not used to retune it.

## Gates

On both discovery and internal validation, the 95% Wilson lower bounds must satisfy:

- coverage >= 0.90;
- protected recall >= 0.95;
- rejection precision >= 0.90;
- savings >= 0.10.

Passing these internal development gates is necessary but not sufficient. Scientific confirmation additionally requires the same frozen formula to pass on unseen, source-qualified, force-converged Alexandria shards.

## Interpretation

The target is structural response under DFT relaxation. It is a useful DFT-before-screening endpoint but does not by itself prove convex-hull, phonon, or finite-temperature dynamical stability.
