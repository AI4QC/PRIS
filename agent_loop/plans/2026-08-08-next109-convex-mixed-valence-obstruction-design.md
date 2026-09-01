# NEXT109 Convex Mixed-Valence Obstruction Certificate Design

Date: 2026-08-08

## Objective and immutable boundary

NEXT109 adds a necessary-condition obstruction certificate for one raw,
unrelaxed periodic structure before DFT.  The executable may use only `x0`, a
frozen oxidation-state catalogue, elemental electronegativity, deterministic
Voronoi geometry, and linear programming.  It may not use DFT values or
calculations, relaxed structures or trajectories, energies, forces, stresses,
learned proxies, MLIPs, or same-composition alternatives.

The branch is additive.  NEXT104--NEXT108 and every older source, result, and
report remain unchanged.  NEXT109 gets new source and tests; NEXT110 gets a
discovery-only feature artifact; NEXT111 gets a finite frozen search.  A new
standalone report precedes any canonical report or paper edit.  Validation and
replication files remain physically unopened unless every discovery gate passes.

## Label-free opportunity audit

The immutable NEXT105 feature artifacts show that CMVF currently abstains on a
large, mechanistically relevant population:

| source/mode | global interval imbalance | isolated opposite-sign site | cation without opposite-sign neighbor | graph-interval LP infeasible |
|---|---:|---:|---:|---:|
| SCIGEN core | 3,630 | 69 | 794 | 11 |
| SCIGEN expanded | 2,065 | 128 | 524 | 12 |
| WyFormer core | 1,127 | 55 | 153 | 7 |
| WyFormer expanded | 704 | 38 | 198 | 5 |

These counts use no endpoint labels.  Missing Brown parameters account for a
separate 459/1,125/210/446 rows respectively.  Those are missing-data cases,
not evidence of structural impossibility, and must never become automatic
rejections.

## Alternatives considered

### A. Normalized minimum interval-slack LP with a dual cut interpretation (selected)

This solves a bounded primal distance-to-feasibility problem.  Its optimum is
zero exactly when the original mixed-valence interval flow is feasible.  Strong
LP duality makes a positive optimum the magnitude of a normalized Farkas/cut
obstruction without depending on solver-specific access to an infeasible ray.
It detects global, disconnected-component, and within-component Hall-type
obstructions.

### B. Component charge-balance gaps only

This is cheap and interpretable, but insufficient.  A connected bipartite graph
can have globally and component-wise balanced charge intervals while a cation
subset is connected to too little anion capacity.  Component gaps are retained
as diagnostics, not used as the complete certificate.

### C. Extract an infeasible dual ray directly from HiGHS

This would expose a literal witness, but SciPy's `linprog` result does not offer
a stable, portable infeasibility-ray contract.  Solver-version behavior would
become part of the scientific definition.  This option is rejected.

## Frozen mathematical definition

For one electronegativity-oriented sign pattern, let site `i` have nonzero
same-sign charge-magnitude interval `[l_i,u_i]`.  Construct the raw Voronoi
opposite-sign graph and deduplicate its unit-cell endpoint pairs.  Let `B` be
the unsigned site-edge incidence matrix.  `y_e >= 0` is normalized edge flow,
`sum_e y_e = 1`, `r >= 0` is inverse total charge, and `s_i >= 0` is normalized
site-interval violation.  Solve

\[
\begin{aligned}
D^*=\min_{y,r,s}\;&\frac12\sum_i s_i\\
\text{s.t. }&B_i y-u_i r\le s_i,\\
&l_i r-B_i y\le s_i,\\
&\sum_e y_e=1,\quad y,r,s\ge0.
\end{aligned}
\]

`D* = 0` if and only if the unrelaxed graph can realize all site charge
intervals.  For any nonempty graph, `r=0` and `s=By` is a feasible fallback
with objective one, hence `0 <= D* <= 1`.  Common scaling of all charge bounds
is absorbed by inverse scaling of `r`.  Repeating an integer supercell divides
normalized flow and `r` among copies while preserving the summed slack.

Three deterministic decompositions accompany `D*`:

- global balance gap: distance between total positive and negative charge
  intervals, divided by the larger total lower bound;
- component balance gap: maximum similarly normalized gap across projected
  bipartite connected components, with a one-sign component assigned one;
- unserved-site fraction: fraction of sites with zero incident opposite-sign
  endpoint.

All lie in `[0,1]`.  A graph with no endpoint receives `D*=1`, component gap
one, and unserved fraction one.  Neighbor-construction exceptions abstain.

## Sign-pattern and missing-data policy

NEXT109 reuses the frozen NEXT104 `core` and `expanded` oxidation catalogues,
catalogue digest, electronegativity orientation, and exact sign-pattern bound of
128.  For each pattern it builds a Brown-free topology: distance and Voronoi
weight are used only to validate a neighbor; no Brown parameter enters.

The structure certificate selects one coherent pattern by lexicographically
minimizing `(D*, global_gap, component_gap, unserved_fraction, pattern)`.
This is the conservative existential rule: a structure is obstructed only if
every allowed oriented sign interpretation is obstructed.  Features are not
minimized independently across different patterns.

Empty oxidation or electronegativity catalogues, no oriented sign pattern,
sign-pattern overflow, single-element structures, and neighbor-finder or solver
exceptions abstain with an auditable reason.  Missing Brown parameters are not
consulted and therefore cannot raise risk.

## Frozen feature schema

Each supported catalogue mode emits four finite high-is-risk terms:

- `cmvo_min_interval_slack`;
- `cmvo_global_balance_gap`;
- `cmvo_component_balance_gap`;
- `cmvo_unserved_site_fraction`.

The wrapper records support, failure reason, catalogue SHA-256, pymatgen/scipy
versions, and the graph policy.  No selected charges, endpoints, labels, DFT
quantities, or learned predictions are scientific features.

## Discovery-only build and finite search

NEXT110 may read only the physically isolated SCIGEN and WyFormer discovery
geometries and their metadata.  Its CLI has no validation or replication path.
It emits both catalogue modes, status provenance, immutable input/source/output
hashes, and a manifest declaring forbidden inputs unopened.

Before any endpoint join, NEXT111 freezes label-free eligibility and scaling for
the eight core/expanded terms.  A term is eligible only with at least 15% finite
coverage and eight unique finite values in each source.  Calibration uses
pooled discovery geometries without endpoints: subtract the pooled median and
divide by the pooled interquartile range; clip the nonnegative high-risk tail at
the pooled 99.5th percentile.  Unsupported values contribute exactly zero and
must not alter the base support mask.

The finite search starts from the 353 NEXT108 bases that were within 0.01 of all
six AUC gates.  It tests each eligible CMVO term singly at weights
`{0.25, 0.5, 1, 2, 4}` and same-mode pairs at weights `{0.25, 0.5, 1, 2}`.  It
does not add CMVF weights, tune signs, or change existing base coefficients.
Candidate ordering and tie breaks are frozen before endpoints are read.

Scientific gates remain unchanged: pooled, macro-crystal-system, and
worst-crystal-system AUC gates in both sources; SAFE precision/recall gates in
both source aggregates and ten fixed reduced-formula folds; and one shared BROAD
threshold that Pareto-dominates Pauling in all 12 cells.  Replication may open
only if one candidate passes every discovery gate.

## Required tests

Tests must observe RED before implementation and cover exact feasibility,
global imbalance, disconnected-component imbalance, a connected Hall/cut
obstruction, isolated and empty graphs, site/edge permutation, integer
supercell replication, common charge scaling, deterministic raw-structure
evaluation, core/expanded behavior, raw-structure immutability, missing Brown
independence, explicit technical abstention, forbidden-schema tokens,
discovery-only input purity, optional-term fail-open semantics, and frozen
candidate enumeration.

Engineering tests prove only implementation and boundary compliance.  They do
not establish screening quality; that requires the frozen discovery gates and,
only after they pass, untouched external replication.
