# NEXT104 Convex Mixed-Valence Flow Certificate Design

Date: 2026-08-04

## Objective and boundary

Build a new analytic certificate for screening one raw, unrelaxed periodic structure before DFT. The executable may use only the input `x0`, frozen oxidation-state catalogues, frozen elemental data, deterministic Voronoi geometry, and linear programming. It must not use DFT values, relaxed structures, relaxation trajectories, learned energy/force/stress proxies, MLIPs, same-composition alternatives, or validation/replication labels.

The mechanism is additive. It does not replace NEXT19, NEXT22, NEXT38, NEXT101, NEXT101b, or NEXT103. It creates independent source, tests, feature artifacts, search artifacts, and a standalone report. Canonical paper/report files remain untouched until user review.

## Why the previous mechanism is insufficient

NEXT19 and NEXT38 optimize bond-valence flow only after a single site-charge vector has already been fixed. NEXT101 and NEXT101b enumerate several element-uniform charge vectors, but all sites of one element must share one integer oxidation state. That excludes valid mixed-valence compositions and lets an arbitrary best assignment determine the result.

A label-free composition audit on the two discovery sources found:

| source | expanded element-uniform neutral | exact site-mixed neutral | convex mixed-valence interval |
|---|---:|---:|---:|
| SCIGEN | 7,136/13,470 (52.98%) | 8,385/13,470 (62.25%) | 8,459/13,470 (62.80%) |
| WyFormer | 3,010/5,232 (57.53%) | 3,491/5,232 (66.72%) | 3,495/5,232 (66.80%) |

The exact site-level integer formulation is not representation invariant: a supercell can realize charge disproportionation that a smaller cell cannot, even when both encode the same periodic structure. The convex hull of allowed integer states represents arbitrary mixed-valence fractions without tying the certificate to the chosen cell size.

## Alternatives considered

### A. Exact site-level mixed-integer charge and flow assignment

This is chemically literal and computationally feasible for most rows: the median minimum binary-variable counts are 19 for SCIGEN and 40 for WyFormer. It is rejected as the primary certificate because its feasibility and optimum may improve solely after making a supercell. It may be retained later as a diagnostic, never as the frozen representation-invariant decision score.

### B. Deterministic site ordering followed by fixed-charge transport

This is fast but assigns oxidation states from an arbitrary ordering or local ranking. It can violate site-permutation invariance and would repeat the fixed-charge limitation of NEXT19/38. It is rejected.

### C. Convex mixed-valence periodic flow certificate (selected)

For each electronegativity-oriented element sign pattern, every site receives a continuous charge magnitude inside the convex hull of the allowed same-sign integer oxidation states. A periodic bond flow is optimized jointly with those charges. This is a necessary-condition certificate: a high incompatibility is evidence against the structure, while a low incompatibility is not claimed to prove stability.

## Mathematical certificate

For a fixed sign pattern, construct the raw Voronoi cation-anion edge set. Each edge receives a charge-blind Brown generic bond-valence strength

\[
b_e=\exp[(R_{0,e}-d_e)/0.37],
\qquad p_e=b_e/\sum_j b_j.
\]

No oxidation-state-specific fitted parameter is used in `b_e`; the element-only Brown parameters and raw distance determine it.

Let `y_e` be normalized nonnegative edge flow with `sum(y)=1`, and let `r=1/Q`, where `Q` is the total positive charge. For site `i`, the incident normalized flow is `z_i`. If its same-sign oxidation magnitudes span `[l_i,u_i]`, the mixed-valence domain is

\[
l_i r \le z_i \le u_i r.
\]

The first LP minimizes total-variation reallocation from the geometric prior:

\[
T^*=\frac12\min\sum_e |y_e-p_e|.
\]

On the `T*` optimum face, a second LP minimizes the maximum edge overload `kappa` subject to `y_e <= kappa p_e`. On both optimum faces, a third LP chooses `r` closest to the raw-scale target `1/sum(b)`. The scale mismatch is

\[
S=|\log(Q/\sum_e b_e)|.
\]

The three reported risks `T*`, `max(kappa-1,0)`, and `S` are optimum values, not properties of an arbitrary charge assignment. The normalized formulation makes them invariant to site permutation, edge ordering, uniform edge-prior scaling, and integer supercell replication.

## Catalogue modes and sign patterns

Compute two independent modes:

- `core`: nonzero integer states from the union of pymatgen common and ICSD catalogues;
- `expanded`: all nonzero integer pymatgen oxidation states.

For each element, enumerate only signs represented in the selected catalogue. Require both signs in the structure and require the count-weighted mean electronegativity of negative sites to exceed that of positive sites. Enumerate every valid sign pattern up to a frozen bound of 128; exceeding the bound causes explicit abstention and never truncation.

Use Voronoi edges with the existing frozen `solid_angle`, `tol=0`, `cutoff=13 Å` policy. Dummy neutral decoration is allowed only to reuse the geometry builder; its magnitudes cannot enter priors or LP constraints. Missing Brown parameters, isolated opposite-sign sites, invalid numeric values, solver failure, or frozen-bound overflow return an auditable unsupported result.

## Frozen feature schema

Each catalogue mode emits only finite values when supported:

- `cmvf_reallocation`: `T*`;
- `cmvf_overload`: `max(kappa-1,0)`;
- `cmvf_log_scale_mismatch`: `S`;
- `cmvf_domain_width_mean`: mean `(u_i-l_i)/(u_i+l_i)`;
- `cmvf_domain_width_max`: maximum normalized domain width;
- `cmvf_sign_pattern_log_count`: `log1p` of the number of valid sign patterns.

The wrapper also records support, failure reason, catalogue hash, pymatgen/scipy versions, and graph policy. It does not expose a selected site charge vector as a scientific feature.

## Discovery search protocol

NEXT105 will compute both modes only for the physically isolated SCIGEN and WyFormer discovery geometries. It must have no validation/replication geometry or endpoint arguments. A corruption test will prove that inaccessible non-discovery files cannot affect the output.

NEXT106 will start from the same 67 NEXT98b bases that passed both-source AUC gates. CMVF terms are optional guards: when a mode is unsupported, the correction is exactly zero and the base score/support mask is unchanged. Physics-prespecified directions are high reallocation, high overload, high scale mismatch, and high domain ambiguity. Term calibration is label-free and occurs before discovery endpoints are joined. Each term must have at least 15% active coverage in both sources and at least eight unique values.

Candidate weights remain `{0.25, 0.5, 1, 2, 4}`. The unchanged gates are both-source pooled/macro/worst crystal-system AUC, SAFE operation in both source aggregates plus ten fixed reduced-formula folds, and a shared BROAD threshold that Pareto-dominates Pauling in all 12 cells. Replication remains physically unopened unless one frozen candidate passes every gate.

## Required tests

Tests must observe RED before implementation and cover:

- an exactly compatible synthetic mixed-valence network;
- a network whose charge intervals cannot balance;
- an isolated opposite-sign site;
- site and edge permutation invariance;
- integer supercell replication invariance;
- invariance to common edge-prior scaling for reallocation/overload, with predictable scale shift;
- core versus expanded catalogue behavior;
- deterministic repeated execution;
- raw-structure immutability;
- explicit abstention on invalid graph, missing parameter, solver failure, and bound overflow;
- schema exclusion of DFT, energy, force, stress, relaxation, and learned-proxy fields;
- discovery-only builder purity and optional-guard missing semantics.

Engineering tests do not establish scientific quality. Scientific success requires the unchanged cross-source frozen gates.
