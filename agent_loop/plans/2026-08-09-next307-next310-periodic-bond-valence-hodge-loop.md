# NEXT307--NEXT310 Periodic Bond-Valence Hodge Loop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether the cycle-space component of raw-geometry bond valences supplies a transferable, fully pre-DFT crystal-plausibility certificate beyond the frozen NEXT224/Pauling frontier.

**Architecture:** NEXT307 constructs an oriented periodic opposite-sign Voronoi multigraph and projects its observed bond-valence edge field onto the graph cycle space. NEXT308 audits exactly four frozen low-loop-residual hypotheses on the unchanged two-source discovery cohort. NEXT309 and NEXT310 are conditional: they run only if the preceding fixed gate authorizes them, use the unchanged one-term triangular margin-local grammar, and never open validation or replication.

**Tech Stack:** Python 3.11, NumPy/SVD, pandas/Parquet, pymatgen/ASE, pytest, existing NEXT19/NEXT38 graph and bond-valence helpers, and existing NEXT224/NEXT268/NEXT261/NEXT164 frozen evaluators.

**Repository constraint:** This checkout is intentionally dirty and additive. Do not create a worktree, branch, commit, merge, PR, or cleanup. Preserve all prior scripts and artifacts. Do not modify `paper/`, `tex/`, `notes/`, `README.md`, or `PREREG.md`.

## 1. Scientific motivation and novelty boundary

Bond-valence theory contains two independent network requirements. The
valence-sum rule is a modernization of Pauling's second rule; the path or loop
rule requires the signed bond-valence sum around every closed path to vanish.
The two rules can be written as network equations for a fixed bond topology
and used to derive a-priori bond valences and structural strain
([Gagné, Mercier, and Hawthorne, 2018](https://doi.org/10.1107/S2052520618010442)).
Bond-valence parameters are also used explicitly for crystal-structure
plausibility tests
([Chen and Adams, 2017](https://doi.org/10.1107/S2052252517010211)).

The repository already implements the site-sum side extensively:

- NEXT19 transports valence over an opposite-sign periodic graph while
  satisfying cation supply and anion demand;
- NEXT22 measures local bond-valence mismatch;
- NEXT38 projects the site-conserving correction through a geometric
  Jacobian;
- NEXT109--NEXT125 test finite-capacity/Hall obstructions.

None directly projects the *observed geometry-derived bond-valence edge field*
onto the periodic graph's loop space. NEXT307 therefore tests the independent
path-rule residual rather than another cation/anion sum, force closure,
directional cage, topology rank, or post-outcome transformation.

Three alternatives were compared before any new outcome was opened:

1. **Selected:** the unnormalized observed bond-valence field, because the
   path rule is stated for bond valences themselves and this retains the
   calibrated bond-length relation.
2. A cation-star-normalized edge field. An 80+80 label-free probe found it
   strongly redundant with the selected field: paired Spearman correlations
   were `0.834--0.934` in SCIGEN and `0.971--0.982` in WyFormer. Searching both
   would add redundant degrees of freedom, so it is excluded.
3. A local metric-strain or signed-contact-persistence branch. These overlap
   substantially with NEXT37/NEXT38 and NEXT123/NEXT166 respectively and are
   not selected.

The deterministic label-free probe used 80 evenly spaced discovery identities
from each physically isolated geometry inventory. It supported 80/80 in both
sources. All SCIGEN loop features had 80 distinct values at ten decimal
places. WyFormer had 53 distinct values and a 35% exact-zero population; every
probed graph nevertheless had positive cycle dimension (minimum 6 in SCIGEN
and 2 in WyFormer). No endpoint, label, validation, or replication payload was
read by this probe.

The literature motivates the descriptor only. A small loop residual is not a
proof of energetic stability, realizability, synthesizability, a correct
oxidation assignment, or DFT equivalence.

## 2. Hard executable information boundary

The NEXT307 executable may read exactly one raw initial periodic structure:
elements, stoichiometry, cell, coordinates, and fixed elemental/bond-valence
tables. It may use:

- the frozen NEXT19 valence-assignment cascade;
- the frozen NEXT19 opposite-sign Voronoi periodic multigraph;
- the frozen NEXT38 exact/fallback bond-valence parameter resolver;
- deterministic arithmetic, SVD, and empirical inverse-CDF aggregation.

It may not:

- execute DFT or read any per-structure DFT value;
- use a learned energy, force, stress, stability, or relaxation proxy;
- invoke a model/proxy potential or compare same-composition alternatives;
- read a relaxed structure, trajectory, validation geometry, replication
  geometry, or any validation/replication endpoint;
- update coordinates or the cell, solve a physical relaxation, or emit an
  energy/force/stress proxy.

Discovery outcomes are permitted only in NEXT308--NEXT310 as offline labels
after NEXT307 source, tests, feature catalogue, directions, design hash, and
label-free feature tables are frozen. All failures abstain; unsupported rows
are never automatic rejections.

## 3. Frozen periodic Hodge construction

For the frozen neutral formal charges `z_i`, orient every retained periodic
Voronoi edge from cation `i` to anion `j`. Distinct periodic images remain
distinct multiedges. For each edge `e`, NEXT38's unchanged parameter policy
gives `(R0_e, B_e)` and the raw geometry gives distance `d_e`. Define the
positive observed bond valence

```text
b_e = exp((R0_e - d_e) / B_e).
```

Let `D` be the signed site-edge incidence matrix:

```text
D[i,e] = +1  at the cation endpoint,
D[j,e] = -1  at the anion endpoint.
```

Using the single frozen tolerance

```text
tol = eps * max(D.shape) * sigma_max(D),
```

let `V_r` contain the right singular vectors for singular values above `tol`.
The row-space projection and loop residual are

```text
b_gradient = V_r.T @ (V_r @ b)
c_loop     = b - b_gradient.
```

Thus `c_loop` is the orthogonal projection of `b` into `ker(D)`, and
`D @ c_loop = 0` up to the frozen numerical tolerance. The path rule is
satisfied exactly when `c_loop = 0`. Parallel periodic images automatically
form valid two-edge quotient cycles; longer graph cycles require no explicit
cycle-basis choice.

Let `m = mean(b_e)`, which is positive on supported structures, and
`x_e = |c_loop,e| / m`. For every site, let `r_i` be the RMS of `x_e` over
incident edges. Quantiles use NumPy's empirical `inverted_cdf` method.

## 4. Frozen feature catalogue and directions

Exactly four NEXT307 hypotheses are frozen, all `protected_low`:

1. `pbvhl_cycle_fraction = ||c_loop||_2 / ||b||_2`;
2. `pbvhl_cycle_rms = sqrt(mean_e x_e^2)`;
3. `pbvhl_cycle_q90 = inverted_cdf_q90_e(x_e)`;
4. `pbvhl_site_rms_q90 = inverted_cdf_q90_i(r_i)`.

The first quantity is bounded in `[0,1]`; the other three are finite and
nonnegative. Diagnostics only are site/edge counts, incidence rank, cycle
dimension and fraction, valence-assignment policy, and bond-valence parameter
source fractions. No maximum, alternate quantile, normalized-edge version,
feature conjunction, direction reversal, source-conditioned variant, or
outcome-derived transformation may be added after outcomes are opened.

Required invariants:

- equal-valued edges on a four-cycle have zero loop residual;
- perturbing one edge produces the analytic cycle-space projection;
- equal parallel periodic images have zero residual and unequal images do not;
- `D @ c_loop` is numerically zero;
- common positive scaling of all `b_e`, edge ordering, atom ordering, rigid
  rotation, periodic translation, equivalent lattice rebasing, and exact
  integral supercell representation preserve all four features;
- malformed/nonperiodic/decorated inputs fail open without feature values.

## 5. NEXT307 label-free formal gate

NEXT307 consumes only the physically isolated discovery partitions from:

- `$PRIS_ARCHIVE/next84_scigen_geometry_lockbox_v1`;
- `$PRIS_ARCHIVE/next93b_wyformer_blind_source_lockbox_v1`.

It must publish exactly 13,470 SCIGEN rows and 5,232 WyFormer rows to
`$PRIS_ARCHIVE/next307_periodic_bond_valence_hodge_loop_v1`.
Formal authorization for NEXT308 requires source coverage at least `0.95`, all
four values finite on every supported row, no value present on unsupported
rows, a positive cycle dimension on supported rows, and nondegeneracy in both
sources. The manifest must record every DFT/proxy/relaxation/endpoint/opening
flag as false and the exact source, test, design, input, and output hashes.

## 6. NEXT308 fixed cross-source audit

Only after NEXT307 is frozen may NEXT308 reconstruct the unchanged NEXT224
frontier and exact rejected-extreme cohort. It uses the same physical discovery
partitions, reduced-formula five-fold assignment, cell-count/coverage gates,
and source thresholds as NEXT268/NEXT304:

- combined-discovery `1/16` and `15/16` inverted-CDF normalization;
- aggregate AUC at least the unchanged frozen minimum in both sources;
- macro-fold and worst-fold AUC at least the unchanged frozen minima;
- opposite-direction veto and minimum cell support/coverage;
- no source-specific normalization, cutoff, direction, or hypothesis.

All four hypotheses are audited. If zero pass, publish the negative NEXT308
artifact, set `next309_search_authorized = false`, and stop the branch. If one
or more pass, freeze the complete eligible set and its identity digest before
NEXT309.

## 7. NEXT309 bounded search and NEXT310 residual

For `K` eligible NEXT308 hypotheses, NEXT309 evaluates exactly one unchanged
NEXT224 reproduction control plus `K * 7 * 3` new candidates. It reuses the
NEXT261 triangular margin-local grammar without modification:

```text
local_width_fraction in {1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1}
amplitude_fraction   in {1/4, 1/2, 1}
missing feature      -> term off, exact NEXT224 score retained
support              -> unchanged NEXT214/NEXT224 support
score                 -> max(0, NEXT224 score + one signed local term)
```

No interaction, second term, new amplitude, cutoff, threshold, or selected
subset is allowed. NEXT309 freezes a law only if a new candidate passes both
source AUC gates, all 12 SAFE cells, all BROAD cells, and every inherited
discovery gate. A passing discovery candidate still does not authorize opening
validation or replication without a separate frozen validation action.

If no new candidate passes all gates but at least one passes AUC+SAFE and not
BROAD, NEXT310 reproduces exactly that complete diagnostic population and
computes the unchanged NEXT164 BROAD threshold residual. It compares
lexicographically against the frozen NEXT235 reference:

```text
(failed constraint count, normalized shortfall sum)
= (5, 0.12339543654931197).
```

NEXT310 searches no new formula. Failure to strictly improve closes this
branch. No residual result may be used to alter NEXT307 features or NEXT309's
grid.

## 8. Additive files and TDD execution tasks

### Task 1: Freeze this design

**Files:**

- Create: `docs/plans/2026-08-09-next307-next310-periodic-bond-valence-hodge-loop.md`

1. Save this plan before any production source or endpoint access.
2. Record its SHA-256 in NEXT307 and every downstream manifest.
3. Do not edit the plan after the first endpoint is opened; amendments, if ever
   scientifically required before endpoints, must be separate additive files.

### Task 2: Implement the pure NEXT307 kernel with TDD

**Files:**

- Create: `tests/test_next307_periodic_bond_valence_hodge_loop.py`
- Create: `src/next307_periodic_bond_valence_hodge_loop.py`

1. Write failing tests for the four-cycle, parallel-edge, divergence-free,
   scale/order/replication invariants, invalid inputs, raw NaCl geometry
   transformations, and geometry-only firewall.
2. Run the focused test and confirm failure because the module is absent.
3. Implement the minimal pure kernel and raw-structure wrapper.
4. Re-run focused and adjacent NEXT19/NEXT38/NEXT303 tests.
5. Add the no-replace two-source formal publisher and its manifest tests.

### Task 3: Publish NEXT307 without labels

1. Hash-lock this plan, required sources, both geometry manifests, discovery
   geometry files, and discovery metadata.
2. Run the formal builder with 16 workers.
3. Verify row identity, coverage, feature finiteness/ranges, cycle dimensions,
   output hashes, source hashes, and false boundary flags.
4. Do not open or load an endpoint table in this task.

### Task 4: Implement and run NEXT308 with TDD

**Files:**

- Create: `tests/test_next308_pbvhl_feature_audit.py`
- Create: `src/next308_pbvhl_feature_audit.py`

1. Write failing tests for material-ID alignment, both protection directions,
   fixed eligible selection, formal-input identity, no-replace behavior, and
   validation/replication closure.
2. Implement only the four-hypothesis audit by reusing the frozen NEXT268
   evaluator.
3. Run it once on the already-open discovery endpoints.
4. Follow the frozen zero/nonzero eligibility stop rule.

### Task 5: Conditionally implement NEXT309 and NEXT310 with TDD

**Files:**

- Conditionally create: `tests/test_next309_pbvhl_margin_local_search.py`
- Conditionally create: `src/next309_pbvhl_margin_local_search.py`
- Conditionally create: `tests/test_next310_pbvhl_broad_diagnostic.py`
- Conditionally create: `src/next310_pbvhl_broad_diagnostic.py`

1. Do not create NEXT309 if NEXT308 authorizes zero hypotheses.
2. If authorized, write failing tests before each production module.
3. Reuse the unchanged NEXT261 grid/evaluator and NEXT164 residual machinery.
4. Do not create NEXT310 unless NEXT309 authorizes a nonempty exact diagnostic
   population.

### Task 6: Report and verify

**Files:**

- Modify only the independent report:
  `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`

1. Append mechanism, literature boundary, pre-label probe, formal coverage,
   audit/search/diagnostic results, hashes, and the exact scientific claim
   boundary.
2. Do not edit canonical paper/report assets pending user confirmation.
3. Run focused tests, the full repository suite, an independent manifest/output
   hash verifier, boundary-flag assertions, canonical-path diff checks, and
   CodeGraph status/search.
4. Report a new law only if every discovery gate passes and later sealed
   validation evidence separately justifies it. Otherwise state the exact
   negative or intermediate result and keep the overall goal active.
