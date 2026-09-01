# NEXT311--NEXT314 Periodic Charge-Alternation Eigenmode Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether a formal-charge signal that is an approximate alternating eigenmode of the full periodic radical-facet contact graph provides a transferable, DFT-free crystal-screening law beyond the existing Pauling, charge-spectrum, and bond-valence features.

**Architecture:** NEXT311 builds three prospectively directed, label-free graph-spectral defects from raw discovery geometries in the physically isolated SCIGEN and WyFormer inventories. NEXT312 opens only the two discovery endpoint tables and audits the three frozen hypotheses in the exact NEXT224 rejected-extreme cohort. NEXT313 and NEXT314 are conditional formula-search and exact-BROAD-diagnostic stages and must not be created unless the preceding frozen gate authorizes them.

**Tech Stack:** Python 3.11, NumPy, pandas/Parquet, ASE, pymatgen, SciPy/Qhull, pytest, SHA-256 manifests, and the existing NEXT19/NEXT279/NEXT295/NEXT308 frozen infrastructure.

---

## Scientific hypothesis and prior-art boundary

The executable descriptor treats a site-aligned formal-charge vector as a
signal on the *full* periodic radical-facet contact graph, including like-sign
and unlike-sign contacts.  In a perfectly alternating weighted bipartite graph,
the symmetric normalized adjacency has eigenvalue `-1`; normalized-Laplacian
spectra therefore provide an exact mathematical measure of bipartiteness
([Bauer and Jost](https://doi.org/10.4310/CAG.2013.v21.n4.a2)).  Exact
charge-correlation sum rules motivate rapid charge screening in equilibrium
Coulomb systems ([Stillinger and Lovett](https://doi.org/10.1063/1.1670358)).
Applying these ideas to a finite periodic crystal contact graph is a new,
explicitly tested inference, not a theorem that a small defect proves energetic
or dynamical stability.

This branch is distinct from:

- Pauling/D6 one-hop like-charge exclusions, because it tests a normalized
  multi-site eigen-equation and a two-step return relation;
- NEXT36/NEXT148 global reciprocal charge spectra, because it is site-resolved
  on a radius-weighted active-facet graph;
- NEXT19/NEXT38/NEXT104--125 transport, because the graph contains all contact
  signs and no bond-flow optimization is performed;
- NEXT267/NEXT279 radical-cell scalar/autocorrelation features, because the
  signal is formal charge rather than cell volume or Chebyshev radius; and
- NEXT307 bond-valence Hodge loops, because it acts on site charge through a
  symmetric normalized adjacency rather than projecting an edge bond-valence
  field into a cycle space.

## Frozen executable information boundary

The executable may read only one supplied composition and its initial raw,
unrelaxed periodic cell and coordinates, plus the frozen elemental/radius and
oxidation-state tables already used by NEXT19 and NEXT267.  It may use
deterministic geometry, power-cell construction, graph algebra, and linear
algebra.  It must not:

- execute DFT or read a per-structure DFT quantity;
- read an endpoint, relaxed structure, trajectory, validation record, or
  replication record;
- use an MLIP, learned energy/force/stress proxy, model/proxy potential, or
  physical relaxation; or
- compare the input with another structure of the same composition.

Unsupported structures fail open.  Discovery outcomes may be opened only by
NEXT312 or a later authorized offline stage.  Validation and replication
geometry and endpoints remain physically sealed.

## Label-blind feasibility choices frozen before outcomes

A deterministic 80+80 discovery-geometry probe was run without opening any
endpoint.  A standard all-site Voronoi graph supported 80/80 structures from
each source but was rejected before outcomes because its primary defects were
strongly redundant with existing charge-spectrum descriptors (maximum absolute
Spearman correlation approximately `0.96` in SCIGEN and `0.76` in WyFormer).

The retained radical-facet graph supported 79/80 SCIGEN and 76/80 WyFormer
probe structures.  Its primary/unit, local-tail, and two-step defects were
continuous and nondegenerate.  Their maximum absolute probe correlations with
the audited charge-spectrum/Madelung subset were, respectively, approximately
`0.62`, `0.58`, and `0.20` in SCIGEN and `0.21`, `0.56`, and `0.11` in
WyFormer.  Existing frozen label-free support masks imply exact whole-discovery
upper-bound intersection populations of 12,926/13,470 (`0.959614`) SCIGEN and
4,841/5,232 (`0.925268`) WyFormer.  Therefore NEXT311 freezes `0.90`, the
unchanged NEXT207/NEXT227 audit coverage floor, as its minimum source coverage.

The scalar Rayleigh defect is excluded before outcomes because it is an
algebraic component of the full unit-eigenmode defect and was nearly redundant
with it in the label-blind graph comparison.  No graph definition, direction,
feature, normalization, or coverage gate may change after NEXT312 opens
outcomes.

## Frozen mathematics

1. Apply the NEXT295 raw-geometry guard and NEXT267 standardized periodic-cell
   representation.
2. Infer the site-aligned neutral formal charges `q_i` with the unchanged
   NEXT19 policy.  Require both charge signs.
3. Use NEXT267 radii and NEXT279's certified periodic radical-cell active-facet
   incidences.  Require every labelled power cell to be nonempty and the
   directed contact multiset to be exactly reciprocal.
4. Construct the symmetric quotient adjacency `W`, where each directed active
   facet incidence adds one to `W_ij`, including valid periodic self-image
   incidences.  Require exact `W = W.T` and positive degree
   `d_i = sum_j W_ij` for every site.
5. Define

   ```text
   A = D^(-1/2) W D^(-1/2)
   x_i = q_i / sqrt(d_i)
   y = A x
   ```

6. Freeze exactly three `protected_low` features:

   ```text
   pcae_unit_eigen_defect = ||A x + x||_2 / ||x||_2
   pcae_local_eigen_defect_q90 = q90_i(|(A x + x)_i| / RMS(x))
   pcae_two_step_return_defect = ||A^2 x - x||_2 / ||x||_2
   ```

   The q90 uses the existing deterministic `inverted_cdf` convention.

The implementation must certify finite nonnegative outputs, neutrality,
reciprocity, positive degree, atom-order invariance, rigid-rotation invariance,
and diagonal-supercell invariance on supported test fixtures.  Uniform scaling
is intentionally *not* an invariance because the frozen elemental radii supply
a physical length scale and can change the active radical-facet topology.  It
must fail open on invalid/nonperiodic geometry, unsupported
formal charges, empty radical cells, nonreciprocal contacts, or a zero signal.

## Frozen sequential gates

### NEXT311 label-free build

- Exact rows: 13,470 SCIGEN and 5,232 WyFormer discovery structures.
- Minimum supported fraction: `0.90` independently in both sources.
- Every retained feature must be finite on every supported row and nondegenerate
  in both sources.
- Labels/endpoints and validation/replication artifacts must remain unopened.
- If any condition fails, terminate the PCAE branch and do not create NEXT312.

### NEXT312 fixed cross-source audit

Audit only the three frozen `protected_low` hypotheses in the exact NEXT224
rejected-extreme cohort.  Normalize each feature with discovery-only inverse-CDF
bounds `q_lo=1/16` and `q_hi=15/16`.  Reuse the unchanged source/fold cells and
gates:

- minimum cell coverage `0.90`;
- minimum protected and severe class count `20`;
- pooled AUC at least `0.55` in each source;
- macro fold AUC at least `0.53` in each source; and
- worst-fold AUC at least `0.50` in each source.

Rank eligible hypotheses by minimum worst-fold AUC, then minimum pooled AUC,
then mean pooled AUC, with deterministic hypothesis-name tie breaking.  If no
hypothesis is eligible, publish an empty-set digest, set
`next313_search_authorized=false`, terminate the branch, and do not create
NEXT313 or NEXT314.

### Conditional NEXT313 search

Only if NEXT312 has at least one eligible hypothesis, reuse the exact NEXT305
one-term triangular margin-local grammar on the unchanged NEXT224 closest
candidate and threshold.  For every eligible PCAE hypothesis enumerate exactly:

- local-width fractions `(1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1)` of the frozen
  repair interval; and
- amplitudes `(1/4, 1/2, 1)`.

Evaluate the unchanged AUC, SAFE12, and BROAD gates.  No interaction, second
PCAE term, sign reversal, alternate graph, continuous fit, or new grid point is
allowed.  If no candidate passes all discovery gates, set
`next314_diagnostic_authorized` only for the deterministic failed-candidate
population already defined by the unchanged NEXT305 rule; do not freeze a law.

### Conditional NEXT314 diagnostic

Only if NEXT313 creates its frozen diagnostic population, reproduce every
record exactly and apply the unchanged NEXT306 BROAD residual certificate.
This stage is diagnostic only and cannot authorize another tuning round.

## Implementation tasks

### Task 1: NEXT311 kernel and builder

**Files:**

- Create: `tests/test_next311_periodic_charge_alternation_eigenmode.py`
- Create: `src/next311_periodic_charge_alternation_eigenmode.py`

1. Write failing tests for exact alternating graphs, feature schema/directions,
   invalid inputs, invariances, fail-open behavior, discovery-only provenance,
   output identity, and boundary flags.
2. Run the new test and verify import failure before creating production code.
3. Implement only the frozen kernel, per-structure adapter, parallel source
   build, catalogue, and manifest.
4. Run the focused and adjacent NEXT19/NEXT267/NEXT279/NEXT295 tests.
5. Run the formal NEXT311 build under
   `$PRIS_ARCHIVE/next311_periodic_charge_alternation_eigenmode_v1`.

### Task 2: NEXT312 audit

**Files:**

- Create: `tests/test_next312_pcae_feature_audit.py`
- Create: `src/next312_pcae_feature_audit.py`

1. Write failing tests for the exact three hypotheses, protected-low direction,
   fixed bounds/gates, deterministic eligibility order/digest, identity joins,
   provenance, and zero-eligible stopping behavior.
2. Verify import failure, then implement by reusing the unchanged NEXT308 audit
   and NEXT224 reconstruction infrastructure.
3. Run NEXT312 exactly once on the two discovery endpoint lockboxes and formal
   NEXT311 artifacts.
4. Inspect only the preregistered table and stop immediately if the eligible set
   is empty.

### Task 3: Conditional NEXT313/NEXT314

**Files, only if authorized:**

- Create: `tests/test_next313_pcae_margin_local_search.py`
- Create: `src/next313_pcae_margin_local_search.py`
- Create: `tests/test_next314_pcae_broad_diagnostic.py`
- Create: `src/next314_pcae_broad_diagnostic.py`

Follow the frozen conditional grammar and TDD.  Do not create these files merely
to record that the gate failed.

### Task 4: Independent report and verification

**Files:**

- Modify only: `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`

Append the design, label-blind probe, formal coverage/failures, exact audit
metrics, sequential stopping decision, artifact paths, and SHA-256 identities.
Do not modify `paper/`, `tex/`, `notes/`, `README.md`, or `PREREG.md`.  Verify
artifact hashes, source hashes, boundary flags, focused tests, the full test
suite, canonical-path status, and CodeGraph synchronization before reporting.
