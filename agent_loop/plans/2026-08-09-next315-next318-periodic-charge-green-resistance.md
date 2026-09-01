# Periodic Charge Green Resistance Implementation Plan

> **For Codex:** Execute this plan task-by-task with test-driven development. The repository is intentionally dirty and the user requires additive work, so do not create a worktree, commit, delegate to subagents, or modify canonical paper/report sources.

**Goal:** Test whether a parameter-free multi-source effective-resistance certificate on the raw periodic radical contact graph yields a transferable, fully DFT-free crystal plausibility law beyond the existing Pauling/frontier score.

**Architecture:** NEXT315 constructs an auxiliary unit-conductance graph from the full reciprocal NEXT279 periodic radical active-facet incidences and solves one graph Poisson equation for the formal-charge injection vector. NEXT316 applies the unchanged NEXT224 rejected-extreme cross-source audit to three frozen protected-low summaries. NEXT317 and NEXT318 are conditional: an exact one-term triangular margin-local search and, only if authorized, the unchanged BROAD residual diagnostic.

**Tech Stack:** Python 3.11, NumPy symmetric eigensolver, pandas/Parquet, ASE, pymatgen, existing NEXT19/NEXT267/NEXT279/NEXT295 geometry and provenance guards, pytest.

---

## 1. Scientific question and novelty audit

The existing repository already contains all of the following and they must not
be presented as new:

- NEXT19 minimum-cost formal-valence transport and its overload/reallocation
  summaries;
- NEXT119--NEXT125 bounded transport, Hall-cut, and multiscale contact
  feasibility;
- NEXT36/NEXT148 reciprocal-space long-wavelength charge spectrum;
- NEXT166 periodic contact topology;
- NEXT299 opposite-sign local cages, NEXT303 reciprocal cage balance, NEXT307
  bond-valence loop residuals, and NEXT311 charge-alternation eigenmodes.

Repository-wide literal searches found no graph-Laplacian pseudoinverse,
effective-resistance, or `q^T L^+ q` implementation. The new mechanism is not
shortest-path/minimum-cost transport: all parallel contact paths contribute
simultaneously through the Green operator.

The mathematical electrical-network interpretation follows resistance
distance and graph-Laplacian theory
([Klein and Randic](https://doi.org/10.1007/BF01164627)). The physical reason
to test locality of charge screening follows exact charge-correlation
restrictions
([Stillinger and Lovett](https://doi.org/10.1063/1.1670358)). Applying an
auxiliary unit-conductance contact graph to a finite crystal cell is our tested
inference, not a theorem of stability. The auxiliary potential and dissipation
below are graph quantities; they are not electrostatic energies, DFT values,
or a fitted/model potential.

## 2. Frozen information boundary

All executable features and candidate scores may read only:

- element identities and deterministic tabulated radii;
- the initial, unrelaxed, fully periodic geometry;
- deterministic formal-valence assignment from NEXT19;
- the reciprocal active-facet contact multigraph from NEXT279.

They must not:

- execute DFT or read any per-structure DFT value;
- read energy, force, stress, calculator, outcome, trajectory, relaxed
  structure, or model-potential payloads;
- use a learned energy/force/stress proxy or any analytic pair potential;
- perform structural relaxation or use validation/replication geometry;
- fit graph conductances, radii, feature directions, normalizations, or search
  widths to discovery outcomes.

Discovery outcomes may be opened only in NEXT316 and later as offline labels.
Internal validation and replication geometry/endpoints remain physically
sealed. Canonical `paper/`, `tex/`, `notes/`, `README.md`, and `PREREG.md`
remain untouched before user review.

## 3. Frozen PCGR construction

For the full reciprocal directed contact population, define the symmetric
quotient count matrix

`W_ij = number of directed NEXT279 incidences i -> j`.

Every active radical facet has its reciprocal incidence, so `W = W^T`. Use one
unit graph conductance per incidence and define

`L = diag(W 1) - W`.

Periodic self-image contacts appear on the diagonal of `W` and cancel in `L`,
as required because an edge from a quotient vertex to itself cannot transport
net charge between distinct site potentials. Let `q` be the neutral formal
charge vector. Using the Moore-Penrose pseudoinverse with the deterministic
symmetric-eigensolver tolerance

`eps * max(L.shape) * max(1, max(abs(eigenvalues)))`,

solve the zero-gauge auxiliary field

`phi = L^+ q`.

Fail open unless the projection of `q` onto every null mode is at most
`1e-8 * max(1, ||q||_2)`. This includes disconnected-component charge
imbalance. Require at least one positive Laplacian mode and one non-self
contact.

Freeze exactly three protected-low features:

1. `pcgr_charge_resistance = (q^T phi) / (q^T q)`;
2. `pcgr_voltage_drop_q90 = inverted_cdf_q90(|phi_i-phi_j| / RMS(q))`
   over directed non-self active-facet incidences;
3. `pcgr_voltage_drop_max = max(|phi_i-phi_j| / RMS(q))` over the same
   incidence population.

All outputs are quantized to the existing NEXT267 `1e-10` grid. Lower values
mean that the frozen charge injection is neutralized with lower graph
dissipation and smaller local auxiliary voltage drops; all three directions
are therefore frozen as `protected_low`.

The three retained summaries passed label-free rigid rotation, translation,
site permutation, and `2x1x1` supercell probes. Their internal Spearman
correlations were `0.60--0.82` in SCIGEN and `0.79--0.88` in WyFormer. Two
other label-free candidates are excluded before any outcome access:

- charge resistance divided by mean `1/lambda_k`, because it changed under a
  supercell representation;
- current participation, because it also changed under a supercell
  representation and was strongly redundant with graph size in SCIGEN.

Uniform coordinate scaling is not a required invariance because the frozen
tabulated radii give the radical tessellation a physical length scale.

## 4. Label-blind probe evidence and fixed coverage rule

An inventory-spanning deterministic 80-record sample from each discovery
geometry store was selected without reading endpoints. PCGR supported 79/80
SCIGEN and 77/80 WyFormer records; all four failures were pre-existing empty
radical cells. Every retained feature was nondegenerate.

Against 359 SCIGEN and 360 WyFormer old numeric features, maximum absolute
Spearman correlations were:

| new feature | SCIGEN | WyFormer |
|---|---:|---:|
| charge resistance | `0.551` | `0.574` |
| voltage-drop q90 | `0.601` | `0.588` |
| voltage-drop maximum | `0.687` | `0.577` |

The formal NEXT315 source coverage floor is fixed at `0.90` for each source.
No feature may be imputed. The formal build must process all 13,470 SCIGEN and
5,232 WyFormer discovery structures and publish exact failure counts before
opening outcomes.

## 5. Sequential protocol and stop rules

### Task 1: NEXT315 pure kernel and invariance tests

**Files:**

- Create: `tests/test_next315_periodic_charge_green_resistance.py`
- Create: `src/next315_periodic_charge_green_resistance.py`

**Steps:**

1. Write failing tests for exact two-site and charge-matched star graphs,
   invalid/null-space inputs, rigid equivalences, lattice rebase, supercell
   replication, geometry-only guards, and builder interface.
2. Run
   `python -m pytest -q tests/test_next315_periodic_charge_green_resistance.py`
   and require a missing-module failure.
3. Implement a pure `charge_green_resistance_features` kernel and the
   NEXT19/NEXT267/NEXT279-backed raw-geometry wrapper.
4. Run the focused test to green, then run adjacent NEXT19, NEXT267, NEXT279,
   NEXT295, and NEXT311 tests.

### Task 2: NEXT315 full label-free build

**Files:**

- Modify only the new NEXT315 source/test if required by failing tests.
- Create formal artifacts only under
  `$PRIS_ARCHIVE/next315_periodic_charge_green_resistance_v1`.

**Steps:**

1. Reuse the exact NEXT311 discovery-geometry provenance and payload readers.
2. Build all rows with 16 workers and no endpoint paths in the interface.
3. Require source coverage at least `0.90`, three finite features on every
   supported row, zero finite feature values on unsupported rows, and publish
   label-free quantiles/unique counts.
4. Record source/test/input/output SHA-256 identities and all no-DFT boundary
   flags before atomically publishing the directory.

### Task 3: NEXT316 fixed discovery audit

**Files:**

- Create: `tests/test_next316_pcgr_feature_audit.py`
- Create: `src/next316_pcgr_feature_audit.py`
- Create formal artifacts only under
  `$PRIS_ARCHIVE/next316_pcgr_feature_audit_v1`.

**Steps:**

1. Write the failing audit tests first.
2. Adapt the exact NEXT312 audit implementation without changing the NEXT224
   rejected-extreme cohort, source/fold cells, class counts, coverage floor,
   AUC gates, quantiles `(0.05, 0.95)`, or ranking/eligibility rule.
3. Audit exactly the three frozen `protected_low` hypotheses.
4. If zero hypotheses are eligible, set
   `next317_search_authorized = false`, terminate the PCGR branch, and do not
   create NEXT317 or NEXT318.
5. If at least one is eligible, publish its sorted digest and authorize only
   the frozen NEXT317 search below.

### Task 4: conditional NEXT317 one-term search

**Files (only if NEXT316 authorizes):**

- Create: `tests/test_next317_pcgr_margin_local_search.py`
- Create: `src/next317_pcgr_margin_local_search.py`
- Create formal artifacts only under
  `$PRIS_ARCHIVE/next317_pcgr_margin_local_search_v1`.

**Frozen grammar:**

- unchanged NEXT224 nonnegative base score and support;
- one signed triangular margin-local PCGR term;
- local width fractions `(1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1)`;
- amplitude fractions `(1/4, 1/2, 1)`;
- exactly `1 + 21 * H` candidates for `H` eligible hypotheses, including one
  no-op reproduction control;
- `ALL_FINITE_COMBINED_DISCOVERY` normalization, with no endpoint use in the
  normalization;
- missing term means keep the unchanged NEXT224 score and support.

If any new candidate passes all unchanged cross-source discovery gates, mark
only `freeze_authorized = true`; do not open validation/replication. If none
passes, authorize NEXT318 only for the deterministic population that passes
source AUC and all SAFE cells but fails BROAD.

### Task 5: conditional NEXT318 unchanged BROAD diagnostic

**Files (only if NEXT317 authorizes):**

- Create: `tests/test_next318_pcgr_broad_diagnostic.py`
- Create: `src/next318_pcgr_broad_diagnostic.py`
- Create formal artifacts only under
  `$PRIS_ARCHIVE/next318_pcgr_broad_diagnostic_v1`.

Reproduce the exact published NEXT317 diagnostic population and apply the
unchanged NEXT314/NEXT164 BROAD residual analysis. Compare failure count first
and normalized shortfall second against the frozen NEXT235 reference. Do not
search another threshold family. Close the PCGR branch unless it strictly
improves the reference; an improvement still requires a new pre-outcome freeze
and does not authorize validation.

### Task 6: independent report and verification

**Files:**

- Add a new section to
  `reports/2026-08-08-next115-next117-hcid-no-dft-search.md` only after the
  sequential branch reaches its authorized stop.

Report the formula, literature scope, probe exclusions, full coverage/failure
counts, all audit/search/diagnostic metrics, artifact hashes, and the exact
negative or positive conclusion. Explicitly distinguish a reporting-selected
AUC+SAFE representative from a frozen law.

Fresh verification must include focused and adjacent tests, the full pytest
suite, `git diff --check`, formal manifest/output/source hash verification,
CodeGraph status/search, and confirmation that canonical paper paths have no
changes. Do not claim a law, Pauling replacement, or DFT-like screening unless
all frozen gates and subsequent sealed validation evidence actually prove it.
