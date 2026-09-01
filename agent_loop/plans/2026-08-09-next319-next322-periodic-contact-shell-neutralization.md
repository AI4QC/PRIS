# Periodic Contact-Shell Neutralization Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to
> implement this plan task-by-task with test-driven development. The repository
> is intentionally dirty and the user requires additive work, so do not create
> a worktree, commit, delegate to subagents, or modify canonical paper sources.

**Goal:** Test whether the rate at which formal charge neutralizes over exact
shells of the infinite periodic radical-contact cover supplies a transferable,
fully DFT-free crystal plausibility law beyond the existing Pauling/frontier
score.

**Architecture:** NEXT319 reconstructs integer translations for the reciprocal
NEXT279 active-facet contacts, performs an exact two-shell breadth-first search
on the infinite periodic cover, and publishes two frozen local-neutrality
summaries. NEXT320 applies the unchanged NEXT316/NEXT224 cross-source feature
audit. NEXT321 and NEXT322 are conditional: the same bounded one-term
margin-local search and, only when authorized, the unchanged BROAD residual
diagnostic.

**Tech Stack:** Python 3.11, NumPy, pandas/Parquet, ASE, pymatgen, existing
NEXT19/NEXT267/NEXT279/NEXT295 geometry and provenance guards, pytest.

---

## 1. Scientific question, alternatives, and novelty boundary

The unresolved failure is not merely cross-source ranking. Recent PCAE and
PCGR terms passed ranking and SAFE gates but could not simultaneously preserve
the protected SCIGEN population and reach the frozen WyFormer BROAD savings
bound. A new mechanism must therefore supply information that is local enough
to protect plausible structures yet extends beyond first-neighbor Pauling and
bond-valence summaries.

Three label-free mechanisms were compared before outcome access:

1. A low-wavevector formal-charge structure factor was rejected because exact
   primitive/supercell representation invariance would require a symmetry or
   reciprocal-grid convention. NEXT36/NEXT148 also already cover long-wave
   charge-spectrum information.
2. A periodic contact-shell charge-neutralization curve is selected. It is
   defined on the infinite translated contact cover, so it is independent of
   which exact supercell represents the crystal. It tests how quickly a site's
   formal charge is compensated as successive geometric contact shells are
   included.
3. Local charge multipole cancellation was rejected because cell-origin and
   polarization-branch choices introduce a gauge ambiguity absent from the
   selected definition.

The repository already contains formal-valence transport (NEXT19 and
NEXT119--NEXT125), Ewald/Madelung summaries, local opposite-sign cages
(NEXT299), reciprocal cage balance (NEXT303), bond-valence Hodge loops
(NEXT307), charge-alternation graph modes (NEXT311), and graph Green resistance
(NEXT315). CodeGraph found no `neutralization`, `cumulative_charge`,
`charge_shell`, or equivalent shell-residual implementation. PCSN does not
solve an optimization, graph Poisson equation, learned model, or physical
potential. It counts unique translated sites at exact unweighted contact
distance and evaluates their formal-charge cancellation.

The physical motivation is the local-charge-neutrality principle summarized
for bond-valence models by Brown
(<https://doi.org/10.1021/cr900053k>) and the exact importance of charge
screening/correlation constraints established by Stillinger and Lovett
(<https://doi.org/10.1063/1.1670358>). The discrete two-shell contact-cover
statistic below is our hypothesis. Neither source makes it a universal crystal
stability theorem, and no such claim is permitted.

## 2. Frozen information boundary

All executable features and candidate scores may read only:

- element identities and deterministic tabulated radii;
- the initial, unrelaxed, fully periodic geometry;
- deterministic neutral formal-valence assignment from NEXT19;
- the reciprocal active-facet contact incidences from NEXT279.

They must not:

- execute DFT or read any per-structure DFT value;
- read energy, force, stress, calculator, outcome, trajectory, relaxed
  structure, or model-potential payloads;
- use a learned energy/force/stress proxy, analytic pair potential, fitted
  charge, or fitted contact weight;
- perform structural relaxation or open validation/replication geometry;
- fit shell depth, feature directions, quantiles, normalizations, search
  widths, or amplitudes to discovery outcomes.

Discovery outcomes may be opened only in NEXT320 and later as offline labels.
Internal validation and replication geometry/endpoints remain physically
sealed. Canonical `paper/`, `tex/`, `notes/`, `README.md`, and `PREREG.md`
remain untouched pending user review.

## 3. Frozen PCSN construction

Let each NEXT279 directed active-facet incidence from site `i` to the periodic
image of site `j` be represented as `(i, j, s)`, where `s` is an integer
translation in `Z^3`. Recover `s` from the published Cartesian displacement
and raw positions/cell, round to the nearest integer, and require a maximum
fractional residual at most `1e-8`. Require exact reciprocal multiplicities:
every `(i, j, s)` must be paired with `(j, i, -s)`. Duplicate incidences to the
same translated node are collapsed for shell population only.

Define the infinite periodic contact cover with vertices `(i, t)`,
`t in Z^3`, and edges

`(i, t) -> (j, t + s)`.

For every raw root site `i`, start at `(i, 0)` and use exact unweighted
breadth-first distance. Let `B_h(i)` be the unique cover vertices at distance
at most `h`, including the root. For the neutral NEXT19 formal-charge vector
`q`, freeze

`R_h(i) = |sum_(j,t in B_h(i)) q_j| / sum_(j,t in B_h(i)) |q_j|`,

for exactly `h = 1, 2`. The denominator must be finite and positive. Require a
nonempty new shell at both depths for every root and enforce explicit
population guards to prevent pathological expansion. Freeze the guard at
most `4,096` unique cover vertices through depth two for any root; exceeding
it makes the record unsupported and does not permit truncation or imputation.

Freeze exactly two protected-low features:

1. `pcsn_shell1_residual_q90 = inverted_cdf_q90_i R_1(i)`;
2. `pcsn_shell2_residual_q90 = inverted_cdf_q90_i R_2(i)`.

Use the existing inverted-CDF convention and quantize both outputs to the
NEXT267 `1e-10` grid. Lower residual means faster local formal-charge
neutralization, so both directions are frozen as `protected_low`. Uniform
coordinate scaling is not a required invariance because tabulated radii make
the radical tessellation a physical-length construction.

The shell-2 maximum is excluded before outcome access: its Spearman
correlation with shell-2 q90 was `0.9209` in the SCIGEN probe and `0.9886` in
the WyFormer probe. No signed overscreening, ratio, fitted shell depth, or
alternate threshold is authorized.

## 4. Label-blind evidence and formal coverage rule

The deterministic probe selected 80 discovery records per source by sorting
only label-free inventory fields `(natoms, chemical_system, material_id)` and
taking evenly spaced indices. It opened the complete geometry inventories only
to satisfy their identity-checking lockbox APIs, then computed the selected
records in memory. It did not open an endpoint, label, validation or
replication payload, relaxed structure, DFT value, or model potential.

PCSN supported 78/80 SCIGEN and 76/80 WyFormer probe records. Rigid rotation,
translation, site permutation, and exact `2 x 1 x 1` supercell representation
all changed each retained feature by exactly zero in the probe. Both retained
features were nondegenerate:

| source | feature | minimum | median | q90 | maximum | unique at 1e-10 |
|---|---|---:|---:|---:|---:|---:|
| SCIGEN | shell-1 q90 | `0.066667` | `0.329702` | `0.565217` | `1.000000` | 61 |
| SCIGEN | shell-2 q90 | `0.015385` | `0.164179` | `0.314286` | `0.521368` | 71 |
| WyFormer | shell-1 q90 | `0.000000` | `0.465740` | `0.801961` | `1.000000` | 60 |
| WyFormer | shell-2 q90 | `0.015385` | `0.213069` | `0.374345` | `1.000000` | 72 |

Against label-free formal feature tables from NEXT267, NEXT279, NEXT295,
NEXT299, NEXT303, NEXT307, NEXT311, and NEXT315, maximum absolute Spearman
correlations were:

| retained feature | SCIGEN | WyFormer |
|---|---:|---:|
| shell-1 q90 | `0.761` | `0.765` |
| shell-2 q90 | `0.485` | `0.695` |

The retained pair had internal Spearman correlation `0.471` in SCIGEN and
`0.709` in WyFormer. The formal NEXT319 source coverage floor is frozen at
`0.90` for each source. No feature may be imputed. The formal build must
process all 13,470 SCIGEN and 5,232 WyFormer discovery structures and publish
exact failure counts before outcomes are opened.

## 5. Sequential implementation and stop rules

### Task 1: NEXT319 pure kernel and invariance tests

**Files:**

- Create: `tests/test_next319_periodic_contact_shell_neutralization.py`
- Create: `src/next319_periodic_contact_shell_neutralization.py`

**Steps:**

1. Write failing tests for translated two-site covers, reciprocal-contact
   validation, exact shell populations/residuals, deterministic q90,
   translation-recovery guards, rigid rotation, translation, site permutation,
   exact supercell representation, geometry-only inputs, and builder boundary.
2. Run
   `python -m pytest -q tests/test_next319_periodic_contact_shell_neutralization.py`
   and require a missing-module collection failure.
3. Implement the minimal pure cover kernel, raw-geometry wrapper, builder,
   atomic publication, hashes, label-free statistics, and boundary flags.
4. Run focused and adjacent NEXT19/NEXT267/NEXT279/NEXT295/NEXT311/NEXT315
   tests to green.

### Task 2: NEXT319 full label-free build

**Formal output:**

`$PRIS_ARCHIVE/next319_periodic_contact_shell_neutralization_v1`

**Steps:**

1. Reuse the exact NEXT315 physically isolated discovery-geometry provenance
   and full-inventory payload readers.
2. Build all rows with 16 workers and no endpoint path in the interface.
3. Require coverage at least `0.90` per source, both features finite on every
   supported row, no finite feature imputation on unsupported rows, and at
   least two distinct values per feature.
4. Record source/test/input/output SHA-256 identities and all no-DFT boundary
   flags before atomically publishing the new directory.

### Task 3: NEXT320 fixed discovery audit

**Files:**

- Create: `tests/test_next320_pcsn_feature_audit.py`
- Create: `src/next320_pcsn_feature_audit.py`
- Formal output:
  `$PRIS_ARCHIVE/next320_pcsn_feature_audit_v1`

**Steps:**

1. Write the failing audit tests first.
2. Adapt the exact NEXT316 audit without changing the NEXT224
   rejected-extreme cohorts, fixed source/fold cells, class counts, quantiles
   `(0.05, 0.95)`, coverage floor, pooled/macro/worst-lattice AUC gates, or
   deterministic eligibility rule.
3. Audit exactly the two frozen `protected_low` hypotheses.
4. If none is eligible, publish `next321_search_authorized = false`, close the
   PCSN branch, and do not create NEXT321 or NEXT322.
5. If one or more are eligible, publish their sorted digest and authorize only
   the frozen NEXT321 search below.

### Task 4: conditional NEXT321 one-term margin-local search

**Files (only if NEXT320 authorizes):**

- Create: `tests/test_next321_pcsn_margin_local_search.py`
- Create: `src/next321_pcsn_margin_local_search.py`
- Formal output:
  `$PRIS_ARCHIVE/next321_pcsn_margin_local_search_v1`

**Frozen grammar:**

- unchanged NEXT224 nonnegative base score, threshold, and support;
- one signed triangular margin-local PCSN term;
- local width fractions `(1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1)`;
- amplitude fractions `(1/4, 1/2, 1)`;
- exactly `1 + 21 * H` candidates for `H` eligible hypotheses, including one
  exact no-op reproduction control;
- `ALL_FINITE_COMBINED_DISCOVERY` feature normalization, with no endpoint use
  in the normalization;
- missing PCSN term means retain the unchanged NEXT224 score and support.

If a new candidate passes every unchanged cross-source discovery gate, mark
only `freeze_authorized = true`; do not open validation or replication. If no
candidate passes all gates, authorize NEXT322 only for the deterministic
population that passes both source AUC gates and all SAFE cells while failing
BROAD.

### Task 5: conditional NEXT322 unchanged BROAD residual diagnostic

**Files (only if NEXT321 authorizes):**

- Create: `tests/test_next322_pcsn_broad_diagnostic.py`
- Create: `src/next322_pcsn_broad_diagnostic.py`
- Formal output:
  `$PRIS_ARCHIVE/next322_pcsn_broad_diagnostic_v1`

Reproduce the exact NEXT321 record population and candidate universe. Apply
the unchanged NEXT318/NEXT314 BROAD residual analysis, comparing failed
constraint count first and normalized shortfall sum second against the frozen
NEXT235 reference. Do not search a new threshold family or formula. Close the
PCSN branch unless it strictly improves the reference; even a diagnostic
improvement requires a new pre-outcome freeze and does not authorize opening
validation.

### Task 6: independent report and fresh verification

**Files:**

- Add a new PCSN section to
  `reports/2026-08-08-next115-next117-hcid-no-dft-search.md` only after the
  sequential branch reaches its authorized stop.

Report the formula, literature scope, label-blind exclusions, full formal
coverage/failure counts, all audit/search/diagnostic metrics, artifact hashes,
and the exact negative or positive conclusion. A deterministic reporting
representative is not a frozen law.

Fresh verification must include focused and adjacent tests, the complete
pytest suite, Python compilation, manifest/output/source hash verification,
CodeGraph status/context, and confirmation that canonical paper paths have no
changes. Do not claim a Pauling replacement, DFT-like screening, or a crystal
stability theorem unless every frozen gate plus subsequent sealed validation
and replication evidence actually proves it.
