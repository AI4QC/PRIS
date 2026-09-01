# Periodic Global Contact Equilibrium Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to
> implement this plan task-by-task.

**Goal:** Test whether a representation-invariant, global, positive contact-
equilibrium certificate computed only from composition and an initial raw
periodic geometry supplies a transferable pre-DFT crystal-validity signal.

**Architecture:** NEXT323 reuses the label-free periodic radical/power-cell
active-facet graph from NEXT279. It assigns one dimensionless dual coefficient
to every directed facet incidence, constrains reciprocal incidences to have the
same coefficient, and solves one fixed linear program that maximizes the
minimum participation of every incidence while all site resultants vanish.
NEXT324 opens discovery outcomes only as offline audit labels and applies the
unchanged NEXT224/NEXT268 cross-source gates. NEXT325/NEXT326 are contingent:
they may be created only if NEXT324 authorizes the frozen hypothesis.

**Tech Stack:** Python 3.11, NumPy, SciPy HiGHS linear programming, pandas,
ASE, the existing NEXT267/NEXT279 geometry pipeline, pytest, CodeGraph.

## Scientific and information-boundary freeze

For directed active-facet incidences `c = (i, j, s)` with unit displacement
`u_c`, let `bar(c) = (j, i, -s)` be the exact reciprocal incidence. With `M`
directed incidences, solve

```text
alpha* = max alpha
subject to
    sum_(c:center(c)=i) f_c u_c = 0       for every raw site i,
    f_c = f_bar(c)                         for every reciprocal pair,
    sum_c f_c = 1,
    f_c >= alpha / M                       for every c,
    f_c >= 0 and 0 <= alpha <= 1.
```

The sole protected hypothesis is
`pgce_all_facet_participation_floor = alpha*` in the `protected_high`
direction. `alpha*=1` means the uniform reciprocal coefficients already give
sitewise equilibrium; positive values certify an all-incidence equilibrium
with a finite participation margin. Coefficients are dimensionless dual
certificates, not predicted forces, stresses, energies, a potential, or a
physical relaxation. Coordinates and cells are never moved.

The mechanism is motivated by equilibrium stresses in periodic frameworks and
by linear-programming rigidity/jamming tests, including Donev, Torquato,
Stillinger, and Connelly, JCP 197 (2004), DOI
`10.1016/j.jcp.2003.11.022`, and Malestein and Theran, *Ultrarigid periodic
frameworks*, arXiv:`1404.2319`. These works motivate the dual equilibrium
construction; they do not make this statistic a theorem of crystal stability.

Executable inputs are limited to element identities, deterministic tabulated
radii, and one initial raw unrelaxed periodic geometry. The branch must not
execute or consume DFT calculations or per-structure DFT values; learned
energy/force/stress proxies; MLIPs; model/proxy potentials; relaxed structures;
trajectories; later geometries; validation outcomes; or replication outcomes.
Discovery outcomes may enter NEXT324 only as offline labels. Validation and
replication files remain physically unopened.

## Alternatives rejected before outcome access

1. A periodic rigidity spectral gap is rejected because NEXT20, NEXT37,
   NEXT168, and NEXT173 already cover global/affine rank and local directional
   rigidity.
2. A capacitated bond-valence circulation is rejected because NEXT38 and
   NEXT307 already cover transport compatibility and Hodge-loop frustration.
3. A low-wavevector charge structure factor remains rejected because its
   primitive/supercell convention is nontrivial and it overlaps earlier charge
   spectrum mechanisms.

No graph, direction, quantile, threshold, transform, conjunction, or second
PGCE feature may be added after discovery outcomes are opened.

## Frozen gates

- Label-blind probe: deterministically select 80 discovery geometries per
  source in memory after the complete identifier-bearing geometry inventory is
  opened; read no endpoint, label, validation, replication, relaxed geometry,
  DFT, or model-potential field.
- Probe engineering gates: exact schema; finite `[0,1]` values; support at
  least 72/80 per source; at least 20 quantized unique values per source; rigid
  rotation, periodic translation, site permutation, and exact integral
  supercell error at most `1e-8` before `1e-10` output quantization.
- Probe novelty gate: maximum absolute Spearman correlation below `0.90`
  against the available label-free NEXT267, NEXT279, NEXT295, NEXT299,
  NEXT303, NEXT307, NEXT311, NEXT315, and NEXT319 feature populations on the
  same records. If the feature fails any probe gate, terminate before NEXT323's
  formal full build and do not open discovery outcomes.
- Formal NEXT323 source coverage: at least `0.90` independently in SCIGEN and
  WyFormer, with unsupported structures retained as abstentions and never
  imputed.
- NEXT324 source gates, unchanged from NEXT207/NEXT224/NEXT268: minimum cell
  coverage `0.90`, minimum class count `20`, pooled AUC `0.55`, macro AUC
  `0.53`, and worst-fold AUC `0.50` in both sources using reduced-formula
  folds. Quantile normalization is frozen at inverse-CDF `1/16` and `15/16`.
- Any empty eligible set sets `next325_search_authorized=false` and
  `pgce_branch_terminated=true`; NEXT325/NEXT326 must not exist.

## Task 1: NEXT323 core and geometry wrapper

**Files:**

- Create: `src/next323_periodic_global_contact_equilibrium.py`
- Create: `tests/test_next323_periodic_global_contact_equilibrium.py`
- Reuse without modification: `src/next267_periodic_radical_voronoi_packing.py`
- Reuse without modification: `src/next279_radical_packing_autocorrelation.py`
- Reuse without modification: `src/next319_periodic_contact_shell_neutralization.py`

1. Write failing analytic tests for uniform balanced directions (`alpha=1`),
   an imbalanced but positive five-incidence equilibrium (`alpha=5/6`), and a
   one-sided infeasible population (`alpha=0`).
2. Run the new test file and record the expected import failure.
3. Implement strict array validation, exact reciprocal-pair validation, the
   HiGHS LP with `1e-10` primal/dual feasibility tolerances, independent
   `1e-9` residual checks, and `1e-10` quantization.
4. Add geometry-only guards and build the directed translated incidence table
   from NEXT279 contacts plus NEXT319's integer-translation recovery.
5. Add real-structure tests for NaCl/CsCl/ZnS, rigid rotation, periodic
   translation, site permutation, exact `2 x 1 x 1` replication, calculator /
   metadata / extra-array refusal, and deterministic repeated execution.
6. Run the test file to green and run Python byte compilation.

## Task 2: Label-blind probe and final pre-outcome freeze

**Files:**

- Create only if useful for reproducibility:
  `experiments/next323_pgce_label_blind_probe.py`
- Create only if the script is retained:
  `tests/test_next323_pgce_label_blind_probe.py`

1. Materialize the complete discovery geometry inventory required by the
   existing archive loaders, then select the frozen 80/source subset in memory.
2. Compute support, range, quantized uniqueness, invariance error, internal
   numerical residuals, and correlations with the listed label-free features.
3. Record exact probe statistics and the final design SHA-256 before opening
   any discovery outcome.
4. If a frozen probe gate fails, stop this branch and report the label-blind
   negative result. Do not change the feature or direction in this branch.

## Task 3: NEXT323 full label-free build

**Files:**

- Modify only the new file:
  `src/next323_periodic_global_contact_equilibrium.py`
- Modify only the new test:
  `tests/test_next323_periodic_global_contact_equilibrium.py`

1. Write failing builder tests for complete ID identity, strict geometry-only
   reads, unsupported-row abstention, exact manifests, and false boundary flags.
2. Implement the additive SCIGEN/WyFormer builder following NEXT319's
   multiprocessing and provenance pattern.
3. Run the formal build into
   `$PRIS_ARCHIVE/next323_periodic_global_contact_equilibrium_v1`.
4. Recompute manifest/output/source hashes, coverage, finite-value assertions,
   and boundary flags. Stop before outcomes if either source misses coverage.

## Task 4: NEXT324 fixed discovery-only audit

**Files:**

- Create: `src/next324_pgce_feature_audit.py`
- Create: `tests/test_next324_pgce_feature_audit.py`

1. Write failing tests for the exact one-hypothesis universe, protected-high
   normalization, unchanged gates/folds, authorization, empty-set stop, and
   validation/replication refusal.
2. Implement the audit by reusing NEXT268/NEXT320 gate and provenance helpers.
3. Open only the physically isolated discovery endpoints as offline labels and
   run into
   `$PRIS_ARCHIVE/next324_pgce_feature_audit_v1`.
4. Freeze the eligible-set digest. If empty, stop without NEXT325/NEXT326.

## Task 5: Contingent law search and BROAD diagnostic

**Files, only when NEXT324 authorizes search:**

- Create: `src/next325_pgce_margin_local_search.py`
- Create: `tests/test_next325_pgce_margin_local_search.py`
- Create: `src/next326_pgce_broad_diagnostic.py`
- Create: `tests/test_next326_pgce_broad_diagnostic.py`

1. Reuse the unchanged NEXT269 margin-local search space and fixed
   NEXT224/NEXT135 base score.
2. Permit only the authorized PGCE hypothesis; do not add transformations or
   reverse its direction.
3. Freeze the selected formula before any BROAD diagnostic.
4. Run NEXT326 only as a discovery diagnostic. Keep validation and replication
   sealed regardless of the BROAD result.

## Task 6: Verification and independent report

**Files:**

- Modify additively:
  `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`
- Do not modify: `paper/`, `tex/`, `notes/`, `README.md`, `PREREG.md`

1. Run focused tests, adjacent NEXT267--NEXT324 tests, `py_compile`, and the
   complete repository `pytest -q` suite.
2. Verify all artifact hashes and boundary flags, no forbidden directories,
   CodeGraph synchronization, and clean status for every protected canonical
   path.
3. Append a standalone PGCE section with the frozen formula, alternatives,
   literature boundary, probe evidence, formal results, hashes, and an explicit
   claim limit.
4. Do not edit the canonical report or paper until the user confirms.

## Execution note

The repository is an intentionally dirty shared checkout and the user requires
all existing scripts/content to remain. Therefore this plan overrides the
generic worktree/commit checkpoints from the process skill: work additively in
the current checkout, make no Git commit/merge/cleanup, and do not delegate to
subagents.
