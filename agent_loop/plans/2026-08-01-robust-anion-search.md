# Robust Anion-Aware Additive Law Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the existing law-discovery loop with discovery-only robust anion constraints and physically motivated fixed guards, seeking a balanced rule set without changing or overwriting any previous script, output, report, or manuscript file.

**Architecture:** Add a new `next4_*` layer on top of the physically isolated P1--P10 tables. First recompute corrected, consistently valenced P2/P6/P7/P9 descriptors after an independent pre-outcome audit found periodic-neighbour and graph-degree defects; substitute these corrected columns for the affected old columns without changing old caches. The new search then evaluates deterministic anion-by-fold constraints against the existing-loop reference and ranks feasible beam states by worst perturbation-kind rejection before pooled rejection. Calibration remains an adaptively reused diagnostic, the 295 DFT-relaxed set remains a falsification set, and the lockbox is never read.

**Tech Stack:** Python 3.10, pandas, NumPy, pytest, existing `better_law_search` and `next3_law_search` utilities, Parquet caches, JSON aggregate outputs, SHA-256 manifests.

## Scientific freeze for experiment `np-next-20260801d`

This section is frozen before any `next4` candidate outcome is computed. The
P2/P6/P7/P9 correction below is a pre-outcome amendment prompted by an independent
implementation audit; no `next4` descriptor values or search results had been
computed when it was added.

- Scope is additive. Create only new `next4_*` code/tests, a new output directory, and a new standalone report.
- Search only the physically isolated discovery real/bad tables. Do not read `lockbox_real.parquet`, `lockbox_bad.parquet`, `lockbox_eval.json`, or any monolithic all-split source.
- Reuse the prior one-sided/band threshold grid. Keep unaffected P1/P3/P4/P5/P8/P10 columns, exclude the old P2/P6/P7/P9 columns from the `next4` pool, and replace them one-for-one with corrected `p2c_`, `p6c_`, `p7c_`, and `p9c_` columns. Formula ranking is not rerun because the new changes are local structural descriptors/guards and do not provide a new preregistered within-formula ranking hypothesis.
- Use one dataset-independent formal-valence cascade for corrected real, perturbed, and 295 false-positive structures: `discriminate.guess_oxi`, then the repository's public `apply_rules.frac_oxi`, then `polymorph_rank2.balance` mapped back to sites. Record the selected source and failures; no dataset-specific branch is allowed.
- Correct P6 by discarding only neighbours at zero distance, retaining periodic images even when their original site index equals the centre index.
- Correct P7 by the same zero-distance rule. `p7c_an_contact_frac` uses every anion site in the denominator; a site with no same-species neighbour within 8 Angstrom contributes zero short contacts. `p7c_an_contact_min` is the minimum observed ratio, while the guard uses the directly observed boolean `p7c_an_short_contact_frac == 0`, so a censored continuous minimum is never interpreted as an exact distance.
- Correct P9 so each Lewis acidity/basicity denominator is the degree in the opposite-sign CrystalNN graph used by the mismatch edges, not the degree in the all-neighbour graph.
- Add two fixed, mechanism-derived guarded candidate families:
  - corrected P2 same-charge solid-angle candidates apply only when `p7c_an_short_contact_frac == 0`.
  - P1 bond-valence-local candidates apply only when `bvloc_parameter_exact_fraction > 0.9`.
- Fit the existing-loop reference at satisfaction floor `0.98` on discovery.
- Define eligible anions as those with at least 200 discovery-real rows. Assign four deterministic folds with `crc32(source_id) % 4`. Create one full-anion stratum plus every nonempty anion-by-fold stratum containing at least 50 rows.
- For every robust stratum require candidate satisfaction to be at least the paired existing-loop satisfaction in that same stratum (zero empirical drop on discovery). This is stricter than the frozen downstream gate of worst-anion drop no worse than `-0.01`.
- Run beam width 96, at most 12 rules, minimum incremental bad-pass reduction `0.0015`. Among feasible states rank lexicographically by minimum S1--S5 rejection, pooled rejection, real satisfaction, and then fewer rules/deterministic index order. Keep an efficiency branch for search diversity.
- Freeze the full-discovery rule set before inspecting calibration, false-positive, or LOKO outcomes. Calibration is explicitly historical/adaptively reused and cannot confirm the result.
- Apply unchanged downstream gates: satisfaction at least baseline minus `0.005`; pooled rejection gain at least `0.02` or minimum-kind gain at least `0.03`; worst shared-anion satisfaction delta at least `-0.01`; selected target and guard coverage at least `0.90`; DFT-relaxed pass-rate drop no worse than `-0.03`; signed LOKO outcomes disclosed without cancellation. Before viewing 295 pass/rejection outcomes, additionally require the selected rule set's joint required-feature coverage on all 295 rows to be at least `0.90`; the main denominator remains all 295 with unknown failing closed, and known-only is sensitivity analysis only.
- Success status is at most `development-promising`. No new lockbox opening is authorised by this plan.

### Task 1: Correct P2/P6/P7/P9 definitions and unify valence inference

**Files:**
- Create: `tests/test_next4_features.py`
- Create later: `src/next4_features.py`
- Create external caches later: `$PRIS_LAW_TABLES/desc4_{real,bad,fp}.parquet`

**Step 1: Write failing tests**

Use tiny periodic structures to prove that periodic self-images are retained, only
zero-distance neighbours are discarded, a no-contact anion contributes zero to the
P7 fraction denominator, and P9 uses opposite-sign graph degree. Add a valence-policy
test showing the same ordered cascade is used independent of dataset role.

**Step 2: Run tests to verify RED**

Run: `python -m pytest -q tests/test_next4_features.py`

Expected: import failure because `next4_features` does not exist.

**Step 3: Implement minimal corrected helpers and featurization CLI**

Reuse the existing structure readers and the unaffected Voronoi implementation, but
write only new corrected columns/caches. Emit row counts, valence-source counts,
failure reasons, coverage, and exact input/implementation hashes.

**Step 4: Run tests to verify GREEN, then build all three caches**

Do not inspect search/calibration outcomes while building descriptors.

### Task 2: Specify robust strata and ranking behavior

**Files:**
- Create: `tests/test_next4_law_search.py`
- Create later: `src/next4_law_search.py`

**Step 1: Write failing tests**

Add tests showing that:

1. deterministic fold assignment is stable under row reordering;
2. robust strata include full-anion and eligible anion-by-fold cells but exclude cells below 50 rows;
3. every floor equals the paired existing-loop satisfaction for that exact mask;
4. a candidate that violates any robust real stratum is rejected even when its pooled satisfaction passes;
5. the feasible result is selected by minimum-kind rejection before pooled rejection.

**Step 2: Run tests to verify RED**

Run: `python -m pytest -q tests/test_next4_law_search.py`

Expected: import failure because `next4_law_search` does not exist.

**Step 3: Implement the minimal search helpers**

Create `deterministic_real_folds`, `paired_robust_strata`, and `robust_pareto_beam` in `src/next4_law_search.py`. Reuse `LawCandidate` and `BeamResult`; do not modify `better_law_search.py`.

**Step 4: Run tests to verify GREEN**

Run the Task 1 pytest command. Expected: all Task 1 tests pass.

### Task 3: Add fixed physics guards without expanding the target vocabulary

**Files:**
- Modify: `tests/test_next4_law_search.py`
- Modify later: `src/next4_law_search.py`

**Step 1: Write failing tests**

Test that only corrected P2 same-charge solid-angle targets receive the fixed corrected-P7 no-short-contact guard, only P1 `bvloc_` targets receive the exact-parameter guard, missing guard values make the rule inapplicable, and fixed guards are serialized with their exact thresholds.

**Step 2: Run tests to verify RED**

Expected: missing fixed-guard candidate builder.

**Step 3: Implement the minimal candidate extension**

Build on `build_next3_candidate_sets`, add only fixed guarded copies, deduplicate them with the existing key, and retain the existing/additive pool separation.

**Step 4: Run tests to verify GREEN**

Expected: all `test_next4_law_search.py` tests pass.

### Task 4: Run the frozen full-discovery search

**Files:**
- Complete: `src/next4_law_search.py`
- Create output: `outputs/20260801_robust_anion_search/law_robust_anion.json`

**Step 1: Add CLI contract tests**

Test overwrite refusal, lockbox-free protocol fields, correct input hashing, and identifier-free aggregate schema.

**Step 2: Run tests to verify RED, implement CLI, then verify GREEN**

The CLI loads only:

- `$PRIS_LAW_TABLES/law_real.parquet`
- `$PRIS_LAW_TABLES/law_bad.parquet`
- `desc_real.parquet`, `desc_bad.parquet`
- `desc3_real.parquet`, `desc3_bad.parquet`
- `desc4_real.parquet`, `desc4_bad.parquet`
- `guards_real.parquet`, `guards_bad.parquet`

It must refuse output overwrite and record hashes once per input byte stream.

**Step 3: Run the frozen search once**

Run the new CLI at floor 0.98, width 96, four folds, minimum stratum size 50, and at most 12 rules. Do not inspect calibration until the selected discovery rule set has been written into the output object.

### Task 5: Falsify the frozen candidate

**Files:**
- Create: `src/next4_law_stability.py`
- Create: `src/next4_falsification.py`
- Create: `tests/test_next4_falsification.py`
- Create outputs under: `outputs/20260801_robust_anion_search/`

**Step 1: Write failing tests**

Test true held-kind removal, signed held-kind deltas, unknown-fails-closed behavior, and rule/report hash linkage.

**Step 2: Implement and run**

Run true S1--S5 LOKO refits using the same robust real constraints and run the frozen 295-structure false-positive evaluation. Do not select or alter rules from either result.

### Task 6: Verify artifacts and write a standalone report

**Files:**
- Create: `outputs/20260801_robust_anion_search/README.md`
- Create: `outputs/20260801_robust_anion_search/MANIFEST.sha256`
- Create: `reports/2026-08-01-robust-anion-search.md`

**Step 1: Run full verification**

Run the complete pytest suite, validate every JSON with `jq empty`, verify the manifest from inside its output directory, scan aggregate outputs for row identifiers, and confirm `git diff -- README.md PREREG.md paper notes tex` is empty.

**Step 2: Write the report**

Report discovery, reused-calibration, LOKO, and false-positive metrics separately; include all per-anion and per-kind deltas, exact selected rules, coverage, provenance, limitations, and whether every frozen gate passed. Explicitly state that one prior lockbox opening exists, this round did not access it, and no confirmation claim is possible.

**Step 3: Stop before canonical edits**

Do not modify the manuscript, canonical report, preregistration, README, or old artifacts. Wait for user confirmation after delivering the new standalone report.
