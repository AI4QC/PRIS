# P2/P3/P5 Descriptors on Physically Isolated Splits: Additive Search Design

**Date:** 2026-08-01 (frozen before any descriptor matrix or candidate outcome is inspected)

**Experiment ID:** `np-next-20260801`

**Status:** frozen. The searchable vocabulary in §3, the gates in §6, and the analysis
plan in §5 are fixed by this document before the search runs. Any deviation discovered
later is labelled post-freeze exploration.

## 1. Scope boundary

Identical to `np-next-20260731`: this experiment is additive. It must not modify or
overwrite `PREREG.md`, `README.md`, `src/README.md`, `paper/`, `notes/`, `tex/`, any
existing script, or any existing feature-store artifact. New code lives in new files.
Row-level caches are written outside the repository. Only aggregate, identifier-free
JSON and a new report are added to `outputs/` and `reports/`.

## 2. Relation to the previous round

`np-next-20260731` (report `reports/2026-07-31-better-laws-formulas.md`) ended in a
strict negative result for frozen-vocabulary laws and left four explicit gaps:

1. the monolithic loaders materialized lockbox rows' metadata before filtering
   (formal source-access failure);
2. P2 Voronoi solid-angle, P3 Hawthorne network residuals, and P5 strict iterative
   Hoppe diagnostics were named but not implemented;
3. the exact `bvparm2020` policy failed the 0.90 coverage gate, pushing P1 onto a
   post-freeze parameter fallback;
4. no genuinely untouched source/temporal holdout exists for the expanded formula.

This round closes gaps 1 and 2. Gap 3 is retained exactly as before (the exact branch
is recomputed as a diagnostic; the fallback branch stays labelled post-freeze
exploration). Gap 4 is not closed here; nothing in this round is `confirmed`.

## 3. Frozen descriptor vocabulary

All families use formal valences from composition-only `guess_oxi`
(`discriminate.guess_oxi`); `BVAnalyzer` remains banned. Per-site quantities are
aggregated over cation sites (`cat`) and anion sites (`an`) separately. Aggregates are
taken over sites with a defined value; site-level coverage is emitted as a diagnostic
column and is never searchable.

Searchable columns (61 total):

### P1 — bond-valence local triplet (recomputed, identical definitions to last round)

The 18 frozen columns
`bvloc_{cat,an}_{absolute_mismatch,effective_cn,vector_asymmetry}_{mean,q95,max}`
on CrystalNN topology. Computed under two parameter policies in one pass:

- `exact`: committed `bvparm2020.cif` tuples only. This branch emits only coverage
  diagnostics (`bvlocx_*`, not searchable) to re-test the 0.90 gate.
- `frozen-fallback`: exact tuple, then nearest frozen element-pair valence tuple, then
  pymatgen Brown generic. The 18 searchable `bvloc_*` columns come from this branch,
  which remains labelled post-freeze parameter exploration, exactly as last round.

### P2 — Voronoi solid-angle family (new this round)

From one `VoronoiNN` tessellation per structure (`get_all_voronoi_polyhedra`), per site
\(i\) with solid angles \(\Omega_{ij}\) to tessellation neighbours \(j\),
\(p_{ij}=\Omega_{ij}/\sum_k\Omega_{ik}\):

- `sa_effective_cn` \(=\exp(-\sum_j p_{ij}\log p_{ij})\) — O'Keeffe-style entropy CN;
- `sa_like_fraction` \(=\sum_{j:\,\mathrm{sign}(v_j)=\mathrm{sign}(v_i)} p_{ij}\) —
  solid-angle share of same-charge neighbours;
- `sa_max_fraction` \(=\max_j p_{ij}\) — dominant-neighbour share.

Searchable: `p2vor_{cat,an}_{sa_effective_cn,sa_like_fraction,sa_max_fraction}_{mean,q95,max}`
(18 columns). No fitted parameters; expected coverage ≈ 1.

### P3 — Hawthorne prior-bond-strength network (new this round)

Bond graph = CrystalNN topology restricted to opposite-sign site pairs (the same
neighbour pass as P1; no bond lengths enter this family). With site–bond incidence
\(A\) and target \(v_i = |V_i|\):

- `nnls_relres` \(=\min_{s\ge0}\|As-v\|_2/\|v\|_2\) — feasibility of a physical
  (nonnegative) bond-strength assignment;
- `minnorm_relres` \(=\|As^*-v\|_2/\|v\|_2\) for the minimum-norm \(s^*\);
- `pauling_gap` \(=\|s^*-s^{P}\|_2/\|s^{P}\|_2\), \(s^{P}\) = Pauling equal-strength
  \(z_i/\mathrm{CN}_i\) on the same graph — Hawthorne's critique as a number;
- `rank_deficiency` \(=(n_{bonds}-\mathrm{rank}(A))/n_{bonds}\);
- `unbonded_charged_fraction` — fraction of charged sites with zero opposite-sign
  neighbours in the topology;
- `site_relres_q95`, `site_relres_max` — quantiles of per-site NNLS residuals
  \(|(As^+)_i-v_i|/|V_i|\).

Searchable: `p3haw_{nnls_relres,minnorm_relres,pauling_gap,rank_deficiency,unbonded_charged_fraction,site_relres_q95,site_relres_max}`
(7 columns). No fitted parameters and no bond lengths; expected coverage ≈ 1 up to
solver failures, which are counted.

### P5 — strict iterative Hoppe diagnostics (new this round)

Neighbour sphere: all periodic neighbours within 8.0 Å
  (`Structure.get_all_neighbors`), weights \(w_{ij}=\exp(1-(d_{ij}/d_{\min,i})^6)\) with
  \(d_{\min,i}\) the shortest distance at site \(i\) inside the sphere.

  **Solver note (pre-outcome amendment, 2026-08-01):** Hoppe's alternating iteration is
  run in block-partition (Gauss-Seidel) form with damping 0.5 to
  \(\max|\Delta R|<10^{-4}\) Å, capped at 400 iterations; the fixed point and feature
  definitions are unchanged from the frozen text. The tolerance was loosened from the
  initially drafted \(10^{-6}\) after convergence profiling showed a stiff slow mode
  needing ~300+ steps; \(10^{-4}\) Å is two orders below any chemically meaningful
  radius difference. Non-convergence is always disclosed per structure.

- `econ_strict_i` \(=\sum_j w_{ij}\) over the sphere (Hoppe ECoN without a CrystalNN
  prefilter);
- `econ_delta_i` = strict minus the existing approximation recomputed on the CrystalNN
  neighbour set with the same weight formula (per site, same structure);
- MEFIR strict: Hoppe's alternating iteration for effective radii,
  \(R_i^{(t+1)}=\sum_j w_{ij}(d_{ij}-R_j^{(t)})/\sum_j w_{ij}\) over opposite-sign
  sphere neighbours, initialised at the repository's existing Shannon-radius table
  (same fallback semantics as `geom_feat.py`), at most 200 iterations to
  \(\max|\Delta R|<10^{-6}\) Å, values clipped to [0.2, 3.0] Å against divergence;
  `mefir_rel_i` \(=(R_i-r^{Shannon}_i)/r^{Shannon}_i\);
- `mefir_delta_i` = strict minus the existing one-sided approximation
  (\(\sum_j w_{ij}(d_{ij}-r^{Shannon}_j)/\sum_j w_{ij}\) on CrystalNN neighbours),
  per site.

Searchable: `p5hop_{cat,an}_econ_strict_{mean,max}` (4),
`p5hop_{cat,an}_econ_delta_{mean,max}` (4),
`p5hop_{cat,an}_mefir_rel_{mean,min,max}` (6),
`p5hop_{cat,an}_mefir_delta_{mean,max}` (4) — 18 columns.
Sites without opposite-sign sphere neighbours are excluded from the MEFIR system and
counted in a coverage diagnostic.

### Guard vocabulary

Unchanged from the existing loop: `mean_cn_cat`, `z_cat_max`, `cn_an_mean`, `n_el`,
`cat_an_ratio`, `fi`, `dchi` at quantiles 0.25/0.5/0.75, both directions.

## 4. Physical split isolation

A new builder (`src/next_isolate.py`) calls the existing loaders
(`rules_final.load(phys=True)`, `formula2.load(phys=True)`) exactly once, filters rows
to `split ∈ {discovery, calibration}` (real) / `psplit ∈ {discovery, calibration}`
(bad), asserts zero lockbox/unknown rows in the output, and writes isolated tables plus
SHA-256 manifests to the external cache. Every downstream step in this round loads only
the isolated tables, so after the builder runs, no code path can materialize a lockbox
row. Disclosure kept: the builder itself scans monolithic files (it is the single,
audited exception); its outputs are verified lockbox-free by assertion and by a
post-write re-read check.

## 5. Analysis plan

1. Build isolated tables; verify row counts against the frozen split sizes
   (real discovery 12,632 / calibration 5,297; bad 8,590 / 3,612; ranking
   discovery 3,268 / calibration 1,348 after the existing inner merges).
2. Reproduce the historical L1/L1′/L2/L3 calibration numbers
   (0.991882/0.289037, 0.989428/0.383721, 0.957901/0.612126, 0.917123/0.700443)
   from the isolated tables before accepting any new result.
3. Compute descriptors for the exact same real lineage (discovery+calibration,
   n_sites ≤ 80) and the exact deterministic S1–S5 perturbation lineage
   (`seed_of(parent)`, canonical kind order), plus the frozen 295 DFT-relaxed
   false-positive structures.
4. Law loop, unchanged from last round: one-sided + band + guarded candidates,
   thresholds from real discovery quantiles only, Pareto beam (width 24, max 12 rules),
   floors {0.99, 0.98, 0.95}, paired-anion guard variant, calibration as historical
   diagnostic. Two pools: `existing_loop` (old features) and
   `additive_p235_loop` (old + the 61 frozen columns).
5. True LOKO refits at floor 0.98 with all five signed changes and mean absolute
   change; no cancellation claims.
6. False-positive evaluation on the 295 DFT-relaxed structures, unknown fails closed.
7. Formula loop (secondary): same grouped nested sparse-linear protocol as last round
   on the isolated ranking table, same seed 20260731 for fold comparability, pools
   `existing` vs `additive` (existing + the 61 frozen columns), ≤ 7 terms, fixed
   inner-OOF abstention thresholds, paired group bootstrap.

## 6. Frozen success gates

Unchanged from `np-next-20260731`, with the source-access condition now evaluated
honestly under the isolated pipeline:

- **Law promising** iff at a matched real-satisfaction operating point
  (≥ baseline − 0.005): pooled exclusion +≥ 0.02 or minimum-kind exclusion +≥ 0.03;
  same direction in deterministic discovery sub-splits and historical calibration;
  real and perturbed descriptor coverage ≥ 0.90 for every selected feature;
  worst-anion satisfaction drop ≤ 0.01; DFT-relaxed false-positive pass-rate drop
  ≤ 0.03; and zero lockbox rows materialized anywhere downstream of the isolation
  builder.
- **Formula promising** iff group-equal outer accuracy improves the reproducible
  sparse-linear refit by ≥ 0.02 at full coverage or a fixed commitment target; not
  single-group driven; direction gate (all folds positive, or one negative fold within
  0.01); ≤ 7 terms; fold statistics saved; gap-stratified table and bootstrap interval
  reported regardless of sign.
- **Confirmed** remains reserved for a genuinely untouched source/temporal holdout or
  an authorised lockbox opening. Neither happens in this round.

## 7. Stopping rule

If no candidate passes its gate, the report is a negative-result report. Existing
reports and the paper remain unchanged regardless.
