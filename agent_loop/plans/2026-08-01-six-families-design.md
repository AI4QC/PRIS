# Six Descriptor Families: Additive Search Design (round np-next-20260801c)

**Date:** 2026-08-01 (frozen before any descriptor matrix or candidate outcome of this
round is inspected)

**Experiment ID:** `np-next-20260801c`

**Status:** frozen. The searchable vocabulary in §2, the guard vocabulary in §3, and
the gates in §5 are fixed before the search runs.

## 1. Scope and evidence basis

Additive again: no existing file is modified, including the np-next-20260731 /
np-next-20260801 / np-next-20260802 additions. New code in `next3_*` files.

The six families were selected against the project's own negative evidence:
symmetry features are rejected (perturbation-generator fingerprint, +8.3–21.8 pt
false-positive gap on DFT-relaxed candidates, documented in `notes/结果总结.md`);
composition-only features have identically zero exclusion (guard-only); deep
trees/symbolic search overfit perturbation kinds. Every family below is local,
literature-anchored, and must survive the 295 DFT-relaxed gate and true LOKO.

The lockbox is **not** touched in this round: the authorised opening
(opening_index=1) covered the frozen V4 evaluation only. Nothing in this round reads
`lockbox_*.parquet` or any lockbox row.

## 2. Frozen descriptor vocabulary (33 new searchable columns)

Formal valences from `guess_oxi` throughout; `BVAnalyzer` banned. Aggregates over
sites with defined values, cation (`cat`) and anion (`an`) sites separately.

### P4 — ChemEnv continuous symmetry measure (the deferred P4 from round 1)

Per cation site, CSM of the coordination environment computed with the repository's
exact recipe (`build_bonds.py`): `LocalGeometryFinder(centering_type="centroid",
include_central_site_in_centroid=True, STRUCTURE_REFINEMENT_NONE)`,
`compute_structure_environments(only_cations=True, valences=rounded formal,
maximum_distance_factor=1.41)`, `LightStructureEnvironments` under
`MultiWeightsChemenvStrategy.stats_article_weights_parameters()`, CSM of the
best-matching ideal polyhedron (`coordination_environments[i][0]["csm"]`).

Searchable: `p4csm_cat_{mean,q95,max}` (3). Coverage diagnostic only for anions
(not computed).

### P6 — coordination shell gap (cutoff-free bondedness)

Per site, distances to all periodic neighbours within 8.0 Å sorted
\(d_1\le d_2\le\dots\); ratios \(r_k=d_{k+1}/d_k\) for \(k=1..\min(m-1,12)\);
`gap_ratio` \(=\max_k r_k\); `gap_pos` \(=\arg\max_k r_k\) (the cutoff-free CN);
`shell_width` \(=d_{\mathrm{gap\_pos}}/d_1\).

Searchable: `p6gap_{cat,an}_{gap_ratio,shell_width}_{mean,max}` (8),
`p6gap_{cat,an}_gap_pos_{mean,max}` (4) — 12.

### P7 — polyanion / homonuclear anion-contact detector

For each anion species X, the minimum X–X distance within the 8 Å sphere divided by
\(2 r_{\mathrm{atom}}(X)\). **(Pre-outcome amendment, 2026-08-01:** pymatgen 2026
removed `Element.covalent_radius`; the frozen table is `Element.atomic_radius`, the
remaining fixed unfitted length scale. Separation is unchanged: azide N–N ≈ 0.88,
peroxide O–O ≈ 1.2, ordinary ionic O–O ≈ 2.1.**)** Structure level:
`p7poly_an_contact_min` (minimum over all same-species anion pairs),
`p7poly_an_contact_frac` (fraction of anion sites with a contact ratio < 1.3).
Searchable: 2 columns.

### P8 — neighbour-algorithm disagreement

Per site, the Jaccard distance between the CrystalNN neighbour set and the Voronoi
face set (all tessellation faces, no threshold), both as sets of original site
indices. Searchable: `p8nnj_{cat,an}_jaccard_{mean,max}` (4).

### P9 — Hawthorne R3 Lewis acid–base matching

On the CrystalNN bond graph (opposite-sign pairs): Lewis acidity
\(a_i=z_i/\mathrm{CN}_i\), basicity \(b_j=|z_j|/\mathrm{CN}_j\); bond mismatch
\(|a_i-b_j|\). Searchable: `p9lew_bond_mismatch_{mean,q95,max}` (3) plus
`p9lew_cat_site_mismatch_max` (per-cation-site mean of bond mismatches, then max)
(1) — 4.

### P10 — Voronoi free volume and off-centering

Per site from the tessellation: cell volume \(V_i\) (sum of face pyramid volumes);
`freevol` \(=V_i/(\tfrac43\pi r_{\rm Shannon}^3)\) with the repository's existing
Shannon table (same fallback semantics as the MEFIR family); `offcenter`
\(=\lVert\sum_j \Omega_{ij}\hat n_{ij}\rVert/\sum_j\Omega_{ij}\), the
solid-angle-weighted face-normal asymmetry. **(Pre-outcome amendment, 2026-08-01:**
in pymatgen 2026 the face `verts` field carries vertex *indices* without
coordinates, so the drafted cell-centroid definition is not computable from this
API; the normal-asymmetry form is the same physical quantity — a centred site
cancels to zero.**)** Searchable: `p10vor_{cat,an}_{freevol,offcenter}_{mean,max}`
(8).

## 3. Guard vocabulary

The ten frozen columns of np-next-20260802 (`mean_cn_cat`, `z_cat_max`,
`cn_an_mean`, `n_el`, `cat_an_ratio`, `fi`, `dchi`, `z_an_abs`, `oxi_n_guesses`,
`oxi_unique`) **plus** `p7poly_an_contact_min` (structural, evaluated per row).
Guards at quantiles 0.25/0.5/0.75, both directions, as before.

## 4. Analysis plan

1. Featurize the isolated real/bad lineages and the 295 DFT-relaxed set with the
   six families (new `next3_*` caches; P1–P5 caches from np-next-20260801 reused
   unchanged; guard tables from np-next-20260802 reused unchanged).
2. Identical law loop (floors 0.99/0.98/0.95, min coverage 0.90, width 24, max 12
   rules, paired-anion-guarded variant), pools: `existing_loop` vs additive
   (existing + 61 previous + 33 new = 94 frozen new columns).
3. True LOKO refits at 0.98; 295 DFT-relaxed falsification, unknown fails closed.
4. Formula loop is not re-run (the previous round's negative stands; the ranking
   target is noise-floor-limited, PREREG limitation 4).
5. Report; if any candidate passes every frozen gate, it is a candidate for a
   **new** lockbox opening (2 remain) — not opened in this round.

## 5. Frozen success gates

Identical to np-next-20260801/02 (matched satisfaction −0.005; pooled +0.02 or
min-kind +0.03; direction consistency; coverage ≥ 0.90; worst-anion ≥ −0.01;
DFT-relaxed pass drop ≤ 0.03; zero lockbox materialization downstream of the
isolation builder). `confirmed` remains reserved.

## 6. Stopping rule

No gate pass ⇒ negative-result report. Existing reports and the paper unchanged.
