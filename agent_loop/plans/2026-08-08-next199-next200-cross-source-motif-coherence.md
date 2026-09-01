# NEXT199 Cross-Source Motif Features and NEXT200 Audit Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add and test a genuinely new initial-geometry information axis—local coordination-motif coherence—that has not entered the current SCIGEN/WyFormer law loop, while preserving the strict pre-DFT and never-read validation boundary.

**Architecture:** NEXT199 reuses the frozen NEXT46 CrystalNN/Voronoi motif implementation to compute exactly 21 x0-only features from only the physically isolated discovery geometry files for SCIGEN and blind WyFormer. NEXT200 merges those identifier-bearing feature tables with discovery endpoints only after the feature outputs are frozen, then audits one prospectively assigned physical direction per feature on the exact NEXT164 repair shell and full extremes. No formula or cutoff is searched unless a feature passes every cross-source gate.

**Tech Stack:** Python 3.11, ASE, pymatgen, matminer CrystalNNFingerprint, NumPy, pandas/Parquet, multiprocessing, pytest, and existing NEXT46/NEXT84/NEXT93b/NEXT151/NEXT164 evaluation helpers.

## Frozen scientific design

NEXT197/NEXT198 ruled out a hard decision exception built from the existing signed-local/closure information. Its closest residual was `0.799874`, still failing the same six SCIGEN protected-retention constraints. Earlier NEXT151--NEXT165 work already ruled out top-two trimming, fixed capping, hard family concurrence, smooth family attenuation, and family identity. A new feature family, rather than another transform of the same contributions, is required.

NEXT46 implemented 21 geometry-only motif features, but they were evaluated only in an older Alexandria development loop and are absent from both current NEXT85 and NEXT94 catalogues. NEXT199 freezes their first SCIGEN/WyFormer cross-source materialization. The exact feature universe is `src.next46_motif_coherence_features.FEATURE_NAMES`:

```text
motif_weight_sum_mean
motif_weight_sum_min
motif_weight_sum_std
motif_cn_dominance_mean
motif_cn_dominance_min
motif_cn_dominance_std
motif_cn_entropy_mean
motif_cn_entropy_q95
motif_effective_cn_mean
motif_effective_cn_std
motif_effective_cn_range
motif_order_strength_mean
motif_order_strength_min
motif_order_strength_std
motif_fingerprint_norm_mean
motif_fingerprint_norm_std
motif_same_element_dispersion_rms
motif_same_element_dispersion_q95
motif_same_element_dispersion_max
motif_global_dispersion_rms
motif_species_centroid_separation_mean
```

NEXT199 may read only:

```text
NEXT84 MANIFEST.json, scigen_x0_metadata.parquet, geometry_discovery.zip
NEXT93b MANIFEST.json, wyformer_x0_metadata.parquet,
    wyformer_x0_geometry_discovery.parquet
```

The internal-validation and internal-replication geometry files are neither path arguments nor manifest inputs. NEXT199 has no endpoint argument. It publishes one discovery feature table per source, a catalogue, and a manifest; identifier-bearing tables remain outside the repository.

NEXT200 preregisters exactly one protection direction per feature. Coherent structures are expected to have higher completeness, dominance, effective CN, motif strength, fingerprint norm, and species-centroid separation, but lower dispersion and entropy:

```text
high protection: weight_sum_mean, weight_sum_min,
                 cn_dominance_mean, cn_dominance_min,
                 effective_cn_mean,
                 order_strength_mean, order_strength_min,
                 fingerprint_norm_mean,
                 species_centroid_separation_mean

low protection:  weight_sum_std, cn_dominance_std,
                 cn_entropy_mean, cn_entropy_q95,
                 effective_cn_std, effective_cn_range,
                 order_strength_std, fingerprint_norm_std,
                 same_element_dispersion_rms,
                 same_element_dispersion_q95,
                 same_element_dispersion_max,
                 global_dispersion_rms
```

There are exactly 21 hypotheses. Opposite signs, interactions, source-specific directions, endpoint-derived transforms, empirical cutoffs, and extra motif summaries are forbidden after outcomes are opened.

Audit populations and gates are unchanged from NEXT194:

```text
repair shell: BROAD <= exact NEXT164 base score < SAFE, protected <= 1,
              severe >= 2
full extremes: protected <= 1 or severe >= 2

SCIGEN full support >= 0.90
WyFormer full support >= 0.90
SCIGEN shell worst-fold AUC >= 0.55 with all 5 folds evaluable
WyFormer shell pooled AUC >= 0.55
SCIGEN full pooled AUC >= 0.50
WyFormer full pooled AUC >= 0.50
```

Eligible hypotheses are ranked by decreasing minimum of the four key AUCs, decreasing mean, then lexical hypothesis name. NEXT200 searches no formula and opens no validation/replication endpoint. If at least one hypothesis is eligible, NEXT201 requires a new result-bound frozen design before any formula search.

Every motif quantity is computed from species, cell, and coordinates of the raw unrelaxed x0 structure. No DFT calculation/value, learned energy/force/stress proxy, relaxed structure, trajectory, or physical relaxation is used. Discovery endpoints enter NEXT200 only as offline labels after NEXT199 publication.

### Task 1: TDD and build NEXT199

**Files:**

- Create: `tests/test_next199_cross_source_motif_features.py`
- Create: `src/next199_cross_source_motif_features.py`
- Create atomically: `$PRIS_ARCHIVE/next199_cross_source_motif_features_v1/`

Test the exact 21-feature schema, finite/error-row behavior, source parsers, discovery-only interface, absence of validation/replication parameters, deterministic row identity, and missing-input failure. Confirm RED before implementation. Bind exact cohort manifests, metadata, discovery geometry, NEXT46 source, plan, and executed source hashes. Do not materialize any other partition.

### Task 2: TDD and run NEXT200

**Files:**

- Create: `tests/test_next200_cross_source_motif_audit.py`
- Create: `src/next200_cross_source_motif_audit.py`
- Create atomically: `$PRIS_ARCHIVE/next200_cross_source_motif_audit_v1/`

Test exact hypothesis identities/directions, unchanged eligibility gates, deterministic selection, discovery-only endpoint interface, and missing-input failure. Reconstruct the exact NEXT164 base score and frozen folds, merge only NEXT199 discovery rows, evaluate the 21 hypotheses, and publish the audit plus Parquet table.

### Task 3: Result branch and verification

If zero hypotheses are eligible, terminate motif coherence without a search. If one or more are eligible, freeze a separate NEXT201 formula-search plan before inspecting any threshold result. In either case, update only the standalone report, run target and full tests, compile sources, verify formal hashes and no-DFT flags, confirm validation/replication remain unopened, check canonical zero-diff, and keep the persistent goal active unless all required discovery and unseen-data gates are truly passed.
