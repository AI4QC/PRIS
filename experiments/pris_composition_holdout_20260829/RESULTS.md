# PRIS composition-held-out sensitivity: results

The frozen PRIS rules and thresholds were evaluated without any refitting. Only the physically isolated discovery/calibration law tables were read; lockbox and unlabeled structures were not part of this analysis.

## Overlap audit

Of 5,297 held-out experimental structures, 3,631 (68.55%) have a reduced composition seen in discovery and 1,666 (31.45%) do not. The corresponding damaged-structure counts are 2,406 and 1,206.

## Frozen Set 4

| Cohort | Experimental satisfaction | Damage detection |
|---|---:|---:|
| All held-out | 81.80% [80.50%, 83.08%] | 91.11% [90.16%, 92.06%] |
| Composition shared | 81.38% [79.73%, 82.99%] | 90.73% [89.53%, 91.92%] |
| Composition unseen | 82.71% [80.65%, 84.67%] | 91.87% [90.28%, 93.42%] |
| Chemical system unseen | 83.17% [80.68%, 85.54%] | 91.49% [89.52%, 93.38%] |

The composition-unseen point estimates do not decrease relative to the composition-shared subset: Set 4 satisfaction changes by +1.33 percentage points and damage detection by +1.14 percentage points.

## Set 4 damage classes on unseen compositions

| Class | n | Detection |
|---|---:|---:|
| S1 | 260 | 74.23% [68.97%, 79.39%] |
| S2 | 198 | 92.93% [89.23%, 96.41%] |
| S3 | 261 | 100.00% [100.00%, 100.00%] |
| S4 | 226 | 92.92% [89.33%, 96.04%] |
| S5 | 261 | 99.62% [98.84%, 100.00%] |

Intervals are percentile intervals from whole-cluster resampling. The analysis used 10,000 replicates with seed 20260829; composition is the cluster except for the chemical-system-unseen cohort, where the exact element set is the cluster.

This is an outcome-blind subgroup sensitivity analysis of the existing calibration partition, not a newly collected external holdout. `Chemical system` means the exact sorted element set; no broader, post-hoc notion of chemical family is claimed.

Missing frozen-law features retain the published convention of counting as satisfied. Feature coverage is therefore reported separately in `results/feature_coverage.csv`.
