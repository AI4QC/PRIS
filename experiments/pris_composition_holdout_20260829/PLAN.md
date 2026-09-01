# PRIS composition-held-out sensitivity analysis

## Scope

This is an additive evaluation of the already frozen PRIS predicates. It does
not search for laws, alter thresholds, contact the lockbox, or modify any
manuscript, Supplementary Information, canonical report, or existing figure.

## Population and grouping

1. Read the pre-partitioned PRIS law tables containing only `discovery` and
   `calibration` rows.
2. Map every real `source_id` and every damaged `parent` to the unique formula
   and chemical system in `provenance.parquet`.
3. Canonicalize formulae with the exact-stoichiometry reduction already used by
   `src.next6_wbm_protocol.reduced_formula_key`.
4. Define, without using outcomes:
   - `heldout_all`: every calibration row;
   - `composition_shared`: calibration compositions present in discovery;
   - `composition_unseen`: calibration compositions absent from discovery;
   - `chemical_system_unseen`: calibration element sets absent from discovery.
5. Damaged structures inherit the group of their parent. Assert that parent
   and damaged split labels agree.

## Frozen endpoints

Evaluate Set 1, Set 1-prime, Set 2, Set 3 and Set 4 exactly at their published
thresholds and missing-value convention. Report experimental-structure
satisfaction and damaged-structure detection, both row-weighted and with each
composition weighted equally. Reproduce the published all-held-out values
before accepting any subgroup result.

## Uncertainty and outputs

Use a deterministic composition-cluster bootstrap (10,000 replicates, seed
20260829) for 95% percentile intervals. Save machine-readable cohort counts,
overall metrics, per-damage-class metrics and provenance fingerprints. Produce
two SI-ready figures: a three-panel overview of overlap and frozen-set transfer,
and a Set 4 per-damage-class sensitivity plot. Export PDF, SVG and 400-dpi PNG.

## Verification gates

- Formula multiples map to one composition key.
- Unknown, missing and lockbox split labels are rejected.
- Every damaged row inherits its parent's metadata and split.
- Frozen masks reproduce the five published calibration points to numerical
  precision.
- The run is deterministic and all plotted numbers come from emitted CSVs.
- `git diff` confirms that only this independent experiment directory changed.
