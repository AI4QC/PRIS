# NEXT551--NEXT554 blind HEA law search

## Objective and hard boundary

Search for an explicit pre-DFT screening law on the previously unused
high-entropy-materials dataset (Zenodo record 10854500).  The published CSV
contains 83,797 rows with both an initial unrelaxed structure and a final DFT
relaxed structure.  The executable law may use only composition and the one
raw, fully periodic, unrelaxed initial structure.  It may not use energies,
forces, stresses, final structures, trajectories, ML/MLIP outputs, or any
coordinate/cell relaxation.

All work is additive.  Existing scripts, reports, notes, and paper files are
preserved.  A standalone report is authorized only after NEXT554 passes its
frozen confirmation gates.

## Source firewall

Formal source: `hea.2023-04-06.csv`, 452,858,376 bytes, published MD5
`4754d35ac163bb8804ef2f24ace659f7`.  The only label-free columns allowed before
NEXT553 are `fid`, `reduced_formula`, `chemical_system`, `nelements`, `NIONS`,
and `structure_ini_as_dict`.

Use a byte-stream RFC4180 projector that copies only requested fields.  It may
scan delimiters and quoting in skipped fields but must not copy, decode,
convert, log, or expose their bytes.  The schema-audit prefix materialized 31
records whose initial structures are empty; list those FIDs as permanently
excluded even though they are ineligible anyway.

## NEXT551 label-free cohort

1. Scan label-free metadata and presence/absence of the initial-structure
   field.  Require 84,024 unique source FIDs and 83,797 eligible initial
   structures.
2. Define `ordered` as `NIONS in {2,...,8}` and `sqs` as
   `NIONS in {27,64,125}`.
3. Within each family select the 1,200 lowest
   `SHA256("NEXT551-cohort-v1|" + fid)` values.  No endpoint or final-structure
   value may affect selection.
4. Assign every chemical system wholly to `development` or `validation` with
   a deterministic label-free balancing pass.  Sort systems by decreasing
   selected-row count and then `SHA256("NEXT551-split-v2|" + chemical_system)`;
   assign each system to the side that lexicographically minimizes the maximum
   ordered/SQS count imbalance, total count imbalance, and finally disagreement
   with the hash-parity preferred side.  This split is frozen before any
   endpoint is opened.
5. In a second source pass, copy and decode only the selected
   `structure_ini_as_dict` values.  Strip all site properties and publish a
   deterministic geometry-only archive plus label-free metadata.

Blind gates: exactly 2,400 unique rows, 1,200 per size family, at least 900 rows
and 400 rows per size family in each partition, at least 100 chemical systems
per partition, all frames fully periodic and free of calculators/non-geometric
arrays, and at least 99% unique geometry hashes.

## NEXT552 frozen analytic feature bank

Compute the fixed NEXT43 analytic feature bank plus the eight NEXT546 primitive
geometry descriptors for every NEXT551 frame.  Add no learned energy, force,
stress, or stability proxy.  Before endpoint opening, convert every feature and
both risk directions to cohort-wide midrank percentiles.  Publish the raw and
ranked table and an exact feature catalogue.  Require at least 95% support in
both partitions, at least 100 distinct rounded values, and no greater than 10%
point mass for a direction to be searchable.

## Frozen DFT endpoints

For a row whose endpoint is authorized to open, decode only
`e_above_hull` and `structure_as_dict`.  Require identical site count and
ordered atomic numbers between initial and final structures.  Compute
minimum-image site displacement using the average initial/final cell, maximum
absolute logarithmic cell strain, and absolute volume log change.

Define before looking at values:

- energetic instability: `e_above_hull >= 0.10 eV/atom`;
- large geometric response: displacement p90 at least 0.25 A, cell log strain
  at least 0.08, or volume log change at least 0.10;
- primary `dft_waste`: energetic instability OR large geometric response;
- continuous severity: the maximum of the four endpoint values divided by
  their thresholds;
- protected: `e_above_hull <= 0.025`, displacement p90 at most 0.10 A, cell
  log strain at most 0.03, and volume log change at most 0.04.

## NEXT553 development-only bounded search

Open endpoint fields only for development FIDs.  Validation endpoint fields
must be skipped by the byte projector and remain unmaterialized.  Abort before
search if endpoint coverage is below 95%, either primary class has fewer than
100 rows, or either size family has fewer than 30 rows in either class.

For each label-free feature, evaluate both already-frozen risk directions.
Retain a direction only when its AUC is at least 0.60 in both ordered and SQS
development rows.  Prune absolute Spearman redundancy above 0.95 and retain at
most 16 directions.  Search every unordered retained pair with exactly four
coefficient-free symmetric formulas: mean, maximum, probabilistic union, and
minimum.  No weights, thresholds, or feature transformations may be added.

A pair is eligible to freeze only if it has at least 95% coverage; AUC at least
0.70 overall and 0.65 in both size families; cluster-bootstrap AUC lower bound
at least 0.62; top-15% precision at least 1.75 times prevalence; zero protected
rows in the top 15%; Spearman severity at least 0.30; and AUC at least 0.02
above both components in both size families.  Choose deterministically by the
minimum size-family AUC, then overall AUC, Spearman, and lexical formula ID.  If
none is eligible, NEXT554 is forbidden.

## NEXT554 sealed chemical-system confirmation

Only after an eligible NEXT553 formula and checksum freeze, open validation
endpoints and apply the formula without refitting ranks, components, weights,
thresholds, or the operating fraction.  Confirmation gates are at least 95%
coverage; at least 100 rows per class; AUC at least 0.70 with chemical-system
cluster-bootstrap lower bound at least 0.65; AUC at least 0.62 in ordered and
SQS rows; Spearman severity at least 0.30 with lower bound at least 0.20;
top-15% precision at least 1.75 times prevalence; protected top-15% fraction at
most 2%; and AUC at least 0.02 above the best component.  Pauling controls are
reported but unsupported Pauling rows cannot count as victories.

Passing NEXT554 supports only HEA-domain pre-DFT screening of the frozen DFT
waste endpoint.  It does not establish universal thermodynamic stability or
replacement of DFT; those require a further unseen-domain replication.
