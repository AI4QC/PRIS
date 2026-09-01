# NEXT543--NEXT545: Li--Si random-DFT-relaxation confirmation

Date frozen: 2026-08-13

## Purpose and immutable boundary

This branch prospectively confirms whether the coefficient-free NEXT541 MUPR
screen can identify expensive or unproductive DFT random-structure relaxations
from the raw initial cell alone.  The executable screen may read only species,
cell, and coordinates of the first unrelaxed periodic structure.  It may not
read or use DFT energies, forces, stresses, convergence states, later structures,
trajectories, ML predictions, potentials, or physically/virtually relaxed
coordinates.  DFT information is opened only as an offline endpoint after all
scores and gates have immutable hashes.

All work is additive.  Existing scripts, reports, and paper files remain
unchanged.  A new independent success report is authorized only if every
confirmatory gate passes.

## Source and permanent audit exclusion

The source is the public Google Research bucket
`gs://gresearch/crystal-relaxations/`, accompanying Cheon et al., *Dataset of
Random Relaxations for Crystal Structure Search of Li-Si System*.  The paper
reports 116,200 VASP relaxations from AIRSS random initial cells and deliberately
retains electronic/ionic non-convergence cases.

`Li13Si4_02` was used only to learn the public JSON and summary schemas.  Its
entire prefix is permanently excluded from discovery, score normalization,
validation, and reporting.  Audit files and hashes are:

- `Li13Si4_02/data/02_10000.json`:
  `9fb5239fa41870d3d2ee2f865b0257fa2ca31cd981eb823a4d2143546b7a1df6`
- `Li13Si4_02/summary.txt`:
  `4e0905d0fe7748e364271a6e1a718c71f066d44a1bc67ab14c91c9b04b4b2ded`

No summary, convergence status, final energy, or later structure has been read
from the four confirmation prefixes before this design freeze.

## Cohort selection before labels

The four confirmation prefixes are frozen:

- `Li1Si1_02`
- `Li2Si1_02`
- `Li7Si2_03`
- `Li15Si4_02`

NEXT543 lists public GCS object metadata under each `<prefix>/data/` directory.
Objects must be non-empty JSON files because empty objects contain no recoverable
initial structure.  This is a data-availability condition, not an endpoint
decision.  Within each prefix objects are ordered by

```
SHA256("NEXT543-v1|" + full_object_name)
```

and the first 50 are selected, for exactly 200 trajectories.  Selection cannot
use object size ordering, summary rows, energies, convergence messages, number
of steps, or any structure-derived feature.  Remote generation, size, MD5, and
name are frozen in the inventory before endpoint access.

## Initial-only extraction firewall

Each public JSON has top-level arrays for forces, stress, energy, and structure.
The initial-only extractor memory-maps the bytes, locates the literal top-level
`"structure"` array, and isolates only its first balanced JSON object with a
string-aware brace scanner.  It decodes that one object and never decodes,
summarizes, logs, or materializes the preceding DFT arrays or any later
structure.  The extracted structure is required to be a fully periodic
pymatgen `Structure` and is copied to a new geometry-only archive.

NEXT543 passes only with exactly 50 valid initial structures per prefix and 200
overall, matching remote MD5s and local SHA256s, unique IDs, finite nonzero
cells, and no summary file present in the formal confirmation source directory.
Its manifest records that endpoint bytes may exist inside sealed source files
but no endpoint value or later structure was parsed.

## Frozen candidate law and blind prediction gates

NEXT544 applies the already frozen NEXT541 mechanisms to the 200 geometry-only
structures:

- contact risk `-cov_q05` (NEXT32),
- same-sign shell risk `-SSSP` (NEXT411),
- affine accommodation risk `PBAAA` (NEXT537 v2),
- Pauling 2--5 controls.

The primary score remains exactly:

```
u_j = (midrank(x_j) - 0.5) / n_j
R_MUPR = 1 - product_j(1 - u_j)
```

Contact is required; unsupported SSSP/PBAAA is omitted.  No endpoint coefficient
or threshold is fitted.  The operational screen rejects the top 15% of MUPR
within the 200-structure batch, using prefix then trajectory ID as stable tie
breakers.

NEXT544 passes the label-blind freeze if contact and MUPR coverage are at least
0.95, MUPR has at least 50 distinct values, maximum point mass is at most 0.10,
all scores are finite and bounded, and deterministic/invariance tests pass.
Only then may NEXT545 download/read the four summary files and final structures.

## Offline endpoint and primary label

Summary rows provide a calculation message and final total energy.  For a
selected trajectory:

```
failed = (message is non-empty) OR (final energy is missing/non-finite)
```

For successful rows, final energy per atom is ranked within its frozen prefix
using deterministic midrank percentiles in the high-energy direction.

The primary binary endpoint `dft_waste` is true when either:

- `failed` is true; or
- a successful row is in the highest-energy quartile of its prefix
  (`energy_percentile > 0.75`).

The continuous endpoint is

```
waste_severity = 1.25                         if failed
                 energy_percentile            otherwise.
```

This endpoint directly represents a DFT run that fails or ends in the
least-favorable energy quartile for the same stoichiometry.  It does not claim
formation energy, energy above hull, or cross-composition thermodynamic
stability.

As a secondary response-only diagnostic, NEXT545 may extract the last structure
from the same frozen JSON object using the same balanced-object scanner and
report normalized initial-to-final RMS/max displacement and per-atom volume
change.  This diagnostic cannot rescue the primary endpoint.

## Frozen comparisons and statistics

The primary hypothesis is MUPR versus `dft_waste`.  Prespecified comparators are
Pauling violation fraction/individual rules and the three single MUPR mechanism
percentiles.  Directions are fixed as risk-high.

Metrics are evaluated on identical supported rows.  Confidence intervals use
10,000 deterministic stratified bootstrap draws: within each of the four
prefixes, rows are resampled with replacement at that prefix's original size.
Degenerate draws are skipped and counted.

## Confirmatory success gates

Every gate is required:

1. endpoint coverage at least 0.95 (at least 190/200);
2. at least 30 `dft_waste` positives and 30 negatives overall, and at least five
   of each class in every prefix;
3. mapped MUPR support at least 0.95;
4. MUPR ROC AUC at least 0.65 with stratified-bootstrap 95% lower bound above
   0.55;
5. MUPR AUC exceeds the best supported Pauling comparator by at least 0.05 on
   identical rows;
6. MUPR AUC exceeds the best single mechanism percentile by at least 0.02 on
   identical rows, demonstrating value beyond feature accumulation;
7. top-15% rejection precision at least 0.60, Wilson 95% lower bound at least
   0.42, and recall at least 0.20;
8. bottom-50% protected-set non-waste fraction at least 0.75;
9. Spearman correlation with `waste_severity` at least 0.35 and its stratified
   bootstrap 95% lower bound above 0.20;
10. within-prefix MUPR AUC exceeds 0.55 in at least three of four prefixes.

If any gate fails, this branch is a negative confirmation and cannot be rescued
by retuning on these endpoints.  The exact result is preserved, and the search
moves to a new hypothesis/source.  Passing all gates authorizes only a new
standalone report for user review; canonical report and paper edits remain
forbidden until explicit confirmation.

## Decision log

- Use convergence plus within-stoichiometry high final energy to avoid the
  degenerate structural-response label observed in NEXT542.
- Preserve failures rather than restricting to converged trajectories, because
  failed DFT effort is precisely relevant to pre-screening.
- Exclude empty objects only because no executable x0 can be reconstructed.
- Keep the MUPR equation unchanged so NEXT545 is confirmation, not a new fit.
- Require improvement over both Pauling and every single mechanism so a pass
  reflects a genuinely better combined law rather than feature accumulation.
