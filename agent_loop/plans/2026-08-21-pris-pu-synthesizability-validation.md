# PRIS--PU synthesizability experiment plan

**Status:** additive experiment only. Do not edit the manuscript, Supplementary
Information, or their canonical data products until the results are reviewed.

## Scientific question

Test whether explicit violations of the Plausibility Rules for Inorganic
Structures (PRIS) are enriched among structures assigned low crystal-likeness
scores by the CSAgent positive--unlabelled (PU) models, and quantify whether
PRIS can remove part of that difficult queue before a learned synthesizability
model or later theoretical validation is run.

The experiment has two complementary parts.

1. **Binary cohort comparison.** Evaluate the frozen 99,162-structure
   experimental cohort and all 364,771 released PU-predicted hard negatives.
2. **Continuous-score analysis.** Evaluate the original scored candidate pool
   and measure how each PRIS law and rung changes across the continuous CLscore
   distribution. High CLscore means a larger learned probability of belonging
   to the experimentally reported class.

The first part tests separation at the released decision boundary. The second
tests whether the association is graded rather than an artefact of selecting a
single low-score tail.

## Frozen inputs and provenance

- CSAgent checkout: `<other-repo>/`
- Released negative structures:
  `data/from_hpc/release/negatives/{train,val}.csv`
- Released negative metadata:
  `data/from_hpc/release/negatives/meta.tsv`
- Frozen experimental mapping: rows with `source_index < 99162` in the
  original CSAgent experimental snapshot. The exact structure bytes are read
  through the frozen 99,162-row provenance table and its recorded blob offsets.
- Continuous outputs: original `clscore_all.csv`, `clscore_b_all.csv`, and the
  CSAgent pool-index mapping recovered from the HPC run recorded in the CSAgent
  Claude session. Any local or remote copy must be hashed before use.
- PRIS implementation: reuse
  `experiments/property_design_20260821/evaluate_generated.py` so thresholds and
  applicability guards are identical to the current published definitions.

Each run writes an input manifest containing absolute source paths, byte sizes,
SHA-256 hashes, row counts, the newpauling commit, and the CSAgent commit.

## Prespecified verdicts and outcomes

PRIS is three-valued for every individual rule and rung:

- `pass`: every applicable, computable condition is satisfied;
- `explicit_violation`: at least one applicable condition is violated;
- `no_verdict`: no violation is observed but at least one required value cannot
  be computed.

Only `explicit_violation` removes a structure. `no_verdict` must never be
silently counted as a pass or a violation.

Primary binary outcomes for every PRIS rung are:

- experimental-structure satisfaction = pass / (pass + explicit violation);
- experimental queue retained = (pass + no verdict) / all experimental rows;
- PU-hard-negative detection = explicit violation / all PU-negative rows;
- queue reduction at the experimental-satisfaction operating point.

The report also gives all raw state counts, individual D1--D8 mechanism rates,
0.5 and 0.7 Angstrom distance-cutoff baselines, and sensitivity results for the
high-confidence subset and the LeMat and ELEMENTA sources separately.

Primary continuous outcomes are:

- PRIS explicit-violation and no-verdict rates in fixed CLscore quantiles;
- monotonic trend across quantiles;
- Spearman association between CLscore and violation count;
- CLscore distributions for pass, explicit-violation, and no-verdict states;
- score--rule concordance and disagreement cases.

CLscore is a learned proxy, not a synthesis outcome. The binary experiment is a
descriptive association rather than a fully independent external validation,
because the PRIS discovery resources include subsets of the same experimental
collection. The PU-negative structures did not fit PRIS thresholds.

## Confounding controls

Raw separation can be driven by composition or cell complexity. In addition to
the full-cohort result, compare experimental structures and PU negatives within
prespecified strata based on:

- element set / reduced chemical system when exact support overlaps;
- number of elements;
- number of sites (log-spaced bins);
- ionic-applicability and charge-assignment route;
- data source for the PU negatives.

Use common-support weighting as the primary adjusted summary. A deterministic,
seeded matched sample is a sensitivity analysis. Report loss of coverage rather
than extrapolating beyond common support.

## Data-integrity gates

The pipeline fails closed if any of the following occurs:

- the frozen experimental index is not exactly the contiguous range 0--99,161;
- negative train/validation indices do not join one-to-one to all 364,771
  metadata rows;
- a material identifier, source index, or structure hash is duplicated across
  comparison cohorts without being explicitly reported;
- a malformed four-field CGCNN CLscore row is misread because its merged header
  contains only three labels;
- resumed shards use different input hashes, evaluator hashes, or schemas;
- a parsing/evaluation failure is converted to an explicit violation.

## Implementation and execution

1. Write unit tests for frozen-cohort selection, blob decoding, released-data
   joins, malformed-header CLscore parsing, three-state aggregation, fixed
   quantiles with ties, common-support weighting, and resumable shard manifests.
2. Implement read-only dataset adapters and a multiprocessing PRIS batch runner.
3. Run a stratified pilot to measure CIF parsing success and throughput, then
   freeze shard size and worker count.
4. Evaluate the entire binary cohort with resumable, immutable shards.
5. Recover and hash the continuous raw scores and their source structures, then
   evaluate either the full common pool or a predeclared deterministic sample if
   the full remote structure store cannot be made available. Never substitute
   the released negative tail for the requested continuous experiment.
6. Generate machine-readable summaries, compact figures, concordant and
   discordant example CIFs, and an independent Chinese report.
7. Verify row accounting, hashes, deterministic reruns, tests, and report claims.

## Interpretation gates

- A useful pre-screen requires both measurable PU-hard-negative detection and
  high experimental queue retention. Report both together.
- A binary difference without a graded continuous trend is not evidence that
  PRIS tracks synthesizability across the pool.
- A graded trend that disappears within chemistry/size strata is reported as
  dataset-composition association, not a structure-level signal.
- Agreement between PRIS and CLscore establishes complementarity only after
  feature overlap and discordant cases are examined.
- No result is added to the paper before review of the independent report.
