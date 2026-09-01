# NEXT83--NEXT89: independent SCIGEN DFT validation and new-law loop

Date: 2026-08-03

Status: pre-endpoint design freeze. No SCIGEN endpoint value has been parsed,
summarized, plotted, or used to choose a feature, transformation, threshold, or
gate at the time this document is first published.

## Objective

Use a genuinely external generated-crystal cohort to discover and validate an
interpretable pre-DFT plausibility law. The executable law may use only one
unrelaxed generated structure (`x0`) plus frozen elemental tables and
deterministic analytic geometry, graph, Voronoi, bond-valence,
electrostatics, linear-algebra, symmetry, and rigidity calculations. It must
not use DFT values, relaxed structures, trajectory frames, a learned
energy/force/stress proxy, an ML interatomic potential, physical relaxation,
or a same-composition alternative at execution time.

This work is additive. It does not replace earlier scripts, artifacts, reports,
or any canonical paper text.

## Independent source and eligibility audit

Primary source:

- SCIGEN official repository: <https://github.com/RyotaroOKabe/SCIGEN>
- Figshare collection: <https://doi.org/10.6084/m9.figshare.c.7283062>
- DFT article: <https://doi.org/10.6084/m9.figshare.26082733.v3>
- Source file: `03_scigen_materials_relaxed.zip`
- Figshare file id: `57245942`
- Published MD5: `fc217e45c5dd8920d08c523177546d45`
- Locally measured SHA-256:
  `7eb1b48200329e8d294d013c56767c2219020731dc9a44e36c23b83ac0914068`
- License: CC BY 4.0
- Cohort: 24,742 successfully converged DFT relaxations selected from 26,000
  SCIGEN candidates.

The archive has exactly one `POSCAR` and one `CONTCAR` for each of 24,742
material IDs, plus aggregate endpoint tables. The Figshare description defines
`POSCAR` as the structure before DFT relaxation and `CONTCAR` as the structure
after relaxation.

The upstream four-stage prescreen is selection only: (1) SMACT charge
neutrality, (2) space-occupation ratio, (3) a GNN energy-above-hull
classifier, and (4) a GNN pristine-versus-diffused classifier. The official
screening code performs classification and writes surviving generated
structures; it does not relax or otherwise update their coordinates. Thus the
archived `POSCAR` is eligible as an unrelaxed generated x0. The upstream GNN
scores are not inputs to this project and will not be reconstructed.

Important scope limitation: this is a post-prescreen cohort. Results measure
additional DFT structural-instability discrimination among candidates that
already survived SCIGEN's ML prescreen, not performance on all 7.86 million raw
generations and not thermodynamic hull stability.

The source was not used in NEXT1--NEXT82 formula development. WBM,
Alexandria/Elementa, SSAGEN, OMatG/OMat24, OMC25, QMOF, ODAC23, MatterSim, and
the CHGNet-based `matgen_baselines` samples are treated as previously used or
ineligible sources and are not rebranded as independent evidence here.

## NEXT83: immutable source receipt

1. Download the official Figshare file to an additive external artifact
   directory.
2. Verify the published MD5 and record SHA-256, byte count, DOI, license,
   publication timestamps, and the Figshare API response hash.
3. Inspect only the ZIP central directory. Do not read `CONTCAR`, `output.dat`,
   or `si_table_*.csv` payloads.
4. Publish a no-overwrite receipt and manifest.

## NEXT84: geometry-only cohort and physical label split

The sanitizer may read only members matching
`03_scigen_materials_relaxed/<material_id>/POSCAR`. It must reject any duplicate
ID, missing pair, unexpected path grammar, invalid/nonperiodic structure, or
non-finite geometry. It must never deserialize `CONTCAR`, `output.dat`, or a
supplementary CSV.

For each valid x0, publish:

- canonical geometry-only EXTXYZ bytes with no calculator, energy, force,
  stress, tag, constraint, or extra array;
- `material_id`, lattice-class prefix, reduced formula, chemical system,
  atom count, source member name, source member CRC and uncompressed byte
  count;
- no endpoint value or endpoint-derived flag.

Composition groups are assigned as indivisible units using
`sha256("NEXT84_SCIGEN_COMPOSITION_SPLIT_V1|" + reduced_formula)`. The first
eight bytes are interpreted as an unsigned integer and divided by `2**64`:

- `[0.00, 0.55)`: discovery;
- `[0.55, 0.75)`: internal validation;
- `[0.75, 1.00)`: internal replication.

The same reduced formula may not occur in more than one partition. Partition
counts and identity hashes are label-free and may be reported. Geometry
archives and metadata are physically separate by partition.

An automated endpoint router may subsequently read the aggregate endpoint
table once and route records by the frozen identity map into three physically
separate endpoint artifacts. It must not print, summarize, rank, plot, or
otherwise expose validation or replication values. The validation and
replication artifacts remain unopened until their respective frozen-prediction
gates authorize a one-shot evaluation.

## Published endpoint definition

SCIGEN defines a structurally stable relaxation using the final maximum force
and lattice/coordinate distortion. We preserve the published thresholds:

| lattice classes | `d_latt` threshold (A) | `d_xyz` threshold (A) |
|---|---:|---:|
| `tri`, `hon`, `kag`, `sqr`, `elt`, `sns`, `lieb` | 1.0 | 0.5 |
| `tsq`, `srt`, `snh` | 2.0 | 1.0 |
| `trh` | 3.0 | 1.5 |

The final-force threshold is 0.01 eV/A for every class. Define

`D = max(F_max / 0.01, d_latt / T_latt, d_xyz / T_xyz)`.

Precommitted evaluation strata are:

- protected: `D <= 1` (the published small-distortion stability criterion);
- middle: `1 < D < 2`;
- severe: `D >= 2`.

Only offline labels use DFT. No DFT quantity or label-derived statistic may be
serialized into an executable feature artifact.

## NEXT85: label-free analytic feature bank

Before any endpoint partition is opened, compute deterministic x0-only
features for every partition:

- NEXT43 finite analytic families: contact, rigidity, normalized Madelung,
  bond-valence equilibrium, symmetry recovery, directional sterics, valence
  transport, analytic electrostatics, Coulomb/steric balance, charge spectrum,
  self-stress compatibility, and bond-valence transport compatibility;
- NEXT44 rich geometry/contact/packing and full rigidity/Madelung/bond-valence
  families;
- NEXT80 periodic repulsive-load resolvability;
- frozen classical Pauling-rule controls using the existing definitions.

Unsupported families fail open (`KEEP`) and record a reason. Feature building
must accept no endpoint, relaxed-structure, model, checkpoint, or proxy
argument. Raw feature artifacts are frozen by SHA-256 before discovery labels
are opened.

Pure element-identity shortcuts (`atomic_number_*`, element one-hot fields),
raw lattice-class IDs, material IDs, formulas, and chemical-system labels are
excluded from candidate formulas. They may be retained only as audit and
stratification metadata.

## NEXT86: discovery endpoint opening

Open only the physically isolated discovery endpoint artifact and join it to
the already frozen discovery feature rows by exact material ID. Do not access
internal-validation or internal-replication endpoint files.

## NEXT87: finite sparse-law search and freeze

The new-law family is a nonnegative sum of at most three interpretable
one-sided hinge risks:

`R(x) = sum_j w_j * max(0, direction_j * (g_j(x)-center_j) / scale_j)`.

- `g_j` is identity, `log1p(max(x,0))`, or a signed `asinh` transform selected
  from a feature-specific predeclared catalogue;
- `direction_j` is fixed from the physical meaning before endpoint access;
- centers and positive scales are frozen from discovery x0 values only;
- weights are from `{0.25, 0.5, 1, 2, 4}` up to a common multiplier;
- missing/non-finite required terms produce `KEEP`;
- maximum three terms, no interaction tree, neural model, or hidden fallback.

Selection uses only discovery labels with five composition-group folds. A term
list must be supported in every fold and selected in at least four of five
foldwise searches before a final discovery-only refit. Search records must
retain every evaluated candidate and deterministic tie-break.

The rejection threshold is chosen only on discovery to maximize the
Wilson-score lower bound of severe-rejection precision subject to all gates
below. It is then immutable.

Discovery gates:

- support coverage Wilson lower bound >= 0.90;
- protected recall Wilson lower bound >= 0.95;
- severe-rejection precision Wilson lower bound >= 0.80;
- rejection-fraction/savings Wilson lower bound >= 0.02;
- protected-versus-severe pooled ROC AUC >= 0.75;
- macro lattice-class ROC AUC >= 0.65;
- worst eligible lattice-class ROC AUC >= 0.55;
- at least eight lattice classes contain both protected and severe examples;
- all five group folds satisfy precision >= 0.70 and protected recall >= 0.93;
- at the candidate's protected-recall operating point, it must reject more
  severe structures than the frozen Pauling P2--P5 union and have a higher
  severe-rejection precision lower bound.

If no candidate passes, stop this finite catalogue and retain the negative
result. Do not rescue it with validation labels.

After a discovery pass, serialize the complete formula and freeze predictions
for both internal validation and internal replication before either endpoint
file is opened.

## NEXT88: one-shot internal validation

Open only internal-validation endpoints and evaluate the frozen predictions.
Required gates are the discovery gates above, except the five-fold stability
condition is replaced by exact per-lattice reporting. In addition, the
precision lower bound must remain >= 0.80. No term, transform, center, scale,
weight, threshold, missing policy, or subgroup exception may change.

Failure is a precommitted stop. The internal-replication endpoint remains
unopened.

## NEXT89: one-shot internal replication

Only if NEXT88 passes, open the physically isolated internal-replication
endpoint. The same frozen gates apply. A successful replication supports a
new standalone report and a candidate-law claim for the SCIGEN post-prescreen
domain. It does not by itself justify a universal-crystal or DFT-equivalence
claim.

## Reporting and preservation

Whether the loop passes or stops, write a new standalone report containing
source provenance, immutable hashes, cohort limitations, formula, Pauling
comparison, discovery/validation/replication state, all failures, and exact
reproduction commands. Do not edit `paper/`, `tex/`, `README.md`, prior
reports, or existing scripts/artifacts.
