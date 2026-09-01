# NEXT541--NEXT542: source-independent CDVAE x0-to-DFT response test

Date frozen: 2026-08-13

## Scientific question and hard boundary

Can a completely DFT-free, coefficient-free combination of three initial-cell
mechanisms rank generated inorganic crystals by the severity of their later DFT
structural response, and does it do better than Pauling controls on a new data
source?

The executable screen may read only composition and one raw, fully periodic,
unrelaxed initial structure (`x0`).  It may not read energies, forces, stresses,
relaxed structures, trajectories, ML predictions, ML potentials, proxy
potentials, or coordinates produced by any physical or virtual relaxation.  DFT
outputs are endpoints used once, offline, only after every candidate score and
decision has been frozen.  No existing script, artifact, report, or paper file
is replaced.

## Independent source and endpoint seal

The source is Figshare article 23681025 version 2, associated with the published
JARVIS-CDVAE superconductors study.  It was not used in NEXT31--NEXT540.

Inputs already downloaded from the official Figshare file API:

- `all_pred_data.json.zip`: SHA256
  `8f879acec0e6b6f8a201d9c4ff59de55e2bfd26b790c7aceace05511f753a55b`
- `CDVAE_relax_DFT.zip`: SHA256
  `1308bb32ec05bfcad1e9858dfaa25927f30c1fec61886fea47bd3b298e5db4cd`
- `cdvae_plotting.ipynb`: SHA256
  `2d1d274528726b1bc88347555069ef7953113c1245b9ef85a691d23a80b0a1af`

Before this design was frozen, only the central-directory names of the endpoint
archive were inventoried.  No POSCAR bytes, final lattice, final coordinates,
or response value were read.  The endpoint inventory contains 61 POSCAR names
covering 48 reduced compositions.  The initial archive contains 2,895 records;
149 records have one of those 48 compositions.  Because names are not unique
identifiers, all 149 possible initial candidates must receive frozen scores.

The initial JSON fields `pred`, `fenp`, and `bg` are model-derived quantities.
Their values are forbidden and must not enter the feature table, selection,
score, mapping, or evaluation.  NEXT541 reads only `formula` and `atoms` and
records the discarded field names as a firewall check.

## Frozen mechanisms

For each of the 149 possible initial candidates, NEXT541 computes:

1. Steric contact risk: `C_raw = -cov_q05`, using the frozen NEXT32 periodic
   covalent-radius contact kernel.
2. Same-sign shell intrusion risk: `S_raw = -SSSP`, using frozen NEXT411.
3. Periodic affine accommodation risk: `A_raw = PBAAA`, using frozen NEXT537
   protocol v2.
4. Pauling rules 2--5 as a comparator, using the frozen classical control.

No coefficients are fitted.  Within the complete 149-record candidate batch,
each supported raw mechanism is converted to a deterministic midrank percentile
in its risk-high direction:

```
u_j = (rank_mid(x_j) - 0.5) / n_j
```

The new mechanism-union percentile risk (MUPR) is

```
R_MUPR = 1 - product_j(1 - u_j),  j in {C, S, A} when supported.
```

At least the contact mechanism must be supported.  Missing SSSP or PBAAA is
omitted rather than imputed from endpoints.  `mechanism_count` is retained and
reported.  The operational batch screen rejects the top 15% of mapped records
by frozen `R_MUPR`, with stable ties broken by the frozen initial record index.
This is a batch pre-screening rule, not a fitted probability.

The three raw mechanisms are secondary, prespecified diagnostics.  MUPR is the
single primary hypothesis; the secondary results cannot rescue a failed MUPR
test.

## NEXT541 label-blind gates and freeze artifacts

NEXT541 is authorized to publish predictions only if all of the following hold:

- exactly 2,895 initial records and 61 endpoint filenames are inventoried;
- exactly 149 possible initial candidates and 48 compositions are recovered;
- no forbidden model-property value is materialized in the feature table;
- contact support is at least 0.95;
- MUPR support is at least 0.95;
- MUPR has at least 30 distinct values and no one value occupies more than 0.25;
- every value is finite and in `[0, 1]`;
- deterministic rerun hashes and geometry-only invariance tests pass.

Published artifacts include the 149-row initial-only feature/prediction table,
the 61-name endpoint inventory, all source/design hashes, the exact screening
formula, feature protocols, support/failure counts, and an explicit
`endpoint_contents_read=false` declaration.  Once published, the table and
manifest are immutable inputs to NEXT542.

## Frozen mapping after predictions

Only after NEXT541 passes and its prediction hashes are recorded may NEXT542
read the 61 endpoint POSCAR members.

For each reduced composition, mapping is a deterministic rectangular one-to-one
assignment from the final files to distinct initial candidates.  Pair costs are
obtained with pymatgen `StructureMatcher` using primitive reduction, volume
scaling, species matching, and two prespecified tolerance tiers:

- tier 0: `ltol=0.20`, `stol=0.30`, `angle_tol=5` degrees;
- tier 1 fallback: `ltol=0.50`, `stol=0.50`, `angle_tol=15` degrees.

`attempt_supercell=true`, `scale=true`, and `allow_subset=false`.  Tier 0 costs
precede tier 1 costs lexicographically.  Within a tier the cost is normalized
RMS displacement, then normalized maximum displacement, then initial index.
The global minimum one-to-one assignment is solved per composition.  A final
file with no finite candidate pair is unmapped.  Mapping coverage is reported;
no tolerance may be changed after endpoint access.

## Frozen DFT-response endpoint

For each mapped pair, let `rms` and `max` be the normalized displacement values
from the chosen StructureMatcher result and let

```
v = abs(log((V_final / N_final) / (V_initial / N_initial))).
severity = max(rms / 0.15, max / 0.30, v / log(1.25)).
```

The binary endpoint `severe_response` is true when any of:

- tier 0 did not match and only tier 1 did;
- `rms > 0.15`;
- `max > 0.30`;
- `v > log(1.25)`.

An unmapped file is excluded from discrimination metrics and counted as a
mapping failure, not silently labeled severe.  This endpoint measures a large
DFT structural response; it is not an energy-above-hull or thermodynamic
stability label.

## One-shot success gates

NEXT542 is a one-shot evaluation.  No threshold, score, mapping setting, or
endpoint definition may be changed after reading endpoint coordinates.

All gates are required for a genuine source-independent success:

1. mapping coverage at least 0.90 (at least 55 of 61 files);
2. at least 10 severe and at least 10 non-severe mapped endpoints;
3. MUPR support at least 0.90 on mapped endpoints;
4. MUPR ROC AUC point estimate at least 0.65 and its composition-cluster
   bootstrap 95% lower bound greater than 0.50;
5. MUPR AUC exceeds the best supported Pauling scalar/control AUC by at least
   0.05 on the identical rows;
6. top-15% MUPR rejection precision at least 0.70, Wilson 95% lower bound at
   least 0.45, and severe-response recall at least 0.20;
7. the bottom 50% protected set contains at least 80% non-severe responses;
8. Spearman correlation between MUPR and continuous severity is at least 0.30,
   with composition-cluster bootstrap 95% lower bound greater than zero.

Bootstrap uses 10,000 deterministic draws, resampling the 48 reduced-
composition clusters with replacement.  A degenerate draw is skipped and its
count reported.  Gate 5 is evaluated only where both candidates are supported;
Pauling missingness cannot be treated as a favorable decision.

If any gate fails, the branch is a documented negative result.  It does not
authorize a report claiming a replacement law, endpoint-driven retuning, or
opening another sealed dataset.  A new hypothesis or source must be frozen in a
new additive NEXT stage.

## Decision log

- Use all 149 same-composition initial candidates because filenames alone do
  not identify which generated geometry was relaxed.
- Freeze predictions for every possible candidate before structural matching,
  preventing final coordinates from influencing the law.
- Prefer a coefficient-free union of physical ranks over another endpoint-fitted
  linear formula.
- Keep PBAAA as a mechanism rather than assuming its earlier ODAC failure makes
  it universally irrelevant.
- Treat structural response as an external proxy only; do not overclaim DFT
  energy-level stability.
