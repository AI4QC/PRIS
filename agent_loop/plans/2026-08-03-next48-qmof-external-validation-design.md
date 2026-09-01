# NEXT48 QMOF external zero-refit validation design

## Objective

Test whether the frozen NEXT31 two-term, DFT-free OMC25 packing law transfers
without refitting to an independent framework-crystal source.  The law sees one
raw QMOF unrelaxed structure only.  QMOF PBE-D3(BJ)-relaxed structures are
opened only after identities, predictions, endpoint definitions, thresholds,
and gates have been sealed.

## Alternatives considered

1. **Frozen NEXT31 transfer with a structural-relaxation endpoint (selected).**
   This is the strongest available external test because QMOF contains paired
   unrelaxed and PBE-D3(BJ)-optimized structures, but no public optimization
   trajectories or initial energies.  It preserves the executable law exactly
   and measures genuine cross-source and cross-domain migration.
2. **Refit the packing rule on QMOF.**  This could improve in-domain numbers but
   would destroy the zero-refit external test and make the opened QMOF endpoint
   a development label.  It is excluded from NEXT48.
3. **Use final QMOF total energy as the endpoint.**  QMOF does not publish the
   matching initial energy needed for a relaxation-energy drop.  Comparing raw
   total energies across compositions would be scientifically invalid.
4. **Use cell-volume change only.**  It is cheap and mapping-free, but misses
   local coordination and internal framework rearrangement.  It will be
   retained as a secondary diagnostic, not the primary endpoint.

## Frozen inputs and cohort

- Source archive:
  `<path>`
- Required archive SHA-256:
  `97d23c0b4f9e5a30888e53dc16222b90443ad7167c3284d2258615d9f44eceef`
- Frozen NEXT31 rule SHA-256:
  `993d64b851c755fc5cc0d4b68ca7ca6994d4bdb7ed666f860d43a04925e254a8`
- Initial candidate files: all 4,147 CIF members under
  `unrelaxed_structures/other/`, sorted by member name.
- Eligible cohort: every initial member whose filename stem maps to exactly one
  whitelisted `qmof.csv` metadata row and whose mapped QMOF ID has a relaxed CIF.
  The pre-coordinate audit yields 4,119 rows.  The 28 unmapped filenames are
  recorded verbatim and excluded without guessing.
- No coordinate-, endpoint-, energy-, volume-, density-, or band-gap-based
  selection is allowed.  Only `qmof_id`, `name`, `info.formula`,
  `info.formula_reduced`, `info.natoms`, and `info.source` are selected from the
  endpoint-bearing CSV member.  The protocol truthfully records that archive
  bytes are accessible and that this is not a physical never-read lockbox.

## Prediction contract

For each eligible unrelaxed CIF, construct a new geometry-only ASE object with
atomic numbers, Cartesian positions, cell, and periodicity.  Drop all CIF
metadata, occupancies, charges, calculators, and auxiliary arrays.  Compute the
existing NEXT27 periodic nonbond descriptors and apply the exact NEXT31 formula:

```text
R_E(x0) = -(q05 - 0.9831717659022418) / 0.0955608356181108
          +(C1.05 - 2.152680652680653) / 3.418499623150786

REJECT iff R_E >= 2.4463648618269622
```

Any CIF or feature failure abstains and does not reject.  In the same x0-only
pass, apply the already frozen Pauling 2--5 control implementation.  Seal a
deterministic geometry-only archive, feature table, prediction table, source
hashes, input hashes, support counts, and prediction hashes before any relaxed
CIF payload is read.  No coefficient, feature, rule, or threshold can change.

## Frozen endpoint and gates

The primary endpoint is the L2 change in the historical Matbench Discovery
CrystalNN site-stat fingerprint (`ops`; mean, standard deviation, minimum, and
maximum).  Unlike trajectory evaluation, QMOF CIF atom order is not assumed to
be preserved: exact element-count equality is required, then each structure is
featurized independently.  Unsupported endpoint pairs are retained in the
accounting and excluded from scientific metrics.  Secondary diagnostics are
absolute log-volume change and endpoint failure counts.

Thresholds are inherited unchanged from NEXT23/NEXT42:

- protected: fingerprint change at most 0.10;
- substantial positive: fingerprint change at least 0.20;
- severe: fingerprint change at least 0.50.

The frozen NEXT31 migration gates are also inherited from its confirmation
protocol: one-sided 95% coverage lower bound at least 0.95, protected-recall
lower bound at least 0.95, rejection-precision lower bound at least 0.70,
savings lower bound at least 0.02, and substantial-change AUC at least 0.85.
The overall cohort is primary.  BoydWoo, GMOF, CoRE, ToBaCCo, and Anderson
source slices and Pauling results are descriptive; no slice can rescue an
overall failure.

## Claim boundary

A pass would establish external-source, zero-refit evidence that a two-term
raw-x0 analytic packing law conservatively anticipates large DFT structural
relaxation in hypothetical and curated MOFs.  A failure would be equally
informative evidence that the molecular-crystal law does not transfer to
periodic frameworks.  Neither outcome establishes formation energy,
convex-hull stability, phonon/dynamical stability, thermodynamic stability, or
replacement of DFT.  QMOF final structures are published optimized geometries;
per-row force convergence cannot be independently verified from the CIF archive.

## Error handling and verification

All output directories publish atomically and refuse replacement.  Protocol,
cohort, rule, source, prediction, and endpoint hashes are rechecked during
publication.  Tests cover mapping without coordinate parsing, strict hash
binding, initial-only prediction, fail-open behavior, relaxed-payload opening
only after prediction validation, composition-invariant endpoint comparison,
metric arithmetic, immutability, and injected deterministic fixtures.  The
final standalone report must distinguish the OMC25 energy endpoint from the
QMOF structural endpoint and must not modify any existing report, script,
paper, README, or prior artifact.
