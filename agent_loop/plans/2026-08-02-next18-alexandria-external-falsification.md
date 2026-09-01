# NEXT18 Alexandria external falsification design

Date: 2026-08-02  
Status: frozen after NEXT17 production freeze and before external feature execution  
Scope: additive source sanitation, label-free execution, private endpoint extraction, aggregate evaluation

## Objective

Falsify the frozen NEXT17 rule on an external x0-to-DFT source without changing
its formula, threshold, relaxation settings, model, support policy, or success
gates. Compare against the operational Pauling P2-P5 control on exactly the same
initial geometries.

## Frozen NEXT17 candidate

Use `outputs/20260802_next17_strict_relax_gap_freeze/FROZEN_PROTOCOL.json`:

```text
R64s(i) = E_MatterSim_strict_relaxed(i)/N_i
          - min_j_in_same_composition E_MatterSim_strict_relaxed(j)/N_j
REJECT iff R64s >= 0.06 eV/atom
```

MatterSim 1.2.3 5M, FIRE, FrechetCellFilter, `fmax=0.005 eV/Angstrom`,
maximum 64 prediction steps, atom budget 512. Any incomplete/unsupported group
must ABSTAIN in full.

## External source and fixed cohort

Source: official Alexandria 2025-07-02 PBE geometry-optimization paths.

Only the first two numerically ordered official shards are in scope:

```text
pbe_0000.json.bz2
pbe_0001.json.bz2
```

They were fixed before the second shard composition inventory was opened. No
later shard may be added after performance is visible.

The raw JSON containers interleave initial geometries and DFT trajectory fields.
Therefore this cohort cannot satisfy a literal raw-byte never-read lockbox:
the sanitizer must parse each JSON value to reach its first structure. The
evidence boundary is instead:

1. a reviewed sanitizer may read the raw containers but may access only the
   first nonempty calculation's first `structure` field;
2. selection uses only material ID, reduced composition, atom count, and shard;
3. selected geometry is serialized without calculator, energy, forces, stress,
   or endpoint metadata;
4. the MatterSim/Pauling feature process receives only this physically isolated
   geometry archive and metadata;
5. endpoint energy is extracted into a separate private artifact only after the
   feature artifact is sealed.

This is an external-source falsification test with a threshold frozen on
ELEMENTA, but not a fresh never-read lockbox.

## Deterministic selection

Inventory all 20,000 material IDs in the two shards. Group by reduced formula
derived from the x0 structure. Include every composition occurring at least
twice in this fixed two-shard batch; do not cap group size and do not sample.

The label-free inventory audit found:

- 20,000 unique IDs and valid x0 structures;
- 19,806 reduced compositions;
- 185 repeated-composition groups;
- 379 selected rows;
- group-size histogram: 178 groups of size 2, 5 of size 3, 2 of size 4.

These counts are formal invariants. A mismatch must fail closed.

## DFT endpoint definition

After feature sealing, extract the last step of the last nonempty calculation
for each selected ID. Define total endpoint energy from its finite `energy`
field and divide by the final structure atom count. Verify that initial and
final reduced compositions agree. Within each frozen x0 composition group:

```text
DFT_regret(i) = E_DFT_endpoint(i)/N_i
                - min_j_in_group E_DFT_endpoint(j)/N_j
```

Definitions:

- exact group minimum: regret <= 1e-8 eV/atom;
- valuable: regret <= 0.05 eV/atom;
- high regret: regret >= 0.20 eV/atom.

## External success gates

The frozen NEXT17 rule must satisfy all of:

1. coverage 100% or every abstention is explicitly fail-open at group level;
2. exact group-minimum recall Wilson 95% lower bound >= 0.95;
3. valuable recall Wilson 95% lower bound >= 0.95;
4. reject precision above the group minimum Wilson lower bound >= 0.95;
5. DFT savings Wilson lower bound >= 0.10;
6. no selected composition group is fully rejected;
7. high-regret rejection recall is reported with Wilson interval and must not be
   lower than Pauling by more than 0.05 under composition-cluster bootstrap;
8. minimum-recall, valuable-recall, and reject-precision differences versus
   Pauling must each have cluster-bootstrap lower bound > 0 unless Pauling has a
   zero denominator, in which case the exact counts are reported and the gate
   fails closed.

Passing is external retrospective support for a DFT pre-screening candidate. It
still does not establish universal DFT equivalence because the shards may mix
selection strategies and MatterSim training overlap has not been excluded.

## Stop and reporting rules

- Do not tune the threshold, add a score, exclude capped structures, or download
  another shard after endpoint performance is visible.
- If any gate fails, preserve the negative result and stop this branch.
- Do not modify existing reports, the paper, README, notes, tex, or PREREG.
- A new standalone NEXT18 report may be proposed only after the external result
  is complete; canonical integration still requires explicit user approval.
