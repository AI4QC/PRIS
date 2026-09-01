# Complex-Anion Contact Correspondence Implementation Plan

> **For Codex:** Execute additively and test first. Preserve every prior
> artifact. Keep validation and replication sealed.

**Goal:** Test Hawthorne's Appendix 4 Lewis-basicity rule as a topology-only
correspondence between an isolated complex oxyanion's formal group charge and
the number of external cation--oxygen contacts it can support.

**Architecture:** NEXT485 recognizes a frozen non-hydroxylated subset of simple
isolated oxyanions using exact element identities and the unchanged periodic
opposite-sign Voronoi multigraph. It compares observed external contacts with
`|Q_group|/Lewis_basicity`. NEXT486--NEXT489 are conditional full label-free
build, cross-source discovery audit, bounded search, and BROAD diagnostic.

## 1. Frozen Appendix 4 subset and recognition

Transcribe the eleven non-hydroxylated Appendix 4 rows whose central atom and
oxygen coordination uniquely specify the group:

```text
BO3, BO4, SiO4, AlO4, PO4, AsO3, AsO4, VO4, CO3, NO3, SO4.
```

The executable identifies a central positive site of the stated element whose
entire opposite-sign contact population consists of exactly three or four
distinct O sites as specified. A group is isolated only if every ligand O has
exactly one incident bond from any recognized oxyanion centre. Shared-O,
polymerized or overlapping candidates are excluded. If H occurs anywhere in
the structure, no group is recognized because Appendix 4 assigns different
basicities after hydroxylation. Structures with no recognized isolated group
receive supported physical zero; the formula does not guess a group.

No distance cutoff, bond length, oxidation-state magnitude, fitted tolerance,
group subset, proton assignment, interpolation or nearest-template fallback is
available. The frozen CSV records group formula, centre element, centre
coordination, formal group charge, Lewis basicity, DOI and CC BY attribution.

## 2. Frozen CACC formula

For isolated group `g`, let `N_g` be the number of translated cation--O
contacts incident to its ligand O sites after removing the centre--O contacts.
Appendix 4 implies the characteristic external-contact count

```text
T_g = |Q_g| / beta_g,
```

where `beta_g` is the tabulated Lewis basicity. Freeze

```text
D = sum_g |N_g-T_g| / sum_g (N_g+T_g),
CACC(x0) = round_1e-10(1-D),
```

with physical zero when no eligible group exists. The sole feature is
`cacc_complex_anion_contact_correspondence`, direction `protected_high`, and
range `[0,1]`. Malformed contacts, zero-charge sites and isolated charged sites
are unsupported; an absent opposite-sign graph is supported zero.

## 3. Hard no-DFT boundary and invariance

The executable reads only composition, charge signs, one raw initial unrelaxed
periodic geometry, translated contact topology and fixed public Appendix 4
constants. It must not run/read DFT, energy/force/stress, learned proxies,
MLIPs/potentials, relaxation, trajectories, later geometry,
same-composition alternatives, validation or replication. Edge order,
disjoint exact replication, rigid motion, translation, site permutation,
unimodular rebasing and exact supercells must be invariant within `1e-8`.

## 4. Frozen ordered blind gates

Use the unchanged 80+80 discovery probes. Before opening any prior feature
table, require support `>=72/80`, `[0,1]`, at least 20 values distinct at
`1e-10`, and invariance error `<=1e-8` in each source. Only if all four
engineering gates pass, compare with all 32 prior formal families, recomputed
ZBVVG through PFPU, CLAM, MV-CLAM, ECCC, CCCB and SBCC, requiring maximum
adequate absolute Spearman `<0.90` with at least 40 joint finite rows. This
short-circuit order changes no threshold or cohort.

Only if all pass: NEXT486 requires full discovery coverage `>=0.95` per source;
NEXT487 applies unchanged NEXT224/NEXT413 outcome gates; NEXT488/NEXT489 remain
conditional. Validation and replication stay sealed.

## 5. Artifact order

1. Add attributed asset and RED recognition/kernel/invariance/firewall tests.
2. Implement independent pure core and raw-periodic wrapper.
3. Run ordered support and full novelty probes.
4. Continue only on mechanical authorization.
5. Update the independent report and rerun complete verification.
