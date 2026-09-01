# NEXT53 ODAC23 framework-development and OOD-confirmation protocol

## Frozen source

Use only the official ODAC23 IS2RS/IS2RE archive at
`https://dl.fbaipublicfiles.com/dac/datasets/odac23_is2r.tar.gz`:

- bytes: 848,157,819;
- published MD5: `f7f2f58669a30abae8cb9ba1b7f2bcd2`;
- local SHA-256:
  `13a26f00a6a26a95ab0706bf77b5dc1598cc689bda8846913bec0cc643152496`;
- level of theory: PBE+D3 in VASP;
- license: CC BY 4.0.

The opaque receipt was sealed before archive member names or payloads were
opened.  FAIR-MOFs is not substituted because its optimization used GFN-xTB.

## Split firewall

After this document is frozen, archive headers may be inventoried without
deserializing any LMDB value.  Member paths are assigned by their official
split names:

- development: `train` members only;
- confirmation lockbox: every `val`, `test-id`, `test-ood-big`,
  `test-ood-linker`, `test-ood-topology`, and
  `test-ood-linker-and-topology` member.

No confirmation LMDB value may be deserialized until a single executable
formula, domain gate, endpoint transformation, exclusions, and pass/fail gates
are sealed.  ODAC23's framework-stratified official split prevents pristine
and defective variants of the same parent framework from crossing splits.

## Allowed development fields

Only trusted official LMDB objects are deserialized.  The reader whitelists:
initial positions, relaxed positions, atomic numbers, cell, periodic boundary
flags, atom tags, and stable sample/framework identity.  Energies and forces
are not needed for the primary endpoint and must not enter the executable law.
Unexpected object types or fields fail closed during ingestion; unsupported
records fail open during law execution.

Adsorbates are removed before feature calculation.  Atom tags are the primary
selector; if tags are absent, only disconnected covalent components with exact
CO2 or H2O stoichiometry may be removed.  The removal method and counts are
recorded.  The law then receives one raw framework x0 only.

## Offline DFT endpoint

For each DFT relaxation, compute minimum-image framework-atom displacement
between initial and relaxed positions, requiring unchanged atom identity/order
and a fixed cell.  Aggregate repeated placements of one identical framework x0
by the median of their per-relaxation 95th-percentile framework displacement.

- protected: median p95 displacement <= 0.05 angstrom;
- severe: median p95 displacement >= 0.20 angstrom.

These thresholds are frozen before reading the train payload.  They measure
large DFT relaxation, not formation energy, convex-hull stability, phonons, or
finite-temperature lifetime.

## Formula development

The executable candidates are finite one-to-three-term formulas over one x0:
periodic translation rank, framework fraction, metal/donor CrystalNN
coordination clarity and entropy, metal-ligand distance strain, packing, and
predefined net-cycle/connectivity descriptors.  No DFT value, relaxed
coordinate, force, stress, energy, MLIP, or learned energy/force/stress proxy is
an input.  Missing required terms force `KEEP`.

Train frameworks are internally separated by a fixed framework-identity hash.
Only discovery labels select features, signs, scales, weights, and threshold;
the internal validation subset is opened once for advancement.  A candidate is
sealed for external confirmation only if both subsets pass.

## Frozen advancement gates

- coverage one-sided 95% Wilson lower bound >= 0.95;
- protected recall lower bound >= 0.95;
- severe-rejection precision lower bound >= 0.70;
- savings lower bound >= 0.02;
- severe AUC >= 0.75;
- source/split macro AUC >= 0.65 and worst evaluable split AUC >= 0.55.

The final official validation/test evaluation is one-shot.  A failed split or
gate yields an additive failure report and no universal-law claim.  A passing
result yields a standalone report; prior reports and the paper remain unchanged
until user confirmation.
