# NEXT26 OMC25 packing-law design

## Objective

Develop an additive, interpretable law that screens a single raw generated
molecular-crystal structure before any DFT calculation.  The executable law
may use only composition, the unrelaxed cell and coordinates, deterministic
geometry, and tabulated elemental properties.  DFT energies, forces, stresses,
relaxed structures, trajectories, MLIPs, and relaxation proxies are evaluation
labels only.

No existing script, report, paper, README, or frozen result is replaced.  This
experiment will publish a new standalone NEXT26 report before any canonical
document is considered for modification.

## Source audit

- Dataset: official `facebook/OMC25` validation archive and starting-crystal
  catalogue.
- The catalogue has 218,841 rows: 207,271 train and 11,570 validation starting
  crystals.  Validation CSD refcodes are disjoint from train refcodes.
- The validation archive is one 8,110,386,232-byte gzip-compressed tar stream
  containing sharded ASE-LMDB files.
- The first complete extracted shard is `data0031.aselmdb`, SHA-256
  `d1a2e915427703149fda5511e6de4aa5f0179a886dceb49c5d68b4bc9b43bcac`.
  It contains 34,670 frames from 290 trajectories; 288 trajectories are
  complete against the official `nframes` catalogue.  Their labels have been
  opened and this shard is permanently development-only.
- The next archive member is `data0037.aselmdb` (291,971,072 bytes).  It is the
  candidate prospective cohort.  Byte extraction alone does not inspect its
  scientific contents.  The x0 sanitizer must read only each trajectory's
  earliest frame and must never publish DFT-labelled fields.

## Endpoint and claim boundary

The primary DFT-only endpoint is a severe initial/relaxation response.  A
trajectory is positive when at least one fixed condition holds:

1. initial maximum atomic force >= 1.0 eV/Angstrom;
2. initial RMS atomic force >= 0.40 eV/Angstrom;
3. DFT relaxation energy decrease >= 0.040 eV/atom;
4. initial six-component stress norm >= 0.030 eV/Angstrom^3.

These thresholds are frozen from physical scale and the development-shard
audit before the prospective endpoint is opened.  They measure whether the
raw candidate provokes a substantial DFT response; they do not prove
thermodynamic instability or exclude a different stable polymorph.

Secondary, non-gating endpoints are atomic displacement, cell logarithmic
strain, volume change, final force, and continuous rank correlation.

## Candidate law family

The law family is intentionally small and analytic:

- covalent-sphere packing fraction;
- mass-density proxy;
- first-percentile nonbonded van-der-Waals distance ratio after excluding
  inferred 1-2, 1-3, and 1-4 covalent neighbours;
- nonbonded clash fraction below 0.85 of summed van-der-Waals radii;
- inferred covalent-bond length dispersion;
- cell anisotropy and volume per atom.

Development search may robust-standardize these terms using medians and IQRs
from `data0031`, use fixed-sign equal-weight sums of at most two terms, and use
one-sided or two-sided hinge/tail forms.  Missing or unsupported cases are
fail-open.  No learned model, optimizer-derived dense coefficients, energy
surrogate, force surrogate, or relaxation is eligible.

The intended physical form is a packing-pressure mismatch: overpacking and
short nonbonded contacts indicate repulsive pressure, while severe
underpacking indicates collapse risk.  Search results must retain this
interpretation and are not accepted solely because a metric improves.

## Split and freeze protocol

1. Develop feature definitions, endpoint code, and the candidate search only
   on complete `data0031` trajectories.
2. Freeze the selected formula, constants, threshold, feature schema, source
   hashes, endpoint definition, and gates in immutable JSON plus a manifest.
3. Extract `data0037` as an opaque archive member.  Sanitize x0 records without
   decoding or exporting DFT labels.  Exclude incomplete trajectories and all
   CSD refcodes seen in development using metadata/frame-count checks only.
4. Compute DFT-free features and publish checksum-locked predictions.
5. Only after prediction freeze may a separate evaluator decode holdout forces,
   energies, stresses, and final structures.

This is a checksum-locked prospective protocol, but not a physically isolated
never-readable lockbox because x0 and endpoints share one LMDB member.  The
report must state that limitation explicitly.

## Frozen primary gates

Use two-sided 95% Wilson intervals.  All four gates must pass:

- supported coverage lower bound >= 0.95;
- protection of endpoint-negative structures lower bound >= 0.95;
- endpoint-positive precision among rejected structures lower bound >= 0.90;
- rejected-fraction lower bound >= 0.10.

Pauling controls are reported with identical support and endpoint accounting.
For molecular crystals, broad abstention or domain inapplicability is a result,
not permission to impute a favourable decision.

## Stop conditions

- No eligible development formula: report a negative result and do not open a
  new prospective endpoint.
- Any sanitizer output contains energy, force, stress, relaxed geometry, or
  more than one frame per trajectory: hard fail.
- Any prediction artifact is created after holdout labels are opened: the
  prospective claim is ineligible.
- Gate failure: retain the continuous diagnostic evidence, but make no claim of
  replacing or surpassing Pauling or DFT.

