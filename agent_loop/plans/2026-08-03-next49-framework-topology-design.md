# NEXT49 topology-aware framework branch: design

## Objective

Develop a new, additive crystal-screening branch for periodic coordination
frameworks.  The executable rule must inspect one raw, unrelaxed structure
only.  DFT calculations, DFT-derived scalar inputs, relaxed coordinates,
trajectories, alternative structures, MLIPs, learned energies, forces, and
stresses are forbidden at execution time.

NEXT48 established that the frozen two-term molecular-packing rule cannot be
claimed as a universal MOF law: it retained high precision on eleven rejections
but had only 0.27% savings and AUC 0.576 for substantial structural change.
NEXT49 therefore does not refit or overwrite NEXT31.  It adds a topology gate
and a separately developed framework formula.

## Domain split

Construct a periodic covalent graph from frozen elemental covalent radii and
analytic minimum-image geometry.  Every graph edge carries its integer lattice
translation.  For each quotient-graph component, assign lattice offsets along
a spanning tree; residual translations on non-tree edges span the component's
periodic translation space.

- translation rank 0: discrete molecular component;
- translation rank 1 or 2: chain or layer framework;
- translation rank 3: three-dimensional framework.

The largest periodic-component atom fraction and maximum translation rank form
the domain gate.  The original NEXT31 formula remains the molecular branch;
NEXT49 develops only the periodic-framework branch.

## Candidate x0 feature families

All features are deterministic, permutation/translation/supercell invariant or
explicitly normalized per atom/site:

1. periodic topology: maximum translation rank, periodic atom fraction,
   quotient-component density, and covalent cycle density;
2. bond geometry: lower/upper quantiles and spread of covalent-radius-normalized
   edge lengths;
3. metal coordination: coordination quantiles, under-coordinated metal fraction,
   metal-ligand bond-ratio spread, and local unit-vector imbalance;
4. cell packing: frozen covalent/van-der-Waals sphere volume per cell volume and
   lattice condition number.

No feature may contain a token suggesting a label or calculated endpoint.
Unsupported structures fail open and are recorded rather than rejected.

## Development and validation separation

QMOF is already exposed and is development-only.  Its initial structures may
be featurized; its PBE-D3(BJ)-relaxed structures may be used only after the x0
feature artifact is sealed.  The primary development endpoint is severe local
environment change (CrystalNN fingerprint L2 >= 0.50), with protection for
change <= 0.10.  Feature and threshold selection must report source-family
leave-one-source-out behavior and may not describe it as blind validation.

The future external lockbox is ODAC23 IS2RS/IS2RE, whose official artifact is
PBE+D3 (VASP), CC BY 4.0, MD5
`f7f2f58669a30abae8cb9ba1b7f2bcd2`.  The archive may be downloaded and hashed
opaquely now, but its records, structures, energies, forces, and relaxed
coordinates must not be opened until the NEXT49 executable formula, threshold,
exclusions, and gates are frozen.  Framework atoms will be evaluated separately
from analytically removable CO2/H2O adsorbates; QMOF-overlapping framework
identities will be excluded before opening endpoints.

FAIR-MOFs is not the primary lockbox because its 33,361 optimized structures
were produced with GFN-xTB rather than periodic DFT.

## Claim gate

NEXT49 is only a candidate until the independent lockbox passes all frozen
gates.  At minimum the gate must cover supported fraction, protected-structure
recall, rejection precision, useful savings, AUC, and official ODAC source/OOD
split robustness.  Failure produces an additive failure report and no universal
claim.  Success produces a standalone new report; the paper and prior reports
remain unchanged until user confirmation.
