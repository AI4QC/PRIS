# NEXT55 ODAC23 x0 analytic mechanics feature design

## Boundary

Every feature is calculated from one selected, raw, unrelaxed framework x0.
Allowed inputs are atom identities, Cartesian coordinates, periodic cell,
frozen elemental covalent/van-der-Waals radii, and frozen tabulated Pauling
electronegativities.  No DFT value, relaxed coordinate, alternative structure,
energy/force/stress proxy, MLIP, physical relaxation, or fitted representation
is allowed.  Unsupported rows force `KEEP` downstream.

## Existing deterministic family retained

Retain all NEXT49 periodic topology descriptors: periodic translation rank and
framework fraction, covalent graph coordination, metal-ligand length ratios,
metal-vector balance, bond spread, and covalent/van-der-Waals packing.

## New finite analytic family

The following intensive descriptors are frozen before any selected ODAC23 row
label is inspected:

- elemental composition: H, C, donor, and heavy-nonmetal fractions; atomic
  number and electronegativity mean/dispersion;
- packing: atom density and volume per atom;
- constraint/connectivity: degree dispersion and quantiles, degree-one and
  degree-two fractions, low-degree heavy fraction, degree-two organic fraction,
  and covalent edge excess per atom;
- hinge geometry: mean and upper-tail normalized bend of degree-two heavy
  nonmetals and their bent-hinge fraction;
- bond chemistry: heteroatomic, metal-donor, and organic-organic edge fractions,
  donor contact with metals, metal-neighbour element diversity, and
  electronegativity contrast;
- strain/orientation: covalent-radius bond-ratio mean, dispersion, maximum,
  short/long tails, and minimum/maximum eigenvalues plus anisotropy of the
  global bond-direction second moment.

All quantities are deterministic, translation/permutation invariant, and
intensive under exact supercell replication.  No learned transformation is
used.  The sealed feature table is built from selected metadata and x0 archive
only; the selected offline-label parquet is not an input.
