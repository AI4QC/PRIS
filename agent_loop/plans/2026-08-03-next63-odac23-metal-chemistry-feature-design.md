# NEXT63 frozen metal-chemistry x0 feature design

## Boundary and motivation

NEXT61/NEXT62 show that framework density, bond orientation, and metal-ligand
geometry carry signal but do not meet precision.  The current catalogue largely
forgets metal identity.  NEXT63 adds deterministic chemical-table descriptors
from one raw x0.  Internal validation, replication, and official validation
remain unopened.

Allowed values are atomic numbers, raw periodic geometry/covalent graph, and
the installed frozen pymatgen elemental table: group, row, Pauling
electronegativity, atomic radius, Mendeleev number, first ionization energy, and
listed common/max oxidation states.  No oxidation state is fitted to the
structure; each metal uses the mean positive common oxidation state, falling
back to its positive maximum.  No DFT value, relaxed geometry, learned proxy,
alternative structure, or physical relaxation is used.

## Frozen features

- O, N, S, P, and halogen atom fractions;
- metal species count and composition entropy;
- metal-only atomic-number, group, row, electronegativity, Mendeleev-number,
  ionization-energy, atomic-radius, common-oxidation, and maximum-oxidation
  aggregates;
- analytic metal hardness (`ionization_energy / radius`) and charge-density
  (`common_oxidation / radius^3`) proxies;
- covalent-graph metal coordination divided by common oxidation state;
- metal-donor electronegativity-gap, distance, and covalent-radius bond-ratio
  distributions.

All aggregates are intensive under exact supercell replication except the
explicit metal species count, which counts distinct elements and is likewise
replication invariant.  Missing elemental fields fail open.  NEXT63 appends
these fixed features to NEXT58 without accepting a label path.
