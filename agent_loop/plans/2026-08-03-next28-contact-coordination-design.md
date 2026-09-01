# NEXT28 fixed contact-coordination law design

## Boundary

- Runtime input is one unrelaxed periodic x0 only.
- Runtime uses tabulated van der Waals/covalent radii, exact periodic images,
  deterministic graph paths, counting, and division only.
- No DFT value, relaxed structure, trajectory endpoint, MLIP, energy, force,
  stress, or same-composition alternative is available to the executable law.
- DFT response values are development/evaluation labels only.

## Law

After exact 1-2/1-3/1-4 periodic path exclusion, define

`C_1.05 = N^-1 sum_i n_i(q_vdW <= 1.05)`.

The fixed law is `REJECT iff C_1.05 >= 6.3`; missing or unsupported geometry
fails open to `KEEP`.  It has one term and one rounded cutoff.

## Validation

The six opened NEXT27 prospective shards become development-only.  Freeze the
law before decoding any later archive member.  On later CSD-refcode-disjoint
shards require aggregate Wilson lower bounds: coverage 0.95, protected-negative
rate 0.95, rejection precision 0.75, and savings 0.04.

Passing these gates supports only screening for the frozen severe initial-DFT-
response endpoint in OMC25 molecular crystals.  It does not establish
thermodynamic stability or a beyond-Pauling claim.
