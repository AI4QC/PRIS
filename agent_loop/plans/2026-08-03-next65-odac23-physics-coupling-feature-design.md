# NEXT65 predefined x0 physics-coupling features

NEXT61--NEXT64 repeatedly surface packing, bond-direction anisotropy,
metal-donor strain, hinge bending, and local motif order.  An additive score may
miss the mechanistic condition that two effects coexist.  Before evaluating any
new result, NEXT65 freezes twelve algebraic couplings over already sealed
NEXT63 x0 features:

- atom density times metal-donor ratio dispersion, distance q95, or
  metal-ligand ratio q95;
- atom density or heteroatomic-edge fraction divided by bond-orientation
  lambda-min;
- metal-donor ratio dispersion divided by donor motif order minimum;
- metal-donor ratio maximum divided by global motif order minimum;
- degree-two bend q95 times metal-donor ratio dispersion or metal-ligand q95;
- volume per atom times metal-donor ratio dispersion;
- donor motif entropy q95 times metal-donor ratio dispersion;
- metal-donor electronegativity-gap q95 times metal-donor ratio dispersion.

Divisors use a fixed floor of `1e-6`.  Inputs must all be finite; otherwise the
interaction row is unsupported and any formula using it keeps the structure.
The builder reads only the sealed label-free NEXT63 table.  All expressions are
deterministic, intensive, and contain no DFT, relaxed geometry, learned proxy,
or alternative structure.
