# NEXT86 clarification: freeze the term catalogue before endpoint routing

Date: 2026-08-03

This is an additive clarification to
`2026-08-03-next83-next89-scigen-independent-dft-validation-design.md`. The
original design and every artifact that hashes it remain unchanged.

Before any SCIGEN endpoint payload is opened, NEXT86 will publish a finite
candidate-term catalogue from the already frozen discovery x0 features. It may
use only label-free finite coverage, unique-value count, and robust location
and scale statistics. It may not read `output.dat`, `CONTCAR`, a supplementary
CSV, or any endpoint artifact.

Eligibility rules are fixed as follows:

- the feature must be in the explicit physics-direction map implemented in
  `src/next86_scigen_term_catalogue.py`;
- discovery finite coverage must be at least 0.90;
- at least 16 distinct finite transformed values must exist;
- the robust transformed scale `(q90 - q10) / 2` must exceed `1e-12`;
- element identity, material/formula/chemical-system identity, lattice-class
  identity, raw atom/edge/site counts, and ambiguous unsigned energy-like
  quantities are excluded;
- each feature has exactly one predeclared direction (`+1` means high is risky,
  `-1` means low is risky) and one transform (`log1p_nonnegative` or `asinh`);
- the hinge center is the transformed discovery median and the scale is the
  robust transformed scale above.

The catalogue records both eligible and excluded prespecified terms with the
reason for exclusion. Formula search may use only eligible terms exactly as
serialized. Validation and replication feature distributions do not set any
center, scale, direction, transform, or eligibility decision.

After the catalogue and its hashes are frozen, a separate deterministic router
may mechanically split the official endpoint table into physically separate
discovery, internal-validation, and internal-replication artifacts. The router
must not print or summarize endpoint values. Only discovery endpoints may then
be opened for law search.
