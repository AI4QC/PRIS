# NEXT379--NEXT382 periodic skeletal-net bottleneck

Date: 2026-08-13

Status: frozen before computing any NEXT379 value or opening any NEXT379
outcome. This is a new additive branch after the frozen negative NEXT377 PCRL
audit. PCRL is not modified, reversed, thresholded, or combined here.

## Scope and executable boundary

The target remains an interpretable pre-DFT screen for a generated or
theoretical periodic crystal. The executable may use element identities and
one submitted raw initial unrelaxed periodic geometry only. It must not
execute or consume DFT, endpoint values, energies, forces, stresses, their
learned proxies, model potentials, relaxation, trajectories, or later
geometries. Discovery outcomes may be opened only after all label-blind gates
and the complete label-free formal build pass. Validation and replication
remain physically sealed.

All files are additive. Existing scripts and content remain unchanged. Only
the independent no-DFT report may be extended before user review; canonical
`paper/`, `tex/`, `notes/`, `README.md`, and `PREREG.md` are outside this
branch.

## Literature mechanism and existing-coverage audit

Blatova and Blatov, *Acta Crystallographica A* 80 (2024),
"Hierarchical topological analysis of crystal structures: the skeletal net
concept", describes a hierarchy obtained by successively removing weaker
contacts and defines a skeletal net as the minimal strongest-contact network
that preserves crystal periodicity. The executable below imports only this
purely geometric/topological principle. It does not call ToposPro, query a
crystal database, assign interaction energies, or use an electronic-structure
quantity.

The repository already covers related but non-equivalent mechanisms:

1. NEXT166 computes translation ranks on two already fixed contact graphs. It
   does not filter contacts by strength or measure the strength at which a
   site's component first becomes three-periodic.
2. NEXT259 computes periodic bottleneck persistence on the *void* graph. Its
   nodes are Voronoi vertices and its capacities are free-sphere passage
   radii; it is not an atomic skeletal contact net.
3. NEXT239/NEXT251 summarize Voronoi face weights and topology without a
   periodic-connectivity filtration. NEXT331 tests only the weakest radical
   facet. NEXT375 tests independently chosen local coordination prefixes and
   endpoint reciprocity, not global periodic growth.
4. NEXT295/NEXT323 and NEXT351/NEXT355 test force-closure or kinematic
   rigidity. They do not ask when strong contacts alone acquire topological
   rank three.

The retained missing statistic is therefore a strength-filtered periodic
growth threshold on the ordinary atomic Voronoi contact graph.

## Frozen graph and formula

First apply the unchanged deterministic NEXT267 Minkowski reduction and
periodic wrapping. Construct the ordinary periodic Voronoi tessellation with
`VoronoiNN(weight="solid_angle", tol=0, cutoff=13)`. A directed facet is
identified exactly by `(i,j,T)` and must have the reverse `(j,i,-T)`. Reverse
solid angles must agree within `1e-8`; their arithmetic mean is the shared
facet solid angle `omega_e`. Duplicate or incomplete incidences fail closed.

For every site `i`, let `Omega_i` be the largest shared solid angle among all
incident periodic facets. For undirected facet edge `e=(i,j,T)`, freeze its
mutual local salience as

```text
s_e = min(omega_e / Omega_i, omega_e / Omega_j),  0 < s_e <= 1.
```

For a self-image edge the two endpoint normalizers are the same. Saliences
are quantized at `1e-10`. For each threshold `tau`, retain all edges with
`s_e >= tau`, treating exact salience ties simultaneously. Each connected
component has an exact periodic translation rank: choose integer quotient-
graph vertex potentials and take the rational rank of all cycle-closure
translations in `Z^3`.

For each quotient site `i`, define

```text
b_i = max tau such that i belongs to a retained component of translation rank 3,
```

and set `b_i=0` if no rank-three component ever contains it. The sole public
feature is

```text
psnb_skeletal_3d_bottleneck_q10
    = inverted-CDF 0.10 quantile of {b_i over all quotient sites},
```

quantized at `1e-10`. The sole direction is `protected_high`: a structure is
hypothesized to be better protected when at least 90% of its sites enter a
three-periodic component using only mutually salient contacts. This is a
topological screening hypothesis, not a theorem of energetic stability.

No alternative facet weight, graph, normalization, rank target, quantile,
threshold, element subgroup, transformation, composite, or opposite
direction is searched. Solid angle rather than face area makes the formula
invariant to uniform geometric scale. The mutual minimum requires a contact
to be salient from both endpoints and is fixed before any value is computed.

## Analytic and representation tests

The pure quotient-graph kernel must cover a one-site three-axis periodic net
with an exact known bottleneck, a rank-deficient net, multiple components,
simultaneous ties, edge-order invariance, uniform weight-scale invariance,
and disjoint exact replication. Malformed translations, duplicate edges,
missing reverse incidences, and inconsistent reverse weights fail closed.

Structure tests must cover rigid rotation, periodic translation, site
permutation, unimodular lattice rebasing, and an explicit integer supercell.
The geometry-only firewall fails closed on calculators, metadata, extra
arrays, nonperiodicity, and nonfinite coordinates.

## Label-blind sequential gates

The deterministic probe uses the same 80 discovery structures per source and
reads raw discovery geometry plus label-free controls only. The novelty
population contains every numeric NEXT85/NEXT94 base feature and every formal
later label-free feature through NEXT375 PCRL. A control is eligible only with
at least 40 jointly finite, nonconstant probe rows.

Each source must independently satisfy:

- support at least `72/80`;
- all supported values in `[0,1]` and at least 20 distinct values at 10
  decimals;
- maximum equivalent-representation error at most `1e-8`;
- maximum adequate absolute label-free Spearman correlation strictly below
  `0.90`.

No endpoint is read during this probe. Failure of any gate terminates the
branch, records `next380_formal_build_authorized=false`, and prohibits
NEXT380--NEXT382.

Only a passing probe authorizes NEXT380's immutable all-row discovery
feature build. Formal coverage must be at least `0.90` independently in all
13,470 SCIGEN and 5,232 WyFormer discovery rows. Only then may NEXT381 reuse
the unchanged NEXT224/NEXT268/NEXT324 rejected-extreme discovery audit:
reduced-formula five-folds, inverted-CDF `1/16` and `15/16` normalization,
minimum coverage `0.90`, minimum class count `20`, pooled AUC `0.55`, macro
AUC `0.53`, worst-fold AUC `0.50`, and the frozen `protected_high` direction.

Failure in either source yields zero eligible hypotheses and terminates the
branch. NEXT382 may exist only if NEXT381 explicitly authorizes the
predeclared formula-search stage. Nothing may be reversed, imputed, repaired,
or tuned after outcomes are opened.

## Decision log

- This branch follows a published purely topological skeletal-net principle,
  but the precise bounded statistic is a prospective project hypothesis.
- It does not use DFT, electron density, a force field, an MLIP, relaxation,
  or another structure of the same composition.
- Atom-contact and void-network bottlenecks are kept distinct; the strict
  novelty gate tests whether that distinction produces new information.
- The q10 population aggregate is fixed to require broad site participation
  while remaining invariant under exact supercelling.
- Any unsupported structure abstains; no missing feature is imputed.
