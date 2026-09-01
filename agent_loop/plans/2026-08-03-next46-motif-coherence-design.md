# NEXT46 local-motif coherence design

NEXT43--45 mainly test contact magnitudes, global analytic fields, rigidity, and low-dimensional conditional laws. NEXT46 adds a different x0-only mechanism: whether each site has a clear local coordination/polyhedral signature, and whether sites of the same element have mutually consistent local environments.

For every raw x0 site, the deterministic CrystalNN/Voronoi `ops` fingerprint is computed. The executable descriptors aggregate coordination-weight completeness, dominant-CN clarity, CN entropy, effective-CN dispersion, ideal-polyhedron strength, fingerprint norm, within-element fingerprint dispersion, global dispersion, and between-species centroid separation. Only geometry and frozen elemental radii are used. No endpoint, DFT value, model potential, learned energy/force/stress proxy, or relaxation is available to the feature program.

The new label-free table remains separate. After sealing, it is joined with NEXT43/NEXT44 and searched with the unchanged deterministic split, finite formula catalogue, KEEP-on-missing policy, and primary gates. It is development-only until a formula passes both internal splits and is then frozen before unseen-shard endpoint opening.
