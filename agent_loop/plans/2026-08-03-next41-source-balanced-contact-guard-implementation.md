# NEXT41 source-balanced contact-guard implementation plan

1. Add failing tests for generic frozen-geometry validation, deterministic
   contact-feature publication, forbidden endpoint flags, hashes, and no replace.
2. Implement and run the label-free contact feature builder on WBM and OMat24.
3. Add failing tests for the finite two-source conjunction catalogue and
   per-source gates.
4. Implement the source-balanced search, run it once, and freeze at most one
   eligible law without touching any confirmation source.
5. If eligible, audit and freeze a physically separate confirmation source; if
   none is eligible, write an independent negative NEXT41 report.
