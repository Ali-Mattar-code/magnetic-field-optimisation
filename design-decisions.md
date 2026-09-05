# Engineering and research decisions

## Explicit ampere-turns

Current constraints are meaningless without turns. The configuration therefore exposes `turns`, and the field model multiplies each segment contribution accordingly.

## Power is not inductance

For a DC resistive system, the heat loss is `I^T R I`. Mutual inductance affects transients and stored energy, `0.5 I^T L I`; it is not an extra steady-state loss term. The electrical module keeps these quantities separate.

## Complete inductance matrices

Approximate mutual-inductance matrices are symmetrised and projected to the positive semidefinite cone before use in convex energy terms. This prevents numerical artefacts from masquerading as physical energy reduction.

## Physics remains authoritative

The ML surrogate is used for screening and warm starts. Candidate designs are always checked using the forward model and constrained solver.

## Claims are versioned

Historical thesis metrics live in their own configuration. Reproduction scripts never copy those numbers into new results.

