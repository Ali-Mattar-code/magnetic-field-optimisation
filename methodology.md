# Methodology

## Forward model

Each circular loop is discretised into straight current elements. For observation point `r`,

\[
d\mathbf B = \frac{\mu_0 N I}{4\pi}\frac{d\boldsymbol\ell\times(\mathbf r-\mathbf r')}{\lVert\mathbf r-\mathbf r'\rVert^3}.
\]

Evaluating each coil at unit current constructs the influence matrix `A`; arbitrary current vectors then use linear superposition.

## Inverse problem

The solver balances target accuracy, off-target leakage, and Ohmic heating:

\[
\min_I \; \lVert W(AI-b)\rVert_2^2 + \lambda I^T R I
\]

subject to a target-field tolerance and per-channel current limits. The SciPy backend solves the convex quadratic with SLSQP. The optional CVXPY backend represents the same program directly.

## Metrics

- target relative error;
- full width at half maximum (FWHM);
- leakage ratio outside a fixed guard region;
- peak side-lobe ratio;
- Ohmic power and peak current;
- stored magnetic energy;
- robustness quantiles under manufacturing and control perturbations.

## Robustness

Monte Carlo draws perturb coil currents, resistance, position, and tilt. Every perturbed design is evaluated by the forward model. Random seeds and distributions are stored in the output JSON.

## Learning aid

The surrogate predicts solver outputs over a restricted geometry family. In addition to raw radius, spacing, and depth, it receives dimensionless ratios and the analytical on-axis loop-field scale. These physics-informed features encode known geometry relationships without replacing the numerical model. The surrogate proposes candidates, but all reported candidates are re-evaluated using the physics solver.
