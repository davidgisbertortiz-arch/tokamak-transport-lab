# Synthetic ML and uncertainty protocol

## What is learned

Version-2 data contain five active inputs, in this exact order:
`a_over_LTe`, `chi_s`, `a_over_LTe_crit`, `alpha_s`, `chi_neo`.
The target is **normalized diffusivity**, evaluated from the analytic closure.
The old safety factor, magnetic shear and collisionality inputs were removed
because they did not affect the labels. No experimental data or gyrokinetic
calculations are used.

Training uses 20,000 Latin-hypercube samples. Validation, local calibration
and local test each use 3,000 independent uniform samples, with different
seeds. IID calibration/test sampling is intentional: a dependent LHS design
should not be treated as an IID conformal experiment.

The MLP has three 64-unit SiLU layers and a **linear scalar output**. Inputs
are standardized using training statistics. The target is standardized
`log(chi)`, so its normalized values can have either sign. This removes the
old incompatibility between centered targets and a nonnegative output head.
Checkpoints are selected by validation log-MSE. Test data do not select weights.

Inference maps back with `exp`. The known floor is exact below threshold.
Above threshold, the learned excess is anchored continuously:

```
chi = chi_neo + max(exp(predicted_log_chi(g))
                   - exp(predicted_log_chi(g_crit)), 0)
```

This uses a specified structural prior of the synthetic closure. It does not
learn the threshold or floor physics from observations. Anchoring prevents a
jump at the threshold that can obstruct a nonlinear solver even when global
regression metrics look good. Monotonicity and extrapolation accuracy of the
neural component are not guaranteed.

`TransportBundle` contains weights, feature order, bounds, architecture,
normalization, inverse transform and inference-rule identifier. Loading uses
PyTorch's weights-only mode. Adapters consume raw physical-feature values
and return `chi` directly, avoiding division by a vanishing gradient. Inference
uses CPU float64 and one PyTorch thread for small solver batches. Model hashes
appear in result metadata and calibration files.

A single trained MLP and a three-member independent ensemble are available.
Ensemble predictions average member **diffusivities**, not normalized logits.
The learner is more expensive than this very cheap analytic toy law; no
acceleration claim is made. This exercise demonstrates the integration pattern
that would matter for a costly closure.

## Three different uncertainty outputs

| Output | Construction | Meaning |
|---|---|---|
| Local diffusivity interval | Split conformal residual in log diffusivity | Marginal coverage over IID local inputs from the declared box |
| Ensemble profile range | Solve the full stationary problem for each member | Descriptive disagreement; no coverage probability |
| Calibrated profile interval | Split conformal maximum error across both entire profiles | Simultaneous nodal Te/Ti coverage, marginal over the specified synthetic scenario law |

Local prediction intervals are never rescaled arbitrarily into temperature
intervals. Local inputs encountered along a solved profile are not IID samples
from the training box, so the local coverage statement does not automatically
apply to that trajectory.

For split conformal calibration, sort the `n` scores and use order statistic
`ceil((n+1)*(1-alpha))`. There is no interpolation. If the rank exceeds `n`,
the interval is unbounded. Invalid shapes, empty arrays and nonfinite values
are rejected. JSON persistence represents an unbounded interval explicitly.
The construction and its exchangeability assumption follow
[Angelopoulos and Bates](https://arxiv.org/abs/2107.07511).

## Profile experiment

`python -m scripts.calibrate_profiles` independently draws 64 calibration and
64 test scenarios. Each uses 40 radial nodes and uniform independent draws of:

- Electron source amplitude: 300–1000 eV per normalized time.
- Electron pedestal: 300–700 eV; ion pedestal is 0.8 times the electron value.

Geometry, exchange, source shape and closure parameters are fixed and recorded
in `profile_uq.FAMILY` and the calibration JSON. For each scenario, solve both
the analytic reference and the ensemble-mean closure. No failed scenario is
dropped. Its score is the largest absolute temperature discrepancy across
both channels and all radial nodes. A common conformal radius is then added
and subtracted from each predicted profile; known pedestal values remain exact.

The reference is the analytic **discrete synthetic model on the same grid**.
These intervals measure surrogate error, not discretization error, measurement
noise or uncertainty in real transport physics. The app enables the calibrated
profile option only for this experiment family and checks the model hash.
Coverage is marginal over random scenarios; selecting a particular slider
value does not acquire a conditional guarantee.

The saved evaluation covered 58/64 independent test scenarios (90.625%) for
a nominal 90% interval. With 64 cases the empirical rate is imprecise; it is
a reproducible finite experiment, not proof of device-level predictive validity.
See [the generated summary](validation/summary.json) for all metrics and hashes.
