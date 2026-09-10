# Model, units and numerical methods

This is an educational, synthetic radial heat-diffusion model. It develops
numerical modelling and ML integration skills. It does not solve equilibrium,
gyrokinetics, particle transport or current diffusion, and has no experimental
validation. The scenarios are heating examples, not Ohmic, H-mode or ITER predictions.

## Normalization

Let `rho = r/a` and `t_star = t/t_ref`. Temperature remains in eV. The stored
coefficient called `chi` is **normalized diffusivity**:

\[
\chi_\star = \frac{t_{\rm ref}}{a^2}\chi_{\rm physical},\qquad
S_\star = t_{\rm ref} S_{\rm eV/s}.
\]

Neither a minor radius nor a reference time is assigned in these examples.
Consequently `chi=1` is not a claim of 1 m²/s. `S0_e` is a peak heating rate
in eV per normalized time, not total heating power. The code does not infer MW,
density-dependent heat capacity or a gyroBohm scale. Unused device parameters
have been removed from the default configuration.

`rho` is a normalized cylindrical radius. Calling it a normalized toroidal
flux coordinate would require an equilibrium mapping that is absent here.

## Coupled equations

With prescribed equal, constant normalized heat capacities:

\[
\partial_{t_\star}T_e = \frac1\rho\partial_\rho
 (\rho\chi_e\partial_\rho T_e)+S_e-\nu(T_e-T_i),
\]
\[
\partial_{t_\star}T_i = \frac1\rho\partial_\rho
 (\rho\chi_i\partial_\rho T_i)+S_i+\nu(T_e-T_i).
\]

The exchange convention is `nu = 1.5 * density / tau_eq`. Here the legacy
parameter name `density` denotes a normalized exchange multiplier; `tau_eq`
is an effective normalized time. This is a prescribed relaxation law, not a
self-consistent collisional model. The exchange terms cancel in the summed
equation. In an isolated exchange step, `Te + Ti` is conserved and `Te - Ti`
decays as `exp(-2*nu*dt)`.

At the axis the solution is symmetric; at `rho=1` both pedestal temperatures
are imposed. The boundary is a heat reservoir. A weakly heated electron core
can be colder than its electron pedestal when it transfers heat to an ion
channel with a colder pedestal. `Ti <= Te` is tested in selected examples,
not asserted as a universal theorem for arbitrary heating and diffusivities.

The source is `S0 * exp(-(rho-rho_dep)^2/(2*sigma^2))`. The analytic closure is

\[
g=\max(-\partial_\rho T/T,0),\qquad
\chi=\chi_{\rm floor}+\chi_s\max(g-g_{\rm crit},0)^\alpha.
\]

The parameter `chi_neo` is a **prescribed floor**, not a neoclassical transport
calculation. The helper `normalised_flux` returns the algebraic proxy `chi*g`;
it is not a physical or gyroBohm-normalized heat flux.

## Discretization and nonlinear convergence

A uniform node grid includes the axis and outer boundary. Interior diffusion
uses shared face coefficients `F[i+1/2] = (V'[i]*chi[i] + V'[i+1]*chi[i+1])/2`.
Interior control-volume weights are `V'[i]*h`; the axis weight is
`V'[1]*h/8`. The same face flux enters adjacent cells with opposite signs.
The boundary row imposes the pedestal exactly. Global source-to-boundary-flux
balance and a quadratic manufactured solution are regression tests.

Two distinct entry points prevent confusing elapsed time with equilibrium:

- `solver.solve` and `solve_two_channel` advance a specified duration. Their
  per-step change history is a transient diagnostic. `steady_residual` and
  `converged` separately report the final stationary balance. The two-channel
  theta scheme includes exchange in the block matrix; it has no splitting error.
- `run_picard` and `run_picard_multichannel` solve stationary equations.
  Frozen transport is solved elliptically with simultaneous exchange. A
  residual line search under-relaxes the update. Newton iteration and, when
  needed, Powell's hybrid method solve the **same closure** when Picard stalls.
  Positivity is enforced by accepting positive trial profiles, not clipping.

For each channel the stopping defect is
`||L(chi(T))*T + S +/- exchange||_2 / max(||S_interior||_2, 1)`;
the maximum channel defect must be below `tol`. Boundary defects are included.
The denominator is independent of the temperature iterate and mixing factor.
The returned diffusivity is recomputed from the returned profile.

`n_iters` counts outer updates/checks; `hybrid_function_evaluations` separately
records any internal hybrid rescue work. `transport_evaluations` includes
trial profiles and numerical Jacobians. The colored Jacobian assumes each
transport output depends only on the corresponding local gradient and fixed
parameters. Globally coupled custom closures are outside this interface.

The legacy stationary arguments `dt`, `theta`, `sub_steps` and
`divergence_patience` remain accepted but do not control the stationary solve.
New configurations omit them. Model fallback requires an explicitly provided
fallback; app and CLI failures are visible, not replaced by analytic output.

Crank–Nicolson (`theta=0.5`) is second order in time and A-stable, but is not
unconditionally positivity-preserving: unresolved transients can oscillate.
`theta=1` is backward Euler. Spatial refinement is measured against independent
fine quadrature; temporal order is measured against the exact semidiscrete
matrix exponential. Nonlinear profile accuracy still depends on grid resolution.

## Geometry scope

The supported examples use circular geometry, `V'=rho`. Historical
`vprime_miller` computes `kappa*(1+delta²/2)*rho`. This constant factor cancels
from diffusion and cannot model shaping. The function is retained to reproduce
old configurations and is explicitly labelled as a legacy rescaling. The app
no longer offers misleading shaping controls. A future shaped model needs
consistent volume derivatives and radial metric coefficients, with verification.

## Context and references

Critical-gradient closures are educational approximations also discussed in
[the TORAX physics documentation](https://torax.readthedocs.io/en/stable/physics_models.html).
[TORAX (Citrin et al.)](https://arxiv.org/abs/2406.06718) is a substantially more
complete transport simulator, with coupled heat, particle and current equations
and verification against another transport code. It provides context for the
field; this repository does not claim comparable scope or validation.
