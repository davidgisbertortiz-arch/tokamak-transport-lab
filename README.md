# Tokamak Transport Lab

[![CI](https://github.com/davidgisbertortiz-arch/tokamak-transport-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/davidgisbertortiz-arch/tokamak-transport-lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

**A personal scientific-computing project for learning radial heat transport,
nonlinear solvers, neural surrogates and uncertainty quantification.**

The lab solves coupled electron–ion temperature equations in circular geometry.
An analytic critical-gradient closure provides synthetic data for a PyTorch MLP
and an ensemble. The learned closures run inside the stationary solver, and
separate experiments measure local prediction error and full-profile uncertainty.

The physics is deliberately simplified and normalized. There are no experimental
plasma data, gyrokinetic calculations or validated device predictions.

![Converged synthetic heating sweep](assets/demo.gif)

*Electron heating amplitude varies from 300 to 1000 eV per normalized time.
Every displayed frame converged. [Frame configurations and metrics](assets/demo.json).*

## Run locally

Requires Python 3.11 or newer; CI is configured for 3.11 and 3.12.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,demo,gif]'

# Analytic calculations need no trained model or PyTorch.
python -m scripts.run_solver_verification
python -m scripts.run_scenarios
streamlit run src/tokamak_transport_lab/app/demo.py
```

For the complete ML workflow, install PyTorch. This CPU-only installation is
sufficient for the small networks in this project:

```bash
python -m pip install 'torch>=2.1,<2.6' --index-url https://download.pytorch.org/whl/cpu
python -m scripts.reproduce --gif
```

The reproduction command generates data, trains a single MLP and three-member
ensemble, runs the scenarios, calibrates and tests two uncertainty experiments,
and produces `outputs/validation/summary.json` and `validation.png`. It stops
on a failed calculation. Reuse trained models with `--skip-training`.

For the exact tested Linux/CPython 3.11 dependency set:

```bash
python -m pip install -r requirements-linux-cpu.txt
python -m pip install --no-deps -e .
```

Generated datasets and model weights are in ignored `data/` and `outputs/`.
The repository includes compact validation results, figures and their provenance.

## What has been checked

![Numerical and surrogate validation](docs/validation/validation.png)

| Check | Recorded result |
|---|---|
| Spatial refinement against independent quadrature | Observed order 2.00 |
| Stationary initial-condition independence | Initial temperatures of 500, 1500 and 10,000 eV give the same solution within 1e-6 eV |
| Coupled energy balance | Sources match net boundary heat flux; exchange cancels between channels |
| Actual YAML scenarios | All converge without temperature clipping or model fallback |
| Ensemble diffusivity prediction | Test R² = 0.99991 on 3,000 synthetic examples |
| Coupled reference profile | Maximum Te/Ti discrepancy ≈ 3.26 eV for the documented reference case |
| Local diffusivity interval | 89.43% coverage on 3,000 independent test examples; nominal 90% |
| Simultaneous temperature interval | 58/64 independent test scenarios covered (90.625%); nominal 90% |

Full values, configuration scope, model hashes and environment versions are in
[the generated summary](docs/validation/summary.json).
The temperature interval is calibrated against the analytic **discrete synthetic
model**, simultaneously across both channels and the 40-node grid. Its coverage
is marginal over a specified random source/pedestal experiment. It does not
cover arbitrary user-selected cases, grid error or uncertainty in real physics.

## The model

For normalized radius `rho = r/a` and normalized time:

\[
\partial_t T_{e,i} = \frac{1}{\rho}\partial_\rho
\left(\rho\chi_{e,i}\partial_\rho T_{e,i}\right)
+S_{e,i}\mp\nu(T_e-T_i).
\]

Axis symmetry and imposed outer pedestal temperatures close the equations.
The closure is

\[
\chi = \chi_{\rm floor}+
\chi_s\max\left(0,-\frac{\partial_\rho T}{T}-g_{\rm crit}\right)^\alpha.
\]

`chi` is a normalized coefficient and `S0_e` is a peak heating rate; neither
is labelled as a dimensional diffusivity or device power. The radial operator
is conservative. Crank–Nicolson handles transients, while the stationary solver
uses elliptic Picard updates, Newton iteration and a hybrid rescue method.
Convergence is assessed from the PDE balance, independently of step size or
under-relaxation. Electron–ion exchange is included in the coupled matrix.

See [the equations and numerical conventions](docs/physics.md) and
[the ML and uncertainty protocol](docs/ml-and-uncertainty.md).
The historical constant “Miller” rescaling is retained only for legacy
configuration regression; it provides no shaping physics and is absent from
the interactive controls.

## Experiments and individual commands

| Command | Purpose |
|---|---|
| `python -m scripts.run_solver --config configs/solver.yaml` | Run a finite transient and report its stationary residual |
| `python -m scripts.run_picard --config configs/picard.yaml --run-id example` | Single-channel stationary solve |
| `python -m scripts.run_picard --config configs/scenarios/mid_power.yaml --run-id coupled` | Coupled stationary solve |
| `python -m scripts.run_scenarios` | Compare the three normalized heating scenarios |
| `python -m scripts.generate_dataset` | Generate train, validation, calibration and test data |
| `python -m scripts.train_surrogate` | Train and save a complete single-model bundle |
| `python -m scripts.train_ensemble` | Train three independent members |
| `python -m scripts.calibrate_conformal` | Local log-diffusivity intervals with independent evaluation |
| `python -m scripts.calibrate_profiles` | Simultaneous temperature intervals with independent evaluation |
| `python -m scripts.validate_project` | Generate numerical/ML validation evidence |
| `python -m scripts.make_readme_gif` | Render converged experiments and save frame provenance |

To use a trained closure from YAML, add:

```yaml
model:
  type: ensemble  # analytic, mlp, or ensemble
  path: outputs/ensemble/model.pt
```

The app and CLIs use the same configuration dispatcher. A missing model raises
an explicit error. The app records which model produced the displayed result,
marks stale controls, shows nonconvergence and extrapolation, supports source
sweeps, and downloads profiles with their configuration and metrics.

Ensemble member ranges propagate each member through a complete solve. They
are descriptive ranges; the separately calibrated profile interval is available
only for the **Profile benchmark** experiment and matching model artifact.

## Development and project scope

```bash
python -m pytest
ruff check src/ tests/ scripts/
ruff format --check src/ tests/ scripts/
```

Tests cover analytic and manufactured solutions, spatial and temporal order,
energy balance, false convergence, the original problematic scenario configs,
training-to-inference persistence, real model dispatch, conformal order
statistics and the Streamlit run/error/stale-result paths. CI also executes
the solver verification and YAML scenarios and uploads their outputs.

The main modules are `solver/` (linear operators and transients), `integration/`
(stationary iteration and shared dispatch), `transport/` (closures/adapters),
`data/`, `surrogate/`, and `app/`. Entry points live in `scripts/`.

For the repository repair rationale and a restrained portfolio description,
see [the validation notes](docs/validation-notes.md) and
[CV and interview notes](docs/portfolio.md).

## License

[MIT](LICENSE).
