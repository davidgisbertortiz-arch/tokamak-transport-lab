# tokamak-transport-lab

[![CI](https://github.com/davidgisbertortiz-arch/tokamak-transport-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/davidgisbertortiz-arch/tokamak-transport-lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

> Interactive integrated modelling playground: 1-D diffusion solver, ML transport surrogate, calibrated uncertainty — all on synthetic data, runnable on a laptop.

<p align="center">
  <img src="assets/demo.gif" alt="P_heat sweep demo" width="600">
</p>

## Quick Demo

```bash
git clone https://github.com/davidgisbertortiz-arch/tokamak-transport-lab.git
cd tokamak-transport-lab
pip install -e ".[dev,demo,gif]"

# Interactive app (opens in browser)
streamlit run src/tokamak_transport_lab/app/demo.py

# Headless GIF for README
python -m scripts.make_readme_gif --config configs/scenarios/mid_power.yaml --n_frames 10
```

The Streamlit app includes a **sweep mode** — vary P_heat (or any parameter)
across N steps and overlay all Te(ρ)/Ti(ρ) curves with a summary table.
Outputs go to `outputs/` (git-ignored).

## Overview

**tokamak-transport-lab** couples a custom 1-D Crank–Nicolson diffusion solver with a deep-ensemble ML surrogate of turbulent transport in a self-consistent Picard iteration loop, producing predictive plasma temperature profiles with calibrated uncertainty.  The solver supports **coupled electron–ion** ($T_e$–$T_i$) channels with collisional equilibration.

```
Config (YAML)  ──▶  Solver (CN-FDM)  ──▶  Profiles + Plots + NPZ
                        ▲
                 Surrogate / Stiffness model
```

## Physics

### Transport equation

The solver advances the electron temperature $T_e(\rho, t)$ on a normalised
toroidal flux coordinate $\rho \in [0, 1]$:

$$\frac{\partial T_e}{\partial t} = \frac{1}{V'(\rho)}\frac{\partial}{\partial \rho}\!\left[V'(\rho)\,\chi(\rho)\,\frac{\partial T_e}{\partial \rho}\right] + S(\rho)$$

For circular geometry the flux-surface volume element is $V'(\rho) = \rho$.

**Boundary conditions:**
- Symmetry (Neumann) at the magnetic axis: $\left.\dfrac{\partial T_e}{\partial \rho}\right|_{\rho=0} = 0$
- Pedestal (Dirichlet) at the separatrix: $T_e(1) = T_\text{ped}$

### Heating source

A Gaussian deposition profile centred at $\rho_\text{dep}$:

$$S(\rho) = S_0 \exp\!\left(-\frac{(\rho - \rho_\text{dep})^2}{2\sigma^2}\right)$$

### Semi-analytic steady-state reference

At steady state ($\partial T_e/\partial t = 0$) the PDE reduces to an ODE.
Defining the cumulative source integral $I(\rho) = \int_0^\rho \rho' S(\rho')\,d\rho'$,
the solution is:

$$T_e(\rho) = T_\text{ped} + \int_\rho^1 \frac{I(\rho'')}{\rho''\,\chi}\,d\rho''$$

Both integrals are evaluated by cumulative trapezoidal quadrature (4000 points).

### Stiffness transport model

The turbulent diffusivity follows a critical-gradient (stiffness) law:

$$\chi_\text{turb} = \chi_s \cdot \max\!\left(0,\; \frac{a}{L_{T_e}} - \frac{a}{L_{T_e,\text{crit}}}\right)^{\!\alpha_s}$$

where $a/L_{T_e} = -(a/T_e)\,\partial T_e/\partial r$ is the normalised inverse
gradient length. A neoclassical floor prevents zero transport below threshold:

$$\chi_\text{total} = \chi_\text{turb} + \chi_\text{neo}$$

The ML surrogate learns to predict the normalised heat flux
$Q_e/Q_{gB} = \chi_\text{total} \cdot (a/L_{T_e})$ as a function of six
dimensionless local parameters $(a/L_{T_e},\, q,\, \hat{s},\, \nu^*,\, \chi_s,\, a/L_{T_e,\text{crit}})$.

## Installation

```bash
git clone https://github.com/davidgisbertortiz-arch/tokamak-transport-lab.git
cd tokamak-transport-lab
pip install -e ".[dev]"          # core + lint + tests
pip install -e ".[demo]"         # adds Streamlit interactive app
pip install -e ".[gif]"          # adds imageio for headless GIF generation
pip install -e ".[dev,demo,gif]" # everything at once
```

## Quick Start

Run the 1-D solver with the default config:

```bash
python -m scripts.run_solver --config configs/solver.yaml
```

This produces `outputs/solver_verification.npz` with the converged temperature
profile, diffusivity, source, and residual history.

Generate synthetic transport dataset:

```bash
python -m scripts.generate_dataset --config configs/dataset.yaml
```

This produces `data/transport_train.npz` (10 k samples) and
`data/transport_test.npz` (2 k samples) via Latin Hypercube sampling of the
stiffness model.  Override sizes with `--size_train` / `--size_test`.

Train the MLP transport surrogate:

```bash
pip install -e ".[surrogate]"   # adds torch
python -m scripts.train_surrogate --train data/transport_train.npz --test data/transport_test.npz
```

Outputs to `outputs/surrogate/`: `model.pt`, `metrics.json`, `calibration.png`.

### Picard Iteration Loop

Run a self-consistent Picard loop coupling the stiffness transport model
with the CN solver:

```bash
python -m scripts.run_picard --config configs/scenarios/mid_power.yaml --run-id demo
```

Outputs to `outputs/picard/demo/`:
- `result.npz` — rho, Te, chi, residual & alpha histories
- `metrics.json` — convergence metadata + profile stats
- `profiles.png` — Te(rho) and chi(rho) side-by-side
- `convergence.png` — residual and relaxation-parameter history

Three scenarios are provided: `low_power`, `mid_power`, `high_power`
(the last uses Miller geometry).  Each runs end-to-end in under 60 s.

### Multi-Channel Scenarios (Te + Ti)

The solver supports coupled electron–ion temperature evolution.  A
collisional equilibration term transfers energy between channels:

$$S_{ei} = \frac{3}{2}\,n\,\frac{T_e - T_i}{\tau_{eq}}$$

This appears as a sink in the $T_e$ equation and a source in the $T_i$
equation.  In electron-heated scenarios ($S_i = 0$) the physics constraint
$T_i \leq T_e$ is automatically satisfied.

#### Running scenarios

```bash
python -m scripts.run_scenarios                       # all 3 scenarios
python -m scripts.run_scenarios --scenarios-dir configs/scenarios
python -m scripts.run_scenarios --output-dir outputs/scenarios
```

#### Outputs produced

Per scenario (e.g. `outputs/scenarios/mid_power/`):

| File | Description |
|------|-------------|
| `result.npz` | Converged profiles: rho, Te, Ti, chi_e, chi_i, residual & alpha histories |
| `metrics.json` | Convergence metadata (n_iters, wall_time, final_residual, Te/Ti core) |

Global (under `outputs/scenarios/`):

| File | Description |
|------|-------------|
| `summary.json` | Comparison table with `n_iters`, `final_residual`, `runtime_s`, `Te0`, `Ti0` per scenario |
| `scenarios_compare.png` | Te(ρ) and Ti(ρ) overlays for all scenarios |
| `comparison_convergence.png` | Residual histories overlaid |

#### Reference scenarios

| Scenario | P_heat (S0_e) | Te_ped | Ti_ped | Geometry |
|----------|:------------:|:------:|:------:|:--------:|
| `low_power` | 0.3 | 300 eV | 250 eV | Circular |
| `mid_power` | 1.0 | 500 eV | 400 eV | Circular |
| `high_power` | 4.0 | 1500 eV | 1200 eV | Miller |

> **Simplifications & caveats:**  This is a *toy educational model*, not a
> production transport code.  Key simplifications include: (1) synthetic
> stiffness-based transport — not derived from gyrokinetic simulations;
> (2) 1-D radial geometry only — no poloidal/toroidal asymmetries beyond
> a lowest-order Miller shaping factor; (3) a single equilibration time
> $\tau_{eq}$ rather than a self-consistent collision operator; (4) no
> impurity radiation, neutral fuelling, or current diffusion.  Results are
> suitable for learning and prototyping, not for device prediction.

### Uncertainty (Ensemble + Conformal) & Geometry (Miller-lite)

Train a Deep Ensemble (5 members by default, each with its own seed).
The `DeepEnsemble` class takes a *model_factory* and handles per-member
seeding, training, and aggregation:

```bash
python -m scripts.train_ensemble --config configs/ensemble.yaml
```

Outputs to `outputs/ensemble/`: `member_0.pt` … `member_4.pt`,
`norm_stats.npz`, `metadata.json`, `metrics.json`.

Calibrate split conformal prediction intervals on a held-out set:

```bash
python -m scripts.calibrate_conformal --config configs/conformal.yaml
```

Outputs to `outputs/conformal/`: `conformal.json` ($\hat{q}$, $\alpha$, timestamp),
`summary.json` (empirical coverage check).

Quick demo — mean ± conformal band in Python:

```python
from tokamak_transport_lab.surrogate import DeepEnsemble, SplitConformal, TransportMLP
import torch, numpy as np

ens = DeepEnsemble.load("outputs/ensemble", TransportMLP, n_features=6, hidden=(128,128,64))
sc  = SplitConformal.load("outputs/conformal/conformal.json")
x   = torch.randn(100, 6)
out = ens.predict(x)
lo, hi = sc.predict_interval(out["mean"].squeeze().numpy())
```

**Miller geometry** — the solver can use a shaped flux-surface volume element
$V'(\rho) \approx \kappa(1 + \tfrac{1}{2}\delta^2)\rho$ instead of the
circular $V'=\rho$.  Set `kappa` and `delta` in your config or call
`vprime_miller()` directly.

> **Limitations:** The ensemble captures epistemic uncertainty only (no
> aleatoric head).  The conformal wrapper provides marginal (not conditional)
> coverage guarantees.  The Miller model is a lowest-order analytic
> approximation — not a full Grad–Shafranov solution.

### Streamlit Interactive Demo

Launch the interactive app (requires `[demo]` extra):

```bash
pip install -e ".[demo]"
streamlit run src/tokamak_transport_lab/app/demo.py
```

<!-- Optional: replace with a real screenshot when available -->
<!-- ![Streamlit screenshot](assets/screenshot.png) -->

The app provides:
- **Sidebar** — P_heat, ρ_dep, Te/Ti pedestal, density, τ_eq, geometry (circular/Miller), transport model (analytic/MLP/ensemble), UQ toggle, solver knobs.
- **Run button** — triggers Picard loop; results are cached by config hash.
- **Three-panel layout** (tabs):
  - 📊 **Temperature profiles** — Te(ρ) + Ti(ρ), optional 90 % UQ band (ensemble + conformal).
  - 📉 **Convergence** — residual (log) + relaxation α on dual y-axes.
  - 🔧 **χ profiles** — χ_turb + χ_neo decomposition for electron and ion channels.
- **Sweep mode** — sweep P_heat (or any parameter) across N steps (default 8), overlay Te/Ti curves, summary table with `Te(0)`, `Ti(0)`, `n_iters`, `runtime_s`.
- **Metrics row** — Te(0), Ti(0), Picard iters, converged, runtime.

### Headless GIF Generation

Generate an animated demo GIF for the README without Streamlit (requires `[gif]` extra):

```bash
pip install -e ".[gif]"
python -m scripts.make_readme_gif --config configs/scenarios/mid_power.yaml --n_frames 10
python -m scripts.make_readme_gif --param S0_e --start 0.5 --end 6 --n_frames 12
python -m scripts.make_readme_gif --out assets/demo.gif
```

Outputs `assets/demo.gif` — a looping animation of Te/Ti profiles with a text
overlay (P_heat, residual, n_iters) as the swept parameter varies. Target < 5 MB.

Run the tests:

```bash
pytest
```

Lint:

```bash
ruff check src/ tests/ scripts/
ruff format --check src/ tests/ scripts/
```

## Project Structure

```
tokamak-transport-lab/
├── configs/                 # YAML configuration files
├── src/tokamak_transport_lab/
│   ├── solver/              # Crank–Nicolson 1-D diffusion solver
│   ├── transport/           # Stiffness model, neoclassical, normalizations
│   ├── geometry/            # Circular & Miller flux-surface geometry
│   ├── surrogate/           # MLP, ensemble, UQ (weeks 3-4)
│   ├── data/                # Dataset generation (week 2)
│   ├── integration/         # Picard loop (week 5)
│   ├── app/                 # Streamlit demo + headless GIF export
│   ├── evaluation/          # Metrics, benchmarks
│   └── visualization/       # Plotting helpers
├── scripts/                 # Runnable entry points
├── tests/                   # pytest test suite
├── assets/                  # Visual assets for README
└── outputs/                 # Runtime outputs (git-ignored)
```

## Milestones

| PR | Milestone | Status |
|----|-----------|--------|
| 1  | Scaffold + CI + smoke tests | ✅ |
| 2  | CN solver + semi-analytic verification + plot | ✅ |
| 3  | Stiffness model + dataset generator (LHS) | ✅ |
| 4  | Surrogate baseline (torch MLP) + training | ✅ |
| 5  | CI sanity + formatting + LaTeX physics README | ✅ |
| 6  | Ensemble UQ + conformal + Miller geometry | ✅ |
| 7  | Picard loop + safeguards + convergence + CLI | ✅ |
| 8  | Multi-channel Te–Ti + scenario configs + runner | ✅ |
| 9  | Streamlit demo + sweep + headless GIF + README polish | ✅ |

## License

[MIT](LICENSE)
