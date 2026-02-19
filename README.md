# tokamak-transport-lab

[![CI](https://github.com/davidgisbertortiz-arch/tokamak-transport-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/davidgisbertortiz-arch/tokamak-transport-lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

> Interactive integrated modelling playground: 1-D diffusion solver, ML transport surrogate, calibrated uncertainty — all on synthetic data, runnable on a laptop.

## Overview

**tokamak-transport-lab** couples a custom 1-D Crank–Nicolson diffusion solver with a deep-ensemble ML surrogate of turbulent transport in a self-consistent Picard iteration loop, producing predictive plasma temperature profiles with calibrated uncertainty.

```
Config (YAML)  ──▶  Solver (CN-FDM)  ──▶  Profiles + Plots + NPZ
                        ▲
                 Surrogate / Stiffness model
```

## Installation

```bash
git clone https://github.com/davidgisbertortiz-arch/tokamak-transport-lab.git
cd tokamak-transport-lab
pip install -e ".[dev]"
```

## Quick Start

Run the 1-D solver with the default config:

```bash
python -m scripts.run_solver --config configs/solver.yaml
```

This produces `outputs/solver_verification.npz` with the converged temperature
profile, diffusivity, source, and residual history.

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
| 2  | CN solver + semi-analytic verification + plot | 🔜 |
| 3  | Stiffness model + dataset generator (LHS) | 🔜 |
| 4  | Surrogate baseline (torch MLP) + training | 🔜 |
| 5  | Picard loop + safeguards + convergence | 🔜 |
| 6+ | UQ, visuals, Streamlit | 🔜 |

## License

[MIT](LICENSE)
