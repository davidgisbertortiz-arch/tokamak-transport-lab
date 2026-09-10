"""Simultaneous profile intervals calibrated on an explicit synthetic scenario law."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

# Coverage is marginal over this random experiment, not arbitrary device cases.
FAMILY = {
    "source_e": [300.0, 1000.0],
    "te_ped": [300.0, 700.0],
    "ti_ped_ratio": 0.8,
    "n_rho": 40,
    "rho_dep": 0.3,
    "sigma": 0.1,
    "density": 1.0,
    "tau_eq": 1.0,
    "transport_e": {"chi_s": 1.0, "a_over_LTe_crit": 3.0, "alpha_s": 1.5, "chi_neo": 0.01},
    "transport_i": {"chi_s": 0.5, "a_over_LTe_crit": 3.5, "alpha_s": 1.5, "chi_neo": 0.01},
}


def family_config(source=650.0, te_ped=500.0):
    return {
        "grid": {"n_rho": FAMILY["n_rho"]},
        "source": {
            "S0_e": float(source),
            "S0_i": 0.0,
            "rho_dep": FAMILY["rho_dep"],
            "sigma": FAMILY["sigma"],
        },
        "bc": {"Te_ped": float(te_ped), "Ti_ped": FAMILY["ti_ped_ratio"] * float(te_ped)},
        "coupling": {"density": FAMILY["density"], "tau_eq": FAMILY["tau_eq"]},
        "transport_e": deepcopy(FAMILY["transport_e"]),
        "transport_i": deepcopy(FAMILY["transport_i"]),
        "geometry": "circular",
        "ti_transport_mode": "stiffness",
        "picard": {"max_iters": 100, "tol": 1e-5, "alpha0": 0.5},
    }


def sample_configs(n, seed):
    rng = np.random.default_rng(seed)
    values = rng.uniform(
        [FAMILY["source_e"][0], FAMILY["te_ped"][0]],
        [FAMILY["source_e"][1], FAMILY["te_ped"][1]],
        size=(n, 2),
    )
    return [family_config(*row) for row in values]


def profile_interval(result, q_hat):
    """One common score controls both channels and every node simultaneously."""
    if q_hat < 0 or not result.metadata["converged"]:
        raise ValueError("Intervals require nonnegative radius and converged prediction")
    bands = {}
    for channel, t in [("te", result.te_final), ("ti", result.ti_final)]:
        lo, hi = np.maximum(t - q_hat, 0), t + q_hat
        lo[-1] = hi[-1] = t[-1]  # imposed Dirichlet values are known exactly
        bands[channel + "_lower"], bands[channel + "_upper"] = lo, hi
    return bands
