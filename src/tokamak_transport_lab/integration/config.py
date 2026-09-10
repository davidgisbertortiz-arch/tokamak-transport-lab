"""Shared configuration and model dispatch for the app, CLI and benchmarks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tokamak_transport_lab.integration.multichannel_picard import run_picard_multichannel
from tokamak_transport_lab.transport.ion_transport import make_ion_transport
from tokamak_transport_lab.transport.stiffness import chi_total


def validate_config(cfg):
    """Reject misspelled sections/keys before running expensive experiments."""
    sections = {
        "grid": {"n_rho", "rho_min", "rho_max"},
        "source": {"S0", "S0_e", "S0_i", "rho_dep", "sigma"},
        "bc": {"Te_ped", "Ti_ped"},
        "coupling": {"density", "tau_eq"},
        "picard": {"max_iters", "tol", "alpha0", "alpha_min", "alpha_max", "divergence_patience"},
        "solver": {"dt", "theta", "sub_steps", "n_steps"},
        "model": {"type", "path"},
        "miller": {"kappa", "delta"},
    }
    transport_keys = {"chi_s", "a_over_LTe_crit", "alpha_s", "chi_neo", "model", "chi0"}
    sections.update({name: transport_keys for name in ("transport", "transport_e", "transport_i")})
    allowed = set(sections) | {"geometry", "ti_transport_mode", "qi_factor", "output_dir", "seed"}
    if not isinstance(cfg, dict) or set(cfg) - allowed:
        raise ValueError("Unknown configuration section or non-mapping configuration")
    for name, keys in sections.items():
        value = cfg.get(name, {})
        if not isinstance(value, dict) or set(value) - keys:
            raise ValueError(f"Unknown keys or invalid mapping in {name}")
    if cfg.get("grid", {}).get("rho_min", 0) != 0 or cfg.get("grid", {}).get("rho_max", 1) != 1:
        raise ValueError("The radial grid must cover [0, 1]")
    if not np.isfinite(cfg.get("qi_factor", 1.0)) or cfg.get("qi_factor", 1.0) <= 0:
        raise ValueError("qi_factor must be finite and strictly positive")


def transport_params(cfg, section="transport_e"):
    values = cfg.get(section, cfg.get("transport", {}))
    return {
        k: values.get(k, v)
        for k, v in {"chi_s": 1.0, "a_over_LTe_crit": 3.0, "alpha_s": 1.5, "chi_neo": 0.01}.items()
    }


def geometry_function(cfg):
    geometry = cfg.get("geometry", "circular")
    if geometry == "circular":
        return None
    if geometry == "miller":
        from functools import partial

        from tokamak_transport_lab.geometry.miller import vprime_miller

        return partial(vprime_miller, **cfg.get("miller", {}))
    raise ValueError(f"Unknown geometry: {geometry}")


def build_transport(cfg, *, bundle=None, member=None):
    mode = cfg.get("model", {}).get("type", cfg.get("transport", {}).get("model", "analytic"))
    if mode in ("analytic", "stiffness"):
        return chi_total, None, "analytic"
    if mode not in ("mlp", "ensemble"):
        raise ValueError(f"Unknown transport model: {mode}")
    from tokamak_transport_lab.surrogate.bundle import TransportBundle
    from tokamak_transport_lab.transport.bundle_adapter import BundleAdapter

    if bundle is None:
        default = "outputs/surrogate/model.pt" if mode == "mlp" else "outputs/ensemble/model.pt"
        bundle = TransportBundle.load(Path(cfg.get("model", {}).get("path", default)))
    if mode == "ensemble" and len(bundle.models) < 2:
        raise ValueError("Ensemble mode requires at least two trained members")
    if mode == "mlp" and len(bundle.models) != 1:
        raise ValueError("MLP mode requires a single-member artifact")
    return BundleAdapter(bundle, member=member), bundle, mode


def run_config(cfg, *, bundle=None, member=None):
    """Run a coupled stationary calculation with explicit model provenance."""
    validate_config(cfg)
    e_model, bundle, mode = build_transport(cfg, bundle=bundle, member=member)
    params_e, params_i = transport_params(cfg), transport_params(cfg, "transport_i")
    ion_mode = cfg.get("ti_transport_mode", "stiffness")
    if mode != "analytic" and ion_mode == "stiffness":
        from tokamak_transport_lab.transport.bundle_adapter import BundleAdapter

        i_model = BundleAdapter(bundle, member=member)
    else:
        i_model, params_i = make_ion_transport(
            ion_mode,
            electron_model=e_model,
            electron_params=params_e,
            qi_factor=cfg.get("qi_factor", 1.0),
            ion_params=params_i,
        )
    source = cfg.get("source", {})
    bc = cfg.get("bc", {})
    controls = {
        k: v
        for k, v in cfg.get("picard", {}).items()
        if k in ("max_iters", "tol", "alpha0", "alpha_min", "alpha_max", "divergence_patience")
    }
    result = run_picard_multichannel(
        n_rho=cfg.get("grid", {}).get("n_rho", 60),
        te_ped=bc.get("Te_ped", 500.0),
        ti_ped=bc.get("Ti_ped", 400.0),
        s0_e=source.get("S0_e", source.get("S0", 1.0)),
        s0_i=source.get("S0_i", 0.0),
        rho_dep=source.get("rho_dep", 0.3),
        sigma=source.get("sigma", 0.1),
        **cfg.get("coupling", {}),
        v_prime_fn=geometry_function(cfg),
        transport_model_e=e_model,
        transport_params_e=params_e,
        transport_model_i=i_model,
        transport_params_i=params_i,
        **controls,
    )
    ood = max(
        getattr(e_model, "out_of_domain_fraction", 0.0),
        getattr(i_model, "out_of_domain_fraction", 0.0),
    )
    result.metadata.update(
        tol=cfg.get("picard", {}).get("tol", 1e-4),
        transport_model=mode,
        geometry=cfg.get("geometry", "circular"),
        artifact_sha256=bundle.fingerprint if bundle else None,
        out_of_domain_fraction=ood,
    )
    return result


def ensemble_profiles(cfg, *, bundle=None):
    """Solve each closure member; return an uncalibrated profile range.

    Members are propagated through the entire nonlinear solve. A member that
    does not converge invalidates the band rather than being silently removed.
    """
    _, bundle, mode = build_transport(cfg, bundle=bundle)
    if mode != "ensemble":
        raise ValueError("Profile ensemble requires ensemble mode")
    results = [run_config(cfg, bundle=bundle, member=i) for i in range(len(bundle.models))]
    if not all(r.metadata["converged"] for r in results):
        raise RuntimeError("A member profile did not converge; no uncertainty band is reported")
    te = np.array([r.te_final for r in results])
    ti = np.array([r.ti_final for r in results])
    return {
        "te_lower": te.min(0),
        "te_upper": te.max(0),
        "ti_lower": ti.min(0),
        "ti_upper": ti.max(0),
        "te_members": te,
        "ti_members": ti,
        "label": "Ensemble member range (uncalibrated)",
    }
