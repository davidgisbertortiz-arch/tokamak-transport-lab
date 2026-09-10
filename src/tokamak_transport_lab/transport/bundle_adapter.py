"""Connect complete, normalized ML artifacts to the stationary solver."""

from __future__ import annotations

import numpy as np

from tokamak_transport_lab.data.generate import FEATURE_NAMES


class BundleAdapter:
    """Return learned chi directly; feature order comes from the dataset schema."""

    def __init__(self, bundle, member=None):
        self.bundle, self.member = bundle, member
        self.last_inputs = None

    def __call__(self, a_over_lte, **params):
        defaults = {"chi_s": 1.0, "a_over_LTe_crit": 3.0, "alpha_s": 1.5, "chi_neo": 0.01}
        values = {**defaults, **params}
        a = np.asarray(a_over_lte)
        self.last_inputs = np.column_stack(
            [a] + [np.broadcast_to(values[k], a.shape) for k in FEATURE_NAMES[1:]]
        )
        return self.bundle.predict_chi(self.last_inputs, member=self.member)

    @property
    def out_of_domain_fraction(self):
        return (
            0.0
            if self.last_inputs is None
            else float(self.bundle.out_of_domain(self.last_inputs).mean())
        )
