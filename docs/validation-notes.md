# Repository diagnosis and corrections

The audit baseline was commit `fca9b6976e5b76de9ca8a082ad145b5b80a16a5b`.
The original implementation had useful structure, a real diffusion solver,
synthetic transport models and tests, but several advertised integrations were
incomplete or numerically misleading.

| Original issue | Correction and evidence |
|---|---|
| Small mixed transient updates declared stationary convergence | Elliptic stationary iteration; source-scaled PDE defect; initial-condition regression |
| Electron-ion exchange frozen over long transient blocks | Exchange solved in the coupled operator for both stationary and transient paths |
| Temperature caps concealed instability | Positive trial acceptance, explicit nonconvergence, no clipping |
| Default transient stopped too early | Sufficient default duration and separate stationary residual; verification rejects incomplete equilibration |
| Incorrect reference integrand limit at the axis | `I(rho)/rho` tends to zero; independent refinement uses the corrected quadrature |
| Test scenarios missed the actual failing YAML settings | Original settings retained in `tests/fixtures/legacy_scenarios`; both old and current files are executed |
| Centered targets with nonnegative network output | Linear output for standardized log diffusivity |
| Inputs and outputs lacked transforms at inference | Versioned bundle contains normalization, architecture and feature schema |
| Three sampled inputs did not affect labels | Five inputs now all parameterize the closure |
| Converting predicted flux to chi was unstable near zero gradient | Direct diffusivity target and continuous, prescribed subcritical floor |
| App selector did not change the solver model | App and CLI use one tested dispatcher with artifact provenance |
| Local flux uncertainty arbitrarily mapped to temperature | Independent local and full-profile conformal experiments; complete member propagation |
| Calibration used the test set and an interpolated quantile | Independent IID calibration/test, exact finite-sample order statistic |
| Constant volume rescaling advertised as Miller physics | Removed shaping controls and claims; legacy behavior explicitly documented |
| Device-like labels and unused machine parameters implied physical scaling | Explicit normalization, synthetic scenario names, no MW or device interpretation |
| GIF overrides and plotted transport could differ from the supplied config | Full configuration dispatch, actual chi plots and per-frame JSON provenance |

The current heating scenarios also use source amplitudes and exchange times
that exercise subcritical and turbulent behavior. They are intentionally distinct
from the original settings, which remain as regression fixtures. Comparisons
between old and new numerical implementations must use those fixtures, not
compare changed scenarios as if the parameters were identical.

The generated validation figure and JSON are in `docs/validation/`. Scripts
record source/model hashes and all relevant calibration parameters. Timings are
hardware-dependent. The recorded Git base is the starting revision; the source
hash identifies the implementation used before the final local commit.

This work establishes numerical consistency and a working synthetic ML
integration. It does not establish fidelity to tokamak experiments. Further
scientific work could compare other discretizations, implement metric-consistent
shaped geometry, use an external transport database, or validate a dimensional
model against measurements. None of those is claimed as already completed.

## Final local checks

The completed workspace was checked on Python 3.11.12: **202 tests passed**,
Ruff lint and formatting passed, and `pip check` found no broken requirements.
The test run emits third-party Matplotlib/Pyparsing deprecation warnings.
The recorded JUnit report is available locally at `outputs/test-results.xml`.

Streamlit's analytic, trained MLP, trained ensemble, calibrated interval and
four-point sweep flows were exercised. The local server returned `ok` from
its health endpoint. The GIF was regenerated from eight converged solves and
its rendered frame and the validation figure were visually inspected.

The GitHub workflow is configured for Python 3.11 and 3.12; its remote run
requires publishing the branch. Python 3.12 has not been executed locally.
