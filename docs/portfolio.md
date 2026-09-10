# Portfolio positioning

## Suggested CV entry

**Tokamak Transport Lab — personal project / preparatory work**

Developing an educational Python model of radial electron–ion heat transport,
combining conservative numerical methods, synthetic neural transport surrogates
and uncertainty quantification. Verified solver convergence and energy balance,
built reproducible training and calibration workflows, and integrated an
interactive Streamlit demonstration.

Optional evidence-based bullet:

- Implemented and tested coupled stationary and Crank–Nicolson solvers, neural
  inference with versioned preprocessing, and simultaneous temperature-profile
  intervals evaluated on independent synthetic scenarios.

Use this wording only to the extent that you can explain the implementation
and the experiments. Present it as personal learning and development; do not
present it as employment, published research, experimental validation or
experience running a production fusion transport code.

## Concepts to be able to explain in an interview

1. Why a small time-step update does not demonstrate a stationary solution.
2. How shared face fluxes and implicit exchange support conservation.
3. Why an accurate surrogate on local samples can still fail inside a solver.
4. What the five ML inputs mean, what is synthetic, and which structural
   assumptions are imposed in the inference rule.
5. Why separate training, validation, calibration and test sets are needed.
6. Why local diffusivity intervals and temperature-profile intervals are
   different statistical objects.
7. What simultaneous, marginal coverage means and why 58/64 test successes
   do not establish a 90% guarantee for any selected real plasma.
8. Why the constant legacy “Miller” factor cannot model geometric shaping.

The useful evidence is the executable reasoning behind these points. The
headline R² reflects emulation of a known synthetic closure, not discovery of
plasma physics or validation against a fusion device.
