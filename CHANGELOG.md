# Changelog

## 0.2.0 — 2026-11-03

- New `WakeResult.sample_cross_velocity(x, y, z, *, bounds="raise")`: cross-plane
  velocities `(v, w)` in m/s, the total in-plane flow (vortex-induced plus
  entrainment) that the solver uses to transport the wake. Same coordinates and
  bounds rules as `sample_deficit`; computed on first use. No numerical change to
  the model: deficits, vorticity and all existing outputs are unchanged.
- Zenodo DOI in CITATION.cff and README (concept 10.5281/zenodo.23888293; 0.1.0: 10.5281/zenodo.23888295).

## 0.1.0 — 2026-11-02

First public release.

- Single-turbine steady wake model with curled wake shape: the recommended
  configuration of the authors' research solver (thrust exponent 2, vortex
  diffusivity factor 0.5, momentum fraction 0.95, near-wake display ramp).
- `WakeModel`, `Numerics` and `WakeResult`: immutable inputs, explicit deficit
  and velocity sampling, read-only histories, live progress display.
- `Turbine`: power/CT and CP/CT tables, three reference presets pinned from
  FLORIS 4.6.2 (NREL 5 MW, IEA 10 MW, IEA 15 MW), rotor-equivalent operating
  point. The tabulated speed range is the operating range; table CT ≥ 1 is capped
  at 0.97 for the wake with a warning. `Turbine.fixed` for constant coefficients.
- Input limits: H ≥ D/2 and |yaw| ≤ 50° (validated to 40°).
- Runtime dependencies NumPy ≥ 1.24 and SciPy ≥ 1.10; Python ≥ 3.11.
- Four example notebooks (API, resolution/cost, turbines, optional public PIV
  comparison).
