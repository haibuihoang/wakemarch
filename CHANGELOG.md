# Changelog

## 0.3.1 — 2026-11-12

Fix found in the critical review of the accompanying paper (R1).
- The initial momentum deficit is now `eta * T_x / rho` at every yaw angle, as documented. In 0.3.0 the
  momentum target used the normal-velocity thrust coefficient `eta C_T thrust_ratio / cos^3(yaw)` capped at
  0.97, so strongly yawed or heavily loaded rotors started with less momentum (LES rotor, C_T 0.83: 0.894 instead
  of 0.95 T_x at 40 deg; C_T 0.95: 0.884 at 30 deg). Results differ from 0.3.0 only where that cap was active.
  The cap remains in the amplitude of the initial vortex sheet (V = Gamma0 / (2 R_n cos(yaw))) and in the axial
  induction of the hub vortex, where it keeps normal-velocity momentum theory real; the sheet carries 0.88-1.03
  Gamma0. Normalising the sheet to exactly Gamma0 was tested and rejected (wakes over-deflected).
- New module constant `FIXED_POINT_PASSES` (3, unchanged) of the continuity coupling, for convergence checks.
- Zenodo DOI of 0.3.0 (10.5281/zenodo.24057965) in CITATION.cff and release metadata.

## 0.3.0 — 2026-11-08

New model (manuscript of the accompanying paper, Section 2); outputs differ from 0.2.
- Eddy viscosity `cos^2(yaw) k_w sigma u_bar F(x/x0)` with `F(xi) = 1 - (1 - 2 xi^2) exp(-xi^2)`
  (it conserves the cumulative mixing of the Gaussian-wake viscosity; no tuned constant), x0 the near-wake
  length of the unyawed rotor; vortex diffusivity 0.25 of the eddy viscosity.
- Initial deficit of the unyawed developed wake projected on the stream (proportional to cos(yaw)),
  area from the streamwise momentum.
- Hub deficit (input `hub_depth`, default 0.6; radius 0.35 of the wake radius) and initial edge thickness 0.2.
- Wake rotation from the torque: new inputs `CQ` and `rotor_sense`.
- New optional input `thrust_ratio` (measured yawed/unyawed streamwise thrust); default `cos^2(yaw)`.
- Near field of the rotor (display only; replaces the near-wake ramp of 0.1/0.2): the initial deficit profile,
  hub term included, as a superposition of semi-infinite vortex cylinders (actuator-disk wake). Half the
  developed deficit at the rotor plane, the speed-up beside the near wake and the induction zone upstream of
  the rotor; no constant. New input `near_field` (default True). `sample_deficit` is therefore nonzero upstream
  of the rotor. The march, the wake position and `sample_cross_velocity` do not depend on it.
- New optional input `domain_area` (m^2): blockage of a confined domain (LES domain, closed wind tunnel), a
  uniform speed-up `-Q(x)/A` from the wake's own volume-flux deficit. Default None (open atmosphere).
- `Turbine.wake_model` and `FixedTurbine.wake_model` pass `CQ`, `rotor_sense`, `hub_depth`, `near_field` and
  `domain_area`.
- `bounds="zero"` returns exactly zero outside the stored downstream support.
- Kernel reduced to the final model; the research switches of 0.1/0.2 are removed. The march reproduces the
  authors' research solver in the same configuration on eight cases (relative difference 1.5e-9).
- Release candidates 0.3.0rc1 and 0.3.0rc2 were not published.

## 0.2.x follow-up

- Zenodo DOI of 0.2.0 (10.5281/zenodo.23920372) in CITATION.cff and release metadata.

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
