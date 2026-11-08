# Provenance

WakeMarch 0.3 packages the final single-turbine model of the accompanying paper.
It was developed with the authors' research solver `src/curl_pns.py` (with
`src/wake_inputs.py`) in the paper companion repository; the research solver
keeps the experimental switches of the development, the package only the final
model. The solver is project research code, not the official implementation of
any comparison model.

## Kernel

`src/wakemarch/_kernel.py` contains the final model only:

- the march of the developed wake (deficit and streamwise vorticity, entrainment
  from continuity, open far field and impermeable ground);
- the rotor force, the projected initial deficit with hub term and smoothed edge,
  the force-based circulation and the hub vortex from the torque;
- the eddy viscosity `cos^2(yaw) k_w sigma u_bar F(x/x0)` with `A = 2`;
- the near field of the rotor (display only), as a superposition of semi-infinite
  vortex cylinders of the initial profile, and the optional blockage offset of a
  confined domain;
- caching of fixed advection and multipole geometry, an optional progress
  observer, and reuse of the variable-coefficient diffusion factorization.

Equality of the march is checked numerically: in the same configuration the
package reproduces the research solver on eight cases (zero, positive, negative
and 40° yaw; shear; low and high turbulence; measured thrust ratio; no hub term;
coarse and fine grids) with a maximum relative difference of 1.5e-9 in the deficit,
the vorticity and the sampled field (research solver SHA-256
`ec4739276b0bf7f41035f6dee7906c56cbd668779f1e697f1fe6a57653be87f2`, 2026-11-08).
The near-field display and the blockage offset are package code; they do not
change the march. The results of the paper were computed with this kernel
(release candidate 0.3.0rc2, numerically identical to 0.3.0).

## Added code

- The public API (`WakeModel`, `Numerics`, `WakeResult`), input checks,
  explicitly named sampling methods and tests.
- `Turbine` (power/CT and CP/CT tables, presets, rotor-equivalent operating
  point) and `Turbine.fixed` (constant coefficients). Three reference tables are
  pinned from FLORIS 4.6.2 with their source hashes and licence
  (`src/wakemarch/data/`).

Model options tested during the development and not adopted are not part of the
package. Packaging does not establish new scientific claims; see
[scientific scope](docs/SCIENCE.md).
