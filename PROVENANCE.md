# Provenance

WakeMarch packages the recommended configuration
(`YawedWakeSolver.recommended(...)`) of the authors' research solver
`src/curl_pns.py`, SHA-256
`f3dce8e95c77942caf59c8a0745d9ff6d2f9ed4ab642062cec46e752856a33fc`
(research repository commit `3027932`). The solver is project research code,
not the official implementation of any comparison model.

## Kernel

`src/wakemarch/_kernel.py` is a copy of that solver with:

- explanatory comments and an optional progress observer;
- caching of fixed advection and multipole geometry;
- reuse of the exact variable-coefficient diffusion factorization as the next
  preconditioner after a stalled refinement.

The discrete operators, residual threshold (1e-9), precision, grid, physics and
fitted constants are unchanged. Iteration paths can differ, so equality is
checked numerically, not textually: the installed package reproduces eight
frozen reference cases of the research solver (zero, positive, negative and
40° yaw; shear; low/high turbulence; coarse and fine grids) at rtol=1e-7,
atol=1e-8. This holds from NumPy 1.24.4/SciPy 1.10.1 to NumPy 2.4/SciPy 1.17.
The reference arrays are kept with the authors' development records and are not
shipped.

## Added code

- The public API (`WakeModel`, `Numerics`, `WakeResult`), input checks,
  explicitly named sampling methods and tests. No change to model equations or
  fitted constants.
- `Turbine` (power/CT and CP/CT tables, presets, rotor-equivalent operating
  point) and `Turbine.fixed` (constant coefficients). Three reference tables are
  pinned from FLORIS 4.6.2 with their source hashes and licence
  (`src/wakemarch/data/`).

An independently developed set of additional wake corrections was evaluated and
not adopted; it is not part of the package. Packaging does not establish new
scientific claims; see [scientific scope](docs/SCIENCE.md).
