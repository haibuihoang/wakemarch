# WakeMarch user guide

WakeMarch predicts the steady wake of **one yawed turbine**. Start with the
[API notebook](../examples/wakemarch_api_test.ipynb) for editable inputs, yaw
comparisons and wake plots; use the [performance notebook](../examples/wakemarch_performance.ipynb)
to examine grid sensitivity. Neither requires measurement data.
The optional [PIV notebook](../examples/wakemarch_piv_validation.ipynb) compares
public measurements with inherited calibration; download its dataset separately.

## Installation and first run

In a fresh virtual environment:

```sh
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install "wakemarch[notebooks]"   # or ".[notebooks]" from a source checkout
python -m jupyterlab examples/wakemarch_api_test.ipynb
```

JupyterLab started from this environment uses its Python 3 kernel; run all cells
from a fresh kernel. If you use a shared Jupyter installation instead, register
the environment with `python -m ipykernel install --user --name wakemarch` and
select that kernel. Install `wakemarch` without `[notebooks]` for core-only use.

The examples below are executed by the documentation check on the installed package.

```python
import numpy as np
from wakemarch import Numerics, WakeModel

model = WakeModel(D=126, H=90, U=8, CT=0.8, yaw_deg=25, TI=0.06,
                  numerics=Numerics(h_D=1/20))
result = model.solve(max_dist_D=6, progress=False)
print(result.sample_streamwise_velocity(5*model.D, 0, model.H))
```

## Physical inputs

All inputs are explicit scalars; physical settings are immutable. Create another
`WakeModel` to change them. Accepted input bounds prevent some computational
errors; they are **not** a validated operating envelope.

| Argument | Meaning | Units / computational bounds |
|---|---|---|
| `D` | Rotor diameter | m, >0 |
| `H` | Hub height above ground | m, ≥ D/2 |
| `U` | Undisturbed speed at hub height | m/s, >0 |
| `CT` | Unyawed thrust coefficient | 0 < CT < 1 |
| `yaw_deg` | Yaw angle | degrees, \|yaw\| ≤ 50 (validated to 40) |
| `TI` | Ambient turbulence intensity | fraction, 0–1; 6% is **0.06** |
| `shear_alpha` | Power-law exponent | ≥0; default 0 |

The background speed is `U * (max(z, 0.05*H)/H)**shear_alpha`. The height floor
matches the accepted implementation; it is not a surface-layer model. The API
only requires H ≥ D/2 (rotor above the ground); choose a physically meaningful
rotor clearance yourself.

## Coordinates and reconstruction

Global `x` is downstream from the turbine, `y` is lateral from its centre, and
`z` is height above ground. Positive yaw deflects the wake toward **negative y**.
Rotate/mirror reference data explicitly if its convention differs.

The solver marches in `x_s = x + y*tan(yaw)` with yaw in radians in that formula.
Thus a physical plane at fixed x samples different marching positions across y.
Use the sampling API for physical-space plots, rather than selecting a raw
history slice. Internally the vertical coordinate is relative to the hub.

```python
y = np.linspace(-model.D, model.D, 81)
z = np.linspace(0, 2*model.H, 51)
yy, zz = np.meshgrid(y, z)
cross_plane = result.sample_deficit(5*model.D, yy, zz)
assert cross_plane.shape == (51, 81)
velocity = result.sample_streamwise_velocity(5*model.D, yy, zz)
assert np.isfinite(velocity).all()
```

Both methods accept scalars, lists and broadcastable arrays. Deficit means
background minus streamwise velocity, in m/s. It is not normalized by inflow.
Sampling includes the near field of the rotor (the actuator-disk vortex cylinder:
half the deficit at the rotor plane, a slight speed-up beside the near wake, the
induction zone upstream); raw histories do not.

Upstream of the tilted rotor (`x_s < 0`) the deficit is the induction zone (zero
with `near_field=False`). Downstream samples
outside the stored x/y/z support raise `ValueError`. Explicit `bounds="zero"`
extends deficit by zero and streamwise velocity by the background value. This
extension is not a prediction that a truncated wake has recovered. Negative
heights and nonfinite coordinates always raise errors.

```python
assert result.sample_deficit(-model.D, 0, model.H) > 0      # induction zone
try:
    result.sample_deficit(100*model.D, 0, model.H)
except ValueError:
    pass
else:
    raise AssertionError("Expected an out-of-support error")
assert result.sample_deficit(100*model.D, 0, model.H, bounds="zero") == 0
```

### Cross-plane velocities

`sample_cross_velocity` returns the lateral (`v`, along +y) and vertical (`w`,
along +z) velocities in m/s, with the same coordinates and `bounds` rules as
`sample_deficit`. They are the total in-plane flow that transports the wake in the
solver: the flow induced by the streamwise vorticity (the counter-rotating vortex
pair of a yawed rotor) plus the entrainment flow from the streamwise velocity
change, with `w = 0` at the ground. The near-field display is not applied.
The fields are computed on the first call (one Poisson solve per stored plane).

```python
v, w = result.sample_cross_velocity(5*model.D, yy, zz)
assert v.shape == w.shape == (51, 81)
assert result.sample_cross_velocity(-model.D, 0, model.H) == (0.0, 0.0)
```

## Switching model parts on and off

For comparisons with other rotors and for sensitivity tests, four parts are inputs of `WakeModel`
(and of `Turbine.wake_model`):

| input | default | effect |
| --- | --- | --- |
| `hub_depth` | 0.6 | depth of the hub hole of the initial wake; 0 removes it (e.g. a lab rotor with a nacelle) |
| `near_field` | True | near field of the rotor (display only); False shows the developed wake from the rotor plane |
| `domain_area` | None | cross-section in m^2 of a confined domain (LES, closed wind tunnel): uniform blockage speed-up from the wake's own flux deficit |
| `CQ` | None | rotor torque coefficient for the wake rotation; None omits it |

`near_field` and `domain_area` change only what is sampled, not the march: the wake position and the
cross-plane velocities are the same.

```python
open_flow = WakeModel(D=126, H=90, U=8, CT=0.8, yaw_deg=25, TI=0.06)
tunnel = WakeModel(D=126, H=90, U=8, CT=0.8, yaw_deg=25, TI=0.06, hub_depth=0.0, domain_area=30 * 126**2)
```

## Numerical settings and results

| `Numerics` field | Default | Meaning |
|---|---:|---|
| `h_D` | 1/30 | Cross-plane grid spacing / D |
| `Ly_D` | 8 | Full lateral width / D, centred on turbine |
| `ztop_D` | 3.5 | Top **above hub height** / D; bottom at ground |
| `dx_D` | 0.1 | Marching step / D |
| `store_every_D` | 0.25 | Requested minimum stored-plane interval / D |

Choose `h_D` so that `Ly_D/h_D` and `ztop_D/h_D` are integers; otherwise the far
boundary is rounded down to the last whole grid cell (e.g. h_D=0.07 gives a
lateral range of −4D to 3.98D).

Storage happens on marching steps: the default request gives actual 0.3D
intervals. Inspect `result.x`; do not assume requested spacing. `max_dist_D`
must be positive and at least `dx_D` and `store_every_D`. The march adds a
yaw-dependent reconstruction buffer. Its presence does not guarantee support
for every lateral position on every requested physical plane.

| Result property | Meaning / units |
|---|---|
| `x` | Stored marching positions x_s, m |
| `y` | Lateral mesh, m |
| `z`, `z_relative` | Absolute and hub-relative heights, m |
| `marching_deficit` | Marched (developed-wake) deficit without the near-field display, m/s |
| `streamwise_vorticity` | Streamwise vorticity, 1/s |
| `streamwise_velocity_derivative` | Stored u_x field, 1/s; initial plane zero |

Fields have shape `(len(x), len(z_relative), len(y))`. Array properties return
read-only views to prevent accidental edits; use `.copy()` for analysis that
modifies arrays. This is not a serialization or security guarantee. Every solve
constructs a new independent result. The `_kernel` and result `_solver` are private.

```python
assert result.marching_deficit.shape == (len(result.x), len(result.z), len(result.y))
assert not result.marching_deficit.flags.writeable
np.testing.assert_allclose(np.diff(result.x)/model.D, 0.3)
```

## Progress and runtime

`solve(progress=True)` (default) shows HTML progress in Jupyter and a text bar in
terminals. `False` disables it; a callable receives `(completed_D, total_D)`.
Total includes reconstruction buffering. Progress measures distance, not remaining
time. Errors and interrupts close the display; exceptions from callbacks propagate.

Use progress off and repeated, serial runs for timing. Imports and construction
should be reported separately or explicitly included. There is no JIT compiler.
On one macOS arm64 machine, the difficult 40° yaw/shear case took 5.3 s (median)
and the standard D/20 case about 4 s. These are local measurements, not
performance promises. Cached geometry costs memory.

D/20 is a useful exploratory choice in the supplied examples; it does not replace
the D/30 default or establish adequate accuracy for every application. Assess
grid spacing, domain size and marching step for your outputs of interest.
The performance notebook compares to D/30, which is a numerical reference, not
truth. A rotor-average velocity or u³ proxy does not model a downstream active
turbine or validate a farm power prediction.

## Reproducibility and troubleshooting

Record package/Python/NumPy/SciPy versions, all physical/numerical inputs, actual
coordinates, sampling definitions and data preprocessing. The supplied figure
example writes a JSON record beside its image. Keep calibration and validation
cases distinguishable; see [scientific scope](SCIENCE.md).

- Wrong package/version in a notebook: select the registered kernel, restart,
  and inspect `wakemarch.__file__` and `wakemarch.__version__`.
- Blank/truncated downstream samples: inspect stored support and yaw mapping;
  avoid using `bounds="zero"` to conceal insufficient domain/extent.
- A percentage entered as TI: convert to a fraction.
- Long runs or large memory use: compare a coarser grid for exploration and
  quantify its differences; do not silently change resolution in benchmarks.
- A progress bar not visible: run the cell in Jupyter, or use a callback/text
  terminal. No widget extension is required.

See [release verification](RELEASING.md) for commands that test built artifacts,
documentation, notebooks and dependency environments.

For the optional measurement comparison install `.[notebooks,validation]`, set
`WAKEMARCH_PIV_DATA` to the downloaded `Bastankhah_jfm_2016.h5` (or edit `DATA`
in the notebook), and open `examples/wakemarch_piv_validation.ipynb`. Read its
calibration and coordinate notes before interpreting its error metrics.

## Turbine inputs

Use the [turbine guide](TURBINES.md) for reusable CP/CT or power/CT inputs,
standard presets and rotor-weighted operating points. The scalar-CT interface
remains supported.
