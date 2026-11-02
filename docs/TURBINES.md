# Turbine curves and presets

`Turbine` separates turbine performance data from wind conditions. The existing
`WakeModel` takes a scalar unyawed CT; it never required CP. For varying operating
conditions, define a turbine once and let its tables determine power and CT.

## Fixed coefficients for controlled studies

The direct `WakeModel(D=..., H=..., U=..., CT=..., yaw_deg=..., TI=...)`
remains the original wake-only interface and does not require CP. For a fixed
coefficient power estimate as well, select the separate fixed mode explicitly:

```python
from wakemarch import Turbine
fixed = Turbine.fixed(D=126, H=90, CT=.8, CP=.45)
point = fixed.operating_point([6., 10.], yaw_deg=20, rho=1.225)
model = fixed.wake_model([6., 10.], yaw_deg=20, TI=.1)
```

This mode follows the constant-coefficient conventions of the authors' farm
study: the wake source speed is the arithmetic disk mean, while power is
`0.5 * rho * A * CP * mean(u**3) * cos(yaw)**3`. Weights may represent unequal
sample areas. CP represents the chosen power coefficient, including any assumed
efficiency. Fixed mode has no rated cap, cut-in/out or parked-drag model. It is
an approximation, not an active-turbine validation. Curve mode instead uses
cubic rotor-equivalent speed for its wake source and its documented power mapping.
Manuscript comparisons retain their original inputs and metrics; selecting either
new power mode does not reproduce them automatically.

```python
from wakemarch import Turbine, Numerics, available_turbines

print(available_turbines())
turbine = Turbine.preset('nrel_5MW')
# Equal-area incoming rotor samples; a scalar represents uniform inflow.
point = turbine.operating_point([6.0, 10.0], yaw_deg=20)
print(point.rotor_equivalent_m_s, point.CT, point.power_W)
model = turbine.wake_model([6.0, 10.0], yaw_deg=20, TI=0.1,
                           numerics=Numerics(h_D=0.1))
# result = model.solve(6, progress=True)
```

Built-in names: `nrel_5MW`, `iea_10MW`, `iea_15MW`. They include their own D/H,
reference density, yaw-power exponent and immutable power/CT tables. Use the
[interactive notebook](../examples/wakemarch_turbines.ipynb) to inspect curves,
create a custom turbine and initialize a wake from rotor samples.

## Custom inputs and units

Supply electrical power directly when available:

```python
from wakemarch import Turbine
custom_power = Turbine.from_power_ct(
    name='illustrative', D=126, H=90,
    wind_speeds=[0, 4, 8, 12, 16, 25, 26, 50],
    power_W=[0, 200000, 1800000, 5000000, 5000000, 5000000, 0, 0],
    ct=[0, .85, .8, .4, .2, .06, 0, 0],
    rho_ref=1.225, yaw_power_exponent=1.88,
)
```

Or supply CP/CT and explicitly account for generator efficiency:

```python
custom_cp = Turbine.from_cp_ct(
    name='illustrative-cp', D=126, H=90,
    wind_speeds=[0, 4, 8, 12, 16, 25, 26, 50],
    cp=[0, .30, .46, .30, .1265625, .0331776, 0, 0],
    ct=[0, .85, .80, .40, .20, .06, 0, 0],
    generator_efficiency=.94,
)
```

These custom values are examples, not documented standard turbine curves.
Speeds are m/s; D/H are metres; density is kg/m³; `power_W` is **watts**, not kW.
CT is nominal and unyawed. All arrays must be finite, nonnegative, equal length,
with at least two strictly increasing speed nodes. Data are copied to immutable
tuples so changing input arrays cannot change the model. Source values CT ≥ 1
may be stored and plotted; `operating_point` caps them for the wake (see below).

CP conversion is performed **at table nodes**:
`P_i = efficiency * 0.5 * rho_ref * A * CP_i * U_i**3`.
For electrical-equivalent CP use efficiency=1 (the default); for aerodynamic CP,
specify generator efficiency. The canonical power and CT tables are interpolated
linearly. CP is subsequently derived from interpolated power. This retains a
rated plateau between equal-power nodes; it is not direct linear CP interpolation.
No automatic rating or operating-region fit is inferred. The first and last
tabulated speeds act as cut-in and cut-out of `operating_point` (see below);
supply zero-power nodes for any other behaviour. Do not count efficiency twice.

## Rotor averaging, yaw and wake initialization

`operating_point(rotor_velocities, weights=..., yaw_deg=..., rho=...)` computes
`U_RE = (sum(w*u**3)/sum(w))**(1/3)`. Omit weights only for equal-area samples.
Weights must broadcast to the rotor sample shape and have positive finite sum;
negative/nonfinite weights or velocities fail. Speeds are incoming streamwise
velocities without the turbine's self-wake. Zero speed is supported where the
curve table includes it. This API does not create the rotor sample geometry.

CT is evaluated at U_RE. Power uses the explicit effective-speed approximation
`P0(U_RE * (rho/rho_ref)**(1/3) * cos(yaw)**(p/3))`, with p=1.88 for these presets.
No second yaw multiplier is applied. This follows the FLORIS cosine-loss power
mapping, not a dynamic pitch/torque controller. CT is not corrected for yaw here:
the accepted WakeMarch kernel applies its existing thrust-yaw model once.
Reference tilt is embedded in the presets; tilted wake geometry is not added.

`point` reports arithmetic mean, cubic rotor-equivalent speed, power-effective
speed, unyawed CT, electrical power estimate and operating status. `power_at`,
`ct_at` and `cp_at` accept scalar or array speeds. Lookup outside tabulated support
raises by default; the lookup methods permit an explicit `bounds='zero'` option.

`operating_point` never extrapolates and never raises for out-of-table speeds.
The tabulated speed range is the operating range:

| Situation | Result |
| --- | --- |
| U_RE below the first or above the last tabulated speed | Not operating: zero power, CT 0, no wake |
| Yaw lowers the power-effective speed below the first tabulated speed | Zero power; CT and wake still from U_RE |
| Air density raises the power-effective speed above the last tabulated speed | Power at the last tabulated speed (the real wind is still below cut-out) |
| Table CT ≥ 1 at U_RE | CT 0.97 (`wakemarch.turbine.CT_CAP`) with a `RuntimeWarning`; power unchanged |

A table CT above 1 describes a heavily loaded rotor (turbulent-wake state), where
the momentum-theory initial wake of WakeMarch does not apply. The cap is the same
value the solver already uses to limit the effective CT of yawed rotors. The NREL
5 MW preset reaches CT ≥ 1 between 3.0 and about 4.0 m/s (one table node with
CT 1.13 at cut-in); the IEA presets do not.

Non-operating points return zero power. `wake_model()` rejects non-operating
points; it does not return a fictitious active wake or model parked drag.
The factory constructs a
uniform-background `WakeModel` with U=U_RE and the curve CT. It does **not**
propagate the incoming rotor field's detailed spatial structure into the new
wake. For the original power-law background option, use `WakeModel` directly.

## Sources and limitations

The bundled JSON presets are pinned from FLORIS **4.6.2**, not fetched at runtime.
They retain original YAML hashes, source URLs, reference tilt and source version.
The third-party license is bundled in `wakemarch/data/FLORIS_LICENSE.txt` and
applies to those inputs; WakeMarch itself is distributed under its own BSD
3-Clause licence. The presets contain reference data, not measurements or LES data.

| Preset | D / H [m] | Rated power | Source generator-efficiency convention |
|---|---|---|---|
| nrel_5MW | 125.88 / 90 | 5 MW | 94.4% |
| iea_10MW | 198 / 119 | 10 MW | 94% |
| iea_15MW | 242.24 / 150 | 15 MW | Table notes 100% |

Source files: [NREL 5 MW](https://github.com/NatLabRockies/floris/blob/main/floris/turbine_library/nrel_5MW.yaml),
[IEA 10 MW](https://github.com/NatLabRockies/floris/blob/main/floris/turbine_library/iea_10MW.yaml),
[IEA 15 MW](https://github.com/NatLabRockies/floris/blob/main/floris/turbine_library/iea_15MW.yaml).
These online main-branch pages can change; the bundled version and hashes define
our inputs. The original headers cite the NREL turbine-models repository and
IEA reference-turbine design data and describe flattening small above-rated power
variations. See [FLORIS operation models](https://natlabrockies.github.io/floris/operation_models_user.html)
for power-mapping context. Our hybrid retains WakeMarch's own thrust-yaw law.

No farm solver or independent farm validation is included in the package.
Nonuniform loading, shear, veer, controller dynamics and yaw losses are only
approximated by a rotor-equivalent input. Keep these limits explicit in research.
