"""Immutable turbine tables and an explicit rotor-equivalent operating approximation."""
from dataclasses import dataclass
from importlib.resources import files
import json
import warnings
import numpy as np
from .model import _finite

_PRESETS = ('nrel_5MW', 'iea_10MW', 'iea_15MW')
# Thrust coefficient used for the wake when a table gives CT >= 1 (heavily loaded
# rotor near cut-in). Same value as the solver's existing cap for yawed rotors.
CT_CAP = 0.97


def available_turbines():
    """Names of bundled, version-pinned reference turbine presets."""
    return _PRESETS


def _vector(name, values):
    try:
        a = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite numeric vector') from exc
    if a.ndim != 1 or a.size < 2 or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError(f'{name} must be a finite nonnegative vector of length >=2')
    return tuple(float(x) for x in a)


@dataclass(frozen=True)
class OperatingPoint:
    """Power estimate and nominal CT; CT has NOT received a yaw correction."""
    rotor_mean_m_s: float
    rotor_equivalent_m_s: float
    power_effective_m_s: float
    CT: float
    power_W: float
    operating: bool


@dataclass(frozen=True, kw_only=True)
class Turbine:
    """Geometry and immutable unyawed power/CT tables in SI units.

    Prefer preset(), from_power_ct(), or from_cp_ct(). Power is electrical W.
    Tabulated power and CT use linear interpolation. No automatic extrapolation.
    The tabulated speed range is the operating range: operating_point treats
    speeds below the first or above the last tabulated speed as not operating.
    Table CT >= 1 is capped at CT_CAP (0.97) for the wake, with a RuntimeWarning.
    yaw_power_exponent controls an effective-speed approximation, not a controller.
    """
    name: str
    D: float
    H: float
    wind_speeds: tuple
    power_values_W: tuple
    ct_values: tuple
    rho_ref: float = 1.225
    yaw_power_exponent: float = 1.88
    provenance: str = 'User-supplied turbine tables'

    @classmethod
    def fixed(cls, *, D, H, CT, CP):
        """Fixed coefficients using the historical farm prototype conventions."""
        return FixedTurbine(D=D, H=H, CT=CT, CP=CP)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('name must be a nonempty string')
        if not isinstance(self.provenance, str):
            raise TypeError('provenance must be a string')
        for key in ('D', 'H', 'rho_ref', 'yaw_power_exponent'):
            object.__setattr__(self, key, _finite(key, getattr(self, key)))
        if self.D <= 0 or self.H < self.D/2 or self.rho_ref <= 0 or self.yaw_power_exponent < 0:
            raise ValueError('Require D>0, H>=D/2, rho_ref>0, yaw_power_exponent>=0')
        for key in ('wind_speeds', 'power_values_W', 'ct_values'):
            object.__setattr__(self, key, _vector(key, getattr(self, key)))
        if len({len(self.wind_speeds), len(self.power_values_W), len(self.ct_values)}) != 1:
            raise ValueError('Speed, power and CT tables must have equal lengths')
        if np.any(np.diff(self.wind_speeds) <= 0):
            raise ValueError('Wind speeds must be strictly increasing')
        if self.wind_speeds[0] == 0 and self.power_values_W[0] != 0:
            raise ValueError('Power at zero speed must be zero')

    @classmethod
    def preset(cls, name):
        """Load an exact FLORIS 4.6.2 reference table, with its original geometry."""
        if name not in _PRESETS:
            raise ValueError(f'Unknown preset {name!r}; choose from {_PRESETS}')
        d = json.loads(files('wakemarch').joinpath('data', name+'.json').read_text())
        return cls.from_power_ct(name=d['name'], D=d['D'], H=d['H'],
            wind_speeds=d['wind_speed_m_s'], power_W=d['power_W'], ct=d['CT'],
            rho_ref=d['rho_ref'], yaw_power_exponent=d['yaw_exponent'],
            provenance=f"FLORIS {d['floris_version']}; {d['source_url']}; source YAML SHA256={d['source_yaml_sha256']}")

    @classmethod
    def from_power_ct(cls, *, wind_speeds, power_W, ct, **settings):
        """Input electrical power in W, nominal unyawed CT, and geometry/settings."""
        return cls(wind_speeds=wind_speeds, power_values_W=power_W, ct_values=ct, **settings)

    @classmethod
    def from_cp_ct(cls, *, wind_speeds, cp, ct, D, rho_ref=1.225,
                   generator_efficiency=1.0, **settings):
        """Convert Cp nodes to electrical-power nodes before interpolation.

        P_i=efficiency*0.5*rho_ref*A*Cp_i*U_i**3. Use efficiency=1 for electrical
        Cp, or supply an explicit efficiency for aerodynamic Cp. Power rather
        than Cp is interpolated, so equal-power input nodes preserve a plateau.
        """
        u = np.asarray(_vector('wind_speeds', wind_speeds))
        c = np.asarray(_vector('cp', cp))
        D, rho_ref = _finite('D', D), _finite('rho_ref', rho_ref)
        efficiency = _finite('generator_efficiency', generator_efficiency)
        if c.shape != u.shape or np.any(c > 1) or not 0 < efficiency <= 1:
            raise ValueError('Cp must match speeds and lie in [0,1]; efficiency in (0,1]')
        power = efficiency*.5*rho_ref*np.pi*D**2/4*c*u**3
        return cls.from_power_ct(wind_speeds=u, power_W=power, ct=ct, D=D,
                                 rho_ref=rho_ref, **settings)

    def _interpolate(self, values, speed, bounds):
        if bounds not in ('raise', 'zero'):
            raise ValueError("bounds must be 'raise' or 'zero'")
        u = np.asarray(speed, dtype=float)
        if not np.isfinite(u).all() or np.any(u < 0):
            raise ValueError('Speeds must be finite and nonnegative')
        if bounds == 'raise' and np.any((u < self.wind_speeds[0]) | (u > self.wind_speeds[-1])):
            raise ValueError('Speed outside turbine table; extend inputs or explicitly use bounds="zero"')
        answer = np.interp(u, self.wind_speeds, values, left=0., right=0.)
        return float(answer) if u.ndim == 0 else answer

    def power_at(self, speed, *, bounds='raise'):
        """Unyawed electrical power in W at reference density."""
        return self._interpolate(self.power_values_W, speed, bounds)

    def ct_at(self, speed, *, bounds='raise'):
        """Nominal unyawed CT, without clipping or yaw correction."""
        return self._interpolate(self.ct_values, speed, bounds)

    def cp_at(self, speed, *, bounds='raise'):
        """Equivalent electrical Cp derived from the interpolated power curve."""
        u = np.asarray(speed, float)
        power = self.power_at(u, bounds=bounds)
        cp = np.divide(power, .5*self.rho_ref*np.pi*self.D**2/4*u**3,
                       out=np.zeros_like(u), where=u > 0)
        return float(cp) if cp.ndim == 0 else cp

    def operating_point(self, rotor_velocities, *, yaw_deg=0., rho=None, weights=None):
        """Area-weighted cubic rotor speed; effective-speed yaw power approximation.

        rotor_velocities are incoming streamwise speeds, with no self-wake.
        Equal-area samples use equal weights by default. Supplied weights must
        broadcast to the sample shape, be finite/nonnegative, and have positive
        total. CT and wake speed use the same unyawed rotor-equivalent speed U_RE.
        Non-operating points return zero power; parked drag is not modelled.

        Table limits (the tabulated range is the operating range):
        U_RE below the first or above the last tabulated speed: not operating
        (zero power, CT=0, no wake). Yaw lowering the power-effective speed below
        the first tabulated speed: zero power, CT and wake still from U_RE.
        Air density raising it above the last tabulated speed: power at the last
        tabulated speed (the real wind is still below cut-out). Table CT >= 1 is
        replaced by CT_CAP for the returned CT, with a RuntimeWarning.
        """
        u = np.asarray(rotor_velocities, float)
        if u.size == 0 or not np.isfinite(u).all() or np.any(u < 0):
            raise ValueError('Rotor velocities must be nonempty, finite and nonnegative')
        yaw_deg = _finite('yaw_deg', yaw_deg)
        rho = self.rho_ref if rho is None else _finite('rho', rho)
        if abs(yaw_deg) >= 90 or rho <= 0:
            raise ValueError('Require |yaw|<90 and rho>0')
        try:
            w = np.ones_like(u) if weights is None else np.broadcast_to(np.asarray(weights, float), u.shape)
        except ValueError as exc:
            raise ValueError('Weights must broadcast to the rotor sample shape') from exc
        if not np.isfinite(w).all() or np.any(w < 0) or not np.isfinite(w.sum()) or w.sum() <= 0:
            raise ValueError('Weights must be finite/nonnegative with positive finite sum')
        w = w/w.sum()
        with np.errstate(over='ignore'):
            rews = float(np.cbrt(np.sum(w*u**3)))
        if not np.isfinite(rews):
            raise ValueError('Rotor-equivalent speed overflow')
        mean = float(np.sum(w*u))
        effective = float(rews*(rho/self.rho_ref)**(1/3)*np.cos(np.radians(yaw_deg))**(self.yaw_power_exponent/3))
        low, high = self.wind_speeds[0], self.wind_speeds[-1]
        if not low <= rews <= high:   # below cut-in or above cut-out of this table
            return OperatingPoint(mean, rews, effective, 0., 0., False)
        ct = self.ct_at(rews)
        if ct >= 1:
            warnings.warn(f"Table CT={ct:.4g} at rotor-equivalent speed {rews:.4g} m/s for "
                          f"turbine '{self.name}' is >= 1; the wake uses CT={CT_CAP}",
                          RuntimeWarning, stacklevel=2)
            ct = CT_CAP
        operating = ct > 0 and self.power_at(rews) > 0
        if not operating or effective < low:
            power = 0.
        elif effective > high:
            power = self.power_values_W[-1]
        else:
            power = self.power_at(effective)
        return OperatingPoint(mean, rews, effective, ct, power, bool(operating))

    def wake_model(self, rotor_velocities, *, TI, yaw_deg=0., rho=None, weights=None, numerics=None):
        """Build the unchanged uniform-background wake model at this rotor operating point.

        The incoming distribution is reduced to U_RE; its spatial structure is
        not transported into the new wake. Use the direct WakeModel constructor
        for its existing power-law-shear background option.
        """
        from .model import WakeModel, Numerics
        point = self.operating_point(rotor_velocities, yaw_deg=yaw_deg, rho=rho, weights=weights)
        if not point.operating:
            raise ValueError('Non-operating turbine has no supported active-wake solve')
        return WakeModel(D=self.D, H=self.H, U=point.rotor_equivalent_m_s, CT=point.CT,
                         TI=TI, yaw_deg=yaw_deg, numerics=Numerics() if numerics is None else numerics)


@dataclass(frozen=True, kw_only=True)
class FixedTurbine:
    """Constant coefficients with disk-mean wake speed and cubic power averaging."""
    D: float
    H: float
    CT: float
    CP: float

    def __post_init__(self):
        for key in ('D', 'H', 'CT', 'CP'):
            object.__setattr__(self, key, _finite(key, getattr(self, key)))
        if self.D <= 0 or self.H < self.D / 2 or not 0 < self.CT < 1 or not 0 < self.CP < 1:
            raise ValueError('Require D>0, H>=D/2, 0<CT<1 and 0<CP<1')

    def operating_point(self, rotor_velocities, *, yaw_deg=0., rho=1.225, weights=None):
        """Estimate power using mean(u**3) and a cos(yaw)**3 loss."""
        u = np.asarray(rotor_velocities, dtype=float)
        yaw_deg, rho = _finite('yaw_deg', yaw_deg), _finite('rho', rho)
        if u.size == 0 or not np.isfinite(u).all() or np.any(u < 0):
            raise ValueError('Rotor velocities must be nonempty, finite and nonnegative')
        if abs(yaw_deg) >= 90 or rho <= 0:
            raise ValueError('Require |yaw|<90 and rho>0')
        try:
            w = np.ones_like(u) if weights is None else np.broadcast_to(np.asarray(weights, float), u.shape)
        except ValueError as exc:
            raise ValueError('Weights must broadcast to the rotor sample shape') from exc
        if not np.isfinite(w).all() or np.any(w < 0) or not np.isfinite(w.sum()) or w.sum() <= 0:
            raise ValueError('Weights must be finite/nonnegative with positive finite sum')
        w = w / w.sum()
        with np.errstate(over='ignore', invalid='ignore'):
            mean = float(np.sum(w * u))
            rews = float(np.cbrt(np.sum(w * u**3)))
            effective = rews * np.cos(np.radians(yaw_deg))
            power = .5 * rho * np.pi * self.D**2 / 4 * self.CP * effective**3
        if not np.isfinite([mean, rews, effective, power]).all():
            raise ValueError('Rotor operating point overflow')
        return OperatingPoint(mean, rews, float(effective), self.CT, float(power), True)

    def wake_model(self, rotor_velocities, *, TI, yaw_deg=0., rho=1.225, weights=None, numerics=None):
        """Use the arithmetic disk-mean source speed, as in the fixed farm prototype."""
        from .model import WakeModel, Numerics
        point = self.operating_point(rotor_velocities, yaw_deg=yaw_deg, rho=rho, weights=weights)
        return WakeModel(D=self.D, H=self.H, U=point.rotor_mean_m_s, CT=self.CT,
                         TI=TI, yaw_deg=yaw_deg, numerics=Numerics() if numerics is None else numerics)
