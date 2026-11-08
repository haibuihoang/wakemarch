"""Public API of the WakeMarch 0.3 wake model (one yawed turbine, curled wake).

Distances and velocities use SI units. TI is a fraction; yaw uses degrees.
Positive yaw deflects towards negative y. Sampling z is height above ground.
"""

from dataclasses import dataclass
import math

import numpy as np

from ._kernel import YawedWakeSolver

# Largest supported |yaw| in degrees. Validation covers 40 degrees; the solver's
# initial-state construction fails near 84 degrees.
MAX_YAW_DEG = 50.0


def _finite(name, value):
    if isinstance(value, (bool, str)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class Numerics:
    """Grid and marching settings in rotor diameters.

    ``ztop_D`` is measured above hub height; the bottom is ground level.
    Storage occurs at the next marching step reaching ``store_every_D``.
    Thus dx_D=0.1 and store_every_D=0.25 currently stores every 0.3D.
    Choose h_D so that Ly_D/h_D and ztop_D/h_D are integers; otherwise the far
    boundary is rounded down to the last whole grid cell.
    """

    h_D: float = 1 / 30
    Ly_D: float = 8.0
    ztop_D: float = 3.5
    dx_D: float = 0.1
    store_every_D: float = 0.25

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = _finite(name, getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.Ly_D / self.h_D < 4 or self.ztop_D / self.h_D < 4:
            raise ValueError("Grid must span at least four intervals laterally and above hub")


@dataclass(frozen=True, kw_only=True)
class WakeModel:
    """The WakeMarch 0.3 model.

    D/H: rotor diameter/hub height in m (H >= D/2); U: undisturbed hub speed in m/s;
    CT: unyawed thrust coefficient (0 < CT < 1); TI: fraction (0..1);
    yaw_deg: degrees, |yaw| <= 50 (validated to 40); shear_alpha: nonnegative exponent.
    CQ: rotor torque coefficient Q / (0.5 rho pi R^3 U^2) at the operating point, or
    C_P / tip-speed ratio; it sets the wake rotation (hub vortex). None omits the rotation.
    rotor_sense: 'clockwise' (default) or 'counterclockwise', seen from upstream.
    thrust_ratio: measured yawed/unyawed streamwise thrust T_x/T_x0; None uses cos^2(yaw).
    hub_depth: depth of the hub deficit of the initial wake, a rotor property (0.6 for an
    actuator disk without nacelle, fitted on LES; smaller or 0 for a rotor with a nacelle).
    near_field: True (default) displays the near field of the rotor: the actuator-disk wake as a
    semi-infinite vortex cylinder (half the deficit at the rotor, the speed-up beside the near wake,
    the induction zone upstream); no constant. False shows the developed wake from the rotor plane on.
    Display only: the march, the wake position and the cross-plane velocities are unchanged.
    domain_area: cross-section in m^2 of a confined domain (LES domain, closed wind-tunnel test
    section). Adds the uniform blockage speed-up -Q(x)/A_domain from the wake's own volume-flux
    deficit Q. None (default): open atmosphere, no blockage. Useful for comparisons with confined
    simulations or experiments and for sensitivity tests.
    These input checks are computational guards, not a validated operating range.
    """

    D: float
    H: float
    U: float
    CT: float
    yaw_deg: float
    TI: float
    shear_alpha: float = 0.0
    CQ: float = None
    rotor_sense: str = "clockwise"
    thrust_ratio: float = None
    hub_depth: float = 0.6
    near_field: bool = True
    domain_area: float = None
    numerics: Numerics = Numerics()

    def __post_init__(self):
        for name in ("D", "H", "U", "CT", "yaw_deg", "TI", "shear_alpha"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if min(self.D, self.H, self.U) <= 0:
            raise ValueError("D, H and U must be positive")
        if self.H < self.D / 2:
            raise ValueError("H must be at least D/2 (rotor above the ground)")
        if not 0 < self.CT < 1:
            raise ValueError("CT must lie strictly between 0 and 1")
        if abs(self.yaw_deg) > MAX_YAW_DEG:
            raise ValueError(f"yaw_deg must lie within [-{MAX_YAW_DEG:g}, {MAX_YAW_DEG:g}] degrees "
                             "(validated to 40 degrees)")
        if not 0 <= self.TI <= 1 or self.shear_alpha < 0:
            raise ValueError("TI must be a fraction in [0, 1]; shear_alpha must be nonnegative")
        if self.CQ is not None:
            object.__setattr__(self, "CQ", _finite("CQ", self.CQ))
            if self.CQ < 0:
                raise ValueError("CQ must be nonnegative")
        if self.rotor_sense not in ("clockwise", "counterclockwise"):
            raise ValueError("rotor_sense must be 'clockwise' or 'counterclockwise' (seen from upstream)")
        if self.thrust_ratio is not None:
            object.__setattr__(self, "thrust_ratio", _finite("thrust_ratio", self.thrust_ratio))
            if not 0 < self.thrust_ratio <= 1.2:
                raise ValueError("thrust_ratio must lie in (0, 1.2]")
        object.__setattr__(self, "hub_depth", _finite("hub_depth", self.hub_depth))
        if not 0 <= self.hub_depth < 1:
            raise ValueError("hub_depth must lie in [0, 1)")
        if not isinstance(self.near_field, (bool, np.bool_)):
            raise TypeError("near_field must be True or False")
        object.__setattr__(self, "near_field", bool(self.near_field))
        if self.domain_area is not None:
            object.__setattr__(self, "domain_area", _finite("domain_area", self.domain_area))
            if self.domain_area <= np.pi * self.D ** 2 / 4:
                raise ValueError("domain_area must exceed the rotor area (m^2)")
        if not isinstance(self.numerics, Numerics):
            raise TypeError("numerics must be a Numerics instance")

    def solve(self, max_dist_D=10.0, *, progress=True):
        """March a fresh solver; return independent results for this run.

        The kernel includes extra downstream distance for yaw reconstruction.
        No previous result is mutated or reused. No resolution preset is
        selected implicitly.
        ``progress=True`` displays a notebook/terminal bar (the default).
        Use False for silent execution, or a callable accepting
        (completed_D, total_D) for custom reporting. Distances include the
        reconstruction buffer. Callback errors propagate to the caller.
        """
        distance = _finite("max_dist_D", max_dist_D)
        n = self.numerics
        if distance <= 0 or n.dx_D > distance or n.store_every_D > distance:
            raise ValueError("max_dist_D must be positive and at least dx_D and store_every_D")
        if progress is not True and progress is not False and not callable(progress):
            raise TypeError("progress must be True, False, or a callable")
        bar = None
        if progress is True:
            from ._progress import ProgressBar
            bar = ProgressBar()
        callback = bar if bar is not None else (progress if callable(progress) else None)
        success = False
        try:
            solver = YawedWakeSolver(
                D=self.D, H=self.H, U=self.U, CT=self.CT, yaw_deg=self.yaw_deg,
                TI=self.TI, shear_alpha=self.shear_alpha, CQ=self.CQ,
                rotor_sense=self.rotor_sense, thrust_ratio=self.thrust_ratio, hub_depth=self.hub_depth,
                near_field=self.near_field, domain_area=self.domain_area,
                h_D=n.h_D, Ly_D=n.Ly_D, ztop_D=n.ztop_D,
            )
            solver.solve(max_dist_D=distance, dx_D=n.dx_D, store_every_D=n.store_every_D,
                         progress_callback=callback)
            result = WakeResult(solver)
            success = True
            return result
        finally:
            if bar is not None:
                bar.close(success)


class WakeResult:
    """Results from ``WakeModel.solve``; create through the model, not directly.

    History arrays are in marching coordinates (x_s,z_relative,y), without the
    near-field display. Use sampling methods for reconstructed physical-space values.
    History properties return read-only views to prevent accidental edits.
    """

    def __init__(self, solver):
        self._solver = solver

    @staticmethod
    def _view(array):
        result = array.view()
        result.flags.writeable = False
        return result

    @property
    def x(self):
        """Stored marching coordinates x_s in m, including reconstruction buffer."""
        return self._view(self._solver.xs)

    @property
    def y(self):
        """Lateral mesh coordinates in m, relative to the turbine centre."""
        return self._view(self._solver.y)

    @property
    def z(self):
        """Mesh heights above ground in m."""
        return self._view(self._solver.z + self._solver.H)

    @property
    def z_relative(self):
        """Mesh heights relative to hub in m, for marching-array diagnostics."""
        return self._view(self._solver.z)

    @property
    def marching_deficit(self):
        """Marched (developed-wake) streamwise deficit without the near-field display, shape (x_s, z_relative, y), m/s."""
        return self._view(self._solver.du_hist)

    @property
    def streamwise_vorticity(self):
        """Streamwise vorticity history in 1/s, same ordering as deficit."""
        return self._view(self._solver.om_hist)

    @property
    def streamwise_velocity_derivative(self):
        """Stored marching u_x field in 1/s (initial plane is zero)."""
        return self._view(self._solver.dudx_hist)

    def sample_deficit(self, x, y, z, *, bounds="raise"):
        """Reconstructed deficit in m/s at broadcastable physical coordinates.

        Includes the near field of the rotor (``WakeModel.near_field``): upstream
        of the tilted rotor (x+y*tan(yaw)<0) this is the induction zone, zero
        with near_field=False. With ``domain_area`` the blockage offset is added.
        Downstream points outside stored support raise by default;
        ``bounds='zero'`` explicitly requests the zero-deficit extension of the
        marched wake. Negative heights are invalid.
        """
        x, y, z, outside = self._sampling_points(x, y, z, bounds)
        # Flatten first so the kernel also supports scalar/list inputs.
        out = self._solver.sample_global(x.ravel(), y.ravel(), z.ravel()).reshape(x.shape)
        out = np.where(outside, 0.0, out)   # bounds='zero': zero outside the stored support
        return out.item() if out.ndim == 0 else out

    def sample_cross_velocity(self, x, y, z, *, bounds="raise"):
        """Cross-plane velocities ``(v, w)`` in m/s at broadcastable physical coordinates.

        v is lateral (+y) and w vertical (+z). This is the total in-plane flow the
        solver uses to transport the wake: the part induced by the streamwise
        vorticity plus the entrainment flow from the streamwise velocity change,
        with w = 0 at the ground. The near-field display and the blockage offset of
        ``sample_deficit`` are not applied. Coordinates and ``bounds`` behave as in ``sample_deficit``;
        upstream of the tilted rotor both components are zero. The fields are
        computed once, on the first call.
        """
        x, y, z, _ = self._sampling_points(x, y, z, bounds)
        v, w = self._solver.sample_velocity(x.ravel(), y.ravel(), z.ravel())
        v, w = v.reshape(x.shape), w.reshape(x.shape)
        if v.ndim == 0:
            return v.item(), w.item()
        return v, w

    def _sampling_points(self, x, y, z, bounds):
        """Validate and broadcast physical sampling coordinates (shared by all samplers);
        also return the mask of downstream points outside the stored support."""
        if bounds not in ("raise", "zero"):
            raise ValueError("bounds must be 'raise' or 'zero'")
        x, y, z = np.broadcast_arrays(np.asarray(x, dtype=float),
                                       np.asarray(y, dtype=float), np.asarray(z, dtype=float))
        if not all(np.isfinite(a).all() for a in (x, y, z)) or np.any(z < 0):
            raise ValueError("Coordinates must be finite and z must be nonnegative")
        s = self._solver
        xs = x + y * np.tan(s.yaw)
        outside = (xs >= 0) & ((xs > s.xs[-1]) | (y < s.y[0]) | (y > s.y[-1])
                               | (z - s.H < s.z[0]) | (z - s.H > s.z[-1]))
        if bounds == "raise" and np.any(outside):
            raise ValueError("Sampling point is outside stored wake support; use bounds='zero' explicitly")
        return x, y, z, outside

    def sample_streamwise_velocity(self, x, y, z, *, bounds="raise"):
        """Background power-law velocity minus reconstructed deficit, m/s.

        Under bounds='zero', outside-support values equal background velocity.
        This is a velocity output, not a turbine power or farm prediction.
        """
        deficit = self.sample_deficit(x, y, z, bounds=bounds)
        s = self._solver
        background = s.U * (np.maximum(np.asarray(z, dtype=float), 0.05 * s.H) / s.H) ** s.shear_alpha
        out = background - deficit
        return out.item() if np.ndim(out) == 0 else out
