"""Small public API around the accepted recommended numerical kernel.

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
    """Accepted recommended model with no optional empirical corrections.

    D/H: rotor diameter/hub height in m (H >= D/2); U: undisturbed hub speed in m/s;
    CT: unyawed thrust coefficient (0 < CT < 1); TI: fraction (0..1);
    yaw_deg: degrees, |yaw| <= 50 (validated to 40); shear_alpha: nonnegative exponent.
    These input checks are computational guards, not a validated operating range.
    """

    D: float
    H: float
    U: float
    CT: float
    yaw_deg: float
    TI: float
    shear_alpha: float = 0.0
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
        if not isinstance(self.numerics, Numerics):
            raise TypeError("numerics must be a Numerics instance")

    def solve(self, max_dist_D=10.0, *, progress=True):
        """March a fresh solver; return independent results for this run.

        The kernel includes extra downstream distance for yaw reconstruction.
        No previous result is mutated or reused. Numerical defaults match the
        research solver; no resolution preset is selected implicitly.
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
            solver = YawedWakeSolver.recommended(
                D=self.D, H=self.H, U=self.U, CT=self.CT, yaw_deg=self.yaw_deg,
                TI=self.TI, shear_alpha=self.shear_alpha,
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
    display ramp. Use sampling methods for reconstructed physical-space values.
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
        """Unramped streamwise deficit, shape (x_s, z_relative, y), m/s."""
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

        Includes the accepted near-wake display ramp. Upstream of the tilted
        rotor (x+y*tan(yaw)<0) the deficit is zero. Downstream points outside
        stored support raise by default; ``bounds='zero'`` explicitly requests
        the historical zero-deficit extension. Negative heights are invalid.
        """
        if bounds not in ("raise", "zero"):
            raise ValueError("bounds must be 'raise' or 'zero'")
        x, y, z = np.broadcast_arrays(np.asarray(x, dtype=float),
                                       np.asarray(y, dtype=float), np.asarray(z, dtype=float))
        if not all(np.isfinite(a).all() for a in (x, y, z)) or np.any(z < 0):
            raise ValueError("Coordinates must be finite and z must be nonnegative")
        s = self._solver
        xs = x + y * np.tan(s.yaw)
        outside = ((xs > s.xs[-1]) | (y < s.y[0]) | (y > s.y[-1])
                   | (z - s.H < s.z[0]) | (z - s.H > s.z[-1]))
        if bounds == "raise" and np.any((xs >= 0) & outside):
            raise ValueError("Sampling point is outside stored wake support; use bounds='zero' explicitly")
        # Flatten first so the original kernel also supports scalar/list inputs.
        out = s.sample_global(x.ravel(), y.ravel(), z.ravel()).reshape(x.shape)
        return out.item() if out.ndim == 0 else out

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
