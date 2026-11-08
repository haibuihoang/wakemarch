"""Numerical kernel of WakeMarch 0.3 (internal module; use WakeModel).

A cross-plane marching model of the wake of one yawed turbine (manuscript, Section 2):
streamwise deficit and streamwise vorticity marched downstream on planes parallel to
the rotor; cross-plane velocity = vortex-induced part + entrainment from continuity;
impermeable ground, open far field, optional power-law shear. Only the features of the
final model are included; the research solver (src/curl_pns.py of the paper companion)
keeps the experimental switches.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
from scipy.special import ellipe, ellipk

# Constants of the model (manuscript, Section 2.8)
THRUST_EXP = 2.0          # T_x = T_x0 cos^p(yaw)
ETA = 0.95                # share of T_x carried as wake momentum after the pressure recovery
A_DEV = 2.0               # near-wake development constant of F(x / x0); 2 conserves the cumulative mixing (no tuning)
KAPPA_OMEGA = 0.25        # vorticity diffusivity / eddy viscosity
EDGE = 0.2                # thickness of the smoothed initial wake edge (units of the wake radius)
HUB_DEPTH, HUB_RADIUS = 0.6, 0.35   # hub deficit f = 1 - h exp(-(rho/rho_h)^2)
HUB_CORE_D = 0.11         # Gaussian core of the hub vortex (units of D)
CT_CAP = 0.97             # largest normal-velocity thrust coefficient used for the induction factor
NMULT = 14                # multipole terms of the open far field
NF_CORE = 0.02            # vortex core of the near-field sheet (units of R_w); regularises points on the sheet


def lap1d(n, h):
    return sp.diags([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)], [-1, 0, 1]) / h**2


def lap1d_neumann_first(n, h):
    """Rows 0..n-1: Neumann (mirror ghost) at row 0, Dirichlet zero beyond row n-1."""
    T = lap1d(n, h).tolil()
    T[0, 1] = 2.0 / h**2
    return T.tocsc()


def laplacian_2d(ny, nz, dy, dz):
    """Laplacian on a (nz, ny) C-ordered field (index = iz*ny + iy), zero Dirichlet outside."""
    return sp.kron(sp.eye(nz), lap1d(ny, dy)) + sp.kron(lap1d(nz, dz), sp.eye(ny))


# ---- near field of the rotor (display only) ---------------------------------------------------------------
# The marched state is the developed wake. Close to the rotor the deficit is displayed with the near field of the
# actuator-disk wake (manuscript, Section 2.8): the initial profile p(rho), hub term included, written as a
# superposition of semi-infinite vortex cylinders starting at the rotor plane. It has no constant. The field of a
# unit cylinder is tabulated once in (xi, rho) = (x_s, r) / R_w by summing vortex rings; a cylinder of radius c
# follows by scale invariance.
_NF_XI = np.concatenate([np.linspace(-6, -2.2, 20), np.linspace(-2, 4, 121), np.linspace(4.25, 20, 64)])
_NF_RHO = np.concatenate([np.linspace(0, 0.6, 13), np.linspace(0.625, 1.6, 79), np.linspace(1.65, 6, 88)])
_NF_S = np.concatenate([np.arange(0.0, 25.0, 0.02), np.geomspace(25.0, 400.0, 200)])
_NF_TABLES = {}


def _ring_axial_velocity(b, z, r):
    """Axial velocity of a vortex ring (radius b, unit circulation) at axial offset z and radius r, with core NF_CORE."""
    den = (b + r) ** 2 + z ** 2 + NF_CORE ** 2
    m = np.clip(4 * b * r / den, 0, 1 - 1e-12)
    return (ellipk(m) + (b * b - r * r - z * z) / ((b - r) ** 2 + z ** 2 + NF_CORE ** 2) * ellipe(m)) / (2 * np.pi * np.sqrt(den))


def wake_mask(rho, edge=None):
    """Smoothed top hat of the initial wake, m = (1 - tanh((rho - 1) / edge)) / 2."""
    return 0.5 * (1 - np.tanh((rho - 1.0) / (EDGE if edge is None else edge)))


def _sharp_cylinder():
    """u/gamma of a sharp semi-infinite vortex cylinder of radius 1 on (_NF_XI, _NF_RHO), as a sum of rings (cached)."""
    if "sharp" not in _NF_TABLES:
        sm, ds = 0.5 * (_NF_S[1:] + _NF_S[:-1]), np.diff(_NF_S)
        Xg, Rg = np.meshgrid(_NF_XI, _NF_RHO, indexing="ij")
        u1 = np.zeros_like(Xg)
        for s_, d_ in zip(sm, ds):
            u1 += d_ * _ring_axial_velocity(1.0, Xg - s_, Rg)
        _NF_TABLES["sharp"] = RegularGridInterpolator((_NF_XI, _NF_RHO), u1, bounds_error=False, fill_value=None)
    return _NF_TABLES["sharp"]


def initial_profile(rho, edge, hub_depth, hub_radius, norm):
    """Normalised initial deficit p(rho)/dU0 of the march: smoothed top hat times the hub term (manuscript Eqs. 2 and 7)."""
    return wake_mask(rho, edge) * (1.0 - hub_depth * np.exp(-(rho / hub_radius) ** 2)) * norm


def near_field_table(edge, hub_depth, hub_radius, norm):
    """Interpolator of the near-field deficit /dU0 on (xi, rho): the initial profile p(rho) written as a superposition of
    semi-infinite vortex cylinders of radius c with strengths -dp/dc (scale invariance u_c(xi, rho) = u_1(xi/c, rho/c)).
    Far downstream it tends to p(rho); at the rotor plane it is p(rho)/2. Cached per profile."""
    key = tuple(round(float(v), 9) for v in (edge, hub_depth, hub_radius, norm))
    if key not in _NF_TABLES:
        sharp = _sharp_cylinder()
        Xg, Rg = np.meshgrid(_NF_XI, _NF_RHO, indexing="ij")
        ce = np.linspace(0.0, 1 + 6 * edge, 482)                    # layer edges; cylinders at the midpoints
        c = 0.5 * (ce[1:] + ce[:-1])
        w = -(initial_profile(ce[1:], edge, hub_depth, hub_radius, norm) - initial_profile(ce[:-1], edge, hub_depth, hub_radius, norm))
        u = np.zeros_like(Xg)
        for ci, wi in zip(c, w):
            if abs(wi) < 1e-12:
                continue
            xi, rh = Xg / ci, Rg / ci
            pts = np.stack([np.clip(xi, _NF_XI[0], _NF_XI[-1]).ravel(), np.clip(rh, 0, _NF_RHO[-1]).ravel()], -1)
            ui = sharp(pts).reshape(Xg.shape)
            ui = ui * np.minimum(1.0, (_NF_XI[0] / np.minimum(xi, -1e-9)) ** 2) * (xi < 0) + ui * (xi >= 0)   # beyond the table:
            ui = ui * (_NF_RHO[-1] / np.maximum(rh, _NF_RHO[-1])) ** 3                                           # far-field decay
            u += wi * ui
        _NF_TABLES[key] = RegularGridInterpolator((_NF_XI, _NF_RHO), u, bounds_error=False, fill_value=None)
    return _NF_TABLES[key]


def near_wake_length(CT, TI):
    """Near-wake length x0 / D of the unyawed rotor (Bastankhah & Porte-Agel 2016)."""
    sq = np.sqrt(1.0 - min(CT, 0.99))
    return (1.0 + sq) / (np.sqrt(2.0) * (2.32 * TI + 0.154 * (1.0 - sq)))


class YawedWakeSolver:
    """Grid, initial state and march of the final model. Distances in m, yaw in degrees.

    CQ: torque coefficient Q / (0.5 rho pi R^3 U^2) at the operating point (None: no wake rotation).
    rotor_sense: 'clockwise' or 'counterclockwise' seen from upstream.
    thrust_ratio: measured T_x / T_x0 (None: cos^2(yaw)).
    hub_depth: depth h of the hub deficit (0.6: actuator disk without nacelle; 0: none).
    near_field: display the near field of the rotor (semi-infinite vortex cylinder); False shows the developed wake from x_s = 0.
    domain_area: cross-section (m^2) of a confined domain (LES, closed wind tunnel): adds the uniform blockage speed-up
        -Q(x)/A of the wake's own volume-flux deficit Q; None (default) for the open atmosphere.
    """

    def __init__(self, D, H, U, CT, yaw_deg, TI, shear_alpha=0.0, CQ=None, rotor_sense="clockwise",
                 thrust_ratio=None, hub_depth=HUB_DEPTH, near_field=True, domain_area=None, Ly_D=8.0, ztop_D=3.5, h_D=1.0 / 30.0):
        self.D, self.H, self.U, self.CT, self.TI = D, H, U, CT, TI
        self.yaw = np.radians(yaw_deg)
        self.k_wake = 0.3837 * TI + 0.003678          # wake growth rate (Niayifar & Porte-Agel 2016)
        self.shear_alpha = shear_alpha
        self.CQ, self.rotor_sense, self.hub_depth = CQ, rotor_sense, hub_depth
        c = np.cos(self.yaw)
        self.thrust_ratio = float(c ** THRUST_EXP) if thrust_ratio is None else float(thrust_ratio)
        self.CT_n = float(min(CT * self.thrust_ratio * ETA / c**3, max(CT_CAP, CT)))   # normal-velocity thrust coefficient
        self.CT_0 = float(min(CT * ETA, max(CT_CAP, CT)))                              # unyawed wake thrust coefficient
        self.x0 = near_wake_length(CT, TI) * D
        self.near_field = bool(near_field)                      # display of the rotor near field (vortex cylinder)
        self.domain_area = domain_area                          # confined domain cross-section (m^2) for the blockage offset; None: open
        h = h_D * D
        self.y = np.arange(-Ly_D / 2 * D, Ly_D / 2 * D + h / 2, h)
        self.z = np.arange(-H, ztop_D * D + h / 2, h)           # relative to the hub; row 0 is the ground
        self.ny, self.nz, self.h = len(self.y), len(self.z), h
        self.Y, self.Z = np.meshgrid(self.y, self.z)             # shape (nz, ny)
        self._advection_grid = np.meshgrid(np.arange(self.ny, dtype=float), np.arange(self.nz, dtype=float))
        nyi, nzi = self.ny - 2, self.nz - 2
        self._poisson = splu(laplacian_2d(nyi, nzi, h, h).tocsc())
        self._L_int = laplacian_2d(nyi, nzi, h, h).tocsr()
        self._nzphi = self.nz - 1                                 # entrainment potential: rows 0..nz-2, Neumann ground
        zabs = np.maximum(self.Z + H, 0.05 * H)
        self.Ubg = U * (zabs / H) ** shear_alpha
        self.dUbg_dz = shear_alpha * self.Ubg / zabs * (self.Z + H > 0.05 * H)
        self._L_du = (sp.kron(sp.eye(self.nz - 1), lap1d(nyi, h)) + sp.kron(lap1d_neumann_first(self.nz - 1, h), sp.eye(nyi))).tocsr()
        self._build_far_field()
        self._phi_solver = splu((sp.kron(sp.eye(self._nzphi), lap1d(nyi, h))
                                 + sp.kron(lap1d_neumann_first(self._nzphi, h), sp.eye(nyi))).tocsc())

    # ---- rotor force and initial state ---------------------------------------
    def initial_state_params(self):
        """Rotor force, initial deficit and circulation (manuscript Eqs. 1, 3, 5)."""
        U, g, c = self.U, self.yaw, np.cos(self.yaw)
        A = np.pi * self.D**2 / 4
        dU0 = (1 - np.sqrt(1 - self.CT_0)) * U * c                       # projected deficit
        uw0 = U - dU0
        Tx = 0.5 * A * U**2 * self.CT_n * c**3                           # wake momentum eta T_x / rho
        s_n = np.sqrt(1 - self.CT_n)                                     # normal-velocity actuator-disk theory:
        Rn = np.sqrt(0.5 * A * U * c * (1 + s_n) / uw0 / (np.pi * c))    # wake radius (sets the sheet amplitude and the root bracket)
        Gamma0 = 0.5 * self.D * U * self.CT * np.sin(g) * c**2 * self.thrust_ratio / c**3   # 4 F_y / (pi rho D U)
        a = 0.5 * (1 - s_n)                                              # axial induction
        return dict(dU0=dU0, uw0=uw0, Tx=Tx, Rn=Rn, Gamma0=Gamma0, a=a)

    def initialize(self):
        p = self.initial_state_params()
        c, d0, hh = np.cos(self.yaw), p["dU0"], self.h**2

        def mask_of(R):
            r = np.hypot(self.Y / (R * c), self.Z / R)
            return 0.5 * (1 - np.tanh((r - 1.0) / EDGE))

        def profile(R):
            m = mask_of(R)
            rho2 = (self.Y / (R * c)) ** 2 + (self.Z / R) ** 2
            fm = m * (1.0 - self.hub_depth * np.exp(-rho2 / HUB_RADIUS**2))
            return fm * (m.sum() / fm.sum())                                 # same mean deficit as the top hat

        def theta(R):
            m = profile(R)
            return np.sum((self.U - d0 * m) * d0 * m) * hh

        # the smoothed edge and the hub term change the momentum: find the radius whose discrete momentum is eta T_x
        Reff = brentq(lambda R: theta(R) - p["Tx"], 0.4 * p["Rn"], 2.0 * p["Rn"])
        self.Reff0 = Reff
        m0 = mask_of(Reff)
        rho0 = (self.Y / (Reff * c)) ** 2 + (self.Z / Reff) ** 2
        self._profile_norm = m0.sum() / (m0 * (1.0 - self.hub_depth * np.exp(-rho0 / HUB_RADIUS**2))).sum()
        mask = m0
        du = d0 * profile(Reff)
        # omega = V dm/dz with V = Gamma0 / (2 R_n cos(yaw)); on the smoothed mask the upper half carries 0.91-1.00 Gamma0
        # (normalising to exactly Gamma0 increased the LES wake-centre error by 11 %)
        omega = p["Gamma0"] / (2 * p["Rn"] * c) * np.gradient(mask, self.h, axis=0)
        if self.CQ is not None and self.CQ > 0:                              # hub vortex from the torque, opposite circulation on the wake edge
            sign = -1.0 if self.rotor_sense == "clockwise" else 1.0
            Gh = 0.5 * np.pi * self.CQ * self.U * self.D / (1.0 - p["a"])
            sc = HUB_CORE_D * self.D
            gk = np.exp(-(self.Y**2 + self.Z**2) / (2 * sc**2))
            om_rot = sign * Gh * gk / (gk.sum() * hh)
            gz_, gy_ = np.gradient(mask, self.h, self.h)
            e = np.hypot(gy_, gz_)
            om_rot = om_rot - sign * Gh * e / (e.sum() * hh)
            omega = omega + om_rot
        self.sigma0 = Reff / 2
        self._p0 = p
        return du, omega

    # ---- open far field ------------------------------------------------------
    def _build_far_field(self):
        ny, nz = self.ny, self.nz
        mask = np.zeros((nz, ny), bool)
        mask[:, 0] = mask[:, -1] = True; mask[-1, :] = True
        self._psi_bmask = mask
        self._psi_zt = (self.Y + 1j * self.Z)[mask]
        pm = np.zeros((nz, ny), bool)
        pm[0:nz - 1, 0] = pm[0:nz - 1, -1] = True; pm[-1, :] = True
        self._phi_bmask = pm
        self._phi_zt = (self.Y + 1j * self.Z)[pm]
        self._zs = self.Y + 1j * self.Z
        self._zs_img = self.Y + 1j * (-2 * self.H - self.Z)
        w = np.zeros((nz, ny)); w[1:-1, 1:-1] = self.h**2
        self._w_psi = w
        wp = np.zeros((nz, ny)); wp[0:nz - 1, 1:-1] = self.h**2; wp[0, 1:-1] = 0.5 * self.h**2
        self._w_phi = wp
        self._multipole_powers = []
        zk = np.ones_like(self._zs.ravel()); zik = np.ones_like(self._zs_img.ravel())
        for _ in range(NMULT):
            zk = zk * self._zs.ravel(); zik = zik * self._zs_img.ravel()
            self._multipole_powers.append((zk, zik))
        self._multipole_targets = {}
        for targets in (self._psi_zt, self._phi_zt):
            inv = 1.0 / targets; ik = np.ones_like(targets); powers = []
            for _ in range(NMULT):
                ik = ik * inv
                powers.append(ik)
            self._multipole_targets[id(targets)] = (powers, np.log(np.abs(targets)))

    def _far_field(self, f, w, zt, mask, even_image):
        """Dirichlet data for lap(u) = -f in free space plus the ground image (odd for psi, even for phi), by multipoles about the rotor axis."""
        out = np.zeros((self.nz, self.ny))
        src = (w * f).ravel()
        sg = 1.0 if even_image else -1.0
        powers, log_radius = self._multipole_targets[id(zt)]
        val = np.zeros(len(zt), dtype=complex)
        for k in range(1, NMULT + 1):
            zk, zik = self._multipole_powers[k - 1]
            m = (src * zk).sum() + sg * (src * zik).sum()
            val += m * powers[k - 1] / k
        S = src.sum() * (1.0 + sg) * log_radius - val.real
        out[mask] = -S / (2 * np.pi)
        return out

    # ---- operators -------------------------------------------------------------
    def solve_psi(self, omega):
        psi = self._far_field(omega, self._w_psi, self._psi_zt, self._psi_bmask, even_image=False)
        B = np.zeros((self.nz - 2, self.ny - 2))
        B[:, 0] += psi[1:-1, 0]; B[:, -1] += psi[1:-1, -1]; B[0, :] += psi[0, 1:-1]; B[-1, :] += psi[-1, 1:-1]
        rhs = -omega[1:-1, 1:-1] - B / self.h**2
        psi[1:-1, 1:-1] = self._poisson.solve(rhs.ravel()).reshape(self.nz - 2, self.ny - 2)
        return psi

    def solve_phi(self, dudx):
        """lap(phi) = -u_x (continuity); open far field, Neumann (impermeable) ground."""
        phi = self._far_field(dudx, self._w_phi, self._phi_zt, self._phi_bmask, even_image=True)
        B = np.zeros((self._nzphi, self.ny - 2))
        B[:, 0] += phi[0:self.nz - 1, 0]; B[:, -1] += phi[0:self.nz - 1, -1]; B[-1, :] += phi[self.nz - 1, 1:-1]
        rhs = -dudx[0:self.nz - 1, 1:-1] - B / self.h**2
        phi[0:self.nz - 1, 1:-1] = self._phi_solver.solve(rhs.ravel()).reshape(self._nzphi, self.ny - 2)
        return phi

    def velocities(self, psi):
        dpsi_dz, dpsi_dy = np.gradient(psi, self.h, self.h, edge_order=2)
        return dpsi_dz, -dpsi_dy

    def vortex_velocity(self, om, dt):
        """Solenoidal velocity induced by om, time-centred over the step (predictor-corrector)."""
        v, w = self.velocities(self.solve_psi(om))
        om_half = self.advect([om], v, w, 0.5 * dt)[0]
        return self.velocities(self.solve_psi(om_half))

    def _sample(self, f, iy, iz, mirror_ground=False):
        if mirror_ground:   # deficit: reflect about the ground (zero flux)
            iz = np.abs(iz)
        return map_coordinates(f, [iz, iy], order=3, mode="constant", cval=0.0)

    def advect(self, fields, v, w, dt, mirror_first=False):
        """Semi-Lagrangian RK2 backward trace; v, w frozen over the step; dt scalar or per-point array."""
        iy0, iz0 = self._advection_grid
        iy_mid = iy0 - 0.5 * dt * v / self.h
        iz_mid = iz0 - 0.5 * dt * w / self.h
        v2, w2 = self._sample(v, iy_mid, iz_mid), self._sample(w, iy_mid, iz_mid)
        iy_src = iy0 - dt * v2 / self.h
        iz_src = iz0 - dt * w2 / self.h
        return [self._sample(f, iy_src, iz_src, mirror_ground=(mirror_first and i == 0)) for i, f in enumerate(fields)]

    def diffuse(self, f, nu, dt, neumann_ground=False):
        """Backward Euler on interior nodes with coefficient dt*nu (scalar or array): a cached factorisation of
        (I - cbar L) with iterative refinement to a relative residual of 1e-9; refreshed when the mean
        coefficient moves by 12 per cent or the refinement stalls."""
        ng = bool(neumann_ground)
        if ng:
            c = np.broadcast_to(dt * nu, f.shape)[:-1, 1:-1]; rhs = f[:-1, 1:-1].ravel(); L = self._L_du; shp = (self.nz - 1, self.ny - 2)
        else:
            c = np.broadcast_to(dt * nu, f.shape)[1:-1, 1:-1]; rhs = f[1:-1, 1:-1].ravel(); L = self._L_int; shp = (self.nz - 2, self.ny - 2)
        cv = np.ascontiguousarray(c).ravel(); n = len(cv)
        out = np.zeros_like(f)
        cbar = float(cv.mean())
        if cbar <= 0.0:
            sol = rhs
        else:
            cache = self.__dict__.setdefault("_dcache", {})
            ent = cache.get(ng)
            if ent is None or ent[0] <= 0.0 or abs(cbar / ent[0] - 1.0) > 0.12:
                ent = (cbar, splu((sp.identity(n, format="csc") - cbar * L).tocsc()))
                cache[ng] = ent
            lu = ent[1]
            x = lu.solve(rhs); bn = np.linalg.norm(rhs) + 1e-300
            ok = False
            for _ in range(14):
                r = rhs - (x - cv * (L @ x))
                if np.linalg.norm(r) < 1e-9 * bn:
                    ok = True; break
                x = x + lu.solve(r)
            if not ok:
                lu = splu((sp.identity(n, format="csc") - sp.diags(cv) @ L).tocsc())
                x = lu.solve(rhs)
                cache[ng] = (cbar, lu)
            sol = x
        if ng:
            out[:-1, 1:-1] = sol.reshape(shp)
        else:
            out[1:-1, 1:-1] = sol.reshape(shp)
        return out

    def laplacian(self, f):
        """Five-point Laplacian with the mirrored (Neumann) ground row."""
        out = np.zeros_like(f)
        out[1:-1, 1:-1] = (f[1:-1, 2:] + f[1:-1, :-2] + f[2:, 1:-1] + f[:-2, 1:-1] - 4 * f[1:-1, 1:-1]) / self.h**2
        out[0, 1:-1] = (f[0, 2:] + f[0, :-2] + 2 * f[1, 1:-1] - 4 * f[0, 1:-1]) / self.h**2
        return out

    # ---- eddy viscosity ------------------------------------------------------
    def eddy_viscosity(self, x, du):
        """nu = cos^2(yaw) k_w sigma(x) u_bar F(x / x0) (manuscript Eq. 18)."""
        sigma = self.sigma0 + self.k_wake * x
        sum_du = du.sum()
        u_bar = self.U - (du**2).sum() / sum_du if sum_du > 0 else self.U
        nu = self.k_wake * sigma * u_bar
        q = (x / self.x0) ** 2.0
        return np.cos(self.yaw) ** 2 * nu * (1.0 - (1.0 - A_DEV * q) * np.exp(-q))   # cos^2: yaw factor of the mixing

    # ---- marching ------------------------------------------------------------
    def solve(self, max_dist_D=10.0, dx_D=0.1, store_every_D=0.25, progress_callback=None):
        buffer = (3.0 * np.tan(abs(self.yaw)) + 1.0) * self.D       # room for the reconstruction on tilted planes
        L = max_dist_D * self.D + buffer
        if progress_callback is not None:
            progress_callback(0.0, L / self.D)
        du, om = self.initialize()
        xs, dus, oms, dudxs = [0.0], [du.copy()], [om.copy()], [np.zeros_like(du)]
        x, dx = 0.0, dx_D * self.D
        dudx = np.zeros_like(du)
        while x < L:
            u0 = np.maximum(self.Ubg - du, 0.3 * self.Ubg)
            vs, ws = self.vortex_velocity(om, dx / u0)
            u = np.maximum(self.Ubg - du, 0.3 * self.Ubg)
            nu = self.eddy_viscosity(x, du)
            uz, uy = np.gradient(u, self.h, self.h, edge_order=2)
            lap_u = self.laplacian(u - self.Ubg)
            v, w = vs, ws
            for _ in range(3):                                          # u_x depends on (v, w), which depend on u_x through phi
                dudx = (nu * lap_u - (v * uy + w * uz)) / u
                gz, gy = np.gradient(self.solve_phi(dudx), self.h, self.h, edge_order=2)
                v, w = vs + gy, ws + gz
                w[0, :] = 0.0
            dt = dx / u
            du, om = self.advect([du, om], v, w, dt, mirror_first=True)
            if self.shear_alpha != 0.0:
                du = du + dt * w * self.dUbg_dz                         # vertical advection of the background shear
            div = np.gradient(v / u, self.h, axis=1, edge_order=2) + np.gradient(w / u, self.h, axis=0, edge_order=2)
            om = om * np.exp(-dx * div)                                 # conservative form of the vorticity transport
            du = self.diffuse(du, nu / u, dx, neumann_ground=True)
            om = self.diffuse(om, KAPPA_OMEGA * nu / u, dx)
            x += dx
            if progress_callback is not None:
                progress_callback(min(x / self.D, L / self.D), L / self.D)
            if x - xs[-1] >= store_every_D * self.D - 1e-9:
                xs.append(x); dus.append(du.copy()); oms.append(om.copy()); dudxs.append(dudx.copy())
        self.xs = np.array(xs)
        self.du_hist = np.stack(dus)
        self.om_hist = np.stack(oms)
        self.dudx_hist = np.stack(dudxs)
        self._interp = RegularGridInterpolator((self.xs, self.z, self.y), self.du_hist, bounds_error=False, fill_value=0.0)

    # ---- cross-plane velocities ------------------------------------------------
    def build_velocity_history(self):
        """Cross-plane velocities (v, w) of every stored plane: vortex-induced part plus the entrainment flow."""
        vs_, ws_ = [], []
        for om, dudx in zip(self.om_hist, self.dudx_hist):
            v, w = self.velocities(self.solve_psi(om))
            gz, gy = np.gradient(self.solve_phi(dudx), self.h, self.h, edge_order=2)
            v, w = v + gy, w + gz
            w[0, :] = 0.0
            vs_.append(v); ws_.append(w)
        self.v_hist, self.w_hist = np.stack(vs_), np.stack(ws_)
        self._vint = RegularGridInterpolator((self.xs, self.z, self.y), self.v_hist, bounds_error=False, fill_value=0.0)
        self._wint = RegularGridInterpolator((self.xs, self.z, self.y), self.w_hist, bounds_error=False, fill_value=0.0)

    def sample_velocity(self, X, Y, Z):
        """(v, w) at global (X, Y, Z) with the same mapping as the deficit."""
        if not hasattr(self, "_vint"):
            self.build_velocity_history()
        xs = X + Y * np.tan(self.yaw)
        pts = np.stack([xs.ravel(), (Z - self.H).ravel(), Y.ravel()], axis=1)
        v, w = self._vint(pts).reshape(X.shape), self._wint(pts).reshape(X.shape)
        v[xs < 0] = 0.0; w[xs < 0] = 0.0
        return v, w

    # ---- diagnostics / reconstruction -------------------------------------------
    def circulation_top(self, om):
        """Circulation above hub height (z > 0) and below (z < 0)."""
        return np.sum(om[self.Z > 0]) * self.h**2, np.sum(om[self.Z < 0]) * self.h**2

    def momentum_deficit(self, du):
        """Theta = int u dU dA (per unit density)."""
        return np.sum((self.U - du) * du) * self.h**2

    def sample_global(self, X, Y, Z):
        """Deficit at global (X, Y, Z) (Z above ground) on the tilted marching planes, with the near field of the rotor
        (vortex cylinder; also upstream of the rotor) and, for a confined domain, the blockage offset."""
        X, Y, Z = (np.asarray(a, dtype=float) for a in (X, Y, Z))
        xs = X + Y * np.tan(self.yaw)
        pts = np.stack([xs.ravel(), (Z - self.H).ravel(), Y.ravel()], axis=1)
        out = self._interp(pts).reshape(X.shape)
        out[xs < 0] = 0.0
        R = self.Reff0
        if self.near_field:
            c = np.cos(self.yaw)
            rho = np.hypot(Y / (R * c), (Z - self.H) / R)
            T = near_field_table(EDGE, self.hub_depth, HUB_RADIUS, self._profile_norm)
            xi = xs / R
            u = T(np.stack([np.clip(xi, _NF_XI[0], _NF_XI[-1]).ravel(), np.clip(rho, 0, _NF_RHO[-1]).ravel()], -1)).reshape(X.shape)
            u = u - initial_profile(rho, EDGE, self.hub_depth, HUB_RADIUS, self._profile_norm) * (xs >= 0)   # correction to the
            # developed (marched) wake; upstream of the rotor the induction itself
            u = u * np.minimum(1.0, (_NF_XI[0] / np.minimum(xi, -1e-9)) ** 2)   # far upstream: decay ~ xi^-2 beyond the table
            u = u * np.minimum(1.0, (_NF_XI[-1] / np.maximum(xi, 1e-9)) ** 2)   # far downstream: the end effect decays ~ xi^-2
            u = u * np.minimum(1.0, (_NF_RHO[-1] / np.maximum(rho, 1e-9)) ** 3)  # far to the side: decay ~ rho^-3
            out = out + self._p0["dU0"] * u
        if self.domain_area:
            out = out - self.blockage(X)
        return out

    def blockage(self, X):
        """Uniform blockage deficit (negative: speed-up) of a confined domain at physical x, -Q(x)/A f(x), with Q the
        volume-flux deficit of the marched wake and f = (1 + x/sqrt(x^2 + R_w^2))/2 its rise through the rotor plane."""
        if not self.domain_area:
            return np.zeros_like(np.asarray(X, dtype=float))
        if not hasattr(self, "_Q"):
            self._Q = self.du_hist.sum(axis=(1, 2)) * self.h ** 2
        X = np.asarray(X, dtype=float); R = self.Reff0
        return np.interp(np.maximum(X, 0.0), self.xs, self._Q) / self.domain_area * 0.5 * (1 + X / np.sqrt(X ** 2 + R ** 2))
