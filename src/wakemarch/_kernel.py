"""Numerical kernel copied from the authors' research solver (internal module).

The recommended configuration is preserved. This module retains historical
research switches for provenance, not as a supported public API. Use WakeModel.
The cross-plane conservative advection form does not establish exact circulation
conservation with variable-coefficient diffusion. Force-based initialization is
an engineering closure; the discrete circulation need not equal its target.
Power-law background shear is included; a resolved ground boundary layer and
wake pressure gradients are not. See README.md and PROVENANCE.md.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator


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


class YawedWakeSolver:
    def __init__(self, D=126.0, H=90.0, U=8.0, CT=0.8, yaw_deg=25.0, TI=0.06,
                 Ly_D=8.0, ztop_D=3.5, h_D=1.0 / 30.0, sigma0_D=0.25, k_wake=None, ground=True,
                 gamma_scale=1.0, nu_omega_factor=1.0, theory="v3", entrainment=True, stretching=False, zbot_D=None, time_order=2, jacobian=True,
                 gamma_model="ref", nu_factor=1.0, freeze_vorticity=False, profile_q=0.0, shear_alpha=0.0, open_bc=True, thrust_ratio=None, swirl_CQ=0.0, swirl_sign=1.0, thrust_exp=None, hole_depth=0.0, hole_radius=0.25, hole_shape="gauss", hub_vortex_CQ=0.0, hub_vortex_sign=1.0, thrust_momentum=None, ct_cap=0.97, momentum_fraction=None, near_ramp=None, nu_near=None, edge_width=0.1, kappa_far=None, ground_mix=None, surface_visc=None):
        self.D, self.H, self.U, self.CT, self.TI = D, H, U, CT, TI
        self.yaw = np.radians(yaw_deg)
        self.k_wake = 0.3837 * TI + 0.003678 if k_wake is None else k_wake  # Niayifar-type growth rate
        self.sigma0_D = sigma0_D            # used by v1 only; v3 derives sigma0 = R_w/2
        self.gamma_scale, self.nu_omega_factor = gamma_scale, nu_omega_factor
        self.theory, self.ground = theory, ground
        self.entrainment, self.stretching = entrainment, stretching   # ablation switches (v3)
        self.jacobian = jacobian      # Jacobian factor exp(-dx div(v/u)) on omega advection (v3)
        self.gamma_model = gamma_model      # 'ref': Gamma0 = 0.5 D CT U sin(yaw) cos^2(yaw) (Shapiro 2018); 'impulse': lateral-impulse closed form
        self.shear_alpha = shear_alpha      # power-law background shear U(z) = U (z/H)^alpha (0 = uniform inflow); z floored at 0.05 H
        self.open_bc = open_bc              # open far field: Dirichlet data of psi and phi from a multipole expansion of the free-space (plus ground image) potential
        if thrust_ratio is None and thrust_exp is not None:
            thrust_ratio = float(np.cos(np.radians(yaw_deg)) ** thrust_exp)   # T_x = T_x0 cos^p(yaw) with T_x0 the unyawed thrust; p = 3 reproduces the constant-C_T default
        if thrust_momentum is None:
            thrust_momentum = thrust_ratio is not None          # default: when a thrust law is given, the initial momentum follows it as well as the circulation
        _final = thrust_momentum and thrust_ratio is not None            # 'final-model mode': a thrust law is given
        if momentum_fraction is None:
            momentum_fraction = 0.95 if _final else 1.0               # wake momentum / T_x downstream of the pressure recovery (LES: 0.91-0.95 at 3-5 D, tail-corrected)
        if near_ramp is None and _final:
            near_ramp = (0.5, 1.0)                                    # display of the first diameter (LES disk-mean deficit 0.53, 0.80, 0.92, 1 of its 1 D value at 0, 0.25, 0.5, 1 D); False switches it off
        if near_ramp is False:
            near_ramp = None
        self.edge_width = edge_width  # width of the smoothed edge of the initial wake mask in units of the wake radius (tanh)
        self.surface_visc = surface_visc  # (zc, fmin, on_deficit): surface-layer viscosity f(z) = max(fmin, min(1, z_abs/(zc D))) on the vorticity (and optionally the deficit) diffusivity (None: off)
        self.ground_mix = ground_mix  # (a, zr, on_deficit): vortex (and optionally deficit) diffusivity multiplied by 1 + a max(0, 1 - z_abs/(zr D)) near the ground (None: off)
        self.kappa_far = kappa_far  # (k1, xa, xb): vortex-diffusivity factor ramps linearly from nu_omega_factor at xa to k1 at xb diameters (None: constant)
        self.nu_near = nu_near      # (f0, x_c): eddy viscosity (deficit and vorticity) reduced to f0 at the rotor and growing linearly to 1 at x_c diameters (thin shear layers of the near wake); None: off
        self.near_ramp = near_ramp   # (g0, x_r): near-wake display ramp, deficit multiplied by g(x_s) = 1 - (1 - g0)(1 - x_s/x_r)^2 for x_s < x_r (x_s: distance from the tilted rotor plane, in D); the marching is not affected
        self.CT_eff = CT      # thrust coefficient on the normal velocity that carries the initial streamwise momentum
        if thrust_momentum and thrust_ratio is not None:   # momentum-consistent with the adopted thrust law: T_x = T_x0 * ratio, i.e. C_T,eff = C_T ratio / cos^3(yaw), limited by momentum theory (C_T,eff <= ct_cap)
            self.CT_eff = float(min(CT * thrust_ratio * momentum_fraction / np.cos(np.radians(yaw_deg)) ** 3, max(ct_cap, CT)))   # momentum_fraction: share of T_x that the wake carries downstream of the pressure recovery (LES: 0.91-0.95 at 3-5 D); the circulation is not scaled
        self.thrust_ratio = thrust_ratio    # None: constant C_T on the normal velocity (Gamma0 = Gamma_ref). Else the measured ratio Theta_yawed/Theta_unyawed of the streamwise momentum deficit: Gamma0 = Gamma_ref thrust_ratio / cos^3(yaw), i.e. 4 F_y/(pi rho D U) with the actual lateral force
        self.swirl_CQ = swirl_CQ            # wake-rotation torque coefficient C_Q = Q/(0.5 rho A U^2 R) = C_P/lambda (0: no swirl)
        self.swirl_sign = swirl_sign        # +1: wake rotates counter-clockwise seen from upstream (clockwise rotor); -1: opposite
        self.hole_shape = hole_shape      # 'gauss': f = 1 - h exp(-(rho/rho_h)^2); 'ramp': f = 1 - h (1 - smoothstep(rho/rho_h)), a hub-to-ring ramp of radius rho_h
        self.hole_depth, self.hole_radius = hole_depth, hole_radius   # hub hole of the initial deficit: f = 1 - depth exp(-(rho/hole_radius)^2), rho = distance from the axis in units of the wake radius; mean over the disc kept (same mass flux)
        self.hub_vortex_CQ, self.hub_vortex_sign = hub_vortex_CQ, hub_vortex_sign   # optional rotation module: hub vortex Gamma_c = 2.25 C_Q U D (C_Q = unyawed torque coefficient, scaled by cos^1.5 yaw), radius 0.11 D, compensating ring of opposite circulation at 0.55 D; sign +1: clockwise rotor seen from upstream
        self.profile_q = profile_q          # initial deficit shape (1-rho^2)^q (q = 0: top-hat); same mass flux and, via the radius, same thrust
        self.nu_factor = nu_factor          # multiplier of the deficit eddy viscosity (closure parameter, default 1)
        self.freeze_vorticity = freeze_vorticity   # ablation: vortices fixed at the rotor axis, no transport, no decay (as in the 2021 curled wake model)
        self.time_order = time_order  # 2: time-centred (predictor-corrector) induced velocity; 1: frozen at the start of the step
        h = h_D * D
        zbot = -H if ground else (-2.5 * D if zbot_D is None else -zbot_D * D)   # zbot_D: bottom of a no-ground domain
        self.y = np.arange(-Ly_D / 2 * D, Ly_D / 2 * D + h / 2, h)
        self.z = np.arange(zbot, ztop_D * D + h / 2, h)
        self.ny, self.nz, self.h = len(self.y), len(self.z), h
        self.Y, self.Z = np.meshgrid(self.y, self.z)  # shape (nz, ny)
        self._advection_grid = np.meshgrid(np.arange(self.ny, dtype=float),
                                          np.arange(self.nz, dtype=float))
        nyi, nzi = self.ny - 2, self.nz - 2
        self._poisson = splu(laplacian_2d(nyi, nzi, h, h).tocsc())
        self._L_int = laplacian_2d(nyi, nzi, h, h).tocsr()
        self._ivec = sp.identity(nyi * nzi, format="csc")
        # entrainment potential: Dirichlet far field, Neumann (impermeable) ground
        if ground:
            self._nzphi, self._phi_z0 = self.nz - 1, 0   # rows 0..nz-2, row 0 is the ground
            Lz = lap1d_neumann_first(self._nzphi, h)
        else:
            self._nzphi, self._phi_z0 = self.nz - 2, 1
            Lz = lap1d(self._nzphi, h)
        zabs = np.maximum(self.Z + H, 0.05 * H)
        self.Ubg = U * (zabs / H) ** shear_alpha
        self.dUbg_dz = shear_alpha * self.Ubg / zabs * (self.Z + H > 0.05 * H)
        # deficit diffusion: zero-flux (Neumann) ground, zero Dirichlet elsewhere; omega keeps Dirichlet (image vortex is antisymmetric)
        if ground:
            self._L_du = (sp.kron(sp.eye(self.nz - 1), lap1d(nyi, h)) + sp.kron(lap1d_neumann_first(self.nz - 1, h), sp.eye(nyi))).tocsr()
        if self.open_bc:
            self._build_far_field()
        self._phi_solver = splu((sp.kron(sp.eye(self._nzphi), lap1d(nyi, h)) + sp.kron(Lz, sp.eye(nyi))).tocsc())

    @classmethod
    def recommended(cls, D, H, U, CT, yaw_deg, TI, shear_alpha=0.0, **kw):
        """Minimal-parameter configuration of the methods report: inputs D, H, U (hub speed), C_T (unyawed thrust coefficient), yaw, TI, shear exponent. Constants: growth rate
        k_w = 0.3837 TI + 0.003678, thrust-yaw law T_x = T_x0 cos^2(yaw) with Gamma_0 = 4 F_y / (pi rho D U), vortex diffusivity kappa_omega = 0.5 nu (trade-off: 0.25 for the kidney shape, 1 for the yaw-steering gain)."""
        kw.setdefault("thrust_exp", 2.0); kw.setdefault("nu_omega_factor", 0.5)          # defaults of the final model, overridable for sensitivity studies
        return cls(D=D, H=H, U=U, CT=CT, yaw_deg=yaw_deg, TI=TI, shear_alpha=shear_alpha, **kw)

    # ---- initial state -----------------------------------------------------
    def initial_state_params(self):
        """Closed-form initial-state parameters of the accepted engineering model (theory.tex, 'Near-wake initial state')."""
        U, CT, g = self.U, self.CT_eff, self.yaw
        c, sn = np.cos(g), np.sin(g)
        s = np.sqrt(1 - CT)
        A = np.pi * self.D**2 / 4
        dU0 = (1 - s) * U * c**2                  # streamwise deficit
        uw0 = U - dU0                             # wake speed
        mdot = 0.5 * A * U * c * (1 + s)          # mass flux / rho through the yawed disk
        Aw = mdot / uw0                           # cross-plane wake area
        Rw = np.sqrt(Aw / (np.pi * c))            # ellipse semi-axes (Rw c, Rw)
        vj = (1 - s) * U * sn * c                 # lateral jump of wake fluid
        Gamma_imp = vj * 2 * Rw * c               # lateral-impulse closed form
        Gamma_ref = 0.5 * self.D * U * self.CT * sn * c**2   # Shapiro (2018) / curl-model value
        gscale = self.gamma_scale * (Gamma_ref / Gamma_imp if (self.gamma_model == "ref" and abs(Gamma_imp) > 1e-12) else 1.0)
        if self.thrust_ratio is not None and self.gamma_model == "ref":
            gscale *= self.thrust_ratio / c**3
        Gamma0 = gscale * Gamma_imp
        Fy = 0.5 * A * U**2 * CT * c**2 * sn      # lateral force / rho
        Tx = 0.5 * A * U**2 * CT * c**3           # streamwise thrust / rho
        return dict(s=s, dU0=dU0, uw0=uw0, Aw=Aw, Rw=Rw, vj=vj, Gamma0=Gamma0, Gamma0_impulse=Gamma_imp, Gamma_ref=Gamma_ref, gscale=gscale, Fy=Fy, Tx=Tx, mdot=mdot)

    def circulation_target(self):
        if self.theory == "v1":
            return self.gamma_scale * 0.5 * self.CT * self.U * self.D * np.sin(self.yaw) * np.cos(self.yaw) ** 2
        return self.initial_state_params()["Gamma0"]

    def initialize(self):
        return self._initialize_v1() if self.theory == "v1" else self._initialize_v3()

    def _initialize_v1(self):
        R = self.D / 2
        r = np.hypot(self.Y / (R * np.cos(self.yaw)), self.Z / R)
        mask = 0.5 * (1 - np.tanh((r - 1.0) / 0.1))
        a = 0.5 * (1 - np.sqrt(1 - self.CT))
        du = 2 * a * self.U * mask
        iz0 = np.argmin(np.abs(self.z))
        width = np.sum(mask[iz0]) * self.h
        A = self.circulation_target() / width
        v0 = -A * mask
        omega = -np.gradient(v0, self.h, axis=0)
        self.sigma0 = self.sigma0_D * self.D
        return du, omega

    def _initialize_v3(self):
        p = self.initial_state_params()
        c = np.cos(self.yaw)
        d0 = p["dU0"]

        def build(R):
            r = np.hypot(self.Y / (R * c), self.Z / R)
            return 0.5 * (1 - np.tanh((r - 1.0) / self.edge_width))

        q = self.profile_q

        def prof(R):
            m = build(R)
            rho2 = (self.Y / (R * c)) ** 2 + (self.Z / R) ** 2
            if self.hole_depth > 0.0:
                if self.hole_shape == "ramp":
                    t = np.clip(np.sqrt(rho2) / self.hole_radius, 0.0, 1.0); hf = 1.0 - (3 * t**2 - 2 * t**3)
                else:
                    hf = np.exp(-rho2 / self.hole_radius**2)
                fm = m * (1.0 - self.hole_depth * hf)
                return fm * (m.sum() / fm.sum())          # mean over the disc equals the top-hat value
            if q <= 0:
                return m
            return m * (q + 1.0) * np.clip(1.0 - rho2, 0.0, None) ** q   # mean over the disc equals the top-hat value

        def theta(R):
            m = prof(R)
            return np.sum((self.U - d0 * m) * d0 * m) * self.h**2

        # the smoothed mask edge adds momentum: adjust the mask radius (well conditioned) so the discrete Theta0 equals Tx
        from scipy.optimize import brentq
        Reff = brentq(lambda R: theta(R) - p["Tx"], 0.7 * p["Rw"], 1.3 * p["Rw"])
        self.Reff0 = Reff
        mask = build(Reff)
        du = d0 * prof(Reff)
        omega = p["gscale"] * p["vj"] * np.gradient(mask, self.h, axis=0)   # omega = -dv/dz, v = -vj*mask
        if self.swirl_CQ != 0.0:   # solid-body wake rotation from the rotor torque: x-component of the angular momentum flux = Q cos(yaw)
            R = self.D / 2
            Qx = self.swirl_CQ * 0.5 * (np.pi * R**2) * self.U**2 * R * c
            a_, b_ = Reff * c, Reff
            Om = Qx / (p["uw0"] * np.pi * a_ * b_ * (a_**2 + b_**2) / 4)
            omega = omega - self.swirl_sign * 2.0 * Om * mask
            self.swirl_Omega = Om
        if self.hub_vortex_CQ > 0.0:     # hub vortex of the rotor rotation (LES: sigma_c = 0.11 D, Gamma_c = 2.25 C_Q U D, constant to 3 % over yaw 0-40), with the compensating tip-radius ring
            Gc = self.hub_vortex_sign * 2.25 * self.hub_vortex_CQ * np.cos(self.yaw) ** 1.5 * self.U * self.D
            r2 = self.Y**2 + self.Z**2; sc = 0.11 * self.D
            gring = np.exp(-((np.sqrt(r2) - 0.55 * self.D) ** 2) / (2 * (0.05 * self.D) ** 2))
            omega = omega + Gc / (2 * np.pi * sc**2) * np.exp(-r2 / (2 * sc**2)) - Gc * gring / (gring.sum() * self.h**2)
        self.sigma0 = Reff / 2
        self._p0 = p
        return du, omega

    # ---- open far field ----------------------------------------------------
    NMULT = 14

    def _build_far_field(self):
        ny, nz = self.ny, self.nz
        # psi: boundary nodes (left/right columns, top row, bottom row when there is no ground)
        mask = np.zeros((nz, ny), bool)
        mask[:, 0] = mask[:, -1] = True; mask[-1, :] = True
        if not self.ground:
            mask[0, :] = True
        self._psi_bmask = mask
        zt = (self.Y + 1j * self.Z)[mask]
        self._psi_zt = zt
        # phi: columns 0 and ny-1 (rows z0..nz-2), top row nz-1, bottom row 0 when there is no ground
        pm = np.zeros((nz, ny), bool)
        z0 = self._phi_z0
        pm[z0:nz - 1, 0] = pm[z0:nz - 1, -1] = True; pm[-1, :] = True
        if not self.ground:
            pm[0, :] = True
        self._phi_bmask = pm
        self._phi_zt = (self.Y + 1j * self.Z)[pm]
        self._zs = self.Y + 1j * self.Z
        self._zs_img = self.Y + 1j * (-2 * self.H - self.Z)
        w = np.zeros((nz, ny)); w[1:-1, 1:-1] = self.h**2
        self._w_psi = w
        wp = np.zeros((nz, ny)); wp[z0:nz - 1, 1:-1] = self.h**2
        if self.ground:
            wp[0, 1:-1] = 0.5 * self.h**2
        self._w_phi = wp
        # Geometry is fixed throughout marching. Preserve the original repeated
        # multiplication and reduction order (no reassociation of the moments).
        self._multipole_powers = []
        zk = np.ones_like(self._zs.ravel())
        zik = np.ones_like(self._zs_img.ravel())
        for _ in range(self.NMULT):
            zk = zk * self._zs.ravel()
            if self.ground:
                zik = zik * self._zs_img.ravel()
            self._multipole_powers.append((zk, zik))
        self._multipole_targets = {}
        for targets in (self._psi_zt, self._phi_zt):
            inv = 1.0 / targets
            ik = np.ones_like(targets)
            powers = []
            for _ in range(self.NMULT):
                ik = ik * inv
                powers.append(ik)
            self._multipole_targets[id(targets)] = (powers, np.log(np.abs(targets)))

    def _far_field(self, f, w, zt, mask, even_image):
        """Dirichlet data at the boundary targets for lap(u) = -f in free space (plus the ground image: odd for psi, even for phi):
        u = -(1/2 pi) S,  S = sum_s w f [ln|zt - zs| + sg ln|zt - zs_image|],  expanded in multipoles about the rotor axis:
        ln|zt - zs| = ln|zt| - Re sum_k (zs/zt)^k / k."""
        K = self.NMULT
        out = np.zeros((self.nz, self.ny))
        src = (w * f).ravel()
        sg = (1.0 if even_image else -1.0) if self.ground else 0.0
        powers, log_radius = self._multipole_targets[id(zt)]
        val = np.zeros(len(zt), dtype=complex)
        for k in range(1, K + 1):
            zk, zik = self._multipole_powers[k - 1]
            ik = powers[k - 1]
            m = (src * zk).sum()
            if self.ground:
                m = m + sg * (src * zik).sum()
            val += m * ik / k
        S = src.sum() * (1.0 + sg) * log_radius - val.real
        out[mask] = -S / (2 * np.pi)
        return out

    # ---- operators ---------------------------------------------------------
    def solve_psi(self, omega):
        rhs = -omega[1:-1, 1:-1]
        psi = np.zeros_like(omega)
        if self.open_bc:
            psi = self._far_field(omega, self._w_psi, self._psi_zt, self._psi_bmask, even_image=False)
            B = np.zeros((self.nz - 2, self.ny - 2))
            B[:, 0] += psi[1:-1, 0]; B[:, -1] += psi[1:-1, -1]; B[0, :] += psi[0, 1:-1]; B[-1, :] += psi[-1, 1:-1]
            rhs = rhs - B / self.h**2
        psi[1:-1, 1:-1] = self._poisson.solve(rhs.ravel()).reshape(self.nz - 2, self.ny - 2)
        return psi

    def solve_phi(self, dudx):
        """lap(phi) = -u_x (continuity); Dirichlet far field, Neumann (impermeable) ground."""
        z0 = self._phi_z0
        rhs = -dudx[z0:self.nz - 1, 1:-1]
        phi = np.zeros((self.nz, self.ny))
        if self.open_bc:
            phi = self._far_field(dudx, self._w_phi, self._phi_zt, self._phi_bmask, even_image=True)
            B = np.zeros((self._nzphi, self.ny - 2))
            B[:, 0] += phi[z0:self.nz - 1, 0]; B[:, -1] += phi[z0:self.nz - 1, -1]; B[-1, :] += phi[self.nz - 1, 1:-1]
            if not self.ground:
                B[0, :] += phi[0, 1:-1]
            rhs = rhs - B / self.h**2
        sol = self._phi_solver.solve(rhs.ravel()).reshape(self._nzphi, self.ny - 2)
        phi[z0:self.nz - 1, 1:-1] = sol
        return phi

    def velocities(self, psi):
        dpsi_dz, dpsi_dy = np.gradient(psi, self.h, self.h, edge_order=2)
        return dpsi_dz, -dpsi_dy  # v, w (solenoidal part)

    def vortex_velocity(self, om, dt):
        """Solenoidal velocity induced by om, time-centred over a step dt (scalar or array) when time_order == 2."""
        v, w = self.velocities(self.solve_psi(om))
        if self.time_order == 2:
            om_half = self.advect([om], v, w, 0.5 * dt)[0]
            v, w = self.velocities(self.solve_psi(om_half))
        return v, w

    def _sample(self, f, iy, iz, mirror_ground=False):
        if mirror_ground:   # deficit: reflect about the ground (zero flux), constant zero elsewhere
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
        return [self._sample(f, iy_src, iz_src, mirror_ground=(mirror_first and self.ground and i == 0)) for i, f in enumerate(fields)]

    def diffuse(self, f, nu, dt, neumann_ground=False, fast=True):
        """Backward Euler on interior nodes, zero Dirichlet outside; coefficient dt*nu scalar or array.
        neumann_ground: zero normal flux at the bottom row (used for the deficit when a ground is present).
        fast: reuse a cached factorisation of (I - cbar L) with iterative refinement towards the exact variable-coefficient system
        (relative residual 1e-9). After a stalled refinement, reuse the exact
        factorization as the next preconditioner; refresh when the mean
        coefficient moves by 12 per cent or refinement stalls again."""
        ng = bool(neumann_ground and self.ground)
        if ng:
            c = np.broadcast_to(dt * nu, f.shape)[:-1, 1:-1]; rhs = f[:-1, 1:-1].ravel(); L = self._L_du; shp = (self.nz - 1, self.ny - 2)
        else:
            c = np.broadcast_to(dt * nu, f.shape)[1:-1, 1:-1]; rhs = f[1:-1, 1:-1].ravel(); L = self._L_int; shp = (self.nz - 2, self.ny - 2)
        cv = np.ascontiguousarray(c).ravel(); n = len(cv)
        out = np.zeros_like(f)

        def exact_factor():
            A = (sp.identity(n, format="csc") - sp.diags(cv) @ L).tocsc()
            return splu(A)

        if not fast:
            sol = exact_factor().solve(rhs)
        else:
            cbar = float(cv.mean())
            if cbar <= 0.0:                      # no diffusion (e.g. kappa_omega = 0): identity
                out_ = np.zeros_like(f)
                if ng:
                    out_[:-1, 1:-1] = rhs.reshape(shp)
                else:
                    out_[1:-1, 1:-1] = rhs.reshape(shp)
                return out_
            cache = self.__dict__.setdefault("_dcache", {})
            ent = cache.get(ng)
            if ent is None or ent[0] <= 0.0 or abs(cbar / ent[0] - 1.0) > 0.12:
                ent = (cbar, splu((sp.identity(n, format="csc") - cbar * L).tocsc()))
                cache[ng] = ent
            cb, lu = ent
            x = lu.solve(rhs); bn = np.linalg.norm(rhs) + 1e-300
            ok = False
            for _ in range(14):
                r = rhs - (x - cv * (L @ x))
                if np.linalg.norm(r) < 1e-9 * bn:
                    ok = True; break
                x = x + lu.solve(r)
            if not ok:    # keep the factorization we already paid for
                lu = exact_factor()
                x = lu.solve(rhs)
                cache[ng] = (cbar, lu)
            sol = x
        if ng:
            out[:-1, 1:-1] = sol.reshape(shp)
        else:
            out[1:-1, 1:-1] = sol.reshape(shp)
        return out

    def laplacian(self, f, neumann_ground=False):
        out = np.zeros_like(f)
        out[1:-1, 1:-1] = (f[1:-1, 2:] + f[1:-1, :-2] + f[2:, 1:-1] + f[:-2, 1:-1] - 4 * f[1:-1, 1:-1]) / self.h**2
        if neumann_ground and self.ground:
            out[0, 1:-1] = (f[0, 2:] + f[0, :-2] + 2 * f[1, 1:-1] - 4 * f[0, 1:-1]) / self.h**2
        return out

    # ---- marching ----------------------------------------------------------
    def mix_field(self):
        """Near-ground enhancement of the diffusivity of `ground_mix` on the grid (1 where off)."""
        f = 1.0
        if self.ground_mix is not None:
            a, zr = self.ground_mix[0], self.ground_mix[1]
            f = f * (1.0 + a * np.maximum(0.0, 1.0 - np.maximum(self.Z + self.H, 0.0) / (zr * self.D)))
        if self.surface_visc is not None:
            zc, fmin = self.surface_visc[0], self.surface_visc[1]
            f = f * np.maximum(fmin, np.minimum(1.0, np.maximum(self.Z + self.H, 0.0) / (zc * self.D)))
        return f

    def mix_deficit(self):
        """Same field for the deficit diffusivity when the options ask for it."""
        on = (self.ground_mix is not None and self.ground_mix[2]) or (self.surface_visc is not None and self.surface_visc[2])
        return self.mix_field() if on else 1.0

    def kappa_at(self, x):
        """Vortex-diffusivity factor at distance x (m): constant nu_omega_factor, or the linear ramp of `kappa_far`."""
        if self.kappa_far is None:
            return self.nu_omega_factor
        k1, xa, xb = self.kappa_far
        return self.nu_omega_factor + (k1 - self.nu_omega_factor) * min(1.0, max(0.0, (x / self.D - xa) / (xb - xa)))

    def solve(self, max_dist_D=10.0, dx_D=0.1, store_every_D=0.25, diffusion=True, init_fields=None, progress_callback=None):
        buffer = (3.0 * np.tan(abs(self.yaw)) + 1.0) * self.D  # room for the shear-mapped reconstruction
        L = max_dist_D * self.D + buffer
        if progress_callback is not None:
            progress_callback(0.0, L / self.D)
        du, om = self.initialize()
        if init_fields is not None:   # (du, om, sigma0) override, used by verification tests
            du, om, self.sigma0 = init_fields[0].copy(), init_fields[1].copy(), init_fields[2]
        xs, dus, oms, dudxs = [0.0], [du.copy()], [om.copy()], [np.zeros_like(du)]
        x, dx = 0.0, dx_D * self.D
        dudx = np.zeros_like(du)
        while x < L:
            if self.theory == "v1":
                vs, ws = self.velocities(self.solve_psi(om))
            else:
                u0 = np.maximum(self.Ubg - du, 0.3 * self.Ubg)
                vs, ws = self.vortex_velocity(om, dx / u0)
            if self.theory == "v1":
                u_loc = self.U * np.cos(self.yaw)
                dt = dx / u_loc
                du, om = self.advect([du, om], vs, ws, dt)
                if diffusion:
                    nu = self.k_wake * u_loc * (self.sigma0 + self.k_wake * x)
                    du, om = self.diffuse(du, nu, dt), self.diffuse(om, self.nu_omega_factor * nu, dt)
            else:
                u = np.maximum(self.Ubg - du, 0.3 * self.Ubg)
                sigma = self.sigma0 + self.k_wake * x
                sum_du = du.sum()
                u_bar = self.U - (du**2).sum() / sum_du if sum_du > 0 else self.U   # deficit-weighted wake speed
                nu = self.k_wake * sigma * u_bar if diffusion else 0.0               # uniform cross-plane eddy viscosity
                if self.nu_near is not None and diffusion:
                    f0, xc = self.nu_near; nu = nu * (f0 + (1.0 - f0) * min(1.0, x / (xc * self.D)))
                uz, uy = np.gradient(u, self.h, self.h, edge_order=2)
                lap_u = self.laplacian(u - self.Ubg, neumann_ground=True)
                v, w = vs, ws
                for _ in range(3 if self.entrainment else 1):          # u_x depends on (v,w), which depend on u_x via phi
                    dudx = (self.nu_factor * nu * lap_u - (v * uy + w * uz)) / u
                    if self.entrainment:
                        gz, gy = np.gradient(self.solve_phi(dudx), self.h, self.h, edge_order=2)
                        v, w = vs + gy, ws + gz
                        if self.ground:
                            w[0, :] = 0.0
                dt = dx / u
                if self.freeze_vorticity:
                    du = self.advect([du], v, w, dt, mirror_first=True)[0]
                else:
                    du, om = self.advect([du, om], v, w, dt, mirror_first=True)
                if self.shear_alpha != 0.0:
                    du = du + dt * w * self.dUbg_dz        # vertical advection of the background shear by the induced flow
                if self.jacobian and not self.freeze_vorticity:   # approximate conservative advection update; not an exact discrete integral guarantee
                    div = np.gradient(v / u, self.h, axis=1, edge_order=2) + np.gradient(w / u, self.h, axis=0, edge_order=2)
                    om = om * np.exp(-dx * div)
                if self.stretching and not self.freeze_vorticity:
                    om = om * np.exp(dx * dudx / u)
                if diffusion:
                    du = self.diffuse(du, self.nu_factor * nu / u * self.mix_deficit(), dx, neumann_ground=True)
                    if not self.freeze_vorticity:
                        om = self.diffuse(om, self.kappa_at(x) * nu / u * self.mix_field(), dx)
            x += dx
            if progress_callback is not None:
                progress_callback(min(x / self.D, L / self.D), L / self.D)
            if x - xs[-1] >= store_every_D * self.D - 1e-9:
                xs.append(x); dus.append(du.copy()); oms.append(om.copy()); dudxs.append(dudx.copy())
        self.xs = np.array(xs)
        self.du_hist = np.stack(dus)
        self.om_hist = np.stack(oms)
        self.dudx_hist = np.stack(dudxs)
        self._interp = RegularGridInterpolator((self.xs, self.z, self.y), self.du_hist,
                                               bounds_error=False, fill_value=0.0)

    # ---- cross-plane velocities ---------------------------------------------
    def build_velocity_history(self):
        """Cross-plane velocities (v lateral, w vertical) of every stored plane: solenoidal part from omega plus the entrainment flow."""
        vs_, ws_ = [], []
        for om, dudx in zip(self.om_hist, self.dudx_hist):
            v, w = self.velocities(self.solve_psi(om))
            if self.entrainment:
                gz, gy = np.gradient(self.solve_phi(dudx), self.h, self.h, edge_order=2)
                v, w = v + gy, w + gz
                if self.ground:
                    w[0, :] = 0.0
            vs_.append(v); ws_.append(w)
        self.v_hist, self.w_hist = np.stack(vs_), np.stack(ws_)
        self._vint = RegularGridInterpolator((self.xs, self.z, self.y), self.v_hist, bounds_error=False, fill_value=0.0)
        self._wint = RegularGridInterpolator((self.xs, self.z, self.y), self.w_hist, bounds_error=False, fill_value=0.0)

    def sample_velocity(self, X, Y, Z):
        """(v, w) at global (X, Y, Z) with the same shear-mapped reconstruction as the deficit."""
        if not hasattr(self, "_vint"):
            self.build_velocity_history()
        xs = X + Y * np.tan(self.yaw)
        pts = np.stack([xs.ravel(), (Z - self.H).ravel(), Y.ravel()], axis=1)
        v, w = self._vint(pts).reshape(X.shape), self._wint(pts).reshape(X.shape)
        v[xs < 0] = 0.0; w[xs < 0] = 0.0
        return v, w

    # ---- diagnostics / reconstruction -------------------------------------
    def circulation_top(self, om):
        """Circulation of the vortex above hub height (z>0) and below (z<0)."""
        up, dn = self.Z > 0, self.Z < 0
        return np.sum(om[up]) * self.h**2, np.sum(om[dn]) * self.h**2

    def momentum_deficit(self, du):
        """Theta = int u dU dA  (per unit density)."""
        return np.sum((self.U - du) * du) * self.h**2

    def sample_global(self, X, Y, Z):
        """Shear-mapped reconstruction onto global (X, Y, Z); Z is height above ground."""
        xs = X + Y * np.tan(self.yaw)
        pts = np.stack([xs.ravel(), (Z - self.H).ravel(), Y.ravel()], axis=1)
        out = self._interp(pts).reshape(X.shape)
        out[xs < 0] = 0.0
        if self.near_ramp is not None:      # pressure recovery and wake expansion of the first diameter: the developed state is reached at x_r
            g0, xr = self.near_ramp; t = np.clip(xs / (xr * self.D), 0.0, 1.0)
            out = out * (1.0 - (1.0 - g0) * (1.0 - t) ** 2)
        return out
