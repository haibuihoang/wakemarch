"""Fast public API tests. Run on the installed wheel: python -I tests/test_api.py."""
from dataclasses import FrozenInstanceError
from contextlib import redirect_stderr
import io
import unittest
from unittest.mock import patch

import numpy as np

from wakemarch import Numerics, WakeModel

PARAMS = dict(D=126, H=90, U=8, CT=0.8, yaw_deg=25, TI=0.06)


class InputTests(unittest.TestCase):
    def test_invalid_physical_inputs(self):
        for key, value in [("D", 0), ("H", -1), ("H", 40), ("U", 0), ("CT", 1), ("CT", 0),
                           ("yaw_deg", 90), ("yaw_deg", -90), ("yaw_deg", 50.5), ("yaw_deg", -50.5),
                           ("TI", 6), ("TI", -0.1),
                           ("shear_alpha", -1), ("U", np.nan), ("D", np.inf),
                           ("CT", [0.8]), ("D", True), ("U", "8")]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                WakeModel(**(PARAMS | {key: value}))

    def test_supported_limits_construct(self):
        for key, value in [("yaw_deg", 50), ("yaw_deg", -50), ("H", 63)]:
            with self.subTest(key=key, value=value):
                WakeModel(**(PARAMS | {key: value}))

    def test_invalid_numerics(self):
        for key in ("h_D", "dx_D", "store_every_D", "Ly_D", "ztop_D"):
            for value in (0, -1, np.nan, np.inf):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    Numerics(**{key: value})
        with self.assertRaises(ValueError):
            Numerics(h_D=2)
        with self.assertRaises(TypeError):
            WakeModel(**PARAMS, numerics={})

    def test_settings_are_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            Numerics().dx_D = 0
        with self.assertRaises(FrozenInstanceError):
            WakeModel(**PARAMS).yaw_deg = 0

    def test_bad_solve_extent(self):
        model = WakeModel(**PARAMS)
        for distance in (0, -1, np.nan, 0.05):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                model.solve(distance)


class ResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = WakeModel(**PARAMS, shear_alpha=0.1,
                              numerics=Numerics(h_D=0.1))
        cls.result = cls.model.solve(1)

    def test_scalar_list_and_broadcast_sampling(self):
        r = self.result
        scalar = r.sample_deficit(126, 0, 90)
        self.assertIsInstance(scalar, float)
        result = r.sample_deficit([[126], [100]], [0, 10, 20], 90)
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result[0, 0], scalar)

    def test_velocity_matches_background_minus_deficit(self):
        r = self.result
        z = np.array([60, 90, 120])
        actual = r.sample_streamwise_velocity(126, 0, z)
        expected = 8 * (z / 90) ** 0.1 - r.sample_deficit(126, 0, z)
        np.testing.assert_array_equal(actual, expected)

    def test_upstream_is_zero(self):
        self.assertEqual(self.result.sample_deficit(-126, 0, 90), 0)
        self.assertEqual(self.result.sample_streamwise_velocity(-126, 0, 90), 8)

    def test_default_bounds_error_and_explicit_extension(self):
        for x, y, z in [(1e6, 0, 90), (126, 1e6, 90), (126, 0, 1e6)]:
            with self.subTest(x=x, y=y, z=z), self.assertRaises(ValueError):
                self.result.sample_deficit(x, y, z)
            self.assertEqual(self.result.sample_deficit(x, y, z, bounds="zero"), 0)

    def test_bad_sampling_inputs(self):
        for x, y, z in [(np.nan, 0, 90), (126, np.inf, 90), (126, 0, -1)]:
            with self.subTest(x=x, y=y, z=z), self.assertRaises(ValueError):
                self.result.sample_deficit(x, y, z)
        with self.assertRaises(ValueError):
            self.result.sample_deficit(126, 0, 90, bounds="ignore")

    def test_history_shapes_coordinates_and_readonly(self):
        r = self.result
        self.assertEqual(r.marching_deficit.shape, (len(r.x), len(r.z), len(r.y)))
        np.testing.assert_array_equal(r.z, r.z_relative + 90)
        self.assertAlmostEqual(r.z[0], 0)
        np.testing.assert_allclose(np.diff(r.x) / 126, 0.3, rtol=0, atol=1e-13)
        for a in (r.x, r.y, r.z, r.marching_deficit, r.streamwise_vorticity,
                  r.streamwise_velocity_derivative):
            self.assertFalse(a.flags.writeable)

    def test_repeated_solve_keeps_old_result(self):
        before = self.result.marching_deficit.copy()
        later = self.model.solve(0.6)
        self.assertIsNot(later, self.result)
        np.testing.assert_array_equal(before, self.result.marching_deficit)

    def test_near_ramp_is_applied(self):
        r = self.result
        iy = np.argmin(np.abs(r.y))
        iz = np.argmin(np.abs(r.z_relative))
        # Map exactly to x_s=0 despite floating-point placement of the y mesh.
        x = -r.y[iy] * np.tan(np.radians(self.model.yaw_deg))
        self.assertAlmostEqual(r.sample_deficit(x, r.y[iy], r.z[iz]),
                               0.5 * r.marching_deficit[0, iz, iy], places=11)


class CrossVelocityTests(unittest.TestCase):
    """Public cross-plane velocities (v, w): sampling, bounds, kernel agreement, symmetry, sign."""

    @classmethod
    def setUpClass(cls):
        numerics = Numerics(h_D=0.1)
        cls.straight = WakeModel(**(PARAMS | {"yaw_deg": 0}), numerics=numerics).solve(3, progress=False)
        cls.yawed = WakeModel(**PARAMS, numerics=numerics).solve(3, progress=False)
        cls.mirrored = WakeModel(**(PARAMS | {"yaw_deg": -25}), numerics=numerics).solve(3, progress=False)

    def test_scalar_list_and_broadcast_sampling(self):
        r = self.yawed
        v, w = r.sample_cross_velocity(252, 0, 90)
        self.assertIsInstance(v, float)
        self.assertIsInstance(w, float)
        v2, w2 = r.sample_cross_velocity([[252], [200]], [0, 10, 20], 90)
        self.assertEqual(v2.shape, (2, 3))
        self.assertEqual(w2.shape, (2, 3))
        self.assertEqual((v2[0, 0], w2[0, 0]), (v, w))

    def test_bounds_upstream_and_bad_inputs(self):
        r = self.yawed
        self.assertEqual(r.sample_cross_velocity(-126, 0, 90), (0.0, 0.0))
        for x, y, z in [(1e6, 0, 90), (126, 1e6, 90), (126, 0, 1e6)]:
            with self.subTest(x=x, y=y, z=z):
                with self.assertRaises(ValueError):
                    r.sample_cross_velocity(x, y, z)
                self.assertEqual(r.sample_cross_velocity(x, y, z, bounds="zero"), (0.0, 0.0))
        for x, y, z in [(np.nan, 0, 90), (126, np.inf, 90), (126, 0, -1)]:
            with self.subTest(x=x, y=y, z=z), self.assertRaises(ValueError):
                r.sample_cross_velocity(x, y, z)
        with self.assertRaises(ValueError):
            r.sample_cross_velocity(126, 0, 90, bounds="ignore")

    def test_matches_kernel_fields_at_grid_nodes(self):
        r = self.straight            # yaw 0: physical x equals marching x_s
        k = len(r.x) // 2
        # The top row is left out: z + H - H can round just above the stored top.
        zz, yy = np.meshgrid(r.z[:-1], r.y, indexing="ij")
        v, w = r.sample_cross_velocity(r.x[k], yy, zz)
        s = r._solver
        np.testing.assert_allclose(v, s.v_hist[k, :-1], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(w, s.w_hist[k, :-1], rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(w[0], 0.0)          # no flow through the ground
        self.assertGreater(np.abs(v).max(), 1e-3)         # the field is not trivially zero

    def test_symmetry_without_yaw_and_mirror_with_yaw(self):
        y = np.linspace(-150, 150, 13)
        z = np.array([[30.], [90.], [150.]])
        v, w = self.straight.sample_cross_velocity(252, y, z)
        vm, wm = self.straight.sample_cross_velocity(252, -y, z)
        np.testing.assert_allclose(v, -vm, rtol=0, atol=1e-12)
        np.testing.assert_allclose(w, wm, rtol=0, atol=1e-12)
        v, w = self.yawed.sample_cross_velocity(252, y, z)
        vm, wm = self.mirrored.sample_cross_velocity(252, -y, z)
        np.testing.assert_allclose(v, -vm, rtol=0, atol=1e-10)
        np.testing.assert_allclose(w, wm, rtol=0, atol=1e-10)

    def test_positive_yaw_pushes_wake_core_towards_negative_y(self):
        v, _ = self.yawed.sample_cross_velocity(252, 0, 90)
        self.assertLess(v, -0.02 * 8)

    def test_no_side_effect_on_deficit(self):
        model = WakeModel(**PARAMS, numerics=Numerics(h_D=0.1))
        r = model.solve(1, progress=False)
        before = (r.marching_deficit.copy(), r.sample_deficit(126, [0, 20], 90))
        r.sample_cross_velocity(126, 0, 90)
        np.testing.assert_array_equal(r.marching_deficit, before[0])
        np.testing.assert_array_equal(r.sample_deficit(126, [0, 20], 90), before[1])


class ProgressTests(unittest.TestCase):
    def test_progress_is_monotonic_and_preserves_arrays(self):
        model = WakeModel(**PARAMS, numerics=Numerics(h_D=0.1))
        quiet = io.StringIO()
        with redirect_stderr(quiet):
            reference = model.solve(0.6, progress=False)
        self.assertEqual(quiet.getvalue(), '')
        events = []
        reported = model.solve(0.6, progress=lambda completed, total: events.append((completed, total)))
        self.assertEqual(events[0][0], 0)
        self.assertEqual(events[-1][0], events[-1][1])
        self.assertGreater(events[-1][1], 0.6)
        self.assertTrue(all(a[0] <= b[0] for a, b in zip(events, events[1:])))
        self.assertTrue(all(total == events[0][1] for _, total in events))
        for name in ('marching_deficit', 'streamwise_vorticity', 'streamwise_velocity_derivative'):
            np.testing.assert_array_equal(getattr(reported, name), getattr(reference, name))

    def test_default_bar_finishes(self):
        output = io.StringIO()
        with redirect_stderr(output):
            WakeModel(**PARAMS, numerics=Numerics(h_D=0.1)).solve(0.6)
        self.assertIn('100%', output.getvalue())
        self.assertIn('Complete', output.getvalue())

    def test_bar_reports_failure_without_false_success(self):
        output = io.StringIO()
        with patch('wakemarch.model.YawedWakeSolver.recommended', side_effect=RuntimeError('test failure')):
            with redirect_stderr(output), self.assertRaisesRegex(RuntimeError, 'test failure'):
                WakeModel(**PARAMS).solve(0.6)
        self.assertIn('Stopped before completion', output.getvalue())
        self.assertNotIn('Complete', output.getvalue())

    def test_callback_errors_propagate_and_bad_progress_is_rejected(self):
        def fail(completed, total):
            raise RuntimeError('observer failure')
        model = WakeModel(**PARAMS, numerics=Numerics(h_D=0.1))
        with self.assertRaisesRegex(RuntimeError, 'observer failure'):
            model.solve(0.6, progress=fail)
        with self.assertRaises(TypeError):
            model.solve(0.6, progress='yes')


class DiffusionTests(unittest.TestCase):
    def test_reused_factors_satisfy_variable_coefficient_system(self):
        from wakemarch._kernel import YawedWakeSolver
        solver = YawedWakeSolver.recommended(**PARAMS, h_D=0.1)
        rng = np.random.default_rng(418)
        field = rng.normal(size=(solver.nz, solver.ny))
        # Large coefficient contrast exercises stalled refinement and reuse.
        coefficient = np.where(solver.Y > 0, 2000., 1.)
        for ground in (False, True):
            for scale in (1., 1.01, 1.02, 1.4):
                with self.subTest(ground=ground, scale=scale):
                    fast = solver.diffuse(field, scale*coefficient, 1., neumann_ground=ground)
                    direct = solver.diffuse(field, scale*coefficient, 1., neumann_ground=ground, fast=False)
                    np.testing.assert_allclose(fast, direct, rtol=1e-7, atol=1e-8)
                    rows = slice(None, -1) if ground else slice(1, -1)
                    x = fast[rows, 1:-1].ravel()
                    rhs = field[rows, 1:-1].ravel()
                    cv = (scale*coefficient)[rows, 1:-1].ravel()
                    operator = solver._L_du if ground else solver._L_int
                    residual = rhs - (x - cv*(operator @ x))
                    self.assertLess(np.linalg.norm(residual)/np.linalg.norm(rhs), 1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
