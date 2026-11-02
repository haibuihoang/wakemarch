"""Installed-package turbine tables, rotor averaging and wake construction checks."""
from dataclasses import FrozenInstanceError
import json
from importlib.resources import files
import unittest
import warnings
import numpy as np
from wakemarch import Turbine, Numerics, WakeModel, available_turbines
from wakemarch.turbine import CT_CAP


def custom(**kw):
    args=dict(name='test',D=100,H=80,wind_speeds=[0,5,10,15,25],
              power_W=[0,1e5,1e6,2e6,0],ct=[0,.8,.7,.3,0])
    args.update(kw);return Turbine.from_power_ct(**args)


class TurbineTests(unittest.TestCase):
    def test_fixed_farm_conventions(self):
        t = Turbine.fixed(D=126, H=90, CT=.8, CP=.45)
        u = np.array([4., 8., 10.])
        p = t.operating_point(u, yaw_deg=20, rho=1.2)
        expected = .5 * 1.2 * (np.pi * 126**2 / 4) * .45 * np.mean(u**3) * np.cos(np.radians(20))**3
        self.assertAlmostEqual(p.power_W, expected)
        self.assertAlmostEqual(p.rotor_mean_m_s, np.mean(u))
        self.assertNotAlmostEqual(p.rotor_mean_m_s, p.rotor_equivalent_m_s)
        direct = WakeModel(D=126, H=90, U=np.mean(u), CT=.8, yaw_deg=20, TI=.1)
        self.assertEqual(t.wake_model(u, TI=.1, yaw_deg=20), direct)
        weighted = t.operating_point(u, weights=[1, 2, 1])
        self.assertAlmostEqual(weighted.rotor_mean_m_s, 7.5)
        for kw in (dict(CT=1), dict(CP=0), dict(H=1)):
            args = dict(D=126, H=90, CT=.8, CP=.45); args.update(kw)
            with self.assertRaises(ValueError): Turbine.fixed(**args)
        with self.assertRaises(ValueError): t.operating_point(u, weights=[0, 0, 0])

    def test_presets_and_packaged_data(self):
        self.assertEqual(available_turbines(),('nrel_5MW','iea_10MW','iea_15MW'))
        for name,rated in zip(available_turbines(),(5e6,10e6,15e6)):
            t=Turbine.preset(name)
            d=json.loads(files('wakemarch').joinpath('data',name+'.json').read_text())
            np.testing.assert_array_equal(t.power_at(d['wind_speed_m_s']),d['power_W'])
            np.testing.assert_array_equal(t.ct_at(d['wind_speed_m_s']),d['CT'])
            self.assertEqual(t.power_at(16),rated)
            self.assertIn('4.6.2',t.provenance)
        self.assertIn('Copyright',files('wakemarch').joinpath('data','FLORIS_LICENSE.txt').read_text())
        with self.assertRaises(ValueError):Turbine.preset('missing')

    def test_cp_conversion_and_efficiency(self):
        t=Turbine.from_cp_ct(name='cp',D=100,H=80,wind_speeds=[0,8,12],
                            cp=[0,.45,.3],ct=[0,.8,.6],generator_efficiency=.94)
        expected=.94*.5*1.225*np.pi*100**2/4*.45*8**3
        self.assertAlmostEqual(t.power_at(8),expected)
        self.assertAlmostEqual(t.cp_at(8),.94*.45)
        self.assertEqual(t.cp_at(0),0)
        # Canonical linear-power interpolation, not linear-Cp interpolation.
        self.assertAlmostEqual(t.power_at(10),(t.power_at(8)+t.power_at(12))/2)

    def test_invalid_tables(self):
        for kw in (dict(D=0),dict(H=20),dict(rho_ref=0),dict(yaw_power_exponent=-1),
                   dict(wind_speeds=[0,5,5,15,25]),dict(ct=[0,.8]),
                   dict(power_W=[0,1,np.nan,2,0]),dict(ct=[0,-1,.7,.3,0]),
                   dict(wind_speeds=[[0,5,10,15,25]]),dict(power_W=[1,2,3,4,0])):
            with self.subTest(kw=kw),self.assertRaises(ValueError):custom(**kw)
        with self.assertRaises(ValueError):
            Turbine.from_cp_ct(name='x',D=100,H=80,wind_speeds=[0,8],cp=[0,1.1],ct=[0,.8])

    def test_immutable_owned_tables(self):
        values=[0,1e5,1e6,2e6,0];t=custom(power_W=values);values[1]=99
        self.assertEqual(t.power_at(5),1e5)
        with self.assertRaises(FrozenInstanceError):t.D=10
        with self.assertRaises(TypeError):t.ct_values[1]=.2

    def test_query_bounds(self):
        t=custom()
        for u in (-1,np.nan,np.inf,26):
            with self.subTest(u=u),self.assertRaises(ValueError):t.power_at(u)
        np.testing.assert_array_equal(t.ct_at([0,26],bounds='zero'),[0,0])
        with self.assertRaises(ValueError):t.power_at(8,bounds='clip')

    def test_weighted_rotor_average(self):
        t=custom();p=t.operating_point([6,10],weights=[1,3])
        expected=((6**3+3*10**3)/4)**(1/3)
        self.assertAlmostEqual(p.rotor_equivalent_m_s,expected)
        self.assertAlmostEqual(p.rotor_mean_m_s,9)
        self.assertAlmostEqual(p.CT,t.ct_at(expected))
        self.assertAlmostEqual(p.power_W,t.power_at(expected))
        self.assertGreater(abs(p.power_W-t.power_at(9)),1)
        p2=t.operating_point([[6,10],[6,10]],weights=[1,3])
        self.assertEqual(p,p2)

    def test_rotor_input_guards(self):
        t=custom()
        for velocities,weights in (([],None),([-1],None),([np.nan],None),([8],[0]),
                                  ([8],[-1]),([8],[np.inf]),([8,9],[1,2,3])):
            with self.assertRaises(ValueError):t.operating_point(velocities,weights=weights)
        for kw in (dict(yaw_deg=90),dict(rho=0),dict(yaw_deg=np.nan)):
            with self.assertRaises(ValueError):t.operating_point([8],**kw)

    def test_yaw_density_and_no_double_CT_correction(self):
        t=Turbine.preset('nrel_5MW');p=t.operating_point([8],yaw_deg=20,rho=1.3)
        effective=8*(1.3/1.225)**(1/3)*np.cos(np.radians(20))**(1.88/3)
        self.assertAlmostEqual(p.power_W,t.power_at(effective))
        self.assertAlmostEqual(p.CT,t.ct_at(8))
        self.assertAlmostEqual(t.operating_point([16],yaw_deg=20).power_W,5e6)

    def test_nonoperating_and_high_CT(self):
        t=Turbine.preset('nrel_5MW')
        for u in (0,30):
            p=t.operating_point([u]);self.assertFalse(p.operating);self.assertEqual(p.power_W,0)
            with self.assertRaises(ValueError):t.wake_model([u],TI=.1)
        self.assertGreater(t.ct_at(3),1)
        for u in (3,3.5):
            with self.subTest(u=u),self.assertWarnsRegex(RuntimeWarning,"nrel_5MW"):
                p=t.operating_point([u])
            self.assertEqual(p.CT,CT_CAP);self.assertTrue(p.operating)
            # U_RE comes from a cube root; libm differences across platforms shift the last bits.
            np.testing.assert_allclose(p.power_W,t.power_at(u),rtol=1e-12)
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(t.wake_model([3.5],TI=.1).CT,CT_CAP)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self.assertAlmostEqual(t.operating_point([4]).CT,t.ct_at(4))   # 0.9995 < 1: unchanged

    def test_table_range_is_operating_range(self):
        t=Turbine.from_power_ct(name='toy',D=100,H=90,wind_speeds=[3,4,10,25],
                                power_W=[1e4,1e5,2e6,2e6],ct=[.8,.8,.7,.1])
        # Below cut-in / above cut-out of the table: not operating, no wake.
        for u in (2.5,26.):
            p=t.operating_point([u])
            self.assertEqual((p.power_W,p.CT,p.operating),(0.,0.,False))
            with self.assertRaises(ValueError):t.wake_model([u],TI=.1)
        # Yaw lowers the power-effective speed below the table: zero power, wake kept.
        p=t.operating_point([3.2],yaw_deg=30)
        self.assertLess(p.power_effective_m_s,3)
        self.assertEqual(p.power_W,0.);self.assertTrue(p.operating)
        self.assertAlmostEqual(p.CT,t.ct_at(3.2))
        self.assertAlmostEqual(t.operating_point([3.2]).power_W,28000.,places=6)
        # Denser air raises it above the table: power at the last tabulated speed.
        p=t.operating_point([24.9],rho=1.3)
        self.assertGreater(p.power_effective_m_s,25)
        self.assertEqual(p.power_W,2e6);self.assertTrue(p.operating)
        # Lookup methods keep their explicit bounds policy.
        with self.assertRaises(ValueError):t.power_at(2.5)

    def test_exact_wake_constructor_reduction(self):
        t=Turbine.preset('nrel_5MW');n=Numerics(h_D=.1)
        via=t.wake_model([8],TI=.1,yaw_deg=20,numerics=n)
        direct=WakeModel(D=t.D,H=t.H,U=8,CT=t.ct_at(8),yaw_deg=20,TI=.1,numerics=n)
        self.assertEqual(via,direct)
        a=via.solve(2,progress=False);b=direct.solve(2,progress=False)
        np.testing.assert_array_equal(a.marching_deficit,b.marching_deficit)
        with self.assertRaises(ValueError):t.wake_model([8],TI=2)

if __name__=='__main__':unittest.main(verbosity=2)
