import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'engine/src'))
from navigation_frame_filter import NavigationFrameFilter


class NavigationFrameTests(unittest.TestCase):
    def test_motion_and_covariance(self):
        k=NavigationFrameFilter([0,0],[10,0],0)
        for i in range(1,101):
            k.predict(i*.01,[1,0],0); k.update_nonholonomic()
        np.testing.assert_allclose(k.x[:4],[10.5,0,11,0],atol=1e-8)
        self.assertGreater(np.linalg.eigvalsh(k.P).min(),0)

    def test_world_acceleration_not_rotated_twice(self):
        k=NavigationFrameFilter([0,0],[0,10],0)
        for i in range(1,101): k.predict(i*.01,[1,0],.1)
        np.testing.assert_allclose(k.x[:4],[.5,10,1,10],atol=1e-8)

    def test_gnss_bias_learning(self):
        k=NavigationFrameFilter([0,0],[10,0],0)
        for i in range(1,1001):
            t=i*.1; k.predict(t,[.2,0],0)
            if i%10==0: k.update_gnss(t,[10*t,0],[10,0],1.,.2)
        self.assertAlmostEqual(k.x[4],.2,delta=.02)
        self.assertAlmostEqual(k.x[2],10,delta=.1)

    def test_stale_and_invalid(self):
        k=NavigationFrameFilter([0,0],[0,0],0)
        self.assertTrue(k.update_gnss(0,[0,0],[0,0]))
        p=k.P.copy(); self.assertFalse(k.update_gnss(0,[0,0],[0,0]))
        np.testing.assert_array_equal(k.P,p)
        with self.assertRaises(ValueError): k.predict(-1,[0,0],0)

    def test_heading_error_does_not_force_vehicle_to_stop(self):
        k=NavigationFrameFilter([0,0],[10,0],0)
        k.x[6]=.2
        for i in range(1,101):
            k.predict(i*.01,[0,0],0); k.update_nonholonomic()
        self.assertGreater(np.linalg.norm(k.x[2:4]),9.)
        lateral=-np.sin(k.x[6])*k.x[2]+np.cos(k.x[6])*k.x[3]
        self.assertLess(abs(lateral),.05)
        self.assertGreater(np.linalg.eigvalsh(k.P).min(),0)


if __name__=='__main__': unittest.main()
