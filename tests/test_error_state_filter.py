import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'engine/src'))
from error_state_filter import ErrorStateFilter, exp_rotation


class ErrorStateTests(unittest.TestCase):
    def make(self,velocity=(0,0,0)):
        return ErrorStateFilter([0,0,0],velocity,np.eye(3),0.)

    def test_stationary_gravity_and_covariance(self):
        f=self.make()
        for i in range(1,101):
            f.predict(i*.01,[0,0,9.80665],[0,0,0]); f.update_nonholonomic()
        np.testing.assert_allclose(f.p,0,atol=1e-10)
        self.assertGreater(np.linalg.eigvalsh(f.P).min(),0)
        np.testing.assert_allclose(f.R.T@f.R,np.eye(3),atol=1e-10)

    def test_constant_velocity_is_not_zeroed(self):
        f=self.make([10,0,0])
        for i in range(1,101):
            f.predict(i*.01,[0,0,9.80665],[0,0,0]); f.update_nonholonomic()
        np.testing.assert_allclose(f.p,[10,0,0],atol=1e-8)

    def test_acceleration(self):
        f=self.make()
        for i in range(1,101): f.predict(i*.01,[1,0,9.80665],[0,0,0])
        np.testing.assert_allclose(f.p,[.5,0,0],atol=1e-8)

    def test_coordinated_turn(self):
        f=self.make([10,0,0]); rate=.2
        for i in range(1,501):
            f.predict(i*.01,[0,10*rate,9.80665],[0,0,rate])
            f.update_nonholonomic()
        expected=[50*np.sin(1),50*(1-np.cos(1)),0]
        np.testing.assert_allclose(f.p,expected,atol=.02)
        self.assertGreater(np.linalg.eigvalsh(f.P).min(),0)

    def test_nonholonomic_jacobian(self):
        from error_state_filter import skew
        R=exp_rotation(np.array([.1,-.2,.4])); v=np.array([5.,2.,.1])
        H=skew(R.T@v)[1:3,:]; eps=1e-7
        for axis in range(3):
            step=np.eye(3)[axis]*eps
            numerical=((R@exp_rotation(step)).T@v-R.T@v)[1:3]/eps
            np.testing.assert_allclose(numerical,H[:,axis],atol=1e-6)

    def test_rotated_body(self):
        R=exp_rotation(np.array([.3,.2,.4]))
        f=ErrorStateFilter([0]*3,[0]*3,R,0.)
        for i in range(1,101): f.predict(i*.01,R.T@np.array([0,0,9.80665]),[0]*3)
        np.testing.assert_allclose(f.p,0,atol=1e-8)

    def test_invalid_time_does_not_mutate(self):
        f=self.make()
        for t in (0,-1,1,float('nan')):
            with self.assertRaises(ValueError): f.predict(t,[0,0,9.80665],[0]*3)
        self.assertEqual(f.time,0)

    def test_gnss_duplicate_and_outlier(self):
        f=self.make()
        self.assertTrue(f.update_gnss(0,[1,0,0],2))
        p=f.p.copy(); P=f.P.copy()
        self.assertFalse(f.update_gnss(0,[1,0,0],2))
        np.testing.assert_array_equal(f.P,P)
        f.predict(.1,[0,0,9.80665],[0]*3)
        self.assertFalse(f.update_gnss(.1,[1e6]*3,2))
        np.testing.assert_allclose(f.p,p)


if __name__=='__main__': unittest.main()
