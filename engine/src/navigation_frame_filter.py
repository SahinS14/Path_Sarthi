"""Planar EKF for already navigation-frame acceleration, not raw phone IMU.

State [east, north, ve, vn, bias_e, bias_n, heading, gyro_bias]. A caller must verify and provide
ENU acceleration with gravity removed. Heading is independently gyro-propagated
and is used only for optional, uncertainty-weighted non-holonomic observations.
"""
import numpy as np


class NavigationFrameFilter:
    def __init__(self, position, velocity, time_s, acceleration_noise=1., bias_walk=.01):
        p=np.asarray(position,dtype=float); v=np.asarray(velocity,dtype=float)
        if p.shape!=(2,) or v.shape!=(2,) or not np.isfinite(np.r_[p,v,time_s]).all():
            raise ValueError('Invalid initial state')
        if min(acceleration_noise,bias_walk)<=0 or not np.isfinite([acceleration_noise,bias_walk]).all():
            raise ValueError('Invalid noise')
        self.x=np.r_[p,v,np.zeros(2),np.arctan2(v[1],v[0]),0.]
        self.P=np.diag([25,25,4,4,.25,.25,np.deg2rad(10.)**2,.01]).astype(float)
        self.time=float(time_s)
        self.acceleration_noise=acceleration_noise; self.bias_walk=bias_walk
        self.last_gnss_time=None

    def predict(self,time_s,acceleration_enu,yaw_rate_ccw):
        a=np.asarray(acceleration_enu,dtype=float); dt=time_s-self.time
        if a.shape!=(2,) or not np.isfinite(np.r_[a,dt,yaw_rate_ccw]).all() or dt<=0 or dt>.25:
            raise ValueError('Invalid input or time gap')
        F=np.eye(8); F[:2,2:4]=np.eye(2)*dt; F[:2,4:6]=-.5*np.eye(2)*dt**2; F[2:4,4:6]=-np.eye(2)*dt
        F[6,7]=-dt
        B=np.vstack([.5*np.eye(2)*dt**2,np.eye(2)*dt,np.zeros((4,2))])
        self.x=F@self.x+B@a
        self.P=F@self.P@F.T+(B@B.T)*self.acceleration_noise**2/dt
        self.P[4:6,4:6]+=np.eye(2)*self.bias_walk**2*dt
        self.x[6]+=yaw_rate_ccw*dt; self.P[6,6]+=.03**2*dt; self.P[7,7]+=.0005**2*dt
        self.time=float(time_s)

    def _update(self,residual,H,V,gate):
        S=H@self.P@H.T+V
        if float(residual@np.linalg.solve(S,residual))>gate: return False
        K=np.linalg.solve(S,H@self.P).T
        self.x+=K@residual; A=np.eye(8)-K@H
        self.P=A@self.P@A.T+K@V@K.T; self.P=(self.P+self.P.T)/2
        return True

    def update_gnss(self,time_s,position,velocity,position_sigma=5.,velocity_sigma=1.):
        if abs(time_s-self.time)>1e-6 or not np.isfinite(time_s): raise ValueError('Unsynchronized GNSS')
        if self.last_gnss_time is not None and time_s<=self.last_gnss_time: return False
        z=np.r_[position,velocity]
        if z.shape!=(4,) or not np.isfinite(z).all() or min(position_sigma,velocity_sigma)<=0 or not np.isfinite([position_sigma,velocity_sigma]).all():
            raise ValueError('Invalid GNSS measurement')
        self.last_gnss_time=time_s; H=np.eye(8)[:4]
        accepted=self._update(z-self.x[:4],H,np.diag([position_sigma**2]*2+[velocity_sigma**2]*2),18.47)
        return accepted

    def update_nonholonomic(self,variance=.25):
        if not np.isfinite(variance) or variance<=0: raise ValueError('Invalid NHC variance')
        normal=np.array([-np.sin(self.x[6]),np.cos(self.x[6])])
        tangent=np.array([np.cos(self.x[6]),np.sin(self.x[6])])
        H=np.zeros((1,8)); H[0,2:4]=normal; H[0,6]=-tangent@self.x[2:4]
        # Joint heading/velocity covariance allows a lateral residual to correct
        # heading instead of repeatedly reducing speed to satisfy a wrong heading.
        return self._update(np.array([-normal@self.x[2:4]]),H,np.array([[variance]]),10.83)
