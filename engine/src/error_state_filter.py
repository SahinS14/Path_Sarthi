"""15-state multiplicative error-state EKF, independent of Android.

Conventions: local ENU world; right-handed body x-forward/y-left/z-up.
R maps body vectors into world. Accelerometer is specific force in m/s²,
gyro is body angular velocity in rad/s. Biases are expressed in body axes.
Error order: position, velocity, right attitude error, accel bias, gyro bias.
This is an ESKF baseline, not the reference repository's invariant EKF.
"""
import numpy as np


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def exp_rotation(v):
    angle = np.linalg.norm(v)
    k = skew(v)
    if angle < 1e-8:
        return np.eye(3) + k + .5*k@k
    return np.eye(3) + np.sin(angle)/angle*k + (1-np.cos(angle))/angle**2*k@k


def vector(value, size):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'Expected finite vector of length {size}')
    return result.copy()


class ErrorStateFilter:
    def __init__(self, position, velocity, rotation, time_s,
                 accel_noise=.5, gyro_noise=.03, accel_bias_walk=.005,
                 gyro_bias_walk=.0005):
        self.p = vector(position, 3); self.v = vector(velocity, 3)
        self.R = np.asarray(rotation, dtype=float).copy()
        if (self.R.shape != (3,3) or not np.isfinite(self.R).all()
                or not np.allclose(self.R.T@self.R, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(self.R), 1.)):
            raise ValueError('rotation must belong to SO(3)')
        if not np.isfinite(time_s): raise ValueError('Invalid initial time')
        self.time = float(time_s); self.ba = np.zeros(3); self.bg = np.zeros(3)
        self.P = np.diag([25.]*3 + [4.]*3 + [.1]*3 + [.25]*3 + [.01]*3)
        noises = np.array([accel_noise, gyro_noise, accel_bias_walk, gyro_bias_walk])
        if not np.isfinite(noises).all() or (noises<=0).any(): raise ValueError('Invalid noise')
        self.noises = noises
        self.last_gnss_time = None

    def predict(self, time_s, specific_force, angular_rate):
        a = vector(specific_force,3); w = vector(angular_rate,3)
        dt = float(time_s)-self.time
        if not np.isfinite(dt) or dt<=0 or dt>.25:
            raise ValueError('IMU time must increase with a gap <=250 ms; reinitialize after gaps')
        # Substeps limit linearization error at lower input rates.
        steps = max(1, int(np.ceil(dt/.02)))
        for _ in range(steps): self._propagate(dt/steps, a-self.ba, w-self.bg)
        self.time = float(time_s)

    def _propagate(self, dt, a, w):
        F=np.zeros((15,15)); F[0:3,3:6]=np.eye(3)
        F[3:6,6:9]=-self.R@skew(a); F[3:6,9:12]=-self.R
        F[6:9,6:9]=-skew(w); F[6:9,12:15]=-np.eye(3)
        A=np.eye(15)+F*dt+.5*(F@F)*dt**2
        G=np.zeros((15,12)); G[3:6,0:3]=self.R; G[6:9,3:6]=np.eye(3)
        G[9:12,6:9]=np.eye(3); G[12:15,9:12]=np.eye(3)
        noise=np.repeat(self.noises**2,3)
        Q=(G*noise)@G.T
        # First-order noise integration, including position/velocity coupling.
        B=np.eye(15)+.5*F*dt
        self.P=A@self.P@A.T+B@Q@B.T*dt
        midpoint=self.R@exp_rotation(w*dt*.5)
        world_acc=midpoint@a+np.array([0.,0.,-9.80665])
        self.p+=self.v*dt+.5*world_acc*dt**2; self.v+=world_acc*dt
        self.R=self.R@exp_rotation(w*dt)
        self.P=(self.P+self.P.T)*.5

    def _update(self, residual, H, variances, gate):
        variances=vector(variances,len(residual))
        if (variances<=0).any(): raise ValueError('Measurement variances must be positive')
        V=np.diag(variances); S=H@self.P@H.T+V
        nis=float(residual@np.linalg.solve(S,residual))
        if nis>gate: return False
        K=np.linalg.solve(S,H@self.P).T; dx=K@residual
        I=np.eye(15)-K@H
        self.P=I@self.P@I.T+K@V@K.T  # Joseph form
        self.p+=dx[:3]; self.v+=dx[3:6]; self.R=self.R@exp_rotation(dx[6:9])
        self.ba+=dx[9:12]; self.bg+=dx[12:15]
        reset=np.eye(15); reset[6:9,6:9]-=.5*skew(dx[6:9])
        self.P=reset@self.P@reset.T; self.P=(self.P+self.P.T)*.5
        return True

    def update_gnss(self, time_s, position, position_sigma, velocity=None, velocity_sigma=1.):
        # Call at the synchronized IMU timestamp. Delayed measurements require replay.
        if not np.isfinite(time_s) or abs(time_s-self.time)>1e-6:
            raise ValueError('GNSS must be synchronized to the current state')
        if self.last_gnss_time is not None and time_s<=self.last_gnss_time:
            return False
        p=vector(position,3)
        if not np.isfinite(position_sigma) or position_sigma<=0: raise ValueError('Invalid position sigma')
        H=np.zeros((3,15)); H[:,:3]=np.eye(3); residual=p-self.p
        variances=np.full(3,position_sigma**2)
        if velocity is not None:
            vel=vector(velocity,3)
            if not np.isfinite(velocity_sigma) or velocity_sigma<=0: raise ValueError('Invalid velocity sigma')
            Hv=np.zeros((3,15)); Hv[:,3:6]=np.eye(3)
            H=np.vstack([H,Hv]); residual=np.r_[residual,vel-self.v]
            variances=np.r_[variances,np.full(3,velocity_sigma**2)]
        self.last_gnss_time=float(time_s)
        return self._update(residual,H,variances,22.46 if velocity is not None else 16.27)

    def update_nonholonomic(self, lateral_variance=.25, vertical_variance=.25):
        # Requires calibrated vehicle axes. Do not assume these for a tilted phone.
        body_velocity=self.R.T@self.v
        H=np.zeros((2,15)); H[:,3:6]=self.R.T[1:3,:]
        H[:,6:9]=skew(body_velocity)[1:3,:]
        return self._update(-body_velocity[1:3],H,[lateral_variance,vertical_variance],13.82)
