"""Modern-PyTorch compatibility subclass for the MIT-licensed ai-imu-dr filter.

Upstream equations: https://github.com/mbrossar/ai-imu-dr. Original source and
license remain under work/reference/ai-imu-dr. Only CPU float64 is qualified here.
Use differentiable stack operations rather than constructing tensors from tensor
scalars, which detaches their gradients. No global monkey patches.
"""
import torch


def skew(x):
    zero = x[0] * 0
    return torch.stack((zero, -x[2], x[1], x[2], zero, -x[0], -x[1], x[0], zero)).reshape(3, 3)


def rotation_and_jacobian(phi):
    theta2 = phi.square().sum()
    # Branch avoids undefined norm derivatives and division by zero at origin.
    if theta2.detach().item() < 1e-8:
        a = 1-theta2/6+theta2.square()/120
        b = .5-theta2/24+theta2.square()/720
        c = 1/6-theta2/120+theta2.square()/5040
    else:
        theta = theta2.sqrt()
        a = theta.sin()/theta
        b = (1-theta.cos())/theta2
        c = (theta-theta.sin())/(theta2*theta)
    k = skew(phi); identity = torch.eye(3, dtype=phi.dtype, device=phi.device)
    return identity+a*k+b*(k@k), identity+b*k+c*(k@k)


def compatible_filter(base):
    class CompatibleIEKF(base):
        skew = staticmethod(skew)

        @staticmethod
        def so3exp(phi):
            return rotation_and_jacobian(phi)[0]

        @staticmethod
        def sen3exp(xi):
            rotation, jacobian = rotation_and_jacobian(xi[:3])
            return rotation, jacobian @ xi[3:].reshape(-1, 3).T

        def state_and_cov_update(self, Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, H, r, R):
            innovation = H @ P @ H.T + R
            gain = torch.linalg.solve(innovation, (P @ H.T).T).T
            dx = gain @ r.reshape(-1)
            rotation, delta = self.sen3exp(dx[:9])
            correction = self.so3exp(dx[15:18])
            residual = self.IdP-gain@H
            covariance = residual@P@residual.T+gain@R@gain.T
            return (rotation@Rot, rotation@v+delta[:, 0], rotation@p+delta[:, 1],
                    b_omega+dx[9:12], b_acc+dx[12:15], correction@Rot_c_i,
                    t_c_i+dx[18:21], (covariance+covariance.T)/2)

    return CompatibleIEKF
