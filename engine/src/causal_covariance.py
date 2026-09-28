"""Causal learned NHC covariance head. New architecture, no pretrained weights.

Input: normalized [gyro xyz, body specific-force xyz], B x 6 x T.
Output: positive lateral/vertical velocity measurement variances, B x T x 2.
The sample rate and normalization belong in the eventual trained artifact.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CausalCovariance(nn.Module):
    receptive_samples = 17

    def __init__(self, baseline=(0.2, 300.0)):
        super().__init__()
        base = torch.as_tensor(baseline, dtype=torch.float32)
        if base.shape != (2,) or not torch.isfinite(base).all() or not (base > 0).all():
            raise ValueError('Two positive baseline variances required')
        self.register_buffer('baseline', base)
        self.conv1 = nn.Conv1d(6, 32, 5)
        self.conv2 = nn.Conv1d(32, 32, 5, dilation=3)
        self.output = nn.Linear(32, 2)
        # Begin exactly at the fixed-covariance baseline, not random corrections.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, imu):
        if imu.ndim != 3 or imu.shape[1] != 6 or imu.shape[2] < 1:
            raise ValueError('Expected B x 6 x T input with T >= 1')
        y = F.relu(self.conv1(F.pad(imu, (4, 0))))
        y = F.relu(self.conv2(F.pad(y, (12, 0))))
        log_factor = 3 * torch.tanh(self.output(y.transpose(1, 2)))
        return self.baseline * torch.pow(10., log_factor)

    def stream(self, imu, history=None):
        """Inference-only chunk evaluation, returning detached finite history.

        Caller owns state and must clear it on a gap, session or device change.
        No hidden cross-vehicle state is retained inside the module.
        """
        if self.training:
            raise ValueError('Streaming requires eval mode')
        if imu.ndim != 3 or imu.shape[1] != 6 or imu.shape[-1] < 1:
            raise ValueError('Expected a nonempty B x 6 x T chunk')
        if history is not None:
            if (history.ndim != 3 or history.shape[:2] != imu.shape[:2]
                    or history.shape[-1] > self.receptive_samples-1):
                raise ValueError('Invalid streaming history')
            joined = torch.cat((history, imu), dim=-1)
        else:
            joined = imu
        with torch.no_grad():
            result = self(joined)[:, -imu.shape[-1]:]
        return result, joined[:, :, -(self.receptive_samples-1):].detach().clone()
