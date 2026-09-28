import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'idr-benchmark/src'))
from torch_filter_compat import skew, rotation_and_jacobian


class DifferentiableRotationTest(unittest.TestCase):
    def test_skew_gradcheck(self):
        x = torch.tensor([.1, -.2, .3], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(skew, (x,)))

    def test_rotation_jacobian_gradcheck(self):
        for values in ([0., 0., 0.], [.1, -.2, .3], [1e-6, -1e-6, 0.]):
            x = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(rotation_and_jacobian, (x,)))
            rotation, _ = rotation_and_jacobian(x)
            torch.testing.assert_close(rotation.T@rotation, torch.eye(3).double())
            torch.testing.assert_close(torch.linalg.det(rotation), torch.tensor(1.).double())
