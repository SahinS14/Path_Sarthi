import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine/src'))
from causal_covariance import CausalCovariance


class CausalCovarianceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.model = CausalCovariance().eval()
        # Nonzero head is essential: a zero head would hide future-data leakage.
        torch.nn.init.normal_(self.model.output.weight, std=.1)
        self.imu = torch.randn(2, 6, 50)

    def test_future_cannot_change_past(self):
        altered = self.imu.clone(); altered[:, :, 25:] += 10
        torch.testing.assert_close(self.model(self.imu)[:, :25], self.model(altered)[:, :25])

    def test_stream_equals_batch_including_startup(self):
        chunks, history = [], None
        for start, end in [(0, 1), (1, 8), (8, 30), (30, 50)]:
            result, history = self.model.stream(self.imu[:, :, start:end], history)
            chunks.append(result)
            self.assertLessEqual(history.shape[-1], 16)
        torch.testing.assert_close(torch.cat(chunks, dim=1), self.model(self.imu))

    def test_baseline_initialization_and_gradient(self):
        model = CausalCovariance()
        result = model(self.imu)
        torch.testing.assert_close(result, model.baseline.expand(2, 50, 2))
        result.log().mean().backward()
        self.assertGreater(model.output.weight.grad.abs().sum().item(), 0)

    def test_positive_bounded_outputs(self):
        result = self.model(self.imu * 100)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue((result >= self.model.baseline * .001).all())
        self.assertTrue((result <= self.model.baseline * 1000).all())

    def test_training_stream_rejected(self):
        with self.assertRaises(ValueError):
            self.model.train().stream(self.imu)
