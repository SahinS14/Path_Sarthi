import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'idr-benchmark/src'))
from train_trajectory import load_data, train, digest, normalization, PROJECT
from prepare_trajectory import initial_attitude, prepare


def fixture(root):
    root.mkdir()
    records = []
    for i, split in enumerate(('train', 'validation')):
        t = np.arange(31)/10
        imu = np.zeros((31, 6)); imu[:, 5] = 9.80665; imu[:, 2] = .008
        imu[:, 3] = .08+i*.01; imu[:, 4] = .12
        truth = np.zeros((31, 3)); truth[:, 0] = (10+i)*t
        path = root/f'{i}.npz'
        np.savez_compressed(path, t=t, imu=imu, reference_position=truth,
                            initial_velocity=np.array([10+i, 0., 0.]), initial_rpy=np.zeros(3))
        records.append(dict(file=path.name, sha256=digest(path), drive=split,
                            session_group=split, split=split))
    manifest = dict(schema='idr-trajectory-v1', status='synthetic', sample_hz=10, records=records)
    (root/'manifest.json').write_text(json.dumps(manifest))
    return manifest


class TrajectoryTrainingTest(unittest.TestCase):
    def test_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = root/'data'; fixture(data)
            args = argparse.Namespace(data=data, out=root/'resumed', source=PROJECT/'work/reference/ai-imu-dr',
                                      epochs=1, patience=12, lr=.001, resume=False)
            train(args)
            args.epochs = 2; args.resume = True; train(args)
            args.out = root/'continuous'; args.resume = False; train(args)
            resumed = torch.load(root/'resumed/last.pt', weights_only=False)
            continuous = torch.load(root/'continuous/last.pt', weights_only=False)
            for name in resumed['model']:
                torch.testing.assert_close(resumed['model'][name], continuous['model'][name], rtol=0, atol=0)
            self.assertEqual(resumed['epoch'], 2)
            self.assertTrue((root/'resumed/best-validation.json').exists())
            args.out = root/'resumed'; args.resume = True; args.lr = .1
            with self.assertRaises(ValueError): train(args)

    def test_session_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'data'; manifest = fixture(root)
            manifest['records'][1]['session_group'] = 'train'
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): load_data(root)

    def test_normalization_training_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'data'; fixture(root)
            _, windows = load_data(root)
            mean, _ = normalization(windows)
            self.assertAlmostEqual(mean[3], .08)

    def test_source_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'data'; fixture(root)
            (root/'0.npz').write_bytes(b'invalid')
            with self.assertRaises(ValueError): load_data(root)

    def test_no_unverified_preparation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); config = root/'contracts.json'
            config.write_text(json.dumps(dict(schema='idr-recording-contracts-v1', records=[])))
            with self.assertRaises(ValueError): prepare(config, root/'prepared', 10, 10)
            self.assertFalse((root/'prepared').exists())

    def test_heading_convention(self):
        np.testing.assert_allclose(initial_attitude([0, 0, 9.81], np.pi/2), [0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(initial_attitude([0, 0, 9.81], 0), [0, 0, np.pi/2], atol=1e-12)
