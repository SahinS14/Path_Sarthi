"""Small synthetic autograd check, NOT a dataset-trained model or PS benchmark."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'engine/src'))
from causal_covariance import CausalCovariance
from torch_filter_compat import compatible_filter
from reference_adapter import run_reference


def run(source, steps):
    sys.path.insert(0, str(source.resolve() / 'src'))
    from utils_torch_filter import TORCHIEKF
    torch.manual_seed(26168); torch.set_num_threads(2)
    # CPU float64 matches upstream constants and linear-algebra operations.
    f = compatible_filter(TORCHIEKF)()
    f.g = torch.as_tensor(f.g, dtype=torch.float64)
    for parameter in f.parameters():
        parameter.requires_grad_(False)
    # Disable random learned initialization/process multipliers for this check.
    with torch.no_grad():
        for parameter in f.initprocesscov_net.parameters():
            parameter.zero_()
    f.set_Q()
    model = CausalCovariance().double()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    t = torch.arange(81, dtype=torch.float64) / 10
    u = torch.zeros((len(t), 6), dtype=torch.float64)
    u[:, 5] = 9.80665
    u[:, 1] = .006; u[:, 2] = .008
    u[:, 3] = .08; u[:, 4] = .12
    v0 = torch.zeros((len(t), 3), dtype=torch.float64); v0[0, 0] = 10
    truth = torch.zeros_like(v0); truth[:, 0] = 10*t
    # Synthetic-only fixed normalization, not derived from held-out labels.
    network_input = u.clone(); network_input[:, 5] -= 9.80665
    network_input = network_input.T.unsqueeze(0)

    def predict():
        cov = model(network_input)[0]
        _, _, position, *_ = f.run(t, u, cov, v0, torch.zeros_like(v0), len(t), torch.zeros(3).double())
        return position, cov

    def loss():
        return ((predict()[0]-truth)**2).mean()

    # Check forward parity against the independent NumPy implementation.
    with torch.no_grad():
        position, covariance = predict()
        reference = run_reference(t.numpy(), u.numpy(), v0[0].numpy(), np.zeros(3),
                                  covariance=covariance.numpy(), source=source)
        parity_error = float(np.max(np.abs(position.numpy()-reference['position'])))
    if parity_error > 1e-5:
        raise ValueError('Differentiable and NumPy filter trajectories disagree')
    # Finite difference across the complete filter, not just the neural layer.
    initial = loss(); initial.backward()
    analytic = float(model.output.bias.grad[0])
    epsilon = 1e-5
    with torch.no_grad():
        model.output.bias[0] += epsilon
        plus = float(loss())
        model.output.bias[0] -= 2*epsilon
        minus = float(loss())
        model.output.bias[0] += epsilon
    numerical = (plus-minus)/(2*epsilon)
    if not np.isclose(analytic, numerical, rtol=1e-3, atol=1e-5):
        raise ValueError(f'Trajectory gradient mismatch: {analytic} vs {numerical}')

    history = []; start = time.perf_counter()
    for step in range(steps):
        optimizer.zero_grad()
        value = loss()
        if not torch.isfinite(value):
            raise ValueError('Nonfinite trajectory loss')
        value.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise ValueError('Missing or nonfinite covariance-head gradient')
        gradient = float(sum(g.square().sum() for g in grads).sqrt())
        if step == 0 and gradient == 0:
            raise ValueError('No trajectory gradient reaches the neural head')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        history.append(dict(step=step, trajectory_mse=float(value.detach()), gradient_norm=gradient))
        print(json.dumps(history[-1]), flush=True)
    with torch.no_grad():
        final = float(loss())
    return dict(scope='Overfit one synthetic 8-second straight-motion case to verify autograd only',
                device='cpu', dtype='float64', steps=steps, history=history,
                initial_mse=history[0]['trajectory_mse'], final_mse=final,
                improved=final < history[0]['trajectory_mse'],
                numpy_parity_max_position_difference_m=parity_error,
                gradient_check=dict(analytic=analytic, finite_difference=numerical),
                wall_seconds=time.perf_counter()-start,
                dataset_training=False, production_weights_exported=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('work/reference/ai-imu-dr'))
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    if args.out.exists():
        parser.error('Use a new output path')
    report = run(args.source, args.steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'history'}, indent=2))
