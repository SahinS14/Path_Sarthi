"""Controlled causality and adapter smoke tests; no dataset accuracy claims."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from reference_adapter import run_reference

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'engine/src'))
from causal_covariance import CausalCovariance


def check(source):
    sys.path.insert(0, str(source.resolve() / 'src'))
    from utils_torch_filter import MesNet
    from types import SimpleNamespace
    torch.manual_seed(26168)
    torch.set_num_threads(2)
    upstream = MesNet().eval()
    baseline = SimpleNamespace(cov0_measurement=torch.tensor([.2, 300.], dtype=torch.float64))
    x = torch.randn(1, 6, 100, dtype=torch.float64)
    changed = x.clone(); changed[:, :, 50:] += 10
    with torch.no_grad():
        original = upstream(x, baseline)
        perturbed = upstream(changed, baseline)
    upstream_change = float((original[:50]-perturbed[:50]).abs().max())
    causal = CausalCovariance().double().eval()
    torch.nn.init.normal_(causal.output.weight, std=.1)
    with torch.no_grad():
        causal_change = float((causal(x)[:, :50]-causal(changed)[:, :50]).abs().max())
    if causal_change != 0:
        raise AssertionError('Causal head uses future samples')
    # Actual reference implementation, both fixed and neural covariance paths.
    t = np.arange(301)/10
    imu = np.zeros((len(t), 6)); imu[:, 5] = 9.80665
    seed = CausalCovariance().eval()
    with torch.no_grad():
        covariance = seed(torch.from_numpy(imu.T).float().unsqueeze(0))[0].numpy()
    results = {}
    for name, cov in [('fixed', None), ('causal_untrained', covariance)]:
        output = run_reference(t, imu, [10, 0, 0], [0, 0, 0],
                               covariance=cov, source=source, initial_position=[100, 200, 0])
        expected = np.column_stack((100+10*t, np.full(len(t), 200), np.zeros(len(t))))
        error = float(np.linalg.norm(output['position']-expected, axis=1).max())
        if error > 1e-6:
            raise AssertionError('Reference adapter failed ideal straight-motion check')
        results[name] = dict(max_position_error_m=error)
    return dict(scope='Synthetic causality/adapter check, not trained navigation accuracy',
                upstream_max_past_variance_change=upstream_change,
                causal_max_past_variance_change=causal_change,
                altered_input_start_sample=50, scenarios=results,
                learned_weights_used=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('work/reference/ai-imu-dr'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = check(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps(report, indent=2))
