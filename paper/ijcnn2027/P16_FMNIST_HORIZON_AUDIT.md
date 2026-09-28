# P16 Fashion-MNIST horizon audit

## Purpose

The original headline campaigns used 150 communication rounds for the
Fashion-MNIST MLP and 80 rounds for the Fashion-MNIST CNN. Their reconstructed
loss curves were still decreasing at the final evaluation. P16 therefore asks
how long the two tasks must run before the held-out cross-entropy reaches a
practical plateau. The audit precedes, and does not reuse, the final ten-seed
campaign.

Here, a training "horizon" is the number of communication rounds. Every round
retains the frozen five local SGD steps, so the audit does not change the local
optimization depth.

## Frozen design

- One matched client partition and training seed for every method:
  partition seed 2500 and training seed 72500.
- Five frozen quality-selected methods: Event-FedAvg, STrom, EF-TopK, Sign-EF,
  and Dense FedAvg.
- MLP horizons: 600, 900, 1200, and 1500 rounds.
- CNN horizons: 400, 600, 900, 1200, and 1800 rounds.
- All optimizer, partition, encoder, and communication parameters remain
  unchanged.
- The diagnostic curve uses a centered five-evaluation moving average.
- The late window is the final 20% of communication rounds.

A curve is classified as plateaued when the absolute late-window loss slope is
at most \(10^{-4}\) per round, late-window improvement is at most 0.01 CE, and
the final smoothed loss is within 0.01 CE of its minimum. A positive late slope
or a substantially earlier minimum is classified as plateau/overfit rather than
still improving.

## Result

| Architecture | Selected horizon | Event-FedAvg | STrom | EF-TopK | Sign-EF | Dense |
|---|---:|---|---|---|---|---|
| MLP | 1500 | plateau | plateau/overfit | plateau | plateau | plateau |
| CNN | 1800 | plateau | plateau/overfit | plateau | plateau | plateau |

At the selected horizons, the single-seed final endpoints are:

| Architecture | Method | Test CE | Test accuracy |
|---|---|---:|---:|
| MLP | Event-FedAvg | 0.363 | 87.05% |
| MLP | STrom | 0.470 | 83.75% |
| MLP | EF-TopK | 0.366 | 86.68% |
| MLP | Sign-EF | 0.374 | 86.94% |
| MLP | Dense FedAvg | 0.372 | 86.77% |
| CNN | Event-FedAvg | 0.339 | 88.04% |
| CNN | STrom | 0.412 | 86.74% |
| CNN | EF-TopK | 0.344 | 87.21% |
| CNN | Sign-EF | 0.342 | 87.67% |
| CNN | Dense FedAvg | 0.342 | 87.35% |

STrom does not require a longer horizon. Its best smoothed test loss occurs near
round 900 for the MLP and round 1170 for the CNN, after which the held-out loss
slightly increases. This is evidence of saturation or overfitting, not continued
undertraining.

## Decision

Use **1500 rounds for the Fashion-MNIST MLP** and **1800 rounds for the
Fashion-MNIST CNN** in the final matched ten-seed campaign. Evaluate every 15
rounds for the MLP and every 10 rounds for the CNN. Report the predeclared final
round for every method. Do not select method-specific checkpoints from test
performance.

This audit uses one held-out trajectory to choose a conservative common budget.
It is a convergence diagnostic, not inferential evidence and not test-tuned
early stopping. The ten-seed campaign must be rerun from scratch at the frozen
horizons before manuscript results or confidence intervals are updated.

## Reproduction

```bash
PYTHONPATH=src python experiments/p16_fmnist_horizon_audit.py prepare-data
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src python experiments/p16_fmnist_horizon_audit.py campaign --workers 5
PYTHONPATH=src python experiments/p16_fmnist_horizon_audit.py analyze
PYTHONPATH=src python experiments/p16_fmnist_horizon_audit.py progression
PYTHONPATH=src python experiments/p16_fmnist_horizon_audit.py figure
```

The result directory retains every audited horizon, its per-round history, the
plateau analysis, the cross-horizon progression table, and the final diagnostic
figure.
