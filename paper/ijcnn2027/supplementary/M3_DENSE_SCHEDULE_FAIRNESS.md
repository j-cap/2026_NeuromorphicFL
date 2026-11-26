# M3 — Scheduled dense FedAvg fairness control

This supplementary note documents the 1,800-round compact CIFAR-10 CNN
server-gain sensitivity analysis. The control varies only the dense server
gain schedule, not local SGD or modeled bidirectional communication.

## Frozen selection protocol

The gain is
\[
\gamma_r=\gamma_0(1+r/100)^{-\alpha_\gamma}.
\]
The 15 development candidates cross
\(\gamma_0\in\{1,2,4\}\) and
\(\alpha_\gamma\in\{0,0.1,0.2,0.3,0.5\}\).
The dedicated strong-non-IID development partition seed 4500
(training seed 84500) is outside the final ten evaluation partition seeds
3500–4400. Candidates run for 1,800 rounds and are ranked by *final
development training objective*, not test accuracy. The winning
configuration is \(\gamma_0=4,\alpha_\gamma=0.3\)
with development objective 0.990088.

## Ten matched seeds

| Comparison | Scheduled dense minus comparator, accuracy (percentage points) | Paired 95% CI |
|---|---:|---|
| Constant dense (gain 2) | +0.206 | [-0.404, +0.816] |
| EF-TopK | -1.740 | [-2.562, -0.918] |
| Event-FedAvg | -2.331 | [-2.879, -1.783] |

The scheduled dense mean accuracy is 54.82%, compared with 54.62% for
the original dense control, 56.56% for EF-TopK, and 57.15% for
Event-FedAvg. Exact modeled unicast communication is 23.706 Gbit
for either dense variant and 3.363 Gbit for Event-FedAvg (85.8% less).
The scheduled-versus-constant confidence interval includes zero; this
does not prove statistical equivalence.

The result rules out the *tested* polynomial-decay gain family as
a sufficient explanation of the compact-CIFAR Event-FedAvg advantage.
It does not isolate the event encoder's effect, optimize every possible
server schedule, or extend automatically to the other benchmarks.

## Reproducibility

- [Experiment protocol](https://github.com/j-cap/2026_NeuromorphicFL/blob/main/experiments/M3_DENSE_SCHEDULE_PROTOCOL.md)
- [Implementation](https://github.com/j-cap/2026_NeuromorphicFL/blob/main/experiments/m3_dense_schedule.py)
- [Development ranking](https://github.com/j-cap/2026_NeuromorphicFL/blob/main/experiments/results/m3_dense_schedule/development_ranking.csv)
- [Frozen choice](https://github.com/j-cap/2026_NeuromorphicFL/blob/main/experiments/results/m3_dense_schedule/frozen_selection.json)
- [Confirmation runs](https://github.com/j-cap/2026_NeuromorphicFL/blob/main/experiments/results/m3_dense_schedule/confirmation_runs.csv)
- [Paired comparisons](https://github.com/j-cap/2026_NeuromorphicFL/blob/main/experiments/results/m3_dense_schedule/paired_summary.csv)
