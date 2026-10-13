# M3 — Scheduled dense FedAvg fairness control

## Question
Could Event-FedAvg's observed accuracy advantage over dense FedAvg result partly
from comparing a scheduled server quantum with a fixed server gain?

The dense method now optionally uses
`gamma_r = gamma_0 (1 + r/100)^(-alpha_gamma)`, preserving `alpha_gamma=0`
as the previous exactly constant baseline. This is an optimizer-only change;
dense full-precision uploads and exact synchronization accounting are unaltered.

## Frozen protocol
- CIFAR-10 compact CNN, 10 clients, strong label skew; 1,800 rounds and
  five local SGD steps, otherwise frozen P19 hyperparameters.
- Development partition seed **4500** (training seed 84500) is outside the
  P19 final ten-seed set (3500–4400).
- Development candidates: `gamma_0 in {1,2,4}` and
  `alpha_gamma in {0,0.1,0.2,0.3,0.5}`, each evaluated for all 1,800 rounds.
- Select the minimum **final development training objective** only.
  Test accuracy must not be used to select the schedule.
- Freeze `frozen_selection.json` and run ten matched final seeds; compare
  against already committed P19 constant dense, Event-FedAvg, and EF-TopK.
- Point/seed outputs are resumable; use `--workers 1` under memory pressure.

## Workstation commands

```bat
git pull
conda activate neuromorphicfl-resnet14
python experiments\m3_dense_schedule.py smoke
python experiments\m3_dense_schedule.py develop --workers 2
python experiments\m3_dense_schedule.py select
python experiments\m3_dense_schedule.py confirm --workers 2
python experiments\m3_dense_schedule.py verify
python experiments\m3_dense_schedule.py analyze
python experiments\m3_dense_schedule.py status
```

Outputs: `experiments/results/m3_dense_schedule/`, including per-run summary
and history files, development ranking, frozen selection, confirmation runs,
and seed-paired confidence intervals.

The experiment controls server-gain scheduling, **not** the causal effect of
coordinate encoding, threshold reset, or server aggregation in isolation.
A paired accuracy difference that is not statistically distinguishable from
zero does not by itself establish statistical equivalence.
