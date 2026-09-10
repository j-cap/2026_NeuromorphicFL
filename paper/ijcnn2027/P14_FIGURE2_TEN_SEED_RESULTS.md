# P14 Figure 2 ten-seed results

## Decision

P14 passes its acceptance gate. All 100 new runs completed, comprising ten
previously missing frozen operating points over the ten matched P12 partition
and training seed pairs. Combined with the 110 P12 runs, the complete Figure 2
evidence contains 210 seed-level runs: 21 unique operating points with exactly
ten seeds each.

No result was used to revise a hyperparameter or replace a displayed point.
The extension therefore changes the uncertainty estimates, not the frozen
comparison design.

## Missing-point ten-seed estimates

| Benchmark | Selection | Method | Accuracy [%] | Conservative unicast [Mbit] |
|---|---|---|---:|---:|
| Fashion-MNIST MLP | Quality | Strom | 81.43 +/- 0.78 | 1611.6 +/- 5.1 |
|  | Quality | Sign-EF | 81.27 +/- 0.30 | 435.9 +/- 0.0 |
|  | Quality | Dense FedAvg | 82.37 +/- 0.20 | 2487.0 +/- 0.0 |
|  | Nearest traffic | EF-TopK | 81.29 +/- 0.32 | 209.4 +/- 0.0 |
| Fashion-MNIST CNN | Quality | EF-TopK | 77.07 +/- 0.23 | 299.5 +/- 0.0 |
|  | Quality | Sign-EF | 76.25 +/- 0.22 | 133.5 +/- 0.0 |
|  | Quality | Dense FedAvg | 77.61 +/- 0.15 | 749.1 +/- 0.0 |
|  | Nearest traffic | EF-TopK | 75.81 +/- 0.19 | 63.9 +/- 0.0 |
| CIFAR-10 compact CNN | Quality | Sign-EF | 41.50 +/- 0.30 | 279.4 +/- 0.0 |
|  | Nearest traffic | EF-TopK | 41.25 +/- 0.25 | 135.3 +/- 0.0 |

Values are means over ten matched seeds. Sample standard deviations and all
other recorded metrics are retained in `summary.csv`.

## Interpretation

The expanded uncertainty assessment does not change the visual conclusion:
Event-FedAvg remains on the observed communication--accuracy frontier in all
three panels. It is not uniformly best in accuracy--the quality-selected Strom
CNN remains more accurate--but it occupies a favorable tradeoff relative to
the communication it uses.

The first three rerun means remain close to the historical three-seed
aggregates. Small deviations in threshold-based methods are consistent with
the documented last-bit threshold-branch sensitivity; P14 is now the single
internally consistent authority for Figure 2.

## Authoritative products

- `experiments/results/p14_figure2_ten_seed/heldout_runs.csv`: all 210 combined
  P12+P14 seed-level results;
- `experiments/results/p14_figure2_ten_seed/summary.csv`: 21 ten-seed
  aggregates;
- `paper/ijcnn2027/evidence/p14_figure2_ten_seed.csv`: checked manuscript-facing
  summary;
- `paper/ijcnn2027/figures/communication_frontier.pdf`: regenerated Figure 2.
