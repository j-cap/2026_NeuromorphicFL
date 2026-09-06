# P12 headline ten-seed results

## Decision

P12 passes its predeclared gate. The single pinned campaign completed all 110
runs: eleven frozen method/configuration points over ten paired partition and
training seeds. The seven added pairs preserve every headline direction. The
manuscript therefore reports the ten-seed means and sample standard deviations
and gives paired 95% t intervals for the new-seven validation subset.

## Ten-seed headline estimates

| Benchmark | Point | Accuracy [%] | Worst class [%] | Conservative unicast [Mbit] |
|---|---|---:|---:|---:|
| Fashion-MNIST MLP | Event-FedAvg | 83.05 +/- 0.42 | 55.0 +/- 5.9 | 182.8 +/- 2.4 |
|  | EF-TopK quality | 82.40 +/- 0.24 | 52.9 +/- 3.1 | 1010.5 +/- 0.0 |
|  | Strom nearest traffic | 78.82 +/- 1.62 | 40.3 +/- 12.0 | 181.5 +/- 4.3 |
| Fashion-MNIST CNN | Event-FedAvg | 81.20 +/- 0.49 | 42.5 +/- 7.4 | 72.1 +/- 1.0 |
|  | Strom quality | 82.07 +/- 0.77 | 38.8 +/- 11.4 | 357.1 +/- 5.5 |
|  | Strom nearest traffic | 79.06 +/- 1.30 | 23.8 +/- 10.9 | 81.9 +/- 1.5 |
| CIFAR-10 CNN | Event-FedAvg | 48.13 +/- 0.89 | 26.0 +/- 4.0 | 201.0 +/- 4.0 |
|  | Strom quality | 46.98 +/- 0.84 | 20.8 +/- 6.3 | 927.5 +/- 6.6 |
|  | Strom nearest traffic | 34.87 +/- 4.41 | 9.8 +/- 6.0 | 214.0 +/- 17.4 |
|  | EF-TopK qualification | 42.32 +/- 0.22 | 24.9 +/- 1.9 | 645.2 +/- 0.0 |
|  | Dense FedAvg, gain 2 | 43.88 +/- 1.38 | 20.8 +/- 4.2 | 1586.6 +/- 0.0 |

## Predeclared seven-seed validation

Differences are Event-FedAvg minus the named comparator in percentage points.

| Comparison | Mean difference | 95% paired t interval | Positive pairs |
|---|---:|---:|---:|
| Fashion-MNIST MLP vs EF-TopK quality | +0.69 | [0.25, 1.12] | 6/7 |
| Fashion-MNIST MLP vs Strom nearest traffic | +4.56 | [2.63, 6.49] | 7/7 |
| Fashion-MNIST CNN vs Strom quality | -0.86 | [-1.66, -0.05] | 1/7 |
| Fashion-MNIST CNN vs Strom nearest traffic | +2.38 | [1.00, 3.77] | 7/7 |
| CIFAR-10 vs Strom quality | +1.14 | [0.09, 2.19] | 6/7 |
| CIFAR-10 vs Strom nearest traffic | +14.75 | [10.75, 18.74] | 7/7 |
| CIFAR-10 vs dense FedAvg, gain 2 | +4.38 | [3.18, 5.59] | 7/7 |

The CIFAR-10 worst-class qualification remains unresolved. Across all ten
pairs, Event-FedAvg minus EF-TopK is +1.16 points with interval
[-1.72, 4.04]. This result does not support a uniform class-wise superiority
claim.

## Reproduction boundary

P12 reruns all ten seeds in one pinned environment rather than combining seven
new runs with historical aggregates. The first three Event-FedAvg CIFAR-10
runs reproduce the historical P3 summary exactly at displayed precision. Some
threshold-based baseline runs, and the Fashion-MNIST reruns, are statistically
consistent but not bitwise identical to their historical artifacts. This is
the previously documented last-bit threshold-branch sensitivity: tiny numeric
changes can alter a pulse decision and then the later trajectory. The current
headline table is therefore frozen from the internally consistent P12
campaign; the P1/P3 artifacts remain the historical audit trail.

## Authoritative products

- `experiments/results/p12_headline_ten_seed/heldout_runs.csv`: 110 seed-level
  results.
- `experiments/results/p12_headline_ten_seed/summary.csv`: ten-seed aggregates.
- `experiments/results/p12_headline_ten_seed/paired_differences.csv`: all-ten
  and new-seven paired differences.
- `experiments/results/p12_headline_ten_seed/protocol.json`: frozen matrix and
  environment record.
- `paper/ijcnn2027/evidence/p12_headline_ten_seed.csv` and
  `p12_paired_differences.csv`: manuscript-facing checked copies.

`build_evidence.py --check`, `build_visuals.py --check`, and
`check_manuscript_claims.py` jointly prevent the raw results, displayed table,
figure, and narrative values from drifting apart.
