# P12 headline ten-seed extension protocol

## Purpose

P12 strengthens the uncertainty assessment for the operating points reported
in the manuscript's main comparison table. It adds seven independent paired
partition/training seeds to the three existing seeds. Development selection is
closed. No method, hyperparameter, training rule, endpoint, or traffic rule may
change after inspecting the extension results.

This is distinct from P11. P11 uses ten seeds to audit the alignment mechanism
on the Fashion-MNIST MLP. P12 evaluates the predictive and communication
results used in the headline comparison.

## Frozen scope

The following points are evaluated because they appear in Table I or support a
headline qualification immediately adjacent to it:

| Benchmark | Role | Frozen method/configuration |
|---|---|---|
| Fashion-MNIST MLP | Event operating point | Event-FedAvg, existing fixed setting |
| Fashion-MNIST MLP | Quality comparator | EF-TopK, fraction 0.05 |
| Fashion-MNIST MLP | Nearest-traffic comparator | Strom, threshold 0.02 |
| Fashion-MNIST CNN | Event operating point | Event-FedAvg, existing fixed setting |
| Fashion-MNIST CNN | Quality comparator | Strom, threshold 0.005 |
| Fashion-MNIST CNN | Nearest-traffic comparator | Strom, threshold 0.02 |
| CIFAR-10 CNN | Event operating point | Event-FedAvg, threshold 0.025 and initial quantum 0.005 |
| CIFAR-10 CNN | Quality comparator | Strom, threshold 0.0025 |
| CIFAR-10 CNN | Nearest-traffic comparator | Strom, threshold 0.01 |
| CIFAR-10 CNN | Worst-class qualification | EF-TopK, fraction 0.05 |
| CIFAR-10 CNN | Dense headline reference | dense FedAvg, server gain 2 |

All local training, model, partition, and communication settings remain those
already frozen in `EVIDENCE_FREEZE.md`.

## Paired seeds

The first three pairs reproduce the current paper evidence. The final seven
pairs are the extension and had not been used for method selection.

| Dataset | Partition seeds | Training-seed rule |
|---|---|---|
| Fashion-MNIST | 2500, 2600, 2700, **2800, 2900, 3000, 3100, 3200, 3300, 3400** | `70000 + partition_seed` |
| CIFAR-10 | 3500, 3600, 3700, **3800, 3900, 4000, 4100, 4200, 4300, 4400** | `80000 + partition_seed` |

The use of paired seeds means every compared method sees the same partition,
initialization, and minibatch stream within a dataset and seed index.

## Outputs and analysis

The campaign produces:

- all 110 seed-level rows;
- ten-seed means and sample standard deviations for accuracy, worst-class
  accuracy, cross-entropy, event count, and all traffic measures;
- paired Event-FedAvg-minus-comparator differences;
- two-sided 95% paired t intervals for all ten seeds and, separately, for the
  seven new seeds.

Table I will report mean plus sample standard deviation over all ten seeds. The
seven-seed subset is retained as the cleaner validation view because the
methods and configurations were fixed before those outcomes existed. P12 does
not introduce a new hyperparameter search or a new winner-selection step.

## Interpretation rule

Claims must follow the observed extension, including adverse or mixed results.
The manuscript may describe effect direction, variability, and paired
intervals. It must not claim broad statistical generality from ten seeds. If a
previous qualitative comparison reverses, the text and conclusion must be
revised rather than changing the comparator or configuration.

## Reproduction

The complete matrix is implemented in
`experiments/p12_headline_ten_seed.py` and executed by
`.github/workflows/p12_headline_ten_seed.yml`. The first command below validates
the immutable protocol without downloading data:

```bash
PYTHONPATH=src python experiments/p12_headline_ten_seed.py validate-protocol
PYTHONPATH=src python experiments/p12_headline_ten_seed.py point \
  --point-id fmnist_mlp_event --seed-index 3
```
