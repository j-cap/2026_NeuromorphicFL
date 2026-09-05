# P8 targeted-revision audit

## Gate decision

P8 is **closed**. The targeted campaign resolves the P7 dense-FedAvg fairness
blocker, isolates two operator choices, and supplies the requested manuscript
clarifications without reopening the conference scope. T5 is not required.

The authoritative campaign is GitHub Actions run
[33880856470](https://github.com/j-cap/2026_NeuromorphicFL/actions/runs/33880856470).
Its code entered the branch in commit
1a74846ee1874937d6ba25ed59cbafd0009f048b. Selection and held-out artifacts
are frozen under experiments/results/p8_targeted_revision/.

## Development-first design

- Development partition: 3400; training seed: 83400.
- Held-out partitions: 3500, 3600, and 3700, paired with training seeds
  83500, 83600, and 83700.
- Dense server gains: 0.25, 0.5, 0.75, 1, 1.5, 2, and 4.
- Dense selection metric: minimum final development training cross-entropy.
- Frozen mechanism variants: the P3 Event-FedAvg operating point,
  no leakage (rho=1), and a coupled quantum (q_r=theta=0.025).

The development rule selected dense server gain 2. Held-out values were not
used to select the gain or mechanism variants.

## Held-out findings

| Configuration | Accuracy [%] | Worst class [%] | Total [Mbit] |
|---|---:|---:|---:|
| Dense FedAvg, gain 2 | 44.13 ± 1.12 | 23.3 ± 5.6 | 1586.6 ± 0.0 |
| Event-FedAvg, P8 rerun | 48.25 ± 1.07 | 26.7 ± 7.8 | 200.2 ± 2.2 |
| Event-FedAvg, rho=1 | 48.32 ± 1.05 | 25.4 ± 5.4 | 202.1 ± 5.6 |
| Event-FedAvg, q=theta | 18.42 ± 7.33 | 0.0 ± 0.0 | 44.8 ± 17.7 |

The manuscript retains the original frozen P3 Event-FedAvg aggregate
(48.06 ± 1.12%, 201.5 ± 4.3 Mbit) as its primary result. Against the
P8-tuned dense baseline, that frozen result is 3.93 accuracy points higher at
12.7% of dense traffic.

Leakage at rho=0.999 has no material isolated effect relative to rho=1 in
this audit: the accuracy difference is 0.06 point and the traffic ratio is
1.009. The neuromorphic interpretation is therefore narrowed to persistent
threshold-and-reset event state; the evidence does not attribute the operating
point to weak leakage. Coupling the server quantum to the trigger loses 29.84
points relative to the P8 Event rerun. This supports independent trigger and
update resolutions at the selected operating point, not a universal statement
that all coupled schedules fail.

## Rerun sensitivity

The P8 Event-FedAvg rerun exactly reproduces the frozen P3 accuracy,
worst-class accuracy, and traffic for partitions 3600 and 3700. Partition
3500 follows a nearby threshold branch: P3 reports 47.66% and 206.316 Mbit,
whereas P8 reports 48.25% and 202.472 Mbit. Threshold decisions can amplify
last-bit floating-point differences, as already documented in the T4 report.
To avoid post-hoc replacement, the paper keeps the frozen P3 aggregate and
uses the P8 rerun only inside the like-for-like mechanism audit. This
cross-platform branch sensitivity remains an explicit P9 reproducibility item.

## P7 finding closure

- **FL-B1:** closed by development-only dense server-gain selection and
  held-out evaluation.
- **FL-I1:** “traffic-matched” display wording is replaced by
  “development-selected nearest-traffic.”
- **FL-I2:** the main table exposes worst-class accuracy and names the
  CIFAR-10 EF-TopK qualification.
- **FL-I3/FL-I4:** the fixed strong-skew template and three-seed uncertainty
  boundary are explicit.
- **TH-I1/TH-I2:** the empirical alignment statistic is defined as a ratio of
  weighted sums and remains a finite-trajectory diagnostic.
- **NN-I1:** the compact leakage/quantum audit is reported, including its
  negative leakage finding.
- **NN-I2:** novelty wording is limited to the dated targeted search.

## Validation contract

paper/ijcnn2027/build_evidence.py checks the P8 selection, the 12-run held-out
grid, the frozen seed mapping, and the generated evidence. The main CIFAR-10
evidence view replaces only the quality-view dense row with the P8-tuned
baseline; the primary Event-FedAvg row remains sourced from P3.
