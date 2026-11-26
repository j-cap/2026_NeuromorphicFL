# Fashion-MNIST MLP calibration: results and confirmation decision

Evidence: `experiments/results/fmnist_mlp_calibrated_v1/`, user-results commit
`8f51dbef864a143b0fbb9c7570f2755cafef5eec`. Protocol and local commands:
[`FMNIST_MLP_CALIBRATED.md`](FMNIST_MLP_CALIBRATED.md).

## Decision

**Freeze the five validation-selected configurations and proceed to the already
prepared ten-seed confirmation.** The completed calibration supports a promising
communication/quality operating point for Event. It covers a single development
partition used for selection; the predictive test comparison remains the next
stage.

All **90 screens and 15 continuations are complete and finite**. Every family
received 18 screens of 500 rounds and three exact continuations to 1,500 rounds:
12,000 unique rounds per family, 60,000 in total. Promotion and final selection
reproduce exactly from validation CE, with the frozen traffic/identifier tie
breakers. There are no completed confirmation receipts in this checkpoint.

## Selected validation endpoints

These are single-run development results after 1,500 rounds, not means across
seeds. Accuracy is measured on the 6,000-image validation split. Traffic is total
bidirectional conservative unicast, including serialized frames, padding,
initialization, per-client requests and ten recipient copies. **1 Gbit = 10^9
bits.** Every method shares the training partition and local-learning protocol.

| Method | Validation accuracy, % | Validation CE | Uplink, Gbit | Synchronization/control, Gbit | Total unicast, Gbit |
|---|---:|---:|---:|---:|---:|
| Event | 87.77 | 0.340446 | 0.068605 | 0.420511 | **0.489116** |
| Ström-style | 88.07 | 0.340460 | 0.086185 | 0.491690 | 0.577875 |
| Dense FedAvg | 87.77 | 0.341165 | 12.393600 | 12.402342 | 24.795942 |
| STC | 87.70 | 0.346582 | 0.393404 | 0.401862 | 0.795266 |
| EF-TopK | 87.43 | 0.346587 | 1.466154 | 8.269867 | 9.736021 |

The synchronization/control column follows the existing panel's
`unicast_downlink_bits` convention and includes initialization and request
costs. The primary total accounts for every component.

Event uses the least total traffic among the five selected points. Its
validation CE differs from Ström by only **-0.0000143**, so that numerical
ranking should not be presented as a meaningful quality advantage. Ström has
**0.30 percentage points higher validation accuracy**, at greater communication.
No confidence interval or equivalence claim can be inferred from one selected
development run.

| Reference | Event's total unicast saving, % | Event-minus-reference accuracy, pp | Event-minus-reference CE |
|---|---:|---:|---:|
| Ström-style | 15.36 | -0.300 | -0.0000143 |
| Dense | 98.03 | 0.000 | -0.0007190 |
| STC | 38.50 | +0.067 | -0.0061361 |
| EF-TopK | 94.98 | +0.333 | -0.0061407 |

These percentages compare the two complete 1,500-round trajectories. They are
neither paired-seed estimates nor cost-to-test-target results. The 80%, 85% and
87% targets remain reserved for the predeclared confirmation analysis.

The directional qualification remains relevant: versus selected STC, Event
uploads **82.56% fewer bits** but uses **4.64% more synchronization/control
traffic**. The resulting total saving is 38.50%; the gain does not hold in
every traffic component.

## Frozen configurations

| Method | Selected setting |
|---|---|
| Event | Threshold 0.025, initial quantum 0.005, decay exponent 0.1; leak 0.999, persistent state, full reset, gain one |
| Ström-style | Threshold 0.02, native gain 0.3; FedAvg-equivalent gain three at local LR 0.1 |
| Dense | Initial gain two, decay exponent 0.3, scale 100 |
| STC | Upload/download retention one, gain two; ternarization and both residuals retained |
| EF-TopK | Global retained fraction 0.1, gain two |

Each is the lowest-final-validation-CE point among its three continued
finalists. All families have the same bounded accumulated budget, with
method-specific grid geometry. Ström selects the largest screened threshold;
Dense selects the largest screened decay exponent. These are bounded-search
winners rather than proven global optima. The next step uses the agreed frozen
settings and budget.

The Event setting also has lower CE and total traffic than its other two
continued settings. The lowest-CE EF finalist retains 10% of coordinates;
the continued 2.5%-retention alternatives have lower communication but higher
validation CE. Thus the confirmation will compare the preselected quality
points, rather than complete rate/quality frontiers.

## Verification

`fmnist_mlp_development_audit.py` checks the original source contract without
requiring the Windows numerical runtime for read-only analysis. It reproduces
all selections, checks the exact screening prefixes, and independently sums
integer frame counters at every saved validation checkpoint.

Coverage: **105 complete development stages, 67,500 saved round records
(including continued screening prefixes), and 1,455 saved validation model
fingerprints**. It verifies packet byte alignment, dense frame sizes, pulse
header/portfolio constraints, train/validation allocation, receipt hashes,
model identities and final metrics. The 533 original calibration files remain
unchanged. It does not rerun training, load official test images or infer
ten-seed predictive performance.

Contract SHA-256:
`0b8838e535fccd9e70e3babe4416b7f43f7b335c13e5393ad723913ffff717d8`.
Selection SHA-256:
`104950f27d8c601f9fbf31ba92059859756a3869e2e2a661aadac588728f5703`.

Derivative tables and the audit inventory use the
`analysis_development_` prefix; the study drivers, original outputs and
CIFAR panel freezes are untouched.

## Run confirmation locally

Use the same numerical environment as tuning. The new analysis files do not
alter the frozen training-source contract. `confirm` verifies the existing
selection, resumes any compatible confirmation checkpoints and skips completed
fits. It trains **all five methods on ten common seeds**, using all 60,000
training images, the official test set and the fixed 1,500-round endpoint.
It also produces the final comparative tables and plots automatically.

```bat
cd /d C:\Users\WeberJ\2026_NeuromorphicFL
conda activate neuromorphicfl-resnet14
git pull --ff-only
python experiments/fmnist_mlp_calibrated.py confirm --workers auto

git add experiments/results/fmnist_mlp_calibrated_v1
git diff --cached --check
git commit -m "Add Fashion-MNIST MLP ten-seed five-method confirmation"
git push origin main
```

To independently recheck this calibration audit from saved files:

```bat
python experiments/fmnist_mlp_development_audit.py --check
```
