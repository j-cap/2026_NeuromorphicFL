# Tuned Event versus calibrated Ström: fresh paired confirmation

Repository: **j-cap/2026_NeuromorphicFL**, the research-code repository.
Entry point: `experiments/event_strom_fresh_confirmation.py`.
Results: `experiments/results/event_strom_fresh_confirmation_v1/`.

## Question and frozen methods

Does the validation-selected Event setting retain its predictive performance
and communication benefit against calibrated Ström across fresh partition/SGD
seeds after refitting on the full training set?

| Setting | Tuned Event | Calibrated Ström |
|---|---:|---:|
| Threshold | 0.05 | 0.05 |
| Initial Event quantum | 0.01 | Coupled to threshold |
| Native server gain | 1 | 0.10 |
| FedAvg-equivalent Ström gain | — | 2, with local learning rate 0.05 |
| Pulse state | Persistent leaky state, full reset | Nonleaky residual, subtractive reset |

Event retains rho 0.999 and its existing quantum decay with exponent 0.2 and
scale 100. Ström retains at most one pulse per coordinate/client/round. Both
use the frozen lossless position encoder and exact aggregate-downlink codec
portfolio. Only the already selected Event threshold/initial quantum differ
from the previous Event confirmation.

The Event development study is interpreted in
[`EVENT_CODEC_DEVELOPMENT_RESULTS.md`](EVENT_CODEC_DEVELOPMENT_RESULTS.md).
Its quality and Ström-budget selection roles chose the same configuration.
The driver verifies the saved split, receipts, source hashes, selected configs
and development decisions before starting confirmation. There is no new tuning
grid or automatic extension at the quantum-grid boundary.

## Twenty new runs on ten fresh pairs

Each method starts from the shared initialization on **all ten pairs**, giving
**20 new 1800-round trajectories**. No previous full-training result or
development model/checkpoint is reused, including for the first pair. The
development split contained 45000 training and 5000 validation images; this
study refits on all **50000 training images** and evaluates on the separate
**10000-image official test set**.

| Seed index | Partition seed | SGD seed for both methods |
|---:|---:|---:|
| 0 | 6000 | 86000 |
| 1 | 6100 | 86100 |
| 2 | 6200 | 86200 |
| 3 | 6300 | 86300 |
| 4 | 6400 | 86400 |
| 5 | 6500 | 86500 |
| 6 | 6600 | 86600 |
| 7 | 6700 | 86700 |
| 8 | 6800 | 86800 |
| 9 | 6900 | 86900 |

The pre-campaign scan of recorded JSON partition-seed metadata at commit
`838167711b474764b5056c7a7270bbe486b06904` found prior seeds 2500–5100 in
increments of 100, including ResNet results using 4600–5100. The new grid has
no overlap with that snapshot. This defines freshness relative to recorded
experiment metadata; these are new partition/SGD seeds on the same previously
inspected official test dataset. They do not create a blind new test dataset.
All ten pairs enter the summary; no pilot seed is excluded.

All runs use the **compact CIFAR CNN (20570 parameters)**, ten fully
participating clients, strong 55%-dominant label skew, initialization seed
7777, five local SGD steps, batch size 32, local learning rate 0.05, and L2
coefficient 0.0005. The paired methods receive the same partition, initial
model and minibatch RNG seed. Training diagnostics use 5000 training images.
Evaluation is saved at round 1 and every 60 rounds through round 1800, giving
31 samples per run. These are **communication rounds**, not epochs. The conda
environment name does not change the model into ResNet14.

## Run locally and publish results

From Anaconda Prompt in the code repository:

```bat
cd /d C:\Users\WeberJ\2026_NeuromorphicFL
conda activate neuromorphicfl-resnet14
git pull --ff-only

python experiments/event_strom_fresh_confirmation.py verify
python experiments/event_strom_fresh_confirmation.py preflight
python experiments/event_strom_fresh_confirmation.py run --workers 2
python experiments/event_strom_fresh_confirmation.py analyze

git add experiments/results/event_strom_fresh_confirmation_v1
git commit -m "Add fresh tuned Event and Strom confirmation results"
git push origin main
```

`verify` checks bookkeeping and analysis on synthetic data, then compares
uninterrupted versus resumed execution for both methods on the first and last
fresh seeds, including spawned workers. It performs no CIFAR research training
or download. In the original conda environment, `preflight` should report
`Runtime matches recorded Windows environment: True`. Training also verifies
the six raw CIFAR training/test batch fingerprints.

The run command queues all missing method/seed pairs. Two concurrent workers
are suggested; use `--workers 1` to reduce CPU/RAM concurrency. The CLI defaults
to one worker if the option is omitted. Optional `--point event` or
`--point strom` and `--seed-index 0` through `9` restrict execution to a subset;
they do not change scientific settings. The final analysis requires all twenty
results. Use the same result commit commands after publishing a completed subset.

Check progress without training:

```bat
python experiments/event_strom_fresh_confirmation.py status
```

After interruption, repeat the run command. Existing valid completed receipts
are skipped, and unfinished local checkpoints resume the exact trajectory:

```bat
python experiments/event_strom_fresh_confirmation.py run --workers 2
python experiments/event_strom_fresh_confirmation.py analyze
git add experiments/results/event_strom_fresh_confirmation_v1
git commit -m "Add completed fresh Event and Strom confirmation results"
git push origin main
```

Local NPZ checkpoints retain the model, client replica, pulse/residual state,
RNG, traffic records and sampled predictive metrics. Provenance includes the
method, fresh seeds, selected config and campaign/source hashes. Source,
runtime, raw-data or frozen-selection changes cannot be mixed into this
campaign. Checkpoints and locks remain local and ignored by Git; scientific
CSV/JSON evidence and the confirmation PNG/PDF are tracked.

Numerical failures receive explicit receipts and are not automatically retried
with altered settings. Missing, invalid or failed runs block the final paired
summary and remain visible in `completion_audit.json`. Do not change thresholds,
quantum, gains, horizon or seed inclusion after inspecting these test results.

## Analysis and interpretation

The **primary predictive endpoint** is Event minus calibrated Ström
round-1800 test accuracy, in percentage points. Always report final test
cross-entropy alongside accuracy. Additional endpoints are regularized training
objective, its CE/L2 separation, upload, download and total traffic.

**Total unicast traffic** under the `both_codecs` representation is the primary
communication measure. Logical-broadcast traffic is supplementary. Per-seed
paired differences use Event minus Ström; traffic ratios use Event divided by
Ström. A cost ratio above one means Event uses more communication. The analysis
does not enforce a favourable direction.

Report all ten pairs with descriptive, unadjusted paired 95% Student-t intervals;
cost ratios are summarized on the log scale. These intervals describe
partition/SGD variation on this fixed dataset. They do not establish formal
accuracy equivalence, noninferiority or field-wide state of the art. A final
accuracy interval containing zero is not evidence of equivalent accuracy.

Communication to 45%, 50% and 55% test accuracy uses the **first saved
crossing**, without interpolation. Record rounds and bit cost separately.
Keep each method's reached/not-reached status and censor horizon. Target-cost
ratios condition on both methods reaching the target; always report the total
number of pairs, each method's hit count and common-hit count. With no common
hits, the ratio is undefined. These retrospective crossings do not select a
best test checkpoint or implement early stopping.

## Outputs

- `protocol.json`: frozen configs, selection provenance, seed audit snapshot,
  source/runtime/data hashes and statistical rules.
- Twenty fresh receipts and per-run round, evaluation, summary and history CSVs,
  retaining model fingerprints and actual codec counters.
- `completion_audit.json`: all twenty expected outcomes, including failures.
- `comparison.csv`, `comparison_history.csv`: all four encoding variants.
- `main_runs.csv`, `endpoint_summary.csv`: `both_codecs` endpoints and descriptive
  mean/std/count summaries, including CE/L2 diagnostics.
- `paired_per_seed.csv`, `paired_summary.csv`: eight endpoint contrasts and
  their paired intervals, with the primary predictive endpoint flagged.
- `traffic_to_target.csv`, `paired_target_summary.csv`: sampled first crossings,
  censoring, conditional cost ratios and complete hit coverage.
- `analysis_audit.json`, `confirmation.png` / `.pdf`: verification scope and
  descriptive accuracy, CE and communication plots.

This focused confirmation compares the two frozen pulse methods. Dense FedAvg,
STC and EF-TopK remain relevant to the final numerical core; this experiment
does not rerun or retune them.
