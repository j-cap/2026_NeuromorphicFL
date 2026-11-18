# Ten-seed Event/STC codec confirmation

Repository: **j-cap/2026_NeuromorphicFL**, the research-code repository.
Entry point: `experiments/event_codec_ten_seed.py`.

## Fixed question

Does the frozen Event codec retain its predictive-quality/communication tradeoff
against the two frozen STC operating points across the existing ten matched
partition and training seeds? The horizon is **1800 communication rounds** for
every method. There is no compressor tuning, change to the training rule, or
early stopping in this extension.

| Point | Training rule | Primary message representation |
|---|---|---|
| `event` | Event-FedAvg, persistent state/full reset, server gain 1 | Packed pulse positions on upload; exact aggregated downlink |
| `stc_primary` | STC p=1 on upload and download, server gain 2 | Lossless STC recoding |
| `stc_sparse` | STC p=0.25 on upload and download, server gain 2 | Lossless STC recoding |

These are the previously frozen operating points. The primary STC comparator is
the selected p=1 configuration. The p=0.25 point is a secondary comparator;
neither is a claim about all possible STC configurations.

Use the compact CIFAR CNN (**20570 parameters**), ten fully participating
clients, the full 50000-example training allocation with strong label skew, and
the fixed 5000-example training diagnostic subset. Each round uses five local
SGD steps, batch size 32, local learning rate 0.05 and L2 coefficient 0.0005.
Initialization remains seed 7777 with scale 1. The conda environment name does
not change this experiment's architecture to ResNet-14.

| Seed index | Partition seed | SGD seed | Work |
|---|---:|---:|---|
| 0 | 3500 | 83500 | Reuse all three completed full-horizon trajectories |
| 1–9 | 3600, 3700, …, 4400 | 83600, 83700, …, 84400 | Train three points per seed |

There are **30 point/seed results and 27 new training runs**. Seed 3500 is
referenced through verified receipts and hashes; its CSVs are not duplicated
or retrained. New seeds require execution of the codec host because the older
prediction summaries do not contain the actual pulse-position/message streams.
All methods share the partition and local-SGD seed within each seed pair.

We have already inspected seed 3500. Report both the full ten-seed summary
(`all_10`) and a separate summary over only the nine additional seeds
(`additional_9`). This records the distinction between the observed diagnostic
and the extension, without choosing seeds according to their outcomes.

## Measurements and analysis

Evaluate the unchanged CIFAR-10 test set at round 1 and every 60 rounds through
round 1800: **31 saved evaluations per trajectory**. Record test accuracy,
unregularized test cross-entropy, macro accuracy and worst-class accuracy.
The endpoint is the final round 1800, rather than a selected test checkpoint.
Training objective, training cross-entropy and derived L2 penalty are kept
separate from predictive performance.

The primary endpoint is the paired **Event minus STC p=1 final test accuracy**,
in percentage points. Additional paired endpoints are test cross-entropy,
training objective/L2 penalty, and upload, download, unicast-total and
broadcast-total traffic. Accuracy differences above zero favor Event;
Event/STC traffic ratios below one favor Event. Cost ratios are summarized as
geometric means of the within-seed ratios.

Report descriptive, unadjusted 95% paired Student-t intervals. Ratios use the
log scale. These intervals describe variation across partition/SGD seeds on
the fixed dataset and test set. They do not establish equivalence or
noninferiority, and secondary intervals have no multiple-comparison correction.

Traffic-to-accuracy targets stay at **45%, 50% and 55%**. Use the first saved
evaluation reaching each target, with no interpolation between samples.
Every seed remains in the coverage table. Unreached targets are censored at
1800 rounds and have no achievement cost. Paired cost ratios use only seeds
where both methods reach the target, and explicitly report Event hits,
reference hits and common-hit counts out of all ten (or nine) seeds. Such a
ratio is conditional on both reaching the target, so coverage must accompany it.

Record the same eight message representations as the single-seed experiment:
four Event representations and two for each STC point. Primary comparisons use
Event `both_codecs` and STC `lossless_stc`. The native representations provide
within-trajectory coding comparisons; they do not require extra training.
Count actual bytes, framing/padding, initialization, control requests and
receiver replication under the existing unicast and logical-broadcast models.

## Exactness and recovery

The existing codec host and full-horizon evaluator are imported unchanged.
Every message is encoded and decoded; the synchronized client model must remain
bitwise equal to the host update. Event retains integer-count roundoff
corrections or its ordered fallback. STC retains its native selection,
magnitudes, server gain and both residuals.

Prediction evaluations are saved atomically with the model, RNG, residuals,
per-round counters and training history. Checkpoints occur at round 1 and then
every 60 rounds. Repeating `run` skips verified completions and resumes unfinished
trajectories from the latest complete checkpoint. Numerical failures are
recorded explicitly and block the complete-grid analysis; they are not dropped
or automatically rerun with a different seed.

Sources, runtime, raw data batches and reused evidence are fingerprinted. Use
the original Windows environment and CIFAR batches from the codec pilot and
single-seed full-horizon run. No new package module is added, preserving the
frozen pilot's package-source inventory. New output is isolated from all older
results and checkpoints.

The ten archived Event trajectories are mandatory regression references.
Training/test curves must match their printed precision (rtol 1e-8, atol 1e-9),
and native upload prefixes must match exactly. Original archived full model
files are unavailable; this curve check is distinct from the live bitwise
message/client-replica checks. If matching native STC confirmation receipts are
present locally, their prediction curves and upload/server prefixes are also
checked. Pending native STC runs do not need to finish for this campaign to run.

## Run locally

From Anaconda Prompt in the **code** repository:

```bat
cd /d C:\Users\WeberJ\2026_NeuromorphicFL
conda activate neuromorphicfl-resnet14
git pull --ff-only
python experiments/event_codec_ten_seed.py verify
python experiments/event_codec_ten_seed.py preflight
python experiments/event_codec_ten_seed.py run --workers 2
python experiments/event_codec_ten_seed.py analyze

git add experiments/results/event_codec_ten_seed_v1
git commit -m "Add ten-seed Event and STC codec confirmation results"
git push origin main
```

`verify` uses synthetic inputs and the existing read-only evidence; it performs
no CIFAR training or download. It checks seed mapping, reference-only reuse,
receipt validation, paired/censored summaries, the complete analysis path,
exact checkpoint continuation and two spawned seed workers.

The preflight should print `Runtime matches full-horizon run: True` on the
original Windows environment. `run` additionally checks the actual data batches
and refuses execution if the frozen runtime/data contract differs. Use
`--workers 1` if two concurrent jobs leave insufficient CPU/RAM for other work.

Progress and restart use the same driver:

```bat
python experiments/event_codec_ten_seed.py status
python experiments/event_codec_ten_seed.py run --workers 2
python experiments/event_codec_ten_seed.py analyze
git add experiments/results/event_codec_ten_seed_v1
git commit -m "Add completed ten-seed codec confirmation results"
git push origin main
```

For a selected point/seed, use e.g.
`python experiments/event_codec_ten_seed.py run --point event --seed-index 1 --workers 1`.
Index 1 is partition 3600 / SGD 83600. Subsequent unfiltered `run` completes the
remaining grid. Index 0 only registers reuse and never enters the training queue.

## Outputs

Directory: `experiments/results/event_codec_ten_seed_v1/`.

- `protocol.json`: fixed settings, source/runtime/data hashes, ten Event reference
  identities and the three reused full-horizon result identities.
- Three seed-zero receipts referencing original evidence; no duplicate CSVs.
- For each new point/seed: 1800-row `rounds.csv`, 31-row `evaluation.csv`,
  representation-level `history.csv`, endpoint `summary.csv` and a receipt.
- `completion_audit.json`: status of all 30 required point/seed results, written
  before the complete-grid analysis and retained if it is blocked.
- `comparison.csv`: 80 representation/seed endpoints; `comparison_history.csv`:
  2480 representation/seed/sample rows.
- `main_runs.csv`: the 30 optimized-codec endpoints with training L2 diagnostics.
- `endpoint_summary.csv`: per-method mean, seed standard deviation and count
  for the all-ten and additional-nine scopes.
- `paired_per_seed.csv`: paired differences/ratios; `paired_summary.csv`:
  estimates and intervals for both scopes.
- `traffic_to_target.csv`: 240 representation/seed/target outcomes, including
  unreached targets; `paired_target_summary.csv`: conditional cost ratios and
  complete hit coverage for both scopes.
- `analysis_audit.json`: evidence checks and regression-reference availability.
- `confirmation.png` and `.pdf`: accuracy versus rounds/traffic, paired endpoint
  accuracy intervals and target-cost ratios with common-hit counts.

Root CSV/JSON evidence and these two figures are Git-enabled. Checkpoints and
lock files remain ignored. Codec audit timings include reconstruction and
reference-encoding work and are not a deployment timing benchmark.
