# Common-seed five-method core comparison

Repository: **j-cap/2026_NeuromorphicFL**, the research-code repository.
Entry point: `experiments/core_fresh_comparison.py`.
Results: `experiments/results/core_fresh_comparison_v1/`.

## Scope and fixed choices

Complete the numerical core on the ten partition/SGD pairs used in the fresh
Event/Ström confirmation. Run **30 new trajectories** for the three remaining
baseline methods and reuse the **20 completed Event/Ström trajectories** after
verifying their frozen configs, sources, receipts, metrics and traffic histories.
The completed pulse comparison is interpreted in
[`EVENT_STROM_FRESH_CONFIRMATION_RESULTS.md`](EVENT_STROM_FRESH_CONFIRMATION_RESULTS.md).

| Method | Fixed operating point | Execution |
|---|---|---|
| Tuned Event | Threshold 0.05, initial quantum 0.01 | Reuse ten fresh runs |
| Calibrated Ström | Threshold 0.05, native gain 0.10 | Reuse ten fresh runs |
| Dense FedAvg | Previously selected server gain `4*(1+r/100)^(-0.3)` | Ten new runs |
| STC | Upload/download fraction 1, server gain 2 | Ten new runs |
| EF-TopK | Global retained fraction 0.05, server gain 1 | Ten new runs |

Dense uses the frozen M3 schedule selected by final development training
objective. This retains the existing control for the server schedule.
STC uses the original frozen primary selected by training objective, with both
client and server error feedback. At p=1 it is dense ternary compression, not
full-precision Dense FedAvg. The earlier p=0.25 secondary point is outside this
five-method core; the command does not silently add another ten training runs.
EF-TopK uses the archived P3 quality setting selected by training CE. Event and
Ström use their completed validation-CE selections. These are explicit historical
selection policies, not an equal tuning-budget experiment or a new tuning sweep.

The driver checks the saved STC primary, Dense schedule and EF-TopK selection
records. It does not select configurations, horizons or seeds using the fresh
test curves. Validation models/checkpoints and older baseline trajectories from
different seed grids cannot be reused for the new runs.

## Matched learning and evaluation

All five methods use the compact CIFAR CNN (20570 parameters), ten fully
participating clients, strong 55%-dominant label skew, all 50000 training
images, the same 10000-image official test set, and **1800 communication
rounds**. Shared local training is five SGD steps per round, batch size 32,
learning rate 0.05 and L2 coefficient 0.0005. Initialization seed is 7777 and
the initialization scale is 1. Training diagnostics use 5000 training images.
Evaluate at round 1 and every 60 rounds, giving 31 predictive samples per run.

| Seed index | Partition seed | SGD seed for every method |
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

Thus each pair uses the same partition, initial model and minibatch RNG seed.
Freshness refers to these partition/SGD seeds relative to the earlier recorded
studies; the official test dataset has already been inspected. The completed
Event/Ström runs were inspected before this extension. All ten pairs remain in
every final summary; there is no favourable-seed or pilot exclusion.

## Lossless representation and accounting

Dense sends actual full-precision float32 update frames in both directions,
including a 64-bit frame header. Its scheduled gain is applied once at the
server, and clients reconstruct the same model from the decoded server update.

STC reuses the frozen native compressor and lossless STC representation. Its
server residual, ternarization, gain-before-server-compression rule and native
compressed server downlink are unchanged.

EF-TopK retains its native global `argpartition` support selection and weighted
client-delta residual rule. Each selected magnitude remains float32. A lossless
wire extension represents the sparse float32 vector using the shared support
codec plus sorted float32 values, with a dense fallback. The support frame,
mode, dimensions, lengths, padding and all magnitudes are charged. Signed zero
and subnormal bit patterns round-trip exactly. Omitting positive-zero entries
from this representation changes no vector value or residual update.

The EF-TopK server aggregates decoded client messages in the original client
order and applies its native gain once. It sends the smaller actual frame from
an exact aggregate-vector representation and length-prefixed ordered replay.
Both decoders reproduce the server update bit for bit. Clients verify exact
model reconstruction every round. This adds no lossy server compressor or
server residual. Native fixed-address/replay accounting remains available as
an additional representation of the same trajectory.

Synthetic native-runner regression checks compare model bits, residual bits,
RNG, sampled metrics and native traffic for all three methods. The wire extension
therefore changes representation costs while preserving the native algorithms.
The original package files and earlier experiment drivers remain frozen.

Primary communication accounting is **upload plus conservative-unicast
download**, including initial model synchronization, actual frame sizes,
32-bit client requests per round and replication to every client. Logical
broadcast is supplementary and uses the same convention as the completed pulse
study. Main representations are `both_codecs` for Event/Ström, `dense_float32`
for Dense, `lossless_stc` for STC and `lossless_ef_topk` for EF-TopK.

## Run locally and publish results

From Anaconda Prompt in the code repository:

```bat
cd /d C:\Users\WeberJ\2026_NeuromorphicFL
conda activate neuromorphicfl-resnet14
git pull --ff-only

python experiments/core_fresh_comparison.py verify
python experiments/core_fresh_comparison.py preflight
python experiments/core_fresh_comparison.py run --workers 2
python experiments/core_fresh_comparison.py analyze

git add experiments/results/core_fresh_comparison_v1
git commit -m "Add common-seed five-method core comparison results"
git push origin main
```

`verify` uses synthetic images and saved evidence, with no research training or
CIFAR download. It checks native fidelity, codec handling, result provenance,
complete-grid analysis, and exact resume/spawn execution on the first and last
seed for all three new methods. The original conda environment should give
`Runtime matches recorded Windows environment: True` in `preflight`. Training
also checks all six raw CIFAR batch fingerprints.

Two concurrent workers are suggested; use `--workers 1` to reduce CPU/RAM
concurrency. The CLI defaults to one worker if the option is omitted. Optional
`--point dense`, `--point stc` or `--point ef_topk` and `--seed-index 0` through
`9` limit execution to a subset. The final analysis requires all fifty outcomes.
The environment name does not change this compact CNN into ResNet14.

Check progress without training:

```bat
python experiments/core_fresh_comparison.py status
```

Repeat the run command after interruption to resume exact local checkpoints:

```bat
python experiments/core_fresh_comparison.py run --workers 2
python experiments/core_fresh_comparison.py analyze
git add experiments/results/core_fresh_comparison_v1
git commit -m "Add completed common-seed core comparison results"
git push origin main
```

The checkpoints preserve model/client-replica/residual state, RNG, round
counters and sampled predictive metrics. Changed configs, sources, runtime,
raw data or pulse reference evidence cannot be mixed into the campaign.
Numerical failures receive explicit receipts and are not automatically retried
with changed settings. Missing, invalid or failed outcomes block the final
paired summary and remain visible in `completion_audit.json`. Checkpoints and
locks remain local; CSV/JSON evidence and the comparison PNG/PDF are tracked.

## Analysis

Report round-1800 accuracy and CE, upload, download, total unicast cost,
supplementary logical-broadcast cost and training CE/L2 diagnostics for all five
methods. Four paired contrasts compare Event with each baseline, including the
previously completed Event/Ström contrast. Accuracy/CE differences are Event
minus reference; traffic ratios are Event divided by reference. Retain all ten
paired values and descriptive, unadjusted 95% Student-t intervals, with ratios
on the log scale. These are not formal equivalence/noninferiority tests or
simultaneously adjusted claims.

Accuracy targets remain 45%, 50% and 55%. Use first saved crossings with no
interpolation. Always report the total seed count, each method's hit count,
common hits, rounds and bits. Unreached targets remain censored. Cost ratios
condition on both methods reaching the target and are undefined with zero
common hits. Test curves do not control stopping or parameter selection.

Outputs include twenty reference receipts, thirty new complete/failure
receipts, four per-run CSVs for each new run, `protocol.json`,
`completion_audit.json`, `analysis_audit.json`, `comparison.csv`,
`comparison_history.csv`, `main_runs.csv`, `endpoint_summary.csv`,
`paired_per_seed.csv`, `paired_summary.csv`, `traffic_to_target.csv`,
`paired_target_summary.csv`, and `comparison.png` / `.pdf`. Reuse receipts link
to the original Event/Ström files; their raw CSVs are not copied or modified.

With the full grid complete, the analysis covers 90000 round records and 1550
sampled model fingerprints, including 36000 rounds/620 samples from the reused
pulse controls. Dense/STC/EF-TopK add 54000 rounds and 930 evaluated models.
