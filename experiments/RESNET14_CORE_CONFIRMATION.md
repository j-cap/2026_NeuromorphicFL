# ResNet-14: selected five-method common-seed confirmation

Repository: **j-cap/2026_NeuromorphicFL**, the research-code repository.
Entry point: `experiments/resnet14_core_confirmation.py`.
Results: `experiments/results/resnet14_core_confirmation_v1/`.

Status: all thirty new fits and twenty verified reuses completed. The
[results and independent audit](RESNET14_CORE_CONFIRMATION_RESULTS.md) report
the five-method test comparison, traffic-to-target results and paper scope.

## Scope and fixed choices

Complete the original three-baseline panel after the
[validation calibration](RESNET14_BASELINE_DEVELOPMENT_RESULTS.md). Run exactly
**30 new full-training fits** and reuse all **20 completed Event/Ström outcomes**.
The selected STC is already **p=1, gain 4**; it is one configuration, not two.
No sparse-STC secondary points or new tuning are included in this campaign.

| Method | Fixed ResNet setting | Work |
|---|---|---|
| Event | Threshold 0.01, initial quantum 0.0025, leak 0.999, gain 1 | Reuse ten completed outcomes |
| Ström | Threshold 0.005, native gain 0.10 / FedAvg-equivalent gain 2 | Reuse ten completed outcomes |
| Dense FedAvg | Constant server gain 1.5 | Ten new fits |
| EF-TopK | Global retained fraction 0.01, server gain 4 | Ten new fits |
| STC | Upload/download retained fractions 1, server gain 4 | Ten new fits |

The baseline point IDs are `dense_g1p5_a0`, `ef_topk_p0p01_g4` and
`stc_u1_d1_g4`. They are the frozen round-1800 validation-CE winners from the
600-round screen / top-three continuation design. The driver reconstructs
the frozen promotion, selection, configurations and source identities before
reading existing pulse test results or parsing new test images. Test scores
do not select methods, configurations, seeds or stopping times.

STC p=1 still ternarizes each parameter tensor with client and server error
feedback; it is not full-precision Dense FedAvg. EF retains its native global
TopK selection, weighted client error feedback and float32 values. All three
use the frozen native NumPy float32 compressor/server conventions, with the
same Torch local SGD and model application as the pulse experiment. Lossless
representations preserve each method's native update; they do not add a new
lossy server operator to EF or Dense.

The runner uses the [faster CPU codec](RESNET14_CODEC_SPEED.md), preserving the
exact original message bytes and modes. The acceleration sources/backend are
pinned in the confirmation contract. The frozen optimizer, package files,
calibration drivers and original baseline host remain unchanged.

## Same network, learning problem and ten seed pairs

All methods use **CIFAR-10 Torch ResNet-14 with GroupNorm, 175258 parameters**,
ten fully participating clients and strong 55%-dominant label skew. Shared
local training is five SGD steps per round, batch 32, learning rate 0.05,
L2 coefficient 0.0005 and initialization seed 7777. Training diagnostics use
the same 5000-image subset within each seed. Every new fit starts from
initialization on **all 50000 training images**, evaluates the fixed official
10000-image test set, and runs to **1800 communication rounds**. Evaluate at
round 1 and every 60 rounds, giving 31 saved predictive samples per fit.

| Partition seed | SGD seed for every method |
|---:|---:|
| 7100 | 97100 |
| 7200 | 97200 |
| 7300 | 97300 |
| 7400 | 97400 |
| 7500 | 97500 |
| 7600 | 97600 |
| 7700 | 97700 |
| 7800 | 97800 |
| 7900 | 97900 |
| 8000 | 98000 |

The development fits on seed 7000 and the 45000/5000 split are not eligible
confirmation results. **All ten seeds are new fits for each added baseline.**
Neither development checkpoints nor compact-CNN results initialize or replace
them. The full-training client allocation, diagnostic subset and data hashes
must match the completed pulse experiment exactly.

Work is **54000 new rounds / 930 new evaluation points**, giving **50 total
outcomes / 90000 verified round records / 1550 sampled model fingerprints**
including reuse. This is a fixed-horizon experiment; these counts are not
epochs or elapsed-time guarantees. Execution is sequential on one GPU, using
the recorded Windows environment, four Torch threads, evaluation batches of
256, deterministic operations and no mixed precision or TF32.

## Read-only reuse, resume and failures

Reuse all twenty pulse results from `resnet14_ten_seed_v1`, retaining their
original provenance. The first three pairs ultimately reference the original
`resnet14_transfer_v1` pilot files; the other seven reference the completed
ten-seed extension. The driver verifies all source receipts, full-training
manifests, hashes, sampled model identities and traffic prefixes. It records
new reuse receipts rather than copying or rewriting those original files.

New checkpoints live only in this campaign's own `checkpoints/` directory.
They preserve model/client replica, both error-feedback arrays, NumPy/Torch
RNG, completed round, metrics and traffic. Repeat the same run command to
resume from the last atomic checkpoint and skip completed fits. Changed
configuration, sources, data, runtime or reference evidence blocks reuse.

Numerical failures receive terminal receipts and remain in the completion
ledger. Other planned fits continue. Failed fits are not retried under new
settings, and no seeds are replaced. Final analysis requires all fifty
complete/reused outcomes; missing or failed outcomes are reported explicitly.
Structural errors propagate while preserving completed work. A controller
lock prevents overlapping instances of this campaign.

## Communication and analysis

Each new primary message is actually encoded/decoded, including complete
frame headers and padding. Dense uses full float32 frames; STC uses the
existing lossless SCL1 representation of its ternary values; EF uses lossless
support-plus-float32 uplink and the smaller exact aggregate/ordered-replay
downlink representation. Native-versus-encoded model, residual and prediction
parity is checked synthetically. The training host verifies client/server
model reconstruction each round.

**Upload plus conservative-unicast download** is primary. Count initial
synchronization, 32-bit client requests each round, actual frames and server
message replication to ten recipients. Logical broadcast remains supplementary.
Per-round codec CPU times are not physical communication latency or energy.

Automatic analysis writes:

- `confirmation_completion.json`, including every planned outcome and its origin;
- `confirmation_runs.csv`, `confirmation_history.csv`, `endpoint_summary.csv`;
- `paired_per_seed.csv`, `paired_summary.csv` for Event versus each comparator;
- `traffic_to_target.csv`, `paired_target_summary.csv` with complete hit coverage;
- `analysis_audit.json`, `confirmation.png` and `confirmation.pdf`.

The fixed round-1800 predictive endpoint is test accuracy, accompanied by CE,
worst-class accuracy, training CE/objective/L2 diagnostics and upload/download
costs. Paired differences use the same ten seeds. Ratios use log-scale paired
Student-t intervals; all intervals are descriptive and unadjusted. Negative
effects and unfavorable seeds are retained.

Keep the existing 45%, 50% and 55% accuracy targets. Crossings use the first
saved evaluation, without interpolation or requiring sustained achievement.
Unreached targets are censored at 1800 and remain visible; ratios are computed
only on common hits with full coverage counts. These diagnostics do not define
a test-driven early-stopping policy or asymptotic convergence-rate claim.

This campaign compares five **selected operating points**. The sparse-STC
development results remain an unresolved broader frontier comparison; omitting
their confirmation here does not establish a general advantage over every STC
setting. The test set and existing pulse outcomes have already been inspected.
Uncertainty concerns partition/SGD variation under fixed initialization and
configurations, not unseen datasets or a globally optimized/SOTA comparison.

## Run locally and commit results

From Anaconda Prompt in the research repository:

```bat
cd /d C:\Users\WeberJ\2026_NeuromorphicFL
conda activate neuromorphicfl-resnet14
git pull --ff-only origin main
python experiments\resnet14_core_confirmation.py verify --device cuda
python experiments\resnet14_core_confirmation.py preflight --device cuda
python -u experiments\resnet14_core_confirmation.py run --device cuda >> resnet14_core_confirmation.log 2>&1
git add experiments/results/resnet14_core_confirmation_v1
git commit -m "Add selected ResNet14 common-seed baseline confirmation"
git push origin main
```

`verify` uses unit tests, saved evidence and small synthetic real-ResNet
trajectories; it performs no research training or CIFAR download. `preflight`
checks scientific provenance, prints the faster codec identity and checks the
recorded execution environment. Synthetic ResNet verification compares against
both the original encoded runner and the native update, as well as exact
checkpoint resume. `run`
performs only the thirty new baseline fits and automatically analyzes the
fifty-outcome panel when complete.

From a second prompt, inspect progress without training:

```bat
python experiments\resnet14_core_confirmation.py status
```

After an interruption, repeat the same `run` and publication commands above.
Console output is appended to the ignored log file. Checkpoints and locks
remain local; the CSV/JSON evidence and diagnostic figures are tracked.

To rebuild a completed analysis without Torch, images or training:

```bat
python experiments\resnet14_core_confirmation.py analyze
git add experiments/results/resnet14_core_confirmation_v1
git commit -m "Refresh selected ResNet14 core confirmation analysis"
git push origin main
```
