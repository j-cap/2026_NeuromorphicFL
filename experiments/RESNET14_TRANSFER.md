# Weekend ResNet-14 transfer study

Repository: **j-cap/2026_NeuromorphicFL**, the research-code repository.
Entry point: `experiments/resnet14_transfer.py`.
Results: `experiments/results/resnet14_transfer_v1/`.

## Question and automatic stages

Does the shared-codec Event/Ström comparison retain its communication-to-accuracy
behaviour on a larger ResNet-14, after bounded validation-only calibration?
This study keeps CIFAR-10 and starts with the closest pulse comparator. It does
not start another full dataset/architecture matrix or retrain the compact-CNN
core. A single `run` command advances through the entire study:

| Stage | Work | Data and decision |
|---|---|---|
| 1. Screen | 15 Event + 15 Ström settings, 600 rounds each | 45000 training / 5000 validation images; promote the two lowest validation-CE settings per method |
| 2. Finalists | At most four exact continuations from 600 to 1800 | Same training holdout; freeze one lowest-validation-CE setting per method |
| 3. Confirmation | Two selected methods × three fresh seeds × 1800 rounds | Refit from initialization on all 50000 training images; evaluate the official 10000-image test set |
| 4. Analysis | Automatic after all six confirmation runs complete | Paired accuracy/CE, upload/download/unicast costs, and first crossings of fixed accuracy targets |

With all candidates finite, this is **36 fresh trajectories**, **40 stage
receipts** and **33600 training rounds**: 18000 for screening, 4800 for finalist
continuation, and 10800 for confirmation. There are 596 sampled evaluations,
each covering the training diagnostic subset and validation/test set. These
are communication rounds, not epochs. Checkpoint recovery or replacing lost
local work can add execution time.

The next stage does not require a manual decision. Confirmation runs proceed
after the selections freeze even if Ström has better development results.
There is no Event-wins gate or automatic search extension at a grid boundary.

## Architecture, fixed learning protocol and bounded grids

Use the existing Torch **CIFAR ResNet-14 with GroupNorm, 175258 parameters**.
Ten clients participate in every round with 55%-dominant label skew. Local
training is five SGD steps, batch size 32, learning rate 0.05, L2 coefficient
0.0005 over all parameters, and initialization seed 7777. Training diagnostics
use 5000 training images. Evaluation is saved at round 1 and every 60 rounds.
There is no data augmentation, mixed precision or TF32. Deterministic Torch
algorithms are enabled. One device worker runs sequentially, avoiding multiple
training processes competing for the same GPU.

Event retains the current method's persistent state, leak 0.999, full reset,
server gain 1 and quantum schedule `q_r=q0*(1+r/100)^(-0.2)`. In particular,
the exponent remains **0.2**, as in the current compact-CNN core. It does not
silently inherit the older P13 ResNet exponent 0.3. Ström retains its native
nonleaky residual, subtractive reset and one pulse per coordinate/client/round.

| Method | Threshold grid | Second parameter | Candidates |
|---|---|---|---:|
| Event | 0.005, 0.01, 0.0125, 0.025, 0.05 | Initial quantum 0.0025, 0.005, 0.01 | 15 |
| Ström | Same five thresholds | FedAvg-equivalent gain 2, 10, 20; native gain 0.10, 0.50, 1.00 with local lr 0.05 | 15 |

These grids include the compact-CNN choices and the earlier ResNet Ström
threshold 0.01/native gain 1.00. The earlier ResNet Event threshold/quantum
pair 0.0125/0.005 is also represented, with the current fixed quantum exponent.
Both families receive fifteen screening settings and at most two full-horizon
finalists. Rank by final validation CE, then total unicast traffic, then point
ID. Accuracy and training objective do not select the winners. The final choice
is the best promoted finalist, not a claim of optimality over a full 1800-round
grid. No settings are changed after inspecting confirmation test curves.

Numerical failures receive terminal receipts and remain in the calibration
ledger. Promote the best two finite complete settings per family, or one if
only one is finite. If a family has no finite eligible setting, stop with an
explicit error rather than inventing a fallback. Structural errors such as CUDA
or checkpoint incompatibility propagate and preserve completed work.

## Data isolation and fresh seeds

The development partition uses seed **7000**, training RNG seed **97000**, and
holdout split seed **17000**. Hold out 500 images per class from the training
set. The remaining 45000 images form ten clients of 4500 images each, with
2475 dominant-class and 225 images from each other class. Training diagnostics
are sampled only from those 45000 images. Exact split indices and hashes are
saved and verified.

The development loader parses only the five training batches. The official
test batch may be fingerprinted as an input file but is not parsed, allocated
or evaluated until both final selections have been frozen and revalidated.

| Confirmation pair | Partition seed | Shared SGD seed |
|---:|---:|---:|
| 0 | 7100 | 97100 |
| 1 | 7200 | 97200 |
| 2 | 7300 | 97300 |

These seeds do not overlap the recorded prior partition-seed snapshot through
commit `4640f164ccafd84f95364545ed3bb3945a5fc14e`. Each confirmation uses the full
50000-image training set, 5000 images/client, shared initialization and matched
minibatch RNG across methods. Development models/checkpoints cannot be reused
for confirmation. The official test set has already been inspected in earlier
studies; fresh partition/SGD seeds do not create a blind new test dataset.

## Communication fidelity and checkpointing

Both methods use the same actual lossless position encoder and exact aggregate
downlink portfolio as the compact-CNN study. Every round encodes and decodes
the client pulses, checks the aggregate against the native Torch pulse update
bit for bit, applies the gain once, and checks model reconstruction. This
preserves each Torch method's native leak/reset/residual rule. It does not claim
bitwise trajectory parity between the NumPy compact CNN and Torch ResNet.

Primary accounting is conservative unicast: actual complete frames, padding,
initial model synchronization, 32-bit requests per client per round and downlink
replication to all clients. Upload and download remain separate; logical
broadcast is supplementary. Codec timings are audit overhead, not measured
physical-network latency or energy.

NPZ checkpoints atomically preserve the model, client replica, pulse/residual
state, NumPy/Torch/CUDA RNG states, all round counters and sampled evaluations.
The configured identity is 1800 rounds from the beginning, so finalists resume
the exact 600-round trajectory. Saved 600-round prefixes are independently
checked against the final records before confirmation. Missing finalist NPZs
produce an explicit error. Complete scientific results remain usable without
the NPZ files; unfinished trajectories require the local checkpoints.

Source files, configs, data fingerprints, split, runtime/GPU and evaluation batch
size are pinned. Resume with the same command/options/environment. Old package
sources, datasets and previous experiment results remain unchanged.

## Run today

Use the existing `neuromorphicfl-resnet14` environment. It already specifies
Torch 2.5.1, torchvision 0.20.1 and CUDA 12.1. From Anaconda Prompt:

```bat
cd /d C:\Users\WeberJ\2026_NeuromorphicFL
conda activate neuromorphicfl-resnet14
git pull --ff-only

python experiments/resnet14_transfer.py verify --device cuda
python experiments/resnet14_transfer.py preflight --device cuda
python experiments/resnet14_transfer.py benchmark --device cuda
python experiments/resnet14_transfer.py run --device cuda > resnet14_transfer_weekend.log 2>&1

git add experiments/results/resnet14_transfer_v1
git commit -m "Add bounded ResNet14 pulse transfer results"
git push origin main
```

`verify` uses synthetic images, tests the split/selection/controller/evidence
handling, and runs actual ResNet-14 codec/resume checks on the requested device.
It performs no CIFAR research training or download. `preflight` reports hardware,
versions, determinism and the declared workload. `benchmark` uses synthetic
images to estimate training-plus-codec time for the planned round count. It
excludes full-data evaluations, checkpointing and I/O; firing patterns and load
also change runtime. Use its output as a planning estimate, not a finish-time
guarantee. The preparation and research training happen in `run`.

`--device cuda` fails clearly if CUDA is unavailable. `--device auto` selects
CUDA if available and otherwise CPU; `--device cpu` forces CPU. Choose one device
and keep it fixed once the campaign starts. The default Torch thread count is
4 and evaluation batch size is 256. Optional `--threads` and `--eval-batch-size`
must also stay fixed for resumption. There is no multi-worker GPU option.

The redirected command stays running while progress is written to
`resnet14_transfer_weekend.log`. Keep the machine awake and the terminal running.
In a second prompt, activate the same environment and inspect progress:

```bat
python experiments/resnet14_transfer.py status
```

If interrupted, repeat the same run command. Valid completed stages are skipped;
unfinished checkpoints resume. Analysis runs automatically after all six
confirmation trajectories complete. To resume and publish:

```bat
python experiments/resnet14_transfer.py run --device cuda >> resnet14_transfer_weekend.log 2>&1
git add experiments/results/resnet14_transfer_v1
git commit -m "Add completed ResNet14 transfer study results"
git push origin main
```

## Outputs and interpretation

The folder contains `protocol.json`, split indices/manifest, per-stage receipts
and round/evaluation/summary/history CSVs, `screening.csv`, `promotion.json`,
`finalists.csv`, `frozen_selection.json`, three confirmation partition manifests,
`confirmation_completion.json`, `confirmation_runs.csv`,
`confirmation_history.csv`, `paired_per_seed.csv`, `paired_summary.csv`,
`traffic_to_target.csv`, `paired_target_summary.csv`, `analysis_audit.json`,
and `confirmation.png` / `.pdf`. Checkpoints and locks remain local and ignored.

Final confirmation requires all three paired seeds. A missing or numerically
failed confirmation remains visible and blocks the paired summary. Report
round-1800 accuracy and CE alongside communication, with descriptive unadjusted
paired 95% intervals; cost ratios use the log scale. With n=3, this is an initial
architecture-transfer study, not the ten-seed compact-CNN confirmation or a
formal equivalence/noninferiority test.

Targets remain 45%, 50% and 55% accuracy, using first saved crossings without
interpolation. Keep method hit counts and common-hit denominators; unreached
targets remain censored. No test checkpoint is selected as the winner and no
test-driven early stopping is implemented. Architecture and backend both change
relative to the compact CNN, so this study tests the protocol on a larger Torch
network rather than isolating a causal model-size effect. Results determine
whether to extend the ResNet comparator set or narrow the paper's scope.

To regenerate the confirmation analysis after completion:

```bat
python experiments/resnet14_transfer.py analyze
git add experiments/results/resnet14_transfer_v1
git commit -m "Refresh ResNet14 transfer confirmation analysis"
git push origin main
```
