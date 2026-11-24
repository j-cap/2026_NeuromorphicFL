# ResNet baseline calibration: Dense FedAvg, STC and EF-TopK

Repository: **j-cap/2026_NeuromorphicFL**, the research-code repository.
Entry point: `experiments/resnet14_baseline_development.py`.
Results: `experiments/results/resnet14_baseline_development_v1/`.

## Why calibrate these baselines

STC and EF-TopK need architecture-specific calibration before the ResNet
comparison. Transferring only their compact-CNN settings would leave a tuning
asymmetry after calibrating Event and Ström for this network. Dense FedAvg
also receives calibration of its server gain and decay schedule.

Keep the common local learning problem fixed: ResNet-14/GroupNorm, 175258
parameters, ten fully participating clients, strong 55%-dominant label skew,
initialization 7777, five local SGD steps, batch size 32, learning rate 0.05
and L2 coefficient 0.0005. These shared parameters are not tuned separately
for any comparator.

Use the same **45000/5000 training/validation split**, partition seed 7000,
split seed 17000 and SGD seed 97000 as the Event/Ström calibration. Training
diagnostics use the same 5000-image training subset. Select by final validation
CE, then total conservative unicast traffic and point ID for ties. Accuracy,
training objective and official-test metrics do not select the settings.

## Fixed grids and matched calibration work

| Method | Main grid | Additional anchors | Candidates |
|---|---|---|---:|
| Dense FedAvg | Initial server gain 0.25, 0.5, 1, 1.5, 2, 4 × decay exponent 0, 0.2, 0.3 | None | 18 |
| STC | Symmetric upload/download fractions 0.01, 0.05, 0.25, 1 × server gain 0.5, 1, 2, 4 | Upload 0.01 or 0.25, download 1, gain 1 | 18 |
| EF-TopK | Global retained fraction 0.01, 0.025, 0.05, 0.1 × server gain 0.5, 1, 2, 4 | Retained fraction 0.005 or 0.25, gain 1 | 18 |

Dense gain is `gain0*(1+r/100)^(-exponent)`; exponent zero includes ordinary
constant-gain Dense FedAvg and gain one includes the usual FedAvg update.
The previous compact-CNN Dense schedule (gain 4, exponent 0.3) is included.
STC and EF gains are constant and scale their native server updates.

STC's fraction is a **per-parameter-tensor retention fraction**. Its client
and server error feedback remain active, and the gain is applied once before
server error feedback/compression. The main grid ties upload and download
fractions; two predeclared asymmetric anchors allow a full-retention server
compressor. **STC at p=1 still ternarizes each tensor. It is not full-precision
Dense FedAvg.** Both p=1 and p=0.25 therefore remain meaningful candidates.

EF-TopK retains its **global support selection**, float32 values and weighted
client-delta error feedback. Its gain is applied after client-message
aggregation. The search covers compressed operating points from 0.5% to 25%
retention and includes the compact-CNN 5%/gain-1 setting. Full retention is
used as a native-parity verification control, where the update equals Dense
FedAvg; it is not a second Dense candidate in the calibration grid.

| Stage | Work per method | Decision |
|---|---:|---|
| Screen | 18 × 600 rounds = 10800 rounds | Promote the three lowest final validation-CE settings |
| Continue | Three exact continuations × 1200 additional rounds = 3600 rounds | Freeze one winner at round 1800 |
| Analyze | No new training | Compare the selected settings with the read-only pulse development incumbents |

Maximum work is **14400 unique rounds per method**, or **43200 total rounds**,
with 774 unique evaluation points. This matches the accumulated ResNet work
already spent per Event/Ström family: 15 original screens plus three boundary
screens, two original continuations plus one boundary continuation. Their
historical search order and parameter geometry differ; matching total
candidate/continuation work does not establish identical tuning effectiveness
or global optimality.

There are 54 fresh trajectories and at most 63 stage receipts. Full finalist
archives repeat nine 600-round prefixes, giving at maximum 48600 archived
round rows and 873 archived evaluation rows. These are communication rounds,
not epochs or elapsed-time guarantees. No extra settings or boundary extension
are launched automatically.

## Native rules and lossless communication

Local SGD, prediction, initialization and global-model application use the
same Torch ResNet functions as the frozen pulse study. Compressor and server
aggregation arithmetic preserve the existing NumPy float32 baseline
conventions, with float64-normalized client weights. STC retains its exact-k
tie convention and float64 mean-magnitude reduction rounded to float32.
EF retains its NumPy global `argpartition` selection and weighted residual
units. Cross-backend bitwise parity to a different model/implementation is
not claimed; native-versus-encoded parity within each new host is checked.

Every primary message is actually encoded and decoded. Dense uses full
float32 frames. EF receives the existing float32 support codec and exact
aggregate/ordered-replay downlink portfolio, without a second lossy operator.
STC uses the existing lossless SCL1 format for its already ternarized values.
The new direct SCL1 emitter produces the same bytes as native Rice encoding
followed by lossless re-encoding, avoiding a redundant encode/decode pass.
It changes neither quantization nor either error-feedback state.

Costs count complete frames and padding, initial synchronization, per-client
requests and downlink replication. Conservative bidirectional unicast is
primary; logical broadcast is supplementary. Only actual primary wire
representations are reported, without hypothetical native-Rice traffic
estimates. Codec CPU overhead is not physical communication latency or energy.

The verification compares against native STCState and native Dense/EF
reference rules, including nonunit gain, zero-message charges, ternary p=1
and full-retention EF identity. Synthetic real-ResNet checks verify model,
client model, client/server feedback, NumPy/Torch RNG and predictions against
native updates, and exact traffic on interruption/resume. Scientific tables
reject the reference-only verification mode.

## Data isolation, failures and stopping point

The new calibration reads only training batches and validation evidence. It
does not parse official-test images or read the saved official-test metrics.
Raw batch fingerprints may include the test file as an input identity, just
as in the earlier calibration; its contents are not parsed for prediction.
The completed Event/Ström test results and their configurations stay frozen.

Numerical failures remain terminal receipts and ledger rows. If no finite
candidate remains for a family, stop explicitly. Finalist failures do not
trigger additional promotions. Structural errors propagate while preserving
completed work. Missing or changed evidence blocks final selection.

The run stops after writing `screening.csv`, `promotion.json`, `finalists.csv`,
`selected.csv`, `frozen_selection.json`, `validation_comparison.csv` and an
audit JSON. The comparison table contains the three selected new settings
and the two read-only pulse development incumbents, all on the same holdout.
These are validation results, not five-method test evidence.

After examining this calibration, prepare the **thirty new full-training
confirmation runs** for the three selected baselines on the same ten seeds
7100–8000, reusing all twenty completed Event/Ström outcomes. This stage does
not launch those runs or reuse development checkpoints for them.

## Run locally and commit results

Use the same `neuromorphicfl-resnet14` environment and GPU as the pulse study.
Defaults match its four threads and evaluation batch size 256. One worker
uses the device sequentially.

```bat
cd /d C:\Users\WeberJ\2026_NeuromorphicFL
conda activate neuromorphicfl-resnet14
git pull --ff-only origin main
python experiments\resnet14_baseline_development.py verify --device cuda
python experiments\resnet14_baseline_development.py preflight --device cuda
python -u experiments\resnet14_baseline_development.py run --device cuda >> resnet14_baseline_development.log 2>&1
```

After completion:

```bat
git add experiments/results/resnet14_baseline_development_v1
git commit -m "Add ResNet14 baseline validation calibration"
git push origin main
```

Progress is appended to `resnet14_baseline_development.log`. From a second prompt:

```bat
python experiments\resnet14_baseline_development.py status
```

Repeat the same `run` and result-commit commands after an interruption.
Complete stages are verified/skipped, and partial trajectories resume from
their last atomic numeric checkpoint. Finalists continue the exact screened
trajectory with its original 1800-round identity. Keep local checkpoints
until completion. Checkpoints, locks and the console log are ignored by Git.

To rebuild completed validation summaries without training or loading images:

```bat
python experiments\resnet14_baseline_development.py analyze
git add experiments/results/resnet14_baseline_development_v1
git commit -m "Refresh ResNet14 baseline calibration analysis"
git push origin main
```
