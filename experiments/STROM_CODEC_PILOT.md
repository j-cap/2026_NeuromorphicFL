# Comparable Ström-style residual-pulse pilot

Repository: **j-cap/2026_NeuromorphicFL**, the research-code repository.
Entry point: `experiments/strom_codec_pilot.py`.
Codec adapter: `experiments/strom_codec_host.py`.

## Question and scope

Does a Ström-style residual-pulse method close Event's early accuracy-per-bit
advantage when given **the same lossless position coding and exact aggregated
downlink**, with explicit and reasonable server-gain scaling?

This is an implementation/scaling pilot: **one seed, 600 communication rounds,
six new Ström trajectories**. It is neither a ten-seed confirmation nor a new
selection campaign. Report every operating point. No test-based winner is
selected, and no confidence intervals are calculated from a single seed.

The common setting is the compact CIFAR CNN (20570 parameters), ten fully
participating clients, strong label skew, partition seed **3500**, SGD seed
**83500**, initialization seed 7777/scale 1, five local SGD steps, batch size 32,
local learning rate **0.05**, and L2 coefficient **0.0005**. The existing full
50000-example training allocation and 5000-example training diagnostic subset
are retained. This is the compact CNN, despite the conda environment name.

Reuse the first 600 rounds of the completed Event, STC p=1/gain 2, and
STC p=0.25/gain 2 trajectories. Their update schedules do not depend on the
terminal horizon; models and traffic at these prefixes are unchanged. Source
receipts/hashes and the 1800-round parent identities are recorded. No additional
Event/STC training or duplicate per-run CSVs are needed.

## Preserve the compressor and expose its units

This is a **Ström-style compressor within synchronous server-based FedAvg**.
The original work studies distributed SGD and peer communication, including
asynchronous execution, and also discusses Golomb-Rice coding of sorted index
gaps. We give the adapted baseline the same coding opportunity as Event.
Reference: [Ström, INTERSPEECH 2015](https://www.isca-archive.org/interspeech_2015/strom15_interspeech.pdf),
Sections 2.4--2.6. This experiment does not reproduce the original distributed
systems implementation.

The adapter preserves the existing `run_final_baseline(method="strom")`
recurrence, including one pulse per coordinate/client/round and the `>=`
threshold comparison. For local-model delta Delta_i and normalized weight p_i:

1. Accumulate `a_i = residual_i + (p_i / local_lr) * Delta_i`.
2. Emit `s_i = sign(a_i)` on coordinates with `abs(a_i) >= tau`; emit zero elsewhere.
3. Retain `residual_i = a_i - tau * s_i`.
4. Apply `native_gain * ordered_sum_i(tau * s_i)` to the global model.

There is **no leakage, full reset, Event threshold, or Event quantum schedule**
in this compressor. Overshoot is retained, and a large residual can remain
above threshold after one pulse. The gain is applied once after ordered float32
accumulation, exactly as in the native runner.

The archived state is in gradient-like units because it accumulates model
deltas divided by the local learning rate. For a constant native gain gamma,
the model-unit residual `gamma * residual_i` conserves inputs
`(gamma / local_lr) * p_i * Delta_i` and transmitted increments
`gamma * tau * s_i` in real arithmetic. Consequently,
**FedAvg-equivalent gain = native gain / local learning rate**.

The old native gain 1 therefore corresponds to gain **20** in model-delta units
at local_lr=0.05. This is an explicit gain convention, not a claim that
normalization alone fixes the method's performance. Keep the legacy points as
regression controls and include model-delta gains 1 and 2 in the pilot:

| Point ID | Threshold in archived state units | Native gain | FedAvg-equivalent gain |
|---|---:|---:|---:|
| `strom_t0025_g005` | 0.0025 | 0.05 | 1 |
| `strom_t0025_g010` | 0.0025 | 0.10 | 2 |
| `strom_t0025_g100` | 0.0025 | 1.00 | 20, legacy control |
| `strom_t0100_g005` | 0.0100 | 0.05 | 1 |
| `strom_t0100_g010` | 0.0100 | 0.10 | 2 |
| `strom_t0100_g100` | 0.0100 | 1.00 | 20, legacy control |

The two thresholds are the previously archived choices. Their old
"quality-selected" and "nearest-traffic" labels describe the earlier campaign;
they do not establish optimality or traffic matching under the new encoding.
No new settings are chosen from the known test results. A later paper-level
selection requires a separate development/validation procedure and sufficient
horizon, followed by matched multi-seed evaluation.

## Common message representations and accounting

For each Ström trajectory, measure the same four alternatives as Event:

- `legacy`: address/sign uploads and per-round replay/checkpoint fallback.
- `uplink_codec_replay`: packed positions, with replay/checkpoint fallback.
- `aggregate_downlink`: legacy uploads with the exact aggregated server message.
- `both_codecs`: packed uploads and the exact aggregated server message.

The new encodings preserve the trajectory. The actual packet bytes are decoded
on every round. The server sends signed integer pulse counts plus float32
corrections, or the exact ordered fallback when smaller. Decode at Ström's own
fixed threshold amplitude, then apply its native gain. The synchronized client
replica must remain bitwise equal to the server model. This does not insert
Event's independently annealed quantum into Ström.

Primary comparisons use **conservative total unicast traffic**, with upload
and download reported separately. Initialization, complete frame metadata,
padding and receiver copies are charged. The existing 32-bit per-client request
is charged on each unicast round. Logical-broadcast results remain secondary
and retain the frozen references' convention: common initialization/server data
charged once, without those per-client requests. This is application-message
accounting, not a network airtime or energy benchmark.

Layout, gain, threshold and round ordering are agreed by the surrounding
protocol, just as Event's jump schedule is agreed in its existing protocol.
Position-code mode, support and correction metadata are carried in the frames.

## Measurements, verification and interpretation

Evaluate the fixed test set at **round 1 and every 60 rounds**, through 600:
11 samples per trajectory. Save test CE, accuracy, macro/worst-class accuracy,
training CE/objective, per-round message counts, residual norm/max magnitude
and model fingerprints. Training loss and its L2 contribution remain separate
from predictive performance.

The unchanged **45%, 50%, 55%** targets use first saved crossings without
interpolation. A target not reached by round 600 is censored with missing
achievement cost. It must not be called achieved, omitted, or interpreted as
unreachable at 1800. The analysis exports all nine optimized-codec points,
all representations and all target outcomes, including censored outcomes.

The two native-gain-1 curves must match the archived Ström histories at printed
precision (rtol 1e-8, atol 1e-9), and native upload prefixes must match exactly.
All six configurations are checked synthetically against the native runner for
bitwise model/residual equality, RNG state, predictions and original traffic.
Live codec/client-replica checks preserve exact updates within each new run.
These checks do not supply archived full model files, which are unavailable.

Checkpoints atomically save model, client replica, persistent residual, RNG,
training history, prediction metrics and byte counters. Restarting the same
command skips verified completions and resumes incomplete points. Numerical
failures remain explicit and block complete-grid analysis. Source, runtime,
data and reference changes cannot be mixed into a running campaign.

No existing package module, frozen pilot, STC runner or old result is changed.
The two new runtime files live in `experiments/`, preserving the older pilot's
complete package-source hash inventory.

## Run locally

From Anaconda Prompt in the **code** repository:

```bat
cd /d C:\Users\WeberJ\2026_NeuromorphicFL
conda activate neuromorphicfl-resnet14
git pull --ff-only
python experiments/strom_codec_pilot.py verify
python experiments/strom_codec_pilot.py preflight
python experiments/strom_codec_pilot.py run --workers 2
python experiments/strom_codec_pilot.py analyze

git add experiments/results/strom_codec_pilot_v1
git commit -m "Add comparable Strom-style codec pilot results"
git push origin main
```

`verify` uses synthetic images and existing read-only references. It does not
train on or download CIFAR. The preflight should show
`Runtime matches full-horizon evidence: True` in the original Windows
environment. Actual data hashes are additionally checked before training.

Use `--workers 1` to reduce CPU/RAM concurrency. For a single point, e.g. a
FedAvg-equivalent gain-1 run, use
`python experiments/strom_codec_pilot.py run --point strom_t0025_g005 --workers 1`.
The unfiltered command subsequently completes the remaining points.

Progress/restart and committing completed outputs:

```bat
python experiments/strom_codec_pilot.py status
python experiments/strom_codec_pilot.py run --workers 2
python experiments/strom_codec_pilot.py analyze
git add experiments/results/strom_codec_pilot_v1
git commit -m "Add completed Strom-style codec pilot results"
git push origin main
```

## Outputs and next decision

Directory: `experiments/results/strom_codec_pilot_v1/`.

- `protocol.json`: six settings, source/runtime/data hashes, legacy reference
  identities and reused Event/STC source identities.
- Per new point: 600-row `rounds.csv`, 11-row `evaluation.csv`, 44-row
  representation `history.csv`, four-row `summary.csv`, and a completion receipt.
- `completion_audit.json`: status of all six required trajectories.
- `comparison.csv`: 32 representation endpoints; `comparison_history.csv`:
  352 representation/sample rows, including reused prefixes.
- `main_runs.csv`: nine optimized-codec endpoints with explicit Ström gain units
  and the L2-penalty diagnostic.
- `event_contrasts.csv`: eight descriptive endpoint comparisons.
- `traffic_to_target.csv`: 96 representation/target outcomes, including censoring.
- `analysis_audit.json`: regression checks and the single-seed scope.
- `pilot_comparison.png` and `.pdf`: accuracy versus rounds and unicast bits.

CSV/JSON evidence and these figures are Git-enabled. Local NPZ checkpoints and
lock files remain ignored.

After the pilot, inspect whether reasonable residual-conserving gains and
common coding close Event's early advantage. Resolve that mechanism question
before selecting settings for a full confirmation or moving to the larger model.
Use development/validation data for any selection; retain this pilot's complete
grid and the distinction between scaling controls and final tuned baselines.
