# Full-horizon Event/STC codec diagnostic

Repository: **j-cap/2026_NeuromorphicFL**, the research-code repository.
Entry point: `experiments/event_codec_full_horizon.py`.

## Fixed question and scope

Does the communication advantage of the validated codec persist over **1800
communication rounds**, and what held-out predictive quality does each method
achieve per communicated bit? This is one fixed-seed diagnostic before any
change to the Event trigger or residual protocol. It is not a statistical
confirmation or a state-of-the-art claim.

| Point | Existing training rule | Codec comparison |
|---|---|---|
| `event` | Persistent/full-reset Event, gain 1 | Address/replay, packed-upload replay, aggregate downlink, both codecs |
| `stc_primary` | STC p=1 in both directions, gain 2 | Native framing versus lossless recoding |
| `stc_sparse` | STC p=.25 in both directions, gain 2 | Native framing versus lossless recoding |

All use the compact CIFAR CNN (20570 parameters), ten fully participating
clients, the existing strong partition with **seed 3500**, and **SGD seed 83500**.
This is the first predeclared evaluation seed of the existing ten-seed
confirmation, distinct from development partition 4500. The full 50000-example
training allocation and fixed 5000-example training diagnostic subset are
unchanged. Local learning uses five SGD steps per round, batch 32, learning rate
.05, L2 coefficient .0005, and initialization seed 7777 with scale 1. Compressor
settings are the frozen development choices. There is no new parameter search,
test-driven selection, early stopping or best-test-checkpoint reporting.

Only **three trajectories** are trained. Alternative message representations
are measured on each point's identical updates. Training is regenerated because
the archived summaries do not contain full pulse-position/message streams.

## Prediction evaluation and targets

The real, fixed CIFAR-10 test set is evaluated at **round 1 and every 60 rounds**
through round 1800. Record test accuracy, unregularized test cross-entropy,
macro accuracy and worst-class accuracy. Training objective and training
cross-entropy remain separate diagnostics. The fixed endpoint is the **final
round 1800**.

Report actual cumulative upload, download, total conservative-unicast traffic
and logical-broadcast traffic. The history associates every evaluated model
with the exact prefix cost of each representation, including initialization,
headers, byte padding, requests and receiver replication. Historical Event
replay preserves the cheaper-of-replay-or-float32-checkpoint rule.

Traffic-to-accuracy targets are fixed at **45%, 50% and 55%**. The first saved
evaluation meeting a target is used. There is no interpolation between 60-round
samples. Unreached targets remain explicitly censored at 1800, with achievement
costs missing. No confidence intervals or significance claims are produced for
one seed. Both accuracy-versus-round and accuracy-versus-bits are needed to
interpret the experiment.

## Exactness, resume and references

The driver imports the existing Event/STC codec host unchanged. It adds a
checkpoint adapter that evaluates the test set and atomically saves prediction
metrics with the model, RNG, residuals, per-round byte counters and training
history. A one-time round-1 checkpoint is followed by exact continuation at
round 2. Later checkpoints are every 60 rounds. An interruption during evaluation
leaves the preceding complete checkpoint intact.

Every communicated update and synchronized client-model replica must remain
bitwise equal to the unchanged host's server update. Event retains integer-count
roundoff corrections or the exact ordered fallback. STC recoding preserves
native selection, magnitudes, gains and both residual levels.

The existing archived Event seed-3500 curve is a required regression reference.
Saved training and test diagnostics must match its printed precision, and native
upload prefixes must match exactly. Original P19 CSVs have about ten significant
digits, so metric comparisons use the existing archive tolerance (rtol 1e-8,
atol 1e-9). They are not a comparison against archived full model files, which
are unavailable. Some legacy configuration/runtime fields were not recorded per
run and remain subject to the existing declared-driver compatibility scope.

If matching seed-3500 STC confirmation receipts are present when analyzing, their
native prediction/traffic curves are checked too. If they are still pending,
analysis records that status and uses the newly measured fixed STC trajectories.
No confirmation files or checkpoints are overwritten.

Use the same environment and raw dataset batches as the 120-round pilot. Source,
dataset and runtime changes cannot be mixed into a running campaign. Frozen
pilot and ten-seed confirmation source files remain unchanged. This follow-up
adds no new package module, preserving the pilot's complete package-source hash
inventory.

## Run locally

From Anaconda Prompt in the **code** repository:

```bat
cd /d C:\Users\WeberJ\2026_NeuromorphicFL
conda activate neuromorphicfl-resnet14
git pull --ff-only
python experiments/event_codec_full_horizon.py verify
python experiments/event_codec_full_horizon.py preflight
python experiments/event_codec_full_horizon.py run --workers 2
python experiments/event_codec_full_horizon.py analyze

git add experiments/results/event_codec_full_horizon_v1
git commit -m "Add full-horizon single-seed codec results"
git push origin main
```

The preflight should print `Runtime matches codec pilot: True` on the original
Windows environment. `verify` uses synthetic inputs only and exercises exact
prediction/model resume and spawned workers without a dataset download.

If another confirmation is still using the machine, `--workers 1` reduces
concurrency. For Event first, use `run --point event --workers 1`; the default
`run` command subsequently completes the two STC controls and skips verified
completed points. Analysis requires all three points and retains numerical
failures rather than silently omitting them.

For progress and recovery:

```bat
python experiments/event_codec_full_horizon.py status
python experiments/event_codec_full_horizon.py run --workers 2
python experiments/event_codec_full_horizon.py analyze
git add experiments/results/event_codec_full_horizon_v1
git commit -m "Add completed full-horizon codec results"
git push origin main
```

## Outputs

Directory: `experiments/results/event_codec_full_horizon_v1/`.

- `protocol.json`: fixed protocol, source/data/runtime fingerprints and Event reference identity.
- Per point: 1800-row `rounds.csv`, 31-row `evaluation.csv`, long-form predictive/cost
  `history.csv`, endpoint `summary.csv`, and a completion receipt.
- `comparison.csv`: final predictive metrics and byte costs for all eight representations.
- `comparison_history.csv`: sampled learning curves with exact prefix costs.
- `traffic_to_target.csv`: 24 representation/target outcomes, including censoring.
- `analysis_audit.json`: source/receipt checks and available trajectory regression checks.
- `accuracy_traffic.png` and `.pdf`: test accuracy versus rounds, test cross-entropy
  versus rounds, and test accuracy versus communicated bits.

Root CSV/JSON evidence and these figures are Git-enabled. NPZ checkpoints and
lock files remain ignored. `codec_audit_seconds` includes reference encodings
and reconstruction checks and is not a deployment encoding-time benchmark.
