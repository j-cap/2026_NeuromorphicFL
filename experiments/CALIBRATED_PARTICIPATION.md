# Calibrated five-method participation study

This is a separate robustness extension. The completed full-participation
panels, manuscript products, original P20 outputs and all frozen package files
remain unchanged. The new entry point is experiments/calibrated_participation.py
in the research repository 2026_NeuromorphicFL.

## Scientific question and fixed design

Does the current Event protocol retain a favorable communication/quality point
when clients participate intermittently and exact returning-client catch-up
is charged? This is a transfer study of the full-participation-selected
configurations, not a claim of optimal hyperparameters at each fraction.
No threshold/gain search or selection from the new results is performed.

| Registered clients | Active per round | Participation | Role |
|---:|---:|---:|---|
| 50 | 50 | 100% | Larger-population full-participation control |
| 50 | 10 | 20% | Primary partial-participation setting |
| 50 | 5 | 10% | Sampling stress |

All five methods are included at each fraction: Event-FedAvg, Ström-style
residual pulses, STC, EF-TopK and Dense FedAvg. The model is the original
20,570-parameter compact CIFAR-10 CNN. Frozen settings are read directly
from the completed common-seed core protocol and checked against its freeze:

| Method | Transferred configuration |
|---|---|
| Event | Threshold .05, initial quantum .01, exponent .2, scale 100, leak .999, persistent state/full reset, gain 1 |
| Ström-style | Threshold .05, native gain .1 (FedAvg-equivalent gain 2), nonleaky subtractive residual |
| STC | Upload/download retention 1, gain 2, ternarization and both residuals |
| EF-TopK | Global retention 5%, gain 1, float32 values and client error feedback |
| Dense | Gain 4*(1+r/100)^(-.3), dense float32 messages |

All share five local SGD steps, batch 32, local learning rate .05,
L2 coefficient .0005 and initialization seed 7777. Each client has 55%
dominant-class and 5% of each other class. Within a seed, the client partition
is identical across methods and fractions. A shared stream of uniform
permutations makes each round's five clients a subset of its ten clients.
Client order and RNG use remain sequential inside a run. Within each fraction,
all methods see the same selection schedule and local minibatches.

Sampling weights are p_i/pi_i, which equals 1/K here. Their placement preserves
each original operator: Event/Ström use weighted gradient-scale evidence;
EF-TopK retains weighted delta residuals; STC keeps UNWEIGHTED client delta
residuals and weights each decoded upload once, before its server residual.
STC weighting must not be moved into its client compressor. Inclusion correction
does not imply unbiasedness after nonlinear compression.

## Pilot and confirmation

The pilot has 15 jobs at 300 rounds on development partition 13000, SGD seed
113000, and a fixed 45,000/5,000 training/validation split (split seed 23000).
Each client has 900 training images. It parses/evaluates training and validation
data only; the official CIFAR distribution archive inherently contains both
training and test files. Pilot output columns are labeled validation.
The frozen 1,800-round configurations are stopped at round 300 without changing
the schedules. No development checkpoint is reused for confirmation.

Confirmation has 150 new jobs: five methods x three fractions x ten partition
seeds 13100--14000 in steps of 100. SGD seed = partition seed + 100000;
selection seed = partition seed + 200000. Every fit starts fresh on all 50,000
training images (1,000/client), uses the official 10,000-image test set, and
runs 1,800 rounds. Evaluation is at round one, every sixty rounds and the final
round. Targets 45%, 50% and 55% are fixed in advance.

The confirm command requires all fifteen pilot jobs to have valid terminal
receipts, then writes an immutable confirmation freeze BEFORE parsing test
data. The gate does not require Event to win and does not select parameters.
Numerical failures are retained as terminal results, including failed-round
metadata. They are not silently retried, replaced or excluded. A protocol or
source mismatch stops execution. Source changes after prepare cannot be mixed
into that campaign; use a separately identified output directory if a deliberate
scientific protocol revision becomes necessary.

## Exact synchronization and accounting

All fifty clients initially receive the dense model. Only selected clients
request catch-up, compute local updates and receive the current server frame.
Returning clients replay the actual missed server packets IN ORDER, or receive
a dense model checkpoint when smaller. Each reconstruction is verified bit for
bit. Summing missed floating-point deltas is not used as a shortcut.

Event and Ström share the current lossless pulse-support/aggregate portfolio,
including float32 fidelity corrections and the exact ordered fallback.
EF-TopK keeps its exact float32 aggregate/replay representation. STC keeps its
native lossy server update and both residuals, transported losslessly.
The generic bound-one codec accelerations preserve the native bytes.

Inactive Event state decays by rho**gap on return; other client residuals remain
unchanged. The server journal discards an old frame when all clients have
already received it, or when any client that could still need it would choose
the smaller checkpoint. Thus full participation retains no missed-frame
backlog, and intermittent clients do not require storing every decoded
model-sized update for the full campaign. Checkpoints save
the exact model, all cached client models, residuals, RNG, counters, evaluations
and retained actual wire frames. Restart verifies every saved client cursor.

Record both:

- Operational unicast: actual uploads, initial model distribution, selected-client
  requests, missed-update catch-up and replicated current server messages.
- Conservative total: operational traffic plus exact final synchronization of
  every registered client, with terminal requests/payload reported separately.

At each saved evaluation, terminal cost is the hypothetical extra cost IF
training stopped there. It does not mutate state or get charged repeatedly
during training. The same rule is used for each run's first saved accuracy
crossing. Missing targets are censored, never assigned zero cost.

Requests are recorded in the physical uplink in this new study. Update-payload
uplink is also exported separately. At full participation, total unicast and
all data-frame lengths match the existing full-participation hosts; the older
tables put request bits into their synchronization/control column.
These are serialized-payload costs, not measured latency or energy.

Fixed round counts involve different total computation across fractions.
Exported active-client rounds, local SGD steps and sampled examples make
this difference explicit. Methods are compute-matched WITHIN each fraction.
Do not interpret a low-fraction fixed-round result as equal total training work.

## First local run: verification and pilot

Windows Command Prompt:

    cd C:\Users\WeberJ\2026_NeuromorphicFL
    conda activate neuromorphicfl-resnet14
    git pull --ff-only
    python experiments/calibrated_participation.py verify
    python experiments/calibrated_participation.py pilot --workers auto
    python experiments/calibrated_participation.py status
    python experiments/calibrated_participation.py analyze --phase pilot
    git add experiments/results/calibrated_participation_v1
    git diff --cached --check
    git commit -m "Add calibrated five-method participation pilot"
    git push origin main

The pilot command prepares its cache/immutable contract automatically and
generates the analysis when the queued jobs finish. Running analyze again is
optional. Verify uses only synthetic data, including a tiny native-CNN smoke
check, and does not download CIFAR or produce paper evidence.

The default worker count is half the reported CPU threads, capped at eight.
Each worker has one BLAS thread. Images are shared read-only raw uint8 mmaps;
only requested minibatches are materialized. Set --workers 4 for a conservative
run or --workers 12 if the machine has enough physical cores/RAM. Parallelism
changes which independent jobs run together; it does not change a run's local
update order. Repeating a command validates completed receipts and resumes
unfinished jobs from their last atomic checkpoint (at most sixty rounds lost).
Ctrl+C or closing a process can be followed by the same command to resume;
ordinary graceful shutdown of a process pool may wait for active jobs.

Optional prepare without training:

    python experiments/calibrated_participation.py prepare
    git add experiments/results/calibrated_participation_v1
    git commit -m "Freeze calibrated participation protocol and schedules"
    git push origin main

## Long confirmation, after examining the pilot

No further code generation is required:

    python experiments/calibrated_participation.py confirm --workers auto
    python experiments/calibrated_participation.py analyze --phase confirmation
    python experiments/calibrated_participation.py status
    git add experiments/results/calibrated_participation_v1
    git diff --cached --check
    git commit -m "Add ten-seed calibrated participation confirmation"
    git push origin main

Subset/resume filters are --method, --active-clients (50, 10 or 5), and
--seed-index (0--9, confirmation only). They never change the prescribed
analysis denominator or erase missing jobs. Completed receipts are reusable
only under the same frozen source/runtime/configuration identities.

## Analysis products

Each phase exports complete per-run raw round/history/summary CSVs and hashed
terminal receipts. The analysis regenerates coverage, completion, endpoint
means/SDs, paired accuracy/CE differences and traffic ratios, target coverage/
costs, two rows of trajectory plots, and an input/output audit.

Ratios use paired geometric means and descriptive unadjusted 95% Student-t
intervals on the log scale. A single pilot seed has point estimates only,
never a confidence interval. Failed/missing confirmation pairs block the
affected endpoint estimates; they are not dropped to create a smaller
successful panel. Target ratios additionally report both coverage counts
and the conditional common-hit denominator.

All result files are under experiments/results/calibrated_participation_v1.
Only root CSV/JSON and comparison PDFs/PNGs are tracked; image caches, binary
checkpoints, lock files and transient progress remain local. The original
full-participation freeze and manuscript remain untouched by this preparation.
