# Event threshold/quantum development against calibrated Ström

Repository: **j-cap/2026_NeuromorphicFL**, the research-code repository.
Entry point: `experiments/event_codec_development.py`.
Output: `experiments/results/event_codec_development_v1/`.

## Question and scope

Can a different Event threshold/quantum pair improve validation predictive
quality and communication relative to the frozen current Event setting and
the calibrated Ström reference? This is an **additional development study after
inspecting confirmation results**, not a new held-out test confirmation.

The earlier compact-CIFAR Event search used a 120-round training-CE criterion.
This study uses the current long-horizon, validation-CE and exact end-to-end
communication protocol. It preserves the Event algorithm, optimizer, codecs
and all previous evidence. No decay, leak, reset, regularization or local-SGD
change is folded into this search.

## Fixed data and computation

Reuse the exact training-derived split from
[`STROM_CODEC_DEVELOPMENT.md`](STROM_CODEC_DEVELOPMENT.md): **45000 training
images and 5000 validation images**, with 500 validation images per class.
Split seed is 64500, partition seed 4500, and SGD seed 84500. Ten fully
participating clients each receive 4500 training images with 55% dominant-class
allocation. Training diagnostics use a disjoint 5000-image subset of the
training images. Validation images participate in no local gradient step.

The shared `strom_development_data.py` adapter parses only the five CIFAR
training batches. The official test batch is never parsed/evaluated and the
generic host's test slots remain empty. The runner reads only previous
validation-study evidence for references; it loads no confirmation test curves.
The exact index ledger, ordered index hashes, source/runtime and training-batch
identities are checked. A local copy of the split ledger is saved with this study.

Use the compact CIFAR CNN (20570 parameters), initialization seed 7777, five
local SGD steps, batch 32, local learning rate 0.05, and L2 coefficient 0.0005.
Evaluation is at round 1 and every 60 rounds. This is the compact CNN experiment,
not the larger ResNet model named in the conda environment.

## Fifteen fixed Event candidates

| Parameter | Values |
|---|---|
| Event threshold | 0.0125, 0.025, 0.05, 0.10, 0.20 |
| Initial model quantum | 0.0025, 0.005, 0.010 |
| Leak | 0.999, fixed |
| Memory/reset | Persistent memory and full reset, fixed |
| Quantum schedule | `q(r) = q0 * (1 + r/100)^(-0.2)` |
| Server gain | 1, fixed |
| Screening/final horizons | 600 / 1800 communication rounds |

The Cartesian grid contains **15 points**, including the current threshold
0.025 / quantum 0.005 setting. No automatic boundary extension occurs. A
boundary winner is a limitation of the finite search, not an instruction to
continue searching until Event wins.

The calibrated Ström reference has threshold 0.05, equivalent gain 2, and
native gain 0.10. Its algorithm and representation remain frozen. Both methods
use the same lossless position and exact aggregated-downlink portfolio, with
float32 correction patches or ordered fallback as needed.

## Reuse and maximum new work

The completed Ström validation study already contains **current Event and
calibrated Ström at both 600 and 1800 rounds on this exact split**. Reuse those
four stages after verifying their original frozen selection, source receipts,
tables, configs and model fingerprints. New reuse receipts point to the original
files; the original results/checkpoints are not edited or relabelled in place.

Only **14 new Event trajectories** need a 600-round screen. Continue at most
two distinct selected Event candidates to 1800. If the current Event setting is
selected, reuse its completed 1800-round result. Maximum new training is 10800
communication rounds: `14*600 + 2*1200`. Reused references are not new runs.

## Selection rules

At round 600 select among all 15 Event candidates:

1. **Quality:** minimum final unregularized validation cross-entropy.
2. **Within Ström budget:** minimum validation CE among candidates at or below
   calibrated Ström's actual total unicast cost at the same horizon.

Break exact CE ties by lower traffic, then point ID. Accuracy, training objective
and any previously observed test score do not select settings. Retain the full
trial ledger, including explicit numerical failures.

If no Event candidate fits the budget, record a **null** constrained choice.
There is no nearest-traffic fallback and no automatic promotion of an infeasible
point as if it met the budget. Deduplicate the two selected roles, producing
one or two promoted IDs.

At round 1800, apply the same rules **among those promoted candidates**, using
Ström's actual round-1800 cost. No additional candidate is promoted after
inspecting final outcomes. The current Event reference remains visible at the
final horizon, but is eligible for final selection only if the round-600 rule
promoted it. This is a bounded two-stage search, not an exhaustive full-horizon
optimum. An unpromoted setting could improve later, and a selected finalist
need not outperform the current full-horizon control. Report that outcome if it
occurs; the code does not require a favourable result or replace the method.

The budget uses the **validation split**, not the old full-training ten-seed
mean. The matching frozen Ström costs are 158684888 bits at 600 and 518019672
bits at 1800. Primary cost is `both_codecs` conservative unicast, including
initialization, complete frames/padding, per-client requests and downlink
replication. Logical-broadcast counters remain supplementary.

## Continuation and evidence

Each new run has an 1800-round config identity from its first round. Screening
stops at 600; the same checkpoint resumes at 601 with model, client replica,
Event state, RNG, traffic counters and validation history intact. Validation
metrics and model state are saved atomically. Before accepting a final result,
its screening prefix must reproduce exactly, except codec audit timings.

Repeating a command resumes incomplete jobs and skips verified complete/reused
or explicitly failed outcomes. Source, config, runtime, raw training-data, split
or reference changes cannot be mixed into existing results. A numerical failure
does not trigger changed settings or silent retries. Configuration/codec errors
stop the runner for investigation.

If local NPZ checkpoints are lost, a promoted trajectory can be regenerated from
round 1 in the same recorded environment, but its saved screen prefix must still
match. NPZ files and locks remain local and ignored. Scientific CSV/JSON results
and the two frontier figures are Git-enabled.

## Run locally

From Anaconda Prompt in the code repository:

```bat
cd /d C:\Users\WeberJ\2026_NeuromorphicFL
conda activate neuromorphicfl-resnet14
git pull --ff-only

python experiments/event_codec_development.py verify
python experiments/event_codec_development.py preflight
python experiments/event_codec_development.py run --workers 2
python experiments/event_codec_development.py analyze

git add experiments/results/event_codec_development_v1
git commit -m "Add validation-only Event threshold quantum development results"
git push origin main
```

`verify` uses synthetic images and committed validation metadata; it performs
no CIFAR research training or download. In the original Windows conda environment,
the preflight should report `Runtime matches recorded Windows environment: True`.
Training also verifies all five raw training-batch hashes.

The default `run` completes screening and finalist continuation automatically.
Use `--workers 1` to reduce CPU/RAM concurrency. Check progress with:

```bat
python experiments/event_codec_development.py status
```

After interruption, continue and publish results with:

```bat
python experiments/event_codec_development.py run --workers 2
python experiments/event_codec_development.py analyze
git add experiments/results/event_codec_development_v1
git commit -m "Add completed Event development selection results"
git push origin main
```

To inspect screening before the final horizon, substitute `screen --workers 2`
for `run --workers 2`, then use `analyze` and the same result commit commands.
Later use `finalize --workers 2`, `analyze`, and the result commit commands.
Selection rules are identical in both workflows.

## Outputs and subsequent decision

- `protocol.json`, `split_indices.csv`, `split_manifest.json`: frozen grid,
  sources, runtime/data, reference identities and exact training/validation split.
- Four reuse receipts linking current Event and calibrated Ström at both horizons.
- Per-new-point/stage round, evaluation, summary and history CSVs plus hashed
  completion or numerical-failure receipt. No `test_*` metrics are exported.
- `screening.csv`, `screening_selection.json`: complete screen ledger and
  immutable promotion decision.
- `finalists.csv`, `frozen_selection.json`: final quality/budget choices, exact
  configs and explicit infeasibility where applicable.
- `controls.csv`: both fixed controls at both horizons, kept separate from final
  selection eligibility.
- `analysis_audit.json`, `development_frontier.png` / `.pdf`: validation-only
  verification scope and quality/traffic scatter plots.

The campaign ends with frozen development choices. It does not run the old
confirmation seeds again or start a new test campaign. A useful selected
operating point should subsequently receive a separately declared paired
confirmation on fresh partition/SGD seeds with calibrated Ström fixed. Any gain
does not by itself isolate the causal value of Event's leak or full reset.
