# Bounded validation-only Ström development study

Repository: **j-cap/2026_NeuromorphicFL**, the research-code repository.
Entry point: `experiments/strom_codec_development.py`.
Data/evaluation adapter: `experiments/strom_development_data.py`.

## Purpose

Calibrate the existing Ström-style residual-pulse baseline using validation
predictive quality and actual unicast traffic before a full comparative
confirmation. The pilot validated the common codecs and revealed large residual
backlogs, but did not select a final baseline. Event remains fixed throughout.

This is a **development study**, with no official CIFAR-10 test evaluation and
no held-out multi-seed confirmation command. The final output is a reproducible,
frozen configuration selection, not a new claim of superiority.

## Genuine training/validation separation

Only `data_batch_1` through `data_batch_5` are parsed. The official `test_batch`
is never parsed or evaluated. No prior official-test curve is loaded to make
the new decisions. Source/runtime identities are checked against the recorded
environment using the old protocol and source hashes only.

Before partitioning, hold out **500 randomly selected images per class** from
the 50000 training images, using split seed **64500**. This creates a balanced
5000-image validation set. Partition the remaining **45000 images** across ten
clients using development partition seed **4500**. Each client receives 4500
examples: 2475 from its dominant class and 225 from each other class, preserving
the 55% dominant-class allocation of the strong-skew regime.

Training diagnostics use a fixed 5000-image subset of those 45000 training
images. It is disjoint from validation. Validation images participate in no
local gradient step. The generic host's official-test slots are kept empty;
a separate checkpoint adapter evaluates only the explicit validation arrays.

`split_indices.csv` records every original training-row index, class, client,
split membership, local ordering and training-diagnostic ordering.
`split_manifest.json` records ordered index hashes and the split specification.
Disjointness, full index coverage, class balance and exact client allocations
are verified before fitting and again when analyzing results.

All methods in this study use this same split. **Event must run again** to
supply the matching communication budget. Its older full-training pilot curves
cannot be reused on the new validation split. The resulting validation scores
must not be compared directly with old official-test scores.

## Fixed candidate grid and optimizer

| Quantity | Fixed choice |
|---|---|
| Model | Compact CIFAR CNN, 20570 parameters |
| Participation | Ten clients, full participation |
| Partition / SGD seeds | 4500 / 84500 |
| Initialization | Seed 7777, scale 1 |
| Local computation | Five SGD steps, batch 32, learning rate 0.05 |
| L2 coefficient | 0.0005 |
| Ström thresholds | 0.0025, 0.01, 0.025, 0.05, 0.10 |
| FedAvg-equivalent Ström gains | 1, 2, 4 |
| Native server gains | 0.05, 0.10, 0.20 |
| Screening / final horizons | 600 / 1800 communication rounds |
| Validation samples | Round 1 and every 60 rounds |

The Cartesian grid contains **15 Ström settings**, plus one fixed Event
reference. The nonleaky accumulated residual, subtractive reset, one pulse per
coordinate/client/round, coupled threshold/pulse amplitude and exact float32
update order are unchanged. Both pulse methods receive the same lossless
position coding and exact aggregated-downlink portfolio.

The archived residual accumulates `(p_i / local_lr) * local_delta_i`.
Native gain `gamma = local_lr * equivalent_gain` gives model pulse magnitude
`gamma * threshold`. Vary threshold and gain together rather than assuming that
the old threshold remains calibrated after changing the gain. The Event
threshold, leak, reset and independently scheduled quantum remain frozen.

Primary communication cost is `both_codecs` total conservative-unicast traffic,
including initialization, framing, padding, per-client requests and downlink
replication. Logical-broadcast counters are supplementary and preserve the
earlier accounting convention. Audit timings include reference encodings and
decoding checks; they are not deployment runtimes.

## Selection rules and bounded extension

All selection uses **unregularized validation cross-entropy at a fixed horizon**.
Break exact CE ties by lower total unicast traffic, then lexicographic point ID.
Do not select a best validation checkpoint or stop early using validation.
Training objective, accuracy, residual norm and residual maximum are diagnostics.
Residual growth alone never excludes a finite candidate.

At round 600, select:

1. **Quality:** lowest validation CE among finite Ström candidates.
2. **Within Event budget:** lowest validation CE among candidates whose total
   unicast cost is at most the same-horizon Event reference cost on this split.

If no candidate meets the budget, record the constrained choice as **null**.
The nearest-traffic candidate, selected by minimum absolute log cost ratio,
is then only a diagnostic/promotable fallback; it is never relabelled feasible.

After the initial 15-point grid, perform **one boundary-extension pass**:

- Examine the initial quality winner and budget winner (or diagnostic fallback).
- For a winner on the initial threshold boundary, add the adjacent value:
  half the minimum or twice the maximum, holding the winner's gain fixed.
- For a winner on the initial gain boundary, add half the minimum or twice the
  maximum, holding its threshold fixed.
- If both coordinates are on boundaries, include their outward corner too.
- Deduplicate both winners' proposals. This adds **at most six** candidates.
  There is no recursive extension after inspecting their results.

Freeze the boundary decision with its source receipt hashes before executing
the extra points. Then recompute quality and budget choices over the complete
screened grid. Promote the distinct selected Ström IDs: **at most two**. If the
budget is infeasible, promote quality plus the nearest-traffic diagnostic.

Continue those finalists and Event to **1800 rounds**, then apply the same
quality and actual-Event-budget rules at round 1800. An infeasible final budget
remains null; no extra candidate is automatically promoted after inspecting
the final results. Numerical failures stay visible in the full ledger and are
excluded from selection as failed runs. A failed Event reference or absence
of any finite Ström candidate blocks selection with an explicit error.

The final choice is among the promoted finalists. It is **not an exhaustive
1800-round optimum over all initial candidates**. Screening can miss a setting
that improves late, and a boundary winner does not establish global optimality.
The search and its single extension are bounded rather than open-ended.

This campaign uses validation CE because the question concerns predictive
quality. The earlier STC development campaign used regularized training loss;
the final paper must disclose that distinction and align selection policies
where needed before claiming uniform tuning parity.

## Exact continuation and evidence

Every host starts with an **1800-round configuration identity**, while screening
stops at round 600. The same local checkpoint resumes at 601, preserving model,
client replica, residuals, RNG, message counters, training diagnostics and
validation history. The 600-round receipt/CSVs remain intact. The extended
trajectory's screening prefix is checked against them before final results are
accepted; wall-clock audit timings are excluded from that equality check.

Validation metrics and the corresponding model are saved atomically in one
NPZ. Repeating a command resumes incomplete jobs and skips verified completed
or explicitly failed jobs. Configuration, source, runtime, raw training-data
or split changes cannot be mixed into an existing campaign.

If local checkpoints are lost, the same environment can regenerate a promoted
trajectory from round 1. Its saved 600-round prefix must still reproduce before
the 1800-round result is accepted. NPZ files are local recovery files and are
not committed. The index ledger and scientific CSV/JSON evidence are committed.

## Run locally

The default `run` completes screening, the optional bounded extension, and
full-horizon finalist checks automatically. It starts **16 initial runs to
600**, may add **up to six** more, then extends **Event plus at most two Ström
finalists** to 1800. It does not start a multi-seed test campaign.

From Anaconda Prompt in the **code** repository:

```bat
cd /d C:\Users\WeberJ\2026_NeuromorphicFL
conda activate neuromorphicfl-resnet14
git pull --ff-only
python experiments/strom_codec_development.py verify
python experiments/strom_codec_development.py preflight
python experiments/strom_codec_development.py run --workers 2
python experiments/strom_codec_development.py analyze

git add experiments/results/strom_codec_development_v1
git commit -m "Add validation-only Strom development results"
git push origin main
```

`verify` uses synthetic inputs, synthetic training-row identities and frozen
source metadata. It parses no real CIFAR images and performs no research
training. The preflight should print
`Runtime matches original environment: True` in the original Windows conda
environment. Training additionally checks all five raw training-batch hashes.

For progress or recovery:

```bat
python experiments/strom_codec_development.py status
python experiments/strom_codec_development.py run --workers 2
python experiments/strom_codec_development.py analyze
git add experiments/results/strom_codec_development_v1
git commit -m "Add completed Strom development selection results"
git push origin main
```

Use `--workers 1` to reduce CPU/RAM concurrency. To review the screening stage
first, use `screen --workers 2` in place of `run --workers 2`, followed by
`analyze`; later use `finalize --workers 2` and `analyze`. The frozen selection
rules apply in both workflows. Publish each completed stage with the same
`git add` / `git commit` / `git push` pattern above.

## Outputs

Directory: `experiments/results/strom_codec_development_v1/`.

- `protocol.json`, `split_indices.csv`, `split_manifest.json`: frozen study,
  source/runtime/training-data identities and exact split allocation.
- Per point/stage: round records, validation/training evaluations, four-variant
  traffic summaries/history, and hashed completion or numerical-failure receipt.
  No `test_*` metric columns are written.
- `screening_initial.csv`: all 16 initial outcomes, including failures.
- `boundary_extension.json`: one immutable, receipt-linked extension decision.
- `screening.csv`, `screening_selection.json`: complete grid and promotion choices.
- `finalists.csv`: all promoted 1800-round outcomes and Event's matching budget.
- `frozen_selection.json`: final quality/constrained choices, configurations,
  source receipt hashes, and explicit infeasibility where applicable.
- `development_frontier.png` / `.pdf`: validation CE versus unicast traffic at
  screening and finalist horizons.
- `analysis_audit.json`: evidence verification and validation-only scope.

Root CSV/JSON evidence and these two figures are Git-enabled. Checkpoints and
lock files remain ignored. After reviewing these outputs, decide how to carry
the frozen Ström settings into the common full-training multi-seed comparison.
