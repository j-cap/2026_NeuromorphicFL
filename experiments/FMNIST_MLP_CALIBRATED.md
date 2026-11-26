# Fashion-MNIST MLP: parallel calibration and common-seed confirmation

Research repository: **2026_NeuromorphicFL**. This is a new third-architecture
panel. The completed CIFAR panels and manuscript evidence remain frozen.

## Run the complete campaign in Windows Command Prompt

The existing `neuromorphicfl-resnet14` environment is suitable. This MLP uses
**NumPy on the CPU**; it does not require a GPU or new packages beyond NumPy,
pandas and Matplotlib already used by the project.

```bat
cd /d C:\Users\WeberJ\2026_NeuromorphicFL
conda activate neuromorphicfl-resnet14
git pull --ff-only
python experiments/fmnist_mlp_calibrated.py verify
python experiments/fmnist_mlp_calibrated.py preflight --workers auto
python experiments/fmnist_mlp_calibrated.py run --workers auto

git add experiments/results/fmnist_mlp_calibrated_v1
git diff --cached --check
git commit -m "Add calibrated Fashion-MNIST MLP five-method results"
git push origin main
```

`run` completes screening, continuation, validation-only selection, all fifty
fresh confirmation fits, and the saved-results analysis. Confirmation begins
only after `frozen_selection.json` is written and independently reproduced from
the validation receipts. There is no intermediate manual selection step.

`auto` uses approximately one process per physical core (half the detected
logical CPUs, at least one, capped at twelve). Each process has **one BLAS
thread**. On a 16-logical-CPU machine this means eight workers; on a
24-logical-CPU machine, twelve. This is a heuristic; explicitly request a
worker count if desired:

```bat
python experiments/fmnist_mlp_calibrated.py run --workers 8
git add experiments/results/fmnist_mlp_calibrated_v1
git diff --cached --check
git commit -m "Add calibrated Fashion-MNIST MLP five-method results"
git push origin main
```

Use one coordinator at a time. Repeating the **same** run command resumes
checkpoints and skips finished or terminally failed jobs. The worker count can
be changed on resume. Ctrl+C requests a stop after active workers save their
next checkpoint; keep the terminal open until they finish saving.

Monitor from another terminal:

```bat
python experiments/fmnist_mlp_calibrated.py status
```

## Scientific protocol

| Item | Frozen choice |
|---|---|
| Task/model | Fashion-MNIST; tanh MLP 784–32–16–10, 25,818 parameters |
| Federation | Ten synchronous clients; full participation; 55% dominant label and 5% each other label |
| Shared local learning | Five SGD steps, batch 32, learning rate 0.1, L2 coefficient 0.0001 |
| Initialization | Seed 7777, scale 0.5 for every method/seed |
| Development | Training split only: 54,000 train / 6,000 validation; 600 validation examples per class; 5,400 train examples/client |
| Development seeds | Partition 11000; split 21000; SGD 101000 |
| Budget per family | 18 candidates × 500 rounds; top three continue to 1,500; at most 12,000 unique training rounds |
| Promotion/selection | Lowest final validation CE; ties by total unicast bits, then configuration identifier |
| Confirmation | Ten new common partition seeds 11100–12000, step 100; SGD seeds 101100–102000 |
| Full-training data | All 60,000 train examples, 6,000/client; official 10,000-image test set |
| Predictive endpoints | Fixed round 1,500 accuracy and unregularized test CE; worst-class accuracy secondary |
| Evaluation | Round 1 and every 50 rounds; training diagnostic subset of 5,000; one forward pass per evaluation batch of 512 |
| Target accounting | First saved crossing of **80%, 85%, 87%** test accuracy; no interpolation/stability requirement |
| Primary communication | Actual encoded uplink and replicated downlink, initialization, byte padding and 32-bit per-client round requests |
| Secondary communication | Logical broadcast, charging the common server payload once |
| Uncertainty | Descriptive paired 95% Student-t intervals; cost ratios on log scale; no multiplicity adjustment |

Every confirmation run starts fresh; development checkpoints are used only to
continue the corresponding tuning candidate. The 1,500-round common horizon
comes from the earlier MLP horizon audit, with no method-specific test stopping.
The three accuracy targets are declared in code before this campaign runs.
A target miss remains censored, with coverage reported explicitly. A ratio
uses only common hits, and fewer than two pairs receive no confidence interval.
The fixed final-round endpoints complement the possibly transient crossings.

The total upper budget is **60,000 unique tuning rounds + 75,000 confirmation
rounds = 135,000**, spread across independent jobs. Screening and finalist files
retain the common prefix, so saved round-record counts include those deliberate
duplicates. There are at most 105 tuning-stage receipts and 50 confirmation
receipts. Numerical failures are retained; there are no replacement candidates,
extra seeds, automatic gain extensions or test-driven method substitutions.

Fresh-seed provenance is checked against recorded prior JSON metadata. The
official Fashion-MNIST test set was inspected in earlier work; new seed pairs
are fresh runs, not a new untouched dataset. Selection uses this campaign's
validation CE. Bounded equal budgets do not guarantee global hyperparameter
optimality or identical search efficacy.

## Search grids: 18 settings for each method

| Method | Grid |
|---|---|
| Event | Threshold {0.0125, 0.025, 0.05} × initial quantum {0.0025, 0.005, 0.01} × decay exponent {0.1, 0.2}; leak 0.999, full reset, persistent state, gain one |
| Ström-style | Threshold {0.000625, 0.00125, 0.0025, 0.005, 0.01, 0.02} × native gain {0.1, 0.3, 1}; no leak, subtractive residual |
| Dense | Initial gain {0.25, 0.5, 1, 1.5, 2, 4} × decay exponent {0, 0.1, 0.3} |
| STC | Symmetric per-tensor upload/download retention {0.01, 0.05, 0.25, 1} × gain {0.5, 1, 2, 4}, plus upload 0.01 or 0.25 / download 1 at gain 1 |
| EF-TopK | Global retained fraction {0.01, 0.025, 0.05, 0.1} × gain {0.5, 1, 2, 4}, plus 0.005 and 0.25 at gain 1 |

The schedule scale is 100. Ström's native gain equals local learning rate times
its FedAvg-equivalent gain, so this native grid corresponds to effective gains
{1, 3, 10}. It includes the earlier residual-pulse quality anchor. Dense's
gain is the FedAvg server gain directly. STC retention one still ternarizes
and retains both client/server error-feedback residuals; it is not Dense.
EF-TopK retains its native global support, float32 values and client residual.
The common FedAvg host adapts Ström's residual encoder; it is not a reproduction
of the original peer-to-peer system.

## What speeds up the campaign

- Independent configurations and independent method/seed fits use a **spawn**
  process pool, compatible with Windows. Clients within one fit keep their
  original local-update and RNG order.
- Workers map the same read-only normalized image files. Client minibatches
  use indexed views instead of copying the full training dataset per worker.
- One BLAS thread per worker prevents nested multithreading from oversubscribing
  the CPU. No threadpoolctl, multiprocessing server or GPU setup is required.
- Selected 500-round fits continue with their model, all residual/evidence
  states, RNG, counters and evaluations intact. Their exact prefix is checked
  against the saved screening files before selection is frozen.
- Existing byte-equivalent bound-one support codecs accelerate all compressed
  families. Pulse downlink selection calculates the exact ordered-frame length
  first and constructs that frame only when it wins; selected bytes and tie
  decisions match the original encoder.
- Evaluation computes CE and predictions from one batched forward pass. Its
  float64 loss accumulation is fixed for every family and never affects SGD.

Worker scheduling and worker count do not change model or wire results. Timing
columns describe this execution environment; parallel campaign timings are not
a controlled deployment-latency comparison between methods.

## Files and checks

The output directory is `experiments/results/fmnist_mlp_calibrated_v1/`.
The source/runtime/data contract, development allocation and selection are
immutable. Per-stage receipts verify all saved CSV hashes. Each raw output
identifies Fashion-MNIST and the MLP architecture.

Important outputs: `screen_summary.csv`, `finalist_summary.csv`,
`frozen_selection.json`, `selected.csv`, `confirmation_completion.json`,
`confirmation_runs.csv`, `confirmation_history.csv`, `endpoint_summary.csv`,
`paired_endpoints.csv`, `target_crossings.csv`, `paired_targets.csv`,
`comparison.pdf`, `comparison.png`, `analysis_audit.json`.

Models, residuals, RNG checkpoints, shared data and live progress remain local
and are ignored by Git. Keep the local checkpoints until the campaign finishes.
The tracked CSV/JSON files and plots are sufficient for saved-result analysis;
full training replay additionally needs the original data and numerical runtime.
Training resume enforces the original Python/NumPy/pandas/BLAS build profile.
Saved-result analysis can run on another platform and does not require Torch,
the image cache or training checkpoints.

`verify` uses only synthetic inputs. It checks native operator/model/residual
parity, exact resume, byte-identical wire selection, one-versus-two-worker
results, immutable receipts, validation-only selection, portable read-only
analysis, target censoring and independent cumulative accounting.
It provides no predictive-performance evidence for this new panel.

## Optional: split tuning and confirmation

To inspect the validation choices before launching final fits:

```bat
python experiments/fmnist_mlp_calibrated.py tune --workers auto
git add experiments/results/fmnist_mlp_calibrated_v1
git diff --cached --check
git commit -m "Add Fashion-MNIST MLP validation calibration and frozen selections"
git push origin main
```

Then run the fresh common-seed confirmation:

```bat
python experiments/fmnist_mlp_calibrated.py confirm --workers auto
git add experiments/results/fmnist_mlp_calibrated_v1
git diff --cached --check
git commit -m "Add Fashion-MNIST MLP ten-seed five-method confirmation"
git push origin main
```

`confirm` also runs the saved-result analysis. Rebuild analysis after editing
only its figure/report code with:

```bat
python experiments/fmnist_mlp_calibrated.py analyze
git add experiments/results/fmnist_mlp_calibrated_v1
git diff --cached --check
git commit -m "Update calibrated Fashion-MNIST MLP analysis"
git push origin main
```
