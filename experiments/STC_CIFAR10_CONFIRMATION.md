# STC compact-CIFAR confirmation, version 1

Research-code repository: **j-cap/2026_NeuromorphicFL**.
Entry point: `experiments/stc_cifar10_confirmation.py`.
This protocol is committed before the new STC test results are generated.

## Scientific question and fixed choices

Does Event-FedAvg provide a useful test-accuracy/communication operating point
when compared with the development-selected STC protocol in the same synchronous
server-based FL setting?

| Role | Development point | Upload/down retained fractions | Constant server gain |
|---|---|---|---:|
| Primary | `stc_u1_d1_g2` | 1 / 1 | 2 |
| Predeclared secondary | `stc_u0p25_d0p25_g2` | 0.25 / 0.25 | 2 |

The primary is the original frozen development selection for both quality and
within-Event-budget roles. It uses dense ternary messages with two levels of
error feedback. The secondary was chosen during development analysis because it
had nearly the same regularized objective at substantially lower traffic. Its
addition is explicit and precedes the new test evaluations. The original
development selection and all its source/data/receipt records are preserved.
There is no test-driven reselection or further compression/gain search here.

## Common training protocol

- Existing compact CIFAR-10 CNN, 20570 parameters; ten fully participating clients.
- Existing strong partition: each client has 55% dominant-class and 5% of each
  other-class samples; the full 50000-example training set is retained.
- Ten partition seeds: 3500, 3600, ..., 4400. SGD seeds: 83500, 83600, ..., 84400.
  Development partition 4500 is excluded.
- Existing initialization: seed 7777, He initialization with scale 1.
- 1800 rounds, five local SGD steps, batch size 32, local learning rate 0.05.
- **Regularization stays at 0.0005 for every method.** It enters the local gradient
  exactly as in the existing benchmark. No zero-regularization ablation is added.
- Evaluation at round 1 and every 60 rounds through 1800. The real CIFAR-10 test
  slot is used in this confirmation; its results never select configurations.

Total: **20 new STC runs**, using the unchanged NumPy runner and STC codec.
This experiment precedes any broader benchmark or PyTorch ResNet campaign.

## Archived references and compatibility

The preflight validates ten matched runs for each of:

| Comparison ID | Archived operating point | Role |
|---|---|---|
| `event` | Event-FedAvg, existing threshold/jump schedule | Primary comparator |
| `dense_constant` | Dense FedAvg, constant server gain 2 | Core reference |
| `dense_scheduled` | Dense FedAvg, gain 4 times (1+round/100)^(-0.3) | Existing development-selected fairness control |
| `ef_topk` | EF-TopK, retained fraction 0.05 | Core reference |
| `strom_style` | Ström-style FedAvg residual pulses, threshold 0.0025 | Archived adaptation |

The four P19 references and M3 scheduled-dense control are reused rather than
rerun automatically. Checks cover model dimension, seeds, horizon, local steps,
learning rate, recorded compressor/gain settings, history metadata, endpoints
and exact communication reconstruction. The frozen dense schedule is retained
explicitly; the analysis does not pick between dense controls after seeing STC.

Some shared legacy settings, including batch size and regularization, are not
present in old per-run CSVs. They are recovered from the declared archived
drivers and common CIFAR configuration. The preflight reports these missing
fields. Legacy per-run runtime and raw-batch fingerprints are unavailable, so
compatibility means recorded metadata plus declared-driver compatibility, not
independent verification of the original execution environment.

The Ström row represents the explicitly labeled server-FedAvg adaptation. Its
existing implementation integrates weighted local deltas divided by the local
learning rate, then transmits and subtracts a threshold quantum. The comparison
does not claim to reproduce Ström's original asynchronous system. Broader baseline
fidelity claims remain subject to the manuscript's separate audit.

## Predeclared outcomes and analysis

The primary endpoint is the paired final-test-accuracy difference in percentage
points: **STC primary minus Event-FedAvg**, at the fixed 1800-round horizon.
The secondary STC point is reported with its role intact.

Additional reported outcomes are unregularized test cross-entropy, upload and
download separately, broadcast and conservative-unicast total traffic, and
traffic to fixed test-accuracy targets **45%, 50% and 55%**. Training objective,
training cross-entropy, L2 penalty and inferred parameter norm are diagnostics.
The L2 contribution is never presented as prediction improvement.

Accuracy-target cost is measured at the **first saved checkpoint** meeting the
target. No crossing time or traffic is interpolated between 60-round samples.
An unreached target is censored at the horizon; the horizon is not counted as
an achievement. Target comparisons report both methods' hit counts and the
number of seed pairs in which both hit. Ratios on those pairs are explicitly
conditional on common hits.

For archived references, exact prefix traffic follows from their recorded
synchronization modes: all Event/EF rounds replay and all dense/Ström rounds
use checkpoints in this archive. Endpoint formulas must reproduce recorded
costs. Mixed-mode archives without per-round choices are rejected rather than
assigned estimated prefix traffic. STC prefix costs come from the actual native
server-packet counters.

Endpoint differences use paired, unadjusted descriptive 95% Student-t intervals
across the ten seed pairs. Traffic ratios use paired log-scale intervals and a
geometric-mean estimate. Secondary comparisons are identified as such; no
multiplicity-adjusted significance claim is made. An accuracy interval spanning
zero is not evidence of equivalence. These repetitions quantify training-seed
and partition variation on the fixed dataset, not a new sample of test datasets.

All 20 declared attempts must be accounted for. Numerical failures are retained
with seed and reason. The analysis writes a completion audit and blocks fixed-
horizon inference if it would require silently dropping failed seeds. There is
no check that requires Event-FedAvg or STC to win or remain on a frontier.

## Local execution

Use the same `neuromorphicfl-resnet14` environment used for development. From
Anaconda Prompt, in the **code** repository:

```bat
cd /d C:\Users\WeberJ\2026_NeuromorphicFL
git pull --ff-only
conda activate neuromorphicfl-resnet14
python experiments/stc_stage1.py verify
python experiments/stc_cifar10_confirmation.py smoke
python experiments/stc_cifar10_confirmation.py preflight
python experiments/stc_cifar10_confirmation.py confirm --workers 2
```

The smoke uses only synthetic data and temporary files; it exercises both STC
points, exact checkpoint/resume, native traffic identities and spawned-worker
signatures. The preflight reads committed development and reference evidence
without downloading data. Its runtime-match line should be `True` on the original
training environment. The actual training command requires the recorded
development runtime and matching data-batch hashes. Read-only verification and
analysis can run on another machine without that runtime or raw dataset.

Each CPU worker uses one BLAS/OpenMP thread; 1--8 workers are supported. Restart
the same `confirm --workers 2` command to validate/skip completed runs and resume
unfinished runs from atomic 60-round checkpoints. The checkpoints preserve
parameters, RNG, both residual levels, history and all communication counters.

```bat
python experiments/stc_cifar10_confirmation.py status
```

For a single declared job, use `point --point-id stc_primary --seed-index 0`
(or `stc_sparse_secondary`, with an index from 0 through 9).

After the full campaign:

```bat
python experiments/stc_cifar10_confirmation.py verify
python experiments/stc_cifar10_confirmation.py analyze
```

The analysis produces endpoint tables, paired per-seed differences/ratios,
paired summaries, target-crossing records and target summaries. Actual research
results are evaluated only after the user runs the campaign; smoke outputs are
implementation checks.

## Evidence and restart rules

Outputs are under `experiments/results/stc_cifar10_confirmation_v1/`.
`protocol.json` freezes the two roles, all configurations, seeds, endpoints,
targets, source fingerprints, development-selection identity, reference-file
fingerprints, execution runtime and dataset-batch identities before training.
Existing incompatible contracts are rejected.

New text receipts hash UTF-8 files after LF/CRLF normalization, with an explicit
`sha256_utf8_lf_v1` format label. Writers emit LF. Thus Git checkout policies do
not invalidate the result, while altered numeric values still fail verification.
This does not modify the original development receipts or their hash convention.

CSV/JSON evidence is trackable. Checkpoint NPZ files, OS lock files and temporary
writes remain ignored. Commit results from the code repository:

```bat
git add experiments/results/stc_cifar10_confirmation_v1
git commit -m "Add paired STC compact-CIFAR confirmation results"
git push
```

The resulting comparison decides whether to expand the numerical scope or
reconsider the current efficiency claim. Manuscript evidence and text are updated
after that assessment; new manuscript changes remain red.
