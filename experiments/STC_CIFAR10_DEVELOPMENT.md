# STC compact-CIFAR development campaign, version 1

Repository: **j-cap/2026_NeuromorphicFL**, the research-code repository.
The separate **2026_NeuromorphicFL_Paper** repository contains the manuscript.

This stage selects STC configurations and checks their numerical behaviour in
the existing compact-CIFAR setting. It does not run the final ten-seed evaluation
or update manuscript claims. The search and rules below are declared before its
research results are generated.

## Fixed common protocol

| Item | Setting |
|---|---|
| Dataset/model | CIFAR-10, existing compact CNN |
| Clients | 10; full participation |
| Partition | Existing strong skew: 55% dominant class, 5% for each other class |
| Development partition seed | 4500 |
| Local-SGD random seed | 84500 |
| Local work | 5 SGD steps per client per round; batch size 32 |
| Local learning rate | 0.05 |
| Regularization | 0.0005 |
| Initialization | Existing He initialization, scale 1, seed 7777 |
| Horizon | 1800 rounds for every development point |
| Selection metric | Final regularized objective on the fixed 5000-example training-evaluation subset |
| Evaluation interval | 60 rounds, plus round 1 |
| Resume checkpoints | Every 60 rounds and at completion |

All points share the same partition, initialization, minibatch RNG sequence,
local optimizer and horizon. Seed 4500 is outside the compact-CIFAR final
partition-seed set 3500, 3600, ..., 4400. These are separate random partitions of
the same CIFAR-10 training set, not separate datasets or unseen validation images.
The training-objective selection rule follows the existing dense-schedule audit.

The development runner redirects its secondary evaluation slot to the same
training subset and removes generic `test_*` and accuracy columns from exports.
It never evaluates the CIFAR-10 test set while selecting configurations. Training
objective/traffic values in this campaign are not final generalization results.

## Declared STC search

The symmetric retained fractions are:

```
p_up = p_down in {0.001, 0.0025, 0.005, 0.01, 0.025,
                 0.05, 0.1, 0.25, 0.5, 1.0}
```

Also include `(p_up, p_down) = (0.0025, 0.01)` and `(0.01, 0.0025)` to check
whether a moderate upload/download asymmetry matters. Each of these 12 pairs is
run with constant server gain 1 and 2: **24 STC configurations**. Gain 1 uses the
reference server update; gain 2 is the documented common optimizer extension,
applied before server error feedback. No gain schedule is added to STC here.

The wide symmetric range covers aggressive compression through dense ternary
messages. `p=1` still ternarizes; it is not Dense FedAvg. The two asymmetric pairs
provide a bounded sensitivity check rather than an exhaustive two-dimensional
optimization. Any ranking is limited to the declared search.

One additional Event-FedAvg reference uses the existing compact-CIFAR parameters
(`rho=0.999`, threshold 0.025, initial jump 0.005, jump exponent 0.2, server gain 1)
on the same development partition and horizon. It defines the communication
budget. Total: **25 development runs**, all CPU/NumPy runs.

This is an extension of STC development evidence, not a replacement tuning sweep
for every comparator. The existing Dense FedAvg gain/schedule audit and frozen
Ström-style/EF-TopK results remain part of the eventual comparison. Search budgets
and available tuning dimensions must be reported rather than claimed identical.

## Selection before final evaluation

All 25 points must have a validated completion receipt or an explicit numerical
failure record. A failed Event reference blocks budget selection. Numerical STC
failures are retained in the audit trail and excluded from the finite ranking;
configuration, filesystem and codec errors stop execution for investigation.

Select two roles among finite STC candidates:

1. **Quality:** lowest final training objective.
2. **Within Event budget:** lowest final training objective among candidates whose
   total conservative-unicast traffic is at or below the Event reference.

Ties use lower traffic, then point ID. If no candidate meets the budget, the
second role is explicitly empty. One candidate may fill both roles, in which
case the later evaluation need not duplicate it.

Separately record the candidate with the smallest absolute log traffic ratio to
Event, its actual ratio and whether its relative budget gap is at most 10%.
This is a diagnostic, not an automatic claim of matched traffic. STC's native
compressed downlink may make close traffic matching unattainable. Budget-feasible
comparisons remain meaningful without requiring both methods to spend all bits.

`frozen_selection.json` records both roles, protocol/evidence hashes and any
failures. An existing different selection is not overwritten. No final test
accuracy or evaluation-partition result enters this selection.

## Local commands

Use the existing `neuromorphicfl-resnet14` environment. From Anaconda Prompt:

```bat
cd /d C:\Users\WeberJ\2026_NeuromorphicFL
git pull --ff-only
conda activate neuromorphicfl-resnet14
python experiments/stc_stage1.py verify
python experiments/stc_cifar10_development.py smoke
python experiments/stc_cifar10_development.py develop --workers 2
```

The smoke uses synthetic inputs, tests exact checkpoint/resume and spawned-worker
signatures, and writes no research results. `develop` downloads CIFAR-10 only if
needed. Each worker uses one BLAS/OpenMP thread. Two workers are a conservative
default; the CLI permits 1--8 workers. The compact CNN is trained on CPU even when
the environment also has CUDA/PyTorch installed.

To inspect the grid or current completion state:

```bat
python experiments/stc_cifar10_development.py plan
python experiments/stc_cifar10_development.py status
```

After all runs finish:

```bat
python experiments/stc_cifar10_development.py verify
python experiments/stc_cifar10_development.py select
python experiments/stc_cifar10_development.py analyze
```

Verification and analysis read CSV/JSON evidence without requiring a dataset
download or the original training machine. Running/resuming additionally requires
the recorded runtime, data files and source version to match. `select` freezes
choices; it does not launch final evaluation runs. Analysing a numerical failure
does not turn it into a successful result.

## Restart and evidence handling

Rerun the same `develop --workers 2` command after interruption. Completed results
are checksum validated and skipped; unfinished points resume their latest atomic
checkpoint, repeating up to 60 rounds of work. A completed checkpoint can also
recover an interruption between training completion and CSV/receipt writing.
Models, RNG, client/server residuals, history and traffic counters are all saved.
OS point locks prevent duplicate workers and release automatically on exit.

`protocol.json` freezes configurations, normalized source hashes, runtime/NumPy
build information, thread settings and input-batch hashes. Changed inputs or
versions are rejected rather than mixed into one campaign. Resolve the cause
before deliberately introducing a new campaign version.

Outputs are in `experiments/results/stc_cifar10_development_v1/`:

- `protocol.json`: the common execution contract.
- Per-point summary/history CSVs and completion/failure receipt JSONs.
- `frozen_selection.json`: the chosen roles and supporting evidence hashes.
- `development_summary.csv`, `development_ranking.csv`,
  `development_frontier.csv`: development-only audit products.

The CSV/JSON files are allowed by `.gitignore`. Checkpoint NPZ files, OS lock
files and temporary writes remain local and ignored. After completion:

```bat
git add experiments/results/stc_cifar10_development_v1
git commit -m "Add STC compact-CIFAR development results"
git push
```

The next decision is based on these development results: inspect stability,
selected configurations and actual traffic separation, then prepare the frozen
ten-seed evaluation and the PyTorch ResNet integration. Existing manuscript
evidence remains untouched at this stage; subsequent manuscript changes stay red.
