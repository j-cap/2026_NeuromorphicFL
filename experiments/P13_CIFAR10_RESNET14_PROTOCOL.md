# P13: CIFAR-10 ResNet-14 architecture extension

Date frozen: 2026-09-09

## Question

Does Event-FedAvg retain a useful communication--quality trade-off on a deeper
residual architecture, without starving complete parameter groups under its
global coordinate threshold?

This is an architecture-extension experiment. It does not claim scalability to
large contemporary networks.

## Architecture

- CIFAR-style ResNet-14 with depth rule `6n+2`, `n=2`.
- One 3x3, 16-channel stem followed by stages with 16, 32, and 64 channels.
- Two residual blocks per stage. The first block of stages 2 and 3 downsamples.
- Learned 1x1 projection shortcuts at the two stage transitions.
- GroupNorm with four groups after each convolution, including projections.
- Global average pooling and a ten-class linear head.
- Exactly 175,258 trainable float32 parameters.
- No data augmentation, momentum, or architecture-specific optimizer is added,
  so local learning remains comparable with the existing CIFAR campaign.

GroupNorm is used to avoid an additional decision about local versus aggregated
BatchNorm statistics under the non-IID partition.

## Data and training

The dataset and strong label-skew partition construction match P3: ten clients,
5,000 training examples per client, and a 55% dominant-class fraction. Each
round has ten active clients, five local SGD steps, batch size 32, local learning
rate 0.05, and weight decay 5e-4. The initial evaluation horizon is 180 rounds,
with metrics recorded every 15 rounds.

- Development partition: 4100.
- Three-seed pilot: 4200, 4300, 4400.
- Seven extension seeds, used only after passing the validity gate: 4500--5100
  in increments of 100.
- Training seed: `90000 + partition_seed`, paired across methods.

## Development selection

Only final training cross-entropy on partition 4100 selects a quality setting.
The grid contains four dense server gains, seven Event-FedAvg
`(threshold, initial quantum)` combinations, Sign-EF, three EF-TopK fractions,
and three Strom thresholds. Event-FedAvg fixes `rho=0.999`, `r0=100`, and
`alpha=0.3` during this grid. All selected settings are frozen before held-out
evaluation.

The grid is intentionally modest. It tests the scale of the threshold and
quantum without turning the architecture extension into a broad post-hoc
search.

## Predeclared validity gate

Proceed from the three-seed pilot to the seven extension seeds only when:

1. every recorded metric is finite;
2. dense FedAvg achieves at least 20% mean test accuracy, establishing that the
   training setup learns beyond chance;
3. every semantic ResNet group emits at least one Event-FedAvg coordinate event
   in every pilot seed;
4. for every selected method, the absolute mean relative training-CE change
   over the final 20% of rounds is below 1%, establishing a practical plateau
   at the selected horizon.

The gate tests experimental validity. Event-FedAvg is not required to outperform
a baseline. A valid unfavorable comparison remains part of the scientific
result. If only the stability condition fails, extend the horizon without
retuning the selected method settings and repeat the stability assessment.

## Layer diagnostics

For every Event-FedAvg round and semantic parameter group, record:

- coordinate-event count and firing fraction;
- number of clients with at least one event;
- mean absolute pre-reset evidence.

The groups are the stem, six residual blocks, and classifier head. These
diagnostics detect layer starvation that aggregate event counts would hide.

## Commands

### RTX workstation

The CUDA-native implementation is
`experiments/p13_cifar10_resnet14_torch.py`. From the repository root, create
the pinned Conda environment and verify the GPU:

```bash
bash tools/p13_workstation.sh setup
bash tools/p13_workstation.sh smoke
```

The smoke run downloads CIFAR-10 and executes two dense rounds. The test
campaign then runs all development configurations on the single development
partition, freezes the selected configuration for each method, and runs one
held-out seed per selected method:

```bash
bash tools/p13_workstation.sh dense-audit 900
```

This exploratory convergence audit runs the four dense server gains for 900
rounds on development partition 4100. It uses the distinct tag
`dense-audit-r900`, writes one history and metadata set per gain, and creates a
combined `dense-audit-r900_seed4100_summary.csv`. These exploratory results are
not consumed by development selection. Interrupted runs resume at their latest
15-round checkpoint. A different horizon can be supplied in place of `900`.

After choosing an adequate fixed horizon from the learning curves, update the
frozen protocol consistently before running the test campaign:

```bash
bash tools/p13_workstation.sh test
```

Inspect `experiments/results/p13_cifar10_resnet14/` after the test. Every point
has a final CSV, learning-history CSV, and JSON environment record. Event points
also have a semantic-group activity CSV. Interrupted points resume from a
checkpoint written at every 15-round evaluation. Completed checkpoints are
removed automatically.

Once the timing and single-seed results are sensible, run the remaining pilot
and conditional extension:

```bash
bash tools/p13_workstation.sh full
```

The full command fills the remaining two pilot seeds, evaluates the predeclared
gate, and launches the seven extension seeds only if the gate passes. Existing
completed points are skipped, so both commands are safe to rerun after an
interruption.

The result CSV and JSON files are intentionally unignored for Git. Checkpoints
remain ignored and must not be committed.

### NumPy reference

```bash
PYTHONPATH=src python experiments/p13_cifar10_resnet14.py smoke
PYTHONPATH=src python experiments/p13_cifar10_resnet14.py protocol
```

Run each configuration on the development partition, then freeze the selection:

```bash
PYTHONPATH=src python experiments/p13_cifar10_resnet14.py point --config CONFIG --partition-seed 4100 --tag dev
PYTHONPATH=src python experiments/p13_cifar10_resnet14.py select
```

For each frozen method and each pilot seed:

```bash
PYTHONPATH=src python experiments/p13_cifar10_resnet14.py heldout --selection experiments/results/p13_cifar10_resnet14/selection.json --method METHOD --partition-seed SEED
PYTHONPATH=src python experiments/p13_cifar10_resnet14.py assess --selection experiments/results/p13_cifar10_resnet14/selection.json
```

The machine-readable protocol is generated at
`experiments/results/p13_cifar10_resnet14/protocol.json`.
