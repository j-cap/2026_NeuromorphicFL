# ResNet-14 validation boundary check

Repository: **j-cap/2026_NeuromorphicFL**, the research-code repository.
Entry point: `experiments/resnet14_boundary.py`.
New results: `experiments/results/resnet14_boundary_v1/`.

## Question and fixed scope

Does one small extension beyond the selected lower calibration boundaries
change the validation choice for Event or Ström? This is a separate follow-up
to the completed [`ResNet transfer`](RESNET14_TRANSFER_RESULTS.md), designed
after inspecting that pilot. The completed pilot, its choices and its three
official-test confirmation pairs stay frozen.

Use the same 45000/5000 training/validation split, partition seed 7000, split
seed 17000, SGD seed 97000 and initialization 7777. Keep ResNet-14/GroupNorm,
ten fully participating clients, strong 55%-dominant skew, local SGD, L2,
evaluation cadence and actual shared lossless codecs unchanged. No official
test images or saved official-test metrics are read by the new study.

The reused development incumbents are Event threshold **0.01 / quantum
0.0025** and Ström threshold **0.005 / equivalent gain 2**, native gain **0.10**.
Their verified 1800-round development evidence comes from results commit
`f8e740b1a888eee83200d82f9a92333543b16bba`. Their original NPZ checkpoints are
not required because these two trajectories are not continued or retrained.

## Additional grid and equal budget

| Method | Threshold | Initial quantum / equivalent gain | Native server gain |
|---|---:|---:|---:|
| Event | 0.005 | Quantum 0.00125 | 1 |
| Event | 0.01 | Quantum 0.00125 | 1 |
| Event | 0.0125 | Quantum 0.00125 | 1 |
| Ström | 0.0025 | Equivalent gain 2 | 0.10 |
| Ström | 0.005 | Equivalent gain 1 | 0.05 |
| Ström | 0.0025 | Equivalent gain 1 | 0.05 |

Event halves the smallest tested initial quantum, at the selected threshold
and its two neighboring tested thresholds. Leak 0.999, full reset and quantum
schedule exponent 0.2 remain fixed. Ström halves the selected threshold, the
selected gain, or both, retaining its nonleaky subtractive residual and one
pulse per coordinate/client/round. Native Ström gain is local learning rate
0.05 times the equivalent gain. The parameterizations differ; equal budget
means the same number of new candidates and continuation slots per family.

| Stage | New training | Decision |
|---|---:|---|
| Screen | Three candidates per method × 600 rounds = 3600 rounds | Promote one candidate per method by final validation CE |
| Continue | Two finalists × 1200 additional rounds = 2400 rounds | Continue the exact screened trajectory to round 1800 |
| Compare and freeze | No new training | Compare each new finalist with its own reused 1800-round incumbent |

The maximum is **6000 additional communication rounds**, equally divided as
3000 per method, six fresh trajectories and eight new stage receipts. There
are 106 unique evaluation points (round 1 and every 60 rounds), each evaluating
the training diagnostic subset and validation set. Full finalist files repeat
their 600-round prefixes: at maximum the archive contains 7200 round rows and
128 evaluation rows, with two exact prefixes verified. No further grid
extension or official-test confirmation is launched automatically.

## Selection, failures and interpretation

Use final **validation cross-entropy**, then total unicast bits, then point ID
for ties, for both screen promotion and final choice. Accuracy, training
objective and official-test results do not select the configuration. Train the
one promoted finalist for each method even if its screening result is worse
than the incumbent. Final selection includes the incumbent, so an extension
that loses the declared ranking leaves the old choice in place.

Numerical failures remain explicit terminal receipts and ledger rows. If all
three new screens fail for a family, stop with an error and the screening
ledger. If a promoted continuation fails numerically, retain its failure and
the incumbent; do not substitute an extra candidate. Structural errors such
as a missing checkpoint or incompatible runtime propagate and preserve saved
work. Missing or corrupted evidence blocks final analysis.

At completion, `boundary_summary.csv` reports incumbent/new validation CE,
accuracy, traffic ratio and whether the selected point changed.
`comparison.csv` retains all four full-horizon outcomes, including failure
states. `frozen_selection.json` pins the two final choices and records whether
fresh matched confirmation is required. The complete screening and finalist
ledgers and `analysis_audit.json` retain selection provenance and coverage.

If either choice changes, the resulting comparison needs a subsequent fresh
matched full-training confirmation. The already inspected 7100/7200/7300
official-test outcomes cannot confirm a changed configuration. If neither
choice changes, this adds validation robustness evidence to the existing
pilot; it adds no new test-seed pairs. In either case, stop this bounded check
and analyze it before another study. A new boundary choice is not a reason
for an automatic second extension.

This remains adaptation to one development split after a pilot. It does not
establish global optimality, equal tuning over all historical experiments or
new predictive superiority. The final choice is limited to the incumbent
and one screen-promoted new finalist per family, rather than a full 1800-round
search over all six new points.

## Run locally and commit the results

Use the **same environment, GPU, thread count and evaluation batch size** as
the completed parent campaign. The defaults match its four threads and
256-image evaluation batches. Preflight checks the recorded runtime and
inherited source/data/selection fingerprints before the research run.

From Anaconda Prompt:

```bat
cd /d C:\Users\WeberJ\2026_NeuromorphicFL
conda activate neuromorphicfl-resnet14
git pull --ff-only origin main
python experiments\resnet14_boundary.py verify --device cuda
python experiments\resnet14_boundary.py preflight --device cuda
python experiments\resnet14_boundary.py run --device cuda
git add experiments/results/resnet14_boundary_v1
git commit -m "Add bounded ResNet14 validation boundary check"
git push origin main
```

`verify` runs the boundary and inherited ResNet regression tests, then a short
synthetic ResNet check of the smaller quantum and threshold/gain settings,
native codec fidelity, and exact uninterrupted/resumed model/state/RNG/metrics
and traffic. It downloads no CIFAR data and produces no paper evidence.
`run` advances all stages and performs analysis automatically. One device
worker runs sequentially.

After an interruption, repeat the same `run --device cuda` command and the
three result-commit commands above. Complete stages are verified and skipped;
unfinished trajectories resume from atomic local NPZ checkpoints. Keep these
checkpoints locally until the check finishes. CSVs and JSONs are committed;
checkpoints, locks and an optional `resnet14_boundary.log` are ignored.

Progress can be inspected separately:

```bat
python experiments\resnet14_boundary.py status
```

To rebuild the boundary summaries from the completed saved evidence, without
training or loading image arrays:

```bat
python experiments\resnet14_boundary.py analyze
git add experiments/results/resnet14_boundary_v1
git commit -m "Refresh ResNet14 boundary validation analysis"
git push origin main
```
