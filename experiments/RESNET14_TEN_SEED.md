# ResNet-14: ten matched Event/Ström pairs

Repository: **j-cap/2026_NeuromorphicFL**, the research-code repository.
Entry point: `experiments/resnet14_ten_seed.py`.
Results: `experiments/results/resnet14_ten_seed_v1/`.

## Fixed question and scope

Does the selected Event/Ström accuracy and communication contrast persist
across ten matched ResNet-14 partitions/SGD seeds? Reuse the six completed
full-training outcomes on the three pilot pairs and add **fourteen new runs
on seven fresh pairs**. All new runs use 1800 communication rounds.

The [`bounded validation boundary check`](RESNET14_BOUNDARY_RESULTS.md) leaves
both original configurations selected. This confirmation adds no tuning or
new method family. Configurations and validation evidence are verified before
the existing pilot test results are read as reuse references. No test result
selects a setting, replaces a seed or stops a trajectory early. All fourteen
new trajectories are attempted even if Event loses the pilot comparison.

| Method | Frozen configuration |
|---|---|
| Event | Threshold 0.01; initial quantum 0.0025; quantum exponent 0.2 and scale 100; leak 0.999; persistent state; full reset; server gain 1 |
| Ström | Threshold 0.005; equivalent gain 2 / native gain 0.10; nonleaky subtractive residual; one pulse per coordinate/client/round |

Keep Torch ResNet-14 with GroupNorm (175258 parameters), ten fully participating
clients, 55%-dominant label skew, initialization 7777, five local SGD steps,
batch size 32, learning rate 0.05 and L2 coefficient 0.0005 unchanged. Each run
uses all 50000 CIFAR-10 training images, 5000 images/client, a 5000-image
training diagnostic subset and the fixed official 10000-image test set.
Evaluation is saved at round 1 and every 60 rounds, including round 1800.

## Seeds, reuse and workload

| Pair | Partition seed | Shared SGD seed | Evidence |
|---:|---:|---:|---|
| 0 | 7100 | 97100 | Verified pilot reuse |
| 1 | 7200 | 97200 | Verified pilot reuse |
| 2 | 7300 | 97300 | Verified pilot reuse |
| 3 | 7400 | 97400 | Two new runs |
| 4 | 7500 | 97500 | Two new runs |
| 5 | 7600 | 97600 | Two new runs |
| 6 | 7700 | 97700 | Two new runs |
| 7 | 7800 | 97800 | Two new runs |
| 8 | 7900 | 97900 | Two new runs |
| 9 | 8000 | 98000 | Two new runs |

The seven new partition seeds do not overlap the recorded earlier grid through
7300 at evidence snapshot `bd729848a9fcb89edd4e40f71ed09c7eb461372c`.
Both methods share partition, training diagnostic indices, initialization and
minibatch RNG within each pair. Each new run starts from initialization;
development or pilot model checkpoints cannot initialize it.

New work is **25200 communication rounds and 434 evaluation points**. Including
reuse, the final comparison verifies twenty outcomes, **36000 round records
and 620 sampled model fingerprints**. One evaluation point covers both the
training diagnostic subset and the official test set. These counts do not
measure epochs or elapsed GPU time.

The six reused outcomes retain their original source-contract identity. Six
small `*_reuse.json` receipts pin their files and source hashes; original
pilot CSVs, receipts and partition manifests remain in
`experiments/results/resnet14_transfer_v1/`. Their NPZ checkpoints are not
needed and their trajectories are not retrained. Combined analysis marks
`reused_pilot` versus `new_additional`, retains original `contract_sha256`, and
adds `source_contract_sha256` and `analysis_contract_sha256`.

New provenance paths use repository-relative POSIX spelling. The boundary
source's Windows dependency-key spelling is checked through the separate
portable reader, preserving its original stored digests. The parent protocol,
unchanged boundary selections, all referenced evidence, scientific sources,
runtime, raw-data fingerprints and exact full-training partition manifests
are pinned. Changes to those identities block resume rather than silently
mixing different experiments.

## Accounting and automatic analysis

Use the frozen native Torch pulse rules and actual shared lossless uplink and
exact aggregate downlink codecs. Primary traffic is conservative unicast:
complete frames, padding, initial synchronization, per-client requests and
recipient replication. Upload and download are also reported separately;
logical broadcast is supplementary. Codec audit timings are not physical
network latency, energy or measured deployment speed.

Final analysis requires all **six reused and fourteen new outcomes**. Missing
or numerically failed runs remain in `confirmation_completion.json` and block
the final comparison. Numerical failures produce terminal receipts, the
remaining new runs continue, and no replacement seed or setting is invented.
Structural errors preserve completed work and propagate for correction.

The driver automatically writes:

- `confirmation_runs.csv` and `confirmation_history.csv`: all twenty outcomes,
  with source identities, predictive metrics, regularization diagnostics and
  communication variants.
- `paired_per_seed.csv` and `paired_summary.csv`: Event-minus-Ström final
  accuracy in percentage points and CE, plus Event/Ström upload, download,
  total unicast and logical-broadcast ratios.
- `traffic_to_target.csv` and `paired_target_summary.csv`: first saved crossings
  of the unchanged **45%, 50% and 55% accuracy targets**, with censoring and
  achievement counts for every method and scope. Ratios use only common hits;
  an unreached target is not assigned zero or free communication.
- `confirmation.png` / `.pdf`: descriptive mean accuracy versus round and
  unicast traffic, and mean test CE. The accuracy band is sample SD.
- `analysis_audit.json`: total coverage, reuse counts and selection provenance.

Summary tables distinguish **`all_10`** from **`additional_7`**. The latter
contains only the seven new prospective pairs, excluding the already
inspected three-pair pilot. Report all ten pairs and use the seven-pair
analysis to assess the prospective confirmation separately.

Intervals are descriptive, paired, unadjusted 95% Student-t intervals; traffic
ratios use the log scale. Uncertainty covers partition/SGD variation under
one fixed initialization, configuration pair and previously inspected CIFAR-10
test set. This adds no new test dataset, no equivalence/noninferiority result,
and no comparison to Dense/STC/EF-TopK under the new ResNet protocol.

## Run overnight

Use the same `neuromorphicfl-resnet14` environment and GPU as the completed
pilot. Defaults match its four threads and evaluation batch size 256. One
process uses the GPU sequentially. There is no `--workers` option.

From Anaconda Prompt:

```bat
cd /d C:\Users\WeberJ\2026_NeuromorphicFL
conda activate neuromorphicfl-resnet14
git pull --ff-only origin main
python experiments\resnet14_ten_seed.py verify --device cuda
python experiments\resnet14_ten_seed.py preflight --device cuda
python -u experiments\resnet14_ten_seed.py run --device cuda >> resnet14_ten_seed.log 2>&1
```

The run appends progress to `resnet14_ten_seed.log` and performs final analysis
automatically. After it completes, commit and push the results:

```bat
git add experiments/results/resnet14_ten_seed_v1
git commit -m "Add ResNet14 ten-seed Event Strom confirmation"
git push origin main
```

`verify` runs the boundary, inherited ResNet and ten-seed regression tests,
then actual synthetic ResNet checks for both selected configurations: native
codec fidelity and exact uninterrupted/resumed model, pulse state, RNG,
metrics and traffic. Verification loads no CIFAR image arrays and produces
no paper evidence. Preflight validates source/reuse/selection identity and
the recorded runtime before the training run.

In a separate prompt, inspect progress without launching another GPU worker:

```bat
python experiments\resnet14_ten_seed.py status
```

After an interruption, repeat the same `run` command, then the result-commit
commands above. Complete stages are verified and skipped; partial trajectories
resume from their last atomic NPZ checkpoint, saved every 60 rounds. Keep
local checkpoints until completion. If all fourteen new outcomes already
exist, repeating `run` rebuilds the combined analysis without loading image
arrays. Checkpoints, locks and the console log stay out of Git.

To rebuild final analysis from completed saved evidence without training:

```bat
python experiments\resnet14_ten_seed.py analyze
git add experiments/results/resnet14_ten_seed_v1
git commit -m "Refresh ResNet14 ten-seed confirmation analysis"
git push origin main
```
