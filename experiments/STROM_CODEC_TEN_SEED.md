# Calibrated Ström: ten matched full-training seeds

Repository: **j-cap/2026_NeuromorphicFL**, the research-code repository.
Entry point: `experiments/strom_codec_ten_seed.py`.
Results: `experiments/results/strom_codec_ten_seed_v1/`.

## Purpose and frozen choice

Confirm the validation-selected Ström-style residual-pulse setting against the
completed Event/STC comparison on the same ten full-training partitions and
SGD seeds. Keep the existing Event and STC algorithms and trajectories fixed.
The development study is interpreted in
[`STROM_CODEC_DEVELOPMENT_RESULTS.md`](STROM_CODEC_DEVELOPMENT_RESULTS.md).

Ström uses **threshold 0.05**, **FedAvg-equivalent gain 2**, and **native server
gain 0.10** with local learning rate 0.05. This setting won both development
selection roles. Its nonleaky residual, subtractive reset, one pulse per
coordinate/client/round, and coupled threshold/pulse magnitude are unchanged.
The shared lossless position codec and exact aggregate-downlink portfolio are
unchanged. No Event leak/reset/quantum schedule is added to the comparator.

The driver verifies the development study's split, receipts, frozen decisions,
and selected configuration before linking the comparison evidence. It provides
no threshold/gain tuning command or test-based early stopping.

## Ten new runs, thirty reused controls

The development run used **45000 training images and 5000 validation images**
on development partition 4500/SGD seed 84500. The existing comparison uses
**all 50000 training images and the separate 10000-image official test set**.
Therefore, the development model/checkpoint cannot serve as one confirmation
seed or be continued into the full-training experiment.

Run **all ten calibrated Ström trajectories from round 1**, including seed 3500.
The earlier small-threshold Ström pilot cannot be reused for the new setting.
Reuse **all thirty** completed Event/STC trajectories (ten Event, ten STC p=1,
ten STC p=0.25), after their sources, receipts, model fingerprints, sample
coverage and traffic histories verify. Reuse receipts point to the existing
CSV files; those methods are not retrained or copied into new raw-result CSVs.

| Seed index | Partition seed | SGD seed |
|---:|---:|---:|
| 0 | 3500 | 83500 |
| 1 | 3600 | 83600 |
| 2 | 3700 | 83700 |
| 3 | 3800 | 83800 |
| 4 | 3900 | 83900 |
| 5 | 4000 | 84000 |
| 6 | 4100 | 84100 |
| 7 | 4200 | 84200 |
| 8 | 4300 | 84300 |
| 9 | 4400 | 84400 |

All runs use the compact CIFAR CNN (20570 parameters), ten fully participating
clients, strong 55%-dominant label skew, initialization seed 7777, five local
SGD steps, batch size 32, local learning rate 0.05, L2 coefficient 0.0005, and
**1800 communication rounds**. Training diagnostics use 5000 training images.
Save prediction metrics at round 1 and every 60 rounds, matching the controls.
This campaign does not run the larger ResNet model named in the conda environment.

Refitting on all training images after freezing the setting is intentional.
The development validation scores remain separate from confirmation test scores.
The existing official test set and earlier pilot have already been inspected;
this is a matched confirmation across partition/SGD variation, not a new blind
test dataset. No further setting selection may use these test curves.

## Run locally

From Anaconda Prompt in the code repository:

```bat
cd /d C:\Users\WeberJ\2026_NeuromorphicFL
conda activate neuromorphicfl-resnet14
git pull --ff-only

python experiments/strom_codec_ten_seed.py verify
python experiments/strom_codec_ten_seed.py preflight
python experiments/strom_codec_ten_seed.py run --workers 2
python experiments/strom_codec_ten_seed.py analyze

git add experiments/results/strom_codec_ten_seed_v1
git commit -m "Add calibrated Strom ten-seed confirmation results"
git push origin main
```

`verify` uses synthetic images and validates the committed evidence. It performs
no CIFAR research training or download. The preflight should report
`Runtime matches recorded Windows environment: True` in the original conda
environment. Training additionally checks the raw CIFAR batch fingerprints.

The default command starts all ten missing Ström runs, with two concurrent
workers. Use `--workers 1` to reduce CPU/RAM concurrency. Optional
`--seed-index 0` limits `run` to the first full-horizon Ström seed; it does not
select or change the scientific configuration. Use the same result commit
commands after publishing any completed subset.

Check progress without training:

```bat
python experiments/strom_codec_ten_seed.py status
```

After interruption, repeat the run command to resume unfinished checkpoints:

```bat
python experiments/strom_codec_ten_seed.py run --workers 2
python experiments/strom_codec_ten_seed.py analyze
git add experiments/results/strom_codec_ten_seed_v1
git commit -m "Add completed calibrated Strom confirmation results"
git push origin main
```

Complete receipts are verified and skipped. Local NPZ files retain the model,
client replica, residual, RNG, traffic records and sampled prediction history;
continuation preserves the exact trajectory. Each checkpoint also carries the
configuration and campaign provenance. Source, runtime, raw-data, development
selection or reused-evidence changes cannot be mixed into this campaign.

Numerical failures receive explicit receipts and are not automatically retried
with altered settings. Analysis requires every expected result to be complete
or a verified reuse; a failed/missing seed blocks the paired summary and remains
visible in `completion_audit.json`. Checkpoints and locks remain local and ignored
by Git; CSV/JSON scientific evidence and the two confirmation figures are tracked.

## Analysis and interpretation

The **primary endpoint** is Event minus calibrated Ström round-1800 test accuracy
in percentage points. Secondary endpoints are test CE, regularized training
diagnostics, upload, download, total unicast traffic, supplementary logical
broadcast traffic, and communication to fixed 45%, 50% and 55% accuracy targets.
Every method receives its improved lossless representation; Event/Ström use
`both_codecs`, while STC uses `lossless_stc`.

The comparison includes Event/Ström, Ström/STC p=1, Ström/STC p=0.25, and the
two existing Event/STC contrasts. Paired accuracy/CE differences use candidate
minus reference; cost ratios use candidate divided by reference. Values above
one for a cost ratio mean that the candidate costs more. No favourable direction
is enforced or used as an inclusion rule.

Report both all ten seeds and a sensitivity analysis excluding the previously
inspected pilot partition 3500. Use descriptive, unadjusted paired 95% Student-t
intervals; traffic ratios use the log scale. These intervals describe partition
and SGD variation on the fixed dataset and do not establish formal equivalence,
noninferiority or field-wide state of the art. The older STC tuning objective
and this Ström validation-CE selection policy must remain explicit in the paper.

Target costs use the **first saved crossing**, without interpolation or best
test-checkpoint selection. Keep each method's reached/not-reached result and
censor horizon. Ratios are conditional on both methods reaching a target;
always report the total seed count, each method's hit count and common-hit
count. An unreached target is not a free or zero-cost achievement. First
crossings are retrospective diagnostics, not a deployed early-stopping rule.

## Outputs

- `protocol.json`: frozen selection, sources, runtime/data, all method configs,
  reference paths/hashes and statistical rules.
- Thirty reuse receipts referencing completed Event/STC evidence.
- Ten new Ström receipts plus per-seed round, evaluation, summary and history
  CSVs, including model fingerprints and actual codec counters.
- `completion_audit.json`: all forty expected outcomes, including failures.
- `comparison.csv`, `comparison_history.csv`: all encoding variants and methods.
- `main_runs.csv`, `endpoint_summary.csv`: optimized-encoding endpoints, training
  CE/L2 separation and all-ten/excluding-pilot descriptive summaries.
- `paired_per_seed.csv`, `paired_summary.csv`: five paired contrasts with
  predeclared directions and primary endpoint flag.
- `traffic_to_target.csv`, `paired_target_summary.csv`: first saved crossings,
  censoring and paired target-cost coverage.
- `analysis_audit.json`, `confirmation.png` / `.pdf`: verification scope and
  descriptive curves/target-cost intervals.

All previous hosts, packages, experiments, calibration results and completed
comparison archives remain unchanged. This campaign adds the calibrated pulse
comparator; dense FedAvg and EF-TopK remain part of the agreed final core set.
