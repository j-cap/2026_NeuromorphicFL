# P1 evidence freeze

This file records the Fashion-MNIST evidence that may be used in the IJCNN
manuscript. The freeze is gate-based and has no internal calendar dates.

## Frozen provenance

- Repository branch: `theory/t1-formal-semantics`
- Audited source commit: `f998f554079079629006d59f537c504a488ab010`
- Point runner: `experiments/final_baseline_point.py`
- Aggregator: `experiments/final_baseline_aggregate.py`
- Core implementation: `src/neuromorphicfl/final_baseline_campaign.py`
- Quality-selected source artifact:
  `experiments/results/final_baseline_campaign/observed_heldout_summary.csv`
- Traffic-matched source artifact:
  `experiments/results/final_baseline_campaign/observed_traffic_matched_summary.csv`
- Paper evidence generator: `paper/ijcnn2027/build_evidence.py`
- Authoritative merged artifact:
  `paper/ijcnn2027/evidence/fmnist_master_results.csv`

The source CSVs, the final report tables, and the generated paper tables were
reconciled at the displayed precision. `build_evidence.py --check` verifies the
campaign shape, selected configurations, duplicated Event-FedAvg control rows,
communication-total ordering, and byte-for-byte currency of the generated
products.

## Frozen experimental protocol

All final comparisons use Fashion-MNIST, ten clients, synchronous full
participation, the strong non-IID partition, five local SGD steps, local
learning rate 0.1, and method-matched minibatch streams.

| Item | MLP | Compact CNN |
|---|---:|---:|
| Parameters | 25,818 | 14,538 |
| Rounds | 150 | 80 |
| Event leak `rho` | 0.999 | 0.999 |
| Event threshold | 0.025 | 0.025 |
| Initial model quantum | 0.005 | 0.01 |
| Quantum exponent | 0.1 | 0.3 |

The model quantum is `q_r = q_0 (1 + r/100)^(-a)`. Event-FedAvg is frozen and
is not retuned in the closest-baseline campaign.

Development-only tuning uses partition seed 2400 and training seed 70707.
EF-TopK is searched over `{0.005, 0.01, 0.025, 0.05}` and Strom over
`{0.00125, 0.0025, 0.005, 0.01, 0.02}`, with final training objective as the
selection criterion. The quality-selected settings are:

| Architecture | EF-TopK fraction | Strom threshold |
|---|---:|---:|
| MLP | 0.05 | 0.00125 |
| Compact CNN | 0.05 | 0.005 |

The held-out comparison uses partition seeds `2500, 2600, 2700` and training
seeds `72500, 72600, 72700`. The traffic-matched controls, selected before
held-out evaluation, use EF-TopK fraction 0.01 and Strom threshold 0.02 on both
architectures.

## Tuning-leakage audit

The tuning, quality-selected evaluation, and traffic-matched evaluation are
separate GitHub Actions workflows. The tuning workflow contains only partition
seed 2400. Both evaluation workflows contain only the three held-out partition
seeds and hard-code the previously selected configurations. No held-out metric
is used by the selection code.

## Communication-accounting audit

The implementation uses `ceil(log2(d))` address bits and one sign bit per
coordinate event. Every nonempty client packet receives a 64-bit header.
EF-TopK sends a float32 value plus an address; Sign-EF sends one sign bit per
coordinate plus a float32 scale; dense FedAvg sends a float32 delta. Initial
model synchronization is charged explicitly.

For each round, the server chooses the cheaper of exact update replay and a
dense float32 checkpoint. The logical-broadcast total is packetized uplink plus
one selected downlink representation. The conservative unicast total is
packetized uplink plus the initial checkpoint to every client and, per round,
a 32-bit request plus the selected representation for each of the ten clients.
The frozen paper headline always uses conservative unicast total traffic.

The code audit confirmed that the same header, checkpoint fallback, initial
synchronization, and request policy is applied to every method. The generated
evidence build additionally rejects any row that does not satisfy
`uplink < broadcast total < unicast total`.

## Three-seed decision

The existing three held-out seeds are retained for the frozen Fashion-MNIST
campaign. They are sufficient for the paper's bounded claim about these tested
operating points, and compute effort is better spent on the independent second
benchmark required by P3. The manuscript must report mean plus sample standard
deviation, must not claim statistical significance from three seeds, and must
retain the high-variance CNN Strom worst-class qualification. More
Fashion-MNIST seeds become necessary only if review makes a significance claim
or a stability objection submission-blocking.

## Frozen environment and reproduction

The campaign workflows use Python 3.11 with the exact packages in
`paper/ijcnn2027/requirements-evidence.txt`:

```text
numpy==2.4.6
pandas==3.0.5
```

The minimum local evidence build is:

```bash
python -m pip install -r paper/ijcnn2027/requirements-evidence.txt
python paper/ijcnn2027/build_evidence.py
python paper/ijcnn2027/build_evidence.py --check
```

To reproduce one held-out point:

```bash
PYTHONPATH=src python experiments/final_baseline_point.py \
  --architecture mlp --method event \
  --partition-seed 2500 --train-seed 72500 --tag eval
```

The complete matrices are defined by:

- `.github/workflows/final_baseline_tuning.yml`
- `.github/workflows/final_baseline_evaluation.yml`
- `.github/workflows/final_baseline_traffic_match.yml`

After all matrix artifacts are in
`experiments/results/final_baseline_campaign/`, run:

```bash
python experiments/final_baseline_aggregate.py --tag eval
python experiments/final_baseline_aggregate.py --tag traffic
python paper/ijcnn2027/build_evidence.py
python paper/ijcnn2027/build_evidence.py --check
```

## Permitted paper claims from this freeze

- On the MLP, Event-FedAvg has the best mean predictive metrics among the
  quality-selected methods while using substantially less total traffic.
- On the CNN, quality-selected Strom has slightly better mean CE and accuracy
  but uses about 4.94 times the total traffic; this is a Pareto trade-off, not
  uniform Event-FedAvg dominance.
- At the frozen traffic-matched points, Event-FedAvg has better mean CE,
  accuracy, and worst-class accuracy than Strom and EF-TopK on both tested
  architectures.

These statements do not imply statistical significance, generality beyond the
tested setting, or measured hardware-energy savings.
