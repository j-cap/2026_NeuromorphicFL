# P3 CIFAR-10 gate decision

> **P8 amendment.** P3 froze the original benchmark campaign. P8 subsequently
> tuned dense FedAvg's server gain on the development partition and selected
> gain 2. The held-out dense result is 44.13 +/- 1.12%, so the frozen P3
> Event-FedAvg result remains 3.93 points higher while using 12.7% of its
> traffic. The original gain-1 row and comparison below are retained as the P3
> historical record; the manuscript and generated evidence use the P8-tuned
> dense row. See `P8_REVISION_AUDIT.md`.

## Decision

**Pass:** Event-FedAvg remains on the held-out communication--performance
frontier for CIFAR-10 with the frozen compact CNN and strong non-IID partition.

The decision was made after the complete GitHub Actions campaign finished. The
development partition selected `event_t025_q005`; the three held-out partitions
were not used for tuning.

## Held-out evidence

| Comparison | Method | Accuracy [%] | Worst class [%] | Total unicast [Mbit] |
|---|---|---:|---:|---:|
| Quality | Event-FedAvg | 48.06 +/- 1.12 | 24.6 +/- 4.6 | 201.5 +/- 4.3 |
| Quality | Strom | 47.68 +/- 0.49 | 18.9 +/- 3.8 | 930.9 +/- 2.0 |
| Quality | EF-TopK | 42.19 +/- 0.37 | 26.7 +/- 0.8 | 645.2 +/- 0.0 |
| Quality | Sign-EF | 41.59 +/- 0.33 | 24.6 +/- 3.2 | 279.4 +/- 0.0 |
| Quality | Dense FedAvg | 42.42 +/- 0.24 | 26.8 +/- 1.7 | 1586.6 +/- 0.0 |
| Traffic neighbor | Strom | 35.50 +/- 4.00 | 10.3 +/- 3.8 | 221.8 +/- 14.8 |
| Traffic neighbor | EF-TopK | 40.95 +/- 0.14 | 23.6 +/- 0.5 | 135.3 +/- 0.0 |

Values are mean plus or minus sample standard deviation over partition seeds
3500, 3600, and 3700. The Event-FedAvg row is shared by the quality and traffic
comparisons.

## Interpretation boundary

Event-FedAvg has the best mean CE and accuracy and the lowest total traffic
among the quality-selected methods. Relative to dense FedAvg, it improves mean
accuracy by 5.63 percentage points while using 12.7% of the traffic. The
nearest Strom point uses 10.1% more traffic and has 12.56 percentage points
lower mean accuracy. The nearest EF-TopK point uses 32.9% less traffic and has
7.10 percentage points lower mean accuracy, so those two points describe a
trade-off rather than strict Event-FedAvg dominance.

The paper must not turn this pass into a significance claim. It must also show
that dense and quality-selected EF-TopK have slightly better mean worst-class
accuracy, and it must not infer hardware energy, latency, partial-participation,
or asynchronous-FL behavior from this campaign.
