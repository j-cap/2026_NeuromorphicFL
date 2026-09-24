# P15 compact Event-FedAvg operator ablation

## Question

At matched bidirectional traffic, how much do persistent cross-round state and
full reset contribute to the selected compact CIFAR-10 CNN operating point?

## Scope

This is a focused mechanism ablation, not a new architecture benchmark. The
four-model P12--P14 campaign evaluates the complete operator across models.
P15 uses only the compact CIFAR-10 CNN because it is the established operating
point for the encoder audit and is inexpensive enough for ten paired seeds.

## Unique variants

1. **Frozen reference:** persistent state with full reset.
2. **Memoryless:** state is cleared before adding each round's evidence.
3. **Subtractive reset:** persistent state subtracts one signed threshold after
   firing and therefore retains overshoot.

A memoryless/subtractive combination is redundant under one threshold test per
round because its post-fire state is discarded before the next round. The
coupled threshold/quantum control belongs to the existing schedule audit and is
not repeated here.

## Fixed training protocol

- Dataset/model: strong-skew CIFAR-10, compact CNN.
- Clients: 10 with full participation.
- Local training: 5 steps, batch size 32, learning rate 0.05.
- Horizon: 120 rounds.
- Persistent-state retention: `rho = 0.999`.
- Server quantum: `q_r = 0.005 (1 + r/100)^(-0.2)` for every variant.
- Memoryless threshold candidates: `0.0025`, `0.005`, `0.01`, and `0.025`.
- Subtractive-reset threshold candidates: `0.0125`, `0.025`, `0.05`, and
  `0.1`. The larger range anticipates additional firing when overshoot is
  retained.
- Communication: the manuscript's conservative bidirectional accounting.

Only the encoder memory/reset rule and the development-selected threshold may
change.

## Selection and held-out evaluation

On development partition 2400, select one threshold per ablated family by
minimum absolute log-traffic distance to the frozen Event-FedAvg development
run. Final training cross-entropy breaks an exact traffic-distance tie. Freeze
the selected thresholds before held-out evaluation.

Evaluate the two selected variants on CIFAR-10 partitions 3500, 3600, ..., 4400
with paired training seeds 83500, 83600, ..., 84400. Reuse the matching ten
frozen Event-FedAvg runs from P12 rather than retraining them.

## Reported quantities

- test accuracy,
- worst-class accuracy,
- coordinate-event count,
- conservative unicast traffic,
- logical-broadcast traffic,
- paired accuracy difference and 95% confidence interval versus the frozen
  reference,
- paired traffic ratio and number of seed-pair wins.

The result supports an isolated mechanism claim only when a variant differs at
approximately matched traffic. A null or reversed result is retained and the
manuscript continues to attribute performance to the complete operator.
