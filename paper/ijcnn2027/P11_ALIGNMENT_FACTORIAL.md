# P11 alignment-factorial contract

## Question

Does the aggregate-alignment interface behave as predicted when data
heterogeneity and local-update depth are changed independently?

## Frozen design

P11 changes exactly two factors from the frozen Fashion-MNIST MLP
Event-FedAvg setting:

| Factor | Levels |
|---|---|
| Partition regime | IID, strong non-IID |
| Local SGD steps | `E=1`, `E=5` |

Everything else remains fixed: ten clients, full participation, 150 rounds,
local learning rate 0.1, batch size 32, `rho=0.999`, threshold 0.025, initial
server quantum 0.005, scale 100, and decay exponent 0.1. The three existing
held-out partition/training-seed pairs are used: 2500/72500, 2600/72600, and
2700/72700. No result-dependent tuning is allowed.

The primary theory-interface quantities are the trajectory alignment ratio,
positive-alignment fraction, objective-descent fraction, and the exact
decomposition

`A_r = P_r - R_r + L_r + B_r`,

where `P_r` is current local-update alignment, `R_r` is memory opposition,
`L_r` is local drift and stochastic-gradient mismatch, and `B_r` is client
heterogeneity. Final accuracy and conservative bidirectional traffic remain
secondary operating-point checks.

## Audit protocol

Rounds 1, 5, 10, ..., 150 are audited. Each snapshot is reconstructed by an
independent replay from initialization. The event decision is completed before
the exact full-client empirical gradients are evaluated on all 6000 examples
per client. This prevents the additional diagnostic computation from changing
later threshold decisions.

The implementation must verify the aggregate-alignment identity, the exact
four-term decomposition, the server-update reconstruction, and the nonnegative
defect bound at every snapshot.

## Predeclared interpretation

- Evidence for the theory bridge requires positive aggregate alignment in all
  four cells on the finite audited trajectories.
- The heterogeneity interpretation is supported if moving from IID to strong
  non-IID makes `B_r` more adverse under paired seeds.
- The local-depth interpretation is supported if moving from `E=1` to `E=5`
  materially changes `L_r` under paired seeds.
- The experiment does not prove the conditional expectation assumption or
  establish asymptotic convergence.
- With three seeds, effect estimates remain descriptive. Statistical claims
  are deferred to the planned seed extension.

## Reproduction

The workflow `.github/workflows/p11_alignment_factorial.yml` is manual-only.
Its final aggregate job validates the complete 12-point design and exports all
per-round, per-seed, aggregate, and paired-effect artifacts.

## Result

P11 is complete. The campaign was executed with Python 3.11 and the frozen
NumPy 2.4.6, pandas 3.0.5, and Matplotlib 3.10.8 environment. All 12 runs and
all 372 independently replayed snapshots passed the alignment identity,
four-term decomposition, server-update reconstruction, and defect-bound
checks.

| Partition | E | weighted alignment | positive alignment | objective descent | Test accuracy | Total unicast |
|---|---:|---:|---:|---:|---:|---:|
| IID | 1 | 10.12 ± 0.02 | 95.7 ± 1.9% | 95.7 ± 1.9% | 71.52 ± 0.40% | 20.8 ± 0.2 Mbit |
| IID | 5 | 45.78 ± 1.28 | 100.0 ± 0.0% | 96.8 ± 3.2% | 83.17 ± 0.06% | 54.2 ± 0.5 Mbit |
| Strong non-IID | 1 | 20.26 ± 1.51 | 97.8 ± 1.9% | 86.0 ± 4.9% | 72.58 ± 1.21% | 99.8 ± 2.9 Mbit |
| Strong non-IID | 5 | 35.76 ± 0.76 | 98.9 ± 1.9% | 67.7 ± 6.5% | 83.22 ± 0.08% | 183.6 ± 1.9 Mbit |

Values are mean ± sample standard deviation across the three predeclared seed
pairs. The alignment and descent percentages summarize 31 snapshots per seed.

## Analysis

The finite-trajectory alignment requirement passes in all four cells. Between
95.7% and 100.0% of audited updates have positive first-order alignment. The
weighted mean alignment ratio is positive for every individual seed, not only
after aggregation.

The heterogeneity prediction is strongly supported. Normalizing the signed
`B` term by the weighted gradient-squared denominator gives -0.27 and -5.37
under IID data for `E=1` and `E=5`. The corresponding strong-non-IID means are
-360.98 and -700.62. The move to strong non-IID makes `B` more adverse for
every paired seed at both local depths.

Local depth materially changes `L`, but the result is an interaction rather
than a universal local-drift penalty. Under IID data, its normalized mean moves
from -21.74 at `E=1` to -57.80 at `E=5`. Under strong non-IID data, it moves
from -95.23 to +411.36. At the same time, the ideal local mass and `B` also
change. Their large signed cancellation explains why bounding the terms
independently remains vacuous.

Positive alignment does not guarantee an objective decrease at the finite
model quantum. Under strong non-IID data, increasing local depth raises the
mean audited event count from 3,393 to 6,496 and the mean curvature remainder
from 0.0100 to 0.0166. The objective-decrease fraction falls from 86.0% to
67.7%. In the strong-non-IID, `E=5` cell, 29 of 93 snapshots have positive
alignment but no descent. This is consistent with the second-order term in the
one-step smoothness inequality.

No emitted event opposes the current local-update proxy in any of the 372
snapshots, so `R=0` throughout this factorial. The observed limitation is not
stale-state sign reversal. It is the coupled effect of heterogeneity, local
depth, event energy, and curvature.

## Decision

The exact decomposition and compact factorial belong in the conference paper.
They strengthen the theory section by making the conditional alignment
interface measurable, while the negative descent result prevents an inflated
convergence claim. The next experiment is the already planned seed extension
of the main communication--performance comparison, not another expansion of
this factorial.
