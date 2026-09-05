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
server quantum 0.005, scale 100, and decay exponent 0.1. The original three
held-out partition/training-seed pairs are retained and seven new pairs are
added before inspecting their results. Partition seeds are 2500, 2600, ...,
3400, and each paired training seed is defined as 70000 plus the partition
seed. No result-dependent tuning is allowed.

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
- The ten paired seeds are used to assess the stability and direction of the
  factor contrasts. The experiment still does not claim broad robustness
  outside the frozen Fashion-MNIST operating point.

## Reproduction

The workflow `.github/workflows/p11_alignment_factorial.yml` is manual-only.
Its final aggregate job validates the complete 40-point design and exports all
per-round, per-seed, aggregate, and paired-effect artifacts.

## Result

P11 and its seed extension are complete. The campaign was executed with Python
3.11 and the frozen NumPy 2.4.6, pandas 3.0.5, and Matplotlib 3.10.8
environment. All 40 runs and all 1,240 independently replayed snapshots passed
the alignment identity,
four-term decomposition, server-update reconstruction, and defect-bound
checks.

| Partition | E | weighted alignment | positive alignment | objective descent | Test accuracy | Total unicast |
|---|---:|---:|---:|---:|---:|---:|
| IID | 1 | 10.36 ± 0.23 | 96.1 ± 1.4% | 96.1 ± 1.4% | 71.43 ± 0.43% | 20.7 ± 0.2 Mbit |
| IID | 5 | 45.31 ± 1.66 | 99.0 ± 2.2% | 94.8 ± 4.1% | 83.33 ± 0.24% | 54.5 ± 0.7 Mbit |
| Strong non-IID | 1 | 20.14 ± 0.87 | 97.7 ± 1.6% | 87.4 ± 4.4% | 72.92 ± 1.04% | 100.6 ± 2.0 Mbit |
| Strong non-IID | 5 | 35.40 ± 1.52 | 99.0 ± 1.6% | 69.7 ± 9.3% | 83.22 ± 0.23% | 182.9 ± 2.4 Mbit |

Values are mean ± sample standard deviation across the ten predeclared seed
pairs. The alignment and descent percentages summarize 31 snapshots per seed.

## Analysis

The finite-trajectory alignment requirement passes in all four cells. Between
96.1% and 99.0% of audited updates have positive first-order alignment. The
weighted mean alignment ratio is positive for every individual seed, not only
after aggregation.

The heterogeneity prediction is strongly supported. Normalizing the signed
`B` term by the weighted gradient-squared denominator gives -0.25 and -5.65
under IID data for `E=1` and `E=5`. The corresponding strong-non-IID means are
-350.31 and -678.29. The paired heterogeneity effects are -350.06 at `E=1`
and -672.64 at `E=5`, with 95% paired t intervals [-363.05, -337.07] and
[-704.56, -640.71]. The effect is adverse for every paired seed at both local
depths.

Local depth materially changes `L`, but the result is an interaction rather
than a universal local-drift penalty. Under IID data, its normalized mean moves
from -22.11 at `E=1` to -59.18 at `E=5`. Under strong non-IID data, it moves
from -90.17 to +400.45. The paired local-depth effects are -37.07 under IID
data and +490.63 under strong non-IID data, with 95% paired t intervals
[-39.61, -34.53] and [473.95, 507.31]. Both directions hold for all ten seed
pairs. At the same time, the ideal local mass and `B` also change. Their large
signed cancellation explains why bounding the terms independently remains
vacuous.

Positive alignment does not guarantee an objective decrease at the finite
model quantum. Under strong non-IID data, increasing local depth raises the
mean audited event count from 3,365 to 6,458 and the mean curvature remainder
from 0.0096 to 0.0167. The objective-decrease fraction falls from 87.4% to
69.7%. The paired decrease is 17.74 percentage points, with a 95% paired t
interval of [12.39, 23.10] points, and occurs for all ten seed pairs. In the
strong-non-IID, `E=5` cell, 91 of 310 snapshots have positive alignment but no
descent. This is consistent with the second-order term in the one-step
smoothness inequality.

No emitted event opposes the current local-update proxy in any of the 1,240
snapshots, so `R=0` throughout this factorial. The observed limitation is not
stale-state sign reversal. It is the coupled effect of heterogeneity, local
depth, event energy, and curvature.

## Decision

The exact decomposition and compact factorial belong in the conference paper.
The ten-seed extension confirms that the initial three-seed mechanism result
was not driven by a favorable seed subset. It strengthens the theory section by
making the conditional alignment interface measurable, while the negative
descent result prevents an inflated convergence claim. No further expansion of
this factorial is required for the conference submission.
