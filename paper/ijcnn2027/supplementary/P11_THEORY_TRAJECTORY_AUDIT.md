# Supplementary analysis: P11 theory-to-trajectory audit

This report supports the conditional-theory discussion in the Event-FedAvg
manuscript. It is a retrospective analysis of **31 independently reconstructed
audit checkpoints per seed**, for 10 matched seeds and four Fashion-MNIST MLP
conditions (IID / strong non-IID crossed with 1 / 5 local steps).

**Reproducible source and full outputs:**
- [Audit code](https://github.com/j-cap/2026_NeuromorphicFL/blob/main/experiments/p11_theory_trajectory_audit.py)
- [Three-panel vector figure](https://github.com/j-cap/2026_NeuromorphicFL/blob/main/experiments/results/p11_alignment_factorial/theory_trajectory/theory_trajectory.pdf)
- [Three-panel PNG](https://github.com/j-cap/2026_NeuromorphicFL/blob/main/experiments/results/p11_alignment_factorial/theory_trajectory/theory_trajectory.png)
- [Full generated report](https://github.com/j-cap/2026_NeuromorphicFL/blob/main/experiments/results/p11_alignment_factorial/theory_trajectory/audit_report.md)
- [Paired seed contrasts](https://github.com/j-cap/2026_NeuromorphicFL/blob/main/experiments/results/p11_alignment_factorial/theory_trajectory/paired_contrasts.csv)
- [Checkpoint diagnostics](https://github.com/j-cap/2026_NeuromorphicFL/blob/main/experiments/results/p11_alignment_factorial/theory_trajectory/checkpoint_diagnostics.csv)

## Interpretation and measurements

For each audited round the recorded reference-objective change obeys the
algebraic decomposition

\[
\Delta F_r = -q_r A_r + \mathcal R_r,\qquad
\mathcal R_r := \Delta F_r + q_r A_r .
\]

The event-scale diagnostic is \(M q_r^2N_r\), the quantity multiplying
\(L/2\) in the smoothness upper bound. The empirical reference defect uses
\(\kappa_0=8\), selected from exploratory runs. Neither \(\kappa_0\)
nor the observed curvature remainder estimates the theorem's conditional
alignment constant or a global smoothness constant.

| Regime | Local steps | Positive alignment (%) | Objective decrease (%) | Aligned without descent (%) | Mean remainder | Sampled event scale |
|---|---:|---:|---:|---:|---:|---:|
| IID | 1 | 96.13 | 96.13 | 0.00 | 0.000413 | 20.01 |
| IID | 5 | 99.03 | 94.84 | 4.19 | 0.007144 | 83.70 |
| Strong non-IID | 1 | 97.74 | 87.42 | 10.32 | 0.009600 | 158.07 |
| Strong non-IID | 5 | 99.03 | 69.68 | 29.35 | 0.016660 | 306.56 |

Values are condition-level averages over ten seeds. **Aligned without
descent** is the fraction of *all audited checkpoints*, not the conditional
fraction among positively aligned checkpoints. The event-scale column is
\(\sum_{r\in\mathcal A} Mq_r^2N_r/\sum_{r\in\mathcal A}q_r\), where
\(\mathcal A\) contains the audited checkpoints.

The strong-non-IID minus IID paired contrast in aligned non-descent
frequency is +10.32 percentage points (95% Student-t interval:
[7.10, 13.55]) for one local step and +25.16 points ([18.84, 31.49])
for five local steps. Raising the local steps from 1 to 5 in the strong
non-IID setting increases it by +19.03 points ([13.33, 24.73]).

## Limitations

The 31 checkpoints per seed are *not* the complete 150-round trajectories
for purposes of full-round theorem sums; they are selected audit rounds.
The reference objective is evaluated on the audit reference data and should
not automatically be identified with the population objective \(F\).
The correlations and four-condition ordering are descriptive, not an
identified physical or smoothness relationship. No conditional expectation,
global \(L\), or convergence certificate is estimated or proved by this audit.
The paired confidence intervals use ten seeds as independent units rather
than treating 1,240 checkpoint observations as independent.
