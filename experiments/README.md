# Experiments

All experiments should import reusable code from `src/neuromorphicfl` and write generated artifacts to `experiments/results/<experiment-name>/`.

## 01 — stationary stochastic quadratic

**Question.** Does temporal gradient evidence accumulation improve signed-update reliability and reduce communication while retaining SGD-like steady-state accuracy?

**Core setup.** `F(w)=0.5 w^2`, noisy stochastic gradients, fixed-amplitude update events. Compare SGD, signSGD, IF-SGD, and LIF-SGD. Characterize event rate, wrong-sign probability, communication reduction, tail MSE, and a leak sweep.

Run:

```bash
PYTHONPATH=src python experiments/01_stochastic_quadratic.py
```

## 02 — suddenly moving optimum

**Question.** How do IF/LIF communication states behave when the objective changes while residual gradient evidence is stored?

**Core setup.** `F_k(w)=0.5(w-theta_k)^2`, with an abrupt large change `theta: 0 -> 2`. Compare IF-SGD, LIF-SGD, periodic baselines, and an oracle membrane reset. Primary error plots are MAE/RMSE; the signed ensemble mean is only a bias diagnostic because it can hide signSGD oscillations.

**Current conclusion.** The large objective shift overwhelms the residual membrane quickly. Oracle resetting gives only a small benefit; leak mainly trades communication for responsiveness.

Run:

```bash
PYTHONPATH=src python experiments/02_moving_optimum.py
```

## 03 — moderate optimum shift after stationary convergence

**Question.** Does leakage reveal a useful stale-evidence advantage when the new gradient is much smaller than in Experiment 02?

**Core setup.** Use the same stochastic quadratic and parameters, but first allow the methods to settle near `theta=0` until `k=2000`, then shift only to `theta=0.5`. The long pre-switch phase is essential: switching earlier confounds the result because strongly leaky methods have not yet reached the old optimum and can accidentally start closer to the new target.

Compare IF-SGD, an oracle-reset IF baseline, and several LIF retention factors. Measure pre-switch stationarity, short-horizon post-switch MAE, recovery time, first-event delay/correctness, and communication.

**Diagnostic interpretation.** If stale membrane evidence is practically important, oracle resetting and/or mild leakage should improve post-switch adaptation after the methods begin from comparable parameter distributions. If not, leakage should continue to appear primarily as a communication/noise trade-off.

Run:

```bash
PYTHONPATH=src python experiments/03_moderate_optimum_shift.py
```

## 04 — controlled stale membrane

**Question.** What does leak actually do to a known amount of stale evidence when the gradient direction reverses?

**Core setup.** Remove optimization-trajectory confounders. Initialize a membrane at a prescribed stale negative value `z0` and then apply a constant new gradient requiring a positive event. Compare deterministic and stochastic first-passage behavior for IF and LIF.

Track two distinct quantities:

1. time until the stale membrane sign is erased (`z >= 0`), and
2. time until a useful communication event is actually emitted (`z >= +Delta`).

The distinction is crucial. Leak can accelerate stale-sign cancellation while simultaneously slowing or even preventing threshold crossing because the same leak attenuates newly accumulated evidence.

Run:

```bash
PYTHONPATH=src python experiments/04_controlled_stale_membrane.py
```

## Conventions

- Use fixed random seeds and Monte Carlo ensembles.
- Report both optimization quality and communication cost.
- Do not use signed ensemble means as the primary convergence metric under stochastic updates.
- Preserve threshold `Delta` and parameter jump `q` as separate quantities.
- Prefer subtractive membrane reset so discrete threshold overshoot is not discarded.
- When studying nonstationarity, ensure methods have comparable states at the change point or explicitly control the state; otherwise apparent tracking gains can be initialization artifacts.
