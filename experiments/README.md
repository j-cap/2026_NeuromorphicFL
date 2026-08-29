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

**Current conclusion.** Once the change-point state is controlled, leakage does not improve tracking. It reduces communication but slows adaptation; an oracle membrane reset changes IF only marginally.

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

**Current conclusion.** Leak erases stale sign sooner, but does not make a useful threshold event occur sooner. Strong leak can prevent deterministic firing entirely through the LIF deadzone.

Run:

```bash
PYTHONPATH=src python experiments/04_controlled_stale_membrane.py
```

## 05 — controlled asynchronous model drift

**Question.** Does the same stale-evidence mechanism appear when the *objective stays fixed* but the global model moves because of remote-client activity?

**Core setup.** A single diagnostic client has fixed local objective `F_A(w)=0.5*(w-0.25)^2`. It has stored positive membrane evidence accumulated near `w=0`. Other clients are abstracted as an instantaneous server-model drift to `w=0.5`, which reverses the diagnostic client's local gradient while leaving its loss unchanged. Start from a controlled stale membrane `z=0.4` and compare IF/LIF first-passage behavior.

Measure both stale-sign erasure and the delay until the first correct negative event. This is the FL-specific counterpart of Experiment 04: the gradient changes because `w` changes, not because the objective changes.

**Current conclusion.** Mild leak erases the obsolete membrane sign earlier, but the first useful event is not accelerated; stronger leak again creates a responsiveness/deadzone cost. This isolates the mechanism but does not by itself establish an optimization benefit.

Run:

```bash
PYTHONPATH=src python experiments/05_controlled_async_drift.py
```

## 06 — endogenous two-client asynchronous learning

**Question.** Can finite LIF memory provide a net systems benefit when stale evidence is created endogenously by another client's server updates?

**Core setup.** Two clients optimize the same scalar quadratic `F_i(w)=0.5 w^2`, eliminating statistical heterogeneity so stale/harmful event signs are unambiguous. The fast client evaluates a gradient every wall-clock tick; the slow client every five ticks. Server events are applied immediately. All client membranes decay every wall-clock tick for LIF, including while a client is idle. The slow client can therefore hold evidence accumulated at old server models while the fast client moves `w` underneath it.

Compare:

- IF (`rho=1`): infinite evidence memory,
- an oracle hard-reset baseline that clears every other client's membrane after each server update,
- LIF with several finite memory factors.

Measure optimization error, total communication, fraction of events that oppose the *current* true descent direction, and client-specific harmful-event fractions.

**Current diagnostic result.** With the present configuration (`slow period=5`, `fast period=1`, `sigma=0.5`, `Delta=0.5`, `q=0.05`), mild LIF memory (`rho=0.995`) reduces the harmful-event fraction from about 8.7% for IF to about 3.7%, reduces communication, and improves tail RMSE. Stronger leak removes harmful events further but degrades tracking through the already identified deadzone/responsiveness mechanism. The hard-reset oracle is also not ideal: frequent remote updates repeatedly destroy useful slow-client evidence. This suggests a candidate interpretation of mild LIF as a **soft freshness mechanism between infinite memory and hard invalidation**. The result is still a toy diagnostic and requires parameter sweeps and heterogeneous-client tests before any broader claim.

Run:

```bash
PYTHONPATH=src python experiments/06_two_client_async.py
```

## Conventions

- Use fixed random seeds and Monte Carlo ensembles.
- Report both optimization quality and communication cost.
- Do not use signed ensemble means as the primary convergence metric under stochastic updates.
- Preserve threshold `Delta` and parameter jump `q` as separate quantities.
- Prefer subtractive membrane reset so discrete threshold overshoot is not discarded.
- When studying nonstationarity, ensure methods have comparable states at the change point or explicitly control the state; otherwise apparent tracking gains can be initialization artifacts.
- In asynchronous experiments, define whether leakage acts per local gradient evaluation or per wall-clock interval. Experiments 05--06 use wall-clock leakage because the intended LIF memory represents information age, including periods when a client is idle.
