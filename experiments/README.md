# Experiments

All experiments should import reusable code from `src/neuromorphicfl` where appropriate and write generated artifacts to `experiments/results/<experiment-name>/`.

## 01 — stationary stochastic quadratic

**Question.** Does temporal gradient evidence accumulation improve signed-update reliability and reduce communication while retaining SGD-like steady-state accuracy?

**Core setup.** `F(w)=0.5 w^2`, noisy stochastic gradients, fixed-amplitude update events. Compare SGD, signSGD, IF-SGD, and LIF-SGD. Characterize event rate, wrong-sign probability, communication reduction, tail MSE, and a leak sweep.

```bash
PYTHONPATH=src python experiments/01_stochastic_quadratic.py
```

## 02 — suddenly moving optimum

**Question.** How do IF/LIF communication states behave when the objective changes while residual gradient evidence is stored?

**Core setup.** `F_k(w)=0.5(w-theta_k)^2`, with an abrupt large change `theta: 0 -> 2`. Compare IF-SGD, LIF-SGD, periodic baselines, and an oracle membrane reset. Primary error plots are MAE/RMSE; the signed ensemble mean is only a bias diagnostic because it can hide signSGD oscillations.

**Current conclusion.** The large objective shift overwhelms the residual membrane quickly. Oracle resetting gives only a small benefit; leak mainly trades communication for responsiveness.

```bash
PYTHONPATH=src python experiments/02_moving_optimum.py
```

## 03 — moderate optimum shift after stationary convergence

**Question.** Does leakage reveal a useful stale-evidence advantage when the new gradient is much smaller than in Experiment 02?

**Core setup.** Use the same stochastic quadratic and parameters, but first allow the methods to settle near `theta=0` until `k=2000`, then shift only to `theta=0.5`. The long pre-switch phase is essential: switching earlier confounds the result because strongly leaky methods have not yet reached the old optimum and can accidentally start closer to the new target.

**Current conclusion.** Once the change-point state is controlled, leakage does not improve tracking. It reduces communication but slows adaptation; an oracle membrane reset changes IF only marginally.

```bash
PYTHONPATH=src python experiments/03_moderate_optimum_shift.py
```

## 04 — controlled stale membrane

**Question.** What does leak actually do to a known amount of stale evidence when the gradient direction reverses?

**Core setup.** Remove optimization-trajectory confounders. Initialize a membrane at a prescribed stale value and then apply a constant new gradient requiring the opposite event direction. Compare deterministic and stochastic first-passage behavior for IF and LIF.

**Current conclusion.** Leak erases stale sign sooner, but does not make a useful threshold event occur sooner. Strong leak can prevent deterministic firing entirely through the LIF deadzone.

```bash
PYTHONPATH=src python experiments/04_controlled_stale_membrane.py
```

## 05 — controlled asynchronous model drift

**Question.** Does the same stale-evidence mechanism appear when the objective stays fixed but the global model moves because of remote-client activity?

**Core setup.** A single diagnostic client has fixed local objective `F_A(w)=0.5*(w-0.25)^2`. It has stored positive membrane evidence accumulated near `w=0`. Other clients are abstracted as an instantaneous server-model drift to `w=0.5`, which reverses the diagnostic client's local gradient while leaving its loss unchanged.

**Current conclusion.** Mild leak erases the obsolete membrane sign earlier, but the first useful event is not accelerated; stronger leak again creates a responsiveness/deadzone cost.

```bash
PYTHONPATH=src python experiments/05_controlled_async_drift.py
```

## 06 — endogenous two-client asynchronous learning

**Question.** Can finite LIF memory provide a net systems benefit when stored evidence is aged while another client moves the server model?

**Core setup.** Two clients optimize the same scalar quadratic `F_i(w)=0.5 w^2`. The fast client evaluates every wall-clock tick and the slow client every five ticks. Server events are immediate and LIF membranes decay with wall-clock time.

**Initial diagnostic result.** A mild finite-memory setting improved tail RMSE and reduced communication relative to IF at one selected noisy operating point. This initially suggested a possible soft stale-evidence invalidation effect. Experiment 07 shows that this interpretation is too strong: the advantage persists even when asynchrony is removed and disappears in the zero-noise control. It should therefore currently be interpreted mainly as stochastic-gradient filtering / communication regularization, not as an established FL-specific freshness benefit.

```bash
PYTHONPATH=src python experiments/06_two_client_async.py
```

## 07 — asynchrony × memory regime map

**Question.** Is the finite-memory advantage from Experiment 06 a robust consequence of client asynchrony, or a noise-regularization effect that happens to appear in the asynchronous example?

**Core setup.** Keep homogeneous local objectives `F_1(w)=F_2(w)=0.5 w^2` and vary the slow/fast compute-period ratio

`R in {1, 2, 5, 10, 20}`

against

`rho in {1, 0.999, 0.995, 0.99, 0.98, 0.95}`.

The default stochastic run uses `sigma=0.5`, 4000 wall-clock ticks, and 300 Monte Carlo seeds. For fairness, every `rho` at a fixed `R` receives the same stochastic-gradient noise realizations. An event is classified as harmful only if the fixed jump actually increases the current quadratic objective, not by a sign-only proxy.

The experiment additionally includes:

- IF, hard-reset IF, and instantaneous-sign baselines;
- a `local_step` leakage diagnostic to check whether the result is merely caused by wall-clock decay silencing slow clients;
- a zero-gradient-noise control to separate temporal noise filtering from asynchronous stale-evidence effects.

**Current conclusion.** In the noisy homogeneous problem, `rho≈0.99` gives the lowest tail RMSE at every tested `R`, including `R=1`. The same optimum remains when leakage is applied only on local gradient evaluations. In the zero-noise control, however, IF (`rho=1`) is optimal for every `R`, reaches the exact optimum, and produces no harmful events. Therefore the present regime map does **not** support the hypothesis that the optimal memory horizon shortens as asynchrony increases. The observed finite-memory benefit is primarily stochastic evidence filtering / event regularization. At large `R`, finite wall-clock memory can also silence the slow client entirely, which is a failure mode rather than a benefit.

This is a deliberate falsification result. A genuine FL-specific stale-sign test now requires heterogeneous local objectives so that remote server movement can reverse a client's current local gradient. Such an experiment must also correct for unequal compute rates so that fast clients do not implicitly receive larger optimization weight.

Full run:

```bash
PYTHONPATH=src python experiments/07_asynchrony_memory_map.py
```

Smoke test:

```bash
PYTHONPATH=src python experiments/07_asynchrony_memory_map.py --quick
```

## Conventions

- Use fixed random seeds and Monte Carlo ensembles.
- Use common random numbers across parameter sweeps whenever possible.
- Report both optimization quality and communication cost.
- Do not use signed ensemble means as the primary convergence metric under stochastic updates.
- Preserve threshold `Delta` and parameter jump `q` as separate quantities.
- Prefer subtractive membrane reset so discrete threshold overshoot is not discarded.
- Define harmful events through actual objective change whenever the objective is available.
- When studying nonstationarity, ensure methods have comparable states at the change point or explicitly control the state.
- In asynchronous experiments, state explicitly whether leakage acts per wall-clock interval or per local gradient evaluation.
- Always check whether reduced communication is caused by useful compression or by effectively silencing slow clients.
