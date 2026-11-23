# Experiments

The bounded follow-up to the ResNet calibration boundaries is documented in
[`RESNET14_BOUNDARY.md`](RESNET14_BOUNDARY.md). It adds three validation-only
600-round candidates per method, continues one new finalist each exactly to
1800 rounds, and compares them with the verified frozen incumbents. Maximum
additional work is 6000 rounds, equally divided between Event and Ström.
Run `python experiments/resnet14_boundary.py verify --device cuda`,
`preflight --device cuda`, then `run --device cuda`. Selection uses validation
CE and the original holdout; the new study reads no official-test images or
metrics and launches no confirmation or further extension. Afterwards use
`git add experiments/results/resnet14_boundary_v1`,
`git commit -m "Add bounded ResNet14 validation boundary check"`, and
`git push origin main`. Repeating `run` resumes the pipeline.

The completed ResNet-14 transfer is interpreted in
[`RESNET14_TRANSFER_RESULTS.md`](RESNET14_TRANSFER_RESULTS.md). All thirty
screening runs, four finalists and six fresh confirmation runs are complete.
Across three matched pairs, selected Event reaches 75.797% mean test accuracy
versus 72.257% for selected Ström, with 31.04% less final unicast traffic and
53.24% less traffic to the declared 50% accuracy target. Both selections touch
lower search boundaries; these are bounded-calibration results, not optimal
tuning claims. Run `python experiments/resnet14_transfer_audit.py` to verify
all forty receipts, four exact continuations, independent traffic/statistical
reconstruction and the eight original analysis CSVs without training or
overwriting original analysis. Afterwards use
`git add experiments/results/resnet14_transfer_v1`,
`git commit -m "Refresh ResNet14 transfer evidence audit"`, and
`git push origin main`.

The automatic weekend ResNet-14 study is documented in
[`RESNET14_TRANSFER.md`](RESNET14_TRANSFER.md). It screens fifteen settings each
for Event and Ström on a training-only holdout, continues at most two finalists
per method exactly to 1800 rounds, freezes one choice each, and automatically
trains six fresh full-training confirmation runs on seeds 7100/7200/7300.
The normal maximum budget is 33600 rounds. Run
`python experiments/resnet14_transfer.py verify --device cuda`,
`preflight --device cuda`, `benchmark --device cuda`, then `run --device cuda`.
Analysis is automatic when all six confirmation runs complete. Afterwards use
`git add experiments/results/resnet14_transfer_v1`,
`git commit -m "Add bounded ResNet14 pulse transfer results"`, and
`git push origin main`. Repeating the run command resumes the complete pipeline;
official test images are parsed only after selection is frozen.

The completed five-method core is interpreted in
[`CORE_FRESH_COMPARISON_RESULTS.md`](CORE_FRESH_COMPARISON_RESULTS.md). Event has
the lowest final unicast cost among the tested settings and uses fewer bits
to reach 50% accuracy in every paired seed against all four comparators.
The closest Ström contrast retains the 3.91% final and 29.23% target-50%
traffic savings, with unresolved final accuracy/CE differences. Run
`python experiments/core_fresh_comparison_audit.py` to verify all fifty
outcomes, 90000 rounds, 1550 sampled model fingerprints, independent traffic
and statistical reconstruction, and target censoring without training or
overwriting original analysis. Afterwards use
`git add experiments/results/core_fresh_comparison_v1`,
`git commit -m "Refresh common-seed core comparison evidence audit"`, and
`git push origin main`.

The common-seed five-method core is documented in
[`CORE_FRESH_COMPARISON.md`](CORE_FRESH_COMPARISON.md). It adds thirty new
1800-round trajectories for the frozen scheduled Dense, STC p=1/gain 2 and
EF-TopK 5% baselines on seeds 6000–6900, and verifies/reuses all twenty completed
Event/Ström controls. EF-TopK receives a lossless float32 support codec and
exact aggregate/replay downlink without changing its native update/residual
rule. Run `python experiments/core_fresh_comparison.py verify`, `preflight`,
`run --workers 2`, then `analyze`. Afterwards use
`git add experiments/results/core_fresh_comparison_v1`,
`git commit -m "Add common-seed five-method core comparison results"`, and
`git push origin main`. All fifty expected outcomes are required for the final
paired accuracy/CE and conservative-unicast analysis.

The completed fresh Event/Ström confirmation is interpreted in
[`EVENT_STROM_FRESH_CONFIRMATION_RESULTS.md`](EVENT_STROM_FRESH_CONFIRMATION_RESULTS.md).
Tuned Event uses 3.91% fewer final unicast bits and 29.23% fewer bits to reach
50% test accuracy across ten new matched pairs; final accuracy and CE
differences remain unresolved. All twenty runs are complete. Run
`python experiments/event_strom_fresh_confirmation_audit.py` to verify receipts,
model fingerprints, independent traffic/statistical reconstruction and saved
analysis without training or overwriting the original outputs. Afterwards use
`git add experiments/results/event_strom_fresh_confirmation_v1`,
`git commit -m "Refresh fresh Event and Strom confirmation evidence audit"`,
and `git push origin main`.

The fresh confirmation of tuned Event against calibrated Ström is documented
in [`EVENT_STROM_FRESH_CONFIRMATION.md`](EVENT_STROM_FRESH_CONFIRMATION.md).
It freezes Event threshold 0.05 / quantum 0.01 and Ström threshold 0.05 /
native gain 0.10, then trains both methods on ten fresh matched partitions
(6000–6900): twenty new full-training runs of 1800 rounds, with no reused
trajectory or validation checkpoint. Run
`python experiments/event_strom_fresh_confirmation.py verify`, `preflight`,
`run --workers 2`, then `analyze`. Afterwards use
`git add experiments/results/event_strom_fresh_confirmation_v1`,
`git commit -m "Add fresh tuned Event and Strom confirmation results"`, and
`git push origin main`. All ten pairs enter the accuracy/CE and unicast-traffic
analysis; failed or missing runs block the final summary.

The completed Event threshold/quantum search is interpreted in
[`EVENT_CODEC_DEVELOPMENT_RESULTS.md`](EVENT_CODEC_DEVELOPMENT_RESULTS.md).
It selects threshold 0.05 / quantum 0.01, improving validation accuracy and
traffic over current Event. Relative to calibrated Ström, it has lower traffic
and higher accuracy but worse final validation CE on the single development
seed. Run `python experiments/event_codec_development_audit.py` to verify saved
evidence and regenerate endpoint, target and codec diagnostics without changing
the original analysis. Afterwards use
`git add experiments/results/event_codec_development_v1`,
`git commit -m "Refresh Event threshold quantum development analysis"`, and
`git push origin main`. The fresh paired confirmation above implements the
next experiment.

The bounded Event threshold/quantum study is documented in
[`EVENT_CODEC_DEVELOPMENT.md`](EVENT_CODEC_DEVELOPMENT.md). It screens fifteen
settings on the existing 45000/5000 training/validation split: fourteen new runs
plus reused current Event. Calibrated Ström supplies the fixed communication
budget. At most two promoted settings continue exactly from 600 to 1800 rounds;
only threshold and initial quantum change. Run
`python experiments/event_codec_development.py verify`, `preflight`,
`run --workers 2`, then `analyze`. Afterwards use
`git add experiments/results/event_codec_development_v1`,
`git commit -m "Add validation-only Event threshold quantum development results"`,
and `git push origin main`. The study loads no official test curves and ends
with frozen development choices, without launching confirmation runs.

The completed calibrated Ström comparison is interpreted in
[`STROM_CODEC_TEN_SEED_RESULTS.md`](STROM_CODEC_TEN_SEED_RESULTS.md). The ten-seed
study finds lower final traffic for Ström in every seed, lower mean test CE,
and no established final-accuracy difference from Event. Event reaches 50%
accuracy in fewer sampled rounds on every seed, but this does not establish
lower traffic to that target. Run `python experiments/strom_codec_ten_seed_audit.py`
to reproduce the saved-evidence audit and export paired cost/codec diagnostics
without changing the original analysis. Afterwards use
`git add experiments/results/strom_codec_ten_seed_v1`,
`git commit -m "Refresh calibrated Strom ten-seed evidence audit"`, and
`git push origin main`.

The calibrated Ström confirmation is documented in
[`STROM_CODEC_TEN_SEED.md`](STROM_CODEC_TEN_SEED.md). It freezes threshold 0.05
and native gain 0.10, runs ten new 1800-round Ström trajectories on the full
training set, and reuses all thirty completed Event/STC controls. The validation
checkpoint cannot be reused, including for seed 3500. Run
`python experiments/strom_codec_ten_seed.py verify`, `preflight`,
`run --workers 2`, then `analyze`. Afterwards use
`git add experiments/results/strom_codec_ten_seed_v1`,
`git commit -m "Add calibrated Strom ten-seed confirmation results"`, and
`git push origin main`. Runs resume from local checkpoints; all expected seeds
must be present before the paired analysis is accepted.

The completed Ström development study is interpreted in
[`STROM_CODEC_DEVELOPMENT_RESULTS.md`](STROM_CODEC_DEVELOPMENT_RESULTS.md).
It freezes threshold 0.05 and native gain 0.10 for confirmation. The selected
Ström run uses 31.85% less final unicast traffic than Event and has better
round-1800 validation CE on the single development seed. This is development
evidence, not a multi-seed test conclusion. Run
`python experiments/strom_codec_development_audit.py` to verify saved evidence
and regenerate endpoint, target-cost, codec and trajectory diagnostics without
training. Afterwards use `git add experiments/results/strom_codec_development_v1`,
`git commit -m "Refresh validation-only Strom development analysis"`, and
`git push origin main`.

The bounded validation-only Ström calibration is documented in
[`STROM_CODEC_DEVELOPMENT.md`](STROM_CODEC_DEVELOPMENT.md). It holds out 5000
training images, screens 15 settings plus fixed Event, permits one boundary
extension of at most six settings, and continues at most two Ström finalists
and Event to 1800 rounds. Run `python experiments/strom_codec_development.py verify`,
`preflight`, `run --workers 2`, then `analyze`. Afterwards use
`git add experiments/results/strom_codec_development_v1`,
`git commit -m "Add validation-only Strom development results"`, and
`git push origin main`.

The comparable Ström-style pulse baseline is documented in
[`STROM_CODEC_PILOT.md`](STROM_CODEC_PILOT.md). It uses the same lossless position
and aggregate-downlink codecs as Event, preserves the native residual rule, and
tests six fixed threshold/gain combinations for 600 rounds on seed 3500.
Event and both STC prefixes are reused. Run
`python experiments/strom_codec_pilot.py verify`, `preflight`, `run --workers 2`,
then `analyze`. Afterwards use `git add experiments/results/strom_codec_pilot_v1`,
`git commit -m "Add comparable Strom-style codec pilot results"`, and
`git push origin main`.

The completed pilot and its scaling/backlog qualifications are documented in
[`STROM_CODEC_PILOT_RESULTS.md`](STROM_CODEC_PILOT_RESULTS.md).
`python experiments/strom_codec_pilot_audit.py` verifies the saved evidence and
exports endpoint, target-cost and residual/codec diagnostics without training.

The ten-seed confirmation of the frozen 1800-round codec comparison is documented
in [`EVENT_CODEC_TEN_SEED.md`](EVENT_CODEC_TEN_SEED.md). It reuses the three
completed seed-3500 trajectories and trains 27 remaining point/seed pairs.
Run `python experiments/event_codec_ten_seed.py verify`, `preflight`,
`run --workers 2`, then `analyze`. Afterwards use
`git add experiments/results/event_codec_ten_seed_v1`,
`git commit -m "Add ten-seed Event and STC codec confirmation results"`, and
`git push origin main`.

The completed confirmation is interpreted in
[`EVENT_CODEC_TEN_SEED_RESULTS.md`](EVENT_CODEC_TEN_SEED_RESULTS.md).
`python experiments/event_codec_ten_seed_audit.py` verifies the committed
evidence and adds direction/target/codec summaries without overwriting the
original local analysis.

The single-seed 1800-round codec follow-up with held-out predictive metrics is
documented in [`EVENT_CODEC_FULL_HORIZON.md`](EVENT_CODEC_FULL_HORIZON.md).
Run `python experiments/event_codec_full_horizon.py verify`, `preflight`,
`run --workers 2`, then `analyze`. Afterwards use
`git add experiments/results/event_codec_full_horizon_v1`,
`git commit -m "Add full-horizon single-seed codec results"`, and `git push origin main`.

The short Event/STC message-coding audit is documented in
[`EVENT_CODEC_PILOT.md`](EVENT_CODEC_PILOT.md). Run
`python experiments/event_codec_pilot.py verify`, then `run --workers 2` and
`analyze`. It uses 120 rounds on the fixed development seed and does not alter
the frozen STC confirmation sources.

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

## 08 — heterogeneous delayed gradients with compute-rate normalization

**Question.** Does finite LIF memory provide a genuinely federation-specific freshness advantage once clients have different local objectives and slow computations return gradients evaluated at outdated server snapshots?

**Core setup.** Two equally weighted clients optimize `F_i(w)=0.5*(w-theta_i)^2` with `theta_slow=+0.05` and `theta_fast=-0.05`, so the intended aggregate optimum remains `w*=0`. A computation started by client `i` uses a server snapshot and returns only after `T_i` wall-clock ticks. The slow/fast delay ratio is swept over `R in {5,10,20,40,80}`. To remove compute-rate bias, a returned gradient contributes membrane evidence with gain `gamma_i = gamma * p_i * T_i`, making the mean evidence injection per wall-clock tick approximately proportional to the intended client weight rather than hardware speed.

Two labels are reported separately:

- **locally stale event:** the emitted sign disagrees with the client's exact local descent direction at the current server model;
- **globally harmful event:** applying the event increases the intended weighted aggregate objective.

A **fresh-gradient oracle** keeps the same completion schedule and rate normalization but evaluates the gradient at the current model when the job completes. This isolates computation-delay staleness from ordinary client heterogeneity and fixed-step/event regularization.

**Current conclusion.** True deterministic locally stale events can be produced under sufficiently strong delayed computation; in the stress design they first appear clearly around `R=40`, and every such event is globally harmful. However, the fresh-gradient oracle removes those stale events without materially improving IF's tail error or global-harm rate. Mild LIF can eliminate the stale events and reduce global harm, but its much larger improvement comes from suppressing the broader heterogeneous event tug-of-war / creating a finite-memory deadzone, not from approximating the fresh oracle. With randomized initial conditions at `R=40`, deterministic IF has only about `0.36%` locally stale events, while roughly `36%` of all events are globally harmful; LIF `rho=0.995` removes the stale events and reduces global harm strongly, but the causal oracle confirms that stale gradients are not the main source of the gain. Under noisy gradients the wrong-direction-event fraction grows with delay, but finite memory does not consistently reduce it at high `R`.

This is therefore another falsification of the strong freshness claim. The project now has evidence for temporal filtering/regularization and event suppression under heterogeneity, but not yet for a robust FL-specific stale-gradient benefit from leak itself.

Full run:

```bash
PYTHONPATH=src python experiments/08_heterogeneous_delayed_async.py
```

Smoke test:

```bash
PYTHONPATH=src python experiments/08_heterogeneous_delayed_async.py --quick
```

## P13 — CIFAR-10 ResNet-14 architecture extension

**Question.** Does Event-FedAvg retain a useful communication--quality trade-off
on a deeper residual model without starving complete parameter groups?

**Protocol.** A 175,258-parameter CIFAR-style ResNet-14 with GroupNorm uses the
same strong-skew CIFAR-10 federation and matched local-SGD interface as P3. A
development-only grid is followed by a three-seed validity gate and, only after
that gate passes, seven frozen extension seeds. Layer-level event activity and
final-horizon stability are audited explicitly. See
[`P13_CIFAR10_RESNET14_PROTOCOL.md`](P13_CIFAR10_RESNET14_PROTOCOL.md).

```bash
PYTHONPATH=src python experiments/p13_cifar10_resnet14.py smoke
PYTHONPATH=src python experiments/p13_cifar10_resnet14.py protocol
```

For an NVIDIA workstation, use the CUDA-native PyTorch runner:

```bash
bash tools/p13_workstation.sh setup
bash tools/p13_workstation.sh smoke
bash tools/p13_workstation.sh test
# After inspecting timing and test outputs:
bash tools/p13_workstation.sh full
```

## P15 — Event-encoder operator ablations

**Question.** How much of the selected compact CIFAR-10 CNN operating point is
explained by persistent cross-round state and full reset, rather than by the
complete Event-FedAvg operator?

The development stage retunes the threshold separately for a memoryless
encoder and a persistent subtractive-reset encoder. Selection minimizes the
absolute log-traffic distance to the frozen persistent/full-reset Event-FedAvg
reference, with final training loss used only to break ties. The selected
variants are then evaluated on the same ten CIFAR-10 partition and training
seeds used by P12 and P14. The authoritative frozen Event-FedAvg runs are
reused, so only twenty new held-out runs are required. The earlier coupled
threshold/quantum audit remains separate. Existing files are skipped unless
`--force` is given.

```bash
PYTHONPATH=src python experiments/p15_operator_ablation.py smoke
PYTHONPATH=src python experiments/p15_operator_ablation.py develop
PYTHONPATH=src python experiments/p15_operator_ablation.py select
PYTHONPATH=src python experiments/p15_operator_ablation.py heldout
PYTHONPATH=src python experiments/p15_operator_ablation.py aggregate
# Or run/resume every stage in order:
PYTHONPATH=src python experiments/p15_operator_ablation.py campaign
```

These experiments are not part of the current frozen manuscript evidence. The
paper therefore attributes results to the complete operator and explicitly
does not claim an isolated gain from either reset or persistent accumulation.

## P20 — Larger-client and partial-participation pilot

**Question.** Does Event-FedAvg retain its communication advantage with fifty
registered clients and uniform partial participation, once returning-client
catch-up is charged explicitly?

The one-seed pilot compares Event-FedAvg, quality-selected EF-TopK, and dense
FedAvg on the compact CIFAR-10 CNN. It includes a 50-client full-participation
control and a primary 10-of-50 partial-participation cell. The existing matched
10-of-10 P19 seed-zero runs are reused. See
[`P20_PARTIAL_PARTICIPATION_PROTOCOL.md`](P20_PARTIAL_PARTICIPATION_PROTOCOL.md).

```bash
python experiments/p20_cifar10_partial_participation.py smoke
python experiments/p20_cifar10_partial_participation.py pilot --workers 4
python experiments/p20_cifar10_partial_participation.py verify
python experiments/p20_cifar10_partial_participation.py analyze
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
- Distinguish sparse current-model sampling from true delayed gradients: a slow client is only genuinely stale if its returned gradient was computed from an older server snapshot.
- Under heterogeneous objectives, distinguish locally stale/wrong-direction events from globally harmful events; they are not the same phenomenon.
