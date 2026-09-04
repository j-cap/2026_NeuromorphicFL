# Event-FedAvg IJCNN paper contract

This contract freezes the intended conference-paper story before manuscript
extraction. Changes to the central claim require an explicit update here and in
`CLAIM_EVIDENCE.md`.

## Working title

**Event-FedAvg: Stateful Event-Driven Communication for Federated Learning**

The title deliberately leads with the algorithm and its communication role.
The leaky integrate-and-fire inspiration is explained in the paper rather than
used to imply that the trained model is a spiking neural network.

## Central question

Can a leaky integrate-and-fire-inspired, stateful communication layer provide
a useful communication--performance operating point for conventional
multi-step federated averaging?

## Intended answer

Yes, within the evaluated scope. Event-FedAvg accumulates weighted local-model
deltas in persistent leaky evidence states, emits sparse signed coordinate
events after first passage, fully resets fired coordinates, and applies an
independently scheduled server-model quantum. Across the established
Fashion-MNIST campaigns and the independent CIFAR-10 compact-CNN campaign, it
provides a strong predictive operating point under conservative bidirectional
communication accounting. On CIFAR-10 it remains nondominated and stronger
than the closest tested residual-pulse alternative at comparable traffic.

This answer is bounded to the evaluated synchronous, full-participation,
software-accounted settings. The three-seed CIFAR-10 result supports transfer
to a second image dataset and architecture, not broad dataset generalization or
statistical significance.

## Contributions

1. **A complete Event-FedAvg communication operator.** Conventional multi-step
   local FedAvg deltas drive leaky client evidence states. First-passage signed
   coordinate events, full reset, an independent model quantum, and exact
   replay/checkpoint synchronization define the complete algorithm and
   end-to-end protocol.
2. **A scoped mathematical characterization.** The encoder has a bounded
   post-reset state and a pathwise event budget. Smoothness yields an exact
   descent-versus-harm decomposition, and an explicit aggregate-alignment
   condition yields a finite-horizon stationarity result.
3. **Matched communication--performance evidence.** Development-only tuning,
   held-out partitions, close residual and error-feedback baselines, and
   symmetric uplink-plus-downlink accounting test the practical operating
   point. An exact empirical-gradient audit assesses the theory's alignment
   condition without changing the training rule.

## Novelty boundary

The contribution is the complete combination of:

- persistent leaky evidence;
- one threshold test per active client-coordinate-round;
- signed coordinate transmission;
- full reset that discards threshold overshoot;
- a model-update quantum independent of the trigger threshold;
- algebraic server aggregation without a second client weighting;
- ordered sparse replay with dense-checkpoint fallback.

The paper does not claim novelty for accumulated gradients, threshold
sparsification, sign communication, error feedback, event-triggered learning,
LIF dynamics, or the generic accumulate--threshold--pulse motif. The P2
operator audit treats Strom-style residual-conserving pulses as the closest
algorithmic predecessor, LENA as the closest accumulated-memory/full-clear
precedent, and model-deviation triggers as a separate event-communication
family. The defensible wording is that the targeted search did not identify
the complete frozen operator; this is not a universal priority claim and must
be rechecked at P10.

## Claim boundary

The paper may claim:

- exact algorithm and synchronization semantics;
- encoder-state and event-budget properties;
- a deterministic one-step descent inequality;
- conditional finite-horizon stationarity;
- an empirical communication--performance frontier;
- finite-horizon empirical aggregate alignment;
- negative qualifications concerning late-stage descent and schedule decay.

The paper must not claim:

- unconditional convergence of the complete model-coupled method;
- a uniform pathwise alignment constant with zero defect;
- uniform accuracy superiority over every baseline operating point;
- measured hardware energy savings or deployment latency;
- robustness to partial participation or asynchronous FedAvg unless new
  evidence is added;
- broad dataset generalization beyond the tested Fashion-MNIST and CIFAR-10
  settings.

## Intended reader takeaway

After reading the paper, a federated-learning researcher should be able to
answer four questions:

1. What state does Event-FedAvg retain and when does it communicate?
2. How does it differ from residual-conserving threshold pulses?
3. What can and cannot currently be proven about its optimization behavior?
4. What predictive quality is obtained for a given total communication cost?

## Main-paper structure and page budget

The manuscript must fit the current six-page IJCNN limit as a self-contained
paper. The budget below is a constraint, not a requirement to fill space.

| Section | Target pages | Purpose |
|---|---:|---|
| Abstract and introduction | 0.65 | Problem, result, contributions, limits |
| Related work | 0.40 | Closest operator-level distinctions |
| Method and protocol | 1.15 | Frozen transition, algorithm, communication |
| Theory | 0.80 | Encoder properties and conditional optimization result |
| Experimental protocol | 0.55 | Fair tuning, partitions, metrics, accounting |
| Results and discussion | 1.20 | Frontier, matched controls, alignment audit |
| Limitations and conclusion | 0.25 | Precise boundary and takeaway |
| References | 1.00 | Curated citations only |

## Visual budget

The main paper should use at most three principal visual elements:

1. **Method figure:** local delta, leaky evidence, event/reset, server update,
   and client synchronization.
2. **Frontier figure:** predictive performance versus total bidirectional
   communication for both final benchmarks.
3. **Main result table with compact alignment evidence:** quality-selected and
   traffic-matched comparisons plus the minimum diagnostic required to support
   the conditional theory.

The full T4 decomposition, schedule comparison, observer-effect audit, and
exploratory mechanism figures remain in the living report.

## Author and review roles

The author list is intentionally not inferred from repository history. Before
P0 is closed, the project team must confirm:

- lead author and corresponding author;
- all co-authors and contribution expectations;
- FL-method reviewer;
- mathematical reviewer;
- final presentation/compliance reviewer.

These assignments may be maintained privately if public repository disclosure
would be inappropriate during anonymous review.

## Scope-control decision

The default final scientific addition, the CIFAR-10 compact-CNN campaign, is
complete. T5 is not part of the conference scope unless a focused review shows that the coupled
alignment mechanism is a submission-blocking theoretical gap. Partial
participation, network emulation, hardware energy measurement, and stronger
coupled convergence theory are journal-extension candidates.
