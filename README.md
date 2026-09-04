# NeuromorphicFL

Research workspace for **neuromorphic federated learning via leaky integrate-and-fire (LIF) gradient communication**.

The central idea is to reinterpret communication-efficient distributed/federated optimization as an event-driven hybrid dynamical system. Local stochastic-gradient information is accumulated in LIF-like communication states. A client communicates only when accumulated evidence reaches a threshold, producing fixed signed parameter-update events whose timing/rate carries gradient-magnitude information.

## Repository structure

- `src/` — reusable objectives, optimizers, scalar and asynchronous simulation utilities, and metrics.
- `experiments/` — self-contained experiment scripts built on `src/` where appropriate.
- `report/` — evolving LaTeX knowledge base, bibliography, and figure conventions.
- `paper/ijcnn2027/` — submission plan and, once extracted, the concise IJCNN manuscript.

## Current experiment progression

1. **Stationary stochastic quadratic** — temporal evidence accumulation, event-rate encoding, wrong-sign probability, communication reduction, and leak.
2. **Large moving optimum** — tests externally changed objectives and exposes the communication–responsiveness trade-off.
3. **Moderate moving optimum** — controls the change-point state and shows that leak does not generically improve adaptation speed.
4. **Controlled stale membrane** — separates stale-sign erasure from useful threshold firing and identifies the LIF deadzone.
5. **Controlled asynchronous model drift** — keeps the local objective fixed while the server model moves, isolating the FL-specific source of gradient staleness.
6. **Two-client asynchronous learning** — a selected noisy operating point initially suggested a mild finite-memory advantage.
7. **Asynchrony × memory regime map** — shows that the same finite-memory optimum appears even at `R=1` and disappears in the zero-noise homogeneous control, so the robust explanation is stochastic evidence filtering / event regularization rather than a proven freshness benefit.
8. **Heterogeneous delayed gradients with compute-rate normalization** — introduces true in-flight stale gradients and different local optima. Genuine locally stale events can be observed under strong delays, but a fresh-gradient oracle shows that they are not the main source of IF degradation. LIF's larger gains come from suppressing heterogeneous event activity / creating a finite-memory deadzone, not from robust stale-gradient correction.

Two key negative results are now part of the project knowledge base:

- **reduced communication can be caused by silencing slow clients**, especially under wall-clock leakage;
- **finite LIF memory has not demonstrated a robust FL-specific stale-gradient advantage** in the scalar tests, even when true delayed gradients and heterogeneous local objectives are introduced.

The project has since consolidated the supported mechanism into Event-FedAvg:
conventional multi-step local model deltas drive persistent leaky evidence
states; sparse signed coordinate events update the server; and ordered replay
with checkpoint fallback provides complete bidirectional synchronization. The
matched-baseline campaign and theory tasks T1--T4 are complete. The P3
CIFAR-10 compact-CNN campaign also closes with a pass: Event-FedAvg remains on
the held-out communication--performance frontier. P4 has now extracted the
compact operator, encoder bounds, conditional optimization result, and
exact-gradient alignment audit into the manuscript. P5 has now frozen the
three-element visual argument: method schematic, cross-benchmark frontier, and
compact headline table. The current priority is P6 manuscript construction,
not open-ended mechanism discovery.

## Publication planning

The gate-based plan for extracting the conference paper is maintained in
[`paper/ijcnn2027/PLAN.md`](paper/ijcnn2027/PLAN.md). It intentionally contains
no internal calendar dates: progress is controlled by scientific and
reproducibility gates.

The report in `report/main.tex` is intentionally more detailed than a paper draft. It is the project knowledge base from which the manuscript is distilled, and it records both successful mechanisms and falsified hypotheses.
