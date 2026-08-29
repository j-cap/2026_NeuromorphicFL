# NeuromorphicFL

Research workspace for **neuromorphic federated learning via leaky integrate-and-fire (LIF) gradient communication**.

The central idea is to reinterpret communication-efficient distributed/federated optimization as an event-driven hybrid dynamical system. Local stochastic-gradient information is accumulated in LIF-like communication states. A client communicates only when accumulated evidence reaches a threshold, producing fixed signed parameter-update events whose timing/rate carries gradient-magnitude information.

## Repository structure

- `src/` — reusable objectives, optimizers, scalar and asynchronous simulation utilities, and metrics.
- `experiments/` — self-contained experiment scripts built on `src/`.
- `report/` — evolving LaTeX knowledge base, bibliography, and figure conventions.

## Current experiment progression

1. **Stationary stochastic quadratic** — temporal evidence accumulation, event-rate encoding, wrong-sign probability, communication reduction, and leak.
2. **Large moving optimum** — tests externally changed objectives and exposes the communication–responsiveness trade-off.
3. **Moderate moving optimum** — controls the change-point state and shows that leak does not generically improve adaptation speed.
4. **Controlled stale membrane** — separates stale-sign erasure from useful threshold firing and identifies the LIF deadzone.
5. **Controlled asynchronous model drift** — keeps the local objective fixed while the server model moves, isolating the FL-specific source of gradient staleness.
6. **Two-client asynchronous learning** — lets a fast client move the global model underneath a slow client's stored evidence. Mild LIF memory currently shows a promising soft-freshness trade-off between infinite IF memory and aggressive hard reset.

The report in `report/main.tex` is intentionally more detailed than a paper draft. It is the project knowledge base from which a later manuscript can be distilled. Current results are mechanism diagnostics, not yet paper-level claims; the next stage is systematic asynchronous parameter and heterogeneity sweeps.
