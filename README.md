# NeuromorphicFL

Research workspace for **neuromorphic federated learning via leaky integrate-and-fire (LIF) gradient communication**.

The central idea is to reinterpret communication-efficient distributed/federated optimization as an event-driven hybrid dynamical system. Local stochastic-gradient information is accumulated in LIF-like communication states. A client communicates only when accumulated evidence reaches a threshold, producing fixed signed parameter-update events whose timing/rate carries gradient-magnitude information.

## Repository structure

- `src/` — reusable objectives, optimizers, simulation utilities, and metrics.
- `experiments/` — self-contained experiment scripts built on `src/`.
- `report/` — evolving LaTeX knowledge base, bibliography, and figure conventions.

## Current experiments

1. **Stationary stochastic quadratic**: tests temporal evidence accumulation, event-rate encoding, wrong-sign probability, communication reduction, and the effect of leak.
2. **Moving optimum**: tests adaptation under a suddenly changing objective and distinguishes stale membrane evidence from the more general communication–responsiveness trade-off.

The report in `report/main.tex` is intentionally more detailed than a paper draft. It is the project knowledge base from which a later manuscript can be distilled.
