# Experiment 10A — Vector heterogeneous asynchronous quadratics

## Question

Can sparse coordinate LIF events remain communication efficient once the model is genuinely vector valued and coordinate addresses must be transmitted?

## Primary setup

- 10 equally weighted clients.
- 20-dimensional diagonal strongly-convex local quadratics.
- Client compute periods `[1,1,2,2,5,5,10,10,20,20]`.
- Coordinate base curvatures span `0.2 -> 5.0` logarithmically.
- Client curvature perturbation std: `0.15` in log scale.
- Local optimum std: `0.25` per coordinate.
- Stochastic gradient noise std: `0.25`.
- Evidence/update contributions are normalized by `p_i*T_i`.
- Full run: 80 heterogeneous problem instances, 2000 wall-clock ticks, 500-tick tail window.

## Methods

- asynchronous full precision;
- dense sign communication;
- full-reset coordinate LIF with one global threshold;
- full-reset coordinate LIF with curvature-normalized thresholds;
- EF-TopK.

The reset-EMA trigger is represented by an equivalence check rather than a separate performance curve because it is exactly equivalent to full-reset LIF after coordinatewise rescaling.

## Communication accounting

For `d=20`:

- sparse LIF event: 5 address bits + 1 sign bit = 6 payload bits;
- dense sign: 20 payload bits/message;
- full precision: 640 payload bits/message;
- EF-TopK: `k*(32+5)` payload bits/message.

An additional diagnostic adds a 32-bit illustrative header per nonempty packet. Downlink communication is not included yet.

## Run

```bash
PYTHONPATH=src python experiments/10a_vector_quadratic.py
```

Smoke test:

```bash
PYTHONPATH=src python experiments/10a_vector_quadratic.py --quick
```

Outputs are written to `experiments/results/10a_vector_quadratic/`.

## Current conclusion

Coordinate normalization is essential: a global threshold makes event count almost perfectly track coordinate curvature, whereas normalized thresholds produce a nearly flat firing distribution with full coordinate coverage. A selected normalized-LIF point reaches a tail excess objective of about `0.00625` with about `10.3k` payload bits, compared with `0.00604` and about `1.095M` payload bits for EF-TopK with `k=4`. Packetized with an illustrative 32-bit header, the reduction is still about 26x.

The trade-off is a much slower transient: the selected LIF point has a whole-run excess objective about 4.5x larger than the matched EF-TopK point. The result therefore establishes vector communication viability, not uniform optimizer superiority. The basic full-reset LIF recurrence also remains exactly equivalent to a reset EMA trigger after rescaling.
