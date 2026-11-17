# Core comparator extension

The core numerical comparison will contain Event-FedAvg and four comparators:
Dense FedAvg, STC, Ström-style residual pulses within FedAvg, and EF-TopK.
Sign-EF remains available as supporting evidence. LENA and SPARQ remain
related-work entries unless a separate, explicitly scoped comparison is added.

## Implementation sequence

| Stage | Deliverable | Completion criterion |
|---|---|---|
| 1 | STC compressor, wire codec, two residual levels, NumPy integration | Hand-reference, conservation, wire-length and model-reconstruction checks; bounded CNN smoke; existing-method regression |
| 2 | Compact-CIFAR development campaign | Declared search and selection rules, separate development data, frozen configurations, resumable local commands |
| 3 | PyTorch ResNet integration and main benchmark extension | Same compressor conventions and traffic accounting, checkpointed residuals, ten paired evaluation seeds |
| 4 | Publication evidence | All five methods represented consistently; conclusions follow observed results; manuscript changes in red |

Stage 1 provides implementation verification only. No STC research result,
configuration selection, or submission claim follows from its smoke output.
Stage 2 is implemented in `stc_cifar10_development.py`; its fixed development
search, selection rules, checkpoint convention and local commands are documented
in [STC_CIFAR10_DEVELOPMENT.md](STC_CIFAR10_DEVELOPMENT.md). Development results
must be analysed and selected configurations frozen before held-out campaigns.
The completed development analysis and the recommended ten-seed confirmation
are in [STC_CIFAR10_DEVELOPMENT_RESULTS.md](STC_CIFAR10_DEVELOPMENT_RESULTS.md).
Use `python experiments/stc_cifar10_development_audit.py` for verification after
Git has normalized Windows text line endings; it preserves the frozen source
and original receipt hashes.

## Common experimental contract

Within each benchmark, all five methods share the model, initialization, client
partition, local optimizer, minibatch schedule, local training work, participation
and horizon. Compressor parameters may differ and must be tuned using declared
development-only rules. Keep the existing dense gain/schedule fairness audit.
Preserve eligible frozen results rather than rerunning the existing campaigns.

STC uses the published compression/residual structure inside this common
multi-step local-training host. Record the local step count and client weighting
explicitly; this is not a claim to reproduce the original paper's complete
training recipe. The initial integration covers full participation. The existing
partial-participation runner does not yet implement STC.

## STC state and update convention

Reference: Sattler et al., *Robust and Communication-Efficient Federated Learning
from Non-IID Data*, Algorithms 1--2 and Appendix A,
<https://arxiv.org/abs/1903.02891>.

For local model delta `delta_i`, client residual `r_i`, normalized client weight
`a_i`, server residual `R`, and server gain `gamma`, each round implements:

```
u_i = delta_i + r_i
c_i = decode(encode(STC_up(u_i)))
r_i = u_i - c_i
v   = R + gamma * sum_i(a_i * c_i)
s   = decode(encode(STC_down(v)))
R   = v - s
w   = w + s
```

The server sends the encoded `s` to clients, which reconstruct the same update.
Weights are applied once, at aggregation. Client residuals are unweighted local
model deltas; the server residual has applied-model-update units. Gain one gives
the reference server update. If a constant nonunit server gain is used, it acts
before server compression and error feedback, never twice on the decoded update.
The current shared runner permits gain schedules only for Dense FedAvg.

For constant gain and fixed weights, the retained information satisfies, up to
floating-point roundoff:

```
gamma * sum_i a_i * sum_t delta_i(t)
    = sum_t s(t) + gamma * sum_i a_i * r_i(final) + R(final)
```

This identity and an independent two-round hand example are tested. STC retains
compression error at both levels; it does not leak or clear untransmitted error.

## Frozen implementation conventions

- Compression is per parameter tensor, using the layout's `groups()` order.
- `k = max(1, floor(p*n))`. Equal magnitudes are resolved by lower coordinate
  index, giving deterministic exact-k selection. These integer/tie conventions
  make the paper's pseudocode unambiguous.
- The selected magnitudes are averaged with a float64 reduction and transmitted
  as one float32 scale. Selected zeros remain zero. All arithmetic state is
  float32. `p=1` still ternarizes and is not an identity-compression check.
- Position gaps use the power-of-two Golomb/Rice code in the reference, followed
  by one sign bit per nonzero. The Rice exponent follows its analytic density
  rule, with density clipped below at `1/n` and exponent below at zero.
- The message has a 64-bit format/tensor-count frame header. Each tensor has a
  32-bit nonzero count, float32 scale and 8-bit Rice exponent. Tensor symbol
  streams are individually zero padded to byte boundaries. Tensor sizes/order
  are agreed with the model. All transmitted metadata and padding are charged.
- Even an all-zero tensor has metadata; STC sends an update message every round.

`STCPacket.wire_bits` is eight times the actual byte-message length. It is not an
entropy estimate. The decoder reconstructs the message independently of the
source update.

## Bidirectional accounting

Use the existing common network conventions: an initial float32 model plus a
64-bit header; one 32-bit request per client per round under conservative unicast;
one copy of each server message for broadcast, or one per client for unicast.

STC uploads are charged per client. Its native server-compressed packet is
charged separately on the downlink. Never replace that packet with a replay of
client uploads. Stage 1 does not add a dense-checkpoint fallback to native STC.

For compatibility, `unicast_hybrid_downlink_bits` and
`unicast_hybrid_total_bits` contain STC's native unicast costs when `method=stc`.
`downlink_protocol=stc_compressed_server_update` identifies that interpretation;
`unicast_total_bits` is an explicit alias. Native STC rounds count as neither
`replay_rounds` nor `checkpoint_rounds`. Endpoint and history totals must agree.

## Local verification

Use the existing NumPy benchmark environment with `requirements.txt` installed:

```
python experiments/stc_stage1.py verify
```

This runs the unit/integration tests and a two-round synthetic CIFAR-CNN smoke
for the five core methods. It downloads no data and writes no research results.
For only the smoke, use `python experiments/stc_stage1.py smoke`.

Stage 4 must replace the historical five-method frontier registry (which
includes Sign-EF) with the agreed core set and update its frozen design checks.
New checks must verify evidence completeness and consistency, without requiring
Event-FedAvg to be nondominated. Keep existing publication evidence frozen until
the replacement comparison is complete, and mark manuscript edits in red.
