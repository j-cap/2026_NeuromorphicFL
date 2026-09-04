# P2 literature and novelty audit

This document freezes the literature boundary used by the IJCNN manuscript.
It records a targeted search snapshot, not a proof that no unpublished or
unindexed equivalent exists.

## Search snapshot and protocol

- **Snapshot:** 2026-09-04.
- **Sources:** primary publisher pages, proceedings pages, and author/arXiv
  manuscripts when publisher full text was unavailable.
- **Query families:** event-triggered federated/distributed learning; temporal
  and lazy communication; accumulated/residual/error-feedback compression;
  sign, ternary, and quantized updates; level-crossing and pulse communication;
  federated spiking/neuromorphic learning; and combinations of leak, reset,
  threshold, pulse, and federated averaging.
- **Screening rule:** compare update equations and synchronization semantics,
  not titles or use of the words *event*, *spike*, or *neuromorphic*.
- **Pre-submission action:** repeat the same query families in P10 and append
  any new close neighbor. This is necessary because the submission date is in
  the future relative to this snapshot.

## Frozen operator used for comparison

For client \(i\), coordinate \(j\), and round \(r\), Event-FedAvg integrates a
weighted local FedAvg delta in a persistent state, tests the coordinate once,
emits only its address and sign on threshold crossing, and clears the fired
state. The server applies a model quantum \(q_r\) that is scheduled independently
of the trigger threshold. Client reconstruction uses ordered sparse replay with
a dense-checkpoint fallback. The novelty question concerns this full operator
and protocol, not any ingredient in isolation.

## Operator-level comparison

| Method/family | Persistent local state | Leak | Trigger/granularity | Reset or error treatment | Transmitted value | Server resolution | Silence and synchronization |
|---|---|---|---|---|---|---|---|
| **Event-FedAvg (ours)** | Weighted FedAvg-delta evidence per client-coordinate | \(0<\rho<1\) | One first-passage test per active coordinate and round | Full clear; overshoot is discarded | Coordinate address and sign | Independent, annealed \(q_r\) | No uplink pulse when silent; ordered aggregate replay with dense checkpoint fallback gives exact catch-up |
| **Strom threshold pulses** | Residual gradient/update per coordinate | No forgetting (equivalent to \(\rho=1\)) | Coordinate residual crosses a fixed threshold | Subtracts a threshold quantum and retains residual/overshoot | Coordinate address and sign/pulse | Coupled to the threshold quantum | Silent coordinate retains conserved residual; original work is an uplink distributed-SGD compressor |
| **Error feedback / sparse SGD with memory / EF21** | Compression error or estimator memory | Normally no | Compressor selected every iteration, often top-\(k\) or contractive | Compression residual is retained for correction | Compressed vector and values | Determined by compressor output | Round-based transmission; synchronization follows the host optimizer, not an event replay protocol |
| **STC** | Client/server residual memories | No | Sparsifies each selected round | Subtracts the sparse ternary reconstruction; error is retained | Sparse ternary vector with data-dependent common magnitude | Transmitted magnitude | Compresses both directions, but does not use leaky first-passage state or full-clear pulses |
| **1-bit SGD / signSGD** | Error-feedback state in 1-bit SGD; none in basic signSGD | No | Quantizes every transmitted iteration/coordinate | Error carried forward in 1-bit SGD; no firing reset in signSGD | Sign vector (plus scaling where applicable) | Step size or quantizer scale | Periodic communication; no silence-by-first-passage or replay/checkpoint semantics |
| **LENA** | Accumulated error/memory vector | No | Self-trigger based on memory significance | Uploaded memory can be cleared; silent workers are represented by server drift | Accumulated vector, not a one-bit coordinate pulse | The uploaded vector itself | Server maintains/drifts worker estimates; standard model synchronization |
| **SPARQ-SGD** | Last communicated model and neighbor estimates | No | After local steps, test model deviation at node/vector level | Communicated cache/reference is updated | Quantized and sparsified model change | Compressed change used in decentralized mixing | Silent nodes leave stale neighbor estimates; decentralized, without central sparse replay/checkpoint |
| **DETSGRAD / EventGraD / ETFL** | Last transmitted gradient/model or local reference | No | Aperiodic trigger from model/gradient deviation, usually vector-level | Reference updates on communication | Gradient or model information, optionally top-\(k\) compressed | Message value itself | Event-triggered uplink and, for ETFL, downlink; no coordinate full-clear pulse with independent quantum |
| **Event-triggered gossip (2026)** | Last broadcast model and neighbor-side estimates | No | Adaptive trigger from local-model deviation | Broadcast reference updates on an event | Model information to graph neighbors | Gossip/mixing update | Fully decentralized neighbor exchange; no server, pulse quantum, or replay/checkpoint layer |
| **Federated SNN/SFL** | Membrane/firing state inside the trained SNN; ordinary FL optimizer state | LIF leak is inside neurons, not the communication compressor | Model aggregation rounds; some works add masking, dropout, sparsification, or distillation | Architecture/training-specific, not a communication-state full reset | SNN weights, updates, distilled information, or sparse masks | Aggregation/compression dependent | Trains spiking models; it does not place a LIF encoder around conventional FedAvg deltas |

## Family-by-family conclusion

### Residual, sign, and quantized communication

Strom is the closest algorithmic predecessor because it accumulates
coordinate-wise residuals and sends signed threshold pulses. Its state is
conservative: emitted quanta are subtracted and overshoot remains. Event-FedAvg
instead leaks old evidence, clears a fired coordinate completely, and separates
the threshold that decides *when* to communicate from \(q_r\), which decides
*how far* the model moves. These are semantic differences, not a renaming.
Error-feedback and STC are also essential comparators, but their defining goal
is to preserve and later correct compression error rather than deliberately
forget evidence through leak and full reset.

### Event-triggered and temporal communication

LENA is the closest precedent for an accumulated memory that may be cleared,
but it uploads a vector-valued memory and uses server drift for silent workers.
SPARQ-SGD, DETSGRAD, EventGraD, bidirectional ETFL, the 2025
communication-balancing threshold, and 2026 event-triggered gossip decide
whether a model or gradient estimate has changed enough to justify a message.
They do not expose the same coordinate pulse alphabet, leaky state, full-clear
reset, and independent applied resolution. Level-crossing estimation supplies
a useful signal-processing analogy for information carried by event time and
sign, but it is not a federated optimization and synchronization protocol.

### Neuromorphic and spiking federated learning

Federated SNN work uses spikes and LIF dynamics inside the model being trained.
Representative work trains SNNs with FedAvg, masks model updates or drops
clients, adapts heterogeneous SNN widths or temporal resolutions, or exchanges
distilled/spiking-model information. Event-FedAvg instead leaves the predictive
model conventional and uses LIF-inspired dynamics only to encode client update
evidence. The manuscript must therefore say *LIF-inspired communication
operator*, never *the first neuromorphic FL method*.

## Novelty verdict

The targeted search did **not identify a predecessor containing the complete
frozen operator**: persistent leaky weighted-delta state, coordinate
first-passage signs, full reset with discarded overshoot, a trigger-independent
annealed server quantum, algebraic aggregation without second weighting, and
exact ordered sparse replay with checkpoint fallback. This is a qualified
search result. It does not establish universal priority and must be rechecked
at P10.

Closest neighbors are:

1. **Strom:** closest pulse/residual recurrence and mandatory primary
   comparator.
2. **LENA:** closest accumulated-memory/full-clear communication precedent.
3. **SPARQ-SGD and ETFL:** closest learned-optimization event-trigger families.
4. **Level-crossing estimation:** closest time/sign event-coding analogy.
5. **Federated SNN:** closest use of neuromorphic terminology, but at a
   different layer of the system.

## Manuscript-ready claim

> We do not claim novelty for accumulation, thresholding, sign transmission,
> error feedback, event-triggered learning, or LIF dynamics in isolation. Our
> contribution is the complete Event-FedAvg communication operator: weighted
> local-model deltas drive persistent leaky coordinate states; a first-passage
> event transmits only sign and address; firing fully clears the state; and the
> server applies an independently scheduled model quantum. Combined with
> algebraic aggregation and exact replay/checkpoint synchronization, this
> yields semantics distinct from the closest residual-conserving Strom pulse
> compressor and from model-deviation event triggers.

## Language constraints

Allowed:

- “Our targeted search did not identify the complete operator in prior work.”
- “Strom-style residual pulses are the closest algorithmic predecessor.”
- “Event-FedAvg is LIF-inspired at the communication layer.”

Disallowed:

- “the first neuromorphic federated-learning algorithm”;
- “the first event-triggered FL method”;
- “a novel LIF neuron/threshold/sign mechanism”;
- any implication that an absence search proves universal priority;
- any hardware-energy claim without measurements.

## Primary-source reading set

The manuscript bibliography contains the curated references. Particularly
important source records are Strom (Interspeech 2015), LENA (AISTATS 2021),
STC (TNNLS 2020), SPARQ-SGD (TAC 2023), ETFL (TSP 2023), the 2025
communication-balancing threshold, the 2026 event-triggered gossip preprint,
and representative federated-SNN papers from TSP 2021 and 2026.
