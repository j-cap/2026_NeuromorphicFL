# Short Event uplink/downlink codec pilot

Repository: **j-cap/2026_NeuromorphicFL** (research code, not the manuscript repo).
Entry point: `experiments/event_codec_pilot.py`.

## Purpose and fixed scope

Measure actual lossless-message sizes and reconstruction correctness on **120
rounds**, using one existing development partition. This is an early-trajectory
codec experiment, not a convergence study or a test-accuracy comparison. No
configuration is selected by its final objective or accuracy.

| Point | Training rule | Role |
|---|---|---|
| `event` | Existing persistent/full-reset Event; gain 1 | Four encodings of the same trajectory |
| `stc_primary` | Native STC, upload/download p=1; gain 2 | Lossless coding fairness control |
| `stc_sparse` | Native STC, upload/download p=.25; gain 2 | Lossless coding fairness control |

All three use the compact CIFAR-10 CNN (20570 parameters), 10 fully participating
clients, the existing strong partition with seed **4500**, SGD seed **84500**,
initialization seed 7777, five local SGD steps, batch 32, learning rate .05 and
L2 coefficient .0005. Existing 50000-example training allocation is unchanged.
Training diagnostics use the fixed 5000-example training subset at rounds 1,
30, 60, 90 and 120. The runner never evaluates the CIFAR test subset. Its loader
still reads the standard dataset batches, whose fingerprints are recorded.

Only three short training runs are needed. Alternative encodings are evaluated
on each point's exact same emitted messages, rather than retraining a model for
each encoding. Early-round activity need not represent late-round activity, so
do not extrapolate the measured percentage to 1800 rounds without verification.

## Integer pulse codec

`src/neuromorphicfl/event_codec.py` implements independently decodable byte
frames. Its 16-byte EVS1 header transmits the format version, dimension, nonzero
count, maximum absolute integer, support/value coding modes and Rice exponent.
This is **128 bits**, compared with the historical Event accounting's abstract
64-bit header. The extra metadata is charged in every new frame.

Candidate formats are sorted-position Rice gaps, explicit positions, activity
bitmap, full-support values, dense signed unary integers and dense fixed-width
integers. Sparse/full-support values use either fixed-width or unary magnitudes
plus signs. For uplink pulses the magnitude is implicitly one. Selection
minimizes complete byte length using exact bit-length formulas, which are checked
against serialization. Padding is charged and validated. The first version
selects globally per message; it does not add blockwise or temporal predictors.

Model ordering, round context and the shared pulse schedule are agreed by the
surrounding synchronous protocol. These frames do not implement networking,
timestamps, retransmission, TCP/TLS overhead or asynchronous catch-up.

## Aggregate downlink and floating-point arithmetic

The server sums signed pulses into integer counts, preserving multiplicity and
cancellation. The count codec reconstructs those integers exactly. Multiplying
the count by the common jump can nevertheless round differently from the frozen
runner's sequential float32 pulse additions. The pilot records that difference
and measures two **bitwise exact** server-message options:

1. Encoded aggregate counts plus a sparse correction mask and original float32
   values wherever count-times-jump differs. Envelope, mask, values and padding
   are all charged. The corrections retain cancelled-coordinate roundoff too.
2. A fixed coordinate scan containing each coordinate's unary pulse count and
   ordered signs. It retains the historical addition order without client IDs.
   This reference costs `d + 2*K` payload bits per round, plus its 128-bit header
   and byte padding.

The shorter fully serialized exact frame is used. The count-only frame size is
also recorded, but is **a diagnostic**, not an assertion of bitwise equivalence.
Every uplink is decoded before server aggregation. Every chosen downlink is
decoded independently; its update and a synchronized client-model replica must
match the server bit for bit, including the existing server gain application.

## Fairness controls and reported variants

Event reports historical address/replay accounting, packed-uplink replay,
aggregate exact downlink with old upload accounting, and both codecs together.
Historical replay retains the cheaper-of-replay-or-float32-checkpoint rule.
The new downlink variants charge their actual exact frames without a fallback.

STC reports its native actual byte counts and a losslessly recoded variant.
The adapter preserves each already-quantized tensor's float32 magnitude and
encodes the concatenated signs/support using the same integer codec as Event.
It **does not** change top-k selection, ternarization, gains, client residuals or
server residuals. Both uploads and native server messages are checked bitwise.
Full-support STC signs can thus omit redundant position delimiters.

Every variant charges the initial float32 model (32*d+64 bits), a 32-bit request
per receiver per round in the conservative unicast model, and the appropriate
replication of the server stream. Logical broadcast is reported separately.
The saved `codec_audit_seconds` includes reference encodings and reconstruction
checks; it is not a deployment encoder benchmark or a method-runtime ranking.

## Run locally

Use the same existing environment, from the **code** repository:

```bat
cd /d C:\Users\WeberJ\2026_NeuromorphicFL
git pull --ff-only
conda activate neuromorphicfl-resnet14
python experiments/event_codec_pilot.py verify
python experiments/event_codec_pilot.py preflight
python experiments/event_codec_pilot.py run --workers 2
python experiments/event_codec_pilot.py analyze
```

For lower resource use while STC confirmation is running, use `--workers 1`.
To run Event first, use `run --point event --workers 1`; run the default command
later to finish the two controls. Analysis requires all three points and never
silently drops a point. `status` reports pending and complete points.

Checkpoints are saved every 30 rounds. Repeating `run` resumes compatible
checkpoints and skips verified completed points. Changes to sources, data,
runtime or protocol are rejected. Numerical failures are saved explicitly.
All frozen STC development/confirmation source files remain unchanged.

## Outputs and next decision

`experiments/results/event_codec_pilot_v1/` contains the protocol/data/source/
runtime manifest and, per point, summary, per-round and training-history CSVs
with a completion receipt. `analyze` writes `comparison.csv` and
`analysis_audit.json`. CSV/JSON evidence is Git-enabled; checkpoint NPZ files and
lock files are ignored.

Inspect actual upload/download savings, the selected support modes, aggregate
nonzeros and cancellation, rounding corrections, and the STC dense-support gain.
The next decision is whether the message representation is correct and useful
enough to adopt before developing the two-state trigger. This pilot establishes
neither final predictive performance nor a communication advantage at matched
accuracy.

After completion:

```bat
git add experiments/results/event_codec_pilot_v1
git commit -m "Add short Event and STC codec pilot results"
git push
```
