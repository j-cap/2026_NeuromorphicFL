# Faster execution of the selected ResNet confirmation

The confirmation runner now uses a **byte-equivalent faster CPU codec**.
This change was prepared before the confirmation started. The same thirty
Dense/EF/STC fits, ten seeds, 1800 rounds, local SGD, model initialization,
float32 arithmetic and twenty pulse reuses remain in the
[confirmation protocol](RESNET14_CORE_CONFIRMATION.md).

## The measured bottleneck

The selected development runs each contain 1800 recorded training/codec times.
Their accumulated components on the recorded Windows/RTX 4090 setup are:

| Selected method | Local SGD, hours | Codec section, hours | Codec share of these two recorded components |
|---|---:|---:|---:|
| Dense gain 1.5 | 0.271 | 0.004 | 1.6% |
| EF-TopK p=0.01, gain 4 | 0.270 | 0.113 | 29.5% |
| STC p=1, gain 4 | 0.327 | 1.941 | 85.6% |

Source: round files in `results/resnet14_baseline_development_v1/`, results
commit `1eb535792cc882d9d1da910e5e4bb5863935426b`. Evaluation, checkpointing,
setup and final analysis are outside these two recorded components, so their
sum is not complete wall-clock time. The codec section also includes vector
transfers, compression and model application; it is not serialization alone.

The dominant optimization opportunity is therefore the STC codec, rather than
reducing the statistical confirmation or changing the shared learning problem.

## Implementation and fidelity

The frozen pulse codec emits/reads individual fields in Python. Near-full
STC messages still incur a large position/sign-processing cost, and every
round includes ten client packets plus a server packet.

`fast_pulse_codec.py` accelerates integer-bound-one messages using NumPy
bit packing and C-level regular-expression traversal of Rice symbol boundaries.
It retains the original exact candidate lengths, lexicographic mode/value-code/
exponent ties, headers, signs, bit order and padding. It creates and decodes
the actual bytes, rather than estimating their lengths. Other integer bounds
and the rarely forced dense modes delegate to the frozen codec.

`resnet14_baseline_fast_host.py` loads private instances of the original host
and float32 wire modules, replacing only those instances' pulse encoder and
decoder references. The original imported modules are not mutated. Compression
support selection, ordered reductions, client/server residuals, float32
scales, server gain, training, evaluation and checkpoint code remain the
original functions. Frozen package files, baseline host and calibration
drivers are unchanged. The core contract additionally pins both acceleration
sources and records backend `EVS1_bound1_packbits_regex_v1`.

Verification covers exact packet bytes and metadata for every applicable mode,
zero/full/nearly-full/sparse support, byte boundaries, larger-bound fallback,
handwritten unary value codes, malformed headers/truncation/padding, STC scales
including subnormals, and exact updates/residuals/counters for all three
selected methods. Synthetic actual-ResNet trajectories compare the faster
runner with both the original encoded runner and native reference: models,
client replicas, feedback, NumPy/Torch RNG, predictions and traffic match
exactly. Interrupted/resumed execution matches the uninterrupted trajectory.

The full confirmation verification also exercises the thirty-fit workflow,
twenty read-only reuses, failures, target censoring and unchanged historical
Event/Ström paired statistics. Compatibility is checked with NumPy 1.26.4,
the version used by the recorded Windows environment. No new dependency is
required beyond the existing standard library and NumPy/Torch environment.

## Synthetic local CPU benchmark

The [saved benchmark](results/resnet14_codec_speed_v1/synthetic_linux_cpu.json)
uses ten synthetic client deltas, the real ResNet tensor sizes (175258 total
parameters), the selected method configurations and explicit zeros every
97th coordinate. Thus STC exercises nearly full support, including Rice
positions, rather than only an all-nonzero special case. Each measurement
includes the full compressor/codec step, with one warm-up per backend followed
by three alternating-order repetitions. Median CPU seconds are:

| Method | Original | Faster backend | Original/faster ratio |
|---|---:|---:|---:|
| Dense | 0.00342 | 0.00349 | Same code path; small timing variation |
| EF-TopK p=0.01, gain 4 | 0.16794 | 0.04399 | 3.82x |
| STC p=1, gain 4 | 5.62601 | 0.45131 | 12.47x |

This benchmark ran on this workspace's Linux CPU, Python 3.12.14, NumPy 2.3.5
and Torch 2.5.1+cpu. It verifies exact update/residual bits and every frame
counter during each measurement. It performs no local SGD, image loading or
research training and does not predict the Windows/GPU wall-clock speedup.
The numerical byte-parity checks also run under the target NumPy 1.26.4.

The STC training stage should benefit most. Its complete speedup will be
smaller than the codec speedup because local SGD, evaluation and checkpointing
still take time. Dense has essentially no codec work to remove. These
implementation-specific CPU timings must not be used as a physical
communication/algorithm-latency comparison against the reused pulse runs.

## Run and optionally measure your own CPU

The ordinary confirmation `verify`, `preflight` and `run` commands are unchanged;
pull the new code first. `preflight` prints the faster backend identity.
The [main protocol](RESNET14_CORE_CONFIRMATION.md) includes the complete run,
resume and result-publication commands.

An optional synthetic benchmark in the existing conda environment takes no
research seed or convergence run:

```bat
python experiments\resnet14_codec_benchmark.py --repeats 3 --output experiments/results/resnet14_codec_speed_v1/windows_local_cpu.json
git add experiments/results/resnet14_codec_speed_v1/windows_local_cpu.json
git commit -m "Add local CPU codec speed benchmark"
git push origin main
```

This measurement is optional; it does not select configurations or control
the confirmation. The Linux benchmark file is preserved separately.
