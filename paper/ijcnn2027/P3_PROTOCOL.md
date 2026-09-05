# P3 CIFAR-10 protocol freeze

P3 tests whether the Event-FedAvg communication--performance result transfers
beyond Fashion-MNIST. This protocol was fixed before any CIFAR-10 result was
inspected.

## Dataset and partition

- CIFAR-10 Python archive from the official Toronto source.
- Archive MD5: `c58f30108f718f92721af3b95e74349a`.
- Ten equal clients with 5,000 training examples each.
- Strong label skew: client `i` receives 2,750 examples of class `i` and 250
  examples of every other class. Each class is used exactly once globally.
- Development partition seed: `3400`.
- Untouched held-out partition seeds: `3500`, `3600`, and `3700`.
- Training seed for partition `p`: `80000 + p`.
- All clients participate in every round. There is no asynchrony or partial
  participation claim.

## Compact CNN

The predictive model is conventional, not spiking:

1. `3 x 32 x 32` normalized RGB input;
2. 8-channel `5 x 5` convolution, stride 2, ReLU;
3. 16-channel `3 x 3` convolution, stride 2, ReLU;
4. 32-unit fully connected layer with ReLU;
5. 10-class linear output.

The model has 20,570 trainable parameters. No augmentation, pretrained model,
batch normalization, or test-time transformation is used. This keeps the
campaign deterministic and isolates the communication operator.

## Shared training protocol

- 120 synchronous rounds;
- five local SGD steps per client and round;
- minibatch size 32;
- local learning rate 0.05;
- L2 regularization `5e-4`;
- identical initialization, local minibatch stream, and local learning across
  methods for a given seed;
- full test set for final cross-entropy, accuracy, macro accuracy, and
  worst-class accuracy.

## Methods and development grids

- Dense FedAvg.
- Sign-EF.
- EF-TopK fractions: `0.005`, `0.01`, `0.025`, `0.05`.
- Strom thresholds: `0.0025`, `0.005`, `0.01`, `0.02`, `0.04`.
- Event-FedAvg candidates use `rho=0.999` and
  `q_r=q_0(1+r/100)^(-0.2)`:
  - `(threshold, q0) = (0.0125, 0.0025)`;
  - `(0.025, 0.0025)`;
  - `(0.025, 0.005)`;
  - `(0.05, 0.005)`;
  - `(0.05, 0.01)`.

Each family selects the smallest final training cross-entropy on the
development partition. Held-out results are never used for tuning. For Strom
and EF-TopK, a second development point is selected by minimum log-distance to
the total traffic of the quality-selected Event-FedAvg point.

## Communication accounting

The campaign reuses the frozen symmetric accounting model:

- packet headers and coordinate addresses are charged;
- uplink and downlink are both counted;
- the server update stream is replayed exactly when cheaper than a dense
  float32 checkpoint;
- conservative unicast downlink includes a 32-bit catch-up request per client
  and round;
- the initial dense model synchronization is charged.

The primary communication metric is total bidirectional unicast traffic.

## Execution and gate

The GitHub workflow runs, in order:

1. synthetic gradient/determinism/accounting smoke checks;
2. dense FedAvg development validation;
3. development-only tuning;
4. immutable selection;
5. three held-out quality-selected runs per method;
6. three held-out traffic-matched runs for Strom and EF-TopK;
7. aggregation, frontier rendering, and preliminary classification.

The result is classified according to `PLAN.md`: pass, qualified pass, or
fail. The report and paper evidence are updated only after the combined
artifact has been inspected.
