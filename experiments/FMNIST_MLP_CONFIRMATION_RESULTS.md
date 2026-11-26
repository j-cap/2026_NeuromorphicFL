# Fashion-MNIST MLP: complete common-seed confirmation

Evidence: `experiments/results/fmnist_mlp_calibrated_v1/`, user-results commit
`b2e2ce8226cacefa8c36855f474f0769e87b20ad`. Calibration analysis:
[`FMNIST_MLP_DEVELOPMENT_RESULTS.md`](FMNIST_MLP_DEVELOPMENT_RESULTS.md).

## Scientific conclusion and decision

**The MLP confirms Event's lower final unicast communication at competitive
predictive quality among the five frozen operating points.** Event has the
lowest final total traffic in every paired comparison against every reference.
Its final saving is **37.58% [37.23%, 37.92%] versus STC**, **14.51% [14.20%,
14.82%] versus Ström**, **94.89% versus EF-TopK**, and **98.00% versus Dense**.
All final paired accuracy intervals span zero; formal equivalence or
noninferiority is not established.

**Communication to an accuracy target depends on the target and comparator.**
At 85%, Event saves **17.57% [11.95%, 22.84%] versus STC**, but uses **10.03%
[0.66%, 20.27%] more traffic than Ström**. Ström reaches this target earlier
and with fewer bits in nine pairs. The MLP therefore qualifies any claim that
Event always reaches a given quality with less communication or fewer rounds.

**Freeze this completed panel and integrate it into the manuscript.** It closes
the agreed third-architecture comparison. The next work is manuscript and
figure consolidation with the Ström target exception visible. No additional
training is needed for the agreed five-method, ten-seed matrix.

## Fixed protocol and audit coverage

The model is the 25,818-parameter tanh MLP, 784–32–16–10, on Fashion-MNIST.
Ten clients participate in every synchronous round; each has 6,000 examples,
with 55% from its dominant label and 5% from each other label. Every method
shares full-training data, initialization seed 7777, five local SGD steps,
batch 32, learning rate 0.1, L2 coefficient 0.0001 and fixed 1,500-round horizon.
The official test set contains 10,000 images. Evaluation is after round one
and every fifty rounds, with a separate 5,000-image training diagnostic subset.

All **50 new confirmation fits are complete and finite**: five methods times
ten common partition seeds 11100–12000, step 100, with matched SGD seeds
101100–102000. There is no development-checkpoint transfer into confirmation.
The source/runtime contract and all selections are unchanged from calibration.

Every family used 18 development candidates screened for 500 rounds and three
exact continuations to 1,500 rounds: at most 12,000 unique tuning rounds per
family. All 105 development stages completed. Promotion and final selection
minimize validation CE on the 54,000/6,000 training/validation split, with total
unicast traffic and identifier as tie breakers. Search geometry differs by
family; this is a common bounded budget, not proof of global hyperparameter
optimality or a complete method frontier.

Frozen settings are Event threshold 0.025/initial quantum 0.005/exponent 0.1,
leak 0.999 and full reset; Ström threshold 0.02/native gain 0.3; Dense initial
gain two/exponent 0.3; STC upload/download retention one/gain two; EF-TopK global
retention 10%/gain two. Schedule scale is 100. Ström is the residual-pulse
encoder adapted to the common FedAvg host. STC p=1 retains ternarization and
both residuals and is not full-precision Dense FedAvg.

The audit verifies **47 source hashes, 50 complete receipts, 75,000 raw round
records, 1,550 sampled model fingerprints and 50 distinct final model hashes**.
It checks full-training/test manifests, all recorded model identities, frame
alignment and portfolio constraints, and cumulative accounting at each saved
checkpoint. The nine original numerical CSV/JSON products are reproduced in a
temporary copy. Independent Python scalar calculations check all endpoint
means/SDs, paired effects, first sampled target crossings and denominators.
The **809 original campaign files remain unchanged**. Images, model arrays and
training checkpoints are not loaded or rerun by this audit; encoding/decoding
and reconstruction checks were performed by the frozen runners and their
pre-execution synthetic verification.

## Final predictive quality and total communication

Accuracy and unregularized test CE are arithmetic mean ± sample SD across ten
paired seeds. Traffic columns are arithmetic means. **1 Gbit = 10^9 bits.**
Primary total unicast includes actual serialized upload/server frames and
padding, initialization, per-client requests and recipient replication.
The synchronization/control column uses the existing `unicast_downlink_bits`
convention, including initialization and request costs.

| Method | Test accuracy, % | Test CE | Uplink, Gbit | Synchronization/control, Gbit | Total unicast, Gbit |
|---|---:|---:|---:|---:|---:|
| Event | 86.939 ± 0.183 | 0.369931 ± 0.004464 | 0.069993 | 0.426658 | **0.496651** |
| Ström-style | 86.901 ± 0.238 | 0.377107 ± 0.006796 | 0.087003 | 0.493921 | 0.580924 |
| Dense | 86.935 ± 0.213 | 0.368068 ± 0.003970 | 12.393600 | 12.402342 | 24.795942 |
| STC p=1 | 86.981 ± 0.190 | 0.369574 ± 0.004755 | 0.393533 | 0.402043 | 0.795576 |
| EF-TopK 10% | 86.844 ± 0.273 | 0.374542 ± 0.005942 | 1.466166 | 8.249944 | 9.716110 |

STC has the highest mean accuracy; Dense has the lowest mean CE. Their small
quality differences from Event remain unresolved under the paired intervals.
Event's smallest traffic is not a claim of predictive superiority.
Worst-class mean accuracy is 64.90% for Event, 64.36% for Ström, 64.01% for
Dense, 65.32% for STC and 64.74% for EF-TopK; this remains a secondary diagnostic.

## Paired final effects

Quality differences are Event minus reference. Communication savings are
`100*(1-geometric_mean(Event/reference))`. Intervals are descriptive,
unadjusted paired 95% Student-t intervals; cost ratios use the log scale.
They describe partition/SGD variation at fixed initialization and frozen
parameters, not variation across independent datasets or tuning splits.

| Reference | Accuracy difference, pp [95% interval] | CE difference [95% interval] | Total unicast saving, % [95% interval] |
|---|---:|---:|---:|
| Ström-style | +0.038 [-0.120, +0.196] | -0.007176 [-0.012942, -0.001411] | 14.51 [14.20, 14.82] |
| Dense | +0.004 [-0.178, +0.186] | +0.001862 [-0.002335, +0.006060] | 98.00 [97.99, 98.01] |
| STC | -0.042 [-0.206, +0.122] | +0.000356 [-0.004572, +0.005284] | 37.58 [37.23, 37.92] |
| EF-TopK | +0.095 [-0.079, +0.269] | -0.004612 [-0.007670, -0.001553] | 94.89 [94.86, 94.92] |

Event has fewer final total bits in **10/10 pairs versus every comparator**.
It has higher accuracy in 7/10 pairs versus Ström, 7/10 versus Dense, 5/10
versus STC and 6/10 versus EF-TopK. It has lower CE in 9/10, 4/10, 6/10 and
8/10, respectively. The CE intervals against Ström and EF-TopK are below
zero before multiplicity adjustment; they do not establish a universal quality
ranking. The accuracy intervals cannot be interpreted as equivalence tests.

Against STC, Event saves **82.21%** of upload bits but uses **6.12% more
synchronization/control bits**. Its total saving is therefore 37.58%, with a
directional tradeoff. Logical-broadcast accounting gives a larger **74.03%**
final total saving versus STC, but is supplementary. The primary conclusion
uses replicated unicast and does not depend on free broadcast.

## Communication to the three predeclared targets

All methods reach 80% and 85% on every seed. At 87%, target coverage is
incomplete. The rule is the first saved per-run crossing, with no interpolation
or requirement to remain above the target. All fits continue to 1,500 rounds.
The table reports Event's paired geometric-mean saving; a negative value means
Event spends more than the reference.

| Reference | To 80%, saving % [95% interval] | To 85%, saving % [95% interval] | To 87%, saving % [95% interval] | Common 87% hits |
|---|---:|---:|---:|---:|
| Ström-style | 6.34 [-5.74, 17.03] | **-10.03 [-20.27, -0.66]** | 3.22 [-21.05, 22.62] | 7/10 |
| Dense | 96.76 [95.89, 97.45] | 97.63 [97.50, 97.75] | 98.05 [97.71, 98.35] | 8/10 |
| STC | 14.71 [-3.68, 29.83] | **17.57 [11.95, 22.84]** | 25.22 [11.17, 37.04] | 9/10 |
| EF-TopK | 89.90 [87.00, 92.15] | 93.30 [92.72, 93.84] | 94.51 [92.83, 95.80] | 6/10 |

At 85%, all ten seeds contribute to every comparison. The arithmetic mean
first crossing/cost, distinct from the paired geometric ratio above, is:

| Method | Mean first sampled round | Mean total unicast to 85%, Mbit |
|---|---:|---:|
| Event | 330 | 121.69 |
| Ström-style | 255 | 110.49 |
| Dense | 310 | 5,131.05 |
| STC | 265 | 147.30 |
| EF-TopK | 280 | 1,814.35 |

Event reaches 85% later than Ström and STC in nine pairs each, tying the
remaining pair. Its cheaper updates offset this delay against STC, yielding
lower cost in all ten pairs, but not against Ström, which uses fewer bits in
nine pairs. There is no general faster-round-convergence result on the MLP.
At 80%, Event's cost intervals versus Ström and STC include no saving; those
point estimates should not be presented as resolved advantages.

The 87% contrast is conditional on both methods reaching the target:

| Method | Ever reaches saved 87% | Missed partition seeds | Ends at or above 87% |
|---|---:|---|---:|
| Event | 9/10 | 11700 | 3/10 |
| Ström-style | 8/10 | 11300, 11900 | 2/10 |
| Dense | 9/10 | 11400 | 4/10 |
| STC | 9/10 | 11700 | 4/10 |
| EF-TopK | 7/10 | 11500, 11800, 12000 | 4/10 |

Thus some high-target crossings are transient for every family. A missed
crossing is censored rather than assigned zero cost. The conditional 87%
ratios do not summarize all ten seeds or guarantee sustained/final 87%
accuracy. Fixed-horizon accuracy/CE remain visible alongside them.

## Consequence for the manuscript

The agreed matrix is complete: compact CNN and ResNet-14 on CIFAR-10, plus
MLP on Fashion-MNIST. Event has the lowest final unicast cost at the evaluated
quality-selected operating points in all three panels. This supports a
communication-efficiency contribution across architectures and datasets,
with predictive tradeoffs reported explicitly. ResNet's STC/EF accuracy
tradeoffs and the compact CNN's different historical calibration rules remain
part of the evidence boundary.

The new MLP must not extend the two-CIFAR claim of lower cost at every declared
target to all three architectures. Its 85% Ström comparison is a concrete
exception. Do not retune against these confirmation seeds to erase it.
The complete method-frontier, unconditional convergence, network latency and
energy claims remain outside these experiments.

A suitable integration paragraph is:

> On Fashion-MNIST with a fully connected MLP, Event-FedAvg attains
> 86.94 ± 0.18% accuracy with 0.497 Gbit of total unicast traffic, compared with
> 86.98 ± 0.19% at 0.796 Gbit for the validation-selected STC configuration.
> The paired final traffic reduction is 37.6% [37.2%, 37.9%]; the accuracy
> interval spans zero and does not establish equivalence. Event also reduces
> traffic to the declared 85% accuracy target by 17.6% [11.9%, 22.8%] versus
> STC, while requiring 10.0% [0.7%, 20.3%] more than the calibrated residual-pulse
> comparator. The advantage is a favorable communication/quality operating
> point rather than a uniform reduction in convergence rounds or target cost.

Next: integrate the MLP endpoint and all three target contrasts into the main
comparison figures/tables, revise abstract and conclusion around the scoped
three-architecture claim, and preserve the original two-CIFAR numerical freezes.

## Recheck without training

The audit is separate from the frozen training sources and writes only new
`analysis_confirmation_` products. It reproduces the original analysis in a
temporary copy and leaves all original files untouched.

```bat
python experiments/fmnist_mlp_confirmation_audit.py --check
```

To deliberately regenerate the derivative audit files and commit them:

```bat
python experiments/fmnist_mlp_confirmation_audit.py
git add experiments/results/fmnist_mlp_calibrated_v1/analysis_confirmation_*.csv experiments/results/fmnist_mlp_calibrated_v1/analysis_confirmation_audit.json
git diff --cached --check
git commit -m "Refresh independent Fashion-MNIST MLP confirmation audit"
git push origin main
```
