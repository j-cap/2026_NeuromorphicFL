# ResNet-14 boundary check: results and decision

Evidence: `experiments/results/resnet14_boundary_v1/`, results commit
`dbb354b8541a57ad810116d0597b4972a268214b`. Protocol:
[`RESNET14_BOUNDARY.md`](RESNET14_BOUNDARY.md). The six 600-round screens and
both promoted 1800-round finalists are complete, with no numerical failures
or omitted candidates.

## Decision

**Keep both original configurations and stop this bounded tuning study.**
Neither promoted boundary extension improves its incumbent. At round 1800,
the new Event finalist has worse validation CE, 0.86 percentage points lower
validation accuracy and **48.14% more unicast traffic**. The new Ström finalist
has worse validation CE, 0.62 percentage points lower validation accuracy and
**32.25% more traffic**. There is no quality/traffic tradeoff favouring either
promoted finalist on these endpoints.

The frozen choices remain:

| Method | Threshold | Quantum / gain | Other fixed method settings |
|---|---:|---|---|
| Event | 0.01 | Initial quantum 0.0025; server gain 1 | Leak 0.999, persistent state, full reset, quantum exponent 0.2 |
| Ström | 0.005 | FedAvg-equivalent gain 2; native server gain 0.10 | Nonleaky residual, subtractive reset, one pulse per coordinate/client/round |

This answers the specific concern that a small extension below the selected
grid boundaries might change the validation choices. It does not establish
a global optimum or exhaust all threshold/gain interactions. No further
extension is justified by this completed, deliberately bounded check.

## Matched validation protocol and endpoints

The check uses the same Torch ResNet-14/GroupNorm (175258 parameters), ten
fully participating clients, strong 55%-dominant skew, initialization seed
7777, development partition seed 7000 and SGD seed 97000 as its parent. The
same 45000 training / 5000 validation split, five local SGD steps, batch size
32, learning rate 0.05, L2 coefficient 0.0005 and shared lossless pulse and
exact aggregate downlink codecs remain fixed.

Each family receives three new 600-round candidates and one exact continuation
to 1800 rounds. Promotion and final selection use validation CE, then total
unicast bits and point ID for ties. The original 1800-round development winners
are reused as read-only incumbents. No official-test images or saved
official-test metrics enter this boundary check or its audit.

All numbers below are **validation results on one development partition**,
not test means or additional confirmation seeds. Costs use conservative
unicast accounting, including complete frames/padding, initialization,
client requests and recipient replication; 1 Gbit = 10^9 bits.

| Method and configuration | Validation CE | Validation accuracy | Uplink, Gbit | Downlink, Gbit | Total unicast, Gbit |
|---|---:|---:|---:|---:|---:|
| Event incumbent: threshold 0.01, quantum 0.0025 | **0.717497** | **75.78%** | 0.951 | 4.837 | **5.788** |
| Event new finalist: threshold 0.005, quantum 0.00125 | 0.751896 | 74.92% | 1.716 | 6.859 | 8.574 |
| Ström incumbent: threshold 0.005, equivalent gain 2 | **0.788567** | **72.50%** | 1.816 | 6.467 | **8.283** |
| Ström new finalist: threshold 0.0025, equivalent gain 2 | 0.813102 | 71.88% | 3.034 | 7.920 | 10.954 |

| New finalist minus its incumbent | Event | Ström |
|---|---:|---:|
| Validation CE difference | +0.034399 | +0.024535 |
| Validation accuracy difference, pp | -0.86 | -0.62 |
| Total unicast increase | +48.14% | +32.25% |

Both upload and download increase for both promoted extensions. The lower
training CE of the incumbents (Event 0.418892 versus 0.450074; Ström 0.613187
versus 0.678815) is consistent with the validation ranking. There is no
statistical interval across independent seeds for this one-partition check.

## Screening results and parameter interactions

| New candidate | Validation CE at 600 | Validation accuracy | Unicast Gbit | Promoted |
|---|---:|---:|---:|---|
| Event threshold 0.005, quantum 0.00125 | 1.072459 | 61.28% | 2.676 | Yes |
| Event threshold 0.01, quantum 0.00125 | 1.270305 | 54.08% | 1.888 | No |
| Event threshold 0.0125, quantum 0.00125 | 1.343838 | 51.06% | 1.671 | No |
| Ström threshold 0.0025, equivalent gain 2 | 1.235946 | 55.04% | 3.555 | Yes |
| Ström threshold 0.005, equivalent gain 1 | 1.431272 | 47.74% | 2.714 | No |
| Ström threshold 0.0025, equivalent gain 1 | 1.448139 | 47.10% | 3.549 | No |

The Event result is not evidence that smaller quantum is always worse. At
threshold 0.005, halving quantum from 0.0025 to 0.00125 improves the 600-round
validation CE from 1.496816 to 1.072459, while increasing traffic by 7.48%.
At threshold 0.01, the same quantum change worsens CE from 1.064054 to
1.270305 and increases traffic by 2.81%. At threshold 0.0125, CE worsens from
1.112873 to 1.343838 with 1.63% more traffic. Threshold and quantum interact;
the new small-threshold/small-quantum finalist still loses to the incumbent
at the declared final horizon.

For Ström at threshold 0.005, halving equivalent gain from 2 to 1 worsens
600-round CE from 1.213382 to 1.431272, with essentially unchanged traffic
(+0.51%). At equivalent gain 2, halving threshold from 0.005 to 0.0025 gives
CE 1.235946 instead of 1.213382 and 31.67% more traffic at round 600. The
continued lower-threshold candidate remains worse at round 1800.

Smaller pulse amplitude does not by itself imply fewer encoded bits. The
amplitude is shared metadata, while positions, signs and aggregate messages
depend on the firing pattern and resulting optimization trajectory. The
promoted Event extension changes both threshold and quantum; its final traffic
increase cannot be assigned to quantum alone. The study does not isolate a
causal role for leak, reset or quantum scheduling.

## Consequence for confirmation and the paper

The two choices are unchanged, so the completed three-pair full-training
ResNet confirmation still evaluates these same frozen configurations. Its
reported test result remains **75.80% versus 72.26% mean accuracy**, with a
**31.04% paired final unicast saving** and **53.24% saving to the declared 50%
accuracy target**; see [`RESNET14_TRANSFER_RESULTS.md`](RESNET14_TRANSFER_RESULTS.md).
Those predictive numbers are earlier evidence, not new results of this
validation check. The check adds **zero new test-seed pairs** and does not
shrink the uncertainty of the three-pair estimate.

The next useful computation is to freeze these settings and extend the ResNet
confirmation from **three to ten matched pairs**, retaining the three completed
pairs and adding seven fresh pairs: **fourteen new 1800-round runs, or 25200
additional rounds**. Report all ten pairs and distinguish the seven prospective
new pairs. No development checkpoint should seed the full-training runs.

The other ResNet baselines still require matching calibration/evaluation before
claiming a five-method ResNet ranking. The completed compact-CNN five-method
comparison remains separate. This boundary study supports a bounded calibration
decision, not field-wide SOTA, universal hyperparameter transfer or an
architecture-dependent convergence theorem.

## Evidence audit and portability

The supplementary audit verifies eight complete new stage receipts, both reused
development incumbents, **7200 archived new round rows**, **128 sampled model
fingerprints**, four distinct final development fingerprints and **two exact
600-to-1800 prefixes**. The duplicate prefixes account for the difference from
the actual new work: **6000 unique training rounds and 106 unique evaluation
points**. It reproduces the four original analysis CSVs, independently rebuilds
the validation rankings and all four traffic variants' saved prefixes/endpoints,
and verifies the original promotion, selection and completion audit.

The original Windows driver records parent-dependency dictionary keys using
Windows separators. Its strict Linux reader therefore rejects that spelling
even though every dependency path/hash and other contract field agrees. The
separate audit normalizes only those key separators for comparison, rejects
duplicate canonical names and other source/data/runtime/configuration changes,
and retains the original recorded contract and receipt digests. No frozen
driver, result, hash identity or selection is rewritten. The four portability
regression tests pass, and the real saved-evidence audit passes.

The audit reads saved metrics, counters and hashes, not image arrays, packet
bytes or NPZ model/state arrays, and does not replay training. Runtime native
codec decoding and model reconstruction belong to the frozen training runner.
The eight original protocol/selection/analysis files and all forty new stage
files remain unchanged. The audit writes only supplementary `analysis_*` files.

Reproduce the audit without training on either Windows or Linux:

```bat
python experiments/resnet14_boundary_audit.py
git add experiments/results/resnet14_boundary_v1
git commit -m "Refresh ResNet14 boundary evidence audit"
git push origin main
```
