# P5 visual-argument audit

## Outcome

P5 is closed. The main-paper visual package contains exactly three principal
elements: a method schematic, a cross-benchmark communication--performance
frontier, and one compact headline table with the minimum alignment diagnostic.
All quantitative content is generated from frozen evidence rather than copied
into the manuscript by hand.

## Visual-to-claim map

| Element | Claim carried | Source | Deliberate boundary |
|---|---|---|---|
| `figures/event_fedavg_method.pdf` | C1 and C3: complete stateful operator and exact synchronization | Frozen transition in `main.tex` and implementation contract checked in P4 | Diagram does not imply asynchronous or partial-participation support. |
| `figures/communication_frontier.pdf` | C4 and C9: communication--performance operating point on three benchmark/model settings | `evidence/fmnist_master_results.csv` and `evidence/cifar10_master_results.csv` | Uses total conservative bidirectional unicast traffic; no energy or latency axis. |
| `generated/main_results_table.tex` | C4, C7, and C9: strongest-quality and nearest-traffic comparisons plus finite-trajectory alignment evidence | The three paper evidence CSVs, with the alignment aggregate traced to the authoritative T4 Actions run | Three seeds are shown as mean $\pm$ sample standard deviation, not significance claims. The alignment audit does not prove the conditional assumption. |

## Selection rules

The figure includes every frozen quality-selected point and every available
traffic-matched control. Event-FedAvg appears once because its row is identical
in the two selection views. Filled markers mean quality-selected; hollow
markers mean traffic-matched. Method identity is also encoded by marker shape,
so interpretation does not depend on color.

For each benchmark/model setting, the compact table selects:

1. the frozen Event-FedAvg operating point;
2. the non-Event quality-selected method with the highest held-out mean
   accuracy; and
3. the non-Event traffic-matched method whose total traffic is closest to
   Event-FedAvg on a log-ratio scale.

These deterministic rules produce EF-TopK/Strom for Fashion-MNIST MLP,
Strom/Strom for Fashion-MNIST CNN, and Strom/Strom for CIFAR-10. The repeated
Strom rows represent different development-selected thresholds, not duplicated
measurements.

## Scientific reading

- Event-FedAvg is nondominated in all three panels.
- It exceeds the strongest quality-selected control in mean accuracy on
  Fashion-MNIST MLP and CIFAR-10 while using substantially less total traffic.
- Fashion-MNIST CNN is the visible qualification: quality-selected Strom gains
  0.90 accuracy points but uses 4.9 times more total traffic.
- At nearby traffic, Event-FedAvg has higher mean accuracy than the selected
  Strom control in all three settings.
- The table retains the exact-gradient alignment audit only as finite-trajectory
  support for the conditional theory interface.

## Readability and reproducibility checks

`build_visuals.py` generates deterministic vector PDFs with embedded TrueType
fonts and a source-controlled LaTeX fragment. Its `--check` mode validates the
frozen row design, distinct grayscale marker shapes, Event-FedAvg's
nondominated status, selection rules, and byte-for-byte rendered products.

The figures target IEEE two-column width. The method diagram is grayscale by
construction. The frontier remains interpretable in grayscale through distinct
markers, filled versus hollow selection encoding, black edges, and a dotted
nondominated trace. Captions state the scientific finding and expose the CNN
qualification. Full campaign tables, T4 decomposition plots, schedule plots,
and exploratory report figures remain outside the six-page visual budget.

Rebuild and check with:

```bash
python paper/ijcnn2027/build_evidence.py --check
python paper/ijcnn2027/build_visuals.py
python paper/ijcnn2027/build_visuals.py --check
```
