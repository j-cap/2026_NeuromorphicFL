# Calibrated panel freeze and manuscript consolidation

Historical two-CIFAR consolidation. The completed third-architecture
integration and current commands are in THREE_ARCHITECTURE_CONSOLIDATION.md.
The two-CIFAR numerical freeze below remains unchanged.

Frozen on 2026-11-26. Research source before this edit: `14216ff`.
The agreed five methods and all completed CIFAR runs are closed for this panel.
No additional fitting or tuning is required to reproduce the manuscript update.

## Scientific position

Event is the communication winner at the selected quality-calibrated operating
points: it uses the least final total unicast traffic on both architectures and
less paired geometric-mean traffic to each declared target than all four
comparators. At 50% test accuracy:

| Reference | Compact CNN saving, % [95% interval] | ResNet-14 saving, % [95% interval] |
|---|---:|---:|
| Ström-style | 29.2 [21.0, 36.6] | 51.4 [48.7, 54.0] |
| STC | 57.8 [52.6, 62.3] | 23.4 [20.3, 26.4] |
| EF-TopK | 95.8 [95.4, 96.1] | 65.4 [62.5, 68.1] |
| Dense FedAvg | 99.1 [99.0, 99.2] | 98.3 [98.2, 98.4] |

Every method reaches this target in every run. Intervals are descriptive,
unadjusted paired Student-t intervals on log cost ratios. Final quality is
reported alongside cost: Event has the highest compact-CNN mean accuracy;
ResNet EF-TopK and STC finish at higher mean accuracy at greater cost. A
zero-containing accuracy interval does not establish equivalence.

Every method is calibrated for the dataset/model. ResNet uses equal accumulated
screen/continuation budgets and validation CE. The compact-CNN archived tuning
rules differ. The text says this explicitly and retains the sparse-STC
validation-frontier qualification.

## Manuscript and figure changes

- Abstract/introduction now lead with calibrated communication to predictive
  quality, conservative unicast, and the completed two-architecture evidence.
- Table I contains the actual four numerical comparators, their residuals,
  server semantics, and the explicit Ström-to-FedAvg adaptation. LENA/SPARQ
  remain structural context.
- Upload support encoding, aggregate-first downlink, exact float32 fidelity
  patches, and actual frame/header/padding accounting replace the stale replay-only
  description. The event-budget accounting upper bound is corrected.
- New vector figures show the current protocol, accuracy against total traffic,
  and paired cost ratios at every declared target. New endpoint and target
  tables include STC and predictive CE.
- The reviewed mechanism/alignment mathematics and earlier partial-participation
  qualifications remain; old Fashion-MNIST/MLP results are supporting historical
  evidence until their separate calibration stage.
- New changes are blue. Set `\coreColor` to empty in `main.tex` for a clean copy.
  Original campaign outputs, study drivers and the implementation package remain
  unchanged.

## Rebuild locally (no training)

In Windows Command Prompt, from the research repository:

```bat
cd C:\Users\WeberJ\2026_NeuromorphicFL
conda activate neuromorphicfl-resnet14
git pull --ff-only
python paper/ijcnn2027/build_calibrated_core.py
python paper/ijcnn2027/build_calibrated_core.py --check
python paper/ijcnn2027/check_manuscript_claims.py
```

The script uses NumPy/Matplotlib and reads committed CSVs. It does not load
CIFAR images, import Torch, retrain, alter trajectories, or change result files.
Outputs are three vector PDFs, two LaTeX tables, and four evidence CSVs.
They are already generated in this commit. Repeating the build in the pinned
artifact environment should leave no changes. If deliberately regenerating
new figure products after editing the builder, include the results commit:

```bat
git add paper/ijcnn2027/build_calibrated_core.py paper/ijcnn2027/figures/calibrated_*.pdf paper/ijcnn2027/generated/calibrated_*.tex paper/ijcnn2027/evidence/calibrated_core_*.csv
git diff --cached --check
git commit -m "Regenerate calibrated Event-FedAvg manuscript figures"
git push origin main
```

Changes in plot-generation dependencies can change PDF bytes; the checked
artifact dependencies are pinned in `requirements-reproduction.txt`. The
training pins in `requirements-evidence.txt` remain unchanged. This consolidation
was checked under Python 3.12.14, NumPy 2.3.5, Matplotlib 3.10.8, and pypdf 6.21.4;
the existing strict CI profile uses Python 3.11 and its pinned pypdf version. Full document checks use `reproduce.py`, including
legacy historical product checks; they do not rerun training.

## Next stage

Prepare the remaining MLP/model calibration after reviewing this consolidated
manuscript. Use a separate bounded development protocol for every numerical
method, freeze the validation criterion and budget before confirmation, and
retain unicast cost, test CE, final accuracy and predeclared targets. Do not
retune or reopen these completed CIFAR panels as part of that stage.
