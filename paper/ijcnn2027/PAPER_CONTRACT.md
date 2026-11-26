# Event-FedAvg manuscript contract - completed 2026-11-26

The compact-CNN and ResNet-14 CIFAR-10 panels and Fashion-MNIST MLP panel are
closed: three architectures, two datasets, five selected operating points,
ten paired seeds per architecture. Research input is commit 63bc32b; MLP results
are b2e2ce8. The reviewed paper base is 757cfe9. No further fitting, tuning or
method replacement is part of this matrix. Original CIFAR freezes remain intact.

The central question is whether persistent leaky coordinate evidence, full reset,
independent model quanta and exact aggregate transport yield favorable
bidirectional communication cost for predictive quality in synchronous FedAvg.
The numerical methods are Event-FedAvg, Dense FedAvg, STC, Ström-style residual
pulses in FedAvg, and EF-TopK. CIFAR panels use 1,800 rounds; the MLP uses 1,500.

Every family uses dataset/model-specific development calibration. ResNet uses
18 x 600-round screens and three 1,200-round continuations per family; MLP uses
18 x 500-round screens and three 1,000-round continuations. Both use validation
CE for promotion/selection, with traffic and identifier as tie breakers.
Compact-CNN historical budgets/criteria differ. No global hyperparameter
optimality or complete method frontier is established.

Event has the lowest final total unicast traffic at every panel and in every
paired comparison against every reference. This final-cost lead is distinct
from quality: compact CNN has the highest mean accuracy, ResNet improves on
residual pulses but selected STC/EF finish at higher mean accuracy, and all MLP
paired final accuracy intervals span zero. Equivalence is not established.

For CIFAR, Event has lower paired cost to all declared 45/50/55% targets;
compact Dense at 55% has nine common hits. The MLP has declared 80/85/87%
targets. At 85%, Event saves 17.6% versus STC but spends 10.0% more than Ström.
At 80%, the STC/Ström cost intervals span no saving; at 87%, ratios are
conditional on common hits, with censoring and transient crossings explicit.
Never extend the all-target CIFAR claim to all three architectures.

Do not claim universal predictive dominance, accuracy equivalence, a general
faster-convergence rate, unconditional convergence, measured energy or latency
savings, or a field-wide SOTA ranking. Competitive sparse STC development points
remain a visible limitation of selecting one validation-CE operating point.

The state/event-budget proof and conditional alignment result remain. The event
uplink bound includes 128-bit frames plus padding. Mechanism/alignment and
partial-participation studies retain original settings/accounting as supporting
evidence; their conclusions are not transferred to the new calibrated matrix.

Use at most ten pages with standard IEEE class and bibliography size. New MLP
integration blocks are orange through the mlpColor/mlpRev commands; earlier
blue consolidation edits remain. Set both color commands to empty for a clean
copy. Rebuild plots/tables only from committed CSVs. Never modify campaign
outputs, training drivers or the frozen src/neuromorphicfl inventory.

See THREE_ARCHITECTURE_CONSOLIDATION.md for the input freeze, regeneration,
commit/push and paper synchronization commands.
