# Event-FedAvg manuscript contract — consolidated 2026-11-26

The completed compact-CNN and ResNet-14 CIFAR-10 five-method panels are frozen.
The research snapshot before consolidation is `14216ff50455774fce31386621f344fc3f07d1b8`.
The reviewed manuscript base is paper-repository main `310a0805d133ab2f0873a36f9f08fb6e8036887c`.
The later paper-source M1, M3, P11 and participation clarifications are preserved.

The central question is whether persistent leaky coordinate evidence, full reset,
independent model quanta and exact aggregate transport yield a favorable
bidirectional communication cost for predictive quality in synchronous FedAvg.

The primary evidence is two CIFAR-10 architectures, five fixed operating points,
ten paired seeds per architecture, 1,800 rounds. The methods are Event-FedAvg,
Dense FedAvg, STC, Ström-style residual pulses in FedAvg, and EF-TopK. There is
no further training, tuning, additional STC selection, or method replacement in
this panel. Remaining MLP tuning is the separate next stage.

All methods use architecture-specific development-calibrated settings. ResNet
uses equal accumulated budgets (18 x 600-round screens, three 1,200-round
continuations per family), with validation-CE selection. Compact-CNN historical
budgets and criteria differ and must be identified accurately. Neither panel
establishes global hyperparameter optimality or a full method frontier.

The primary result is lower paired geometric-mean unicast cost to the declared
45%, 50%, and 55% accuracy targets versus all four references. Compact-CNN Dense
at 55% has only nine common hits. All other target contrasts have ten. Event
also has the lowest final total traffic at both selected panels, the largest
compact-CNN mean accuracy, and a clear ResNet improvement over residual pulses.
EF-TopK and STC have higher mean final ResNet accuracy at higher traffic.

The paper must not claim universal accuracy dominance, final-accuracy equivalence,
a general faster-convergence rate, unconditional convergence, measured energy or
latency savings, or a field-wide SOTA ranking. Sparse STC's competitive development
points remain a visible limitation of quality-point selection.

The state/event-budget proof and conditional alignment result remain. The event
uplink bound is updated for 128-bit frames plus padding. The mechanism audit and
50-client partial-participation study retain their original settings/accounting
and are supporting evidence, not newly calibrated five-method panels. Earlier
Fashion-MNIST tables are historical pending the separate calibration stage.

The paper can use up to ten pages under the user's agreed page budget. Use the
standard IEEE conference class and bibliography size. New consolidation edits
are blue through `\coreColor`; earlier review wrappers are black. Do not shrink
the bibliography to fit. Rebuild plots/tables only from committed evidence;
never modify raw campaign outputs or the frozen `src/neuromorphicfl` inventory.
