# Completed three-architecture manuscript integration

Frozen on 2026-11-26 from research commit 63bc32b8406f370fd476259514f4dce7bc909d0b.
The new MLP results are in commit b2e2ce8226cacefa8c36855f474f0769e87b20ad.
The authoritative manuscript base is paper main 757cfe9634f024274de2d6a114fbe34d72085a40.
The complete matrix is closed: three architectures, two datasets, five methods
and ten paired seeds per architecture. No further training or tuning is needed.

## Claim and evidence boundary

Event uses the lowest final total unicast traffic in all ten paired comparisons
against every reference in every panel. Each family uses dataset/model-specific
development calibration. ResNet and MLP have equal bounded budgets within each
panel; the compact CNN preserves its historical, different selection rules.
These are selected operating points, not complete method frontiers.

The Fashion-MNIST MLP ends at 86.94 +/- 0.18% accuracy and 0.497 Gbit.
Its paired final savings are 37.6% [37.2,37.9] versus STC and 14.5%
[14.2,14.8] versus Ström. All four final paired accuracy intervals span zero;
equivalence is not established.

At the predeclared 85% target, Event saves 17.6% [11.9,22.8] versus STC
but uses 10.0% [0.7,20.3] more than Ström. At 80%, cost intervals against
STC and Ström span no saving. At 87%, common-hit denominators are 7/9/6/8
against Ström/STC/EF-TopK/Dense; crossings can be transient for every family.
The manuscript emphasizes final communication efficiency, with target-specific
exceptions and the higher final ResNet mean accuracy of selected STC/EF visible.

The old two-CIFAR claim of a cost advantage at every declared target remains
scoped to those two panels. It is not extended to the MLP.

## Manuscript and artifact changes

- New or rewritten text, captions and MLP table cells are orange through
  \mlpRev / \mlpColor. Earlier blue revisions remain visible. Set both
  \coreColor and \mlpColor to empty to make a clean copy.
- Abstract, contribution statement, protocol, results, limitations and conclusion
  now cover all three architectures.
- Two new vector figures show all three communication/accuracy trajectories
  and all 36 paired target contrasts, including the MLP Ström exception. The
  protocol diagram also uses shorter labels that stay inside its boxes.
- The endpoint table includes accuracy, CE, upload, synchronization/control
  and total traffic. The middle-target table uses 50% for CIFAR and 85% for MLP;
  different datasets' absolute target levels are not compared as equal tasks.
- The original two-CIFAR builder, freeze, source results and generated products
  are preserved. A separate three_architecture_freeze.json locks 866 MLP/source
  inputs and the prior freeze. Original training sources remain unchanged.
- The new builder checks all 150 endpoints and first saved crossings against
  original campaign summaries, the previous CIFAR products, and the independent
  MLP audit. Coverage and final-above-target counts are exported explicitly.
- The reviewed conditional-theory, mechanism and original participation
  qualifications remain. No new asymptotic, latency or energy claim is added.

## Regenerate figures and tables locally (no training)

From Windows Command Prompt in the research repository:

    cd C:\Users\WeberJ\2026_NeuromorphicFL
    conda activate neuromorphicfl-resnet14
    git pull --ff-only
    python paper/ijcnn2027/build_three_architecture.py
    python paper/ijcnn2027/build_three_architecture.py --check
    python paper/ijcnn2027/check_manuscript_claims.py
    git add paper/ijcnn2027/build_three_architecture.py paper/ijcnn2027/figures/three_architecture_*.pdf paper/ijcnn2027/generated/three_architecture_*.tex paper/ijcnn2027/evidence/three_architecture_*.csv
    git diff --cached --check
    git commit -m "Regenerate three-architecture Event-FedAvg comparison"
    git push origin main

The products are already generated and committed. An unchanged rebuild should
produce nothing new to commit in the pinned artifact environment.
Only NumPy and Matplotlib are required; no images, Torch, training models or
optimizer checkpoints are loaded.

To copy regenerated products to a sibling paper checkout:

    copy /Y paper\ijcnn2027\figures\three_architecture_*.pdf ..\2026_NeuromorphicFL_Paper\figures\
    copy /Y paper\ijcnn2027\generated\three_architecture_*.tex ..\2026_NeuromorphicFL_Paper\generated\
    git -C ..\2026_NeuromorphicFL_Paper add figures/three_architecture_*.pdf generated/three_architecture_*.tex
    git -C ..\2026_NeuromorphicFL_Paper diff --cached --check
    git -C ..\2026_NeuromorphicFL_Paper commit -m "Update three-architecture comparison products"
    git -C ..\2026_NeuromorphicFL_Paper push origin main

The paper repository remains authoritative for manuscript edits. Product
synchronization does not replace its main.tex.

## Validation

The standalone build and --check verify the original 480 CIFAR freeze inputs
and all 866 new freeze entries, reconstruct every summary and paired contrast,
and require byte-identical generated products. The MLP campaign's separate
audit covers 75,000 raw round records and all original numerical products.
The current reproducibility driver also checks the preserved historical
products, manuscript claims, theory contract and anonymous supplement.
Compilation keeps the ten-page limit and the standard IEEE bibliography size.
