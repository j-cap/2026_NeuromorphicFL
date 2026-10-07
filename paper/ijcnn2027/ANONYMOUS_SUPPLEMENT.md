# Event-FedAvg anonymous reproducibility supplement

This archive contains the frozen aggregate and per-seed evidence used for the
paper's central tables, figures, operator ablation, and alignment audit. It also
contains the scripts that regenerate those products and the implementation used
to check the paper's encoder equations. No repository history, account name,
email address, or public project URL is included.

From the source repository, create the deterministic submission archive with:

```bash
python paper/ijcnn2027/build_anonymous_supplement.py
```

## Quick reproduction

Use Python 3.11 from the extracted archive root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r paper/ijcnn2027/requirements-reproduction.txt
python paper/ijcnn2027/reproduce_anonymous.py --regenerate
```

The command reconstructs the machine-readable evidence, comparison plots,
main results table, alignment table, and kappa-reference sensitivity file. It
then checks the empirical prose against the frozen CSVs and executes the
theory-to-implementation contract tests. A successful run leaves no changes in
the generated products.

The archive-level checksum file can be checked independently with
`sha256sum -c MANIFEST.sha256` on Linux or with any SHA-256 utility on Windows.

## Scope

- `experiments/results/` contains the frozen campaign summaries, histories,
  paired comparisons, protocols, and the ResNet-14 per-seed companion files
  required by the builders.
- `paper/ijcnn2027/evidence/` contains normalized evidence tables.
- `paper/ijcnn2027/generated/` contains generated LaTeX table fragments.
- `paper/ijcnn2027/figures/` contains the generated vector plots and the
  editable method diagram.
- `src/neuromorphicfl/` and `experiments/*.py` contain the communication-layer
  implementation and campaign entry points.

The archive supports exact regeneration of the reported analysis from the
committed numerical outputs. Full training reruns require the evidence
environment in `requirements-evidence.txt`, the public Fashion-MNIST and
CIFAR-10 datasets, and appropriate compute. Training can be statistically
reproduced across supported systems, but bitwise identity is not promised
across numerical libraries and hardware.

## Anonymity and submission use

The archive is intended for anonymous supplementary upload with the initial
submission. Do not add repository links, commit history, author names, account
names, or machine-specific paths before review. The archive builder performs a
text scan for known identity-bearing strings before writing the ZIP file.
