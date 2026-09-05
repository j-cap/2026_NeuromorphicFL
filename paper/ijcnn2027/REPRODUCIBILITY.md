# IJCNN reproducibility guide

This guide distinguishes two reproducibility levels. The submission gate uses
the first level: a clean checkout must rebuild every central table and figure
from the committed, frozen experiment outputs. Full training reruns are the
second level and remain manual because they are computational campaigns, not
ordinary document checks.

## Artifact-level reproduction

Use Python 3.11 from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r paper/ijcnn2027/requirements-reproduction.txt
python paper/ijcnn2027/reproduce.py --regenerate --strict-environment
git diff --exit-code -- \
  paper/ijcnn2027/evidence/ \
  paper/ijcnn2027/figures/ \
  paper/ijcnn2027/generated/
```

The command regenerates the six evidence products, two vector figures, and
headline table; checks all 19 empirical prose claims against the source CSVs;
exercises the theory/implementation contract; validates the Actions policy;
and verifies SHA-256 hashes for the frozen sources, products, manuscript, and
central implementation.

To include a manuscript compile, install `latexmk`, `pdfinfo` (Poppler), and a
LaTeX distribution containing the standard IEEE dependencies, then run:

```bash
python paper/ijcnn2027/reproduce.py --compile --strict-environment
```

The compile step rejects a manuscript over six pages and fails on overfull
boxes or unresolved citations/references. The stricter submission-format audit
is P10.

## Full experiment reproduction

The numerical campaign environment is pinned separately:

```bash
python -m pip install -r paper/ijcnn2027/requirements-evidence.txt
```

One Fashion-MNIST held-out point can be rerun with:

```bash
PYTHONPATH=src python experiments/final_baseline_point.py \
  --architecture mlp --method event \
  --partition-seed 2500 --train-seed 72500 --tag eval
```

The complete Fashion-MNIST, CIFAR-10, P8, and T4 matrices are encoded in their
manual-only GitHub Actions workflows. Run them only intentionally from the
Actions tab and retain their per-seed products before aggregating. The source
run URLs and selected configurations are recorded in `EVIDENCE_FREEZE.md`,
`P3_DECISION.md`, `P4_THEORY_AUDIT.md`, and `P8_REVISION_AUDIT.md`.

The frozen P3 Event-FedAvg headline remains the authoritative CIFAR-10 result.
The P8 like-for-like rerun reproduced two of three seeds exactly; seed 3500
followed a different last-bit threshold branch on a later execution platform.
Accordingly, artifact-level reproduction is byte-exact, while a fresh training
rerun is expected to be statistically consistent rather than guaranteed
bitwise identical across platforms.

## Updating the freeze

Do not update the checksum manifest to make an unexplained difference pass.
After an intentional, reviewed change, regenerate all products, inspect the
diff, run the claim and manuscript checks, and only then accept the new bytes:

```bash
python paper/ijcnn2027/reproduce.py --regenerate --update-manifest
python paper/ijcnn2027/reproduce.py --compile
```

P10 must tag the exact commit whose PDF is uploaded to the submission system;
an earlier planning or reproducibility commit must not be labeled as submitted.
