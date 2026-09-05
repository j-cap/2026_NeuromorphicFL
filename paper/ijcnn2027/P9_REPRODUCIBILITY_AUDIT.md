# P9 reproducibility audit

## Decision

P9 closes at the artifact-reproduction level. A clean checkout regenerates the
central evidence tables and vector figures from committed source results,
checks every empirical number written directly in the manuscript, compiles the
paper independently, and verifies a checksum manifest. The expensive training
campaigns remain manual and are not triggered by paper changes.

## Frozen bundle

`reproducibility_manifest.json` records SHA-256 hashes for the submission
bundle, including:

- authoritative Fashion-MNIST, CIFAR-10, P8, T4, and P11 result sources;
- all manuscript evidence CSVs, generated tables, and vector figures;
- the manuscript, bibliography, and IEEE template-provenance record;
- the deterministic builders and validation scripts;
- the pinned artifact and campaign environments; and
- the central experiment implementation.

`reproduce.py` is the single entry point. It can regenerate products, enforce
the Python 3.11 environment, validate manuscript claims and theory semantics,
check the manifest, and optionally compile the paper. The automatic IJCNN
evidence workflow now performs regeneration in a clean GitHub checkout and
requires the generated directories to remain byte-identical.

## Portability audit

Repository-relative paths are derived from each script's location. No tracked
paper source, workflow, or reproduction script contains a workstation-specific
absolute path. LaTeX auxiliary files and the local PDF are excluded explicitly;
the two manuscript figures remain intentionally tracked under `figures/`.

The direct artifact dependencies are pinned in
`requirements-reproduction.txt`; NumPy and Pandas for full experiment reruns
remain pinned in `requirements-evidence.txt`. The LaTeX toolchain is a system
dependency and is provisioned explicitly by the manuscript workflow.

## Reproduction boundary

The gate does not claim bitwise-identical retraining across BLAS, CPU, or NumPy
implementations. P8 documented one threshold-sensitive CIFAR-10 seed whose
last-bit numerical branch changed on a later platform, while the other two
seeds reproduced exactly. The paper therefore retains the predeclared P3
headline and uses P8 only for the targeted mechanism audit. Committed source
results and every paper artifact derived from them are byte-frozen.

## Tagging boundary

The P9 commit is a reproducibility freeze, not the submitted artifact. P10 may
still change author/anonymization fields, disclosures, or format details after
checking the current instructions. The exact uploaded commit must be tagged
only after those checks and author approval, so the submission tag is moved to
the P10 gate rather than created prematurely.
