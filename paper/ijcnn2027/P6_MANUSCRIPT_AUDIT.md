# P6 conference-manuscript audit

## Outcome

P6 is closed. `main.tex` is a complete, standalone conference manuscript with
no drafting placeholders or report-chronology language. The author block
remains intentionally anonymous until the project team supplies the author
list and the applicable IJCNN policy is checked at P10.

## Completed narrative

- The abstract states the operator, scoped theory, evaluation standard, and
  held-out CIFAR-10 headline result.
- The introduction motivates temporal evidence rather than per-round-only
  compression and states the algorithmic, theoretical, and empirical
  contributions together with their boundary.
- The related-work section positions the complete operator against residual
  pulses, error feedback, event-triggered learning, and federated SNNs.
- The method and theory sections retain the frozen P4 semantics and conditional
  result without importing the living report's derivation history.
- The experimental protocol records tasks, models, strong non-IID setting,
  matched local learning, development-only selection, held-out seeds, and
  conservative bidirectional accounting.
- The results distinguish quality-selected from traffic-matched controls,
  state the Fashion-MNIST CNN qualification, and avoid significance or uniform
  worst-class-superiority claims.
- The conclusion states the synchronous, full-participation, finite-benchmark,
  software-accounting, and conditional-theory limitations.

## Mechanical checks

The manuscript was compiled from `paper/ijcnn2027/main.tex` with IEEEtran
V1.8b. The validated PDF has five US-letter pages, resolved citations and
cross-references, and no overfull boxes. All three P5 visual elements render
legibly at IEEE two-column scale. The intentionally short fifth page leaves
room for revisions from P7; P6 does not add filler merely to consume the
provisional page budget.

The workflow `.github/workflows/ijcnn_manuscript_check.yml` reproduces these
checks on manuscript changes. It installs IEEEtran through the system TeX
distribution rather than vendoring an unverified class file, compiles with
`latexmk`, rejects internal drafting language, enforces at most six pages, and
fails on unresolved references or overfull boxes.

## Claim audit

Every substantive claim in the manuscript remains within `CLAIM_EVIDENCE.md`:

- C1--C3 support the complete operator and synchronization protocol;
- C4 and C9 support the three communication--performance frontiers;
- C5--C7 support the encoder properties, conditional optimization interface,
  and finite-trajectory alignment audit.

The manuscript does not activate excluded C10 or C11 claims. The P2 literature
boundary, official page limit, anonymization requirements, and IEEE/IJCNN
submission policy still require their planned P10 refresh.

## P6 gate

The gate is satisfied: the paper compiles as a complete, self-contained draft,
contains no placeholders, fits the provisional page budget, and can now enter
the three-perspective peer-review simulation in P7.
