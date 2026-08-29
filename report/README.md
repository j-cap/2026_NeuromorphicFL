# Report knowledge base

`main.tex` is the living technical report for the project. It is intentionally broader and more explicit than a future paper manuscript.

## Build

From the `report/` directory:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Maintenance convention

- Add mechanism/theory knowledge to `sections/03_method.tex`.
- Add validated experimental evidence, including negative results, to `sections/04_experiments.tex`.
- Keep literature collisions and novelty constraints in `sections/02_literature.tex`.
- Add proposed experiments only to `sections/05_roadmap.tex` until they are actually run.
- Keep `references.bib` source-complete enough that the report can later be distilled into a paper without reconstructing the literature trail.
- Generated figures should eventually be copied or linked into `report/figures/` only when they are stable enough to reference in the report.
