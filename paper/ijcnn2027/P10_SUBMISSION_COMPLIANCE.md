# P10 submission-compliance audit

**Audit snapshot:** 2026-09-06 UTC

**Candidate base:** `119cf759d99ae191a6fd863b1a42f8c9e2dbbd20`

P10 separates repository-verifiable checks from decisions that require the
authors or the live submission system. The first group is complete. The P10
gate remains open until the four external actions at the end of this document
are complete.

## Live venue facts

The official [IJCNN 2027 call for papers](https://ijcnn.org/2027/authors/call-for-papers)
was checked on the audit date. It states:

- regular papers are limited to no more than six pages;
- papers use the IEEE two-column conference proceedings format;
- the regular-paper deadline is January 31, 2027;
- the conference is June 14--18, 2027 in Cape Town;
- federated and distributed learning, communication-efficient learning,
  neuromorphic systems, and event-driven neural processing are in scope.

The site does not yet expose a detailed IJCNN 2027 initial-author-instruction
page. In particular, the live call does not currently settle page size,
double-blind implementation details, paid extra pages, required author
identifiers, the final CMT fields, or whether the conference will issue an IEEE
PDF eXpress link. Those items are therefore mandatory rechecks, not inferred
facts. The 2025 instructions are useful precedent but are not treated as 2027
authority.

## IEEE format and PDF checks

The manuscript uses `\documentclass[conference]{IEEEtran}` and the IEEEtran
bibliography style. The official [IEEE Author Center template page](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/)
and interactive template selector were rechecked. The selector returned the
route **Conferences -> Original Research -> LaTeX**, but its generated download
endpoint returned no archive in the automated browser and a direct request
returned HTTP 404. `IEEEtran.cls` is consequently not vendored from an
unverifiable mirror. TeX Live and Overleaf provide the standard class.

The current candidate is a uniform six-page US Letter PDF produced by pdfTeX.
All fonts are embedded and subset, the PDF is unencrypted, and it contains no
links, bookmarks, annotations, forms, or attachments. It uses PDF 1.5. These
checks implement the machine-verifiable items listed by the [IEEE Xplore PDF
requirements](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/meet-ieee-xplore-requirements/).
The final 2027 page-size instruction is not yet available, so US Letter remains
provisional. If IJCNN 2027 requires A4, add `a4paper` to the class options and
rerun the entire visual and automated audit.

Run the bounded compliance check from `paper/ijcnn2027/` after compiling:

```bash
python -m pip install -r requirements-compliance.txt
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
python check_submission_compliance.py
```

The checker enforces the six-page maximum, recognized uniform page size,
anonymous source and PDF, empty author metadata, PDF version, font embedding
and subsetting, absence of Type 3 fonts, and absence of links, bookmarks, and
embedded files. It also requires the IEEE bibliography style and the AI-use
acknowledgment.

## Anonymous review, repository, and self-citations

The candidate PDF uses `Anonymous Authors`, contains no affiliation, author
footnote, email address, ORCID, project repository URL, or author metadata, and
does not describe prior work as belonging to the authors. The manuscript does
not cite this public repository. The public repository can still make the work
discoverable through its title and commit history. Historical IJCNN guidance
allowed public preprints and technical reports, but the authors must recheck
the live 2027 rule before submission. Do not add the repository URL to an
anonymous PDF unless the 2027 instructions explicitly permit the chosen
handling.

## Originality and concurrent submission

The manuscript bibliography and targeted novelty review were checked, but a
repository cannot prove originality or absence of concurrent submission. Every
author must make the explicit attestations in `SUBMISSION_ATTESTATIONS.md`.
Any related manuscript, thesis, preprint, or report must be disclosed and
handled under the live IEEE/IJCNN rules.

## AI-use disclosure

IEEE policy requires disclosure of AI-generated content and identification of
the system, affected portions, and level of assistance. The anonymous
manuscript now states that OpenAI ChatGPT and Codex assisted drafting and
language editing throughout the paper and experimental and validation code,
and that the authors verified all content. This follows the
current IEEE policy summarized by the [IEEE Robotics and Automation Society](https://www.ieee-ras.org/publications/guidelines-for-generative-ai-usage/).
The authors must confirm that the wording accurately covers the full project
history and update it if another AI system was used.

## Literature refresh

The P2 query families were rerun on 2026-09-06. Newly screened 2026 federated
SNN work includes BKT-DSNN and federated SNN training on Raspberry Pi hardware.
Both place spiking dynamics inside the trained model and exchange model or
compressed-model information. Neither implements the frozen Event-FedAvg
communication operator. The qualified novelty boundary therefore remains
unchanged. `RELATED_WORK.md` records the refresh and must be checked once more
near the actual submission date because the deadline is still months away.

## External actions that keep P10 open

1. Finalize the author list and obtain approval and ethics attestations from
   every author.
2. Recheck the detailed IJCNN 2027 instructions when published, especially
   anonymity, page size, identifiers, and AI-disclosure wording.
3. Upload the exact six-page PDF to CMT and the conference-provided IEEE PDF
   checker, if supplied, then inspect the uploaded rendering.
4. Only after successful submission, tag the exact submitted commit. Do not tag
   the present candidate as submitted.

The P10 gate closes only when the submission system confirms compliance and
the immutable commit/tag record points to the exact uploaded PDF.
