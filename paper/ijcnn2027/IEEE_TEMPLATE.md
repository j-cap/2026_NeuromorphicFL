# IEEE conference template provenance

The manuscript uses `\documentclass[conference]{IEEEtran}`.

The official IEEE Author Center directs conference authors to the IEEE
conference template selector:

- Author Center: <https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/>
- Template selector: <https://template-selector.ieee.org/>

The selector path was rechecked on 2026-09-06 and remains **Conferences ->
Original Research -> LaTeX**. It identified an IEEE conference LaTeX template.
Its generated download endpoint did not deliver an archive in the automated
browser, and a direct request returned HTTP 404. Therefore `IEEEtran.cls` is
intentionally not vendored here: committing a copy from CTAN or another mirror
would not satisfy the requested IEEE-website provenance.

For local compilation, install the `IEEEtran` package from the system TeX
distribution. Overleaf also supplies the standard class. Before final
submission, retry the then-current conference-template download, verify the
IJCNN-specific author instructions, and record the archive checksum here if the
official endpoint succeeds. Do not modify or hand-copy `IEEEtran.cls`.
