# IEEE conference template provenance

The manuscript uses `\documentclass[conference]{IEEEtran}`.

The official IEEE Author Center directs conference authors to the IEEE
conference template selector:

- Author Center: <https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/>
- Template selector: <https://template-selector.ieee.org/>

The selector path used was **Conferences -> Original Research -> LaTeX**. It
identified an IEEE conference LaTeX template, but the download endpoint did not
deliver a file in the automated browser. Therefore `IEEEtran.cls` is
intentionally not vendored here: committing a copy from CTAN or another mirror
would not satisfy the requested IEEE-website provenance.

For local compilation, install the `IEEEtran` package from the system TeX
distribution. Before final submission, download the then-current conference
template from the selector, verify the IJCNN-specific author instructions, and
record the archive checksum here. Do not modify or hand-copy `IEEEtran.cls`.
