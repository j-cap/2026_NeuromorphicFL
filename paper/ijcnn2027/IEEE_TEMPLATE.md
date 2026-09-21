# IEEE conference template provenance

The manuscript uses `\documentclass[conference]{IEEEtran}`.

The official IEEE Author Center directs conference authors to the IEEE
conference template selector:

- Author Center: <https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/>
- Template selector: <https://template-selector.ieee.org/>

The selector path was rechecked on 2026-09-06 and remains **Conferences ->
Original Research -> LaTeX**. An official `IEEEtran.cls` downloaded through
that route was supplied by the author on 2026-09-21 and is vendored beside the
manuscript. It identifies itself as IEEEtran V1.8b dated 2015-08-26. Its SHA-256
checksum as downloaded is
`c972aca108fda004c3514d63658e02816da2e54d9a1451e870b9bd970e003f55`.

The supplied file has the same textual contents as IEEEtran V1.8b from the TeX
installation used for the preceding validation build. The byte-level files
differed only in line endings: the official download uses CRLF and the TeX
installation copy uses LF. Keeping the official file in this directory makes
local and Overleaf builds resolve this verified copy before a system-wide
installation. Do not modify `IEEEtran.cls` directly.
