from pathlib import Path
import re

SECTIONS = Path("report/sections")


def wrap_tabulars(text: str) -> str:
    """Guard ordinary tabular environments with max-width adjustbox.

    The guard is idempotent: if an adjustbox begins immediately before the
    tabular (ignoring whitespace), the table is left untouched.
    """
    pos = 0
    out = []
    while True:
        start = text.find(r"\begin{tabular}", pos)
        if start < 0:
            out.append(text[pos:])
            break
        out.append(text[pos:start])
        prefix = text[max(0, start - 90):start]
        end = text.find(r"\end{tabular}", start)
        if end < 0:
            raise RuntimeError("Unterminated tabular environment")
        end2 = end + len(r"\end{tabular}")
        block = text[start:end2]
        if re.search(r"\begin\{adjustbox\}\{max width=\\(?:textwidth|linewidth)\}\s*$", prefix):
            out.append(block)
        else:
            out.append(
                r"\begin{adjustbox}{max width=\textwidth}" + "\n" +
                block + "\n" + r"\end{adjustbox}"
            )
        pos = end2
    return "".join(out)


def brace_argument(text: str, brace_start: int):
    depth = 0
    i = brace_start
    while i < len(text):
        c = text[i]
        if c == "{" and (i == 0 or text[i-1] != "\\"):
            depth += 1
        elif c == "}" and (i == 0 or text[i-1] != "\\"):
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def safe_long_boxed_equations(text: str) -> str:
    """Fit long historical boxed callouts to the text width.

    Only unlabelled equation environments are changed. Short verdict boxes are
    kept exactly as authored. Publication-facing long prose boxes have already
    been rewritten as wrapping tcolorboxes and are therefore unaffected.
    """
    pattern = re.compile(r"\\begin\{equation\}(.*?)\\end\{equation\}", re.DOTALL)

    def repl(match: re.Match) -> str:
        body = match.group(1)
        if r"\boxed" not in body or r"\label" in body:
            return match.group(0)
        # Long/structured callouts are the problematic historical cases.
        normalized = re.sub(r"\s+", " ", body).strip()
        if len(normalized) < 105 and r"\begin{array}" not in body and r"\begin{matrix}" not in body and r"\begin{aligned}" not in body:
            return match.group(0)
        # Avoid double-sanitizing a previously converted block.
        if r"\resizebox" in body:
            return match.group(0)
        return (
            "\\begin{center}\n"
            "\\resizebox{0.96\\linewidth}{!}{$\\displaystyle " + body.strip() + "$}\n"
            "\\end{center}"
        )

    return pattern.sub(repl, text)


changed = []
for path in sorted(SECTIONS.glob("*.tex")):
    text = path.read_text(encoding="utf-8")
    new = wrap_tabulars(text)
    new = safe_long_boxed_equations(new)
    if new != text:
        path.write_text(new, encoding="utf-8")
        changed.append(str(path))

print(f"Width sanitizer changed {len(changed)} section files")
for p in changed:
    print(p)
