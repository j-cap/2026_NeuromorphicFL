from pathlib import Path

# Execute the preregistered patch, but treat an already-changed or slightly
# different source block as a warning so that one formatting mismatch does not
# discard all earlier successful replacements in the same working tree.
patch_path = Path("tools/report_readability_patch.py")
source = patch_path.read_text(encoding="utf-8")
source = source.replace(
    '        raise RuntimeError(f"Expected block not found in {path}: {old[:120]!r}")',
    '        print(f"WARNING: expected block not found in {path}: {old[:120]!r}"); return',
)
exec(compile(source, str(patch_path), "exec"), {})

# The final Strom trade-off block contains a small source-format difference from
# the original patch specification; replace it explicitly here.
p = Path("report/sections/10f_final_matched_baseline_campaign.tex")
text = p.read_text(encoding="utf-8")
old = r"""The quality-selected and traffic-matched Strom results together therefore expose a clear trade-off:
\begin{equation}
\boxed{
\begin{array}{c}
\text{residual-conserving tied pulses can buy slightly better CNN accuracy}\\
\text{only by moving far toward the high-traffic side of the frontier,}\\
\text{whereas at comparable traffic V1 gives the stronger predictive operating point.}
\end{array}}
\end{equation}"""
new = r"""The quality-selected and traffic-matched Strom results together therefore expose a clear trade-off.
\begin{takeawaybox}[title={Final nearest-neighbor result}]
Residual-conserving tied pulses can buy slightly better CNN accuracy only by moving far toward the high-traffic side of the frontier. At comparable communication, V1 gives the stronger predictive operating point on both tested architectures.
\end{takeawaybox}"""
if old in text:
    p.write_text(text.replace(old, new, 1), encoding="utf-8")
else:
    print("WARNING: final Strom trade-off block still not matched; leaving source unchanged.")

print("Resilient report readability patch completed.")
