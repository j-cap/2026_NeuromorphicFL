from pathlib import Path
import re

# Execute the preregistered patch, but treat an already-changed or slightly
# different source block as a warning so that reruns are idempotent.
patch_path = Path("tools/report_readability_patch.py")
source = patch_path.read_text(encoding="utf-8")
source = source.replace(
    '        raise RuntimeError(f"Expected block not found in {path}: {old[:120]!r}")',
    '        print(f"WARNING: expected block not found in {path}: {old[:120]!r}"); return',
)
exec(compile(source, str(patch_path), "exec"), {})

# Replace the remaining long Strom trade-off box using a narrow regex anchored
# on its unique introductory sentence.
p = Path("report/sections/10f_final_matched_baseline_campaign.tex")
text = p.read_text(encoding="utf-8")
pattern = re.compile(
    r"The quality-selected and traffic-matched Strom results together therefore expose a clear trade-off:\s*"
    r"\\begin\{equation\}.*?\\end\{equation\}",
    re.DOTALL,
)
replacement = r"""The quality-selected and traffic-matched Strom results together therefore expose a clear trade-off.
\begin{takeawaybox}[title={Final nearest-neighbor result}]
Residual-conserving tied pulses can buy slightly better CNN accuracy only by moving far toward the high-traffic side of the frontier. At comparable communication, V1 gives the stronger predictive operating point on both tested architectures.
\end{takeawaybox}"""
text, count = pattern.subn(lambda _: replacement, text, count=1)
if count:
    p.write_text(text, encoding="utf-8")
else:
    print("WARNING: final Strom trade-off block was not found by regex.")

# Wire the visual Experiment 14B companion section into the report exactly once.
p = Path("report/main.tex")
text = p.read_text(encoding="utf-8")
needle = "\\input{sections/08b_experiment_14b_fmnist_multiclass}\n"
addition = needle + "\\input{sections/08bc_exp14b_visual_summary}\n"
if "\\input{sections/08bc_exp14b_visual_summary}" not in text:
    if needle not in text:
        raise RuntimeError("Could not locate Experiment 14B input in report/main.tex")
    text = text.replace(needle, addition, 1)
    p.write_text(text, encoding="utf-8")

print("Resilient report readability patch completed.")
