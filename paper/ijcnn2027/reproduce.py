"""Rebuild and validate the frozen IJCNN submission artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper" / "ijcnn2027"
MANIFEST = PAPER / "reproducibility_manifest.json"
REQUIREMENTS = PAPER / "requirements-reproduction.txt"


def run(*command: str, cwd: Path = REPO) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def pinned_requirements() -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise AssertionError(f"unpinned reproduction dependency: {line}")
        package, version = line.split("==", 1)
        pins[package] = version
    return pins


def check_environment() -> None:
    if sys.version_info[:2] != (3, 11):
        raise AssertionError(
            f"reproduction requires Python 3.11, found {sys.version_info.major}.{sys.version_info.minor}"
        )
    for package, expected in pinned_requirements().items():
        observed = importlib.metadata.version(package)
        if observed != expected:
            raise AssertionError(
                f"{package} version drift: expected {expected}, found {observed}"
            )
    print("validated pinned Python 3.11 reproduction environment")


def check_manifest() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise AssertionError("unsupported reproducibility manifest schema")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise AssertionError("reproducibility manifest has no files")
    seen: set[str] = set()
    for entry in entries:
        relative = entry["path"]
        if relative in seen:
            raise AssertionError(f"duplicate manifest path: {relative}")
        seen.add(relative)
        path = REPO / relative
        if not path.is_file():
            raise AssertionError(f"missing frozen artifact: {relative}")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != entry["sha256"]:
            raise AssertionError(
                f"checksum drift for {relative}: expected {entry['sha256']}, found {observed}"
            )
    print(f"validated {len(entries)} frozen artifact checksums")
    return len(entries)


def update_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = REPO / entry["path"]
        if not path.is_file():
            raise AssertionError(f"cannot freeze missing artifact: {entry['path']}")
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"updated {len(manifest['files'])} frozen artifact checksums")


def compile_manuscript() -> None:
    for executable in ("latexmk", "pdfinfo"):
        if shutil.which(executable) is None:
            raise SystemExit(f"{executable} is required for --compile")
    run("latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex", cwd=PAPER)
    info = subprocess.run(
        ["pdfinfo", "main.pdf"],
        cwd=PAPER,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    pages = next(
        (int(line.split(":", 1)[1]) for line in info.splitlines() if line.startswith("Pages:")),
        None,
    )
    if pages is None or pages > 6:
        raise AssertionError(f"manuscript page limit failed: {pages}")
    log = (PAPER / "main.log").read_text(encoding="utf-8", errors="replace")
    forbidden = ("Overfull \\hbox", "Overfull \\vbox", "Citation `", "Reference `")
    warnings = [marker for marker in forbidden if marker in log]
    if warnings:
        raise AssertionError(f"manuscript log contains forbidden warnings: {warnings}")
    print(f"compiled manuscript successfully: {pages} pages")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="rewrite central tables and figures before validating them",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="compile main.tex and enforce the P9 page/warning checks",
    )
    parser.add_argument(
        "--strict-environment",
        action="store_true",
        help="require the frozen Python version and direct dependencies",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="accept current bytes as a new deliberate freeze (never used by CI)",
    )
    args = parser.parse_args()

    if args.strict_environment:
        check_environment()
    if args.regenerate:
        run(sys.executable, str(PAPER / "build_evidence.py"))
        run(sys.executable, str(PAPER / "build_visuals.py"))

    run(sys.executable, str(PAPER / "build_evidence.py"), "--check")
    run(sys.executable, str(PAPER / "build_visuals.py"), "--check")
    run(sys.executable, str(PAPER / "check_manuscript_claims.py"))
    run(sys.executable, str(PAPER / "check_theory_contract.py"))
    run(sys.executable, str(REPO / ".github" / "scripts" / "check_actions_policy.py"))
    if args.update_manifest:
        update_manifest()
    check_manifest()
    if args.compile:
        compile_manuscript()
    print("P9 reproducibility bundle validated")


if __name__ == "__main__":
    main()
