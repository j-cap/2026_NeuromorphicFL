"""Regenerate the anonymous supplement's central empirical products."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper" / "ijcnn2027"


def run(*command: str) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="rewrite the central evidence, figures, and tables before checking",
    )
    args = parser.parse_args()

    builders = (
        PAPER / "build_evidence.py",
        PAPER / "build_visuals.py",
        PAPER / "build_alignment_factorial.py",
    )
    if args.regenerate:
        for builder in builders:
            run(sys.executable, str(builder))
    for builder in builders:
        run(sys.executable, str(builder), "--check")
    run(sys.executable, str(PAPER / "check_manuscript_claims.py"))
    run(sys.executable, str(PAPER / "check_theory_contract.py"))
    print("Anonymous Event-FedAvg supplement validated")


if __name__ == "__main__":
    main()
