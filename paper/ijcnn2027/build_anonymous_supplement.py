"""Build a deterministic, identity-screened supplementary ZIP archive."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile


REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper" / "ijcnn2027"
DEFAULT_OUTPUT = PAPER / "dist" / "event-fedavg-anonymous-supplement.zip"
ARCHIVE_ROOT = "event-fedavg-anonymous"
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".tex", ".txt", ".bib"}
FORBIDDEN = (
    "github.com/j-cap",
    "j-cap/",
    "weberj",
    "c:\\users\\weber",
)


def add_if_present(paths: set[Path], relative: str) -> None:
    path = REPO / relative
    if not path.is_file():
        raise AssertionError(f"missing supplement input: {relative}")
    paths.add(path)


def selected_files() -> set[Path]:
    paths: set[Path] = set()
    exact = (
        "paper/ijcnn2027/main.tex",
        "paper/ijcnn2027/references.bib",
        "paper/ijcnn2027/IEEEtran.cls",
        "paper/ijcnn2027/IEEEtran.bst",
        "paper/ijcnn2027/requirements-reproduction.txt",
        "paper/ijcnn2027/requirements-evidence.txt",
        "paper/ijcnn2027/build_evidence.py",
        "paper/ijcnn2027/build_visuals.py",
        "paper/ijcnn2027/build_alignment_factorial.py",
        "paper/ijcnn2027/check_manuscript_claims.py",
        "paper/ijcnn2027/check_theory_contract.py",
        "paper/ijcnn2027/reproduce_anonymous.py",
        "paper/ijcnn2027/figures/event_fedavg_method.drawio",
        "paper/ijcnn2027/figures/event_fedavg_method.pdf",
        "experiments/results/final_baseline_campaign/observed_heldout_summary.csv",
        "experiments/results/final_baseline_campaign/observed_traffic_matched_summary.csv",
    )
    for relative in exact:
        add_if_present(paths, relative)

    for pattern in (
        "src/neuromorphicfl/*.py",
        "paper/ijcnn2027/evidence/*.csv",
        "paper/ijcnn2027/generated/*.tex",
        "paper/ijcnn2027/figures/communication_frontier_*.pdf",
        "paper/ijcnn2027/figures/convergence_*.pdf",
        "experiments/p*.py",
        "experiments/final_baseline_*.py",
        "experiments/results/p3_cifar10/*.csv",
        "experiments/results/p3_cifar10/*.json",
        "experiments/results/p8_targeted_revision/*.csv",
        "experiments/results/p8_targeted_revision/*.json",
        "experiments/results/p11_alignment_factorial/*.csv",
        "experiments/results/p12_headline_ten_seed/*.csv",
        "experiments/results/p12_headline_ten_seed/*.json",
        "experiments/results/p14_figure2_ten_seed/*.csv",
        "experiments/results/p15_operator_ablation/*.csv",
        "experiments/results/p15_operator_ablation/*.json",
        "experiments/results/p17_fmnist_final_horizons/*.csv",
        "experiments/results/p17_fmnist_final_horizons/*.json",
        "experiments/results/p19_cifar10_cnn_final_horizons/aggregate.csv",
        "experiments/results/p19_cifar10_cnn_final_horizons/histories.csv",
        "experiments/results/p19_cifar10_cnn_final_horizons/runs.csv",
        "experiments/results/p19_cifar10_cnn_final_horizons/paired_differences.csv",
        "experiments/results/p19_cifar10_cnn_final_horizons/protocol.json",
        "experiments/results/p13_cifar10_resnet14/*heldout-r3000.csv",
        "experiments/results/p13_cifar10_resnet14/*heldout-r3000_history.csv",
        "experiments/results/p13_cifar10_resnet14/*heldout-r3000_run.json",
        "experiments/results/p13_cifar10_resnet14/heldout-r3000_summary.csv",
        "experiments/results/p13_cifar10_resnet14/selection.json",
        "experiments/results/p13_cifar10_resnet14/gate.json",
    ):
        matches = [
            path
            for path in REPO.glob(pattern)
            if path.is_file()
            and path.relative_to(REPO).as_posix()
            != "paper/ijcnn2027/evidence/t4_alignment_audit.csv"
        ]
        if not matches:
            raise AssertionError(f"supplement pattern matched no files: {pattern}")
        paths.update(matches)
    return paths


def archive_name(path: Path) -> str:
    return f"{ARCHIVE_ROOT}/{path.relative_to(REPO).as_posix()}"


def scan_text(path: Path, content: bytes) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    text = content.decode("utf-8", errors="replace").lower()
    matched = [pattern for pattern in FORBIDDEN if pattern in text]
    if matched:
        raise AssertionError(
            f"identity-bearing text in {path.relative_to(REPO)}: {matched}"
        )


def write_member(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, content, compresslevel=9)


def build(output: Path) -> tuple[int, str]:
    files = selected_files()
    payloads: list[tuple[str, bytes]] = []
    readme = (PAPER / "ANONYMOUS_SUPPLEMENT.md").read_bytes()
    scan_text(PAPER / "ANONYMOUS_SUPPLEMENT.md", readme)
    payloads.append((f"{ARCHIVE_ROOT}/README.md", readme))
    for path in sorted(files, key=lambda item: item.relative_to(REPO).as_posix()):
        content = path.read_bytes()
        scan_text(path, content)
        payloads.append((archive_name(path), content))

    checksums = "".join(
        f"{hashlib.sha256(content).hexdigest()}  {name.removeprefix(ARCHIVE_ROOT + '/')}\n"
        for name, content in payloads
    ).encode("utf-8")
    payloads.append((f"{ARCHIVE_ROOT}/MANIFEST.sha256", checksums))

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for name, content in payloads:
            write_member(archive, name, content)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return len(payloads), digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    count, digest = build(args.output.resolve())
    print(f"wrote {args.output}: {count} files, sha256={digest}")


if __name__ == "__main__":
    main()
