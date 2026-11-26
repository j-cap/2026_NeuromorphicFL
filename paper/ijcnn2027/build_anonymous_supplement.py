"""Build a deterministic, identity-screened supplementary ZIP archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
OMITTED = {
    "experiments/stc_cifar10_confirmation.py",
    "experiments/FMNIST_MLP_DEVELOPMENT_RESULTS.md",
}


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
        "paper/ijcnn2027/build_calibrated_core.py",
        "paper/ijcnn2027/build_three_architecture.py",
        "paper/ijcnn2027/evidence/calibrated_core_freeze.json",
        "paper/ijcnn2027/evidence/three_architecture_freeze.json",
        "paper/ijcnn2027/build_alignment_factorial.py",
        "paper/ijcnn2027/check_manuscript_claims.py",
        "paper/ijcnn2027/check_theory_contract.py",
        "paper/ijcnn2027/reproduce_anonymous.py",
        "paper/ijcnn2027/figures/event_fedavg_method.drawio",
        "paper/ijcnn2027/figures/event_fedavg_method.pdf",
        "experiments/results/final_baseline_campaign/observed_heldout_summary.csv",
        "experiments/results/final_baseline_campaign/observed_traffic_matched_summary.csv",
    )
    # The archive is an artifact-reproduction profile. One historical driver
    # contains an identifying repository name in its help examples; it is not
    # required by any builder. Preserve that omission explicitly in the archive
    # freeze rather than rewriting a frozen scientific source.
    for name in ("calibrated_core_freeze.json", "three_architecture_freeze.json"):
        freeze = json.loads((PAPER / "evidence" / name).read_text())
        for relative in freeze["files"]:
            if relative not in OMITTED:
                add_if_present(paths, relative)

    for relative in exact:
        add_if_present(paths, relative)

    for pattern in (
        "src/neuromorphicfl/*.py",
        "paper/ijcnn2027/evidence/*.csv",
        "paper/ijcnn2027/generated/*.tex",
        "paper/ijcnn2027/figures/communication_frontier_*.pdf",
        "paper/ijcnn2027/figures/convergence_*.pdf",
        "paper/ijcnn2027/figures/calibrated_*.pdf",
        "paper/ijcnn2027/figures/three_architecture_*.pdf",
        "paper/ijcnn2027/figures/p20_*.pdf",
        "paper/ijcnn2027/supplementary/*.md",
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


def anonymous_freezes() -> dict[Path, bytes]:
    """Declare omitted identifying extras without rewriting scientific inputs."""
    output = {}
    for name in ("calibrated_core_freeze.json", "three_architecture_freeze.json"):
        path = PAPER / "evidence" / name
        manifest = json.loads(path.read_text())
        manifest["artifact_profile"] = "anonymous_output_reproduction"
        manifest["omitted_artifact_extras"] = {
            item: manifest["files"].pop(item)
            for item in sorted(OMITTED) if item in manifest["files"]
        }
        # The third-panel freeze includes the previous manifest itself.
        # Its artifact-profile hash must describe the copy actually shipped.
        previous = PAPER / "evidence/calibrated_core_freeze.json"
        previous_name = previous.relative_to(REPO).as_posix()
        if previous_name in manifest["files"]:
            manifest["files"][previous_name] = hashlib.sha256(output[previous]).hexdigest()
        output[path] = (json.dumps(manifest, indent=2) + "\n").encode()
    return output


def build(output: Path) -> tuple[int, str]:
    files = selected_files()
    profiles = anonymous_freezes()
    payloads: list[tuple[str, bytes]] = []
    readme = (PAPER / "ANONYMOUS_SUPPLEMENT.md").read_bytes()
    scan_text(PAPER / "ANONYMOUS_SUPPLEMENT.md", readme)
    payloads.append((f"{ARCHIVE_ROOT}/README.md", readme))
    for path in sorted(files, key=lambda item: item.relative_to(REPO).as_posix()):
        content = path.read_bytes()
        if path in profiles:
            content = profiles[path]
        if path.parent == PAPER / "supplementary" and path.suffix == ".md":
            note = content.decode("utf-8")
            note = re.sub(r"https://github[.]com/j-cap/[^\s)]+", "[source snapshot]", note)
            note = note.replace("j-cap/2026_NeuromorphicFL", "[source repository]")
            content = note.encode()
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
