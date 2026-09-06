"""Validate the repository's bounded GitHub Actions execution policy."""

from __future__ import annotations

from pathlib import Path

import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
AUTOMATIC = {
    "ijcnn_evidence_check.yml",
    "ijcnn_manuscript_check.yml",
    "report_compile.yml",
}


class StrictBaseLoader(yaml.BaseLoader):
    """Keep YAML scalars as strings and reject duplicate mapping keys."""


def strict_mapping(loader: StrictBaseLoader, node: yaml.MappingNode, deep: bool = False):
    mapping: dict[str, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key: {key}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


StrictBaseLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, strict_mapping)


def as_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def validate_workflow(path: Path) -> None:
    with path.open(encoding="utf-8") as stream:
        data = as_mapping(yaml.load(stream, Loader=StrictBaseLoader), path.name)

    triggers = as_mapping(data.get("on"), f"{path.name}: on")
    expected = (
        {"pull_request", "workflow_dispatch"}
        if path.name in AUTOMATIC
        else {"workflow_dispatch"}
    )
    if set(triggers) != expected:
        raise ValueError(
            f"{path.name}: triggers {sorted(triggers)} != {sorted(expected)}"
        )
    if path.name in AUTOMATIC:
        pull_request = as_mapping(
            triggers["pull_request"], f"{path.name}: pull_request"
        )
        if not pull_request.get("paths"):
            raise ValueError(f"{path.name}: automatic check requires path filters")

    concurrency = as_mapping(data.get("concurrency"), f"{path.name}: concurrency")
    if concurrency.get("cancel-in-progress") != "true" or not concurrency.get("group"):
        raise ValueError(f"{path.name}: superseded runs must be cancelled")

    jobs = as_mapping(data.get("jobs"), f"{path.name}: jobs")
    limit = 30 if path.name in AUTOMATIC else 180
    for job_name, job_value in jobs.items():
        job = as_mapping(job_value, f"{path.name}: job {job_name}")
        timeout = int(str(job.get("timeout-minutes", "0")))
        if timeout <= 0 or timeout > limit:
            raise ValueError(
                f"{path.name}: job {job_name} timeout {timeout} exceeds {limit}"
            )
        if path.name in AUTOMATIC:
            strategy = job.get("strategy")
            if isinstance(strategy, dict) and "matrix" in strategy:
                raise ValueError(f"{path.name}: automatic checks may not use matrices")

        steps = job.get("steps", [])
        if not isinstance(steps, list):
            raise ValueError(f"{path.name}: job {job_name} steps must be a list")
        for step in steps:
            if not isinstance(step, dict):
                continue
            if step.get("uses") == "actions/upload-artifact@v4":
                options = as_mapping(
                    step.get("with"), f"{path.name}: upload-artifact with"
                )
                retention = int(str(options.get("retention-days", "0")))
                if retention <= 0 or retention > 7:
                    raise ValueError(
                        f"{path.name}: artifact retention must be 1--7 days"
                    )


def main() -> None:
    paths = sorted(WORKFLOWS.glob("*.yml"))
    if not paths:
        raise SystemExit("no GitHub Actions workflows found")
    for path in paths:
        validate_workflow(path)
    print(
        f"validated Actions policy for {len(paths)} workflows: "
        f"{len(AUTOMATIC)} automatic, {len(paths) - len(AUTOMATIC)} manual-only"
    )


if __name__ == "__main__":
    main()
