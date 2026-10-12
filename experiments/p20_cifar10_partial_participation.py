"""Larger-client and partial-participation pilot for Event-FedAvg.

The primary comparison keeps ten active clients per round while increasing the
registered population from ten to fifty.  A full-participation fifty-client
cell separates client-count effects from sampling effects.  The existing P19
ten-client seed-zero runs are imported as the matched reference, so the default
pilot computes only six new 1,800-round runs.  A five-of-fifty stress cell is
available as a separate command.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

# Each worker already parallelizes heavily inside NumPy unless constrained.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from neuromorphicfl.cifar10_benchmark import (
    LAYOUT,
    download_cifar10,
    make_cifar10_federation,
)
from neuromorphicfl.partial_participation import (
    participation_schedule,
    run_partial_participation,
    schedule_sha256,
)
from p12_headline_ten_seed import CIFAR_BASE


DATA = REPO / "data" / "cifar-10"
OUT = REPO / "experiments" / "results" / "p20_cifar10_partial_participation"
P19 = REPO / "experiments" / "results" / "p19_cifar10_cnn_final_horizons"
PARTITION_SEED = 3500
TRAIN_SEED = 80_000 + PARTITION_SEED
ROUNDS = 1800
EVAL_STRIDE = 60


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    n_clients: int
    active_clients: int
    role: str


@dataclass(frozen=True)
class Point:
    point_id: str
    scenario_id: str
    method: str
    configuration: str


SCENARIOS = (
    Scenario("m10_k10", 10, 10, "matched reference"),
    Scenario("m50_k50", 50, 50, "client-count scaling control"),
    Scenario("m50_k10", 50, 10, "primary 20-percent participation"),
    Scenario("m50_k05", 50, 5, "optional 10-percent stress test"),
)
SCENARIO_BY_ID = {scenario.scenario_id: scenario for scenario in SCENARIOS}
METHODS = (
    ("event", "event_t025_q005"),
    ("ef_topk", "ef_k05"),
    ("dense", "dense_gain_2p0"),
)
POINTS = tuple(
    Point(
        f"{scenario.scenario_id}_{method}",
        scenario.scenario_id,
        method,
        configuration,
    )
    for scenario in SCENARIOS
    for method, configuration in METHODS
)
POINT_BY_ID = {point.point_id: point for point in POINTS}
PILOT_SCENARIOS = ("m10_k10", "m50_k50", "m50_k10")
STRESS_SCENARIOS = ("m50_k05",)
P19_REFERENCE_IDS = {
    "event": "cifar_event",
    "ef_topk": "cifar_ef_quality",
    "dense": "cifar_dense_gain2",
}


def selection_seed(scenario: Scenario) -> int:
    return 91_000 + PARTITION_SEED + 100 * scenario.n_clients + scenario.active_clients


def config_for(point: Point, rounds: int = ROUNDS):
    config = replace(CIFAR_BASE, rounds=rounds, eval_stride=min(EVAL_STRIDE, rounds))
    if point.method == "dense":
        return replace(config, server_gain=2.0)
    if point.method == "ef_topk":
        return replace(config, topk_fraction=0.05)
    if point.method == "event":
        return config
    raise ValueError(point.method)


def output_paths(point: Point) -> tuple[Path, Path, Path]:
    stem = f"{point.point_id}_p{PARTITION_SEED}_r{ROUNDS}"
    return (
        OUT / f"{stem}_summary.csv",
        OUT / f"{stem}_history.csv",
        OUT / f"{stem}_activity.csv",
    )


def complete(point: Point) -> bool:
    summary, history, activity = output_paths(point)
    return summary.exists() and history.exists() and (
        point.method != "event"
        or point.scenario_id == "m10_k10"
        or activity.exists()
    )


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False, float_format="%.10g")
    os.replace(temporary, path)


def atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def schedule_for(scenario: Scenario, rounds: int = ROUNDS) -> np.ndarray:
    return participation_schedule(
        n_rounds=rounds,
        n_clients=scenario.n_clients,
        active_clients=scenario.active_clients,
        seed=selection_seed(scenario),
    )


def write_schedule(scenario: Scenario) -> Path:
    schedule = schedule_for(scenario)
    frame = pd.DataFrame(
        schedule,
        columns=[f"slot_{slot}" for slot in range(scenario.active_clients)],
    )
    frame.insert(0, "round", np.arange(1, ROUNDS + 1))
    path = OUT / f"selection_schedule_{scenario.scenario_id}.csv"
    atomic_csv(frame, path)
    return path


def metadata(point: Point, schedule: np.ndarray) -> dict[str, object]:
    scenario = SCENARIO_BY_ID[point.scenario_id]
    return {
        "point_id": point.point_id,
        "scenario_id": scenario.scenario_id,
        "scenario_role": scenario.role,
        "method": point.method,
        "configuration": point.configuration,
        "n_clients": scenario.n_clients,
        "active_clients": scenario.active_clients,
        "participation_rate": scenario.active_clients / scenario.n_clients,
        "partition_seed": PARTITION_SEED,
        "train_seed": TRAIN_SEED,
        "selection_seed": selection_seed(scenario),
        "selection_schedule_sha256": schedule_sha256(schedule),
        "rounds": ROUNDS,
    }


def prepare_data() -> None:
    download_cifar10(DATA)
    print(f"CIFAR-10 prepared at {DATA}", flush=True)


def prepare_schedules() -> None:
    for scenario in SCENARIOS:
        path = write_schedule(scenario)
        schedule = schedule_for(scenario)
        print(
            f"{scenario.scenario_id}: {path.name}, sha256={schedule_sha256(schedule)}",
            flush=True,
        )


def _reference_source(point: Point) -> tuple[Path, Path]:
    source_id = P19_REFERENCE_IDS[point.method]
    stem = f"{source_id}_s0_p{PARTITION_SEED}_r{ROUNDS}"
    return P19 / f"{stem}_summary.csv", P19 / f"{stem}_history.csv"


def import_p19_reference() -> None:
    scenario = SCENARIO_BY_ID["m10_k10"]
    schedule = schedule_for(scenario)
    checkpoint_packet = 32 * int(LAYOUT.dimension) + 64
    imported = 0
    for point in POINTS:
        if point.scenario_id != scenario.scenario_id:
            continue
        destination_summary, destination_history, _ = output_paths(point)
        if destination_summary.exists() and destination_history.exists():
            continue
        source_summary, source_history = _reference_source(point)
        if not source_summary.exists() or not source_history.exists():
            raise RuntimeError(
                f"missing frozen P19 reference for {point.method}: {source_summary}"
            )
        summary = pd.read_csv(source_summary)
        history = pd.read_csv(source_history)
        if len(summary) != 1:
            raise RuntimeError(f"invalid P19 summary: {source_summary}")
        row = summary.iloc[0]
        if int(row.partition_seed) != PARTITION_SEED or int(row.train_seed) != TRAIN_SEED:
            raise RuntimeError(f"P19 seed mismatch: {source_summary}")
        if int(row.rounds) != ROUNDS:
            raise RuntimeError(f"P19 horizon mismatch: {source_summary}")

        additions = metadata(point, schedule)
        additions.update(
            {
                "source_campaign": "p19_cifar10_cnn_final_horizons",
                "sampling_estimator": "p_i_over_pi_i",
                "initial_sync_downlink_bits": scenario.n_clients * checkpoint_packet,
                "request_bits": scenario.n_clients * ROUNDS * 32,
                "operational_unicast_hybrid_downlink_bits": int(
                    row.unicast_hybrid_downlink_bits
                ),
                "terminal_catchup_downlink_bits": 0,
                "operational_total_bits": int(row.unicast_hybrid_total_bits),
                "active_client_rounds": scenario.active_clients * ROUNDS,
                "events_per_active_client_round": float(row.coordinate_events)
                / (scenario.active_clients * ROUNDS),
                "catchup_replay_syncs": 0,
                "catchup_checkpoint_syncs": 0,
                "round_replay_syncs": int(row.replay_rounds)
                * scenario.active_clients,
                "round_checkpoint_syncs": int(row.checkpoint_rounds)
                * scenario.active_clients,
                "terminal_replay_syncs": 0,
                "terminal_checkpoint_syncs": 0,
                "min_client_participations": ROUNDS,
                "max_client_participations": ROUNDS,
            }
        )
        for column, value in additions.items():
            summary[column] = value
            history[column] = value
        atomic_csv(summary, destination_summary)
        atomic_csv(history, destination_history)
        imported += 1
        print(f"imported P19 reference: {point.point_id}", flush=True)
    print(f"P19 reference import complete: {imported} newly imported", flush=True)


def run_point(point_id: str, force: bool = False) -> str:
    point = POINT_BY_ID[point_id]
    scenario = SCENARIO_BY_ID[point.scenario_id]
    if scenario.scenario_id == "m10_k10":
        import_p19_reference()
        return f"reuse frozen P19 reference {point_id}"

    summary_path, history_path, activity_path = output_paths(point)
    if complete(point) and not force:
        return f"reuse {point_id}"

    schedule = schedule_for(scenario)
    federation = make_cifar10_federation(
        root=DATA,
        regime="strong",
        seed=PARTITION_SEED,
        n_clients=scenario.n_clients,
    )
    result = run_partial_participation(
        federation=federation,
        architecture="cifar_cnn",
        method=point.method,
        config=config_for(point),
        schedule=schedule,
        seed=TRAIN_SEED,
        record_history=True,
        record_activity=point.method == "event",
    )
    history = result.pop("history")
    if not isinstance(history, pd.DataFrame):
        raise TypeError("partial runner did not return a history DataFrame")
    activity = result.pop("activity", None)
    info = metadata(point, schedule)
    info["source_campaign"] = "p20_cifar10_partial_participation"
    result.update(info)
    atomic_csv(pd.DataFrame([result]), summary_path)
    atomic_csv(history.assign(**info), history_path)
    if point.method == "event":
        if not isinstance(activity, pd.DataFrame):
            raise TypeError("event run did not return an activity DataFrame")
        atomic_csv(activity.assign(**info), activity_path)
    return (
        f"done {point_id}: test_acc={100 * float(result['final_test_accuracy']):.2f}%, "
        f"traffic={float(result['unicast_hybrid_total_bits']) / 1e9:.3f} Gbit"
    )


def campaign(scenario_ids: tuple[str, ...], workers: int, force: bool) -> None:
    prepare_data()
    import_p19_reference()
    for scenario_id in scenario_ids:
        write_schedule(SCENARIO_BY_ID[scenario_id])
    jobs = [
        point.point_id
        for point in POINTS
        if point.scenario_id in scenario_ids
        and point.scenario_id != "m10_k10"
        and (force or not complete(point))
    ]
    total = sum(
        point.scenario_id in scenario_ids and point.scenario_id != "m10_k10"
        for point in POINTS
    )
    print(
        f"P20 {','.join(scenario_ids)}: {total - len(jobs)}/{total} new runs "
        f"complete, {len(jobs)} pending, workers={workers}",
        flush=True,
    )
    if workers == 1:
        for point_id in jobs:
            print(run_point(point_id, force=force), flush=True)
        return
    executor = ProcessPoolExecutor(max_workers=workers)
    futures = {
        executor.submit(run_point, point_id, force): point_id for point_id in jobs
    }
    try:
        for future in as_completed(futures):
            point_id = futures[future]
            try:
                print(future.result(), flush=True)
            except BaseException as error:
                for other in futures:
                    other.cancel()
                raise RuntimeError(f"P20 point failed: {point_id}") from error
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def smoke() -> None:
    prepare_data()
    scenario = Scenario("smoke_m10_k05", 10, 5, "implementation smoke test")
    schedule = participation_schedule(
        n_rounds=2,
        n_clients=scenario.n_clients,
        active_clients=scenario.active_clients,
        seed=selection_seed(scenario),
    )
    federation = make_cifar10_federation(
        root=DATA,
        regime="strong",
        seed=PARTITION_SEED,
        n_clients=scenario.n_clients,
        train_eval_size=100,
    )
    for method, configuration in METHODS:
        point = Point(f"smoke_{method}", scenario.scenario_id, method, configuration)
        result = run_partial_participation(
            federation=federation,
            architecture="cifar_cnn",
            method=method,
            config=config_for(point, rounds=2),
            schedule=schedule,
            seed=TRAIN_SEED,
            record_history=True,
            record_activity=method == "event",
        )
        print(
            f"smoke {method}: accuracy={100 * float(result['final_test_accuracy']):.2f}%, "
            f"traffic={int(result['unicast_hybrid_total_bits'])} bits",
            flush=True,
        )
    print("P20 smoke test passed", flush=True)


def selected_points(scenario_ids: tuple[str, ...]) -> list[Point]:
    return [point for point in POINTS if point.scenario_id in scenario_ids]


def load_summaries(scenario_ids: tuple[str, ...]) -> pd.DataFrame:
    points = selected_points(scenario_ids)
    missing = [point.point_id for point in points if not complete(point)]
    if missing:
        raise RuntimeError(f"P20 incomplete: {', '.join(missing)}")
    return pd.concat(
        [pd.read_csv(output_paths(point)[0]) for point in points],
        ignore_index=True,
        sort=False,
    )


def analyze(include_stress: bool = False) -> None:
    scenario_ids = PILOT_SCENARIOS + (STRESS_SCENARIOS if include_stress else ())
    runs = load_summaries(scenario_ids)
    rows = []
    for scenario_id, group in runs.groupby("scenario_id", sort=False):
        dense = group[group.method == "dense"]
        if len(dense) != 1:
            raise RuntimeError(f"missing dense reference for {scenario_id}")
        dense_row = dense.iloc[0]
        dense_accuracy = float(dense_row.final_test_accuracy)
        dense_traffic = float(dense_row.unicast_hybrid_total_bits)
        for row in group.itertuples(index=False):
            syncs = sum(
                float(getattr(row, column, 0.0) or 0.0)
                for column in (
                    "catchup_replay_syncs",
                    "catchup_checkpoint_syncs",
                    "round_replay_syncs",
                    "round_checkpoint_syncs",
                    "terminal_replay_syncs",
                    "terminal_checkpoint_syncs",
                )
            )
            checkpoint_syncs = sum(
                float(getattr(row, column, 0.0) or 0.0)
                for column in (
                    "catchup_checkpoint_syncs",
                    "round_checkpoint_syncs",
                    "terminal_checkpoint_syncs",
                )
            )
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_role": row.scenario_role,
                    "method": row.method,
                    "n_clients": int(row.n_clients),
                    "active_clients": int(row.active_clients),
                    "final_test_accuracy": float(row.final_test_accuracy),
                    "accuracy_minus_dense_points": 100
                    * (float(row.final_test_accuracy) - dense_accuracy),
                    "final_worst_class_accuracy": float(
                        row.final_worst_class_accuracy
                    ),
                    "total_traffic_gbit": float(row.unicast_hybrid_total_bits)
                    / 1e9,
                    "traffic_fraction_of_dense": float(
                        row.unicast_hybrid_total_bits
                    )
                    / dense_traffic,
                    "traffic_reduction_vs_dense": 1.0
                    - float(row.unicast_hybrid_total_bits) / dense_traffic,
                    "terminal_catchup_fraction": float(
                        getattr(row, "terminal_catchup_downlink_bits", 0.0)
                        or 0.0
                    )
                    / float(row.unicast_hybrid_total_bits),
                    "checkpoint_sync_fraction": (
                        checkpoint_syncs / syncs if syncs > 0 else 0.0
                    ),
                    "events_per_active_client_round": float(
                        getattr(row, "events_per_active_client_round", 0.0)
                        or 0.0
                    ),
                    "return_burst_ratio": float(
                        getattr(row, "return_burst_ratio", np.nan)
                    ),
                }
            )
    comparison = pd.DataFrame(rows)
    atomic_csv(comparison, OUT / "pilot_comparison.csv")
    primary_event = comparison[
        (comparison.scenario_id == "m50_k10")
        & (comparison.method == "event")
    ].iloc[0]
    gate = {
        "scenario": "m50_k10",
        "accuracy_within_one_point_of_dense": bool(
            primary_event.accuracy_minus_dense_points >= -1.0
        ),
        "less_total_traffic_than_dense": bool(
            primary_event.traffic_fraction_of_dense < 1.0
        ),
        "checkpoint_sync_fraction_below_0p9": bool(
            primary_event.checkpoint_sync_fraction < 0.9
        ),
        "candidate_for_ten_seed_extension": bool(
            primary_event.accuracy_minus_dense_points >= -1.0
            and primary_event.traffic_fraction_of_dense < 1.0
            and primary_event.checkpoint_sync_fraction < 0.9
        ),
        "note": (
            "This is a prespecified pilot screen, not a statistical conclusion. "
            "Inspect learning curves and return-burst diagnostics before extension."
        ),
    }
    atomic_json(gate, OUT / "pilot_gate.json")
    display = comparison[
        [
            "scenario_id",
            "method",
            "final_test_accuracy",
            "accuracy_minus_dense_points",
            "total_traffic_gbit",
            "traffic_fraction_of_dense",
            "checkpoint_sync_fraction",
            "return_burst_ratio",
        ]
    ]
    print(display.to_string(index=False), flush=True)
    print(json.dumps(gate, indent=2), flush=True)


def verify(include_stress: bool = False) -> None:
    scenario_ids = PILOT_SCENARIOS + (STRESS_SCENARIOS if include_stress else ())
    runs = load_summaries(scenario_ids)
    problems: list[str] = []
    for scenario_id in scenario_ids:
        scenario = SCENARIO_BY_ID[scenario_id]
        schedule = schedule_for(scenario)
        if schedule.shape != (ROUNDS, scenario.active_clients):
            problems.append(f"schedule shape mismatch: {scenario_id}")
        if any(len(np.unique(row)) != scenario.active_clients for row in schedule):
            problems.append(f"duplicate client in schedule: {scenario_id}")
        scenario_runs = runs[runs.scenario_id == scenario_id]
        if len(scenario_runs) != len(METHODS):
            problems.append(f"method grid incomplete: {scenario_id}")
        if scenario_runs.selection_schedule_sha256.nunique() != 1:
            problems.append(f"methods did not share one schedule: {scenario_id}")
        if len(scenario_runs) and (
            scenario_runs.selection_schedule_sha256.iloc[0]
            != schedule_sha256(schedule)
        ):
            problems.append(f"schedule digest mismatch: {scenario_id}")

    required = (
        "final_test_ce",
        "final_test_accuracy",
        "final_worst_class_accuracy",
        "uplink_packetized_bits",
        "unicast_hybrid_total_bits",
        "operational_total_bits",
    )
    for metric in required:
        values = pd.to_numeric(runs[metric], errors="coerce")
        if not np.isfinite(values).all():
            problems.append(f"invalid metric: {metric}")
    if np.any(
        runs.unicast_hybrid_total_bits.astype(float)
        < runs.operational_total_bits.astype(float)
    ):
        problems.append("terminally synchronized traffic is below operational traffic")

    for method, source_id in P19_REFERENCE_IDS.items():
        imported = runs[
            (runs.scenario_id == "m10_k10") & (runs.method == method)
        ]
        source = pd.read_csv(
            P19 / f"{source_id}_s0_p{PARTITION_SEED}_r{ROUNDS}_summary.csv"
        )
        if len(imported) != 1 or len(source) != 1:
            problems.append(f"reference row missing: {method}")
            continue
        for metric in ("final_test_accuracy", "unicast_hybrid_total_bits"):
            if float(imported.iloc[0][metric]) != float(source.iloc[0][metric]):
                problems.append(f"P19 reference drift: {method}/{metric}")

    for point in selected_points(scenario_ids):
        summary_path, history_path, activity_path = output_paths(point)
        history = pd.read_csv(history_path)
        if int(history["round"].max()) != ROUNDS:
            problems.append(f"history horizon mismatch: {point.point_id}")
        if point.scenario_id != "m10_k10":
            summary = pd.read_csv(summary_path).iloc[0]
            final_history = history.sort_values("round").iloc[-1]
            if int(final_history.unicast_hybrid_total_bits) != int(
                summary.unicast_hybrid_total_bits
            ):
                problems.append(f"final traffic mismatch: {point.point_id}")
        if point.method == "event" and point.scenario_id != "m10_k10":
            activity = pd.read_csv(activity_path)
            expected = ROUNDS * SCENARIO_BY_ID[point.scenario_id].active_clients
            if len(activity) != expected:
                problems.append(f"event activity rows mismatch: {point.point_id}")

    if problems:
        raise RuntimeError("\n".join(problems))
    print(
        f"P20 verification passed for {len(runs)} matched scenario/method runs",
        flush=True,
    )


def protocol(write: bool = False) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "purpose": "larger-client and partial-participation pilot",
        "dataset": "CIFAR-10",
        "architecture": "compact CNN",
        "partition": {
            "regime": "strong class skew",
            "fifty_client_design": (
                "1,000 examples per client; 550 dominant-class and 50 from "
                "each other class; five dominant clients per class"
            ),
        },
        "rounds": ROUNDS,
        "eval_stride": EVAL_STRIDE,
        "partition_seed": PARTITION_SEED,
        "train_seed": TRAIN_SEED,
        "selection": (
            "uniform without replacement each round; one schedule shared by methods"
        ),
        "sampling_estimator": "p_i / pi_i before compression",
        "state_aging": "lazy event-state decay rho ** participation_gap",
        "synchronization": (
            "selected clients catch up by ordered replay or the smaller dense "
            "checkpoint; selected clients receive the current round update; a "
            "terminal catch-up makes all registered clients exact"
        ),
        "traffic": (
            "initial model, selected-client requests, packet headers, addresses, "
            "signs or float values, catch-up, and terminal exact synchronization"
        ),
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
        "methods": [
            {"method": method, "configuration": configuration}
            for method, configuration in METHODS
        ],
        "reference_reuse": (
            "m10_k10 imports the frozen P19 seed-zero runs after seed and horizon checks"
        ),
        "extension_rule": (
            "consider ten matched seeds only after the m50_k10 pilot is stable, "
            "within one accuracy point of dense, uses less total traffic, and "
            "does not use checkpoints for at least 90 percent of nonempty syncs"
        ),
    }
    if write:
        atomic_json(payload, OUT / "protocol.json")
    else:
        print(json.dumps(payload, indent=2))
    return payload


def status() -> None:
    print("P20 larger-client and partial-participation pilot")
    for scenario in SCENARIOS:
        points = [point for point in POINTS if point.scenario_id == scenario.scenario_id]
        finished = [point.method for point in points if complete(point)]
        print(
            f"  {scenario.scenario_id}: {len(finished)}/{len(points)} {finished}",
            flush=True,
        )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-data")
    sub.add_parser("prepare-schedules")
    sub.add_parser("import-reference")
    sub.add_parser("protocol")
    sub.add_parser("status")
    sub.add_parser("smoke")
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    pilot.add_argument("--force", action="store_true")
    stress = sub.add_parser("stress")
    stress.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    stress.add_argument("--force", action="store_true")
    point = sub.add_parser("point")
    point.add_argument("--point-id", choices=sorted(POINT_BY_ID), required=True)
    point.add_argument("--force", action="store_true")
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--include-stress", action="store_true")
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--include-stress", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare-data":
        prepare_data()
    elif args.command == "prepare-schedules":
        prepare_schedules()
    elif args.command == "import-reference":
        import_p19_reference()
    elif args.command == "protocol":
        protocol()
    elif args.command == "status":
        status()
    elif args.command == "smoke":
        smoke()
    elif args.command == "pilot":
        protocol(write=True)
        campaign(PILOT_SCENARIOS, args.workers, args.force)
    elif args.command == "stress":
        protocol(write=True)
        campaign(STRESS_SCENARIOS, args.workers, args.force)
    elif args.command == "point":
        prepare_data()
        protocol(write=True)
        print(run_point(args.point_id, args.force), flush=True)
    elif args.command == "analyze":
        analyze(args.include_stress)
    elif args.command == "verify":
        verify(args.include_stress)
    else:
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
