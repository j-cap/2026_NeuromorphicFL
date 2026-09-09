from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch.nn.utils import parameters_to_vector

import p13_cifar10_resnet14 as protocol
from neuromorphicfl.cifar10_resnet_torch import (
    CIFARResNet14,
    load_federation,
    normalize,
    parameter_count,
    semantic_parameter_groups,
)


OUT = protocol.OUT
DATA = protocol.DATA
ARCHITECTURE = "cifar_resnet14_groupnorm_torch"
METHODS = protocol.METHODS


def _device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(requested)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _copy_vector_to_model(vector: torch.Tensor, model: CIFARResNet14) -> None:
    """Copy values without rebinding parameter storage to the source vector."""
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            stop = offset + parameter.numel()
            parameter.copy_(vector[offset:stop].view_as(parameter))
            offset = stop
    if offset != vector.numel():
        raise ValueError("parameter vector length mismatch")


def _prefix(config_name: str, partition_seed: int, tag: str) -> Path:
    return OUT / f"{config_name}_p{partition_seed}_{tag}"


def _evaluate(
    model: CIFARResNet14, images: torch.Tensor, labels: torch.Tensor,
    regularization: float, batch_size: int = 512,
) -> tuple[float, float, float, float, float, list[float]]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    class_correct = torch.zeros(10, dtype=torch.int64, device=images.device)
    class_count = torch.zeros(10, dtype=torch.int64, device=images.device)
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            stop = min(start + batch_size, len(labels))
            batch_labels = labels[start:stop]
            logits = model(normalize(images[start:stop]))
            loss_sum += float(functional.cross_entropy(
                logits, batch_labels, reduction="sum"
            ).item())
            prediction = logits.argmax(dim=1)
            correct += int((prediction == batch_labels).sum().item())
            class_count += torch.bincount(batch_labels, minlength=10)
            class_correct += torch.bincount(
                batch_labels[prediction == batch_labels], minlength=10
            )
    cross_entropy = loss_sum / len(labels)
    squared_norm = sum(float((parameter * parameter).sum().item()) for parameter in model.parameters())
    objective = cross_entropy + 0.5 * regularization * squared_norm
    per_class = class_correct.float() / class_count.clamp_min(1)
    return (
        objective,
        cross_entropy,
        correct / len(labels),
        float(per_class.mean().item()),
        float(per_class.min().item()),
        [float(value) for value in per_class.cpu().tolist()],
    )


def _local_delta(
    local_model: CIFARResNet14,
    global_vector: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    config,
    rng: np.random.Generator,
) -> torch.Tensor:
    _copy_vector_to_model(global_vector, local_model)
    local_model.train()
    for _ in range(config.local_steps):
        indices = rng.integers(0, len(labels), size=config.batch_size)
        batch_indices = torch.as_tensor(indices, dtype=torch.long, device=labels.device)
        local_model.zero_grad(set_to_none=True)
        logits = local_model(normalize(images[batch_indices]))
        objective = functional.cross_entropy(logits, labels[batch_indices])
        objective = objective + 0.5 * config.regularization * sum(
            (parameter * parameter).sum() for parameter in local_model.parameters()
        )
        objective.backward()
        with torch.no_grad():
            for parameter in local_model.parameters():
                parameter.add_(parameter.grad, alpha=-config.local_lr)
    return parameters_to_vector(local_model.parameters()).detach() - global_vector


def run_point(
    config_name: str,
    partition_seed: int,
    tag: str,
    device_name: str = "cuda",
    resume: bool = True,
    force: bool = False,
    max_rounds: int | None = None,
) -> Path:
    if config_name not in protocol.CONFIGS:
        raise ValueError(config_name)
    method, original_config = protocol.CONFIGS[config_name]
    config = original_config
    if config.init_scale != 1.0:
        raise ValueError("the torch ResNet runner currently requires init_scale=1")
    if max_rounds is not None:
        from dataclasses import replace
        config = replace(config, rounds=min(max_rounds, original_config.rounds))
    device = _device(device_name)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    OUT.mkdir(parents=True, exist_ok=True)
    prefix = _prefix(config_name, partition_seed, tag)
    result_path = prefix.with_suffix(".csv")
    checkpoint_path = prefix.with_name(prefix.name + "_checkpoint.pt")
    metadata_path = prefix.with_name(prefix.name + "_run.json")
    if result_path.exists() and metadata_path.exists() and not force:
        existing_metadata = json.loads(metadata_path.read_text())
        if existing_metadata.get("status") == "completed":
            print(f"skip completed {result_path}")
            return result_path
    if force and checkpoint_path.exists():
        checkpoint_path.unlink()

    torch.manual_seed(7777)
    torch.cuda.manual_seed_all(7777)
    model = CIFARResNet14().to(device)
    local_model = CIFARResNet14().to(device)
    dimension = parameter_count(model)
    if dimension != 175258:
        raise RuntimeError(f"unexpected parameter count {dimension}")
    groups = semantic_parameter_groups(model)
    federation = load_federation(DATA, partition_seed, device)
    train_seed = 90000 + partition_seed
    rng = np.random.default_rng(train_seed)
    address_bits = math.ceil(math.log2(dimension))
    pulse_bits = address_bits + 1
    topk = max(1, round(config.topk_fraction * dimension))
    checkpoint_packet = 32 * dimension + 64
    request_bits = 32

    event_state = torch.zeros(
        (federation.n_clients, dimension), dtype=torch.float32, device=device
    )
    strom_state = torch.zeros_like(event_state)
    residual = torch.zeros_like(event_state)
    start_round = 1
    history: list[dict] = []
    activity: list[dict] = []
    counters = {
        "uplink_payload_bits": 0,
        "uplink_packetized_bits": 0,
        "messages": 0,
        "coordinate_events": 0,
        "delta_norm_sum": 0.0,
        "delta_norm_count": 0,
        "broadcast_downlink_bits": checkpoint_packet,
        "unicast_hybrid_downlink_bits": federation.n_clients * checkpoint_packet,
        "replay_rounds": 0,
        "checkpoint_rounds": 0,
    }
    elapsed_before = 0.0
    if resume and checkpoint_path.exists() and not force:
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved["config_name"] != config_name or saved["partition_seed"] != partition_seed:
            raise RuntimeError("checkpoint identity mismatch")
        if saved["config"] != asdict(config):
            raise RuntimeError("checkpoint configuration mismatch")
        _copy_vector_to_model(saved["global_vector"], model)
        event_state.copy_(saved["event_state"])
        strom_state.copy_(saved["strom_state"])
        residual.copy_(saved["residual"])
        rng.bit_generator.state = saved["numpy_rng_state"]
        counters = saved["counters"]
        history = saved["history"]
        activity = saved["activity"]
        elapsed_before = float(saved["elapsed_seconds"])
        start_round = int(saved["round"]) + 1
        print(f"resume {config_name} at round {start_round}")

    metadata = {
        "status": "running",
        "architecture": ARCHITECTURE,
        "dimension": dimension,
        "method": method,
        "config_name": config_name,
        "config": asdict(config),
        "frozen_config": asdict(original_config),
        "partition_seed": partition_seed,
        "train_seed": train_seed,
        "tag": tag,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "git_commit": _git_commit(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else None,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32 if device.type == "cuda" else None,
        "mixed_precision": False,
        "deterministic_algorithms": True,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_json(metadata_path, metadata)
    started = time.perf_counter()
    evidence_gain = 1.0 / config.local_lr

    for round_index in range(start_round, config.rounds + 1):
        events_before_round = counters["coordinate_events"]
        messages_before_round = counters["messages"]
        uplink_before_round = counters["uplink_packetized_bits"]
        global_vector = parameters_to_vector(model.parameters()).detach().clone()
        deltas = []
        for client in range(federation.n_clients):
            delta = _local_delta(
                local_model,
                global_vector,
                federation.client_images[client],
                federation.client_labels[client],
                config,
                rng,
            )
            deltas.append(delta)
            counters["delta_norm_sum"] += float(torch.linalg.vector_norm(delta).item())
            counters["delta_norm_count"] += 1

        aggregate = torch.zeros(dimension, dtype=torch.float32, device=device)
        replay_bits = 0
        if method == "event":
            event_state.mul_(config.rho)
            jump = config.jump0 * (1 + round_index / config.jump_scale) ** (-config.jump_exponent)
            group_events = np.zeros(len(groups), dtype=np.int64)
            group_clients = np.zeros(len(groups), dtype=np.int64)
            group_evidence = np.zeros(len(groups), dtype=np.float64)
            for client, delta in enumerate(deltas):
                event_state[client].add_(
                    delta, alpha=evidence_gain * float(federation.weights[client].item())
                )
                mask = event_state[client].abs() >= config.threshold
                for group_index, (_, group_slice, group_size) in enumerate(groups):
                    group_mask = mask[group_slice]
                    count = int(group_mask.sum().item())
                    group_events[group_index] += count
                    group_clients[group_index] += int(count > 0)
                    group_evidence[group_index] += float(
                        event_state[client, group_slice].abs().mean().item()
                    )
                count = int(mask.sum().item())
                if count:
                    aggregate[mask] = (
                        aggregate[mask] + jump * event_state[client, mask].sign()
                    )
                    event_state[client, mask] = 0.0
                    payload = count * pulse_bits
                    packet = payload + 64
                    counters["uplink_payload_bits"] += payload
                    counters["uplink_packetized_bits"] += packet
                    counters["messages"] += 1
                    counters["coordinate_events"] += count
                    replay_bits += packet
            for group_index, (group_name, _, group_size) in enumerate(groups):
                activity.append({
                    "round": round_index,
                    "group": group_name,
                    "parameter_count": group_size,
                    "coordinate_events": int(group_events[group_index]),
                    "active_clients": int(group_clients[group_index]),
                    "firing_fraction": float(
                        group_events[group_index] / (federation.n_clients * group_size)
                    ),
                    "mean_abs_pre_reset_evidence": float(
                        group_evidence[group_index] / federation.n_clients
                    ),
                })
        elif method == "strom":
            for client, delta in enumerate(deltas):
                strom_state[client].add_(
                    delta, alpha=evidence_gain * float(federation.weights[client].item())
                )
                mask = strom_state[client].abs() >= config.strom_threshold
                count = int(mask.sum().item())
                if count:
                    signs = strom_state[client, mask].sign()
                    aggregate[mask] = (
                        aggregate[mask] + config.strom_threshold * signs
                    )
                    strom_state[client, mask] = (
                        strom_state[client, mask] - config.strom_threshold * signs
                    )
                    payload = count * pulse_bits
                    packet = payload + 64
                    counters["uplink_payload_bits"] += payload
                    counters["uplink_packetized_bits"] += packet
                    counters["messages"] += 1
                    counters["coordinate_events"] += count
                    replay_bits += packet
        elif method == "ef_topk":
            for client, delta in enumerate(deltas):
                residual[client].add_(delta, alpha=float(federation.weights[client].item()))
                indices = torch.topk(residual[client].abs(), topk, sorted=False).indices
                values = residual[client, indices].clone()
                aggregate.scatter_add_(0, indices, values)
                residual[client, indices] = 0.0
                payload = topk * (32 + address_bits)
                packet = payload + 64
                counters["uplink_payload_bits"] += payload
                counters["uplink_packetized_bits"] += packet
                counters["messages"] += 1
                replay_bits += packet
        elif method == "sign_ef":
            for client, delta in enumerate(deltas):
                residual[client].add_(delta, alpha=float(federation.weights[client].item()))
                scale = residual[client].abs().mean()
                compressed = scale * residual[client].sign()
                aggregate.add_(compressed)
                residual[client].sub_(compressed)
                payload = dimension + 32
                packet = payload + 64
                counters["uplink_payload_bits"] += payload
                counters["uplink_packetized_bits"] += packet
                counters["messages"] += 1
                replay_bits += packet
        elif method == "dense":
            for client, delta in enumerate(deltas):
                aggregate.add_(delta, alpha=float(federation.weights[client].item()))
                payload = 32 * dimension
                packet = payload + 64
                counters["uplink_payload_bits"] += payload
                counters["uplink_packetized_bits"] += packet
                counters["messages"] += 1
                replay_bits += packet
        else:
            raise ValueError(method)

        aggregate.mul_(config.server_gain)
        server_update_norm = float(torch.linalg.vector_norm(aggregate).item())
        aggregate_nonzeros = int(torch.count_nonzero(aggregate).item())
        global_vector.add_(aggregate)
        _copy_vector_to_model(global_vector, model)
        if replay_bits <= checkpoint_packet:
            counters["replay_rounds"] += 1
        else:
            counters["checkpoint_rounds"] += 1
        downlink = min(replay_bits, checkpoint_packet)
        counters["broadcast_downlink_bits"] += downlink
        counters["unicast_hybrid_downlink_bits"] += federation.n_clients * (
            request_bits + downlink
        )

        evaluate = (
            round_index == 1
            or round_index % config.eval_stride == 0
            or round_index == config.rounds
        )
        if evaluate:
            train_metrics = _evaluate(
                model, federation.train_eval_images, federation.train_eval_labels,
                config.regularization,
            )
            test_metrics = _evaluate(
                model, federation.test_images, federation.test_labels,
                config.regularization,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = elapsed_before + time.perf_counter() - started
            history.append({
                "round": round_index,
                "train_objective": train_metrics[0],
                "train_ce": train_metrics[1],
                "test_ce": test_metrics[1],
                "test_accuracy": test_metrics[2],
                "macro_accuracy": test_metrics[3],
                "worst_class_accuracy": test_metrics[4],
                "uplink_packetized_bits": counters["uplink_packetized_bits"],
                "broadcast_downlink_bits": counters["broadcast_downlink_bits"],
                "unicast_hybrid_downlink_bits": counters["unicast_hybrid_downlink_bits"],
                "coordinate_events": counters["coordinate_events"],
                "messages": counters["messages"],
                "replay_rounds": counters["replay_rounds"],
                "checkpoint_rounds": counters["checkpoint_rounds"],
                "round_coordinate_events": counters["coordinate_events"] - events_before_round,
                "round_messages": counters["messages"] - messages_before_round,
                "round_uplink_packetized_bits": counters["uplink_packetized_bits"] - uplink_before_round,
                "server_update_norm": server_update_norm,
                "aggregate_nonzeros": aggregate_nonzeros,
                "elapsed_seconds": elapsed,
            })
            _atomic_csv(pd.DataFrame(history).assign(
                config_name=config_name, method=method,
                partition_seed=partition_seed, train_seed=train_seed, tag=tag,
            ), prefix.with_name(prefix.name + "_history.csv"))
            if activity:
                _atomic_csv(pd.DataFrame(activity).assign(
                    config_name=config_name, partition_seed=partition_seed,
                    train_seed=train_seed,
                ), prefix.with_name(prefix.name + "_activity.csv"))
            _atomic_checkpoint(checkpoint_path, {
                "round": round_index,
                "config_name": config_name,
                "partition_seed": partition_seed,
                "config": asdict(config),
                "global_vector": global_vector,
                "event_state": event_state,
                "strom_state": strom_state,
                "residual": residual,
                "numpy_rng_state": rng.bit_generator.state,
                "counters": counters,
                "history": history,
                "activity": activity,
                "elapsed_seconds": elapsed,
            })
            print(
                f"{config_name} seed={partition_seed} round={round_index}/{config.rounds} "
                f"test_acc={100 * test_metrics[2]:.2f}% elapsed={elapsed / 60:.1f} min",
                flush=True,
            )

    train_metrics = _evaluate(
        model, federation.train_eval_images, federation.train_eval_labels,
        config.regularization,
    )
    test_metrics = _evaluate(
        model, federation.test_images, federation.test_labels,
        config.regularization,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = elapsed_before + time.perf_counter() - started
    total = counters["uplink_packetized_bits"] + counters["unicast_hybrid_downlink_bits"]
    result = {
        "architecture": ARCHITECTURE,
        "method": method,
        "config_name": config_name,
        "partition_seed": partition_seed,
        "train_seed": train_seed,
        "tag": tag,
        "dimension": dimension,
        "rounds": config.rounds,
        "local_steps": config.local_steps,
        "local_lr": config.local_lr,
        "final_train_objective": train_metrics[0],
        "final_train_ce": train_metrics[1],
        "final_test_ce": test_metrics[1],
        "final_test_accuracy": test_metrics[2],
        "final_macro_accuracy": test_metrics[3],
        "final_worst_class_accuracy": test_metrics[4],
        "whole_train_objective": float(pd.DataFrame(history).train_objective.mean()),
        "uplink_payload_bits": counters["uplink_payload_bits"],
        "uplink_packetized_bits": counters["uplink_packetized_bits"],
        "broadcast_downlink_bits": counters["broadcast_downlink_bits"],
        "unicast_hybrid_downlink_bits": counters["unicast_hybrid_downlink_bits"],
        "broadcast_total_bits": counters["uplink_packetized_bits"] + counters["broadcast_downlink_bits"],
        "unicast_hybrid_total_bits": total,
        "messages": counters["messages"],
        "coordinate_events": counters["coordinate_events"],
        "events_per_message": counters["coordinate_events"] / counters["messages"] if counters["messages"] else 0.0,
        "mean_delta_norm": counters["delta_norm_sum"] / counters["delta_norm_count"],
        "replay_rounds": counters["replay_rounds"],
        "checkpoint_rounds": counters["checkpoint_rounds"],
        "rho": config.rho,
        "threshold": config.threshold,
        "jump0": config.jump0,
        "jump_exponent": config.jump_exponent,
        "topk_fraction": config.topk_fraction,
        "strom_threshold": config.strom_threshold,
        "init_scale": config.init_scale,
        "server_gain": config.server_gain,
        "elapsed_seconds": elapsed,
        "seconds_per_round": elapsed / config.rounds,
    }
    for class_index, accuracy in enumerate(test_metrics[5]):
        result[f"class_{class_index}_accuracy"] = accuracy
    _atomic_csv(pd.DataFrame([result]), result_path)
    metadata.update({
        "status": "completed",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "result_file": result_path.name,
    })
    _atomic_json(metadata_path, metadata)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(pd.DataFrame([result])[[
        "config_name", "method", "partition_seed", "final_train_ce",
        "final_test_accuracy", "final_worst_class_accuracy",
        "unicast_hybrid_total_bits", "elapsed_seconds",
    ]].to_string(index=False))
    return result_path


def select_development() -> None:
    protocol.select()


def run_frozen_method(method: str, partition_seed: int, device: str) -> None:
    selection_path = OUT / "selection.json"
    if not selection_path.exists():
        raise RuntimeError("selection.json is missing; complete and select development first")
    selection = json.loads(selection_path.read_text())
    run_point(selection["quality"][method], partition_seed, "heldout", device)


def test_campaign(device: str) -> None:
    protocol.write_protocol()
    for config_name in protocol.CONFIGS:
        run_point(config_name, protocol.DEVELOPMENT_SEED, "dev", device)
    select_development()
    for method in METHODS:
        run_frozen_method(method, protocol.PILOT_SEEDS[0], device)
    protocol.assess(OUT / "selection.json")


def full_campaign(device: str) -> None:
    if not (OUT / "selection.json").exists():
        raise RuntimeError("run the test campaign through development selection first")
    for seed in protocol.PILOT_SEEDS:
        for method in METHODS:
            run_frozen_method(method, seed, device)
    protocol.assess(OUT / "selection.json")
    gate = json.loads((OUT / "gate.json").read_text())
    if not gate["proceed_to_extension"]:
        raise RuntimeError("pilot validity gate did not pass; inspect gate.json")
    for seed in protocol.EXTENSION_SEEDS:
        for method in METHODS:
            run_frozen_method(method, seed, device)
    protocol.assess(OUT / "selection.json")


def verify_outputs() -> None:
    problems = []
    for result_path in OUT.glob("*.csv"):
        if result_path.name.endswith(("_history.csv", "_activity.csv")):
            continue
        if result_path.name in {"development_summary.csv", "heldout_summary.csv"}:
            continue
        run_path = result_path.with_name(result_path.stem + "_run.json")
        history_path = result_path.with_name(result_path.stem + "_history.csv")
        if not run_path.exists() or not history_path.exists():
            problems.append(f"missing companion for {result_path.name}")
            continue
        result = pd.read_csv(result_path)
        history = pd.read_csv(history_path)
        if len(result) != 1 or not np.isfinite(result.select_dtypes(include=[np.number])).all().all():
            problems.append(f"invalid result {result_path.name}")
        if int(history.iloc[-1]["round"]) != int(result.iloc[0]["rounds"]):
            problems.append(f"incomplete history {history_path.name}")
        if result.iloc[0]["method"] == "event":
            activity_path = result_path.with_name(result_path.stem + "_activity.csv")
            if not activity_path.exists():
                problems.append(f"missing activity {activity_path.name}")
    if problems:
        raise RuntimeError("\n".join(problems))
    manifest_files = sorted(
        path for path in OUT.iterdir()
        if path.is_file()
        and path.suffix in {".csv", ".json"}
        and path.name != "result_manifest.json"
    )
    manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in manifest_files
        ],
    }
    _atomic_json(OUT / "result_manifest.json", manifest)
    print("P13 output companions and numeric values validated")


def status() -> None:
    completed = []
    for path in OUT.glob("*_run.json"):
        metadata = json.loads(path.read_text())
        if metadata.get("status") == "completed":
            completed.append(metadata)
    result_files = [
        path for path in OUT.glob("*.csv")
        if not path.name.endswith(("_history.csv", "_activity.csv"))
        and path.name not in {"development_summary.csv", "heldout_summary.csv"}
    ]
    elapsed = [float(item["elapsed_seconds"]) for item in completed]
    print(f"completed points: {len(result_files)}")
    if elapsed:
        print(f"median point runtime: {np.median(elapsed) / 60:.1f} min")
        print(f"total completed GPU time: {np.sum(elapsed) / 3600:.2f} h")
    checkpoints = sorted(OUT.glob("*_checkpoint.pt"))
    print(f"resumable incomplete points: {len(checkpoints)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--rounds", type=int, default=2)
    point = commands.add_parser("point")
    point.add_argument("--config", choices=sorted(protocol.CONFIGS), required=True)
    point.add_argument("--partition-seed", type=int, required=True)
    point.add_argument("--tag", default="dev")
    point.add_argument("--max-rounds", type=int)
    point.add_argument("--force", action="store_true")
    commands.add_parser("select")
    heldout = commands.add_parser("heldout")
    heldout.add_argument("--method", choices=METHODS, required=True)
    heldout.add_argument("--partition-seed", type=int, required=True)
    commands.add_parser("test-campaign")
    commands.add_parser("full-campaign")
    commands.add_parser("verify")
    commands.add_parser("status")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "smoke":
        run_point(
            "dense_g10", protocol.DEVELOPMENT_SEED, "smoke",
            args.device, force=True, max_rounds=args.rounds,
        )
    elif args.command == "point":
        run_point(
            args.config, args.partition_seed, args.tag, args.device,
            force=args.force, max_rounds=args.max_rounds,
        )
    elif args.command == "select":
        select_development()
    elif args.command == "heldout":
        run_frozen_method(args.method, args.partition_seed, args.device)
    elif args.command == "test-campaign":
        test_campaign(args.device)
    elif args.command == "full-campaign":
        full_campaign(args.device)
    elif args.command == "verify":
        verify_outputs()
    elif args.command == "status":
        status()
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
