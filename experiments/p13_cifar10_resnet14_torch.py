from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

# Required by cuBLAS when torch deterministic algorithms are enabled. This must
# be set before importing torch or creating a CUDA context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
    device = torch.device("cuda:0" if requested == "cuda" else requested)
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


def _resume_config_matches(saved: dict, requested: dict) -> bool:
    """Allow an exact checkpoint trajectory to be extended to a later round."""
    saved_without_horizon = {key: value for key, value in saved.items() if key != "rounds"}
    requested_without_horizon = {
        key: value for key, value in requested.items() if key != "rounds"
    }
    return saved_without_horizon == requested_without_horizon


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
    keep_checkpoint: bool = False,
) -> Path:
    if config_name not in protocol.CONFIGS:
        raise ValueError(config_name)
    method, original_config = protocol.CONFIGS[config_name]
    config = original_config
    if config.init_scale != 1.0:
        raise ValueError("the torch ResNet runner currently requires init_scale=1")
    if max_rounds is not None:
        from dataclasses import replace
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        config = replace(config, rounds=max_rounds)
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
            completed_rounds = int(existing_metadata["config"]["rounds"])
            can_extend = (
                resume
                and checkpoint_path.exists()
                and config.rounds > completed_rounds
                and _resume_config_matches(existing_metadata["config"], asdict(config))
            )
            if not can_extend:
                print(f"skip completed {result_path}")
                return result_path
            print(
                f"extend completed {config_name} from {completed_rounds} "
                f"to {config.rounds} rounds"
            )
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
        if not _resume_config_matches(saved["config"], asdict(config)):
            raise RuntimeError("checkpoint configuration mismatch")
        if int(saved["round"]) >= config.rounds:
            raise RuntimeError("checkpoint is already at or beyond the requested horizon")
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
        "keep_checkpoint": keep_checkpoint,
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
    if checkpoint_path.exists() and not keep_checkpoint:
        checkpoint_path.unlink()
    elif checkpoint_path.exists():
        metadata["checkpoint_file"] = checkpoint_path.name
        _atomic_json(metadata_path, metadata)
    print(pd.DataFrame([result])[[
        "config_name", "method", "partition_seed", "final_train_ce",
        "final_test_accuracy", "final_worst_class_accuracy",
        "unicast_hybrid_total_bits", "elapsed_seconds",
    ]].to_string(index=False))
    return result_path


def select_development() -> None:
    protocol.select()


def dense_audit(device: str, rounds: int, partition_seed: int) -> None:
    """Run the dense-gain grid at an exploratory convergence horizon."""
    tag = f"dense-audit-r{rounds}"
    result_paths = []
    for config_name in ("dense_g05", "dense_g10", "dense_g15", "dense_g20"):
        result_paths.append(
            run_point(
                config_name,
                partition_seed,
                tag,
                device,
                max_rounds=rounds,
            )
        )
    summary = pd.concat(
        [pd.read_csv(result_path) for result_path in result_paths],
        ignore_index=True,
    ).sort_values("server_gain")
    summary_path = OUT / f"{tag}_seed{partition_seed}_summary.csv"
    _atomic_csv(summary, summary_path)
    print("\nDense convergence-audit summary")
    print(summary[[
        "config_name", "server_gain", "rounds", "final_train_ce",
        "final_test_accuracy", "final_worst_class_accuracy",
        "seconds_per_round", "elapsed_seconds",
    ]].to_string(index=False))
    print(f"saved {summary_path}")


def dense_horizon_audit(device: str, rounds: int, partition_seed: int) -> None:
    """Extend the selected dense development trajectory and retain its state."""
    if rounds <= protocol.BASE.rounds:
        raise ValueError(
            f"dense horizon audit must exceed the frozen {protocol.BASE.rounds} rounds"
        )
    # Keep a stable prefix so a retained completed checkpoint can be extended
    # by rerunning this command with a larger horizon.
    tag = "dense-horizon-audit"
    result_path = run_point(
        "dense_g15",
        partition_seed,
        tag,
        device,
        max_rounds=rounds,
        keep_checkpoint=True,
    )
    history_path = result_path.with_name(result_path.stem + "_history.csv")
    history = pd.read_csv(history_path).sort_values("round")
    milestones = sorted(
        set(
            [protocol.BASE.rounds, rounds]
            + [round_index for round_index in range(600, rounds + 1, 600)]
        )
    )
    milestone_rows = history[history["round"].isin(milestones)].copy()
    report_path = result_path.with_name(result_path.stem + "_milestones.csv")
    _atomic_csv(
        milestone_rows[
            [
                "round",
                "train_ce",
                "test_ce",
                "test_accuracy",
                "worst_class_accuracy",
                "unicast_hybrid_downlink_bits",
                "uplink_packetized_bits",
                "elapsed_seconds",
            ]
        ],
        report_path,
    )

    reference_path = OUT / (
        f"dense_g15_p{partition_seed}_dense-horizon-r{protocol.BASE.rounds}_history.csv"
    )
    comparison: dict[str, object] = {
        "reference_history": reference_path.name,
        "candidate_history": history_path.name,
        "reference_available": reference_path.exists(),
        "overlap_target_round": protocol.BASE.rounds,
    }
    if reference_path.exists():
        reference = pd.read_csv(reference_path).sort_values("round")
        overlap = history[history["round"] <= protocol.BASE.rounds]
        if not np.array_equal(reference["round"].to_numpy(), overlap["round"].to_numpy()):
            raise RuntimeError("existing and rerun histories use different evaluation rounds")
        continuous_columns = ("train_ce", "test_ce")
        comparison["accuracy_exact_match"] = bool(
            np.array_equal(
                reference["test_accuracy"].to_numpy(),
                overlap["test_accuracy"].to_numpy(),
            )
        )
        for column in continuous_columns:
            comparison[f"max_abs_{column}_difference"] = float(
                np.max(np.abs(reference[column].to_numpy() - overlap[column].to_numpy()))
            )
        comparison["numerically_reproduced"] = bool(
            comparison["accuracy_exact_match"]
            and all(
                float(comparison[f"max_abs_{column}_difference"]) <= 1e-6
                for column in continuous_columns
            )
        )
    comparison_path = result_path.with_name(result_path.stem + "_overlap_check.json")
    _atomic_json(comparison_path, comparison)

    print("\nDense horizon milestones")
    print(milestone_rows[["round", "train_ce", "test_ce", "test_accuracy"]].to_string(index=False))
    print(f"saved {report_path}")
    print(f"saved {comparison_path}")
    if comparison.get("reference_available") and not comparison.get("numerically_reproduced"):
        print(
            "WARNING: the rerun did not reproduce the existing 1,800-round "
            "trajectory within the audit tolerance; inspect the environment metadata"
        )


def run_frozen_method(method: str, partition_seed: int, device: str) -> None:
    selection_path = OUT / "selection.json"
    if not selection_path.exists():
        raise RuntimeError("selection.json is missing; complete and select development first")
    selection = json.loads(selection_path.read_text())
    run_point(
        selection["quality"][method], partition_seed, protocol.HELDOUT_TAG, device
    )


def _final_campaign_tasks() -> list[tuple[str, str, int]]:
    """Return frozen (method, config, seed) tasks in a stable order."""
    selection_path = OUT / "selection.json"
    if not selection_path.exists():
        raise RuntimeError("selection.json is missing; the frozen method settings are required")
    selection = json.loads(selection_path.read_text())
    missing = [method for method in METHODS if method not in selection.get("quality", {})]
    if missing:
        raise RuntimeError(f"selection.json is missing frozen methods: {missing}")
    seeds = (*protocol.PILOT_SEEDS, *protocol.EXTENSION_SEEDS)
    return [
        (method, str(selection["quality"][method]), seed)
        for seed in seeds
        for method in METHODS
    ]


def _completed_final_tasks() -> set[tuple[str, int]]:
    completed: set[tuple[str, int]] = set()
    for metadata_path in OUT.glob(f"*_{protocol.FINAL_HELDOUT_TAG}_run.json"):
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("status") == "completed"
            and int(metadata.get("config", {}).get("rounds", -1))
            == protocol.FINAL_HORIZON
        ):
            completed.add((str(metadata["config_name"]), int(metadata["partition_seed"])))
    return completed


def _run_final_subprocess(
    config_name: str,
    partition_seed: int,
    device: str,
    torch_threads: int,
) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--device", device,
        "--torch-threads", str(torch_threads),
        "point",
        "--config", config_name,
        "--partition-seed", str(partition_seed),
        "--tag", protocol.FINAL_HELDOUT_TAG,
        "--max-rounds", str(protocol.FINAL_HORIZON),
    ]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(torch_threads)
    environment["MKL_NUM_THREADS"] = str(torch_threads)
    print(f"launch {config_name} seed={partition_seed}", flush=True)
    subprocess.run(command, check=True, env=environment)


def summarize_final_campaign(require_complete: bool = True) -> Path:
    tasks = _final_campaign_tasks()
    rows = []
    missing = []
    for _, config_name, seed in tasks:
        result_path = _prefix(config_name, seed, protocol.FINAL_HELDOUT_TAG).with_suffix(".csv")
        metadata_path = result_path.with_name(result_path.stem + "_run.json")
        if not result_path.exists() or not metadata_path.exists():
            missing.append(result_path.name)
            continue
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("status") != "completed":
            missing.append(result_path.name)
            continue
        row = pd.read_csv(result_path)
        if len(row) != 1 or int(row.iloc[0]["rounds"]) != protocol.FINAL_HORIZON:
            raise RuntimeError(f"invalid final campaign result: {result_path}")
        rows.append(row)
    if require_complete and missing:
        raise RuntimeError(
            f"final campaign is incomplete: {len(missing)} of {len(tasks)} points missing"
        )
    if not rows:
        raise RuntimeError("no completed final campaign results")
    data = pd.concat(rows, ignore_index=True)
    metrics = (
        "final_train_ce",
        "final_test_ce",
        "final_test_accuracy",
        "final_worst_class_accuracy",
        "unicast_hybrid_total_bits",
        "elapsed_seconds",
    )
    summary = data.groupby(["method", "config_name"], as_index=False).agg(**{
        f"{metric}_{stat}": (metric, stat)
        for metric in metrics for stat in ("mean", "std")
    })
    seed_counts = data.groupby(
        ["method", "config_name"], as_index=False
    ).agg(completed_seeds=("partition_seed", "nunique"))
    summary = summary.merge(seed_counts, on=["method", "config_name"], validate="one_to_one")
    path = OUT / f"{protocol.FINAL_HELDOUT_TAG}_summary.csv"
    _atomic_csv(summary, path)
    print(summary.to_string(index=False))
    print(f"saved {path}")
    return path


def final_campaign(
    device: str,
    workers: int,
    torch_threads: int,
    dry_run: bool = False,
) -> None:
    """Run the frozen five-method, ten-seed campaign at 3,000 rounds."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if torch_threads < 1:
        raise ValueError("torch_threads must be positive")
    tasks = _final_campaign_tasks()
    completed = _completed_final_tasks()
    pending = [task for task in tasks if (task[1], task[2]) not in completed]
    plan = {
        "status": "dry_run" if dry_run else "running",
        "horizon": protocol.FINAL_HORIZON,
        "tag": protocol.FINAL_HELDOUT_TAG,
        "workers": workers,
        "torch_threads_per_worker": torch_threads,
        "device": device,
        "total_points": len(tasks),
        "completed_points_before_launch": len(tasks) - len(pending),
        "pending_points_before_launch": len(pending),
        "tasks": [
            {"method": method, "config_name": config, "partition_seed": seed}
            for method, config, seed in tasks
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / f"{protocol.FINAL_HELDOUT_TAG}_campaign.json"
    _atomic_json(plan_path, plan)
    print(
        f"final campaign: {len(tasks) - len(pending)}/{len(tasks)} complete, "
        f"{len(pending)} pending, workers={workers}, threads/worker={torch_threads}"
    )
    if dry_run:
        for method, config, seed in pending:
            print(f"pending method={method} config={config} seed={seed}")
        return
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_final_subprocess, config, seed, device, torch_threads
            ): (method, config, seed)
            for method, config, seed in pending
        }
        for future in as_completed(futures):
            method, config, seed = futures[future]
            try:
                future.result()
                print(f"completed method={method} config={config} seed={seed}", flush=True)
            except Exception as error:
                failures.append((method, config, seed, str(error)))
                print(
                    f"FAILED method={method} config={config} seed={seed}: {error}",
                    flush=True,
                )
    plan["status"] = "failed" if failures else "completed"
    plan["failures"] = [
        {"method": method, "config_name": config, "partition_seed": seed, "error": error}
        for method, config, seed, error in failures
    ]
    plan["completed_points_after_run"] = len(_completed_final_tasks())
    _atomic_json(plan_path, plan)
    if failures:
        raise RuntimeError(f"{len(failures)} final campaign point(s) failed; rerun to resume")
    summarize_final_campaign(require_complete=True)


def development_campaign(device: str) -> None:
    """Run the amended single-seed grid and freeze one setting per method."""
    protocol.write_protocol()
    for config_name in protocol.DEVELOPMENT_CONFIGS:
        run_point(
            config_name,
            protocol.DEVELOPMENT_SEED,
            protocol.DEVELOPMENT_TAG,
            device,
        )
    select_development()


def test_campaign(device: str) -> None:
    development_campaign(device)
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
        if result_path.name.endswith(
            ("_history.csv", "_activity.csv", "_milestones.csv", "_summary.csv")
        ):
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
        and not path.name.endswith("_summary.csv")
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
    parser.add_argument("--torch-threads", type=int, default=8)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--rounds", type=int, default=2)
    point = commands.add_parser("point")
    point.add_argument("--config", choices=sorted(protocol.CONFIGS), required=True)
    point.add_argument("--partition-seed", type=int, required=True)
    point.add_argument("--tag", default="dev")
    point.add_argument("--max-rounds", type=int)
    point.add_argument("--force", action="store_true")
    point.add_argument("--keep-checkpoint", action="store_true")
    dense_audit_parser = commands.add_parser("dense-audit")
    dense_audit_parser.add_argument("--rounds", type=int, default=900)
    dense_audit_parser.add_argument(
        "--partition-seed", type=int, default=protocol.DEVELOPMENT_SEED
    )
    dense_horizon_parser = commands.add_parser("dense-horizon")
    dense_horizon_parser.add_argument("--rounds", type=int, default=3600)
    dense_horizon_parser.add_argument(
        "--partition-seed", type=int, default=protocol.DEVELOPMENT_SEED
    )
    commands.add_parser("development-campaign")
    commands.add_parser("select")
    heldout = commands.add_parser("heldout")
    heldout.add_argument("--method", choices=METHODS, required=True)
    heldout.add_argument("--partition-seed", type=int, required=True)
    commands.add_parser("test-campaign")
    commands.add_parser("full-campaign")
    final = commands.add_parser("final-campaign")
    final.add_argument("--workers", type=int, default=2)
    final.add_argument("--dry-run", action="store_true")
    commands.add_parser("final-summary")
    commands.add_parser("verify")
    commands.add_parser("status")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    if args.command == "smoke":
        run_point(
            "dense_g10", protocol.DEVELOPMENT_SEED, "smoke",
            args.device, force=True, max_rounds=args.rounds,
        )
    elif args.command == "point":
        run_point(
            args.config, args.partition_seed, args.tag, args.device,
            force=args.force, max_rounds=args.max_rounds,
            keep_checkpoint=args.keep_checkpoint,
        )
    elif args.command == "dense-audit":
        dense_audit(args.device, args.rounds, args.partition_seed)
    elif args.command == "dense-horizon":
        dense_horizon_audit(args.device, args.rounds, args.partition_seed)
    elif args.command == "development-campaign":
        development_campaign(args.device)
    elif args.command == "select":
        select_development()
    elif args.command == "heldout":
        run_frozen_method(args.method, args.partition_seed, args.device)
    elif args.command == "test-campaign":
        test_campaign(args.device)
    elif args.command == "full-campaign":
        full_campaign(args.device)
    elif args.command == "final-campaign":
        final_campaign(
            args.device,
            workers=args.workers,
            torch_threads=args.torch_threads,
            dry_run=args.dry_run,
        )
    elif args.command == "final-summary":
        summarize_final_campaign(require_complete=False)
    elif args.command == "verify":
        verify_outputs()
    elif args.command == "status":
        status()
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
