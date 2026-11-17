"""Bounded STC implementation checks; no downloads or research training runs.

From the repository root, on Linux or Windows:
    python experiments/stc_stage1.py verify
    python experiments/stc_stage1.py smoke
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np

from neuromorphicfl.final_baseline_campaign import FinalBaselineConfig, run_final_baseline
from neuromorphicfl.fmnist_multiclass_benchmark import MulticlassFederation
from neuromorphicfl.stc import STC_PROTOCOL


def smoke() -> None:
    rng = np.random.default_rng(12)
    images = tuple(rng.integers(0, 256, size=(3, 3, 32, 32), dtype=np.uint8) for _ in range(2))
    labels = tuple(rng.integers(0, 10, size=3, dtype=np.int64) for _ in range(2))
    federation = MulticlassFederation(
        client_X=images, client_y=labels,
        X_train_eval=np.concatenate(images), y_train_eval=np.concatenate(labels),
        X_test=rng.integers(0, 256, size=(10, 3, 32, 32), dtype=np.uint8),
        y_test=np.arange(10, dtype=np.int64), regime="synthetic-smoke",
        client_class_counts=np.ones((2, 10), dtype=int),
        periods=np.ones(2, dtype=int), weights=np.array([0.5, 0.5]),
    )
    config = FinalBaselineConfig(
        local_steps=1, local_lr=0.05, batch_size=2, rounds=2,
        eval_stride=1, init_scale=1.0,
        stc_upload_fraction=0.01, stc_download_fraction=0.01,
    )
    print(f"STC implementation smoke: {STC_PROTOCOL}", flush=True)
    print("Synthetic inputs, two rounds; output is not paper evidence.", flush=True)
    for method in ("event", "dense", "stc", "strom", "ef_topk"):
        result = run_final_baseline(
            federation=federation, architecture="cifar_cnn", method=method,
            config=config, seed=83500, record_history=True,
        )
        if not np.isfinite(result["final_train_objective"]):
            raise RuntimeError(f"nonfinite smoke objective: {method}")
        if method == "stc":
            if result["stc_downlink_messages"] != config.rounds:
                raise RuntimeError("missing native STC server messages")
            if result["history"].iloc[-1]["unicast_total_bits"] != result["unicast_total_bits"]:
                raise RuntimeError("STC cumulative accounting disagrees with endpoint")
        print(
            f"{method:8s} objective={result['final_train_objective']:.6f} "
            f"uplink={result['uplink_packetized_bits']} bits "
            f"unicast_total={result['unicast_hybrid_total_bits']} bits",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "smoke"))
    args = parser.parse_args()
    if args.command == "verify":
        env = os.environ.copy()
        paths = [str(REPO / "src"), str(REPO)]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=REPO, env=env, check=True,
        )
    smoke()


if __name__ == "__main__":
    main()
