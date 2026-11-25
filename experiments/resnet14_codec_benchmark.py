"""Synthetic CPU codec benchmark; no SGD, CIFAR data or paper results.

python experiments/resnet14_codec_benchmark.py --repeats 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))
import resnet14_baseline_development as development
import resnet14_baseline_fast_host as fast
import numpy as np


def benchmark(repeats=3):
    import torch
    from neuromorphicfl.cifar10_resnet_torch import CIFARResNet14
    sizes = tuple(p.numel() for p in CIFARResNet14().parameters())
    dimension, clients = sum(sizes), 10
    rng = np.random.default_rng(20261125)
    deltas = rng.normal(0, .003, size=(clients, dimension)).astype(np.float32)
    deltas[:, ::97] = 0  # Near-full support exercises Rice, not only full-support bypass.
    weights = np.full(clients, .1, dtype=np.float64)
    rows = []
    for key in ("dense_g1p5_a0", "ef_topk_p0p01_g4", "stc_u1_d1_g4"):
        point = development.BY_ID[key]
        timings = {"original": [], "fast": []}
        for repeat in range(repeats + 1):
            results = {}
            order = ("original", "fast") if repeat % 2 == 0 else ("fast", "original")
            for label in order:
                implementation = development.host if label == "original" else fast
                state = {"client_residuals": np.zeros((clients, dimension), dtype=np.float32),
                         "server_residual": np.zeros(dimension, dtype=np.float32)}
                began = time.perf_counter()
                update, record = implementation.step(state, deltas, weights, method=point.method,
                                                     config=development.config_for(point), sizes=sizes, rnd=1)
                elapsed = time.perf_counter() - began
                if repeat:  # One unrecorded warm-up per backend before timing.
                    timings[label].append(elapsed)
                results[label] = update, record, state
            a, b = results["original"], results["fast"]
            np.testing.assert_array_equal(a[0].view(np.uint32), b[0].view(np.uint32))
            if a[1] != b[1]:
                raise AssertionError("Faster codec changed wire modes/counters")
            for name in a[2]:
                np.testing.assert_array_equal(a[2][name].view(np.uint32), b[2][name].view(np.uint32))
        original, accelerated = (float(np.median(timings[k])) for k in ("original", "fast"))
        rows.append({"point_id": key, "original_median_seconds": original, "fast_median_seconds": accelerated,
                     "codec_speedup": original / accelerated, "samples_seconds": timings,
                     "models_residuals_and_counters_identical": True})
    return {"scope": "synthetic_CPU_codec_only_not_training_or_paper_evidence", "implementation": fast.IMPLEMENTATION,
            "platform": platform.system(), "python": platform.python_version(), "numpy": np.__version__,
            "torch": str(torch.__version__), "dimension": dimension, "clients": clients, "tensor_sizes": list(sizes),
            "seed": 20261125, "repeats": repeats, "warmup_steps_per_backend": 1,
            "dense_uses_the_identical_original_code_path": True,
            "source_sha256": {name: development.parent.text_hash(REPO / name) for name in
                              ("experiments/resnet14_codec_benchmark.py", "experiments/fast_pulse_codec.py",
                               "experiments/resnet14_baseline_fast_host.py", "experiments/resnet14_baseline_host.py",
                               "experiments/core_float_host.py", "src/neuromorphicfl/event_codec.py", "src/neuromorphicfl/stc.py")},
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    result = benchmark(args.repeats)
    if args.output is not None:
        development.parent.atomic_json(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
