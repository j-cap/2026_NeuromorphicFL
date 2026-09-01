from __future__ import annotations

from pathlib import Path
import pandas as pd

from neuromorphicfl.fmnist_event_benchmark import make_binary_federation, run_federated_method

RESULT_DIR = Path("experiments/results/14a_fmnist_binary")


def run_method(fed, label, method, **kwargs):
    result = run_federated_method(
        federation=fed,
        method=method,
        n_ticks=450,
        seed=60606,
        init_scale=0.5,
        step=0.02,
        gamma=0.3,
        threshold=0.025,
        jump0=0.0025,
        batch_size=32,
        eval_stride=50,
        **kwargs,
    )
    return {"configuration": label, **result}


if __name__ == "__main__":
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    robustness = []
    for data_seed in [1400, 1500, 1700]:
        fed = make_binary_federation(
            root="/tmp/fashion-mnist",
            class_pair=(2, 4),
            regime="strong",
            seed=data_seed,
        )
        for label, method, kwargs in [
            ("events_m3", "events", {}),
            ("ef_0p5pct", "ef_topk", {"topk_fraction": 0.005}),
            ("ef_2p5pct", "ef_topk", {"topk_fraction": 0.025}),
            ("full_precision", "full", {}),
        ]:
            robustness.append(
                {
                    "data_seed": data_seed,
                    **run_method(fed, label, method, **kwargs),
                }
            )
    robustness_df = pd.DataFrame(robustness)
    robustness_df.to_csv(RESULT_DIR / "robustness_selected.csv", index=False)

    heterogeneity = []
    for regime in ["iid", "moderate", "strong"]:
        fed = make_binary_federation(
            root="/tmp/fashion-mnist",
            class_pair=(2, 4),
            regime=regime,
            seed=1400,
        )
        for label, method, kwargs in [
            ("events_m3", "events", {}),
            ("ef_0p5pct", "ef_topk", {"topk_fraction": 0.005}),
            ("ef_2p5pct", "ef_topk", {"topk_fraction": 0.025}),
            ("full_precision", "full", {}),
        ]:
            heterogeneity.append(
                {
                    "regime": regime,
                    "positive_rate_std": float(fed.client_positive_rates.std()),
                    **run_method(fed, label, method, **kwargs),
                }
            )
    heterogeneity_df = pd.DataFrame(heterogeneity)
    heterogeneity_df.to_csv(RESULT_DIR / "heterogeneity_selected.csv", index=False)

    challenge = []
    for data_seed in [1400, 1500]:
        fed = make_binary_federation(
            root="/tmp/fashion-mnist",
            class_pair=(0, 6),
            regime="strong",
            seed=data_seed,
        )
        for label, method, kwargs in [
            ("events_m3", "events", {}),
            ("ef_0p5pct", "ef_topk", {"topk_fraction": 0.005}),
            ("ef_2p5pct", "ef_topk", {"topk_fraction": 0.025}),
            ("full_precision", "full", {}),
        ]:
            challenge.append(
                {
                    "data_seed": data_seed,
                    **run_method(fed, label, method, **kwargs),
                }
            )
    challenge_df = pd.DataFrame(challenge)
    challenge_df.to_csv(RESULT_DIR / "challenge_selected.csv", index=False)

    print("=== ROBUSTNESS STRONG 2 VS 4 ===")
    print(robustness_df.to_string(index=False))
    print("=== HETEROGENEITY 2 VS 4 ===")
    print(heterogeneity_df.to_string(index=False))
    print("=== CHALLENGE STRONG 0 VS 6 ===")
    print(challenge_df.to_string(index=False))
