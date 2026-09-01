from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from neuromorphicfl.fmnist_event_benchmark import (
    CLASS_NAMES,
    LAYOUT,
    federation_audit,
    make_binary_federation,
    run_federated_method,
)


RESULT_DIR = Path("experiments/results/14a_fmnist_binary")


def run_audit(data_dir: str | Path):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fed = make_binary_federation(
        root=data_dir,
        class_pair=(2, 4),
        regime="strong",
        seed=1400,
    )
    audit = federation_audit(fed)
    audit.to_csv(RESULT_DIR / "client_partition_strong.csv", index=False)

    dense_rows = []
    for init_scale in [0.3, 0.5, 0.8]:
        for step in [0.01, 0.02]:
            result = run_federated_method(
                federation=fed,
                method="full",
                n_ticks=160,
                seed=60606,
                init_scale=init_scale,
                step=step,
                batch_size=32,
                eval_stride=40,
            )
            dense_rows.append(result)
    dense = pd.DataFrame(dense_rows)
    dense.to_csv(RESULT_DIR / "dense_audit.csv", index=False)

    event_settings = [
        (0.10, 0.025, 0.005),
        (0.20, 0.025, 0.005),
        (0.20, 0.050, 0.005),
        (0.20, 0.025, 0.010),
        (0.10, 0.050, 0.010),
        (0.30, 0.025, 0.005),
    ]
    event_rows = []
    for gamma, threshold, jump0 in event_settings:
        result = run_federated_method(
            federation=fed,
            method="events",
            n_ticks=160,
            seed=60606,
            init_scale=0.5,
            gamma=gamma,
            threshold=threshold,
            jump0=jump0,
            batch_size=32,
            eval_stride=40,
        )
        event_rows.append(result)
    events = pd.DataFrame(event_rows)
    events.to_csv(RESULT_DIR / "event_audit.csv", index=False)

    print("=== DATA AUDIT ===")
    print(audit.to_string(index=False))
    print("=== DENSE AUDIT ===")
    print(dense.to_string(index=False))
    print("=== EVENT AUDIT ===")
    print(events.to_string(index=False))
    print(
        f"pair=2/4 ({CLASS_NAMES[2]} vs {CLASS_NAMES[4]}), "
        f"dimension={LAYOUT.dimension}"
    )


def run_full(
    data_dir: str | Path,
    *,
    init_scale: float,
    dense_step: float,
    event_gamma: float,
    event_threshold: float,
    event_jump: float,
):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    primary_rows = []
    histories = {}
    fed = make_binary_federation(
        root=data_dir,
        class_pair=(2, 4),
        regime="strong",
        seed=1400,
    )
    primary_configs = [
        ("events", {"method": "events"}),
        ("ef_0p5pct", {"method": "ef_topk", "topk_fraction": 0.005}),
        ("ef_1pct", {"method": "ef_topk", "topk_fraction": 0.01}),
        ("ef_2p5pct", {"method": "ef_topk", "topk_fraction": 0.025}),
        ("full_precision", {"method": "full"}),
    ]
    for label, config in primary_configs:
        result = run_federated_method(
            federation=fed,
            n_ticks=450,
            seed=60606,
            init_scale=init_scale,
            step=dense_step,
            gamma=event_gamma,
            threshold=event_threshold,
            jump0=event_jump,
            batch_size=32,
            eval_stride=50,
            record_history=True,
            **config,
        )
        history = result.pop("history")
        primary_rows.append({"configuration": label, **result})
        histories[label] = history
    primary = pd.DataFrame(primary_rows)
    primary.to_csv(RESULT_DIR / "primary_strong.csv", index=False)
    for label, history in histories.items():
        history.to_csv(RESULT_DIR / f"history_{label}.csv", index=False)

    # Heterogeneity sweep. Use one representative EF point to keep the
    # real-data scaling experiment computationally focused.
    heterogeneity_rows = []
    for regime in ["iid", "moderate", "strong"]:
        federation = make_binary_federation(
            root=data_dir,
            class_pair=(2, 4),
            regime=regime,
            seed=1400,
        )
        federation_audit(federation).to_csv(
            RESULT_DIR / f"client_partition_{regime}.csv", index=False
        )
        for label, config in [
            ("events", {"method": "events"}),
            ("ef_1pct", {"method": "ef_topk", "topk_fraction": 0.01}),
            ("full_precision", {"method": "full"}),
        ]:
            result = run_federated_method(
                federation=federation,
                n_ticks=320,
                seed=60606,
                init_scale=init_scale,
                step=dense_step,
                gamma=event_gamma,
                threshold=event_threshold,
                jump0=event_jump,
                batch_size=32,
                eval_stride=40,
                **config,
            )
            heterogeneity_rows.append(
                {
                    "regime": regime,
                    "configuration": label,
                    "positive_rate_std": float(
                        federation.client_positive_rates.std()
                    ),
                    **result,
                }
            )
    heterogeneity = pd.DataFrame(heterogeneity_rows)
    heterogeneity.to_csv(RESULT_DIR / "heterogeneity.csv", index=False)

    # Harder binary pair as a challenge check.
    challenge_rows = []
    challenge = make_binary_federation(
        root=data_dir,
        class_pair=(0, 6),
        regime="strong",
        seed=1400,
    )
    for label, config in [
        ("events", {"method": "events"}),
        ("ef_1pct", {"method": "ef_topk", "topk_fraction": 0.01}),
        ("full_precision", {"method": "full"}),
    ]:
        result = run_federated_method(
            federation=challenge,
            n_ticks=320,
            seed=60606,
            init_scale=init_scale,
            step=dense_step,
            gamma=event_gamma,
            threshold=event_threshold,
            jump0=event_jump,
            batch_size=32,
            eval_stride=40,
            **config,
        )
        challenge_rows.append({"configuration": label, **result})
    challenge_df = pd.DataFrame(challenge_rows)
    challenge_df.to_csv(RESULT_DIR / "challenge_0_vs_6.csv", index=False)

    print("=== PRIMARY STRONG ===")
    print(primary.to_string(index=False))
    print("=== HETEROGENEITY ===")
    print(heterogeneity.to_string(index=False))
    print("=== CHALLENGE 0 VS 6 ===")
    print(challenge_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["audit", "full"], default="audit")
    parser.add_argument("--data-dir", default="/tmp/fashion-mnist")
    parser.add_argument("--init-scale", type=float, default=0.5)
    parser.add_argument("--dense-step", type=float, default=0.02)
    parser.add_argument("--event-gamma", type=float, default=0.2)
    parser.add_argument("--event-threshold", type=float, default=0.025)
    parser.add_argument("--event-jump", type=float, default=0.005)
    args = parser.parse_args()

    if args.phase == "audit":
        run_audit(args.data_dir)
    else:
        run_full(
            args.data_dir,
            init_scale=args.init_scale,
            dense_step=args.dense_step,
            event_gamma=args.event_gamma,
            event_threshold=args.event_threshold,
            event_jump=args.event_jump,
        )
