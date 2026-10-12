from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from neuromorphicfl.cifar10_benchmark import class_count_matrix
from neuromorphicfl.final_baseline_campaign import (
    FinalBaselineConfig,
    run_final_baseline,
)
from neuromorphicfl.fmnist_multiclass_benchmark import MulticlassFederation
from neuromorphicfl.partial_participation import (
    participation_schedule,
    run_partial_participation,
)


class PartitionTests(unittest.TestCase):
    def test_fifty_client_strong_partition_is_exact(self) -> None:
        counts = class_count_matrix("strong", n_clients=50)
        self.assertEqual(counts.shape, (50, 10))
        np.testing.assert_array_equal(counts.sum(axis=1), np.full(50, 1000))
        np.testing.assert_array_equal(counts.sum(axis=0), np.full(10, 5000))
        self.assertEqual(int(counts[0, 0]), 550)
        self.assertTrue(np.all(counts[0, 1:] == 50))

    def test_schedule_is_reproducible_and_without_replacement(self) -> None:
        first = participation_schedule(
            n_rounds=20, n_clients=50, active_clients=10, seed=7
        )
        second = participation_schedule(
            n_rounds=20, n_clients=50, active_clients=10, seed=7
        )
        np.testing.assert_array_equal(first, second)
        self.assertTrue(all(len(np.unique(row)) == 10 for row in first))


class FullParticipationRegressionTests(unittest.TestCase):
    @staticmethod
    def federation() -> MulticlassFederation:
        rng = np.random.default_rng(12)
        client_X = tuple(
            rng.integers(0, 256, size=(3, 3, 32, 32), dtype=np.uint8)
            for _ in range(2)
        )
        client_y = tuple(
            rng.integers(0, 10, size=3, dtype=np.int64) for _ in range(2)
        )
        return MulticlassFederation(
            client_X=client_X,
            client_y=client_y,
            X_train_eval=np.concatenate(client_X),
            y_train_eval=np.concatenate(client_y),
            X_test=rng.integers(
                0, 256, size=(10, 3, 32, 32), dtype=np.uint8
            ),
            y_test=np.arange(10, dtype=np.int64),
            regime="test",
            client_class_counts=np.ones((2, 10), dtype=int),
            periods=np.ones(2, dtype=int),
            weights=np.full(2, 0.5),
        )

    def test_full_participation_matches_frozen_runner(self) -> None:
        federation = self.federation()
        base = FinalBaselineConfig(
            local_steps=1,
            local_lr=0.05,
            batch_size=2,
            rounds=2,
            eval_stride=1,
            rho=0.999,
            threshold=0.025,
            jump0=0.005,
            jump_scale=100.0,
            jump_exponent=0.2,
            topk_fraction=0.05,
            init_scale=1.0,
        )
        schedule = participation_schedule(
            n_rounds=2, n_clients=2, active_clients=2, seed=1
        )
        for method, config in (
            ("event", base),
            ("ef_topk", base),
            ("dense", replace(base, server_gain=2.0)),
        ):
            with self.subTest(method=method):
                frozen = run_final_baseline(
                    federation=federation,
                    architecture="cifar_cnn",
                    method=method,
                    config=config,
                    seed=83500,
                    record_history=True,
                )
                partial = run_partial_participation(
                    federation=federation,
                    architecture="cifar_cnn",
                    method=method,
                    config=config,
                    schedule=schedule,
                    seed=83500,
                    record_history=True,
                )
                self.assertAlmostEqual(
                    float(frozen["final_test_accuracy"]),
                    float(partial["final_test_accuracy"]),
                    places=7,
                )
                self.assertAlmostEqual(
                    float(frozen["final_test_ce"]),
                    float(partial["final_test_ce"]),
                    places=6,
                )
                self.assertEqual(
                    int(frozen["uplink_packetized_bits"]),
                    int(partial["uplink_packetized_bits"]),
                )
                self.assertEqual(
                    int(frozen["unicast_hybrid_total_bits"]),
                    int(partial["unicast_hybrid_total_bits"]),
                )


if __name__ == "__main__":
    unittest.main()
