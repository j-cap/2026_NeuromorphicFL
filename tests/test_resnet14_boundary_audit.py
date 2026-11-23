from copy import deepcopy
import unittest

from experiments.resnet14_boundary_audit import verify_portable_contract


class PortableContractTests(unittest.TestCase):
    def setUp(self):
        self.expected = {"parent_evidence_sha256": {"experiments/results/parent/file.csv": "digest"},
                         "source_hashes": {"experiments/runner.py": "source"}, "execution": {"device": "cuda:0"}}

    def test_separator_only_difference_returns_original_hashed_contract(self):
        recorded = deepcopy(self.expected)
        recorded["parent_evidence_sha256"] = {"experiments\\results\\parent\\file.csv": "digest"}
        self.assertIs(verify_portable_contract(recorded, self.expected), recorded)
        self.assertIn("experiments\\results\\parent\\file.csv", recorded["parent_evidence_sha256"])

    def test_changed_dependency_hash_is_rejected(self):
        recorded = deepcopy(self.expected)
        recorded["parent_evidence_sha256"]["experiments/results/parent/file.csv"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "hashes, runtime or parent evidence differ"):
            verify_portable_contract(recorded, self.expected)

    def test_changed_source_and_execution_are_not_normalized_away(self):
        for field in ("source_hashes", "execution"):
            recorded = deepcopy(self.expected)
            recorded[field] = {"changed": True}
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                verify_portable_contract(recorded, self.expected)

    def test_duplicate_canonical_dependency_names_are_rejected(self):
        recorded = deepcopy(self.expected)
        recorded["parent_evidence_sha256"]["experiments\\results\\parent\\file.csv"] = "digest"
        with self.assertRaisesRegex(RuntimeError, "Duplicate canonical"):
            verify_portable_contract(recorded, self.expected)


if __name__ == "__main__":
    unittest.main()
