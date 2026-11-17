from __future__ import annotations

import hashlib
import unittest

from experiments.stc_cifar10_development_audit import recover_recorded_bytes


class TransportAuditTests(unittest.TestCase):
    def test_windows_bytes_are_recovered_after_git_normalizes_to_lf(self):
        original = b"name,value\r\npoint,1.23456789\r\n"
        expected = hashlib.sha256(original).hexdigest()
        recovered, kind = recover_recorded_bytes(original.replace(b"\r\n", b"\n"), expected)
        self.assertEqual(recovered, original)
        self.assertEqual(kind, "CRLF")

    def test_lf_bytes_are_recovered_after_windows_checkout(self):
        original = b"name,value\npoint,1.23456789\n"
        expected = hashlib.sha256(original).hexdigest()
        recovered, kind = recover_recorded_bytes(original.replace(b"\n", b"\r\n"), expected)
        self.assertEqual(recovered, original)
        self.assertEqual(kind, "LF")

    def test_numeric_edits_are_rejected(self):
        original = b"name,value\r\npoint,1.23456789\r\n"
        expected = hashlib.sha256(original).hexdigest()
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            recover_recorded_bytes(b"name,value\npoint,1.23456788\n", expected)


if __name__ == "__main__":
    unittest.main()
