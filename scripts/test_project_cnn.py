import json
import tempfile
import unittest
from pathlib import Path

from cnn.release import RELEASE, verify_integrity


class FrozenReleaseTests(unittest.TestCase):
    def test_original_artifact_is_intact_and_scope_is_explicit(self):
        manifest = verify_integrity()
        self.assertEqual(
            manifest["files_sha256"]["model.pt"],
            "33621e4a90aafbb6c925ff52f7b6c45d74301a66f902cc561a05b83bc3da14b5",
        )
        self.assertEqual(
            manifest["validation_status"], "fixed_diagnostic_limited_sensitivity"
        )
        self.assertFalse(manifest["prespecified_sensitivity_gate_passed"])

    def test_modified_weight_cannot_be_loaded_as_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "version": "cnn-v1",
                "frozen": True,
                "files_sha256": {"model.pt": "original"},
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "model.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_integrity(root)

    def test_reference_ids_are_unique_and_reserved_for_test(self):
        rows = [
            json.loads(line)
            for line in (RELEASE / "evaluation-prompts.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(rows), 160)
        self.assertEqual(len({r["id"] for r in rows}), len(rows))
        self.assertEqual(len({r["group_id"] for r in rows}), 135)
        self.assertTrue(all(r["fold"] == "test" for r in rows))
        self.assertTrue(
            all(len(r["prefix"]) == 256 and len(r["text"]) == 512 for r in rows)
        )


if __name__ == "__main__":
    unittest.main()
