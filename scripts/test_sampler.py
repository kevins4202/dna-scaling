import gzip
import importlib.util
import io
import pathlib
import types
import unittest

spec = importlib.util.spec_from_file_location(
    "sampler", pathlib.Path(__file__).with_name("sample_opengenome.py")
)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class SamplerTests(unittest.TestCase):
    def test_quotas(self):
        for total in (10000, 100000, 100000000):
            q = s.quotas(total)
            self.assertEqual(sum(q.values()), total)
            self.assertTrue(
                all(
                    abs(q[k] - total * w / sum(s.WEIGHTS.values())) < 1
                    for k, w in s.WEIGHTS.items()
                )
            )

    def test_complete_scan_deterministic_preserves_text(self):
        raw = b"".join(
            s.json.dumps({"text": ("acGTN|" * 100) + str(i)}).encode() + b"\n"
            for i in range(100)
        )
        original_list, original_request = s.list_files, s.request
        s.list_files = lambda *a: [
            {"path": "fixture_train_chunk1.jsonl.gz", "size": len(raw)}
        ]
        s.request = lambda *a: io.BytesIO(gzip.compress(raw))
        args = types.SimpleNamespace(
            seed=6200, mode="full", shards_per_category=0, smoke_scan_chars=100
        )
        try:
            first, report = s.sample_category("mrna", "train", 10000, args)
            second, _ = s.sample_category("mrna", "train", 10000, args)
            self.assertEqual(first, second)
            self.assertEqual(sum(len(x["text"]) for x in first), 10000)
            self.assertEqual(report["shards"][0]["records_scanned"], 100)
            self.assertTrue(any("acGTN|" in x["text"] for x in first))
            with self.assertRaises(ValueError):
                s.sample_category("mrna", "train", 1000000, args)
        finally:
            s.list_files, s.request = original_list, original_request


if __name__ == "__main__":
    unittest.main()
