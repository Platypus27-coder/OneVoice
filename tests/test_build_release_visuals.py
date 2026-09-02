import json
import unittest
from pathlib import Path

from scripts.build_release_visuals import _rows, build_svg


class ReleaseVisualTests(unittest.TestCase):
    def test_summary_produces_current_release_rows(self):
        summary = json.loads(Path("summary.json").read_text(encoding="utf-8"))
        rows = _rows(summary)
        self.assertEqual(len(rows), 10)
        self.assertIn("VI→EN ASR / clean", {row["label"] for row in rows})
        self.assertIn("EN→VI MT / safety", {row["label"] for row in rows})

    def test_svg_contains_three_distinct_metrics_and_gate(self):
        svg = build_svg({"generated_at": "now", "artifacts": [
            {"family": "ASR", "direction": "vi2en", "audio": "clean", "samples": 1,
             "critical_term_recall": 0.98, "wer": 0.02, "latency_p95_ms": 120},
        ]})
        self.assertIn("Critical preservation", svg)
        self.assertIn("Error rate", svg)
        self.assertIn("p95 model-stage latency", svg)
        self.assertIn("95% gate", svg)


if __name__ == "__main__":
    unittest.main()
