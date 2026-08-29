import unittest
from pathlib import Path

from scripts.build_benchmark_report import classify, html_report, load_aggregates, markdown_report, metric_value


class BenchmarkReportTests(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(classify("mt/candidate_vi2en/vi2en/test/raw", {"samples": 1}), "MT")
        self.assertEqual(classify("gipformer/pytorch/dev/noisy", {"samples": 1, "wer": 0.1}), "ASR")

    def test_critical_metric_supports_asr_and_mt_names(self):
        self.assertEqual(metric_value({"critical_term_recall": 0.81}, "critical_term_recall"), 0.81)
        self.assertEqual(metric_value({"critical_field_preservation": 0.99}, "critical_term_recall"), 0.99)

    def test_report_reads_aggregate_and_writes_visual_html(self):
        root = Path("tests/fixtures/benchmark_report_root")
        rows = load_aggregates(root)
        self.assertEqual(len(rows), 1)
        self.assertIn("svg", html_report(rows, "now"))
        self.assertIn("candidate", markdown_report(rows, "now"))


if __name__ == "__main__":
    unittest.main()
