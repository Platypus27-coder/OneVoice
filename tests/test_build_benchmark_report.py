import unittest
from pathlib import Path

from scripts.benchmark_selection import select_release_rows
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

    def test_release_profile_excludes_historical_runs(self):
        rows = [
            {"report": "denoiser/passthrough/clean", "family": "ASR"},
            {"report": "gipformer_vi_adaptation_v1/pytorch_finetuned_dev_noisy", "family": "ASR"},
        ]
        selected = select_release_rows(rows, "release")
        self.assertEqual([row["report"] for row in selected], ["denoiser/passthrough/clean"])
        self.assertIn("release_label", selected[0])

    def test_report_profile_is_explicit(self):
        rows = [{"family": "ASR", "report": "denoiser/passthrough/clean", "samples": 1}]
        self.assertIn("report (release)", markdown_report(rows, "now", "release"))
        self.assertIn("Current runtime", html_report(rows, "now", "release"))


if __name__ == "__main__":
    unittest.main()
