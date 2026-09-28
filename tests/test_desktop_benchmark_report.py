from __future__ import annotations

import json
import csv
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.build_desktop_benchmark_report import html_report, overview_svg, read_complete_run


def write_reports(root: Path) -> None:
    for direction in ("vi2en", "en2vi"):
        reports = {
            "pipeline_full_noisy/summary.json": {
                "available_cases": {"test_noisy": 2, "approved_safety": 1},
                "full_selected_corpus": True, "measurement_completed": True,
                "completed_cases": 3, "requested_cases": 3,
                "passed_cases": 2, "failed_cases": 1,
                "git_revision": "fixture",
                "suites": {"test_noisy": {"samples": 2}, "approved_safety": {"samples": 1}},
                "failures": [{"case_id": "case-2", "error": "blocked translation"}],
            },
            "tts_full_test/aggregate.json": {
                "full_prompt_csv": True,
                "target_language": "en" if direction == "vi2en" else "vi",
                "samples": 2, "passed_samples": 1, "failed_samples": 1,
            },
            "profile/profile.json": {"measurement_completed": True, "passed": False},
            "tts_full_test/benchmark_contract.json": {"prompts": [{"prompt_id": "1", "text": "one"}, {"prompt_id": "2", "text": "two"}]},
            "pipeline_full_noisy/benchmark_contract.json": {"runtime_code_sha256": "fixture", "dependencies": {}, "cases": [
                {"case_id": f"case-{index}", "suite": "test_noisy" if index < 2 else "approved_safety"} for index in range(3)
            ]},
        }
        for name, report in reports.items():
            report.update({"direction": direction, "offline": True})
            path = root / direction / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report), encoding="utf-8")
        with (root / direction / "tts_full_test/predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["prompt_id", "text", "status"])
            writer.writeheader()
            writer.writerows([{"prompt_id": "1", "text": "one", "status": "pass"}, {"prompt_id": "2", "text": "two", "status": "fail"}])
        for index in range(3):
            path = root / direction / f"pipeline_full_noisy/cases/case-{index}/result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "case_id": f"case-{index}", "status": "pass" if index < 2 else "fail",
                "error": "blocked translation" if index == 2 else None,
                "suite": "test_noisy" if index < 2 else "approved_safety",
                "reference_critical_fields_valid": index == 0,
                "reference_validation_errors": [] if index == 0 else ["missing_negation"],
            }))


class DesktopBenchmarkReportTests(unittest.TestCase):
    def test_complete_measurement_can_report_failures_honestly(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_reports(root)
            data = read_complete_run(root)
            self.assertEqual(data["directions"]["vi2en"]["tts"]["failed_samples"], 1)
            self.assertFalse(data["directions"]["vi2en"]["profile"]["passed"])

    def test_smoke_is_never_published_as_full(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_reports(root)
            path = root / "vi2en/pipeline_full_noisy/summary.json"
            report = json.loads(path.read_text())
            report["full_selected_corpus"] = False
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, "sampled"):
                read_complete_run(root)

    def test_old_swapped_language_tts_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_reports(root)
            path = root / "vi2en/tts_full_test/aggregate.json"
            report = json.loads(path.read_text())
            report["target_language"] = "vi"
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, "correct output language"):
                read_complete_run(root)

    def test_missing_stage_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_reports(root)
            (root / "en2vi/profile/profile.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "missing en2vi/profile"):
                read_complete_run(root)

    def test_tts_partial_corpus_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_reports(root)
            path = root / "vi2en/tts_full_test/benchmark_contract.json"
            path.write_text(json.dumps({"prompts": [{}, {}, {}]}))
            with self.assertRaisesRegex(RuntimeError, "every contracted prompt"):
                read_complete_run(root)

    def test_missing_failures_are_not_hidden(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_reports(root)
            path = root / "vi2en/pipeline_full_noisy/summary.json"
            report = json.loads(path.read_text())
            report["failures"] = []
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, "failure list is incomplete"):
                read_complete_run(root)

    def test_reference_findings_remain_separate_from_functional_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_reports(root)
            data = read_complete_run(root)
            diagnostics = data["directions"]["vi2en"]["diagnostics"]
            self.assertEqual(diagnostics["functional_pass_but_reference_invalid"], 1)
            self.assertEqual(diagnostics["reference_error_occurrences"]["missing_negation"], 2)

    def test_missing_case_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_reports(root)
            (root / "vi2en/pipeline_full_noisy/cases/case-0/result.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "Per-case evidence"):
                read_complete_run(root)

    def test_case_reference_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_reports(root)
            path = root / "vi2en/pipeline_full_noisy/cases/case-0/result.json"
            row = json.loads(path.read_text())
            row["reference_text"] = "changed reference"
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(RuntimeError, "differs from contract"):
                read_complete_run(root)

    def test_missing_tts_prompt_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_reports(root)
            (root / "vi2en/tts_full_test/predictions.csv").unlink()
            with self.assertRaisesRegex(RuntimeError, "per-prompt TTS evidence"):
                read_complete_run(root)

    def test_mixed_runtime_revisions_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_reports(root)
            path = root / "en2vi/pipeline_full_noisy/summary.json"
            report = json.loads(path.read_text())
            report["git_revision"] = "different revision"
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, "different runtime revisions"):
                read_complete_run(root)

    def test_svg_has_separate_scales_and_successful_latency_sample_counts(self):
        reports = {
            direction: {"pipeline": {
                "suites": {"test_noisy": {"samples": 10, "passed": 8, "reference_critical_field_preservation": 0.5}},
                "routes": {"normal": {"passed": 8, "commit_to_first_audio_p95_ms": 1800}},
            }} for direction in ("vi2en", "en2vi")
        }
        svg = overview_svg({"directions": reports})
        root = ET.fromstring(svg)
        self.assertTrue(root.tag.endswith("svg"))
        self.assertIn("Tỷ lệ lượt (%)", svg)
        self.assertIn("Độ trễ p95 (ms)", svg)
        self.assertIn("n=8", svg)
        self.assertIn("1800.0 ms", svg)

    def test_html_escapes_errors_and_uses_actual_commit_policy(self):
        data = {"generated_at": "fixture", "scope": "fixture", "directions": {
            "vi2en": {
                "pipeline": {
                    "git_revision": "fixture", "tts_engine": "fixture", "normal_commit_policy": "progressive",
                    "suites": {}, "failures": [{"case_id": "fixture", "error": "<script>unsafe</script>"}],
                },
                "tts": {"passed_samples": 1, "samples": 2, "target_language": "en", "latency_p95_ms": None},
                "profile": {"passed": False},
            }
        }}
        output = html_report(data)
        self.assertNotIn("<script>unsafe</script>", output)
        self.assertIn("&lt;script&gt;unsafe&lt;/script&gt;", output)
        self.assertIn("policy progressive", output)


if __name__ == "__main__":
    unittest.main()
