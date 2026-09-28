from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.benchmark_desktop_pipeline import (
    select_full_cases, summarize_results, validate_resume_contract,
)
from scripts.benchmark_offline_tts import load_prompts


class DesktopBenchmarkTests(unittest.TestCase):
    def test_tts_uses_target_language_for_both_directions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prompts.csv"
            path.write_text("id,vi,en\n1,Kiểm tra mũ bảo hộ.,Check the helmet.\n", encoding="utf-8")
            self.assertEqual(load_prompts(path, "vi2en", None)[0]["text"], "Check the helmet.")
            self.assertEqual(load_prompts(path, "en2vi", None)[0]["text"], "Kiểm tra mũ bảo hộ.")

    def test_full_selection_deduplicates_clean_and_never_uses_train(self):
        context = SimpleNamespace(analyze=lambda *_: SimpleNamespace(safety_candidates=[]))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "clean").mkdir()
            (root / "clean/one.wav").touch()
            manifest = root / "manifest.jsonl"
            test = {"split": "test", "clean_audio": "one.wav", "text": "Xin chào", "translation": "Hello"}
            manifest.write_text("\n".join(json.dumps(row) for row in [test, test, {**test, "split": "train", "clean_audio": "missing.wav"}]), encoding="utf-8")
            cases = select_full_cases(manifest, "vi2en", "clean", context)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0]["expected_route"], "normal")

    def test_missing_test_audio_fails_instead_of_shrinking_full_coverage(self):
        context = SimpleNamespace(analyze=lambda *_: SimpleNamespace(safety_candidates=[]))
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "manifest.jsonl"
            manifest.write_text(json.dumps({"split": "test", "audio": "missing.wav", "text": "a", "translation": "b"}), encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "not skipped"):
                select_full_cases(manifest, "vi2en", "noisy", context)

    def test_backend_change_refuses_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "contract.json"
            path.write_text(json.dumps({"backend": "espeak", "code_hash": "old"}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Refusing resume"):
                validate_resume_contract(path, {"backend": "sapi", "code_hash": "old"})

    def test_quality_metrics_include_failed_empty_outputs(self):
        rows = [
            {"status": "pass", "reference_text": "hello", "asr_text": "hello", "reference_translation": "xin chào", "translation": "xin chào", "reference_critical_fields_valid": True, "reference_translation_exact": True},
            {"status": "fail", "reference_text": "hello", "asr_text": "", "reference_translation": "xin chào", "translation": "", "reference_critical_fields_valid": False, "reference_translation_exact": False},
        ]
        summary = summarize_results(rows)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["asr_wer"], 0.5)
        self.assertEqual(summary["reference_critical_field_preservation"], 0.5)
        self.assertIsNone(summary["tts_p95_ms"])


if __name__ == "__main__":
    unittest.main()
