from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.run_release_e2e import select_cases, select_safety_audio_cases, sha256, valid_resume


ROOT = Path(__file__).resolve().parents[1]


class _Context:
    def analyze(self, text: str, direction: str):
        del direction
        return SimpleNamespace(safety_candidates=[object()] if text.startswith("STOP") else [])


class ReleaseE2ETests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "tests" / ".tmp" / "release-e2e"
        shutil.rmtree(self.root, ignore_errors=True)
        (self.root / "clean").mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_case_selection_is_split_route_and_audio_unique(self):
        for name in ("normal.wav", "safety.wav"):
            (self.root / "clean" / name).write_bytes(b"wav-placeholder")
        manifest = self.root / "manifest.jsonl"
        rows = [
            {
                "utterance_id": "n1",
                "split": "test",
                "language": "vi",
                "clean_audio": "normal.wav",
                "text": "Kiểm tra dây đai",
                "translation": "Check the harness",
            },
            {
                "utterance_id": "s1",
                "split": "test",
                "language": "vi",
                "clean_audio": "safety.wav",
                "text": "STOP ngay",
                "translation": "Stop now",
            },
        ]
        manifest.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        cases = select_cases(manifest, "vi2en", _Context(), 1)
        self.assertEqual([case["expected_route"] for case in cases], ["normal", "safety"])
        self.assertEqual(len({case["input_path"] for case in cases}), 2)

    def test_resume_requires_matching_output_checksum(self):
        output = self.root / "output.wav"
        result = self.root / "result.json"
        output.write_bytes(b"verified")
        result.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "output_path": str(output),
                    "output_sha256": sha256(output),
                }
            ),
            encoding="utf-8",
        )
        self.assertTrue(valid_resume(result))
        output.write_bytes(b"corrupt")
        self.assertFalse(valid_resume(result))

    def test_safety_cases_are_selected_from_verified_audio_manifest(self):
        safety_dir = self.root / "safety"
        safety_dir.mkdir()
        (safety_dir / "S1_vi2en.wav").write_bytes(b"en")
        (safety_dir / "S1_en2vi.wav").write_bytes(b"vi")
        manifest = safety_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "approval_id": "review-v1",
                    "entries": [
                        {"safety_id": "S1", "direction": "vi2en", "path": "S1_vi2en.wav"},
                        {"safety_id": "S1", "direction": "en2vi", "path": "S1_en2vi.wav"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        source = safety_dir / "review.csv"
        source.write_text("safety_id,vi,en\nS1,Dừng lại,Stop\n", encoding="utf-8")
        cases = select_safety_audio_cases(manifest, source, "vi2en", 1)
        self.assertEqual(cases[0]["safety_id"], "S1")
        self.assertTrue(cases[0]["input_path"].endswith("S1_en2vi.wav"))


if __name__ == "__main__":
    unittest.main()
