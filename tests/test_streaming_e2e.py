from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_streaming_e2e import validate_stream_result


def safety_result() -> tuple[dict, list[dict]]:
    return (
        {
            "direction": "vi2en",
            "fatal_error": None,
            "dropped_audio_frames": 0,
            "commit_ids": [1],
            "chunks": [{"samples": 32, "engine": "safety_audio"}],
            "hypothesis_trace": [{"decision": "COMMIT_SAFETY"}],
        },
        [{"route": "safety_audio", "validation_errors": []}],
    )


class StreamingE2ETests(unittest.TestCase):
    def test_accepts_verified_safety_turn(self):
        result, latency = safety_result()
        validate_stream_result({"direction": "vi2en", "expected_route": "safety"}, result, latency)

    def test_rejects_duplicate_commit(self):
        result, latency = safety_result()
        result["commit_ids"] = [1, 1]
        result["chunks"].append({"samples": 32, "engine": "safety_audio"})
        with self.assertRaisesRegex(RuntimeError, "duplicate or reordered"):
            validate_stream_result({"direction": "vi2en", "expected_route": "safety"}, result, latency)

    def test_accepts_normal_turn(self):
        result = {
            "direction": "en2vi",
            "fatal_error": None,
            "dropped_audio_frames": 0,
            "commit_ids": [1],
            "chunks": [{"samples": 32, "engine": "espeak_ng"}],
            "hypothesis_trace": [{"decision": "COMMIT_NORMAL"}],
        }
        validate_stream_result(
            {"direction": "en2vi", "expected_route": "normal"},
            result,
            [{"route": "mt", "validation_errors": []}],
        )


if __name__ == "__main__":
    unittest.main()
