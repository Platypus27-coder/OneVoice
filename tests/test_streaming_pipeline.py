from __future__ import annotations

import threading
import unittest
import queue
import wave
import contextlib
import io
from pathlib import Path
import sys
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import OneVoicePipeline
from streaming.session import RollingUtteranceSession


class _CaptureStub:
    dropped_frames = 0

    def stop(self) -> None:
        return None


class _SoundFileStub:
    @staticmethod
    def read(path, dtype="float32", always_2d=False):
        with wave.open(str(path), "rb") as handle:
            samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
            return (samples.astype(np.float32) / 32768.0, handle.getframerate())


class StreamingPipelineTests(unittest.TestCase):
    def test_stream_file_submits_fixed_frames_and_flushes_endpoint(self):
        pipeline = OneVoicePipeline.__new__(OneVoicePipeline)
        pipeline.direction = "vi2en"
        pipeline.profile = "development"
        pipeline.offline = True
        pipeline.report_dir = None
        pipeline.cfg = {
            "audio": {
                "sample_rate": 16000,
                "chunk_size": 512,
                "vad_endpoint_ms": 64,
            },
            "pipeline": {
                "queue_maxsize": 2,
                "rolling_stride_ms": 64,
                "rolling_window_ms": 64,
            },
        }
        pipeline.stop_event = threading.Event()
        pipeline._fatal_error = None
        pipeline._worker_threads = []
        pipeline._stream_playback_enabled = True
        pipeline._stream_chunks = []
        pipeline._stream_trace = []
        pipeline._latency_log = []
        pipeline.q_audio_raw = queue.Queue(maxsize=2)
        pipeline.q_audio_clean = queue.Queue(maxsize=2)
        pipeline.q_text_src = queue.Queue(maxsize=2)
        pipeline.q_text_tgt = queue.Queue(maxsize=2)
        pipeline.capture = _CaptureStub()
        pipeline.streaming_session = RollingUtteranceSession(
            pipeline.cfg["audio"], pipeline.cfg["pipeline"]
        )
        from streaming.semantic_commit import (
            RollingHypothesisAssembler,
            SemanticCommitController,
            StablePrefixAligner,
        )
        from utils.srt_generator import SRTGenerator

        pipeline.aligner = StablePrefixAligner()
        pipeline.hypothesis_assembler = RollingHypothesisAssembler()
        pipeline.committer = SemanticCommitController()
        pipeline.srt = SRTGenerator(bilingual=True)
        pipeline.load_models = lambda: None
        pipeline._save_reports = lambda: None

        # Keep the worker graph real, but consume frames without loading models;
        # the production path uses the same bounded queue and shutdown logic.
        def consume_raw() -> None:
            while not pipeline.stop_event.is_set() or not pipeline.q_audio_raw.empty():
                try:
                    pipeline.q_audio_raw.get(timeout=0.01)
                except Exception:
                    continue
                pipeline.q_audio_raw.task_done()

        pipeline._denoise_worker = consume_raw
        temporary = Path(__file__).resolve().parent / ".tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        input_path = temporary / "streaming_pipeline_input.wav"
        report_dir = temporary / "streaming-pipeline-report"
        report_dir.mkdir(parents=True, exist_ok=True)
        pipeline.report_dir = report_dir
        pcm = (np.ones(1024, dtype=np.float32) * 0.1 * 32767).astype(np.int16)
        try:
            with wave.open(str(input_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(pcm.tobytes())
            with mock.patch.dict(sys.modules, {"soundfile": _SoundFileStub}):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = pipeline.stream_file(str(input_path))
        finally:
            input_path.unlink(missing_ok=True)

        self.assertEqual(result["frame_samples"], 512)
        self.assertEqual(result["frame_ms"], 32.0)
        self.assertEqual(result["frames_submitted"], 5)
        self.assertEqual(result["commits"], 0)
        self.assertEqual(result["dropped_audio_frames"], 0)
        self.assertGreaterEqual(result["complete_turn_ms"], 0.0)
        self.assertTrue((report_dir / "stream_result.json").is_file())


if __name__ == "__main__":
    unittest.main()
