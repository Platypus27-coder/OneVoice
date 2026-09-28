from __future__ import annotations

import threading
import unittest
import queue
import wave
import contextlib
import io
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import OneVoicePipeline, main as pipeline_main
from streaming.session import RollingUtteranceSession
from contracts import CommitKind


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
    def test_suppressed_translation_keeps_the_reason_without_emitting_audio(self):
        pipeline = OneVoicePipeline.__new__(OneVoicePipeline)
        pipeline.stop_event = threading.Event()
        pipeline.q_text_src = queue.Queue()
        pipeline.q_text_tgt = queue.Queue()
        pipeline._translation_log = []
        context = SimpleNamespace(translation_memory=None, risk_level="critical")
        def reject(*args):
            pipeline.stop_event.set()
            return ["missing_negation"]
        pipeline.context = SimpleNamespace(canonicalize_source=lambda text, *_: text,
                                           validate_translation=reject)
        pipeline.translator = SimpleNamespace(translate=lambda *_: "Raise the load.")
        pipeline.q_text_src.put({"text": "Do not raise the load.", "direction": "en2vi",
                                 "context": context, "decision": SimpleNamespace(kind=CommitKind.NORMAL)})
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline._mt_worker()
        self.assertTrue(pipeline.q_text_tgt.empty())
        self.assertTrue(pipeline._translation_log[0]["suppressed"])
        self.assertEqual(pipeline._translation_log[0]["validation_errors"], ["missing_negation"])
        self.assertEqual(pipeline.q_text_src.unfinished_tasks, 0)

    def test_cli_forwards_explicit_laptop_microphone_and_headphones(self):
        with mock.patch.object(sys, "argv", ["pipeline.py", "--direction", "vi2en",
                              "--input-device", "Microphone Array Realtek MME",
                              "--output-device", "Headphones Realtek WASAPI"]):
            with mock.patch("pipeline.OneVoicePipeline") as factory:
                pipeline_main()
        self.assertEqual(factory.call_args.kwargs["input_device"], "Microphone Array Realtek MME")
        self.assertEqual(factory.call_args.kwargs["output_device"], "Headphones Realtek WASAPI")
        self.assertEqual(factory.call_args.kwargs["direction"], "vi2en")

    def test_cli_forwards_output_device(self):
        with mock.patch.object(sys, "argv", ["pipeline.py", "--output-device", "3"]):
            with mock.patch("pipeline.OneVoicePipeline") as factory:
                pipeline_main()
        self.assertEqual(factory.call_args.kwargs["output_device"], 3)

    def test_tts_worker_records_repeated_device_playbacks(self):
        pipeline = OneVoicePipeline.__new__(OneVoicePipeline)
        pipeline.stop_event = threading.Event()
        pipeline.q_text_tgt = queue.Queue()
        pipeline.cfg = {"pipeline": {}}
        pipeline._stream_chunks = []
        pipeline._playback_log = []
        pipeline._latency_log = []
        pipeline._stream_playback_enabled = True
        pipeline.srt = SimpleNamespace(add_entry=mock.Mock())
        pipeline.tts = mock.Mock()
        pipeline.tts.synthesize.return_value = (np.asarray([0.1, -0.1], np.float32), 22050)
        pipeline.tts.is_silence.return_value = False
        pipeline.tts.engine_name.return_value = "espeak-ng-offline-demo"

        def play(*args, **kwargs):
            if pipeline.tts.play.call_count == 3:
                pipeline.stop_event.set()
            return {"output_device": "Headphones", "device_playback_completed": True}

        pipeline.tts.play.side_effect = play
        for _ in range(3):
            pipeline.q_text_tgt.put({
                "decision": SimpleNamespace(kind=CommitKind.NORMAL, safety_match=None, decided_at=time.perf_counter()),
                "direction": "en2vi", "text": "Hello", "translated": "Xin chào",
                "translation_route": "mt",
            })
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline._tts_worker()
        self.assertEqual([row["commit_id"] for row in pipeline._playback_log], [1, 2, 3])
        self.assertEqual(pipeline.q_text_tgt.unfinished_tasks, 0)
        self.assertEqual(pipeline.srt.add_entry.call_count, 3)

    def test_cli_forwards_vad_override_to_live_pipeline(self):
        with mock.patch.object(sys, "argv", ["pipeline.py", "--vad-energy-threshold", "0.005"]):
            with mock.patch("pipeline.OneVoicePipeline") as factory:
                pipeline_main()
        self.assertEqual(factory.call_args.kwargs["vad_energy_threshold"], 0.005)
        factory.return_value.start.assert_called_once_with()

    def test_cli_rejects_invalid_vad_threshold_before_loading_models(self):
        for value in ("0", "-0.1", "1.5", "nan"):
            with self.subTest(value=value):
                with mock.patch.object(sys, "argv", ["pipeline.py", "--vad-energy-threshold", value]):
                    with mock.patch("pipeline.OneVoicePipeline") as factory:
                        with contextlib.redirect_stderr(io.StringIO()):
                            with self.assertRaises(SystemExit) as error:
                                pipeline_main()
                self.assertEqual(error.exception.code, 2)
                factory.assert_not_called()

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
