from __future__ import annotations

import contextlib
import io
from pathlib import Path
import queue
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from scripts.run_desktop_live import LiveTextState, DesktopLiveWindow
from pipeline import OneVoicePipeline, main
from contracts import CommitKind


class DesktopLiveTests(unittest.TestCase):
    def test_partial_is_not_final_and_never_uses_a_sample_sentence(self):
        state = LiveTextState()
        state.apply({"kind": "asr", "text": "KHÔNG", "endpoint": False})
        self.assertEqual(state.raw_asr, "KHÔNG")
        self.assertEqual(state.final_asr, "")
        state.apply({"kind": "asr", "text": "KHÔNG ĐI", "hypothesis": "Không đi", "endpoint": True})
        self.assertEqual(state.final_asr, "Không đi")

    def test_translation_stays_paired_with_its_source_when_asr_moves_ahead(self):
        state = LiveTextState()
        state.apply({"kind": "translation", "source": "Máy chạy.", "text": "The machine runs."})
        state.apply({"kind": "asr", "text": "CÂU TIẾP THEO", "endpoint": True})
        self.assertEqual(state.source, "Máy chạy.")
        self.assertEqual(state.translation, "The machine runs.")

    def test_suppressed_translation_is_not_hidden_from_the_listener(self):
        state = LiveTextState()
        state.apply({"kind": "translation", "source": "Không nâng.", "text": "Raise it.", "suppressed": True})
        self.assertTrue(state.suppressed)
        self.assertEqual(state.translation, "Raise it.")

    def test_observer_failure_does_not_kill_pipeline(self):
        pipeline = OneVoicePipeline.__new__(OneVoicePipeline)
        pipeline.direction = "vi2en"
        pipeline.event_callback = mock.Mock(side_effect=RuntimeError("UI unavailable"))
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline._emit_event("asr", text="Máy chạy")
        self.assertEqual(pipeline.event_callback.call_args.args[0]["text"], "Máy chạy")

    def test_cli_forwards_half_duplex_flag(self):
        with mock.patch.object(sys, "argv", ["pipeline.py", "--pause-mic-during-playback"]), \
             mock.patch("pipeline.OneVoicePipeline") as factory:
            main()
        self.assertTrue(factory.call_args.kwargs["pause_mic_during_playback"])

    def _tts_pipeline(self):
        pipeline = OneVoicePipeline.__new__(OneVoicePipeline)
        pipeline.direction = "vi2en"
        pipeline.profile = "edge"
        pipeline.cfg = {"pipeline": {}}
        pipeline.stop_event = threading.Event()
        pipeline.q_text_tgt = queue.Queue()
        pipeline._stream_chunks = []
        pipeline._playback_log = []
        pipeline._latency_log = []
        pipeline._stream_playback_enabled = True
        pipeline.pause_mic_during_playback = True
        pipeline.capture = mock.Mock()
        pipeline.safety_audio = None
        pipeline.srt = mock.Mock()
        pipeline.tts = mock.Mock()
        pipeline.tts.synthesize.return_value = (np.asarray([0.1, -0.1], np.float32), 22050)
        pipeline.tts.is_silence.return_value = False
        pipeline.tts.engine_name.return_value = "windows-sapi-offline"
        pipeline.q_text_tgt.put({"text": "Máy chạy", "translated": "The machine runs.",
            "direction": "vi2en", "translation_route": "mt",
            "decision": SimpleNamespace(kind=CommitKind.NORMAL, safety_match=None, decided_at=0.0)})
        return pipeline

    def test_half_duplex_resumes_microphone_even_when_playback_fails(self):
        pipeline = self._tts_pipeline()
        pipeline.tts.play.side_effect = RuntimeError("device disconnected")
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline._run_worker("TTS", pipeline._tts_worker)
        pipeline.capture.pause.assert_called_once()
        pipeline.capture.resume.assert_called_once()
        self.assertIsInstance(pipeline._fatal_error, RuntimeError)
        self.assertEqual(pipeline.q_text_tgt.unfinished_tasks, 0)

    def test_stop_during_synthesis_does_not_start_playback_afterwards(self):
        pipeline = self._tts_pipeline()
        def synthesize(*args, **kwargs):
            pipeline.stop_event.set()
            return np.asarray([0.1, -0.1], np.float32), 22050
        pipeline.tts.synthesize.side_effect = synthesize
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline._tts_worker()
        pipeline.tts.play.assert_not_called()
        pipeline.capture.pause.assert_not_called()

    def test_cancelled_loading_never_opens_microphone(self):
        pipeline = OneVoicePipeline.__new__(OneVoicePipeline)
        pipeline.stop_event = threading.Event()
        pipeline.stop_event.set()
        pipeline.load_models = mock.Mock()
        pipeline._save_reports = mock.Mock()
        pipeline.capture = mock.Mock()
        pipeline.start()
        pipeline.capture.start.assert_not_called()

    def test_gui_wires_real_pipeline_and_per_run_playback_without_config_writes(self):
        app = DesktopLiveWindow.__new__(DesktopLiveWindow)
        app.args = SimpleNamespace(bundle_dir=Path("bundle"), report_dir=Path("reports"),
            input_device="Microphone Array MME", output_device="Speakers Realtek WASAPI",
            windows_en_voice="Microsoft Zira Desktop", windows_en_rate=-2)
        app.events = queue.SimpleQueue()
        app.stop_requested = threading.Event()
        with mock.patch("os.chdir"), mock.patch.dict(sys.modules, {"onnxruntime": mock.Mock(), "sounddevice": mock.Mock()}), \
             mock.patch("pipeline.OneVoicePipeline") as factory:
            app.run_pipeline(0.015)
        self.assertEqual(factory.call_args.kwargs["direction"], "vi2en")
        self.assertTrue(factory.call_args.kwargs["offline"])
        self.assertTrue(factory.call_args.kwargs["pause_mic_during_playback"])
        factory.return_value.tts.playback_options.update.assert_called_once_with(gain=0.7, stereo=True, latency="high")
        factory.return_value.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
