from __future__ import annotations

import queue
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio.capture import AudioCapture
from streaming.session import RollingUtteranceSession


class AudioCaptureTests(unittest.TestCase):
    def test_half_duplex_pause_drops_echo_without_counting_queue_overflow(self):
        frames = queue.Queue()
        capture = AudioCapture(frames, {"audio": {"sample_rate": 16000, "chunk_size": 512}})
        samples = np.ones((512, 1), np.float32) * 0.1
        capture.pause()
        capture._callback(samples, 512, None, None)
        self.assertTrue(frames.empty())
        self.assertEqual(capture.paused_frames, 1)
        self.assertEqual(capture.dropped_frames, 0)
        capture.resume()
        capture._callback(samples, 512, None, None)
        self.assertEqual(frames.get_nowait().sequence, 2)

    def test_queued_mono_frame_survives_driver_buffer_reuse(self):
        frames = queue.Queue()
        capture = AudioCapture(frames, {"audio": {"sample_rate": 16000, "chunk_size": 512}})
        native_buffer = np.linspace(-0.1, 0.1, 512, dtype=np.float32).reshape(-1, 1)
        expected = native_buffer[:, 0].copy()

        capture._callback(native_buffer, 512, None, None)
        native_buffer.fill(0)
        frame = frames.get_nowait()

        np.testing.assert_array_equal(frame.samples, expected)
        self.assertFalse(np.shares_memory(frame.samples, native_buffer))
        self.assertTrue(frame.samples.flags.c_contiguous)

    def test_delayed_vad_retains_each_frame_from_reused_driver_buffer(self):
        frames = queue.Queue()
        audio_config = {
            "sample_rate": 16000,
            "chunk_size": 512,
            "vad_energy_threshold": 0.01,
            "vad_min_speech_ms": 32,
            "vad_endpoint_ms": 32,
        }
        capture = AudioCapture(frames, {"audio": audio_config})
        session = RollingUtteranceSession(audio_config, {"rolling_stride_ms": 1000})
        native_buffer = np.empty((512, 1), dtype=np.float32)
        values = (0.02, 0.03, 0.04, 0.0)
        for value in values:
            native_buffer.fill(value)
            capture._callback(native_buffer, 512, None, None)
        # Simulate the driver reusing its memory before workers consume it.
        native_buffer.fill(0)

        events = []
        while not frames.empty():
            event = session.accept(frames.get_nowait())
            if event is not None:
                events.append(event)

        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].endpoint)
        np.testing.assert_array_equal(
            events[0].audio,
            np.repeat(np.asarray(values, dtype=np.float32), 512),
        )


if __name__ == "__main__":
    unittest.main()
