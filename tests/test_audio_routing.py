from __future__ import annotations

import contextlib
import io
from pathlib import Path
import queue
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio.capture import AudioCapture
from audio.devices import parse_input_device
from audio.playback import prepare_playback_audio


class AudioRoutingTests(unittest.TestCase):
    def test_resampling_keeps_duration_and_frequency(self):
        for source_rate in (16000, 22050, 24000):
            source = (0.1 * np.sin(2 * np.pi * 440 * np.arange(source_rate) / source_rate)).astype(np.float32)
            played = prepare_playback_audio(source, source_rate, 48000)
            self.assertEqual(len(played), 48000)
            self.assertEqual(played.dtype, np.float32)
            self.assertEqual(np.argmax(np.abs(np.fft.rfft(played))), 440)

    def test_matching_rate_does_not_change_samples(self):
        source = np.asarray([0.1, -0.1], np.float32)
        self.assertIs(prepare_playback_audio(source, 48000, 48000), source)

    def test_stereo_resampling_keeps_channels(self):
        source = np.full((240, 2), 0.1, dtype=np.float32)
        self.assertEqual(prepare_playback_audio(source, 24000, 48000).shape, (480, 2))

    def test_invalid_sample_rates_are_rejected(self):
        source = np.ones(8, np.float32)
        for invalid in (0, -1, 48000.5, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                prepare_playback_audio(source, invalid, 48000)
            with self.assertRaises(ValueError):
                prepare_playback_audio(source, 24000, invalid)

    def test_input_selector_does_not_accept_negative_or_empty_device(self):
        self.assertEqual(parse_input_device("Microphone Array Realtek MME"), "Microphone Array Realtek MME")
        self.assertEqual(parse_input_device("2"), 2)
        for invalid in (" ", "-1"):
            with self.assertRaises(ValueError):
                parse_input_device(invalid)

    def test_capture_uses_the_selected_microphone(self):
        capture = AudioCapture(queue.Queue(), {"audio": {"sample_rate": 16000,
                               "input_device": "Microphone Array Realtek MME"}})
        capture._running = True
        device = mock.Mock()
        device.query_devices.return_value = {"name": "Microphone Array", "index": 2}
        device.InputStream.return_value.__enter__ = mock.Mock()
        device.InputStream.return_value.__exit__ = mock.Mock()
        device.sleep.side_effect = lambda _: setattr(capture, "_running", False)
        with mock.patch("audio.capture.sd", device), contextlib.redirect_stdout(io.StringIO()):
            capture._stream_loop()
        device.query_devices.assert_called_once_with("Microphone Array Realtek MME", kind="input")
        self.assertEqual(device.InputStream.call_args.kwargs["device"], 2)
        self.assertIsNone(capture.error)

    def test_missing_microphone_is_an_error_not_a_silent_fallback(self):
        capture = AudioCapture(queue.Queue(), {"audio": {"sample_rate": 16000,
                               "input_device": "Microphone Array Realtek MME"}})
        device = mock.Mock()
        device.query_devices.side_effect = ValueError("No matching microphone")
        with mock.patch("audio.capture.sd", device):
            capture._stream_loop()
        self.assertIsInstance(capture.error, ValueError)
        device.InputStream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
