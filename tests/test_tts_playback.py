from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tts.tts_engine import TTSEngine
from scripts.check_desktop_audio import run_check
from audio.devices import parse_output_device


def config() -> dict:
    return {"audio": {"sample_rate": 16000, "output_device": 3},
            "tts": {"offline_engine": "pyttsx3", "en_speed": 1.0},
            "profiles": {"edge": {"tts_tier": "edge"}}}


class SystemTTSTests(unittest.TestCase):
    def test_windows_loads_native_backend_for_both_languages_without_pyttsx3(self):
        broken_pyttsx3 = SimpleNamespace(init=mock.Mock(side_effect=AssertionError("must not enter SAPI loop")))
        with mock.patch("sys.platform", "win32"), mock.patch.object(TTSEngine, "_find_espeak_executable", return_value="espeak-ng.exe"), \
             mock.patch.dict(sys.modules, {"pyttsx3": broken_pyttsx3}), contextlib.redirect_stdout(io.StringIO()):
            engine = TTSEngine(config(), profile="edge", offline=True)
            engine.load()
        broken_pyttsx3.init.assert_not_called()
        self.assertEqual(engine.engine_name("vi2en"), "espeak-ng-offline-demo")
        self.assertEqual(engine.engine_name("en2vi"), "espeak-ng-offline-demo")

    def test_windows_fails_at_startup_without_native_tts(self):
        with mock.patch("sys.platform", "win32"), mock.patch.object(TTSEngine, "_find_espeak_executable", return_value=None), \
             contextlib.redirect_stdout(io.StringIO()):
            for direction in ("vi2en", "en2vi"):
                engine = TTSEngine(config(), profile="edge", offline=True)
                with self.assertRaisesRegex(RuntimeError, "Windows system TTS requires local"):
                    engine.load(direction)

    def test_explicit_conda_python_discovers_local_espeak_without_path_hook(self):
        with mock.patch("sys.platform", "win32"), mock.patch("shutil.which", return_value=None), \
             mock.patch("sys.prefix", str(Path("environment"))), mock.patch.object(Path, "is_file", return_value=True):
            found = TTSEngine._find_espeak_executable()
        self.assertEqual(Path(found), Path("environment/espeak-ng-runtime/eSpeak NG/espeak-ng.exe"))

    def test_native_router_can_synthesize_repeated_calls_with_explicit_language(self):
        engine = TTSEngine(config(), profile="edge", offline=True)
        engine._en_tts_engine = engine._vi_tts_engine_name = "espeak-ng-offline-demo"
        engine._en_tts_executable = engine._vi_tts_executable = "espeak-ng"
        with mock.patch.object(engine, "_synthesize_espeak", return_value=(np.ones(8, np.float32), 22050)) as synth, \
             contextlib.redirect_stdout(io.StringIO()):
            for _ in range(5):
                engine.synthesize("Hello", "vi2en")
                engine.synthesize("Xin chào", "en2vi")
        self.assertEqual(synth.call_count, 10)
        self.assertEqual([call.args[2] for call in synth.call_args_list], ["en-us", "vi"] * 5)

    def test_native_subprocess_uses_utf8_stdin_timeout_and_cleans_wav(self):
        engine = TTSEngine(config(), profile="edge", offline=True)
        created = []

        def write_wav(command, **kwargs):
            self.assertEqual(kwargs["input"], "Kiểm tra mũ bảo hộ.")
            self.assertEqual(kwargs["encoding"], "utf-8")
            self.assertEqual(kwargs["timeout"], 15.0)
            self.assertIn("env", kwargs)
            self.assertIn("--stdin", command)
            destination = command[command.index("-w") + 1]
            created.append(Path(destination))
            with wave.open(destination, "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(22050)
                handle.writeframes(np.asarray([2000, -2000], dtype="<i2").tobytes())

        with mock.patch("subprocess.run", side_effect=write_wav):
            audio, sr = engine._synthesize_espeak("Kiểm tra mũ bảo hộ.", "espeak-ng", "vi", 165)
        self.assertEqual(sr, 22050)
        self.assertFalse(engine.is_silence(audio))
        self.assertFalse(created[0].exists())

    def test_portable_espeak_sets_child_data_path_without_changing_parent(self):
        executable = Path("portable/eSpeak NG/espeak-ng.exe")
        with mock.patch.dict(os.environ, {"ESPEAK_DATA_PATH": "wrong-data", "ONEVOICE_TEST": "keep"}), \
             mock.patch.object(Path, "is_dir", return_value=True):
            child_env = TTSEngine._espeak_environment(str(executable))
            self.assertEqual(child_env["ESPEAK_DATA_PATH"], str(executable.resolve().parent / "espeak-ng-data"))
            self.assertEqual(child_env["ONEVOICE_TEST"], "keep")
            self.assertEqual(os.environ["ESPEAK_DATA_PATH"], "wrong-data")

    def test_system_espeak_preserves_existing_environment_without_sibling_data(self):
        with mock.patch.dict(os.environ, {"ESPEAK_DATA_PATH": "system-data"}), \
             mock.patch.object(Path, "is_dir", return_value=False):
            self.assertEqual(TTSEngine._espeak_environment("espeak-ng")["ESPEAK_DATA_PATH"], "system-data")

    def test_native_crash_is_visible_and_cleans_partial_wav(self):
        engine = TTSEngine(config(), profile="edge", offline=True)
        failure = subprocess.CalledProcessError(3221225477, "espeak-ng", stderr="")
        with mock.patch("subprocess.run", side_effect=failure) as process:
            with self.assertRaisesRegex(RuntimeError, "Native eSpeak failed.*0xC0000005"):
                engine._synthesize_espeak("Xin chào", "espeak-ng", "vi", 165)
        command = process.call_args.args[0]
        self.assertFalse(Path(command[command.index("-w") + 1]).exists())

    def test_native_timeout_is_visible_and_cleans_partial_wav(self):
        engine = TTSEngine(config(), profile="edge", offline=True)
        with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("espeak-ng", 15)) as process:
            with self.assertRaisesRegex(RuntimeError, "exceeded 15s"):
                engine._synthesize_espeak("Hello", "espeak-ng", "en-us", 160)
        command = process.call_args.args[0]
        self.assertFalse(Path(command[command.index("-w") + 1]).exists())

    def test_timeout_setting_must_be_positive_and_finite(self):
        for value in (0, -1, float("nan"), float("inf")):
            cfg = config()
            cfg["tts"]["synthesis_timeout_s"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                TTSEngine(cfg, profile="edge", offline=True)


class PlaybackTests(unittest.TestCase):
    def test_device_selector_accepts_indices_and_names_without_silent_fallback(self):
        self.assertEqual(parse_output_device("3"), 3)
        self.assertEqual(parse_output_device(" GZUT-MUSIC MME "), "GZUT-MUSIC MME")
        for invalid in ("-1", " "):
            with self.assertRaises(ValueError):
                parse_output_device(invalid)

    def test_named_headset_is_resolved_to_its_current_index(self):
        device = mock.Mock()
        device.query_devices.return_value = {"name": "Headphones (GZUT-MUSIC)", "index": 7}
        device.wait.return_value = None
        cfg = config()
        cfg["audio"]["output_device"] = "GZUT-MUSIC MME"
        engine = TTSEngine(cfg, profile="edge", offline=True)
        audio = np.asarray([0.1, -0.1], np.float32)
        with mock.patch("tts.tts_engine.sd", device), contextlib.redirect_stdout(io.StringIO()):
            engine.play(audio, 22050)
        device.query_devices.assert_called_once_with("GZUT-MUSIC MME", kind="output")
        device.play.assert_called_once_with(audio, samplerate=22050, device=7)

    def test_missing_named_headset_does_not_fall_back_to_default_speakers(self):
        device = mock.Mock()
        device.query_devices.side_effect = ValueError("No output device matching GZUT-MUSIC")
        cfg = config()
        cfg["audio"]["output_device"] = "GZUT-MUSIC MME"
        engine = TTSEngine(cfg, profile="edge", offline=True)
        with mock.patch("tts.tts_engine.sd", device), self.assertRaisesRegex(RuntimeError, "No output device matching"):
            engine.play(np.ones(8, np.float32), 22050)
        device.play.assert_not_called()

    def test_silent_or_non_finite_audio_never_reaches_output_device(self):
        device = mock.Mock()
        engine = TTSEngine(config(), profile="edge", offline=True)
        with mock.patch("tts.tts_engine.sd", device):
            for audio in (np.zeros(8, np.float32), np.asarray([float("nan")], np.float32)):
                with self.assertRaisesRegex(RuntimeError, "silent or non-finite"):
                    engine.play(audio, 22050)
        device.play.assert_not_called()

    def test_playback_records_the_actual_output_and_waits_for_completion(self):
        device = mock.Mock()
        device.query_devices.return_value = {"name": "Headphones", "index": 3}
        device.wait.return_value = None
        engine = TTSEngine(config(), profile="edge", offline=True)
        audio = np.asarray([0.1, -0.1], dtype=np.float32)
        with mock.patch("tts.tts_engine.sd", device), contextlib.redirect_stdout(io.StringIO()):
            result = engine.play(audio, sample_rate=22050)
        device.play.assert_called_once_with(audio, samplerate=22050, device=3)
        device.wait.assert_called_once_with(ignore_errors=False)
        self.assertTrue(result["device_playback_completed"])
        self.assertEqual(result["output_device"], "Headphones")

    def test_playback_device_error_is_not_swallowed(self):
        device = mock.Mock()
        device.query_devices.return_value = {"name": "Headphones", "index": 3}
        device.play.side_effect = RuntimeError("device disconnected")
        engine = TTSEngine(config(), profile="edge", offline=True)
        with mock.patch("tts.tts_engine.sd", device), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Audio playback failed.*disconnected"):
                engine.play(np.ones(8, np.float32), 22050)

    def test_playback_underrun_is_not_reported_as_success(self):
        device = mock.Mock()
        device.query_devices.return_value = {"name": "Headphones", "index": 3}
        device.wait.return_value = "output underflow"
        engine = TTSEngine(config(), profile="edge", offline=True)
        with mock.patch("tts.tts_engine.sd", device), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "underrun/overflow"):
                engine.play(np.ones(8, np.float32), 22050)

    def test_desktop_check_without_playback_cannot_claim_audible_success(self):
        engine = mock.Mock()
        engine.synthesize.return_value = (np.ones(8, np.float32), 22050)
        engine.is_silence.return_value = False
        engine.engine_name.return_value = "espeak-ng-offline-demo"
        with mock.patch("scripts.check_desktop_audio.TTSEngine", return_value=engine), contextlib.redirect_stdout(io.StringIO()):
            report = run_check(config(), "en2vi", 5, play=False)
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["calls"]), 5)
        self.assertFalse(any(row["device_playback_completed"] for row in report["calls"]))
        self.assertEqual(report["listener_confirmation"], "not_recorded")
        self.assertIsNone(report["intelligibility_passed"])
        engine.play.assert_not_called()

    def test_desktop_check_fails_on_output_error(self):
        engine = mock.Mock()
        engine.synthesize.return_value = (np.ones(8, np.float32), 22050)
        engine.is_silence.return_value = False
        engine.engine_name.return_value = "espeak-ng-offline-demo"
        engine.play.side_effect = RuntimeError("device disconnected")
        with mock.patch("scripts.check_desktop_audio.TTSEngine", return_value=engine), contextlib.redirect_stdout(io.StringIO()):
            report = run_check(config(), "en2vi", 5, play=True)
        self.assertFalse(report["passed"])
        self.assertEqual(report["calls"], [])
        self.assertIn("device disconnected", report["error"])


if __name__ == "__main__":
    unittest.main()
