import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from scripts.benchmark_asr_v2 import load_audio_mono_16k, load_construction_context


class ASRBenchmarkHelpersTests(unittest.TestCase):
    def test_load_construction_context_uses_configured_safety_source(self):
        config = {
            "pipeline": {
                "construction_data_dir": "data/onevoice_construction_v2",
                "safety_source_csv": "data/onevoice_construction_v2/safety_fast_path_review.csv",
            }
        }

        with patch(
            "scripts.benchmark_asr_v2.ConstructionContextEngine.from_data_dir"
        ) as from_data_dir:
            load_construction_context(config)

        from_data_dir.assert_called_once_with(
            "data/onevoice_construction_v2",
            safety_path="data/onevoice_construction_v2/safety_fast_path_review.csv",
        )

    def test_load_construction_context_keeps_legacy_default_safety_source(self):
        config = {"pipeline": {"construction_data_dir": "data/onevoice_construction_v2"}}

        with patch(
            "scripts.benchmark_asr_v2.ConstructionContextEngine.from_data_dir"
        ) as from_data_dir:
            load_construction_context(config)

        from_data_dir.assert_called_once_with(
            "data/onevoice_construction_v2", safety_path=None
        )

    def test_load_audio_downmixes_stereo_without_resampling_16khz(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stereo.wav"
            stereo = np.array([[0.2, -0.2], [0.4, 0.0]], dtype=np.float32)
            sf.write(path, stereo, 16000)

            audio = load_audio_mono_16k(path)

        np.testing.assert_allclose(
            audio, np.array([0.0, 0.2], dtype=np.float32), atol=2e-5
        )
        self.assertEqual(audio.dtype, np.float32)
