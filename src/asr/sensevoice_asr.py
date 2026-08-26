"""SenseVoice English ASR with explicit local/offline loading."""

from __future__ import annotations

import os
import re

import numpy as np


class SenseVoiceASR:
    def __init__(self, config: dict, offline: bool = False):
        cfg = config.get("sensevoice", {})
        self.model_dir = cfg.get("model_path", "models/sensevoice")
        self.remote_model = cfg.get("remote_model", "iic/SenseVoiceSmall")
        self.quantize = bool(cfg.get("quantize", True))
        self.offline = offline
        self.model = None
        self._numeric_tag_api = False

    def load(self) -> None:
        try:
            from funasr_onnx import SenseVoiceSmall
        except ImportError as exc:
            # funasr_onnx renamed this public class in newer wheels while the
            # stable releases still expose SenseVoiceSmall. Both constructors
            # accept model_dir/batch_size/quantize.
            try:
                from funasr_onnx import SenseVoiceSmallONNX as SenseVoiceSmall
                self._numeric_tag_api = True
            except ImportError:
                raise ImportError("Install a funasr_onnx build with SenseVoice support to use EN→VI ASR") from exc

        model_dir = self.model_dir
        if not os.path.isdir(model_dir):
            if self.offline:
                raise FileNotFoundError(f"Missing offline SenseVoice model: {model_dir}")
            try:
                from modelscope import snapshot_download
                model_dir = snapshot_download(self.remote_model)
            except Exception as exc:
                raise RuntimeError(f"Could not prepare SenseVoice model: {exc}") from exc
        self.model = SenseVoiceSmall(model_dir, batch_size=1, quantize=self.quantize)
        precision = "INT8" if self.quantize else "FP32"
        print(f"[ASR] ✅ SenseVoice ONNX ({precision}) loaded from {model_dir}")

    @staticmethod
    def _parse_output(raw_text: str) -> dict:
        emotion_match = re.search(
            r"<\|(HAPPY|SAD|ANGRY|NEUTRAL|FEARFUL|DISGUSTED|SURPRISED)\|>",
            raw_text,
            re.IGNORECASE,
        )
        event_match = re.search(
            r"<\|(BGM|Speech|Applause|Laughter|Cry|Sneeze|Breath|Cough)\|>",
            raw_text,
            re.IGNORECASE,
        )
        return {
            "text": re.sub(r"<\|.*?\|>", "", raw_text).strip(),
            "emotion": emotion_match.group(1).lower() if emotion_match else "neutral",
            "event": event_match.group(1).lower() if event_match else "speech",
            "raw": raw_text,
        }

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> dict:
        if self.model is None:
            raise RuntimeError("SenseVoice not loaded. Call .load() first.")
        if len(audio) == 0:
            return {"text": "", "emotion": "neutral", "event": "speech"}
        audio_f32 = np.asarray(audio, dtype=np.float32)
        if np.max(np.abs(audio_f32), initial=0.0) > 1.0:
            audio_f32 /= 32768.0
        # ``funasr_onnx`` interprets a list as a list of audio *paths*.
        # Pass the mono waveform directly so its ndarray branch is selected.
        # Its public option is ``textnorm`` (not ``use_itn``).
        if self._numeric_tag_api:
            # New funasr_onnx API consumes already-encoded prompt IDs. These
            # are the SenseVoice runtime's fixed English and with-ITN values.
            result = self.model(audio_f32, language=4, textnorm=14)
        else:
            result = self.model(audio_f32, language="en", textnorm="withitn")
        if not result:
            return {"text": "", "emotion": "neutral", "event": "speech"}
        return self._parse_output(str(result[0]))
