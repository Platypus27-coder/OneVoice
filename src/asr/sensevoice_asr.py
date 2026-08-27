"""SenseVoice English ASR with explicit local/offline loading."""

from __future__ import annotations

import json
import os
import re

import numpy as np


class _NumericTokenDecoder:
    """Adapter expected by newer ``SenseVoiceSmallONNX.__call__``.

    That API calls ``tokenizer.tokens2text(token_ids)``. Its SentencePiece
    helper accepts token strings rather than token IDs, so bridge it with the
    bundle's ``tokens.json`` ID vocabulary first.
    """

    def __init__(self, converter, text_tokenizer):
        self._converter = converter
        self._text_tokenizer = text_tokenizer

    def tokens2text(self, token_ids) -> str:
        return self._text_tokenizer.tokens2text(
            self._converter.ids2tokens(token_ids)
        )


class SenseVoiceASR:
    def __init__(self, config: dict, offline: bool = False):
        cfg = config.get("sensevoice", {})
        self.model_dir = cfg.get("model_path", "models/sensevoice")
        self.remote_model = cfg.get("remote_model", "iic/SenseVoiceSmall")
        self.quantize = bool(cfg.get("quantize", True))
        self.offline = offline
        self.model = None
        self._numeric_tag_api = False
        self._tokenizer = None

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
        if self._numeric_tag_api:
            self._tokenizer = self._load_numeric_api_tokenizer(model_dir)
        precision = "INT8" if self.quantize else "FP32"
        print(f"[ASR] ✅ SenseVoice ONNX ({precision}) loaded from {model_dir}")

    def _load_numeric_api_tokenizer(self, model_dir: str):
        """Load the local SentencePiece decoder required by newer FunASR ONNX.

        ``SenseVoiceSmallONNX`` returns raw token IDs when its optional
        ``tokenizer`` argument is omitted. Decoding those IDs is mandatory:
        treating their Python-list representation as ASR text prevents context
        and safety matching.
        """
        try:
            from funasr_onnx.utils.sentencepiece_tokenizer import SentencepiecesTokenizer
            from funasr_onnx.utils.utils import TokenIDConverter
        except ImportError as exc:
            raise ImportError(
                "The installed funasr_onnx package lacks its SentencePiece tokenizer"
            ) from exc
        root = os.path.abspath(model_dir)
        token_path = os.path.join(root, "tokens.json")
        if not os.path.isfile(token_path):
            raise FileNotFoundError(f"SenseVoice ONNX bundle is missing tokens.json: {root}")
        token_payload = json.loads(open(token_path, encoding="utf-8").read())
        if isinstance(token_payload, dict):
            token_list = [
                token
                for token, _ in sorted(token_payload.items(), key=lambda item: int(item[1]))
            ]
        elif isinstance(token_payload, list):
            token_list = token_payload
        else:
            raise ValueError(f"Invalid SenseVoice tokens.json format: {token_path}")
        for filename in (
            "chn_jpn_yue_eng_ko_spectok.bpe.model",
            "spiece.model",
        ):
            candidate = os.path.join(root, filename)
            if os.path.isfile(candidate):
                return _NumericTokenDecoder(
                    TokenIDConverter(token_list),
                    SentencepiecesTokenizer(bpemodel=candidate),
                )
        raise FileNotFoundError(
            "SenseVoice ONNX bundle is missing its SentencePiece model "
            "(expected chn_jpn_yue_eng_ko_spectok.bpe.model or spiece.model)"
        )

    @staticmethod
    def _prompt_tag(value: int) -> list[int]:
        """Build a rank-1 SenseVoice prompt tensor for newer FunASR wheels.

        The public wrapper's metadata may report a scalar while the underlying
        ONNX session requires rank one (both staged FP32 and INT8 bundles do).
        Legacy ``SenseVoiceSmall`` uses string options and does not enter this
        numeric branch.
        """
        return [value]

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
            # The public wrapper converts these one-element lists with
            # ``np.array`` into the rank-1 tensors required by the ONNX graph.
            kwargs = {
                "language": self._prompt_tag(4),
                "textnorm": self._prompt_tag(14),
            }
            if self._tokenizer is not None:
                kwargs["tokenizer"] = self._tokenizer
            print(
                "[ASR] SenseVoice numeric prompt "
                f"language={kwargs['language']} textnorm={kwargs['textnorm']}"
            )
            result = self.model(audio_f32, **kwargs)
        else:
            result = self.model(audio_f32, language="en", textnorm="withitn")
        if not result:
            return {"text": "", "emotion": "neutral", "event": "speech"}
        return self._parse_output(str(result[0]))
