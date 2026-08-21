"""Waveform denoising backends for OneVoice V2.

GIPFormer is an ASR model and is deliberately not used as a denoiser.  The
passthrough backend is the reproducible baseline; optional enhancement backends
must return waveform audio and pass downstream ASR quality gates.
"""

from __future__ import annotations

import importlib
from typing import Protocol

import numpy as np


SAMPLE_RATE = 16000


class DenoiserBackend(Protocol):
    name: str

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray: ...

    def reset(self) -> None: ...


class PassthroughBackend:
    name = "passthrough"

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        del sample_rate
        value = np.asarray(audio, dtype=np.float32)
        if value.ndim > 1:
            value = value.mean(axis=1)
        return np.ascontiguousarray(value)

    def reset(self) -> None:
        return None


class DeepFilterBackend:
    """Optional DeepFilterNet quality-reference backend.

    This wrapper is intended for utterance/offline A/B evaluation. It is not
    selected by the edge profile until stateful streaming and latency gates pass.
    """

    name = "deepfilter"

    def __init__(self):
        try:
            import torch
            from df.enhance import enhance, init_df
        except ImportError as exc:
            raise RuntimeError(
                "DeepFilterNet backend requires the optional 'deepfilternet' package"
            ) from exc
        self._torch = torch
        self._enhance = enhance
        self._model, self._df_state, _ = init_df()

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        value = np.asarray(audio, dtype=np.float32).reshape(-1)
        model_rate = int(self._df_state.sr())
        if sample_rate != model_rate:
            value = RNNoiseBackend._resample(value, sample_rate, model_rate)
        tensor = self._torch.as_tensor(value, dtype=self._torch.float32).reshape(1, -1)
        enhanced = self._enhance(self._model, self._df_state, tensor)
        result = enhanced.detach().cpu().numpy().reshape(-1).astype(np.float32)
        if sample_rate != model_rate:
            result = RNNoiseBackend._resample(result, model_rate, sample_rate)
        target_size = len(np.asarray(audio).reshape(-1))
        if len(result) < target_size:
            result = np.pad(result, (0, target_size - len(result)))
        return np.ascontiguousarray(result[:target_size], dtype=np.float32)

    def reset(self) -> None:
        return None


class RNNoiseBackend:
    """RNNoise backend contract.

    RNNoise uses 48 kHz/480-sample frames. The native binding must expose a
    stateful ``process_frame(float32[480])`` method. Requiring that small
    contract avoids silently using an incompatible Python package.
    """

    name = "rnnoise"
    frame_size = 480
    sample_rate = 48000

    def __init__(self, processor=None):
        if processor is None:
            raise RuntimeError(
                "RNNoise requires a native processor exposing process_frame(); "
                "configure it only after the RNNoise candidate is installed"
            )
        if not callable(getattr(processor, "process_frame", None)):
            raise TypeError("RNNoise processor must expose process_frame(frame)")
        self._processor = processor

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        value = np.asarray(audio, dtype=np.float32).reshape(-1)
        original_size = len(value)
        if sample_rate != self.sample_rate:
            value = self._resample(value, sample_rate, self.sample_rate)
        padding = (-len(value)) % self.frame_size
        if padding:
            value = np.pad(value, (0, padding))
        output = [
            np.asarray(self._processor.process_frame(frame), dtype=np.float32)
            for frame in value.reshape(-1, self.frame_size)
        ]
        enhanced = np.concatenate(output) if output else np.zeros(0, dtype=np.float32)
        if padding:
            enhanced = enhanced[:-padding]
        if sample_rate != self.sample_rate:
            enhanced = self._resample(enhanced, self.sample_rate, sample_rate)
        if len(enhanced) < original_size:
            enhanced = np.pad(enhanced, (0, original_size - len(enhanced)))
        return np.ascontiguousarray(enhanced[:original_size], dtype=np.float32)

    @staticmethod
    def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        try:
            from scipy.signal import resample_poly

            from math import gcd

            divisor = gcd(source_rate, target_rate)
            return resample_poly(
                audio, target_rate // divisor, source_rate // divisor
            ).astype(np.float32)
        except ImportError:
            import librosa

            return librosa.resample(
                audio, orig_sr=source_rate, target_sr=target_rate
            ).astype(np.float32)

    def reset(self) -> None:
        reset = getattr(self._processor, "reset", None)
        if callable(reset):
            reset()


class Denoiser:
    """Stable denoiser facade retained for V1 call-site compatibility."""

    def __init__(self, config: dict | None = None, num_threads: int | None = None):
        del num_threads
        cfg = config or {}
        self.backend_name = str(cfg.get("backend", "passthrough")).casefold()
        self._rnnoise_factory = str(cfg.get("rnnoise_factory", "")).strip()
        self._backend: DenoiserBackend | None = None

    def load(self) -> None:
        if self.backend_name == "passthrough":
            self._backend = PassthroughBackend()
        elif self.backend_name == "deepfilter":
            self._backend = DeepFilterBackend()
        elif self.backend_name == "rnnoise":
            if ":" not in self._rnnoise_factory:
                raise RuntimeError(
                    "RNNoise candidate requires rnnoise_factory='module:callable'; "
                    "the callable must return a process_frame(float32[480]) processor"
                )
            module_name, callable_name = self._rnnoise_factory.split(":", 1)
            factory = getattr(importlib.import_module(module_name), callable_name)
            self._backend = RNNoiseBackend(factory())
        else:
            raise ValueError(f"Unsupported denoiser backend: {self.backend_name}")
        print(f"[Denoiser] Backend ready: {self._backend.name}")

    def process(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        if self._backend is None:
            raise RuntimeError("Denoiser not loaded. Call .load() first.")
        return self._backend.process(audio, sample_rate)

    def denoise(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        return self.process(audio, sample_rate)

    def reset(self) -> None:
        if self._backend is not None:
            self._backend.reset()
