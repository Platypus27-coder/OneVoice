"""
Trạm 0: Industrial Noise Denoising
====================================
Wraps the GIPFormer ONNX model to filter out industrial noise
(machinery, engines, hammering) from the captured audio stream.

Reference: gipformer — G-Group AI Lab (MIT License)
           https://huggingface.co/g-group-ai-lab/gipformer-65M-rnnt
"""

import time
import numpy as np
import onnxruntime as ort


class Denoiser:
    """
    Runs GIPFormer ONNX inference to clean industrial noise from raw audio.
    Designed for CPU inference on edge devices (< 10ms per chunk).
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._session = None

    def load(self):
        """Load ONNX model into ONNXRuntime session."""
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(
            self.model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        print(f"[Denoiser] ✅ GIPFormer ONNX loaded from: {self.model_path}")

    def denoise(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Run noise filtering on a raw audio chunk.

        Args:
            audio: numpy float32 array, shape (N,)
            sample_rate: audio sample rate (default 16000)

        Returns:
            Denoised audio as float32 numpy array.
        """
        if self._session is None:
            raise RuntimeError("Denoiser not loaded. Call .load() first.")

        t0 = time.perf_counter()

        # Prepare input — shape expected by GIPFormer: (1, samples)
        input_data = audio.astype(np.float32)[np.newaxis, :]
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: input_data})
        clean_audio = outputs[0].squeeze()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[Denoiser] ⏱ {elapsed_ms:.1f}ms")

        return clean_audio.astype(np.float32)

    def passthrough(self, audio: np.ndarray) -> np.ndarray:
        """Return audio unchanged. Used when model is not available."""
        print("[Denoiser] ⚠ Passthrough mode (model not loaded)")
        return audio
