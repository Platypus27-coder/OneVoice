"""Profile-aware VI↔EN translation adapters.

Development/premium may use the original Transformers EnViT5 checkpoint.
Edge accepts only a pre-exported ONNX Runtime GenAI encoder/decoder bundle and
never downloads at runtime. Model export is a separate quality-gated step.
"""

from __future__ import annotations

import queue
import re
import time
from pathlib import Path


class Translator:
    PREFIX = {"vi2en": "vi: ", "en2vi": "en: "}

    def __init__(self, config: dict, offline: bool = False, profile: str = "development"):
        cfg = config["translation"]
        self.profile = profile
        self.offline = bool(offline or profile == "edge")
        self.model_name = str(cfg.get("model", "VietAI/envit5-translation"))
        self.model_dir = Path(cfg.get("model_dir", "models/envit5"))
        self.edge_model_dir = Path(cfg.get("edge_model_dir", "models/envit5_ort_genai"))
        self.max_length = int(cfg.get("max_length", 512))
        self._backend: str | None = None
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._device = "cpu"
        self._og = None

    def load(self) -> None:
        if self.profile == "edge":
            self._load_edge()
        else:
            self._load_transformers()

    def _load_edge(self) -> None:
        config_file = self.edge_model_dir / "genai_config.json"
        if not config_file.is_file():
            raise FileNotFoundError(
                "Edge MT requires a validated ONNX Runtime GenAI bundle at "
                f"{self.edge_model_dir}; genai_config.json is missing"
            )
        try:
            import onnxruntime_genai as og
        except ImportError as exc:
            raise RuntimeError(
                "Edge MT requires the local 'onnxruntime-genai' CPU package"
            ) from exc
        self._og = og
        self._model = og.Model(str(self.edge_model_dir))
        self._tokenizer = og.Tokenizer(self._model)
        self._backend = "onnxruntime_genai_cpu"
        print(f"[MT] ONNX Runtime GenAI edge bundle loaded: {self.edge_model_dir}")

    def _load_transformers(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Development MT requires torch and transformers"
            ) from exc
        local_config = self.model_dir / "config.json"
        if local_config.is_file():
            source = str(self.model_dir)
            local_only = True
        elif self.offline:
            raise FileNotFoundError(f"Offline translation checkpoint not found: {self.model_dir}")
        else:
            source = self.model_name
            local_only = False
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_only)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            source, local_files_only=local_only
        ).to(self._device)
        self._model.eval()
        self._backend = "transformers"
        print(f"[MT] Transformers EnViT5 loaded from {source} on {self._device}")

    def translate(self, text: str, direction: str = "vi2en") -> str:
        if direction not in self.PREFIX:
            raise ValueError("direction must be 'vi2en' or 'en2vi'")
        if not text.strip():
            return ""
        if self._backend is None:
            raise RuntimeError("Translator not loaded. Call .load() first.")
        prompt = self.PREFIX[direction] + " ".join(text.split())
        started = time.perf_counter()
        if self._backend == "transformers":
            result = self._translate_transformers(prompt)
        else:
            result = self._translate_ort_genai(prompt)
        result = re.sub(r"^(en|vi):\s*", "", result, flags=re.IGNORECASE).strip()
        print(
            f"[MT] {direction} backend={self._backend} "
            f"latency_ms={(time.perf_counter() - started) * 1000:.0f}"
        )
        return result

    def _translate_transformers(self, prompt: str) -> str:
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self._device)
        with self._torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=5,
                early_stopping=True,
            )
        return self._tokenizer.decode(output[0], skip_special_tokens=True)

    def _translate_ort_genai(self, prompt: str) -> str:
        tokens = self._tokenizer.encode(prompt)
        params = self._og.GeneratorParams(self._model)
        params.set_search_options(max_length=self.max_length, num_beams=5)
        generator = self._og.Generator(self._model, params)
        generator.append_tokens(tokens)
        while not generator.is_done():
            generator.generate_next_token()
        sequence = generator.get_sequence(0)
        return self._tokenizer.decode(sequence)

    def run(self, text_in_queue: queue.Queue, text_out_queue: queue.Queue) -> None:
        while True:
            try:
                item = text_in_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                item["translated"] = self.translate(
                    item["text"], item.get("direction", "vi2en")
                )
                text_out_queue.put(item)
            finally:
                text_in_queue.task_done()
