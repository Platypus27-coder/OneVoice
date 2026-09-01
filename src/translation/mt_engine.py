"""Profile-aware VI↔EN translation adapters.

Development/premium may use the original Transformers EnViT5 checkpoint.
Edge accepts only a pre-exported ONNX Runtime Seq2Seq bundle and never
downloads at runtime. EnViT5 is a T5 encoder-decoder model, so it cannot use
the decoder-only ONNX Runtime GenAI bundle format. Model export is a separate
quality-gated step.
"""

from __future__ import annotations

import queue
import re
import time
from pathlib import Path


class Translator:
    PREFIX = {"vi2en": "vi: ", "en2vi": "en: "}

    def __init__(
        self,
        config: dict,
        offline: bool = False,
        profile: str = "development",
        direction: str | None = None,
        model_source: str | None = None,
        model_revision: str | None = None,
    ):
        cfg = config["translation"]
        configured_direction = direction or cfg.get("direction", "vi2en")
        if configured_direction not in self.PREFIX:
            raise ValueError("direction must be 'vi2en' or 'en2vi'")
        direction_cfg = cfg.get("directions", {}).get(configured_direction, {})
        if not isinstance(direction_cfg, dict):
            raise ValueError(f"translation.directions.{configured_direction} must be a mapping")
        self.direction = configured_direction
        self.profile = profile
        self.offline = bool(offline or profile == "edge")
        self.model_name = str(
            model_source
            or direction_cfg.get("release_model")
            or direction_cfg.get("development_model")
            or cfg.get("model", "VietAI/envit5-translation")
        )
        self.model_revision = (
            model_revision
            or direction_cfg.get("release_revision")
            or direction_cfg.get("model_revision")
            or cfg.get("model_revision")
        )
        self.model_dir = Path(
            direction_cfg.get("local_model_dir") or cfg.get("model_dir", "models/envit5")
        )
        self.edge_model_dir = Path(
            direction_cfg.get("edge_model_dir")
            or cfg.get("edge_model_dir", "models/envit5_ort")
        )
        self.max_length = int(cfg.get("max_length", 512))
        self._backend: str | None = None
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._device = "cpu"

    @property
    def model_reference(self) -> dict[str, str | None]:
        """Exact logical model requested for a measured run or runtime startup."""
        return {
            "direction": self.direction,
            "source": self.model_name,
            "revision": self.model_revision,
            "local_model_dir": str(self.model_dir),
            "edge_model_dir": str(self.edge_model_dir),
        }

    def load(self) -> None:
        if self.profile == "edge":
            self._load_edge()
        else:
            self._load_transformers()

    def _load_edge(self) -> None:
        required = ("config.json", "encoder_model.onnx")
        missing = [name for name in required if not (self.edge_model_dir / name).is_file()]
        has_decoder = any(self.edge_model_dir.glob("decoder*.onnx"))
        if missing or not has_decoder:
            detail = ", ".join(missing + ([] if has_decoder else ["decoder*.onnx"]))
            raise FileNotFoundError(
                "Edge MT requires a validated ONNX Runtime Seq2Seq T5 bundle at "
                f"{self.edge_model_dir}; missing {detail}"
            )
        try:
            from optimum.onnxruntime import ORTModelForSeq2SeqLM
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Edge MT requires local optimum-onnx/onnxruntime and transformers packages"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.edge_model_dir), local_files_only=True
        )
        self._model = ORTModelForSeq2SeqLM.from_pretrained(
            str(self.edge_model_dir),
            provider="CPUExecutionProvider",
            local_files_only=True,
        )
        self._backend = "onnxruntime_seq2seq_cpu"
        print(f"[MT] ONNX Runtime Seq2Seq edge bundle loaded: {self.edge_model_dir}")

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
        load_kwargs = {"local_files_only": local_only}
        if self.model_revision and not local_only:
            load_kwargs["revision"] = self.model_revision
        self._tokenizer = AutoTokenizer.from_pretrained(source, **load_kwargs)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            source, **load_kwargs
        ).to(self._device)
        self._model.eval()
        self._backend = "transformers"
        print(f"[MT] Transformers EnViT5 loaded from {source} on {self._device}")

    def translate(self, text: str, direction: str = "vi2en") -> str:
        if direction not in self.PREFIX:
            raise ValueError("direction must be 'vi2en' or 'en2vi'")
        if direction != self.direction:
            raise ValueError(
                f"Translator loaded for {self.direction}; create a direction-specific "
                f"Translator for {direction}"
            )
        if not text.strip():
            return ""
        if self._backend is None:
            raise RuntimeError("Translator not loaded. Call .load() first.")
        prompt = self.PREFIX[direction] + " ".join(text.split())
        started = time.perf_counter()
        if self._backend == "transformers":
            result = self._translate_transformers(prompt)
        else:
            result = self._translate_ort_seq2seq(prompt)
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

    def _translate_ort_seq2seq(self, prompt: str) -> str:
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        output = self._model.generate(
            **inputs,
            max_length=self.max_length,
            num_beams=5,
            early_stopping=True,
        )
        return self._tokenizer.decode(output[0], skip_special_tokens=True)

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
