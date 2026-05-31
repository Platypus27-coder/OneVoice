"""
Trạm 2: Bilingual Machine Translation — MarianMT (VI ↔ EN)
===========================================================
Focused exclusively on Vietnamese ↔ English translation using
Helsinki-NLP's MarianMT models. Supports fine-tuned checkpoints
for industrial/engineering terminology accuracy.

References:
  Helsinki-NLP/opus-mt-vi-en — University of Helsinki (Apache 2.0)
  Helsinki-NLP/opus-mt-en-vi — University of Helsinki (Apache 2.0)
  https://huggingface.co/Helsinki-NLP
"""

import time
import queue
import re
import torch
from transformers import MarianMTModel, MarianTokenizer


# ── Industrial terminology override dictionary ────────────────────────────────
# Loaded from data/industrial_terms.csv at runtime if available.
# These are direct substitutions BEFORE translation to improve accuracy.
_TERM_OVERRIDE_VI_EN: dict[str, str] = {}
_TERM_OVERRIDE_EN_VI: dict[str, str] = {}


def load_terminology(csv_path: str):
    """Load industrial term pairs from CSV into override dictionaries."""
    import csv
    try:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vi = row["vi_text"].strip().strip('"').lower()
                en = row["en_text"].strip().strip('"').lower()
                if vi and en:
                    _TERM_OVERRIDE_VI_EN[vi] = en
                    _TERM_OVERRIDE_EN_VI[en] = vi
        print(f"[MT] ✅ Loaded {len(_TERM_OVERRIDE_VI_EN)} technical term pairs.")
    except FileNotFoundError:
        print(f"[MT] ⚠ Terminology file not found: {csv_path}")
    except Exception as e:
        print(f"[MT] ⚠ Could not load terminology: {e}")


def _apply_terminology(text: str, override: dict[str, str]) -> str:
    """Replace known technical terms before passing to neural MT."""
    text_lower = text.lower()
    result = text
    for src, tgt in sorted(override.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(re.escape(src), re.IGNORECASE)
        result = pattern.sub(tgt, result)
    return result


class Translator:
    """
    Bidirectional translator: Vietnamese ↔ English (VI2EN, EN2VI).

    Loads fine-tuned MarianMT checkpoints if present in models/marianmt/,
    otherwise downloads pretrained Helsinki-NLP models from HuggingFace.
    """

    VI_EN_REMOTE = "Helsinki-NLP/opus-mt-vi-en"
    EN_VI_REMOTE = "Helsinki-NLP/opus-mt-en-vi"

    def __init__(self, config: dict):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = config["translation"]
        self.model_dir = cfg.get("model_dir", None)
        self.max_length = cfg.get("max_length", 128)
        self._models: dict = {}
        self._tokenizers: dict = {}

    def _model_source(self, direction: str) -> str:
        """Return local fine-tuned path if available, else HuggingFace remote."""
        if self.model_dir:
            import os
            local_path = os.path.join(self.model_dir, direction)
            config_file = os.path.join(local_path, "config.json")
            if os.path.exists(config_file):
                print(f"[MT] Using fine-tuned model: {local_path}")
                return local_path
        remote = self.VI_EN_REMOTE if direction == "vi2en" else self.EN_VI_REMOTE
        print(f"[MT] Using pretrained: {remote}")
        return remote

    def load(self):
        """Load both VI→EN and EN→VI models."""
        print(f"[MT] Loading MarianMT models on {self.device}...")
        for direction in ["vi2en", "en2vi"]:
            src = self._model_source(direction)
            self._tokenizers[direction] = MarianTokenizer.from_pretrained(src)
            self._models[direction] = (
                MarianMTModel.from_pretrained(src).to(self.device)
            )
        print("[MT] ✅ Both VI↔EN models loaded.")

        # Load industrial terminology overrides
        import os
        term_path = os.path.join(
            os.path.dirname(__file__), "../../data/industrial_terms.csv"
        )
        load_terminology(os.path.abspath(term_path))

    def translate(self, text: str, direction: str = "vi2en") -> str:
        """
        Translate text between Vietnamese and English.

        Args:
            text: input text
            direction: "vi2en" (Vietnamese→English) or "en2vi" (English→Vietnamese)

        Returns:
            Translated text string.
        """
        if not text.strip():
            return ""
        if direction not in self._models:
            raise ValueError(f"Unsupported direction: {direction}. Use 'vi2en' or 'en2vi'.")

        # Apply technical terminology substitution first
        override_map = _TERM_OVERRIDE_VI_EN if direction == "vi2en" else _TERM_OVERRIDE_EN_VI
        text_with_terms = _apply_terminology(text, override_map)

        t0 = time.perf_counter()
        tokenizer = self._tokenizers[direction]
        model = self._models[direction]

        inputs = tokenizer(
            text_with_terms,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=self.max_length)

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        arrow = "VI→EN" if direction == "vi2en" else "EN→VI"
        print(f"[MT] ⏱ {elapsed_ms:.0f}ms | {arrow} | \"{text}\" → \"{result}\"")

        return result

    def run(self, text_in_queue: queue.Queue, text_out_queue: queue.Queue):
        """
        Worker loop: reads ASR results, translates, pushes to TTS queue.
        Each item: {"text": str, "lang": str, "direction": str}
        """
        print("[MT Worker] ✅ Started (VI↔EN)")
        while True:
            try:
                item = text_in_queue.get(timeout=1)
                direction = item.get("direction", "vi2en")
                text = item["text"]
                translated = self.translate(text, direction=direction)

                if translated:
                    text_out_queue.put({
                        "text": translated,
                        "direction": direction,
                        "original": text,
                    })
                text_in_queue.task_done()
            except queue.Empty:
                continue


if __name__ == "__main__":
    import yaml, os
    cfg_path = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    t = Translator(cfg)
    t.load()

    tests = [
        ("vi2en", "Máy xúc số 3 đang bị lỗi thủy lực, cần kỹ sư kiểm tra ngay."),
        ("vi2en", "Van an toàn trên đường ống số 5 bị rò rỉ."),
        ("en2vi", "The hydraulic jack on excavator number 3 has failed."),
        ("en2vi", "Please check the safety valve on pipeline 5 immediately."),
    ]
    print("\n── Translation Tests ──")
    for direction, text in tests:
        result = t.translate(text, direction)
        print(f"  [{direction}] {text}")
        print(f"         → {result}\n")
