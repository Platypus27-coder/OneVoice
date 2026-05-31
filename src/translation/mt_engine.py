"""
Trạm 2: Bilingual Machine Translation (MarianMT)
=================================================
Translates text between Vietnamese and English using Helsinki-NLP's
lightweight MarianMT models. Supports fine-tuned checkpoints
for industrial terminology accuracy.

Reference: Helsinki-NLP/opus-mt — University of Helsinki (Apache 2.0)
           https://huggingface.co/Helsinki-NLP/opus-mt-vi-en
           https://huggingface.co/Helsinki-NLP/opus-mt-en-vi
"""

import time
import queue
import torch
from transformers import MarianMTModel, MarianTokenizer


class Translator:
    """
    Bidirectional translator: Vietnamese ↔ English.
    Loads fine-tuned MarianMT checkpoints if available,
    otherwise falls back to pretrained Helsinki-NLP models.
    """

    VI_EN_MODEL = "Helsinki-NLP/opus-mt-vi-en"
    EN_VI_MODEL = "Helsinki-NLP/opus-mt-en-vi"

    def __init__(self, config: dict):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = config["translation"]
        self.model_dir = cfg.get("model_dir", None)
        self.max_length = cfg.get("max_length", 128)
        self._models: dict = {}
        self._tokenizers: dict = {}

    def load(self):
        """Load both translation direction models."""
        print(f"[MT] Loading MarianMT models on {self.device}...")
        for direction, name in [("vi2en", self.VI_EN_MODEL), ("en2vi", self.EN_VI_MODEL)]:
            local = f"{self.model_dir}/{direction}" if self.model_dir else None
            source = local if local else name
            print(f"[MT]   Loading {direction}: {source}")
            self._tokenizers[direction] = MarianTokenizer.from_pretrained(source)
            self._models[direction] = MarianMTModel.from_pretrained(source).to(self.device)
        print("[MT] ✅ Both translation models loaded.")

    def translate(self, text: str, direction: str = "vi2en") -> str:
        """
        Translate a piece of text.

        Args:
            text: input text string
            direction: "vi2en" or "en2vi"

        Returns:
            Translated text string.
        """
        if not text.strip():
            return ""
        if direction not in self._models:
            raise ValueError(f"Unknown direction: {direction}. Use 'vi2en' or 'en2vi'.")

        t0 = time.perf_counter()
        tokenizer = self._tokenizers[direction]
        model = self._models[direction]

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_length).to(self.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=self.max_length)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[MT] ⏱ {elapsed_ms:.0f}ms | {direction} | \"{text}\" → \"{result}\"")

        return result

    def run(self, text_in_queue: queue.Queue, text_out_queue: queue.Queue):
        """
        Worker loop: reads text items from queue, translates, pushes to output queue.
        Each item is a dict: {"text": str, "lang": str}
        """
        print("[MT Worker] ✅ Started")
        while True:
            try:
                item = text_in_queue.get(timeout=1)
                text = item["text"]
                detected_lang = item.get("lang", "vi")

                # Auto-select direction based on detected language
                direction = "vi2en" if detected_lang == "vi" else "en2vi"
                translated = self.translate(text, direction=direction)

                if translated:
                    text_out_queue.put({"text": translated, "direction": direction})
                text_in_queue.task_done()
            except queue.Empty:
                continue


if __name__ == "__main__":
    # Quick test
    import yaml
    with open("../../config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    t = Translator(cfg)
    t.load()
    result = t.translate("Máy xúc số 3 đang bị lỗi thủy lực, cần chuyên gia kiểm tra.", "vi2en")
    print("Result:", result)
