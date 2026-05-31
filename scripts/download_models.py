"""
Model Download Setup Script
============================
Downloads all required model files to the correct local directories.
Run this once before starting the pipeline.

Models downloaded:
  - GIPFormer (INT8 ONNX) — from HuggingFace
  - Whisper-Tiny           — from HuggingFace / OpenAI
  - MarianMT VI↔EN         — from HuggingFace Helsinki-NLP
"""

import os
import sys
import time

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def download_gipformer():
    print("\n[1/3] Downloading GIPFormer (Vietnamese ASR — INT8 ONNX)...")
    from huggingface_hub import hf_hub_download

    REPO = "g-group-ai-lab/gipformer-65M-rnnt"
    FILES = [
        "encoder-epoch-35-avg-6.int8.onnx",
        "decoder-epoch-35-avg-6.int8.onnx",
        "joiner-epoch-35-avg-6.int8.onnx",
        "tokens.txt",
    ]
    out_dir = os.path.join(MODELS_DIR, "gipformer")
    ensure_dir(out_dir)

    for fname in FILES:
        t0 = time.time()
        path = hf_hub_download(repo_id=REPO, filename=fname,
                                local_dir=out_dir)
        print(f"  ✅ {fname} → {path} ({time.time()-t0:.1f}s)")


def download_whisper():
    print("\n[2/3] Downloading Whisper-Tiny (English ASR)...")
    try:
        import whisper
        model = whisper.load_model("tiny")
        print("  ✅ Whisper-Tiny loaded and cached.")
    except Exception as e:
        print(f"  ⚠ {e}")


def download_marianmt():
    print("\n[3/3] Downloading MarianMT VI↔EN models...")
    from transformers import MarianTokenizer, MarianMTModel

    models = [
        ("Helsinki-NLP/opus-mt-vi-en", os.path.join(MODELS_DIR, "marianmt", "vi2en")),
        ("Helsinki-NLP/opus-mt-en-vi", os.path.join(MODELS_DIR, "marianmt", "en2vi")),
    ]
    for model_name, out_dir in models:
        ensure_dir(out_dir)
        print(f"  Downloading {model_name}...")
        t0 = time.time()
        MarianTokenizer.from_pretrained(model_name).save_pretrained(out_dir)
        MarianMTModel.from_pretrained(model_name).save_pretrained(out_dir)
        print(f"  ✅ Saved to {out_dir} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    print("=" * 55)
    print("  OneVoice Edge — Model Setup")
    print("=" * 55)
    ensure_dir(MODELS_DIR)

    try:
        download_gipformer()
    except Exception as e:
        print(f"  ❌ GIPFormer download failed: {e}")

    try:
        download_whisper()
    except Exception as e:
        print(f"  ❌ Whisper download failed: {e}")

    try:
        download_marianmt()
    except Exception as e:
        print(f"  ❌ MarianMT download failed: {e}")

    print("\n✅ Setup complete. Run: python src/pipeline.py --direction vi2en")
