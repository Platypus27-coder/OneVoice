"""
Model Download Setup Script
============================
Downloads all required model files to the correct local directories.
Run this once before starting the pipeline.

Models downloaded:
  - GIPFormer (INT8 ONNX) — from HuggingFace
  - SenseVoiceSmall        — from ModelScope / HuggingFace
  - VietAI/envit5          — from HuggingFace
"""

import os
import sys
import time

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def download_gipformer():
    print("\n[1/3] Downloading GIPFormer (Vietnamese ASR / Denoise — INT8 ONNX)...")
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


def download_sensevoice():
    print("\n[2/3] Downloading SenseVoiceSmall (EN/VI ASR)...")
    try:
        from modelscope import snapshot_download
        model_dir = snapshot_download('iic/SenseVoiceSmall')
        print(f"  ✅ SenseVoiceSmall loaded and cached at {model_dir}")
    except Exception as e:
        print(f"  ⚠ SenseVoice download failed: {e}")


def download_envit5():
    print("\n[3/3] Downloading VietAI/envit5-translation (MT)...")
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        print("  Downloading tokenizer...")
        AutoTokenizer.from_pretrained("VietAI/envit5-translation")
        print("  Downloading model...")
        AutoModelForSeq2SeqLM.from_pretrained("VietAI/envit5-translation")
        print("  ✅ Envit5 loaded and cached via HuggingFace.")
    except Exception as e:
        print(f"  ⚠ Envit5 download failed: {e}")


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
        download_sensevoice()
    except Exception as e:
        print(f"  ❌ SenseVoice download failed: {e}")

    try:
        download_envit5()
    except Exception as e:
        print(f"  ❌ Envit5 download failed: {e}")

    print("\n✅ Setup complete. Run: python src/pipeline.py --direction vi2en")
