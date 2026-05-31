"""
[Colab] Quantize & Compile Models for Qualcomm Snapdragon
==========================================================
Run this on Google Colab with GPU after fine-tuning is complete.
Uses Qualcomm AI Hub to compile models targeting Snapdragon NPU (INT8).

Prerequisites:
  - Set QAI_HUB_API_TOKEN in Colab Secrets or as environment variable
  - Have fine-tuned model ready (from finetune_marian.py)

Usage:
  1. Open this file in Colab
  2. Set your QAI_HUB_API_TOKEN
  3. Run all cells
  4. Download quantized_models/ folder to local machine
"""

# !pip install qai-hub qai-hub-models onnx onnxruntime transformers

import os
import torch
import onnx
import onnxruntime as ort
from transformers import MarianMTModel, MarianTokenizer, WhisperForConditionalGeneration, WhisperProcessor

os.environ["QAI_HUB_API_TOKEN"] = "YOUR_API_KEY_HERE"  # Replace or use Colab Secrets

# import qai_hub as hub  # Uncomment when API key is set

OUTPUT_DIR = "quantized_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Export Whisper-Tiny to ONNX
# ═══════════════════════════════════════════════════════════════════════════
print("── Exporting Whisper-Tiny to ONNX ──")

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model_whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model_whisper.eval()

dummy_audio = torch.randn(1, 80, 3000)  # Mel spectrogram input
dummy_ids   = torch.tensor([[50258]])   # decoder_input_ids

onnx_path = f"{OUTPUT_DIR}/whisper_tiny.onnx"
torch.onnx.export(
    model_whisper,
    (dummy_audio, dummy_ids),
    onnx_path,
    input_names=["input_features", "decoder_input_ids"],
    output_names=["logits"],
    opset_version=14,
    dynamic_axes={"input_features": {0: "batch"}, "decoder_input_ids": {0: "batch"}},
)
print(f"✅ Whisper-Tiny ONNX saved: {onnx_path}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Export MarianMT (fine-tuned) to ONNX
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Exporting MarianMT to ONNX ──")

MARIAN_MODEL = "marianmt_finetuned_vi_en"  # or "Helsinki-NLP/opus-mt-vi-en"
tokenizer_mt = MarianTokenizer.from_pretrained(MARIAN_MODEL)
model_mt = MarianMTModel.from_pretrained(MARIAN_MODEL).eval()

sample_text = "Máy xúc số 3 đang bị lỗi thủy lực."
inputs = tokenizer_mt(sample_text, return_tensors="pt", padding=True, truncation=True)

onnx_mt_path = f"{OUTPUT_DIR}/marianmt_vi_en.onnx"
torch.onnx.export(
    model_mt,
    (inputs["input_ids"], inputs["attention_mask"]),
    onnx_mt_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    opset_version=14,
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
    },
)
print(f"✅ MarianMT ONNX saved: {onnx_mt_path}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Submit to Qualcomm AI Hub for INT8 Quantization
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Submitting to Qualcomm AI Hub ──")

# Uncomment and run this block after setting your QAI_HUB_API_TOKEN:
#
# import qai_hub as hub
# device = hub.Device("Snapdragon 8 Gen 3")
#
# for model_name, onnx_file in [
#     ("whisper_tiny",     f"{OUTPUT_DIR}/whisper_tiny.onnx"),
#     ("marianmt_vi_en",   f"{OUTPUT_DIR}/marianmt_vi_en.onnx"),
# ]:
#     print(f"  Compiling: {model_name}...")
#     job = hub.submit_compile_job(
#         model=onnx_file,
#         device=device,
#         options="--target_runtime qnn --quantize_full_type int8",
#     )
#     print(f"  Job ID: {job.job_id} | Status: {job.get_status()}")
#     compiled = job.get_target_model()
#     out_path = f"{OUTPUT_DIR}/{model_name}_snapdragon.bin"
#     compiled.download(out_path)
#     print(f"  ✅ Downloaded: {out_path}")

print("\n✅ Export complete.")
print("📦 Upload quantized_models/ to onevoice-edge/models/ on your local machine.")
