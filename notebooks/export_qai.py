"""
[Colab] Quantize & Compile Production Models for Qualcomm Snapdragon
====================================================================
Uses Qualcomm AI Hub to compile OneVoice Edge target models for Snapdragon NPU (INT8).

Target Architecture Models:
  1. ASR (VI): GIPFormer INT8 ONNX (g-group-ai-lab/gipformer-65M-rnnt)
  2. ASR (EN): SenseVoice Small ONNX (iic/SenseVoiceSmall)
  3. MT (VI↔EN): VietAI EnViT5 ONNX (VietAI/envit5-translation)

Prerequisites:
  - Set QAI_HUB_API_TOKEN in Colab Secrets or environment variable
"""

import os
import torch
import onnx
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

os.environ["QAI_HUB_API_TOKEN"] = os.getenv("QAI_HUB_API_TOKEN", "YOUR_API_KEY_HERE")

OUTPUT_DIR = "quantized_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Export VietAI/envit5-translation to ONNX for Snapdragon NPU
# ═══════════════════════════════════════════════════════════════════════════
print("── Exporting VietAI/envit5-translation to ONNX ──")

MODEL_NAME = "VietAI/envit5-translation"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).eval()

sample_text = "vi: Máy xúc số 3 đang bị lỗi thủy lực."
inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)

onnx_mt_path = f"{OUTPUT_DIR}/envit5_translation.onnx"
torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    onnx_mt_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    opset_version=14,
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
    },
)
print(f"✅ VietAI/envit5-translation ONNX saved: {onnx_mt_path}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Submit to Qualcomm AI Hub for INT8 Quantization
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Submitting to Qualcomm AI Hub ──")

# Uncomment when QAI_HUB_API_TOKEN is active:
# import qai_hub as hub
# device = hub.Device("Snapdragon 8 Gen 3")
#
# job = hub.submit_compile_job(
#     model=onnx_mt_path,
#     device=device,
#     options="--target_runtime qnn --quantize_full_type int8",
# )
# print(f"  Job ID: {job.job_id} | Status: {job.get_status()}")

print("\n✅ Qualcomm AI Hub Export Setup Complete.")
