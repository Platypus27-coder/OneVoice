"""
[Colab] Fine-tune MarianMT with Industrial Terminology
=======================================================
Run this script on Google Colab (T4 GPU recommended).
It fine-tunes Helsinki-NLP/opus-mt-vi-en on a custom dataset
of industrial/engineering terms to improve domain accuracy.

Usage:
  1. Upload this file + data/industrial_terms.csv to Colab
  2. Run all cells
  3. Download the fine-tuned model and place in onevoice-edge/models/marianmt/
"""

# ── Install dependencies ──────────────────────────────────────────────────
# !pip install transformers datasets sentencepiece sacremoses evaluate sacrebleu

import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MarianMTModel, MarianTokenizer, AdamW
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────
MODEL_NAME  = "Helsinki-NLP/opus-mt-vi-en"
DATA_FILE   = "industrial_terms.csv"
OUTPUT_DIR  = "marianmt_finetuned_vi_en"
EPOCHS      = 5
BATCH_SIZE  = 16
LR          = 5e-5
MAX_LENGTH  = 64
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ── Dataset ───────────────────────────────────────────────────────────────
class IndustrialTermsDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=64):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vi = row["vi_text"].strip().strip('"')
                en = row["en_text"].strip().strip('"')
                if vi and en:
                    self.samples.append((vi, en))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt = self.samples[idx]
        enc = self.tokenizer(
            src, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        with self.tokenizer.as_target_tokenizer():
            tgt_enc = self.tokenizer(
                tgt, max_length=self.max_length, padding="max_length",
                truncation=True, return_tensors="pt"
            )
        labels = tgt_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": labels,
        }

# ── Load Model & Tokenizer ────────────────────────────────────────────────
print("Loading model and tokenizer...")
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME).to(DEVICE)

dataset = IndustrialTermsDataset(DATA_FILE, tokenizer, MAX_LENGTH)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
optimizer = AdamW(model.parameters(), lr=LR)

print(f"Dataset size: {len(dataset)} term pairs")

# ── Training Loop ─────────────────────────────────────────────────────────
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} — Loss: {avg_loss:.4f}")

# ── Save Fine-tuned Model ─────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n✅ Fine-tuned model saved to: {OUTPUT_DIR}/")
print("Download this folder and place it in: onevoice-edge/models/marianmt/vi2en/")

# ── Quick Inference Test ──────────────────────────────────────────────────
model.eval()
test_sentences = [
    "Máy xúc số 3 đang bị lỗi thủy lực.",
    "Cần kiểm tra van an toàn ngay lập tức.",
    "Kỹ thuật viên đang hiệu chỉnh cảm biến nhiệt độ.",
]
print("\n── Inference test ──")
for vi in test_sentences:
    inputs = tokenizer(vi, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs)
    en = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"  VI: {vi}")
    print(f"  EN: {en}\n")
