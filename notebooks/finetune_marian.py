"""Fine-tune EnViT5 without train/dev/test leakage.

Despite the legacy filename this trains VietAI/envit5-translation, not MarianMT.
Test data is never opened by this training program.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.reporting import create_run_manifest


class ConstructionTranslationDataset(Dataset):
    def __init__(self, filepath: str | Path, tokenizer, max_length: int = 128):
        self.samples: list[tuple[str, str]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                vi, en = row.get("vi", "").strip(), row.get("en", "").strip()
                if vi and en:
                    self.samples.extend((("vi: " + vi, en), ("en: " + en, vi)))
        if not self.samples:
            raise ValueError(f"No valid vi/en pairs found in {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source, target = self.samples[index]
        encoded = self.tokenizer(
            source,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_encoded = self.tokenizer(
            target,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = target_encoded["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels,
        }


def evaluate_loss(model, loader, device: str) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            total += float(model(**batch).loss.item())
    model.train()
    return total / max(len(loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="VietAI/envit5-translation")
    parser.add_argument("--train", default="data/onevoice_construction_v2/train.csv")
    parser.add_argument("--dev", default="data/onevoice_construction_v2/dev.csv")
    parser.add_argument("--output", default="models/envit5_finetuned_construction")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    train_data = ConstructionTranslationDataset(args.train, tokenizer, args.max_length)
    dev_data = ConstructionTranslationDataset(args.dev, tokenizer, args.max_length)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best_dev = float("inf")
    history = []

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        dev_loss = evaluate_loss(model, dev_loader, device)
        print(
            f"Epoch {epoch + 1}: train_loss={total / len(train_loader):.4f}, "
            f"dev_loss={dev_loss:.4f}"
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total / len(train_loader),
                "dev_loss": dev_loss,
            }
        )
        model.save_pretrained(output / f"epoch-{epoch + 1}")
        tokenizer.save_pretrained(output / f"epoch-{epoch + 1}")
        if dev_loss < best_dev:
            best_dev = dev_loss
            model.save_pretrained(output / "best")
            tokenizer.save_pretrained(output / "best")

    (output / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    create_run_manifest(
        output / "run_manifest.json",
        command="finetune_envit5",
        inputs=[args.train, args.dev, output / "best"],
        metadata={**vars(args), "device": device, "best_dev_loss": best_dev},
    )
    print(f"Best dev loss: {best_dev:.4f}; checkpoint: {output / 'best'}")


if __name__ == "__main__":
    main()
