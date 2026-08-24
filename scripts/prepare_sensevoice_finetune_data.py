"""Build leakage-safe FunASR/SenseVoice JSONL data from an English audio manifest.

This is a *training-preparation* tool.  It never modifies the source audio
manifest and it never reads the test split.  The resulting JSONL follows the
ChatML schema required by FunASR's ``train_ds.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable


SYSTEM_PROMPT = "You are a helpful assistant."
USER_PREFIX = "Speech transcription: <|startofspeech|>!"
USER_SUFFIX = "<|endofspeech|>"


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object on line {line_number} of {path}")
        rows.append(row)
    return rows


def _word_length(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text)))


def _qwen_length_function(model_name: str) -> Callable[[str], int]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Qwen text lengths require transformers. Install it or pass "
            "--text-length-mode words for an explicitly approximate dry run."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count(text: str) -> int:
        return max(1, len(tokenizer.encode(text, add_special_tokens=False)))

    return count


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(row: dict, audio_name: str, variant: str, root: Path, text_length: Callable[[str], int]) -> dict:
    transcript = str(row.get("text", "")).strip()
    if not transcript:
        raise ValueError(f"Missing transcript for {audio_name}")
    try:
        duration_s = float(row["duration_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing/invalid duration_s for {audio_name}") from exc
    if duration_s <= 0:
        raise ValueError(f"Non-positive duration_s for {audio_name}")

    audio_path = (root / audio_name).resolve()
    item_id = f"{Path(audio_name).stem}__{variant}"
    return {
        "id": item_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{USER_PREFIX}{audio_path}{USER_SUFFIX}"},
            {"role": "assistant", "content": transcript},
        ],
        # FunASR documents this as the number of 10 ms fbank frames.
        "speech_length": max(1, round(duration_s * 100)),
        "text_length": text_length(transcript),
        "onevoice": {
            "variant": variant,
            "utterance_id": row.get("utterance_id"),
            "speaker_id": row.get("speaker_id"),
            "risk_level": row.get("risk_level"),
            "noise_type": row.get("noise_type") if variant == "noisy" else None,
        },
    }


def build_split_records(
    rows: list[dict], split: str, root: Path, text_length: Callable[[str], int], include_clean: bool, include_noisy: bool
) -> list[dict]:
    selected = [row for row in rows if row.get("language") == "en" and row.get("split") == split]
    records: list[dict] = []
    seen_audio: set[str] = set()

    def append_unique(row: dict, field: str, variant: str) -> None:
        audio_name = str(row.get(field, "")).strip()
        if not audio_name:
            raise ValueError(f"Missing {field} in {split} split")
        if audio_name in seen_audio:
            return
        seen_audio.add(audio_name)
        records.append(_record(row, audio_name, variant, root, text_length))

    if include_clean:
        for row in selected:
            append_unique(row, "clean_audio", "clean")
    if include_noisy:
        for row in selected:
            append_unique(row, "noisy_audio", "noisy")
    if not records:
        raise ValueError(f"No English {split} records selected from manifest")
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare(
    manifest_path: Path,
    output_dir: Path,
    text_length: Callable[[str], int],
    include_clean: bool = True,
    include_noisy: bool = True,
) -> dict:
    rows = _read_jsonl(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = manifest_path.parent
    train = build_split_records(rows, "train", root, text_length, include_clean, include_noisy)
    dev = build_split_records(rows, "dev", root, text_length, include_clean, include_noisy)
    _write_jsonl(output_dir / "train.jsonl", train)
    _write_jsonl(output_dir / "dev.jsonl", dev)

    report = {
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": _source_sha256(manifest_path),
        "language": "en",
        "test_split_included": False,
        "include_clean": include_clean,
        "include_noisy": include_noisy,
        "splits": {
            "train": {"records": len(train), "variants": dict(Counter(r["onevoice"]["variant"] for r in train))},
            "dev": {"records": len(dev), "variants": dict(Counter(r["onevoice"]["variant"] for r in dev))},
        },
        "format": "FunASR ChatML / 10ms speech_length",
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-length-mode", choices=("qwen", "words"), default="qwen")
    parser.add_argument("--qwen-tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--include-clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-noisy", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not args.include_clean and not args.include_noisy:
        parser.error("Select at least one audio variant")
    length = _qwen_length_function(args.qwen_tokenizer) if args.text_length_mode == "qwen" else _word_length
    print(json.dumps(prepare(args.manifest, args.output_dir, length, args.include_clean, args.include_noisy), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
