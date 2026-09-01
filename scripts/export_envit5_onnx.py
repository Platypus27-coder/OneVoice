"""Export a local fine-tuned EnViT5 checkpoint to a portable ONNX Seq2Seq bundle.

This deliberately uses Optimum's standard encoder-decoder ONNX Runtime format.
EnViT5 is T5-based and therefore must not be exported as a decoder-only
``onnxruntime-genai`` model.  The resulting bundle stays local and is safe for
offline runtime only after its separate quality gate has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path


REQUIRED_FILES = ("config.json", "encoder_model.onnx")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_bundle(root: Path) -> list[Path]:
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    decoders = sorted(root.glob("decoder*.onnx"))
    if missing or not decoders:
        detail = missing + ([] if decoders else ["decoder*.onnx"])
        raise FileNotFoundError(f"Invalid ONNX Seq2Seq bundle {root}: missing {', '.join(detail)}")
    return sorted(path for path in root.rglob("*") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Local fine-tuned T5 checkpoint.")
    parser.add_argument("--output", required=True, type=Path, help="New local output bundle directory.")
    parser.add_argument("--direction", choices=["vi2en", "en2vi"], required=True)
    parser.add_argument("--smoke-text", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"Local EnViT5 checkpoint is incomplete: {source / 'config.json'}")
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"Output already exists: {output}; pass --resume after checking it")
        files = validate_bundle(output)
        manifest = output / "export_manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"Existing output has no export manifest: {manifest}")
        print(json.dumps({"status": "reused", "output": str(output), "files": len(files)}, ensure_ascii=False))
        return

    try:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install the export dependencies first: pip install 'optimum-onnx[onnxruntime]' "
            "onnxruntime transformers sentencepiece"
        ) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.export-", dir=output.parent))
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
        model = ORTModelForSeq2SeqLM.from_pretrained(
            str(source), export=True, provider="CPUExecutionProvider", local_files_only=True
        )
        model.save_pretrained(staging)
        tokenizer.save_pretrained(staging)
        files = validate_bundle(staging)

        # A fresh reload proves the exported graph and tokenizer can run without
        # consulting the source checkpoint or network.
        reloaded = ORTModelForSeq2SeqLM.from_pretrained(
            str(staging), provider="CPUExecutionProvider", local_files_only=True
        )
        tokenized = tokenizer(
            args.smoke_text, return_tensors="pt", truncation=True, max_length=args.max_length
        )
        generated = reloaded.generate(
            **tokenized, max_length=args.max_length, num_beams=5, early_stopping=True
        )
        smoke_output = tokenizer.decode(generated[0], skip_special_tokens=True)
        if not smoke_output.strip():
            raise RuntimeError("ONNX export smoke inference produced an empty translation")

        manifest = {
            "schema_version": 1,
            "kind": "onevoice.envit5.onnx_seq2seq",
            "direction": args.direction,
            "source": str(source),
            "source_config_sha256": sha256(source / "config.json"),
            "created_at": datetime.now(UTC).isoformat(),
            "smoke": {"input": args.smoke_text, "output": smoke_output},
            "files": [
                {
                    "path": path.relative_to(staging).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in files
            ],
        }
        (staging / "export_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(staging, output)
        print(json.dumps({"status": "exported", "output": str(output), "smoke": smoke_output}, ensure_ascii=False))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
