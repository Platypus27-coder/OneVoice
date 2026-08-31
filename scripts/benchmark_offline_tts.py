"""Benchmark the normal-path offline TTS contract for one direction.

The benchmark intentionally exercises the runtime TTS router instead of a
standalone backend.  It writes a resumable prediction ledger and fails closed
on silence, corrupt output, clipping, or an online/unavailable engine.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

try:
    import resource
except ImportError:  # pragma: no cover - Windows has no resource module
    resource = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from evaluation.reporting import create_run_manifest
from tts.tts_engine import TTSEngine


def peak_rss_mb() -> float:
    """Return process peak RSS in MB on Linux/macOS/Windows where available."""
    try:
        if resource is None:
            raise ImportError
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes. Colab is Linux.
        if value > 1024 * 1024:
            return value / (1024 * 1024)
        return value / 1024
    except (AttributeError, ImportError):
        try:
            import psutil

            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0


def load_prompts(path: Path, direction: str, limit: int | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    source_column = "en" if direction == "en2vi" else "vi"
    prompts = []
    for index, row in enumerate(rows, start=1):
        text = str(row.get(source_column, "")).strip()
        if text:
            prompts.append({"prompt_id": str(row.get("id") or row.get("utterance_id") or index), "text": text})
    if limit is not None:
        prompts = prompts[:limit]
    if not prompts:
        raise ValueError(f"No {source_column!r} prompts found in {path}")
    return prompts


def load_partial(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid partial TTS report {path}:{line_number}") from exc
        if not isinstance(row, dict) or not row.get("prompt_id"):
            raise ValueError(f"Invalid partial TTS row {path}:{line_number}")
        rows.append(row)
    return rows


def write_partial(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=("vi2en", "en2vi"), required=True)
    parser.add_argument("--prompt-csv", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--profile", choices=("development", "edge"), default="edge")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-audio", action="store_true", help="Keep validated WAVs under report-dir/audio")
    args = parser.parse_args()
    if args.max_samples <= 0 or args.progress_every < 0 or args.save_every < 0:
        parser.error("max-samples must be positive; progress/save intervals must be non-negative")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prompt_csv = args.prompt_csv
    if prompt_csv is None:
        prompt_csv = Path(config["pipeline"]["construction_data_dir"]) / "test.csv"
    prompts = load_prompts(prompt_csv, args.direction, args.max_samples)
    output = args.report_dir
    output.mkdir(parents=True, exist_ok=True)
    partial_path = output / "predictions.partial.jsonl"
    predictions = load_partial(partial_path) if args.resume else []
    completed = {str(row["prompt_id"]) for row in predictions}
    if len(completed) != len(predictions):
        raise ValueError("Duplicate prompt_id in the resume ledger")
    expected_ids = {row["prompt_id"] for row in prompts}
    if not completed.issubset(expected_ids):
        raise ValueError("Resume ledger contains prompts outside this benchmark")

    # Offline is mandatory here: development is only a profile label, not permission
    # to use gTTS. The engine may still select a local system voice/eSpeak fallback.
    tts = TTSEngine(config, profile=args.profile, offline=True)
    tts.load(direction=args.direction)
    engine_name = tts.engine_name(args.direction)
    if engine_name in {"gtts", "gtts-development", "unavailable", "None"}:
        raise RuntimeError(f"Offline TTS selected an invalid engine: {engine_name}")
    print(f"[TTS benchmark] {args.direction}: {len(prompts)} prompts; engine={engine_name}", flush=True)
    audio_dir = output / "audio"
    if args.keep_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()
    for index, prompt in enumerate(prompts, start=1):
        prompt_id = prompt["prompt_id"]
        if prompt_id in completed:
            continue
        row = {"prompt_id": prompt_id, "text": prompt["text"], "status": "fail"}
        temporary_path: Path | None = None
        started = time.perf_counter()
        try:
            audio, sample_rate = tts.synthesize(prompt["text"], args.direction)
            elapsed_ms = (time.perf_counter() - started) * 1000
            audio = np.asarray(audio, dtype=np.float32)
            peak = float(np.max(np.abs(audio), initial=0.0))
            if tts.is_silence(audio):
                raise RuntimeError("silence")
            if not np.isfinite(audio).all():
                raise RuntimeError("non-finite samples")
            if peak >= 0.9999:
                raise RuntimeError(f"clipping peak={peak:.6f}")
            if int(sample_rate) <= 0:
                raise RuntimeError(f"invalid sample rate {sample_rate}")
            if args.keep_audio:
                destination = audio_dir / f"{index:04d}_{prompt_id}.wav"
                sf.write(destination, audio, int(sample_rate))
                decoded, decoded_sr = sf.read(destination, dtype="float32", always_2d=False)
            else:
                with tempfile.NamedTemporaryFile(suffix=".wav", dir=output, delete=False) as handle:
                    temporary_path = Path(handle.name)
                sf.write(temporary_path, audio, int(sample_rate))
                decoded, decoded_sr = sf.read(temporary_path, dtype="float32", always_2d=False)
            if decoded.size == 0 or int(decoded_sr) != int(sample_rate) or tts.is_silence(decoded):
                raise RuntimeError("WAV decode/sample-rate/silence validation failed")
            row.update({
                "status": "pass", "elapsed_ms": round(elapsed_ms, 3),
                "sample_rate": int(sample_rate), "samples": int(audio.size),
                "peak_abs": round(peak, 6), "engine": tts.engine_name(args.direction),
            })
        except Exception as exc:
            row.update({"elapsed_ms": round((time.perf_counter() - started) * 1000, 3), "engine": tts.engine_name(args.direction), "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if temporary_path:
                temporary_path.unlink(missing_ok=True)
        predictions.append(row)
        completed.add(prompt_id)
        if args.save_every and len(predictions) % args.save_every == 0:
            write_partial(partial_path, predictions)
        if args.progress_every and (index % args.progress_every == 0 or index == len(prompts)):
            print(f"[TTS benchmark] processed {index}/{len(prompts)}", flush=True)

    latencies = sorted(float(row["elapsed_ms"]) for row in predictions if row.get("status") == "pass")
    passes = sum(row.get("status") == "pass" for row in predictions)
    failures = [row for row in predictions if row.get("status") != "pass"]
    percentile = lambda p: latencies[min(round((len(latencies) - 1) * p), len(latencies) - 1)] if latencies else None
    aggregate = {
        "schema_version": 1, "direction": args.direction, "profile": args.profile,
        "offline": True, "prompt_csv": str(prompt_csv.resolve()),
        "samples": len(predictions), "passed_samples": passes, "failed_samples": len(failures),
        "engine": tts.engine_name(args.direction),
        "sample_rates": dict(sorted((str(rate), sum(row.get("sample_rate") == rate for row in predictions)) for rate in {row.get("sample_rate") for row in predictions if row.get("sample_rate")})),
        "silence_or_validation_failures": len(failures),
        "latency_p50_ms": percentile(0.50), "latency_p95_ms": percentile(0.95),
        "peak_rss_mb": round(peak_rss_mb(), 3),
        "elapsed_seconds": round(time.perf_counter() - started_all, 3),
        "failures": failures[:20],
        "passed": len(predictions) == len(prompts) and not failures and tts.engine_name(args.direction) not in {"gtts", "gtts-development", "unavailable"},
        "scope": "offline development/edge system voice benchmark; not a production voice-quality claim",
    }
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in predictions for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(predictions)
    (output / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    create_run_manifest(output / "run_manifest.json", command="benchmark_offline_tts", inputs=[args.config, prompt_csv], metadata={**vars(args), "engine": tts.engine_name(args.direction), "aggregate": str(output / "aggregate.json")})
    partial_path.unlink(missing_ok=True)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    if not aggregate["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
