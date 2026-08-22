# ============================================================
# OneVoice Edge — Synthetic Noisy Construction Speech Generator
# Run on: Google Colab (T4 GPU) or Kaggle (P100 GPU)
# Output: ~16,000 .wav + manifest.jsonl
# ============================================================

# ─────────────────────────────────────────────
# CELL 1 — Mount Drive & Install dependencies
# ─────────────────────────────────────────────
import os
import sys
import json
import random
import re
import time
import unicodedata
import asyncio
import hashlib
import urllib.request
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm

from reconcile_manifest_splits import normalized_text, SPLIT_PRIORITY

# This script is intentionally Colab-agnostic.  The calling notebook mounts
# Drive in its own kernel, then passes durable paths through the environment.
OUTPUT_ROOT = os.environ.get("ONEVOICE_OUTPUT_ROOT", "./onevoice_audio_v2_1")
DATA_DIR_DEFAULT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "onevoice_construction_v2")
)
DATA_DIR = os.environ.get("ONEVOICE_DATA_DIR", DATA_DIR_DEFAULT)

print(f"Output Directory: {OUTPUT_ROOT}")

# ─────────────────────────────────────────────
# CELL 2 — Config
# ─────────────────────────────────────────────
NOISE_DIR    = os.path.join(OUTPUT_ROOT, "noise_bank")
CLEAN_DIR    = os.path.join(OUTPUT_ROOT, "clean")
NOISY_DIR    = os.path.join(OUTPUT_ROOT, "noisy")
MANIFEST     = os.path.join(OUTPUT_ROOT, "manifest.jsonl")
NOISE_MANIFEST = os.path.join(NOISE_DIR, "sources.json")
STATE_PATH = os.path.join(OUTPUT_ROOT, "generation_state.json")

for d in [OUTPUT_ROOT, NOISE_DIR, CLEAN_DIR, NOISY_DIR]:
    os.makedirs(d, exist_ok=True)

SAMPLES_PER_TEXT   = int(os.environ.get("ONEVOICE_SAMPLES_PER_TEXT", "2"))
SAMPLE_RATE        = 16000
MAX_UTTERANCES     = int(os.environ["ONEVOICE_MAX_UTTERANCES"]) if os.environ.get("ONEVOICE_MAX_UTTERANCES") else None

VI_SPEAKERS = [
    "vi-VN-HoaiMyNeural",
    "vi-VN-NamMinhNeural",
    "en-US-AvaMultilingualNeural",
    "en-US-AndrewMultilingualNeural",
    "en-US-EmmaMultilingualNeural",
    "en-US-BrianMultilingualNeural",
]

EN_SPEAKERS = [
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-AU-NatashaNeural",
    "en-AU-WilliamNeural",
    "en-IN-NeerjaNeural",
]

# v2.1 is a new dataset. Never append EN samples to onevoice_audio_v1.
GENERATE_LANGUAGES = tuple(
    item.strip() for item in os.environ.get("ONEVOICE_LANGUAGES", "en").split(",")
    if item.strip() in {"vi", "en"}
)

NOISE_CLASSES = [
    "excavator.wav",
    "angle_grinder.wav",
    "drilling.wav",
    "hammer.wav",
    "diesel_engine.wav",
    "generator.wav",
    "truck.wav",
    "wind.wav",
    "worker_babble.wav",
]

SNR_OPTIONS = [0, 5, 10, 15, 20]
RATE_OPTIONS = ["-15%", "-8%", "+0%", "+8%", "+15%"]
RIR_PROFILES = {
    "small_room": ((0.025, 0.22), (0.051, 0.10)),
    "concrete_corridor": ((0.045, 0.30), (0.092, 0.16), (0.141, 0.08)),
    "open_structure": ((0.072, 0.16),),
    "metal_enclosure": ((0.018, 0.28), (0.036, 0.18), (0.055, 0.11)),
}

# ─────────────────────────────────────────────
# CELL 3 — Download & Auto-Repair Noise Bank
# ─────────────────────────────────────────────
ESC50_BASE = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio"

NOISE_URLS = {
    "excavator.wav":     f"{ESC50_BASE}/1-116765-A-41.wav",
    "angle_grinder.wav": f"{ESC50_BASE}/3-156897-A-13.wav",
    "drilling.wav":      f"{ESC50_BASE}/4-182368-A-12.wav",
    "hammer.wav":        f"{ESC50_BASE}/3-149189-A-13.wav",
    "diesel_engine.wav": f"{ESC50_BASE}/1-26143-A-43.wav",
    "generator.wav":     f"{ESC50_BASE}/2-109371-A-43.wav",
    "truck.wav":         f"{ESC50_BASE}/5-219213-A-11.wav",
    "wind.wav":          f"{ESC50_BASE}/1-179701-A-25.wav",
    "worker_babble.wav": f"{ESC50_BASE}/4-167642-A-26.wav",
}
NOISE_ORIGINS = {}

def generate_synthetic_noise(noise_type: str, duration_sec: int = 10, sr: int = 16000) -> np.ndarray:
    """Bulletproof fallback: generate realistic industrial noise if download fails/corrupt."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec))
    if "grinder" in noise_type or "drill" in noise_type:
        noise = 0.6 * np.sin(2 * np.pi * 3200 * t + np.sin(2 * np.pi * 50 * t)) + 0.4 * np.random.normal(0, 1, len(t))
    elif any(k in noise_type for k in ("engine", "excavator", "generator", "truck")):
        noise = 0.5 * np.sin(2 * np.pi * 60 * t) + 0.3 * np.sin(2 * np.pi * 120 * t) + 0.3 * np.random.normal(0, 1, len(t))
    elif "hammer" in noise_type:
        noise = 0.2 * np.random.normal(0, 1, len(t))
        pulse_idx = np.arange(0, len(t), int(sr * 0.8), dtype=int)
        for idx in pulse_idx:
            end = min(idx + int(sr * 0.05), len(t))
            noise[idx:end] += np.random.normal(0, 3, end - idx)
    else: # wind / babble
        noise = np.convolve(np.random.normal(0, 1, len(t)), [0.05, -0.09, 0.05], mode='same')
    return np.clip(noise / (np.max(np.abs(noise)) + 1e-9), -1.0, 1.0)

def ok_wav(p):
    try:
        return os.path.exists(p) and os.path.getsize(p) > 10000 and sf.read(p) is not None
    except Exception:
        return False

def download_noise_bank():
    print("[Noise Bank] Verifying industrial noise files...")
    if os.path.exists(NOISE_MANIFEST):
        try:
            with open(NOISE_MANIFEST, "r", encoding="utf-8") as handle:
                NOISE_ORIGINS.update(json.load(handle))
        except (OSError, json.JSONDecodeError):
            pass
    for fname, url in NOISE_URLS.items():
        dst = os.path.join(NOISE_DIR, fname)
        if ok_wav(dst):
            NOISE_ORIGINS.setdefault(fname, {"kind": "cached_unknown", "url": url})
            continue
        if os.path.exists(dst):
            os.remove(dst)
        print(f"  Downloading {fname}...")
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception:
            pass
        if not ok_wav(dst):
            print(f"  ⚡ Auto-generating synthetic noise fallback for {fname}...")
            if os.path.exists(dst):
                os.remove(dst)
            sf.write(dst, generate_synthetic_noise(fname), SAMPLE_RATE)
            NOISE_ORIGINS[fname] = {"kind": "synthetic_fallback", "url": url}
        else:
            NOISE_ORIGINS[fname] = {"kind": "downloaded", "url": url}
    with open(NOISE_MANIFEST, "w", encoding="utf-8") as handle:
        json.dump(NOISE_ORIGINS, handle, ensure_ascii=False, indent=2)

# ─────────────────────────────────────────────
# CELL 4 — TTS + Audio Functions
# ─────────────────────────────────────────────
try:
    import edge_tts
    import nest_asyncio
    nest_asyncio.apply()
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False
    print("⚠ edge-tts or nest_asyncio not installed. Please run: pip install edge-tts nest_asyncio pydub")

def clean_text(text):
    if not text or pd.isna(text):
        return None
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF.,!?;:()\'\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text if len(text) >= 2 else None

def tts_one(text: str, out_wav_path: str, voice: str, rate: str) -> bool:
    """
    Synthesize ONE file via edge_tts.
    CRITICAL FIX: edge_tts outputs MP3 data. We MUST save to temporary .mp3 first,
    then decode & convert to proper 16kHz mono PCM WAV via librosa / soundfile / pydub.
    """
    if not HAS_EDGE_TTS:
        return False

    tmp_mp3 = out_wav_path.replace(".wav", "_tmp.mp3")
    
    for attempt in range(3):
        try:
            if attempt > 0:
                time.sleep(1.5 * attempt)
            
            # Synthesize MP3 stream
            loop = asyncio.get_event_loop()
            loop.run_until_complete(
                edge_tts.Communicate(text, voice, rate=rate).save(tmp_mp3)
            )

            # Convert MP3 -> 16kHz Mono WAV
            if os.path.exists(tmp_mp3) and os.path.getsize(tmp_mp3) > 100:
                try:
                    # Load MP3 with librosa (uses ffmpeg / audioread)
                    audio, _ = librosa.load(tmp_mp3, sr=SAMPLE_RATE, mono=True)
                    sf.write(out_wav_path, audio, SAMPLE_RATE)
                    if os.path.exists(tmp_mp3):
                        os.remove(tmp_mp3)
                    return True
                except Exception as load_err:
                    # Fallback to pydub if librosa load fails
                    from pydub import AudioSegment
                    seg = AudioSegment.from_file(tmp_mp3)
                    seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(1)
                    seg.export(out_wav_path, format="wav")
                    if os.path.exists(tmp_mp3):
                        os.remove(tmp_mp3)
                    return True

        except Exception as e:
            if attempt == 2:
                print(f"  [TTS FAIL] {os.path.basename(out_wav_path)}: {e}")
                if os.path.exists(tmp_mp3):
                    try: os.remove(tmp_mp3)
                    except: pass
                return False
    return False

def apply_rir(speech: np.ndarray, sr: int, rir_id: str) -> np.ndarray:
    out = speech.astype(np.float32).copy()
    for delay_s, gain in RIR_PROFILES[rir_id]:
        delay = int(sr * delay_s)
        if 0 < delay < len(speech):
            out[delay:] += speech[:-delay] * gain
    return out / (np.max(np.abs(out)) + 1e-9)

def mix_noise(speech: np.ndarray, noise: np.ndarray, snr_db: float, rng: random.Random) -> tuple[np.ndarray, float, int]:
    if len(noise) < len(speech):
        repeats = int(np.ceil(len(speech) / len(noise)))
        noise = np.tile(noise, repeats)
    max_offset = max(0, len(noise) - len(speech))
    crop_offset = rng.randint(0, max_offset) if max_offset else 0
    noise = noise[crop_offset:crop_offset + len(speech)]
    
    rms_speech = np.sqrt(np.mean(speech**2) + 1e-9)
    rms_noise  = np.sqrt(np.mean(noise**2) + 1e-9)
    scaled_noise = noise * (rms_speech / (rms_noise * (10**(snr_db / 20))))
    realized_snr = 20 * np.log10(
        (np.sqrt(np.mean(speech**2)) + 1e-9)
        / (np.sqrt(np.mean(scaled_noise**2)) + 1e-9)
    )
    mixed = speech + scaled_noise
    mixed = np.clip(mixed / (np.max(np.abs(mixed)) + 1e-9), -1.0, 1.0)
    return mixed, float(realized_snr), crop_offset

def augment(audio: np.ndarray, rng: random.Random) -> tuple[np.ndarray, float]:
    """Apply a linear gain only; nonlinear clipping would invalidate realized SNR."""
    requested_gain = rng.uniform(-3.0, 3.0)
    peak = float(np.max(np.abs(audio)) + 1e-9)
    max_safe_gain = 20 * np.log10(0.95 / peak)
    applied_gain = min(requested_gain, max_safe_gain)
    return (audio * (10**(applied_gain / 20))).astype(np.float32), float(applied_gain)

# ─────────────────────────────────────────────
# CELL 5 — Main Generation Loop
# ─────────────────────────────────────────────
def write_generation_state(status: str, generated: int, skipped: int, expected: int) -> None:
    payload = {
        "status": status,
        "generated_noisy": generated,
        "skipped": skipped,
        "expected_noisy": expected,
        "languages": list(GENERATE_LANGUAGES),
        "samples_per_text": SAMPLES_PER_TEXT,
        "manifest": MANIFEST,
    }
    temporary = STATE_PATH + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, STATE_PATH)


def generate_dataset():
    download_noise_bank()
    for language, speakers in (("vi", VI_SPEAKERS), ("en", EN_SPEAKERS)):
        if language in GENERATE_LANGUAGES and len(set(speakers)) < 6:
            raise ValueError(f"{language} generation requires at least six distinct real voice IDs")

    # Load existing manifest entries
    existing = set()
    if os.path.exists(MANIFEST):
        with open(MANIFEST, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        existing.add(json.loads(line.strip()).get("audio"))
                    except Exception:
                        pass
        print(f"🔄 Resuming checkpoint: {len(existing)} samples done.")

    # Find dataset CSV
    possible_csvs = [
        os.path.join(DATA_DIR, "utterances_all.csv"),
        os.path.join(os.path.dirname(__file__), "../data/onevoice_construction_v2/utterances_all.csv"),
    ]
    csv_path = None
    for p in possible_csvs:
        if os.path.exists(p):
            csv_path = p
            break

    if not csv_path:
        raise FileNotFoundError(f"Cannot find utterances CSV in {possible_csvs}")

    df = pd.read_csv(csv_path)
    if MAX_UTTERANCES:
        df = df.head(MAX_UTTERANCES)
    split_by_english_text = {}
    if "en" in GENERATE_LANGUAGES:
        groups = {}
        for index, value in df["en"].items():
            groups.setdefault(normalized_text(value), set()).add(str(df.at[index, "split"]))
        split_by_english_text = {
            text: max(splits, key=lambda split: SPLIT_PRIORITY.get(split, -1))
            for text, splits in groups.items()
        }
    print(f"Loaded {len(df)} utterances from {os.path.basename(csv_path)}.")

    # Cache noise files
    NC = {}
    for nc in NOISE_CLASSES:
        p = os.path.join(NOISE_DIR, nc)
        if not os.path.exists(p):
            continue
        try:
            NC[nc], _ = librosa.load(p, sr=SAMPLE_RATE, mono=True)
        except Exception:
            s = generate_synthetic_noise(nc)
            sf.write(p, s, SAMPLE_RATE)
            NC[nc] = s

    if not NC:
        NC["_silence"] = np.zeros(SAMPLE_RATE)

    noises = list(NC.keys())
    print(f"Active noise types: {noises}")

    expected_total = len(df) * len(GENERATE_LANGUAGES) * SAMPLES_PER_TEXT
    total_generated = len(existing)
    skipped = 0
    write_generation_state("running", total_generated, skipped, expected_total)

    with open(MANIFEST, "a", encoding="utf-8") as mf:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating synthetic audio"):
            uid  = str(row.get("utterance_id", row.get("id", f"utt_{_}")))
            vi   = str(row.get("vi", row.get("vi_text", "")))
            en   = str(row.get("en", row.get("en_text", "")))
            dom  = str(row.get("domain", "unknown"))
            inte = str(row.get("intent", "unknown"))
            risk = str(row.get("risk_level", "unknown"))
            spl  = str(row.get("split", "train"))
            pattern_id = str(row.get("frame_pattern_id", uid))
            pair_id = str(row.get("pair_id", ""))

            for language in GENERATE_LANGUAGES:
                source_text = vi if language == "vi" else en
                translation = en if language == "vi" else vi
                speakers = VI_SPEAKERS if language == "vi" else EN_SPEAKERS
                digest = hashlib.sha256(f"{uid}:{language}".encode()).digest()
                voice = speakers[int.from_bytes(digest[:2], "big") % len(speakers)]
                speaking_rate = RATE_OPTIONS[digest[2] % len(RATE_OPTIONS)]
                rir_id = list(RIR_PROFILES)[digest[3] % len(RIR_PROFILES)]
                prefix = f"{uid}_{language}"

                if all(
                    f"{prefix}_n{v+1:02d}.wav" in existing
                    and os.path.exists(os.path.join(NOISY_DIR, f"{prefix}_n{v+1:02d}.wav"))
                    for v in range(SAMPLES_PER_TEXT)
                ):
                    continue
                cf = f"{prefix}_clean.wav"
                cp = os.path.join(CLEAN_DIR, cf)
                if not os.path.exists(cp):
                    txt = clean_text(source_text)
                    if not txt or not tts_one(txt, cp, voice, speaking_rate):
                        skipped += 1
                        continue
                try:
                    ca, _ = librosa.load(cp, sr=SAMPLE_RATE, mono=True)
                except Exception as e:
                    print(f"⚠ Could not read clean audio {cp}: {e}")
                    skipped += 1
                    continue

                rev = apply_rir(ca, SAMPLE_RATE, rir_id)
                for v in range(SAMPLES_PER_TEXT):
                    nf = f"{prefix}_n{v+1:02d}.wav"
                    np_path = os.path.join(NOISY_DIR, nf)
                    if nf in existing and os.path.exists(np_path):
                        continue
                    sample_seed = int.from_bytes(
                        hashlib.sha256(f"{uid}:{language}:{v}".encode()).digest()[:8], "big"
                    )
                    rng = random.Random(sample_seed)
                    nn = rng.choice(noises)
                    target_snr = rng.choice(SNR_OPTIONS)
                    rev_on = rng.random() > 0.35
                    mixed, realized_snr, crop_offset = mix_noise(
                        rev if rev_on else ca, NC[nn], target_snr, rng
                    )
                    mixed, applied_gain_db = augment(mixed, rng)
                    sf.write(np_path, mixed, SAMPLE_RATE)
                    entry = {
                        "utterance_id": uid,
                        "pair_id": pair_id,
                        "frame_pattern_id": pattern_id,
                        "audio": nf,
                        "clean_audio": cf,
                        "noisy_audio": nf,
                        "language": language,
                        "text": source_text,
                        "translation": translation,
                        "domain": dom,
                        "intent": inte,
                        "risk_level": risk,
                        "split": split_by_english_text.get(normalized_text(en), spl) if language == "en" else spl,
                        "source_split": spl,
                        "speaker_id": voice,
                        "voice_id": voice,
                        "voice_engine": "edge-tts",
                        "speaking_rate": speaking_rate,
                        "noise_type": nn.replace(".wav", ""),
                        "noise_source": NOISE_ORIGINS.get(nn, {"kind": "unknown"}),
                        "target_snr_db": target_snr,
                        "realized_snr_db": round(realized_snr, 3),
                        "applied_gain_db": round(applied_gain_db, 3),
                        "noise_crop_offset_samples": crop_offset,
                        "reverb": rev_on,
                        "rir_id": rir_id if rev_on else "none",
                        "synthetic_speech": True,
                        "synthetic_noise_mix": True,
                        "sample_rate": SAMPLE_RATE,
                        "duration_s": round(len(mixed) / SAMPLE_RATE, 3),
                        "generation_seed": sample_seed,
                    }
                    mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    mf.flush()
                    existing.add(nf)
                    total_generated += 1
                    if total_generated % 25 == 0:
                        write_generation_state("running", total_generated, skipped, expected_total)

    write_generation_state("complete", total_generated, skipped, expected_total)
    print(f"\n✅ Generation finished! Total: {total_generated} samples | Skipped: {skipped}")

if __name__ == "__main__":
    generate_dataset()
