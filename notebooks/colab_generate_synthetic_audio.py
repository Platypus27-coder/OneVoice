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
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm

# Check environment
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    drive.mount('/content/drive')
    OUTPUT_ROOT = "/content/drive/MyDrive/onevoice_audio_v1"
else:
    OUTPUT_ROOT = "./onevoice_audio_v1"

print(f"Output Directory: {OUTPUT_ROOT}")

# ─────────────────────────────────────────────
# CELL 2 — Config
# ─────────────────────────────────────────────
DATA_DIR     = "/content/OneVoice/data/onevoice_construction_v2" if IN_COLAB else "../data/onevoice_construction_v2"
NOISE_DIR    = os.path.join(OUTPUT_ROOT, "noise_bank")
CLEAN_DIR    = os.path.join(OUTPUT_ROOT, "clean")
NOISY_DIR    = os.path.join(OUTPUT_ROOT, "noisy")
MANIFEST     = os.path.join(OUTPUT_ROOT, "manifest.jsonl")

for d in [OUTPUT_ROOT, NOISE_DIR, CLEAN_DIR, NOISY_DIR]:
    os.makedirs(d, exist_ok=True)

SAMPLES_PER_TEXT   = 2          # each utterance → 2 noisy versions
SAMPLE_RATE        = 16000
MAX_UTTERANCES     = None       # None = all 8,064

VI_SPEAKERS = [
    "vi-VN-HoaiMyNeural",
    "vi-VN-NamMinhNeural",
]

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
    for fname, url in NOISE_URLS.items():
        dst = os.path.join(NOISE_DIR, fname)
        if ok_wav(dst):
            continue
        if os.path.exists(dst):
            os.remove(dst)
        print(f"  Downloading {fname}...")
        os.system(f'wget -q -O "{dst}" "{url}"')
        if not ok_wav(dst):
            print(f"  ⚡ Auto-generating synthetic noise fallback for {fname}...")
            if os.path.exists(dst):
                os.remove(dst)
            sf.write(dst, generate_synthetic_noise(fname), SAMPLE_RATE)

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

def tts_one(text: str, out_wav_path: str, voice: str) -> bool:
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
                edge_tts.Communicate(text, voice).save(tmp_mp3)
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

def apply_rir(speech: np.ndarray, sr: int) -> np.ndarray:
    delay = int(sr * random.uniform(0.03, 0.08))
    echo = np.zeros_like(speech)
    echo[delay:] = speech[:-delay] * random.uniform(0.2, 0.4)
    out = speech + echo
    return out / (np.max(np.abs(out)) + 1e-9)

def mix_noise(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(speech):
        repeats = int(np.ceil(len(speech) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[:len(speech)]
    
    rms_speech = np.sqrt(np.mean(speech**2) + 1e-9)
    rms_noise  = np.sqrt(np.mean(noise**2) + 1e-9)
    scaled_noise = noise * (rms_speech / (rms_noise * (10**(snr_db / 20))))
    mixed = speech + scaled_noise
    return np.clip(mixed / (np.max(np.abs(mixed)) + 1e-9), -1.0, 1.0)

def augment(audio: np.ndarray) -> np.ndarray:
    gain = random.uniform(-3.0, 3.0)
    audio = audio * (10**(gain / 20))
    if random.random() < 0.05:
        threshold = random.uniform(0.7, 0.95)
        audio = np.clip(audio, -threshold, threshold)
    return audio

# ─────────────────────────────────────────────
# CELL 5 — Main Generation Loop
# ─────────────────────────────────────────────
def generate_dataset():
    download_noise_bank()

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
        os.path.join(os.path.dirname(__file__), "../data/synthetic_dataset_10k.csv"),
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

    total_generated = len(existing)
    skipped = 0

    with open(MANIFEST, "a", encoding="utf-8") as mf:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating synthetic audio"):
            uid  = str(row.get("utterance_id", row.get("id", f"utt_{_}")))
            vi   = str(row.get("vi", row.get("vi_text", "")))
            en   = str(row.get("en", row.get("en_text", "")))
            dom  = str(row.get("domain", "unknown"))
            inte = str(row.get("intent", "unknown"))
            risk = str(row.get("risk_level", "unknown"))
            spl  = str(row.get("split", "train"))

            # Skip if all variants already generated
            if all(f"{uid}_n{v+1:02d}.wav" in existing for v in range(SAMPLES_PER_TEXT)):
                continue

            # TTS Clean WAV synthesis
            cf = f"{uid}_clean.wav"
            cp = os.path.join(CLEAN_DIR, cf)
            if not os.path.exists(cp):
                txt = clean_text(vi)
                if not txt:
                    skipped += 1
                    continue
                voice = random.choice(VI_SPEAKERS)
                if not tts_one(txt, cp, voice):
                    skipped += 1
                    continue

            # Load clean audio PCM WAV
            try:
                ca, _ = librosa.load(cp, sr=SAMPLE_RATE, mono=True)
            except Exception as e:
                print(f"⚠ Could not read clean audio {cp}: {e}")
                skipped += 1
                continue

            rev = apply_rir(ca, SAMPLE_RATE)
            voice = random.choice(VI_SPEAKERS)

            # Generate SAMPLES_PER_TEXT noisy variants
            for v in range(SAMPLES_PER_TEXT):
                nf = f"{uid}_n{v+1:02d}.wav"
                np_path = os.path.join(NOISY_DIR, nf)

                if nf in existing and os.path.exists(np_path):
                    continue

                nn     = random.choice(noises)
                snr    = random.choice(SNR_OPTIONS)
                rev_on = random.random() > 0.35

                mixed = augment(mix_noise(rev if rev_on else ca, NC[nn], snr))
                sf.write(np_path, mixed, SAMPLE_RATE)

                entry = {
                    "audio": nf,
                    "clean_audio": cf,
                    "text": vi,
                    "translation": en,
                    "domain": dom,
                    "intent": inte,
                    "risk_level": risk,
                    "split": spl,
                    "speaker_id": voice,
                    "noise_type": nn.replace(".wav", ""),
                    "snr_db": snr,
                    "reverb": rev_on,
                    "rir_id": "simulated_echo" if rev_on else "none",
                    "synthetic_speech": True,
                    "synthetic_noise_mix": True,
                    "sample_rate": SAMPLE_RATE,
                }
                mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
                mf.flush()
                existing.add(nf)
                total_generated += 1

    print(f"\n✅ Generation finished! Total: {total_generated} samples | Skipped: {skipped}")

if __name__ == "__main__":
    generate_dataset()
