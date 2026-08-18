
# ============================================================
# OneVoice Edge — Synthetic Noisy Construction Speech Generator
# Run on: Google Colab (T4 GPU) or Kaggle (P100 GPU)
# Output: ~16,000 .wav + manifest.jsonl
# ============================================================

# ─────────────────────────────────────────────
# CELL 1 — Mount Drive & Install dependencies
# ─────────────────────────────────────────────
# In Colab:
# from google.colab import drive
# drive.mount('/content/drive')
# OUTPUT_ROOT = "/content/drive/MyDrive/onevoice_audio_v1"

# In Kaggle:
# OUTPUT_ROOT = "/kaggle/working/onevoice_audio_v1"

# !pip install -q TTS soundfile librosa audiomentations numpy pandas tqdm

# ─────────────────────────────────────────────
# CELL 2 — Upload / clone dataset CSV
# ─────────────────────────────────────────────
# Option A: clone from GitHub
# !git clone --depth 1 https://github.com/Platypus27-coder/OneVoice.git
# DATA_DIR = "OneVoice/onevoice-edge/data/onevoice_construction_v2"

# Option B: upload utterances_all.csv manually via Colab Files panel
# DATA_DIR = "/content"

# ─────────────────────────────────────────────
# CELL 3 — Config (edit these before running)
# ─────────────────────────────────────────────
import os, json, random, hashlib
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────
DATA_DIR     = "/content/OneVoice/onevoice-edge/data/onevoice_construction_v2"
OUTPUT_ROOT  = "/content/drive/MyDrive/onevoice_audio_v1"   # change for Kaggle
NOISE_DIR    = os.path.join(OUTPUT_ROOT, "noise_bank")      # pre-downloaded noise files
CLEAN_DIR    = os.path.join(OUTPUT_ROOT, "clean")
NOISY_DIR    = os.path.join(OUTPUT_ROOT, "noisy")
MANIFEST     = os.path.join(OUTPUT_ROOT, "manifest.jsonl")

# ── Sampling config ────────────────────────────────────────
SAMPLES_PER_TEXT   = 2          # each utterance → 2 noisy versions ≈ 16,000 total
SAMPLE_RATE        = 16000
MAX_UTTERANCES     = None       # None = all 8,064; set int to limit for quick test

# ── Speaker pool (Coqui TTS model IDs) ────────────────────
# Each entry: (tts_model, speaker_name_or_None)
# Requires: pip install TTS
VI_SPEAKERS = [
    ("tts_models/vi/vivos/vits",         None),           # Vietnamese VITS
]
EN_SPEAKERS = [
    ("tts_models/en/vctk/vits",          "p225"),         # EN female
    ("tts_models/en/vctk/vits",          "p226"),         # EN male
    ("tts_models/en/ljspeech/tacotron2-DDC", None),       # EN neutral
]

# ── Noise classes (filenames in NOISE_DIR/*.wav) ───────────
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

# ── SNR options (dB) ───────────────────────────────────────
SNR_OPTIONS = [0, 5, 10, 15, 20]

# ─────────────────────────────────────────────
# CELL 4 — Download construction noise bank
# (freesound.org / ESC-50 / DEMAND — verify licences)
# ─────────────────────────────────────────────
NOISE_URLS = {
    # Replace with actual URLs from freesound or your Drive
    "excavator.wav":    "https://your-bucket/excavator.wav",
    "angle_grinder.wav":"https://your-bucket/angle_grinder.wav",
    "drilling.wav":     "https://your-bucket/drilling.wav",
    "hammer.wav":       "https://your-bucket/hammer.wav",
    "diesel_engine.wav":"https://your-bucket/diesel_engine.wav",
    "generator.wav":    "https://your-bucket/generator.wav",
    "truck.wav":        "https://your-bucket/truck.wav",
    "wind.wav":         "https://your-bucket/wind.wav",
    "worker_babble.wav":"https://your-bucket/worker_babble.wav",
}

def download_noise_bank():
    os.makedirs(NOISE_DIR, exist_ok=True)
    for fname, url in NOISE_URLS.items():
        dst = os.path.join(NOISE_DIR, fname)
        if not os.path.exists(dst):
            print(f"Downloading {fname}...")
            os.system(f'wget -q -O "{dst}" "{url}"')
        else:
            print(f"  {fname} already exists, skipping.")

# ─────────────────────────────────────────────
# CELL 5 — Core pipeline functions
# ─────────────────────────────────────────────
import numpy as np
import soundfile as sf
import librosa

def tts_synthesize(text: str, lang: str, out_path: str,
                   tts_model: str, speaker: str = None) -> bool:
    """
    Synthesize text → clean wav using Coqui TTS.
    Returns True on success.
    """
    try:
        from TTS.api import TTS
        tts = TTS(model_name=tts_model, progress_bar=False, gpu=True)
        if speaker:
            tts.tts_to_file(text=text, speaker=speaker, file_path=out_path)
        else:
            tts.tts_to_file(text=text, file_path=out_path)
        return True
    except Exception as e:
        print(f"  [TTS ERROR] {e}")
        return False


def apply_rir(speech: np.ndarray, sr: int,
              rir_wav_path: str = None) -> np.ndarray:
    """
    Convolve speech with a Room Impulse Response (RIR).
    If no RIR file, simulate simple reverb via librosa.
    """
    if rir_wav_path and os.path.exists(rir_wav_path):
        rir, _ = librosa.load(rir_wav_path, sr=sr, mono=True)
        convolved = np.convolve(speech, rir, mode="full")[:len(speech)]
        return convolved / (np.max(np.abs(convolved)) + 1e-9)
    else:
        # Lightweight: add small reverb via echo
        delay = int(sr * 0.05)
        reverb = np.zeros_like(speech)
        reverb[delay:] = speech[:-delay] * 0.3
        return (speech + reverb) / 1.15


def mix_noise(speech: np.ndarray, noise: np.ndarray,
              snr_db: float, sr: int) -> np.ndarray:
    """Mix speech + noise at target SNR (dB)."""
    # Trim / loop noise to match speech length
    if len(noise) < len(speech):
        repeats = int(np.ceil(len(speech) / len(noise)))
        noise   = np.tile(noise, repeats)
    start = random.randint(0, max(0, len(noise) - len(speech)))
    noise = noise[start: start + len(speech)]

    # RMS-based SNR
    rms_speech = np.sqrt(np.mean(speech ** 2) + 1e-9)
    rms_noise  = np.sqrt(np.mean(noise  ** 2) + 1e-9)
    scale = rms_speech / (rms_noise * (10 ** (snr_db / 20)))
    mixed = speech + scale * noise
    return np.clip(mixed / (np.max(np.abs(mixed)) + 1e-9), -1.0, 1.0)


def gain_variation(audio: np.ndarray, min_db=-3.0, max_db=3.0) -> np.ndarray:
    gain = random.uniform(min_db, max_db)
    return audio * (10 ** (gain / 20))


def random_clip(audio: np.ndarray, p=0.05) -> np.ndarray:
    """Very light optional clipping to simulate mic saturation."""
    if random.random() < p:
        threshold = random.uniform(0.7, 0.95)
        return np.clip(audio, -threshold, threshold)
    return audio

# ─────────────────────────────────────────────
# CELL 6 — Main generation loop
# ─────────────────────────────────────────────
import pandas as pd
from tqdm import tqdm

def generate_dataset():
    # ── Setup dirs ──
    for d in [CLEAN_DIR, NOISY_DIR, OUTPUT_ROOT]:
        os.makedirs(d, exist_ok=True)

    # ── Load utterances ──
    utt_path = os.path.join(DATA_DIR, "utterances_all.csv")
    df = pd.read_csv(utt_path)
    if MAX_UTTERANCES:
        df = df.head(MAX_UTTERANCES)
    print(f"Loaded {len(df)} utterances.")

    # ── Load noise files ──
    noise_cache = {}
    for nc in NOISE_CLASSES:
        path = os.path.join(NOISE_DIR, nc)
        if os.path.exists(path):
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            noise_cache[nc] = audio
        else:
            print(f"  [WARN] Noise file missing: {nc}")

    manifest_lines = []
    sample_idx = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        utt_id   = row["utterance_id"]
        vi_text  = str(row["vi"])
        en_text  = str(row.get("en", ""))
        domain   = str(row.get("domain", "unknown"))
        intent   = str(row.get("intent", "unknown"))
        risk     = str(row.get("risk_level", "unknown"))
        split    = str(row.get("split", "train"))

        # ── Pick language (VI or EN depending on pipeline direction) ──
        lang = "vi"
        text = vi_text
        speaker_pool = VI_SPEAKERS

        # ── Synthesize clean wav ──
        clean_fname = f"{utt_id}_clean.wav"
        clean_path  = os.path.join(CLEAN_DIR, clean_fname)

        if not os.path.exists(clean_path):
            tts_model, speaker = random.choice(speaker_pool)
            ok = tts_synthesize(text, lang, clean_path, tts_model, speaker)
            if not ok:
                continue
        else:
            tts_model, speaker = random.choice(speaker_pool)

        clean_audio, _ = librosa.load(clean_path, sr=SAMPLE_RATE, mono=True)

        # ── Apply RIR ──
        rir_applied = apply_rir(clean_audio, SAMPLE_RATE)

        # ── Generate SAMPLES_PER_TEXT noisy variants ──
        for v in range(SAMPLES_PER_TEXT):
            # Random choices
            noise_name = random.choice([n for n in NOISE_CLASSES if n in noise_cache])
            snr_db     = random.choice(SNR_OPTIONS)
            use_reverb = random.random() > 0.4

            speech_for_mix = rir_applied if use_reverb else clean_audio
            mixed = mix_noise(speech_for_mix, noise_cache[noise_name],
                              snr_db, SAMPLE_RATE)
            mixed = gain_variation(mixed)
            mixed = random_clip(mixed)

            noisy_fname = f"{utt_id}_n{v+1:02d}.wav"
            noisy_path  = os.path.join(NOISY_DIR, noisy_fname)
            sf.write(noisy_path, mixed, SAMPLE_RATE)

            # ── Write manifest entry ──
            entry = {
                "audio":              noisy_fname,
                "clean_audio":        clean_fname,
                "text":               text,
                "translation":        en_text,
                "domain":             domain,
                "intent":             intent,
                "risk_level":         risk,
                "split":              split,
                "speaker_id":         f"{tts_model}_{speaker or 'default'}",
                "noise_type":         noise_name.replace(".wav", ""),
                "snr_db":             snr_db,
                "reverb":             use_reverb,
                "rir_id":             "simulated_echo" if use_reverb else "none",
                "synthetic_speech":   True,
                "synthetic_noise_mix":True,
                "sample_rate":        SAMPLE_RATE,
            }
            manifest_lines.append(json.dumps(entry, ensure_ascii=False))
            sample_idx += 1

    # ── Write manifest ──
    with open(MANIFEST, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines))

    print(f"\n✅ Done! Generated {sample_idx} noisy samples.")
    print(f"   Clean wavs : {CLEAN_DIR}")
    print(f"   Noisy wavs : {NOISY_DIR}")
    print(f"   Manifest   : {MANIFEST}")


# ─────────────────────────────────────────────
# CELL 7 — Stats verification
# ─────────────────────────────────────────────
def verify_stats():
    entries = []
    with open(MANIFEST) as f:
        for line in f:
            entries.append(json.loads(line))

    df_m = pd.DataFrame(entries)
    print("=== Dataset Stats ===")
    print(f"Total samples   : {len(df_m)}")
    print(f"\nBy domain:\n{df_m['domain'].value_counts()}")
    print(f"\nBy intent:\n{df_m['intent'].value_counts()}")
    print(f"\nBy noise_type:\n{df_m['noise_type'].value_counts()}")
    print(f"\nBy snr_db:\n{df_m['snr_db'].value_counts().sort_index()}")

# ─────────────────────────────────────────────
# CELL 8 — ENTRY POINT (run in Colab)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Step 0: download noise bank (fill NOISE_URLS above first)
    # download_noise_bank()

    # Step 1: generate
    generate_dataset()

    # Step 2: verify
    verify_stats()
