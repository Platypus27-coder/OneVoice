# ============================================================
# OneVoice Edge — Colab Full Pipeline Test (VI ↔ EN)
# ============================================================
# Hướng dẫn dùng:
#   1. Upload file này lên Google Colab
#   2. Runtime → Change runtime type → T4 GPU
#   3. Chạy từng cell theo thứ tự
# ============================================================

# %% [Cell 1] GPU Check
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB" if torch.cuda.is_available() else "")

# %% [Cell 2] Clone repos
import os

# Clone project repo (thay bằng URL GitHub thật của bạn)
if not os.path.exists("onevoice-edge"):
    os.system("git clone https://github.com/your-team/onevoice-edge.git")

# Clone BetterBox-TTS (TTS tiếng Việt)
if not os.path.exists("BetterBox-TTS"):
    os.system("git clone https://github.com/nowtranminh1-TTS/BetterBox-TTS.git")

os.chdir("onevoice-edge")
print("Working dir:", os.getcwd())

# %% [Cell 3] Install dependencies
os.system("pip install -q sherpa-onnx openai-whisper transformers torch torchaudio")
os.system("pip install -q soundfile sounddevice pyttsx3 PyYAML huggingface_hub")
os.system("pip install -q pedalboard pydub librosa sentencepiece sacremoses")
os.system("pip install -q -r ../BetterBox-TTS/general/requirements.txt")
print("✅ Dependencies installed")

# %% [Cell 4] Download GIPFormer (Vietnamese ASR)
from huggingface_hub import hf_hub_download
import os

REPO = "g-group-ai-lab/gipformer-65M-rnnt"
os.makedirs("models/gipformer", exist_ok=True)
files = [
    "encoder-epoch-35-avg-6.int8.onnx",
    "decoder-epoch-35-avg-6.int8.onnx",
    "joiner-epoch-35-avg-6.int8.onnx",
    "tokens.txt",
]
for f in files:
    path = hf_hub_download(repo_id=REPO, filename=f, local_dir="models/gipformer")
    print(f"  ✅ {f}")

# %% [Cell 5] Download Whisper-Tiny (English ASR)
import whisper
model_whisper = whisper.load_model("tiny")
print("✅ Whisper-Tiny loaded")

# %% [Cell 6] Download MarianMT (VI↔EN Translation)
from transformers import MarianMTModel, MarianTokenizer

os.makedirs("models/marianmt/vi2en", exist_ok=True)
os.makedirs("models/marianmt/en2vi", exist_ok=True)

print("Downloading VI→EN...")
tok_vi_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-vi-en")
mdl_vi_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-vi-en")
tok_vi_en.save_pretrained("models/marianmt/vi2en")
mdl_vi_en.save_pretrained("models/marianmt/vi2en")

print("Downloading EN→VI...")
tok_en_vi = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-vi")
mdl_en_vi = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-vi")
tok_en_vi.save_pretrained("models/marianmt/en2vi")
mdl_en_vi.save_pretrained("models/marianmt/en2vi")
print("✅ MarianMT VI↔EN saved")

# %% [Cell 7] Download & Load OmniVoice (Vietnamese TTS — cần GPU!)
import sys
sys.path.insert(0, "../BetterBox-TTS")
sys.path.insert(0, "../BetterBox-TTS/OmniVoice")

from OmniVoice.omnivoice_inference.ttsOmni import Omni, generate_speech_omni

# Dùng bản gốc k2-fsa/OmniVoice nếu chưa có fine-tune
omni = Omni(model_path="k2-fsa/OmniVoice")
omni.loadModelOmni()
print(f"✅ OmniVoice loaded | device={omni.device} | sr={omni.sampling_rate}")

# %% [Cell 8] Test ASR — GIPFormer (Vietnamese)
import sherpa_onnx
import soundfile as sf
import numpy as np
import time

recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder="models/gipformer/encoder-epoch-35-avg-6.int8.onnx",
    decoder="models/gipformer/decoder-epoch-35-avg-6.int8.onnx",
    joiner="models/gipformer/joiner-epoch-35-avg-6.int8.onnx",
    tokens="models/gipformer/tokens.txt",
    num_threads=2,
    sample_rate=16000,
    feature_dim=80,
    decoding_method="greedy_search",
)

# Dùng audio mẫu từ gipformer repo
test_audio_url = "https://huggingface.co/g-group-ai-lab/gipformer-65M-rnnt/resolve/main/test.wav"
os.system(f"wget -q -O test_vi.wav {test_audio_url} 2>/dev/null || echo 'No sample, using silence'")

if os.path.exists("test_vi.wav"):
    audio, sr = sf.read("test_vi.wav", dtype="float32")
    t0 = time.perf_counter()
    stream = recognizer.create_stream()
    stream.accept_waveform(sr, audio)
    recognizer.decode_streams([stream])
    text = stream.result.text.strip()
    print(f"✅ GIPFormer ASR: \"{text}\" ({(time.perf_counter()-t0)*1000:.0f}ms)")

# %% [Cell 9] Test Translation — MarianMT VI→EN
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
mdl_vi_en = mdl_vi_en.to(device)

def translate(text, tokenizer, model):
    t0 = time.perf_counter()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=128)
    result = tokenizer.decode(out[0], skip_special_tokens=True)
    ms = (time.perf_counter() - t0) * 1000
    return result, ms

tests_vi = [
    "Máy xúc số 3 đang bị lỗi thủy lực.",
    "Van an toàn trên đường ống số 5 bị rò rỉ.",
    "Kỹ thuật viên hãy kiểm tra cầu dao số 12.",
    "Cẩu tháp khu A gặp sự cố, dừng hoạt động ngay.",
]
print("\n── VI→EN Translation ──")
for vi in tests_vi:
    en, ms = translate(vi, tok_vi_en, mdl_vi_en)
    print(f"  [{ms:.0f}ms] \"{vi}\"")
    print(f"         → \"{en}\"\n")

# %% [Cell 10] Test Translation — MarianMT EN→VI
mdl_en_vi = mdl_en_vi.to(device)

tests_en = [
    "The hydraulic jack on excavator 3 has failed.",
    "Safety valve on pipeline 5 is leaking.",
    "Technician please check circuit breaker 12.",
    "Tower crane in zone A has malfunctioned.",
]
print("── EN→VI Translation ──")
for en in tests_en:
    vi, ms = translate(en, tok_en_vi, mdl_en_vi)
    print(f"  [{ms:.0f}ms] \"{en}\"")
    print(f"         → \"{vi}\"\n")

# %% [Cell 11] Test TTS — OmniVoice (Vietnamese output)
import IPython.display as ipd
import soundfile as sf

tts_tests = [
    "Máy xúc số ba đang bị lỗi thủy lực, cần kiểm tra ngay.",
    "Van an toàn trên đường ống số năm bị rò rỉ.",
]

# Tạo reference audio mẫu (nếu không có file thật)
# Bạn có thể upload file .wav bất kỳ lên Colab làm reference
ref_audio_path = None  # None = dùng default của OmniVoice

for i, text in enumerate(tts_tests):
    print(f"\nSynthesizing [{i+1}]: \"{text}\"")
    t0 = time.perf_counter()
    result, status, srt_path = generate_speech_omni(
        omni=omni,
        text=text,
        language="vi",
        reference_audio=ref_audio_path,
        speed=1.0,
    )
    ms = (time.perf_counter() - t0) * 1000
    print(f"  {status} | {ms:.0f}ms")

    if result is not None:
        sr_out, audio_out = result
        out_file = f"tts_output_{i+1}.wav"
        sf.write(out_file, audio_out, sr_out)
        print(f"  Saved: {out_file}")
        ipd.display(ipd.Audio(out_file))  # Phát thẳng trong Colab

# %% [Cell 12] Full E2E Pipeline Test (VI→EN với audio file)
print("\n" + "="*60)
print("  E2E Pipeline: Audio VI → Text VI → Text EN → Audio EN")
print("="*60)

import pyttsx3, tempfile

def e2e_vi_to_en(audio_path: str):
    """Full pipeline: VI audio → EN speech."""
    timings = {}

    # Trạm 1: ASR (GIPFormer — Vietnamese)
    audio, sr = sf.read(audio_path, dtype="float32")
    t0 = time.perf_counter()
    stream = recognizer.create_stream()
    stream.accept_waveform(sr, audio)
    recognizer.decode_streams([stream])
    vi_text = stream.result.text.strip()
    timings["asr_ms"] = (time.perf_counter() - t0) * 1000

    # Trạm 2: MT (MarianMT VI→EN)
    en_text, timings["mt_ms"] = translate(vi_text, tok_vi_en, mdl_vi_en)

    # Trạm 3: TTS (pyttsx3 cho EN — nhẹ, phù hợp test)
    t0 = time.perf_counter()
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    engine.save_to_file(en_text, tmp_path)
    engine.runAndWait()
    timings["tts_ms"] = (time.perf_counter() - t0) * 1000

    total = sum(timings.values())
    status = "✅" if total < 1000 else "⚠️"

    print(f"\n  VI (ASR):  \"{vi_text}\" [{timings['asr_ms']:.0f}ms]")
    print(f"  EN (MT) :  \"{en_text}\" [{timings['mt_ms']:.0f}ms]")
    print(f"  TTS     :  {tmp_path} [{timings['tts_ms']:.0f}ms]")
    print(f"  {status} Total E2E: {total:.0f}ms")
    return tmp_path, timings

if os.path.exists("test_vi.wav"):
    out, timings = e2e_vi_to_en("test_vi.wav")
    ipd.display(ipd.Audio(out))

# %% [Cell 13] Benchmark Summary
print("\n" + "="*60)
print("  BENCHMARK — MarianMT Latency on GPU")
print("="*60)

import statistics

latencies_vi_en = []
latencies_en_vi = []

for vi in tests_vi:
    _, ms = translate(vi, tok_vi_en, mdl_vi_en)
    latencies_vi_en.append(ms)

for en in tests_en:
    _, ms = translate(en, tok_en_vi, mdl_en_vi)
    latencies_en_vi.append(ms)

print(f"\n  VI→EN avg: {statistics.mean(latencies_vi_en):.0f}ms | max: {max(latencies_vi_en):.0f}ms")
print(f"  EN→VI avg: {statistics.mean(latencies_en_vi):.0f}ms | max: {max(latencies_en_vi):.0f}ms")
print(f"\n  Target < 100ms for MT stage: {'✅' if max(latencies_vi_en + latencies_en_vi) < 100 else '⚠️'}")

# %% [Cell 14] Save fine-tuned/tested model checkpoints
# Sau khi verify chất lượng OK, lưu model để dùng tiếp
print("\nModel paths ready to download:")
print("  models/marianmt/vi2en/  ← Copy về local")
print("  models/marianmt/en2vi/  ← Copy về local")
print("  models/gipformer/       ← Copy về local")
print("\nDownload bằng lệnh:")
print("  from google.colab import files")
print("  import shutil")
print("  shutil.make_archive('marianmt_vi_en', 'zip', 'models/marianmt/vi2en')")
print("  files.download('marianmt_vi_en.zip')")
