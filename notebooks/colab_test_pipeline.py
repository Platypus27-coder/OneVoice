# ============================================================
# OneVoice Edge — Full Pipeline Test (Google Colab / Kaggle)
# ============================================================
#
# Hướng dẫn:
#   1. Upload toàn bộ folder onevoice-edge lên Colab,
#      hoặc push lên GitHub rồi clone về bằng URL của team.
#   2. Runtime → Change runtime type → T4 GPU
#   3. Chạy từng cell theo thứ tự
#
# KHÔNG clone repo bên thứ ba — tất cả code đã được
# tích hợp sẵn trong src/ của project này.
# ============================================================

# %% [Cell 1] Setup & GPU check
import os, sys, time
import torch

print("="*55)
print("  OneVoice Edge — Colab Test")
print("="*55)
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("⚠️  No GPU detected — OmniVoice TTS will be slow")

# Thêm src/ vào sys.path để import trực tiếp
PROJECT_ROOT = os.path.abspath(".")
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
print(f"\nProject root   : {PROJECT_ROOT}")

# %% [Cell 2] Install dependencies
os.system("pip install -q sherpa-onnx openai-whisper transformers torch torchaudio")
os.system("pip install -q soundfile sounddevice pyttsx3 PyYAML huggingface_hub")
os.system("pip install -q pedalboard pydub librosa sentencepiece sacremoses")
print("✅ Core dependencies installed")

# %% [Cell 3] Download model weights (lần đầu ~10 phút, sau đó cache)
# Weights được download từ HuggingFace — không clone repo người khác
from scripts.download_models import download_gipformer, download_whisper, download_marianmt

download_gipformer()   # 65MB — GIPFormer INT8 ONNX (Vietnamese ASR)
download_whisper()     # 150MB — Whisper-Tiny (English ASR)
download_marianmt()    # 600MB — MarianMT VI↔EN

# %% [Cell 4] Download OmniVoice model weights (Vietnamese TTS — cần GPU!)
# Chỉ download weights — code TTS đã nằm trong src/tts/ của project
from huggingface_hub import snapshot_download

os.makedirs("models/omnivoice", exist_ok=True)

print("Downloading OmniVoice model weights...")
print("(Lần đầu ~3-7GB, mất 10-15 phút)")

# Dùng bản fine-tune tiếng Việt nếu có, fallback về bản gốc
try:
    snapshot_download(
        repo_id="splendor1811/omnivoice-vietnamese",
        local_dir="models/omnivoice",
    )
    OMNI_MODEL_PATH = "models/omnivoice"
    print("✅ OmniVoice Vietnamese fine-tune downloaded")
except Exception:
    snapshot_download(
        repo_id="k2-fsa/OmniVoice",
        local_dir="models/omnivoice",
    )
    OMNI_MODEL_PATH = "models/omnivoice"
    print("✅ OmniVoice original downloaded")

# %% [Cell 5] Test GIPFormer — ASR tiếng Việt
import sherpa_onnx
import soundfile as sf
import numpy as np

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

def asr_vi(audio: np.ndarray, sr: int = 16000) -> tuple:
    t0 = time.perf_counter()
    stream = recognizer.create_stream()
    stream.accept_waveform(sr, audio.astype(np.float32))
    recognizer.decode_streams([stream])
    text = stream.result.text.strip()
    return text, (time.perf_counter() - t0) * 1000

# Tạo audio mẫu test (hoặc upload file .wav của bạn lên Colab)
dummy_audio = np.random.randn(16000).astype(np.float32) * 0.01
text_vi, ms = asr_vi(dummy_audio)
print(f"[GIPFormer] ✅ Ready | Test: \"{text_vi}\" ({ms:.0f}ms)")
print("💡 Upload file .wav tiếng Việt thật để test chính xác hơn")

# %% [Cell 6] Test Whisper-Tiny — ASR tiếng Anh
import whisper

whisper_model = whisper.load_model("tiny")

def asr_en(audio: np.ndarray) -> tuple:
    t0 = time.perf_counter()
    audio_f = audio.astype(np.float32)
    if audio_f.max() > 1.0:
        audio_f /= 32768.0
    result = whisper_model.transcribe(audio_f, language="en", fp16=torch.cuda.is_available())
    return result["text"].strip(), (time.perf_counter() - t0) * 1000

text_en, ms = asr_en(dummy_audio)
print(f"[Whisper-Tiny] ✅ Ready | Test: \"{text_en}\" ({ms:.0f}ms)")

# %% [Cell 7] Test MarianMT — Translation VI↔EN
from transformers import MarianMTModel, MarianTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tok_vi_en = MarianTokenizer.from_pretrained("models/marianmt/vi2en")
mdl_vi_en = MarianMTModel.from_pretrained("models/marianmt/vi2en").to(DEVICE)

tok_en_vi = MarianTokenizer.from_pretrained("models/marianmt/en2vi")
mdl_en_vi = MarianMTModel.from_pretrained("models/marianmt/en2vi").to(DEVICE)

def translate(text: str, direction: str = "vi2en") -> tuple:
    tok = tok_vi_en if direction == "vi2en" else tok_en_vi
    mdl = mdl_vi_en if direction == "vi2en" else mdl_en_vi
    t0 = time.perf_counter()
    inputs = tok(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_length=128)
    result = tok.decode(out[0], skip_special_tokens=True)
    return result, (time.perf_counter() - t0) * 1000

print("\n── MarianMT Translation Tests ──")
tests = [
    ("vi2en", "Máy xúc số 3 đang bị lỗi thủy lực."),
    ("vi2en", "Van an toàn trên đường ống số 5 bị rò rỉ."),
    ("en2vi", "The hydraulic jack on excavator 3 has failed."),
    ("en2vi", "Safety valve on pipeline 5 is leaking immediately."),
]
for direction, text in tests:
    result, ms = translate(text, direction)
    arrow = "VI→EN" if direction == "vi2en" else "EN→VI"
    print(f"  [{arrow} {ms:.0f}ms] \"{text}\"")
    print(f"                → \"{result}\"\n")

# %% [Cell 8] Test OmniVoice TTS — tiếng Việt (cần GPU)
# OmniVoice model class được import từ HuggingFace hub trực tiếp
# Không cần clone BetterBox-TTS repo
from omnivoice.models.omnivoice import OmniVoice
from src.utils.audio_tools import segment_text, fix_silent_audio, apply_pitch_shift
from src.utils.vad import vad_trim

model_omni = OmniVoice.from_pretrained(OMNI_MODEL_PATH, dtype=torch.float32)
model_omni = model_omni.to(DEVICE)
SR_OMNI = model_omni.sampling_rate
print(f"✅ OmniVoice loaded | device={DEVICE} | sr={SR_OMNI}")

def tts_vi(text: str, reference_audio: str = None, speed: float = 1.0) -> np.ndarray:
    """Synthesize Vietnamese speech using OmniVoice."""
    t0 = time.perf_counter()
    # Import inference function từ omnivoice hub package
    from omnivoice.inference import generate as omni_generate
    result = omni_generate(
        model=model_omni,
        text=text,
        reference_audio=reference_audio,
        language="vi",
        speed=speed,
    )
    audio = result[0] if isinstance(result, (list, tuple)) else result
    audio = vad_trim(audio, SR_OMNI, margin_s=0.05)
    audio = fix_silent_audio(audio, SR_OMNI)
    ms = (time.perf_counter() - t0) * 1000
    print(f"[OmniVoice] ✅ {ms:.0f}ms | \"{text[:50]}\"")
    return audio.astype(np.float32)

import IPython.display as ipd

vi_tests = [
    "Máy xúc số ba đang bị lỗi thủy lực, cần kiểm tra ngay.",
    "Van an toàn trên đường ống số năm bị rò rỉ.",
    "Kỹ thuật viên hãy kiểm tra cầu dao số mười hai.",
]
for text in vi_tests:
    audio = tts_vi(text)
    out_path = f"tts_{hash(text)%10000}.wav"
    sf.write(out_path, audio, SR_OMNI)
    print(f"  Saved: {out_path}")
    ipd.display(ipd.Audio(out_path))

# %% [Cell 9] Full E2E Pipeline Test — VI→EN
print("\n" + "="*55)
print("  E2E Test: VI Audio → VI Text → EN Text → EN Audio")
print("="*55)

# Upload file .wav tiếng Việt lên Colab, hoặc dùng dummy
# from google.colab import files
# uploaded = files.upload()  # Chọn file .wav tiếng Việt

# Giả lập với dummy audio (thay bằng file thật khi có)
sample_audio = np.random.randn(16000 * 3).astype(np.float32) * 0.01

t_start = time.perf_counter()

# Trạm 1: ASR
vi_text, asr_ms = asr_vi(sample_audio)
print(f"  [ASR  {asr_ms:.0f}ms] VI: \"{vi_text}\"")

# Trạm 2: MT
en_text, mt_ms = translate(vi_text, "vi2en")
print(f"  [MT   {mt_ms:.0f}ms] EN: \"{en_text}\"")

# Trạm 3: TTS (pyttsx3 cho EN — offline, nhẹ)
import pyttsx3, tempfile
engine = pyttsx3.init()
engine.setProperty("rate", 160)
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    en_audio_path = f.name
engine.save_to_file(en_text, en_audio_path)
engine.runAndWait()
tts_ms = (time.perf_counter() - t_start)*1000 - asr_ms - mt_ms

total_ms = (time.perf_counter() - t_start) * 1000
status = "✅" if total_ms < 1000 else "⚠️"
print(f"\n  {status} Total E2E: {total_ms:.0f}ms (ASR:{asr_ms:.0f} MT:{mt_ms:.0f} TTS:{tts_ms:.0f})")
ipd.display(ipd.Audio(en_audio_path))

# %% [Cell 10] Benchmark & Summary
print("\n" + "="*55)
print("  BENCHMARK SUMMARY")
print("="*55)

bench_sentences = [
    ("vi2en", "Máy xúc số 3 đang bị lỗi thủy lực."),
    ("vi2en", "Cần dừng máy và kiểm tra van an toàn ngay."),
    ("vi2en", "Áp suất đường ống vượt mức, cẩu tháp dừng khẩn cấp."),
    ("en2vi", "The hydraulic system of excavator 3 has failed."),
    ("en2vi", "Stop the machine and check the safety valve immediately."),
]

mt_times = []
for direction, text in bench_sentences:
    _, ms = translate(text, direction)
    mt_times.append(ms)
    arrow = "VI→EN" if direction == "vi2en" else "EN→VI"
    print(f"  [{arrow} {ms:.0f}ms] {text[:45]}")

import statistics
print(f"\n  MT avg latency : {statistics.mean(mt_times):.0f}ms")
print(f"  MT max latency : {max(mt_times):.0f}ms")
print(f"  GPU device     : {DEVICE}")
passed = sum(1 for t in mt_times if t < 200)
print(f"  Target <200ms  : {passed}/{len(mt_times)} passed")

# %% [Cell 11] Save & download models
# Sau khi verify xong, zip và download về local
print("\nModels available for download:")
print("  models/marianmt/vi2en/")
print("  models/marianmt/en2vi/")
print("  models/gipformer/")
print()
print("# Chạy cell này để download:")
print("from google.colab import files")
print("import shutil")
print("shutil.make_archive('models_onevoice', 'zip', 'models')")
print("files.download('models_onevoice.zip')")
