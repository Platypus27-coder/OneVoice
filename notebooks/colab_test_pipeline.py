# ============================================================
# OneVoice Edge — Full Pipeline Test (Google Colab / Kaggle)
# ============================================================
#
# Hướng dẫn:
#   1. Upload toàn bộ folder onevoice-edge lên Colab,
#      hoặc push lên GitHub rồi clone về bằng URL của repo.
#   2. Runtime → Change runtime type → T4 GPU
#   3. Chạy từng cell theo thứ tự
#
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
    print("⚠️  No GPU detected — Pipeline will be very slow")

# Thêm src/ vào sys.path để import trực tiếp
PROJECT_ROOT = os.path.abspath(".")
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
TTS_PATH = os.path.join(SRC_PATH, "tts")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if TTS_PATH not in sys.path:
    sys.path.insert(0, TTS_PATH)
print(f"\nProject root   : {PROJECT_ROOT}")

# %% [Cell 2] Install dependencies
print("Installing dependencies...")
os.system("apt-get update && apt-get install -y portaudio19-dev ffmpeg espeak espeak-ng")
os.system("pip install -q git+https://github.com/huggingface/transformers.git")
os.system("pip install -q funasr_onnx modelscope f5-tts torch torchvision torchaudio soundfile sounddevice PyYAML pedalboard pydub deepmultilingualpunctuation librosa sentencepiece sacremoses huggingface_hub accelerate tqdm unidecode cn2an zhconv zhon webdataset")
print("✅ Core dependencies installed")

# %% [Cell 3] Test SenseVoice ASR (Trạm 1)
print("\n── SenseVoice ASR Test ──")
from funasr_onnx import SenseVoiceSmall
import numpy as np

# Load SenseVoice
model_dir = "iic/SenseVoiceSmall"
asr_model = SenseVoiceSmall(model_dir, batch_size=1, quantize=True)
print("[SenseVoice] ✅ Ready")

def asr_infer(audio_path: str) -> str:
    # Có thể upload file wav thật lên Colab để test
    res = asr_model(audio_path, language="auto", use_itn=True)
    return res[0][0]['text']

# Tạo dummy audio file để test
import soundfile as sf
dummy_audio = np.random.randn(16000 * 2).astype(np.float32) * 0.01
sf.write("dummy.wav", dummy_audio, 16000)
text = asr_infer("dummy.wav")
print(f"Test ASR output: {text}")

# %% [Cell 4] Test Envit5 MT (Trạm 2)
print("\n── Envit5 Translation Test ──")
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading VietAI/envit5-translation...")
mt_tokenizer = AutoTokenizer.from_pretrained("VietAI/envit5-translation")
mt_model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/envit5-translation").to(DEVICE)
print("[Envit5] ✅ Ready")

def translate(text: str, direction: str = "vi2en") -> tuple:
    prefix = "vi: " if direction == "vi2en" else "en: "
    input_text = prefix + text
    t0 = time.perf_counter()
    inputs = mt_tokenizer(input_text, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        out = mt_model.generate(**inputs, max_length=128)
    result = mt_tokenizer.decode(out[0], skip_special_tokens=True)
    if result.startswith("en: "): result = result[4:]
    if result.startswith("vi: "): result = result[4:]
    return result.strip(), (time.perf_counter() - t0) * 1000

tests = [
    ("vi2en", "Máy xúc số 3 đang bị lỗi thủy lực."),
    ("en2vi", "The hydraulic system of excavator 3 has failed."),
]
for direction, text in tests:
    res, ms = translate(text, direction)
    print(f"[{direction} {ms:.0f}ms] {text} → {res}")

# %% [Cell 5] Test OmniVoice TTS (Trạm 3 - VI)
print("\n── OmniVoice TTS Test ──")
# Yêu cầu src/tts/omnivoice có sẵn trong repo của bạn
try:
    from omnivoice.models.omnivoice import OmniVoice
    print("Loading OmniVoice...")
    # Tự động tải weights từ HuggingFace nếu chưa có
    omni_model = OmniVoice.from_pretrained("splendor1811/omnivoice-vietnamese", dtype=torch.float32).to(DEVICE)
    print("[OmniVoice] ✅ Ready")
except ImportError:
    print("⚠️  OmniVoice không tìm thấy trong thư mục src/tts/omnivoice. Vui lòng đảm bảo đã clone đủ module.")

# %% [Cell 6] E2E Pipeline (Mô phỏng 1 luồng xử lý)
print("\n" + "="*55)
print("  E2E Test: VI Audio → VI Text → EN Text")
print("="*55)

t_start = time.perf_counter()

# 1. ASR
vi_text = asr_infer("dummy.wav")
asr_ms = (time.perf_counter() - t_start) * 1000
print(f"  [ASR  {asr_ms:.0f}ms] VI: \"{vi_text}\"")

# 2. MT
t0_mt = time.perf_counter()
en_text, mt_ms = translate("Xin chào, tôi cần kiểm tra van an toàn", "vi2en")
print(f"  [MT   {mt_ms:.0f}ms] EN: \"{en_text}\"")

total_ms = (time.perf_counter() - t_start) * 1000
print(f"\n  Total E2E (ASR+MT): {total_ms:.0f}ms")
