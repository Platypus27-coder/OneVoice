# 🎙️ OneVoice Edge — Real-time Industrial Speech Translation

> **OneVoice AI Challenge 2026 — Team Impact**

A real-time, **100% offline** Speech-to-Speech translation system designed for
industrial environments (factories, construction sites). Built to run on
**Qualcomm Snapdragon** NPU with end-to-end latency under **1 second** and
RAM consumption under **200 MB**.

---

## 🏭 Problem Statement

Language barriers between foreign experts and local engineers in industrial
settings cause productivity loss and safety hazards. Existing translation apps
(e.g., Google Translate) fail in these environments because they:
- Require continuous internet connection (cloud-based).
- Cannot handle heavy industrial noise (machinery, engines).

## 🚀 Solution — 4-Stage Edge AI Pipeline

```
Microphone
    │
    ▼
┌─────────────────────────────────┐
│  Trạm 0: Denoise                │  GIPFormer ONNX (INT8)
│  Industrial noise filtering     │  ~10ms
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Trạm 1: ASR (Streaming)        │  Whisper-Tiny QNN + GIPFormer
│  Speech → Text (VI / EN)        │  ~200ms
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Trạm 2: Translation            │  MarianMT (INT8 fine-tuned)
│  Text VI ↔ EN / ZH / KR        │  ~100ms
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Trạm 3: TTS                    │  BetterBox-TTS / Tiny VITS
│  Text → Natural Speech          │  ~200ms
└─────────────────────────────────┘
    │
    ▼
Speaker / Earphone
```

**Total latency target: < 600ms (well within the 1s requirement)**

---

## 🌐 Supported Language Pairs

| Direction | Status |
|-----------|--------|
| Vietnamese 🇻🇳 ↔ English 🇬🇧 | ✅ Core |
| Vietnamese 🇻🇳 ↔ Chinese 🇨🇳 | 🔄 Planned |
| Vietnamese 🇻🇳 ↔ Korean 🇰🇷 | 🔄 Planned |

---

## 🔧 Hardware Target

| Spec | Requirement |
|------|-------------|
| Chip | Qualcomm Snapdragon (NPU) |
| RAM  | < 200 MB |
| Latency | < 1 second end-to-end |
| Network | **None — 100% Offline** |

---

## 📁 Project Structure

```
onevoice-edge/
├── src/
│   ├── audio/
│   │   ├── capture.py        # Microphone capture + Silero VAD
│   │   └── denoise.py        # Trạm 0: GIPFormer ONNX denoising
│   ├── asr/
│   │   └── whisper_asr.py    # Trạm 1: Whisper-Tiny streaming ASR
│   ├── translation/
│   │   └── mt_engine.py      # Trạm 2: MarianMT bilingual translation
│   ├── tts/
│   │   └── tts_engine.py     # Trạm 3: BetterBox TTS / Tiny VITS
│   └── pipeline.py           # Main orchestrator (Queue-based threading)
├── notebooks/
│   ├── finetune_marian.ipynb # [Colab] Fine-tune MT with industrial terms
│   └── export_qai.ipynb      # [Colab] Quantize & compile for Snapdragon
├── data/
│   ├── industrial_terms.csv  # VI↔EN technical terminology dictionary
│   └── calibration/          # PTQ calibration audio/text samples
├── scripts/
│   └── export_qai.py         # Qualcomm AI Hub export script
├── config/
│   └── config.yaml           # Pipeline configuration
├── LICENSE                   # CC BY-NC 4.0 (see attributions)
└── requirements.txt
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-team/onevoice-edge.git
cd onevoice-edge
conda create -n onevoice python=3.11.8
conda activate onevoice
pip install -r requirements.txt
```

---

## 🚀 Quick Start

```bash
# Run the full pipeline (microphone → speaker)
python src/pipeline.py

# Test individual modules
python src/translation/mt_engine.py
python src/asr/whisper_asr.py
```

---

## 🧪 Fine-tuning on Colab (for industrial terminology)

Open `notebooks/finetune_marian.ipynb` on Google Colab.
This notebook will fine-tune MarianMT using the technical dictionary
in `data/industrial_terms.csv`.

---

## 📦 Model Export for Snapdragon (Qualcomm AI Hub)

Open `notebooks/export_qai.ipynb` on Google Colab.
Requires a Qualcomm AI Hub API key (set as `QAI_HUB_API_TOKEN`).

---

## ⚠️ License & Attribution

This project is licensed under **CC BY-NC 4.0** (Non-Commercial).

It builds upon the following open-source works — please see [LICENSE](./LICENSE)
for full attribution details:

| Component | Author | License |
|-----------|--------|---------|
| [BetterBox-TTS](https://github.com/nowtranminh1-TTS/BetterBox-TTS) | Dolly VN / ContextBoxAI | CC BY-NC 4.0 |
| [gipformer](https://huggingface.co/g-group-ai-lab/gipformer-65M-rnnt) | G-Group AI Lab | MIT |
| [VALL-E X](https://github.com/Plachtaa/VALL-E-X) | Plachtaa / Songting | MIT |
| [MarianMT (Helsinki-NLP)](https://huggingface.co/Helsinki-NLP) | University of Helsinki | Apache 2.0 |
| [Whisper](https://github.com/openai/whisper) | OpenAI | MIT |
| [Silero VAD](https://github.com/snakers4/silero-vad) | Silero Team | MIT |

> ❌ **Commercial use is strictly prohibited** due to the CC BY-NC 4.0 license
> of BetterBox-TTS components used in this project.

---

## 👥 Team Impact
*OneVoice AI Challenge 2026*
