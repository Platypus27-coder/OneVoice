# OneVoice Edge V2 — Hệ Thống Phiên Dịch Giọng Nói Thời Gian Thực (Edge AI)

<img width="2352" height="1792" alt="OneVoice Edge Banner" src="https://github.com/user-attachments/assets/f4747894-01d8-4889-bbf5-a0d2a5c01de7" />

OneVoice là hệ thống dịch thuật Speech-to-Speech Việt ↔ Anh dành cho môi trường công nghiệp (nhà máy, công trường). Mục tiêu V2 là vận hành **100% offline** trên thiết bị Edge / chip **Qualcomm Snapdragon**, đạt độ trễ dưới **1 giây** và peak RAM dưới **200 MB**. Các chỉ tiêu này là cổng nghiệm thu đang được đo bằng notebook và báo cáo; dự án chưa tuyên bố production-ready khi chưa vượt dữ liệu thực địa.

---


## 🚧 Trạng Thái OneVoice V2

V2 nâng cấp trực tiếp runtime hiện tại nhưng vẫn giữ tag `v1-working-baseline` để rollback. Trạng thái được ghi theo bằng chứng, không suy diễn từ notebook:

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| Context Engine, Site Pack, Safety Matching | `VERIFIED` | Logic deterministic và unit test |
| Streaming 32 ms, stable prefix, semantic commit | `PARTIAL` | Đã test logic, chưa có p95 model thật |
| Denoising | `FALLBACK` | Passthrough là baseline; DeepFilterNet/RNNoise phải qua quality gate |
| VI-ASR, EN-ASR, MT và TTS | `PARTIAL` | Adapter đã có; benchmark thực chạy trên Colab |
| Offline Edge và RAM < 200 MB | `PLANNED` | Chờ artifact bundle và profile đầy đủ |
| Real-site robustness | `PLANNED` | Chưa có fixed real-site holdout |

Chi tiết bằng chứng: [V1 baseline](docs/V1_BASELINE_STATUS.md), [kế hoạch V1 → V2](ONEVOICE_V1_TO_V2_PLAN.md) và [hướng dẫn notebook V2](notebooks/README_V2.md).

---

## 🎯 Vấn Đề Thực Tế & Giải Pháp Edge AI

Rào cản ngôn ngữ giữa chuyên gia nước ngoài và kỹ sư bản địa gây giảm năng suất và nguy cơ mất an toàn lao động. Các ứng dụng như Google Translate không thể dùng được tại công trường vì:
- Bắt buộc phải có Internet (Cloud-based).
- Không hoạt động được trong môi trường tiếng ồn máy móc lớn (máy xúc, máy cắt, gió công trường).

---

## 🏗️ Kiến Trúc Luồng Xử Lý 4 Trạm Cục Bộ (Production Architecture)

### 🇻🇳 Luồng 1: VI → EN (Tiếng Việt → Tiếng Anh)

```text
Microphone
    │
    ▼
┌─────────────────────────────────┐
│  Trạm 0: Capture & VAD          │  Silero VAD + SoundDevice
│  Tách đoạn thoại thực tế        │  ~10ms
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Trạm 1: Nhận Diện ASR (VI)     │  GIPFormer INT8 ONNX (VietAI)
│  Giọng nói → Văn bản Tiếng Việt │  Noise-Robust Acoustic Modeling
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Trạm 2: Dịch Thuật MT (VI→EN)  │  VietAI/envit5-translation
│  Văn bản VI → EN                │  Tích hợp Terminology & Normalizer
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Trạm 3: Tổng Hợp Âm Thanh (TTS)│  F5-TTS (Voice Clone) / Fallback Engine
│  Văn bản EN → Giọng nói         │  Bảo toàn chất giọng / Đọc tức thì
└─────────────────────────────────┘
    │
    ▼
Speaker / Earphone
```

### 🇬🇧 Luồng 2: EN → VI (Tiếng Anh → Tiếng Việt)

```text
Microphone
    │
    ▼
┌─────────────────────────────────┐
│  Trạm 0: Capture & VAD          │  Silero VAD + SoundDevice
│  Tách đoạn thoại thực tế        │  ~10ms
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Trạm 1: Nhận Diện ASR (EN)     │  SenseVoice Small ONNX / Whisper
│  Giọng nói → Văn bản Tiếng Anh  │  + Trích xuất Cảm Xúc (Emotion)
└─────────────────────────────────┘
    │ metadata: [text, emotion]
    ▼
┌─────────────────────────────────┐
│  Trạm 2: Dịch Thuật MT (EN→VI)  │  VietAI/envit5-translation
│  Văn bản EN → VI                │  Luân chuyển metadata cảm xúc
└─────────────────────────────────┘
    │ metadata: [translated_text, emotion]
    ▼
┌─────────────────────────────────┐
│  Trạm 3: Tổng Hợp Âm Thanh (TTS)│  OmniVoice (Voice Design) / PyTTSx3
│  Văn bản VI → Giọng nói         │  Đọc Tiếng Việt mô phỏng cảm xúc
└─────────────────────────────────┘
    │
    ▼
Speaker / Earphone
```

**⚡ Mục tiêu nghiệm thu V2:** normal commit→first-audio p95 < **1000 ms**, safety p95 < **300 ms**, runtime Edge không truy cập mạng và peak RAM < **200 MB**. Chỉ công nhận khi có report đo thật.

---


### 🧠 Các lớp V2 bổ sung quanh hai luồng trên

```text
AudioFrame 32 ms
    → Denoiser
    → Stateful VAD + rolling utterance
    → Rolling ASR + stable prefix
    → Construction Context Engine
    → Semantic Commit (WAIT / NORMAL / SAFETY)
    → Safety Audio hoặc MT
    → Critical-field Validator
    → Ordered TTS
    → Audio đầu tiên + báo cáo latency
```

Hai sơ đồ VI→EN và EN→VI ở trên mô tả hướng model; chuỗi V2 này mô tả cơ chế streaming, an toàn và đo lường dùng chung cho cả hai hướng.

---

## 🎛️ Các Chế Độ Hoạt Động (Translation Directions)

Hệ thống là một đường ống hai chiều, cho phép chuyển đổi linh hoạt qua cờ lệnh runtime:

| Hướng (Direction) | Đầu vào (Người nói) | Đầu ra (Loa phát) | Lệnh chạy (Flag) |
|---|---|---|---|
| **VI → EN** (Mặc định) | Người Việt | Người Anh | `python src/pipeline.py --direction vi2en` |
| **EN → VI** | Người Anh | Người Việt | `python src/pipeline.py --direction en2vi` |

---

## 🎬 Demo Kết Quả Dịch Thuật & Voice Cloning

Dưới đây là 7 kịch bản demo đã có từ baseline V1, gồm thuật ngữ chuyên ngành, từ lóng thi công và tình huống công trường. Đây là bằng chứng demo lịch sử, không thay thế benchmark V2 trên fixed test set. Các bản MP4 vẫn được lưu trong `demo_outputs/`:

1. **Test 1 (VI→EN)**
   - **Đầu vào**: Cậu đã làm gì với nó vậy thêm năng lượng hả nó hoạt động như thế nào vậy cho mình mượn chút đừng có keo kiệt vậy chứ hôm nay lớp mình có bài kiểm tra môn thể dục nên mình rất là cần nó luôn xài xong mình trả lại liền
   - **Bản dịch**: *What did you do with it? More power, huh? How it works. Well, let me borrow some. don't be mean, because we have a gym test today, so... I really need it. I'll give it back when I'm done.*
<details>
  <summary><h5>🔗 Nghe Audio</h5></summary>

[test_1_output_vi2en.webm](https://github.com/user-attachments/assets/274cef99-a640-4cc5-8ed6-2c7836ec417b)

[▶ Xem/nghe file MP4 trong repository](demo_outputs/test_1_output_vi2en.mp4)

</details>

2. **Test 2 (VI→EN)**
   - **Đầu vào**: Ê bạn ơi cái máy xúc số ba nó bị xì nhớt thủy lực rồi bơm bê tông cũng kẹt luôn qua kiểm tra lẹ giùm mình đi chứ để vậy là cháy van an toàn nha
   - **Bản dịch**: *Hey, buddy, that excavator number three, it's leaking hydraulic fluid. The pump's jammed, too. please check it immediately. the safety valve will blow out.*
<details>
  <summary><h5>🔗 Nghe Audio</h5></summary>

[test_2_output_vi2en.webm](https://github.com/user-attachments/assets/9bad0263-e075-4f59-a08b-67be54f38863)

[▶ Xem/nghe file MP4 trong repository](demo_outputs/test_2_output_vi2en.mp4)

</details>

3. **Test 3 (EN→VI)**
   - **Đầu vào**: the gantry crane at berth seven is malfunctioning we cannot unload the containers the draft survey shows the vessel is listing to port side
   - **Bản dịch**: *Cần cẩu ở cầu cảng số 7 bị trục trặc. Chúng ta không thể dỡ các container. Cuộc giám định mớn nước cho thấy con tàu đang nghiêng sang mạn trái.*
<details>
  <summary><h5>🔗 Nghe Audio</h5></summary>

[test_3_output_en2vi.webm](https://github.com/user-attachments/assets/abb1cbe9-16e4-49b8-abe6-37017fae85c3)

[▶ Xem/nghe file MP4 trong repository](demo_outputs/test_3_output_en2vi.mp4)

</details>

4. **Test 4 (EN→VI)**
   - **Đầu vào**: the solar inverter tripped again check the photovoltaic panels on the rooftop and make sure the string combiner box is not overheating
   - **Bản dịch**: *Bộ đảo lưu năng lượng mặt trời lại bị hỏng. Kiểm tra các tấm pin quang điện trên mái nhà và đảm bảo bộ tổng hợp dây không bị quá nóng.*
<details>
  <summary><h5>🔗 Nghe Audio</h5></summary>

[test_4_output_en2vi.webm](https://github.com/user-attachments/assets/0111b682-b7c0-4d84-83cf-17091a52361a)

[▶ Xem/nghe file MP4 trong repository](demo_outputs/test_4_output_en2vi.mp4)

</details>

5. **Test 5 (VI→EN)**
   - **Đầu vào**: anh ơi cái xe tải nó bị hộp số trục trặc rồi mà két nước cũng rỉ nước ra nữa bạc biên kêu to lắm chắc phải thay rồi mà ống bô cũng bị thủng luôn
   - **Bản dịch**: *Hey, man, the truck's got a malfunctioning gearbox, and the cooling system's leaking water. Connecting rod ball bearings knocked. It's gotta be replaced, and the exhaust pipe's leaking too.*
<details>
  <summary><h5>🔗 Nghe Audio</h5></summary>

[test_5_output_vi2en.webm](https://github.com/user-attachments/assets/0ac1d9d2-23a3-4715-b085-d7b28689a677)

[▶ Xem/nghe file MP4 trong repository](demo_outputs/test_5_output_vi2en.mp4)

</details>

6. **Test 6 (EN→VI)**
   - **Đầu vào**: one worker collapsed from heatstroke bring the first aid kit and check if we have tourniquets and a portable defibrillator in the emergency cabinet
   - **Bản dịch**: *Một công nhân bị ngã do say nắng. Mang theo bộ sơ cứu và kiểm tra xem có ga-rô và máy khử rung cầm tay không trong tủ cấp cứu.*
<details>
  <summary><h5>🔗 Nghe Audio</h5></summary>

[test_6_output_en2vi.webm](https://github.com/user-attachments/assets/b8d8ee00-a9bc-4739-8419-e1bb333d9cad)

[▶ Xem/nghe file MP4 trong repository](demo_outputs/test_6_output_en2vi.mp4)

</details>

7. **Test 7 (EN→VI)**
   - **Đầu vào**: the project manager said that if the geotechnical report confirms the soil bearing capacity is sufficient we can proceed with the shallow foundation design instead of using deep piles which would save us approximately thirty percent of the budget
   - **Bản dịch**: *Giám đốc dự án nói rằng nếu báo cáo địa kỹ thuật xác nhận sức chịu tải của đất là đủ, chúng tôi có thể tiến hành thiết kế móng nông thay vì sử dụng cọc sâu, mà sẽ tiết kiệm cho chúng ta khoảng 30% ngân sách.*
<details>
  <summary><h5>🔗 Nghe Audio</h5></summary>

[test_7_output_en2vi.webm](https://github.com/user-attachments/assets/f24d1c5d-ec8b-4aab-bc2f-8668b2f1eb46)

[▶ Xem/nghe file MP4 trong repository](demo_outputs/test_7_output_en2vi.mp4)

</details>

---


## ☁️ Chạy Trực Tiếp Trên Google Colab

Source code được clone từ GitHub vào `/content/OneVoice`; dataset giữ nguyên trên Google Drive và report được lưu tại `MyDrive/OneVoice/reports`.

Cấu trúc Drive hiện dùng:

```text
MyDrive/
├── onevoice_audio_v1/
│   ├── clean/
│   ├── noisy/
│   ├── noise_bank/
│   └── manifest.jsonl              # notebook audit tự phục hồi nếu thiếu
└── OneVoice/
    ├── model_cache/
    └── reports/
```

Chạy theo thứ tự:

1. [Data Audit V2](https://colab.research.google.com/github/Platypus27-coder/OneVoice/blob/main/notebooks/colab_data_audit_v2.ipynb)
2. [Vietnamese ASR V2](https://colab.research.google.com/github/Platypus27-coder/OneVoice/blob/main/notebooks/colab_vi_asr_v2.ipynb)
3. [Denoiser V2](https://colab.research.google.com/github/Platypus27-coder/OneVoice/blob/main/notebooks/colab_denoiser_v2.ipynb)
4. [Machine Translation V2](https://colab.research.google.com/github/Platypus27-coder/OneVoice/blob/main/notebooks/colab_mt_v2.ipynb)
5. [English ASR V2](https://colab.research.google.com/github/Platypus27-coder/OneVoice/blob/main/notebooks/colab_en_asr_v2.ipynb) — chỉ chạy khi đã có audio English V2.1
6. [Qualcomm Edge Profile V2](https://colab.research.google.com/github/Platypus27-coder/OneVoice/blob/main/notebooks/colab_edge_profile_v2.ipynb) — cần frozen ONNX và `QAI_HUB_API_TOKEN`

Mỗi benchmark đo thật phải xuất `run_manifest.json`, `predictions.csv` và `aggregate.json`. Không dùng predictions sao chép hoặc hệ số WER giả lập.

---

## 🛠️ Chạy Local (Tùy Chọn)

```bash
# 1. Tạo môi trường Conda
conda create -n onevoice python=3.11.8 -y
conda activate onevoice

# 2. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# 3. Tải voice presets (Voice Reference Files)
python scripts/download_voice_preset.py

# 4. Chạy hệ thống dịch thời gian thực (VI → EN)
python src/pipeline.py --direction vi2en

# 5. Chạy hệ thống dịch thời gian thực (EN → VI)
python src/pipeline.py --direction en2vi
```

---

## 📜 Giấy Phép & Tri Ạn Tác Giả

Dự án tuân thủ Giấy phép **CC BY-NC 4.0**.
Chúng tôi trân trọng tri ân các công trình mã nguồn mở được tích hợp:
- **BetterBox-TTS & OmniVoice**: Dolly VN / ContextBoxAI (CC BY-NC 4.0)
- **GIPFormer**: G-Group AI Lab (MIT)
- **SenseVoice**: FunAudioLLM / Alibaba (MIT)
- **VietAI/envit5**: VietAI (MIT)
