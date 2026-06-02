# 🎙️ OneVoice Edge — Hệ Thống Phiên Dịch Giọng Nói Thời Gian Thực
> **Cuộc thi OneVoice AI Challenge 2026 — Team Impact**

Hệ thống dịch thuật Speech-to-Speech chạy **100% Offline**, được thiết kế đặc biệt cho môi trường công nghiệp (nhà máy, công trường). Dự án được tối ưu hóa để chạy trên chip **Qualcomm Snapdragon NPU** với độ trễ (latency) dưới **1 giây** và mức ngốn RAM dưới **200 MB**.

---

## 🏭 Vấn Đề Thực Tế
Rào cản ngôn ngữ giữa chuyên gia nước ngoài và kỹ sư bản địa gây giảm năng suất và nguy cơ mất an toàn. Các ứng dụng như Google Translate không thể dùng được vì:
- Bắt buộc phải có Internet (Cloud-based).
- Chết hoàn toàn khi gặp tiếng ồn máy móc công trường.

## 🚀 Giải pháp — Kiến Trúc 4 Trạm Cục Bộ (Edge AI)

### Luồng 1: VI → EN

```text
Microphone 
    │
    ▼
┌─────────────────────────────────┐
│  Trạm 0: Lọc Ồn (Denoise)       │  GIPFormer ONNX (INT8)
│  Khử tiếng máy cắt, gió, ồn...  │  ~10ms
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Trạm 1: Nhận Diện (ASR)        │  GIPFormer
│  Giọng nói → Văn bản Tiếng Việt │  Chuyên dụng cho tiếng ồn công nghiệp
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Trạm 2: Dịch Thuật (MT)        │  VietAI/envit5-translation
│  Văn bản VI → EN                │  1 model cho cả 2 chiều (~600MB)
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Trạm 3: Phát Âm (TTS)          │  F5-TTS (voice clone) / OmniVoice
│  Văn bản EN → Giọng nói         │  Bảo toàn giọng nói qua ngôn ngữ
└─────────────────────────────────┘
    │
    ▼
Speaker / Earphone 
```

### Luồng 2: EN → VI

```text
Microphone
    │
    ▼
┌─────────────────────────────────┐
│  Trạm 0: Lọc Ồn (Denoise)       │  GIPFormer ONNX (INT8)
│  Khử tiếng máy cắt, gió, ồn...  │  ~10ms
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Trạm 1: Nhận Diện (ASR)        │  SenseVoice Small
│  Giọng nói → Văn bản Tiếng Anh  │  + Trích xuất Cảm Xúc (Emotion)
└─────────────────────────────────┘
    │ metadata: [text, emotion]
    ▼
┌─────────────────────────────────┐
│  Trạm 2: Dịch Thuật (MT)        │  VietAI/envit5-translation
│  Văn bản EN → VI                │  Luân chuyển metadata cảm xúc
└─────────────────────────────────┘
    │ metadata: [translated_text, emotion]
    ▼
┌─────────────────────────────────┐
│  Trạm 3: Phát Âm (TTS)          │  OmniVoice (Voice Design)
│  Văn bản VI → Giọng nói         │  Đọc Tiếng Việt mô phỏng cảm xúc
└─────────────────────────────────┘
    │
    ▼
Speaker / Earphone 
```

**Mục tiêu Độ trễ tổng: < 600ms (Vượt chỉ tiêu 1s của giải)**

---

## 🌐 Các Chế Độ Hoạt Động (Translation Directions)
Hệ thống là một đường ống hai chiều, cho phép bạn chuyển đổi linh hoạt.

| Hướng (Direction) | Đầu vào (Người nói) | Đầu ra (Loa phát) | Lệnh chạy (Flag) |
|-------------------|---------------------|-------------------|------------------|
| **VI → EN** (Mặc định) | Kỹ sư Việt Nam 🇻🇳 | Chuyên gia Anh 🇬🇧 | `--direction vi2en` |
| **EN → VI** | Chuyên gia Anh 🇬🇧 | Kỹ sư Việt Nam 🇻🇳 | `--direction en2vi` |

---

## ⚙️ Hướng dẫn Cài Đặt (Self-Contained)

```bash
# 1. Tạo môi trường Conda
conda create -n onevoice python=3.11.8
conda activate onevoice

# 2. Cài đặt các thư viện (Không dính repo ngoài)
pip install -r requirements.txt
```

*(Lưu ý: Lần chạy đầu tiên, hệ thống sẽ tự động tải các file weights của GIPFormer và SenseVoice từ HuggingFace/ModelScope về cache cục bộ. Để chạy 100% Offline không cần Wifi, hãy đảm bảo bạn đã chạy pipeline ít nhất 1 lần khi có mạng).*

---

## 🚀 Cách Chạy Dự Án

### 1. Dịch từ Kỹ sư Việt Nam sang Tiếng Anh (VI → EN)
Đây là chế độ mặc định. Hệ thống sẽ bật mic, nghe bạn nói Tiếng Việt, khử ồn bằng GIPFormer, dịch sang Tiếng Anh và đọc ra loa.

```bash
python src/pipeline.py --direction vi2en
```

### 2. Dịch từ Chuyên gia Anh sang Tiếng Việt (EN → VI)
Hệ thống sẽ nghe tiếng Anh. Đặc biệt, **SenseVoice** sẽ tự động trích xuất cảm xúc (Ví dụ: Giận dữ, Vui vẻ). Thái độ này sẽ được truyền thẳng xuống **OmniVoice** để đọc Tiếng Việt với đúng tông giọng gắt gỏng hoặc vui nhộn của người gốc.

```bash
python src/pipeline.py --direction en2vi
```

---

## ⚠️ Giấy phép & Tri ân tác giả
Dự án tuân thủ Giấy phép **CC BY-NC 4.0** (Tuyệt đối không dùng cho mục đích thương mại).
Chúng tôi đã tích hợp trực tiếp, trích xuất và tinh chỉnh mã nguồn từ các tác giả:
- **BetterBox-TTS & OmniVoice**: Dolly VN / ContextBoxAI (CC BY-NC 4.0)
- **GIPFormer**: G-Group AI Lab (MIT)
- **SenseVoice**: FunAudioLLM / Alibaba (MIT)
- **VietAI/envit5**: VietAI (MIT)

> **Cảm ơn Ban Tổ Chức OneVoice AI Challenge 2026!**
