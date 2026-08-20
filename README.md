# OneVoice Edge — Hệ Thống Phiên Dịch Giọng Nói Thời Gian Thực (Edge AI)

<img width="2352" height="1792" alt="OneVoice Edge Banner" src="https://github.com/user-attachments/assets/f4747894-01d8-4889-bbf5-a0d2a5c01de7" />

Hệ thống dịch thuật Speech-to-Speech chạy **100% Offline**, được thiết kế đặc biệt cho môi trường công nghiệp (nhà máy, công trường). Dự án được tối ưu hóa để chạy trên thiết bị Edge / chip **Qualcomm Snapdragon NPU** với độ trễ (latency) dưới **1 giây** và mức ngốn RAM cực thấp (< **200 MB**).

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

**⚡ Chỉ tiêu hiệu năng tổng:** Độ trễ < **1000ms**, 100% Offline, chống crash tuyệt đối với cơ chế Fallback thông minh.

---

## 🎛️ Các Chế Độ Hoạt Động (Translation Directions)

Hệ thống là một đường ống hai chiều, cho phép chuyển đổi linh hoạt qua cờ lệnh runtime:

| Hướng (Direction) | Đầu vào (Người nói) | Đầu ra (Loa phát) | Lệnh chạy (Flag) |
|---|---|---|---|
| **VI → EN** (Mặc định) | Người Việt | Người Anh | `python src/pipeline.py --direction vi2en` |
| **EN → VI** | Người Anh | Người Việt | `python src/pipeline.py --direction en2vi` |

---

## 🎬 Demo Kết Quả Dịch Thuật & Voice Cloning

Dưới đây là 7 kịch bản kiểm thử (Test Scenarios) với các thuật ngữ chuyên ngành hóc búa, từ lóng thi công và các tình huống thực tế tại công trường. Hệ thống dịch chuẩn xác và xuất file âm thanh thành công vào thư mục `demo_outputs/`:

1. **Test 1 (VI→EN)**
   - **Đầu vào**: Cậu đã làm gì với nó vậy thêm năng lượng hả nó hoạt động như thế nào vậy cho mình mượn chút đừng có keo kiệt vậy chứ hôm nay lớp mình có bài kiểm tra môn thể dục nên mình rất là cần nó luôn xài xong mình trả lại liền
   - **Bản dịch**: *What did you do with it? More power, huh? How it works. Well, let me borrow some. don't be mean, because we have a gym test today, so... I really need it. I'll give it back when I'm done.*
<details>
  <summary><h5>🔗 Nghe Audio</h5></summary>

[test_1_output_vi2en.webm](https://github.com/user-attachments/assets/274cef99-a640-4cc5-8ed6-2c7836ec417b)

</details>

2. **Test 2 (VI→EN)**
   - **Đầu vào**: Ê bạn ơi cái máy xúc số ba nó bị xì nhớt thủy lực rồi bơm bê tông cũng kẹt luôn qua kiểm tra lẹ giùm mình đi chứ để vậy là cháy van an toàn nha
   - **Bản dịch**: *Hey, buddy, that excavator number three, it's leaking hydraulic fluid. The pump's jammed, too. please check it immediately. the safety valve will blow out.*
<details>
  <summary><h5>🔗 Nghe Audio</h5></summary>

[test_2_output_vi2en.webm](https://github.com/user-attachments/assets/9bad0263-e075-4f59-a08b-67be54f38863)

</details>

3. **Test 3 (EN→VI)**
   - **Đầu vào**: the gantry crane at berth seven is malfunctioning we cannot unload the containers the draft survey shows the vessel is listing to port side
   - **Bản dịch**: *Cần cẩu ở cầu cảng số 7 bị trục trặc. Chúng ta không thể dỡ các container. Cuộc giám định mớn nước cho thấy con tàu đang nghiêng sang mạn trái.*
<details>
  <summary><h5>🔗 Nghe Audio</h5></summary>

[test_3_output_en2vi.webm](https://github.com/user-attachments/assets/abb1cbe9-16e4-49b8-abe6-37017fae85c3)

</details>

4. **Test 4 (EN→VI)**
   - **Đầu vào**: the solar inverter tripped again check the photovoltaic panels on the rooftop and make sure the string combiner box is not overheating
   - **Bản dịch**: *Bộ đảo lưu năng lượng mặt trời lại bị hỏng. Kiểm tra các tấm pin quang điện trên mái nhà và đảm bảo bộ tổng hợp dây không bị quá nóng.*
<details>
  <summary><h5>🔗 Nghe Audio</h5></summary>

[test_4_output_en2vi.webm](https://github.com/user-attachments/assets/0111b682-b7c0-4d84-83cf-17091a52361a)

</details>

5. **Test 5 (VI→EN)**
   - **Đầu vào**: anh ơi cái xe tải nó bị hộp số trục trặc rồi mà két nước cũng rỉ nước ra nữa bạc biên kêu to lắm chắc phải thay rồi mà ống bô cũng bị thủng luôn
   - **Bản dịch**: *Hey, man, the truck's got a malfunctioning gearbox, and the cooling system's leaking water. Connecting rod ball bearings knocked. It's gotta be replaced, and the exhaust pipe's leaking too.*
<details>
  <summary><h5>🔗 Nghe Audio</h5></summary>

[test_5_output_vi2en.webm](https://github.com/user-attachments/assets/0ac1d9d2-23a3-4715-b085-d7b28689a677)

</details>

6. **Test 6 (EN→VI)**
   - **Đầu vào**: one worker collapsed from heatstroke bring the first aid kit and check if we have tourniquets and a portable defibrillator in the emergency cabinet
   - **Bản dịch**: *Một công nhân bị ngã do say nắng. Mang theo bộ sơ cứu và kiểm tra xem có ga-rô và máy khử rung cầm tay không trong tủ cấp cứu.*
<details>
  <summary><h5>🔗 Nghe Audio</h5></summary>

[test_6_output_en2vi.webm](https://github.com/user-attachments/assets/b8d8ee00-a9bc-4739-8419-e1bb333d9cad)

</details>

7. **Test 7 (EN→VI)**
   - **Đầu vào**: the project manager said that if the geotechnical report confirms the soil bearing capacity is sufficient we can proceed with the shallow foundation design instead of using deep piles which would save us approximately thirty percent of the budget
   - **Bản dịch**: *Giám đốc dự án nói rằng nếu báo cáo địa kỹ thuật xác nhận sức chịu tải của đất là đủ, chúng tôi có thể tiến hành thiết kế móng nông thay vì sử dụng cọc sâu, mà sẽ tiết kiệm cho chúng ta khoảng 30% ngân sách.*
<details>
  <summary><h5>🔗 Nghe Audio</h5></summary>

[test_7_output_en2vi.webm](https://github.com/user-attachments/assets/f24d1c5d-ec8b-4aab-bc2f-8668b2f1eb46)

</details>

---

## 🛠️ Hướng Dẫn Cài Đặt & Chạy Hệ Thống

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
