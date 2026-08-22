# OneVoice V2 — Colab / Google Drive runbook

Mục tiêu của workflow này là có thể mất GPU, restart runtime hoặc đổi tài khoản Colab mà không mất dữ liệu đã chạy.
`/content` chỉ là vùng tạm; mọi thứ cần giữ nằm trên Google Drive.

## 1. Chuẩn bị Drive một lần

```text
MyDrive/
├── onevoice_audio_v1/
│   ├── clean/
│   ├── noisy/
│   ├── noise_bank/
│   └── manifest.jsonl              # audit sẽ phục hồi nếu thiếu
└── OneVoice/
    ├── model_cache/                # Hugging Face / Torch cache
    ├── models/
    │   └── envit5_finetuned_vi2en_v2/
    │       ├── training_state.pt   # epoch + optimizer để resume
    │       ├── checkpoints/
    │       ├── best/
    │       ├── training_history.json
    │       └── run_manifest.json
    └── reports/                    # benchmark outputs không bị ghi đè khi complete
```

Không xóa `OneVoice/model_cache`, `OneVoice/models` hay `OneVoice/reports` khi muốn đổi GPU. Chúng là checkpoint
durable, không phải cache runtime tạm.

Nếu chạy bằng tài khoản Colab khác, hãy share **cả** thư mục `OneVoice` và `onevoice_audio_v1` cho tài khoản đó với
quyền Editor, rồi chọn “Add shortcut to Drive” vào `My Drive`. Sau đó path trong notebook vẫn là
`/content/drive/MyDrive/OneVoice` và `/content/drive/MyDrive/onevoice_audio_v1`. Cách an toàn nhất vẫn là dùng cùng
một Google Drive làm nơi lưu durable state.

## 2. Mở notebook đúng phiên bản

Mỗi link dưới đây luôn clone/pull code từ GitHub; dataset và output không nằm trong GitHub.

1. [Data audit](https://colab.research.google.com/github/Platypus27-coder/OneVoice/blob/main/notebooks/colab_data_audit_v2.ipynb)
2. [VI ASR benchmark](https://colab.research.google.com/github/Platypus27-coder/OneVoice/blob/main/notebooks/colab_vi_asr_v2.ipynb)
3. [Denoiser benchmark](https://colab.research.google.com/github/Platypus27-coder/OneVoice/blob/main/notebooks/colab_denoiser_v2.ipynb)
4. [EnViT5 VI→EN fine-tune](https://colab.research.google.com/github/Platypus27-coder/OneVoice/blob/main/notebooks/colab_mt_finetune_v2.ipynb)
5. [MT benchmark](https://colab.research.google.com/github/Platypus27-coder/OneVoice/blob/main/notebooks/colab_mt_v2.ipynb)

Vào `Runtime → Change runtime type → T4 GPU` (hoặc GPU đang có) trước khi fine-tune/benchmark MT. Sau đó chọn
`Runtime → Run all`, và cho phép mount Drive khi Colab hỏi.

## 3. Thứ tự chạy khuyến nghị

1. `colab_data_audit_v2.ipynb`: chạy logical audit trước. Physical audit để `False` nếu chưa cần quét 24.192 WAV.
2. `colab_vi_asr_v2.ipynb`: tạo baseline GIPFormer clean/noisy nếu Drive chưa có report đầy đủ.
3. `colab_denoiser_v2.ipynb`: benchmark passthrough/DeepFilterNet khi dependency khả dụng.
4. Không cần fine-tune lại checkpoint đã upload tại `platypus123/onevoice-envit5-vi-en`. Chỉ dùng
   `colab_mt_finetune_v2.ipynb` khi tạo một candidate VI→EN mới.
5. `colab_mt_v2.ipynb`: chạy 12 baseline jobs và 6 candidate VI→EN jobs. Cuối notebook tạo
   `MyDrive/OneVoice/reports/benchmark_dashboard.md`.

English ASR và Qualcomm profile chỉ chạy sau khi có English V2.1 / model ONNX frozen tương ứng.

## 4. Khi Colab bị ngắt giữa chừng

Mở lại **đúng notebook**, chọn GPU bất kỳ, mount cùng Drive, rồi `Run all` từ đầu.

- Benchmark: job có đủ `aggregate.json`, `predictions.csv`, `run_manifest.json` sẽ được in là `already complete` và bị
  bỏ qua. Job dở dang chạy lại từ đầu; kết quả job hoàn chỉnh cũ vẫn giữ trên Drive.
- Fine-tune: script đọc `models/envit5_finetuned_vi2en_v2/training_state.pt`, nạp checkpoint và optimizer state, rồi
  bắt đầu từ epoch kế tiếp. Dữ liệu trong epoch đang dang dở có thể chạy lại; epoch đã lưu thì không.
- Để train thêm epoch sau này, tăng `TOTAL_EPOCHS` trong notebook (ví dụ `3` lên `5`). Giữ nguyên output folder.

Không đổi `CHECKPOINT_ROOT` giữa chừng. Không đổi direction của một run đang resume; checkpoint hiện được thiết kế
cho VI→EN. `best/` chỉ là candidate, chưa được promote vào runtime production cho đến khi held-out MT gate đạt.

## 5. Kiểm tra kết quả

Sau mỗi benchmark, kiểm tra folder report có ba file: `predictions.csv`, `aggregate.json`, `run_manifest.json`.
`run_manifest.json` ghi revision source, dependency và model reference. Dashboard chỉ tổng hợp report đã có; ô `—`
nghĩa là thiếu số liệu, không phải pass.

Chi tiết task và gate: [ONEVOICE_END_TO_END_MASTER_PLAN.md](../ONEVOICE_END_TO_END_MASTER_PLAN.md).
