# OneVoice V2 notebooks

Các notebook chỉ làm nhiệm vụ điều phối Colab/Google Drive: chúng clone hoặc fast-forward `main` từ
`https://github.com/Platypus27-coder/OneVoice.git` vào `/content/OneVoice`, rồi gọi logic đã được quản lý phiên bản
trong `scripts/` và `src/`. Dataset giữ ở `/content/drive/MyDrive/onevoice_audio_v1` và
`/content/drive/MyDrive/onevoice_audio_v2_1`; cache model và report giữ ở `/content/drive/MyDrive/OneVoice`.

Mỗi benchmark hợp lệ phải sinh `run_manifest.json`, `predictions.csv` và `aggregate.json`. Physical audit sinh
`audit.json` thay cho prediction. Notebook sẽ in toàn bộ log lỗi theo đúng job, không chỉ ném `CalledProcessError`.

## Danh sách chuẩn

1. `colab_generate_english_v2_1.ipynb` — sinh English V2.1 trực tiếp; không phụ thuộc data audit V1 và resume từ Drive.
2. `colab_data_audit_v2.ipynb` — khôi phục manifest nếu cần, logical audit nhanh; physical audit đầy đủ là opt-in
   vì Google Drive phải mở 24.192 WAV nhỏ. Chỉ cần trước benchmark/ nghiệm thu V1.
3. `colab_vi_asr_v2.ipynb` — benchmark GIPFormer tiếng Việt với passthrough trên clean/noisy, split `test` cố định.
4. `colab_denoiser_v2.ipynb` — baseline passthrough và DeepFilterNet khi dependency tương thích; kết quả denoiser
   chỉ được giữ khi vượt quality gate. RNNoise sẽ được bổ sung khi backend runtime hoàn thiện.
5. `colab_mt_finetune_v2.ipynb` — fine-tune EnViT5 VI→EN có checkpoint/optimizer state trên Drive; chạy lại sẽ resume.
6. `colab_mt_finetune_en2vi_v2.ipynb` — fine-tune EnViT5 EN→VI từ base model vào checkpoint Drive riêng; không ghi đè VI→EN.
7. `colab_mt_v2.ipynb` — benchmark EnViT5 raw/context trên `test`, `minimal_pairs` và `safety`; tự nhận candidate Drive theo từng chiều.
8. `colab_en_asr_v2.ipynb` — chỉ chạy sau khi có audio English V2.1 và audit tối thiểu 6 speaker/voice đạt.
9. `colab_edge_profile_v2.ipynb` — export/compile/profile model ONNX đã freeze trên Qualcomm AI Hub hosted device.
10. `colab_sensevoice_evaluate_en_v1.ipynb` — đánh giá checkpoint SenseVoice EN đã fine-tune trên held-out clean/noisy, có resume prediction trên Drive.
11. `colab_sensevoice_export_onnx_v1.ipynb` — export candidate SenseVoice EN đã qua quality gate sang bundle ONNX FP32 riêng; chưa thay runtime tới khi benchmark ONNX đạt.

`colab_vi_asr_finetune_submission.ipynb` đã bị loại: nó fine-tune Whisper Tiny, không phải kiến trúc GIPFormer của
OneVoice. Fine-tune GIPFormer chỉ bắt đầu khi có checkpoint PyTorch/icefall tương thích và khi benchmark/context gate
chứng minh cần thiết.

Fine-tune MT chỉ dùng `train.csv`, chọn checkpoint bằng `dev.csv`; `test.csv` luôn được giữ cho đánh giá cuối. Fine-tune
VI→EN ghi `training_state.pt`, `checkpoints/epoch-*`, `best/`, `training_history.json` và `run_manifest.json` vào
`MyDrive/OneVoice/models/envit5_finetuned_vi2en_v2`, nên có thể đổi Colab GPU/máy/tài khoản miễn là mount cùng Drive.
EN→VI dùng cùng cơ chế nhưng ghi độc lập vào `MyDrive/OneVoice/models/envit5_finetuned_en2vi_v1`; hai checkpoint
không được ghi đè hoặc dùng lẫn nhau.
Nếu
Drive V1 chưa có `manifest.jsonl`, notebook audit có thể phục hồi pairing/transcript/split từ filename và
`utterances_all.csv`. Speaker, crop noise, SNR, RIR ngẫu nhiên của V1 không thể phục hồi nên vẫn là giới hạn phải
báo cáo, không được dùng để tuyên bố chất lượng theo speaker/noise.
## Release decision — 2026-08-29

Production VI ASR is pinned to the verified pretrained GIPFormer ONNX bundle at
`MyDrive/OneVoice/models/gipformer`. Construction-domain GIPFormer fine-tune
experiments (`head_ft_v1`, `icefall_ft_v1` through `icefall_ft_v4`) failed the
development quality gate and must not replace the baseline. The fine-tune
notebook remains experimental only; it is not part of the runtime path.

## Release benchmark (current runtime only)

`colab_benchmark_report_v2.ipynb` now uses `PROFILE = 'release'` and writes
`MyDrive/OneVoice/reports/onevoice_release_benchmark_v1/`. It includes only the
reviewed runtime artifacts: the official GIPFormer VI->EN baseline, fine-tuned
SenseVoice FP32 EN->VI, and the promoted EnViT5 validators. Historical,
diagnostic, INT8, and rejected GIPFormer fine-tune reports remain available in
the original `onevoice_benchmark_report_v1/` output and are never presented as
the current release.

To intentionally rebuild the complete historical report, change `PROFILE` to
`'all'`; this is an audit view, not the release result.

12. `colab_streaming_v2.ipynb` — replay một WAV qua pipeline streaming thật (frame 32 ms, VAD rolling, stable-prefix/semantic commit, bounded workers và ordered TTS); ghi `stream_result.json`, `latency.json` và `latency_summary.json` trên Drive. Notebook chỉ mount Drive/clone GitHub; không chứa logic runtime.
