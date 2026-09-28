# Sửa đầu vào giữa ASR, safety và MT

Benchmark model đo khả năng của model với đầu vào xác định. Khi đưa vào pipeline,
VAD chia audio, ASR tạo transcript, các bước xử lý văn bản và context chuẩn bị
đầu vào dịch. Một lỗi ở các bước này có thể thay đổi câu trước khi model dịch
nhận được nó. Kết quả benchmark model vẫn cần được đối chiếu riêng với chất
lượng của pipeline trên WAV có nhiễu.

## Thay đổi runtime

- `normalize_asr` chỉ chuẩn hóa Unicode, khoảng trắng và chữ hoa của ASR.
  Giữ số, đơn vị, mã thiết bị và từ phủ định. Các phép mở rộng phục vụ đọc TTS
  như “giàn giáo” thành “hệ thống giàn giáo” không chạy trước safety lookup.
- Mạo từ tiếng Anh “a” được giữ nguyên; bộ mở rộng TTS chỉ đọc A thành ampere
  khi có số đo đi trước.
- Commit safety nhận biết cùng hành động đã phát trong một câu, kể cả khi
  ASR thêm “Ê” và chuyển sang một safety ID khác có cùng bản dịch đã duyệt.
- Safety lookup nhận biết hai dạng ngữ pháp giới hạn: “wear your/the ...” và
  “climb on/onto ...”. Match gần đúng bảo vệ phủ định, hướng, số và các cặp
  hành động đối lập; kết quả mơ hồ không được chọn tự động.
- Validator nhận biết cách viết đầy đủ của cùng đơn vị, ví dụ km/kilometers,
  và một số từ đồng nghĩa kỹ thuật. kg và km vẫn là hai đơn vị khác nhau.
- eSpeak tổng hợp với amplitude 80 để tạo khoảng dự phòng trong PCM từ đầu.
  Không giảm âm lượng của WAV đã bị clipping để che lỗi.
- Báo cáo streaming lưu transcript ASR nguyên gốc cùng các bản dịch đã bị
  validator chặn. Benchmark báo lý do chặn cụ thể thay cho lỗi chung “latency
  report is empty”.

Các bản sửa này giữ các trọng số model đang dùng. Không dùng reference của
benchmark để sửa transcript hay làm đầu vào suy luận.

## Kết quả kiểm tra ngày 28/09/2026

Đã replay 557 lượt trên các WAV của bộ benchmark cục bộ: 305 VI→EN và
252 EN→VI. Bộ này gồm toàn bộ ca FAIL cũ, toàn bộ safety canonical và
100 noisy control từng PASS mỗi chiều. Đây là kiểm tra hồi quy có chủ đích,
không phải một đợt full test mới.

- VI→EN: 27 ca FAIL cũ chuyển sang PASS gate chức năng; số FAIL trong bộ
  hồi quy giảm từ 97 xuống 70. Safety đầy đủ tăng từ 108/126 lên 114/126.
- EN→VI: 24 ca FAIL cũ chuyển sang PASS gate chức năng; số FAIL trong bộ
  hồi quy giảm từ 86 xuống 62. Safety đầy đủ tăng từ 66/126 lên 90/126.
- Không có ca từng PASS nào trong bộ hồi quy chuyển thành FAIL. Kết luận này
  chỉ áp dụng cho các ca đã replay, không suy rộng sang mọi noisy test WAV.
- TTS tiếng Việt độc lập đã chạy đủ 979 prompt test: 979/979 qua gate WAV,
  xử lý được 3 ca clipping của đợt trước. Gate này kiểm tra tín hiệu đầu ra,
  không thay thế đánh giá độ rõ giọng bằng người nghe.
- 165 unit test đều PASS.

51 ca hồi phục là kết quả của **gate chức năng**, không phải 51 câu đã được
chứng minh đúng nghĩa hoàn toàn. Validator giảm một số false positive do
đơn vị/từ đồng nghĩa, nhưng vẫn phải kiểm tra bản dịch so với lời gốc.
126 safety entry mỗi chiều là dữ liệu tổng hợp duyệt nội bộ; các entry có
thể dùng lại câu hoặc WAV, không phải 126 người nói/câu độc lập.

Hai chiều và kiểm tra TTS đã chạy song song. Không dùng latency của đợt này
để kết luận nhanh/chậm hơn benchmark cũ; phép so sánh latency cần chạy tuần tự
trên cùng máy, cùng cấu hình và cùng corpus.

Bằng chứng cục bộ được giữ trong `reports/pc_pipeline_integrity_v3`: mỗi chiều
có `benchmark_contract.json`, `summary.json`, `runtime.log` và kết quả từng ca;
TTS nằm ở `tts_vi_full/aggregate.json`. Contract kiểm tra hash cấu hình, WAV,
dependencies và mã runtime. Hash mã runtime của cả hai lượt đo là:

```text
b93635b5fe482eecdaf0614cc559a849bb1ca1edc8c5ee6f62421bbdaafd0f10
```

## Ưu tiên còn lại

1. Tách lỗi ASR khỏi lỗi MT bằng cách đối chiếu transcript với lời gốc và
   dịch cùng câu bằng transcript ASR/gold. Đã xác nhận một số safety EN ngắn
   vẫn sai cả khi suy luận nguyên WAV: “Smoke!” → “S.”, “Fire!” → “5re.”.
   Không mở rộng safety matcher để đoán các đầu vào này thành lệnh nguy hiểm.
2. Xử lý nhóm thuật ngữ dịch sai hoặc thiếu, ví dụ radiator, đất đào và rào
   chắn; giữ cơ chế chặn mất phủ định/đơn vị. Nếu cần fine-tune thêm, lấy dữ
   liệu từ train và chọn checkpoint trên dev; không đưa các câu test vào train.
3. Khi chốt runtime, chạy full suite mới và P5 tuần tự. Giữ báo cáo model,
   hồi quy và full pipeline riêng; không đổi mẫu số hoặc bỏ ca FAIL để tăng số.

## Cách kiểm tra bản sửa

`scripts/run_desktop_regression.py` dùng WAV và cấu hình của đợt full benchmark
trước đó. Nó chạy lại mọi ca FAIL, tất cả 126 safety entry mỗi chiều và 100 ca
noisy từng PASS được chọn theo hash cố định. Nó kiểm tra hash WAV trước khi
chạy và ghi một contract mới; các báo cáo cũ được giữ riêng.

```powershell
$Repo = "D:\code\.vscode\OneVoice\onevoice-edge"
$BundleRoot = "D:\OneVoiceDesktop\onevoice-v2-rc1"
$Python = "D:\MINICONDA\envs\onevoice\python.exe"
$env:PYTHONIOENCODING = "utf-8"

foreach ($Direction in @("vi2en", "en2vi")) {
    Set-Location (Join-Path $BundleRoot $Direction)
    & $Python -u "$Repo\scripts\run_desktop_regression.py" `
        --baseline-dir "$Repo\reports\pc_pipeline_afterfix_v2\$Direction\pipeline_full_noisy" `
        --config ".\runtime_config.yaml" `
        --output-dir "$Repo\reports\pc_pipeline_integrity_v3\$Direction" `
        --controls 100 --resume
}
```

Exit code 2 nghĩa là đã chạy xong nhưng còn ca FAIL, cần đọc `summary.json`.
Không coi tỷ lệ PASS của bộ chọn theo lỗi là độ chính xác của full test.
PASS chức năng cũng cần đối chiếu `reference_validation_errors`, WER và đầu ra
thực tế: ASR nghe “ki-lô-gam” thành “km” vẫn làm sai lời gốc dù bản dịch có
thể đúng theo transcript.

Sau khi chốt runtime, dùng thư mục mới để chạy lại full suite:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "$Repo\scripts\run_desktop_benchmarks.ps1" `
    -OutputRoot "$Repo\reports\pc_pipeline_full_integrity_v3"
```

Đợt full này gồm mọi noisy test WAV cục bộ, mọi canonical safety entry,
TTS đúng ngôn ngữ đầu ra và P5. Full benchmark trước sửa tiếp tục nằm trong
`docs/desktop_runtime_rebenchmark`; không ghép kết quả kiểm tra hồi quy vào
báo cáo đó để tạo một tỷ lệ full mới.
