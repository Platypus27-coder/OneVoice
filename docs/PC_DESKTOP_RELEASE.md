# OneVoice V2 — Windows desktop release candidate

## Scope

The current delivery target is a **Windows PC offline demo** for a portfolio
and CV. A single headset selected as the Windows output is a local playback
acceptance step. Android, Snapdragon acceleration and two-participant headset
routing remain future work.

Reference machine for the first acceptance run:

| Component | Value |
|---|---|
| Laptop | ASUS TUF Gaming F15 FX506HF |
| CPU | Intel Core i5-11400H |
| RAM | 24 GB |
| GPU | NVIDIA RTX 2050, 4 GB VRAM |
| Runtime mode | local, offline, `edge` profile |

The named P5 budget `desktop_tuf_f15_rtx2050` is deliberately different from
the portable `edge_200mb` budget. It records the actual desktop target rather
than making an unsupported 200 MB mobile claim.

## What “complete for CV” means

The release candidate is complete when the following evidence is checked in or
linked from the release report:

1. Current ASR/MT artifacts are hash-locked and copied into a local bundle.
2. The bundle loads without network access and passes one smoke per direction.
3. The fixed streaming suite passes normal and approved-safety routes in both
   directions.
4. A 30-minute offline streaming soak report has zero failed turns.
5. P5 is measured on this Windows PC using the named desktop hardware profile.
6. `report.md`, `report.html`, `summary.json`, and the README SVG summarize the
   selected runtime rather than historical candidates.

This is a demonstrable offline prototype, not a claim of site-ready safety
deployment. The data is synthetic/hosted, safety WAVs are internally approved
demo assets, the Vietnamese GIPFormer baseline remains below the 95% critical
term gate, and system-voice TTS is functional rather than a voice-quality
claim.

## Make a local bundle once

Run these commands on a machine that currently has the verified artifacts
(for example, after copying them off Drive). They never download model files.
`copy` can be re-run safely: hash-matching files are reused.

```powershell
$Repo = "D:\code\.vscode\OneVoice\onevoice-edge"
$Artifacts = "D:\OneVoiceArtifacts\runtime_demo_manifest.json"
$Lock = "D:\OneVoiceArtifacts\release_lock_v2.json"
$BundleRoot = "D:\OneVoiceDesktop\onevoice-v2-rc1"
$RuntimeConfig = "D:\OneVoiceArtifacts\runtime_demo_local.yaml"

Set-Location $Repo
python scripts/build_release_bundle.py `
  --artifact-manifest $Artifacts `
  --output-dir "$BundleRoot\vi2en" `
  --direction vi2en --mode copy --release-lock $Lock `
  --runtime-config $RuntimeConfig
python scripts/build_release_bundle.py `
  --artifact-manifest $Artifacts `
  --output-dir "$BundleRoot\en2vi" `
  --direction en2vi --mode copy --release-lock $Lock `
  --runtime-config $RuntimeConfig
```

Each copied bundle contains `manifest.json`, `receipt.json`, the selected
models, approved safety audio, construction data, and a `runtime_config.yaml`
whose paths are local to that bundle. A bundle is an immutable model/data
package; the Git repository provides the executable Python runtime.

## Offline smoke on Windows

Install the locked Python dependencies and the required local audio backends
once. In particular, the selected runtime needs `sherpa_onnx` for VI ASR,
`funasr_onnx` for EN ASR, ONNX Runtime/Transformers for MT, and an offline
system TTS backend. The preflight check fails clearly if one is absent.

For the local eSpeak NG demo fallback installed inside the `onevoice` Conda
environment, register its executable and voice-data paths once:

```powershell
$Repo = "D:\code\.vscode\OneVoice\onevoice-edge"
& "$Repo\scripts\install_espeak_conda_hooks.ps1" `
  -Prefix "D:\MINICONDA\envs\onevoice"
```

Then open a new PowerShell session and run `conda activate onevoice`. The
activation hook makes `espeak-ng` discoverable and sets `ESPEAK_DATA_PATH`; it
does not install eSpeak NG or change the voice model. This lightweight system
voice is a functional offline demo fallback, not a production voice-quality
claim.

Run each command from the corresponding bundle directory so
`runtime_config.yaml` resolves only local paths:

```powershell
$Repo = "D:\code\.vscode\OneVoice\onevoice-edge"
Set-Location "D:\OneVoiceDesktop\onevoice-v2-rc1\vi2en"
$env:PYTHONPATH = "$Repo\src"
python "$Repo\scripts\verify_release_bundle.py" `
  --bundle-dir . --direction vi2en --config runtime_config.yaml --profile edge `
  --input-file artifacts\safety_audio\SAFE2_0001_en2vi.wav `
  --output-file reports\vi2en_smoke.wav --report-dir reports

Set-Location "D:\OneVoiceDesktop\onevoice-v2-rc1\en2vi"
python "$Repo\scripts\verify_release_bundle.py" `
  --bundle-dir . --direction en2vi --config runtime_config.yaml --profile edge `
  --input-file artifacts\safety_audio\SAFE2_0001_vi2en.wav `
  --output-file reports\en2vi_smoke.wav --report-dir reports
```

Expected result: two `bundle_verify_*.json` reports with `passed: true` and
`network_blocked: true`.

## Thử trực tiếp bằng mic laptop

Chọn **Microphone Array (Realtek)** làm input mặc định trong Windows và tai
nghe làm output. Chạy từ thư mục bundle sau khi kích hoạt Conda `onevoice`.
Runtime dùng bản sao riêng của từng frame mic để buffer của driver không ghi
đè âm thanh đang chờ ASR; các bài replay WAV không kiểm tra vòng đời buffer này.
Đây cũng là cách giữ dữ liệu trong
[ví dụ callback chính thức của sounddevice](https://github.com/spatialaudio/python-sounddevice/blob/master/examples/rec_unlimited.py).

Nếu mức RMS giọng nói đo trên mic thấp hơn ngưỡng `0.015`, có thể thử
`--vad-energy-threshold 0.005`. Tùy chọn chỉ áp dụng cho lần chạy này, được ghi
trong `runtime_summary.json` khi dùng `--report-dir`, và không sửa config gốc.
Ngưỡng cần kiểm tra cả khi nói lẫn khi im lặng; `0.005` là mức thử cho mic thu
nhỏ, không phải giá trị đã được xác nhận phù hợp cho mọi thiết bị.

```powershell
conda activate onevoice
$Repo = "D:\code\.vscode\OneVoice\onevoice-edge"
$OneVoicePython = "D:\MINICONDA\envs\onevoice\python.exe"
$env:PYTHONPATH = "$Repo\src"
Set-Location "D:\OneVoiceDesktop\onevoice-v2-rc1\vi2en"
& $OneVoicePython -u "$Repo\src\pipeline.py" --config runtime_config.yaml --direction vi2en --profile edge --offline --vad-energy-threshold 0.005 --report-dir reports\live_mic_vi2en
```

Đợi `OneVoice V2 LIVE`, nói một câu tiếng Việt rõ rồi nghỉ khoảng một giây
để xác nhận hết câu. Dừng bằng `Ctrl+C` để lưu báo cáo. Xác nhận thêm bằng
nghe đầu ra và so sánh câu nhận dạng với lời nói; chỉ có log nạp model chưa
đủ để kết luận live microphone chạy đúng.

## Kiểm tra TTS lặp và tai nghe Bluetooth trên Windows

Windows English TTS ở chế độ `auto` ưu tiên giọng Microsoft SAPI đã cài sẵn,
chạy trong tiến trình PowerShell riêng để tránh vòng lặp COM `pyttsx3` bị kẹt.
Nếu Windows không có giọng tiếng Anh dùng được, OneVoice mới chuyển sang
eSpeak NG cục bộ. Tiếng Việt tiếp tục dùng giọng cục bộ hiện có. Không tải
model TTS trong edge profile. Mỗi lần tổng hợp có timeout mặc định 15 giây
(có thể đặt `tts.synthesis_timeout_s` trong config).

Với runtime eSpeak portable trên Windows, tiến trình con nhận
`ESPEAK_DATA_PATH` trỏ vào `espeak-ng-data` đi cùng executable; không sửa biến
môi trường của máy. Thiếu đường dẫn này đã tái hiện lỗi native `0xC0000005`
ở `--version`/`--help` khi gọi Python trực tiếp mà chưa chạy hook Conda.
Đặt đúng đường dẫn đã kiểm tra các lệnh đó trả mã 0. CLI `--path` riêng
không sửa được nhánh in phiên bản của eSpeak 1.52.0.

eSpeak nhận văn bản UTF-8 và ghi WAV trước khi phát, theo
[giao diện eSpeak NG chính thức](https://github.com/espeak-ng/espeak-ng/blob/master/src/espeak-ng.1.ronn).
Đây là giọng hệ thống cho demo, **không phải giọng Nobita**. Không thay đổi
model ASR/MT hay nội dung safety WAV đã duyệt.

Để nghe gTTS đọc câu mới qua tai nghe trên PC, cài backend online tùy chọn
rồi chạy bài kiểm tra TTS độc lập:

```powershell
$Repo = "D:\code\.vscode\OneVoice\onevoice-edge"
$OneVoicePython = "D:\MINICONDA\envs\onevoice\python.exe"
$Bundle = "D:\OneVoiceDesktop\onevoice-v2-rc1\vi2en"
$OneVoicePython -m pip install -r "$Repo\requirements-online-tts.txt"
$OneVoicePython -u "$Repo\scripts\check_desktop_audio.py" `
  --config "$Bundle\runtime_config.yaml" --direction vi2en `
  --tts-backend gtts --allow-online-tts --play --repeats 3 `
  --output-device "Headphones Realtek WASAPI" `
  --report-dir "$Repo\reports\desktop_tts_gtts_online"
```

Bài kiểm tra này không bật mic và không chạy ASR/MT. Mỗi câu sẽ được gửi tới
Google Translate TTS để tạo giọng. Dùng `--tts-backend gtts` đồng thời với
`--allow-online-tts` là lựa chọn tường minh; chế độ edge/offline không tự gọi
mạng. Các gói online nằm riêng trong `requirements-online-tts.txt`.

**Chất lượng nghe tiếng Việt chưa được chấp nhận:** người dùng nghe được
âm thanh từ GZUT-MUSIC nhưng báo khó hiểu ở bài thử ba câu eSpeak. Không dùng
PASS kỹ thuật (WAV không im lặng, API phát xong) làm bằng chứng giọng rõ hoặc
chất lượng TTS đạt production. `intelligibility_passed: null` trong smoke
report nghĩa là script không tự đánh giá khả năng nghe hiểu; cần kiểm thử
người nghe riêng trước khi nghiệm thu luồng live Bluetooth.

Sau khi dừng phiên live cũ bằng `Ctrl+C`, kiểm tra thiết bị:

```powershell
conda activate onevoice
$Repo = "D:\code\.vscode\OneVoice\onevoice-edge"
$OneVoicePython = "D:\MINICONDA\envs\onevoice\python.exe"
& $OneVoicePython -m sounddevice
```

Chọn theo tên tai nghe và host API, ví dụ `GZUT-MUSIC MME`. Nếu dùng index
như `3`, cần kiểm tra lại vì index có thể thay đổi sau khi kết nối lại.
Khi chỉ định tên mà tai nghe không có/không duy nhất, runtime báo lỗi,
không tự chuyển về loa mặc định. Chạy kiểm tra 5 câu tiếng Việt liên tiếp:

```powershell
Set-Location "D:\OneVoiceDesktop\onevoice-v2-rc1\en2vi"
& $OneVoicePython "$Repo\scripts\check_desktop_audio.py" --config runtime_config.yaml --direction en2vi --repeats 5 --play --output-device "GZUT-MUSIC MME" --report-dir reports\desktop_audio_check_en2vi
```

Script này **không thu mic và không chạy ASR/MT**. Nếu không dùng `--play`,
nó chỉ kiểm tra tạo WAV, không được dùng làm bằng chứng đã phát ra tai nghe.
Report `desktop_audio_check.json` phân biệt hai trường hợp. Khi có `--play`,
`[Playback] Started/Finished` ghi tên thiết bị và chờ phát xong; lỗi thiết bị
hoặc underrun làm kiểm tra FAIL thay vì bị bỏ qua. Cách chọn thiết bị và
chờ phát được mô tả trong
[API sounddevice](https://python-sounddevice.readthedocs.io/en/0.5.3/api/convenience-functions.html).

Sau khi nghe đủ 5 câu, mới kiểm tra chuỗi live đầy đủ:

```powershell
$env:PYTHONPATH = "$Repo\src"
$env:PYTHONIOENCODING = "utf-8"
& $OneVoicePython -u "$Repo\src\pipeline.py" --config runtime_config.yaml --direction en2vi --profile edge --offline --vad-energy-threshold 0.005 --output-device "GZUT-MUSIC MME" --report-dir reports\live_mic_en2vi
```

Nói ít nhất 3 câu tiếng Anh khác nhau, nghỉ khoảng một giây giữa các câu,
nghe và đối chiếu đầu ra tiếng Việt. Sau `Ctrl+C`, giữ
`playback_events.json`, `runtime_summary.json` và SRT. Với VI→EN, chạy
từ bundle `vi2en`, dùng `--direction vi2en` và nói tiếng Việt.

Log phát xong chỉ chứng minh API output đã hoàn tất, không chứng minh người
nghe thực sự nghe đúng; report để `listener_confirmation: not_recorded`.
Cần xác nhận của người dùng hoặc video demo cho tuyên bố “mic laptop →
tai nghe Bluetooth”. Các mốc `commit→audio` hiện tại là thời điểm buffer
đã được tạo, **không phải latency âm thanh tới tai người nghe qua Bluetooth**.
Benchmark model, replay WAV và soak không thay thế bước xác nhận này.

Luồng normal mặc định chờ endpoint rồi dịch **nguyên câu**, vì dịch từng
stable prefix ngắn riêng lẻ có thể làm sai ngữ cảnh hoặc lặp ý. ASR vẫn nhận
frame và cập nhật hypothesis trong khi nói. Safety fast path vẫn được phép
commit sớm sau xác nhận; không bị buộc chờ endpoint bởi policy normal.
Report ghi `normal_commit_policy: endpoint`. Đặt
`pipeline.normal_commit_policy: progressive` chỉ để thử nghiệm thuật toán
cũ, không dùng kết quả đó làm tuyên bố chất lượng của bản desktop hiện tại.
Thay đổi policy/backend cần đo lại latency của bản đang chạy, không gán
các số đo cũ cho bản mới.

## Mic laptop → tai nghe Realtek (VI→EN)

Ngày 28/09/2026, người dùng xác nhận nghe rõ một WAV tiếng Việt có sẵn qua
`Headphones (Realtek(R) Audio)` với Windows WASAPI, resample đúng từ 24 kHz
sang 48 kHz. Độ dài câu giữ nguyên 2,952 giây. Đây là xác nhận cho **một WAV
và một đường phát**, không phải nghiệm thu TTS sinh câu mới, mic live hay
Bluetooth. Các lượt phát nhầm vào loa không được dùng làm bằng chứng tai nghe.

Bài nghe tiếp theo gồm ba câu tiếng Anh sinh bằng eSpeak qua cùng tai nghe:
API phát đủ ba câu nhưng người dùng vẫn báo giọng khó hiểu. Vì vậy eSpeak
tiếng Anh chưa được nghiệm thu. Windows `auto` nay ưu tiên giọng SAPI có sẵn;
gTTS có thể tạo câu mới khi người dùng bật tùy chọn online. WAV safety tiếng
Anh tạo sẵn bằng gTTS, nhưng phát WAV đó offline không đồng nghĩa việc tạo
câu dịch mới bằng gTTS cũng chạy offline.

Runtime phát theo tần số mặc định mà thiết bị khai báo và resample waveform
tương ứng bằng [SciPy resample_poly](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html).
Không chỉ đổi nhãn sample rate: cách đó sẽ làm sai tốc độ/độ cao giọng.
`playback_events.json` ghi tần số nguồn/phát, số mẫu, host API và độ dài
nguồn/phát để kiểm tra. Sample rate nguồn của ASR/TTS không bị thay đổi.

Chọn mic và tai nghe riêng bằng tên + host API. `--input-device` tránh việc
Windows đang mặc định chọn mic tai nghe thay cho mic laptop. Thiết bị không
có/không duy nhất hoặc định dạng không hỗ trợ thì báo lỗi; không tự đổi mic
hay chuyển ra loa. Tùy chọn không sửa thiết lập âm thanh Windows hoặc config
của bundle.

Sau khi giọng tiếng Anh trong kiểm tra TTS đã nghe rõ, chạy toàn bộ luồng
VI→EN. Block dưới đây dùng array để tránh lỗi xuống dòng PowerShell:

```powershell
conda activate onevoice
$Repo = "D:\code\.vscode\OneVoice\onevoice-edge"
$OneVoicePython = "D:\MINICONDA\envs\onevoice\python.exe"
$env:PYTHONPATH = "$Repo\src"
$env:PYTHONIOENCODING = "utf-8"
Set-Location "D:\OneVoiceDesktop\onevoice-v2-rc1\vi2en"
$oneVoiceLiveArgs = @(
  "-u", "$Repo\src\pipeline.py",
  "--config", "runtime_config.yaml",
  "--direction", "vi2en",
  "--profile", "edge", "--offline",
  "--input-device", "Microphone Array Realtek MME",
  "--output-device", "Headphones Realtek WASAPI",
  "--vad-energy-threshold", "0.005",
  "--report-dir", "$Repo\reports\live_mic_vi2en_realtek"
)
& $OneVoicePython @oneVoiceLiveArgs
```

Chờ `OneVoice V2 LIVE` và log `Input=Microphone Array` rồi nói rõ một câu
tiếng Việt, nghỉ một giây để kết thúc câu. Tai nghe phải phát tiếng Anh.
Thử ít nhất ba câu khác nhau rồi `Ctrl+C`, giữ SRT, `runtime_summary.json`
và `playback_events.json`. Ngưỡng mic `0.005` vẫn là mức thử cần kiểm tra.
Nếu chuyển sang loa laptop, phải đổi `--output-device` sang loa đã kiểm tra;
mic có thể thu lại giọng máy, vì chưa có nghiệm thu chống vọng cho chế độ đó.

## Measure the actual PC (P5)

### Đo lại sau các bản sửa desktop

Sau khi đổi normal commit policy sang endpoint hoặc đổi tiếng Anh từ eSpeak
sang SAPI, số liệu P5 cũ chỉ mô tả phiên bản trước đó. Chạy lại bản đang dùng
bằng bộ benchmark đầy đủ dưới đây. Script gọi giọng Windows SAPI cho VI→EN
ở backend `auto`, eSpeak tiếng Việt cho EN→VI và WAV duyệt sẵn cho safety.

```powershell
$Repo = "D:\code\.vscode\OneVoice\onevoice-edge"
& "$Repo\scripts\run_desktop_benchmarks.ps1"
```

Mặc định không giới hạn số mẫu: 1.958 noisy test VI và 2.546 noisy test EN
trong manifest local, cộng toàn bộ 126 mục safety canonical mỗi chiều.
Mỗi WAV đi qua 32 ms streaming → VAD/ASR → context/MT → TTS. Script tiếp tục
qua các lỗi để thống kê đầy đủ, giữ từng kết quả dưới `cases/`, ghi
`runtime.log`, `progress.json`, `summary.json` và hash code/config/model/data.
Độ trễ tổng hợp TTS và commit→audio được đo lại với engine thực tế.

Sau streaming full, bộ chạy đo TTS trên toàn bộ 979 cặp trong `test.csv`
và P5 clean/safety lặp 5 lần mỗi chiều. Benchmark TTS dùng đúng **ngôn ngữ
đầu ra**: EN cho VI→EN, VI cho EN→VI. Script cũ trước bản sửa chọn nhầm cột
nguồn; báo cáo TTS đó không chứng minh đã sinh đúng ngôn ngữ đầu ra.

Output mặc định: `reports/pc_pipeline_afterfix_v2`. Có thể chạy lại để resume
các case đã lưu cùng code, backend và dữ liệu. Khi thay đổi chúng, dùng
`-OutputRoot` mới; không ghép các latency của hai runtime vào cùng báo cáo.
Case lỗi vẫn nằm trong mẫu số chất lượng; p50/p95 audio chỉ tính các lượt
qua invariant chức năng. Ca đã tạo audio nhưng bị rule/route đánh FAIL cũng
không góp latency vào các percentile này. Reference critical-field validator
so với văn bản gốc phát hiện
cả mất thuật ngữ/phủ định do ASR, thay vì chỉ đối chiếu với bản nhận dạng.

Bộ full dùng replay tăng tốc (`realtime=false`) để kiểm tra toàn bộ corpus.
Đây là thời gian tạo buffer audio trên PC; chưa đo thời gian âm thanh qua
driver/tai nghe và chưa kiểm tra mic thực. Các WAV clean đầy đủ chưa có ở
local, nên không ghi nhận full clean test từ bộ chạy này. Có thể kiểm tra
kế hoạch mà chưa nạp model với `-PrepareOnly`; `-SmokeTestCases 3` chỉ là
smoke và được ghi `full_selected_corpus=false`.

Khi cả hai chiều đã hoàn tất streaming, TTS và P5, tạo báo cáo GitHub từ
chính các số liệu vừa đo:

```powershell
$Repo = "D:\code\.vscode\OneVoice\onevoice-edge"
$OneVoicePython = "D:\MINICONDA\envs\onevoice\python.exe"
Set-Location $Repo
& $OneVoicePython scripts/build_desktop_benchmark_report.py `
  --input-root reports/pc_pipeline_afterfix_v2 `
  --output-dir docs/desktop_runtime_rebenchmark
```

Lệnh tạo `summary.json`, `report.md`, `report.html` và `overview.svg`, từ chối
ghép báo cáo chưa chạy đủ hoặc TTS sai ngôn ngữ. Các ca FAIL vẫn được công bố;
hoàn tất phép đo không có nghĩa mọi ca đều PASS. Không sửa runtime giữa lượt
đo để làm các ca lỗi biến mất khỏi mẫu số.

### P5 với input cố định

P5 measures the **whole chain**—audio, ASR, translation, safety/context, TTS
and generated audio—not only a model benchmark. It should be run after the
offline smokes, from each bundle directory.

```powershell
$Repo = "D:\code\.vscode\OneVoice\onevoice-edge"
Set-Location "D:\OneVoiceDesktop\onevoice-v2-rc1\vi2en"
python "$Repo\scripts\profile_release_runtime.py" `
  --config runtime_config.yaml --direction vi2en --profile edge `
  --hardware-profile desktop_tuf_f15_rtx2050 --repeats 5 `
  --normal-input artifacts\safety_audio\SAFE2_0001_en2vi.wav `
  --safety-input artifacts\safety_audio\SAFE2_0001_en2vi.wav `
  --report-dir reports\p5_vi2en
```

For a meaningful normal-route result, replace `--normal-input` with a local
non-safety WAV. Do not use a safety WAV for both routes in the final report.
The output records the CPU/RAM target source, p50/p95 timing and any blocker;
it does not silently relax a budget.

## CV-safe project wording

> Built OneVoice V2, an offline Vietnamese–English speech-to-speech prototype
> for industrial communication. Integrated streaming ASR, domain-aware MT,
> deterministic approved-safety audio routing, hash-locked local artifacts and
> no-network release verification; evaluated component quality and a 30-minute
> streaming soak before packaging a Windows desktop release candidate.

Keep the accompanying report links and the limitations above. Do not claim
real-site validation, Android deployment, certified safety operation, or
sub-second desktop latency until a measurement on the target PC proves it.

## Future roadmap

The Android/Snapdragon/headset plan remains at
[Android Snapdragon execution gate](ANDROID_SNAPDRAGON_EXECUTION.md). It starts
only after this desktop candidate is packaged and demonstrated successfully.
