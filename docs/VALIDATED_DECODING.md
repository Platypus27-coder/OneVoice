# Sửa giải mã ASR/MT có kiểm tra nội dung

Đây là bản sửa tiếp theo của [input integrity](PIPELINE_INPUT_INTEGRITY.md).
Giữ nguyên checkpoint SenseVoice, GIPFormer baseline và hai EnViT5 ONNX.
Không tải model mới, không dùng câu reference làm đầu vào suy luận và không
đưa dữ liệu test vào train. Báo cáo full trước đó được giữ nguyên.

## Thuật ngữ MT

EnViT5 vốn chạy beam search với 5 beam nhưng chỉ trả về bản đầu tiên. Runtime
nay nhận tối đa 5 bản dịch model sinh ra, theo thứ tự xếp hạng. Nếu bản đầu
qua validator thì giữ nguyên; nếu không, chọn bản đầu tiên qua toàn bộ rule.
Không có bản nào qua thì giữ lỗi của bản đầu, không chèn thuật ngữ vào câu để
ép PASS. Luồng safety và translation memory không bị thay bằng beam search.

Thử trên 56 câu dev mỗi chiều: toàn bộ 16 câu thuộc bốn nhóm radiator, spoil,
barricade, pedestrian walkway và 40 câu đối chứng chọn bằng hash cố định.
Đầu vào đã bỏ dấu kết câu để mô phỏng transcript ASR. Trước khi sửa validator,
EN→VI có 53/56 bản top-1 qua rule và 55/56 có ít nhất một beam qua; VI→EN
có 55/56 ở cả hai cách. Đây là kết quả rule trên tập dev được chọn, không
phải độ chính xác dịch của toàn bộ dev hoặc test.

Ví dụ dev: `Do not tighten the radiator` sinh top-1 `Không siết sàn làm mát.`
và beam tiếp theo `Không siết két làm mát.`. Bản thứ hai giữ đúng thiết bị và
phủ định; không sửa chuỗi đầu bằng một mapping lấy từ câu test.

API `Translator.translate` vẫn trả top-1 cho benchmark model thành phần.
Pipeline dùng `translate_candidates` và lưu các beam đã xét cùng lỗi validation.
Do đó không gán kết quả sau reranking cho độ chính xác raw của model.
Tham khảo [generation strategies của Transformers 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/generation_strategies).

## SenseVoice: giữ ITN chính, retry tại cuối câu

Đã kiểm tra vocabulary của bundle: 25.055 token khớp hoàn toàn SentencePiece.
Thử trên 412 WAV EN dev: mọi câu ngắn tối đa 5 từ, mọi câu có chữ số và 40
câu đối chứng chọn theo hash; mỗi câu có hai biến thể nhiễu. WER dạng văn bản
với ITN là 0,9397%, không ITN là 9,2796%. Chỉ số này cũng chịu ảnh hưởng
cách viết số, không dùng nó một mình để kết luận mọi âm nhận dạng đều sai.
Không chuyển ASR toàn bộ sang chế độ không ITN.

Khi hypothesis cuối câu chưa được nhận diện là safety, adapter có thể giải mã
lại cùng audio bằng `woitn`. Kết quả chỉ được chọn nếu khớp chính xác câu
đã duyệt hoặc biến thể ngữ pháp giới hạn đã hỗ trợ; không dùng edit distance
để chọn bản retry. Từ chối thay đổi phủ định, hành động đã nhận diện, số,
đơn vị, hướng, mã thiết bị và concept đã nhận diện trong bản chính.
Chính sách này không thay đổi transcript nào trong bộ 412 WAV dev trên.
Không khẳng định nó sẽ không có false positive trên giọng/ngữ cảnh mới.
Mã thiết bị có/không có dấu nối như `XC-03`, `XC03`, `BM1` phải giữ nguyên
token mã, không chấp nhận bằng cách tìm substring. Đây vẫn là rule nhận diện
mã theo mẫu chữ-số, không phải parser mọi định dạng mã thiết bị.

Ví dụ: một WAV cảnh báo cho `5re.` ở ITN và `fire` ở không ITN. Dùng chính
kết quả model thứ hai, không sửa `5re` thành `Fire` bằng từ điển test.
`S.` → `s` và `pling object` → `folding object` không khớp câu đã duyệt nên
không được sửa thành `Smoke`/`Falling object` để tăng điểm.
Prompt của FunASR là [withitn=14, woitn=15](https://github.com/modelscope/FunASR/blob/main/runtime/python/onnxruntime/funasr_onnx/sensevoice_bin.py).

Retry chạy tối đa một lần tại endpoint/file, không trên mọi cửa sổ partial và
không khi bản chính đã khớp safety. Nó có thêm chi phí ASR cho các câu cần
retry; phải đo latency bản mới, không giữ nguyên số đo của bản cũ.
Trace giữ riêng bản chính, bản retry, quyết định chọn và bản dùng downstream.

## Validator chính xác hơn

- `barricade` ở câu mệnh lệnh rào mép/quanh khu vực không bị ép dịch thành
  danh từ `rào chắn`. Không áp dụng tương đương này cho mọi câu chứa từ đó.
- `không ổn định` → `unstable/unstably` giữ nghĩa phủ định. Antonym chỉ bao
  phủ predicate tương ứng, không bao phủ thêm một lệnh `đừng ...` khác.
- So sánh giá trị số bằng token số/Decimal: 500 không khớp 1500; -20 không
  khớp 20; 500.0 có cùng giá trị với 500. Đây chưa phải kiểm tra đầy đủ việc
  ghép từng số với từng đơn vị hoặc mọi số viết bằng chữ.
- Khi nguồn đã có một safety translation cố định được duyệt, đầu ra khác
  bị ghi `unverified_safety_translation`. `Smoke!` → `Hàn.` không còn qua
  rule chỉ vì nguồn không có thuật ngữ hoặc chữ số. Cờ này yêu cầu review,
  không khẳng định mọi cách diễn đạt khác đều sai nghĩa.

Rule PASS vẫn không thay thế review ngữ nghĩa của con người. TTS, mic và
Bluetooth cũng không được nghiệm thu chỉ bằng các rule văn bản này.

## Kiểm chứng và giới hạn

Unit test bổ sung bao phủ chọn beam, giữ top-1 hợp lệ, giữ FAIL nếu không có
beam đúng trường, từ chối đổi phủ định/mã/số và cả hai API SenseVoice.
178 unit test của repo đã đạt sau bản sửa. WAV regression và các probe được
lưu ở `reports/pc_pipeline_integrity_v4`, `reports/pc_pipeline_integrity_v4_final`
và `reports/pc_runtime_investigation_v4`; không ghi đè báo cáo cũ.

Replay dùng đúng 557 WAV của lần input-integrity trước: toàn bộ ca FAIL của
full benchmark cũ, toàn bộ 126 safety WAV mỗi chiều và 100 ca noisy từng
PASS chọn bằng hash mỗi chiều. Kiểm tra SHA-256 WAV/config trước khi chạy,
chặn socket mạng Python, dùng cùng bundle và không bật mic/loa.

- VI→EN: lỗi gate chức năng từ 97 ở full baseline cũ, xuống 70 ở bản
  input-integrity, còn 35 ở bản validated-decoding; khôi phục thêm 35 ca.
  Safety giữ mức 114/126 PASS, noisy được chọn đạt 156/179.
- EN→VI: lỗi từ 86, xuống 62, còn 47; khôi phục thêm 15 ca. Safety từ
  90/126 lên 94/126, noisy được chọn đạt 111/126.
- Không có ca từ PASS thành FAIL so với bản input-integrity trên bộ replay
  này. Hai bản sửa cộng lại khôi phục 101/183 ca lỗi cũ; bản sửa lần này
  riêng khôi phục thêm 50 ca. Không có nghĩa full test mới đã đạt tỷ lệ đó.

V4 dùng runtime-code SHA-256
`cd1bb12eae2d203ba250cfbf64eebcfe40e3063900fafb85af8ab9cb193db273`.
Sau đó chỉ siết guard mã thiết bị cho retry EN: kiểm tra lại toàn bộ 163
retry trace lưu từ V4, không thay đổi quyết định nào. EN được replay thêm
riêng ở `pc_pipeline_integrity_v4_final`: vẫn 205/252 PASS, 47 FAIL,
94/126 safety PASS, không có quyết định gate nào thay đổi. Runtime-code
SHA-256 bản guard cuối là
`2429985063ada7c962f6944b935449457b39f11b81c91748bfff5de885a33cb9`.
Đây không phải một tập test mới độc lập. Thời gian đo của replay đồng thời
không dùng để tuyên bố tối ưu latency.

Retry EN được chọn ở 5/163 lần thử, trong đó khôi phục thêm 4 ca safety
`Lock out the energy source before maintenance`. WAV `Fire!` cho `fire`
khi giải mã file nguyên trong probe, nhưng đoạn qua VAD/streaming cho bản
retry rỗng và vẫn FAIL. Không gán thành công ở probe cho toàn pipeline.

Không thay GIPFormer greedy baseline: thử trên toàn bộ VI dev có 1.432
WAV nhiễu, modified beam search giảm WER từ 9,9655% xuống 9,6685%, chỉ
0,2970 điểm phần trăm. Recall concept trọng yếu từ 73,3271% lên 73,4201%,
chỉ 0,0929 điểm phần trăm. Đó là so sánh hai decoder trên cùng checkpoint
INT8, không phải benchmark PyTorch hay một model fine-tune mới. Chưa đủ
bằng chứng cải thiện lớn để chuyển cấu hình release hoặc chi thêm thời
gian fine-tune tùy tiện.

82 ca gate FAIL còn lại gồm 35 VI→EN và 47 EN→VI. Safety không vào đúng
route ở 12 và 32 WAV; đây là lỗi cần xử lý tiếp, không nghiệm thu safety
trong vận hành thực tế. Một số ca còn lại bị chặn do thiếu thuật ngữ,
phủ định/hướng hoặc sai trường so với reference. Cần phân loại review
ngữ nghĩa để tách ASR/MT thật sự sai với validator còn quá cứng.

## Latency toàn pipeline sau sửa

Đo ngày 28/09/2026, sau khi các replay/probe đã kết thúc, chạy từng chiều
tuần tự. Cùng hai WAV/chiều, cùng config, endpoint commit, backend TTS và
budget với P5 desktop cũ: 8.192 MB RSS, 3.000 ms normal commit→audio,
300 ms safety commit→audio. Mỗi route lặp 5 lần, chặn mạng Python.

- VI→EN normal commit→audio p95: 825,7 → 830,9 ms; toàn lượt p95:
  1.279,0 → 1.265,4 ms. Gần như tương đương trên mẫu đo này.
- EN→VI normal commit→audio p95: 566,0 → 696,3 ms; toàn lượt p95:
  1.364,1 → 1.566,7 ms. Có tăng độ trễ, không tuyên bố bản sửa nhanh hơn.
  Retry tại endpoint có thêm chi phí ASR; 5 lần lặp khác phiên đo không
  đủ để quy toàn bộ chênh lệch cho một thay đổi riêng lẻ.
- Safety commit→audio p95 bản mới khoảng 2,1 ms VI→EN và 1,4 ms EN→VI.
  Đây chỉ là sau commit tới buffer audio, không phải thời gian người nghe
  đã nghe xong, và không bao gồm nghiệm thu Bluetooth.

Cả hai hướng vẫn qua **budget desktop đã giữ nguyên**. Báo cáo bản mới
nằm ở `reports/pc_pipeline_integrity_v4_final/{direction}/profile/profile.json`;
đối chiếu cũ ở `reports/pc_pipeline_afterfix_v2/{direction}/profile/profile.json`.
Peak RSS là process Python, chưa cộng mọi child process SAPI. Năm lần lặp
trên một câu thường/một câu safety chỉ là profile đối chứng, không đại diện
phân bố latency của toàn bộ corpus hoặc giọng người nói mới.

Các thay đổi này không giải quyết mọi lỗi âm học của baseline GIPFormer hay
SenseVoice. Những câu vẫn nhận sai hoặc không có bản dịch hợp lệ được giữ
trong danh sách FAIL. Chưa có bằng chứng nghiệm thu giọng công trường thực tế.
