# Các vấn đề còn lại sau benchmark desktop

Phạm vi: runtime `a34118a`, Windows PC offline, replay WAV tăng tốc. Báo cáo này
phân biệt lỗi có bằng chứng với giả thuyết cần kiểm tra; không khẳng định mọi
cảnh báo tự động đều là lỗi ngữ nghĩa. Số đo và ví dụ nằm trong
[summary.json](summary.json) và [báo cáo đầy đủ](report.md).

## Kết luận nghiệm thu

Hoàn tất benchmark không có nghĩa toàn bộ pipeline đã đạt chất lượng sản phẩm.
TTS tạo được WAV và P5 đạt budget cũng không thay thế kiểm tra nội dung, nghe
hiểu, mic live hoặc truyền âm thanh tới tai nghe. Các lỗi dưới đây phải được
xử lý trước khi tuyên bố dùng cho tình huống an toàn thực tế.

Lượt đo có tổng 4.756 case, 183 FAIL streaming. VI safety có 126 văn bản
nguồn riêng; EN safety có 126 entry nhưng chỉ 33 văn bản nguồn và 78 hash
WAV riêng. Đây không phải 126 câu/người nói độc lập ở chiều EN.

## VI → EN: baseline ASR vẫn là hạn chế chất lượng

Trong 126 WAV safety canonical, 18 lượt không qua gate chức năng. Đối chiếu
trace của từng lượt cho thấy 12 lượt có lỗi nhận dạng, 5 lượt bị chuẩn hóa
thuật ngữ làm mất safety match và 1 lượt phát lặp. Đây là phân loại thủ công
theo nguyên nhân quan sát được, không phải chỉ dựa vào tên exception.

Ví dụ: “máy xúc” thành “máy súp”, “có rò khí” thành “có giò khí”, hoặc
“thành hố đang sạt” thành “thành hố đang xa”. Khi nguồn đã sai, bản dịch có
thể giữ nghĩa của bản nhận dạng sai mà vẫn qua một số rule runtime.
Runtime này dùng **GIPFormer ONNX baseline**, không dùng các checkpoint
fine-tune GIPFormer đã thất bại trước đây.

## Safety matching bị ảnh hưởng bởi thứ tự chuẩn hóa

Các case `vi2en-safety-084`, `085`, `088`, `089`, `090` có cụm “giàn giáo”
được nhận dạng nhưng trace chứa “hệ thống giàn giáo”.
[colloquial_terms.csv](../../data/colloquial_terms.csv) có mapping này;
[_asr_worker](../../src/pipeline.py) gọi `normalize` trước khi phân tích
safety. Các phrase đã duyệt không khớp với bản mở rộng nên luồng đi vào MT
thường hoặc bị validator chặn, thay vì dùng WAV safety đã duyệt.

Hướng sửa cần kiểm thử: giữ nguyên nhận dạng để safety matching và dùng
chuẩn hóa chuyên ngành ở bước thích hợp; không mở rộng fuzzy matching tùy ý
cho câu nguy hiểm. Regression test phải gồm các câu canonical trên, câu gần
giống nhưng không phải safety và trường hợp phủ định.

## Hai safety ID tương đương gây phát lặp

Case `vi2en-safety-065`: trace chốt `SAFE2_0063` ở hypothesis “Dừng xe nâng”,
rồi chốt thêm `SAFE2_0065` ở endpoint “Ê dừng xe nâng”. Kết quả là
“Stop the forklift! Stop the forklift!”. Controller hiện chống lặp theo
ID, trong khi hai ID này là hai biến thể của cùng mệnh lệnh.

Hướng sửa: review nhóm mệnh lệnh tương đương và cơ chế commit trong cùng
utterance; phải có test cho hypothesis đổi ID nhưng giữ nghĩa, lẫn trường
hợp người nói thật sự đổi mệnh lệnh. Không gộp các câu chỉ vì giống từ khóa.

## EN → VI: thuật ngữ MT và lượt bị chặn

Noisy test có 26/2.546 lượt không qua gate; safety có 60/126 lượt không vào
đúng luồng safety đã duyệt. Không được dùng kết quả smoke 20 câu trước đây
để khẳng định mọi câu safety đều hoạt động đúng.

Các case `en2vi-test-noisy-00297/00298` nhận dạng đúng “Check the radiator
before continuing.” nhưng bản dịch là “Kiểm tra sàn làm mát trước khi tiếp
tục.”. Một số case khác dùng “băng làm mát”. Đây là lỗi thuật ngữ có bằng
chứng, không phải lỗi tạo WAV hoặc thiết bị phát.

Một số câu chứa “spoil” hoặc “barricade” không có audio; runtime log ghi
`[Safety Validator] Translation suppressed: missing_term:...`.
`Streaming latency report is empty` ở các lượt đó là triệu chứng sau khi
nội dung bị chặn, không chứng minh TTS bị treo. Ledger giữ output rỗng khi
không có câu nào được phát, nhưng chưa lưu riêng văn bản MT đã bị chặn;
cần bổ sung dấu vết này để review chính xác lỗi model hay false positive.

Hướng sửa: review bản dịch raw, glossary/canonicalization và dữ liệu dev
của các thuật ngữ trên; không bỏ validator để biến FAIL thành PASS.

## Normalizer tiếng Anh nhầm mạo từ thành đơn vị

[normalize_en](../../src/utils/text_normalizer.py) hiện thay mọi từ `a`
thành `amperes`, không yêu cầu có số đo phía trước. Đã tái hiện bằng hàm
thuần, không cần nạp model:

```text
There is a person below! → There is amperes person below!
Do not stand under a suspended load! → Do not stand under amperes suspended load!
The current is 5 A. → The current is 5 amperes.
```

Trace các case `en2vi-safety-055` đến `058` có “amperes suspended load” và
không khớp safety phrase. Đây là lỗi code chuẩn hóa đã được tái hiện. Với
những câu khác như “Do nott the machine”, “S.” thay “Smoke!”, còn cần ghi
raw ASR và so sánh WAV nguyên bản với đoạn VAD để tách lỗi ASR, cắt âm hoặc
chuẩn hóa; chưa có đủ bằng chứng quy tất cả về fine-tune.

Hướng sửa: mở rộng ký hiệu đơn vị khi có ngữ cảnh đo lường được xác nhận,
giữ nguyên mạo từ và tên/mã thiết bị. Regression test phải có cả câu thường,
câu safety và số đo; không thay từ bừa để khớp benchmark.

## Rule bảo toàn trường trọng yếu có false positive

Ví dụ VI → EN: `material store` có thể bị báo thiếu `material storage area`;
`five hundred` có thể bị báo thiếu số `500`. Các cách diễn đạt tương đương
cần được review trước khi bổ sung alias hoặc chuẩn hóa số. Cảnh báo phủ định
phải kiểm tra ngữ cảnh câu hỏi, không chỉ dò từ.

Tỷ lệ câu qua rule reference không phải độ chính xác dịch và không phải
critical-term recall của benchmark ASR. WER văn bản dịch so với một reference
cũng không phải đánh giá nghĩa. Dùng người review và danh sách thuật ngữ đã
duyệt để kiểm chứng; không bỏ các ca khó khỏi mẫu số.

Đặc biệt, EN safety có tỷ lệ qua rule reference 100% dù 60/126 lượt không
qua safety gate. Ví dụ “Smoke!” nhận dạng thành “S.” và dịch “Hàn.” vẫn
không bị rule reference báo lỗi. Rule chưa bao phủ đủ nghĩa của các cảnh báo
này; không dùng tỷ lệ đó làm bằng chứng nội dung safety đúng.

## TTS tiếng Việt: 3 cảnh báo nguy cơ clipping

TTS độc lập tiếng Anh qua 979/979 câu, tiếng Việt qua 976/979. Các prompt
`OV2_002585`, `OV2_007251`, `OV2_007330` có peak gần full-scale
(`0.999908` đến `0.999969`) và bị gate clipping đánh FAIL. Đây là cảnh báo
định lượng, chưa chứng minh người nghe bị méo ở mức nào.

Cần kiểm tra waveform và cấu hình amplitude của giọng eSpeak, nghe đối
chiếu rồi đo lại. Giảm mức phát sau tổng hợp không phục hồi tín hiệu nếu
nó đã bị clip trong engine. Không bỏ ngưỡng kiểm tra để báo 100% PASS.

## Trình tự tiếp theo

1. Giữ nguyên báo cáo này làm bằng chứng của phiên bản đã đo; không ghi đè
   bằng kết quả sau sửa.
2. Thêm regression test từ các mẫu lỗi, rồi sửa thứ tự normalize/safety
   matching, nhầm mạo từ/đơn vị và cơ chế chống commit lặp bằng tập dev.
3. Review thuật ngữ MT và rule validator; không tiếp tục tune trên test.
   Khi cần thay đổi model, dùng dev để chọn cấu hình và thêm holdout mới.
4. Chạy lại regression rồi full benchmark vào thư mục phiên bản mới.
5. Nghe kiểm thử giọng và chạy mic thật → thiết bị đầu ra đã chọn; latency
   buffer trong báo cáo này không phải thời gian âm thanh tới tai nghe.

Chưa thay đổi model, dữ liệu bundle hoặc runtime giữa lượt benchmark này.
README và báo cáo nêu rõ prototype/demo, không tuyên bố production-ready.
