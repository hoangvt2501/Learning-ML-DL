# Ghi chú biên tập

Những chỗ trong giáo trình phát biểu đúng về ý nhưng **rộng hơn mức chứng minh được**, hoặc
có một chi tiết kỹ thuật đáng nói thêm. Nguyên văn giáo trình trong `content/quantization.md`
**được giữ nguyên**; mỗi ghi chú dưới đây được chèn thành một khung riêng ngay cuối mục tương ứng
trên bản web.

Cú pháp: `## <số mục> — <tiêu đề ghi chú>`, rồi phần thân viết bằng Markdown.

## 5.2 — Nói cho chặt: ràng buộc nằm ở trục bị lấy tổng, không phải ở “activation”

Câu “activation **không** được lượng tử theo kênh” đúng với tình huống phổ biến nhất, nhưng phát
biểu chặt hơn là:

> Một hệ số scale **không rút được ra ngoài một phép tích vô hướng duy nhất** nếu nó phụ thuộc
> vào chỉ số đang bị lấy tổng (trục *thu gọn*, reduction axis).

Từ cách phát biểu này rút ra ba hệ quả mà cách nói ngắn che mất:

1. **Chia nhóm thì vẫn đặt được scale trên trục thu gọn.** Nếu ta chấp nhận **tách tổng thành
   từng nhóm** — cộng dồn int32 trong mỗi nhóm, nhân scale của nhóm đó, rồi cộng các nhóm lại —
   thì scale hoàn toàn được phép nằm trên trục $k$. Đây chính là *per-group* ở Mục 5.1 và là cách
   lượng tử 4 bit cho LLM ở Chương 11 vẫn chạy được. Cái giá là một phép nhân scale cho mỗi nhóm
   thay vì mỗi kênh đầu ra — rẻ khi nhóm đủ lớn (32–128 phần tử).

2. **Có ngoại lệ do cấu trúc lớp.** Trong *depthwise convolution*, chỉ số kênh đầu vào không hề bị
   lấy tổng (mỗi kênh đầu ra chỉ đọc đúng một kênh đầu vào), nên scale activation theo kênh rút ra
   ngoài tổng được một cách sạch sẽ. Xem lời giải [Bài 4](bai-tap.html#bai-4).

3. **Cái thực sự không làm được** là có scale activation theo kênh **miễn phí** trong một phép GEMM
   chuẩn với một bộ cộng dồn duy nhất. Và đúng ràng buộc đó — chứ không phải một điều cấm tuyệt
   đối — mới là nguồn gốc bài toán outlier của LLM ở Mục 11.1–11.3.

Cách nhớ gọn: **được phép hay không là do trục, không do tensor.** Trục bị lấy tổng thì scale phải
trả giá; trục không bị lấy tổng (kênh đầu ra của trọng số, token của activation, batch) thì miễn phí.

## 6.4 — Phạm vi áp dụng của `quantize_multiplier` và quy ước làm tròn số âm

Hai chi tiết cần nói rõ trước khi ai đó chép đoạn mã này vào việc thật.

**a) Đoạn mã chỉ đúng với $0 < M < 1$**, đúng như chuỗi docstring của nó ghi. Vòng lặp
`while M < 0.5` chỉ chuẩn hoá *lên*, nên:

| $M$ | Kết quả | Nhận xét |
|---|---|---|
| $0{,}75$ | `M0 = 1610612736`, `shift = 0` | đúng |
| $1{,}0$ | `M0 = 1073741824`, `shift = -1` | tình cờ vẫn đúng, nhờ nhánh `if M0 == (1 << 31)` |
| $2{,}5$ | `M0 = 5368709120` | **tràn int32**, sai âm thầm |

Cài đặt thật (`QuantizeMultiplier` của gemmlowp/TFLite) cho phép `shift` **âm**, tức một phép dịch
*trái*, nên mọi $M > 0$ đều xử lý được. Thêm đúng một vòng lặp là đủ:

```python
while M >= 1.0:
    M /= 2; shift -= 1
```

$M \ge 1$ xảy ra khi $S_w S_x \ge S_y$, tức dải đầu ra hẹp hơn dải tự nhiên của bộ cộng dồn — hiếm,
nhưng gặp thật ở lớp có đầu ra bị cắt mạnh hoặc khi activation để ở 16 bit.

**b) `rounding_right_shift` làm tròn nửa về phía $+\infty$**, vì `>>` của Python là dịch số học
(tương đương lấy sàn). Trong khi đó `RoundingDivideByPOT` của gemmlowp làm tròn nửa **ra xa số 0**.
Hai quy ước chỉ khác nhau ở đúng điểm giữa của giá trị âm:

| `acc` | `acc / 4` | Mã trong bài | gemmlowp |
|---|---|---|---|
| $-6$ | $-1{,}5$ | $-1$ | $-2$ |
| $-2$ | $-0{,}5$ | $0$ | $-1$ |
| $+6$ | $+1{,}5$ | $+2$ | $+2$ |

Chênh lệch tối đa 1 LSB và không ảnh hưởng tới lập luận của chương. Nhưng nếu bạn muốn khớp
**từng bit** với một backend cụ thể thì phải dùng đúng quy ước của backend đó — đây là loại chi
tiết khiến hai cài đặt “cùng công thức” vẫn lệch nhau vài LSB.

## 6.5 — “Trùng khớp từng bit” chứng minh điều gì và không chứng minh điều gì

Kết quả `max |q_y(integer) - q_y(float sim)| = 0 LSB` rất đáng giá, nhưng cần đọc đúng phạm vi.

**Nó chứng minh:** phép suy ra ở Mục 6.1–6.4 là đúng, và cách mô phỏng bằng số thực trong tài liệu
này tái hiện chính xác đường tính toán số nguyên trong tài liệu này — với cấu hình đã nêu (một lớp
Linear $256 \to 64$, trọng số đối xứng, activation uint8, requantization 32 bit).

**Nó không chứng minh:** rằng mô phỏng fake quantization luôn khớp từng bit với *một backend thật*.
Đây là phép tự đối chiếu giữa hai cài đặt của cùng một tác giả, không phải phép đối chiếu với
TFLite, ONNX Runtime hay QNNPACK. Mức khớp với backend thật còn phụ thuộc:

- **quy ước làm tròn** ở mỗi bước (xem ghi chú Mục 6.4);
- **độ chính xác của bước requantization** — có backend dùng 32 bit thay vì 64;
- **cách gộp lớp**, vì gộp hay không gộp sẽ đổi số lần lượng tử trung gian;
- **op không có kernel INT8** sẽ rơi về FP32 và sinh thêm cặp quantize/dequantize.

Phát biểu an toàn: **fake quantization là công cụ dự đoán độ chính xác rất tốt, không phải bảo đảm
về đầu ra từng bit.** Chính vì vậy mà mục 8 trong danh sách kiểm tra ở Mục 12.3 vẫn bắt buộc đo lại
trên thiết bị đích.

## 11.3 — Giá trị $\alpha$ tốt nhất trên chính dữ liệu của thí nghiệm này là 0,60

$\alpha = 0{,}5$ là mặc định của bài báo SmoothQuant và là điểm làm **biên độ** hai vế bằng nhau.
Nhưng biên độ bằng nhau không có nghĩa **sai số** hai vế bằng nhau. Quét $\alpha$ trên chính dữ
liệu của Hình 14 (`python code/sweep_alpha.py`) cho:

| $\alpha$ | 0,00 | 0,30 | 0,50 | **0,60** | 0,70 | 1,00 |
|---|---|---|---|---|---|---|
| Sai số đầu ra | 8,02% | 2,48% | 1,44% | **1,34%** | 1,46% | 3,64% |

Lý do cực tiểu lệch sang phải: ở $\alpha = 0{,}5$, phần sai số do trọng số gây ra là 0,60% còn phần
do activation gây ra là 1,31% — activation vẫn là vế đắt hơn, vì nó chỉ có **một** scale cho cả
tensor trong khi trọng số có **một scale cho mỗi kênh đầu ra**. Đẩy thêm chút độ khó sang phía
trọng số vì thế còn lợi.

Phân tích đầy đủ cùng cách chạy lại nằm ở lời giải [Bài 8](bai-tap.html#bai-8).

## 10.1.1 — Mốc thời gian của thông tin phụ thuộc phiên bản

Mọi phát biểu về API trong Chương 10 gắn với một phiên bản cụ thể và sẽ cũ đi. Đối chiếu lại vào
**28-09-2026**, các phiên bản mà tài liệu nêu vẫn là bản mới nhất:

| Gói | Tài liệu nêu | Bản mới nhất khi đối chiếu |
|---|---|---|
| PyTorch | 2.14 | 2.14.0 |
| torchao | 0.18 | 0.18.0 (phát hành 03-08-2026) |
| NumPy | 2.4 | dòng 2.4 |

Việc TensorFlow Lite được Google đổi thương hiệu thành **LiteRT** (2024) và vẫn giữ API
`tf.lite.TFLiteConverter` cũng khớp với tài liệu hiện hành. Dù vậy, cảnh báo deprecated cụ thể,
tên cấu hình và đường dẫn module trong torchao là thứ đổi nhanh nhất giữa các bản phát hành — khi
đọc lại sau vài tháng, hãy kiểm tra trực tiếp trên tài liệu của PyTorch và torchao thay vì tin
vào đoạn mã ở đây.
