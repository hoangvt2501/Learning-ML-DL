# Từ điển thuật ngữ

Cú pháp: `## Nhóm`, rồi `### Tiếng Việt | English` và phần định nghĩa bên dưới.
Mọi định nghĩa đều bám sát cách dùng trong giáo trình.

## Khái niệm nền

### Lượng tử hoá | quantization
Ánh xạ từ một tập giá trị lớn (thường liên tục) vào một tập hữu hạn nhỏ hơn. Trong deep learning, đây là phép **nén có mất mát**: giảm số bit mỗi giá trị để tiết kiệm bộ nhớ, băng thông và năng lượng, đổi lại chấp nhận sai số. Xem Mục 3.1.

### Lượng tử hoá đều | uniform quantization
Dạng lượng tử trong đó các mức đích **cách đều nhau**. Được ưa chuộng vì cho phép tính toán trực tiếp bằng phép toán số nguyên (Chương 6). Đối lập là các dạng không đều như NF4.

### Bước nhảy, thang | scale, step size — $S$
Khoảng cách giữa hai mức lượng tử liền kề, tính bằng đơn vị số thực. Luôn dương. $S = (\beta-\alpha)/(q_{max}-q_{min})$.

### Điểm không | zero-point — $Z$
Số nguyên ứng với số thực 0. Bắt buộc là **số nguyên** để $x = 0$ được biểu diễn chính xác tuyệt đối (Mục 3.4).

### Dải cắt | clipping range — $[\alpha, \beta]$
Đoạn số thực được ánh xạ vào miền số nguyên. Giá trị ngoài đoạn này bị cắt về mép. Dải luôn được mở rộng để chứa số 0.

### Độ rộng bit | bit-width — $b$
Số bit của kiểu số nguyên đích. Ký hiệu W8A8 nghĩa là trọng số 8 bit và activation 8 bit; W4A16 là trọng số 4 bit, activation 16 bit.

### Lượng tử–giải lượng tử | fake quantization, quantize–dequantize
Hàm hợp $x \mapsto \hat{x} = S(\text{quantize}(x) - Z)$. Kết quả vẫn là số thực nhưng chỉ nhận giá trị trên lưới lượng tử. Là công cụ trung tâm để mô phỏng và huấn luyện (Mục 7.2).

### Lượng tử đối xứng | symmetric quantization
Ép $Z = 0$ và dùng dải đối xứng quanh 0. Thường dùng miền hạn chế $[-(2^{b-1}-1), 2^{b-1}-1]$ để lưới đối xứng hoàn hảo. Hợp với trọng số, và rẻ hơn khi tính (Mục 6.2).

### Lượng tử bất đối xứng | asymmetric quantization, affine quantization
Dùng cả $S$ và $Z$ tuỳ ý. Hợp với dữ liệu lệch một phía như activation sau ReLU.

## Sai số và đo lường

### Sai số làm tròn | rounding error
Sai số khi $x$ nằm **trong** dải. Độ lớn luôn $\le S/2$. Theo mô hình nhiễu đều, $\mathbb{E}[e^2] = S^2/12$.

### Sai số cắt | clipping error
Sai số khi $x$ nằm **ngoài** dải, bằng khoảng cách từ $x$ tới mép dải. Không bị chặn trên.

### Tỉ số tín hiệu trên nhiễu lượng tử | SQNR — signal-to-quantization-noise ratio
$\text{SQNR} = 10\log_{10}\bigl(\mathbb{E}[x^2]/\mathbb{E}[(x-\hat{x})^2]\bigr)$ dB. Quy tắc nhớ: **mỗi bit đáng giá khoảng 6 dB** (Mục 4.2).

### Epsilon máy | machine epsilon — $\varepsilon$
Khoảng cách từ 1 đến số biểu diễn được kế tiếp lớn hơn 1, bằng $2^{-p}$ với $p$ là số bit độ chính xác. Đo **độ chính xác tương đối**.

### Cường độ số học | arithmetic intensity
Số phép toán chia cho số byte đọc ghi. Đại lượng quyết định một phép tính là compute-bound hay memory-bound. Mô hình *roofline* so nó với tỉ số giữa thông lượng tính toán và băng thông bộ nhớ.

### Bị chặn bởi tính toán | compute-bound
Trạng thái mà tốc độ bị giới hạn bởi thông lượng tính toán, ví dụ nhân hai ma trận lớn với batch lớn. Lượng tử giúp nhờ thông lượng số nguyên cao hơn.

### Bị chặn bởi bộ nhớ | memory-bound
Trạng thái mà tốc độ bị giới hạn bởi băng thông bộ nhớ, ví dụ sinh từng token của LLM với batch nhỏ. Lượng tử giúp nhờ giảm số byte phải đọc.

## Độ mịn

### Độ mịn | granularity
Mức độ chia nhỏ tensor để mỗi phần dùng một cặp $(S, Z)$ riêng. Càng mịn thì sai số càng nhỏ nhưng càng nhiều scale phải lưu và xử lý.

### Theo tensor | per-tensor
Một cặp $(S, Z)$ cho cả tensor. Đơn giản nhất; một kênh biên độ lớn ép mọi kênh khác dùng bước nhảy thô.

### Theo kênh | per-channel, per-axis
Mỗi kênh **đầu ra** của trọng số có $S$ riêng. Mặc định cho Conv và Linear trong TFLite và PyTorch hiện nay. Bắt buộc theo trục đầu ra vì đó là trục không bị lấy tổng (Mục 5.2).

### Theo nhóm, theo khối | per-group, per-block
Chia mỗi hàng trọng số thành các nhóm liên tiếp 32, 64 hoặc 128 phần tử, mỗi nhóm một scale. Chuẩn cho lượng tử 4 bit của LLM.

### Theo token | per-token
Mỗi hàng (token) của ma trận activation có scale riêng. Hợp lệ vì chỉ số token không bị lấy tổng trong phép nhân ma trận.

## Đường tính toán số nguyên

### Bộ cộng dồn | accumulator
Thanh ghi chứa tổng tích luỹ của phép nhân ma trận. Với đầu vào int8 phải là **int32**: thí nghiệm ở Mục 6.5 đo được $\max|\text{acc}| = 66\,876$, đã vượt xa giới hạn int16.

### Tái lượng tử | requantization
Bước đưa kết quả int32 của bộ cộng dồn về lại miền int8/uint8 của lớp sau, bằng cách nhân với $M = S_w S_x / S_y$ rồi cộng $Z_y$ và cắt.

### Nhân dấu chấm tĩnh | fixed-point multiplier — $M_0$, $n$
Cách nhân với số thực $M$ mà không dùng số thực: viết $M = 2^{-n} M_0$ với $M_0 \in [0{,}5;1)$, lưu $M_0^{\text{int}} = \lfloor M_0 \cdot 2^{31}\rceil$, rồi nhân int64 và dịch phải có làm tròn.

### Gộp BatchNorm | BN folding
Gấp phép biến đổi affine cố định của BatchNorm vào trọng số và bias của tích chập đứng trước. **Phải làm trước khi lượng tử**, vì trên phần cứng BN không tồn tại như một lớp riêng (Mục 7.3).

### Gộp lớp | layer fusion
Gộp chuỗi như Conv → BN → ReLU thành một phép toán. Bớt các cặp quantize/dequantize trung gian và tránh ghi activation trung gian ra bộ nhớ.

## Quy trình

### Lượng tử sau huấn luyện | PTQ — post-training quantization
Lượng tử mô hình đã huấn luyện xong, không cập nhật trọng số bằng gradient. Nhanh (vài phút), không cần nhãn. Luôn nên thử trước.

### Huấn luyện có nhận thức lượng tử | QAT — quantization-aware training
Fine-tune mô hình với các nút fake-quant trong đồ thị, để mạng học cách bù lại sai số lượng tử. Đáng dùng khi số bit thấp hoặc mô hình nhỏ, nhạy cảm.

### Hiệu chỉnh | calibration
Cho một tập dữ liệu nhỏ chạy qua mô hình để thống kê phân phối activation, rồi chọn $[\alpha,\beta]$ cho mỗi điểm lượng tử. Thường cần 100–1000 mẫu đại diện.

### Tập dữ liệu đại diện | representative dataset
Tập mẫu nhỏ dùng cho calibration. Phải cùng phân phối và cùng tiền xử lý với lúc triển khai; nếu không, dải chọn ra sẽ sai.

### Lượng tử tĩnh | static quantization
Dải activation được cố định trước nhờ calibration. Nhanh nhất và là điều kiện để chạy thuần số nguyên.

### Lượng tử động | dynamic quantization
Trọng số lượng tử sẵn; activation được lượng tử **lúc chạy** từ min/max của chính tensor đó. Không cần dữ liệu calibration. Trong `torch.ao` chỉ hỗ trợ Linear và các lớp hồi quy, **không** hỗ trợ Conv2d.

### Chỉ lượng tử trọng số | weight-only quantization
Chỉ nén trọng số; lúc chạy giải lượng tử về FP16/FP32 rồi tính. Lợi về bộ nhớ và băng thông, không lợi về số phép tính.

### Ước lượng thẳng | STE — straight-through estimator
Thủ thuật cho lượt ngược: coi phép làm tròn như hàm đồng nhất bên trong dải, đạo hàm 0 ở vùng bị cắt. Không có nó thì gradient về trọng số luôn bằng 0.

### Lượng tử với bước nhảy học được | LSQ — learned step size quantization
Coi $S$ là tham số huấn luyện được thay vì cố định bằng EMA. Cần hệ số chỉnh gradient $g = 1/\sqrt{N Q_P}$ để ổn định.

### Cân bằng giữa các lớp | CLE — cross-layer equalization
Dựa vào $\text{ReLU}(sz) = s\,\text{ReLU}(z)$, chia kênh $i$ của lớp trước cho $s_i$ và nhân kênh đầu vào $i$ của lớp sau với $s_i$ để dải giữa các kênh đồng đều hơn. Không cần dữ liệu.

### Hiệu chỉnh bias | bias correction
Bù lại độ lệch trung bình của đầu ra do sai số lượng tử trọng số gây ra: $b \leftarrow b - \varepsilon\,\mathbb{E}[x]$.

### Làm tròn thích nghi | AdaRound
Học quyết định làm tròn lên hay xuống cho **từng trọng số** sao cho tối thiểu sai số **đầu ra của lớp**, thay vì luôn làm tròn đến số gần nhất.

### Làm tròn đến số gần nhất | RTN — round-to-nearest
Cách lượng tử cơ bản nhất: mỗi trọng số làm tròn độc lập về mức gần nhất. Mốc so sánh của GPTQ và AWQ.

## Kiểu dữ liệu

### Bù hai | two's complement
Cách biểu diễn số nguyên có dấu với $b$ bit, cho miền $[-2^{b-1}, 2^{b-1}-1]$.

### Số dưới chuẩn | subnormal, denormal
Số dấu phẩy động có exponent bằng 0, giá trị $(-1)^s (0.m)_2 \times 2^{1-\text{bias}}$, cho phép biểu diễn các giá trị rất gần 0.

### Bit ẩn | implicit bit, hidden bit
Chữ số 1 đứng trước dấu chấm trong số chuẩn hoá. Không được lưu nhưng luôn có mặt, nên mantissa $p$ bit cho độ chính xác $p+1$ bit.

### BF16 | bfloat16
16 bit với 8 bit exponent và 7 bit mantissa. Cùng dải với FP32 nên ít tràn số — lý do nó được ưa dùng để huấn luyện.

### FP8 E4M3 / E5M2 | FP8 formats
Hai biến thể 8 bit. E4M3 chính xác hơn, thường cho trọng số và activation; E5M2 dải rộng hơn, thường cho gradient. Bản OCP của E4M3 bỏ vô cực và chỉ giữ một mã NaN để nâng giá trị lớn nhất lên 448.

### NF4 | NormalFloat 4-bit
Kiểu dữ liệu 4 bit **không đều**, đặt 16 mức tại các phân vị của phân phối chuẩn. Là định dạng **lưu trữ**, phải giải lượng tử về BF16 trước khi tính.

### Lượng tử hai lần | double quantization
Lượng tử tiếp chính các hằng số absmax của từng khối, để giảm chi phí phụ trội. QLoRA đưa chi phí từ 0,5 xuống còn ≈ 0,127 bit mỗi tham số.

### Microscaling | MX formats
Chuẩn OCP: mỗi khối 32 phần tử dùng chung một scale là luỹ thừa của 2, lưu 8 bit (E8M0). Gồm MXFP8, MXFP6, MXFP4, MXINT8. Thực chất là per-group quantization được chuẩn hoá ở mức phần cứng.

## Riêng cho LLM

### Giá trị ngoại lai | outlier
Trong LLM đủ lớn (từ khoảng 6,7 tỉ tham số), một số ít chiều ẩn **cố định** có biên độ lớn gấp hàng chục lần phần còn lại, lặp lại ở hầu hết token và hầu hết lớp. Vì activation không được lượng tử theo kênh, chúng phá scale của mọi kênh khác.

### Phân rã hỗn hợp độ chính xác | mixed-precision decomposition
Cách của LLM.int8(): tách các chiều có $|x| > 6$ ra nhân bằng FP16, phần còn lại nhân bằng INT8, rồi cộng hai kết quả.

### SmoothQuant
Biến đổi tương đương toán học $Y = (X\,\text{diag}(s)^{-1})(W\,\text{diag}(s))^\top$ để chuyển bớt độ khó từ activation sang trọng số. Tham số $\alpha$ điều chỉnh mức chuyển giao; 0,5 là mặc định.

### GPTQ
Lượng tử từng cột trọng số, sau mỗi cột thì **bù sai số vào các cột chưa lượng tử** theo công thức kế thừa từ Optimal Brain Surgeon. Tối thiểu sai số **đầu ra của lớp** chứ không phải sai số trọng số.

### AWQ | activation-aware weight quantization
Xác định kênh trọng số "quan trọng" bằng **biên độ activation** đi vào kênh đó, rồi nhân chúng lên trước khi lượng tử và chia lại ở activation. Không cần lan truyền ngược.

### QLoRA
Giữ mô hình gốc đông cứng ở NF4 và chỉ huấn luyện các adapter LoRA ở BF16, cho phép fine-tune mô hình hàng chục tỉ tham số trên một GPU.

### Bộ nhớ đệm KV | KV cache
Các vector key và value của mọi token trước đó, được lưu lại khi sinh văn bản. Dung lượng tỉ lệ với số token và batch, nên với ngữ cảnh dài nó có thể vượt cả dung lượng trọng số (Mục 11.7).

### GQA | grouped-query attention
Kiến trúc cho nhiều query head dùng chung một KV head, giảm KV cache theo đúng tỉ lệ nhóm. Cách cắt KV cache ngay từ kiến trúc, trước khi cần đến lượng tử.

### GGUF
Định dạng file của llama.cpp, với nhiều mức lượng tử (`Q4_0`, `Q4_K_M`, `Q5_K_M`, `Q8_0`…). Ký hiệu `Q4_K` tốn 4,5 bit thực tế mỗi trọng số.

## Công cụ

### Eager mode | torch.ao.quantization eager mode
API "cổ điển" của PyTorch, đòi hỏi sửa mã mô hình: thêm `QuantStub`/`DeQuantStub`, mỗi ReLU là một module riêng, phép cộng skip-connection dùng `FloatFunctional`. Đã deprecated ở PyTorch 2.14 nhưng vẫn hữu ích để học cơ chế.

### PT2E | PyTorch 2 Export quantization
Luồng lượng tử dựa trên `torch.export`: xuất mô hình thành đồ thị nên **không cần sửa mã mô hình**. Nay nằm trong torchao.

### Quantizer
Thành phần quyết định op nào được lượng tử và theo cấu hình nào, cho khớp với kernel mà backend đó có. Mỗi phần cứng một quantizer: `X86InductorQuantizer`, `ArmInductorQuantizer`, các quantizer của ExecuTorch…

### LiteRT
Tên mới (2024) của TensorFlow Lite. API chuyển đổi `tf.lite.TFLiteConverter` vẫn giữ nguyên; trình thông dịch cài riêng qua gói `ai-edge-litert`.
