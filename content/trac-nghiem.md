# Ngân hàng câu hỏi tự kiểm tra

Cú pháp: `## Chương N` mở một nhóm, `### …` là câu hỏi, `- [x]` đánh dấu đáp án đúng,
dòng `>` là phần giải thích hiện ra sau khi trả lời.

## Chương 1

### Theo Hình 1, đọc 64 bit từ DRAM tốn năng lượng gấp khoảng bao nhiêu lần một phép nhân FP32?
- [ ] Xấp xỉ bằng nhau
- [ ] Khoảng 10 lần
- [x] Vài trăm lần
- [ ] DRAM còn rẻ hơn
> Đọc 64 bit từ DRAM tốn 1300–2600 pJ, còn một phép nhân FP32 tốn 3,7 pJ — chênh nhau hàng trăm lần. Đây là lý do giảm số byte phải đọc thường có lợi **hơn** việc làm phép tính rẻ đi.

### Sinh từng token của một LLM với batch nhỏ là bài toán thuộc loại nào?
- [x] Memory-bound — bị chặn bởi băng thông bộ nhớ
- [ ] Compute-bound — bị chặn bởi thông lượng tính toán
- [ ] Bị chặn bởi độ trễ mạng
- [ ] Không bị chặn bởi gì cả
> Mỗi trọng số được đọc một lần nhưng chỉ tham gia rất ít phép tính, nên cường độ số học rất thấp. Vì vậy chỉ cần **nén trọng số** (weight-only) là đã nhanh hơn gần tỉ lệ với mức nén, kể cả khi phép nhân vẫn làm bằng FP16.

### "Lượng tử hoá luôn làm mô hình chạy nhanh hơn." Phát biểu này:
- [ ] Đúng, vì phép toán số nguyên rẻ hơn
- [ ] Đúng, vì mô hình nhẹ hơn
- [x] Sai — chỉ nhanh khi phần cứng và thư viện có kernel số nguyên tương ứng
- [ ] Sai — lượng tử hoá luôn làm chậm đi
> Mục 10.1.5 đo được `Int8WeightOnly` chạy eager **chậm hơn FP32 hơn 2 lần**, vì không có `torch.compile` thì trọng số bị giải lượng tử về số thực ở mỗi lần gọi. Không có kernel phù hợp thì mô hình chỉ nhẹ hơn chứ không nhanh hơn.

## Chương 2

### Phát biểu nào đúng về cách FP32 lưu một số?
- [ ] 16 bit cho phần nguyên, 16 bit cho phần thập phân
- [ ] 1 bit dấu, 8 bit phần nguyên, 23 bit phần thập phân
- [x] 1 bit dấu, 8 bit exponent, 23 bit mantissa; vị trí dấu chấm mã hoá qua exponent
- [ ] 1 bit dấu và 31 bit giá trị tuyệt đối
> Không hề có "bit dành cho phần nguyên". Giá trị là $(-1)^s \times (1.m)_2 \times 2^{e-127}$, nên dấu chấm trôi theo số mũ — đó chính là nghĩa của "dấu phẩy động".

### Vì sao BF16 được ưa dùng để huấn luyện hơn FP16?
- [ ] Vì BF16 có mantissa dài hơn nên chính xác hơn
- [x] Vì BF16 giữ nguyên 8 bit exponent của FP32 nên có cùng dải, ít tràn số
- [ ] Vì BF16 dùng ít bộ nhớ hơn FP16
- [ ] Vì BF16 là số nguyên nên tính nhanh hơn
> Cả hai đều 16 bit. FP16 chỉ có 5 bit exponent, giá trị lớn nhất 65 504 nên rất dễ tràn. BF16 đánh đổi mantissa ngắn hơn (7 bit) để lấy dải rộng bằng FP32.

### Khác biệt cốt lõi giữa lưới của số dấu phẩy động và của số nguyên là gì?
- [ ] Số dấu phẩy động cách đều, số nguyên thì không
- [x] Số dấu phẩy động dày gần 0 và thưa dần ra xa (sai số tương đối gần như không đổi); số nguyên cách đều (sai số tuyệt đối không đổi)
- [ ] Cả hai đều cách đều, chỉ khác số lượng mức
- [ ] Số nguyên dày gần 0 hơn
> Chính vì số nguyên có lưới cách đều mà ta phải chọn khéo **vị trí và độ rộng** của lưới cho khớp dữ liệu — đó là vai trò của $S$ và $Z$.

## Chương 3

### Công thức đúng của bước nhảy $S$ là gì?
- [x] $S = \dfrac{\beta - \alpha}{q_{max} - q_{min}}$
- [ ] $S = \dfrac{q_{max} - q_{min}}{\beta - \alpha}$
- [ ] $S = \dfrac{\beta - \alpha}{2^b}$
- [ ] $S = \max(|\alpha|, |\beta|)$
> $S$ là *bước nhảy giữa hai mức liền kề tính bằng đơn vị số thực*, nên nó phải là (độ rộng dải thực) chia cho (số khoảng trong miền số nguyên).

### Vì sao $Z$ bắt buộc phải là một số nguyên?
- [ ] Để tiết kiệm bộ nhớ
- [x] Để số 0 được biểu diễn **chính xác tuyệt đối**
- [ ] Để phép nhân ma trận nhanh hơn
- [ ] Vì phần cứng không lưu được số thực
> Số 0 xuất hiện dày đặc: zero-padding, đầu ra ReLU, phần tử bị mask trong attention. Nếu 0 lệch thành $\delta \ne 0$ thì sai số **có hệ thống** và cộng dồn qua hàng nghìn phần tử. Ép $Z$ nguyên bảo đảm $x = 0 \Rightarrow q = Z \Rightarrow \hat{x} = 0$.

### Quy ước phổ biến trong TFLite và PyTorch là gì?
- [ ] Trọng số bất đối xứng, activation đối xứng
- [x] Trọng số đối xứng, activation bất đối xứng
- [ ] Cả hai đều đối xứng
- [ ] Cả hai đều bất đối xứng
> Trọng số phân bố quanh 0 nên đối xứng là hợp (và rẻ hơn khi tính, Mục 6.2). Activation sau ReLU lệch hẳn một phía nên bất đối xứng mới không phí mức.

### Dùng lưới đối xứng cho activation sau ReLU ở 3 bit thì lãng phí bao nhiêu mức?
- [ ] Không lãng phí mức nào
- [ ] 1/7 số mức
- [x] 3/7 số mức, dành cho vùng âm không bao giờ xuất hiện
- [ ] Toàn bộ số mức
> Hình 5 vẽ đúng hiện tượng này. Đo trên 200 000 giá trị sau ReLU, MSE của đối xứng so với bất đối xứng là $2{,}67 \times 10^{-2}$ so với $5{,}79 \times 10^{-3}$ ở 4 bit — gấp 4,6 lần.

## Chương 4

### Thêm một bit vào độ rộng bit thì SQNR thay đổi thế nào?
- [ ] Tăng khoảng 3 dB
- [x] Tăng khoảng 6 dB
- [ ] Tăng gấp đôi
- [ ] Không đổi nếu dải giữ nguyên
> Thêm một bit làm $S$ giảm một nửa, công suất nhiễu $S^2/12$ giảm 4 lần, tức $10\log_{10} 4 \approx 6{,}02$ dB.

### Theo mô hình nhiễu đều, kỳ vọng bình phương sai số làm tròn bằng bao nhiêu?
- [ ] $S/2$
- [ ] $S^2/4$
- [x] $S^2/12$
- [ ] $S^2/2$
> Sai số phân phối đều trên $[-S/2, S/2]$, nên $\mathbb{E}[e^2] = \frac{1}{S}\int_{-S/2}^{S/2} e^2\,de = S^2/12$. Điểm mấu chốt: sai số tỉ lệ với **bình phương** bước nhảy.

### Trên $10^6$ trọng số Laplace lượng tử INT4 đối xứng, dùng dải min–max thay vì ngưỡng tối ưu làm MSE tăng bao nhiêu lần?
- [ ] Không đổi — min–max không cắt giá trị nào nên an toàn nhất
- [ ] Khoảng 1,5 lần
- [x] Khoảng 6,4 lần
- [ ] Khoảng 100 lần
> Hình 7: ngưỡng tối ưu $c^* \approx 4{,}83$ cho MSE $= 0{,}0547$; min–max ($c = 15{,}28$) cho MSE $= 0{,}348$. "Không cắt giá trị nào" nghe an toàn nhưng phải trả bằng bước nhảy thô cho toàn bộ dữ liệu.

## Chương 5

### Vì sao scale per-channel của trọng số phải đặt theo trục **đầu ra**?
- [ ] Vì trục đầu ra thường dài hơn
- [x] Vì chỉ số đầu ra không bị lấy tổng, nên scale đưa được ra ngoài tổng và phần còn lại vẫn thuần số nguyên
- [ ] Vì framework quy định như vậy
- [ ] Vì trục đầu vào không có outlier
> Với $y_j = \sum_k W_{jk} x_k$, scale phụ thuộc $j$ thì rút ra ngoài $\sum_k$ được; phụ thuộc $k$ thì nằm kẹt bên trong và phải nhân số thực cho từng phần tử.

### Phát biểu nào **chặt nhất** về việc lượng tử activation theo kênh?
- [ ] Luôn làm được, giống hệt trọng số
- [ ] Không bao giờ làm được, vì activation không có khái niệm kênh đầu ra
- [x] Scale đặt trên **trục thu gọn** thì không rút ra ngoài một phép tích vô hướng duy nhất được — nhưng chia nhóm thì vẫn làm được, và có ngoại lệ do cấu trúc lớp
- [ ] Chỉ làm được khi dùng INT4
> Ràng buộc nằm ở **trục**, không ở tensor. Chấp nhận tách tổng thành từng nhóm (per-group) là đặt được scale trên trục thu gọn — đó là cách lượng tử 4 bit cho LLM vẫn chạy. Depthwise convolution thì kênh đầu vào không hề bị lấy tổng nên miễn phí luôn. Cái không làm được là có scale activation theo kênh **miễn phí** trong một GEMM chuẩn — và đúng ràng buộc đó sinh ra bài toán outlier của LLM (Mục 11.1).

### Vì sao lượng tử per-group (nhóm 32–128) đặt được scale trên trục thu gọn?
- [ ] Vì nhóm nhỏ nên sai số không đáng kể
- [x] Vì ta tách tổng thành từng nhóm: cộng dồn số nguyên trong mỗi nhóm, nhân scale của nhóm đó, rồi mới cộng các nhóm lại
- [ ] Vì phần cứng có lệnh riêng cho per-group
- [ ] Không đặt được — per-group chỉ áp dụng cho trục đầu ra
> Cái giá phải trả là một phép nhân scale cho mỗi nhóm thay vì mỗi kênh đầu ra — chấp nhận được khi nhóm đủ lớn. Đây là lý do "mô hình 4 bit" của LLM tốn 4,125–4,5 bit mỗi trọng số (Mục 11.8) chứ không phải đúng 4.

### Trong thí nghiệm Mục 5.3, granularity ảnh hưởng ra sao ở 8 bit so với 4 bit?
- [ ] Ảnh hưởng mạnh ở cả hai
- [x] Ở 8 bit gần như không đáng kể (99,26% cho cả hai); ở 4 bit per-channel hơn per-tensor hơn 6 điểm phần trăm
- [ ] Ở 8 bit mạnh hơn ở 4 bit
- [ ] Không ảnh hưởng ở cả hai
> Ở 8 bit bước nhảy đã đủ mịn nên chênh lệch dải giữa các kênh không kịp gây hại. Ở 4 bit (91,30% so với 97,59%) thì có.

## Chương 6

### Vì sao bộ cộng dồn trong phép nhân ma trận INT8 phải là int32?
- [ ] Vì phần cứng không có thanh ghi int16
- [x] Vì tổng tích luỹ vượt xa giới hạn int16 — thí nghiệm đo được $\max|\text{acc}| = 66\,876$
- [ ] Vì bias được lưu ở int32
- [ ] Vì scale là số thực
> Giới hạn int16 là 32 767. Cận trên lý thuyết $255 \times 127 \times K = 8{,}3 \times 10^6$ vẫn nằm rất xa giới hạn int32 ($\approx 2{,}1 \times 10^9$), nên int32 là lựa chọn vừa đủ.

### Trọng số lượng tử đối xứng ($Z_w = 0$) mang lại lợi ích tính toán gì?
- [ ] Giảm một nửa dung lượng lưu trữ
- [x] Triệt tiêu hai số hạng $Z_w \sum_k q_{x,k}$ và $K Z_w Z_x$ trong khai triển
- [ ] Bỏ được bộ cộng dồn int32
- [ ] Bỏ được bước requantization
> Số hạng $Z_w \sum_k q_{x,k}$ phụ thuộc đầu vào nên **phải tính lúc chạy**; bỏ được nó là bỏ hẳn một lượt quét toàn bộ ma trận activation. Đây là lý do kỹ thuật khiến trọng số gần như luôn đối xứng.

### Bias được lưu như thế nào trong đường tính toán số nguyên?
- [ ] int8, scale riêng
- [x] int32, scale $S_w S_x$, zero-point 0 — cùng đơn vị với bộ cộng dồn
- [ ] FP32, cộng vào sau khi requantize
- [ ] Không lưu bias, đã gộp vào trọng số
> Nhờ cùng scale với bộ cộng dồn, bias cộng thẳng vào kết quả int32 mà không cần đổi đơn vị. Nó dùng 32 bit vì rất ít tham số nhưng ảnh hưởng trực tiếp lên mọi đầu ra.

### Requantization nhân với hệ số thực $M$ bằng cách nào mà không dùng số thực?
- [ ] Làm tròn $M$ về số nguyên gần nhất
- [x] Viết $M = 2^{-n} M_0$ với $M_0 \in [0{,}5; 1)$, lưu $M_0$ dạng int32 dấu chấm tĩnh, rồi nhân int64 và dịch phải có làm tròn
- [ ] Dùng bảng tra cho mọi giá trị có thể
- [ ] Chuyển tạm sang FP16
> $M_0^{\text{int}} = \lfloor M_0 \cdot 2^{31} \rceil$ và $\text{acc} \cdot M \approx (\text{acc} \cdot M_0^{\text{int}}) \gg (31 + n)$. Toàn bộ đường đi vẫn thuần số nguyên nên chạy được trên NPU/DSP không có FPU.

### Đoạn mã `quantize_multiplier` ở Mục 6.4 dùng được cho hệ số $M$ nào?
- [ ] Mọi $M > 0$
- [x] Chỉ $0 < M \le 1$ — với $M > 1$ thì $M_0$ tràn int32
- [ ] Mọi $M$, kể cả âm
- [ ] Chỉ $M < 0{,}5$
> Vòng lặp `while M < 0.5` chỉ chuẩn hoá **lên**. Cài đặt thật (`QuantizeMultiplier` của gemmlowp/TFLite) cho phép `shift` **âm**, tức một phép dịch trái, nên mọi $M > 0$ đều xử lý được; chỉ cần thêm `while M >= 1: M /= 2; shift -= 1`. Xem [ghi chú Mục 6.4](ghi-chu.html#gc-sec-6-4).

### Hàm dịch phải có làm tròn trong Mục 6.4 xử lý giá trị **âm** theo quy ước nào?
- [ ] Làm tròn nửa về số chẵn, giống NumPy
- [x] Làm tròn nửa về phía $+\infty$ — khác với gemmlowp/TFLite vốn làm tròn nửa **ra xa số 0**
- [ ] Luôn làm tròn xuống, không có ngoại lệ
- [ ] Mọi cài đặt đều giống nhau nên không cần quan tâm
> `>>` của Python là dịch số học, tương đương lấy sàn, nên $(x + 2^{n-1}) \gg n$ làm tròn nửa lên. Hai quy ước chỉ khác nhau ở đúng điểm giữa của giá trị âm: $-6/4$ cho $-1$ ở đây nhưng $-2$ ở gemmlowp. Lệch tối đa 1 LSB — vô hại với lập luận của chương, nhưng phải khớp nếu bạn cần kết quả trùng từng bit với một backend cụ thể.

### Kết quả "trùng khớp từng bit" ở Mục 6.5 chứng minh điều gì?
- [ ] Mô phỏng fake quantization luôn cho đầu ra y hệt mọi backend INT8
- [x] Phép suy ra ở Chương 6 là đúng, và hai cài đặt **trong chính tài liệu này** khớp nhau trên cấu hình đã nêu
- [ ] Lượng tử hoá không gây sai số nào
- [ ] Backend nào cũng dùng cùng một quy ước làm tròn
> Đây là phép **tự đối chiếu** giữa đường số nguyên và bản mô phỏng số thực của cùng một tác giả, không phải phép đối chiếu với TFLite hay ONNX Runtime. Mức khớp với backend thật còn phụ thuộc quy ước làm tròn, độ chính xác bước requantization, cách gộp lớp và op nào rơi về FP32 — nên mục 8 của danh sách kiểm tra (Mục 12.3) vẫn bắt buộc đo trên thiết bị đích.

## Chương 7

### BatchNorm phải được gộp vào tích chập ở thời điểm nào?
- [x] **Trước** khi lượng tử
- [ ] Sau khi lượng tử
- [ ] Không cần gộp, phần cứng tự xử lý
- [ ] Chỉ gộp khi dùng QAT
> Trên phần cứng BN không tồn tại như một lớp riêng. Nếu lượng tử $W$ rồi mới gộp thì trọng số thật sự chạy trên thiết bị là $W'$ với dải khác hẳn. Đây cũng là nguyên nhân làm dải giữa các kênh chênh lệch, vì $\gamma/\sqrt{\sigma^2+\epsilon}$ khác nhau theo kênh.

### Nút fake-quant làm gì?
- [ ] Thay phép nhân FP32 bằng phép nhân INT8 thật
- [x] Cho giá trị vẫn là số thực nhưng chỉ nhận các giá trị nằm trên lưới lượng tử
- [ ] Làm tròn trọng số về số nguyên rồi giữ nguyên như vậy
- [ ] Tắt gradient của lớp đó
> Đó là hàm hợp quantize → dequantize. Mục 6.5 đã kiểm chứng cách mô phỏng này trùng khớp **từng bit** với phép tính số nguyên thật, nên nó dự đoán đúng độ chính xác khi triển khai.

### Chế độ nào **không cần** dữ liệu calibration?
- [ ] Static quantization
- [x] Dynamic quantization (và weight-only)
- [ ] Cả ba chế độ đều cần
- [ ] Chỉ QAT mới không cần
> Dynamic lượng tử activation *lúc chạy* với dải tính từ chính tensor đó. Đổi lại phải tốn chi phí tìm min/max mỗi lần chạy, và không chạy thuần số nguyên được như Hình 9.

## Chương 8

### Tập calibration thường cần bao nhiêu mẫu?
- [ ] 1–10 mẫu
- [x] Khoảng 100–1000 mẫu
- [ ] Ít nhất 100 000 mẫu
- [ ] Toàn bộ tập huấn luyện
> Nhiều hơn 1000 ít khi cải thiện thêm. Quan trọng hơn số lượng là **tính đại diện**: đủ các lớp, đủ điều kiện, và tiền xử lý giống hệt lúc suy luận, với mô hình ở chế độ `eval()`.

### Phương pháp calibration nào nhạy nhất với outlier?
- [x] Min–max
- [ ] Percentile 99,9%
- [ ] Tối thiểu MSE
- [ ] Tối thiểu KL divergence
> Min–max lấy đúng giá trị lớn nhất nên chỉ một mẫu lạc cũng kéo dải ra rất rộng, làm bước nhảy thô cho toàn bộ phần còn lại.

### Ý tưởng của calibration theo KL divergence (entropy) là gì?
- [ ] Giữ cho giá trị lớn nhất không bị cắt
- [ ] Tối thiểu sai số bình phương trung bình
- [x] Giữ cho **phân phối** sau lượng tử gần phân phối gốc nhất
- [ ] Chọn ngưỡng bằng phân vị 99,99%
> Với mỗi ngưỡng ứng viên, ta lập histogram tham chiếu $P$ (dồn phần bị cắt vào bin cuối), gộp bin để được $Q$, rồi chọn ngưỡng cho $D_{KL}(P\,\|\,Q)$ nhỏ nhất.

## Chương 9

### Straight-Through Estimator giải quyết vấn đề gì?
- [ ] Làm cho phép làm tròn chính xác hơn
- [x] Đạo hàm của hàm làm tròn bằng 0 gần như khắp nơi, nên lượt ngược coi nó như hàm đồng nhất
- [ ] Giảm bộ nhớ khi huấn luyện
- [ ] Tự động chọn số bit cho từng lớp
> Nếu dùng đạo hàm thật, gradient về trọng số luôn bằng 0 và mạng không học được gì. STE đặt $\partial \hat{x}/\partial x \approx 1$ bên trong dải và 0 ở vùng bị cắt. Trong PyTorch viết gọn: `x + (fake_quant(x) - x).detach()`.

### Ở W8A8, QAT so với PTQ như thế nào trong thí nghiệm Mục 9.3?
- [ ] QAT tốt hơn rõ rệt
- [x] Ngang nhau trong phạm vi dao động — PTQ đã đủ, QAT không đáng chi phí
- [ ] PTQ tốt hơn rõ rệt
- [ ] Cả hai đều sụp đổ
> PTQ 99,26% so với QAT 99,07%; chênh 0,19 điểm đúng bằng **1 ảnh** trên 540 ảnh test. Quy tắc: luôn bắt đầu bằng PTQ.

### Ở W3A3, kết quả ra sao?
- [ ] PTQ và QAT vẫn ngang nhau
- [x] PTQ sụp đổ còn 66–72%, QAT giữ được 94,01%
- [ ] Cả hai đều dưới 50%
- [ ] QAT kém hơn PTQ
> Đây chính là khoảng mà QAT thực sự đáng giá. Ở 2 bit thì ngay cả QAT cũng chỉ còn 49,44%, cần thêm kỹ thuật chuyên biệt như học scale kiểu LSQ hoặc chưng cất tri thức.

### Cross-Layer Equalization dựa trên tính chất nào của ReLU?
- [ ] $\text{ReLU}(x) \ge 0$
- [x] $\text{ReLU}(s z) = s\,\text{ReLU}(z)$ với mọi $s > 0$
- [ ] $\text{ReLU}$ có đạo hàm bằng 1 với $x > 0$
- [ ] $\text{ReLU}(x) + \text{ReLU}(-x) = |x|$
> Nhờ tính thuần nhất dương đó, ta chia kênh $i$ của lớp trước cho $s_i$ và nhân kênh đầu vào $i$ của lớp sau với $s_i$ mà đầu ra **không đổi**. Chọn $s_i$ khéo thì dải giữa các kênh đồng đều hơn. SmoothQuant dùng đúng ý tưởng này.

## Chương 10

### `torch.ao.quantization.quantize_dynamic` lượng tử những lớp nào?
- [ ] Mọi lớp, kể cả Conv2d
- [x] Chỉ `nn.Linear` và các lớp hồi quy (LSTM, GRU…) — **không** hỗ trợ `nn.Conv2d`
- [ ] Chỉ Conv2d
- [ ] Chỉ các lớp chuẩn hoá
> Mô hình tích chập cần static quantization. Đây là một trong những hiểu lầm được liệt kê ở Chương 13.

### Vì sao `Int8WeightOnly` của torchao chạy eager lại **chậm hơn** FP32?
- [ ] Vì INT8 vốn chậm hơn FP32
- [x] Vì không có `torch.compile` để sinh kernel hợp nhất, trọng số bị giải lượng tử về số thực ở mỗi lần gọi
- [ ] Vì mô hình lớn hơn sau khi lượng tử
- [ ] Vì phải tính min/max của activation mỗi lần chạy
> Tốn thêm công giải lượng tử mà không bớt được phép tính nào. torchao được thiết kế để dùng **cùng** `torch.compile`.

### Một báo cáo lượng tử hoá đáng tin cần đủ những con số nào?
- [ ] Chỉ cần độ chính xác
- [ ] Độ chính xác và kích thước
- [x] Kích thước file triển khai, độ trễ (có warm-up, cố định số luồng), và độ chính xác — đo **trên thiết bị đích**
- [ ] Chỉ cần độ trễ trên máy phát triển
> Đo `state_dict` trong bộ nhớ thay vì file triển khai, hay đo độ trễ trên máy dev thay vì thiết bị đích, là hai cách phổ biến nhất để tự lừa mình.

## Chương 11

### Vì sao weight-only 4 bit (W4A16) tăng tốc LLM gần tỉ lệ với mức nén?
- [ ] Vì phép nhân được làm bằng INT4
- [x] Vì sinh văn bản là bài toán memory-bound: mỗi token phải đọc toàn bộ trọng số, nên nén trọng số là nới đúng nút thắt
- [ ] Vì giảm số lớp phải tính
- [ ] Vì KV cache nhỏ đi
> Phép nhân vẫn làm ở FP16. Cái được cải thiện là số byte phải kéo từ bộ nhớ về — đúng nhận định ở Mục 1.3.

### SmoothQuant làm gì?
- [ ] Tách các kênh outlier ra tính bằng FP16
- [x] Dùng một biến đổi tương đương toán học để chuyển bớt độ khó từ activation sang trọng số
- [ ] Huấn luyện lại mô hình với nút fake-quant
- [ ] Lượng tử KV cache theo kênh
> $Y = XW^\top = (X\,\text{diag}(s)^{-1})(W\,\text{diag}(s))^\top$. Phép chia cho $s$ được gộp sẵn vào LayerNorm đứng trước nên lúc chạy không tốn thêm gì. Phương án "tách outlier sang FP16" là LLM.int8().

### Kết quả nào của GPTQ nghe nghịch lý nhưng rất đáng suy ngẫm?
- [ ] Nó giảm cả sai số trọng số lẫn sai số đầu ra
- [x] Nó làm sai số trọng số **tăng 2,7 lần** nhưng sai số đầu ra **giảm 3,7 lần**
- [ ] Nó nhanh hơn RTN nhưng kém chính xác hơn
- [ ] Nó chỉ chạy được với mô hình dưới 1 tỉ tham số
> Các sai số được sắp xếp để triệt tiêu nhau *theo những hướng mà dữ liệu thực sự đi qua*. Cái cần bảo toàn là **hàm số mà lớp biểu diễn**, không phải từng con số trong ma trận.

### NF4 đặt 16 mức lượng tử ở đâu?
- [ ] Cách đều nhau trên $[-1, 1]$
- [x] Tại các **phân vị** của phân phối chuẩn, để mỗi mức gánh một lượng dữ liệu xấp xỉ bằng nhau
- [ ] Tại các luỹ thừa của 2
- [ ] Tại các giá trị học được bằng gradient
> Trọng số đã huấn luyện xấp xỉ phân phối chuẩn; sau khi chia mỗi khối 64 phần tử cho absmax, chúng nằm trong $[-1,1]$. Tập mức bất đối xứng (8 dương, 7 âm) để vừa có số 0 chính xác vừa dùng hết 16 mã. NF4 là **định dạng lưu trữ**, không phải định dạng tính toán.

### "Mô hình 4 bit" trong thực tế tốn bao nhiêu bit mỗi trọng số?
- [ ] Đúng 4,0
- [x] Khoảng 4,1–4,5, vì còn phải lưu scale (và zero-point) cho mỗi nhóm
- [ ] Khoảng 6
- [ ] Khoảng 8
> Ví dụ: INT4 nhóm 128 scale FP16 là $4 + 16/128 = 4{,}125$; GGUF `Q4_0` và `Q4_K` đều là 4,5. Cộng thêm một số lớp (embedding, lớp đầu ra) thường giữ ở độ chính xác cao hơn.

### Quét $\alpha$ của SmoothQuant trên chính dữ liệu của Mục 11.3 cho kết quả nào?
- [ ] Sai số giảm đều khi $\alpha$ tăng từ 0 lên 1
- [ ] $\alpha = 0$ tốt nhất, vì trọng số được chuẩn hoá hoàn toàn
- [x] Đường cong hình chữ U, cực tiểu ở $\alpha \approx 0{,}60$; còn $\alpha = 0$ gần như không giúp được gì
- [ ] Sai số không phụ thuộc $\alpha$ vì phép biến đổi là tương đương toán học
> Đo thật bằng `code/sweep_alpha.py`: 8,02% tại $\alpha = 0$, 1,44% tại 0,5, **1,34% tại 0,60**, 3,64% tại 1,0. Biến đổi tuy tương đương về mặt toán học nhưng nó đổi *dữ liệu đem đi lượng tử*, nên sai số đổi theo. Xem [Bài 8](bai-tap.html#bai-8).

### Vì sao cực tiểu lại lệch sang phải điểm cân bằng $\alpha = 0{,}5$?
- [ ] Vì bài báo SmoothQuant chọn sai giá trị mặc định
- [x] Vì ở $\alpha = 0{,}5$ hai vế có **biên độ** bằng nhau nhưng **sai số** thì không: activation chỉ có một scale cho cả tensor, còn trọng số có một scale cho mỗi kênh đầu ra
- [ ] Vì dữ liệu mô phỏng có đúng 6 kênh outlier
- [ ] Vì trọng số luôn dễ lượng tử hơn activation ở mọi mức chênh lệch
> Đo riêng từng nguồn ở $\alpha = 0{,}5$: sai số do trọng số 0,60%, do activation 1,31%. Vế có 512 scale hấp thụ cùng một mức chênh lệch rẻ hơn hẳn vế chỉ có 1 scale, nên đẩy thêm độ khó sang trọng số vẫn còn lợi. Quy tắc mang đi được: **cực tiểu lệch về phía tensor có granularity mịn hơn.**

## Chương 12

### Trong phân tích độ nhạy ở Mục 12.2, lớp nào nhạy nhất?
- [ ] conv1, lớp đầu
- [ ] fc, lớp cuối
- [x] pw, lớp pointwise 1×1 — giảm 11,30 điểm phần trăm
- [ ] dw, lớp depthwise
> Kết quả này **trái với quy tắc kinh nghiệm** "lớp đầu và lớp cuối nhạy nhất": ở đây lớp cuối không nhạy chút nào (giảm 0,00 điểm). Quy tắc kinh nghiệm chỉ là điểm xuất phát.

### Cách đơn giản và hiệu quả nhất để tìm lớp gây mất độ chính xác là gì?
- [ ] Xem lớp nào có nhiều tham số nhất
- [ ] Xem lớp nào có dải trọng số rộng nhất
- [x] Lượng tử **từng lớp một** trong khi giữ các lớp khác ở FP32, rồi đo độ chính xác
- [ ] Xem lớp nào có gradient lớn nhất
> Cách này tốn $O(\text{số lớp})$ lần đánh giá nhưng cho kết quả trực tiếp trên chính đại lượng ta quan tâm. Có kết quả rồi mới quyết định mixed precision hay QAT.

## Chương 13

### "INT8 luôn giảm kích thước file đúng 4 lần." Phát biểu này:
- [ ] Đúng trong mọi trường hợp
- [x] Sai — file còn chứa siêu dữ liệu, scale và các lớp giữ FP32; mô hình nhỏ ở Mục 10.1.2 chỉ giảm 2,4 lần
- [ ] Sai — thực ra luôn giảm 8 lần
- [ ] Sai — kích thước không đổi
> Tỉ lệ 4 lần đúng với **phần trọng số**. Mô hình càng lớn thì tỉ lệ thực tế càng tiến gần 4.

### "Sai số trọng số càng nhỏ thì mô hình càng chính xác." Phát biểu này:
- [ ] Đúng — đó là mục tiêu của mọi phương pháp lượng tử
- [x] Sai — cái cần bảo toàn là **đầu ra**; GPTQ tăng sai số trọng số 2,7 lần mà giảm sai số đầu ra 3,7 lần
- [ ] Đúng, trừ trường hợp dùng per-channel
- [ ] Sai — thực ra hai đại lượng không liên quan gì nhau
> Đây là hiểu lầm nền tảng nhất trong danh sách, vì nó quyết định ta **tối ưu hàm mục tiêu nào**. AdaRound và GPTQ đều xuất phát từ việc sửa lại mục tiêu này.
