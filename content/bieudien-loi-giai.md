# Lời giải chi tiết — Biểu diễn, Sinh và Căn chỉnh

Mỗi mục ứng với một bài trong Chương 16. Dòng `@meta` được script build đọc để gắn nhãn
chương, dạng bài và độ khó; nó không hiện ra trên trang.

## Bài 1
@meta chuong=2 | dang=Suy luận | kho=Trung bình

**(a) Hàm mục tiêu và đạo hàm.** Với một cặp $(i,j)$, đặt $s = \langle w_i, c_j\rangle$. Hàm mục tiêu kỳ vọng của skip-gram với lấy mẫu âm gồm hai phần: phần **cặp dương**, cân nặng theo số lần đồng hiện thật, và phần **cặp âm**, cân nặng theo tần suất mà phép lấy mẫu âm sinh ra cặp ấy:

$$\ell(s) = \#(i,j)\,\log\sigma(s) \;+\; k\,\#(i)\,P_n(j)\,\log\sigma(-s).$$

Dùng $\frac{d}{ds}\log\sigma(s) = 1-\sigma(s)$ và $\frac{d}{ds}\log\sigma(-s) = -\sigma(s)$:

$$\ell'(s) = \#(i,j)\,(1-\sigma(s)) \;-\; k\,\#(i)\,P_n(j)\,\sigma(s) = 0.$$

**(b) Suy ra dạng nghiệm.** Chuyển vế:

$$\frac{\sigma(s)}{1-\sigma(s)} = \frac{\#(i,j)}{k\,\#(i)\,P_n(j)}.$$

Vế trái chính là $e^{s}$, vì $\sigma(s) = \frac{e^s}{1+e^s}$ nên $\frac{\sigma}{1-\sigma} = e^s$. Thay $P_n(j) = \#(j)/N$:

$$e^{s} = \frac{\#(i,j)\,N}{k\,\#(i)\,\#(j)} \;\Longrightarrow\; \boxed{\;s = \log\frac{\#(i,j)\,N}{\#(i)\,\#(j)} - \log k = \mathrm{PMI}(i,j) - \log k\;}$$

**(c) Tăng $k$ từ 1 lên 15.** Ma trận đích dịch xuống một lượng $\log 15 = 2{,}708$. Vì PMI của các cặp không liên quan vốn đã bằng 0, việc trừ đi 2,708 làm **rất nhiều ô trở thành âm**, và sau khi cắt phần âm (PPMI) thì chúng thành 0.

Nói cách khác, $k$ lớn làm ma trận đích **thưa hơn nhiều**. Trên kho ngữ liệu thật thì đó là điều tốt — nó lọc bỏ các cặp đồng hiện yếu vốn ước lượng rất nhiễu. Nhưng trên kho ngữ liệu đồ chơi 20 từ của tài liệu này, PMI lớn nhất chỉ là $1{,}19$, nên $k = 15$ làm **toàn bộ** ma trận về 0 và không còn gì để phân rã. Đây đúng là điều tôi quan sát được khi thử, và là lý do các thí nghiệm dùng $k = 1$.

**(d) Hạng đầy đủ so với hạng thấp.** Kết quả ở (b) là một điều kiện **theo từng ô**: nó nói giá trị $\langle w_i, c_j\rangle$ phải bằng bao nhiêu cho **mỗi** cặp $(i,j)$.

Nếu embedding có $V$ chiều thì ma trận $WC^\top$ có thể là **bất kỳ** ma trận $V\times V$ nào, nên tối ưu hoá thoải mái chạm tới điều kiện ấy ở mọi ô. Đo được ở Mục 2.6: tương quan **0,9995**, lệch trung bình 0,0106.

Nếu embedding chỉ có 8 chiều thì $WC^\top$ bị ép có hạng $\le 8$, tức nằm trên một mặt cong con trong không gian các ma trận. Điều kiện từng ô nói chung **không thoả được**, và tối ưu hoá phải thoả hiệp. Đo được: tương quan chỉ 0,3727.

Và đây là điểm đáng nhớ: sự thoả hiệp ấy **chính là thứ ta muốn**. Nếu khớp được hoàn hảo thì ta chỉ có một cách viết lại ma trận PMI, không có khái quát hoá nào. Ràng buộc hạng buộc mô hình phải gom các từ hành xử giống nhau lại với nhau — và đó là ý nghĩa của embedding.

## Bài 2
@meta chuong=3 | dang=Chẩn đoán | kho=Cơ bản

**(a) Chẩn đoán: bất đẳng hướng.**

Cơ chế: cosine đo góc, mà góc chỉ có nghĩa nếu đám mây điểm trải đều **quanh gốc toạ độ**. Trạng thái ẩn của mô hình Transformer thì không — chúng dồn về một vùng hẹp lệch hẳn khỏi gốc. Khi mọi vector đều có một thành phần chung lớn theo cùng một hướng, thì tích vô hướng giữa hai vector bất kỳ bị thành phần chung ấy chi phối.

Kết quả là điểm cosine phản ánh **hướng chung của cả tập** chứ không phản ánh quan hệ giữa hai câu cụ thể.

Cần nói rõ đây không phải lỗi cài đặt. Nó là đặc tính của biểu diễn sinh ra từ mô hình được huấn luyện để đoán token tiếp theo — không có sức ép nào trong quá trình ấy buộc các biểu diễn phải trải đều.

**(b) Phép sửa rẻ nhất: trừ vector trung bình.**

Tính $\bar v$ trên một mẫu đại diện (vài nghìn câu là đủ) rồi dùng $u - \bar v$ thay cho $u$ khi so sánh.

Hiệu quả đo được ở Mục 3.2:

| Mức lệch tâm | Cosine TB trước | Cosine TB sau |
|---|---|---|
| Nhẹ | 0,3899 | $-0{,}0011$ |
| Mạnh | **0,8724** | $-0{,}0011$ |

Chỉ một phép trừ đưa cosine trung bình về gần 0 ở cả hai mức. Chi phí là một phép trừ vector cho mỗi truy vấn — không đáng kể.

**(c) Vì sao không nên nâng ngưỡng lên 0,93.**

Ba lý do, theo thứ tự nặng dần:

1. **Nó chữa triệu chứng chứ không chữa nguyên nhân.** Phần thông tin hữu ích vẫn bị chôn dưới thành phần chung; ta chỉ đang cắt ở một chỗ khác trên cùng một phân bố tệ.
2. **Ngưỡng ấy không bền.** Vector trung bình đổi theo miền dữ liệu, theo mô hình, theo cả độ dài câu. Nâng ngưỡng cho tập kiểm thử hôm nay thì mai đổi dữ liệu là sai.
3. **Nó thu hẹp dải phân biệt.** Nếu mọi điểm nằm trong $[0{,}82;\ 0{,}94]$ thì toàn bộ thông tin nằm trong một dải rộng 0,12. Sau khi trừ trung bình, dải ấy giãn ra gần hết khoảng $[-1, 1]$, nên xếp hạng ổn định hơn nhiều.

**(d) Cách chữa triệt để: huấn luyện với mục tiêu tương phản.**

Thay vì sửa embedding sau khi có, huấn luyện ngay từ đầu để cặp cùng nghĩa gần nhau và cặp khác nghĩa xa nhau. Hàm mục tiêu có dạng entropy chéo softmax, với các câu khác trong cùng lô làm mẫu âm.

Cái giá: cần **dữ liệu cặp** — câu hỏi với câu trả lời, câu với bản dịch, hai cách diễn đạt của cùng một ý. Thu thập nó tốn công hơn nhiều so với việc chỉ lấy một mô hình có sẵn. Và cần lô lớn, vì lô càng lớn thì càng nhiều mẫu âm.

Nên thứ tự nên thử là: trừ trung bình trước (rẻ, thường đã đủ), làm trắng nếu chưa đủ, và chỉ huấn luyện tương phản khi hai cách kia không đạt yêu cầu.

## Bài 3
@meta chuong=7 | dang=Tính tay | kho=Cơ bản

**(a) Công thức và các giá trị.** Mỗi lớp có hai ma trận được gắn LoRA ($W_Q$ và $W_V$), mỗi ma trận tốn $2dr$:

$$N_{\text{LoRA}} = L \times 2 \times 2dr = 4Ldr = 4 \times 32 \times 4096 \times r = 524\,288\,r.$$

| $r$ | Tham số huấn luyện | % của 6 476 005 376 |
|---|---|---|
| 4 | 2 097 152 | 0,0324% |
| 8 | 4 194 304 | 0,0648% |
| 16 | 8 388 608 | 0,1295% |
| 64 | 33 554 432 | 0,5181% |

**(b) Trạng thái Adam.** Adam giữ **hai** trạng thái cho mỗi tham số, mỗi cái 4 byte ở FP32:

$$4\,194\,304 \times 2 \times 4 = 33\,554\,432 \text{ byte} = \boxed{32{,}0 \text{ MiB}}.$$

Tinh chỉnh toàn phần:

$$6\,476\,005\,376 \times 2 \times 4 = 51\,808\,043\,008 \text{ byte} = \boxed{48{,}2 \text{ GiB}}.$$

Tỉ lệ: $48{,}2 \times 1024 / 32 = \mathbf{1\,542}$ lần.

Đây là con số quyết định. 32 MiB nằm gọn trong bộ nhớ dư của bất kỳ GPU nào; 48,2 GiB thì ngay cả A100 80 GB cũng chật sau khi cộng thêm trọng số (24,1 GiB) và gradient (24,1 GiB) — tổng 96,4 GiB, tức **không vừa**.

**(c) Tỉ lệ $2r/d$ theo $d$.**

| $d$ | $2r/d$ với $r=8$ |
|---|---|
| 768 | 2,083% |
| 1 280 | 1,250% |
| 4 096 | 0,391% |
| 8 192 | 0,195% |

Tỉ lệ **giảm tuyến tính theo $d$**. Nghĩa là **mô hình càng lớn thì LoRA càng có lợi**, và lợi theo cấp số.

Lý do trực giác: số tham số của một lớp tăng như $d^2$ còn LoRA chỉ tăng như $d$. Với mô hình khổng lồ, phần "chỉnh sửa hạng thấp" trở thành một phần cực nhỏ so với bản thân mô hình. Đây là một trong những lý do LoRA trở nên phổ biến đúng vào lúc mô hình bắt đầu lớn lên.

**(d) Lưu trữ cho 50 khách hàng.**

Tinh chỉnh toàn phần: mỗi khách một bản sao đầy đủ.

$$50 \times 13\text{ GB} = \boxed{650 \text{ GB}}$$

LoRA: một bản gốc dùng chung, cộng 50 adapter.

$$13\text{ GB} + 50 \times 16\text{ MB} = 13 + 0{,}8 = \boxed{13{,}8 \text{ GB}}$$

Tiết kiệm **47 lần**. Nhưng món lợi thật còn lớn hơn con số lưu trữ: vì tất cả dùng chung trọng số gốc, ta giữ **một** bản mô hình trên GPU và chỉ hoán đổi adapter theo từng yêu cầu. Với tinh chỉnh toàn phần thì mỗi khách hàng cần một bản mô hình riêng trên GPU — tức 50 GPU, hoặc nạp lại 13 GB mỗi lần đổi khách, cả hai đều không chấp nhận được.

Đây là lý do LoRA được dùng trong sản phẩm thật, và nó **độc lập** với chuyện tiết kiệm bộ nhớ huấn luyện.

## Bài 4
@meta chuong=11 | dang=Suy luận | kho=Trung bình

**(a) Chứng minh bằng quy nạp.**

*Cơ sở.* Với $t = 1$: $q(x_1\mid x_0) = \mathcal{N}(\sqrt{\alpha_1}x_0,\ \beta_1 I)$ theo định nghĩa, và $\bar\alpha_1 = \alpha_1$, $1-\bar\alpha_1 = 1-\alpha_1 = \beta_1$. Khớp.

*Bước quy nạp.* Giả sử $x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,x_0 + \sqrt{1-\bar\alpha_{t-1}}\,\varepsilon_1$ với $\varepsilon_1\sim\mathcal{N}(0,I)$. Theo định nghĩa một bước:

$$x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{\beta_t}\,\varepsilon_2, \qquad \varepsilon_2\sim\mathcal{N}(0,I) \text{ độc lập với } \varepsilon_1.$$

Thay vào:

$$x_t = \sqrt{\alpha_t\bar\alpha_{t-1}}\,x_0 + \sqrt{\alpha_t(1-\bar\alpha_{t-1})}\,\varepsilon_1 + \sqrt{\beta_t}\,\varepsilon_2.$$

Hệ số của $x_0$ là $\sqrt{\bar\alpha_t}$ theo định nghĩa $\bar\alpha_t = \alpha_t\bar\alpha_{t-1}$.

Hai số hạng nhiễu độc lập nên **phương sai cộng được** (đây là chỗ cần tính chất độc lập — xem [Mục 3.2 của *Nền tảng*](nentang-ch03.html)):

$$\alpha_t(1-\bar\alpha_{t-1}) + \beta_t = \alpha_t - \alpha_t\bar\alpha_{t-1} + 1 - \alpha_t = 1 - \bar\alpha_t.$$

Tổng hai Gauss độc lập là một Gauss, nên

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon, \qquad \varepsilon\sim\mathcal{N}(0,I). \qquad \blacksquare$$

**(b) Vì sao bắt buộc.** Huấn luyện đòi lấy mẫu $x_t$ với $t$ **ngẫu nhiên** trong $\{1,\dots,1000\}$, và làm việc này hàng triệu lần. Nếu phải mô phỏng từng bước thì trung bình mất 500 bước cho mỗi mẫu huấn luyện — tức chậm hơn khoảng **500 lần**.

Với dạng đóng, ta rút $\varepsilon$ một lần và dựng $x_t$ bằng **một** phép tính, bất kể $t$ bằng bao nhiêu. Không có tính chất này thì mô hình khuếch tán không huấn luyện được ở quy mô thực tế.

Và tính chất ấy không phải may mắn — nó là hệ quả của việc chọn mỗi bước là **Gauss tuyến tính**. Nếu quá trình thuận dùng một phép biến đổi phi tuyến thì không gộp được.

**(c) Tín hiệu còn lại ở $t = 999$.**

Biên độ tín hiệu là $\sqrt{\bar\alpha_t}$, không phải $\bar\alpha_t$:

$$\sqrt{4{,}0\times10^{-5}} = 6{,}3\times10^{-3} = \boxed{0{,}63\%}$$

Con số này quan trọng vì khi lấy mẫu, ta **bắt đầu từ $\mathcal{N}(0,I)$**. Điều đó chỉ hợp lệ nếu $q(x_T\mid x_0)$ thật sự gần $\mathcal{N}(0,I)$ với mọi $x_0$. Còn 0,63% tín hiệu nghĩa là phân phối cuối gần như không phụ thuộc điểm xuất phát nữa — đúng điều cần.

Nếu lịch nhiễu dừng quá sớm, chẳng hạn $\bar\alpha_T = 0{,}1$, thì $x_T$ vẫn mang 32% biên độ của $x_0$. Khi ấy bắt đầu lấy mẫu từ nhiễu thuần tuý là **sai phân phối**, và mẫu sinh ra sẽ lệch.

**(d) Vì sao $t$ nhỏ học chi tiết còn $t$ lớn học bố cục.**

Ở $t$ nhỏ, SNR rất cao (9 999 ở $t=0$): ảnh gần như nguyên vẹn, chỉ có một lớp nhiễu mỏng. Để đoán được lớp nhiễu ấy, mô hình phải phân biệt được "nhiễu" với "kết cấu thật" — tức phải nắm rất kỹ **chi tiết ở mức điểm ảnh**. Bố cục thì nó không cần nhìn, vì bố cục vẫn còn nguyên đó.

Ở $t$ lớn, SNR rất thấp (0,0015 ở $t=800$): ảnh gần như chỉ còn nhiễu, chi tiết đã mất sạch. Thông tin duy nhất còn sót lại là **những gì có biên độ lớn nhất** — bố cục thô, vùng sáng tối, màu tổng thể. Mô hình chỉ có thể học những thứ đó, vì không còn gì khác.

Hệ quả khi lấy mẫu: đi ngược từ $t=T$ về $t=0$, mô hình quyết định bố cục trước rồi mới điền chi tiết — giống hệt cách người ta vẽ tranh. Điều đó **không được lập trình vào**; nó rơi ra từ lịch nhiễu.

## Bài 5
@meta chuong=10 | dang=Suy luận | kho=Trung bình

**(a) Hai đạo hàm.** Với $s$ là điểm số (logit) của bộ phân biệt trên mẫu giả, và $D = \sigma(s)$:

$$\frac{\partial}{\partial s}\log\big(1-\sigma(s)\big) = \frac{-\sigma'(s)}{1-\sigma(s)} = \frac{-\sigma(s)(1-\sigma(s))}{1-\sigma(s)} = -\sigma(s).$$

$$\frac{\partial}{\partial s}\log\sigma(s) = \frac{\sigma'(s)}{\sigma(s)} = \frac{\sigma(s)(1-\sigma(s))}{\sigma(s)} = 1-\sigma(s).$$

**(b) Độ lớn lúc bắt đầu.** Lúc bắt đầu, bộ sinh còn tệ nên bộ phân biệt nhận ra hàng giả dễ dàng: $\sigma(s)\approx 0$.

| Hàm | Đạo hàm | Giá trị khi $\sigma(s)\approx 0$ |
|---|---|---|
| Tối thiểu $\log(1-D)$ | $-\sigma(s)$ | $\approx \mathbf{0}$ |
| Tối đa $\log D$ | $1-\sigma(s)$ | $\approx \mathbf{1}$ |

Chênh lệch là **toàn bộ** so với **không có gì**.

**(c) Hậu quả, đo được.**

| Hàm mất mát của bộ sinh | Chế độ phủ được (4 hạt giống) | % điểm rơi vào cụm |
|---|---|---|
| Minimax gốc | $[0,0,0,0]$ | 0,0% |
| Không bão hoà | $[8,8,8,8]$ | 15,3% |

Dạng gốc **thất bại hoàn toàn**, ở cả bốn hạt giống, không học được gì.

Cơ chế là một vòng luẩn quẩn: bộ sinh tệ → bộ phân biệt tự tin → gradient của bộ sinh gần 0 → bộ sinh không cải thiện → vẫn tệ. Và vòng này khép lại ngay ở bước đầu tiên, khi bộ sinh **luôn luôn** tệ.

Hàm không bão hoà phá vòng ấy vì đạo hàm $1-\sigma(s)$ **lớn nhất đúng lúc bộ sinh tệ nhất** — tức tín hiệu mạnh nhất xuất hiện đúng khi cần nhất.

**(d) Bài học tổng quát và một ví dụ khác.**

Bài học: **hai hàm mất mát có cùng điểm tối ưu vẫn có thể cho hành vi huấn luyện khác nhau một trời một vực**, vì thứ quyết định việc học là **gradient dọc đường đi**, không phải vị trí đích.

Đây là một sai lầm dễ mắc, vì khi phân tích một hàm mất mát ta hay nhìn xem nó tối ưu ở đâu. Nhưng thuật toán không nhảy thẳng tới điểm tối ưu — nó đi từng bước theo gradient, nên hình dạng của gradient trên cả đường đi mới là thứ quan trọng.

**Ví dụ khác trong bộ giáo trình này:** [Mục 6.1 của *Nền tảng*](nentang-ch06.html) so sánh mất mát perceptron, hinge và logistic. Cả ba đều là chặn trên lồi của mất mát 0–1 và đều "đúng" theo nghĩa giảm chúng thì giảm tỉ lệ đoán sai. Nhưng mất mát perceptron **phẳng hoàn toàn khi $m>0$**, nên thuật toán buông ngay khi vừa đúng và dao động mãi trên dữ liệu không tách được — đo được 17 977 lần sai qua 2 000 lượt quét mà không hội tụ. Hinge và logistic không phẳng ngay tại $m=0$ nên vẫn còn động lực đẩy điểm ra xa biên.

Cùng một bài học, hai lĩnh vực khác nhau.

## Bài 6
@meta chuong=14 | dang=Tính tay | kho=Nâng cao

**(a) Suy ra dạng đóng.** Viết bài toán trên không gian rời rạc, bỏ điều kiện $x$ cho gọn:

$$\max_\pi \sum_y \pi(y)\,r(y) - \beta\sum_y \pi(y)\log\frac{\pi(y)}{\pi_{\text{ref}}(y)} \quad\text{với}\quad \sum_y\pi(y) = 1.$$

Lagrange với nhân tử $\lambda$ cho ràng buộc chuẩn hoá:

$$\mathcal{L} = \sum_y \pi(y)r(y) - \beta\sum_y\pi(y)\log\frac{\pi(y)}{\pi_{\text{ref}}(y)} + \lambda\Big(\sum_y\pi(y) - 1\Big).$$

Đạo hàm theo $\pi(y)$, chú ý $\frac{\partial}{\partial \pi}\big[\pi\log\frac{\pi}{\pi_{\text{ref}}}\big] = \log\frac{\pi}{\pi_{\text{ref}}} + 1$:

$$\frac{\partial\mathcal{L}}{\partial\pi(y)} = r(y) - \beta\Big(\log\frac{\pi(y)}{\pi_{\text{ref}}(y)} + 1\Big) + \lambda = 0.$$

Giải theo $\pi(y)$:

$$\log\frac{\pi(y)}{\pi_{\text{ref}}(y)} = \frac{r(y) + \lambda}{\beta} - 1 \;\Longrightarrow\; \pi(y) = \pi_{\text{ref}}(y)\,\exp\!\Big(\frac{r(y)}{\beta}\Big)\cdot\underbrace{e^{\lambda/\beta - 1}}_{\text{hằng số}}.$$

Hằng số được xác định bởi ràng buộc $\sum_y\pi(y)=1$, tức nó đúng bằng $1/Z$ với $Z = \sum_y \pi_{\text{ref}}(y)e^{r(y)/\beta}$:

$$\boxed{\;\pi^{*}(y) = \frac{1}{Z}\,\pi_{\text{ref}}(y)\,\exp\!\Big(\frac{r(y)}{\beta}\Big)\;}$$

**(b) Hai giới hạn.**

$\beta\to 0$: số mũ $r(y)/\beta$ nổ ra, nên $\exp(r/\beta)$ bị chi phối hoàn toàn bởi $y$ có $r$ lớn nhất. Chính sách **dồn toàn bộ xác suất vào câu trả lời thưởng cao nhất** và bỏ qua $\pi_{\text{ref}}$. Bằng lời: không quan tâm gì tới mô hình gốc, chỉ đuổi theo điểm thưởng.

$\beta\to\infty$: số mũ về 0, nên $\exp(r/\beta)\to 1$ với mọi $y$, và $\pi^* \to \pi_{\text{ref}}$. Bằng lời: giữ nguyên mô hình gốc, **không học được gì** từ phần thưởng.

Con số đo được ở Mục 14.4 khớp với hai giới hạn ấy: ở $\beta = 0{,}05$ thì $\mathrm{KL} = 4{,}74$ và $\mathbb{E}[r] = 2{,}44$; ở $\beta = 5$ thì $\mathrm{KL} = 0{,}017$ và $\mathbb{E}[r] = -0{,}35$, tức gần như không nhúc nhích khỏi $\pi_{\text{ref}}$.

**(c) Khi $\pi_{\text{ref}}(y_0) = 0$.**

$$\pi^*(y_0) = \frac{1}{Z}\cdot 0 \cdot \exp\!\Big(\frac{r(y_0)}{\beta}\Big) = \boxed{0}$$

bất kể $r(y_0)$ lớn tới đâu, vì nhân với 0 vẫn là 0.

**Ý nghĩa về an toàn:** ràng buộc KL không chỉ giữ mô hình "gần" bản gốc theo nghĩa mờ — nó **bảo đảm tuyệt đối** rằng những gì mô hình gốc coi là không thể thì vẫn không thể. Nếu $\pi_{\text{ref}}$ gán xác suất 0 cho một kiểu đầu ra, thì không mô hình thưởng nào, dù hỏng tới đâu, có thể làm nó xuất hiện.

**Nhưng phải nói ngay mặt trái**, vì đây là chỗ dễ hiểu nhầm: trong thực tế $\pi_{\text{ref}}$ **gần như không bao giờ** gán đúng 0 cho bất cứ chuỗi nào — một mô hình ngôn ngữ luôn gán xác suất dương, dù rất nhỏ, cho mọi chuỗi. Nên bảo đảm ở trên đúng về toán mà gần như vô dụng trong thực tế. Xem thêm phần (c) của Bài 9.

**(d) Vì sao $\beta$ là vị trí trên đường đánh đổi.**

Từ bảng ở Mục 14.4:

| $\beta$ | KL | $\mathbb{E}[r]$ |
|---|---|---|
| 0,05 | 4,741 | 2,438 |
| 0,20 | 4,658 | 2,424 |
| 1,00 | 0,417 | 0,303 |
| 5,00 | 0,017 | $-0{,}350$ |

Hai cột **đi cùng chiều**: muốn thưởng cao hơn thì phải chấp nhận KL lớn hơn. Không có giá trị $\beta$ nào cho cả hai tốt cùng lúc, vì chúng nằm trên một đường cong đánh đổi.

Nên câu hỏi "$\beta$ bằng bao nhiêu là đúng?" **không có câu trả lời kỹ thuật**. Câu hỏi đúng là: *ta chấp nhận lệch khỏi mô hình gốc bao nhiêu để đổi lấy bao nhiêu điểm thưởng?* Và đó là quyết định sản phẩm, phụ thuộc mức độ ta tin mô hình thưởng.

Cách đọc này giống hệt cách đọc đường ROC ở [Mục 8.3 của *Nền tảng*](nentang-ch08.html): không có điểm nào "tốt nhất", chỉ có một đường, và chọn điểm nào là quyết định nghiệp vụ.

## Bài 7
@meta chuong=15 | dang=Suy luận | kho=Nâng cao

**(a) Đảo ngược dạng đóng.** Từ Bài 6:

$$\pi^{*}(y\mid x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp\!\Big(\frac{r(x,y)}{\beta}\Big).$$

Lấy log hai vế:

$$\log\pi^{*}(y\mid x) = \log\pi_{\text{ref}}(y\mid x) + \frac{r(x,y)}{\beta} - \log Z(x).$$

Giải theo $r$:

$$\boxed{\;r(x,y) = \beta\log\frac{\pi^{*}(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x)\;}$$

Đọc câu này cho kỹ: **mọi chính sách đều ngầm định nghĩa một hàm thưởng.** Quan hệ giữa $\pi$ và $r$ là song ánh (với $\pi_{\text{ref}}$ và $\beta$ cho trước), nên tham số hoá một cái là tham số hoá cái kia.

**(b) Thay vào Bradley–Terry và chỉ chỗ triệt tiêu.** Hợp lý Bradley–Terry là

$$P(y_w \succ y_l\mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big).$$

Thay biểu thức ở (a) vào **hiệu**:

$$r(x,y_w) - r(x,y_l) = \left[\beta\log\frac{\pi(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} + \beta\log Z(x)\right] - \left[\beta\log\frac{\pi(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)} + \beta\log Z(x)\right]$$

$$= \beta\log\frac{\pi(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)} + \underbrace{\beta\log Z(x) - \beta\log Z(x)}_{\textbf{= 0}}.$$

Chỗ triệt tiêu là **hai số hạng $\beta\log Z(x)$ ở cuối mỗi ngoặc**. Chúng bằng nhau nên trừ đi hết.

Kết quả là hàm mất mát DPO:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right].$$

Mọi đại lượng trong đó đều tính được trực tiếp: chỉ cần chạy $\pi_\theta$ và $\pi_{\text{ref}}$ trên hai câu trả lời.

**(c) Vì sao phải cùng một $x$.**

$Z(x) = \sum_y \pi_{\text{ref}}(y\mid x)\exp(r(x,y)/\beta)$ — nó **phụ thuộc $x$** nhưng **không phụ thuộc $y$**.

Nên khi lấy hiệu hai phần thưởng cho hai câu trả lời của **cùng một** câu hỏi $x$, ta trừ $\beta\log Z(x)$ cho chính nó và được 0.

Nếu hai câu trả lời thuộc **hai** câu hỏi khác nhau $x_1 \ne x_2$, ta sẽ có $\beta\log Z(x_1) - \beta\log Z(x_2) \ne 0$, và số hạng ấy **không tính được**. Cả phương pháp sụp đổ.

Đây là lý do dữ liệu DPO bắt buộc phải là **cặp so sánh trong cùng một lời nhắc**, không phải các đánh giá rời rạc. Và đó cũng là hạn chế mà KTO (Mục 15.5) sinh ra để gỡ.

**(d) Hai tình huống RLHF vẫn hơn.**

**Thứ nhất — cần học trên phân phối mà mô hình đang thực sự sinh ra.** RLHF **lấy mẫu từ chính sách hiện tại** rồi chấm bằng mô hình thưởng, nên nó luôn học trên đúng những câu mà mô hình đang sinh lúc này. DPO chỉ học trên tập so sánh **cố định** đã thu thập trước. Khi huấn luyện tiến triển, chính sách đi xa dần khỏi phân phối đã sinh ra tập ấy, và tín hiệu học yếu đi — mô hình đang được dạy về những câu nó không còn sinh ra nữa.

**Thứ hai — mô hình thưởng là tài sản dùng lại được.** Nó dùng để chấm dữ liệu mới mà không cần người, để so sánh các mô hình khác nhau trên cùng một thang, để dò suy giảm chất lượng trong sản xuất, và để lọc dữ liệu huấn luyện. DPO không tạo ra thứ đó — nó cho một chính sách tốt hơn và không gì khác.

Với một đội đang vận hành lâu dài, món thứ hai đôi khi quan trọng hơn chính chất lượng mô hình, vì nó là công cụ **giám sát**.

## Bài 8
@meta chuong=9 | dang=Chẩn đoán | kho=Trung bình

**(a) VAE: tái dựng kém, KL gần 0 ở mọi chiều.**

**Vấn đề: sụp hậu nghiệm.** $q(z\mid x)$ đã trùng với tiên nghiệm ở mọi chiều, nghĩa là $z$ không mang thông tin gì về $x$, và bộ giải mã chỉ còn đoán giá trị trung bình. Đo được ở Mục 9.3: tại $\beta = 16$, cả 6 chiều chết và sai số tái dựng nhảy lên 15,28.

**Hai việc nên làm:**
1. **Giảm $\beta$.** Nếu đang dùng $\beta > 1$ thì hạ về 1 hoặc thấp hơn. Bảng ở Mục 9.3 cho thấy quan hệ rất trực tiếp giữa $\beta$ và số chiều sống.
2. **Hâm nóng KL** — bắt đầu với $\beta = 0$ rồi tăng dần lên giá trị đích trong vài nghìn bước đầu. Cách này cho bộ giải mã kịp học cách dùng $z$ trước khi số hạng KL bắt đầu ép. Nếu ép ngay từ đầu thì mô hình phát hiện ra rằng cách rẻ nhất để giảm mất mát là vứt bỏ $z$, và nó không bao giờ quay lại.

**(b) Tinh chỉnh trên 80 mẫu tệ hơn đóng băng.**

**Vấn đề: quá khớp.** Với 80 mẫu, tinh chỉnh toàn phần có quá nhiều tham số so với lượng dữ liệu. Bảng ở Mục 6.3 cho thấy đúng vùng này: dưới 150 mẫu, tinh chỉnh **không** thắng, và ở 20 mẫu nó còn thua.

**Hai việc nên làm:**
1. **Dùng LoRA thay vì tinh chỉnh toàn phần.** Nó cho phép sửa sâu hơn một lớp tuyến tính mà chỉ thêm dưới 1% tham số, nên nằm giữa hai cực và thường hợp nhất ở vùng dữ liệu này.
2. **Hạ tốc độ học và dừng sớm**, kèm mở đóng băng dần từ lớp trên xuống thay vì mở hết một lúc.

Và một việc thứ ba đáng làm trước cả hai: **kiểm tra xem 80 mẫu có đủ để đo tin cậy không.** Với 80 mẫu chia cho một tập kiểm định nhỏ, chênh lệch quan sát được có thể chỉ là nhiễu — đúng như hàng $n=20$ và $n=50$ trong bảng ở Mục 6.3 mà tài liệu từ chối kết luận.

**(c) Đóng băng, dữ liệu tăng 10 lần, độ chính xác chỉ nhích 0,02.**

**Vấn đề: trần chặn của đặc trưng đóng băng.** Đây chính là hiện tượng đo được ở Mục 6.3 (0,6220 → 0,6375 → 0,6452). Đặc trưng cố định đặt ra một giới hạn mà thêm dữ liệu không phá được, vì mô hình chỉ còn là hồi quy softmax trên chúng.

Nhìn theo khung ở [Mục 2.4 của *Mô hình & Kiến trúc*](models-ch02.html), đây là **thiên lệch cao**: sai số huấn luyện và kiểm định sát nhau và cùng phẳng.

**Hai việc nên làm:**
1. **Mở đóng băng**, ít nhất vài lớp trên cùng. Bảng ở Mục 6.3 cho thấy ở 8 000 mẫu, tinh chỉnh toàn phần đạt 0,8183 so với 0,6452 của đóng băng — chênh 0,17.
2. **Nếu không đủ bộ nhớ để tinh chỉnh toàn phần, dùng LoRA.** Nó cho phép sửa phần đặc trưng mà vẫn nằm trong ngân sách bộ nhớ.

Điều **không** nên làm: thu thập thêm dữ liệu. Bảng đã nói rõ nó không giúp gì trong cấu hình hiện tại.

**(d) GAN chạy 20 000 vòng, mất mát bộ sinh tăng đều. Có nên dừng?**

**Không — hoặc ít nhất, không thể kết luận từ thông tin đó.**

Lý do đã nêu ở Mục 10.4: mất mát của bộ sinh tăng có thể vì nó tệ đi, **mà cũng có thể vì bộ phân biệt vừa giỏi lên**. Hai mạng đuổi nhau, nên giá trị mất mát của một bên không đo được chất lượng tuyệt đối của bên đó. Đây là điều gây bực bội nhất khi làm việc với GAN, và nó khác hẳn mọi mô hình khác trong bộ giáo trình này.

**Hai việc nên làm:**
1. **Nhìn mẫu sinh ra bằng mắt**, lấy mẫu định kỳ theo số vòng. Đây nghe thô sơ nhưng là cách đánh giá đáng tin nhất, và là lý do mọi bài báo về GAN đều in một lưới mẫu.
2. **Dùng một chỉ số ngoài** — FID, hoặc với dữ liệu tổng hợp thì đếm số chế độ phủ được như Mục 10.2. Chúng đều là chỉ số **thay thế** với điểm mù riêng, nhưng vẫn hơn hẳn việc đọc mất mát.

Có một dấu hiệu **thật sự** đáng lo, và nó khác với mất mát tăng: nếu mất mát của bộ phân biệt về gần 0 và đứng yên, nghĩa là nó thắng tuyệt đối — khi ấy gradient của bộ sinh triệt tiêu đúng như cơ chế ở Mục 10.2, và huấn luyện đã chết thật.

## Bài 9
@meta chuong=14 | dang=Thiết kế | kho=Nâng cao

**(a) 3 000 đoạn hội thoại mẫu — chọn cách nào.**

**Chọn LoRA**, hạng 8 hoặc 16.

Lý do dựa trên số liệu:

- Bảng ở Mục 6.3 cho thấy từ 150 mẫu trở lên, tinh chỉnh **thắng rõ** so với đóng băng, và khoảng cách giãn dần. Với 3 000 mẫu ta đã ở sâu trong vùng đó, nên **chỉ nhắc lệnh hoặc chỉ đóng băng là bỏ phí dữ liệu**.
- Nhưng tinh chỉnh toàn phần mô hình 7B cần 96,4 GiB (Mục 7.1), tức vượt một GPU 80 GB.
- LoRA $r=8$ cần 32 MiB trạng thái tối ưu hoá thay vì 48,2 GiB, và cho phép sửa sâu hơn một lớp tuyến tính.

Nhiệm vụ ở đây — học văn phong và quy trình trả lời của bộ phận hỗ trợ — đúng là loại "nhấn mạnh lại thứ mô hình đã biết" mà giả thiết hạng thấp của LoRA phù hợp. Nó không đòi mô hình học kiến thức hoàn toàn mới.

**Nên làm trước khi huấn luyện bất cứ thứ gì:** thử chỉ nhắc lệnh với vài ví dụ trong ngữ cảnh, rồi đo. Nếu đã đủ tốt thì không cần huấn luyện gì cả. Đây là đường cơ sở mà nhiều đội bỏ qua.

**(b) 40 000 cặp so sánh — thiết kế bước tiếp theo.**

**Chọn DPO.**

Lý do:

1. **Quy mô đội phù hợp.** RLHF cần bốn mô hình trong bộ nhớ và rất nhiều công chỉnh. DPO cần hai, và hàm mất mát là entropy chéo nhị phân bình thường.
2. **Không mất gì về chất lượng lý thuyết.** Mục 15.3 kiểm chứng rằng hai phương pháp cho **cùng nghiệm** tới $4{,}2\times10^{-8}$ — DPO không phải xấp xỉ.
3. **40 000 cặp là một tập cố định,** đúng dạng dữ liệu mà DPO được thiết kế cho.

**Cách làm:** lấy mô hình sau bước (a) làm $\pi_{\text{ref}}$, chạy DPO với $\beta$ khoảng 0,1 tới 0,5, và **theo dõi KL giữa $\pi_\theta$ và $\pi_{\text{ref}}$** trong lúc huấn luyện. Nếu KL tăng quá nhanh thì mô hình đang đi xa khỏi hành vi đã kiểm chứng ở bước (a).

**Khi nào nên cân nhắc lại sang RLHF:** nếu sau này cần chấm dữ liệu mới liên tục, hoặc cần một công cụ đánh giá độc lập cho việc giám sát. Khi ấy mô hình thưởng có giá trị riêng ngoài việc huấn luyện — xem phần (d).

**(c) Vì sao ràng buộc KL không đủ để bảo đảm "không bao giờ hứa chi trả".**

Ba lý do, và cả ba đều phải nói rõ với bộ phận pháp chế:

1. **KL là một ràng buộc trung bình, không phải ràng buộc theo từng đầu ra.** $\mathrm{KL}(\pi\|\pi_{\text{ref}})$ nhỏ nghĩa là hai phân phối gần nhau **nói chung**. Nó hoàn toàn cho phép một số đầu ra cụ thể có xác suất tăng lên rất nhiều, miễn là phần còn lại bù lại. Bài 6(c) cho thấy bảo đảm tuyệt đối chỉ có khi $\pi_{\text{ref}}(y) = 0$ **đúng bằng 0**.

2. **Mô hình ngôn ngữ gần như không bao giờ gán xác suất đúng 0.** Softmax trên từ vựng luôn cho giá trị dương cho mọi token. Nên $\pi_{\text{ref}}(\text{"chúng tôi sẽ chi trả"}) > 0$, dù rất nhỏ — và do đó $\pi^*$ của nó cũng dương. Bảo đảm ở Bài 6(c) đúng về toán nhưng **không kích hoạt được** trong thực tế.

3. **Mô hình thưởng không biết về ràng buộc pháp lý** trừ khi dữ liệu so sánh đã dạy nó điều đó. Nếu người dán nhãn chỉ chấm theo "câu nào hữu ích hơn", thì một câu hứa chi trả có thể được chấm **cao hơn** vì nó làm khách hàng hài lòng.

**Cần thêm gì:** một lớp chặn **ngoài mô hình**, tức một bộ lọc cứng chạy trên đầu ra trước khi gửi đi. Nó có thể là luật chuỗi, một bộ phân loại riêng, hoặc cả hai. Điểm mấu chốt là nó **không phải một phần của mô hình được tối ưu**, nên không bị tối ưu hoá lách qua.

Nói thẳng với pháp chế: *"Huấn luyện làm cho hành vi ấy rất hiếm. Chỉ có bộ lọc đầu ra mới làm nó không thể xảy ra."* Đừng hứa rằng huấn luyện bảo đảm được điều gì.

**(d) Giám sát trong sản xuất.**

Bốn lớp, xếp theo độ trễ phát hiện từ nhanh tới chậm:

| Lớp | Đo gì | Vì sao |
|---|---|---|
| Chặn cứng | tỉ lệ đầu ra bị bộ lọc chặn | tăng đột ngột nghĩa là mô hình đang trôi về phía vùng cấm |
| Phân phối đầu vào | dịch chuyển của lời nhắc người dùng | [Chương 9 của *MLOps*](mlops-ch09.html) — câu hỏi đổi thì mô hình có thể không còn hợp |
| Chất lượng theo mẫu | chấm một mẫu nhỏ bằng người, hoặc bằng mô hình thưởng nếu có | tín hiệu trực tiếp nhất về chất lượng |
| Phản hồi người dùng | tỉ lệ chuyển sang nhân viên thật, đánh giá sao | trễ nhất nhưng đo đúng thứ quan trọng |

Hai điều đặc thù cho mô hình ngôn ngữ mà [Chương 13 của *MLOps*](mlops-ch13.html) nêu và nên nhắc lại ở đây:

- **Theo dõi KL giữa mô hình đang phục vụ và $\pi_{\text{ref}}$** trên lưu lượng thật. Nếu nó trôi thì có gì đó đã đổi mà ta chưa biết.
- **Cẩn thận với vòng phản hồi.** Nếu dùng chính dữ liệu do mô hình sinh ra để huấn luyện vòng sau, phân phối sẽ hẹp dần — đúng hiện tượng đo được ở [Mục 12.5 của *MLOps*](mlops-ch12.html), nơi hệ thống không ngẫu nhiên hoá vĩnh viễn chỉ nhìn thấy 9,5% danh mục.

## Bài 10
@meta chuong=10 | dang=Thí nghiệm | kho=Nâng cao

**(a) Ba thay đổi có thể làm sụp chế độ lộ ra.**

1. **Tăng số chiều của dữ liệu.** Đặt 8 chế độ trong không gian 50 chiều thay vì 2 chiều.
   *Cơ chế kỳ vọng:* ở chiều cao, các chế độ cách xa nhau hơn so với độ rộng của chúng, và bộ sinh khó "phủ" nhiều vùng rời rạc bằng một ánh xạ liên tục từ $z$. Việc đi từ chế độ này sang chế độ khác đòi đi qua vùng trống, mà vùng trống thì bị bộ phân biệt phạt.

2. **Làm bộ phân biệt mạnh hơn hẳn bộ sinh**, ví dụ cho nó nhiều lớp hơn và cập nhật 5 lần mỗi lần cập nhật bộ sinh.
   *Cơ chế kỳ vọng:* bộ phân biệt quá mạnh sẽ tìm ra ranh giới rất sắc, và bộ sinh chỉ còn cách chạy trốn vào một vùng hẹp mà nó đang tạm thắng. Đây là cách sụp chế độ hay được mô tả nhất trong tài liệu.

3. **Giảm số chiều của $z$ xuống rất thấp**, ví dụ $d_z = 1$.
   *Cơ chế kỳ vọng:* với $z$ một chiều, ảnh của bộ sinh là một **đường cong** trong mặt phẳng. Một đường cong liên tục đi qua 8 cụm rời rạc phải đi qua cả vùng trống giữa chúng, nên bộ sinh có động cơ chỉ phủ vài cụm gần nhau.

Lý do thứ ba là lý do tôi đặt cược cao nhất, vì nó là một ràng buộc **hình học cứng** chứ không phải một xu hướng của tối ưu hoá.

**(b) Chạy lại và báo cáo trung thực.**

Phần này để người học tự làm, nhưng yêu cầu quan trọng nhất nằm ở chữ "kể cả khi nó không ra như bạn đoán". Chính tài liệu này đã gặp đúng tình huống ấy: tôi vào thí nghiệm với dự định minh hoạ sụp chế độ, thử nhiều cấu hình, và **không** tái hiện được. Phần chữ ở Mục 10.3 được viết lại theo số liệu.

Nếu bạn chạy và cũng không tái hiện được, đó là một kết quả hợp lệ và nên ghi lại như vậy. Một thí nghiệm bác bỏ giả thuyết của chính mình có giá trị hơn một thí nghiệm được chỉnh cho tới khi ra kết quả mong muốn.

**(c) Một chỉ số phủ tốt hơn.**

Cách đếm hiện tại — số cụm có ít nhất một điểm rơi vào — là một chỉ số **nhị phân và rất dễ dãi**: một cụm nhận 1 điểm cũng được tính ngang một cụm nhận 1 000 điểm.

Chỉ số tốt hơn: gọi $p_i$ là tỉ lệ điểm sinh ra rơi vào cụm $i$ (trên tổng số điểm rơi vào cụm nào đó), và $q_i = 1/8$ là tỉ lệ thật. Rồi đo khoảng cách giữa hai phân phối:

$$\text{KL}(p\,\|\,q) \qquad\text{hoặc}\qquad \text{số chế độ hiệu dụng} = \exp\big(H(p)\big) = \exp\Big(-\sum_i p_i\log p_i\Big).$$

**Số chế độ hiệu dụng** đặc biệt dễ đọc: nó bằng 8 khi phân bố đều hoàn hảo, và bằng 1 khi tất cả dồn vào một cụm. Nó cho một con số liên tục trong khoảng $[1, 8]$ thay vì một số nguyên nhị phân.

Đây chính là entropy mũ, cùng đại lượng dùng để đo các chiến lược giải mã ở [Mục 11.3 của *Mô hình & Kiến trúc*](models-ch11.html) — một công cụ, hai lĩnh vực.

**(d) Vì sao "phủ hết chế độ" chưa đủ.**

Vì phủ hết chỉ nói rằng bộ sinh **chạm tới** mọi vùng, không nói nó phân bố khối lượng xác suất cho đúng.

Con số 15,3% ở Mục 10.3 cho thấy đúng điều đó: chỉ 15,3% số điểm sinh ra rơi vào phạm vi một cụm, tức **gần 85% nằm rải rác giữa các cụm** — ở những chỗ mà dữ liệu thật gần như không có điểm nào. Bộ sinh đã tìm đúng vị trí 8 chế độ nhưng phân phối thì sai nặng.

Đây là một cách hỏng khác, nhẹ hơn sụp chế độ nhưng vẫn là hỏng, và nó cho thấy vì sao cần cả hai loại chỉ số: một đo **phủ** (có chạm tới mọi chế độ không) và một đo **chính xác** (có bao nhiêu mẫu thật sự nằm trên đa tạp dữ liệu). Chỉ số FID dùng trong thực tế cố gắng nắm cả hai cùng lúc, và đó là lý do nó phức tạp hơn cả hai.
