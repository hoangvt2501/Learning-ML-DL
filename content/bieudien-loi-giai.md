# Lời giải chi tiết — Biểu diễn, mô hình sinh và căn chỉnh

Mỗi mục ứng với một bài trong Chương 16. Dòng `@meta` được script build đọc để gắn nhãn
chương, dạng bài và độ khó; nó không hiện ra trên trang.

## Bài 1
@meta chuong=2 | dang=Suy luận | kho=Trung bình

**(a) Hàm mục tiêu và đạo hàm.** Với cặp $(i, j)$, đặt $s = \langle w_i, c_j\rangle$. Phần hàm mục tiêu kỳ vọng phụ thuộc $s$ gồm phần cặp thật, có trọng số là số lần đồng hiện, và phần cặp âm, có trọng số là số lần lấy mẫu âm sinh ra cặp đó:

$$\ell(s) = \#(i,j)\,\log\sigma(s) + k\,\#(i)\,P_n(j)\,\log\sigma(-s).$$

Dùng $\frac{d}{ds}\log\sigma(s) = 1 - \sigma(s)$ và $\frac{d}{ds}\log\sigma(-s) = -\sigma(s)$:

$$\ell'(s) = \#(i,j)\,\big(1 - \sigma(s)\big) - k\,\#(i)\,P_n(j)\,\sigma(s) = 0.$$

**(b) Nghiệm.** Chuyển vế được $\frac{\sigma(s)}{1 - \sigma(s)} = \frac{\#(i,j)}{k\,\#(i)\,P_n(j)}$. Vì $\sigma(s) = \frac{e^s}{1 + e^s}$ nên vế trái bằng $e^s$. Thay $P_n(j) = \#(j)/N$:

$$\begin{gathered} e^{s} = \frac{\#(i,j)\,N}{k\,\#(i)\,\#(j)} \\ \Longrightarrow\; s = \log\frac{\#(i,j)\,N}{\#(i)\,\#(j)} - \log k = \mathrm{PMI}(i,j) - \log k. \end{gathered}$$

**(c) Tăng $k$ từ 1 lên 15.** Ma trận đích dịch xuống $\log 15 \approx 2{,}71$. Các cặp không liên quan vốn có PMI khoảng 0 trở thành âm, và sau khi cắt phần âm thì bằng 0: ma trận đích thưa hơn nhiều. Trên kho ngữ liệu thật, điều này lọc bỏ các cặp đồng hiện yếu vốn ước lượng rất nhiễu. Trên kho ngữ liệu 20 từ của Mục 2.2, PMI lớn nhất chỉ là 1,19, nên với $k = 15$ mọi phần tử đều âm và bị cắt về 0, không còn gì để phân rã, nên thí nghiệm dùng $k = 1$.

**(d) Hạng đầy đủ và hạng thấp.** Kết quả ở (b) là điều kiện cho từng phần tử của ma trận $WC^\top$. Nếu embedding có đủ $V = 20$ chiều, $WC^\top$ có thể là bất kỳ ma trận $20 \times 20$ nào, nên tối ưu đạt được điều kiện ở mọi phần tử: Mục 2.6 đo được tương quan 0,9995 và sai lệch trung bình 0,0106.

Nếu embedding chỉ có 8 chiều, $WC^\top$ có hạng không quá 8, và nói chung không thể thoả điều kiện ở mọi phần tử; tối ưu phải chọn một xấp xỉ. Mục 2.6 đo được tương quan chỉ 0,3727. Xấp xỉ đó phụ thuộc vào trọng số của hàm mục tiêu: skip-gram đặt trọng số theo tần suất của các cặp, khác với SVD đặt trọng số như nhau. Hệ quả đo được: embedding skip-gram hạng 8 chỉ giải đúng 20,4% phép loại suy, trong khi SVD trên PPMI cùng hạng giải đúng 100%. Giới hạn số chiều là điều kiện để có embedding gọn và tổng quát hoá được, nhưng không tự bảo đảm giữ đúng cấu trúc cần thiết; cấu trúc nào được giữ lại tuỳ thuộc vào hàm mục tiêu và cách tối ưu.

## Bài 2
@meta chuong=3 | dang=Chẩn đoán | kho=Cơ bản

**(a) Chẩn đoán: tính bất đẳng hướng.** Cosine đo góc, và góc chỉ phản ánh độ giống nhau khi đám mây điểm phân bố quanh gốc toạ độ. Trạng thái ẩn của Transformer thường tập trung trong một vùng hẹp lệch khỏi gốc, nên mọi vector có chung một thành phần lớn theo cùng một hướng. Tích vô hướng giữa hai vector bất kỳ bị thành phần chung đó chi phối, và cosine phản ánh hướng chung của cả tập thay vì quan hệ giữa hai câu. Đây không phải lỗi cài đặt mà là đặc điểm của biểu diễn từ một mô hình không được huấn luyện để các câu cùng nghĩa gần nhau.

**(b) Cách sửa rẻ nhất: trừ vector trung bình.** Tính $\bar v$ trên một mẫu đại diện, vài nghìn câu là đủ, rồi so sánh $u - \bar v$ thay cho $u$. Ở Mục 3.2:

| Mức lệch tâm | Cosine trung bình trước | Sau khi trừ trung bình |
|---|---|---|
| Nhẹ | 0,3899 | $-0{,}0011$ |
| Mạnh | 0,8724 | $-0{,}0011$ |

Chi phí chỉ là một phép trừ vector cho mỗi truy vấn.

**(c) Vì sao không nâng ngưỡng lên 0,93.**

1. Cách này xử lý triệu chứng, không xử lý nguyên nhân: phần thông tin hữu ích vẫn bị thành phần chung che lấp, và thứ hạng giữa các kết quả vẫn bị nó chi phối.
2. Ngưỡng không ổn định: vector trung bình thay đổi theo miền dữ liệu, theo mô hình, cả theo độ dài câu. Ngưỡng chỉnh cho dữ liệu hôm nay sẽ sai khi dữ liệu thay đổi.
3. Dải giá trị quá hẹp: mọi điểm nằm trong $[0{,}82;\ 0{,}94]$, tức toàn bộ thông tin nằm trong một dải rộng 0,12, nên một chênh lệch nhỏ do nhiễu cũng đảo được thứ hạng. Sau khi trừ trung bình, dải này giãn ra gần hết khoảng $[-1, 1]$.

**(d) Cách sửa triệt để: huấn luyện với mục tiêu tương phản.** Huấn luyện embedding sao cho cặp câu liên quan gần nhau và cặp không liên quan xa nhau, với hàm mất mát InfoNCE và các câu khác trong lô làm mẫu âm (Mục 4.2). Đổi lại, cách này cần dữ liệu cặp (câu hỏi và câu trả lời, câu và bản dịch, hai cách diễn đạt của cùng một ý), và cần lô lớn để có nhiều mẫu âm. Thứ tự nên thử: trừ trung bình trước, làm trắng nếu chưa đủ, và chỉ huấn luyện tương phản khi hai cách kia không đạt yêu cầu; hoặc dùng ngay một mô hình embedding đã được huấn luyện tương phản sẵn.

## Bài 3
@meta chuong=7 | dang=Tính tay | kho=Cơ bản

**(a) Số tham số.** Mỗi lớp có hai ma trận được gắn LoRA ($W_Q$ và $W_V$), mỗi ma trận có $2dr$ tham số:

$$\begin{aligned} N_{\text{LoRA}} &= L \times 2 \times 2dr = 4Ldr \\ &= 4 \times 32 \times 4\,096 \times r = 524\,288\,r. \end{aligned}$$

| $r$ | Tham số huấn luyện | Tỉ lệ so với 6 476 005 376 |
|---|---|---|
| 4 | 2 097 152 | 0,0324% |
| 8 | 4 194 304 | 0,0648% |
| 16 | 8 388 608 | 0,1295% |
| 64 | 33 554 432 | 0,5181% |

**(b) Trạng thái Adam.** Adam giữ hai trạng thái cho mỗi tham số, mỗi trạng thái 4 byte ở FP32:

$$4\,194\,304 \times 2 \times 4 = 33\,554\,432 \text{ byte} = 32 \text{ MiB}.$$

Với tinh chỉnh toàn phần: $6\,476\,005\,376 \times 2 \times 4 = 51\,808\,043\,008$ byte, tức 48,2 GiB. Tỉ lệ khoảng 1 540 lần. Cộng thêm trọng số (24,1 GiB) và gradient (24,1 GiB) ở FP32, tinh chỉnh toàn phần cần 96,4 GiB, vượt một GPU 80 GB.

**(c) Tỉ số $2r/d$ theo $d$.**

| $d$ | $2r/d$ với $r = 8$ |
|---|---|
| 768 | 2,083% |
| 1 280 | 1,250% |
| 4 096 | 0,391% |
| 8 192 | 0,195% |

Tỉ số tỉ lệ nghịch với $d$: số tham số của một lớp tăng theo $d^2$, còn của LoRA chỉ tăng theo $d$. Mô hình càng lớn, phần điều chỉnh hạng thấp càng nhỏ so với bản thân mô hình, nên LoRA càng tiết kiệm.

**(d) Lưu trữ cho 50 khách hàng.** Tinh chỉnh toàn phần: $50 \times 13$ GB $= 650$ GB. LoRA: một bản gốc dùng chung cộng 50 adapter, $13 + 50 \times 0{,}016 = 13{,}8$ GB, nhỏ hơn khoảng 47 lần. Lợi ích khi phục vụ còn lớn hơn: giữ một bản mô hình gốc trên GPU và chọn adapter theo từng yêu cầu, trong khi với tinh chỉnh toàn phần, mỗi khách hàng cần một bản mô hình riêng trên GPU hoặc phải nạp lại 13 GB mỗi lần đổi khách hàng.

## Bài 4
@meta chuong=11 | dang=Suy luận | kho=Trung bình

**(a) Quy nạp.** Với $t = 1$: $q(x_1 \mid x_0) = \mathcal{N}(\sqrt{\alpha_1}\,x_0, \beta_1 I)$, và $\bar\alpha_1 = \alpha_1$, $1 - \bar\alpha_1 = \beta_1$, nên mệnh đề đúng.

Giả sử $x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,x_0 + \sqrt{1 - \bar\alpha_{t-1}}\,\varepsilon_1$ với $\varepsilon_1 \sim \mathcal{N}(0, I)$. Theo định nghĩa một bước, $x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{\beta_t}\,\varepsilon_2$ với $\varepsilon_2 \sim \mathcal{N}(0, I)$ độc lập với $\varepsilon_1$. Thay vào:

$$x_t = \sqrt{\alpha_t\bar\alpha_{t-1}}\,x_0 + \sqrt{\alpha_t(1 - \bar\alpha_{t-1})}\,\varepsilon_1 + \sqrt{\beta_t}\,\varepsilon_2.$$

Hệ số của $x_0$ là $\sqrt{\bar\alpha_t}$. Hai số hạng nhiễu là biến Gauss độc lập có trung bình 0, nên tổng của chúng là biến Gauss có phương sai bằng tổng các phương sai:

$$\alpha_t(1 - \bar\alpha_{t-1}) + \beta_t = \alpha_t - \bar\alpha_t + 1 - \alpha_t = 1 - \bar\alpha_t.$$

Vậy $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\varepsilon$ với $\varepsilon \sim \mathcal{N}(0, I)$.

**(b) Vì sao cần dạng đóng.** Huấn luyện cần lấy $x_t$ với $t$ ngẫu nhiên trong $\{1, \dots, 1\,000\}$, hàng triệu lần. Nếu phải mô phỏng từng bước, trung bình mỗi mẫu huấn luyện tốn khoảng 500 bước, tức chậm hơn khoảng 500 lần. Với dạng đóng, rút một $\varepsilon$ và tính $x_t$ trong một phép tính với mọi $t$. Tính chất này có được vì mỗi bước là một phép biến đổi Gauss tuyến tính; với phép biến đổi phi tuyến thì không gộp được như vậy.

**(c) Tín hiệu còn lại.** Biên độ tín hiệu là $\sqrt{\bar\alpha}$: $\sqrt{4{,}0 \times 10^{-5}} \approx 6{,}3 \times 10^{-3}$, tức khoảng 0,63%. Con số này quan trọng vì khi sinh mẫu, quá trình bắt đầu từ $\mathcal{N}(0, I)$, và điều đó chỉ đúng nếu $q(x_T \mid x_0)$ gần $\mathcal{N}(0, I)$ với mọi $x_0$. Nếu lịch nhiễu dừng sớm, chẳng hạn $\bar\alpha_T = 0{,}1$, thì $x_T$ còn giữ khoảng 32% biên độ của $x_0$, và bắt đầu sinh từ nhiễu thuần tuý là sai phân phối.

**(d) Chi tiết và bố cục.** Ở $t$ nhỏ, SNR rất cao: ảnh gần như nguyên vẹn, chỉ có một lớp nhiễu mỏng. Để dự đoán lớp nhiễu đó, mô hình phải phân biệt nhiễu với kết cấu thật ở mức điểm ảnh; bố cục thì vẫn còn nguyên nên không cần dự đoán. Ở $t$ lớn, SNR rất thấp: gần như chỉ còn nhiễu, các chi tiết đã mất, chỉ các thành phần biên độ lớn như bố cục, vùng sáng tối và màu tổng thể còn để lại dấu vết. Khi sinh mẫu đi ngược từ $t = T$ về 0, mô hình vì vậy quyết định bố cục trước rồi mới thêm chi tiết; trình tự này là hệ quả của lịch nhiễu, không được lập trình riêng.

## Bài 5
@meta chuong=10 | dang=Suy luận | kho=Trung bình

**(a) Hai đạo hàm.** Với $D = \sigma(s)$ và $\sigma'(s) = \sigma(s)(1 - \sigma(s))$:

$$\begin{aligned} \frac{\partial}{\partial s}\log\big(1 - \sigma(s)\big) &= \frac{-\sigma(s)(1 - \sigma(s))}{1 - \sigma(s)} = -\sigma(s), \\ \frac{\partial}{\partial s}\log\sigma(s) &= \frac{\sigma(s)(1 - \sigma(s))}{\sigma(s)} = 1 - \sigma(s). \end{aligned}$$

**(b) Lúc bắt đầu.** Bộ phân biệt dễ dàng nhận ra mẫu giả nên $\sigma(s) \approx 0$. Dạng gốc cho gradient cỡ $-\sigma(s) \approx 0$; dạng không bão hoà cho $1 - \sigma(s) \approx 1$.

**(c) Hậu quả đo được.**

| Hàm mất mát của bộ sinh | Số cụm được phủ (4 lần khởi tạo) | Tỉ lệ điểm nằm trong cụm |
|---|---|---|
| Dạng gốc | 0, 0, 0, 0 | 0,0% |
| Dạng không bão hoà | 8, 8, 8, 8 | 15,3% |

Dạng gốc không học được gì ở cả bốn lần. Cơ chế là một vòng luẩn quẩn: bộ sinh kém, bộ phân biệt tự tin, gradient của bộ sinh gần 0, bộ sinh không cải thiện. Vòng này khép lại ngay từ bước đầu, vì lúc đầu bộ sinh luôn kém. Dạng không bão hoà phá vòng này vì gradient lớn nhất đúng khi bộ sinh kém nhất.

**(d) Nhận định tổng quát.** Hai hàm mất mát có cùng điểm tối ưu vẫn có thể cho quá trình huấn luyện rất khác nhau, vì thuật toán không nhảy thẳng tới điểm tối ưu mà đi từng bước theo gradient; hình dạng của gradient dọc đường đi mới quyết định việc học.

Ví dụ khác: [Chương 6 của *Nền tảng*](nentang-ch06.html) so sánh hàm mất mát của perceptron, hinge và logistic. Cả ba đều là chặn trên lồi của mất mát 0–1, nhưng mất mát perceptron bằng 0 ngay khi một điểm vừa được phân loại đúng, nên thuật toán ngừng điều chỉnh điểm đó dù nó nằm sát biên. Trên dữ liệu không tách được, Mục 6.2 của *Nền tảng* đo được 17 977 lần phân loại sai sau 2 000 lượt quét mà không hội tụ.

## Bài 6
@meta chuong=14 | dang=Tính tay | kho=Nâng cao

**(a) Nghiệm dạng đóng.** Với $x$ cố định, bài toán là

$$\max_\pi \sum_y\pi(y)\,r(y) - \beta\sum_y\pi(y)\log\frac{\pi(y)}{\pi_{\text{ref}}(y)} \quad\text{với}\quad \sum_y\pi(y) = 1.$$

Hàm Lagrange với nhân tử $\lambda$:

$$\mathcal{L} = \sum_y\pi(y)\,r(y) - \beta\sum_y\pi(y)\log\frac{\pi(y)}{\pi_{\text{ref}}(y)} + \lambda\Big(\sum_y\pi(y) - 1\Big).$$

Dùng $\frac{\partial}{\partial\pi}\big[\pi\log\frac{\pi}{\pi_{\text{ref}}}\big] = \log\frac{\pi}{\pi_{\text{ref}}} + 1$, cho đạo hàm theo $\pi(y)$ bằng 0:

$$\begin{gathered} r(y) - \beta\Big(\log\frac{\pi(y)}{\pi_{\text{ref}}(y)} + 1\Big) + \lambda = 0 \\ \Longrightarrow\; \pi(y) = \pi_{\text{ref}}(y)\exp\Big(\frac{r(y)}{\beta}\Big)\,e^{\lambda/\beta - 1}. \end{gathered}$$

Thừa số cuối là hằng số, xác định bởi ràng buộc tổng bằng 1, nên

$$\pi^*(y) = \frac{1}{Z}\,\pi_{\text{ref}}(y)\exp\Big(\frac{r(y)}{\beta}\Big), \qquad Z = \sum_y\pi_{\text{ref}}(y)\,e^{r(y)/\beta}.$$

**(b) Hai giới hạn.** Khi $\beta \to 0$, $\exp(r/\beta)$ bị chi phối hoàn toàn bởi câu trả lời có phần thưởng lớn nhất, nên chính sách dồn toàn bộ xác suất vào câu trả lời đó, bỏ qua $\pi_{\text{ref}}$ (miễn là $\pi_{\text{ref}}$ của nó dương). Khi $\beta \to \infty$, $\exp(r/\beta) \to 1$ và $\pi^* \to \pi_{\text{ref}}$: mô hình giữ nguyên, không học gì từ phần thưởng. Mục 14.4 cho thấy hai xu hướng này: ở $\beta = 0{,}05$, KL bằng 4,74 và phần thưởng kỳ vọng 2,44; ở $\beta = 5$, KL chỉ 0,017 và phần thưởng kỳ vọng $-0{,}35$.

**(c) Khi $\pi_{\text{ref}}(y_0) = 0$.** $\pi^*(y_0) = 0 \cdot \exp(r(y_0)/\beta)/Z = 0$, bất kể $r(y_0)$ lớn tới đâu. Về mặt an toàn, điều này có nghĩa là những gì mô hình tham chiếu coi là không thể thì vẫn không thể sau khi tối ưu, dù mô hình thưởng có sai lệch thế nào.

Nhưng bảo đảm này hầu như không áp dụng được trong thực tế: một mô hình ngôn ngữ dùng softmax gán xác suất dương, dù rất nhỏ, cho mọi chuỗi token, nên $\pi_{\text{ref}}(y) > 0$ với mọi $y$. Ràng buộc KL chỉ làm những đầu ra đó ít xảy ra hơn, không loại bỏ chúng. Bài 9(c) bàn tiếp hệ quả này.

**(d) $\beta$ là một vị trí trên đường đánh đổi.**

| $\beta$ | KL | Phần thưởng kỳ vọng |
|---|---|---|
| 0,05 | 4,741 | 2,438 |
| 0,20 | 4,658 | 2,424 |
| 1,00 | 0,417 | 0,303 |
| 5,00 | 0,017 | $-0{,}350$ |

KL và phần thưởng tăng giảm cùng nhau: muốn phần thưởng cao hơn thì phải chấp nhận đi xa $\pi_{\text{ref}}$ hơn. Không có giá trị $\beta$ nào làm cả hai cùng tốt, nên câu hỏi "$\beta$ bằng bao nhiêu là đúng" không có câu trả lời kỹ thuật chung. Câu hỏi thực tế là chấp nhận lệch khỏi mô hình tham chiếu bao nhiêu để đổi lấy bao nhiêu phần thưởng, và câu trả lời phụ thuộc mức tin cậy vào mô hình thưởng. Cách đọc giống đường ROC ([Mục 8.3 của *Nền tảng*](nentang-ch08.html)): không có điểm tốt nhất chung, chỉ có một đường đánh đổi.

## Bài 7
@meta chuong=15 | dang=Suy luận | kho=Nâng cao

**(a) Đảo ngược nghiệm dạng đóng.** Lấy logarit hai vế của $\pi^*(y \mid x) = \pi_{\text{ref}}(y \mid x)\exp(r(x, y)/\beta)/Z(x)$:

$$\begin{gathered} \log\pi^*(y \mid x) = \log\pi_{\text{ref}}(y \mid x) + \frac{r(x, y)}{\beta} - \log Z(x) \\ \Longrightarrow\; r(x, y) = \beta\log\frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta\log Z(x). \end{gathered}$$

Với $\pi_{\text{ref}}$ và $\beta$ cho trước, mỗi chính sách ứng với một hàm thưởng (sai khác một hàm chỉ phụ thuộc $x$), nên tham số hoá chính sách cũng là tham số hoá hàm thưởng.

**(b) Thay vào Bradley–Terry.** Mô hình Bradley–Terry dùng hiệu hai phần thưởng:

$$\begin{aligned} r(x, y_w) - r(x, y_l) &= \Big[\beta\log\frac{\pi(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \beta\log Z(x)\Big] \\ &\quad - \Big[\beta\log\frac{\pi(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \beta\log Z(x)\Big]. \end{aligned}$$

Hai số hạng $\beta\log Z(x)$ ở cuối mỗi ngoặc bằng nhau và triệt tiêu. Kết quả là hàm mất mát DPO:

$$\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta\log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right],$$

trong đó mọi đại lượng đều tính được bằng một lượt xuôi qua $\pi_\theta$ và $\pi_{\text{ref}}$ trên hai câu trả lời.

**(c) Vì sao cần cùng một câu hỏi.** $Z(x) = \sum_y\pi_{\text{ref}}(y \mid x)\exp(r(x, y)/\beta)$ phụ thuộc $x$ nhưng không phụ thuộc $y$. Với hai câu trả lời cho cùng câu hỏi $x$, hai số hạng $\beta\log Z(x)$ bằng nhau. Nếu hai câu trả lời thuộc hai câu hỏi khác nhau $x_1 \ne x_2$, hiệu $\beta\log Z(x_1) - \beta\log Z(x_2)$ khác 0 và không tính được. Vì vậy dữ liệu của DPO phải là các cặp so sánh cho cùng một câu hỏi; KTO (Mục 15.5) được đề xuất để bỏ yêu cầu này.

**(d) Hai tình huống RLHF có thể tốt hơn.**

1. **Cần học trên các câu trả lời mà chính sách hiện tại sinh ra.** RLHF lấy mẫu từ chính sách đang học rồi chấm bằng mô hình thưởng, nên luôn học trên đúng phân phối mô hình đang sinh ra. DPO học trên một tập so sánh cố định; khi chính sách đi xa khỏi phân phối đã sinh ra tập đó, mô hình được dạy về những câu trả lời nó không còn sinh ra nữa, và tín hiệu học yếu đi.
2. **Cần một mô hình thưởng dùng lại được.** Mô hình thưởng có thể chấm dữ liệu mới mà không cần người, so sánh các phiên bản mô hình trên cùng một thang, phát hiện suy giảm chất lượng khi vận hành, và lọc dữ liệu huấn luyện. DPO không tạo ra thành phần này.

## Bài 8
@meta chuong=9 | dang=Chẩn đoán | kho=Trung bình

**(a) VAE: tái dựng kém, KL gần 0 ở mọi chiều.** Đây là sụp hậu nghiệm: hậu nghiệm xấp xỉ trùng tiên nghiệm ở mọi chiều, $z$ không mang thông tin về $x$, và bộ giải mã chỉ dự đoán giá trị trung bình. Mục 9.3 đo được hiện tượng này ở $\beta = 16$: mọi chiều có KL bằng 0 và sai số tái dựng tăng lên 15,28. Hai việc nên làm:

1. **Giảm $\beta$**, ví dụ về 1 hoặc nhỏ hơn; Mục 9.3 cho thấy số chiều mang thông tin phụ thuộc trực tiếp vào $\beta$.
2. **Tăng dần $\beta$ từ 0 (KL annealing)** trong những bước đầu, để bộ giải mã học cách dùng $z$ trước khi số hạng KL kéo hậu nghiệm về tiên nghiệm (Bowman và cộng sự, 2016). Nếu bộ giải mã quá mạnh, như một mô hình tự hồi quy, có thể cần giảm dung lượng của nó.

**(b) Tinh chỉnh trên 80 mẫu kém hơn đóng băng.** Với 80 mẫu, tinh chỉnh toàn phần có quá nhiều tham số so với dữ liệu, nên dễ overfitting; Mục 6.3 cho thấy dưới 150 mẫu, tinh chỉnh toàn phần không vượt được các cách khác. Hai việc nên làm:

1. **Dùng LoRA hạng nhỏ** thay cho tinh chỉnh toàn phần: điều chỉnh sâu hơn một lớp tuyến tính mà chỉ thêm rất ít tham số.
2. **Giảm tốc độ học, dừng sớm theo tập xác thực**, và mở đóng băng dần từ các lớp trên.

Trước đó nên kiểm tra xem chênh lệch có đáng tin không: với 80 mẫu, tập xác thực rất nhỏ, và chênh lệch quan sát được có thể chỉ là dao động giữa các lần chạy, như hai hàng đầu của bảng ở Mục 6.3.

**(c) Đóng băng, dữ liệu tăng mười lần, độ chính xác chỉ tăng 0,02.** Đây là giới hạn của đặc trưng đóng băng, giống kết quả đo ở Mục 6.3 (0,622; 0,638; 0,645): khi đặc trưng cố định, mô hình chỉ còn là hồi quy softmax trên các đặc trưng đó. Theo [Mục 2.4 của *Học sâu*](models-ch02.html), đây là tình trạng độ chệch cao: sai số huấn luyện và sai số xác thực gần nhau và cùng đã nằm ngang. Hai việc nên làm:

1. **Cho phép cập nhật các lớp phía dưới**, ít nhất vài lớp trên cùng; ở Mục 6.3, với 8 000 mẫu, tinh chỉnh toàn phần đạt 0,818 so với 0,645.
2. **Dùng LoRA** nếu không đủ bộ nhớ để tinh chỉnh toàn phần.

Không nên thu thập thêm dữ liệu cho cấu hình hiện tại, vì đường cong đã cho thấy dữ liệu không giúp được.

**(d) GAN: mất mát của bộ sinh tăng đều sau 20 000 vòng.** Không đủ thông tin để kết luận nên dừng. Mất mát của bộ sinh tăng có thể vì bộ sinh kém đi, cũng có thể vì bộ phân biệt vừa tốt lên (Mục 10.4). Hai việc nên làm:

1. **Xem trực tiếp mẫu sinh ra**, định kỳ theo số vòng; đây là cách đánh giá thô nhưng đáng tin.
2. **Dùng một chỉ số bên ngoài** như FID, hoặc với dữ liệu mô phỏng thì đếm số chế độ được phủ và tỉ lệ điểm nằm trong các cụm như Mục 10.2.

Một dấu hiệu thật sự đáng lo là mất mát của bộ phân biệt về gần 0 và đứng yên: bộ phân biệt thắng hoàn toàn, gradient của bộ sinh triệt tiêu như cơ chế ở Mục 10.2, và quá trình huấn luyện đã hỏng.

## Bài 9
@meta chuong=14 | dang=Thiết kế | kho=Nâng cao

**(a) 3 000 đoạn hội thoại mẫu.** Trước hết nên thử chỉ viết prompt với vài ví dụ trong ngữ cảnh và đo trên một bộ đánh giá; nếu đã đủ tốt thì không cần huấn luyện. Nếu chưa đủ, chọn LoRA hạng 8 hoặc 16:

- Với 3 000 mẫu, dữ liệu nằm trong vùng mà ở Mục 6.3 tinh chỉnh vượt rõ đóng băng, nên chỉ học lớp cuối là bỏ phí dữ liệu.
- Tinh chỉnh toàn phần mô hình 7B với Adam ở FP32 cần khoảng 96 GiB (Mục 7.1), vượt một GPU 80 GB; LoRA $r = 8$ chỉ cần 32 MiB cho trạng thái bộ tối ưu.
- Nhiệm vụ, học văn phong và quy trình trả lời của bộ phận hỗ trợ, là loại điều chỉnh mà giả định hạng thấp của LoRA phù hợp: nhấn mạnh lại những gì mô hình đã biết thay vì học kiến thức mới.

**(b) 40 000 cặp so sánh.** Chọn DPO:

1. DPO cần 2 mô hình trong bộ nhớ thay vì 4, hàm mất mát là cross-entropy nhị phân, ít siêu tham số và ổn định hơn PPO.
2. DPO có cùng nghiệm tối ưu với RLHF (Mục 15.3), nên về lý thuyết không mất gì.
3. 40 000 cặp là một tập so sánh cố định, đúng dạng dữ liệu DPO được thiết kế cho.

Cách làm: lấy mô hình sau bước (a) làm $\pi_{\text{ref}}$, chạy DPO với $\beta$ trong khoảng 0,1 tới 0,5, theo dõi KL giữa $\pi_\theta$ và $\pi_{\text{ref}}$ và chất lượng trên một bộ đánh giá giữ riêng. Nên cân nhắc RLHF về sau nếu cần học liên tục trên các câu trả lời mô hình đang sinh ra, hoặc cần một mô hình thưởng cho việc đánh giá và giám sát.

**(c) Ràng buộc KL không bảo đảm "không bao giờ hứa chi trả".**

1. **KL là ràng buộc trung bình, không phải ràng buộc cho từng đầu ra.** KL nhỏ nghĩa là hai phân phối gần nhau nói chung, vẫn cho phép một số đầu ra cụ thể tăng xác suất đáng kể.
2. **Mô hình ngôn ngữ không gán xác suất bằng 0 cho chuỗi nào.** Bảo đảm ở Bài 6(c) chỉ áp dụng khi $\pi_{\text{ref}}(y) = 0$ đúng bằng 0, điều không xảy ra với softmax.
3. **Mô hình thưởng không biết ràng buộc pháp lý** nếu dữ liệu so sánh không thể hiện nó. Nếu người đánh giá chỉ chọn câu trả lời "hữu ích hơn", một câu hứa chi trả có thể được chấm cao hơn vì làm khách hàng hài lòng.

Cần thêm một lớp kiểm tra nằm ngoài mô hình: một bộ lọc chạy trên đầu ra trước khi gửi cho khách hàng, gồm quy tắc và một bộ phân loại riêng ([Mục 11.5 của *Ứng dụng LLM*](ungdung-ch11.html)). Bộ lọc không phải một phần của mô hình được tối ưu, nên không bị quá trình tối ưu lách qua. Huấn luyện làm hành vi này hiếm đi; chỉ bộ lọc đầu ra mới chặn được nó, và nên nói rõ điều đó với bộ phận pháp chế.

**(d) Giám sát khi vận hành.** Bốn lớp, xếp theo độ trễ phát hiện từ nhanh tới chậm:

| Lớp | Đo gì | Lý do |
|---|---|---|
| Bộ lọc đầu ra | tỉ lệ đầu ra bị chặn | tăng đột ngột cho thấy mô hình hoặc loại câu hỏi đã thay đổi |
| Phân phối đầu vào | dịch chuyển của câu hỏi người dùng ([Chương 9 của *MLOps*](mlops-ch09.html)) | câu hỏi thay đổi thì chất lượng có thể thay đổi |
| Chất lượng theo mẫu | chấm một mẫu nhỏ bằng người hoặc bằng mô hình giám khảo | tín hiệu trực tiếp nhất về chất lượng |
| Phản hồi người dùng | tỉ lệ chuyển sang nhân viên thật, đánh giá sau hội thoại | đến muộn nhất nhưng đo đúng điều quan trọng |

Thêm hai điểm riêng cho mô hình ngôn ngữ ([Chương 13 của *MLOps*](mlops-ch13.html)): kiểm tra định kỳ trên một bộ câu hỏi cố định để phát hiện thay đổi hành vi khi đổi phiên bản mô hình, và cẩn thận khi dùng dữ liệu do chính mô hình sinh ra để huấn luyện vòng sau, vì phân phối có thể hẹp dần, như hiện tượng vòng phản hồi đo được ở [Mục 12.3 của *MLOps*](mlops-ch12.html).

## Bài 10
@meta chuong=10 | dang=Thí nghiệm | kho=Nâng cao

**(a) Ba thay đổi có thể làm sụp chế độ xuất hiện.**

1. **Tăng số chiều của dữ liệu**, ví dụ đặt 8 cụm trong không gian 50 chiều. Cơ chế dự kiến: ở số chiều cao, các cụm cách xa nhau hơn so với độ rộng của chúng, và bộ sinh, một ánh xạ liên tục từ $z$, khó phủ nhiều vùng rời rạc cùng lúc.
2. **Làm bộ phân biệt mạnh hơn nhiều so với bộ sinh**, ví dụ thêm lớp và cập nhật bộ phân biệt 5 lần cho mỗi lần cập nhật bộ sinh. Cơ chế dự kiến: bộ phân biệt tìm ra ranh giới rất sắc, và bộ sinh dồn vào một vùng hẹp mà nó tạm thời đánh lừa được.
3. **Giảm số chiều của $z$ xuống 1.** Cơ chế dự kiến: ảnh của bộ sinh khi đó là một đường cong liên tục trong mặt phẳng; để đi qua 8 cụm rời rạc, đường cong phải đi qua cả vùng trống giữa chúng, nên bộ sinh có xu hướng chỉ phủ vài cụm gần nhau. Đây là một ràng buộc hình học, không phụ thuộc vào quá trình tối ưu, nên là thay đổi có khả năng tạo ra hiện tượng cao nhất.

**(b) Chạy lại và báo cáo.** Phần này để người học tự làm. Yêu cầu quan trọng là ghi lại kết quả đúng như nó xảy ra, kể cả khi khác với dự đoán. Thí nghiệm gốc ở Mục 10.3 chính là một ví dụ: dự định ban đầu là minh hoạ sụp chế độ, nhưng sau nhiều cấu hình vẫn không tái hiện được, và nội dung được viết theo số liệu thay vì theo dự định. Một thí nghiệm bác bỏ giả thuyết ban đầu có giá trị hơn một thí nghiệm được điều chỉnh cho tới khi ra kết quả mong muốn.

**(c) Chỉ số phủ tốt hơn.** Cách đếm hiện tại, số cụm có ít nhất một điểm rơi vào, là chỉ số nhị phân và rất dễ dãi: một cụm nhận 1 điểm được tính ngang một cụm nhận 1 000 điểm. Chỉ số tốt hơn: gọi $p_i$ là tỉ lệ điểm sinh ra rơi vào cụm $i$ (trong số các điểm rơi vào một cụm nào đó), so với tỉ lệ thật $q_i = 1/8$, rồi đo

$$\begin{gathered} \mathrm{KL}(p \,\|\, q) \quad\text{hoặc} \\ \text{số chế độ hiệu dụng} = \exp\big(H(p)\big) = \exp\Big(-\sum_i p_i\log p_i\Big). \end{gathered}$$

Số chế độ hiệu dụng bằng 8 khi các cụm nhận số điểm bằng nhau và bằng 1 khi mọi điểm dồn vào một cụm, cho một giá trị liên tục trong $[1, 8]$. Đây là cùng đại lượng $e^H$ dùng để đo số lựa chọn hiệu dụng của các chiến lược giải mã ở [Mục 11.3 của *Học sâu*](models-ch11.html).

**(d) Phủ đủ chế độ chưa đủ.** Chỉ 15,3% số điểm sinh ra nằm trong phạm vi một cụm: bộ sinh đặt khối lượng xác suất vào đúng vùng của 8 cụm, nhưng phần lớn điểm nằm rải rác giữa các cụm, nơi dữ liệu thật gần như không có. Phủ đủ các chế độ là điều kiện cần, không phải điều kiện đủ, để học đúng phân phối; cần thêm các chỉ số so sánh mật độ, như tỉ lệ điểm trong cụm hay phân kỳ giữa phân phối sinh ra và phân phối thật.
