# Học sâu và các kiến trúc mô hình

> **Giáo trình 2 của lộ trình.** Giáo trình *Nền tảng* dừng ở các mô hình tuyến tính, SVM và các phương pháp không giám sát cổ điển. Giáo trình này đi tiếp từ chỗ đó theo một mạch: so sánh các họ mô hình qua giả định của chúng, dùng phân rã độ chệch – phương sai để hiểu vì sao kết hợp nhiều cây lại tốt hơn một cây, rồi chuyển sang mạng nơ-ron, cách huấn luyện chúng khi chúng sâu, các kiến trúc cho ảnh và cho chuỗi, và cuối cùng là Transformer cùng các thành phần của mô hình ngôn ngữ lớn hiện nay.
>
> **Kiến thức cần có.** Đại số tuyến tính, xác suất, gradient descent và hồi quy logistic ở mức của *Nền tảng*. Chỗ nào dùng lại kiến thức cũ đều có liên kết tới đúng mục.
>
> **Về số liệu.** Giáo trình có hai loại con số. Loại thứ nhất trích từ bài báo gốc, mỗi lần dùng đều ghi nguồn, và danh sách nguồn nằm ở Chương 15. Loại thứ hai được đo trong chính repo này bằng các script trong `code/models/` với hạt giống cố định, nên chạy lại cho kết quả giống hệt. Một số kết quả là phép đối chiếu với con số đã công bố, như số tham số của GPT-2 và Llama 2, và các phép đối chiếu đó khớp chính xác.

---

## Mục lục

0. Ký hiệu và quy ước
1. Các họ mô hình
2. Độ chệch và phương sai
3. Cây quyết định và phương pháp tập hợp
4. Mạng nơ-ron nhiều lớp
5. Lan truyền ngược và các thuật toán tối ưu
6. Khởi tạo, chuẩn hoá và kết nối tắt
7. Mạng nơ-ron tích chập
8. Mạng nơ-ron hồi quy
9. Attention và Transformer
10. Các cải tiến của Transformer
11. Các chiến lược giải mã
12. Tham số, FLOP và bộ nhớ của Transformer
13. Bài tập
14. Câu hỏi phỏng vấn
15. Tài liệu tham khảo

---

## 0. Ký hiệu và quy ước

Giáo trình dùng lại ký hiệu và quy ước thuật ngữ của *Nền tảng* ([Chương 0 của *Nền tảng*](nentang-ch00.html)), và bổ sung các ký hiệu riêng cho mạng nơ-ron và Transformer.

### 0.1. Bảng ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| $x \in \mathbb{R}^{d}$ | một điểm dữ liệu đầu vào có $d$ đặc trưng |
| $X \in \mathbb{R}^{n \times d}$ | ma trận dữ liệu gồm $n$ điểm |
| $y$, $\hat{y}$ | nhãn thật và giá trị dự đoán |
| $W$, $b$ | ma trận trọng số và vector hệ số chặn của một lớp |
| $h^{(l)}$, $h_t$ | đầu ra của lớp ẩn thứ $l$; trạng thái ẩn của RNN tại bước $t$ |
| $\phi$, $\sigma$ | hàm kích hoạt nói chung; hàm sigmoid |
| $\mathcal{L}$ | hàm mất mát |
| $\eta$ | tốc độ học (learning rate) |
| $B$ | kích thước lô (batch size) |
| $T$ | độ dài chuỗi: số token hoặc số bước thời gian |
| $d$ hay $d_{\text{model}}$ | số chiều biểu diễn của Transformer |
| $h$ | số đầu attention |
| $d_k = d/h$ | số chiều của mỗi đầu |
| $d_{\text{ff}}$ | số chiều lớp ẩn của khối FFN |
| $L$ | số lớp của mạng; với Transformer là số khối |
| $V$ | kích thước từ vựng; trong công thức attention là ma trận giá trị (value) |
| $N$ | số tham số của mô hình (không tính embedding, trừ khi ghi rõ) |
| $D$ | số token dữ liệu huấn luyện |

Giáo trình *Nền tảng* viết hàm mất mát là $L$, vì ở đó chưa có mạng nhiều lớp. Từ giáo trình này, chữ $L$ được dành cho số lớp, xuất hiện từ định nghĩa mạng nhiều lớp ở Chương 4 tới phép đếm tham số ở Chương 12, nên hàm mất mát viết là $\mathcal{L}$. Chữ $h$ chỉ vector ẩn khi nói về mạng nhiều lớp, mạng hồi quy và kết nối tắt (Chương 4 tới 8), và chỉ số đầu attention trong các chương về Transformer (Chương 9 tới 12).

### 0.2. Quy ước thuật ngữ

Giống *Nền tảng*, giáo trình dùng thuật ngữ theo cách phổ biến trong cộng đồng học máy tiếng Việt, có tham khảo bản dịch tiếng Việt các cheatsheet CS229 và CS230 của Đại học Stanford. Bảng dưới ghi các thuật ngữ mới xuất hiện trong giáo trình này.

| Tiếng Anh | Dùng trong giáo trình |
|---|---|
| bias (của một ước lượng) | độ chệch |
| bias (tham số $b$ của một lớp) | hệ số chặn (bias) |
| inductive bias | thiên kiến quy nạp (inductive bias) |
| layer, hidden layer | lớp, lớp ẩn |
| activation function | hàm kích hoạt |
| forward / backward pass | lượt xuôi / lượt ngược |
| backpropagation | lan truyền ngược |
| weight decay | weight decay (suy giảm trọng số) |
| residual / skip connection | kết nối tắt |
| convolution, filter, stride, padding, pooling | tích chập, bộ lọc, bước nhảy (stride), phần đệm (padding), gộp (pooling) |
| receptive field | trường tiếp nhận |
| recurrent neural network | mạng nơ-ron hồi quy (RNN) |
| gate, forget gate, cell state | cổng, cổng quên, trạng thái ô nhớ |
| query, key, value | truy vấn, khoá, giá trị (query, key, value) |
| head, mask | đầu, mặt nạ |
| decoding, beam search, temperature | giải mã, beam search, nhiệt độ |
| ensemble | phương pháp tập hợp (ensemble) |
| bagging, boosting, dropout, embedding, softmax, attention | giữ nguyên tiếng Anh |

Chữ *bias* có ba nghĩa trong giáo trình, và mỗi nghĩa có một tên riêng. **Độ chệch** là sai lệch có hệ thống của một ước lượng, như trong "ước lượng không chệch". **Hệ số chặn** là tham số cộng thêm $b$ trong $Wx + b$. **Thiên kiến quy nạp** là tập các giả định mà một thuật toán học dùng để chọn giữa các hàm cùng khớp dữ liệu (Mục 1.4).

### 0.3. Quy ước về chiều

Trong công thức, $x$ là vector cột và một lớp viết là $Wx + b$ với $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$. Trong mã, dữ liệu xếp theo hàng nên cả lô viết là `X @ W + b` với `W.shape == (d_in, d_out)`. Hai cách viết chỉ khác nhau một phép chuyển vị. Phần chữ dùng dấu phẩy thập phân (0,15); phần mã và kết quả in ra từ script giữ dấu chấm (0.15).

---

## 1. Các họ mô hình

Ở giáo trình *Nền tảng*, mỗi chương giới thiệu một mô hình: hồi quy tuyến tính, hồi quy logistic, SVM, k láng giềng gần nhất. Khi đứng trước một bài toán mới, câu hỏi thực tế lại khác: nên dùng mô hình nào? Câu trả lời hay nghe là "thử vài mô hình rồi chọn cái tốt nhất". Lời khuyên đó không sai, nhưng có thể đoán trước khá nhiều điều nếu biết mỗi họ mô hình giả định gì về hàm cần học. Chẳng hạn, cây tăng cường gradient vẫn thường thắng mạng nơ-ron trên dữ liệu dạng bảng, trong khi mạng tích chập và Transformer vượt xa mọi thứ khác trên ảnh và văn bản; cả hai sự kiện đều giải thích được bằng giả định của mô hình. Ta bắt đầu bằng việc dựng một khung để so sánh các họ mô hình theo cách đó.

### 1.1. Các câu hỏi khi thiết kế một mô hình

Mọi thuật toán học có giám sát đều phải trả lời cùng một nhóm câu hỏi, và mỗi thuật toán là một cách trả lời cụ thể. Câu hỏi đầu tiên là xét họ hàm nào: hàm tuyến tính của đặc trưng, hàm hằng trên từng vùng hình hộp, hay hợp của nhiều phép biến đổi. Đây chính là thành phần "mô hình" trong ba thành phần của một thuật toán học ([Mục 1.1 của *Nền tảng*](nentang-ch01.html)). Câu hỏi thứ hai là số tham số được cố định trước hay tăng theo dữ liệu. Hồi quy tuyến tính có $d + 1$ tham số dù có bao nhiêu dữ liệu, còn cây quyết định có thể mọc thêm nút khi có thêm dữ liệu; những mô hình như cây được gọi là **phi tham số** (nonparametric), dù chúng vẫn có tham số.

Câu hỏi thứ ba là tìm tham số bằng cách nào: bằng nghiệm dạng đóng như hồi quy tuyến tính, giải bài toán tối ưu lồi như hồi quy logistic và SVM, tìm kiếm tham lam như cây quyết định, hay chạy gradient descent trên một hàm không lồi như mạng nơ-ron. Câu hỏi cuối cùng là kiểm soát overfitting bằng cách nào: phạt chuẩn của trọng số, giới hạn độ sâu của cây, lấy trung bình nhiều mô hình, hay dừng sớm quá trình tối ưu.

Trả lời được bốn câu hỏi này cho một mô hình là nắm được những đặc điểm quan trọng nhất của nó: nó biểu diễn được gì, cần bao nhiêu dữ liệu, huấn luyện tốn kém thế nào và dễ hỏng ở đâu. Hình 1 xếp bốn họ mô hình lớn theo mức độ giả định của chúng.

![Hình 1](figs/models01_families.png)

**Hình 1.** Bốn họ mô hình lớn. Từ trái sang phải, mô hình càng ít giả định về dạng của hàm cần học, nên càng linh hoạt, nhưng càng cần nhiều dữ liệu và càng khó diễn giải.

### 1.2. So sánh các họ mô hình

Bảng dưới trả lời các câu hỏi ở Mục 1.1 cho bốn họ mô hình, kèm trường hợp phù hợp và hạn chế của mỗi họ.

| | Mô hình tuyến tính | Cây và phương pháp tập hợp | Mạng nơ-ron nhiều lớp | Transformer |
|---|---|---|---|---|
| Giả định về dạng hàm | tổ hợp tuyến tính của các đặc trưng | hằng số trên các hình hộp có cạnh song song với trục | hợp của nhiều phép biến đổi affine xen với hàm phi tuyến | như mạng nơ-ron, thêm cơ chế kết hợp thông tin giữa mọi cặp vị trí |
| Số tham số | $d + 1$, cố định | tăng theo dữ liệu | rất nhiều, cố định trước | rất nhiều, cố định trước |
| Cách tối ưu | nghiệm dạng đóng hoặc tối ưu lồi | tham lam theo từng lần chia | gradient descent và lan truyền ngược | như mạng nơ-ron |
| Kiểm soát overfitting | regularization $\ell_1$, $\ell_2$ | giới hạn độ sâu, cắt tỉa, lấy trung bình | dừng sớm, weight decay, dropout, tăng cường dữ liệu | như mạng nơ-ron |
| Phù hợp khi | ít dữ liệu, cần diễn giải, quan hệ gần tuyến tính | dữ liệu dạng bảng, đặc trưng lẫn số và hạng mục | dữ liệu thô có cấu trúc: ảnh, âm thanh | chuỗi dài, văn bản, và ngày càng nhiều loại dữ liệu khác |
| Hạn chế | không biểu diễn được quan hệ phi tuyến nếu không tự tạo đặc trưng | không ngoại suy được; kém với dữ liệu thô | cần nhiều dữ liệu; kém với dữ liệu bảng thuần | chi phí tăng theo bình phương độ dài chuỗi |

Bảng này giải thích được một kết quả thực nghiệm mà người mới thường ngạc nhiên: trên dữ liệu dạng bảng, các mô hình dựa trên cây, nhất là gradient boosting, vẫn thường cho kết quả bằng hoặc tốt hơn mạng nơ-ron sâu, với chi phí thấp hơn nhiều. Grinsztajn, Oyallon và Varoquaux (2022) so sánh trên 45 bộ dữ liệu dạng bảng và chỉ ra ba nguyên nhân. Dữ liệu bảng thường có nhiều đặc trưng không liên quan. Hàm cần học thường không trơn mà có những bước nhảy theo ngưỡng. Và mỗi cột có ý nghĩa riêng, nên một phép quay không gian đặc trưng làm mất thông tin. Cả ba đặc điểm này hợp với cách chia theo từng trục của cây, và không hợp với phép biến đổi affine của mạng nơ-ron.

### 1.3. Các yếu tố quyết định kết quả

Họ mô hình không phải yếu tố duy nhất, và thường cũng không phải yếu tố quan trọng nhất. Với cùng một bài toán, kinh nghiệm thực tế cho thấy thứ tự ảnh hưởng tới kết quả thường là

$$\begin{gathered} \text{dữ liệu và nhãn} \;>\; \text{đặc trưng} \;>\; \text{họ mô hình} \\ \;>\; \text{siêu tham số} \;>\; \text{chi tiết kiến trúc}. \end{gathered}$$

Đây là quy tắc kinh nghiệm, không phải định lý, và có ngoại lệ: với ảnh và văn bản, chọn đúng họ mô hình như mạng tích chập hay Transformer quan trọng hơn nhiều so với tự thiết kế đặc trưng. Nhưng quy tắc này nhắc một điều hay bị quên: nhãn sai, hoặc dữ liệu không đại diện cho tình huống sử dụng thật, là những lỗi mà không mô hình nào sửa được. Giáo trình này tập trung vào ba mức sau; hai mức đầu được bàn trong [*MLOps*](mlops-ch03.html).

### 1.4. Thiên kiến quy nạp

Một tập dữ liệu hữu hạn luôn khớp được với vô số hàm khác nhau. Qua 10 điểm dữ liệu có vô số đường cong đi qua chính xác mọi điểm, và chúng dự đoán rất khác nhau ở những điểm chưa thấy. Muốn dự đoán được, thuật toán học phải ưu tiên một số hàm hơn các hàm khác, bằng những giả định không đến từ dữ liệu.

> **Định nghĩa 1.1 (Thiên kiến quy nạp).** **Thiên kiến quy nạp** (inductive bias) của một thuật toán học là tập các giả định nó dùng để chọn giữa các hàm cùng khớp dữ liệu huấn luyện, và do đó quyết định cách nó dự đoán cho các điểm chưa thấy.

Mỗi họ mô hình có một thiên kiến quy nạp riêng. Hồi quy tuyến tính giả định quan hệ tuyến tính. Regularization $\ell_2$ ưu tiên trọng số nhỏ ([Mục 9.2 của *Nền tảng*](nentang-ch09.html)). k láng giềng gần nhất giả định các điểm gần nhau có nhãn giống nhau. Mạng tích chập giả định đặc trưng có tính cục bộ và giống nhau ở mọi vị trí trong ảnh (Mục 7.1). Transformer giả định rất ít, và vì thế cần rất nhiều dữ liệu.

Bốn căn hộ ở [Chương 4 của *Nền tảng*](nentang-ch04.html), rộng 40, 60, 80, 100 m² với giá 1,5; 2,1; 2,4; 3,0 tỉ đồng, cho thấy thiên kiến quy nạp quyết định dự đoán ra sao. Ba mô hình học từ cùng bốn điểm này. Hồi quy tuyến tính cho đường $0{,}57 + 0{,}024x$. Cây quyết định một lần chia cắt ở 70 m², dự đoán 1,8 tỉ cho mọi căn nhỏ hơn và 2,7 tỉ cho mọi căn lớn hơn. Còn 1 láng giềng gần nhất lấy giá của căn gần nhất. Với căn 120 m², ba mô hình dự đoán lần lượt 3,45; 2,7 và 3,0 tỉ. Cả ba đều khớp bốn căn hộ ở mức chấp nhận được, nhưng khác nhau ở giả định: hồi quy tuyến tính giả định giá tăng đều theo diện tích nên ngoại suy theo đường thẳng, cây giả định giá là hằng số trên từng khoảng nên mọi dự đoán của nó nằm trong khoảng giá đã thấy, còn 1 láng giềng gần nhất giả định một căn hộ đáng giá như căn giống nó nhất đã thấy. Dữ liệu không cho biết giả định nào đúng.

Không có thiên kiến quy nạp nào tốt nhất cho mọi bài toán. Định lý "không có bữa trưa miễn phí" (no free lunch; Wolpert, 1996) phát biểu chặt chẽ điều này: lấy trung bình trên mọi bài toán có thể, mọi thuật toán học có cùng sai số trên các điểm ngoài tập huấn luyện. Một thuật toán chỉ tốt hơn thuật toán khác trên những bài toán mà giả định của nó phù hợp. Vì vậy câu hỏi "mô hình nào tốt nhất" không có câu trả lời chung; câu hỏi đúng là giả định của mô hình nào khớp với cấu trúc của dữ liệu đang có.

### 1.5. Tóm tắt

Mỗi họ mô hình là một cách trả lời bốn câu hỏi: xét họ hàm nào, số tham số cố định hay tăng theo dữ liệu, tìm tham số bằng cách nào, và kiểm soát overfitting ra sao. Các câu trả lời đó quy về thiên kiến quy nạp của mô hình, tức những giả định nó dùng để chọn giữa các hàm cùng khớp dữ liệu; từ cùng bốn căn hộ, hồi quy tuyến tính, cây một lần chia và 1 láng giềng gần nhất định giá căn 120 m² là 3,45; 2,7 và 3,0 tỉ. Không có thiên kiến nào tốt nhất cho mọi bài toán: cây hợp với dữ liệu bảng vì nó chia theo từng trục, còn mạng tích chập và Transformer hợp với ảnh và văn bản vì giả định của chúng khớp với cấu trúc của các loại dữ liệu đó. Dù vậy, dữ liệu và nhãn thường ảnh hưởng tới kết quả nhiều hơn việc chọn mô hình.

Giả định càng mạnh thì họ hàm càng hẹp. Nếu giả định đúng, mô hình học được từ ít dữ liệu; nếu sai, nó mắc một sai số có hệ thống mà thêm dữ liệu cũng không sửa được. Chương 2 đặt tên cho loại sai số này và loại sai số đối lập với nó, rồi đo cả hai.

---

## 2. Độ chệch và phương sai

[Mục 3.6 của *Nền tảng*](nentang-ch03.html) đã tách sai số của một ước lượng thành độ chệch và phương sai. Ở đây ta áp dụng cùng phép tách cho dự đoán của cả một mô hình học máy, rồi đo từng thành phần bằng mô phỏng. Phép tách này có giá trị thực tế rất rõ: biết sai số chủ yếu đến từ độ chệch hay từ phương sai thì biết nên đổi mô hình hay nên thu thêm dữ liệu, hai hướng đi tốn kém theo những cách rất khác nhau.

### 2.1. Vai trò của phân rã độ chệch – phương sai

Nhiều quyết định trong các chương sau thực chất là điều chỉnh độ linh hoạt của mô hình: thêm hay bớt lớp, regularization mạnh hay nhẹ, cây sâu hay nông, lấy trung bình bao nhiêu mô hình. Mỗi quyết định như vậy làm hai loại sai số thay đổi theo hai chiều ngược nhau. Phân rã độ chệch – phương sai cho biết chính xác hai loại sai số đó là gì, và vì sao tổng của chúng thường nhỏ nhất ở một mức linh hoạt vừa phải.

### 2.2. Phân rã sai số dự đoán

Giả sử dữ liệu sinh từ mô hình $y = f(x) + \varepsilon$, trong đó $f$ là hàm thật, $\varepsilon$ là nhiễu có $\mathbb{E}[\varepsilon] = 0$, $\operatorname{Var}(\varepsilon) = \sigma^2$ và độc lập với $x$. Một thuật toán học nhận tập huấn luyện $\mathcal{D}$ và trả về hàm dự đoán $\hat f_{\mathcal{D}}$. Vì $\mathcal{D}$ là ngẫu nhiên, giá trị dự đoán $\hat f_{\mathcal{D}}(x_0)$ tại một điểm cố định $x_0$ cũng là một biến ngẫu nhiên.

> **Định lý 2.1 (Phân rã độ chệch – phương sai).** Tại một điểm $x_0$, với $y_0 = f(x_0) + \varepsilon_0$ là một quan sát mới độc lập với tập huấn luyện, sai số bình phương kỳ vọng tách thành ba phần:
> $$\begin{aligned} \mathbb{E}\big[(y_0 - \hat f_{\mathcal{D}}(x_0))^2\big] &= \underbrace{\big(\mathbb{E}[\hat f_{\mathcal{D}}(x_0)] - f(x_0)\big)^2}_{\text{độ chệch}^2} \\ &\quad + \underbrace{\operatorname{Var}\big(\hat f_{\mathcal{D}}(x_0)\big)}_{\text{phương sai}} + \underbrace{\sigma^2}_{\text{nhiễu}}, \end{aligned}$$
> trong đó kỳ vọng và phương sai lấy theo phân phối của tập huấn luyện $\mathcal{D}$ và của nhiễu $\varepsilon_0$.

> **Chứng minh.** Viết gọn $\hat f = \hat f_{\mathcal{D}}(x_0)$, $f = f(x_0)$ và $m = \mathbb{E}[\hat f]$. Ta có $y_0 - \hat f = (f - \hat f) + \varepsilon_0$. Bình phương và lấy kỳ vọng: số hạng chéo $2\,\mathbb{E}[(f - \hat f)\varepsilon_0]$ bằng 0 vì $\varepsilon_0$ độc lập với $\hat f$ và có kỳ vọng 0, còn $\mathbb{E}[\varepsilon_0^2] = \sigma^2$. Phần còn lại $\mathbb{E}[(f - \hat f)^2]$ được tách giống [Định lý 3.2 của *Nền tảng*](nentang-ch03.html): viết $f - \hat f = (f - m) + (m - \hat f)$, số hạng chéo $2(f - m)\,\mathbb{E}[m - \hat f]$ bằng 0, còn lại $(f - m)^2 + \operatorname{Var}(\hat f)$.

Ba thành phần có ý nghĩa khác nhau. Độ chệch đo sai lệch có hệ thống: lấy trung bình trên rất nhiều tập huấn luyện, mô hình vẫn dự đoán lệch khỏi giá trị thật, vì họ hàm quá hẹp so với hàm thật. Độ chệch cao ứng với underfitting. Phương sai đo mức dao động: đổi tập huấn luyện thì dự đoán thay đổi nhiều, vì mô hình đủ linh hoạt để khớp cả phần nhiễu riêng của từng tập. Phương sai cao ứng với overfitting. Còn nhiễu $\sigma^2$ không phụ thuộc vào mô hình; nó là cận dưới của sai số mà không mô hình nào vượt qua được.

> **Lưu ý.** Độ chệch và phương sai được định nghĩa qua kỳ vọng trên **các tập huấn luyện có thể rút ra**, tại một điểm $x_0$ cố định. Chúng không đo mức dao động của dự đoán giữa các điểm dữ liệu khác nhau. Một mô hình có dự đoán thay đổi mạnh theo $x$ vẫn có thể có phương sai thấp, nếu huấn luyện trên tập dữ liệu khác thì nó cho gần như cùng một hàm.

Bốn căn hộ ở Mục 1.4 cho một phép tính tay. Giả sử giá thật là ${f(x) = 0{,}57 + 0{,}024x}$ cộng nhiễu có phương sai $\sigma^2$ ở mỗi căn, và ta dự đoán giá căn 120 m², nơi $f(120) = 3{,}45$. Ba mô hình dưới đây đều cho dự đoán là một tổ hợp tuyến tính $\sum_i c_i y_i$ của bốn mức giá quan sát được, với các hệ số $c_i$ chỉ phụ thuộc vào diện tích. Nhiễu ở bốn căn độc lập, nên phương sai của dự đoán là $\sigma^2\sum_i c_i^2$.

| Mô hình | Hệ số $c_i$ tại 120 m² | Độ chệch | Phương sai |
|---|---|---|---|
| Hằng số: trung bình bốn giá | 0,25 cho mỗi căn | $-1{,}2$ | $0{,}25\,\sigma^2$ |
| Đường thẳng | $-0{,}5;\ 0;\ 0{,}5;\ 1$ | 0 | $1{,}5\,\sigma^2$ |
| Đa thức bậc ba qua bốn điểm | $-1;\ 4;\ -6;\ 4$ | 0 | $69\,\sigma^2$ |

Mô hình hằng số có phương sai nhỏ nhất nhưng kỳ vọng của nó là ${f(70) = 2{,}25}$, lệch 1,2 tỉ so với giá thật 3,45 tỉ. Đường thẳng và đa thức bậc ba đều không chệch, vì hàm thật là một đường thẳng, nhưng phương sai của đa thức gấp 46 lần đường thẳng: hệ số $-6$ của căn 80 m² nghĩa là nhiễu ở căn đó bị khuếch đại sáu lần khi ngoại suy ra 120 m². Với $\sigma^2 = 0{,}009$, ước lượng ở [Mục 10.2 của *Nền tảng*](nentang-ch10.html), phần sai số ngoài nhiễu là 1,442 cho mô hình hằng số, 0,0135 cho đường thẳng và 0,621 cho đa thức bậc ba.

Ba mô hình này là một trường hợp của quy luật chung. Khi tăng độ linh hoạt của mô hình, độ chệch thường giảm vì họ hàm rộng hơn, còn phương sai thường tăng vì có nhiều cách hơn để khớp theo nhiễu. Tổng của chúng vì vậy thường có dạng chữ U theo độ linh hoạt. Chữ "thường" ở đây là cần thiết: định lý chỉ khẳng định phép tách, không khẳng định hai thành phần thay đổi đơn điệu, và Mục 2.3 sẽ cho thấy những trường hợp ngoại lệ.

### 2.3. Đo độ chệch và phương sai bằng mô phỏng

Với dữ liệu mô phỏng, hàm thật $f$ đã biết, nên có thể đo trực tiếp cả ba thành phần: sinh nhiều tập huấn luyện độc lập, huấn luyện một mô hình trên mỗi tập, rồi tính trung bình và phương sai của các dự đoán tại từng điểm.

Thí nghiệm trong `code/models/experiments.py` dùng hàm thật $f(x) = \sin(2{,}2x) + 0{,}35x$ trên đoạn $[-3, 3]$ và nhiễu Gauss có độ lệch chuẩn $\sigma = 0{,}35$, tức $\sigma^2 = 0{,}1225$. Mỗi tập huấn luyện gồm 40 điểm có $x$ rút đều trên đoạn. Với mỗi bậc đa thức từ 1 tới 14, thí nghiệm khớp đa thức trên 250 tập huấn luyện độc lập, rồi tính độ chệch² và phương sai tại 200 điểm cách đều và lấy trung bình theo các điểm.

![Hình 2](figs/models02_biasvar.png)

**Hình 2.** Độ chệch², phương sai và tổng sai số theo bậc đa thức, trục dọc theo thang log. Đường chấm ngang là mức nhiễu $\sigma^2$; đường đứt dọc đánh dấu bậc có tổng sai số nhỏ nhất.

| Bậc đa thức | Độ chệch² | Phương sai | Nhiễu $\sigma^2$ | Tổng sai số |
|---|---|---|---|---|
| 1 | 0,4221 | 0,0231 | 0,1225 | 0,5677 |
| 2 | 0,4223 | 0,0418 | 0,1225 | 0,5866 |
| 3 | 0,4098 | 0,0857 | 0,1225 | 0,6180 |
| 4 | 0,4191 | 0,1852 | 0,1225 | 0,7268 |
| **5** | **0,0836** | **0,0767** | 0,1225 | **0,2828** |
| 7 | 0,0058 | 0,3415 | 0,1225 | 0,4699 |
| 9 | 0,0024 | 1,2645 | 0,1225 | 1,3894 |
| 12 | 0,0276 | 52,2156 | 0,1225 | 52,3657 |
| 14 | 35,1402 | 7 392,2751 | 0,1225 | 7 427,5379 |

Ta đọc bảng từ trên xuống. Ở bậc thấp, sai số đến từ độ chệch: với bậc 1, độ chệch² bằng 0,4221, chiếm 74% tổng sai số. Một đường thẳng không thể theo được hình sin dù có bao nhiêu dữ liệu, nên thu thêm dữ liệu không giúp gì; phải dùng họ hàm linh hoạt hơn.

Bảng còn cho thấy một chi tiết bất ngờ: tăng lên bậc chẵn gần như không làm giảm độ chệch. Độ chệch² ở bậc 2 bằng bậc 1 (0,4223 so với 0,4221), ở bậc 4 xấp xỉ bậc 3 (0,4191 so với 0,4098). Lý do là hàm thật là hàm lẻ, $f(-x) = -f(x)$, còn dữ liệu rải đối xứng quanh 0, nên các số hạng bậc chẵn không giúp xấp xỉ $f$ tốt hơn; chúng chỉ thêm tham số để khớp nhiễu, nên phương sai tăng. Đây là một ví dụ cụ thể cho điều Mục 2.2 đã lưu ý: độ chệch không nhất thiết giảm đều khi mô hình phức tạp hơn, mà chỉ giảm khi họ hàm mở rộng theo hướng có ích cho việc xấp xỉ hàm thật.

Tổng sai số nhỏ nhất ở bậc 5, nơi độ chệch² 0,0836 và phương sai 0,0767 gần bằng nhau. Tổng 0,2828 vẫn cao hơn mức nhiễu 0,1225, và phần chênh này là sai số không tránh được khi chỉ có 40 điểm dữ liệu.

Từ bậc 7 trở đi, sai số đến từ phương sai. Ở bậc 12, phương sai bằng 52,2, gấp khoảng 1 900 lần độ chệch². Đa thức bậc cao dao động rất mạnh ở hai đầu đoạn, nơi có ít điểm dữ liệu, và mỗi tập huấn luyện cho một kiểu dao động khác nhau. Khác với độ chệch, phương sai giảm khi có thêm dữ liệu, nên với mô hình này thu thêm dữ liệu là cách sửa đúng.

Dòng cuối của bảng cần đọc thận trọng. Độ chệch² đo được ở bậc 13 (dòng này không có trong bảng) và bậc 14 tăng vọt lên 6,2 và 35,1, nhưng đó không phải độ chệch thật. Trung bình của 250 dự đoán có sai số ngẫu nhiên, và bình phương của sai số đó cộng thêm vào độ chệch² ước lượng một lượng cỡ $\text{phương sai}/250$, bằng khoảng 3,5 ở bậc 13 và 29,6 ở bậc 14. Khi phương sai lớn tới mức này, cần nhiều tập huấn luyện hơn hẳn mới ước lượng được độ chệch; kết luận chắc chắn duy nhất là phương sai chi phối.

Giá trị thực tế của phép phân rã nằm ở chỗ nó phân biệt hai tình huống cần xử lý ngược nhau. Độ chệch cao cần mô hình linh hoạt hơn, thêm đặc trưng hoặc giảm regularization, và thu thêm dữ liệu gần như vô ích. Phương sai cao thì cần thêm dữ liệu, tăng regularization hoặc dùng mô hình đơn giản hơn. Chẩn đoán sai có thể dẫn tới việc tốn nhiều tuần thu thập dữ liệu cho một mô hình mà dữ liệu không cứu được.

### 2.4. Chẩn đoán bằng đường cong học

Với dữ liệu thật, hàm $f$ không biết trước và không thể rút lại hàng trăm tập huấn luyện, nên không đo trực tiếp được độ chệch và phương sai. Công cụ thay thế là **đường cong học** (learning curve): đồ thị sai số trên tập huấn luyện và sai số trên tập xác thực theo số điểm dữ liệu huấn luyện, hoặc theo số vòng huấn luyện. Bảng dưới tóm tắt cách đọc các dấu hiệu thường gặp.

| Dấu hiệu | Chẩn đoán | Hướng xử lý |
|---|---|---|
| Sai số huấn luyện cao, sai số xác thực cao, hai đường sát nhau và đã nằm ngang | độ chệch cao (underfitting) | mô hình linh hoạt hơn, thêm đặc trưng, giảm regularization, huấn luyện lâu hơn |
| Sai số huấn luyện thấp, sai số xác thực cao, khoảng cách lớn và chưa thu hẹp | phương sai cao (overfitting) | thêm dữ liệu, tăng regularization, tăng cường dữ liệu, mô hình đơn giản hơn, dừng sớm |
| Cả hai đều cao và vẫn đang giảm theo số vòng huấn luyện | huấn luyện chưa xong | huấn luyện tiếp |
| Sai số xác thực thấp hơn sai số huấn luyện | thường là dấu hiệu có lỗi | kiểm tra rò rỉ dữ liệu; kiểm tra dropout hoặc tăng cường dữ liệu chỉ bật khi huấn luyện |

Dòng cuối cần chú ý vì nó trông như tin tốt. Tình huống vô hại duy nhất là khi dropout hoặc tăng cường dữ liệu chỉ bật lúc huấn luyện, vì khi đó sai số huấn luyện được đo trên một bài toán khó hơn bài toán lúc xác thực. Trong các trường hợp còn lại, nguyên nhân thường là rò rỉ dữ liệu ([Mục 8.5 của *Nền tảng*](nentang-ch08.html)), chẳng hạn tập xác thực chứa bản sao gần giống của dữ liệu huấn luyện.

Hai đường cong theo số điểm dữ liệu còn trả lời một câu hỏi có giá trị kinh tế: thu thêm dữ liệu có đáng không. Nếu đường sai số xác thực đã nằm ngang và sát đường sai số huấn luyện, thêm dữ liệu không giúp gì; nếu nó vẫn đang giảm và còn cách xa đường huấn luyện, thêm dữ liệu nhiều khả năng giúp được.

### 2.5. Double descent

Đường chữ U ở Mục 2.3 là bức tranh cổ điển, đúng khi số tham số nhỏ hơn số điểm dữ liệu. Với mô hình có số tham số vượt xa số điểm dữ liệu, như các mạng nơ-ron hiện đại, người ta quan sát được hiện tượng **double descent** (Belkin và cộng sự, 2019): sai số xác thực tăng lên tới một đỉnh tại ngưỡng mà mô hình vừa đủ tham số để khớp chính xác mọi điểm huấn luyện, gọi là ngưỡng nội suy, rồi giảm trở lại khi số tham số tiếp tục tăng. Nakkiran và cộng sự (2020) quan sát cùng hiện tượng với mạng tích chập và Transformer, theo cả số tham số lẫn số vòng huấn luyện.

Double descent không mâu thuẫn với Định lý 2.1, vì phân rã là một đẳng thức và luôn đúng. Điều thay đổi là cách phương sai phụ thuộc vào số tham số. Khi có nhiều tham số hơn số điểm dữ liệu, có vô số nghiệm khớp chính xác dữ liệu huấn luyện, và thuật toán tối ưu chọn một trong số đó. Với hồi quy tuyến tính, có thể chứng minh gradient descent khởi tạo từ 0 hội tụ tới nghiệm có chuẩn nhỏ nhất, tức nghiệm giả nghịch đảo ([Mục 4.3 của *Nền tảng*](nentang-ch04.html)). Việc chọn nghiệm chuẩn nhỏ nhất tác dụng như một dạng regularization ngầm, và càng nhiều tham số thì nghiệm chuẩn nhỏ nhất càng trơn. Với mạng sâu, cơ chế tương tự được nhiều nghiên cứu ủng hộ nhưng chưa được chứng minh đầy đủ.

Vì vậy câu "mô hình càng nhiều tham số càng dễ overfitting" chỉ đúng trong chế độ cổ điển, khi số tham số nhỏ hơn số điểm dữ liệu. Trong chế độ nhiều tham số hơn dữ liệu, nó không còn đúng hiển nhiên, và đó là một phần lời giải thích cho việc các mô hình hàng tỉ tham số vẫn tổng quát hoá được. Tuy vậy, ngay trong chế độ này, regularization, dừng sớm và dữ liệu nhiều hơn vẫn có ích; double descent không có nghĩa là có thể bỏ qua overfitting.

### 2.6. Tóm tắt

Sai số kỳ vọng của một mô hình tại một điểm tách thành độ chệch², phương sai và nhiễu. Độ chệch đến từ họ hàm quá hẹp, phương sai đến từ việc khớp theo nhiễu riêng của từng tập huấn luyện, còn nhiễu là cận dưới không vượt qua được. Với bốn căn hộ và căn 120 m², mô hình hằng số lệch 1,2 tỉ, đường thẳng không chệch với phương sai $1{,}5\,\sigma^2$, còn đa thức bậc ba qua bốn điểm không chệch nhưng có phương sai $69\,\sigma^2$. Mô phỏng với đa thức cho thấy tổng sai số nhỏ nhất ở bậc 5, nơi hai thành phần gần bằng nhau, và độ chệch chỉ giảm khi họ hàm mở rộng theo hướng có ích, như các bậc lẻ với một hàm lẻ. Với dữ liệu thật, đường cong học thay cho phép đo trực tiếp và cho biết nên đổi mô hình hay thu thêm dữ liệu. Ở chế độ nhiều tham số hơn dữ liệu, double descent cho thấy phương sai không nhất thiết tăng theo số tham số.

Một mô hình minh hoạ rất rõ cả hai loại sai số là cây quyết định: cây nông có độ chệch cao, cây sâu có phương sai cao. Chương 3 xét cây quyết định, rồi hai cách kết hợp nhiều cây để giảm từng loại sai số.

---

## 3. Cây quyết định và phương pháp tập hợp

Cây quyết định có những ưu điểm mà các mô hình khác hiếm có: dễ diễn giải, không cần chuẩn hoá đặc trưng, xử lý tự nhiên cả đặc trưng số lẫn đặc trưng hạng mục. Điểm yếu của nó là phương sai: một cây sâu thay đổi gần như hoàn toàn khi dữ liệu thay đổi một chút. Hai ý tưởng kết hợp nhiều cây biến điểm yếu đó thành điểm mạnh theo hai cách khác nhau, và phân rã ở Chương 2 cho biết chính xác mỗi cách tác động vào đâu: bagging nhắm vào phương sai, còn boosting nhắm vào độ chệch. Rừng ngẫu nhiên và gradient boosting, hai sản phẩm của hai ý tưởng đó, nằm trong số các mô hình được dùng nhiều nhất cho dữ liệu dạng bảng. Ta bắt đầu từ một cây.

### 3.1. Cây quyết định

Cây quyết định chia không gian đặc trưng thành các vùng hình hộp bằng các phép chia dạng "$x_j \le t$", mỗi phép chia dùng một đặc trưng và một ngưỡng. Mỗi lá của cây ứng với một vùng, và dự đoán trong vùng đó là một hằng số: lớp chiếm đa số với bài toán phân loại, hoặc giá trị trung bình với bài toán hồi quy, của các điểm huấn luyện rơi vào vùng.

![Hình 3](figs/models03_tree.png)

**Hình 3.** Trái: cây chia mặt phẳng bằng các đường thẳng song song với trục. Phải: cùng mô hình đó vẽ dưới dạng cây nhị phân, mỗi nút trong là một phép chia.

Vì mọi phép chia đều song song với trục, cây có ba tính chất đi liền với nhau. Thứ nhất, cây bất biến với phép biến đổi đơn điệu của từng đặc trưng: lấy logarit hay chuẩn hoá một đặc trưng không đổi thứ tự các giá trị, nên không đổi tập các phép chia có thể có. Vì vậy cây không cần chuẩn hoá đặc trưng, khác với hồi quy logistic, SVM hay k-NN. Thứ hai, cây không ngoại suy được: ngoài miền dữ liệu huấn luyện, nó trả về hằng số của vùng ngoài cùng. Một cây dự đoán giá nhà theo diện tích sẽ cho căn nhà 500 m² cùng giá với căn lớn nhất trong dữ liệu huấn luyện. Cây một lần chia ở Mục 1.4 định giá mọi căn từ 70 m² trở lên là 2,7 tỉ, dù căn đó rộng 100 m² hay 500 m². Thứ ba, cây khó biểu diễn biên chéo. Biên quyết định $x_1 + x_2 > 1$ phải được xấp xỉ bằng một đường bậc thang gồm nhiều phép chia, trong khi mô hình tuyến tính chỉ cần ba tham số.

### 3.2. Chọn phép chia

Cây được xây từ gốc xuống. Tại mỗi nút, thuật toán xét mọi đặc trưng và mọi ngưỡng có thể, rồi chọn phép chia làm giảm **độ không thuần nhất** (impurity) nhiều nhất. Với phân loại, gọi $p_c$ là tỉ lệ điểm thuộc lớp $c$ trong nút; hai thước đo thông dụng là

$$\text{Gini}(p) = 1 - \sum_c p_c^2, \qquad H(p) = -\sum_c p_c \log_2 p_c.$$

Cả hai bằng 0 khi nút chỉ chứa một lớp và lớn nhất khi các lớp chia đều. **Độ lợi** của một phép chia là độ không thuần nhất của nút cha trừ trung bình có trọng số, theo số điểm, của hai nút con. Với hồi quy, độ không thuần nhất là phương sai của $y$ trong nút, và phép chia tốt nhất là phép chia làm giảm tổng bình phương sai số nhiều nhất. Với bốn căn hộ, có ba ngưỡng để thử: giữa 40 và 60 m², giữa 60 và 80 m², giữa 80 và 100 m². Khi mọi căn được đoán bằng giá trung bình 2,25 tỉ, tổng bình phương sai số là 1,17. Chia ở 50 m² tách căn nhỏ nhất ra một mình và đoán 2,5 tỉ cho ba căn còn lại, tổng còn 0,42; chia ở 70 m² cho hai nút với giá trung bình 1,8 và 2,7 tỉ, tổng còn 0,36; chia ở 90 m² cũng còn 0,42. Thuật toán chọn ngưỡng 70 m², và đó là cây một lần chia ở Mục 1.4.

> **Ví dụ 3.1.** Một nút có 10 điểm, 6 điểm lớp A và 4 điểm lớp B, nên $\text{Gini} = 1 - 0{,}6^2 - 0{,}4^2 = 0{,}48$. Phép chia thứ nhất cho nút trái 4 điểm (4A, 0B) và nút phải 6 điểm (2A, 4B). Gini của nút trái bằng 0, của nút phải bằng $1 - (1/3)^2 - (2/3)^2 = 0{,}444$, nên trung bình có trọng số là $0{,}4 \cdot 0 + 0{,}6 \cdot 0{,}444 = 0{,}267$ và độ lợi là $0{,}48 - 0{,}267 = 0{,}213$. Phép chia thứ hai cho hai nút 5 điểm: (4A, 1B) có Gini 0,32 và (2A, 3B) có Gini 0,48, trung bình 0,40, nên độ lợi chỉ là 0,08. Thuật toán chọn phép chia thứ nhất. Dùng entropy cũng cho cùng kết luận: độ lợi thông tin là 0,420 bit so với 0,125 bit.

Trong thực tế, Gini và entropy gần như luôn chọn cùng một phép chia, và Gini tính nhanh hơn vì không cần logarit. Các siêu tham số đáng chỉnh hơn nhiều là độ sâu tối đa và số điểm tối thiểu trong mỗi lá, vì chúng quyết định độ linh hoạt của cây.

Cách xây cây như trên là **tham lam**: tại mỗi nút, nó chọn phép chia tốt nhất lúc đó mà không xét ảnh hưởng tới các phép chia phía dưới. Tìm cây tối ưu toàn cục là bài toán NP-khó (Hyafil và Rivest, 1976). Vì vậy một cây đơn lẻ có phương sai cao: chỉ cần vài điểm dữ liệu thay đổi là phép chia ở gốc có thể đổi, kéo theo toàn bộ cấu trúc phía dưới thay đổi.

### 3.3. Bagging, rừng ngẫu nhiên và boosting

Phương sai cao là điểm yếu khi dùng một cây, nhưng lại làm cây trở thành thành phần phù hợp cho phương pháp tập hợp. Có hai cách kết hợp, và chúng tác động vào hai thành phần khác nhau của phân rã ở Chương 2.

Cách thứ nhất là **bagging** (bootstrap aggregating; Breiman, 1996): huấn luyện $M$ mô hình, mỗi mô hình trên một mẫu bootstrap, tức $n$ điểm rút có hoàn lại từ $n$ điểm huấn luyện, rồi lấy trung bình dự đoán. Theo [Ví dụ 3.1 của *Nền tảng*](nentang-ch03.html), nếu mỗi mô hình có phương sai $\sigma^2$ và mỗi cặp có hệ số tương quan $\rho$ thì trung bình của $M$ mô hình có phương sai

$$\rho\,\sigma^2 + \frac{1 - \rho}{M}\,\sigma^2.$$

Khi $M$ tăng, số hạng thứ hai tiến về 0 và chỉ còn lại $\rho\,\sigma^2$. Công thức này nói hai điều. Thêm mô hình không bao giờ làm phương sai tăng, nên thêm cây vào bagging không gây overfitting. Và muốn giảm phương sai tiếp thì phải giảm tương quan $\rho$ giữa các mô hình. **Rừng ngẫu nhiên** (random forest; Breiman, 2001) làm đúng việc đó: tại mỗi nút, cây chỉ được chọn phép chia trong một tập con ngẫu nhiên các đặc trưng, thường là $\sqrt d$ đặc trưng với bài toán phân loại, nên các cây khác nhau hơn và ít tương quan hơn. Còn độ chệch thì bagging gần như không thay đổi, vì trung bình của các mô hình có cùng kỳ vọng với từng mô hình.

Cách thứ hai là **boosting**: huấn luyện các mô hình tuần tự, mỗi mô hình mới sửa phần sai của tổng các mô hình trước. **Gradient boosting** (Friedman, 2001) viết điều này dưới dạng

$$F_m(x) = F_{m-1}(x) + \eta\, h_m(x),$$

trong đó cây $h_m$ được huấn luyện để xấp xỉ gradient âm của hàm mất mát theo giá trị dự đoán, $-\partial \mathcal{L}(y, F)/\partial F$ tại $F = F_{m-1}(x)$, và $\eta$ là tốc độ học, thường từ 0,01 tới 0,3. Với mất mát bình phương, gradient âm chính là phần dư $y - F_{m-1}(x)$, nên mỗi cây mới học phần dư của các cây trước. Với bốn căn hộ và cây một lần chia, $F_0$ là giá trung bình 2,25 tỉ, và phần dư là $-0{,}75$; $-0{,}15$; $0{,}15$; $0{,}75$. Cây đầu tiên khớp phần dư này bằng phép chia ở 70 m², dự đoán $-0{,}45$ cho hai căn nhỏ và $0{,}45$ cho hai căn lớn. Với $\eta = 0{,}1$, $F_1$ chỉ đi một phần mười quãng đường đó: 2,205 tỉ cho hai căn nhỏ, 2,295 tỉ cho hai căn lớn, và sai số bình phương trung bình giảm từ 0,2925 xuống 0,254. Phần dư mới lớn nhất ở hai căn ngoài cùng, nên cây thứ hai chia ở 50 m² để tách căn 40 m², và cây thứ ba chia ở 90 m² để tách căn 100 m². Đây là gradient descent trong không gian hàm: thay vì cập nhật tham số, mỗi bước cộng thêm một hàm theo hướng làm giảm mất mát. Vì mỗi cây chỉ cần sửa một phần nhỏ, các cây trong boosting thường rất nông, độ sâu từ 2 tới 6. Tổ hợp của nhiều cây nông có độ chệch thấp hơn nhiều so với từng cây, nên boosting chủ yếu giảm độ chệch.

### 3.4. Đo tác dụng của bagging và boosting

Phân tích trên có thể kiểm tra trực tiếp. Thí nghiệm trong `code/models/experiments.py` dùng cùng hàm thật và mức nhiễu như Chương 2, mỗi tập huấn luyện 120 điểm, và đo độ chệch² cùng phương sai của ba mô hình trên 40 tập huấn luyện độc lập: một cây sâu 8, bagging 100 cây sâu 8, và gradient boosting 100 cây sâu 2 với tốc độ học 0,1.

![Hình 4](figs/models04_ensemble.png)

**Hình 4.** Độ chệch² và phương sai của ba mô hình trên cùng bài toán.

| Mô hình | Độ chệch² | Phương sai | Tổng (không tính nhiễu) |
|---|---|---|---|
| Một cây, sâu 8 | 0,0031 | 0,1089 | 0,1120 |
| Bagging, 100 cây sâu 8 | 0,0016 | **0,0534** | 0,0550 |
| Gradient boosting, 100 cây sâu 2 | 0,0032 | 0,0371 | **0,0403** |

Cây sâu 8 đủ linh hoạt để có độ chệch rất thấp, nên sai số của nó gần như hoàn toàn là phương sai: phương sai gấp khoảng 35 lần độ chệch². Bagging giảm phương sai đi một nửa, từ 0,1089 xuống 0,0534, và gần như không đổi độ chệch, đúng như phân tích ở Mục 3.3. Phương sai không giảm tới 0 dù có 100 cây, vì các cây được huấn luyện trên các mẫu bootstrap của cùng một tập dữ liệu nên vẫn tương quan với nhau; phần còn lại đó chính là số hạng $\rho\sigma^2$.

Gradient boosting cho tổng sai số thấp nhất. Mỗi cây sâu 2 chỉ có 4 lá, nên nếu dùng một mình thì có độ chệch rất cao. Cộng 100 cây như vậy hạ độ chệch xuống ngang cây sâu 8, trong khi phương sai giữ ở mức thấp vì mỗi cây rất đơn giản và tốc độ học 0,1 làm mỗi cây chỉ đóng góp một phần nhỏ.

### 3.5. Các hiểu lầm thường gặp

Cây và các phương pháp tập hợp đi kèm khá nhiều hiểu lầm phổ biến. Một hiểu lầm thấy được ngay trên bốn căn hộ: tiếp tục ví dụ ở Mục 3.3 tới 100 cây thì sai số huấn luyện gần như bằng 0. Tổng các cây khi đó là một hàm bậc thang đi qua đúng bốn mức giá, và nó định giá căn 85 m² là 2,4 tỉ, bằng căn 80 m², trong khi đường thẳng định giá 2,61 tỉ. Bốn điểm dữ liệu không đủ cho 100 cây. Bảng dưới gom những hiểu lầm hay gặp nhất cùng điều đúng tương ứng.

| Phát biểu | Thực tế |
|---|---|
| "Cây quyết định cần chuẩn hoá đặc trưng." | Không cần: cây bất biến với mọi phép biến đổi đơn điệu của từng đặc trưng. |
| "Rừng ngẫu nhiên không bị overfitting." | Thêm cây không gây overfitting, nhưng cây quá sâu trên dữ liệu nhiễu vẫn làm rừng overfitting. |
| "Thêm cây vào gradient boosting luôn tốt hơn." | Không: boosting có thể overfitting khi số cây quá lớn. Số cây phải chọn bằng tập xác thực, thường kết hợp dừng sớm. |
| "Độ quan trọng đặc trưng tính từ cây là đáng tin." | Độ quan trọng tính theo tổng độ giảm độ không thuần nhất thiên về các đặc trưng có nhiều giá trị khác nhau. Permutation importance (Breiman, 2001) hoặc SHAP (Lundberg và Lee, 2017) đáng tin hơn. |
| "Boosting chỉ dùng được với cây." | Boosting dùng được với mọi mô hình yếu; cây phổ biến vì rẻ và xử lý tốt đặc trưng hỗn hợp. |
| "XGBoost là một thuật toán khác gradient boosting." | XGBoost (Chen và Guestrin, 2016) là một cài đặt hiệu quả của gradient boosting, có thêm regularization trên cấu trúc cây và cách xử lý giá trị thiếu. |

### 3.6. Gradient boosting trong thực hành

Các thư viện phổ biến là XGBoost, LightGBM (Ke và cộng sự, 2017) và CatBoost (Prokhorenkova và cộng sự, 2018). Chúng khác nhau ở chi tiết cài đặt, như cách tìm ngưỡng chia trên dữ liệu lớn hay cách xử lý đặc trưng hạng mục, nhưng có chung các siêu tham số quan trọng.

Quan trọng nhất là cặp tốc độ học và số cây, hai siêu tham số luôn đi cùng nhau: tốc độ học nhỏ cần nhiều cây hơn nhưng thường tổng quát hoá tốt hơn. Cách làm thông dụng là cố định tốc độ học ở một mức nhỏ, chẳng hạn 0,05, rồi chọn số cây bằng dừng sớm trên tập xác thực, tức dừng khi sai số xác thực không giảm sau một số vòng. Tiếp theo là độ phức tạp của mỗi cây, đặt bằng độ sâu tối đa hoặc số lá tối đa và số điểm tối thiểu trong mỗi lá. Cây sâu hơn biểu diễn được tương tác bậc cao hơn giữa các đặc trưng, nhưng làm tăng phương sai.

Hai nhóm siêu tham số còn lại đóng vai trò regularization. Lấy mẫu con cho mỗi cây chỉ dùng một phần ngẫu nhiên các điểm dữ liệu, chẳng hạn 80%, và một phần các đặc trưng; cách này giảm tương quan giữa các cây theo đúng ý tưởng của rừng ngẫu nhiên, và thường cải thiện tổng quát hoá (Friedman, 2002). Regularization trên lá phạt $\ell_2$ lên giá trị của các lá và yêu cầu một độ lợi tối thiểu cho mỗi phép chia.

So với rừng ngẫu nhiên, gradient boosting thường chính xác hơn khi được chỉnh tốt, nhưng nhạy hơn với siêu tham số. Rừng ngẫu nhiên là lựa chọn mặc định an toàn: các cây độc lập nên dễ song song hoá, và thêm cây không làm kết quả kém đi. Với cả hai, cần chia tập xác thực đúng cách, theo thời gian hoặc theo nhóm khi dữ liệu có cấu trúc đó ([Mục 8.5 của *Nền tảng*](nentang-ch08.html)).

### 3.7. Tóm tắt

Cây quyết định chia không gian bằng các phép chia song song với trục, chọn tham lam theo độ giảm độ không thuần nhất. Nhờ vậy nó không cần chuẩn hoá đặc trưng, nhưng không ngoại suy được và có phương sai cao. Bagging lấy trung bình nhiều cây trên các mẫu bootstrap để giảm phương sai, tới giới hạn $\rho\sigma^2$ đặt bởi tương quan giữa các cây, và rừng ngẫu nhiên hạ giới hạn đó bằng cách chọn ngẫu nhiên đặc trưng ở mỗi nút. Gradient boosting cộng dần các cây nông, mỗi cây đi theo gradient âm của hàm mất mát, nên chủ yếu giảm độ chệch. Với bốn căn hộ, cây chọn ngưỡng 70 m² vì nó giảm tổng bình phương sai số từ 1,17 xuống 0,36, và gradient boosting với cây một lần chia khớp đúng bốn mức giá sau 100 cây, thành một hàm bậc thang. Trên bài toán mô phỏng, bagging giảm một nửa phương sai của một cây sâu, còn gradient boosting cho tổng sai số thấp nhất.

Gradient boosting là gradient descent trong không gian hàm, mỗi bước cộng thêm một cây. Chương 4 chuyển sang một họ mô hình mà gradient descent chạy trên chính các tham số: mạng nơ-ron nhiều lớp, bắt đầu từ chỗ hồi quy logistic đã dừng.

---

## 4. Mạng nơ-ron nhiều lớp

Hồi quy logistic ở [Chương 6 của *Nền tảng*](nentang-ch06.html) là một mạng nơ-ron một lớp: một phép biến đổi tuyến tính theo sau là hàm sigmoid. Nó chỉ vẽ được biên quyết định thẳng, nên ở *Nền tảng*, mỗi khi biên cong ta phải tự tạo thêm đặc trưng như bình phương hay tích của các đặc trưng gốc. Mạng nhiều lớp tự học những đặc trưng đó bằng cách chồng nhiều lớp lên nhau. Ta sẽ thấy vì sao giữa các lớp bắt buộc phải có hàm phi tuyến, mạng một lớp ẩn biểu diễn được tới đâu theo định lý xấp xỉ phổ quát, và độ sâu tiết kiệm được bao nhiêu tham số qua một phép dựng tường minh.

### 4.1. Từ mô hình tuyến tính tới mạng nhiều lớp

Một lớp tuyến tính tính $Wx + b$. Chồng hai lớp tuyến tính không tạo ra gì mới, vì

$$W_2(W_1 x + b_1) + b_2 = (W_2 W_1)\,x + (W_2 b_1 + b_2)$$

vẫn là một phép biến đổi affine của $x$. Chồng bao nhiêu lớp tuyến tính cũng chỉ tương đương một lớp. Muốn nhiều lớp biểu diễn được nhiều hơn một lớp, phải xen giữa chúng một hàm phi tuyến.

> **Định nghĩa 4.1 (Mạng nơ-ron truyền thẳng nhiều lớp).** Một **mạng nơ-ron truyền thẳng nhiều lớp** (multilayer perceptron, MLP) có $L$ lớp tính
> $$\begin{aligned} h^{(0)} &= x, \\ h^{(l)} &= \phi\big(W^{(l)} h^{(l-1)} + b^{(l)}\big), \qquad l = 1, \dots, L-1, \\ \hat y &= W^{(L)} h^{(L-1)} + b^{(L)}, \end{aligned}$$
> trong đó $\phi$ là hàm kích hoạt áp dụng cho từng phần tử. Các $h^{(l)}$ với $1 \le l \le L-1$ gọi là **lớp ẩn**, số phần tử của mỗi lớp ẩn là **bề rộng** của lớp đó.

Lớp cuối không có hàm kích hoạt vì nó được chọn theo bài toán. Với hồi quy, $\hat y$ được dùng trực tiếp và hàm mất mát là bình phương sai số. Với phân loại nhị phân, $\hat y$ đi qua sigmoid; với nhiều lớp, qua softmax; và hàm mất mát là cross-entropy. Cả hai cặp lựa chọn đều đến từ ước lượng hợp lý cực đại ([Chương 10 của *Nền tảng*](nentang-ch10.html)): bình phương sai số ứng với nhiễu Gauss, cross-entropy ứng với phân phối Bernoulli hoặc phân phối phân loại.

Một lớp có $d_{\text{in}}$ đầu vào và $d_{\text{out}}$ đầu ra có $d_{\text{in}} d_{\text{out}} + d_{\text{out}}$ tham số. Chẳng hạn, mạng 784–256–10 cho ảnh chữ số MNIST có $784 \cdot 256 + 256 = 200\,960$ tham số ở lớp ẩn và $256 \cdot 10 + 10 = 2\,570$ tham số ở lớp ra, tổng cộng 203 530.

> **Ví dụ 4.1 (Hàm XOR).** Bốn điểm $(0,0)$, $(1,1)$ có nhãn 0 và $(1,0)$, $(0,1)$ có nhãn 1 không tách được bằng một đường thẳng, nên không mô hình tuyến tính nào phân loại đúng cả bốn. Một mạng có hai đơn vị ReLU ở lớp ẩn làm được:
> $$\begin{aligned} h_1 &= \operatorname{ReLU}(x_1 + x_2), \\ h_2 &= \operatorname{ReLU}(x_1 + x_2 - 1), \\ \hat y &= h_1 - 2h_2. \end{aligned}$$
> Kiểm tra: tại $(0,0)$ có $h = (0, 0)$ nên $\hat y = 0$; tại $(1,0)$ và $(0,1)$ có $h = (1, 0)$ nên $\hat y = 1$; tại $(1,1)$ có $h = (2, 1)$ nên $\hat y = 2 - 2 = 0$. Lớp ẩn đã biến đổi bốn điểm sang một không gian mới, trong đó chúng tách được bằng một hàm tuyến tính. Đó là cách hiểu hữu ích về mạng nhiều lớp: các lớp ẩn học một phép biến đổi đặc trưng, lớp cuối là một mô hình tuyến tính trên đặc trưng đã học.

### 4.2. Hàm kích hoạt

Bảng dưới so sánh các hàm kích hoạt thường gặp.

| Hàm | Công thức | Đạo hàm | Ưu điểm | Nhược điểm |
|---|---|---|---|---|
| Sigmoid | $\sigma(z) = 1/(1+e^{-z})$ | $\sigma(z)(1-\sigma(z)) \le 1/4$ | đầu ra trong $(0,1)$ | bão hoà ở hai phía, đạo hàm gần 0; đầu ra không có trung bình 0 |
| Tanh | $\tanh z$ | $1 - \tanh^2 z \le 1$ | đầu ra có trung bình quanh 0 | vẫn bão hoà ở hai phía |
| ReLU | $\max(0, z)$ | 1 nếu $z > 0$, 0 nếu $z < 0$ | không bão hoà phía dương, tính rất rẻ | đơn vị có thể "chết" (luôn có $z < 0$) |
| Leaky ReLU | $\max(\alpha z, z)$, $\alpha \approx 0{,}01$ | $1$ hoặc $\alpha$ | tránh đơn vị chết | thêm một siêu tham số |
| GELU | $z\,\Phi(z)$ | trơn | dùng phổ biến trong Transformer | tính đắt hơn ReLU một chút |
| SiLU (Swish) | $z\,\sigma(z)$ | trơn | dùng trong nhiều mô hình ngôn ngữ (Mục 10.5) | như GELU |

Trong bảng, $\Phi$ là hàm phân phối tích luỹ của phân phối chuẩn tắc. GELU do Hendrycks và Gimpel (2016) đề xuất; SiLU do Elfwing, Uchibe và Doya (2018) và Ramachandran, Zoph và Le (2017) đề xuất độc lập.

Từ bảng có ba điều cần nắm. Điều đầu tiên giải thích vì sao ReLU thay thế sigmoid trong mạng sâu: đạo hàm của ReLU bằng 1 ở phía dương. Gradient qua nhiều lớp là tích của các đạo hàm (Chương 5), mà đạo hàm của sigmoid không vượt quá 1/4, nên qua 10 lớp sigmoid, riêng phần đạo hàm của hàm kích hoạt đã nhân gradient với tối đa $0{,}25^{10} \approx 9{,}5 \times 10^{-7}$. Đây là một nguồn của gradient tiêu biến (Chương 6).

Điều thứ hai là ReLU có một kiểu hỏng riêng, gọi là đơn vị chết. Nếu một bước cập nhật lớn đẩy hệ số chặn của một đơn vị xuống đủ âm, đơn vị đó cho $z < 0$ với mọi đầu vào, đạo hàm bằng 0, và nó không bao giờ được cập nhật lại. Nguyên nhân thường gặp là tốc độ học quá lớn.

Điều thứ ba là hàm kích hoạt của lớp ra khác với lớp ẩn. Lớp ra dùng sigmoid, softmax hoặc không dùng gì, tuỳ bài toán (Mục 4.1). Dùng ReLU ở lớp ra của một bài toán hồi quy có giá trị âm là một lỗi hay gặp: mô hình khi đó không bao giờ dự đoán được giá trị âm.

### 4.3. Định lý xấp xỉ phổ quát

Một câu hỏi tự nhiên là mạng nhiều lớp biểu diễn được những hàm nào. Câu trả lời cổ điển là: gần như mọi hàm, ngay cả khi chỉ có một lớp ẩn.

> **Định lý 4.1 (Xấp xỉ phổ quát).** Cho $\phi$ là một hàm liên tục không phải đa thức. Với mọi hàm liên tục $f$ trên một tập compact $K \subset \mathbb{R}^d$ và mọi $\varepsilon > 0$, tồn tại một mạng một lớp ẩn $g(x) = \sum_{j=1}^{m} a_j\,\phi(w_j^\top x + b_j)$ sao cho $|f(x) - g(x)| < \varepsilon$ với mọi $x \in K$.

Cybenko (1989) chứng minh định lý cho hàm sigmoid, Hornik (1991) mở rộng, và Leshno và cộng sự (1993) chứng minh điều kiện "không phải đa thức" là đủ, nên định lý áp dụng cho cả ReLU.

Định lý nói rằng mạng một lớp ẩn đủ rộng biểu diễn được mọi hàm liên tục, nhưng nó chỉ khẳng định sự tồn tại. Nó không cho biết cần bao nhiêu đơn vị ẩn, và với một số hàm, con số này tăng theo hàm mũ của số chiều $d$ hoặc của độ phức tạp của hàm. Nó không cho biết gradient descent có tìm được bộ trọng số đó hay không. Và nó cũng không cho biết cần bao nhiêu dữ liệu để tìm được bộ trọng số đó mà không overfitting. Ba câu hỏi này mới là những câu quan trọng trong thực tế.

Vì vậy định lý không giải thích được vì sao mạng sâu hoạt động tốt hơn mạng nông mà rộng. Mục 4.4 trả lời một phần câu hỏi này: có những hàm mà mạng sâu biểu diễn bằng số tham số tăng tuyến tính, trong khi mạng một lớp ẩn cần số tham số tăng theo hàm mũ.

### 4.4. Độ sâu và số tham số cần thiết

Phép dựng sau, theo ý tưởng của Telgarsky (2016), cho thấy lợi ích của độ sâu mà không phụ thuộc vào việc huấn luyện có thành công hay không. Ta xuất phát từ **hàm lều** dùng đúng hai đơn vị ReLU:

$$g(x) = 2\operatorname{ReLU}(x) - 4\operatorname{ReLU}\!\left(x - \tfrac12\right).$$

Trên $[0, 1]$, hàm này tăng tuyến tính từ $g(0) = 0$ lên $g(1/2) = 1$ rồi giảm tuyến tính về $g(1) = 0$, nên nó ánh xạ $[0, 1]$ lên $[0, 1]$. Bây giờ hợp $g$ với chính nó $k$ lần. Mỗi nhánh của $g$ phủ trọn đoạn $[0, 1]$, nên mỗi lần hợp làm số đoạn tuyến tính tăng gấp đôi. Với $k = 2$: $g(1/4) = 1/2$ và $g(1/2) = 1$, nên $g(g(1/4)) = 1$; tương tự $g(3/4) = 1/2$ nên $g(g(3/4)) = 1$, còn $g(g(1/2)) = g(1) = 0$. Hàm $g \circ g$ đi từ 0 lên 1 ở $x = 1/4$, xuống 0 ở $x = 1/2$, lên 1 ở $x = 3/4$ rồi về 0 ở $x = 1$: bốn đoạn tuyến tính, từ hai lớp, mỗi lớp hai đơn vị. Kết quả $g^{(k)} = g \circ \dots \circ g$ là một hàm răng cưa có đúng $2^k$ đoạn tuyến tính, biểu diễn bằng một mạng $k$ lớp, mỗi lớp 2 đơn vị ReLU, tổng cộng $6k$ tham số.

Mạng một lớp ẩn cần bao nhiêu đơn vị để biểu diễn cùng hàm đó? Với đầu vào một chiều, mỗi đơn vị ReLU $\operatorname{ReLU}(w x + b)$ tạo ra đúng một điểm gãy tại $x = -b/w$, và tổng của $m$ đơn vị là một hàm tuyến tính từng khúc có tối đa $m$ điểm gãy. Hàm răng cưa có $2^k - 1$ điểm gãy bên trong đoạn, nên mạng một lớp ẩn cần ít nhất $2^k - 1$ đơn vị để biểu diễn chính xác nó.

Thí nghiệm trong `code/models/experiments.py` kiểm tra điều này theo hướng xấp xỉ. Với mỗi bề rộng $m$ trong dãy 2, 4, 8, …, 256, thí nghiệm đặt $m$ điểm gãy cách đều trên $(0, 1)$ rồi giải bình phương tối thiểu cho các hệ số của lớp ra. Như vậy mạng một lớp được cho sẵn nghiệm tối ưu với các điểm gãy cố định, thay vì phải tự tìm bằng gradient descent. Bề rộng cần thiết là bề rộng nhỏ nhất trong dãy cho sai số bình phương trung bình dưới một nửa phương sai của hàm răng cưa, tức tốt hơn rõ rệt so với dự đoán bằng một hằng số.

![Hình 5](figs/models05_depth.png)

**Hình 5.** Trái: sai số nhỏ nhất của mạng một lớp ẩn theo bề rộng, với từng số lớp $k$ của mạng sâu; hai trục theo thang log. Sai số chỉ bắt đầu giảm khi bề rộng đạt cỡ $2^k$. Phải: số tham số cần thiết của hai loại mạng theo $k$.

| $k$ | Số đoạn $2^k$ | Tham số mạng sâu | Bề rộng mạng một lớp cần | Tham số mạng một lớp | Tỉ lệ |
|---|---|---|---|---|---|
| 2 | 4 | 12 | 4 | 14 | 1,2 |
| 3 | 8 | 18 | 8 | 26 | 1,4 |
| 4 | 16 | 24 | 16 | 50 | 2,1 |
| 5 | 32 | 30 | 32 | 98 | 3,3 |
| 6 | 64 | 36 | 64 | 194 | 5,4 |
| 7 | 128 | 42 | 128 | 386 | 9,2 |

Với mọi $k$, bề rộng cần thiết là $2^k$, khớp với cận dưới $2^k - 1$ ở trên. Dưới bề rộng đó, mạng một lớp không làm tốt hơn dự đoán hằng số: với $k = 7$, mọi bề rộng từ 2 tới 64 đều cho sai số 0,083, đúng bằng phương sai $1/12$ của hàm răng cưa. Số tham số của mạng sâu tăng tuyến tính theo $k$, của mạng một lớp tăng theo $2^k$, nên tỉ lệ giữa chúng tăng không giới hạn: 9,2 lần ở $k = 7$, và hơn 26 000 lần ở $k = 20$.

Kết quả này cần được đọc đúng phạm vi. Đây là so sánh về khả năng biểu diễn, không phải về khả năng học: mạng sâu được cho trọng số tường minh, không qua huấn luyện. Trên thực tế, huấn luyện một mạng 7 lớp chỉ có 2 đơn vị mỗi lớp để học hàm này bằng gradient descent gần như luôn thất bại, và Chương 6 giải thích một phần nguyên nhân. Hàm răng cưa cũng là trường hợp thuận lợi nhất cho độ sâu, vì nó có cấu trúc lặp lại mà phép hợp hàm khai thác trực tiếp; kết quả không có nghĩa là mạng càng sâu càng tốt với mọi dữ liệu, và với dữ liệu dạng bảng, thêm lớp thường không giúp gì (Mục 1.2). Cuối cùng, độ sâu không làm tăng những gì mạng biểu diễn được về nguyên tắc, vì theo Định lý 4.1 mạng một lớp đủ rộng đã biểu diễn được mọi hàm liên tục. Thứ độ sâu mang lại là khả năng biểu diễn một số hàm bằng ít tham số hơn rất nhiều.

### 4.5. Thiết kế MLP trong thực hành

Với dữ liệu không có cấu trúc không gian hay thời gian, một số lựa chọn mặc định thường là điểm khởi đầu tốt. Đầu vào cần được chuẩn hoá: đưa mỗi đặc trưng số về trung bình 0, độ lệch chuẩn 1, dùng thống kê tính trên tập huấn luyện. Khác với cây quyết định, mạng nơ-ron nhạy với thang đo của đặc trưng, vì thang đo ảnh hưởng tới số điều kiện của bài toán tối ưu ([Mục 5.3 của *Nền tảng*](nentang-ch05.html)). Mạng nên có 2 tới 4 lớp ẩn, bề rộng bằng nhau hoặc giảm dần; mạng sâu hơn cần kết nối tắt và chuẩn hoá (Chương 6). Hàm kích hoạt nên là ReLU hoặc GELU, và thuật toán tối ưu là Adam hoặc AdamW với tốc độ học khoảng $10^{-3}$ (Mục 5.4). Để chống overfitting, dừng sớm dựa trên sai số xác thực là biện pháp rẻ và hiệu quả nhất, nên làm trước; sau đó mới tới weight decay và dropout với tỉ lệ 0,1 tới 0,3 (Mục 6.6).

Trên dữ liệu dạng bảng, nên so sánh MLP với gradient boosting (Mục 3.6) trước khi kết luận, vì như Mục 1.2 đã nêu, gradient boosting thường tốt hơn.

### 4.6. Tóm tắt

Mạng nơ-ron nhiều lớp chồng các phép biến đổi affine xen với hàm kích hoạt phi tuyến; không có hàm phi tuyến thì chồng bao nhiêu lớp cũng chỉ bằng một lớp. Các lớp ẩn học một phép biến đổi đặc trưng, như ví dụ XOR cho thấy, còn lớp ra là một mô hình tuyến tính trên đặc trưng đã học, với hàm kích hoạt và hàm mất mát chọn theo bài toán. ReLU thay sigmoid trong mạng sâu vì đạo hàm của nó bằng 1 ở phía dương. Định lý xấp xỉ phổ quát bảo đảm mạng một lớp ẩn đủ rộng biểu diễn được mọi hàm liên tục, nhưng không nói gì về số đơn vị cần thiết hay về việc học. Với hàm răng cưa $2^k$ đoạn, mạng sâu cần $6k$ tham số, trong khi mạng một lớp cần cỡ $3 \cdot 2^k$.

Biểu diễn được một hàm chưa có nghĩa là học được nó. Để học, ta cần gradient của hàm mất mát theo mọi tham số của mạng, và một cách dùng gradient đó để cập nhật tham số. Chương 5 trình bày cả hai việc.

---

## 5. Lan truyền ngược và các thuật toán tối ưu

Huấn luyện một mạng nơ-ron gồm hai việc tách biệt: tính gradient của hàm mất mát theo mọi tham số, và dùng gradient đó để cập nhật tham số. Lan truyền ngược làm việc thứ nhất; SGD, momentum và Adam làm việc thứ hai. Người mới hay gộp hai việc làm một, nói "huấn luyện bằng lan truyền ngược" như thể lan truyền ngược là một thuật toán học. Tách chúng ra giúp trả lời những câu hỏi rất thực tế: vì sao tính gradient của một mô hình hàng tỉ tham số chỉ tốn cỡ ba lần tính hàm mất mát, vì sao huấn luyện tốn bộ nhớ hơn suy luận nhiều, và vì sao Adam phải giữ thêm hai giá trị cho mỗi tham số.

### 5.1. Lan truyền ngược

> **Định nghĩa 5.1 (Lan truyền ngược).** **Lan truyền ngược** (backpropagation) là thuật toán tính gradient của một hàm mất mát vô hướng theo mọi tham số của mạng, bằng cách áp dụng quy tắc dây chuyền theo thứ tự từ đầu ra ngược về đầu vào và dùng lại các giá trị trung gian đã tính ở lượt xuôi.

Lan truyền ngược không phải là thuật toán học. Nó chỉ tính gradient; dùng gradient để cập nhật tham số thế nào là việc của thuật toán tối ưu (Mục 5.4). Hai thành phần độc lập với nhau: cùng một gradient có thể dùng cho SGD hay Adam, và cùng một thuật toán tối ưu nhận được gradient tính bằng lan truyền ngược hay bằng cách khác.

Thuật toán gồm hai lượt. **Lượt xuôi** tính đầu ra từ đầu vào, lần lượt qua từng lớp, và lưu lại mọi giá trị trung gian: đầu vào của mỗi lớp, giá trị trước và sau hàm kích hoạt. **Lượt ngược** bắt đầu từ $\partial \mathcal{L}/\partial \mathcal{L} = 1$ và đi ngược qua từng lớp. Với một lớp $z = Wh + b$, $a = \phi(z)$, khi đã có $\delta_a = \partial \mathcal{L}/\partial a$ thì

$$\delta_z = \delta_a \odot \phi'(z), \qquad \frac{\partial \mathcal{L}}{\partial W} = \delta_z\, h^\top, \qquad \frac{\partial \mathcal{L}}{\partial b} = \delta_z, \qquad \delta_h = W^\top \delta_z.$$

Vector $\delta_h$ lại là đầu vào cho lớp phía trước, và mỗi công thức chỉ cần các giá trị $z$, $h$ đã lưu ở lượt xuôi. Rumelhart, Hinton và Williams (1986) phổ biến thuật toán này cho mạng nơ-ron; trong toán học ứng dụng, nó là trường hợp riêng của vi phân tự động chế độ ngược.

![Hình 6](figs/models06_backprop.png)

**Hình 6.** Lượt xuôi tính và lưu lại các giá trị trung gian; lượt ngược nhân dần các đạo hàm cục bộ theo thứ tự ngược lại, dùng lại các giá trị đã lưu.

Ví dụ sau thực hiện đầy đủ một lượt xuôi, một lượt ngược và một bước cập nhật trên mạng nhỏ nhất có lớp ẩn, để thấy từng công thức trên hoạt động thế nào.

> **Ví dụ 5.1.** Mạng có đầu vào $x = (1, 2)$, lớp ẩn 2 đơn vị ReLU và một đầu ra tuyến tính, hàm mất mát $\mathcal{L} = \tfrac12(\hat y - y)^2$ với $y = 1$. Tham số:
> $$\begin{gathered} W_1 = \begin{pmatrix} 0{,}1 & 0{,}2 \\ -0{,}3 & 0{,}1 \end{pmatrix}, \quad b_1 = (0, 0), \\ w_2 = (0{,}5;\; -0{,}4), \quad b_2 = 0{,}2. \end{gathered}$$
>
> **Lượt xuôi.** $z = W_1 x + b_1 = (0{,}1 + 0{,}4;\; -0{,}3 + 0{,}2) = (0{,}5;\; -0{,}1)$. Qua ReLU: $h = (0{,}5;\; 0)$. Đầu ra $\hat y = w_2^\top h + b_2 = 0{,}25 + 0 + 0{,}2 = 0{,}45$. Mất mát $\mathcal{L} = \tfrac12(0{,}45 - 1)^2 = 0{,}15125$.
>
> **Lượt ngược.** $\partial \mathcal{L}/\partial \hat y = \hat y - y = -0{,}55$. Lớp ra: $\partial \mathcal{L}/\partial w_2 = -0{,}55 \cdot h = (-0{,}275;\; 0)$, $\partial \mathcal{L}/\partial b_2 = -0{,}55$, và $\delta_h = -0{,}55 \cdot w_2 = (-0{,}275;\; 0{,}22)$. Qua ReLU: $\delta_z = \delta_h \odot \mathbb{1}[z > 0] = (-0{,}275;\; 0)$. Lớp ẩn: $\partial \mathcal{L}/\partial W_1 = \delta_z\, x^\top = \begin{pmatrix} -0{,}275 & -0{,}55 \\ 0 & 0 \end{pmatrix}$ và $\partial \mathcal{L}/\partial b_1 = (-0{,}275;\; 0)$.
>
> **Kiểm tra bằng sai phân.** Tăng rồi giảm $W_1[1,2]$ một lượng $10^{-4}$: mất mát thành $0{,}151195$ và $0{,}151305$, nên đạo hàm xấp xỉ $(0{,}151195 - 0{,}151305)/(2 \cdot 10^{-4}) = -0{,}55$, khớp với lượt ngược.
>
> **Cập nhật.** Một bước gradient descent với $\eta = 0{,}1$ đưa $\hat y$ từ 0,45 lên 0,606 và mất mát từ 0,151 xuống 0,078.

Ví dụ cho thấy một chi tiết quan trọng: đơn vị ẩn thứ hai có $z_2 = -0{,}1 < 0$, nên đạo hàm của ReLU tại đó bằng 0, và toàn bộ hàng thứ hai của $W_1$ nhận gradient bằng 0. Đơn vị nào không kích hoạt với một đầu vào thì không học được gì từ đầu vào đó. Nếu một đơn vị không kích hoạt với mọi đầu vào, nó không bao giờ được cập nhật nữa; đó chính là đơn vị ReLU chết ở Mục 4.2. Cách kiểm tra bằng sai phân, và cách chọn bước sai phân, được trình bày ở [Mục 5.5 của *Nền tảng*](nentang-ch05.html).

### 5.2. Thứ tự nhân các ma trận Jacobi

Vì sao phải đi ngược mà không đi xuôi? Theo quy tắc dây chuyền, với mạng $x \to h_1 \to \dots \to h_{L-1} \to \mathcal{L}$ trong đó $\mathcal{L}$ là một số, gradient theo đầu vào là tích các ma trận Jacobi:

$$\frac{\partial \mathcal{L}}{\partial x} = \underbrace{\frac{\partial \mathcal{L}}{\partial h_{L-1}}}_{1 \times m}\;\underbrace{\frac{\partial h_{L-1}}{\partial h_{L-2}}}_{m \times m}\;\cdots\;\underbrace{\frac{\partial h_1}{\partial x}}_{m \times d}.$$

Phép nhân ma trận có tính kết hợp, nên có thể nhân theo bất kỳ thứ tự nào, nhưng chi phí rất khác nhau. Nhân từ trái sang phải, tức chế độ ngược của lan truyền ngược, thì kết quả trung gian luôn là một vector hàng $1 \times m$, và mỗi bước chỉ là một phép nhân vector với ma trận, tốn $O(m^2)$. Nhân từ phải sang trái, tức chế độ xuôi, thì kết quả trung gian là một ma trận $m \times d$, và mỗi bước là một phép nhân ma trận với ma trận, tốn $O(m^2 d)$. Với 10 lớp có $m = d = 1\,000$, chế độ ngược tốn khoảng $10 \times 10^6 = 10^7$ phép nhân, chế độ xuôi khoảng $10 \times 10^9 = 10^{10}$, chênh nhau 1 000 lần, đúng bằng số đầu vào $d$.

Tổng quát hơn, với một hàm từ $n$ đầu vào ra $k$ đầu ra, chế độ ngược tính toàn bộ gradient với chi phí cỡ $k$ lần tính hàm, còn chế độ xuôi cần cỡ $n$ lần (Baydin và cộng sự, 2018). Huấn luyện mạng nơ-ron có $k = 1$, vì hàm mất mát là một số, còn $n$ bằng số tham số, có thể tới hàng tỉ, nên chế độ ngược rẻ hơn hàng tỉ lần. Trên thực tế, một lượt ngược tốn khoảng gấp đôi một lượt xuôi (Mục 12.4), nên tính gradient chỉ tốn cỡ ba lần tính hàm mất mát.

Ngược lại, khi cần đạo hàm của nhiều đầu ra theo ít đầu vào, chế độ xuôi rẻ hơn. Các thư viện như PyTorch và JAX cài đặt cả hai chế độ.

### 5.3. Bộ nhớ cho lan truyền ngược

Lượt ngược cần các giá trị trung gian của lượt xuôi, nên chúng phải được giữ trong bộ nhớ cho tới khi lượt ngược dùng xong. Phần bộ nhớ này, gọi là **bộ nhớ kích hoạt**, tỉ lệ với

$$\text{số lớp} \times \text{kích thước lô} \times \text{độ dài chuỗi} \times \text{bề rộng} \times \text{số byte mỗi số}.$$

Bộ nhớ kích hoạt không tỉ lệ với số tham số mà tỉ lệ với kích thước lô và độ dài chuỗi. Vì vậy khi huấn luyện bị hết bộ nhớ, cách xử lý đầu tiên thường là giảm kích thước lô, không phải thu nhỏ mô hình.

Có hai kỹ thuật đổi thời gian tính lấy bộ nhớ. **Gradient checkpointing** (Chen và cộng sự, 2016) chỉ lưu giá trị trung gian tại một số lớp mốc; các lớp khác được tính lại từ mốc gần nhất khi lượt ngược cần tới. Đặt mốc cách nhau $\sqrt{L}$ lớp thì bộ nhớ kích hoạt giảm từ $O(L)$ xuống $O(\sqrt L)$, đổi lại phải tính thêm khoảng một lượt xuôi, tức thời gian tăng khoảng một phần ba. **Tích luỹ gradient** (gradient accumulation) chia một lô lớn thành nhiều lô nhỏ, cộng dồn gradient của các lô nhỏ rồi mới cập nhật một lần. Kết quả tương đương huấn luyện với lô lớn, bộ nhớ chỉ bằng của lô nhỏ, nhưng không nhanh hơn.

### 5.4. SGD, momentum và Adam

Có gradient rồi, ta chuyển sang việc thứ hai. Gradient descent cập nhật $\theta_{t+1} = \theta_t - \eta\, g_t$ với $g_t = \nabla_\theta \mathcal{L}(\theta_t)$ ([Chương 5 của *Nền tảng*](nentang-ch05.html)). Với mạng nơ-ron, $L$ là trung bình trên hàng triệu điểm dữ liệu, nên tính gradient đầy đủ ở mỗi bước là quá đắt. Cách giải quyết là **SGD theo mini-batch**: ước lượng $g_t$ bằng gradient trên một lô nhỏ $B$ điểm chọn ngẫu nhiên. Ước lượng này không chệch, và phương sai của nó giảm theo $1/B$. Nhiễu của ước lượng không hoàn toàn có hại: nhiều nghiên cứu cho thấy nó giúp tránh một số điểm cực tiểu hẹp và có tác dụng như một dạng regularization, dù cơ chế chính xác vẫn đang được nghiên cứu.

Cải tiến đầu tiên là **momentum**, giữ một trung bình trượt của các gradient:

$$v_{t+1} = \beta v_t + g_t, \qquad \theta_{t+1} = \theta_t - \eta\, v_{t+1}.$$

Trong một thung lũng hẹp và dài, gradient đổi dấu liên tục theo chiều ngang thung lũng và giữ cùng dấu theo chiều dọc. Trung bình trượt triệt tiêu phần dao động ngang và cộng dồn phần dọc, nên với $\beta = 0{,}9$, bước đi theo hướng ổn định lớn gấp khoảng $1/(1-\beta) = 10$ lần. [Mục 5.3 của *Nền tảng*](nentang-ch05.html) đã đo được tác dụng này: số vòng lặp giảm từ bậc $\kappa$ xuống bậc $\sqrt\kappa$.

**Adam** (Kingma và Ba, 2015) đi xa hơn: giữ trung bình trượt của cả gradient và bình phương gradient, rồi chia từng toạ độ cho căn bậc hai của trung bình bình phương, nên mỗi tham số có một bước đi riêng:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2,$$
$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}, \qquad \theta_{t+1} = \theta_t - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}.$$

Các giá trị mặc định là $\beta_1 = 0{,}9$, $\beta_2 = 0{,}999$, $\epsilon = 10^{-8}$.

Hai phép chia cho $1 - \beta^t$ gọi là **hiệu chỉnh độ chệch** (bias correction), và tên gọi này đúng theo nghĩa thống kê của Mục 2.2. Vì khởi tạo $m_0 = 0$, khai triển công thức truy hồi được $m_t = (1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i} g_i$. Nếu các $g_i$ có cùng kỳ vọng $\mathbb{E}[g]$ thì

$$\mathbb{E}[m_t] = (1-\beta_1)\big(1 + \beta_1 + \dots + \beta_1^{t-1}\big)\,\mathbb{E}[g] = (1 - \beta_1^t)\,\mathbb{E}[g].$$

Như vậy $m_t$ là ước lượng chệch về phía 0 của $\mathbb{E}[g]$, và chia cho $1 - \beta_1^t$ cho ước lượng không chệch. Cùng lập luận cho $\mathbb{E}[v_t] = (1 - \beta_2^t)\,\mathbb{E}[g^2]$. Hai độ chệch này không bù trừ cho nhau, vì $v_t$ bị kéo về 0 mạnh hơn nhiều. Ở bước đầu, $m_1 = 0{,}1\,g_1$ và $v_1 = 0{,}001\,g_1^2$, nên nếu không hiệu chỉnh, bước đi là $\eta \cdot 0{,}1\,g_1/\sqrt{0{,}001\,g_1^2} \approx 3{,}16\,\eta$ theo dấu của $g_1$, trong khi bước đã hiệu chỉnh là đúng $\eta$. Với gradient không đổi, tỉ số giữa hai bước là $(1 - \beta_1^t)/\sqrt{1 - \beta_2^t}$: bằng 6,5 ở bước 10, 3,2 ở bước 100 và vẫn còn 1,26 ở bước 1 000, vì $1 - \beta_2^t$ tăng rất chậm. Bỏ hiệu chỉnh vì vậy làm các bước đầu lớn bất thường, không phải nhỏ đi; Kingma và Ba (2015) nêu đúng điều này. Khi $t$ lớn, $\beta^t \to 0$ và phép hiệu chỉnh không còn tác dụng gì.

Cuối cùng, **AdamW** (Loshchilov và Hutter, 2019) sửa cách Adam xử lý weight decay. Nếu thêm $\lambda\|\theta\|^2$ vào hàm mất mát rồi dùng Adam, gradient của thành phần phạt $2\lambda\theta$ cũng bị chia cho $\sqrt{\hat v_t}$, nên tham số có gradient lớn lại bị phạt ít hơn, ngược với mục đích của regularization. AdamW tách weight decay ra khỏi gradient và trừ trực tiếp $\eta\lambda\theta_t$ ở mỗi bước. AdamW là lựa chọn mặc định khi huấn luyện Transformer. Bảng dưới so sánh hai nhóm thuật toán.

| | SGD với momentum | Adam, AdamW |
|---|---|---|
| Độ nhạy với tốc độ học | cao | thấp hơn nhiều |
| Tốc độ giảm mất mát ban đầu | chậm hơn | nhanh hơn |
| Tổng quát hoá | trên một số bài toán thị giác, tốt hơn (Wilson và cộng sự, 2017) | thường tốt, là mặc định cho Transformer |
| Bộ nhớ trạng thái | 1 giá trị mỗi tham số | 2 giá trị mỗi tham số |

Dòng cuối quan trọng khi huấn luyện mô hình lớn. Adam giữ $m$ và $v$ cho mọi tham số, thường ở FP32. Với mô hình 7 tỉ tham số, riêng trạng thái Adam đã chiếm $7 \times 10^9 \times 2 \times 4 = 56$ GB. Mục 12.5 sẽ tính đầy đủ bộ nhớ huấn luyện.

### 5.5. Lịch tốc độ học

Tốc độ học là siêu tham số quan trọng nhất khi huấn luyện mạng nơ-ron, và cách thay đổi nó theo thời gian cũng quan trọng gần như vậy. Phần lớn lịch tốc độ học hiện nay bắt đầu bằng **warmup**: tăng tuyến tính tốc độ học từ gần 0 lên giá trị đích trong vài trăm tới vài nghìn bước đầu. Với Adam, $\hat v_t$ ở các bước đầu được ước lượng từ rất ít gradient nên rất nhiễu, và bước đi có thể lớn bất thường; warmup tránh được những bước như vậy. Warmup cũng cần thiết với Transformer dùng post-LN (Mục 10.2).

Sau warmup, tốc độ học được giảm dần. Lịch phổ biến khi huấn luyện mô hình ngôn ngữ là **cosine decay** (Loshchilov và Hutter, 2017), giảm tốc độ học theo nửa chu kỳ hàm cosin về gần 0 ở bước cuối. Trong thị giác máy tính, lịch giảm theo bậc thang, tức chia tốc độ học cho 10 tại một số mốc định trước, vẫn còn phổ biến.

Khi tăng kích thước lô lên $k$ lần, gradient ít nhiễu hơn nên có thể dùng tốc độ học lớn hơn. Với SGD, Goyal và cộng sự (2017) dùng **quy tắc tuyến tính**: nhân tốc độ học với $k$, kèm warmup, và huấn luyện được ResNet-50 trên ImageNet với lô 8 192 ảnh mà không giảm độ chính xác. Với Adam, Malladi và cộng sự (2022) lập luận rằng quy tắc phù hợp là nhân với $\sqrt{k}$. Cả hai là quy tắc kinh nghiệm có cơ sở lý thuyết một phần, và đều không còn đúng khi lô quá lớn.

### 5.6. Các hiểu lầm thường gặp

Bảng dưới gom những hiểu lầm hay gặp về lan truyền ngược và tối ưu.

| Phát biểu | Thực tế |
|---|---|
| "Lan truyền ngược là thuật toán học." | Nó chỉ tính gradient. Việc cập nhật tham số do SGD, Adam hay thuật toán tối ưu khác đảm nhận. |
| "Mạng nơ-ron hay bị kẹt ở cực tiểu địa phương xấu." | Trong không gian nhiều chiều, điểm yên ngựa phổ biến hơn nhiều so với cực tiểu địa phương xấu (Dauphin và cộng sự, 2014), và thực nghiệm cho thấy các cực tiểu mà gradient descent tìm được trong mạng lớn thường có mất mát gần nhau. |
| "Adam luôn tốt hơn SGD." | Adam giảm mất mát nhanh hơn và ít phải chỉnh; SGD với momentum có thể tổng quát hoá tốt hơn trên một số bài toán thị giác. |
| "Gradient bằng 0 nghĩa là đã hội tụ." | Cũng có thể là điểm yên ngựa, đơn vị ReLU đã chết, hoặc gradient đã tiêu biến qua nhiều lớp. |
| "Lô càng lớn càng tốt." | Lô rất lớn làm mất phần nhiễu có ích, cần chỉnh lại tốc độ học, và lợi ích về thời gian giảm dần khi lô tăng. |

### 5.7. Tóm tắt

Lan truyền ngược tính gradient của hàm mất mát theo mọi tham số bằng quy tắc dây chuyền, đi từ đầu ra về đầu vào và dùng lại các giá trị đã lưu ở lượt xuôi. Nhân các ma trận Jacobi theo chiều ngược làm mọi kết quả trung gian là vector, nên với hàm mất mát là một số, tính toàn bộ gradient chỉ tốn cỡ ba lần tính hàm. Đổi lại, các giá trị trung gian phải được giữ trong bộ nhớ, và phần bộ nhớ kích hoạt này tăng theo kích thước lô và độ dài chuỗi. Gradient được dùng bởi SGD theo mini-batch, momentum hoặc Adam; Adam chia bước đi theo từng tham số, cần hiệu chỉnh độ chệch ở các bước đầu và giữ hai giá trị trạng thái cho mỗi tham số, còn AdamW tách weight decay khỏi gradient. Tốc độ học thường đi theo lịch warmup rồi giảm dần.

Với các công cụ này, ví dụ tính tay đã chạy đúng trên một mạng hai lớp. Khi mạng có hàng chục lớp, một vấn đề mới xuất hiện ngay trong công thức lan truyền ngược: gradient là tích của nhiều ma trận, và tích đó có thể co về 0 hoặc phình ra rất nhanh. Chương 6 xét vấn đề này.

---

## 6. Khởi tạo, chuẩn hoá và kết nối tắt

Ở Mục 4.4, một mạng 7 lớp, mỗi lớp 2 đơn vị, biểu diễn được hàm răng cưa 128 đoạn, nhưng gradient descent gần như không bao giờ tìm ra bộ trọng số đó. Một phần nguyên nhân nằm trong công thức lan truyền ngược của Chương 5: gradient tại một lớp là tích của các ma trận Jacobi của mọi lớp phía sau, và tích của nhiều thừa số hơi nhỏ hơn hoặc hơi lớn hơn 1 sẽ co về 0 hoặc phình ra rất nhanh. Trong một thời gian dài, đây là trở ngại khiến các mạng sâu vài chục lớp rất khó huấn luyện. Ba kỹ thuật giữ độ lớn của tín hiệu ổn định đã gỡ trở ngại đó: khởi tạo trọng số đúng phương sai, chuẩn hoá, và kết nối tắt. Một thí nghiệm trên mạng 40 lớp sẽ đo tác dụng của từng kỹ thuật, và cho một kết quả hay bị hiểu sai: kết nối tắt một mình không đủ.

### 6.1. Tín hiệu co lại hoặc phình ra qua nhiều lớp

Theo Mục 5.2, gradient theo đầu vào của một lớp là tích của các ma trận Jacobi của mọi lớp phía sau. Nếu mỗi lớp nhân độ lớn của gradient với một hệ số trung bình $\gamma$, thì qua $L$ lớp gradient bị nhân với khoảng $\gamma^L$. Với $\gamma = 0{,}9$ và $L = 50$, hệ số là $0{,}005$; với $\gamma = 1{,}1$, hệ số là 117. Chỉ cần $\gamma$ lệch khỏi 1 một chút, gradient ở các lớp đầu sẽ **tiêu biến** về gần 0 hoặc **bùng nổ**. Lượt xuôi gặp vấn đề tương tự với độ lớn của các giá trị kích hoạt.

Ba kỹ thuật trong chương này đều nhằm giữ $\gamma$ gần 1, và chúng bổ sung cho nhau chứ không thay thế nhau.

### 6.2. Khởi tạo Xavier và He

Kỹ thuật đầu tiên tác động ngay từ bước khởi tạo. Xét một lớp $z = Wa$ với $W \in \mathbb{R}^{n_{\text{out}} \times n_{\text{in}}}$. Giả sử các phần tử của $W$ độc lập, có trung bình 0 và phương sai $\sigma_W^2$, và độc lập với $a$. Khi đó mỗi thành phần của $z$ có

$$\operatorname{Var}(z_i) = \sum_{j=1}^{n_{\text{in}}} \operatorname{Var}(W_{ij} a_j) = n_{\text{in}}\,\sigma_W^2\,\mathbb{E}[a_j^2].$$

Nếu $a$ có trung bình 0 thì $\mathbb{E}[a_j^2] = \operatorname{Var}(a_j)$, và muốn phương sai không đổi qua lớp thì cần $\sigma_W^2 = 1/n_{\text{in}}$. Áp dụng cùng lập luận cho lượt ngược, nơi gradient nhân với $W^\top$, cho điều kiện $\sigma_W^2 = 1/n_{\text{out}}$. **Khởi tạo Xavier** (Glorot và Bengio, 2010) dung hoà hai điều kiện bằng

$$\sigma_W^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}.$$

Lập luận trên đúng với hàm kích hoạt gần tuyến tính quanh 0 như tanh. Với ReLU thì phải sửa lại: nếu $z$ đối xứng quanh 0 thì $a = \operatorname{ReLU}(z)$ bằng 0 với một nửa số giá trị, nên

$$\mathbb{E}[a^2] = \tfrac12\,\mathbb{E}[z^2] = \tfrac12 \operatorname{Var}(z).$$

Thay vào công thức trên, phương sai giữ nguyên qua các lớp khi $n_{\text{in}}\,\sigma_W^2 \cdot \tfrac12 = 1$, tức

$$\sigma_W^2 = \frac{2}{n_{\text{in}}}.$$

Đây là **khởi tạo He** (He và cộng sự, 2015). Hệ số 2 đến trực tiếp từ việc ReLU loại bỏ một nửa phân phối. Với các lớp rộng 128 đơn vị ở Mục 6.3, khởi tạo He lấy $\sigma_W^2 = 2/128 \approx 0{,}0156$, tức độ lệch chuẩn 0,125, còn Xavier lấy $2/256$, tức độ lệch chuẩn khoảng 0,088, nhỏ hơn $\sqrt2$ lần. Dùng Xavier cho mạng ReLU thì mỗi lớp nhân độ lớn tín hiệu với khoảng $1/\sqrt2 \approx 0{,}71$, và qua 40 lớp còn $2^{-20} \approx 10^{-6}$. Nguyên tắc chung là khởi tạo phải khớp với hàm kích hoạt: Xavier cho tanh và sigmoid, He cho ReLU và các biến thể gần giống.

### 6.3. Thí nghiệm với mạng 40 lớp

Thí nghiệm trong `code/models/experiments.py` dùng một mạng 40 lớp, mỗi lớp rộng 128 đơn vị ReLU, với một lô 128 đầu vào ngẫu nhiên. Sau lượt xuôi, một gradient ngẫu nhiên có chuẩn 1 được truyền ngược từ lớp cuối, và thí nghiệm đo chuẩn của gradient theo giá trị kích hoạt $\|\partial \mathcal{L}/\partial a_l\|$ tại từng lớp. Kết quả lấy trung bình trên 5 lần khởi tạo và chia cho giá trị ở lớp cuối. Trọng số được khởi tạo với phương sai $\text{gain}^2/n_{\text{in}}$, nên khởi tạo He ứng với $\text{gain} = \sqrt 2$.

Sáu cách dựng lớp được so sánh. Ba cách đầu là mạng thường, mỗi lớp tính $a_{l+1} = \operatorname{ReLU}(W a_l)$, với ba mức gain 0,5, $\sqrt2$ và 2,0. Cách thứ tư thêm LayerNorm trước ReLU: $a_{l+1} = \operatorname{ReLU}(\operatorname{LN}(W a_l))$. Cách thứ năm thêm kết nối tắt mà không chuẩn hoá: $a_{l+1} = a_l + \operatorname{ReLU}(W a_l)$. Cách thứ sáu dùng cả hai, $a_{l+1} = a_l + \operatorname{ReLU}(\operatorname{LN}(W a_l))$, với chuẩn hoá nằm trong nhánh còn đường tắt đi thẳng, cùng nguyên tắc với pre-LN của Transformer (Mục 10.2). Ba cách sau đều dùng khởi tạo He.

![Hình 7](figs/models07_vanish.png)

**Hình 7.** Chuẩn của gradient tại từng lớp, chia cho giá trị ở lớp cuối, với sáu cách dựng lớp. Trục dọc theo thang log.

| Cách dựng lớp | Gradient tại lớp 1 so với lớp 40 | Kết luận |
|---|---|---|
| Khởi tạo với gain 0,5 (quá nhỏ) | $1{,}2 \times 10^{-18}$ | tiêu biến |
| Khởi tạo He, gain $\sqrt2$ | $1{,}42$ | ổn định |
| Khởi tạo với gain 2,0 (quá lớn) | $1{,}5 \times 10^{6}$ | bùng nổ |
| He và LayerNorm | $1{,}15$ | ổn định |
| He và kết nối tắt, không chuẩn hoá | $2{,}4 \times 10^{8}$ | bùng nổ |
| He, chuẩn hoá trong nhánh và kết nối tắt | $2{,}22$ | ổn định |

Ba dòng đầu cho thấy khởi tạo có ảnh hưởng rất lớn. Trên cùng một kiến trúc, chỉ đổi gain từ 0,5 sang 2,0 làm gradient ở lớp đầu thay đổi từ $10^{-18}$ tới $10^{6}$, tức 24 bậc độ lớn. Với gain 0,5, mỗi lớp nhân gradient với khoảng $0{,}5/\sqrt2 \approx 0{,}35$, và $0{,}35^{40} \approx 10^{-18}$. Khởi tạo He thì giữ gradient gần như không đổi, với tỉ lệ 1,42 sau 40 lớp, đúng như lập luận ở Mục 6.2.

Dòng thứ năm là kết quả hay bị hiểu sai nhất: kết nối tắt một mình làm gradient bùng nổ. Với khởi tạo He, nhánh $\operatorname{ReLU}(Wa_l)$ có độ lớn cùng cỡ với $a_l$, nên mỗi lớp cộng thêm vào tín hiệu một lượng xấp xỉ bằng chính nó, và độ lớn của tín hiệu tăng gần gấp đôi sau mỗi lớp. Ở lượt ngược, Jacobi của mỗi lớp là $I + J_l$ với $J_l$ có độ lớn cỡ 1, và tích của 40 ma trận như vậy tăng theo cấp số nhân. Vì vậy mạng ResNet luôn có chuẩn hoá đi kèm kết nối tắt.

Dòng cuối cho thấy cách kết hợp đúng: chuẩn hoá trong nhánh kèm đường tắt đi thẳng cho gradient ổn định, với tỉ lệ 2,22. Chuẩn hoá giữ đầu ra của mỗi nhánh ở độ lớn cố định, nên tín hiệu chỉ tăng theo kiểu cộng dồn thay vì nhân lên, còn đường tắt bảo đảm gradient luôn có một đường đi không qua ma trận trọng số nào. Đây là cấu hình mà các Transformer hiện nay dùng (Mục 10.2).

Trang thí nghiệm tương tác của giáo trình cho phép tự chọn gain, số lớp và các thành phần để quan sát các đường trong Hình 7.

### 6.4. BatchNorm, LayerNorm và RMSNorm

Kỹ thuật thứ hai là chuẩn hoá. Các lớp chuẩn hoá đưa giá trị kích hoạt về trung bình 0, phương sai 1, rồi nhân với một hệ số $\gamma$ và cộng một độ dịch $\beta$ học được, để mạng vẫn biểu diễn được mọi thang đo nếu cần:

$$\operatorname{Norm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta.$$

Các phương pháp khác nhau ở tập giá trị dùng để tính $\mu$ và $\sigma^2$, như bảng dưới cho thấy.

| | BatchNorm (Ioffe và Szegedy, 2015) | LayerNorm (Ba, Kiros và Hinton, 2016) |
|---|---|---|
| Tính thống kê trên | mọi mẫu trong lô, riêng cho từng đặc trưng (từng kênh) | mọi đặc trưng của một mẫu, riêng cho từng mẫu |
| Phụ thuộc kích thước lô | có: lô nhỏ cho thống kê nhiễu | không |
| Khác nhau giữa huấn luyện và suy luận | có: suy luận dùng trung bình trượt tích luỹ khi huấn luyện | không |
| Dùng phổ biến trong | mạng tích chập cho ảnh | Transformer, RNN |

Transformer dùng LayerNorm thay cho BatchNorm, và ba hàng đầu của bảng giải thích vì sao. Các chuỗi trong một lô có độ dài khác nhau và được đệm thêm, nên thống kê theo lô trên trục thời gian bị phần đệm làm sai lệch. BatchNorm dùng thống kê của lô khi huấn luyện và trung bình trượt khi suy luận; khi sinh văn bản từng token, mỗi bước chỉ xử lý một vị trí, và khác biệt giữa hai chế độ trở thành một nguồn sai lệch. Ngoài ra, với BatchNorm, dự đoán cho một mẫu phụ thuộc vào các mẫu khác cùng lô, điều không mong muốn khi phục vụ nhiều người dùng cùng lúc.

**RMSNorm** (Zhang và Sennrich, 2019) đơn giản hoá LayerNorm bằng cách bỏ bước trừ trung bình và độ dịch $\beta$, chỉ chia cho căn trung bình bình phương:

$$\operatorname{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}}.$$

Với vector $x = (1;\ 2;\ 3;\ 6)$, $\gamma = 1$ và $\beta = 0$, LayerNorm trừ trung bình 3, chia cho độ lệch chuẩn $\sqrt{3{,}5} \approx 1{,}87$ và cho $(-1{,}07;\ -0{,}53;\ 0;\ 1{,}60)$. RMSNorm chỉ chia cho căn trung bình bình phương $\sqrt{12{,}5} \approx 3{,}54$ và cho $(0{,}28;\ 0{,}57;\ 0{,}85;\ 1{,}70)$: các phần tử giữ nguyên dấu và tỉ lệ với nhau, chỉ thang đo thay đổi. Phép tính rẻ hơn LayerNorm, và thực nghiệm cho chất lượng tương đương. RMSNorm là lựa chọn của nhiều mô hình ngôn ngữ hiện nay, như Llama.

Vì sao chuẩn hoá giúp huấn luyện vẫn còn được thảo luận. Bài báo gốc của BatchNorm giải thích bằng việc giảm "dịch chuyển hiệp biến nội tại" (internal covariate shift), nhưng Santurkar và cộng sự (2018) cho thấy lợi ích chủ yếu đến từ việc làm hàm mất mát trơn hơn, cho phép dùng tốc độ học lớn hơn. Tác dụng giữ ổn định độ lớn của tín hiệu, đo được ở Mục 6.3, là một phần của lời giải thích đó.

### 6.5. Kết nối tắt

Kỹ thuật thứ ba thay $h = F(x)$ bằng

$$h = x + F(x),$$

trong đó $F$ là một khối gồm vài lớp. Có hai cách hiểu bổ sung cho nhau.

Cách hiểu thứ nhất nhìn theo đường đi của gradient. Jacobi của một khối là $\partial h/\partial x = I + \partial F/\partial x$, và số hạng $I$ bảo đảm gradient luôn có một đường đi thẳng từ đầu ra về đầu vào mà không nhân với ma trận trọng số nào. Qua nhiều khối, tích $\prod_l (I + J_l)$ khai triển thành tổng của mọi tích con, trong đó có số hạng $I$, nên không tự co về 0 như $\prod_l J_l$.

Cách hiểu thứ hai nhìn theo bài toán cần học. Thay vì học trực tiếp hàm $H(x)$, khối học phần dư $F(x) = H(x) - x$. Nếu ánh xạ đồng nhất đã là một xấp xỉ tốt, khối chỉ cần học một hiệu chỉnh nhỏ, và khởi tạo $F \approx 0$ đã là một điểm xuất phát tốt. Đây là lập luận của bài báo ResNet (He và cộng sự, 2016), xuất phát từ **vấn đề suy thoái** (degradation problem): trên CIFAR-10, mạng thường 56 lớp có sai số huấn luyện cao hơn mạng 20 lớp. Mạng 56 lớp biểu diễn được mọi hàm mà mạng 20 lớp biểu diễn được, chỉ cần 36 lớp thừa là ánh xạ đồng nhất, nên sai số huấn luyện cao hơn không phải do overfitting mà do khó tối ưu. Kết nối tắt làm ánh xạ đồng nhất trở nên dễ biểu diễn, và mạng ResNet với hơn 100 lớp huấn luyện được bình thường.

> **Lưu ý.** Kết nối tắt ra đời để giải quyết vấn đề suy thoái, tức sai số huấn luyện tăng theo độ sâu, không phải để chống overfitting. Và như Mục 6.3 cho thấy, kết nối tắt cần đi kèm chuẩn hoá thì gradient mới ổn định.

### 6.6. Dropout và các phương pháp regularization khác

Ba kỹ thuật trên giúp huấn luyện được mạng sâu; còn lại là chống overfitting. Phương pháp regularization riêng của mạng nơ-ron được dùng nhiều nhất là **dropout** (Srivastava và cộng sự, 2014). Khi huấn luyện, mỗi đơn vị bị đặt bằng 0 với xác suất $p$, độc lập ở mỗi bước, và các đơn vị còn lại được chia cho $1 - p$ để kỳ vọng của đầu ra không đổi; cách làm này gọi là inverted dropout. Chẳng hạn với $p = 0{,}5$, một lớp có đầu ra $(2;\ 4;\ 6;\ 8)$ có thể bị giữ lại phần tử thứ nhất và thứ ba ở một bước, thành $(4;\ 0;\ 12;\ 0)$ sau khi chia cho $0{,}5$. Mỗi phần tử được nhân đôi với xác suất 0,5 và bằng 0 với xác suất 0,5, nên kỳ vọng của nó đúng bằng giá trị ban đầu. Khi suy luận, dropout bị tắt và mạng dùng mọi đơn vị. Mỗi bước huấn luyện vì vậy dùng một mạng con khác nhau chia sẻ trọng số, và mạng đầy đủ khi suy luận xấp xỉ trung bình của rất nhiều mạng con. Cơ chế giảm phương sai vì thế tương tự bagging ở Mục 3.3, nhưng không phải huấn luyện nhiều mô hình.

Với mô hình ngôn ngữ lớn, vai trò của dropout giảm nhiều. Khi tiền huấn luyện trên kho dữ liệu cực lớn, mỗi mẫu thường chỉ được xem một lần nên overfitting không phải vấn đề chính, và một số mô hình như PaLM (Chowdhery và cộng sự, 2022) không dùng dropout khi tiền huấn luyện. Dropout vẫn hữu ích khi tinh chỉnh trên tập dữ liệu nhỏ. Bảng dưới đặt dropout cạnh các phương pháp regularization khác.

| Phương pháp | Cơ chế | Dùng ở đâu |
|---|---|---|
| Weight decay | phạt chuẩn $\ell_2$ của trọng số ([Mục 9.2 của *Nền tảng*](nentang-ch09.html)); với Adam phải dùng dạng AdamW (Mục 5.4) | hầu hết mọi nơi |
| Dropout | huấn luyện trên các mạng con ngẫu nhiên | MLP, mạng tích chập, tinh chỉnh; ít dùng khi tiền huấn luyện LLM |
| Dừng sớm | dừng khi sai số xác thực ngừng giảm | hầu hết mọi nơi; rẻ nhất |
| Tăng cường dữ liệu | tạo thêm mẫu bằng các biến đổi giữ nguyên nhãn: lật, cắt, xoay ảnh, thêm nhiễu vào âm thanh | ảnh, âm thanh |
| Làm mượt nhãn | thay nhãn one-hot $e_y$ bằng $(1 - \epsilon)\,e_y + \epsilon/K$, để mô hình không bị đẩy tới xác suất tuyệt đối 0 và 1 (Szegedy và cộng sự, 2016) | phân loại nhiều lớp |

Weight decay có một cách hiểu từ *Nền tảng*: regularization $\ell_2$ tương đương ước lượng MAP với tiên nghiệm Gauss trên trọng số ([Chương 10 của *Nền tảng*](nentang-ch10.html)). Dừng sớm cũng có liên hệ với regularization: với hồi quy tuyến tính, dừng gradient descent sau $t$ bước cho kết quả gần giống hồi quy ridge với $\lambda$ tỉ lệ nghịch với $\eta t$.

### 6.7. Tóm tắt

Qua nhiều lớp, gradient bị nhân với tích của các ma trận Jacobi, nên chỉ cần mỗi lớp lệch khỏi hệ số 1 một chút là gradient tiêu biến hoặc bùng nổ. Khởi tạo phải khớp với hàm kích hoạt: Xavier cho tanh, He với phương sai $2/n_{\text{in}}$ cho ReLU. Chuẩn hoá giữ đầu ra mỗi lớp ở độ lớn cố định; Transformer dùng LayerNorm hoặc RMSNorm thay BatchNorm vì không muốn phụ thuộc vào lô. Kết nối tắt cho gradient một đường đi thẳng và giải quyết vấn đề suy thoái, nhưng thí nghiệm với mạng 40 lớp cho thấy nó làm gradient bùng nổ tới $2{,}4 \times 10^{8}$ lần nếu không có chuẩn hoá trong nhánh. Dropout, weight decay và dừng sớm lo phần chống overfitting, dù vai trò của dropout giảm nhiều khi tiền huấn luyện mô hình ngôn ngữ lớn.

Với các kỹ thuật này, ta huấn luyện được mạng hàng trăm lớp. Nhưng mạng kết nối đầy đủ vẫn rất lãng phí khi đầu vào là ảnh: chỉ riêng một lớp nhận ảnh $224 \times 224 \times 3$ và cho 1 000 đơn vị ẩn đã cần khoảng 150 triệu tham số. Chương 7 trình bày một kiến trúc khai thác cấu trúc không gian của ảnh, nhờ đó số tham số giảm đi hàng nghìn lần.

---

## 7. Mạng nơ-ron tích chập

Ảnh có một cấu trúc mà dữ liệu dạng bảng không có: các điểm ảnh gần nhau liên quan với nhau, và một chi tiết như cạnh hay góc có ý nghĩa như nhau dù xuất hiện ở đâu trong ảnh. Một mạng kết nối đầy đủ bỏ qua cấu trúc đó, nên phải học riêng cho từng vị trí và cần số tham số khổng lồ. **Mạng nơ-ron tích chập** (convolutional neural network, CNN) đưa hai tính chất trên vào chính kiến trúc, và nhờ vậy giảm số tham số đi hàng nghìn lần. Ta bắt đầu từ phép tích chập và các phép tính kích thước, tham số, chi phí đi kèm, rồi xem các khối mà CNN hiện đại ghép từ đó, so sánh CNN với Vision Transformer, và cuối cùng là hai bài toán thị giác ngoài phân loại ảnh: phát hiện đối tượng và phân đoạn ảnh.

### 7.1. Phép tích chập

Một lớp kết nối đầy đủ nhận ảnh $224 \times 224 \times 3$ và cho 1 000 đơn vị ẩn cần $224 \cdot 224 \cdot 3 \cdot 1\,000 \approx 1{,}5 \times 10^8$ tham số, chỉ cho một lớp. Mạng tích chập giảm con số đó bằng ba giả định về ảnh, tóm tắt trong bảng dưới.

| Giả định | Cách thực hiện | Tác dụng |
|---|---|---|
| Đặc trưng có tính cục bộ | mỗi đơn vị chỉ nối với một vùng nhỏ $k \times k$ của lớp trước | số kết nối của mỗi đơn vị không phụ thuộc kích thước ảnh |
| Một đặc trưng hữu ích ở một vị trí thì cũng hữu ích ở vị trí khác | **chia sẻ trọng số**: cùng một bộ lọc quét qua mọi vị trí | số tham số không phụ thuộc kích thước ảnh |
| Vị trí chính xác của đặc trưng ít quan trọng hơn sự có mặt của nó | gộp (pooling) hoặc bước nhảy lớn hơn 1 | giảm độ phân giải; bất biến gần đúng với dịch chuyển nhỏ |

> **Định nghĩa 7.1 (Lớp tích chập hai chiều).** Cho đầu vào $X \in \mathbb{R}^{H \times W \times C_{\text{in}}}$ và bộ lọc $K \in \mathbb{R}^{k_h \times k_w \times C_{\text{in}} \times C_{\text{out}}}$. Lớp tích chập với bước nhảy $s$ tính
> $$Z[i, j, o] = b_o + \sum_{c=1}^{C_{\text{in}}} \sum_{u=0}^{k_h - 1} \sum_{v=0}^{k_w - 1} K[u, v, c, o]\; X[s\,i + u,\; s\,j + v,\; c].$$
> Mỗi kênh đầu ra $o$ ứng với một bộ lọc, và giá trị tại vị trí $(i, j)$ là tích vô hướng giữa bộ lọc và vùng ảnh tương ứng.

Về mặt toán học, công thức trên là phép tương quan chéo (cross-correlation), vì bộ lọc không bị lật như trong định nghĩa tích chập của xử lý tín hiệu. Bộ lọc được học từ dữ liệu, nên lật hay không cũng không thay đổi những gì mạng biểu diễn được, và trong học sâu cả hai được gọi chung là tích chập.

Một ví dụ nhỏ cho thấy bộ lọc làm gì. Một ảnh xám cao 3, rộng 5 điểm, có mọi hàng là $(0, 0, 0, 1, 1)$: phần trái tối, phần phải sáng, và một cạnh dọc ở giữa. Bộ lọc $3 \times 3$ có mọi hàng là $(-1, 0, 1)$; tại mỗi vị trí, nó cộng trên ba hàng hiệu giữa điểm ảnh bên phải và điểm ảnh bên trái của cửa sổ. Không đệm và bước nhảy 1, đầu ra có kích thước $1 \times 3$ và bằng $(0, 3, 3)$: bằng 0 ở vùng đồng nhất, bằng 3 ở hai vị trí mà cửa sổ chứa cạnh. Đây là một bộ dò cạnh dọc, và các lớp tích chập đầu tiên của một CNN đã huấn luyện thường học được những bộ dò cạnh theo nhiều hướng như vậy.

Khi thiết kế một mạng tích chập, ba phép tính được dùng liên tục. Phép tính đầu tiên là kích thước đầu ra. Với đầu vào cao $H$, bộ lọc cao $k$, bước nhảy $s$ và phần đệm (padding) $p$ ô mỗi phía, chiều cao đầu ra là

$$H_{\text{out}} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1,$$

và chiều rộng tính tương tự. Phép tính thứ hai là số tham số: một lớp tích chập có $k_h k_w C_{\text{in}} C_{\text{out}} + C_{\text{out}}$ tham số, không phụ thuộc kích thước ảnh. Phép tính thứ ba là số phép tính, và con số này thì có phụ thuộc vào kích thước ảnh: mỗi giá trị đầu ra cần $k_h k_w C_{\text{in}}$ phép nhân và cộng, nên cả lớp tốn khoảng

$$\text{FLOP} \approx 2\, H_{\text{out}}\, W_{\text{out}}\, k_h k_w\, C_{\text{in}}\, C_{\text{out}}.$$

> **Ví dụ 7.1 (Lớp đầu của ResNet-50).** Lớp tích chập đầu tiên nhận ảnh $224 \times 224 \times 3$, dùng 64 bộ lọc $7 \times 7$, bước nhảy 2, phần đệm 3. Kích thước đầu ra là $\lfloor (224 + 6 - 7)/2 \rfloor + 1 = 112$, tức $112 \times 112 \times 64$. Số tham số là $7 \cdot 7 \cdot 3 \cdot 64 = 9\,408$ (lớp này không có hệ số chặn vì có BatchNorm theo sau). Số phép tính là $2 \cdot 112 \cdot 112 \cdot 9\,408 \approx 2{,}4 \times 10^8$ FLOP. So với lớp kết nối đầy đủ ở đầu mục, số tham số nhỏ hơn khoảng 16 000 lần.

Sự tách rời giữa số tham số và số phép tính hay gây nhầm lẫn khi ước lượng chi phí. Một mô hình "nhẹ" về tham số vẫn có thể rất tốn tính toán khi chạy trên ảnh độ phân giải cao, vì số phép tính tăng theo diện tích ảnh.

Chia sẻ trọng số còn làm lớp tích chập **đẳng biến** với phép dịch chuyển (translation equivariant): dịch ảnh đầu vào đi một số ô thì bản đồ đặc trưng đầu ra dịch theo đúng số ô đó, nếu bỏ qua hiệu ứng ở biên. Với ảnh cạnh dọc ở trên, thêm một cột tối vào bên trái, tức dịch cạnh sang phải một ô, thì đầu ra thành $(0, 0, 3, 3)$: hai giá trị 3 dịch theo đúng một ô. Gộp và bước nhảy biến tính đẳng biến này thành **bất biến gần đúng**: một dịch chuyển nhỏ không làm thay đổi đầu ra sau khi gộp. Tính bất biến này chỉ gần đúng; Azulay và Weiss (2019) cho thấy các CNN hiện đại có thể đổi dự đoán khi ảnh chỉ dịch đi một điểm ảnh, vì bước nhảy lớn hơn 1 phá vỡ tính đẳng biến.

### 7.2. Trường tiếp nhận

Mỗi đơn vị của lớp tích chập chỉ nhìn một vùng nhỏ của lớp trước. Muốn nhận ra một đối tượng lớn, mạng phải chồng nhiều lớp để vùng nhìn thấy của các đơn vị ở sâu lớn dần lên.

> **Định nghĩa 7.2 (Trường tiếp nhận).** **Trường tiếp nhận** (receptive field) của một đơn vị là vùng của ảnh đầu vào có ảnh hưởng tới giá trị của đơn vị đó.

![Hình 8](figs/models08_cnn.png)

**Hình 8.** Mỗi lớp tích chập $3 \times 3$ với bước nhảy 1 mở rộng trường tiếp nhận thêm 2 ô theo mỗi chiều.

Với lớp thứ $i$ có bộ lọc kích thước $k_i$ và bước nhảy $s_i$, kích thước trường tiếp nhận tính theo công thức truy hồi

$$r_0 = 1, \qquad r_i = r_{i-1} + (k_i - 1)\prod_{j < i} s_j.$$

Tích các bước nhảy phía trước cho biết một ô ở lớp $i - 1$ ứng với bao nhiêu điểm ảnh của đầu vào. Với bước nhảy 1 và bộ lọc $3 \times 3$ ở mọi lớp, $r_L = 1 + 2L$: trường tiếp nhận chỉ tăng tuyến tính theo số lớp, và phải cần tới 112 lớp mới phủ được ảnh rộng 224 điểm. Vì vậy các CNN thực tế đều giảm độ phân giải sớm: sau mỗi lần bước nhảy 2, mọi lớp phía sau mở rộng trường tiếp nhận nhanh gấp đôi. Chẳng hạn, lớp $7 \times 7$ bước nhảy 2 cho $r = 7$; lớp gộp $3 \times 3$ bước nhảy 2 phía sau đưa lên $r = 7 + 2 \cdot 2 = 11$; mỗi lớp $3 \times 3$ tiếp theo thêm $2 \cdot 4 = 8$ điểm ảnh.

Công thức này dẫn tới hai hệ quả trong thiết kế. Hai lớp $3 \times 3$ có cùng trường tiếp nhận với một lớp $5 \times 5$, nhưng dùng $2 \cdot 9 = 18$ trọng số cho mỗi cặp kênh thay vì 25, và có thêm một hàm phi tuyến ở giữa. Tương tự, ba lớp $3 \times 3$ thay được một lớp $7 \times 7$ với 27 thay vì 49 trọng số. Đây là lập luận của mạng VGG (Simonyan và Zisserman, 2015), và giải thích vì sao bộ lọc lớn ít được dùng trong các mạng sau đó.

Hệ quả thứ hai là trường tiếp nhận hiệu dụng nhỏ hơn nhiều so với con số lý thuyết. Luo và cộng sự (2016) cho thấy ảnh hưởng của các điểm ảnh lên một đơn vị có phân bố gần với phân phối Gauss tập trung ở giữa trường tiếp nhận, và độ rộng hiệu dụng chỉ tăng theo $\sqrt L$ thay vì $L$. Các điểm ảnh ở rìa trường tiếp nhận lý thuyết gần như không có ảnh hưởng.

### 7.3. Các khối thường dùng

Các CNN hiện đại được ghép từ một số khối chuẩn. Bảng dưới liệt kê những khối hay gặp nhất cùng tác dụng của chúng.

| Khối | Cấu tạo | Tác dụng |
|---|---|---|
| Tích chập $1 \times 1$ | bộ lọc kích thước $1 \times 1$ | trộn thông tin giữa các kênh tại từng vị trí; thay đổi số kênh với chi phí thấp |
| Khối cổ chai (bottleneck) | $1 \times 1$ giảm số kênh, $3 \times 3$, rồi $1 \times 1$ tăng lại số kênh | giảm mạnh số phép tính ở cùng độ sâu; dùng trong ResNet-50 trở lên |
| Tích chập tách theo chiều sâu (depthwise separable) | tích chập $k \times k$ riêng cho từng kênh, rồi tích chập $1 \times 1$ | giảm chi phí khoảng $k^2$ lần; là nền tảng của MobileNet (Howard và cộng sự, 2017) |
| Khối dư (residual block) | $h = x + F(x)$ | cho phép mạng sâu hàng trăm lớp (Mục 6.5) |
| Gộp trung bình toàn cục | lấy trung bình mỗi kênh trên toàn bộ không gian trước lớp phân loại | bỏ được lớp kết nối đầy đủ lớn ở cuối mạng; nhận ảnh có kích thước bất kỳ |

Với tích chập tách theo chiều sâu, số trọng số của tích chập thường là $k^2 C_{\text{in}} C_{\text{out}}$, còn của phiên bản tách là $k^2 C_{\text{in}} + C_{\text{in}} C_{\text{out}}$. Tỉ lệ giữa hai con số là

$$\frac{k^2 C_{\text{in}} + C_{\text{in}} C_{\text{out}}}{k^2 C_{\text{in}} C_{\text{out}}} = \frac{1}{C_{\text{out}}} + \frac{1}{k^2}.$$

Với $k = 3$ và $C_{\text{out}} = 256$, tỉ lệ là 0,115, tức rẻ hơn khoảng 8,7 lần, và số phép tính giảm theo cùng tỉ lệ. Đổi lại, mỗi kênh của lớp depthwise chỉ có 9 trọng số, và dải giá trị của các kênh có thể chênh nhau rất nhiều; [Mục 5.3 của *Quantization*](ch05.html) cho thấy đây là loại lớp nhạy nhất khi lượng tử hoá với một thang đo chung cho cả lớp.

### 7.4. CNN và Vision Transformer

Từ năm 2020, CNN có một đối thủ trên chính sân nhà của nó. **Vision Transformer** (ViT; Dosovitskiy và cộng sự, 2021) cắt ảnh thành các mảnh $16 \times 16$ điểm ảnh, chiếu mỗi mảnh thành một vector, rồi xử lý dãy vector như dãy token bằng Transformer (Chương 9). Ảnh $224 \times 224$ cho $14 \times 14 = 196$ token.

Khác biệt giữa hai kiến trúc là khác biệt về thiên kiến quy nạp (Mục 1.4). CNN đưa sẵn vào kiến trúc tính cục bộ và tính đẳng biến với dịch chuyển; đây là giả định đúng với ảnh tự nhiên, nên CNN học được từ lượng dữ liệu vừa phải. ViT gần như không giả định gì về cấu trúc không gian: mọi mảnh có thể tương tác với mọi mảnh ngay từ lớp đầu, và mô hình phải tự học từ dữ liệu những quan hệ mà CNN có sẵn. Kết quả trong bài báo ViT phản ánh đúng đánh đổi này: khi chỉ huấn luyện trên ImageNet (1,3 triệu ảnh), ViT kém các mạng ResNet cùng cỡ; khi tiền huấn luyện trên tập JFT-300M (300 triệu ảnh), ViT vượt ResNet.

Theo ngôn ngữ của Chương 2, thiên kiến quy nạp mạnh ứng với độ chệch cao hơn và phương sai thấp hơn: có lợi khi dữ liệu ít, trở thành giới hạn khi dữ liệu nhiều. Khoảng cách giữa hai họ cũng không cố định. ConvNeXt (Liu và cộng sự, 2022) cho thấy một CNN được hiện đại hoá theo các lựa chọn thiết kế của Transformer cạnh tranh được với ViT ở cùng quy mô, và nhiều kiến trúc hiện nay kết hợp tích chập với attention.

### 7.5. Phát hiện đối tượng

Phân loại ảnh trả lời câu hỏi "ảnh có gì". **Phát hiện đối tượng** (object detection) trả lời thêm câu hỏi "ở đâu": đầu ra là một danh sách **hộp bao** (bounding box), mỗi hộp kèm nhãn lớp và một điểm tin cậy. Số đối tượng thay đổi theo từng ảnh, nên đầu ra không có kích thước cố định như bài toán phân loại, và phần lớn cấu trúc của các bộ phát hiện đến từ việc giải quyết điều này.

Một hộp thường được biểu diễn bằng toạ độ góc trên bên trái và góc dưới bên phải $(x_1, y_1, x_2, y_2)$, hoặc bằng tâm và kích thước $(x, y, w, h)$. Để so sánh hai hộp, cần một thước đo mức trùng khớp.

> **Định nghĩa 7.3 (IoU).** Độ trùng khớp **IoU** (intersection over union) giữa hai hộp $A$ và $B$ là tỉ số giữa diện tích phần giao và diện tích phần hợp:
>
> $$\operatorname{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}.$$
>
> IoU nằm trong $[0, 1]$, bằng 1 khi hai hộp trùng nhau hoàn toàn và bằng 0 khi chúng không giao nhau.

> **Ví dụ 7.2.** Hộp $A = (0, 0, 4, 4)$ và $B = (2, 2, 6, 6)$: phần giao là hình vuông $2 \times 2$, diện tích 4; phần hợp là $16 + 16 - 4 = 28$; nên $\operatorname{IoU} = 4/28 \approx 0{,}143$. Hộp $B' = (1, 0, 5, 4)$ chỉ là $A$ dịch sang phải 1 đơn vị: giao $3 \times 4 = 12$, hợp $20$, IoU $= 0{,}6$. Hộp $B'' = (0, 0, 4, 2)$ là nửa trên của $A$: IoU $= 8/16 = 0{,}5$. Ví dụ cuối cho thấy một hộp chỉ phủ nửa đối tượng vẫn đạt ngưỡng 0,5 thường dùng khi đánh giá.

Các bộ phát hiện dựa trên CNN chia thành hai họ. Họ **hai giai đoạn** (two-stage) đề xuất trước các vùng có khả năng chứa đối tượng, rồi mới phân loại từng vùng và tinh chỉnh hộp. Đại diện đầu tiên là R-CNN (Girshick và cộng sự, 2014), lấy khoảng 2 000 vùng đề xuất từ thuật toán selective search và chạy CNN riêng trên từng vùng, nên rất chậm. Fast R-CNN (Girshick, 2015) chạy CNN một lần cho cả ảnh rồi cắt đặc trưng của từng vùng bằng phép RoI pooling. Faster R-CNN (Ren và cộng sự, 2015) thay selective search bằng một **mạng đề xuất vùng** (region proposal network, RPN) dùng chung đặc trưng với phần phát hiện, nên toàn bộ hệ thống trở thành một mạng nơ-ron huấn luyện được từ đầu tới cuối.

Họ **một giai đoạn** (one-stage) bỏ bước đề xuất vùng và dự đoán trực tiếp hộp cùng lớp trên một lưới dày đặc. YOLO (Redmon và cộng sự, 2016) chia ảnh thành lưới $7 \times 7$; mỗi ô dự đoán 2 hộp và xác suất của 20 lớp, nên đầu ra là một tensor $7 \times 7 \times 30$, và mô hình chạy được 45 khung hình mỗi giây. SSD (Liu và cộng sự, 2016) dự đoán trên nhiều bản đồ đặc trưng có độ phân giải khác nhau, để bản đồ chi tiết bắt đối tượng nhỏ còn bản đồ thô bắt đối tượng lớn.

Bộ một giai đoạn thường nhanh hơn, còn bộ hai giai đoạn từng chính xác hơn. Lin và cộng sự (2017) chỉ ra nguyên nhân chính của khoảng cách này: bộ một giai đoạn đánh giá hàng chục nghìn vị trí trên mỗi ảnh, phần lớn là nền dễ nhận ra, và tổng hàm mất mát của số đông mẫu dễ này lấn át số ít đối tượng thật. Họ đề xuất **focal loss**:

$$\operatorname{FL}(p_t) = -\alpha_t\,(1 - p_t)^{\gamma}\,\log p_t,$$

trong đó $p_t$ là xác suất mô hình gán cho lớp đúng và $\alpha_t$ là trọng số theo lớp. Hệ số $(1 - p_t)^{\gamma}$ giảm trọng số của các mẫu đã được phân loại tốt: với $\gamma = 2$, một mẫu có $p_t = 0{,}9$ nhận trọng số $0{,}01$, tức đóng góp nhỏ hơn 100 lần so với cross-entropy, còn một mẫu khó với $p_t = 0{,}5$ nhận trọng số $0{,}25$. Bộ phát hiện một giai đoạn RetinaNet dùng hàm mất mát này đạt độ chính xác của các bộ hai giai đoạn cùng thời.

Một chi tiết chung của Faster R-CNN, SSD và RetinaNet là **hộp neo** (anchor box). Tại mỗi vị trí trên bản đồ đặc trưng, mô hình đặt một số hộp có kích thước và tỉ lệ cạnh định trước; RPN của Faster R-CNN dùng 3 kích thước và 3 tỉ lệ, tức 9 hộp neo mỗi vị trí. Mạng không dự đoán toạ độ tuyệt đối mà dự đoán độ lệch so với hộp neo:

$$t_x = \frac{x - x_a}{w_a}, \qquad t_y = \frac{y - y_a}{h_a}, \qquad t_w = \log\frac{w}{w_a}, \qquad t_h = \log\frac{h}{h_a},$$

với $(x, y, w, h)$ là tâm và kích thước của hộp, chỉ số $a$ chỉ hộp neo. Chia cho kích thước hộp neo làm độ lệch không phụ thuộc vào kích thước đối tượng, còn logarit làm phép co giãn đối xứng: hộp lớn gấp đôi hay nhỏ đi một nửa cho $t_w$ bằng $+\log 2$ hay $-\log 2$. Khi huấn luyện, mỗi hộp neo được gán nhãn theo IoU với các hộp thật: trong RPN, hộp neo có IoU lớn hơn 0,7 với một hộp thật là mẫu dương, nhỏ hơn 0,3 với mọi hộp thật là mẫu âm, còn ở giữa thì bỏ qua. Các bộ phát hiện không dùng hộp neo (anchor-free) như FCOS (Tian và cộng sự, 2019) và CenterNet (Zhou và cộng sự, 2019) dự đoán trực tiếp từ từng điểm của bản đồ đặc trưng, bớt được các siêu tham số về kích thước và tỉ lệ của hộp neo.

Vì dự đoán trên một lưới dày đặc, một đối tượng thường được nhiều hộp gần nhau cùng phát hiện. Bước **non-maximum suppression** (NMS) lọc các hộp trùng theo thuật toán sau.

1. Sắp xếp các hộp theo điểm tin cậy giảm dần.
2. Đưa hộp có điểm cao nhất vào kết quả.
3. Bỏ mọi hộp còn lại có IoU với hộp vừa chọn lớn hơn một ngưỡng, thường là 0,5.
4. Lặp lại bước 2 và 3 với các hộp còn lại cho tới khi hết.

> **Ví dụ 7.3 (Hình 9, trái).** Sáu hộp dự đoán có điểm A 0,92; B 0,85; D 0,80; C 0,60; E 0,55; F 0,40. IoU giữa A và B là 0,822, giữa A và C là 0,566, giữa A và F là 0,143, giữa D và E là 0,716. Với ngưỡng 0,5: chọn A, bỏ B và C; trong các hộp còn lại D có điểm cao nhất, chọn D và bỏ E; còn F, chọn F. Kết quả là ba hộp A, D, F.

Hộp F chỉ trùng 0,143 với A nên được giữ, vì NMS coi đó là một đối tượng khác. Đây cũng là giới hạn của NMS: hai đối tượng thật đứng sát nhau, như người trong đám đông, có hộp trùng nhau nhiều, và hộp có điểm thấp hơn bị loại nhầm. DETR (Carion và cộng sự, 2020) bỏ hẳn NMS và hộp neo. Một Transformer encoder–decoder nhận đặc trưng của ảnh và 100 vector truy vấn, mỗi truy vấn sinh ra một hộp hoặc "không có đối tượng"; khi huấn luyện, các dự đoán được ghép một–một với hộp thật bằng thuật toán Hungary, nên mỗi đối tượng chỉ có một dự đoán chịu trách nhiệm. Đổi lại, DETR huấn luyện lâu hơn nhiều: bài báo gốc dùng 500 epoch trên COCO.

Còn lại câu hỏi đánh giá một bộ phát hiện thế nào. Cách đánh giá dựa trên precision và recall ([Mục 8.3 của *Nền tảng*](nentang-ch08.html)), với định nghĩa đúng sai qua IoU. Ta xét các hộp dự đoán theo điểm giảm dần. Một hộp là **dương tính thật** nếu IoU của nó với một hộp thật cùng lớp, chưa được ghép với dự đoán nào trước đó, đạt ngưỡng, thường là 0,5; ngược lại nó là **dương tính giả**. Hộp thật không được ghép là âm tính giả. Mỗi hộp thật chỉ được ghép một lần, nên các hộp trùng của cùng một đối tượng bị tính là dương tính giả.

Hạ dần ngưỡng điểm, ta được đường precision–recall. **AP** (average precision) là diện tích dưới đường này sau khi thay mỗi giá trị precision bằng precision lớn nhất ở các mức recall lớn hơn hoặc bằng: $p_{\text{nội suy}}(r) = \max_{r' \ge r} p(r')$. PASCAL VOC 2007 lấy trung bình tại 11 mức recall $0; 0{,}1; \dots; 1$; từ VOC 2010, AP là diện tích tính tại mọi điểm (Everingham và cộng sự, 2010). **mAP** là trung bình AP của các lớp. COCO (Lin và cộng sự, 2014) tính AP tại 101 mức recall và lấy trung bình trên 10 ngưỡng IoU $0{,}50; 0{,}55; \dots; 0{,}95$. Con số này được ký hiệu AP@[0,5:0,95], hoặc gọn là AP trong các bài báo dùng COCO, còn AP50 và AP75 là AP tại một ngưỡng IoU.

Để thấy các quy ước này ảnh hưởng ra sao, script `code/models/thi_giac.py` mô phỏng một bộ phát hiện trên 300 ảnh kích thước $100 \times 100$, mỗi ảnh 1 tới 4 đối tượng, tổng cộng 765 hộp thật. Mỗi đối tượng được phát hiện với xác suất 0,9 bằng một hộp lệch ngẫu nhiên khoảng 6% kích thước ở mỗi cạnh; với xác suất 0,3 có thêm một hộp trùng lệch hơn và điểm thấp hơn; ngoài ra mỗi ảnh có trung bình 0,6 hộp báo nhầm ở chỗ không có gì. Không có mô hình nào được huấn luyện: mô phỏng cô lập đúng phần cần hiểu là cách ghép hộp, lọc trùng và tính AP.

![Hình 9](figs/models15_detect.png)

**Hình 9.** Trái: NMS với ngưỡng IoU 0,5 trên sáu hộp của Ví dụ 7.3; nét liền là hộp được giữ, nét đứt là hộp bị loại. Phải: đường precision–recall của bộ phát hiện mô phỏng, khi không lọc trùng và khi có NMS, ở ngưỡng IoU 0,5 và 0,75.

| Cách đánh giá | AP (mọi điểm) | AP (101 điểm) | Recall lớn nhất |
|---|---|---|---|
| IoU 0,5, không lọc trùng | 0,885 | 0,879 | 0,928 |
| IoU 0,5, có NMS | 0,913 | 0,908 | 0,916 |
| IoU 0,75, có NMS | 0,708 | 0,708 | 0,792 |

Với NMS, AP theo cách tính của COCO tại các ngưỡng IoU 0,5; 0,6; 0,7; 0,8; 0,9 lần lượt là 0,908; 0,894; 0,829; 0,425; 0,014, và trung bình trên 10 ngưỡng là AP@[0,5:0,95] = 0,569.

Hai dòng đầu cho thấy NMS tăng AP nhưng loại nhầm một ít hộp đúng. Không lọc trùng, các hộp trùng bị tính là dương tính giả và AP50 là 0,885; sau NMS, AP50 tăng lên 0,913. Nhưng recall lớn nhất giảm từ 0,928 xuống 0,916, vì NMS đã loại một số hộp đúng của các đối tượng đứng sát nhau, đúng giới hạn nêu ở trên.

Dòng cuối và dãy AP theo ngưỡng cho thấy ngưỡng IoU thay đổi con số rất nhiều. Cùng một bộ phát hiện, AP giảm từ 0,908 ở ngưỡng 0,5 xuống 0,708 ở 0,75 và gần 0 ở 0,9: hộp lệch khoảng 6% kích thước mỗi cạnh là đủ tốt cho ngưỡng 0,5 nhưng hiếm khi đạt 0,9. AP@[0,5:0,95] thưởng cho việc định vị chính xác, nên hai bộ phát hiện có cùng AP50 có thể khác xa nhau theo thước đo này. Khi đọc một con số AP, cần biết nó được tính theo quy ước nào.

### 7.6. Phân đoạn ảnh

Phát hiện đối tượng khoanh vùng bằng hộp, nhưng một chiếc xe hay một khối u không có hình hộp. **Phân đoạn ảnh** (image segmentation) đi xa hơn: gán nhãn cho từng điểm ảnh thay vì cho cả ảnh hay cho một hộp. Có ba bài toán phân đoạn, khác nhau ở việc có phân biệt các đối tượng cùng lớp hay không.

| Bài toán | Đầu ra | Ví dụ |
|---|---|---|
| Phân đoạn ngữ nghĩa (semantic segmentation) | mỗi điểm ảnh một nhãn lớp; không phân biệt các đối tượng cùng lớp | đường, vỉa hè, xe, người trong ảnh từ camera của xe |
| Phân đoạn thể hiện (instance segmentation) | một mặt nạ riêng cho từng đối tượng | từng chiếc xe, từng người |
| Phân đoạn toàn cảnh (panoptic segmentation; Kirillov và cộng sự, 2019) | mọi điểm ảnh có nhãn lớp, và các đối tượng đếm được có thêm định danh riêng | bản đồ đầy đủ của một cảnh |

Khó khăn đầu tiên là độ phân giải. Một CNN phân loại giảm độ phân giải dần qua các lớp, chẳng hạn từ $224 \times 224$ xuống $7 \times 7$, để trường tiếp nhận đủ lớn (Mục 7.2), trong khi phân đoạn lại cần đầu ra ở độ phân giải gốc. FCN (fully convolutional network; Long, Shelhamer và Darrell, 2015) giải quyết bằng ba thay đổi. Các lớp kết nối đầy đủ ở cuối được thay bằng tích chập, để mạng cho ra một bản đồ lớp thô thay vì một vector. Bản đồ đó được tăng độ phân giải bằng tích chập chuyển vị. Và đặc trưng từ các lớp đầu, nơi độ phân giải còn cao, được cộng thêm vào để khôi phục chi tiết ở biên đối tượng.

**Tích chập chuyển vị** (transposed convolution) là phép tăng kích thước không gian. Với đầu vào kích thước $i$, bộ lọc $k$, bước nhảy $s$ và phần đệm $p$, đầu ra có kích thước

$$o = (i - 1)\,s - 2p + k.$$

Ví dụ, $i = 7$, $k = 4$, $s = 2$, $p = 1$ cho $o = 6 \cdot 2 - 2 + 4 = 14$: độ phân giải tăng gấp đôi. Khi $k$ không chia hết cho $s$, các vùng ảnh hưởng của bộ lọc chồng lên nhau không đều và có thể tạo hoa văn ô cờ trên ảnh đầu ra (Odena và cộng sự, 2016); một cách thay thế phổ biến là phóng to bằng nội suy song tuyến tính rồi mới tích chập.

Kiến trúc phân đoạn được dùng rộng rãi nhất là **U-Net** (Ronneberger, Fischer và Brox, 2015). Nó gồm một nhánh mã hoá giảm độ phân giải và tăng số kênh như một CNN phân loại, và một nhánh giải mã đối xứng tăng dần độ phân giải. Ở mỗi mức độ phân giải, nhánh giải mã ghép (concatenate) bản đồ đặc trưng của nhánh mã hoá ở cùng mức: đặc trưng sâu cho biết "đây là gì", đặc trưng nông cho biết chính xác "ở đâu". U-Net được thiết kế cho ảnh y sinh, nơi chỉ có vài chục ảnh có nhãn (một bài toán trong bài báo có 30 ảnh huấn luyện), và dựa nhiều vào tăng cường dữ liệu. Ngày nay kiến trúc này còn là mạng khử nhiễu trong nhiều mô hình khuếch tán (Ho và cộng sự, 2020; [Chương 11 của *Biểu diễn & Căn chỉnh*](bieudien-ch11.html)).

Một cách khác để có trường tiếp nhận lớn mà không phải giảm độ phân giải là **tích chập giãn** (dilated convolution, atrous convolution), chèn $d - 1$ ô trống giữa các phần tử của bộ lọc. Bộ lọc $k \times k$ với hệ số giãn $d$ phủ một vùng kích thước $k + (k - 1)(d - 1)$ mà vẫn chỉ có $k^2$ trọng số: bộ lọc $3 \times 3$ với $d = 2$ phủ $5 \times 5$, với $d = 4$ phủ $9 \times 9$. DeepLab (Chen và cộng sự, 2018) dùng tích chập giãn, và chạy song song nhiều hệ số giãn khác nhau để bắt ngữ cảnh ở nhiều tỉ lệ.

Với phân đoạn thể hiện, Mask R-CNN (He và cộng sự, 2017) thêm vào Faster R-CNN một nhánh dự đoán mặt nạ nhị phân $28 \times 28$ cho mỗi vùng, song song với nhánh phân loại và nhánh hộp. Nó cũng thay RoI pooling bằng **RoIAlign**. RoI pooling làm tròn toạ độ của vùng về lưới của bản đồ đặc trưng, gây lệch tới nửa ô; sai lệch đó chấp nhận được với hộp nhưng không chấp nhận được khi cần chính xác tới từng điểm ảnh, nên RoIAlign lấy giá trị tại toạ độ thực bằng nội suy song tuyến tính.

Đánh giá phân đoạn cũng có bẫy riêng. Độ chính xác theo điểm ảnh (pixel accuracy) dễ gây hiểu lầm khi các lớp mất cân bằng: nếu nền chiếm 95% số điểm ảnh, một mô hình gán mọi điểm là nền đạt 95%. Thước đo chuẩn cho phân đoạn ngữ nghĩa là IoU theo lớp và **mIoU**, trung bình IoU của các lớp, với IoU tính trên tập điểm ảnh thay vì trên hộp. Trong phân đoạn ảnh y tế, người ta hay dùng **hệ số Dice** $= \dfrac{2|A \cap B|}{|A| + |B|}$. Dice và IoU liên hệ bởi $\text{Dice} = \dfrac{2\,\text{IoU}}{1 + \text{IoU}}$, nên Dice luôn lớn hơn hoặc bằng IoU: IoU 0,3; 0,5; 0,7; 0,9 ứng với Dice 0,462; 0,667; 0,824; 0,947. Vì vậy khi so sánh kết quả từ hai bài báo, cần chú ý chúng dùng thước đo nào.

Hàm mất mát cơ bản là cross-entropy trên từng điểm ảnh. Khi đối tượng chỉ chiếm một phần rất nhỏ của ảnh, như khối u trên ảnh chụp, cross-entropy bị số đông điểm ảnh nền chi phối, giống tình huống của focal loss ở Mục 7.5. Các cách xử lý gồm đặt trọng số theo lớp, dùng focal loss, và dùng **Dice loss** (Milletari và cộng sự, 2016): $1 - \text{Dice}$, với Dice tính từ xác suất dự đoán thay vì nhãn đã làm tròn để lấy đạo hàm được. Dice loss chỉ quan tâm tới mức trùng khớp của vùng đối tượng, nên không bị số điểm ảnh nền lấn át.

Xu hướng mới nhất trong phân đoạn là mô hình nền. SAM (Segment Anything; Kirillov và cộng sự, 2023) nhận một ảnh và một gợi ý, có thể là một vài điểm hoặc một hộp, rồi trả về mặt nạ của đối tượng được chỉ tới. Mô hình được huấn luyện trên bộ dữ liệu SA-1B gồm hơn 1 tỉ mặt nạ trên 11 triệu ảnh, và dùng được trên nhiều loại ảnh chưa từng thấy mà không cần huấn luyện thêm. Trong ứng dụng, SAM thường được ghép với một bộ phát hiện: bộ phát hiện đưa ra hộp, SAM biến hộp thành mặt nạ. Đây là cùng xu hướng với mô hình ngôn ngữ ở các Chương 9 tới 12: một mô hình nền huấn luyện trên dữ liệu rất lớn, rồi được dùng cho nhiều nhiệm vụ cụ thể.

### 7.7. Tóm tắt

Mạng tích chập dựa trên ba giả định về ảnh: tính cục bộ, chia sẻ trọng số giữa các vị trí, và tính bất biến gần đúng với dịch chuyển nhỏ. Nhờ vậy số tham số của một lớp không phụ thuộc kích thước ảnh, dù số phép tính thì có, như lớp đầu của ResNet-50 với 9 408 tham số và khoảng $2{,}4 \times 10^8$ FLOP. Trường tiếp nhận tăng tuyến tính theo số lớp, nên mạng phải giảm độ phân giải sớm, và chồng các lớp $3 \times 3$ rẻ hơn dùng bộ lọc lớn. So với ViT, CNN có thiên kiến quy nạp mạnh hơn, nên tốt hơn khi dữ liệu vừa phải và kém hơn khi dữ liệu rất lớn. Phát hiện đối tượng thêm câu hỏi "ở đâu", với IoU để so hộp, hộp neo, NMS để lọc trùng và AP để đánh giá; mô phỏng cho thấy AP giảm từ 0,908 ở ngưỡng IoU 0,5 xuống 0,014 ở 0,9, nên con số AP chỉ có nghĩa khi biết quy ước tính. Phân đoạn gán nhãn cho từng điểm ảnh, với FCN, U-Net, tích chập giãn và các thước đo IoU, Dice.

CNN khai thác cấu trúc không gian của ảnh. Văn bản, tiếng nói và chuỗi thời gian có một cấu trúc khác: thứ tự. Chương 8 xét họ mô hình đầu tiên được thiết kế cho dữ liệu dạng chuỗi.

---

## 8. Mạng nơ-ron hồi quy

Văn bản, tiếng nói và chuỗi thời gian là dữ liệu dạng chuỗi có độ dài thay đổi, và ý nghĩa của mỗi phần tử phụ thuộc vào những gì đứng trước nó. Một mạng truyền thẳng nhận đầu vào có kích thước cố định nên không xử lý trực tiếp được loại dữ liệu này. **Mạng nơ-ron hồi quy** (recurrent neural network, RNN) giải quyết bằng một ý tưởng đơn giản: đọc chuỗi từng phần tử một và giữ một trạng thái tóm tắt những gì đã đọc. Ý tưởng này gặp lại đúng vấn đề gradient của Chương 6, ở dạng nặng hơn, và cuối cùng bị Transformer thay thế vì một giới hạn mà không kỹ thuật huấn luyện nào sửa được.

### 8.1. Mô hình

> **Định nghĩa 8.1 (Mạng nơ-ron hồi quy).** Với chuỗi đầu vào $x_1, \dots, x_T$, một RNN cập nhật **trạng thái ẩn** $h_t$ theo
> $$h_t = \tanh\big(W_{hh}\, h_{t-1} + W_{xh}\, x_t + b\big), \qquad \hat y_t = W_{hy}\, h_t,$$
> với $h_0$ cho trước, thường bằng 0. Cùng các ma trận $W_{hh}$, $W_{xh}$, $W_{hy}$ được dùng ở mọi bước.

Dùng chung trọng số ở mọi bước là dạng chia sẻ trọng số theo trục thời gian, giống chia sẻ trọng số theo không gian của tích chập (Mục 7.1). Nhờ đó RNN xử lý được chuỗi dài tuỳ ý với số tham số cố định, và cách xử lý một mẫu ở vị trí thứ 5 giống ở vị trí thứ 500.

Trường hợp một chiều cho thấy trạng thái ẩn nhớ và quên ra sao. Lấy $h_t = \tanh(0{,}5\,h_{t-1} + x_t)$ với $h_0 = 0$ và chuỗi đầu vào $1, 0, 0$. Ba trạng thái lần lượt là $h_1 = \tanh 1 = 0{,}76$, $h_2 = \tanh 0{,}38 = 0{,}36$ và $h_3 = \tanh 0{,}18 = 0{,}18$: sau mỗi bước, dấu vết của đầu vào đầu tiên còn chưa tới một nửa, vì nó bị nhân với hệ số 0,5 rồi đi qua tanh.

Để huấn luyện, RNN được **trải ra theo thời gian** thành một mạng truyền thẳng $T$ lớp dùng chung trọng số, rồi áp dụng lan truyền ngược lên mạng đó. Thuật toán này gọi là lan truyền ngược theo thời gian (backpropagation through time). Gradient theo trọng số dùng chung là tổng các gradient tại mọi bước.

### 8.2. Gradient tiêu biến và bùng nổ theo thời gian

Mạng trải ra theo thời gian là một mạng sâu $T$ lớp, nên nó gặp đúng vấn đề của Chương 6. Theo quy tắc dây chuyền, đạo hàm của trạng thái ở bước $T$ theo trạng thái ở bước $t < T$ là

$$\frac{\partial h_T}{\partial h_t} = \prod_{i=t+1}^{T} \frac{\partial h_i}{\partial h_{i-1}} = \prod_{i=t+1}^{T} \operatorname{diag}\big(\tanh'(z_i)\big)\, W_{hh},$$

với $z_i$ là giá trị trước hàm tanh ở bước $i$. Với RNN một chiều ở Mục 8.1, mỗi thừa số là $0{,}5 \cdot \tanh'(z_i)$, nên $\partial h_3/\partial h_1 = 0{,}5\,(1 - 0{,}36^2) \cdot 0{,}5\,(1 - 0{,}18^2) \approx 0{,}21$, và qua 20 bước đạo hàm không vượt quá $0{,}5^{20} \approx 10^{-6}$. Tình huống giống Mục 6.1, nhưng nghiêm trọng hơn ở một điểm: cùng một ma trận $W_{hh}$ được nhân lặp lại $T - t$ lần, thay vì mỗi lớp một ma trận khác nhau. Pascanu, Mikolov và Bengio (2013) chỉ ra rằng nếu giá trị suy biến lớn nhất của $W_{hh}$ nhỏ hơn 1 thì gradient chắc chắn tiêu biến theo hàm mũ, vì $|\tanh'| \le 1$; còn muốn gradient bùng nổ thì bán kính phổ của $W_{hh}$ phải lớn hơn 1.

Thí nghiệm trong `code/models/experiments.py` đo trực tiếp hiện tượng này. Một RNN tanh 64 chiều có $W_{hh}$ với các phần tử độc lập, phương sai $g^2/64$, nên bán kính phổ xấp xỉ $g$; trạng thái đầu nhỏ, và mạng chạy 100 bước. Một gradient ngẫu nhiên được truyền ngược từ bước cuối, và thí nghiệm đo chuẩn của gradient sau 10, 50 và 100 bước lùi, chia cho chuẩn sau 1 bước và lấy trung bình trên 8 lần khởi tạo.

![Hình 10](figs/models09_rnn.png)

**Hình 10.** Chuẩn tương đối của gradient khi truyền ngược qua thời gian, theo số bước lùi; trục dọc theo thang log.

| Mô hình | Sau 10 bước | Sau 50 bước | Sau 100 bước |
|---|---|---|---|
| RNN, $g = 0{,}9$ | $3{,}2 \times 10^{-1}$ | $6{,}8 \times 10^{-2}$ | $3{,}2 \times 10^{-2}$ |
| RNN, $g = 1{,}0$ | $4{,}2 \times 10^{-1}$ | $1{,}8 \times 10^{-1}$ | $1{,}1$ |
| RNN, $g = 1{,}2$ | $6{,}4 \times 10^{-1}$ | $3{,}8 \times 10^{-1}$ | $3{,}0$ |
| Đường ô nhớ của LSTM, hệ số chặn cổng quên 1 | $5{,}0 \times 10^{-2}$ | $8{,}0 \times 10^{-8}$ | $3{,}0 \times 10^{-15}$ |
| Đường ô nhớ của LSTM, hệ số chặn cổng quên 4 | $8{,}3 \times 10^{-1}$ | $3{,}6 \times 10^{-1}$ | $1{,}3 \times 10^{-1}$ |

Với $g = 0{,}9$, gradient giảm đều. Với $g = 1{,}0$ và $1{,}2$, gradient giảm trong khoảng 50 bước lùi đầu rồi tăng trở lại, tới 3,0 sau 100 bước với $g = 1{,}2$. Lý do nằm ở trạng thái đầu nhỏ. Ở các bước đầu của chuỗi, $h$ còn nhỏ nên tanh gần như tuyến tính và Jacobi xấp xỉ $W_{hh}$, có bán kính phổ lớn hơn 1; ở các bước sau, với $g > 1$, trạng thái lớn dần, tanh bão hoà và đạo hàm của nó kéo Jacobi xuống. Như vậy độ lớn của gradient phụ thuộc cả vào $W_{hh}$ lẫn vào mức bão hoà của các đơn vị dọc theo chuỗi, và trong khi huấn luyện, cả hai đều thay đổi. Hai dòng cuối của bảng nói về LSTM và sẽ được giải thích ở Mục 8.3.

Hai loại sự cố cần cách xử lý khác nhau. Bùng nổ dễ xử lý: **cắt ngưỡng gradient** (gradient clipping) nhân gradient với $\tau/\|g\|$ mỗi khi chuẩn của nó vượt ngưỡng $\tau$, giữ nguyên hướng và chỉ giới hạn độ lớn (Pascanu, Mikolov và Bengio, 2013). Kỹ thuật này được dùng rộng rãi, kể cả khi huấn luyện Transformer. Tiêu biến thì khó xử lý hơn nhiều. Một gradient đã quá nhỏ không thể phóng to lại một cách có ý nghĩa, vì thông tin về ảnh hưởng của các bước xa đã lẫn vào nhiễu. Muốn sửa phải thay đổi kiến trúc, và đó là việc LSTM làm.

### 8.3. LSTM và GRU

Kiến trúc giải quyết được vấn đề tiêu biến là **LSTM** (long short-term memory; Hochreiter và Schmidhuber, 1997, với cổng quên do Gers, Schmidhuber và Cummins, 2000 bổ sung). LSTM thêm một **trạng thái ô nhớ** $c_t$ bên cạnh trạng thái ẩn $h_t$, và ba cổng có giá trị trong $(0, 1)$:

$$\begin{aligned} f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f), & \tilde c_t &= \tanh(W_c [h_{t-1}, x_t] + b_c), \\ i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i), & c_t &= f_t \odot c_{t-1} + i_t \odot \tilde c_t, \\ o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o), & h_t &= o_t \odot \tanh(c_t). \end{aligned}$$

Cổng quên $f_t$ quyết định giữ lại bao nhiêu phần của ô nhớ cũ, cổng vào $i_t$ quyết định ghi thêm bao nhiêu thông tin mới, cổng ra $o_t$ quyết định đưa bao nhiêu phần của ô nhớ ra trạng thái ẩn. Chẳng hạn với một ô nhớ một chiều có $c_{t-1} = 2$, nếu ở bước này $f_t = 0{,}9$, $i_t = 0{,}5$, $\tilde c_t = -1$ và $o_t = 0{,}8$ thì $c_t = 0{,}9 \cdot 2 + 0{,}5 \cdot (-1) = 1{,}3$ và $h_t = 0{,}8 \tanh 1{,}3 \approx 0{,}69$. Ô nhớ giữ lại 90% giá trị cũ, ghi thêm một nửa giá trị mới, và chỉ một phần của nó được đưa ra ngoài.

Chỗ khác biệt quyết định là ô nhớ được cập nhật bằng phép cộng có trọng số, không qua phép nhân ma trận. Dọc theo đường đi qua ô nhớ, nếu bỏ qua các đường gián tiếp qua $h$, ta có

$$\frac{\partial c_T}{\partial c_t} = \prod_{i=t+1}^{T} \operatorname{diag}(f_i).$$

Đây là tích của các số trong $(0, 1)$, không phải tích của các ma trận, và mạng học được giá trị của $f_i$: khi cần giữ thông tin lâu, nó có thể đặt $f_i$ gần 1. Ở ví dụ trên, đạo hàm của $c_t$ theo $c_{t-1}$ dọc theo ô nhớ đúng bằng $f_t = 0{,}9$, không qua hàm tanh hay ma trận trọng số nào. Cơ chế này tương tự kết nối tắt ở Mục 6.5, nhưng theo trục thời gian.

Hai dòng cuối của bảng ở Mục 8.2 mô phỏng riêng đường đi này: gradient nhân với $f_t = \sigma(b_f + \xi_t)$ ở mỗi bước, với $\xi_t$ là nhiễu nhỏ. Kết quả cho thấy lợi ích của LSTM phụ thuộc vào giá trị cổng quên. Với hệ số chặn $b_f = 1$, cổng quên trung bình khoảng $\sigma(1) = 0{,}73$, và sau 100 bước gradient chỉ còn $3 \times 10^{-15}$. Với $b_f = 4$, cổng quên khoảng $0{,}98$, và gradient còn 0,13.

Có thể hiểu các con số này qua **thang thời gian của bộ nhớ**. Với cổng quên không đổi $f$, thông tin suy giảm theo $f^t$, tức có thang thời gian khoảng $1/(1-f)$ bước: với $f = 0{,}73$ là khoảng 4 bước, với $f = 0{,}98$ là khoảng 50 bước. Khuyến nghị phổ biến là khởi tạo $b_f = 1$ (Jozefowicz, Zaremba và Sutskever, 2015), đủ cho các phụ thuộc cỡ vài bước tới vài chục bước ngay từ đầu huấn luyện. Khi biết trước chuỗi có phụ thuộc rất xa, Tallec và Ollivier (2018) đề xuất khởi tạo $b_f$ theo thang thời gian mong muốn, để các đơn vị khác nhau có bộ nhớ dài ngắn khác nhau.

Một biến thể gọn hơn là **GRU** (gated recurrent unit; Cho và cộng sự, 2014). GRU gộp cổng quên và cổng vào thành một cổng cập nhật, bỏ ô nhớ riêng và cổng ra, nên có ba khối trọng số thay vì bốn, tức ít hơn khoảng 25% tham số. Chung và cộng sự (2014) so sánh trên nhiều bài toán và không thấy bên nào tốt hơn một cách nhất quán, nên chọn giữa hai loại là việc phải quyết định bằng thực nghiệm.

### 8.4. Giới hạn về tính song song

Vấn đề gradient giảm nhẹ được bằng LSTM, cắt ngưỡng gradient và khởi tạo tốt. Giới hạn không khắc phục được nằm ở chính định nghĩa: $h_t$ phụ thuộc $h_{t-1}$, nên các bước phải tính tuần tự. Xử lý một chuỗi dài $T$ cần $T$ bước nối tiếp nhau, dù có bao nhiêu phần cứng song song. Bảng sau so sánh một lớp RNN với một lớp self-attention (Chương 9), theo Vaswani và cộng sự (2017).

| | Lớp RNN | Lớp self-attention |
|---|---|---|
| Số phép tính mỗi lớp | $O(T d^2)$ | $O(T^2 d)$ |
| Số bước phải thực hiện tuần tự | $O(T)$ | $O(1)$ |
| Độ dài đường đi dài nhất giữa hai vị trí | $O(T)$ | $O(1)$ |

Self-attention tốn nhiều phép tính hơn khi $T > d$, nhưng các phép tính đó thực hiện song song được trên GPU, còn các bước tuần tự thì không. Đường đi ngắn giữa hai vị trí bất kỳ còn có nghĩa là gradient nối hai token cách xa nhau không phải đi qua $T$ phép nhân như trong RNN. Hai điểm này khiến Transformer thay thế RNN trong phần lớn bài toán xử lý ngôn ngữ.

Hướng hồi quy chưa kết thúc. Các mô hình không gian trạng thái (state space model) như Mamba (Gu và Dao, 2023) dùng phép truy hồi tuyến tính theo thời gian, cho phép huấn luyện song song bằng thuật toán quét (scan) và suy luận với bộ nhớ không đổi theo độ dài chuỗi, và cạnh tranh được với Transformer ở một số quy mô.

### 8.5. Tóm tắt

RNN xử lý chuỗi bằng một trạng thái ẩn cập nhật qua từng bước với trọng số dùng chung, và được huấn luyện bằng lan truyền ngược trên mạng trải ra theo thời gian. Vì cùng một ma trận được nhân lặp lại, gradient tiêu biến hoặc bùng nổ theo số bước; bùng nổ xử lý được bằng cắt ngưỡng gradient, còn tiêu biến cần thay đổi kiến trúc. LSTM cập nhật ô nhớ bằng phép cộng có cổng, nên gradient dọc theo ô nhớ chỉ nhân với các giá trị cổng quên mà mạng tự học; thang thời gian của bộ nhớ khoảng $1/(1-f)$ bước, và mô phỏng cho thấy sau 100 bước gradient còn 0,13 với cổng quên 0,98 nhưng chỉ $3 \times 10^{-15}$ với cổng quên 0,73. Giới hạn cuối cùng của RNN là các bước phải tính tuần tự, và đường đi giữa hai vị trí xa nhau dài $O(T)$.

Self-attention bỏ hẳn phép truy hồi: mọi vị trí lấy thông tin trực tiếp từ mọi vị trí khác trong một bước. Chương 9 xây Transformer từ phép toán đó.

---

## 9. Attention và Transformer

Mục 8.4 kết thúc ở bảng so sánh một lớp RNN với một lớp self-attention: self-attention tốn nhiều phép tính hơn nhưng không có bước tuần tự nào, và mọi cặp vị trí chỉ cách nhau một bước. Transformer (Vaswani và cộng sự, 2017) xây toàn bộ kiến trúc quanh phép toán đó, và ngày nay là kiến trúc của gần như mọi mô hình ngôn ngữ lớn. Ta bắt đầu từ attention như một phép tra cứu, rồi lần lượt ghép các thành phần lại thành một khối Transformer hoàn chỉnh.

### 9.1. Attention như phép tra cứu mềm

Một từ điển trả về giá trị ứng với khoá khớp đúng truy vấn. Attention làm việc tương tự nhưng mềm hơn: nó so truy vấn với mọi khoá, biến các độ giống nhau thành trọng số dương có tổng bằng 1, rồi trả về trung bình có trọng số của các giá trị. Vì mọi bước đều khả vi, cả phép tra cứu có thể học bằng gradient descent.

> **Định nghĩa 9.1 (Scaled dot-product attention).** Cho ma trận truy vấn $Q \in \mathbb{R}^{T_q \times d_k}$, khoá $K \in \mathbb{R}^{T_k \times d_k}$ và giá trị $V \in \mathbb{R}^{T_k \times d_v}$,
> $$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V,$$
> trong đó softmax áp dụng cho từng hàng. Ma trận $A = \operatorname{softmax}(QK^\top/\sqrt{d_k}) \in \mathbb{R}^{T_q \times T_k}$ gọi là ma trận trọng số attention; mỗi hàng của nó không âm và có tổng bằng 1.

Ba ma trận đóng ba vai khác nhau trong phép tra cứu, như bảng dưới tóm tắt.

| Ký hiệu | Vai trò |
|---|---|
| $Q$ (query, truy vấn) | thông tin mà vị trí đang xét cần tìm |
| $K$ (key, khoá) | đặc điểm để so khớp của từng vị trí |
| $V$ (value, giá trị) | nội dung được lấy về từ từng vị trí |

Trong **self-attention**, cả ba ma trận là phép chiếu tuyến tính của cùng một chuỗi đầu vào $X \in \mathbb{R}^{T \times d}$: $Q = XW_Q$, $K = XW_K$, $V = XW_V$. Trong **cross-attention**, $Q$ lấy từ một chuỗi còn $K$ và $V$ lấy từ chuỗi khác, chẳng hạn câu đang dịch và câu nguồn. Cơ chế attention được Bahdanau, Cho và Bengio (2015) đưa vào dịch máy trước khi có Transformer, dùng kết hợp với RNN.

> **Ví dụ 9.1.** Truy vấn $q = (1, 0)$, ba khoá $k_1 = (1, 0)$, $k_2 = (0, 1)$, $k_3 = (1, 1)$ và ba giá trị $v_1 = (1, 0)$, $v_2 = (0, 1)$, $v_3 = (2, 2)$, với $d_k = 2$. Điểm số $q \cdot k_j$ là $(1, 0, 1)$; chia cho $\sqrt2$ được $(0{,}707;\; 0;\; 0{,}707)$. Softmax cho trọng số $(0{,}401;\; 0{,}198;\; 0{,}401)$: truy vấn "giống" khoá 1 và khoá 3 như nhau và giống khoá 2 ít hơn. Đầu ra là $0{,}401\,v_1 + 0{,}198\,v_2 + 0{,}401\,v_3 = (1{,}203;\; 1{,}000)$.

### 9.2. Phép chia cho $\sqrt{d_k}$

Trong Định nghĩa 9.1, điểm số được chia cho $\sqrt{d_k}$ trước softmax. Phép chia này trông như một chi tiết nhỏ, nhưng thiếu nó thì attention gần như không học được khi $d_k$ lớn.

> **Mệnh đề 9.1.** Nếu các thành phần của $q, k \in \mathbb{R}^{d_k}$ là các biến ngẫu nhiên độc lập, có trung bình 0 và phương sai 1, thì $q \cdot k$ có trung bình 0 và phương sai $d_k$.

> **Chứng minh.** $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$. Mỗi số hạng có $\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$ và $\operatorname{Var}(q_i k_i) = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1$. Các số hạng độc lập với nhau, nên phương sai của tổng bằng tổng các phương sai, tức $d_k$.

Như vậy độ lệch chuẩn của điểm số tăng theo $\sqrt{d_k}$: với $d_k = 64$, các điểm số trải trong khoảng cỡ $\pm 8$; với $d_k = 1\,024$ là $\pm 32$. Softmax của các số chênh nhau hàng chục đơn vị gần như là một vector one-hot. Đạo hàm của softmax là $\partial p_i/\partial s_j = p_i(\delta_{ij} - p_j)$, và khi $p$ gần one-hot thì mọi phần tử của ma trận này đều gần 0, nên gradient gần như không truyền qua được. Chia điểm số cho $\sqrt{d_k}$ đưa phương sai về 1 với mọi $d_k$. Vaswani và cộng sự (2017) nêu lập luận này trong chú thích 4 của bài báo.

Thí nghiệm trong `code/models/experiments.py` kiểm tra cả hai vế của lập luận: tạo 4 000 truy vấn và 64 khoá ngẫu nhiên theo phân phối chuẩn tắc, đo phương sai của điểm số và entropy của phân phối attention trên 64 khoá, có và không có phép chia.

![Hình 11](figs/models10_scaling.png)

**Hình 11.** Trái: phương sai đo được của $q \cdot k$ so với giá trị lý thuyết $d_k$ (đường chấm). Phải: entropy của phân phối attention trên 64 khoá theo $d_k$, có và không có phép chia cho $\sqrt{d_k}$.

| $d_k$ | $\operatorname{Var}(q \cdot k)$ đo được | Lý thuyết | Entropy, không chia | Entropy, có chia | Trọng số lớn nhất, không chia |
|---|---|---|---|---|---|
| 4 | 3,63 | 4 | 2,892 | 3,739 | 0,258 |
| 16 | 15,99 | 16 | 1,360 | 3,685 | 0,584 |
| 64 | 64,66 | 64 | 0,582 | 3,680 | 0,791 |
| 256 | 251,67 | 256 | 0,275 | 3,692 | 0,894 |
| 1 024 | 1 021,97 | 1 024 | 0,118 | 3,685 | 0,953 |

Phương sai đo được khớp lý thuyết ở mọi $d_k$. Không chia thì entropy giảm dần về 0: ở $d_k = 1\,024$, entropy chỉ còn 0,118 nat, trong khi giá trị lớn nhất có thể là $\ln 64 = 4{,}159$ nat khi trọng số chia đều, và trọng số lớn nhất trung bình là 0,953, tức attention gần như chỉ chọn một khoá. Có chia thì entropy giữ quanh 3,68 với mọi $d_k$, nghĩa là hành vi của attention không còn phụ thuộc vào số chiều được chọn cho mỗi đầu.

### 9.3. Attention nhiều đầu

Thay vì một phép attention trên $d$ chiều, **attention nhiều đầu** (multi-head attention) chiếu đầu vào xuống $h$ không gian con, mỗi không gian $d_k = d/h$ chiều, thực hiện $h$ phép attention song song, rồi nối kết quả và chiếu lại:

$$\begin{aligned} \operatorname{head}_i &= \operatorname{Attention}(XW_Q^{(i)},\, XW_K^{(i)},\, XW_V^{(i)}), \\ \operatorname{MultiHead}(X) &= \operatorname{Concat}(\operatorname{head}_1, \dots, \operatorname{head}_h)\, W_O. \end{aligned}$$

Bài báo gốc dùng $d = 512$, $h = 8$, $d_k = 64$. Vì mỗi đầu hẹp đi $h$ lần, tổng số tham số, $4d^2$ cho $W_Q$, $W_K$, $W_V$, $W_O$, và tổng số phép tính gần như bằng attention một đầu trên đủ $d$ chiều. Nhiều đầu không mua thêm khả năng tính toán mà cho phép mô hình cùng lúc chú ý tới những loại quan hệ khác nhau ở những vị trí khác nhau, chẳng hạn một đầu theo quan hệ cú pháp, một đầu theo từ đồng tham chiếu. Với một đầu duy nhất, phép lấy trung bình có trọng số sẽ trộn lẫn các quan hệ đó.

Không phải đầu nào cũng quan trọng như nhau. Michel, Levy và Neubig (2019) cho thấy có thể bỏ phần lớn các đầu của một mô hình đã huấn luyện mà chất lượng giảm rất ít, dù một số ít đầu là thiết yếu. Nhận xét này là cơ sở cho các cách chia sẻ khoá và giá trị giữa các đầu ở Mục 10.4.

### 9.4. Mặt nạ

Đôi khi một vị trí không được phép nhìn một số vị trí khác. Hai loại mặt nạ thường gặp có mục đích khác nhau.

| Loại | Che các vị trí nào | Mục đích |
|---|---|---|
| Mặt nạ nhân quả (causal mask) | mọi vị trí $j > i$ khi tính đầu ra tại vị trí $i$ | mô hình sinh văn bản không được nhìn thấy các token phía sau |
| Mặt nạ đệm (padding mask) | các vị trí đệm thêm để các chuỗi trong lô dài bằng nhau | phần đệm không đóng góp vào kết quả |

Mặt nạ được cài đặt bằng cách cộng $-\infty$, trong thực tế là một số âm rất lớn, vào điểm số của các vị trí bị che trước khi tính softmax, để sau softmax trọng số của chúng bằng đúng 0. Chẳng hạn với ba token có điểm số bằng nhau ở mọi cặp, không có mặt nạ thì mỗi hàng của ma trận trọng số là $(1/3;\ 1/3;\ 1/3)$. Có mặt nạ nhân quả, hàng thứ nhất chỉ còn chính nó, $(1;\ 0;\ 0)$; hàng thứ hai chia đều cho hai vị trí đầu, $(0{,}5;\ 0{,}5;\ 0)$; hàng thứ ba giữ nguyên $(1/3;\ 1/3;\ 1/3)$.

Mặt nạ nhân quả còn cho phép một điều quan trọng khi huấn luyện. Với một chuỗi $T$ token, một lượt xuôi duy nhất tính được dự đoán cho cả $T$ vị trí cùng lúc, và vị trí nào cũng chỉ thấy các token phía trước nó. Mỗi chuỗi huấn luyện vì vậy cho $T$ bài toán dự đoán token tiếp theo, tính song song. RNN cũng cho $T$ dự đoán, nhưng phải tính tuần tự.

### 9.5. Cấu trúc một khối Transformer

Attention nhiều đầu, mặt nạ, chuẩn hoá và kết nối tắt ở Chương 6 ghép lại thành một khối Transformer.

![Hình 12](figs/models11_block.png)

**Hình 12.** Một khối Transformer, vẽ theo thứ tự pre-LN (Mục 10.2): hai khối con, mỗi khối con có một kết nối tắt đi vòng qua.

Một khối Transformer theo thứ tự pre-LN tính

$$x \leftarrow x + \operatorname{MultiHead}(\operatorname{LN}(x)), \qquad x \leftarrow x + \operatorname{FFN}(\operatorname{LN}(x)),$$

trong đó $\operatorname{FFN}(x) = W_2\,\phi(W_1 x + b_1) + b_2$ là một MLP hai lớp áp dụng riêng cho từng vị trí, thường với $d_{\text{ff}} = 4d$. Một mô hình gồm nhiều khối như vậy xếp chồng, cộng với lớp embedding ở đầu và lớp chiếu ra từ vựng ở cuối. Hai khối con phân công rõ ràng, như bảng dưới cho thấy.

| | Khối con attention | Khối con FFN |
|---|---|---|
| Kết hợp thông tin theo trục nào | giữa các vị trí | giữa các chiều đặc trưng của cùng một vị trí |
| Các vị trí có được xử lý độc lập không | không: mỗi vị trí nhìn thấy các vị trí khác | có: từng vị trí xử lý riêng |
| Số tham số, với $d_{\text{ff}} = 4d$, bỏ qua hệ số chặn | $4d^2$ | $8d^2$ |
| Số phép tính theo độ dài chuỗi | $O(T^2 d + T d^2)$ | $O(T d^2)$ |

Từ bảng rút ra hai điều mà người mới thường không để ý. FFN chiếm hai phần ba số tham số của mỗi khối, dù attention là phần được nói tới nhiều hơn; Chương 12 kiểm chứng con số này trên GPT-2. Và attention là thành phần duy nhất kết hợp thông tin giữa các vị trí: bỏ attention đi, Transformer chỉ còn là một MLP áp dụng độc lập cho từng token.

### 9.6. Mã hoá vị trí

Self-attention không có khái niệm thứ tự. Nếu hoán vị các token đầu vào, khi không có mặt nạ, thì các vector đầu ra chỉ hoán vị theo, giá trị của từng vector không đổi. Hai câu "chó cắn người" và "người cắn chó" vì vậy sẽ cho cùng một tập biểu diễn. Ví dụ 9.1 cho thấy điều này ở quy mô nhỏ: đổi thứ tự ba cặp khoá–giá trị thì ba trọng số đổi chỗ theo, còn đầu ra vẫn là $(1{,}203;\ 1{,}000)$, vì trung bình có trọng số không phụ thuộc thứ tự cộng. Thông tin vị trí phải được đưa vào riêng, và có ba cách chính.

Cách đầu tiên, dùng trong bài báo gốc (Vaswani và cộng sự, 2017), là mã hoá sin–cos cố định: cộng vào embedding của token ở vị trí $i$ vector $p_i$ với $p_{i,2t} = \sin(i/10000^{2t/d})$ và $p_{i,2t+1} = \cos(i/10000^{2t/d})$. Cách này không có tham số học. Các tần số khác nhau cho phép biểu diễn cả khoảng cách gần lẫn xa, và $p_{i+k}$ là một phép biến đổi tuyến tính của $p_i$, nên mô hình có thể học quan hệ theo khoảng cách tương đối.

Cách thứ hai, dùng trong BERT và GPT-2, là embedding vị trí học được: một bảng tra kích thước $T_{\max} \times d$, học như embedding của từ. Cách này đơn giản, nhưng không dùng được cho vị trí vượt quá $T_{\max}$, tức 512 với BERT và 1 024 với GPT-2.

Cách thứ ba đưa khoảng cách giữa hai vị trí trực tiếp vào điểm số attention thay vì cộng vào embedding. RoPE (Mục 10.3) xoay truy vấn và khoá theo vị trí; ALiBi (Press, Smith và Lewis, 2022) cộng vào điểm số một đại lượng tỉ lệ với khoảng cách. Đây là hướng mà phần lớn mô hình ngôn ngữ hiện nay đi theo.

### 9.7. Encoder, decoder và encoder–decoder

Bài báo gốc của Transformer dùng kiến trúc encoder–decoder cho dịch máy. Các mô hình về sau dùng một trong ba kiểu, khác nhau ở loại attention và mục tiêu tiền huấn luyện.

| Kiểu | Attention | Mục tiêu tiền huấn luyện | Mô hình tiêu biểu | Dùng cho |
|---|---|---|---|---|
| Chỉ encoder | hai chiều: mỗi token thấy mọi token | đoán các token bị che (masked language modeling) | BERT (Devlin và cộng sự, 2019), 110 triệu tham số với bản base | phân loại, trích xuất thông tin, embedding cho tìm kiếm |
| Chỉ decoder | nhân quả: mỗi token chỉ thấy các token phía trước | đoán token tiếp theo | GPT-2 (Radford và cộng sự, 2019), các mô hình ngôn ngữ lớn hiện nay | sinh văn bản, và qua sinh văn bản là hầu hết các nhiệm vụ khác |
| Encoder–decoder | encoder hai chiều; decoder nhân quả, thêm cross-attention tới đầu ra của encoder | khôi phục các đoạn văn bị che | T5 (Raffel và cộng sự, 2020) | dịch máy, tóm tắt, các bài toán biến chuỗi thành chuỗi |

BERT che ngẫu nhiên 15% số token và học đoán lại chúng từ cả hai phía, nên biểu diễn của mỗi token chứa thông tin của toàn câu; vì vậy các mô hình embedding văn bản thường dựa trên encoder ([Chương 4 của *Biểu diễn & Căn chỉnh*](bieudien-ch04.html)). Mô hình chỉ decoder học từ mọi vị trí của mọi chuỗi (Mục 9.4), dùng một mục tiêu duy nhất, và sinh văn bản một cách tự nhiên; khi mở rộng quy mô, kiểu kiến trúc này trở thành lựa chọn của gần như mọi mô hình ngôn ngữ lớn. Các chương tiếp theo tập trung vào kiểu chỉ decoder.

### 9.8. Tóm tắt

Attention là một phép tra cứu mềm: so truy vấn với mọi khoá, biến độ giống nhau thành trọng số bằng softmax, rồi lấy trung bình có trọng số của các giá trị. Điểm số phải chia cho $\sqrt{d_k}$, vì phương sai của tích vô hướng bằng $d_k$; thí nghiệm cho thấy không chia thì entropy của attention ở $d_k = 1\,024$ chỉ còn 0,118 nat và attention gần như chỉ chọn một khoá. Nhiều đầu cho phép chú ý tới nhiều loại quan hệ cùng lúc với cùng số tham số, mặt nạ nhân quả cho phép huấn luyện song song trên mọi vị trí, và mã hoá vị trí đưa thứ tự vào một phép toán vốn không có thứ tự. Một khối Transformer gồm attention và FFN, mỗi phần có kết nối tắt và chuẩn hoá, trong đó FFN chiếm hai phần ba tham số còn attention là phần duy nhất trộn thông tin giữa các vị trí. Kiểu chỉ decoder trở thành kiến trúc của các mô hình ngôn ngữ lớn.

Khung của khối Transformer gần như giữ nguyên tới nay, nhưng các thành phần bên trong đã được thay gần hết kể từ bài báo gốc: vị trí của lớp chuẩn hoá, cách mã hoá vị trí, cách chia sẻ khoá và giá trị giữa các đầu, hàm kích hoạt của FFN, và cả cách tính attention trên phần cứng. Chương 10 xét từng thay đổi đó.

---

## 10. Các cải tiến của Transformer

Nếu đặt sơ đồ một khối của Llama 2 cạnh sơ đồ trong bài báo Transformer năm 2017, ta thấy cùng một khung: attention, FFN, kết nối tắt và chuẩn hoá. Nhìn kỹ từng thành phần thì gần như thành phần nào cũng đã đổi. Mỗi thay đổi ra đời để giải quyết một vấn đề cụ thể, có khi về độ ổn định khi huấn luyện, có khi về bộ nhớ khi suy luận, có khi về tốc độ trên phần cứng, và mỗi thay đổi cũng có điểm yếu riêng. Ta xét chúng theo thứ tự trong bảng tổng quan ở Mục 10.1.

### 10.1. Những thay đổi so với Transformer gốc

![Hình 13](figs/models14_modern.png)

**Hình 13.** Các thành phần của Transformer năm 2017 và lựa chọn phổ biến hiện nay. Khung kiến trúc giữ nguyên; các thành phần thay đổi vì những lý do cụ thể.

| Thành phần | Transformer 2017 | Phổ biến hiện nay | Lý do thay đổi | Mục |
|---|---|---|---|---|
| Vị trí lớp chuẩn hoá | sau khối con (post-LN) | trước khối con (pre-LN) | huấn luyện ổn định hơn | 10.2 |
| Loại chuẩn hoá | LayerNorm | RMSNorm | rẻ hơn, chất lượng tương đương | 6.4 |
| Thông tin vị trí | sin–cos cộng vào embedding | RoPE | quan hệ tương đối, mở rộng ngữ cảnh tốt hơn | 10.3 |
| Khoá và giá trị | mỗi đầu một cặp (MHA) | nhiều đầu chung một cặp (GQA) | KV cache nhỏ hơn khi suy luận | 10.4 |
| Khối FFN | ReLU, $d_{\text{ff}} = 4d$ | SwiGLU, $d_{\text{ff}} \approx \tfrac83 d$ | chất lượng tốt hơn ở cùng số tham số | 10.5 |

Llama 2 (Touvron và cộng sự, 2023) là một ví dụ dùng cả năm lựa chọn ở cột thứ ba, với một ngoại lệ: bản 7B và 13B dùng MHA, chỉ bản 70B dùng GQA. Hai thay đổi còn lại của chương, mixture of experts và FlashAttention, không nằm trong bảng vì chúng không thay một thành phần mà thay cách tổ chức hoặc cách tính.

### 10.2. Pre-LN và post-LN

Bài báo gốc đặt chuẩn hoá sau phép cộng của kết nối tắt, $x \leftarrow \operatorname{LN}(x + F(x))$, cách làm gọi là post-LN. Phần lớn mô hình hiện nay đặt chuẩn hoá trước khối con, $x \leftarrow x + F(\operatorname{LN}(x))$, gọi là pre-LN, và thêm một lớp chuẩn hoá sau khối cuối cùng.

Khác biệt nằm ở đường tắt. Với post-LN, tín hiệu đi từ đầu vào tới đầu ra phải qua một lớp chuẩn hoá ở mỗi khối. Với pre-LN, đường tắt đi thẳng qua mọi khối mà không qua phép biến đổi nào, đúng cấu hình "chuẩn hoá trong nhánh, đường tắt đi thẳng" mà Mục 6.3 đo được là ổn định. Xiong và cộng sự (2020) phân tích gradient lúc khởi tạo và chỉ ra rằng với post-LN, gradient của các tham số gần đầu ra rất lớn, nên cần warmup để tránh các bước cập nhật quá lớn ở đầu quá trình huấn luyện; với pre-LN, gradient có độ lớn hợp lý ở mọi lớp, và mô hình huấn luyện được với ít hoặc không cần warmup.

Pre-LN cũng có nhược điểm: độ lớn của tín hiệu trên đường tắt tăng dần theo độ sâu, nên các khối ở sâu đóng góp tương đối ít hơn. Một số nghiên cứu cho thấy post-LN, khi huấn luyện được ổn định, có thể cho chất lượng cuối cao hơn một chút, và đã có các biến thể kết hợp ưu điểm của cả hai. Trong thực hành, pre-LN vẫn là lựa chọn mặc định, vì khi huấn luyện một mô hình lớn tốn kém, độ ổn định quan trọng hơn một chút chất lượng.

### 10.3. RoPE

**RoPE** (rotary position embedding; Su và cộng sự, 2024) đưa thông tin vị trí vào attention bằng cách quay vector truy vấn và khoá một góc tỉ lệ với vị trí, thay vì cộng một vector vị trí vào embedding.

> **Định nghĩa 10.1 (RoPE).** Chia $d$ chiều của một vector thành $d/2$ cặp. Với vị trí $m$, phép biến đổi $R_m$ quay cặp thứ $i$ một góc $m\theta_i$ trong mặt phẳng của cặp đó, với $\theta_i = 10000^{-2i/d}$, $i = 0, \dots, d/2 - 1$. Truy vấn tại vị trí $m$ và khoá tại vị trí $n$ được thay bằng $R_m q$ và $R_n k$ trước khi tính điểm số attention.

> **Mệnh đề 10.1.** Với mọi $q$, $k$ và mọi vị trí $m$, $n$: $\langle R_m q,\, R_n k\rangle = \langle q,\, R_{n-m}\, k\rangle$. Như vậy điểm số attention chỉ phụ thuộc vào khoảng cách tương đối $n - m$.

> **Chứng minh.** $R_m$ là ma trận khối đường chéo gồm các ma trận quay $2 \times 2$, nên nó trực giao: $R_m^\top = R_m^{-1} = R_{-m}$. Hai phép quay cùng mặt phẳng cộng góc, nên $R_{-m} R_n = R_{n-m}$. Do đó $\langle R_m q, R_n k\rangle = q^\top R_m^\top R_n k = q^\top R_{n-m}\, k$.

Trường hợp $d = 2$ cho thấy cơ chế mà không cần ký hiệu khối: chỉ có một cặp, và $R_m$ là phép quay góc $m\theta$. Lấy $q = k = (1, 0)$ và $\theta = 0{,}5$. Tại $m = 1$ và $n = 3$, $R_1 q = (\cos 0{,}5;\ \sin 0{,}5)$ và $R_3 k = (\cos 1{,}5;\ \sin 1{,}5)$, nên tích vô hướng là $\cos 1{,}5 \cos 0{,}5 + \sin 1{,}5 \sin 0{,}5 = \cos 1 \approx 0{,}540$. Dời cả hai sang vị trí 10 và 12 thì góc chênh vẫn là $2\theta = 1$ và tích vô hướng vẫn là 0,540. Dãy tần số $\theta_i$ giống dãy tần số của mã hoá sin–cos ở Mục 9.6: cặp đầu quay nhanh, phù hợp để phân biệt khoảng cách gần; cặp cuối quay rất chậm, phân biệt được khoảng cách xa.

Thí nghiệm trong `code/models/experiments.py` kiểm tra mệnh đề với $d = 64$ và hai vector ngẫu nhiên.

![Hình 14](figs/models12_rope.png)

**Hình 14.** Trái: tích vô hướng sau khi quay chỉ phụ thuộc khoảng cách tương đối, dù vị trí tuyệt đối khác nhau hàng trăm. Phải: độ tương đồng của một vector với chính nó ở hai vị trí, theo khoảng cách giữa hai vị trí.

| $m$ | $n$ | $m - n$ | $\langle R_m q, R_n k\rangle$ |
|---|---|---|---|
| 5 | 2 | 3 | 6,851342 |
| 105 | 102 | 3 | 6,851342 |
| 500 | 497 | 3 | 6,851342 |
| 9 | 3 | 6 | 5,861634 |
| 109 | 103 | 6 | 5,861634 |

Các cặp có cùng khoảng cách cho cùng giá trị tới chữ số thứ sáu sau dấu phẩy, đúng như mệnh đề. Thí nghiệm còn đo độ tương đồng của một vector với chính nó ở hai vị trí cách nhau $\Delta$, trung bình trên 300 vector ngẫu nhiên.

| $\Delta$ | 0 | 8 | 64 | 255 |
|---|---|---|---|---|
| $\langle R_\Delta v, v\rangle / \lVert v\rVert^2$ | 1,000 | 0,708 | 0,439 | 0,319 |

Kết quả này có công thức rõ ràng: $\langle R_\Delta v, v\rangle = \sum_i (v_{2i}^2 + v_{2i+1}^2)\cos(\Delta\theta_i)$, tức một trung bình có trọng số của $\cos(\Delta \theta_i)$. Khi $\Delta$ tăng, các cặp có tần số khác nhau lệch pha nhau và trung bình giảm dần. Đây là tính chất **suy giảm theo khoảng cách** mà bài báo RoPE nêu: cùng một nội dung, ở xa thì có xu hướng nhận điểm attention thấp hơn.

RoPE có ba ưu điểm. Nó mã hoá vị trí tương đối mà không cần bảng tham số riêng. Nó chỉ tác động lên $Q$ và $K$, không thay đổi $V$, không thêm tham số và không đổi kích thước tensor. Và nó cho phép mở rộng ngữ cảnh sau khi huấn luyện: vì không có bảng vị trí với giới hạn cứng, có thể co giãn các góc quay để mô hình huấn luyện với ngữ cảnh 4 096 token dùng được cho ngữ cảnh dài hơn, kèm một bước tinh chỉnh ngắn. Hai phương pháp phổ biến là nội suy vị trí (position interpolation; Chen và cộng sự, 2023), chia chỉ số vị trí cho hệ số mở rộng, và YaRN (Peng và cộng sự, 2024), co giãn khác nhau cho các dải tần số.

### 10.4. MQA và GQA

Khi sinh văn bản, mỗi token mới cần khoá và giá trị của mọi token trước đó ở mọi lớp. Để không phải tính lại, chúng được lưu trong **KV cache**. Với attention nhiều đầu thông thường (multi-head attention, MHA), mỗi đầu có khoá và giá trị riêng, nên dung lượng KV cache tỉ lệ với số đầu:

$$\text{KV cache} = 2 \times L \times n_{\text{kv}} \times d_{\text{head}} \times T \times B \times \text{số byte mỗi số},$$

trong đó hệ số 2 cho khoá và giá trị, $n_{\text{kv}}$ là số đầu có khoá và giá trị riêng, $B$ là số chuỗi xử lý cùng lúc.

Ý tưởng giảm KV cache là cho nhiều đầu truy vấn dùng chung khoá và giá trị. Ở mức cực đoan, **MQA** (multi-query attention; Shazeer, 2019) cho mọi đầu truy vấn dùng chung một cặp khoá và giá trị, tức $n_{\text{kv}} = 1$. **GQA** (grouped-query attention; Ainslie và cộng sự, 2023) đi giữa hai thái cực: chia $h$ đầu truy vấn thành $g$ nhóm, mỗi nhóm dùng chung một cặp khoá và giá trị, nên $n_{\text{kv}} = g$. MQA là trường hợp $g = 1$, còn MHA là trường hợp $g = h$.

KV cache giảm đúng $h/g$ lần. Với một mô hình cỡ Llama 2 70B (80 lớp, $d_{\text{head}} = 128$, 64 đầu truy vấn), ngữ cảnh 4 096 token, 8 chuỗi cùng lúc, lưu ở FP16, MHA với 64 đầu khoá–giá trị cần 80 GiB, còn GQA với 8 đầu cần 10 GiB. Mỗi token của mỗi chuỗi tốn 2,5 MiB với MHA và 0,3125 MiB với GQA.

MQA giảm KV cache nhiều nhất nhưng có thể làm giảm chất lượng và làm huấn luyện kém ổn định. Ainslie và cộng sự (2023) cho thấy GQA cho chất lượng gần MHA với tốc độ suy luận gần MQA, và một mô hình MHA đã huấn luyện có thể chuyển sang GQA bằng cách huấn luyện thêm với khoảng 5% lượng tính toán ban đầu. GQA vì vậy là một quyết định thiết kế vì bộ nhớ lúc suy luận, không phải vì chất lượng. [Mục 2.2 của *Ứng dụng LLM*](ungdung-ch02.html) tính KV cache của Llama 3 8B, một mô hình dùng GQA, và hệ quả của nó lên chi phí phục vụ.

### 10.5. SwiGLU

Khối FFN gốc là $W_2\,\operatorname{ReLU}(W_1 x)$ với $d_{\text{ff}} = 4d$, gồm hai ma trận. Shazeer (2020) thử các biến thể có cổng, trong đó đầu ra của một nhánh tuyến tính được nhân từng phần tử với đầu ra của một nhánh khác:

$$\operatorname{FFN}_{\text{SwiGLU}}(x) = W_2\big(\operatorname{SiLU}(W x) \odot V x\big).$$

Biến thể dùng SiLU (Swish), gọi là SwiGLU, cho chất lượng tốt nhất trong các thí nghiệm của bài báo và được nhiều mô hình ngôn ngữ dùng.

SwiGLU có ba ma trận $W$, $V$, $W_2$ thay vì hai. Để số tham số và số phép tính bằng khối FFN gốc, số chiều ẩn phải giảm: $3\,d\,d_{\text{ff}}' = 2\,d \cdot 4d$ cho

$$d_{\text{ff}}' = \frac{2}{3} \cdot 4d = \frac{8}{3}\,d.$$

Bài báo làm đúng như vậy: mô hình gốc có $d_{\text{ff}} = 3\,072$, các biến thể có cổng dùng $d_{\text{ff}} = 2\,048$. Trong thực tế, $\tfrac83 d$ được làm tròn lên bội của một số như 128 hoặc 256 cho hiệu quả phần cứng. Với Llama 2 7B, $d = 4\,096$ nên $\tfrac83 d \approx 10\,923$, và giá trị thật $d_{\text{ff}} = 11\,008 = 43 \times 256$ chính là số đó làm tròn lên bội của 256.

### 10.6. Mixture of experts

Các thay đổi trên giữ nguyên số phép tính cho mỗi token. Mixture of experts đi theo hướng khác: tăng số tham số mà không tăng số phép tính. Trong một lớp **mixture of experts** (MoE; Shazeer và cộng sự, 2017), khối FFN được thay bằng $E$ khối FFN song song gọi là chuyên gia (expert), cùng một **bộ định tuyến** (router) chọn $k$ chuyên gia cho mỗi token, thường $k = 1$ hoặc 2. Đầu ra là tổng có trọng số của các chuyên gia được chọn, với trọng số do bộ định tuyến tính.

Như vậy MoE tách số tham số khỏi số phép tính: tổng số tham số tăng theo $E$, nhưng mỗi token chỉ đi qua $k$ chuyên gia. Chẳng hạn, Mixtral 8x7B (Jiang và cộng sự, 2024) có 8 chuyên gia mỗi lớp, mỗi token dùng 2, tổng khoảng 47 tỉ tham số nhưng mỗi token chỉ dùng khoảng 13 tỉ.

Đổi lại, MoE kéo theo ba vấn đề kỹ thuật. Vấn đề đầu tiên là cân bằng tải: bộ định tuyến có xu hướng dồn phần lớn token vào vài chuyên gia, làm các chuyên gia khác không được huấn luyện, và cách xử lý thông dụng là thêm một hàm mất mát phụ phạt sự mất cân bằng (Fedus, Zoph và Shazeer, 2022). Vấn đề thứ hai là bộ nhớ: dù mỗi token chỉ dùng một phần, mọi chuyên gia đều phải nằm sẵn trong bộ nhớ, vì token tiếp theo có thể cần bất kỳ chuyên gia nào. Vấn đề thứ ba là truyền dữ liệu: khi các chuyên gia nằm trên nhiều GPU khác nhau, mỗi lớp MoE phải gửi token tới GPU chứa chuyên gia được chọn rồi nhận kết quả về. Vì vậy MoE có lợi khi giới hạn là số phép tính; nếu giới hạn là bộ nhớ, MoE không giúp được.

### 10.7. FlashAttention

Thay đổi cuối cùng không động tới mô hình mà tới cách tính. Tính attention trực tiếp theo Định nghĩa 9.1 tạo ra ma trận điểm số $T \times T$ cho mỗi đầu và lưu nó trong bộ nhớ chính của GPU (HBM). Với $T = 4\,096$, 32 đầu và số 16 bit, riêng các ma trận này đã chiếm 1 GiB cho một chuỗi ở một lớp. Ngoài dung lượng, việc ghi và đọc lại ma trận lớn này từ HBM còn tốn thời gian hơn chính các phép tính, vì trên GPU hiện đại tốc độ tính toán vượt xa tốc độ truy cập bộ nhớ.

**FlashAttention** (Dao và cộng sự, 2022) tính chính xác cùng kết quả mà không bao giờ tạo ra toàn bộ ma trận $T \times T$. Thuật toán chia $Q$, $K$, $V$ thành các khối đủ nhỏ để nằm trong bộ nhớ nhanh trên chip (SRAM), và tính softmax theo từng khối bằng kỹ thuật softmax trực tuyến (Milakov và Gimelshein, 2018). Với mỗi hàng, thuật toán giữ giá trị lớn nhất $m$ và tổng $\ell = \sum_j e^{s_j - m}$ của các điểm số đã xử lý; khi gặp khối mới có giá trị lớn nhất $m' > m$, nó nhân tổng cũ và kết quả tích luỹ cũ với $e^{m - m'}$ rồi cộng phần của khối mới. Chẳng hạn một hàng có bốn điểm số chia thành hai khối $(1, 2)$ và $(3, 0)$. Sau khối đầu, $m = 2$ và $\ell = e^{-1} + e^{0} = 1{,}368$. Khối thứ hai có giá trị lớn nhất 3, nên tổng cũ được nhân với $e^{2 - 3}$ rồi cộng thêm $e^{0} + e^{-3}$: $\ell = 1{,}368 \cdot 0{,}368 + 1 + 0{,}050 = 1{,}553$, đúng bằng tổng $\sum_j e^{s_j - 3}$ tính trực tiếp trên cả bốn điểm số. Ở lượt ngược, các giá trị cần thiết được tính lại từ các khối thay vì đọc từ bộ nhớ.

Nhờ vậy bộ nhớ cho attention tăng tuyến tính theo $T$ thay vì theo $T^2$, và thời gian chạy giảm đáng kể vì ít truy cập HBM hơn, dù số phép tính tăng nhẹ do phải tính lại ở lượt ngược. FlashAttention không thay đổi bậc $O(T^2)$ của số phép tính; nó thay đổi cách thực hiện chúng trên phần cứng. Các thư viện huấn luyện và phục vụ mô hình ngôn ngữ hiện nay đều dùng FlashAttention hoặc các cài đặt cùng ý tưởng (Dao, 2024).

### 10.8. Tóm tắt

Khung của Transformer giữ nguyên từ năm 2017, nhưng các thành phần đã thay đổi vì những lý do cụ thể. Pre-LN cho đường tắt đi thẳng nên huấn luyện ổn định hơn post-LN. RoPE quay truy vấn và khoá theo vị trí, nên điểm số attention chỉ phụ thuộc khoảng cách tương đối, và thí nghiệm xác nhận điều này tới chữ số thứ sáu. GQA cho nhiều đầu truy vấn dùng chung khoá và giá trị, giảm KV cache của một mô hình cỡ Llama 2 70B từ 80 GiB xuống 10 GiB trong ví dụ đã tính. SwiGLU thêm cổng vào FFN và giảm $d_{\text{ff}}$ xuống $\tfrac83 d$ để giữ số tham số. MoE tăng số tham số mà không tăng số phép tính cho mỗi token, đổi lại phải cân bằng tải và giữ mọi chuyên gia trong bộ nhớ. FlashAttention tính đúng attention mà không tạo ma trận $T \times T$, nên bộ nhớ tăng tuyến tính theo độ dài chuỗi.

Đến đây ta có đủ các thành phần của một mô hình ngôn ngữ hiện đại. Sau mỗi lượt xuôi, mô hình đó chỉ cho ra một phân phối xác suất trên từ vựng; muốn có văn bản, còn phải quyết định chọn token nào từ phân phối ấy. Chương 11 bàn về quyết định này.

---

## 11. Các chiến lược giải mã

Sau mỗi lượt xuôi, mô hình ngôn ngữ cho ra một phân phối xác suất trên toàn bộ từ vựng cho token tiếp theo. Để có văn bản, cần một quy tắc chọn token từ phân phối đó, gọi là chiến lược giải mã. Quy tắc này nằm hoàn toàn ngoài mô hình, nhưng ảnh hưởng của nó lớn hơn nhiều người nghĩ: cùng một mô hình với hai chiến lược khác nhau cho ra hai hệ thống có hành vi rất khác nhau, một bên lặp lại nhạt nhẽo, một bên lan man khó kiểm soát.

### 11.1. Bài toán giải mã

Tại mỗi bước, mô hình cho vector logit $z \in \mathbb{R}^{V}$ trên toàn bộ từ vựng, và phân phối của token tiếp theo là $p = \operatorname{softmax}(z)$. **Giải mã** (decoding) là quy tắc chọn một token từ $p$, nối vào chuỗi, rồi lặp lại cho tới khi gặp token kết thúc hoặc đạt độ dài tối đa.

Để so sánh các chiến lược, ta dùng hai đại lượng mô tả một phân phối. Đại lượng thứ nhất là **entropy** $H(p) = -\sum_i p_i \ln p_i$, tính bằng nat, đo mức không chắc chắn. Giá trị $e^{H}$ gọi là perplexity của phân phối, và có thể hiểu là số lựa chọn hiệu dụng: phân phối đều trên $n$ token có $e^H = n$. Đại lượng thứ hai là số token phủ 90% xác suất, tức số token ít nhất, lấy theo thứ tự xác suất giảm dần, có tổng xác suất đạt 0,9.

### 11.2. Các chiến lược

Bảng dưới tóm tắt các chiến lược giải mã phổ biến và trường hợp dùng của mỗi chiến lược.

| Chiến lược | Quy tắc | Phù hợp với |
|---|---|---|
| Greedy | luôn chọn token có xác suất cao nhất | bài toán có một đáp án đúng: trích xuất, phân loại, trả lời ngắn |
| Beam search | giữ $b$ chuỗi có xác suất cao nhất ở mỗi bước, mở rộng song song | dịch máy, nhận dạng tiếng nói |
| Lấy mẫu với nhiệt độ | lấy mẫu từ $p_i \propto \exp(z_i / \tau)$ | văn bản mở, hội thoại |
| Top-$k$ (Fan, Lewis và Dauphin, 2018) | chỉ giữ $k$ token có xác suất cao nhất, chuẩn hoá lại rồi lấy mẫu | cắt đuôi phân phối với ngưỡng cố định |
| Top-$p$, hay nucleus (Holtzman và cộng sự, 2020) | giữ tập nhỏ nhất các token có tổng xác suất ít nhất $p$, chuẩn hoá lại rồi lấy mẫu | cắt đuôi với ngưỡng thích ứng theo phân phối |

Nhiệt độ $\tau < 1$ làm phân phối nhọn hơn, $\tau > 1$ làm phân phối phẳng hơn; $\tau \to 0$ cho greedy, $\tau = 1$ giữ nguyên phân phối của mô hình. Các biến thể mới hơn tiếp tục điều chỉnh cách cắt đuôi, chẳng hạn min-$p$ (Nguyen và cộng sự, 2025) chỉ giữ các token có xác suất ít nhất một tỉ lệ cố định của xác suất lớn nhất.

### 11.3. Ảnh hưởng của các chiến lược tới phân phối

Để thấy các chiến lược khác nhau ra sao, thí nghiệm trong `code/models/experiments.py` dựng một phân phối trên 50 000 token có đuôi dài, với logit lấy từ phân phối chuẩn có độ lệch chuẩn 2,2, rồi đo entropy, số token phủ 90% xác suất và xác suất của token đứng đầu sau mỗi phép biến đổi.

![Hình 15](figs/models13_decode.png)

**Hình 15.** Trái: xác suất theo thứ hạng của token sau mỗi phép biến đổi, trục dọc theo thang log. Phải: số token phủ 90% xác suất.

| Chiến lược | Entropy (nat) | Số lựa chọn hiệu dụng $e^H$ | Số token phủ 90% | Xác suất token đầu |
|---|---|---|---|---|
| Phân phối gốc, $\tau = 1$ | 8,223 | 3 726 | 8 362 | 0,027 |
| $\tau = 0{,}7$ | 5,886 | 360 | 1 215 | 0,128 |
| $\tau = 1{,}3$ | 9,305 | 10 993 | 16 766 | 0,008 |
| Top-$k$, $k = 40$ | 3,449 | 31 | 33 | 0,139 |
| Top-$p$, $p = 0{,}9$ | 7,669 | 2 141 | 4 141 | 0,030 |
| Top-$p$, $p = 0{,}5$ | 5,673 | 291 | 376 | 0,055 |

Dòng đầu cho thấy phân phối gốc có đuôi rất dài: cần 8 362 token để phủ 90% xác suất. Lấy mẫu trực tiếp từ phân phối này nghĩa là ở mỗi bước có một xác suất đáng kể chọn phải một token rất ít khả năng, và một token sai có thể kéo lệch cả phần văn bản sau đó.

So các dòng còn lại, ta thấy hai cơ chế khác nhau. Nhiệt độ co giãn cả phân phối, còn top-$k$ và top-$p$ cắt bỏ phần đuôi. Với $\tau = 0{,}7$, vẫn còn 1 215 token trong vùng 90%, trong khi top-$k$ với $k = 40$ chỉ còn 33. Vì hai cơ chế khác nhau, chúng thường được dùng cùng nhau: cắt đuôi bằng top-$p$, rồi điều chỉnh độ tập trung bằng nhiệt độ.

Giữa hai cách cắt đuôi, top-$p$ thích ứng theo phân phối, còn top-$k$ thì không. Khi mô hình rất chắc chắn, gần như toàn bộ xác suất nằm ở một vài token, nhưng top-$k$ với $k = 40$ vẫn giữ 40 lựa chọn; khi mô hình không chắc chắn, hàng trăm token đều hợp lý, nhưng top-$k$ cắt bớt còn 40. Top-$p$ giữ ít token ở trường hợp đầu và nhiều token ở trường hợp sau.

Số lựa chọn hiệu dụng $e^H$ còn được dùng để đo độ đa dạng của mẫu sinh ra, chẳng hạn số chế độ dữ liệu mà một mạng đối sinh phủ được ([Chương 10 của *Biểu diễn & Căn chỉnh*](bieudien-ch10.html)).

### 11.4. Beam search và văn bản mở

Greedy chọn token tốt nhất ở từng bước, nhưng chuỗi gồm các lựa chọn tốt nhất từng bước chưa chắc là chuỗi có xác suất cao nhất; đây là cùng kiểu hạn chế của thuật toán tham lam khi xây cây ở Mục 3.2. Beam search giảm hạn chế này bằng cách giữ nhiều ứng viên song song.

Với văn bản mở, beam search lại cho kết quả kém: văn bản thường lặp lại, nhạt và chung chung. Holtzman và cộng sự (2020) giải thích rằng văn bản do con người viết không có xác suất cao một cách đều đặn theo mô hình, mà thường có những lựa chọn ít khả năng. Tối đa hoá xác suất của cả chuỗi vì vậy đẩy kết quả về phía văn bản an toàn và đơn điệu. Do đó sinh văn bản mở thường dùng lấy mẫu, còn các bài toán có một đáp án đúng rõ ràng như dịch máy vẫn dùng beam search.

Hai kỹ thuật thường dùng để giảm lặp là phạt lặp, tức giảm logit của các token đã xuất hiện (Keskar và cộng sự, 2019), và cấm lặp lại một cụm $n$ token đã có. Cả hai là biện pháp điều chỉnh bên ngoài, không giải quyết nguyên nhân từ phía mô hình.

### 11.5. Giải mã trong hệ thống thực tế

Khi đưa vào hệ thống thật, giải mã có mấy điểm cần biết. Điểm đầu tiên là nhiệt độ 0 chưa chắc cho kết quả tất định. Với $\tau = 0$, quy tắc là greedy, nhưng phép cộng số thực không có tính kết hợp, và thứ tự cộng trong các kernel trên GPU có thể phụ thuộc vào kích thước lô, tức vào những yêu cầu khác đang được xử lý cùng lúc. Hai lần gọi giống hệt nhau vì vậy có thể cho kết quả khác nhau khi hai logit gần bằng nhau (He, 2025). Không nên hứa với người dùng rằng đầu ra là tất định nếu hệ thống không được thiết kế riêng cho điều đó.

Điểm thứ hai là độ trễ. Mỗi token đầu ra cần một lượt xuôi qua toàn bộ mô hình, nên chi phí tỉ lệ với số token sinh ra, và yêu cầu mô hình trả lời ngắn gọn giảm chi phí và độ trễ một cách trực tiếp. **Giải mã suy đoán** (speculative decoding; Leviathan, Kalman và Matias, 2023; Chen và cộng sự, 2023) giảm độ trễ theo một cách khác: một mô hình nhỏ đề xuất trước vài token, mô hình lớn kiểm tra tất cả trong một lượt xuôi, và một quy tắc chấp nhận–từ chối bảo đảm phân phối của đầu ra đúng bằng phân phối của mô hình lớn. Kỹ thuật này giảm độ trễ mà không thay đổi chất lượng.

Điểm cuối cùng là giải mã có thể được dùng để ép định dạng. **Giải mã có ràng buộc** đặt xác suất bằng 0 cho các token làm đầu ra vi phạm một cú pháp cho trước, chẳng hạn một lược đồ JSON, nên bảo đảm đầu ra đúng định dạng. [Mục 4.4 của *Ứng dụng LLM*](ungdung-ch04.html) trình bày cách dùng kỹ thuật này.

### 11.6. Tóm tắt

Giải mã là quy tắc chọn token từ phân phối mà mô hình cho ra, và nó quyết định hành vi của hệ thống không kém gì mô hình. Greedy và beam search hợp với bài toán có một đáp án đúng; văn bản mở dùng lấy mẫu, vì tối đa hoá xác suất của cả chuỗi cho văn bản nhạt và lặp. Trên một phân phối có đuôi dài, cần 8 362 token để phủ 90% xác suất; nhiệt độ co giãn cả phân phối, còn top-$k$ và top-$p$ cắt đuôi, với top-$p$ thích ứng theo mức chắc chắn của mô hình. Trong hệ thống thật, nhiệt độ 0 chưa chắc cho kết quả tất định, chi phí tỉ lệ với số token sinh ra, giải mã suy đoán giảm độ trễ mà không đổi phân phối, và giải mã có ràng buộc bảo đảm định dạng.

Cả giải mã lẫn các cải tiến ở Chương 10 đều quay về cùng một câu hỏi thực tế: mô hình tốn bao nhiêu bộ nhớ và bao nhiêu phép tính. Chương 12 trả lời câu hỏi đó bằng các công thức tính được trên giấy, và kiểm chứng chúng với các mô hình thật.

---

## 12. Tham số, FLOP và bộ nhớ của Transformer

Khi làm việc với mô hình lớn, nhiều câu hỏi quan trọng có thể trả lời trên giấy trong vài phút, trước khi chạy bất cứ thứ gì: mô hình có vừa bộ nhớ GPU không, huấn luyện mất bao lâu, tăng gấp đôi độ dài ngữ cảnh thì chi phí tăng bao nhiêu. Các công thức để trả lời đều dựng được từ những gì đã học ở Chương 9 và 10. Ta sẽ dựng chúng rồi kiểm chứng với các con số đã công bố của GPT-2 và Llama 2; như sẽ thấy, công thức đếm tham số khớp tới từng tham số.

### 12.1. Mục đích của việc ước lượng

Câu hỏi thường gặp nhất là mô hình có vừa hay không. Số tham số nhân số byte mỗi tham số cho dung lượng trọng số, và cần cộng thêm KV cache khi suy luận, hoặc gradient, trạng thái của bộ tối ưu và giá trị kích hoạt khi huấn luyện. Câu hỏi thứ hai là tốn bao nhiêu thời gian: số phép tính chia cho tốc độ tính thực tế của phần cứng cho thời gian huấn luyện hoặc độ trễ. Câu hỏi thứ ba là khi thay đổi một cấu hình, như tăng độ dài ngữ cảnh, dùng GQA hay lượng tử hoá trọng số xuống 4 bit, thì chi phí thay đổi thế nào. Cả ba đều bắt đầu từ việc đếm tham số.

### 12.2. Số tham số của một khối Transformer

Với $d$ là số chiều của mô hình và $d_{\text{ff}}$ là số chiều ẩn của FFN, bỏ qua hệ số chặn và tham số của các lớp chuẩn hoá, bảng dưới đếm tham số của một khối.

| Thành phần | Số tham số |
|---|---|
| $W_Q$, $W_K$, $W_V$ | $3d^2$ |
| $W_O$ | $d^2$ |
| FFN, hai ma trận $W_1$, $W_2$ | $2\,d\,d_{\text{ff}}$ |
| Tổng một khối, với $d_{\text{ff}} = 4d$ | $4d^2 + 8d^2 = 12d^2$ |

Với $L$ khối, số tham số không tính embedding là

$$N \approx 12\,L\,d^2.$$

Đây là công thức Kaplan và cộng sự (2020) dùng trong nghiên cứu về quy luật co giãn. Phần embedding thêm $Vd$ tham số cho bảng embedding từ vựng, cộng $T_{\max} d$ nếu dùng embedding vị trí học được. Nếu lớp chiếu ra từ vựng không dùng chung trọng số với bảng embedding, cộng thêm $Vd$ nữa.

### 12.3. Kiểm chứng trên GPT-2 và Llama 2

Thí nghiệm trong `code/models/experiments.py` tính đầy đủ số tham số, kể cả hệ số chặn và tham số của LayerNorm, rồi so với số tham số của các mô hình GPT-2 đã công bố.

| Mô hình | Theo công thức | Đã công bố | Chênh lệch | $12Ld^2$ |
|---|---|---|---|---|
| GPT-2 small ($L = 12$, $d = 768$) | 124 439 808 | 124 439 808 | 0 | 84 934 656 |
| GPT-2 medium ($L = 24$, $d = 1\,024$) | 354 823 168 | 354 823 168 | 0 | 301 989 888 |
| GPT-2 large ($L = 36$, $d = 1\,280$) | 774 030 080 | 774 030 080 | 0 | 707 788 800 |

Công thức khớp chính xác tới từng tham số với cả ba kích thước. Chia nhỏ GPT-2 small cho thấy tham số nằm ở đâu.

| Thành phần | Số tham số | Tỉ lệ |
|---|---|---|
| Embedding từ vựng ($50\,257 \times 768$) | 38 597 376 | 31,0% |
| Embedding vị trí ($1\,024 \times 768$) | 786 432 | 0,6% |
| Attention, 12 khối | 28 348 416 | 22,8% |
| FFN, 12 khối | 56 669 184 | 45,5% |

Bảng chia nhỏ cho thấy hai điều. Embedding chiếm 31% số tham số của GPT-2 small: với mô hình nhỏ và từ vựng lớn, phần embedding chiếm tỉ trọng lớn, nên các nghiên cứu về quy luật co giãn tách riêng số tham số không tính embedding. Còn trong các khối Transformer, FFN chiếm 67% số tham số, đúng tỉ lệ $8d^2/12d^2$ ở Mục 12.2.

Cùng cách đếm áp dụng được cho kiến trúc kiểu Llama, chỉ cần sửa bốn chỗ: FFN dạng SwiGLU có ba ma trận, nên mỗi khối có $4d^2 + 3\,d\,d_{\text{ff}}$ tham số; RMSNorm chỉ có $\gamma$; không có hệ số chặn và không có embedding vị trí, vì vị trí được mã hoá bằng RoPE; và lớp chiếu ra không dùng chung trọng số với embedding. Kết quả cho Llama 2 7B là 6 738 415 616 và cho Llama 2 13B là 13 015 864 320, khớp chính xác với số đã công bố. Bài tập 1 yêu cầu tự tính con số của bản 7B.

### 12.4. Số FLOP

Phép nhân ma trận $(m \times k)$ với $(k \times n)$ tốn $2mkn$ FLOP, vì mỗi phần tử kết quả cần $k$ phép nhân và $k$ phép cộng. Với mỗi token, mỗi tham số của một lớp tuyến tính tham gia đúng một phép nhân và một phép cộng ở lượt xuôi, nên

$$C_{\text{xuôi}} \approx 2N \quad \text{FLOP mỗi token}.$$

Lượt ngược tốn khoảng gấp đôi lượt xuôi: một lần cho gradient theo đầu vào của mỗi lớp, để truyền tiếp về phía trước, và một lần cho gradient theo trọng số. Vì vậy huấn luyện tốn khoảng $6N$ FLOP mỗi token, và toàn bộ quá trình huấn luyện trên $D$ token tốn

$$C \approx 6\,N\,D \quad \text{FLOP}.$$

Với GPT-2 small, $N = 84\,934\,656$, nên suy luận tốn khoảng 170 triệu FLOP mỗi token và huấn luyện khoảng 510 triệu FLOP mỗi token.

> **Ví dụ 12.1 (Ngân sách huấn luyện Llama 2 7B).** Llama 2 7B có $N = 6\,476\,005\,376$ tham số không tính embedding và được huấn luyện trên $D = 2 \times 10^{12}$ token, nên $C \approx 6ND = 7{,}77 \times 10^{22}$ FLOP. GPU A100 đạt tối đa 312 TFLOPS với số bf16, nhưng trong thực tế chỉ dùng được một phần; tỉ lệ này gọi là mức sử dụng phần cứng (model FLOPs utilization, MFU), thường 30% tới 50% với mô hình lớn. Với MFU 30%, 37,6% và 40%, ước lượng lần lượt là 230 627, 184 011 và 172 970 giờ-GPU. Meta công bố 184 320 giờ-GPU (Touvron và cộng sự, 2023), ứng với MFU khoảng 37,5%, nằm trong khoảng thông thường. Công thức $6ND$ vì vậy cho đúng bậc độ lớn của chi phí huấn luyện; con số chính xác phụ thuộc hiệu suất của hệ thống.

Công thức $6ND$ dẫn tới một câu hỏi thiết kế: với một ngân sách tính toán $C$ cố định, nên chia $C$ giữa kích thước mô hình $N$ và lượng dữ liệu $D$ thế nào? Hoffmann và cộng sự (2022) huấn luyện hơn 400 mô hình và kết luận rằng $N$ và $D$ nên tăng cùng tỉ lệ khi $C$ tăng, với khoảng 20 token dữ liệu cho mỗi tham số; mô hình Chinchilla 70B huấn luyện trên 1,4 nghìn tỉ token theo nguyên tắc này tốt hơn các mô hình lớn hơn nhiều nhưng ít dữ liệu hơn. Kết quả này được gọi là **quy luật co giãn** tối ưu về tính toán. Trong thực tế, nhiều mô hình được huấn luyện trên lượng dữ liệu vượt xa tỉ lệ 20 token mỗi tham số, vì mô hình nhỏ hơn rẻ hơn khi phục vụ; Llama 2 7B dùng khoảng 300 token mỗi tham số.

Công thức $2N$ chỉ tính các phép nhân với ma trận trọng số, chưa tính hai phép nhân phụ thuộc độ dài ngữ cảnh trong attention: điểm số $QK^\top$ và tích với $V$. Với mỗi token và mỗi lớp, mỗi phép tốn khoảng $2Td$ FLOP khi tính trên cả ngữ cảnh $T$ token, tổng $4LTd$ cho cả mô hình. So với $2N = 24Ld^2$, ta có

$$\frac{\text{phần attention}}{\text{phần còn lại}} \approx \frac{4LTd}{24Ld^2} = \frac{T}{6d}.$$

Với GPT-2 small ($d = 768$, $T = 1\,024$), tỉ lệ là 0,22, tức attention chiếm phần nhỏ. Với $d = 4\,096$, tỉ lệ là 0,17 ở $T = 4\,096$, nhưng 1,33 ở $T = 32\,768$ và 5,33 ở $T = 131\,072$, khi attention trở thành phần chính. Như vậy chi phí tăng theo bình phương độ dài chỉ thực sự chi phối khi $T$ vượt khoảng $6d$. Nếu cài đặt bỏ qua phần bị mặt nạ nhân quả che, như FlashAttention, chi phí attention giảm khoảng một nửa và ngưỡng thành $T \approx 12d$, là con số mà Kaplan và cộng sự (2020) dùng.

### 12.5. Bộ nhớ

Bộ nhớ gồm nhiều khoản, mỗi khoản phụ thuộc vào những đại lượng khác nhau, và nhầm lẫn giữa chúng là nguyên nhân của nhiều sự cố hết bộ nhớ. Bảng dưới tách riêng từng khoản.

| Khoản | Công thức | Phụ thuộc vào |
|---|---|---|
| Trọng số | $N \times$ số byte mỗi tham số | số tham số |
| Gradient | $N \times$ số byte | số tham số; chỉ khi huấn luyện |
| Trạng thái của bộ tối ưu | Adam: $2N \times 4$ byte, cộng bản sao FP32 của trọng số khi huấn luyện với độ chính xác hỗn hợp | số tham số; chỉ khi huấn luyện |
| Giá trị kích hoạt | tỉ lệ với $L \times B \times T \times d$, cộng một phần tỉ lệ với $T^2$ nếu lưu ma trận attention | kích thước lô, độ dài chuỗi; chỉ khi huấn luyện |
| KV cache | $2 \times L \times n_{\text{kv}} \times d_{\text{head}} \times T \times B \times$ số byte | độ dài ngữ cảnh, số chuỗi; chỉ khi suy luận |

Khi huấn luyện với độ chính xác hỗn hợp và Adam, mỗi tham số cần khoảng 16 byte (Rajbhandari và cộng sự, 2020): 2 byte cho trọng số 16 bit, 2 byte cho gradient 16 bit, 4 byte cho bản sao trọng số FP32 mà bộ tối ưu cập nhật, và 8 byte cho hai trạng thái $m$, $v$ của Adam ở FP32. Với mô hình 7 tỉ tham số, con số là 112 GB, chưa tính giá trị kích hoạt, nên không vừa một GPU 80 GB. Các kỹ thuật như ZeRO (Rajbhandari và cộng sự, 2020) chia gradient và trạng thái bộ tối ưu cho nhiều GPU để giải quyết vấn đề này.

Giá trị kích hoạt có thể còn lớn hơn thế. Korthikanti và cộng sự (2022) ước lượng mỗi khối Transformer huấn luyện với số 16 bit cần khoảng $sbd\,(34 + 5as/d)$ byte cho giá trị kích hoạt, với $s$ là độ dài chuỗi, $b$ là kích thước lô và $a$ là số đầu attention. Với cấu hình của Llama 2 7B ($d = 4\,096$, $a = 32$, 32 khối), một chuỗi 4 096 token cần khoảng 97 GiB, trong đó 80 GiB là số hạng $5as/d$ đến từ các ma trận attention $T \times T$. FlashAttention (Mục 10.7) không lưu các ma trận này, đưa con số xuống khoảng 17 GiB, và gradient checkpointing (Mục 5.3) giảm tiếp bằng cách tính lại.

Khi suy luận, trọng số và KV cache là hai khoản chính. Với Llama 2 7B dùng MHA (32 lớp, 32 đầu, $d_{\text{head}} = 128$) ở FP16, KV cache tốn $2 \cdot 32 \cdot 32 \cdot 128 \cdot 2 = 524\,288$ byte, tức 0,5 MiB mỗi token; một chuỗi 4 096 token cần 2 GiB, bằng khoảng 16% dung lượng trọng số 13,5 GB. Với nhiều chuỗi và ngữ cảnh dài, KV cache có thể vượt dung lượng trọng số. [Mục 3.3 của *Ứng dụng LLM*](ungdung-ch03.html) dùng các con số này để ước lượng bộ nhớ khi tự triển khai mô hình.

Giảm số byte mỗi số là cách tác động trực tiếp lên mọi dòng của bảng trên, và là chủ đề của giáo trình [*Quantization*](ch01.html).

### 12.6. Tóm tắt

Các công thức của chương được gom trong bảng dưới.

| Đại lượng | Công thức | Dùng để |
|---|---|---|
| Số tham số không tính embedding | $12 L d^2$ (FFN hai ma trận, $d_{\text{ff}} = 4d$) | ước lượng kích thước mô hình |
| FLOP suy luận mỗi token | $2N$, cộng $4LTd$ cho attention | ước lượng độ trễ |
| FLOP huấn luyện | $6ND$ | ước lượng ngân sách tính toán |
| Ngưỡng attention chi phối chi phí | $T \approx 6d$ (tính đủ ma trận), $12d$ (bỏ phần bị che) | quyết định có cần attention hiệu quả hơn không |
| KV cache mỗi token | $2 L\, n_{\text{kv}}\, d_{\text{head}} \times$ số byte | ước lượng bộ nhớ khi suy luận |
| Bộ nhớ huấn luyện với Adam | khoảng 16 byte mỗi tham số, cộng giá trị kích hoạt | ước lượng bộ nhớ khi huấn luyện |

Công thức đếm tham số khớp tới từng tham số với GPT-2 và Llama 2, còn công thức $6ND$ cho đúng bậc độ lớn của 184 320 giờ-GPU mà Meta công bố cho Llama 2 7B khi mức sử dụng phần cứng khoảng 37,5%. Attention chỉ chi phối chi phí tính toán khi độ dài ngữ cảnh vượt khoảng $6d$. Về bộ nhớ, huấn luyện tốn khoảng 16 byte mỗi tham số cộng giá trị kích hoạt, còn suy luận tốn trọng số cộng KV cache, và mỗi khoản phụ thuộc vào những đại lượng khác nhau.

Chương này khép lại phần kiến trúc của giáo trình. Đi từ các họ mô hình qua cây, mạng nhiều lớp, mạng tích chập, mạng hồi quy tới Transformer, ta đã thấy cùng một câu hỏi lặp lại dưới nhiều dạng: giả định nào của mô hình khớp với cấu trúc của dữ liệu, và làm sao để gradient đi được qua một mạng sâu. Hai giáo trình tiếp theo dùng trực tiếp các kết quả ở đây: *Biểu diễn & Căn chỉnh* xét những gì một mô hình như vậy học được và cách căn chỉnh nó theo ý người dùng, còn *Quantization* giảm số byte mỗi số trong mọi dòng của bảng bộ nhớ ở Mục 12.5. Chương 13 gồm các bài tập để luyện các phép tính của giáo trình, và Chương 14 gom các câu hỏi phỏng vấn thường gặp.

---

## 13. Bài tập

Các bài tập trải đều trên các chương của giáo trình. Phần lớn là phép tính tay với số liệu của các mô hình thật như Llama 2, vài bài là chứng minh ngắn, một bài là bài thiết kế và một bài là thí nghiệm sửa mã. Lời giải chi tiết, kèm nhãn chương và độ khó của từng bài, nằm ở trang Lời giải.

**Bài 1 (tính tay).** Tính số tham số của Llama 2 7B từ cấu hình: 32 khối, $d = 4\,096$, $d_{\text{ff}} = 11\,008$, từ vựng 32 000 token, 32 đầu attention và 32 đầu khoá–giá trị, RoPE (không có embedding vị trí), RMSNorm (chỉ có $\gamma$), không có hệ số chặn, FFN dạng SwiGLU, lớp chiếu ra **không** dùng chung trọng số với embedding.
(a) Tính riêng số tham số của attention, FFN, các lớp chuẩn hoá, embedding và lớp chiếu ra.
(b) So tổng với con số đã công bố 6 738 415 616.
(c) Tỉ số $d_{\text{ff}}/d$ bằng bao nhiêu? Vì sao nó không bằng 4?
*Gợi ý: SwiGLU có ba ma trận. So $d_{\text{ff}}$ với $\tfrac83 d$ (Mục 10.5).*

**Bài 2 (tính tay).** Dùng kết quả Bài 1.
(a) Tính số FLOP để huấn luyện Llama 2 7B trên $2 \times 10^{12}$ token theo công thức $C \approx 6ND$.
(b) Trên GPU A100 (312 TFLOPS với bf16), ước lượng số giờ-GPU khi mức sử dụng phần cứng (MFU) là 30% và 40%.
(c) Meta công bố 184 320 giờ-GPU. Từ con số này, suy ra MFU thực tế. Giá trị đó có hợp lý không?
(d) Vì sao $N$ trong công thức không tính tham số của embedding?

**Bài 3 (suy luận).** (a) Chứng minh: nếu các thành phần của $q, k \in \mathbb{R}^{d_k}$ độc lập, có trung bình 0 và phương sai 1 thì $\operatorname{Var}(q \cdot k) = d_k$.
(b) Dùng đạo hàm của softmax để giải thích vì sao điểm số có phương sai lớn làm gradient qua attention gần như bằng 0.
(c) Nếu chia điểm số cho $d_k$ thay vì $\sqrt{d_k}$ thì điều gì xảy ra khi $d_k$ lớn?

**Bài 4 (tính tay).** Một CNN chỉ gồm các lớp tích chập $3 \times 3$.
(a) Với bước nhảy 1 ở mọi lớp, cần bao nhiêu lớp để trường tiếp nhận phủ hết ảnh rộng 224 điểm ảnh?
(b) Nếu các lớp thứ 2, 4, 6, … có bước nhảy 2 và các lớp còn lại có bước nhảy 1, tính trường tiếp nhận sau 10 lớp. Cần bao nhiêu lớp để phủ 224 điểm ảnh?
(c) Vì sao trường tiếp nhận hiệu dụng nhỏ hơn con số tính ở trên?
*Gợi ý: $r_i = r_{i-1} + (k_i - 1)\prod_{j<i} s_j$ (Mục 7.2).*

**Bài 5 (chẩn đoán).** Với mỗi tình huống, cho biết vấn đề chủ yếu là độ chệch hay phương sai, hay là dấu hiệu của lỗi, và nêu hai việc nên làm:
(a) Sai số huấn luyện 2%, sai số xác thực 18%, khoảng cách không thu hẹp khi thêm dữ liệu vào tập huấn luyện nhưng vẫn giảm dần.
(b) Sai số huấn luyện 24%, sai số xác thực 25%, cả hai đã nằm ngang.
(c) Sai số xác thực 12%, sai số huấn luyện 15%.

**Bài 6 (suy luận).** Cho hàm lều $g(x) = 2\operatorname{ReLU}(x) - 4\operatorname{ReLU}(x - \tfrac12)$.
(a) Chứng minh $g$ ánh xạ $[0, 1]$ lên $[0, 1]$.
(b) Chứng minh bằng quy nạp rằng $g$ hợp với chính nó $k$ lần là một hàm tuyến tính từng khúc có $2^k$ đoạn.
(c) Chứng minh một mạng một lớp ẩn với đầu vào một chiều cần ít nhất $2^k - 1$ đơn vị ReLU để biểu diễn chính xác hàm đó, và so với $2k$ đơn vị của mạng sâu.

**Bài 7 (suy luận).** (a) Bỏ qua hàm tanh, gradient của một RNN tuyến tính sau 100 bước lùi có độ lớn cỡ $\rho^{100}$ với $\rho$ là bán kính phổ của $W_{hh}$. Tính giá trị này khi $\rho = 0{,}9$ và $\rho = 1{,}1$. So với số đo ở Mục 8.2 và giải thích vì sao chúng khác nhau.
(b) Dọc theo ô nhớ của LSTM, gradient nhân với cổng quên $f$ ở mỗi bước. Tính $f^{100}$ khi hệ số chặn của cổng quên là 1 và là 4 (bỏ qua phần phụ thuộc đầu vào), và thang thời gian $1/(1 - f)$ tương ứng.
(c) Vì sao gradient bùng nổ dễ xử lý hơn gradient tiêu biến?

**Bài 8 (tính tay).** Một mô hình 70 tỉ tham số có 80 lớp, $d_{\text{head}} = 128$ và 64 đầu truy vấn. Tính dung lượng KV cache ở FP16 cho ngữ cảnh 4 096 token và 8 chuỗi cùng lúc, trong hai trường hợp:
(a) MHA với 64 đầu khoá–giá trị;
(b) GQA với 8 đầu khoá–giá trị.
(c) So với dung lượng trọng số ở FP16. Vì sao GQA là quyết định về bộ nhớ khi suy luận chứ không phải về chất lượng?

**Bài 9 (thiết kế).** Với mỗi bài toán, chọn họ mô hình và giải thích bằng thiên kiến quy nạp của mô hình và cấu trúc của dữ liệu, không dựa vào mức độ phổ biến của mô hình:
(a) Dự đoán khách hàng rời bỏ dịch vụ từ 40 đặc trưng dạng bảng, 30 000 khách hàng, kết quả cần giải thích được cho bộ phận kinh doanh.
(b) Phân loại ảnh sản phẩm lỗi trên dây chuyền, 3 000 ảnh có nhãn, các ảnh có bố cục rất giống nhau.
(c) Trích xuất các trường thông tin từ hợp đồng dài 50 trang.
(d) Dự báo phụ tải điện theo giờ cho 12 tháng tới.

**Bài 10 (thí nghiệm).** Trong phần (E) của `code/models/experiments.py`, thay softmax bằng một phép chuẩn hoá không có hàm mũ, ví dụ $w_i = \operatorname{ReLU}(s_i)/\sum_j \operatorname{ReLU}(s_j)$, rồi đo lại entropy theo $d_k$ khi **không** chia cho $\sqrt{d_k}$.
(a) Hiện tượng bão hoà còn xuất hiện không? Giải thích.
(b) Từ đó cho biết phép chia $\sqrt{d_k}$ xử lý vấn đề của softmax hay của tích vô hướng.
(c) Nêu lý do attention vẫn dùng softmax.

**Bài 11 (tính tay).** Phát hiện đối tượng (Mục 7.5).
(a) Tính IoU của hai hộp $A = (10, 10, 50, 40)$ và $B = (30, 20, 70, 60)$, toạ độ dạng $(x_1, y_1, x_2, y_2)$.
(b) Chạy NMS với ngưỡng IoU 0,5 trên bốn hộp: $P_1 = (0, 0, 10, 10)$ điểm 0,9; $P_2 = (1, 1, 11, 11)$ điểm 0,8; $P_3 = (5, 5, 15, 15)$ điểm 0,7; $P_4 = (20, 20, 30, 30)$ điểm 0,6. Những hộp nào được giữ?
(c) Một lớp có 4 hộp thật. Bộ phát hiện trả về 6 hộp; xếp theo điểm giảm dần, kết quả ghép ở ngưỡng IoU 0,5 lần lượt là: đúng, đúng, sai, đúng, sai, sai. Tính precision và recall sau mỗi hộp, rồi tính AP theo cách tính mọi điểm của PASCAL VOC.
(d) Vì sao một hộp dự đoán trùng với hộp đúng của cùng đối tượng bị tính là dương tính giả, và điều đó liên quan gì tới NMS?

---

## 14. Câu hỏi phỏng vấn

### 14.1. Cách trình bày câu trả lời

Phần lớn câu hỏi phỏng vấn về mô hình và kiến trúc có dạng "vì sao X". Một câu trả lời tốt thường đi qua ba bước. Đầu tiên là vấn đề mà X giải quyết: không có X thì điều gì hỏng. Tiếp theo là cơ chế: X giải quyết vấn đề đó bằng cách nào, tốt nhất bằng một công thức ngắn hoặc một con số. Cuối cùng là giới hạn: X tốn kém ở đâu, và khi nào nó không còn tác dụng. Bước cuối thường là chỗ phân biệt người hiểu với người chỉ nhớ.

Các câu trả lời mẫu dưới đây dẫn số liệu từ các thí nghiệm của giáo trình. Khi phỏng vấn không cần nhớ chính xác từng con số, nhưng cần nắm độ lớn và chiều của hiệu ứng.

### 14.2. Học máy cổ điển và mạng nơ-ron

**Câu hỏi: Giải thích phân rã độ chệch – phương sai.**

> **Trả lời.** Sai số bình phương kỳ vọng tại một điểm tách thành độ chệch², phương sai và nhiễu, với kỳ vọng lấy trên các tập huấn luyện có thể rút ra. Độ chệch là sai lệch có hệ thống do họ hàm quá hẹp; phương sai là mức dự đoán thay đổi khi đổi tập huấn luyện; nhiễu không giảm được. Giá trị thực tế của phân rã là phân biệt hai tình huống cần xử lý ngược nhau. Ở Mục 2.3, đa thức bậc 1 có độ chệch² chiếm 74% sai số, nên thêm dữ liệu không giúp gì; đa thức bậc 12 có phương sai gấp khoảng 1 900 lần độ chệch², nên thêm dữ liệu là cách sửa đúng.

**Câu hỏi: Vì sao mạng nơ-ron cần hàm kích hoạt phi tuyến?**

> **Trả lời.** Vì hợp của các phép biến đổi affine vẫn là một phép biến đổi affine: $W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2)$. Không có hàm phi tuyến thì mạng bao nhiêu lớp cũng tương đương một lớp tuyến tính. Ví dụ đơn giản nhất là hàm XOR: không mô hình tuyến tính nào biểu diễn được, nhưng hai đơn vị ReLU ở lớp ẩn là đủ.

**Câu hỏi: Mạng sâu hay mạng rộng?**

> **Trả lời.** Định lý xấp xỉ phổ quát nói mạng một lớp ẩn đủ rộng biểu diễn được mọi hàm liên tục, nhưng chỉ khẳng định sự tồn tại, không cho biết cần bao nhiêu đơn vị. Có những hàm mà mạng sâu biểu diễn với ít tham số hơn theo cấp số mũ. Ở Mục 4.4, hàm răng cưa $2^k$ đoạn biểu diễn được bằng mạng $k$ lớp với $6k$ tham số, trong khi mạng một lớp ẩn cần ít nhất $2^k - 1$ đơn vị; với $k = 7$ là 42 so với 386 tham số. Đổi lại, mạng sâu khó tối ưu hơn, cần khởi tạo đúng, chuẩn hoá và kết nối tắt.

**Câu hỏi: Rừng ngẫu nhiên hay gradient boosting?**

> **Trả lời.** Hai phương pháp tác động vào hai thành phần khác nhau của sai số. Rừng ngẫu nhiên lấy trung bình các cây sâu huấn luyện độc lập, nên giảm phương sai; thêm cây không gây overfitting, dễ chỉnh và dễ song song hoá. Gradient boosting cộng dần các cây nông, mỗi cây sửa phần sai của các cây trước, nên giảm độ chệch; thường chính xác hơn khi chỉnh tốt, nhưng nhạy với tốc độ học và số cây, và có thể overfitting nếu không dừng sớm. Ở Mục 3.4, bagging giảm phương sai của một cây sâu 8 đi một nửa, còn boosting 100 cây sâu 2 cho tổng sai số thấp nhất.

**Câu hỏi: Vì sao cây quyết định không cần chuẩn hoá đặc trưng?**

> **Trả lời.** Vì mỗi phép chia có dạng $x_j \le t$ chỉ phụ thuộc thứ tự các giá trị của đặc trưng. Mọi phép biến đổi đơn điệu, như chuẩn hoá hay lấy logarit, giữ nguyên thứ tự, nên không đổi tập các phép chia có thể có và không đổi cây. Mạng nơ-ron, hồi quy logistic, SVM và k-NN thì nhạy với thang đo.

### 14.3. Huấn luyện mạng sâu

**Câu hỏi: Vì sao có gradient tiêu biến, và xử lý thế nào?**

> **Trả lời.** Gradient qua $L$ lớp là tích của $L$ ma trận Jacobi; nếu mỗi lớp nhân độ lớn với một hệ số lệch khỏi 1, tích co về 0 hoặc tăng vọt theo hàm mũ. Ở Mục 6.3, cùng một mạng 40 lớp, chỉ đổi hệ số khởi tạo từ 0,5 sang 2,0 làm gradient ở lớp đầu thay đổi từ $10^{-18}$ tới $10^{6}$. Ba cách xử lý bổ sung cho nhau: khởi tạo đúng phương sai (He cho ReLU), chuẩn hoá, và kết nối tắt. Kết nối tắt một mình làm gradient bùng nổ ($2{,}4 \times 10^8$ sau 40 lớp) vì mỗi lớp cộng thêm vào tín hiệu một lượng cùng cỡ; phải kết hợp với chuẩn hoá trong nhánh (tỉ lệ 2,22), đúng như cấu hình pre-LN của Transformer.

**Câu hỏi: Hệ số 2 trong khởi tạo He đến từ đâu?**

> **Trả lời.** Từ ReLU. Với $z$ đối xứng quanh 0, $\mathbb{E}[\operatorname{ReLU}(z)^2] = \tfrac12 \operatorname{Var}(z)$ vì ReLU đặt một nửa phân phối bằng 0. Muốn phương sai giữ nguyên qua lớp thì $n_{\text{in}}\sigma_W^2 \cdot \tfrac12 = 1$, tức $\sigma_W^2 = 2/n_{\text{in}}$. Với tanh, gần tuyến tính quanh 0, không có hệ số này và dùng khởi tạo Xavier.

**Câu hỏi: BatchNorm khác LayerNorm thế nào? Vì sao Transformer dùng LayerNorm?**

> **Trả lời.** BatchNorm tính trung bình và phương sai trên các mẫu trong lô, riêng cho từng đặc trưng; LayerNorm tính trên các đặc trưng của một mẫu. Transformer dùng LayerNorm vì ba lý do: các chuỗi trong lô có độ dài khác nhau và có phần đệm; BatchNorm dùng thống kê của lô khi huấn luyện nhưng trung bình trượt khi suy luận, gây sai lệch khi sinh từng token; và BatchNorm làm dự đoán của một mẫu phụ thuộc các mẫu khác trong lô. RMSNorm bỏ bước trừ trung bình, rẻ hơn và chất lượng tương đương, nên được nhiều mô hình ngôn ngữ dùng.

**Câu hỏi: Adam khác SGD thế nào? Hiệu chỉnh độ chệch trong Adam để làm gì?**

> **Trả lời.** Adam giữ trung bình trượt của gradient ($m$) và của bình phương gradient ($v$), chia $m$ cho $\sqrt v$, nên mỗi tham số có bước đi riêng và ít nhạy với tốc độ học hơn SGD. Vì $m_0 = 0$, ta có $\mathbb{E}[m_t] = (1 - \beta_1^t)\,\mathbb{E}[g]$, tức $m_t$ là ước lượng chệch về 0 ở các bước đầu; chia cho $1 - \beta_1^t$ cho ước lượng không chệch, và tương tự với $v$. Đổi lại, Adam cần hai trạng thái cho mỗi tham số: với mô hình 7 tỉ tham số là 56 GB ở FP32. AdamW tách weight decay khỏi gradient, vì với Adam thường, thành phần phạt bị chia cho $\sqrt v$ và tham số có gradient lớn lại bị phạt ít hơn.

**Câu hỏi: Kết nối tắt giải quyết vấn đề gì?**

> **Trả lời.** Vấn đề suy thoái: trên CIFAR-10, mạng thường 56 lớp có sai số huấn luyện cao hơn mạng 20 lớp. Đây là vấn đề tối ưu, không phải overfitting, vì mạng sâu hơn biểu diễn được mọi thứ mạng nông biểu diễn được. Kết nối tắt $h = x + F(x)$ làm ánh xạ đồng nhất dễ biểu diễn và cho gradient một đường đi không qua ma trận trọng số nào, vì Jacobi là $I + \partial F/\partial x$. Kết nối tắt cần đi kèm chuẩn hoá mới ổn định.

**Câu hỏi: Dropout hoạt động thế nào, và vì sao ít dùng khi tiền huấn luyện mô hình ngôn ngữ lớn?**

> **Trả lời.** Khi huấn luyện, mỗi đơn vị bị tắt với xác suất $p$ và các đơn vị còn lại được chia cho $1 - p$; khi suy luận dùng mọi đơn vị. Mạng đầy đủ xấp xỉ trung bình của rất nhiều mạng con, nên dropout giảm phương sai tương tự bagging. Khi tiền huấn luyện trên kho dữ liệu cực lớn, mỗi mẫu thường chỉ được thấy một lần nên overfitting không phải vấn đề chính, và một số mô hình như PaLM không dùng dropout. Dropout vẫn hữu ích khi tinh chỉnh trên tập nhỏ.

### 14.4. Transformer

**Câu hỏi: Giải thích attention.**

> **Trả lời.** Attention là phép tra cứu mềm, khả vi: mỗi vị trí tạo một truy vấn, so với khoá của mọi vị trí bằng tích vô hướng, chuẩn hoá các điểm số bằng softmax, rồi lấy trung bình có trọng số của các giá trị: $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$. Trong một khối Transformer, attention là thành phần duy nhất kết hợp thông tin giữa các vị trí; FFN xử lý từng vị trí độc lập.

**Câu hỏi: Vì sao chia điểm số cho $\sqrt{d_k}$?**

> **Trả lời.** Nếu các thành phần của $q$ và $k$ độc lập, trung bình 0, phương sai 1 thì $q \cdot k$ có phương sai $d_k$. Điểm số lớn đẩy softmax về gần one-hot, nơi đạo hàm $p_i(\delta_{ij} - p_j)$ gần bằng 0. Chia cho $\sqrt{d_k}$ đưa phương sai về 1 với mọi $d_k$. Ở Mục 9.2, với $d_k = 1\,024$ và 64 khoá, không chia thì entropy của attention chỉ còn 0,118 nat so với tối đa 4,159, trọng số lớn nhất trung bình 0,953; có chia thì entropy giữ quanh 3,68 với mọi $d_k$.

**Câu hỏi: Attention nhiều đầu mang lại gì?**

> **Trả lời.** Không mang lại thêm khả năng tính toán: mỗi đầu hẹp đi $h$ lần nên tổng số tham số và số phép tính gần như bằng attention một đầu ở đủ số chiều. Nó cho phép mô hình cùng lúc theo dõi nhiều loại quan hệ khác nhau ở những vị trí khác nhau, điều mà một phép lấy trung bình duy nhất sẽ trộn lẫn. Các đầu có mức độ dư thừa nhất định: nhiều đầu có thể bỏ đi sau khi huấn luyện mà chất lượng giảm ít, và đó là cơ sở của MQA, GQA.

**Câu hỏi: Vì sao Transformer thay thế RNN?**

> **Trả lời.** Không phải vì ít phép tính hơn: một lớp self-attention tốn $O(T^2 d)$, một lớp RNN tốn $O(Td^2)$. Lý do là số bước tuần tự: RNN cần $O(T)$ bước không song song hoá được, self-attention cần $O(1)$. Thêm vào đó, đường đi giữa hai vị trí bất kỳ trong self-attention dài $O(1)$ thay vì $O(T)$, nên gradient nối hai token cách xa không phải qua hàng trăm phép nhân. Khi huấn luyện với mặt nạ nhân quả, một lượt xuôi tính được dự đoán cho mọi vị trí của chuỗi cùng lúc.

**Câu hỏi: Vì sao cần mã hoá vị trí, và RoPE hoạt động thế nào?**

> **Trả lời.** Self-attention không có khái niệm thứ tự: hoán vị các token đầu vào thì đầu ra chỉ hoán vị theo, nên "chó cắn người" và "người cắn chó" cho cùng một tập biểu diễn. RoPE quay từng cặp chiều của truy vấn và khoá một góc tỉ lệ với vị trí, với tần số $\theta_i = 10000^{-2i/d}$. Vì phép quay trực giao, $\langle R_m q, R_n k\rangle = \langle q, R_{n-m}k\rangle$: điểm số chỉ phụ thuộc khoảng cách tương đối. Ở Mục 10.3, các cặp vị trí $(5, 2)$, $(105, 102)$ và $(500, 497)$ cho cùng giá trị 6,851342. RoPE không thêm tham số, chỉ tác động lên $Q$ và $K$, và cho phép mở rộng ngữ cảnh sau huấn luyện bằng cách co giãn góc quay.

**Câu hỏi: GQA là gì, và vì sao được dùng?**

> **Trả lời.** Nhiều đầu truy vấn dùng chung một cặp khoá và giá trị, nên KV cache giảm đúng theo tỉ số số đầu truy vấn trên số đầu khoá–giá trị. Với mô hình cỡ 70B, ngữ cảnh 4 096 token, 8 chuỗi, FP16: MHA với 64 đầu cần 80 GiB KV cache, GQA với 8 đầu cần 10 GiB. GQA là quyết định vì bộ nhớ và băng thông khi suy luận; chất lượng gần MHA nhưng không tốt hơn.

**Câu hỏi: Vì sao $d_{\text{ff}}$ của nhiều mô hình mới không bằng $4d$?**

> **Trả lời.** Vì FFN dạng SwiGLU có ba ma trận thay vì hai. Để giữ số tham số và số phép tính, số chiều ẩn giảm còn $\tfrac23 \cdot 4d = \tfrac83 d$. Với Llama 2 7B, $\tfrac83 \cdot 4\,096 \approx 10\,923$, làm tròn lên bội của 256 được 11 008, đúng giá trị trong cấu hình.

**Câu hỏi: Encoder, decoder và encoder–decoder khác nhau thế nào?**

> **Trả lời.** Encoder dùng attention hai chiều, tiền huấn luyện bằng cách đoán token bị che, cho biểu diễn tốt cho phân loại và tìm kiếm (BERT). Decoder dùng mặt nạ nhân quả, tiền huấn luyện bằng cách đoán token tiếp theo, dùng để sinh văn bản (GPT và gần như mọi mô hình ngôn ngữ lớn hiện nay). Encoder–decoder đọc đầu vào hai chiều rồi sinh đầu ra có cross-attention tới đầu vào, phù hợp với dịch máy và tóm tắt (T5).

**Câu hỏi: FlashAttention làm gì?**

> **Trả lời.** Tính chính xác attention mà không lưu ma trận $T \times T$ trong bộ nhớ chính của GPU: chia $Q$, $K$, $V$ thành khối nằm vừa bộ nhớ trên chip, tính softmax theo từng khối bằng softmax trực tuyến, và tính lại ở lượt ngược. Bộ nhớ cho attention tăng tuyến tính theo $T$ thay vì bậc hai, và thời gian giảm vì ít truy cập bộ nhớ. Số phép tính vẫn là $O(T^2)$.

### 14.5. Tính toán tài nguyên

**Câu hỏi: Một Transformer $L$ lớp, số chiều $d$ có bao nhiêu tham số?**

> **Trả lời.** Khoảng $12Ld^2$ không tính embedding: $4d^2$ cho $W_Q$, $W_K$, $W_V$, $W_O$ và $8d^2$ cho FFN với $d_{\text{ff}} = 4d$. Cộng $Vd$ cho embedding từ vựng. FFN chiếm hai phần ba tham số của mỗi khối. Với mô hình nhỏ, embedding chiếm tỉ trọng lớn: 31% ở GPT-2 small. Mục 12.3 kiểm chứng công thức khớp chính xác với GPT-2 và Llama 2.

**Câu hỏi: Huấn luyện một mô hình tốn bao nhiêu phép tính?**

> **Trả lời.** $C \approx 6ND$: mỗi token tốn khoảng $2N$ FLOP ở lượt xuôi và $4N$ ở lượt ngược, với $N$ là số tham số không tính embedding. Với Llama 2 7B, $N \approx 6{,}48$ tỉ và $D = 2 \times 10^{12}$ token, $C \approx 7{,}8 \times 10^{22}$ FLOP. Con số 184 320 giờ-GPU A100 mà Meta công bố ứng với mức sử dụng phần cứng khoảng 37,5%, nằm trong khoảng thông thường 30–50%.

**Câu hỏi: Khi nào chi phí bậc hai của attention trở thành vấn đề?**

> **Trả lời.** Tỉ số giữa phần tính toán của attention và phần còn lại xấp xỉ $T/(6d)$ khi tính đủ ma trận điểm số. Với $d = 4\,096$, attention chỉ chiếm phần chính khi $T$ vượt khoảng 24 000 token; ở $T = 4\,096$ nó mới bằng 17% phần còn lại. Về bộ nhớ, vấn đề xuất hiện sớm hơn nhiều nếu lưu ma trận attention, nên FlashAttention được dùng rộng rãi.

**Câu hỏi: Vì sao hết bộ nhớ khi huấn luyện, và xử lý thế nào?**

> **Trả lời.** Bộ nhớ gồm trọng số, gradient và trạng thái bộ tối ưu (khoảng 16 byte mỗi tham số với Adam và độ chính xác hỗn hợp, tức 112 GB cho mô hình 7B), cộng giá trị kích hoạt tỉ lệ với kích thước lô và độ dài chuỗi. Nếu phần tăng theo lô là nguyên nhân, giảm kích thước lô và dùng tích luỹ gradient; tiếp theo là gradient checkpointing (bộ nhớ kích hoạt từ $O(L)$ xuống $O(\sqrt L)$, đổi lấy khoảng một lượt xuôi tính lại), FlashAttention, và chia trạng thái bộ tối ưu cho nhiều GPU như ZeRO.

### 14.6. Thị giác máy tính

**Câu hỏi: IoU là gì, và vì sao dùng nó để đánh giá bộ phát hiện?**

> **Trả lời.** IoU là diện tích phần giao chia cho diện tích phần hợp của hai hộp, nằm trong khoảng từ 0 tới 1. Nó đo mức trùng khớp mà không phụ thuộc vào kích thước tuyệt đối của đối tượng, nên một ngưỡng như 0,5 dùng được cho cả đối tượng lớn lẫn nhỏ. Cần biết giới hạn của ngưỡng 0,5: một hộp chỉ phủ nửa đối tượng đã đạt IoU 0,5 (Ví dụ 7.2), vì vậy COCO lấy trung bình AP trên các ngưỡng từ 0,5 tới 0,95.

**Câu hỏi: NMS hoạt động thế nào, và nó có nhược điểm gì?**

> **Trả lời.** Sắp xếp các hộp theo điểm, lấy hộp điểm cao nhất, bỏ các hộp còn lại có IoU với nó vượt ngưỡng, rồi lặp lại. NMS cần thiết vì bộ phát hiện dự đoán trên lưới dày nên một đối tượng có nhiều hộp. Nhược điểm: với các đối tượng đứng sát nhau, hộp đúng của đối tượng thứ hai có thể bị loại vì trùng nhiều với hộp của đối tượng thứ nhất; trong thí nghiệm ở Mục 7.5, NMS làm recall lớn nhất giảm từ 0,928 xuống 0,916. DETR tránh NMS bằng cách ghép một–một giữa dự đoán và hộp thật khi huấn luyện.

**Câu hỏi: AP50 và AP@[0,5:0,95] khác nhau thế nào?**

> **Trả lời.** AP50 là diện tích dưới đường precision–recall khi một dự đoán được tính là đúng nếu IoU với hộp thật đạt 0,5. AP@[0,5:0,95], thước đo chính của COCO, là trung bình AP trên 10 ngưỡng IoU từ 0,5 tới 0,95, nên thưởng cho việc định vị chính xác. Cùng một bộ phát hiện mô phỏng ở Mục 7.5 có AP50 bằng 0,908 nhưng AP@[0,5:0,95] chỉ bằng 0,569. So sánh hai con số AP chỉ có nghĩa khi chúng cùng quy ước.

**Câu hỏi: Vì sao focal loss giúp bộ phát hiện một giai đoạn?**

> **Trả lời.** Bộ một giai đoạn đánh giá hàng chục nghìn vị trí mỗi ảnh, phần lớn là nền dễ. Với cross-entropy, tổng mất mát nhỏ của rất nhiều mẫu dễ lấn át mất mát của số ít đối tượng thật. Focal loss nhân thêm hệ số $(1 - p_t)^{\gamma}$: với $\gamma = 2$, mẫu đã đúng với xác suất 0,9 chỉ còn 1% trọng số, nên gradient tập trung vào các mẫu khó.

**Câu hỏi: U-Net khác một CNN phân loại ở điểm nào?**

> **Trả lời.** U-Net có nhánh mã hoá giống CNN phân loại và thêm một nhánh giải mã tăng dần độ phân giải về kích thước ảnh gốc. Ở mỗi mức độ phân giải, nhánh giải mã ghép đặc trưng của nhánh mã hoá cùng mức, để kết hợp thông tin "là gì" của lớp sâu với thông tin "ở đâu" của lớp nông. Nhờ vậy nó cho ra nhãn cho từng điểm ảnh với biên đối tượng chính xác, và học được từ ít ảnh có nhãn nếu tăng cường dữ liệu tốt.

### 14.7. Các câu trả lời chưa đạt

| Câu trả lời | Vì sao chưa đạt |
|---|---|
| "Chia $\sqrt{d_k}$ để huấn luyện ổn định." | Đúng nhưng chưa đủ. Cần nêu phương sai của tích vô hướng bằng $d_k$ và hiện tượng bão hoà của softmax. |
| "Nhiều đầu attention làm mô hình mạnh hơn." | Không nêu được rằng tổng chi phí không đổi và cái nhận được là khả năng theo dõi nhiều loại quan hệ. |
| "ResNet ra đời để chống overfitting." | Sai: nó giải quyết vấn đề suy thoái, tức sai số huấn luyện tăng theo độ sâu. |
| "Kết nối tắt giải quyết gradient tiêu biến." | Chưa đủ: một mình kết nối tắt làm gradient bùng nổ, cần đi kèm chuẩn hoá. |
| "Mô hình càng nhiều tham số càng overfitting." | Đúng khi số tham số nhỏ hơn số điểm dữ liệu; với mô hình nhiều tham số hơn dữ liệu có hiện tượng double descent. |
| "Cây quyết định cần chuẩn hoá đặc trưng." | Sai: cây bất biến với phép biến đổi đơn điệu của từng đặc trưng. |
| "Adam luôn tốt hơn SGD." | Adam hội tụ nhanh hơn và ít phải chỉnh, nhưng SGD với momentum có thể tổng quát hoá tốt hơn trên một số bài toán thị giác. |
| "Transformer nhanh hơn RNN vì ít phép tính hơn." | Sai: self-attention tốn nhiều phép tính hơn khi $T > d$. Transformer thắng ở số bước tuần tự. |
| "Mạng nào cũng cần dropout." | Nhiều mô hình ngôn ngữ lớn không dùng dropout khi tiền huấn luyện. |
| "GQA làm tăng chất lượng." | GQA giảm chất lượng một chút so với MHA; lý do dùng là KV cache nhỏ hơn. |
| "Bộ phát hiện này có AP 0,9 nên rất tốt." | Phải biết ngưỡng IoU và quy ước tính: cùng một bộ phát hiện có thể có AP50 bằng 0,91 và AP@[0,5:0,95] bằng 0,57 (Mục 7.5). |

---

## 15. Tài liệu tham khảo

**Sách và học máy cổ điển**

1. T. Hastie, R. Tibshirani, J. Friedman. *The Elements of Statistical Learning*, 2nd ed. Springer, 2009. Nguồn chuẩn cho phân rã độ chệch – phương sai, cây quyết định, bagging và boosting.
2. C. M. Bishop. *Pattern Recognition and Machine Learning*. Springer, 2006.
3. I. Goodfellow, Y. Bengio, A. Courville. *Deep Learning*. MIT Press, 2016. Nguồn chuẩn cho lan truyền ngược, khởi tạo, regularization và tối ưu.
4. D. H. Wolpert. The Lack of A Priori Distinctions Between Learning Algorithms. *Neural Computation*, 1996. Định lý "không có bữa trưa miễn phí" ở Mục 1.4.
5. L. Hyafil, R. L. Rivest. Constructing Optimal Binary Decision Trees is NP-Complete. *Information Processing Letters*, 1976.
6. L. Breiman. Bagging Predictors. *Machine Learning*, 1996.
7. L. Breiman. Random Forests. *Machine Learning*, 2001. Lập luận về tương quan giữa các cây và permutation importance.
8. J. H. Friedman. Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*, 2001.
9. J. H. Friedman. Stochastic Gradient Boosting. *Computational Statistics & Data Analysis*, 2002.
10. T. Chen, C. Guestrin. XGBoost: A Scalable Tree Boosting System. *KDD*, 2016.
11. G. Ke và cộng sự. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*, 2017.
12. L. Prokhorenkova và cộng sự. CatBoost: Unbiased Boosting with Categorical Features. *NeurIPS*, 2018.
13. S. M. Lundberg, S.-I. Lee. A Unified Approach to Interpreting Model Predictions. *NeurIPS*, 2017. Phương pháp SHAP.
14. L. Grinsztajn, E. Oyallon, G. Varoquaux. Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data? *NeurIPS Datasets and Benchmarks*, 2022. Nguồn của nhận định ở Mục 1.2.
15. M. Belkin, D. Hsu, S. Ma, S. Mandal. Reconciling Modern Machine-Learning Practice and the Classical Bias–Variance Trade-off. *PNAS*, 2019. Double descent.
16. P. Nakkiran và cộng sự. Deep Double Descent: Where Bigger Models and More Data Hurt. *ICLR*, 2020.

**Mạng nơ-ron và cách huấn luyện**

17. G. Cybenko. Approximation by Superpositions of a Sigmoidal Function. *Mathematics of Control, Signals and Systems*, 1989.
18. K. Hornik. Approximation Capabilities of Multilayer Feedforward Networks. *Neural Networks*, 1991.
19. M. Leshno, V. Y. Lin, A. Pinkus, S. Schocken. Multilayer Feedforward Networks with a Nonpolynomial Activation Function Can Approximate Any Function. *Neural Networks*, 1993.
20. M. Telgarsky. Benefits of Depth in Neural Networks. *COLT*, 2016. Ý tưởng của phép dựng hàm răng cưa ở Mục 4.4.
21. D. Hendrycks, K. Gimpel. Gaussian Error Linear Units (GELUs). arXiv:1606.08415, 2016.
22. S. Elfwing, E. Uchibe, K. Doya. Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning. *Neural Networks*, 2018.
23. P. Ramachandran, B. Zoph, Q. V. Le. Searching for Activation Functions. arXiv:1710.05941, 2017.
24. D. E. Rumelhart, G. E. Hinton, R. J. Williams. Learning Representations by Back-Propagating Errors. *Nature*, 1986.
25. A. G. Baydin, B. A. Pearlmutter, A. A. Radul, J. M. Siskind. Automatic Differentiation in Machine Learning: a Survey. *JMLR*, 2018. Chế độ xuôi và chế độ ngược ở Mục 5.2.
26. T. Chen, B. Xu, C. Zhang, C. Guestrin. Training Deep Nets with Sublinear Memory Cost. arXiv:1604.06174, 2016. Gradient checkpointing.
27. D. P. Kingma, J. Ba. Adam: A Method for Stochastic Optimization. *ICLR*, 2015.
28. I. Loshchilov, F. Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. *ICLR*, 2017. Lịch cosine.
29. I. Loshchilov, F. Hutter. Decoupled Weight Decay Regularization. *ICLR*, 2019. AdamW.
30. P. Goyal và cộng sự. Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour. arXiv:1706.02677, 2017. Quy tắc tuyến tính cho tốc độ học.
31. S. Malladi, K. Lyu, A. Panigrahi, S. Arora. On the SDEs and Scaling Rules for Adaptive Gradient Algorithms. *NeurIPS*, 2022. Quy tắc căn bậc hai cho Adam.
32. A. C. Wilson và cộng sự. The Marginal Value of Adaptive Gradient Methods in Machine Learning. *NeurIPS*, 2017.
33. Y. N. Dauphin và cộng sự. Identifying and Attacking the Saddle Point Problem in High-Dimensional Non-Convex Optimization. *NeurIPS*, 2014.
34. X. Glorot, Y. Bengio. Understanding the Difficulty of Training Deep Feedforward Neural Networks. *AISTATS*, 2010. Khởi tạo Xavier.
35. K. He, X. Zhang, S. Ren, J. Sun. Delving Deep into Rectifiers. *ICCV*, 2015. Khởi tạo He.
36. S. Ioffe, C. Szegedy. Batch Normalization. *ICML*, 2015.
37. J. L. Ba, J. R. Kiros, G. E. Hinton. Layer Normalization. arXiv:1607.06450, 2016.
38. B. Zhang, R. Sennrich. Root Mean Square Layer Normalization. *NeurIPS*, 2019.
39. S. Santurkar, D. Tsipras, A. Ilyas, A. Madry. How Does Batch Normalization Help Optimization? *NeurIPS*, 2018.
40. K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. *CVPR*, 2016. Vấn đề suy thoái và kết nối tắt.
41. N. Srivastava và cộng sự. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*, 2014.
42. C. Szegedy và cộng sự. Rethinking the Inception Architecture for Computer Vision. *CVPR*, 2016. Làm mượt nhãn.
43. A. Chowdhery và cộng sự. PaLM: Scaling Language Modeling with Pathways. arXiv:2204.02311, 2022.

**Mạng tích chập và mạng hồi quy**

44. K. Simonyan, A. Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR*, 2015.
45. A. G. Howard và cộng sự. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv:1704.04861, 2017.
46. W. Luo, Y. Li, R. Urtasun, R. Zemel. Understanding the Effective Receptive Field in Deep Convolutional Neural Networks. *NeurIPS*, 2016.
47. A. Azulay, Y. Weiss. Why Do Deep Convolutional Networks Generalize So Poorly to Small Image Transformations? *JMLR*, 2019.
48. A. Dosovitskiy và cộng sự. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*, 2021.
49. Z. Liu và cộng sự. A ConvNet for the 2020s. *CVPR*, 2022.
50. S. Hochreiter, J. Schmidhuber. Long Short-Term Memory. *Neural Computation*, 1997.
51. F. A. Gers, J. Schmidhuber, F. Cummins. Learning to Forget: Continual Prediction with LSTM. *Neural Computation*, 2000.
52. R. Pascanu, T. Mikolov, Y. Bengio. On the Difficulty of Training Recurrent Neural Networks. *ICML*, 2013. Điều kiện về giá trị suy biến và cắt ngưỡng gradient.
53. R. Jozefowicz, W. Zaremba, I. Sutskever. An Empirical Exploration of Recurrent Network Architectures. *ICML*, 2015.
54. C. Tallec, Y. Ollivier. Can Recurrent Neural Networks Warp Time? *ICLR*, 2018.
55. K. Cho và cộng sự. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. *EMNLP*, 2014. Mạng GRU.
56. J. Chung, C. Gulcehre, K. Cho, Y. Bengio. Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. arXiv:1412.3555, 2014.
57. A. Gu, T. Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752, 2023.

**Thị giác máy tính: phát hiện đối tượng và phân đoạn**

58. R. Girshick, J. Donahue, T. Darrell, J. Malik. Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation. *CVPR*, 2014. R-CNN.
59. R. Girshick. Fast R-CNN. *ICCV*, 2015.
60. S. Ren, K. He, R. Girshick, J. Sun. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. *NeurIPS*, 2015. Mạng đề xuất vùng và hộp neo.
61. J. Redmon, S. Divvala, R. Girshick, A. Farhadi. You Only Look Once: Unified, Real-Time Object Detection. *CVPR*, 2016.
62. W. Liu và cộng sự. SSD: Single Shot MultiBox Detector. *ECCV*, 2016.
63. T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár. Focal Loss for Dense Object Detection. *ICCV*, 2017. RetinaNet.
64. Z. Tian, C. Shen, H. Chen, T. He. FCOS: Fully Convolutional One-Stage Object Detection. *ICCV*, 2019.
65. X. Zhou, D. Wang, P. Krähenbühl. Objects as Points. arXiv:1904.07850, 2019. CenterNet.
66. N. Carion và cộng sự. End-to-End Object Detection with Transformers. *ECCV*, 2020. DETR.
67. M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, A. Zisserman. The PASCAL Visual Object Classes (VOC) Challenge. *IJCV*, 2010. Cách tính AP 11 điểm và mọi điểm.
68. T.-Y. Lin và cộng sự. Microsoft COCO: Common Objects in Context. *ECCV*, 2014. AP@[0,5:0,95].
69. J. Long, E. Shelhamer, T. Darrell. Fully Convolutional Networks for Semantic Segmentation. *CVPR*, 2015.
70. O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*, 2015.
71. V. Dumoulin, F. Visin. A Guide to Convolution Arithmetic for Deep Learning. arXiv:1603.07285, 2016. Kích thước đầu ra của tích chập chuyển vị.
72. A. Odena, V. Dumoulin, C. Olah. Deconvolution and Checkerboard Artifacts. *Distill*, 2016.
73. L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, A. L. Yuille. DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. *TPAMI*, 2018.
74. K. He, G. Gkioxari, P. Dollár, R. Girshick. Mask R-CNN. *ICCV*, 2017. RoIAlign.
75. A. Kirillov, K. He, R. Girshick, C. Rother, P. Dollár. Panoptic Segmentation. *CVPR*, 2019.
76. F. Milletari, N. Navab, S.-A. Ahmadi. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. *3DV*, 2016. Dice loss.
77. A. Kirillov và cộng sự. Segment Anything. *ICCV*, 2023. SAM và bộ dữ liệu SA-1B.
78. J. Ho, A. Jain, P. Abbeel. Denoising Diffusion Probabilistic Models. *NeurIPS*, 2020. U-Net làm mạng khử nhiễu.

**Transformer**

79. A. Vaswani và cộng sự. Attention Is All You Need. *NeurIPS*, 2017. Scaled dot-product attention, lập luận $\sqrt{d_k}$ ở chú thích 4, attention nhiều đầu và bảng so sánh với RNN.
80. D. Bahdanau, K. Cho, Y. Bengio. Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR*, 2015.
81. P. Michel, O. Levy, G. Neubig. Are Sixteen Heads Really Better than One? *NeurIPS*, 2019.
82. O. Press, N. A. Smith, M. Lewis. Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation. *ICLR*, 2022. ALiBi.
83. J. Devlin, M.-W. Chang, K. Lee, K. Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL*, 2019.
84. A. Radford và cộng sự. *Language Models are Unsupervised Multitask Learners*. OpenAI, 2019. GPT-2.
85. C. Raffel và cộng sự. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. *JMLR*, 2020. T5.
86. R. Xiong và cộng sự. On Layer Normalization in the Transformer Architecture. *ICML*, 2020. Pre-LN và post-LN.
87. J. Su và cộng sự. RoFormer: Enhanced Transformer with Rotary Position Embedding. *Neurocomputing*, 2024. RoPE.
88. S. Chen, S. Wong, L. Chen, Y. Tian. Extending Context Window of Large Language Models via Positional Interpolation. arXiv:2306.15595, 2023.
89. B. Peng, J. Quesnelle, H. Fan, E. Shippole. YaRN: Efficient Context Window Extension of Large Language Models. *ICLR*, 2024.
90. N. Shazeer. Fast Transformer Decoding: One Write-Head is All You Need. arXiv:1911.02150, 2019. MQA.
91. J. Ainslie và cộng sự. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. *EMNLP*, 2023.
92. N. Shazeer. GLU Variants Improve Transformer. arXiv:2002.05202, 2020. SwiGLU và việc giảm $d_{\text{ff}}$ còn hai phần ba.
93. N. Shazeer và cộng sự. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR*, 2017.
94. W. Fedus, B. Zoph, N. Shazeer. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. *JMLR*, 2022.
95. A. Q. Jiang và cộng sự. Mixtral of Experts. arXiv:2401.04088, 2024.
96. H. Touvron và cộng sự. Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv:2307.09288, 2023. Cấu hình mô hình và con số 184 320 giờ-GPU.
97. T. Dao, D. Y. Fu, S. Ermon, A. Rudra, C. Ré. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS*, 2022.
98. T. Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. *ICLR*, 2024.
99. M. Milakov, N. Gimelshein. Online Normalizer Calculation for Softmax. arXiv:1805.02867, 2018.

**Giải mã**

100. A. Fan, M. Lewis, Y. Dauphin. Hierarchical Neural Story Generation. *ACL*, 2018. Lấy mẫu top-$k$.
101. A. Holtzman, J. Buys, L. Du, M. Forbes, Y. Choi. The Curious Case of Neural Text Degeneration. *ICLR*, 2020. Top-$p$ và nhận xét về beam search.
102. M. Nguyen và cộng sự. Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs. *ICLR*, 2025.
103. N. S. Keskar và cộng sự. CTRL: A Conditional Transformer Language Model for Controllable Generation. arXiv:1909.05858, 2019. Phạt lặp.
104. Y. Leviathan, M. Kalman, Y. Matias. Fast Inference from Transformers via Speculative Decoding. *ICML*, 2023.
105. C. Chen và cộng sự. Accelerating Large Language Model Decoding with Speculative Sampling. arXiv:2302.01318, 2023.
106. H. He. *Defeating Nondeterminism in LLM Inference*. Thinking Machines Lab, 2025.

**Tài nguyên tính toán**

107. J. Kaplan và cộng sự. Scaling Laws for Neural Language Models. arXiv:2001.08361, 2020. Công thức $12Ld^2$, $2N$ và $6ND$.
108. J. Hoffmann và cộng sự. Training Compute-Optimal Large Language Models. *NeurIPS*, 2022. Chinchilla.
109. S. Rajbhandari, J. Rasley, O. Ruwase, Y. He. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. *SC*, 2020. Con số 16 byte mỗi tham số.
110. V. Korthikanti và cộng sự. Reducing Activation Recomputation in Large Transformer Models. arXiv:2205.05198, 2022. Công thức bộ nhớ kích hoạt ở Mục 12.5.

---

## Phụ lục: chạy lại thí nghiệm

```text
code/models/
├── experiments.py            # Hình 2, 4, 5, 7, 10, 11, 14, 15 và mọi số liệu đo được
├── experiments_output.txt    # kết quả in ra của script trên
├── thi_giac.py               # Hình 9 và số liệu về IoU, NMS, AP ở Mục 7.5–7.6
├── thi_giac_output.txt       # kết quả in ra của thi_giac.py
└── fig_diagrams.py           # Hình 1, 3, 6, 8, 12, 13 (sơ đồ khái niệm)
```

```bash
pip install numpy scipy matplotlib scikit-learn
python code/models/experiments.py     # khoảng một tới vài phút trên CPU
python code/models/thi_giac.py        # vài giây
python code/models/fig_diagrams.py
```

Các script đặt hạt giống cố định nên mọi con số trong giáo trình lặp lại được trên cùng phiên bản thư viện. Môi trường đã dùng: Python 3.13, NumPy 2.3, SciPy 1.16, scikit-learn 1.7, matplotlib 3.10. Trên Windows, đặt `PYTHONIOENCODING=utf-8` khi ghi kết quả ra tệp.

Một số kết quả là phép đối chiếu với con số đã công bố, không phải mô phỏng: số tham số của GPT-2 small, medium, large và của Llama 2 7B, 13B; ngân sách giờ-GPU của Llama 2 7B; và phương sai của tích vô hướng trong chú thích 4 của bài báo Transformer. Các phép đối chiếu này kiểm tra rằng công thức ở Chương 12 đúng, không chỉ hợp lý.

Các thí nghiệm dùng dữ liệu mô phỏng nhỏ để quan sát cơ chế một cách rõ ràng. Hai điểm cần nhớ khi đọc kết quả: thí nghiệm ở Mục 4.4 so sánh khả năng biểu diễn chứ không phải khả năng học bằng gradient descent; và thí nghiệm về LSTM ở Mục 8.2 chỉ mô phỏng đường đi qua ô nhớ, không phải một mạng LSTM đầy đủ.
