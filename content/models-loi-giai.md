# Lời giải chi tiết — Học sâu

Mỗi mục ứng với một bài trong Chương 13. Dòng `@meta` được script build đọc để gắn nhãn
chương, dạng bài và độ khó; nó không hiện ra trên trang.

## Bài 1
@meta chuong=12 | dang=Tính tay | kho=Cơ bản

**(a) Từng thành phần.**

*Attention.* Với 32 đầu truy vấn và 32 đầu khoá–giá trị, mỗi đầu $d_{\text{head}} = 4\,096/32 = 128$ chiều, nên $W_Q$, $W_K$, $W_V$, $W_O$ đều là ma trận $4\,096 \times 4\,096$. Không có hệ số chặn:

$$4d^2 \times L = 4 \times 4\,096^2 \times 32 = 2\,147\,483\,648.$$

*FFN.* SwiGLU có ba ma trận $W$, $V$ (kích thước $d \times d_{\text{ff}}$) và $W_2$ (kích thước $d_{\text{ff}} \times d$):

$$3 \times d \times d_{\text{ff}} \times L = 3 \times 4\,096 \times 11\,008 \times 32 = 4\,328\,521\,728.$$

*Chuẩn hoá.* RMSNorm chỉ có vector $\gamma$. Mỗi khối có hai lớp chuẩn hoá (trước attention và trước FFN), cộng một lớp sau khối cuối:

$$32 \times 2 \times 4\,096 + 4\,096 = 266\,240.$$

*Embedding và lớp chiếu ra.* Không dùng chung trọng số nên tính hai lần:

$$2 \times 32\,000 \times 4\,096 = 262\,144\,000.$$

**(b) Tổng.**

| Thành phần | Số tham số | Tỉ lệ |
|---|---|---|
| Attention | 2 147 483 648 | 31,9% |
| FFN (SwiGLU) | 4 328 521 728 | 64,2% |
| RMSNorm | 266 240 | 0,004% |
| Embedding và lớp chiếu ra | 262 144 000 | 3,9% |
| Tổng | 6 738 415 616 | |

Tổng khớp chính xác con số đã công bố 6 738 415 616.

**(c) Tỉ số $d_{\text{ff}}/d$.** $11\,008/4\,096 = 2{,}6875$, không phải 4. Vì SwiGLU có ba ma trận thay vì hai, muốn giữ số tham số bằng FFN gốc với $d_{\text{ff}} = 4d$ thì phải có $3d\,d_{\text{ff}} = 8d^2$, tức $d_{\text{ff}} = \tfrac83 d = 10\,922{,}67$. Làm tròn lên bội của 256 được $43 \times 256 = 11\,008$ (Mục 10.5).

> **Nhận xét.** Trong phạm vi các khối Transformer, FFN chiếm $3 \times 2{,}6875/(4 + 3 \times 2{,}6875) = 66{,}8\%$ số tham số, gần với tỉ lệ $2/3$ của FFN gốc ở Mục 12.2. Đó chính là mục đích của việc chọn $d_{\text{ff}} \approx \tfrac83 d$.

## Bài 2
@meta chuong=12 | dang=Tính tay | kho=Cơ bản

**(a) Số FLOP.** Từ Bài 1, số tham số không tính embedding là

$$N = 2\,147\,483\,648 + 4\,328\,521\,728 = 6\,476\,005\,376 \approx 6{,}48 \times 10^9,$$

nên

$$C = 6ND = 6 \times 6{,}476 \times 10^9 \times 2 \times 10^{12} = 7{,}771 \times 10^{22} \text{ FLOP}.$$

**(b) Số giờ-GPU.** Tốc độ thực tế là $312 \times 10^{12} \times \text{MFU}$ FLOP mỗi giây:

- MFU 30%: $7{,}771 \times 10^{22} / (312 \times 10^{12} \times 0{,}30) = 8{,}30 \times 10^8$ giây, tức 230 627 giờ-GPU;
- MFU 40%: 172 970 giờ-GPU.

**(c) MFU thực tế.** Giải ngược từ con số công bố:

$$\text{MFU} = \frac{7{,}771 \times 10^{22}}{312 \times 10^{12} \times 184\,320 \times 3\,600} = 0{,}375.$$

Mức 37,5% nằm trong khoảng 30% tới 50% thường gặp khi huấn luyện mô hình lớn, nên công thức $6ND$ nhất quán với thực tế. Cần hiểu đúng ý nghĩa của phép so sánh: công thức cho đúng bậc độ lớn của lượng tính toán, còn thời gian thực phụ thuộc vào MFU, một đại lượng công thức không dự đoán được.

**(d) Vì sao không tính embedding.** Với mỗi token, lớp embedding chỉ là một phép tra bảng, không có phép nhân ma trận nào, nên gần như không tốn FLOP. Lớp chiếu ra từ vựng thì có phép nhân ma trận ($2Vd$ FLOP mỗi token) nhưng thường được bỏ qua trong công thức gần đúng; với Llama 2 7B, phần này chỉ khoảng 2% so với $2N$. Nếu dùng tổng 6,74 tỉ thay cho 6,48 tỉ, kết quả lệch khoảng 4%.

## Bài 3
@meta chuong=9 | dang=Suy luận | kho=Trung bình

**(a) Phương sai của tích vô hướng.** Viết $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$. Vì $q_i$ và $k_i$ độc lập và có trung bình 0,

$$\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\,\mathbb{E}[k_i] = 0, \qquad \operatorname{Var}(q_i k_i) = \mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = 1.$$

Các số hạng độc lập với nhau nên phương sai của tổng bằng tổng các phương sai: $\operatorname{Var}(q \cdot k) = d_k$. Số đo ở Mục 9.2 với $d_k = 4, 16, 64, 256, 1\,024$ là 3,63; 15,99; 64,66; 251,67; 1 021,97, khớp lý thuyết trong phạm vi sai số lấy mẫu.

**(b) Vì sao gradient gần bằng 0.** Đạo hàm của softmax là

$$\frac{\partial p_i}{\partial s_j} = p_i(\delta_{ij} - p_j).$$

Khi một $p_i$ gần 1 và các $p_j$ còn lại gần 0, mọi phần tử của ma trận này đều gần 0: với $i = j$ là $p_i(1 - p_i) \approx 0$, với $i \ne j$ là $-p_i p_j \approx 0$. Điểm số có độ lệch chuẩn $\sqrt{d_k}$; với $d_k = 1\,024$, các điểm số chênh nhau hàng chục đơn vị, và tỉ số $e^{s_i - s_j}$ giữa hai trọng số có thể lên tới $10^{13}$, nên softmax gần như luôn ở vùng gần one-hot. Bảng ở Mục 9.2 cho thấy điều đó: không chia thì trọng số lớn nhất trung bình là 0,953 và entropy chỉ còn 0,118 nat.

**(c) Chia cho $d_k$.** Phương sai của điểm số thành $d_k/d_k^2 = 1/d_k$, tiến về 0 khi $d_k$ lớn. Các điểm số gần bằng nhau, softmax cho phân phối gần đều, và attention mất khả năng phân biệt các vị trí: mỗi token nhận gần như trung bình của cả chuỗi. Chia cho $\sqrt{d_k}$ là cách duy nhất trong họ $d_k^{\alpha}$ làm phương sai không phụ thuộc $d_k$; chia ít hơn thì bão hoà, chia nhiều hơn thì phân phối bị làm phẳng.

## Bài 4
@meta chuong=7 | dang=Tính tay | kho=Cơ bản

**(a) Bước nhảy 1 ở mọi lớp.** Với $k_i = 3$, $s_j = 1$, công thức cho $r_i = r_{i-1} + 2$, nên $r_L = 2L + 1$. Giải $2L + 1 \ge 224$ được $L \ge 111{,}5$, tức cần **112 lớp**.

**(b) Bước nhảy 2 ở các lớp chẵn.** Dãy bước nhảy là $(1, 2, 1, 2, \dots)$. Mức tăng của lớp $i$ là $2\prod_{j<i}s_j$:

| Lớp $i$ | $\prod_{j<i} s_j$ | Mức tăng | $r_i$ |
|---|---|---|---|
| 1 | 1 | 2 | 3 |
| 2 | 1 | 2 | 5 |
| 3 | 2 | 4 | 9 |
| 4 | 2 | 4 | 13 |
| 5 | 4 | 8 | 21 |
| 6 | 4 | 8 | 29 |
| 7 | 8 | 16 | 45 |
| 8 | 8 | 16 | 61 |
| 9 | 16 | 32 | 93 |
| 10 | 16 | 32 | 125 |
| 11 | 32 | 64 | 189 |
| 12 | 32 | 64 | 253 |

Sau 10 lớp, trường tiếp nhận rộng 125 điểm ảnh; cần 12 lớp để phủ 224 điểm ảnh, so với 112 lớp khi không có bước nhảy. Bước nhảy nhân vào mọi mức tăng phía sau, nên trường tiếp nhận tăng theo cấp số nhân thay vì tuyến tính. Đổi lại, độ phân giải của bản đồ đặc trưng giảm một nửa theo mỗi chiều sau mỗi lớp có bước nhảy 2.

**(c) Trường tiếp nhận hiệu dụng.** Các điểm trong trường tiếp nhận lý thuyết không ảnh hưởng như nhau. Số đường đi từ một điểm ảnh tới đơn vị đang xét lớn nhất ở tâm và giảm dần ra rìa; ảnh hưởng tổng hợp qua nhiều lớp tích chập giống kết quả của nhiều lần tích chập liên tiếp, và theo định lý giới hạn trung tâm, nó có dạng gần Gauss. Luo và cộng sự (2016) cho thấy độ rộng hiệu dụng chỉ tăng theo $\sqrt L$. Vì vậy chỉ thêm lớp không đủ để mở rộng vùng nhìn; cần bước nhảy, gộp, tích chập giãn cách hoặc attention.

## Bài 5
@meta chuong=2 | dang=Chẩn đoán | kho=Cơ bản

**(a) Huấn luyện 2%, xác thực 18%, khoảng cách lớn và thu hẹp dần khi thêm dữ liệu.** Phương sai cao (overfitting). Hai việc nên làm: thêm dữ liệu hoặc tăng cường dữ liệu, vì phương sai giảm khi có thêm dữ liệu và đường cong cho thấy dữ liệu đang có tác dụng; và tăng regularization (weight decay, dropout, dừng sớm), rẻ hơn thu thập dữ liệu và thường đủ.

**(b) Huấn luyện 24%, xác thực 25%, cả hai đã nằm ngang.** Độ chệch cao (underfitting). Thêm dữ liệu sẽ không giúp gì, và đây là kết luận quan trọng nhất của chẩn đoán này. Hai việc nên làm: dùng mô hình linh hoạt hơn (thêm lớp, tăng bề rộng hoặc đổi họ mô hình), và thêm đặc trưng hoặc giảm regularization.

**(c) Xác thực 12%, huấn luyện 15%.** Sai số xác thực thấp hơn sai số huấn luyện không phải là một chẩn đoán về độ chệch hay phương sai mà là dấu hiệu cần kiểm tra quy trình. Ba nguyên nhân thường gặp:

1. **Dropout hoặc tăng cường dữ liệu chỉ bật khi huấn luyện.** Sai số huấn luyện khi đó được đo trên bài toán khó hơn. Đây là nguyên nhân phổ biến nhất và không phải lỗi.
2. **Sai số huấn luyện được tính trung bình trong suốt một epoch**, gồm cả những bước đầu khi mô hình còn kém, còn sai số xác thực đo ở cuối epoch.
3. **Tập xác thực dễ hơn hoặc có rò rỉ dữ liệu**: chia không ngẫu nhiên, thiếu các trường hợp khó, hoặc chứa bản sao gần giống của dữ liệu huấn luyện.

Nên kiểm tra hai nguyên nhân đầu trước, vì chúng là vấn đề đo lường. Nếu loại trừ được cả hai mà hiện tượng vẫn còn, cần xem lại cách chia dữ liệu ([Mục 8.5 của *Nền tảng*](nentang-ch08.html)).

## Bài 6
@meta chuong=4 | dang=Suy luận | kho=Trung bình

**(a) $g$ ánh xạ $[0,1]$ lên $[0,1]$.** Với $x \in [0, \tfrac12]$: $\operatorname{ReLU}(x) = x$ và $\operatorname{ReLU}(x - \tfrac12) = 0$, nên $g(x) = 2x$, tăng từ 0 tới 1. Với $x \in [\tfrac12, 1]$: $g(x) = 2x - 4(x - \tfrac12) = 2 - 2x$, giảm từ 1 về 0. Vậy $g$ liên tục, có giá trị trong $[0, 1]$, và mỗi nửa đoạn được ánh xạ lên toàn bộ $[0, 1]$.

**(b) Quy nạp.** Mệnh đề: $g^{(k)}$ là hàm tuyến tính từng khúc gồm $2^k$ đoạn, mỗi đoạn ánh xạ lên toàn bộ $[0, 1]$ (tăng từ 0 lên 1 hoặc giảm từ 1 về 0). Với $k = 1$, mệnh đề đúng theo (a). Giả sử mệnh đề đúng với $k$. Trên mỗi đoạn của $g^{(k)}$, giá trị đi qua toàn bộ $[0, 1]$ một cách đơn điệu, nên đi qua cả hai nửa $[0, \tfrac12]$ và $[\tfrac12, 1]$ trên đó $g$ tuyến tính. Do đó mỗi đoạn của $g^{(k)}$ tách thành hai đoạn của $g^{(k+1)} = g \circ g^{(k)}$, mỗi đoạn mới lại ánh xạ lên toàn bộ $[0, 1]$. Vậy $g^{(k+1)}$ có $2^{k+1}$ đoạn.

Mạng sâu thực hiện $g$ ở mỗi lớp bằng 2 đơn vị ReLU, nên $g^{(k)}$ dùng $2k$ đơn vị trong $k$ lớp, tức $6k$ tham số (mỗi đơn vị một trọng số vào, một hệ số chặn, một trọng số ra).

**(c) Mạng một lớp ẩn.** Một mạng một lớp ẩn với $m$ đơn vị ReLU và đầu vào một chiều có dạng

$$f(x) = \sum_{j=1}^{m} c_j \operatorname{ReLU}(a_j x + b_j) + e.$$

Mỗi số hạng là hàm tuyến tính từng khúc với đúng một điểm gãy tại $x = -b_j/a_j$, nên $f$ có nhiều nhất $m$ điểm gãy. Hàm $g^{(k)}$ có $2^k$ đoạn, tức $2^k - 1$ điểm gãy bên trong $(0, 1)$, nên cần $m \ge 2^k - 1$.

So sánh: mạng sâu cần $2k$ đơn vị, mạng một lớp cần ít nhất $2^k - 1$. Với $k = 7$ là 14 so với 127 đơn vị; với $k = 20$ là 40 so với hơn một triệu. Thí nghiệm ở Mục 4.4 cho thấy cận dưới này sát với thực tế: sai số của mạng một lớp chỉ giảm rõ khi bề rộng đạt $2^k$.

> **Lưu ý.** Bài này chứng minh khả năng biểu diễn, không chứng minh rằng gradient descent tìm được nghiệm của mạng sâu. Trong thực tế, huấn luyện mạng hẹp và sâu để học hàm này thường thất bại.

## Bài 7
@meta chuong=8 | dang=Suy luận | kho=Trung bình

**(a) RNN tuyến tính.**

$$0{,}9^{100} = 2{,}66 \times 10^{-5}, \qquad 1{,}1^{100} = 1{,}38 \times 10^{4}.$$

Bán kính phổ chỉ khác nhau 20% nhưng kết quả chênh nhau 9 bậc độ lớn, vì $\rho \mapsto \rho^T$ rất nhạy khi $T$ lớn.

Số đo ở Mục 8.2 khác xa các con số này: với $g = 0{,}9$ là $3{,}2 \times 10^{-2}$, với $g = 1{,}2$ là 3,0. Có hai lý do. Thứ nhất, RNN trong thí nghiệm có hàm tanh, và khi các đơn vị bão hoà, đạo hàm $\tanh' < 1$ làm Jacobi nhỏ đi; với $g > 1$, trạng thái lớn dần theo thời gian nên hiệu ứng này mạnh ở cuối chuỗi. Thứ hai, với ma trận ngẫu nhiên $64 \times 64$, bán kính phổ thực tế chỉ xấp xỉ $g$ và độ lớn của $W^T v$ phụ thuộc cả vào các trị riêng khác. Ước lượng $\rho^T$ đúng về xu hướng nhưng không đúng về con số.

**(b) Đường ô nhớ của LSTM.**

$$b_f = 1: \; f = \sigma(1) = 0{,}731, \quad 0{,}731^{100} = 2{,}5 \times 10^{-14}, \quad 1/(1-f) \approx 3{,}7 \text{ bước}.$$
$$b_f = 4: \; f = \sigma(4) = 0{,}982, \quad 0{,}982^{100} = 0{,}16, \quad 1/(1-f) \approx 56 \text{ bước}.$$

Mô phỏng ở Mục 8.2 cho $3{,}0 \times 10^{-15}$ và $0{,}13$, cùng bậc với tính tay; phần chênh đến từ nhiễu cộng vào đầu vào của cổng quên ở mỗi bước. Với $b_f = 1$, bộ nhớ có thang thời gian chỉ vài bước; với $b_f = 4$, khoảng 50 bước. Cấu trúc cổng cho phép mạng điều khiển tích $\prod f_t$, nhưng giá trị của cổng quyết định bộ nhớ dài hay ngắn.

**(c) Bùng nổ và tiêu biến.** Gradient bùng nổ vẫn giữ đúng hướng, chỉ có độ lớn quá mức; cắt ngưỡng gradient giới hạn độ lớn và giữ nguyên hướng, nên sửa được với một dòng mã. Gradient tiêu biến thì thông tin về ảnh hưởng của các bước xa đã nhỏ hơn nhiều so với nhiễu của SGD và sai số làm tròn; phóng to nó lên chỉ phóng to nhiễu. Tiêu biến phải được ngăn từ kiến trúc: cổng như LSTM, kết nối tắt, chuẩn hoá, khởi tạo phù hợp.

## Bài 8
@meta chuong=10 | dang=Tính tay | kho=Cơ bản

**Công thức.** Mỗi lớp lưu khoá và giá trị, mỗi thứ $n_{\text{kv}} \times d_{\text{head}}$ số cho mỗi token của mỗi chuỗi:

$$\text{số byte} = 2 \times L \times n_{\text{kv}} \times d_{\text{head}} \times T \times B \times 2,$$

với hệ số 2 đầu cho khoá và giá trị, hệ số 2 cuối là số byte của FP16.

**(a) MHA, 64 đầu khoá–giá trị.**

$$2 \times 80 \times 64 \times 128 \times 4\,096 \times 8 \times 2 = 85\,899\,345\,920 \text{ byte} = 80 \text{ GiB}.$$

**(b) GQA, 8 đầu khoá–giá trị.** Chỉ $n_{\text{kv}}$ thay đổi, nên dung lượng giảm đúng $64/8 = 8$ lần: 10 GiB.

**(c) So với trọng số.** Mô hình 70 tỉ tham số ở FP16 chiếm khoảng 140 GB trọng số. Với MHA, KV cache 80 GiB (86 GB) đã bằng hơn một nửa dung lượng trọng số, chỉ với 8 chuỗi 4 096 token; với 64 chuỗi, KV cache là 640 GiB, gần gấp năm lần trọng số. GQA là quyết định vì bộ nhớ khi suy luận vì ba lẽ:

1. **KV cache giới hạn số yêu cầu phục vụ được cùng lúc.** Trọng số cố định, còn KV cache tăng tuyến tính theo độ dài ngữ cảnh và số chuỗi. Với GQA, mỗi token của mỗi chuỗi chỉ tốn 0,3125 MiB thay vì 2,5 MiB.
2. **Sinh token bị giới hạn bởi băng thông bộ nhớ.** Mỗi token mới phải đọc lại toàn bộ KV cache; KV cache nhỏ hơn 8 lần thì lượng dữ liệu phải đọc cũng nhỏ hơn, và độ trễ giảm theo.
3. **Chất lượng không tăng.** Các đầu truy vấn trong một nhóm buộc phải dùng chung khoá và giá trị, nên GQA giảm chất lượng một chút so với MHA; Ainslie và cộng sự (2023) cho thấy mức giảm nhỏ khi số nhóm không quá ít.

## Bài 9
@meta chuong=1 | dang=Thiết kế | kho=Trung bình

Điểm chính của bài là lý do dựa trên cấu trúc của dữ liệu và thiên kiến quy nạp của mô hình. "Mô hình X thắng nhiều cuộc thi" không phải là lý do.

**(a) Khách hàng rời bỏ: 40 đặc trưng bảng, 30 000 mẫu, cần giải thích.** Chọn gradient boosting trên cây (XGBoost, LightGBM), hoặc rừng ngẫu nhiên nếu cần một mô hình dễ chỉnh. Dữ liệu bảng không có cấu trúc không gian hay thời gian để tích chập hay attention khai thác; nó thường có quan hệ dạng ngưỡng, tương tác giữa các đặc trưng và các cột có thang đo khác nhau, đúng loại cấu trúc mà phép chia theo từng trục của cây biểu diễn tự nhiên (Mục 1.2). 30 000 mẫu cũng ít đối với mạng nơ-ron. Về giải thích, có thể dùng SHAP để giải thích từng dự đoán cho bộ phận kinh doanh.

**(b) Ảnh sản phẩm lỗi: 3 000 ảnh có nhãn, bố cục giống nhau.** Chọn CNN đã tiền huấn luyện rồi tinh chỉnh, ví dụ ResNet hoặc EfficientNet. Các giả định của tích chập đúng với ảnh, nên CNN cần ít dữ liệu hơn ViT ở quy mô này (Mục 7.4). 3 000 ảnh không đủ để huấn luyện từ đầu, nên dùng đặc trưng đã học từ ImageNet ([Chương 6 của *Biểu diễn & Căn chỉnh*](bieudien-ch06.html)). Chi tiết "bố cục giống nhau" cần chú ý: nếu vị trí của khuyết tật có ý nghĩa, các phép tăng cường dữ liệu như lật hay dịch mạnh có thể có hại, và tính bất biến với dịch chuyển của lớp gộp toàn cục có thể không mong muốn.

**(c) Trích xuất thông tin từ hợp đồng 50 trang.** Chọn Transformer đã tiền huấn luyện, có ngữ cảnh đủ dài, hoặc kết hợp chia văn bản thành đoạn với truy xuất. Bài toán có nhiều phụ thuộc xa, ví dụ điều khoản ở trang 40 tham chiếu định nghĩa ở trang 2; attention cho đường đi độ dài $O(1)$ giữa hai vị trí bất kỳ. Tiền huấn luyện là cần thiết vì không có đủ hợp đồng có nhãn để học ngôn ngữ pháp lý từ đầu. Về chi phí, 50 trang tương ứng khoảng 25 000 tới 35 000 token; với một mô hình có $d = 4\,096$, tỉ số $T/(6d)$ vào khoảng 1 tới 1,4, tức phần tính toán của attention đã ngang phần còn lại (Mục 12.4), nên cân nhắc FlashAttention hoặc chỉ đưa vào các đoạn liên quan.

**(d) Dự báo phụ tải điện theo giờ cho 12 tháng.** Chọn mô hình khai thác trực tiếp cấu trúc đã biết: gradient boosting trên các đặc trưng lịch (giờ trong ngày, ngày trong tuần, mùa, ngày lễ), đặc trưng trễ và dự báo thời tiết, hoặc mô hình chuỗi thời gian có thành phần mùa vụ tường minh. Phụ tải bị chi phối bởi các chu kỳ đã biết trước và nhiệt độ; đưa chúng vào làm đặc trưng thì mô hình không phải học lại chúng từ dữ liệu. Quan trọng hơn cả việc chọn mô hình là cách đánh giá: phải chia dữ liệu theo thời gian, huấn luyện trên quá khứ và kiểm tra trên giai đoạn sau, không bao giờ chia ngẫu nhiên ([Mục 8.5 của *Nền tảng*](nentang-ch08.html)).

## Bài 10
@meta chuong=9 | dang=Thí nghiệm | kho=Nâng cao

**Cách làm.** Trong phần (E), thay `softmax(scores)` bằng một hàm chuẩn hoá không có hàm mũ:

```python
def relu_norm(s, axis=-1, eps=1e-9):
    """Chuan hoa l1 cua ReLU: khong co ham mu nen khong bao hoa."""
    w = np.maximum(s, 0.0)
    return w / (w.sum(axis=axis, keepdims=True) + eps)
```

rồi đo entropy của trọng số theo $d_k = 4, 16, 64, 256, 1\,024$ khi không chia cho $\sqrt{d_k}$.

**(a) Kết quả dự kiến.** Hiện tượng bão hoà biến mất: entropy gần như không đổi theo $d_k$. Với softmax, tỉ số giữa hai trọng số là $e^{s_i - s_j}$, hàm mũ của hiệu hai điểm số, nên khi điểm số có độ lệch chuẩn cỡ 32, một phần tử áp đảo các phần tử còn lại. Với chuẩn hoá $\ell_1$ của ReLU, tỉ số là $s_i/s_j$ (với các điểm số dương); nhân mọi điểm số với cùng một hằng số dương không thay đổi tỉ số này, nên phân phối trọng số không phụ thuộc thang đo của điểm số.

**(b) Softmax hay tích vô hướng.** Tích vô hướng làm phương sai của điểm số tăng theo $d_k$ (Bài 3). Nhưng phương sai lớn chỉ gây hại vì softmax nhạy với thang đo tuyệt đối của đầu vào. Nguyên nhân trực tiếp nằm ở tích vô hướng, còn cơ chế gây hại nằm ở softmax. Cách sửa được chọn ở phía tích vô hướng vì chỉ tốn một phép chia, trong khi thay softmax sẽ thay đổi tính chất của mô hình.

**(c) Vì sao vẫn dùng softmax.**

1. **Luôn dương và trơn.** Chuẩn hoá ReLU cho trọng số bằng đúng 0 và gradient bằng 0 ở mọi vị trí có điểm số âm, nên các vị trí đó không bao giờ được xét lại, tương tự đơn vị ReLU chết.
2. **Mẫu số luôn dương.** Với ReLU, nếu mọi điểm số đều âm thì tổng bằng 0; phải thêm `eps`, và `eps` trở thành một siêu tham số ngầm.
3. **Có thể chọn dứt khoát khi cần.** Một số nhiệm vụ cần phân phối gần one-hot, ví dụ sao chép chính xác một token từ ngữ cảnh. Tính chất hàm mũ của softmax cho phép điều đó với điểm số vừa phải.

> **Nhận xét.** Khi chạy, nên ghi thêm entropy khi có chia $\sqrt{d_k}$ cho cả hai hàm để có bốn cột so sánh: chuẩn hoá ReLU cho entropy gần như không đổi trong cả hai trường hợp, còn softmax chỉ ổn định khi có phép chia.

## Bài 11
@meta chuong=7 | dang=Tính tay | kho=Trung bình

**(a) IoU.** Hộp $A$ có diện tích $40 \times 30 = 1\,200$, hộp $B$ có diện tích $40 \times 40 = 1\,600$. Phần giao theo trục ngang là $[30, 50]$, theo trục dọc là $[20, 40]$, diện tích $20 \times 20 = 400$. Phần hợp là $1\,200 + 1\,600 - 400 = 2\,400$, nên

$$\operatorname{IoU}(A, B) = \frac{400}{2\,400} = \frac{1}{6} \approx 0{,}167.$$

**(b) NMS.** Các IoU cần dùng: $\operatorname{IoU}(P_1, P_2) = 81/119 \approx 0{,}681$ (giao $9 \times 9$, hợp $100 + 100 - 81$); $\operatorname{IoU}(P_1, P_3) = 25/175 \approx 0{,}143$; $\operatorname{IoU}(P_1, P_4) = 0$; $\operatorname{IoU}(P_3, P_4) = 0$.

1. Chọn $P_1$ (điểm 0,9). Bỏ $P_2$ vì IoU 0,681 vượt 0,5; giữ $P_3$ và $P_4$.
2. Trong các hộp còn lại, chọn $P_3$ (0,7); IoU với $P_4$ bằng 0 nên giữ $P_4$.
3. Chọn $P_4$.

Kết quả: $P_1$, $P_3$, $P_4$. Lưu ý rằng $P_2$ bị loại vì trùng với $P_1$ chứ không vì trùng với $P_3$ (IoU giữa $P_2$ và $P_3$ chỉ là $36/164 \approx 0{,}22$).

**(c) Precision, recall và AP.** Với 4 hộp thật:

| Hộp thứ | Kết quả | TP tích luỹ | Precision | Recall |
|---|---|---|---|---|
| 1 | đúng | 1 | 1,000 | 0,25 |
| 2 | đúng | 2 | 1,000 | 0,50 |
| 3 | sai | 2 | 0,667 | 0,50 |
| 4 | đúng | 3 | 0,750 | 0,75 |
| 5 | sai | 3 | 0,600 | 0,75 |
| 6 | sai | 3 | 0,500 | 0,75 |

Precision nội suy tại mỗi mức recall là precision lớn nhất ở các mức recall lớn hơn hoặc bằng: tại recall 0,25 và 0,50 là 1; tại recall 0,75 là 0,75. Recall không bao giờ đạt 1 vì một hộp thật không được phát hiện, nên đoạn từ 0,75 tới 1 đóng góp 0. AP theo cách tính mọi điểm:

$$\text{AP} = 0{,}25 \times 1 + 0{,}25 \times 1 + 0{,}25 \times 0{,}75 = 0{,}6875.$$

Theo cách tính 11 điểm của VOC 2007: precision nội suy bằng 1 tại 6 mức recall $0; 0{,}1; \dots; 0{,}5$, bằng 0,75 tại hai mức 0,6 và 0,7, và bằng 0 tại 0,8; 0,9; 1. AP $= (6 + 1{,}5)/11 \approx 0{,}682$, gần nhưng không bằng cách tính mọi điểm.

**(d) Hộp trùng.** Mỗi hộp thật chỉ được ghép với một hộp dự đoán. Hộp dự đoán thứ hai của cùng một đối tượng không còn hộp thật nào để ghép, nên bị tính là dương tính giả và làm giảm precision. Vì vậy NMS làm tăng AP: nó loại các hộp trùng trước khi đánh giá. Trong thí nghiệm ở Mục 7.5, AP50 tăng từ 0,885 lên 0,913 sau NMS. Nhưng nếu ngưỡng NMS quá thấp, hộp đúng của một đối tượng khác đứng sát bên cũng bị loại, và recall giảm.
