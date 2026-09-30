# Lời giải chi tiết — Mô hình và kiến trúc

Mỗi mục ứng với một bài trong Chương 13. Dòng `@meta` được script build đọc để gắn nhãn
chương, dạng bài và độ khó; nó không hiện ra trên trang.

## Bài 1
@meta chuong=12 | dang=Tính tay | kho=Cơ bản

**Bước 1 — attention.** Với GQA thì $W_K$ và $W_V$ có chiều ra là $n_{\text{kv}} \times d_{\text{head}}$ chứ không phải $d$. Ở bản 7B thì $n_{\text{kv}} = n_{\text{heads}} = 32$, nên $32 \times 128 = 4096 = d$ và bốn ma trận đều vuông:

$$4d^2 \times L = 4 \times 4096^2 \times 32 = 2\,147\,483\,648.$$

**Bước 2 — FFN.** Đây là chỗ dễ sai nhất. SwiGLU có **ba** ma trận, không phải hai:

$$3 \times d \times d_{\text{ff}} \times L = 3 \times 4096 \times 11\,008 \times 32 = 4\,328\,521\,728.$$

**Bước 3 — chuẩn hoá.** RMSNorm chỉ có $\gamma$, không có $\beta$. Mỗi lớp có hai cái (trước attention và trước FFN), cộng một cái cuối cùng trước lớp ra:

$$32 \times 2 \times 4096 + 4096 = 266\,240.$$

**Bước 4 — embedding và lớp ra.** Đề nói rõ **không** dùng chung trọng số, nên phải đếm hai lần:

$$2 \times 32\,000 \times 4096 = 262\,144\,000.$$

**Bước 5 — cộng lại.**

| Thành phần | Số tham số | Tỉ lệ |
|---|---|---|
| Attention | 2 147 483 648 | 31,9% |
| FFN (SwiGLU) | 4 328 521 728 | 64,2% |
| RMSNorm | 266 240 | 0,004% |
| Embedding + lớp ra | 262 144 000 | 3,9% |
| **Tổng** | **6 738 415 616** | |

Con số đã công bố cho Llama-2 7B là **6 738 415 616**. Khớp **chính xác tới từng tham số** — không phải xấp xỉ.

**Tỉ lệ $d_{\text{ff}}/d$.** $11\,008 / 4096 = 2{,}6875$, chứ không phải 4. Lý do:

$$\tfrac{8}{3} \times 4096 = 10\,922{,}67 \;\longrightarrow\; \text{làm tròn lên bội của } 256 \;=\; 11\,008.$$

Vì SwiGLU dùng ba ma trận thay vì hai, muốn giữ nguyên ngân sách tham số thì bề rộng ẩn phải giảm còn $2/3$: $\tfrac{2}{3} \times 4d = \tfrac{8}{3}d$. Đây là một quy ước kiến trúc để lại **dấu vết kiểm chứng được** trong tệp cấu hình của một mô hình thật, và biết đọc dấu vết ấy là bằng chứng hiểu chứ không phải nhớ.

> Chú ý: FFN chiếm **64%** tham số, gần đúng con số hai phần ba ở Mục 12.2. Với $d_{\text{ff}} = 4d$ thì tỉ lệ là đúng $8/12 = 2/3$; ở đây SwiGLU làm nó thành $3 \times 2{,}6875 / (4 + 3 \times 2{,}6875) = 66{,}8\%$ trong phạm vi khối, và 64,2% khi tính cả embedding.

## Bài 2
@meta chuong=12 | dang=Tính tay | kho=Cơ bản

**Bước 1 — lấy $N$ đúng.** Quy tắc $C \approx 6ND$ dùng số tham số **phi-embedding**. Từ Bài 1:

$$N = 2\,147\,483\,648 + 4\,328\,521\,728 = 6\,476\,005\,376 \approx 6{,}48 \text{ tỉ}.$$

Dùng tổng 6,74 tỉ thay vào đây là sai lệch 4% — nhỏ, nhưng nó cho thấy người trả lời không biết vì sao công thức loại embedding ra. Lý do: embedding chỉ là một phép tra bảng, không có phép nhân ma trận nào cho mỗi token.

**Bước 2 — tổng FLOP.**

$$C = 6ND = 6 \times 6{,}476 \times 10^{9} \times 2 \times 10^{12} = 7{,}7712 \times 10^{22} \text{ FLOP}.$$

**Bước 3 — quy ra giờ-GPU.** A100 cho 312 TFLOPS ở bf16, nhưng đó là con số **đỉnh**; phần thực dùng được là hiệu suất sử dụng:

$$t = \frac{7{,}7712 \times 10^{22}}{312 \times 10^{12} \times 0{,}376} = 6{,}624 \times 10^{8}\ \text{s} = 184\,011\ \text{giờ-GPU}.$$

**Bước 4 — đối chiếu.** Meta công bố **184 320** giờ-GPU cho bản 7B.

$$\text{lệch} = \frac{|184\,011 - 184\,320|}{184\,320} = 0{,}17\%.$$

**Vì sao kết quả này đáng chú ý.** Một công thức ước lượng gồm đúng một hằng số — số 6 — dự đoán ngân sách tính của một lần huấn luyện thật với sai lệch dưới hai phần nghìn. Nó nói rằng $C \approx 6ND$ không phải quy tắc ngón tay mà là một phép đếm gần như chính xác, và mọi thứ khác (chi phí attention, chuẩn hoá, tải dữ liệu) đều nằm trong phần nhiễu.

> Cẩn thận khi kể con số này: 37,6% là hiệu suất mà tôi **khớp ngược** từ số giờ đã công bố, không phải số Meta báo cáo. Điều bài toán chứng minh là công thức nhất quán với thực tế ở một mức hiệu suất hợp lý (30–50% là khoảng thường thấy), chứ không phải nó tiên đoán được hiệu suất.

## Bài 3
@meta chuong=9 | dang=Suy luận | kho=Trung bình

**Bước 1 — phương sai của tích vô hướng.** Viết $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$. Với mỗi số hạng, vì $q_i$ và $k_i$ độc lập và đều trung bình 0:

$$\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\,\mathbb{E}[k_i] = 0, \qquad \operatorname{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] = \mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = 1.$$

Các số hạng độc lập nên phương sai cộng được:

$$\operatorname{Var}(q \cdot k) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i) = d_k. \qquad \blacksquare$$

Đối chiếu số đo ở Mục 9.2: $d_k$ = 4, 16, 64, 256, 1024 cho phương sai đo được 3,63 / 15,99 / 64,66 / 251,67 / **1021,97**. Khớp với lý thuyết trong sai số lấy mẫu.

**Bước 2 — vì sao phương sai lớn làm gradient biến mất.** Đạo hàm của softmax là

$$\frac{\partial p_i}{\partial s_j} = p_i(\delta_{ij} - p_j).$$

Khi một $p_i$ tiến tới 1 và các $p_j$ còn lại tiến tới 0, **mọi** số hạng của ma trận Jacobi này tiến tới 0. Nói cách khác softmax bão hoà y như sigmoid: ở vùng gần one-hot nó gần như hằng số cục bộ nên không truyền gradient.

Và phương sai lớn đẩy đúng vào vùng đó: độ lệch chuẩn của điểm số là $\sqrt{d_k}$, nên với $d_k = 1024$ thì các điểm số trải rộng chừng $\pm 30$, và $e^{30}$ so với $e^{-30}$ là tỉ lệ $10^{26}$. Không có cách nào tránh being one-hot.

Số đo ở Mục 9.2 nói đúng điều đó bằng entropy (chuỗi dài 64, nên mức tối đa là $\ln 64 = 4{,}159$ nat):

| $d_k$ | Entropy khi **không** chia | Entropy khi **có** chia | Trọng số lớn nhất (không chia) |
|---|---|---|---|
| 4 | 2,8920 | 3,7391 | 0,258 |
| 16 | 1,3597 | 3,6854 | 0,584 |
| 64 | 0,5818 | 3,6803 | 0,791 |
| 256 | 0,2746 | 3,6923 | 0,894 |
| 1024 | **0,1179** | 3,6847 | **0,953** |

Cột thứ ba **đứng yên** ở khoảng 3,68 với mọi $d_k$: đó chính là điều phép chia được thiết kế để làm. Cột cuối cho thấy cái giá của việc không chia dưới dạng dễ thấy hơn — ở $d_k = 1024$, một khoá duy nhất chiếm 95,3% toàn bộ trọng số attention.

**Bước 3 — nếu chia cho $d_k$ thay vì $\sqrt{d_k}$.** Vì $\operatorname{Var}(q\cdot k) = d_k$, chia điểm số cho $d_k$ làm phương sai thành $d_k / d_k^2 = 1/d_k$, tức **tiến tới 0** khi $d_k$ lớn. Mọi điểm số co về gần nhau nên softmax cho phân phối **gần đều**: attention mất khả năng phân biệt vị trí, và mỗi token chỉ nhận trung bình của toàn chuỗi.

Đây là hỏng theo chiều ngược lại, và nó cho thấy $\sqrt{d_k}$ không phải một lựa chọn tuỳ ý trong nhiều lựa chọn "đủ nhỏ": nó là **giá trị duy nhất** làm phương sai độc lập với $d_k$. Chia ít hơn thì bão hoà, chia nhiều hơn thì mất phân giải.

## Bài 4
@meta chuong=7 | dang=Tính tay | kho=Cơ bản

**(a) Số lớp $3\times3$ bước nhảy 1 để phủ $224 \times 224$.** Với $k_i = 3$ và mọi $s_j = 1$, công thức $r_i = r_{i-1} + (k_i-1)\prod_{j<i}s_j$ thành $r_i = r_{i-1} + 2$, nên $r_L = 2L + 1$. Giải $2L + 1 \geq 224$:

$$L \geq 111{,}5 \;\Longrightarrow\; \boxed{112 \text{ lớp}}.$$

Đây là con số đáng nhớ, vì nó giải thích **vì sao CNN buộc phải có bước nhảy hoặc lớp gộp**: xếp 112 lớp chỉ để một nơ-ron nhìn thấy hết ảnh là vô lý về cả chi phí lẫn khả năng huấn luyện.

**(b) Trường tiếp nhận sau 10 lớp, cứ hai lớp một lần bước nhảy 2.** Điểm then chốt: bước nhảy **nhân** vào mọi mức tăng về sau, nên nó là cơ chế tăng theo hàm mũ chứ không phải tuyến tính. Với $s = (1,2,1,2,1,2,1,2,1,2)$:

| Lớp $i$ | $\prod_{j<i} s_j$ | Mức tăng | $r_i$ |
|---|---|---|---|
| 1 | 1 | +2 | 3 |
| 2 | 1 | +2 | 5 |
| 3 | 2 | +4 | 9 |
| 4 | 2 | +4 | 13 |
| 5 | 4 | +8 | 21 |
| 6 | 4 | +8 | 29 |
| 7 | 8 | +16 | 45 |
| 8 | 8 | +16 | 61 |
| 9 | 16 | +32 | 93 |
| 10 | 16 | +32 | **125** |

**10 lớp cho $r = 125$**, so với 112 lớp cần thiết khi không có bước nhảy. Cùng một mục tiêu, chi phí giảm hơn 11 lần. Đó là toàn bộ lý do các kiến trúc thật giảm độ phân giải dần theo chiều sâu.

**(c) Vì sao trường tiếp nhận hiệu dụng nhỏ hơn.** Vì không phải mọi điểm trong trường lý thuyết đều ảnh hưởng như nhau. Số **đường đi** từ một điểm đầu vào tới nơ-ron đích nhiều nhất ở tâm và ít dần ra biên — về bản chất là một phép tích chập lặp lại, mà tích chập lặp lại hội tụ về dạng Gauss theo định lý giới hạn trung tâm. Kết quả của Luo và cộng sự (2016): trường tiếp nhận hiệu dụng có dạng Gauss và đường kính chỉ tăng theo $O(\sqrt{L})$, chậm hơn nhiều so với $O(L)$ của con số lý thuyết.

Hệ quả thực tế: **không thể chỉ xếp thêm lớp** để mở rộng tầm nhìn. Phải dùng bước nhảy, gộp, tích chập giãn cách, hoặc attention.

## Bài 5
@meta chuong=2 | dang=Chẩn đoán | kho=Cơ bản

**(a) Huấn luyện 2%, kiểm định 18%, khoảng cách chưa khép.** Khoảng cách 16 điểm và chưa khép khi thêm dữ liệu là dấu hiệu kinh điển của **phương sai cao** (quá khớp).

Hai việc nên làm: **thêm dữ liệu hoặc tăng cường dữ liệu** (vì phương sai giảm theo cỡ mẫu — đây là cách chữa nhắm đúng cơ chế), và **tăng phạt chuẩn** (weight decay, dropout, hoặc dừng sớm — rẻ hơn nhiều và thường đủ).

**(b) Huấn luyện 24%, kiểm định 25%, cả hai đã phẳng.** Hai đường sát nhau và đã phẳng ở mức cao: **thiên lệch cao** (thiếu khớp). Thêm dữ liệu sẽ **không** giúp gì — đó là điểm quan trọng nhất của chẩn đoán này.

Hai việc nên làm: **dùng mô hình mạnh hơn** (thêm lớp, thêm bề rộng, hoặc đổi sang họ mô hình khác) và **thêm đặc trưng hoặc giảm phạt chuẩn**. Nếu đang dùng mô hình tuyến tính trên quan hệ phi tuyến thì mọi cố gắng về dữ liệu đều vô ích.

**(c) Kiểm định 12%, huấn luyện 15%.** Sai số kiểm định **thấp hơn** sai số huấn luyện. Đây không phải chẩn đoán về thiên lệch hay phương sai — nó là **dấu hiệu có lỗi trong quy trình**. Ba nguyên nhân thường gặp:

1. **Tập kiểm định dễ hơn tập huấn luyện** — chia dữ liệu không ngẫu nhiên, hoặc tập kiểm định thiếu các lát cắt khó.
2. **Phạt chuẩn chỉ bật khi huấn luyện.** Dropout và tăng cường dữ liệu làm sai số huấn luyện **đo được** cao hơn thực chất, vì mô hình đang bị làm khó. Đây là nguyên nhân phổ biến nhất và **không phải lỗi**.
3. **Sai số huấn luyện được tính trung bình trong lúc chạy epoch** — tức gồm cả các bước đầu khi mô hình còn kém — còn sai số kiểm định đo ở cuối epoch.

Việc cần làm: kiểm tra 2 và 3 trước, vì chúng là hiện tượng đo lường chứ không phải vấn đề mô hình. Nếu loại cả hai mà hiện tượng còn thì phải xem lại cách chia dữ liệu — rất có thể có rò rỉ.

## Bài 6
@meta chuong=4 | dang=Suy luận | kho=Trung bình

**Bước 1 — $g$ ánh xạ $[0,1]$ lên $[0,1]$.** Xét hai khoảng riêng.

Với $x \in [0, \tfrac12]$: $\text{ReLU}(x) = x$ và $\text{ReLU}(x - \tfrac12) = 0$, nên $g(x) = 2x$, chạy từ 0 tới 1.

Với $x \in [\tfrac12, 1]$: $g(x) = 2x - 4(x - \tfrac12) = 2 - 2x$, chạy từ 1 xuống 0.

Vậy $g$ là hàm "lều": lên tuyến tính từ $(0,0)$ tới $(\tfrac12, 1)$ rồi xuống tuyến tính tới $(1,0)$. Miền giá trị đúng bằng $[0,1]$, và **mỗi nhánh phủ trọn $[0,1]$**. Tính chất cuối này là chìa khoá.

**Bước 2 — hợp $k$ lần cho $2^k$ đoạn.** Chứng minh bằng quy nạp. Gọi $g^{(k)}$ là hợp $k$ lần và giả sử nó có $2^k$ đoạn tuyến tính, mỗi đoạn chạy hết biên độ từ 0 lên 1 hoặc từ 1 xuống 0.

Xét $g^{(k+1)} = g \circ g^{(k)}$. Trên mỗi đoạn của $g^{(k)}$, giá trị đi qua toàn bộ $[0,1]$, nên nó đi qua cả hai nhánh của $g$ — tức mỗi đoạn bị **chẻ làm hai**. Vậy $g^{(k+1)}$ có $2 \times 2^k = 2^{k+1}$ đoạn. Cơ sở quy nạp $k=1$ đã có ở Bước 1. $\blacksquare$

Chi phí: mỗi lần hợp thêm 2 đơn vị ReLU, nên $g^{(k)}$ dùng $2k$ đơn vị xếp thành $k$ lớp, tổng khoảng $6k$ tham số (mỗi đơn vị một trọng số vào, một độ lệch, một trọng số ra).

**Bước 3 — mạng một lớp ẩn cần bao nhiêu.** Một mạng một lớp ẩn với $w$ đơn vị ReLU là

$$f(x) = \sum_{j=1}^{w} c_j \,\text{ReLU}(a_j x + b_j) + dx + e,$$

tức một hàm tuyến tính từng khúc với **nhiều nhất $w$ điểm gãy** — mỗi đơn vị ReLU đóng góp đúng một điểm gãy, ở $x = -b_j/a_j$. Mà $g^{(k)}$ có $2^k$ đoạn, tức $2^k - 1$ điểm gãy bên trong. Vậy

$$w \geq 2^k - 1 \;\Longrightarrow\; \boxed{w = \Theta(2^k)}.$$

Số đo ở Mục 4.4 xác nhận chặn này là **chặt**: cho mạng một lớp lời giải bình phương tối thiểu tối ưu chính xác, sai số sụp đúng khi bề rộng chạm $2^k$, với mọi $k$ từ 2 tới 7.

**Kết luận.** $O(k)$ tham số so với $O(2^k)$ tham số cho cùng một hàm — **tuyến tính so với hàm mũ**. Ở $k = 7$ là 42 so với 386 tham số (9,2 lần); ở $k = 20$ sẽ là 120 so với hơn 3 triệu.

> Điều bài này **không** chứng minh, và cần nói rõ khi trình bày: rằng xuống dốc tìm được lời giải gọn ấy. Ta dựng nó bằng tay. Khoảng cách giữa "biểu diễn được" và "học được" là lý do Chương 6 tồn tại.

## Bài 7
@meta chuong=8 | dang=Suy luận | kho=Trung bình

**(a) RNN.** Gradient qua $T$ bước có dạng $\prod_{t} W_{hh}^\top \operatorname{diag}(\phi')$, và độ lớn của nó bị chi phối bởi $\rho^T$ với $\rho$ là bán kính phổ.

$$\rho = 0{,}9: \quad 0{,}9^{100} = 2{,}66 \times 10^{-5}.$$
$$\rho = 1{,}1: \quad 1{,}1^{100} = 1{,}38 \times 10^{4}.$$

Chênh 9 bậc độ lớn từ một thay đổi 20% ở $\rho$. Đây là bản chất của vấn đề: hàm $\rho \mapsto \rho^T$ cực kỳ nhạy, nên **không tồn tại vùng an toàn rộng** — chỉ $\rho$ rất gần 1 là dùng được, và huấn luyện thì làm $\rho$ trôi.

Số đo ở Mục 8.2 cho thấy đúng hình ảnh ấy trên mạng thật: hệ số 0,9 cho $3{,}2 \times 10^{-2}$, hệ số 1,0 cho 1,1, hệ số 1,2 cho 3,0.

**(b) LSTM.** Dọc đường ô nhớ, gradient là $\prod_{t} f_t$ — tích các **vô hướng** trong $(0,1)$, không phải tích các ma trận.

$$\text{độ lệch } = 1: \quad f = \sigma(1) = 0{,}7311, \qquad 0{,}7311^{100} = 2{,}48 \times 10^{-14}.$$
$$\text{độ lệch } = 4: \quad f = \sigma(4) = 0{,}9820, \qquad 0{,}9820^{100} = 0{,}163.$$

Chênh **12,8 bậc độ lớn** chỉ từ một giá trị khởi tạo. Số đo ở Mục 8.3 trên LSTM thật: $3{,}0 \times 10^{-15}$ so với $1{,}3 \times 10^{-1}$ — cùng bậc với tính tay; phần lệch còn lại là do các cổng khác cũng tham gia.

Điều rút ra, và đây là điều đáng nói trong phỏng vấn: **cơ chế của LSTM không phải bản thân cấu trúc cổng mà là việc giữ $f$ gần 1**. Cấu trúc cổng chỉ làm cho điều đó *khả thi* — nó biến một tích ma trận không điều khiển được thành một tích vô hướng điều khiển được. Thủ thuật đặt độ lệch cổng quên dương lúc khởi tạo (Jozefowicz và cộng sự, 2015) là cách khai thác nó, và chênh lệch gần 13 bậc độ lớn ở trên là giá của việc không làm.

**(c) Vì sao bùng nổ dễ chữa hơn tiêu biến.** Vì bùng nổ giữ nguyên **hướng** và chỉ làm sai **độ lớn**. Cắt ngưỡng gradient — co vector lại khi chuẩn vượt ngưỡng — bỏ phần độ lớn quá đáng và giữ lại toàn bộ thông tin về hướng đi. Một dòng mã, không mất gì.

Tiêu biến thì ngược: thông tin đã **mất**. Khi gradient là $10^{-14}$, nó đã chìm dưới độ phân giải của số thực dấu phẩy động và dưới mức nhiễu của SGD. Nhân nó lên $10^{14}$ không hồi phục được tín hiệu, chỉ khuếch đại nhiễu làm tròn. Phải chữa bằng **kiến trúc** — cổng, kết nối tắt, chuẩn hoá — tức phải ngăn từ đầu chứ không sửa được sau.

## Bài 8
@meta chuong=10 | dang=Tính tay | kho=Cơ bản

**Công thức.** Mỗi lớp giữ một $K$ và một $V$, mỗi cái có $n_{\text{kv}} \times d_{\text{head}}$ giá trị cho mỗi token mỗi chuỗi:

$$\text{bytes} = 2 \times L \times n_{\text{kv}} \times d_{\text{head}} \times T \times B \times \text{bpe}.$$

Số 2 đầu tiên là cho $K$ **và** $V$; `bpe` = 2 byte cho FP16.

**(a) MHA, 64 đầu KV.**

$$2 \times 80 \times 64 \times 128 \times 4096 \times 8 \times 2 = 85\,899\,345\,920 \text{ byte} = \boxed{80{,}00 \text{ GiB}}.$$

**(b) GQA, 8 đầu KV.** Mọi thứ giống hệt trừ $n_{\text{kv}}$:

$$\frac{80{,}00}{8} = \boxed{10{,}00 \text{ GiB}}.$$

**Tỉ lệ giảm: đúng 8 lần** — đúng bằng tỉ lệ nhóm $64/8$, vì $n_{\text{kv}}$ vào công thức một cách tuyến tính.

**Vì sao đây là quyết định về bộ nhớ lúc suy luận.** Ba lý do, và cả ba đều là con số:

1. **KV cache lớn hơn trọng số.** Mô hình 70B ở FP16 chiếm 140 GB trọng số. Với MHA, cache 80 GiB là hơn một nửa nữa — và đó chỉ ở batch 8, ngữ cảnh 4096. Ở batch 64 thì cache là 640 GiB, gấp 4,5 lần trọng số.
2. **Cache tăng tuyến tính theo cả $T$ và $B$**, còn trọng số thì cố định. Nên cache là thứ chặn **thông lượng**: nó quyết định phục vụ được bao nhiêu yêu cầu đồng thời. Với GQA, mỗi token mỗi chuỗi chỉ tốn 0,3125 MiB, nên 10 GiB đủ cho 32 768 token-chuỗi.
3. **Suy luận sinh token bị chặn bởi băng thông bộ nhớ, không bởi phép tính.** Mỗi token mới phải **đọc lại toàn bộ** cache. Cache nhỏ đi 8 lần thì số byte phải đọc cũng nhỏ đi 8 lần, nên độ trễ giảm theo.

Và cái giá: GQA **giảm** chất lượng một chút so với MHA, vì các đầu query buộc phải dùng chung cách "mô tả" vị trí. Bài báo GQA chọn số nhóm bằng số GPU trong một nút tensor-parallel, để mỗi GPU giữ đúng một nhóm — tức lựa chọn kiến trúc được quyết định bởi hình dạng phần cứng.

> Nói "GQA để tăng chất lượng" là một trong những câu tự tố cáo ở Mục 14.6. Nó là một **đánh đổi**, và biết đánh đổi cái gì lấy cái gì mới là hiểu.

## Bài 9
@meta chuong=1 | dang=Thiết kế | kho=Trung bình

Điểm chấm của bài này không phải chọn đúng mô hình mà là **lý do dựa trên giả thiết quy nạp**. Câu "vì XGBoost thắng nhiều thi Kaggle" không phải lý do.

**(a) Rời bỏ khách hàng: 40 đặc trưng bảng, 30 000 mẫu, cần giải thích được.**

Chọn: **gradient boosting cây** (XGBoost hoặc LightGBM), hoặc random forest nếu muốn dễ chỉnh hơn.

Lý do: dữ liệu bảng **không có cấu trúc không gian hay thời gian** để khai thác, nên các giả thiết của tích chập và attention đều vô dụng ở đây. Cái dữ liệu bảng *có* là các quan hệ ngưỡng và tương tác giữa các đặc trưng không cùng thang đo — đúng là thứ nhát cắt trực giao của cây biểu diễn tự nhiên. Thêm nữa, 30 000 mẫu với 40 đặc trưng là quá ít cho mạng nơ-ron ở chế độ mà nó mạnh. Về giải thích: cây cho SHAP values ở mức từng dự đoán, thứ bộ phận kinh doanh dùng được.

**(b) Khuyết tật trên dây chuyền: 3 000 ảnh có nhãn, bố cục rất giống nhau.**

Chọn: **CNN tiền huấn luyện, tinh chỉnh** (ResNet hoặc EfficientNet).

Lý do: ba giả thiết của tích chập — cục bộ, bất biến dịch chuyển, phân cấp — đúng với ảnh, nên nó tiết kiệm dữ liệu hơn ViT ở quy mô này rất nhiều. Với 3 000 ảnh thì huấn luyện từ đầu là không đủ dù dùng kiến trúc gì; tinh chỉnh từ trọng số ImageNet cho ta các đặc trưng cạnh và kết cấu miễn phí. Chi tiết "bố cục rất giống nhau" còn có nghĩa **không nên** tăng cường dữ liệu bằng phép lật hay xoay mạnh: nếu khuyết tật có ý nghĩa phụ thuộc vị trí thì bất biến dịch chuyển lại thành có hại, và đó là trường hợp nên bỏ lớp gộp toàn cục.

**(c) Trích xuất trường từ hợp đồng 50 trang.**

Chọn: **Transformer tiền huấn luyện có ngữ cảnh dài**, dạng encoder cho trích xuất trường (hoặc decoder nếu đầu ra là văn bản tự do).

Lý do: đây là bài toán mà **quan hệ phụ thuộc xa** là bản chất — điều khoản ở trang 40 tham chiếu định nghĩa ở trang 2. Attention cho đường đi $O(1)$ giữa hai vị trí bất kỳ, còn RNN phải đi qua hàng nghìn bước. Và tiền huấn luyện là điều kiện bắt buộc: không có nó thì không đủ dữ liệu hợp đồng có nhãn để học tiếng Việt hay tiếng Anh pháp lý từ đầu.

Chú ý về chi phí: 50 trang khoảng 25 000 token, mà $T/(6d) = 25\,000/(6 \times 768) \approx 5{,}4$ nếu dùng mô hình nhỏ — tức attention **thật sự** chi phối ở đây, không như phần lớn trường hợp. Đây là một trong những bài toán mà attention thưa hoặc chia đoạn có cửa sổ trượt là đáng cân nhắc.

**(d) Dự báo phụ tải điện theo giờ cho 12 tháng.**

Chọn: **mô hình chuỗi thời gian có cấu trúc tường minh** — gradient boosting trên các đặc trưng lịch và trễ, hoặc một mô hình có thành phần mùa vụ tường minh.

Lý do: phụ tải điện bị chi phối bởi **chu kỳ đã biết trước** — giờ trong ngày, ngày trong tuần, mùa trong năm, ngày lễ — cộng với nhiệt độ. Đó là thông tin ta có thể **đưa thẳng vào đặc trưng**, và đưa vào thì không phải học. Mạng nơ-ron sẽ phải học lại các chu kỳ ấy từ dữ liệu, tức tiêu dữ liệu cho việc đã biết đáp án.

Điều quan trọng hơn cả việc chọn mô hình: **cách chia dữ liệu**. Dự báo 12 tháng tới bắt buộc phải kiểm định theo thời gian (huấn luyện trên quá khứ, kiểm trên tương lai), không bao giờ chia ngẫu nhiên. Chia ngẫu nhiên trên chuỗi thời gian cho kết quả kiểm định tốt giả tạo vì mô hình được thấy cả hai phía của mỗi điểm cần dự báo.

## Bài 10
@meta chuong=9 | dang=Thí nghiệm | kho=Nâng cao

**Việc phải làm.** Trong phần (E), thay `softmax(scores)` bằng một hàm chuẩn hoá không bão hoà, ví dụ

```python
def relu_norm(s, axis=-1, eps=1e-9):
    """Chuan hoa l1 cua ReLU: khong bao hoa, vi no khong co ham exp."""
    w = np.maximum(s, 0.0)
    return w / (w.sum(axis=axis, keepdims=True) + eps)
```

rồi đo lại entropy của trọng số theo $d_k$ = 4, 16, 64, 256, 1024, **không** chia $\sqrt{d_k}$.

**Kết quả mong đợi, và lý do.** Hiện tượng bão hoà **biến mất**. Với softmax, tỉ lệ giữa hai trọng số là $e^{s_i - s_j}$ — **hàm mũ** của hiệu điểm số. Khi độ lệch chuẩn của điểm số là $\sqrt{d_k} \approx 32$, các hiệu cỡ vài chục cho tỉ lệ cỡ $e^{30} \approx 10^{13}$, nên một phần tử áp đảo tất cả.

Với chuẩn hoá $\ell_1$ của ReLU, tỉ lệ giữa hai trọng số là $s_i / s_j$ — **tuyến tính** theo điểm số. Nhân mọi điểm số với một hằng số $c$ thì tỉ lệ **không đổi chút nào**, nên phân phối trọng số hoàn toàn bất biến với thang đo của điểm số. Entropy đứng yên theo $d_k$ mà không cần chia gì cả.

**Câu trả lời cho câu hỏi.** Phép chia $\sqrt{d_k}$ chữa vấn đề của **softmax**, không phải của tích vô hướng.

Nói cho chính xác hơn: tích vô hướng làm phương sai tăng theo $d_k$ — đó là sự thật, và là điều Bài 3 chứng minh. Nhưng phương sai lớn chỉ **thành vấn đề** vì softmax là hàm mũ nên nhạy với thang đo tuyệt đối của đầu vào. Một hàm chuẩn hoá thuần nhất bậc 1 thì miễn nhiễm. Vậy nguyên nhân gần là tích vô hướng, nhưng cơ chế gây hại nằm ở softmax — và người ta chọn chữa ở phía tích vô hướng vì nó chỉ tốn một phép chia, còn đổi softmax thì đổi cả tính chất của mô hình.

**Vì sao vẫn dùng softmax dù nó có điểm yếu này.** Ba lý do đáng nêu:

1. **Nó luôn dương và trơn ở mọi nơi.** Chuẩn hoá ReLU có gradient bằng 0 cứng cho mọi vị trí bị cắt, nên các vị trí ấy không bao giờ được xét lại — giống hệt vấn đề nơ-ron ReLU chết, nhưng ở mức trọng số attention.
2. **Mẫu số không bao giờ bằng 0.** Với ReLU, nếu mọi điểm số đều âm thì tổng bằng 0 và phép chia vỡ. Phải thêm `eps`, và `eps` đó thành một siêu tham số ngầm.
3. **Tính mũ chính là thứ cho attention khả năng chọn dứt khoát khi cần.** Có những bài toán cần gần one-hot — ví dụ sao chép chính xác một token từ ngữ cảnh. Một hàm tuyến tính không làm được điều đó nếu không có điểm số cực lớn.

Nói cách khác, độ nhạy của softmax là một **tính năng** bị dùng sai thang đo, và $\sqrt{d_k}$ đưa thang đo về đúng chỗ. Đó là cách đọc đúng chú thích 4 của bài báo Transformer.

> Nếu chạy thí nghiệm này, ghi luôn entropy **có** chia $\sqrt{d_k}$ cho cả hai hàm để so bốn cột. Kết quả sẽ cho thấy chuẩn hoá ReLU bất biến với cả hai cách, còn softmax chỉ ổn định khi có chia — bảng ấy tự nó là câu trả lời.
