# Mô hình và kiến trúc: từ hồi quy tuyến tính tới Transformer

> **Giáo trình tự học, viết theo lối bài giảng.** Mỗi khái niệm đi theo trình tự *động cơ → định nghĩa → suy luận → ví dụ số → thí nghiệm kiểm chứng*, giống hai giáo trình còn lại trong repo này.
>
> **Về độ tin cậy của số liệu.** Tài liệu có hai loại con số, phân biệt rạch ròi. Loại thứ nhất **trích từ bài báo gốc** — mỗi lần dùng đều nói rõ nguồn, và nguồn nằm ở Chương 15. Loại thứ hai là **số liệu đo được trong chính repo này**, sinh bởi hai script trong `code/models/`, hạt giống cố định, chạy lại cho kết quả y hệt. Chỗ nào là quy ước của ngành thì được gọi đúng tên là quy ước.
>
> **Tài liệu này dạy cái gì.** Nó không dạy dùng framework. Nó trả lời câu hỏi: *mỗi họ mô hình giả định điều gì về thế giới, và giả định ấy bắt nó mạnh ở đâu, yếu ở đâu.* Nắm được điều đó thì đọc một kiến trúc mới chỉ còn là đọc xem nó đổi giả định nào.

---

## Mục lục

0. [Kiến thức nền và quy ước](#0-kiến-thức-nền-và-quy-ước)
1. [Bản đồ các họ mô hình](#1-bản-đồ-các-họ-mô-hình)
2. [Đánh đổi thiên lệch – phương sai](#2-đánh-đổi-thiên-lệch--phương-sai)
3. [Cây quyết định và học tập hợp](#3-cây-quyết-định-và-học-tập-hợp)
4. [Mạng nơ-ron nhiều lớp](#4-mạng-nơ-ron-nhiều-lớp)
5. [Lan truyền ngược và tối ưu hoá](#5-lan-truyền-ngược-và-tối-ưu-hoá)
6. [Khởi tạo, chuẩn hoá và kết nối tắt](#6-khởi-tạo-chuẩn-hoá-và-kết-nối-tắt)
7. [Mạng tích chập](#7-mạng-tích-chập)
8. [Mạng hồi quy và giới hạn của nó](#8-mạng-hồi-quy-và-giới-hạn-của-nó)
9. [Attention và Transformer](#9-attention-và-transformer)
10. [Transformer hiện đại](#10-transformer-hiện-đại)
11. [Sinh văn bản: các chiến lược giải mã](#11-sinh-văn-bản-các-chiến-lược-giải-mã)
12. [Đếm tham số, FLOP và bộ nhớ](#12-đếm-tham-số-flop-và-bộ-nhớ)
13. [Bài tập](#13-bài-tập)
14. [Ôn phỏng vấn](#14-ôn-phỏng-vấn)
15. [Tài liệu tham khảo](#15-tài-liệu-tham-khảo)

---

## 0. Kiến thức nền và quy ước

Người đọc cần biết: đại số tuyến tính (nhân ma trận, chuẩn vector), giải tích nhiều biến (đạo hàm riêng, quy tắc dây chuyền), xác suất cơ bản (kỳ vọng, phương sai), và Python/NumPy ở mức đọc hiểu.

| Ký hiệu | Ý nghĩa |
|---|---|
| $x \in \mathbb{R}^{d}$ | một mẫu đầu vào, $d$ chiều |
| $X \in \mathbb{R}^{n \times d}$ | ma trận dữ liệu: $n$ mẫu, mỗi mẫu $d$ chiều |
| $y$, $\hat{y}$ | nhãn thật và dự đoán |
| $W$, $b$ | ma trận trọng số và vector độ lệch |
| $\sigma$, $\phi$ | hàm kích hoạt |
| $L$ | hàm mất mát |
| $\eta$ | tốc độ học (learning rate) |
| $T$ | độ dài chuỗi (số token, số bước thời gian) |
| $d$ hay $d_{\text{model}}$ | số chiều biểu diễn của Transformer |
| $h$ | số đầu attention |
| $d_k = d/h$ | số chiều mỗi đầu |
| $d_{\text{ff}}$ | bề rộng lớp ẩn của FFN |
| $L_{\text{layers}}$ | số khối Transformer |
| $V$ | kích thước từ vựng |

**Một quy ước về chữ.** Tài liệu dùng *thiên lệch* cho bias (theo nghĩa thống kê) và *độ lệch* cho bias (theo nghĩa tham số cộng thêm trong $Wx + b$) — hai thứ khác hẳn nhau mà tiếng Anh gọi trùng tên. Dùng *lớp* cho layer, *khối* cho block, *đầu* cho head, *kết nối tắt* cho skip/residual connection, *trường tiếp nhận* cho receptive field, *hàm kích hoạt* cho activation function. Khi một thuật ngữ đã thành tên riêng (softmax, attention, Transformer, dropout, embedding) thì giữ nguyên. Toàn bộ đối chiếu Việt – Anh nằm ở trang **Từ điển thuật ngữ**.

**Quy ước về chiều.** Tài liệu viết $x W$ (vector hàng nhân ma trận) khi nói về mã nguồn thực tế, và $W x$ (ma trận nhân vector cột) khi viết công thức toán. Hai cách chỉ khác nhau ở phép chuyển vị; chỗ nào dễ nhầm thì có ghi rõ chiều.

---

## 1. Bản đồ các họ mô hình

### 1.1. Vì sao cần một bản đồ

Học máy có hàng trăm thuật toán. Nhưng chúng chỉ trả lời **bốn câu hỏi thiết kế**, và mọi thuật toán là một bộ câu trả lời cụ thể:

1. **Giả thiết gì về dạng của hàm cần học?** Tuyến tính? Bậc thang theo từng trục? Hợp thành nhiều tầng?
2. **Tham số hoá thế nào?** Cố định số tham số, hay để số tham số lớn dần theo dữ liệu?
3. **Tối ưu bằng cách nào?** Có nghiệm đóng? Tìm tham lam? Xuống dốc?
4. **Chống quá khớp bằng gì?** Phạt chuẩn? Cắt tỉa? Lấy trung bình nhiều mô hình? Dừng sớm?

Trả lời được bốn câu này cho một mô hình là hiểu mô hình ấy. Phần còn lại là chi tiết cài đặt.

![Hình 1](figs/models01_families.png)

**Hình 1.** Bốn họ lớn và vị trí của chúng trên cùng một trục. Càng sang phải, mô hình càng ít giả thiết về dạng hàm — nên càng linh hoạt, nhưng càng cần nhiều dữ liệu và càng khó diễn giải.

### 1.2. Bốn họ, bốn bộ câu trả lời

| | Mô hình tuyến tính | Cây và tập hợp | Mạng nơ-ron | Transformer |
|---|---|---|---|---|
| **Giả thiết dạng hàm** | tổ hợp tuyến tính của đặc trưng | hằng số trên các hộp chữ nhật song song trục | hợp thành nhiều tầng biến đổi affine + phi tuyến | như mạng nơ-ron, cộng thêm **quan hệ cặp giữa các vị trí** |
| **Số tham số** | $d + 1$, cố định | lớn dần theo dữ liệu (phi tham số) | rất nhiều, cố định trước | rất nhiều, cố định trước |
| **Tối ưu** | nghiệm đóng hoặc lồi | tham lam theo từng nhát cắt | xuống dốc + lan truyền ngược | như mạng nơ-ron |
| **Chống quá khớp** | phạt $\ell_1$, $\ell_2$ | giới hạn độ sâu, cắt tỉa, lấy trung bình | dừng sớm, dropout, phạt chuẩn, tăng cường dữ liệu | như mạng nơ-ron |
| **Mạnh nhất khi** | dữ liệu ít, cần giải thích, quan hệ gần tuyến tính | dữ liệu **bảng**, đặc trưng lẫn số và hạng mục, thang đo khác nhau | dữ liệu **thô có cấu trúc**: ảnh, âm thanh, văn bản | chuỗi dài, quan hệ xa, mọi phương thức dữ liệu |
| **Yếu nhất khi** | quan hệ phi tuyến | phải ngoại suy ra ngoài miền dữ liệu; dữ liệu thô | dữ liệu ít; dữ liệu bảng thuần | chuỗi rất dài (chi phí bậc hai) |

Một điểm hay bị bỏ qua và rất đáng nhớ: **trên dữ liệu bảng, cây tăng cường gradient vẫn thường thắng mạng nơ-ron sâu.** Đây không phải nghịch lý. Dữ liệu bảng có đặc trưng đã được con người thiết kế, thang đo lệch nhau, nhiều biến hạng mục và quan hệ dạng bậc thang — đúng loại cấu trúc mà nhát cắt song song trục mô tả tự nhiên, còn phép biến đổi affine thì không.

### 1.3. Cái gì thực sự quyết định

Với cùng một bài toán, thứ tự ảnh hưởng tới kết quả thường là:

$$\text{chất lượng dữ liệu và nhãn} \;>\; \text{đặc trưng} \;>\; \text{họ mô hình} \;>\; \text{siêu tham số} \;>\; \text{kiến trúc chi tiết}.$$

Tài liệu này nói về ba mức sau, nhưng nói trước điều ấy để bạn đặt chúng đúng chỗ. Hai giáo trình còn lại trong repo lo hai mức đầu (MLOps) và mức triển khai (Quantization).

---

## 2. Đánh đổi thiên lệch – phương sai

### 2.1. Vì sao chương này đứng đầu

Gần như mọi quyết định trong phần còn lại của tài liệu — thêm lớp hay bớt lớp, phạt chuẩn mạnh hay nhẹ, lấy trung bình bao nhiêu mô hình — đều là một điểm trên cùng một đường cong. Hiểu đường cong ấy rồi thì các quyết định sau không còn là ghi nhớ mà là suy luận.

### 2.2. Phân rã

Giả sử dữ liệu sinh ra từ $y = f(x) + \varepsilon$ với $\mathbb{E}[\varepsilon] = 0$ và $\operatorname{Var}(\varepsilon) = \sigma^2$. Ta huấn luyện được $\hat{f}$, phụ thuộc vào tập huấn luyện ngẫu nhiên. Tại một điểm $x_0$, lấy kỳ vọng trên **mọi tập huấn luyện có thể**:

$$\mathbb{E}\big[(y_0 - \hat{f}(x_0))^2\big] \;=\; \underbrace{\big(\mathbb{E}[\hat{f}(x_0)] - f(x_0)\big)^2}_{\text{thiên lệch}^2} \;+\; \underbrace{\operatorname{Var}\big(\hat{f}(x_0)\big)}_{\text{phương sai}} \;+\; \underbrace{\sigma^2}_{\text{nhiễu}}$$

Đọc ba số hạng bằng lời:

- **Thiên lệch** — sai lệch có hệ thống: mô hình *trung bình* đã sai so với sự thật. Nguyên nhân là giả thiết về dạng hàm quá hẹp. Gọi là **thiếu khớp** (underfitting).
- **Phương sai** — mức dao động: đổi tập huấn luyện một chút thì mô hình đổi nhiều. Nguyên nhân là mô hình đủ linh hoạt để bám cả nhiễu. Gọi là **quá khớp** (overfitting).
- **Nhiễu** — phần không thể giảm được bằng bất kỳ mô hình nào. Đây là **trần trên của mọi cố gắng**.

> Điểm cần nhớ khi trả lời phỏng vấn: thiên lệch và phương sai được định nghĩa **qua kỳ vọng trên các tập huấn luyện**, không phải trên các điểm dữ liệu. Nói "mô hình này phương sai cao vì dự đoán của nó dao động nhiều giữa các điểm" là nhầm định nghĩa.

### 2.3. Đo thật phân rã ấy

Phân rã trên không chỉ là hình vẽ minh hoạ — nó đo được. Thí nghiệm trong `code/models/experiments.py`: hàm thật là $f(x) = \sin(2{,}2x) + 0{,}35x$, nhiễu $\sigma = 0{,}35$, mỗi lần lấy 40 điểm huấn luyện, lặp lại 400 lần với 400 tập huấn luyện khác nhau, rồi khớp đa thức bậc 1 tới 14.

![Hình 2](figs/models02_biasvar.png)

**Hình 2.** Ba đường đo được trên cùng một dữ liệu. Thiên lệch² giảm đơn điệu theo độ phức tạp; phương sai tăng; tổng có một cực tiểu. Đường chấm ngang là $\sigma^2$ — mức sàn không thể vượt qua.

**Kết quả.** (Bảng đầy đủ nằm trong `experiments_output.txt`.)

| Bậc đa thức | Thiên lệch² | Phương sai | Nhiễu² | Tổng MSE |
|---|---|---|---|---|
| 1 | 0,4221 | 0,0231 | 0,1225 | 0,5677 |
| 3 | 0,4098 | 0,0857 | 0,1225 | 0,6180 |
| **5** | **0,0836** | **0,0767** | 0,1225 | **0,2828** |
| 12 | 0,0276 | 52,2156 | 0,1225 | 52,3657 |
| 14 | 35,1402 | 7392,2751 | 0,1225 | 7427,5379 |

Ba điều đọc ra được, và cả ba đều dùng được:

1. **Bậc 1 sai vì thiên lệch**: nó chiếm 74% tổng MSE. Thêm dữ liệu sẽ **không** cứu được — đường thẳng không bao giờ thành hình sin. Phải đổi mô hình.
2. **Bậc cao sai vì phương sai**: ở bậc 12 phương sai đã gấp 1892 lần thiên lệch², và ở bậc 14 nó chiếm gần như toàn bộ MSE. Thêm dữ liệu **sẽ** cứu được, vì phương sai giảm khi cỡ mẫu tăng.
3. **Vì vậy chẩn đoán đúng quyết định hành động đúng.** Đây là giá trị thực tế của phân rã: nó phân biệt *"cần mô hình mạnh hơn"* với *"cần nhiều dữ liệu hơn"* — hai kết luận trái ngược, và chọn sai thì mất hàng tháng.

### 2.4. Cách chẩn đoán trong việc thật

Không cần chạy 400 tập huấn luyện. Chỉ cần vẽ **đường cong học** (learning curve): sai số huấn luyện và sai số kiểm định theo cỡ tập huấn luyện.

| Dấu hiệu | Chẩn đoán | Việc nên làm |
|---|---|---|
| Sai số huấn luyện **cao**, sai số kiểm định cao, hai đường **sát nhau** và đã phẳng | thiên lệch cao | mô hình mạnh hơn, thêm đặc trưng, bớt phạt chuẩn, huấn luyện lâu hơn |
| Sai số huấn luyện **thấp**, sai số kiểm định cao, **khoảng cách lớn** và chưa khép | phương sai cao | thêm dữ liệu, tăng phạt chuẩn, tăng cường dữ liệu, mô hình đơn giản hơn, dừng sớm |
| Cả hai đều cao và **vẫn đang giảm** | huấn luyện chưa xong | cứ huấn luyện tiếp |
| Sai số kiểm định thấp hơn sai số huấn luyện | thường là **lỗi** | nghi rò rỉ dữ liệu, hoặc dropout/augmentation chỉ bật lúc huấn luyện |

Hàng cuối đáng nhớ: nó trông như tin mừng nhưng gần như luôn là lỗi. Trường hợp vô hại duy nhất là khi dropout hoặc tăng cường dữ liệu chỉ bật ở lượt huấn luyện, làm sai số huấn luyện bị đo trên bài toán khó hơn.

### 2.5. Chú thích về "double descent"

Bức tranh chữ U ở trên là bức tranh cổ điển và nó đúng trong chế độ *thiếu tham số*. Với mô hình rất lớn, người ta quan sát được hiện tượng **double descent**: khi số tham số vượt qua điểm nội suy hoàn hảo tập huấn luyện, sai số kiểm định **giảm trở lại** thay vì tiếp tục tăng.

Điều này **không bác bỏ** phân rã ở Mục 2.2 — phân rã ấy là một đẳng thức toán học, luôn đúng. Nó chỉ nói rằng trong chế độ quá tham số, phương sai không đơn điệu tăng theo số tham số như trực giác cổ điển. Lý do là ở chế độ ấy, thuật toán tối ưu (xuống dốc) **ngầm chọn nghiệm có chuẩn nhỏ nhất** trong vô số nghiệm khớp hoàn hảo, tức nó tự phạt chuẩn.

Tài liệu này nói ra điều đó để bạn không bị hớ khi được hỏi *"mô hình càng lớn càng quá khớp, đúng không?"*. Câu trả lời đúng: **đúng trong chế độ cổ điển, và không còn đúng hiển nhiên ở chế độ quá tham số** — đó là lý do các mô hình hàng tỉ tham số vẫn tổng quát hoá được.

---

## 3. Cây quyết định và học tập hợp

### 3.1. Một cây làm gì

Cây quyết định chia không gian đặc trưng bằng các nhát cắt **vuông góc với trục**, rồi dự đoán một hằng số trên mỗi hộp.

![Hình 3](figs/models03_tree.png)

**Hình 3.** Trái: cây chia mặt phẳng bằng các nhát cắt song song trục. Phải: cùng một mô hình viết dưới dạng cây nhị phân. Đây là hai cách nhìn cùng một vật.

Từ hình này rút ra ngay ba tính chất, và cả ba đều là hệ quả trực tiếp của "nhát cắt song song trục":

1. **Bất biến với phép biến đổi đơn điệu từng đặc trưng.** Lấy log một đặc trưng không đổi cây, vì thứ tự không đổi. Đây là lý do **cây không cần chuẩn hoá đặc trưng** — một điểm hay bị hỏi.
2. **Không ngoại suy được.** Ngoài miền dữ liệu huấn luyện, cây trả về hằng số của hộp ngoài cùng. Dự đoán giá nhà cho diện tích lớn hơn mọi căn từng thấy sẽ cho đúng giá của căn lớn nhất đã thấy.
3. **Khó biểu diễn quan hệ chéo.** Một biên quyết định $x_1 + x_2 > 1$ phải được xấp xỉ bằng hình bậc thang, tốn rất nhiều nhát cắt.

### 3.2. Chọn nhát cắt thế nào

Tại mỗi nút, thuật toán duyệt mọi đặc trưng và mọi ngưỡng, chọn nhát cắt làm **giảm độ tạp nhất**. Với phân loại, hai thước đo độ tạp phổ biến:

$$\text{Gini}(p) = 1 - \sum_{c} p_c^2, \qquad \text{Entropy}(p) = -\sum_{c} p_c \log p_c.$$

Với hồi quy thì dùng phương sai trong nút. Độ lợi của một nhát cắt là độ tạp của nút cha trừ trung bình có trọng số của hai nút con.

> Gini và entropy cho cây gần như giống nhau trong thực tế. Gini rẻ hơn vì không có log. Đừng mất thời gian chọn giữa hai cái; chọn độ sâu và số mẫu tối thiểu mỗi lá mới là việc đáng làm.

**Điểm quan trọng:** thuật toán này **tham lam** — nó chọn nhát cắt tốt nhất ở bước hiện tại mà không nhìn xa. Cây tối ưu toàn cục là bài toán NP-khó. Vì vậy một cây đơn lẻ vừa **phương sai cao** (đổi vài điểm dữ liệu là đổi nhát cắt đầu, đổi luôn cả cây) vừa **không đảm bảo tối ưu**.

### 3.3. Hai cách ghép nhiều mô hình lại

Chính vì phương sai cao mà cây trở thành nguyên liệu lý tưởng cho học tập hợp. Có **hai** cách ghép, và chúng tấn công **hai thành phần khác nhau** của phân rã ở Chương 2.

**Bagging** (bootstrap aggregating): huấn luyện $B$ mô hình trên $B$ mẫu bootstrap **độc lập**, rồi lấy trung bình. Lập luận rất gọn: nếu $B$ mô hình có cùng phương sai $\sigma^2$ và tương quan từng cặp $\rho$, thì trung bình của chúng có phương sai

$$\rho\,\sigma^2 + \frac{1-\rho}{B}\sigma^2.$$

Cho $B \to \infty$, số hạng thứ hai biến mất và **chỉ còn $\rho \sigma^2$**. Đây là toàn bộ lý thuyết của bagging trong một dòng, và nó nói luôn điều phải làm: muốn giảm tiếp thì phải **giảm $\rho$**, tức làm các mô hình bớt giống nhau. Đó đúng là điều **rừng ngẫu nhiên** (random forest) thêm vào so với bagging thường: ở mỗi nút chỉ xét một tập con ngẫu nhiên các đặc trưng, nên các cây bớt tương quan.

**Boosting**: huấn luyện tuần tự, mỗi mô hình mới sửa phần **dư** của tổng các mô hình trước:

$$F_m(x) = F_{m-1}(x) + \eta\, h_m(x), \qquad h_m \approx -\frac{\partial L}{\partial F}\bigg|_{F = F_{m-1}}.$$

Đây chính là **xuống dốc trong không gian hàm**: mỗi cây mới xấp xỉ gradient âm của hàm mất mát. Vì mỗi cây chỉ phải sửa phần còn thiếu, cây con có thể rất nông (thường sâu 2–6) — và chính vì thế boosting **giảm thiên lệch**.

### 3.4. Đo thật xem cái nào cắt vào đâu

Thí nghiệm trong `code/models/experiments.py`: cùng một hàm thật và cùng mức nhiễu như Chương 2, đo thiên lệch² và phương sai của ba mô hình trên 60 tập huấn luyện.

![Hình 4](figs/models04_ensemble.png)

**Hình 4.** Cùng một bài toán, ba mô hình. Cột trái là thiên lệch², cột phải là phương sai.

| Mô hình | Thiên lệch² | Phương sai | Tổng |
|---|---|---|---|
| Một cây (sâu 8) | 0,0031 | 0,1089 | 0,1120 |
| Bagging 100 cây | 0,0016 | **0,0534** | 0,0550 |
| Boosting 100 gốc nông | **0,0032** | 0,0371 | 0,0403 |

Đọc bảng:

- **Bagging cắt phương sai đúng một nửa** (0,1089 → 0,0534) và gần như không đụng tới thiên lệch. Đúng như lý thuyết: lấy trung bình các mô hình gần như không thiên lệch.
- **Boosting đạt tổng thấp nhất** bằng cách xuất phát từ các cây rất nông — thiên lệch ban đầu cao — rồi cộng dồn để hạ nó xuống, đồng thời giữ phương sai thấp vì mỗi cây rất yếu.
- Cây đơn lẻ ở đây có thiên lệch rất thấp (0,0031) vì sâu 8 đã đủ linh hoạt; **phương sai lớn gấp 35 lần thiên lệch²**, nên toàn bộ vấn đề của nó là phương sai — đó chính xác là thứ bagging sinh ra để chữa.

> Câu hỏi phỏng vấn hay đi kèm: *"random forest hay gradient boosting?"* Câu trả lời tốt không phải tên mô hình mà là: **random forest dễ chỉnh hơn và khó hỏng hơn vì mỗi cây độc lập nên thêm cây không bao giờ làm tệ đi; gradient boosting thường chính xác hơn nhưng nhạy với tốc độ học và số vòng, và thêm vòng thì CÓ THỂ quá khớp.** Nói được tính "thêm cây không hại" của bagging là dấu hiệu hiểu bản chất.

### 3.5. Những chỗ hay nhầm

| Phát biểu | Thực tế |
|---|---|
| "Cây cần chuẩn hoá đặc trưng." | Không. Cây bất biến với mọi phép biến đổi đơn điệu từng đặc trưng. |
| "Random forest không quá khớp." | Thêm **cây** thì không, nhưng để cây quá **sâu** trên dữ liệu nhiễu thì vẫn quá khớp. |
| "Độ quan trọng đặc trưng của cây là đáng tin." | Độ quan trọng theo độ giảm độ tạp **thiên vị đặc trưng nhiều giá trị**. Dùng permutation importance hoặc SHAP thì đáng tin hơn. |
| "Boosting chỉ dùng cây được." | Bất kỳ học máy yếu nào cũng được; cây chỉ là lựa chọn phổ biến vì rẻ và xử lý được đặc trưng hỗn hợp. |
| "XGBoost là một thuật toán mới." | Nó là một **cài đặt** rất tối ưu của gradient boosting, cộng thêm phạt chuẩn trên cấu trúc cây và xử lý giá trị thiếu. |

---

## 4. Mạng nơ-ron nhiều lớp

### 4.1. Từ hồi quy tuyến tính tới mạng

Một mô hình tuyến tính là $\hat{y} = Wx + b$. Chồng hai mô hình tuyến tính lên nhau không cho gì mới:

$$W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2)$$

vẫn là một hàm tuyến tính. **Vì vậy phi tuyến không phải là phụ gia — nó là điều kiện để "nhiều lớp" có nghĩa.** Đây là câu hỏi phỏng vấn cơ bản nhất về mạng nơ-ron, và câu trả lời đúng là một dòng đại số.

Một lớp ẩn là:

$$h = \phi(W_1 x + b_1), \qquad \hat{y} = W_2 h + b_2$$

với $\phi$ áp dụng theo từng phần tử.

### 4.2. Các hàm kích hoạt và vì sao chúng thay nhau

| Hàm | Công thức | Ưu | Nhược |
|---|---|---|---|
| **Sigmoid** | $1/(1+e^{-z})$ | đầu ra trong $(0,1)$, đọc được như xác suất | **bão hoà hai đầu** → gradient $\approx 0$; đầu ra không quanh 0 |
| **Tanh** | $\tanh z$ | quanh 0, tốt hơn sigmoid | vẫn bão hoà |
| **ReLU** | $\max(0, z)$ | không bão hoà phía dương, rẻ, gradient đúng bằng 1 | **nơ-ron chết**: rơi vào vùng âm là gradient bằng 0 mãi |
| **Leaky ReLU** | $\max(\alpha z, z)$ | chữa nơ-ron chết | thêm một siêu tham số |
| **GELU** | $z\,\Phi(z)$ | trơn, hoạt động tốt trong Transformer | đắt hơn ReLU một chút |
| **SiLU / Swish** | $z\,\sigma(z)$ | trơn, không đơn điệu | như trên |

Ba điều đáng nói:

- **Đạo hàm của ReLU đúng bằng 1 ở phía dương** là lý do chính khiến nó thay được sigmoid trong mạng sâu: chuỗi nhân các đạo hàm không co lại. Sigmoid có đạo hàm tối đa $0{,}25$, nên qua 10 lớp là $0{,}25^{10} \approx 10^{-6}$ — đó chính là gradient tiêu biến ở Chương 6.
- **Nơ-ron chết** không phải lỗi lý thuyết mà là hiện tượng thật: nếu tốc độ học quá lớn, một bước cập nhật có thể đẩy độ lệch xuống đủ thấp để nơ-ron không bao giờ kích hoạt lại với bất kỳ đầu vào nào.
- **Ở đầu ra thì khác**: dùng sigmoid cho phân loại nhị phân, softmax cho đa lớp, và **không dùng gì** cho hồi quy. Trộn lẫn hàm kích hoạt của lớp ẩn với hàm của lớp đầu ra là lỗi hay gặp.

### 4.3. Định lý xấp xỉ phổ quát, và vì sao nó ít hữu ích hơn ta tưởng

> **Định lý xấp xỉ phổ quát.** Một mạng **một lớp ẩn** với hàm kích hoạt không đa thức, đủ rộng, xấp xỉ được mọi hàm liên tục trên một tập compact tới độ chính xác tuỳ ý.

Nghe như mọi chuyện đã xong. Nhưng định lý chỉ nói **tồn tại**, và không nói ba điều quyết định:

1. Cần **bao nhiêu** nơ-ron — và câu trả lời có thể là số mũ theo số chiều.
2. Có **tìm ra** được bộ trọng số ấy bằng xuống dốc hay không.
3. Với **bao nhiêu dữ liệu** thì tìm được mà không quá khớp.

Điều thực sự quan trọng là **độ sâu mua được sự gọn gàng**: có những hàm mà mạng sâu biểu diễn với số nơ-ron tăng đa thức, trong khi mạng một lớp cần số nơ-ron tăng theo hàm mũ.

### 4.4. Đo thật: sâu hay rộng

Câu "độ sâu mua được sự gọn gàng" thường được nói như một niềm tin. Thực ra nó **chứng minh được bằng kiến tạo**, và cách chứng minh ấy đẹp vì không phụ thuộc chút nào vào việc huấn luyện có thành công hay không.

Xuất phát từ **hàm lều**, dùng đúng hai đơn vị ReLU:

$$g(x) = 2\,\text{ReLU}(x) - 4\,\text{ReLU}\!\left(x - \tfrac12\right).$$

Hàm này ánh xạ $[0,1]$ lên $[0,1]$: nó đi lên từ $g(0)=0$ tới $g(1/2)=1$ rồi xuống lại $g(1)=0$. Bây giờ **hợp nó với chính nó** $k$ lần. Vì mỗi nhánh của $g$ phủ trọn $[0,1]$, mỗi lần hợp **gấp đôi số đoạn tuyến tính**: hợp $k$ lần cho một sóng răng cưa đúng $2^k$ đoạn, dùng $2k$ đơn vị ReLU xếp thành $k$ lớp.

Câu hỏi: mạng **một lớp ẩn** cần bao nhiêu đơn vị để khớp cùng hàm ấy? Câu trả lời đo được — đặt các điểm gãy đều trên $(0,1)$ rồi giải bình phương tối thiểu cho lớp ra, tức là cho mạng một lớp **lời giải tối ưu chính xác** thay vì bắt nó tự tìm bằng xuống dốc:

![Hình 5](figs/models05_depth.png)

**Hình 5.** Trái: sai số nhỏ nhất mà mạng một lớp ẩn đạt được, theo bề rộng, cho từng độ sâu $k$ (hai trục log). Mỗi đường **sụp thẳng đứng** đúng khi bề rộng chạm $2^k$. Phải: số tham số cần thiết — tuyến tính theo $k$ so với hàm mũ theo $k$.

| $k$ lớp | Số đoạn $2^k$ | Tham số mạng sâu | Bề rộng một lớp cần | Tham số một lớp | Gấp |
|---|---|---|---|---|---|
| 2 | 4 | 12 | 4 | 14 | 1,2× |
| 3 | 8 | 18 | 8 | 26 | 1,4× |
| 4 | 16 | 24 | 16 | 50 | 2,1× |
| 5 | 32 | 30 | 32 | 98 | 3,3× |
| 6 | 64 | 36 | 64 | 194 | 5,4× |
| 7 | 128 | 42 | 128 | 386 | **9,2×** |

Bề rộng cần thiết **đúng bằng $2^k$**, không xấp xỉ. Lý do thì hiển nhiên khi đã thấy: một đơn vị ReLU tạo được đúng một điểm gãy, nên muốn có $2^k$ đoạn thì phải có $2^k$ điểm gãy, nên phải có $2^k$ đơn vị. Còn mạng sâu chỉ cần $6k$ tham số. Cột cuối là tỉ lệ giữa hai con số ấy, và nó **tăng không giới hạn**: ở $k = 7$ đã là 9,2 lần, ở $k = 20$ sẽ là hơn 26 000 lần.

Ba điều cần nói rõ về cách đọc kết quả này:

1. **Đây là chặn trên cho mạng sâu và gần chặn dưới cho mạng một lớp.** Mạng sâu được cho trọng số **tường minh** — không huấn luyện gì cả. Mạng một lớp được cho lời giải bình phương tối thiểu **tối ưu chính xác** với các điểm gãy ở vị trí tốt nhất cho hàm tuần hoàn này. Nên khoảng cách đo được không phải do bên nào tối ưu kém.
2. **Bảng nói về khả năng biểu diễn, không nói về khả năng học.** Rằng mạng sâu *biểu diễn được* hàm ấy gọn gàng không có nghĩa xuống dốc *tìm ra được* bộ trọng số ấy. Thực tế thì huấn luyện mạng 7 lớp với 2 đơn vị mỗi lớp trên hàm này gần như luôn thất bại. Đó là lý do Chương 6 tồn tại.
3. **Không suy ra được "cứ sâu hơn là tốt hơn".** Hàm răng cưa được chọn có chủ đích vì nó là trường hợp *tốt nhất* cho độ sâu — nó có cấu trúc tự lặp mà phép hợp khai thác được. Với dữ liệu bảng, thêm lớp thường không giúp gì (Mục 1.2).

> Cách nói gọn khi phỏng vấn: *"Định lý xấp xỉ phổ quát nói một lớp là đủ, nhưng nó chỉ nói tồn tại. Có những hàm mà mạng $k$ lớp cần $O(k)$ tham số còn mạng một lớp cần $O(2^k)$ — cấu trúc răng cưa dựng từ hàm lều là ví dụ đo được. Nên độ sâu không mua thêm sức biểu diễn, nó mua sự gọn gàng."*

### 4.5. Kiến trúc MLP trong thực tế

Với dữ liệu không có cấu trúc không gian hay thời gian, công thức mặc định đáng tin:

- **Độ sâu**: 2–4 lớp ẩn. Sâu hơn cần kết nối tắt và chuẩn hoá (Chương 6).
- **Bề rộng**: thường giữ bằng nhau giữa các lớp, hoặc thu hẹp dần. Bề rộng đáng chỉnh hơn độ sâu.
- **Kích hoạt**: ReLU hoặc GELU.
- **Chuẩn hoá**: LayerNorm hoặc BatchNorm sau biến đổi tuyến tính, trước kích hoạt (hoặc sau — xem Mục 6.4).
- **Chống quá khớp**: dropout 0,1–0,3 và **dừng sớm**. Dừng sớm là biện pháp rẻ nhất và hiệu quả nhất; đáng làm trước mọi thứ khác.

---

## 5. Lan truyền ngược và tối ưu hoá

### 5.1. Lan truyền ngược không phải một thuật toán học

Đây là hiểu lầm phổ biến nhất về lan truyền ngược, và sửa nó là bước đầu để hiểu đúng:

> **Lan truyền ngược chỉ là cách tính gradient một cách hiệu quả.** Nó không quyết định cập nhật tham số thế nào — việc đó là của thuật toán tối ưu (SGD, Adam…). Hai thứ độc lập nhau.

Ý tưởng nằm gọn trong quy tắc dây chuyền. Với chuỗi hàm hợp $L = \ell(f_n(\cdots f_1(x)))$, đạo hàm theo tham số của lớp $i$ là một tích các đạo hàm địa phương. Điều thông minh duy nhất là **thứ tự nhân**.

![Hình 6](figs/models06_backprop.png)

**Hình 6.** Lượt xuôi tính và **ghi lại** mọi giá trị trung gian; lượt ngược nhân dần các đạo hàm địa phương theo chiều ngược, dùng lại đúng những giá trị đã ghi.

### 5.2. Vì sao phải nhân từ phải sang trái

Giả sử mạng biến $\mathbb{R}^{d} \to \mathbb{R}^{m} \to \mathbb{R}^{1}$. Gradient là tích hai ma trận Jacobi:

$$\frac{\partial L}{\partial x} \;=\; \underbrace{\frac{\partial L}{\partial h}}_{1 \times m} \; \underbrace{\frac{\partial h}{\partial x}}_{m \times d}$$

- Nhân **từ trái sang phải** (chiều ngược, tức lan truyền ngược): mỗi bước là *vector × ma trận*, chi phí $O(md)$.
- Nhân **từ phải sang trái** (chiều xuôi, tức lan truyền xuôi): mỗi bước là *ma trận × ma trận*, chi phí $O(md^2)$.

Với $d$ lớn — và trong học sâu $d$ là hàng triệu — khác biệt là một bậc độ lớn theo $d$. Đây là toàn bộ lý do kỹ thuật khiến ta huấn luyện ngược chứ không xuôi:

> **Lan truyền ngược rẻ khi có ÍT đầu ra và NHIỀU tham số.** Hàm mất mát là một số vô hướng, nên trường hợp này đúng tuyệt đối.

Ngược lại, nếu cần đạo hàm của **nhiều** đầu ra theo **một** đầu vào thì chiều xuôi mới rẻ. Đó là lý do có hai chế độ vi phân tự động: *reverse mode* và *forward mode*.

### 5.3. Cái giá: bộ nhớ

Lan truyền ngược phải **giữ lại các giá trị trung gian của lượt xuôi** để dùng ở lượt ngược. Đây là khoản chi phí bị đánh giá thấp nhất khi huấn luyện mô hình lớn:

$$\text{bộ nhớ kích hoạt} \;\approx\; \text{số lớp} \times \text{kích thước lô} \times \text{độ dài chuỗi} \times d \times \text{số byte}.$$

Nó **không** tỉ lệ với số tham số mà tỉ lệ với **kích thước lô và độ dài chuỗi**. Đó là vì sao hết bộ nhớ khi huấn luyện thường được chữa bằng giảm kích thước lô, chứ không bằng giảm mô hình.

Hai kỹ thuật đổi thời gian lấy bộ nhớ:

- **Gradient checkpointing**: chỉ giữ kích hoạt ở một số lớp mốc, các lớp còn lại **tính lại** khi cần ở lượt ngược. Với $L$ lớp và mốc mỗi $\sqrt{L}$ lớp, bộ nhớ giảm từ $O(L)$ xuống $O(\sqrt{L})$, đổi lại khoảng **một lần lượt xuôi thêm**, tức tăng khoảng 30% thời gian.
- **Tích luỹ gradient**: chạy nhiều lô nhỏ, cộng dồn gradient rồi mới cập nhật một lần. Cho hiệu ứng của lô lớn với bộ nhớ của lô nhỏ, nhưng **không** nhanh hơn.

### 5.4. Xuống dốc và các biến thể

Cập nhật cơ bản:

$$\theta_{t+1} = \theta_t - \eta\, g_t, \qquad g_t = \nabla_\theta L(\theta_t).$$

**Xuống dốc ngẫu nhiên theo lô nhỏ (mini-batch SGD)** ước lượng $g_t$ trên một lô nhỏ. Nhiễu của phép ước lượng này không chỉ là cái giá phải trả — nó còn là một dạng phạt chuẩn ngầm, giúp thoát khỏi các cực tiểu hẹp.

**Momentum** cộng dồn hướng đi:

$$v_{t+1} = \beta v_t + g_t, \qquad \theta_{t+1} = \theta_t - \eta\, v_{t+1}.$$

Trực giác đúng: trong một khe hẹp và dài, gradient dao động mạnh theo chiều ngang khe và yếu theo chiều dọc khe. Lấy trung bình trượt **triệt tiêu dao động ngang** và **cộng dồn phần dọc**. Với $\beta = 0{,}9$, bước đi hiệu dụng lớn gấp khoảng $1/(1-\beta) = 10$ lần.

**Adam** thêm một ý nữa: chia mỗi toạ độ cho căn bậc hai của trung bình trượt bình phương gradient, tức **mỗi tham số có tốc độ học riêng**:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}, \qquad \theta_{t+1} = \theta_t - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.$$

**Hai phép chia cho $1-\beta^t$ là hiệu chỉnh thiên lệch**, và đây là chi tiết hay bị hỏi. Lý do: $m_0 = 0$, nên ở bước đầu $m_1 = (1-\beta_1) g_1$ — nhỏ hơn $g_1$ khoảng 10 lần với $\beta_1 = 0{,}9$. Không hiệu chỉnh thì các bước đầu tiên nhỏ một cách giả tạo. Chia cho $1-\beta_1^t$ đúng bằng việc bù lại phần khuyết đó, và ảnh hưởng của nó tắt dần khi $t$ lớn.

| | SGD + momentum | Adam |
|---|---|---|
| Siêu tham số nhạy | rất nhạy với $\eta$ | ít nhạy hơn nhiều |
| Hội tụ ban đầu | chậm hơn | nhanh hơn |
| Tổng quát hoá | thường **tốt hơn** trên thị giác máy tính | thường kém hơn một chút ở đó |
| Bộ nhớ trạng thái | 1× số tham số | **2×** số tham số |
| Mặc định hợp lý | thị giác, khi có thời gian chỉnh | NLP, Transformer, khi muốn chạy được ngay |

Hàng "bộ nhớ trạng thái" rất đáng nhớ khi nói về huấn luyện mô hình lớn: Adam giữ **hai** trạng thái cho mỗi tham số, nên bộ nhớ tối ưu hoá gấp đôi bộ nhớ trọng số. Với mô hình 7 tỉ tham số ở FP32, riêng trạng thái Adam đã là $7 \times 10^9 \times 2 \times 4 = 56$ GB.

**AdamW** tách phần suy giảm trọng số ra khỏi gradient thay vì cộng vào nó. Khác biệt không phải chuyện nhỏ: với Adam thường, phạt $\ell_2$ bị chia cho $\sqrt{\hat{v}}$ nên tham số có gradient lớn bị phạt **ít** hơn — ngược hẳn với ý định. AdamW là mặc định cho Transformer vì lý do đó.

### 5.5. Lịch tốc độ học

Tốc độ học là siêu tham số quan trọng nhất, và **lịch** của nó gần như cũng quan trọng ngang thế.

- **Warmup** — tăng dần $\eta$ từ 0 trong vài trăm tới vài nghìn bước đầu. Lý do với Adam: ở các bước đầu, $\hat{v}_t$ được ước lượng từ rất ít mẫu nên rất nhiễu, khiến bước đi có thể lớn bất thường và đẩy mô hình vào vùng xấu. Warmup là bắt buộc trên thực tế với Transformer post-LN (Mục 6.4).
- **Cosine decay** — giảm $\eta$ theo hình cos về gần 0 ở cuối. Đây là lịch mặc định của gần như mọi mô hình ngôn ngữ lớn hiện nay.
- **Giảm theo bậc thang** — chia $\eta$ cho 10 tại các mốc. Vẫn phổ biến trong thị giác máy tính.

Một quy tắc kinh nghiệm hữu ích: **tăng kích thước lô lên $k$ lần thì tăng tốc độ học lên khoảng $k$ lần** (quy tắc tuyến tính) hoặc $\sqrt{k}$ lần (quy tắc căn). Quy tắc tuyến tính hợp với SGD, quy tắc căn hợp hơn với Adam. Cả hai đều là quy tắc kinh nghiệm, không phải định lý, và đều hỏng khi lô quá lớn.

### 5.6. Những chỗ hay nhầm

| Phát biểu | Thực tế |
|---|---|
| "Lan truyền ngược là thuật toán học." | Nó chỉ tính gradient. Thuật toán học là SGD/Adam. |
| "Mạng nơ-ron kẹt ở cực tiểu địa phương." | Ở số chiều lớn, **điểm yên ngựa** phổ biến hơn cực tiểu địa phương xấu rất nhiều. Phần lớn cực tiểu địa phương trong mạng lớn có chất lượng gần như nhau. |
| "Adam luôn tốt hơn SGD." | Adam hội tụ nhanh hơn; SGD + momentum thường **tổng quát hoá tốt hơn** trên thị giác. |
| "Gradient bằng 0 nghĩa là đã hội tụ." | Cũng có thể là điểm yên ngựa, hoặc nơ-ron ReLU đã chết, hoặc gradient đã tiêu biến. |
| "Lô càng lớn càng tốt." | Lô rất lớn làm mất phần nhiễu có ích và thường cần chỉnh lại lịch tốc độ học; lợi ích về thời gian cũng bão hoà. |

---

## 6. Khởi tạo, chuẩn hoá và kết nối tắt

### 6.1. Bài toán: tín hiệu co lại hoặc phình ra

Một mạng sâu là một tích các ma trận Jacobi. Nếu mỗi lớp nhân tín hiệu với một hệ số trung bình $\gamma$, thì qua $L$ lớp tín hiệu bị nhân với $\gamma^L$. Chỉ cần $\gamma$ lệch khỏi 1 một chút là sau vài chục lớp mọi thứ hoặc **tiêu biến** về 0 hoặc **bùng nổ** ra vô cực.

Toàn bộ chương này là về ba cách giữ $\gamma \approx 1$, và ba cách ấy **cộng dồn** chứ không thay thế nhau.

### 6.2. Khởi tạo: chọn phương sai cho đúng

Xét một lớp $z = Wx$ với $W \in \mathbb{R}^{n_{\text{out}} \times n_{\text{in}}}$, các phần tử độc lập có phương sai $\sigma_W^2$, và $x$ có các thành phần độc lập phương sai $\sigma_x^2$. Khi đó

$$\operatorname{Var}(z_i) = \sum_{j=1}^{n_{\text{in}}} \operatorname{Var}(W_{ij} x_j) = n_{\text{in}}\,\sigma_W^2\,\sigma_x^2.$$

Muốn $\operatorname{Var}(z) = \operatorname{Var}(x)$ thì phải có $\sigma_W^2 = 1/n_{\text{in}}$. Đó là **khởi tạo Xavier/Glorot**, thường lấy trung bình hài hoà của hai chiều để lượt ngược cũng được giữ:

$$\sigma_W^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}.$$

Với **ReLU** thì phải sửa: ReLU vứt đi một nửa số giá trị, nên nó cắt phương sai đi đúng một nửa. Bù lại bằng cách nhân đôi:

$$\boxed{\sigma_W^2 = \frac{2}{n_{\text{in}}}} \qquad \text{(khởi tạo He)}.$$

> Hệ số 2 trong khởi tạo He **đến từ ReLU**, không phải từ đâu khác. Đây là một câu hỏi phỏng vấn rất hay: nó kiểm tra xem bạn có hiểu rằng khởi tạo phải khớp với hàm kích hoạt hay không. Với tanh thì dùng Xavier; với ReLU dùng He; với SELU lại có hằng số riêng.

### 6.3. Đo thật: điều gì xảy ra qua 40 lớp

Thí nghiệm trong `code/models/experiments.py`: một mạng 40 lớp, mỗi lớp rộng 128, đo **chuẩn của gradient theo kích hoạt** $\|\partial L / \partial a_i\|$ ở từng lớp, chuẩn hoá theo lớp trên cùng.

![Hình 7](figs/models07_vanish.png)

**Hình 7.** Cùng độ sâu 40, chỉ khác khởi tạo và cách dựng lớp. Trục dọc là thang log, trải hơn 26 bậc độ lớn.

| Cách dựng lớp | $\|\nabla a\|$ tại lớp 1 so với lớp 40 | Kết luận |
|---|---|---|
| gain 0,5 — khởi tạo quá nhỏ | $1{,}2 \times 10^{-18}$ | **tiêu biến** |
| gain $\sqrt{2}$ — khởi tạo He | $1{,}42$ | **ổn định** |
| gain 2,0 — khởi tạo quá lớn | $1{,}5 \times 10^{6}$ | **bùng nổ** |
| He + LayerNorm | $1{,}15$ | ổn định, phẳng hơn nữa |
| He + kết nối tắt, **không** chuẩn hoá | $2{,}4 \times 10^{8}$ | **bùng nổ** |
| He + chuẩn hoá + kết nối tắt (pre-LN) | $2{,}22$ | **ổn định** |

Bốn điều rút ra, và điều thứ ba là điều hay bị nói sai nhất:

1. **Khởi tạo là cái nút chính.** Chỉ đổi hệ số từ 0,5 sang 2,0 đã làm gradient đi từ $10^{-18}$ tới $10^{6}$ — trải 24 bậc độ lớn, trên **cùng một kiến trúc**.
2. **Khởi tạo He làm đúng việc nó được thiết kế để làm**: tỉ lệ 1,42 sau 40 lớp, tức gần như giữ nguyên.
3. **Kết nối tắt một mình thì KHÔNG đủ — nó làm bùng nổ.** Lý do đơn giản: mỗi lớp *cộng thêm* vào tín hiệu, nên chuẩn của kích hoạt lớn dần theo chiều xuôi, và đường tắt ở lượt ngược cũng cộng dồn theo. Đây là lý do ResNet **luôn** có chuẩn hoá đi kèm, và vì sao nói "residual chữa gradient tiêu biến" là nói chưa đủ.
4. **Kết hợp chuẩn hoá với kết nối tắt mới cho cấu hình vừa ổn định vừa huấn luyện được** — đúng cấu hình pre-LN mà mọi Transformer hiện đại dùng (Chương 10).

### 6.4. Chuẩn hoá: BatchNorm và LayerNorm

Cả hai cùng làm một việc: chuẩn hoá về trung bình 0, phương sai 1, rồi học lại một phép co giãn và dịch chuyển $\gamma, \beta$. Khác nhau ở **trục nào được lấy thống kê**.

| | BatchNorm | LayerNorm |
|---|---|---|
| Lấy thống kê trên | **cả lô**, cho mỗi kênh đặc trưng | **các đặc trưng**, cho mỗi mẫu |
| Phụ thuộc kích thước lô | **có** — lô nhỏ thì thống kê nhiễu | **không** |
| Khác nhau giữa huấn luyện và suy luận | **có** — suy luận dùng trung bình trượt | **không** |
| Hợp với | CNN, thị giác | Transformer, RNN, chuỗi độ dài thay đổi |

**Vì sao Transformer dùng LayerNorm chứ không BatchNorm** — câu hỏi phỏng vấn rất hay. Ba lý do, nêu được hai là tốt:

1. **Độ dài chuỗi thay đổi.** Thống kê theo lô trên trục thời gian trở nên vô nghĩa khi các chuỗi trong lô dài ngắn khác nhau và có phần đệm.
2. **Có sự khác biệt giữa huấn luyện và suy luận.** BatchNorm dùng thống kê lô lúc huấn luyện nhưng trung bình trượt lúc suy luận — một nguồn lệch kinh điển, và khi sinh văn bản từng token thì lô hiệu dụng bằng 1.
3. **Phụ thuộc lẫn nhau giữa các mẫu.** BatchNorm làm dự đoán cho một mẫu phụ thuộc các mẫu khác trong cùng lô — điều khó chấp nhận khi phục vụ.

**RMSNorm** bỏ luôn bước trừ trung bình, chỉ chia cho chuẩn bậc hai:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot \gamma.$$

Rẻ hơn (bớt một lần quét để tính trung bình, bớt tham số $\beta$) và gần như không mất chất lượng. Nó là mặc định của phần lớn mô hình ngôn ngữ hiện nay.

### 6.5. Kết nối tắt

$$h = x + f(x) \qquad \text{thay cho} \qquad h = f(x).$$

Hai cách hiểu, cả hai đều đúng và nên nói được cả hai:

**Cách 1 — đường đi của gradient.** $\dfrac{\partial h}{\partial x} = I + \dfrac{\partial f}{\partial x}$. Số hạng $I$ bảo đảm gradient luôn có một đường đi **không bị nhân với ma trận trọng số nào**. Qua $L$ lớp, tích trở thành $\prod (I + J_i)$ thay vì $\prod J_i$ — và tích thứ nhất không tự triệt tiêu về 0.

**Cách 2 — bài toán được đặt lại.** Thay vì học hàm $H(x)$, mạng học **phần dư** $F(x) = H(x) - x$. Nếu ánh xạ đồng nhất đã gần đúng thì mạng chỉ cần học một hiệu chỉnh nhỏ, tức khởi điểm đã tốt. Đây chính là lập luận của bài báo ResNet, và nó giải thích **vấn đề suy thoái**: mạng 56 lớp *thường* có sai số **huấn luyện** cao hơn mạng 20 lớp — không phải do quá khớp mà do khó tối ưu hơn.

> Nói được rằng ResNet sinh ra để chữa **vấn đề suy thoái** (sai số *huấn luyện* tăng theo độ sâu), chứ không phải để chữa quá khớp, là một dấu hiệu rõ của người đã đọc bài báo.

### 6.6. Dropout và các cách phạt chuẩn khác

**Dropout**: lúc huấn luyện, tắt ngẫu nhiên mỗi nơ-ron với xác suất $p$, rồi chia phần còn lại cho $1-p$ để giữ kỳ vọng (gọi là *inverted dropout*). Lúc suy luận thì tắt hẳn dropout. Cách hiểu đúng: nó xấp xỉ việc lấy trung bình một tập hợp khổng lồ các mạng con chia sẻ trọng số — tức đúng cơ chế bagging ở Mục 3.3, nhưng miễn phí.

Điều đáng nói: **dropout đã giảm hẳn vai trò trong Transformer lớn.** Với mô hình ngôn ngữ huấn luyện một lượt qua kho dữ liệu khổng lồ, quá khớp không phải vấn đề chính, nên nhiều mô hình lớn đặt dropout bằng 0. Nói được điều này cho thấy bạn theo dõi thực hành chứ không chỉ đọc giáo trình cũ.

| Cách | Cơ chế | Hay dùng ở đâu |
|---|---|---|
| Suy giảm trọng số ($\ell_2$) | phạt chuẩn trọng số | khắp nơi; với Adam thì phải dùng **AdamW** (Mục 5.4) |
| Dropout | trung bình nhiều mạng con | MLP, CNN; ít dần ở LLM |
| Dừng sớm | hạn chế số bước tối ưu | khắp nơi — rẻ nhất, nên làm trước |
| Tăng cường dữ liệu | mở rộng tập huấn luyện | thị giác, âm thanh |
| Làm mượt nhãn | ngăn mô hình quá tự tin | phân loại nhiều lớp |

---

## 7. Mạng tích chập

### 7.1. Ba giả thiết, ba món lợi

Một lớp kết nối đầy đủ trên ảnh $224 \times 224 \times 3$ với 1000 nơ-ron cần $224 \times 224 \times 3 \times 1000 \approx 1{,}5 \times 10^8$ tham số — cho **một** lớp. Tích chập cắt con số ấy bằng ba giả thiết về cấu trúc của ảnh:

| Giả thiết | Tên | Món lợi |
|---|---|---|
| Đặc trưng có ý nghĩa là **cục bộ** | kết nối cục bộ | mỗi nơ-ron chỉ nhìn một vùng $k \times k$ |
| Một đặc trưng hữu ích ở chỗ này thì cũng hữu ích ở chỗ khác | **chia sẻ trọng số** | cùng một bộ lọc quét khắp ảnh |
| Vị trí chính xác ít quan trọng hơn sự có mặt | gộp / bước nhảy | bất biến dịch chuyển gần đúng, giảm độ phân giải |

Số tham số của một lớp tích chập **không phụ thuộc kích thước ảnh**:

$$\text{tham số} = k_h \times k_w \times C_{\text{in}} \times C_{\text{out}} + C_{\text{out}}.$$

Còn số phép tính thì có: $\;\text{FLOP} \approx 2 \times H_{\text{out}} \times W_{\text{out}} \times k_h k_w C_{\text{in}} C_{\text{out}}$.

> Sự tách rời này — **tham số không phụ thuộc kích thước ảnh, FLOP thì có** — là nguồn của nhiều nhầm lẫn khi ước lượng chi phí. Một mô hình "nhẹ" về tham số vẫn có thể rất đắt khi chạy trên ảnh lớn.

### 7.2. Trường tiếp nhận

**Trường tiếp nhận** của một nơ-ron là vùng đầu vào ảnh hưởng tới giá trị của nó.

![Hình 8](figs/models08_cnn.png)

**Hình 8.** Mỗi lớp $3\times3$ nới trường tiếp nhận thêm 2 ô về mỗi chiều. Với bước nhảy 1, nó tăng **tuyến tính** theo số lớp.

Công thức đệ quy, với $k_i$ là kích thước hạt nhân và $s_i$ là bước nhảy của lớp $i$:

$$r_0 = 1, \qquad r_i = r_{i-1} + (k_i - 1)\prod_{j<i} s_j.$$

Với bước nhảy toàn bộ bằng 1 và hạt nhân $3\times3$: $r_L = 1 + 2L$. Muốn trường tiếp nhận phủ hết ảnh $224\times224$ thì cần hơn 110 lớp — **không khả thi**. Đó là lý do CNN thật luôn giảm độ phân giải: mỗi lần bước nhảy 2 thì mọi lớp sau đó nới trường tiếp nhận nhanh gấp đôi.

Hai chi tiết tinh tế đáng biết:

- **Hai lớp $3\times3$ có cùng trường tiếp nhận với một lớp $5\times5$**, nhưng dùng $2 \times 9 = 18$ tham số thay vì 25, và có **thêm một phi tuyến** ở giữa. Đây là lập luận trung tâm của VGG và là lý do hạt nhân lớn gần như biến mất.
- **Trường tiếp nhận hiệu dụng nhỏ hơn nhiều trường tiếp nhận lý thuyết.** Đóng góp của các điểm ở rìa rất yếu; phân bố ảnh hưởng gần như Gauss và bề rộng hiệu dụng tăng theo $O(\sqrt{L})$ chứ không phải $O(L)$.

### 7.3. Các khối đã thành chuẩn

| Khối | Ý tưởng | Vì sao quan trọng |
|---|---|---|
| **Tích chập $1\times1$** | trộn kênh mà không trộn không gian | đổi số kênh rẻ; là "nút cổ chai" của ResNet |
| **Nút cổ chai** ($1\times1 \to 3\times3 \to 1\times1$) | hạ số kênh, tính, rồi nâng lại | giảm mạnh FLOP ở cùng độ sâu |
| **Depthwise separable** | tách thành tích chập theo từng kênh rồi $1\times1$ | giảm chi phí khoảng $k^2$ lần; nền tảng của MobileNet |
| **Khối residual** | $h = x + f(x)$ | cho phép độ sâu hàng trăm lớp (Mục 6.5) |
| **Gộp trung bình toàn cục** | trung bình theo không gian trước lớp phân loại | bỏ được lớp kết nối đầy đủ khổng lồ; nhận ảnh kích thước bất kỳ |

**Depthwise separable đáng tính thử một lần.** Tích chập thường: $k^2 C_{\text{in}} C_{\text{out}}$. Depthwise separable: $k^2 C_{\text{in}} + C_{\text{in}} C_{\text{out}}$. Tỉ lệ:

$$\frac{k^2 C_{\text{in}} + C_{\text{in}}C_{\text{out}}}{k^2 C_{\text{in}} C_{\text{out}}} = \frac{1}{C_{\text{out}}} + \frac{1}{k^2}.$$

Với $k=3$ và $C_{\text{out}}$ lớn, tỉ lệ $\approx 1/9$ — **rẻ hơn khoảng 9 lần**. Đây cũng chính là lớp mà giáo trình Quantization chỉ ra là nhạy nhất với lượng tử per-tensor, vì mỗi kênh chỉ có 9 trọng số.

### 7.4. CNN so với Transformer trong thị giác

Vision Transformer cắt ảnh thành các mảnh rồi coi chúng như token. Điểm cần hiểu là **đánh đổi về thiên lệch quy nạp**:

- CNN **cài sẵn** tính cục bộ và bất biến dịch chuyển. Đó là thiên lệch quy nạp mạnh: rất có lợi khi dữ liệu ít, nhưng là một ràng buộc khi dữ liệu nhiều.
- ViT gần như **không cài sẵn gì**, nên phải học cả những thứ CNN được cho không. Vì vậy ViT thua CNN khi huấn luyện trên dữ liệu vừa phải, và **vượt** khi có đủ dữ liệu hoặc được tiền huấn luyện quy mô lớn.

Đây là một ví dụ sạch của đánh đổi thiên lệch – phương sai ở Chương 2, nhưng ở mức **kiến trúc** thay vì mức số tham số: thiên lệch quy nạp mạnh = thiên lệch cao, phương sai thấp.

---

## 8. Mạng hồi quy và giới hạn của nó

### 8.1. Ý tưởng và công thức

RNN xử lý chuỗi bằng cách giữ một **trạng thái ẩn** và cập nhật nó ở mỗi bước:

$$h_t = \phi(W_{hh} h_{t-1} + W_{xh} x_t + b), \qquad \hat{y}_t = W_{hy} h_t.$$

Cùng một bộ trọng số dùng cho mọi bước thời gian — đây là "chia sẻ trọng số" của Mục 7.1, nhưng theo trục thời gian thay vì trục không gian. Nhờ đó RNN xử lý được chuỗi dài tuỳ ý với số tham số cố định.

### 8.2. Vì sao nó hỏng

Gradient của mất mát tại bước $T$ theo trạng thái tại bước $t$ là một **tích các ma trận Jacobi**:

$$\frac{\partial h_T}{\partial h_t} = \prod_{i=t+1}^{T} \frac{\partial h_i}{\partial h_{i-1}} = \prod_{i=t+1}^{T} W_{hh}^\top \operatorname{diag}\big(\phi'(z_i)\big).$$

Đây đúng là tình huống ở Mục 6.1, nhưng tệ hơn ở một điểm: **cùng một ma trận $W_{hh}$ được nhân lặp lại**. Nếu bán kính phổ của nó nhỏ hơn 1 thì tích co về 0 theo hàm mũ; lớn hơn 1 thì bùng nổ.

Thí nghiệm trong `code/models/experiments.py` đo trực tiếp điều đó: khởi tạo $W_{hh}$ với các phần tử phương sai $g^2/n$ (nên bán kính phổ $\approx g$), rồi truyền ngược 100 bước.

![Hình 9](figs/models09_rnn.png)

**Hình 9.** Chuẩn gradient tương đối khi truyền ngược qua thời gian. Trục dọc là thang log.

| Mô hình | sau 10 bước | sau 50 bước | sau 100 bước |
|---|---|---|---|
| RNN, gain 0,9 | $3{,}2 \times 10^{-1}$ | $6{,}8 \times 10^{-2}$ | $3{,}2 \times 10^{-2}$ |
| RNN, gain 1,0 | $4{,}2 \times 10^{-1}$ | $1{,}8 \times 10^{-1}$ | $1{,}1$ |
| RNN, gain 1,2 | $6{,}4 \times 10^{-1}$ | $3{,}8 \times 10^{-1}$ | $3{,}0$ |
| **LSTM, bias cổng quên = 1** | $5{,}0 \times 10^{-2}$ | $8{,}0 \times 10^{-8}$ | $3{,}0 \times 10^{-15}$ |
| **LSTM, bias cổng quên = 4** | $8{,}3 \times 10^{-1}$ | $3{,}6 \times 10^{-1}$ | $1{,}3 \times 10^{-1}$ |

Bảng này nói ba điều, và điều thứ ba là điều quan trọng nhất:

1. **Bán kính phổ quyết định tất cả** với RNN thường: dưới 1 thì tiêu biến, trên 1 thì bùng nổ. Điểm 1,0 là ranh giới, và nó **không ổn định** — trong huấn luyện thật trọng số thay đổi nên không ở lại đó được.
2. **Bùng nổ dễ chữa, tiêu biến thì không.** Bùng nổ chữa bằng **cắt ngưỡng gradient** (gradient clipping): chia cho chuẩn nếu chuẩn vượt ngưỡng. Tiêu biến thì không có mẹo nào tương đương, vì thông tin đã mất chứ không phải quá lớn.
3. **LSTM không tự động cứu được gì — cái cứu là giá trị của cổng quên.**

### 8.3. LSTM: vì sao nó giúp, và giúp tới đâu

LSTM thêm một **đường trạng thái ô nhớ** $c_t$ được cập nhật bằng phép **cộng** thay vì nhân ma trận:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t, \qquad h_t = o_t \odot \tanh(c_t),$$

với $f_t$ (quên), $i_t$ (vào), $o_t$ (ra) là các cổng sigmoid. Gradient dọc theo đường $c$ là

$$\frac{\partial c_T}{\partial c_t} = \prod_{i=t+1}^{T} f_i$$

— một tích các **số vô hướng trong $(0,1)$**, không phải tích các ma trận. Khác biệt then chốt: mạng **học được** giá trị $f_i$, nên nó có thể chọn $f_i \approx 1$ để giữ thông tin lâu.

Nhưng bảng trên cho thấy điều mà sách giáo khoa hay bỏ: **nếu $f \approx 0{,}73$ (tức bias cổng quên bằng 1) thì sau 100 bước gradient còn $10^{-15}$ — tệ hơn cả RNN.** Với bias bằng 4, tức $f \approx 0{,}98$, gradient còn 0,13. Đây chính là lý do có thủ thuật kinh điển **khởi tạo độ lệch của cổng quên bằng một số dương** (thường 1 tới 3): nó đặt điểm xuất phát ở chỗ ô nhớ có xu hướng *giữ* thay vì *quên*.

> Nói được rằng LSTM biến *tích các ma trận Jacobi* thành *tích các số vô hướng học được*, và rằng lợi ích ấy phụ thuộc giá trị cổng quên, là câu trả lời đầy đủ cho "vì sao LSTM tốt hơn RNN".

**GRU** gộp cổng quên và cổng vào thành một, bỏ cổng ra, còn 2 cổng thay vì 3. Ít tham số hơn khoảng 25%, chất lượng thường tương đương. Chọn cái nào là chuyện thực nghiệm, không có câu trả lời lý thuyết.

### 8.4. Giới hạn thật sự: tính tuần tự

Vấn đề gradient có thể giảm nhẹ. Cái không giảm nhẹ được là:

> $h_t$ phụ thuộc $h_{t-1}$, nên **không thể tính song song theo trục thời gian.** Huấn luyện trên chuỗi dài $T$ cần $T$ bước tuần tự, bất kể có bao nhiêu GPU.

Đây mới là lý do RNN bị thay thế. Với độ dài chuỗi $T$ và mô hình chiều $d$:

| | RNN | Self-attention |
|---|---|---|
| Phép tính mỗi lớp | $O(T d^2)$ | $O(T^2 d)$ |
| **Số bước tuần tự** | $O(T)$ | $O(1)$ |
| Đường đi dài nhất giữa hai vị trí | $O(T)$ | $O(1)$ |

Hai hàng cuối là toàn bộ câu chuyện. Attention đắt hơn về phép tính khi $T > d$, nhưng phép tính thì song song hoá được còn **các bước tuần tự thì không**. Và đường đi ngắn giữa hai vị trí bất kỳ nghĩa là gradient không phải đi qua $T$ phép nhân để nối hai token xa nhau.

Đó là nội dung của chương tiếp theo.

---

## 9. Attention và Transformer

### 9.1. Ý tưởng: tra cứu mềm

Attention là một **phép tra cứu từ điển có thể lấy đạo hàm**. Trong một từ điển thường, ta có khoá và giá trị, đưa vào một truy vấn, khớp đúng một khoá, lấy về một giá trị. Attention làm y hệt, chỉ khác là thay vì khớp cứng thì nó **khớp mềm**: tính độ giống giữa truy vấn và mọi khoá, chuẩn hoá thành trọng số, rồi lấy tổ hợp có trọng số của các giá trị.

$$\text{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V.$$

Ba vai trò, và phân biệt được chúng là bước đầu để hiểu:

| Ký hiệu | Vai trò | Câu hỏi nó trả lời |
|---|---|---|
| $Q$ (query) | vị trí đang hỏi | "tôi đang cần thông tin gì?" |
| $K$ (key) | nhãn của từng vị trí | "tôi có loại thông tin gì?" |
| $V$ (value) | nội dung của từng vị trí | "thông tin đó cụ thể là gì" |

Trong **self-attention**, cả ba đều là phép chiếu tuyến tính của **cùng một** dãy đầu vào: $Q = XW_Q$, $K = XW_K$, $V = XW_V$. Trong **cross-attention**, $Q$ đến từ một dãy còn $K, V$ đến từ dãy khác.

### 9.2. Vì sao phải chia cho $\sqrt{d_k}$

Bài báo gốc nêu lý do trong đúng một chú thích, và lập luận rất gọn: nếu các thành phần của $q$ và $k$ là biến ngẫu nhiên độc lập có trung bình 0 và phương sai 1, thì

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i \quad\text{có trung bình } 0 \text{ và phương sai } d_k.$$

Tức độ lệch chuẩn của điểm số tăng theo $\sqrt{d_k}$. Với $d_k = 64$, điểm số dao động trong khoảng $\pm 8$; với $d_k = 1024$ là $\pm 32$. Softmax của các số cách nhau hàng chục là một vector gần như one-hot — và **đạo hàm của softmax tại vùng bão hoà gần bằng 0**. Chia cho $\sqrt{d_k}$ đưa phương sai về đúng 1, bất kể $d_k$.

Thí nghiệm trong `code/models/experiments.py` đo trực tiếp cả hai vế:

![Hình 10](figs/models10_scaling.png)

**Hình 10.** Trái: phương sai của $q \cdot k$ đo được, so với dự đoán lý thuyết $d_k$ (đường chấm). Phải: entropy của phân phối attention trên 64 khoá, có và không có phép chia.

| $d_k$ | $\operatorname{Var}(q\cdot k)$ đo được | Lý thuyết | Entropy **không** chia | Entropy **có** chia | softmax lớn nhất |
|---|---|---|---|---|---|
| 4 | 3,63 | 4 | 2,892 | 3,739 | 0,258 |
| 64 | 64,66 | 64 | 0,582 | 3,680 | 0,791 |
| 1024 | 1021,97 | 1024 | **0,118** | 3,685 | **0,953** |

Đọc bảng:

- **Phương sai khớp lý thuyết tới hai chữ số** ở mọi $d_k$. Đây không phải trực giác mà là một đẳng thức kiểm chứng được.
- **Không chia thì attention hoá cứng.** Ở $d_k = 1024$, entropy còn 0,118 nat trong khi mức tối đa là $\ln 64 = 4{,}159$ — tức phân phối đã gần như one-hot, và trọng số lớn nhất trung bình là 0,953. Softmax ở trạng thái ấy gần như không có gradient.
- **Có chia thì entropy đứng yên ở khoảng 3,68 bất kể $d_k$.** Đúng mục tiêu thiết kế: hành vi của attention không được phụ thuộc vào việc ta chọn bao nhiêu chiều cho mỗi đầu.

> Câu trả lời phỏng vấn đầy đủ gồm ba phần: **(1)** phương sai của tích vô hướng bằng $d_k$; **(2)** điểm số lớn đẩy softmax vào vùng bão hoà nơi gradient gần 0; **(3)** chia cho $\sqrt{d_k}$ đưa phương sai về 1. Nói được cả ba, kèm con số, là đã hơn hẳn mức "để ổn định huấn luyện".

### 9.3. Nhiều đầu

Thay vì một phép attention trên $d$ chiều, ta chiếu xuống $h$ không gian con $d_k = d/h$ chiều và làm $h$ phép attention song song, rồi nối lại và chiếu một lần nữa:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,W_O.$$

Bài báo gốc dùng $h = 8$ và $d_k = d_v = d_{\text{model}}/h = 64$, và nêu rõ **tổng chi phí tính toán xấp xỉ bằng attention một đầu ở chiều đầy đủ** — vì mỗi đầu hẹp đi đúng $h$ lần.

Vậy nhiều đầu mua được gì nếu không phải sức tính? Bài báo trả lời: nó cho phép mô hình **cùng lúc chú ý tới thông tin từ nhiều không gian biểu diễn khác nhau ở những vị trí khác nhau**; với một đầu duy nhất thì phép lấy trung bình sẽ làm nhoè điều đó. Nói cách khác, nhiều đầu mua **sự đa dạng của quan hệ**, không mua sức tính.

### 9.4. Mặt nạ

Hai loại mặt nạ, khác mục đích, hay bị gộp làm một:

| Loại | Che gì | Vì sao |
|---|---|---|
| **Mặt nạ nhân quả** (causal) | mọi vị trí $j > i$ | để mô hình sinh không nhìn thấy tương lai — nếu không thì bài toán dự đoán token kế tiếp trở nên tầm thường |
| **Mặt nạ đệm** (padding) | các vị trí đệm | để phần đệm không đóng góp vào tổng có trọng số |

Cài đặt: cộng $-\infty$ (thực tế là một số âm rất lớn) vào điểm số **trước** softmax, để sau softmax trọng số đúng bằng 0.

### 9.5. Bên trong một khối

![Hình 11](figs/models11_block.png)

**Hình 11.** Một khối Transformer đầy đủ, vẽ theo thứ tự pre-LN. Hai khối con, mỗi khối con có một kết nối tắt đi vòng qua nó.

Điều đáng nhớ nhất về cấu trúc này là **phân công lao động giữa hai khối con**:

| | Khối con 1 — attention | Khối con 2 — FFN |
|---|---|---|
| Trộn theo trục nào | **giữa các vị trí** | **giữa các chiều đặc trưng** |
| Mỗi vị trí có độc lập không | không — mọi vị trí nhìn thấy nhau | **có** — hoàn toàn độc lập |
| Số tham số (với $d_{\text{ff}} = 4d$) | $4d^2$ | $8d^2$ |
| Chi phí theo độ dài chuỗi | $O(T^2 d)$ | $O(T d^2)$ |

Hai điều rút ra:

1. **FFN chiếm hai phần ba số tham số của khối**, dù attention mới là phần được nói tới nhiều. Chương 12 kiểm chứng con số này trên GPT-2 thật.
2. **Attention là phần duy nhất trộn thông tin giữa các vị trí.** Bỏ nó đi thì Transformer chỉ còn là một MLP áp dụng độc lập cho từng token. Đây là cách trả lời gọn nhất cho "attention làm gì".

### 9.6. Vị trí: vì sao cần mã hoá thêm

Self-attention **hoán vị bất biến**: đảo thứ tự các token đầu vào thì đầu ra cũng chỉ đảo theo, giá trị không đổi. Với ngôn ngữ thì đó là tai hoạ — "chó cắn người" và "người cắn chó" sẽ cho cùng biểu diễn.

Vì thế phải đưa thông tin vị trí vào bằng tay. Ba thế hệ:

1. **Sin/cos cố định** (bài báo gốc): $p_{i,2t} = \sin(i/10000^{2t/d})$, $p_{i,2t+1} = \cos(i/10000^{2t/d})$, cộng thẳng vào embedding. Không có tham số học được; các tần số khác nhau cho phép mô hình suy ra khoảng cách.
2. **Embedding vị trí học được** (BERT, GPT-2): một bảng tra $T_{\max} \times d$. Đơn giản, nhưng **không ngoại suy** được quá $T_{\max}$.
3. **RoPE** — nội dung của Mục 10.3.

---

## 10. Transformer hiện đại

### 10.1. Cùng bộ khung, năm chi tiết đã đổi

![Hình 12](figs/models14_modern.png)

**Hình 12.** Kiến trúc năm 2017 so với thứ phổ biến hiện nay. Bộ khung không đổi; năm chi tiết đổi, mỗi chi tiết vì một lý do cụ thể.

### 10.2. Pre-LN thay cho post-LN

Bài báo gốc đặt chuẩn hoá **sau** khối con: $x \leftarrow \text{LN}(x + \text{Sublayer}(x))$. Cách làm phổ biến hiện nay đặt chuẩn hoá **trước**: $x \leftarrow x + \text{Sublayer}(\text{LN}(x))$.

Khác biệt nằm ở chỗ **đường tắt có bị chuẩn hoá hay không**. Với post-LN, mọi đường tắt đều đi qua một phép LN, nên tín hiệu từ đầu vào tới đầu ra bị co lại nhiều lần. Với pre-LN, đường tắt đi **thẳng** từ đầu vào tới đầu ra mà không qua phép biến đổi nào — đúng cấu hình đã đo ở Mục 6.3 cho tỉ lệ gradient 2,22 sau 40 lớp.

Hệ quả thực tế: **post-LN cần warmup dài và nhạy với tốc độ học**; pre-LN huấn luyện ổn định hơn nhiều và bớt phụ thuộc warmup. Cái giá là pre-LN thường cho chất lượng cuối hơi thấp hơn ở cùng ngân sách, nên một số mô hình dùng các biến thể lai.

### 10.3. RoPE

Ý tưởng: thay vì **cộng** thông tin vị trí vào embedding, hãy **xoay** các vector $Q$ và $K$ một góc tỉ lệ với vị trí.

Chia $d$ chiều thành $d/2$ cặp. Cặp thứ $i$ được xoay một góc $m\theta_i$ với $\theta_i = 10000^{-2i/d}$ — đúng dãy tần số của mã hoá sin/cos gốc. Viết $R_m$ là phép xoay khối tại vị trí $m$, tính chất then chốt là

$$\langle R_m q,\; R_n k \rangle = \langle R_{m-n}\, q,\; k \rangle.$$

Tức **điểm số attention chỉ phụ thuộc khoảng cách tương đối $m-n$**, dù mỗi vector được mã hoá bằng vị trí tuyệt đối.

Thí nghiệm trong `code/models/experiments.py` kiểm chứng trực tiếp:

![Hình 13](figs/models12_rope.png)

**Hình 13.** Trái: cùng khoảng cách tương đối cho cùng một tích vô hướng, dù vị trí tuyệt đối khác nhau hàng trăm. Phải: độ tương đồng của cùng một vector ở hai vị trí, theo khoảng cách.

| $m$ | $n$ | $m-n$ | $\langle R_m q, R_n k\rangle$ |
|---|---|---|---|
| 5 | 2 | 3 | 6,851342 |
| 105 | 102 | 3 | 6,851342 |
| 500 | 497 | 3 | **6,851342** |
| 9 | 3 | 6 | 5,861634 |
| 109 | 103 | 6 | 5,861634 |

Giá trị **trùng khít tới chữ số cuối** — đây là một đẳng thức, không phải xấp xỉ. Và đo độ tương đồng của cùng một vector ở hai vị trí cách nhau $\Delta$:

| $\Delta$ | 0 | 8 | 64 | 255 |
|---|---|---|---|---|
| $\langle R_m q, R_0 q\rangle / \|q\|^2$ | 1,000 | 0,708 | 0,439 | **0,319** |

Đây là tính chất **suy giảm theo khoảng cách** mà bài báo RoPE nêu: hai token cách xa nhau thì có xu hướng liên hệ yếu hơn. Nó được cài sẵn vào hình học chứ không phải học ra.

Ba món lợi của RoPE, nên nói được cả ba:

1. **Quan hệ tương đối miễn phí** — không cần bảng tra bias tương đối như một số phương án khác.
2. **Ngoại suy độ dài tốt hơn** — không có bảng vị trí nào để hết chỗ; và có các kỹ thuật kéo giãn tần số để mở rộng ngữ cảnh sau khi đã huấn luyện.
3. **Chỉ tác động vào $Q$ và $K$** — không đụng $V$, không thêm tham số, không đổi hình dạng tensor.

### 10.4. GQA và MQA

Trong attention nhiều đầu chuẩn, mỗi đầu có $K$ và $V$ riêng. Khi sinh văn bản, toàn bộ $K$ và $V$ của các token trước phải được lưu lại — đó là **KV cache**, và dung lượng của nó tỉ lệ với số đầu.

- **MQA** (multi-query): mọi đầu dùng chung **một** cặp $K, V$.
- **GQA** (grouped-query): chia $h$ đầu thành $g$ nhóm, mỗi nhóm chung một cặp $K, V$. MQA là trường hợp $g = 1$, MHA là $g = h$.

KV cache giảm đúng theo tỉ lệ $h/g$. Chất lượng giảm rất ít khi $g$ không quá nhỏ — đó là lý do GQA với $g = 8$ trở thành mặc định. Phép tính dung lượng cụ thể nằm ở giáo trình MLOps, Mục 11.7; ở đây chỉ cần nhớ rằng **GQA là một quyết định kiến trúc được đưa ra vì lý do bộ nhớ lúc suy luận, không phải vì chất lượng**.

### 10.5. SwiGLU

FFN gốc là $W_2\,\phi(W_1 x)$ với $\phi$ = ReLU và $d_{\text{ff}} = 4d$, dùng **hai** ma trận. Các biến thể GLU thay lớp tuyến tính thứ nhất và hàm kích hoạt bằng một **cổng nhân**:

$$\text{SwiGLU}(x) = \big(\text{Swish}(xW) \odot xV\big)W_2.$$

Điểm phải nhớ: **có ba ma trận thay vì hai**. Để giữ nguyên số tham số và số phép tính, bề rộng ẩn phải giảm đi đúng $2/3$. Bài báo đề xuất SwiGLU làm đúng điều đó: mô hình nền có $d_{\text{ff}} = 3072$, còn các biến thể GLU dùng $d_{\text{ff}} = 2048$ để **khớp cả số tham số lẫn số phép tính** với mô hình nền.

Vì vậy con số hay gặp trong các mô hình hiện đại là

$$d_{\text{ff}} \approx \frac{2}{3} \times 4d = \frac{8}{3}d,$$

thường được làm tròn lên bội của 128 hoặc 256 cho vừa phần cứng. Biết vì sao có phân số $2/3$ là dấu hiệu đã đọc bài báo chứ không chỉ chép cấu hình.

### 10.6. Mixture of Experts

Ý tưởng: thay lớp FFN bằng $E$ lớp FFN song song ("chuyên gia") và một **bộ định tuyến** chọn $k$ chuyên gia cho mỗi token (thường $k = 1$ hoặc 2).

Hệ quả là **tách rời số tham số khỏi số phép tính**: mô hình có tổng tham số rất lớn nhưng mỗi token chỉ kích hoạt một phần nhỏ. Cái giá là ba vấn đề kỹ thuật thật:

1. **Cân bằng tải** — bộ định tuyến có xu hướng dồn vào vài chuyên gia; phải thêm một hàm mất mát phụ để phạt sự mất cân bằng.
2. **Bộ nhớ** — dù chỉ kích hoạt một phần, **toàn bộ** trọng số vẫn phải nằm sẵn trong bộ nhớ.
3. **Giao tiếp** — chuyên gia thường nằm trên các thiết bị khác nhau, nên định tuyến sinh ra trao đổi dữ liệu qua mạng.

> Câu trả lời gọn cho "MoE mua được gì": nó mua **dung lượng tham số trên mỗi FLOP**, và trả bằng **bộ nhớ và giao tiếp**. Nếu nút thắt của bạn là bộ nhớ chứ không phải phép tính thì MoE không giúp.

---

## 11. Sinh văn bản: các chiến lược giải mã

### 11.1. Bài toán

Mô hình cho một phân phối $p(\text{token kế tiếp} \mid \text{ngữ cảnh})$ trên toàn bộ từ vựng. **Giải mã** là quy tắc biến phân phối ấy thành một token cụ thể. Đây là một quyết định **hoàn toàn nằm ngoài mô hình** — cùng một mô hình với hai chiến lược giải mã khác nhau cho ra hai hệ thống rất khác nhau.

### 11.2. Các chiến lược

| Chiến lược | Quy tắc | Hợp với |
|---|---|---|
| **Greedy** | luôn lấy token xác suất cao nhất | bài toán có một đáp án đúng: dịch, trích xuất, phân loại |
| **Beam search** | giữ $B$ chuỗi tốt nhất, mở rộng song song | dịch máy, tóm tắt — nơi cần xác suất chuỗi cao |
| **Lấy mẫu + nhiệt độ** | chia logits cho $T$ rồi lấy mẫu | sáng tạo, hội thoại |
| **Top-$k$** | chỉ giữ $k$ token cao nhất rồi lấy mẫu | chặn đuôi rác với ngưỡng cố định |
| **Top-$p$ (nucleus)** | giữ tập nhỏ nhất có tổng xác suất $\ge p$ | chặn đuôi rác với ngưỡng **thích ứng** |

### 11.3. Đo thật: mỗi chiến lược nhào nặn phân phối thế nào

Thí nghiệm trong `code/models/experiments.py`: một phân phối 50 000 token có dạng đuôi dài giống thực tế, rồi đo entropy, số token phủ 90% xác suất, và xác suất của token đứng đầu.

![Hình 14](figs/models13_decode.png)

**Hình 14.** Trái: hình dạng phân phối sau mỗi phép biến đổi (thang log). Phải: số token còn thực sự nằm trong cuộc chơi.

| Chiến lược | Entropy (nat) | Số token phủ 90% | Xác suất top-1 |
|---|---|---|---|
| Gốc ($T = 1$) | 8,223 | **8 362** | 0,027 |
| $T = 0{,}7$ | 5,886 | 1 215 | 0,128 |
| $T = 1{,}3$ | 9,305 | 16 766 | 0,008 |
| Top-$k$ = 40 | 3,449 | **33** | 0,139 |
| Top-$p$ = 0,9 | 7,669 | 4 141 | 0,030 |
| Top-$p$ = 0,5 | 5,673 | 376 | 0,055 |

Ba điều đọc ra:

1. **Phân phối gốc có đuôi khổng lồ.** Cần 8 362 token để phủ 90% xác suất. Nếu lấy mẫu thẳng từ nó thì xác suất chọn phải một token vô nghĩa là rất đáng kể ở mỗi bước — và một bước sai sẽ kéo theo cả phần còn lại.
2. **Nhiệt độ co giãn toàn bộ phân phối, top-$k$/top-$p$ thì cắt.** $T = 0{,}7$ vẫn để lại 1 215 token trong cuộc chơi; top-$k$ = 40 chỉ để lại 33. Hai cơ chế khác hẳn nhau, và **thường dùng chung**: cắt đuôi bằng top-$p$ rồi chỉnh độ sắc bằng nhiệt độ.
3. **Vì sao top-$p$ thường hơn top-$k$**: với ngữ cảnh mà mô hình rất chắc chắn, top-$k$ = 40 vẫn ép giữ 40 lựa chọn, trong đó 39 là rác; với ngữ cảnh mơ hồ, nó lại cắt mất những lựa chọn hợp lý. Top-$p$ **tự co giãn theo độ chắc chắn** của từng bước.

### 11.4. Vì sao không phải lúc nào cũng greedy

Greedy tối đa hoá xác suất **từng bước**, không phải xác suất của cả chuỗi — một dạng tham lam như thuật toán dựng cây ở Mục 3.2. Beam search giảm nhẹ điều đó bằng cách giữ nhiều ứng viên.

Nhưng với văn bản mở, **beam search cho kết quả tệ một cách đáng ngạc nhiên**: nó sinh ra văn bản lặp lại, nhạt nhẽo và chung chung. Lý do được chấp nhận rộng rãi: văn bản do người viết **không** có xác suất cao một cách đều đặn — nó có những chỗ bất ngờ. Tối đa hoá xác suất chuỗi vì thế đẩy mô hình về phía văn bản "an toàn" và nhàm chán. Đây là lý do văn bản mở dùng **lấy mẫu**, còn dịch máy — nơi có một đáp án đúng — vẫn dùng beam.

Hai công cụ chữa lặp thường gặp: **phạt lặp** (giảm logit của token đã xuất hiện) và **chặn n-gram lặp** (cấm lặp lại một cụm $n$ từ). Cả hai là biện pháp vá, không phải lời giải gốc.

### 11.5. Giải mã và hệ thống

Ba điều thuộc về vận hành nhưng phải quyết ở đây:

- **Nhiệt độ 0 không phải lúc nào cũng tất định.** Với $T = 0$ thì quy tắc là greedy, nhưng thứ tự cộng dồn trên GPU có thể đổi giữa các lần chạy, và với kích thước lô khác nhau thì kết quả cũng có thể khác. Hứa "tất định" với người dùng là hứa liều.
- **Giải mã suy đoán** (speculative decoding) dùng một mô hình nhỏ đề xuất vài token rồi mô hình lớn kiểm tra một lượt. Nó tăng tốc mà **không đổi phân phối đầu ra** — một trong số rất ít bữa trưa miễn phí trong lĩnh vực này.
- **Chi phí sinh tỉ lệ với số token sinh ra**, và mỗi token cần một lượt xuôi đầy đủ. Đó là vì sao "bảo mô hình trả lời ngắn gọn" là một đòn bẩy chi phí thật sự, không phải mẹo vặt.

---

## 12. Đếm tham số, FLOP và bộ nhớ

### 12.1. Vì sao chương này quan trọng

Đây là chương ứng dụng nhất của cả tài liệu. Biết đếm tham số và FLOP cho phép trả lời, **trên giấy và trong vài phút**, những câu hỏi mà nếu không biết thì phải thử nghiệm hàng giờ: mô hình này có vừa GPU không, huấn luyện mất bao lâu, tăng ngữ cảnh gấp đôi thì đắt lên bao nhiêu.

### 12.2. Đếm tham số của một khối Transformer

Với $d$ = chiều mô hình, $d_{\text{ff}}$ = bề rộng FFN:

| Thành phần | Tham số |
|---|---|
| $W_Q, W_K, W_V$ | $3d^2$ |
| $W_O$ | $d^2$ |
| FFN ($W_1$, $W_2$) | $2 d\, d_{\text{ff}}$ |
| **Tổng một khối, với $d_{\text{ff}} = 4d$** | $4d^2 + 8d^2 = \mathbf{12d^2}$ |

Nhân với số khối $L$:

$$\boxed{N_{\text{phi-embedding}} \approx 12\,L\,d^2}$$

Đây đúng là công thức mà bài báo về quy luật co giãn của Kaplan và cộng sự dùng, với quy ước $d_{\text{attn}} = d_{\text{ff}}/4 = d_{\text{model}}$.

Cộng thêm phần embedding: $V d$ cho từ vựng và $T_{\max} d$ cho vị trí học được (RoPE thì không có phần này).

### 12.3. Kiểm chứng trên GPT-2 thật

Công thức chỉ đáng tin nếu nó tái lập được con số đã công bố. Thí nghiệm trong `code/models/experiments.py` tính đầy đủ — kể cả độ lệch và LayerNorm — rồi đối chiếu:

| Mô hình | Công thức | Đã công bố | Lệch | Phi-embedding |
|---|---|---|---|---|
| GPT-2 small | 124 439 808 | 124 439 808 | **0** | 84 934 656 |
| GPT-2 medium | 354 823 168 | 354 823 168 | **0** | 301 989 888 |
| GPT-2 large | 774 030 080 | 774 030 080 | **0** | 707 788 800 |

Khớp **chính xác tới từng tham số** trên cả ba kích thước. Và phần phi-embedding của GPT-2 small là 84 934 656, đúng bằng $12 \times 12 \times 768^2$ — tức công thức $12Ld^2$ không phải xấp xỉ mà là đẳng thức khi bỏ độ lệch và LayerNorm.

Tách nhỏ GPT-2 small:

| Thành phần | Tham số | Tỉ lệ |
|---|---|---|
| Embedding từ vựng | 38 597 376 | **31,0%** |
| Embedding vị trí | 786 432 | 0,6% |
| Attention (12 khối) | 28 348 416 | 22,8% |
| FFN (12 khối) | 56 669 184 | **45,5%** |

Hai con số đáng nhớ:

- **Embedding chiếm 31% tham số của GPT-2 small.** Với mô hình nhỏ và từ vựng lớn, phần embedding **chi phối**. Đây là lý do các bài báo về quy luật co giãn tách riêng phần phi-embedding: chỉ khi tách ra thì quy luật mới sạch.
- **Trong khối Transformer, FFN chiếm 67% tham số** (56,7 triệu trên tổng 85 triệu của hai khối con). Attention được nói tới nhiều hơn, nhưng phần lớn trọng số nằm ở FFN.

### 12.4. Đếm FLOP

Quy tắc nền: một phép nhân ma trận $(m \times k) \times (k \times n)$ tốn $2mkn$ FLOP — hệ số 2 vì mỗi phần tử cần một phép nhân và một phép cộng.

Từ đó, cho **mỗi token**:

$$C_{\text{xuôi}} \approx 2N, \qquad C_{\text{xuôi + ngược}} \approx 6N$$

với $N$ là số tham số phi-embedding. Hệ số 3 giữa xuôi và toàn bộ là vì lượt ngược tốn khoảng **hai lần** lượt xuôi: một lần cho gradient theo đầu vào, một lần cho gradient theo trọng số.

Vì vậy chi phí huấn luyện toàn bộ trên $D$ token là

$$\boxed{C \approx 6 N D \ \text{FLOP}}$$

— một trong những công thức hữu dụng nhất trong toàn bộ lĩnh vực.

**Kiểm chứng với GPT-2 small:** $N = 84\,934\,656$, nên suy luận tốn $2N = 169{,}9$ triệu FLOP mỗi token và huấn luyện tốn $6N = 509{,}6$ triệu FLOP mỗi token.

**Phần attention thì sao?** Công thức $2N$ bỏ qua phép tính $QK^\top$ và $\cdot V$, vốn tốn thêm khoảng $2 L T d$ FLOP mỗi token với ngữ cảnh $T$. Phần này **tỉ lệ với $T$**, còn $2N$ thì không. Tỉ lệ giữa chúng:

$$\frac{\text{attention}}{\text{phần còn lại}} \approx \frac{2LTd}{12Ld^2} = \frac{T}{6d}.$$

Với GPT-2 ($d = 768$, $T = 1024$): tỉ lệ $= 1024/4608 \approx 22\%$ — đáng kể nhưng chưa chi phối. Với $T = 32\,768$ và cùng $d$: tỉ lệ $\approx 7{,}1$, tức **attention chiếm phần lớn chi phí**. Đây là lý do định lượng cho việc "chi phí bậc hai theo độ dài" chỉ trở thành vấn đề khi $T$ vượt quá $6d$.

### 12.5. Đếm bộ nhớ

Ba khoản khác nhau, và nhầm lẫn giữa chúng là nguồn gốc của phần lớn sự cố hết bộ nhớ:

| Khoản | Công thức | Phụ thuộc |
|---|---|---|
| **Trọng số** | $N \times$ số byte | chỉ số tham số |
| **Trạng thái tối ưu hoá** | $2N \times$ số byte (Adam) | chỉ số tham số |
| **Kích hoạt** | $\approx L \times B \times T \times d \times$ số byte $\times c$ | **kích thước lô và độ dài chuỗi** |
| **KV cache** (lúc sinh) | $2 \times L \times n_{kv} \times d_{\text{head}} \times T \times B \times$ số byte | độ dài và batch |

Ví dụ cụ thể cho một mô hình 7 tỉ tham số huấn luyện ở hỗn hợp độ chính xác với Adam:

- trọng số FP16: $7 \times 10^9 \times 2 = 14$ GB
- bản sao trọng số FP32 cho bộ tối ưu: 28 GB
- trạng thái Adam ($m$ và $v$) ở FP32: 56 GB
- **cộng lại đã 98 GB, chưa tính kích hoạt**

Đây là lý do huấn luyện mô hình 7B không vừa một GPU 80 GB nếu không có kỹ thuật phân mảnh trạng thái tối ưu hoá. Và cũng là lý do giáo trình Quantization tồn tại: giảm số byte mỗi tham số là đòn bẩy trực tiếp lên cả bốn dòng của bảng trên.

### 12.6. Bảng bỏ túi

| Đại lượng | Công thức | Dùng khi |
|---|---|---|
| Tham số phi-embedding | $12 L d^2$ | ước lượng cỡ mô hình |
| FLOP suy luận mỗi token | $2N$ | ước lượng độ trễ |
| FLOP huấn luyện | $6ND$ | ước lượng ngân sách tính toán |
| Khi nào attention chi phối | $T > 6d$ | quyết định có cần attention thưa không |
| KV cache mỗi token | $2 L n_{kv} d_{\text{head}} \times$ byte | ước lượng bộ nhớ lúc sinh |
| Bộ nhớ Adam | $2N \times$ byte, cộng bản sao FP32 | ước lượng bộ nhớ huấn luyện |

---

## 13. Bài tập

**Bài 1 (tính tay).** Tính số tham số của một mô hình kiểu Llama-2 7B: 32 lớp, $d = 4096$, $d_{\text{ff}} = 11008$, từ vựng 32 000, 32 đầu attention và 32 đầu KV, dùng RoPE (không có embedding vị trí), RMSNorm (chỉ có $\gamma$), không có độ lệch, FFN kiểu SwiGLU, lớp ra **không** dùng chung trọng số với embedding. Đối chiếu với con số đã công bố 6 738 415 616. Tỉ lệ $d_{\text{ff}}/d$ bằng bao nhiêu, và vì sao nó không phải 4?
*Gợi ý: SwiGLU có ba ma trận chứ không phải hai. So $d_{\text{ff}}$ với $\tfrac{8}{3}d$.*

**Bài 2 (tính tay).** Dùng kết quả Bài 1, tính tổng FLOP để huấn luyện mô hình ấy trên 2 nghìn tỉ token. Nếu chạy trên A100 (312 TFLOPS ở bf16) với hiệu suất sử dụng 37,6%, mất bao nhiêu giờ-GPU? Đối chiếu với con số Meta công bố là 184 320 giờ-GPU và cho biết phép ước lượng lệch bao nhiêu phần trăm.
*Gợi ý: $C \approx 6ND$ với $N$ là số tham số **phi-embedding**.*

**Bài 3 (suy luận).** Chứng minh rằng nếu các thành phần của $q$ và $k$ độc lập, trung bình 0, phương sai 1 thì $\operatorname{Var}(q \cdot k) = d_k$. Rồi giải thích, bằng đạo hàm của softmax, vì sao phương sai lớn làm gradient gần như biến mất. Cuối cùng: nếu ai đó đề xuất chia cho $d_k$ thay vì $\sqrt{d_k}$ thì điều gì xảy ra?

**Bài 4 (tính tay).** Một CNN chỉ gồm các lớp tích chập $3\times3$, bước nhảy 1, không gộp.
(a) Cần bao nhiêu lớp để trường tiếp nhận phủ hết ảnh $224 \times 224$?
(b) Nếu cứ hai lớp lại có một lớp bước nhảy 2, tính trường tiếp nhận sau 10 lớp.
(c) Giải thích vì sao trường tiếp nhận **hiệu dụng** nhỏ hơn con số lý thuyết.
*Gợi ý: $r_i = r_{i-1} + (k_i - 1)\prod_{j<i} s_j$.*

**Bài 5 (chẩn đoán).** Với mỗi tình huống, cho biết đó là vấn đề thiên lệch hay phương sai, và nêu **hai** việc nên làm:
(a) Sai số huấn luyện 2%, sai số kiểm định 18%, khoảng cách chưa khép khi thêm dữ liệu.
(b) Sai số huấn luyện 24%, sai số kiểm định 25%, cả hai đã phẳng.
(c) Sai số kiểm định 12%, sai số huấn luyện 15%.

**Bài 6 (suy luận).** Hàm lều $g(x) = 2\,\text{ReLU}(x) - 4\,\text{ReLU}(x - \tfrac12)$ dùng đúng 2 đơn vị ReLU. Chứng minh $g$ ánh xạ $[0,1]$ lên $[0,1]$ và hợp $k$ lần cho một sóng răng cưa $2^k$ đoạn tuyến tính. Suy ra số đơn vị mà mạng **một lớp ẩn** cần để biểu diễn hàm ấy, và so với $2k$ đơn vị của mạng sâu.

**Bài 7 (suy luận).** Cho một RNN có $W_{hh}$ với bán kính phổ $\rho$.
(a) Ước lượng gradient còn lại sau 100 bước truyền ngược khi $\rho = 0{,}9$ và khi $\rho = 1{,}1$.
(b) Với LSTM, gradient dọc đường ô nhớ là $\prod f_i$. Nếu độ lệch cổng quên đặt sao cho $f \approx 0{,}73$, sau 100 bước còn lại bao nhiêu? Nếu $f \approx 0{,}98$?
(c) Vì sao bùng nổ dễ chữa hơn tiêu biến?

**Bài 8 (tính tay).** Một mô hình 70 tỉ tham số có 80 lớp, $d_{\text{head}} = 128$, 64 đầu query. Tính dung lượng KV cache ở FP16 cho ngữ cảnh 4096 token, batch 8, trong hai trường hợp: (a) MHA với 64 đầu KV; (b) GQA với 8 đầu KV. Nêu tỉ lệ giảm và giải thích vì sao GQA là quyết định về **bộ nhớ lúc suy luận** chứ không phải về chất lượng.

**Bài 9 (thiết kế).** Với mỗi bài toán, chọn họ mô hình và nêu **lý do dựa trên giả thiết quy nạp**, không dựa trên độ phổ biến:
(a) Dự đoán rời bỏ từ 40 đặc trưng bảng, 30 000 khách hàng, cần giải thích được cho bộ phận kinh doanh.
(b) Phân loại ảnh khuyết tật trên dây chuyền, 3 000 ảnh có nhãn, ảnh rất giống nhau về bố cục.
(c) Trích xuất trường thông tin từ hợp đồng dài 50 trang.
(d) Dự báo phụ tải điện theo giờ cho 12 tháng tới.

**Bài 10 (thí nghiệm).** Trong phần (E) của `code/models/experiments.py`, thay softmax bằng một hàm không bão hoà (ví dụ chuẩn hoá $\ell_1$ của ReLU) rồi đo lại entropy theo $d_k$ khi **không** chia $\sqrt{d_k}$. Hiện tượng bão hoà có biến mất không? Từ đó rút ra: phép chia $\sqrt{d_k}$ là để chữa vấn đề của **softmax** hay của **tích vô hướng**?

---

## 14. Ôn phỏng vấn

### 14.1. Khung trả lời

Câu hỏi về mô hình và kiến trúc hầu hết có dạng "vì sao X". Khung ba bước dùng được cho gần hết:

1. **Nêu vấn đề mà X sinh ra để chữa.** Không có X thì hỏng chuyện gì?
2. **Nêu cơ chế.** X chữa bằng cách nào — tốt nhất là một dòng toán hoặc một con số.
3. **Nêu cái giá.** X đắt ở đâu, hỏng khi nào.

Bước 3 là bước phân biệt rõ nhất. Người học thuộc chỉ nói được bước 1 và 2.

### 14.2. Nhóm nền tảng

**"Giải thích đánh đổi thiên lệch – phương sai."**

> Sai số kỳ vọng tách thành ba phần: thiên lệch² (mô hình *trung bình* đã sai vì giả thiết quá hẹp), phương sai (mô hình đổi nhiều khi đổi tập huấn luyện), và nhiễu không giảm được.
>
> *Ghi điểm thêm:* nhấn rằng kỳ vọng lấy **trên các tập huấn luyện**, và nêu giá trị thực tế của phân rã — nó phân biệt *"cần mô hình mạnh hơn"* với *"cần nhiều dữ liệu hơn"*, hai kết luận trái ngược. Kể một con số: trong thí nghiệm ở Mục 2.3, đa thức bậc 1 có thiên lệch² chiếm 74% MSE nên thêm dữ liệu vô ích; bậc 12 có phương sai chiếm gần như toàn bộ nên thêm dữ liệu thì cứu được.

**"Vì sao mạng nơ-ron cần hàm kích hoạt phi tuyến?"**

> Vì hợp của hai phép biến đổi affine vẫn là một phép affine: $W_2(W_1x + b_1) + b_2 = (W_2W_1)x + (\ldots)$. Không có phi tuyến thì mọi mạng sâu sụp về một lớp tuyến tính duy nhất.

**"Sâu hơn hay rộng hơn?"**

> Định lý xấp xỉ phổ quát nói một lớp ẩn đủ rộng là đủ, nhưng nó chỉ nói **tồn tại**, không nói cần bao nhiêu nơ-ron.
>
> *Kể một con số:* trong thí nghiệm ở Mục 4.4, một mạng $k$ lớp với 2 đơn vị mỗi lớp biểu diễn chính xác một sóng răng cưa $2^k$ đoạn, dùng $6k$ tham số. Mạng một lớp ẩn cần bề rộng **đúng bằng $2^k$**, tức $3 \cdot 2^k + 2$ tham số. Ở $k = 7$ là 42 so với 386 tham số. Đó là **tuyến tính so với hàm mũ** — "độ sâu mua được sự gọn gàng".
>
> *Cái giá:* mạng sâu khó tối ưu hơn, và cần kết nối tắt cùng chuẩn hoá mới huấn luyện được.

### 14.3. Nhóm huấn luyện

**"Vì sao gradient tiêu biến, và chữa thế nào?"**

> Vì gradient qua $L$ lớp là một tích $L$ ma trận Jacobi; hệ số trung bình lệch khỏi 1 một chút là sau vài chục lớp tích co về 0 hoặc bùng nổ.
>
> *Kể một con số:* trong thí nghiệm ở Mục 6.3, cùng một mạng 40 lớp, chỉ đổi hệ số khởi tạo từ 0,5 sang 2,0 làm gradient ở lớp đầu đi từ $10^{-18}$ tới $10^{6}$ — trải 24 bậc độ lớn.
>
> *Ba cách chữa, và chúng cộng dồn:* khởi tạo đúng phương sai (He cho ReLU: $2/n_{\text{in}}$, hệ số 2 đến từ việc ReLU vứt một nửa), chuẩn hoá, và kết nối tắt.
>
> *Ghi điểm thêm — chi tiết hay bị nói sai:* **kết nối tắt một mình thì làm bùng nổ**, vì mỗi lớp cộng thêm vào tín hiệu. Đo được $2{,}4 \times 10^{8}$ sau 40 lớp. Phải **kết hợp với chuẩn hoá** mới ổn định (đo được 2,22). Đó đúng là cấu hình pre-LN của Transformer hiện đại.

**"Vì sao hệ số 2 trong khởi tạo He?"** — Vì ReLU đưa một nửa số giá trị về 0, nên nó cắt phương sai đi một nửa; nhân đôi phương sai trọng số để bù lại. Với tanh thì không cần, nên dùng Xavier.

**"BatchNorm khác LayerNorm thế nào, vì sao Transformer dùng LayerNorm?"**

> BatchNorm lấy thống kê **trên cả lô** cho mỗi kênh; LayerNorm lấy **trên các đặc trưng** cho mỗi mẫu.
>
> *Ba lý do Transformer chọn LayerNorm:* độ dài chuỗi thay đổi nên thống kê theo lô trên trục thời gian vô nghĩa; BatchNorm có hành vi **khác nhau giữa huấn luyện và suy luận** mà lúc sinh từng token thì lô hiệu dụng bằng 1; và BatchNorm làm dự đoán của một mẫu phụ thuộc các mẫu khác trong lô.
>
> *Ghi điểm thêm:* RMSNorm bỏ luôn bước trừ trung bình, rẻ hơn và gần như không mất chất lượng — là mặc định hiện nay.

**"Adam khác SGD thế nào? Vì sao có hiệu chỉnh thiên lệch?"**

> Adam giữ trung bình trượt của gradient ($m$) và của bình phương gradient ($v$), rồi chia — tức **mỗi tham số một tốc độ học riêng**.
>
> *Hiệu chỉnh thiên lệch:* vì $m_0 = v_0 = 0$, các bước đầu bị kéo về 0 một cách giả tạo — với $\beta_1 = 0{,}9$ thì $m_1$ chỉ bằng 10% của $g_1$. Chia cho $1 - \beta^t$ bù đúng phần khuyết ấy và tắt dần khi $t$ lớn.
>
> *Cái giá:* Adam giữ **hai** trạng thái mỗi tham số, nên bộ nhớ tối ưu hoá gấp đôi bộ nhớ trọng số. Với mô hình 7B ở FP32 đó là 56 GB.
>
> *Ghi điểm thêm:* AdamW tách suy giảm trọng số ra khỏi gradient, vì với Adam thường thì phạt $\ell_2$ bị chia cho $\sqrt{\hat v}$ nên tham số có gradient lớn lại bị phạt ít — ngược ý định.

### 14.4. Nhóm Transformer

**"Giải thích attention."**

> Là phép tra cứu từ điển lấy đạo hàm được: truy vấn $Q$ hỏi, khoá $K$ mô tả từng vị trí có gì, giá trị $V$ là nội dung. Độ giống $QK^\top$ đi qua softmax thành trọng số, rồi lấy tổ hợp có trọng số của $V$.
>
> *Ghi điểm thêm:* attention là **phần duy nhất trong khối Transformer trộn thông tin giữa các vị trí**; FFN xử lý từng vị trí hoàn toàn độc lập. Bỏ attention đi thì Transformer chỉ còn là một MLP áp cho từng token.

**"Vì sao chia cho $\sqrt{d_k}$?"** — Câu này được hỏi nhiều nhất.

> Nếu các thành phần của $q$, $k$ độc lập, trung bình 0, phương sai 1 thì $q \cdot k$ có **phương sai đúng bằng $d_k$**. Điểm số lớn đẩy softmax vào vùng bão hoà, nơi gradient gần 0. Chia cho $\sqrt{d_k}$ đưa phương sai về 1 bất kể $d_k$.
>
> *Kể một con số:* đo được ở Mục 9.2 — với $d_k = 1024$, phương sai đo được là 1022 (lý thuyết 1024); entropy của attention rơi xuống **0,118 nat** khi không chia, trong khi mức tối đa là $\ln 64 = 4{,}159$; trọng số lớn nhất trung bình là 0,953, tức gần như one-hot. Có chia thì entropy đứng yên ở 3,68 với mọi $d_k$.

**"Nhiều đầu attention mua được gì?"**

> Không mua sức tính — bài báo gốc nêu rõ tổng chi phí **xấp xỉ bằng** attention một đầu ở chiều đầy đủ, vì mỗi đầu hẹp đi đúng $h$ lần. Nó mua **sự đa dạng của quan hệ**: cho phép chú ý tới nhiều không gian biểu diễn khác nhau ở những vị trí khác nhau, điều mà một đầu duy nhất sẽ làm nhoè đi vì lấy trung bình.

**"Vì sao Transformer thay được RNN?"**

> Không phải vì rẻ hơn — attention tốn $O(T^2 d)$ còn RNN tốn $O(Td^2)$. Lý do là **số bước tuần tự**: RNN cần $O(T)$ bước không song song hoá được, attention cần $O(1)$. Và đường đi giữa hai vị trí bất kỳ là $O(1)$ thay vì $O(T)$, nên gradient không phải đi qua $T$ phép nhân.

**"Vì sao cần mã hoá vị trí?"** — Vì self-attention **hoán vị bất biến**: đảo thứ tự token thì đầu ra chỉ đảo theo. Với ngôn ngữ thì đó là tai hoạ.

**"RoPE hoạt động thế nào?"**

> Thay vì cộng vị trí vào embedding, nó **xoay** từng cặp chiều của $Q$ và $K$ một góc tỉ lệ với vị trí, với tần số $\theta_i = 10000^{-2i/d}$. Tính chất then chốt: $\langle R_m q, R_n k\rangle = \langle R_{m-n}q, k\rangle$ — điểm số **chỉ phụ thuộc khoảng cách tương đối**.
>
> *Kể một con số:* đo được ở Mục 10.3 — cặp $(5,2)$, $(105,102)$ và $(500,497)$ đều cho **đúng cùng một giá trị 6,851342**. Và độ tương đồng của cùng một vector ở hai vị trí suy giảm 1,000 → 0,439 khi khoảng cách đi từ 0 tới 64.
>
> *Ba món lợi:* quan hệ tương đối miễn phí, ngoại suy độ dài tốt hơn, và chỉ chạm $Q$, $K$ nên không thêm tham số.

**"GQA là gì, vì sao dùng?"**

> Nhiều đầu query dùng chung một cặp $K$, $V$. **Lý do là bộ nhớ lúc suy luận**, không phải chất lượng: KV cache giảm đúng theo tỉ lệ nhóm.
>
> *Kể một con số:* mô hình 70B, ngữ cảnh 4096, batch 8, FP16 — MHA với 64 đầu KV cần **80 GiB**, GQA với 8 đầu KV cần **10 GiB**. Đúng 8 lần.

**"Vì sao $d_{\text{ff}}$ của các mô hình mới không phải $4d$?"**

> Vì SwiGLU dùng **ba** ma trận thay vì hai. Để giữ nguyên số tham số và số phép tính, bề rộng ẩn phải giảm còn $2/3$, tức $d_{\text{ff}} \approx \tfrac{8}{3}d$.
>
> *Kiểm chứng:* Llama-2 7B có $d = 4096$ nên $\tfrac{8}{3}d = 10\,923$, và giá trị thật là **11 008** — đúng là 10 923 làm tròn lên bội của 256.

### 14.5. Nhóm tính toán

**"Một mô hình $L$ lớp, chiều $d$ có bao nhiêu tham số?"**

> $12 L d^2$ cho phần phi-embedding: $4d^2$ cho attention ($W_Q, W_K, W_V, W_O$) và $8d^2$ cho FFN với $d_{\text{ff}} = 4d$. Cộng $Vd$ cho embedding.
>
> *Ghi điểm thêm:* nêu rằng **FFN chiếm hai phần ba** tham số của khối, và rằng với mô hình nhỏ thì embedding chi phối — ở GPT-2 small nó là **31%** tổng tham số, đó là lý do các bài báo về quy luật co giãn tách riêng phần phi-embedding.

**"Huấn luyện tốn bao nhiêu FLOP?"**

> $C \approx 6ND$: mỗi token tốn $2N$ cho lượt xuôi và $4N$ cho lượt ngược.
>
> *Kể một con số kiểm chứng:* Llama-2 7B có $N = 6{,}48$ tỉ tham số phi-embedding, huấn luyện trên 2 nghìn tỉ token, nên $C = 7{,}77 \times 10^{22}$ FLOP. Trên A100 ở 37,6% hiệu suất sử dụng, đó là **184 011 giờ-GPU** — so với **184 320 giờ-GPU** mà Meta công bố, tức lệch 0,17%.

**"Khi nào chi phí bậc hai của attention mới thành vấn đề?"**

> Tỉ lệ giữa chi phí attention và phần còn lại là $T/(6d)$. Với $d = 4096$, attention chỉ chi phối khi $T > 24\,576$. Ở ngữ cảnh 4096 nó mới chiếm 17%.
>
> Đây là câu trả lời định lượng cho "vì sao chưa cần attention thưa": với phần lớn hệ thống hiện nay thì $T$ chưa vượt $6d$.

**"Vì sao hết bộ nhớ khi huấn luyện?"** — Vì bộ nhớ kích hoạt tỉ lệ với **kích thước lô × độ dài chuỗi × số lớp**, chứ không tỉ lệ với số tham số. Đó là lý do cách chữa đầu tiên là giảm kích thước lô, rồi tới gradient checkpointing (đổi khoảng 30% thời gian lấy bộ nhớ từ $O(L)$ xuống $O(\sqrt{L})$).

### 14.6. Những câu trả lời tự tố cáo

| Câu trả lời | Vì sao nó tố cáo |
|---|---|
| "Chia $\sqrt{d_k}$ để ổn định huấn luyện." | Đúng nhưng rỗng. Không nói được phương sai bằng $d_k$ là chưa hiểu. |
| "Nhiều đầu để mô hình mạnh hơn." | Không nói được rằng tổng chi phí **không đổi** và cái mua được là sự đa dạng. |
| "ResNet chữa quá khớp." | Sai. Nó sinh ra để chữa **vấn đề suy thoái**: sai số *huấn luyện* tăng theo độ sâu. |
| "Kết nối tắt chữa gradient tiêu biến." | Chưa đủ — một mình nó làm **bùng nổ**; phải kèm chuẩn hoá. |
| "Càng nhiều tham số càng quá khớp." | Đúng ở chế độ cổ điển; chế độ quá tham số có double descent. |
| "Cây cần chuẩn hoá đặc trưng." | Cây bất biến với mọi phép biến đổi đơn điệu từng đặc trưng. |
| "Adam luôn tốt hơn SGD." | Adam hội tụ nhanh hơn; SGD + momentum thường tổng quát hoá tốt hơn trên thị giác. |
| "Transformer nhanh hơn RNN." | Về **FLOP** thì không. Cái nó thắng là **số bước tuần tự**. |
| "Dropout luôn cần." | Nhiều LLM lớn đặt dropout bằng 0, vì quá khớp không phải vấn đề chính ở quy mô đó. |
| "GQA để tăng chất lượng." | Nó **giảm** chất lượng một chút; lý do dùng là bộ nhớ KV cache. |

---

## 15. Tài liệu tham khảo

**Nền tảng**

1. T. Hastie, R. Tibshirani, J. Friedman. *The Elements of Statistical Learning.* Springer, 2009. — nguồn chuẩn cho phân rã thiên lệch–phương sai, cây, bagging và boosting.
2. C. Bishop. *Pattern Recognition and Machine Learning.* Springer, 2006.
3. I. Goodfellow, Y. Bengio, A. Courville. *Deep Learning.* MIT Press, 2016. — nguồn chuẩn cho lan truyền ngược, khởi tạo, tối ưu hoá.
4. L. Breiman. *Random Forests.* Machine Learning, 2001. — nguồn của lập luận về tương quan giữa các cây.
5. J. Friedman. *Greedy Function Approximation: A Gradient Boosting Machine.* Annals of Statistics, 2001.
6. T. Chen, C. Guestrin. *XGBoost: A Scalable Tree Boosting System.* KDD 2016.

**Huấn luyện mạng sâu**

7. X. Glorot, Y. Bengio. *Understanding the difficulty of training deep feedforward neural networks.* AISTATS 2010. — khởi tạo Xavier.
8. K. He và cộng sự. *Delving Deep into Rectifiers.* ICCV 2015. — khởi tạo He và hệ số 2 cho ReLU.
9. S. Ioffe, C. Szegedy. *Batch Normalization.* ICML 2015.
10. J. L. Ba, J. Kiros, G. Hinton. *Layer Normalization.* arXiv:1607.06450, 2016.
11. B. Zhang, R. Sennrich. *Root Mean Square Layer Normalization.* NeurIPS 2019. — RMSNorm.
12. K. He và cộng sự. *Deep Residual Learning for Image Recognition.* CVPR 2016. — nguồn của **vấn đề suy thoái** và kết nối tắt.
13. D. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015. — nguồn của hiệu chỉnh thiên lệch.
14. I. Loshchilov, F. Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019. — AdamW.
15. N. Srivastava và cộng sự. *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.* JMLR 2014.

**Tích chập và hồi quy**

16. K. Simonyan, A. Zisserman. *Very Deep Convolutional Networks.* ICLR 2015. — lập luận hai lớp $3\times3$ thay một lớp $5\times5$.
17. A. Howard và cộng sự. *MobileNets.* arXiv:1704.04861, 2017. — depthwise separable.
18. W. Luo và cộng sự. *Understanding the Effective Receptive Field in Deep CNNs.* NeurIPS 2016. — nguồn của kết quả trường tiếp nhận hiệu dụng tăng theo $O(\sqrt{L})$.
19. S. Hochreiter, J. Schmidhuber. *Long Short-Term Memory.* Neural Computation, 1997.
20. R. Pascanu, T. Mikolov, Y. Bengio. *On the difficulty of training Recurrent Neural Networks.* ICML 2013. — bán kính phổ, cắt ngưỡng gradient.
21. R. Jozefowicz, W. Zaremba, I. Sutskever. *An Empirical Exploration of Recurrent Network Architectures.* ICML 2015. — nguồn của thủ thuật khởi tạo độ lệch cổng quên dương.

**Transformer**

22. A. Vaswani và cộng sự. *Attention Is All You Need.* NeurIPS 2017. — nguồn của scaled dot-product attention, lập luận $\sqrt{d_k}$ (chú thích 4), multi-head, và bảng so sánh số bước tuần tự.
23. J. Su và cộng sự. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864. — nguồn của RoPE, $\theta_i = 10000^{-2i/d}$ và tính chất suy giảm theo khoảng cách.
24. N. Shazeer. *GLU Variants Improve Transformer.* arXiv:2002.05202, 2020. — nguồn của SwiGLU và của việc giảm $d_{\text{ff}}$ còn $2/3$ để khớp tham số.
25. N. Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need.* arXiv:1911.02150, 2019. — MQA.
26. J. Ainslie và cộng sự. *GQA: Training Generalized Multi-Query Transformer Models.* EMNLP 2023.
27. R. Xiong và cộng sự. *On Layer Normalization in the Transformer Architecture.* ICML 2020. — phân tích pre-LN so với post-LN.
28. J. Kaplan và cộng sự. *Scaling Laws for Neural Language Models.* arXiv:2001.08361, 2020. — nguồn của $N = 12 L d^2$, $C_{\text{xuôi}} \approx 2N$ và $C \approx 6NBS$.
29. J. Hoffmann và cộng sự. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — Chinchilla.
30. A. Dosovitskiy và cộng sự. *An Image is Worth 16x16 Words.* ICLR 2021. — ViT và đánh đổi thiên lệch quy nạp.
31. H. Touvron và cộng sự. *Llama 2: Open Foundation and Fine-Tuned Chat Models.* arXiv:2307.09288, 2023. — nguồn của cấu hình và con số 184 320 giờ-GPU dùng để đối chiếu ở Mục 12.4.

**Giải mã**

32. A. Holtzman và cộng sự. *The Curious Case of Neural Text Degeneration.* ICLR 2020. — nguồn của top-$p$ và của lập luận vì sao beam search cho văn bản nhạt.
33. Y. Leviathan, M. Kalman, Y. Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023.

**Chế độ quá tham số**

34. M. Belkin và cộng sự. *Reconciling modern machine-learning practice and the classical bias–variance trade-off.* PNAS, 2019. — double descent.

---

## Phụ lục: chạy lại toàn bộ thí nghiệm

```text
code/models/
├── experiments.py            # Hình 2, 4, 5, 7, 9, 10, 13, 14 và mọi số liệu đo được
├── experiments_output.txt    # kết quả in ra của script trên
└── fig_diagrams.py           # Hình 1, 3, 6, 8, 11, 12 (các sơ đồ khái niệm)
```

```bash
pip install numpy scipy matplotlib scikit-learn
python code/models/experiments.py     # vài phút
python code/models/fig_diagrams.py
```

Hai script đặt hạt giống cố định nên mọi con số trong tài liệu lặp lại được y hệt trên cùng phiên bản thư viện. Môi trường đã dùng: Python 3.13, NumPy 2.3, SciPy 1.16, scikit-learn 1.7, matplotlib 3.10.

Bốn kết quả trong tài liệu là **đối chiếu với con số đã công bố**, không phải mô phỏng: số tham số của GPT-2 small/medium/large, số tham số của Llama-2 7B/13B, ngân sách giờ-GPU của Llama-2 7B, và phương sai của tích vô hướng trong chú thích 4 của bài báo Transformer. Bốn kết quả ấy là phép kiểm tra rằng các công thức trong Chương 12 đúng chứ không chỉ hợp lý.
