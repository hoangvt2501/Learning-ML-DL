# Nền tảng Machine Learning: từ đại số tuyến tính tới SVM

> **Giáo trình tự học, viết theo lối bài giảng.** Mỗi khái niệm đi theo trình tự *động cơ → định nghĩa → suy luận → ví dụ số → thí nghiệm kiểm chứng*, giống ba giáo trình còn lại trong repo này.
>
> **Đây là phần đứng trước.** Giáo trình *Mô hình & Kiến trúc* mở đầu bằng đánh đổi thiên lệch – phương sai, tức đã giả định người đọc biết sẵn hồi quy tuyến tính, hồi quy logistic, gradient descent và xác suất. Tài liệu này lấp đúng khoảng trống ấy, và dừng lại đúng chỗ giáo trình kia bắt đầu.
>
> **Về độ tin cậy của số liệu.** Mọi con số trong tài liệu đều **đo được trong chính repo này**, sinh bởi hai script trong `code/nentang/`, hạt giống cố định, chạy lại cho kết quả y hệt. Một số kết quả không phải mô phỏng mà là **kiểm chứng một đẳng thức** — ridge đúng bằng MAP với tiên nghiệm Gauss, sai số tái dựng của PCA đúng bằng tổng các trị riêng bị bỏ, hồi quy logistic đúng bằng softmax với $K = 2$. Những chỗ ấy phải khớp tới sai số máy, và tài liệu in ra sai số thật để người đọc tự kiểm.
>
> **Tài liệu này dạy cái gì.** Không dạy dùng thư viện. Nó trả lời câu hỏi: *mỗi thuật toán cổ điển giả định điều gì, giả định ấy hỏng khi nào, và vì sao người ta vẫn dùng nó.* Nắm được điều đó thì học mạng nơ-ron về sau chỉ còn là thay một trong ba thành phần ở Chương 1.

---

## Mục lục

0. [Kiến thức nền và quy ước](#0-kiến-thức-nền-và-quy-ước)
1. [Bài toán học máy là gì](#1-bài-toán-học-máy-là-gì)
2. [Đại số tuyến tính cho học máy](#2-đại-số-tuyến-tính-cho-học-máy)
3. [Xác suất và thống kê cho học máy](#3-xác-suất-và-thống-kê-cho-học-máy)
4. [Hồi quy tuyến tính](#4-hồi-quy-tuyến-tính)
5. [Gradient descent](#5-gradient-descent)
6. [Phân loại tuyến tính: perceptron, logistic, softmax](#6-phân-loại-tuyến-tính-perceptron-logistic-softmax)
7. [K láng giềng gần nhất và Naive Bayes](#7-k-láng-giềng-gần-nhất-và-naive-bayes)
8. [Đánh giá mô hình phân loại](#8-đánh-giá-mô-hình-phân-loại)
9. [Quá khớp, phạt chuẩn và kiểm định chéo](#9-quá-khớp-phạt-chuẩn-và-kiểm-định-chéo)
10. [Hợp lý cực đại và hậu nghiệm cực đại](#10-hợp-lý-cực-đại-và-hậu-nghiệm-cực-đại)
11. [Tập lồi và hàm lồi](#11-tập-lồi-và-hàm-lồi)
12. [Tối ưu lồi và đối ngẫu Lagrange](#12-tối-ưu-lồi-và-đối-ngẫu-lagrange)
13. [Support Vector Machine](#13-support-vector-machine)
14. [Giảm chiều: PCA, SVD và LDA](#14-giảm-chiều-pca-svd-và-lda)
15. [Phân cụm: K-means](#15-phân-cụm-k-means)
16. [Hệ gợi ý](#16-hệ-gợi-ý)
17. [Bài tập](#17-bài-tập)
18. [Ôn phỏng vấn](#18-ôn-phỏng-vấn)
19. [Tài liệu tham khảo](#19-tài-liệu-tham-khảo)

---

## 0. Kiến thức nền và quy ước

Người đọc cần biết: đạo hàm một biến, nhân ma trận ở mức cơ học, và Python/NumPy ở mức đọc hiểu. **Không** cần biết trước xác suất hay đại số tuyến tính ở mức nâng cao — Chương 2 và Chương 3 ôn lại đúng phần cần dùng, và không ôn thừa.

| Ký hiệu | Ý nghĩa |
|---|---|
| $x \in \mathbb{R}^{d}$ | một mẫu đầu vào, $d$ chiều |
| $X \in \mathbb{R}^{n \times d}$ | ma trận dữ liệu: $n$ mẫu xếp theo hàng |
| $y_i$, $\hat{y}_i$ | nhãn thật và nhãn dự đoán của mẫu thứ $i$ |
| $w$, $b$ | vector trọng số và độ lệch |
| $\theta$ | gọi chung mọi tham số của mô hình |
| $L(\theta)$ | hàm mất mát |
| $\eta$ | tốc độ học |
| $\lambda$ | hệ số phạt chuẩn |
| $\nabla_\theta L$ | gradient của $L$ theo $\theta$ |
| $\Sigma$ | ma trận hiệp phương sai |
| $\lambda_i$, $v_i$ | trị riêng và vector riêng thứ $i$ |
| $d_i$ | trị kỳ dị thứ $i$ trong phân rã SVD |
| $\kappa$ | số điều kiện |
| $\alpha_i$ | nhân tử Lagrange ứng với ràng buộc thứ $i$ |
| $K$ | số lớp trong bài toán phân loại |
| $k$ | số cụm, số láng giềng, hoặc số chiều giữ lại — tuỳ ngữ cảnh, luôn nói rõ |

**Một quy ước về chữ, và nó quan trọng.** Tiếng Anh dùng chung chữ *bias* cho hai thứ khác hẳn nhau. Tài liệu này phân biệt rạch ròi:

- **thiên lệch** — bias theo nghĩa thống kê: sai lệch có hệ thống giữa ước lượng trung bình và giá trị thật.
- **độ lệch** — bias theo nghĩa tham số cộng thêm $b$ trong $w^\top x + b$.

Ngoài ra: *phạt chuẩn* cho regularization, *quá khớp* cho overfitting, *thiếu khớp* cho underfitting, *kiểm định chéo* cho cross-validation, *lề* cho margin, *nhân* cho kernel, *hàm hợp lý* cho likelihood, *tiên nghiệm* / *hậu nghiệm* cho prior / posterior, *trị riêng* / *vector riêng* cho eigenvalue / eigenvector, *trị kỳ dị* cho singular value. Khi một thuật ngữ đã thành tên riêng (softmax, sigmoid, hinge, kernel RBF, bootstrap) thì giữ nguyên. Toàn bộ đối chiếu Việt – Anh nằm ở trang **Từ điển thuật ngữ**.

**Quy ước về chiều.** Ma trận dữ liệu $X$ có $n$ hàng (mẫu) và $d$ cột (đặc trưng) — đúng như `X.shape` trong NumPy. Công thức toán viết $w^\top x$ với $x$ là vector cột; trong mã thì thành `X @ w`. Hai cách chỉ khác phép chuyển vị, chỗ nào dễ nhầm thì có ghi rõ.

**Quy ước về dấu phẩy thập phân.** Phần chữ dùng dấu phẩy theo lối Việt (0,15); phần mã và phần in ra từ script giữ dấu chấm theo lối Python (0.15).

---

## 1. Bài toán học máy là gì

### 1.1. Vì sao chương này đứng đầu

Người mới học thường tiếp cận học máy như một danh sách thuật toán cần thuộc: hồi quy tuyến tính, SVM, cây quyết định, mạng nơ-ron. Danh sách ấy dài ra mỗi năm và không bao giờ học hết.

Có một cách nhìn khác, gọn hơn nhiều và không cũ đi. **Mọi thuật toán học có giám sát đều gồm đúng ba thứ**, và chỉ ba:

![Hình 1](figs/nt01_bando.png)

**Hình 1.** Ba thành phần, và năm thuật toán đọc theo đúng ba cột ấy. Đổi một cột là ra một thuật toán khác.

1. **Mô hình** — tập các hàm $f_\theta$ mà ta cho phép mình dùng. Hồi quy tuyến tính cho phép các hàm affine; cây quyết định cho phép các hàm hằng từng khúc; mạng nơ-ron cho phép hợp của nhiều lớp affine và phi tuyến.
2. **Hàm mất mát** — định nghĩa thế nào là đoán sai, và sai bao nhiêu. Đây là chỗ ta phát biểu cái mình thực sự muốn.
3. **Thuật toán tối ưu** — cách đi tìm $\theta$ làm tổng mất mát nhỏ nhất.

Giá trị của cách nhìn này là nó biến việc học một thuật toán mới thành việc trả lời ba câu hỏi, và biến rất nhiều "thuật toán khác nhau" thành cùng một thứ nhìn từ góc khác. Chương 6 sẽ cho thấy perceptron, hồi quy logistic và SVM **dùng chung một mô hình** (hàm tuyến tính) và **chung một thuật toán tối ưu** (xuống dốc); chúng chỉ khác nhau ở cột thứ hai.

### 1.2. Học có giám sát, không giám sát, và ranh giới giữa chúng

| | Có nhãn $y$ | Không có nhãn |
|---|---|---|
| **Đoán một số** | hồi quy (Chương 4) | — |
| **Đoán một lớp** | phân loại (Chương 6, 7, 13) | phân cụm (Chương 15) |
| **Tìm cấu trúc** | LDA (Mục 14.4) | PCA, SVD (Chương 14) |
| **Điền chỗ trống** | — | phân rã ma trận (Chương 16) |

Ranh giới này không cứng như bảng gợi ý. Hệ gợi ý ở Chương 16 có "nhãn" là các ô đánh giá đã quan sát được, nhưng phần lớn ma trận lại trống — nó nằm giữa hai cột. Và Chương 10 sẽ cho thấy rất nhiều thuật toán ở cả hai cột đều là **ước lượng hợp lý cực đại** dưới các giả thiết phân phối khác nhau, tức chúng có chung một gốc.

### 1.3. Ba câu hỏi phải trả lời trước khi chọn thuật toán

Trước khi nghĩ tới mô hình nào, có ba câu hỏi quyết định nhiều hơn:

**Dữ liệu có bao nhiêu mẫu so với bao nhiêu đặc trưng?** Tỉ lệ $n/d$ quyết định gần như mọi thứ. $n \gg d$ thì mô hình linh hoạt được; $n \approx d$ hoặc $n < d$ thì bắt buộc phải phạt chuẩn mạnh (Chương 9), và một số phương pháp sụp đổ hẳn — Mục 4.3 cho thấy phương trình chuẩn tắc không giải được khi các cột phụ thuộc tuyến tính.

**Quan hệ cần học có gì đặc biệt về cấu trúc?** Ảnh có tính cục bộ và bất biến dịch chuyển; chuỗi có thứ tự; dữ liệu bảng thì không có gì cả. Đây là chỗ mà mô hình tuyến tính và cây quyết định thường thắng mạng nơ-ron trên dữ liệu bảng — không có cấu trúc để mạng sâu khai thác.

**Ta cần dự báo hay cần giải thích?** Hai mục tiêu này khác nhau, và đôi khi ngược nhau. Mục 4.4 đo được một trường hợp rất rõ: khi hai đặc trưng gần cộng tuyến, **hệ số dao động gấp 25 lần trong khi dự báo gần như không đổi**. Nếu chỉ cần dự báo thì không việc gì; nếu cần nói "đặc trưng này ảnh hưởng bao nhiêu" thì con số ấy vô nghĩa.

### 1.4. Tài liệu này đứng ở đâu trong bộ bốn giáo trình

![Hình 13](figs/nt13_mach.png)

**Hình 13.** Bốn giáo trình trong repo đọc như một mạch. Mũi tên là thứ tự ít phải quay lại nhất, không phải thứ tự bắt buộc.

Cụ thể mối nối giữa tài liệu này và các tài liệu kia:

- **Chương 5 (gradient descent)** ở đây dựng nền cho [Chương 5 của *Mô hình & Kiến trúc*](models-ch05.html), nơi cùng một ý tưởng được mở rộng thành momentum, Adam và lan truyền ngược.
- **Chương 6 (hồi quy logistic)** là trường hợp một lớp của mạng nơ-ron: [Chương 4 của *Mô hình & Kiến trúc*](models-ch04.html) mở đầu đúng bằng câu "hồi quy logistic chồng lên nhau thì được gì".
- **Chương 9 (phạt chuẩn)** và **Chương 10 (MAP)** giải thích vì sao weight decay tồn tại, thứ mà [Chương 5 của *Mô hình & Kiến trúc*](models-ch05.html) dùng như một mặc định.
- **Chương 8 (đánh giá)** là nền cho [Chương 6 của *MLOps*](mlops-ch06.html), nơi bàn đánh giá theo lát cắt và ML Test Score.
- **Chương 14 (PCA, SVD)** cho công cụ để đọc [Chương 11 của *Quantization*](ch11.html), nơi các phương pháp như GPTQ dựa trên phân rã ma trận.

---

## 2. Đại số tuyến tính cho học máy

### 2.1. Chỉ cần đúng bằng này

Chương này không ôn lại đại số tuyến tính. Nó chỉ nêu **những thứ sẽ được dùng ở các chương sau**, cùng lý do chúng được dùng. Ai đã vững có thể đọc lướt bảng dưới rồi sang Chương 3.

| Khái niệm | Dùng ở đâu trong tài liệu này |
|---|---|
| Tích vô hướng $w^\top x$ | mọi mô hình tuyến tính (Chương 4, 6, 13) |
| Chuẩn $\|w\|_2$, $\|w\|_1$ | phạt chuẩn (Chương 9), lề SVM (Chương 13) |
| Hạng của ma trận | khi nào phương trình chuẩn tắc giải được (Mục 4.3) |
| Trị riêng, vector riêng | PCA (Chương 14), số điều kiện (Mục 5.3) |
| Ma trận xác định dương | hàm lồi bậc hai (Chương 11) |
| Phân rã SVD | PCA, ridge, phân rã ma trận (Chương 14, 16) |
| Đạo hàm theo vector | mọi phép suy ra công thức cập nhật |

### 2.2. Tích vô hướng là phép đo sự giống nhau

$$w^\top x = \sum_{j=1}^{d} w_j x_j = \|w\|\,\|x\|\cos\vartheta.$$

Hai cách đọc, và cả hai đều cần:

- **Đại số:** tổng có trọng số của các đặc trưng. Đây là cách đọc tự nhiên khi $w$ là "mức quan trọng" của từng đặc trưng.
- **Hình học:** phép chiếu của $x$ lên hướng $w$, nhân với độ dài $w$. Đây là cách đọc cần thiết để hiểu vì sao $w^\top x + b = 0$ là một **siêu phẳng** có vector pháp tuyến $w$ — nền tảng của toàn bộ Chương 13.

Khoảng cách từ điểm $x_0$ tới siêu phẳng ấy là

$$\frac{|w^\top x_0 + b|}{\|w\|}.$$

Công thức này xuất hiện lại ở Mục 13.2 và là toàn bộ lý do SVM tối thiểu hoá $\|w\|$.

### 2.3. Các chuẩn, và vì sao chọn chuẩn nào lại quan trọng

$$\|w\|_1 = \sum_j |w_j|, \qquad \|w\|_2 = \sqrt{\textstyle\sum_j w_j^2}, \qquad \|w\|_\infty = \max_j |w_j|.$$

Nghe như ba cách đo cùng một thứ, nhưng lựa chọn giữa $\ell_1$ và $\ell_2$ cho ra hai thuật toán có hành vi khác hẳn nhau. Mục 9.4 đo được điều ấy: phạt $\ell_2$ **không bao giờ** đưa một hệ số về đúng 0, còn phạt $\ell_1$ thì có — ở $\lambda = 100$ nó đưa đúng 9 trong 12 hệ số về 0, mà mô hình sinh dữ liệu cũng có đúng 9 hệ số bằng 0.

Lý do nằm ở đạo hàm tại gốc: $|w|$ có **điểm gãy** ở 0 nên nghiệm bị "dính" vào đó, còn $w^2$ trơn nên lực kéo về 0 yếu dần khi đã gần 0.

### 2.4. Trị riêng: hướng mà ma trận chỉ kéo dãn

$$A v = \lambda v.$$

Vector riêng là hướng mà phép biến đổi $A$ **không làm đổi hướng**, chỉ kéo dãn hệ số $\lambda$. Với ma trận đối xứng — và mọi ma trận hiệp phương sai đều đối xứng — ta luôn có $d$ vector riêng trực giao nhau và $d$ trị riêng thực. Đây là định lý phổ, và nó là toàn bộ nền móng của PCA.

Hai đại lượng dùng liên tục về sau:

$$\operatorname{tr}(A) = \sum_i \lambda_i, \qquad \det(A) = \prod_i \lambda_i.$$

**Số điều kiện** của một ma trận đối xứng xác định dương:

$$\kappa(A) = \frac{\lambda_{\max}}{\lambda_{\min}}.$$

Con số này quyết định gradient descent chạy nhanh hay chậm — Mục 5.3 đo được rằng số vòng lặp tỉ lệ thuận với $\kappa$, và với $\kappa = 10\,000$ thì cần **92 104 vòng** thay vì 1 vòng khi $\kappa = 1$.

### 2.5. Ma trận xác định dương

$A$ đối xứng gọi là **xác định dương** nếu $z^\top A z > 0$ với mọi $z \neq 0$; tương đương: mọi trị riêng đều dương. Ba chỗ cần đến:

1. Hàm bậc hai $f(z) = \tfrac12 z^\top A z$ **lồi** khi và chỉ khi $A$ nửa xác định dương (Chương 11).
2. $X^\top X$ luôn nửa xác định dương, và **xác định dương khi và chỉ khi $X$ đủ hạng cột** — đây đúng là điều kiện để phương trình chuẩn tắc giải được (Mục 4.3).
3. Thêm $\lambda I$ vào một ma trận nửa xác định dương làm nó thành xác định dương với mọi $\lambda > 0$. Đây là lý do toán học khiến ridge **luôn** có nghiệm duy nhất kể cả khi OLS thì không (Mục 9.2).

### 2.6. Phân rã SVD

Mọi ma trận $X \in \mathbb{R}^{n \times d}$ đều viết được thành

$$X = U D V^\top,$$

với $U$, $V$ trực giao và $D$ đường chéo chứa các **trị kỳ dị** $d_1 \geq d_2 \geq \dots \geq 0$. Ba hệ quả dùng về sau:

- $X^\top X = V D^2 V^\top$, nên **trị riêng của ma trận hiệp phương sai bằng $d_i^2/(n-1)$**. Mục 14.2 kiểm chứng điều này bằng số: hai đường tính cho kết quả lệch nhau $7{,}1 \times 10^{-15}$.
- $\kappa(X^\top X) = \kappa(X)^2$ — số điều kiện **bình phương lên** khi lập phương trình chuẩn tắc. Đây là lý do thực tế người ta giải bình phương tối thiểu bằng phân rã QR chứ không bằng $(X^\top X)^{-1}X^\top y$.
- Cắt bớt $D$ còn $k$ giá trị lớn nhất cho **xấp xỉ hạng $k$ tốt nhất** theo chuẩn Frobenius (định lý Eckart–Young). Mục 14.3 kiểm chứng đẳng thức này tới $10^{-13}$.

### 2.7. Đạo hàm theo vector: bốn công thức đủ dùng

| Hàm | Gradient |
|---|---|
| $a^\top w$ | $a$ |
| $w^\top A w$ | $(A + A^\top)w$, bằng $2Aw$ khi $A$ đối xứng |
| $\|w\|_2^2 = w^\top w$ | $2w$ |
| $\|y - Xw\|_2^2$ | $-2X^\top(y - Xw)$ |

Công thức cuối là công thức được dùng nhiều nhất trong cả tài liệu. Cho nó bằng 0 là ra ngay phương trình chuẩn tắc của Chương 4.

> **Cách kiểm tra mọi công thức đạo hàm.** Đừng tin công thức mình vừa suy ra — hãy so nó với sai phân số. Mục 5.5 nêu cách làm đúng, kể cả cái bẫy khiến chọn $\varepsilon$ **quá nhỏ** lại cho kết quả tệ hơn.

---

## 3. Xác suất và thống kê cho học máy

### 3.1. Vì sao cần xác suất

Ba lý do cụ thể, và cả ba đều được dùng ở các chương sau:

1. **Để phát biểu "mô hình không chắc chắn".** Hồi quy logistic (Chương 6) không trả về nhãn mà trả về xác suất; sự khác biệt ấy quyết định cách chọn ngưỡng ở Chương 8.
2. **Để giải thích hàm mất mát đến từ đâu.** Chương 10 cho thấy bình phương sai lệch và entropy chéo không phải hai lựa chọn tuỳ tiện — chúng là hợp lý cực đại dưới hai giả thiết nhiễu khác nhau.
3. **Để giải thích phạt chuẩn đến từ đâu.** Mục 10.4 kiểm chứng bằng số rằng ridge **đúng bằng** MAP với tiên nghiệm Gauss.

### 3.2. Kỳ vọng và phương sai

$$\mathbb{E}[X] = \sum_x x\,p(x) \;\;\text{hoặc}\;\; \int x\,p(x)\,dx, \qquad \operatorname{Var}(X) = \mathbb{E}[(X - \mathbb{E}X)^2] = \mathbb{E}[X^2] - (\mathbb{E}X)^2.$$

Bốn tính chất được dùng đi dùng lại:

$$\mathbb{E}[aX + b] = a\,\mathbb{E}[X] + b, \qquad \operatorname{Var}(aX + b) = a^2 \operatorname{Var}(X),$$
$$\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y] \;\text{(luôn đúng)}, \qquad \operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) \;\text{(chỉ khi độc lập)}.$$

Dòng cuối là chỗ hay sai nhất. Kỳ vọng thì cộng được vô điều kiện; phương sai thì **không**. Nếu $X$ và $Y$ có tương quan $\rho$ thì

$$\operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) + 2\rho\sqrt{\operatorname{Var}(X)\operatorname{Var}(Y)}.$$

Công thức này chính là toàn bộ lý thuyết của bagging — xem [Mục 3.3 của *Mô hình & Kiến trúc*](models-ch03.html), nơi nó giải thích vì sao lấy trung bình nhiều mô hình **tương quan** thì không giảm phương sai được bao nhiêu.

### 3.3. Ma trận hiệp phương sai

Với vector ngẫu nhiên $x \in \mathbb{R}^d$:

$$\Sigma = \mathbb{E}\big[(x - \mu)(x - \mu)^\top\big], \qquad \Sigma_{jk} = \operatorname{Cov}(x_j, x_k).$$

Ước lượng từ dữ liệu, với $X_c$ là ma trận đã trừ trung bình theo cột:

$$\hat\Sigma = \frac{1}{n-1} X_c^\top X_c.$$

Ba điều cần nhớ về $\Sigma$, mỗi điều ứng với một chương sau:

- Nó **đối xứng và nửa xác định dương**, nên có phân rã phổ — đó là PCA (Chương 14).
- Đường chéo là phương sai từng đặc trưng; ngoài đường chéo là hiệp phương sai.
- **Ép $\Sigma$ thành ma trận đường chéo chính là giả thiết "naive" của Naive Bayes.** Mục 7.4 kiểm chứng điều này bằng số và cho thấy Naive Bayes, LDA và QDA chỉ khác nhau đúng ở ràng buộc đặt lên $\Sigma$.

### 3.4. Phân phối Gauss

Một chiều:

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right).$$

Nhiều chiều:

$$p(x) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\!\left(-\tfrac12 (x-\mu)^\top \Sigma^{-1}(x-\mu)\right).$$

Đại lượng $(x-\mu)^\top \Sigma^{-1}(x-\mu)$ gọi là **khoảng cách Mahalanobis** bình phương. Nó là khoảng cách Euclid sau khi đã "chuẩn hoá" theo hình dạng của đám mây dữ liệu: đi xa 1 đơn vị theo hướng dữ liệu trải rộng thì rẻ, theo hướng dữ liệu bó hẹp thì đắt.

Một quan sát nhỏ nhưng dùng nhiều: **lấy logarit của mật độ Gauss thì được một hàm bậc hai**. Đó là lý do hợp lý cực đại dưới giả thiết nhiễu Gauss cho ra đúng bình phương tối thiểu (Mục 10.2), và là lý do bộ phân lớp Gauss có biên quyết định bậc hai (Mục 7.4).

### 3.5. Quy tắc Bayes

$$\underbrace{p(\theta \mid \mathcal{D})}_{\text{hậu nghiệm}} = \frac{\overbrace{p(\mathcal{D} \mid \theta)}^{\text{hợp lý}} \; \overbrace{p(\theta)}^{\text{tiên nghiệm}}}{\underbrace{p(\mathcal{D})}_{\text{bằng chứng}}}.$$

Bốn chữ này là xương sống của Chương 10. Điểm cần nhớ ngay: mẫu số $p(\mathcal{D})$ **không phụ thuộc $\theta$**, nên khi đi tìm $\theta$ tốt nhất thì bỏ được. Vì vậy

$$\arg\max_\theta p(\theta \mid \mathcal{D}) = \arg\max_\theta \; p(\mathcal{D} \mid \theta)\,p(\theta) = \arg\min_\theta \;\big[-\log p(\mathcal{D} \mid \theta) - \log p(\theta)\big].$$

Vế phải đọc là: **mất mát + phạt chuẩn**. Toàn bộ Chương 10 là khai triển của dòng này.

### 3.6. Ước lượng, thiên lệch và phương sai

Một **ước lượng** $\hat\theta$ là một hàm của dữ liệu, nên bản thân nó là biến ngẫu nhiên. Hai đại lượng mô tả nó:

$$\text{thiên lệch} = \mathbb{E}[\hat\theta] - \theta, \qquad \text{phương sai} = \operatorname{Var}(\hat\theta),$$

và sai số bình phương trung bình tách ra đúng thành

$$\mathbb{E}\big[(\hat\theta - \theta)^2\big] = \text{thiên lệch}^2 + \text{phương sai}.$$

> **Chú ý về từ "kỳ vọng" ở đây.** Kỳ vọng lấy **trên các tập dữ liệu khác nhau có thể rút ra được**, không phải trên các điểm trong một tập. Đây là chỗ nhầm phổ biến nhất khi trả lời phỏng vấn, và [Chương 2 của *Mô hình & Kiến trúc*](models-ch02.html) khai triển đầy đủ ý này cho trường hợp mô hình dự báo.

Một ví dụ cụ thể sẽ dùng lại ở Chương 9: **ridge là ước lượng có thiên lệch**, trong khi OLS không thiên lệch. Vậy mà ridge thường cho sai số nhỏ hơn — vì nó đổi một ít thiên lệch lấy rất nhiều phương sai. Mục 9.5 đo được phần phương sai ấy bị cắt ở đâu.

---

## 4. Hồi quy tuyến tính

### 4.1. Bài toán, và vì sao nó đáng học kỹ

Hồi quy tuyến tính là thuật toán đơn giản nhất có ích, nhưng lý do học nó kỹ không phải vì nó hay được dùng. Lý do là **nó có nghiệm dạng đóng**, nên mọi hiện tượng về sau — quá khớp, phạt chuẩn, số điều kiện, thiên lệch đổi lấy phương sai — đều nhìn thấy được ở đây mà không bị che bởi chuyện tối ưu hoá có hội tụ hay không.

Mô hình, hàm mất mát, và cách giải, đúng theo ba cột của Hình 1:

$$f_w(x) = w^\top x, \qquad L(w) = \frac{1}{n}\sum_{i=1}^{n}(y_i - w^\top x_i)^2 = \frac{1}{n}\|y - Xw\|_2^2.$$

Quy ước: cột đầu của $X$ là toàn số 1, nên $w_0$ chính là độ lệch. Cách này gọn hơn việc viết $b$ riêng và được dùng suốt tài liệu.

### 4.2. Nghiệm giải tích

Dùng công thức cuối trong bảng ở Mục 2.7 rồi cho gradient bằng 0:

$$\nabla_w L = -\frac{2}{n}X^\top(y - Xw) = 0 \;\Longrightarrow\; X^\top X\,w = X^\top y.$$

Đây là **phương trình chuẩn tắc**. Khi $X^\top X$ khả nghịch:

$$\boxed{\;\hat w = (X^\top X)^{-1}X^\top y\;}$$

Vì sao đây là **cực tiểu** chứ không phải cực đại hay điểm yên ngựa? Vì ma trận Hessian là $\tfrac{2}{n}X^\top X$, luôn nửa xác định dương (Mục 2.5), nên $L$ là hàm lồi và mọi điểm dừng đều là cực tiểu toàn cục. Chương 11 nói kỹ hơn, nhưng ý ấy đã đủ dùng ngay tại đây.

**Ba cách tính, một nghiệm.** Thí nghiệm trong `code/nentang/experiments.py` giải cùng một bài toán bằng ba đường khác hẳn nhau:

| Cách giải | $w_0$ | $w_1$ | $w_2$ | $w_3$ |
|---|---|---|---|---|
| Phương trình chuẩn tắc | 1,9906 | −1,4985 | 0,7992 | 2,9749 |
| Giả nghịch đảo Moore–Penrose | 1,9906 | −1,4985 | 0,7992 | 2,9749 |
| `lstsq` (phân rã QR) | 1,9906 | −1,4985 | 0,7992 | 2,9749 |

Lệch lớn nhất giữa ba cách là $1{,}2 \times 10^{-15}$ — tức bằng nhau tới sai số máy.

> **Nhưng đừng dùng công thức nghịch đảo trong mã thật.** Mục 2.6 đã nêu lý do: $\kappa(X^\top X) = \kappa(X)^2$, nên lập phương trình chuẩn tắc **bình phương** số điều kiện và mất một nửa số chữ số có nghĩa. `np.linalg.lstsq` dùng phân rã QR hoặc SVD, tránh được việc ấy. Công thức $(X^\top X)^{-1}X^\top y$ là để **suy luận**, không phải để chạy.

### 4.3. Khi nào không giải được, và làm gì

$X^\top X$ khả nghịch khi và chỉ khi $X$ đủ hạng cột, tức không cột nào là tổ hợp tuyến tính của các cột khác. Hai trường hợp hỏng thường gặp:

- **Nhiều đặc trưng hơn mẫu** ($d > n$). Khi ấy hạng tối đa là $n < d$, nên chắc chắn suy biến.
- **Đặc trưng lặp hoặc phụ thuộc tuyến tính.** Ví dụ hay gặp nhất: mã hoá one-hot đủ $K$ mức cho một biến hạng mục **và** vẫn giữ cột hằng số — tổng $K$ cột ấy đúng bằng cột hằng số.

Thí nghiệm lặp lại một cột rồi thử giải:

```text
hang cua X = 4 nhung X co 5 cot  ->  X^T X suy bien
  solve: BAO LOI 'Singular matrix'
  pinv : van cho mot nghiem, chuan ||w|| = 3.8176
  mot nghiem khac cung du bao y het: ||w|| = 4.0711 (sai so du bao lech 0.00e+00)
```

Đọc kỹ dòng cuối: có **vô số** vector $w$ cho ra **đúng cùng một dự báo**. Bài toán không phải là "không có nghiệm" mà là "có quá nhiều nghiệm". Giả nghịch đảo chọn giúp ta một cái — cái có **chuẩn nhỏ nhất**. Đó là một lựa chọn hợp lý, nhưng phải biết là mình đang chọn, vì nó chính là ridge với $\lambda \to 0^+$ (Mục 9.2).

### 4.4. Cộng tuyến: hệ số loạn, dự báo thì không

Ngay cả khi chưa suy biến hẳn, gần suy biến đã đủ gây chuyện. Thí nghiệm dựng hai cột có tương quan $\rho$ tăng dần, rồi lặp lại 300 lần với nhiễu khác nhau để đo độ dao động:

![Hình 2](figs/nt02_linreg.png)

**Hình 2.** Trái: số điều kiện bùng nổ theo cộng tuyến. Phải: hệ số dao động dữ dội trong khi dự báo đứng yên.

| Tương quan hai cột | $\mathrm{cond}(X^\top X)$ | Độ lệch chuẩn của $w_1$ | Độ lệch chuẩn của dự báo |
|---|---|---|---|
| 0,000 | 1,3 | 0,0324 | 0,0586 |
| 0,900 | 20,0 | 0,0772 | 0,0588 |
| 0,990 | 180,3 | 0,2373 | 0,0590 |
| 0,999 | 1958,9 | **0,8196** | **0,0585** |

Hai cột cuối là cả bài học. Khi tương quan đi từ 0 tới 0,999:

- độ dao động của **hệ số** tăng **25,3 lần**;
- độ dao động của **dự báo** đi từ 0,0586 xuống 0,0585, tức không đổi.

**Vì sao vậy?** Khi hai cột gần trùng nhau, mô hình không phân biệt được nên gán bao nhiêu cho cột này, bao nhiêu cho cột kia — nó chỉ xác định được **tổng** ảnh hưởng của chúng. Tổng ấy ổn định, nên dự báo ổn định; cách chia thì tuỳ nhiễu, nên hệ số loạn.

> **Hệ quả thực tế, và nó trả lời một câu hỏi phỏng vấn hay gặp.** Cộng tuyến là vấn đề của **diễn giải**, không phải của dự báo. Nếu mô hình chỉ để dự báo thì không cần làm gì. Nếu ai đó định đọc hệ số để nói "đặc trưng này ảnh hưởng chừng này" thì con số đó vô nghĩa — và cách chữa là ridge (Chương 9), gộp các đặc trưng, hoặc bỏ bớt.

### 4.5. Ba giả định, và cách kiểm

Hồi quy tuyến tính chỉ có ý nghĩa nếu:

1. **Quan hệ là tuyến tính theo tham số.** Chú ý "theo tham số": $y = w_0 + w_1 x + w_2 x^2$ vẫn là hồi quy tuyến tính, vì nó tuyến tính theo $w$. Kiểm bằng cách vẽ phần dư theo giá trị dự báo — có hình cong là hỏng.
2. **Nhiễu có phương sai không đổi.** Nếu phương sai nhiễu tăng theo $y$ (rất hay gặp với dữ liệu tiền tệ) thì bình phương tối thiểu vẫn cho ước lượng không thiên lệch nhưng **không còn hiệu quả nhất**, và mọi khoảng tin cậy đều sai. Cách chữa thường dùng: lấy logarit của $y$.
3. **Các quan sát độc lập.** Chuỗi thời gian gần như luôn vi phạm điều này.

Giả định 2 và 3 không ảnh hưởng tới **giá trị** của $\hat w$, chỉ ảnh hưởng tới **độ tin cậy** ta gán cho nó. Đây là phân biệt đáng nhớ.

---

## 5. Gradient descent

### 5.1. Vì sao cần, khi đã có nghiệm giải tích

Chương 4 cho nghiệm dạng đóng. Vậy sao vẫn cần phương pháp lặp? Ba lý do, theo thứ tự quan trọng tăng dần:

1. **Đa số bài toán không có nghiệm dạng đóng.** Cho gradient của mất mát logistic bằng 0 sẽ ra một hệ phương trình siêu việt, không giải tay được.
2. **Nghiệm dạng đóng có thể quá đắt.** Giải $X^\top X w = X^\top y$ tốn $O(nd^2 + d^3)$. Với $d = 10^6$ thì $d^3 = 10^{18}$ — bất khả thi.
3. **Dữ liệu có thể không vừa bộ nhớ.** Xuống dốc ngẫu nhiên chỉ cần một lô nhỏ mỗi bước.

### 5.2. Ý tưởng và công thức

Gradient chỉ hướng **tăng nhanh nhất** của hàm. Muốn giảm thì đi ngược lại:

$$\theta_{t+1} = \theta_t - \eta\,\nabla_\theta L(\theta_t).$$

Vì sao "ngược gradient" là hướng giảm nhanh nhất? Khai triển Taylor bậc nhất: $L(\theta + \delta) \approx L(\theta) + \nabla L^\top \delta$. Trong mọi $\delta$ có cùng độ dài, cái làm $\nabla L^\top \delta$ âm nhất là $\delta \propto -\nabla L$, theo bất đẳng thức Cauchy–Schwarz. Đó là toàn bộ chứng minh — và nó cũng cho thấy giới hạn: lập luận chỉ đúng **cục bộ**, với bước đi đủ nhỏ.

### 5.3. Tốc độ hội tụ phụ thuộc số điều kiện, không phụ thuộc số chiều

Đây là kết quả quan trọng nhất của chương, và nó đo được chính xác.

Xét hàm bậc hai $f(x) = \tfrac12(x_1^2 + \kappa x_2^2)$, cực tiểu tại gốc. Lý thuyết nói: với tốc độ học tối ưu $\eta = 2/(\kappa+1)$, sai số co mỗi vòng theo hệ số $(\kappa-1)/(\kappa+1)$, nên số vòng cần để đạt sai số $\varepsilon$ là $O(\kappa \log \tfrac1\varepsilon)$.

Nếu thêm **quán tính** (heavy ball) với tham số tối ưu, tốc độ co thành $(\sqrt\kappa-1)/(\sqrt\kappa+1)$, tức số vòng chỉ còn $O(\sqrt\kappa \log\tfrac1\varepsilon)$.

![Hình 3](figs/nt03_gd.png)

**Hình 3.** Số vòng lặp tới sai số $10^{-8}$, hai trục log. Quán tính đổi $\kappa$ thành $\sqrt\kappa$ — đó là một thay đổi về **bậc**, không phải về hằng số.

| $\kappa$ | Gradient descent | Thêm quán tính | Tỉ lệ | $\sqrt\kappa$ |
|---|---|---|---|---|
| 1 | 1 | 1 | 1,00 | 1,00 |
| 10 | 92 | 34 | 2,71 | 3,16 |
| 100 | 922 | 117 | 7,88 | 10,00 |
| 1 000 | 9 211 | 391 | 23,56 | 31,62 |
| 10 000 | **92 104** | **1 297** | 71,01 | 100,00 |

Ba điều đọc ra được:

1. **Cột gradient descent tăng đúng 10 lần mỗi hàng** — tuyến tính theo $\kappa$, đúng như lý thuyết.
2. **Cột quán tính tăng khoảng 3,3 lần mỗi hàng**, tức xấp xỉ $\sqrt{10} = 3{,}16$.
3. Tỉ lệ giữa hai cột bám sát $\sqrt\kappa$ với hệ số khoảng 0,7. Hằng số ấy không quan trọng; **bậc** mới quan trọng.

> **Vì sao kết quả này đáng nhớ.** Nó nói rằng tốc độ hội tụ **không phụ thuộc số chiều** mà phụ thuộc **hình dạng** của hàm mất mát. Một bài toán 1 triệu chiều nhưng tròn trịa ($\kappa$ nhỏ) dễ hơn nhiều một bài toán 2 chiều nhưng dẹt. Và nó nói luôn cách chữa: **chuẩn hoá đặc trưng về cùng thang đo là cách rẻ nhất để giảm $\kappa$.** Đó là lý do bước tiền xử lý tưởng như tầm thường ấy lại quan trọng đến vậy.

### 5.4. Chọn tốc độ học

| $\eta$ | Hiện tượng | Dấu hiệu nhận ra |
|---|---|---|
| Quá nhỏ | hội tụ đúng nhưng rất chậm | mất mát giảm đều, gần như tuyến tính, mãi không tới |
| Vừa | hội tụ nhanh | mất mát giảm dốc rồi phẳng dần |
| Hơi lớn | dao động quanh nghiệm | mất mát giảm rồi nhấp nhô, không phẳng hẳn |
| Quá lớn | phân kỳ | mất mát tăng, rồi thành `inf` hoặc `nan` |

Với hàm bậc hai có trị riêng lớn nhất $\lambda_{\max}$, điều kiện hội tụ là $\eta < 2/\lambda_{\max}$ — vượt ngưỡng ấy là phân kỳ chắc chắn, không phải chậm. Đây là một trong số rất ít trường hợp có ngưỡng chính xác.

[Chương 5 của *Mô hình & Kiến trúc*](models-ch05.html) mở rộng phần này thành momentum, RMSProp, Adam và AdamW, cùng lý do có bước hiệu chỉnh thiên lệch.

### 5.5. Kiểm tra đạo hàm, và cái bẫy về $\varepsilon$

Suy ra gradient bằng tay rất dễ sai. Cách kiểm duy nhất đáng tin là so với sai phân số:

$$\text{sai phân tiến:}\;\; \frac{f(x+\varepsilon) - f(x)}{\varepsilon}, \qquad \text{sai phân trung tâm:}\;\; \frac{f(x+\varepsilon) - f(x-\varepsilon)}{2\varepsilon}.$$

Câu hỏi: chọn $\varepsilon$ bao nhiêu? Trực giác nói "càng nhỏ càng chính xác". **Trực giác ấy sai**, và sai một cách đo được.

![Hình 4](figs/nt04_gradcheck.png)

**Hình 4.** Sai số theo $\varepsilon$, hai trục log. Hình chữ V: nhánh phải là sai số cắt cụt, nhánh trái là sai số làm tròn.

| $\varepsilon$ | Sai phân trung tâm | Sai phân tiến |
|---|---|---|
| $10^{-1}$ | $2{,}22 \times 10^{-3}$ | $6{,}08 \times 10^{-2}$ |
| $10^{-4}$ | $2{,}23 \times 10^{-9}$ | $5{,}86 \times 10^{-5}$ |
| $3{,}2 \times 10^{-6}$ | $\mathbf{1{,}72 \times 10^{-11}}$ | $1{,}85 \times 10^{-6}$ |
| $10^{-10}$ | $4{,}25 \times 10^{-7}$ | $8{,}01 \times 10^{-7}$ |
| $10^{-13}$ | $6{,}27 \times 10^{-4}$ | $9{,}38 \times 10^{-4}$ |

Có hai nguồn sai số đánh nhau:

- **Sai số cắt cụt** — do bỏ các số hạng bậc cao trong khai triển Taylor. Nó tỉ lệ $\varepsilon^2$ với sai phân trung tâm và $\varepsilon$ với sai phân tiến. Nguồn này **giảm** khi $\varepsilon$ nhỏ.
- **Sai số làm tròn** — do trừ hai số gần bằng nhau rồi chia cho một số rất nhỏ. Nó tỉ lệ $u/\varepsilon$ với $u \approx 2{,}22\times10^{-16}$ là epsilon máy. Nguồn này **tăng** khi $\varepsilon$ nhỏ.

Cân bằng hai nguồn:

$$\varepsilon^{*}_{\text{trung tâm}} \sim u^{1/3} = 6{,}06 \times 10^{-6}, \qquad \varepsilon^{*}_{\text{tiến}} \sim u^{1/2} = 1{,}49 \times 10^{-8}.$$

Giá trị đo được: **$5{,}62 \times 10^{-6}$** và **$1{,}78 \times 10^{-8}$**. Cả hai khớp với lý thuyết trong phạm vi lưới quét.

> **Quy tắc dùng được ngay.** Dùng sai phân **trung tâm** với $\varepsilon \approx 10^{-5}$ hoặc $10^{-6}$. Xem **sai số tương đối** $\frac{|g_{\text{giải tích}} - g_{\text{số}}|}{\max(|g_{\text{giải tích}}|, |g_{\text{số}}|, 10^{-8})}$; dưới $10^{-7}$ là yên tâm, trên $10^{-4}$ là gần như chắc chắn có lỗi. Và đừng bao giờ đặt $\varepsilon = 10^{-12}$ vì nghĩ rằng nhỏ hơn thì tốt hơn.

### 5.6. Xuống dốc theo lô, ngẫu nhiên và lô nhỏ

| Biến thể | Mỗi bước dùng | Ưu | Nhược |
|---|---|---|---|
| Theo lô đầy đủ | cả $n$ mẫu | gradient chính xác, hội tụ trơn | một bước rất đắt; cần cả dữ liệu trong bộ nhớ |
| Ngẫu nhiên (SGD) | 1 mẫu | bước cực rẻ, thoát được điểm yên ngựa | nhiễu lớn, đường đi nhấp nhô |
| Lô nhỏ | $B$ mẫu | dung hoà, tận dụng được phép tính song song | thêm một siêu tham số $B$ |

Lô nhỏ thắng trên thực tế không phải vì toán học đẹp hơn mà vì **phần cứng**: nhân ma trận $B \times d$ tận dụng GPU tốt hơn $B$ phép nhân vector rời rạc. Nhiễu của SGD hoá ra còn **có ích** ở bài toán không lồi, vì nó giúp thoát khỏi cực tiểu địa phương nông — nhưng với hàm lồi thì nhiễu chỉ là cái giá phải trả.

---

## 6. Phân loại tuyến tính: perceptron, logistic, softmax

### 6.1. Cùng một mô hình, ba hàm mất mát

Ba thuật toán của chương này dùng **chung một mô hình** — điểm số tuyến tính $s = w^\top x + b$ — và **chung một thuật toán tối ưu** — xuống dốc. Theo khung ở Hình 1, chúng chỉ khác nhau ở cột giữa.

Định nghĩa **biên** (margin) của một mẫu:

$$m_i = y_i\,(w^\top x_i + b), \qquad y_i \in \{-1, +1\}.$$

$m_i > 0$ nghĩa là đoán đúng, và $|m_i|$ đo mức độ tự tin. Mọi hàm mất mát phân loại đều là hàm của $m$:

![Hình 5](figs/nt05_matmat.png)

**Hình 5.** Bốn hàm mất mát theo biên $m$. Cái ta thực sự muốn là mất mát 0–1; ba cái còn lại là các cách làm trơn nó.

| Hàm mất mát | Công thức | Tính chất | Cho ra thuật toán |
|---|---|---|---|
| 0–1 | $\mathbb{1}[m \le 0]$ | không lồi, đạo hàm 0 khắp nơi | — (không tối ưu trực tiếp được) |
| Perceptron | $\max(0, -m)$ | lồi, phẳng khi $m > 0$ | perceptron |
| Hinge | $\max(0, 1-m)$ | lồi, phẳng khi $m > 1$ | SVM (Chương 13) |
| Logistic | $\log(1 + e^{-m})$ | lồi, trơn, không bao giờ phẳng hẳn | hồi quy logistic |

**Đây là ý trung tâm của chương.** Thứ ta thực sự muốn tối thiểu là tỉ lệ đoán sai, tức mất mát 0–1. Nhưng nó không lồi và có đạo hàm bằng 0 ở mọi nơi nó khả vi, nên không tối ưu được. Ba hàm còn lại đều là **chặn trên lồi** của nó, nên giảm chúng là gián tiếp giảm tỉ lệ đoán sai. Chọn chặn trên nào là chọn thuật toán.

### 6.2. Perceptron: cái đầu tiên, và giới hạn của nó

Quy tắc cập nhật, chỉ chạm vào các mẫu đang bị đoán sai:

$$\text{nếu } y_i(w^\top x_i) \le 0: \quad w \leftarrow w + y_i x_i.$$

Đây chính là xuống dốc trên mất mát perceptron với $\eta = 1$: đạo hàm của $\max(0,-m)$ theo $w$ bằng $-y_i x_i$ khi $m < 0$ và bằng 0 khi $m > 0$.

**Định lý Novikoff.** Nếu dữ liệu tách được tuyến tính với lề $\gamma$ và mọi điểm nằm trong hình cầu bán kính $R$, thì perceptron mắc **nhiều nhất $(R/\gamma)^2$ lần sai** trước khi dừng hẳn — bất kể có bao nhiêu điểm và bao nhiêu chiều.

Thí nghiệm sinh dữ liệu nằm sát lề để chặn có ý nghĩa:

| Lề $\gamma$ | $R = \max\|x\|$ | Chặn $(R/\gamma)^2$ | Số lần sai đo được | Hội tụ |
|---|---|---|---|---|
| 0,300 | 1,3273 | 19,5 | 4 | có |
| 0,201 | 1,2631 | 39,5 | 4 | có |
| 0,100 | 1,1703 | 136,8 | 10 | có |
| 0,050 | 1,1519 | 526,9 | 14 | có |

Chặn đúng ở cả bốn dòng, và số lần sai tăng lên khi lề co lại — đúng chiều lý thuyết dự đoán. Chặn khá lỏng, nhưng điều đáng giá là nó **không chứa $n$ và không chứa $d$**.

**Và đây là chỗ perceptron gãy.** Cùng thuật toán ấy chạy trên dữ liệu **không** tách được tuyến tính:

```text
hoi tu = False, so lan sai sau 2000 luot = 17,977
```

Nó không dừng, và không có bảo đảm gì cả. Định lý Novikoff chỉ nói về trường hợp tách được; ngoài trường hợp ấy perceptron có thể dao động mãi mãi. Lý do nằm ở Hình 5: mất mát perceptron **phẳng hoàn toàn khi $m > 0$**, nên thuật toán mất động lực ngay khi vừa đúng, không có khái niệm "đúng chắc chắn hơn".

### 6.3. Hồi quy logistic

Chữa đúng hai khuyết điểm trên bằng cách đổi sang hàm mất mát trơn và không bao giờ phẳng hẳn. Mô hình cho ra **xác suất**:

$$p(y=1 \mid x) = \sigma(w^\top x) = \frac{1}{1 + e^{-w^\top x}}.$$

Hàm mất mát là **entropy chéo**, mà Mục 10.3 sẽ cho thấy chính là âm log hợp lý:

$$L(w) = -\frac{1}{n}\sum_{i=1}^{n}\Big[y_i \log p_i + (1-y_i)\log(1-p_i)\Big], \qquad p_i = \sigma(w^\top x_i).$$

Gradient của nó gọn đến bất ngờ:

$$\nabla_w L = \frac{1}{n}X^\top(p - y).$$

So với gradient của hồi quy tuyến tính $\tfrac{2}{n}X^\top(Xw - y)$ thì **cùng một dạng**: ma trận thiết kế chuyển vị nhân với sai lệch. Đây không phải trùng hợp — Mục 10.3 giải thích rằng cả hai đều thuộc họ mũ, và mọi mô hình trong họ ấy đều có gradient dạng này.

**Vì sao không dùng bình phương sai lệch cho phân loại?** Hai lý do. Thứ nhất, $\sum(y_i - \sigma(w^\top x_i))^2$ **không lồi** theo $w$, nên mất hết bảo đảm của Chương 11. Thứ hai, khi mô hình đoán sai mà rất tự tin, đạo hàm của bình phương sai lệch **triệt tiêu** (vì $\sigma$ bão hoà), còn entropy chéo thì không — nó phạt mạnh đúng lúc cần.

### 6.4. Softmax: nhiều lớp

Với $K$ lớp, thay $K$ vector trọng số và chuẩn hoá bằng softmax:

$$p(y = k \mid x) = \frac{\exp(w_k^\top x)}{\sum_{j=1}^{K}\exp(w_j^\top x)}.$$

Softmax có một tính chất cần biết: **cộng cùng một vector vào mọi $w_k$ thì xác suất không đổi**, vì tử và mẫu cùng nhân thêm một hệ số. Nên chỉ **hiệu** giữa các vector trọng số là xác định được; bản thân chúng thì không.

Hệ quả đẹp, và kiểm chứng được:

> **Hồi quy logistic chính là softmax với $K = 2$.**

Thí nghiệm khớp cả hai mô hình trên cùng dữ liệu, không phạt chuẩn, chạy tới hội tụ:

```text
Sai khac lon nhat giua xac suat cua logistic va cua softmax(K=2): 3.331e-16
Hieu hai cot trong so cua softmax:  [ 0.10864027  1.53215349 -1.83422929]
Trong so cua hoi quy logistic:      [ 0.10864027  1.53215349 -1.83422929]
  -> lech 1.554e-15
Tong hai cot trong so cua softmax (phai bang 0): 6.661e-16
```

Khớp tới $1{,}6\times10^{-15}$, tức sai số máy. Đây không phải "hai mô hình cho kết quả giống nhau" mà là **một mô hình viết theo hai cách**.

### 6.5. Cái bẫy: dữ liệu tách được thì hợp lý cực đại không có nghiệm

Đây là chi tiết ít được dạy nhưng gặp thật, và nó giải thích vì sao thư viện nào cũng bật phạt chuẩn mặc định.

Nếu dữ liệu **tách được hoàn toàn**, thì với mọi $w$ tách đúng, nhân $w$ lên gấp đôi sẽ làm mọi $p_i$ tiến gần 0 hoặc 1 hơn, tức hợp lý **tăng**. Nên không tồn tại cực đại hữu hạn: $\|w\| \to \infty$.

| Số vòng lặp | $\|w\|$ khi $\lambda = 0$ | $\|w\|$ khi $\lambda = 0{,}01$ |
|---|---|---|
| 500 | 10,164 | 3,993 |
| 2 000 | 15,930 | 3,993 |
| 10 000 | 24,703 | 3,993 |
| 50 000 | **34,784** | **3,993** |

Cột giữa không có dấu hiệu dừng — nó sẽ cứ tăng mãi. Cột phải đứng yên từ vòng 500.

> **Ba hệ quả thực tế.** (1) `sklearn.linear_model.LogisticRegression` mặc định $C = 1{,}0$, tức **luôn có phạt chuẩn** — đó là chủ ý, không phải mặc định tuỳ tiện. (2) Nếu thấy hệ số hồi quy logistic lớn bất thường, hãy kiểm xem có đặc trưng nào rò rỉ nhãn không: tách được hoàn toàn thường là dấu hiệu rò rỉ. (3) Cùng hiện tượng ấy xuất hiện lại ở mạng nơ-ron, và là một trong các lý do weight decay tồn tại — xem [Mục 6.6 của *Mô hình & Kiến trúc*](models-ch06.html).

### 6.6. Những chỗ hay nhầm

| Phát biểu | Thực tế |
|---|---|
| "Hồi quy logistic là thuật toán hồi quy." | Nó là phân loại. Chữ "hồi quy" đến từ việc nó hồi quy **log tỉ lệ cược** theo $x$. |
| "Perceptron và SVM là hai thứ khác hẳn nhau." | Cùng mô hình tuyến tính, khác đúng một chỗ: hinge phẳng ở $m>1$ thay vì $m>0$. Chỗ khác ấy sinh ra khái niệm lề. |
| "Softmax cần $K$ vector trọng số độc lập." | Chỉ $K-1$ là tự do; cộng cùng một vector vào tất cả thì không đổi gì. |
| "Xác suất của hồi quy logistic là xác suất thật." | Chỉ khi mô hình được **hiệu chuẩn**. Dữ liệu mất cân bằng hoặc phạt chuẩn mạnh đều làm lệch — xem [Mục 9.4 của *MLOps*](mlops-ch09.html). |
| "Đoán sai thì tăng tốc độ học." | Với dữ liệu không tách được, perceptron dao động bất kể tốc độ học. Vấn đề ở hàm mất mát chứ không ở $\eta$. |

---

## 7. K láng giềng gần nhất và Naive Bayes

### 7.1. Hai cách nghĩ hoàn toàn khác nhau

Chương 6 học một **biên** rồi dùng nó để phân loại. Chương này trình bày hai cách khác hẳn, và chúng đại diện cho hai trường phái lớn:

- **KNN** không học gì cả. Nó giữ nguyên dữ liệu và trả lời bằng cách tra cứu. Gọi là phương pháp **phi tham số** và **lười**.
- **Naive Bayes** mô hình hoá $p(x \mid y)$ — tức mô tả dữ liệu của từng lớp trông thế nào — rồi lật ngược bằng quy tắc Bayes. Gọi là phương pháp **sinh**.

Phân biệt **sinh** (generative) và **phân biệt** (discriminative) đáng nắm vì nó hay được hỏi:

| | Mô hình hoá gì | Ví dụ | Mạnh khi |
|---|---|---|---|
| Phân biệt | $p(y \mid x)$ trực tiếp | logistic, SVM, cây | nhiều dữ liệu; chỉ cần dự báo |
| Sinh | $p(x \mid y)$ rồi lật | Naive Bayes, LDA, QDA | ít dữ liệu; cần sinh mẫu mới; cần xử lý dữ liệu thiếu |

### 7.2. K láng giềng gần nhất

Thuật toán vỏn vẹn: để phân loại $x$, tìm $k$ điểm huấn luyện gần $x$ nhất rồi lấy nhãn theo đa số.

Ba quyết định phải làm, và cả ba đều quan trọng hơn vẻ ngoài:

1. **Chọn $k$.** $k=1$ cho biên rất gấp khúc, thiên lệch thấp phương sai cao; $k$ lớn làm trơn biên, thiên lệch cao phương sai thấp. Đây là đánh đổi thiên lệch–phương sai ở dạng dễ thấy nhất, và $k$ chính là nút điều chỉnh.
2. **Chọn khoảng cách.** Euclid là mặc định nhưng chỉ hợp lý khi các đặc trưng cùng thang đo. **Không chuẩn hoá đặc trưng thì KNN gần như luôn sai** — một đặc trưng đo bằng đồng sẽ át hoàn toàn một đặc trưng đo bằng năm.
3. **Chọn cách đánh trọng số.** Lấy đa số đơn thuần, hay đánh trọng số theo $1/d$?

**Lời nguyền số chiều.** Đây là giới hạn nghiêm trọng nhất của KNN, và nó là một sự thật hình học chứ không phải vấn đề kỹ thuật. Trong không gian $d$ chiều, để một hình cầu chứa được tỉ lệ $f$ số điểm phân bố đều trong khối lập phương đơn vị, bán kính của nó phải là $f^{1/d}$. Với $d = 100$ và $f = 0{,}01$, bán kính cần là $0{,}01^{1/100} \approx 0{,}955$ — tức gần bằng toàn bộ cạnh khối lập phương. **"Láng giềng gần nhất" ở chiều cao thì không còn gần.**

Hệ quả: KNN chỉ dùng được ở số chiều thấp, hoặc sau khi đã giảm chiều (Chương 14), hoặc trong không gian nhúng đã học được — đó đúng là cách các hệ tìm kiếm ảnh hiện đại làm.

### 7.3. Naive Bayes

Xuất phát từ quy tắc Bayes:

$$p(y = k \mid x) \propto p(x \mid y=k)\,p(y=k).$$

Khó khăn là ước lượng $p(x \mid y=k)$ khi $x$ có nhiều chiều. Giả thiết **naive** cắt phăng khó khăn ấy: **các đặc trưng độc lập có điều kiện khi đã biết lớp.**

$$p(x \mid y=k) = \prod_{j=1}^{d} p(x_j \mid y=k).$$

Thay vì một phân phối $d$ chiều, ta chỉ cần $d$ phân phối một chiều. Số tham số giảm từ cỡ $d^2$ xuống cỡ $d$.

Giả thiết này gần như luôn **sai** — từ ngữ trong văn bản rõ ràng không độc lập. Nhưng Naive Bayes vẫn chạy tốt, vì để phân loại đúng ta chỉ cần **thứ tự** giữa các $p(y=k \mid x)$ đúng, chứ không cần bản thân các xác suất đúng. Ước lượng xác suất của nó thường rất tệ (dồn về 0 hoặc 1); nhãn thì thường ổn.

**Làm trơn Laplace.** Với dữ liệu đếm, nếu một từ chưa từng xuất hiện cùng lớp $k$ thì $p(x_j \mid y=k) = 0$, và vì ta **nhân** các xác suất nên cả tích thành 0 — một từ duy nhất phủ quyết toàn bộ bằng chứng còn lại. Cách chữa là cộng thêm một lượng nhỏ vào mọi ô đếm:

$$p(x_j = v \mid y=k) = \frac{N_{kjv} + \alpha}{N_k + \alpha\,V}.$$

Mục 10.5 cho thấy đây **chính là** MAP với tiên nghiệm Dirichlet, chứ không phải một thủ thuật kỹ thuật.

### 7.4. Naive Bayes, LDA và QDA chỉ khác nhau đúng một chỗ

Đây là kết quả gọn nhất của chương, và nó đo được.

Khi $p(x \mid y=k)$ được mô hình bằng phân phối Gauss, **cả ba phương pháp là một**, chỉ khác ràng buộc đặt lên ma trận hiệp phương sai $\Sigma$:

| Mô hình | Ràng buộc trên $\Sigma$ | Biên quyết định | Độ chính xác đo được |
|---|---|---|---|
| Naive Bayes Gauss | riêng từng lớp, **chéo** | bậc hai | 0,8550 |
| LDA | **dùng chung**, đầy đủ | **tuyến tính** | 0,8550 |
| QDA | riêng từng lớp, đầy đủ | bậc hai | 0,8600 |
| NB dùng chung $\Sigma$ chéo | dùng chung, chéo | tuyến tính | 0,8550 |

**Giả thiết "naive" chính là ép $\Sigma$ thành ma trận đường chéo.** Nói cách khác, Naive Bayes không phải một thuật toán riêng biệt — nó là bộ phân lớp Gauss với một ràng buộc cụ thể.

Thí nghiệm kiểm chứng đẳng thức ấy bằng hai cài đặt **hoàn toàn độc lập** trong mã nguồn: một bên nhân các mật độ Gauss **một chiều** lại với nhau (đúng định nghĩa Naive Bayes, không đụng tới ma trận hiệp phương sai ở bất kỳ đâu), một bên dùng mật độ Gauss **nhiều chiều** với $\Sigma$ chéo.

```text
sai khac lon nhat cua hieu diem phan biet: 1.222e-08
hai mo hinh du bao giong nhau o 100.0% so mau
ham quyet dinh cua LDA co TUYEN TINH khong? sai so khop tuyen tinh = 4.441e-15  -> co
con QDA thi sao? sai so khop tuyen tinh = 1.780e+01 -> khong, no bac hai
```

Hai dòng cuối kiểm chứng một khẳng định lý thuyết quan trọng. Khi hai lớp **dùng chung** $\Sigma$, các số hạng bậc hai $x^\top\Sigma^{-1}x$ triệt tiêu nhau lúc lấy hiệu, nên biên quyết định là **tuyến tính** — khớp tuyến tính sai lệch $4{,}4\times10^{-15}$. Khi mỗi lớp có $\Sigma$ riêng thì không triệt tiêu, và biên là bậc hai — khớp tuyến tính sai lệch 17,8, tức trật hẳn.

> **Cách trả lời khi được hỏi "LDA khác QDA thế nào".** Đừng nói "một cái tuyến tính một cái bậc hai" rồi dừng. Nói **vì sao**: dùng chung $\Sigma$ thì số hạng bậc hai triệt tiêu khi lấy hiệu hai hàm phân biệt. Và nói **cái giá**: QDA cần ước lượng $K$ ma trận $d\times d$ thay vì một, nên tốn dữ liệu gấp $K$ lần và hỏng sớm khi $n$ nhỏ.

---

## 8. Đánh giá mô hình phân loại

### 8.1. Vì sao độ chính xác không đủ

Dữ liệu thật hiếm khi cân bằng: gian lận thẻ, lỗi thiết bị, bệnh hiếm — lớp cần quan tâm luôn là lớp thiểu số. Trên dữ liệu như vậy độ chính xác không chỉ kém thông tin mà còn **đánh lừa một cách chủ động**.

Thí nghiệm dựng 20 000 mẫu với 0,98% thuộc lớp dương:

| Bộ phân loại | Độ chính xác | Precision | Recall | F1 |
|---|---|---|---|---|
| **Đoán TẤT CẢ là âm** | **0,9902** | — | 0,0000 | 0,0000 |
| Ngưỡng 2,6 | 0,9900 | 0,4928 | 0,5204 | **0,5062** |
| Ngưỡng 1,0 | 0,8408 | 0,0559 | 0,9592 | 0,1056 |
| Ngưỡng −1,0 | 0,1699 | 0,0117 | 1,0000 | 0,0231 |

Đọc kỹ hai dòng đầu. **Bộ phân loại vô dụng có độ chính xác 0,9902, cao hơn bộ phân loại dùng được (0,9900).** Nếu chọn mô hình theo độ chính xác thì ta đã chọn cái không bắt được một ca dương nào.

### 8.2. Ma trận nhầm lẫn và các đại lượng dẫn xuất

| | Dự đoán dương | Dự đoán âm |
|---|---|---|
| **Thật dương** | TP | FN |
| **Thật âm** | FP | TN |

$$\text{precision} = \frac{TP}{TP + FP}, \qquad \text{recall} = \frac{TP}{TP + FN}, \qquad F_1 = \frac{2\,PR}{P + R}.$$

Cách nhớ bằng câu hỏi, và đây là cách nhớ đúng bản chất:

- **Precision** trả lời: *trong những ca tôi báo động, bao nhiêu phần là thật?* Mẫu số là những gì **mô hình nói**.
- **Recall** trả lời: *trong những ca thật sự có, tôi bắt được bao nhiêu phần?* Mẫu số là những gì **thực tế có**.

Bảng ở Mục 8.1 cho thấy chúng đánh đổi nhau theo ngưỡng: hạ ngưỡng thì recall lên (0,52 → 0,96 → 1,00) còn precision xuống (0,49 → 0,056 → 0,012). **Không có ngưỡng nào tốt cho cả hai**, và chọn ngưỡng là quyết định nghiệp vụ chứ không phải quyết định kỹ thuật: nó phụ thuộc giá của một lần bỏ sót so với giá của một lần báo động nhầm.

$F_1$ là trung bình điều hoà của precision và recall. Dùng trung bình điều hoà chứ không phải trung bình cộng vì nó **phạt nặng sự mất cân đối**: bộ phân loại ở dòng cuối có recall hoàn hảo 1,00 nhưng $F_1$ chỉ 0,023.

### 8.3. ROC và Precision–Recall

![Hình 6](figs/nt06_metrics.png)

**Hình 6.** Cùng một bộ phân loại, cùng một bộ dữ liệu. Trái: ROC trông rất tốt. Phải: Precision–Recall kể một câu chuyện khác hẳn.

Trên đúng dữ liệu của Mục 8.1:

$$\text{ROC-AUC} = 0{,}9715 \qquad \text{nhưng} \qquad \text{PR-AUC} = 0{,}4931.$$

**Vì sao chênh lệch lớn đến thế?** Nằm ở mẫu số của trục hoành ROC:

$$\text{FPR} = \frac{FP}{FP + TN}.$$

Mẫu số là **toàn bộ lớp âm**, mà lớp âm ở đây có 19 804 mẫu. Thêm 100 báo động nhầm chỉ làm FPR nhúc nhích 0,005 — gần như không thấy trên đồ thị. Nhưng cũng 100 báo động nhầm ấy có thể làm precision sụp đổ, vì mẫu số của precision chỉ gồm những ca được báo động.

Một mốc cần nhớ: **PR-AUC của bộ đoán ngẫu nhiên bằng đúng tỉ lệ lớp dương**, ở đây là 0,0100. Còn ROC-AUC của bộ đoán ngẫu nhiên luôn là 0,5 bất kể mất cân bằng. Nên 0,4931 phải so với 0,0100, và khi so như vậy thì mô hình thực ra khá tốt — nhưng con số 0,9715 vẫn là con số không nên đem đi báo cáo.

> **Quy tắc chọn.** Lớp cân bằng, và cả hai lớp đều đáng quan tâm ngang nhau → ROC-AUC. Lớp mất cân bằng, và chỉ lớp thiểu số đáng quan tâm → **PR-AUC**. Đây là câu hỏi phỏng vấn rất hay gặp, và trả lời được kèm lý do về mẫu số là dấu hiệu hiểu thật.

### 8.4. Nhiều lớp: ba cách lấy trung bình

| Cách | Tính thế nào | Khi nào dùng |
|---|---|---|
| Macro | tính chỉ số cho từng lớp rồi lấy trung bình **không trọng số** | mọi lớp quan trọng như nhau, kể cả lớp hiếm |
| Micro | gộp toàn bộ TP, FP, FN rồi tính một lần | quan tâm hiệu năng tổng thể; lớp lớn chi phối |
| Weighted | trung bình có trọng số theo số mẫu mỗi lớp | dung hoà, nhưng lại che mất lớp hiếm |

Với phân loại đơn nhãn nhiều lớp, micro-F1 **đúng bằng** độ chính xác — nên nếu ai báo cáo "micro-F1" trên bài toán đơn nhãn thì họ đang báo cáo độ chính xác dưới một cái tên khác.

### 8.5. Một cảnh báo về chia dữ liệu

Mọi con số ở chương này đều vô nghĩa nếu tập kiểm tra bị nhiễm. Ba lỗi hay gặp nhất:

1. **Chuẩn hoá trước khi chia.** Tính trung bình và độ lệch chuẩn trên toàn bộ dữ liệu rồi mới chia là đã để thông tin tập kiểm tra rò vào tập huấn luyện. Phải khớp bộ chuẩn hoá **chỉ trên tập huấn luyện**.
2. **Chia ngẫu nhiên trên dữ liệu chuỗi thời gian.** Phải chia theo thời gian, nếu không mô hình được nhìn cả tương lai lẫn quá khứ của điểm cần dự báo.
3. **Chia ngẫu nhiên khi có nhóm.** Nếu một bệnh nhân có nhiều bản ghi, phải chia theo **bệnh nhân** chứ không theo bản ghi.

Ba lỗi này đều làm kết quả kiểm định tốt lên một cách giả tạo, và đều chỉ lộ ra khi mô hình đã ra sản xuất. [Chương 5 của *MLOps*](mlops-ch05.html) bàn kỹ hơn về tính đúng theo thời điểm.

---

## 9. Quá khớp, phạt chuẩn và kiểm định chéo

### 9.1. Quá khớp nhìn từ góc độ chương này

[Chương 2 của *Mô hình & Kiến trúc*](models-ch02.html) trình bày đầy đủ phân rã thiên lệch–phương sai. Chương này không lặp lại mà đi thẳng vào **công cụ chữa**, và vào chỗ mà hồi quy tuyến tính cho ta nhìn thấy cơ chế rõ nhất.

Nhắc lại đủ dùng: quá khớp là trạng thái **phương sai cao** — mô hình đổi nhiều khi đổi tập huấn luyện vì nó đủ linh hoạt để bám cả nhiễu.

### 9.2. Ridge: thêm $\lambda I$ và mọi thứ đổi

$$L(w) = \|y - Xw\|_2^2 + \lambda\|w\|_2^2 \;\Longrightarrow\; \hat w_{\text{ridge}} = (X^\top X + \lambda I)^{-1}X^\top y.$$

Số hạng $\lambda I$ nhỏ bé ấy làm được ba việc cùng lúc:

1. **Luôn khả nghịch.** Mục 2.5 đã nêu: thêm $\lambda I$ vào ma trận nửa xác định dương làm nó thành xác định dương với mọi $\lambda > 0$. Nên ridge có nghiệm duy nhất **kể cả khi $d > n$**, còn OLS thì không.
2. **Giảm số điều kiện**, nên gradient descent chạy nhanh hơn (Mục 5.3).
3. **Co các hệ số lại**, nên giảm phương sai.

### 9.3. Ridge co ngót không đều — và đó là cái hay của nó

Viết ridge qua SVD $X = UDV^\top$:

$$\hat w_{\text{ridge}} = \sum_{i} v_i \,\frac{d_i}{d_i^2 + \lambda}\, u_i^\top y, \qquad\text{so với}\qquad \hat w_{\text{OLS}} = \sum_{i} v_i \,\frac{1}{d_i}\, u_i^\top y.$$

Tỉ số giữa hai công thức cho **hệ số co ngót** theo từng hướng riêng:

$$\frac{d_i^2}{d_i^2 + \lambda}.$$

Đo được với $\lambda = 10$ (và nghiệm qua SVD khớp với dạng đóng tới $2{,}3\times10^{-15}$):

| Trị kỳ dị $d_i$ | Hệ số co ngót |
|---|---|
| 12,2680 | 0,9377 |
| 12,0678 | 0,9357 |
| 8,3651 | 0,8750 |
| 6,4450 | 0,8060 |
| 6,1730 | 0,7921 |

**Hướng có trị kỳ dị lớn gần như không bị động tới; hướng yếu bị co mạnh.** Đây là điều ridge làm đúng: nó co mạnh nhất đúng những hướng mà dữ liệu nói ít nhất — tức những hướng mà ước lượng OLS kém tin cậy nhất. Đây cũng là chỗ nối trực tiếp với Mục 4.4: cộng tuyến sinh ra trị kỳ dị nhỏ, và ridge chính là cách chữa cộng tuyến.

### 9.4. Lasso: điểm gãy ở gốc tạo ra hệ số bằng 0

$$L(w) = \|y - Xw\|_2^2 + \lambda\|w\|_1.$$

Khác biệt tưởng nhỏ mà hệ quả lớn:

![Hình 7](figs/nt07_regular.png)

**Hình 7.** Đường đi của các hệ số theo $\lambda$. Trái: ridge co dần nhưng không bao giờ chạm 0. Phải: lasso cắt hẳn về 0, từng hệ số một.

| $\lambda$ | $\|w\|_2$ (ridge) | Số hệ số bằng 0 — **ridge** | Số hệ số bằng 0 — **lasso** |
|---|---|---|---|
| 0,01 | 4,0106 | 0 | 0 |
| 0,10 | 4,0056 | 0 | 1 |
| 1,00 | 3,9559 | 0 | 1 |
| 10,00 | 3,5234 | 0 | 7 |
| 100,00 | 1,7090 | **0** | **9** |

Mô hình sinh dữ liệu có đúng **9** hệ số bằng 0 trong 12. Ở $\lambda = 100$, lasso tìm ra đúng 9; ridge tìm ra 0.

**Vì sao?** Đạo hàm của $|w|$ tại 0 không tồn tại — có một **điểm gãy**, và dưới vi phân của nó là cả đoạn $[-1, 1]$. Nghĩa là để kéo một hệ số ra khỏi 0, gradient của phần mất mát phải thắng được một lực có độ lớn cố định $\lambda$. Nếu không thắng nổi, hệ số **dính** ở 0. Với $w^2$ thì đạo hàm là $2w$, tiến về 0 khi $w$ tiến về 0 — lực kéo yếu dần nên không bao giờ đẩy được tới đúng 0.

> **Chọn cái nào.** Cần **chọn đặc trưng** hoặc tin rằng chỉ vài đặc trưng thực sự quan trọng → lasso. Các đặc trưng đều có chút ít tác dụng, hoặc có nhóm đặc trưng tương quan cao → ridge. Không chắc → elastic net, tức dùng cả hai. Lưu ý lasso xử lý nhóm tương quan khá tuỳ tiện: nó thường chọn **một** đại diện và vứt phần còn lại, mà chọn cái nào thì phụ thuộc nhiễu.

### 9.5. Kiểm định chéo

$\lambda$ không học được từ dữ liệu huấn luyện — tăng $\lambda$ luôn làm mất mát huấn luyện tăng. Nên phải ước lượng hiệu năng ngoài mẫu, và cách chuẩn là **kiểm định chéo $k$ phần**: chia dữ liệu thành $k$ phần, lần lượt giữ một phần để kiểm và huấn luyện trên $k-1$ phần còn lại, rồi lấy trung bình.

| Biến thể | Dùng khi |
|---|---|
| $k$ phần ($k = 5$ hoặc 10) | mặc định |
| Phân tầng | phân loại, nhất là khi mất cân bằng — giữ tỉ lệ lớp trong mỗi phần |
| Bỏ một mẫu ($k = n$) | dữ liệu rất ít; tốn kém và có phương sai cao |
| Theo thời gian | chuỗi thời gian — chỉ huấn luyện trên quá khứ |
| Theo nhóm | có nhóm tự nhiên trong dữ liệu (bệnh nhân, người dùng) |

> **Lỗi nghiêm trọng nhất khi dùng kiểm định chéo.** Dùng cùng một vòng kiểm định chéo để (a) chọn siêu tham số và (b) báo cáo hiệu năng. Con số báo cáo khi ấy **lạc quan có hệ thống**, vì ta đã chọn cấu hình tốt nhất *trên chính tập đó*. Cách đúng là **kiểm định chéo lồng nhau**, hoặc giữ riêng một tập kiểm tra không bao giờ được chạm tới cho tới lần đo cuối cùng.

---

## 10. Hợp lý cực đại và hậu nghiệm cực đại

### 10.1. Vì sao chương này đặt ở đây

Chín chương trước đã dùng nhiều hàm mất mát: bình phương sai lệch, entropy chéo, hinge, cộng thêm phạt $\ell_1$ hoặc $\ell_2$. Chúng trông như những lựa chọn rời rạc mà ta phải nhớ.

Chương này cho thấy chúng **không rời rạc**. Gần hết chúng đều suy ra được từ một nguyên lý duy nhất, và bản thân phạt chuẩn cũng vậy.

### 10.2. Hợp lý cực đại

Hàm **hợp lý** là xác suất của dữ liệu đã quan sát, xem như hàm của tham số:

$$\mathcal{L}(\theta) = p(\mathcal{D} \mid \theta) = \prod_{i=1}^{n} p(y_i \mid x_i, \theta).$$

Ta lấy logarit rồi đổi dấu — tích thành tổng, và cực đại thành cực tiểu:

$$\hat\theta_{\text{MLE}} = \arg\min_\theta \; -\sum_{i=1}^{n}\log p(y_i \mid x_i, \theta).$$

**Kết quả then chốt: giả thiết nhiễu Gauss cho ra đúng bình phương tối thiểu.** Giả sử $y_i = w^\top x_i + \epsilon_i$ với $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$. Khi ấy

$$-\log p(y_i \mid x_i, w) = \frac{(y_i - w^\top x_i)^2}{2\sigma^2} + \underbrace{\tfrac12\log(2\pi\sigma^2)}_{\text{không phụ thuộc } w}.$$

Cộng lại và bỏ hằng số:

$$\hat w_{\text{MLE}} = \arg\min_w \sum_i (y_i - w^\top x_i)^2 = \hat w_{\text{OLS}}.$$

> **Đây là câu trả lời cho "vì sao lại bình phương sai lệch, sao không phải trị tuyệt đối".** Bình phương sai lệch không phải một lựa chọn thẩm mỹ — nó là hệ quả của giả thiết **nhiễu Gauss**. Nếu tin rằng nhiễu có đuôi dày hơn Gauss thì bình phương tối thiểu là lựa chọn sai, và MLE dưới giả thiết Laplace sẽ cho ra **hồi quy trị tuyệt đối**, thứ bền vững hơn với điểm ngoại lai. Nói được mối nối này là dấu hiệu hiểu chứ không thuộc.

### 10.3. Entropy chéo cũng vậy

Với phân loại nhị phân, $y_i \sim \text{Bernoulli}(p_i)$ với $p_i = \sigma(w^\top x_i)$:

$$p(y_i \mid x_i, w) = p_i^{y_i}(1-p_i)^{1-y_i} \;\Longrightarrow\; -\log p(y_i \mid x_i, w) = -\big[y_i\log p_i + (1-y_i)\log(1-p_i)\big].$$

Vế phải đúng là hàm mất mát entropy chéo ở Mục 6.3. Nên **entropy chéo không phải định nghĩa mà là hệ quả** — nó là âm log hợp lý dưới giả thiết Bernoulli.

Điều này giải thích luôn hình dạng gradient gọn gàng đã thấy ở Mục 6.3. Cả hồi quy tuyến tính lẫn hồi quy logistic đều thuộc **họ mũ**, và mọi mô hình tuyến tính tổng quát trong họ ấy đều có gradient dạng $X^\top(\hat y - y)$.

### 10.4. MAP: phạt chuẩn chính là tiên nghiệm

Hợp lý cực đại không có chỗ nào để đưa vào hiểu biết sẵn có. Quy tắc Bayes ở Mục 3.5 cho chỗ ấy:

$$\hat\theta_{\text{MAP}} = \arg\max_\theta \; p(\mathcal{D}\mid\theta)\,p(\theta) = \arg\min_\theta\;\big[\underbrace{-\log p(\mathcal{D}\mid\theta)}_{\text{mất mát}} \underbrace{- \log p(\theta)}_{\text{phạt chuẩn}}\big].$$

Đặt tiên nghiệm Gauss $w \sim \mathcal{N}(0, \tau^2 I)$:

$$-\log p(w) = \frac{\|w\|_2^2}{2\tau^2} + \text{hằng số}.$$

Cộng với âm log hợp lý Gauss ở Mục 10.2 và nhân $2\sigma^2$:

$$\hat w_{\text{MAP}} = \arg\min_w\; \|y - Xw\|_2^2 + \frac{\sigma^2}{\tau^2}\|w\|_2^2.$$

So với Mục 9.2 thì đây **đúng là ridge**, với

$$\boxed{\;\lambda = \frac{\sigma^2}{\tau^2}\;}$$

**Kiểm chứng bằng số.** Thí nghiệm tính nghiệm theo hai đường hoàn toàn khác nhau: một bên giải hệ tuyến tính dạng đóng của ridge, một bên chạy BFGS trực tiếp trên âm log hậu nghiệm.

```text
lambda ly thuyet = sigma^2/tau^2 = 1.0/0.49 = 2.040816
Sai khac giua ridge dang dong va MAP toi uu bang so: 1.973e-08
```

Khớp tới dung sai của bộ tối ưu. Đây không phải sự tương tự — nó là **cùng một bài toán**.

Tương tự, tiên nghiệm Laplace $p(w_j) \propto e^{-|w_j|/b}$ cho ra $-\log p(w) = \sum_j |w_j|/b$, tức **lasso**. Và hai tiên nghiệm ấy nói đúng điều ta trông đợi: Gauss nói "các hệ số nên nhỏ", Laplace nói "phần lớn các hệ số nên bằng 0" — vì phân phối Laplace có đỉnh nhọn tại 0.

| Phạt chuẩn | Tiên nghiệm tương ứng | Niềm tin nó phát biểu |
|---|---|---|
| $\ell_2$ (ridge) | Gauss $\mathcal{N}(0, \tau^2)$ | hệ số nên nhỏ và trải đều |
| $\ell_1$ (lasso) | Laplace | phần lớn hệ số nên đúng bằng 0 |
| Không phạt | đều (không chính quy) | không có ý kiến gì trước |
| Làm trơn Laplace (Mục 7.3) | Dirichlet | mọi kết cục đều có thể xảy ra |

### 10.5. Bảng tổng kết: mọi thứ từ một nguyên lý

| Giả thiết về phân phối | Tiên nghiệm | Cho ra |
|---|---|---|
| Nhiễu Gauss | không | bình phương tối thiểu (Chương 4) |
| Nhiễu Gauss | Gauss | ridge (Mục 9.2) |
| Nhiễu Gauss | Laplace | lasso (Mục 9.4) |
| Nhiễu Laplace | không | hồi quy trị tuyệt đối |
| Nhãn Bernoulli | không | hồi quy logistic (Mục 6.3) |
| Nhãn Bernoulli | Gauss | hồi quy logistic có phạt $\ell_2$ |
| Nhãn nhiều lớp | không | hồi quy softmax (Mục 6.4) |
| Đếm đa thức + độc lập có điều kiện | Dirichlet | Naive Bayes có làm trơn (Chương 7) |

Cột trái là **giả thiết về thế giới**; cột phải là **thuật toán**. Học máy cổ điển phần lớn là bảng này.

> **Một ranh giới cần nói rõ.** MAP **không phải** suy luận Bayes đầy đủ. Suy luận Bayes giữ cả phân phối hậu nghiệm và lấy trung bình dự báo trên đó; MAP chỉ lấy **một điểm** — đỉnh của hậu nghiệm — rồi vứt phần còn lại. Nên MAP không cho ta độ bất định, và với hậu nghiệm lệch hoặc nhiều đỉnh thì điểm đỉnh có thể chẳng đại diện cho gì. SVM ở Chương 13 thì nằm ngoài bảng này hẳn: hinge không phải âm log hợp lý của phân phối tự nhiên nào, nên SVM không có cách đọc xác suất trực tiếp.

---

## 11. Tập lồi và hàm lồi

### 11.1. Vì sao dành hẳn hai chương cho tính lồi

Vì nó là ranh giới giữa hai thế giới khác hẳn nhau.

Với bài toán lồi, ta có một bảo đảm mạnh đến mức hiếm gặp trong học máy: **mọi cực tiểu địa phương đều là cực tiểu toàn cục**. Nghĩa là điểm khởi tạo không quan trọng, không cần chạy lại nhiều lần, và khi thuật toán dừng thì ta *biết* mình đã ở nghiệm tốt nhất.

Với bài toán không lồi — tức mọi mạng nơ-ron — không có bảo đảm nào cả. Ta chạy nhiều lần, giữ kết quả tốt nhất, và không bao giờ biết còn nghiệm nào tốt hơn hay không.

Chương này và chương sau cũng là **công cụ để dựng SVM**. Không có đối ngẫu Lagrange thì không có thủ thuật nhân, và không có thủ thuật nhân thì SVM chỉ là một bộ phân loại tuyến tính nữa.

### 11.2. Tập lồi

Tập $C$ là **lồi** nếu đoạn thẳng nối hai điểm bất kỳ của nó nằm trọn trong nó:

$$x, y \in C,\; t \in [0,1] \;\Longrightarrow\; t x + (1-t) y \in C.$$

| Lồi | Không lồi |
|---|---|
| Hình cầu, hình hộp, nửa không gian | Hình vành khuyên, hình chữ U |
| Siêu phẳng và giao của nhiều nửa không gian | Hợp của hai hình tròn rời nhau |
| Bao lồi của một tập điểm bất kỳ | Mặt cầu (chỉ phần vỏ) |

Hai tính chất được dùng ngay ở Chương 13: **giao của các tập lồi là lồi** (nên miền chấp nhận được của một hệ ràng buộc tuyến tính luôn lồi), nhưng **hợp thì thường không**.

### 11.3. Hàm lồi

$f$ là **lồi** nếu với mọi $x, y$ và $t \in [0,1]$:

$$f\big(t x + (1-t)y\big) \;\le\; t f(x) + (1-t) f(y).$$

Đọc bằng lời: **dây cung luôn nằm trên đồ thị.**

Ba cách kiểm, theo thứ tự dễ dùng tăng dần:

1. **Theo định nghĩa** — luôn đúng nhưng thường khó.
2. **Đạo hàm bậc hai** — với hàm một biến, $f'' \ge 0$ trên toàn miền. Với hàm nhiều biến, ma trận Hessian nửa xác định dương.
3. **Theo phép dựng** — xem Mục 11.5.

### 11.4. Kiểm bằng số: bác bỏ thì dễ, chứng minh thì không

Thí nghiệm thử trực tiếp bất đẳng thức định nghĩa trên 20 000 cặp điểm ngẫu nhiên:

| Hàm | Vi phạm lớn nhất | Tỉ lệ vi phạm | Kết luận |
|---|---|---|---|
| $x^2$ | $-1{,}50\times10^{-9}$ | 0,0000 | lồi |
| $\lvert x\rvert$ | $0$ | 0,0000 | lồi |
| $e^x$ | $-3{,}99\times10^{-9}$ | 0,0000 | lồi |
| $\log(1+e^x)$ — mất mát logistic | $-2{,}54\times10^{-11}$ | 0,0000 | lồi |
| $x^4 - 3x^2$ | $+2{,}237$ | 0,2075 | **không** |
| $\sin x$ | $+1{,}972$ | 0,5051 | **không** |
| $x^3$ | $+13{,}17$ | 0,4985 | **không** |

Ba dòng đầu có "vi phạm lớn nhất" **âm**, nghĩa là bất đẳng thức luôn đúng với lề dương — số âm nhỏ ấy chỉ là sai số làm tròn.

> **Phép thử này không chứng minh tính lồi.** Không vi phạm trên 20 000 cặp không loại trừ được một chỗ lõm hẹp ở đâu đó. Nhưng **một** phản ví dụ thì đủ để bác bỏ, và với hàm mất mát tự viết thì đây là cách rẻ nhất để bắt lỗi trước khi ngồi suy luận.

Dòng thứ tư đáng chú ý riêng: nó xác nhận mất mát logistic là hàm lồi, tức bài toán ở Mục 6.3 có nghiệm toàn cục duy nhất (khi có phạt chuẩn). Đó là lý do hồi quy logistic **không cần chạy lại nhiều lần với nhiều khởi tạo**, khác hẳn mạng nơ-ron.

### 11.5. Các phép dựng giữ nguyên tính lồi

Trong thực tế hiếm ai kiểm định nghĩa. Người ta dựng hàm lồi từ các mảnh đã biết là lồi:

| Phép | Giữ tính lồi không |
|---|---|
| $f_1 + f_2$ | có |
| $\alpha f$ với $\alpha \ge 0$ | có |
| $\max(f_1, f_2)$ | **có** |
| $\min(f_1, f_2)$ | **không** |
| $f(Ax + b)$ — hợp với hàm affine | có |
| $g(f(x))$ với $g$ lồi **và không giảm** | có |
| $f_1 \cdot f_2$ | **không** nói chung |

Dòng $\max$ giải thích ngay vì sao mất mát hinge $\max(0, 1-m)$ lồi: nó là max của hai hàm affine. Và dòng "hợp với affine" giải thích vì sao $\|y - Xw\|^2$ lồi theo $w$: nó là hàm lồi $\|\cdot\|^2$ hợp với một ánh xạ affine của $w$.

Cộng hai dòng đầu lại: **mất mát lồi cộng phạt chuẩn lồi thì vẫn lồi.** Nên ridge, lasso, hồi quy logistic có phạt chuẩn và SVM đều là bài toán lồi.

### 11.6. Hậu quả lên tối ưu hoá, đo được

![Hình 8](figs/nt08_convex.png)

**Hình 8.** Cùng một thuật toán, 21 điểm xuất phát. Trái: hàm lồi, mọi đường đều về cùng một chỗ. Phải: hàm không lồi, kết quả tuỳ chỗ xuất phát.

```text
ham LOI      x^2             : so diem dung khac nhau = 1
ham KHONG LOI x^4-3x^2+x/2   : so diem dung khac nhau = 2, tai x = -1.2645 va 1.1807
gia tri ham tai hai diem do  : -2.8725 va -1.6484
```

Hai điểm dừng ấy có giá trị hàm **khác nhau**: $-2{,}8725$ và $-1{,}6484$. Nghĩa là gần một nửa số lần chạy kết thúc ở một cực tiểu địa phương **không phải** toàn cục, và thuật toán không có cách nào biết điều đó.

> **Đây là lý do thực tế của mọi thứ ở Chương 6 của *Mô hình & Kiến trúc*.** Hàm mất mát của mạng nơ-ron không lồi, nên khởi tạo, chuẩn hoá và kết nối tắt mới thành vấn đề sống còn — xem [Chương 6 của *Mô hình & Kiến trúc*](models-ch06.html). Với hồi quy logistic thì những thứ ấy không cần thiết, và Hình 8 cho thấy vì sao.

---

## 12. Tối ưu lồi và đối ngẫu Lagrange

### 12.1. Dạng chuẩn

$$\min_x \; f_0(x) \quad\text{với}\quad f_i(x) \le 0 \;\; (i=1,\dots,m), \qquad h_j(x) = 0 \;\;(j=1,\dots,p).$$

Bài toán là **lồi** khi $f_0$ và mọi $f_i$ lồi, còn mọi $h_j$ affine. Lưu ý ràng buộc đẳng thức phải **affine**, không chỉ lồi — vì $\{h(x) = 0\}$ với $h$ lồi nhưng không affine thì không phải tập lồi.

### 12.2. Hàm Lagrange

Ý tưởng: thay vì cấm vi phạm ràng buộc, hãy **tính tiền** mỗi lần vi phạm. Gắn cho mỗi ràng buộc một giá $\alpha_i \ge 0$:

$$\mathcal{L}(x, \alpha, \nu) = f_0(x) + \sum_{i=1}^{m}\alpha_i f_i(x) + \sum_{j=1}^{p}\nu_j h_j(x).$$

Các $\alpha_i$, $\nu_j$ gọi là **nhân tử Lagrange**. Điều kiện $\alpha_i \ge 0$ là bắt buộc và có nghĩa rõ ràng: vi phạm ($f_i > 0$) phải **làm tăng** giá trị hàm.

Hàm **đối ngẫu** là cực tiểu của $\mathcal{L}$ theo $x$:

$$g(\alpha, \nu) = \inf_x \mathcal{L}(x, \alpha, \nu).$$

$g$ **luôn lõm** — kể cả khi bài toán gốc không lồi — vì nó là infimum của một họ hàm affine theo $(\alpha,\nu)$. Đây là một trong những sự thật hữu dụng nhất của toàn bộ lĩnh vực.

### 12.3. Đối ngẫu yếu và đối ngẫu mạnh

Với mọi $\alpha \ge 0$:

$$g(\alpha, \nu) \;\le\; p^{*},$$

trong đó $p^*$ là giá trị tối ưu của bài toán gốc. Đây là **đối ngẫu yếu**, và nó luôn đúng. Bài toán **đối ngẫu** là đi tìm chặn dưới tốt nhất:

$$d^{*} = \max_{\alpha \ge 0,\, \nu} g(\alpha, \nu) \;\le\; p^{*}.$$

Hiệu $p^* - d^*$ gọi là **khe đối ngẫu**. Khi nó bằng 0 ta có **đối ngẫu mạnh**, và khi ấy giải bài toán đối ngẫu là giải xong bài toán gốc.

**Điều kiện Slater:** với bài toán lồi, nếu tồn tại một điểm thoả mọi ràng buộc bất đẳng thức một cách **nghiêm ngặt** ($f_i(x) < 0$), thì đối ngẫu mạnh xảy ra. Điều kiện này nhẹ và hầu như luôn thoả trong các bài toán học máy — nên trên thực tế ta được dùng đối ngẫu mạnh gần như miễn phí.

### 12.4. Điều kiện KKT

Khi đối ngẫu mạnh xảy ra và các hàm khả vi, nghiệm tối ưu $(x^*, \alpha^*, \nu^*)$ thoả bốn nhóm điều kiện:

| Điều kiện | Công thức | Nghĩa |
|---|---|---|
| Chấp nhận được (gốc) | $f_i(x^*) \le 0$, $h_j(x^*) = 0$ | nghiệm hợp lệ |
| Chấp nhận được (đối ngẫu) | $\alpha_i^* \ge 0$ | giá không âm |
| **Bù trừ** | $\alpha_i^{*} f_i(x^{*}) = 0$ | **ràng buộc lỏng thì giá bằng 0** |
| Dừng | $\nabla_x \mathcal{L}(x^*,\alpha^*,\nu^*) = 0$ | điểm dừng của Lagrange |

Dòng **bù trừ** là dòng quan trọng nhất, và nó là chìa khoá của cả Chương 13. Nó nói: với mỗi ràng buộc, hoặc ràng buộc đó **chặt** ($f_i = 0$), hoặc giá của nó **bằng 0**. Không thể vừa lỏng vừa có giá.

Áp vào SVM ở chương sau, điều này có nghĩa: **chỉ những điểm nằm đúng trên lề mới có $\alpha > 0$**, và chỉ chúng mới ảnh hưởng tới nghiệm. Đó chính là định nghĩa của *vector hỗ trợ*, và nó không phải một thiết kế mà là một **hệ quả** của KKT.

### 12.5. Vì sao lại đi đường vòng qua đối ngẫu

Ba lý do, và với SVM thì lý do thứ ba mới là lý do thật:

1. **Đôi khi đối ngẫu dễ giải hơn.** Bài toán gốc của SVM có $d$ biến; bài toán đối ngẫu có $n$ biến. Khi $d \gg n$ thì đối ngẫu rẻ hơn hẳn.
2. **Đối ngẫu cho chặn dưới.** Kể cả khi chưa giải xong, $g(\alpha)$ cho ta biết còn cách tối ưu bao xa.
3. **Đối ngẫu làm lộ ra cấu trúc.** Bài toán đối ngẫu của SVM chỉ phụ thuộc dữ liệu qua các **tích vô hướng** $x_i^\top x_j$. Ngay khi thấy điều đó, thủ thuật nhân trở thành hiển nhiên: thay tích vô hướng bằng một hàm nhân là xong. Không có bước chuyển sang đối ngẫu thì không nhìn ra được điều này.

---

## 13. Support Vector Machine

### 13.1. Ý tưởng: trong vô số biên đúng, chọn biên nào

Với dữ liệu tách được tuyến tính, có **vô số** siêu phẳng tách đúng. Perceptron dừng ở cái đầu tiên nó gặp (Mục 6.2), và cái ấy có thể sát ngay một điểm dữ liệu.

SVM đặt một tiêu chí chọn: **lấy siêu phẳng có lề rộng nhất**, tức cách đều hai lớp nhiều nhất có thể. Trực giác biện minh: biên càng xa dữ liệu thì càng chịu được nhiễu, nên càng tổng quát hoá tốt.

### 13.2. Từ trực giác tới bài toán tối ưu

Khoảng cách từ $x_i$ tới siêu phẳng $w^\top x + b = 0$ là $|w^\top x_i + b| / \|w\|$ (Mục 2.2). Ta được tự do co giãn $w$ và $b$ cùng lúc mà không đổi siêu phẳng, nên **chuẩn hoá** sao cho điểm gần nhất có $|w^\top x_i + b| = 1$. Khi ấy lề — khoảng cách giữa hai mặt $w^\top x + b = \pm 1$ — bằng $2/\|w\|$.

Cực đại $2/\|w\|$ tương đương cực tiểu $\tfrac12\|w\|^2$, nên:

$$\min_{w,b} \;\tfrac12\|w\|^2 \qquad\text{với}\qquad y_i(w^\top x_i + b) \ge 1 \;\; \forall i.$$

Hàm mục tiêu lồi (bậc hai với Hessian $I$), ràng buộc affine — đúng dạng chuẩn ở Mục 12.1.

### 13.3. Bài toán đối ngẫu, và vì sao nó quan trọng

Lập Lagrange rồi cho đạo hàm theo $w$ và $b$ bằng 0:

$$w = \sum_{i}\alpha_i y_i x_i, \qquad \sum_i \alpha_i y_i = 0.$$

Thế ngược vào cho bài toán đối ngẫu:

$$\max_{\alpha \ge 0} \;\sum_i \alpha_i - \tfrac12\sum_{i,j}\alpha_i\alpha_j y_i y_j \,\underbrace{x_i^\top x_j}_{\text{chỉ qua tích vô hướng}} \qquad\text{với}\quad \sum_i\alpha_i y_i = 0.$$

Hai điều rút ra ngay:

- $w$ là **tổ hợp tuyến tính của các điểm dữ liệu**, với trọng số $\alpha_i y_i$.
- Theo điều kiện bù trừ (Mục 12.4), $\alpha_i > 0$ **chỉ với** những điểm có $y_i(w^\top x_i + b) = 1$, tức nằm đúng trên lề. Đó là các **vector hỗ trợ**. Mọi điểm khác có $\alpha_i = 0$ và **không ảnh hưởng gì** tới nghiệm.

**Kiểm chứng bằng số.** Thí nghiệm giải bài toán bằng libsvm rồi dựng lại mọi thứ từ nghiệm đối ngẫu:

```text
Dung lai w tu nghiem doi ngau: lech so voi coef_ cua sklearn = 0.000e+00
Vector ho tro co thoa y_i(w.x_i + b) = 1 khong? lech lon nhat = 2.523e-08
So vector ho tro: 3 tren 120 diem (2.5%)

Doi ngau manh:
  gia tri bai toan goc  0.5*||w||^2  = 1.1037868651
  gia tri bai toan doi ngau          = 1.1037868254
  khe doi ngau                       = 3.961e-08
Le = 2/||w|| = 1.346085; khoang cach nho nhat tu diem toi sieu phang x 2 = 1.346085
```

Bốn khẳng định lý thuyết, bốn lần khớp: công thức $w = \sum\alpha_i y_i x_i$ khớp **chính xác**; vector hỗ trợ nằm đúng trên lề; khe đối ngẫu bằng 0 trong phạm vi dung sai của bộ giải; và lề $2/\|w\|$ đúng bằng hai lần khoảng cách nhỏ nhất.

Chi tiết đáng chú ý nhất: **3 trên 120 điểm** quyết định toàn bộ nghiệm. Xoá 117 điểm còn lại đi thì kết quả không đổi. Đây là nghĩa thực sự của từ "thưa" khi nói về SVM.

### 13.4. Lề mềm: khi dữ liệu không tách được

Bài toán ở Mục 13.2 **vô nghiệm** nếu không có siêu phẳng nào tách đúng. Thêm biến bù $\xi_i \ge 0$ cho phép vi phạm, và tính tiền:

$$\min_{w,b,\xi}\;\tfrac12\|w\|^2 + C\sum_i \xi_i \qquad\text{với}\qquad y_i(w^\top x_i + b) \ge 1 - \xi_i,\quad \xi_i \ge 0.$$

Bài toán đối ngẫu gần như không đổi — chỉ thêm chặn trên $\alpha_i \le C$. Và KKT chia các điểm thành **đúng ba nhóm**:

| $\alpha_i$ | Vị trí của điểm | Ảnh hưởng tới nghiệm |
|---|---|---|
| $\alpha_i = 0$ | nằm ngoài lề, đúng phía | không |
| $0 < \alpha_i < C$ | nằm **đúng trên** lề | có |
| $\alpha_i = C$ | vi phạm lề (lọt vào trong, hoặc sai phía) | có |

Thí nghiệm trên hai lớp chồng lấn, đếm đúng ba nhóm ấy:

![Hình 9](figs/nt09_svm.png)

**Hình 9.** Trái và giữa: cùng dữ liệu, hai giá trị $C$. Đường liền là biên, hai đường đứt là lề. Phải: nhân RBF trên dữ liệu hai vòng tròn đồng tâm.

| $C$ | Lề $2/\|w\|$ | Số VTHT | Đúng trên lề | Vi phạm lề | Sai số huấn luyện | Sai số kiểm tra |
|---|---|---|---|---|---|---|
| 0,003 | 5,6121 | 114 | 0 | 114 | 0,1500 | 0,1795 |
| 0,030 | 2,6409 | 66 | 3 | 63 | 0,1750 | **0,1727** |
| 0,300 | 1,8286 | 48 | 3 | 45 | 0,1583 | 0,1790 |
| 3,000 | 1,7312 | 45 | 3 | 42 | 0,1667 | 0,1795 |
| 300,000 | 1,7311 | 45 | 3 | 42 | 0,1667 | 0,1795 |

$C$ là nút điều chỉnh giữa hai mong muốn trái nhau:

- **$C$ nhỏ** — lề rộng (5,61), chịu nhiều vi phạm, mô hình đơn giản hơn. Ở $C = 0{,}003$ thì **cả 114 vector hỗ trợ đều là điểm vi phạm** và không điểm nào nằm đúng trên lề: lề rộng tới mức nuốt gần hết dữ liệu.
- **$C$ lớn** — lề hẹp (1,73), cố ép từng điểm cho đúng.

> **Một lưu ý trung thực về bảng này.** Sai số kiểm tra chỉ chênh nhau trong khoảng 0,173–0,180, tức $C$ ảnh hưởng **ít** ở đây. Đó không phải thiếu sót của thí nghiệm mà là một sự thật: với **nhân tuyến tính** trên dữ liệu mà biên tối ưu vốn dĩ tuyến tính, SVM không có mấy chỗ để quá khớp. $C$ trở nên quan trọng hơn nhiều khi dùng nhân phi tuyến, vì khi ấy mô hình đủ linh hoạt để bám nhiễu.

### 13.5. Thủ thuật nhân

Mục 13.3 đã nêu chỗ mấu chốt: bài toán đối ngẫu chỉ phụ thuộc dữ liệu qua **tích vô hướng** $x_i^\top x_j$. Nên nếu muốn làm việc trong một không gian đặc trưng $\varphi(x)$ nào đó, ta **không cần** tính $\varphi(x)$ — chỉ cần tính được

$$K(x, x') = \varphi(x)^\top \varphi(x').$$

| Nhân | Công thức | Không gian đặc trưng |
|---|---|---|
| Tuyến tính | $x^\top x'$ | chính nó |
| Đa thức bậc $p$ | $(\gamma x^\top x' + r)^p$ | mọi đơn thức bậc $\le p$ |
| RBF (Gauss) | $\exp(-\gamma\|x - x'\|^2)$ | **vô hạn chiều** |

Dòng cuối là chỗ thủ thuật này trở nên đáng kinh ngạc: ta làm việc trong không gian vô hạn chiều mà mỗi phép tính chỉ tốn $O(d)$.

Thí nghiệm trên hai vòng tròn đồng tâm — bài toán không thể tách tuyến tính:

| Nhân | Độ chính xác |
|---|---|
| Tuyến tính | 0,6150 |
| Đa thức bậc 2 | **1,0000** |
| RBF | **1,0000** |

Nhân đa thức bậc 2 đạt 100% là điều đoán trước được: không gian đặc trưng của nó chứa $x_1^2 + x_2^2$, mà đó đúng là đại lượng phân biệt hai vòng tròn. Đây là minh hoạ sạch cho ý "chọn nhân là phát biểu một giả thiết về dữ liệu".

### 13.6. SVM nhiều lớp, và giới hạn thực tế

SVM sinh ra cho hai lớp. Ba cách mở rộng:

| Cách | Số mô hình | Ghi chú |
|---|---|---|
| Một–chọi–tất cả | $K$ | rẻ; điểm số giữa các mô hình không cùng thang đo |
| Một–chọi–một | $K(K-1)/2$ | mỗi mô hình nhỏ hơn nhiều; là mặc định của libsvm |
| Crammer–Singer | 1 | một bài toán duy nhất; đẹp về lý thuyết, ít dùng |

**Vì sao SVM ít được dùng ở quy mô lớn ngày nay.** Không phải vì nó kém chính xác. Lý do là chi phí: huấn luyện SVM nhân tốn khoảng $O(n^2)$ tới $O(n^3)$, và khi **suy luận** phải tính $K(x, x_i)$ với mọi vector hỗ trợ — mà số vector hỗ trợ thường tăng tuyến tính theo $n$. Với $n$ cỡ triệu thì cả hai đều không chấp nhận được, trong khi xuống dốc ngẫu nhiên trên mạng nơ-ron chỉ tốn $O(n)$ mỗi vòng.

> **SVM vẫn là lựa chọn tốt khi:** $n$ vừa phải (dưới vài chục nghìn), $d$ lớn so với $n$, và ta cần một mô hình mạnh mà không muốn chỉnh nhiều siêu tham số. Trên dữ liệu văn bản đã vector hoá thì SVM tuyến tính tới nay vẫn là một chuẩn mực khó vượt.

---

## 14. Giảm chiều: PCA, SVD và LDA

### 14.1. Ba lý do giảm chiều

1. **Tính toán** — thuật toán chạy nhanh hơn ở ít chiều hơn.
2. **Thống kê** — số chiều ít đi thì phương sai ước lượng nhỏ đi; đây là phạt chuẩn ở dạng khác.
3. **Con người** — không ai nhìn được 100 chiều, nhưng ai cũng nhìn được 2.

Và một lý do ngầm quan trọng hơn cả ba: **lời nguyền số chiều** ở Mục 7.2. Khoảng cách mất ý nghĩa ở chiều cao, nên mọi phương pháp dựa trên khoảng cách đều cần giảm chiều trước.

### 14.2. PCA: tìm hướng phương sai lớn nhất

Cho dữ liệu đã trừ trung bình $X_c$, PCA tìm hướng đơn vị $v$ làm phương sai của phép chiếu lớn nhất:

$$\max_{\|v\|=1} \; v^\top \Sigma v, \qquad \Sigma = \frac{1}{n-1}X_c^\top X_c.$$

Lập Lagrange với ràng buộc $\|v\|^2 = 1$ rồi cho đạo hàm bằng 0:

$$\Sigma v = \lambda v.$$

Tức **nghiệm chính là vector riêng của $\Sigma$**, và phương sai đạt được bằng đúng trị riêng $\lambda$. Các trục tiếp theo là các vector riêng tiếp theo, trực giao với các trục trước. Toàn bộ PCA nằm trong ba dòng này.

**Hai đường tính, một kết quả.** Có thể tính PCA bằng phân rã phổ của $\Sigma$, hoặc bằng SVD của $X_c$ trực tiếp:

```text
tri rieng cua ma tran hiep phuong sai : [11.52671  4.855133  2.064187  0.142466]
d_i^2/(n-1) tu SVD                    : [11.52671  4.855133  2.064187  0.142466]
lech lon nhat tren ca 8 tri: 7.105e-15
goc giua cac truc chinh: [1. 1. 1. 1.]
```

Khớp tới $7{,}1\times10^{-15}$. Trên thực tế **luôn dùng đường SVD**, vì nó tránh phải lập $X_c^\top X_c$ — thứ bình phương số điều kiện (Mục 2.6).

### 14.3. Định lý Eckart–Young: sai số bằng đúng tổng trị riêng bị bỏ

Đây là kết quả đẹp nhất của chương, và nó đúng **chính xác**, không phải xấp xỉ.

Giữ $k$ thành phần đầu rồi tái dựng, sai số Frobenius bằng đúng tổng các trị riêng bị vứt, nhân $(n-1)$:

$$\|X_c - X_k\|_F^2 = (n-1)\sum_{i > k}\lambda_i.$$

![Hình 10](figs/nt10_pca.png)

**Hình 10.** Trái: phương sai từng trục và luỹ kế. Phải: một trường hợp PCA chọn đúng hướng vô dụng cho phân loại.

| $k$ | $\|X_c - X_k\|_F^2$ | $(n-1)\sum_{i>k}\lambda_i$ | Lệch | % phương sai giữ |
|---|---|---|---|---|
| 1 | 2242,010603 | 2242,010603 | $9{,}1\times10^{-13}$ | 60,59 |
| 2 | 790,325815 | 790,325815 | $4{,}6\times10^{-13}$ | 86,11 |
| 3 | 173,133807 | 173,133807 | $6{,}3\times10^{-13}$ | **96,96** |
| 4 | 130,536554 | 130,536554 | $4{,}3\times10^{-13}$ | 97,71 |
| 8 | 0,000000 | 0,000000 | $3{,}6\times10^{-27}$ | 100,00 |

Dữ liệu được sinh từ 3 thành phần thật cộng nhiễu, và đúng là $k=3$ giữ được 96,96% phương sai — bước nhảy từ 86,11% lên 96,96% rồi chỉ nhích lên 97,71% chính là dấu hiệu nhận ra số chiều thật.

Định lý này nói thêm một điều mạnh hơn: trong **mọi** xấp xỉ hạng $k$ có thể có, PCA là cái tốt nhất theo chuẩn Frobenius. Không có phép chiếu tuyến tính nào làm tốt hơn.

### 14.4. Khi PCA làm hỏng việc — và LDA

PCA có một điểm mù lớn: **nó không nhìn nhãn**. Nó tối đa hoá phương sai, và phương sai lớn không đồng nghĩa có ích.

Thí nghiệm dựng dữ liệu có một hướng nhiễu biên độ lớn và một hướng tín hiệu biên độ nhỏ:

```text
truc chinh thu nhat cua PCA: [-1.  0.0067]   (giu 96.1% phuong sai)
huong cua LDA              : [0.0025  1.]
AUC khi chieu len truc PCA : 0.5020
AUC khi chieu len huong LDA: 0.9987
```

PCA giữ 96,1% phương sai mà **AUC chỉ 0,5020** — tức hoàn toàn vô dụng cho phân loại, không hơn đoán bừa. LDA chọn hướng gần như vuông góc với nó và đạt AUC 0,9987.

**LDA** sửa đúng điểm mù ấy bằng cách tối đa hoá tỉ số giữa độ tách giữa các lớp và độ tản bên trong lớp:

$$\max_w \;\frac{w^\top S_B\, w}{w^\top S_W\, w} \;\Longrightarrow\; w \propto S_W^{-1}(\mu_1 - \mu_0),$$

với $S_W$ là ma trận tản trong lớp và $S_B$ giữa các lớp.

| | PCA | LDA |
|---|---|---|
| Dùng nhãn | không | **có** |
| Tối đa hoá | phương sai toàn phần | tỉ số tách giữa lớp / tản trong lớp |
| Số chiều ra tối đa | $d$ | $K - 1$ |
| Dùng khi | nén, khử nhiễu, trực quan hoá | tiền xử lý cho phân loại |

Giới hạn $K-1$ chiều của LDA là hệ quả trực tiếp: $S_B$ dựng từ $K$ vector trung bình quanh trung bình chung nên chỉ có hạng $K-1$. Với bài toán hai lớp, LDA cho đúng **một** chiều.

Và như Mục 7.4 đã cho thấy, LDA ở đây **chính là** bộ phân lớp Gauss dùng chung ma trận hiệp phương sai — hai cách phát biểu, một thuật toán.

> **Quy tắc dùng.** Không có nhãn, hoặc muốn nén/khử nhiễu → PCA. Có nhãn và mục tiêu cuối là phân loại → thử LDA trước. Cần cấu trúc phi tuyến → PCA nhân hoặc t-SNE/UMAP để trực quan hoá (nhưng **không** dùng t-SNE làm bước tiền xử lý cho mô hình, vì nó không cho phép ánh xạ điểm mới).

---

## 15. Phân cụm: K-means

### 15.1. Bài toán và thuật toán

Không có nhãn, chỉ có $X$. Chia $n$ điểm thành $k$ cụm sao cho tổng bình phương khoảng cách tới tâm cụm nhỏ nhất:

$$\min_{S, \mu}\;\sum_{j=1}^{k}\sum_{x \in S_j}\|x - \mu_j\|^2.$$

Bài toán này **NP-khó**. Thuật toán Lloyd giải xấp xỉ bằng cách lặp hai bước:

1. **Gán:** mỗi điểm về tâm gần nhất.
2. **Cập nhật:** mỗi tâm thành trung bình các điểm thuộc về nó.

Cả hai bước đều **không làm tăng** hàm mục tiêu, và số cách phân hoạch là hữu hạn, nên thuật toán chắc chắn dừng. Nhưng nó chỉ dừng ở **cực tiểu địa phương** — đây là hệ quả trực tiếp của Chương 11: hàm mục tiêu không lồi theo $(S,\mu)$.

### 15.2. Khởi tạo quyết định kết quả

Thí nghiệm chạy 200 lần trên dữ liệu 8 cụm, mỗi lần một hạt giống khác:

| Cách khởi tạo | Quán tính nhỏ nhất | Trung bình | % lần kẹt ở nghiệm tồi |
|---|---|---|---|
| Ngẫu nhiên thuần | 407,992 | 517,838 | **71,5** |
| k-means++ | 407,992 | 476,529 | **46,5** |

Khởi tạo ngẫu nhiên thuần kẹt ở nghiệm tồi **71,5% số lần**. Con số ấy nói rằng chạy K-means đúng một lần rồi tin kết quả là sai lầm nghiêm trọng.

**k-means++** chọn các tâm ban đầu tuần tự, mỗi tâm mới được rút với xác suất tỉ lệ $d^2$ tới tâm gần nhất đã chọn — tức ưu tiên những chỗ xa. Nó hạ tỉ lệ kẹt xuống 46,5%, tức cải thiện rõ nhưng **không** giải quyết triệt để.

> **Hệ quả thực tế.** Luôn chạy nhiều lần rồi giữ kết quả có quán tính nhỏ nhất. `sklearn.cluster.KMeans` mặc định `n_init=10` và `init='k-means++'` đúng vì lý do này.

### 15.3. K-means giả định gì

![Hình 11](figs/nt11_kmeans.png)

**Hình 11.** Cùng một thuật toán trên ba hình dạng dữ liệu. Nó chỉ làm tốt ở hình đầu.

| Hình dạng cụm | Độ chính xác gom cụm |
|---|---|
| Ba cụm tròn, tách rời | **1,0000** |
| Hai hình lưỡi liềm lồng nhau | 0,7517 |
| Hai dải dẹt nằm ngang | **0,5317** |

Dòng cuối gần như bằng đoán bừa (0,5) — K-means thất bại hoàn toàn.

**Vì sao?** Vì nó gán mỗi điểm cho tâm **gần nhất theo khoảng cách Euclid**, nên biên giữa hai cụm bất kỳ luôn là một **siêu phẳng** — chính xác là mặt trung trực của đoạn nối hai tâm. Ba giả định ngầm theo sau:

1. Cụm có dạng **lồi và gần hình cầu**.
2. Các cụm có **kích thước tương đương**.
3. Mọi chiều **cùng thang đo** — nếu không thì khoảng cách Euclid vô nghĩa.

Hai dải dẹt vi phạm giả định 1: cụm đúng bị kéo dài theo chiều ngang, nên mặt phẳng chia tối ưu theo Euclid lại cắt ngang chúng thay vì tách chúng.

Khi các giả định ấy sai, có lựa chọn khác: **mô hình hỗn hợp Gauss** cho phép cụm hình elip; **DBSCAN** cho phép cụm hình dạng tuỳ ý và tự xác định số cụm; **phân cụm phổ** biến đổi không gian trước rồi mới dùng K-means.

### 15.4. Chọn $k$

Hàm mục tiêu **luôn giảm** khi $k$ tăng — với $k = n$ thì quán tính bằng 0. Nên không thể chọn $k$ bằng cách tối thiểu hoá nó.

| $k$ | Quán tính | Giảm so với $k-1$ |
|---|---|---|
| 5 | 902,24 | 21,6% |
| 6 | 727,79 | 19,3% |
| 7 | 563,93 | 22,5% |
| **8** | **407,99** | **27,7%** |
| 9 | 385,82 | **5,4%** |
| 10 | 364,76 | 5,5% |

Dữ liệu có đúng 8 cụm, và bảng chỉ đúng chỗ ấy: mức giảm là 27,7% khi lên $k=8$ rồi rơi xuống 5,4% khi lên $k=9$. Đó là **khuỷu tay** — chỗ mà thêm cụm không còn mua được gì nhiều.

> **Nhưng khuỷu tay không phải một định nghĩa toán học.** Nó là một quy tắc nhìn bằng mắt, và trên dữ liệu thật nó thường không rõ ràng như bảng trên. Các lựa chọn khác: **điểm bóng** (silhouette) đo mức một điểm hợp với cụm của nó hơn cụm gần nhất, hoặc **thống kê khoảng trống** (gap statistic) so quán tính với quán tính trên dữ liệu ngẫu nhiên. Và thường thì $k$ được quyết định bởi nghiệp vụ chứ không bởi dữ liệu.

---

## 16. Hệ gợi ý

### 16.1. Bài toán, và vì sao nó khác

Cho ma trận đánh giá $R \in \mathbb{R}^{n_u \times n_i}$ mà **phần lớn ô để trống**, hãy đoán các ô trống. Đây không hẳn là học có giám sát cũng không hẳn là không giám sát: "nhãn" là các ô đã quan sát, nhưng chúng cũng chính là đặc trưng.

Hai họ phương pháp:

- **Dựa trên nội dung** — mô tả mỗi mục bằng đặc trưng (thể loại, đạo diễn, từ khoá), rồi học sở thích của từng người theo các đặc trưng ấy. Về bản chất là một bài hồi quy cho mỗi người.
- **Lọc cộng tác** — không dùng đặc trưng của mục nào cả; chỉ dùng *mẫu hình* đánh giá. Ý tưởng: những người đồng ý với nhau trong quá khứ thì sẽ đồng ý tiếp.

### 16.2. Phân rã ma trận

Cách lọc cộng tác hiệu quả nhất. Giả sử $R$ có **hạng thấp**:

$$R \approx P Q^\top, \qquad P \in \mathbb{R}^{n_u \times k},\; Q \in \mathbb{R}^{n_i \times k}.$$

Mỗi người dùng thành một vector $k$ chiều, mỗi mục cũng vậy, và đánh giá dự đoán là tích vô hướng của chúng. Các chiều ấy là **yếu tố ẩn** — không ai định nghĩa chúng, chúng nổi lên từ dữ liệu.

Hàm mất mát chỉ tính trên các ô đã quan sát:

$$\min_{P,Q}\;\sum_{(u,i) \in \Omega}\big(r_{ui} - p_u^\top q_i\big)^2 + \lambda\big(\|P\|_F^2 + \|Q\|_F^2\big).$$

Bài toán này **không lồi** theo $(P, Q)$ cùng lúc, nhưng **lồi theo từng cái khi cố định cái kia** — mỗi bước chính là một bài ridge. Đó là cơ sở của **bình phương tối thiểu luân phiên** (ALS): cố định $Q$ giải $P$, cố định $P$ giải $Q$, lặp lại.

Giả thiết hạng thấp là thứ làm bài toán có nghĩa. Với 300 người và 200 mục, mô hình hạng 4 có 2 000 tham số tự do thay cho 60 000 ô — chỉ **3,3%**. Không có giả thiết ấy thì các ô chưa thấy là vô phương đoán.

### 16.3. Cần bao nhiêu dữ liệu thì nó mới chạy

Đây là kết quả quan trọng nhất của chương, và nó có một ngưỡng sắc nét.

![Hình 12](figs/nt12_recsys.png)

**Hình 12.** Trái: RMSE trên ô chưa thấy theo số đánh giá mỗi người (trục dọc log). Đường chấm là mức đoán bừa bằng 0; đường đứt là mốc $5k$.

| % ô thấy được | Đánh giá / người | RMSE ô chưa thấy | So với đoán bừa |
|---|---|---|---|
| 2 | 4,1 | 2,4759 | **1,19** |
| 3 | 5,9 | 3,0167 | **1,45** |
| 5 | 10,1 | 3,5444 | **1,71** |
| 10 | 19,8 | 0,3048 | 0,15 |
| 20 | 39,6 | 0,1834 | 0,09 |
| 40 | 80,5 | 0,1237 | 0,06 |

Cột cuối là tỉ số giữa RMSE của mô hình và độ lệch chuẩn của dữ liệu. **Lớn hơn 1 nghĩa là mô hình còn tệ hơn đoán bừa bằng 0.**

Ba dòng đầu đều lớn hơn 1. Với dưới khoảng 10 đánh giá mỗi người, phân rã ma trận không chỉ kém mà **có hại**. Rồi giữa 10,1 và 19,8 đánh giá, RMSE rơi từ 3,54 xuống 0,30 — **giảm hơn 11 lần**.

**Vì sao có ngưỡng, và ngưỡng ở đâu.** Mỗi người dùng có $k = 4$ ẩn số cần ước lượng. Về nguyên tắc cần ít nhất 4 ràng buộc, nhưng ràng buộc lại nhiễu và bản thân $Q$ cũng đang được ước lượng, nên thực tế cần vài lần con số ấy. Ngưỡng đo được rơi vào khoảng $5k$.

> **Quy tắc bỏ túi:** cần cỡ **5 lần số yếu tố ẩn** đánh giá cho mỗi người dùng thì phân rã ma trận mới bắt đầu có ích. Đây là lý do các hệ thực tế dùng $k$ nhỏ đáng ngạc nhiên (10–200) dù có hàng triệu người dùng.

### 16.4. Chọn hạng $k$

Đo ở mật độ 30%, nơi việc khớp là đáng tin:

| Hạng $k$ | RMSE ô **chưa** thấy | RMSE ô **đã** thấy |
|---|---|---|
| 1 | 1,8073 | 1,7769 |
| 2 | 1,4764 | 1,4206 |
| 3 | 1,0569 | 1,0128 |
| **4** | **0,1448** | 0,3762 |
| 6 | 0,2217 | 0,3535 |
| 10 | 0,3501 | 0,3076 |
| 20 | 0,6291 | **0,1903** |

Hai cột kể hai câu chuyện trái ngược, và đó chính là điều cần thấy:

- Cột **đã thấy** giảm đơn điệu theo $k$ — mô hình mạnh hơn thì khớp dữ liệu quan sát chặt hơn. Ở $k=20$ nó đạt 0,1903, tốt nhất bảng.
- Cột **chưa thấy** chạm đáy tại $k = 4$ — đúng hạng thật — rồi **tăng trở lại**. Ở $k=20$ nó là 0,6291, tệ hơn 4,3 lần so với $k=4$.

Đây là đánh đổi thiên lệch–phương sai xuất hiện lần nữa, lần này ở dạng chọn hạng. $k$ nhỏ quá thì thiếu khớp; $k$ lớn quá thì bắt đầu khớp cả nhiễu quan sát.

### 16.5. Khởi đầu lạnh

Người dùng mới không có đánh giá nào. Phân rã ma trận khi ấy không có gì để dựa vào:

```text
RMSE cho nguoi dung cu   : 0.1459
RMSE cho nguoi dung moi  : 2.2364
RMSE neu doan bua bang 0 : 2.0772
||P|| trung binh, nguoi cu : 2.0841
||P|| trung binh, nguoi moi: 1.54e-01
```

Ba con số cuối giải thích cơ chế: vector của người dùng mới có chuẩn **0,154** so với 2,084 của người dùng cũ — tức bị kéo gần về 0. Lý do nằm ngay trong hàm mất mát ở Mục 16.2: với người không có ô quan sát nào, số hạng thứ nhất trống rỗng, nên chỉ còn phạt chuẩn $\lambda\|p_u\|^2$, và cực tiểu của nó là $p_u = 0$.

Kết quả là mô hình dự báo 0 cho mọi mục, và RMSE 2,2364 **tệ hơn cả đoán bừa** 2,0772.

| Dạng khởi đầu lạnh | Cách chữa |
|---|---|
| Người dùng mới | hỏi vài sở thích lúc đăng ký; dùng thông tin nhân khẩu; gợi ý mục phổ biến |
| Mục mới | dùng **đặc trưng nội dung** của mục |
| Hệ thống mới | bắt đầu bằng phương pháp dựa trên nội dung, chuyển dần sang cộng tác |

> **Đây là lý do mọi hệ gợi ý thực tế đều là hệ lai.** Lọc cộng tác mạnh hơn khi có đủ dữ liệu, nhưng nó **không có đường lùi** khi không có. Một mô hình dựa trên nội dung yếu hơn nhưng luôn trả lời được là thứ bắt buộc phải có bên cạnh.

Và còn một vấn đề nữa mà chương này không đo được: **vòng phản hồi thoái hoá** — hệ chỉ gợi ý thứ nó đã biết, nên chỉ thu được phản hồi về thứ đó, nên càng ngày càng hẹp. [Mục 12.5 của *MLOps*](mlops-ch12.html) đo hiện tượng ấy và cho thấy không ngẫu nhiên hoá thì hệ vĩnh viễn chỉ nhìn thấy 9,5% danh mục.

---

## 17. Bài tập

**Bài 1 (tính tay).** Cho bốn điểm $(x, y) = (1,2), (2,3), (3,5), (4,6)$ và mô hình $y = w_0 + w_1 x$.
(a) Lập ma trận $X$ có cột hằng số, rồi tính $X^\top X$ và $X^\top y$.
(b) Giải phương trình chuẩn tắc để tìm $w_0$, $w_1$.
(c) Tính phần dư của từng điểm. Tổng của chúng bằng bao nhiêu, và vì sao con số ấy **không** phải trùng hợp?
(d) Lặp lại với ridge $\lambda = 1$. Hệ số đổi theo chiều nào?

**Bài 2 (suy luận).** Giả sử $y_i = w^\top x_i + \epsilon_i$ với $\epsilon_i \sim \mathcal{N}(0,\sigma^2)$ độc lập, và tiên nghiệm $w \sim \mathcal{N}(0, \tau^2 I)$.
(a) Viết $-\log p(w \mid \mathcal{D})$, bỏ các hằng số không phụ thuộc $w$.
(b) Suy ra rằng cực tiểu của nó đúng bằng nghiệm ridge, và chỉ ra $\lambda = \sigma^2/\tau^2$.
(c) Tiên nghiệm càng **chặt** (nghĩa là $\tau$ càng nhỏ) thì $\lambda$ đi theo chiều nào? Giải thích bằng lời tại sao chiều ấy hợp lý.
(d) Nếu đổi tiên nghiệm sang Laplace thì được thuật toán gì, và điều đó phát biểu niềm tin gì về $w$?

**Bài 3 (tính tay).** Mục 5.3 đo được: với $\kappa = 10\,000$, gradient descent cần 92 104 vòng còn thêm quán tính chỉ cần 1 297 vòng.
(a) Ước lượng số vòng của cả hai phương pháp khi $\kappa = 40\,000$, dựa vào bậc lý thuyết $O(\kappa)$ và $O(\sqrt\kappa)$.
(b) Một bạn nói: "bài toán của tôi có 1 triệu chiều nên gradient descent sẽ rất chậm." Câu ấy đúng hay sai, và vì sao?
(c) Nêu **một** bước tiền xử lý rẻ tiền làm giảm $\kappa$, và giải thích cơ chế.

**Bài 4 (suy luận).** Sai số của sai phân trung tâm gồm hai phần: cắt cụt cỡ $C_1\varepsilon^2$ và làm tròn cỡ $C_2 u/\varepsilon$, với $u \approx 2{,}22\times10^{-16}$.
(a) Cực tiểu hoá tổng hai phần theo $\varepsilon$ và chứng tỏ $\varepsilon^* \propto u^{1/3}$.
(b) Làm tương tự cho sai phân tiến (cắt cụt cỡ $C_1\varepsilon$) và chứng tỏ $\varepsilon^* \propto u^{1/2}$.
(c) Đối chiếu với hai giá trị đo được ở Mục 5.5.
(d) Vì sao sai phân trung tâm đắt gấp đôi mà vẫn được ưa dùng hơn?

**Bài 5 (tính tay).** Một tập dữ liệu tách được tuyến tính có mọi điểm nằm trong hình cầu bán kính $R = 2$ và lề $\gamma = 0{,}1$.
(a) Chặn Novikoff cho số lần sai là bao nhiêu?
(b) Nếu thêm 10 000 điểm nữa (vẫn tách được, vẫn cùng $R$ và $\gamma$) thì chặn đổi thế nào? Điều đó nói lên gì?
(c) Nếu nhân mọi $x_i$ với 10 thì $R$, $\gamma$ và chặn đổi thế nào?
(d) Vì sao perceptron **không** có bảo đảm nào khi dữ liệu không tách được? Trả lời bằng hình dạng của hàm mất mát ở Hình 5.

**Bài 6 (tính tay).** Một mô hình phát hiện gian lận cho ma trận nhầm lẫn sau trên 20 000 giao dịch:

| | Dự đoán gian lận | Dự đoán bình thường |
|---|---|---|
| **Thật gian lận** | 102 | 94 |
| **Thật bình thường** | 105 | 19 699 |

(a) Tính độ chính xác, precision, recall và $F_1$.
(b) Tính độ chính xác của bộ phân loại "luôn đoán bình thường". So với (a).
(c) Kết quả (b) nói gì về việc dùng độ chính xác để chọn mô hình ở đây?
(d) Bộ phận rủi ro nói: "bỏ sót một ca gian lận tốn gấp 20 lần một lần báo động nhầm." Nên dịch ngưỡng theo chiều nào, và chỉ số nào nên dùng để chọn ngưỡng?

**Bài 7 (tính tay).** SVM lề cứng trên đúng hai điểm: $x_1 = (1,1)$ nhãn $+1$ và $x_2 = (-1,-1)$ nhãn $-1$.
(a) Dùng đối xứng để lập luận rằng $b = 0$, rồi giải bài toán gốc tìm $w$.
(b) Tính lề $2/\|w\|$ và so với khoảng cách giữa hai điểm. Giải thích kết quả.
(c) Tìm $\alpha_1$, $\alpha_2$ từ $w = \sum\alpha_i y_i x_i$ và ràng buộc $\sum\alpha_i y_i = 0$.
(d) Tính giá trị bài toán gốc $\tfrac12\|w\|^2$ và giá trị bài toán đối ngẫu. Khe đối ngẫu bằng bao nhiêu?
(e) Thêm một điểm $x_3 = (5,5)$ nhãn $+1$. Nghiệm có đổi không? Trả lời bằng điều kiện bù trừ.

**Bài 8 (tính tay).** Một ma trận hiệp phương sai có các trị riêng $\lambda = (10;\, 5;\, 3;\, 1{,}5;\, 0{,}5)$.
(a) Giữ bao nhiêu thành phần thì được ít nhất 85% phương sai?
(b) Với $k = 2$ và $n = 101$ mẫu, sai số tái dựng $\|X_c - X_k\|_F^2$ bằng bao nhiêu?
(c) Một bạn nói: "giữ 90% phương sai nên mô hình phân loại sẽ chỉ kém đi một chút." Dùng kết quả ở Mục 14.4 để bác bỏ.
(d) Nếu đổi đơn vị của một đặc trưng từ mét sang milimét thì các trị riêng đổi thế nào? Rút ra điều gì về bước chuẩn hoá trước PCA?

**Bài 9 (chẩn đoán).** Với mỗi tình huống, cho biết vấn đề và **hai** việc nên làm:
(a) Chạy K-means 5 lần trên cùng dữ liệu, được 5 kết quả khác hẳn nhau.
(b) K-means chia đôi một cụm dài và dẹt rõ ràng, trong khi gộp hai cụm tròn nhỏ gần nhau.
(c) Quán tính giảm đều đặn theo $k$, không thấy khuỷu tay nào.
(d) Một cụm chứa 98% số điểm, bốn cụm còn lại mỗi cụm vài điểm.

**Bài 10 (thiết kế).** Một hệ gợi ý có 1 triệu người dùng và 100 nghìn mục, dự định dùng phân rã ma trận hạng $k = 50$.
(a) Mô hình có bao nhiêu tham số? Bằng bao nhiêu phần trăm số ô của ma trận đầy đủ?
(b) Dùng quy tắc ở Mục 16.3, mỗi người cần khoảng bao nhiêu đánh giá? Tổng cộng bao nhiêu, và bằng bao nhiêu phần trăm số ô?
(c) Thực tế chỉ có trung bình 30 đánh giá mỗi người. Dự đoán chuyện gì xảy ra, và nêu **hai** cách xử lý.
(d) Sản phẩm yêu cầu mọi người dùng mới đều phải nhận được gợi ý ngay từ phiên đầu tiên. Thiết kế đường dự phòng, và nói rõ khi nào thì chuyển sang phân rã ma trận.

---

## 18. Ôn phỏng vấn

### 18.1. Khung trả lời

Giống ba giáo trình còn lại, khung ba bước dùng được cho gần hết câu hỏi dạng "vì sao X":

1. **Nêu vấn đề X sinh ra để chữa.**
2. **Nêu cơ chế** — tốt nhất là một dòng toán hoặc một con số.
3. **Nêu cái giá** — X đắt ở đâu, hỏng khi nào.

Với phần nền tảng có thêm một bước đáng giá nữa: **nối nó với một thứ khác**. Người phỏng vấn nhớ rất lâu câu trả lời chỉ ra được rằng hai thứ tưởng khác nhau thực ra là một.

### 18.2. Nhóm hồi quy và tối ưu hoá

**"Vì sao dùng bình phương sai lệch?"**

> Vì nó là **âm log hợp lý dưới giả thiết nhiễu Gauss**. Không phải lựa chọn thẩm mỹ — nó suy ra được.
>
> *Nối sang chỗ khác:* nếu tin nhiễu có đuôi dày hơn, MLE dưới giả thiết Laplace cho ra hồi quy trị tuyệt đối, bền hơn với điểm ngoại lai. Đổi giả thiết về nhiễu là đổi hàm mất mát.

**"Khi nào phương trình chuẩn tắc không giải được, và làm gì?"**

> Khi $X$ không đủ hạng cột — nhiều đặc trưng hơn mẫu, hoặc có đặc trưng phụ thuộc tuyến tính (kinh điển: one-hot đủ $K$ mức mà vẫn giữ cột hằng số).
>
> *Đọc đúng bản chất:* vấn đề không phải "không có nghiệm" mà là **có vô số nghiệm cho cùng một dự báo**. Giả nghịch đảo chọn cái chuẩn nhỏ nhất, và đó chính là ridge khi $\lambda \to 0^+$.

**"Cộng tuyến gây hại thế nào?"**

> *Kể một con số:* trong thí nghiệm ở Mục 4.4, khi tương quan giữa hai cột đi từ 0 lên 0,999 thì độ dao động của **hệ số** tăng **25 lần**, còn độ dao động của **dự báo** đi từ 0,0586 xuống 0,0585 — tức không đổi.
>
> *Kết luận:* cộng tuyến là vấn đề của **diễn giải**, không phải của dự báo. Nếu chỉ cần dự báo thì bỏ qua được.

**"Gradient descent chạy nhanh hay chậm phụ thuộc gì?"**

> Phụ thuộc **số điều kiện** $\kappa$, không phụ thuộc số chiều. Số vòng lặp là $O(\kappa\log\frac1\varepsilon)$.
>
> *Kể một con số:* đo ở Mục 5.3 — $\kappa$ đi từ 1 tới 10 000 làm số vòng đi từ 1 tới **92 104**, tức tăng đúng tuyến tính theo $\kappa$. Thêm quán tính đưa nó về $O(\sqrt\kappa)$, chỉ còn **1 297** vòng.
>
> *Điều nên nói thêm:* vì vậy **chuẩn hoá đặc trưng về cùng thang đo** là cách rẻ nhất để tăng tốc huấn luyện — nó giảm $\kappa$ trực tiếp.

**"Kiểm tra gradient thế nào?"**

> Sai phân trung tâm với $\varepsilon \approx 10^{-5}$ tới $10^{-6}$, so sai số **tương đối**.
>
> *Ghi điểm thêm — chỗ hầu hết mọi người nói sai:* $\varepsilon$ **không** phải càng nhỏ càng tốt. Có hai nguồn sai số đánh nhau: cắt cụt cỡ $\varepsilon^2$ và làm tròn cỡ $u/\varepsilon$. Điểm tối ưu là $\varepsilon^* \sim u^{1/3} \approx 6\times10^{-6}$, và đo được là $5{,}6\times10^{-6}$. Đặt $\varepsilon = 10^{-12}$ cho kết quả **tệ hơn** $10^{-6}$ khoảng sáu bậc độ lớn.

### 18.3. Nhóm phân loại

**"Perceptron, hồi quy logistic và SVM khác nhau ở đâu?"**

> Câu trả lời tốt bắt đầu bằng chỗ chúng **giống** nhau: cùng mô hình tuyến tính $w^\top x + b$, cùng tối ưu bằng xuống dốc. Chúng chỉ khác **hàm mất mát**, và cả ba đều là chặn trên lồi của mất mát 0–1:
>
> | | Mất mát | Hệ quả |
> |---|---|---|
> | Perceptron | $\max(0,-m)$ | phẳng khi $m>0$ → dừng ngay khi vừa đúng |
> | Hinge | $\max(0,1-m)$ | phẳng khi $m>1$ → sinh ra khái niệm **lề** |
> | Logistic | $\log(1+e^{-m})$ | không bao giờ phẳng → cho ra **xác suất** |
>
> *Vì sao phải thay mất mát 0–1:* nó không lồi và có đạo hàm bằng 0 ở mọi chỗ khả vi, nên không tối ưu trực tiếp được.

**"Hồi quy logistic và softmax quan hệ thế nào?"**

> Softmax với $K=2$ **chính là** hồi quy logistic. Softmax có $K$ vector trọng số nhưng chỉ **hiệu** của chúng là xác định được, và với $K=2$ thì hiệu ấy đúng là $w$ của logistic.
>
> *Kể một con số:* đo ở Mục 6.4 — khớp cả hai tới hội tụ, xác suất lệch nhau $3{,}3\times10^{-16}$ và trọng số lệch $1{,}6\times10^{-15}$, tức sai số máy.

**"Vì sao hồi quy logistic luôn cần phạt chuẩn?"**

> Vì nếu dữ liệu **tách được hoàn toàn** thì hợp lý cực đại không có nghiệm hữu hạn: nhân $w$ lên bất kỳ hệ số nào cũng làm hợp lý tăng, nên $\|w\| \to \infty$.
>
> *Kể một con số:* đo ở Mục 6.5 — không phạt chuẩn thì $\|w\|$ đi 10,2 → 15,9 → 24,7 → **34,8** qua các mốc 500 tới 50 000 vòng, không có dấu hiệu dừng; có phạt chuẩn $\lambda = 0{,}01$ thì nó đứng yên ở 3,993 từ vòng 500.
>
> *Hệ quả dùng được:* `LogisticRegression` của scikit-learn mặc định $C=1$ chính vì lý do này. Và nếu thấy hệ số lớn bất thường thì nên nghi có đặc trưng rò rỉ nhãn.

**"Naive Bayes khác LDA và QDA thế nào?"**

> Cả ba là **cùng một bộ phân lớp sinh Gauss**, chỉ khác ràng buộc đặt lên ma trận hiệp phương sai:
>
> | | $\Sigma$ | Biên |
> |---|---|---|
> | Naive Bayes Gauss | riêng từng lớp, **chéo** | bậc hai |
> | LDA | **dùng chung**, đầy đủ | tuyến tính |
> | QDA | riêng từng lớp, đầy đủ | bậc hai |
>
> *Giả thiết "naive" chính là ép $\Sigma$ thành ma trận đường chéo* — tức giả định các đặc trưng độc lập khi đã biết lớp.
>
> *Vì sao LDA cho biên tuyến tính:* khi hai lớp dùng chung $\Sigma$, số hạng bậc hai $x^\top\Sigma^{-1}x$ **triệt tiêu** khi lấy hiệu hai hàm phân biệt. Đo được ở Mục 7.4: hàm quyết định của LDA khớp tuyến tính với sai số $4{,}4\times10^{-15}$, còn QDA thì sai số 17,8 — trật hẳn.
>
> *Cái giá của QDA:* phải ước lượng $K$ ma trận $d\times d$ thay vì một, nên tốn dữ liệu gấp $K$ lần.

**"Vì sao Naive Bayes vẫn chạy tốt dù giả thiết độc lập gần như luôn sai?"**

> Vì để phân loại đúng ta chỉ cần **thứ tự** giữa các $p(y=k\mid x)$ đúng, không cần bản thân các xác suất đúng. Ước lượng xác suất của Naive Bayes thường rất tệ (dồn về 0 hoặc 1); nhãn thì thường ổn. Đây là lý do không nên dùng đầu ra của nó làm xác suất.

### 18.4. Nhóm đánh giá

**"Vì sao không dùng độ chính xác?"**

> *Kể một con số, và đây là con số đắt nhất trong cả giáo trình:* trên dữ liệu có 0,98% lớp dương, bộ phân loại **đoán tất cả là âm** đạt độ chính xác **0,9902**, trong khi bộ phân loại thật sự dùng được ($F_1 = 0{,}51$) chỉ đạt **0,9900**. Chọn theo độ chính xác là chọn đúng cái vô dụng.

**"ROC-AUC hay PR-AUC?"**

> *Cùng một mô hình, cùng một bộ dữ liệu:* ROC-AUC = **0,9715** nhưng PR-AUC = **0,4931**.
>
> *Cơ chế, và đây là chỗ phân biệt hiểu với thuộc:* mẫu số của FPR là **toàn bộ lớp âm**, mà lớp âm chiếm 99%. Thêm hàng trăm báo động nhầm gần như không làm FPR nhúc nhích. Precision thì có mẫu số chỉ gồm những ca được báo động, nên nó cảm nhận được ngay.
>
> *Mốc so sánh:* PR-AUC của bộ đoán ngẫu nhiên bằng đúng **tỉ lệ lớp dương** (ở đây 0,01), còn ROC-AUC của nó luôn là 0,5 bất kể mất cân bằng.

**"Precision và recall đánh đổi thế nào?"**

> Qua ngưỡng. *Đo được ở Mục 8.1:* hạ ngưỡng từ 2,6 xuống −1,0 làm recall đi 0,52 → 1,00 còn precision đi 0,49 → 0,012.
>
> *Điểm cần nói:* chọn ngưỡng là **quyết định nghiệp vụ**, phụ thuộc giá của một lần bỏ sót so với một lần báo động nhầm. Không có ngưỡng "đúng" về mặt kỹ thuật.

### 18.5. Nhóm phạt chuẩn và xác suất

**"Ridge khác lasso thế nào?"**

> Ridge phạt $\|w\|_2^2$, lasso phạt $\|w\|_1$. Lasso cho hệ số **đúng bằng 0**, ridge thì không bao giờ.
>
> *Cơ chế:* $|w|$ có **điểm gãy** tại 0, dưới vi phân là cả đoạn $[-1,1]$, nên cần gradient thắng một lực có độ lớn cố định mới kéo được hệ số ra khỏi 0. Với $w^2$ thì đạo hàm là $2w$, yếu dần khi gần 0.
>
> *Kể một con số:* ở $\lambda = 100$, lasso đưa **9 trong 12** hệ số về đúng 0, và mô hình sinh dữ liệu cũng có đúng 9 hệ số bằng 0. Ridge đưa được **0**.

**"Ridge co ngót đều mọi hướng phải không?"**

> **Không** — và đây là chỗ hay bị nói sai. Qua SVD, hệ số co ngót theo hướng riêng thứ $i$ là $d_i^2/(d_i^2+\lambda)$.
>
> *Kể một con số:* với $\lambda=10$, hướng có $d_i = 12{,}27$ chỉ bị co còn 0,938, trong khi hướng có $d_i = 6{,}17$ bị co còn 0,792. Ridge **co mạnh nhất đúng những hướng dữ liệu nói ít nhất** — đó chính là lý do nó chữa được cộng tuyến.

**"Phạt chuẩn từ đâu ra?"**

> Từ **tiên nghiệm**. MAP cực tiểu $-\log p(\mathcal{D}\mid\theta) - \log p(\theta)$, trong đó số hạng thứ nhất là mất mát và số hạng thứ hai là phạt chuẩn.
>
> | Phạt | Tiên nghiệm | Niềm tin |
> |---|---|---|
> | $\ell_2$ | Gauss | hệ số nên nhỏ |
> | $\ell_1$ | Laplace | phần lớn hệ số nên bằng 0 |
> | Làm trơn Laplace | Dirichlet | mọi kết cục đều có thể |
>
> *Kể một con số:* ridge dạng đóng và MAP tối ưu bằng BFGS trên hậu nghiệm — hai đường tính hoàn toàn khác nhau — cho nghiệm lệch $2{,}0\times10^{-8}$, với $\lambda = \sigma^2/\tau^2$ đúng như lý thuyết.
>
> *Ranh giới cần nói rõ:* MAP **không phải** suy luận Bayes đầy đủ. Nó lấy một điểm — đỉnh hậu nghiệm — rồi vứt phần còn lại, nên không cho độ bất định.

### 18.6. Nhóm tối ưu lồi và SVM

**"Vì sao quan tâm tới tính lồi?"**

> Vì với hàm lồi, **mọi cực tiểu địa phương đều là cực tiểu toàn cục**. Nên điểm khởi tạo không quan trọng và khi dừng thì ta biết mình đã ở nghiệm tốt nhất.
>
> *Kể một con số:* thí nghiệm ở Mục 11.6 chạy xuống dốc từ 21 điểm xuất phát. Hàm lồi cho **một** điểm dừng; hàm $x^4-3x^2+x/2$ cho **hai**, với giá trị $-2{,}87$ và $-1{,}65$ — tức gần một nửa số lần chạy kết thúc ở nghiệm không phải toàn cục.
>
> *Nối sang chỗ khác:* hàm mất mát của mạng nơ-ron không lồi, và đó chính là lý do khởi tạo, chuẩn hoá và kết nối tắt trở thành vấn đề sống còn ở đó.

**"Vì sao SVM đi đường vòng qua bài toán đối ngẫu?"**

> Ba lý do, nhưng lý do thật là lý do thứ ba: bài toán đối ngẫu chỉ phụ thuộc dữ liệu qua **tích vô hướng** $x_i^\top x_j$. Ngay khi thấy điều đó thì **thủ thuật nhân** trở thành hiển nhiên — thay tích vô hướng bằng $K(x_i,x_j)$ là xong. Không chuyển sang đối ngẫu thì không nhìn ra.
>
> Hai lý do còn lại: đối ngẫu có $n$ biến thay vì $d$ (thắng khi $d \gg n$), và nó cho chặn dưới để biết còn cách tối ưu bao xa.

**"Vector hỗ trợ là gì, và vì sao chỉ chúng quan trọng?"**

> Là những điểm có $\alpha_i > 0$. Theo **điều kiện bù trừ** của KKT, $\alpha_i f_i(x^*) = 0$, nên ràng buộc lỏng thì $\alpha_i = 0$ — tức điểm nằm ngoài lề **không ảnh hưởng gì** tới nghiệm.
>
> *Điều nên nhấn:* đây không phải một thiết kế mà là một **hệ quả** của KKT.
>
> *Kể một con số:* đo ở Mục 13.3 — **3 trên 120 điểm** quyết định toàn bộ nghiệm. Xoá 117 điểm còn lại thì kết quả không đổi.

**"$C$ trong SVM lề mềm làm gì?"**

> Điều khiển đánh đổi giữa lề rộng và số điểm được phép vi phạm. Và KKT chia các điểm thành **đúng ba nhóm**: $\alpha=0$ nằm ngoài lề, $0<\alpha<C$ nằm đúng trên lề, $\alpha=C$ vi phạm lề.
>
> *Kể một con số:* ở $C = 0{,}003$ lề rộng 5,61 với **cả 114 vector hỗ trợ đều là điểm vi phạm**; ở $C=300$ lề hẹp 1,73 với 45 vector hỗ trợ.
>
> *Trung thực về giới hạn:* với **nhân tuyến tính** trên dữ liệu mà biên tối ưu vốn tuyến tính, $C$ ảnh hưởng ít — sai số kiểm tra chỉ chênh 0,173 so với 0,180. $C$ quan trọng hơn nhiều với nhân phi tuyến.

**"Thủ thuật nhân là gì?"**

> Tính $K(x,x') = \varphi(x)^\top\varphi(x')$ mà **không bao giờ** tính $\varphi(x)$. Với nhân RBF thì $\varphi(x)$ nằm ở không gian **vô hạn chiều**, nhưng mỗi phép tính chỉ tốn $O(d)$.
>
> *Kể một con số:* trên dữ liệu hai vòng tròn đồng tâm, nhân tuyến tính đạt 0,615 còn nhân đa thức bậc 2 và RBF đều đạt **1,000**. Nhân bậc 2 đạt trọn vì không gian đặc trưng của nó chứa $x_1^2+x_2^2$ — đúng đại lượng phân biệt hai vòng tròn.

**"Vì sao SVM ít dùng ở quy mô lớn?"**

> Không phải vì kém chính xác mà vì chi phí: huấn luyện cỡ $O(n^2)$–$O(n^3)$, và khi suy luận phải tính $K(x, x_i)$ với **mọi** vector hỗ trợ, mà số ấy thường tăng tuyến tính theo $n$. Xuống dốc ngẫu nhiên chỉ tốn $O(n)$ mỗi vòng.

### 18.7. Nhóm không giám sát

**"PCA làm gì, và khi nào nó phản tác dụng?"**

> Tìm các hướng trực giao có phương sai lớn nhất; nghiệm là vector riêng của ma trận hiệp phương sai. Sai số tái dựng **đúng bằng** tổng các trị riêng bị bỏ — định lý Eckart–Young, kiểm chứng được tới $10^{-13}$.
>
> *Khi nào phản tác dụng, kèm con số:* PCA **không nhìn nhãn**. Ở Mục 14.4, trục chính thứ nhất giữ **96,1% phương sai** nhưng cho AUC **0,502** — vô dụng hoàn toàn cho phân loại, trong khi LDA đạt 0,9987.
>
> *Kết luận đáng nhớ:* phương sai lớn không đồng nghĩa có ích.

**"PCA khác LDA thế nào?"**

> PCA tối đa hoá phương sai toàn phần và không dùng nhãn; LDA tối đa hoá tỉ số giữa độ tách giữa lớp và độ tản trong lớp, và **dùng nhãn**. LDA cho tối đa $K-1$ chiều, vì $S_B$ chỉ có hạng $K-1$.
>
> *Nối sang chỗ khác:* LDA ở đây chính là bộ phân lớp Gauss dùng chung $\Sigma$ ở Mục 7.4 — hai cách phát biểu của một thuật toán.

**"K-means có bảo đảm gì?"**

> Chỉ bảo đảm **hội tụ tới cực tiểu địa phương**, vì bài toán gốc NP-khó và hàm mục tiêu không lồi.
>
> *Kể một con số:* chạy 200 lần trên dữ liệu 8 cụm — khởi tạo ngẫu nhiên thuần kẹt ở nghiệm tồi **71,5%** số lần; k-means++ hạ xuống **46,5%**, tức cải thiện rõ nhưng không triệt để. Đó là lý do luôn phải chạy nhiều lần.
>
> *Ba giả định ngầm:* cụm lồi và gần hình cầu, kích thước tương đương, mọi chiều cùng thang đo. Đo được: ba cụm tròn → đúng **100%**; hai hình lưỡi liềm → 75%; hai dải dẹt → **53%**, gần như đoán bừa.

**"Chọn $k$ thế nào?"**

> Không thể tối thiểu hoá quán tính, vì nó **luôn giảm** theo $k$ và bằng 0 khi $k=n$. Quy tắc khuỷu tay là nhìn chỗ mức giảm chậm hẳn lại.
>
> *Kể một con số:* trên dữ liệu 8 cụm, mức giảm là **27,7%** khi lên $k=8$ rồi rơi xuống **5,4%** khi lên $k=9$.
>
> *Trung thực:* khuỷu tay không phải định nghĩa toán học và thường không rõ trên dữ liệu thật. Các lựa chọn khác là điểm bóng và thống kê khoảng trống — nhưng thường $k$ do nghiệp vụ quyết định.

**"Phân rã ma trận cần bao nhiêu dữ liệu?"**

> *Có một ngưỡng sắc nét, và đây là con số đáng kể:* với hạng thật $k=4$, ở **10,1 đánh giá mỗi người** thì RMSE là 3,54 — **tệ hơn đoán bừa bằng 0** (2,08). Ở **19,8 đánh giá** thì RMSE rơi xuống 0,30. Giảm hơn 11 lần chỉ vì tăng gấp đôi dữ liệu.
>
> *Quy tắc bỏ túi:* cần cỡ **5 lần số yếu tố ẩn** đánh giá cho mỗi người. Đó là lý do các hệ thật dùng $k$ nhỏ đáng ngạc nhiên (10–200) dù có hàng triệu người dùng.

**"Khởi đầu lạnh vì sao nghiêm trọng?"**

> Vì với người dùng không có đánh giá nào, hàm mất mát chỉ còn số hạng phạt chuẩn $\lambda\|p_u\|^2$, mà cực tiểu của nó là $p_u = 0$.
>
> *Kể một con số:* $\|p_u\|$ của người dùng mới đo được là **0,154** so với 2,084 của người cũ, và RMSE của họ là 2,24 — **tệ hơn cả đoán bừa** 2,08.
>
> *Hệ quả thiết kế:* mọi hệ thực tế phải lai, vì lọc cộng tác **không có đường lùi** khi không có dữ liệu.

### 18.8. Những câu trả lời tự tố cáo

| Câu trả lời | Vì sao nó tố cáo |
|---|---|
| "Bình phương sai lệch vì nó phạt sai số lớn nặng hơn." | Đúng nhưng hời hợt. Không nói được nó là MLE dưới nhiễu Gauss là chưa hiểu gốc. |
| "Cộng tuyến làm mô hình dự báo kém." | Sai. Nó làm **hệ số** loạn; dự báo gần như không đổi. |
| "Gradient descent chậm vì nhiều chiều quá." | Sai. Phụ thuộc $\kappa$, không phụ thuộc $d$. |
| "Kiểm tra gradient thì $\varepsilon$ càng nhỏ càng tốt." | Sai, và sai theo hướng ngược hẳn. $\varepsilon^* \sim u^{1/3}$. |
| "Mô hình đạt 99% độ chính xác nên rất tốt." | Trên dữ liệu 1% lớp dương thì bộ đoán tất cả là âm đạt 99%. |
| "ROC-AUC 0,97 là mô hình rất tốt." | Trên dữ liệu mất cân bằng, cùng mô hình ấy có PR-AUC 0,49. |
| "Ridge và lasso về cơ bản giống nhau." | Lasso cho hệ số bằng đúng 0, ridge không bao giờ. Khác nhau về **chất**. |
| "Phạt chuẩn là một thủ thuật để chống quá khớp." | Nó là **tiên nghiệm**. Ridge = MAP Gauss, kiểm chứng được tới $2\times10^{-8}$. |
| "SVM tốt vì nó tìm biên tối ưu." | "Tối ưu" theo nghĩa nào? Phải nói: lề rộng nhất, và vì sao lề rộng thì tổng quát hoá tốt. |
| "Nhân RBF ánh xạ dữ liệu lên chiều cao hơn rồi phân loại ở đó." | Gần đúng nhưng bỏ mất điểm hay nhất: nó **không bao giờ tính** ánh xạ ấy. |
| "PCA giữ 95% phương sai nên gần như không mất gì." | Đo được: giữ 96,1% phương sai mà AUC còn 0,502. |
| "K-means tìm được cụm tối ưu." | NP-khó; nó chỉ cho cực tiểu địa phương, kẹt 46,5% số lần ngay cả với k-means++. |
| "Naive Bayes giả định các đặc trưng độc lập." | Thiếu hai chữ quan trọng: độc lập **có điều kiện** khi đã biết lớp. |
| "Chọn $k$ cho K-means bằng cách tối thiểu quán tính." | Quán tính luôn giảm theo $k$ và bằng 0 khi $k=n$. |

---

## 19. Tài liệu tham khảo

**Sách nền tảng**

1. T. Hastie, R. Tibshirani, J. Friedman. *The Elements of Statistical Learning.* Springer, 2009. — nguồn chuẩn cho ridge, lasso, LDA/QDA, và công thức co ngót $d_i^2/(d_i^2+\lambda)$ ở Mục 9.3.
2. C. Bishop. *Pattern Recognition and Machine Learning.* Springer, 2006. — nguồn cho cách đọc MAP như phạt chuẩn, và cho phần phân lớp sinh Gauss.
3. K. Murphy. *Machine Learning: A Probabilistic Perspective.* MIT Press, 2012. — trình bày thống nhất theo góc nhìn xác suất, đúng lối Chương 10 dùng.
4. S. Boyd, L. Vandenberghe. *Convex Optimization.* Cambridge University Press, 2004. — nguồn chuẩn cho Chương 11 và 12: tập lồi, hàm lồi, đối ngẫu Lagrange, điều kiện Slater và KKT. Sách và bài giảng đều đọc miễn phí.
5. G. Strang. *Introduction to Linear Algebra.* Wellesley–Cambridge Press. — nền cho Chương 2.
6. T. Hastie, R. Tibshirani, M. Wainwright. *Statistical Learning with Sparsity.* CRC Press, 2015. — chuyên sâu về lasso và vì sao $\ell_1$ cho nghiệm thưa.

**Bài báo gốc của từng thuật toán**

7. A. Hoerl, R. Kennard. *Ridge Regression: Biased Estimation for Nonorthogonal Problems.* Technometrics, 1970. — bài báo giới thiệu ridge, và chính tiêu đề của nó đã nói rõ ridge là ước lượng **có thiên lệch**.
8. R. Tibshirani. *Regression Shrinkage and Selection via the Lasso.* JRSS-B, 1996.
9. H. Zou, T. Hastie. *Regularization and Variable Selection via the Elastic Net.* JRSS-B, 2005.
10. F. Rosenblatt. *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain.* Psychological Review, 1958.
11. A. Novikoff. *On Convergence Proofs for Perceptrons.* Symposium on the Mathematical Theory of Automata, 1962. — nguồn của chặn $(R/\gamma)^2$ dùng ở Mục 6.2.
12. M. Minsky, S. Papert. *Perceptrons.* MIT Press, 1969. — phân tích giới hạn của perceptron trên dữ liệu không tách được.
13. B. Boser, I. Guyon, V. Vapnik. *A Training Algorithm for Optimal Margin Classifiers.* COLT 1992. — nguồn của thủ thuật nhân.
14. C. Cortes, V. Vapnik. *Support-Vector Networks.* Machine Learning, 1995. — nguồn của lề mềm và tham số $C$.
15. V. Vapnik. *The Nature of Statistical Learning Theory.* Springer, 1995.
16. J. Platt. *Sequential Minimal Optimization.* Microsoft Research TR, 1998. — thuật toán mà libsvm dùng để giải bài toán đối ngẫu.
17. R. Fisher. *The Use of Multiple Measurements in Taxonomic Problems.* Annals of Eugenics, 1936. — nguồn của LDA.
18. K. Pearson. *On Lines and Planes of Closest Fit to Systems of Points in Space.* Philosophical Magazine, 1901. — nguồn của PCA.
19. C. Eckart, G. Young. *The Approximation of One Matrix by Another of Lower Rank.* Psychometrika, 1936. — định lý kiểm chứng ở Mục 14.3.
20. S. Lloyd. *Least Squares Quantization in PCM.* IEEE Trans. Information Theory, 1982 (bản thảo 1957). — thuật toán K-means.
21. D. Arthur, S. Vassilvitskii. *k-means++: The Advantages of Careful Seeding.* SODA 2007. — nguồn của cách khởi tạo đo ở Mục 15.2.
22. Y. Koren, R. Bell, C. Volinsky. *Matrix Factorization Techniques for Recommender Systems.* IEEE Computer, 2009. — nguồn chuẩn cho Chương 16.
23. Y. Hu, Y. Koren, C. Volinsky. *Collaborative Filtering for Implicit Feedback Datasets.* ICDM 2008. — ALS cho phản hồi ngầm.

**Tối ưu hoá**

24. B. Polyak. *Some Methods of Speeding up the Convergence of Iteration Methods.* USSR Computational Mathematics, 1964. — heavy ball, tức phương pháp quán tính đo ở Mục 5.3.
25. Y. Nesterov. *A Method of Solving a Convex Programming Problem with Convergence Rate $O(1/k^2)$.* Soviet Mathematics Doklady, 1983.
26. L. Bottou, F. Curtis, J. Nocedal. *Optimization Methods for Large-Scale Machine Learning.* SIAM Review, 2018. — khảo sát tốt về SGD và các biến thể.

**Đánh giá mô hình**

27. T. Fawcett. *An Introduction to ROC Analysis.* Pattern Recognition Letters, 2006.
28. J. Davis, M. Goadrich. *The Relationship Between Precision-Recall and ROC Curves.* ICML 2006. — nguồn chính cho lập luận ở Mục 8.3 về vì sao PR hợp hơn khi mất cân bằng.
29. T. Saito, M. Rehmsmeier. *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.* PLOS ONE, 2015.

**Số học dấu phẩy động**

30. N. Higham. *Accuracy and Stability of Numerical Algorithms.* SIAM, 2002. — nguồn cho phân tích $\varepsilon^* \sim u^{1/3}$ ở Mục 5.5 và cho cảnh báo về $\kappa(X^\top X) = \kappa(X)^2$.

**Tài liệu tiếng Việt**

31. Vũ Hữu Tiệp. *Machine Learning cơ bản.* machinelearningcoban.com — bộ bài giảng tiếng Việt đầy đủ nhất về mảng này, với ảnh động minh hoạ và mã Python cho từng thuật toán. Cấu trúc của giáo trình hiện tại có tham khảo cách tổ chức theo phần của bản ebook ấy.
32. Phạm Đình Khánh. *Deep AI KhanhBlog.* phamdinhkhanh.github.io — thiên về mảng ứng dụng thị giác máy tính và xử lý ngôn ngữ.

> **Về hai nguồn tiếng Việt trên.** Chúng được nêu ở đây như tài liệu đọc thêm. Repo này **không** sao chép nội dung của chúng: mọi phần chữ đều tự viết, và mọi con số đều sinh từ mã trong `code/nentang/`. Riêng ebook *Machine Learning cơ bản* có ghi rõ yêu cầu xin phép tác giả trước khi sao chép hay in ấn, nên ai muốn dùng lại nội dung của tài liệu ấy cần liên hệ trực tiếp.

---

## Phụ lục: chạy lại toàn bộ thí nghiệm

```text
code/nentang/
├── experiments.py            # Hình 2, 3, 4, 6, 7, 8, 9, 10, 11, 12 và mọi số liệu đo được
├── experiments_output.txt    # kết quả in ra của script trên
└── fig_diagrams.py           # Hình 1, 5, 13 (các sơ đồ khái niệm)
```

```bash
pip install numpy scipy matplotlib scikit-learn
python code/nentang/experiments.py      # vài phút
python code/nentang/fig_diagrams.py
```

Hai script đặt hạt giống cố định nên mọi con số trong tài liệu lặp lại được y hệt trên cùng phiên bản thư viện. Môi trường đã dùng: Python 3.13, NumPy 2.3, SciPy 1.16, scikit-learn 1.7, matplotlib 3.10.

**Về loại số liệu.** Khác với giáo trình *Mô hình & Kiến trúc* — nơi có bốn kết quả đối chiếu với con số đã công bố của GPT-2 và Llama-2 — tài liệu này không có con số nào trích từ bài báo. Mọi số đều sinh tại chỗ, nhưng chúng chia làm hai loại rất khác nhau:

**Loại thứ nhất — kiểm chứng một đẳng thức.** Những kết quả này phải khớp tới sai số máy, và nếu chúng không khớp thì có lỗi trong mã:

| Đẳng thức | Sai số đo được |
|---|---|
| Hồi quy logistic $\equiv$ softmax với $K=2$ | $1{,}6\times10^{-15}$ |
| Ba cách giải bình phương tối thiểu cho cùng nghiệm | $1{,}2\times10^{-15}$ |
| PCA qua hiệp phương sai $\equiv$ PCA qua SVD | $7{,}1\times10^{-15}$ |
| Eckart–Young: sai số tái dựng $= (n-1)\sum_{i>k}\lambda_i$ | $\sim 10^{-13}$ |
| Ridge qua SVD $\equiv$ ridge dạng đóng | $2{,}3\times10^{-15}$ |
| Hàm quyết định của LDA là tuyến tính | $4{,}4\times10^{-15}$ |
| SVM: $w = \sum_i \alpha_i y_i x_i$ | $0$ |
| SVM: khe đối ngẫu bằng 0 | $4{,}0\times10^{-8}$ |
| Tích các Gauss một chiều $\equiv$ Gauss nhiều chiều $\Sigma$ chéo | $1{,}2\times10^{-8}$ |
| Ridge dạng đóng $\equiv$ MAP tối ưu bằng BFGS | $2{,}0\times10^{-8}$ |

Bốn dòng cuối không đạt sai số máy vì chúng bị giới hạn bởi dung sai của bộ giải (libsvm, BFGS) hoặc bởi lượng chính quy nhỏ thêm vào để nghịch đảo ma trận — tài liệu in ra con số thật thay vì làm tròn cho đẹp.

**Loại thứ hai — mô phỏng trên dữ liệu sinh ra.** Các bảng về số vòng lặp, độ chính xác, RMSE và tỉ lệ kẹt. Chúng chứng minh *cơ chế* tồn tại và có độ lớn đáng kể, không dùng để suy ra con số cho một bộ dữ liệu cụ thể nào. Chỗ nào kết quả yếu hơn mong đợi — ví dụ ảnh hưởng của $C$ trong SVM nhân tuyến tính ở Mục 13.4 — thì tài liệu nói thẳng như vậy thay vì chọn một thiết lập khác cho số đẹp hơn.
