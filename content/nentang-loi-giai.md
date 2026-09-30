# Lời giải chi tiết — Nền tảng Machine Learning

Mỗi mục ứng với một bài trong Chương 17. Dòng `@meta` được script build đọc để gắn nhãn
chương, dạng bài và độ khó; nó không hiện ra trên trang.

## Bài 1
@meta chuong=4 | dang=Tính tay | kho=Cơ bản

**(a) Lập ma trận.** Thêm cột hằng số 1 cho độ lệch:

$$X = \begin{pmatrix}1&1\\1&2\\1&3\\1&4\end{pmatrix}, \qquad y = \begin{pmatrix}2\\3\\5\\6\end{pmatrix}.$$

$$X^\top X = \begin{pmatrix}4 & 10\\ 10 & 30\end{pmatrix}, \qquad X^\top y = \begin{pmatrix}16\\ 47\end{pmatrix}.$$

Kiểm nhanh: phần tử $(1,1)$ của $X^\top X$ là $n = 4$; phần tử $(1,2)$ là $\sum x_i = 10$; phần tử $(2,2)$ là $\sum x_i^2 = 30$. Phần tử đầu của $X^\top y$ là $\sum y_i = 16$, phần tử sau là $\sum x_i y_i = 2 + 6 + 15 + 24 = 47$.

**(b) Giải.** Định thức $= 4\cdot30 - 10^2 = 20$, nên

$$(X^\top X)^{-1} = \frac{1}{20}\begin{pmatrix}30 & -10\\ -10 & 4\end{pmatrix}.$$

$$\hat w = \frac{1}{20}\begin{pmatrix}30\cdot16 - 10\cdot47\\ -10\cdot16 + 4\cdot47\end{pmatrix} = \frac{1}{20}\begin{pmatrix}10\\ 28\end{pmatrix} = \begin{pmatrix}0{,}5\\ 1{,}4\end{pmatrix}.$$

Vậy $\hat y = 0{,}5 + 1{,}4x$.

**(c) Phần dư.**

| $x$ | $y$ | $\hat y$ | dư |
|---|---|---|---|
| 1 | 2 | 1,9 | $+0{,}1$ |
| 2 | 3 | 3,3 | $-0{,}3$ |
| 3 | 5 | 4,7 | $+0{,}3$ |
| 4 | 6 | 6,1 | $-0{,}1$ |

Tổng phần dư bằng **0** (giá trị máy tính ra là $4{,}4\times10^{-16}$), và $RSS = 0{,}20$.

**Vì sao tổng bằng 0 không phải trùng hợp.** Phương trình chuẩn tắc là $X^\top(y - X\hat w) = 0$, tức $X^\top r = 0$. Hàng đầu của $X^\top$ là cột hằng số, gồm toàn số 1, nên hàng ấy cho đúng $\sum_i r_i = 0$.

Điều này đúng **bất cứ khi nào mô hình có hệ số chặn**, và nó còn cho một cách đọc hình học: vector phần dư trực giao với **mọi** cột của $X$, tức trực giao với toàn bộ không gian cột. Bình phương tối thiểu chính là phép chiếu vuông góc của $y$ lên không gian cột của $X$.

**(d) Ridge với $\lambda = 1$.**

$$X^\top X + I = \begin{pmatrix}5 & 10\\ 10 & 31\end{pmatrix}, \quad \det = 155 - 100 = 55,$$

$$\hat w_{\text{ridge}} = \frac{1}{55}\begin{pmatrix}31\cdot16 - 10\cdot47\\ -10\cdot16 + 5\cdot47\end{pmatrix} = \frac{1}{55}\begin{pmatrix}26\\ 75\end{pmatrix} = \begin{pmatrix}0{,}4727\\ 1{,}3636\end{pmatrix}.$$

Cả hai hệ số **co về phía 0**: $0{,}5 \to 0{,}4727$ và $1{,}4 \to 1{,}3636$. Đúng như tên gọi "co ngót".

> **Một lưu ý thực hành.** Ở đây ta phạt cả $w_0$, và điều đó thường **không** mong muốn: phạt hệ số chặn làm mô hình phụ thuộc vào việc ta có dịch chuyển $y$ hay không. Các cài đặt thật (kể cả `sklearn.linear_model.Ridge`) đều **loại hệ số chặn khỏi phần bị phạt**.

## Bài 2
@meta chuong=10 | dang=Suy luận | kho=Trung bình

**(a) Âm log hậu nghiệm.** Theo quy tắc Bayes, $p(w \mid \mathcal{D}) \propto p(\mathcal{D}\mid w)\,p(w)$.

Phần hợp lý, với nhiễu Gauss độc lập:

$$-\log p(\mathcal{D}\mid w) = \sum_{i=1}^{n}\frac{(y_i - w^\top x_i)^2}{2\sigma^2} + \text{hằng số} = \frac{\|y - Xw\|^2}{2\sigma^2} + \text{hằng số}.$$

Phần tiên nghiệm, với $w \sim \mathcal{N}(0, \tau^2 I)$:

$$-\log p(w) = \frac{\|w\|^2}{2\tau^2} + \text{hằng số}.$$

Cộng lại và bỏ mọi hằng số không phụ thuộc $w$:

$$-\log p(w\mid\mathcal{D}) \;\dot=\; \frac{\|y - Xw\|^2}{2\sigma^2} + \frac{\|w\|^2}{2\tau^2}.$$

**(b) Suy ra ridge.** Nhân cả hai vế với $2\sigma^2$ — phép này không đổi vị trí cực tiểu:

$$\hat w_{\text{MAP}} = \arg\min_w\; \|y - Xw\|^2 + \frac{\sigma^2}{\tau^2}\|w\|^2.$$

So với định nghĩa ridge ở Mục 9.2 thì đây đúng là cùng một bài toán, với

$$\boxed{\lambda = \frac{\sigma^2}{\tau^2}}$$

Kiểm chứng bằng số ở Mục 10.4: với $\sigma = 1$ và $\tau = 0{,}7$ thì $\lambda = 2{,}040816$, và nghiệm ridge dạng đóng lệch so với MAP tối ưu bằng BFGS đúng $1{,}97\times10^{-8}$ — tức bằng nhau trong phạm vi dung sai của bộ tối ưu.

**(c) Tiên nghiệm chặt hơn.** $\tau$ nhỏ đi $\Rightarrow$ $\lambda = \sigma^2/\tau^2$ **lớn lên**, tức phạt chuẩn mạnh hơn.

Chiều này hợp lý vì $\tau$ nhỏ nghĩa là ta tin chắc rằng các hệ số phải gần 0. Niềm tin càng chắc thì dữ liệu càng phải đưa ra bằng chứng mạnh mới kéo được hệ số ra xa 0 — và "phải đưa bằng chứng mạnh hơn" chính là ý nghĩa của phạt chuẩn lớn.

Tỉ số $\sigma^2/\tau^2$ đọc được như một **cuộc đấu giữa hai nguồn bất định**: $\sigma^2$ là mức nhiễu của dữ liệu, $\tau^2$ là mức mở của niềm tin trước. Dữ liệu nhiễu nhiều (σ lớn) thì nên nghe tiên nghiệm hơn; tiên nghiệm mơ hồ (τ lớn) thì nên nghe dữ liệu hơn.

**(d) Tiên nghiệm Laplace.** Với $p(w_j) \propto \exp(-|w_j|/b)$:

$$-\log p(w) = \frac{1}{b}\sum_j |w_j| + \text{hằng số} = \frac{\|w\|_1}{b} + \text{hằng số},$$

cho ra **lasso**.

Niềm tin mà nó phát biểu: **phần lớn các hệ số nên đúng bằng 0**. Lý do nằm ở hình dạng phân phối — Laplace có **đỉnh nhọn** tại 0 (mật độ không khả vi ở đó), nên nó dồn nhiều khối lượng xác suất sát 0 hơn Gauss, đồng thời có đuôi dày hơn nên vẫn cho phép một vài hệ số lớn. Đúng là mô tả của một vector thưa.

## Bài 3
@meta chuong=5 | dang=Tính tay | kho=Cơ bản

**(a) Ước lượng ở $\kappa = 40\,000$.** Số điều kiện tăng 4 lần so với $10\,000$.

| | Bậc | Hệ số nhân | Ước lượng |
|---|---|---|---|
| Gradient descent | $O(\kappa)$ | $\times 4$ | $92\,104 \times 4 \approx \mathbf{368\,000}$ vòng |
| Thêm quán tính | $O(\sqrt\kappa)$ | $\times 2$ | $1\,297 \times 2 \approx \mathbf{2\,600}$ vòng |

Khoảng cách giữa hai phương pháp giãn từ 71 lần lên khoảng 142 lần. Đó là điều đáng nhớ: **lợi ích của quán tính tăng lên khi bài toán khó đi**, chứ không phải là một hệ số cố định.

**(b) "Bài toán tôi có 1 triệu chiều nên GD sẽ rất chậm."**

**Sai.** Tốc độ hội tụ của gradient descent phụ thuộc **số điều kiện** $\kappa$ chứ không phụ thuộc số chiều $d$. Một bài toán 1 triệu chiều nhưng có $\kappa = 5$ hội tụ nhanh hơn nhiều một bài toán 2 chiều có $\kappa = 10\,000$.

Số chiều ảnh hưởng tới **chi phí của mỗi vòng lặp** ($O(nd)$ cho một lần tính gradient), chứ không ảnh hưởng tới **số vòng lặp cần thiết**. Hai thứ ấy phải tách bạch, và nhầm lẫn giữa chúng là một trong những câu tự tố cáo ở Mục 18.8.

**(c) Một bước tiền xử lý rẻ tiền: chuẩn hoá đặc trưng.**

Trừ trung bình và chia cho độ lệch chuẩn từng cột. Cơ chế: nếu một đặc trưng có biên độ $10^6$ còn một đặc trưng khác có biên độ $10^{-3}$, thì các trị riêng của $X^\top X$ chênh nhau cỡ $10^{18}$, tức $\kappa \approx 10^{18}$. Đưa mọi đặc trưng về cùng phương sai làm các trị riêng về cùng cỡ độ lớn, và $\kappa$ giảm theo.

Đây là lý do một bước tưởng như tầm thường lại quyết định việc huấn luyện có chạy hay không. Và nó cũng giải thích vì sao chuẩn hoá lại quan trọng với mạng nơ-ron — xem [Chương 6 của *Mô hình & Kiến trúc*](models-ch06.html).

## Bài 4
@meta chuong=5 | dang=Suy luận | kho=Trung bình

**(a) Sai phân trung tâm.** Tổng sai số:

$$E(\varepsilon) = C_1\varepsilon^2 + \frac{C_2 u}{\varepsilon}.$$

Lấy đạo hàm và cho bằng 0:

$$E'(\varepsilon) = 2C_1\varepsilon - \frac{C_2u}{\varepsilon^2} = 0 \;\Longrightarrow\; \varepsilon^3 = \frac{C_2 u}{2C_1} \;\Longrightarrow\; \boxed{\varepsilon^* = \left(\frac{C_2u}{2C_1}\right)^{1/3} \propto u^{1/3}}$$

Giá trị sai số tại đó cỡ $u^{2/3} \approx 3{,}7\times10^{-11}$ — và đây là **độ chính xác tốt nhất có thể đạt được**, không thể làm hơn bằng cách chọn $\varepsilon$ khéo hơn.

**(b) Sai phân tiến.** Sai số cắt cụt chỉ bậc nhất:

$$E(\varepsilon) = C_1\varepsilon + \frac{C_2u}{\varepsilon}, \qquad E'(\varepsilon) = C_1 - \frac{C_2u}{\varepsilon^2} = 0 \;\Longrightarrow\; \boxed{\varepsilon^* \propto u^{1/2}}$$

Sai số tốt nhất đạt được cỡ $u^{1/2} \approx 1{,}5\times10^{-8}$.

**(c) Đối chiếu với số đo.**

| | Lý thuyết | Đo được |
|---|---|---|
| Trung tâm | $u^{1/3} = 6{,}06\times10^{-6}$ | $5{,}62\times10^{-6}$ |
| Tiến | $u^{1/2} = 1{,}49\times10^{-8}$ | $1{,}78\times10^{-8}$ |

Cả hai khớp trong phạm vi độ phân giải của lưới quét (53 điểm trên 13 bậc độ lớn, nên bước lưới là hệ số $1{,}8$ — cả hai độ lệch đều nhỏ hơn một bước lưới).

Và sai số tốt nhất đạt được cũng khớp: trung tâm cho $1{,}72\times10^{-11}$ so với dự đoán $\sim u^{2/3} = 3{,}7\times10^{-11}$.

**(d) Vì sao vẫn ưa sai phân trung tâm.** Nó tốn hai lần gọi hàm thay vì một (thực ra là hai thay vì một, vì $f(x)$ thường đã có sẵn), nhưng đổi lại độ chính xác tốt nhất đi từ $10^{-8}$ xuống $10^{-11}$ — **hơn ba bậc độ lớn**.

Khoảng cách ấy quan trọng vì mục đích của việc kiểm tra là **phân biệt lỗi thật với nhiễu số học**. Với sai phân tiến, một sai số tương đối $10^{-7}$ có thể là lỗi cài đặt mà cũng có thể chỉ là nhiễu — không kết luận được. Với sai phân trung tâm thì $10^{-7}$ chắc chắn là lỗi, vì nhiễu chỉ ở mức $10^{-11}$.

## Bài 5
@meta chuong=6 | dang=Tính tay | kho=Cơ bản

**(a) Chặn Novikoff.**

$$\left(\frac{R}{\gamma}\right)^2 = \left(\frac{2}{0{,}1}\right)^2 = 20^2 = \boxed{400 \text{ lần sai}}$$

**(b) Thêm 10 000 điểm.** Chặn **không đổi**: vẫn là 400.

Đây là điều đáng nói nhất về định lý Novikoff. Công thức $(R/\gamma)^2$ **không chứa $n$ và không chứa $d$**. Nghĩa là số lần perceptron mắc lỗi bị chặn bởi **hình học của bài toán** — độ lớn của dữ liệu và độ rộng của khe hở giữa hai lớp — chứ không bởi lượng dữ liệu hay số chiều.

Một hệ quả cụ thể: perceptron vẫn chạy được trên dữ liệu có số chiều rất cao (ví dụ văn bản với hàng trăm nghìn đặc trưng), miễn là dữ liệu tách được với lề không quá nhỏ.

**(c) Nhân mọi $x_i$ với 10.**

- $R \to 10R = 20$.
- $\gamma \to 10\gamma = 1$, vì lề là $\min_i y_i(w^{*\top}x_i)$ với $\|w^*\| = 1$, nên nó co giãn cùng $x$.
- Chặn: $(20/1)^2 = 400$ — **không đổi**.

Chặn **bất biến với phép co giãn**, đúng như phải thế: nhân toàn bộ dữ liệu với một hằng số không làm bài toán dễ hay khó hơn, nó chỉ đổi đơn vị.

**(d) Vì sao không có bảo đảm khi dữ liệu không tách được.**

Chứng minh của Novikoff dựa vào sự tồn tại của một $w^*$ tách đúng **mọi** điểm với lề $\gamma > 0$. Không có $w^*$ như thế thì lập luận sụp ngay từ dòng đầu.

Nhìn từ hàm mất mát (Hình 5) thì rõ hơn: mất mát perceptron $\max(0, -m)$ **phẳng hoàn toàn khi $m > 0$**. Nghĩa là thuật toán không có khái niệm "đúng chắc chắn hơn" — vừa vượt qua 0 là nó buông. Khi có các điểm mâu thuẫn nhau, mỗi lần sửa cho điểm này lại làm hỏng điểm kia, và vòng lặp không bao giờ đóng.

Đo được ở Mục 6.2: trên dữ liệu không tách được, perceptron mắc **17 977 lần sai qua 2 000 lượt quét** và không hội tụ. Đây đúng là vấn đề mà hinge và logistic sinh ra để chữa: cả hai đều **không phẳng ngay tại $m = 0$**, nên vẫn còn động lực đẩy điểm ra xa biên.

## Bài 6
@meta chuong=8 | dang=Tính tay | kho=Cơ bản

**(a) Bốn chỉ số.** Từ bảng: $TP = 102$, $FN = 94$, $FP = 105$, $TN = 19\,699$; tổng $n = 20\,000$.

$$\text{độ chính xác} = \frac{102 + 19\,699}{20\,000} = \frac{19\,801}{20\,000} = \mathbf{0{,}9900}$$

$$\text{precision} = \frac{102}{102 + 105} = \frac{102}{207} = \mathbf{0{,}4928}$$

$$\text{recall} = \frac{102}{102 + 94} = \frac{102}{196} = \mathbf{0{,}5204}$$

$$F_1 = \frac{2 \times 0{,}4928 \times 0{,}5204}{0{,}4928 + 0{,}5204} = \mathbf{0{,}5062}$$

**(b) Bộ "luôn đoán bình thường".** Nó đúng ở toàn bộ lớp âm và sai ở toàn bộ lớp dương:

$$\text{độ chính xác} = \frac{19\,699 + 105}{20\,000} = \frac{19\,804}{20\,000} = \mathbf{0{,}9902}$$

**(c) Kết luận.** Bộ phân loại **vô dụng đạt 0,9902**, cao hơn bộ phân loại dùng được (0,9900). Nếu chọn mô hình theo độ chính xác thì ta chọn đúng cái không bắt được một ca gian lận nào.

Lý do là độ chính xác bị chi phối hoàn toàn bởi lớp đa số: lớp âm chiếm 99,02% nên đoán bừa theo nó đã được 99,02%. Mọi cải thiện thật trên lớp dương chỉ có thể đóng góp nhiều nhất 0,98 điểm phần trăm — tức nằm trong phần nhiễu của chỉ số.

**Độ chính xác chỉ có ý nghĩa khi các lớp cân bằng.** Ở đây phải dùng precision, recall, $F_1$ hoặc PR-AUC.

**(d) Bỏ sót đắt gấp 20 lần báo động nhầm.**

**Dịch ngưỡng xuống.** Ngưỡng thấp hơn làm mô hình báo động nhiều hơn: recall tăng, precision giảm. Đó là chiều đúng khi FN đắt hơn FP.

**Chỉ số nên dùng:** không phải $F_1$, vì $F_1$ coi precision và recall **quan trọng như nhau** — giả định ấy sai ở đây. Hai lựa chọn đúng:

1. **$F_\beta$ với $\beta = \sqrt{20} \approx 4{,}5$**, vì $F_\beta$ đánh trọng số recall gấp $\beta^2$ lần precision:
   $$F_\beta = \frac{(1+\beta^2)\,PR}{\beta^2 P + R}.$$
2. **Tốt hơn nữa: tối thiểu hoá chi phí kỳ vọng trực tiếp.** Đặt $\text{chi phí} = 20 \cdot FN + 1 \cdot FP$ rồi quét ngưỡng tìm chỗ nhỏ nhất. Cách này không cần chọn chỉ số thay thế nào cả — nó tối ưu đúng thứ mà bộ phận rủi ro quan tâm.

Cách thứ hai luôn đáng ưu tiên khi biết được chi phí thật. Chỉ số như $F_1$ là thứ dùng khi ta **không** biết chi phí.

## Bài 7
@meta chuong=13 | dang=Tính tay | kho=Trung bình

**(a) Lập luận đối xứng và giải bài toán gốc.**

Hai điểm $x_1 = (1,1)$ và $x_2 = (-1,-1)$ đối xứng qua gốc toạ độ, và nhãn của chúng đối nhau. Nên siêu phẳng tối ưu phải đi qua trung điểm của chúng, tức qua gốc: $b = 0$.

Cũng theo đối xứng, $w$ phải cùng phương với $x_1 - x_2 = (2,2)$, tức $w = (a, a)$.

Ràng buộc $y_i(w^\top x_i + b) \ge 1$:

- Với $x_1$: $a + a = 2a \ge 1$.
- Với $x_2$: $-(-a - a) = 2a \ge 1$.

Cả hai cho $a \ge 1/2$. Ta tối thiểu $\tfrac12\|w\|^2 = a^2$, nên lấy $a$ nhỏ nhất:

$$\boxed{w = (0{,}5;\; 0{,}5), \qquad b = 0}$$

**(b) Lề.** $\|w\| = \sqrt{0{,}5} = 0{,}7071$, nên

$$\text{lề} = \frac{2}{\|w\|} = \frac{2}{0{,}7071} = \mathbf{2{,}8284}.$$

Khoảng cách giữa hai điểm: $\|x_1 - x_2\| = \|(2,2)\| = 2\sqrt2 = \mathbf{2{,}8284}$.

**Chúng bằng nhau đúng.** Điều này hợp lý: chỉ có hai điểm, cả hai đều là vector hỗ trợ, và siêu phẳng nằm chính giữa chúng. Lề — khoảng cách giữa hai mặt $w^\top x = \pm1$ — khi ấy trải trọn khoảng cách giữa hai điểm.

**(c) Nhân tử Lagrange.** Từ $w = \sum_i \alpha_i y_i x_i$:

$$(0{,}5;\,0{,}5) = \alpha_1 \cdot (+1) \cdot (1,1) + \alpha_2 \cdot (-1) \cdot (-1,-1) = (\alpha_1 + \alpha_2)(1,1),$$

nên $\alpha_1 + \alpha_2 = 0{,}5$. Ràng buộc $\sum_i\alpha_i y_i = 0$ cho $\alpha_1 - \alpha_2 = 0$. Giải hệ:

$$\boxed{\alpha_1 = \alpha_2 = 0{,}25}$$

**(d) Đối ngẫu mạnh.**

Bài toán gốc: $\tfrac12\|w\|^2 = \tfrac12 (0{,}5^2 + 0{,}5^2) = \tfrac12 \cdot 0{,}5 = \mathbf{0{,}25}$.

Bài toán đối ngẫu: với $x_1^\top x_1 = 2$, $x_2^\top x_2 = 2$, $x_1^\top x_2 = -2$ và $y_1y_2 = -1$:

$$\sum_i\alpha_i - \tfrac12\sum_{i,j}\alpha_i\alpha_j y_iy_j x_i^\top x_j = 0{,}5 - \tfrac12\big[0{,}0625\cdot2 + 0{,}0625\cdot2 + 2\cdot0{,}0625\cdot(-1)\cdot(-2)\big]$$
$$= 0{,}5 - \tfrac12 \cdot 0{,}5 = \mathbf{0{,}25}.$$

**Khe đối ngẫu bằng 0 chính xác.** Đây là đối ngẫu mạnh, đúng như điều kiện Slater đảm bảo.

**(e) Thêm điểm $x_3 = (5,5)$ nhãn $+1$.**

**Nghiệm không đổi.** Kiểm ràng buộc của điểm mới với nghiệm cũ:

$$y_3(w^\top x_3 + b) = +1 \cdot (0{,}5\cdot5 + 0{,}5\cdot5) = 5 \;\ge\; 1. \;\checkmark$$

Ràng buộc thoả một cách **lỏng** (5 chứ không phải 1). Theo điều kiện bù trừ $\alpha_i f_i(x^*) = 0$ ở Mục 12.4, ràng buộc lỏng buộc $\alpha_3 = 0$, nên $x_3$ **không đóng góp gì** vào $w = \sum\alpha_i y_i x_i$.

Đây chính là cơ chế tạo ra tính thưa của SVM, và là lý do ở Mục 13.3 chỉ **3 trên 120 điểm** quyết định toàn bộ nghiệm. Có thể thêm bao nhiêu điểm cũng được, miễn chúng nằm ngoài lề, mà không đụng tới mô hình.

## Bài 8
@meta chuong=14 | dang=Tính tay | kho=Cơ bản

Trị riêng $\lambda = (10;\, 5;\, 3;\, 1{,}5;\, 0{,}5)$, tổng $= 20$.

**(a) Số thành phần cho ≥ 85% phương sai.**

| $k$ | Tổng giữ lại | % phương sai |
|---|---|---|
| 1 | 10 | 50,0% |
| 2 | 15 | 75,0% |
| 3 | 18 | **90,0%** |
| 4 | 19,5 | 97,5% |

Cần $\boxed{k = 3}$.

**(b) Sai số tái dựng với $k = 2$, $n = 101$.** Theo định lý Eckart–Young (Mục 14.3):

$$\|X_c - X_k\|_F^2 = (n-1)\sum_{i>k}\lambda_i = 100 \times (3 + 1{,}5 + 0{,}5) = 100 \times 5 = \boxed{500}.$$

**(c) Bác bỏ "giữ 90% phương sai nên chỉ kém đi một chút".**

Phát biểu ấy lẫn lộn hai thứ khác hẳn nhau: **phương sai được giữ lại** và **thông tin phân loại được giữ lại**. PCA tối đa hoá cái thứ nhất và **hoàn toàn không nhìn** cái thứ hai, vì nó không dùng nhãn.

Phản ví dụ đo được ở Mục 14.4: trục chính thứ nhất giữ **96,1% phương sai** nhưng cho **AUC = 0,502** — tức ngang đoán bừa. Cùng dữ liệu ấy, hướng LDA cho AUC 0,9987.

Cơ chế: dữ liệu có một hướng nhiễu biên độ rất lớn (không mang thông tin) và một hướng tín hiệu biên độ nhỏ. PCA chọn hướng nhiễu vì nó có phương sai lớn hơn, và vứt đúng hướng cần giữ.

**Kết luận đáng nhớ: phương sai lớn không đồng nghĩa có ích.** Khi mục tiêu cuối là phân loại và có sẵn nhãn, nên thử LDA, hoặc ít nhất kiểm tra hiệu năng sau khi giảm chiều chứ đừng tin vào tỉ lệ phương sai.

**(d) Đổi đơn vị từ mét sang milimét.**

Đặc trưng ấy được nhân với 1 000, nên **phương sai của nó nhân với $10^6$**. Ma trận hiệp phương sai đổi theo, và trị riêng nào gắn với hướng ấy cũng phình lên cỡ $10^6$ lần.

Hệ quả: đặc trưng đó sẽ **chiếm gần trọn** trục chính thứ nhất, bất kể nó có quan trọng hay không. PCA khi ấy chỉ còn phản ánh việc ta chọn đơn vị đo nào.

**Rút ra: PCA không bất biến với phép co giãn từng đặc trưng, nên hầu như luôn phải chuẩn hoá trước.** Tương đương với việc làm PCA trên ma trận **tương quan** thay vì ma trận hiệp phương sai. Ngoại lệ duy nhất là khi mọi đặc trưng vốn đã cùng đơn vị và cùng thang đo có ý nghĩa — ví dụ cường độ điểm ảnh.

> Chú ý rằng cây quyết định **không** có vấn đề này, vì chúng bất biến với mọi phép biến đổi đơn điệu từng đặc trưng — xem [Mục 3.5 của *Mô hình & Kiến trúc*](models-ch03.html).

## Bài 9
@meta chuong=15 | dang=Chẩn đoán | kho=Trung bình

**(a) Năm lần chạy cho năm kết quả khác hẳn nhau.**

**Vấn đề:** kẹt ở các cực tiểu địa phương khác nhau. Đây là hành vi **bình thường** của K-means, không phải lỗi — bài toán NP-khó và hàm mục tiêu không lồi. Đo được ở Mục 15.2: khởi tạo ngẫu nhiên thuần kẹt ở nghiệm tồi 71,5% số lần.

**Hai việc nên làm:**
1. Chạy nhiều lần (10–50) với hạt giống khác nhau rồi **giữ kết quả có quán tính nhỏ nhất**. Đây đúng là điều `n_init` trong scikit-learn làm.
2. Dùng **k-means++** thay cho khởi tạo ngẫu nhiên. Nó hạ tỉ lệ kẹt xuống 46,5% — cải thiện rõ, dù không triệt để.

**(b) Chia đôi cụm dài dẹt, gộp hai cụm tròn nhỏ gần nhau.**

**Vấn đề:** vi phạm giả định về hình dạng. K-means gán điểm theo khoảng cách Euclid tới tâm, nên biên giữa hai cụm luôn là siêu phẳng, và nó ưu tiên các cụm **tròn, kích thước tương đương**. Một cụm dài dẹt có các điểm ở hai đầu xa tâm hơn là xa tâm của cụm bên cạnh — nên bị xé đôi.

Đo được ở Mục 15.3: trên hai dải dẹt, độ chính xác chỉ **0,5317**, gần như đoán bừa.

**Hai việc nên làm:**
1. Đổi sang **mô hình hỗn hợp Gauss**, vốn cho phép cụm hình elip với hướng và độ dẹt riêng.
2. Hoặc **DBSCAN** / **phân cụm phổ**, cả hai đều xử lý được cụm hình dạng tuỳ ý. DBSCAN còn tự xác định số cụm.

Nếu bắt buộc phải dùng K-means, có thể biến đổi không gian trước (ví dụ chuẩn hoá theo từng trục hoặc làm trắng dữ liệu) để cụm tròn lại — nhưng đó là chữa triệu chứng.

**(c) Quán tính giảm đều, không thấy khuỷu tay.**

**Vấn đề:** rất có thể dữ liệu **không có cấu trúc cụm rõ ràng**. Với dữ liệu phân bố đều hoặc một đám mây Gauss duy nhất, quán tính giảm trơn theo $k$ và không có chỗ gãy — vì không có $k$ nào "đúng" cả.

Cũng có thể cụm tồn tại nhưng chồng lấn nhiều, hoặc số cụm thật lớn hơn khoảng $k$ đang quét.

**Hai việc nên làm:**
1. Dùng chỉ số khác: **điểm bóng** (đo mức một điểm hợp với cụm của nó hơn cụm gần nhất) hoặc **thống kê khoảng trống** (so quán tính với quán tính trên dữ liệu ngẫu nhiên cùng phạm vi). Thống kê khoảng trống đặc biệt hợp ở đây vì nó phát hiện được trường hợp "không có cụm nào".
2. **Trực quan hoá** bằng PCA hoặc UMAP trước khi phân cụm. Nếu mắt không thấy cụm thì thuật toán cũng khó tìm ra thứ có ý nghĩa.

Và một khả năng cần cân nhắc nghiêm túc: **phân cụm có thể không phải công cụ đúng** cho bài toán này.

**(d) Một cụm chứa 98% số điểm.**

**Vấn đề:** hầu như chắc chắn là do **thang đo hoặc điểm ngoại lai**. Nếu một đặc trưng có biên độ lớn hơn hẳn các đặc trưng khác, khoảng cách Euclid gần như chỉ phản ánh đặc trưng ấy. Còn vài điểm ngoại lai ở rất xa sẽ hút hẳn các tâm cụm về phía chúng, để lại một tâm duy nhất gánh toàn bộ phần còn lại.

**Hai việc nên làm:**
1. **Chuẩn hoá đặc trưng** về cùng thang đo trước khi phân cụm — đây là giả định ngầm thứ ba ở Mục 15.3, và là nguyên nhân thường gặp nhất.
2. **Phát hiện và xử lý điểm ngoại lai** trước, hoặc đổi sang **K-medoids**, vốn dùng điểm dữ liệu thật làm tâm nên ít bị điểm ngoại lai kéo đi.

## Bài 10
@meta chuong=16 | dang=Thiết kế | kho=Nâng cao

**(a) Số tham số.**

$$(n_u + n_i)\,k = (10^6 + 10^5) \times 50 = 1{,}1\times10^6 \times 50 = \boxed{55\,000\,000}$$

Số ô của ma trận đầy đủ: $10^6 \times 10^5 = 10^{11}$.

$$\frac{5{,}5\times10^7}{10^{11}} = \mathbf{0{,}055\%}$$

Mô hình nén $10^{11}$ ô xuống 55 triệu số — tỉ lệ nén hơn 1 800 lần. **Toàn bộ sức mạnh của phương pháp nằm ở giả thiết hạng thấp**, và con số 0,055% chính là mức độ của giả thiết ấy.

**(b) Lượng đánh giá cần.**

Theo quy tắc ở Mục 16.3, cần khoảng $5k$ đánh giá mỗi người:

$$5 \times 50 = \boxed{250 \text{ đánh giá mỗi người}}$$

Tổng cộng: $10^6 \times 250 = \mathbf{250}$ **triệu đánh giá**, bằng

$$\frac{2{,}5\times10^8}{10^{11}} = \mathbf{0{,}25\%} \text{ số ô}.$$

**(c) Thực tế chỉ có 30 đánh giá mỗi người.**

$30$ so với $250$ là thiếu hơn **8 lần**. Theo bảng đo ở Mục 16.3, ở mức $\approx 10$ đánh giá với $k=4$ thì RMSE **tệ hơn đoán bừa** — và ở đây tỉ lệ $30/50 = 0{,}6$ đánh giá trên mỗi yếu tố ẩn còn thấp hơn tỉ lệ $10/4 = 2{,}5$ của trường hợp hỏng ấy.

**Dự đoán:** mô hình sẽ khớp rất tốt trên các ô đã thấy nhưng **dự đoán kém hơn cả việc đoán điểm trung bình** trên các ô chưa thấy. Đây là quá khớp ở dạng nghiêm trọng nhất, và nó sẽ **không** lộ ra nếu chỉ nhìn sai số huấn luyện.

**Hai cách xử lý:**

1. **Giảm $k$ mạnh.** Với 30 đánh giá mỗi người, quy tắc $5k$ cho $k \approx 6$. Bảng chọn hạng ở Mục 16.4 cho thấy đúng hiện tượng này: chọn $k$ quá lớn làm RMSE trên ô chưa thấy tăng — ở $k=20$ nó tệ hơn 4,3 lần so với $k$ đúng.
2. **Tăng phạt chuẩn $\lambda$**, và chọn nó bằng kiểm định chéo trên các ô bị giữ lại chứ không phải trên các ô đã thấy. Phạt chuẩn mạnh làm mô hình hạng $50$ hành xử gần như mô hình hạng thấp hơn.

Một cách thứ ba đáng cân nhắc: **bổ sung đặc trưng phụ** (thể loại, nhân khẩu học) vào mô hình, tức chuyển sang mô hình lai. Đặc trưng phụ không cần ước lượng từ dữ liệu đánh giá nên không tiêu tốn "ngân sách" 30 đánh giá ấy.

**(d) Người dùng mới phải nhận gợi ý ngay từ phiên đầu.**

Phân rã ma trận **không làm được việc này**, và không phải vì cài đặt kém. Như đo ở Mục 16.5, người dùng không có đánh giá nào thì hàm mất mát chỉ còn $\lambda\|p_u\|^2$, nên $p_u \to 0$ và mô hình dự đoán 0 cho mọi mục — RMSE 2,24 so với 2,08 của việc đoán bừa, tức **tệ hơn đoán bừa**.

**Thiết kế đường dự phòng, ba tầng:**

| Tầng | Điều kiện | Cách gợi ý |
|---|---|---|
| 0 | 0 tương tác | **Mục phổ biến**, lọc theo ngữ cảnh có sẵn (quốc gia, thiết bị, thời điểm). Kèm một ít ngẫu nhiên hoá để thu thập tín hiệu. |
| 1 | 1–30 tương tác | **Dựa trên nội dung**: khớp đặc trưng mục với hồ sơ suy ra từ vài tương tác đầu. |
| 2 | > 30 tương tác | **Phân rã ma trận**, hoặc mô hình lai kết hợp cả hai nguồn. |

**Khi nào chuyển tầng.** Đừng chọn ngưỡng bằng cảm tính. Cách đúng: với người dùng **đã có nhiều dữ liệu**, cắt bớt lịch sử của họ xuống $m$ tương tác rồi đo hiệu năng của cả hai phương pháp theo $m$. Ngưỡng chuyển là điểm giao — chỗ phân rã ma trận bắt đầu vượt phương pháp dựa trên nội dung. Quy tắc $5k$ cho ta chỗ **bắt đầu tìm**, còn đường cong đo được mới cho ngưỡng thật.

**Hai điều nữa phải thiết kế ngay từ đầu:**

1. **Hỏi sở thích lúc đăng ký.** Cho người dùng mới chọn vài thể loại là cách rẻ nhất để nhảy thẳng từ tầng 0 lên tầng 1.
2. **Ngẫu nhiên hoá một tỉ lệ nhỏ số gợi ý.** Nếu hệ chỉ gợi ý thứ nó đã biết thì nó chỉ nhận phản hồi về thứ đó, và càng ngày càng hẹp. [Mục 12.5 của *MLOps*](mlops-ch12.html) đo được rằng không ngẫu nhiên hoá thì hệ vĩnh viễn chỉ nhìn thấy **9,5%** danh mục. Đây là quyết định kiến trúc, không phải tính năng thêm vào sau.
