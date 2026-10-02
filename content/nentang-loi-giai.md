# Lời giải chi tiết — Nền tảng học máy

Mỗi mục ứng với một bài trong Chương 17. Dòng `@meta` được script build đọc để gắn nhãn
chương, dạng bài và độ khó; nó không hiện ra trên trang.

## Bài 1
@meta chuong=4 | dang=Tính tay | kho=Cơ bản

**(a) Lập ma trận.** Thêm cột hằng số 1 ứng với hệ số chặn:

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

Tổng phần dư bằng 0 (máy tính cho $4{,}4\times10^{-16}$, tức sai số làm tròn), và tổng bình phương phần dư là $RSS = 0{,}20$.

**Vì sao tổng bằng 0.** Phương trình chuẩn là $X^\top(y - X\hat w) = 0$, tức $X^\top r = 0$. Hàng đầu của $X^\top$ là cột hằng số, gồm toàn số 1, nên hàng ấy cho đúng $\sum_i r_i = 0$.

Điều này đúng với mọi mô hình có hệ số chặn. Nó còn có cách hiểu hình học: vector phần dư trực giao với mọi cột của $X$, tức trực giao với toàn bộ không gian cột. Bình phương tối thiểu chính là phép chiếu vuông góc của $y$ lên không gian cột của $X$.

**(d) Ridge với $\lambda = 1$.**

$$X^\top X + I = \begin{pmatrix}5 & 10\\ 10 & 31\end{pmatrix}, \quad \det = 155 - 100 = 55,$$

$$\hat w_{\text{ridge}} = \frac{1}{55}\begin{pmatrix}31\cdot16 - 10\cdot47\\ -10\cdot16 + 5\cdot47\end{pmatrix} = \frac{1}{55}\begin{pmatrix}26\\ 75\end{pmatrix} = \begin{pmatrix}0{,}4727\\ 1{,}3636\end{pmatrix}.$$

Cả hai hệ số co về phía 0: $0{,}5 \to 0{,}4727$ và $1{,}4 \to 1{,}3636$.

> **Lưu ý.** Ở đây hệ số chặn $w_0$ cũng bị phạt, điều thường không mong muốn: phạt hệ số chặn làm kết quả phụ thuộc vào việc $y$ có bị dịch đi một hằng số hay không. Các cài đặt thực tế, kể cả `sklearn.linear_model.Ridge`, không phạt hệ số chặn.

## Bài 2
@meta chuong=10 | dang=Suy luận | kho=Trung bình

**(a) Âm log hậu nghiệm.** Theo định lý Bayes, $p(w \mid \mathcal{D}) \propto p(\mathcal{D}\mid w)\,p(w)$.

Phần hợp lý, với nhiễu Gauss độc lập:

$$-\log p(\mathcal{D}\mid w) = \sum_{i=1}^{n}\frac{(y_i - w^\top x_i)^2}{2\sigma^2} + \text{hằng số} = \frac{\|y - Xw\|^2}{2\sigma^2} + \text{hằng số}.$$

Phần tiên nghiệm, với $w \sim \mathcal{N}(0, \tau^2 I)$:

$$-\log p(w) = \frac{\|w\|^2}{2\tau^2} + \text{hằng số}.$$

Cộng lại và bỏ mọi hằng số không phụ thuộc $w$:

$$-\log p(w\mid\mathcal{D}) \;\dot=\; \frac{\|y - Xw\|^2}{2\sigma^2} + \frac{\|w\|^2}{2\tau^2}.$$

**(b) Suy ra ridge.** Nhân cả hai vế với $2\sigma^2$ — phép này không đổi vị trí cực tiểu:

$$\hat w_{\text{MAP}} = \arg\min_w\; \|y - Xw\|^2 + \frac{\sigma^2}{\tau^2}\|w\|^2.$$

So với Định nghĩa 9.2, đây chính là bài toán ridge với

$$\boxed{\lambda = \frac{\sigma^2}{\tau^2}}$$

Kiểm chứng bằng số ở Mục 10.4: với $\sigma = 1$ và $\tau = 0{,}7$ thì $\lambda = 2{,}040816$, và nghiệm ridge dạng đóng lệch so với nghiệm MAP tìm bằng BFGS $1{,}97\times10^{-8}$, trong phạm vi dung sai của thuật toán tối ưu.

**(c) Tiên nghiệm chặt hơn.** $\tau$ nhỏ đi thì $\lambda = \sigma^2/\tau^2$ lớn lên, tức regularization mạnh hơn.

Chiều này hợp lý vì $\tau$ nhỏ nghĩa là ta tin chắc rằng các hệ số phải gần 0. Niềm tin càng chắc thì dữ liệu càng phải đưa ra bằng chứng mạnh mới kéo được hệ số ra xa 0, và đó chính là tác dụng của regularization mạnh.

Tỉ số $\sigma^2/\tau^2$ so sánh hai nguồn không chắc chắn: $\sigma^2$ là mức nhiễu của dữ liệu, $\tau^2$ là độ rộng của tiên nghiệm. Dữ liệu càng nhiễu ($\sigma$ lớn) thì tiên nghiệm càng được coi trọng; tiên nghiệm càng rộng ($\tau$ lớn) thì dữ liệu càng được coi trọng.

**(d) Tiên nghiệm Laplace.** Với $p(w_j) \propto \exp(-|w_j|/b)$:

$$-\log p(w) = \frac{1}{b}\sum_j |w_j| + \text{hằng số} = \frac{\|w\|_1}{b} + \text{hằng số},$$

cho ra lasso.

Tiên nghiệm này diễn đạt niềm tin rằng nhiều hệ số nên bằng đúng 0. Phân phối Laplace có đỉnh nhọn tại 0 (mật độ không khả vi ở đó), nên dồn nhiều xác suất sát 0 hơn phân phối Gauss, đồng thời có đuôi dày hơn nên vẫn cho phép một vài hệ số lớn. Đó là mô tả của một vector thưa.

## Bài 3
@meta chuong=5 | dang=Tính tay | kho=Cơ bản

**(a) Ước lượng ở $\kappa = 40\,000$.** Số điều kiện tăng 4 lần so với $10\,000$.

| | Bậc | Hệ số nhân | Ước lượng |
|---|---|---|---|
| Gradient descent | $O(\kappa)$ | $\times 4$ | $92\,104 \times 4 \approx \mathbf{368\,000}$ vòng |
| Có momentum | $O(\sqrt\kappa)$ | $\times 2$ | $1\,297 \times 2 \approx \mathbf{2\,600}$ vòng |

Tỉ số giữa hai phương pháp tăng từ 71 lên khoảng 142 lần: lợi ích của momentum tăng theo độ khó của bài toán, không phải một hệ số cố định.

**(b) "Bài toán tôi có 1 triệu chiều nên GD sẽ rất chậm."**

Nhận định này sai. Tốc độ hội tụ của gradient descent phụ thuộc số điều kiện $\kappa$ chứ không phụ thuộc số chiều $d$. Một bài toán 1 triệu chiều nhưng có $\kappa = 5$ hội tụ nhanh hơn nhiều một bài toán 2 chiều có $\kappa = 10\,000$.

Số chiều ảnh hưởng tới chi phí của mỗi vòng lặp ($O(nd)$ cho một lần tính gradient), không ảnh hưởng tới số vòng lặp cần thiết. Nhầm lẫn giữa hai điều này là một trong các câu trả lời chưa đạt ở Mục 18.8.

**(c) Chuẩn hoá đặc trưng.**

Trừ trung bình và chia cho độ lệch chuẩn từng cột. Cơ chế: nếu một đặc trưng có biên độ $10^6$ còn một đặc trưng khác có biên độ $10^{-3}$, thì các trị riêng của $X^\top X$ chênh nhau cỡ $10^{18}$, tức $\kappa \approx 10^{18}$. Đưa mọi đặc trưng về cùng phương sai làm các trị riêng về cùng cỡ độ lớn, và $\kappa$ giảm theo.

Cùng lý do giải thích vì sao chuẩn hoá quan trọng với mạng nơ-ron; xem [Chương 6 của *Học sâu*](models-ch06.html).

## Bài 4
@meta chuong=5 | dang=Suy luận | kho=Trung bình

**(a) Sai phân trung tâm.** Tổng sai số:

$$E(\varepsilon) = C_1\varepsilon^2 + \frac{C_2 u}{\varepsilon}.$$

Lấy đạo hàm và cho bằng 0:

$$E'(\varepsilon) = 2C_1\varepsilon - \frac{C_2u}{\varepsilon^2} = 0 \;\Longrightarrow\; \varepsilon^3 = \frac{C_2 u}{2C_1} \;\Longrightarrow\; \boxed{\varepsilon^* = \left(\frac{C_2u}{2C_1}\right)^{1/3} \propto u^{1/3}}$$

Sai số tại đó cỡ $u^{2/3} \approx 3{,}7\times10^{-11}$. Đây là độ chính xác tốt nhất có thể đạt với sai phân trung tâm; không chọn $\varepsilon$ nào cho kết quả tốt hơn.

**(b) Sai phân tiến.** Sai số cắt cụt chỉ bậc nhất:

$$E(\varepsilon) = C_1\varepsilon + \frac{C_2u}{\varepsilon}, \qquad E'(\varepsilon) = C_1 - \frac{C_2u}{\varepsilon^2} = 0 \;\Longrightarrow\; \boxed{\varepsilon^* \propto u^{1/2}}$$

Sai số tốt nhất đạt được cỡ $u^{1/2} \approx 1{,}5\times10^{-8}$.

**(c) Đối chiếu với số đo.**

| | Lý thuyết | Đo được |
|---|---|---|
| Trung tâm | $u^{1/3} = 6{,}06\times10^{-6}$ | $5{,}62\times10^{-6}$ |
| Tiến | $u^{1/2} = 1{,}49\times10^{-8}$ | $1{,}78\times10^{-8}$ |

Cả hai khớp trong phạm vi độ phân giải của lưới quét: lưới có 53 điểm trên 13 bậc độ lớn, tức mỗi bước là hệ số $1{,}8$, và cả hai độ lệch đều nhỏ hơn một bước lưới.

Sai số tốt nhất cũng khớp về bậc: sai phân trung tâm đạt $1{,}16\times10^{-11}$, so với dự đoán cỡ $u^{2/3} = 3{,}7\times10^{-11}$.

**(d) Vì sao ưa sai phân trung tâm.** Với mỗi toạ độ, sai phân trung tâm cần hai lần tính hàm, $f(x+\varepsilon)$ và $f(x-\varepsilon)$, còn sai phân tiến chỉ cần một lần tính mới vì $f(x)$ đã có sẵn. Đổi lại, sai số tốt nhất giảm từ cỡ $10^{-8}$ xuống cỡ $10^{-11}$, hơn ba bậc độ lớn.

Khoảng cách này quan trọng vì mục đích của việc kiểm tra là phân biệt lỗi cài đặt với sai số số học. Với sai phân tiến, sai số tương đối $10^{-7}$ có thể là lỗi mà cũng có thể chỉ là sai số số học, nên không kết luận được. Với sai phân trung tâm, sai số số học chỉ cỡ $10^{-11}$, nên $10^{-7}$ là dấu hiệu rõ của lỗi.

## Bài 5
@meta chuong=6 | dang=Tính tay | kho=Cơ bản

**(a) Chặn Novikoff.**

$$\left(\frac{R}{\gamma}\right)^2 = \left(\frac{2}{0{,}1}\right)^2 = 20^2 = \boxed{400 \text{ lần cập nhật}}$$

**(b) Thêm 10 000 điểm.** Chặn không đổi, vẫn là 400.

Công thức $(R/\gamma)^2$ không chứa $n$ và không chứa $d$: số lần perceptron cập nhật bị chặn bởi hình học của bài toán, tức kích thước của dữ liệu và độ rộng khoảng trống giữa hai lớp, chứ không bởi lượng dữ liệu hay số chiều.

Một hệ quả cụ thể: perceptron vẫn chạy được trên dữ liệu có số chiều rất cao (ví dụ văn bản với hàng trăm nghìn đặc trưng), miễn là dữ liệu tách được với lề không quá nhỏ.

**(c) Nhân mọi $x_i$ với 10.**

- $R \to 10R = 20$.
- $\gamma \to 10\gamma = 1$, vì lề là $\min_i y_i(w^{*\top}x_i)$ với $\|w^*\| = 1$, nên nó co giãn cùng $x$.
- Chặn: $(20/1)^2 = 400$, không đổi.

Chặn bất biến với phép co giãn, đúng như mong đợi: nhân toàn bộ dữ liệu với một hằng số chỉ đổi đơn vị đo, không làm bài toán dễ hơn hay khó hơn.

**(d) Vì sao không có bảo đảm khi dữ liệu không tách được.**

Chứng minh của Novikoff dựa vào sự tồn tại của một vector $w^*$ tách đúng mọi điểm với lề $\gamma > 0$. Khi dữ liệu không tách được, vector đó không tồn tại và lập luận không áp dụng được.

Nhìn từ hàm mất mát (Hình 5): mất mát perceptron $\max(0, -m)$ bằng 0 với mọi $m \ge 0$, nên thuật toán ngừng điều chỉnh một điểm ngay khi nó vừa được phân loại đúng, dù nó nằm sát biên. Khi các lớp chồng lấn, sửa một điểm lại làm sai một điểm khác, và quá trình không kết thúc.

Ở Mục 6.2, trên dữ liệu không tách được, perceptron mắc 17 977 lỗi qua 2 000 lượt duyệt và không hội tụ. Mất mát hinge và logistic vẫn dương tại $m = 0$, nên vẫn tiếp tục đẩy các điểm ra xa biên; kết hợp với regularization, chúng cho bài toán lồi có nghiệm xác định kể cả khi dữ liệu không tách được.

## Bài 6
@meta chuong=8 | dang=Tính tay | kho=Cơ bản

**(a) Bốn chỉ số.** Từ bảng: $TP = 102$, $FN = 94$, $FP = 105$, $TN = 19\,699$; tổng $n = 20\,000$.

$$\text{độ chính xác} = \frac{102 + 19\,699}{20\,000} = \frac{19\,801}{20\,000} = \mathbf{0{,}9900}$$

$$\text{precision} = \frac{102}{102 + 105} = \frac{102}{207} = \mathbf{0{,}4928}$$

$$\text{recall} = \frac{102}{102 + 94} = \frac{102}{196} = \mathbf{0{,}5204}$$

$$F_1 = \frac{2 \times 0{,}4928 \times 0{,}5204}{0{,}4928 + 0{,}5204} = \mathbf{0{,}5062}$$

**(b) Bộ "luôn đoán bình thường".** Nó đúng ở toàn bộ lớp âm và sai ở toàn bộ lớp dương:

$$\text{độ chính xác} = \frac{19\,699 + 105}{20\,000} = \frac{19\,804}{20\,000} = \mathbf{0{,}9902}$$

**(c) Kết luận.** Bộ phân loại không phát hiện được ca gian lận nào có độ chính xác 0,9902, cao hơn bộ phân loại ở (a) (0,9900). Chọn mô hình theo độ chính xác sẽ dẫn tới chọn mô hình vô dụng.

Độ chính xác bị chi phối bởi lớp đa số: lớp âm chiếm 99,02% nên luôn đoán lớp âm đã được 99,02%. Mọi cải thiện trên lớp dương chỉ có thể đóng góp tối đa 0,98 điểm phần trăm vào độ chính xác.

Độ chính xác chỉ phù hợp khi các lớp tương đối cân bằng. Ở đây nên dùng precision, recall, $F_1$ hoặc PR-AUC.

**(d) Bỏ sót đắt gấp 20 lần báo động nhầm.**

Nên hạ ngưỡng. Ngưỡng thấp hơn làm mô hình báo động nhiều hơn: recall tăng, precision giảm, đúng chiều cần thiết khi FN đắt hơn FP.

Thước đo để chọn ngưỡng không nên là $F_1$, vì $F_1$ coi precision và recall quan trọng như nhau, điều không đúng ở đây. Hai lựa chọn phù hợp:

1. $F_\beta$ với $\beta = \sqrt{20} \approx 4{,}5$, vì $F_\beta$ đặt trọng số cho recall gấp $\beta^2$ lần precision:
   $$F_\beta = \frac{(1+\beta^2)\,PR}{\beta^2 P + R}.$$
2. Cực tiểu trực tiếp chi phí kỳ vọng: đặt $\text{chi phí} = 20 \cdot FN + 1 \cdot FP$ rồi quét ngưỡng để tìm giá trị nhỏ nhất. Cách này tối ưu đúng đại lượng mà bộ phận rủi ro quan tâm.

Khi biết chi phí thật, cách thứ hai nên được ưu tiên; các thước đo như $F_1$ dùng khi không biết chi phí.

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

Hai giá trị bằng nhau. Điều này hợp lý: chỉ có hai điểm, cả hai đều là vector hỗ trợ và nằm đúng trên hai mép lề, siêu phẳng nằm chính giữa chúng, nên độ rộng lề bằng đúng khoảng cách giữa hai điểm.

**(c) Nhân tử Lagrange.** Từ $w = \sum_i \alpha_i y_i x_i$:

$$(0{,}5;\,0{,}5) = \alpha_1 \cdot (+1) \cdot (1,1) + \alpha_2 \cdot (-1) \cdot (-1,-1) = (\alpha_1 + \alpha_2)(1,1),$$

nên $\alpha_1 + \alpha_2 = 0{,}5$. Ràng buộc $\sum_i\alpha_i y_i = 0$ cho $\alpha_1 - \alpha_2 = 0$. Giải hệ:

$$\boxed{\alpha_1 = \alpha_2 = 0{,}25}$$

**(d) Đối ngẫu mạnh.**

Bài toán gốc: $\tfrac12\|w\|^2 = \tfrac12 (0{,}5^2 + 0{,}5^2) = \tfrac12 \cdot 0{,}5 = \mathbf{0{,}25}$.

Bài toán đối ngẫu: với $x_1^\top x_1 = 2$, $x_2^\top x_2 = 2$, $x_1^\top x_2 = -2$ và $y_1y_2 = -1$:

$$\sum_i\alpha_i - \tfrac12\sum_{i,j}\alpha_i\alpha_j y_iy_j x_i^\top x_j = 0{,}5 - \tfrac12\big[0{,}0625\cdot2 + 0{,}0625\cdot2 + 2\cdot0{,}0625\cdot(-1)\cdot(-2)\big]$$
$$= 0{,}5 - \tfrac12 \cdot 0{,}5 = \mathbf{0{,}25}.$$

Khe đối ngẫu bằng 0, tức đối ngẫu mạnh, đúng như điều kiện Slater bảo đảm.

**(e) Thêm điểm $x_3 = (5,5)$ nhãn $+1$.**

Nghiệm không đổi. Kiểm tra ràng buộc của điểm mới với nghiệm cũ:

$$y_3(w^\top x_3 + b) = +1 \cdot (0{,}5\cdot5 + 0{,}5\cdot5) = 5 \;\ge\; 1. \;\checkmark$$

Ràng buộc thoả nhưng không chặt (bằng 5 chứ không bằng 1). Theo điều kiện bù $\alpha_i f_i(x^*) = 0$ ở Mục 12.4, ràng buộc không chặt kéo theo $\alpha_3 = 0$, nên $x_3$ không đóng góp vào $w = \sum_i\alpha_i y_i x_i$.

Đây là cơ chế tạo ra tính thưa của SVM, và là lý do ở Mục 13.3 chỉ 3 trên 120 điểm quyết định nghiệm. Thêm bao nhiêu điểm nằm ngoài lề cũng không làm mô hình thay đổi.

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

**(b) Sai số tái tạo với $k = 2$, $n = 101$.** Theo định lý Eckart–Young (Mục 14.3):

$$\|X_c - X_k\|_F^2 = (n-1)\sum_{i>k}\lambda_i = 100 \times (3 + 1{,}5 + 0{,}5) = 100 \times 5 = \boxed{500}.$$

**(c) Bác bỏ "giữ 90% phương sai nên chỉ kém đi một chút".**

Phát biểu này lẫn lộn hai đại lượng khác nhau: phương sai được giữ lại và thông tin phân loại được giữ lại. PCA cực đại đại lượng thứ nhất và không xét đại lượng thứ hai, vì nó không dùng nhãn.

Phản ví dụ ở Mục 14.4: thành phần chính thứ nhất giữ 96,1% phương sai nhưng cho AUC 0,502, ngang đoán ngẫu nhiên. Trên cùng dữ liệu, hướng của LDA cho AUC 0,9987.

Cơ chế: dữ liệu có một hướng nhiễu biên độ lớn, không mang thông tin về lớp, và một hướng tín hiệu biên độ nhỏ. PCA chọn hướng nhiễu vì phương sai của nó lớn hơn, và bỏ đi đúng hướng cần giữ.

Phương sai lớn không có nghĩa là hữu ích cho bài toán. Khi mục tiêu là phân loại và có nhãn, nên thử LDA, hoặc ít nhất đo hiệu năng phân loại sau khi giảm chiều thay vì dựa vào tỉ lệ phương sai.

**(d) Đổi đơn vị từ mét sang milimét.**

Đặc trưng đó được nhân với 1 000, nên phương sai của nó nhân với $10^6$. Ma trận hiệp phương sai thay đổi theo, và trị riêng ứng với hướng gần trục của đặc trưng đó tăng lên cỡ $10^6$ lần.

Hệ quả là thành phần chính thứ nhất gần như trùng với đặc trưng đó, bất kể nó có quan trọng hay không. Kết quả PCA khi đó phản ánh việc chọn đơn vị đo nhiều hơn là cấu trúc của dữ liệu.

Vì PCA không bất biến với phép co giãn từng đặc trưng, cần chuẩn hoá trước khi áp dụng khi các đặc trưng có đơn vị khác nhau. Việc này tương đương với làm PCA trên ma trận tương quan thay vì ma trận hiệp phương sai. Ngoại lệ là khi mọi đặc trưng đã cùng đơn vị và thang đo có ý nghĩa, như cường độ điểm ảnh.

Cây quyết định không gặp vấn đề này, vì cây bất biến với mọi phép biến đổi đơn điệu trên từng đặc trưng; xem [Chương 3 của *Học sâu*](models-ch03.html).

## Bài 9
@meta chuong=15 | dang=Chẩn đoán | kho=Trung bình

**(a) Năm lần chạy cho năm kết quả khác hẳn nhau.**

**Nguyên nhân:** mỗi lần chạy dừng ở một cực tiểu địa phương khác nhau. Đây là hành vi bình thường của K-means, không phải lỗi, vì bài toán NP-khó và hàm mục tiêu không lồi. Ở Mục 15.2, khởi tạo ngẫu nhiên cho nghiệm tồi trong 71,5% số lần chạy.

**Hai việc nên làm:**
1. Chạy nhiều lần (10 tới 50) với hạt giống khác nhau rồi giữ kết quả có inertia nhỏ nhất. Trong scikit-learn, đặt tham số `n_init` tường minh, vì từ phiên bản 1.4 giá trị mặc định `'auto'` chỉ chạy một lần với k-means++.
2. Dùng k-means++ thay cho khởi tạo ngẫu nhiên. Trong thí nghiệm, nó giảm tỉ lệ nghiệm tồi xuống 46,5%.

**(b) Chia đôi cụm dài dẹt, gộp hai cụm tròn nhỏ gần nhau.**

**Nguyên nhân:** dữ liệu vi phạm giả định về hình dạng cụm. K-means gán điểm theo khoảng cách Euclid tới tâm, nên biên giữa hai cụm luôn là siêu phẳng, và thuật toán ưu tiên các cụm tròn có kích thước tương đương. Các điểm ở hai đầu một cụm dài có thể gần tâm của cụm bên cạnh hơn tâm của chính cụm đó, nên cụm dài bị chia đôi.

Ở Mục 15.3, trên hai dải dẹt, độ chính xác chỉ đạt 0,5317, gần như đoán ngẫu nhiên.

**Hai việc nên làm:**
1. Dùng mô hình hỗn hợp Gauss, cho phép mỗi cụm có dạng elip với hướng và độ dẹt riêng.
2. Dùng DBSCAN hoặc phân cụm phổ, hai phương pháp xử lý được cụm có hình dạng tuỳ ý. DBSCAN còn tự xác định số cụm.

Nếu bắt buộc dùng K-means, có thể biến đổi dữ liệu trước (chuẩn hoá theo từng trục hoặc làm trắng dữ liệu) để các cụm tròn hơn, nhưng cách này chỉ hiệu quả khi mọi cụm dẹt theo cùng một hướng.

**(c) Inertia giảm đều, không thấy khuỷu tay.**

**Nguyên nhân:** rất có thể dữ liệu không có cấu trúc cụm rõ ràng. Với dữ liệu phân bố đều hoặc một đám mây Gauss duy nhất, inertia giảm đều theo $k$ và không có chỗ gãy, vì không có số cụm nào đúng hơn các số khác.

Cũng có thể cụm tồn tại nhưng chồng lấn nhiều, hoặc số cụm thật lớn hơn khoảng $k$ đang quét.

**Hai việc nên làm:**
1. Dùng tiêu chí khác: hệ số silhouette (mức một điểm hợp với cụm của nó hơn cụm gần nhất khác) hoặc gap statistic (so inertia với inertia trên dữ liệu ngẫu nhiên cùng phạm vi). Gap statistic đặc biệt phù hợp ở đây vì phát hiện được trường hợp dữ liệu không có cụm.
2. Trực quan hoá bằng PCA hoặc UMAP trước khi phân cụm. Nếu không thấy cụm trên hình chiếu, khả năng thuật toán tìm được cụm có ý nghĩa cũng thấp.

Cũng cần cân nhắc khả năng phân cụm không phải công cụ phù hợp cho bài toán này.

**(d) Một cụm chứa 98% số điểm.**

**Nguyên nhân:** thường do thang đo hoặc điểm ngoại lai. Nếu một đặc trưng có độ lớn vượt hẳn các đặc trưng khác, khoảng cách Euclid gần như chỉ phản ánh đặc trưng đó. Vài điểm ngoại lai ở rất xa có thể kéo các tâm cụm về phía chúng, để lại một tâm cho toàn bộ phần dữ liệu còn lại.

**Hai việc nên làm:**
1. Chuẩn hoá đặc trưng về cùng thang đo trước khi phân cụm, theo giả định thứ ba ở Mục 15.3.
2. Phát hiện và xử lý điểm ngoại lai trước, hoặc dùng K-medoids, phương pháp chọn tâm là điểm dữ liệu thật nên ít bị điểm ngoại lai ảnh hưởng.

## Bài 10
@meta chuong=16 | dang=Thiết kế | kho=Nâng cao

**(a) Số tham số.**

$$(n_u + n_i)\,k = (10^6 + 10^5) \times 50 = 1{,}1\times10^6 \times 50 = \boxed{55\,000\,000}$$

Số ô của ma trận đầy đủ: $10^6 \times 10^5 = 10^{11}$.

$$\frac{5{,}5\times10^7}{10^{11}} = \mathbf{0{,}055\%}$$

Mô hình biểu diễn $10^{11}$ ô bằng 55 triệu số, nén hơn 1 800 lần. Khả năng dự đoán các ô trống đến từ giả thiết hạng thấp, và con số 0,055% cho thấy giả thiết đó mạnh tới mức nào.

**(b) Lượng đánh giá cần.**

Theo Mục 16.3, lấy mức an toàn khoảng $5k$ đánh giá mỗi người:

$$5 \times 50 = \boxed{250 \text{ đánh giá mỗi người}}$$

Tổng cộng: $10^6 \times 250 = \mathbf{250}$ **triệu đánh giá**, bằng

$$\frac{2{,}5\times10^8}{10^{11}} = \mathbf{0{,}25\%} \text{ số ô}.$$

**(c) Thực tế chỉ có 30 đánh giá mỗi người.**

30 so với 250 là thiếu hơn 8 lần. Trong bảng ở Mục 16.3, với khoảng 10 đánh giá mỗi người và $k=4$, RMSE còn tệ hơn đoán mọi ô bằng 0; tỉ lệ đánh giá trên mỗi nhân tố ẩn ở đây là $30/50 = 0{,}6$, còn thấp hơn tỉ lệ $10/4 = 2{,}5$ của trường hợp đó.

**Dự đoán:** mô hình sẽ khớp rất tốt các ô đã quan sát nhưng dự đoán các ô chưa quan sát còn kém hơn đoán điểm trung bình. Đây là overfitting nghiêm trọng, và không phát hiện được nếu chỉ nhìn sai số huấn luyện.

**Hai cách xử lý:**

1. Giảm $k$ mạnh. Với 30 đánh giá mỗi người, mức $5k$ cho $k \approx 6$. Bảng ở Mục 16.4 cho thấy chọn $k$ quá lớn làm RMSE trên ô chưa quan sát tăng: ở $k=20$, sai số lớn gấp 4,3 lần so với hạng đúng.
2. Tăng hệ số regularization $\lambda$, và chọn nó bằng cách đánh giá trên các ô được giữ lại, không phải trên các ô dùng để huấn luyện. Regularization mạnh làm mô hình hạng 50 hoạt động gần giống một mô hình hạng thấp hơn.

Một cách thứ ba là bổ sung đặc trưng phụ (thể loại, thông tin nhân khẩu học), tức chuyển sang mô hình lai. Đặc trưng phụ không phải ước lượng từ dữ liệu đánh giá nên không tiêu tốn 30 đánh giá ít ỏi của mỗi người.

**(d) Người dùng mới phải nhận gợi ý ngay từ phiên đầu.**

Phân rã ma trận không làm được việc này. Như ở Mục 16.5, với người dùng chưa có đánh giá, hàm mất mát chỉ còn $\lambda\|p_u\|^2$, nên $p_u$ về 0 và mô hình dự đoán gần 0 cho mọi sản phẩm: RMSE 2,24, tệ hơn đoán mọi ô bằng 0 (2,08).

**Thiết kế phương án dự phòng theo ba tầng:**

| Tầng | Điều kiện | Cách gợi ý |
|---|---|---|
| 0 | 0 tương tác | Sản phẩm phổ biến, lọc theo ngữ cảnh có sẵn (quốc gia, thiết bị, thời điểm), kèm một tỉ lệ nhỏ gợi ý ngẫu nhiên để thu thập tín hiệu. |
| 1 | 1 tới 30 tương tác | Lọc dựa trên nội dung: khớp đặc trưng sản phẩm với hồ sơ suy ra từ vài tương tác đầu. |
| 2 | trên 30 tương tác | Phân rã ma trận, hoặc mô hình lai kết hợp cả hai nguồn. |

**Khi nào chuyển tầng.** Ngưỡng nên được đo chứ không chọn theo cảm tính: với những người dùng đã có nhiều dữ liệu, cắt lịch sử của họ xuống $m$ tương tác rồi đo hiệu năng của cả hai phương pháp theo $m$. Ngưỡng chuyển là điểm mà phân rã ma trận bắt đầu vượt lọc dựa trên nội dung. Mức $5k$ chỉ là điểm xuất phát để tìm ngưỡng đó.

**Hai điều cần thiết kế ngay từ đầu:**

1. Hỏi sở thích khi đăng ký. Cho người dùng mới chọn vài thể loại là cách rẻ nhất để chuyển ngay từ tầng 0 lên tầng 1.
2. Dành một tỉ lệ nhỏ gợi ý cho thăm dò ngẫu nhiên. Nếu hệ thống chỉ gợi ý những gì nó đã biết, nó chỉ nhận phản hồi về những thứ đó. [Mục 12.3 của *MLOps*](mlops-ch12.html) mô phỏng hiện tượng này: không có thăm dò ngẫu nhiên, hệ thống chỉ quan sát được 9,5% danh mục. Đây là quyết định kiến trúc, cần có từ đầu.
