# Ngân hàng câu hỏi tự kiểm tra — Nền tảng học máy

Cú pháp: `## Chương N` mở một nhóm, `### …` là câu hỏi, `- [x]` đánh dấu đáp án đúng,
dòng `>` là phần giải thích hiện ra sau khi trả lời.

## Chương 1

### Ba thành phần của mọi thuật toán học có giám sát là gì?
- [ ] Dữ liệu, mô hình, độ chính xác
- [x] Mô hình, hàm mất mát, thuật toán tối ưu
- [ ] Đặc trưng, nhãn, siêu tham số
- [ ] Huấn luyện, xác thực, kiểm tra
> Thay đổi một thành phần là được một thuật toán khác. Chương 6 cho thấy perceptron, hồi quy logistic và SVM dùng chung mô hình tuyến tính và chung cách tối ưu; chúng chỉ khác nhau ở hàm mất mát.

### Yếu tố nào nên xét trước khi chọn thuật toán?
- [ ] Thư viện nào đang được dùng nhiều nhất
- [x] Tỉ lệ giữa số điểm dữ liệu và số đặc trưng, cấu trúc của dữ liệu, và mục tiêu là dự đoán hay giải thích
- [ ] Mô hình nào đạt điểm cao nhất trên các cuộc thi
- [ ] Có bao nhiêu GPU
> Tỉ lệ $n/d$ quyết định mô hình được phép linh hoạt tới đâu. Mục tiêu dự đoán và mục tiêu giải thích có thể dẫn tới lựa chọn khác nhau; Mục 4.4 cho một ví dụ đo được.

### Tập kiểm tra nên được dùng thế nào?
- [ ] Dùng nhiều lần để chọn mô hình tốt nhất
- [x] Chỉ dùng một lần ở bước cuối để báo cáo kết quả; việc chọn mô hình và siêu tham số làm trên tập xác thực
- [ ] Gộp vào tập huấn luyện để có thêm dữ liệu
- [ ] Dùng để chuẩn hoá đặc trưng
> Nếu tập kiểm tra được dùng nhiều lần để chọn mô hình, nó trở thành một tập xác thực thứ hai và con số báo cáo sẽ lạc quan hơn thực tế.

## Chương 2

### Số điều kiện của $X^\top X$ so với số điều kiện của $X$ thế nào?
- [ ] Bằng nhau
- [x] Bằng bình phương số điều kiện của $X$
- [ ] Bằng một nửa
- [ ] Không liên quan
> Vì $X^\top X = VD^2V^\top$, các trị riêng của $X^\top X$ là bình phương các giá trị suy biến của $X$. Số điều kiện vì thế bị bình phương, nên các thư viện giải bình phương tối thiểu bằng phân tích QR hoặc SVD thay vì dùng công thức $(X^\top X)^{-1}X^\top y$.

### Vì sao regularization $\ell_1$ cho hệ số bằng đúng 0 còn $\ell_2$ thì không?
- [ ] Vì $\ell_1$ phạt nặng hơn
- [x] Vì $\lvert w\rvert$ không khả vi tại 0: để hệ số rời khỏi 0, gradient của phần sai số phải thắng một lực có độ lớn cố định $\lambda$
- [ ] Vì $\ell_1$ không khả vi nên thuật toán bị lỗi và trả về 0
- [ ] Vì $\ell_2$ luôn dùng $\lambda$ nhỏ hơn
> Với $w^2$ thì đạo hàm là $2w$, nhỏ dần khi $w$ tiến về 0, nên lực kéo không bao giờ đưa hệ số tới đúng 0. Ở Mục 9.4, với $\lambda = 100$, lasso đưa 9 trong 12 hệ số về 0 còn ridge không đưa hệ số nào về 0.

### Cộng $\lambda I$ ($\lambda > 0$) vào một ma trận nửa xác định dương thì được gì?
- [x] Một ma trận xác định dương, nên luôn khả nghịch
- [ ] Ma trận vẫn có thể suy biến
- [ ] Một ma trận trực giao
- [ ] Một ma trận có định thức bằng 0
> Các trị riêng tăng thêm $\lambda$ nên đều dương. Vì vậy hồi quy ridge luôn có nghiệm duy nhất, kể cả khi $d > n$.

### Khoảng cách từ điểm $x_0 = (2, 1)$ tới đường thẳng $3x_1 + 4x_2 - 5 = 0$ bằng bao nhiêu?
- [ ] 5
- [x] 1
- [ ] 0,2
- [ ] 2
> Theo Định lý 2.1: $|3\cdot2 + 4\cdot1 - 5|/\sqrt{3^2 + 4^2} = 5/5 = 1$.

## Chương 3

### Phát biểu nào đúng?
- [ ] $\operatorname{Var}(X+Y) = \operatorname{Var}(X) + \operatorname{Var}(Y)$ luôn đúng
- [x] $\mathbb{E}[X+Y] = \mathbb{E}[X] + \mathbb{E}[Y]$ luôn đúng, còn phương sai của tổng chỉ bằng tổng các phương sai khi hai biến không tương quan
- [ ] Cả kỳ vọng và phương sai của tổng chỉ cộng được khi độc lập
- [ ] Cả hai luôn cộng được
> Khi có tương quan, $\operatorname{Var}(X+Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) + 2\operatorname{Cov}(X,Y)$. Hệ quả ở Ví dụ 3.1: trung bình nhiều mô hình có tương quan $\rho$ không giảm phương sai xuống dưới $\rho\sigma^2$.

### Ép ma trận hiệp phương sai của mỗi lớp thành ma trận đường chéo tương đương với giả thiết nào?
- [ ] Dữ liệu có phân phối chuẩn
- [x] Các đặc trưng độc lập có điều kiện khi đã biết lớp, tức giả thiết của Naive Bayes (với đặc trưng Gauss)
- [ ] Các lớp có cùng số điểm
- [ ] Mọi đặc trưng có cùng phương sai
> Mục 7.4 kiểm chứng bằng số: với đặc trưng Gauss, Naive Bayes, LDA và QDA là cùng một bộ phân loại, chỉ khác ràng buộc trên $\Sigma$.

### Bệnh có tỉ lệ mắc 1%, xét nghiệm dương tính với 99% người bệnh và dương tính nhầm với 5% người khoẻ. Một người có kết quả dương tính. Xác suất người đó mắc bệnh khoảng bao nhiêu?
- [ ] 99%
- [ ] 95%
- [ ] 50%
- [x] 17%
> Theo định lý Bayes: $0{,}99\cdot0{,}01/(0{,}99\cdot0{,}01 + 0{,}05\cdot0{,}99) \approx 0{,}167$. Người khoẻ đông gấp 99 lần nên 5% dương tính nhầm của họ nhiều hơn số người bệnh được phát hiện.

### Vì sao ước lượng phương sai mẫu chia cho $n-1$ thay vì $n$?
- [ ] Để phương sai luôn dương
- [x] Vì chia cho $n$ cho ước lượng có chệch, kỳ vọng bằng $\frac{n-1}{n}\sigma^2$; chia cho $n-1$ cho ước lượng không chệch
- [ ] Vì một điểm dữ liệu luôn là điểm ngoại lai
- [ ] Để công thức tính nhanh hơn
> Các điểm luôn gần trung bình mẫu $\bar x$ hơn so với gần trung bình thật, vì $\bar x$ được tính từ chính các điểm đó.

## Chương 4

### Khi nào phương trình chuẩn không có nghiệm duy nhất?
- [ ] Khi dữ liệu có nhiễu
- [x] Khi $X$ không đủ hạng cột: nhiều đặc trưng hơn số điểm, hoặc có cột phụ thuộc tuyến tính vào các cột khác
- [ ] Khi $y$ có giá trị âm
- [ ] Khi chưa chuẩn hoá đặc trưng
> Trường hợp hay gặp: mã hoá one-hot đủ $K$ giá trị trong khi vẫn giữ cột hằng số, vì tổng $K$ cột one-hot bằng đúng cột hằng số.

### Khi $X^\top X$ suy biến, bài toán bình phương tối thiểu có đặc điểm gì?
- [ ] Không có nghiệm nào
- [x] Có vô số nghiệm cho cùng một dự đoán; giả nghịch đảo chọn nghiệm có chuẩn nhỏ nhất
- [ ] Có nghiệm duy nhất nhưng khó tính
- [ ] Dữ liệu bị lỗi và phải thu thập lại
> Ở Mục 4.3, hai vector trọng số có chuẩn 3,8176 và 4,0711 cho cùng một dự đoán. Nghiệm chuẩn nhỏ nhất của giả nghịch đảo là giới hạn của nghiệm ridge khi $\lambda \to 0^+$.

### Đa cộng tuyến ảnh hưởng chủ yếu tới điều gì?
- [ ] Độ chính xác của dự đoán
- [x] Độ ổn định của các hệ số, tức khả năng diễn giải; dự đoán gần như không bị ảnh hưởng
- [ ] Tốc độ huấn luyện
- [ ] Cả dự đoán lẫn hệ số như nhau
> Ở Mục 4.4, khi tương quan tăng từ 0 lên 0,999, độ lệch chuẩn của $w_1$ tăng khoảng 25 lần (0,0324 lên 0,8196) trong khi độ lệch chuẩn của dự đoán gần như không đổi. Lý thuyết dự đoán mức tăng $\sqrt{\text{VIF}} = 1/\sqrt{1-\rho^2} \approx 22$.

### Trong hồi quy tuyến tính có hệ số chặn, tổng các phần dư trên tập huấn luyện bằng bao nhiêu?
- [ ] Bằng tổng bình phương sai số
- [x] Bằng 0
- [ ] Bằng số điểm dữ liệu
- [ ] Không xác định được
> Phương trình chuẩn nói $X^\top r = 0$. Hàng ứng với cột hằng số của $X^\top$ gồm toàn số 1, nên $\sum_i r_i = 0$.

## Chương 5

### Tốc độ hội tụ của gradient descent trên hàm bậc hai phụ thuộc vào điều gì?
- [ ] Số chiều $d$
- [x] Số điều kiện $\kappa$: số vòng lặp tỉ lệ với $\kappa\log(1/\varepsilon)$
- [ ] Số điểm dữ liệu $n$
- [ ] Giá trị khởi tạo
> Ở Mục 5.3, khi $\kappa$ tăng từ 1 lên 10 000, số vòng lặp tăng từ 1 lên 92 104, khớp với ước lượng $\frac{\kappa}{2}\ln 10^8 \approx 92\,103$. Số chiều ảnh hưởng tới chi phí mỗi vòng lặp, không ảnh hưởng tới số vòng lặp.

### Momentum (heavy ball) thay đổi bậc của số vòng lặp thế nào?
- [ ] Từ bậc $\kappa$ xuống bậc $\log\kappa$
- [x] Từ bậc $\kappa$ xuống bậc $\sqrt\kappa$
- [ ] Từ bậc $\kappa^2$ xuống bậc $\kappa$
- [ ] Không đổi bậc, chỉ đổi hằng số
> Ở $\kappa = 10\,000$, gradient descent cần 92 104 vòng lặp, có momentum chỉ cần 1 297. Lợi ích càng lớn khi $\kappa$ càng lớn.

### Bước tiền xử lý đơn giản nào giúp gradient descent nhanh hơn nhiều nhất?
- [x] Chuẩn hoá đặc trưng về cùng thang đo, vì nó giảm trực tiếp số điều kiện
- [ ] Sắp xếp dữ liệu theo nhãn
- [ ] Bỏ bớt điểm dữ liệu
- [ ] Đổi kiểu dữ liệu sang float32
> Nếu một đặc trưng có độ lớn cỡ $10^6$ và một đặc trưng khác cỡ $10^{-3}$, các trị riêng của $X^\top X$ chênh nhau rất nhiều bậc. Chuẩn hoá đưa chúng về cùng cỡ.

### Với $L(w) = (w-3)^2$, gradient descent hội tụ khi tốc độ học $\eta$ nằm trong khoảng nào?
- [ ] $0 < \eta < 3$
- [x] $0 < \eta < 1$
- [ ] $0 < \eta < 2$
- [ ] Mọi $\eta > 0$
> Mỗi bước nhân khoảng cách tới cực tiểu với $1 - 2\eta$, nên cần $|1 - 2\eta| < 1$. Tổng quát, với hàm bậc hai cần $\eta < 2/\lambda_{\max}$; ở đây $\lambda_{\max} = L'' = 2$.

### Khi kiểm tra gradient bằng sai phân trung tâm, nên chọn $\varepsilon$ thế nào?
- [ ] Càng nhỏ càng chính xác
- [x] Có điểm tối ưu cỡ $u^{1/3} \approx 6\times10^{-6}$; nhỏ hơn thì sai số tăng lên
- [ ] Luôn dùng $\varepsilon = 10^{-12}$
- [ ] Bằng tốc độ học
> Sai số cắt cụt tỉ lệ $\varepsilon^2$, sai số làm tròn tỉ lệ $u/\varepsilon$. Thí nghiệm đo được điểm tối ưu $5{,}62\times10^{-6}$, khớp lý thuyết $6{,}06\times10^{-6}$; với $\varepsilon = 10^{-13}$ sai số lớn hơn khoảng $5\times10^{7}$ lần.

## Chương 6

### Perceptron, hồi quy logistic và SVM khác nhau ở đâu?
- [ ] Ở mô hình: tuyến tính, phi tuyến và dựa trên kernel
- [x] Ở hàm mất mát; cả ba dùng chung mô hình tuyến tính
- [ ] Ở thuật toán tối ưu
- [ ] Ở cách chuẩn hoá dữ liệu
> Mất mát 0–1 không tối ưu được bằng gradient, nên cả ba dùng hàm thay thế lồi. Hinge và logistic (theo logarit cơ số 2) còn là chặn trên của mất mát 0–1; mất mát perceptron $\max(0,-m)$ thì không, vì nó bằng 0 tại $m = 0$.

### Vì sao perceptron không hội tụ trên dữ liệu không tách được?
- [ ] Vì tốc độ học quá lớn
- [x] Vì mất mát $\max(0,-m)$ bằng 0 ngay khi $m \ge 0$: thuật toán ngừng điều chỉnh một điểm ngay khi nó vừa được phân loại đúng, nên khi các lớp chồng lấn, sửa điểm này lại làm sai điểm khác
- [ ] Vì hàm mất mát không lồi
- [ ] Vì dữ liệu chưa được chuẩn hoá
> Ở Mục 6.2, trên dữ liệu không tách được, perceptron mắc 17 977 lỗi qua 2 000 lượt duyệt mà không dừng. Định lý Novikoff chỉ áp dụng cho dữ liệu tách được.

### Hồi quy logistic và hồi quy softmax với $K=2$ quan hệ thế nào?
- [ ] Softmax mạnh hơn vì có nhiều tham số hơn
- [x] Chúng là cùng một mô hình; hiệu hai vector trọng số của softmax chính là vector trọng số của logistic
- [ ] Chúng cho kết quả gần giống nhau nhưng khác về bản chất
- [ ] Softmax chỉ dùng được khi $K \ge 3$
> Ở Mục 6.4, khớp cả hai mô hình tới hội tụ, xác suất lệch nhau $3{,}3\times10^{-16}$ và trọng số lệch nhau $1{,}6\times10^{-15}$, tức trùng nhau tới sai số làm tròn.

### Vì sao hồi quy logistic thường cần regularization?
- [ ] Để chạy nhanh hơn
- [x] Vì nếu dữ liệu tách được hoàn toàn, hàm mất mát không có cực tiểu hữu hạn và $\lVert w\rVert \to \infty$
- [ ] Vì hàm mất mát không lồi
- [ ] Vì gradient không tính được
> Ở Mục 6.5, không có regularization thì $\|w\|$ tăng từ 10,2 lên 34,8 khi số vòng lặp tăng từ 500 lên 50 000; với $\lambda = 0{,}01$, $\|w\|$ đứng yên ở 3,993.

### Gradient của cross-entropy trong hồi quy logistic có dạng nào?
- [ ] $X^\top(y - p)\,p(1-p)$
- [x] $\frac{1}{n}X^\top(p - y)$
- [ ] $\frac{2}{n}X^\top(Xw - y)$
- [ ] $X^\top y$
> Nhờ $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, các thừa số rút gọn còn $(p_i - y_i)x_i$ cho mỗi điểm. Dạng này giống gradient của hồi quy tuyến tính vì cả hai là mô hình tuyến tính tổng quát với hàm liên kết chính tắc.

## Chương 7

### Lời nguyền số chiều ảnh hưởng tới k-NN thế nào?
- [x] Ở số chiều cao, láng giềng gần nhất không còn gần: với 100 chiều, vùng lân cận chứa 1% dữ liệu cần cạnh dài 0,955 lần cạnh của miền dữ liệu
- [ ] Bộ nhớ không đủ để lưu dữ liệu
- [ ] Nhãn trở nên nhiễu hơn
- [ ] Thuật toán không dùng được khoảng cách Euclid
> Cạnh cần thiết là $f^{1/d}$. Vì vậy k-NN chỉ hiệu quả ở số chiều thấp, sau khi giảm chiều, hoặc trong không gian embedding đã được học.

### Naive Bayes Gauss, LDA và QDA khác nhau ở đâu?
- [ ] Ở loại phân phối được dùng
- [x] Ở ràng buộc trên ma trận hiệp phương sai; cả ba là bộ phân loại sinh dùng phân phối Gauss cho mỗi lớp
- [ ] Ở thuật toán tối ưu
- [ ] Ở cách xử lý giá trị thiếu
> Naive Bayes Gauss dùng ma trận đường chéo riêng cho từng lớp; LDA dùng một ma trận đầy đủ chung; QDA dùng ma trận đầy đủ riêng cho từng lớp.

### Vì sao biên quyết định của LDA là tuyến tính còn của QDA là bậc hai?
- [ ] Vì LDA dùng hàm tuyến tính còn QDA dùng đa thức
- [x] Vì khi các lớp dùng chung $\Sigma$, số hạng bậc hai $x^\top\Sigma^{-1}x$ triệt tiêu khi lấy hiệu hai hàm phân biệt
- [ ] Vì LDA chỉ xử lý được hai lớp
- [ ] Vì QDA có nhiều tham số hơn
> Ở Mục 7.4, khớp hàm quyết định của LDA bằng hàm tuyến tính cho sai số $4{,}4\times10^{-15}$; với QDA sai số là 17,8.

### Vì sao Naive Bayes vẫn phân loại tốt dù giả thiết độc lập có điều kiện hiếm khi đúng?
- [x] Vì để chọn đúng nhãn chỉ cần thứ tự của các xác suất hậu nghiệm đúng, không cần giá trị của chúng đúng
- [ ] Vì giả thiết đó thường đúng
- [ ] Vì làm trơn Laplace sửa được sai lệch
- [ ] Vì nó chỉ dùng cho dữ liệu văn bản
> Hệ quả là xác suất do Naive Bayes đưa ra thường bị đẩy về gần 0 hoặc 1 quá mức, nên không nên dùng như xác suất đã hiệu chuẩn.

### Làm trơn Laplace giải quyết vấn đề gì trong Naive Bayes?
- [ ] Làm mô hình chạy nhanh hơn
- [x] Tránh xác suất bằng 0 cho giá trị chưa gặp trong một lớp, vì một thừa số bằng 0 làm cả tích bằng 0
- [ ] Làm các đặc trưng trở nên độc lập
- [ ] Chuẩn hoá đặc trưng về cùng thang đo
> Không làm trơn, một từ chưa từng xuất hiện trong thư rác sẽ làm mọi thư chứa từ đó có xác suất là rác bằng đúng 0. Làm trơn Laplace tương ứng với ước lượng MAP khi tiên nghiệm là Dirichlet.

## Chương 8

### Trên dữ liệu có 0,98% lớp dương, bộ phân loại luôn đoán lớp âm có độ chính xác bao nhiêu?
- [ ] Khoảng 50%
- [ ] Khoảng 90%
- [x] 0,9902, cao hơn một bộ phân loại phát hiện được hơn nửa số ca dương (0,9900)
- [ ] Không tính được
> Chọn mô hình theo độ chính xác trên dữ liệu mất cân bằng dẫn tới chọn mô hình vô dụng. Nên dùng precision, recall, F1 hoặc PR-AUC.

### Vì sao ROC-AUC có thể rất cao trên dữ liệu mất cân bằng dù phần lớn cảnh báo là sai?
- [ ] Vì nó chỉ tính trên lớp dương
- [x] Vì mẫu số của FPR là toàn bộ lớp âm, rất đông, nên thêm nhiều dương giả cũng chỉ làm FPR tăng rất ít
- [ ] Vì nó dùng thang logarit
- [ ] Vì nó bỏ qua các ngưỡng cực đoan
> Ở Mục 8.3, cùng mô hình và dữ liệu cho ROC-AUC 0,9715 nhưng PR-AUC chỉ 0,4931. Precision có mẫu số là số điểm được báo dương nên nhạy với dương giả.

### PR-AUC của bộ phân loại đoán ngẫu nhiên bằng bao nhiêu?
- [ ] Luôn bằng 0,5
- [x] Bằng tỉ lệ lớp dương trong dữ liệu, ở ví dụ của Chương 8 là 0,0100
- [ ] Bằng 0
- [ ] Phụ thuộc số điểm dữ liệu
> ROC-AUC của bộ đoán ngẫu nhiên luôn là 0,5. Vì vậy PR-AUC 0,4931 cần so với mốc 0,0100 chứ không so với 0,5.

### Lỗi nào làm kết quả đánh giá tốt lên một cách giả tạo?
- [x] Chuẩn hoá trên toàn bộ dữ liệu trước khi chia tập
- [ ] Dùng $k = 10$ thay vì $k = 5$ trong cross-validation
- [ ] Đặt hạt giống ngẫu nhiên cố định
- [ ] Dùng F1 thay vì độ chính xác
> Trung bình và độ lệch chuẩn tính trên cả dữ liệu chứa thông tin của tập kiểm tra. Mọi bước tiền xử lý có tham số chỉ được khớp trên tập huấn luyện.

### Với bài toán phân loại đơn nhãn nhiều lớp, micro-F1 bằng đại lượng nào?
- [ ] Macro-F1
- [x] Độ chính xác
- [ ] Recall của lớp đông nhất
- [ ] Trung bình precision của các lớp
> Mỗi dự đoán sai vừa là một dương giả của lớp được dự đoán vừa là một âm giả của lớp thật, nên tổng FP bằng tổng FN và micro-precision bằng micro-recall bằng độ chính xác.

## Chương 9

### Dấu hiệu nào cho thấy mô hình bị overfitting?
- [ ] Sai số huấn luyện và sai số xác thực đều lớn
- [x] Sai số huấn luyện nhỏ nhưng sai số xác thực lớn hơn nhiều
- [ ] Sai số xác thực nhỏ hơn sai số huấn luyện
- [ ] Mô hình huấn luyện quá lâu
> Sai số huấn luyện và xác thực cùng lớn là dấu hiệu underfitting. Overfitting tương ứng với phương sai cao: mô hình khớp cả nhiễu của tập huấn luyện.

### Hồi quy ridge co các hướng riêng như thế nào?
- [ ] Đều nhau ở mọi hướng
- [x] Theo hệ số $d_i^2/(d_i^2+\lambda)$, co mạnh nhất những hướng có giá trị suy biến nhỏ
- [ ] Chỉ co hướng có giá trị suy biến lớn nhất
- [ ] Không co, chỉ xoay trục
> Ở Mục 9.3 với $\lambda=10$, hướng có $d_i = 12{,}27$ giữ lại 0,938, hướng có $d_i = 6{,}17$ giữ lại 0,792. Hướng có giá trị suy biến nhỏ là hướng ước lượng kém tin cậy nhất, nên ridge xử lý được đa cộng tuyến.

### Khi các cột của $X$ trực chuẩn, lasso làm gì với nghiệm bình phương tối thiểu $z_j$?
- [ ] Nhân với $1/(1+\lambda)$
- [x] Trừ trị tuyệt đối đi $\lambda$ và cắt về 0 nếu kết quả âm (ngưỡng mềm)
- [ ] Giữ nguyên
- [ ] Làm tròn tới số nguyên gần nhất
> Nghiệm là $\operatorname{sign}(z_j)\max(|z_j| - \lambda, 0)$ với quy ước $\tfrac12\|y - Xw\|^2 + \lambda\|w\|_1$. Ridge thì nhân với $1/(1+\lambda)$ nên không bao giờ cho hệ số bằng 0.

### Lỗi nghiêm trọng nhất khi dùng cross-validation là gì?
- [ ] Chọn $k$ quá lớn
- [x] Dùng cùng một vòng cross-validation để vừa chọn siêu tham số vừa báo cáo hiệu năng cuối cùng
- [ ] Không dùng stratified k-fold
- [ ] Chạy trên máy nhiều nhân
> Cấu hình được chọn là cấu hình tốt nhất trên chính các phần dữ liệu đó, nên con số báo cáo lạc quan một cách có hệ thống. Cách đúng là cross-validation lồng nhau, hoặc giữ riêng một tập kiểm tra chỉ dùng một lần.

## Chương 10

### Bình phương sai số trong hồi quy tuyến tính có nguồn gốc từ đâu?
- [ ] Một quy ước toán học thuận tiện
- [x] Là âm log hợp lý khi nhiễu có phân phối Gauss
- [ ] Định lý giới hạn trung tâm
- [ ] Bất đẳng thức Cauchy–Schwarz
> Nếu nhiễu có đuôi dày hơn Gauss, giả thiết nhiễu Laplace cho ra hồi quy trị tuyệt đối, ít nhạy với điểm ngoại lai hơn. Đổi giả thiết về nhiễu là đổi hàm mất mát.

### Quan hệ giữa hồi quy ridge và ước lượng MAP là gì?
- [ ] Hai phương pháp tương tự nhau về trực giác
- [x] Ridge trùng với MAP khi tiên nghiệm là Gauss, với $\lambda = \sigma^2/\tau^2$
- [ ] MAP là trường hợp riêng của ridge
- [ ] Không có quan hệ
> Ở Mục 10.4, nghiệm ridge dạng đóng và nghiệm MAP tìm bằng BFGS, hai cách tính độc lập, lệch nhau $1{,}97\times10^{-8}$, trong phạm vi dung sai của thuật toán tối ưu.

### Tiên nghiệm càng chặt ($\tau$ càng nhỏ) thì $\lambda$ thay đổi thế nào?
- [x] $\lambda$ tăng, vì $\lambda = \sigma^2/\tau^2$
- [ ] $\lambda$ giảm
- [ ] $\lambda$ không đổi
- [ ] Phụ thuộc vào dữ liệu
> Tin chắc rằng hệ số gần 0 thì cần bằng chứng mạnh hơn từ dữ liệu mới kéo được hệ số ra xa, và đó đúng là ý nghĩa của regularization mạnh.

### Ước lượng MAP có phải là suy luận Bayes đầy đủ không?
- [ ] Có, đó là hai tên gọi của cùng một phương pháp
- [x] Không: MAP chỉ lấy đỉnh của phân phối hậu nghiệm, nên không cho biết mức độ không chắc chắn của tham số
- [ ] Có, nhưng chỉ khi tiên nghiệm là Gauss
- [ ] Không, vì MAP không dùng định lý Bayes
> Suy luận Bayes đầy đủ giữ toàn bộ phân phối hậu nghiệm và lấy trung bình dự đoán trên đó. Khi hậu nghiệm lệch hoặc có nhiều đỉnh, điểm đỉnh có thể không đại diện cho phân phối.

## Chương 11

### Vì sao tính lồi quan trọng trong tối ưu?
- [x] Vì với hàm lồi, mọi điểm có gradient bằng 0 là cực tiểu toàn cục, nên kết quả không phụ thuộc điểm khởi tạo
- [ ] Vì hàm lồi tính nhanh hơn
- [ ] Vì hàm lồi luôn khả vi
- [ ] Vì hàm lồi luôn có cực tiểu
> Ở Mục 11.6, gradient descent từ 21 điểm xuất phát cho một điểm dừng với hàm lồi, nhưng cho hai điểm dừng có giá trị $-2{,}87$ và $-1{,}65$ với hàm $x^4-3x^2+x/2$. Lưu ý hàm lồi không nhất thiết có cực tiểu, ví dụ $e^x$.

### Phép toán nào không bảo toàn tính lồi?
- [ ] Cộng hai hàm lồi
- [ ] Lấy max của hai hàm lồi
- [x] Lấy min của hai hàm lồi
- [ ] Hợp một hàm lồi với một ánh xạ affine
> Phép max giải thích vì sao mất mát hinge $\max(0,1-m)$ lồi. Min của hai parabol có đỉnh khác nhau có thể có hai đáy, nên không lồi.

### Kiểm tra tính lồi bằng cách thử bất đẳng thức định nghĩa trên nhiều bộ điểm ngẫu nhiên cho phép kết luận gì?
- [ ] Chứng minh được hàm lồi nếu không có vi phạm
- [x] Chỉ bác bỏ được: một phản ví dụ là đủ để kết luận hàm không lồi, còn không có vi phạm thì chưa chứng minh được gì
- [ ] Không kết luận được gì
- [ ] Chứng minh được nếu thử đủ 10 000 bộ
> Không có vi phạm trên 20 000 bộ không loại trừ một vùng lõm hẹp chưa được lấy mẫu. Với hàm mất mát tự viết, đây vẫn là cách rẻ để phát hiện lỗi.

## Chương 12

### Điều kiện bù trong KKT phát biểu điều gì?
- [ ] Mọi nhân tử Lagrange đều dương
- [x] $\alpha_i f_i(x^*) = 0$: nếu ràng buộc không chặt tại nghiệm thì nhân tử của nó bằng 0
- [ ] Gradient của hàm mục tiêu bằng 0
- [ ] Bài toán gốc và bài toán đối ngẫu có cùng số biến
> Điều kiện này sinh ra khái niệm vector hỗ trợ trong SVM: chỉ những điểm có ràng buộc chặt mới có $\alpha > 0$ và mới ảnh hưởng tới nghiệm.

### Hàm đối ngẫu $g(\alpha,\nu)$ có tính chất gì?
- [x] Luôn lõm, kể cả khi bài toán gốc không lồi
- [ ] Luôn lồi
- [ ] Luôn bằng giá trị tối ưu của bài toán gốc
- [ ] Chỉ xác định khi bài toán gốc lồi
> Với mỗi $x$, hàm Lagrange là hàm affine theo $(\alpha,\nu)$, và cận dưới đúng của một họ hàm affine là hàm lõm. Mỗi giá trị của $g$ là một chặn dưới của giá trị tối ưu.

### Với bài toán cực tiểu $x^2$ với ràng buộc $x \ge 1$, nhân tử Lagrange tối ưu bằng bao nhiêu?
- [ ] 0
- [ ] 1
- [x] 2
- [ ] 4
> Điều kiện dừng $2x - \alpha = 0$ và ràng buộc chặt $x^* = 1$ cho $\alpha^* = 2$. Hàm đối ngẫu $g(\alpha) = \alpha - \alpha^2/4$ đạt cực đại 1 tại $\alpha = 2$, bằng giá trị tối ưu của bài toán gốc.

### Vì sao SVM được giải qua bài toán đối ngẫu?
- [ ] Vì bài toán gốc không có nghiệm
- [x] Vì bài toán đối ngẫu chỉ phụ thuộc dữ liệu qua tích vô hướng $x_i^\top x_j$, nên thay bằng kernel là làm việc được trong không gian đặc trưng phi tuyến
- [ ] Vì bài toán đối ngẫu luôn nhanh hơn
- [ ] Vì nó cho nghiệm chính xác hơn
> Hai lý do phụ: bài toán đối ngẫu có $n$ biến thay vì $d$ biến, và giá trị đối ngẫu cho chặn dưới để biết nghiệm hiện tại còn cách tối ưu bao xa.

## Chương 13

### SVM chọn siêu phẳng nào trong các siêu phẳng tách đúng dữ liệu?
- [ ] Siêu phẳng đầu tiên tìm được
- [x] Siêu phẳng có lề lớn nhất
- [ ] Siêu phẳng đi qua trọng tâm dữ liệu
- [ ] Siêu phẳng có ít tham số nhất
> Cực đại độ rộng lề $2/\|w\|$ tương đương cực tiểu $\tfrac12\|w\|^2$ với các ràng buộc $y_i(w^\top x_i + b) \ge 1$, một bài toán quy hoạch toàn phương lồi.

### Trong thí nghiệm ở Mục 13.3, có bao nhiêu vector hỗ trợ?
- [ ] Toàn bộ 120 điểm
- [x] 3 trên 120 điểm; bỏ 117 điểm còn lại thì nghiệm không đổi
- [ ] Một nửa số điểm
- [ ] Phụ thuộc vào $C$
> Tính thưa này là hệ quả của điều kiện bù: điểm nằm ngoài lề có $\alpha_i = 0$ và không đóng góp vào $w = \sum_i\alpha_i y_i x_i$.

### Điều kiện KKT chia các điểm trong SVM lề mềm thành những nhóm nào?
- [x] Ba nhóm: $\alpha=0$ nằm ngoài lề, $0<\alpha<C$ nằm đúng trên lề, $\alpha=C$ vi phạm lề
- [ ] Hai nhóm: vector hỗ trợ và không phải vector hỗ trợ
- [ ] Bốn nhóm
- [ ] Không chia nhóm
> Ở Mục 13.4, với $C = 0{,}003$ cả 114 vector hỗ trợ đều thuộc nhóm vi phạm lề và không điểm nào nằm đúng trên lề.

### SVM lề mềm tương đương với bài toán nào?
- [ ] Hồi quy logistic không có regularization
- [x] Cực tiểu tổng mất mát hinge cộng regularization $\ell_2$
- [ ] Perceptron với tốc độ học nhỏ
- [ ] Bình phương tối thiểu có ràng buộc
> Tại nghiệm, biến bù $\xi_i = \max(0, 1 - y_i f(x_i))$, đúng bằng mất mát hinge. Tham số $C$ đóng vai trò nghịch đảo của hệ số regularization.

### Thủ thuật kernel cho phép làm gì?
- [ ] Giảm số chiều của dữ liệu
- [x] Tính tích vô hướng $\varphi(x)^\top\varphi(x')$ trong không gian đặc trưng mà không cần tính $\varphi(x)$
- [ ] Biến hàm mất mát không lồi thành lồi
- [ ] Tăng tốc độ huấn luyện
> Với kernel RBF, không gian đặc trưng có vô hạn chiều nhưng mỗi lần tính kernel chỉ tốn $O(d)$. Ở Mục 13.5, trên dữ liệu hai đường tròn đồng tâm, kernel tuyến tính đạt 0,615, còn kernel đa thức bậc 2 và RBF đều đạt 1,000.

### Vì sao SVM với kernel ít được dùng cho dữ liệu rất lớn?
- [ ] Vì kém chính xác hơn mạng nơ-ron
- [x] Vì huấn luyện tốn khoảng $O(n^2)$ tới $O(n^3)$, và dự đoán phải tính kernel với mọi vector hỗ trợ, số lượng thường tăng theo $n$
- [ ] Vì không xử lý được nhiều lớp
- [ ] Vì bắt buộc cần GPU
> Mỗi epoch của SGD trên mạng nơ-ron chỉ tốn chi phí tuyến tính theo $n$. SVM vẫn phù hợp khi $n$ vừa phải và $d$ lớn, như văn bản biểu diễn bằng TF-IDF.

## Chương 14

### Định lý Eckart–Young phát biểu điều gì?
- [x] Xấp xỉ bằng $k$ thành phần chính đầu tiên là xấp xỉ hạng $k$ tốt nhất theo chuẩn Frobenius, và sai số tái tạo bằng $(n-1)$ nhân tổng các trị riêng bị bỏ
- [ ] PCA luôn giữ được 95% phương sai
- [ ] Số thành phần tối ưu bằng căn bậc hai của số chiều
- [ ] PCA và LDA cho cùng kết quả khi dữ liệu là Gauss
> Ở Mục 14.3, đẳng thức khớp tới cỡ $10^{-13}$ với mọi giá trị $k$.

### Khi nào PCA không phù hợp làm bước tiền xử lý cho phân loại?
- [ ] Khi dữ liệu có nhiều chiều
- [x] Khi hướng có phương sai lớn nhất không chứa thông tin phân biệt các lớp, vì PCA không dùng nhãn
- [ ] Khi số điểm nhỏ hơn số chiều
- [ ] Khi dữ liệu không có phân phối chuẩn
> Ở Mục 14.4, thành phần chính thứ nhất giữ 96,1% phương sai nhưng cho AUC 0,502, trong khi hướng của LDA cho AUC 0,9987.

### LDA cho tối đa bao nhiêu chiều?
- [ ] $d$
- [x] $K - 1$, vì ma trận tán xạ giữa các lớp có hạng không quá $K-1$
- [ ] $\min(n, d)$
- [ ] Không giới hạn
> Với hai lớp, LDA cho đúng một chiều, theo hướng $S_W^{-1}(\mu_1 - \mu_0)$.

### Vì sao thường phải chuẩn hoá đặc trưng trước PCA?
- [x] Vì PCA phụ thuộc đơn vị đo: đổi một đặc trưng từ mét sang milimét làm phương sai của nó tăng $10^6$ lần và thành phần chính thứ nhất gần như trùng với đặc trưng đó
- [ ] Vì thuật toán không hội tụ nếu chưa chuẩn hoá
- [ ] Vì trị riêng phải dương
- [ ] Vì nếu không thì ma trận hiệp phương sai suy biến
> Chuẩn hoá mỗi đặc trưng về phương sai 1 tương đương với làm PCA trên ma trận tương quan. Ngoại lệ là khi mọi đặc trưng đã cùng đơn vị, như cường độ điểm ảnh.

## Chương 15

### Thuật toán K-means (Lloyd) bảo đảm điều gì?
- [ ] Tìm được cách chia tối ưu toàn cục
- [x] Dừng sau hữu hạn bước ở một cực tiểu địa phương; bài toán tìm nghiệm tối ưu là NP-khó
- [ ] Hội tụ trong nhiều nhất $k$ vòng lặp
- [ ] Không bảo đảm dừng
> Cả bước gán và bước cập nhật đều không làm tăng hàm mục tiêu, và số cách chia là hữu hạn, nên thuật toán chắc chắn dừng; điểm dừng phụ thuộc vào khởi tạo.

### Trong thí nghiệm ở Mục 15.2, k-means++ cải thiện được bao nhiêu?
- [ ] Loại bỏ hoàn toàn trường hợp kẹt ở nghiệm tồi
- [x] Giảm tỉ lệ kẹt ở nghiệm tồi từ 71,5% xuống 46,5%
- [ ] Không cải thiện đáng kể
- [ ] Làm chậm thuật toán mà không cải thiện chất lượng
> Vẫn cần chạy nhiều lần và giữ kết quả có inertia nhỏ nhất. Trong scikit-learn từ phiên bản 1.4, `KMeans` mặc định `n_init='auto'`, tức chỉ chạy một lần với k-means++, nên cần đặt `n_init` tường minh.

### Các giả định ngầm của K-means là gì?
- [x] Cụm lồi và gần tròn, kích thước tương đương, mọi chiều cùng thang đo
- [ ] Dữ liệu có phân phối chuẩn, độc lập và cùng phân phối
- [ ] Số cụm đã biết, không có nhiễu, không có điểm ngoại lai
- [ ] Dữ liệu tách được tuyến tính
> Vì mỗi điểm được gán cho tâm gần nhất theo khoảng cách Euclid, biên giữa hai cụm luôn là một siêu phẳng. Ở Mục 15.3: ba cụm tròn đạt độ chính xác 100%, hai hình lưỡi liềm 75%, hai dải dẹt 53%.

### Vì sao không chọn số cụm $k$ bằng cách cực tiểu inertia?
- [x] Vì inertia luôn giảm khi $k$ tăng và bằng 0 khi $k = n$
- [ ] Vì inertia không tính được với $k$ lớn
- [ ] Vì inertia không liên quan tới chất lượng cụm
- [ ] Vì thuật toán không hội tụ với $k$ lớn
> Phương pháp elbow tìm chỗ inertia ngừng giảm nhanh: trên dữ liệu 8 cụm, mức giảm là 27,7% khi lên $k=8$ và 5,4% khi lên $k=9$. Trên dữ liệu thật, có thể dùng thêm hệ số silhouette hoặc gap statistic.

## Chương 16

### Phân rã ma trận cần bao nhiêu đánh giá mỗi người dùng?
- [ ] Chỉ cần nhiều hơn 1 đánh giá
- [x] Lớn hơn hạng $k$ nhiều lần; trong thí nghiệm với $k = 4$, ngưỡng nằm giữa khoảng 10 và 20 đánh giá mỗi người
- [ ] Càng nhiều càng tốt, không có ngưỡng
- [ ] Bằng số sản phẩm trong danh mục
> Ở Mục 16.3, với 10,1 đánh giá mỗi người, RMSE là 3,54, tệ hơn đoán mọi ô bằng 0 (2,08); với 19,8 đánh giá mỗi người, RMSE giảm xuống 0,30.

### Chọn hạng $k$ lớn hơn hạng thật thì điều gì xảy ra?
- [ ] Mô hình luôn tốt hơn vì linh hoạt hơn
- [x] Sai số trên ô đã quan sát tiếp tục giảm nhưng sai số trên ô chưa quan sát tăng lên, tức overfitting
- [ ] Thuật toán không hội tụ
- [ ] Không có ảnh hưởng
> Ở Mục 16.4, RMSE trên ô chưa quan sát nhỏ nhất tại $k=4$ (0,1448) và tăng lên 0,6291 tại $k=20$, trong khi RMSE trên ô đã quan sát giảm đều xuống 0,1903.

### Vì sao phân rã ma trận không dự đoán được cho người dùng mới?
- [ ] Vì thuật toán phải khởi tạo lại
- [x] Vì với người dùng chưa có đánh giá, hàm mất mát chỉ còn thành phần regularization $\lambda\lVert p_u\rVert^2$, có cực tiểu tại $p_u = 0$
- [ ] Vì ma trận đánh giá trở nên suy biến
- [ ] Vì số chiều không khớp
> Ở Mục 16.5, chuẩn trung bình của vector người dùng mới là 0,154 so với 2,084 của người dùng cũ, và RMSE cho họ là 2,24, tệ hơn đoán mọi ô bằng 0 (2,08). Hệ thống thực tế cần kết hợp thêm lọc dựa trên nội dung.
