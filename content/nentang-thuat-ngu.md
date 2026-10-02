# Từ điển thuật ngữ — Nền tảng học máy

Cú pháp: `## Nhóm`, rồi `### Tiếng Việt | English` và phần định nghĩa bên dưới.
Mỗi định nghĩa khớp với cách dùng trong giáo trình. Thuật ngữ nào giáo trình giữ nguyên tiếng Anh thì tên tiếng Anh đứng trước, tên tiếng Việt (nếu có) đặt trong ngoặc.

## Khái niệm chung

### Học có giám sát | supervised learning
Học từ các cặp $(x, y)$ trong đó mỗi đầu vào có nhãn. Gồm hồi quy (nhãn là số thực) và phân loại (nhãn là một lớp).

### Học không giám sát | unsupervised learning
Học từ dữ liệu $x$ không có nhãn, nhằm tìm cấu trúc như cụm, hướng chứa nhiều thông tin hay mật độ phân phối.

### Hàm mất mát | loss function
Hàm đo mức sai của một dự đoán so với nhãn thật. Cùng với họ mô hình và thuật toán tối ưu, nó là một trong ba thành phần của thuật toán học có giám sát (Mục 1.1).

### Tổng quát hoá | generalization
Khả năng dự đoán tốt trên dữ liệu chưa gặp khi huấn luyện. Đây là mục tiêu thật của học máy; cực tiểu mất mát huấn luyện chỉ là phương tiện.

### Mô hình sinh | generative model
Mô hình hoá $p(x \mid y)$ và $p(y)$, rồi dùng định lý Bayes để tính $p(y \mid x)$. Naive Bayes, LDA và QDA thuộc loại này.

### Mô hình phân biệt | discriminative model
Mô hình hoá trực tiếp $p(y \mid x)$ hoặc biên quyết định. Hồi quy logistic, SVM và cây quyết định thuộc loại này.

### Phương pháp phi tham số | non-parametric method
Phương pháp có số tham số tăng theo lượng dữ liệu. k-NN là ví dụ điển hình: nó lưu lại toàn bộ dữ liệu huấn luyện.

### Siêu tham số | hyperparameter
Tham số không học được từ dữ liệu huấn luyện mà phải chọn bằng tập xác thực hoặc cross-validation, như $\lambda$ của ridge, $C$ của SVM, $k$ của k-NN và K-means.

### Tập huấn luyện, tập xác thực, tập kiểm tra | training, validation, test set
Tập huấn luyện dùng để học tham số; tập xác thực dùng để chọn siêu tham số và so sánh mô hình; tập kiểm tra chỉ dùng một lần ở cuối để báo cáo kết quả.

## Đại số tuyến tính

### Siêu phẳng | hyperplane
Tập $\{x : w^\top x + b = 0\}$. Vector $w$ vuông góc với siêu phẳng (vector pháp tuyến), và khoảng cách từ điểm $x_0$ tới siêu phẳng là $|w^\top x_0 + b| / \|w\|$.

### Tích vô hướng | inner product, dot product
$w^\top x = \sum_j w_j x_j = \|w\|\,\|x\|\cos\vartheta$. Chia cho tích hai độ dài được độ tương đồng cosine.

### Chuẩn | norm
Cách đo độ dài của vector. Chuẩn $\ell_1$ là tổng trị tuyệt đối các thành phần, chuẩn $\ell_2$ là căn bậc hai của tổng bình phương, chuẩn $\ell_\infty$ là thành phần có trị tuyệt đối lớn nhất.

### Hạng | rank
Số cột độc lập tuyến tính lớn nhất của ma trận. Ma trận $X$ đủ hạng cột khi và chỉ khi $X^\top X$ khả nghịch.

### Trị riêng, vector riêng | eigenvalue, eigenvector
Số $\lambda$ và vector khác không $v$ thoả $Av = \lambda v$: phép nhân với $A$ chỉ kéo dãn $v$ mà không đổi hướng. Mọi ma trận đối xứng thực có một hệ vector riêng trực chuẩn (định lý phổ).

### Ma trận xác định dương | positive definite matrix
Ma trận đối xứng thoả $z^\top A z > 0$ với mọi $z \neq 0$, tương đương mọi trị riêng dương. Nếu $A$ nửa xác định dương thì $A + \lambda I$ xác định dương với mọi $\lambda > 0$, nên hồi quy ridge luôn có nghiệm duy nhất.

### Số điều kiện | condition number
Với ma trận đối xứng xác định dương, $\kappa(A) = \lambda_{\max}/\lambda_{\min}$. Số vòng lặp của gradient descent trên hàm bậc hai tỉ lệ với $\kappa$. Lập ma trận $X^\top X$ làm bình phương số điều kiện: $\kappa(X^\top X) = \kappa(X)^2$.

### Phân tích giá trị suy biến (SVD) | singular value decomposition
Viết $X = UDV^\top$ với $U, V$ trực giao và $D$ chứa các giá trị suy biến $d_1 \ge d_2 \ge \dots \ge 0$. Trị riêng của $X^\top X$ là $d_i^2$; giữ $k$ giá trị suy biến lớn nhất cho xấp xỉ hạng $k$ tốt nhất.

### Giả nghịch đảo Moore–Penrose | pseudoinverse
Ký hiệu $X^{+}$. Khi phương trình chuẩn có vô số nghiệm, $X^{+}y$ là nghiệm có chuẩn nhỏ nhất, trùng với giới hạn của nghiệm ridge khi $\lambda \to 0^{+}$.

### Gradient, ma trận Hessian | gradient, Hessian matrix
Gradient là vector các đạo hàm riêng cấp một, chỉ hướng hàm tăng nhanh nhất. Ma trận Hessian chứa các đạo hàm riêng cấp hai; hàm khả vi hai lần là lồi khi và chỉ khi Hessian nửa xác định dương ở mọi điểm.

## Xác suất và thống kê

### Kỳ vọng, phương sai | expectation, variance
Kỳ vọng là giá trị trung bình khi lặp lại phép thử rất nhiều lần; phương sai là trung bình của bình phương độ lệch khỏi kỳ vọng. Kỳ vọng của tổng luôn bằng tổng các kỳ vọng; phương sai của tổng chỉ bằng tổng các phương sai khi các biến không tương quan.

### Hiệp phương sai, hệ số tương quan | covariance, correlation coefficient
$\operatorname{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}X)(Y - \mathbb{E}Y)]$. Chia cho tích hai độ lệch chuẩn được hệ số tương quan $\rho \in [-1, 1]$.

### Ma trận hiệp phương sai | covariance matrix
$\Sigma = \mathbb{E}[(x-\mu)(x-\mu)^\top]$, đối xứng và nửa xác định dương. Giả thiết của Naive Bayes Gauss tương đương với việc ép ma trận hiệp phương sai của mỗi lớp thành ma trận đường chéo.

### Phân phối Gauss | Gaussian (normal) distribution
Phân phối có mật độ tỉ lệ với $\exp(-\tfrac12 (x-\mu)^\top\Sigma^{-1}(x-\mu))$. Logarit của mật độ là hàm bậc hai, nên giả thiết nhiễu Gauss dẫn tới bình phương tối thiểu.

### Khoảng cách Mahalanobis | Mahalanobis distance
$\sqrt{(x-\mu)^\top\Sigma^{-1}(x-\mu)}$: khoảng cách Euclid sau khi quy mỗi hướng về đơn vị độ lệch chuẩn của dữ liệu theo hướng đó.

### Định lý Bayes | Bayes' theorem
$p(A \mid B) = p(B \mid A)\,p(A)/p(B)$. Cho phép tính xác suất hậu nghiệm từ hàm hợp lý và phân phối tiên nghiệm.

### Hàm hợp lý (likelihood) | likelihood
$p(\mathcal{D}\mid\theta)$ xem như hàm của tham số $\theta$ khi dữ liệu cố định. Hàm hợp lý không phải là phân phối xác suất của $\theta$.

### Ước lượng hợp lý cực đại (MLE) | maximum likelihood estimation
Tham số làm hàm hợp lý lớn nhất. Với nhiễu Gauss cho ra bình phương tối thiểu; với nhãn Bernoulli cho ra cross-entropy.

### Phân phối tiên nghiệm, phân phối hậu nghiệm | prior, posterior
Tiên nghiệm $p(\theta)$ mô tả hiểu biết về tham số trước khi thấy dữ liệu; hậu nghiệm $p(\theta\mid\mathcal{D})$ là hiểu biết sau khi thấy dữ liệu. Âm logarit của tiên nghiệm đóng vai trò thành phần regularization.

### Ước lượng hậu nghiệm cực đại (MAP) | maximum a posteriori estimation
Tham số làm hậu nghiệm lớn nhất. Với tiên nghiệm Gauss cho ra ridge, với tiên nghiệm Laplace cho ra lasso. MAP chỉ lấy đỉnh của hậu nghiệm nên không cho biết mức độ không chắc chắn của tham số.

### Làm trơn Laplace | Laplace smoothing, additive smoothing
Cộng $\alpha$ vào mọi ô đếm để không xác suất nào bằng 0. Tương ứng với ước lượng MAP khi tiên nghiệm là phân phối Dirichlet.

### Độ chệch của ước lượng | bias of an estimator
$\mathbb{E}[\hat\theta] - \theta$, với kỳ vọng lấy trên mọi tập dữ liệu có thể rút ra. Ước lượng có độ chệch bằng 0 gọi là ước lượng không chệch. Sai số bình phương trung bình bằng bình phương độ chệch cộng phương sai.

## Hồi quy và tối ưu

### Bình phương tối thiểu | ordinary least squares, OLS
Hồi quy tuyến tính cực tiểu tổng bình phương sai số $\|y - Xw\|^2$. Về hình học, đây là phép chiếu vuông góc của $y$ lên không gian cột của $X$.

### Phương trình chuẩn | normal equations
$X^\top X w = X^\top y$, thu được khi cho gradient của bình phương sai số bằng 0. Nó nói rằng vector phần dư vuông góc với mọi cột của $X$.

### Đa cộng tuyến | multicollinearity
Tình trạng các cột của $X$ gần phụ thuộc tuyến tính. Làm hệ số dao động mạnh nhưng hầu như không ảnh hưởng tới dự đoán; mức tăng độ lệch chuẩn của hệ số được đo bằng $\sqrt{\text{VIF}}$.

### Hệ số phóng đại phương sai | variance inflation factor, VIF
$\text{VIF}_j = 1/(1 - R_j^2)$, với $R_j^2$ là hệ số xác định khi hồi quy đặc trưng $j$ theo các đặc trưng còn lại. VIF lớn hơn 10 thường được coi là dấu hiệu đa cộng tuyến đáng kể.

### Gradient descent | gradient descent
Thuật toán lặp $\theta_{t+1} = \theta_t - \eta\nabla_\theta L(\theta_t)$. Hướng ngược gradient là hướng giảm nhanh nhất trong một lân cận nhỏ.

### Tốc độ học | learning rate
Hệ số $\eta$ nhân với gradient ở mỗi bước. Với hàm bậc hai, gradient descent hội tụ khi và chỉ khi $0 < \eta < 2/\lambda_{\max}$.

### Momentum | momentum, heavy ball method
Cộng thêm một phần của bước trước vào bước hiện tại. Với hàm bậc hai và tham số tối ưu, số vòng lặp giảm từ bậc $\kappa$ xuống bậc $\sqrt\kappa$.

### Stochastic gradient descent (SGD) | stochastic gradient descent
Gradient descent dùng gradient tính trên một điểm hoặc một mini-batch chọn ngẫu nhiên. Gradient này là ước lượng không chệch của gradient đầy đủ, có phương sai giảm theo $1/B$.

### Mini-batch | mini-batch
Một nhóm $B$ điểm dữ liệu dùng để tính gradient ở mỗi bước. Cách làm phổ biến nhất vì tận dụng được tính toán song song trên GPU.

### Kiểm tra gradient | gradient checking
So sánh gradient tính bằng công thức với gradient xấp xỉ bằng sai phân trung tâm, dùng $\varepsilon$ khoảng $10^{-5}$ tới $10^{-6}$. Bước sai phân tối ưu là $\varepsilon^* \sim u^{1/3}$; $\varepsilon$ nhỏ hơn làm sai số lớn lên.

### Sai số cắt cụt, sai số làm tròn | truncation error, round-off error
Hai nguồn sai số của sai phân hữu hạn: sai số cắt cụt giảm khi $\varepsilon$ nhỏ đi (tỉ lệ $\varepsilon^2$ với sai phân trung tâm), sai số làm tròn tăng khi $\varepsilon$ nhỏ đi (tỉ lệ $u/\varepsilon$).

## Phân loại

### Lề của một điểm | margin (of an example)
$m_i = y_i(w^\top x_i + b)$ với $y_i \in \{-1,+1\}$. Dương nghĩa là phân loại đúng; trị tuyệt đối tỉ lệ với khoảng cách tới biên quyết định.

### Lề của bộ phân loại | margin (of a classifier)
Khoảng cách giữa hai siêu phẳng $w^\top x + b = \pm1$, bằng $2/\|w\|$. SVM chọn siêu phẳng có lề lớn nhất.

### Mất mát 0–1 | zero-one loss
$\mathbb{1}[m \le 0]$, đếm số điểm bị phân loại sai. Không tối ưu trực tiếp bằng gradient được vì đạo hàm bằng 0 hầu khắp nơi.

### Hàm mất mát thay thế | surrogate loss
Hàm lồi dùng thay cho mất mát 0–1 để có thể tối ưu bằng gradient. Hinge và logistic (theo logarit cơ số 2) còn là chặn trên của mất mát 0–1; mất mát perceptron là hàm thay thế lồi nhưng không phải chặn trên.

### Mất mát hinge | hinge loss
$\max(0, 1-m)$. Bằng 0 khi $m \ge 1$, nên một điểm chỉ có mất mát bằng 0 khi được phân loại đúng và nằm ngoài lề. SVM lề mềm tương đương với mất mát hinge cộng regularization $\ell_2$.

### Cross-entropy | cross-entropy loss
$-[y\log p + (1-y)\log(1-p)]$ cho hai lớp. Là âm log hợp lý khi nhãn có phân phối Bernoulli; cực tiểu cross-entropy tương đương cực tiểu phân kỳ KL từ phân phối dữ liệu tới phân phối của mô hình.

### Perceptron | perceptron
Thuật toán phân loại tuyến tính cập nhật $w \leftarrow w + y_i x_i$ mỗi khi gặp điểm bị phân loại sai. Theo định lý Novikoff, số lần cập nhật không quá $(R/\gamma)^2$ khi dữ liệu tách được; khi không tách được, thuật toán không dừng.

### Hàm sigmoid | sigmoid function
$\sigma(z) = 1/(1+e^{-z})$, biến một số thực thành một số trong khoảng $(0, 1)$. Đạo hàm $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.

### Hồi quy logistic | logistic regression
Thuật toán phân loại hai lớp với $p(y=1\mid x) = \sigma(w^\top x)$, huấn luyện bằng cross-entropy. Tên "hồi quy" đến từ việc mô hình hồi quy log tỉ lệ cược theo $x$.

### Hồi quy softmax | softmax regression
Mở rộng hồi quy logistic cho $K$ lớp bằng hàm softmax. Chỉ hiệu giữa các vector trọng số được xác định; với $K=2$ nó trùng với hồi quy logistic.

### Tách được tuyến tính | linearly separable
Tồn tại siêu phẳng phân loại đúng mọi điểm. Khi đó hồi quy logistic không có regularization không có nghiệm hữu hạn: $\|w\| \to \infty$.

### k láng giềng gần nhất (k-NN) | k-nearest neighbors
Phân loại một điểm theo lớp chiếm đa số trong $k$ điểm huấn luyện gần nó nhất. Cần chuẩn hoá đặc trưng; hoạt động kém ở số chiều cao.

### Naive Bayes | naive Bayes
Bộ phân loại sinh giả định các đặc trưng độc lập có điều kiện khi đã biết lớp. Thường phân loại tốt nhưng xác suất đưa ra kém hiệu chuẩn.

### LDA, QDA | linear / quadratic discriminant analysis
Các bộ phân loại dùng phân phối Gauss cho mỗi lớp. LDA dùng chung một ma trận hiệp phương sai và có biên tuyến tính; QDA dùng ma trận riêng cho mỗi lớp và có biên bậc hai.

### Hiệu chuẩn | calibration
Mức độ xác suất do mô hình đưa ra khớp với tần suất thật. Kiểm tra bằng biểu đồ độ tin cậy; hiệu chỉnh bằng Platt scaling hoặc hồi quy isotonic.

## Đánh giá mô hình

### Độ chính xác (accuracy) | accuracy
Tỉ lệ điểm được phân loại đúng. Dễ gây hiểu lầm trên dữ liệu mất cân bằng: bộ phân loại luôn đoán lớp đông nhất cũng có độ chính xác cao.

### Ma trận nhầm lẫn | confusion matrix
Bảng bốn ô TP, FP, FN, TN. Mọi thước đo phân loại hai lớp đều tính từ bốn ô này.

### Precision | precision
$TP/(TP+FP)$: trong những trường hợp mô hình báo là dương, bao nhiêu phần đúng là dương.

### Recall | recall, sensitivity, true positive rate
$TP/(TP+FN)$: trong những trường hợp thật sự dương, mô hình phát hiện được bao nhiêu phần.

### Điểm F1 | F1 score
Trung bình điều hoà của precision và recall. Chỉ cao khi cả hai cùng cao.

### Điểm F-beta | F-beta score
$F_\beta = (1+\beta^2)PR/(\beta^2 P + R)$. Với $\beta > 1$, recall được coi trọng hơn precision.

### ROC-AUC | area under the ROC curve
Diện tích dưới đường cong TPR theo FPR; bằng xác suất một điểm dương ngẫu nhiên có điểm số cao hơn một điểm âm ngẫu nhiên. Bộ phân loại ngẫu nhiên cho 0,5. Trên dữ liệu mất cân bằng có thể cao trong khi phần lớn cảnh báo dương là sai.

### PR-AUC | area under the precision–recall curve, average precision
Diện tích dưới đường cong precision theo recall. Bộ phân loại ngẫu nhiên cho giá trị bằng tỉ lệ lớp dương. Phù hợp hơn ROC-AUC khi lớp dương hiếm.

### Macro, micro, weighted | macro, micro, weighted averaging
Ba cách gộp thước đo của bài toán nhiều lớp. Với phân loại đơn nhãn, micro-F1 bằng độ chính xác.

### Rò rỉ dữ liệu | data leakage
Thông tin không có ở thời điểm dự đoán lọt vào quá trình huấn luyện, làm kết quả đánh giá tốt hơn thực tế. Ba dạng hay gặp: chuẩn hoá trước khi chia dữ liệu, chia ngẫu nhiên dữ liệu có thời gian, chia ngẫu nhiên dữ liệu có nhóm.

## Overfitting và regularization

### Overfitting (quá khớp) | overfitting
Mô hình khớp tốt dữ liệu huấn luyện nhưng dự đoán kém trên dữ liệu mới; tương ứng với phương sai cao.

### Underfitting (chưa khớp) | underfitting
Mô hình quá đơn giản, sai số lớn trên cả dữ liệu huấn luyện lẫn dữ liệu mới; tương ứng với độ chệch cao.

### Regularization (điều chuẩn) | regularization
Thêm vào hàm mất mát một thành phần phạt độ phức tạp của mô hình, thường là chuẩn của vector trọng số. Tương ứng với âm logarit của một phân phối tiên nghiệm.

### Hồi quy ridge | ridge regression, Tikhonov regularization
Bình phương tối thiểu cộng $\lambda\|w\|_2^2$. Luôn có nghiệm duy nhất; co thành phần theo hướng riêng thứ $i$ với hệ số $d_i^2/(d_i^2+\lambda)$, tức co mạnh nhất những hướng dữ liệu cung cấp ít thông tin.

### Số bậc tự do hiệu dụng | effective degrees of freedom
Với ridge, $\operatorname{df}(\lambda) = \sum_i d_i^2/(d_i^2+\lambda)$. Bằng số đặc trưng khi $\lambda = 0$ và giảm dần về 0 khi $\lambda$ tăng.

### Lasso | lasso
Bình phương tối thiểu cộng $\lambda\|w\|_1$. Cho nhiều hệ số bằng đúng 0 nên đồng thời chọn đặc trưng. Với một nhóm đặc trưng tương quan cao, thường chỉ giữ một đặc trưng đại diện.

### Ngưỡng mềm | soft thresholding
Phép toán $\operatorname{sign}(z)\max(|z| - \lambda, 0)$: trừ trị tuyệt đối đi $\lambda$ và cắt về 0 nếu âm. Là nghiệm của lasso khi các cột của $X$ trực chuẩn.

### Elastic net | elastic net
Kết hợp phạt $\ell_1$ và $\ell_2$: vừa cho nghiệm thưa như lasso vừa ổn định với nhóm đặc trưng tương quan như ridge.

### Cross-validation | cross-validation
Chia dữ liệu thành $k$ phần, lần lượt dùng một phần để đánh giá và $k-1$ phần để huấn luyện, rồi lấy trung bình. Không dùng cùng một vòng cross-validation để vừa chọn siêu tham số vừa báo cáo kết quả; khi cần cả hai, dùng cross-validation lồng nhau.

## Tối ưu lồi

### Tập lồi | convex set
Tập chứa trọn đoạn thẳng nối hai điểm bất kỳ của nó. Giao của các tập lồi là tập lồi; hợp thì thường không.

### Hàm lồi | convex function
Hàm thoả $f(tx + (1-t)y) \le tf(x) + (1-t)f(y)$: dây cung nằm trên đồ thị. Với hàm lồi khả vi, mọi điểm có gradient bằng 0 là cực tiểu toàn cục.

### Bài toán tối ưu lồi | convex optimization problem
Bài toán cực tiểu một hàm lồi với các ràng buộc bất đẳng thức cho bởi hàm lồi và ràng buộc đẳng thức cho bởi hàm affine.

### Hàm Lagrange | Lagrangian
$\mathcal{L} = f_0 + \sum_i\alpha_i f_i + \sum_j\nu_j h_j$: hàm mục tiêu cộng các ràng buộc nhân với nhân tử, như một khoản tính giá cho mỗi đơn vị vi phạm.

### Nhân tử Lagrange | Lagrange multiplier
Các hệ số $\alpha_i \ge 0$ và $\nu_j$ trong hàm Lagrange. Nhân tử của một ràng buộc cho biết giá trị tối ưu thay đổi bao nhiêu khi ràng buộc được nới lỏng (giá bóng).

### Hàm đối ngẫu | dual function
$g(\alpha,\nu) = \inf_x \mathcal{L}(x, \alpha, \nu)$. Luôn là hàm lõm, và mỗi giá trị của nó là một chặn dưới của giá trị tối ưu.

### Đối ngẫu yếu, đối ngẫu mạnh | weak duality, strong duality
Đối ngẫu yếu: $d^* \le p^*$, luôn đúng. Đối ngẫu mạnh: $d^* = p^*$, xảy ra với bài toán lồi thoả điều kiện Slater.

### Khe đối ngẫu | duality gap
Hiệu $p^* - d^*$ giữa giá trị tối ưu của bài toán gốc và bài toán đối ngẫu.

### Điều kiện Slater | Slater's condition
Bài toán lồi có một điểm thoả chặt mọi ràng buộc bất đẳng thức. Khi đó đối ngẫu mạnh xảy ra.

### Điều kiện KKT | Karush–Kuhn–Tucker conditions
Bốn nhóm điều kiện đặc trưng cho nghiệm tối ưu của bài toán lồi khả vi thoả điều kiện Slater: chấp nhận được của bài toán gốc, chấp nhận được của bài toán đối ngẫu, điều kiện bù và điều kiện dừng.

### Điều kiện bù | complementary slackness
$\alpha_i f_i(x^{*}) = 0$: với mỗi ràng buộc, hoặc ràng buộc chặt tại nghiệm, hoặc nhân tử của nó bằng 0. Trong SVM, điều kiện này sinh ra khái niệm vector hỗ trợ.

## Máy vector hỗ trợ

### Máy vector hỗ trợ (SVM) | support vector machine
Bộ phân loại tuyến tính chọn siêu phẳng có lề lớn nhất. Kết hợp với kernel để tạo biên quyết định phi tuyến.

### Vector hỗ trợ | support vector
Điểm dữ liệu có nhân tử $\alpha_i > 0$ trong nghiệm đối ngẫu. Chỉ các điểm này ảnh hưởng tới nghiệm: trong thí nghiệm ở Mục 13.3, 3 trên 120 điểm là vector hỗ trợ.

### SVM lề cứng | hard-margin SVM
Đòi hỏi mọi điểm thoả $y_i(w^\top x_i+b)\ge1$. Vô nghiệm khi dữ liệu không tách được tuyến tính.

### SVM lề mềm | soft-margin SVM
Cho phép vi phạm lề qua các biến bù $\xi_i \ge 0$, với mức phạt điều khiển bởi tham số $C$. Tương đương với mất mát hinge cộng regularization $\ell_2$.

### Tham số C | the C parameter
Cân bằng giữa lề rộng và mức phạt vi phạm; đóng vai trò nghịch đảo của hệ số regularization. $C$ nhỏ cho lề rộng, $C$ lớn cho lề hẹp và mô hình bám dữ liệu hơn.

### Kernel | kernel function
Hàm $K(x,x') = \varphi(x)^\top\varphi(x')$ cho tích vô hướng trong một không gian đặc trưng. Một hàm là kernel khi mọi ma trận kernel dựng từ nó đều đối xứng và nửa xác định dương (định lý Mercer).

### Thủ thuật kernel | kernel trick
Thay tích vô hướng trong bài toán đối ngẫu bằng một kernel, để làm việc trong không gian đặc trưng mà không cần tính $\varphi(x)$.

### Kernel RBF | RBF kernel, Gaussian kernel
$K(x,x') = \exp(-\gamma\|x-x'\|^2)$. Không gian đặc trưng tương ứng có vô hạn chiều, nhưng mỗi lần tính chỉ tốn $O(d)$. Tham số $\gamma$ lớn làm biên uốn lượn và dễ overfitting.

## Giảm chiều và phân cụm

### Phân tích thành phần chính (PCA) | principal component analysis
Tìm các hướng trực giao có phương sai lớn nhất, chính là các vector riêng của ma trận hiệp phương sai. Không dùng nhãn.

### Thành phần chính | principal component
Vector riêng của ma trận hiệp phương sai, xếp theo trị riêng giảm dần. Tỉ lệ trị riêng trên tổng các trị riêng là tỉ lệ phương sai được giải thích.

### Định lý Eckart–Young | Eckart–Young theorem
Xấp xỉ bằng $k$ thành phần chính đầu tiên là xấp xỉ hạng $k$ tốt nhất theo chuẩn Frobenius, và sai số tái tạo bằng $(n-1)$ nhân tổng các trị riêng bị bỏ.

### Phân tích biệt thức tuyến tính (LDA) | linear discriminant analysis
Tìm hướng cực đại tỉ số giữa tán xạ giữa các lớp và tán xạ trong lớp; với hai lớp, $w \propto S_W^{-1}(\mu_1 - \mu_0)$. Dùng nhãn, cho tối đa $K-1$ chiều.

### Lời nguyền số chiều | curse of dimensionality
Ở số chiều cao, dữ liệu trở nên rất thưa và khoảng cách mất dần ý nghĩa: với dữ liệu đều trong khối lập phương đơn vị, vùng lân cận chứa 1% dữ liệu cần cạnh dài 0,955 khi $d = 100$.

### Phân cụm | clustering
Chia dữ liệu không có nhãn thành các nhóm sao cho điểm cùng nhóm giống nhau hơn điểm khác nhóm.

### K-means | k-means clustering
Thuật toán phân cụm cực tiểu tổng bình phương khoảng cách từ mỗi điểm tới tâm cụm, giải bằng cách lặp bước gán và bước cập nhật (thuật toán Lloyd). Chỉ bảo đảm hội tụ tới cực tiểu địa phương.

### Inertia | inertia, within-cluster sum of squares
Hàm mục tiêu của K-means. Luôn giảm khi số cụm tăng và bằng 0 khi mỗi điểm là một cụm.

### k-means++ | k-means++
Cách khởi tạo tâm cụm tuần tự, mỗi tâm mới được chọn với xác suất tỉ lệ bình phương khoảng cách tới tâm gần nhất đã chọn. Kỳ vọng inertia không quá $O(\log k)$ lần tối ưu.

### Phương pháp elbow | elbow method
Chọn số cụm tại chỗ inertia ngừng giảm nhanh. Là quy tắc quan sát, không có định nghĩa toán học chặt chẽ.

### Hệ số silhouette | silhouette coefficient
$s = (b - a)/\max(a, b)$, với $a$ là khoảng cách trung bình tới các điểm cùng cụm và $b$ tới các điểm của cụm gần nhất khác. Giá trị gần 1 nghĩa là điểm thuộc đúng cụm.

## Hệ thống gợi ý

### Lọc cộng tác | collaborative filtering
Gợi ý dựa trên mẫu hình đánh giá của mọi người dùng, không cần đặc trưng của sản phẩm.

### Lọc dựa trên nội dung | content-based filtering
Mô tả sản phẩm bằng đặc trưng rồi học sở thích của từng người dùng theo các đặc trưng đó. Luôn trả lời được, kể cả với người dùng hoặc sản phẩm mới.

### Phân rã ma trận | matrix factorization
Xấp xỉ ma trận đánh giá bằng tích hai ma trận hạng thấp $PQ^\top$; mỗi người dùng và mỗi sản phẩm được biểu diễn bằng một vector $k$ chiều.

### Nhân tố ẩn | latent factor
Các toạ độ của vector người dùng và vector sản phẩm trong phân rã ma trận. Hình thành từ dữ liệu, không được định nghĩa trước.

### Bình phương tối thiểu luân phiên (ALS) | alternating least squares
Cố định $Q$ để giải $P$, rồi cố định $P$ để giải $Q$, lặp lại. Mỗi bước là một tập các bài hồi quy ridge độc lập.

### Phản hồi tường minh, phản hồi ngầm | explicit feedback, implicit feedback
Phản hồi tường minh là điểm số người dùng chủ động cho; phản hồi ngầm là hành vi như lượt xem hay lượt nhấp.

### Khởi đầu lạnh | cold start
Người dùng hoặc sản phẩm mới chưa có tương tác nào. Với phân rã ma trận, vector của họ không được dữ liệu xác định và mô hình dự đoán gần 0 cho mọi sản phẩm.

### Vòng phản hồi | feedback loop
Hệ thống chỉ thu được phản hồi về những gì nó đã gợi ý, nên dữ liệu huấn luyện ngày càng lệch về các lựa chọn cũ. Cách xử lý thông dụng là dành một tỉ lệ nhỏ gợi ý cho thăm dò ngẫu nhiên.
