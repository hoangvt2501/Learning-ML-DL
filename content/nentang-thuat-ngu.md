# Từ điển thuật ngữ — Nền tảng Machine Learning

Cú pháp: `## Nhóm`, rồi `### Tiếng Việt | English` và phần định nghĩa bên dưới.
Mọi định nghĩa bám sát đúng cách dùng trong giáo trình.

## Khung chung

### Học có giám sát | supervised learning
Học từ các cặp $(x, y)$ có nhãn. Gồm hồi quy (nhãn là số) và phân loại (nhãn là lớp).

### Học không giám sát | unsupervised learning
Học từ $x$ không có nhãn. Gồm phân cụm, giảm chiều và ước lượng mật độ.

### Mô hình sinh | generative model
Mô hình hoá $p(x \mid y)$ rồi lật ngược bằng quy tắc Bayes. Naive Bayes, LDA và QDA đều thuộc loại này. Mạnh khi ít dữ liệu, và sinh được mẫu mới.

### Mô hình phân biệt | discriminative model
Mô hình hoá $p(y \mid x)$ trực tiếp, hoặc thậm chí chỉ mô hình biên quyết định. Hồi quy logistic, SVM và cây quyết định thuộc loại này. Mạnh khi nhiều dữ liệu.

### Phương pháp phi tham số | non-parametric method
Phương pháp mà số tham số tăng theo lượng dữ liệu, thay vì cố định trước. KNN là ví dụ cực đoan: nó giữ nguyên toàn bộ dữ liệu.

### Siêu tham số | hyperparameter
Tham số không học được từ dữ liệu huấn luyện, phải chọn bằng kiểm định — $\lambda$ của ridge, $C$ của SVM, $k$ của KNN và K-means.

## Đại số tuyến tính

### Siêu phẳng | hyperplane
Tập $\{x : w^\top x + b = 0\}$. Vector $w$ là pháp tuyến của nó, và khoảng cách từ điểm $x_0$ tới nó là $|w^\top x_0 + b| / \|w\|$ — công thức là toàn bộ lý do SVM tối thiểu hoá $\|w\|$.

### Chuẩn | norm
Phép đo độ dài của vector. $\ell_1$ là tổng trị tuyệt đối, $\ell_2$ là căn tổng bình phương. Lựa chọn giữa hai chuẩn này cho ra lasso và ridge — hai thuật toán khác nhau về **chất**, không chỉ về mức độ.

### Hạng | rank
Số cột độc lập tuyến tính của ma trận. $X^\top X$ khả nghịch khi và chỉ khi $X$ **đủ hạng cột**.

### Trị riêng, vector riêng | eigenvalue, eigenvector
Nghiệm của $Av = \lambda v$: hướng mà phép biến đổi chỉ kéo dãn chứ không xoay. Mọi ma trận đối xứng có $d$ vector riêng trực giao — đó là nền móng của PCA.

### Ma trận xác định dương | positive definite matrix
Ma trận đối xứng có $z^\top A z > 0$ với mọi $z \neq 0$; tương đương mọi trị riêng đều dương. Thêm $\lambda I$ vào ma trận nửa xác định dương làm nó thành xác định dương — đó là lý do ridge luôn có nghiệm.

### Số điều kiện | condition number
$\kappa(A) = \lambda_{\max}/\lambda_{\min}$. Quyết định gradient descent chạy nhanh hay chậm: số vòng lặp là $O(\kappa\log\frac1\varepsilon)$. Lưu ý $\kappa(X^\top X) = \kappa(X)^2$ — lập phương trình chuẩn tắc **bình phương** số điều kiện.

### Phân rã SVD | singular value decomposition
$X = UDV^\top$ với $U, V$ trực giao. Trị riêng của ma trận hiệp phương sai bằng $d_i^2/(n-1)$. Cắt bớt $D$ còn $k$ giá trị lớn nhất cho xấp xỉ hạng $k$ **tốt nhất** theo chuẩn Frobenius.

### Giả nghịch đảo Moore–Penrose | pseudoinverse
Ký hiệu $X^{+}$. Khi $X^\top X$ suy biến, $X^{+}y$ cho nghiệm có **chuẩn nhỏ nhất** trong vô số nghiệm — và đó chính là ridge khi $\lambda \to 0^{+}$.

### Cộng tuyến | collinearity, multicollinearity
Tình trạng các cột của $X$ gần phụ thuộc tuyến tính. Làm hệ số dao động dữ dội nhưng gần như **không** ảnh hưởng dự báo, nên nó là vấn đề của **diễn giải**.

## Xác suất

### Ma trận hiệp phương sai | covariance matrix
$\Sigma = \mathbb{E}[(x-\mu)(x-\mu)^\top]$. Đối xứng và nửa xác định dương nên luôn có phân rã phổ. **Ép nó thành ma trận đường chéo chính là giả thiết "naive"** của Naive Bayes.

### Khoảng cách Mahalanobis | Mahalanobis distance
$\sqrt{(x-\mu)^\top\Sigma^{-1}(x-\mu)}$. Khoảng cách Euclid sau khi chuẩn hoá theo hình dạng đám mây dữ liệu: đi xa theo hướng dữ liệu trải rộng thì rẻ, theo hướng bó hẹp thì đắt.

### Hàm hợp lý | likelihood
$p(\mathcal{D}\mid\theta)$ xem như hàm của $\theta$ với dữ liệu cố định. **Không phải** xác suất của $\theta$ — đó là chỗ nhầm phổ biến.

### Hợp lý cực đại | maximum likelihood estimation, MLE
$\arg\max_\theta p(\mathcal{D}\mid\theta)$. Với nhiễu Gauss cho ra bình phương tối thiểu; với nhãn Bernoulli cho ra entropy chéo.

### Tiên nghiệm, hậu nghiệm | prior, posterior
$p(\theta)$ là niềm tin trước khi thấy dữ liệu; $p(\theta\mid\mathcal{D})$ là niềm tin sau. **Phạt chuẩn chính là $-\log$ của tiên nghiệm.**

### Hậu nghiệm cực đại | maximum a posteriori, MAP
$\arg\max_\theta p(\theta\mid\mathcal{D})$. Không phải suy luận Bayes đầy đủ — nó lấy **một điểm** là đỉnh hậu nghiệm rồi vứt phần còn lại, nên không cho độ bất định.

### Làm trơn Laplace | Laplace smoothing, additive smoothing
Cộng $\alpha$ vào mọi ô đếm để tránh xác suất bằng 0. Không phải thủ thuật kỹ thuật mà **chính là** MAP với tiên nghiệm Dirichlet.

### Thiên lệch của ước lượng | bias of an estimator
$\mathbb{E}[\hat\theta] - \theta$, với kỳ vọng lấy **trên các tập dữ liệu có thể rút ra**, không phải trên các điểm trong một tập. Ridge là ước lượng **có thiên lệch**, và bài báo gốc năm 1970 nói rõ điều đó ngay trong tiêu đề.

## Hồi quy và tối ưu hoá

### Phương trình chuẩn tắc | normal equations
$X^\top X w = X^\top y$. Gọi là "chuẩn tắc" vì nó phát biểu rằng vector phần dư **trực giao** với mọi cột của $X$ — bình phương tối thiểu là phép chiếu vuông góc.

### Gradient descent | gradient descent
$\theta_{t+1} = \theta_t - \eta\nabla_\theta L$. Hướng ngược gradient là hướng giảm nhanh nhất **cục bộ**, theo bất đẳng thức Cauchy–Schwarz.

### Quán tính | momentum, heavy ball
Cộng dồn một phần bước trước vào bước hiện tại. Đổi bậc hội tụ từ $O(\kappa)$ xuống $O(\sqrt\kappa)$ — một thay đổi về **bậc**, không phải về hằng số.

### Xuống dốc ngẫu nhiên | stochastic gradient descent, SGD
Ước lượng gradient từ một mẫu hoặc một lô nhỏ. Thắng trên thực tế không vì toán học đẹp hơn mà vì phần cứng: nhân ma trận lô nhỏ tận dụng GPU tốt hơn.

### Kiểm tra gradient | gradient checking
So gradient giải tích với sai phân số. Dùng sai phân **trung tâm** với $\varepsilon \approx 10^{-5}$–$10^{-6}$; điểm tối ưu là $\varepsilon^{*} \sim u^{1/3}$, và $\varepsilon$ **nhỏ hơn thì tệ hơn** vì sai số làm tròn.

### Sai số cắt cụt, sai số làm tròn | truncation error, round-off error
Hai nguồn sai số đánh nhau trong sai phân số: cắt cụt tỉ lệ $\varepsilon^2$ (giảm khi $\varepsilon$ nhỏ), làm tròn tỉ lệ $u/\varepsilon$ (tăng khi $\varepsilon$ nhỏ). Cân bằng chúng cho $\varepsilon^{*}$.

## Phân loại

### Biên | margin (của một mẫu)
$m_i = y_i(w^\top x_i + b)$ với $y_i \in \{-1,+1\}$. Dương là đoán đúng, và $|m_i|$ đo mức tự tin. Mọi hàm mất mát phân loại đều là hàm của $m$.

### Lề | margin (của siêu phẳng)
Khoảng cách giữa hai mặt $w^\top x + b = \pm1$, bằng $2/\|w\|$. Đừng lẫn với **biên** của một mẫu ở trên — tiếng Anh dùng chung chữ *margin*.

### Mất mát 0–1 | zero-one loss
$\mathbb{1}[m \le 0]$. Thứ ta **thực sự** muốn tối thiểu, nhưng không lồi và có đạo hàm bằng 0 ở mọi chỗ khả vi nên không tối ưu trực tiếp được.

### Hàm mất mát thay thế | surrogate loss
Hàm lồi chặn trên mất mát 0–1, dùng thay cho nó. Perceptron, hinge và logistic đều là hàm thay thế — giảm chúng là gián tiếp giảm tỉ lệ đoán sai.

### Mất mát hinge | hinge loss
$\max(0, 1-m)$. Phẳng khi $m > 1$, tức **ép có lề** chứ không chỉ ép đúng. Đây là chỗ sinh ra khái niệm lề của SVM.

### Entropy chéo | cross-entropy loss
$-[y\log p + (1-y)\log(1-p)]$. Không phải định nghĩa mà là **hệ quả**: nó là âm log hợp lý dưới giả thiết Bernoulli.

### Perceptron | perceptron
Thuật toán phân loại tuyến tính chỉ cập nhật ở các mẫu đang sai. Có chặn **Novikoff**: số lần sai $\le (R/\gamma)^2$, **không phụ thuộc $n$ hay $d$** — nhưng chỉ khi dữ liệu tách được.

### Hồi quy logistic | logistic regression
Bất chấp tên gọi, đây là thuật toán **phân loại**. Chữ "hồi quy" đến từ việc nó hồi quy **log tỉ lệ cược** theo $x$.

### Softmax | softmax regression
Mở rộng hồi quy logistic cho $K$ lớp. Cộng cùng một vector vào mọi $w_k$ thì xác suất không đổi, nên chỉ **hiệu** giữa chúng là xác định được. Với $K=2$ nó **chính là** hồi quy logistic.

### Tách được tuyến tính | linearly separable
Tồn tại siêu phẳng phân đúng mọi điểm. Nghe như điều tốt, nhưng nó làm **hợp lý cực đại của hồi quy logistic vô nghiệm**: $\|w\| \to \infty$. Đây là lý do thư viện nào cũng bật phạt chuẩn mặc định.

### Hiệu chuẩn | calibration
Mức mà xác suất mô hình đưa ra phản ánh đúng tần suất thật. Naive Bayes phân loại khá nhưng **hiệu chuẩn rất tệ**, nên không nên dùng đầu ra của nó làm xác suất.

## Đánh giá

### Ma trận nhầm lẫn | confusion matrix
Bảng TP, FP, FN, TN. Mọi chỉ số phân loại đều dẫn xuất từ bốn con số này.

### Precision | precision
$TP/(TP+FP)$. Trả lời: *trong những ca tôi báo động, bao nhiêu phần là thật?* Mẫu số là những gì **mô hình nói**.

### Recall | recall, sensitivity
$TP/(TP+FN)$. Trả lời: *trong những ca thật sự có, tôi bắt được bao nhiêu phần?* Mẫu số là những gì **thực tế có**.

### F1 | F1 score
Trung bình **điều hoà** của precision và recall. Dùng điều hoà chứ không phải cộng vì nó phạt nặng sự mất cân đối: recall 1,00 với precision 0,012 chỉ cho $F_1 = 0{,}023$.

### F-beta | F-beta score
$F_\beta$ đánh trọng số recall gấp $\beta^2$ lần precision. Dùng khi chi phí của bỏ sót khác chi phí của báo động nhầm.

### ROC-AUC | ROC-AUC
Diện tích dưới đường TPR theo FPR. Trên dữ liệu mất cân bằng nó **trông đẹp giả tạo**, vì mẫu số của FPR là toàn bộ lớp âm vốn rất đông. Bộ đoán ngẫu nhiên luôn cho 0,5.

### PR-AUC | precision-recall AUC, average precision
Diện tích dưới đường precision theo recall. Hợp hơn ROC khi mất cân bằng. Bộ đoán ngẫu nhiên cho đúng **tỉ lệ lớp dương**, nên mốc so sánh thay đổi theo dữ liệu.

### Rò rỉ dữ liệu | data leakage
Thông tin từ tập kiểm tra lọt vào quá trình huấn luyện. Ba nguồn thường gặp: chuẩn hoá trước khi chia, chia ngẫu nhiên trên chuỗi thời gian, và chia ngẫu nhiên khi dữ liệu có nhóm.

### Kiểm định chéo | cross-validation
Chia dữ liệu thành $k$ phần, lần lượt giữ một phần để kiểm. Lỗi nghiêm trọng nhất là dùng **cùng một vòng** để vừa chọn siêu tham số vừa báo cáo hiệu năng — cách đúng là kiểm định chéo **lồng nhau**.

## Phạt chuẩn

### Phạt chuẩn | regularization
Thêm một số hạng phạt vào hàm mất mát để hạn chế độ phức tạp. Nhìn từ Chương 10, nó **chính là** $-\log$ của tiên nghiệm.

### Ridge | ridge regression, Tikhonov regularization
Phạt $\lambda\|w\|_2^2$. Luôn có nghiệm duy nhất kể cả khi $d > n$. Co ngót **không đều**: hệ số co theo hướng riêng thứ $i$ là $d_i^2/(d_i^2+\lambda)$, tức co mạnh nhất những hướng dữ liệu nói ít nhất.

### Lasso | lasso
Phạt $\lambda\|w\|_1$. Cho hệ số **bằng đúng 0** nên kiêm luôn việc chọn đặc trưng. Xử lý nhóm đặc trưng tương quan khá tuỳ tiện: thường chọn một đại diện và vứt phần còn lại.

### Elastic net | elastic net
Kết hợp cả phạt $\ell_1$ và $\ell_2$. Giữ được tính thưa của lasso mà xử lý nhóm tương quan ổn định hơn.

## Tối ưu lồi

### Tập lồi | convex set
Tập chứa trọn đoạn thẳng nối hai điểm bất kỳ của nó. **Giao** của các tập lồi là lồi; hợp thì thường không.

### Hàm lồi | convex function
$f(tx + (1-t)y) \le tf(x) + (1-t)f(y)$ — dây cung nằm trên đồ thị. Hệ quả quyết định: **mọi cực tiểu địa phương đều là cực tiểu toàn cục**.

### Hàm Lagrange | Lagrangian
$\mathcal{L} = f_0 + \sum\alpha_i f_i + \sum\nu_j h_j$. Ý tưởng: thay vì cấm vi phạm ràng buộc, hãy **tính tiền** mỗi lần vi phạm.

### Nhân tử Lagrange | Lagrange multiplier
Các $\alpha_i \ge 0$, đọc được như "giá" của từng ràng buộc. Điều kiện $\alpha_i \ge 0$ có nghĩa rõ ràng: vi phạm phải **làm tăng** giá trị hàm.

### Hàm đối ngẫu | dual function
$g(\alpha,\nu) = \inf_x \mathcal{L}$. **Luôn lõm**, kể cả khi bài toán gốc không lồi — vì nó là infimum của một họ hàm affine.

### Khe đối ngẫu | duality gap
$p^{*} - d^{*}$. Bằng 0 gọi là **đối ngẫu mạnh**, khi ấy giải bài toán đối ngẫu là giải xong bài toán gốc.

### Điều kiện Slater | Slater's condition
Nếu bài toán lồi và tồn tại điểm thoả mọi ràng buộc bất đẳng thức một cách **nghiêm ngặt**, thì đối ngẫu mạnh xảy ra. Điều kiện nhẹ và hầu như luôn thoả trong học máy.

### Điều kiện KKT | Karush–Kuhn–Tucker conditions
Bốn nhóm điều kiện tại nghiệm tối ưu. Nhóm quan trọng nhất là **bù trừ**: $\alpha_i f_i(x^{*}) = 0$ — ràng buộc lỏng thì giá bằng 0.

### Bù trừ | complementary slackness
$\alpha_i f_i(x^{*}) = 0$. Với mỗi ràng buộc, hoặc nó **chặt**, hoặc giá của nó **bằng 0**. Đây là thứ sinh ra khái niệm vector hỗ trợ — một hệ quả, không phải một thiết kế.

## SVM

### Vector hỗ trợ | support vector
Điểm có $\alpha_i > 0$. Chỉ chúng ảnh hưởng tới nghiệm; mọi điểm khác xoá đi cũng không đổi gì. Đo được: **3 trên 120 điểm** quyết định toàn bộ nghiệm trong thí nghiệm ở Mục 13.3.

### Lề cứng | hard margin
SVM đòi mọi điểm thoả $y_i(w^\top x_i+b)\ge1$. **Vô nghiệm** nếu dữ liệu không tách được.

### Lề mềm | soft margin
Thêm biến bù $\xi_i \ge 0$ cho phép vi phạm, và tính tiền qua tham số $C$. KKT chia các điểm thành đúng ba nhóm theo giá trị $\alpha_i$.

### Tham số C | the C parameter
Điều khiển đánh đổi giữa lề rộng và số điểm được phép vi phạm. $C$ nhỏ cho lề rộng và mô hình đơn giản; $C$ lớn cho lề hẹp và bám dữ liệu chặt hơn.

### Thủ thuật nhân | kernel trick
Tính $K(x,x') = \varphi(x)^\top\varphi(x')$ mà **không bao giờ** tính $\varphi(x)$. Khả dĩ vì bài toán đối ngẫu chỉ phụ thuộc dữ liệu qua tích vô hướng.

### Nhân RBF | RBF kernel, Gaussian kernel
$K(x,x') = \exp(-\gamma\|x-x'\|^2)$. Không gian đặc trưng tương ứng là **vô hạn chiều**, nhưng mỗi phép tính chỉ tốn $O(d)$.

## Giảm chiều và phân cụm

### Thành phần chính | principal component
Vector riêng của ma trận hiệp phương sai, xếp theo trị riêng giảm dần. Trục thứ nhất là hướng có phương sai lớn nhất.

### Định lý Eckart–Young | Eckart–Young theorem
Phép cắt SVD cho xấp xỉ hạng $k$ **tốt nhất** theo chuẩn Frobenius, và sai số của nó đúng bằng tổng các trị riêng bị bỏ. Kiểm chứng được tới $10^{-13}$.

### LDA | linear discriminant analysis
Tìm hướng tối đa hoá tỉ số giữa độ tách giữa các lớp và độ tản trong lớp. Khác PCA ở chỗ **dùng nhãn**. Cho tối đa $K-1$ chiều, và chính là bộ phân lớp Gauss dùng chung $\Sigma$.

### Lời nguyền số chiều | curse of dimensionality
Ở chiều cao, khoảng cách mất ý nghĩa: để hình cầu chứa tỉ lệ $f$ số điểm cần bán kính $f^{1/d}$, mà với $d=100$, $f=0{,}01$ thì bán kính là 0,955. Đây là lý do KNN hỏng ở chiều cao.

### Quán tính | inertia, within-cluster sum of squares
Hàm mục tiêu của K-means. **Luôn giảm** khi $k$ tăng và bằng 0 khi $k=n$, nên không thể chọn $k$ bằng cách tối thiểu hoá nó.

### k-means++ | k-means++
Cách khởi tạo rải các tâm ban đầu cho xa nhau, rút mỗi tâm mới với xác suất tỉ lệ $d^2$ tới tâm gần nhất đã chọn. Hạ tỉ lệ kẹt ở nghiệm tồi từ 71,5% xuống 46,5% — cải thiện rõ nhưng không triệt để.

### Khuỷu tay | elbow method
Chọn $k$ ở chỗ mức giảm của quán tính chậm hẳn lại. **Không phải định nghĩa toán học** mà là quy tắc nhìn bằng mắt, và thường không rõ trên dữ liệu thật.

### Điểm bóng | silhouette score
Đo mức một điểm hợp với cụm của nó hơn cụm gần nhất. Một lựa chọn thay cho khuỷu tay khi chọn $k$.

## Hệ gợi ý

### Lọc cộng tác | collaborative filtering
Chỉ dùng mẫu hình đánh giá, không dùng đặc trưng của mục. Ý tưởng: những người đồng ý với nhau trong quá khứ sẽ đồng ý tiếp.

### Dựa trên nội dung | content-based filtering
Mô tả mục bằng đặc trưng rồi học sở thích của từng người theo các đặc trưng ấy. Yếu hơn lọc cộng tác khi có nhiều dữ liệu, nhưng **luôn trả lời được** — nên là đường dự phòng bắt buộc.

### Phân rã ma trận | matrix factorization
$R \approx PQ^\top$. Mỗi người dùng và mỗi mục thành một vector $k$ chiều. Giả thiết hạng thấp là thứ làm bài toán có nghĩa.

### Yếu tố ẩn | latent factor
Các chiều của $P$ và $Q$. Không ai định nghĩa chúng; chúng nổi lên từ dữ liệu.

### Bình phương tối thiểu luân phiên | alternating least squares, ALS
Cố định $Q$ giải $P$, cố định $P$ giải $Q$, lặp lại. Bài toán không lồi theo cả hai cùng lúc nhưng **lồi theo từng cái** — mỗi bước là một bài ridge.

### Khởi đầu lạnh | cold start
Tình trạng người dùng hoặc mục mới chưa có tương tác nào. Với phân rã ma trận, hàm mất mát khi ấy chỉ còn phạt chuẩn nên vector của họ bị kéo về 0 và mô hình dự báo 0 cho mọi thứ — **tệ hơn cả đoán bừa**.

### Vòng phản hồi thoái hoá | degenerate feedback loop
Hệ chỉ gợi ý thứ nó đã biết, nên chỉ thu được phản hồi về thứ đó, nên càng ngày càng hẹp. Cách chữa là ngẫu nhiên hoá một tỉ lệ nhỏ số gợi ý — một quyết định kiến trúc, không phải tính năng thêm sau.
