# Ngân hàng câu hỏi tự kiểm tra — Nền tảng Machine Learning

Cú pháp: `## Chương N` mở một nhóm, `### …` là câu hỏi, `- [x]` đánh dấu đáp án đúng,
dòng `>` là phần giải thích hiện ra sau khi trả lời.

## Chương 1

### Ba thành phần của mọi thuật toán học có giám sát là gì?
- [ ] Dữ liệu, mô hình, độ chính xác
- [x] Mô hình, hàm mất mát, thuật toán tối ưu
- [ ] Đặc trưng, nhãn, siêu tham số
- [ ] Huấn luyện, kiểm định, kiểm tra
> Đổi một trong ba cột là ra một thuật toán khác. Chương 6 cho thấy perceptron, hồi quy logistic và SVM dùng **chung** mô hình và **chung** thuật toán tối ưu — chúng chỉ khác nhau ở hàm mất mát.

### Câu hỏi nào quyết định nhiều nhất trước khi chọn thuật toán?
- [ ] Thư viện nào đang được dùng nhiều nhất
- [x] Tỉ lệ giữa số mẫu và số đặc trưng, cấu trúc của dữ liệu, và việc cần dự báo hay cần giải thích
- [ ] Mô hình nào đạt điểm cao nhất trên các cuộc thi
- [ ] Có bao nhiêu GPU khả dụng
> Tỉ lệ $n/d$ quyết định gần như mọi thứ. Và mục tiêu dự báo so với mục tiêu giải thích có thể dẫn tới hai lựa chọn ngược nhau — Mục 4.4 cho một ví dụ đo được.

## Chương 2

### Số điều kiện của $X^\top X$ so với của $X$ thì thế nào?
- [ ] Bằng nhau
- [x] Bị **bình phương** lên
- [ ] Bằng một nửa
- [ ] Không có quan hệ gì
> Vì $X^\top X = VD^2V^\top$ nên các trị riêng là bình phương của các trị kỳ dị. Đây là lý do thực tế người ta giải bình phương tối thiểu bằng phân rã QR chứ không bằng công thức $(X^\top X)^{-1}X^\top y$ — lập ma trận chuẩn tắc làm mất một nửa số chữ số có nghĩa.

### Vì sao phạt $\ell_1$ cho hệ số bằng đúng 0 còn $\ell_2$ thì không?
- [ ] Vì $\ell_1$ phạt nặng hơn
- [x] Vì $\lvert w\rvert$ có điểm gãy tại 0, nên cần gradient thắng một lực có độ lớn cố định mới kéo được hệ số ra khỏi 0
- [ ] Vì $\ell_1$ không khả vi nên thuật toán bị lỗi và trả về 0
- [ ] Vì $\ell_2$ luôn dùng $\lambda$ nhỏ hơn
> Với $w^2$ thì đạo hàm là $2w$, yếu dần khi $w$ tiến về 0, nên không bao giờ đẩy được tới đúng 0. Đo ở Mục 9.4: ở $\lambda = 100$, lasso đưa 9 trong 12 hệ số về 0 còn ridge đưa được 0.

### Thêm $\lambda I$ vào một ma trận nửa xác định dương thì được gì?
- [x] Ma trận xác định dương, nên luôn khả nghịch với mọi $\lambda > 0$
- [ ] Ma trận vẫn có thể suy biến
- [ ] Ma trận trực giao
- [ ] Ma trận có định thức bằng 0
> Đây là lý do toán học khiến ridge **luôn** có nghiệm duy nhất, kể cả khi $d > n$ và OLS thì không có.

## Chương 3

### Phát biểu nào đúng?
- [ ] $\operatorname{Var}(X+Y) = \operatorname{Var}(X) + \operatorname{Var}(Y)$ luôn đúng
- [x] $\mathbb{E}[X+Y] = \mathbb{E}[X] + \mathbb{E}[Y]$ luôn đúng, còn phương sai chỉ cộng được khi độc lập
- [ ] Cả kỳ vọng và phương sai đều chỉ cộng được khi độc lập
- [ ] Cả hai đều cộng được vô điều kiện
> Đây là chỗ hay sai nhất. Khi có tương quan $\rho$ thì $\operatorname{Var}(X+Y)$ có thêm số hạng $2\rho\sqrt{\operatorname{Var}X \operatorname{Var}Y}$ — và công thức ấy chính là toàn bộ lý thuyết của bagging.

### Ép ma trận hiệp phương sai thành ma trận đường chéo tương đương với giả thiết nào?
- [ ] Dữ liệu tuân theo phân phối chuẩn
- [x] Các đặc trưng độc lập có điều kiện khi đã biết lớp — tức giả thiết "naive" của Naive Bayes
- [ ] Các lớp có cùng số mẫu
- [ ] Phương sai bằng nhau ở mọi đặc trưng
> Mục 7.4 kiểm chứng bằng số: Naive Bayes, LDA và QDA là cùng một bộ phân lớp Gauss, chỉ khác ràng buộc đặt lên $\Sigma$.

## Chương 4

### Khi nào phương trình chuẩn tắc không giải được?
- [ ] Khi dữ liệu có nhiễu
- [x] Khi $X$ không đủ hạng cột — nhiều đặc trưng hơn mẫu, hoặc có cột phụ thuộc tuyến tính
- [ ] Khi $y$ có giá trị âm
- [ ] Khi chưa chuẩn hoá đặc trưng
> Trường hợp kinh điển: mã hoá one-hot đủ $K$ mức mà vẫn giữ cột hằng số — tổng $K$ cột ấy đúng bằng cột hằng số.

### Khi $X^\top X$ suy biến thì bài toán thực chất là gì?
- [ ] Không có nghiệm nào
- [x] Có **vô số** nghiệm cho cùng một dự báo; giả nghịch đảo chọn cái có chuẩn nhỏ nhất
- [ ] Nghiệm là duy nhất nhưng khó tính
- [ ] Dữ liệu bị lỗi và phải thu thập lại
> Đo ở Mục 4.3: hai vector $w$ có chuẩn 3,8176 và 4,0711 cho sai số dự báo lệch nhau **đúng bằng 0**. Và lựa chọn "chuẩn nhỏ nhất" của pinv chính là ridge khi $\lambda \to 0^+$.

### Cộng tuyến gây hại cho cái gì?
- [ ] Dự báo
- [x] Hệ số — tức khả năng **diễn giải**; dự báo gần như không bị ảnh hưởng
- [ ] Tốc độ huấn luyện
- [ ] Cả dự báo lẫn hệ số như nhau
> Đo ở Mục 4.4: khi tương quan đi từ 0 lên 0,999, độ dao động của $w_1$ tăng **25 lần** (0,0324 → 0,8196) trong khi độ dao động của dự báo đi từ 0,0586 xuống 0,0585 — tức không đổi. Mô hình chỉ xác định được **tổng** ảnh hưởng của hai cột, không xác định được cách chia.

## Chương 5

### Tốc độ hội tụ của gradient descent phụ thuộc gì?
- [ ] Số chiều $d$
- [x] Số điều kiện $\kappa$ — số vòng lặp là $O(\kappa\log\frac{1}{\varepsilon})$
- [ ] Số mẫu $n$
- [ ] Giá trị khởi tạo
> Đo ở Mục 5.3: $\kappa$ đi từ 1 tới 10 000 làm số vòng đi từ 1 tới **92 104**, tăng đúng tuyến tính. Số chiều ảnh hưởng tới **chi phí mỗi vòng**, không ảnh hưởng tới **số vòng**.

### Thêm quán tính (heavy ball) đổi bậc hội tụ thế nào?
- [ ] Từ $O(\kappa)$ xuống $O(\log\kappa)$
- [x] Từ $O(\kappa)$ xuống $O(\sqrt\kappa)$
- [ ] Từ $O(\kappa^2)$ xuống $O(\kappa)$
- [ ] Không đổi bậc, chỉ đổi hằng số
> Đo ở $\kappa = 10\,000$: gradient descent cần 92 104 vòng, thêm quán tính chỉ cần **1 297**. Và lợi ích ấy **tăng lên** khi bài toán khó đi.

### Bước tiền xử lý rẻ tiền nào làm gradient descent nhanh hơn nhiều nhất?
- [x] Chuẩn hoá đặc trưng về cùng thang đo, vì nó giảm số điều kiện trực tiếp
- [ ] Sắp xếp dữ liệu theo nhãn
- [ ] Bỏ bớt mẫu để dữ liệu nhỏ hơn
- [ ] Đổi kiểu dữ liệu sang float32
> Nếu một đặc trưng có biên độ $10^6$ và một đặc trưng khác có biên độ $10^{-3}$ thì $\kappa$ cỡ $10^{18}$. Chuẩn hoá đưa các trị riêng về cùng cỡ độ lớn.

### Khi kiểm tra đạo hàm bằng sai phân, chọn $\varepsilon$ thế nào?
- [ ] Càng nhỏ càng chính xác
- [x] Có điểm tối ưu ở khoảng $u^{1/3} \approx 6\times10^{-6}$ với sai phân trung tâm; nhỏ hơn thì **tệ đi**
- [ ] Luôn dùng $\varepsilon = 10^{-12}$
- [ ] Chọn bằng đúng tốc độ học
> Hai nguồn sai số đánh nhau: cắt cụt cỡ $\varepsilon^2$ và làm tròn cỡ $u/\varepsilon$. Đo được điểm tối ưu ở $5{,}62\times10^{-6}$, khớp với lý thuyết $6{,}06\times10^{-6}$. Đặt $\varepsilon = 10^{-13}$ cho sai số tệ hơn khoảng bảy bậc độ lớn.

## Chương 6

### Perceptron, hồi quy logistic và SVM khác nhau ở đâu?
- [ ] Ở mô hình: tuyến tính, phi tuyến và dựa trên nhân
- [x] Chỉ ở **hàm mất mát**; cả ba dùng chung mô hình tuyến tính và cùng tối ưu bằng xuống dốc
- [ ] Ở thuật toán tối ưu
- [ ] Ở cách chuẩn hoá dữ liệu
> Và cả ba hàm mất mát đều là **chặn trên lồi** của mất mát 0–1, thứ ta thực sự muốn tối thiểu nhưng không tối ưu trực tiếp được vì nó không lồi và có đạo hàm bằng 0 ở mọi chỗ khả vi.

### Vì sao mất mát perceptron làm thuật toán không hội tụ trên dữ liệu không tách được?
- [ ] Vì tốc độ học quá lớn
- [x] Vì $\max(0,-m)$ **phẳng hoàn toàn khi $m>0$**, nên thuật toán buông ngay khi vừa đúng và không có khái niệm "đúng chắc chắn hơn"
- [ ] Vì nó không lồi
- [ ] Vì dữ liệu chưa được chuẩn hoá
> Đo ở Mục 6.2: trên dữ liệu không tách được, perceptron mắc **17 977 lần sai qua 2 000 lượt quét** và không dừng. Hinge và logistic không phẳng ngay tại $m=0$ nên vẫn còn động lực đẩy điểm ra xa biên.

### Hồi quy logistic và softmax với $K=2$ quan hệ thế nào?
- [ ] Softmax mạnh hơn vì có nhiều tham số hơn
- [x] Chúng là **một mô hình viết theo hai cách**; hiệu hai cột trọng số của softmax chính là $w$ của logistic
- [ ] Chúng cho kết quả gần giống nhau nhưng khác về bản chất
- [ ] Softmax chỉ dùng được khi $K \ge 3$
> Đo ở Mục 6.4: khớp cả hai tới hội tụ, xác suất lệch $3{,}3\times10^{-16}$ và trọng số lệch $1{,}6\times10^{-15}$ — tức sai số máy. Softmax có $K$ vector trọng số nhưng chỉ **hiệu** của chúng là xác định được.

### Vì sao hồi quy logistic gần như luôn cần phạt chuẩn?
- [ ] Để chạy nhanh hơn
- [x] Vì nếu dữ liệu tách được hoàn toàn thì hợp lý cực đại **không có nghiệm hữu hạn**: $\lVert w\rVert \to \infty$
- [ ] Vì hàm mất mát không lồi
- [ ] Vì gradient không tính được
> Đo ở Mục 6.5: không phạt chuẩn thì $\|w\|$ đi 10,2 → 15,9 → 24,7 → **34,8** qua các mốc 500 tới 50 000 vòng, không dấu hiệu dừng; có $\lambda = 0{,}01$ thì đứng yên ở 3,993. Đây là lý do `LogisticRegression` mặc định $C=1$.

## Chương 7

### Lời nguyền số chiều ảnh hưởng tới KNN thế nào?
- [x] Ở chiều cao, "láng giềng gần nhất" không còn gần: để chứa 1% số điểm ở $d=100$ chiều cần bán kính bằng 0,955 cạnh khối lập phương
- [ ] Bộ nhớ không đủ để lưu dữ liệu
- [ ] Thuật toán chạy chậm đi theo $d$
- [ ] Nhãn trở nên nhiễu hơn
> Đây là sự thật hình học chứ không phải vấn đề kỹ thuật: bán kính cần là $f^{1/d}$. Hệ quả là KNN chỉ dùng được ở chiều thấp, hoặc sau khi giảm chiều, hoặc trong không gian nhúng đã học được.

### Naive Bayes, LDA và QDA khác nhau ở đâu?
- [ ] Ở loại phân phối được dùng
- [x] Chỉ ở **ràng buộc đặt lên ma trận hiệp phương sai** $\Sigma$; cả ba là cùng một bộ phân lớp sinh Gauss
- [ ] Ở thuật toán tối ưu
- [ ] Ở cách xử lý dữ liệu thiếu
> Naive Bayes ép $\Sigma$ chéo; LDA dùng chung $\Sigma$ đầy đủ; QDA cho mỗi lớp một $\Sigma$ riêng đầy đủ.

### Vì sao biên quyết định của LDA là tuyến tính còn của QDA là bậc hai?
- [ ] Vì LDA dùng hàm tuyến tính còn QDA dùng đa thức
- [x] Vì khi hai lớp **dùng chung** $\Sigma$, số hạng bậc hai $x^\top\Sigma^{-1}x$ triệt tiêu lúc lấy hiệu hai hàm phân biệt
- [ ] Vì LDA chỉ xử lý được hai lớp
- [ ] Vì QDA có nhiều tham số hơn nên phức tạp hơn
> Kiểm chứng bằng số ở Mục 7.4: hàm quyết định của LDA khớp tuyến tính với sai số $4{,}4\times10^{-15}$, còn QDA thì sai số 17,8 — trật hẳn.

### Vì sao Naive Bayes vẫn chạy tốt dù giả thiết độc lập gần như luôn sai?
- [x] Vì phân loại đúng chỉ cần **thứ tự** giữa các xác suất hậu nghiệm đúng, không cần bản thân xác suất đúng
- [ ] Vì giả thiết ấy thực ra thường đúng
- [ ] Vì làm trơn Laplace sửa được sai lệch
- [ ] Vì nó chỉ dùng với dữ liệu văn bản
> Hệ quả thực tế: ước lượng xác suất của Naive Bayes thường rất tệ (dồn về 0 hoặc 1), nên **không nên** dùng đầu ra của nó làm xác suất.

## Chương 8

### Trên dữ liệu có 0,98% lớp dương, bộ phân loại "đoán tất cả là âm" đạt độ chính xác bao nhiêu?
- [ ] Khoảng 50%
- [ ] Khoảng 90%
- [x] **0,9902** — cao hơn cả bộ phân loại thật sự dùng được (0,9900)
- [ ] Không tính được
> Đây là con số đắt giá nhất của chương. Chọn mô hình theo độ chính xác trên dữ liệu mất cân bằng là chọn đúng cái vô dụng.

### Vì sao ROC-AUC trông đẹp trên dữ liệu mất cân bằng?
- [ ] Vì nó chỉ tính trên lớp dương
- [x] Vì mẫu số của FPR là **toàn bộ lớp âm**, mà lớp âm rất đông, nên thêm hàng trăm báo động nhầm cũng không làm FPR nhúc nhích
- [ ] Vì nó dùng thang logarit
- [ ] Vì nó bỏ qua các ngưỡng cực đoan
> Đo ở Mục 8.3: cùng mô hình, cùng dữ liệu — ROC-AUC 0,9715 nhưng PR-AUC chỉ 0,4931. Precision có mẫu số chỉ gồm những ca được báo động nên nó cảm nhận được ngay.

### PR-AUC của bộ đoán ngẫu nhiên bằng bao nhiêu?
- [ ] Luôn bằng 0,5
- [x] Bằng đúng **tỉ lệ lớp dương** — ở ví dụ trong bài là 0,0100
- [ ] Bằng 0
- [ ] Phụ thuộc số mẫu
> Còn ROC-AUC của bộ đoán ngẫu nhiên luôn là 0,5 bất kể mất cân bằng. Vì vậy PR-AUC 0,4931 phải so với mốc 0,0100 chứ không so với 0,5.

### Lỗi nào làm kết quả kiểm định tốt lên một cách giả tạo?
- [x] Chuẩn hoá trên toàn bộ dữ liệu **trước khi** chia tập
- [ ] Dùng $k = 10$ thay vì $k = 5$ trong kiểm định chéo
- [ ] Đặt hạt giống ngẫu nhiên cố định
- [ ] Dùng F1 thay vì độ chính xác
> Tính trung bình và độ lệch chuẩn trên cả dữ liệu là để thông tin tập kiểm tra rò vào tập huấn luyện. Phải khớp bộ chuẩn hoá **chỉ trên tập huấn luyện**.

## Chương 9

### Ridge co ngót các hướng riêng như thế nào?
- [ ] Đều nhau ở mọi hướng
- [x] Theo hệ số $d_i^2/(d_i^2+\lambda)$ — co **mạnh nhất** đúng những hướng có trị kỳ dị nhỏ
- [ ] Chỉ co hướng có trị kỳ dị lớn nhất
- [ ] Không co, chỉ xoay trục
> Đo ở Mục 9.3 với $\lambda=10$: hướng có $d_i = 12{,}27$ chỉ co còn 0,938, hướng có $d_i = 6{,}17$ co còn 0,792. Ridge co mạnh nhất những hướng dữ liệu nói ít nhất — đó là lý do nó chữa được cộng tuyến.

### Lỗi nghiêm trọng nhất khi dùng kiểm định chéo là gì?
- [ ] Chọn $k$ quá lớn
- [x] Dùng **cùng một vòng** kiểm định chéo để vừa chọn siêu tham số vừa báo cáo hiệu năng
- [ ] Không phân tầng theo lớp
- [ ] Chạy trên máy nhiều nhân
> Con số báo cáo khi ấy lạc quan có hệ thống, vì ta đã chọn cấu hình tốt nhất *trên chính tập đó*. Cách đúng là kiểm định chéo lồng nhau, hoặc giữ riêng một tập kiểm tra không chạm tới cho tới lần đo cuối.

## Chương 10

### Bình phương sai lệch đến từ đâu?
- [ ] Từ một quy ước toán học thuận tiện
- [x] Là **âm log hợp lý dưới giả thiết nhiễu Gauss**
- [ ] Từ định lý giới hạn trung tâm
- [ ] Từ bất đẳng thức Cauchy–Schwarz
> Nếu tin nhiễu có đuôi dày hơn Gauss thì MLE dưới giả thiết Laplace cho ra hồi quy trị tuyệt đối, bền hơn với điểm ngoại lai. Đổi giả thiết về nhiễu là đổi hàm mất mát.

### Quan hệ giữa ridge và MAP là gì?
- [ ] Tương tự nhau về mặt trực giác
- [x] Ridge **đúng bằng** MAP với tiên nghiệm Gauss, và $\lambda = \sigma^2/\tau^2$
- [ ] MAP là trường hợp riêng của ridge
- [ ] Không có quan hệ gì
> Kiểm chứng bằng số ở Mục 10.4: ridge dạng đóng và MAP tối ưu bằng BFGS — hai đường tính hoàn toàn khác nhau — cho nghiệm lệch $1{,}97\times10^{-8}$, tức bằng nhau trong dung sai của bộ tối ưu.

### Tiên nghiệm càng chặt ($\tau$ nhỏ) thì $\lambda$ đi theo chiều nào?
- [x] $\lambda$ **lớn lên**, vì $\lambda = \sigma^2/\tau^2$
- [ ] $\lambda$ nhỏ đi
- [ ] $\lambda$ không đổi
- [ ] Phụ thuộc vào dữ liệu
> Chiều này hợp lý: tin chắc rằng hệ số phải gần 0 thì dữ liệu phải đưa bằng chứng mạnh hơn mới kéo được nó ra xa — mà "cần bằng chứng mạnh hơn" chính là ý nghĩa của phạt chuẩn lớn.

### MAP có phải suy luận Bayes đầy đủ không?
- [ ] Có, đó là hai tên gọi của một thứ
- [x] **Không** — MAP chỉ lấy một điểm là đỉnh hậu nghiệm rồi vứt phần còn lại, nên không cho độ bất định
- [ ] Có, nhưng chỉ khi tiên nghiệm là Gauss
- [ ] Không, vì MAP không dùng quy tắc Bayes
> Suy luận Bayes đầy đủ giữ cả phân phối hậu nghiệm và lấy trung bình dự báo trên đó. Với hậu nghiệm lệch hoặc nhiều đỉnh, điểm đỉnh có thể chẳng đại diện cho gì.

## Chương 11

### Vì sao tính lồi quan trọng đến thế?
- [x] Vì khi ấy **mọi cực tiểu địa phương đều là cực tiểu toàn cục**, nên điểm khởi tạo không quan trọng
- [ ] Vì hàm lồi tính nhanh hơn
- [ ] Vì hàm lồi luôn khả vi
- [ ] Vì hàm lồi không cần gradient
> Đo ở Mục 11.6: chạy xuống dốc từ 21 điểm xuất phát, hàm lồi cho **một** điểm dừng; hàm $x^4-3x^2+x/2$ cho **hai**, với giá trị $-2{,}87$ và $-1{,}65$ khác hẳn nhau.

### Phép nào **không** giữ nguyên tính lồi?
- [ ] Cộng hai hàm lồi
- [ ] Lấy $\max$ của hai hàm lồi
- [x] Lấy $\min$ của hai hàm lồi
- [ ] Hợp hàm lồi với một ánh xạ affine
> Phép $\max$ giải thích vì sao mất mát hinge $\max(0,1-m)$ lồi — nó là max của hai hàm affine. Phép $\min$ thì không: min của hai parabol có thể có hai đáy.

### Kiểm tính lồi bằng cách thử bất đẳng thức định nghĩa trên nhiều cặp điểm thì kết luận được gì?
- [ ] Chứng minh được hàm lồi nếu không có vi phạm
- [x] Chỉ **bác bỏ** được: một phản ví dụ là đủ, nhưng không vi phạm thì chưa chứng minh được gì
- [ ] Không kết luận được gì cả
- [ ] Chứng minh được nếu thử đủ 10 000 cặp
> Không vi phạm trên 20 000 cặp không loại trừ được một chỗ lõm hẹp. Nhưng với hàm mất mát tự viết thì đây là cách rẻ nhất để bắt lỗi trước khi ngồi suy luận.

## Chương 12

### Điều kiện bù trừ của KKT phát biểu điều gì?
- [ ] Mọi nhân tử Lagrange đều dương
- [x] $\alpha_i f_i(x^*) = 0$ — ràng buộc **lỏng** thì nhân tử bằng 0, và ngược lại
- [ ] Gradient của hàm mục tiêu bằng 0
- [ ] Bài toán gốc và đối ngẫu có cùng số biến
> Đây là điều kiện sinh ra khái niệm **vector hỗ trợ**: chỉ những điểm nằm đúng trên lề mới có $\alpha > 0$ và mới ảnh hưởng tới nghiệm. Nó là một **hệ quả**, không phải một thiết kế.

### Hàm đối ngẫu $g(\alpha,\nu)$ có tính chất gì đặc biệt?
- [x] Nó **luôn lõm**, kể cả khi bài toán gốc không lồi
- [ ] Nó luôn lồi
- [ ] Nó luôn bằng giá trị tối ưu của bài toán gốc
- [ ] Nó chỉ xác định khi bài toán gốc lồi
> Vì nó là infimum của một họ hàm affine theo $(\alpha,\nu)$. Đây là một trong những sự thật hữu dụng nhất của tối ưu hoá.

### Vì sao SVM chuyển sang bài toán đối ngẫu?
- [ ] Vì bài toán gốc không giải được
- [x] Vì bài toán đối ngẫu chỉ phụ thuộc dữ liệu qua **tích vô hướng** $x_i^\top x_j$, và từ đó thủ thuật nhân trở thành hiển nhiên
- [ ] Vì đối ngẫu luôn nhanh hơn
- [ ] Vì nó cho nghiệm chính xác hơn
> Hai lý do phụ: đối ngẫu có $n$ biến thay vì $d$ (thắng khi $d \gg n$), và nó cho chặn dưới để biết còn cách tối ưu bao xa.

## Chương 13

### SVM chọn siêu phẳng nào trong vô số siêu phẳng tách đúng?
- [ ] Cái đầu tiên tìm được
- [x] Cái có **lề rộng nhất**, tức cách đều hai lớp nhiều nhất có thể
- [ ] Cái đi qua trọng tâm dữ liệu
- [ ] Cái có ít tham số nhất
> Cực đại $2/\|w\|$ tương đương cực tiểu $\tfrac12\|w\|^2$, cho ra một bài toán lồi với ràng buộc affine.

### Bao nhiêu điểm quyết định nghiệm SVM trong thí nghiệm ở Mục 13.3?
- [ ] Toàn bộ 120 điểm
- [x] **3 trên 120 điểm**; xoá 117 điểm còn lại thì kết quả không đổi
- [ ] Một nửa số điểm
- [ ] Phụ thuộc vào $C$
> Đây là nghĩa thực sự của từ "thưa" khi nói về SVM, và nó là hệ quả trực tiếp của điều kiện bù trừ.

### KKT chia các điểm trong SVM lề mềm thành mấy nhóm?
- [x] Ba: $\alpha=0$ ngoài lề, $0<\alpha<C$ đúng trên lề, $\alpha=C$ vi phạm lề
- [ ] Hai: vector hỗ trợ và không phải vector hỗ trợ
- [ ] Bốn
- [ ] Không chia nhóm nào
> Đo ở Mục 13.4: ở $C = 0{,}003$ thì **cả 114 vector hỗ trợ đều là điểm vi phạm** và không điểm nào nằm đúng trên lề — lề rộng tới mức nuốt gần hết dữ liệu.

### Thủ thuật nhân cho phép làm gì?
- [ ] Ánh xạ dữ liệu lên chiều cao rồi phân loại ở đó
- [x] Tính $\varphi(x)^\top\varphi(x')$ mà **không bao giờ** tính $\varphi(x)$ — với nhân RBF thì $\varphi(x)$ ở không gian vô hạn chiều
- [ ] Giảm số chiều của dữ liệu
- [ ] Làm cho hàm mất mát trở nên lồi
> Phương án đầu gần đúng nhưng bỏ mất điểm hay nhất. Đo ở Mục 13.5: trên dữ liệu hai vòng tròn đồng tâm, nhân tuyến tính đạt 0,615 còn nhân đa thức bậc 2 và RBF đều đạt 1,000.

### Vì sao SVM ít được dùng ở quy mô rất lớn?
- [ ] Vì nó kém chính xác hơn mạng nơ-ron
- [x] Vì huấn luyện tốn $O(n^2)$–$O(n^3)$ và suy luận phải tính nhân với **mọi** vector hỗ trợ, mà số ấy tăng tuyến tính theo $n$
- [ ] Vì nó không xử lý được nhiều lớp
- [ ] Vì nó cần GPU
> Xuống dốc ngẫu nhiên chỉ tốn $O(n)$ mỗi vòng. SVM vẫn là lựa chọn tốt khi $n$ vừa phải và $d$ lớn — ví dụ văn bản đã vector hoá.

## Chương 14

### Định lý Eckart–Young phát biểu điều gì?
- [x] Sai số tái dựng của PCA **đúng bằng** tổng các trị riêng bị bỏ, và không phép chiếu tuyến tính nào làm tốt hơn
- [ ] PCA luôn giữ được 95% phương sai
- [ ] Số thành phần tối ưu bằng căn bậc hai của số chiều
- [ ] PCA và LDA cho cùng kết quả khi dữ liệu là Gauss
> Kiểm chứng ở Mục 14.3: đẳng thức khớp tới $\sim10^{-13}$ trên cả tám giá trị $k$.

### Khi nào PCA phản tác dụng?
- [ ] Khi dữ liệu có nhiều chiều
- [x] Khi hướng phương sai lớn nhất **không mang thông tin phân loại** — PCA không nhìn nhãn
- [ ] Khi số mẫu nhỏ hơn số chiều
- [ ] Khi dữ liệu không tuân theo phân phối chuẩn
> Đo ở Mục 14.4: trục chính thứ nhất giữ **96,1% phương sai** nhưng cho AUC **0,502** — ngang đoán bừa, trong khi LDA đạt 0,9987. Phương sai lớn không đồng nghĩa có ích.

### LDA cho tối đa bao nhiêu chiều?
- [ ] $d$
- [x] $K - 1$, vì $S_B$ dựng từ $K$ vector trung bình quanh trung bình chung nên chỉ có hạng $K-1$
- [ ] $\min(n, d)$
- [ ] Không giới hạn
> Với bài toán hai lớp, LDA cho đúng **một** chiều.

### Vì sao hầu như luôn phải chuẩn hoá trước khi làm PCA?
- [x] Vì PCA **không bất biến** với phép co giãn từng đặc trưng: đổi đơn vị từ mét sang milimét làm phương sai nhân $10^6$ và đặc trưng ấy chiếm trọn trục chính
- [ ] Vì thuật toán không hội tụ nếu chưa chuẩn hoá
- [ ] Vì trị riêng phải dương
- [ ] Vì không thì ma trận hiệp phương sai suy biến
> Tương đương với việc làm PCA trên ma trận **tương quan** thay vì hiệp phương sai. Ngoại lệ là khi mọi đặc trưng vốn cùng đơn vị, ví dụ cường độ điểm ảnh.

## Chương 15

### K-means bảo đảm điều gì?
- [ ] Tìm được phân hoạch tối ưu toàn cục
- [x] Chỉ bảo đảm **hội tụ tới cực tiểu địa phương**; bài toán gốc NP-khó
- [ ] Hội tụ trong nhiều nhất $k$ vòng
- [ ] Không bảo đảm gì, có thể chạy mãi
> Cả hai bước đều không làm tăng hàm mục tiêu và số cách phân hoạch là hữu hạn, nên nó chắc chắn dừng — nhưng dừng ở đâu thì tuỳ khởi tạo.

### k-means++ cải thiện được bao nhiêu trong thí nghiệm ở Mục 15.2?
- [ ] Loại bỏ hoàn toàn việc kẹt ở nghiệm tồi
- [x] Hạ tỉ lệ kẹt từ **71,5%** xuống **46,5%** — cải thiện rõ nhưng không triệt để
- [ ] Không cải thiện gì đáng kể
- [ ] Làm chậm thuật toán mà không cải thiện chất lượng
> Vì vậy vẫn phải chạy nhiều lần rồi giữ kết quả có quán tính nhỏ nhất. `KMeans` của scikit-learn mặc định `n_init=10` chính vì lý do này.

### Ba giả định ngầm của K-means là gì?
- [x] Cụm lồi và gần hình cầu, kích thước tương đương, mọi chiều cùng thang đo
- [ ] Dữ liệu tuân theo phân phối chuẩn, độc lập, cùng phân phối
- [ ] Số cụm đã biết, không có nhiễu, không có điểm ngoại lai
- [ ] Dữ liệu tách được tuyến tính
> Vì nó gán điểm cho tâm gần nhất theo khoảng cách Euclid, biên giữa hai cụm luôn là siêu phẳng. Đo được: ba cụm tròn → đúng 100%; hai lưỡi liềm → 75%; hai dải dẹt → **53%**, gần như đoán bừa.

### Vì sao không thể chọn $k$ bằng cách tối thiểu hoá quán tính?
- [x] Vì quán tính **luôn giảm** khi $k$ tăng, và bằng 0 khi $k = n$
- [ ] Vì quán tính không tính được với $k$ lớn
- [ ] Vì quán tính không liên quan tới chất lượng cụm
- [ ] Vì thuật toán không hội tụ với $k$ lớn
> Quy tắc khuỷu tay nhìn chỗ mức giảm chậm hẳn lại. Đo trên dữ liệu 8 cụm: mức giảm **27,7%** khi lên $k=8$ rồi rơi xuống **5,4%** khi lên $k=9$. Nhưng khuỷu tay không phải định nghĩa toán học và thường không rõ trên dữ liệu thật.

## Chương 16

### Phân rã ma trận cần khoảng bao nhiêu đánh giá mỗi người dùng?
- [ ] Ít nhất 2 đánh giá
- [x] Khoảng **5 lần số yếu tố ẩn** $k$; dưới ngưỡng ấy nó còn tệ hơn đoán bừa
- [ ] Càng nhiều càng tốt, không có ngưỡng
- [ ] Bằng số mục trong danh mục
> Đo ở Mục 16.3 với $k=4$: ở 10,1 đánh giá mỗi người, RMSE là 3,54 — tệ hơn đoán bừa bằng 0 (2,08). Ở 19,8 đánh giá thì RMSE rơi xuống 0,30, tức giảm hơn 11 lần chỉ vì tăng gấp đôi dữ liệu.

### Chọn hạng $k$ quá lớn thì sao?
- [ ] Mô hình luôn tốt hơn vì linh hoạt hơn
- [x] Sai số trên ô **đã thấy** tiếp tục giảm nhưng sai số trên ô **chưa thấy** tăng lên — quá khớp
- [ ] Thuật toán không hội tụ
- [ ] Không có ảnh hưởng gì
> Đo ở Mục 16.4: RMSE ô chưa thấy chạm đáy tại $k=4$ (đúng hạng thật) với 0,1448, rồi ở $k=20$ tăng lên 0,6291 — tệ hơn 4,3 lần. Trong khi RMSE ô đã thấy vẫn giảm đều xuống 0,1903.

### Vì sao phân rã ma trận thất bại hoàn toàn với người dùng mới?
- [ ] Vì thuật toán cần khởi tạo lại
- [x] Vì hàm mất mát khi ấy chỉ còn số hạng phạt chuẩn $\lambda\lVert p_u\rVert^2$, mà cực tiểu của nó là $p_u = 0$
- [ ] Vì ma trận trở nên suy biến
- [ ] Vì số chiều không khớp
> Đo ở Mục 16.5: $\|p_u\|$ của người dùng mới là **0,154** so với 2,084 của người cũ, và RMSE 2,24 — **tệ hơn cả đoán bừa** 2,08. Đây là lý do mọi hệ thực tế đều phải lai.
