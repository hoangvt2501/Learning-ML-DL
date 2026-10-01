# Ngân hàng câu hỏi tự kiểm tra — Học sâu

Cú pháp: `## Chương N` mở một nhóm, `### …` là câu hỏi, `- [x]` đánh dấu đáp án đúng,
dòng `>` là phần giải thích hiện ra sau khi trả lời.

## Chương 1

### Thiên kiến quy nạp của một thuật toán học là gì?
- [ ] Sai lệch có hệ thống của dự đoán so với giá trị thật
- [x] Tập các giả định thuật toán dùng để chọn giữa các hàm cùng khớp dữ liệu huấn luyện
- [ ] Tham số cộng thêm $b$ trong $Wx + b$
- [ ] Xu hướng của mô hình dự đoán lớp đa số
> Dữ liệu hữu hạn khớp được với vô số hàm, nên muốn dự đoán cho điểm chưa thấy, thuật toán phải ưu tiên một số hàm bằng những giả định không đến từ dữ liệu. Phương án đầu là độ chệch, phương án thứ ba là hệ số chặn (Mục 0.2, Mục 1.4).

### Vì sao gradient boosting thường tốt hơn mạng nơ-ron trên dữ liệu dạng bảng?
- [ ] Vì mạng nơ-ron không xử lý được đặc trưng số
- [x] Vì dữ liệu bảng thường có đặc trưng không liên quan, quan hệ dạng ngưỡng và mỗi cột có ý nghĩa riêng, hợp với phép chia theo từng trục của cây
- [ ] Vì cây quyết định luôn có ít tham số hơn
- [ ] Vì gradient boosting không cần chọn siêu tham số
> Grinsztajn và cộng sự (2022) so sánh trên 45 bộ dữ liệu bảng và chỉ ra ba đặc điểm đó. Phép biến đổi affine của mạng nơ-ron trộn các cột với nhau, trong khi cây xử lý từng cột riêng (Mục 1.2).

## Chương 2

### Độ chệch và phương sai trong phân rã sai số được định nghĩa qua kỳ vọng trên cái gì?
- [ ] Trên các điểm dữ liệu trong tập kiểm tra
- [x] Trên các tập huấn luyện có thể rút ra từ cùng một phân phối, tại một điểm $x_0$ cố định
- [ ] Trên các lần khởi tạo trọng số khác nhau
- [ ] Trên các giá trị siêu tham số khác nhau
> Phương sai đo mức dự đoán tại một điểm thay đổi khi đổi tập huấn luyện, không đo mức dự đoán dao động giữa các điểm dữ liệu (Mục 2.2).

### Sai số huấn luyện 2%, sai số xác thực 18%, khoảng cách lớn và vẫn thu hẹp dần khi thêm dữ liệu. Nên làm gì?
- [ ] Dùng mô hình linh hoạt hơn, thêm đặc trưng
- [x] Thêm dữ liệu, tăng regularization, tăng cường dữ liệu hoặc dừng sớm
- [ ] Huấn luyện thêm nhiều vòng
- [ ] Giảm regularization để mô hình khớp tốt hơn
> Khoảng cách lớn giữa hai sai số là dấu hiệu phương sai cao. Phương sai giảm khi có thêm dữ liệu, nên thêm dữ liệu có tác dụng; ngược với trường hợp độ chệch cao (Mục 2.4).

### Trong thí nghiệm ở Mục 2.3, vì sao thêm dữ liệu không giúp được đa thức bậc 1?
- [ ] Vì nhiễu quá lớn so với tín hiệu
- [x] Vì 74% sai số của nó là độ chệch², và độ chệch không giảm khi có thêm dữ liệu
- [ ] Vì phương sai của nó đã bằng 0
- [ ] Vì đa thức bậc 1 luôn underfitting với mọi dữ liệu
> Một đường thẳng không thể theo được hình sin dù có bao nhiêu dữ liệu; phải đổi họ hàm. Ngược lại, ở bậc 12 phương sai gấp khoảng 1 900 lần độ chệch², nên thêm dữ liệu giúp được.

### Trong thí nghiệm ở Mục 2.3, vì sao đa thức bậc 2 có độ chệch² gần như bằng bậc 1?
- [ ] Vì thuật toán khớp đa thức bậc 2 không hội tụ
- [x] Vì hàm thật là hàm lẻ và dữ liệu đối xứng quanh 0, nên số hạng bậc chẵn không giúp xấp xỉ hàm thật
- [ ] Vì 40 điểm dữ liệu quá ít cho đa thức bậc 2
- [ ] Vì nhiễu lấn át số hạng bậc 2
> $f(x) = \sin(2{,}2x) + 0{,}35x$ thoả $f(-x) = -f(x)$. Số hạng bậc chẵn chỉ thêm tham số để khớp nhiễu nên làm tăng phương sai mà không giảm độ chệch; độ chệch không nhất thiết giảm đều khi mô hình phức tạp hơn.

### Hiện tượng double descent nói điều gì?
- [ ] Phân rã độ chệch – phương sai là sai
- [x] Khi số tham số vượt ngưỡng khớp chính xác dữ liệu huấn luyện, sai số xác thực có thể giảm trở lại
- [ ] Mô hình càng nhiều tham số thì càng ít overfitting trong mọi trường hợp
- [ ] Cần huấn luyện hai lần để có kết quả tốt
> Phân rã là một đẳng thức nên luôn đúng. Điều thay đổi là phương sai không tăng đều theo số tham số, vì thuật toán tối ưu chọn một nghiệm cụ thể trong vô số nghiệm khớp dữ liệu, ví dụ nghiệm có chuẩn nhỏ nhất (Mục 2.5).

## Chương 3

### Phương sai của trung bình $M$ mô hình, mỗi mô hình có phương sai $\sigma^2$ và mỗi cặp có tương quan $\rho$, là bao nhiêu?
- [ ] $\sigma^2 / M$
- [x] $\rho\sigma^2 + \frac{1-\rho}{M}\sigma^2$
- [ ] $\rho\sigma^2 / M$
- [ ] $(1-\rho)\sigma^2$
> Khi $M$ tăng, số hạng thứ hai tiến về 0 và chỉ còn $\rho\sigma^2$. Muốn giảm tiếp phải giảm tương quan giữa các mô hình, và đó là điều rừng ngẫu nhiên làm (Mục 3.3).

### Gradient boosting chủ yếu giảm thành phần nào của sai số?
- [ ] Phương sai, vì lấy trung bình nhiều mô hình
- [x] Độ chệch, vì mỗi cây mới sửa phần sai của tổng các cây trước
- [ ] Nhiễu, vì nó loại được điểm ngoại lai
- [ ] Cả ba như nhau
> Mỗi cây xấp xỉ gradient âm của hàm mất mát theo dự đoán hiện tại, tức gradient descent trong không gian hàm. Cộng nhiều cây nông, mỗi cây có độ chệch cao, cho một mô hình có độ chệch thấp (Mục 3.3, Mục 3.4).

### Phát biểu nào về cây quyết định là đúng?
- [ ] Cây cần chuẩn hoá đặc trưng về cùng thang đo
- [x] Cây bất biến với mọi phép biến đổi đơn điệu áp dụng riêng cho từng đặc trưng
- [ ] Cây không bao giờ overfitting
- [ ] Độ quan trọng đặc trưng tính từ độ giảm độ không thuần nhất là ước lượng không chệch
> Phép chia $x_j \le t$ chỉ phụ thuộc thứ tự các giá trị, nên chuẩn hoá hay lấy logarit không đổi cây. Độ quan trọng tính từ độ giảm độ không thuần nhất thiên về các đặc trưng có nhiều giá trị; permutation importance đáng tin hơn (Mục 3.5).

### Rừng ngẫu nhiên khác bagging thông thường ở điểm nào?
- [ ] Huấn luyện các cây tuần tự
- [x] Mỗi nút chỉ chọn phép chia trong một tập con ngẫu nhiên các đặc trưng, để giảm tương quan giữa các cây
- [ ] Dùng cây nông hơn
- [ ] Không dùng mẫu bootstrap
> Vì phương sai của trung bình dừng ở $\rho\sigma^2$, giảm tương quan $\rho$ là cách duy nhất để giảm tiếp phương sai.

### Một nút có 6 điểm lớp A và 4 điểm lớp B. Độ không thuần nhất Gini của nút là bao nhiêu?
- [ ] 0,24
- [x] 0,48
- [ ] 0,52
- [ ] 0,971
> $1 - 0{,}6^2 - 0{,}4^2 = 1 - 0{,}36 - 0{,}16 = 0{,}48$. Con số 0,971 là entropy của nút tính theo bit (Ví dụ 3.1).

## Chương 4

### Vì sao mạng nơ-ron cần hàm kích hoạt phi tuyến?
- [ ] Để gradient không tiêu biến
- [x] Vì hợp của các phép biến đổi affine vẫn là một phép affine, nên mạng nhiều lớp sẽ tương đương một lớp
- [ ] Để đầu ra nằm trong khoảng $[0, 1]$
- [ ] Để huấn luyện hội tụ nhanh hơn
> $W_2(W_1x + b_1) + b_2 = (W_2W_1)x + (W_2b_1 + b_2)$. Không có hàm phi tuyến thì mạng bao nhiêu lớp cũng chỉ biểu diễn được hàm affine, và không giải được cả bài toán XOR (Ví dụ 4.1).

### Định lý xấp xỉ phổ quát **không** cho biết điều gì?
- [ ] Mạng một lớp ẩn xấp xỉ được mọi hàm liên tục trên tập compact
- [x] Cần bao nhiêu đơn vị ẩn, và gradient descent có tìm được bộ trọng số đó không
- [ ] Sai số xấp xỉ có thể nhỏ tuỳ ý
- [ ] Hàm kích hoạt không được là đa thức
> Đây là định lý tồn tại. Số đơn vị cần thiết, khả năng tìm được bằng tối ưu và lượng dữ liệu cần thiết đều nằm ngoài phạm vi của nó (Mục 4.3).

### Một mạng một lớp ẩn với đầu vào một chiều cần ít nhất bao nhiêu đơn vị ReLU để biểu diễn chính xác hàm răng cưa $2^k$ đoạn?
- [ ] $2k$
- [ ] $k^2$
- [x] $2^k - 1$
- [ ] Không số đơn vị nào đủ
> Mỗi đơn vị ReLU tạo đúng một điểm gãy, và hàm răng cưa có $2^k - 1$ điểm gãy bên trong. Mạng sâu $k$ lớp chỉ cần $2k$ đơn vị, tức $6k$ tham số; với $k = 7$ là 42 so với 386 tham số (Mục 4.4).

### Kết quả ở Mục 4.4 cho phép kết luận điều gì?
- [ ] Mạng càng sâu càng tốt với mọi bài toán
- [x] Với một số hàm, độ sâu cho phép biểu diễn bằng ít tham số hơn rất nhiều so với một lớp ẩn
- [ ] Gradient descent luôn tìm được nghiệm gọn của mạng sâu
- [ ] Mạng một lớp ẩn không xấp xỉ được hàm răng cưa
> Hàm răng cưa là trường hợp thuận lợi nhất cho độ sâu, và thí nghiệm chỉ nói về khả năng biểu diễn. Huấn luyện một mạng 7 lớp, mỗi lớp 2 đơn vị, để học hàm này bằng gradient descent gần như luôn thất bại.

### Vì sao đạo hàm của sigmoid làm gradient qua nhiều lớp nhỏ đi nhanh?
- [ ] Vì sigmoid không khả vi tại 0
- [x] Vì đạo hàm của sigmoid không vượt quá 1/4, nên qua 10 lớp riêng phần này đã nhân gradient với tối đa khoảng $10^{-6}$
- [ ] Vì sigmoid có đầu ra âm
- [ ] Vì sigmoid tốn nhiều phép tính
> $\sigma'(z) = \sigma(z)(1 - \sigma(z)) \le 1/4$ và $0{,}25^{10} \approx 9{,}5 \times 10^{-7}$. Đạo hàm của ReLU bằng 1 ở phía dương, nên không có vấn đề này (Mục 4.2).

## Chương 5

### Vì sao lan truyền ngược, tức vi phân chế độ ngược, rẻ hơn chế độ xuôi khi huấn luyện mạng nơ-ron?
- [ ] Vì quy tắc dây chuyền chỉ đúng theo chiều ngược
- [x] Vì hàm mất mát là một số, nên đi từ đầu ra về đầu vào chỉ cần nhân vector với ma trận và tính được mọi gradient trong một lượt
- [ ] Vì chế độ ngược không cần lưu giá trị trung gian
- [ ] Vì thư viện chỉ cài đặt chế độ ngược
> Quy tắc dây chuyền đúng theo mọi thứ tự nhân. Chế độ ngược tốn chi phí cỡ số đầu ra lần tính hàm, chế độ xuôi cỡ số đầu vào lần; với một đầu ra và hàng tỉ tham số, chế độ ngược rẻ hơn rất nhiều (Mục 5.2).

### Trong Ví dụ 5.1, vì sao hàng thứ hai của $W_1$ nhận gradient bằng 0?
- [ ] Vì trọng số của hàng đó bằng 0
- [x] Vì đơn vị ẩn thứ hai có $z_2 = -0{,}1 < 0$, nên đạo hàm của ReLU tại đó bằng 0
- [ ] Vì mất mát đã bằng 0
- [ ] Vì hệ số chặn bằng 0
> Đơn vị nào không kích hoạt với một đầu vào thì không nhận gradient từ đầu vào đó. Nếu không kích hoạt với mọi đầu vào, nó thành đơn vị ReLU chết (Mục 5.7).

### Vì sao Adam có bước hiệu chỉnh độ chệch?
- [ ] Để bù cho độ chệch của mô hình
- [x] Vì khởi tạo $m_0 = 0$ làm $m_t$ có kỳ vọng $(1 - \beta_1^t)\,\mathbb{E}[g]$, tức bị kéo về 0 ở các bước đầu
- [ ] Để tránh chia cho 0
- [ ] Để tốc độ học nhỏ lại ở cuối quá trình huấn luyện
> Với $\beta_1 = 0{,}9$, $m_1 = 0{,}1\,g_1$. Chia cho $1 - \beta_1^t$ cho ước lượng không chệch của kỳ vọng gradient; khi $t$ lớn, $\beta_1^t \to 0$ và phép hiệu chỉnh không còn tác dụng (Mục 5.4).

### Trạng thái của Adam cho mô hình 7 tỉ tham số, lưu ở FP32, chiếm bao nhiêu bộ nhớ?
- [ ] 28 GB
- [x] 56 GB
- [ ] 14 GB
- [ ] 112 GB
> Hai trạng thái $m$ và $v$ cho mỗi tham số, mỗi trạng thái 4 byte: $7 \times 10^9 \times 2 \times 4 = 56$ GB. Con số 112 GB là toàn bộ bộ nhớ huấn luyện với độ chính xác hỗn hợp, 16 byte mỗi tham số (Mục 12.5).

### AdamW khác Adam kèm regularization $\ell_2$ ở điểm nào?
- [ ] AdamW dùng hệ số $\beta$ khác
- [x] AdamW trừ weight decay trực tiếp vào tham số, không để thành phần phạt đi qua phép chia cho $\sqrt{\hat v}$
- [ ] AdamW không dùng momentum
- [ ] Hai cách tương đương về mặt toán học
> Với Adam thường, gradient của thành phần phạt bị chia cho $\sqrt{\hat v}$, nên tham số có gradient lớn lại bị phạt ít hơn, ngược với mục đích của regularization.

## Chương 6

### Hệ số 2 trong khởi tạo He, $\sigma_W^2 = 2/n_{\text{in}}$, đến từ đâu?
- [ ] Từ việc có hai lớp liên tiếp
- [x] Từ việc ReLU đặt một nửa phân phối bằng 0, nên $\mathbb{E}[\operatorname{ReLU}(z)^2] = \tfrac12 \operatorname{Var}(z)$
- [ ] Từ đạo hàm của ReLU bằng 2
- [ ] Từ thực nghiệm, không có lý do lý thuyết
> Muốn phương sai giữ nguyên qua lớp thì $n_{\text{in}}\sigma_W^2 \cdot \tfrac12 = 1$. Với tanh, gần tuyến tính quanh 0, không cần hệ số này và dùng khởi tạo Xavier (Mục 6.2).

### Trong thí nghiệm ở Mục 6.3, kết nối tắt không kèm chuẩn hoá cho kết quả gì trên mạng 40 lớp?
- [ ] Gradient ổn định, tỉ lệ khoảng 1
- [ ] Gradient tiêu biến về khoảng $10^{-18}$
- [x] Gradient bùng nổ, tới khoảng $2{,}4 \times 10^{8}$
- [ ] Không khác mạng thường
> Mỗi lớp cộng thêm vào tín hiệu một lượng cùng cỡ, nên độ lớn tăng theo cấp số nhân. Kết hợp kết nối tắt với chuẩn hoá trong nhánh cho tỉ lệ 2,22, là cấu hình giống pre-LN.

### Vì sao Transformer dùng LayerNorm thay cho BatchNorm?
- [ ] LayerNorm có nhiều tham số hơn
- [x] Vì chuỗi có độ dài khác nhau, BatchNorm hoạt động khác nhau khi huấn luyện và suy luận, và làm dự đoán của một mẫu phụ thuộc các mẫu khác trong lô
- [ ] Vì BatchNorm không dùng được với ReLU
- [ ] Vì LayerNorm luôn cho chất lượng cao hơn
> LayerNorm tính thống kê trên các đặc trưng của từng mẫu nên không có ba vấn đề trên (Mục 6.4).

### Kết nối tắt trong ResNet được đề xuất để giải quyết vấn đề gì?
- [ ] Overfitting
- [x] Vấn đề suy thoái: mạng thường sâu hơn có sai số huấn luyện cao hơn mạng nông hơn
- [ ] Gradient bùng nổ
- [ ] Bộ nhớ kích hoạt quá lớn
> Bằng chứng là sai số huấn luyện, không phải sai số xác thực, tăng theo độ sâu, dù mạng sâu hơn biểu diễn được mọi hàm của mạng nông hơn. Đây là vấn đề tối ưu (Mục 6.5).

### RMSNorm khác LayerNorm ở điểm nào?
- [ ] Chuẩn hoá theo lô thay vì theo đặc trưng
- [x] Bỏ bước trừ trung bình và độ dịch $\beta$, chỉ chia cho căn trung bình bình phương
- [ ] Không có tham số học được
- [ ] Luôn đặt sau khối con
> RMSNorm rẻ hơn và cho chất lượng tương đương, nên được Llama và nhiều mô hình ngôn ngữ khác dùng.

## Chương 7

### Trường tiếp nhận sau $L$ lớp tích chập $3 \times 3$ với bước nhảy 1 rộng bao nhiêu điểm ảnh?
- [ ] $3L$
- [x] $2L + 1$
- [ ] $3^L$
- [ ] $L^2$
> Theo $r_i = r_{i-1} + (k_i - 1)\prod_{j<i}s_j$ với $k = 3$ và $s = 1$, mỗi lớp thêm 2. Phủ ảnh rộng 224 điểm cần 112 lớp, nên CNN thực tế đều giảm độ phân giải bằng bước nhảy hoặc gộp (Mục 7.2).

### Vì sao trường tiếp nhận hiệu dụng nhỏ hơn con số lý thuyết?
- [ ] Vì lớp gộp làm mất thông tin
- [x] Vì ảnh hưởng của các điểm ảnh giảm dần từ tâm ra rìa theo dạng gần Gauss, và độ rộng hiệu dụng chỉ tăng theo $\sqrt L$
- [ ] Vì hàm kích hoạt cắt bớt tín hiệu
- [ ] Vì trọng số được khởi tạo nhỏ
> Kết quả của Luo và cộng sự (2016). Các điểm ở rìa trường tiếp nhận lý thuyết gần như không ảnh hưởng tới đơn vị.

### Ba giả định mà mạng tích chập đưa vào kiến trúc là gì?
- [ ] Tuyến tính, độc lập, cùng phân phối
- [x] Đặc trưng có tính cục bộ; đặc trưng hữu ích ở mọi vị trí (chia sẻ trọng số); vị trí chính xác ít quan trọng hơn sự có mặt (gộp, bước nhảy)
- [ ] Trơn, lồi, khả vi
- [ ] Thưa, hạng thấp, có cấu trúc
> Các giả định này đúng với ảnh tự nhiên nên CNN học được từ ít dữ liệu hơn ViT, vốn phải tự học những quan hệ đó (Mục 7.1, Mục 7.4).

### Một lớp tích chập $7 \times 7$, 64 bộ lọc, bước nhảy 2, phần đệm 3, nhận ảnh $224 \times 224 \times 3$. Đầu ra có kích thước không gian bao nhiêu?
- [ ] $224 \times 224$
- [x] $112 \times 112$
- [ ] $109 \times 109$
- [ ] $56 \times 56$
> $\lfloor (224 + 2 \cdot 3 - 7)/2 \rfloor + 1 = 112$. Lớp có $7 \cdot 7 \cdot 3 \cdot 64 = 9\,408$ trọng số (Ví dụ 7.1).

### Hai hộp $(0, 0, 4, 4)$ và $(2, 2, 6, 6)$ có IoU bằng bao nhiêu?
- [ ] $1/4$
- [x] $1/7$
- [ ] $1/8$
- [ ] $1/2$
> Phần giao là hình vuông $2 \times 2$, diện tích 4; phần hợp là $16 + 16 - 4 = 28$; IoU $= 4/28 = 1/7 \approx 0{,}143$ (Ví dụ 7.2). Lỗi hay gặp là chia cho tổng diện tích 32 thay vì diện tích phần hợp.

### Vì sao bộ phát hiện cần bước NMS?
- [ ] Để tăng độ phân giải của bản đồ đặc trưng
- [x] Vì dự đoán trên lưới dày nên một đối tượng có nhiều hộp gần nhau; NMS giữ hộp điểm cao nhất và bỏ các hộp trùng với nó quá ngưỡng IoU
- [ ] Để gán nhãn cho các hộp neo khi huấn luyện
- [ ] Để chuẩn hoá điểm tin cậy về khoảng từ 0 tới 1
> Khi đánh giá, hộp trùng của cùng một đối tượng bị tính là dương tính giả, nên NMS làm AP tăng; đổi lại, nó có thể loại nhầm hộp đúng của đối tượng đứng sát bên (Mục 7.5).

### Cùng một bộ phát hiện, vì sao AP@[0,5:0,95] thấp hơn nhiều so với AP50?
- [ ] Vì AP@[0,5:0,95] chỉ tính các đối tượng nhỏ
- [x] Vì nó lấy trung bình AP trên các ngưỡng IoU từ 0,5 tới 0,95, và ở ngưỡng cao chỉ những hộp định vị rất chính xác mới được tính là đúng
- [ ] Vì nó không dùng NMS
- [ ] Vì nó tính trên tập kiểm tra khác
> Trong thí nghiệm ở Mục 7.5, AP giảm từ 0,908 ở ngưỡng 0,5 xuống 0,014 ở ngưỡng 0,9, và trung bình trên 10 ngưỡng là 0,569.

### Trong U-Net, các kết nối giữa nhánh mã hoá và nhánh giải mã có tác dụng gì?
- [ ] Giảm số tham số của mạng
- [ ] Thay thế cho hàm kích hoạt
- [x] Đưa đặc trưng độ phân giải cao của các lớp nông sang nhánh giải mã, để khôi phục chính xác vị trí và biên của đối tượng
- [ ] Giúp mạng không cần tăng cường dữ liệu
> Đặc trưng sâu cho biết "đây là gì" nhưng đã mất chi tiết vị trí sau nhiều lần giảm độ phân giải; ghép với đặc trưng nông cùng mức cho biết "ở đâu" (Mục 7.6).

### Hai mô hình phân đoạn có IoU 0,5. Hệ số Dice tương ứng là bao nhiêu?
- [ ] 0,5
- [ ] 0,25
- [x] Khoảng 0,667
- [ ] 1
> $\text{Dice} = 2\,\text{IoU}/(1 + \text{IoU}) = 1/1{,}5 \approx 0{,}667$. Dice luôn lớn hơn hoặc bằng IoU, nên khi so sánh kết quả phải cùng một thước đo (Mục 7.6).

## Chương 8

### Giới hạn nào của RNN là lý do chính khiến Transformer thay thế nó?
- [ ] Số phép tính trên mỗi token
- [x] Số bước phải tính tuần tự: $O(T)$ so với $O(1)$ của self-attention
- [ ] Số tham số
- [ ] Dung lượng bộ nhớ
> Về số phép tính, self-attention tốn $O(T^2d)$, nhiều hơn $O(Td^2)$ của RNN khi $T > d$. Transformer thắng ở khả năng song song hoá theo thời gian và đường đi ngắn giữa hai vị trí bất kỳ (Mục 8.4).

### Dọc theo trạng thái ô nhớ của LSTM, gradient nhân với gì ở mỗi bước?
- [ ] Ma trận trọng số $W_{hh}$ như RNN thường
- [x] Giá trị của cổng quên $f_t$
- [ ] Một hằng số
- [ ] Tổng các gradient của từng bước
> $\partial c_T/\partial c_t = \prod_i \operatorname{diag}(f_i)$ khi bỏ qua các đường gián tiếp. Mạng học được $f$, nên có thể giữ $f$ gần 1 để nhớ lâu (Mục 8.3).

### Theo mô phỏng ở Mục 8.2, gradient dọc ô nhớ sau 100 bước còn bao nhiêu khi hệ số chặn của cổng quên bằng 1 và bằng 4?
- [ ] Hai trường hợp gần như nhau
- [x] Khoảng $3 \times 10^{-15}$ và $0{,}13$
- [ ] Khoảng $10^{-3}$ và $10^{-2}$
- [ ] Cả hai đều bùng nổ
> Cổng quên trung bình là $\sigma(1) \approx 0{,}73$ và $\sigma(4) \approx 0{,}98$, ứng với thang thời gian của bộ nhớ khoảng 4 và 50 bước. Thí nghiệm chỉ mô phỏng đường đi qua ô nhớ, không phải một mạng LSTM đầy đủ.

### Vì sao gradient bùng nổ dễ xử lý hơn gradient tiêu biến?
- [ ] Vì bùng nổ hiếm khi xảy ra
- [x] Vì cắt ngưỡng gradient giữ nguyên hướng và chỉ giới hạn độ lớn, còn gradient đã tiêu biến thì thông tin đã bị lẫn trong nhiễu
- [ ] Vì bùng nổ tự hết sau vài bước
- [ ] Vì tiêu biến chỉ xảy ra với sigmoid
> Tiêu biến cần sửa bằng kiến trúc: cổng như LSTM, kết nối tắt, chuẩn hoá (Mục 8.2).

## Chương 9

### Nếu các thành phần của $q, k \in \mathbb{R}^{d_k}$ độc lập, trung bình 0, phương sai 1, thì $q \cdot k$ có phương sai bao nhiêu?
- [ ] 1
- [ ] $\sqrt{d_k}$
- [x] $d_k$
- [ ] $d_k^2$
> $q \cdot k$ là tổng của $d_k$ số hạng độc lập, mỗi số hạng có phương sai 1. Ở Mục 9.2, với $d_k = 1\,024$ phương sai đo được là 1 022.

### Không chia điểm số cho $\sqrt{d_k}$ thì điều gì xảy ra khi $d_k$ lớn?
- [ ] Đầu ra nằm ngoài khoảng hợp lệ
- [x] Softmax bão hoà: với $d_k = 1\,024$, entropy chỉ còn 0,118 nat so với tối đa 4,159, và gradient gần như bằng 0
- [ ] Attention chia đều trọng số cho mọi vị trí
- [ ] Không có ảnh hưởng đáng kể
> Trọng số lớn nhất trung bình đạt 0,953. Đạo hàm của softmax là $p_i(\delta_{ij} - p_j)$, gần bằng 0 khi $p$ gần one-hot. Có chia thì entropy giữ quanh 3,68 với mọi $d_k$.

### Attention nhiều đầu mang lại điều gì?
- [ ] Nhiều khả năng tính toán hơn attention một đầu
- [x] Khả năng theo dõi nhiều loại quan hệ cùng lúc, với tổng chi phí gần như không đổi vì mỗi đầu hẹp đi $h$ lần
- [ ] Ngữ cảnh dài hơn
- [ ] Ít tham số hơn
> Một đầu duy nhất phải lấy trung bình có trọng số, trộn lẫn các loại quan hệ khác nhau (Mục 9.3).

### Thành phần nào trong một khối Transformer kết hợp thông tin giữa các vị trí?
- [ ] Cả attention và FFN
- [x] Chỉ attention; FFN xử lý từng vị trí độc lập
- [ ] Chỉ FFN
- [ ] Lớp chuẩn hoá
> FFN là cùng một MLP áp dụng riêng cho từng token. Bỏ attention đi, Transformer chỉ còn là một MLP áp dụng cho từng token (Mục 9.5).

### Vì sao self-attention cần thông tin vị trí?
- [ ] Vì softmax cần chuẩn hoá
- [x] Vì hoán vị các token đầu vào thì các vector đầu ra chỉ hoán vị theo, nên mô hình không phân biệt được thứ tự
- [ ] Vì cần giới hạn độ dài chuỗi
- [ ] Vì gradient phụ thuộc thứ tự token
> Không có thông tin vị trí thì "chó cắn người" và "người cắn chó" cho cùng một tập biểu diễn (Mục 9.6).

### Mô hình chỉ decoder khác mô hình chỉ encoder ở điểm nào?
- [ ] Decoder không dùng attention
- [x] Decoder dùng mặt nạ nhân quả và học đoán token tiếp theo; encoder dùng attention hai chiều và học đoán token bị che
- [ ] Encoder không có khối FFN
- [ ] Decoder chỉ dùng được cho dịch máy
> BERT là mô hình chỉ encoder, phù hợp cho phân loại và embedding; GPT và phần lớn mô hình ngôn ngữ lớn là mô hình chỉ decoder (Mục 9.7).

## Chương 10

### Tính chất quan trọng nhất của RoPE là gì?
- [ ] RoPE không thêm tham số
- [x] $\langle R_m q, R_n k\rangle = \langle q, R_{n-m}k\rangle$: điểm số chỉ phụ thuộc khoảng cách tương đối
- [ ] RoPE làm chi phí attention tuyến tính theo độ dài
- [ ] RoPE thay được lớp chuẩn hoá
> Ở Mục 10.3, các cặp vị trí $(5, 2)$, $(105, 102)$ và $(500, 497)$ cho cùng giá trị 6,851342. Không thêm tham số là một ưu điểm đi kèm.

### GQA được dùng vì lý do gì?
- [ ] Để tăng chất lượng mô hình
- [x] Để giảm KV cache, tức bộ nhớ và băng thông khi suy luận
- [ ] Để giảm số tham số khi huấn luyện
- [ ] Để tăng độ dài ngữ cảnh tối đa
> Với mô hình cỡ 70B, ngữ cảnh 4 096 token, 8 chuỗi, FP16: MHA cần 80 GiB KV cache, GQA với 8 nhóm cần 10 GiB. Chất lượng của GQA gần MHA nhưng không cao hơn (Mục 10.4).

### Vì sao $d_{\text{ff}}$ của Llama 2 7B là 11 008 chứ không phải $4d = 16\,384$?
- [ ] Để tiết kiệm bộ nhớ khi suy luận
- [x] Vì SwiGLU có ba ma trận nên số chiều ẩn giảm còn $\tfrac83 d \approx 10\,923$, làm tròn lên bội của 256
- [ ] Vì mô hình được tỉa bớt sau khi huấn luyện
- [ ] Vì $d_{\text{ff}}$ phải chia hết cho số đầu attention
> $11\,008 = 43 \times 256$ là bội của 256 nhỏ nhất không nhỏ hơn 10 923 (Mục 10.5).

### Mixture of experts tách hai đại lượng nào khỏi nhau?
- [ ] Thời gian huấn luyện và thời gian suy luận
- [x] Số tham số và số phép tính cho mỗi token
- [ ] Attention và FFN
- [ ] Bộ nhớ và độ trễ
> Mỗi token chỉ đi qua vài chuyên gia, nên mô hình có nhiều tham số nhưng mỗi token chỉ dùng một phần. Mọi chuyên gia vẫn phải nằm trong bộ nhớ (Mục 10.6).

### FlashAttention thay đổi điều gì so với cách tính attention trực tiếp?
- [ ] Giảm số phép tính từ bậc hai xuống tuyến tính theo độ dài chuỗi
- [x] Không lưu ma trận $T \times T$ trong bộ nhớ chính của GPU, nên bộ nhớ tăng tuyến tính theo $T$ và thời gian chạy giảm
- [ ] Tính xấp xỉ attention để nhanh hơn
- [ ] Bỏ bớt các đầu attention không quan trọng
> FlashAttention tính chính xác cùng kết quả, theo từng khối trong bộ nhớ trên chip, dùng softmax trực tuyến. Số phép tính vẫn là $O(T^2)$ (Mục 10.7).

## Chương 11

### Vì sao beam search thường cho văn bản mở nhạt và lặp?
- [ ] Vì nó chọn sai token đầu tiên
- [x] Vì nó tối đa hoá xác suất của chuỗi, trong khi văn bản do người viết không có xác suất cao một cách đều đặn
- [ ] Vì nó quá chậm nên phải cắt ngắn
- [ ] Vì nó không dùng được mặt nạ nhân quả
> Nhận xét của Holtzman và cộng sự (2020). Beam search vẫn phù hợp với dịch máy, nơi đầu ra bị ràng buộc chặt bởi đầu vào (Mục 11.4).

### Top-p khác top-k ở điểm cơ bản nào?
- [ ] Top-p nhanh hơn
- [x] Số token được giữ lại của top-p thay đổi theo mức chắc chắn của mô hình ở từng bước
- [ ] Top-p không dùng được cùng nhiệt độ
- [ ] Top-k cho kết quả tất định
> Ở Mục 11.3, với phân phối đuôi dài, top-p với $p = 0{,}9$ giữ 4 141 token, còn top-k với $k = 40$ luôn giữ 40 token dù mô hình chắc chắn hay không.

## Chương 12

### Số tham số không tính embedding của một Transformer $L$ lớp, số chiều $d$, với $d_{\text{ff}} = 4d$ là bao nhiêu?
- [ ] $4Ld^2$
- [ ] $8Ld^2$
- [x] $12Ld^2$
- [ ] $16Ld^2$
> $4d^2$ cho $W_Q$, $W_K$, $W_V$, $W_O$ cộng $8d^2$ cho FFN. Với GPT-2 small, $12 \cdot 12 \cdot 768^2 = 84\,934\,656$ (Mục 12.3).

### Trong GPT-2 small, embedding chiếm khoảng bao nhiêu phần tổng số tham số?
- [ ] 5%
- [ ] 15%
- [x] 31%
- [ ] 50%
> Bảng embedding có $50\,257 \times 768$ tham số. Với mô hình nhỏ và từ vựng lớn, embedding chiếm tỉ trọng lớn; tỉ lệ này giảm khi mô hình lớn hơn.

### Công thức $C \approx 6ND$ phân bổ chi phí cho mỗi token thế nào?
- [ ] $3N$ ở lượt xuôi và $3N$ ở lượt ngược
- [x] $2N$ ở lượt xuôi và $4N$ ở lượt ngược
- [ ] $N$ ở lượt xuôi và $5N$ ở lượt ngược
- [ ] $6N$ ở lượt xuôi, lượt ngược không tốn gì
> Lượt ngược tính gradient theo cả đầu vào và trọng số của mỗi lớp, nên tốn gấp đôi lượt xuôi. Với Llama 2 7B, con số 184 320 giờ-GPU đã công bố ứng với mức sử dụng phần cứng khoảng 37,5% (Ví dụ 12.1).

### Khi tính đủ ma trận điểm số, phần tính toán của attention lớn hơn phần còn lại khi nào?
- [ ] Luôn luôn, vì attention là $O(T^2)$
- [x] Khi $T$ lớn hơn khoảng $6d$; với $d = 4\,096$ là khoảng 24 500 token
- [ ] Khi $T > d$
- [ ] Khi số lớp vượt 32
> Tỉ số giữa hai phần là $T/(6d)$. Nếu cài đặt bỏ qua phần bị mặt nạ nhân quả che, ngưỡng thành khoảng $12d$ (Mục 12.4).

### Huấn luyện với Adam và độ chính xác hỗn hợp cần khoảng bao nhiêu byte cho mỗi tham số, chưa tính giá trị kích hoạt?
- [ ] 2 byte
- [ ] 4 byte
- [ ] 8 byte
- [x] 16 byte
> 2 byte trọng số 16 bit, 2 byte gradient, 4 byte bản sao FP32 của trọng số và 8 byte cho hai trạng thái của Adam. Mô hình 7 tỉ tham số cần 112 GB (Mục 12.5).

### Khi huấn luyện bị hết bộ nhớ, cách xử lý đầu tiên thường là gì, và vì sao?
- [ ] Tỉa bớt tham số, vì bộ nhớ tỉ lệ với số tham số
- [x] Giảm kích thước lô, vì bộ nhớ kích hoạt tỉ lệ với kích thước lô và độ dài chuỗi
- [ ] Chuyển từ Adam sang SGD
- [ ] Khởi động lại quá trình huấn luyện
> Sau đó có thể dùng tích luỹ gradient để giữ kích thước lô hiệu dụng, gradient checkpointing, FlashAttention, và chia trạng thái bộ tối ưu cho nhiều GPU (Mục 5.3, Mục 12.5).
