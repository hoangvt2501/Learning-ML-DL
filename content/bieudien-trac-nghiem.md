# Ngân hàng câu hỏi tự kiểm tra — Biểu diễn, mô hình sinh và căn chỉnh

Cú pháp: `## Chương N` mở một nhóm, `### …` là câu hỏi, `- [x]` đánh dấu đáp án đúng,
dòng `>` là phần giải thích hiện ra sau khi trả lời.

## Chương 2

### Skip-gram với lấy mẫu âm thực chất phân rã ma trận nào?
- [ ] Ma trận số lần đồng hiện thô
- [x] Ma trận PMI dịch đi một lượng $\log k$
- [ ] Ma trận hiệp phương sai của các từ
- [ ] Không phân rã ma trận nào, vì nó là mạng nơ-ron sâu
> Kết quả của Levy và Goldberg (2014): cho đạo hàm của hàm mục tiêu kỳ vọng bằng 0 được $\langle w_i, c_j\rangle = \mathrm{PMI}(i,j) - \log k$. Ở Mục 2.6, với hạng đầy đủ, tích vô hướng học được có tương quan 0,9995 với PMI.

### Vì sao dùng PMI thay cho số lần đồng hiện thô?
- [ ] Vì PMI dễ tính hơn
- [x] Vì số đếm thô bị các từ phổ biến chi phối; PMI đo hai từ cùng xuất hiện nhiều hơn mức ngẫu nhiên bao nhiêu
- [ ] Vì PMI luôn dương nên dễ phân rã
- [ ] Vì PMI không cần chuẩn hoá
> Một từ như "của" đồng hiện với hầu hết các từ vì nó phổ biến, không vì nó liên quan. Mẫu số $p(w)p(c)$ của PMI là xác suất đồng hiện nếu hai từ độc lập, nên tỉ số đo phần vượt mức ngẫu nhiên.

### Lấy mẫu âm giải quyết vấn đề gì?
- [x] Mẫu số của softmax là tổng trên toàn bộ từ vựng, tính lại cho mỗi cặp huấn luyện
- [ ] Dữ liệu huấn luyện có quá nhiều nhiễu
- [ ] Embedding bị overfitting
- [ ] Gradient bị tiêu biến
> Với từ vựng 2 triệu từ và vector 300 chiều, softmax đầy đủ tốn khoảng 1,2 tỉ phép tính mỗi cặp, lấy mẫu âm với $k = 5$ tốn 3 600. Chi phí của lấy mẫu âm không phụ thuộc kích thước từ vựng.

### Trong thí nghiệm ở Mục 2.6, embedding skip-gram hạng 8 giải đúng bao nhiêu phép loại suy, so với SVD trên PPMI cùng số chiều?
- [ ] Cả hai đều đạt 100%
- [x] Khoảng 20%, so với 100% của SVD
- [ ] Khoảng 90%, gần bằng SVD
- [ ] Cao hơn SVD vì skip-gram tối ưu trực tiếp
> Hai phương pháp nhắm tới cùng một ma trận đích nhưng xấp xỉ khác nhau khi bị giới hạn số chiều: SVD cho xấp xỉ tốt nhất với trọng số như nhau cho mọi phần tử, skip-gram cho xấp xỉ có trọng số theo tần suất và dùng hai bộ vector. Trên dữ liệu thật, khác biệt giữa các phương pháp phụ thuộc nhiều vào siêu tham số (Levy, Goldberg và Dagan, 2015).

### Vì sao độ chính xác loại suy của SVD giảm khi giữ 12 hoặc 16 chiều, rồi lại đạt 100% ở 20 chiều?
- [ ] Vì giữ nhiều chiều hơn luôn gây overfitting
- [x] Vì ma trận PPMI chỉ có 8 trị riêng dương; các hướng tiếp theo ứng với trị riêng âm, và SVD không phân biệt dấu
- [ ] Vì thuật toán SVD không hội tụ ở số chiều cao
- [ ] Vì phép loại suy chỉ dùng 8 từ
> Tám trị riêng dương ứng với cấu trúc hai trục của dữ liệu ($1 + 4 + 3 = 8$); mười hai trị riêng âm sinh ra vì đường chéo của ma trận bằng 0. Giữ đủ 20 chiều thì ma trận được tái tạo chính xác (Mục 2.4).

### Số mũ 3/4 trong phân phối lấy mẫu âm có cơ sở gì?
- [ ] Được suy ra từ lý thuyết thông tin
- [x] Được chọn bằng thực nghiệm, như bài báo gốc ghi rõ
- [ ] Là hệ quả của định lý giới hạn trung tâm
- [ ] Để phân phối lấy mẫu trở thành phân phối chuẩn
> Số mũ 3/4 cho kết quả tốt hơn số mũ 1 (phân phối unigram) và số mũ 0 (phân phối đều) trong thí nghiệm của Mikolov và cộng sự (2013); không có lý thuyết nào dẫn tới con số này.

## Chương 3

### Vì sao hai embedding không liên quan vẫn có thể có cosine rất cao?
- [x] Vì đám mây embedding lệch khỏi gốc toạ độ (tính bất đẳng hướng), nên mọi vector có chung một thành phần lớn
- [ ] Vì số chiều quá cao
- [ ] Vì các vector chưa được chuẩn hoá độ dài
- [ ] Vì cosine luôn dương
> Ở Mục 3.2, 900 vector ngẫu nhiên độc lập cộng một vector hằng có cosine trung bình 0,8724, dù không có quan hệ gì với nhau.

### Cách rẻ nhất để giảm tác động của tính bất đẳng hướng là gì?
- [x] Trừ vector trung bình của cả tập trước khi tính cosine
- [ ] Nâng ngưỡng cosine lên
- [ ] Giảm số chiều bằng PCA
- [ ] Dùng khoảng cách Euclid thay cho cosine
> Trừ trung bình đưa cosine trung bình về $-0{,}0011$ ở cả ba mức lệch tâm trong thí nghiệm. Nâng ngưỡng chỉ xử lý triệu chứng, và thứ hạng giữa các kết quả vẫn bị thành phần chung chi phối.

### Chiều thứ 37 của một embedding mang nghĩa gì?
- [ ] Một đặc trưng ngữ nghĩa cụ thể mà mô hình học được
- [x] Thường không mang nghĩa riêng, vì xoay toàn bộ không gian không thay đổi mọi tích vô hướng
- [ ] Tần suất của từ trong kho ngữ liệu
- [ ] Vị trí của từ trong câu
> Nếu $EE^\top \approx M$ thì $(EQ)(EQ)^\top = EE^\top$ với mọi ma trận trực giao $Q$, nên hệ trục toạ độ là tuỳ ý. Thứ có nghĩa là quan hệ giữa các vector (Mục 3.4).

## Chương 4

### Vì sao trung bình các trạng thái ẩn của BERT chưa phải embedding câu tốt?
- [ ] Vì BERT quá nhỏ
- [x] Vì mô hình được huấn luyện để đoán token bị che, không được huấn luyện để các câu cùng nghĩa có biểu diễn gần nhau
- [ ] Vì trạng thái ẩn có quá nhiều chiều
- [ ] Vì BERT không dùng attention
> Reimers và Gurevych (2019) cho thấy trung bình các vector đầu ra của BERT kém cả trung bình vector GloVe trên các bộ đánh giá độ tương đồng câu. Cần huấn luyện thêm với mục tiêu tương phản (Mục 4.2).

### Trong học tương phản với hàm mất mát InfoNCE, mẫu âm thường lấy từ đâu?
- [ ] Phải thu thập riêng một tập mẫu âm có gán nhãn
- [x] Từ các câu khác trong cùng một lô huấn luyện
- [ ] Từ các từ ngẫu nhiên trong từ vựng
- [ ] Không cần mẫu âm
> Mỗi câu còn lại trong lô là một "lớp sai" trong hàm cross-entropy, nên lô càng lớn thì càng nhiều mẫu âm. Đây là cùng ý tưởng với lấy mẫu âm của word2vec.

### Một kho 10 triệu vector 768 chiều lưu ở FP32 chiếm bao nhiêu bộ nhớ?
- [ ] Khoảng 3 GB
- [x] Khoảng 30,7 GB
- [ ] Khoảng 7,7 GB
- [ ] Khoảng 307 GB
> $10^7 \times 768 \times 4$ byte $= 30{,}7$ GB. Con số này quyết định nhiều lựa chọn kiến trúc: giảm độ chính xác số, nén bằng product quantization, hay chia kho cho nhiều máy (Mục 4.3).

## Chương 5

### Vì sao dự đoán token tiếp theo dạy được nhiều kỹ năng như vậy?
- [ ] Vì mô hình được dạy riêng từng kỹ năng
- [x] Vì muốn dự đoán đúng trong mọi ngữ cảnh, mô hình buộc phải nắm sự kiện, ngữ pháp, phép tính và cấu trúc mã nguồn
- [ ] Vì dữ liệu tiền huấn luyện đã được gán nhãn theo kỹ năng
- [ ] Vì mô hình rất nhiều tham số
> Các kỹ năng là hệ quả của việc giảm mất mát trên một kho văn bản đủ lớn và đa dạng, không được dạy riêng (Mục 5.2).

### Trường hợp nào tiền huấn luyện giúp được ít nhất?
- [ ] Nhiệm vụ phân loại văn bản tiếng Việt
- [x] Miền dữ liệu đích rất xa miền tiền huấn luyện, như tín hiệu cảm biến công nghiệp với một mô hình học trên văn bản web
- [ ] Nhiệm vụ trả lời câu hỏi
- [ ] Nhiệm vụ có ít nhãn
> Khi miền quá xa, các đặc trưng đã học không dùng lại được. Ngược lại, ít nhãn chính là tình huống tiền huấn luyện có ích nhất (Mục 5.4).

## Chương 6

### Trong thí nghiệm ở Mục 6.3, từ khoảng bao nhiêu mẫu có nhãn thì tinh chỉnh toàn phần vượt rõ các cách khác?
- [ ] 20 mẫu
- [x] Khoảng 150 mẫu trở lên
- [ ] 8 000 mẫu
- [ ] Không bao giờ
> Với 20 và 50 mẫu, ba cách không phân biệt được. Từ 150 mẫu, tinh chỉnh toàn phần vượt lên và khoảng cách tăng dần, tới 0,17 so với đóng băng ở 8 000 mẫu.

### Điều gì xảy ra với cách đóng băng khi dữ liệu tăng từ 500 lên 8 000 mẫu?
- [ ] Độ chính xác tăng đều
- [x] Độ chính xác gần như không tăng (0,622; 0,638; 0,645), vì đặc trưng cố định đặt một giới hạn trên
- [ ] Độ chính xác giảm vì overfitting
- [ ] Độ chính xác vượt tinh chỉnh toàn phần
> Khi đặc trưng bị cố định, mô hình chỉ còn là hồi quy softmax trên các đặc trưng đó. Thêm dữ liệu không vượt được giới hạn của đặc trưng; đây là tình trạng độ chệch cao.

### Vì sao thí nghiệm ở Mục 6.2 tự cài mạng bằng NumPy thay cho `MLPClassifier` của scikit-learn?
- [ ] Vì NumPy chạy nhanh hơn
- [x] Vì `MLPClassifier` không cho nạp trọng số ban đầu, nên "tinh chỉnh" bằng nó thực chất là huấn luyện từ đầu
- [ ] Vì scikit-learn không hỗ trợ hàm tanh
- [ ] Vì cần chạy trên GPU
> Tên cấu hình phải khớp với việc mã thực sự làm; nếu không, so sánh giữa hai cột trở nên vô nghĩa.

## Chương 7

### LoRA tiết kiệm điều gì?
- [ ] Thời gian suy luận
- [x] Bộ nhớ khi huấn luyện, chủ yếu là gradient và trạng thái bộ tối ưu, và dung lượng lưu các phiên bản tinh chỉnh
- [ ] Số phép tính khi suy luận
- [ ] Dung lượng của mô hình gốc
> Khi suy luận, $BA$ được gộp vào $W_0$ nên tốc độ bằng mô hình gốc. Với Llama 2 7B, trạng thái Adam giảm từ 48,2 GiB xuống 32 MiB với $r = 8$.

### LoRA dựa trên giả định nào?
- [ ] Mô hình gốc có hạng thấp
- [x] Phần thay đổi cần thiết khi chuyển sang nhiệm vụ mới có hạng thấp
- [ ] Dữ liệu huấn luyện có hạng thấp
- [ ] Mọi ma trận trọng số đều thưa
> Mô hình gốc đã đúng gần hết; phần cần điều chỉnh nằm trong một không gian con ít chiều. Giả định này yếu đi khi nhiệm vụ đòi học kiến thức thật sự mới.

### Vì sao LoRA khởi tạo $B = 0$?
- [ ] Để tiết kiệm bộ nhớ
- [x] Để $BA = 0$ lúc đầu, tức mô hình bắt đầu đúng bằng mô hình gốc
- [ ] Để gradient của $A$ bằng 0
- [ ] Để $A$ và $B$ trực giao
> Nếu khởi tạo cả $A$ và $B$ ngẫu nhiên, ngay từ bước đầu mô hình đã bị cộng một nhiễu ngẫu nhiên vào trọng số.

### Với một ma trận $d \times d$, tỉ lệ số tham số của LoRA hạng $r$ so với tinh chỉnh toàn phần là bao nhiêu?
- [ ] $r/d$
- [x] $2r/d$
- [ ] $r^2/d^2$
- [ ] $2r/d^2$
> LoRA có $2dr$ tham số so với $d^2$. Tỉ lệ giảm khi $d$ tăng: 2,08% với GPT-2 small, 0,39% với một lớp của Llama 2 7B khi $r = 8$.

## Chương 8

### Vì sao không áp dụng trực tiếp ước lượng hợp lý cực đại cho một mô hình mật độ tổng quát trên ảnh?
- [ ] Vì ảnh không có phân phối
- [x] Vì hằng số chuẩn hoá $Z(\theta)$ là tích phân trên không gian rất nhiều chiều, không tính được và phụ thuộc $\theta$
- [ ] Vì hàm mất mát không khả vi
- [ ] Vì cần quá nhiều nhãn
> Các họ mô hình sinh khác nhau ở cách tránh $Z$: mô hình tự hồi quy chuẩn hoá từng bước trên tập nhỏ, VAE dùng chặn dưới, GAN dùng trò chơi, mô hình khuếch tán dùng chuỗi bài hồi quy (Mục 8.2).

### Vì sao mô hình khuếch tán thay thế GAN trong phần lớn ứng dụng sinh ảnh?
- [ ] Vì lấy mẫu nhanh hơn
- [x] Vì đạt độ sắc nét tương đương mà huấn luyện ổn định và phủ phân phối tốt hơn
- [ ] Vì có ít tham số hơn
- [ ] Vì không cần dữ liệu
> Nhược điểm của mô hình khuếch tán là lấy mẫu nhiều bước, và nhược điểm đó cải thiện được bằng DDIM, chưng cất và khuếch tán trong không gian ẩn.

## Chương 9

### Trong thí nghiệm ở Mục 9.3, hệ số $\beta$ của VAE quyết định điều gì?
- [ ] Tốc độ học của bộ mã hoá
- [x] Số chiều ẩn còn mang thông tin: 6 chiều với $\beta = 0$, đúng 2 chiều với $\beta = 1$, không chiều nào với $\beta = 16$
- [ ] Số lớp của bộ giải mã
- [ ] Kích thước lô
> Dữ liệu sinh từ đúng 2 yếu tố, và với $\beta = 1$ VAE giữ đúng 2 chiều mang thông tin; mỗi chiều phải "trả" một chi phí KL, và chỉ những chiều giảm sai số tái dựng đủ nhiều mới được giữ.

### Sụp hậu nghiệm là gì?
- [ ] Bộ giải mã sinh ra mẫu giống hệt nhau
- [x] Hậu nghiệm xấp xỉ trùng với tiên nghiệm ở mọi chiều, nên $z$ không còn mang thông tin về $x$
- [ ] Mô hình quên dữ liệu cũ khi học dữ liệu mới
- [ ] Gradient của bộ mã hoá bằng 0
> Ở $\beta = 16$, mọi chiều có KL bằng 0 và sai số tái dựng tăng lên 15,28, gấp khoảng 125 lần so với $\beta = 0$. Hiện tượng này cũng gặp khi bộ giải mã quá mạnh.

### Vì sao mẫu của VAE thường mờ?
- [ ] Vì mô hình chưa đủ lớn
- [x] Vì hàm mất mát bình phương có nghiệm tối ưu là kỳ vọng có điều kiện, tức trung bình của nhiều ảnh khả dĩ
- [ ] Vì lấy mẫu từ phân phối Gauss
- [ ] Vì số chiều ẩn quá nhỏ
> Trung bình của nhiều ảnh sắc nét lệch nhau vài điểm ảnh là một ảnh nhoè ở các đường viền. Độ mờ là hệ quả của lựa chọn hàm mất mát.

### Mẹo tái tham số hoá dùng để làm gì?
- [x] Cho gradient truyền qua phép lấy mẫu, bằng cách viết $z = \mu + \sigma \odot \varepsilon$ với $\varepsilon$ không phụ thuộc tham số
- [ ] Giảm số tham số của bộ mã hoá
- [ ] Chuẩn hoá không gian ẩn
- [ ] Tăng tốc lấy mẫu khi suy luận
> Đây là kỹ thuật cài đặt để huấn luyện VAE bằng gradient descent, không phải ý tưởng chính của VAE.

## Chương 10

### Đạo hàm của $\log(1 - \sigma(s))$ theo $s$ là gì, và hệ quả với dạng gốc của hàm mất mát cho bộ sinh?
- [ ] $1 - \sigma(s)$; bộ sinh học nhanh lúc đầu
- [x] $-\sigma(s)$; lúc đầu $\sigma(s) \approx 0$ với mẫu giả nên gradient gần 0 và bộ sinh không học được
- [ ] $\sigma(s)$; gradient bùng nổ
- [ ] Bằng 0; hàm mất mát là hằng số
> Ở Mục 10.2, dạng gốc phủ 0 trong 8 cụm ở cả bốn lần khởi tạo. Bộ sinh càng kém càng nhận ít tín hiệu để sửa.

### Dạng không bão hoà của hàm mất mát cho bộ sinh khác gì?
- [x] Cực đại $\log D(G(z))$, có đạo hàm $1 - \sigma(s)$, lớn nhất khi bộ sinh kém nhất
- [ ] Dùng khoảng cách Wasserstein
- [ ] Bỏ bộ phân biệt
- [ ] Huấn luyện bộ sinh nhiều bước hơn bộ phân biệt
> Hai dạng có cùng điểm cố định nhưng gradient khác hẳn; với dạng không bão hoà, cả bốn lần khởi tạo đều phủ đủ 8 cụm.

### Thí nghiệm GAN ở Mục 10.3 cho thấy gì về sụp chế độ?
- [ ] Sụp chế độ xảy ra ở mọi lần khởi tạo
- [x] Không tái hiện được sụp chế độ: dạng không bão hoà phủ đủ 8 cụm, nhưng chỉ 15,3% điểm sinh ra nằm trong phạm vi một cụm
- [ ] GAN luôn học đúng phân phối trong không gian hai chiều
- [ ] Sụp chế độ chỉ xảy ra với dạng không bão hoà
> Phủ đủ các chế độ chưa có nghĩa là học đúng phân phối: phần lớn điểm sinh ra nằm rải rác giữa các cụm. Sụp chế độ là hiện tượng có thật, nhưng một ví dụ hai chiều đơn giản không đủ để nó xuất hiện.

### Vì sao đường cong mất mát của GAN khó dùng để theo dõi chất lượng?
- [ ] Vì mất mát luôn bằng 0
- [x] Vì mất mát của bộ sinh tăng có thể do bộ sinh kém đi hoặc do bộ phân biệt vừa tốt lên; bài toán tìm điểm cân bằng chứ không tìm cực tiểu
- [ ] Vì mất mát chỉ tính được trên tập kiểm tra
- [ ] Vì GAN không có hàm mất mát
> Đây là một trong ba nguyên nhân cấu trúc khiến GAN khó huấn luyện, cùng với việc không có bảo đảm hội tụ tới điểm yên ngựa và không có đánh giá dựa trên hợp lý.

## Chương 11

### Dạng đóng của quá trình thuận cho biết gì, và vì sao nó quan trọng?
- [x] $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0, (1 - \bar\alpha_t)I)$, nên lấy được $x_t$ ở bước bất kỳ trong một phép tính khi huấn luyện
- [ ] $x_t$ luôn là nhiễu thuần tuý
- [ ] Quá trình ngược cũng có dạng đóng
- [ ] Mô hình không cần huấn luyện
> Không có dạng đóng thì mỗi mẫu huấn luyện phải mô phỏng $t$ bước. Ở Mục 11.2, dạng đóng khớp với mô phỏng 200 000 quỹ đạo.

### Ở các bước $t$ lớn (SNR rất thấp), việc khử nhiễu liên quan tới điều gì?
- [ ] Chi tiết nhỏ và kết cấu
- [x] Bố cục và màu sắc tổng thể
- [ ] Không liên quan gì, vì chỉ còn nhiễu
- [ ] Chỉ tới độ sáng trung bình
> Ở $t$ lớn gần như chỉ còn nhiễu, nên việc khử nhiễu quyết định các thành phần biên độ lớn. Khi sinh mẫu đi từ $t = T$ về 0, mô hình quyết định bố cục trước rồi mới thêm chi tiết.

### Huấn luyện mô hình khuếch tán thực chất là bài toán gì?
- [ ] Trò chơi giữa hai mạng
- [x] Hồi quy: dự đoán nhiễu $\varepsilon$ từ $x_t$ và $t$ với mất mát bình phương sai số
- [ ] Phân loại nhiều lớp
- [ ] Học tăng cường
> Mất mát giảm nghĩa là mô hình tốt lên, không có điểm yên ngựa hay hai mạng cạnh tranh. Nhờ vậy mô hình khuếch tán dễ huấn luyện hơn GAN nhiều.

### Vì sao mô hình khuếch tán thường dự đoán nhiễu $\varepsilon$ thay vì dự đoán $x_0$?
- [ ] Vì dự đoán $x_0$ là không thể
- [x] Hai cách tương đương về toán học; dự đoán $\varepsilon$ là lựa chọn thực nghiệm cho chất lượng mẫu tốt hơn
- [ ] Vì $\varepsilon$ có ít chiều hơn
- [ ] Vì dự đoán $\varepsilon$ không cần mạng nơ-ron
> Biết $\varepsilon$ thì suy ra được $x_0$ từ dạng đóng và ngược lại. Ho, Jain và Abbeel (2020) thấy dạng đơn giản hoá với dự đoán nhiễu cho kết quả tốt nhất.

## Chương 12

### Khác biệt có hệ quả lớn nhất giữa học tăng cường và học có giám sát là gì?
- [ ] Học tăng cường dùng mạng nơ-ron sâu hơn
- [x] Dữ liệu do chính chính sách đang học sinh ra, tạo thành vòng phản hồi
- [ ] Học tăng cường không có hàm mất mát
- [ ] Học tăng cường chỉ dùng được trong trò chơi
> Chính sách kém chỉ thu thập được dữ liệu kém, rồi học từ dữ liệu đó lại ra chính sách kém. Ngoài ra, phần thưởng không cho biết hành động đúng là gì và có thể đến rất muộn.

### Thí nghiệm ở Mục 12.4 cho thấy điều gì mà câu "cần $\varepsilon > 0$ để khám phá" bỏ sót?
- [ ] $\varepsilon$ càng lớn càng tốt
- [x] Khởi tạo lạc quan cũng là cơ chế khám phá: với phạt $-0{,}1$ mỗi bước và $Q_0 = 0$, $\varepsilon = 0$ vẫn tới đích 100%
- [ ] Q-learning không cần khám phá
- [ ] Chỉ cần tăng số lượt huấn luyện
> Khi mọi phần thưởng của bước đi đều âm mà $Q$ khởi tạo bằng 0, hành động chưa thử luôn có vẻ tốt hơn hành động đã thử, nên chính sách tham lam tự khám phá.

### Khởi tạo bi quan $Q_0 = -20$ gây ra điều gì trong thí nghiệm?
- [ ] Tác tử khám phá nhiều hơn
- [x] Tác tử bám lấy hành động đầu tiên được thử; ngay cả $\varepsilon = 0{,}3$ cũng không đủ để tới đích trong 6 000 lượt
- [ ] Không khác khởi tạo bằng 0
- [ ] Tác tử luôn rơi vào bẫy
> Hành động đã thử có $Q$ tăng lên gần giá trị thật, còn các hành động khác vẫn ở $-20$, nên chính sách tham lam gần như không thử hành động mới. Chỉ $\varepsilon = 1$ mới học được chính sách tối ưu.

## Chương 13

### Vì sao trừ một đường nền không làm lệch ước lượng gradient chính sách?
- [ ] Vì đường nền luôn bằng 0
- [x] Vì $\mathbb{E}[b\,\nabla_\theta\log\pi_\theta(a)] = b\,\nabla_\theta\sum_a\pi_\theta(a) = b\,\nabla_\theta 1 = 0$
- [ ] Vì đường nền được học cùng chính sách
- [ ] Vì gradient được chuẩn hoá sau khi trừ
> Tổng xác suất luôn bằng 1, nên số hạng thêm vào có kỳ vọng 0 với mọi hằng số $b$. Phép trừ chỉ làm giảm phương sai.

### Giảm độ lệch chuẩn của ước lượng gradient 2,3 lần tương đương cần ít mẫu hơn bao nhiêu lần?
- [ ] 2,3 lần
- [x] Khoảng 5,3 lần
- [ ] 1,5 lần
- [ ] 23 lần
> Số mẫu cần để đạt một độ chính xác tỉ lệ với phương sai, tức bình phương độ lệch chuẩn: $2{,}3^2 \approx 5{,}3$.

### Vì sao phần thưởng luôn dương làm ước lượng REINFORCE nhiễu?
- [ ] Vì gradient bằng 0
- [x] Vì mỗi mẫu đều đẩy log xác suất của hành động được chọn lên, kể cả hành động tệ; việc học chỉ dựa vào chênh lệch nhỏ về độ mạnh
- [ ] Vì phần thưởng dương làm chính sách hội tụ quá nhanh
- [ ] Vì hàm softmax không nhận giá trị dương
> Trừ đường nền gần phần thưởng trung bình làm hành động kém hơn trung bình bị đẩy xuống thay vì được đẩy lên ít hơn.

## Chương 14

### Nghiệm của bài toán $\max_\pi \mathbb{E}_\pi[r] - \beta\,\mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})$ là gì?
- [ ] $\pi^* = \pi_{\text{ref}}$
- [x] $\pi^*(y) \propto \pi_{\text{ref}}(y)\exp(r(y)/\beta)$
- [ ] $\pi^*(y) \propto \exp(r(y))$
- [ ] $\pi^*$ dồn toàn bộ xác suất vào câu trả lời có phần thưởng cao nhất
> Suy ra bằng nhân tử Lagrange với ràng buộc tổng xác suất bằng 1. Ở Mục 14.4, công thức khớp với nghiệm tìm bằng BFGS tới cỡ $10^{-7}$.

### Nếu $\pi_{\text{ref}}(y_0) = 0$ thì $\pi^*(y_0)$ bằng bao nhiêu?
- [ ] Tỉ lệ với $r(y_0)$
- [x] Bằng 0, bất kể $r(y_0)$ lớn tới đâu
- [ ] Bằng $1/N$
- [ ] Không xác định
> Nhân với 0 vẫn bằng 0. Điều này giữ mô hình trong phạm vi hành vi của mô hình tham chiếu, nhưng không ngăn được những hành vi mà mô hình tham chiếu đã có xác suất dương.

### Vì sao học từ so sánh có thể vượt chất lượng của học bắt chước?
- [ ] Vì dữ liệu so sánh nhiều hơn
- [x] Vì đánh giá câu trả lời nào tốt hơn dễ hơn tự viết câu trả lời tốt nhất
- [ ] Vì học tăng cường luôn tốt hơn học có giám sát
- [ ] Vì mô hình thưởng lớn hơn mô hình ngôn ngữ
> Học bắt chước chỉ đưa mô hình tới chất lượng của câu trả lời mẫu; so sánh cho tín hiệu về chất lượng cao hơn mức người gán nhãn tự viết ra.

### Nên hiểu hệ số $\beta$ trong RLHF thế nào?
- [ ] Một siêu tham số có giá trị đúng duy nhất cần tìm
- [x] Một vị trí trên đường đánh đổi giữa phần thưởng đạt được và độ lệch khỏi $\pi_{\text{ref}}$
- [ ] Tốc độ học của PPO
- [ ] Nhiệt độ khi lấy mẫu
> $\beta$ nhỏ cho phần thưởng theo mô hình thưởng cao hơn nhưng đi xa $\pi_{\text{ref}}$ hơn, tăng nguy cơ lách phần thưởng.

## Chương 15

### DPO dựa trên quan sát nào?
- [ ] Mô hình thưởng không cần thiết vì phần thưởng luôn bằng nhau
- [x] Nghiệm dạng đóng của RLHF đảo ngược được, nên mọi chính sách ngầm xác định một hàm thưởng $r = \beta\log(\pi/\pi_{\text{ref}}) + \beta\log Z(x)$
- [ ] Học tăng cường luôn cho kết quả kém
- [ ] Có thể bỏ ràng buộc KL
> Từ đó thay trực tiếp biểu thức của $r$ vào mô hình Bradley–Terry và tối ưu theo chính sách.

### Vì sao $\log Z(x)$ triệt tiêu trong hàm mất mát DPO?
- [ ] Vì $Z(x) = 1$
- [x] Vì mô hình Bradley–Terry chỉ dùng hiệu phần thưởng của hai câu trả lời cho cùng một câu hỏi, và $Z(x)$ không phụ thuộc câu trả lời
- [ ] Vì $Z(x)$ được xấp xỉ bằng lấy mẫu
- [ ] Vì hàm sigmoid bỏ qua hằng số
> Phép triệt tiêu đòi hỏi hai câu trả lời phải cho cùng một câu hỏi $x$; đó là cấu trúc của dữ liệu so sánh.

### DPO có phải một phép xấp xỉ của RLHF không?
- [ ] Có, DPO đơn giản hoá mục tiêu nên cho nghiệm gần đúng
- [x] Không, DPO tối ưu cùng mục tiêu và có cùng nghiệm; ở Mục 15.3 hai chính sách khác nhau không quá $4{,}2 \times 10^{-8}$
- [ ] Có, vì DPO bỏ ràng buộc KL
- [ ] Không so sánh được vì hai phương pháp khác mục tiêu
> Khác biệt thực tế nằm ở dữ liệu dùng khi huấn luyện: DPO học trên tập so sánh cố định, RLHF lấy mẫu từ chính sách hiện tại.

### RLHF với PPO có lợi thế nào so với DPO?
- [ ] Cần ít bộ nhớ hơn
- [x] Học được trên các câu trả lời mà chính sách hiện tại sinh ra, và tạo ra một mô hình thưởng dùng lại được
- [ ] Ổn định hơn khi huấn luyện
- [ ] Không cần dữ liệu so sánh
> DPO đơn giản và ổn định hơn nên là mặc định hợp lý; RLHF có lợi khi chính sách đi xa khỏi phân phối của tập so sánh, hoặc khi cần mô hình thưởng cho đánh giá và giám sát.
