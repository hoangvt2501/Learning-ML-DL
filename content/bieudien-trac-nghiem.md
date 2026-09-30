# Ngân hàng câu hỏi tự kiểm tra — Biểu diễn, Sinh và Căn chỉnh

Cú pháp: `## Chương N` mở một nhóm, `### …` là câu hỏi, `- [x]` đánh dấu đáp án đúng,
dòng `>` là phần giải thích hiện ra sau khi trả lời.

## Chương 2

### Skip-gram với lấy mẫu âm thực chất đang phân rã cái gì?
- [ ] Ma trận đồng hiện thô
- [x] Ma trận PMI dịch đi một lượng $\log k$
- [ ] Ma trận hiệp phương sai của các từ
- [ ] Không phân rã gì cả, nó là mạng nơ-ron sâu
> Kết quả của Levy và Goldberg (2014). Cho đạo hàm của hàm mục tiêu kỳ vọng bằng 0 sẽ ra $\langle w_i, c_j\rangle = \mathrm{PMI}(i,j) - \log k$. Kiểm chứng ở Mục 2.6: ở hạng đầy đủ, xuống dốc hội tụ về đúng PMI với tương quan **0,9995**.

### Vì sao dùng PMI thay vì đếm đồng hiện thô?
- [ ] Vì PMI dễ tính hơn
- [x] Vì đếm thô bị chi phối bởi từ **phổ biến**; PMI hỏi đúng câu "đồng hiện nhiều hơn mức ngẫu nhiên bao nhiêu"
- [ ] Vì PMI luôn dương nên dễ phân rã
- [ ] Vì PMI không cần chuẩn hoá
> Từ "của" đồng hiện với mọi thứ không phải vì nó liên quan tới mọi thứ. Mẫu số $p(w)p(c)$ của PMI chính là xác suất nếu hai từ độc lập, nên tỉ số đo đúng phần vượt trội.

### Lấy mẫu âm giải quyết vấn đề gì?
- [x] Mẫu số của softmax cộng trên toàn bộ từ vựng cho **mỗi** cặp huấn luyện
- [ ] Dữ liệu huấn luyện có quá nhiều nhiễu
- [ ] Embedding bị quá khớp
- [ ] Gradient bị tiêu biến
> Với $V = 2$ triệu và $d = 300$, softmax đầy đủ tốn 1,2 tỉ phép tính mỗi cặp còn lấy mẫu âm $k=5$ tốn 3 600 — rẻ hơn **333 000 lần**. Điểm mấu chốt: chi phí không còn phụ thuộc $V$.

### Vì sao ở hạng thấp skip-gram lại cho embedding có ích hơn ở hạng đầy đủ?
- [ ] Vì hạng thấp tính nhanh hơn
- [x] Vì nó **không thể** khớp chính xác nên buộc phải chọn giữ lại cái gì — chính sự ép buộc ấy tạo ra khái quát hoá
- [ ] Vì hạng đầy đủ luôn bị quá khớp
- [ ] Vì PMI chỉ có hạng thấp
> Nếu khớp được hoàn hảo thì ta chỉ có một cách viết lại ma trận PMI chứ không có sự khái quát hoá nào. Đây là cùng ý với xấp xỉ hạng thấp ở Chương 14 của *Nền tảng*.

### Mũ 3/4 trong phân phối lấy mẫu âm có cơ sở gì?
- [ ] Suy ra từ lý thuyết thông tin
- [x] **Chọn bằng thực nghiệm** — bài báo gốc nói rõ như vậy
- [ ] Là hệ quả của định lý giới hạn trung tâm
- [ ] Để phân phối trở thành phân phối chuẩn
> Nó chạy tốt hơn mũ 1 và mũ 0; vì sao thì không ai chứng minh. Bịa một lý thuyết cho nó khi phỏng vấn là tự tố cáo.

## Chương 3

### Vì sao hai vector embedding ngẫu nhiên vẫn có thể cho cosine rất cao?
- [x] Vì đám mây embedding **lệch khỏi gốc toạ độ** — hiện tượng bất đẳng hướng
- [ ] Vì số chiều quá cao
- [ ] Vì chúng chưa được chuẩn hoá độ dài
- [ ] Vì cosine luôn dương
> Đo được ở Mục 3.2: 900 vector ngẫu nhiên độc lập cộng một hằng số cho cosine trung bình **0,8724**. Không có quan hệ ngữ nghĩa nào ở đó, chỉ là mọi vector cùng lệch về một hướng.

### Cách chữa rẻ nhất cho bất đẳng hướng là gì?
- [x] Trừ vector trung bình của cả tập trước khi so sánh
- [ ] Nâng ngưỡng điểm cosine lên
- [ ] Giảm số chiều bằng PCA
- [ ] Dùng khoảng cách Euclid thay cosine
> Đo được: cosine trung bình về $-0{,}0011$ ở cả ba mức lệch tâm. Nâng ngưỡng là chữa triệu chứng chứ không chữa nguyên nhân, và sẽ hỏng ngay khi phân bố đổi.

### Chiều thứ 37 của một embedding mang nghĩa gì?
- [ ] Một đặc trưng ngữ nghĩa cụ thể mà mô hình học được
- [x] Thường **không gì cả**, vì bài toán chỉ xác định embedding tới một phép xoay
- [ ] Tần suất của từ
- [ ] Độ dài của từ
> Nếu $EE^\top \approx M$ thì $(EQ)(EQ)^\top = EE^\top$ với mọi ma trận trực giao $Q$. Thứ có nghĩa là **quan hệ** giữa các vector, không phải toạ độ — và đó cũng là lý do phép loại suy hoạt động, vì nó chỉ dùng hiệu vector.

## Chương 4

### Vì sao trạng thái ẩn thô của một mô hình ngôn ngữ không phải embedding câu tốt?
- [x] Vì mô hình được huấn luyện để **đoán token tiếp theo**, không có sức ép nào buộc hai câu cùng nghĩa phải gần nhau
- [ ] Vì số chiều quá nhỏ
- [ ] Vì nó chưa được lượng tử hoá
- [ ] Vì nó chỉ chứa thông tin ngữ pháp
> Cộng thêm việc các biểu diễn ấy bất đẳng hướng nặng. Cách chữa là huấn luyện thêm một bước với mục tiêu tương phản.

### Trong học tương phản, mẫu âm lấy từ đâu?
- [x] Từ **các câu khác trong cùng lô** — nên lô càng lớn càng nhiều mẫu âm
- [ ] Từ một kho mẫu âm chuẩn bị sẵn
- [ ] Sinh ngẫu nhiên
- [ ] Không cần mẫu âm
> Đây cũng là một dạng lấy mẫu âm, đúng tinh thần Chương 2. Đó là lý do các mô hình embedding tốt thường huấn luyện với lô rất lớn.

### Một kho 10 triệu vector 768 chiều ở FP32 chiếm bao nhiêu?
- [ ] 3,1 GB
- [x] **30,7 GB**
- [ ] 307 GB
- [ ] 7,7 GB
> $10^7 \times 768 \times 4 = 30{,}7$ GB. Con số này quyết định phần lớn kiến trúc: nó không vừa RAM máy thường, nên hoặc phải nén hoặc phải chia máy. Tính nó **trước** khi chọn thư viện.

## Chương 5

### Vì sao đoán từ tiếp theo lại dạy được nhiều thứ đến vậy?
- [ ] Vì mô hình thấy rất nhiều dữ liệu
- [x] Vì để đoán đúng trong **mọi** ngữ cảnh, mô hình buộc phải học sự thật về thế giới, ngữ pháp, số học, cấu trúc mã — không cái nào được dạy riêng
- [ ] Vì kiến trúc Transformer rất mạnh
- [ ] Vì dữ liệu web đã được lọc kỹ
> Đây là nhiệm vụ **đại diện**: không phải mục tiêu cuối mà là cái cớ để mô hình phải xây dựng biểu diễn hữu ích về thế giới.

### Tiền huấn luyện hỏng khi nào?
- [x] Khi miền đích quá xa miền tiền huấn luyện, hoặc nhiệm vụ đích cần thứ mà nhiệm vụ đại diện không đụng tới
- [ ] Khi có quá nhiều dữ liệu có nhãn
- [ ] Khi dùng tốc độ học nhỏ
- [ ] Khi mô hình quá lớn
> Mô hình học trên văn bản web không giúp được mấy cho tín hiệu cảm biến công nghiệp. Và nó giỏi ngữ pháp nhưng không giỏi số học nhiều chữ số, vì đoán từ tiếp theo hiếm khi đòi tính toán chính xác.

## Chương 6

### Ở bao nhiêu mẫu có nhãn thì tinh chỉnh toàn phần tách hẳn lên so với đóng băng?
- [ ] Ngay từ 20 mẫu
- [x] Từ khoảng **150 mẫu**, và khoảng cách giãn dần tới 0,17 ở 8 000 mẫu
- [ ] Chỉ khi có trên 100 000 mẫu
- [ ] Không bao giờ — đóng băng luôn thắng
> Ở 20 và 50 mẫu, ba cách nằm trong khoảng nhiễu của nhau và **không kết luận được gì**. Kết luận từ hàng đầu tiên của một bảng như vậy là một lỗi thường gặp.

### Hiện tượng gì xảy ra với cột đóng băng khi dữ liệu tăng từ 500 lên 8 000 mẫu?
- [ ] Nó tiếp tục tăng đều
- [x] Nó **phẳng ra** (0,6220 → 0,6375 → 0,6452) — đây là trần chặn do đặc trưng cố định đặt ra
- [ ] Nó giảm vì quá khớp
- [ ] Nó vượt cột tinh chỉnh
> Khi đặc trưng bị cố định, mô hình chỉ còn là hồi quy softmax trên chúng. Nhận ra trần này có giá trị ngay: nếu thêm dữ liệu không giúp gì thì thêm nữa cũng vô ích — phải mở đóng băng ra.

### Vì sao thí nghiệm ở Mục 6.2 phải tự cài mạng bằng NumPy thay vì dùng scikit-learn?
- [x] Vì scikit-learn **không cho nạp trọng số ban đầu**, nên "tinh chỉnh" hoá ra lại là huấn luyện từ đầu — so sánh vô nghĩa
- [ ] Vì scikit-learn chạy chậm
- [ ] Vì cần GPU
- [ ] Vì scikit-learn không hỗ trợ nhiều lớp
> Lần đầu làm sai đúng chỗ này, và hai cột cho số gần như y hệt (0,6975 so với 0,6987) — dấu hiệu rõ ràng rằng phép so sánh hỏng.

## Chương 7

### LoRA tiết kiệm cái gì?
- [ ] Thời gian suy luận
- [x] **Bộ nhớ lúc huấn luyện** — trạng thái Adam 32 MiB so với 48,2 GiB
- [ ] Dung lượng dữ liệu huấn luyện
- [ ] Số lớp của mô hình
> Khi suy luận thì gộp $BA$ vào $W_0$ nên tốc độ **bằng** bản gốc. Chênh lệch bộ nhớ là 1 540 lần — đó là khác biệt giữa "chạy được trên một GPU tiêu dùng" và "cần một cụm máy chủ".

### Giả thiết cốt lõi của LoRA là gì?
- [x] Phần **cần sửa** khi chuyển sang nhiệm vụ mới có hạng thấp
- [ ] Ma trận trọng số gốc có hạng thấp
- [ ] Dữ liệu đích có ít chiều
- [ ] Gradient luôn thưa
> Chú ý phân biệt: LoRA **không** giả định $W_0$ hạng thấp — nó giả định $\Delta W$ hạng thấp. Nhiệm vụ đích thường chỉ đòi nhấn mạnh lại thứ mô hình đã biết, và loại chỉnh sửa ấy đúng là ít chiều.

### Vì sao khởi tạo $B = 0$?
- [x] Để $BA = 0$ lúc bắt đầu, nên mô hình **khởi đầu đúng bằng bản gốc** rồi đi dần ra
- [ ] Để tiết kiệm bộ nhớ
- [ ] Để gradient lớn hơn
- [ ] Để tránh quá khớp
> Nếu khởi tạo cả $A$ và $B$ ngẫu nhiên thì ngay bước đầu đã phá mô hình bằng một phần thêm vào ngẫu nhiên.

### Tỉ lệ tham số LoRA trên tham số toàn phần của một lớp là bao nhiêu?
- [x] $2r/d$ — nên mô hình càng lớn thì LoRA càng lợi
- [ ] $r/d^2$
- [ ] $r^2/d$
- [ ] $2r/d^2$
> Với $d = 768$ và $r = 8$ là 2,08%; với $d = 8192$ cùng $r$ là 0,195%. Đây là lý do LoRA đặc biệt hợp với mô hình rất lớn.

## Chương 8

### Vì sao không dùng thẳng hợp lý cực đại cho bài toán sinh?
- [x] Vì hằng số chuẩn hoá $Z(\theta)$ là tích phân trên toàn bộ không gian, không tính được và **phụ thuộc $\theta$**
- [ ] Vì hợp lý cực đại chỉ dùng cho phân loại
- [ ] Vì dữ liệu sinh không có nhãn
- [ ] Vì hợp lý cực đại luôn bị quá khớp
> Ba họ mô hình sinh là ba cách né nó: VAE tối đa một chặn dưới, GAN thay bằng trò chơi, khuếch tán biến thành chuỗi bài hồi quy.

### Vì sao khuếch tán thay thế được GAN từ khoảng 2021?
- [ ] Vì nó lấy mẫu nhanh hơn
- [x] Vì nó giữ được độ sắc nét **mà không phải trả giá bằng sự bất ổn** — và chậm thì còn tối ưu được chứ bất ổn thì không
- [ ] Vì nó cần ít dữ liệu hơn
- [ ] Vì nó có ít tham số hơn
> Khuếch tán lấy mẫu **chậm hơn** GAN hàng trăm lần. Nhưng huấn luyện nó là một bài hồi quy bình thường với mất mát đọc được.

## Chương 9

### $\beta$ trong VAE thực sự điều khiển cái gì?
- [ ] Chỉ là hệ số cân giữa hai số hạng, không có ý nghĩa cụ thể hơn
- [x] **Số chiều ẩn còn mang thông tin** — ở $\beta=1$ mô hình tự tìm ra đúng số yếu tố thật
- [ ] Tốc độ hội tụ
- [ ] Kích thước không gian ẩn
> Đo được với dữ liệu sinh từ đúng 2 yếu tố, VAE cho 6 chiều ẩn: $\beta=0$ giữ cả 6, $\beta=1$ giữ **đúng 2**, $\beta=16$ giữ 0. Số hạng KL hoạt động như một phép chọn số chiều tự động.

### Sụp hậu nghiệm là gì?
- [x] $q(z\mid x)$ trùng với tiên nghiệm, tức $z$ **không còn mang thông tin gì** về $x$
- [ ] Bộ giải mã không hội tụ
- [ ] Không gian ẩn bị quá khớp
- [ ] KL trở nên vô hạn
> Đo được ở $\beta=16$: cả 6 chiều đều chết, sai số tái dựng nhảy lên 15,28 — tệ hơn 125 lần so với $\beta=0$. Mô hình đã "thắng" số hạng KL bằng cách vứt bỏ toàn bộ thông tin.

### Vì sao mẫu của VAE thường mờ?
- [x] Vì bình phương sai lệch tối ưu ở **kỳ vọng có điều kiện**, mà trung bình của nhiều ảnh sắc nét lệch nhau thì nhoè
- [ ] Vì mô hình chưa đủ lớn
- [ ] Vì không gian ẩn quá nhỏ
- [ ] Vì huấn luyện chưa đủ lâu
> Độ mờ là **hệ quả trực tiếp của việc chọn hàm mất mát**, tức của giả định nhiễu Gauss. Không phải lỗi huấn luyện. Đổi giả định thì đổi hiện tượng — và đó chính là điều GAN làm.

### Mẹo tái tham số hoá dùng để làm gì?
- [x] Để gradient chảy được qua phép lấy mẫu: viết $z = \mu + \sigma\odot\varepsilon$ nên phần ngẫu nhiên nằm ở $\varepsilon$ vốn không phụ thuộc tham số
- [ ] Để giảm phương sai của ELBO
- [ ] Để bảo đảm hậu nghiệm là Gauss
- [ ] Để tăng tốc huấn luyện
> Đây là một **mẹo cài đặt**, không phải ý tưởng của VAE. Không nên nhầm hai thứ khi trả lời phỏng vấn.

## Chương 10

### Đạo hàm của $\log(1-\sigma(s))$ theo $s$ bằng bao nhiêu, và hậu quả là gì?
- [x] $-\sigma(s)$ — nên khi bộ sinh còn tệ ($\sigma(s)\approx 0$) thì gradient cũng gần 0
- [ ] $1-\sigma(s)$ — gradient lớn khi bộ sinh tệ
- [ ] $\sigma(s)(1-\sigma(s))$ — luôn nhỏ
- [ ] $-1/\sigma(s)$ — gradient bùng nổ
> Đây là vòng luẩn quẩn: bộ sinh càng tệ càng ít tín hiệu để sửa. Đo được ở Mục 10.2: dạng minimax gốc phủ **0/8** chế độ ở cả bốn hạt giống.

### Bản "không bão hoà" khác ở đâu?
- [x] Bộ sinh **tối đa** $\log D(G(z))$, đạo hàm là $1-\sigma(s)$ — lớn nhất đúng khi bộ sinh đang tệ nhất
- [ ] Nó thêm một số hạng phạt chuẩn
- [ ] Nó dùng bộ phân biệt mạnh hơn
- [ ] Nó đổi kiến trúc bộ sinh
> Hai hàm có **cùng điểm tối ưu** nhưng hành vi huấn luyện khác một trời một vực. Bài học tổng quát: cái quyết định việc học là **gradient**, không phải vị trí điểm tối ưu.

### Thí nghiệm GAN trong tài liệu này cho thấy gì về sụp chế độ?
- [ ] Nó xác nhận sụp chế độ xảy ra ở mọi hạt giống
- [x] Nó **không** quan sát được sụp chế độ — bản không bão hoà phủ đủ 8/8 ở mọi hạt giống, và tài liệu nói rõ điều đó
- [ ] Nó cho thấy sụp chế độ chỉ xảy ra với lô nhỏ
- [ ] Nó không đo được gì về sụp chế độ
> Sụp chế độ là hiện tượng có thật và hay gặp, nhưng khẳng định nó từ một thí nghiệm hai chiều mà chính thí nghiệm ấy bác bỏ thì là nói quá. Điều thí nghiệm **có** cho thấy: chỉ 15,3% điểm sinh ra rơi vào phạm vi một cụm.

### Vì sao không đọc được đường cong mất mát của GAN?
- [x] Mất mát của bộ sinh tăng có thể vì nó tệ đi, **mà cũng có thể** vì bộ phân biệt vừa giỏi lên
- [ ] Vì mất mát luôn bằng 0
- [ ] Vì thư viện không ghi lại
- [ ] Vì mất mát dao động quá nhanh
> Cộng thêm hai lý do cấu trúc: đây là bài toán **điểm yên ngựa** chứ không phải cực tiểu, và không có cách đánh giá khách quan như chặn dưới hợp lý.

## Chương 11

### Dạng đóng của quá trình thuận nói gì, và vì sao nó quan trọng?
- [x] $q(x_t\mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$ — cho phép **nhảy thẳng** tới bước $t$ bất kỳ thay vì mô phỏng $t$ bước
- [ ] Nó cho biết cách đi ngược lại
- [ ] Nó bảo đảm quá trình hội tụ
- [ ] Nó chỉ dùng để phân tích, không dùng khi huấn luyện
> Không có nó thì huấn luyện chậm gấp hàng trăm lần. Kiểm chứng trên 200 000 quỹ đạo mô phỏng thật, khớp mô men tới 4–5 chữ số.

### Ở $t$ lớn (SNR thấp) mô hình học gì?
- [ ] Chi tiết nhỏ và kết cấu
- [x] **Bố cục thô và màu tổng thể**
- [ ] Không học gì vì chỉ còn nhiễu
- [ ] Học cách phân loại ảnh
> Nên khi lấy mẫu và đi ngược từ $t=T$ về $t=0$, mô hình quyết định bố cục trước rồi mới điền chi tiết — giống cách người ta vẽ tranh. Điều đó không được lập trình vào; nó là hệ quả của lịch nhiễu.

### Huấn luyện mô hình khuếch tán thực chất là bài toán gì?
- [x] **Hồi quy bình phương sai lệch**: đoán lại $\varepsilon$ đã thêm vào
- [ ] Một trò chơi hai bên như GAN
- [ ] Tối đa một chặn dưới như VAE
- [ ] Học tăng cường
> Đó là lý do nó ổn định: không có điểm yên ngựa, mất mát giảm nghĩa là mô hình tốt lên, và đọc được đường cong mất mát.

### Vì sao đoán $\varepsilon$ thay vì đoán thẳng $x_0$?
- [x] Hai cách tương đương về toán, nhưng đoán $\varepsilon$ cho mục tiêu có **phương sai đồng đều hơn giữa các $t$** nên ổn định hơn
- [ ] Vì $x_0$ không biết khi huấn luyện
- [ ] Vì đoán $\varepsilon$ chính xác hơn về mặt lý thuyết
- [ ] Vì $\varepsilon$ có ít chiều hơn
> Biết một cái thì suy ra cái kia từ dạng đóng. Đây là lựa chọn **thực nghiệm**, và bài báo DDPM nói rõ như vậy.

## Chương 12

### Khác biệt nặng nhất giữa học tăng cường và học có giám sát là gì?
- [ ] Phần thưởng là số thay vì nhãn
- [x] **Dữ liệu do chính chính sách sinh ra** — chính sách tồi chỉ thu được dữ liệu tồi, rồi lại học ra chính sách tồi
- [ ] Phần thưởng có thể đến trễ
- [ ] Không có tập kiểm tra
> Đây là một vòng phản hồi, cùng bản chất với vòng phản hồi thoái hoá ở Chương 12 của *MLOps*. Nó là gốc của mọi khó khăn về sau.

### Thí nghiệm ở Mục 12.4 cho thấy điều gì mà câu "phải có ε để khám phá" bỏ sót?
- [x] **Khởi tạo hàm giá trị cũng là một cơ chế khám phá** — với khởi tạo lạc quan, $\varepsilon = 0$ vẫn đạt 100%
- [ ] ε lớn luôn tốt hơn ε nhỏ
- [ ] Q-learning không cần khám phá
- [ ] Chỉ cần chạy đủ lâu là được
> Khởi tạo $Q = 0$ trong khi mọi phần thưởng đều âm nghĩa là hành động chưa thử luôn trông hấp dẫn hơn hành động đã thử, nên tham lam tự đi khám phá. Bỏ phạt bước đi thì mất tính lạc quan ấy và $\varepsilon = 0$ thất bại 0%.

### Khởi tạo bi quan ($Q_0 = -20$) gây ra chuyện gì?
- [x] Tác tử **bám chặt** lấy hành động đầu tiên tình cờ thử; ngay cả $\varepsilon = 0{,}3$ cũng không đủ để thoát
- [ ] Tác tử khám phá nhiều hơn
- [ ] Không ảnh hưởng gì
- [ ] Thuật toán không hội tụ
> Hành động đầu tiên được thử sẽ được kéo **lên** gần 0 trong khi các hành động khác vẫn ở $-20$. Bi quan tạo ra sự cố chấp.

## Chương 13

### Vì sao trừ một đường nền không làm lệch ước lượng gradient?
- [x] Vì $\mathbb{E}[b\nabla\log\pi] = b\nabla\big(\sum_a\pi(a)\big) = b\nabla 1 = 0$ với **mọi** hằng số $b$
- [ ] Vì đường nền được chọn bằng cách tối thiểu phương sai
- [ ] Vì nó nhỏ so với phần thưởng
- [ ] Nó **có** làm lệch, nhưng lệch ít
> Tổng xác suất luôn bằng 1 nên đạo hàm của nó bằng 0. Ta được đổi phương sai mà không mất gì — đo được: giảm độ lệch chuẩn 2,3 lần, gradient trung bình lệch chỉ $1{,}2\times10^{-3}$.

### Giảm độ lệch chuẩn 2,3 lần tương đương tiết kiệm bao nhiêu mẫu?
- [ ] 2,3 lần
- [x] Khoảng **5,3 lần**, vì số mẫu cần tỉ lệ với **bình phương** độ lệch chuẩn
- [ ] 1,5 lần
- [ ] Không tiết kiệm mẫu, chỉ ổn định hơn
> $2{,}3^2 = 5{,}3$. Đây là lý do đường nền gần như luôn được dùng trong mọi cài đặt gradient chính sách thực tế.

### Vì sao phần thưởng toàn dương làm ước lượng gradient nhiễu?
- [x] Vì ta **đẩy log xác suất của mọi hành động lên**, kể cả hành động tệ; việc học chỉ diễn ra nhờ chênh lệch tương đối vốn chìm trong nhiễu
- [ ] Vì gradient bị tràn số
- [ ] Vì softmax bão hoà
- [ ] Vì phần thưởng dương không hợp lệ
> Trừ đường nền biến phần thưởng thành "tốt hơn hay tệ hơn mức trung bình", đúng thứ ta cần.

## Chương 14

### Nghiệm tối ưu của $\max_\pi \mathbb{E}_\pi[r] - \beta\mathrm{KL}(\pi\|\pi_{\text{ref}})$ là gì?
- [x] $\pi^*(y) \propto \pi_{\text{ref}}(y)\exp(r(y)/\beta)$ — chính sách tham chiếu **đánh trọng số lại theo hàm mũ của thưởng**
- [ ] $\pi^*(y) \propto \exp(r(y)/\beta)$, không phụ thuộc $\pi_{\text{ref}}$
- [ ] $\pi^*(y) \propto \pi_{\text{ref}}(y) + r(y)/\beta$
- [ ] Không có dạng đóng
> Suy ra bằng nhân tử Lagrange trên ràng buộc $\sum_y\pi(y)=1$. Kiểm chứng bằng số: giải theo hai đường độc lập cho kết quả lệch $10^{-9}$ tới $10^{-7}$.

### Nếu $\pi_{\text{ref}}(y_0) = 0$ thì $\pi^*(y_0)$ bằng bao nhiêu?
- [x] **0**, bất kể $r(y_0)$ lớn tới đâu — vì nhân với 0 vẫn là 0
- [ ] Bằng $\exp(r(y_0)/\beta)/Z$
- [ ] Không xác định
- [ ] Bằng $\beta$
> Đây chính là cơ chế giữ cho mô hình không nói năng lung tung: những gì chính sách tham chiếu coi là không thể thì vẫn không thể.

### Vì sao học từ so sánh vượt được trần của học bắt chước?
- [x] Vì **đánh giá dễ hơn sáng tạo** — người ta chỉ ra được bài nào hay hơn dù không viết nổi bài hay nhất
- [ ] Vì dữ liệu so sánh nhiều hơn
- [ ] Vì mô hình thưởng mạnh hơn mô hình ngôn ngữ
- [ ] Vì RLHF dùng nhiều tính toán hơn
> Cộng thêm một lý do thực tế: so sánh **nhất quán giữa những người dán nhãn** hơn là chấm điểm tuyệt đối. Hỏi mười người "mấy điểm trên 10" thì được mười câu trả lời khác nhau.

### Nên hiểu $\beta$ của RLHF thế nào?
- [ ] Một siêu tham số cần chỉnh cho đúng
- [x] **Vị trí ta chọn trên một đường đánh đổi** giữa thưởng đạt được và độ lệch khỏi $\pi_{\text{ref}}$
- [ ] Tốc độ học của PPO
- [ ] Trọng số của mô hình thưởng
> Muốn thưởng cao hơn thì phải chấp nhận lệch xa $\pi_{\text{ref}}$ hơn, mà lệch xa hơn nghĩa là rủi ro lách điểm thưởng cao hơn. Không có giá trị nào "đúng".

## Chương 15

### Quan sát then chốt của DPO là gì?
- [x] Dạng đóng **đảo ngược được**: mọi chính sách đều ngầm định nghĩa một hàm thưởng $r = \beta\log\frac{\pi}{\pi_{\text{ref}}} + \beta\log Z$
- [ ] Mô hình thưởng không cần thiết vì nó không chính xác
- [ ] PPO có thể thay bằng xuống dốc thường
- [ ] Dữ liệu so sánh có thể chuyển thành dữ liệu có nhãn
> Không cần huấn luyện mô hình thưởng riêng — chính sách **đã là** một mô hình thưởng, viết ở dạng khác.

### Vì sao $\log Z(x)$ triệt tiêu trong hàm mất mát DPO?
- [x] Vì hợp lý Bradley–Terry chỉ dùng **hiệu** hai phần thưởng, mà $\log Z(x)$ phụ thuộc $x$ chứ không phụ thuộc $y$
- [ ] Vì nó nhỏ nên bỏ qua được
- [ ] Vì nó được xấp xỉ bằng lấy mẫu
- [ ] Vì nó bằng 1 khi chuẩn hoá
> Chi tiết nhỏ ấy là thứ làm cả phương pháp chạy được. Và nó đòi hỏi hai câu trả lời phải cho **cùng một** câu hỏi $x$ — nếu khác $x$ thì không triệt tiêu.

### DPO có phải một xấp xỉ của RLHF không?
- [ ] Có, nó đơn giản hoá nên kém chính xác hơn
- [x] **Không** — nó tối ưu cùng mục tiêu và cho cùng nghiệm, kiểm chứng được tới $4{,}2\times10^{-8}$
- [ ] Không, nó tối ưu một mục tiêu hoàn toàn khác
- [ ] Có, nó chỉ đúng khi $\beta$ nhỏ
> Thí nghiệm ở Mục 15.3 giải cả hai đường trên không gian 8 câu trả lời với 52 588 cặp so sánh. Lệch lớn nhất giữa hai chính sách là $4{,}16\times10^{-8}$.

### RLHF vẫn hơn DPO ở điểm nào?
- [x] Nó **lấy mẫu từ chính sách hiện tại** nên học trên đúng phân phối mô hình đang sinh ra; và mô hình thưởng là tài sản dùng lại được
- [ ] Nó luôn cho chất lượng cao hơn
- [ ] Nó cần ít bộ nhớ hơn
- [ ] Nó ổn định hơn
> DPO chỉ học trên tập so sánh **cố định**, nên khi chính sách đi xa khỏi phân phối của tập ấy thì tín hiệu học yếu đi. Và mô hình thưởng còn dùng để chấm dữ liệu mới, đánh giá mô hình khác, dò suy giảm chất lượng.
