# Ngân hàng câu hỏi tự kiểm tra — Mô hình và kiến trúc

Cú pháp: `## Chương N` mở một nhóm, `### …` là câu hỏi, `- [x]` đánh dấu đáp án đúng,
dòng `>` là phần giải thích hiện ra sau khi trả lời.

## Chương 2

### Thiên lệch và phương sai được định nghĩa qua kỳ vọng trên cái gì?
- [ ] Trên các điểm dữ liệu trong tập kiểm tra
- [x] Trên các tập huấn luyện khác nhau lấy từ cùng phân phối
- [ ] Trên các lần khởi tạo trọng số khác nhau
- [ ] Trên các siêu tham số khác nhau
> Đây là chỗ nhầm phổ biến nhất. Phương sai đo mức **mô hình đổi khi đổi tập huấn luyện**, không phải mức dự đoán dao động giữa các điểm dữ liệu. Nói sai chỗ này là dấu hiệu chỉ học thuộc công thức.

### Sai số huấn luyện 2%, sai số kiểm định 18%, khoảng cách chưa khép khi thêm dữ liệu. Nên làm gì?
- [ ] Dùng mô hình mạnh hơn, thêm đặc trưng
- [x] Thêm dữ liệu, tăng phạt chuẩn, tăng cường dữ liệu, dừng sớm
- [ ] Huấn luyện thêm nhiều vòng nữa
- [ ] Giảm phạt chuẩn để mô hình khớp tốt hơn
> Khoảng cách lớn và chưa khép là dấu hiệu **phương sai cao**. Phương sai giảm khi cỡ mẫu tăng, nên thêm dữ liệu có tác dụng. Đây là trường hợp ngược với thiên lệch cao, nơi thêm dữ liệu vô ích.

### Trong thí nghiệm ở Mục 2.3, vì sao đa thức bậc 1 không cứu được bằng cách thêm dữ liệu?
- [ ] Vì nhiễu quá lớn so với tín hiệu
- [x] Vì 74% MSE của nó là thiên lệch², và thiên lệch không giảm theo cỡ mẫu
- [ ] Vì phương sai của nó đã ở mức sàn
- [ ] Vì đa thức bậc 1 luôn thiếu khớp với mọi dữ liệu
> Thiên lệch là sai lệch của mô hình **trung bình**, nên dù có vô hạn dữ liệu thì đường thẳng vẫn không thành hình sin. Phải đổi lớp hàm. Ngược lại, ở bậc 12 phương sai chiếm gần như toàn bộ MSE nên thêm dữ liệu thì cứu được.

### "Double descent" nói điều gì?
- [ ] Phân rã thiên lệch–phương sai là sai
- [x] Ở chế độ quá tham số, sai số kiểm định giảm lần thứ hai sau đỉnh nội suy
- [ ] Mô hình càng nhiều tham số thì càng ít quá khớp
- [ ] Cần huấn luyện hai lần để có kết quả tốt
> Nó không phủ định phân rã — phân rã là một đồng nhất thức toán học, luôn đúng. Nó chỉ cho thấy đường cong **phương sai theo độ phức tạp** không đơn điệu tăng như bức tranh cổ điển vẽ.

## Chương 3

### Phương sai của trung bình $B$ mô hình có phương sai $\sigma^2$ và tương quan từng cặp $\rho$ là bao nhiêu?
- [ ] $\sigma^2 / B$
- [x] $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$
- [ ] $\rho\sigma^2 / B$
- [ ] $(1-\rho)\sigma^2$
> Cho $B \to \infty$ thì số hạng thứ hai biến mất và **chỉ còn $\rho\sigma^2$**. Đây là toàn bộ lý thuyết của bagging trong một dòng, và nó nói luôn điều phải làm: muốn giảm tiếp phải giảm tương quan — đó đúng là điều random forest thêm vào.

### Boosting giảm thành phần nào của sai số, và vì sao?
- [ ] Phương sai, vì lấy trung bình nhiều mô hình
- [x] Thiên lệch, vì mỗi mô hình mới sửa phần dư của tổng các mô hình trước
- [ ] Nhiễu, vì nó lọc được các điểm ngoại lai
- [ ] Cả ba như nhau
> Boosting là **xuống dốc trong không gian hàm**: mỗi cây mới xấp xỉ gradient âm của hàm mất mát. Vì mỗi cây chỉ phải sửa phần còn thiếu, cây con rất nông — thiên lệch ban đầu cao rồi được hạ dần xuống.

### Phát biểu nào về cây quyết định là đúng?
- [ ] Cây cần chuẩn hoá đặc trưng về cùng thang đo
- [x] Cây bất biến với mọi phép biến đổi đơn điệu áp riêng lên từng đặc trưng
- [ ] Cây không bao giờ quá khớp
- [ ] Độ quan trọng đặc trưng theo độ giảm độ tạp là ước lượng không thiên vị
> Cây chỉ so sánh giá trị với ngưỡng, nên đổi thang đo không đổi thứ tự và không đổi nhát cắt. Phương án cuối sai: độ quan trọng theo độ giảm độ tạp **thiên vị đặc trưng có nhiều giá trị**; permutation importance đáng tin hơn.

### Random forest khác bagging thường ở điểm nào?
- [ ] Dùng boosting thay vì lấy trung bình
- [x] Mỗi nút chỉ xét một tập con ngẫu nhiên các đặc trưng, cốt để giảm tương quan giữa các cây
- [ ] Dùng cây nông hơn
- [ ] Không lấy mẫu bootstrap
> Vì phương sai của trung bình dừng ở $\rho\sigma^2$, giảm $\rho$ là cách duy nhất để đi tiếp. Lấy mẫu đặc trưng ở mỗi nút chính là để làm việc đó.

## Chương 4

### Vì sao mạng nơ-ron cần hàm kích hoạt phi tuyến?
- [ ] Để gradient không tiêu biến
- [x] Vì hợp của hai phép affine vẫn là một phép affine, nên mạng sâu sẽ sụp về một lớp
- [ ] Để đầu ra nằm trong khoảng $[0,1]$
- [ ] Để tăng tốc hội tụ
> $W_2(W_1x + b_1) + b_2 = (W_2W_1)x + (W_2b_1 + b_2)$ — vẫn là affine. Không có phi tuyến thì 100 lớp cũng chỉ mạnh bằng hồi quy tuyến tính.

### Định lý xấp xỉ phổ quát **không** nói điều gì?
- [ ] Mạng một lớp ẩn xấp xỉ được mọi hàm liên tục trên tập compact
- [x] Cần bao nhiêu nơ-ron, và xuống dốc có tìm ra được bộ trọng số ấy hay không
- [ ] Sai số có thể nhỏ tuỳ ý
- [ ] Hàm kích hoạt cần phi tuyến
> Nó là định lý **tồn tại**. Ba điều nó không nói — số nơ-ron cần, khả năng tìm được bằng tối ưu hoá, và lượng dữ liệu cần — lại chính là ba điều quyết định trong thực tế.

### Trong thí nghiệm ở Mục 4.4, mạng một lớp ẩn cần bề rộng bao nhiêu để khớp sóng răng cưa $2^k$ đoạn?
- [ ] Khoảng $2k$
- [ ] Khoảng $k^2$
- [x] Đúng bằng $2^k$
- [ ] Không có bề rộng nào đủ
> Một đơn vị ReLU tạo được đúng một điểm gãy, nên $2^k$ đoạn cần $2^k$ điểm gãy nên cần $2^k$ đơn vị. Mạng sâu chỉ cần $6k$ tham số. Ở $k=7$ là 42 so với 386 tham số — **tuyến tính so với hàm mũ**.

### Kết quả ở Mục 4.4 cho phép kết luận điều gì?
- [ ] Mạng càng sâu càng tốt cho mọi bài toán
- [x] Độ sâu không mua thêm sức biểu diễn mà mua sự gọn gàng, trên những hàm có cấu trúc tự lặp
- [ ] Xuống dốc luôn tìm được lời giải gọn của mạng sâu
- [ ] Mạng một lớp không xấp xỉ được sóng răng cưa
> Hàm răng cưa được chọn vì nó là trường hợp **tốt nhất** cho độ sâu. Và bảng chỉ nói về khả năng **biểu diễn**: huấn luyện mạng 7 lớp × 2 đơn vị trên hàm ấy gần như luôn thất bại — đó là lý do Chương 6 tồn tại.

## Chương 5

### Vì sao lan truyền ngược nhân từ phải sang trái?
- [ ] Vì quy tắc chuỗi chỉ đúng theo thứ tự đó
- [x] Vì hàm mất mát có đầu ra một chiều, nên nhân từ phải cho toàn bộ gradient trong một lượt
- [ ] Vì tiết kiệm bộ nhớ hơn
- [ ] Vì thư viện cài như vậy
> Quy tắc chuỗi đúng theo cả hai thứ tự; cái khác nhau là **chi phí**. Với $f: \mathbb{R}^d \to \mathbb{R}$, chế độ ngược tốn $O(md)$ còn chế độ xuôi tốn $O(md^2)$ vì phải chạy lại một lượt cho mỗi tham số.

### Vì sao Adam có bước hiệu chỉnh thiên lệch?
- [ ] Để bù cho thiên lệch thống kê của mô hình
- [x] Vì khởi tạo $m_0 = v_0 = 0$ kéo các bước đầu về 0 một cách giả tạo
- [ ] Để tránh chia cho 0
- [ ] Để tốc độ học không quá lớn ở cuối
> Với $\beta_1 = 0{,}9$ thì $m_1 = 0{,}1 g_1$ — chỉ bằng 10% của gradient thật. Chia cho $1 - \beta_1^t$ bù đúng phần khuyết ấy, và vì $\beta^t \to 0$ nên hiệu chỉnh tự tắt dần.

### Trạng thái của Adam cho mô hình 7 tỉ tham số ở FP32 chiếm bao nhiêu?
- [ ] 28 GB
- [x] 56 GB
- [ ] 14 GB
- [ ] 112 GB
> Adam giữ **hai** trạng thái ($m$ và $v$) cho mỗi tham số, mỗi cái 4 byte: $7 \times 10^9 \times 2 \times 4 = 56$ GB — gấp đôi bộ nhớ trọng số. Đây là lý do có các biến thể 8-bit và sharding trạng thái tối ưu hoá.

### AdamW khác Adam có suy giảm trọng số $\ell_2$ thường ở đâu?
- [ ] AdamW dùng $\beta$ khác
- [x] AdamW áp suy giảm trực tiếp lên tham số, không cho nó đi qua phép chia $\sqrt{\hat v}$
- [ ] AdamW không dùng momentum
- [ ] Hai cái tương đương về mặt toán học
> Với Adam thường, phạt $\ell_2$ bị cộng vào gradient rồi chia cho $\sqrt{\hat v}$, nên tham số có gradient lớn lại bị phạt **ít** — ngược hẳn ý định của phạt chuẩn.

## Chương 6

### Hệ số 2 trong khởi tạo He ($2/n_{\text{in}}$) đến từ đâu?
- [ ] Từ việc có hai lớp liên tiếp
- [x] Từ việc ReLU đưa một nửa giá trị về 0, tức cắt phương sai đi một nửa
- [ ] Từ đạo hàm của ReLU
- [ ] Từ thực nghiệm, không có lý do lý thuyết
> Vì vậy phải nhân đôi phương sai trọng số để bù lại. Với tanh, hàm không cắt nửa miền giá trị nên không cần hệ số này — đó là khác biệt giữa He và Xavier.

### Trong thí nghiệm ở Mục 6.3, kết nối tắt **một mình** (không chuẩn hoá) cho kết quả gì trên mạng 40 lớp?
- [ ] Gradient ổn định, tỉ lệ khoảng 1
- [ ] Gradient tiêu biến về $10^{-18}$
- [x] Gradient **bùng nổ** tới khoảng $2{,}4 \times 10^{8}$
- [ ] Không đổi so với mạng thường
> Đây là chi tiết hay bị nói sai. Mỗi lớp **cộng thêm** vào tín hiệu, nên gradient tích luỹ chứ không co lại. Phải kết hợp với chuẩn hoá mới ổn định — đo được 2,22, và đó đúng là cấu hình pre-LN.

### Vì sao Transformer dùng LayerNorm chứ không BatchNorm?
- [ ] LayerNorm nhanh hơn
- [x] Độ dài chuỗi thay đổi, và lúc sinh từng token thì lô hiệu dụng bằng 1
- [ ] BatchNorm không hoạt động với ReLU
- [ ] LayerNorm có nhiều tham số hơn
> Ba lý do cộng lại: thống kê theo lô trên trục thời gian vô nghĩa khi độ dài khác nhau; BatchNorm có hành vi **khác nhau** giữa huấn luyện và suy luận; và nó làm dự đoán của một mẫu phụ thuộc các mẫu khác trong lô.

### ResNet sinh ra để chữa vấn đề gì?
- [ ] Quá khớp
- [x] Vấn đề suy thoái: mạng sâu hơn có sai số **huấn luyện** cao hơn mạng nông hơn
- [ ] Gradient bùng nổ
- [ ] Bộ nhớ kích hoạt quá lớn
> Nói "ResNet chữa quá khớp" là tố cáo mình chưa đọc bài báo. Bằng chứng trong bài là sai số **huấn luyện** — không phải kiểm định — tăng theo độ sâu, dù về lý thuyết các lớp thêm chỉ cần học hàm đồng nhất.

### RMSNorm khác LayerNorm ở đâu?
- [ ] Chuẩn hoá theo lô thay vì theo đặc trưng
- [x] Bỏ bước trừ trung bình, chỉ chia cho căn bậc hai của bình phương trung bình
- [ ] Không có tham số học được
- [ ] Đặt sau khối con thay vì trước
> Rẻ hơn vì bớt một lượt quét và bớt phép trừ, mà gần như không mất chất lượng. Là mặc định của Llama và phần lớn mô hình hiện nay.

## Chương 7

### Trường tiếp nhận sau $L$ lớp tích chập $3\times3$ bước nhảy 1 là bao nhiêu?
- [ ] $3L$
- [x] $2L + 1$
- [ ] $3^L$
- [ ] $L^2$
> Từ $r_i = r_{i-1} + (k_i - 1)\prod_{j<i} s_j$ với $k=3$, $s=1$: mỗi lớp thêm 2. Nên phủ hết ảnh $224 \times 224$ cần 112 lớp — đó là lý do phải có bước nhảy và lớp gộp.

### Vì sao trường tiếp nhận **hiệu dụng** nhỏ hơn con số lý thuyết?
- [ ] Vì lớp gộp làm mất thông tin
- [x] Vì ảnh hưởng của các điểm xa trung tâm nhỏ dần theo dạng Gauss, và nó chỉ tăng theo $O(\sqrt{L})$
- [ ] Vì hàm kích hoạt cắt tín hiệu
- [ ] Vì trọng số được khởi tạo nhỏ
> Không phải mọi điểm trong trường lý thuyết đều ảnh hưởng như nhau. Kết quả này là lý do các kiến trúc thực tế dùng bước nhảy, gộp hoặc tích chập giãn cách thay vì chỉ xếp thêm lớp.

### Ba giả thiết quy nạp của tích chập là gì?
- [ ] Tuyến tính, độc lập, đồng nhất phân phối
- [x] Tính cục bộ, chia sẻ trọng số (bất biến dịch chuyển), và cấu trúc phân cấp
- [ ] Trơn, lồi, khả vi
- [ ] Thưa, thấp hạng, có cấu trúc
> Ba giả thiết này đúng với ảnh nên tiết kiệm được rất nhiều dữ liệu. Đó cũng là lý do ViT cần nhiều dữ liệu hơn CNN ở quy mô nhỏ: nó không có sẵn các giả thiết ấy nên phải học từ dữ liệu.

## Chương 8

### Điều gì là giới hạn **thật sự** của RNN mà Transformer phá bỏ?
- [ ] Số phép tính trên mỗi token
- [x] Số bước tuần tự: $O(T)$ so với $O(1)$
- [ ] Số tham số
- [ ] Dung lượng bộ nhớ
> Về FLOP thì attention **đắt hơn**: $O(T^2 d)$ so với $O(Td^2)$. Cái Transformer thắng là song song hoá theo trục thời gian, và đường đi giữa hai vị trí bất kỳ là $O(1)$ thay vì $O(T)$.

### Trong LSTM, gradient dọc theo đường ô nhớ là gì?
- [ ] Tích các ma trận Jacobi như RNN thường
- [x] Tích các **vô hướng** cổng quên $\prod f_t$
- [ ] Một hằng số
- [ ] Tổng các gradient của từng bước
> Vì là tích các vô hướng trong $(0,1)$ chứ không phải tích các ma trận, ta **điều khiển được** nó — cụ thể bằng cách đặt độ lệch cổng quên dương lúc khởi tạo để $f$ khởi đầu gần 1.

### Đo ở Mục 8.3: LSTM với độ lệch cổng quên bằng 1 so với bằng 4, sau 100 bước gradient còn lại bao nhiêu?
- [ ] Hai trường hợp gần như nhau
- [x] $3{,}0 \times 10^{-15}$ so với $1{,}3 \times 10^{-1}$
- [ ] $10^{-3}$ so với $10^{-2}$
- [ ] Cả hai đều bùng nổ
> Chênh 14 bậc độ lớn chỉ do một giá trị khởi tạo. Đây là bằng chứng rằng **cơ chế** của LSTM không phải bản thân cấu trúc cổng mà là việc giữ $f$ gần 1 ở đầu quá trình huấn luyện.

## Chương 9

### Phương sai của $q \cdot k$ khi các thành phần độc lập, trung bình 0, phương sai 1 là bao nhiêu?
- [ ] 1
- [ ] $\sqrt{d_k}$
- [x] $d_k$
- [ ] $d_k^2$
> Vì $q\cdot k = \sum_{i=1}^{d_k} q_i k_i$ là tổng $d_k$ số hạng độc lập, mỗi số hạng có phương sai 1. Đo được ở Mục 9.2: với $d_k = 1024$ thì phương sai đo là 1022. Chia $\sqrt{d_k}$ đưa nó về 1.

### Hậu quả của việc **không** chia $\sqrt{d_k}$ khi $d_k$ lớn là gì?
- [ ] Đầu ra không nằm trong khoảng hợp lệ
- [x] Softmax bão hoà: entropy rơi xuống 0,118 nat trên tối đa 4,159, và gradient gần như bằng 0
- [ ] Mô hình chú ý đều tới mọi vị trí
- [ ] Không có hậu quả gì đáng kể
> Trọng số lớn nhất trung bình đạt 0,953, tức gần như one-hot. Đạo hàm của softmax là $p_i(\delta_{ij} - p_j)$, nên khi $p$ gần 0 hoặc 1 thì gradient tắt. Có chia thì entropy đứng yên ở 3,68 với mọi $d_k$.

### Nhiều đầu attention mua được gì?
- [ ] Nhiều sức tính hơn
- [x] Sự đa dạng của quan hệ, với tổng chi phí xấp xỉ **không đổi** vì mỗi đầu hẹp đi $h$ lần
- [ ] Ngữ cảnh dài hơn
- [ ] Ít tham số hơn
> Bài báo gốc nêu rõ tổng chi phí xấp xỉ bằng attention một đầu ở chiều đầy đủ. Một đầu duy nhất phải **lấy trung bình** các loại quan hệ khác nhau và làm nhoè chúng đi.

### Phần nào trong khối Transformer trộn thông tin giữa các vị trí?
- [ ] Cả attention và FFN
- [x] Chỉ attention; FFN xử lý từng vị trí hoàn toàn độc lập
- [ ] Chỉ FFN
- [ ] Chuẩn hoá lớp
> Nói được điều này cho thấy hiểu cấu trúc chứ không chỉ nhớ sơ đồ. FFN là cùng một MLP áp riêng cho từng token, nên bỏ attention đi thì Transformer chỉ còn là một MLP theo token.

### Vì sao self-attention buộc phải có mã hoá vị trí?
- [ ] Vì softmax cần chuẩn hoá
- [x] Vì nó hoán vị bất biến: đảo thứ tự token thì đầu ra chỉ đảo theo
- [ ] Vì cần giới hạn độ dài chuỗi
- [ ] Vì gradient phụ thuộc thứ tự
> Không có thông tin vị trí thì "mèo đuổi chuột" và "chuột đuổi mèo" cho cùng tập biểu diễn. Với ngôn ngữ thì đó là tai hoạ.

## Chương 10

### Tính chất then chốt của RoPE là gì?
- [ ] Nó không thêm tham số
- [x] $\langle R_m q, R_n k\rangle = \langle R_{m-n} q, k\rangle$ — điểm số chỉ phụ thuộc khoảng cách tương đối
- [ ] Nó làm attention tuyến tính theo độ dài
- [ ] Nó thay thế được chuẩn hoá lớp
> Đo được ở Mục 10.3: các cặp $(5,2)$, $(105,102)$ và $(500,497)$ cho **đúng cùng giá trị 6,851342**. Việc không thêm tham số là món lợi kèm theo, không phải tính chất định nghĩa.

### GQA được dùng vì lý do gì?
- [ ] Để tăng chất lượng mô hình
- [x] Để giảm bộ nhớ KV cache lúc suy luận
- [ ] Để giảm số tham số huấn luyện
- [ ] Để tăng độ dài ngữ cảnh tối đa
> GQA **giảm** chất lượng một chút so với MHA. Đo ở Mục 10.4: mô hình 70B, ngữ cảnh 4096, batch 8 — MHA cần 80 GiB, GQA 8 nhóm cần 10 GiB. Đúng 8 lần, đúng bằng tỉ lệ nhóm.

### Vì sao $d_{\text{ff}}$ của Llama-2 7B là 11 008 chứ không phải $4d = 16\,384$?
- [ ] Để tiết kiệm bộ nhớ
- [x] Vì SwiGLU dùng ba ma trận, nên bề rộng giảm còn $2/3$: $\frac{8}{3} \times 4096 = 10\,923$, làm tròn lên bội của 256
- [ ] Vì mô hình được tỉa bớt sau khi huấn luyện
- [ ] Vì $d_{\text{ff}}$ phải là bội của số đầu attention
> Đây là ví dụ đẹp của việc một quy ước kiến trúc để lại dấu vết kiểm chứng được trong cấu hình mô hình thật. Chênh lệch 11 008 − 10 923 = 85 là phần làm tròn lên cho hợp với phần cứng.

### Mixture of Experts tách được hai thứ nào ra khỏi nhau?
- [ ] Huấn luyện và suy luận
- [x] Số tham số và số phép tính cho mỗi token
- [ ] Attention và FFN
- [ ] Bộ nhớ và độ trễ
> Vì bộ định tuyến chỉ chọn vài chuyên gia cho mỗi token, mô hình có rất nhiều tham số nhưng mỗi token chỉ dùng một phần nhỏ. Cái giá là bộ nhớ phải chứa **toàn bộ** chuyên gia, và cân bằng tải trở thành vấn đề huấn luyện.

## Chương 11

### Vì sao beam search cho văn bản mở thường nhạt và lặp?
- [ ] Vì nó chọn sai token đầu tiên
- [x] Vì nó tối đa hoá xác suất, và văn bản người viết **không** phải văn bản xác suất cao nhất
- [ ] Vì nó quá chậm nên phải cắt ngắn
- [ ] Vì nó không dùng được mặt nạ nhân quả
> Đây là nội dung chính của bài báo về top-$p$: văn bản người viết có entropy dao động, còn chuỗi xác suất cao nhất thì đơn điệu và lặp. Beam search vẫn tốt cho dịch máy, nơi đầu ra bị ràng buộc chặt bởi đầu vào.

### Top-p khác top-k ở điểm bản chất nào?
- [ ] Top-p nhanh hơn
- [x] Kích thước tập ứng viên của top-p **tự thích ứng** theo độ chắc chắn của từng bước
- [ ] Top-p không cần nhiệt độ
- [ ] Top-k cho kết quả xác định
> Đo ở Mục 11.3: với top-$p = 0{,}9$ số token còn sống là 4141 ở bước lưỡng lự, trong khi top-$k = 40$ luôn cố định ở 40 dù mô hình chắc chắn hay không.

## Chương 12

### Số tham số phi-embedding của một Transformer $L$ lớp, chiều $d$ (với $d_{\text{ff}} = 4d$) là bao nhiêu?
- [ ] $4Ld^2$
- [ ] $8Ld^2$
- [x] $12Ld^2$
- [ ] $16Ld^2$
> $4d^2$ cho attention ($W_Q, W_K, W_V, W_O$) cộng $8d^2$ cho FFN ($d \times 4d$ hai lần). Kiểm chứng: GPT-2 small có $12 \times 12 \times 768^2 = 84\,934\,656$, khớp chính xác.

### Trong GPT-2 small, embedding chiếm bao nhiêu phần tổng tham số?
- [ ] Khoảng 5%
- [ ] Khoảng 15%
- [x] Khoảng 31%
- [ ] Khoảng 50%
> Vì $d$ nhỏ mà từ vựng lớn ($50\,257 \times 768$). Tỉ lệ này giảm nhanh theo cỡ mô hình, và đó là lý do các bài báo về quy luật co giãn tách riêng phần phi-embedding.

### Quy tắc $C \approx 6ND$ phân bổ chi phí thế nào?
- [ ] $3N$ xuôi và $3N$ ngược
- [x] $2N$ xuôi và $4N$ ngược, cho mỗi token
- [ ] $N$ xuôi và $5N$ ngược
- [ ] $6N$ xuôi, lượt ngược miễn phí
> Lượt xuôi tốn $2N$ (một nhân và một cộng cho mỗi tham số); lượt ngược tốn gấp đôi vì phải tính gradient theo cả đầu vào và trọng số. Kiểm chứng ở Mục 12.4: Llama-2 7B cho 184 011 giờ-GPU so với 184 320 công bố — lệch 0,17%.

### Chi phí bậc hai của attention chỉ chi phối khi nào?
- [ ] Luôn luôn, vì nó là $O(T^2)$
- [x] Khi $T > 6d$ — với $d = 4096$ là khoảng 24 500 token
- [ ] Khi $T > d$
- [ ] Khi số lớp vượt 32
> Tỉ lệ giữa chi phí attention và phần còn lại là $T/(6d)$. Ở ngữ cảnh 4096 với $d = 4096$, attention chỉ chiếm 17%. Đây là câu trả lời định lượng cho "vì sao chưa cần attention thưa".

### Vì sao hết bộ nhớ khi huấn luyện, và cách chữa đầu tiên là gì?
- [ ] Vì số tham số quá lớn; cách chữa là tỉa mô hình
- [x] Vì bộ nhớ kích hoạt tỉ lệ với cỡ lô × độ dài chuỗi × số lớp; cách chữa đầu tiên là giảm cỡ lô
- [ ] Vì trạng thái Adam quá lớn; cách chữa là dùng SGD
- [ ] Vì phân mảnh bộ nhớ; cách chữa là khởi động lại
> Bộ nhớ kích hoạt thường lớn hơn bộ nhớ trọng số rất nhiều trong lúc huấn luyện, và nó **không** tỉ lệ với số tham số. Sau khi giảm cỡ lô thì tới gradient checkpointing: đổi khoảng 30% thời gian để hạ từ $O(L)$ xuống $O(\sqrt{L})$.
