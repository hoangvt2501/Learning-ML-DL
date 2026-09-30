# Từ điển thuật ngữ — Biểu diễn, Sinh và Căn chỉnh

Cú pháp: `## Nhóm`, rồi `### Tiếng Việt | English` và phần định nghĩa bên dưới.
Mọi định nghĩa bám sát đúng cách dùng trong giáo trình.

## Biểu diễn và embedding

### Biểu diễn dày | dense representation
Vector số thực ngắn (vài chục tới vài trăm chiều) thay cho one-hot thưa. Khác one-hot ở chỗ **khoảng cách giữa hai vector có nghĩa**, còn one-hot thì mọi cặp cách nhau đúng $\sqrt2$.

### Giả thiết phân bố | distributional hypothesis
Một từ được đặc trưng bởi những từ đi kèm nó. Đây là phép đổi một câu hỏi **không đo được** ("hai từ có gần nghĩa không?") lấy một câu hỏi **đo được** ("chúng có hay xuất hiện trong ngữ cảnh giống nhau không?").

### Thông tin tương hỗ điểm | pointwise mutual information, PMI
$\log \frac{p(w,c)}{p(w)p(c)}$ — đo mức đồng hiện **vượt trên** mức ngẫu nhiên. Bằng 0 nghĩa là đồng hiện đúng như thể hai từ độc lập. PMI dương cắt về 0 gọi là **PPMI**.

### Skip-gram | skip-gram
Mô hình word2vec đoán các từ ngữ cảnh từ một từ trung tâm. **Không phải học sâu** — nó chỉ có một lớp và không có phi tuyến ở giữa.

### Lấy mẫu âm | negative sampling
Thay bài toán "chọn đúng 1 trong $V$" bằng $k+1$ bài nhị phân. Điểm mấu chốt: chi phí **không còn phụ thuộc kích thước từ vựng**. Với $V = 2$ triệu thì nó rẻ hơn softmax đầy đủ 333 000 lần.

### Kết quả Levy–Goldberg | implicit matrix factorization
Chứng minh rằng skip-gram với lấy mẫu âm ngầm phân rã ma trận $\mathrm{PMI} - \log k$. Kiểm chứng được: ở hạng đầy đủ, xuống dốc hội tụ về đúng PMI với tương quan 0,9995.

### Phép loại suy vector | vector analogy
$a : b :: c : ?$ giải bằng $\vec b - \vec a + \vec c$. Nó hoạt động được vì chỉ dùng **hiệu vector**, mà hiệu vector bất biến với phép xoay không gian.

### Bất đẳng hướng | anisotropy
Tình trạng đám mây embedding lệch khỏi gốc toạ độ, làm mọi cặp vector đều có cosine dương lớn. Đo được: 900 vector **ngẫu nhiên độc lập** vẫn cho cosine trung bình 0,8724. Trừ vector trung bình đi thì về $-0{,}0011$.

### Làm trắng | whitening
Trừ trung bình rồi nhân $\Sigma^{-1/2}$ để chuẩn hoá phương sai theo mọi hướng. Mạnh hơn việc chỉ trừ trung bình, nhưng cần ước lượng ma trận hiệp phương sai.

### Bất biến với phép xoay | rotation invariance
Nếu $EE^\top \approx M$ thì $(EQ)(EQ)^\top = EE^\top$ với mọi ma trận trực giao $Q$. Hệ quả: **trục toạ độ của embedding hoàn toàn tuỳ tiện**, nên từng chiều riêng lẻ thường không mang nghĩa gì.

### Embedding theo ngữ cảnh | contextual embedding
Gán vector cho **một lần xuất hiện của từ trong một câu cụ thể** thay vì cho chuỗi ký tự. Đây là cách duy nhất xử lý được từ đa nghĩa.

### Học tương phản | contrastive learning
Huấn luyện để cặp cùng nghĩa gần nhau và cặp khác nghĩa xa nhau. Hàm mục tiêu có dạng y hệt entropy chéo softmax, với các câu khác trong cùng lô đóng vai trò mẫu âm.

### Tìm kiếm láng giềng gần xấp xỉ | approximate nearest neighbour search, ANN
Bỏ yêu cầu trả về chính xác $k$ mục gần nhất, đổi lấy tốc độ nhanh hơn hàng trăm lần. Cần thiết vì lời nguyền số chiều làm các cấu trúc cổ điển như cây k-d thoái hoá về tìm kiếm tuyến tính.

### HNSW | hierarchical navigable small world
Chỉ mục ANN dạng đồ thị nhiều tầng, đi tham lam từ tầng thô xuống tầng mịn. Nhanh và chính xác nhất hiện nay, đổi lại tốn bộ nhớ vì phải lưu cả đồ thị.

### Lượng tử hoá tích | product quantization, PQ
Chia vector thành nhiều đoạn rồi lượng tử hoá từng đoạn bằng một bảng mã nhỏ. Chính là uniform affine quantization áp cho từng đoạn, với cùng đánh đổi giữa sai số và dung lượng.

### recall@k | recall@k
Tỉ lệ kết quả trả về thuộc về $k$ kết quả đúng thật sự. Lưu ý: **recall@k cao không đảm bảo hệ thống hữu ích** — nếu embedding không nắm được thứ người dùng coi là liên quan thì tìm chính xác 100% vẫn vô dụng.

## Học chuyển giao

### Tiền huấn luyện | pre-training
Huấn luyện trên nhiệm vụ mà **nhãn tự sinh từ chính dữ liệu** (đoán từ tiếp theo, che từ rồi đoán), nên dùng được dữ liệu không nhãn vốn gần như miễn phí.

### Nhiệm vụ đại diện | pretext task, proxy task
Nhiệm vụ không phải mục tiêu cuối, được đặt ra để ép mô hình xây dựng biểu diễn hữu ích. Đoán từ tiếp theo là nhiệm vụ đại diện rộng nhất hiện biết.

### Đóng băng | freezing
Giữ nguyên trọng số một phần mô hình, không cập nhật. Thắng khi rất ít nhãn, nhưng đặt ra một **trần chặn** mà thêm dữ liệu không phá được.

### Trần chặn của đặc trưng đóng băng | frozen-feature ceiling
Hiện tượng độ chính xác phẳng ra dù thêm dữ liệu, vì mô hình chỉ còn là hồi quy softmax trên các đặc trưng cố định. Đo được: 0,6220 → 0,6375 → 0,6452 khi dữ liệu tăng từ 500 lên 8 000. Đây là chẩn đoán thiên lệch cao ở một dạng khác.

### Tinh chỉnh | fine-tuning
Cập nhật toàn bộ trọng số từ điểm khởi đầu là mô hình tiền huấn luyện. Thắng từ khoảng 150 mẫu trở lên, và khoảng cách giãn dần theo lượng dữ liệu.

### Quên tai hại | catastrophic forgetting
Mô hình mất khả năng ở những việc nó từng làm được sau khi tinh chỉnh trên miền hẹp. Cách chữa: trộn dữ liệu miền gốc vào, hoặc dùng LoRA vốn giữ nguyên trọng số gốc theo đúng nghĩa đen.

### LoRA | low-rank adaptation
Đóng băng $W_0$, học phần thêm vào $\Delta W = BA$ với $A$ là $d\times r$, $B$ là $r\times d$. Giả thiết cốt lõi: phần cần sửa cho nhiệm vụ mới có **hạng thấp**. Tỉ lệ tham số là $2r/d$, nên mô hình càng lớn thì LoRA càng lợi.

### Khởi tạo B = 0 | zero-init of B
Chi tiết cài đặt của LoRA: khởi tạo $B = 0$ để $BA = 0$ lúc bắt đầu, nên mô hình khởi đầu **đúng bằng bản gốc** rồi đi dần ra. Khởi tạo cả hai ngẫu nhiên thì phá mô hình ngay bước đầu.

### Gộp adapter | merging
Cộng $BA$ vào $W_0$ một lần khi suy luận. Vì vậy LoRA **không làm chậm cũng không làm nhanh** mô hình — nó chỉ tiết kiệm bộ nhớ lúc huấn luyện.

### PEFT | parameter-efficient fine-tuning
Gọi chung các cách tinh chỉnh chỉ cập nhật một phần rất nhỏ tham số: LoRA, adapter, prefix tuning, BitFit.

### QLoRA | QLoRA
LoRA trên một mô hình gốc đã lượng tử hoá 4 bit. Chạy được vì mô hình gốc **chỉ đọc chứ không cập nhật**, nên lượng tử hoá nó không ảnh hưởng tới việc huấn luyện adapter.

## Mô hình sinh

### Đa tạp dữ liệu | data manifold
Tập con rất nhỏ của không gian mà dữ liệu thật nằm trên đó. Với ảnh $256\times256$, số điểm khả dĩ là $256^{196608}$ nhưng ảnh trông như thật chỉ chiếm một phần cực nhỏ.

### Hằng số chuẩn hoá | partition function, normalising constant
$Z(\theta) = \int \exp(f_\theta(x))dx$ — tích phân trên toàn bộ không gian, không tính được. Đây là thứ chặn đường dùng thẳng hợp lý cực đại cho bài toán sinh, và ba họ mô hình sinh là ba cách né nó.

### ELBO | evidence lower bound
Chặn dưới của log hợp lý, gồm số hạng tái dựng trừ đi số hạng KL. VAE tối đa nó thay vì tối đa hợp lý trực tiếp.

### Mẹo tái tham số hoá | reparameterisation trick
Viết $z = \mu + \sigma\odot\varepsilon$ với $\varepsilon\sim\mathcal{N}(0,I)$ để gradient chảy được qua phép lấy mẫu. Là **mẹo cài đặt**, không phải ý tưởng của VAE.

### Sụp hậu nghiệm | posterior collapse
Trạng thái $q(z\mid x)$ trùng với tiên nghiệm, tức $z$ không còn mang thông tin gì về $x$. Đo được ở $\beta = 16$: cả 6 chiều ẩn đều chết, sai số tái dựng nhảy lên 15,28.

### Chiều ẩn còn sống | active latent dimension
Chiều có KL riêng lớn hơn 0 đáng kể. Số chiều còn sống là thứ $\beta$ thực sự điều khiển: ở $\beta=1$, mô hình giữ lại **đúng** số yếu tố thật sinh ra dữ liệu.

### Bộ sinh, bộ phân biệt | generator, discriminator
Hai mạng của GAN. Điểm đáng chú ý: **bộ sinh không bao giờ nhìn thấy dữ liệu thật** — mọi thông tin về "thế nào là thật" đi qua một kênh duy nhất là ý kiến của bộ phân biệt.

### Mất mát bão hoà | saturating loss
Dạng minimax gốc: bộ sinh tối thiểu $\log(1-D(G(z)))$. Đạo hàm là $-\sigma(s)$, gần 0 đúng lúc bộ sinh đang tệ nhất. Đo được: **phủ 0/8 chế độ** ở mọi hạt giống.

### Mất mát không bão hoà | non-saturating loss
Bộ sinh tối đa $\log D(G(z))$. Cùng điểm tối ưu nhưng đạo hàm là $1-\sigma(s)$, **lớn nhất khi bộ sinh đang tệ nhất**. Đo được: phủ 8/8.

### Sụp chế độ | mode collapse
Bộ sinh chỉ phủ một phần các chế độ của phân phối thật. Hiện tượng có thật và hay gặp, nhưng **không quan sát được** trong thí nghiệm hai chiều của tài liệu này — và tài liệu nói rõ điều đó thay vì khẳng định.

### Quá trình thuận | forward process
Chuỗi bước thêm nhiễu Gauss của mô hình khuếch tán. Vì mọi bước đều Gauss tuyến tính nên gộp được thành **dạng đóng** $q(x_t\mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$.

### Dạng đóng của quá trình thuận | closed form of the forward process
Chi tiết làm cho khuếch tán huấn luyện được: cho phép nhảy thẳng tới bước $t$ bất kỳ thay vì mô phỏng $t$ bước. Kiểm chứng trên 200 000 quỹ đạo, khớp mô men tới 4–5 chữ số.

### Tỉ số tín hiệu trên nhiễu | signal-to-noise ratio, SNR
$\bar\alpha_t/(1-\bar\alpha_t)$. Cho biết mô hình học gì ở mỗi bước: $t$ nhỏ (SNR cao) thì học chi tiết và kết cấu, $t$ lớn (SNR thấp) thì học bố cục thô.

### Đoán nhiễu | noise prediction
Cách đặt mục tiêu của DDPM: mạng đoán lại $\varepsilon$ đã thêm vào, mất mát là bình phương sai lệch. Tương đương về toán với đoán $x_0$, nhưng cho phương sai đồng đều hơn giữa các $t$ nên huấn luyện ổn định hơn.

### Khuếch tán trong không gian ẩn | latent diffusion
Chạy toàn bộ quá trình trong không gian ẩn nén của một VAE. Ví dụ đẹp của việc ghép hai họ mô hình: VAE lo phần nén, khuếch tán lo phần sinh.

## Học tăng cường và căn chỉnh

### Chính sách | policy
$\pi(a\mid s)$ — quy tắc chọn hành động. Với mô hình ngôn ngữ, chính sách **chính là** mô hình.

### Hàm giá trị hành động | action-value function, Q
Tổng phần thưởng **kỳ vọng trong tương lai** nếu làm $a$ tại $s$ rồi chơi tốt về sau. Chữ "trong tương lai" là chìa khoá: một hành động có thể cho thưởng tức thì tệ mà vẫn đúng.

### Phương trình Bellman | Bellman equation
$Q(s,a) = \mathbb{E}[r + \gamma\max_{a'}Q(s',a')]$. Q-learning biến nó thành quy tắc cập nhật.

### Sai số thời gian | temporal-difference error
Chênh lệch giữa điều vừa quan sát và điều đang tin, tức phần trong ngoặc của quy tắc cập nhật Q-learning.

### Khám phá và khai thác | exploration–exploitation
Đánh đổi giữa thử cái mới và tận dụng cái đã biết. **Không tồn tại trong học có giám sát**, vì ở đó nhãn được cho sẵn chứ không phải đi tìm.

### Khởi tạo lạc quan | optimistic initialisation
Khởi tạo $Q$ cao hơn giá trị thật, làm hành động chưa thử luôn trông hấp dẫn hơn hành động đã thử. **Là một cơ chế khám phá hoàn chỉnh** — đo được rằng nó cho 100% thành công ngay cả với $\varepsilon = 0$.

### REINFORCE | REINFORCE
Thuật toán gradient chính sách: lấy mẫu hành động, xem thưởng bao nhiêu, đẩy log xác suất của nó lên với cường độ tỉ lệ thưởng.

### Thủ thuật log | log-derivative trick
$\nabla_\theta\pi_\theta(a) = \pi_\theta(a)\nabla_\theta\log\pi_\theta(a)$. Nó biến gradient của một kỳ vọng thành một kỳ vọng của gradient, nên ước lượng được bằng lấy mẫu.

### Đường nền | baseline
Hằng số trừ khỏi phần thưởng trong REINFORCE. **Không làm lệch ước lượng** vì $\mathbb{E}[b\nabla\log\pi] = b\nabla 1 = 0$, nhưng cắt độ lệch chuẩn đi 2,3 lần — tức cần ít hơn 5,3 lần số mẫu.

### Hàm lợi thế | advantage function
$A(s,a) = Q(s,a) - V(s)$ — đường nền tốt nhất phụ thuộc trạng thái. Trả lời đúng câu hỏi cần hỏi: hành động này tốt hơn mức trung bình bao nhiêu.

### PPO | proximal policy optimisation
Thuật toán actor–critic có **giới hạn mức thay đổi chính sách mỗi lần cập nhật**, vì bước quá lớn trong không gian chính sách có thể phá hỏng mọi thứ mà không quay lại được.

### Mô hình Bradley–Terry | Bradley–Terry model
$P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))$ — hồi quy logistic trên hiệu hai điểm thưởng. Hàm thưởng chỉ xác định **tới một hằng số cộng**, nhưng điều đó vô hại vì hằng số bị hằng số chuẩn hoá nuốt mất.

### RLHF | reinforcement learning from human feedback
Quy trình ba giai đoạn, trong đó giai đoạn cuối học từ **so sánh** thay vì ví dụ mẫu. Đây là chỗ mô hình vượt được chất lượng của người viết mẫu, vì **đánh giá dễ hơn sáng tạo**.

### Lách điểm thưởng | reward hacking
Mô hình khai thác điểm mù của mô hình thưởng thay vì thực sự trả lời tốt hơn. Xảy ra vì mô hình thưởng chỉ là một xấp xỉ học từ dữ liệu hữu hạn.

### Ràng buộc KL | KL constraint
Số hạng $-\beta\,\mathrm{KL}(\pi\|\pi_{\text{ref}})$ chặn mô hình đi quá xa bản gốc. Nghiệm tối ưu có dạng đóng $\pi^* \propto \pi_{\text{ref}}\exp(r/\beta)$ — tức **chính sách tham chiếu đánh trọng số lại theo hàm mũ của thưởng**.

### Hệ số beta của RLHF | RLHF beta
**Không phải** siêu tham số cần chỉnh đúng, mà là **vị trí ta chọn trên một đường đánh đổi** giữa thưởng đạt được và độ lệch khỏi $\pi_{\text{ref}}$. Đừng lẫn với $\beta$ của VAE.

### DPO | direct preference optimisation
Tối ưu thẳng trên chính sách, bỏ hẳn mô hình thưởng. Chạy được vì dạng đóng đảo ngược được thành $r = \beta\log\frac{\pi}{\pi_{\text{ref}}} + \beta\log Z$, và $\log Z$ **triệt tiêu** khi lấy hiệu hai câu trả lời cho cùng một câu hỏi. **Không phải xấp xỉ** — kiểm chứng được là cho cùng nghiệm tới $4{,}2\times10^{-8}$.

### IPO, KTO | IPO, KTO
Hai hướng sau DPO. IPO thêm phạt chuẩn để chặn hiện tượng đẩy tỉ lệ xác suất ra vô cùng khi dữ liệu so sánh gần như tất định. KTO bỏ yêu cầu dữ liệu phải là cặp, chỉ cần nhãn tốt/tệ cho từng câu riêng lẻ.
