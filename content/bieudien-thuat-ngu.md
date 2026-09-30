# Từ điển thuật ngữ — Biểu diễn, mô hình sinh và căn chỉnh

Cú pháp: `## Nhóm`, rồi `### Tiếng Việt | English` và phần định nghĩa bên dưới.
Mỗi định nghĩa khớp với cách dùng trong giáo trình. Thuật ngữ nào giáo trình giữ nguyên tiếng Anh thì tên tiếng Anh đứng trước.

## Biểu diễn và embedding

### Embedding | embedding
Vector số thực, thường vài trăm chiều, biểu diễn một từ, một câu hay một đối tượng, sao cho các đối tượng giống nhau có vector gần nhau. Khác với mã hoá one-hot, trong đó mọi cặp từ cách nhau như nhau.

### Giả thuyết phân bố | distributional hypothesis
Các từ xuất hiện trong những ngữ cảnh giống nhau thường có nghĩa giống nhau (Harris, 1954). Là cơ sở của mọi phương pháp học embedding từ thống kê đồng hiện (Mục 2.1).

### Thông tin tương hỗ điểm | pointwise mutual information, PMI
$\log\frac{p(w,c)}{p(w)p(c)}$: hai từ cùng xuất hiện nhiều hơn mức kỳ vọng nếu độc lập bao nhiêu. PPMI là PMI với các giá trị âm thay bằng 0 (Mục 2.3).

### Skip-gram | skip-gram
Mô hình của word2vec học hai vector cho mỗi từ, một khi là từ trung tâm và một khi là từ ngữ cảnh, sao cho tích vô hướng lớn với các cặp từ cùng xuất hiện (Mục 2.5).

### Lấy mẫu âm | negative sampling
Thay softmax trên toàn bộ từ vựng bằng $k + 1$ bài toán phân loại nhị phân: cặp thật và $k$ cặp ghép ngẫu nhiên. Chi phí mỗi cặp huấn luyện không phụ thuộc kích thước từ vựng.

### Phân rã ma trận ngầm | implicit matrix factorization
Kết quả của Levy và Goldberg (2014): ở điểm tối ưu, skip-gram với lấy mẫu âm cho $\langle w_i, c_j\rangle = \mathrm{PMI}(i,j) - \log k$, tức word2vec phân rã ma trận PMI mà không lập ma trận đó (Mục 2.6).

### Phép loại suy | analogy
Kiểm tra embedding bằng câu hỏi $a : b :: c : ?$, tìm từ có vector gần nhất với $\vec b - \vec a + \vec c$. Chỉ dùng vector hiệu nên không phụ thuộc phép xoay của không gian.

### Tính bất đẳng hướng | anisotropy
Hiện tượng đám mây embedding lệch khỏi gốc toạ độ, khiến mọi cặp vector có cosine cao dù không liên quan. Ở Mục 3.2, 900 vector ngẫu nhiên lệch tâm có cosine trung bình 0,87.

### Làm trắng | whitening
Trừ trung bình rồi nhân với $\Sigma^{-1/2}$ để phương sai theo mọi hướng bằng nhau. Mạnh hơn trừ trung bình nhưng cần ước lượng ma trận hiệp phương sai.

### Embedding theo ngữ cảnh | contextual embedding
Vector gán cho một lần xuất hiện của từ trong một câu cụ thể, do Transformer tạo ra, nên cùng một từ có vector khác nhau trong các ngữ cảnh khác nhau (Mục 4.2).

### Học tương phản | contrastive learning
Huấn luyện embedding sao cho cặp liên quan gần nhau và cặp không liên quan xa nhau, thường bằng hàm mất mát InfoNCE với các mẫu khác trong lô làm mẫu âm.

### Tìm kiếm láng giềng gần đúng | approximate nearest neighbor search, ANN
Tìm các vector gần truy vấn mà không bảo đảm tìm đúng $k$ vector gần nhất, đổi lại nhanh hơn nhiều lần so với quét toàn bộ. Các họ phổ biến: IVF, HNSW, product quantization (Mục 4.3).

### Product quantization | product quantization, PQ
Chia vector thành nhiều đoạn và lượng tử hoá mỗi đoạn bằng một bảng mã nhỏ, nén vector xuống vài chục byte với sai số chấp nhận được.

### Recall@k của chỉ mục | index recall@k
Tỉ lệ trong $k$ kết quả trả về thuộc về $k$ kết quả đúng tìm bằng quét toàn bộ. Đo chất lượng của chỉ mục, khác với chất lượng của bản thân embedding (Mục 4.4).

## Học chuyển giao

### Tiền huấn luyện | pre-training
Huấn luyện một mô hình lớn trên dữ liệu không nhãn bằng một nhiệm vụ mà nhãn tự sinh ra từ dữ liệu, như đoán token tiếp theo, trước khi dùng cho các nhiệm vụ cụ thể (Chương 5).

### Học tự giám sát | self-supervised learning
Học từ dữ liệu không nhãn bằng những nhiệm vụ có nhãn tự sinh, ví dụ che một từ rồi đoán lại, hoặc đoán từ tiếp theo.

### Đóng băng | freezing
Giữ nguyên trọng số của một phần mô hình khi huấn luyện, thường là phần trích đặc trưng, và chỉ học các lớp phía trên.

### Tinh chỉnh | fine-tuning
Tiếp tục huấn luyện một mô hình đã tiền huấn luyện trên dữ liệu của nhiệm vụ cụ thể, thường với tốc độ học nhỏ hơn 10 tới 100 lần so với huấn luyện từ đầu.

### Giới hạn của đặc trưng đóng băng | frozen-feature ceiling
Khi đặc trưng bị cố định, thêm dữ liệu chỉ giúp ước lượng lớp cuối chính xác hơn, không vượt được giới hạn do đặc trưng đặt ra. Ở Mục 6.3, độ chính xác của cách đóng băng gần như không tăng từ 500 mẫu trở đi.

### Quên thảm hoạ | catastrophic forgetting
Hiện tượng mô hình mất khả năng ở những việc từng làm được sau khi tinh chỉnh trên dữ liệu mới.

### LoRA | low-rank adaptation
Đóng băng $W_0$ và học phần cộng thêm $\Delta W = BA$, với $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$ và $r \ll d$; $B$ khởi tạo bằng 0 để mô hình bắt đầu đúng bằng mô hình gốc (Mục 7.2).

### Gộp adapter | merging
Cộng $BA$ vào $W_0$ sau khi huấn luyện để suy luận với một ma trận duy nhất, nên LoRA không làm chậm mô hình khi suy luận.

### Tinh chỉnh tiết kiệm tham số | parameter-efficient fine-tuning, PEFT
Các phương pháp chỉ học một phần nhỏ tham số: LoRA, adapter, prefix tuning, prompt tuning, BitFit.

### QLoRA | QLoRA
LoRA trên mô hình gốc đã lượng tử hoá 4 bit. Mô hình gốc chỉ được đọc nên lượng tử hoá không cản trở việc huấn luyện adapter.

## Mô hình sinh

### Đa tạp dữ liệu | data manifold
Tập con rất nhỏ của không gian dữ liệu chứa các mẫu trông giống thật, thường được xem như có số chiều thấp hơn nhiều so với không gian chứa nó.

### Hằng số chuẩn hoá | normalizing constant, partition function
$Z(\theta) = \int\exp(f_\theta(x))\,dx$, cần để một hàm tuỳ ý trở thành mật độ xác suất. Không tính được trong không gian nhiều chiều, nên ngăn áp dụng trực tiếp hợp lý cực đại (Mục 8.2).

### Mô hình tự hồi quy | autoregressive model
Mô hình viết $p(x) = \prod_t p(x_t \mid x_{<t})$, chỉ cần chuẩn hoá từng bước trên một tập nhỏ như từ vựng. Các mô hình ngôn ngữ thuộc loại này.

### ELBO | evidence lower bound
Chặn dưới của log hợp lý: $\mathbb{E}_{q(z \mid x)}[\log p(x \mid z)] - \mathrm{KL}(q(z \mid x) \,\|\, p(z))$. Hàm mục tiêu của VAE (Mục 9.2).

### Tái tham số hoá | reparameterization trick
Viết $z = \mu + \sigma \odot \varepsilon$ với $\varepsilon \sim \mathcal{N}(0, I)$ để gradient truyền qua phép lấy mẫu.

### Sụp hậu nghiệm | posterior collapse
Hậu nghiệm xấp xỉ trùng tiên nghiệm ở mọi chiều, nên biến ẩn không mang thông tin về dữ liệu. Xảy ra khi $\beta$ quá lớn hoặc bộ giải mã quá mạnh.

### Chiều ẩn mang thông tin | active latent dimension
Chiều ẩn có KL giữa hậu nghiệm và tiên nghiệm lớn hơn 0 đáng kể. Ở Mục 9.3, số chiều này giảm từ 6 xuống 2 rồi 0 khi $\beta$ tăng.

### Bộ sinh và bộ phân biệt | generator, discriminator
Hai mạng của GAN: bộ sinh biến nhiễu thành mẫu, bộ phân biệt cho xác suất một mẫu là thật. Bộ phân biệt tối ưu là $p_{\text{data}}/(p_{\text{data}} + p_g)$.

### Hàm mất mát không bão hoà | non-saturating loss
Cho bộ sinh cực đại $\log D(G(z))$ thay vì cực tiểu $\log(1 - D(G(z)))$. Gradient $1 - \sigma(s)$ lớn khi bộ sinh còn kém, nên bộ sinh học được ngay từ đầu (Mục 10.2).

### Sụp chế độ | mode collapse
Bộ sinh chỉ sinh ra một phần của phân phối dữ liệu, bỏ qua các chế độ khác.

### Quá trình thuận | forward process
Chuỗi các bước thêm nhiễu Gauss $q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1 - \beta_t}\,x_{t-1}, \beta_t I)$, có dạng đóng $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0, (1 - \bar\alpha_t)I)$ (Mục 11.2).

### Tỉ số tín hiệu trên nhiễu | signal-to-noise ratio, SNR
$\bar\alpha_t/(1 - \bar\alpha_t)$ ở bước $t$ của quá trình khuếch tán. SNR cao ứng với chi tiết nhỏ, SNR thấp ứng với bố cục tổng thể (Mục 11.3).

### Dự đoán nhiễu | noise prediction
Hàm mục tiêu của mô hình khuếch tán: hồi quy $\|\varepsilon - \varepsilon_\theta(x_t, t)\|^2$.

### Khuếch tán trong không gian ẩn | latent diffusion
Nén ảnh bằng một bộ tự mã hoá rồi chạy khuếch tán trong không gian ẩn nhỏ hơn, giảm chi phí mỗi bước.

### Classifier-free guidance | classifier-free guidance
Kết hợp dự đoán có điều kiện và không điều kiện, $\varepsilon(\varnothing) + w\,(\varepsilon(c) - \varepsilon(\varnothing))$ với $w > 1$, để mẫu bám điều kiện chặt hơn (Mục 11.6).

## Học tăng cường và căn chỉnh

### Chính sách | policy
Quy tắc chọn hành động $\pi(a \mid s)$ theo trạng thái. Một mô hình ngôn ngữ là một chính sách chọn token tiếp theo.

### Hàm giá trị hành động | action-value function
$Q(s, a)$: tổng phần thưởng chiết khấu kỳ vọng nếu làm $a$ tại $s$ rồi tiếp tục theo chính sách.

### Phương trình Bellman | Bellman equation
$Q^*(s, a) = \mathbb{E}[r + \gamma\max_{a'}Q^*(s', a')]$: giá trị của hành động bằng phần thưởng nhận ngay cộng giá trị chiết khấu của hành động tốt nhất ở trạng thái kế tiếp.

### Q-learning | Q-learning
Cập nhật $Q(s, a)$ theo sai số thời gian $r + \gamma\max_{a'}Q(s', a') - Q(s, a)$. Là thuật toán ngoài chính sách: học giá trị của chính sách tham lam bất kể dữ liệu thu bằng chính sách nào.

### Khám phá và khai thác | exploration, exploitation
Cân bằng giữa thử hành động mới để biết thêm và chọn hành động tốt nhất theo hiểu biết hiện tại. $\varepsilon$-tham lam là cách thông dụng nhất.

### Khởi tạo lạc quan | optimistic initialization
Khởi tạo hàm giá trị cao hơn giá trị thật để các hành động chưa thử trông hấp dẫn, tạo ra khám phá mà không cần yếu tố ngẫu nhiên (Mục 12.4).

### Gradient chính sách | policy gradient
$\nabla_\theta J = \mathbb{E}_{a \sim \pi_\theta}[R(a)\,\nabla_\theta\log\pi_\theta(a)]$: tối ưu trực tiếp tham số của chính sách theo phần thưởng kỳ vọng. REINFORCE ước lượng nó bằng lấy mẫu.

### Đường nền | baseline
Hằng số $b$ trừ khỏi phần thưởng trong gradient chính sách. Không đổi kỳ vọng của ước lượng mà giảm phương sai; ở Mục 13.4, độ lệch chuẩn giảm khoảng 2,3 lần.

### Hàm lợi thế | advantage function
$A(s, a) = Q(s, a) - V(s)$: hành động tốt hơn mức trung bình ở trạng thái đó bao nhiêu.

### PPO | proximal policy optimization
Thuật toán gradient chính sách giới hạn mức thay đổi của chính sách mỗi lần cập nhật bằng cách cắt tỉ số xác suất mới trên cũ. Được dùng trong RLHF.

### Mô hình Bradley–Terry | Bradley–Terry model
$P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))$: xác suất chọn câu trả lời này hơn câu kia phụ thuộc hiệu phần thưởng. Là hồi quy logistic trên hiệu điểm (Mục 14.3).

### RLHF | reinforcement learning from human feedback
Huấn luyện mô hình thưởng từ dữ liệu so sánh của con người, rồi tối ưu mô hình ngôn ngữ theo mô hình thưởng có ràng buộc KL với mô hình tham chiếu (Chương 14).

### Lách phần thưởng | reward hacking
Chính sách khai thác điểm yếu của mô hình thưởng để đạt điểm cao mà không thật sự tốt hơn. Ràng buộc KL là một cách hạn chế.

### Nghiệm có ràng buộc KL | KL-constrained optimum
$\pi^*(y \mid x) \propto \pi_{\text{ref}}(y \mid x)\exp(r(x, y)/\beta)$: chính sách tham chiếu đánh trọng số lại theo hàm mũ của phần thưởng (Mục 14.4).

### DPO | direct preference optimization
Tối ưu trực tiếp chính sách trên dữ liệu so sánh bằng hàm mất mát cross-entropy nhị phân trên hiệu các tỉ số log xác suất, không cần mô hình thưởng hay học tăng cường; có cùng nghiệm tối ưu với RLHF (Chương 15).

### IPO, KTO | IPO, KTO
Các biến thể sau DPO: IPO sửa hàm mất mát để tránh đẩy tỉ số xác suất ra vô cùng khi dữ liệu so sánh gần tất định; KTO chỉ cần nhãn tốt hoặc không tốt cho từng câu trả lời riêng lẻ.
