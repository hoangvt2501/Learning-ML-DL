# Từ điển thuật ngữ — Mô hình và kiến trúc

Cú pháp: `## Nhóm`, rồi `### Tiếng Việt | English` và phần định nghĩa bên dưới.
Mọi định nghĩa bám sát đúng cách dùng trong giáo trình.

## Sai số và tổng quát hoá

### Thiên lệch | bias (statistical)
Sai lệch có hệ thống giữa dự đoán **trung bình** của mô hình (trung bình lấy trên các tập huấn luyện khác nhau) và giá trị thật. Thiên lệch cao đồng nghĩa giả thiết về dạng hàm quá hẹp. Đừng lẫn với **độ lệch** — tham số cộng $b$ trong $Wx + b$; tiếng Anh dùng chung một chữ *bias* cho cả hai.

### Phương sai | variance
Mức dao động của mô hình khi đổi tập huấn luyện. Phương sai cao đồng nghĩa mô hình đủ linh hoạt để bám cả nhiễu.

### Thiếu khớp | underfitting
Trạng thái thiên lệch cao: sai số huấn luyện và sai số kiểm định đều cao và **sát nhau**. Thêm dữ liệu không cứu được.

### Quá khớp | overfitting
Trạng thái phương sai cao: sai số huấn luyện thấp, sai số kiểm định cao, **khoảng cách lớn**. Thêm dữ liệu thì cứu được.

### Nhiễu không giảm được | irreducible error, Bayes error
Phần sai số do bản thân dữ liệu, không mô hình nào vượt qua được. Là trần trên của mọi cố gắng.

### Đường cong học | learning curve
Đồ thị sai số huấn luyện và sai số kiểm định theo cỡ tập huấn luyện. Là công cụ chẩn đoán thiên lệch/phương sai dùng được trong việc thật, không cần chạy nhiều tập huấn luyện.

### Double descent | double descent
Hiện tượng ở chế độ **quá tham số**: sau đỉnh nội suy, sai số kiểm định **giảm lần thứ hai** khi mô hình tiếp tục lớn lên. Nó không phủ định phân rã thiên lệch–phương sai mà chỉ cho thấy đường cong phương sai không đơn điệu như bức tranh cổ điển.

### Giả thiết quy nạp | inductive bias
Tập các giả định mà kiến trúc áp lên bài toán trước khi thấy dữ liệu. Tích chập giả định tính cục bộ và bất biến dịch chuyển; attention gần như không giả định gì. Giả thiết đúng thì tiết kiệm dữ liệu; giả thiết sai thì thành trần chặn.

## Học tập hợp

### Bagging | bootstrap aggregating
Huấn luyện $B$ mô hình trên $B$ mẫu bootstrap rồi lấy trung bình. Phương sai của trung bình là $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$, nên khi $B \to \infty$ chỉ còn $\rho\sigma^2$ — muốn giảm tiếp phải giảm **tương quan** giữa các mô hình.

### Rừng ngẫu nhiên | random forest
Bagging cộng thêm việc chỉ xét một tập con ngẫu nhiên các đặc trưng ở mỗi nút, cốt để giảm $\rho$. Thêm cây thì không bao giờ làm tệ đi.

### Boosting | boosting
Huấn luyện tuần tự, mỗi mô hình mới xấp xỉ gradient âm của hàm mất mát tại tổ hợp hiện tại — tức **xuống dốc trong không gian hàm**. Vì mỗi cây chỉ sửa phần còn thiếu, cây con rất nông và cơ chế chính là **giảm thiên lệch**.

### Độ tạp Gini | Gini impurity
$1 - \sum_k p_k^2$. Xấp xỉ bậc hai của entropy, rẻ hơn vì không cần tính logarit; hai tiêu chí gần như luôn chọn cùng nhát cắt.

## Mạng nơ-ron

### Perceptron nhiều lớp | multilayer perceptron, MLP
Chuỗi các lớp affine xen kẽ hàm kích hoạt phi tuyến. Không có phi tuyến thì cả mạng sụp về một phép affine duy nhất.

### Định lý xấp xỉ phổ quát | universal approximation theorem
Mạng một lớp ẩn đủ rộng xấp xỉ được mọi hàm liên tục trên tập compact với sai số tuỳ ý. Nó chỉ nói **tồn tại** — không nói cần bao nhiêu nơ-ron, có tìm được bằng xuống dốc hay không, hay cần bao nhiêu dữ liệu.

### Nơ-ron ReLU chết | dying ReLU
Đơn vị ReLU rơi vào vùng đầu vào luôn âm nên gradient luôn bằng 0 và không hồi phục được. Là lý do có Leaky ReLU và GELU.

### Lan truyền ngược | backpropagation
Quy tắc chuỗi áp theo thứ tự **từ phải sang trái** trên đồ thị tính toán. Không phải thuật toán học — nó chỉ tính gradient; việc cập nhật tham số là của bộ tối ưu.

### Chế độ ngược so với chế độ xuôi | reverse-mode vs forward-mode AD
Với hàm $\mathbb{R}^d \to \mathbb{R}$, chế độ ngược tính toàn bộ gradient trong một lượt với chi phí $O(md)$, còn chế độ xuôi cần $O(md^2)$ vì phải chạy lại một lượt cho mỗi tham số. Đó là toàn bộ lý do của thứ tự nhân.

### Gradient checkpointing | gradient checkpointing
Chỉ lưu một phần kích hoạt và tính lại phần còn lại trong lượt ngược. Đổi khoảng 30% thời gian để hạ bộ nhớ kích hoạt từ $O(L)$ xuống $O(\sqrt{L})$.

## Tối ưu hoá

### Momentum | momentum
Trung bình trượt của gradient. Làm phẳng dao động theo các hướng có độ cong lớn và cộng dồn theo hướng nhất quán.

### Adam | Adam
Giữ trung bình trượt của gradient ($m$) và của bình phương gradient ($v$) rồi chia, cho ra **một tốc độ học riêng cho mỗi tham số**. Giá phải trả là hai trạng thái cho mỗi tham số.

### Hiệu chỉnh thiên lệch | bias correction
Chia $m_t$, $v_t$ cho $1 - \beta^t$. Cần thiết vì khởi tạo $m_0 = v_0 = 0$ kéo các bước đầu về 0 một cách giả tạo; ảnh hưởng tắt dần khi $t$ lớn.

### AdamW | AdamW
Adam với suy giảm trọng số **tách khỏi** gradient. Với Adam thường, phạt $\ell_2$ bị chia cho $\sqrt{\hat v}$ nên tham số có gradient lớn lại bị phạt ít — ngược ý định.

### Khởi động | warmup
Tăng tốc độ học từ gần 0 trong vài nghìn bước đầu. Cần với Adam vì các bước đầu có ước lượng $v$ chưa đáng tin.

## Khởi tạo và chuẩn hoá

### Khởi tạo Xavier | Xavier/Glorot initialization
Phương sai trọng số $2/(n_{\text{in}} + n_{\text{out}})$, giữ phương sai tín hiệu ổn định theo cả hai chiều. Phù hợp với tanh và sigmoid.

### Khởi tạo He | He initialization
Phương sai trọng số $2/n_{\text{in}}$. Hệ số 2 bù cho việc **ReLU đưa một nửa giá trị về 0**, tức cắt phương sai đi một nửa. Là mặc định cho mạng ReLU.

### BatchNorm | batch normalization
Chuẩn hoá theo thống kê **trên cả lô** cho từng kênh. Có hành vi khác nhau giữa huấn luyện và suy luận, và làm dự đoán của một mẫu phụ thuộc các mẫu khác trong lô.

### LayerNorm | layer normalization
Chuẩn hoá theo thống kê **trên các đặc trưng** của từng mẫu. Không phụ thuộc lô nên dùng được khi độ dài chuỗi thay đổi và khi sinh từng token.

### RMSNorm | root mean square normalization
LayerNorm bỏ bước trừ trung bình: chỉ chia cho căn bậc hai của bình phương trung bình. Rẻ hơn và gần như không mất chất lượng; là mặc định của các mô hình hiện nay.

### Kết nối tắt | residual connection, skip connection
Cho lớp học phần **thêm vào** thay vì học cả phép biến đổi: $h = x + F(x)$. Sinh ra để chữa **vấn đề suy thoái**, không phải chữa quá khớp. Một mình nó làm gradient **bùng nổ**; phải kết hợp chuẩn hoá.

### Vấn đề suy thoái | degradation problem
Hiện tượng mạng sâu hơn có sai số **huấn luyện** cao hơn mạng nông hơn, dù về lý thuyết nó chỉ cần học hàm đồng nhất ở các lớp thêm. Đây mới là động cơ của ResNet.

### Pre-LN so với post-LN | pre-norm vs post-norm
Pre-LN đặt chuẩn hoá **trước** khối con nên có đường tắt sạch từ đầu tới cuối, huấn luyện được mà không cần khởi động kỹ. Post-LN của bài báo 2017 đặt chuẩn hoá sau phép cộng.

### Dropout | dropout
Trong lúc huấn luyện, tắt ngẫu nhiên một tỉ lệ đơn vị. Đọc như huấn luyện một tập hợp ngầm gồm rất nhiều mạng con dùng chung trọng số. Nhiều LLM lớn đặt tỉ lệ này bằng 0.

## Tích chập

### Trường tiếp nhận | receptive field
Vùng đầu vào ảnh hưởng tới một đơn vị ở lớp sâu: $r_i = r_{i-1} + (k_i - 1)\prod_{j<i} s_j$. Trường tiếp nhận **hiệu dụng** nhỏ hơn con số lý thuyết và tăng theo $O(\sqrt{L})$.

### Tích chập tách theo chiều sâu | depthwise separable convolution
Tách tích chập thành một bước theo không gian cho từng kênh và một bước $1\times1$ trộn kênh. Chi phí giảm còn $\frac{1}{C_{\text{out}}} + \frac{1}{k^2}$ so với tích chập thường.

### Bước nhảy | stride
Khoảng dịch của cửa sổ tích chập. Bước nhảy lớn hơn 1 làm giảm độ phân giải và **nhân** với mọi mức tăng trường tiếp nhận về sau.

## Hồi quy

### Bán kính phổ | spectral radius
Trị riêng có độ lớn lớn nhất của $W_{hh}$. Gradient qua $T$ bước co như $\rho^T$: $\rho < 1$ cho tiêu biến, $\rho > 1$ cho bùng nổ.

### Gradient tiêu biến | vanishing gradient
Gradient co về 0 khi truyền qua nhiều bước hoặc nhiều lớp, vì nó là một **tích** nhiều hệ số. Khó chữa hơn bùng nổ vì thông tin đã mất, không cắt ngưỡng lại được.

### Cắt ngưỡng gradient | gradient clipping
Co vector gradient lại khi chuẩn của nó vượt ngưỡng. Chữa được bùng nổ vì nó chỉ giữ lại hướng và bỏ độ lớn quá đáng.

### Đường ô nhớ | cell state path
Trong LSTM, gradient dọc theo ô nhớ là tích các **vô hướng** $f_t$ chứ không phải tích các ma trận. Vì thế đặt độ lệch cổng quên dương lúc khởi tạo là cách giữ gradient sống.

## Attention

### Attention tích vô hướng có tỉ lệ | scaled dot-product attention
$\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$. Là phép tra cứu từ điển lấy đạo hàm được: truy vấn hỏi, khoá mô tả, giá trị là nội dung.

### Vì sao chia $\sqrt{d_k}$ | the √dₖ scaling
Nếu các thành phần của $q$, $k$ độc lập, trung bình 0, phương sai 1 thì $\operatorname{Var}(q \cdot k) = d_k$. Điểm số lớn đẩy softmax vào vùng bão hoà nơi gradient gần 0; chia $\sqrt{d_k}$ đưa phương sai về 1 với mọi $d_k$.

### Nhiều đầu | multi-head attention
Chia $d$ thành $h$ đầu hẹp, chạy attention song song rồi ghép lại. **Không** mua thêm sức tính — tổng chi phí xấp xỉ bằng attention một đầu ở chiều đầy đủ — mà mua **sự đa dạng của quan hệ**.

### Mặt nạ nhân quả | causal mask
Chặn mọi vị trí nhìn về tương lai, cần cho mô hình sinh. Cài bằng cách cộng $-\infty$ vào điểm số trước softmax.

### Hoán vị bất biến | permutation equivariance
Tính chất của self-attention: đảo thứ tự token thì đầu ra chỉ đảo theo. Đây là lý do **buộc phải** có mã hoá vị trí.

### RoPE | rotary position embedding
Xoay từng cặp chiều của $Q$ và $K$ một góc tỉ lệ với vị trí, tần số $\theta_i = 10000^{-2i/d}$. Tính chất then chốt $\langle R_m q, R_n k\rangle = \langle R_{m-n} q, k\rangle$ làm điểm số **chỉ phụ thuộc khoảng cách tương đối**, và không thêm tham số nào.

### GQA | grouped-query attention
Nhiều đầu query dùng chung một cặp khoá–giá trị. Lý do là **bộ nhớ KV cache lúc suy luận**, không phải chất lượng: cache giảm đúng theo tỉ lệ nhóm.

### MQA | multi-query attention
Trường hợp cực đoan của GQA: **một** cặp khoá–giá trị cho toàn bộ các đầu query.

### KV cache | KV cache
Bộ nhớ đệm giữ khoá và giá trị của các token đã sinh, để mỗi token mới không phải tính lại. Dung lượng tỉ lệ với số lớp × số đầu KV × chiều đầu × độ dài × cỡ lô.

### SwiGLU | SwiGLU
FFN dạng $(\text{Swish}(xW) \odot xV)W_2$, dùng **ba** ma trận thay vì hai. Để giữ nguyên số tham số, bề rộng ẩn giảm còn $2/3$, tức $d_{\text{ff}} \approx \frac{8}{3}d$.

### Mixture of Experts | mixture of experts, MoE
Thay một FFN bằng nhiều FFN chuyên gia và một bộ định tuyến chỉ chọn vài cái cho mỗi token. Tách **số tham số** khỏi **số phép tính cho mỗi token**.

## Giải mã

### Giải mã tham lam | greedy decoding
Luôn chọn token có xác suất cao nhất. Xác định nhưng dễ lặp, và không tối ưu toàn cục.

### Nhiệt độ | temperature
Chia logit cho $T$ trước softmax. $T < 1$ làm phân phối nhọn hơn, $T > 1$ làm phẳng hơn.

### Top-k | top-k sampling
Chỉ lấy mẫu trong $k$ token xác suất cao nhất. Nhược điểm: $k$ cố định không thích ứng với độ chắc chắn của từng bước.

### Top-p | nucleus sampling
Lấy tập token nhỏ nhất có tổng xác suất vượt $p$. Kích thước tập **tự thích ứng**: hẹp khi mô hình chắc chắn, rộng khi mô hình lưỡng lự.

### Giải mã suy đoán | speculative decoding
Dùng một mô hình nháp nhỏ sinh trước nhiều token rồi để mô hình lớn kiểm một lượt. Cho **đúng cùng phân phối** với giải mã thường, chỉ nhanh hơn.

## Đếm tài nguyên

### Tham số phi-embedding | non-embedding parameters
$N \approx 12 L d^2$: $4d^2$ cho attention và $8d^2$ cho FFN ở mỗi lớp. Các bài báo về quy luật co giãn dùng con số này thay vì tổng tham số, vì embedding chiếm tỉ lệ rất khác nhau theo cỡ mô hình.

### Quy tắc 6ND | the 6ND rule
Tổng FLOP huấn luyện $C \approx 6ND$ với $N$ tham số phi-embedding và $D$ token: $2N$ cho lượt xuôi và $4N$ cho lượt ngược, cho mỗi token.

### Hiệu suất sử dụng phần cứng | model FLOPs utilization, MFU
Tỉ lệ giữa FLOP hữu ích và FLOP đỉnh mà phần cứng có thể làm trong cùng thời gian. Các lần huấn luyện lớn thường đạt 30–50%.

### Ngưỡng $T > 6d$ | the T > 6d threshold
Tỉ lệ giữa chi phí attention và phần còn lại là $T/(6d)$. Chi phí bậc hai của attention chỉ chi phối khi độ dài chuỗi vượt $6d$ — với $d = 4096$ là khoảng 24 500 token.

### Quy luật co giãn | scaling laws
Các quan hệ luỹ thừa giữa mất mát và cỡ mô hình, cỡ dữ liệu, ngân sách tính. Kết quả Chinchilla nêu rằng ở ngân sách cố định, số token nên tăng **cùng tỉ lệ** với số tham số.
